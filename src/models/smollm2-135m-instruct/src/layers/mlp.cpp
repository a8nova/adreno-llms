// Reference: model_info/transformers_src/modeling_llama.py:171-186 LlamaMLP.forward

#include "layers/mlp.h"
#include "opencl_context.h"
#include "weights.h"
#include "debug_utils.h"
#include "utils.h"
#include "model_config.h"
#include "prof.h"
#include <clblast.h>
#include <cmath>
#include <string>
#include <vector>

static inline bool set_arg_checked(cl_kernel k, cl_uint idx, size_t sz, const void* val, const char* what) {
    cl_int err = clSetKernelArg(k, idx, sz, val);
    if (err != CL_SUCCESS) {
        NNOPT_ERROR_FMT("clSetKernelArg %s failed: %d", what, err);
        return false;
    }
    return true;
}

Mlp::Mlp(OpenCLContext& cl_ctx, Weights& weights, int layer_idx)
    : cl_ctx_(cl_ctx), weights_(weights), layer_idx_(layer_idx) {}

Mlp::~Mlp() {
    if (decode_act_buf_) clReleaseMemObject(decode_act_buf_);
    if (silu_mul_kernel_) clReleaseKernel(silu_mul_kernel_);
    if (mlp_program_) clReleaseProgram(mlp_program_);
    if (fused_gate_up_silu_m1_) clReleaseKernel(fused_gate_up_silu_m1_);
    if (fused_down_res_m1_) clReleaseKernel(fused_down_res_m1_);
    if (block_fused_prog_) clReleaseProgram(block_fused_prog_);
}

bool Mlp::initialize() {
    mlp_program_ = cl_ctx_.build_program_from_file(
        "kernels/mlp_swiglu.cl",
#ifdef NNOPT_USE_FP16
        "-DNNOPT_USE_FP16=1 -DUSE_FP16=1"
#else
        ""
#endif
    );
    if (!mlp_program_) { NNOPT_ERROR("Failed to build kernels/mlp_swiglu.cl"); return false; }

    cl_int err = CL_SUCCESS;
    silu_mul_kernel_ = clCreateKernel(mlp_program_, "silu_mul", &err);
    if (err != CL_SUCCESS || !silu_mul_kernel_) {
        NNOPT_ERROR_FMT("clCreateKernel silu_mul failed: %d", err);
        return false;
    }

    // ── Decode fast-path kernels (block_fused.cl)
    block_fused_prog_ = cl_ctx_.build_program_from_file(
        "kernels/block_fused.cl",
#ifdef NNOPT_USE_FP16
        "-DNNOPT_USE_FP16=1 -DUSE_FP16=1"
#else
        ""
#endif
    );
    if (!block_fused_prog_) { NNOPT_ERROR("Failed to build block_fused.cl (mlp)"); return false; }

    fused_gate_up_silu_m1_ = clCreateKernel(block_fused_prog_, "fused_gate_up_silu_m1", &err);
    if (err != CL_SUCCESS || !fused_gate_up_silu_m1_) {
        NNOPT_ERROR_FMT("clCreateKernel fused_gate_up_silu_m1 failed: %d", err);
        return false;
    }
    fused_down_res_m1_ = clCreateKernel(block_fused_prog_, "fused_down_residual_m1", &err);
    if (err != CL_SUCCESS || !fused_down_res_m1_) {
        NNOPT_ERROR_FMT("clCreateKernel fused_down_residual_m1 failed: %d", err);
        return false;
    }

    if (!set_weights()) return false;

    // Pre-allocate persistent decode intermediate buffer.
    cl_int berr = CL_SUCCESS;
    decode_act_buf_ = clCreateBuffer(cl_ctx_.context(), CL_MEM_READ_WRITE,
                                     (size_t)MODEL_CONFIG::INTERMEDIATE_SIZE * sizeof(nnopt_storage_t),
                                     nullptr, &berr);
    if (berr != CL_SUCCESS || !decode_act_buf_) {
        NNOPT_ERROR_FMT("Mlp: alloc decode_act_buf_ failed: %d", berr);
        return false;
    }

    NNOPT_LAYER_INIT_FMT("block%d_sub_mlp", layer_idx_);
    return true;
}

bool Mlp::set_weights() {
    const std::string prefix = "model.layers." + std::to_string(layer_idx_) + ".mlp.";
    w_gate_ = weights_.get_buffer(prefix + "gate_proj.weight");
    w_up_   = weights_.get_buffer(prefix + "up_proj.weight");
    w_down_ = weights_.get_buffer(prefix + "down_proj.weight");
    if (!w_gate_ || !w_up_ || !w_down_) {
        NNOPT_ERROR_FMT("Mlp[%d]: missing weight buffer(s)", layer_idx_);
        return false;
    }
    return true;
}

cl_mem Mlp::forward(cl_command_queue queue, cl_mem input, int seq_len) {
    // Note: NNOPT_LAYER_CHECK_INPUT_FMT does not exist. Use formatted name.
    char name[64];
    snprintf(name, sizeof(name), "block%d_sub_mlp", layer_idx_);
    NNOPT_LAYER_CHECK_INPUT(name, queue, input, (size_t)seq_len * MODEL_CONFIG::HIDDEN_SIZE);

    cl_context ctx = cl_ctx_.context();
    cl_int err = CL_SUCCESS;

    const int H = MODEL_CONFIG::HIDDEN_SIZE;
    const int I = MODEL_CONFIG::INTERMEDIATE_SIZE;

    cl_mem gate = clCreateBuffer(ctx, CL_MEM_READ_WRITE, (size_t)seq_len * I * sizeof(nnopt_storage_t), nullptr, &err);
    if (err != CL_SUCCESS || !gate) { NNOPT_ERROR_FMT("alloc gate failed: %d", err); return nullptr; }
    cl_mem up = clCreateBuffer(ctx, CL_MEM_READ_WRITE, (size_t)seq_len * I * sizeof(nnopt_storage_t), nullptr, &err);
    if (err != CL_SUCCESS || !up) {
        NNOPT_ERROR_FMT("alloc up failed: %d", err);
        clReleaseMemObject(gate);
        return nullptr;
    }

    auto fail = [&]() -> cl_mem {
        if (gate) clReleaseMemObject(gate);
        if (up) clReleaseMemObject(up);
        return nullptr;
    };

    if (!pytorch_linear(queue, seq_len, I, H, input, w_gate_, gate)) return fail();
    if (!pytorch_linear(queue, seq_len, I, H, input, w_up_, up)) return fail();

    // out = silu(gate) * up  (SwiGLU)
    // kernels/mlp_swiglu.cl:silu_mul signature: (gate, up, out, total)
    const int total = seq_len * I;
    if (!set_arg_checked(silu_mul_kernel_, 0, sizeof(cl_mem), &gate, "gate")) return fail();
    if (!set_arg_checked(silu_mul_kernel_, 1, sizeof(cl_mem), &up, "up")) return fail();
    if (!set_arg_checked(silu_mul_kernel_, 2, sizeof(cl_mem), &gate, "out")) return fail();  // in-place write into gate buffer
    if (!set_arg_checked(silu_mul_kernel_, 3, sizeof(int), &total, "total")) return fail();

    size_t gws = (size_t)total;
    err = nnopt_prof::enqueue(queue, silu_mul_kernel_, 1, nullptr, &gws, nullptr, 0, nullptr, nullptr);
    if (err != CL_SUCCESS) {
        NNOPT_ERROR_FMT("swiglu_mul dispatch failed: %d", err);
        return fail();
    }

    cl_mem out = clCreateBuffer(ctx, CL_MEM_READ_WRITE, (size_t)seq_len * H * sizeof(nnopt_storage_t), nullptr, &err);
    if (err != CL_SUCCESS || !out) {
        NNOPT_ERROR_FMT("alloc down_proj out failed: %d", err);
        return fail();
    }

    if (!pytorch_linear(queue, seq_len, H, I, gate, w_down_, out)) {
        clReleaseMemObject(out);
        return fail();
    }

    NNOPT_LAYER_CHECK(name, queue, out, (size_t)seq_len * MODEL_CONFIG::HIDDEN_SIZE);

    clReleaseMemObject(gate);
    clReleaseMemObject(up);
    return out;
}

// ── Decode fast path (M=1) ───────────────────────────────────────────────────
// Replaces 2 CLBlast gate/up GEMMs + silu_mul + CLBlast down GEMM + element_add
// with 2 custom GEMV kernel dispatches. Updates residual in-place.
bool Mlp::forward_decode_into_residual(cl_command_queue queue, cl_mem x, cl_mem residual) {
    const int H = MODEL_CONFIG::HIDDEN_SIZE;
    const int I = MODEL_CONFIG::INTERMEDIATE_SIZE;

    cl_int err = CL_SUCCESS;

    // Use persistent decode buffer — no allocation per step.
    cl_mem act = decode_act_buf_;

    // 1. fused_gate_up_silu_m1: silu(Wgate*x) * (Wup*x) -> act[I]
    if (!set_arg_checked(fused_gate_up_silu_m1_, 0, sizeof(cl_mem), &x,      "x"))     return false;
    if (!set_arg_checked(fused_gate_up_silu_m1_, 1, sizeof(cl_mem), &w_gate_,"w_gate"))return false;
    if (!set_arg_checked(fused_gate_up_silu_m1_, 2, sizeof(cl_mem), &w_up_,  "w_up"))  return false;
    if (!set_arg_checked(fused_gate_up_silu_m1_, 3, sizeof(cl_mem), &act,    "out"))   return false;
    if (!set_arg_checked(fused_gate_up_silu_m1_, 4, sizeof(int),    &H,      "H"))     return false;
    if (!set_arg_checked(fused_gate_up_silu_m1_, 5, sizeof(int),    &I,      "INTER")) return false;
    {
        const size_t WG = 64;
        size_t gws = (size_t)I * WG;
        size_t lws = WG;
        err = nnopt_prof::enqueue(queue, fused_gate_up_silu_m1_, 1, nullptr, &gws, &lws, 0, nullptr, nullptr);
        if (err != CL_SUCCESS) { NNOPT_ERROR_FMT("fused_gate_up_silu_m1 failed: %d", err); return false; }
    }

    // 2. fused_down_residual_m1: Wdown*act + residual -> residual in-place.
    if (!set_arg_checked(fused_down_res_m1_, 0, sizeof(cl_mem), &act,      "mlp_in"))  return false;
    if (!set_arg_checked(fused_down_res_m1_, 1, sizeof(cl_mem), &w_down_,  "w_down"))  return false;
    if (!set_arg_checked(fused_down_res_m1_, 2, sizeof(cl_mem), &residual, "residual"))return false;
    if (!set_arg_checked(fused_down_res_m1_, 3, sizeof(int),    &H,        "K"))       return false;
    if (!set_arg_checked(fused_down_res_m1_, 4, sizeof(int),    &I,        "N"))       return false;
    {
        const size_t WG = 64;
        size_t gws = (size_t)H * WG;
        size_t lws = WG;
        err = nnopt_prof::enqueue(queue, fused_down_res_m1_, 1, nullptr, &gws, &lws, 0, nullptr, nullptr);
        if (err != CL_SUCCESS) { NNOPT_ERROR_FMT("fused_down_residual_m1 failed: %d", err); return false; }
    }

    return true;
}
