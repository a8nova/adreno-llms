// Reference: model_info/transformers_src/modeling_qwen2.py:14-33 Qwen2MLP.forward
#include "layers/mlp.h"

#include "debug_utils.h"
#include "opencl_context.h"
#include "utils.h"
#include "prof.h"
#include "weights.h"
#include "model_config.h"

#include <CL/cl.h>
#include <cstdio>
#include <string>

Mlp::Mlp(OpenCLContext& cl_ctx, Weights& weights, const std::string& prefix, int layer_idx)
    : cl_ctx_(cl_ctx), weights_(weights), prefix_(prefix), layer_idx_(layer_idx) {}

Mlp::~Mlp() {
    if (swiglu_kernel_) {
        clReleaseKernel(swiglu_kernel_);
        swiglu_kernel_ = nullptr;
    }
    if (program_) {
        clReleaseProgram(program_);
        program_ = nullptr;
    }
    if (buf_gate_) clReleaseMemObject(buf_gate_);
    if (buf_up_)   clReleaseMemObject(buf_up_);
    if (buf_out_)  clReleaseMemObject(buf_out_);
}

bool Mlp::ensure_activation_buffers_(int max_rows) {
    if (buf_gate_ && buf_up_ && buf_out_ && buf_capacity_rows_ >= max_rows) return true;

    if (buf_gate_) { clReleaseMemObject(buf_gate_); buf_gate_ = nullptr; }
    if (buf_up_)   { clReleaseMemObject(buf_up_);   buf_up_   = nullptr; }
    if (buf_out_)  { clReleaseMemObject(buf_out_);  buf_out_  = nullptr; }

    cl_int err = CL_SUCCESS;
    cl_context ctx = cl_ctx_.context();
    const int inter  = MODEL_CONFIG::INTERMEDIATE_SIZE;
    const int hidden = MODEL_CONFIG::HIDDEN_SIZE;
    const size_t inter_bytes = (size_t)max_rows * (size_t)inter * sizeof(nnopt_storage_t);
    const size_t out_bytes   = (size_t)max_rows * (size_t)hidden * sizeof(nnopt_storage_t);

    buf_gate_ = clCreateBuffer(ctx, CL_MEM_READ_WRITE, inter_bytes, nullptr, &err);
    if (err != CL_SUCCESS || !buf_gate_) {
        NNOPT_ERROR_FMT("Mlp: ensure_activation_buffers_ gate alloc failed: %d", (int)err);
        return false;
    }
    buf_up_ = clCreateBuffer(ctx, CL_MEM_READ_WRITE, inter_bytes, nullptr, &err);
    if (err != CL_SUCCESS || !buf_up_) {
        NNOPT_ERROR_FMT("Mlp: ensure_activation_buffers_ up alloc failed: %d", (int)err);
        return false;
    }
    buf_out_ = clCreateBuffer(ctx, CL_MEM_READ_WRITE, out_bytes, nullptr, &err);
    if (err != CL_SUCCESS || !buf_out_) {
        NNOPT_ERROR_FMT("Mlp: ensure_activation_buffers_ out alloc failed: %d", (int)err);
        return false;
    }
    buf_capacity_rows_ = max_rows;
    return true;
}

bool Mlp::initialize() {
    // Weights are lazy GPU buffers; just verify keys exist
    if (!weights_.has_tensor(prefix_ + ".gate_proj.weight")) {
        NNOPT_ERROR_FMT("Mlp missing weight: %s.gate_proj.weight", prefix_.c_str());
        return false;
    }
    if (!weights_.has_tensor(prefix_ + ".up_proj.weight")) {
        NNOPT_ERROR_FMT("Mlp missing weight: %s.up_proj.weight", prefix_.c_str());
        return false;
    }
    if (!weights_.has_tensor(prefix_ + ".down_proj.weight")) {
        NNOPT_ERROR_FMT("Mlp missing weight: %s.down_proj.weight", prefix_.c_str());
        return false;
    }

    program_ = cl_ctx_.build_program_from_file("kernels/mlp.cl");
    if (!program_) {
        NNOPT_ERROR("Mlp: failed to build kernels/mlp.cl");
        return false;
    }

    cl_int err = CL_SUCCESS;
    // Kernel name must match kernels/mlp.cl
    swiglu_kernel_ = clCreateKernel(program_, "swiglu_inplace", &err);
    if (err != CL_SUCCESS) {
        NNOPT_ERROR_FMT("Mlp: clCreateKernel(swiglu_inplace) failed: %d", (int)err);
        return false;
    }

    return true;
}

cl_mem Mlp::forward(cl_command_queue queue, cl_mem input, int seq_len,
                    cl_mem residual_dest) {
    NNOPT_LAYER_FWD("mlp");
    NNOPT_LAYER_CHECK_INPUT(prefix_.c_str(), queue, input, (size_t)seq_len * MODEL_CONFIG::HIDDEN_SIZE);

    const int hidden = MODEL_CONFIG::HIDDEN_SIZE;
    const int inter = MODEL_CONFIG::INTERMEDIATE_SIZE;

    // Reuse persistent buf_gate_/buf_up_/buf_out_ — saves 3 alloc/free per
    // layer per token at decode (Mamba Step 6 lesson).
    if (!ensure_activation_buffers_(seq_len)) return nullptr;
    cl_mem gate = buf_gate_;
    cl_mem up   = buf_up_;
    cl_mem out  = buf_out_;

    cl_mem gate_w = weights_.get_buffer(prefix_ + ".gate_proj.weight");
    cl_mem up_w   = weights_.get_buffer(prefix_ + ".up_proj.weight");
    if (!gate_w || !up_w) {
        NNOPT_ERROR("Mlp: gate_proj.weight or up_proj.weight buffer null");
        return nullptr;
    }

    // ── M=1 decode fast path: fused gate_proj + up_proj + silu*mul.
    // EMPIRICALLY DISABLED on Adreno 620 — measured 1.78× regression
    // vs the unfused 2-GEMV + swiglu path. The 8 fp32 accumulators
    // per thread (4 gate + 4 up) push past the register budget and
    // cause spill-to-local-mem; the saved launch + tensor round-trip
    // (~2 ms/token) is more than wiped out by the slower per-WG
    // throughput. Mamba's "fused norm+GEMV redundancy" memory captures
    // the same general lesson — fusion at decode-M=1 only pays off if
    // the per-WG resource pressure stays the same.
    // To re-enable: split into 2 fused kernels (one per output of the
    // pair) so each thread holds ≤4 accs, OR drop no4 and keep 1 output
    // per WG so 4 accs total. Skipped for now.
    const bool kFuseGateUpSilu = false;
    if (kFuseGateUpSilu && seq_len == 1 &&
        fused_gate_up_silu_m1(queue, /*N=*/inter, /*K=*/hidden,
                              input, gate_w, up_w, /*out=*/gate)) {
        // gate now holds silu(gate_proj(x)) * up_proj(x). Skip the unfused
        // gate/up/swiglu sequence and head straight to down_proj.
        if (layer_idx_ == 0) {
            NNOPT_LAYER_CHECK("block0_sub_mlp_down_proj_in", queue, gate, (size_t)seq_len * inter);
        }
        cl_mem down_w = weights_.get_buffer(prefix_ + ".down_proj.weight");
        if (!down_w) {
            NNOPT_ERROR("Mlp: down_proj.weight buffer null");
            return nullptr;
        }
        if (!pytorch_linear(queue, /*M=*/seq_len, /*N=*/hidden, /*K=*/inter, gate, down_w, out)) {
            return nullptr;
        }
        if (layer_idx_ == 0) {
            NNOPT_LAYER_CHECK("block0_sub_mlp_down_proj_out", queue, out, (size_t)seq_len * hidden);
        }
        clRetainMemObject(out);
        NNOPT_LAYER_FWD_DONE("mlp");
        return out;
    }

    // ── Prefill / fallback: separate gate_proj, up_proj, swiglu launches.
    // gate = gate_proj(x)
    if (!pytorch_linear(queue, /*M=*/seq_len, /*N=*/inter, /*K=*/hidden, input, gate_w, gate)) {
        return nullptr;
    }
    if (layer_idx_ == 0) {
        NNOPT_LAYER_CHECK("block0_sub_mlp_gate_proj_out", queue, gate, (size_t)seq_len * inter);
    }

    // up = up_proj(x)
    if (!pytorch_linear(queue, /*M=*/seq_len, /*N=*/inter, /*K=*/hidden, input, up_w, up)) {
        return nullptr;
    }
    if (layer_idx_ == 0) {
        NNOPT_LAYER_CHECK("block0_sub_mlp_up_proj_out", queue, up, (size_t)seq_len * inter);
    }

    // gate = silu(gate) * up
    {
        const int total = seq_len * inter;
        cl_int err = CL_SUCCESS;
        err = clSetKernelArg(swiglu_kernel_, 0, sizeof(cl_mem), &gate);
        if (err != CL_SUCCESS) { NNOPT_ERROR_FMT("Mlp: setArg 0: %d", (int)err); clReleaseMemObject(gate); clReleaseMemObject(up); return nullptr; }
        err = clSetKernelArg(swiglu_kernel_, 1, sizeof(cl_mem), &up);
        if (err != CL_SUCCESS) { NNOPT_ERROR_FMT("Mlp: setArg 1: %d", (int)err); clReleaseMemObject(gate); clReleaseMemObject(up); return nullptr; }
        err = clSetKernelArg(swiglu_kernel_, 2, sizeof(int), &total);
        if (err != CL_SUCCESS) { NNOPT_ERROR_FMT("Mlp: setArg 2: %d", (int)err); clReleaseMemObject(gate); clReleaseMemObject(up); return nullptr; }

        // Vec4 dispatch: 1 thread per 4 fp16 group. Qwen INTERMEDIATE_SIZE=4864
        // is divisible by 4 (config-derived; build refuses if it ever isn't).
        size_t gws[1] = {(size_t)(total >> 2)};
        err = nnopt_prof::enqueue(queue, swiglu_kernel_, 1, nullptr, gws, nullptr, 0, nullptr, nullptr);
        if (err != CL_SUCCESS) {
            NNOPT_ERROR_FMT("Mlp: enqueue swiglu_inplace failed: %d", (int)err);
            return nullptr;
        }
        NNOPT_DEBUG_SYNC(queue);
    }

    if (layer_idx_ == 0) {
        NNOPT_LAYER_CHECK("block0_sub_mlp_down_proj_in", queue, gate, (size_t)seq_len * inter);
    }

    // out = down_proj(gate)
    cl_mem down_w = weights_.get_buffer(prefix_ + ".down_proj.weight");
    if (!down_w) {
        NNOPT_ERROR("Mlp: down_proj.weight buffer null");
        return nullptr;
    }

    // Decode fast path: when seq_len==1 and a residual_dest was provided,
    // dispatch down_proj as a fused gemv+residual_add directly into
    // residual_dest. Saves the explicit element_add_inplace launch in
    // Model::forward (24 layers × 1 launch/token saved).
    if (seq_len == 1 && residual_dest != nullptr &&
        pytorch_linear_add(queue, /*M=*/seq_len, /*N=*/hidden, /*K=*/inter, gate, down_w, residual_dest)) {
        if (layer_idx_ == 0) {
            NNOPT_LAYER_CHECK("block0_sub_mlp_down_proj_out_fused", queue, residual_dest, (size_t)seq_len * hidden);
        }
        clRetainMemObject(residual_dest);
        NNOPT_LAYER_FWD_DONE("mlp");
        return residual_dest;
    }

    if (!pytorch_linear(queue, /*M=*/seq_len, /*N=*/hidden, /*K=*/inter, gate, down_w, out)) {
        return nullptr;
    }

    if (layer_idx_ == 0) {
        NNOPT_LAYER_CHECK("block0_sub_mlp_down_proj_out", queue, out, (size_t)seq_len * hidden);
    }

    // Caller owns the returned cl_mem and will clReleaseMemObject() it.
    // Since out is our persistent buf_out_ we retain it before handing
    // the reference out — caller's release just decrements back to our
    // own owned reference.
    clRetainMemObject(out);

    // Layer output check under canonical name: "mlp_<idx>" is emitted by model.cpp.
    NNOPT_LAYER_FWD_DONE("mlp");
    return out;
}

cl_mem Mlp::create_buffer_(int rows, int cols) {
    cl_int err = CL_SUCCESS;
    const size_t bytes = (size_t)rows * (size_t)cols * sizeof(nnopt_storage_t);
    cl_mem buf = clCreateBuffer(cl_ctx_.context(), CL_MEM_READ_WRITE, bytes, nullptr, &err);
    if (err != CL_SUCCESS) {
        NNOPT_ERROR_FMT("Mlp: clCreateBuffer(%zu) failed: %d", bytes, (int)err);
        return nullptr;
    }
    return buf;
}
