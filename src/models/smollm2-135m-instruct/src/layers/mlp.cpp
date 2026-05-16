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
#include <cstring>
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
    if (fused_gate_up_silu_m1_v4_) clReleaseKernel(fused_gate_up_silu_m1_v4_);
    if (fused_gate_up_silu_m1_v4_img_) clReleaseKernel(fused_gate_up_silu_m1_v4_img_);
    if (fused_down_res_m1_) clReleaseKernel(fused_down_res_m1_);
    if (fused_down_no4_img_) clReleaseKernel(fused_down_no4_img_);
    if (w_down_img_) clReleaseMemObject(w_down_img_);
    if (w_gate_img_) clReleaseMemObject(w_gate_img_);
    if (w_up_img_)   clReleaseMemObject(w_up_img_);
    if (block_fused_prog_) clReleaseProgram(block_fused_prog_);
    if (fused_gate_up_silu_m1_v4_img_int8_) clReleaseKernel(fused_gate_up_silu_m1_v4_img_int8_);
    if (fused_down_no4_img_int8_)           clReleaseKernel(fused_down_no4_img_int8_);
    if (w_gate_int8_img_) clReleaseMemObject(w_gate_int8_img_);
    if (w_up_int8_img_)   clReleaseMemObject(w_up_int8_img_);
    if (w_down_int8_img_) clReleaseMemObject(w_down_int8_img_);
    if (block_fused_int8_prog_) clReleaseProgram(block_fused_int8_prog_);
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
#ifdef NNOPT_USE_FP16
    // vec4 variant — 2 fp32 accumulators per thread, vec4 inner loop.
    // K=576 = 2 vec4 waves + 16-thread tail. Falls back to scalar if missing.
    fused_gate_up_silu_m1_v4_ = clCreateKernel(block_fused_prog_, "fused_gate_up_silu_m1_v4", &err);
    if (err != CL_SUCCESS) fused_gate_up_silu_m1_v4_ = nullptr;
    fused_gate_up_silu_m1_v4_img_ = clCreateKernel(block_fused_prog_, "fused_gate_up_silu_m1_v4_img", &err);
    if (err != CL_SUCCESS) fused_gate_up_silu_m1_v4_img_ = nullptr;
#endif
    fused_down_res_m1_ = clCreateKernel(block_fused_prog_, "fused_down_residual_m1", &err);
    if (err != CL_SUCCESS || !fused_down_res_m1_) {
        NNOPT_ERROR_FMT("clCreateKernel fused_down_residual_m1 failed: %d", err);
        return false;
    }

    if (!set_weights()) return false;

#ifdef NNOPT_USE_FP16
    // Image2d_t-backed Wdown view (Adreno texture cache).
    // Wdown shape [H=576, INTER=1536]. K=INTER=1536 → K_PIX=384, height=576.
    // Both well under image2d limits.
    fused_down_no4_img_ = clCreateKernel(block_fused_prog_, "fused_down_residual_m1_no4_img", &err);
    if (err != CL_SUCCESS) fused_down_no4_img_ = nullptr;

    {
        const int H     = MODEL_CONFIG::HIDDEN_SIZE;
        const int INTER = MODEL_CONFIG::INTERMEDIATE_SIZE;
        cl_image_format fmt;
        fmt.image_channel_order     = CL_RGBA;
        fmt.image_channel_data_type = CL_HALF_FLOAT;

        auto wrap = [&](cl_mem buf, size_t pix_w, size_t pix_h) -> cl_mem {
            cl_image_desc desc;
            std::memset(&desc, 0, sizeof(desc));
            desc.image_type      = CL_MEM_OBJECT_IMAGE2D;
            desc.image_width     = pix_w;
            desc.image_height    = pix_h;
            desc.image_row_pitch = 0;
            desc.buffer          = buf;
            cl_int e = CL_SUCCESS;
            cl_mem img = clCreateImage(cl_ctx_.context(), CL_MEM_READ_ONLY, &fmt, &desc, nullptr, &e);
            return (e == CL_SUCCESS) ? img : nullptr;
        };

        // Wdown: shape [H=576, INTER=1536]. K_PIX = 384, height = 576.
        if (fused_down_no4_img_ && w_down_) {
            w_down_img_ = wrap(w_down_, (size_t)(INTER / 4), (size_t)H);
            down_img_ready_ = (w_down_img_ != nullptr);
        }
        // Wgate / Wup: shape [INTER=1536, H=576]. K_PIX = 144, height = 1536.
        if (fused_gate_up_silu_m1_v4_img_ && w_gate_ && w_up_) {
            w_gate_img_ = wrap(w_gate_, (size_t)(H / 4), (size_t)INTER);
            w_up_img_   = wrap(w_up_,   (size_t)(H / 4), (size_t)INTER);
            gate_up_img_ready_ = (w_gate_img_ && w_up_img_);
        }
    }

    // ── int8 quantized image path (model.int8.bin) — overrides fp16 images on dispatch.
    if (quantized_) {
        block_fused_int8_prog_ = cl_ctx_.build_program_from_file(
            "kernels/block_fused_int8.cl", "-DNNOPT_USE_FP16=1 -DUSE_FP16=1");
        if (!block_fused_int8_prog_) {
            NNOPT_ERROR("Mlp: failed to build block_fused_int8.cl");
            return false;
        }
        fused_gate_up_silu_m1_v4_img_int8_ =
            clCreateKernel(block_fused_int8_prog_, "fused_gate_up_silu_m1_v4_img_int8", &err);
        if (err != CL_SUCCESS) { NNOPT_ERROR_FMT("clCreateKernel gate_up_int8 failed: %d", err); return false; }
        fused_down_no4_img_int8_ =
            clCreateKernel(block_fused_int8_prog_, "fused_down_residual_m1_no4_img_int8", &err);
        if (err != CL_SUCCESS) { NNOPT_ERROR_FMT("clCreateKernel down_int8 failed: %d", err); return false; }

        const int H     = MODEL_CONFIG::HIDDEN_SIZE;
        const int INTER = MODEL_CONFIG::INTERMEDIATE_SIZE;
        cl_image_format fmt_i8;
        fmt_i8.image_channel_order     = CL_RGBA;
        fmt_i8.image_channel_data_type = CL_SIGNED_INT8;
        auto wrap_i8 = [&](cl_mem buf, size_t pix_w, size_t pix_h) -> cl_mem {
            cl_image_desc desc;
            std::memset(&desc, 0, sizeof(desc));
            desc.image_type   = CL_MEM_OBJECT_IMAGE2D;
            desc.image_width  = pix_w;
            desc.image_height = pix_h;
            desc.buffer       = buf;
            cl_int e = CL_SUCCESS;
            cl_mem img = clCreateImage(cl_ctx_.context(), CL_MEM_READ_ONLY, &fmt_i8, &desc, nullptr, &e);
            return (e == CL_SUCCESS) ? img : nullptr;
        };
        // Same dimensions as fp16 path — int8 image pixel is still RGBA × 4 bytes = 4 ints,
        // so K/4 pixels carries K int8 values per row, identical to fp16 H/4 image width.
        w_down_int8_img_ = wrap_i8(w_down_, (size_t)(INTER / 4), (size_t)H);
        w_gate_int8_img_ = wrap_i8(w_gate_, (size_t)(H / 4),     (size_t)INTER);
        w_up_int8_img_   = wrap_i8(w_up_,   (size_t)(H / 4),     (size_t)INTER);
        if (!w_down_int8_img_ || !w_gate_int8_img_ || !w_up_int8_img_) {
            NNOPT_ERROR_FMT("Mlp[%d]: int8 image creation failed (down=%p gate=%p up=%p)",
                            layer_idx_, (void*)w_down_int8_img_, (void*)w_gate_int8_img_, (void*)w_up_int8_img_);
            return false;
        }
    }
#endif

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
    // Int8 quant path: meta.json says dtype=int8 for the linear sites; load paired
    // per-row fp16 scale buffers (scripts/quantize_weights.py emits .weight.scale).
    quantized_ = (weights_.get_dtype(prefix + "gate_proj.weight") == "int8");
    if (quantized_) {
        w_gate_scale_ = weights_.get_buffer(prefix + "gate_proj.weight.scale");
        w_up_scale_   = weights_.get_buffer(prefix + "up_proj.weight.scale");
        w_down_scale_ = weights_.get_buffer(prefix + "down_proj.weight.scale");
        if (!w_gate_scale_ || !w_up_scale_ || !w_down_scale_) {
            NNOPT_ERROR_FMT("Mlp[%d]: int8 weights but missing scale buffer(s)", layer_idx_);
            return false;
        }
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

    // 1. silu(Wgate*x) * (Wup*x) -> act[I]. Path priority:
    //    int8 image > fp16 image+vec4 > buffer+vec4 > buffer scalar.
    if (quantized_ && fused_gate_up_silu_m1_v4_img_int8_ && w_gate_int8_img_ && w_up_int8_img_) {
        cl_kernel gu = fused_gate_up_silu_m1_v4_img_int8_;
        if (!set_arg_checked(gu, 0, sizeof(cl_mem), &x,                "x"))      return false;
        if (!set_arg_checked(gu, 1, sizeof(cl_mem), &w_gate_int8_img_, "Wg_i8"))  return false;
        if (!set_arg_checked(gu, 2, sizeof(cl_mem), &w_up_int8_img_,   "Wu_i8"))  return false;
        if (!set_arg_checked(gu, 3, sizeof(cl_mem), &w_gate_scale_,    "sg"))     return false;
        if (!set_arg_checked(gu, 4, sizeof(cl_mem), &w_up_scale_,      "su"))     return false;
        if (!set_arg_checked(gu, 5, sizeof(cl_mem), &act,              "out"))    return false;
        if (!set_arg_checked(gu, 6, sizeof(int),    &I,                "INTER"))  return false;
        const size_t WG = 64;
        size_t gws = (size_t)I * WG;
        size_t lws = WG;
        err = nnopt_prof::enqueue(queue, gu, 1, nullptr, &gws, &lws, 0, nullptr, nullptr);
        if (err != CL_SUCCESS) { NNOPT_ERROR_FMT("gate_up_silu_int8 failed: %d", err); return false; }
    } else if (gate_up_img_ready_ && fused_gate_up_silu_m1_v4_img_) {
        cl_kernel gu = fused_gate_up_silu_m1_v4_img_;
        if (!set_arg_checked(gu, 0, sizeof(cl_mem), &x,           "x"))         return false;
        if (!set_arg_checked(gu, 1, sizeof(cl_mem), &w_gate_img_, "Wg_img"))    return false;
        if (!set_arg_checked(gu, 2, sizeof(cl_mem), &w_up_img_,   "Wu_img"))    return false;
        if (!set_arg_checked(gu, 3, sizeof(cl_mem), &act,         "out"))       return false;
        if (!set_arg_checked(gu, 4, sizeof(int),    &I,           "INTER"))     return false;
        const size_t WG = 64;
        size_t gws = (size_t)I * WG;
        size_t lws = WG;
        err = nnopt_prof::enqueue(queue, gu, 1, nullptr, &gws, &lws, 0, nullptr, nullptr);
        if (err != CL_SUCCESS) { NNOPT_ERROR_FMT("fused_gate_up_silu_m1_v4_img failed: %d", err); return false; }
    } else {
        cl_kernel gu_kernel = fused_gate_up_silu_m1_v4_ ? fused_gate_up_silu_m1_v4_ : fused_gate_up_silu_m1_;
        if (!set_arg_checked(gu_kernel, 0, sizeof(cl_mem), &x,      "x"))     return false;
        if (!set_arg_checked(gu_kernel, 1, sizeof(cl_mem), &w_gate_,"w_gate"))return false;
        if (!set_arg_checked(gu_kernel, 2, sizeof(cl_mem), &w_up_,  "w_up"))  return false;
        if (!set_arg_checked(gu_kernel, 3, sizeof(cl_mem), &act,    "out"))   return false;
        if (!set_arg_checked(gu_kernel, 4, sizeof(int),    &H,      "H"))     return false;
        if (!set_arg_checked(gu_kernel, 5, sizeof(int),    &I,      "INTER")) return false;
        const size_t WG = 64;
        size_t gws = (size_t)I * WG;
        size_t lws = WG;
        err = nnopt_prof::enqueue(queue, gu_kernel, 1, nullptr, &gws, &lws, 0, nullptr, nullptr);
        if (err != CL_SUCCESS) { NNOPT_ERROR_FMT("fused_gate_up_silu_m1[_v4] failed: %d", err); return false; }
    }

    // 2. Wdown*act + residual: int8 image > fp16 image > buffer fallback.
    if (quantized_ && fused_down_no4_img_int8_ && w_down_int8_img_) {
        cl_kernel d = fused_down_no4_img_int8_;
        if (!set_arg_checked(d, 0, sizeof(cl_mem), &act,              "mlp_in"))   return false;
        if (!set_arg_checked(d, 1, sizeof(cl_mem), &w_down_int8_img_, "Wd_i8"))    return false;
        if (!set_arg_checked(d, 2, sizeof(cl_mem), &w_down_scale_,    "sd"))       return false;
        if (!set_arg_checked(d, 3, sizeof(cl_mem), &residual,         "residual")) return false;
        if (!set_arg_checked(d, 4, sizeof(int),    &H,                "H"))        return false;
        const size_t WG = 64;
        size_t gws = (size_t)(H / 4) * WG;
        size_t lws = WG;
        err = nnopt_prof::enqueue(queue, d, 1, nullptr, &gws, &lws, 0, nullptr, nullptr);
        if (err != CL_SUCCESS) { NNOPT_ERROR_FMT("down_residual_int8 failed: %d", err); return false; }
    } else if (down_img_ready_ && fused_down_no4_img_) {
        if (!set_arg_checked(fused_down_no4_img_, 0, sizeof(cl_mem), &act,         "mlp_in"))   return false;
        if (!set_arg_checked(fused_down_no4_img_, 1, sizeof(cl_mem), &w_down_img_, "Wd_img"))   return false;
        if (!set_arg_checked(fused_down_no4_img_, 2, sizeof(cl_mem), &residual,    "residual")) return false;
        if (!set_arg_checked(fused_down_no4_img_, 3, sizeof(int),    &H,           "H"))        return false;
        const size_t WG = 64;
        size_t gws = (size_t)(H / 4) * WG;
        size_t lws = WG;
        err = nnopt_prof::enqueue(queue, fused_down_no4_img_, 1, nullptr, &gws, &lws, 0, nullptr, nullptr);
        if (err != CL_SUCCESS) { NNOPT_ERROR_FMT("fused_down_residual_m1_no4_img failed: %d", err); return false; }
    } else {
        if (!set_arg_checked(fused_down_res_m1_, 0, sizeof(cl_mem), &act,      "mlp_in"))  return false;
        if (!set_arg_checked(fused_down_res_m1_, 1, sizeof(cl_mem), &w_down_,  "w_down"))  return false;
        if (!set_arg_checked(fused_down_res_m1_, 2, sizeof(cl_mem), &residual, "residual"))return false;
        if (!set_arg_checked(fused_down_res_m1_, 3, sizeof(int),    &H,        "K"))       return false;
        if (!set_arg_checked(fused_down_res_m1_, 4, sizeof(int),    &I,        "N"))       return false;
        const size_t WG = 64;
        size_t gws = (size_t)H * WG;
        size_t lws = WG;
        err = nnopt_prof::enqueue(queue, fused_down_res_m1_, 1, nullptr, &gws, &lws, 0, nullptr, nullptr);
        if (err != CL_SUCCESS) { NNOPT_ERROR_FMT("fused_down_residual_m1 failed: %d", err); return false; }
    }

    return true;
}
