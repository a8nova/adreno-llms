// Reference: model_info/transformers_src/modeling_granitemoehybrid.py:761-778 GraniteMoeHybridMLP.forward

#include "layers/shared_mlp.h"
#include "opencl_context.h"
#include "weights.h"
#include "debug_utils.h"
#include "utils.h"
#include "model_config.h"
#include "kernel_profiler.h"

#include <clblast.h>
#include <cmath>
#include <string>
#include <vector>

static std::string format_key(const std::string& templ, int i) {
    std::string out = templ;
    auto pos = out.find("{i}");
    if (pos != std::string::npos) out.replace(pos, 3, std::to_string(i));
    return out;
}

SharedMlp::SharedMlp(OpenCLContext& cl_ctx, Weights& weights, int layer_idx)
    : cl_ctx_(cl_ctx), weights_(weights), layer_idx_(layer_idx) {}

SharedMlp::~SharedMlp() {
    if (silu_mul_kernel_) clReleaseKernel(silu_mul_kernel_);
    if (fused_output_kernel_) clReleaseKernel(fused_output_kernel_);
    if (fused_output_kernel_v2_) clReleaseKernel(fused_output_kernel_v2_);
    if (fused_output_kernel_v2_img_) clReleaseKernel(fused_output_kernel_v2_img_);
    if (fused_gate_up_silu_kernel_) clReleaseKernel(fused_gate_up_silu_kernel_);
    if (mlp_down_residual_kernel_v2_) clReleaseKernel(mlp_down_residual_kernel_v2_);
    if (program_) clReleaseProgram(program_);
    if (proj_) clReleaseMemObject(proj_);
    if (gated_) clReleaseMemObject(gated_);
    if (w_out_img_) clReleaseMemObject(w_out_img_);
    if (w_in_img_) clReleaseMemObject(w_in_img_);
}

// Geometric-growth scratch ensure (mirrors attention.cpp helper).
static bool ensure_mlp_scratch(cl_context ctx, cl_mem& buf, size_t& cap_bytes,
                               size_t needed_bytes, const char* name, int layer_idx) {
    if (buf && cap_bytes >= needed_bytes) return true;
    if (buf) { clReleaseMemObject(buf); buf = nullptr; }
    size_t new_cap = (cap_bytes == 0) ? needed_bytes : (cap_bytes * 2);
    while (new_cap < needed_bytes) new_cap *= 2;
    cl_int err = CL_SUCCESS;
    buf = clCreateBuffer(ctx, CL_MEM_READ_WRITE, new_cap, nullptr, &err);
    if (err != CL_SUCCESS || !buf) {
        NNOPT_ERROR_FMT("SharedMlp[%d]: scratch '%s' clCreateBuffer(%zu B) failed (%d)",
                        layer_idx, name, new_cap, err);
        cap_bytes = 0; buf = nullptr;
        return false;
    }
    cap_bytes = new_cap;
    return true;
}

bool SharedMlp::initialize() {
    // PROGRAM-INIT-OK
    program_ = cl_ctx_.build_program_from_file("kernels/shared_mlp.cl");
    if (!program_) {
        NNOPT_ERROR("SharedMlp: build_program_from_file(kernels/shared_mlp.cl) failed");
        return false;
    }

    cl_int err = CL_SUCCESS;
    // Legacy kernel — kept for non-Granite SwiGLU paths but unused here.
    silu_mul_kernel_ = clCreateKernel(program_, "swiglu_silu_mul", &err);
    if (err != CL_SUCCESS || !silu_mul_kernel_) {
        NNOPT_ERROR_FMT("SharedMlp: clCreateKernel(swiglu_silu_mul) failed (%d)", err);
        return false;
    }
    // Fused SwiGLU + output_linear kernel: required for Granite/muP because
    // silu(gate)*up overflows fp16 (max 65504) for some elements; fusing
    // keeps the reduction in fp32 and only stores the final (in-range)
    // output_linear result.
    fused_output_kernel_ = clCreateKernel(program_, "swiglu_fused_output", &err);
    if (err != CL_SUCCESS || !fused_output_kernel_) {
        NNOPT_ERROR_FMT("SharedMlp: clCreateKernel(swiglu_fused_output) failed (%d)", err);
        return false;
    }
    // GEMV-style cooperative reduction (WG=64, vec4) — 89% of decode GPU
    // time was in the legacy fused_output kernel; this v2 fixes that.
    fused_output_kernel_v2_ = clCreateKernel(program_, "swiglu_fused_output_m1_v2", &err);
    if (err != CL_SUCCESS || !fused_output_kernel_v2_) {
        NNOPT_ERROR_FMT("SharedMlp: clCreateKernel(swiglu_fused_output_m1_v2) failed (%d)", err);
        return false;
    }
    // Step #5: fused gate+up+silu and down+residual kernels (M=1, K=1024).
    fused_gate_up_silu_kernel_ = clCreateKernel(program_, "fused_gate_up_silu_m1_k1024", &err);
    if (err != CL_SUCCESS || !fused_gate_up_silu_kernel_) {
        NNOPT_ERROR_FMT("SharedMlp: clCreateKernel(fused_gate_up_silu_m1_k1024) failed (%d) — falling back", err);
        fused_gate_up_silu_kernel_ = nullptr;
    }
    cl_int err2 = CL_SUCCESS;
    mlp_down_residual_kernel_v2_ = clCreateKernel(program_, "mlp_down_residual_m1_v2", &err2);
    if (err2 != CL_SUCCESS || !mlp_down_residual_kernel_v2_) {
        NNOPT_ERROR_FMT("SharedMlp: clCreateKernel(mlp_down_residual_m1_v2) failed (%d) — falling back", err2);
        mlp_down_residual_kernel_v2_ = nullptr;
    }
    // Step #8: image2d_t variant of swiglu_fused_output_v2.
    cl_int err_img = CL_SUCCESS;
    fused_output_kernel_v2_img_ = clCreateKernel(program_, "swiglu_fused_output_m1_v2_img", &err_img);
    if (err_img != CL_SUCCESS || !fused_output_kernel_v2_img_) {
        NNOPT_ERROR_FMT("SharedMlp: clCreateKernel(swiglu_fused_output_m1_v2_img) failed (%d) — falling back", err_img);
        fused_output_kernel_v2_img_ = nullptr;
    }

    const std::string w_in_key = format_key("model.layers.{i}.shared_mlp.input_linear.weight", layer_idx_);
    const std::string w_out_key = format_key("model.layers.{i}.shared_mlp.output_linear.weight", layer_idx_);

    w_in_ = weights_.get_buffer(w_in_key);
    w_out_ = weights_.get_buffer(w_out_key);
    if (!w_in_ || !w_out_) return false;

    // Step #8: wrap w_out_ as a CL_RGBA + CL_HALF_FLOAT image2d. Uses
    // cl_khr_image2d_from_buffer (clCreateImage with desc.buffer set). Falls
    // back gracefully (kernel not selected) if image creation fails.
    if (fused_output_kernel_v2_img_) {
        const int hidden = MODEL_CONFIG::HIDDEN_SIZE;
        const int inter  = MODEL_CONFIG::SHARED_INTERMEDIATE_SIZE;
        cl_image_format fmt{};
        fmt.image_channel_order     = CL_RGBA;
        fmt.image_channel_data_type = CL_HALF_FLOAT;
        cl_image_desc desc{};
        desc.image_type   = CL_MEM_OBJECT_IMAGE2D;
        desc.image_width  = (size_t)(inter / 4);  // 2048 / 4 = 512 RGBA pixels
        desc.image_height = (size_t)hidden;       // 1024
        desc.buffer       = w_out_;
        cl_int e = CL_SUCCESS;
        w_out_img_ = clCreateImage(cl_ctx_.context(), CL_MEM_READ_ONLY, &fmt, &desc, nullptr, &e);
        if (e != CL_SUCCESS || !w_out_img_) {
            NNOPT_ERROR_FMT("SharedMlp[%d]: clCreateImage(w_out) failed (%d) — image2d disabled", layer_idx_, e);
            w_out_img_ = nullptr;
        }
    }

    // Step #9: wrap w_in_ (the gate_up [4096, 1024] matrix) as an image2d.
    // Width = K/4 = 256, height = 2*intermediate = 4096.
    {
        const int hidden = MODEL_CONFIG::HIDDEN_SIZE;
        const int inter  = MODEL_CONFIG::SHARED_INTERMEDIATE_SIZE;
        cl_image_format fmt{};
        fmt.image_channel_order     = CL_RGBA;
        fmt.image_channel_data_type = CL_HALF_FLOAT;
        cl_image_desc desc{};
        desc.image_type   = CL_MEM_OBJECT_IMAGE2D;
        desc.image_width  = (size_t)(hidden / 4);   // 1024 / 4 = 256
        desc.image_height = (size_t)(2 * inter);    // 4096
        desc.buffer       = w_in_;
        cl_int e = CL_SUCCESS;
        w_in_img_ = clCreateImage(cl_ctx_.context(), CL_MEM_READ_ONLY, &fmt, &desc, nullptr, &e);
        if (e != CL_SUCCESS || !w_in_img_) {
            NNOPT_ERROR_FMT("SharedMlp[%d]: clCreateImage(w_in) failed (%d) — image2d for gate_up disabled", layer_idx_, e);
            w_in_img_ = nullptr;
        }
    }

    NNOPT_LAYER_INIT_FMT("shared_mlp_%d", layer_idx_);
    return true;
}

cl_mem SharedMlp::forward(cl_command_queue queue, cl_mem input, cl_mem residual, float residual_scale, int seq_len) {
    const size_t in_elems = (size_t)seq_len * MODEL_CONFIG::HIDDEN_SIZE;
    // NOTE: no NNOPT_LAYER_CHECK_INPUT_FMT macro exists.
    NNOPT_LAYER_CHECK_INPUT("shared_mlp", queue, input, in_elems);

    cl_context ctx = cl_ctx_.context();
    const int hidden = MODEL_CONFIG::HIDDEN_SIZE;
    const int inter = MODEL_CONFIG::SHARED_INTERMEDIATE_SIZE;

    cl_int err = CL_SUCCESS;

    // Step #5: fused gate+up+silu fast path was REVERTED — measured 7.7%
    // regression (10.56 → 9.75 tok/s). Cause: the 8 fp32 accumulators per
    // thread (4 gate + 4 up) caused register spill on Adreno A6xx, exactly
    // as SmolLM2 documented in their no4 attempt (1.78× regression). The
    // fused_gate_up_silu_m1_k1024 + mlp_down_residual_m1_v2 kernels are
    // kept in the .cl file but the `use_fused_v5` predicate is disabled to
    // force the legacy fast path. A single-output (2 accs/thread) variant
    // could work but isn't implemented yet.
    const bool use_fused_v5 = false;

    cl_mem proj = nullptr;
    if (!use_fused_v5) {
        // Legacy path: persistent proj[2*inter] then swiglu_v2.
        const size_t proj_bytes = (size_t)seq_len * (size_t)(2 * inter) * sizeof(nnopt_storage_t);
        if (!ensure_mlp_scratch(ctx, proj_, proj_cap_bytes_, proj_bytes, "proj", layer_idx_)) {
            return nullptr;
        }
        proj = proj_;
        // Step #9: image2d_t for gate_up REVERTED — measured 5-run median 9.53
        // tok/s vs #8's 10.33 (regression). The w_in matrix is [4096, 1024]
        // → image height 4096; the texture cache locality across 4 read_imagef
        // calls per WG (rows n0..n0+3) is worse than the buffer L1 path,
        // unlike w_out at [1024, 2048] (height 1024) which fit the cache well
        // and gave +7.7%. Buffer path retained for input_linear.
        if (!pytorch_linear(queue, seq_len, 2 * inter, hidden, input, w_in_, proj)) {
            NNOPT_ERROR_FMT("SharedMlp: pytorch_linear input_linear failed layer %d", layer_idx_);
            return nullptr;
        }
    } else {
        // #5 fused path: ensure persistent gated[inter] buffer.
        const size_t gated_bytes = (size_t)inter * sizeof(nnopt_storage_t);
        if (!ensure_mlp_scratch(ctx, gated_, gated_cap_bytes_, gated_bytes, "gated", layer_idx_)) {
            return nullptr;
        }
        // Dispatch fused_gate_up_silu_m1_k1024(input, w_in, gated, inter)
        if (clSetKernelArg(fused_gate_up_silu_kernel_, 0, sizeof(cl_mem), &input)  != CL_SUCCESS ||
            clSetKernelArg(fused_gate_up_silu_kernel_, 1, sizeof(cl_mem), &w_in_)  != CL_SUCCESS ||
            clSetKernelArg(fused_gate_up_silu_kernel_, 2, sizeof(cl_mem), &gated_) != CL_SUCCESS ||
            clSetKernelArg(fused_gate_up_silu_kernel_, 3, sizeof(int),    &inter)  != CL_SUCCESS) {
            NNOPT_ERROR_FMT("SharedMlp: setarg fused_gate_up_silu failed layer %d", layer_idx_);
            return nullptr;
        }
        const size_t WG = 64;
        const size_t gws = (size_t)(inter / 4) * WG;
        const size_t lws = WG;
        err = clEnqueueNDRangeKernel(queue, fused_gate_up_silu_kernel_, 1, nullptr, &gws, &lws,
                                     0, nullptr, KernelProfiler::event_for("fused_gate_up_silu_m1_k1024"));
        if (err != CL_SUCCESS) {
            NNOPT_ERROR_FMT("SharedMlp: dispatch fused_gate_up_silu failed (%d) layer %d", err, layer_idx_);
            return nullptr;
        }
    }

    // Allocate output buffer (still per-call — caller takes ownership).
    cl_mem out = clCreateBuffer(ctx, CL_MEM_READ_WRITE, (size_t)seq_len * hidden * sizeof(nnopt_storage_t), nullptr, &err);
    if (err != CL_SUCCESS || !out) {
        NNOPT_ERROR_FMT("SharedMlp: clCreateBuffer out failed (%d)", err);
        return nullptr;
    }

    if (use_fused_v5) {
        // Down-proj + residual via mlp_down_residual_m1_v2 (reads gated, not proj_halves).
        if (clSetKernelArg(mlp_down_residual_kernel_v2_, 0, sizeof(cl_mem), &gated_)         != CL_SUCCESS ||
            clSetKernelArg(mlp_down_residual_kernel_v2_, 1, sizeof(cl_mem), &w_out_)         != CL_SUCCESS ||
            clSetKernelArg(mlp_down_residual_kernel_v2_, 2, sizeof(cl_mem), &residual)       != CL_SUCCESS ||
            clSetKernelArg(mlp_down_residual_kernel_v2_, 3, sizeof(cl_mem), &out)            != CL_SUCCESS ||
            clSetKernelArg(mlp_down_residual_kernel_v2_, 4, sizeof(int),    &inter)          != CL_SUCCESS ||
            clSetKernelArg(mlp_down_residual_kernel_v2_, 5, sizeof(int),    &hidden)         != CL_SUCCESS ||
            clSetKernelArg(mlp_down_residual_kernel_v2_, 6, sizeof(float),  &residual_scale) != CL_SUCCESS) {
            NNOPT_ERROR_FMT("SharedMlp: setarg mlp_down_residual_v2 failed layer %d", layer_idx_);
            clReleaseMemObject(out);
            return nullptr;
        }
        const size_t WG = 64;
        const size_t gws = (size_t)hidden * WG;
        const size_t lws = WG;
        err = clEnqueueNDRangeKernel(queue, mlp_down_residual_kernel_v2_, 1, nullptr, &gws, &lws,
                                     0, nullptr, KernelProfiler::event_for("mlp_down_residual_m1_v2"));
        if (err != CL_SUCCESS) {
            NNOPT_ERROR_FMT("SharedMlp: dispatch mlp_down_residual_v2 failed (%d) layer %d", err, layer_idx_);
            clReleaseMemObject(out);
            return nullptr;
        }
        NNOPT_LAYER_CHECK_FMT("shared_mlp_%d", layer_idx_, queue, out, in_elems);
        return out;
    }

    // Fused SwiGLU + output_linear: out[s,h] = sum_i silu(proj[s,i]) * proj[s,inter+i] * w_out[h,i]
    // Reduction kept in fp32 — never stores the silu(gate)*up intermediate (which can
    // exceed fp16 max 65504 in Granite's bf16-trained activations).
    //
    // Use the v2 GEMV-cooperative kernel when cols is divisible by WG_SIZE*4
    // (=256), which is the case for Granite's intermediate=2048. WG=64 means
    // gws[0] = seq * hidden * 64, lws[0] = 64. Each workgroup computes one
    // output element via a 64-lane vec4 reduction.
    const int WG_SIZE = 64;
    const bool use_v2 = (inter % (WG_SIZE * 4) == 0);
    // Step #8: prefer the image2d_t variant when M==1 and the image was
    // successfully created at init. The image kernel has the same signature
    // shape as v2 except arg 1 is `image2d_t w_out_img` instead of cl_mem.
    const bool use_v2_img = use_v2 && (seq_len == 1) && fused_output_kernel_v2_img_ && w_out_img_;
    cl_kernel k = use_v2_img ? fused_output_kernel_v2_img_
                              : (use_v2 ? fused_output_kernel_v2_ : fused_output_kernel_);
    const char* prof_label = use_v2_img ? "swiglu_fused_output_v2_img"
                                         : (use_v2 ? "swiglu_fused_output_v2" : "swiglu_fused_output");
    // Step 8: kernel signature (proj, w_out, residual, out, seq, cols, hidden, scale).
    // The v1 (legacy) kernel has the old (proj, w_out, out, seq, cols, hidden)
    // signature — only the v2 path supports fusion. v1 is the fallback.
    if (use_v2) {
        err = clSetKernelArg(k, 0, sizeof(cl_mem), &proj);
        if (err != CL_SUCCESS) { NNOPT_ERROR_FMT("SharedMlp v2: setarg proj (%d)", err); clReleaseMemObject(out); return nullptr; }
        // Arg 1 is either cl_mem buffer (v2) or cl_mem image (v2_img — same C type).
        cl_mem w_arg = use_v2_img ? w_out_img_ : w_out_;
        err = clSetKernelArg(k, 1, sizeof(cl_mem), &w_arg);
        if (err != CL_SUCCESS) { NNOPT_ERROR_FMT("SharedMlp v2: setarg w_out (%d)", err); clReleaseMemObject(out); return nullptr; }
        err = clSetKernelArg(k, 2, sizeof(cl_mem), &residual);
        if (err != CL_SUCCESS) { NNOPT_ERROR_FMT("SharedMlp v2: setarg residual (%d)", err); clReleaseMemObject(out); return nullptr; }
        err = clSetKernelArg(k, 3, sizeof(cl_mem), &out);
        if (err != CL_SUCCESS) { NNOPT_ERROR_FMT("SharedMlp v2: setarg out (%d)", err); clReleaseMemObject(out); return nullptr; }
        err = clSetKernelArg(k, 4, sizeof(int), &seq_len);
        if (err != CL_SUCCESS) { NNOPT_ERROR_FMT("SharedMlp v2: setarg seq (%d)", err); clReleaseMemObject(out); return nullptr; }
        err = clSetKernelArg(k, 5, sizeof(int), &inter);
        if (err != CL_SUCCESS) { NNOPT_ERROR_FMT("SharedMlp v2: setarg cols (%d)", err); clReleaseMemObject(out); return nullptr; }
        err = clSetKernelArg(k, 6, sizeof(int), &hidden);
        if (err != CL_SUCCESS) { NNOPT_ERROR_FMT("SharedMlp v2: setarg hidden (%d)", err); clReleaseMemObject(out); return nullptr; }
        err = clSetKernelArg(k, 7, sizeof(float), &residual_scale);
        if (err != CL_SUCCESS) { NNOPT_ERROR_FMT("SharedMlp v2: setarg residual_scale (%d)", err); clReleaseMemObject(out); return nullptr; }
    } else {
        // v1 fallback: no residual fusion. Caller must do element_add after.
        err = clSetKernelArg(k, 0, sizeof(cl_mem), &proj);
        if (err != CL_SUCCESS) { NNOPT_ERROR_FMT("SharedMlp v1: setarg proj (%d)", err); clReleaseMemObject(out); return nullptr; }
        err = clSetKernelArg(k, 1, sizeof(cl_mem), &w_out_);
        if (err != CL_SUCCESS) { NNOPT_ERROR_FMT("SharedMlp v1: setarg w_out (%d)", err); clReleaseMemObject(out); return nullptr; }
        err = clSetKernelArg(k, 2, sizeof(cl_mem), &out);
        if (err != CL_SUCCESS) { NNOPT_ERROR_FMT("SharedMlp v1: setarg out (%d)", err); clReleaseMemObject(out); return nullptr; }
        err = clSetKernelArg(k, 3, sizeof(int), &seq_len);
        if (err != CL_SUCCESS) { NNOPT_ERROR_FMT("SharedMlp v1: setarg seq (%d)", err); clReleaseMemObject(out); return nullptr; }
        err = clSetKernelArg(k, 4, sizeof(int), &inter);
        if (err != CL_SUCCESS) { NNOPT_ERROR_FMT("SharedMlp v1: setarg cols (%d)", err); clReleaseMemObject(out); return nullptr; }
        err = clSetKernelArg(k, 5, sizeof(int), &hidden);
        if (err != CL_SUCCESS) { NNOPT_ERROR_FMT("SharedMlp v1: setarg hidden (%d)", err); clReleaseMemObject(out); return nullptr; }
    }

    const int out_total = seq_len * hidden;
    if (use_v2) {
        size_t gws_v2 = (size_t)out_total * (size_t)WG_SIZE;
        size_t lws_v2 = (size_t)WG_SIZE;
        err = clEnqueueNDRangeKernel(queue, k, 1, nullptr, &gws_v2, &lws_v2, 0, nullptr, KernelProfiler::event_for(prof_label));
    } else {
        size_t gws_fused = (size_t)out_total;
        err = clEnqueueNDRangeKernel(queue, k, 1, nullptr, &gws_fused, nullptr, 0, nullptr, KernelProfiler::event_for(prof_label));
    }
    if (err != CL_SUCCESS) {
        NNOPT_ERROR_FMT("SharedMlp: enqueue %s failed (%d)", prof_label, err);
        clReleaseMemObject(out);
        return nullptr;
    }
    // proj_ is persistent; do NOT release.
    // Per-instance dump name matching reference/layers/shared_mlp_K_output.bin.
    // The legacy "block%d_sub_mlp" name didn't match the reference manifest
    // (dump_name_map expects "shared_mlp_K"), so SxS reported every MLP as
    // "missing dump" and the agent had no signal on MLP correctness.
    NNOPT_LAYER_CHECK_FMT("shared_mlp_%d", layer_idx_, queue, out, in_elems);
    return out;
}
