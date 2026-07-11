// Gemma3RMSNorm — shared impl for all RMSNorm nodes (input/post-attn/pre-ff/
// post-ff/q_norm/k_norm/final norm).
//
// Reference: model_info/transformers_src/modeling_gemma3.py:34-48 Gemma3RMSNorm.forward
//   _norm(x) = x * rsqrt(mean(x^2, -1) + eps)         (computed in fp32)
//   output   = _norm(x) * (1.0 + weight.float())      (the +1 offset is CRITICAL)
//
// Universal signature (see backbone.cpp). cols = last-dim size, derived from
// the weight tensor shape (640 for hidden norms, 256 for q/k head-dim norms).

#include "../opencl_context.h"
#include "../weights.h"
#include "../debug_utils.h"
#include "../model_config.h"
#include "../utils.h"
#include "../profiler.h"
#include <string>
#include <vector>

namespace {
cl_program g_rmsnorm_prog = nullptr;
cl_kernel  g_rmsnorm_kern = nullptr;
cl_kernel  g_rmsnorm_f32in_kern = nullptr;
cl_kernel  g_f16_to_f32_kern = nullptr;

bool ensure_kernel(OpenCLContext& cl_ctx) {
    if (g_rmsnorm_kern && g_rmsnorm_f32in_kern && g_f16_to_f32_kern) return true;
    if (!g_rmsnorm_prog) {
        g_rmsnorm_prog = cl_ctx.build_program_from_file("kernels/gemma3_ops.cl");  // PROGRAM-INIT-OK
        if (!g_rmsnorm_prog) { NNOPT_ERROR("RMSNorm: build gemma3_ops.cl failed"); return false; }
    }
    cl_int err = CL_SUCCESS;
    if (!g_rmsnorm_kern) {
        g_rmsnorm_kern = clCreateKernel(g_rmsnorm_prog, "gemma3_rmsnorm", &err);
        if (err != CL_SUCCESS || !g_rmsnorm_kern) { NNOPT_ERROR_FMT("RMSNorm: clCreateKernel %d", err); return false; }
    }
    if (!g_rmsnorm_f32in_kern) {
        g_rmsnorm_f32in_kern = clCreateKernel(g_rmsnorm_prog, "gemma3_rmsnorm_f32in", &err);
        if (err != CL_SUCCESS || !g_rmsnorm_f32in_kern) { NNOPT_ERROR_FMT("RMSNorm: clCreateKernel f32in %d", err); return false; }
    }
    if (!g_f16_to_f32_kern) {
        g_f16_to_f32_kern = clCreateKernel(g_rmsnorm_prog, "gemma3_f16_to_f32", &err);
        if (err != CL_SUCCESS || !g_f16_to_f32_kern) { NNOPT_ERROR_FMT("RMSNorm: clCreateKernel f16_to_f32 %d", err); return false; }
    }
    return true;
}
} // namespace

// Internal entry usable by DecoderLayer for sub-norms. rows*cols elements.
cl_mem gemma3_rmsnorm_run(OpenCLContext& cl_ctx, cl_command_queue queue,
                          cl_mem input, int rows, int cols, float eps,
                          cl_mem weight_buf) {
    if (!ensure_kernel(cl_ctx)) return nullptr;
    cl_int err = CL_SUCCESS;
    cl_mem out = nnopt_pool_alloc(cl_ctx.context(), (size_t)rows * cols * sizeof(nnopt_storage_t), &err);
    if (err != CL_SUCCESS || !out) { NNOPT_ERROR_FMT("RMSNorm: alloc out %d", err); return nullptr; }
    auto cleanup = [&]() -> cl_mem { if (out) { nnopt_pool_free(out); out = nullptr; } return nullptr; };

    if (!set_arg_checked(g_rmsnorm_kern, 0, sizeof(cl_mem), &input, "x")) return cleanup();
    if (!set_arg_checked(g_rmsnorm_kern, 1, sizeof(cl_mem), &weight_buf, "weight")) return cleanup();
    if (!set_arg_checked(g_rmsnorm_kern, 2, sizeof(cl_mem), &out, "out")) return cleanup();
    if (!set_arg_checked(g_rmsnorm_kern, 3, sizeof(int), &cols, "cols")) return cleanup();
    if (!set_arg_checked(g_rmsnorm_kern, 4, sizeof(float), &eps, "eps")) return cleanup();

    size_t lws = 64;  // must match RMS_WG in gemma3_ops.cl
    size_t gws = (size_t)rows * lws;
    err = clEnqueueNDRangeKernel(queue, g_rmsnorm_kern, 1, nullptr, &gws, &lws,
                                 0, nullptr, KernelProfiler::event_for("op_gemma3_rmsnorm"));
    if (err != CL_SUCCESS) { NNOPT_ERROR_FMT("RMSNorm: dispatch %d", err); return cleanup(); }
    return out;
}

// fp32-residual variant: input is a RAW fp32 buffer (the residual stream).
// Output is fp16 (normalized values are small). Used by DecoderLayer for the
// input_layernorm and pre_feedforward_layernorm (both read the residual), and
// by the final model.norm. Gemma3's residual exceeds fp16 max mid-stack, so it
// must be stored in fp32; this kernel reads it without overflow.
cl_mem gemma3_rmsnorm_run_f32in(OpenCLContext& cl_ctx, cl_command_queue queue,
                                cl_mem input_f32, int rows, int cols, float eps,
                                cl_mem weight_buf) {
    if (!ensure_kernel(cl_ctx)) return nullptr;
    cl_int err = CL_SUCCESS;
    cl_mem out = nnopt_pool_alloc(cl_ctx.context(), (size_t)rows * cols * sizeof(nnopt_storage_t), &err);
    if (err != CL_SUCCESS || !out) { NNOPT_ERROR_FMT("RMSNorm(f32in): alloc out %d", err); return nullptr; }
    auto cleanup = [&]() -> cl_mem { if (out) { nnopt_pool_free(out); out = nullptr; } return nullptr; };

    if (!set_arg_checked(g_rmsnorm_f32in_kern, 0, sizeof(cl_mem), &input_f32, "x")) return cleanup();
    if (!set_arg_checked(g_rmsnorm_f32in_kern, 1, sizeof(cl_mem), &weight_buf, "weight")) return cleanup();
    if (!set_arg_checked(g_rmsnorm_f32in_kern, 2, sizeof(cl_mem), &out, "out")) return cleanup();
    if (!set_arg_checked(g_rmsnorm_f32in_kern, 3, sizeof(int), &cols, "cols")) return cleanup();
    if (!set_arg_checked(g_rmsnorm_f32in_kern, 4, sizeof(float), &eps, "eps")) return cleanup();

    size_t lws = 64;  // must match RMS_WG in gemma3_ops.cl
    size_t gws = (size_t)rows * lws;
    err = clEnqueueNDRangeKernel(queue, g_rmsnorm_f32in_kern, 1, nullptr, &gws, &lws,
                                 0, nullptr, KernelProfiler::event_for("op_gemma3_rmsnorm_f32in"));
    if (err != CL_SUCCESS) { NNOPT_ERROR_FMT("RMSNorm(f32in): dispatch %d", err); return cleanup(); }
    return out;
}

// Convert an fp16 storage buffer to a fresh fp32 buffer (caller owns).
// Used in the backbone to seed the fp32 residual stream from the fp16
// embedding output.
cl_mem gemma3_f16_to_f32_run(OpenCLContext& cl_ctx, cl_command_queue queue,
                             cl_mem in_f16, int n) {
    if (!ensure_kernel(cl_ctx)) return nullptr;
    cl_int err = CL_SUCCESS;
    cl_mem out = nnopt_pool_alloc(cl_ctx.context(), (size_t)n * sizeof(float), &err);
    if (err != CL_SUCCESS || !out) { NNOPT_ERROR_FMT("f16_to_f32: alloc out %d", err); return nullptr; }
    auto cleanup = [&]() -> cl_mem { if (out) { nnopt_pool_free(out); out = nullptr; } return nullptr; };

    if (!set_arg_checked(g_f16_to_f32_kern, 0, sizeof(cl_mem), &in_f16, "in")) return cleanup();
    if (!set_arg_checked(g_f16_to_f32_kern, 1, sizeof(cl_mem), &out, "out")) return cleanup();
    if (!set_arg_checked(g_f16_to_f32_kern, 2, sizeof(int), &n, "n")) return cleanup();

    size_t gws = (size_t)n;
    err = clEnqueueNDRangeKernel(queue, g_f16_to_f32_kern, 1, nullptr, &gws, nullptr,
                                 0, nullptr, KernelProfiler::event_for("op_gemma3_f16_to_f32"));
    if (err != CL_SUCCESS) { NNOPT_ERROR_FMT("f16_to_f32: dispatch %d", err); return cleanup(); }
    return out;
}

extern "C" {
cl_mem Gemma3RMSNorm_forward(
    OpenCLContext& cl_ctx,
    Weights& weights,
    cl_command_queue queue,
    cl_mem input,
    int seq_len,
    int layer_idx,
    int start_pos,
    cl_mem* k_cache_inout,
    cl_mem* v_cache_inout,
    cl_mem encoder_hidden_states,
    const char* weight_prefix)
{
    (void)layer_idx; (void)start_pos; (void)k_cache_inout; (void)v_cache_inout;
    (void)encoder_hidden_states;
    const std::string wp = weight_prefix ? std::string(weight_prefix) : std::string();
    cl_mem w = weights.get_buffer(wp + ".weight");
    if (!w) { NNOPT_ERROR_FMT("RMSNorm: missing weight %s.weight", wp.c_str()); return nullptr; }
    std::vector<int> shp = weights.get_shape(wp + ".weight");
    int cols = shp.empty() ? MODEL_CONFIG::HIDDEN_SIZE : shp.back();
    // The final model.norm receives the fp32 residual stream from the decoder
    // stack, so it must use the f32-input variant.
    return gemma3_rmsnorm_run_f32in(cl_ctx, queue, input, seq_len, cols,
                                    MODEL_CONFIG::RMS_NORM_EPS, w);
}
}
