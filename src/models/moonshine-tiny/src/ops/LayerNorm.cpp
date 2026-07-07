// LayerNorm.cpp — nn.LayerNorm(hidden_size, bias=False) used everywhere in
// Moonshine (encoder/decoder input/post_attention/final/model norms).
// Reference: model_info/transformers_src/modeling_moonshine.py
//   self.input_layernorm = nn.LayerNorm(config.hidden_size, bias=False)
//   → normalize over last dim, scale by weight, NO bias, eps=1e-5.
//
// weight_prefix is the FULL weight-key base, e.g.
//   "model.encoder.layers.0.input_layernorm" → loads ".weight"
// seq_len = rows. Feature dim (cols) derived from weight shape.

#include "../opencl_context.h"
#include "../weights.h"
#include "../nnopt_error.h"
#include "../model_config.h"
#include "../utils.h"
#include "../profiler.h"
#include <string>
#include <vector>

// OPT-6: process-lifetime kernel cache (moonshine_common.cpp) — do not release.
cl_kernel moonshine_kernel(OpenCLContext& cl_ctx, const char* name);

extern "C" {
cl_mem LayerNorm_forward(
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
    (void)layer_idx; (void)start_pos;
    (void)k_cache_inout; (void)v_cache_inout; (void)encoder_hidden_states;

    const std::string wp = weight_prefix ? std::string(weight_prefix) : std::string();
    if (wp.empty()) { NNOPT_ERROR("LayerNorm: empty weight_prefix"); return nullptr; }

    std::vector<int> wshape = weights.get_shape(wp + ".weight");
    if (wshape.size() != 1) {
        NNOPT_ERROR_FMT("LayerNorm: bad weight shape for %s (rank=%zu)", wp.c_str(), wshape.size());
        return nullptr;
    }
    const int cols = wshape[0];
    const int rows = seq_len;

    cl_mem w_buf = weights.get_buffer(wp + ".weight");
    if (!w_buf) { NNOPT_ERROR_FMT("LayerNorm: missing %s.weight", wp.c_str()); return nullptr; }

    cl_int err = CL_SUCCESS;
    cl_mem out = clCreateBuffer(cl_ctx.context(), CL_MEM_READ_WRITE,
                               (size_t)rows * cols * sizeof(nnopt_storage_t), nullptr, &err);
    if (err != CL_SUCCESS || !out) { NNOPT_ERROR_FMT("LayerNorm: alloc out %d", err); return nullptr; }

    cl_kernel kernel = moonshine_kernel(cl_ctx, "layernorm_nobias");
    if (!kernel) { NNOPT_ERROR("LayerNorm: create kernel"); clReleaseMemObject(out); return nullptr; }

    const float eps = 1e-5f;
    auto cleanup = [&]() -> cl_mem { clReleaseMemObject(out); return nullptr; };
    const size_t lws = 64;   // one WG per row; kernel requires gws == rows*lws
    if (!set_arg_checked(kernel, 0, sizeof(cl_mem), &input, "input")) return cleanup();
    if (!set_arg_checked(kernel, 1, sizeof(cl_mem), &w_buf, "weight")) return cleanup();
    if (!set_arg_checked(kernel, 2, sizeof(cl_mem), &out, "output")) return cleanup();
    if (!set_arg_checked(kernel, 3, sizeof(int), &rows, "rows")) return cleanup();
    if (!set_arg_checked(kernel, 4, sizeof(int), &cols, "cols")) return cleanup();
    if (!set_arg_checked(kernel, 5, sizeof(float), &eps, "eps")) return cleanup();
    if (clSetKernelArg(kernel, 6, lws * sizeof(float), nullptr) != CL_SUCCESS) return cleanup();

    size_t gws = (size_t)rows * lws;
    err = clEnqueueNDRangeKernel(queue, kernel, 1, nullptr, &gws, &lws, 0, nullptr,
                                 KernelProfiler::event_for("layernorm_nobias"));
    if (err != CL_SUCCESS) { NNOPT_ERROR_FMT("LayerNorm: dispatch %d", err); return cleanup(); }
    return out;
}
}
