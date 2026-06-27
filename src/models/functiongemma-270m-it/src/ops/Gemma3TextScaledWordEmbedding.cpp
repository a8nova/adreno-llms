// Gemma3TextScaledWordEmbedding — token embedding lookup scaled by sqrt(hidden).
//
// Reference: model_info/transformers_src/modeling_gemma3.py:11-18
//   Gemma3TextScaledWordEmbedding.forward(input_ids):
//     return super().forward(input_ids) * embed_scale
//   embed_scale = config.hidden_size ** 0.5   (= sqrt(640))
//
// `input` is the int32 token-ids buffer (NOT storage_t). Output is
// [seq_len, hidden] in storage_t.

#include "../opencl_context.h"
#include "../weights.h"
#include "../debug_utils.h"
#include "../model_config.h"
#include "../utils.h"
#include "../profiler.h"
#include <string>
#include <cmath>

namespace {
cl_program g_prog = nullptr;
cl_kernel  g_kern = nullptr;
bool ensure_kernel(OpenCLContext& cl_ctx) {
    if (g_kern) return true;
    g_prog = cl_ctx.build_program_from_file("kernels/gemma3_ops.cl");  // PROGRAM-INIT-OK
    if (!g_prog) { NNOPT_ERROR("Embedding: build gemma3_ops.cl failed"); return false; }
    cl_int err = CL_SUCCESS;
    g_kern = clCreateKernel(g_prog, "gemma3_embed", &err);
    if (err != CL_SUCCESS || !g_kern) { NNOPT_ERROR_FMT("Embedding: clCreateKernel %d", err); return false; }
    return true;
}
} // namespace

extern "C" {
cl_mem Gemma3TextScaledWordEmbedding_forward(
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
    if (!ensure_kernel(cl_ctx)) return nullptr;
    const std::string wp = weight_prefix ? std::string(weight_prefix) : std::string();
    cl_mem table = weights.get_buffer(wp + ".weight");
    if (!table) { NNOPT_ERROR_FMT("Embedding: missing %s.weight", wp.c_str()); return nullptr; }

    const int hidden = MODEL_CONFIG::HIDDEN_SIZE;
    const float embed_scale = std::sqrt((float)hidden);

    cl_int err = CL_SUCCESS;
    cl_mem out = nnopt_pool_alloc(cl_ctx.context(),
                                (size_t)seq_len * hidden * sizeof(nnopt_storage_t), &err);
    if (err != CL_SUCCESS || !out) { NNOPT_ERROR_FMT("Embedding: alloc out %d", err); return nullptr; }
    auto cleanup = [&]() -> cl_mem { if (out) { nnopt_pool_free(out); out = nullptr; } return nullptr; };

    if (!set_arg_checked(g_kern, 0, sizeof(cl_mem), &input, "ids")) return cleanup();
    if (!set_arg_checked(g_kern, 1, sizeof(cl_mem), &table, "table")) return cleanup();
    if (!set_arg_checked(g_kern, 2, sizeof(cl_mem), &out, "out")) return cleanup();
    if (!set_arg_checked(g_kern, 3, sizeof(int), &hidden, "hidden")) return cleanup();
    if (!set_arg_checked(g_kern, 4, sizeof(float), &embed_scale, "embed_scale")) return cleanup();

    size_t gws = (size_t)seq_len * hidden;
    err = clEnqueueNDRangeKernel(queue, g_kern, 1, nullptr, &gws, nullptr,
                                 0, nullptr, KernelProfiler::event_for("op_gemma3_embed"));
    if (err != CL_SUCCESS) { NNOPT_ERROR_FMT("Embedding: dispatch %d", err); return cleanup(); }
    return out;
}
}
