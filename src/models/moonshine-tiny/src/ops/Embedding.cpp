// Embedding.cpp — decoder token embedding (nn.Embedding).
// Reference: model_info/transformers_src/modeling_moonshine.py:624-660 MoonshineDecoder.forward
//   self.embed_tokens = nn.Embedding(vocab_size, hidden_size, padding_idx)
//   inputs_embeds = self.embed_tokens(input_ids)
// weight_prefix = "model.decoder.embed_tokens"; weight [vocab, hidden].
// input is a cl_mem of int32 token ids [seq_len]. Output [seq_len, hidden].

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
cl_mem Embedding_forward(
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

    const std::string wp = weight_prefix && *weight_prefix
        ? std::string(weight_prefix)
        : std::string("model.decoder.embed_tokens");

    std::vector<int> wshape = weights.get_shape(wp + ".weight");
    if (wshape.size() != 2) {
        NNOPT_ERROR_FMT("Embedding: bad weight shape for %s (rank=%zu)", wp.c_str(), wshape.size());
        return nullptr;
    }
    const int dim = wshape[1];  // hidden_size
    const int T = seq_len;

    cl_mem table = weights.get_buffer(wp + ".weight");
    if (!table) { NNOPT_ERROR_FMT("Embedding: missing %s.weight", wp.c_str()); return nullptr; }
    if (!input) { NNOPT_ERROR("Embedding: null input ids buffer"); return nullptr; }

    cl_int err = CL_SUCCESS;
    cl_mem out = clCreateBuffer(cl_ctx.context(), CL_MEM_READ_WRITE,
                                (size_t)T * dim * sizeof(nnopt_storage_t), nullptr, &err);
    if (err != CL_SUCCESS || !out) { NNOPT_ERROR_FMT("Embedding: alloc out %d", err); return nullptr; }

    cl_kernel k = moonshine_kernel(cl_ctx, "embed_gather");
    if (!k) { NNOPT_ERROR("Embedding: create kernel"); clReleaseMemObject(out); return nullptr; }

    bool ok = set_arg_checked(k, 0, sizeof(cl_mem), &input, "ids") &&
              set_arg_checked(k, 1, sizeof(cl_mem), &table, "table") &&
              set_arg_checked(k, 2, sizeof(cl_mem), &out, "output") &&
              set_arg_checked(k, 3, sizeof(int), &T, "T") &&
              set_arg_checked(k, 4, sizeof(int), &dim, "dim");
    if (!ok) { clReleaseMemObject(out); return nullptr; }

    size_t gws = (size_t)T * dim;
    err = clEnqueueNDRangeKernel(queue, k, 1, nullptr, &gws, nullptr, 0, nullptr,
                                 KernelProfiler::event_for("embed_gather"));
    if (err != CL_SUCCESS) { NNOPT_ERROR_FMT("Embedding: dispatch %d", err); clReleaseMemObject(out); return nullptr; }
    // OPT-8 (r6): no clFinish — the queue is in-order, downstream kernels see
    // this output without an explicit sync (SYNC-01). The old blocking sync
    // here stalled the pipeline once per decode step.
    return out;
}
}
