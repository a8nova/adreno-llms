// QuantizedEmbedding.cpp — shared implementation for ALL 2 QuantizedEmbedding node(s).
//
// You write the QuantizedEmbedding forward() math ONCE here. backbone.cpp calls
// this function once per traced node of class QuantizedEmbedding, passing the
// appropriate weight_prefix string for each call — your implementation
// uses that prefix to load weights, so the same C++ math correctly
// handles every sibling instance.
//
// Sibling nodes (order  dump_name  weight_prefix):
//      0  sampler.depthformer.decoder.embedder.layers.0._embedding  weight_prefix=sampler.depthformer.decoder.embedder.layers.0._embedding
//      1  sampler.depthformer.encoder.body.layers.0.layers.0.layers.1.layers.1._embedding  weight_prefix=sampler.depthformer.encoder.body.layers.0.layers.0.layers.1.layers.1._embedding
//
// Representative shape (from sibling 0):
//   input:  [1, 1, 12]
//   output: []
//
// Primary reference dump for cosine validation:
//   reference/layers/sampler.depthformer.decoder.embedder.layers.0._embedding_output.bin
//   (per-node cosine validation runs against EVERY sibling's dump too —
//    if your math is correct for one weight_prefix, it's correct for all.)
//
// ─── SIGNATURE ──────────────────────────────────────────────────────────
//
// Hardened universal signature. EVERY <Class>_forward in this scaffold
// uses these 10 params, so backbone.cpp can call any op uniformly:
//
//   cl_ctx          — OpenCLContext& (queue, device, context)
//   weights         — Weights& (use `weights.get_buffer(wp + ".<param>")`)
//   queue           — cl_command_queue for kernel dispatch
//   input           — cl_mem of the input tensor (int32 for Embedding;
//                     nnopt_storage_t for everything else)
//   seq_len         — T dimension of input
//   layer_idx       — 0..NUM_LAYERS-1 for per-layer ops; -1 for global
//                     ops (Embedding, final norm, lm_head). Use this for
//                     model_config arrays like NUM_QUERY_HEADS[layer_idx].
//   start_pos       — generation step offset for decode (0 during prefill;
//                     prompt_len + step during decode). Needed by ops with
//                     persistent state (attention KV cache, embedding wpe).
//   k_cache_inout   — pointer to layer's K-cache cl_mem (attention only;
//                     other ops ignore). Caller owns the buffer.
//   v_cache_inout   — pointer to layer's V-cache cl_mem (attention only).
//   encoder_hidden_states — cl_mem of the encoder's output for cross-attention
//                     (encoder-decoder models like Whisper, T5, SeamlessM4T).
//                     PASSED for decoder.encoder_attn calls; nullptr everywhere
//                     else. When non-null, the attention op uses
//                     encoder_hidden_states as the K/V source (Q from input).
//   weight_prefix   — state_dict prefix (use `std::string(wp) + ".<p>"`).
//
// Returns: cl_mem (newly-allocated) holding this op's output. Caller owns
// and releases. Return `nullptr` on any internal error (after calling
// NNOPT_ERROR_FMT for the log).
//
// ─── IMPLEMENTATION CHECKLIST ──────────────────────────────────────────
//
// 1. Add a citation comment in the first 40 lines:
//    `// Reference: model_info/transformers_src/<file>.py:<lines> QuantizedEmbedding.forward`
//    (Build refuses to compile without it.)
// 2. Load every weight the PyTorch forward() touches via
//    `weights.get_buffer(std::string(weight_prefix) + ".<param>")`.
//    Use `weights.get_shape(wp + ".weight")` for dimensions when needed.
// 3. Allocate output buffer:
//    `cl_mem out = pool_alloc(cl_ctx.context(), CL_MEM_READ_WRITE,
//                                  numel * sizeof(nnopt_storage_t),
//                                  nullptr, &err);`
// 4. Dispatch your kernel(s); `clFinish(queue)` before returning.
// 5. NEVER return a passthrough or zeros — implement the real math or
//    keep the NNOPT_ERROR sentinel (per AUTONOMOUS_PORTING.md §0a).
//
// ─── ARGS YOU PROBABLY DON'T NEED ──────────────────────────────────────
//
// Mark unused args with `(void)arg;` to silence warnings. Common per-class:
//   - Embedding/Linear/Norm/Activation: ignore layer_idx, start_pos, k/v_cache, encoder_hidden_states
//   - RotaryEmbedding: ignore k/v_cache, encoder_hidden_states
//   - Self-attention: USE input, k_cache_inout, v_cache_inout; ignore encoder_hidden_states
//   - Cross-attention (decoder.encoder_attn): USE input (Q source), encoder_hidden_states (K/V source); ignore k/v_cache
//   - WhisperAttention etc. (dual-mode): branch on `encoder_hidden_states != nullptr` to pick self vs cross
//   - MLP: ignore layer_idx, start_pos, k/v_cache, encoder_hidden_states
//   - DecoderLayer (encoder-decoder): USE layer_idx, start_pos, k/v_cache, encoder_hidden_states

#include "../opencl_context.h"
#include "../weights.h"
#include "../debug_utils.h"
#include "../utils.h"

#include <clblast.h>
#include <string>
#include <vector>

// Reference: /Users/alazarshenkute/.nnopt/repos/magenta-realtime/magenta_rt/mlx/gptq.py (QuantizedLinear packing conventions)
// Reference: reference/forward_graph.json nodes[0..1] (op=QuantizedEmbedding, input=int ids, output=embed vectors)
// NOTE: This project is MLX-native; there is no modeling_*.py. We cite the MLX repo as the source-of-truth.

namespace {
static cl_program g_prog = nullptr;
static cl_kernel g_kernel = nullptr;

static bool ensure_kernel(OpenCLContext& cl_ctx) {
  if (g_kernel) return true;
  // fp16-storage gather: weights are dequantized dense fp16 (vload_half); output fp32.
  g_prog = cl_ctx.build_program_from_file("kernels/embedding_gather_f16.cl");
  if (!g_prog) {
    NNOPT_ERROR("QuantizedEmbedding: failed to build kernels/embedding_gather_f16.cl");
    return false;
  }
  cl_int err = CL_SUCCESS;
  g_kernel = clCreateKernel(g_prog, "embedding_gather_f16", &err);
  if (err != CL_SUCCESS || !g_kernel) {
    NNOPT_ERROR_FMT("QuantizedEmbedding: clCreateKernel(embedding_gather_f16) failed (err=%d)", err);
    return false;
  }
  return true;
}

static bool set_arg(cl_kernel k, int idx, size_t sz, const void* v, const char* name) {
  cl_int err = clSetKernelArg(k, idx, sz, v);
  if (err != CL_SUCCESS) {
    NNOPT_ERROR_FMT("QuantizedEmbedding: clSetKernelArg(%d:%s) failed (err=%d)", idx, name, err);
    return false;
  }
  return true;
}
}  // namespace

extern "C" {
cl_mem QuantizedEmbedding_forward(
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
    const char* weight_prefix) {
  (void)layer_idx;
  (void)start_pos;
  (void)k_cache_inout;
  (void)v_cache_inout;
  (void)encoder_hidden_states;

  if (!weight_prefix || !*weight_prefix) {
    NNOPT_ERROR("QuantizedEmbedding: null/empty weight_prefix");
    return nullptr;
  }
  if (!input) {
    NNOPT_ERROR("QuantizedEmbedding: input is null");
    return nullptr;
  }

  if (!ensure_kernel(cl_ctx)) return nullptr;

  const std::string wp(weight_prefix);

  // Expect an embedding table at <wp>.weight with shape [vocab, dim]
  const std::string w_key = wp + ".weight";
  cl_mem table = weights.get_buffer(w_key);
  if (!table) {
    NNOPT_ERROR_FMT("QuantizedEmbedding: missing weight tensor %s", w_key.c_str());
    return nullptr;
  }

  const std::vector<int> shape = weights.get_shape(w_key);
  if (shape.size() != 2) {
    NNOPT_ERROR_FMT("QuantizedEmbedding: expected %s rank=2, got rank=%d", w_key.c_str(), (int)shape.size());
    return nullptr;
  }
  const int vocab = shape[0];
  const int dim = shape[1];

  cl_int err = CL_SUCCESS;
  const size_t out_elems = (size_t)seq_len * (size_t)dim;
  cl_mem out = pool_alloc(cl_ctx.context(), CL_MEM_READ_WRITE,
                              out_elems * sizeof(float), nullptr, &err);
  if (err != CL_SUCCESS || !out) {
    NNOPT_ERROR_FMT("QuantizedEmbedding: clCreateBuffer(out) failed (err=%d)", err);
    return nullptr;
  }

  // Kernel: embedding_gather_f32(__global const float* table,
  //                             __global const int* ids,
  //                             __global float* out,
  //                             int dim, int n_ids, int vocab)
  if (!set_arg(g_kernel, 0, sizeof(cl_mem), &table, "table")) return nullptr;
  if (!set_arg(g_kernel, 1, sizeof(cl_mem), &input, "ids")) return nullptr;
  if (!set_arg(g_kernel, 2, sizeof(cl_mem), &out, "out")) return nullptr;
  if (!set_arg(g_kernel, 3, sizeof(int), &dim, "dim")) return nullptr;
  if (!set_arg(g_kernel, 4, sizeof(int), &seq_len, "n_ids")) return nullptr;
  if (!set_arg(g_kernel, 5, sizeof(int), &vocab, "vocab")) return nullptr;

  const size_t gws[2] = {(size_t)seq_len, (size_t)dim};
  err = clEnqueueNDRangeKernel(queue, g_kernel, 2, nullptr, gws, nullptr, 0, nullptr, nullptr);
  if (err != CL_SUCCESS) {
    NNOPT_ERROR_FMT("QuantizedEmbedding: enqueue kernel failed (err=%d)", err);
    pool_free(out);
    return nullptr;
  }
  

  return out;
}
}
