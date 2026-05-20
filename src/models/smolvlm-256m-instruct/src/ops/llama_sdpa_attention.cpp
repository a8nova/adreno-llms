// Reference: https://raw.githubusercontent.com/huggingface/transformers/main/src/transformers/models/llama/modeling_llama.py LlamaAttention.forward / eager_attention_forward
// This op implements the Llama-style decoder self-attention used by SmolVLM's text_model.
// It is a CAUSAL attention with KV cache (decode-aware) using:
//  - q_proj/k_proj/v_proj/o_proj weights
//  - RoPE applied to Q and K (absolute positions)
//  - GQA: num_q_heads != num_kv_heads (repeat_kv via nrep = q/k)
//  - attention kernels in kernels/attention.cl: gqa_attn_scores, gqa_softmax, gqa_attn_out
//
// NOTE: This is a minimal correctness-first implementation. Performance work (fused GEMV for decode) comes later.

#include "opencl_context.h"
#include "weights.h"
#include "debug_utils.h"
#include "utils.h"
#include "model_config.h"
#include "rotary_tables.h"
#include "profile.h"

#include <CL/cl.h>
#include <clblast.h>

#include <cctype>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <string>
#include <vector>

namespace {

// ─────────────────────────────────────────────────────────────────────
// Per-layer KV cache. Allocated lazily on first use per layer. Persists
// across forward() calls for the lifetime of the process.
//
// Layout: K[layer] is [KV_CACHE_MAX_LEN, num_kv_heads * head_dim] row-major,
// which matches the attention kernel's expected k_cache shape
// [seq_k, num_kv_heads, head_dim].
//
// RoPE has already been applied to K before it is written into the cache,
// per PyTorch convention (Q,K rotated, V un-rotated).
// ─────────────────────────────────────────────────────────────────────
static constexpr int KV_CACHE_MAX_LEN = 2048;  // > 874 prefill + decode budget

struct LayerKVCache {
  cl_mem K = nullptr;
  cl_mem V = nullptr;
  int kv_dim = 0;
  // Persistent decode scratch (seq_q=1): max-sized scores buffer reused across steps.
  cl_mem scores_decode = nullptr;
  size_t scores_decode_bytes = 0;
  // Per-layer decode scratch buffers, sized once at first call.
  cl_mem Q_decode = nullptr;
  cl_mem ctx_q_decode = nullptr;
  cl_mem out_decode = nullptr;
  int decode_q_dim = 0;       // == num_q_heads * head_dim
  int decode_hidden = 0;
  // Image2D views of V cache (for attn_out reads via L1 texture cache).
  // Width = kv_dim/4 (CL_RGBA half), height = KV_CACHE_MAX_LEN. Populated by
  // clEnqueueCopyBufferToImage immediately after each kv_cache_write.
  cl_mem V_image = nullptr;
  cl_mem K_image = nullptr;
};

std::vector<LayerKVCache>& kv_caches() {
  static std::vector<LayerKVCache> v;
  return v;
}

// Parse "model.text_model.layers.<N>.self_attn.<...>" → N.
// Returns -1 on no match.
int extract_layer_idx(const char* weight_key) {
  if (!weight_key) return -1;
  const std::string s(weight_key);
  const std::string anchor = "layers.";
  size_t pos = s.find(anchor);
  if (pos == std::string::npos) return -1;
  size_t start = pos + anchor.size();
  size_t end = start;
  while (end < s.size() && std::isdigit((unsigned char)s[end])) end++;
  if (end == start) return -1;
  return std::atoi(s.substr(start, end - start).c_str());
}

bool ensure_kv_cache(OpenCLContext& cl_ctx, int layer_idx, int kv_dim) {
  if (layer_idx < 0) return false;
  auto& caches = kv_caches();
  if ((int)caches.size() <= layer_idx) caches.resize((size_t)layer_idx + 1);
  LayerKVCache& c = caches[(size_t)layer_idx];
  if (c.K && c.V && c.kv_dim == kv_dim) return true;
  cl_int err = CL_SUCCESS;
  const size_t bytes = (size_t)KV_CACHE_MAX_LEN * (size_t)kv_dim * sizeof(nnopt_storage_t);
  c.K = clCreateBuffer(cl_ctx.context(), CL_MEM_READ_WRITE, bytes, nullptr, &err);
  if (err != CL_SUCCESS || !c.K) {
    NNOPT_ERROR_FMT("op_LlamaSdpaAttention: KV cache K alloc failed layer=%d (%d)", layer_idx, (int)err);
    return false;
  }
  c.V = clCreateBuffer(cl_ctx.context(), CL_MEM_READ_WRITE, bytes, nullptr, &err);
  if (err != CL_SUCCESS || !c.V) {
    NNOPT_ERROR_FMT("op_LlamaSdpaAttention: KV cache V alloc failed layer=%d (%d)", layer_idx, (int)err);
    return false;
  }
  c.kv_dim = kv_dim;

  // Allocate image2D views of K + V (CL_RGBA half, width=kv_dim/4, height=KV_CACHE_MAX_LEN).
  // Populated via clEnqueueCopyBufferToImage after every kv_cache_write. Bytes
  // are not duplicated by the device side — these are independent allocations,
  // but the cost (~768KB × 30 = 23MB per side) is tolerable on a 4GB phone for
  // the bandwidth win on attn_out / attn_scores reads.
  if ((kv_dim & 3) == 0) {
    cl_image_format fmt = {CL_RGBA, CL_HALF_FLOAT};
    cl_image_desc desc{};
    desc.image_type = CL_MEM_OBJECT_IMAGE2D;
    desc.image_width = (size_t)(kv_dim / 4);
    desc.image_height = (size_t)KV_CACHE_MAX_LEN;
    c.K_image = clCreateImage(cl_ctx.context(), CL_MEM_READ_WRITE, &fmt, &desc, nullptr, &err);
    if (err != CL_SUCCESS || !c.K_image) {
      fprintf(stderr, "[kv_image] clCreateImage(K) failed (err=%d) — falling back to buffer-only\n", (int)err);
      c.K_image = nullptr;
    }
    c.V_image = clCreateImage(cl_ctx.context(), CL_MEM_READ_WRITE, &fmt, &desc, nullptr, &err);
    if (err != CL_SUCCESS || !c.V_image) {
      fprintf(stderr, "[kv_image] clCreateImage(V) failed (err=%d) — falling back to buffer-only\n", (int)err);
      c.V_image = nullptr;
    }
  }

  return true;
}

struct TextAttnState {
  bool initialized = false;
  cl_program program = nullptr;
  cl_kernel k_scores = nullptr;
  cl_kernel k_scores_tiled = nullptr;
  cl_kernel k_softmax = nullptr;
  cl_kernel k_out = nullptr;
  cl_kernel k_out_tiled = nullptr;
  cl_kernel k_out_decode = nullptr;
  cl_kernel k_out_decode_image = nullptr; // gqa_attn_out_decode_image — decode V via texture cache
  cl_kernel k_out_image = nullptr;   // gqa_attn_out_image — V via texture cache
  cl_kernel k_scores_image = nullptr; // gqa_attn_scores_image — K via texture cache
  cl_kernel k_kv_write_decode = nullptr; // R6.3: fused 1-dispatch K/V write to buf+img
};

TextAttnState& st() {
  static TextAttnState s;
  return s;
}

bool ensure_initialized(OpenCLContext& cl_ctx) {
  auto& s = st();
  if (s.initialized) return true;

  s.program = cl_ctx.build_program_from_file("kernels/attention.cl");  // PROGRAM-INIT-OK
  if (!s.program) {
    NNOPT_ERROR("op_LlamaSdpaAttention: failed to build kernels/attention.cl");
    return false;
  }
  cl_int err = CL_SUCCESS;
  s.k_scores = clCreateKernel(s.program, "gqa_attn_scores", &err);
  if (err != CL_SUCCESS || !s.k_scores) {
    NNOPT_ERROR_FMT("op_LlamaSdpaAttention: clCreateKernel(gqa_attn_scores) failed (%d)", (int)err);
    return false;
  }
  s.k_scores_tiled = clCreateKernel(s.program, "gqa_attn_scores_tiled", &err);
  if (err != CL_SUCCESS || !s.k_scores_tiled) {
    NNOPT_ERROR_FMT("op_LlamaSdpaAttention: clCreateKernel(gqa_attn_scores_tiled) failed (%d)", (int)err);
    // Non-fatal: prefill fall-back to k_scores.
    s.k_scores_tiled = nullptr;
  }
  s.k_softmax = clCreateKernel(s.program, "gqa_softmax", &err);
  if (err != CL_SUCCESS || !s.k_softmax) {
    NNOPT_ERROR_FMT("op_LlamaSdpaAttention: clCreateKernel(gqa_softmax) failed (%d)", (int)err);
    return false;
  }
  s.k_out = clCreateKernel(s.program, "gqa_attn_out", &err);
  if (err != CL_SUCCESS || !s.k_out) {
    NNOPT_ERROR_FMT("op_LlamaSdpaAttention: clCreateKernel(gqa_attn_out) failed (%d)", (int)err);
    return false;
  }
  s.k_out_tiled = clCreateKernel(s.program, "gqa_attn_out_tiled", &err);
  if (err != CL_SUCCESS || !s.k_out_tiled) {
    NNOPT_ERROR_FMT("op_LlamaSdpaAttention: clCreateKernel(gqa_attn_out_tiled) failed (%d)", (int)err);
    s.k_out_tiled = nullptr;
  }
  s.k_out_decode = clCreateKernel(s.program, "gqa_attn_out_decode", &err);
  if (err != CL_SUCCESS || !s.k_out_decode) {
    NNOPT_ERROR_FMT("op_LlamaSdpaAttention: clCreateKernel(gqa_attn_out_decode) failed (%d)", (int)err);
    return false;
  }
  s.k_out_decode_image = clCreateKernel(s.program, "gqa_attn_out_decode_image", &err);
  if (err != CL_SUCCESS || !s.k_out_decode_image) {
    fprintf(stderr, "[attn] clCreateKernel(gqa_attn_out_decode_image) failed (%d) — image-backed decode attn_out disabled\n", (int)err);
    s.k_out_decode_image = nullptr;
  }
  s.k_out_image = clCreateKernel(s.program, "gqa_attn_out_image", &err);
  if (err != CL_SUCCESS || !s.k_out_image) {
    fprintf(stderr, "[attn] clCreateKernel(gqa_attn_out_image) failed (%d) — image-backed prefill attn_out disabled\n", (int)err);
    s.k_out_image = nullptr;
  }
  s.k_scores_image = clCreateKernel(s.program, "gqa_attn_scores_image", &err);
  if (err != CL_SUCCESS || !s.k_scores_image) {
    fprintf(stderr, "[attn] clCreateKernel(gqa_attn_scores_image) failed (%d) — image-backed scores disabled\n", (int)err);
    s.k_scores_image = nullptr;
  }
  s.k_kv_write_decode = clCreateKernel(s.program, "kv_cache_write_decode_fused", &err);
  if (err != CL_SUCCESS || !s.k_kv_write_decode) {
    fprintf(stderr, "[attn] clCreateKernel(kv_cache_write_decode_fused) failed (%d) — falling back to 4-dispatch path\n", (int)err);
    s.k_kv_write_decode = nullptr;
  }

  s.initialized = true;
  return true;
}

// Apply RoPE to q/k in-place.
// q: [seq_q, num_heads, head_dim]
// k: [seq_q, num_kv_heads, head_dim]
// cos/sin: [max_seq, head_dim]
// positions: absolute: start_pos + t
static cl_program build_rope_program(OpenCLContext& cl_ctx) {
  // PROGRAM-INIT-OK: embedded kernel built once.
  static cl_program prog = nullptr;
  static bool init = false;
  if (init) return prog;

  const char* src = R"CLC(
#ifdef USE_FP16
  #pragma OPENCL EXTENSION cl_khr_fp16 : enable
  typedef half storage_t;
  #define LOAD(p, i)     vload_half((i), (p))
  #define STORE(p, i, v) vstore_half((v), (i), (p))
#else
  typedef float storage_t;
  #define LOAD(p, i)     ((p)[(i)])
  #define STORE(p, i, v) ((p)[(i)] = (v))
#endif

// Applies RoPE on last dimension (head_dim) with precomputed cos/sin tables.
// x: [seq, heads, head_dim]
// cos/sin: [max_seq, head_dim]
__kernel void apply_rope_inplace(
    __global storage_t* x,
    __global const storage_t* cos,
    __global const storage_t* sin,
    const int seq,
    const int heads,
    const int head_dim,
    const int start_pos) {
  const int gid = (int)get_global_id(0); // one (t, h)
  const int t = gid / heads;
  const int h = gid - t * heads;
  if (t >= seq) return;

  const int pos = start_pos + t;
  const int base = (t * heads + h) * head_dim;
  const int cbase = pos * head_dim;
  const int half_dim = head_dim >> 1;

  for (int i = 0; i < half_dim; ++i) {
    const float c = (float)LOAD(cos, cbase + i);
    const float s = (float)LOAD(sin, cbase + i);

    const int i0 = base + i;
    const int i1 = base + i + half_dim;
    const float x0 = (float)LOAD(x, i0);
    const float x1 = (float)LOAD(x, i1);

    // rotate_half([x0,x1]) => [-x1, x0]
    const float y0 = x0 * c - x1 * s;
    const float y1 = x0 * s + x1 * c;

    STORE(x, i0, (storage_t)y0);
    STORE(x, i1, (storage_t)y1);
  }
}

// Combined Q/K RoPE in a single dispatch. gws = seq * (num_q_heads + num_kv_heads).
// Lanes [0, seq*num_q_heads) rotate q; lanes [seq*num_q_heads, gws) rotate k.
// Saves one kernel dispatch per layer per token on decode.
__kernel void apply_rope_qk_inplace(
    __global storage_t* q,
    __global storage_t* k,
    __global const storage_t* cos,
    __global const storage_t* sin,
    const int seq,
    const int num_q_heads,
    const int num_kv_heads,
    const int head_dim,
    const int start_pos) {
  const int gid = (int)get_global_id(0);
  const int q_total = seq * num_q_heads;
  const int kv_total = seq * num_kv_heads;
  if (gid >= q_total + kv_total) return;

  __global storage_t* x;
  int heads, h, t;
  if (gid < q_total) {
    x = q;
    heads = num_q_heads;
    t = gid / heads;
    h = gid - t * heads;
  } else {
    x = k;
    heads = num_kv_heads;
    const int g = gid - q_total;
    t = g / heads;
    h = g - t * heads;
  }
  if (t >= seq) return;

  const int pos = start_pos + t;
  const int base = (t * heads + h) * head_dim;
  const int cbase = pos * head_dim;
  const int half_dim = head_dim >> 1;

  for (int i = 0; i < half_dim; ++i) {
    const float c = (float)LOAD(cos, cbase + i);
    const float s = (float)LOAD(sin, cbase + i);
    const int i0 = base + i;
    const int i1 = base + i + half_dim;
    const float x0 = (float)LOAD(x, i0);
    const float x1 = (float)LOAD(x, i1);
    const float y0 = x0 * c - x1 * s;
    const float y1 = x0 * s + x1 * c;
    STORE(x, i0, (storage_t)y0);
    STORE(x, i1, (storage_t)y1);
  }
}
)CLC";

  prog = cl_ctx.build_program(std::string(src));
  init = true;
  return prog;
}

static cl_kernel rope_kernel(OpenCLContext& cl_ctx) {
  static cl_kernel k = nullptr;
  static bool init = false;
  if (init) return k;
  cl_program p = build_rope_program(cl_ctx);
  if (!p) return nullptr;
  cl_int err = CL_SUCCESS;
  k = clCreateKernel(p, "apply_rope_inplace", &err);
  if (err != CL_SUCCESS || !k) {
    NNOPT_ERROR_FMT("op_LlamaSdpaAttention: clCreateKernel(apply_rope_inplace) failed (%d)", (int)err);
    return nullptr;
  }
  init = true;
  return k;
}

// Combined Q+K RoPE kernel — one dispatch covers both buffers.
static cl_kernel rope_qk_kernel(OpenCLContext& cl_ctx) {
  static cl_kernel k = nullptr;
  static bool init = false;
  if (init) return k;
  cl_program p = build_rope_program(cl_ctx);
  if (!p) return nullptr;
  cl_int err = CL_SUCCESS;
  k = clCreateKernel(p, "apply_rope_qk_inplace", &err);
  if (err != CL_SUCCESS || !k) {
    NNOPT_ERROR_FMT("op_LlamaSdpaAttention: clCreateKernel(apply_rope_qk_inplace) failed (%d)", (int)err);
    return nullptr;
  }
  init = true;
  return k;
}

}  // namespace

extern "C" cl_mem op_LlamaSdpaAttention(OpenCLContext& cl_ctx,
                                         Weights& weights,
                                         cl_command_queue queue,
                                         cl_mem hidden_states, // [rows, hidden]
                                         int rows,
                                         int hidden_size,
                                         int num_q_heads,
                                         int num_kv_heads,
                                         int head_dim,
                                         int start_pos,
                                         const char* q_w,
                                         const char* k_w,
                                         const char* v_w,
                                         const char* o_w,
                                         const char* fused_in_norm_w,
                                         float rms_eps) {
  if (!ensure_initialized(cl_ctx)) return nullptr;
  if (!queue || !hidden_states) {
    NNOPT_ERROR("op_LlamaSdpaAttention: null queue/hidden_states");
    return nullptr;
  }

  const int seq_q = rows;
  const int q_dim = num_q_heads * head_dim;
  const int kv_dim = num_kv_heads * head_dim;
  if (q_dim != hidden_size) {
    NNOPT_ERROR("op_LlamaSdpaAttention: hidden_size != num_q_heads*head_dim");
    return nullptr;
  }

  cl_mem Wq = weights.get_buffer(std::string(q_w));
  cl_mem Wk = weights.get_buffer(std::string(k_w));
  cl_mem Wv = weights.get_buffer(std::string(v_w));
  cl_mem Wo = weights.get_buffer(std::string(o_w));
  if (!Wq || !Wk || !Wv || !Wo) {
    NNOPT_ERROR("op_LlamaSdpaAttention: missing weights");
    return nullptr;
  }

  cl_int err = CL_SUCCESS;
  cl_context ctx = cl_ctx.context();

  // Resolve layer_idx + ensure KV cache up front (needed by both decode and prefill paths).
  const int layer_idx = extract_layer_idx(q_w);
  if (layer_idx < 0) {
    NNOPT_ERROR_FMT("op_LlamaSdpaAttention: cannot extract layer_idx from q_w='%s'", q_w ? q_w : "(null)");
    return nullptr;
  }
  if (!ensure_kv_cache(cl_ctx, layer_idx, kv_dim)) {
    return nullptr;
  }
  if (start_pos + seq_q > KV_CACHE_MAX_LEN) {
    NNOPT_ERROR_FMT("op_LlamaSdpaAttention: KV cache overflow (start_pos=%d seq_q=%d max=%d)",
                    start_pos, seq_q, KV_CACHE_MAX_LEN);
    return nullptr;
  }

  // Note: Q + ctx_q persistence tried and regressed slightly on Adreno 620;
  // clCreateBuffer is cheap enough on this driver that per-call alloc wins.
  bool Q_is_persistent = false;
  cl_mem Q = clCreateBuffer(ctx, CL_MEM_READ_WRITE, (size_t)seq_q * (size_t)q_dim * sizeof(nnopt_storage_t), nullptr, &err);
  cl_mem K = nullptr;
  cl_mem V = nullptr;
  if (!Q) { NNOPT_ERROR("op_LlamaSdpaAttention: alloc Q failed"); return nullptr; }
  cl_mem cache_K = kv_caches()[(size_t)layer_idx].K;
  cl_mem cache_V = kv_caches()[(size_t)layer_idx].V;

  // (Opt #15 attempt — fuse kv_cache_write into QKV via sub-buffers — regressed
  // on Adreno 620: clCreateSubBuffer overhead per call was higher than the
  // single clEnqueueCopyBuffer it replaced. Reverted.)
  // (Opt #10a / #28 — fuse RoPE + kv_cache_write into QKV via
  //  `gemv_m1_qkv_fused_fp16` — re-tried 2026-05-16 on the post-fusion baseline
  //  and regressed AGAIN: decode 5.19 → 4.62 tok/s 3-run median. Same root
  //  cause as #10a: WG count halves 960→480 (each WG handles 2 output rows for
  //  a RoPE pair) and Adreno 620's single CU likes more, smaller WGs even
  //  though we save 4 dispatches. The dispatch savings are real, but the lost
  //  compute parallelism is bigger. Reverted.)
  bool kv_writes_into_cache = false;
  cl_mem v_slot = nullptr;

  if (!kv_writes_into_cache) {
    // Prefill / fallback path: allocate K, V scratch and run separate QKV → RoPE → cache write.
    K = clCreateBuffer(ctx, CL_MEM_READ_WRITE, (size_t)seq_q * (size_t)kv_dim * sizeof(nnopt_storage_t), nullptr, &err);
    V = clCreateBuffer(ctx, CL_MEM_READ_WRITE, (size_t)seq_q * (size_t)kv_dim * sizeof(nnopt_storage_t), nullptr, &err);
    if (!K || !V) {
      if (K) clReleaseMemObject(K);
      if (V) clReleaseMemObject(V);
      clReleaseMemObject(Q);
      NNOPT_ERROR("op_LlamaSdpaAttention: alloc K/V failed");
      return nullptr;
    }
    NNOPT_PROFILE_BEGIN(queue, "23a_qkv_proj");
#ifdef NNOPT_USE_FP16
    // Decode fast path: try image-backed fused QKV first (opt #34), fall back
    // to buffer-backed fused QKV (opt #6), then to 3 separate GEMVs.
    // R6.4: when caller passes fused_in_norm_w, prefer the rmsnorm+QKV fused
    // image GEMV — folds the per-layer rmsnorm dispatch into this kernel.
    bool qkv_done = false;
    if (seq_q == 1) {
      cl_mem Wq_img = get_or_create_weight_image(ctx, queue, Wq, q_dim,  hidden_size);
      cl_mem Wk_img = get_or_create_weight_image(ctx, queue, Wk, kv_dim, hidden_size);
      cl_mem Wv_img = get_or_create_weight_image(ctx, queue, Wv, kv_dim, hidden_size);
      if (Wq_img && Wk_img && Wv_img && fused_in_norm_w) {
        cl_mem gamma_in = weights.get_buffer(std::string(fused_in_norm_w));
        if (gamma_in) {
          qkv_done = gemv_m1_rmsnorm_qkv_image_fp16_dispatch(
              queue, q_dim, kv_dim, hidden_size, rms_eps,
              hidden_states, gamma_in, Wq_img, Wk_img, Wv_img, Q, K, V);
        }
      }
      if (!qkv_done && Wq_img && Wk_img && Wv_img) {
        qkv_done = gemv_m1_qkv_image_fp16_dispatch(queue, q_dim, kv_dim, hidden_size,
                                                   hidden_states, Wq_img, Wk_img, Wv_img,
                                                   Q, K, V);
      }
    }
    if (qkv_done) {
      NNOPT_PROFILE_END(queue, "23a_qkv_proj");
    } else if (seq_q == 1 && gemv_m1_qkv_fp16_dispatch(queue, q_dim, kv_dim, hidden_size,
                                                       hidden_states, Wq, Wk, Wv, Q, K, V)) {
      NNOPT_PROFILE_END(queue, "23a_qkv_proj");
    } else
#endif
    {
      if (!pytorch_linear(queue, seq_q, q_dim, hidden_size, hidden_states, Wq, Q)) {
        NNOPT_PROFILE_END(queue, "23a_qkv_proj");
        NNOPT_ERROR("op_LlamaSdpaAttention: q_proj failed");
        clReleaseMemObject(Q); clReleaseMemObject(K); clReleaseMemObject(V);
        return nullptr;
      }
      if (!pytorch_linear(queue, seq_q, kv_dim, hidden_size, hidden_states, Wk, K)) {
        NNOPT_PROFILE_END(queue, "23a_qkv_proj");
        NNOPT_ERROR("op_LlamaSdpaAttention: k_proj failed");
        clReleaseMemObject(Q); clReleaseMemObject(K); clReleaseMemObject(V);
        return nullptr;
      }
      if (!pytorch_linear(queue, seq_q, kv_dim, hidden_size, hidden_states, Wv, V)) {
        NNOPT_PROFILE_END(queue, "23a_qkv_proj");
        NNOPT_ERROR("op_LlamaSdpaAttention: v_proj failed");
        clReleaseMemObject(Q); clReleaseMemObject(K); clReleaseMemObject(V);
        return nullptr;
      }
      NNOPT_PROFILE_END(queue, "23a_qkv_proj");
    }
  }

  // Reshape to [seq, heads, head_dim] already matches our flat layout if we interpret dims.
  // Apply RoPE. K is either scratch (prefill) or a sub-buffer view into cache_K
  // (decode fast path) — in either case RoPE rotates in-place.
  {
  NNOPT_PROFILE_BEGIN(queue, "23b_rope");
  {
    cl_mem cos = nullptr;
    cl_mem sin = nullptr;
    if (!nnopt_get_rotary_tables(cl_ctx, MODEL_CONFIG::MAX_POSITION_EMBEDDINGS, head_dim, (float)MODEL_CONFIG::ROPE_THETA, cos, sin)) {
      NNOPT_ERROR("op_LlamaSdpaAttention: nnopt_get_rotary_tables failed");
      clReleaseMemObject(Q); clReleaseMemObject(K); clReleaseMemObject(V);
      return nullptr;
    }
    // Opt #28: combined Q+K RoPE — one dispatch covers both buffers
    // (gws=seq*(num_q+num_kv); branch inside the kernel on gid). Same per-
    // workitem math as the two-dispatch path, but skips one kernel-launch
    // round-trip per layer per token.
    cl_kernel kqk = rope_qk_kernel(cl_ctx);
    if (!kqk) {
      clReleaseMemObject(Q); clReleaseMemObject(K); clReleaseMemObject(V);
      return nullptr;
    }
    if (!set_arg_checked(kqk, 0, sizeof(cl_mem), &Q, "q")) return nullptr;
    if (!set_arg_checked(kqk, 1, sizeof(cl_mem), &K, "k")) return nullptr;
    if (!set_arg_checked(kqk, 2, sizeof(cl_mem), &cos, "cos")) return nullptr;
    if (!set_arg_checked(kqk, 3, sizeof(cl_mem), &sin, "sin")) return nullptr;
    if (!set_arg_checked(kqk, 4, sizeof(int), &seq_q, "seq")) return nullptr;
    if (!set_arg_checked(kqk, 5, sizeof(int), &num_q_heads, "num_q_heads")) return nullptr;
    if (!set_arg_checked(kqk, 6, sizeof(int), &num_kv_heads, "num_kv_heads")) return nullptr;
    if (!set_arg_checked(kqk, 7, sizeof(int), &head_dim, "head_dim")) return nullptr;
    if (!set_arg_checked(kqk, 8, sizeof(int), &start_pos, "start_pos")) return nullptr;
    size_t gws_qk[1] = {(size_t)seq_q * (size_t)(num_q_heads + num_kv_heads)};
    err = clEnqueueNDRangeKernel(queue, kqk, 1, nullptr, gws_qk, nullptr, 0, nullptr, nullptr);
    if (err != CL_SUCCESS) {
      NNOPT_ERROR_FMT("op_LlamaSdpaAttention: rope_qk failed (%d)", (int)err);
      clReleaseMemObject(Q); clReleaseMemObject(K); clReleaseMemObject(V);
      return nullptr;
    }
  }
  NNOPT_PROFILE_END(queue, "23b_rope");

  if (!kv_writes_into_cache) {
    NNOPT_PROFILE_BEGIN(queue, "23c_kv_cache_write");
    {
      LayerKVCache& kc = kv_caches()[(size_t)layer_idx];

      // R6.3: decode fast path — 1 fused kernel writes K + V to both buffer
      // and image at once. At decode seq_q=1 and the row is only kv_dim/4
      // pixels (48 for SmolVLM); the previous 4-dispatch chain was ~600 µs
      // of pure dispatch overhead for ~1.5 KB of data.
      bool used_fused = false;
#ifdef NNOPT_USE_FP16
      if (seq_q == 1 && kc.K_image && kc.V_image && st().k_kv_write_decode) {
        cl_kernel kw = st().k_kv_write_decode;
        if (set_arg_checked(kw, 0, sizeof(cl_mem), &K, "k_row") &&
            set_arg_checked(kw, 1, sizeof(cl_mem), &V, "v_row") &&
            set_arg_checked(kw, 2, sizeof(cl_mem), &cache_K, "cache_K") &&
            set_arg_checked(kw, 3, sizeof(cl_mem), &cache_V, "cache_V") &&
            set_arg_checked(kw, 4, sizeof(cl_mem), &kc.K_image, "K_image") &&
            set_arg_checked(kw, 5, sizeof(cl_mem), &kc.V_image, "V_image") &&
            set_arg_checked(kw, 6, sizeof(int), &kv_dim, "kv_dim") &&
            set_arg_checked(kw, 7, sizeof(int), &start_pos, "start_pos")) {
          const size_t gws[1] = {(size_t)(kv_dim / 4)};
          cl_int ke = clEnqueueNDRangeKernel(queue, kw, 1, nullptr, gws, nullptr, 0, nullptr, nullptr);
          if (ke == CL_SUCCESS) {
            used_fused = true;
          } else {
            fprintf(stderr, "[kv_image] kv_cache_write_decode_fused failed (err=%d) — falling back\n", (int)ke);
          }
        }
      }
#endif

      if (!used_fused) {
        // Prefill (seq_q > 1) or fp32 build or kernel unavailable: original
        // 2 buffer copies + 0-2 image mirrors path. The bulk-copy is fine here
        // — at seq_q ≥ ~8 the data dominates the dispatch overhead.
        const size_t bytes_per_tok = (size_t)kv_dim * sizeof(nnopt_storage_t);
        const size_t dst_off = (size_t)start_pos * bytes_per_tok;
        const size_t copy_bytes = (size_t)seq_q * bytes_per_tok;
        err = clEnqueueCopyBuffer(queue, K, cache_K, 0, dst_off, copy_bytes, 0, nullptr, nullptr);
        if (err != CL_SUCCESS) {
          NNOPT_ERROR_FMT("op_LlamaSdpaAttention: KV cache K write failed layer=%d (%d)", layer_idx, (int)err);
          clReleaseMemObject(Q); clReleaseMemObject(K); clReleaseMemObject(V);
          return nullptr;
        }
        err = clEnqueueCopyBuffer(queue, V, cache_V, 0, dst_off, copy_bytes, 0, nullptr, nullptr);
        if (err != CL_SUCCESS) {
          NNOPT_PROFILE_END(queue, "23c_kv_cache_write");
          NNOPT_ERROR_FMT("op_LlamaSdpaAttention: KV cache V write failed layer=%d (%d)", layer_idx, (int)err);
          clReleaseMemObject(Q); clReleaseMemObject(K); clReleaseMemObject(V);
          return nullptr;
        }
        if (kc.K_image) {
          const size_t origin[3] = {0, (size_t)start_pos, 0};
          const size_t region[3] = {(size_t)(kv_dim / 4), (size_t)seq_q, 1};
          cl_int e_ki = clEnqueueCopyBufferToImage(queue, cache_K, kc.K_image,
                                                    /*src_offset=*/dst_off,
                                                    origin, region, 0, nullptr, nullptr);
          if (e_ki != CL_SUCCESS) {
            fprintf(stderr, "[kv_image] CopyBufferToImage(K) failed layer=%d (err=%d)\n",
                    layer_idx, (int)e_ki);
          }
        }
        if (kc.V_image) {
          const size_t origin[3] = {0, (size_t)start_pos, 0};
          const size_t region[3] = {(size_t)(kv_dim / 4), (size_t)seq_q, 1};
          cl_int e_vi = clEnqueueCopyBufferToImage(queue, cache_V, kc.V_image,
                                                    /*src_offset=*/dst_off,
                                                    origin, region, 0, nullptr, nullptr);
          if (e_vi != CL_SUCCESS) {
            fprintf(stderr, "[kv_image] CopyBufferToImage(V) failed layer=%d (err=%d)\n",
                    layer_idx, (int)e_vi);
          }
        }
      }
    }
    NNOPT_PROFILE_END(queue, "23c_kv_cache_write");
  }
  }  // end rope block
  const int seq_k = start_pos + seq_q;
  const float scale = 1.0f / std::sqrt((float)head_dim);

  // (decode_attn_fp16_dispatch fused-attn path was tried but underutilized
  // Adreno 620's single CU at 9 workgroups; reverted. See benchmark.md note.)

  // scores: [num_q_heads, seq_q, seq_k]
  // Decode (seq_q=1): reuse a persistent per-layer scores buffer sized to
  // KV_CACHE_MAX_LEN. Eliminates ~30 clCreateBuffer/clReleaseMemObject pairs
  // per decode token. Prefill keeps the per-call alloc since its shape is
  // huge (seq_q×seq_q×heads × 2B).
  cl_mem scores = nullptr;
  bool scores_is_persistent = false;
  if (seq_q == 1) {
    const size_t need_bytes = (size_t)num_q_heads * (size_t)KV_CACHE_MAX_LEN * sizeof(nnopt_storage_t);
    LayerKVCache& c = kv_caches()[(size_t)layer_idx];
    if (!c.scores_decode || c.scores_decode_bytes < need_bytes) {
      if (c.scores_decode) clReleaseMemObject(c.scores_decode);
      c.scores_decode = clCreateBuffer(ctx, CL_MEM_READ_WRITE, need_bytes, nullptr, &err);
      if (err != CL_SUCCESS || !c.scores_decode) {
        NNOPT_ERROR_FMT("op_LlamaSdpaAttention: alloc persistent scores failed (%d)", (int)err);
        clReleaseMemObject(Q); clReleaseMemObject(K); clReleaseMemObject(V);
        return nullptr;
      }
      c.scores_decode_bytes = need_bytes;
    }
    scores = c.scores_decode;
    scores_is_persistent = true;
  } else {
    scores = clCreateBuffer(ctx, CL_MEM_READ_WRITE,
                            (size_t)num_q_heads * (size_t)seq_q * (size_t)seq_k * sizeof(nnopt_storage_t),
                            nullptr, &err);
    if (!scores) {
      NNOPT_ERROR("op_LlamaSdpaAttention: alloc scores failed");
      clReleaseMemObject(Q); clReleaseMemObject(K); clReleaseMemObject(V);
      return nullptr;
    }
  }

  // scores kernel uses k_cache layout [seq_k, num_kv_heads, head_dim]. Our K is [seq_q, num_kv_heads, head_dim]. OK for prefill-only.
  NNOPT_PROFILE_BEGIN(queue, "23d_attn_scores");
  {
    // Three paths now:
    //   prefill tiled (seq_q >= 16, head_dim==64): __local-tile scores kernel (Round 3 win)
    //   decode / fallback with K image available: image-backed scores (texture cache)
    //   decode / fallback without image: original buffer scores
    LayerKVCache& kc = kv_caches()[(size_t)layer_idx];
    const bool use_tiled = (seq_q >= 16) && (head_dim == 64) && (st().k_scores_tiled != nullptr);
    const bool use_image = !use_tiled && kc.K_image != nullptr && st().k_scores_image != nullptr;
    cl_kernel ks = use_tiled ? st().k_scores_tiled
                             : (use_image ? st().k_scores_image : st().k_scores);
    if (!set_arg_checked(ks, 0, sizeof(cl_mem), &Q, "q")) return nullptr;
    if (use_image) {
      if (!set_arg_checked(ks, 1, sizeof(cl_mem), &kc.K_image, "k_cache_img")) return nullptr;
    } else {
      if (!set_arg_checked(ks, 1, sizeof(cl_mem), &cache_K, "k_cache")) return nullptr;
    }
    if (!set_arg_checked(ks, 2, sizeof(cl_mem), &scores, "scores")) return nullptr;
    if (!set_arg_checked(ks, 3, sizeof(int), &seq_q, "seq_q")) return nullptr;
    if (!set_arg_checked(ks, 4, sizeof(int), &seq_k, "seq_k")) return nullptr;
    if (!set_arg_checked(ks, 5, sizeof(int), &num_q_heads, "num_q_heads")) return nullptr;
    if (!set_arg_checked(ks, 6, sizeof(int), &num_kv_heads, "num_kv_heads")) return nullptr;
    if (!set_arg_checked(ks, 7, sizeof(int), &head_dim, "head_dim")) return nullptr;
    if (!set_arg_checked(ks, 8, sizeof(float), &scale, "scale")) return nullptr;

    if (use_tiled) {
      const int TILED_BR = 16, TILED_BC = 16;
      const size_t lws[3] = {1, (size_t)TILED_BR, (size_t)TILED_BC};
      const size_t q_blocks = (size_t)((seq_q + TILED_BR - 1) / TILED_BR);
      const size_t k_blocks = (size_t)((seq_k + TILED_BC - 1) / TILED_BC);
      const size_t gws[3] = {(size_t)num_q_heads, q_blocks * lws[1], k_blocks * lws[2]};
      err = clEnqueueNDRangeKernel(queue, ks, 3, nullptr, gws, lws, 0, nullptr, nullptr);
    } else {
      const size_t gws[3] = {(size_t)num_q_heads, (size_t)seq_q, (size_t)seq_k};
      err = clEnqueueNDRangeKernel(queue, ks, 3, nullptr, gws, nullptr, 0, nullptr, nullptr);
    }
    if (err != CL_SUCCESS) {
      NNOPT_PROFILE_END(queue, "23d_attn_scores");
      NNOPT_ERROR_FMT("op_LlamaSdpaAttention: gqa_attn_scores failed (%d)", (int)err);
      if (!scores_is_persistent) clReleaseMemObject(scores);
      clReleaseMemObject(Q); clReleaseMemObject(K); clReleaseMemObject(V);
      return nullptr;
    }
  }
  NNOPT_PROFILE_END(queue, "23d_attn_scores");

  // softmax
  NNOPT_PROFILE_BEGIN(queue, "23e_attn_softmax");
  {
    cl_kernel ksm = st().k_softmax;
    const int total_rows = num_q_heads * seq_q;
    if (!set_arg_checked(ksm, 0, sizeof(cl_mem), &scores, "scores")) return nullptr;
    if (!set_arg_checked(ksm, 1, sizeof(int), &seq_q, "seq_q")) return nullptr;
    if (!set_arg_checked(ksm, 2, sizeof(int), &seq_k, "seq_k")) return nullptr;
    if (!set_arg_checked(ksm, 3, sizeof(int), &total_rows, "total_rows")) return nullptr;
    // Workgroup-per-row, 64 lanes parallel-reduce. Matches reqd_work_group_size in attention.cl.
    const size_t lws[1] = {64};
    const size_t gws[1] = {(size_t)total_rows * lws[0]};
    err = clEnqueueNDRangeKernel(queue, ksm, 1, nullptr, gws, lws, 0, nullptr, nullptr);
    if (err != CL_SUCCESS) {
      NNOPT_PROFILE_END(queue, "23e_attn_softmax");
      NNOPT_ERROR_FMT("op_LlamaSdpaAttention: gqa_softmax failed (%d)", (int)err);
      if (!scores_is_persistent) clReleaseMemObject(scores);
      clReleaseMemObject(Q); clReleaseMemObject(K); clReleaseMemObject(V);
      return nullptr;
    }
  }
  NNOPT_PROFILE_END(queue, "23e_attn_softmax");

  // attn_out: [seq_q, num_q_heads, head_dim]
  bool ctx_q_is_persistent = false;
  cl_mem ctx_q = clCreateBuffer(ctx, CL_MEM_READ_WRITE,
                                (size_t)seq_q * (size_t)q_dim * sizeof(nnopt_storage_t),
                                nullptr, &err);
  if (!ctx_q) {
    NNOPT_ERROR("op_LlamaSdpaAttention: alloc ctx_q failed");
    if (!scores_is_persistent) clReleaseMemObject(scores);
    clReleaseMemObject(Q); clReleaseMemObject(K); clReleaseMemObject(V);
    return nullptr;
  }

  NNOPT_PROFILE_BEGIN(queue, "23f_attn_out");
  if (seq_q == 1) {
    // Decode fast path: workgroup-per-(qh, d4), 64-lane reduce over seq_k.
    // R6.1: when the V_image mirror is populated (always, post-#35a), read V
    // via L1 texture cache through `read_imageh` — same trick as prefill's
    // gqa_attn_out_image. At short-prompt seq_k, attn_out is ~470 µs/layer
    // and L2-buffer reads are the main remaining cost.
    LayerKVCache& kc_o = kv_caches()[(size_t)layer_idx];
    const bool use_image = (kc_o.V_image != nullptr) && (st().k_out_decode_image != nullptr);
    cl_kernel ko = use_image ? st().k_out_decode_image : st().k_out_decode;
    if (!set_arg_checked(ko, 0, sizeof(cl_mem), &scores, "scores")) return nullptr;
    if (use_image) {
      if (!set_arg_checked(ko, 1, sizeof(cl_mem), &kc_o.V_image, "v_cache_img")) return nullptr;
    } else {
      if (!set_arg_checked(ko, 1, sizeof(cl_mem), &cache_V, "v_cache")) return nullptr;
    }
    if (!set_arg_checked(ko, 2, sizeof(cl_mem), &ctx_q, "out")) return nullptr;
    if (!set_arg_checked(ko, 3, sizeof(int), &seq_k, "seq_k")) return nullptr;
    if (!set_arg_checked(ko, 4, sizeof(int), &num_q_heads, "num_q_heads")) return nullptr;
    if (!set_arg_checked(ko, 5, sizeof(int), &num_kv_heads, "num_kv_heads")) return nullptr;
    if (!set_arg_checked(ko, 6, sizeof(int), &head_dim, "head_dim")) return nullptr;
    const size_t lws[2] = {1, 64};
    const size_t gws[2] = {(size_t)num_q_heads, (size_t)(head_dim / 4) * lws[1]};
    err = clEnqueueNDRangeKernel(queue, ko, 2, nullptr, gws, lws, 0, nullptr, nullptr);
    if (err != CL_SUCCESS) {
      NNOPT_PROFILE_END(queue, "23f_attn_out");
      NNOPT_ERROR_FMT("op_LlamaSdpaAttention: gqa_attn_out_decode%s failed (%d)",
                      use_image ? "_image" : "", (int)err);
      clReleaseMemObject(ctx_q);
      if (!scores_is_persistent) clReleaseMemObject(scores);
      clReleaseMemObject(Q); clReleaseMemObject(K); clReleaseMemObject(V);
      return nullptr;
    }
  } else {
    // Prefill attn_out: route V reads through L1 texture cache via image2D
    // when available. Was 6 SECONDS (35 % of TTFT) with buffer reads because
    // V (336 KB / layer) blows L2. The image-backed kernel reads via
    // `read_imageh` through the dedicated texture cache — same trick that gave
    // +94 % on decode GEMVs.
    LayerKVCache& kc = kv_caches()[(size_t)layer_idx];
    const bool use_image = (kc.V_image != nullptr) && (st().k_out_image != nullptr);
    cl_kernel ko = use_image ? st().k_out_image : st().k_out;
    if (!set_arg_checked(ko, 0, sizeof(cl_mem), &scores, "scores")) return nullptr;
    if (use_image) {
      if (!set_arg_checked(ko, 1, sizeof(cl_mem), &kc.V_image, "v_cache_img")) return nullptr;
    } else {
      if (!set_arg_checked(ko, 1, sizeof(cl_mem), &cache_V, "v_cache")) return nullptr;
    }
    if (!set_arg_checked(ko, 2, sizeof(cl_mem), &ctx_q, "out")) return nullptr;
    if (!set_arg_checked(ko, 3, sizeof(int), &seq_q, "seq_q")) return nullptr;
    if (!set_arg_checked(ko, 4, sizeof(int), &seq_k, "seq_k")) return nullptr;
    if (!set_arg_checked(ko, 5, sizeof(int), &num_q_heads, "num_q_heads")) return nullptr;
    if (!set_arg_checked(ko, 6, sizeof(int), &num_kv_heads, "num_kv_heads")) return nullptr;
    if (!set_arg_checked(ko, 7, sizeof(int), &head_dim, "head_dim")) return nullptr;

    const size_t gws[3] = {(size_t)num_q_heads, (size_t)seq_q, (size_t)(head_dim / 4)};
    err = clEnqueueNDRangeKernel(queue, ko, 3, nullptr, gws, nullptr, 0, nullptr, nullptr);
    if (err != CL_SUCCESS) {
      NNOPT_PROFILE_END(queue, "23f_attn_out");
      NNOPT_ERROR_FMT("op_LlamaSdpaAttention: gqa_attn_out failed (%d)", (int)err);
      clReleaseMemObject(ctx_q);
      if (!scores_is_persistent) clReleaseMemObject(scores);
      clReleaseMemObject(Q); clReleaseMemObject(K); clReleaseMemObject(V);
      return nullptr;
    }
  }
  NNOPT_PROFILE_END(queue, "23f_attn_out");

  // reshape ctx_q [seq, heads, head_dim] -> [seq, hidden]
  // Layout is already contiguous with head_dim fastest, then head, then seq. That matches [seq, hidden] row-major.
  // So we can directly feed ctx_q into o_proj as [seq, hidden].
  cl_mem out = clCreateBuffer(ctx, CL_MEM_READ_WRITE,
                             (size_t)seq_q * (size_t)hidden_size * sizeof(nnopt_storage_t),
                             nullptr, &err);
  if (!out) {
    NNOPT_ERROR("op_LlamaSdpaAttention: alloc out failed");
    clReleaseMemObject(ctx_q);
    if (!scores_is_persistent) clReleaseMemObject(scores);
    clReleaseMemObject(Q); clReleaseMemObject(K); clReleaseMemObject(V);
    return nullptr;
  }

  NNOPT_PROFILE_BEGIN(queue, "23g_o_proj");
#ifdef NNOPT_USE_FP16
  // Decode fast path: route o_proj GEMV through the L1 texture cache via
  // image2D-backed weight (Round-5 opt #33). One-time clEnqueueCopyBufferToImage
  // amortizes across all decode tokens. Falls back to pytorch_linear on first-
  // call image-creation failure or for prefill (rows>1, image GEMV is M=1-only).
  bool o_proj_ok = false;
  if (seq_q == 1) {
    cl_mem Wo_img = get_or_create_weight_image(cl_ctx.context(), queue,
                                                Wo, hidden_size, hidden_size);
    if (Wo_img) {
      o_proj_ok = gemv_m1_image_fp16_dispatch(queue, hidden_size, hidden_size,
                                              ctx_q, Wo_img, out);
    }
  }
  if (!o_proj_ok) {
    if (!pytorch_linear(queue, seq_q, hidden_size, hidden_size, ctx_q, Wo, out)) {
      NNOPT_ERROR("op_LlamaSdpaAttention: o_proj failed");
      clReleaseMemObject(out);
      out = nullptr;
    }
  }
#else
  if (!pytorch_linear(queue, seq_q, hidden_size, hidden_size, ctx_q, Wo, out)) {
    NNOPT_ERROR("op_LlamaSdpaAttention: o_proj failed");
    clReleaseMemObject(out);
    out = nullptr;
  }
#endif
  NNOPT_PROFILE_END(queue, "23g_o_proj");

  if (!ctx_q_is_persistent) clReleaseMemObject(ctx_q);
  if (!scores_is_persistent) clReleaseMemObject(scores);
  if (!Q_is_persistent) clReleaseMemObject(Q);
  if (K) clReleaseMemObject(K);
  if (V) clReleaseMemObject(V);
  if (v_slot) clReleaseMemObject(v_slot);

  return out;
}
