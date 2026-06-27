// Gemma3Attention — GQA self-attention with QK-norm + dual-theta RoPE.
//
// Reference: model_info/transformers_src/modeling_gemma3.py:99-133 Gemma3Attention.forward
//   q = q_proj(x).view(*, heads, head_dim); k/v = k_proj/v_proj(x).view(*, kv, head_dim)
//   q = q_norm(q); k = k_norm(k)               (RMSNorm over head_dim, BEFORE RoPE)
//   q,k = apply_rotary_pos_emb(q, k, cos, sin) (local theta on sliding layers,
//                                               global theta on full layers)
//   attn = eager_attention_forward(q, k, v, mask, scaling=query_pre_attn_scalar**-0.5)
//        = softmax(Q@K^T * scaling + causal[+sliding]) @ V   (GQA repeat_kv)
//   out = o_proj(attn.reshape(*, heads*head_dim))
//
// PRECISION PATH: the PyTorch reference runs the WHOLE attention in fp32
// (torch default dtype, torch_dtype=None). q_norm/k_norm, the fp32 RoPE
// tables, RoPE itself, QK^T dot, softmax, and the V-weighted accumulation
// are ALL fp32. Storing any of these as fp16 loses ~3 decimal digits and
// accumulates per-layer drift that flips the argmax on near-tie logits
// (15th vs 10th). So:
//   q/k:  gemma3_rmsnorm_f32out (fp32) -> gemma3_rope_f32 (fp32 tables) -> fp32 dot
//   v:    gemma3_v_to_f32 (fp32) -> fp32 weighted-sum
//   cos/sin: nnopt_get_rotary_tables_f32 (raw fp32, no fp16 roundtrip)
// Only the final attention output is narrowed to storage_t for o_proj.
//
// Stateless full-sequence forward (generate() re-runs the whole sequence each
// step, start_pos==0), so no KV cache is used.

#include "../opencl_context.h"
#include "../weights.h"
#include "../debug_utils.h"
#include "../model_config.h"
#include "../utils.h"
#include "../rotary_tables.h"
#include "../profiler.h"
#include <cmath>
#include <string>

namespace {
cl_program g_prog = nullptr;
cl_kernel  g_rmsnorm_f32out = nullptr;
cl_kernel  g_rope_f32 = nullptr;
cl_kernel  g_attn_f32qk = nullptr;
cl_kernel  g_v_to_f32 = nullptr;
bool ensure_kernels(OpenCLContext& cl_ctx) {
    if (g_rmsnorm_f32out && g_rope_f32 && g_attn_f32qk && g_v_to_f32) return true;
    g_prog = cl_ctx.build_program_from_file("kernels/gemma3_ops.cl");  // PROGRAM-INIT-OK
    if (!g_prog) { NNOPT_ERROR("Attn: build gemma3_ops.cl failed"); return false; }
    cl_int err = CL_SUCCESS;
    g_rmsnorm_f32out = clCreateKernel(g_prog, "gemma3_rmsnorm_f32out", &err);
    if (err != CL_SUCCESS || !g_rmsnorm_f32out) { NNOPT_ERROR_FMT("Attn: kernel gemma3_rmsnorm_f32out %d", err); return false; }
    g_rope_f32 = clCreateKernel(g_prog, "gemma3_rope_f32", &err);
    if (err != CL_SUCCESS || !g_rope_f32) { NNOPT_ERROR_FMT("Attn: kernel gemma3_rope_f32 %d", err); return false; }
    g_attn_f32qk = clCreateKernel(g_prog, "gemma3_gqa_attn_f32qk", &err);
    if (err != CL_SUCCESS || !g_attn_f32qk) { NNOPT_ERROR_FMT("Attn: kernel gemma3_gqa_attn_f32qk %d", err); return false; }
    g_v_to_f32 = clCreateKernel(g_prog, "gemma3_v_to_f32", &err);
    if (err != CL_SUCCESS || !g_v_to_f32) { NNOPT_ERROR_FMT("Attn: kernel gemma3_v_to_f32 %d", err); return false; }
    return true;
}

// RMSNorm over head_dim, fp16 input → fp32 output buffer (caller owns).
cl_mem rmsnorm_f32out(OpenCLContext& cl_ctx, cl_command_queue queue,
                      cl_mem input, int rows, int cols, float eps, cl_mem weight_buf) {
    cl_int err = CL_SUCCESS;
    cl_mem out = nnopt_pool_alloc(cl_ctx.context(), (size_t)rows * cols * sizeof(float), &err);
    if (err != CL_SUCCESS || !out) { NNOPT_ERROR_FMT("Attn: alloc rmsnorm_f32out %d", err); return nullptr; }
    auto fail = [&]() -> cl_mem { if (out) { nnopt_pool_free(out); out = nullptr; } return nullptr; };
    if (!set_arg_checked(g_rmsnorm_f32out, 0, sizeof(cl_mem), &input, "x")) return fail();
    if (!set_arg_checked(g_rmsnorm_f32out, 1, sizeof(cl_mem), &weight_buf, "weight")) return fail();
    if (!set_arg_checked(g_rmsnorm_f32out, 2, sizeof(cl_mem), &out, "out")) return fail();
    if (!set_arg_checked(g_rmsnorm_f32out, 3, sizeof(int), &cols, "cols")) return fail();
    if (!set_arg_checked(g_rmsnorm_f32out, 4, sizeof(float), &eps, "eps")) return fail();
    size_t lws = 64;  // must match RMS_WG in gemma3_ops.cl
    size_t gws = (size_t)rows * lws;
    err = clEnqueueNDRangeKernel(queue, g_rmsnorm_f32out, 1, nullptr, &gws, &lws,
                                 0, nullptr, KernelProfiler::event_for("op_gemma3_rmsnorm_f32out"));
    if (err != CL_SUCCESS) { NNOPT_ERROR_FMT("Attn: rmsnorm_f32out dispatch %d", err); return fail(); }
    return out;
}

// Convert fp16 v projection → raw fp32 buffer (caller owns).
cl_mem v_to_f32(OpenCLContext& cl_ctx, cl_command_queue queue, cl_mem v, int n) {
    cl_int err = CL_SUCCESS;
    cl_mem out = nnopt_pool_alloc(cl_ctx.context(), (size_t)n * sizeof(float), &err);
    if (err != CL_SUCCESS || !out) { NNOPT_ERROR_FMT("Attn: alloc v_f32 %d", err); return nullptr; }
    auto fail = [&]() -> cl_mem { if (out) { nnopt_pool_free(out); out = nullptr; } return nullptr; };
    if (!set_arg_checked(g_v_to_f32, 0, sizeof(cl_mem), &v, "v")) return fail();
    if (!set_arg_checked(g_v_to_f32, 1, sizeof(cl_mem), &out, "out")) return fail();
    if (!set_arg_checked(g_v_to_f32, 2, sizeof(int), &n, "n")) return fail();
    size_t gws = (size_t)n;
    err = clEnqueueNDRangeKernel(queue, g_v_to_f32, 1, nullptr, &gws, nullptr,
                                 0, nullptr, KernelProfiler::event_for("op_gemma3_v_to_f32"));
    if (err != CL_SUCCESS) { NNOPT_ERROR_FMT("Attn: v_to_f32 dispatch %d", err); return fail(); }
    return out;
}

// Apply RoPE in-place to an fp32 buf [seq, heads, head_dim]. cos/sin are fp32.
bool apply_rope_f32(cl_command_queue queue, cl_mem buf, cl_mem cos_t, cl_mem sin_t,
                    int seq_len, int heads, int head_dim, int start_pos) {
    if (!set_arg_checked(g_rope_f32, 0, sizeof(cl_mem), &buf,   "buf"))    return false;
    if (!set_arg_checked(g_rope_f32, 1, sizeof(cl_mem), &cos_t, "cos"))    return false;
    if (!set_arg_checked(g_rope_f32, 2, sizeof(cl_mem), &sin_t, "sin"))    return false;
    if (!set_arg_checked(g_rope_f32, 3, sizeof(int), &seq_len, "seq_len")) return false;
    if (!set_arg_checked(g_rope_f32, 4, sizeof(int), &heads,   "heads"))   return false;
    if (!set_arg_checked(g_rope_f32, 5, sizeof(int), &head_dim,"head_dim"))return false;
    if (!set_arg_checked(g_rope_f32, 6, sizeof(int), &start_pos,"start_pos")) return false;
    size_t gws = (size_t)seq_len * heads * head_dim;
    cl_int err = clEnqueueNDRangeKernel(queue, g_rope_f32, 1, nullptr, &gws, nullptr,
                                        0, nullptr, KernelProfiler::event_for("op_gemma3_rope_f32"));
    if (err != CL_SUCCESS) { NNOPT_ERROR_FMT("Attn: rope_f32 dispatch %d", err); return false; }
    return true;
}
} // namespace

extern "C" {
cl_mem Gemma3Attention_forward(
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
    (void)k_cache_inout; (void)v_cache_inout; (void)encoder_hidden_states;
    KernelProfiler::HostTimer _ht_attn("host_attn_total");
    if (!ensure_kernels(cl_ctx)) return nullptr;
    const std::string wp = weight_prefix ? std::string(weight_prefix) : std::string();

    cl_mem q_w = weights.get_buffer(wp + ".q_proj.weight");
    cl_mem k_w = weights.get_buffer(wp + ".k_proj.weight");
    cl_mem v_w = weights.get_buffer(wp + ".v_proj.weight");
    cl_mem o_w = weights.get_buffer(wp + ".o_proj.weight");
    cl_mem q_norm_w = weights.get_buffer(wp + ".q_norm.weight");
    cl_mem k_norm_w = weights.get_buffer(wp + ".k_norm.weight");
    if (!q_w || !k_w || !v_w || !o_w || !q_norm_w || !k_norm_w) {
        NNOPT_ERROR_FMT("Attn: missing weights for %s", wp.c_str());
        return nullptr;
    }

    const int M  = seq_len;
    const int H  = MODEL_CONFIG::HIDDEN_SIZE;          // 640
    const int HD = MODEL_CONFIG::HEAD_DIM;             // 256
    const int QH = MODEL_CONFIG::NUM_ATTENTION_HEADS;  // 4
    const int KVH= MODEL_CONFIG::NUM_KEY_VALUE_HEADS;  // 1
    const int q_dim  = QH * HD;                         // 1024
    const int kv_dim = KVH * HD;                        // 256
    const float scaling = 1.0f / std::sqrt((float)MODEL_CONFIG::QUERY_PRE_ATTN_SCALAR);  // 256**-0.5
    const float eps = MODEL_CONFIG::RMS_NORM_EPS;

    // Layer type → which RoPE theta + sliding window.
    bool is_sliding = (layer_idx % MODEL_CONFIG::_SLIDING_WINDOW_PATTERN) != (MODEL_CONFIG::_SLIDING_WINDOW_PATTERN - 1);
    float theta = is_sliding ? (float)MODEL_CONFIG::ROPE_LOCAL_BASE_FREQ
                             : (float)MODEL_CONFIG::ROPE_THETA;
    int sliding_window = is_sliding ? MODEL_CONFIG::SLIDING_WINDOW : 0;

    cl_int err = CL_SUCCESS;
    cl_mem q = nullptr, k = nullptr, v = nullptr;
    cl_mem q_n = nullptr, k_n = nullptr;   // fp32 buffers (precision path)
    cl_mem v_f = nullptr;                  // fp32 v (precision path)
    cl_mem attn = nullptr, out = nullptr;
    auto cleanup = [&]() -> cl_mem {
        if (q)   { nnopt_pool_free(q);   q   = nullptr; }
        if (k)   { nnopt_pool_free(k);   k   = nullptr; }
        if (v)   { nnopt_pool_free(v);   v   = nullptr; }
        if (q_n) { nnopt_pool_free(q_n); q_n = nullptr; }
        if (k_n) { nnopt_pool_free(k_n); k_n = nullptr; }
        if (v_f) { nnopt_pool_free(v_f); v_f = nullptr; }
        if (attn){ nnopt_pool_free(attn);attn= nullptr; }
        if (out) { nnopt_pool_free(out); out = nullptr; }
        return nullptr;
    };

    // q/k/v projections. Layout [M, q_dim] is naturally [M, QH, HD] row-major.
    // All three read the same x → fused into ONE GEMV over [q_dim+2*kv_dim, H].
    // Removes two dispatches per layer; decode is dispatch-bound here.
    q = nnopt_pool_alloc(cl_ctx.context(), (size_t)M * q_dim * sizeof(nnopt_storage_t), &err);
    if (err != CL_SUCCESS || !q) { NNOPT_ERROR_FMT("Attn: alloc q %d", err); return cleanup(); }
    k = nnopt_pool_alloc(cl_ctx.context(), (size_t)M * kv_dim * sizeof(nnopt_storage_t), &err);
    if (err != CL_SUCCESS || !k) { NNOPT_ERROR_FMT("Attn: alloc k %d", err); return cleanup(); }
    v = nnopt_pool_alloc(cl_ctx.context(), (size_t)M * kv_dim * sizeof(nnopt_storage_t), &err);
    if (err != CL_SUCCESS || !v) { NNOPT_ERROR_FMT("Attn: alloc v %d", err); return cleanup(); }
    if (!pytorch_linear_fused3(queue, M, q_dim, kv_dim, kv_dim, H,
                               input, q_w, k_w, v_w, q, k, v)) {
        NNOPT_ERROR("Attn: q/k/v proj failed"); return cleanup();
    }

    // v -> fp32 (precision path: weighted-sum done in fp32).
    v_f = v_to_f32(cl_ctx, queue, v, M * kv_dim);
    if (!v_f) { NNOPT_ERROR("Attn: v_to_f32 failed"); return cleanup(); }
    nnopt_pool_free(v); v = nullptr;

    // q_norm / k_norm: RMSNorm over head_dim → fp32 output (precision path).
    q_n = rmsnorm_f32out(cl_ctx, queue, q, M * QH, HD, eps, q_norm_w);
    if (!q_n) { NNOPT_ERROR("Attn: q_norm failed"); return cleanup(); }
    k_n = rmsnorm_f32out(cl_ctx, queue, k, M * KVH, HD, eps, k_norm_w);
    if (!k_n) { NNOPT_ERROR("Attn: k_norm failed"); return cleanup(); }
    nnopt_pool_free(q); q = nullptr;
    nnopt_pool_free(k); k = nullptr;

    // fp32 RoPE tables for this layer's theta (raw float, no fp16 roundtrip).
    cl_mem cos_t = nullptr, sin_t = nullptr;
    if (!nnopt_get_rotary_tables_f32(cl_ctx, MODEL_CONFIG::MAX_POSITION_EMBEDDINGS, HD, theta, cos_t, sin_t)) {
        NNOPT_ERROR("Attn: rotary tables unavailable");
        return cleanup();
    }
    // Apply RoPE in-place (fp32) to q_n [M, QH, HD] and k_n [M, KVH, HD].
    if (!apply_rope_f32(queue, q_n, cos_t, sin_t, M, QH, HD, start_pos)) return cleanup();
    if (!apply_rope_f32(queue, k_n, cos_t, sin_t, M, KVH, HD, start_pos)) return cleanup();

    // ── KV cache (incremental decode) ──────────────────────────────────
    // When the backbone supplies a per-layer cache buffer, the K/V for the M
    // input tokens (fp32, post-norm + post-RoPE) are appended to the cache at
    // absolute positions [start_pos .. start_pos+M-1], and attention runs the
    // query rows against ALL cached positions (0..start_pos+M-1). This makes
    // decode M=1 (vs reprocessing the whole sequence each step), which both
    // shrinks GPU work and STABILISES the GEMM problem size so CLBlast stops
    // re-JITing a kernel every step. Cached values are the SAME fp32 numbers a
    // full reprocess would recompute, so per-layer cosine is unchanged.
    cl_mem k_src = k_n, v_src = v_f;
    int seq_q = M, seq_k = M;
    const bool use_cache = (k_cache_inout && v_cache_inout && *k_cache_inout && *v_cache_inout);
    if (use_cache) {
        const size_t off = (size_t)start_pos * (size_t)kv_dim * sizeof(float);
        const size_t cpy = (size_t)M * (size_t)kv_dim * sizeof(float);
        err = clEnqueueCopyBuffer(queue, k_n, *k_cache_inout, 0, off, cpy, 0, nullptr, nullptr);
        if (err != CL_SUCCESS) { NNOPT_ERROR_FMT("Attn: k-cache write %d", err); return cleanup(); }
        err = clEnqueueCopyBuffer(queue, v_f, *v_cache_inout, 0, off, cpy, 0, nullptr, nullptr);
        if (err != CL_SUCCESS) { NNOPT_ERROR_FMT("Attn: v-cache write %d", err); return cleanup(); }
        k_src = *k_cache_inout;
        v_src = *v_cache_inout;
        seq_k = start_pos + M;   // attend over all cached positions
    }

    // GQA attention (fully fp32 q/k/v) → attn [M, QH, HD] (fp16 output).
    attn = nnopt_pool_alloc(cl_ctx.context(), (size_t)M * q_dim * sizeof(nnopt_storage_t), &err);
    if (err != CL_SUCCESS || !attn) { NNOPT_ERROR_FMT("Attn: alloc attn %d", err); return cleanup(); }
    {
        if (!set_arg_checked(g_attn_f32qk, 0, sizeof(cl_mem), &q_n,  "q"))   return cleanup();
        if (!set_arg_checked(g_attn_f32qk, 1, sizeof(cl_mem), &k_src,"k"))   return cleanup();
        if (!set_arg_checked(g_attn_f32qk, 2, sizeof(cl_mem), &v_src,"v"))   return cleanup();
        if (!set_arg_checked(g_attn_f32qk, 3, sizeof(cl_mem), &attn, "out")) return cleanup();
        if (!set_arg_checked(g_attn_f32qk, 4, sizeof(int), &seq_q, "seq_q")) return cleanup();
        if (!set_arg_checked(g_attn_f32qk, 5, sizeof(int), &seq_k, "seq_k")) return cleanup();
        if (!set_arg_checked(g_attn_f32qk, 6, sizeof(int), &QH, "num_q_heads")) return cleanup();
        if (!set_arg_checked(g_attn_f32qk, 7, sizeof(int), &KVH,"num_kv_heads")) return cleanup();
        if (!set_arg_checked(g_attn_f32qk, 8, sizeof(int), &HD, "head_dim")) return cleanup();
        if (!set_arg_checked(g_attn_f32qk, 9, sizeof(float), &scaling, "scaling")) return cleanup();
        if (!set_arg_checked(g_attn_f32qk,10, sizeof(int), &start_pos, "start_pos")) return cleanup();
        if (!set_arg_checked(g_attn_f32qk,11, sizeof(int), &sliding_window, "sliding_window")) return cleanup();
        size_t lws = 64;  // must match ATTN_WG in gemma3_ops.cl
        size_t gws = (size_t)seq_q * QH * lws;  // one workgroup per (query,head)
        err = clEnqueueNDRangeKernel(queue, g_attn_f32qk, 1, nullptr, &gws, &lws,
                                     0, nullptr, KernelProfiler::event_for("op_gemma3_gqa_attn_f32qk"));
        if (err != CL_SUCCESS) { NNOPT_ERROR_FMT("Attn: attn dispatch %d", err); return cleanup(); }
    }
    nnopt_pool_free(q_n); q_n = nullptr;
    nnopt_pool_free(k_n); k_n = nullptr;
    nnopt_pool_free(v_f); v_f = nullptr;

    // o_proj: attn [M, q_dim] -> out [M, H].
    out = nnopt_pool_alloc(cl_ctx.context(), (size_t)M * H * sizeof(nnopt_storage_t), &err);
    if (err != CL_SUCCESS || !out) { NNOPT_ERROR_FMT("Attn: alloc out %d", err); return cleanup(); }
    if (!pytorch_linear(queue, M, H, q_dim, attn, o_w, out)) { NNOPT_ERROR("Attn: o_proj failed"); return cleanup(); }

    nnopt_pool_free(attn); attn = nullptr;
    return out;
}
}
