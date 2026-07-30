// Gated DeltaNet recurrence — the one operator this port cannot borrow.
//
// Factored out of reference_model.h so that (a) the unit test can drive it
// directly against torch's `torch_recurrent_gated_delta_rule`, and (b)
// kernels/delta_net.cl can mirror it line for line.
//
// Per VALUE head, the state S is a [d_k x d_v] fp32 MATRIX (not a vector of
// independent scalars — which is exactly why mamba2-130m's selective_scan.cl
// does not port: it has no counterpart for the `S^T k` reduction below).
//
// Reference (transformers qwen3_next/modeling_qwen3_next.py):
//     q, k = l2norm(q), l2norm(k)          # eps 1e-6
//     q   *= 1/sqrt(d_k)
//     S   *= exp(g)
//     kv_mem = (S * k[:,None]).sum(dim=-2)
//     delta  = (v - kv_mem) * beta
//     S     += k[:,None] * delta[None,:]
//     out    = (S * q[:,None]).sum(dim=-2)
//
// g and beta arrive already reduced to scalars per value head:
//     g    = a_neg * softplus(a + dt_bias)
//     beta = sigmoid(b)
//
// ⚠ `a_neg` is the GGUF `ssm_a` tensor, which ALREADY HOLDS -exp(A_log) — the
// llama.cpp converter pre-computes it. Do NOT apply exp() again here. Confirmed
// against llama.cpp src/models/qwen35.cpp:
//     gate = ggml_mul(ctx0, alpha_softplus, layer.ssm_a);  // -A_log.exp() * softplus
// Applying -exp() a second time makes the decay wrong by a per-layer factor
// (e.g. blk.0 ssm_a = -0.2629: correct g scale 0.2629, doubly-exp'd 0.7688),
// which shortens the state's memory and yields fluent but context-free output.
#pragma once
#include <cmath>
#include <cstddef>
#include <vector>

namespace gdn {

inline float silu(float v)     { return v / (1.0f + expf(-v)); }
inline float sigmoidf(float v) { return 1.0f / (1.0f + expf(-v)); }
// softplus with the same large-input guard torch uses (threshold 20)
inline float softplus(float v) { return v > 20.f ? v : log1pf(expf(v)); }

inline void l2norm(const float* src, int n, float* dst, float eps = 1e-6f) {
    double ss = 0;
    for (int i = 0; i < n; ++i) ss += (double)src[i] * src[i];
    const float inv = 1.0f / sqrtf((float)ss + eps);
    for (int i = 0; i < n; ++i) dst[i] = src[i] * inv;
}

// One decode step for ONE value head.
//   S    : [d_k * d_v] fp32, updated in place
//   q_in : [d_k] raw (pre-l2norm) query for this head's KEY head
//   k_in : [d_k] raw (pre-l2norm) key   for this head's KEY head
//   v_in : [d_v] value for this VALUE head
//   out  : [d_v] written (not accumulated)
//   scratch: caller-owned, >= 2*d_k + d_v floats (avoids per-call allocation)
inline void step_head(float* S, const float* q_in, const float* k_in,
                      const float* v_in, float g, float beta,
                      int d_k, int d_v, float* out, float* scratch) {
    float* qn = scratch;              // [d_k]
    float* kn = scratch + d_k;        // [d_k]
    float* tmp = scratch + 2 * d_k;   // [d_v]  (kv_mem, then delta)

    l2norm(q_in, d_k, qn);
    l2norm(k_in, d_k, kn);
    const float scale = 1.0f / sqrtf((float)d_k);
    for (int i = 0; i < d_k; ++i) qn[i] *= scale;

    const float decay = expf(g);
    for (int i = 0; i < d_k * d_v; ++i) S[i] *= decay;

    // kv_mem[j] = sum_i S[i][j] * k[i]
    for (int j = 0; j < d_v; ++j) tmp[j] = 0.f;
    for (int i = 0; i < d_k; ++i) {
        const float ki = kn[i];
        const float* Si = S + (size_t)i * d_v;
        for (int j = 0; j < d_v; ++j) tmp[j] += Si[j] * ki;
    }
    // delta = (v - kv_mem) * beta
    for (int j = 0; j < d_v; ++j) tmp[j] = (v_in[j] - tmp[j]) * beta;
    // S += outer(k, delta)
    for (int i = 0; i < d_k; ++i) {
        const float ki = kn[i];
        float* Si = S + (size_t)i * d_v;
        for (int j = 0; j < d_v; ++j) Si[j] += ki * tmp[j];
    }
    // out[j] = sum_i S[i][j] * q[i]
    for (int j = 0; j < d_v; ++j) out[j] = 0.f;
    for (int i = 0; i < d_k; ++i) {
        const float qi = qn[i];
        const float* Si = S + (size_t)i * d_v;
        for (int j = 0; j < d_v; ++j) out[j] += Si[j] * qi;
    }
}

// One decode step for ALL value heads.
//   q_all/k_all : [n_k_heads * d_k]   — NOT n_v_heads. See the head-mapping
//                                        note below.
//   v_all       : [n_v_heads * d_v]
//   a, b        : [n_v_heads] raw projections
//   a_neg       : [n_v_heads] GGUF `ssm_a` — ALREADY -exp(A_log)
//   dt_b        : [n_v_heads] GGUF `ssm_dt.bias`
//   S_all       : [n_v_heads * d_k * d_v]
//   out_all     : [n_v_heads * d_v]
inline void step(float* S_all, const float* q_all, const float* k_all,
                 const float* v_all, const float* a, const float* b,
                 const float* a_neg, const float* dt_b,
                 int n_v_heads, int n_k_heads, int d_k, int d_v,
                 float* out_all, std::vector<float>& scratch) {
    if ((int)scratch.size() < 2 * d_k + d_v) scratch.resize(2 * d_k + d_v);
// ⚠ HEAD MAPPING: value head hv reads key head `hv % n_k_heads` (TILE), NOT
// `hv / (n_v/n_k)` (interleave). transformers writes
// `repeat_interleave(n_v//n_k, dim=2)`, but llama.cpp uses `ggml_repeat_4d`,
// which TILES — and against the real Bonsai-27B weights tiling is what matches:
//   block-0 core-attn sum over 5 tokens: tile 3.5452 vs oracle 3.5422 (0.1%),
//   interleave 2.7703 (22% off). Verified with scripts/diff_block.py.
// The two agree only when n_v == n_k, which is why Qwen3.5-2B (16/16) would
// never expose this and Bonsai-27B (48/16) does.
    for (int hv = 0; hv < n_v_heads; ++hv) {
        const int hk = hv % n_k_heads;
        // NOT -expf(a_neg[...]): ssm_a already holds -exp(A_log). See header.
        const float g = a_neg[hv] * softplus(a[hv] + dt_b[hv]);
        const float beta = sigmoidf(b[hv]);
        step_head(S_all + (size_t)hv * d_k * d_v,
                  q_all + (size_t)hk * d_k,
                  k_all + (size_t)hk * d_k,
                  v_all + (size_t)hv * d_v,
                  g, beta, d_k, d_v,
                  out_all + (size_t)hv * d_v, scratch.data());
    }
}

}  // namespace gdn
