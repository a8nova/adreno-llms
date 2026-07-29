// Gated DeltaNet recurrence — the operator with no counterpart in this repo.
//
// MIRRORS src/delta_rule.h LINE FOR LINE. That header is differential-tested
// against transformers' torch_recurrent_gated_delta_rule (scripts/test_delta_net.py,
// max rel err ~5e-7 at both 3:1 and 1:1 v:k ratios). If you change the math here,
// change it there and re-run that test.
//
// Per VALUE head the state S is a [d_k x d_v] fp32 MATRIX:
//     S     *= exp(g)
//     kv_mem = S^T k                       <-- reduction ACROSS d_k
//     delta  = (v - kv_mem) * beta
//     S     += outer(k, delta)
//     out    = S^T q
//
// THE STATE IS ALWAYS fp32, even in an fp16 build. As half it underflows to 0
// or saturates within a few timesteps (the mamba2-130m lesson) and the model
// emits an empty transcript.
//
// Parallelisation: ONE WORKGROUP PER VALUE HEAD, d_v work-items per group —
// work-item j owns COLUMN j of S for the whole step. kv_mem[j] and out[j] are
// then per-column reductions with NO cross-thread communication; only the two
// l2norms need a group reduction. Adjacent j read adjacent S[i*d_v + j], so
// every state access is coalesced.
//
// State traffic per head per token: 2 reads + 1 write of d_k*d_v floats.

#ifdef USE_FP16
  #pragma OPENCL EXTENSION cl_khr_fp16 : enable
  typedef half storage_t;
  #define LOAD(p, i)     vload_half((i), (p))
#else
  typedef float storage_t;
  #define LOAD(p, i)     ((p)[(i)])
#endif

#ifndef DN_WG
#define DN_WG 128          // == d_v; one work-item per state column
#endif

// PRECISION: use exp/sqrt, NOT native_exp/rsqrt. The fast variants cost
// precision the delta rule cannot spare — the recurrent state compounds error
// across every token, and the logit distribution at a typical position is flat
// enough (top-3 within 0.2 nats on the 27B) that a small drift flips the argmax.
// Measured: with native_exp the device emitted ':' where the host reference and
// the llama.cpp oracle both emit ' Paris' (the oracle's #3 vs #1).
inline float dn_silu(float v)    { return v / (1.0f + exp(-v)); }
inline float dn_sigmoid(float v) { return 1.0f / (1.0f + exp(-v)); }
// softplus with torch's large-input guard (threshold 20)
inline float dn_softplus(float v) { return v > 20.0f ? v : log1p(exp(v)); }

// Group sum over `lm` (DN_WG entries). Returns the total to every work-item.
inline float dn_group_sum(__local float* lm, int lid, float v) {
    lm[lid] = v;
    barrier(CLK_LOCAL_MEM_FENCE);
    for (int s = DN_WG >> 1; s > 0; s >>= 1) {
        if (lid < s) lm[lid] += lm[lid + s];
        barrier(CLK_LOCAL_MEM_FENCE);
    }
    const float t = lm[0];
    barrier(CLK_LOCAL_MEM_FENCE);
    return t;
}

// One decode step, all value heads.
//   S    : [n_v_heads * d_k * d_v] fp32, updated in place
//   qkv  : [2*key_dim + value_dim] post-conv, post-silu
//   a, b : [n_v_heads] raw projections (ssm_alpha / ssm_beta outputs)
//   a_neg  : [n_v_heads] f32 — GGUF `ssm_a`, ALREADY -exp(A_log). Do NOT exp
//            again (llama.cpp src/models/qwen35.cpp multiplies softplus by it
//            directly). Double-exp shortens the state's memory.
//   dt_bias: [n_v_heads] f32
//   out  : [n_v_heads * d_v] fp32
__kernel __attribute__((reqd_work_group_size(DN_WG, 1, 1)))
void delta_net_step(__global float* S,
                    __global const storage_t* qkv,
                    __global const storage_t* a,
                    __global const storage_t* b,
                    __global const float* a_neg,
                    __global const float* dt_bias,
                    __global float* out,
                    const int n_v_heads,
                    const int n_k_heads,
                    const int d_k,
                    const int d_v) {
    const int hv = get_group_id(0);          // value head
    const int j = get_local_id(0);           // state column == d_v index
    if (hv >= n_v_heads) return;

    // HEAD MAPPING: TILE, not interleave — value head hv reads key head
    // hv % n_k_heads. llama.cpp's ggml_repeat_4d tiles; transformers'
    // repeat_interleave does not, and tiling is what matches the real weights
    // (block-0 core sum 3.5452 vs oracle 3.5422; interleave gives 2.7703).
    // Identical only when n_v == n_k, so Qwen3.5-2B (16/16) cannot expose this.
    const int hk = hv % n_k_heads;

    const int key_dim = n_k_heads * d_k;
    __global const storage_t* q_in = qkv + (size_t)hk * d_k;
    __global const storage_t* k_in = qkv + key_dim + (size_t)hk * d_k;
    __global const storage_t* v_in = qkv + 2 * key_dim + (size_t)hv * d_v;

    __local float lm[DN_WG];

    // ---- l2norm(q), l2norm(k), eps 1e-6; then q *= 1/sqrt(d_k) ----
    // Work-item j holds element j (d_k == d_v == DN_WG on this model).
    const float qv_raw = (j < d_k) ? (float)LOAD(q_in, j) : 0.0f;
    const float kv_raw = (j < d_k) ? (float)LOAD(k_in, j) : 0.0f;
    const float qss = dn_group_sum(lm, j, qv_raw * qv_raw);
    const float kss = dn_group_sum(lm, j, kv_raw * kv_raw);
    const float q_scale = (1.0f / sqrt(qss + 1e-6f)) * (1.0f / sqrt((float)d_k));
    const float k_scale = 1.0f / sqrt(kss + 1e-6f);

    // Share the normalised q/k across the group: every work-item needs the
    // whole vector to walk down its column.
    __local float lq[DN_WG], lk[DN_WG];
    lq[j] = qv_raw * q_scale;
    lk[j] = kv_raw * k_scale;
    barrier(CLK_LOCAL_MEM_FENCE);

    // ---- per-head scalars ----
    const float g = a_neg[hv] * dn_softplus((float)LOAD(a, hv) + dt_bias[hv]);
    const float decay = exp(g);
    const float beta = dn_sigmoid((float)LOAD(b, hv));

    __global float* Sh = S + (size_t)hv * d_k * d_v;

    // ---- pass 1: kv_mem[j] = sum_i (S[i][j] * decay) * k[i] ----
    // The decay is folded in here rather than written back, so the state is
    // read twice and written once instead of read/written twice.
    float kv_mem = 0.0f;
    for (int i = 0; i < d_k; ++i)
        kv_mem += Sh[(size_t)i * d_v + j] * decay * lk[i];

    // ---- delta[j] = (v[j] - kv_mem[j]) * beta ----
    const float delta = ((float)LOAD(v_in, j) - kv_mem) * beta;

    // ---- pass 2: S = S*decay + outer(k, delta); out[j] = sum_i S[i][j]*q[i] ----
    float acc = 0.0f;
    for (int i = 0; i < d_k; ++i) {
        const size_t idx = (size_t)i * d_v + j;
        const float s = Sh[idx] * decay + lk[i] * delta;
        Sh[idx] = s;
        acc += s * lq[i];
    }
    out[(size_t)hv * d_v + j] = acc;
}

// Gated RMSNorm applied per (value head) row of d_v, then written back.
//
// ORDER: normalise the RAW input, THEN gate. mamba2-130m's
// mamba_rms_norm_gated.cl normalises the GATED PRODUCT — that is a different
// function and produces quietly-wrong logits here. transformers comments this
// explicitly ("# Norm before gate").
//     y = weight * (x * rsqrt(mean(x^2) + eps)) * silu(gate)
__kernel __attribute__((reqd_work_group_size(DN_WG, 1, 1)))
void delta_net_norm_gate(__global float* x,              // [n_v_heads * d_v] in/out
                         __global const storage_t* gate, // [n_v_heads * d_v]
                         __global const float* weight,   // [d_v]
                         const int n_v_heads,
                         const int d_v,
                         const float eps) {
    const int hv = get_group_id(0);
    const int j = get_local_id(0);
    if (hv >= n_v_heads) return;

    __local float lm[DN_WG];
    __global float* row = x + (size_t)hv * d_v;
    const float v = (j < d_v) ? row[j] : 0.0f;
    const float ss = dn_group_sum(lm, j, v * v);
    if (j >= d_v) return;
    const float inv = 1.0f / sqrt(ss / (float)d_v + eps);
    const float gv = (float)LOAD(gate, (size_t)hv * d_v + j);
    row[j] = (v * inv * weight[j]) * dn_silu(gv);
}

// Causal depthwise conv over ALL qkv channels + silu, decode step (seq_len 1).
// conv_state is [channels][conv_k] fp32, rolled left by one each step.
// GGUF `ssm_conv1d.weight` is [conv_k, channels] with the tap index fastest,
// so channel c's taps live at w[c*conv_k + k]. There is no bias tensor.
// State is fp32 for the same reason the recurrent state is.
__kernel void delta_net_conv1d(__global storage_t* qkv,       // [channels] in/out
                               __global float* conv_state,    // [channels][conv_k]
                               __global const float* w,       // [channels][conv_k]
                               const int channels,
                               const int conv_k) {
    const int c = get_global_id(0);
    if (c >= channels) return;
    __global float* win = conv_state + (size_t)c * conv_k;
    for (int k = 0; k < conv_k - 1; ++k) win[k] = win[k + 1];
    win[conv_k - 1] = (float)LOAD(qkv, c);
    float acc = 0.0f;
    __global const float* wc = w + (size_t)c * conv_k;
    for (int k = 0; k < conv_k; ++k) acc += win[k] * wc[k];
#ifdef USE_FP16
    vstore_half(dn_silu(acc), c, qkv);
#else
    qkv[c] = dn_silu(acc);
#endif
}
