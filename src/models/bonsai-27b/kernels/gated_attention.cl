// 'F'-block (gated full attention) helpers for the qwen35 hybrid stack.
//
// The GQA core itself (gqa_attn_scores / gqa_softmax / gqa_attn_out in
// attention.cl) is reused UNCHANGED — those kernels already take head_dim as a
// runtime argument, so they run at D=256 as-is. What Qwen3.5/3.6 adds on top,
// and what lives here, is:
//   1. q_proj emits q AND the output gate, interleaved PER HEAD  -> attn_split_qg
//   2. RoPE rotates only the first `rot` of `D` dims             -> rope_partial
//   3. the attention output is gated by sigmoid(gate)            -> attn_apply_gate
//
// Mirrors Model::full_attn() in src/reference_model.h.

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

// De-interleave q and the output gate.
//
// `q_proj` emits n_heads * head_dim * 2 and upstream views it as (-1, 2*D) then
// chunks on the LAST dim. So within EACH head's 2*D outputs, [0,D) is q and
// [D,2*D) is the gate — they are NOT two contiguous halves of the whole tensor.
// Treating `attn_q.weight` as n_heads*D is off by 2x and silently mixes gate
// values into q.
//
//   qg   : [n_heads * 2 * D]
//   q    : [n_heads * D]
//   gate : [n_heads * D]
__kernel void attn_split_qg(__global const storage_t* qg,
                            __global storage_t* q,
                            __global storage_t* gate,
                            const int n_heads,
                            const int D) {
    const int idx = get_global_id(0);          // over n_heads * D
    if (idx >= n_heads * D) return;
    const int h = idx / D;
    const int i = idx - h * D;
    const size_t base = (size_t)h * 2 * D;
    STORE(q, idx, LOAD(qg, base + i));
    STORE(gate, idx, LOAD(qg, base + D + i));
}

// PARTIAL RoPE, NeoX pairing.
//
// Only the first `rot` dims of each head rotate (64 of 256 here —
// partial_rotary_factor 0.25); dims [rot, D) pass through untouched. Pair
// (i, i + rot/2), NOT (i, i + D/2) — using D/2 reaches outside the rotary
// block entirely and corrupts the pass-through region.
//
// Interleaved mRoPE (mrope_section [11,11,10]) collapses to plain RoPE for
// text-only input, because the T/H/W position ids are identical — so the host
// builds one cos/sin table and this kernel just indexes it.
//
//   x        : [n_heads * D] in/out
//   cos, sin : [max_pos * rot/2] tables, fp32
__kernel void rope_partial(__global storage_t* x,
                           __global const float* cos_tab,
                           __global const float* sin_tab,
                           const int n_heads,
                           const int D,
                           const int rot,
                           const int pos) {
    const int rot_half = rot >> 1;
    const int idx = get_global_id(0);          // over n_heads * half
    if (idx >= n_heads * rot_half) return;
    const int h = idx / rot_half;
    const int i = idx - h * rot_half;

    const float c = cos_tab[(size_t)pos * rot_half + i];
    const float s = sin_tab[(size_t)pos * rot_half + i];

    __global storage_t* p = x + (size_t)h * D;
    const float a = (float)LOAD(p, i);
    const float b = (float)LOAD(p, i + rot_half);
    STORE(p, i, a * c - b * s);
    STORE(p, i + rot_half, a * s + b * c);
}

// attn_out *= sigmoid(gate)
//
// NOTE the activation: the 'F'-block output gate is SIGMOID. The 'L'-block SSM
// gate (delta_net_norm_gate) is SILU. They are different gates on different
// blocks; swapping them is a silent-wrongness bug.
__kernel void attn_apply_gate(__global storage_t* att,
                              __global const storage_t* gate,
                              const int n) {
    const int i = get_global_id(0);
    if (i >= n) return;
    const float g = (float)LOAD(gate, i);
    // precise exp: see the precision note in delta_net.cl
    STORE(att, i, (float)LOAD(att, i) * (1.0f / (1.0f + exp(-g))));
}
