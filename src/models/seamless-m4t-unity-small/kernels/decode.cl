// Token-selection arithmetic on the GPU: log-softmax, logit masking, score
// accumulation, and argmax/top-k. Operates in fp32 (matches the host beam's
// precision) even though the logits arrive as fp16. The beam/greedy *sequence
// bookkeeping* (assembling hypotheses) stays scalar C++ — it is control flow,
// not arithmetic.

// Copy fp16 logits -> fp32 buffer at offset.
__kernel void logits_to_f32(__global const storage_t* in, __global float* out,
                            const int off, const int n) {
    int i = get_global_id(0);
    if (i < n) out[off + i] = (float)LOAD(in, i);
}

// log_softmax of fp16 logits [vocab] into fp32 scored[off .. off+vocab).
// Implemented as a no-barrier, no-big-single-thread-loop sequence (this Adreno
// silently no-ops a single work-item loop of 20k exp() and deadlocks on
// __local barrier reductions). LSM_G work-items each stride over the vocab.
//   1. lsm_partial_max  -> partial[g] = max over chunk g
//   2. lsm_reduce_max   -> scalar[0]  = max over partial   (single thread, 256 cmps)
//   3. lsm_partial_sum  -> partial[g] = sum exp(logit-max) over chunk g
//   4. lsm_reduce_sum   -> scalar[1]  = sum over partial    (single thread, 256 adds)
//   5. lsm_write        -> scored[off+i] = logit[i] - (max + log(sum))   (parallel)
#define LSM_G 256
__kernel void lsm_partial_max(__global const storage_t* logits, const int vocab,
                              __global float* partial) {
    int g = get_global_id(0);
    if (g >= LSM_G) return;
    float m = -1e30f;
    for (int i = g; i < vocab; i += LSM_G) m = fmax(m, (float)LOAD(logits, i));
    partial[g] = m;
}
__kernel void lsm_reduce_max(__global const float* partial, __global float* scalar) {
    if (get_global_id(0) != 0) return;
    float m = partial[0];
    for (int i = 1; i < LSM_G; ++i) m = fmax(m, partial[i]);
    scalar[0] = m;
}
__kernel void lsm_partial_sum(__global const storage_t* logits, const int vocab,
                              __global const float* scalar, __global float* partial) {
    int g = get_global_id(0);
    if (g >= LSM_G) return;
    float mx = scalar[0];
    float s = 0.0f;
    for (int i = g; i < vocab; i += LSM_G) s += native_exp((float)LOAD(logits, i) - mx);
    partial[g] = s;
}
__kernel void lsm_reduce_sum(__global const float* partial, __global float* scalar) {
    if (get_global_id(0) != 0) return;
    float s = 0.0f;
    for (int i = 0; i < LSM_G; ++i) s += partial[i];
    scalar[1] = s;
}
__kernel void lsm_write(__global const storage_t* logits, __global float* scored,
                        const int off, const int vocab, __global const float* scalar) {
    int i = get_global_id(0);
    if (i >= vocab) return;
    float lsum = scalar[0] + log(scalar[1]);
    scored[off + i] = (float)LOAD(logits, i) - lsum;
}

// Apply decode masks to scored[off .. off+vocab): -inf at pad / unk (if >=0),
// force a single token (force_tok>=0 -> all others -inf), suppress eos.
__kernel void mask_region(__global float* buf, const int off, const int vocab,
                          const int pad, const int unk, const int force_tok,
                          const int suppress_eos, const int eos) {
    int i = get_global_id(0);
    if (i >= vocab) return;
    const float NEG = -1e30f;
    int gi = off + i;
    if (i == pad) { buf[gi] = NEG; return; }
    if (unk >= 0 && i == unk) { buf[gi] = NEG; return; }
    if (force_tok >= 0 && i != force_tok) { buf[gi] = NEG; return; }
    if (suppress_eos && i == eos) { buf[gi] = NEG; return; }
}

// buf[off+i] += scalar
__kernel void add_scalar_region(__global float* buf, const int off, const int n, const float scalar) {
    int i = get_global_id(0);
    if (i < n) buf[off + i] += scalar;
}

// argmax over buf[0..n): write best index/value (single thread).
__kernel void argmax_f32(__global const float* buf, const int n,
                         __global int* out_idx, __global float* out_val) {
    if (get_global_id(0) != 0) return;
    int best = 0; float bv = buf[0];
    for (int i = 1; i < n; ++i) if (buf[i] > bv) { bv = buf[i]; best = i; }
    out_idx[0] = best; out_val[0] = bv;
}

// buf[idx] = val (single thread) — used to mask out a found top-k entry.
__kernel void set_at_f32(__global float* buf, const int idx, const float val) {
    if (get_global_id(0) == 0) buf[idx] = val;
}

// Device-index variant for beam top-k: record the argmax pick (out_idx/out_val) into
// cand_idx[c]/cand_val[c] and mask buf[idx]=negval — all on-GPU, so the cand_size
// picks are read back in ONE bulk copy instead of cand_size blocking readbacks/step.
__kernel void set_at_dev(__global float* buf, __global const int* idx_buf,
                         __global const float* val_buf, const float negval,
                         __global int* cand_idx, __global float* cand_val, const int c) {
    if (get_global_id(0) != 0) return;
    int idx = idx_buf[0];
    cand_idx[c] = idx;
    cand_val[c] = val_buf[0];
    buf[idx] = negval;
}

// Cooperative argmax: one wave (64 threads) strides the vocab, then tree-reduces.
// Tie-break = (higher value) OR (equal value AND smaller index) → matches the
// sequential scan's "first max wins" EXACTLY, so the selected token is bit-identical.
// Replaces the single-work-item argmax_f32 (8 ms/call over the 10-20K vocab).
__kernel __attribute__((reqd_work_group_size(64, 1, 1)))
void argmax_coop(__global const float* buf, const int n,
                 __global int* out_idx, __global float* out_val) {
    __local float lv[64];
    __local int li[64];
    int tid = get_local_id(0);
    float bv = -3.0e38f; int bi = 0;
    for (int i = tid; i < n; i += 64) { float v = buf[i]; if (v > bv) { bv = v; bi = i; } }
    lv[tid] = bv; li[tid] = bi;
    barrier(CLK_LOCAL_MEM_FENCE);
    for (int p = 32; p > 0; p >>= 1) {
        if (tid < p) {
            if (lv[tid + p] > lv[tid] || (lv[tid + p] == lv[tid] && li[tid + p] < li[tid])) {
                lv[tid] = lv[tid + p]; li[tid] = li[tid + p];
            }
        }
        barrier(CLK_LOCAL_MEM_FENCE);
    }
    if (tid == 0) { out_idx[0] = li[0]; out_val[0] = lv[0]; }
}
