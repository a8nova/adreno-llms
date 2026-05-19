// flow_wn_fused.cl — single-dispatch fused WaveNet residual layer for
// VITS flow_inverse. Replaces the 8-dispatch sequence:
//   in conv(K=5) → gated_activation → rs conv(K=1) → split_rs_fold (or add)
// with one kernel. Dispatch count for the whole flow_inverse drops from
// ~160 → ~48, which directly attacks the 75-200ms inter-kernel idle that
// dominates Adreno 620 vocoder/flow wall time.
//
// Layout (channels-first, B=1):
//   h          : [C, T]            (input AND output for layers 0..N-2)
//   w_in       : [2C, C, K=5]      conv weights (weight_norm already applied host-side)
//   b_in       : [2C]
//   w_rs       : [rs_oc, C, 1]     (rs_oc = 2C for non-last, C for last layer)
//   b_rs       : [rs_oc]
//   skip_sum   : [C, T]            accumulator across layers (we += into it)
//
// Workgroup model:
//   gws = (C, T)        local = (C, 1)
//   Each workgroup processes ONE timestep `t`.
//   Workitem index `c` in [0, C) handles channel c at this timestep.
//
// Per-WG flow:
//   1. Each WI computes pre_act_t[c,t] and pre_act_g[c,t] by reading the
//      input slice h[*, t-pad..t+pad] from global. Pad = (K-1)/2 = 2.
//   2. Gated activation: tanh(t) * sigmoid(g). Store in __local acts[c].
//   3. Barrier.
//   4. Each WI computes the rs outputs that THIS channel slot owns:
//        - layers 0..N-2: WI c writes h_out[c,t] += res = sum_ci(w_rs[c,ci]*acts[ci])
//                         and        skip_sum[c,t] += skip = sum_ci(w_rs[C+c,ci]*acts[ci])
//        - last layer  :   WI c writes skip_sum[c,t] += sum_ci(w_rs[c,ci]*acts[ci])
//
// Correctness invariant: h and h_out must be DIFFERENT buffers (caller
// pings-pongs across layers). Writing into the same buffer we're reading
// would race across workgroups for the K=5 conv.

#ifdef USE_FP16
  #pragma OPENCL EXTENSION cl_khr_fp16 : enable
  typedef half storage_t;
  #define LOAD(p,i)   vload_half((i),(p))
  #define STORE(p,i,v) vstore_half((v),(i),(p))
#else
  typedef float storage_t;
  #define LOAD(p,i)   ((p)[(i)])
  #define STORE(p,i,v) ((p)[(i)] = (v))
#endif

// Sigmoid with the same ±15 clamp the host implementation uses.
inline float sigmoid_clamp(float x) {
    if (x > 15.0f) x = 15.0f;
    else if (x < -15.0f) x = -15.0f;
    return 1.0f / (1.0f + exp(-x));
}

// Compile-time C; VITS flow always uses C=192.
#ifndef FLOW_C
#define FLOW_C 192
#endif
#ifndef WN_K
#define WN_K 5
#endif
#define WN_PAD ((WN_K - 1) / 2)

__kernel void flow_wn_layer_fused(
    __global const storage_t* h,
    __global storage_t*       h_out,        // ignored when is_last=1
    __global const storage_t* w_in,         // [2C, C, K] row-major
    __global const storage_t* b_in,         // [2C]
    __global const storage_t* w_rs,         // [rs_oc, C] (K=1)
    __global const storage_t* b_rs,         // [rs_oc]
    __global storage_t*       skip_sum,     // [C, T]
    const int T,
    const int is_last)
{
    const int c = get_local_id(0);
    const int t = get_group_id(1);
    const int C = FLOW_C;

    // ── Step 1a: cooperatively load the K×C h-tile into __local memory.
    // Each WG processes a single t — it reads h[*, t-PAD..t+PAD], which is
    // K columns × C rows. All 192 workitems read the SAME h values across
    // the K input timesteps, so global → local once amortizes K*C reads
    // across the WG. Each WI loads K elements (one per kernel position) for
    // its own ci=c row.
    //
    // h_tile is stored as [k][ci] row-major. For vload4 across ci, we need ci
    // to be the innermost stride-1 index — matching the IC-inner layout we
    // also use for w_in below.
    __local float h_tile[WN_K * FLOW_C];        // 5 * 192 * 4 = 3840 bytes
    #pragma unroll
    for (int k = 0; k < WN_K; ++k) {
        const int tk = t + k - WN_PAD;
        float hv = 0.0f;
        if (tk >= 0 && tk < T) {
            hv = (float)LOAD(h, c * T + tk);
        }
        h_tile[k * C + c] = hv;
    }
    barrier(CLK_LOCAL_MEM_FENCE);

    // ── Step 1b: compute pre_acts for this (c, t) using the local h_tile and
    // the REORDERED w_in layout [OC, K, IC] (host transposed ic↔k). This
    // makes the inner ci loop a stride-1 read on w_in — vectorizable as
    // half4 across 4 adjacent IC at a time.
    // tanh weights at w_in[c, *, *]; gate weights at w_in[C + c, *, *].
    float pre_t = (float)LOAD(b_in, c);
    float pre_g = (float)LOAD(b_in, C + c);
    const int w_base_t = c * WN_K * C;        // start of [c, 0, 0]
    const int w_base_g = (C + c) * WN_K * C;
    #pragma unroll
    for (int k = 0; k < WN_K; ++k) {
        const int w_off_t = w_base_t + k * C;
        const int w_off_g = w_base_g + k * C;
        const int h_off   = k * C;
#ifdef USE_FP16
        // Inner loop vectorized 4-wide over ci. C=192 → 48 iters.
        // vload_half4 reads 4 fp16 values and returns float4 (the conversion
        // is performed by the hardware), so accumulation is in fp32.
        for (int ci4 = 0; ci4 < C; ci4 += 4) {
            const float4 wt4 = vload_half4(0, (__global const half*)w_in + w_off_t + ci4);
            const float4 wg4 = vload_half4(0, (__global const half*)w_in + w_off_g + ci4);
            const float4 hv4 = (float4)(h_tile[h_off + ci4 + 0],
                                        h_tile[h_off + ci4 + 1],
                                        h_tile[h_off + ci4 + 2],
                                        h_tile[h_off + ci4 + 3]);
            pre_t += dot(wt4, hv4);
            pre_g += dot(wg4, hv4);
        }
#else
        for (int ci = 0; ci < C; ++ci) {
            const float hv = h_tile[h_off + ci];
            pre_t += (float)LOAD(w_in, w_off_t + ci) * hv;
            pre_g += (float)LOAD(w_in, w_off_g + ci) * hv;
        }
#endif
    }

    // ── Step 2: gated activation.
    const float act_v = tanh(pre_t) * sigmoid_clamp(pre_g);

    // ── Step 3: publish to local memory.
    __local float acts[FLOW_C];
    acts[c] = act_v;
    barrier(CLK_LOCAL_MEM_FENCE);

    // ── Step 4: rs convolution (K=1) — each workitem owns its (c, t) output
    //              for the "res" part, and the (c, t) slot of skip_sum for the
    //              "skip" part. Non-last layers: rs_oc = 2C, do both. Last
    //              layer: rs_oc = C, only do the skip part (whose weights live
    //              at w_rs[0..C-1, *]).
    if (is_last == 0) {
        // res = w_rs[c, *] @ acts + b_rs[c]
        float r_res = (float)LOAD(b_rs, c);
        for (int ci = 0; ci < C; ++ci) {
            r_res += (float)LOAD(w_rs, c * C + ci) * acts[ci];
        }
        // h_out[c, t] = h[c, t] + r_res
        const float h_old = (float)LOAD(h, c * T + t);
        STORE(h_out, c * T + t, h_old + r_res);

        // skip = w_rs[C+c, *] @ acts + b_rs[C+c]
        float r_skip = (float)LOAD(b_rs, C + c);
        for (int ci = 0; ci < C; ++ci) {
            r_skip += (float)LOAD(w_rs, (C + c) * C + ci) * acts[ci];
        }
        const float ss_old = (float)LOAD(skip_sum, c * T + t);
        STORE(skip_sum, c * T + t, ss_old + r_skip);
    } else {
        // Last layer: rs_oc = C, skip only.
        float r_skip = (float)LOAD(b_rs, c);
        for (int ci = 0; ci < C; ++ci) {
            r_skip += (float)LOAD(w_rs, c * C + ci) * acts[ci];
        }
        const float ss_old = (float)LOAD(skip_sum, c * T + t);
        STORE(skip_sum, c * T + t, ss_old + r_skip);
    }
}
