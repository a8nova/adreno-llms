// Reference: model_info/transformers_src/modeling_musicgen.py:250-344 MusicgenDecoderLayer.forward
//
// PER-DECODER-LAYER MEGAKERNEL for seq=1 decode (one CFG row).
// Collapses ~24 dispatches/layer (LN, q/k/v/o GEMV, KV append, scores, softmax,
// out, h2t, cross-attn, fc1, GELU, fc2, residual adds) into ONE dispatch.
// This is the dispatch-count lever (PERFORMANCE_ANALYSIS §6: per-dispatch host
// overhead is the wall) AND the correctness fix (AGENT_DIRECTIVE_FP32_ACCUM.md:
// ALL accumulation fp32; weights load fp16; residual stream fp32 in local).
//
// One workgroup (MEGA_WG=256 threads) executes the whole layer for ONE row:
//   residual(fp32 local) = x
//   h = self_attn_layer_norm(x)             (fp32 mean/var, gamma/beta)
//   q = q_proj(h); k = k_proj(h); v = v_proj(h)   (1024² GEMV, fp16 W, fp32 acc)
//   append k,v to KV cache at start_pos (global, fp16)
//   self causal attn over cache rows [0, start_pos] (fp32 scores/softmax)
//   a = out_proj(attn)
//   residual += a                            (fp32 local)
//   h = encoder_attn_layer_norm(residual)
//   q2 = q_proj_x(h)                          (cross: Q from h)
//   cross attn over PRECOMPUTED k_cross/v_cross [enc_len,1024] (fp32)
//   c = out_proj_x(cross)
//   residual += c
//   h = final_layer_norm(residual)
//   f = gelu(fc1(h))   [1024->4096]           (fp32 acc)
//   o = fc2(f)         [4096->1024]
//   residual += o
//   store residual -> out (fp16)
//
// Weight layout: every nn.Linear weight is row-major [N, K] (out, in); all
// bias-free. LayerNorms carry gamma(weight)+beta(bias) [hidden]. KV cache is
// token-major [pos, hidden] fp16. Cross K/V are precomputed [enc_len, hidden]
// fp16 (decoder_cross_kv_precompute kernel, once per generation).

#ifdef USE_FP16
  #pragma OPENCL EXTENSION cl_khr_fp16 : enable
  typedef half storage_t;
  #define LOAD(p, i)     vload_half((i), (p))
  #define STORE(p, i, v) vstore_half((v), (i), (p))
  // 8-wide vector load → float8 (mega_attn_core KV scan vectorization).
  #define LOADV8(p, i8)  convert_float8(vload_half8((i8), (p)))
#else
  typedef float storage_t;
  #define LOAD(p, i)     ((p)[(i)])
  #define STORE(p, i, v) ((p)[(i)] = (v))
  #define LOADV8(p, i8)  vload8((i8), (p))
#endif

// MEGA_WG is the workgroup size (threads per CFG row). Lever 1 raises it from
// the original 256 to 1024 (4× the streaming threads) and the GEMV below is
// rewritten ROW-COOPERATIVE + COALESCED. Overridable via -D MEGA_WG=N at build.
#ifndef MEGA_WG
  #define MEGA_WG     512
#endif
#define MEGA_HIDDEN   1024
#define MEGA_HEADS    16
#define MEGA_HEAD_DIM 64
#define MEGA_FFN      4096
#define MEGA_MAXK     2048       // KV cache cap (self-attn positions) = decoder
                                 // max_position_embeddings → ~40.9 s @ 50 Hz.
                                 // 256→5.1 s, 512→10.2 s (the old distort-at-10s
                                 // bug), 2048→40.9 s. attn-core local sc[] scales
                                 // with this: 2048 floats = 8 KB (fits 32 KB).
#define MEGA_MAXENC   64         // cross-attn enc length cap (>= 11)

// Lever 1: ROW-COOPERATIVE coalesced GEMV tiling. MEGA_KGROUP threads cooperate
// on ONE output row; consecutive lanes read CONSECUTIVE K elements (coalesced
// weight load along K — the original thread-per-row map read W[n*K+k] at stride
// K across threads, fully uncoalesced). MEGA_ROWS_PAR rows are computed in
// parallel per pass = MEGA_WG / MEGA_KGROUP.
#ifndef MEGA_KGROUP
  #define MEGA_KGROUP 8
#endif
#define MEGA_ROWS_PAR (MEGA_WG / MEGA_KGROUP)

// ── Campaign-2 #2a: MANY-WORKGROUP FFN (MEGA_FFN_EXTERNAL) ───────────────────
// The single-WG-per-row megakernel caps occupancy at 2 resident workgroups, so
// the memory system is latency-bound far below the ~14 GB/s bus. The FFN
// (fc1+fc2 = 16.8 MB of the 29.4 MB/layer weight stream) is factored into the
// dedicated many-WG kernels below: 32-64 WGs per CFG row stream weights
// concurrently. The mega kernel (built with -D MEGA_FFN_EXTERNAL=1) stops after
// final_layer_norm and hands `normed`+`residual` to them as fp32 GLOBAL scratch
// — fp32 so the FFN GEMV consumes bit-identical inputs to the in-kernel path.
// MEGA_MWG_KGROUP MUST equal the in-mega MEGA_KGROUP rounding contract (16 =
// the tf-depth-validated lane split; per-row sums are then BIT-IDENTICAL to the
// in-mega FFN because lane stride, partial order, and tree reduce are the same).
#ifndef MEGA_MWG_WG
  #define MEGA_MWG_WG 64          // Adreno half-wave (adreno-opencl-quirks.md)
#endif
#ifndef MEGA_MWG_KGROUP
  #define MEGA_MWG_KGROUP 16
#endif
#define MEGA_MWG_ROWS_PAR (MEGA_MWG_WG / MEGA_MWG_KGROUP)

// MEGA_MWG_VEC=4: 64-bit weight fetches. The scalar vload_half path issues one
// 2-byte load per instruction — measured 1.9-2.4 GB/s even at 128 WGs (issue-
// rate bound, not occupancy bound). vload_half2/4 are BROKEN on Adreno 6xx
// (err -11, quirks file), but a plain uint2 vector load reinterpreted with
// as_half4() is an ordinary 64-bit load on the load/store pipe — 4 halves per
// instruction, no half-conversion driver path involved. Requires USE_FP16
// (storage_t == half) and K % 4 == 0. Changes per-row summation order vs the
// scalar path (each lane sums 4-element blocks) → tf-depth re-gated per variant.
#ifndef MEGA_MWG_VEC
  #define MEGA_MWG_VEC 1
#endif
// Same 64-bit fetch for the IN-MEGA GEMV (q/k/v/o + cross projections).
#ifndef MEGA_VEC
  #define MEGA_VEC 1
#endif

// ── Campaign-2 #3: TEXTURE-PATH weights (MEGA_TEX) ───────────────────────────
// Weights as image2d_t (CL_RGBA + CL_HALF_FLOAT: one texel = 4 consecutive K
// halves; width = K/4, height = N). read_imageh fetches a guaranteed-vectorized
// half4 through Adreno's DEDICATED texture pipe + L1 texture cache, which
// dual-issues with the load/store pipe (AGENT_DIRECTIVE_ADRENO_TACTICS.md
// PRIMARY variant; Qualcomm 80-NB295-11). Accumulation order per row is the
// SAME 4-element-block pattern as MEGA_VEC=4 (tf-gated together).
#ifdef MEGA_TEX
__constant sampler_t mega_smp =
    CLK_NORMALIZED_COORDS_FALSE | CLK_ADDRESS_NONE | CLK_FILTER_NEAREST;
#endif

// ── In-kernel LayerNorm over a [hidden] vector held in local memory ──────────
// Reads x_local (fp32), writes y_local (fp32). gamma/beta are fp16 global.
// Population variance + rsqrt(var+eps). Cooperative across the workgroup.
static void mega_layernorm(__local float* x, __local float* y,
                           __global const storage_t* gamma,
                           __global const storage_t* beta,
                           float eps, __local float* red, int lid) {
  // mean
  float ls = 0.0f;
  for (int c = lid; c < MEGA_HIDDEN; c += MEGA_WG) ls += x[c];
  red[lid] = ls; barrier(CLK_LOCAL_MEM_FENCE);
  for (int o = MEGA_WG >> 1; o > 0; o >>= 1) { if (lid < o) red[lid] += red[lid+o]; barrier(CLK_LOCAL_MEM_FENCE); }
  float mean = red[0] / (float)MEGA_HIDDEN;
  barrier(CLK_LOCAL_MEM_FENCE);
  // var
  float lv = 0.0f;
  for (int c = lid; c < MEGA_HIDDEN; c += MEGA_WG) { float d = x[c] - mean; lv += d * d; }
  red[lid] = lv; barrier(CLK_LOCAL_MEM_FENCE);
  for (int o = MEGA_WG >> 1; o > 0; o >>= 1) { if (lid < o) red[lid] += red[lid+o]; barrier(CLK_LOCAL_MEM_FENCE); }
  float inv_std = rsqrt(red[0] / (float)MEGA_HIDDEN + eps);
  barrier(CLK_LOCAL_MEM_FENCE);
  for (int c = lid; c < MEGA_HIDDEN; c += MEGA_WG) {
    float val = (x[c] - mean) * inv_std;
    val = val * LOAD(gamma, c) + LOAD(beta, c);
    y[c] = val;
  }
  barrier(CLK_LOCAL_MEM_FENCE);
}

#ifdef MEGA_FUSE_LN
// In-place LayerNorm over the GEMV's staged local row (MEGA_FUSE_LN: the
// standalone mega_ln_rows / mega_ln_rows_f32 dispatches are fused into their
// consumer GEMVs — every consumer already stages the full row into local, so
// the stats reductions ride on that staging and 3 dispatches/layer vanish).
// Stats are computed REDUNDANTLY by every WG of the row (cheap: 2 reductions
// of MEGA_HIDDEN vs a 32-row GEMV). Reduction tree is MEGA_MWG_WG-shaped —
// differs from the standalone MEGA_WG-shaped tree → NOT bit-identical; gated
// on tf-depth (cb0 ≥ 40) + greedy guard.
static void mwg_ln_local(__local float* xs,
                         __global const storage_t* gamma,
                         __global const storage_t* beta,
                         float eps, __local float* part, int lid) {
  float ls = 0.0f;
  for (int c = lid; c < MEGA_HIDDEN; c += MEGA_MWG_WG) ls += xs[c];
  part[lid] = ls; barrier(CLK_LOCAL_MEM_FENCE);
  for (int o = MEGA_MWG_WG >> 1; o > 0; o >>= 1) { if (lid < o) part[lid] += part[lid+o]; barrier(CLK_LOCAL_MEM_FENCE); }
  const float mean = part[0] / (float)MEGA_HIDDEN;
  barrier(CLK_LOCAL_MEM_FENCE);
  float lv = 0.0f;
  for (int c = lid; c < MEGA_HIDDEN; c += MEGA_MWG_WG) { const float d = xs[c] - mean; lv += d * d; }
  part[lid] = lv; barrier(CLK_LOCAL_MEM_FENCE);
  for (int o = MEGA_MWG_WG >> 1; o > 0; o >>= 1) { if (lid < o) part[lid] += part[lid+o]; barrier(CLK_LOCAL_MEM_FENCE); }
  const float inv_std = rsqrt(part[0] / (float)MEGA_HIDDEN + eps);
  barrier(CLK_LOCAL_MEM_FENCE);
  for (int c = lid; c < MEGA_HIDDEN; c += MEGA_MWG_WG)
    xs[c] = (xs[c] - mean) * inv_std * LOAD(gamma, c) + LOAD(beta, c);
  barrier(CLK_LOCAL_MEM_FENCE);
}
#endif

// y[N] = W[N,K] @ x[K]   (W row-major [out,in] fp16; x fp32 local; y fp32 local)
// ROW-COOPERATIVE + COALESCED (Lever 1): the workgroup is partitioned into
// MEGA_ROWS_PAR groups of MEGA_KGROUP threads. Group `grp` owns output rows
// {grp, grp+ROWS_PAR, ...}; within a group, lane `lane` accumulates over
// k = lane, lane+KGROUP, lane+2*KGROUP, ... so CONSECUTIVE LANES READ
// CONSECUTIVE W ADDRESSES (W[n*K + lane + j*KGROUP]) — coalesced along K. The
// KGROUP partials are reduced through `part` (local scratch [MEGA_WG]). fp32 acc.
static void mega_gemv(__global const storage_t* W, __local const float* x,
                      __local float* y, int N, int K, int lid, __local float* part) {
  const int grp  = lid / MEGA_KGROUP;     // which row-group this thread is in
  const int lane = lid % MEGA_KGROUP;     // lane within the row-group
  for (int n = grp; n < N; n += MEGA_ROWS_PAR) {
    const int wbase = n * K;
    float acc = 0.0f;
#if MEGA_VEC == 4 && defined(USE_FP16)
    // 64-bit weight fetch (uint2 → as_half4): 4 halves per load instruction.
    // The scalar vload_half path is ISSUE-RATE bound (~1.6 GB/s measured);
    // this is the same fetch that took mega_ffn_fc1 1.9 → 3.9 GB/s. Changes
    // each lane's summation order (4-element blocks) → tf-depth re-gated.
    {
      __global const uint2* W4 = (__global const uint2*)(W + wbase);
      for (int k4 = lane; k4 < (K >> 2); k4 += MEGA_KGROUP) {
        const float4 wv = convert_float4(as_half4(W4[k4]));
        const int kb = k4 * 4;
        acc += wv.x * x[kb] + wv.y * x[kb + 1] + wv.z * x[kb + 2] + wv.w * x[kb + 3];
      }
    }
#else
    // Coalesced: at iteration j, lanes 0..KGROUP-1 read W[wbase + j*KGROUP + 0..KGROUP-1].
    for (int k = lane; k < K; k += MEGA_KGROUP) acc += LOAD(W, wbase + k) * x[k];
#endif
    part[lid] = acc;
    barrier(CLK_LOCAL_MEM_FENCE);
    // Reduce KGROUP partials within the group (tree reduction over lanes).
    for (int o = MEGA_KGROUP >> 1; o > 0; o >>= 1) {
      if (lane < o) part[lid] += part[lid + o];
      barrier(CLK_LOCAL_MEM_FENCE);
    }
    if (lane == 0) y[n] = part[grp * MEGA_KGROUP];
    barrier(CLK_LOCAL_MEM_FENCE);
  }
  barrier(CLK_LOCAL_MEM_FENCE);
}

#ifdef MEGA_TEX
// Texture-path GEMV: identical row/lane mapping + reduction to mega_gemv, but
// the weight fetch is read_imageh(W, (k4, n)) — half4 per texture instruction.
static void mega_gemv_tex(__read_only image2d_t W, __local const float* x,
                          __local float* y, int N, int K, int lid, __local float* part) {
  const int grp  = lid / MEGA_KGROUP;
  const int lane = lid % MEGA_KGROUP;
  for (int n = grp; n < N; n += MEGA_ROWS_PAR) {
    float acc = 0.0f;
    // NO #pragma unroll here — measured 0.33 GB/s vs 3.53 plain (10× crater)
    // with BOTH a runtime (K >> 2) bound AND a compile-time (MEGA_HIDDEN >> 2)
    // bound (2026-06-04). The crater is the MEGAKERNEL itself: this GEMV is
    // inlined 6× into one giant function (attn + LN + reductions); unrolling
    // multiplies live registers past the file and spills the whole kernel.
    // fc1/fc2 take unroll-4 fine because they are small standalone kernels.
    // Do not re-try ANY unroll inside decoder_layer_mega: #pragma unroll-4
    // cratered 10× (0.33 GB/s), and even a MANUAL ×2 (one extra float4 live)
    // measured 3.52 → 3.01 GB/s (2026-06-04). The kernel sits exactly at its
    // register ceiling. Faster attn GEMVs require a STANDALONE-kernel split.
    for (int k4 = lane; k4 < (K >> 2); k4 += MEGA_KGROUP) {
      const float4 wv = convert_float4(read_imageh(W, mega_smp, (int2)(k4, n)));
      const int kb = k4 * 4;
      acc += wv.x * x[kb] + wv.y * x[kb + 1] + wv.z * x[kb + 2] + wv.w * x[kb + 3];
    }
    part[lid] = acc;
    barrier(CLK_LOCAL_MEM_FENCE);
    for (int o = MEGA_KGROUP >> 1; o > 0; o >>= 1) {
      if (lane < o) part[lid] += part[lid + o];
      barrier(CLK_LOCAL_MEM_FENCE);
    }
    if (lane == 0) y[n] = part[grp * MEGA_KGROUP];
    barrier(CLK_LOCAL_MEM_FENCE);
  }
  barrier(CLK_LOCAL_MEM_FENCE);
}
#endif

// ── Stage 6: PER-GROUP int8 GEMV ─────────────────────────────────────────────
// y[n] = sum_g scale[n,g] * sum_{k in group g}(Wq[n,k] * x[k]).
// Wq int8 row-major [out,in]; scale fp16 [out, K/MEGA_QGROUP], one scale per
// MEGA_QGROUP-wide column group (per-group is far more accurate than per-row on
// a deep decoder's near-tied logits — per-row int8 dropped tf-depth 46→20).
// fp32 accumulation (AGENT_DIRECTIVE_FP32_ACCUM). Halves weight bytes streamed
// (the memory-bound wall, §3) + ~K/MEGA_QGROUP extra fp16 scales (tiny).
#define MEGA_QGROUP 128
// ROW-COOPERATIVE + COALESCED int8 GEMV (mirrors mega_gemv). MEGA_KGROUP lanes
// cooperate per output row; each lane sums its strided slice of K then the
// per-group fp16 scale is applied at the END (valid only because the scale is
// per-128-column-group AND MEGA_QGROUP % MEGA_KGROUP == 0, so each lane's
// stride-KGROUP slice still falls within consistent 128-wide scale groups —
// to keep it exact we apply the scale per QGROUP block as before but split the
// block's inner sum across lanes). lane j accumulates a per-group partial; the
// group's KGROUP partials are reduced via `part`.
static void mega_gemv_i8(__global const char* Wq, __global const storage_t* scale,
                         __local const float* x, __local float* y, int N, int K, int lid,
                         __local float* part) {
  const int ngroups = K / MEGA_QGROUP;
  const int grp  = lid / MEGA_KGROUP;
  const int lane = lid % MEGA_KGROUP;
  for (int n = grp; n < N; n += MEGA_ROWS_PAR) {
    const int wbase = n * K;
    const int sbase = n * ngroups;
    float acc = 0.0f;
    for (int g = 0; g < ngroups; ++g) {
      const int kb = g * MEGA_QGROUP;
      float gacc = 0.0f;
      // coalesced within the 128-wide block: lane reads kb+lane, kb+lane+KGROUP,...
      for (int j = lane; j < MEGA_QGROUP; j += MEGA_KGROUP)
        gacc += (float)Wq[wbase + kb + j] * x[kb + j];
      acc += gacc * LOAD(scale, sbase + g);
    }
    part[lid] = acc;
    barrier(CLK_LOCAL_MEM_FENCE);
    for (int o = MEGA_KGROUP >> 1; o > 0; o >>= 1) {
      if (lane < o) part[lid] += part[lid + o];
      barrier(CLK_LOCAL_MEM_FENCE);
    }
    if (lane == 0) y[n] = part[grp * MEGA_KGROUP];
    barrier(CLK_LOCAL_MEM_FENCE);
  }
  barrier(CLK_LOCAL_MEM_FENCE);
}

// ── M=2-NATIVE megakernel (Stage 1) ─────────────────────────────────────────
// Processes BOTH CFG rows in ONE dispatch: gws={2*MEGA_WG}, lws={MEGA_WG}, so
// workgroup 0 = row0 (cond/KV-bank0/k_cross row0), workgroup 1 = row1
// (uncond/KV-bank32/k_cross row1). Each workgroup is fully independent and runs
// the SAME single-row body as before — the per-WG math is bit-identical to the
// pre-stage-1 per-row dispatch. The shared weights (LN/proj/FFN) are identical
// for both rows; only the residual stream (x_in/x_out), self-attn KV cache, and
// precomputed cross-K/V differ per row, so those are passed as row0/row1 pairs
// and selected by get_group_id(0). This kills the 4 row-copy dispatches/layer
// (the host operates directly on the shared [2,hidden] buffer) and halves the
// layer dispatch count (1 dispatch/layer instead of 2).
__kernel __attribute__((reqd_work_group_size(MEGA_WG, 1, 1)))
void decoder_layer_mega(
    // I/O — row0 and row1 of the shared [2,hidden] buffer (this kernel reads/
    // writes directly; the host no longer extracts/writes-back rows).
    __global const storage_t* x_in0,  __global storage_t* x_out0,   // row0 (cond)
    __global const storage_t* x_in1,  __global storage_t* x_out1,   // row1 (uncond)
    // self-attn weights (shared by both rows). In MEGA_INT8 each GEMV weight is
    // an int8 tensor (char*) + a per-output-row fp16 scale.
    __global const storage_t* self_ln_w, __global const storage_t* self_ln_b,
#ifdef MEGA_INT8
    __global const char* q_w, __global const storage_t* q_s,
    __global const char* k_w, __global const storage_t* k_s,
    __global const char* v_w, __global const storage_t* v_s,
    __global const char* o_w, __global const storage_t* o_s,
#elif defined(MEGA_TEX)
    __read_only image2d_t q_w, __read_only image2d_t k_w,
    __read_only image2d_t v_w, __read_only image2d_t o_w,
#else
    __global const storage_t* q_w, __global const storage_t* k_w,
    __global const storage_t* v_w, __global const storage_t* o_w,
#endif
    // KV cache (per row; global, fp16, token-major [MEGA_MAXK, hidden])
    __global storage_t* k_cache0, __global storage_t* v_cache0,     // row0
    __global storage_t* k_cache1, __global storage_t* v_cache1,     // row1
    // cross-attn weights (shared) + precomputed K/V (per row)
    __global const storage_t* enc_ln_w, __global const storage_t* enc_ln_b,
#ifdef MEGA_INT8
    __global const char* cq_w, __global const storage_t* cq_s,
    __global const char* co_w, __global const storage_t* co_s,
#elif defined(MEGA_TEX)
    __read_only image2d_t cq_w, __read_only image2d_t co_w,
#else
    __global const storage_t* cq_w, __global const storage_t* co_w,
#endif
    __global const storage_t* k_cross0, __global const storage_t* v_cross0,  // row0
    __global const storage_t* k_cross1, __global const storage_t* v_cross1,  // row1
    // FFN weights (shared)
    __global const storage_t* fin_ln_w, __global const storage_t* fin_ln_b,
#if defined(MEGA_INT8) || defined(MEGA_INT8_FFN)
    __global const char* fc1_w, __global const storage_t* fc1_s,
    __global const char* fc2_w, __global const storage_t* fc2_s,
#else
    __global const storage_t* fc1_w, __global const storage_t* fc2_w,
#endif
    // scalars
    const int start_pos, const int enc_len, const float eps, const float scale
#ifdef MEGA_FFN_EXTERNAL
    // #2a outputs: fp32 global handoff to mega_ffn_fc1/fc2 ([row, hidden]).
    , __global float* ffn_normed, __global float* ffn_resid
#endif
#ifdef MEGA_QKV_EXTERNAL
    // Stage-1 split input: q computed by the external mega_qkv dispatch
    // ([row, hidden] fp32); k/v were appended to the caches by the same.
    , __global const float* q_ext_g
#endif
    ) {

  // Row selection: workgroup 0 → cond, workgroup 1 → uncond. All global I/O and
  // per-row caches are bound here; the body below is row-agnostic.
  const int row = get_group_id(0);
  __global const storage_t* x_in    = (row == 0) ? x_in0    : x_in1;
  __global storage_t*       x_out   = (row == 0) ? x_out0   : x_out1;
  __global storage_t*       k_cache = (row == 0) ? k_cache0 : k_cache1;
  __global storage_t*       v_cache = (row == 0) ? v_cache0 : v_cache1;
  __global const storage_t* k_cross = (row == 0) ? k_cross0 : k_cross1;
  __global const storage_t* v_cross = (row == 0) ? v_cross0 : v_cross1;

  const int lid = get_local_id(0);

  // ── Local state (budget: Adreno 620 = 32KB = 8192 floats) ────────────────
  //   residual  1024  (lives whole kernel)
  //   normed    1024  (LN / GEMV output scratch)
  //   qv        1024  (query projection; survives across attention)
  //   big       4096  (fc1 output AND, in disjoint phases, k/v-append scratch,
  //                    attn_out region [0,1024), and softmax scores past 1024)
  //   red       MEGA_WG  (reduction scratch for LN/softmax AND the GEMV's
  //                       row-cooperative partials — disjoint in time, so the
  //                       SAME buffer serves both; sizing it MEGA_WG covers both)
  //   WG=512 → 3*1024 + 4096 + 512 = 7680 floats = 30.7KB. Fits.
  //   WG=1024 → 8192 floats = 32KB (right at the cap; -D MEGA_WG=1024 only if the
  //             device reports >=32KB usable local mem, checked host-side).
  __local float residual[MEGA_HIDDEN];
  __local float normed[MEGA_HIDDEN];
  __local float qv[MEGA_HIDDEN];
  __local float big[MEGA_FFN];
  __local float red[MEGA_WG];
  // Carved views over `big` (used in non-overlapping phases):
  __local float* kv_scratch = big;          // [1024] k/v before cache append
  __local float* attn_out_l = big;          // [1024] attention output (heads concat)
  __local float* sc          = big + 1024;  // [<=256] softmax scores (disjoint from attn_out_l usage window? see note)
  __local float* ffn1        = big;          // [4096] fc1 output for FFN block

  // load input residual
  for (int c = lid; c < MEGA_HIDDEN; c += MEGA_WG) residual[c] = LOAD(x_in, c);
  barrier(CLK_LOCAL_MEM_FENCE);

  // GEMV dispatch macro: fp16 path uses mega_gemv(W,...); int8 path uses
  // mega_gemv_i8(Wq, scale,...). Keeps the body identical between variants.
#ifdef MEGA_INT8
  #define MGEMV(W, S, X, Y, N, K) mega_gemv_i8((W), (S), (X), (Y), (N), (K), lid, red)
#elif defined(MEGA_TEX)
  #define MGEMV(W, S, X, Y, N, K) mega_gemv_tex((W), (X), (Y), (N), (K), lid, red)
#else
  #define MGEMV(W, S, X, Y, N, K) mega_gemv((W), (X), (Y), (N), (K), lid, red)
#endif
// FFN GEMV: int8 in BOTH MEGA_INT8 (full) and MEGA_INT8_FFN (FFN-only) modes.
#if defined(MEGA_INT8) || defined(MEGA_INT8_FFN)
  #define MGEMV_FFN(W, S, X, Y, N, K) mega_gemv_i8((W), (S), (X), (Y), (N), (K), lid, red)
#else
  #define MGEMV_FFN(W, S, X, Y, N, K) mega_gemv((W), (X), (Y), (N), (K), lid, red)
#endif

  // ══ SELF-ATTENTION BLOCK ══════════════════════════════════════════════════
#ifdef MEGA_QKV_EXTERNAL
  // LN1 + q/k/v ran in the external mega_ln_rows + mega_qkv dispatches
  // (multi-WG, packed [3H, H/4] texture weights, unroll-4 — the in-mega
  // register ceiling caps these GEMVs at 3.5 GB/s; standalone they ride the
  // tex-L1 CFG-row dedup like fc1). q arrives via q_ext_g; k/v are already
  // appended to the caches at start_pos.
  for (int c = lid; c < MEGA_HIDDEN; c += MEGA_WG)
    qv[c] = q_ext_g[row * MEGA_HIDDEN + c];
  barrier(CLK_LOCAL_MEM_FENCE);
#else
  mega_layernorm(residual, normed, self_ln_w, self_ln_b, eps, red, lid);

  // q = q_proj(normed) -> qv[hidden]
  MGEMV(q_w, q_s, normed, qv, MEGA_HIDDEN, MEGA_HIDDEN);

  // k,v = k_proj/v_proj(normed) -> append straight into the cache at start_pos.
  // kv_scratch aliases big[0..1024).
  MGEMV(k_w, k_s, normed, kv_scratch, MEGA_HIDDEN, MEGA_HIDDEN);
  for (int c = lid; c < MEGA_HIDDEN; c += MEGA_WG)
    STORE(k_cache, start_pos * MEGA_HIDDEN + c, kv_scratch[c]);
  barrier(CLK_LOCAL_MEM_FENCE);
  MGEMV(v_w, v_s, normed, kv_scratch, MEGA_HIDDEN, MEGA_HIDDEN);
  for (int c = lid; c < MEGA_HIDDEN; c += MEGA_WG)
    STORE(v_cache, start_pos * MEGA_HIDDEN + c, kv_scratch[c]);
  barrier(CLK_LOCAL_MEM_FENCE);
#endif  // MEGA_QKV_EXTERNAL

  // causal attention over cache rows [0, start_pos]. seq_k = start_pos+1.
  // One head per group of threads: 16 heads, head_dim 64. We loop heads with
  // the whole workgroup (parallel over key positions then over dims).
  {
    const int seq_k = start_pos + 1;   // sc aliases big+1024 (carved above)
    for (int h = 0; h < MEGA_HEADS; ++h) {
      const int qb = h * MEGA_HEAD_DIM;
      // scores[t] = scale * dot(q_h, K_h[t])
      for (int t = lid; t < seq_k; t += MEGA_WG) {
        int kb = t * MEGA_HIDDEN + qb;
        float acc = 0.0f;
        for (int d = 0; d < MEGA_HEAD_DIM; ++d) acc += qv[qb + d] * LOAD(k_cache, kb + d);
        sc[t] = acc * scale;
      }
      barrier(CLK_LOCAL_MEM_FENCE);
      // max
      float m = -1e30f;
      for (int t = lid; t < seq_k; t += MEGA_WG) m = fmax(m, sc[t]);
      red[lid] = m; barrier(CLK_LOCAL_MEM_FENCE);
      for (int o = MEGA_WG >> 1; o > 0; o >>= 1) { if (lid < o) red[lid] = fmax(red[lid], red[lid+o]); barrier(CLK_LOCAL_MEM_FENCE); }
      float maxv = red[0]; barrier(CLK_LOCAL_MEM_FENCE);
      // exp + sum
      float s = 0.0f;
      for (int t = lid; t < seq_k; t += MEGA_WG) { float e = exp(sc[t] - maxv); sc[t] = e; s += e; }
      red[lid] = s; barrier(CLK_LOCAL_MEM_FENCE);
      for (int o = MEGA_WG >> 1; o > 0; o >>= 1) { if (lid < o) red[lid] += red[lid+o]; barrier(CLK_LOCAL_MEM_FENCE); }
      float inv = 1.0f / (red[0] + 1e-20f); barrier(CLK_LOCAL_MEM_FENCE);
      // out[d] = sum_t p[t]*V_h[t,d]; thread owns d in [0,64)
      if (lid < MEGA_HEAD_DIM) {
        float acc = 0.0f;
        for (int t = 0; t < seq_k; ++t) acc += sc[t] * inv * LOAD(v_cache, t * MEGA_HIDDEN + qb + lid);
        attn_out_l[qb + lid] = acc;
      }
      barrier(CLK_LOCAL_MEM_FENCE);
    }
  }

  // o = out_proj(attn_out_l) -> normed (reuse), then residual += o
  MGEMV(o_w, o_s, attn_out_l, normed, MEGA_HIDDEN, MEGA_HIDDEN);
  for (int c = lid; c < MEGA_HIDDEN; c += MEGA_WG) residual[c] += normed[c];
  barrier(CLK_LOCAL_MEM_FENCE);

  // ══ CROSS-ATTENTION BLOCK ═════════════════════════════════════════════════
  mega_layernorm(residual, normed, enc_ln_w, enc_ln_b, eps, red, lid);
  MGEMV(cq_w, cq_s, normed, qv, MEGA_HIDDEN, MEGA_HIDDEN);   // Q from decoder hidden
  {   // sc aliases big+1024 (carved above); enc_len <= MEGA_MAXENC
    for (int h = 0; h < MEGA_HEADS; ++h) {
      const int qb = h * MEGA_HEAD_DIM;
      for (int t = lid; t < enc_len; t += MEGA_WG) {
        int kb = t * MEGA_HIDDEN + qb;
        float acc = 0.0f;
        for (int d = 0; d < MEGA_HEAD_DIM; ++d) acc += qv[qb + d] * LOAD(k_cross, kb + d);
        sc[t] = acc * scale;
      }
      barrier(CLK_LOCAL_MEM_FENCE);
      float m = -1e30f;
      for (int t = lid; t < enc_len; t += MEGA_WG) m = fmax(m, sc[t]);
      red[lid] = m; barrier(CLK_LOCAL_MEM_FENCE);
      for (int o = MEGA_WG >> 1; o > 0; o >>= 1) { if (lid < o) red[lid] = fmax(red[lid], red[lid+o]); barrier(CLK_LOCAL_MEM_FENCE); }
      float maxv = red[0]; barrier(CLK_LOCAL_MEM_FENCE);
      float s = 0.0f;
      for (int t = lid; t < enc_len; t += MEGA_WG) { float e = exp(sc[t] - maxv); sc[t] = e; s += e; }
      red[lid] = s; barrier(CLK_LOCAL_MEM_FENCE);
      for (int o = MEGA_WG >> 1; o > 0; o >>= 1) { if (lid < o) red[lid] += red[lid+o]; barrier(CLK_LOCAL_MEM_FENCE); }
      float inv = 1.0f / (red[0] + 1e-20f); barrier(CLK_LOCAL_MEM_FENCE);
      if (lid < MEGA_HEAD_DIM) {
        float acc = 0.0f;
        for (int t = 0; t < enc_len; ++t) acc += sc[t] * inv * LOAD(v_cross, t * MEGA_HIDDEN + qb + lid);
        attn_out_l[qb + lid] = acc;
      }
      barrier(CLK_LOCAL_MEM_FENCE);
    }
  }
  MGEMV(co_w, co_s, attn_out_l, normed, MEGA_HIDDEN, MEGA_HIDDEN);
  for (int c = lid; c < MEGA_HIDDEN; c += MEGA_WG) residual[c] += normed[c];
  barrier(CLK_LOCAL_MEM_FENCE);

  // ══ FFN BLOCK ═════════════════════════════════════════════════════════════
  mega_layernorm(residual, normed, fin_ln_w, fin_ln_b, eps, red, lid);
#ifdef MEGA_FFN_EXTERNAL
  // #2a: hand off to the many-WG FFN kernels (fp32, bit-exact). x_out is
  // written by mega_ffn_fc2 (residual + fc2(gelu(fc1(normed)))).
  for (int c = lid; c < MEGA_HIDDEN; c += MEGA_WG) {
    ffn_normed[row * MEGA_HIDDEN + c] = normed[c];
    ffn_resid [row * MEGA_HIDDEN + c] = residual[c];
  }
#else
  // fc1: [4096,1024] @ normed[1024] -> ffn1[4096], then GELU in place.
  MGEMV_FFN(fc1_w, fc1_s, normed, ffn1, MEGA_FFN, MEGA_HIDDEN);
  // GELU — MUST match the existing validated path (kernels/gelu.cl gelu_tanh),
  // which the n=1 incremental-validation cosine compares against. That kernel
  // uses the tanh approximation: 0.5*x*(1+tanh(sqrt(2/pi)*(x+0.044715*x^3))).
  for (int n = lid; n < MEGA_FFN; n += MEGA_WG) {
    float xf = ffn1[n];
    float x3 = xf * xf * xf;
    float tt = 0.7978845608028654f * (xf + 0.044715f * x3);
    ffn1[n] = 0.5f * xf * (1.0f + tanh(tt));
  }
  barrier(CLK_LOCAL_MEM_FENCE);
  // fc2: [1024,4096] @ ffn1[4096] -> normed[1024]; residual += normed
  MGEMV_FFN(fc2_w, fc2_s, ffn1, normed, MEGA_HIDDEN, MEGA_FFN);
  for (int c = lid; c < MEGA_HIDDEN; c += MEGA_WG) residual[c] += normed[c];
  barrier(CLK_LOCAL_MEM_FENCE);

  // store output (fp16)
  for (int c = lid; c < MEGA_HIDDEN; c += MEGA_WG) STORE(x_out, c, residual[c]);
#endif  // MEGA_FFN_EXTERNAL
#undef MGEMV
}

// ── #2a: many-workgroup FFN kernels ──────────────────────────────────────────
// fc1 + GELU: ffn1[row,n] = gelu(fc1_w[n,:] @ ffn_normed[row,:]).
// gws = {(MEGA_FFN/rows_per_wg) * MEGA_MWG_WG, num_rows}, lws = {MEGA_MWG_WG, 1}.
// WG `wg` owns output rows [wg*rows_per_wg, (wg+1)*rows_per_wg); inside, the
// MEGA_MWG_KGROUP-lane row-groups mirror mega_gemv exactly (same lane stride,
// same tree reduce → per-row fp32 sums BIT-IDENTICAL to the in-mega FFN).
// ── CFG-twin row interleave (MEGA_ROW_ILV; host gate NNOPT_ROW_ILV) ─────────
// The (slices, rows) dispatch launches ALL row-0 WGs before any row-1 WG
// (dim0-major): by the time row 1's WG reads weight slice i, the cache lines
// row 0 fetched are long evicted — both CFG rows stream the full matrix from
// DRAM (measured 2026-06-05: 5.2-6.0 GB/s nominal CFG-2 vs 9.5-11.2 single-row,
// ratio ~1.9 = zero dedup). Interleaved form puts ROW on group-dim0: the twin
// WGs (row0/row1 of the same weight slice) launch adjacently, co-reside, and
// the second's fetches hit the texture L1 — the Opt#5 lockstep, restored by
// scheduling instead of relying on drift-free luck. Pure scheduling change:
// per-WG work, lane mapping, and accumulation order are UNTOUCHED, so output
// is byte-identical by construction (guard gate still applies).
#ifdef MEGA_ROW_ILV
  #define MEGA_ROWWG_ROW   ((int)get_group_id(0))
  #define MEGA_ROWWG_SLICE ((int)get_group_id(1))
  #define MEGA_SK_ROW      ((int)get_group_id(0))
  #define MEGA_SK_SLICE    ((int)get_group_id(2))
#else
  #define MEGA_ROWWG_ROW   ((int)get_group_id(1))
  #define MEGA_ROWWG_SLICE ((int)get_group_id(0))
  #define MEGA_SK_ROW      ((int)get_group_id(2))
  #define MEGA_SK_SLICE    ((int)get_group_id(0))
#endif

__kernel __attribute__((reqd_work_group_size(MEGA_MWG_WG, 1, 1)))
void mega_ffn_fc1(
#ifdef MEGA_TEX
                  __read_only image2d_t fc1_w,           // [MEGA_FFN, MEGA_HIDDEN/4 texels]
#else
                  __global const storage_t* fc1_w,      // [MEGA_FFN, MEGA_HIDDEN] fp16
#endif
                  __global const float* ffn_normed,     // [num_rows, MEGA_HIDDEN] fp32
                                                         // (MEGA_FUSE_LN: the RAW
                                                         // ffn_resid — LN3 runs here)
                  __global float* ffn1_g,                // [num_rows, MEGA_FFN] fp32
                  const int rows_per_wg
#ifdef MEGA_FUSE_LN
                  , __global const storage_t* ln_w, __global const storage_t* ln_b
                  , const float eps
#endif
                  ) {
  const int wg   = MEGA_ROWWG_SLICE;
  const int row  = MEGA_ROWWG_ROW;
  const int lid  = get_local_id(0);
  const int grp  = lid / MEGA_MWG_KGROUP;
  const int lane = lid % MEGA_MWG_KGROUP;
  __local float x[MEGA_HIDDEN];
  __local float part[MEGA_MWG_WG];
#ifdef MEGA_H4ACC
  // Stage x as HALF (packed into the same local buffer; reads go through the
  // half view in the H4ACC loop). One convert per element at stage time.
  for (int c = lid; c < MEGA_HIDDEN; c += MEGA_MWG_WG)
    ((__local half*)x)[c] = (half)ffn_normed[row * MEGA_HIDDEN + c];
#else
  for (int c = lid; c < MEGA_HIDDEN; c += MEGA_MWG_WG)
    x[c] = ffn_normed[row * MEGA_HIDDEN + c];
#endif
  barrier(CLK_LOCAL_MEM_FENCE);
#ifdef MEGA_FUSE_LN
  // LN3 fused (kills the standalone mega_ln3 dispatch).
  mwg_ln_local(x, ln_w, ln_b, eps, part, lid);
#endif
  const int n0 = wg * rows_per_wg;
  for (int n = n0 + grp; n < n0 + rows_per_wg; n += MEGA_MWG_ROWS_PAR) {
    float acc = 0.0f;
#if defined(MEGA_TEX) && defined(MEGA_H4ACC)
    // fp16-ALU probe (NNOPT_H4ACC): Adreno 6xx issues half ops at ~2x the fp32
    // rate, and read_imageh already RETURNS half4 — the convert_float4 per
    // texel is pure issue waste. Multiply-accumulate 4 strided texels in a
    // half4 chain (<=4 half-domain adds per lane: products ~O(0.3), partials
    // ~O(1) — well inside half range), upcast ONCE per 16 weights into the
    // fp32 accumulator. ACCUMULATION ORDER + PRECISION CHANGE -> gated on
    // tf-depth cb0>=40 + tinytemp/spectral, NOT bytes.
    {
      // vload_half4/vload4-of-half is BROKEN-slow on Adreno 6xx (documented
      // quirk; first attempt measured 28x SLOWER). Use the proven pattern:
      // raw 64-bit load + as_half4 reinterpret.
      __local const uint2* xh2 = (__local const uint2*)x;   // staged as half below
      #pragma unroll
      for (int k4 = lane; k4 < MEGA_HIDDEN / 4; k4 += 4 * MEGA_MWG_KGROUP) {
        half4 h = read_imageh(fc1_w, mega_smp, (int2)(k4, n)) * as_half4(xh2[k4]);
        h = fma(read_imageh(fc1_w, mega_smp, (int2)(k4 + MEGA_MWG_KGROUP, n)),
                as_half4(xh2[k4 + MEGA_MWG_KGROUP]), h);
        h = fma(read_imageh(fc1_w, mega_smp, (int2)(k4 + 2 * MEGA_MWG_KGROUP, n)),
                as_half4(xh2[k4 + 2 * MEGA_MWG_KGROUP]), h);
        h = fma(read_imageh(fc1_w, mega_smp, (int2)(k4 + 3 * MEGA_MWG_KGROUP, n)),
                as_half4(xh2[k4 + 3 * MEGA_MWG_KGROUP]), h);
        const float4 f = convert_float4(h);
        acc += f.x + f.y + f.z + f.w;
      }
    }
#elif defined(MEGA_TEX) && defined(MEGA_X4LDS)
    // Issue-bound probe (2026-06-05): the fused-rows experiment proved this
    // loop is ALU/local-issue-bound, NOT DRAM-bound (doubling MADs doubled
    // per-WG time with reads constant). Stage x as float4 so the body is
    // 1 vector local load + 4 fma instead of 4 scalar local loads + 4 mad.
    // Scalar accumulation order preserved -> byte-identical.
    #pragma unroll 4
    for (int k4 = lane; k4 < MEGA_HIDDEN / 4; k4 += MEGA_MWG_KGROUP) {
      const float4 wv = convert_float4(read_imageh(fc1_w, mega_smp, (int2)(k4, n)));
      const float4 xv = ((__local const float4*)x)[k4];
      acc += wv.x * xv.x + wv.y * xv.y + wv.z * xv.z + wv.w * xv.w;
    }
#elif defined(MEGA_TEX)
    #pragma unroll 4
    for (int k4 = lane; k4 < MEGA_HIDDEN / 4; k4 += MEGA_MWG_KGROUP) {
      const float4 wv = convert_float4(read_imageh(fc1_w, mega_smp, (int2)(k4, n)));
      const int kb = k4 * 4;
      acc += wv.x * x[kb] + wv.y * x[kb + 1] + wv.z * x[kb + 2] + wv.w * x[kb + 3];
    }
#elif MEGA_MWG_VEC == 4 && defined(USE_FP16)
    {
      __global const uint2* W4 = (__global const uint2*)(fc1_w + n * MEGA_HIDDEN);
      for (int k4 = lane; k4 < MEGA_HIDDEN / 4; k4 += MEGA_MWG_KGROUP) {
        const float4 wv = convert_float4(as_half4(W4[k4]));
        const int kb = k4 * 4;
        acc += wv.x * x[kb] + wv.y * x[kb + 1] + wv.z * x[kb + 2] + wv.w * x[kb + 3];
      }
    }
#else
    for (int k = lane; k < MEGA_HIDDEN; k += MEGA_MWG_KGROUP)
      acc += LOAD(fc1_w, n * MEGA_HIDDEN + k) * x[k];
#endif
    part[lid] = acc;
    barrier(CLK_LOCAL_MEM_FENCE);
    for (int o = MEGA_MWG_KGROUP >> 1; o > 0; o >>= 1) {
      if (lane < o) part[lid] += part[lid + o];
      barrier(CLK_LOCAL_MEM_FENCE);
    }
    if (lane == 0) {
      // GELU (tanh approx) fused into the epilogue — same constants as in-mega.
      float xf = part[grp * MEGA_MWG_KGROUP];
      float x3 = xf * xf * xf;
      float tt = 0.7978845608028654f * (xf + 0.044715f * x3);
      ffn1_g[row * MEGA_FFN + n] = 0.5f * xf * (1.0f + tanh(tt));
    }
    barrier(CLK_LOCAL_MEM_FENCE);
  }
}

#ifdef MEGA_TEX
// ── TRUE CFG-row fusion, fc1 (NNOPT_ROW_FUSE; dispatched only when num_rows==2)
// One WG stages BOTH rows' x (8 KB local) and reads each weight texel ONCE,
// accumulating into acc0/acc1 — weight traffic halves BY CONSTRUCTION instead
// of by cache luck (ROW_ILV measured only ~1.6x-of-single-row; twins can land
// on different SPs whose texture L1s don't share). Register cost: +1 fp32
// accumulator; local cost: +4 KB x1 + 1 KB part1 (total ~10 KB, >=3 WGs/SP).
// Per-row MAD and reduce order is IDENTICAL to mega_ffn_fc1 (acc0 chain ==
// unfused acc chain; part/part1 trees reduce in the same order) -> output is
// byte-identical per row; the greedy guard gate applies unchanged.
// (Unlike the old FC2_FUSE2 buffer variant — 16 in-flight accumulators, spill —
// this keeps the n-loop's single-accumulator structure.)
__kernel __attribute__((reqd_work_group_size(MEGA_MWG_WG, 1, 1)))
void mega_ffn_fc1_rf(__read_only image2d_t fc1_w,        // [MEGA_FFN, MEGA_HIDDEN/4]
                     __global const float* ffn_normed,   // [2, MEGA_HIDDEN] fp32
                     __global float* ffn1_g,             // [2, MEGA_FFN] fp32
                     const int rows_per_wg) {
  const int wg   = (int)get_group_id(0);
  const int lid  = get_local_id(0);
  const int grp  = lid / MEGA_MWG_KGROUP;
  const int lane = lid % MEGA_MWG_KGROUP;
  __local float x0[MEGA_HIDDEN];
  __local float x1[MEGA_HIDDEN];
  __local float part[MEGA_MWG_WG];
  __local float part1[MEGA_MWG_WG];
  for (int c = lid; c < MEGA_HIDDEN; c += MEGA_MWG_WG) {
    x0[c] = ffn_normed[c];
    x1[c] = ffn_normed[MEGA_HIDDEN + c];
  }
  barrier(CLK_LOCAL_MEM_FENCE);
  const int n0 = wg * rows_per_wg;
  for (int n = n0 + grp; n < n0 + rows_per_wg; n += MEGA_MWG_ROWS_PAR) {
    float acc0 = 0.0f, acc1 = 0.0f;
    #pragma unroll 4
    for (int k4 = lane; k4 < MEGA_HIDDEN / 4; k4 += MEGA_MWG_KGROUP) {
      const float4 wv = convert_float4(read_imageh(fc1_w, mega_smp, (int2)(k4, n)));
      const int kb = k4 * 4;
      acc0 += wv.x * x0[kb] + wv.y * x0[kb + 1] + wv.z * x0[kb + 2] + wv.w * x0[kb + 3];
      acc1 += wv.x * x1[kb] + wv.y * x1[kb + 1] + wv.z * x1[kb + 2] + wv.w * x1[kb + 3];
    }
    part[lid] = acc0; part1[lid] = acc1;
    barrier(CLK_LOCAL_MEM_FENCE);
    for (int o = MEGA_MWG_KGROUP >> 1; o > 0; o >>= 1) {
      if (lane < o) { part[lid] += part[lid + o]; part1[lid] += part1[lid + o]; }
      barrier(CLK_LOCAL_MEM_FENCE);
    }
    if (lane == 0) {
      float xf = part[grp * MEGA_MWG_KGROUP];
      float x3 = xf * xf * xf;
      float tt = 0.7978845608028654f * (xf + 0.044715f * x3);
      ffn1_g[n] = 0.5f * xf * (1.0f + tanh(tt));
      float yf = part1[grp * MEGA_MWG_KGROUP];
      float y3 = yf * yf * yf;
      float tu = 0.7978845608028654f * (yf + 0.044715f * y3);
      ffn1_g[MEGA_FFN + n] = 0.5f * yf * (1.0f + tanh(tu));
    }
    barrier(CLK_LOCAL_MEM_FENCE);
  }
}
#endif  // MEGA_TEX (mega_ffn_fc1_rf)

// fc2 + residual + fp16 store: x_out[row][n] = fp16(ffn_resid[row,n] +
// fc2_w[n,:] @ ffn1_g[row,:]). The 16 KB fc2 input stays in GLOBAL memory
// (a 16 KB __local copy per WG would cap residency at 2 WGs/SP on the 32 KB
// Adreno local store — the exact occupancy wall this kernel exists to break);
// it is shared read-only across all WGs in flight so it serves from cache.
// fc2 stages its 16 KB fp32 input through 4 KB __local chunks: a direct global
// read had every WG in flight re-streaming the same 16 KB (measured 1.1-2.4
// GB/s); the chunked form loads each tile cooperatively once per WG and keeps
// per-WG local usage at 4.3 KB (≥7 WGs resident). Chunk iteration is ascending
// in k, so each lane's accumulation order is IDENTICAL to the unchunked loop.
#define MEGA_MWG_CHUNK 1024
#define MEGA_MWG_MAX_ITERS 16     // rows_per_wg / ROWS_PAR ≤ 16 (host clamps)
// fc2's texture path is gated SEPARATELY (MEGA_TEX_FC2): measured 2026-06-04,
// chunked-tex fc2 = 2.19 GB/s vs 2.46 chunked-buffer — fc2's wide-K (4096)
// rows don't benefit from the tex cache the way attn/fc1 do, so the default
// mixed config keeps fc2 on buffers while attn+fc1 ride the texture pipe.
#ifdef MEGA_FC2_FUSE2
// ── CFG-ROW-FUSED fc2 (MEGA_FC2_FUSE2) ───────────────────────────────────────
// The (nwg, 2) dispatch makes BOTH row-WGs stream the same 8 MB weight matrix;
// fc1's row pairs stay in lockstep so L2 serves the twin (4.35 GB/s nominal ≈
// bus peak actual), but fc2's chunk barriers let its pairs DRIFT → L2 misses →
// each row streams from DRAM → the stuck ~2.2 GB/s nominal. This variant runs
// (nwg, 1): each weight vector is fetched ONCE and mad'd into BOTH rows'
// accumulators — weight DRAM traffic halves by construction. Per-row per-lane
// accumulation order is IDENTICAL to the unfused kernel (tf-gate-preserving).
// num_rows==1 generations skip the row-1 accumulate/writeback (uniform branch).
__kernel __attribute__((reqd_work_group_size(MEGA_MWG_WG, 1, 1)))
void mega_ffn_fc2(__global const storage_t* fc2_w,      // [MEGA_HIDDEN, MEGA_FFN] fp16
                  __global const float* ffn1_g,          // [num_rows, MEGA_FFN] fp32
                  __global const float* ffn_resid,       // [num_rows, MEGA_HIDDEN] fp32
                  __global storage_t* x_out0,            // row0 layer output (fp16)
                  __global storage_t* x_out1,            // row1 layer output (fp16)
                  const int rows_per_wg,
                  const int num_rows) {
  const int wg   = get_group_id(0);
  const int lid  = get_local_id(0);
  const int grp  = lid / MEGA_MWG_KGROUP;
  const int lane = lid % MEGA_MWG_KGROUP;
  __global const float* x0 = ffn1_g;
  __global const float* x1 = ffn1_g + MEGA_FFN;
  __local float xs0[MEGA_MWG_CHUNK];
  __local float xs1[MEGA_MWG_CHUNK];
  __local float part[MEGA_MWG_WG];
  const int n0 = wg * rows_per_wg;
#ifndef MEGA_FC2_RPW
  #define MEGA_FC2_RPW 32
#endif
  const int n_iters = MEGA_FC2_RPW / MEGA_MWG_ROWS_PAR;
  float acc0[MEGA_FC2_RPW / MEGA_MWG_ROWS_PAR];
  float acc1[MEGA_FC2_RPW / MEGA_MWG_ROWS_PAR];
  for (int it = 0; it < n_iters; ++it) { acc0[it] = 0.0f; acc1[it] = 0.0f; }
  const int two = (num_rows == 2);
  for (int kc = 0; kc < MEGA_FFN; kc += MEGA_MWG_CHUNK) {
    for (int c = lid; c < MEGA_MWG_CHUNK; c += MEGA_MWG_WG) {
      xs0[c] = x0[kc + c];
      if (two) xs1[c] = x1[kc + c];
    }
    barrier(CLK_LOCAL_MEM_FENCE);
    for (int it = 0; it < n_iters; ++it) {
      const int n = n0 + grp + it * MEGA_MWG_ROWS_PAR;
#if MEGA_MWG_VEC == 4 && defined(USE_FP16)
      __global const uint2* W4 = (__global const uint2*)(fc2_w + n * MEGA_FFN + kc);
      for (int k4 = lane; k4 < MEGA_MWG_CHUNK / 4; k4 += MEGA_MWG_KGROUP) {
        const float4 wv = convert_float4(as_half4(W4[k4]));
        const int kb = k4 * 4;
        acc0[it] += wv.x * xs0[kb] + wv.y * xs0[kb + 1] + wv.z * xs0[kb + 2] + wv.w * xs0[kb + 3];
        if (two) acc1[it] += wv.x * xs1[kb] + wv.y * xs1[kb + 1] + wv.z * xs1[kb + 2] + wv.w * xs1[kb + 3];
      }
#else
      for (int k = lane; k < MEGA_MWG_CHUNK; k += MEGA_MWG_KGROUP) {
        const float w = LOAD(fc2_w, n * MEGA_FFN + kc + k);
        acc0[it] += w * xs0[k];
        if (two) acc1[it] += w * xs1[k];
      }
#endif
    }
    barrier(CLK_LOCAL_MEM_FENCE);
  }
  for (int it = 0; it < n_iters; ++it) {
    const int n = n0 + grp + it * MEGA_MWG_ROWS_PAR;
    part[lid] = acc0[it];
    barrier(CLK_LOCAL_MEM_FENCE);
    for (int o = MEGA_MWG_KGROUP >> 1; o > 0; o >>= 1) {
      if (lane < o) part[lid] += part[lid + o];
      barrier(CLK_LOCAL_MEM_FENCE);
    }
    if (lane == 0) {
      float out = ffn_resid[n] + part[grp * MEGA_MWG_KGROUP];
      STORE(x_out0, n, out);
    }
    barrier(CLK_LOCAL_MEM_FENCE);
    if (two) {
      part[lid] = acc1[it];
      barrier(CLK_LOCAL_MEM_FENCE);
      for (int o = MEGA_MWG_KGROUP >> 1; o > 0; o >>= 1) {
        if (lane < o) part[lid] += part[lid + o];
        barrier(CLK_LOCAL_MEM_FENCE);
      }
      if (lane == 0) {
        float out = ffn_resid[MEGA_HIDDEN + n] + part[grp * MEGA_MWG_KGROUP];
        STORE(x_out1, n, out);
      }
      barrier(CLK_LOCAL_MEM_FENCE);
    }
  }
}
#else  // !MEGA_FC2_FUSE2 — original per-row dispatch
__kernel __attribute__((reqd_work_group_size(MEGA_MWG_WG, 1, 1)))
void mega_ffn_fc2(
#ifdef MEGA_TEX_FC2
                  __read_only image2d_t fc2_w,           // [MEGA_HIDDEN, MEGA_FFN/4 texels]
#else
                  __global const storage_t* fc2_w,      // [MEGA_HIDDEN, MEGA_FFN] fp16
#endif
                  __global const float* ffn1_g,          // [num_rows, MEGA_FFN] fp32
                  __global const float* ffn_resid,       // [num_rows, MEGA_HIDDEN] fp32
                  __global storage_t* x_out0,            // row0 layer output (fp16)
                  __global storage_t* x_out1,            // row1 layer output (fp16)
                  const int rows_per_wg) {
  const int wg   = get_group_id(0);
  const int row  = get_group_id(1);
  const int lid  = get_local_id(0);
  const int grp  = lid / MEGA_MWG_KGROUP;
  const int lane = lid % MEGA_MWG_KGROUP;
  __global storage_t* x_out = (row == 0) ? x_out0 : x_out1;
  __global const float* x = ffn1_g + (size_t)row * MEGA_FFN;
  __local float part[MEGA_MWG_WG];
  const int n0 = wg * rows_per_wg;
  // CHUNKED in BOTH paths. The unchunked tex variant ("texture pipe frees the
  // load/store pipe for global x reads") was MEASURED 2026-06-04: 1.25 GB/s vs
  // 2.46 chunked-buffer — every WG re-streaming the same 16 KB fp32 input loses
  // 2× regardless of which pipe the weights ride. The __local staging stays.
  //
  // SINGLE-buffer, NOT double-buffered: ping-pong xs[2] (prefetch chunk c+1
  // during compute of c, one barrier/chunk) was MEASURED 2026-06-04 at
  // 1.66 GB/s vs 2.17 single — the extra 4 KB local halves WG residency
  // (7→3 WGs/SP on the 32 KB store) and on Adreno latency hiding comes from
  // resident WGs, not intra-WG prefetch. Do not re-try.
  __local float xs[MEGA_MWG_CHUNK];
  // COMPILE-TIME iteration count (MEGA_FC2_RPW, default 32): with a runtime
  // n_iters the acc[] array is unrollable-proof and gets spilled to scratch
  // memory; a constant count lets the compiler unroll and keep every
  // accumulator in registers. The host passes rows_per_wg == MEGA_FC2_RPW.
#ifndef MEGA_FC2_RPW
  #define MEGA_FC2_RPW 32
#endif
  const int n_iters = MEGA_FC2_RPW / MEGA_MWG_ROWS_PAR;
  float acc[MEGA_FC2_RPW / MEGA_MWG_ROWS_PAR];
  for (int it = 0; it < n_iters; ++it) acc[it] = 0.0f;
  for (int kc = 0; kc < MEGA_FFN; kc += MEGA_MWG_CHUNK) {
    for (int c = lid; c < MEGA_MWG_CHUNK; c += MEGA_MWG_WG) xs[c] = x[kc + c];
    barrier(CLK_LOCAL_MEM_FENCE);
    for (int it = 0; it < n_iters; ++it) {
      const int n = n0 + grp + it * MEGA_MWG_ROWS_PAR;
#ifdef MEGA_TEX_FC2
      const int xc = kc >> 2;   // chunk origin in texels
      #pragma unroll 4
      for (int k4 = lane; k4 < MEGA_MWG_CHUNK / 4; k4 += MEGA_MWG_KGROUP) {
        const float4 wv = convert_float4(read_imageh(fc2_w, mega_smp, (int2)(xc + k4, n)));
        const int kb = k4 * 4;
        acc[it] += wv.x * xs[kb] + wv.y * xs[kb + 1] + wv.z * xs[kb + 2] + wv.w * xs[kb + 3];
      }
#elif MEGA_MWG_VEC == 4 && defined(USE_FP16)
      __global const uint2* W4 = (__global const uint2*)(fc2_w + n * MEGA_FFN + kc);
      for (int k4 = lane; k4 < MEGA_MWG_CHUNK / 4; k4 += MEGA_MWG_KGROUP) {
        const float4 wv = convert_float4(as_half4(W4[k4]));
        const int kb = k4 * 4;
        acc[it] += wv.x * xs[kb] + wv.y * xs[kb + 1] + wv.z * xs[kb + 2] + wv.w * xs[kb + 3];
      }
#else
      for (int k = lane; k < MEGA_MWG_CHUNK; k += MEGA_MWG_KGROUP)
        acc[it] += LOAD(fc2_w, n * MEGA_FFN + kc + k) * xs[k];
#endif
    }
    barrier(CLK_LOCAL_MEM_FENCE);
  }
  for (int it = 0; it < n_iters; ++it) {
    const int n = n0 + grp + it * MEGA_MWG_ROWS_PAR;
    part[lid] = acc[it];
    barrier(CLK_LOCAL_MEM_FENCE);
    for (int o = MEGA_MWG_KGROUP >> 1; o > 0; o >>= 1) {
      if (lane < o) part[lid] += part[lid + o];
      barrier(CLK_LOCAL_MEM_FENCE);
    }
    if (lane == 0) {
      float out = ffn_resid[row * MEGA_HIDDEN + n] + part[grp * MEGA_MWG_KGROUP];
      STORE(x_out, n, out);
    }
    barrier(CLK_LOCAL_MEM_FENCE);
  }
}
#endif  // MEGA_FC2_FUSE2

#if defined(MEGA_QKV_EXTERNAL) && defined(MEGA_TEX)
// ── Stage-1 attn split: external LN1 + QKV (MEGA_QKV_EXTERNAL) ───────────────
// The in-mega q/k/v GEMVs are capped at 3.5 GB/s by the megakernel's register
// ceiling (no unroll possible — measured 10× crater). Standalone they get
// multi-WG parallelism + unroll-4 + the texture-L1 CFG-row dedup (fc1: 6.2
// GB/s nominal). mega_ln_rows computes LN1 (same mega_layernorm → bit-identical
// to in-mega); mega_qkv runs one GEMV over the PACKED [3H, H/4] qkv image
// (rows [0,H)=q → q_ext, [H,2H)=k → k_cache@start_pos, [2H,3H)=v → v_cache).
// Row/lane mapping + tree reduce mirror mega_gemv_tex exactly (KGROUP=16) →
// per-row sums BIT-IDENTICAL to the in-mega path.
__kernel __attribute__((reqd_work_group_size(MEGA_WG, 1, 1)))
void mega_ln_rows(__global const storage_t* x_in0, __global const storage_t* x_in1,
                  __global const storage_t* ln_w, __global const storage_t* ln_b,
                  __global float* normed_g,        // [num_rows, hidden] fp32
                  const float eps) {
  const int row = get_group_id(0);
  const int lid = get_local_id(0);
  __global const storage_t* x_in = (row == 0) ? x_in0 : x_in1;
  __local float xl[MEGA_HIDDEN];
  __local float nl[MEGA_HIDDEN];
  __local float red[MEGA_WG];
  for (int c = lid; c < MEGA_HIDDEN; c += MEGA_WG) xl[c] = LOAD(x_in, c);
  barrier(CLK_LOCAL_MEM_FENCE);
  mega_layernorm(xl, nl, ln_w, ln_b, eps, red, lid);
  for (int c = lid; c < MEGA_HIDDEN; c += MEGA_WG)
    normed_g[row * MEGA_HIDDEN + c] = nl[c];
}

__kernel __attribute__((reqd_work_group_size(MEGA_MWG_WG, 1, 1)))
void mega_qkv(__read_only image2d_t qkv_w,         // [3H, H/4 texels] packed q,k,v
#ifdef MEGA_FUSE_LN
              // LN1 fused: stage the RAW residual row and normalize in local
              // (kills the standalone mega_ln_rows dispatch).
              __global const storage_t* x_in0,     // row0 residual (fp16)
#else
              __global const float* normed_g,      // [num_rows, hidden] fp32
#endif
              __global float* q_ext,               // [num_rows, hidden] fp32
              __global storage_t* k_cache0, __global storage_t* v_cache0,
              __global storage_t* k_cache1, __global storage_t* v_cache1,
              // step-params buffer: sp[0] = start_pos. Was a literal int arg —
              // moved into a buffer so the recordable-queue replay needs NO
              // per-step kernel-arg override (this driver rejects them, -59).
              __global const int* sp, const int rows_per_wg
#ifdef MEGA_FUSE_LN
              , __global const storage_t* x_in1    // row1 residual (fp16)
              , __global const storage_t* ln_w, __global const storage_t* ln_b
              , const float eps
#endif
              ) {
  const int start_pos = sp[0];
  const int wg   = MEGA_ROWWG_SLICE;
  const int row  = MEGA_ROWWG_ROW;
  const int lid  = get_local_id(0);
  const int grp  = lid / MEGA_MWG_KGROUP;
  const int lane = lid % MEGA_MWG_KGROUP;
  __global storage_t* k_cache = (row == 0) ? k_cache0 : k_cache1;
  __global storage_t* v_cache = (row == 0) ? v_cache0 : v_cache1;
  __local float xs[MEGA_HIDDEN];
  __local float part[MEGA_MWG_WG];
#ifdef MEGA_FUSE_LN
  {
    __global const storage_t* xin = (row == 0) ? x_in0 : x_in1;
    for (int c = lid; c < MEGA_HIDDEN; c += MEGA_MWG_WG)
      xs[c] = LOAD(xin, c);
    barrier(CLK_LOCAL_MEM_FENCE);
    mwg_ln_local(xs, ln_w, ln_b, eps, part, lid);
  }
#else
  for (int c = lid; c < MEGA_HIDDEN; c += MEGA_MWG_WG)
    xs[c] = normed_g[row * MEGA_HIDDEN + c];
  barrier(CLK_LOCAL_MEM_FENCE);
#endif
  const int n0 = wg * rows_per_wg;
  for (int n = n0 + grp; n < n0 + rows_per_wg; n += MEGA_MWG_ROWS_PAR) {
    float acc = 0.0f;
    #pragma unroll 4
    for (int k4 = lane; k4 < MEGA_HIDDEN / 4; k4 += MEGA_MWG_KGROUP) {
      const float4 wv = convert_float4(read_imageh(qkv_w, mega_smp, (int2)(k4, n)));
      const int kb = k4 * 4;
      acc += wv.x * xs[kb] + wv.y * xs[kb + 1] + wv.z * xs[kb + 2] + wv.w * xs[kb + 3];
    }
    part[lid] = acc;
    barrier(CLK_LOCAL_MEM_FENCE);
    for (int o = MEGA_MWG_KGROUP >> 1; o > 0; o >>= 1) {
      if (lane < o) part[lid] += part[lid + o];
      barrier(CLK_LOCAL_MEM_FENCE);
    }
    if (lane == 0) {
      const float sum = part[grp * MEGA_MWG_KGROUP];
      const int h   = n % MEGA_HIDDEN;   // position within the projection
      const int sel = n / MEGA_HIDDEN;   // 0=q 1=k 2=v
      if (sel == 0)      q_ext[row * MEGA_HIDDEN + h] = sum;
      else if (sel == 1) STORE(k_cache, start_pos * MEGA_HIDDEN + h, sum);
      else               STORE(v_cache, start_pos * MEGA_HIDDEN + h, sum);
    }
    barrier(CLK_LOCAL_MEM_FENCE);
  }
}
// ── fc2 SPLIT-K (NNOPT_FC2_SPLITK): give fc2 the fc1 shape ───────────────────
// fc2 is the last kernel stuck at ~2.4 GB/s. fc1's winning shape: stage the
// WHOLE input once (4 KB local), short 256-texel weight rows, no chunk
// barriers. Split-K gives fc2 exactly that: each WG owns (row-block, one of
// MEGA_FC2_KSEG K-segments) — stages its 4 KB x-segment once, streams 256
// texels/row, writes per-segment partial sums. A tiny reduce kernel then sums
// resid + the KSEG partials per output element.
// NOTE: this CHANGES the per-row reduction order (segment partials summed
// sequentially vs one lane-stride across full K) → NOT bit-identical; the
// tf-depth gate decides (precedent: MWG_VEC=4's 4-block reorder passed at 47).
#if defined(MEGA_TEX) && defined(MEGA_TEX_FC2)
#ifndef MEGA_FC2_KSEG
  #define MEGA_FC2_KSEG 4
#endif
#define MEGA_FC2_SEG (MEGA_FFN / MEGA_FC2_KSEG)
__kernel __attribute__((reqd_work_group_size(MEGA_MWG_WG, 1, 1)))
void mega_ffn_fc2_sk(__read_only image2d_t fc2_w,      // [H, FFN/4 texels]
                     __global const float* ffn1_g,      // [rows, FFN] fp32
                     __global float* partials,          // [rows, KSEG, H] fp32
                     const int rows_per_wg) {
  const int wg  = MEGA_SK_SLICE;
  const int seg = (int)get_group_id(1);
  const int row = MEGA_SK_ROW;
  const int lid = get_local_id(0);
  const int grp = lid / MEGA_MWG_KGROUP;
  const int lane = lid % MEGA_MWG_KGROUP;
  __local float xs[MEGA_FC2_SEG];
  __local float part[MEGA_MWG_WG];
  __global const float* x = ffn1_g + (size_t)row * MEGA_FFN + seg * MEGA_FC2_SEG;
  for (int c = lid; c < MEGA_FC2_SEG; c += MEGA_MWG_WG) xs[c] = x[c];
  barrier(CLK_LOCAL_MEM_FENCE);
  const int n0 = wg * rows_per_wg;
  const int t0 = seg * (MEGA_FC2_SEG / 4);   // texel origin of this K-segment
  for (int n = n0 + grp; n < n0 + rows_per_wg; n += MEGA_MWG_ROWS_PAR) {
    float acc = 0.0f;
    #pragma unroll 4
    for (int k4 = lane; k4 < MEGA_FC2_SEG / 4; k4 += MEGA_MWG_KGROUP) {
      const float4 wv = convert_float4(read_imageh(fc2_w, mega_smp, (int2)(t0 + k4, n)));
      const int kb = k4 * 4;
      acc += wv.x * xs[kb] + wv.y * xs[kb + 1] + wv.z * xs[kb + 2] + wv.w * xs[kb + 3];
    }
    part[lid] = acc;
    barrier(CLK_LOCAL_MEM_FENCE);
    for (int o = MEGA_MWG_KGROUP >> 1; o > 0; o >>= 1) {
      if (lane < o) part[lid] += part[lid + o];
      barrier(CLK_LOCAL_MEM_FENCE);
    }
    if (lane == 0)
      partials[((size_t)row * MEGA_FC2_KSEG + seg) * MEGA_HIDDEN + n] = part[grp * MEGA_MWG_KGROUP];
    barrier(CLK_LOCAL_MEM_FENCE);
  }
}

// out[row,n] = resid[row,n] + Σ_seg partials[row,seg,n]  (seg-ascending order)
__kernel void mega_ffn_fc2_sk_reduce(__global const float* partials,
                                     __global const float* ffn_resid,
                                     __global storage_t* x_out0,
                                     __global storage_t* x_out1) {
  const int n = get_global_id(0);
  const int row = get_global_id(1);
  if (n >= MEGA_HIDDEN) return;
  __global storage_t* x_out = (row == 0) ? x_out0 : x_out1;
  float acc = ffn_resid[(size_t)row * MEGA_HIDDEN + n];
  for (int s = 0; s < MEGA_FC2_KSEG; ++s)
    acc += partials[((size_t)row * MEGA_FC2_KSEG + s) * MEGA_HIDDEN + n];
  STORE(x_out, n, acc);
}
#endif  // MEGA_TEX && MEGA_TEX_FC2

// WG size for the head-parallel attention core (64 = one lane per head dim).
#ifndef MEGA_ATTN_WG
  #define MEGA_ATTN_WG 64
#endif

// ── Stage-2 attn split (NNOPT_ATTN_SPLIT): the megakernel dissolves ──────────
// decoder_layer_mega is NOT dispatched at all for split layers. Per layer:
//   ln_rows → qkv → attn_core(self) → proj(o,+x_in) → ln_f32 → proj(cq) →
//   attn_core(cross) → proj(co,+resid) → ln_f32 → fc1 → fc2.
// Every kernel gets its own register budget: the o/cq/co GEMVs ride the same
// multi-WG + texture + unroll-4 path that put qkv at 5.7 and fc1 at 6.2 GB/s.
// All loop strides / WG sizes / reduce trees MIRROR the megakernel exactly →
// per-row sums and softmax order BIT-IDENTICAL to the fused path.

// Generic single-query attention core (used for BOTH self and cross attention).
// q_g fp32 [num_rows, hidden]; keys/values storage_t token-major [seq, hidden]
// (self: the KV caches; cross: the precomputed cross K/V).
//
// HEAD-PARALLEL (2026-06-05): one WG per (head, row) — 32 WGs instead of 2.
// The 2-WG head-serial form (megakernel heritage) became the #1 decode kernel
// over a full 250-step clip (27.7%, 14.9 s): self-attn cost grows O(t) with
// the KV scan and the 2-WG dispatch starves the GPU (16 serial heads × ~10
// barriers each). This form has no head loop and reduces over 64 lanes.
// NOTE: 64-lane softmax reductions ≠ the old 512-lane order → NOT bit-identical
// — gated on tf-depth (cb0 ≥ 40).
__kernel __attribute__((reqd_work_group_size(MEGA_ATTN_WG, 1, 1)))
void mega_attn_core(__global const float* q_g,
#ifdef MEGA_ATTN_TEX
                    // K via image2d VIEW over the cache buffer ([rows, H/4]
                    // RGBA half texels, zero-copy — the lm_heads recipe).
                    // Texture-L1 serves the t-strided K rows that the buffer
                    // path read at ~1.1 GB/s effective. V stays a buffer (its
                    // lane-coalesced column reads were never the problem).
                    __read_only image2d_t k0, __global const storage_t* v0,
                    __read_only image2d_t k1, __global const storage_t* v1,
#else
                    __global const storage_t* k0, __global const storage_t* v0,
                    __global const storage_t* k1, __global const storage_t* v1,
#endif
                    __global float* out_g,        // [num_rows, hidden] fp32
                    // seq length via buffer: seq_len = seq_ptr[0] + seq_bias.
                    // Self-attn: seq_ptr = step-params (start_pos), bias = 1.
                    // Cross-attn: seq_ptr = enc-len buffer, bias = 0.
                    // Was a literal int arg — buffered so the recordable-queue
                    // replay needs NO per-step kernel-arg override (driver
                    // rejects them with -59).
                    __global const int* seq_ptr, const int seq_bias,
                    const float scale) {
  const int seq_len = seq_ptr[0] + seq_bias;
  const int h   = get_group_id(0);    // head
  const int row = get_group_id(1);    // CFG row
  const int lid = get_local_id(0);
  __global const storage_t* V = (row == 0) ? v0 : v1;
  const int qb = h * MEGA_HEAD_DIM;
  __local float qh[MEGA_HEAD_DIM];
  __local float sc[MEGA_MAXK];          // >= MEGA_MAXENC (256 >= 64)
  __local float red[MEGA_ATTN_WG];
  if (lid < MEGA_HEAD_DIM) qh[lid] = q_g[row * MEGA_HIDDEN + qb + lid];
  barrier(CLK_LOCAL_MEM_FENCE);
  // scores[t] = scale * dot(q_h, K_h[t]) — VECTORIZED: the original 64 scalar
  // vload_half per (lane, t) ran the kernel at ~1.1 GB/s effective (574 µs avg,
  // 11.6% of decode) — pure issue-rate bound, not bandwidth.
  // NOTE: vector tree summation ≠ the old serial-d order → NOT bit-identical;
  // gated on tf-depth (cb0 ≥ 40) + greedy guard. (vec pass: all 4 cb ≥ 40.)
#ifdef MEGA_ATTN_TEX
  // Images can't be selected through a pointer — expand the loop per row via
  // macro (row is uniform per WG; one branch, no divergence). Two texels per
  // iteration form the SAME float8 grouping/order as the buffer path's
  // vload_half8 → bit-identical math to the gated vectorized-buffer kernel.
  #define MEGA_ATTN_SCORES(KIMG)                                              \
    for (int t = lid; t < seq_len; t += MEGA_ATTN_WG) {                       \
      float8 acc8 = (float8)(0.0f);                                           \
      _Pragma("unroll")                                                       \
      for (int d8 = 0; d8 < MEGA_HEAD_DIM / 8; ++d8) {                        \
        const float8 kv = (float8)(                                           \
          convert_float4(read_imageh(KIMG, mega_smp, (int2)(qb/4 + 2*d8,     t))), \
          convert_float4(read_imageh(KIMG, mega_smp, (int2)(qb/4 + 2*d8 + 1, t)))); \
        acc8 += kv * vload8(d8, qh);                                          \
      }                                                                       \
      const float4 a4t = acc8.lo + acc8.hi;                                   \
      const float2 a2t = a4t.lo + a4t.hi;                                     \
      sc[t] = (a2t.x + a2t.y) * scale;                                        \
    }
  if (row == 0) { MEGA_ATTN_SCORES(k0) } else { MEGA_ATTN_SCORES(k1) }
  #undef MEGA_ATTN_SCORES
#else
  __global const storage_t* K = (row == 0) ? k0 : k1;
  for (int t = lid; t < seq_len; t += MEGA_ATTN_WG) {
    const __global storage_t* Kt = K + (size_t)t * MEGA_HIDDEN + qb;
    float8 acc8 = (float8)(0.0f);
    #pragma unroll
    for (int d8 = 0; d8 < MEGA_HEAD_DIM / 8; ++d8)
      acc8 += LOADV8(Kt, d8) * vload8(d8, qh);
    const float4 a4 = acc8.lo + acc8.hi;
    const float2 a2 = a4.lo + a4.hi;
    sc[t] = (a2.x + a2.y) * scale;
  }
#endif
  barrier(CLK_LOCAL_MEM_FENCE);
  float m = -1e30f;
  for (int t = lid; t < seq_len; t += MEGA_ATTN_WG) m = fmax(m, sc[t]);
  red[lid] = m; barrier(CLK_LOCAL_MEM_FENCE);
  for (int o = MEGA_ATTN_WG >> 1; o > 0; o >>= 1) { if (lid < o) red[lid] = fmax(red[lid], red[lid+o]); barrier(CLK_LOCAL_MEM_FENCE); }
  const float maxv = red[0]; barrier(CLK_LOCAL_MEM_FENCE);
  float s = 0.0f;
  for (int t = lid; t < seq_len; t += MEGA_ATTN_WG) { float e = exp(sc[t] - maxv); sc[t] = e; s += e; }
  red[lid] = s; barrier(CLK_LOCAL_MEM_FENCE);
  for (int o = MEGA_ATTN_WG >> 1; o > 0; o >>= 1) { if (lid < o) red[lid] += red[lid+o]; barrier(CLK_LOCAL_MEM_FENCE); }
  const float inv = 1.0f / (red[0] + 1e-20f); barrier(CLK_LOCAL_MEM_FENCE);
  // out[d] = Σ_t p[t]·V_h[t,d] — lane d owns output dim d (coalesced V reads).
  // 4-way t-unroll (independent accumulators → 4 loads in flight) + inv
  // hoisted out of the loop. Accumulation order changes — same gates as above.
  if (lid < MEGA_HEAD_DIM) {
    const __global storage_t* Vc = V + qb + lid;
    float a0 = 0.0f, a1 = 0.0f, a2v = 0.0f, a3 = 0.0f;
    int t = 0;
    for (; t + 4 <= seq_len; t += 4) {
      a0  += sc[t]     * LOAD(Vc, (size_t)(t)     * MEGA_HIDDEN);
      a1  += sc[t + 1] * LOAD(Vc, (size_t)(t + 1) * MEGA_HIDDEN);
      a2v += sc[t + 2] * LOAD(Vc, (size_t)(t + 2) * MEGA_HIDDEN);
      a3  += sc[t + 3] * LOAD(Vc, (size_t)(t + 3) * MEGA_HIDDEN);
    }
    for (; t < seq_len; ++t) a0 += sc[t] * LOAD(Vc, (size_t)t * MEGA_HIDDEN);
    out_g[row * MEGA_HIDDEN + qb + lid] = ((a0 + a1) + (a2v + a3)) * inv;
  }
}

// Generic [H,H] projection GEMV over a texture weight, multi-WG + unroll-4
// (the fc1/qkv recipe). Epilogue by mode: 0 = y = Wx (cq), 1 = y = x_in + Wx
// (o-proj; x_in fp16 from the layer input), 2 = y = resid_f32 + Wx (co-proj).
__kernel __attribute__((reqd_work_group_size(MEGA_MWG_WG, 1, 1)))
void mega_proj(__read_only image2d_t W,            // [H, H/4 texels]
               __global const float* x_g,           // [num_rows, hidden] fp32
               __global float* y_g,                 // [num_rows, hidden] fp32
               __global const storage_t* xin0,      // mode 1 residual (row 0)
               __global const storage_t* xin1,      // mode 1 residual (row 1)
               __global const float* resid_g,       // mode 2 residual
               const int mode, const int rows_per_wg
#ifdef MEGA_FUSE_LN
               // do_ln=1 (cq): x_g is the RAW resid2 row — LayerNorm it in
               // local after staging (kills the standalone mega_ln2 dispatch).
               , __global const storage_t* ln_w, __global const storage_t* ln_b
               , const float eps_ln, const int do_ln
#endif
               ) {
  const int wg   = MEGA_ROWWG_SLICE;
  const int row  = MEGA_ROWWG_ROW;
  const int lid  = get_local_id(0);
  const int grp  = lid / MEGA_MWG_KGROUP;
  const int lane = lid % MEGA_MWG_KGROUP;
  __global const storage_t* xin = (row == 0) ? xin0 : xin1;
  __local float xs[MEGA_HIDDEN];
  __local float part[MEGA_MWG_WG];
  for (int c = lid; c < MEGA_HIDDEN; c += MEGA_MWG_WG)
    xs[c] = x_g[row * MEGA_HIDDEN + c];
  barrier(CLK_LOCAL_MEM_FENCE);
#ifdef MEGA_FUSE_LN
  if (do_ln) mwg_ln_local(xs, ln_w, ln_b, eps_ln, part, lid);
#endif
  const int n0 = wg * rows_per_wg;
  for (int n = n0 + grp; n < n0 + rows_per_wg; n += MEGA_MWG_ROWS_PAR) {
    float acc = 0.0f;
    #pragma unroll 4
    for (int k4 = lane; k4 < MEGA_HIDDEN / 4; k4 += MEGA_MWG_KGROUP) {
      const float4 wv = convert_float4(read_imageh(W, mega_smp, (int2)(k4, n)));
      const int kb = k4 * 4;
      acc += wv.x * xs[kb] + wv.y * xs[kb + 1] + wv.z * xs[kb + 2] + wv.w * xs[kb + 3];
    }
    part[lid] = acc;
    barrier(CLK_LOCAL_MEM_FENCE);
    for (int o = MEGA_MWG_KGROUP >> 1; o > 0; o >>= 1) {
      if (lane < o) part[lid] += part[lid + o];
      barrier(CLK_LOCAL_MEM_FENCE);
    }
    if (lane == 0) {
      float out = part[grp * MEGA_MWG_KGROUP];
      if (mode == 1)      out += LOAD(xin, n);
      else if (mode == 2) out += resid_g[row * MEGA_HIDDEN + n];
      y_g[row * MEGA_HIDDEN + n] = out;
    }
    barrier(CLK_LOCAL_MEM_FENCE);
  }
}

// fp32-input LayerNorm rows (LN2 / LN3 of the split path; same mega_layernorm
// → bit-identical to the fused kernel's LNs).
__kernel __attribute__((reqd_work_group_size(MEGA_WG, 1, 1)))
void mega_ln_rows_f32(__global const float* x_g,    // [num_rows, hidden] fp32
                      __global const storage_t* ln_w, __global const storage_t* ln_b,
                      __global float* out_g,         // [num_rows, hidden] fp32
                      const float eps) {
  const int row = get_group_id(0);
  const int lid = get_local_id(0);
  __local float xl[MEGA_HIDDEN];
  __local float nl[MEGA_HIDDEN];
  __local float red[MEGA_WG];
  for (int c = lid; c < MEGA_HIDDEN; c += MEGA_WG) xl[c] = x_g[row * MEGA_HIDDEN + c];
  barrier(CLK_LOCAL_MEM_FENCE);
  mega_layernorm(xl, nl, ln_w, ln_b, eps, red, lid);
  for (int c = lid; c < MEGA_HIDDEN; c += MEGA_WG)
    out_g[row * MEGA_HIDDEN + c] = nl[c];
}
#endif  // MEGA_QKV_EXTERNAL && MEGA_TEX

// ── Cross-attn K/V precompute (once per generation, per layer, per CFG row) ──
// k_cross[t,:] = k_proj(enc_states[t,:]); v_cross[t,:] = v_proj(...).
// enc_states: [enc_len, hidden] fp16 (g_enc_states / g_enc_zero).
// W: [hidden,hidden] fp16. Output: [enc_len, hidden] fp16. One row per group.
__kernel __attribute__((reqd_work_group_size(MEGA_WG, 1, 1)))
void decoder_cross_kv_precompute(
    __global const storage_t* enc_states,   // [enc_len, hidden]
    __global const storage_t* k_w,
    __global const storage_t* v_w,
    __global storage_t* k_cross,            // [enc_len, hidden]
    __global storage_t* v_cross,            // [enc_len, hidden]
    const int enc_len) {
  const int t = get_group_id(0);
  const int lid = get_local_id(0);
  if (t >= enc_len) return;
  __local float xrow[MEGA_HIDDEN];
  for (int c = lid; c < MEGA_HIDDEN; c += MEGA_WG) xrow[c] = LOAD(enc_states, t * MEGA_HIDDEN + c);
  barrier(CLK_LOCAL_MEM_FENCE);
  // k
  for (int n = lid; n < MEGA_HIDDEN; n += MEGA_WG) {
    float acc = 0.0f; int wb = n * MEGA_HIDDEN;
    for (int k = 0; k < MEGA_HIDDEN; ++k) acc += LOAD(k_w, wb + k) * xrow[k];
    STORE(k_cross, t * MEGA_HIDDEN + n, acc);
  }
  // v
  for (int n = lid; n < MEGA_HIDDEN; n += MEGA_WG) {
    float acc = 0.0f; int wb = n * MEGA_HIDDEN;
    for (int k = 0; k < MEGA_HIDDEN; ++k) acc += LOAD(v_w, wb + k) * xrow[k];
    STORE(v_cross, t * MEGA_HIDDEN + n, acc);
  }
}

// ── INT8 + FP16-OUTLIER GEMV variants (NNOPT_INT8O) ──────────────────────────
// The Fix-12 verdict's designed quantization: plain int8 (any granularity)
// failed tf-depth (best cb0=33 < 40) because a few large weights move the
// near-tied codebook logits. Scheme: per row, the top ~1.5% largest-|w|
// weights are kept EXACT (fp32 outlier list, applied after the reduce);
// the rest quantize to int8 with one fp32 scale per 128-wide K-group.
// int8 rides the SAME image2d path (CL_RGBA + CL_SIGNED_INT8, read_imagei →
// 4 weights/texel) so the CFG-twin texture-L1 dedup is preserved.
// Bytes/row: K (int8) + K/32 (scales) + ~6% outliers ≈ 0.55× of fp16.
#if defined(MEGA_TEX)
#ifndef MEGA_FC2_KSEG
  #define MEGA_FC2_KSEG 4
#endif
#ifndef MEGA_FC2_SEG
  #define MEGA_FC2_SEG (MEGA_FFN / MEGA_FC2_KSEG)
#endif

// fc1 + GELU, int8-outlier. Geometry identical to mega_ffn_fc1 (tex path).
__kernel __attribute__((reqd_work_group_size(MEGA_MWG_WG, 1, 1)))
void mega_ffn_fc1_i8o(__read_only image2d_t W8,         // [F, H/16] uint32x4 texels (16 int8 each)
                      __global const float* scales,     // [F, H/128]
                      __global const ushort* oidx,      // [F, NOUT]
                      __global const float* ovals,      // [F, NOUT]
                      __global const float* ffn_normed, // [rows, H] fp32
                      __global float* ffn1_g,           // [rows, F] fp32
                      const int rows_per_wg, const int nout) {
  const int wg   = get_group_id(0);
  const int row  = get_group_id(1);
  const int lid  = get_local_id(0);
  const int grp  = lid / MEGA_MWG_KGROUP;
  const int lane = lid % MEGA_MWG_KGROUP;
  __local float xs[MEGA_HIDDEN];
  __local float part[MEGA_MWG_WG];
  for (int c = lid; c < MEGA_HIDDEN; c += MEGA_MWG_WG)
    xs[c] = ffn_normed[row * MEGA_HIDDEN + c];
  barrier(CLK_LOCAL_MEM_FENCE);
  const int n0 = wg * rows_per_wg;
  const int ngroups = MEGA_HIDDEN / 128;
  for (int n = n0 + grp; n < n0 + rows_per_wg; n += MEGA_MWG_ROWS_PAR) {
    float acc = 0.0f;
    // CL_RGBA+UNSIGNED_INT32 texture: one texel = 16 int8 weights, riding the
    // texture L1 (the fp16 path's CFG-twin dedup — buffer int8 lost it and
    // stayed 3.3× slower across 4 variants; round-1 post-mortem in BENCHMARKS).
    // Lanes split across group PAIRS: lanes 0-7 → group gp, lanes 8-15 →
    // gp+1; each lane scales its OWN partial before the tree reduce.
    for (int gp = 0; gp < ngroups; gp += 2) {
      const int g = gp + (lane >> 3);
      const int kb = g * 128 + (lane & 7) * 16;
      const char16 wc = as_char16(read_imageui(W8, mega_smp, (int2)(kb >> 4, n)));
      // VECTOR char4→float4 converts (scalar (float)wc.sN was 16 convert ops
      // per texel — the int8 inner loop was ALU/issue-bound, which is why all
      // five memory-layout variants measured the same 4-7 ms).
      const float4 w0 = convert_float4(wc.s0123);
      const float4 w1 = convert_float4(wc.s4567);
      const float4 w2 = convert_float4(wc.s89ab);
      const float4 w3 = convert_float4(wc.scdef);
      const float4 x0 = vload4(0, xs + kb);
      const float4 x1 = vload4(0, xs + kb + 4);
      const float4 x2 = vload4(0, xs + kb + 8);
      const float4 x3 = vload4(0, xs + kb + 12);
      const float4 pv = w0 * x0 + w1 * x1 + w2 * x2 + w3 * x3;
      acc += scales[(size_t)n * ngroups + g] * (pv.x + pv.y + pv.z + pv.w);
    }
    // Outliers distributed ONE PER LANE and folded into the tree reduce —
    // a lane-0-only loop was 16 serial dependent global loads per row while
    // 15 lanes idled at the barrier (measured: slower than the int8 part).
    for (int j = lane; j < nout; j += MEGA_MWG_KGROUP)
      acc += ovals[(size_t)n * nout + j] * xs[oidx[(size_t)n * nout + j]];
    part[lid] = acc;
    barrier(CLK_LOCAL_MEM_FENCE);
    for (int o = MEGA_MWG_KGROUP >> 1; o > 0; o >>= 1) {
      if (lane < o) part[lid] += part[lid + o];
      barrier(CLK_LOCAL_MEM_FENCE);
    }
    if (lane == 0) {
      float v = part[grp * MEGA_MWG_KGROUP];
      // GELU (tanh approximation — must match the fp16 kernel exactly)
      const float x3 = v * v * v;
      const float tt = 0.7978845608028654f * (v + 0.044715f * x3);
      ffn1_g[(size_t)row * MEGA_FFN + n] = 0.5f * v * (1.0f + tanh(tt));
    }
    barrier(CLK_LOCAL_MEM_FENCE);
  }
}

// fc2 split-K, int8-outlier. Geometry identical to mega_ffn_fc2_sk.
// Outliers are stored sorted by index; each K-segment WG applies only the
// outliers whose index falls inside its segment.
__kernel __attribute__((reqd_work_group_size(MEGA_MWG_WG, 1, 1)))
void mega_ffn_fc2_sk_i8o(__read_only image2d_t W8,      // [H, F/16] uint32x4 texels
                         __global const float* scales,  // [H, F/128]
                         __global const ushort* oidx,   // [H, NOUT] sorted
                         __global const float* ovals,   // [H, NOUT]
                         __global const float* ffn1_g,  // [rows, F] fp32
                         __global float* partials,      // [rows, KSEG, H] fp32
                         const int rows_per_wg, const int nout) {
  const int wg  = get_group_id(0);
  const int seg = get_group_id(1);
  const int row = get_group_id(2);
  const int lid = get_local_id(0);
  const int grp = lid / MEGA_MWG_KGROUP;
  const int lane = lid % MEGA_MWG_KGROUP;
  __local float xs[MEGA_FC2_SEG];
  __local float part[MEGA_MWG_WG];
  __global const float* x = ffn1_g + (size_t)row * MEGA_FFN + seg * MEGA_FC2_SEG;
  for (int c = lid; c < MEGA_FC2_SEG; c += MEGA_MWG_WG) xs[c] = x[c];
  barrier(CLK_LOCAL_MEM_FENCE);
  const int n0 = wg * rows_per_wg;
  const int seg_lo = seg * MEGA_FC2_SEG;
  const int ngroups_seg = MEGA_FC2_SEG / 128;
  const int ngroups_all = MEGA_FFN / 128;
  for (int n = n0 + grp; n < n0 + rows_per_wg; n += MEGA_MWG_ROWS_PAR) {
    float acc = 0.0f;
    // uint32x4 texel = 16 int8; group-pair lane split (see fc1_i8o).
    for (int gp = 0; gp < ngroups_seg; gp += 2) {
      const int g = gp + (lane >> 3);
      const int kb = g * 128 + (lane & 7) * 16;     // local xs offset
      const int kg = seg_lo + kb;                   // global k offset
      const char16 wc = as_char16(read_imageui(W8, mega_smp, (int2)(kg >> 4, n)));
      const float4 w0 = convert_float4(wc.s0123);
      const float4 w1 = convert_float4(wc.s4567);
      const float4 w2 = convert_float4(wc.s89ab);
      const float4 w3 = convert_float4(wc.scdef);
      const float4 x0 = vload4(0, xs + kb);
      const float4 x1 = vload4(0, xs + kb + 4);
      const float4 x2 = vload4(0, xs + kb + 8);
      const float4 x3 = vload4(0, xs + kb + 12);
      const float4 pv = w0 * x0 + w1 * x1 + w2 * x2 + w3 * x3;
      acc += scales[(size_t)n * ngroups_all + (seg * ngroups_seg + g)] * (pv.x + pv.y + pv.z + pv.w);
    }
    // Outliers distributed across lanes (segment-filtered) and folded into
    // the tree reduce — see fc1_i8o note.
    for (int j = lane; j < nout; j += MEGA_MWG_KGROUP) {
      const int oi = oidx[(size_t)n * nout + j];
      if (oi >= seg_lo && oi < seg_lo + MEGA_FC2_SEG)
        acc += ovals[(size_t)n * nout + j] * xs[oi - seg_lo];
    }
    part[lid] = acc;
    barrier(CLK_LOCAL_MEM_FENCE);
    for (int o = MEGA_MWG_KGROUP >> 1; o > 0; o >>= 1) {
      if (lane < o) part[lid] += part[lid + o];
      barrier(CLK_LOCAL_MEM_FENCE);
    }
    if (lane == 0)
      partials[((size_t)row * MEGA_FC2_KSEG + seg) * MEGA_HIDDEN + n] = part[grp * MEGA_MWG_KGROUP];
    barrier(CLK_LOCAL_MEM_FENCE);
  }
}
#endif  // MEGA_TEX (int8-outlier variants)

// ── dot8 INT8 path (cl_qcom_dot_product8 — CONFIRMED on this device) ────────
// Hardware 8-wide int8 dot product. This removes the ALU wall that made the
// six convert-based int8 variants 3-5× slower than fp16 (BENCHMARKS round 2).
// BOTH operands must be int8 → activations are dynamically quantized per
// 128-group by enc-side kernel i8o_actq (fp32 kept too, for the outlier pass).
// The builtin name varies by driver rev — host passes -D MEGA_DOT8_FN=<name>
// (default qcom_sdot8; NNOPT_DOT8_FN overrides without rebuilding).
#ifdef MEGA_DOT8_FN
#pragma OPENCL EXTENSION cl_qcom_dot_product8 : enable

// Quantize fp32 activations to int8, one fp32 scale per 128-group.
// in: [rows, H] fp32 → q8: [rows, H] char, asc: [rows, H/128] float.
// q8 stores OFFSET-BINARY uchar (v+128): the only hardware dot on this driver
// is qcom_dot8_acc(uint, uint, int) with UNSIGNED byte semantics; signed dot
// is recovered via the zero-point identity
//   dot_s = dot_u - 128*(sum_w_s + sum_x_s) - 128*128*count.
// xsum emits the per-group signed activation sums for that correction.
__kernel void i8o_actq(__global const float* in, __global uchar* q8,
                       __global float* asc, __global int* xsum, const int H) {
  const int g = get_global_id(0);     // group index within row
  const int row = get_global_id(1);
  const int ngroups = H / 128;
  if (g >= ngroups) return;
  const int base = row * H + g * 128;
  float m = 0.0f;
  for (int k = 0; k < 128; ++k) m = fmax(m, fabs(in[base + k]));
  const float sc = (m > 0.0f) ? m / 127.0f : 0.0f;
  const float inv = (sc > 0.0f) ? 1.0f / sc : 0.0f;
  asc[row * ngroups + g] = sc;
  int s = 0;
  for (int k = 0; k < 128; ++k) {
    const int v = clamp(convert_int_rte(in[base + k] * inv), -127, 127);
    s += v;
    q8[base + k] = (uchar)(v + 128);
  }
  xsum[row * ngroups + g] = s;
}

// fc1 + GELU via hardware dot8. Geometry identical to mega_ffn_fc1.
__kernel __attribute__((reqd_work_group_size(MEGA_MWG_WG, 1, 1)))
void mega_ffn_fc1_i8d(__global const uchar* W8,        // [F, H] offset-binary int8
                      __global const float* wsc,        // [F, H/128]
                      __global const int* wsum,         // [F, H/128] signed group sums
                      __global const ushort* oidx,      // [F, NOUT]
                      __global const float* ovals,      // [F, NOUT]
                      __global const float* ffn_normed, // [rows, H] fp32 (outliers)
                      __global const uchar* xq8,        // [rows, H] offset-binary acts
                      __global const float* asc,        // [rows, H/128]
                      __global const int* xsum,         // [rows, H/128]
                      __global float* ffn1_g,           // [rows, F]
                      const int rows_per_wg, const int nout) {
  const int wg   = get_group_id(0);
  const int row  = get_group_id(1);
  const int lid  = get_local_id(0);
  const int grp  = lid / MEGA_MWG_KGROUP;
  const int lane = lid % MEGA_MWG_KGROUP;
  __local float xs[MEGA_HIDDEN];        // fp32 acts (outlier pass)
  __local uint  xqu[MEGA_HIDDEN / 4];   // int8 acts PACKED as uints (dot8 operand)
  __local float ascl[MEGA_HIDDEN / 128];
  __local float part[MEGA_MWG_WG];
  const int ngroups = MEGA_HIDDEN / 128;
  for (int c = lid; c < MEGA_HIDDEN; c += MEGA_MWG_WG)
    xs[c] = ffn_normed[row * MEGA_HIDDEN + c];
  for (int c = lid; c < MEGA_HIDDEN / 4; c += MEGA_MWG_WG)
    xqu[c] = ((__global const uint*)(xq8 + (size_t)row * MEGA_HIDDEN))[c];
  __local int wsl[64 * (MEGA_HIDDEN / 128)];   // this WG's rows' weight sums
  if (lid < ngroups) ascl[lid] = asc[row * ngroups + lid];
  (void)xsum;   // unused under mixed-sign dot semantics (no act-sum correction)
  const int n0 = wg * rows_per_wg;
  for (int c = lid; c < rows_per_wg * ngroups; c += MEGA_MWG_WG)
    wsl[c] = ((__global const int*)wsum)[(size_t)n0 * ngroups + c];
  barrier(CLK_LOCAL_MEM_FENCE);
  // (wscl local staging of the weight scales measured WORSE — 2.04 → 2.28 ms:
  // +2 KB local dropped residency. Global wsc reads are cache-served fine.)
  for (int n = n0 + grp; n < n0 + rows_per_wg; n += MEGA_MWG_ROWS_PAR) {
    float acc = 0.0f;
    for (int g = 0; g < ngroups; ++g) {
      const int kb = g * 128 + lane * 8;
      // PROBED SEMANTICS (kernels/dot8_probe.cl, 2026-06-05):
      // qcom_dot8_acc(a, b, acc) = Σ SIGNED(a_byte)·UNSIGNED(b_byte) + acc.
      // → weights ride arg1 as plain signed int8; activations ride arg2 as
      // offset-binary (x_u = x_s + 128). dot(w_s, x_u) = dot(w_s, x_s)
      // + 128·Σw_s, so the only correction is −128·wsum[n][g] (lane 0, once;
      // the tree reduce sums lanes).
      const uint2 wu = vload2(0, (__global const uint*)(W8 + (size_t)n * MEGA_HIDDEN + kb));
      const uint x0 = xqu[(kb >> 2)];
      const uint x1 = xqu[(kb >> 2) + 1];
      // Zero-point correction IN INTEGER DOMAIN (float-domain version lost
      // cb0 46→29 to cancellation: uncorrected dots ~2M, fp32 drops the low
      // bits). BRANCHLESS: lane 0 SEEDS the hardware dot's accumulator with
      // the full -128·wsum for this (n,g); other lanes seed 0 (select, not a
      // divergent if — the if-form measured +0.7 ms). Tree reduce sums lanes
      // → group corrected exactly once, all integer.
      const int seed = (lane == 0) ? (-128 * wsl[(n - n0) * ngroups + g]) : 0;
      int d = MEGA_DOT8_FN(wu.x, x0, seed);
      d = MEGA_DOT8_FN(wu.y, x1, d);
      acc += wsc[(size_t)n * ngroups + g] * ascl[g] * (float)d;
    }
    for (int j = lane; j < nout; j += MEGA_MWG_KGROUP)
      acc += ovals[(size_t)n * nout + j] * xs[oidx[(size_t)n * nout + j]];
    part[lid] = acc;
    barrier(CLK_LOCAL_MEM_FENCE);
    for (int o = MEGA_MWG_KGROUP >> 1; o > 0; o >>= 1) {
      if (lane < o) part[lid] += part[lid + o];
      barrier(CLK_LOCAL_MEM_FENCE);
    }
    if (lane == 0) {
      float v = part[grp * MEGA_MWG_KGROUP];
      const float x3 = v * v * v;
      const float tt = 0.7978845608028654f * (v + 0.044715f * x3);
      ffn1_g[(size_t)row * MEGA_FFN + n] = 0.5f * v * (1.0f + tanh(tt));
    }
    barrier(CLK_LOCAL_MEM_FENCE);
  }
}
#endif  // MEGA_DOT8_FN

// ── lm_heads GEMV (replaces the CLBlast GEMM) ────────────────────────────────
// logits[row, n] = lm_heads_cat[n, :] · x[row, :]  for n in [0, NCB*VOCAB).
// The CLBlast call was the last library kernel in the decode step: unprofiled
// (invisible in every kernel table), suspected of internal event waits (the
// surviving ~20 ms/step wall-vs-kernels gap), and its first-call compile sat
// in TTFT outside our binary cache. Same multi-WG + texture + unroll-4 recipe
// as mega_proj; input x and output logits are storage_t (fp16) like the
// CLBlast path. NOT bit-identical to CLBlast's accumulation order → tf-gated.
#if defined(MEGA_TEX)
__kernel __attribute__((reqd_work_group_size(MEGA_MWG_WG, 1, 1)))
void mega_lmheads(__read_only image2d_t W,        // [N, H/4 texels]
                  __global const storage_t* x_g,   // [num_rows, H] fp16
                  __global storage_t* logits,      // [num_rows, N] fp16
                  const int N, const int rows_per_wg) {
  const int wg   = MEGA_ROWWG_SLICE;
  const int row  = MEGA_ROWWG_ROW;
  const int lid  = get_local_id(0);
  const int grp  = lid / MEGA_MWG_KGROUP;
  const int lane = lid % MEGA_MWG_KGROUP;
  __local float xs[MEGA_HIDDEN];
  __local float part[MEGA_MWG_WG];
  for (int c = lid; c < MEGA_HIDDEN; c += MEGA_MWG_WG)
    xs[c] = LOAD(x_g, row * MEGA_HIDDEN + c);
  barrier(CLK_LOCAL_MEM_FENCE);
  const int n0 = wg * rows_per_wg;
  for (int n = n0 + grp; n < n0 + rows_per_wg; n += MEGA_MWG_ROWS_PAR) {
    float acc = 0.0f;
    #pragma unroll 4
    for (int k4 = lane; k4 < MEGA_HIDDEN / 4; k4 += MEGA_MWG_KGROUP) {
      const float4 wv = convert_float4(read_imageh(W, mega_smp, (int2)(k4, n)));
      const int kb = k4 * 4;
      acc += wv.x * xs[kb] + wv.y * xs[kb + 1] + wv.z * xs[kb + 2] + wv.w * xs[kb + 3];
    }
    part[lid] = acc;
    barrier(CLK_LOCAL_MEM_FENCE);
    for (int o = MEGA_MWG_KGROUP >> 1; o > 0; o >>= 1) {
      if (lane < o) part[lid] += part[lid + o];
      barrier(CLK_LOCAL_MEM_FENCE);
    }
    if (lane == 0)
      STORE(logits, (size_t)row * N + n, part[grp * MEGA_MWG_KGROUP]);
    barrier(CLK_LOCAL_MEM_FENCE);
  }
}
#endif  // MEGA_TEX (lm_heads)

// ── INT4 GEMV PROBE (MEGA_INT4_PROBE) ───────────────────────────────────────
// Weight-only int4 [N,K] GEMV, group-quantized (32-wide groups, one fp32 scale
// per group), nibble-packed as CL_RGBA/CL_UNSIGNED_INT32 (one texel = 4 uint32
// = 32 signed int4 weights → int4's DENSEST possible fetch: 8x fewer texture
// reads than the fp16 read_imageh path). fp32 accumulate. Same thread map /
// tree-reduce as mega_proj so the ONLY difference vs the fp16 GEMV is the
// weight format + nibble unpack. This is the head-to-head speed probe: if it
// does not beat mega_proj here, no int4 decode path can win on this device.
#if defined(MEGA_INT4_PROBE) && defined(MEGA_TEX)
__kernel __attribute__((reqd_work_group_size(MEGA_MWG_WG, 1, 1)))
void mega_proj_int4(__read_only image2d_t W4,        // [N, K/32] RGBA uint32
                    __global const float* scale,     // [N, K/32] fp32 group scale
                    __global const float* x_g,       // [rows, K] fp32
                    __global float* y_g,             // [rows, N] fp32
                    const int rows_per_wg) {
  const int wg   = MEGA_ROWWG_SLICE;
  const int row  = MEGA_ROWWG_ROW;
  const int lid  = get_local_id(0);
  const int grp  = lid / MEGA_MWG_KGROUP;
  const int lane = lid % MEGA_MWG_KGROUP;
  __local float xs[MEGA_HIDDEN];
  __local float part[MEGA_MWG_WG];
  for (int c = lid; c < MEGA_HIDDEN; c += MEGA_MWG_WG) xs[c] = x_g[row * MEGA_HIDDEN + c];
  barrier(CLK_LOCAL_MEM_FENCE);
  const int K32 = MEGA_HIDDEN / 32;
  const int n0  = wg * rows_per_wg;
  for (int n = n0 + grp; n < n0 + rows_per_wg; n += MEGA_MWG_ROWS_PAR) {
    float acc = 0.0f;
    for (int k32 = lane; k32 < K32; k32 += MEGA_MWG_KGROUP) {
      const uint4 pk = read_imageui(W4, mega_smp, (int2)(k32, n));
      const float s  = scale[(size_t)n * K32 + k32];
      const int kb   = k32 * 32;
      float g = 0.0f;
      uint w = pk.x;
      #pragma unroll
      for (int i = 0; i < 8; ++i) { int q = (int)((w >> (4*i)) & 0xF); q = (q ^ 0x8) - 0x8; g += q * xs[kb + i]; }
      w = pk.y;
      #pragma unroll
      for (int i = 0; i < 8; ++i) { int q = (int)((w >> (4*i)) & 0xF); q = (q ^ 0x8) - 0x8; g += q * xs[kb + 8 + i]; }
      w = pk.z;
      #pragma unroll
      for (int i = 0; i < 8; ++i) { int q = (int)((w >> (4*i)) & 0xF); q = (q ^ 0x8) - 0x8; g += q * xs[kb + 16 + i]; }
      w = pk.w;
      #pragma unroll
      for (int i = 0; i < 8; ++i) { int q = (int)((w >> (4*i)) & 0xF); q = (q ^ 0x8) - 0x8; g += q * xs[kb + 24 + i]; }
      acc += s * g;
    }
    part[lid] = acc;
    barrier(CLK_LOCAL_MEM_FENCE);
    for (int o = MEGA_MWG_KGROUP >> 1; o > 0; o >>= 1) {
      if (lane < o) part[lid] += part[lid + o];
      barrier(CLK_LOCAL_MEM_FENCE);
    }
    if (lane == 0) y_g[(size_t)row * MEGA_HIDDEN + n] = part[grp * MEGA_MWG_KGROUP];
    barrier(CLK_LOCAL_MEM_FENCE);
  }
}
#endif  // MEGA_INT4_PROBE
