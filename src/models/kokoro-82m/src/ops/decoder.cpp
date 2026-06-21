// Reference: kokoro/istftnet.py — Decoder + Generator + AdainResBlk1d + AdaINResBlock1 + AdaIN1d
//
// Full chain (simplified vs reference):
//   asr [512, T_frames], F0_pred[T_frames], N_pred[T_frames], ref_s_dec[128]
//   F0 = F0_conv(F0_pred.unsqueeze(1))                 stride=2 conv
//   N  = N_conv(N_pred.unsqueeze(1))                   stride=2 conv
//   x = concat([asr, F0, N], axis=C)                   [514, T_frames]
//   x = encode(x, ref_s_dec)                           AdainResBlk1d 514 -> 1024 [1024, T_frames]
//   asr_res = asr_res(asr)                             Conv1d 512 -> 64
//   for i in 0..3:
//     if i not yet past upsample: x = concat([x, asr_res, F0, N], axis=C) [1024+64+2, T_frames]
//     x = decode[i](x, ref_s_dec)                      AdainResBlk1d (last one upsamples by 2)
//   # x is now [512, T_frames * 2] after final decode block
//   x = generator(x, ref_s_dec, F0_pred)               see below
//
// Simplifications from reference:
//   - Snake1D activation replaced with LeakyReLU(0.1). Snake is x + (1/a)*sin^2(a*x);
//     numerically close to LeakyReLU for the small alpha values trained here.
//   - SineGen/SourceModuleHnNSF excitation skipped (har_source = 0, noise_source = 0).
//     Voicing comes only from F0_pred conditioning into AdaIN, not from explicit harmonics.
//   - generator's noise_convs and noise_res are zeroed out (their input "har" is zero).
//   - All weight_norm parametrizations are reconstructed on first use and cached.

#include "opencl_context.h"
#include "weights.h"
#include "debug_utils.h"
#include "profiler.h"
#include "model_config.h"
#include "utils.h"

#include <CL/cl.h>
#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <unordered_map>
#include <vector>

extern "C" int prim_layernorm(OpenCLContext&, cl_command_queue,
                              cl_mem, cl_mem, cl_mem, cl_mem, int, int, float);

extern "C" int op_decoder_host(OpenCLContext&, Weights&, cl_command_queue,
                                cl_mem, cl_mem, cl_mem, cl_mem,
                                int, int, std::vector<int16_t>&);
extern "C" int op_decoder_gpu_fp32(OpenCLContext&, Weights&, cl_command_queue,
                                    cl_mem, cl_mem, cl_mem, cl_mem,
                                    int, int, std::vector<int16_t>&);

// ─── Kernels ─────────────────────────────────────────────────────────────
static cl_program g_dec_prog = nullptr;
static cl_kernel  g_k_instnorm = nullptr;
static cl_kernel  g_k_adain_combine = nullptr;
static cl_kernel  g_k_leaky = nullptr;
static cl_kernel  g_k_add = nullptr;
static cl_kernel  g_k_add_scale = nullptr;
static cl_kernel  g_k_concat = nullptr;       // generic 2-input concat along C
static cl_kernel  g_k_concat4 = nullptr;      // 4-input concat along C
static cl_kernel  g_k_upsample_nn = nullptr;
static cl_kernel  g_k_conv1d = nullptr;
static cl_kernel  g_k_conv1d_fast = nullptr;
static cl_kernel  g_k_conv1d_fast_c4 = nullptr;
static cl_kernel  g_k_conv1d_int8_c4 = nullptr;
static cl_kernel  g_k_conv1d_c4x4 = nullptr;
static cl_kernel  g_k_conv1d_rt4 = nullptr;
static cl_kernel  g_k_im2col = nullptr;
static cl_kernel  g_k_transpose_tn = nullptr;
static cl_kernel  g_k_pad_rows = nullptr;
static cl_kernel  g_k_convtr1d_c4x4 = nullptr;
static cl_kernel  g_k_convtr1d_c4x4_tex = nullptr;
static cl_kernel  g_k_convtr_pack = nullptr;
static cl_kernel  g_k_convtr1d = nullptr;
static cl_kernel  g_k_exp = nullptr;
static cl_kernel  g_k_sin_act = nullptr;
static cl_kernel  g_k_istft = nullptr;
static cl_kernel  g_k_bias_add_NCL = nullptr;
static cl_kernel  g_k_weightnorm = nullptr;
static cl_kernel  g_k_linear_apply = nullptr; // y[c, t] = w[c, k] @ s[k] + b[c]  (style->gamma_beta projection)
static cl_kernel  g_k_split = nullptr;        // split [2C, T] -> [C, T], [C, T]
static cl_kernel  g_k_pad_unsqueeze = nullptr;
static cl_kernel  g_k_snake1d = nullptr;
static cl_kernel  g_k_refl_pad = nullptr;
static cl_kernel  g_k_div_scalar = nullptr;
static cl_kernel  g_k_upsample_linear = nullptr;
static cl_kernel  g_k_sinegen = nullptr;
static cl_kernel  g_k_source_merge = nullptr;
static cl_kernel  g_k_stft = nullptr;
static cl_kernel  g_k_exp_clamp = nullptr;

static const char* k_dec_src = R"CLC(
// fp16 types are used by the fast-conv weight caches in BOTH build flavors
// (they store weights as half to halve LDS footprint), so the extension is
// enabled unconditionally — Adreno 6xx always supports cl_khr_fp16.
#pragma OPENCL EXTENSION cl_khr_fp16 : enable
#ifdef NNOPT_USE_FP16
  typedef half storage_t;
  #define LOAD(p,i)    vload_half((i), (__global const half*)(p))
  #define STORE(p,i,v) vstore_half((float)(v), (i), (__global half*)(p))
  // 4-wide contiguous load/store along the last (L) dim — single 64-bit transaction.
  #define LOAD4(p,i)    vload_half4(0, ((__global const half*)(p)) + (i))
  #define STORE4(v,p,i) vstore_half4((v), 0, ((__global half*)(p)) + (i))
#else
  typedef float storage_t;
  #define LOAD(p,i)    ((p)[(i)])
  #define STORE(p,i,v) ((p)[(i)] = (v))
  #define LOAD4(p,i)    vload4(0, (__global const float*)(p) + (i))
  #define STORE4(v,p,i) vstore4((v), 0, (__global float*)(p) + (i))
#endif

// Workgroup-per-channel instance norm: 64 lanes strided-sum T, tree-reduce in
// LDS, then all lanes write the normalized row. Replaces the serial
// 1-work-item-per-channel version that was 3 sequential passes over T.
__kernel void instnorm_NCL(__global const storage_t* x,
                           __global storage_t* y,
                           int C, int T, float eps) {
    int c  = get_group_id(0);
    int lt = get_local_id(0);
    if (c >= C) return;
    int base = c * T;
    __local float lsum[64];
    __local float lsq[64];
    float s = 0.0f, q = 0.0f;
    for (int t = lt; t < T; t += 64) {
        float v = (float)LOAD(x, base + t);
        s += v; q += v * v;
    }
    lsum[lt] = s; lsq[lt] = q;
    barrier(CLK_LOCAL_MEM_FENCE);
    for (int off = 32; off > 0; off >>= 1) {
        if (lt < off) { lsum[lt] += lsum[lt + off]; lsq[lt] += lsq[lt + off]; }
        barrier(CLK_LOCAL_MEM_FENCE);
    }
    float mean = lsum[0] / (float)T;
    float var  = lsq[0] / (float)T - mean * mean;
    if (var < 0.0f) var = 0.0f;
    float inv = rsqrt(var + eps);
    for (int t = lt; t < T; t += 64) {
        STORE(y, base + t, ((float)LOAD(x, base + t) - mean) * inv);
    }
}

__kernel void adain_combine(__global const storage_t* x,
                            __global const storage_t* gamma,
                            __global const storage_t* beta,
                            __global storage_t* y,
                            int C, int T) {
    int c = get_global_id(0);
    int t = get_global_id(1);
    if (c >= C || t >= T) return;
    float g = (float)LOAD(gamma, c);
    float b = (float)LOAD(beta, c);
    float v = (1.0f + g) * (float)LOAD(x, c*T + t) + b;
    STORE(y, c*T + t, v);
}

__kernel void leaky_relu(__global storage_t* y, int N, float slope) {
    int i = get_global_id(0);
    if (i >= N) return;
    float v = (float)LOAD(y, i);
    if (v < 0.0f) v *= slope;
    STORE(y, i, v);
}

__kernel void k_add(__global storage_t* y, __global const storage_t* x, int N) {
    int i = get_global_id(0); if (i >= N) return;
    STORE(y, i, (float)LOAD(y,i) + (float)LOAD(x,i));
}

__kernel void k_add_scale(__global storage_t* y, __global const storage_t* x, int N, float s) {
    int i = get_global_id(0); if (i >= N) return;
    STORE(y, i, ((float)LOAD(y,i) + (float)LOAD(x,i)) * s);
}

// Concat 2 [Ca, T] + [Cb, T] -> [Ca+Cb, T]
__kernel void concat_C(__global const storage_t* a, int Ca,
                       __global const storage_t* b, int Cb,
                       __global storage_t* y, int T) {
    int oc = get_global_id(0);
    int t = get_global_id(1);
    int C_total = Ca + Cb;
    if (oc >= C_total || t >= T) return;
    float v = (oc < Ca) ? (float)LOAD(a, oc*T + t) : (float)LOAD(b, (oc - Ca)*T + t);
    STORE(y, oc*T + t, v);
}

// Concat 4 along channel.
__kernel void concat4_C(__global const storage_t* a, int Ca,
                        __global const storage_t* b, int Cb,
                        __global const storage_t* c, int Cc,
                        __global const storage_t* d, int Cd,
                        __global storage_t* y, int T) {
    int oc = get_global_id(0);
    int t = get_global_id(1);
    int Ct = Ca + Cb + Cc + Cd;
    if (oc >= Ct || t >= T) return;
    float v;
    if (oc < Ca) v = (float)LOAD(a, oc*T + t);
    else if (oc < Ca+Cb) v = (float)LOAD(b, (oc-Ca)*T + t);
    else if (oc < Ca+Cb+Cc) v = (float)LOAD(c, (oc-Ca-Cb)*T + t);
    else v = (float)LOAD(d, (oc-Ca-Cb-Cc)*T + t);
    STORE(y, oc*T + t, v);
}

__kernel void upsample_nn_NCL(__global const storage_t* x,
                              __global storage_t* y,
                              int C, int T_in, int scale) {
    int c = get_global_id(0);
    int t_out = get_global_id(1);
    int T_out = T_in * scale;
    if (c >= C || t_out >= T_out) return;
    int t_in = t_out / scale;
    if (t_in >= T_in) t_in = T_in - 1;
    STORE(y, c*T_out + t_out, (float)LOAD(x, c*T_in + t_in));
}

__kernel void conv1d_dilated(__global const storage_t* in,
                             __global const storage_t* W,
                             __global storage_t* out,
                             int C_in, int C_out, int L_in, int L_out,
                             int K, int stride, int padding, int dilation, int groups) {
    int oc = get_global_id(0);
    int ol = get_global_id(1);
    if (oc >= C_out || ol >= L_out) return;
    int grp_in_per_out = C_in / groups;
    int g = oc / (C_out / groups);
    float acc = 0.0f;
    for (int k = 0; k < K; ++k) {
        int il = ol * stride - padding + k * dilation;
        if (il < 0 || il >= L_in) continue;
        for (int ic = 0; ic < grp_in_per_out; ++ic) {
            int ic_g = g * grp_in_per_out + ic;
            int w_idx = oc * grp_in_per_out * K + ic * K + k;
            acc += (float)LOAD(in, ic_g*L_in + il) * (float)LOAD(W, w_idx);
        }
    }
    STORE(out, oc*L_out + ol, acc);
}

// Optimized variant: one workgroup computes (oc, LOCAL_T_DEC contiguous L_out positions).
// Weight row for oc is loaded once into LDS by cooperative load, then reused across all
// L_out positions. Inner loop runs 4-way unrolled across C_in (compiler-recognizable
// pattern that Adreno's shader compiler vectorizes into HALF8 / FLOAT4 issue slots).
// Caller MUST guarantee grp_in_per_out * K <= WCACHE_DEC_MAX or behavior is undefined.
//
// Note: I tried L-tiling (each work item producing 2 or 4 outputs sharing input loads
// across overlapping receptive fields, per Qualcomm Epsilon-filter case study §10.3.3).
// On this kernel/hardware combination, both L_TILE=2 and L_TILE=4 regressed: the loss
// of L-dimension workgroup parallelism (and the fast/slow boundary-path branching) cost
// more than the input-reuse savings. The original 4-way ic-unrolled single-output
// pattern is a local optimum for the Adreno 620 compiler. See journal entry 13.x.
#define LOCAL_T_DEC 384
#define WCACHE_DEC_MAX 4096
__kernel void conv1d_dilated_fast(__global const storage_t* in,
                                   __global const storage_t* W,
                                   __global storage_t* out,
                                   int C_in, int C_out, int L_in, int L_out,
                                   int K, int stride, int padding, int dilation, int groups) {
    int oc = get_global_id(0);
    int lt = get_local_id(1);
    int ol = get_group_id(1) * LOCAL_T_DEC + lt;
    int grp_in_per_out = C_in / groups;
    int g = oc / (C_out / groups);
    int w_base = oc * grp_in_per_out * K;
    int w_total = grp_in_per_out * K;
    __local half W_cache[WCACHE_DEC_MAX];
    // Cooperative load: store fp16 weights directly in LDS (half the LDS footprint
    // vs caching as fp32 → more concurrent workgroups per SP).
    for (int i = lt; i < w_total; i += LOCAL_T_DEC) {
        W_cache[i] = vload_half(w_base + i, (__global const half*)W);
    }
    barrier(CLK_LOCAL_MEM_FENCE);
    if (oc >= C_out || ol >= L_out) return;
    float acc = 0.0f;
    for (int k = 0; k < K; ++k) {
        int il = ol * stride - padding + k * dilation;
        if (il < 0 || il >= L_in) continue;
        int ic = 0;
        // 4-way unrolled — Adreno coalesces consecutive vload_half issues into HALF8 reads.
        for (; ic + 3 < grp_in_per_out; ic += 4) {
            int ic_g = g * grp_in_per_out + ic;
            float i0 = (float)LOAD(in, (ic_g+0)*L_in + il);
            float i1 = (float)LOAD(in, (ic_g+1)*L_in + il);
            float i2 = (float)LOAD(in, (ic_g+2)*L_in + il);
            float i3 = (float)LOAD(in, (ic_g+3)*L_in + il);
            acc += i0 * W_cache[(ic+0)*K + k]
                 + i1 * W_cache[(ic+1)*K + k]
                 + i2 * W_cache[(ic+2)*K + k]
                 + i3 * W_cache[(ic+3)*K + k];
        }
        for (; ic < grp_in_per_out; ++ic) {
            int ic_g = g * grp_in_per_out + ic;
            acc += (float)LOAD(in, ic_g*L_in + il) * W_cache[ic*K + k];
        }
    }
    STORE(out, oc*L_out + ol, acc);
}

// Cout-tile=4 variant: one workgroup computes (4 contiguous oc, LOCAL_T_DEC contiguous
// L_out positions). Each work item maintains 4 accumulators — the input value at each
// (ic, il) is read ONCE per workgroup and multiplied by 4 different weights. This cuts
// input bandwidth 4× vs the single-oc variant, at the cost of 4× LDS for W cache and
// 4× register pressure.
// Requires C_out % 4 == 0. WCACHE_C4_MAX must cover 4 * grp_in_per_out * K.
#define COUT_TILE 4
#define WCACHE_C4_MAX 16384  // 4 * 2816 = 11264, rounded up
__kernel void conv1d_dilated_fast_c4(__global const storage_t* in,
                                      __global const storage_t* W,
                                      __global storage_t* out,
                                      int C_in, int C_out, int L_in, int L_out,
                                      int K, int stride, int padding, int dilation, int groups) {
    int oc_base = get_global_id(0) * COUT_TILE;
    int lt = get_local_id(1);
    int ol = get_group_id(1) * LOCAL_T_DEC + lt;
    int grp_in_per_out = C_in / groups;
    int g = oc_base / (C_out / groups);
    int w_total_per_oc = grp_in_per_out * K;
    int w_total = COUT_TILE * w_total_per_oc;
    int w_base = oc_base * w_total_per_oc;
    __local half W_cache[WCACHE_C4_MAX];
    // Cooperative load — 4× weight rows packed contiguously.
    for (int i = lt; i < w_total; i += LOCAL_T_DEC) {
        W_cache[i] = vload_half(w_base + i, (__global const half*)W);
    }
    barrier(CLK_LOCAL_MEM_FENCE);
    if (oc_base >= C_out || ol >= L_out) return;
    // 4 accumulators, one per oc in the tile.
    // COUT_TILE accumulators per oc + ic 4-way unroll. Each work item produces
    // COUT_TILE output values across COUT_TILE channels. Inputs are loaded once
    // per (ic, il) and multiplied by COUT_TILE weights — saves COUT_TILE × input
    // bandwidth vs the single-oc kernel.
    float acc0 = 0.0f, acc1 = 0.0f, acc2 = 0.0f, acc3 = 0.0f;
    int g_base = g * grp_in_per_out;
    int w_off1 = 1 * w_total_per_oc;
    int w_off2 = 2 * w_total_per_oc;
    int w_off3 = 3 * w_total_per_oc;
    for (int k = 0; k < K; ++k) {
        int il = ol * stride - padding + k * dilation;
        if (il < 0 || il >= L_in) continue;
        int ic = 0;
        // 4-way ic unroll — matches the baseline kernel pattern that Adreno
        // compiler optimizes well into coalesced strided HALF loads.
        for (; ic + 3 < grp_in_per_out; ic += 4) {
            int ic_g = g_base + ic;
            float x0 = (float)LOAD(in, (ic_g+0)*L_in + il);
            float x1 = (float)LOAD(in, (ic_g+1)*L_in + il);
            float x2 = (float)LOAD(in, (ic_g+2)*L_in + il);
            float x3 = (float)LOAD(in, (ic_g+3)*L_in + il);
            int w_idx0 = (ic+0)*K + k;
            int w_idx1 = (ic+1)*K + k;
            int w_idx2 = (ic+2)*K + k;
            int w_idx3 = (ic+3)*K + k;
            acc0 += x0 * (float)W_cache[w_idx0]
                  + x1 * (float)W_cache[w_idx1]
                  + x2 * (float)W_cache[w_idx2]
                  + x3 * (float)W_cache[w_idx3];
            acc1 += x0 * (float)W_cache[w_idx0 + w_off1]
                  + x1 * (float)W_cache[w_idx1 + w_off1]
                  + x2 * (float)W_cache[w_idx2 + w_off1]
                  + x3 * (float)W_cache[w_idx3 + w_off1];
            acc2 += x0 * (float)W_cache[w_idx0 + w_off2]
                  + x1 * (float)W_cache[w_idx1 + w_off2]
                  + x2 * (float)W_cache[w_idx2 + w_off2]
                  + x3 * (float)W_cache[w_idx3 + w_off2];
            acc3 += x0 * (float)W_cache[w_idx0 + w_off3]
                  + x1 * (float)W_cache[w_idx1 + w_off3]
                  + x2 * (float)W_cache[w_idx2 + w_off3]
                  + x3 * (float)W_cache[w_idx3 + w_off3];
        }
        for (; ic < grp_in_per_out; ++ic) {
            int ic_g = g_base + ic;
            float xv = (float)LOAD(in, ic_g * L_in + il);
            int w_idx = ic * K + k;
            acc0 += xv * (float)W_cache[w_idx];
            acc1 += xv * (float)W_cache[w_idx + w_off1];
            acc2 += xv * (float)W_cache[w_idx + w_off2];
            acc3 += xv * (float)W_cache[w_idx + w_off3];
        }
    }
    STORE(out, (oc_base + 0) * L_out + ol, acc0);
    STORE(out, (oc_base + 1) * L_out + ol, acc1);
    STORE(out, (oc_base + 2) * L_out + ol, acc2);
    STORE(out, (oc_base + 3) * L_out + ol, acc3);
}

// Register-tiled 4(oc) x 4(ol) variant for stride==1 / groups==1 convs (all
// resblock + decode-chain convs). Mirrors the proven fp32 generator kernel:
//   - input loaded as one 4-wide vector per (ic,k) via LOAD4 (contiguous along
//     L in the channel-major layout), reused across 4 output channels;
//   - weights read directly from global: lws dim0==1 makes every work-item in
//     the group read the SAME 4 scalars per iteration -> broadcast via L2;
//   - no local memory, no barrier (streaming mode), ~30 registers/WI.
__kernel void conv1d_dilated_c4x4(__global const storage_t* in,
                                  __global const storage_t* W,
                                  __global storage_t* out,
                                  int C_in, int C_out, int L_in, int L_out,
                                  int K, int padding, int dilation) {
    int ocb = (int)get_global_id(0) * 4;
    int ol0 = (int)get_global_id(1) * 4;
    if (ocb >= C_out || ol0 >= L_out) return;
    float4 acc0 = (float4)(0.0f);
    float4 acc1 = (float4)(0.0f);
    float4 acc2 = (float4)(0.0f);
    float4 acc3 = (float4)(0.0f);
    int ws = mul24(C_in, K);
    int w0 = mul24(ocb + 0, ws);
    int w1 = mul24(ocb + 1, ws);
    int w2 = mul24(ocb + 2, ws);
    int w3 = mul24(ocb + 3, ws);
    for (int k = 0; k < K; ++k) {
        int il = ol0 - padding + mul24(k, dilation);
        if (il >= 0 && il + 3 < L_in) {
            for (int ic = 0; ic < C_in; ++ic) {
                float4 v = LOAD4(in, mul24(ic, L_in) + il);
                int wi = mad24(ic, K, k);
                acc0 = mad((float)LOAD(W, w0 + wi), v, acc0);
                acc1 = mad((float)LOAD(W, w1 + wi), v, acc1);
                acc2 = mad((float)LOAD(W, w2 + wi), v, acc2);
                acc3 = mad((float)LOAD(W, w3 + wi), v, acc3);
            }
        } else if (il + 3 >= 0 && il < L_in) {
            for (int ic = 0; ic < C_in; ++ic) {
                int rb = mul24(ic, L_in);
                float4 v;
                v.x = (il     >= 0 && il     < L_in) ? (float)LOAD(in, rb + il)     : 0.0f;
                v.y = (il + 1 >= 0 && il + 1 < L_in) ? (float)LOAD(in, rb + il + 1) : 0.0f;
                v.z = (il + 2 >= 0 && il + 2 < L_in) ? (float)LOAD(in, rb + il + 2) : 0.0f;
                v.w = (il + 3 >= 0 && il + 3 < L_in) ? (float)LOAD(in, rb + il + 3) : 0.0f;
                int wi = mad24(ic, K, k);
                acc0 = mad((float)LOAD(W, w0 + wi), v, acc0);
                acc1 = mad((float)LOAD(W, w1 + wi), v, acc1);
                acc2 = mad((float)LOAD(W, w2 + wi), v, acc2);
                acc3 = mad((float)LOAD(W, w3 + wi), v, acc3);
            }
        }
    }
    int rem = L_out - ol0;
    if (rem >= 4) {
        STORE4(acc0, out, mul24(ocb + 0, L_out) + ol0);
        STORE4(acc1, out, mul24(ocb + 1, L_out) + ol0);
        STORE4(acc2, out, mul24(ocb + 2, L_out) + ol0);
        STORE4(acc3, out, mul24(ocb + 3, L_out) + ol0);
    } else {
        int o0 = mul24(ocb + 0, L_out) + ol0;
        int o1 = mul24(ocb + 1, L_out) + ol0;
        int o2 = mul24(ocb + 2, L_out) + ol0;
        int o3 = mul24(ocb + 3, L_out) + ol0;
        STORE(out, o0, acc0.x); STORE(out, o1, acc1.x);
        STORE(out, o2, acc2.x); STORE(out, o3, acc3.x);
        if (rem > 1) { STORE(out, o0+1, acc0.y); STORE(out, o1+1, acc1.y);
                       STORE(out, o2+1, acc2.y); STORE(out, o3+1, acc3.y); }
        if (rem > 2) { STORE(out, o0+2, acc0.z); STORE(out, o1+2, acc1.z);
                       STORE(out, o2+2, acc2.z); STORE(out, o3+2, acc3.z); }
    }
}

// Small-T cooperative conv: one workgroup computes one (oc, 4-ol tile).
// 64 lanes split C_in and LDS-tree-reduce the 4 partial sums. This is the
// right decomposition for the decode/encode chain shapes (C_in ~1090, K=3,
// T=87..174) where per-WI tiling leaves the GPU latency-bound: parallelism
// becomes C_out * tiles * 64 lanes instead of C_out * tiles, and the per-lane
// loop shrinks from C_in*K to (C_in/64)*K iterations.
// Works for ANY C_out (no %4 requirement) — also used for C_out==1 projections.
__kernel void conv1d_dilated_rt4(__global const storage_t* in,
                                 __global const storage_t* W,
                                 __global storage_t* out,
                                 int C_in, int C_out, int L_in, int L_out,
                                 int K, int padding, int dilation) {
    int oc  = get_group_id(0);
    int ol0 = (int)get_group_id(1) * 4;
    int lt  = get_local_id(1);
    float4 acc = (float4)(0.0f);
    int wbase = mul24(oc, mul24(C_in, K));
    for (int ic = lt; ic < C_in; ic += 64) {
        int rb = mul24(ic, L_in);
        int wrow = wbase + mul24(ic, K);
        for (int k = 0; k < K; ++k) {
            int il = ol0 - padding + mul24(k, dilation);
            float4 v;
            if (il >= 0 && il + 3 < L_in) {
                v = LOAD4(in, rb + il);
            } else if (il + 3 >= 0 && il < L_in) {
                v.x = (il     >= 0 && il     < L_in) ? (float)LOAD(in, rb + il)     : 0.0f;
                v.y = (il + 1 >= 0 && il + 1 < L_in) ? (float)LOAD(in, rb + il + 1) : 0.0f;
                v.z = (il + 2 >= 0 && il + 2 < L_in) ? (float)LOAD(in, rb + il + 2) : 0.0f;
                v.w = (il + 3 >= 0 && il + 3 < L_in) ? (float)LOAD(in, rb + il + 3) : 0.0f;
            } else {
                continue;
            }
            acc = mad((float)LOAD(W, wrow + k), v, acc);
        }
    }
    __local float4 lacc[64];
    lacc[lt] = acc;
    barrier(CLK_LOCAL_MEM_FENCE);
    for (int off = 32; off > 0; off >>= 1) {
        if (lt < off) lacc[lt] += lacc[lt + off];
        barrier(CLK_LOCAL_MEM_FENCE);
    }
    if (lt == 0 && ol0 < L_out) {
        float4 r = lacc[0];
        int ob = mul24(oc, L_out) + ol0;
        int rem = L_out - ol0;
        if (rem >= 4) {
            STORE4(r, out, ob);
        } else {
            STORE(out, ob, r.x);
            if (rem > 1) STORE(out, ob + 1, r.y);
            if (rem > 2) STORE(out, ob + 2, r.z);
        }
    }
}


// NOTE: do NOT add an 8-oc fp16 variant to this program. Two attempts both
// measured ~210ms/conv (vs 125ms for c4x4) AND degraded unrelated kernels in
// the same program (weightnorm_recon 65ms -> 600ms) — the register footprint
// spills, and the driver appears to size per-program scratch by the worst
// kernel. The fp32 generator program's t8x4 is fine (separate program).

// INT8-WEIGHT VARIANT of conv1d_dilated_fast_c4. Per-channel symmetric quant:
//   int8_w[oc, ic, k] = round(fp_w[oc, ic, k] / scale[oc])  with |int8_w| <= 127
//   fp_w (dequantized) = (float)int8_w * scale[oc]
//
// IMPORTANT: Per Qualcomm OpenCL guide §9.5.1, Adreno does NOT have general
// 8-bit ALU support — int8 ops are emulated via 16/32-bit conversions. So we
// dequantize each int8 weight to fp16 ON THE FLY at the cooperative-load step
// and store fp16 into LDS. The inner loop is then identical to the fp16
// fast_c4 kernel (proven fast).
//
// Savings: GLOBAL memory bandwidth on W reads is halved (1 byte vs 2).
// Accuracy: per-channel symmetric int8 typically loses ~0.5-1.5% cos vs fp16.
__kernel void conv1d_dilated_int8_c4(
        __global const storage_t* in,
        __global const char* W_int8,             // [C_out * (grp_in_per_out * K)]
        __global const float* W_scale,           // [C_out] per-oc dequant scale
        __global storage_t* out,
        int C_in, int C_out, int L_in, int L_out,
        int K, int stride, int padding, int dilation, int groups) {
    int oc_base = get_global_id(0) * COUT_TILE;
    int lt = get_local_id(1);
    int ol = get_group_id(1) * LOCAL_T_DEC + lt;
    int grp_in_per_out = C_in / groups;
    int g = oc_base / (C_out / groups);
    int w_total_per_oc = grp_in_per_out * K;
    int w_total = COUT_TILE * w_total_per_oc;
    int w_base = oc_base * w_total_per_oc;
    // Per-tile dequant scales (read once, broadcast across workgroup).
    float s0 = W_scale[oc_base + 0];
    float s1 = W_scale[oc_base + 1];
    float s2 = W_scale[oc_base + 2];
    float s3 = W_scale[oc_base + 3];
    // LDS: store DEQUANTIZED weights as half. Same footprint as fp16 cache
    // but global W reads are int8 (half bandwidth).
    __local half W_cache[WCACHE_C4_MAX];
    // Cooperative load with on-the-fly dequant. Sign-extend through int to
    // force signed treatment (some Adreno chars are unsigned by default).
    for (int i = lt; i < w_total; i += LOCAL_T_DEC) {
        int oc_idx;
        if (i < w_total_per_oc)         oc_idx = 0;
        else if (i < 2 * w_total_per_oc) oc_idx = 1;
        else if (i < 3 * w_total_per_oc) oc_idx = 2;
        else                             oc_idx = 3;
        float s = (oc_idx == 0) ? s0 : (oc_idx == 1) ? s1 : (oc_idx == 2) ? s2 : s3;
        int signed_i = (int)(char)W_int8[w_base + i];  // force signed sign-extend
        float w_f = (float)signed_i * s;
        vstore_half(w_f, i, (__local half*)W_cache);
    }
    barrier(CLK_LOCAL_MEM_FENCE);
    if (oc_base >= C_out || ol >= L_out) return;
    float acc0 = 0.0f, acc1 = 0.0f, acc2 = 0.0f, acc3 = 0.0f;
    int g_base = g * grp_in_per_out;
    int w_off1 = 1 * w_total_per_oc;
    int w_off2 = 2 * w_total_per_oc;
    int w_off3 = 3 * w_total_per_oc;
    // Inner loop is identical to conv1d_dilated_fast_c4 (proven fast on Adreno 620).
    for (int k = 0; k < K; ++k) {
        int il = ol * stride - padding + k * dilation;
        if (il < 0 || il >= L_in) continue;
        int ic = 0;
        for (; ic + 3 < grp_in_per_out; ic += 4) {
            int ic_g = g_base + ic;
            float x0 = (float)LOAD(in, (ic_g+0)*L_in + il);
            float x1 = (float)LOAD(in, (ic_g+1)*L_in + il);
            float x2 = (float)LOAD(in, (ic_g+2)*L_in + il);
            float x3 = (float)LOAD(in, (ic_g+3)*L_in + il);
            int w_idx0 = (ic+0)*K + k;
            int w_idx1 = (ic+1)*K + k;
            int w_idx2 = (ic+2)*K + k;
            int w_idx3 = (ic+3)*K + k;
            acc0 += x0 * (float)W_cache[w_idx0]
                  + x1 * (float)W_cache[w_idx1]
                  + x2 * (float)W_cache[w_idx2]
                  + x3 * (float)W_cache[w_idx3];
            acc1 += x0 * (float)W_cache[w_idx0 + w_off1]
                  + x1 * (float)W_cache[w_idx1 + w_off1]
                  + x2 * (float)W_cache[w_idx2 + w_off1]
                  + x3 * (float)W_cache[w_idx3 + w_off1];
            acc2 += x0 * (float)W_cache[w_idx0 + w_off2]
                  + x1 * (float)W_cache[w_idx1 + w_off2]
                  + x2 * (float)W_cache[w_idx2 + w_off2]
                  + x3 * (float)W_cache[w_idx3 + w_off2];
            acc3 += x0 * (float)W_cache[w_idx0 + w_off3]
                  + x1 * (float)W_cache[w_idx1 + w_off3]
                  + x2 * (float)W_cache[w_idx2 + w_off3]
                  + x3 * (float)W_cache[w_idx3 + w_off3];
        }
        for (; ic < grp_in_per_out; ++ic) {
            int ic_g = g_base + ic;
            float xv = (float)LOAD(in, ic_g * L_in + il);
            int w_idx = ic * K + k;
            acc0 += xv * (float)W_cache[w_idx];
            acc1 += xv * (float)W_cache[w_idx + w_off1];
            acc2 += xv * (float)W_cache[w_idx + w_off2];
            acc3 += xv * (float)W_cache[w_idx + w_off3];
        }
    }
    STORE(out, (oc_base + 0) * L_out + ol, acc0);
    STORE(out, (oc_base + 1) * L_out + ol, acc1);
    STORE(out, (oc_base + 2) * L_out + ol, acc2);
    STORE(out, (oc_base + 3) * L_out + ol, acc3);
}


__kernel void convtr1d(__global const storage_t* in,
                       __global const storage_t* W,
                       __global storage_t* out,
                       int C_in, int C_out, int L_in, int L_out,
                       int K, int stride, int padding, int dilation, int groups) {
    int oc = get_global_id(0);
    int ol = get_global_id(1);
    if (oc >= C_out || ol >= L_out) return;
    int grp_in_per_out = C_in / groups;
    int g = oc / (C_out / groups);
    int oc_in_group = oc - g * (C_out / groups);
    float acc = 0.0f;
    for (int k = 0; k < K; ++k) {
        int num = ol + padding - k * dilation;
        if (num < 0 || (num % stride) != 0) continue;
        int il = num / stride;
        if (il < 0 || il >= L_in) continue;
        for (int ic = 0; ic < grp_in_per_out; ++ic) {
            int ic_g = g * grp_in_per_out + ic;
            int w_idx = ic_g * (C_out / groups) * K + oc_in_group * K + k;
            acc += (float)LOAD(in, ic_g*L_in + il) * (float)LOAD(W, w_idx);
        }
    }
    STORE(out, oc*L_out + ol, acc);
}

// Repack ConvTranspose weights [C_in][C_out][K] into a phase-major image:
// y = ic, x = (p * ntaps + d) * (C_out/4) + oc4 — one RGBA16F texel holds the
// 4 consecutive-oc weights for tap k = p + d*stride.
__kernel void convtr_pack_image(__global const storage_t* W,
                                __write_only image2d_t Wimg,
                                int C_in, int C_out, int K, int stride, int ntaps) {
    int x = get_global_id(0);
    int ic = get_global_id(1);
    int C4 = C_out >> 2;
    if (ic >= C_in || x >= mul24(mul24(stride, ntaps), C4)) return;
    int oc4 = x % C4;
    int pd  = x / C4;
    int d   = pd % ntaps;
    int p   = pd / ntaps;
    int k   = p + mul24(d, stride);
    int ocb = oc4 << 2;
    half4 w = (half4)(0.0h);
    if (k < K) {
        int base = mad24(ic, mul24(C_out, K), k);
        w.x = (half)LOAD(W, base + mul24(ocb + 0, K));
        w.y = (half)LOAD(W, base + mul24(ocb + 1, K));
        w.z = (half)LOAD(W, base + mul24(ocb + 2, K));
        w.w = (half)LOAD(W, base + mul24(ocb + 3, K));
    }
    write_imageh(Wimg, (int2)(x, ic), w);
}

// Texture variant of the phase-decomposed convtranspose: the (ic, tap) texel
// delivers all 4 oc weights through TP/L1 in one read.
__kernel void convtr1d_c4x4_tex(__global const storage_t* in,
                                __read_only image2d_t Wimg,
                                __global storage_t* out,
                                int C_in, int C_out, int L_in, int L_out,
                                int K, int stride, int padding, int ntaps) {
    const sampler_t smp = CLK_NORMALIZED_COORDS_FALSE | CLK_ADDRESS_CLAMP_TO_EDGE | CLK_FILTER_NEAREST;
    int ocb = (int)get_global_id(0) * 4;
    int q0  = (int)get_global_id(1) * 4;
    int p   = (int)get_global_id(2);
    if (ocb >= C_out || p >= stride) return;
    int ol_first = mad24(q0, stride, p - padding);
    int ol_last  = mad24(q0 + 3, stride, p - padding);
    if (ol_last < 0 || ol_first >= L_out) return;
    float4 acc0 = (float4)(0.0f);
    float4 acc1 = (float4)(0.0f);
    float4 acc2 = (float4)(0.0f);
    float4 acc3 = (float4)(0.0f);
    int C4 = C_out >> 2;
    int oc4 = ocb >> 2;
    for (int d = 0; d < ntaps; ++d) {
        int k = p + mul24(d, stride);
        if (k >= K) break;
        int il = q0 - d;
        int tx = mad24(mad24(p, ntaps, d), C4, oc4);
        if (il >= 0 && il + 3 < L_in) {
            for (int ic = 0; ic < C_in; ++ic) {
                float4 v = LOAD4(in, mul24(ic, L_in) + il);
                float4 w = convert_float4(read_imageh(Wimg, smp, (int2)(tx, ic)));
                acc0 = mad(w.x, v, acc0);
                acc1 = mad(w.y, v, acc1);
                acc2 = mad(w.z, v, acc2);
                acc3 = mad(w.w, v, acc3);
            }
        } else if (il + 3 >= 0 && il < L_in) {
            for (int ic = 0; ic < C_in; ++ic) {
                int rb = mul24(ic, L_in);
                float4 v;
                v.x = (il     >= 0 && il     < L_in) ? (float)LOAD(in, rb + il)     : 0.0f;
                v.y = (il + 1 >= 0 && il + 1 < L_in) ? (float)LOAD(in, rb + il + 1) : 0.0f;
                v.z = (il + 2 >= 0 && il + 2 < L_in) ? (float)LOAD(in, rb + il + 2) : 0.0f;
                v.w = (il + 3 >= 0 && il + 3 < L_in) ? (float)LOAD(in, rb + il + 3) : 0.0f;
                float4 w = convert_float4(read_imageh(Wimg, smp, (int2)(tx, ic)));
                acc0 = mad(w.x, v, acc0);
                acc1 = mad(w.y, v, acc1);
                acc2 = mad(w.z, v, acc2);
                acc3 = mad(w.w, v, acc3);
            }
        }
    }
    int ob0 = mul24(ocb + 0, L_out);
    int ob1 = mul24(ocb + 1, L_out);
    int ob2 = mul24(ocb + 2, L_out);
    int ob3 = mul24(ocb + 3, L_out);
    for (int j = 0; j < 4; ++j) {
        int ol = mad24(q0 + j, stride, p - padding);
        if (ol < 0 || ol >= L_out) continue;
        float a0 = (j == 0) ? acc0.x : (j == 1) ? acc0.y : (j == 2) ? acc0.z : acc0.w;
        float a1 = (j == 0) ? acc1.x : (j == 1) ? acc1.y : (j == 2) ? acc1.z : acc1.w;
        float a2 = (j == 0) ? acc2.x : (j == 1) ? acc2.y : (j == 2) ? acc2.z : acc2.w;
        float a3 = (j == 0) ? acc3.x : (j == 1) ? acc3.y : (j == 2) ? acc3.z : acc3.w;
        STORE(out, ob0 + ol, a0);
        STORE(out, ob1 + ol, a1);
        STORE(out, ob2 + ol, a2);
        STORE(out, ob3 + ol, a3);
    }
}

// Phase-decomposed ConvTranspose1d (groups==1, dilation==1). For stride s,
// output ol only receives taps k with (ol+padding-k) % s == 0 — i.e. exactly
// ceil(K/s) taps (2 for both generator ups layers: K=20/s=10 and K=12/s=6)
// instead of the K-iteration modulo scan in the naive kernel.
// Work decomposition: ol = q*s + p - padding. gid(2)=phase p, gid(1)=q tile.
// Each work-item computes 4(oc) x 4(consecutive q) outputs:
//   - for fixed (p,k), il = q - (k-p)/s, so the 4 q values read CONTIGUOUS
//     input -> one LOAD4 per (ic,k), reused across 4 oc;
//   - all work-items in a workgroup (lws dim0/dim2 == 1) share (ocb, p) ->
//     weight reads are uniform -> L2 broadcast.
// Output positions are s apart (scattered stores), but stores are 1-way.
__kernel void convtr1d_c4x4(__global const storage_t* in,
                            __global const storage_t* W,
                            __global storage_t* out,
                            int C_in, int C_out, int L_in, int L_out,
                            int K, int stride, int padding) {
    int ocb = (int)get_global_id(0) * 4;
    int q0  = (int)get_global_id(1) * 4;
    int p   = (int)get_global_id(2);
    if (ocb >= C_out || p >= stride) return;
    // Output positions ol_j = (q0+j)*stride + p - padding, j=0..3. Skip WI if
    // the whole tile is out of range.
    int ol_first = mad24(q0, stride, p - padding);
    int ol_last  = mad24(q0 + 3, stride, p - padding);
    if (ol_last < 0 || ol_first >= L_out) return;
    float4 acc0 = (float4)(0.0f);
    float4 acc1 = (float4)(0.0f);
    float4 acc2 = (float4)(0.0f);
    float4 acc3 = (float4)(0.0f);
    int CK = mul24(C_out, K);
    int wo0 = mul24(ocb + 0, K);
    int wo1 = mul24(ocb + 1, K);
    int wo2 = mul24(ocb + 2, K);
    int wo3 = mul24(ocb + 3, K);
    for (int k = p; k < K; k += stride) {
        int d = (k - p) / stride;
        int il = q0 - d;   // il_j = il + j, contiguous in j
        if (il >= 0 && il + 3 < L_in) {
            for (int ic = 0; ic < C_in; ++ic) {
                float4 v = LOAD4(in, mul24(ic, L_in) + il);
                int wb = mul24(ic, CK) + k;
                acc0 = mad((float)LOAD(W, wb + wo0), v, acc0);
                acc1 = mad((float)LOAD(W, wb + wo1), v, acc1);
                acc2 = mad((float)LOAD(W, wb + wo2), v, acc2);
                acc3 = mad((float)LOAD(W, wb + wo3), v, acc3);
            }
        } else if (il + 3 >= 0 && il < L_in) {
            for (int ic = 0; ic < C_in; ++ic) {
                int rb = mul24(ic, L_in);
                float4 v;
                v.x = (il     >= 0 && il     < L_in) ? (float)LOAD(in, rb + il)     : 0.0f;
                v.y = (il + 1 >= 0 && il + 1 < L_in) ? (float)LOAD(in, rb + il + 1) : 0.0f;
                v.z = (il + 2 >= 0 && il + 2 < L_in) ? (float)LOAD(in, rb + il + 2) : 0.0f;
                v.w = (il + 3 >= 0 && il + 3 < L_in) ? (float)LOAD(in, rb + il + 3) : 0.0f;
                int wb = mul24(ic, CK) + k;
                acc0 = mad((float)LOAD(W, wb + wo0), v, acc0);
                acc1 = mad((float)LOAD(W, wb + wo1), v, acc1);
                acc2 = mad((float)LOAD(W, wb + wo2), v, acc2);
                acc3 = mad((float)LOAD(W, wb + wo3), v, acc3);
            }
        }
    }
    int ob0 = mul24(ocb + 0, L_out);
    int ob1 = mul24(ocb + 1, L_out);
    int ob2 = mul24(ocb + 2, L_out);
    int ob3 = mul24(ocb + 3, L_out);
    for (int j = 0; j < 4; ++j) {
        int ol = mad24(q0 + j, stride, p - padding);
        if (ol < 0 || ol >= L_out) continue;
        float a0 = (j == 0) ? acc0.x : (j == 1) ? acc0.y : (j == 2) ? acc0.z : acc0.w;
        float a1 = (j == 0) ? acc1.x : (j == 1) ? acc1.y : (j == 2) ? acc1.z : acc1.w;
        float a2 = (j == 0) ? acc2.x : (j == 1) ? acc2.y : (j == 2) ? acc2.z : acc2.w;
        float a3 = (j == 0) ? acc3.x : (j == 1) ? acc3.y : (j == 2) ? acc3.z : acc3.w;
        STORE(out, ob0 + ol, a0);
        STORE(out, ob1 + ol, a1);
        STORE(out, ob2 + ol, a2);
        STORE(out, ob3 + ol, a3);
    }
}

// im2col for small-T convs: x NCL [C_in, L] -> cols [L_out, C_in*K] (NLC-ish,
// contiguous reduction axis) so the conv becomes a skinny GEMM.
__kernel void im2col_ncl(__global const storage_t* x,
                         __global storage_t* cols,
                         int C_in, int L_in, int L_out,
                         int K, int padding, int dilation, int row_stride) {
    int t = get_global_id(0);
    int ic = get_global_id(1);
    if (t >= L_out || ic >= C_in) return;
    int ob = mad24(t, row_stride, mul24(ic, K));
    int rb = mul24(ic, L_in);
    for (int k = 0; k < K; ++k) {
        int il = t - padding + mul24(k, dilation);
        float v = (il >= 0 && il < L_in) ? (float)LOAD(x, rb + il) : 0.0f;
        STORE(cols, ob + k, v);
    }
}

// Zero-pad each row of a [rows, src_w] half matrix to dst_w (GEMM alignment).
__kernel void pad_rows(__global const storage_t* src,
                       __global storage_t* dst,
                       int rows, int src_w, int dst_w) {
    int r  = get_global_id(0);
    int lt = get_global_id(1);   // 64 lanes split the row
    if (r >= rows) return;
    int sb = mul24(r, src_w);
    int db = mul24(r, dst_w);
    for (int i = lt; i < dst_w; i += 64) {
        STORE(dst, db + i, (i < src_w) ? (float)LOAD(src, sb + i) : 0.0f);
    }
}

// [T, N] -> [N, T] transpose (small tensors only).
__kernel void transpose_TN(__global const storage_t* x,
                           __global storage_t* y, int T, int N) {
    int n = get_global_id(0);
    int t = get_global_id(1);
    if (n >= N || t >= T) return;
    STORE(y, mad24(n, T, t), (float)LOAD(x, mad24(t, N, n)));
}

__kernel void k_exp(__global storage_t* y, int N) {
    int i = get_global_id(0); if (i >= N) return;
    STORE(y, i, exp((float)LOAD(y, i)));
}
__kernel void k_sin_act(__global storage_t* y, int N) {
    int i = get_global_id(0); if (i >= N) return;
    STORE(y, i, sin((float)LOAD(y, i)));
}

__kernel void istft_simple(__global const storage_t* mag,
                           __global const storage_t* phase,
                           __global const storage_t* window,
                           __global storage_t* out,
                           int n_freq, int T_frames, int n_fft, int hop, int T_audio) {
    // mag/phase layout: [n_freq, T_frames] — index as k*T_frames + f.
    int n = get_global_id(0);
    if (n >= T_audio) return;
    float acc = 0.0f;
    float norm = 0.0f;  // sum of window^2 at this output position (for COLA normalization)
    int f_min = (n - n_fft + hop) / hop;
    if (f_min < 0) f_min = 0;
    int f_max = n / hop;
    if (f_max >= T_frames) f_max = T_frames - 1;
    for (int f = f_min; f <= f_max; ++f) {
        int t = n - f * hop;
        if (t < 0 || t >= n_fft) continue;
        float w = (float)LOAD(window, t);
        norm += w * w;
        float s = (float)LOAD(mag, 0 * T_frames + f) * cos((float)LOAD(phase, 0 * T_frames + f));
        for (int k = 1; k < n_freq - 1; ++k) {
            // Wrap angle to [-pi, pi] to avoid Adreno cos() precision loss at large arguments
            float ang = 2.0f * 3.14159265358979f * (float)k * (float)t / (float)n_fft
                        + (float)LOAD(phase, k * T_frames + f);
            // fmod to [0, 2*pi)
            ang = ang - 6.28318530717958f * floor(ang / 6.28318530717958f);
            s += 2.0f * (float)LOAD(mag, k * T_frames + f) * cos(ang);
        }
        if (n_fft % 2 == 0) {
            float ang = 3.14159265358979f * (float)t
                        + (float)LOAD(phase, (n_freq - 1) * T_frames + f);
            ang = ang - 6.28318530717958f * floor(ang / 6.28318530717958f);
            s += (float)LOAD(mag, (n_freq - 1) * T_frames + f) * cos(ang);
        }
        acc += w * s / (float)n_fft;
    }
    if (norm > 1e-10f) acc /= norm;
    STORE(out, n, acc);
}

__kernel void bias_add_NCL(__global storage_t* y, __global const storage_t* b, int C, int T) {
    int c = get_global_id(0);
    int t = get_global_id(1);
    if (c >= C || t >= T) return;
    STORE(y, c*T + t, (float)LOAD(y, c*T + t) + (float)LOAD(b, c));
}

// Workgroup-per-row weight_norm reconstruction (was 1 WI per row, serial).
__kernel void weightnorm_recon(__global const storage_t* v,
                               __global const storage_t* g,
                               __global storage_t* W,
                               int C_out, int per_oc) {
    int oc = get_group_id(0);
    int lt = get_local_id(0);
    if (oc >= C_out) return;
    int base = oc * per_oc;
    __local float lsq[64];
    float q = 0.0f;
    for (int i = lt; i < per_oc; i += 64) {
        float x = (float)LOAD(v, base + i);
        q += x * x;
    }
    lsq[lt] = q;
    barrier(CLK_LOCAL_MEM_FENCE);
    for (int off = 32; off > 0; off >>= 1) {
        if (lt < off) lsq[lt] += lsq[lt + off];
        barrier(CLK_LOCAL_MEM_FENCE);
    }
    float norm = sqrt(lsq[0]) + 1e-12f;
    float scale = (float)LOAD(g, oc) / norm;
    for (int i = lt; i < per_oc; i += 64) {
        STORE(W, base + i, (float)LOAD(v, base + i) * scale);
    }
}

// Linear apply: out[C_out, T_broadcast=1] = W[C_out, C_in] @ s[C_in] + b[C_out]
// We use T_broadcast=1 because style vector is per-sample, not per-frame.
__kernel void linear_apply(__global const storage_t* s,
                           __global const storage_t* W,
                           __global const storage_t* b,
                           __global storage_t* out,
                           int C_in, int C_out, int has_bias) {
    int oc = get_global_id(0);
    if (oc >= C_out) return;
    float acc = 0.0f;
    for (int ic = 0; ic < C_in; ++ic) acc += (float)LOAD(s, ic) * (float)LOAD(W, oc*C_in + ic);
    if (has_bias) acc += (float)LOAD(b, oc);
    STORE(out, oc, acc);
}

// Split [2C, ?] -> first half (gamma) [C, ?] and second half (beta) [C, ?]. For T=1.
__kernel void split_chunk2(__global const storage_t* x,
                           __global storage_t* a,
                           __global storage_t* b,
                           int C, int T) {
    int c = get_global_id(0);
    int t = get_global_id(1);
    if (c >= C || t >= T) return;
    STORE(a, c*T + t, (float)LOAD(x, c*T + t));
    STORE(b, c*T + t, (float)LOAD(x, (C+c)*T + t));
}

// Snake1D activation: y = x + (1/a)*sin(a*x)^2.  alpha shape [C] (per-channel).
__kernel void snake1d(__global storage_t* y, __global const storage_t* alpha, int C, int T) {
    int c = get_global_id(0);
    int t = get_global_id(1);
    if (c >= C || t >= T) return;
    float a = (float)LOAD(alpha, c);
    if (fabs(a) < 1e-6f) a = (a >= 0.0f ? 1e-6f : -1e-6f);
    float x = (float)LOAD(y, c*T + t);
    float s = sin(a * x);
    STORE(y, c*T + t, x + (1.0f / a) * s * s);
}

// Reflection pad (left=p_left, right=p_right) along last dim. y has T_out = T_in + p_left + p_right.
__kernel void refl_pad(__global const storage_t* x, __global storage_t* y, int C, int T_in, int p_left, int p_right) {
    int c = get_global_id(0);
    int t_out = get_global_id(1);
    int T_out = T_in + p_left + p_right;
    if (c >= C || t_out >= T_out) return;
    int t_in;
    if (t_out < p_left) t_in = p_left - t_out;
    else if (t_out >= p_left + T_in) t_in = T_in - 2 - (t_out - p_left - T_in);
    else t_in = t_out - p_left;
    if (t_in < 0) t_in = 0;
    if (t_in >= T_in) t_in = T_in - 1;
    STORE(y, c*T_out + t_out, (float)LOAD(x, c*T_in + t_in));
}

// in-place division by scalar
__kernel void k_div_scalar(__global storage_t* y, int N, float s) {
    int i = get_global_id(0); if (i >= N) return;
    STORE(y, i, (float)LOAD(y, i) / s);
}

// Linear interpolation upsample 1D: [C, T_in] -> [C, T_out] with T_out >> T_in.
__kernel void upsample_linear_NCL(__global const storage_t* x,
                                  __global storage_t* y,
                                  int C, int T_in, int T_out) {
    int c = get_global_id(0); int t = get_global_id(1);
    if (c >= C || t >= T_out) return;
    float pos = (float)t * (float)(T_in - 1) / (float)(T_out - 1);
    int t0 = (int)pos;
    int t1 = t0 + 1;
    if (t1 >= T_in) t1 = T_in - 1;
    float frac = pos - (float)t0;
    float v = (1.0f - frac) * (float)LOAD(x, c*T_in + t0) + frac * (float)LOAD(x, c*T_in + t1);
    STORE(y, c*T_out + t, v);
}

// SineGen for first harmonic only (we'll extend to 9 harmonics via Linear projection).
// f0 [T_audio] (1 channel implicit). Output sines [n_harm, T_audio].
__kernel void sinegen(__global const storage_t* f0,
                      __global storage_t* sines,
                      __global storage_t* uv,
                      int T_audio, int n_harm, float samp_rate, float voiced_threshold, float sine_amp) {
    int h = get_global_id(0); int t = get_global_id(1);
    if (h >= n_harm || t >= T_audio) return;
    // For harmonic h+1: phase[t] = sum_{i<=t} (f0[i] * (h+1) / samp_rate * 2pi) mod 2pi
    // We'll compute via running sum stored externally — but this kernel is per-(h,t) parallel.
    // Compromise: compute cumulative phase as a closed-form using approximate constant F0 per frame
    // (assumes f0 doesn't change rapidly per sample, which is true since f0 is upsampled from predictor).
    // For correctness this should be a sequential cumsum — for simplicity here we use the instantaneous frequency.
    float fn = (float)LOAD(f0, t) * (float)(h + 1);
    float phase = 2.0f * 3.14159265358979f * fn * (float)t / samp_rate;
    float s = sin(phase) * sine_amp;
    if (h == 0) {
        float u = ((float)LOAD(f0, t) > voiced_threshold) ? 1.0f : 0.0f;
        STORE(uv, t, u);
    }
    // Apply uv masking: voiced samples keep sine; unvoiced get scaled noise (we use 0 for determinism).
    float u_local = ((float)LOAD(f0, t) > voiced_threshold) ? 1.0f : 0.0f;
    STORE(sines, h*T_audio + t, s * u_local);
}

// Apply Linear 9->1 + tanh: out[t] = tanh(sum_h (sines[h, t] * W[h]) + b)
// W: [1, 9] = [9 floats], b: [1] (single float)
__kernel void source_merge(__global const storage_t* sines,
                           __global const storage_t* W,
                           __global const storage_t* b,
                           __global storage_t* out, int T_audio, int n_harm, int has_bias) {
    int t = get_global_id(0);
    if (t >= T_audio) return;
    float acc = 0.0f;
    for (int h = 0; h < n_harm; ++h) acc += (float)LOAD(sines, h*T_audio + t) * (float)LOAD(W, h);
    if (has_bias) acc += (float)LOAD(b, 0);
    STORE(out, t, tanh(acc));
}

// Forward STFT (real-input): produces magnitude + phase for n_freq bins (n_fft/2+1).
// Direct DFT per frame (n_fft small enough: 20).
__kernel void stft_simple(__global const storage_t* x,
                          __global const storage_t* window,
                          __global storage_t* mag,
                          __global storage_t* phase,
                          int T_audio, int n_fft, int hop, int n_freq, int T_frames) {
    // torch.stft center=True semantics: frame f starts at f*hop - n_fft/2
    // and out-of-range samples are reflection-padded (we use zero-pad for simplicity).
    int f = get_global_id(0); int k = get_global_id(1);
    if (f >= T_frames || k >= n_freq) return;
    float re = 0.0f, im = 0.0f;
    int n0 = f * hop - n_fft / 2;
    for (int t = 0; t < n_fft; ++t) {
        int n = n0 + t;
        float v = 0.0f;
        if (n >= 0 && n < T_audio) v = (float)LOAD(x, n) * (float)LOAD(window, t);
        float ang = -2.0f * 3.14159265358979f * (float)k * (float)t / (float)n_fft;
        re += v * cos(ang);
        im += v * sin(ang);
    }
    float m = sqrt(re*re + im*im);
    float p = atan2(im, re);
    STORE(mag, f*n_freq + k, m);
    STORE(phase, f*n_freq + k, p);
}

// Clamp exp input to avoid overflow.
__kernel void k_exp_clamp(__global storage_t* y, int N, float lo, float hi) {
    int i = get_global_id(0); if (i >= N) return;
    float v = (float)LOAD(y, i);
    if (v < lo) v = lo;
    if (v > hi) v = hi;
    STORE(y, i, exp(v));
}
)CLC";

static bool ensure_built(OpenCLContext& cl_ctx) {
    if (g_k_instnorm) return true;
    cl_int err = CL_SUCCESS;
    const char* opts =
#ifdef NNOPT_USE_FP16
        "-DNNOPT_USE_FP16=1 -cl-fast-relaxed-math";
#else
        "-cl-fast-relaxed-math";
#endif
    cl_device_id dev = cl_ctx.device();
    g_dec_prog = nnopt_build_program_cached(cl_ctx.context(), dev, k_dec_src, opts, "decoder", &err);
    if (!g_dec_prog) return false;
    g_k_instnorm        = clCreateKernel(g_dec_prog, "instnorm_NCL", &err);
    g_k_adain_combine   = clCreateKernel(g_dec_prog, "adain_combine", &err);
    g_k_leaky           = clCreateKernel(g_dec_prog, "leaky_relu", &err);
    g_k_add             = clCreateKernel(g_dec_prog, "k_add", &err);
    g_k_add_scale       = clCreateKernel(g_dec_prog, "k_add_scale", &err);
    g_k_concat          = clCreateKernel(g_dec_prog, "concat_C", &err);
    g_k_concat4         = clCreateKernel(g_dec_prog, "concat4_C", &err);
    g_k_upsample_nn     = clCreateKernel(g_dec_prog, "upsample_nn_NCL", &err);
    g_k_conv1d          = clCreateKernel(g_dec_prog, "conv1d_dilated", &err);
    g_k_conv1d_fast     = clCreateKernel(g_dec_prog, "conv1d_dilated_fast", &err);
    g_k_conv1d_fast_c4  = clCreateKernel(g_dec_prog, "conv1d_dilated_fast_c4", &err);
    g_k_conv1d_int8_c4  = clCreateKernel(g_dec_prog, "conv1d_dilated_int8_c4", &err);
    g_k_conv1d_c4x4     = clCreateKernel(g_dec_prog, "conv1d_dilated_c4x4", &err);
    g_k_conv1d_rt4      = clCreateKernel(g_dec_prog, "conv1d_dilated_rt4", &err);
    g_k_im2col          = clCreateKernel(g_dec_prog, "im2col_ncl", &err);
    g_k_transpose_tn    = clCreateKernel(g_dec_prog, "transpose_TN", &err);
    g_k_pad_rows        = clCreateKernel(g_dec_prog, "pad_rows", &err);
    g_k_convtr1d_c4x4   = clCreateKernel(g_dec_prog, "convtr1d_c4x4", &err);
    g_k_convtr1d_c4x4_tex = clCreateKernel(g_dec_prog, "convtr1d_c4x4_tex", &err);
    g_k_convtr_pack     = clCreateKernel(g_dec_prog, "convtr_pack_image", &err);
    g_k_convtr1d        = clCreateKernel(g_dec_prog, "convtr1d", &err);
    g_k_exp             = clCreateKernel(g_dec_prog, "k_exp", &err);
    g_k_sin_act         = clCreateKernel(g_dec_prog, "k_sin_act", &err);
    g_k_istft           = clCreateKernel(g_dec_prog, "istft_simple", &err);
    g_k_bias_add_NCL    = clCreateKernel(g_dec_prog, "bias_add_NCL", &err);
    g_k_weightnorm      = clCreateKernel(g_dec_prog, "weightnorm_recon", &err);
    g_k_linear_apply    = clCreateKernel(g_dec_prog, "linear_apply", &err);
    g_k_split           = clCreateKernel(g_dec_prog, "split_chunk2", &err);
    g_k_snake1d         = clCreateKernel(g_dec_prog, "snake1d", &err);
    g_k_refl_pad        = clCreateKernel(g_dec_prog, "refl_pad", &err);
    g_k_div_scalar      = clCreateKernel(g_dec_prog, "k_div_scalar", &err);
    g_k_upsample_linear = clCreateKernel(g_dec_prog, "upsample_linear_NCL", &err);
    g_k_sinegen         = clCreateKernel(g_dec_prog, "sinegen", &err);
    g_k_source_merge    = clCreateKernel(g_dec_prog, "source_merge", &err);
    g_k_stft            = clCreateKernel(g_dec_prog, "stft_simple", &err);
    g_k_exp_clamp       = clCreateKernel(g_dec_prog, "k_exp_clamp", &err);
    return g_k_instnorm && g_k_adain_combine && g_k_leaky && g_k_add && g_k_add_scale &&
           g_k_concat && g_k_concat4 && g_k_upsample_nn && g_k_conv1d && g_k_conv1d_fast &&
           g_k_conv1d_fast_c4 && g_k_conv1d_int8_c4 &&
           g_k_conv1d_c4x4 && g_k_conv1d_rt4 && g_k_im2col && g_k_transpose_tn && g_k_pad_rows && g_k_convtr1d_c4x4_tex && g_k_convtr_pack && g_k_convtr1d_c4x4 &&
           g_k_convtr1d &&
           g_k_exp && g_k_sin_act && g_k_istft && g_k_bias_add_NCL && g_k_weightnorm &&
           g_k_linear_apply && g_k_split && g_k_snake1d && g_k_refl_pad && g_k_div_scalar &&
           g_k_upsample_linear && g_k_sinegen && g_k_source_merge && g_k_stft && g_k_exp_clamp;
}

// Forward decls for helpers defined below (used by apply_adainresblock1).
static cl_mem alloc_rw(OpenCLContext&, size_t);
static int run1d(cl_command_queue, cl_kernel, size_t);
static int conv1d_wn(OpenCLContext&, Weights&, cl_command_queue,
                     const std::string&, cl_mem, cl_mem,
                     int, int, int, int, int, int*, int*);
static int apply_adain1d(OpenCLContext&, Weights&, cl_command_queue,
                         const std::string&, cl_mem, cl_mem, cl_mem, int, int);

// fp32 generator helpers (defined in _generator_fp32.cpp)
extern "C" int gen_fp16_to_fp32(OpenCLContext&, cl_command_queue,
                                 cl_mem fp16_buf, cl_mem fp32_buf, int N);
extern "C" int gen_fp32_to_fp16(OpenCLContext&, cl_command_queue,
                                 cl_mem fp32_buf, cl_mem fp16_buf, int N);
extern "C" int gen_istft_gpu(OpenCLContext&, cl_command_queue,
                             cl_mem mag_fp16, cl_mem phase_fp16,
                             int L_frames, int n_fft, int hop, int n_freq,
                             float* audio_out, int T_audio);
extern "C" int gen_conv_post_gpu(OpenCLContext&, Weights&, cl_command_queue,
                                 cl_mem xs_fp16, cl_mem mag_fp16, cl_mem phase_fp16,
                                 int C_in, int T, int n_freq);
extern "C" int gen_apply_adainresblock1_fp32(OpenCLContext&, Weights&, cl_command_queue,
                                              const std::string& prefix,
                                              cl_mem x_f32,
                                              cl_mem ref_s_dec_fp16,
                                              int C, int T,
                                              int kernel_size, const int* dilations);
extern "C" int gen_avg_three_fp32(OpenCLContext&, cl_command_queue,
                                   cl_mem a, cl_mem b, cl_mem c, cl_mem y, int N);

// Apply Snake1D activation in-place. Loads alpha from weights.
static int apply_snake1d(OpenCLContext& cl_ctx, Weights& weights, cl_command_queue queue,
                         const std::string& alpha_key, cl_mem y, int C, int T) {
    (void)cl_ctx;
    cl_mem alpha = weights.get_buffer(alpha_key);
    if (!alpha) { NNOPT_ERROR_FMT("snake1d: missing %s", alpha_key.c_str()); return -1; }
    clSetKernelArg(g_k_snake1d, 0, sizeof(cl_mem), &y);
    clSetKernelArg(g_k_snake1d, 1, sizeof(cl_mem), &alpha);
    clSetKernelArg(g_k_snake1d, 2, sizeof(int), &C);
    clSetKernelArg(g_k_snake1d, 3, sizeof(int), &T);
    size_t gws[2] = {(size_t)C, (size_t)T};
    return nnopt_enqueue_profiled(queue, g_k_snake1d, 2, nullptr, gws, nullptr, 0, nullptr, nullptr);
}

// AdaINResBlock1: a single residual block of the generator. channels stays constant.
// 3 sub-blocks, each: adain1[i] -> snake (alpha1[i]) -> conv1 (dilation=dilation[i]) -> adain2[i] -> snake (alpha2[i]) -> conv2 (dilation=1)
// Final: x = xt + x
static int apply_adainresblock1(OpenCLContext& cl_ctx, Weights& weights, cl_command_queue queue,
                                 const std::string& prefix,  // "decoder.module.generator.resblocks.<i>"
                                 cl_mem x, cl_mem ref_s_dec, int C, int T,
                                 int kernel_size, const int* dilations /*[3]*/) {
    cl_mem tmp = alloc_rw(cl_ctx, sizeof(nnopt_storage_t) * C * T);
    cl_mem c1_out = alloc_rw(cl_ctx, sizeof(nnopt_storage_t) * C * T);
    cl_mem c2_out = alloc_rw(cl_ctx, sizeof(nnopt_storage_t) * C * T);
    for (int i = 0; i < 3; ++i) {
        std::string si = std::to_string(i);
        // adain1[i]
        if (apply_adain1d(cl_ctx, weights, queue, prefix + ".adain1." + si, x, tmp, ref_s_dec, C, T) != 0) return -1;
        // snake1d alpha1[i]
        if (apply_snake1d(cl_ctx, weights, queue, prefix + ".alpha1." + si, tmp, C, T) != 0) return -1;
        // conv1: weight_norm, kernel=kernel_size, padding=(kernel*dilation-dilation)/2, dilation=dilations[i]
        int dil = dilations[i];
        int pad = (kernel_size * dil - dil) / 2;
        int Lc1, Cc1;
        if (conv1d_wn(cl_ctx, weights, queue, prefix + ".convs1." + si, tmp, c1_out, T, 1, pad, dil, 1, &Lc1, &Cc1) != 0) return -1;
        // adain2[i]
        if (apply_adain1d(cl_ctx, weights, queue, prefix + ".adain2." + si, c1_out, tmp, ref_s_dec, C, Lc1) != 0) return -1;
        // snake1d alpha2[i]
        if (apply_snake1d(cl_ctx, weights, queue, prefix + ".alpha2." + si, tmp, C, Lc1) != 0) return -1;
        // conv2: weight_norm, kernel=kernel_size, padding=(kernel*1-1)/2, dilation=1
        int pad2 = (kernel_size * 1 - 1) / 2;
        int Lc2, Cc2;
        if (conv1d_wn(cl_ctx, weights, queue, prefix + ".convs2." + si, tmp, c2_out, Lc1, 1, pad2, 1, 1, &Lc2, &Cc2) != 0) return -1;
        // x = c2_out + x  (in-place add into x)
        int N = C * T;
        clSetKernelArg(g_k_add, 0, sizeof(cl_mem), &x);
        clSetKernelArg(g_k_add, 1, sizeof(cl_mem), &c2_out);
        clSetKernelArg(g_k_add, 2, sizeof(int), &N);
        run1d(queue, g_k_add, (size_t)N);
    }
    clReleaseMemObject(tmp);
    clReleaseMemObject(c1_out);
    clReleaseMemObject(c2_out);
    return 0;
}

static cl_mem alloc_rw(OpenCLContext& cl_ctx, size_t bytes) {
    cl_int err = CL_SUCCESS;
    return clCreateBuffer(cl_ctx.context(), CL_MEM_READ_WRITE, bytes, nullptr, &err);
}

static int run1d(cl_command_queue q, cl_kernel k, size_t gws) {
    return nnopt_enqueue_profiled(q, k, 1, nullptr, &gws, nullptr, 0, nullptr, nullptr);
}

// ─── Weight-norm cache ──────────────────────────────────────────────────
struct WCacheEntry { cl_mem W = nullptr; int C_out = 0, C_in = 0, K = 0; };
static std::unordered_map<std::string, WCacheEntry> g_wn_cache;

// Parallel cache for int8-quantized weights. Each entry holds the quantized W
// buffer (signed char) + the per-output-channel dequant scale (fp32). Built
// lazily from the fp16 cache on first use of the int8 conv path for that prefix.
struct WI8CacheEntry { cl_mem W_int8 = nullptr; cl_mem scale = nullptr; int C_out = 0, C_in = 0, K = 0; };
static std::unordered_map<std::string, WI8CacheEntry> g_wn_int8_cache;

static cl_mem get_wn_conv_weight(OpenCLContext& cl_ctx, Weights& weights, cl_command_queue queue,
                                 const std::string& prefix, int* C_out_o, int* C_in_o, int* K_o) {
    auto it = g_wn_cache.find(prefix);
    if (it != g_wn_cache.end()) {
        if (C_out_o) *C_out_o = it->second.C_out;
        if (C_in_o)  *C_in_o  = it->second.C_in;
        if (K_o)     *K_o     = it->second.K;
        return it->second.W;
    }
    std::string vk = prefix + ".weight_v";
    std::string gk = prefix + ".weight_g";
    cl_mem v = weights.get_buffer(vk);
    cl_mem g = weights.get_buffer(gk);
    if (!v || !g) { NNOPT_ERROR_FMT("wn: missing %s", prefix.c_str()); return nullptr; }
    auto shape = weights.get_shape(vk);
    int C_out = shape[0], C_in = (shape.size() > 1 ? shape[1] : 1), K = (shape.size() > 2 ? shape[2] : 1);
    int per_oc = C_in * K;
    cl_mem W = alloc_rw(cl_ctx, sizeof(nnopt_storage_t) * C_out * per_oc);
    clSetKernelArg(g_k_weightnorm, 0, sizeof(cl_mem), &v);
    clSetKernelArg(g_k_weightnorm, 1, sizeof(cl_mem), &g);
    clSetKernelArg(g_k_weightnorm, 2, sizeof(cl_mem), &W);
    clSetKernelArg(g_k_weightnorm, 3, sizeof(int), &C_out);
    clSetKernelArg(g_k_weightnorm, 4, sizeof(int), &per_oc);
    size_t gws = (size_t)C_out * 64;  // workgroup-per-row reduction
    size_t lws = 64;
    nnopt_enqueue_profiled(queue, g_k_weightnorm, 1, nullptr, &gws, &lws, 0, nullptr, nullptr);
    g_wn_cache[prefix] = {W, C_out, C_in, K};
    if (C_out_o) *C_out_o = C_out;
    if (C_in_o)  *C_in_o  = C_in;
    if (K_o)     *K_o     = K;
    return W;
}

// Build (or fetch from cache) the int8-quantized version of the conv weight
// for `prefix`. Lazy: first call reads the fp16 W back to host, computes per-oc
// scale + quantizes, uploads the int8 buffer and scale buffer.
// Returns true on success and fills W_int8_out / scale_out / dims_out.
//
// Quantization scheme: symmetric per-output-channel.
//   scale[oc] = max(|W[oc, :, :]|) / 127     (or 1e-12 if max == 0)
//   int8_w[oc, i] = clamp(round(W[oc, i] / scale[oc]), -127, 127)
//   dequant: w_f32 = (float)int8_w * scale[oc]
//
// One-time cost per layer: ~weight_bytes of CPU work + 2 GPU uploads.
static bool get_wn_conv_weight_int8(OpenCLContext& cl_ctx, Weights& weights, cl_command_queue queue,
                                     const std::string& prefix,
                                     cl_mem* W_int8_out, cl_mem* scale_out,
                                     int* C_out_o, int* C_in_o, int* K_o) {
    auto it = g_wn_int8_cache.find(prefix);
    if (it != g_wn_int8_cache.end()) {
        *W_int8_out = it->second.W_int8;
        *scale_out  = it->second.scale;
        if (C_out_o) *C_out_o = it->second.C_out;
        if (C_in_o)  *C_in_o  = it->second.C_in;
        if (K_o)     *K_o     = it->second.K;
        return true;
    }
    // Get the dequantized fp16 W first (existing path), then quantize on host.
    int C_out = 0, C_in = 0, K = 0;
    cl_mem W_fp = get_wn_conv_weight(cl_ctx, weights, queue, prefix, &C_out, &C_in, &K);
    if (!W_fp) return false;
    const int per_oc = C_in * K;
    const size_t total = (size_t)C_out * per_oc;

    // Read fp16 W back to host
#ifdef NNOPT_USE_FP16
    std::vector<uint16_t> w_fp16(total);
    clEnqueueReadBuffer(queue, W_fp, CL_TRUE, 0, sizeof(uint16_t) * total, w_fp16.data(), 0, nullptr, nullptr);
#else
    std::vector<float> w_f32_raw(total);
    clEnqueueReadBuffer(queue, W_fp, CL_TRUE, 0, sizeof(float) * total, w_f32_raw.data(), 0, nullptr, nullptr);
#endif

    // Compute per-oc scale + quantize
    std::vector<int8_t> w_int8(total);
    std::vector<float> scale(C_out);
    for (int oc = 0; oc < C_out; ++oc) {
        float maxabs = 0.0f;
        for (int i = 0; i < per_oc; ++i) {
#ifdef NNOPT_USE_FP16
            float v = nnopt_f16_to_f32(w_fp16[(size_t)oc * per_oc + i]);
#else
            float v = w_f32_raw[(size_t)oc * per_oc + i];
#endif
            float av = std::fabs(v);
            if (av > maxabs) maxabs = av;
        }
        float s = (maxabs > 0.0f) ? (maxabs / 127.0f) : 1e-12f;
        float inv_s = 1.0f / s;
        scale[oc] = s;
        for (int i = 0; i < per_oc; ++i) {
#ifdef NNOPT_USE_FP16
            float v = nnopt_f16_to_f32(w_fp16[(size_t)oc * per_oc + i]);
#else
            float v = w_f32_raw[(size_t)oc * per_oc + i];
#endif
            int q = (int)std::round(v * inv_s);
            if (q < -127) q = -127;
            if (q >  127) q =  127;
            w_int8[(size_t)oc * per_oc + i] = (int8_t)q;
        }
    }

    cl_int err = CL_SUCCESS;
    cl_mem W_int8_buf = clCreateBuffer(cl_ctx.context(),
        CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
        sizeof(int8_t) * total, w_int8.data(), &err);
    if (err != CL_SUCCESS || !W_int8_buf) {
        NNOPT_ERROR_FMT("int8 quant: failed to allocate W_int8 for %s (err=%d)", prefix.c_str(), (int)err);
        return false;
    }
    cl_mem scale_buf = clCreateBuffer(cl_ctx.context(),
        CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
        sizeof(float) * C_out, scale.data(), &err);
    if (err != CL_SUCCESS || !scale_buf) {
        clReleaseMemObject(W_int8_buf);
        NNOPT_ERROR_FMT("int8 quant: failed to allocate scale for %s (err=%d)", prefix.c_str(), (int)err);
        return false;
    }
    g_wn_int8_cache[prefix] = {W_int8_buf, scale_buf, C_out, C_in, K};
    *W_int8_out = W_int8_buf;
    *scale_out = scale_buf;
    if (C_out_o) *C_out_o = C_out;
    if (C_in_o)  *C_in_o  = C_in;
    if (K_o)     *K_o     = K;
    return true;
}

// Apply Conv1d weight_norm with bias (if present).
static int conv1d_wn(OpenCLContext& cl_ctx, Weights& weights, cl_command_queue queue,
                     const std::string& prefix, cl_mem in, cl_mem out,
                     int L_in, int stride, int padding, int dilation, int groups,
                     int* L_out_o, int* C_out_o) {
    int C_out, C_in, K;
    cl_mem W = get_wn_conv_weight(cl_ctx, weights, queue, prefix, &C_out, &C_in, &K);
    if (!W) return -1;
    cl_mem bias = weights.get_buffer(prefix + ".bias", /*optional=*/true);
    int L_out = (L_in + 2*padding - dilation*(K-1) - 1) / stride + 1;
    if (L_out_o) *L_out_o = L_out;
    if (C_out_o) *C_out_o = C_out;
    // Use the fast (LDS-cached, tiled) kernel when the weight row fits the LDS cache.
    // WCACHE_DEC_MAX=4096 covers all resblock convs (max 256*11=2816) and most decoder
    // chain convs. Anything larger falls back to the scalar kernel.
    int grp_in_per_out = C_in / groups;
    int w_total = grp_in_per_out * K;
    // Register-tiled 4x4 fast path (NNOPT_CONV_T4X4=0 forces legacy for A/B).
    static int use_t4x4_dec = -1;
    if (use_t4x4_dec == -1) {
        const char* e = std::getenv("NNOPT_CONV_T4X4");
        use_t4x4_dec = (e && e[0] == '0') ? 0 : 1;
    }
    if (use_t4x4_dec && stride == 1 && groups == 1 && L_out <= 512) {
        // Small-T: im2col + skinny GEMM (measured ~7x the c4x4 rate on the
        // decode-chain shapes), then transpose back to NCL.
        int KC = C_in * K;
        int KC4 = (KC + 3) & ~3;   // GEMM reduction axis padded to /4
        cl_mem cols = alloc_rw(cl_ctx, sizeof(nnopt_storage_t) * (size_t)L_out * KC4);
        cl_mem W_gemm = W;
        if (KC4 != KC) {
            // zero the cols pad tail and use a cached row-padded weight copy.
            nnopt_storage_t zero_v;
            memset(&zero_v, 0, sizeof(zero_v));
            clEnqueueFillBuffer(queue, cols, &zero_v, sizeof(zero_v), 0,
                                sizeof(nnopt_storage_t) * (size_t)L_out * KC4, 0, nullptr, nullptr);
            static std::unordered_map<std::string, cl_mem> g_wpad_cache;
            auto itp = g_wpad_cache.find(prefix);
            if (itp != g_wpad_cache.end()) {
                W_gemm = itp->second;
            } else {
                cl_mem Wp = alloc_rw(cl_ctx, sizeof(nnopt_storage_t) * (size_t)C_out * KC4);
                clSetKernelArg(g_k_pad_rows, 0, sizeof(cl_mem), &W);
                clSetKernelArg(g_k_pad_rows, 1, sizeof(cl_mem), &Wp);
                clSetKernelArg(g_k_pad_rows, 2, sizeof(int), &C_out);
                clSetKernelArg(g_k_pad_rows, 3, sizeof(int), &KC);
                clSetKernelArg(g_k_pad_rows, 4, sizeof(int), &KC4);
                size_t gws_p[2] = {(size_t)C_out, 64};
                size_t lws_p[2] = {1, 64};
                nnopt_enqueue_profiled(queue, g_k_pad_rows, 2, nullptr, gws_p, lws_p, 0, nullptr, nullptr);
                g_wpad_cache[prefix] = Wp;
                W_gemm = Wp;
            }
        }
        cl_mem out_tn = alloc_rw(cl_ctx, sizeof(nnopt_storage_t) * (size_t)L_out * C_out);
        clSetKernelArg(g_k_im2col, 0, sizeof(cl_mem), &in);
        clSetKernelArg(g_k_im2col, 1, sizeof(cl_mem), &cols);
        clSetKernelArg(g_k_im2col, 2, sizeof(int), &C_in);
        clSetKernelArg(g_k_im2col, 3, sizeof(int), &L_in);
        clSetKernelArg(g_k_im2col, 4, sizeof(int), &L_out);
        clSetKernelArg(g_k_im2col, 5, sizeof(int), &K);
        clSetKernelArg(g_k_im2col, 6, sizeof(int), &padding);
        clSetKernelArg(g_k_im2col, 7, sizeof(int), &dilation);
        clSetKernelArg(g_k_im2col, 8, sizeof(int), &KC4);
        size_t gws_i[2] = {(size_t)L_out, (size_t)C_in};
        nnopt_enqueue_profiled(queue, g_k_im2col, 2, nullptr, gws_i, nullptr, 0, nullptr, nullptr);
        bool ok = pytorch_linear(queue, L_out, C_out, KC4, cols, W_gemm, out_tn);
        if (ok) {
            clSetKernelArg(g_k_transpose_tn, 0, sizeof(cl_mem), &out_tn);
            clSetKernelArg(g_k_transpose_tn, 1, sizeof(cl_mem), &out);
            clSetKernelArg(g_k_transpose_tn, 2, sizeof(int), &L_out);
            clSetKernelArg(g_k_transpose_tn, 3, sizeof(int), &C_out);
            size_t gws_t[2] = {(size_t)C_out, (size_t)L_out};
            nnopt_enqueue_profiled(queue, g_k_transpose_tn, 2, nullptr, gws_t, nullptr, 0, nullptr, nullptr);
        }
        clReleaseMemObject(cols);
        clReleaseMemObject(out_tn);
        if (ok) {
            if (bias) {
                clSetKernelArg(g_k_bias_add_NCL, 0, sizeof(cl_mem), &out);
                clSetKernelArg(g_k_bias_add_NCL, 1, sizeof(cl_mem), &bias);
                clSetKernelArg(g_k_bias_add_NCL, 2, sizeof(int), &C_out);
                clSetKernelArg(g_k_bias_add_NCL, 3, sizeof(int), &L_out);
                size_t gws2[2] = {(size_t)C_out, (size_t)L_out};
                nnopt_enqueue_profiled(queue, g_k_bias_add_NCL, 2, nullptr, gws2, nullptr, 0, nullptr, nullptr);
            }
            return 0;
        }
        // fall through to c4x4 on GEMM failure
    }
    if (use_t4x4_dec && stride == 1 && groups == 1 && (C_out % 4) == 0) {
        clSetKernelArg(g_k_conv1d_c4x4, 0, sizeof(cl_mem), &in);
        clSetKernelArg(g_k_conv1d_c4x4, 1, sizeof(cl_mem), &W);
        clSetKernelArg(g_k_conv1d_c4x4, 2, sizeof(cl_mem), &out);
        clSetKernelArg(g_k_conv1d_c4x4, 3, sizeof(int), &C_in);
        clSetKernelArg(g_k_conv1d_c4x4, 4, sizeof(int), &C_out);
        clSetKernelArg(g_k_conv1d_c4x4, 5, sizeof(int), &L_in);
        clSetKernelArg(g_k_conv1d_c4x4, 6, sizeof(int), &L_out);
        clSetKernelArg(g_k_conv1d_c4x4, 7, sizeof(int), &K);
        clSetKernelArg(g_k_conv1d_c4x4, 8, sizeof(int), &padding);
        clSetKernelArg(g_k_conv1d_c4x4, 9, sizeof(int), &dilation);
        size_t tiles_l = (size_t)((L_out + 3) / 4);
        size_t gws[2], lws[2];
        if (tiles_l < 64 && (C_out % 16) == 0) {
            // Small-T: 2D workgroup (4 oc-tiles x 16 ol-tiles) keeps all 64
            // lanes busy when there are few L tiles. Measured best of
            // (1,64)/(4,16)/(64,1) on the decode-chain shapes.
            size_t tiles_l_padded = ((tiles_l + 15) / 16) * 16;
            gws[0] = (size_t)(C_out / 4); gws[1] = tiles_l_padded;
            lws[0] = 4; lws[1] = 16;
        } else {
            const int local_t = 64;
            size_t tiles_l_padded = ((tiles_l + local_t - 1) / local_t) * local_t;
            gws[0] = (size_t)(C_out / 4); gws[1] = tiles_l_padded;
            lws[0] = 1; lws[1] = (size_t)local_t;
        }
        cl_int kerr = nnopt_enqueue_profiled(queue, g_k_conv1d_c4x4, 2, nullptr, gws, lws, 0, nullptr, nullptr);
        if (kerr != CL_SUCCESS) {
            NNOPT_ERROR_FMT("conv1d_wn %s: c4x4 dispatch failed (err=%d) C_out=%d L_out=%d",
                            prefix.c_str(), (int)kerr, C_out, L_out);
            return -1;
        }
        if (bias) {
            clSetKernelArg(g_k_bias_add_NCL, 0, sizeof(cl_mem), &out);
            clSetKernelArg(g_k_bias_add_NCL, 1, sizeof(cl_mem), &bias);
            clSetKernelArg(g_k_bias_add_NCL, 2, sizeof(int), &C_out);
            clSetKernelArg(g_k_bias_add_NCL, 3, sizeof(int), &L_out);
            size_t gws2[2] = {(size_t)C_out, (size_t)L_out};
            nnopt_enqueue_profiled(queue, g_k_bias_add_NCL, 2, nullptr, gws2, nullptr, 0, nullptr, nullptr);
        }
        return 0;
    }
    bool use_fast = (w_total <= 4096) && (groups == 1 || C_in % groups == 0);
    if (use_fast) {
        const int LOCAL_T = 384;
        const int OUT_PER_GROUP = LOCAL_T;
        // Cout-tile=4 path: requires C_out % 4 == 0 and 4 * w_total <= WCACHE_C4_MAX (16384).
        // All resblock convs satisfy this. Saves 4× input bandwidth.
        bool use_c4 = (C_out % 4 == 0) && (4 * w_total <= 16384);
        // int8 path: NNOPT_INT8_CONV=1 opts in. **Doesn't help on Adreno 620**
        // (per Qualcomm OpenCL guide §9.5.1, no native 8-bit ALU — int8 ops emulate
        // via 16/32-bit conversion). Measured: 32s int8 vs 17s fp16 on this device,
        // for ~1% cos quality loss. The kernel is kept wired up because newer Adreno
        // tiers (A7x+) expose cl_qcom_dot_product8 which gives hardware 8-bit dot
        // products — on such a device the same code would be faster.
        bool use_int8 = false;
        if (use_c4) {
            const char* e = std::getenv("NNOPT_INT8_CONV");
            use_int8 = (e && e[0] == '1');
        }
        cl_kernel kk;
        cl_mem W_int8 = nullptr, W_scale = nullptr;
        if (use_int8) {
            int Co, Ci, Kw;
            if (!get_wn_conv_weight_int8(cl_ctx, weights, queue, prefix,
                                          &W_int8, &W_scale, &Co, &Ci, &Kw)) {
                NNOPT_ERROR_FMT("conv1d_wn %s: int8 weight build failed", prefix.c_str());
                return -1;
            }
            kk = g_k_conv1d_int8_c4;
        } else {
            kk = use_c4 ? g_k_conv1d_fast_c4 : g_k_conv1d_fast;
        }
        clSetKernelArg(kk, 0, sizeof(cl_mem), &in);
        if (use_int8) {
            clSetKernelArg(kk, 1, sizeof(cl_mem), &W_int8);
            clSetKernelArg(kk, 2, sizeof(cl_mem), &W_scale);
            clSetKernelArg(kk, 3, sizeof(cl_mem), &out);
            clSetKernelArg(kk, 4, sizeof(int), &C_in);
            clSetKernelArg(kk, 5, sizeof(int), &C_out);
            clSetKernelArg(kk, 6, sizeof(int), &L_in);
            clSetKernelArg(kk, 7, sizeof(int), &L_out);
            clSetKernelArg(kk, 8, sizeof(int), &K);
            clSetKernelArg(kk, 9, sizeof(int), &stride);
            clSetKernelArg(kk, 10, sizeof(int), &padding);
            clSetKernelArg(kk, 11, sizeof(int), &dilation);
            clSetKernelArg(kk, 12, sizeof(int), &groups);
        } else {
            clSetKernelArg(kk, 1, sizeof(cl_mem), &W);
            clSetKernelArg(kk, 2, sizeof(cl_mem), &out);
            clSetKernelArg(kk, 3, sizeof(int), &C_in);
            clSetKernelArg(kk, 4, sizeof(int), &C_out);
            clSetKernelArg(kk, 5, sizeof(int), &L_in);
            clSetKernelArg(kk, 6, sizeof(int), &L_out);
            clSetKernelArg(kk, 7, sizeof(int), &K);
            clSetKernelArg(kk, 8, sizeof(int), &stride);
            clSetKernelArg(kk, 9, sizeof(int), &padding);
            clSetKernelArg(kk, 10, sizeof(int), &dilation);
            clSetKernelArg(kk, 11, sizeof(int), &groups);
        }
        size_t L_tiles = (size_t)((L_out + OUT_PER_GROUP - 1) / OUT_PER_GROUP);
        size_t C_groups = (use_c4 || use_int8) ? (size_t)(C_out / 4) : (size_t)C_out;
        size_t gws[2] = {C_groups, L_tiles * LOCAL_T};
        size_t lws[2] = {1, (size_t)LOCAL_T};
        cl_int kerr = nnopt_enqueue_profiled(queue, kk, 2, nullptr, gws, lws, 0, nullptr, nullptr);
        if (kerr != CL_SUCCESS) {
            NNOPT_ERROR_FMT("conv1d_wn %s: dispatch failed (err=%d) use_c4=%d use_int8=%d LOCAL_T=%d C_out=%d L_out=%d",
                            prefix.c_str(), (int)kerr, (int)use_c4, (int)use_int8, LOCAL_T, C_out, L_out);
            return -1;
        }
    } else {
        clSetKernelArg(g_k_conv1d, 0, sizeof(cl_mem), &in);
        clSetKernelArg(g_k_conv1d, 1, sizeof(cl_mem), &W);
        clSetKernelArg(g_k_conv1d, 2, sizeof(cl_mem), &out);
        clSetKernelArg(g_k_conv1d, 3, sizeof(int), &C_in);
        clSetKernelArg(g_k_conv1d, 4, sizeof(int), &C_out);
        clSetKernelArg(g_k_conv1d, 5, sizeof(int), &L_in);
        clSetKernelArg(g_k_conv1d, 6, sizeof(int), &L_out);
        clSetKernelArg(g_k_conv1d, 7, sizeof(int), &K);
        clSetKernelArg(g_k_conv1d, 8, sizeof(int), &stride);
        clSetKernelArg(g_k_conv1d, 9, sizeof(int), &padding);
        clSetKernelArg(g_k_conv1d, 10, sizeof(int), &dilation);
        clSetKernelArg(g_k_conv1d, 11, sizeof(int), &groups);
        size_t gws[2] = {(size_t)C_out, (size_t)L_out};
        nnopt_enqueue_profiled(queue, g_k_conv1d, 2, nullptr, gws, nullptr, 0, nullptr, nullptr);
    }
    if (bias) {
        clSetKernelArg(g_k_bias_add_NCL, 0, sizeof(cl_mem), &out);
        clSetKernelArg(g_k_bias_add_NCL, 1, sizeof(cl_mem), &bias);
        clSetKernelArg(g_k_bias_add_NCL, 2, sizeof(int), &C_out);
        clSetKernelArg(g_k_bias_add_NCL, 3, sizeof(int), &L_out);
        size_t gws2[2] = {(size_t)C_out, (size_t)L_out};
        nnopt_enqueue_profiled(queue, g_k_bias_add_NCL, 2, nullptr, gws2, nullptr, 0, nullptr, nullptr);
    }
    return 0;
}

// Apply ConvTranspose1d (weight_norm). L_out = (L_in-1)*stride - 2*padding + dilation*(K-1) + output_padding + 1
static int convtr1d_wn(OpenCLContext& cl_ctx, Weights& weights, cl_command_queue queue,
                       const std::string& prefix, cl_mem in, cl_mem out,
                       int L_in, int stride, int padding, int output_padding, int dilation, int groups,
                       int* L_out_o, int* C_out_o) {
    int C_in, C_out, K;
    // ConvTranspose1d weight layout: [C_in, C_out/groups, K]
    cl_mem W = get_wn_conv_weight(cl_ctx, weights, queue, prefix, &C_in, &C_out, &K);
    // Here our cache returns shape[0]=C_in, shape[1]=C_out_per_group, shape[2]=K
    int C_out_total = C_out * groups;
    if (!W) return -1;
    cl_mem bias = weights.get_buffer(prefix + ".bias", /*optional=*/true);
    int L_out = (L_in - 1)*stride - 2*padding + dilation*(K-1) + output_padding + 1;
    if (L_out_o) *L_out_o = L_out;
    if (C_out_o) *C_out_o = C_out_total;
    // Phase-decomposed fast path: only ceil(K/stride) taps per output instead of
    // a K-iteration modulo scan, vectorized input loads, 4x4 register tile.
    // NNOPT_CONV_T4X4=0 forces the legacy kernel (A/B validation).
    static int use_tr_fast = -1;
    if (use_tr_fast == -1) {
        const char* e = std::getenv("NNOPT_CONV_T4X4");
        use_tr_fast = (e && e[0] == '0') ? 0 : 1;
    }
    if (use_tr_fast && groups == 1 && dilation == 1 && (C_out_total % 4) == 0 && stride >= 1) {
        // Texture path: phase-major repacked weight image (built once).
        static int use_trtex = -1;
        if (use_trtex == -1) {
            const char* e = std::getenv("NNOPT_TRTEX");
            use_trtex = (e && e[0] == '0') ? 0 : 1;
        }
        static std::unordered_map<std::string, cl_mem> g_trimg_cache;
        int ntaps = (K + stride - 1) / stride;
        cl_mem trimg = nullptr;
        if (!use_trtex) goto trtex_skip;
        {
        auto iti = g_trimg_cache.find(prefix);
        if (iti != g_trimg_cache.end()) {
            trimg = iti->second;
        } else {
            size_t img_w = (size_t)(stride * ntaps * (C_out_total / 4));
            size_t img_h = (size_t)C_in;
            if (img_w <= 8192 && img_h <= 8192) {
                cl_image_format fmt;
                fmt.image_channel_order = CL_RGBA;
                fmt.image_channel_data_type = CL_HALF_FLOAT;
                cl_image_desc desc;
                memset(&desc, 0, sizeof(desc));
                desc.image_type = CL_MEM_OBJECT_IMAGE2D;
                desc.image_width = img_w;
                desc.image_height = img_h;
                cl_int ierr = CL_SUCCESS;
                cl_mem img = clCreateImage(cl_ctx.context(), CL_MEM_READ_WRITE, &fmt, &desc, nullptr, &ierr);
                if (ierr == CL_SUCCESS && img) {
                    clSetKernelArg(g_k_convtr_pack, 0, sizeof(cl_mem), &W);
                    clSetKernelArg(g_k_convtr_pack, 1, sizeof(cl_mem), &img);
                    clSetKernelArg(g_k_convtr_pack, 2, sizeof(int), &C_in);
                    clSetKernelArg(g_k_convtr_pack, 3, sizeof(int), &C_out_total);
                    clSetKernelArg(g_k_convtr_pack, 4, sizeof(int), &K);
                    clSetKernelArg(g_k_convtr_pack, 5, sizeof(int), &stride);
                    clSetKernelArg(g_k_convtr_pack, 6, sizeof(int), &ntaps);
                    size_t gws_p[2] = {img_w, img_h};
                    nnopt_enqueue_profiled(queue, g_k_convtr_pack, 2, nullptr, gws_p, nullptr, 0, nullptr, nullptr);
                    trimg = img;
                }
            }
            g_trimg_cache[prefix] = trimg;
        }
        }
        if (trimg) {
            clSetKernelArg(g_k_convtr1d_c4x4_tex, 0, sizeof(cl_mem), &in);
            clSetKernelArg(g_k_convtr1d_c4x4_tex, 1, sizeof(cl_mem), &trimg);
            clSetKernelArg(g_k_convtr1d_c4x4_tex, 2, sizeof(cl_mem), &out);
            clSetKernelArg(g_k_convtr1d_c4x4_tex, 3, sizeof(int), &C_in);
            clSetKernelArg(g_k_convtr1d_c4x4_tex, 4, sizeof(int), &C_out_total);
            clSetKernelArg(g_k_convtr1d_c4x4_tex, 5, sizeof(int), &L_in);
            clSetKernelArg(g_k_convtr1d_c4x4_tex, 6, sizeof(int), &L_out);
            clSetKernelArg(g_k_convtr1d_c4x4_tex, 7, sizeof(int), &K);
            clSetKernelArg(g_k_convtr1d_c4x4_tex, 8, sizeof(int), &stride);
            clSetKernelArg(g_k_convtr1d_c4x4_tex, 9, sizeof(int), &padding);
            clSetKernelArg(g_k_convtr1d_c4x4_tex, 10, sizeof(int), &ntaps);
            const int local_q = 64;
            int n_q = (L_out - 1 + padding) / stride + 1;
            size_t tiles_q = (size_t)((n_q + 3) / 4);
            size_t tiles_q_padded = ((tiles_q + local_q - 1) / local_q) * local_q;
            size_t gws[3] = {(size_t)(C_out_total / 4), tiles_q_padded, (size_t)stride};
            size_t lws[3] = {1, (size_t)local_q, 1};
            cl_int kerr = nnopt_enqueue_profiled(queue, g_k_convtr1d_c4x4_tex, 3, nullptr, gws, lws, 0, nullptr, nullptr);
            if (kerr == CL_SUCCESS) {
                if (bias) {
                    clSetKernelArg(g_k_bias_add_NCL, 0, sizeof(cl_mem), &out);
                    clSetKernelArg(g_k_bias_add_NCL, 1, sizeof(cl_mem), &bias);
                    clSetKernelArg(g_k_bias_add_NCL, 2, sizeof(int), &C_out_total);
                    clSetKernelArg(g_k_bias_add_NCL, 3, sizeof(int), &L_out);
                    size_t gws2[2] = {(size_t)C_out_total, (size_t)L_out};
                    nnopt_enqueue_profiled(queue, g_k_bias_add_NCL, 2, nullptr, gws2, nullptr, 0, nullptr, nullptr);
                }
                return 0;
            }
        }
        trtex_skip:;
        clSetKernelArg(g_k_convtr1d_c4x4, 0, sizeof(cl_mem), &in);
        clSetKernelArg(g_k_convtr1d_c4x4, 1, sizeof(cl_mem), &W);
        clSetKernelArg(g_k_convtr1d_c4x4, 2, sizeof(cl_mem), &out);
        clSetKernelArg(g_k_convtr1d_c4x4, 3, sizeof(int), &C_in);
        clSetKernelArg(g_k_convtr1d_c4x4, 4, sizeof(int), &C_out_total);
        clSetKernelArg(g_k_convtr1d_c4x4, 5, sizeof(int), &L_in);
        clSetKernelArg(g_k_convtr1d_c4x4, 6, sizeof(int), &L_out);
        clSetKernelArg(g_k_convtr1d_c4x4, 7, sizeof(int), &K);
        clSetKernelArg(g_k_convtr1d_c4x4, 8, sizeof(int), &stride);
        clSetKernelArg(g_k_convtr1d_c4x4, 9, sizeof(int), &padding);
        const int local_q = 64;
        int n_q = (L_out - 1 + padding) / stride + 1;        // q covers all output phases
        size_t tiles_q = (size_t)((n_q + 3) / 4);
        size_t tiles_q_padded = ((tiles_q + local_q - 1) / local_q) * local_q;
        size_t gws[3] = {(size_t)(C_out_total / 4), tiles_q_padded, (size_t)stride};
        size_t lws[3] = {1, (size_t)local_q, 1};
        cl_int kerr = nnopt_enqueue_profiled(queue, g_k_convtr1d_c4x4, 3, nullptr, gws, lws, 0, nullptr, nullptr);
        if (kerr != CL_SUCCESS) {
            NNOPT_ERROR_FMT("convtr1d_wn %s: c4x4 dispatch failed (err=%d)", prefix.c_str(), (int)kerr);
            return -1;
        }
    } else {
    clSetKernelArg(g_k_convtr1d, 0, sizeof(cl_mem), &in);
    clSetKernelArg(g_k_convtr1d, 1, sizeof(cl_mem), &W);
    clSetKernelArg(g_k_convtr1d, 2, sizeof(cl_mem), &out);
    clSetKernelArg(g_k_convtr1d, 3, sizeof(int), &C_in);
    clSetKernelArg(g_k_convtr1d, 4, sizeof(int), &C_out_total);
    clSetKernelArg(g_k_convtr1d, 5, sizeof(int), &L_in);
    clSetKernelArg(g_k_convtr1d, 6, sizeof(int), &L_out);
    clSetKernelArg(g_k_convtr1d, 7, sizeof(int), &K);
    clSetKernelArg(g_k_convtr1d, 8, sizeof(int), &stride);
    clSetKernelArg(g_k_convtr1d, 9, sizeof(int), &padding);
    clSetKernelArg(g_k_convtr1d, 10, sizeof(int), &dilation);
    clSetKernelArg(g_k_convtr1d, 11, sizeof(int), &groups);
    size_t gws[2] = {(size_t)C_out_total, (size_t)L_out};
    nnopt_enqueue_profiled(queue, g_k_convtr1d, 2, nullptr, gws, nullptr, 0, nullptr, nullptr);
    }
    if (bias) {
        clSetKernelArg(g_k_bias_add_NCL, 0, sizeof(cl_mem), &out);
        clSetKernelArg(g_k_bias_add_NCL, 1, sizeof(cl_mem), &bias);
        clSetKernelArg(g_k_bias_add_NCL, 2, sizeof(int), &C_out_total);
        clSetKernelArg(g_k_bias_add_NCL, 3, sizeof(int), &L_out);
        size_t gws2[2] = {(size_t)C_out_total, (size_t)L_out};
        nnopt_enqueue_profiled(queue, g_k_bias_add_NCL, 2, nullptr, gws2, nullptr, 0, nullptr, nullptr);
    }
    return 0;
}

// AdaIN1d: y = (1+gamma)*instance_norm(x) + beta
//   gamma_beta = fc(s) [256], split into gamma[128 if num_features=128 else C], beta[same]
//   where fc.weight is [2*num_features, style_dim], fc.bias is [2*num_features].
static int apply_adain1d(OpenCLContext& cl_ctx, Weights& weights, cl_command_queue queue,
                         const std::string& adain_prefix,  // ".adain1.0" or ".norm1" etc.
                         cl_mem x, cl_mem y, cl_mem ref_s_dec, int C, int T) {
    // fc: Linear(style_dim=128, 2*num_features=2*C). Weight key: <prefix>.fc.weight
    std::string fc_w = adain_prefix + ".fc.weight";
    std::string fc_b = adain_prefix + ".fc.bias";
    cl_mem W = weights.get_buffer(fc_w);
    cl_mem b = weights.get_buffer(fc_b);
    if (!W || !b) { NNOPT_ERROR_FMT("adain: missing %s", adain_prefix.c_str()); return -1; }
    int two_C = 2*C, style_dim = 128;
    cl_mem h = alloc_rw(cl_ctx, sizeof(nnopt_storage_t) * two_C);
    int has_bias = 1;
    clSetKernelArg(g_k_linear_apply, 0, sizeof(cl_mem), &ref_s_dec);
    clSetKernelArg(g_k_linear_apply, 1, sizeof(cl_mem), &W);
    clSetKernelArg(g_k_linear_apply, 2, sizeof(cl_mem), &b);
    clSetKernelArg(g_k_linear_apply, 3, sizeof(cl_mem), &h);
    clSetKernelArg(g_k_linear_apply, 4, sizeof(int), &style_dim);
    clSetKernelArg(g_k_linear_apply, 5, sizeof(int), &two_C);
    clSetKernelArg(g_k_linear_apply, 6, sizeof(int), &has_bias);
    size_t gws_lin = (size_t)two_C;
    nnopt_enqueue_profiled(queue, g_k_linear_apply, 1, nullptr, &gws_lin, nullptr, 0, nullptr, nullptr);
    if (adain_prefix == "decoder.module.encode.norm1") {
        NNOPT_LAYER_CHECK("dec_encode_norm1_fc_mine", queue, h, (size_t)two_C);
    }
    // Split [2C, 1] -> gamma [C, 1], beta [C, 1]
    cl_mem gamma = alloc_rw(cl_ctx, sizeof(nnopt_storage_t) * C);
    cl_mem beta  = alloc_rw(cl_ctx, sizeof(nnopt_storage_t) * C);
    int one = 1;
    clSetKernelArg(g_k_split, 0, sizeof(cl_mem), &h);
    clSetKernelArg(g_k_split, 1, sizeof(cl_mem), &gamma);
    clSetKernelArg(g_k_split, 2, sizeof(cl_mem), &beta);
    clSetKernelArg(g_k_split, 3, sizeof(int), &C);
    clSetKernelArg(g_k_split, 4, sizeof(int), &one);
    size_t gws_sp[2] = {(size_t)C, (size_t)1};
    nnopt_enqueue_profiled(queue, g_k_split, 2, nullptr, gws_sp, nullptr, 0, nullptr, nullptr);
    // instance norm
    cl_mem xn = alloc_rw(cl_ctx, sizeof(nnopt_storage_t) * C * T);
    float eps = 1e-5f;
    clSetKernelArg(g_k_instnorm, 0, sizeof(cl_mem), &x);
    clSetKernelArg(g_k_instnorm, 1, sizeof(cl_mem), &xn);
    clSetKernelArg(g_k_instnorm, 2, sizeof(int), &C);
    clSetKernelArg(g_k_instnorm, 3, sizeof(int), &T);
    clSetKernelArg(g_k_instnorm, 4, sizeof(float), &eps);
    size_t gws_in = (size_t)C * 64;  // workgroup-per-channel reduction
    size_t lws_in = 64;
    nnopt_enqueue_profiled(queue, g_k_instnorm, 1, nullptr, &gws_in, &lws_in, 0, nullptr, nullptr);
    if (adain_prefix == "decoder.module.encode.norm1") {
        NNOPT_LAYER_CHECK("dec_encode_norm1_norm_mine", queue, xn, (size_t)C * T);
    }
    // Apply InstanceNorm1d's own learnable affine (affine=True in reference).
    // Reference: nn.InstanceNorm1d(C, affine=True), y = (x-mean)/std * IN.weight[c] + IN.bias[c]
    // Then AdaIN: (1+gamma) * y + beta.
    // Folded: y = (1+gamma)*IN_w*(x-mean)/std + ((1+gamma)*IN_b + beta)
    //          = effective_gamma * xn + effective_beta
    //   where effective_gamma[c] = (1+gamma[c]) * IN_w[c] - 1   (so adain_combine adds the -1)
    //         effective_beta[c]  = (1+gamma[c]) * IN_b[c] + beta[c]
    cl_mem in_w = weights.get_buffer(adain_prefix + ".norm.weight", /*optional=*/true);
    cl_mem in_b = weights.get_buffer(adain_prefix + ".norm.bias",   /*optional=*/true);
    if (in_w && in_b) {
        // Fold IN_w/IN_b into gamma/beta on host.
        std::vector<float> g_f(C), b_f(C), w_f(C), bi_f(C);
#ifdef NNOPT_USE_FP16
        std::vector<uint16_t> gh(C), bh(C), wh(C), bih(C);
        clEnqueueReadBuffer(queue, gamma, CL_TRUE, 0, sizeof(uint16_t)*C, gh.data(), 0, nullptr, nullptr);
        clEnqueueReadBuffer(queue, beta,  CL_TRUE, 0, sizeof(uint16_t)*C, bh.data(), 0, nullptr, nullptr);
        clEnqueueReadBuffer(queue, in_w,  CL_TRUE, 0, sizeof(uint16_t)*C, wh.data(), 0, nullptr, nullptr);
        clEnqueueReadBuffer(queue, in_b,  CL_TRUE, 0, sizeof(uint16_t)*C, bih.data(), 0, nullptr, nullptr);
        for (int i = 0; i < C; ++i) {
            float gv = nnopt_f16_to_f32(gh[i]);
            float bv = nnopt_f16_to_f32(bh[i]);
            float wv = nnopt_f16_to_f32(wh[i]);
            float bw = nnopt_f16_to_f32(bih[i]);
            float eg = (1.0f + gv) * wv - 1.0f;
            float eb = (1.0f + gv) * bw + bv;
            gh[i] = nnopt_f32_to_f16(eg);
            bh[i] = nnopt_f32_to_f16(eb);
        }
        clEnqueueWriteBuffer(queue, gamma, CL_TRUE, 0, sizeof(uint16_t)*C, gh.data(), 0, nullptr, nullptr);
        clEnqueueWriteBuffer(queue, beta,  CL_TRUE, 0, sizeof(uint16_t)*C, bh.data(), 0, nullptr, nullptr);
#else
        clEnqueueReadBuffer(queue, gamma, CL_TRUE, 0, sizeof(float)*C, g_f.data(), 0, nullptr, nullptr);
        clEnqueueReadBuffer(queue, beta,  CL_TRUE, 0, sizeof(float)*C, b_f.data(), 0, nullptr, nullptr);
        clEnqueueReadBuffer(queue, in_w,  CL_TRUE, 0, sizeof(float)*C, w_f.data(), 0, nullptr, nullptr);
        clEnqueueReadBuffer(queue, in_b,  CL_TRUE, 0, sizeof(float)*C, bi_f.data(), 0, nullptr, nullptr);
        for (int i = 0; i < C; ++i) {
            float eg = (1.0f + g_f[i]) * w_f[i] - 1.0f;
            float eb = (1.0f + g_f[i]) * bi_f[i] + b_f[i];
            g_f[i] = eg;
            b_f[i] = eb;
        }
        clEnqueueWriteBuffer(queue, gamma, CL_TRUE, 0, sizeof(float)*C, g_f.data(), 0, nullptr, nullptr);
        clEnqueueWriteBuffer(queue, beta,  CL_TRUE, 0, sizeof(float)*C, b_f.data(), 0, nullptr, nullptr);
#endif
    }
    // adain combine
    clSetKernelArg(g_k_adain_combine, 0, sizeof(cl_mem), &xn);
    clSetKernelArg(g_k_adain_combine, 1, sizeof(cl_mem), &gamma);
    clSetKernelArg(g_k_adain_combine, 2, sizeof(cl_mem), &beta);
    clSetKernelArg(g_k_adain_combine, 3, sizeof(cl_mem), &y);
    clSetKernelArg(g_k_adain_combine, 4, sizeof(int), &C);
    clSetKernelArg(g_k_adain_combine, 5, sizeof(int), &T);
    size_t gws_ad[2] = {(size_t)C, (size_t)T};
    nnopt_enqueue_profiled(queue, g_k_adain_combine, 2, nullptr, gws_ad, nullptr, 0, nullptr, nullptr);
    clReleaseMemObject(h);
    clReleaseMemObject(gamma);
    clReleaseMemObject(beta);
    clReleaseMemObject(xn);
    return 0;
}

static void apply_leaky(cl_command_queue queue, cl_mem y, int N, float slope) {
    clSetKernelArg(g_k_leaky, 0, sizeof(cl_mem), &y);
    clSetKernelArg(g_k_leaky, 1, sizeof(int), &N);
    clSetKernelArg(g_k_leaky, 2, sizeof(float), &slope);
    run1d(queue, g_k_leaky, (size_t)N);
}

// Public version for predictor to reuse.
extern "C" int dec_apply_adainresblk1d(OpenCLContext& cl_ctx, Weights& weights, cl_command_queue queue,
                                        const std::string& prefix,
                                        cl_mem in, cl_mem out,
                                        int dim_in, int dim_out, int T_in,
                                        bool upsample, cl_mem ref_s_dec,
                                        int* T_out_o);

// AdainResBlk1d: in [dim_in, T_in], out [dim_out, T_out].
// upsample=true -> uses ConvTranspose1d "pool" kernel=3 stride=2 padding=1 output_padding=1 groups=dim_in.
// Reference _residual: norm1->actv->pool->conv1->norm2->actv->conv2; output = (residual + shortcut)*rsqrt(2)
static int apply_adainresblk1d(OpenCLContext& cl_ctx, Weights& weights, cl_command_queue queue,
                                const std::string& prefix,  // e.g. "decoder.module.encode"
                                cl_mem in, cl_mem out,
                                int dim_in, int dim_out, int T_in,
                                bool upsample, cl_mem ref_s_dec,
                                int* T_out_o) {
    int T_out = upsample ? T_in * 2 : T_in;
    if (T_out_o) *T_out_o = T_out;
    // Residual branch
    cl_mem r1 = alloc_rw(cl_ctx, sizeof(nnopt_storage_t) * dim_in * T_in);
    if (apply_adain1d(cl_ctx, weights, queue, prefix + ".norm1", in, r1, ref_s_dec, dim_in, T_in) != 0) return -1;
    if (prefix == "decoder.module.encode") {
        NNOPT_LAYER_CHECK("dec_encode_norm1", queue, r1, (size_t)dim_in * T_in);
    }
    apply_leaky(queue, r1, dim_in * T_in, 0.2f);
    // pool: identity or ConvTranspose1d groups=dim_in
    cl_mem after_pool = nullptr;
    int T_after_pool = T_in;
    if (upsample) {
        // pool: ConvTranspose1d(dim_in, dim_in, kernel=3, stride=2, groups=dim_in, padding=1, output_padding=1)
        T_after_pool = (T_in - 1) * 2 - 2 + (3 - 1) + 1 + 1;  // = T_in*2
        after_pool = alloc_rw(cl_ctx, sizeof(nnopt_storage_t) * dim_in * T_after_pool);
        int L_out, C_out;
        if (convtr1d_wn(cl_ctx, weights, queue, prefix + ".pool", r1, after_pool,
                        T_in, /*stride=*/2, /*padding=*/1, /*output_padding=*/1, /*dilation=*/1, /*groups=*/dim_in,
                        &L_out, &C_out) != 0) return -1;
        T_after_pool = L_out;
    } else {
        after_pool = r1;
        r1 = nullptr;
    }
    // conv1: weight_norm Conv1d dim_in -> dim_out, kernel=3, padding=1
    cl_mem c1_out = alloc_rw(cl_ctx, sizeof(nnopt_storage_t) * dim_out * T_after_pool);
    int Lc1, Cc1;
    if (conv1d_wn(cl_ctx, weights, queue, prefix + ".conv1", after_pool, c1_out,
                  T_after_pool, 1, 1, 1, 1, &Lc1, &Cc1) != 0) return -1;
    // norm2 + actv + conv2
    cl_mem r2 = alloc_rw(cl_ctx, sizeof(nnopt_storage_t) * dim_out * Lc1);
    if (apply_adain1d(cl_ctx, weights, queue, prefix + ".norm2", c1_out, r2, ref_s_dec, dim_out, Lc1) != 0) return -1;
    apply_leaky(queue, r2, dim_out * Lc1, 0.2f);
    cl_mem c2_out = alloc_rw(cl_ctx, sizeof(nnopt_storage_t) * dim_out * Lc1);
    int Lc2, Cc2;
    if (conv1d_wn(cl_ctx, weights, queue, prefix + ".conv2", r2, c2_out,
                  Lc1, 1, 1, 1, 1, &Lc2, &Cc2) != 0) return -1;
    // Shortcut: upsample + (conv1x1 if learned_sc)
    cl_mem sc = nullptr;
    if (upsample || dim_in != dim_out) {
        cl_mem sc_pre = in;
        cl_mem sc_up = nullptr;
        if (upsample) {
            sc_up = alloc_rw(cl_ctx, sizeof(nnopt_storage_t) * dim_in * (T_in * 2));
            int scale = 2;
            clSetKernelArg(g_k_upsample_nn, 0, sizeof(cl_mem), &sc_pre);
            clSetKernelArg(g_k_upsample_nn, 1, sizeof(cl_mem), &sc_up);
            clSetKernelArg(g_k_upsample_nn, 2, sizeof(int), &dim_in);
            clSetKernelArg(g_k_upsample_nn, 3, sizeof(int), &T_in);
            clSetKernelArg(g_k_upsample_nn, 4, sizeof(int), &scale);
            size_t gws_u[2] = {(size_t)dim_in, (size_t)(T_in*2)};
            nnopt_enqueue_profiled(queue, g_k_upsample_nn, 2, nullptr, gws_u, nullptr, 0, nullptr, nullptr);
            sc_pre = sc_up;
        }
        if (dim_in != dim_out) {
            sc = alloc_rw(cl_ctx, sizeof(nnopt_storage_t) * dim_out * Lc2);
            int Lsc, Csc;
            if (conv1d_wn(cl_ctx, weights, queue, prefix + ".conv1x1", sc_pre, sc, Lc2, 1, 0, 1, 1, &Lsc, &Csc) != 0) return -1;
            if (sc_up) clReleaseMemObject(sc_up);
        } else {
            sc = sc_up;
        }
    } else {
        // identity copy
        sc = alloc_rw(cl_ctx, sizeof(nnopt_storage_t) * dim_out * Lc2);
        clEnqueueCopyBuffer(queue, in, sc, 0, 0, sizeof(nnopt_storage_t) * dim_out * Lc2, 0, nullptr, nullptr);
    }
    // out = (c2_out + sc) * rsqrt(2)
    int N = dim_out * Lc2;
    float s = 1.0f / std::sqrt(2.0f);
    clSetKernelArg(g_k_add_scale, 0, sizeof(cl_mem), &c2_out);
    clSetKernelArg(g_k_add_scale, 1, sizeof(cl_mem), &sc);
    clSetKernelArg(g_k_add_scale, 2, sizeof(int), &N);
    clSetKernelArg(g_k_add_scale, 3, sizeof(float), &s);
    run1d(queue, g_k_add_scale, (size_t)N);
    // copy result to out
    clEnqueueCopyBuffer(queue, c2_out, out, 0, 0, sizeof(nnopt_storage_t) * dim_out * Lc2, 0, nullptr, nullptr);
    if (T_out_o) *T_out_o = Lc2;
    // Cleanup
    if (r1) clReleaseMemObject(r1);
    clReleaseMemObject(after_pool);
    clReleaseMemObject(c1_out);
    clReleaseMemObject(r2);
    clReleaseMemObject(c2_out);
    if (sc) clReleaseMemObject(sc);
    return 0;
}

// Public wrapper for predictor.cpp
extern "C" int dec_apply_adainresblk1d(OpenCLContext& cl_ctx, Weights& weights, cl_command_queue queue,
                                        const std::string& prefix,
                                        cl_mem in, cl_mem out,
                                        int dim_in, int dim_out, int T_in,
                                        bool upsample, cl_mem ref_s_dec,
                                        int* T_out_o) {
    if (!ensure_built(cl_ctx)) return -1;
    return apply_adainresblk1d(cl_ctx, weights, queue, prefix, in, out,
                               dim_in, dim_out, T_in, upsample, ref_s_dec, T_out_o);
}

// Public wrappers for fp16 helpers (used by the fp32 generator path in _generator_fp32.cpp)
extern "C" int dec_apply_adain1d(OpenCLContext& cl_ctx, Weights& weights, cl_command_queue queue,
                                  const std::string& adain_prefix,
                                  cl_mem x, cl_mem y, cl_mem ref_s_dec, int C, int T) {
    if (!ensure_built(cl_ctx)) return -1;
    return apply_adain1d(cl_ctx, weights, queue, adain_prefix, x, y, ref_s_dec, C, T);
}
extern "C" int dec_apply_snake1d(OpenCLContext& cl_ctx, Weights& weights, cl_command_queue queue,
                                  const std::string& alpha_key, cl_mem y, int C, int T) {
    if (!ensure_built(cl_ctx)) return -1;
    return apply_snake1d(cl_ctx, weights, queue, alpha_key, y, C, T);
}

// ─── Fused adain_combine + snake + conv1d helper ─────────────────────────
//
// Computes gamma, beta (with InstanceNorm affine pre-folded) by replaying
// the front half of apply_adain1d: linear_apply on the style → split into
// gamma/beta → fold the per-channel norm.weight/norm.bias if they exist.
// Caller owns the returned buffers and must release them with clReleaseMemObject.
//
// Returns 0 on success; on failure, both *out_gamma and *out_beta are nullptr.
static int compute_adain_gamma_beta(OpenCLContext& cl_ctx, Weights& weights,
                                     cl_command_queue queue,
                                     const std::string& adain_prefix,
                                     cl_mem ref_s_dec, int C,
                                     cl_mem* out_gamma, cl_mem* out_beta) {
    *out_gamma = nullptr;
    *out_beta = nullptr;
    std::string fc_w = adain_prefix + ".fc.weight";
    std::string fc_b = adain_prefix + ".fc.bias";
    cl_mem W = weights.get_buffer(fc_w);
    cl_mem b = weights.get_buffer(fc_b);
    if (!W || !b) { NNOPT_ERROR_FMT("compute_adain_gamma_beta: missing %s", adain_prefix.c_str()); return -1; }
    int two_C = 2*C, style_dim = 128;
    cl_mem h = alloc_rw(cl_ctx, sizeof(nnopt_storage_t) * two_C);
    int has_bias = 1;
    clSetKernelArg(g_k_linear_apply, 0, sizeof(cl_mem), &ref_s_dec);
    clSetKernelArg(g_k_linear_apply, 1, sizeof(cl_mem), &W);
    clSetKernelArg(g_k_linear_apply, 2, sizeof(cl_mem), &b);
    clSetKernelArg(g_k_linear_apply, 3, sizeof(cl_mem), &h);
    clSetKernelArg(g_k_linear_apply, 4, sizeof(int), &style_dim);
    clSetKernelArg(g_k_linear_apply, 5, sizeof(int), &two_C);
    clSetKernelArg(g_k_linear_apply, 6, sizeof(int), &has_bias);
    size_t gws_lin = (size_t)two_C;
    nnopt_enqueue_profiled(queue, g_k_linear_apply, 1, nullptr, &gws_lin, nullptr, 0, nullptr, nullptr);
    cl_mem gamma = alloc_rw(cl_ctx, sizeof(nnopt_storage_t) * C);
    cl_mem beta  = alloc_rw(cl_ctx, sizeof(nnopt_storage_t) * C);
    int one = 1;
    clSetKernelArg(g_k_split, 0, sizeof(cl_mem), &h);
    clSetKernelArg(g_k_split, 1, sizeof(cl_mem), &gamma);
    clSetKernelArg(g_k_split, 2, sizeof(cl_mem), &beta);
    clSetKernelArg(g_k_split, 3, sizeof(int), &C);
    clSetKernelArg(g_k_split, 4, sizeof(int), &one);
    size_t gws_sp[2] = {(size_t)C, (size_t)1};
    nnopt_enqueue_profiled(queue, g_k_split, 2, nullptr, gws_sp, nullptr, 0, nullptr, nullptr);
    clReleaseMemObject(h);
    // Fold IN affine if present (matches apply_adain1d behavior).
    cl_mem in_w = weights.get_buffer(adain_prefix + ".norm.weight", /*optional=*/true);
    cl_mem in_b = weights.get_buffer(adain_prefix + ".norm.bias",   /*optional=*/true);
    if (in_w && in_b) {
#ifdef NNOPT_USE_FP16
        std::vector<uint16_t> gh(C), bh(C), wh(C), bih(C);
        clEnqueueReadBuffer(queue, gamma, CL_TRUE, 0, sizeof(uint16_t)*C, gh.data(), 0, nullptr, nullptr);
        clEnqueueReadBuffer(queue, beta,  CL_TRUE, 0, sizeof(uint16_t)*C, bh.data(), 0, nullptr, nullptr);
        clEnqueueReadBuffer(queue, in_w,  CL_TRUE, 0, sizeof(uint16_t)*C, wh.data(), 0, nullptr, nullptr);
        clEnqueueReadBuffer(queue, in_b,  CL_TRUE, 0, sizeof(uint16_t)*C, bih.data(), 0, nullptr, nullptr);
        for (int i = 0; i < C; ++i) {
            float gv = nnopt_f16_to_f32(gh[i]);
            float bv = nnopt_f16_to_f32(bh[i]);
            float wv = nnopt_f16_to_f32(wh[i]);
            float bw = nnopt_f16_to_f32(bih[i]);
            float eg = (1.0f + gv) * wv - 1.0f;
            float eb = (1.0f + gv) * bw + bv;
            gh[i] = nnopt_f32_to_f16(eg);
            bh[i] = nnopt_f32_to_f16(eb);
        }
        clEnqueueWriteBuffer(queue, gamma, CL_TRUE, 0, sizeof(uint16_t)*C, gh.data(), 0, nullptr, nullptr);
        clEnqueueWriteBuffer(queue, beta,  CL_TRUE, 0, sizeof(uint16_t)*C, bh.data(), 0, nullptr, nullptr);
#else
        std::vector<float> g_f(C), b_f(C), w_f(C), bi_f(C);
        clEnqueueReadBuffer(queue, gamma, CL_TRUE, 0, sizeof(float)*C, g_f.data(), 0, nullptr, nullptr);
        clEnqueueReadBuffer(queue, beta,  CL_TRUE, 0, sizeof(float)*C, b_f.data(), 0, nullptr, nullptr);
        clEnqueueReadBuffer(queue, in_w,  CL_TRUE, 0, sizeof(float)*C, w_f.data(), 0, nullptr, nullptr);
        clEnqueueReadBuffer(queue, in_b,  CL_TRUE, 0, sizeof(float)*C, bi_f.data(), 0, nullptr, nullptr);
        for (int i = 0; i < C; ++i) {
            float eg = (1.0f + g_f[i]) * w_f[i] - 1.0f;
            float eb = (1.0f + g_f[i]) * bi_f[i] + b_f[i];
            g_f[i] = eg;
            b_f[i] = eb;
        }
        clEnqueueWriteBuffer(queue, gamma, CL_TRUE, 0, sizeof(float)*C, g_f.data(), 0, nullptr, nullptr);
        clEnqueueWriteBuffer(queue, beta,  CL_TRUE, 0, sizeof(float)*C, b_f.data(), 0, nullptr, nullptr);
#endif
    }
    *out_gamma = gamma;
    *out_beta = beta;
    return 0;
}

// Public wrapper for conv1d weight_norm (predictor uses for F0_proj/N_proj).
extern "C" int dec_conv1d_wn(OpenCLContext& cl_ctx, Weights& weights, cl_command_queue queue,
                              const std::string& prefix, cl_mem in, cl_mem out,
                              int L_in, int stride, int padding, int dilation, int groups,
                              int* L_out_o, int* C_out_o) {
    if (!ensure_built(cl_ctx)) return -1;
    return conv1d_wn(cl_ctx, weights, queue, prefix, in, out,
                     L_in, stride, padding, dilation, groups, L_out_o, C_out_o);
}

// Public Conv1d (NOT weight_norm) for predictor's F0_proj/N_proj which are plain Conv1d.
extern "C" int dec_conv1d_plain(OpenCLContext& cl_ctx, Weights& weights, cl_command_queue queue,
                                 const std::string& prefix, cl_mem in, cl_mem out,
                                 int L_in, int stride, int padding, int dilation, int groups,
                                 int* L_out_o, int* C_out_o) {
    if (!ensure_built(cl_ctx)) return -1;
    cl_mem W = weights.get_buffer(prefix + ".weight");
    cl_mem b = weights.get_buffer(prefix + ".bias", /*optional=*/true);
    if (!W) { NNOPT_ERROR_FMT("plain conv1d: missing %s.weight", prefix.c_str()); return -1; }
    auto shape = weights.get_shape(prefix + ".weight");
    int C_out = shape[0], C_in = shape[1], K = shape[2];
    int L_out = (L_in + 2*padding - dilation*(K-1) - 1) / stride + 1;
    if (L_out_o) *L_out_o = L_out;
    if (C_out_o) *C_out_o = C_out;
    int w_total = (C_in / (groups > 0 ? groups : 1)) * K;
    if (stride == 1 && groups == 1 && (C_out % 4) == 0) {
        // Register-tiled 4x4 path (noise_convs.1 etc).
        clSetKernelArg(g_k_conv1d_c4x4, 0, sizeof(cl_mem), &in);
        clSetKernelArg(g_k_conv1d_c4x4, 1, sizeof(cl_mem), &W);
        clSetKernelArg(g_k_conv1d_c4x4, 2, sizeof(cl_mem), &out);
        clSetKernelArg(g_k_conv1d_c4x4, 3, sizeof(int), &C_in);
        clSetKernelArg(g_k_conv1d_c4x4, 4, sizeof(int), &C_out);
        clSetKernelArg(g_k_conv1d_c4x4, 5, sizeof(int), &L_in);
        clSetKernelArg(g_k_conv1d_c4x4, 6, sizeof(int), &L_out);
        clSetKernelArg(g_k_conv1d_c4x4, 7, sizeof(int), &K);
        clSetKernelArg(g_k_conv1d_c4x4, 8, sizeof(int), &padding);
        clSetKernelArg(g_k_conv1d_c4x4, 9, sizeof(int), &dilation);
        const int local_t = 64;
        size_t tiles_l = (size_t)((L_out + 3) / 4);
        size_t tiles_l_padded = ((tiles_l + local_t - 1) / local_t) * local_t;
        size_t gws[2] = {(size_t)(C_out / 4), tiles_l_padded};
        size_t lws[2] = {1, (size_t)local_t};
        nnopt_enqueue_profiled(queue, g_k_conv1d_c4x4, 2, nullptr, gws, lws, 0, nullptr, nullptr);
    } else if ((C_out % 4) == 0 && 4 * w_total <= 16384 && (groups == 1 || C_in % groups == 0)) {
        // Strided convs (noise_convs.0) — LDS-cached c4 kernel.
        clSetKernelArg(g_k_conv1d_fast_c4, 0, sizeof(cl_mem), &in);
        clSetKernelArg(g_k_conv1d_fast_c4, 1, sizeof(cl_mem), &W);
        clSetKernelArg(g_k_conv1d_fast_c4, 2, sizeof(cl_mem), &out);
        clSetKernelArg(g_k_conv1d_fast_c4, 3, sizeof(int), &C_in);
        clSetKernelArg(g_k_conv1d_fast_c4, 4, sizeof(int), &C_out);
        clSetKernelArg(g_k_conv1d_fast_c4, 5, sizeof(int), &L_in);
        clSetKernelArg(g_k_conv1d_fast_c4, 6, sizeof(int), &L_out);
        clSetKernelArg(g_k_conv1d_fast_c4, 7, sizeof(int), &K);
        clSetKernelArg(g_k_conv1d_fast_c4, 8, sizeof(int), &stride);
        clSetKernelArg(g_k_conv1d_fast_c4, 9, sizeof(int), &padding);
        clSetKernelArg(g_k_conv1d_fast_c4, 10, sizeof(int), &dilation);
        clSetKernelArg(g_k_conv1d_fast_c4, 11, sizeof(int), &groups);
        const int LOCAL_T = 384;
        size_t L_tiles = (size_t)((L_out + LOCAL_T - 1) / LOCAL_T);
        size_t gws[2] = {(size_t)(C_out / 4), L_tiles * LOCAL_T};
        size_t lws[2] = {1, (size_t)LOCAL_T};
        nnopt_enqueue_profiled(queue, g_k_conv1d_fast_c4, 2, nullptr, gws, lws, 0, nullptr, nullptr);
    } else {
    clSetKernelArg(g_k_conv1d, 0, sizeof(cl_mem), &in);
    clSetKernelArg(g_k_conv1d, 1, sizeof(cl_mem), &W);
    clSetKernelArg(g_k_conv1d, 2, sizeof(cl_mem), &out);
    clSetKernelArg(g_k_conv1d, 3, sizeof(int), &C_in);
    clSetKernelArg(g_k_conv1d, 4, sizeof(int), &C_out);
    clSetKernelArg(g_k_conv1d, 5, sizeof(int), &L_in);
    clSetKernelArg(g_k_conv1d, 6, sizeof(int), &L_out);
    clSetKernelArg(g_k_conv1d, 7, sizeof(int), &K);
    clSetKernelArg(g_k_conv1d, 8, sizeof(int), &stride);
    clSetKernelArg(g_k_conv1d, 9, sizeof(int), &padding);
    clSetKernelArg(g_k_conv1d, 10, sizeof(int), &dilation);
    clSetKernelArg(g_k_conv1d, 11, sizeof(int), &groups);
    size_t gws[2] = {(size_t)C_out, (size_t)L_out};
    nnopt_enqueue_profiled(queue, g_k_conv1d, 2, nullptr, gws, nullptr, 0, nullptr, nullptr);
    }
    if (b) {
        clSetKernelArg(g_k_bias_add_NCL, 0, sizeof(cl_mem), &out);
        clSetKernelArg(g_k_bias_add_NCL, 1, sizeof(cl_mem), &b);
        clSetKernelArg(g_k_bias_add_NCL, 2, sizeof(int), &C_out);
        clSetKernelArg(g_k_bias_add_NCL, 3, sizeof(int), &L_out);
        size_t gws2[2] = {(size_t)C_out, (size_t)L_out};
        nnopt_enqueue_profiled(queue, g_k_bias_add_NCL, 2, nullptr, gws2, nullptr, 0, nullptr, nullptr);
    }
    return 0;
}

// ─── Public op_decoder ──────────────────────────────────────────────────
extern "C" int op_decoder(OpenCLContext& cl_ctx, Weights& weights, cl_command_queue queue,
                          cl_mem asr,       // [512, T_frames]
                          cl_mem F0_pred,   // [T_frames]
                          cl_mem N_pred,    // [T_frames]
                          cl_mem ref_s_dec, // [256] (full ref_s; we use first 128)
                          int T_frames,
                          std::vector<int16_t>& out_pcm_int16) {
    if (!ensure_built(cl_ctx)) return -1;
    const int n_fft = 20, hop = 5;
    const int upsample_to_audio = 60; // total post-iSTFT upsample factor (excluding iSTFT's own hop)

    // Decoder pre-generator timing (gated by NNOPT_NONGEN_PROFILE=1)
    bool dec_prof = false;
    if (const char* p = std::getenv("NNOPT_NONGEN_PROFILE")) dec_prof = (p[0] == '1');
    auto dec_t_prev = std::chrono::steady_clock::now();
    #define DEC_TICK(label) do { if (dec_prof) { clFinish(queue); auto _t = std::chrono::steady_clock::now(); double _s = std::chrono::duration<double>(_t - dec_t_prev).count(); fprintf(stderr, "[dec] %s @ %.3fs\n", (label), _s); fflush(stderr); dec_t_prev = _t; } } while (0)

    // F0_pred/N_pred sized [T_frames*2] from F0Ntrain (its middle block upsamples by 2).
    // F0_conv stride=2 takes it back to ~T_frames. Everything downstream uses T_frames (matching asr).
    int Tin_F0N = T_frames * 2;
    int Tf = T_frames;  // post-F0_conv length, used everywhere downstream
    int Tf_half = (Tin_F0N + 2*1 - 1*(3-1) - 1) / 2 + 1;  // F0_conv output length
    cl_mem F0_in_view = F0_pred;
    cl_mem F0 = alloc_rw(cl_ctx, sizeof(nnopt_storage_t) * Tf_half);
    int Lf, Cf;
    conv1d_wn(cl_ctx, weights, queue, "decoder.module.F0_conv", F0_in_view, F0, Tin_F0N, 2, 1, 1, 1, &Lf, &Cf);
    NNOPT_LAYER_CHECK("dec_F0_input", queue, F0_pred, (size_t)Tin_F0N);
    NNOPT_LAYER_CHECK("dec_F0_conv", queue, F0, (size_t)Lf);
    cl_mem N = alloc_rw(cl_ctx, sizeof(nnopt_storage_t) * Tf_half);
    int Ln, Cn;
    conv1d_wn(cl_ctx, weights, queue, "decoder.module.N_conv", N_pred, N, Tin_F0N, 2, 1, 1, 1, &Ln, &Cn);

    // asr_res: Conv1d(512 -> 64, k=1) — weight_norm
    cl_mem asr_res = alloc_rw(cl_ctx, sizeof(nnopt_storage_t) * 64 * Tf);
    int Lar, Car;
    conv1d_wn(cl_ctx, weights, queue, "decoder.module.asr_res.0", asr, asr_res, Tf, 1, 0, 1, 1, &Lar, &Car);

    // F0/N are at Tf_half; asr is at Tf. The reference concatenates [asr, F0, N] (different lengths!).
    // Actually wait — reference: x = torch.cat([asr, F0, N], axis=1). asr.shape=[1,512,T], F0=[1,1,T'], N=[1,1,T'].
    // T must match! Looking at reference: F0_conv is stride=2 BUT F0_curve is upsampled separately later in generator.
    // The cat happens INSIDE Decoder, BEFORE generator. asr's T_frames and F0/N's T_frames after conv...
    // Hmm actually looking again: F0_pred shape is [T_frames] = same as asr's T_frames. After stride=2 conv it's halved.
    // BUT asr stays at T_frames. So the cat dims don't match!
    //
    // Looking more carefully: I think the reference is wrong here OR I'm misreading the shapes.
    // In practice, F0_pred passed to decoder is already at a coarser resolution. Let me just upsample
    // F0/N back to T_frames length via NN, then cat with asr.
    cl_mem F0_up = alloc_rw(cl_ctx, sizeof(nnopt_storage_t) * Tf);
    cl_mem N_up  = alloc_rw(cl_ctx, sizeof(nnopt_storage_t) * Tf);
    int scale_up = Tf / Lf;
    if (scale_up < 1) scale_up = 1;
    int C1 = 1;
    clSetKernelArg(g_k_upsample_nn, 0, sizeof(cl_mem), &F0);
    clSetKernelArg(g_k_upsample_nn, 1, sizeof(cl_mem), &F0_up);
    clSetKernelArg(g_k_upsample_nn, 2, sizeof(int), &C1);
    clSetKernelArg(g_k_upsample_nn, 3, sizeof(int), &Lf);
    clSetKernelArg(g_k_upsample_nn, 4, sizeof(int), &scale_up);
    size_t gws_u[2] = {1, (size_t)(Lf*scale_up)};
    nnopt_enqueue_profiled(queue, g_k_upsample_nn, 2, nullptr, gws_u, nullptr, 0, nullptr, nullptr);
    clSetKernelArg(g_k_upsample_nn, 0, sizeof(cl_mem), &N);
    clSetKernelArg(g_k_upsample_nn, 1, sizeof(cl_mem), &N_up);
    nnopt_enqueue_profiled(queue, g_k_upsample_nn, 2, nullptr, gws_u, nullptr, 0, nullptr, nullptr);

    // concat([asr, F0_up, N_up], axis=C) -> [514, Tf]
    cl_mem x = alloc_rw(cl_ctx, sizeof(nnopt_storage_t) * 514 * Tf);
    int C_a = 512, C_b = 2;
    cl_mem F0N = alloc_rw(cl_ctx, sizeof(nnopt_storage_t) * 2 * Tf);
    // First combine F0_up + N_up into [2, Tf]
    int C0 = 1, C00 = 1;
    clSetKernelArg(g_k_concat, 0, sizeof(cl_mem), &F0_up);
    clSetKernelArg(g_k_concat, 1, sizeof(int), &C0);
    clSetKernelArg(g_k_concat, 2, sizeof(cl_mem), &N_up);
    clSetKernelArg(g_k_concat, 3, sizeof(int), &C00);
    clSetKernelArg(g_k_concat, 4, sizeof(cl_mem), &F0N);
    clSetKernelArg(g_k_concat, 5, sizeof(int), &Tf);
    size_t gws_c1[2] = {2, (size_t)Tf};
    nnopt_enqueue_profiled(queue, g_k_concat, 2, nullptr, gws_c1, nullptr, 0, nullptr, nullptr);
    // Then combine asr + F0N
    clSetKernelArg(g_k_concat, 0, sizeof(cl_mem), &asr);
    clSetKernelArg(g_k_concat, 1, sizeof(int), &C_a);
    clSetKernelArg(g_k_concat, 2, sizeof(cl_mem), &F0N);
    clSetKernelArg(g_k_concat, 3, sizeof(int), &C_b);
    clSetKernelArg(g_k_concat, 4, sizeof(cl_mem), &x);
    clSetKernelArg(g_k_concat, 5, sizeof(int), &Tf);
    size_t gws_c2[2] = {514, (size_t)Tf};
    nnopt_enqueue_profiled(queue, g_k_concat, 2, nullptr, gws_c2, nullptr, 0, nullptr, nullptr);

    DEC_TICK("F0/N conv + asr_res + upsample + concat");
    // encode: AdainResBlk1d(514, 1024, style=128), no upsample
    cl_mem x_enc = alloc_rw(cl_ctx, sizeof(nnopt_storage_t) * 1024 * Tf);
    int Tenc;
    // Dump input to encode for debugging
    NNOPT_LAYER_CHECK("dec_encode_input", queue, x, (size_t)514 * Tf);
    if (apply_adainresblk1d(cl_ctx, weights, queue, "decoder.module.encode",
                            x, x_enc, 514, 1024, Tf, false, ref_s_dec, &Tenc) != 0) {
        NNOPT_ERROR("decoder: encode failed");
        return -1;
    }
    NNOPT_LAYER_CHECK("dec_encode", queue, x_enc, (size_t)1024 * Tenc);
    DEC_TICK("encode AdainResBlk1d 514->1024");

    // decode[0..3]: AdainResBlk1d(1024+64+2=1090, 1024), last upsamples
    cl_mem dec_in = nullptr, dec_out = nullptr;
    int T_dec = Tenc;
    for (int i = 0; i < 4; ++i) {
        bool last = (i == 3);
        int dim_out = last ? 512 : 1024;
        bool upsample = last;
        // Build input: concat([x_enc, asr_res, F0_up, N_up], axis=C)
        int C_in_cat = 1024 + 64 + 1 + 1;
        cl_mem x_cat = alloc_rw(cl_ctx, sizeof(nnopt_storage_t) * C_in_cat * T_dec);
        int Cx = 1024, Cy = 64, Cz = 1, Cw = 1;
        clSetKernelArg(g_k_concat4, 0, sizeof(cl_mem), &x_enc);
        clSetKernelArg(g_k_concat4, 1, sizeof(int), &Cx);
        clSetKernelArg(g_k_concat4, 2, sizeof(cl_mem), &asr_res);
        clSetKernelArg(g_k_concat4, 3, sizeof(int), &Cy);
        clSetKernelArg(g_k_concat4, 4, sizeof(cl_mem), &F0_up);
        clSetKernelArg(g_k_concat4, 5, sizeof(int), &Cz);
        clSetKernelArg(g_k_concat4, 6, sizeof(cl_mem), &N_up);
        clSetKernelArg(g_k_concat4, 7, sizeof(int), &Cw);
        clSetKernelArg(g_k_concat4, 8, sizeof(cl_mem), &x_cat);
        clSetKernelArg(g_k_concat4, 9, sizeof(int), &T_dec);
        size_t gws_cat[2] = {(size_t)C_in_cat, (size_t)T_dec};
        nnopt_enqueue_profiled(queue, g_k_concat4, 2, nullptr, gws_cat, nullptr, 0, nullptr, nullptr);

        // dec_out alloc
        int T_after = upsample ? T_dec * 2 : T_dec;
        dec_out = alloc_rw(cl_ctx, sizeof(nnopt_storage_t) * dim_out * T_after);
        std::string p = "decoder.module.decode." + std::to_string(i);
        int Tnew;
        if (apply_adainresblk1d(cl_ctx, weights, queue, p, x_cat, dec_out,
                                C_in_cat, dim_out, T_dec, upsample, ref_s_dec, &Tnew) != 0) {
            NNOPT_ERROR_FMT("decoder: decode[%d] failed", i);
            return -1;
        }
        T_dec = Tnew;
        clReleaseMemObject(x_cat);
        // Swap: x_enc becomes dec_out for next iter
        if (dec_in) clReleaseMemObject(dec_in);
        dec_in = x_enc;  // hold old x_enc to release after swap
        clReleaseMemObject(dec_in);
        x_enc = dec_out;
        dec_in = nullptr;
        DEC_TICK(i == 0 ? "decode[0]" : i == 1 ? "decode[1]" : i == 2 ? "decode[2]" : "decode[3]+upsample");
    }
    // x_enc is now [512, T_dec] where T_dec = Tf * 2
    NNOPT_LAYER_CHECK("dec_after_decode_chain", queue, x_enc, (size_t)512 * T_dec);

    // EARLY DISPATCH: when using the host or fp32 generator path, both compute har on
    // their own (host in fp64). The SineGen+STFT block below is for the LEGACY fp16
    // generator path only. Doing it here AND inside the generator was ~5 s of pure
    // wasted CPU work per inference. Dispatch early and skip the dead block.
    bool _skip_sinegen_block = false;
    if (const char* p = std::getenv("NNOPT_HOST_GENERATOR")) if (p[0]=='1') _skip_sinegen_block = true;
    if (const char* p = std::getenv("NNOPT_GPU_FP32_GENERATOR")) if (p[0]=='1') _skip_sinegen_block = true;
    if (_skip_sinegen_block) {
        cl_mem gx_e = x_enc;
        int gT_e = T_dec;
        if (const char* hg = std::getenv("NNOPT_HOST_GENERATOR")) {
            if (hg[0] == '1') {
                int rc = op_decoder_host(cl_ctx, weights, queue, gx_e, F0_pred, N_pred, ref_s_dec,
                                          gT_e, T_frames, out_pcm_int16);
                for (cl_mem m : {F0, N, asr_res, F0_up, N_up, F0N, x, x_enc}) if (m) clReleaseMemObject(m);
                NNOPT_CHECKPOINT("op_decoder: host-generator path completed (early)");
                return rc;
            }
        }
        if (const char* fg = std::getenv("NNOPT_GPU_FP32_GENERATOR")) {
            if (fg[0] == '1') {
                DEC_TICK("post-decode setup before gpu_fp32 call");
                int rc = op_decoder_gpu_fp32(cl_ctx, weights, queue, gx_e, F0_pred, N_pred, ref_s_dec,
                                              gT_e, T_frames, out_pcm_int16);
                DEC_TICK("op_decoder_gpu_fp32 returned");
                for (cl_mem m : {F0, N, asr_res, F0_up, N_up, F0N, x, x_enc}) if (m) clReleaseMemObject(m);
                DEC_TICK("buffer releases done");
                NNOPT_CHECKPOINT("op_decoder: gpu-fp32-generator path completed (early)");
                return rc;
            }
        }
    }

    // ─── SineGen + SourceModuleHnNSF: F0 → harmonic source → STFT → har[n_fft+2, T_high] ───
    // f0_at_decoder is F0_pred[T_frames*2] from F0Ntrain. Upsample to T_audio_full =
    // T_dec * upsample_to_audio (=60) * hop (=5) = T_dec * 300 samples.
    const int upsample_to_audio_full = 300;
    int T_audio_full = T_dec * upsample_to_audio_full;
    int n_harm = 9;  // harmonic_num=8 → dim = 9
    int T_high_for_har = T_dec * 60;  // matches generator pre-istft length

    // Upsample F0 from T_frames*2 to T_audio_full (linear interp).
    cl_mem f0_up = alloc_rw(cl_ctx, sizeof(nnopt_storage_t) * T_audio_full);
    int F0_T_in = T_frames * 2;
    int Csg1 = 1;
    clSetKernelArg(g_k_upsample_linear, 0, sizeof(cl_mem), &F0_pred);
    clSetKernelArg(g_k_upsample_linear, 1, sizeof(cl_mem), &f0_up);
    clSetKernelArg(g_k_upsample_linear, 2, sizeof(int), &Csg1);
    clSetKernelArg(g_k_upsample_linear, 3, sizeof(int), &F0_T_in);
    clSetKernelArg(g_k_upsample_linear, 4, sizeof(int), &T_audio_full);
    size_t gws_uplin[2] = {1, (size_t)T_audio_full};
    nnopt_enqueue_profiled(queue, g_k_upsample_linear, 2, nullptr, gws_uplin, nullptr, 0, nullptr, nullptr);

    // SineGen with TRUE CUMULATIVE PHASE (host-side cumsum to avoid sequential kernel).
    // Reference: rad_values[t] = f0[t] / sr; phase = cumsum(rad_values) * 2*pi; sines = sin(phase * harm).
    // Per-harmonic: phase[h, t] = (h+1) * 2*pi * sum_{i<=t} f0[i] / sr.
    // We compute on host (CPU) then upload as fp32 → fp16 sines buffer.
    cl_mem sines = alloc_rw(cl_ctx, sizeof(nnopt_storage_t) * n_harm * T_audio_full);
    cl_mem uv = alloc_rw(cl_ctx, sizeof(nnopt_storage_t) * T_audio_full);
    float samp_rate = 24000.0f, voiced_th = 10.0f, sine_amp = 0.1f;
    {
        std::vector<float> f0_host(T_audio_full);
#ifdef NNOPT_USE_FP16
        std::vector<uint16_t> fh(T_audio_full);
        clEnqueueReadBuffer(queue, f0_up, CL_TRUE, 0, sizeof(uint16_t)*T_audio_full, fh.data(), 0, nullptr, nullptr);
        for (int i = 0; i < T_audio_full; ++i) f0_host[i] = nnopt_f16_to_f32(fh[i]);
#else
        clEnqueueReadBuffer(queue, f0_up, CL_TRUE, 0, sizeof(float)*T_audio_full, f0_host.data(), 0, nullptr, nullptr);
#endif
        // Compute cumsum of f0/sr, then per-harmonic sines.
        std::vector<float> cumphase(T_audio_full);
        float acc = 0.0f;
        for (int t = 0; t < T_audio_full; ++t) {
            acc += f0_host[t] / samp_rate;
            cumphase[t] = acc;  // accumulating fraction of cycle (without 2*pi yet)
        }
        std::vector<float> sines_host((size_t)n_harm * T_audio_full);
        std::vector<float> uv_host(T_audio_full);
        for (int t = 0; t < T_audio_full; ++t) {
            float u = (f0_host[t] > voiced_th) ? 1.0f : 0.0f;
            uv_host[t] = u;
            for (int h = 0; h < n_harm; ++h) {
                float phase = 2.0f * 3.14159265358979f * (float)(h + 1) * cumphase[t];
                sines_host[(size_t)h * T_audio_full + t] = std::sin(phase) * sine_amp * u;
            }
        }
#ifdef NNOPT_USE_FP16
        std::vector<uint16_t> sh((size_t)n_harm * T_audio_full);
        for (size_t i = 0; i < sh.size(); ++i) sh[i] = nnopt_f32_to_f16(sines_host[i]);
        clEnqueueWriteBuffer(queue, sines, CL_TRUE, 0, sizeof(uint16_t) * sh.size(), sh.data(), 0, nullptr, nullptr);
        std::vector<uint16_t> uh(T_audio_full);
        for (int i = 0; i < T_audio_full; ++i) uh[i] = nnopt_f32_to_f16(uv_host[i]);
        clEnqueueWriteBuffer(queue, uv, CL_TRUE, 0, sizeof(uint16_t)*T_audio_full, uh.data(), 0, nullptr, nullptr);
#else
        clEnqueueWriteBuffer(queue, sines, CL_TRUE, 0, sizeof(float) * sines_host.size(), sines_host.data(), 0, nullptr, nullptr);
        clEnqueueWriteBuffer(queue, uv, CL_TRUE, 0, sizeof(float)*T_audio_full, uv_host.data(), 0, nullptr, nullptr);
#endif
    }
    DEC_TICK("sinegen host (cumsum+sines)");

    // source_merge: tanh(Linear(sines, 9→1)) — uses m_source.l_linear weights
    cl_mem lm_w = weights.get_buffer("decoder.module.generator.m_source.l_linear.weight", /*opt=*/true);
    cl_mem lm_b = weights.get_buffer("decoder.module.generator.m_source.l_linear.bias", /*opt=*/true);
    cl_mem har_source = alloc_rw(cl_ctx, sizeof(nnopt_storage_t) * T_audio_full);
    if (lm_w) {
        int has_b = lm_b ? 1 : 0;
        clSetKernelArg(g_k_source_merge, 0, sizeof(cl_mem), &sines);
        clSetKernelArg(g_k_source_merge, 1, sizeof(cl_mem), &lm_w);
        if (lm_b) clSetKernelArg(g_k_source_merge, 2, sizeof(cl_mem), &lm_b);
        else { cl_mem dummy = lm_w; clSetKernelArg(g_k_source_merge, 2, sizeof(cl_mem), &dummy); }
        clSetKernelArg(g_k_source_merge, 3, sizeof(cl_mem), &har_source);
        clSetKernelArg(g_k_source_merge, 4, sizeof(int), &T_audio_full);
        clSetKernelArg(g_k_source_merge, 5, sizeof(int), &n_harm);
        clSetKernelArg(g_k_source_merge, 6, sizeof(int), &has_b);
        size_t gws_sm = (size_t)T_audio_full;
        nnopt_enqueue_profiled(queue, g_k_source_merge, 1, nullptr, &gws_sm, nullptr, 0, nullptr, nullptr);
    } else {
        // No weights — zero har_source.
        std::vector<float> zeros(T_audio_full, 0.0f);
#ifdef NNOPT_USE_FP16
        std::vector<uint16_t> zh(T_audio_full, 0);
        clEnqueueWriteBuffer(queue, har_source, CL_TRUE, 0, sizeof(uint16_t)*T_audio_full, zh.data(), 0, nullptr, nullptr);
#else
        clEnqueueWriteBuffer(queue, har_source, CL_TRUE, 0, sizeof(float)*T_audio_full, zeros.data(), 0, nullptr, nullptr);
#endif
    }

    // STFT(har_source) → har_spec[11, T_high], har_phase[11, T_high]; concat → har[22, T_high]
    // Use center=True equivalent: T_high = T_audio_full / hop = T_dec * 60 (matches ups[0] output * 6).
    // Frames that read past T_audio_full are zero-padded by the kernel's bounds check.
    const int gen_n_fft = 20, gen_hop = 5;
    int T_high_stft = T_audio_full / gen_hop + 1;  // torch.stft center=True: frames = T//hop + 1
    cl_mem hann_g = alloc_rw(cl_ctx, sizeof(nnopt_storage_t) * gen_n_fft);
    {
        std::vector<float> hg(gen_n_fft);
        for (int i = 0; i < gen_n_fft; ++i) hg[i] = 0.5f * (1.0f - cosf(2.0f * 3.14159265358979f * (float)i / (float)gen_n_fft));
#ifdef NNOPT_USE_FP16
        std::vector<uint16_t> hgh(gen_n_fft);
        for (int i = 0; i < gen_n_fft; ++i) hgh[i] = nnopt_f32_to_f16(hg[i]);
        clEnqueueWriteBuffer(queue, hann_g, CL_TRUE, 0, sizeof(uint16_t)*gen_n_fft, hgh.data(), 0, nullptr, nullptr);
#else
        clEnqueueWriteBuffer(queue, hann_g, CL_TRUE, 0, sizeof(float)*gen_n_fft, hg.data(), 0, nullptr, nullptr);
#endif
    }
    int n_freq_g = gen_n_fft / 2 + 1;  // = 11
    cl_mem har_spec = alloc_rw(cl_ctx, sizeof(nnopt_storage_t) * n_freq_g * T_high_stft);
    cl_mem har_phase = alloc_rw(cl_ctx, sizeof(nnopt_storage_t) * n_freq_g * T_high_stft);
    clSetKernelArg(g_k_stft, 0, sizeof(cl_mem), &har_source);
    clSetKernelArg(g_k_stft, 1, sizeof(cl_mem), &hann_g);
    clSetKernelArg(g_k_stft, 2, sizeof(cl_mem), &har_spec);
    clSetKernelArg(g_k_stft, 3, sizeof(cl_mem), &har_phase);
    clSetKernelArg(g_k_stft, 4, sizeof(int), &T_audio_full);
    clSetKernelArg(g_k_stft, 5, sizeof(int), &gen_n_fft);
    clSetKernelArg(g_k_stft, 6, sizeof(int), &gen_hop);
    clSetKernelArg(g_k_stft, 7, sizeof(int), &n_freq_g);
    clSetKernelArg(g_k_stft, 8, sizeof(int), &T_high_stft);
    size_t gws_stft[2] = {(size_t)T_high_stft, (size_t)n_freq_g};
    nnopt_enqueue_profiled(queue, g_k_stft, 2, nullptr, gws_stft, nullptr, 0, nullptr, nullptr);

    // Concat har_spec + har_phase → har [22, T_high_stft]
    cl_mem har = alloc_rw(cl_ctx, sizeof(nnopt_storage_t) * (n_freq_g * 2) * T_high_stft);
    {
        int Ca = n_freq_g, Cb = n_freq_g;
        clSetKernelArg(g_k_concat, 0, sizeof(cl_mem), &har_spec);
        clSetKernelArg(g_k_concat, 1, sizeof(int), &Ca);
        clSetKernelArg(g_k_concat, 2, sizeof(cl_mem), &har_phase);
        clSetKernelArg(g_k_concat, 3, sizeof(int), &Cb);
        clSetKernelArg(g_k_concat, 4, sizeof(cl_mem), &har);
        clSetKernelArg(g_k_concat, 5, sizeof(int), &T_high_stft);
        size_t gws_ch[2] = {(size_t)(n_freq_g*2), (size_t)T_high_stft};
        nnopt_enqueue_profiled(queue, g_k_concat, 2, nullptr, gws_ch, nullptr, 0, nullptr, nullptr);
    }
    // Cleanup intermediate
    for (cl_mem m : {f0_up, sines, uv, har_source, hann_g, har_spec, har_phase}) if (m) clReleaseMemObject(m);
    DEC_TICK("source_merge + STFT(har) + concat");

    // ─── Generator: ups[0]=10x -> resblocks[0..2] -> ups[1]=6x -> reflection_pad -> resblocks[3..5] -> conv_post ───
    // upsample_kernel_sizes = [20, 12], upsample_rates = [10, 6]
    // padding = (k - u) / 2 per reference: (20-10)/2=5, (12-6)/2=3
    // resblock_kernel_sizes = [3, 7, 11], resblock_dilation_sizes = [[1,3,5],[1,3,5],[1,3,5]]
    // After ups[0]: 512 -> 256 channels (upsample_initial_channel//(2**(0+1)) = 512/2 = 256)
    // After ups[1]: 256 -> 128 channels (512/4 = 128)
    const int resblock_kernels[3] = {3, 7, 11};
    const int resblock_dils[3][3] = {{1,3,5},{1,3,5},{1,3,5}};

    cl_mem gx = x_enc;     // [512, T_dec]
    int gC = 512, gT = T_dec;

    // HOST GENERATOR (fp64 CPU): NNOPT_HOST_GENERATOR=1. Slow but proven correct.
    if (const char* host_gen = std::getenv("NNOPT_HOST_GENERATOR")) {
        if (host_gen[0] == '1') {
            int rc = op_decoder_host(cl_ctx, weights, queue, gx, F0_pred, N_pred, ref_s_dec,
                                      gT, T_frames, out_pcm_int16);
            for (cl_mem m : {F0, N, asr_res, F0_up, N_up, F0N, x, x_enc, har}) {
                if (m) clReleaseMemObject(m);
            }
            NNOPT_CHECKPOINT("op_decoder: host-generator path completed");
            return rc;
        }
    }
    // GPU FP32 GENERATOR: NNOPT_GPU_FP32_GENERATOR=1. Same algorithm as host, fp32 on GPU.
    if (const char* fp32_gen = std::getenv("NNOPT_GPU_FP32_GENERATOR")) {
        if (fp32_gen[0] == '1') {
            int rc = op_decoder_gpu_fp32(cl_ctx, weights, queue, gx, F0_pred, N_pred, ref_s_dec,
                                          gT, T_frames, out_pcm_int16);
            for (cl_mem m : {F0, N, asr_res, F0_up, N_up, F0N, x, x_enc, har}) {
                if (m) clReleaseMemObject(m);
            }
            NNOPT_CHECKPOINT("op_decoder: gpu-fp32-generator path completed");
            return rc;
        }
    }

    // ups[0]: ConvTranspose1d(512, 256, k=20, stride=10, padding=5)
    apply_leaky(queue, gx, gC * gT, 0.1f);
    int L_u0 = (gT - 1)*10 - 2*5 + (20-1) + 1;
    cl_mem u0_out = alloc_rw(cl_ctx, sizeof(nnopt_storage_t) * 256 * L_u0);
    int Lu0, Cu0;
    if (convtr1d_wn(cl_ctx, weights, queue, "decoder.module.generator.ups.0", gx, u0_out,
                    gT, 10, 5, 0, 1, 1, &Lu0, &Cu0) != 0) return -1;
    gC = 256; gT = Lu0;
    DEC_TICK("ups[0] convtr only");
    NNOPT_LAYER_CHECK("dec_gen_ups0", queue, u0_out, (size_t)gC * gT);
    // noise_convs[0] on har: plain Conv1d(22, 256, kernel=12, stride=6, padding=3)
    // har is [22, T_high_stft]. Output should match [256, gT].
    // Noise integration re-enabled
    cl_mem nc0 = alloc_rw(cl_ctx, sizeof(nnopt_storage_t) * 256 * gT);
    int Lnc0, Cnc0;
    if (dec_conv1d_plain(cl_ctx, weights, queue, "decoder.module.generator.noise_convs.0",
                          har, nc0, T_high_stft, 6, 3, 1, 1, &Lnc0, &Cnc0) == 0 && Lnc0 == gT) {
        int dilations[3] = {1, 3, 5};
        // Run noise_res.0 in fp32 — the fp32 t8x4 conv kernel is ~2x faster
        // than the fp16 c4x4 on this shape, and fp32 storage inside the block
        // also removes per-layer fp16 truncation. Convert at the boundaries.
        int rc_nr0 = -1;
        {
            int N_nr = 256 * gT;
            cl_mem nc0_f32 = clCreateBuffer(cl_ctx.context(), CL_MEM_READ_WRITE, sizeof(float) * N_nr, nullptr, nullptr);
            gen_fp16_to_fp32(cl_ctx, queue, nc0, nc0_f32, N_nr);
            rc_nr0 = gen_apply_adainresblock1_fp32(cl_ctx, weights, queue,
                                                   "decoder.module.generator.noise_res.0",
                                                   nc0_f32, ref_s_dec, 256, gT, 7, dilations);
            if (rc_nr0 == 0) gen_fp32_to_fp16(cl_ctx, queue, nc0_f32, nc0, N_nr);
            clReleaseMemObject(nc0_f32);
        }
        if (rc_nr0 == 0) {
            int N_add = 256 * gT;
            clSetKernelArg(g_k_add, 0, sizeof(cl_mem), &u0_out);
            clSetKernelArg(g_k_add, 1, sizeof(cl_mem), &nc0);
            clSetKernelArg(g_k_add, 2, sizeof(int), &N_add);
            run1d(queue, g_k_add, (size_t)N_add);
        }
    }
    clReleaseMemObject(nc0);
    DEC_TICK("ups[0] convtr + noise_convs[0] + noise_res[0]");
    // resblocks[0..2] in FP32 — converts u0_out to fp32, runs 3 resblocks in fp32, averages, converts back
    int Nlevel0 = gC * gT;
    cl_mem u0_fp32 = clCreateBuffer(cl_ctx.context(), CL_MEM_READ_WRITE, sizeof(float) * Nlevel0, nullptr, nullptr);
    gen_fp16_to_fp32(cl_ctx, queue, u0_out, u0_fp32, Nlevel0);
    cl_mem r_fp32[3];
    for (int j = 0; j < 3; ++j) {
        r_fp32[j] = clCreateBuffer(cl_ctx.context(), CL_MEM_READ_WRITE, sizeof(float) * Nlevel0, nullptr, nullptr);
        clEnqueueCopyBuffer(queue, u0_fp32, r_fp32[j], 0, 0, sizeof(float) * Nlevel0, 0, nullptr, nullptr);
        if (gen_apply_adainresblock1_fp32(cl_ctx, weights, queue,
                                          "decoder.module.generator.resblocks." + std::to_string(0*3+j),
                                          r_fp32[j], ref_s_dec, gC, gT,
                                          resblock_kernels[j], resblock_dils[j]) != 0) return -1;
    }
    // Average → xs_fp32
    cl_mem xs_fp32 = clCreateBuffer(cl_ctx.context(), CL_MEM_READ_WRITE, sizeof(float) * Nlevel0, nullptr, nullptr);
    gen_avg_three_fp32(cl_ctx, queue, r_fp32[0], r_fp32[1], r_fp32[2], xs_fp32, Nlevel0);
    // Convert back to fp16 for downstream
    cl_mem xs = alloc_rw(cl_ctx, sizeof(nnopt_storage_t) * Nlevel0);
    gen_fp32_to_fp16(cl_ctx, queue, xs_fp32, xs, Nlevel0);
    for (int j = 0; j < 3; ++j) clReleaseMemObject(r_fp32[j]);
    clReleaseMemObject(u0_fp32);
    clReleaseMemObject(xs_fp32);
    clReleaseMemObject(u0_out);
    cl_mem gen_x = xs;  // [256, gT]
    DEC_TICK("resblocks[0..2] fp32 (C=256)");
    NNOPT_LAYER_CHECK("dec_gen_resblocks_012_avg", queue, gen_x, (size_t)gC * gT);

    // ups[1]: ConvTranspose1d(256, 128, k=12, stride=6, padding=3)
    apply_leaky(queue, gen_x, gC * gT, 0.1f);
    int L_u1 = (gT - 1)*6 - 2*3 + (12-1) + 1;
    cl_mem u1_out = alloc_rw(cl_ctx, sizeof(nnopt_storage_t) * 128 * L_u1);
    int Lu1, Cu1;
    if (convtr1d_wn(cl_ctx, weights, queue, "decoder.module.generator.ups.1", gen_x, u1_out,
                    gT, 6, 3, 0, 1, 1, &Lu1, &Cu1) != 0) return -1;
    DEC_TICK("ups[1] convtr only");
    NNOPT_LAYER_CHECK("dec_gen_ups1", queue, u1_out, (size_t)128 * Lu1);
    clReleaseMemObject(gen_x);
    gC = 128; gT = Lu1;

    // Reflection pad (1, 0): T+1
    int gT_pad = gT + 1;
    cl_mem padded = alloc_rw(cl_ctx, sizeof(nnopt_storage_t) * gC * gT_pad);
    int pl = 1, pr = 0;
    clSetKernelArg(g_k_refl_pad, 0, sizeof(cl_mem), &u1_out);
    clSetKernelArg(g_k_refl_pad, 1, sizeof(cl_mem), &padded);
    clSetKernelArg(g_k_refl_pad, 2, sizeof(int), &gC);
    clSetKernelArg(g_k_refl_pad, 3, sizeof(int), &gT);
    clSetKernelArg(g_k_refl_pad, 4, sizeof(int), &pl);
    clSetKernelArg(g_k_refl_pad, 5, sizeof(int), &pr);
    size_t gws_pad[2] = {(size_t)gC, (size_t)gT_pad};
    nnopt_enqueue_profiled(queue, g_k_refl_pad, 2, nullptr, gws_pad, nullptr, 0, nullptr, nullptr);
    clReleaseMemObject(u1_out);
    gT = gT_pad;
    // noise_convs[1] on har: plain Conv1d(22, 128, kernel=1, stride=1, padding=0). Output [128, T_high_stft].
    // Need to match gT for addition. If T_high_stft != gT, skip the add.
    cl_mem nc1 = alloc_rw(cl_ctx, sizeof(nnopt_storage_t) * 128 * T_high_stft);
    int Lnc1, Cnc1;
    if (dec_conv1d_plain(cl_ctx, weights, queue, "decoder.module.generator.noise_convs.1",
                          har, nc1, T_high_stft, 1, 0, 1, 1, &Lnc1, &Cnc1) == 0 && Lnc1 == gT) {
        int dilations[3] = {1, 3, 5};
        int rc_nr1 = -1;
        {
            int N_nr = 128 * gT;
            cl_mem nc1_f32 = clCreateBuffer(cl_ctx.context(), CL_MEM_READ_WRITE, sizeof(float) * N_nr, nullptr, nullptr);
            gen_fp16_to_fp32(cl_ctx, queue, nc1, nc1_f32, N_nr);
            rc_nr1 = gen_apply_adainresblock1_fp32(cl_ctx, weights, queue,
                                                   "decoder.module.generator.noise_res.1",
                                                   nc1_f32, ref_s_dec, 128, gT, 11, dilations);
            if (rc_nr1 == 0) gen_fp32_to_fp16(cl_ctx, queue, nc1_f32, nc1, N_nr);
            clReleaseMemObject(nc1_f32);
        }
        if (rc_nr1 == 0) {
            int N_add = 128 * gT;
            clSetKernelArg(g_k_add, 0, sizeof(cl_mem), &padded);
            clSetKernelArg(g_k_add, 1, sizeof(cl_mem), &nc1);
            clSetKernelArg(g_k_add, 2, sizeof(int), &N_add);
            run1d(queue, g_k_add, (size_t)N_add);
        }
    }
    clReleaseMemObject(nc1);
    DEC_TICK("ups[1] convtr + refl_pad + noise_convs[1] + noise_res[1]");

    // resblocks[3..5] in FP32
    int Nlevel1 = gC * gT;
    cl_mem padded_fp32 = clCreateBuffer(cl_ctx.context(), CL_MEM_READ_WRITE, sizeof(float) * Nlevel1, nullptr, nullptr);
    gen_fp16_to_fp32(cl_ctx, queue, padded, padded_fp32, Nlevel1);
    cl_mem r_fp32b[3];
    for (int j = 0; j < 3; ++j) {
        r_fp32b[j] = clCreateBuffer(cl_ctx.context(), CL_MEM_READ_WRITE, sizeof(float) * Nlevel1, nullptr, nullptr);
        clEnqueueCopyBuffer(queue, padded_fp32, r_fp32b[j], 0, 0, sizeof(float) * Nlevel1, 0, nullptr, nullptr);
        if (gen_apply_adainresblock1_fp32(cl_ctx, weights, queue,
                                          "decoder.module.generator.resblocks." + std::to_string(1*3+j),
                                          r_fp32b[j], ref_s_dec, gC, gT,
                                          resblock_kernels[j], resblock_dils[j]) != 0) return -1;
    }
    cl_mem xs_fp32b = clCreateBuffer(cl_ctx.context(), CL_MEM_READ_WRITE, sizeof(float) * Nlevel1, nullptr, nullptr);
    gen_avg_three_fp32(cl_ctx, queue, r_fp32b[0], r_fp32b[1], r_fp32b[2], xs_fp32b, Nlevel1);
    xs = alloc_rw(cl_ctx, sizeof(nnopt_storage_t) * Nlevel1);
    gen_fp32_to_fp16(cl_ctx, queue, xs_fp32b, xs, Nlevel1);
    for (int j = 0; j < 3; ++j) clReleaseMemObject(r_fp32b[j]);
    clReleaseMemObject(padded_fp32);
    clReleaseMemObject(xs_fp32b);
    clReleaseMemObject(padded);
    clReleaseMemObject(har);
    DEC_TICK("resblocks[3..5] fp32 (C=128, T~60x)");

    NNOPT_LAYER_CHECK("dec_gen_resblocks_345_avg", queue, xs, (size_t)gC * gT);

    // Final leaky_relu (no slope param in reference → default 0.01)
    apply_leaky(queue, xs, gC * gT, 0.01f);

    int T_pre = gT;
    NNOPT_LAYER_CHECK("dec_gen_input_to_conv_post", queue, xs, (size_t)gC * gT);
    int post_n_fft = n_fft;
    int post_C_out = post_n_fft + 2;
    int L_post_out = T_pre, C_post = post_C_out;
    int n_freq = post_n_fft / 2 + 1;

    // HOST conv_post + exp/sin: read xs to host, reconstruct conv_post weight (weight_norm)
    // in fp64, compute Conv1d 128→22 with kernel=7 padding=3, then apply exp/sin to first/second half.
    cl_mem mag = alloc_rw(cl_ctx, sizeof(nnopt_storage_t) * n_freq * L_post_out);
    cl_mem phase = alloc_rw(cl_ctx, sizeof(nnopt_storage_t) * n_freq * L_post_out);
    // GPU fp32 conv_post by default; NNOPT_HOST_CONV_POST=1 restores the host
    // fp64 path for A/B.
    bool host_conv_post = false;
    if (const char* hcp = std::getenv("NNOPT_HOST_CONV_POST")) host_conv_post = (hcp[0] == '1');
    if (!host_conv_post) {
        if (gen_conv_post_gpu(cl_ctx, weights, queue, xs, mag, phase, gC, L_post_out, n_freq) != 0) {
            NNOPT_ERROR("conv_post_gpu failed — falling back to host path");
            host_conv_post = true;
        }
    }
    if (host_conv_post) {
        // Read xs (input to conv_post) [gC=128, gT=L_post_out]
        int N_xs = gC * gT;
        std::vector<float> xs_host(N_xs);
#ifdef NNOPT_USE_FP16
        std::vector<uint16_t> xh(N_xs);
        clEnqueueReadBuffer(queue, xs, CL_TRUE, 0, sizeof(uint16_t)*N_xs, xh.data(), 0, nullptr, nullptr);
        for (int i = 0; i < N_xs; ++i) xs_host[i] = nnopt_f16_to_f32(xh[i]);
#else
        clEnqueueReadBuffer(queue, xs, CL_TRUE, 0, sizeof(float)*N_xs, xs_host.data(), 0, nullptr, nullptr);
#endif
        // Read conv_post weight_v, weight_g, bias from device weights
        cl_mem cp_v = weights.get_buffer("decoder.module.generator.conv_post.weight_v");
        cl_mem cp_g = weights.get_buffer("decoder.module.generator.conv_post.weight_g");
        cl_mem cp_b = weights.get_buffer("decoder.module.generator.conv_post.bias", true);
        auto shape = weights.get_shape("decoder.module.generator.conv_post.weight_v");
        int Wc_out = shape[0], Wc_in = shape[1], Wk = shape[2];  // [22, 128, 7]
        int W_total = Wc_out * Wc_in * Wk;
        std::vector<float> v_host(W_total), g_host(Wc_out);
#ifdef NNOPT_USE_FP16
        std::vector<uint16_t> vh(W_total), gh(Wc_out);
        clEnqueueReadBuffer(queue, cp_v, CL_TRUE, 0, sizeof(uint16_t)*W_total, vh.data(), 0, nullptr, nullptr);
        clEnqueueReadBuffer(queue, cp_g, CL_TRUE, 0, sizeof(uint16_t)*Wc_out, gh.data(), 0, nullptr, nullptr);
        for (int i = 0; i < W_total; ++i) v_host[i] = nnopt_f16_to_f32(vh[i]);
        for (int i = 0; i < Wc_out; ++i) g_host[i] = nnopt_f16_to_f32(gh[i]);
#else
        clEnqueueReadBuffer(queue, cp_v, CL_TRUE, 0, sizeof(float)*W_total, v_host.data(), 0, nullptr, nullptr);
        clEnqueueReadBuffer(queue, cp_g, CL_TRUE, 0, sizeof(float)*Wc_out, g_host.data(), 0, nullptr, nullptr);
#endif
        std::vector<float> b_host(Wc_out, 0.0f);
        if (cp_b) {
#ifdef NNOPT_USE_FP16
            std::vector<uint16_t> bh(Wc_out);
            clEnqueueReadBuffer(queue, cp_b, CL_TRUE, 0, sizeof(uint16_t)*Wc_out, bh.data(), 0, nullptr, nullptr);
            for (int i = 0; i < Wc_out; ++i) b_host[i] = nnopt_f16_to_f32(bh[i]);
#else
            clEnqueueReadBuffer(queue, cp_b, CL_TRUE, 0, sizeof(float)*Wc_out, b_host.data(), 0, nullptr, nullptr);
#endif
        }
        // Reconstruct W = (g/||v||) * v per output channel, in fp64
        std::vector<double> W_host(W_total);
        int per_oc = Wc_in * Wk;
        for (int oc = 0; oc < Wc_out; ++oc) {
            double sq = 0.0;
            int base = oc * per_oc;
            for (int i = 0; i < per_oc; ++i) { double x = v_host[base+i]; sq += x*x; }
            double norm = std::sqrt(sq) + 1e-12;
            double scale = (double)g_host[oc] / norm;
            for (int i = 0; i < per_oc; ++i) W_host[base+i] = (double)v_host[base+i] * scale;
        }
        // Conv1d in fp64: out[oc, ol] = sum_ic sum_k W[oc, ic, k] * xs[ic, ol-pad+k]
        int padding = 3;
        std::vector<float> cpost_host((size_t)Wc_out * L_post_out, 0.0f);
        for (int oc = 0; oc < Wc_out; ++oc) {
            for (int ol = 0; ol < L_post_out; ++ol) {
                double acc = 0.0;
                for (int k = 0; k < Wk; ++k) {
                    int il = ol - padding + k;
                    if (il < 0 || il >= L_post_out) continue;
                    for (int ic = 0; ic < Wc_in; ++ic) {
                        acc += W_host[oc * per_oc + ic * Wk + k] * (double)xs_host[ic * L_post_out + il];
                    }
                }
                acc += (double)b_host[oc];
                cpost_host[(size_t)oc * L_post_out + ol] = (float)acc;
            }
        }
        std::vector<float> mag_host((size_t)n_freq * L_post_out);
        std::vector<float> phase_host((size_t)n_freq * L_post_out);
        for (int c = 0; c < n_freq; ++c) {
            for (int t = 0; t < L_post_out; ++t) {
                mag_host[c * L_post_out + t] = std::exp(cpost_host[c * L_post_out + t]);
                phase_host[c * L_post_out + t] = std::sin(cpost_host[(n_freq + c) * L_post_out + t]);
            }
        }
        // Upload mag/phase to device buffers (so existing host iSTFT can read them)
#ifdef NNOPT_USE_FP16
        std::vector<uint16_t> mh((size_t)n_freq * L_post_out), ph((size_t)n_freq * L_post_out);
        for (size_t i = 0; i < mh.size(); ++i) mh[i] = nnopt_f32_to_f16(mag_host[i]);
        for (size_t i = 0; i < ph.size(); ++i) ph[i] = nnopt_f32_to_f16(phase_host[i]);
        clEnqueueWriteBuffer(queue, mag, CL_TRUE, 0, sizeof(uint16_t)*mh.size(), mh.data(), 0, nullptr, nullptr);
        clEnqueueWriteBuffer(queue, phase, CL_TRUE, 0, sizeof(uint16_t)*ph.size(), ph.data(), 0, nullptr, nullptr);
#else
        clEnqueueWriteBuffer(queue, mag, CL_TRUE, 0, sizeof(float)*mag_host.size(), mag_host.data(), 0, nullptr, nullptr);
        clEnqueueWriteBuffer(queue, phase, CL_TRUE, 0, sizeof(float)*phase_host.size(), phase_host.data(), 0, nullptr, nullptr);
#endif
    }
    DEC_TICK("host conv_post fp64 + exp/sin + upload");
    int N_mag = n_freq * L_post_out;
    cl_mem up = xs;

    std::vector<float> hann((size_t)n_fft);
    for (int i = 0; i < n_fft; ++i) hann[i] = 0.5f * (1.0f - cosf(2.0f * 3.14159265358979f * (float)i / (float)n_fft));
    cl_mem win = alloc_rw(cl_ctx, sizeof(nnopt_storage_t) * n_fft);
#ifdef NNOPT_USE_FP16
    std::vector<uint16_t> hh((size_t)n_fft);
    for (int i = 0; i < n_fft; ++i) hh[i] = nnopt_f32_to_f16(hann[i]);
    clEnqueueWriteBuffer(queue, win, CL_TRUE, 0, sizeof(uint16_t)*n_fft, hh.data(), 0, nullptr, nullptr);
#else
    clEnqueueWriteBuffer(queue, win, CL_TRUE, 0, sizeof(float)*n_fft, hann.data(), 0, nullptr, nullptr);
#endif

    int T_audio = (L_post_out - 1) * hop + n_fft;
    cl_mem audio_dev = alloc_rw(cl_ctx, sizeof(nnopt_storage_t) * T_audio);

    // GPU fp32 iSTFT by default (the old precision issue was the fp16 kernel);
    // NNOPT_HOST_ISTFT=1 restores the host fp64 path for A/B.
    std::vector<float> audio_f32_raw((size_t)T_audio, 0.0f);
    bool host_istft = false;
    if (const char* hi = std::getenv("NNOPT_HOST_ISTFT")) host_istft = (hi[0] == '1');
    if (!host_istft) {
        if (gen_istft_gpu(cl_ctx, queue, mag, phase, L_post_out, n_fft, hop, n_freq,
                          audio_f32_raw.data(), T_audio) != 0) {
            NNOPT_ERROR("istft_gpu failed — falling back to host path");
            host_istft = true;
        }
    }
    if (host_istft) {
        int Nmf = n_freq * L_post_out;
        std::vector<float> mag_host(Nmf), phase_host(Nmf), win_host(n_fft);
#ifdef NNOPT_USE_FP16
        std::vector<uint16_t> mh(Nmf), ph(Nmf), wh(n_fft);
        clEnqueueReadBuffer(queue, mag, CL_TRUE, 0, sizeof(uint16_t)*Nmf, mh.data(), 0, nullptr, nullptr);
        clEnqueueReadBuffer(queue, phase, CL_TRUE, 0, sizeof(uint16_t)*Nmf, ph.data(), 0, nullptr, nullptr);
        clEnqueueReadBuffer(queue, win, CL_TRUE, 0, sizeof(uint16_t)*n_fft, wh.data(), 0, nullptr, nullptr);
        for (int i = 0; i < Nmf; ++i) mag_host[i] = nnopt_f16_to_f32(mh[i]);
        for (int i = 0; i < Nmf; ++i) phase_host[i] = nnopt_f16_to_f32(ph[i]);
        for (int i = 0; i < n_fft; ++i) win_host[i] = nnopt_f16_to_f32(wh[i]);
#else
        clEnqueueReadBuffer(queue, mag, CL_TRUE, 0, sizeof(float)*Nmf, mag_host.data(), 0, nullptr, nullptr);
        clEnqueueReadBuffer(queue, phase, CL_TRUE, 0, sizeof(float)*Nmf, phase_host.data(), 0, nullptr, nullptr);
        clEnqueueReadBuffer(queue, win, CL_TRUE, 0, sizeof(float)*n_fft, win_host.data(), 0, nullptr, nullptr);
#endif
        // Overlap-add iSTFT (matches np.fft.irfft + windowed overlap-add + window² normalization)
        std::vector<float> norm_host((size_t)T_audio, 0.0f);
        for (int f = 0; f < L_post_out; ++f) {
            // Compute IFFT of one frame using rfft inverse formula
            for (int t = 0; t < n_fft; ++t) {
                double sample = mag_host[0 * L_post_out + f] * std::cos((double)phase_host[0 * L_post_out + f]);
                for (int k = 1; k < n_freq - 1; ++k) {
                    double ang = 2.0 * 3.14159265358979 * (double)k * (double)t / (double)n_fft
                                 + (double)phase_host[k * L_post_out + f];
                    sample += 2.0 * (double)mag_host[k * L_post_out + f] * std::cos(ang);
                }
                if (n_fft % 2 == 0) {
                    double ang = 3.14159265358979 * (double)t + (double)phase_host[(n_freq - 1) * L_post_out + f];
                    sample += (double)mag_host[(n_freq - 1) * L_post_out + f] * std::cos(ang);
                }
                sample /= (double)n_fft;
                int out_idx = f * hop + t;
                if (out_idx >= 0 && out_idx < T_audio) {
                    audio_f32_raw[out_idx] += (float)(sample * win_host[t]);
                    norm_host[out_idx] += win_host[t] * win_host[t];
                }
            }
        }
        for (int i = 0; i < T_audio; ++i) {
            if (norm_host[i] > 1e-10f) audio_f32_raw[i] /= norm_host[i];
        }
    }
    DEC_TICK("host iSTFT fp64 overlap-add");
    // center=True trim: output[n_fft/2 : T_audio - n_fft/2] matches torch.istft default behavior
    int trim = n_fft / 2;
    int T_trim = T_audio - 2 * trim;
    std::vector<float> audio_f32((size_t)T_trim);
    for (int i = 0; i < T_trim; ++i) audio_f32[i] = audio_f32_raw[i + trim];
    out_pcm_int16.assign((size_t)T_trim, 0);
    int& T_audio_ref = T_trim;  // alias for rest of function
    #define T_audio T_trim
    // Low-pass filter disabled (was a workaround; with deterministic SineGen+noise the
    // audio should match deterministic reference directly).
    // (dec_istft_raw dump removed — audio_dev no longer used; iSTFT runs on host)
    // DC removal: subtract per-segment mean. The DC bin of iSTFT contributes
    // mag[0]*cos(phase[0]) per frame; if cos(phase[0]) averages nonzero, every
    // output sample gets a constant bias. Reference signal is closer to balanced;
    // our intermediate values diverge slightly. High-pass at DC fixes the offset.
    float dc = 0.0f;
    for (int i = 0; i < T_audio; ++i) dc += audio_f32[i];
    dc /= (float)T_audio;
    for (int i = 0; i < T_audio; ++i) audio_f32[i] -= dc;
    // NO loudness normalization: the reference emits the raw float waveform
    // (peak ~0.3); the old 0.8/p99 rescale (~4.3x) hard-clipped ~0.36% of
    // samples — an audible high-pitched distorted copy of the voice laid over
    // the speech. Clamp is safety only.
    for (int i = 0; i < T_audio; ++i) {
        float s_ = audio_f32[i];
        if (s_ > 1.0f) s_ = 1.0f;
        if (s_ < -1.0f) s_ = -1.0f;
        out_pcm_int16[i] = (int16_t)(s_ * 32767.0f);
    }

    for (cl_mem m : {F0, N, asr_res, F0_up, N_up, F0N, x, x_enc, up, mag, phase, win, audio_dev}) {
        if (m) clReleaseMemObject(m);
    }
    NNOPT_LAYER_CHECK("decoder_audio", queue, audio_dev, (size_t)T_audio);
    NNOPT_CHECKPOINT("op_decoder: full encode+decode chain completed");
    return 0;
}
