// Mixed-precision fp32 implementation of the generator's AdaINResBlock1 chain.
// Keeps fp16 weights but uses fp32 storage for activations between layers.
// Eliminates per-layer fp16 truncation that compounds across ~120 round-trips
// in the generator. Final output converted back to fp16 at boundaries.
//
// Public entry:
//   gen_apply_adainresblock1_fp32 — fp32 version of decoder.cpp::apply_adainresblock1
//   gen_fp16_to_fp32 / gen_fp32_to_fp16 — buffer conversion at boundaries
//   gen_avg_three_fp32 — average 3 fp32 buffers

#include "opencl_context.h"
#include "weights.h"
#include "debug_utils.h"
#include "profiler.h"
#include "model_config.h"
#include "utils.h"

#include <CL/cl.h>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <string>
#include <unordered_map>
#include <vector>

static cl_program g_gf_prog = nullptr;
static cl_kernel  g_kf_f16_to_f32 = nullptr;
static cl_kernel  g_kf_f32_to_f16 = nullptr;
static cl_kernel  g_kf_instnorm = nullptr;
static cl_kernel  g_kf_adain_combine = nullptr;
static cl_kernel  g_kf_snake1d = nullptr;
static cl_kernel  g_kf_conv1d = nullptr;
static cl_kernel  g_kf_conv1d_t4x4 = nullptr;
static cl_kernel  g_kf_conv1d_t8x4 = nullptr;
static cl_kernel  g_kf_conv1d_t8x4v = nullptr;
static cl_kernel  g_kf_weightnorm_kpad = nullptr;
static cl_kernel  g_kf_bias_add = nullptr;
static cl_kernel  g_kf_add = nullptr;
static cl_kernel  g_kf_linear_apply = nullptr;
static cl_kernel  g_kf_split2 = nullptr;
static cl_kernel  g_kf_weightnorm = nullptr;
static cl_kernel  g_kf_avg3 = nullptr;
static cl_kernel  g_kf_convtr1d = nullptr;
static cl_kernel  g_kf_convtr1d_c4 = nullptr;
static cl_kernel  g_kf_leaky = nullptr;
static cl_kernel  g_kf_copy_f32 = nullptr;
static cl_kernel  g_kf_nc_pack = nullptr;
static cl_kernel  g_kf_plain_conv1d_tex = nullptr;
static cl_kernel  g_kf_refl_pad = nullptr;
static cl_kernel  g_kf_plain_conv1d = nullptr;
static cl_kernel  g_kf_conv_post_split = nullptr;
static cl_kernel  g_kf_in_adain_snake = nullptr;
static cl_kernel  g_kf_in_adain_snake_h = nullptr;
static cl_kernel  g_kf_conv1d_h8x4 = nullptr;
static cl_kernel  g_kf_conv1d_hh = nullptr;
static cl_kernel  g_kf_weightnorm_h = nullptr;
static cl_kernel  g_kf_in_adain_snake_h2h = nullptr;
static cl_kernel  g_kf_add_from_h = nullptr;
static cl_kernel  g_kf_conv1d_hh_lds = nullptr;
static cl_kernel  g_kf_conv1d_ht = nullptr;
static cl_kernel  g_kf_istft = nullptr;
static cl_kernel  g_kf_conv1d_ht8 = nullptr;
static cl_kernel  g_kf_conv1d_ht48 = nullptr;
static cl_kernel  g_kf_weightnorm_h_kpad = nullptr;
static cl_kernel  g_kf_wn_row_scale = nullptr;
static cl_kernel  g_kf_wn_write_image = nullptr;
static cl_kernel  g_kf_gf_convtr_pack = nullptr;
static cl_kernel  g_kf_convtr1d_c4x4_tex_f32 = nullptr;
static cl_kernel  g_kf_div_scalar = nullptr;
static cl_kernel  g_kf_exp = nullptr;
static cl_kernel  g_kf_sin_act = nullptr;

static const char* k_gf_src = R"CLC(
#pragma OPENCL EXTENSION cl_khr_fp16 : enable

// Convert fp16 storage -> fp32 buffer
__kernel void cvt_f16_to_f32(__global const half* in, __global float* out, int N) {
    int i = get_global_id(0); if (i >= N) return;
    out[i] = (float)vload_half(i, in);
}
__kernel void cvt_f32_to_f16(__global const float* in, __global half* out, int N) {
    int i = get_global_id(0); if (i >= N) return;
    vstore_half(in[i], i, out);
}

// All ops below operate on fp32 activations (in/out) with fp16 weights.
// Workgroup-per-channel instance norm (64-lane LDS tree reduction).
__kernel void instnorm_f32(__global const float* x,
                           __global float* y,
                           int C, int T, float eps) {
    int c  = get_group_id(0);
    int lt = get_local_id(0);
    if (c >= C) return;
    int base = c * T;
    __local float lsum[64];
    __local float lsq[64];
    float s = 0.0f, q = 0.0f;
    for (int t = lt; t < T; t += 64) { float v = x[base + t]; s += v; q += v * v; }
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
    for (int t = lt; t < T; t += 64) y[base + t] = (x[base + t] - mean) * inv;
}

// AdaIN combine: y = (1+gamma)*xn + beta. gamma/beta passed as fp32 (already converted).
__kernel void adain_combine_f32(__global const float* x,
                                __global const float* gamma,
                                __global const float* beta,
                                __global float* y, int C, int T) {
    int c = get_global_id(0); int t = get_global_id(1);
    if (c >= C || t >= T) return;
    y[c*T + t] = (1.0f + gamma[c]) * x[c*T + t] + beta[c];
}

// Snake1D: y = x + (1/a) * sin(a*x)^2
__kernel void snake1d_f32(__global float* y,
                          __global const half* alpha_fp16,
                          int C, int T) {
    int c = get_global_id(0); int t = get_global_id(1);
    if (c >= C || t >= T) return;
    float a = (float)vload_half(c, alpha_fp16);
    if (fabs(a) < 1e-6f) a = (a >= 0.0f ? 1e-6f : -1e-6f);
    float x = y[c*T + t];
    float s = sin(a * x);
    y[c*T + t] = x + (1.0f / a) * s * s;
}

// Fused instance-norm + AdaIN combine + snake activation: one read pass for
// the stats, one write pass for norm+style+activation. Replaces three full
// [C,T] kernel passes (instnorm -> adain_combine -> snake).
__kernel void instnorm_adain_snake_f32(__global const float* x,
                                       __global float* y,
                                       __global const float* gamma,
                                       __global const float* beta,
                                       __global const half* alpha,
                                       int C, int T, float eps) {
    int c  = get_group_id(0);
    int lt = get_local_id(0);
    if (c >= C) return;
    int base = c * T;
    __local float lsum[64];
    __local float lsq[64];
    float s = 0.0f, q = 0.0f;
    for (int t = lt; t < T; t += 64) { float v = x[base + t]; s += v; q += v * v; }
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
    float g = gamma[c], b = beta[c];
    float a = (float)vload_half(c, alpha);
    if (fabs(a) < 1e-6f) a = (a >= 0.0f ? 1e-6f : -1e-6f);
    float ra = 1.0f / a;
    for (int t = lt; t < T; t += 64) {
        float v = (x[base + t] - mean) * inv;
        v = (1.0f + g) * v + b;
        float sn = native_sin(a * v);
        y[base + t] = v + ra * sn * sn;
    }
}

// Same as instnorm_adain_snake_f32 but writes fp16 — used for the buffers
// that only feed the next conv (tmp1/tmp2), halving that conv's input traffic.
// The fp32 residual spine is untouched, so truncation does not compound.
__kernel void instnorm_adain_snake_f16out(__global const float* x,
                                          __global half* y,
                                          __global const float* gamma,
                                          __global const float* beta,
                                          __global const half* alpha,
                                          int C, int T, float eps) {
    int c  = get_group_id(0);
    int lt = get_local_id(0);
    if (c >= C) return;
    int base = c * T;
    __local float lsum[64];
    __local float lsq[64];
    float s = 0.0f, q = 0.0f;
    for (int t = lt; t < T; t += 64) { float v = x[base + t]; s += v; q += v * v; }
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
    float g = gamma[c], b = beta[c];
    float a = (float)vload_half(c, alpha);
    if (fabs(a) < 1e-6f) a = (a >= 0.0f ? 1e-6f : -1e-6f);
    float ra = 1.0f / a;
    for (int t = lt; t < T; t += 64) {
        float v = (x[base + t] - mean) * inv;
        v = (1.0f + g) * v + b;
        float sn = native_sin(a * v);
        vstore_half(v + ra * sn * sn, base + t, y);
    }
}

// fp16-input twin of conv1d_f32_t8x4: activations arrive as half (vload_half4
// = half the bytes of the dominant load stream); weights/accumulation/output
// remain fp32.
__kernel void conv1d_h16in_t8x4(__global const half* in,
                                __global const float* W,
                                __global float* out,
                                __global const float* bias, int has_bias,
                                int C_in, int C_out, int L_in, int L_out,
                                int K, int padding, int dilation) {
    int ocb = (int)get_global_id(0) * 8;
    int ol0 = (int)get_global_id(1) * 4;
    if (ocb >= C_out || ol0 >= L_out) return;
    float4 a0 = (float4)(0.0f), a1 = (float4)(0.0f), a2 = (float4)(0.0f), a3 = (float4)(0.0f);
    float4 a4 = (float4)(0.0f), a5 = (float4)(0.0f), a6 = (float4)(0.0f), a7 = (float4)(0.0f);
    int ws = mul24(C_in, K);
    int oc_last = C_out - 1 - ocb;
    int r1 = (1 <= oc_last) ? 1 : 0, r2 = (2 <= oc_last) ? 2 : 0, r3 = (3 <= oc_last) ? 3 : 0;
    int r4 = (4 <= oc_last) ? 4 : 0, r5 = (5 <= oc_last) ? 5 : 0, r6 = (6 <= oc_last) ? 6 : 0;
    int r7 = (7 <= oc_last) ? 7 : 0;
    __global const float* wp = W + mul24(ocb, ws);
    int o1 = mul24(r1, ws), o2 = mul24(r2, ws), o3 = mul24(r3, ws);
    int o4 = mul24(r4, ws), o5 = mul24(r5, ws), o6 = mul24(r6, ws), o7 = mul24(r7, ws);
    for (int k = 0; k < K; ++k) {
        int il = ol0 - padding + mul24(k, dilation);
        if (il >= 0 && il + 3 < L_in) {
            for (int ic = 0; ic < C_in; ++ic) {
                float4 v = vload_half4(0, in + mul24(ic, L_in) + il);
                int wi = mad24(ic, K, k);
                a0 = mad(wp[wi], v, a0);
                a1 = mad(wp[wi + o1], v, a1);
                a2 = mad(wp[wi + o2], v, a2);
                a3 = mad(wp[wi + o3], v, a3);
                a4 = mad(wp[wi + o4], v, a4);
                a5 = mad(wp[wi + o5], v, a5);
                a6 = mad(wp[wi + o6], v, a6);
                a7 = mad(wp[wi + o7], v, a7);
            }
        } else if (il + 3 >= 0 && il < L_in) {
            for (int ic = 0; ic < C_in; ++ic) {
                int rb = mul24(ic, L_in);
                float4 v;
                v.x = (il     >= 0 && il     < L_in) ? (float)vload_half(rb + il,     in) : 0.0f;
                v.y = (il + 1 >= 0 && il + 1 < L_in) ? (float)vload_half(rb + il + 1, in) : 0.0f;
                v.z = (il + 2 >= 0 && il + 2 < L_in) ? (float)vload_half(rb + il + 2, in) : 0.0f;
                v.w = (il + 3 >= 0 && il + 3 < L_in) ? (float)vload_half(rb + il + 3, in) : 0.0f;
                int wi = mad24(ic, K, k);
                a0 = mad(wp[wi], v, a0);
                a1 = mad(wp[wi + o1], v, a1);
                a2 = mad(wp[wi + o2], v, a2);
                a3 = mad(wp[wi + o3], v, a3);
                a4 = mad(wp[wi + o4], v, a4);
                a5 = mad(wp[wi + o5], v, a5);
                a6 = mad(wp[wi + o6], v, a6);
                a7 = mad(wp[wi + o7], v, a7);
            }
        }
    }
    if (has_bias) {
        a0 += bias[ocb];
        if (1 <= oc_last) a1 += bias[ocb + 1];
        if (2 <= oc_last) a2 += bias[ocb + 2];
        if (3 <= oc_last) a3 += bias[ocb + 3];
        if (4 <= oc_last) a4 += bias[ocb + 4];
        if (5 <= oc_last) a5 += bias[ocb + 5];
        if (6 <= oc_last) a6 += bias[ocb + 6];
        if (7 <= oc_last) a7 += bias[ocb + 7];
    }
    int rem = L_out - ol0;
    if (rem >= 4) {
        #define H8_ST4(J, A) if (J <= oc_last) vstore4(A, 0, out + mul24(ocb + J, L_out) + ol0);
        H8_ST4(0, a0) H8_ST4(1, a1) H8_ST4(2, a2) H8_ST4(3, a3)
        H8_ST4(4, a4) H8_ST4(5, a5) H8_ST4(6, a6) H8_ST4(7, a7)
        #undef H8_ST4
    } else {
        #define H8_TAIL(J, A) if (J <= oc_last) { __global float* o = out + mul24(ocb + J, L_out) + ol0; \
            o[0] = A.x; if (rem > 1) o[1] = A.y; if (rem > 2) o[2] = A.z; }
        H8_TAIL(0, a0) H8_TAIL(1, a1) H8_TAIL(2, a2) H8_TAIL(3, a3)
        H8_TAIL(4, a4) H8_TAIL(5, a5) H8_TAIL(6, a6) H8_TAIL(7, a7)
        #undef H8_TAIL
    }
}

// half-weight weight_norm reconstruction (norm computed in fp32).
__kernel void weightnorm_f16w(__global const half* v_fp16,
                              __global const half* g_fp16,
                              __global half* Wh,
                              int C_out, int per_oc) {
    int oc = get_group_id(0);
    int lt = get_local_id(0);
    if (oc >= C_out) return;
    int base = oc * per_oc;
    __local float lsq[64];
    float q = 0.0f;
    for (int i = lt; i < per_oc; i += 64) { float x = (float)vload_half(base + i, v_fp16); q += x * x; }
    lsq[lt] = q;
    barrier(CLK_LOCAL_MEM_FENCE);
    for (int off = 32; off > 0; off >>= 1) {
        if (lt < off) lsq[lt] += lsq[lt + off];
        barrier(CLK_LOCAL_MEM_FENCE);
    }
    float norm = sqrt(lsq[0]) + 1e-12f;
    float scale = (float)vload_half(oc, g_fp16) / norm;
    for (int i = lt; i < per_oc; i += 64) {
        vstore_half((float)vload_half(base + i, v_fp16) * scale, base + i, Wh);
    }
}

// All-half conv: half inputs, half weights, half4 accumulators (2x ALU rate on
// Adreno 16-bit lanes), half output. 8 oc x 4 ol register tile. The instnorm
// that follows is scale-invariant and computes its stats in fp32, which
// absorbs most of the accumulation noise; the residual spine stays fp32.
__kernel void conv1d_hh_t8x4(__global const half* in,
                             __global const half* W,
                             __global half* out,
                             __global const half* bias, int has_bias,
                             int C_in, int C_out, int L_in, int L_out,
                             int K, int padding, int dilation) {
    int ocb = (int)get_global_id(0) * 8;
    int ol0 = (int)get_global_id(1) * 4;
    if (ocb >= C_out || ol0 >= L_out) return;
    half4 a0 = (half4)(0.0h), a1 = (half4)(0.0h), a2 = (half4)(0.0h), a3 = (half4)(0.0h);
    half4 a4 = (half4)(0.0h), a5 = (half4)(0.0h), a6 = (half4)(0.0h), a7 = (half4)(0.0h);
    int ws = mul24(C_in, K);
    int oc_last = C_out - 1 - ocb;
    int r1 = (1 <= oc_last) ? 1 : 0, r2 = (2 <= oc_last) ? 2 : 0, r3 = (3 <= oc_last) ? 3 : 0;
    int r4 = (4 <= oc_last) ? 4 : 0, r5 = (5 <= oc_last) ? 5 : 0, r6 = (6 <= oc_last) ? 6 : 0;
    int r7 = (7 <= oc_last) ? 7 : 0;
    __global const half* wp = W + mul24(ocb, ws);
    int o1 = mul24(r1, ws), o2 = mul24(r2, ws), o3 = mul24(r3, ws);
    int o4 = mul24(r4, ws), o5 = mul24(r5, ws), o6 = mul24(r6, ws), o7 = mul24(r7, ws);
    for (int k = 0; k < K; ++k) {
        int il = ol0 - padding + mul24(k, dilation);
        if (il >= 0 && il + 3 < L_in) {
            for (int ic = 0; ic < C_in; ++ic) {
                half4 v = vload4(0, in + mul24(ic, L_in) + il);
                int wi = mad24(ic, K, k);
                a0 += wp[wi] * v;
                a1 += wp[wi + o1] * v;
                a2 += wp[wi + o2] * v;
                a3 += wp[wi + o3] * v;
                a4 += wp[wi + o4] * v;
                a5 += wp[wi + o5] * v;
                a6 += wp[wi + o6] * v;
                a7 += wp[wi + o7] * v;
            }
        } else if (il + 3 >= 0 && il < L_in) {
            for (int ic = 0; ic < C_in; ++ic) {
                int rb = mul24(ic, L_in);
                half4 v;
                v.x = (il     >= 0 && il     < L_in) ? in[rb + il]     : (half)0.0h;
                v.y = (il + 1 >= 0 && il + 1 < L_in) ? in[rb + il + 1] : (half)0.0h;
                v.z = (il + 2 >= 0 && il + 2 < L_in) ? in[rb + il + 2] : (half)0.0h;
                v.w = (il + 3 >= 0 && il + 3 < L_in) ? in[rb + il + 3] : (half)0.0h;
                int wi = mad24(ic, K, k);
                a0 += wp[wi] * v;
                a1 += wp[wi + o1] * v;
                a2 += wp[wi + o2] * v;
                a3 += wp[wi + o3] * v;
                a4 += wp[wi + o4] * v;
                a5 += wp[wi + o5] * v;
                a6 += wp[wi + o6] * v;
                a7 += wp[wi + o7] * v;
            }
        }
    }
    if (has_bias) {
        a0 += bias[ocb];
        if (1 <= oc_last) a1 += bias[ocb + 1];
        if (2 <= oc_last) a2 += bias[ocb + 2];
        if (3 <= oc_last) a3 += bias[ocb + 3];
        if (4 <= oc_last) a4 += bias[ocb + 4];
        if (5 <= oc_last) a5 += bias[ocb + 5];
        if (6 <= oc_last) a6 += bias[ocb + 6];
        if (7 <= oc_last) a7 += bias[ocb + 7];
    }
    int rem = L_out - ol0;
    if (rem >= 4) {
        #define HH_ST4(J, A) if (J <= oc_last) vstore4(A, 0, out + mul24(ocb + J, L_out) + ol0);
        HH_ST4(0, a0) HH_ST4(1, a1) HH_ST4(2, a2) HH_ST4(3, a3)
        HH_ST4(4, a4) HH_ST4(5, a5) HH_ST4(6, a6) HH_ST4(7, a7)
        #undef HH_ST4
    } else {
        #define HH_TAIL(J, A) if (J <= oc_last) { __global half* o = out + mul24(ocb + J, L_out) + ol0; \
            o[0] = A.x; if (rem > 1) o[1] = A.y; if (rem > 2) o[2] = A.z; }
        HH_TAIL(0, a0) HH_TAIL(1, a1) HH_TAIL(2, a2) HH_TAIL(3, a3)
        HH_TAIL(4, a4) HH_TAIL(5, a5) HH_TAIL(6, a6) HH_TAIL(7, a7)
        #undef HH_TAIL
    }
}

// Fused instnorm+adain+snake reading HALF input (conv output), writing half.
// Stats still reduced in fp32.
__kernel void instnorm_adain_snake_h2h(__global const half* x,
                                       __global half* y,
                                       __global const float* gamma,
                                       __global const float* beta,
                                       __global const half* alpha,
                                       int C, int T, float eps) {
    int c  = get_group_id(0);
    int lt = get_local_id(0);
    if (c >= C) return;
    int base = c * T;
    __local float lsum[64];
    __local float lsq[64];
    // 4-wide half loads — scalar vload_half here measured 16x slower.
    int T4 = T & ~3;
    float4 s4 = (float4)(0.0f), q4 = (float4)(0.0f);
    for (int t = lt * 4; t < T4; t += 256) {
        float4 v = convert_float4(vload4(0, x + base + t));
        s4 += v; q4 += v * v;
    }
    float s = s4.x + s4.y + s4.z + s4.w;
    float q = q4.x + q4.y + q4.z + q4.w;
    if (lt == 0) {
        for (int t = T4; t < T; ++t) { float v = (float)vload_half(base + t, x); s += v; q += v * v; }
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
    float g = gamma[c], b = beta[c];
    float a = (float)vload_half(c, alpha);
    if (fabs(a) < 1e-6f) a = (a >= 0.0f ? 1e-6f : -1e-6f);
    float ra = 1.0f / a;
    float4 g4 = (float4)(1.0f + g);
    for (int t = lt * 4; t < T4; t += 256) {
        float4 v = (convert_float4(vload4(0, x + base + t)) - mean) * inv;
        v = g4 * v + b;
        float4 sn = native_sin(a * v);
        vstore4(convert_half4(v + ra * sn * sn), 0, y + base + t);
    }
    if (lt == 0) {
        for (int t = T4; t < T; ++t) {
            float v = ((float)vload_half(base + t, x) - mean) * inv;
            v = (1.0f + g) * v + b;
            float sn = native_sin(a * v);
            vstore_half(v + ra * sn * sn, base + t, y);
        }
    }
}

// LDS-weight variant: 4 oc rows cached in local memory (cooperative load,
// one barrier), killing the per-iteration L2 latency on weight reads. Used
// when 4*C_in*K fits the 8192-half cache. 4(oc) x 4(ol) tile, half math.
__kernel void conv1d_hh_lds4x4(__global const half* in,
                               __global const half* W,
                               __global half* out,
                               __global const half* bias, int has_bias,
                               int C_in, int C_out, int L_in, int L_out,
                               int K, int padding, int dilation) {
    int ocb = (int)get_group_id(0) * 4;
    int lt  = (int)get_local_id(1);
    int ol0 = ((int)get_group_id(1) * 128 + lt) * 4;
    int ws = mul24(C_in, K);
    int wtot = ws << 2;
    __local half Wl[8192];
    int wbase = mul24(ocb, ws);
    for (int i = lt; i < wtot; i += 128) Wl[i] = W[wbase + i];
    barrier(CLK_LOCAL_MEM_FENCE);
    if (ocb >= C_out || ol0 >= L_out) return;
    half4 a0 = (half4)(0.0h), a1 = (half4)(0.0h), a2 = (half4)(0.0h), a3 = (half4)(0.0h);
    int ws2 = ws * 2, ws3 = ws * 3;
    for (int k = 0; k < K; ++k) {
        int il = ol0 - padding + mul24(k, dilation);
        if (il >= 0 && il + 3 < L_in) {
            for (int ic = 0; ic < C_in; ++ic) {
                half4 v = vload4(0, in + mul24(ic, L_in) + il);
                int wi = mad24(ic, K, k);
                a0 += Wl[wi] * v;
                a1 += Wl[wi + ws] * v;
                a2 += Wl[wi + ws2] * v;
                a3 += Wl[wi + ws3] * v;
            }
        } else if (il + 3 >= 0 && il < L_in) {
            for (int ic = 0; ic < C_in; ++ic) {
                int rb = mul24(ic, L_in);
                half4 v;
                v.x = (il     >= 0 && il     < L_in) ? in[rb + il]     : (half)0.0h;
                v.y = (il + 1 >= 0 && il + 1 < L_in) ? in[rb + il + 1] : (half)0.0h;
                v.z = (il + 2 >= 0 && il + 2 < L_in) ? in[rb + il + 2] : (half)0.0h;
                v.w = (il + 3 >= 0 && il + 3 < L_in) ? in[rb + il + 3] : (half)0.0h;
                int wi = mad24(ic, K, k);
                a0 += Wl[wi] * v;
                a1 += Wl[wi + ws] * v;
                a2 += Wl[wi + ws2] * v;
                a3 += Wl[wi + ws3] * v;
            }
        }
    }
    if (has_bias) {
        a0 += bias[ocb]; a1 += bias[ocb + 1]; a2 += bias[ocb + 2]; a3 += bias[ocb + 3];
    }
    int rem = L_out - ol0;
    if (rem >= 4) {
        vstore4(a0, 0, out + mul24(ocb + 0, L_out) + ol0);
        vstore4(a1, 0, out + mul24(ocb + 1, L_out) + ol0);
        vstore4(a2, 0, out + mul24(ocb + 2, L_out) + ol0);
        vstore4(a3, 0, out + mul24(ocb + 3, L_out) + ol0);
    } else {
        #define HL_TAIL(J, A) { __global half* o = out + mul24(ocb + J, L_out) + ol0; \
            o[0] = A.x; if (rem > 1) o[1] = A.y; if (rem > 2) o[2] = A.z; }
        HL_TAIL(0, a0) HL_TAIL(1, a1) HL_TAIL(2, a2) HL_TAIL(3, a3)
        #undef HL_TAIL
    }
}

// fp32 iSTFT (overlap-add, hann window, window^2 normalization, irfft via
// direct cosine sum — n_fft=20 so each output sample sums 4 frames x 11 bins).
// Replaces the host fp64 loop + two blocking readbacks of mag/phase.
__kernel void istft_f32(__global const half* mag,     // [n_freq, L_frames]
                        __global const half* phase,   // [n_freq, L_frames]
                        __global float* audio,        // [T_audio]
                        int L_frames, int T_audio,
                        int n_fft, int hop, int n_freq) {
    int i = get_global_id(0);
    if (i >= T_audio) return;
    float acc = 0.0f;
    float norm = 0.0f;
    float inv_nfft = 1.0f / (float)n_fft;
    int f_lo = (i - (n_fft - 1) + hop - 1) / hop;  // ceil((i-(n_fft-1))/hop)
    if (f_lo < 0) f_lo = 0;
    int f_hi = i / hop;
    if (f_hi > L_frames - 1) f_hi = L_frames - 1;
    for (int f = f_lo; f <= f_hi; ++f) {
        int t = i - mul24(f, hop);            // 0..n_fft-1
        float w = 0.5f - 0.5f * native_cos(2.0f * 3.14159265358979f * (float)t * inv_nfft);
        float ph0 = (float)vload_half(f, phase);                       // bin 0 (row stride L_frames)
        float sample = (float)vload_half(f, mag) * native_cos(ph0);
        for (int k = 1; k < n_freq - 1; ++k) {
            int idx = mad24(k, L_frames, f);
            float ang = 2.0f * 3.14159265358979f * (float)k * (float)t * inv_nfft
                        + (float)vload_half(idx, phase);
            sample += 2.0f * (float)vload_half(idx, mag) * native_cos(ang);
        }
        // Nyquist bin (n_fft even)
        int idn = mad24(n_freq - 1, L_frames, f);
        float angn = 3.14159265358979f * (float)t + (float)vload_half(idn, phase);
        sample += (float)vload_half(idn, mag) * native_cos(angn);
        sample *= inv_nfft;
        acc += sample * w;
        norm += w * w;
    }
    audio[i] = (norm > 1e-10f) ? acc / norm : 0.0f;
}

// 8(oc) x 8(ol) texture conv — same TP/L1 weight path, but each texel read
// amortizes over 8 output positions (half8 input vectors).
__kernel void conv1d_ht_t8x8(__global const half* in,
                             __read_only image2d_t Wimg,
                             __global half* out,
                             __global const half* bias, int has_bias,
                             int C_in, int C_out, int L_in, int L_out,
                             int K_pad, int padding, int dilation) {
    const sampler_t smp = CLK_NORMALIZED_COORDS_FALSE | CLK_ADDRESS_CLAMP_TO_EDGE | CLK_FILTER_NEAREST;
    int ocb = (int)get_global_id(0) * 8;
    int ol0 = (int)get_global_id(1) * 8;
    if (ocb >= C_out || ol0 >= L_out) return;
    half8 a0 = (half8)(0.0h), a1 = (half8)(0.0h), a2 = (half8)(0.0h), a3 = (half8)(0.0h);
    half8 a4 = (half8)(0.0h), a5 = (half8)(0.0h), a6 = (half8)(0.0h), a7 = (half8)(0.0h);
    int Kp4 = K_pad >> 2;
    int il_base = ol0 - padding;
    int span = mul24(K_pad - 1, dilation);
    int interior = (il_base >= 0 && il_base + span + 7 < L_in) ? 1 : 0;
    for (int ic = 0; ic < C_in; ++ic) {
        int rb = mul24(ic, L_in);
        int txb = mul24(ic, Kp4);
        for (int k4 = 0; k4 < Kp4; ++k4) {
            int kb = k4 << 2;
            half8 v0, v1, v2, v3;
            if (interior) {
                int il = il_base + mul24(kb, dilation);
                v0 = vload8(0, in + rb + il);
                v1 = vload8(0, in + rb + il + dilation);
                v2 = vload8(0, in + rb + il + 2 * dilation);
                v3 = vload8(0, in + rb + il + 3 * dilation);
            } else {
                #define HT8_GV(VV, KOFF) { int il = il_base + mul24(kb + KOFF, dilation); \
                    VV.s0 = (il     >= 0 && il     < L_in) ? in[rb + il]     : (half)0.0h; \
                    VV.s1 = (il + 1 >= 0 && il + 1 < L_in) ? in[rb + il + 1] : (half)0.0h; \
                    VV.s2 = (il + 2 >= 0 && il + 2 < L_in) ? in[rb + il + 2] : (half)0.0h; \
                    VV.s3 = (il + 3 >= 0 && il + 3 < L_in) ? in[rb + il + 3] : (half)0.0h; \
                    VV.s4 = (il + 4 >= 0 && il + 4 < L_in) ? in[rb + il + 4] : (half)0.0h; \
                    VV.s5 = (il + 5 >= 0 && il + 5 < L_in) ? in[rb + il + 5] : (half)0.0h; \
                    VV.s6 = (il + 6 >= 0 && il + 6 < L_in) ? in[rb + il + 6] : (half)0.0h; \
                    VV.s7 = (il + 7 >= 0 && il + 7 < L_in) ? in[rb + il + 7] : (half)0.0h; }
                HT8_GV(v0, 0) HT8_GV(v1, 1) HT8_GV(v2, 2) HT8_GV(v3, 3)
                #undef HT8_GV
            }
            int tx = txb + k4;
            half4 w;
            #define HT8_ROW(A, J) { w = read_imageh(Wimg, smp, (int2)(tx, ocb + J)); \
                A += w.x * v0 + w.y * v1 + w.z * v2 + w.w * v3; }
            HT8_ROW(a0, 0) HT8_ROW(a1, 1) HT8_ROW(a2, 2) HT8_ROW(a3, 3)
            HT8_ROW(a4, 4) HT8_ROW(a5, 5) HT8_ROW(a6, 6) HT8_ROW(a7, 7)
            #undef HT8_ROW
        }
    }
    int oc_last = C_out - 1 - ocb;
    if (has_bias) {
        a0 += bias[ocb];
        if (1 <= oc_last) a1 += bias[ocb + 1];
        if (2 <= oc_last) a2 += bias[ocb + 2];
        if (3 <= oc_last) a3 += bias[ocb + 3];
        if (4 <= oc_last) a4 += bias[ocb + 4];
        if (5 <= oc_last) a5 += bias[ocb + 5];
        if (6 <= oc_last) a6 += bias[ocb + 6];
        if (7 <= oc_last) a7 += bias[ocb + 7];
    }
    int rem = L_out - ol0;
    if (rem >= 8) {
        #define HT8_ST(J, A) if (J <= oc_last) vstore8(A, 0, out + mul24(ocb + J, L_out) + ol0);
        HT8_ST(0, a0) HT8_ST(1, a1) HT8_ST(2, a2) HT8_ST(3, a3)
        HT8_ST(4, a4) HT8_ST(5, a5) HT8_ST(6, a6) HT8_ST(7, a7)
        #undef HT8_ST
    } else {
        #define HT8_TAIL(J, A) if (J <= oc_last) { __global half* o = out + mul24(ocb + J, L_out) + ol0; \
            o[0] = A.s0; if (rem > 1) o[1] = A.s1; if (rem > 2) o[2] = A.s2; if (rem > 3) o[3] = A.s3; \
            if (rem > 4) o[4] = A.s4; if (rem > 5) o[5] = A.s5; if (rem > 6) o[6] = A.s6; }
        HT8_TAIL(0, a0) HT8_TAIL(1, a1) HT8_TAIL(2, a2) HT8_TAIL(3, a3)
        HT8_TAIL(4, a4) HT8_TAIL(5, a5) HT8_TAIL(6, a6) HT8_TAIL(7, a7)
        #undef HT8_TAIL
    }
}

// 4(oc) x 8(ol) texture conv: same accumulator bytes as 8x4 (4 x half8),
// but each weight texel serves 8 outputs -> half the TP reads per MAC.
__kernel void conv1d_ht_t4x8(__global const half* in,
                             __read_only image2d_t Wimg,
                             __global half* out,
                             __global const half* bias, int has_bias,
                             int C_in, int C_out, int L_in, int L_out,
                             int K_pad, int padding, int dilation) {
    const sampler_t smp = CLK_NORMALIZED_COORDS_FALSE | CLK_ADDRESS_CLAMP_TO_EDGE | CLK_FILTER_NEAREST;
    int ocb = (int)get_global_id(0) * 4;
    int ol0 = (int)get_global_id(1) * 8;
    if (ocb >= C_out || ol0 >= L_out) return;
    half8 a0 = (half8)(0.0h), a1 = (half8)(0.0h), a2 = (half8)(0.0h), a3 = (half8)(0.0h);
    int Kp4 = K_pad >> 2;
    int il_base = ol0 - padding;
    int span = mul24(K_pad - 1, dilation);
    int interior = (il_base >= 0 && il_base + span + 7 < L_in) ? 1 : 0;
    for (int ic = 0; ic < C_in; ++ic) {
        int rb = mul24(ic, L_in);
        int txb = mul24(ic, Kp4);
        for (int k4 = 0; k4 < Kp4; ++k4) {
            int kb = k4 << 2;
            half8 v0, v1, v2, v3;
            if (interior) {
                int il = il_base + mul24(kb, dilation);
                v0 = vload8(0, in + rb + il);
                v1 = vload8(0, in + rb + il + dilation);
                v2 = vload8(0, in + rb + il + 2 * dilation);
                v3 = vload8(0, in + rb + il + 3 * dilation);
            } else {
                #define T48_GV(VV, KOFF) { int il = il_base + mul24(kb + KOFF, dilation); \
                    VV.s0 = (il     >= 0 && il     < L_in) ? in[rb + il]     : (half)0.0h; \
                    VV.s1 = (il + 1 >= 0 && il + 1 < L_in) ? in[rb + il + 1] : (half)0.0h; \
                    VV.s2 = (il + 2 >= 0 && il + 2 < L_in) ? in[rb + il + 2] : (half)0.0h; \
                    VV.s3 = (il + 3 >= 0 && il + 3 < L_in) ? in[rb + il + 3] : (half)0.0h; \
                    VV.s4 = (il + 4 >= 0 && il + 4 < L_in) ? in[rb + il + 4] : (half)0.0h; \
                    VV.s5 = (il + 5 >= 0 && il + 5 < L_in) ? in[rb + il + 5] : (half)0.0h; \
                    VV.s6 = (il + 6 >= 0 && il + 6 < L_in) ? in[rb + il + 6] : (half)0.0h; \
                    VV.s7 = (il + 7 >= 0 && il + 7 < L_in) ? in[rb + il + 7] : (half)0.0h; }
                T48_GV(v0, 0) T48_GV(v1, 1) T48_GV(v2, 2) T48_GV(v3, 3)
                #undef T48_GV
            }
            int tx = txb + k4;
            half4 w;
            #define T48_ROW(A, J) { w = read_imageh(Wimg, smp, (int2)(tx, ocb + J)); \
                A += w.x * v0 + w.y * v1 + w.z * v2 + w.w * v3; }
            T48_ROW(a0, 0) T48_ROW(a1, 1) T48_ROW(a2, 2) T48_ROW(a3, 3)
            #undef T48_ROW
        }
    }
    int oc_last = C_out - 1 - ocb;
    if (has_bias) {
        a0 += bias[ocb];
        if (1 <= oc_last) a1 += bias[ocb + 1];
        if (2 <= oc_last) a2 += bias[ocb + 2];
        if (3 <= oc_last) a3 += bias[ocb + 3];
    }
    int rem = L_out - ol0;
    if (rem >= 8) {
        #define T48_ST(J, A) if (J <= oc_last) vstore8(A, 0, out + mul24(ocb + J, L_out) + ol0);
        T48_ST(0, a0) T48_ST(1, a1) T48_ST(2, a2) T48_ST(3, a3)
        #undef T48_ST
    } else {
        #define T48_TAIL(J, A) if (J <= oc_last) { __global half* o = out + mul24(ocb + J, L_out) + ol0; \
            o[0] = A.s0; if (rem > 1) o[1] = A.s1; if (rem > 2) o[2] = A.s2; if (rem > 3) o[3] = A.s3; \
            if (rem > 4) o[4] = A.s4; if (rem > 5) o[5] = A.s5; if (rem > 6) o[6] = A.s6; }
        T48_TAIL(0, a0) T48_TAIL(1, a1) T48_TAIL(2, a2) T48_TAIL(3, a3)
        #undef T48_TAIL
    }
}

// Residual add: x_f32 += half buffer.
__kernel void add_f32_from_h(__global float* x, __global const half* y, int N) {
    int i = get_global_id(0); if (i >= N) return;
    x[i] += (float)vload_half(i, y);
}

// Conv1d dilated with float4 inner loop + local memory caching of weight row.
// Each work-group: one (oc, tile of L_out). All threads share the weight row for oc.
// Max K*C_in we use: 11*256 = 2816 (level-0 resblocks at K=11).
#define LOCAL_T 32
#define WCACHE_MAX 2816
__kernel void conv1d_f32(__global const float* in,
                         __global const float* W,
                         __global float* out,
                         int C_in, int C_out, int L_in, int L_out,
                         int K, int stride, int padding, int dilation, int groups) {
    int oc = get_global_id(0);
    int lt = get_local_id(1);
    int ol = get_group_id(1) * LOCAL_T + lt;
    int grp_in_per_out = C_in / groups;
    int g = oc / (C_out / groups);
    int w_base = oc * grp_in_per_out * K;
    int w_total = grp_in_per_out * K;
    __local float W_cache[WCACHE_MAX];
    // Cooperative load
    for (int i = lt; i < w_total; i += LOCAL_T) {
        W_cache[i] = W[w_base + i];
    }
    barrier(CLK_LOCAL_MEM_FENCE);
    if (oc >= C_out || ol >= L_out) return;
    float acc = 0.0f;
    for (int k = 0; k < K; ++k) {
        int il = ol * stride - padding + k * dilation;
        if (il < 0 || il >= L_in) continue;
        int ic = 0;
        for (; ic + 3 < grp_in_per_out; ic += 4) {
            int ic_g = g * grp_in_per_out + ic;
            float4 inv = (float4)(in[(ic_g+0)*L_in + il], in[(ic_g+1)*L_in + il],
                                  in[(ic_g+2)*L_in + il], in[(ic_g+3)*L_in + il]);
            float4 wv = (float4)(W_cache[(ic+0)*K + k], W_cache[(ic+1)*K + k],
                                 W_cache[(ic+2)*K + k], W_cache[(ic+3)*K + k]);
            acc += dot(inv, wv);
        }
        for (; ic < grp_in_per_out; ++ic) {
            int ic_g = g * grp_in_per_out + ic;
            acc += in[ic_g*L_in + il] * W_cache[ic*K + k];
        }
    }
    out[oc*L_out + ol] = acc;
}

// Register-tiled conv1d fast path for the generator resblock convs
// (stride==1, groups==1, C_out % 4 == 0). Each work-item computes a
// 4(oc) x 4(ol) output tile:
//   - input is loaded as one float4 per (ic,k) via vload4 (contiguous along L
//     in the channel-major [C,L] layout) and reused across all 4 oc rows;
//   - the 4 weight scalars per (ic,k) are identical across every work-item in
//     the workgroup (lws dim0 == 1), so they coalesce into broadcast reads;
//   - no local memory, no barrier (streaming mode), ~30 registers/WI.
// Boundary taps (il partially outside [0,L_in)) take a guarded scalar path;
// interior taps — the overwhelming majority — run unguarded.
__kernel void conv1d_f32_t4x4(__global const float* in,
                              __global const float* W,
                              __global float* out,
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
    __global const float* w0 = W + mul24(ocb + 0, ws);
    __global const float* w1 = W + mul24(ocb + 1, ws);
    __global const float* w2 = W + mul24(ocb + 2, ws);
    __global const float* w3 = W + mul24(ocb + 3, ws);
    for (int k = 0; k < K; ++k) {
        int il = ol0 - padding + mul24(k, dilation);
        if (il >= 0 && il + 3 < L_in) {
            // interior fast path: unguarded vector loads
            for (int ic = 0; ic < C_in; ++ic) {
                float4 v = vload4(0, in + mul24(ic, L_in) + il);
                int wi = mad24(ic, K, k);
                acc0 = mad(w0[wi], v, acc0);
                acc1 = mad(w1[wi], v, acc1);
                acc2 = mad(w2[wi], v, acc2);
                acc3 = mad(w3[wi], v, acc3);
            }
        } else if (il + 3 >= 0 && il < L_in) {
            // boundary tap: guarded per-element loads (zero outside)
            for (int ic = 0; ic < C_in; ++ic) {
                __global const float* row = in + mul24(ic, L_in);
                float4 v;
                v.x = (il     >= 0 && il     < L_in) ? row[il]     : 0.0f;
                v.y = (il + 1 >= 0 && il + 1 < L_in) ? row[il + 1] : 0.0f;
                v.z = (il + 2 >= 0 && il + 2 < L_in) ? row[il + 2] : 0.0f;
                v.w = (il + 3 >= 0 && il + 3 < L_in) ? row[il + 3] : 0.0f;
                int wi = mad24(ic, K, k);
                acc0 = mad(w0[wi], v, acc0);
                acc1 = mad(w1[wi], v, acc1);
                acc2 = mad(w2[wi], v, acc2);
                acc3 = mad(w3[wi], v, acc3);
            }
        }
        // else: tap fully outside input — contributes zero
    }
    int rem = L_out - ol0;
    if (rem >= 4) {
        vstore4(acc0, 0, out + mul24(ocb + 0, L_out) + ol0);
        vstore4(acc1, 0, out + mul24(ocb + 1, L_out) + ol0);
        vstore4(acc2, 0, out + mul24(ocb + 2, L_out) + ol0);
        vstore4(acc3, 0, out + mul24(ocb + 3, L_out) + ol0);
    } else {
        __global float* o0 = out + mul24(ocb + 0, L_out) + ol0;
        __global float* o1 = out + mul24(ocb + 1, L_out) + ol0;
        __global float* o2 = out + mul24(ocb + 2, L_out) + ol0;
        __global float* o3 = out + mul24(ocb + 3, L_out) + ol0;
        o0[0] = acc0.x; o1[0] = acc1.x; o2[0] = acc2.x; o3[0] = acc3.x;
        if (rem > 1) { o0[1] = acc0.y; o1[1] = acc1.y; o2[1] = acc2.y; o3[1] = acc3.y; }
        if (rem > 2) { o0[2] = acc0.z; o1[2] = acc1.z; o2[2] = acc2.z; o3[2] = acc3.z; }
    }
}


// 8(oc) x 4(ol) variant — halves input traffic vs t4x4 by reusing each input
// vector across 8 output channels. ~45 regs/WI; if the compiler spills, t4x4
// wins — keep both and pick by measurement (NNOPT_T8=0 disables).
__kernel void conv1d_f32_t8x4(__global const float* in,
                              __global const float* W,
                              __global float* out,
                              __global const float* bias, int has_bias,
                              int C_in, int C_out, int L_in, int L_out,
                              int K, int padding, int dilation) {
    int ocb = (int)get_global_id(0) * 8;
    int ol0 = (int)get_global_id(1) * 4;
    if (ocb >= C_out || ol0 >= L_out) return;
    float4 a0 = (float4)(0.0f), a1 = (float4)(0.0f), a2 = (float4)(0.0f), a3 = (float4)(0.0f);
    float4 a4 = (float4)(0.0f), a5 = (float4)(0.0f), a6 = (float4)(0.0f), a7 = (float4)(0.0f);
    int ws = mul24(C_in, K);
    // oc-tail masking: clamp absent rows onto row 0 (their results are
    // discarded at store), so any C_out works — e.g. conv_post's 22 rows.
    int oc_last = C_out - 1 - ocb;   // >= 7 for interior tiles
    int r1 = (1 <= oc_last) ? 1 : 0, r2 = (2 <= oc_last) ? 2 : 0, r3 = (3 <= oc_last) ? 3 : 0;
    int r4 = (4 <= oc_last) ? 4 : 0, r5 = (5 <= oc_last) ? 5 : 0, r6 = (6 <= oc_last) ? 6 : 0;
    int r7 = (7 <= oc_last) ? 7 : 0;
    __global const float* wp = W + mul24(ocb, ws);
    int o1 = mul24(r1, ws), o2 = mul24(r2, ws), o3 = mul24(r3, ws);
    int o4 = mul24(r4, ws), o5 = mul24(r5, ws), o6 = mul24(r6, ws), o7 = mul24(r7, ws);
    for (int k = 0; k < K; ++k) {
        int il = ol0 - padding + mul24(k, dilation);
        if (il >= 0 && il + 3 < L_in) {
            for (int ic = 0; ic < C_in; ++ic) {
                float4 v = vload4(0, in + mul24(ic, L_in) + il);
                int wi = mad24(ic, K, k);
                a0 = mad(wp[wi], v, a0);
                a1 = mad(wp[wi + o1], v, a1);
                a2 = mad(wp[wi + o2], v, a2);
                a3 = mad(wp[wi + o3], v, a3);
                a4 = mad(wp[wi + o4], v, a4);
                a5 = mad(wp[wi + o5], v, a5);
                a6 = mad(wp[wi + o6], v, a6);
                a7 = mad(wp[wi + o7], v, a7);
            }
        } else if (il + 3 >= 0 && il < L_in) {
            for (int ic = 0; ic < C_in; ++ic) {
                __global const float* row = in + mul24(ic, L_in);
                float4 v;
                v.x = (il     >= 0 && il     < L_in) ? row[il]     : 0.0f;
                v.y = (il + 1 >= 0 && il + 1 < L_in) ? row[il + 1] : 0.0f;
                v.z = (il + 2 >= 0 && il + 2 < L_in) ? row[il + 2] : 0.0f;
                v.w = (il + 3 >= 0 && il + 3 < L_in) ? row[il + 3] : 0.0f;
                int wi = mad24(ic, K, k);
                a0 = mad(wp[wi], v, a0);
                a1 = mad(wp[wi + o1], v, a1);
                a2 = mad(wp[wi + o2], v, a2);
                a3 = mad(wp[wi + o3], v, a3);
                a4 = mad(wp[wi + o4], v, a4);
                a5 = mad(wp[wi + o5], v, a5);
                a6 = mad(wp[wi + o6], v, a6);
                a7 = mad(wp[wi + o7], v, a7);
            }
        }
    }
    if (has_bias) {
        a0 += bias[ocb];
        if (1 <= oc_last) a1 += bias[ocb + 1];
        if (2 <= oc_last) a2 += bias[ocb + 2];
        if (3 <= oc_last) a3 += bias[ocb + 3];
        if (4 <= oc_last) a4 += bias[ocb + 4];
        if (5 <= oc_last) a5 += bias[ocb + 5];
        if (6 <= oc_last) a6 += bias[ocb + 6];
        if (7 <= oc_last) a7 += bias[ocb + 7];
    }
    int rem = L_out - ol0;
    if (rem >= 4) {
        #define T8_ST4(J, A) if (J <= oc_last) vstore4(A, 0, out + mul24(ocb + J, L_out) + ol0);
        T8_ST4(0, a0) T8_ST4(1, a1) T8_ST4(2, a2) T8_ST4(3, a3)
        T8_ST4(4, a4) T8_ST4(5, a5) T8_ST4(6, a6) T8_ST4(7, a7)
        #undef T8_ST4
    } else {
        // No private array here: a dynamically indexed local array forces the
        // compiler to spill the accumulators to scratch for EVERY work-item.
        #define T8_TAIL(J, A) if (J <= oc_last) { __global float* o = out + mul24(ocb + J, L_out) + ol0; \
            o[0] = A.x; if (rem > 1) o[1] = A.y; if (rem > 2) o[2] = A.z; }
        T8_TAIL(0, a0) T8_TAIL(1, a1) T8_TAIL(2, a2) T8_TAIL(3, a3)
        T8_TAIL(4, a4) T8_TAIL(5, a5) T8_TAIL(6, a6) T8_TAIL(7, a7)
        #undef T8_TAIL
    }
}

// k-vectorized 8(oc) x 4(ol) conv on K-padded weights: each weight read is a
// float4 of 4 consecutive taps (vs 4 scalar reads), and the 4 input vectors
// for those taps are shared across all 8 output channels.
__kernel void conv1d_f32_t8x4v(__global const float* in,
                               __global const float* Wp,
                               __global float* out,
                               __global const float* bias, int has_bias,
                               int C_in, int C_out, int L_in, int L_out,
                               int K_pad, int padding, int dilation) {
    int ocb = (int)get_global_id(0) * 8;
    int ol0 = (int)get_global_id(1) * 4;
    if (ocb >= C_out || ol0 >= L_out) return;
    float4 a0 = (float4)(0.0f), a1 = (float4)(0.0f), a2 = (float4)(0.0f), a3 = (float4)(0.0f);
    float4 a4 = (float4)(0.0f), a5 = (float4)(0.0f), a6 = (float4)(0.0f), a7 = (float4)(0.0f);
    int ws = mul24(C_in, K_pad);
    int wb = mul24(ocb, ws);
    int Kp4 = K_pad >> 2;
    int il_base = ol0 - padding;
    int span = mul24(K_pad - 1, dilation);
    int interior = (il_base >= 0 && il_base + span + 3 < L_in) ? 1 : 0;
    for (int ic = 0; ic < C_in; ++ic) {
        int rb = mul24(ic, L_in);
        int wrow = wb + mul24(ic, K_pad);
        for (int k4 = 0; k4 < Kp4; ++k4) {
            int kb = k4 << 2;
            float4 v0, v1, v2, v3;
            if (interior) {
                int il = il_base + mul24(kb, dilation);
                v0 = vload4(0, in + rb + il);
                v1 = vload4(0, in + rb + il + dilation);
                v2 = vload4(0, in + rb + il + 2 * dilation);
                v3 = vload4(0, in + rb + il + 3 * dilation);
            } else {
                #define GV(VV, KOFF) { int il = il_base + mul24(kb + KOFF, dilation); \
                    VV.x = (il     >= 0 && il     < L_in) ? in[rb + il]     : 0.0f; \
                    VV.y = (il + 1 >= 0 && il + 1 < L_in) ? in[rb + il + 1] : 0.0f; \
                    VV.z = (il + 2 >= 0 && il + 2 < L_in) ? in[rb + il + 2] : 0.0f; \
                    VV.w = (il + 3 >= 0 && il + 3 < L_in) ? in[rb + il + 3] : 0.0f; }
                GV(v0, 0) GV(v1, 1) GV(v2, 2) GV(v3, 3)
                #undef GV
            }
            int wi = wrow + kb;
            float4 w;
            w = vload4(0, Wp + wi);
            a0 = mad(w.x, v0, mad(w.y, v1, mad(w.z, v2, mad(w.w, v3, a0))));
            w = vload4(0, Wp + wi + ws);
            a1 = mad(w.x, v0, mad(w.y, v1, mad(w.z, v2, mad(w.w, v3, a1))));
            w = vload4(0, Wp + wi + 2 * ws);
            a2 = mad(w.x, v0, mad(w.y, v1, mad(w.z, v2, mad(w.w, v3, a2))));
            w = vload4(0, Wp + wi + 3 * ws);
            a3 = mad(w.x, v0, mad(w.y, v1, mad(w.z, v2, mad(w.w, v3, a3))));
            w = vload4(0, Wp + wi + 4 * ws);
            a4 = mad(w.x, v0, mad(w.y, v1, mad(w.z, v2, mad(w.w, v3, a4))));
            w = vload4(0, Wp + wi + 5 * ws);
            a5 = mad(w.x, v0, mad(w.y, v1, mad(w.z, v2, mad(w.w, v3, a5))));
            w = vload4(0, Wp + wi + 6 * ws);
            a6 = mad(w.x, v0, mad(w.y, v1, mad(w.z, v2, mad(w.w, v3, a6))));
            w = vload4(0, Wp + wi + 7 * ws);
            a7 = mad(w.x, v0, mad(w.y, v1, mad(w.z, v2, mad(w.w, v3, a7))));
        }
    }
    if (has_bias) {
        a0 += bias[ocb + 0]; a1 += bias[ocb + 1]; a2 += bias[ocb + 2]; a3 += bias[ocb + 3];
        a4 += bias[ocb + 4]; a5 += bias[ocb + 5]; a6 += bias[ocb + 6]; a7 += bias[ocb + 7];
    }
    int rem = L_out - ol0;
    if (rem >= 4) {
        vstore4(a0, 0, out + mul24(ocb + 0, L_out) + ol0);
        vstore4(a1, 0, out + mul24(ocb + 1, L_out) + ol0);
        vstore4(a2, 0, out + mul24(ocb + 2, L_out) + ol0);
        vstore4(a3, 0, out + mul24(ocb + 3, L_out) + ol0);
        vstore4(a4, 0, out + mul24(ocb + 4, L_out) + ol0);
        vstore4(a5, 0, out + mul24(ocb + 5, L_out) + ol0);
        vstore4(a6, 0, out + mul24(ocb + 6, L_out) + ol0);
        vstore4(a7, 0, out + mul24(ocb + 7, L_out) + ol0);
    } else {
        #define T8V_TAIL(J, A) { __global float* o = out + mul24(ocb + J, L_out) + ol0; \
            o[0] = A.x; if (rem > 1) o[1] = A.y; if (rem > 2) o[2] = A.z; }
        T8V_TAIL(0, a0) T8V_TAIL(1, a1) T8V_TAIL(2, a2) T8V_TAIL(3, a3)
        T8V_TAIL(4, a4) T8V_TAIL(5, a5) T8V_TAIL(6, a6) T8V_TAIL(7, a7)
        #undef T8V_TAIL
    }
}

__kernel void bias_add_f32(__global float* y, __global const float* b, int C, int T) {
    int c = get_global_id(0); int t = get_global_id(1);
    if (c >= C || t >= T) return;
    y[c*T + t] += b[c];
}

__kernel void add_f32(__global float* y, __global const float* x, int N) {
    int i = get_global_id(0); if (i >= N) return;
    y[i] += x[i];
}

// Linear apply: y[oc] = W[oc, :] @ s + b[oc]. fp32 in/out, fp16 weights/style.
__kernel void linear_apply_f32(__global const half* style_fp16,
                               __global const half* W_fp16,
                               __global const half* b_fp16,
                               __global float* out,
                               int C_in, int C_out, int has_bias) {
    int oc = get_global_id(0); if (oc >= C_out) return;
    float acc = 0.0f;
    for (int ic = 0; ic < C_in; ++ic) {
        acc += (float)vload_half(ic, style_fp16) * (float)vload_half(oc*C_in + ic, W_fp16);
    }
    if (has_bias) acc += (float)vload_half(oc, b_fp16);
    out[oc] = acc;
}

// Split 2C-vector into two C-vectors (no T dim, just [2C])
__kernel void split2_f32(__global const float* x, __global float* a, __global float* b, int C) {
    int c = get_global_id(0); if (c >= C) return;
    a[c] = x[c];
    b[c] = x[C + c];
}

// weight_norm reconstruction: read fp16 v + g, output fp32 W = (g/||v||)*v
// Workgroup-per-row (64-lane LDS tree reduction).
__kernel void weightnorm_f32(__global const half* v_fp16,
                             __global const half* g_fp16,
                             __global float* W_fp32,
                             int C_out, int per_oc) {
    int oc = get_group_id(0);
    int lt = get_local_id(0);
    if (oc >= C_out) return;
    int base = oc * per_oc;
    __local float lsq[64];
    float q = 0.0f;
    for (int i = lt; i < per_oc; i += 64) { float x = (float)vload_half(base + i, v_fp16); q += x * x; }
    lsq[lt] = q;
    barrier(CLK_LOCAL_MEM_FENCE);
    for (int off = 32; off > 0; off >>= 1) {
        if (lt < off) lsq[lt] += lsq[lt + off];
        barrier(CLK_LOCAL_MEM_FENCE);
    }
    float norm = sqrt(lsq[0]) + 1e-12f;
    float scale = (float)vload_half(oc, g_fp16) / norm;
    for (int i = lt; i < per_oc; i += 64) W_fp32[base + i] = (float)vload_half(base + i, v_fp16) * scale;
}

// weight_norm reconstruction with K padded up to a multiple of 4 (zeros in
// the pad taps) so conv kernels can vload4 four taps at once.
__kernel void weightnorm_f32_kpad(__global const half* v_fp16,
                                  __global const half* g_fp16,
                                  __global float* Wp_fp32,
                                  int C_out, int C_in, int K, int K_pad) {
    int oc = get_group_id(0);
    int lt = get_local_id(0);
    if (oc >= C_out) return;
    int per_oc = mul24(C_in, K);
    int base = mul24(oc, per_oc);
    __local float lsq[64];
    float q = 0.0f;
    for (int i = lt; i < per_oc; i += 64) { float x = (float)vload_half(base + i, v_fp16); q += x * x; }
    lsq[lt] = q;
    barrier(CLK_LOCAL_MEM_FENCE);
    for (int off = 32; off > 0; off >>= 1) {
        if (lt < off) lsq[lt] += lsq[lt + off];
        barrier(CLK_LOCAL_MEM_FENCE);
    }
    float norm = sqrt(lsq[0]) + 1e-12f;
    float scale = (float)vload_half(oc, g_fp16) / norm;
    int per_oc_pad = mul24(C_in, K_pad);
    int pbase = mul24(oc, per_oc_pad);
    for (int i = lt; i < per_oc_pad; i += 64) {
        int ic = i / K_pad;
        int kp = i - mul24(ic, K_pad);
        float w = (kp < K) ? (float)vload_half(base + mad24(ic, K, kp), v_fp16) * scale : 0.0f;
        Wp_fp32[pbase + i] = w;
    }
}

// K-padded half weight_norm — rows padded to K_pad (multiple of 4) so each
// RGBA16F texel holds 4 consecutive taps.
__kernel void weightnorm_h_kpad(__global const half* v_fp16,
                                __global const half* g_fp16,
                                __global half* Whp,
                                int C_out, int C_in, int K, int K_pad) {
    int oc = get_group_id(0);
    int lt = get_local_id(0);
    if (oc >= C_out) return;
    int per_oc = mul24(C_in, K);
    int base = mul24(oc, per_oc);
    __local float lsq[64];
    float q = 0.0f;
    int p4 = per_oc & ~3;
    for (int i = lt * 4; i < p4; i += 256) {
        float4 xv = convert_float4(vload4(0, v_fp16 + base + i));
        q += dot(xv, xv);
    }
    if (lt == 0) for (int i = p4; i < per_oc; ++i) { float x = (float)vload_half(base + i, v_fp16); q += x * x; }
    lsq[lt] = q;
    barrier(CLK_LOCAL_MEM_FENCE);
    for (int off = 32; off > 0; off >>= 1) {
        if (lt < off) lsq[lt] += lsq[lt + off];
        barrier(CLK_LOCAL_MEM_FENCE);
    }
    float norm = sqrt(lsq[0]) + 1e-12f;
    float scale = (float)vload_half(oc, g_fp16) / norm;
    // Lanes split the ic axis; 4-wide stores along kp (K_pad % 4 == 0).
    int per_oc_pad = mul24(C_in, K_pad);
    int pbase = mul24(oc, per_oc_pad);
    int Kp4 = K_pad >> 2;
    for (int ic = lt; ic < C_in; ic += 64) {
        int sb = base + mul24(ic, K);
        int db = pbase + mul24(ic, K_pad);
        for (int k4 = 0; k4 < Kp4; ++k4) {
            int kp = k4 << 2;
            float4 w;
            if (kp + 3 < K) {
                w = convert_float4(vload4(0, v_fp16 + sb + kp)) * scale;
            } else {
                w.x = (kp     < K) ? (float)vload_half(sb + kp,     v_fp16) * scale : 0.0f;
                w.y = (kp + 1 < K) ? (float)vload_half(sb + kp + 1, v_fp16) * scale : 0.0f;
                w.z = (kp + 2 < K) ? (float)vload_half(sb + kp + 2, v_fp16) * scale : 0.0f;
                w.w = (kp + 3 < K) ? (float)vload_half(sb + kp + 3, v_fp16) * scale : 0.0f;
            }
            vstore_half4(w, 0, Whp + db + kp);
        }
    }
}

// Row norms for weight_norm: scale[oc] = g[oc] / ||v_row||.
__kernel void wn_row_scale(__global const half* v_fp16,
                           __global const half* g_fp16,
                           __global float* scale_out,
                           int C_out, int per_oc) {
    int oc = get_group_id(0);
    int lt = get_local_id(0);
    if (oc >= C_out) return;
    int base = mul24(oc, per_oc);
    __local float lsq[64];
    float q = 0.0f;
    int p4 = per_oc & ~3;
    for (int i = lt * 4; i < p4; i += 256) {
        float4 xv = convert_float4(vload4(0, v_fp16 + base + i));
        q += dot(xv, xv);
    }
    if (lt == 0) for (int i = p4; i < per_oc; ++i) { float x = (float)vload_half(base + i, v_fp16); q += x * x; }
    lsq[lt] = q;
    barrier(CLK_LOCAL_MEM_FENCE);
    for (int off = 32; off > 0; off >>= 1) {
        if (lt < off) lsq[lt] += lsq[lt + off];
        barrier(CLK_LOCAL_MEM_FENCE);
    }
    if (lt == 0) scale_out[oc] = (float)vload_half(oc, g_fp16) / (sqrt(lsq[0]) + 1e-12f);
}

// Write the K-padded weight texels straight into the image (one WI per texel).
__kernel void wn_write_image(__global const half* v_fp16,
                             __global const float* scale,
                             __write_only image2d_t Wimg,
                             int C_out, int C_in, int K, int K_pad) {
    int tx = get_global_id(0);           // texel x: (ic * K_pad + kp) / 4
    int oc = get_global_id(1);
    int Kp4 = K_pad >> 2;
    if (oc >= C_out || tx >= mul24(C_in, Kp4)) return;
    int ic = tx / Kp4;
    int kp = (tx - mul24(ic, Kp4)) << 2;
    int sb = mad24(oc, mul24(C_in, K), mul24(ic, K));
    float sc = scale[oc];
    float4 w;
    if (kp + 3 < K) {
        w = convert_float4(vload4(0, v_fp16 + sb + kp)) * sc;
    } else {
        w.x = (kp     < K) ? (float)vload_half(sb + kp,     v_fp16) * sc : 0.0f;
        w.y = (kp + 1 < K) ? (float)vload_half(sb + kp + 1, v_fp16) * sc : 0.0f;
        w.z = (kp + 2 < K) ? (float)vload_half(sb + kp + 2, v_fp16) * sc : 0.0f;
        w.w = (kp + 3 < K) ? (float)vload_half(sb + kp + 3, v_fp16) * sc : 0.0f;
    }
    write_imageh(Wimg, (int2)(tx, oc), convert_half4(w));
}

// Texture-path conv: weights come through the TP/L1 pipe as RGBA16F texels
// (image row y = output channel; texel x = 4 consecutive taps), while the
// activations stay on the SP->L2 pipe — the guide's "use both memory paths"
// (80-NB295-11 §7.1.5.4). One read_imageh = 4 taps, sampler does addressing,
// CLAMP_TO_EDGE handles the oc tail for free.
// Wave-size override for the hot conv (Qualcomm guide §9.2.1): built with
// -DHT_WAVE_HALF=1 / -DHT_WAVE_FULL=1 via NNOPT_HT_WAVE=half|full. Default
// leaves the compiler heuristic in charge.
#if defined(HT_WAVE_HALF)
#pragma OPENCL EXTENSION cl_qcom_reqd_sub_group_size : enable
__attribute__((qcom_reqd_sub_group_size("half")))
#elif defined(HT_WAVE_FULL)
#pragma OPENCL EXTENSION cl_qcom_reqd_sub_group_size : enable
__attribute__((qcom_reqd_sub_group_size("full")))
#endif
__kernel void conv1d_ht_t8x4(__global const half* in,
                             __read_only image2d_t Wimg,
                             __global half* out,
                             __global const half* bias, int has_bias,
                             int C_in, int C_out, int L_in, int L_out,
                             int K_pad, int padding, int dilation) {
    const sampler_t smp = CLK_NORMALIZED_COORDS_FALSE | CLK_ADDRESS_CLAMP_TO_EDGE | CLK_FILTER_NEAREST;
    int ocb = (int)get_global_id(0) * 8;
    int ol0 = (int)get_global_id(1) * 4;
    if (ocb >= C_out || ol0 >= L_out) return;
    half4 a0 = (half4)(0.0h), a1 = (half4)(0.0h), a2 = (half4)(0.0h), a3 = (half4)(0.0h);
    half4 a4 = (half4)(0.0h), a5 = (half4)(0.0h), a6 = (half4)(0.0h), a7 = (half4)(0.0h);
    int Kp4 = K_pad >> 2;
    int il_base = ol0 - padding;
    int span = mul24(K_pad - 1, dilation);
    int interior = (il_base >= 0 && il_base + span + 3 < L_in) ? 1 : 0;
    for (int ic = 0; ic < C_in; ++ic) {
        int rb = mul24(ic, L_in);
        int txb = mul24(ic, Kp4);
        for (int k4 = 0; k4 < Kp4; ++k4) {
            int kb = k4 << 2;
            half4 v0, v1, v2, v3;
            if (interior) {
                int il = il_base + mul24(kb, dilation);
                v0 = vload4(0, in + rb + il);
                v1 = vload4(0, in + rb + il + dilation);
                v2 = vload4(0, in + rb + il + 2 * dilation);
                v3 = vload4(0, in + rb + il + 3 * dilation);
            } else {
                #define HT_GV(VV, KOFF) { int il = il_base + mul24(kb + KOFF, dilation); \
                    VV.x = (il     >= 0 && il     < L_in) ? in[rb + il]     : (half)0.0h; \
                    VV.y = (il + 1 >= 0 && il + 1 < L_in) ? in[rb + il + 1] : (half)0.0h; \
                    VV.z = (il + 2 >= 0 && il + 2 < L_in) ? in[rb + il + 2] : (half)0.0h; \
                    VV.w = (il + 3 >= 0 && il + 3 < L_in) ? in[rb + il + 3] : (half)0.0h; }
                HT_GV(v0, 0) HT_GV(v1, 1) HT_GV(v2, 2) HT_GV(v3, 3)
                #undef HT_GV
            }
            int tx = txb + k4;
            half4 w;
            #define HT_ROW(A, J) { w = read_imageh(Wimg, smp, (int2)(tx, ocb + J)); \
                A += w.x * v0 + w.y * v1 + w.z * v2 + w.w * v3; }
            HT_ROW(a0, 0) HT_ROW(a1, 1) HT_ROW(a2, 2) HT_ROW(a3, 3)
            HT_ROW(a4, 4) HT_ROW(a5, 5) HT_ROW(a6, 6) HT_ROW(a7, 7)
            #undef HT_ROW
        }
    }
    int oc_last = C_out - 1 - ocb;
    if (has_bias) {
        a0 += bias[ocb];
        if (1 <= oc_last) a1 += bias[ocb + 1];
        if (2 <= oc_last) a2 += bias[ocb + 2];
        if (3 <= oc_last) a3 += bias[ocb + 3];
        if (4 <= oc_last) a4 += bias[ocb + 4];
        if (5 <= oc_last) a5 += bias[ocb + 5];
        if (6 <= oc_last) a6 += bias[ocb + 6];
        if (7 <= oc_last) a7 += bias[ocb + 7];
    }
    int rem = L_out - ol0;
    if (rem >= 4) {
        #define HT_ST4(J, A) if (J <= oc_last) vstore4(A, 0, out + mul24(ocb + J, L_out) + ol0);
        HT_ST4(0, a0) HT_ST4(1, a1) HT_ST4(2, a2) HT_ST4(3, a3)
        HT_ST4(4, a4) HT_ST4(5, a5) HT_ST4(6, a6) HT_ST4(7, a7)
        #undef HT_ST4
    } else {
        #define HT_TAIL(J, A) if (J <= oc_last) { __global half* o = out + mul24(ocb + J, L_out) + ol0; \
            o[0] = A.x; if (rem > 1) o[1] = A.y; if (rem > 2) o[2] = A.z; }
        HT_TAIL(0, a0) HT_TAIL(1, a1) HT_TAIL(2, a2) HT_TAIL(3, a3)
        HT_TAIL(4, a4) HT_TAIL(5, a5) HT_TAIL(6, a6) HT_TAIL(7, a7)
        #undef HT_TAIL
    }
}

// Average 3 fp32 buffers into out (= (a+b+c)/3)
__kernel void avg3_f32(__global const float* a, __global const float* b, __global const float* c,
                       __global float* y, int N) {
    int i = get_global_id(0); if (i >= N) return;
    y[i] = (a[i] + b[i] + c[i]) / 3.0f;
}

// Hybrid: read fp16 y into fp32 x as += (residual add with fp32 accumulator)

// ConvTranspose1d fp32. Weight layout [C_in, C_out/groups, K].
__kernel void convtr1d_f32(__global const float* in, __global const float* W,
                            __global float* out,
                            int C_in, int C_out, int L_in, int L_out,
                            int K, int stride, int padding, int dilation, int groups) {
    int oc = get_global_id(0); int ol = get_global_id(1);
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
            acc += in[ic_g*L_in + il] * W[w_idx];
        }
    }
    out[oc*L_out + ol] = acc;
}

// ConvTranspose1d Cout-tile=4 variant. Each work item produces 4 outputs across
// 4 contiguous oc, sharing the input load (in[ic, il] read once per ic, multiplied
// by 4 weights for the 4 oc values). NO LDS cache — per-oc W slice is C_in*K floats
// which for ups[0] (C_in=512, K=20) is 10240 floats per oc, ×4 = 160 KB, exceeds
// Adreno 620's 32 KB LDS. Rely on L2 cache for W reuse across workgroups.
// W layout [C_in, C_out, K] (groups=1 for our generator) — 4 consecutive oc weights
// for the same (ic, k) are at strided offsets `ic*C_out*K + (oc_base+i)*K + k`.
#define CONVTR_COUT_TILE 4
__kernel void convtr1d_f32_fast_c4(__global const float* in, __global const float* W,
                                     __global float* out,
                                     int C_in, int C_out, int L_in, int L_out,
                                     int K, int stride, int padding, int dilation, int groups) {
    int oc_base = get_global_id(0) * CONVTR_COUT_TILE;
    int ol = get_global_id(1);
    if (oc_base >= C_out || ol >= L_out) return;
    int grp_in_per_out = C_in / groups;
    int g = oc_base / (C_out / groups);
    int oc_in_group = oc_base - g * (C_out / groups);
    int cout_per_g = C_out / groups;
    float acc0 = 0.0f, acc1 = 0.0f, acc2 = 0.0f, acc3 = 0.0f;
    // For dilation=1 (true for ups[0]/ups[1]): the K loop's modulo check
    // `(ol + padding - k) % stride == 0` is equivalent to `k ≡ (ol + padding) mod stride`.
    // So we iterate only the K/stride values that actually contribute — for ups[0]
    // (K=20, stride=10), that's 2 iterations instead of 20.
    // For dilation != 1, fall back to the original loop.
    if (dilation == 1) {
        int k0 = (ol + padding) % stride;
        for (int k = k0; k < K; k += stride) {
            int il = (ol + padding - k) / stride;
            if (il < 0 || il >= L_in) continue;
            for (int ic = 0; ic < grp_in_per_out; ++ic) {
                int ic_g = g * grp_in_per_out + ic;
                float xv = in[ic_g * L_in + il];
                int w_base = ic_g * cout_per_g * K + oc_in_group * K + k;
                acc0 += xv * W[w_base + 0*K];
                acc1 += xv * W[w_base + 1*K];
                acc2 += xv * W[w_base + 2*K];
                acc3 += xv * W[w_base + 3*K];
            }
        }
    } else {
        for (int k = 0; k < K; ++k) {
            int num = ol + padding - k * dilation;
            if (num < 0 || (num % stride) != 0) continue;
            int il = num / stride;
            if (il < 0 || il >= L_in) continue;
            for (int ic = 0; ic < grp_in_per_out; ++ic) {
                int ic_g = g * grp_in_per_out + ic;
                float xv = in[ic_g * L_in + il];
                int w_base = ic_g * cout_per_g * K + oc_in_group * K + k;
                acc0 += xv * W[w_base + 0*K];
                acc1 += xv * W[w_base + 1*K];
                acc2 += xv * W[w_base + 2*K];
                acc3 += xv * W[w_base + 3*K];
            }
        }
    }
    out[(oc_base + 0) * L_out + ol] = acc0;
    out[(oc_base + 1) * L_out + ol] = acc1;
    out[(oc_base + 2) * L_out + ol] = acc2;
    out[(oc_base + 3) * L_out + ol] = acc3;
}

// Phase-major texture repack of a reconstructed (fp32) ConvTranspose weight
// [C_in, C_out, K]: texel (x=(p*ntaps+d)*C4+oc4, y=ic) holds the 4 oc weights
// of tap k=p+d*stride. Mirror of decoder.cpp::convtr_pack_image but fp32 src.
__kernel void gf_convtr_pack_image(__global const float* W,
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
        w.x = (half)W[base + mul24(ocb + 0, K)];
        w.y = (half)W[base + mul24(ocb + 1, K)];
        w.z = (half)W[base + mul24(ocb + 2, K)];
        w.w = (half)W[base + mul24(ocb + 3, K)];
    }
    write_imageh(Wimg, (int2)(x, ic), w);
}

// fp32-buffer variant of decoder.cpp::convtr1d_c4x4_tex: weights through the
// TP/L1 texture pipe (one texel = 4 oc of one (ic,tap)), float4 input loads,
// fp32 accumulate, fused fp16 bias add (kills the separate bias_add pass).
__kernel void convtr1d_f32_c4x4_tex(__global const float* in,
                                    __read_only image2d_t Wimg,
                                    __global float* out,
                                    __global const half* bias, int has_bias,
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
                float4 v = vload4(0, in + mul24(ic, L_in) + il);
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
                v.x = (il     >= 0 && il     < L_in) ? in[rb + il]     : 0.0f;
                v.y = (il + 1 >= 0 && il + 1 < L_in) ? in[rb + il + 1] : 0.0f;
                v.z = (il + 2 >= 0 && il + 2 < L_in) ? in[rb + il + 2] : 0.0f;
                v.w = (il + 3 >= 0 && il + 3 < L_in) ? in[rb + il + 3] : 0.0f;
                float4 w = convert_float4(read_imageh(Wimg, smp, (int2)(tx, ic)));
                acc0 = mad(w.x, v, acc0);
                acc1 = mad(w.y, v, acc1);
                acc2 = mad(w.z, v, acc2);
                acc3 = mad(w.w, v, acc3);
            }
        }
    }
    float b0 = 0.0f, b1 = 0.0f, b2 = 0.0f, b3 = 0.0f;
    if (has_bias) {
        b0 = (float)bias[ocb + 0];
        b1 = (float)bias[ocb + 1];
        b2 = (float)bias[ocb + 2];
        b3 = (float)bias[ocb + 3];
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
        out[ob0 + ol] = a0 + b0;
        out[ob1 + ol] = a1 + b1;
        out[ob2 + ol] = a2 + b2;
        out[ob3 + ol] = a3 + b3;
    }
}

// Buffer copy as an NDRange kernel: clEnqueueCopyBuffer cannot be captured by
// cl_qcom_recordable_queues, so the resblock input copies use this instead.
__kernel void copy_f32(__global const float* src, __global float* dst, int N) {
    int i = get_global_id(0);
    if (i < N) dst[i] = src[i];
}

__kernel void leaky_relu_f32(__global float* y, int N, float slope) {
    int i = get_global_id(0); if (i >= N) return;
    if (y[i] < 0.0f) y[i] *= slope;
}

__kernel void refl_pad_f32(__global const float* x, __global float* y, int C, int T_in, int p_left, int p_right) {
    int c = get_global_id(0); int t_out = get_global_id(1);
    int T_out = T_in + p_left + p_right;
    if (c >= C || t_out >= T_out) return;
    int t_in;
    if (t_out < p_left) t_in = p_left - t_out;
    else if (t_out >= p_left + T_in) t_in = T_in - 2 - (t_out - p_left - T_in);
    else t_in = t_out - p_left;
    if (t_in < 0) t_in = 0;
    if (t_in >= T_in) t_in = T_in - 1;
    y[c*T_out + t_out] = x[c*T_in + t_in];
}

// Plain Conv1d (NOT weight_norm). Weight layout PyTorch [C_out, C_in, K].
// Pack a plain (non-weight-norm) fp16 conv weight [C_out, C_in, K] into an
// RGBA16F image: texel (x = ic*Kp4 + k4, y = oc) = 4 consecutive taps of
// (oc, ic), zero-padded past K. Built once per prefix.
__kernel void nc_pack_image(__global const half* W,
                            __write_only image2d_t Wimg,
                            int C_in, int C_out, int K, int Kp4) {
    int x = get_global_id(0);
    int oc = get_global_id(1);
    if (oc >= C_out || x >= mul24(C_in, Kp4)) return;
    int ic = x / Kp4;
    int k4 = x - mul24(ic, Kp4);
    int kb = k4 << 2;
    int base = mad24(oc, mul24(C_in, K), mul24(ic, K));
    half4 w = (half4)(0.0h);
    if (kb + 0 < K) w.x = W[base + kb + 0];
    if (kb + 1 < K) w.y = W[base + kb + 1];
    if (kb + 2 < K) w.z = W[base + kb + 2];
    if (kb + 3 < K) w.w = W[base + kb + 3];
    write_imageh(Wimg, (int2)(x, oc), w);
}

// Plain Conv1d with weights through the texture pipe (noise_convs: 22ch in,
// strided). 4 oc x 4 ol per work item; taps fetched 4-wide per texel; fp32
// accumulate; fused fp16 bias.
__kernel void plain_conv1d_tex_f32(__global const float* in,
                                   __read_only image2d_t Wimg,
                                   __global const half* bias, int has_bias,
                                   __global float* out,
                                   int C_in, int C_out, int L_in, int L_out,
                                   int K, int Kp4, int stride, int padding) {
    const sampler_t smp = CLK_NORMALIZED_COORDS_FALSE | CLK_ADDRESS_CLAMP_TO_EDGE | CLK_FILTER_NEAREST;
    int ocb = (int)get_global_id(0) * 4;
    int ol0 = (int)get_global_id(1) * 4;
    if (ocb >= C_out || ol0 >= L_out) return;
    float4 a0 = (float4)(0.0f), a1 = (float4)(0.0f), a2 = (float4)(0.0f), a3 = (float4)(0.0f);
    int il00 = mul24(ol0, stride) - padding;
    int il_last = mad24(ol0 + 3, stride, -padding) + (Kp4 << 2) - 1;
    int interior = (il00 >= 0 && il_last < L_in) ? 1 : 0;
    for (int ic = 0; ic < C_in; ++ic) {
        int rb = mul24(ic, L_in);
        int txb = mul24(ic, Kp4);
        for (int k4 = 0; k4 < Kp4; ++k4) {
            int kb = k4 << 2;
            float4 v0, v1, v2, v3;
            if (interior) {
                v0 = vload4(0, in + rb + il00 + kb);
                v1 = vload4(0, in + rb + il00 + stride + kb);
                v2 = vload4(0, in + rb + il00 + 2 * stride + kb);
                v3 = vload4(0, in + rb + il00 + 3 * stride + kb);
            } else {
                int b0 = il00 + kb, b1 = b0 + stride, b2 = b1 + stride, b3 = b2 + stride;
                v0.x = (b0 >= 0 && b0 < L_in) ? in[rb + b0] : 0.0f;
                v0.y = (b0 + 1 >= 0 && b0 + 1 < L_in) ? in[rb + b0 + 1] : 0.0f;
                v0.z = (b0 + 2 >= 0 && b0 + 2 < L_in) ? in[rb + b0 + 2] : 0.0f;
                v0.w = (b0 + 3 >= 0 && b0 + 3 < L_in) ? in[rb + b0 + 3] : 0.0f;
                v1.x = (b1 >= 0 && b1 < L_in) ? in[rb + b1] : 0.0f;
                v1.y = (b1 + 1 >= 0 && b1 + 1 < L_in) ? in[rb + b1 + 1] : 0.0f;
                v1.z = (b1 + 2 >= 0 && b1 + 2 < L_in) ? in[rb + b1 + 2] : 0.0f;
                v1.w = (b1 + 3 >= 0 && b1 + 3 < L_in) ? in[rb + b1 + 3] : 0.0f;
                v2.x = (b2 >= 0 && b2 < L_in) ? in[rb + b2] : 0.0f;
                v2.y = (b2 + 1 >= 0 && b2 + 1 < L_in) ? in[rb + b2 + 1] : 0.0f;
                v2.z = (b2 + 2 >= 0 && b2 + 2 < L_in) ? in[rb + b2 + 2] : 0.0f;
                v2.w = (b2 + 3 >= 0 && b2 + 3 < L_in) ? in[rb + b2 + 3] : 0.0f;
                v3.x = (b3 >= 0 && b3 < L_in) ? in[rb + b3] : 0.0f;
                v3.y = (b3 + 1 >= 0 && b3 + 1 < L_in) ? in[rb + b3 + 1] : 0.0f;
                v3.z = (b3 + 2 >= 0 && b3 + 2 < L_in) ? in[rb + b3 + 2] : 0.0f;
                v3.w = (b3 + 3 >= 0 && b3 + 3 < L_in) ? in[rb + b3 + 3] : 0.0f;
            }
            int tx = txb + k4;
            float4 w0 = convert_float4(read_imageh(Wimg, smp, (int2)(tx, ocb + 0)));
            float4 w1 = convert_float4(read_imageh(Wimg, smp, (int2)(tx, ocb + 1)));
            float4 w2 = convert_float4(read_imageh(Wimg, smp, (int2)(tx, ocb + 2)));
            float4 w3 = convert_float4(read_imageh(Wimg, smp, (int2)(tx, ocb + 3)));
            a0 += (float4)(dot(w0, v0), dot(w0, v1), dot(w0, v2), dot(w0, v3));
            a1 += (float4)(dot(w1, v0), dot(w1, v1), dot(w1, v2), dot(w1, v3));
            a2 += (float4)(dot(w2, v0), dot(w2, v1), dot(w2, v2), dot(w2, v3));
            a3 += (float4)(dot(w3, v0), dot(w3, v1), dot(w3, v2), dot(w3, v3));
        }
    }
    if (has_bias) {
        a0 += (float4)((float)bias[ocb + 0]);
        a1 += (float4)((float)bias[ocb + 1]);
        a2 += (float4)((float)bias[ocb + 2]);
        a3 += (float4)((float)bias[ocb + 3]);
    }
    int n_ol = L_out - ol0; if (n_ol > 4) n_ol = 4;
    for (int j = 0; j < n_ol; ++j) {
        float s0 = (j == 0) ? a0.x : (j == 1) ? a0.y : (j == 2) ? a0.z : a0.w;
        float s1 = (j == 0) ? a1.x : (j == 1) ? a1.y : (j == 2) ? a1.z : a1.w;
        float s2 = (j == 0) ? a2.x : (j == 1) ? a2.y : (j == 2) ? a2.z : a2.w;
        float s3 = (j == 0) ? a3.x : (j == 1) ? a3.y : (j == 2) ? a3.z : a3.w;
        out[mul24(ocb + 0, L_out) + ol0 + j] = s0;
        out[mul24(ocb + 1, L_out) + ol0 + j] = s1;
        out[mul24(ocb + 2, L_out) + ol0 + j] = s2;
        out[mul24(ocb + 3, L_out) + ol0 + j] = s3;
    }
}

__kernel void plain_conv1d_f32(__global const float* in, __global const half* W_fp16,
                                __global const half* b_fp16,
                                __global float* out,
                                int C_in, int C_out, int L_in, int L_out,
                                int K, int stride, int padding, int dilation, int has_bias) {
    int oc = get_global_id(0); int ol = get_global_id(1);
    if (oc >= C_out || ol >= L_out) return;
    float acc = 0.0f;
    for (int k = 0; k < K; ++k) {
        int il = ol * stride - padding + k * dilation;
        if (il < 0 || il >= L_in) continue;
        for (int ic = 0; ic < C_in; ++ic) {
            acc += in[ic*L_in + il] * (float)vload_half(oc*C_in*K + ic*K + k, W_fp16);
        }
    }
    if (has_bias) acc += (float)vload_half(oc, b_fp16);
    out[oc*L_out + ol] = acc;
}

// conv_post epilogue: mag = exp(rows 0..n_freq), phase = sin(rows n_freq..2n_freq),
// converted straight into the fp16 buffers the host iSTFT consumes.
__kernel void conv_post_split_f32(__global const float* y,
                                  __global half* mag, __global half* phase,
                                  int n_freq, int T) {
    int i = get_global_id(0);
    int NT = mul24(n_freq, T);
    if (i >= NT) return;
    vstore_half(exp(y[i]), i, mag);
    vstore_half(sin(y[NT + i]), i, phase);
}

__kernel void div_scalar_f32(__global float* y, int N, float s) {
    int i = get_global_id(0); if (i >= N) return;
    y[i] /= s;
}

__kernel void exp_f32(__global float* y, int N) {
    int i = get_global_id(0); if (i >= N) return;
    y[i] = exp(y[i]);
}

__kernel void sin_act_f32(__global float* y, int N) {
    int i = get_global_id(0); if (i >= N) return;
    y[i] = sin(y[i]);
}
)CLC";

static bool ensure_built_gf(OpenCLContext& cl_ctx) {
    if (g_kf_f16_to_f32) return true;
    cl_int err = CL_SUCCESS;
    cl_device_id dev = cl_ctx.device();
    // NNOPT_HT_WAVE=half|full forces the wave size on conv1d_ht_t8x4 (§9.2.1
    // sweep knob). The program cache key hashes the options, so each variant
    // gets its own cached binary.
    // Default = FULL wave (swept 2026-06-06: conv 1556→1068 ms with (4,32)+full;
    // half/compiler-default both slower). NNOPT_HT_WAVE=none|half overrides.
    std::string gf_opts = "-cl-fast-relaxed-math -DHT_WAVE_FULL=1";
    if (const char* w = std::getenv("NNOPT_HT_WAVE")) {
        if (w[0] == 'n')      gf_opts = "-cl-fast-relaxed-math";
        else if (w[0] == 'h') gf_opts = "-cl-fast-relaxed-math -DHT_WAVE_HALF=1";
    }
    g_gf_prog = nnopt_build_program_cached(cl_ctx.context(), dev, k_gf_src,
                                            gf_opts.c_str(), "generator_fp32", &err);
    if (!g_gf_prog) return false;
    g_kf_f16_to_f32 = clCreateKernel(g_gf_prog, "cvt_f16_to_f32", &err);
    g_kf_f32_to_f16 = clCreateKernel(g_gf_prog, "cvt_f32_to_f16", &err);
    g_kf_instnorm   = clCreateKernel(g_gf_prog, "instnorm_f32", &err);
    g_kf_adain_combine = clCreateKernel(g_gf_prog, "adain_combine_f32", &err);
    g_kf_snake1d    = clCreateKernel(g_gf_prog, "snake1d_f32", &err);
    g_kf_conv1d     = clCreateKernel(g_gf_prog, "conv1d_f32", &err);
    g_kf_conv1d_t4x4 = clCreateKernel(g_gf_prog, "conv1d_f32_t4x4", &err);
    g_kf_conv1d_t8x4 = clCreateKernel(g_gf_prog, "conv1d_f32_t8x4", &err);
    g_kf_conv1d_t8x4v = clCreateKernel(g_gf_prog, "conv1d_f32_t8x4v", &err);
    g_kf_weightnorm_kpad = clCreateKernel(g_gf_prog, "weightnorm_f32_kpad", &err);
    g_kf_bias_add   = clCreateKernel(g_gf_prog, "bias_add_f32", &err);
    g_kf_add        = clCreateKernel(g_gf_prog, "add_f32", &err);
    g_kf_linear_apply = clCreateKernel(g_gf_prog, "linear_apply_f32", &err);
    g_kf_split2     = clCreateKernel(g_gf_prog, "split2_f32", &err);
    g_kf_weightnorm = clCreateKernel(g_gf_prog, "weightnorm_f32", &err);
    g_kf_avg3       = clCreateKernel(g_gf_prog, "avg3_f32", &err);
    g_kf_convtr1d   = clCreateKernel(g_gf_prog, "convtr1d_f32", &err);
    g_kf_convtr1d_c4 = clCreateKernel(g_gf_prog, "convtr1d_f32_fast_c4", &err);
    g_kf_leaky      = clCreateKernel(g_gf_prog, "leaky_relu_f32", &err);
    g_kf_copy_f32   = clCreateKernel(g_gf_prog, "copy_f32", &err);
    g_kf_nc_pack    = clCreateKernel(g_gf_prog, "nc_pack_image", &err);
    g_kf_plain_conv1d_tex = clCreateKernel(g_gf_prog, "plain_conv1d_tex_f32", &err);
    g_kf_refl_pad   = clCreateKernel(g_gf_prog, "refl_pad_f32", &err);
    g_kf_plain_conv1d = clCreateKernel(g_gf_prog, "plain_conv1d_f32", &err);
    g_kf_conv_post_split = clCreateKernel(g_gf_prog, "conv_post_split_f32", &err);
    g_kf_in_adain_snake = clCreateKernel(g_gf_prog, "instnorm_adain_snake_f32", &err);
    g_kf_in_adain_snake_h = clCreateKernel(g_gf_prog, "instnorm_adain_snake_f16out", &err);
    g_kf_conv1d_h8x4 = clCreateKernel(g_gf_prog, "conv1d_h16in_t8x4", &err);
    g_kf_conv1d_hh = clCreateKernel(g_gf_prog, "conv1d_hh_t8x4", &err);
    g_kf_weightnorm_h = clCreateKernel(g_gf_prog, "weightnorm_f16w", &err);
    g_kf_in_adain_snake_h2h = clCreateKernel(g_gf_prog, "instnorm_adain_snake_h2h", &err);
    g_kf_add_from_h = clCreateKernel(g_gf_prog, "add_f32_from_h", &err);
    g_kf_conv1d_hh_lds = clCreateKernel(g_gf_prog, "conv1d_hh_lds4x4", &err);
    g_kf_conv1d_ht = clCreateKernel(g_gf_prog, "conv1d_ht_t8x4", &err);
    g_kf_istft = clCreateKernel(g_gf_prog, "istft_f32", &err);
    g_kf_conv1d_ht8 = clCreateKernel(g_gf_prog, "conv1d_ht_t8x8", &err);
    g_kf_conv1d_ht48 = clCreateKernel(g_gf_prog, "conv1d_ht_t4x8", &err);
    g_kf_weightnorm_h_kpad = clCreateKernel(g_gf_prog, "weightnorm_h_kpad", &err);
    g_kf_wn_row_scale = clCreateKernel(g_gf_prog, "wn_row_scale", &err);
    g_kf_wn_write_image = clCreateKernel(g_gf_prog, "wn_write_image", &err);
    g_kf_gf_convtr_pack = clCreateKernel(g_gf_prog, "gf_convtr_pack_image", &err);
    g_kf_convtr1d_c4x4_tex_f32 = clCreateKernel(g_gf_prog, "convtr1d_f32_c4x4_tex", &err);
    g_kf_div_scalar = clCreateKernel(g_gf_prog, "div_scalar_f32", &err);
    g_kf_exp        = clCreateKernel(g_gf_prog, "exp_f32", &err);
    g_kf_sin_act    = clCreateKernel(g_gf_prog, "sin_act_f32", &err);
    return g_kf_f16_to_f32 && g_kf_f32_to_f16 && g_kf_instnorm && g_kf_adain_combine
        && g_kf_snake1d && g_kf_conv1d && g_kf_conv1d_t4x4 && g_kf_conv1d_t8x4 && g_kf_conv1d_t8x4v && g_kf_weightnorm_kpad && g_kf_bias_add && g_kf_add && g_kf_linear_apply
        && g_kf_split2 && g_kf_weightnorm && g_kf_avg3
        && g_kf_convtr1d && g_kf_convtr1d_c4 && g_kf_leaky && g_kf_copy_f32 && g_kf_nc_pack && g_kf_plain_conv1d_tex && g_kf_refl_pad && g_kf_plain_conv1d && g_kf_conv_post_split && g_kf_in_adain_snake && g_kf_in_adain_snake_h && g_kf_conv1d_h8x4 && g_kf_conv1d_hh && g_kf_weightnorm_h && g_kf_in_adain_snake_h2h && g_kf_add_from_h && g_kf_conv1d_hh_lds && g_kf_conv1d_ht && g_kf_conv1d_ht8 && g_kf_conv1d_ht48 && g_kf_istft && g_kf_weightnorm_h_kpad && g_kf_wn_row_scale && g_kf_wn_write_image && g_kf_gf_convtr_pack && g_kf_convtr1d_c4x4_tex_f32
        && g_kf_div_scalar && g_kf_exp && g_kf_sin_act;
}

// ── Recording-stable buffer arena ────────────────────────────────────────────
// cl_qcom_recordable_queues bakes cl_mem handles into a recording, so every
// dispatch in the recorded generator span must see IDENTICAL buffers across
// calls with the same T. The arena hands buffers out by call-sequence position
// (deterministic for a fixed T + env-gate configuration) and recycles them on
// the next call instead of create/release churn (Qualcomm guide §5.7.1 also
// wants no clCreate/Release between kernel launches). While the arena is
// active, gen_release() is a no-op — slots are reused via reset().
struct GenArena {
    struct Slot { cl_mem mem = nullptr; size_t bytes = 0; };
    std::vector<Slot> slots;
    size_t pos = 0;
    uint64_t generation = 0;   // bumps on any slot realloc → invalidates recordings
    void reset() { pos = 0; }  // grow-only: slots persist across T changes (streaming
                               // chunks vary in length; dropping per T was full
                               // clCreateBuffer churn on every chunk)
    cl_mem get(cl_context ctx, size_t bytes) {
        if (pos == slots.size()) slots.push_back({});
        Slot& s = slots[pos++];
        if (s.bytes < bytes || !s.mem) {
            if (s.mem) clReleaseMemObject(s.mem);
            // 1/8 headroom rounded to 4 KB: chunk-length jitter below +12.5%
            // reuses the slot instead of regrowing it
            size_t alloc = (bytes + bytes / 8 + 4095) & ~(size_t)4095;
            cl_int err = CL_SUCCESS;
            s.mem = clCreateBuffer(ctx, CL_MEM_READ_WRITE, alloc, nullptr, &err);
            s.bytes = (err == CL_SUCCESS) ? alloc : 0;
            if (err != CL_SUCCESS) s.mem = nullptr;
            ++generation;
        }
        return s.mem;
    }
};
static GenArena g_gen_arena;
static bool g_gen_arena_active = false;

// All generator-path buffer releases funnel through this: forwards to
// clReleaseMemObject normally; no-op while the arena owns the allocations.
static void gen_release(cl_mem m) {
    if (!g_gen_arena_active && m) clReleaseMemObject(m);
}

static cl_mem alloc_f16(OpenCLContext& cl_ctx, size_t n_elems) {
    if (g_gen_arena_active) return g_gen_arena.get(cl_ctx.context(), n_elems * sizeof(cl_half));
    cl_int err = CL_SUCCESS;
    return clCreateBuffer(cl_ctx.context(), CL_MEM_READ_WRITE, n_elems * sizeof(cl_half), nullptr, &err);
}

static cl_mem alloc_fp32(OpenCLContext& cl_ctx, size_t n_elems) {
    if (g_gen_arena_active) return g_gen_arena.get(cl_ctx.context(), n_elems * sizeof(float));
    cl_int err = CL_SUCCESS;
    return clCreateBuffer(cl_ctx.context(), CL_MEM_READ_WRITE, n_elems * sizeof(float), nullptr, &err);
}

// fp16 → fp32 conversion (public)
extern "C" int gen_fp16_to_fp32(OpenCLContext& cl_ctx, cl_command_queue queue,
                                cl_mem fp16_buf, cl_mem fp32_buf, int N) {
    if (!ensure_built_gf(cl_ctx)) return -1;
    clSetKernelArg(g_kf_f16_to_f32, 0, sizeof(cl_mem), &fp16_buf);
    clSetKernelArg(g_kf_f16_to_f32, 1, sizeof(cl_mem), &fp32_buf);
    clSetKernelArg(g_kf_f16_to_f32, 2, sizeof(int), &N);
    size_t gws = (size_t)N;
    return nnopt_enqueue_profiled(queue, g_kf_f16_to_f32, 1, nullptr, &gws, nullptr, 0, nullptr, nullptr);
}

extern "C" int gen_fp32_to_fp16(OpenCLContext& cl_ctx, cl_command_queue queue,
                                cl_mem fp32_buf, cl_mem fp16_buf, int N) {
    if (!ensure_built_gf(cl_ctx)) return -1;
    clSetKernelArg(g_kf_f32_to_f16, 0, sizeof(cl_mem), &fp32_buf);
    clSetKernelArg(g_kf_f32_to_f16, 1, sizeof(cl_mem), &fp16_buf);
    clSetKernelArg(g_kf_f32_to_f16, 2, sizeof(int), &N);
    size_t gws = (size_t)N;
    return nnopt_enqueue_profiled(queue, g_kf_f32_to_f16, 1, nullptr, &gws, nullptr, 0, nullptr, nullptr);
}

// ── int8 conv path via cl_qcom_dot_product8 (NNOPT_DOT8=1, plan-v3 #3) ──────
// The fp16 conv1d_ht_t8x4 is instruction-issue bound (~84 instr / 128 MACs in
// the inner loop). qcom_dot8_acc folds 4 int8 MACs + accumulate into ONE
// instruction, and int8 halves load bytes — the only structural >1.3x left on
// the resblock conv wall. Spec (guide §9.5.1): qcom_dot8_acc(uint p0, uint p1,
// int acc) where p0 = four SIGNED 8-bit (weights), p1 = four UNSIGNED 8-bit
// (activations, asymmetric zp=128). Quantization scheme:
//   weights: per-oc symmetric scale s_w[oc] (packed once from the existing
//            fp16 weight image), plus per-oc wsum = Σ w_q for the zero-point
//            correction: out = s_w·s_a·(acc − 128·Σw_q) + bias.
//   acts:    dynamic per-tensor scale (absmax reduce, stays on GPU — no host
//            sync), quantized + transposed to a channels-last [row][C/4]
//            uchar4-in-uint layout. The dot then runs over 4 consecutive
//            input channels — no strided gathers for dilated taps, and the
//            K=3 kernels waste no lanes. Padding rows are filled 0x80808080
//            (= zp), so the zp correction cancels them exactly and the hot
//            loop needs no bounds checks.
// Separate program: the extension pragma must not poison the main gf build
// on devices without the extension.
static cl_program g_i8_prog = nullptr;
static cl_kernel  g_ki8_absmax_part = nullptr;
static cl_kernel  g_ki8_absmax_fin = nullptr;
static cl_kernel  g_ki8_quant_lc = nullptr;
static cl_kernel  g_ki8_wpack = nullptr;
static cl_kernel  g_ki8_conv = nullptr;

static const char* k_i8_src = R"CLC(
#pragma OPENCL EXTENSION cl_khr_fp16 : enable
#pragma OPENCL EXTENSION cl_qcom_dot_product8 : enable

// Stage 1: grid-stride absmax partials over the half NCL input.
__kernel void i8_absmax_part(__global const half* x, int N, __global float* part) {
    int lt = (int)get_local_id(0);
    float m = 0.0f;
    for (int i = (int)get_global_id(0); i < N; i += (int)get_global_size(0))
        m = fmax(m, fabs((float)x[i]));
    __local float red[256];
    red[lt] = m;
    barrier(CLK_LOCAL_MEM_FENCE);
    for (int off = 128; off > 0; off >>= 1) {
        if (lt < off) red[lt] = fmax(red[lt], red[lt + off]);
        barrier(CLK_LOCAL_MEM_FENCE);
    }
    if (lt == 0) part[get_group_id(0)] = red[0];
}

// Stage 2: one 64-lane WG folds the partials into sa = {s_a, 1/s_a}.
__kernel void i8_absmax_fin(__global const float* part, int NP, __global float* sa) {
    int lt = (int)get_local_id(0);
    float m = (lt < NP) ? part[lt] : 0.0f;
    __local float red[64];
    red[lt] = m;
    barrier(CLK_LOCAL_MEM_FENCE);
    for (int off = 32; off > 0; off >>= 1) {
        if (lt < off) red[lt] = fmax(red[lt], red[lt + off]);
        barrier(CLK_LOCAL_MEM_FENCE);
    }
    if (lt == 0) {
        float s = fmax(red[0], 1e-12f) / 127.0f;
        sa[0] = s;
        sa[1] = 1.0f / s;
    }
}

// Stage 3: quantize + transpose NCL half -> channels-last [row][Cq] packed
// uchar4. Row r covers il = r - padding; out-of-range rows = 0x80808080 (zp).
// LDS-tiled transpose: a 32x32 (rows x channel-quads) tile is read with
// lanes coalesced along L (the NCL fast axis) and written with lanes
// coalesced along Cq (the xq fast axis). Naive version had 16x write-line
// waste (~3.2 ms per big resblock input, x19 calls).
__kernel void i8_quant_lc(__global const half* x, __global const float* sa,
                          __global uint* xq, int C_in, int L_in,
                          int padding, int rows_alloc) {
    int Cq = C_in >> 2;
    int r0   = (int)get_group_id(0) * 32;   // tile origin: rows
    int icq0 = (int)get_group_id(1) * 32;   // tile origin: channel-quads
    int lt = (int)get_local_id(0);          // 0..255 (8 rows of 32 lanes)
    float inv = sa[1];
    __local uint tile[32][33];              // [icq_in_tile][r_in_tile], +1 pad vs bank conflicts
    // read pass: lane = row offset (coalesced along L), 8 icq slices per lane batch
    int rr = lt & 31;
    int r = r0 + rr;
    int il = r - padding;
    bool live = (il >= 0 && il < L_in);
    for (int s = lt >> 5; s < 32; s += 8) {
        int icq = icq0 + s;
        uint v = 0x80808080u;
        if (live && icq < Cq) {
            int ic = icq << 2;
            int q0 = clamp(convert_int_rte((float)x[mad24(ic,     L_in, il)] * inv) + 128, 0, 255);
            int q1 = clamp(convert_int_rte((float)x[mad24(ic + 1, L_in, il)] * inv) + 128, 0, 255);
            int q2 = clamp(convert_int_rte((float)x[mad24(ic + 2, L_in, il)] * inv) + 128, 0, 255);
            int q3 = clamp(convert_int_rte((float)x[mad24(ic + 3, L_in, il)] * inv) + 128, 0, 255);
            v = (uint)q0 | ((uint)q1 << 8) | ((uint)q2 << 16) | ((uint)q3 << 24);
        }
        tile[s][rr] = v;
    }
    barrier(CLK_LOCAL_MEM_FENCE);
    // write pass: lane = icq offset (coalesced along Cq)
    int cc = lt & 31;
    int icq_w = icq0 + cc;
    if (icq_w >= Cq) return;
    for (int s = lt >> 5; s < 32; s += 8) {
        int r_w = r0 + s;
        if (r_w < rows_alloc) xq[mad24(r_w, Cq, icq_w)] = tile[cc][s];
    }
}

// One-time weight pack from the fp16 weight image (texel x = ic*Kp4 + k/4,
// component k&3, row y = oc). One 64-lane WG per oc: pass 1 absmax -> s_w,
// pass 2 pack wq[oc][k][icq] (LSB = ic0) + reduce wsum.
__kernel void i8_wpack(__read_only image2d_t Wimg, __global uint* wq,
                       __global float* sw, __global int* wsum,
                       int C_in, int K, int K_pad) {
    const sampler_t smp = CLK_NORMALIZED_COORDS_FALSE | CLK_ADDRESS_CLAMP_TO_EDGE | CLK_FILTER_NEAREST;
    int oc = (int)get_group_id(0);
    int lt = (int)get_local_id(0);
    int Cq = C_in >> 2;
    int Kp4 = K_pad >> 2;
    int ntex = mul24(C_in, Kp4);
    float m = 0.0f;
    for (int t = lt; t < ntex; t += 64) {
        float4 w = convert_float4(read_imageh(Wimg, smp, (int2)(t, oc)));
        m = fmax(m, fmax(fmax(fabs(w.x), fabs(w.y)), fmax(fabs(w.z), fabs(w.w))));
    }
    __local float red[64];
    red[lt] = m;
    barrier(CLK_LOCAL_MEM_FENCE);
    for (int off = 32; off > 0; off >>= 1) {
        if (lt < off) red[lt] = fmax(red[lt], red[lt + off]);
        barrier(CLK_LOCAL_MEM_FENCE);
    }
    float s = fmax(red[0], 1e-12f) / 127.0f;
    float inv = 1.0f / s;
    barrier(CLK_LOCAL_MEM_FENCE);
    int n_out = mul24(K, Cq);
    int acc = 0;
    for (int t = lt; t < n_out; t += 64) {
        int k = t / Cq;
        int icq = t - mul24(k, Cq);
        int ic0 = icq << 2;
        int kc = k & 3;
        uint v = 0;
        for (int j = 0; j < 4; ++j) {
            float4 w = convert_float4(read_imageh(Wimg, smp,
                           (int2)(mad24(ic0 + j, Kp4, k >> 2), oc)));
            float wc = (kc == 0) ? w.x : (kc == 1) ? w.y : (kc == 2) ? w.z : w.w;
            int q = clamp(convert_int_rte(wc * inv), -127, 127);
            acc += q;
            v |= ((uint)(q & 0xff)) << (8 * j);
        }
        wq[mad24(oc, n_out, t)] = v;
    }
    __local int redi[64];
    redi[lt] = acc;
    barrier(CLK_LOCAL_MEM_FENCE);
    for (int off = 32; off > 0; off >>= 1) {
        if (lt < off) redi[lt] += redi[lt + off];
        barrier(CLK_LOCAL_MEM_FENCE);
    }
    if (lt == 0) { sw[oc] = s; wsum[oc] = redi[0]; }
}

// The conv: OCT oc x 4 ol per WI (OCT = 8 default, 4 via NNOPT_I8_OCT for
// register-pressure relief), dot over 4-channel groups, 4 icq batched per
// iteration (uint4 loads). Per (k, icq4) iteration at OCT=8: 12 x 16B loads
// feed 128 dot8 = 512 MACs — 4x the MAC-per-load-instruction density of the
// fp16 kernel. Padding pre-baked into xq rows (row = ol + k*dilation).
// Wave-size attribute via NNOPT_I8_WAVE=half|full (no barriers here, same
// kernel class as conv1d_ht_t8x4 where full won).
#ifndef I8_OCT
#define I8_OCT 4
#endif
#if defined(I8_WAVE_HALF)
#pragma OPENCL EXTENSION cl_qcom_reqd_sub_group_size : enable
__attribute__((qcom_reqd_sub_group_size("half")))
#elif defined(I8_WAVE_FULL)
#pragma OPENCL EXTENSION cl_qcom_reqd_sub_group_size : enable
__attribute__((qcom_reqd_sub_group_size("full")))
#endif
__kernel void conv1d_i8_t8x4(__global const uint* xq, __global const uint* wq,
                             __global const float* sw, __global const int* wsum,
                             __global const float* sa,
                             __global half* out, __global const half* bias,
                             int has_bias, int C_in, int C_out, int L_out,
                             int K, int dilation) {
    int ocb = (int)get_global_id(0) * I8_OCT;
    int ol0 = (int)get_global_id(1) * 4;
    if (ocb >= C_out || ol0 >= L_out) return;
    int Cq = C_in >> 2;
    int wrow = mul24(K, Cq);
    int wbase = mul24(ocb, wrow);
    int4 a0 = (int4)0, a1 = (int4)0, a2 = (int4)0, a3 = (int4)0;
#if I8_OCT == 8
    int4 a4 = (int4)0, a5 = (int4)0, a6 = (int4)0, a7 = (int4)0;
#endif
    for (int k = 0; k < K; ++k) {
        int r = ol0 + mul24(k, dilation);
        int xb = mul24(r, Cq);
        int wo = wbase + mul24(k, Cq);
        for (int icq = 0; icq < Cq; icq += 4) {
            uint4 x0 = vload4(0, xq + xb + icq);
            uint4 x1 = vload4(0, xq + xb + Cq + icq);
            uint4 x2 = vload4(0, xq + xb + 2 * Cq + icq);
            uint4 x3 = vload4(0, xq + xb + 3 * Cq + icq);
            #define I8_ROW(A, J) { uint4 w = vload4(0, wq + wo + mul24(J, wrow) + icq); \
                A.x = qcom_dot8_acc(w.s3, x0.s3, qcom_dot8_acc(w.s2, x0.s2, qcom_dot8_acc(w.s1, x0.s1, qcom_dot8_acc(w.s0, x0.s0, A.x)))); \
                A.y = qcom_dot8_acc(w.s3, x1.s3, qcom_dot8_acc(w.s2, x1.s2, qcom_dot8_acc(w.s1, x1.s1, qcom_dot8_acc(w.s0, x1.s0, A.y)))); \
                A.z = qcom_dot8_acc(w.s3, x2.s3, qcom_dot8_acc(w.s2, x2.s2, qcom_dot8_acc(w.s1, x2.s1, qcom_dot8_acc(w.s0, x2.s0, A.z)))); \
                A.w = qcom_dot8_acc(w.s3, x3.s3, qcom_dot8_acc(w.s2, x3.s2, qcom_dot8_acc(w.s1, x3.s1, qcom_dot8_acc(w.s0, x3.s0, A.w)))); }
            I8_ROW(a0, 0) I8_ROW(a1, 1) I8_ROW(a2, 2) I8_ROW(a3, 3)
#if I8_OCT == 8
            I8_ROW(a4, 4) I8_ROW(a5, 5) I8_ROW(a6, 6) I8_ROW(a7, 7)
#endif
            #undef I8_ROW
        }
    }
    float s_a = sa[0];
    int oc_last = C_out - 1 - ocb;
    int rem = L_out - ol0;
    #define I8_ST(J, A) if (J <= oc_last) { \
        float sc = sw[ocb + J] * s_a; \
        int zc = wsum[ocb + J] << 7; \
        float4 r4 = (float4)((float)(A.x - zc), (float)(A.y - zc), \
                             (float)(A.z - zc), (float)(A.w - zc)) * sc; \
        if (has_bias) r4 += (float)bias[ocb + J]; \
        __global half* o = out + mad24(ocb + J, L_out, ol0); \
        if (rem >= 4) { vstore_half4(r4, 0, o); } \
        else { vstore_half(r4.x, 0, o); \
               if (rem > 1) vstore_half(r4.y, 1, o); \
               if (rem > 2) vstore_half(r4.z, 2, o); } }
    I8_ST(0, a0) I8_ST(1, a1) I8_ST(2, a2) I8_ST(3, a3)
#if I8_OCT == 8
    I8_ST(4, a4) I8_ST(5, a5) I8_ST(6, a6) I8_ST(7, a7)
#endif
    #undef I8_ST
}
)CLC";

// Compile-time oc-tile for the i8 conv. Default 4 (swept 2026-06-06: OCT=8's
// eight int4 accumulators spill — 949 ms vs 671 ms total at OCT=4; full-wave
// regresses 1.6x, geometry optimum stays (4,32)). NNOPT_I8_OCT=8 re-tests.
static int nnopt_i8_oct() {
    static int oct = 0;
    if (oct == 0) {
        const char* e = std::getenv("NNOPT_I8_OCT");
        oct = (e && atoi(e) == 8) ? 8 : 4;
    }
    return oct;
}

static bool ensure_built_i8(OpenCLContext& cl_ctx) {
    static int checked = 0;   // 0 = not yet, 1 = ok, -1 = unavailable
    if (checked == 1) return true;
    if (checked == -1) return false;
    cl_device_id dev = cl_ctx.device();
    size_t ext_sz = 0;
    clGetDeviceInfo(dev, CL_DEVICE_EXTENSIONS, 0, nullptr, &ext_sz);
    std::string exts(ext_sz, '\0');
    clGetDeviceInfo(dev, CL_DEVICE_EXTENSIONS, ext_sz, &exts[0], nullptr);
    if (exts.find("cl_qcom_dot_product8") == std::string::npos) {
        NNOPT_ERROR("NNOPT_DOT8=1 but device lacks cl_qcom_dot_product8 — falling back to fp16 conv");
        checked = -1;
        return false;
    }
    cl_int err = CL_SUCCESS;
    std::string opts = "-cl-fast-relaxed-math";
    if (nnopt_i8_oct() == 8) opts += " -DI8_OCT=8";
    if (const char* w = std::getenv("NNOPT_I8_WAVE")) {
        if (w[0] == 'h')      opts += " -DI8_WAVE_HALF=1";
        else if (w[0] == 'f') opts += " -DI8_WAVE_FULL=1";
    }
    g_i8_prog = nnopt_build_program_cached(cl_ctx.context(), dev, k_i8_src,
                                           opts.c_str(), "conv_i8_dot8", &err);
    if (!g_i8_prog) { checked = -1; return false; }
    g_ki8_absmax_part = clCreateKernel(g_i8_prog, "i8_absmax_part", &err);
    g_ki8_absmax_fin  = clCreateKernel(g_i8_prog, "i8_absmax_fin", &err);
    g_ki8_quant_lc    = clCreateKernel(g_i8_prog, "i8_quant_lc", &err);
    g_ki8_wpack       = clCreateKernel(g_i8_prog, "i8_wpack", &err);
    g_ki8_conv        = clCreateKernel(g_i8_prog, "conv1d_i8_t8x4", &err);
    bool ok = g_ki8_absmax_part && g_ki8_absmax_fin && g_ki8_quant_lc && g_ki8_wpack && g_ki8_conv;
    checked = ok ? 1 : -1;
    return ok;
}

// Weight_norm cache for fp32 reconstructed weights.
struct WCacheFP32 { cl_mem W = nullptr; cl_mem Wp = nullptr; cl_mem Wh = nullptr; cl_mem Wimg = nullptr; cl_mem bias_f32 = nullptr; int C_out = 0, C_in = 0, K = 0; int K_pad = 0; };
static std::unordered_map<std::string, WCacheFP32> g_wn_cache_fp32;

static cl_mem get_wn_conv_weight_fp32(OpenCLContext& cl_ctx, Weights& weights, cl_command_queue queue,
                                      const std::string& prefix, int* C_out_o, int* C_in_o, int* K_o) {
    auto it = g_wn_cache_fp32.find(prefix);
    if (it != g_wn_cache_fp32.end()) {
        if (C_out_o) *C_out_o = it->second.C_out;
        if (C_in_o)  *C_in_o  = it->second.C_in;
        if (K_o)     *K_o     = it->second.K;
        return it->second.W;
    }
    std::string vk = prefix + ".weight_v";
    std::string gk = prefix + ".weight_g";
    cl_mem v = weights.get_buffer(vk);
    cl_mem g = weights.get_buffer(gk);
    if (!v || !g) { NNOPT_ERROR_FMT("wn_fp32: missing %s", prefix.c_str()); return nullptr; }
    auto shape = weights.get_shape(vk);
    int C_out = shape[0], C_in = (shape.size() > 1 ? shape[1] : 1), K = (shape.size() > 2 ? shape[2] : 1);
    int per_oc = C_in * K;
    cl_mem W = alloc_fp32(cl_ctx, C_out * per_oc);
    clSetKernelArg(g_kf_weightnorm, 0, sizeof(cl_mem), &v);
    clSetKernelArg(g_kf_weightnorm, 1, sizeof(cl_mem), &g);
    clSetKernelArg(g_kf_weightnorm, 2, sizeof(cl_mem), &W);
    clSetKernelArg(g_kf_weightnorm, 3, sizeof(int), &C_out);
    clSetKernelArg(g_kf_weightnorm, 4, sizeof(int), &per_oc);
    size_t gws = (size_t)C_out * 64;  // workgroup-per-row reduction
    size_t lws_wn = 64;
    nnopt_enqueue_profiled(queue, g_kf_weightnorm, 1, nullptr, &gws, &lws_wn, 0, nullptr, nullptr);
    g_wn_cache_fp32[prefix] = {W, nullptr, nullptr, nullptr, nullptr, C_out, C_in, K, 0};
    if (C_out_o) *C_out_o = C_out;
    if (C_in_o) *C_in_o = C_in;
    if (K_o) *K_o = K;
    return W;
}

// Get (or lazily build) the K-padded fp32 weight for the vectorized conv.
static cl_mem get_wn_conv_weight_fp32_kpad(OpenCLContext& cl_ctx, Weights& weights, cl_command_queue queue,
                                           const std::string& prefix, int* C_out_o, int* C_in_o,
                                           int* K_o, int* K_pad_o) {
    // Ensure base cache entry exists (also yields dims).
    int C_out, C_in, K;
    if (!get_wn_conv_weight_fp32(cl_ctx, weights, queue, prefix, &C_out, &C_in, &K)) return nullptr;
    auto it = g_wn_cache_fp32.find(prefix);
    if (it == g_wn_cache_fp32.end()) return nullptr;
    if (!it->second.Wp) {
        int K_pad = (K + 3) & ~3;
        cl_mem v = weights.get_buffer(prefix + ".weight_v");
        cl_mem g = weights.get_buffer(prefix + ".weight_g");
        if (!v || !g) return nullptr;
        cl_mem Wp = alloc_fp32(cl_ctx, (size_t)C_out * C_in * K_pad);
        clSetKernelArg(g_kf_weightnorm_kpad, 0, sizeof(cl_mem), &v);
        clSetKernelArg(g_kf_weightnorm_kpad, 1, sizeof(cl_mem), &g);
        clSetKernelArg(g_kf_weightnorm_kpad, 2, sizeof(cl_mem), &Wp);
        clSetKernelArg(g_kf_weightnorm_kpad, 3, sizeof(int), &C_out);
        clSetKernelArg(g_kf_weightnorm_kpad, 4, sizeof(int), &C_in);
        clSetKernelArg(g_kf_weightnorm_kpad, 5, sizeof(int), &K);
        clSetKernelArg(g_kf_weightnorm_kpad, 6, sizeof(int), &K_pad);
        size_t gws = (size_t)C_out * 64;
        size_t lws = 64;
        nnopt_enqueue_profiled(queue, g_kf_weightnorm_kpad, 1, nullptr, &gws, &lws, 0, nullptr, nullptr);
        it->second.Wp = Wp;
        it->second.K_pad = K_pad;
    }
    if (C_out_o) *C_out_o = it->second.C_out;
    if (C_in_o)  *C_in_o  = it->second.C_in;
    if (K_o)     *K_o     = it->second.K;
    if (K_pad_o) *K_pad_o = it->second.K_pad;
    return it->second.Wp;
}

// Conv1d with fp32 in/out, fp16 weight_norm weights (reconstructed to fp32 on first use).
static int conv1d_wn_fp32(OpenCLContext& cl_ctx, Weights& weights, cl_command_queue queue,
                          const std::string& prefix, cl_mem in_f32, cl_mem out_f32,
                          int L_in, int stride, int padding, int dilation, int groups,
                          int* L_out_o, int* C_out_o) {
    int C_out, C_in, K;
    cl_mem W = get_wn_conv_weight_fp32(cl_ctx, weights, queue, prefix, &C_out, &C_in, &K);
    if (!W) return -1;
    cl_mem bias_fp16 = weights.get_buffer(prefix + ".bias", /*optional=*/true);
    int L_out = (L_in + 2*padding - dilation*(K-1) - 1) / stride + 1;
    if (L_out_o) *L_out_o = L_out;
    if (C_out_o) *C_out_o = C_out;
    // NNOPT_CONV_T4X4=0 forces the legacy scalar kernel (A/B validation only).
    static int use_t4x4 = -1;
    if (use_t4x4 == -1) {
        const char* e = std::getenv("NNOPT_CONV_T4X4");
        use_t4x4 = (e && e[0] == '0') ? 0 : 1;
    }
    static int use_t8 = -1;
    if (use_t8 == -1) {
        const char* e = std::getenv("NNOPT_T8");
        use_t8 = (e && e[0] == '0') ? 0 : 1;
    }
    if (use_t4x4 && use_t8 && stride == 1 && groups == 1 && C_out >= 8 && g_kf_conv1d_t8x4) {
        // Bias is fused into the conv epilogue: convert it to a cached fp32
        // buffer once per weight (saves a full [C,T] bias_add pass per conv).
        cl_mem bias_f32 = nullptr;
        if (bias_fp16) {
            auto itb = g_wn_cache_fp32.find(prefix);
            if (itb != g_wn_cache_fp32.end() && itb->second.bias_f32) {
                bias_f32 = itb->second.bias_f32;
            } else {
                bias_f32 = alloc_fp32(cl_ctx, C_out);
                clSetKernelArg(g_kf_f16_to_f32, 0, sizeof(cl_mem), &bias_fp16);
                clSetKernelArg(g_kf_f16_to_f32, 1, sizeof(cl_mem), &bias_f32);
                clSetKernelArg(g_kf_f16_to_f32, 2, sizeof(int), &C_out);
                size_t gws_b = (size_t)C_out;
                nnopt_enqueue_profiled(queue, g_kf_f16_to_f32, 1, nullptr, &gws_b, nullptr, 0, nullptr, nullptr);
                if (itb != g_wn_cache_fp32.end()) itb->second.bias_f32 = bias_f32;
            }
        }
        int has_bias = bias_f32 ? 1 : 0;
        cl_mem bias_arg = bias_f32 ? bias_f32 : W;  // dummy when unused
        static int use_t8v = -1;
        if (use_t8v == -1) {
            const char* e = std::getenv("NNOPT_T8V");
            use_t8v = (e && e[0] == '1') ? 1 : 0;  // default OFF: measured 7.43s vs 7.03s (register spill)
        }
        if (use_t8v) {
            int Kp = 0, Co2, Ci2, K2;
            cl_mem Wp = get_wn_conv_weight_fp32_kpad(cl_ctx, weights, queue, prefix, &Co2, &Ci2, &K2, &Kp);
            if (Wp) {
                clSetKernelArg(g_kf_conv1d_t8x4v, 0, sizeof(cl_mem), &in_f32);
                clSetKernelArg(g_kf_conv1d_t8x4v, 1, sizeof(cl_mem), &Wp);
                clSetKernelArg(g_kf_conv1d_t8x4v, 2, sizeof(cl_mem), &out_f32);
                clSetKernelArg(g_kf_conv1d_t8x4v, 3, sizeof(cl_mem), &bias_arg);
                clSetKernelArg(g_kf_conv1d_t8x4v, 4, sizeof(int), &has_bias);
                clSetKernelArg(g_kf_conv1d_t8x4v, 5, sizeof(int), &C_in);
                clSetKernelArg(g_kf_conv1d_t8x4v, 6, sizeof(int), &C_out);
                clSetKernelArg(g_kf_conv1d_t8x4v, 7, sizeof(int), &L_in);
                clSetKernelArg(g_kf_conv1d_t8x4v, 8, sizeof(int), &L_out);
                clSetKernelArg(g_kf_conv1d_t8x4v, 9, sizeof(int), &Kp);
                clSetKernelArg(g_kf_conv1d_t8x4v, 10, sizeof(int), &padding);
                clSetKernelArg(g_kf_conv1d_t8x4v, 11, sizeof(int), &dilation);
                const int local_t = 128;
                size_t tiles_l = (size_t)((L_out + 3) / 4);
                size_t tiles_l_padded = ((tiles_l + local_t - 1) / local_t) * local_t;
                size_t gws[2] = {(size_t)(C_out / 8), tiles_l_padded};
                size_t lws[2] = {1, (size_t)local_t};
                nnopt_enqueue_profiled(queue, g_kf_conv1d_t8x4v, 2, nullptr, gws, lws, 0, nullptr, nullptr);
                return 0;
            }
        }
        clSetKernelArg(g_kf_conv1d_t8x4, 0, sizeof(cl_mem), &in_f32);
        clSetKernelArg(g_kf_conv1d_t8x4, 1, sizeof(cl_mem), &W);
        clSetKernelArg(g_kf_conv1d_t8x4, 2, sizeof(cl_mem), &out_f32);
        clSetKernelArg(g_kf_conv1d_t8x4, 3, sizeof(cl_mem), &bias_arg);
        clSetKernelArg(g_kf_conv1d_t8x4, 4, sizeof(int), &has_bias);
        clSetKernelArg(g_kf_conv1d_t8x4, 5, sizeof(int), &C_in);
        clSetKernelArg(g_kf_conv1d_t8x4, 6, sizeof(int), &C_out);
        clSetKernelArg(g_kf_conv1d_t8x4, 7, sizeof(int), &L_in);
        clSetKernelArg(g_kf_conv1d_t8x4, 8, sizeof(int), &L_out);
        clSetKernelArg(g_kf_conv1d_t8x4, 9, sizeof(int), &K);
        clSetKernelArg(g_kf_conv1d_t8x4, 10, sizeof(int), &padding);
        clSetKernelArg(g_kf_conv1d_t8x4, 11, sizeof(int), &dilation);
        const int local_t = 128;
        size_t tiles_l = (size_t)((L_out + 3) / 4);
        size_t tiles_l_padded = ((tiles_l + local_t - 1) / local_t) * local_t;
        size_t gws[2] = {(size_t)((C_out + 7) / 8), tiles_l_padded};
        size_t lws[2] = {1, (size_t)local_t};
        nnopt_enqueue_profiled(queue, g_kf_conv1d_t8x4, 2, nullptr, gws, lws, 0, nullptr, nullptr);
        return 0;  // bias fused — skip the trailing bias_add block
    } else if (use_t4x4 && stride == 1 && groups == 1 && (C_out % 4) == 0) {
        // Register-tiled fast path (all generator resblock convs land here).
        clSetKernelArg(g_kf_conv1d_t4x4, 0, sizeof(cl_mem), &in_f32);
        clSetKernelArg(g_kf_conv1d_t4x4, 1, sizeof(cl_mem), &W);
        clSetKernelArg(g_kf_conv1d_t4x4, 2, sizeof(cl_mem), &out_f32);
        clSetKernelArg(g_kf_conv1d_t4x4, 3, sizeof(int), &C_in);
        clSetKernelArg(g_kf_conv1d_t4x4, 4, sizeof(int), &C_out);
        clSetKernelArg(g_kf_conv1d_t4x4, 5, sizeof(int), &L_in);
        clSetKernelArg(g_kf_conv1d_t4x4, 6, sizeof(int), &L_out);
        clSetKernelArg(g_kf_conv1d_t4x4, 7, sizeof(int), &K);
        clSetKernelArg(g_kf_conv1d_t4x4, 8, sizeof(int), &padding);
        clSetKernelArg(g_kf_conv1d_t4x4, 9, sizeof(int), &dilation);
        const int local_t = 128;  // first dim 1 keeps weight reads uniform per group
        size_t tiles_l = (size_t)((L_out + 3) / 4);
        size_t tiles_l_padded = ((tiles_l + local_t - 1) / local_t) * local_t;
        size_t gws[2] = {(size_t)(C_out / 4), tiles_l_padded};
        size_t lws[2] = {1, (size_t)local_t};
        nnopt_enqueue_profiled(queue, g_kf_conv1d_t4x4, 2, nullptr, gws, lws, 0, nullptr, nullptr);
    } else {
    clSetKernelArg(g_kf_conv1d, 0, sizeof(cl_mem), &in_f32);
    clSetKernelArg(g_kf_conv1d, 1, sizeof(cl_mem), &W);
    clSetKernelArg(g_kf_conv1d, 2, sizeof(cl_mem), &out_f32);
    clSetKernelArg(g_kf_conv1d, 3, sizeof(int), &C_in);
    clSetKernelArg(g_kf_conv1d, 4, sizeof(int), &C_out);
    clSetKernelArg(g_kf_conv1d, 5, sizeof(int), &L_in);
    clSetKernelArg(g_kf_conv1d, 6, sizeof(int), &L_out);
    clSetKernelArg(g_kf_conv1d, 7, sizeof(int), &K);
    clSetKernelArg(g_kf_conv1d, 8, sizeof(int), &stride);
    clSetKernelArg(g_kf_conv1d, 9, sizeof(int), &padding);
    clSetKernelArg(g_kf_conv1d, 10, sizeof(int), &dilation);
    clSetKernelArg(g_kf_conv1d, 11, sizeof(int), &groups);
    const int local_t = 32;
    size_t L_padded = ((L_out + local_t - 1) / local_t) * local_t;
    size_t gws[2] = {(size_t)C_out, L_padded};
    size_t lws[2] = {1, (size_t)local_t};
    nnopt_enqueue_profiled(queue, g_kf_conv1d, 2, nullptr, gws, lws, 0, nullptr, nullptr);
    }
    if (bias_fp16) {
        // Convert bias to fp32 once
        cl_mem bias_fp32 = alloc_fp32(cl_ctx, C_out);
        clSetKernelArg(g_kf_f16_to_f32, 0, sizeof(cl_mem), &bias_fp16);
        clSetKernelArg(g_kf_f16_to_f32, 1, sizeof(cl_mem), &bias_fp32);
        clSetKernelArg(g_kf_f16_to_f32, 2, sizeof(int), &C_out);
        size_t gws_b = (size_t)C_out;
        nnopt_enqueue_profiled(queue, g_kf_f16_to_f32, 1, nullptr, &gws_b, nullptr, 0, nullptr, nullptr);
        clSetKernelArg(g_kf_bias_add, 0, sizeof(cl_mem), &out_f32);
        clSetKernelArg(g_kf_bias_add, 1, sizeof(cl_mem), &bias_fp32);
        clSetKernelArg(g_kf_bias_add, 2, sizeof(int), &C_out);
        clSetKernelArg(g_kf_bias_add, 3, sizeof(int), &L_out);
        size_t gws_ba[2] = {(size_t)C_out, (size_t)L_out};
        nnopt_enqueue_profiled(queue, g_kf_bias_add, 2, nullptr, gws_ba, nullptr, 0, nullptr, nullptr);
        gen_release(bias_fp32);
    }
    return 0;
}

// AdaIN1d (fp32) — takes fp32 input, applies fc(style_fp16) → gamma, beta, then
// instance norm + (1+gamma)*xn + beta. Writes fp32 output.
static int apply_adain1d_fp32(OpenCLContext& cl_ctx, Weights& weights, cl_command_queue queue,
                              const std::string& adain_prefix,
                              cl_mem x_f32, cl_mem y_f32, cl_mem style_fp16, int C, int T) {
    cl_mem fc_w = weights.get_buffer(adain_prefix + ".fc.weight");
    cl_mem fc_b = weights.get_buffer(adain_prefix + ".fc.bias");
    if (!fc_w || !fc_b) { NNOPT_ERROR_FMT("adain_fp32: missing %s.fc", adain_prefix.c_str()); return -1; }
    int two_C = 2*C, style_dim = 128;
    cl_mem h = alloc_fp32(cl_ctx, two_C);
    int has_bias = 1;
    clSetKernelArg(g_kf_linear_apply, 0, sizeof(cl_mem), &style_fp16);
    clSetKernelArg(g_kf_linear_apply, 1, sizeof(cl_mem), &fc_w);
    clSetKernelArg(g_kf_linear_apply, 2, sizeof(cl_mem), &fc_b);
    clSetKernelArg(g_kf_linear_apply, 3, sizeof(cl_mem), &h);
    clSetKernelArg(g_kf_linear_apply, 4, sizeof(int), &style_dim);
    clSetKernelArg(g_kf_linear_apply, 5, sizeof(int), &two_C);
    clSetKernelArg(g_kf_linear_apply, 6, sizeof(int), &has_bias);
    size_t gws_lin = (size_t)two_C;
    nnopt_enqueue_profiled(queue, g_kf_linear_apply, 1, nullptr, &gws_lin, nullptr, 0, nullptr, nullptr);
    // split
    cl_mem gamma = alloc_fp32(cl_ctx, C);
    cl_mem beta  = alloc_fp32(cl_ctx, C);
    clSetKernelArg(g_kf_split2, 0, sizeof(cl_mem), &h);
    clSetKernelArg(g_kf_split2, 1, sizeof(cl_mem), &gamma);
    clSetKernelArg(g_kf_split2, 2, sizeof(cl_mem), &beta);
    clSetKernelArg(g_kf_split2, 3, sizeof(int), &C);
    size_t gws_sp = (size_t)C;
    nnopt_enqueue_profiled(queue, g_kf_split2, 1, nullptr, &gws_sp, nullptr, 0, nullptr, nullptr);
    // instance norm
    cl_mem xn = alloc_fp32(cl_ctx, C * T);
    float eps = 1e-5f;
    clSetKernelArg(g_kf_instnorm, 0, sizeof(cl_mem), &x_f32);
    clSetKernelArg(g_kf_instnorm, 1, sizeof(cl_mem), &xn);
    clSetKernelArg(g_kf_instnorm, 2, sizeof(int), &C);
    clSetKernelArg(g_kf_instnorm, 3, sizeof(int), &T);
    clSetKernelArg(g_kf_instnorm, 4, sizeof(float), &eps);
    size_t gws_in = (size_t)C * 64;  // workgroup-per-channel reduction
    size_t lws_in = 64;
    nnopt_enqueue_profiled(queue, g_kf_instnorm, 1, nullptr, &gws_in, &lws_in, 0, nullptr, nullptr);
    // adain combine
    clSetKernelArg(g_kf_adain_combine, 0, sizeof(cl_mem), &xn);
    clSetKernelArg(g_kf_adain_combine, 1, sizeof(cl_mem), &gamma);
    clSetKernelArg(g_kf_adain_combine, 2, sizeof(cl_mem), &beta);
    clSetKernelArg(g_kf_adain_combine, 3, sizeof(cl_mem), &y_f32);
    clSetKernelArg(g_kf_adain_combine, 4, sizeof(int), &C);
    clSetKernelArg(g_kf_adain_combine, 5, sizeof(int), &T);
    size_t gws_ad[2] = {(size_t)C, (size_t)T};
    nnopt_enqueue_profiled(queue, g_kf_adain_combine, 2, nullptr, gws_ad, nullptr, 0, nullptr, nullptr);
    gen_release(h);
    gen_release(gamma);
    gen_release(beta);
    gen_release(xn);
    return 0;
}

static int apply_snake1d_fp32(OpenCLContext& cl_ctx, Weights& weights, cl_command_queue queue,
                              const std::string& alpha_key, cl_mem y_f32, int C, int T) {
    (void)cl_ctx;
    cl_mem alpha = weights.get_buffer(alpha_key);
    if (!alpha) { NNOPT_ERROR_FMT("snake_fp32: missing %s", alpha_key.c_str()); return -1; }
    clSetKernelArg(g_kf_snake1d, 0, sizeof(cl_mem), &y_f32);
    clSetKernelArg(g_kf_snake1d, 1, sizeof(cl_mem), &alpha);
    clSetKernelArg(g_kf_snake1d, 2, sizeof(int), &C);
    clSetKernelArg(g_kf_snake1d, 3, sizeof(int), &T);
    size_t gws[2] = {(size_t)C, (size_t)T};
    return nnopt_enqueue_profiled(queue, g_kf_snake1d, 2, nullptr, gws, nullptr, 0, nullptr, nullptr);
}

// Fused adain+snake: linear_apply + split for gamma/beta, then ONE kernel for
// instnorm + style combine + snake activation.
static int apply_adain_snake_fp32(OpenCLContext& cl_ctx, Weights& weights, cl_command_queue queue,
                                  const std::string& adain_prefix, const std::string& alpha_key,
                                  cl_mem x_f32, cl_mem y_f32, cl_mem style_fp16, int C, int T) {
    cl_mem fc_w = weights.get_buffer(adain_prefix + ".fc.weight");
    cl_mem fc_b = weights.get_buffer(adain_prefix + ".fc.bias");
    cl_mem alpha = weights.get_buffer(alpha_key);
    if (!fc_w || !fc_b || !alpha) { NNOPT_ERROR_FMT("adain_snake_fp32: missing %s / %s", adain_prefix.c_str(), alpha_key.c_str()); return -1; }
    int two_C = 2*C, style_dim = 128;
    cl_mem h = alloc_fp32(cl_ctx, two_C);
    int has_bias = 1;
    clSetKernelArg(g_kf_linear_apply, 0, sizeof(cl_mem), &style_fp16);
    clSetKernelArg(g_kf_linear_apply, 1, sizeof(cl_mem), &fc_w);
    clSetKernelArg(g_kf_linear_apply, 2, sizeof(cl_mem), &fc_b);
    clSetKernelArg(g_kf_linear_apply, 3, sizeof(cl_mem), &h);
    clSetKernelArg(g_kf_linear_apply, 4, sizeof(int), &style_dim);
    clSetKernelArg(g_kf_linear_apply, 5, sizeof(int), &two_C);
    clSetKernelArg(g_kf_linear_apply, 6, sizeof(int), &has_bias);
    size_t gws_lin = (size_t)two_C;
    nnopt_enqueue_profiled(queue, g_kf_linear_apply, 1, nullptr, &gws_lin, nullptr, 0, nullptr, nullptr);
    cl_mem gamma = alloc_fp32(cl_ctx, C);
    cl_mem beta  = alloc_fp32(cl_ctx, C);
    clSetKernelArg(g_kf_split2, 0, sizeof(cl_mem), &h);
    clSetKernelArg(g_kf_split2, 1, sizeof(cl_mem), &gamma);
    clSetKernelArg(g_kf_split2, 2, sizeof(cl_mem), &beta);
    clSetKernelArg(g_kf_split2, 3, sizeof(int), &C);
    size_t gws_sp = (size_t)C;
    nnopt_enqueue_profiled(queue, g_kf_split2, 1, nullptr, &gws_sp, nullptr, 0, nullptr, nullptr);
    float eps = 1e-5f;
    clSetKernelArg(g_kf_in_adain_snake, 0, sizeof(cl_mem), &x_f32);
    clSetKernelArg(g_kf_in_adain_snake, 1, sizeof(cl_mem), &y_f32);
    clSetKernelArg(g_kf_in_adain_snake, 2, sizeof(cl_mem), &gamma);
    clSetKernelArg(g_kf_in_adain_snake, 3, sizeof(cl_mem), &beta);
    clSetKernelArg(g_kf_in_adain_snake, 4, sizeof(cl_mem), &alpha);
    clSetKernelArg(g_kf_in_adain_snake, 5, sizeof(int), &C);
    clSetKernelArg(g_kf_in_adain_snake, 6, sizeof(int), &T);
    clSetKernelArg(g_kf_in_adain_snake, 7, sizeof(float), &eps);
    size_t gws_f = (size_t)C * 64;
    size_t lws_f = 64;
    nnopt_enqueue_profiled(queue, g_kf_in_adain_snake, 1, nullptr, &gws_f, &lws_f, 0, nullptr, nullptr);
    gen_release(h);
    gen_release(gamma);
    gen_release(beta);
    return 0;
}

// f16-out variant: same as apply_adain_snake_fp32 but writes a half buffer
// (the conv-input scratch). One bounded truncation; residual spine stays fp32.
static int apply_adain_snake_f16out(OpenCLContext& cl_ctx, Weights& weights, cl_command_queue queue,
                                    const std::string& adain_prefix, const std::string& alpha_key,
                                    cl_mem x_f32, cl_mem y_f16, cl_mem style_fp16, int C, int T) {
    cl_mem fc_w = weights.get_buffer(adain_prefix + ".fc.weight");
    cl_mem fc_b = weights.get_buffer(adain_prefix + ".fc.bias");
    cl_mem alpha = weights.get_buffer(alpha_key);
    if (!fc_w || !fc_b || !alpha) { NNOPT_ERROR_FMT("adain_snake_f16: missing %s / %s", adain_prefix.c_str(), alpha_key.c_str()); return -1; }
    int two_C = 2*C, style_dim = 128;
    cl_mem h = alloc_fp32(cl_ctx, two_C);
    int has_bias = 1;
    clSetKernelArg(g_kf_linear_apply, 0, sizeof(cl_mem), &style_fp16);
    clSetKernelArg(g_kf_linear_apply, 1, sizeof(cl_mem), &fc_w);
    clSetKernelArg(g_kf_linear_apply, 2, sizeof(cl_mem), &fc_b);
    clSetKernelArg(g_kf_linear_apply, 3, sizeof(cl_mem), &h);
    clSetKernelArg(g_kf_linear_apply, 4, sizeof(int), &style_dim);
    clSetKernelArg(g_kf_linear_apply, 5, sizeof(int), &two_C);
    clSetKernelArg(g_kf_linear_apply, 6, sizeof(int), &has_bias);
    size_t gws_lin = (size_t)two_C;
    nnopt_enqueue_profiled(queue, g_kf_linear_apply, 1, nullptr, &gws_lin, nullptr, 0, nullptr, nullptr);
    cl_mem gamma = alloc_fp32(cl_ctx, C);
    cl_mem beta  = alloc_fp32(cl_ctx, C);
    clSetKernelArg(g_kf_split2, 0, sizeof(cl_mem), &h);
    clSetKernelArg(g_kf_split2, 1, sizeof(cl_mem), &gamma);
    clSetKernelArg(g_kf_split2, 2, sizeof(cl_mem), &beta);
    clSetKernelArg(g_kf_split2, 3, sizeof(int), &C);
    size_t gws_sp = (size_t)C;
    nnopt_enqueue_profiled(queue, g_kf_split2, 1, nullptr, &gws_sp, nullptr, 0, nullptr, nullptr);
    float eps = 1e-5f;
    clSetKernelArg(g_kf_in_adain_snake_h, 0, sizeof(cl_mem), &x_f32);
    clSetKernelArg(g_kf_in_adain_snake_h, 1, sizeof(cl_mem), &y_f16);
    clSetKernelArg(g_kf_in_adain_snake_h, 2, sizeof(cl_mem), &gamma);
    clSetKernelArg(g_kf_in_adain_snake_h, 3, sizeof(cl_mem), &beta);
    clSetKernelArg(g_kf_in_adain_snake_h, 4, sizeof(cl_mem), &alpha);
    clSetKernelArg(g_kf_in_adain_snake_h, 5, sizeof(int), &C);
    clSetKernelArg(g_kf_in_adain_snake_h, 6, sizeof(int), &T);
    clSetKernelArg(g_kf_in_adain_snake_h, 7, sizeof(float), &eps);
    size_t gws_f = (size_t)C * 64;
    size_t lws_f = 64;
    nnopt_enqueue_profiled(queue, g_kf_in_adain_snake_h, 1, nullptr, &gws_f, &lws_f, 0, nullptr, nullptr);
    gen_release(h);
    gen_release(gamma);
    gen_release(beta);
    return 0;
}

// Get (or build) the half-precision reconstructed weight.
static cl_mem get_wn_conv_weight_f16(OpenCLContext& cl_ctx, Weights& weights, cl_command_queue queue,
                                     const std::string& prefix, int* C_out_o, int* C_in_o, int* K_o) {
    int C_out, C_in, K;
    if (!get_wn_conv_weight_fp32(cl_ctx, weights, queue, prefix, &C_out, &C_in, &K)) return nullptr;
    auto it = g_wn_cache_fp32.find(prefix);
    if (it == g_wn_cache_fp32.end()) return nullptr;
    if (!it->second.Wh) {
        cl_mem v = weights.get_buffer(prefix + ".weight_v");
        cl_mem g = weights.get_buffer(prefix + ".weight_g");
        if (!v || !g) return nullptr;
        int per_oc = C_in * K;
        cl_mem Wh = alloc_f16(cl_ctx, (size_t)C_out * per_oc);
        clSetKernelArg(g_kf_weightnorm_h, 0, sizeof(cl_mem), &v);
        clSetKernelArg(g_kf_weightnorm_h, 1, sizeof(cl_mem), &g);
        clSetKernelArg(g_kf_weightnorm_h, 2, sizeof(cl_mem), &Wh);
        clSetKernelArg(g_kf_weightnorm_h, 3, sizeof(int), &C_out);
        clSetKernelArg(g_kf_weightnorm_h, 4, sizeof(int), &per_oc);
        size_t gws = (size_t)C_out * 64;
        size_t lws = 64;
        nnopt_enqueue_profiled(queue, g_kf_weightnorm_h, 1, nullptr, &gws, &lws, 0, nullptr, nullptr);
        it->second.Wh = Wh;
    }
    if (C_out_o) *C_out_o = it->second.C_out;
    if (C_in_o)  *C_in_o  = it->second.C_in;
    if (K_o)     *K_o     = it->second.K;
    return it->second.Wh;
}

// Get (or build) the weight image: RGBA16F image2d, row y = oc, texel x =
// 4 consecutive taps of the K-padded half weight row.
static cl_mem get_wn_conv_weight_image(OpenCLContext& cl_ctx, Weights& weights, cl_command_queue queue,
                                       const std::string& prefix, int* C_out_o, int* C_in_o,
                                       int* K_o, int* K_pad_o) {
    int C_out, C_in, K;
    if (!get_wn_conv_weight_fp32(cl_ctx, weights, queue, prefix, &C_out, &C_in, &K)) return nullptr;
    auto it = g_wn_cache_fp32.find(prefix);
    if (it == g_wn_cache_fp32.end()) return nullptr;
    if (!it->second.Wimg) {
        int K_pad = (K + 3) & ~3;
        cl_mem v = weights.get_buffer(prefix + ".weight_v");
        cl_mem g = weights.get_buffer(prefix + ".weight_g");
        if (!v || !g) return nullptr;
        // 1. per-row scales (tiny)
        cl_mem scl = alloc_fp32(cl_ctx, (size_t)C_out);
        int per_oc = C_in * K;
        clSetKernelArg(g_kf_wn_row_scale, 0, sizeof(cl_mem), &v);
        clSetKernelArg(g_kf_wn_row_scale, 1, sizeof(cl_mem), &g);
        clSetKernelArg(g_kf_wn_row_scale, 2, sizeof(cl_mem), &scl);
        clSetKernelArg(g_kf_wn_row_scale, 3, sizeof(int), &C_out);
        clSetKernelArg(g_kf_wn_row_scale, 4, sizeof(int), &per_oc);
        size_t gws_s = (size_t)C_out * 64;
        size_t lws_s = 64;
        nnopt_enqueue_profiled(queue, g_kf_wn_row_scale, 1, nullptr, &gws_s, &lws_s, 0, nullptr, nullptr);
        // 2. write texels straight into the image (one WI per texel)
        cl_image_format fmt;
        fmt.image_channel_order = CL_RGBA;
        fmt.image_channel_data_type = CL_HALF_FLOAT;
        cl_image_desc desc;
        memset(&desc, 0, sizeof(desc));
        desc.image_type = CL_MEM_OBJECT_IMAGE2D;
        desc.image_width = (size_t)(C_in * K_pad / 4);
        desc.image_height = (size_t)C_out;
        cl_int err = CL_SUCCESS;
        cl_mem img = clCreateImage(cl_ctx.context(), CL_MEM_READ_WRITE, &fmt, &desc, nullptr, &err);
        if (err != CL_SUCCESS || !img) {
            NNOPT_ERROR_FMT("weight image create failed (%d) for %s w=%zu h=%zu",
                            (int)err, prefix.c_str(), desc.image_width, desc.image_height);
            gen_release(scl);
            return nullptr;
        }
        clSetKernelArg(g_kf_wn_write_image, 0, sizeof(cl_mem), &v);
        clSetKernelArg(g_kf_wn_write_image, 1, sizeof(cl_mem), &scl);
        clSetKernelArg(g_kf_wn_write_image, 2, sizeof(cl_mem), &img);
        clSetKernelArg(g_kf_wn_write_image, 3, sizeof(int), &C_out);
        clSetKernelArg(g_kf_wn_write_image, 4, sizeof(int), &C_in);
        clSetKernelArg(g_kf_wn_write_image, 5, sizeof(int), &K);
        clSetKernelArg(g_kf_wn_write_image, 6, sizeof(int), &K_pad);
        size_t gws_w[2] = {desc.image_width, desc.image_height};
        nnopt_enqueue_profiled(queue, g_kf_wn_write_image, 2, nullptr, gws_w, nullptr, 0, nullptr, nullptr);
        gen_release(scl);
        it->second.Wimg = img;
        it->second.K_pad = K_pad;
    }
    if (C_out_o) *C_out_o = it->second.C_out;
    if (C_in_o)  *C_in_o  = it->second.C_in;
    if (K_o)     *K_o     = it->second.K;
    if (K_pad_o) *K_pad_o = it->second.K_pad;
    return it->second.Wimg;
}

// Packed-int8 weights for the dot8 conv path: wq [oc][k][Cq] uints + per-oc
// scale + per-oc Σw_q. Built once per prefix from the fp16 weight image.
struct I8WCache { cl_mem wq = nullptr; cl_mem sw = nullptr; cl_mem wsum = nullptr; };
static std::unordered_map<std::string, I8WCache> g_i8_wcache;

static bool get_wn_conv_weight_dot8(OpenCLContext& cl_ctx, Weights& weights, cl_command_queue queue,
                                    const std::string& prefix, cl_mem Wimg,
                                    int C_out, int C_in, int K, int K_pad,
                                    I8WCache* out) {
    auto it = g_i8_wcache.find(prefix);
    if (it != g_i8_wcache.end()) { *out = it->second; return it->second.wq != nullptr; }
    I8WCache e;
    cl_int err = CL_SUCCESS;
    size_t wq_bytes = (size_t)C_out * K * (C_in / 4) * sizeof(cl_uint);
    e.wq   = clCreateBuffer(cl_ctx.context(), CL_MEM_READ_WRITE, wq_bytes, nullptr, &err);
    e.sw   = clCreateBuffer(cl_ctx.context(), CL_MEM_READ_WRITE, (size_t)C_out * sizeof(float), nullptr, &err);
    e.wsum = clCreateBuffer(cl_ctx.context(), CL_MEM_READ_WRITE, (size_t)C_out * sizeof(cl_int), nullptr, &err);
    if (!e.wq || !e.sw || !e.wsum) {
        g_i8_wcache[prefix] = I8WCache{};   // cache the failure, don't retry
        return false;
    }
    clSetKernelArg(g_ki8_wpack, 0, sizeof(cl_mem), &Wimg);
    clSetKernelArg(g_ki8_wpack, 1, sizeof(cl_mem), &e.wq);
    clSetKernelArg(g_ki8_wpack, 2, sizeof(cl_mem), &e.sw);
    clSetKernelArg(g_ki8_wpack, 3, sizeof(cl_mem), &e.wsum);
    clSetKernelArg(g_ki8_wpack, 4, sizeof(int), &C_in);
    clSetKernelArg(g_ki8_wpack, 5, sizeof(int), &K);
    clSetKernelArg(g_ki8_wpack, 6, sizeof(int), &K_pad);
    size_t gws = (size_t)C_out * 64;
    size_t lws = 64;
    if (nnopt_enqueue_profiled(queue, g_ki8_wpack, 1, nullptr, &gws, &lws, 0, nullptr, nullptr) != CL_SUCCESS) {
        g_i8_wcache[prefix] = I8WCache{};
        return false;
    }
    g_i8_wcache[prefix] = e;
    *out = e;
    return true;
}

// All-half conv (half in/W/acc/out). Resblock shapes only.
static int conv1d_wn_hh(OpenCLContext& cl_ctx, Weights& weights, cl_command_queue queue,
                        const std::string& prefix, cl_mem in_f16, cl_mem out_f16,
                        int L_in, int stride, int padding, int dilation, int groups,
                        int* L_out_o, int* C_out_o) {
    int C_out, C_in, K;
    cl_mem Wh = get_wn_conv_weight_f16(cl_ctx, weights, queue, prefix, &C_out, &C_in, &K);
    if (!Wh) return -1;
    cl_mem bias_fp16 = weights.get_buffer(prefix + ".bias", /*optional=*/true);
    int L_out = (L_in + 2*padding - dilation*(K-1) - 1) / stride + 1;
    if (L_out_o) *L_out_o = L_out;
    if (C_out_o) *C_out_o = C_out;
    if (!(stride == 1 && groups == 1 && C_out >= 8)) {
        NNOPT_ERROR_FMT("conv1d_wn_hh %s: unsupported shape", prefix.c_str());
        return -1;
    }
    int has_bias = bias_fp16 ? 1 : 0;
    cl_mem bias_arg = bias_fp16 ? bias_fp16 : Wh;
    // int8 dot8 path (see k_i8_src). Default ON since 2026-06-06 (conv 1.61×,
    // cos 0.9940, A/B ×2 + ear check passed); NNOPT_DOT8=0 reverts to the
    // fp16 texture path. Falls through on any setup failure (or no extension).
    static int use_dot8 = -1;
    if (use_dot8 == -1) {
        const char* e = std::getenv("NNOPT_DOT8");
        use_dot8 = (e && e[0] == '0') ? 0 : 1;
    }
    if (use_dot8 && (C_in % 16) == 0 && ensure_built_i8(cl_ctx)) {
        int Kp = 0, Co2, Ci2, K2;
        cl_mem Wimg = get_wn_conv_weight_image(cl_ctx, weights, queue, prefix, &Co2, &Ci2, &K2, &Kp);
        I8WCache wc;
        if (Wimg && get_wn_conv_weight_dot8(cl_ctx, weights, queue, prefix, Wimg,
                                            C_out, C_in, K, Kp, &wc)) {
            int Cq = C_in / 4;
            // +4 rows slack: the 4-ol tile tail reads past L_out-1; quant fills
            // every allocated row (zp for OOB), so those loads are safe.
            int rows_alloc = L_in + 2 * padding + 4;
            cl_mem part = alloc_fp32(cl_ctx, 64);
            cl_mem sa   = alloc_fp32(cl_ctx, 2);
            cl_mem xq   = g_gen_arena_active
                ? g_gen_arena.get(cl_ctx.context(), (size_t)rows_alloc * Cq * sizeof(cl_uint))
                : clCreateBuffer(cl_ctx.context(), CL_MEM_READ_WRITE,
                                 (size_t)rows_alloc * Cq * sizeof(cl_uint), nullptr, nullptr);
            if (part && sa && xq) {
                int N = C_in * L_in;
                clSetKernelArg(g_ki8_absmax_part, 0, sizeof(cl_mem), &in_f16);
                clSetKernelArg(g_ki8_absmax_part, 1, sizeof(int), &N);
                clSetKernelArg(g_ki8_absmax_part, 2, sizeof(cl_mem), &part);
                size_t gws_a = 64 * 256, lws_a = 256;
                nnopt_enqueue_profiled(queue, g_ki8_absmax_part, 1, nullptr, &gws_a, &lws_a, 0, nullptr, nullptr);
                int NP = 64;
                clSetKernelArg(g_ki8_absmax_fin, 0, sizeof(cl_mem), &part);
                clSetKernelArg(g_ki8_absmax_fin, 1, sizeof(int), &NP);
                clSetKernelArg(g_ki8_absmax_fin, 2, sizeof(cl_mem), &sa);
                size_t gws_f = 64, lws_f = 64;
                nnopt_enqueue_profiled(queue, g_ki8_absmax_fin, 1, nullptr, &gws_f, &lws_f, 0, nullptr, nullptr);
                clSetKernelArg(g_ki8_quant_lc, 0, sizeof(cl_mem), &in_f16);
                clSetKernelArg(g_ki8_quant_lc, 1, sizeof(cl_mem), &sa);
                clSetKernelArg(g_ki8_quant_lc, 2, sizeof(cl_mem), &xq);
                clSetKernelArg(g_ki8_quant_lc, 3, sizeof(int), &C_in);
                clSetKernelArg(g_ki8_quant_lc, 4, sizeof(int), &L_in);
                clSetKernelArg(g_ki8_quant_lc, 5, sizeof(int), &padding);
                clSetKernelArg(g_ki8_quant_lc, 6, sizeof(int), &rows_alloc);
                // 32x32 tiles, one 256-lane WG per tile (8 rows-of-32 sweeps).
                size_t gws_q[2] = {(size_t)((rows_alloc + 31) / 32) * 256,
                                   (size_t)((Cq + 31) / 32)};
                size_t lws_q[2] = {256, 1};
                nnopt_enqueue_profiled(queue, g_ki8_quant_lc, 2, nullptr, gws_q, lws_q, 0, nullptr, nullptr);
                clSetKernelArg(g_ki8_conv, 0, sizeof(cl_mem), &xq);
                clSetKernelArg(g_ki8_conv, 1, sizeof(cl_mem), &wc.wq);
                clSetKernelArg(g_ki8_conv, 2, sizeof(cl_mem), &wc.sw);
                clSetKernelArg(g_ki8_conv, 3, sizeof(cl_mem), &wc.wsum);
                clSetKernelArg(g_ki8_conv, 4, sizeof(cl_mem), &sa);
                clSetKernelArg(g_ki8_conv, 5, sizeof(cl_mem), &out_f16);
                clSetKernelArg(g_ki8_conv, 6, sizeof(cl_mem), &bias_arg);
                clSetKernelArg(g_ki8_conv, 7, sizeof(int), &has_bias);
                clSetKernelArg(g_ki8_conv, 8, sizeof(int), &C_in);
                clSetKernelArg(g_ki8_conv, 9, sizeof(int), &C_out);
                clSetKernelArg(g_ki8_conv, 10, sizeof(int), &L_out);
                clSetKernelArg(g_ki8_conv, 11, sizeof(int), &K);
                clSetKernelArg(g_ki8_conv, 12, sizeof(int), &dilation);
                // WG geometry: start from the fp16 winner's shape (4,32);
                // NNOPT_I8_L0/I8_LT sweep it.
                static int s_i8_l0 = -1, s_i8_lt = -1;
                if (s_i8_lt == -1) {
                    const char* el = std::getenv("NNOPT_I8_LT");
                    s_i8_lt = (el && atoi(el) > 0) ? atoi(el) : 32;
                    const char* e0 = std::getenv("NNOPT_I8_L0");
                    s_i8_l0 = (e0 && atoi(e0) > 0) ? atoi(e0) : 4;
                }
                int oct = nnopt_i8_oct();
                size_t tiles_l = (size_t)((L_out + 3) / 4);
                size_t tiles_l_pad = ((tiles_l + s_i8_lt - 1) / s_i8_lt) * s_i8_lt;
                size_t oc8 = (size_t)((C_out + oct - 1) / oct);
                size_t oc8_pad = ((oc8 + s_i8_l0 - 1) / s_i8_l0) * s_i8_l0;
                size_t gws_c[2] = {oc8_pad, tiles_l_pad};
                size_t lws_c[2] = {(size_t)s_i8_l0, (size_t)s_i8_lt};
                cl_int cerr = nnopt_enqueue_profiled(queue, g_ki8_conv, 2, nullptr, gws_c, lws_c, 0, nullptr, nullptr);
                gen_release(part); gen_release(sa); gen_release(xq);
                if (cerr == CL_SUCCESS) return 0;
                NNOPT_ERROR_FMT("conv1d_i8 %s: dispatch failed (%d), falling back", prefix.c_str(), (int)cerr);
            } else {
                gen_release(part); gen_release(sa); gen_release(xq);
            }
        }
    }
    static int use_texw = -1;
    if (use_texw == -1) {
        const char* e = std::getenv("NNOPT_TEXW");
        use_texw = (e && e[0] == '0') ? 0 : 1;
    }
    if (use_texw) {
        int Kp = 0, Co2, Ci2, K2;
        cl_mem Wimg = get_wn_conv_weight_image(cl_ctx, weights, queue, prefix, &Co2, &Ci2, &K2, &Kp);
        if (Wimg) {
            static int use_ht48 = -1;
            if (use_ht48 == -1) {
                const char* e = std::getenv("NNOPT_HT48");
                use_ht48 = (e && e[0] == '1') ? 1 : 0;  // default OFF: half8 spills in any tile shape (5.47s vs 3.93s)
            }
            if (use_ht48 && (C_out % 4) == 0) {
                clSetKernelArg(g_kf_conv1d_ht48, 0, sizeof(cl_mem), &in_f16);
                clSetKernelArg(g_kf_conv1d_ht48, 1, sizeof(cl_mem), &Wimg);
                clSetKernelArg(g_kf_conv1d_ht48, 2, sizeof(cl_mem), &out_f16);
                clSetKernelArg(g_kf_conv1d_ht48, 3, sizeof(cl_mem), &bias_arg);
                clSetKernelArg(g_kf_conv1d_ht48, 4, sizeof(int), &has_bias);
                clSetKernelArg(g_kf_conv1d_ht48, 5, sizeof(int), &C_in);
                clSetKernelArg(g_kf_conv1d_ht48, 6, sizeof(int), &C_out);
                clSetKernelArg(g_kf_conv1d_ht48, 7, sizeof(int), &L_in);
                clSetKernelArg(g_kf_conv1d_ht48, 8, sizeof(int), &L_out);
                clSetKernelArg(g_kf_conv1d_ht48, 9, sizeof(int), &Kp);
                clSetKernelArg(g_kf_conv1d_ht48, 10, sizeof(int), &padding);
                clSetKernelArg(g_kf_conv1d_ht48, 11, sizeof(int), &dilation);
                const int local_t48 = 128;
                size_t tiles48 = (size_t)((L_out + 7) / 8);
                size_t tiles48_padded = ((tiles48 + local_t48 - 1) / local_t48) * local_t48;
                size_t gws48[2] = {(size_t)((C_out + 3) / 4), tiles48_padded};
                size_t lws48[2] = {1, (size_t)local_t48};
                nnopt_enqueue_profiled(queue, g_kf_conv1d_ht48, 2, nullptr, gws48, lws48, 0, nullptr, nullptr);
                return 0;
            }
            static int use_ht8 = -1;
            if (use_ht8 == -1) {
                const char* e = std::getenv("NNOPT_HT8");
                use_ht8 = (e && e[0] == '1') ? 1 : 0;  // default OFF: 8x half8 accs spill (5.97s vs 4.30s)
            }
            if (use_ht8) {
                clSetKernelArg(g_kf_conv1d_ht8, 0, sizeof(cl_mem), &in_f16);
                clSetKernelArg(g_kf_conv1d_ht8, 1, sizeof(cl_mem), &Wimg);
                clSetKernelArg(g_kf_conv1d_ht8, 2, sizeof(cl_mem), &out_f16);
                clSetKernelArg(g_kf_conv1d_ht8, 3, sizeof(cl_mem), &bias_arg);
                clSetKernelArg(g_kf_conv1d_ht8, 4, sizeof(int), &has_bias);
                clSetKernelArg(g_kf_conv1d_ht8, 5, sizeof(int), &C_in);
                clSetKernelArg(g_kf_conv1d_ht8, 6, sizeof(int), &C_out);
                clSetKernelArg(g_kf_conv1d_ht8, 7, sizeof(int), &L_in);
                clSetKernelArg(g_kf_conv1d_ht8, 8, sizeof(int), &L_out);
                clSetKernelArg(g_kf_conv1d_ht8, 9, sizeof(int), &Kp);
                clSetKernelArg(g_kf_conv1d_ht8, 10, sizeof(int), &padding);
                clSetKernelArg(g_kf_conv1d_ht8, 11, sizeof(int), &dilation);
                const int local_t8 = 128;
                size_t tiles8 = (size_t)((L_out + 7) / 8);
                size_t tiles8_padded = ((tiles8 + local_t8 - 1) / local_t8) * local_t8;
                size_t gws8[2] = {(size_t)((C_out + 7) / 8), tiles8_padded};
                size_t lws8[2] = {1, (size_t)local_t8};
                nnopt_enqueue_profiled(queue, g_kf_conv1d_ht8, 2, nullptr, gws8, lws8, 0, nullptr, nullptr);
                return 0;
            }
            clSetKernelArg(g_kf_conv1d_ht, 0, sizeof(cl_mem), &in_f16);
            clSetKernelArg(g_kf_conv1d_ht, 1, sizeof(cl_mem), &Wimg);
            clSetKernelArg(g_kf_conv1d_ht, 2, sizeof(cl_mem), &out_f16);
            clSetKernelArg(g_kf_conv1d_ht, 3, sizeof(cl_mem), &bias_arg);
            clSetKernelArg(g_kf_conv1d_ht, 4, sizeof(int), &has_bias);
            clSetKernelArg(g_kf_conv1d_ht, 5, sizeof(int), &C_in);
            clSetKernelArg(g_kf_conv1d_ht, 6, sizeof(int), &C_out);
            clSetKernelArg(g_kf_conv1d_ht, 7, sizeof(int), &L_in);
            clSetKernelArg(g_kf_conv1d_ht, 8, sizeof(int), &L_out);
            clSetKernelArg(g_kf_conv1d_ht, 9, sizeof(int), &Kp);
            clSetKernelArg(g_kf_conv1d_ht, 10, sizeof(int), &padding);
            clSetKernelArg(g_kf_conv1d_ht, 11, sizeof(int), &dilation);
            // Workgroup geometry sweep knobs (§6.1.5: the (1,128) shape
            // predates the texture rewrite). NNOPT_HT_LT = lws along the
            // ol-tile dim (default 128); NNOPT_HT_L0 = lws along the oc-octet
            // dim (default 1).
            static int s_ht_lt = -1, s_ht_l0 = -1;
            if (s_ht_lt == -1) {
                // Defaults from the 2026-06-06 sweep: (4,32)+full wave = 1068 ms
                // vs 1556 ms at the old (1,128)+default — a full-wave WG needs
                // >=128 work items or the wave starves ((2,32) regressed 1.7x).
                const char* e = std::getenv("NNOPT_HT_LT");
                s_ht_lt = (e && atoi(e) > 0) ? atoi(e) : 32;
                const char* e0 = std::getenv("NNOPT_HT_L0");
                s_ht_l0 = (e0 && atoi(e0) > 0) ? atoi(e0) : 4;
            }
            const int local_t = s_ht_lt;
            size_t tiles_l = (size_t)((L_out + 3) / 4);
            size_t tiles_l_padded = ((tiles_l + local_t - 1) / local_t) * local_t;
            size_t oc8 = (size_t)((C_out + 7) / 8);
            size_t oc8_padded = ((oc8 + s_ht_l0 - 1) / s_ht_l0) * s_ht_l0;
            size_t gws[2] = {oc8_padded, tiles_l_padded};
            size_t lws[2] = {(size_t)s_ht_l0, (size_t)local_t};
            cl_int hterr = nnopt_enqueue_profiled(queue, g_kf_conv1d_ht, 2, nullptr, gws, lws, 0, nullptr, nullptr);
            if (hterr != CL_SUCCESS) {
                NNOPT_ERROR_FMT("conv1d_ht %s: dispatch failed (err=%d) lws=(%d,%d)",
                                prefix.c_str(), (int)hterr, s_ht_l0, local_t);
                return -1;
            }
            return 0;
        }
    }
    static int use_lds = -1;
    if (use_lds == -1) {
        const char* e = std::getenv("NNOPT_CONV_LDS");
        use_lds = (e && e[0] == '1') ? 1 : 0;  // default OFF: 9.6s vs 6.27s — Adreno 620 LDS path loses again
    }
    if (use_lds && (C_out % 4) == 0 && 4 * C_in * K <= 8192) {
        clSetKernelArg(g_kf_conv1d_hh_lds, 0, sizeof(cl_mem), &in_f16);
        clSetKernelArg(g_kf_conv1d_hh_lds, 1, sizeof(cl_mem), &Wh);
        clSetKernelArg(g_kf_conv1d_hh_lds, 2, sizeof(cl_mem), &out_f16);
        clSetKernelArg(g_kf_conv1d_hh_lds, 3, sizeof(cl_mem), &bias_arg);
        clSetKernelArg(g_kf_conv1d_hh_lds, 4, sizeof(int), &has_bias);
        clSetKernelArg(g_kf_conv1d_hh_lds, 5, sizeof(int), &C_in);
        clSetKernelArg(g_kf_conv1d_hh_lds, 6, sizeof(int), &C_out);
        clSetKernelArg(g_kf_conv1d_hh_lds, 7, sizeof(int), &L_in);
        clSetKernelArg(g_kf_conv1d_hh_lds, 8, sizeof(int), &L_out);
        clSetKernelArg(g_kf_conv1d_hh_lds, 9, sizeof(int), &K);
        clSetKernelArg(g_kf_conv1d_hh_lds, 10, sizeof(int), &padding);
        clSetKernelArg(g_kf_conv1d_hh_lds, 11, sizeof(int), &dilation);
        size_t tiles_l = (size_t)((L_out + 3) / 4);
        size_t tile_groups = (tiles_l + 127) / 128;
        size_t gws[2] = {(size_t)(C_out / 4), tile_groups * 128};
        size_t lws[2] = {1, 128};
        nnopt_enqueue_profiled(queue, g_kf_conv1d_hh_lds, 2, nullptr, gws, lws, 0, nullptr, nullptr);
        return 0;
    }
    clSetKernelArg(g_kf_conv1d_hh, 0, sizeof(cl_mem), &in_f16);
    clSetKernelArg(g_kf_conv1d_hh, 1, sizeof(cl_mem), &Wh);
    clSetKernelArg(g_kf_conv1d_hh, 2, sizeof(cl_mem), &out_f16);
    clSetKernelArg(g_kf_conv1d_hh, 3, sizeof(cl_mem), &bias_arg);
    clSetKernelArg(g_kf_conv1d_hh, 4, sizeof(int), &has_bias);
    clSetKernelArg(g_kf_conv1d_hh, 5, sizeof(int), &C_in);
    clSetKernelArg(g_kf_conv1d_hh, 6, sizeof(int), &C_out);
    clSetKernelArg(g_kf_conv1d_hh, 7, sizeof(int), &L_in);
    clSetKernelArg(g_kf_conv1d_hh, 8, sizeof(int), &L_out);
    clSetKernelArg(g_kf_conv1d_hh, 9, sizeof(int), &K);
    clSetKernelArg(g_kf_conv1d_hh, 10, sizeof(int), &padding);
    clSetKernelArg(g_kf_conv1d_hh, 11, sizeof(int), &dilation);
    const int local_t = 256;
    size_t tiles_l = (size_t)((L_out + 3) / 4);
    size_t tiles_l_padded = ((tiles_l + local_t - 1) / local_t) * local_t;
    size_t gws[2] = {(size_t)((C_out + 7) / 8), tiles_l_padded};
    size_t lws[2] = {1, (size_t)local_t};
    nnopt_enqueue_profiled(queue, g_kf_conv1d_hh, 2, nullptr, gws, lws, 0, nullptr, nullptr);
    return 0;
}

// Fused adain+snake reading half conv output, writing half conv input.
static int apply_adain_snake_h2h(OpenCLContext& cl_ctx, Weights& weights, cl_command_queue queue,
                                 const std::string& adain_prefix, const std::string& alpha_key,
                                 cl_mem x_f16, cl_mem y_f16, cl_mem style_fp16, int C, int T) {
    cl_mem fc_w = weights.get_buffer(adain_prefix + ".fc.weight");
    cl_mem fc_b = weights.get_buffer(adain_prefix + ".fc.bias");
    cl_mem alpha = weights.get_buffer(alpha_key);
    if (!fc_w || !fc_b || !alpha) { NNOPT_ERROR_FMT("adain_snake_h2h: missing %s", adain_prefix.c_str()); return -1; }
    int two_C = 2*C, style_dim = 128;
    cl_mem h = alloc_fp32(cl_ctx, two_C);
    int has_bias = 1;
    clSetKernelArg(g_kf_linear_apply, 0, sizeof(cl_mem), &style_fp16);
    clSetKernelArg(g_kf_linear_apply, 1, sizeof(cl_mem), &fc_w);
    clSetKernelArg(g_kf_linear_apply, 2, sizeof(cl_mem), &fc_b);
    clSetKernelArg(g_kf_linear_apply, 3, sizeof(cl_mem), &h);
    clSetKernelArg(g_kf_linear_apply, 4, sizeof(int), &style_dim);
    clSetKernelArg(g_kf_linear_apply, 5, sizeof(int), &two_C);
    clSetKernelArg(g_kf_linear_apply, 6, sizeof(int), &has_bias);
    size_t gws_lin = (size_t)two_C;
    nnopt_enqueue_profiled(queue, g_kf_linear_apply, 1, nullptr, &gws_lin, nullptr, 0, nullptr, nullptr);
    cl_mem gamma = alloc_fp32(cl_ctx, C);
    cl_mem beta  = alloc_fp32(cl_ctx, C);
    clSetKernelArg(g_kf_split2, 0, sizeof(cl_mem), &h);
    clSetKernelArg(g_kf_split2, 1, sizeof(cl_mem), &gamma);
    clSetKernelArg(g_kf_split2, 2, sizeof(cl_mem), &beta);
    clSetKernelArg(g_kf_split2, 3, sizeof(int), &C);
    size_t gws_sp = (size_t)C;
    nnopt_enqueue_profiled(queue, g_kf_split2, 1, nullptr, &gws_sp, nullptr, 0, nullptr, nullptr);
    float eps = 1e-5f;
    clSetKernelArg(g_kf_in_adain_snake_h2h, 0, sizeof(cl_mem), &x_f16);
    clSetKernelArg(g_kf_in_adain_snake_h2h, 1, sizeof(cl_mem), &y_f16);
    clSetKernelArg(g_kf_in_adain_snake_h2h, 2, sizeof(cl_mem), &gamma);
    clSetKernelArg(g_kf_in_adain_snake_h2h, 3, sizeof(cl_mem), &beta);
    clSetKernelArg(g_kf_in_adain_snake_h2h, 4, sizeof(cl_mem), &alpha);
    clSetKernelArg(g_kf_in_adain_snake_h2h, 5, sizeof(int), &C);
    clSetKernelArg(g_kf_in_adain_snake_h2h, 6, sizeof(int), &T);
    clSetKernelArg(g_kf_in_adain_snake_h2h, 7, sizeof(float), &eps);
    size_t gws_f = (size_t)C * 64;
    size_t lws_f = 64;
    nnopt_enqueue_profiled(queue, g_kf_in_adain_snake_h2h, 1, nullptr, &gws_f, &lws_f, 0, nullptr, nullptr);
    gen_release(h);
    gen_release(gamma);
    gen_release(beta);
    return 0;
}

// Conv with fp16 input scratch + fp32 weights/accum/output. Mirrors the t8x4
// dispatch of conv1d_wn_fp32 (the resblock convs always satisfy the fast-path
// conditions: stride 1, groups 1, C_out >= 8).
static int conv1d_wn_h16in(OpenCLContext& cl_ctx, Weights& weights, cl_command_queue queue,
                           const std::string& prefix, cl_mem in_f16, cl_mem out_f32,
                           int L_in, int stride, int padding, int dilation, int groups,
                           int* L_out_o, int* C_out_o) {
    int C_out, C_in, K;
    cl_mem W = get_wn_conv_weight_fp32(cl_ctx, weights, queue, prefix, &C_out, &C_in, &K);
    if (!W) return -1;
    cl_mem bias_fp16 = weights.get_buffer(prefix + ".bias", /*optional=*/true);
    int L_out = (L_in + 2*padding - dilation*(K-1) - 1) / stride + 1;
    if (L_out_o) *L_out_o = L_out;
    if (C_out_o) *C_out_o = C_out;
    if (!(stride == 1 && groups == 1 && C_out >= 8)) {
        NNOPT_ERROR_FMT("conv1d_wn_h16in %s: unsupported shape", prefix.c_str());
        return -1;
    }
    cl_mem bias_f32 = nullptr;
    if (bias_fp16) {
        auto itb = g_wn_cache_fp32.find(prefix);
        if (itb != g_wn_cache_fp32.end() && itb->second.bias_f32) {
            bias_f32 = itb->second.bias_f32;
        } else {
            bias_f32 = alloc_fp32(cl_ctx, C_out);
            clSetKernelArg(g_kf_f16_to_f32, 0, sizeof(cl_mem), &bias_fp16);
            clSetKernelArg(g_kf_f16_to_f32, 1, sizeof(cl_mem), &bias_f32);
            clSetKernelArg(g_kf_f16_to_f32, 2, sizeof(int), &C_out);
            size_t gws_b = (size_t)C_out;
            nnopt_enqueue_profiled(queue, g_kf_f16_to_f32, 1, nullptr, &gws_b, nullptr, 0, nullptr, nullptr);
            if (itb != g_wn_cache_fp32.end()) itb->second.bias_f32 = bias_f32;
        }
    }
    int has_bias = bias_f32 ? 1 : 0;
    cl_mem bias_arg = bias_f32 ? bias_f32 : W;
    clSetKernelArg(g_kf_conv1d_h8x4, 0, sizeof(cl_mem), &in_f16);
    clSetKernelArg(g_kf_conv1d_h8x4, 1, sizeof(cl_mem), &W);
    clSetKernelArg(g_kf_conv1d_h8x4, 2, sizeof(cl_mem), &out_f32);
    clSetKernelArg(g_kf_conv1d_h8x4, 3, sizeof(cl_mem), &bias_arg);
    clSetKernelArg(g_kf_conv1d_h8x4, 4, sizeof(int), &has_bias);
    clSetKernelArg(g_kf_conv1d_h8x4, 5, sizeof(int), &C_in);
    clSetKernelArg(g_kf_conv1d_h8x4, 6, sizeof(int), &C_out);
    clSetKernelArg(g_kf_conv1d_h8x4, 7, sizeof(int), &L_in);
    clSetKernelArg(g_kf_conv1d_h8x4, 8, sizeof(int), &L_out);
    clSetKernelArg(g_kf_conv1d_h8x4, 9, sizeof(int), &K);
    clSetKernelArg(g_kf_conv1d_h8x4, 10, sizeof(int), &padding);
    clSetKernelArg(g_kf_conv1d_h8x4, 11, sizeof(int), &dilation);
    const int local_t = 64;
    size_t tiles_l = (size_t)((L_out + 3) / 4);
    size_t tiles_l_padded = ((tiles_l + local_t - 1) / local_t) * local_t;
    size_t gws[2] = {(size_t)((C_out + 7) / 8), tiles_l_padded};
    size_t lws[2] = {1, (size_t)local_t};
    nnopt_enqueue_profiled(queue, g_kf_conv1d_h8x4, 2, nullptr, gws, lws, 0, nullptr, nullptr);
    return 0;
}

// Public: apply one AdaINResBlock1 in fp32 storage.
// x_f32 and out_f32 are fp32 buffers [C*T_in]. x is modified in-place (used as running residual).
// Actually: caller passes x_f32 as the input buffer; this function does x_f32 += residual per iteration.
// out_f32 == x_f32 is allowed.
// Helper: dump fp32 buffer with given name (only when prefix matches resblocks.3 for bisection)
static void dump_fp32(cl_command_queue queue, cl_mem buf, int N, const std::string& name, const std::string& prefix) {
    // Bisection instrument: OPT-IN ONLY (NNOPT_DUMP_RB3=1). Without the env
    // gate this ran 7 blocking multi-MB readbacks per utterance in production.
    static int dump_on = -1;
    if (dump_on == -1) {
        const char* e = std::getenv("NNOPT_DUMP_RB3");
        dump_on = (e && e[0] == '1') ? 1 : 0;
    }
    if (!dump_on) return;
    if (prefix != "decoder.module.generator.resblocks.3") return;
    // Convert fp32 to fp16 for LAYER_CHECK macro (which expects storage_t)
    // Easier: read fp32 to host and write to file directly.
    std::vector<float> host(N);
    clEnqueueReadBuffer(queue, buf, CL_TRUE, 0, sizeof(float) * N, host.data(), 0, nullptr, nullptr);
    std::string path = "layer_dumps/" + name + "_fp32.bin";
    FILE* f = fopen(path.c_str(), "wb");
    if (f) { fwrite(host.data(), sizeof(float), N, f); fclose(f); }
}

extern "C" int gen_apply_adainresblock1_fp32(OpenCLContext& cl_ctx, Weights& weights, cl_command_queue queue,
                                              const std::string& prefix,
                                              cl_mem x_f32,
                                              cl_mem ref_s_dec_fp16,  // fp16
                                              int C, int T,
                                              int kernel_size, const int* dilations) {
    if (!ensure_built_gf(cl_ctx)) return -1;
    cl_mem tmp1 = alloc_fp32(cl_ctx, C * T);
    cl_mem tmp2 = alloc_fp32(cl_ctx, C * T);
    cl_mem c1_out = alloc_fp32(cl_ctx, C * T);
    cl_mem c2_out = alloc_fp32(cl_ctx, C * T);
    static int use_h16 = -1;
    if (use_h16 == -1) {
        const char* e = std::getenv("NNOPT_H16CONV");
        use_h16 = (e && e[0] == '0') ? 0 : 1;
    }
    cl_mem tmp1_h = use_h16 ? alloc_f16(cl_ctx, (size_t)C * T) : nullptr;
    cl_mem tmp2_h = use_h16 ? alloc_f16(cl_ctx, (size_t)C * T) : nullptr;
    // Full half-math chain (half conv ALU; fp32 residual + fp32 norm stats).
    static int use_hmath = -1;
    if (use_hmath == -1) {
        const char* e = std::getenv("NNOPT_HMATH");
        use_hmath = (e && e[0] == '0') ? 0 : 1;
    }
    cl_mem c1_h = (use_h16 && use_hmath) ? alloc_f16(cl_ctx, (size_t)C * T) : nullptr;
    cl_mem c2_h = (use_h16 && use_hmath) ? alloc_f16(cl_ctx, (size_t)C * T) : nullptr;

    // Dump the INPUT x for bisection
    dump_fp32(queue, x_f32, C * T, "rb3_input", prefix);

    for (int i = 0; i < 3; ++i) {
        std::string si = std::to_string(i);
        // adain1[i] + snake fused: x → tmp1 (fp16 conv scratch when enabled)
        int dil = dilations[i];
        int pad = (kernel_size * dil - dil) / 2;
        int Lc1, Cc1;
        if (use_h16 && use_hmath) {
            if (apply_adain_snake_f16out(cl_ctx, weights, queue, prefix + ".adain1." + si,
                                         prefix + ".alpha1." + si, x_f32, tmp1_h, ref_s_dec_fp16, C, T) != 0) return -1;
            if (conv1d_wn_hh(cl_ctx, weights, queue, prefix + ".convs1." + si, tmp1_h, c1_h, T, 1, pad, dil, 1, &Lc1, &Cc1) != 0) return -1;
        } else if (use_h16) {
            if (apply_adain_snake_f16out(cl_ctx, weights, queue, prefix + ".adain1." + si,
                                         prefix + ".alpha1." + si, x_f32, tmp1_h, ref_s_dec_fp16, C, T) != 0) return -1;
            if (conv1d_wn_h16in(cl_ctx, weights, queue, prefix + ".convs1." + si, tmp1_h, c1_out, T, 1, pad, dil, 1, &Lc1, &Cc1) != 0) return -1;
        } else {
        if (apply_adain_snake_fp32(cl_ctx, weights, queue, prefix + ".adain1." + si,
                                   prefix + ".alpha1." + si, x_f32, tmp1, ref_s_dec_fp16, C, T) != 0) return -1;
        dump_fp32(queue, tmp1, C * T, "rb3_iter" + si + "_after_snake1", prefix);
        if (conv1d_wn_fp32(cl_ctx, weights, queue, prefix + ".convs1." + si, tmp1, c1_out, T, 1, pad, dil, 1, &Lc1, &Cc1) != 0) return -1;
        }
        dump_fp32(queue, c1_out, C * Lc1, "rb3_iter" + si + "_after_conv1", prefix);
        // adain2[i] + snake fused: c1_out → tmp2
        int pad2 = (kernel_size - 1) / 2;
        int Lc2, Cc2;
        if (use_h16 && use_hmath) {
            if (apply_adain_snake_h2h(cl_ctx, weights, queue, prefix + ".adain2." + si,
                                      prefix + ".alpha2." + si, c1_h, tmp2_h, ref_s_dec_fp16, C, Lc1) != 0) return -1;
            if (conv1d_wn_hh(cl_ctx, weights, queue, prefix + ".convs2." + si, tmp2_h, c2_h, Lc1, 1, pad2, 1, 1, &Lc2, &Cc2) != 0) return -1;
        } else if (use_h16) {
            if (apply_adain_snake_f16out(cl_ctx, weights, queue, prefix + ".adain2." + si,
                                         prefix + ".alpha2." + si, c1_out, tmp2_h, ref_s_dec_fp16, C, Lc1) != 0) return -1;
            if (conv1d_wn_h16in(cl_ctx, weights, queue, prefix + ".convs2." + si, tmp2_h, c2_out, Lc1, 1, pad2, 1, 1, &Lc2, &Cc2) != 0) return -1;
        } else {
        if (apply_adain_snake_fp32(cl_ctx, weights, queue, prefix + ".adain2." + si,
                                   prefix + ".alpha2." + si, c1_out, tmp2, ref_s_dec_fp16, C, Lc1) != 0) return -1;
        dump_fp32(queue, tmp2, C * Lc1, "rb3_iter" + si + "_after_snake2", prefix);
        if (conv1d_wn_fp32(cl_ctx, weights, queue, prefix + ".convs2." + si, tmp2, c2_out, Lc1, 1, pad2, 1, 1, &Lc2, &Cc2) != 0) return -1;
        }
        dump_fp32(queue, c2_out, C * Lc2, "rb3_iter" + si + "_after_conv2", prefix);
        // x_f32 += c2 (half or fp32 source)
        int N = C * T;
        if (use_h16 && use_hmath) {
            clSetKernelArg(g_kf_add_from_h, 0, sizeof(cl_mem), &x_f32);
            clSetKernelArg(g_kf_add_from_h, 1, sizeof(cl_mem), &c2_h);
            clSetKernelArg(g_kf_add_from_h, 2, sizeof(int), &N);
            size_t gws = (size_t)N;
            nnopt_enqueue_profiled(queue, g_kf_add_from_h, 1, nullptr, &gws, nullptr, 0, nullptr, nullptr);
        } else {
        clSetKernelArg(g_kf_add, 0, sizeof(cl_mem), &x_f32);
        clSetKernelArg(g_kf_add, 1, sizeof(cl_mem), &c2_out);
        clSetKernelArg(g_kf_add, 2, sizeof(int), &N);
        size_t gws = (size_t)N;
        nnopt_enqueue_profiled(queue, g_kf_add, 1, nullptr, &gws, nullptr, 0, nullptr, nullptr);
        }
        dump_fp32(queue, x_f32, C * T, "rb3_iter" + si + "_after_residual", prefix);
    }
    gen_release(tmp1);
    gen_release(tmp2);
    if (tmp1_h) gen_release(tmp1_h);
    if (tmp2_h) gen_release(tmp2_h);
    if (c1_h) gen_release(c1_h);
    if (c2_h) gen_release(c2_h);
    gen_release(c1_out);
    gen_release(c2_out);
    return 0;
}

// Average 3 fp32 buffers
extern "C" int gen_avg_three_fp32(OpenCLContext& cl_ctx, cl_command_queue queue,
                                   cl_mem a, cl_mem b, cl_mem c, cl_mem y, int N) {
    if (!ensure_built_gf(cl_ctx)) return -1;
    clSetKernelArg(g_kf_avg3, 0, sizeof(cl_mem), &a);
    clSetKernelArg(g_kf_avg3, 1, sizeof(cl_mem), &b);
    clSetKernelArg(g_kf_avg3, 2, sizeof(cl_mem), &c);
    clSetKernelArg(g_kf_avg3, 3, sizeof(cl_mem), &y);
    clSetKernelArg(g_kf_avg3, 4, sizeof(int), &N);
    size_t gws = (size_t)N;
    return nnopt_enqueue_profiled(queue, g_kf_avg3, 1, nullptr, &gws, nullptr, 0, nullptr, nullptr);
}

// ─── Full fp32 generator on GPU ─────────────────────────────────────────

static int run1d_gws(cl_command_queue q, cl_kernel k, size_t n) {
    return nnopt_enqueue_profiled(q, k, 1, nullptr, &n, nullptr, 0, nullptr, nullptr);
}

// Reconstruct ConvTranspose1d weight_norm into fp32 buffer.
// PyTorch ConvTranspose1d weight shape: [C_in, C_out/groups, K]
static cl_mem wn_recon_convtr_fp32(OpenCLContext& cl_ctx, Weights& weights, cl_command_queue queue,
                                    const std::string& prefix, int* C_in_o, int* C_out_o, int* K_o) {
    cl_mem v = weights.get_buffer(prefix + ".weight_v");
    cl_mem g = weights.get_buffer(prefix + ".weight_g");
    if (!v || !g) return nullptr;
    auto shape = weights.get_shape(prefix + ".weight_v");
    int C_in = shape[0], C_out = shape[1], K = shape[2];
    int per_oc = C_out * K;  // For ConvTranspose, norm is per C_in row (over C_out * K)
    cl_mem W = alloc_fp32(cl_ctx, (size_t)C_in * per_oc);
    clSetKernelArg(g_kf_weightnorm, 0, sizeof(cl_mem), &v);
    clSetKernelArg(g_kf_weightnorm, 1, sizeof(cl_mem), &g);
    clSetKernelArg(g_kf_weightnorm, 2, sizeof(cl_mem), &W);
    clSetKernelArg(g_kf_weightnorm, 3, sizeof(int), &C_in);
    clSetKernelArg(g_kf_weightnorm, 4, sizeof(int), &per_oc);
    size_t gws = (size_t)C_in * 64;  // workgroup-per-row reduction
    size_t lws_wn2 = 64;
    nnopt_enqueue_profiled(queue, g_kf_weightnorm, 1, nullptr, &gws, &lws_wn2, 0, nullptr, nullptr);
    if (C_in_o) *C_in_o = C_in;
    if (C_out_o) *C_out_o = C_out;
    if (K_o) *K_o = K;
    return W;
}

// ConvTranspose1d fp32 + bias
static int convtr1d_wn_fp32(OpenCLContext& cl_ctx, Weights& weights, cl_command_queue queue,
                             const std::string& prefix, cl_mem in_f32, cl_mem out_f32,
                             int L_in, int stride, int padding, int* L_out_o, int* C_out_o) {
    auto wv_shape = weights.get_shape(prefix + ".weight_v");
    if (wv_shape.size() != 3) { NNOPT_ERROR_FMT("convtr1d_wn %s: missing weight_v", prefix.c_str()); return -1; }
    int C_in = wv_shape[0], C_out = wv_shape[1], K = wv_shape[2];
    int L_out = (L_in - 1)*stride - 2*padding + (K - 1) + 1;
    if (L_out_o) *L_out_o = L_out;
    if (C_out_o) *C_out_o = C_out;
    int groups = 1;
    int dilation = 1;

    // ── Texture path: weights through TP/L1 (NNOPT_TRTEX=0 reverts to the
    // buffer kernel for A/B). Mirrors decoder.cpp's phase-major image; the
    // image is built ONCE per prefix (also skipping the per-call weight-norm
    // reconstruction) and the bias add is fused into the kernel.
    static int use_trtex = -1;
    if (use_trtex == -1) {
        const char* e = std::getenv("NNOPT_TRTEX");
        use_trtex = (e && e[0] == '0') ? 0 : 1;
    }
    const int ntaps = (K + stride - 1) / stride;
    if (use_trtex && (C_out % 4) == 0) {
        static std::unordered_map<std::string, cl_mem> s_trimg_cache;
        cl_mem trimg = nullptr;
        auto it = s_trimg_cache.find(prefix);
        if (it != s_trimg_cache.end()) {
            trimg = it->second;
        } else {
            size_t img_w = (size_t)(stride * ntaps * (C_out / 4));
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
                    int rc_in, rc_out, rk;
                    cl_mem Wtmp = wn_recon_convtr_fp32(cl_ctx, weights, queue, prefix, &rc_in, &rc_out, &rk);
                    if (Wtmp) {
                        clSetKernelArg(g_kf_gf_convtr_pack, 0, sizeof(cl_mem), &Wtmp);
                        clSetKernelArg(g_kf_gf_convtr_pack, 1, sizeof(cl_mem), &img);
                        clSetKernelArg(g_kf_gf_convtr_pack, 2, sizeof(int), &C_in);
                        clSetKernelArg(g_kf_gf_convtr_pack, 3, sizeof(int), &C_out);
                        clSetKernelArg(g_kf_gf_convtr_pack, 4, sizeof(int), &K);
                        clSetKernelArg(g_kf_gf_convtr_pack, 5, sizeof(int), &stride);
                        clSetKernelArg(g_kf_gf_convtr_pack, 6, sizeof(int), &ntaps);
                        size_t gws_p[2] = {img_w, img_h};
                        nnopt_enqueue_profiled(queue, g_kf_gf_convtr_pack, 2, nullptr, gws_p, nullptr, 0, nullptr, nullptr);
                        clFinish(queue);  // Wtmp released below — pack must be done
                        gen_release(Wtmp);
                        trimg = img;
                    } else {
                        gen_release(img);
                    }
                }
            }
            s_trimg_cache[prefix] = trimg;  // nullptr caches the failure too
        }
        if (trimg) {
            cl_mem bias_f16 = weights.get_buffer(prefix + ".bias", /*optional=*/true);
            int has_bias = bias_f16 ? 1 : 0;
            cl_mem bias_arg = bias_f16 ? bias_f16 : trimg;  // dummy when absent
            clSetKernelArg(g_kf_convtr1d_c4x4_tex_f32, 0, sizeof(cl_mem), &in_f32);
            clSetKernelArg(g_kf_convtr1d_c4x4_tex_f32, 1, sizeof(cl_mem), &trimg);
            clSetKernelArg(g_kf_convtr1d_c4x4_tex_f32, 2, sizeof(cl_mem), &out_f32);
            clSetKernelArg(g_kf_convtr1d_c4x4_tex_f32, 3, sizeof(cl_mem), &bias_arg);
            clSetKernelArg(g_kf_convtr1d_c4x4_tex_f32, 4, sizeof(int), &has_bias);
            clSetKernelArg(g_kf_convtr1d_c4x4_tex_f32, 5, sizeof(int), &C_in);
            clSetKernelArg(g_kf_convtr1d_c4x4_tex_f32, 6, sizeof(int), &C_out);
            clSetKernelArg(g_kf_convtr1d_c4x4_tex_f32, 7, sizeof(int), &L_in);
            clSetKernelArg(g_kf_convtr1d_c4x4_tex_f32, 8, sizeof(int), &L_out);
            clSetKernelArg(g_kf_convtr1d_c4x4_tex_f32, 9, sizeof(int), &K);
            clSetKernelArg(g_kf_convtr1d_c4x4_tex_f32, 10, sizeof(int), &stride);
            clSetKernelArg(g_kf_convtr1d_c4x4_tex_f32, 11, sizeof(int), &padding);
            clSetKernelArg(g_kf_convtr1d_c4x4_tex_f32, 12, sizeof(int), &ntaps);
            const int local_q = 64;
            int n_q = (L_out - 1 + padding) / stride + 1;
            size_t tiles_q = (size_t)((n_q + 3) / 4);
            size_t tiles_q_padded = ((tiles_q + local_q - 1) / local_q) * local_q;
            size_t gws_t[3] = {(size_t)(C_out / 4), tiles_q_padded, (size_t)stride};
            size_t lws_t[3] = {1, (size_t)local_q, 1};
            cl_int kerr = nnopt_enqueue_profiled(queue, g_kf_convtr1d_c4x4_tex_f32, 3, nullptr, gws_t, lws_t, 0, nullptr, nullptr);
            if (kerr == CL_SUCCESS) return 0;
            NNOPT_ERROR_FMT("convtr1d_wn %s: tex dispatch failed (err=%d) — falling back to buffer kernel",
                            prefix.c_str(), (int)kerr);
        }
    }

    cl_mem W = wn_recon_convtr_fp32(cl_ctx, weights, queue, prefix, &C_in, &C_out, &K);
    if (!W) return -1;
    // Use Cout-tile=4 fast path if C_out is divisible by 4 (all our convtr targets are).
    // LOCAL_T tunes how many L outputs per workgroup share weights via L2.
    bool use_c4 = (C_out % 4 == 0);
    cl_kernel kk = use_c4 ? g_kf_convtr1d_c4 : g_kf_convtr1d;
    clSetKernelArg(kk, 0, sizeof(cl_mem), &in_f32);
    clSetKernelArg(kk, 1, sizeof(cl_mem), &W);
    clSetKernelArg(kk, 2, sizeof(cl_mem), &out_f32);
    clSetKernelArg(kk, 3, sizeof(int), &C_in);
    clSetKernelArg(kk, 4, sizeof(int), &C_out);
    clSetKernelArg(kk, 5, sizeof(int), &L_in);
    clSetKernelArg(kk, 6, sizeof(int), &L_out);
    clSetKernelArg(kk, 7, sizeof(int), &K);
    clSetKernelArg(kk, 8, sizeof(int), &stride);
    clSetKernelArg(kk, 9, sizeof(int), &padding);
    clSetKernelArg(kk, 10, sizeof(int), &dilation);
    clSetKernelArg(kk, 11, sizeof(int), &groups);
    const int LOCAL_T = 384;
    size_t gws[2];
    size_t lws[2] = {1, (size_t)LOCAL_T};
    if (use_c4) {
        size_t L_tiles = (size_t)((L_out + LOCAL_T - 1) / LOCAL_T);
        gws[0] = (size_t)(C_out / 4);
        gws[1] = L_tiles * LOCAL_T;
    } else {
        gws[0] = (size_t)C_out;
        gws[1] = (size_t)L_out;
    }
    cl_int kerr = use_c4
        ? nnopt_enqueue_profiled(queue, kk, 2, nullptr, gws, lws, 0, nullptr, nullptr)
        : nnopt_enqueue_profiled(queue, kk, 2, nullptr, gws, nullptr, 0, nullptr, nullptr);
    if (kerr != CL_SUCCESS) {
        NNOPT_ERROR_FMT("convtr1d_wn %s: dispatch failed (err=%d) use_c4=%d LOCAL_T=%d C_out=%d L_out=%d",
                        prefix.c_str(), (int)kerr, (int)use_c4, LOCAL_T, C_out, L_out);
        return -1;
    }
    cl_mem bias_fp16 = weights.get_buffer(prefix + ".bias", /*optional=*/true);
    if (bias_fp16) {
        cl_mem bias_fp32 = alloc_fp32(cl_ctx, C_out);
        clSetKernelArg(g_kf_f16_to_f32, 0, sizeof(cl_mem), &bias_fp16);
        clSetKernelArg(g_kf_f16_to_f32, 1, sizeof(cl_mem), &bias_fp32);
        clSetKernelArg(g_kf_f16_to_f32, 2, sizeof(int), &C_out);
        run1d_gws(queue, g_kf_f16_to_f32, (size_t)C_out);
        clSetKernelArg(g_kf_bias_add, 0, sizeof(cl_mem), &out_f32);
        clSetKernelArg(g_kf_bias_add, 1, sizeof(cl_mem), &bias_fp32);
        clSetKernelArg(g_kf_bias_add, 2, sizeof(int), &C_out);
        clSetKernelArg(g_kf_bias_add, 3, sizeof(int), &L_out);
        size_t gws2[2] = {(size_t)C_out, (size_t)L_out};
        nnopt_enqueue_profiled(queue, g_kf_bias_add, 2, nullptr, gws2, nullptr, 0, nullptr, nullptr);
        gen_release(bias_fp32);
    }
    return 0;
}

// Plain Conv1d (no weight_norm) — for noise_convs
static int plain_conv1d_fp32(OpenCLContext& cl_ctx, Weights& weights, cl_command_queue queue,
                              const std::string& prefix, cl_mem in_f32, cl_mem out_f32,
                              int L_in, int stride, int padding, int* L_out_o, int* C_out_o) {
    cl_mem W = weights.get_buffer(prefix + ".weight");
    cl_mem b = weights.get_buffer(prefix + ".bias", true);
    auto shape = weights.get_shape(prefix + ".weight");
    int C_out = shape[0], C_in = shape[1], K = shape[2];
    int L_out = (L_in + 2*padding - (K-1) - 1) / stride + 1;
    if (L_out_o) *L_out_o = L_out;
    if (C_out_o) *C_out_o = C_out;
    int dilation = 1;
    int has_bias = b ? 1 : 0;
    cl_mem b_arg = b ? b : W;  // dummy if no bias

    // ── Texture path (NNOPT_NCTEX=0 reverts): weights through TP/L1,
    // 4oc×4ol per WI, fused bias. The scalar kernel below was 0.20 s for
    // noise_convs[0] alone. Image built once per prefix.
    static int use_nctex = -1;
    if (use_nctex == -1) {
        const char* e = std::getenv("NNOPT_NCTEX");
        use_nctex = (e && e[0] == '0') ? 0 : 1;
    }
    if (use_nctex && (C_out % 4) == 0) {
        static std::unordered_map<std::string, cl_mem> s_nc_img_cache;
        int Kp4 = (K + 3) / 4;
        cl_mem img = nullptr;
        auto it = s_nc_img_cache.find(prefix);
        if (it != s_nc_img_cache.end()) {
            img = it->second;
        } else {
            size_t img_w = (size_t)C_in * Kp4;
            size_t img_h = (size_t)C_out;
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
                cl_mem im = clCreateImage(cl_ctx.context(), CL_MEM_READ_WRITE, &fmt, &desc, nullptr, &ierr);
                if (ierr == CL_SUCCESS && im) {
                    clSetKernelArg(g_kf_nc_pack, 0, sizeof(cl_mem), &W);
                    clSetKernelArg(g_kf_nc_pack, 1, sizeof(cl_mem), &im);
                    clSetKernelArg(g_kf_nc_pack, 2, sizeof(int), &C_in);
                    clSetKernelArg(g_kf_nc_pack, 3, sizeof(int), &C_out);
                    clSetKernelArg(g_kf_nc_pack, 4, sizeof(int), &K);
                    clSetKernelArg(g_kf_nc_pack, 5, sizeof(int), &Kp4);
                    size_t gws_p[2] = {img_w, img_h};
                    nnopt_enqueue_profiled(queue, g_kf_nc_pack, 2, nullptr, gws_p, nullptr, 0, nullptr, nullptr);
                    img = im;
                }
            }
            s_nc_img_cache[prefix] = img;
        }
        if (img) {
            clSetKernelArg(g_kf_plain_conv1d_tex, 0, sizeof(cl_mem), &in_f32);
            clSetKernelArg(g_kf_plain_conv1d_tex, 1, sizeof(cl_mem), &img);
            clSetKernelArg(g_kf_plain_conv1d_tex, 2, sizeof(cl_mem), &b_arg);
            clSetKernelArg(g_kf_plain_conv1d_tex, 3, sizeof(int), &has_bias);
            clSetKernelArg(g_kf_plain_conv1d_tex, 4, sizeof(cl_mem), &out_f32);
            clSetKernelArg(g_kf_plain_conv1d_tex, 5, sizeof(int), &C_in);
            clSetKernelArg(g_kf_plain_conv1d_tex, 6, sizeof(int), &C_out);
            clSetKernelArg(g_kf_plain_conv1d_tex, 7, sizeof(int), &L_in);
            clSetKernelArg(g_kf_plain_conv1d_tex, 8, sizeof(int), &L_out);
            clSetKernelArg(g_kf_plain_conv1d_tex, 9, sizeof(int), &K);
            clSetKernelArg(g_kf_plain_conv1d_tex, 10, sizeof(int), &Kp4);
            clSetKernelArg(g_kf_plain_conv1d_tex, 11, sizeof(int), &stride);
            clSetKernelArg(g_kf_plain_conv1d_tex, 12, sizeof(int), &padding);
            const int loc0 = 2, loc1 = 64;
            size_t oc4 = (size_t)(C_out / 4);
            size_t oc4_p = ((oc4 + loc0 - 1) / loc0) * loc0;
            size_t tl = (size_t)((L_out + 3) / 4);
            size_t tl_p = ((tl + loc1 - 1) / loc1) * loc1;
            size_t gws_t[2] = {oc4_p, tl_p};
            size_t lws_t[2] = {loc0, loc1};
            cl_int terr = nnopt_enqueue_profiled(queue, g_kf_plain_conv1d_tex, 2, nullptr, gws_t, lws_t, 0, nullptr, nullptr);
            if (terr == CL_SUCCESS) return 0;
            NNOPT_ERROR_FMT("plain_conv1d_tex %s: dispatch failed (%d) — scalar fallback", prefix.c_str(), (int)terr);
        }
    }

    clSetKernelArg(g_kf_plain_conv1d, 0, sizeof(cl_mem), &in_f32);
    clSetKernelArg(g_kf_plain_conv1d, 1, sizeof(cl_mem), &W);
    clSetKernelArg(g_kf_plain_conv1d, 2, sizeof(cl_mem), &b_arg);
    clSetKernelArg(g_kf_plain_conv1d, 3, sizeof(cl_mem), &out_f32);
    clSetKernelArg(g_kf_plain_conv1d, 4, sizeof(int), &C_in);
    clSetKernelArg(g_kf_plain_conv1d, 5, sizeof(int), &C_out);
    clSetKernelArg(g_kf_plain_conv1d, 6, sizeof(int), &L_in);
    clSetKernelArg(g_kf_plain_conv1d, 7, sizeof(int), &L_out);
    clSetKernelArg(g_kf_plain_conv1d, 8, sizeof(int), &K);
    clSetKernelArg(g_kf_plain_conv1d, 9, sizeof(int), &stride);
    clSetKernelArg(g_kf_plain_conv1d, 10, sizeof(int), &padding);
    clSetKernelArg(g_kf_plain_conv1d, 11, sizeof(int), &dilation);
    clSetKernelArg(g_kf_plain_conv1d, 12, sizeof(int), &has_bias);
    size_t gws[2] = {(size_t)C_out, (size_t)L_out};
    return nnopt_enqueue_profiled(queue, g_kf_plain_conv1d, 2, nullptr, gws, nullptr, 0, nullptr, nullptr);
}

static void leaky_f32_inplace(cl_command_queue queue, cl_mem y, int N, float slope) {
    clSetKernelArg(g_kf_leaky, 0, sizeof(cl_mem), &y);
    clSetKernelArg(g_kf_leaky, 1, sizeof(int), &N);
    clSetKernelArg(g_kf_leaky, 2, sizeof(float), &slope);
    run1d_gws(queue, g_kf_leaky, (size_t)N);
}

// fp16 decoder helpers (defined in decoder.cpp)
extern "C" int dec_apply_adain1d(OpenCLContext&, Weights&, cl_command_queue,
                                  const std::string&, cl_mem, cl_mem, cl_mem, int, int);
extern "C" int dec_apply_snake1d(OpenCLContext&, Weights&, cl_command_queue,
                                  const std::string&, cl_mem, int, int);
extern "C" int dec_conv1d_wn(OpenCLContext&, Weights&, cl_command_queue,
                              const std::string&, cl_mem, cl_mem,
                              int, int, int, int, int, int*, int*);

extern "C" int op_decoder_gpu_fp32(OpenCLContext& cl_ctx, Weights& weights, cl_command_queue queue,
                                    cl_mem x_enc_fp16,        // [512, T_dec_final] from decode chain
                                    cl_mem F0_pred_fp16,      // [T_frames*2]
                                    cl_mem N_pred_fp16,       // [T_frames*2]
                                    cl_mem ref_s_dec_fp16,    // [256]
                                    int T_dec_final,
                                    int T_frames,
                                    std::vector<int16_t>& out_pcm_int16) {
    if (!ensure_built_gf(cl_ctx)) return -1;
    (void)N_pred_fp16; (void)T_frames;
    NNOPT_CHECKPOINT("GPU FP32 GENERATOR started");
    auto t0 = std::chrono::steady_clock::now();
    // TICK serializes (clFinish) ONLY in debug builds — in release the
    // checkpoint is compiled out and the clFinish was 8 hidden pipeline
    // serialization points per utterance.
    #define TICK(label) do { NNOPT_DEBUG_SYNC(queue); auto _t = std::chrono::steady_clock::now(); double _s = std::chrono::duration<double>(_t - t0).count(); NNOPT_CHECKPOINT((std::string("[fp32_gen] ") + (label) + " @ " + std::to_string(_s) + "s").c_str()); t0 = _t; } while (0)

    const int n_fft = 20, hop = 5, n_freq = 11;
    const int upsample_to_audio_full = 300;
    int T_audio_full = T_dec_final * upsample_to_audio_full;
    int T_high_stft = T_audio_full / hop + 1;

    // ─── Compute har on host (SineGen + STFT in fp64) then upload to GPU fp32 ───
    // Read F0 to host, compute har, upload
    std::vector<float> F0_host(T_dec_final);  // F0_pred at T_dec_final wait no — F0_pred is at T_frames*2 which equals T_dec_final
    {
        std::vector<uint16_t> fh(T_dec_final);
        clEnqueueReadBuffer(queue, F0_pred_fp16, CL_TRUE, 0, sizeof(uint16_t)*T_dec_final, fh.data(), 0, nullptr, nullptr);
        for (int i = 0; i < T_dec_final; ++i) F0_host[i] = nnopt_f16_to_f32(fh[i]);
    }
    // Read style first 128
    std::vector<float> style_full(256), style(128);
    {
        std::vector<uint16_t> sh(256);
        clEnqueueReadBuffer(queue, ref_s_dec_fp16, CL_TRUE, 0, sizeof(uint16_t)*256, sh.data(), 0, nullptr, nullptr);
        for (int i = 0; i < 256; ++i) style_full[i] = nnopt_f16_to_f32(sh[i]);
        for (int i = 0; i < 128; ++i) style[i] = style_full[i];
    }
    // Upsample F0 → cumsum → SineGen → l_linear → tanh → har_source → STFT → har [22, T_high_stft]
    std::vector<float> har_host((size_t)22 * T_high_stft);
    {
        // F0 upsample to T_audio_full (linear)
        std::vector<float> f0_up(T_audio_full);
        for (int t = 0; t < T_audio_full; ++t) {
            double pos = (double)t * (double)(T_dec_final - 1) / (double)(T_audio_full - 1);
            int t0 = (int)pos; int t1 = std::min(t0 + 1, T_dec_final - 1);
            double frac = pos - t0;
            f0_up[t] = (float)((1.0 - frac) * F0_host[t0] + frac * F0_host[t1]);
        }
        int n_harm = 9;
        double samp_rate = 24000.0, voiced_th = 10.0, sine_amp = 0.1;
        std::vector<double> cumphase(T_audio_full);
        double acc = 0.0;
        for (int t = 0; t < T_audio_full; ++t) { acc += f0_up[t] / samp_rate; cumphase[t] = acc; }
        // l_linear weights
        std::vector<float> lm_w(9), lm_b(1);
        cl_mem lw = weights.get_buffer("decoder.module.generator.m_source.l_linear.weight");
        cl_mem lb = weights.get_buffer("decoder.module.generator.m_source.l_linear.bias");
        if (lw) {
            std::vector<uint16_t> wh(9); clEnqueueReadBuffer(queue, lw, CL_TRUE, 0, sizeof(uint16_t)*9, wh.data(), 0, nullptr, nullptr);
            for (int i = 0; i < 9; ++i) lm_w[i] = nnopt_f16_to_f32(wh[i]);
        }
        if (lb) {
            std::vector<uint16_t> bh(1); clEnqueueReadBuffer(queue, lb, CL_TRUE, 0, sizeof(uint16_t), bh.data(), 0, nullptr, nullptr);
            lm_b[0] = nnopt_f16_to_f32(bh[0]);
        }
        std::vector<float> har_source(T_audio_full);
        for (int t = 0; t < T_audio_full; ++t) {
            double u = f0_up[t] > voiced_th ? 1.0 : 0.0;
            double s = lm_b[0];
            for (int h = 0; h < n_harm; ++h) {
                double phi = 2.0 * M_PI * (double)(h + 1) * cumphase[t];
                s += (double)lm_w[h] * std::sin(phi) * sine_amp * u;
            }
            har_source[t] = (float)std::tanh(s);
        }
        // STFT center=True with Hann periodic window
        std::vector<float> hann(n_fft);
        for (int i = 0; i < n_fft; ++i) hann[i] = (float)(0.5 * (1.0 - std::cos(2.0 * M_PI * i / n_fft)));
        for (int f = 0; f < T_high_stft; ++f) {
            int n0 = f * hop - n_fft / 2;
            for (int k = 0; k < n_freq; ++k) {
                double re = 0, im = 0;
                for (int t = 0; t < n_fft; ++t) {
                    int n = n0 + t;
                    if (n >= 0 && n < T_audio_full) {
                        double v = (double)har_source[n] * hann[t];
                        double ang = -2.0 * M_PI * k * t / n_fft;
                        re += v * std::cos(ang); im += v * std::sin(ang);
                    }
                }
                har_host[k*T_high_stft + f] = (float)std::sqrt(re*re + im*im);
                har_host[(n_freq+k)*T_high_stft + f] = (float)std::atan2(im, re);
            }
        }
    }
    // ─── Recording-stable persistent state (per T) ─────────────────────────
    // The generator span (leaky→ups0→noise→resblocks→ups1→…→conv_post) is a
    // fixed dispatch graph for a fixed T. With cl_qcom_recordable_queues we
    // run it live on call 1 (warms weight-image/wn caches), capture it into a
    // recording on call 2, and replay the recording (4× cheaper per dispatch)
    // from then on. Buffers that cross the span boundary (har input, x input,
    // style vector, conv_post output) live here so their handles are stable;
    // everything inside the span comes from the arena. NNOPT_RECORD=0 reverts.
    struct GenRecState {
        void*  rec = nullptr;
        int    rec_T = -1;        // T the recording was captured at
        int    rec_L_cp = 0;      // L_cp of the captured span (replay calls reuse it)
        uint64_t rec_gen = 0;     // arena generation at capture (handle stability)
        cl_mem har = nullptr, x = nullptr, refs = nullptr, cpost = nullptr;
        cl_mem mag = nullptr, phase = nullptr, audio = nullptr;   // GPU iSTFT chain
        int L_cp = 0;
        int calls = 0;            // consecutive calls at the same T (capture gate)
        int T = -1;               // T of the previous call
        int T_cap = -1;           // grow-only capacity of the boundary buffers
    };
    static GenRecState s_gr;
    static cl_command_queue s_rec_queue = nullptr;

    static int s_rec_enabled = -1;
    if (s_rec_enabled == -1) {
        const char* e = std::getenv("NNOPT_RECORD");
        bool gate = !(e && e[0] == '0');
        bool incompat = false;
        if (const char* p = std::getenv("NNOPT_PROFILE"))     incompat |= (p[0] == '1');
        if (const char* p = std::getenv("NNOPT_DUMP_RB3"))    incompat |= (p[0] == '1');
        if (const char* p = std::getenv("NNOPT_DUMP_LAYERS")) incompat |= (p[0] == '1');
#ifdef NNOPT_DEBUG
        incompat = true;   // TICK clFinish + layer checks can't live in a recording
#endif
        s_rec_enabled = (gate && !incompat && cl_ctx.has_recordable_queues()) ? 1 : 0;
    }
    // GPU fp32 iSTFT by default (the old precision bug was the fp16 kernel);
    // NNOPT_HOST_ISTFT=1 restores the host fp64 path for A/B.
    static int s_gpu_istft = -1;
    if (s_gpu_istft == -1) {
        const char* hi = std::getenv("NNOPT_HOST_ISTFT");
        s_gpu_istft = (hi && hi[0] == '1') ? 0 : 1;
    }

    // Grow-only boundary buffers: every size here is linear in T (audio = 300·T,
    // stft rows = 60·T+1), so capacity T_cap covers any chunk with T ≤ T_cap.
    // Streaming chunks vary in length — realloc per T change was arena churn
    // that kept sustained RTF above 1.0.
    if (T_dec_final > s_gr.T_cap) {
        if (s_gr.har)   clReleaseMemObject(s_gr.har);
        if (s_gr.x)     clReleaseMemObject(s_gr.x);
        if (s_gr.refs)  clReleaseMemObject(s_gr.refs);
        if (s_gr.cpost) clReleaseMemObject(s_gr.cpost);
        if (s_gr.mag)   clReleaseMemObject(s_gr.mag);
        if (s_gr.phase) clReleaseMemObject(s_gr.phase);
        if (s_gr.audio) clReleaseMemObject(s_gr.audio);
        s_gr.har = s_gr.x = s_gr.refs = s_gr.cpost = nullptr;
        s_gr.mag = s_gr.phase = s_gr.audio = nullptr;
        // 1/8 headroom so chunk-length jitter doesn't regrow the buffers
        s_gr.T_cap = T_dec_final + T_dec_final / 8;
        ++g_gen_arena.generation;   // boundary handles changed → recordings stale
    }
    if (T_dec_final != s_gr.T) { s_gr.T = T_dec_final; s_gr.calls = 0; }
    s_gr.calls++;
    g_gen_arena.reset();
    g_gen_arena_active = true;

    if (!s_gr.har) {
        cl_int aerr = CL_SUCCESS;
        // Sizes at T_cap capacity. L_cp = 60*T + 1 (ups 10x then 6x then +1
        // reflection pad); T_high_stft = 60*T + 1 as well (300·T audio / hop 5).
        const size_t l_cp_max = (size_t)60 * s_gr.T_cap + 1;
        const size_t t_high_stft_max = (size_t)s_gr.T_cap * upsample_to_audio_full / hop + 1;
        s_gr.har   = clCreateBuffer(cl_ctx.context(), CL_MEM_READ_WRITE, sizeof(float) * 22 * t_high_stft_max, nullptr, &aerr);
        s_gr.x     = clCreateBuffer(cl_ctx.context(), CL_MEM_READ_WRITE, sizeof(float) * 512 * (size_t)s_gr.T_cap, nullptr, &aerr);
        s_gr.refs  = clCreateBuffer(cl_ctx.context(), CL_MEM_READ_WRITE, sizeof(uint16_t) * 256, nullptr, &aerr);
        s_gr.cpost = clCreateBuffer(cl_ctx.context(), CL_MEM_READ_WRITE, sizeof(float) * 22 * l_cp_max, nullptr, &aerr);
        const size_t t_audio_max = (l_cp_max - 1) * hop + n_fft;
        s_gr.mag   = clCreateBuffer(cl_ctx.context(), CL_MEM_READ_WRITE, sizeof(uint16_t) * n_freq * l_cp_max, nullptr, &aerr);
        s_gr.phase = clCreateBuffer(cl_ctx.context(), CL_MEM_READ_WRITE, sizeof(uint16_t) * n_freq * l_cp_max, nullptr, &aerr);
        s_gr.audio = clCreateBuffer(cl_ctx.context(), CL_MEM_READ_WRITE, sizeof(float) * t_audio_max, nullptr, &aerr);
        if (aerr != CL_SUCCESS || !s_gr.har || !s_gr.x || !s_gr.refs || !s_gr.cpost ||
            !s_gr.mag || !s_gr.phase || !s_gr.audio) {
            g_gen_arena_active = false;
            NNOPT_ERROR("gen persistent buffer alloc failed");
            return -1;
        }
    }

    // Pre-span (live queue; not recordable): har upload, stable style copy,
    // x_enc conversion. x_enc_fp16/ref_s_dec_fp16 handles change per utterance
    // so the span only ever sees the stable copies.
    cl_mem har_fp32 = s_gr.har;
    clEnqueueWriteBuffer(queue, har_fp32, CL_TRUE, 0, sizeof(float) * 22 * T_high_stft, har_host.data(), 0, nullptr, nullptr);
    clEnqueueCopyBuffer(queue, ref_s_dec_fp16, s_gr.refs, 0, 0, sizeof(uint16_t) * 256, 0, nullptr, nullptr);
    cl_mem refs = s_gr.refs;
    gen_fp16_to_fp32(cl_ctx, queue, x_enc_fp16, s_gr.x, 512 * T_dec_final);
    TICK("har_host computed + uploaded");

    // ─── GPU fp32 generator chain (recordable span) ───
    const int resblock_kernels[3] = {3, 7, 11};
    const int resblock_dils[3] = {1, 3, 5};

    auto copy_f32_k = [&](cl_command_queue qq, cl_mem src, cl_mem dst, int N) {
        clSetKernelArg(g_kf_copy_f32, 0, sizeof(cl_mem), &src);
        clSetKernelArg(g_kf_copy_f32, 1, sizeof(cl_mem), &dst);
        clSetKernelArg(g_kf_copy_f32, 2, sizeof(int), &N);
        run1d_gws(qq, g_kf_copy_f32, (size_t)N);
    };

    int span_L_cp = 0;
    auto span = [&](cl_command_queue q) -> int {
        int gC = 512, gT = T_dec_final;
        cl_mem x_fp32 = s_gr.x;

        // Level 0: leaky → ups[0] → noise_convs[0]+noise_res[0] add → resblocks 0..2 avg
        leaky_f32_inplace(q, x_fp32, gC * gT, 0.1f);
        int L_u0, C_u0;
        int L_u0_expect = (gT - 1)*10 - 2*5 + 19 + 1;
        cl_mem u0_out = alloc_fp32(cl_ctx, (size_t)256 * L_u0_expect);
        if (convtr1d_wn_fp32(cl_ctx, weights, q, "decoder.module.generator.ups.0",
                              x_fp32, u0_out, gT, 10, 5, &L_u0, &C_u0) != 0) return -1;
        gC = 256; gT = L_u0;
        TICK("ups[0] done");

        // noise_convs[0]: Conv1d(22, 256, k=12, stride=6, padding=3)
        cl_mem nc0 = alloc_fp32(cl_ctx, (size_t)256 * gT);
        int Lnc0, Cnc0;
        if (plain_conv1d_fp32(cl_ctx, weights, q, "decoder.module.generator.noise_convs.0",
                               har_fp32, nc0, T_high_stft, 6, 3, &Lnc0, &Cnc0) == 0 && Lnc0 == gT) {
            if (gen_apply_adainresblock1_fp32(cl_ctx, weights, q,
                                               "decoder.module.generator.noise_res.0",
                                               nc0, refs, 256, gT, 7, resblock_dils) == 0) {
                int N = 256 * gT;
                clSetKernelArg(g_kf_add, 0, sizeof(cl_mem), &u0_out);
                clSetKernelArg(g_kf_add, 1, sizeof(cl_mem), &nc0);
                clSetKernelArg(g_kf_add, 2, sizeof(int), &N);
                run1d_gws(q, g_kf_add, (size_t)N);
            }
        }
        gen_release(nc0);

        // resblocks 0..2 in fp32
        cl_mem r0[3];
        for (int j = 0; j < 3; ++j) {
            r0[j] = alloc_fp32(cl_ctx, (size_t)gC * gT);
            copy_f32_k(q, u0_out, r0[j], gC * gT);
            if (gen_apply_adainresblock1_fp32(cl_ctx, weights, q,
                                               "decoder.module.generator.resblocks." + std::to_string(j),
                                               r0[j], refs, gC, gT,
                                               resblock_kernels[j], resblock_dils) != 0) return -1;
        }
        cl_mem xs0 = alloc_fp32(cl_ctx, (size_t)gC * gT);
        gen_avg_three_fp32(cl_ctx, q, r0[0], r0[1], r0[2], xs0, gC * gT);
        for (int j = 0; j < 3; ++j) gen_release(r0[j]);
        gen_release(u0_out);
        TICK("resblocks_012 done");

        // Level 1: leaky → ups[1] → refl_pad → noise add → resblocks 3..5 avg
        leaky_f32_inplace(q, xs0, gC * gT, 0.1f);
        int L_u1, C_u1;
        int L_u1_expect = (gT - 1)*6 - 2*3 + 11 + 1;
        cl_mem u1_out = alloc_fp32(cl_ctx, (size_t)128 * L_u1_expect);
        if (convtr1d_wn_fp32(cl_ctx, weights, q, "decoder.module.generator.ups.1",
                              xs0, u1_out, gT, 6, 3, &L_u1, &C_u1) != 0) return -1;
        gen_release(xs0);
        gC = 128; gT = L_u1;
        // reflection_pad (1, 0)
        int gT_pad = gT + 1;
        cl_mem padded = alloc_fp32(cl_ctx, (size_t)gC * gT_pad);
        int pl = 1, pr = 0;
        clSetKernelArg(g_kf_refl_pad, 0, sizeof(cl_mem), &u1_out);
        clSetKernelArg(g_kf_refl_pad, 1, sizeof(cl_mem), &padded);
        clSetKernelArg(g_kf_refl_pad, 2, sizeof(int), &gC);
        clSetKernelArg(g_kf_refl_pad, 3, sizeof(int), &gT);
        clSetKernelArg(g_kf_refl_pad, 4, sizeof(int), &pl);
        clSetKernelArg(g_kf_refl_pad, 5, sizeof(int), &pr);
        size_t gws_p[2] = {(size_t)gC, (size_t)gT_pad};
        nnopt_enqueue_profiled(q, g_kf_refl_pad, 2, nullptr, gws_p, nullptr, 0, nullptr, nullptr);
        gen_release(u1_out);
        gT = gT_pad;
        // noise_convs[1]: Conv1d(22, 128, k=1, stride=1, padding=0)
        cl_mem nc1 = alloc_fp32(cl_ctx, (size_t)128 * T_high_stft);
        int Lnc1, Cnc1;
        if (plain_conv1d_fp32(cl_ctx, weights, q, "decoder.module.generator.noise_convs.1",
                               har_fp32, nc1, T_high_stft, 1, 0, &Lnc1, &Cnc1) == 0 && Lnc1 == gT) {
            if (gen_apply_adainresblock1_fp32(cl_ctx, weights, q,
                                               "decoder.module.generator.noise_res.1",
                                               nc1, refs, 128, gT, 11, resblock_dils) == 0) {
                int N = 128 * gT;
                clSetKernelArg(g_kf_add, 0, sizeof(cl_mem), &padded);
                clSetKernelArg(g_kf_add, 1, sizeof(cl_mem), &nc1);
                clSetKernelArg(g_kf_add, 2, sizeof(int), &N);
                run1d_gws(q, g_kf_add, (size_t)N);
            }
        }
        gen_release(nc1);
        gen_release(har_fp32);
        // resblocks 3..5
        cl_mem r1[3];
        for (int j = 0; j < 3; ++j) {
            r1[j] = alloc_fp32(cl_ctx, (size_t)gC * gT);
            copy_f32_k(q, padded, r1[j], gC * gT);
            if (gen_apply_adainresblock1_fp32(cl_ctx, weights, q,
                                               "decoder.module.generator.resblocks." + std::to_string(3+j),
                                               r1[j], refs, gC, gT,
                                               resblock_kernels[j], resblock_dils) != 0) return -1;
        }
        cl_mem xs1 = alloc_fp32(cl_ctx, (size_t)gC * gT);
        gen_avg_three_fp32(cl_ctx, q, r1[0], r1[1], r1[2], xs1, gC * gT);
        for (int j = 0; j < 3; ++j) gen_release(r1[j]);
        gen_release(padded);
        TICK("resblocks_345 done");

        // Final leaky + conv_post
        leaky_f32_inplace(q, xs1, gC * gT, 0.01f);
        int C_cp;
        if (conv1d_wn_fp32(cl_ctx, weights, q, "decoder.module.generator.conv_post",
                            xs1, s_gr.cpost, gT, 1, 3, 1, 1, &span_L_cp, &C_cp) != 0) return -1;
        gen_release(xs1);
        if (s_gpu_istft) {
            // conv_post raw [22, L_cp] → exp(mag)/sin(phase) fp16 → iSTFT, all
            // on-GPU and inside the recordable span. Replaces the 0.8 MB cpost
            // readback + host fp64 loop with one 0.18 MB audio readback.
            int NT = n_freq * span_L_cp;
            clSetKernelArg(g_kf_conv_post_split, 0, sizeof(cl_mem), &s_gr.cpost);
            clSetKernelArg(g_kf_conv_post_split, 1, sizeof(cl_mem), &s_gr.mag);
            clSetKernelArg(g_kf_conv_post_split, 2, sizeof(cl_mem), &s_gr.phase);
            clSetKernelArg(g_kf_conv_post_split, 3, sizeof(int), &n_freq);
            clSetKernelArg(g_kf_conv_post_split, 4, sizeof(int), &span_L_cp);
            run1d_gws(q, g_kf_conv_post_split, (size_t)NT);
            int T_audio_gpu = (span_L_cp - 1) * hop + n_fft;
            clSetKernelArg(g_kf_istft, 0, sizeof(cl_mem), &s_gr.mag);
            clSetKernelArg(g_kf_istft, 1, sizeof(cl_mem), &s_gr.phase);
            clSetKernelArg(g_kf_istft, 2, sizeof(cl_mem), &s_gr.audio);
            clSetKernelArg(g_kf_istft, 3, sizeof(int), &span_L_cp);
            clSetKernelArg(g_kf_istft, 4, sizeof(int), &T_audio_gpu);
            clSetKernelArg(g_kf_istft, 5, sizeof(int), &n_fft);
            clSetKernelArg(g_kf_istft, 6, sizeof(int), &hop);
            clSetKernelArg(g_kf_istft, 7, sizeof(int), &n_freq);
            size_t gws_i = ((size_t)T_audio_gpu + 63) & ~(size_t)63;
            size_t lws_i = 64;
            nnopt_enqueue_profiled(q, g_kf_istft, 1, nullptr, &gws_i, &lws_i, 0, nullptr, nullptr);
        }
        return 0;
    };

    // ─── Live / record / replay ─────────────────────────────────────────────
    int span_rc = -1;
    bool replayed = false;
    // A recording bakes both the kernel args (T) and the buffer handles, so it
    // is only replayable for the exact T it was captured at AND while no arena
    // slot or boundary buffer has reallocated since (generation match).
    if (s_gr.rec && (s_gr.rec_T != T_dec_final || s_gr.rec_gen != g_gen_arena.generation)) {
        cl_ctx.release_recording(s_gr.rec);
        s_gr.rec = nullptr;
    }
    if (s_rec_enabled && s_gr.rec) {
        cl_int re = cl_ctx.enqueue_recording(queue, s_gr.rec, 0, nullptr);
        if (re == CL_SUCCESS) { replayed = true; span_rc = 0; }
        else {
            NNOPT_ERROR_FMT("gen record: replay failed (%d) — dropping recording, live dispatch", (int)re);
            cl_ctx.release_recording(s_gr.rec);
            s_gr.rec = nullptr;
        }
    }
    if (!replayed && s_rec_enabled && s_gr.calls >= 2) {
        // Call 1 ran live and warmed every weight-image/wn cache; capture now.
        if (!s_rec_queue) s_rec_queue = cl_ctx.create_recordable_queue();
        if (s_rec_queue) {
            void* rec = cl_ctx.new_recording(s_rec_queue);
            if (rec) {
                int rc_rec = span(s_rec_queue);          // captured, NOT executed
                cl_int ee = cl_ctx.end_recording(rec);
                if (rc_rec == 0 && ee == CL_SUCCESS) {
                    cl_int re = cl_ctx.enqueue_recording(queue, rec, 0, nullptr);
                    if (re == CL_SUCCESS) {
                        s_gr.rec = rec;
                        s_gr.rec_T = T_dec_final;
                        s_gr.rec_L_cp = span_L_cp;
                        s_gr.rec_gen = g_gen_arena.generation;
                        replayed = true; span_rc = 0;
                        NNOPT_CHECKPOINT("gen record: captured + replaying");
                    } else {
                        NNOPT_ERROR_FMT("gen record: first replay failed (%d) — live dispatch", (int)re);
                        cl_ctx.release_recording(rec);
                    }
                } else {
                    cl_ctx.release_recording(rec);
                }
            }
        }
    }
    if (!replayed) {
        g_gen_arena.pos = 0;   // restart slot handout if a record attempt consumed slots
        span_rc = span(queue);
    }
    g_gen_arena_active = false;
    if (span_rc != 0) return -1;
    // Live/capture calls compute span_L_cp; replay-only calls take the value
    // captured with the recording (s_gr.L_cp alone could be stale from an
    // intervening different-T chunk).
    const int L_cp = (span_L_cp > 0) ? span_L_cp : s_gr.rec_L_cp;
    s_gr.L_cp = L_cp;
    cl_mem cpost_out = s_gr.cpost;

    int T_audio = (L_cp - 1) * hop + n_fft;
    std::vector<double> audio(T_audio, 0.0);
    if (s_gpu_istft) {
        // GPU split+iSTFT already ran inside the span — one small readback.
        std::vector<float> audio_f((size_t)T_audio);
        clEnqueueReadBuffer(queue, s_gr.audio, CL_TRUE, 0, sizeof(float) * T_audio, audio_f.data(), 0, nullptr, nullptr);
        for (int i = 0; i < T_audio; ++i) audio[i] = (double)audio_f[i];
    } else {
        // Host fp64 path (NNOPT_HOST_ISTFT=1): cpost readback + CPU loop.
        std::vector<float> cpost_host((size_t)22 * L_cp);
        clEnqueueReadBuffer(queue, cpost_out, CL_TRUE, 0, sizeof(float) * 22 * L_cp, cpost_host.data(), 0, nullptr, nullptr);
        std::vector<double> hann(n_fft);
        for (int i = 0; i < n_fft; ++i) hann[i] = 0.5 * (1.0 - std::cos(2.0 * M_PI * i / n_fft));
        std::vector<double> norm_buf(T_audio, 0.0);
        for (int f = 0; f < L_cp; ++f) {
            for (int t = 0; t < n_fft; ++t) {
                double s = std::exp((double)cpost_host[0*L_cp + f]) * std::cos(std::sin((double)cpost_host[n_freq*L_cp + f]));
                for (int k = 1; k < n_freq - 1; ++k) {
                    double mag_k = std::exp((double)cpost_host[k*L_cp + f]);
                    double phase_k = std::sin((double)cpost_host[(n_freq+k)*L_cp + f]);
                    double ang = 2.0 * M_PI * k * t / n_fft + phase_k;
                    s += 2.0 * mag_k * std::cos(ang);
                }
                double mag_n = std::exp((double)cpost_host[(n_freq-1)*L_cp + f]);
                double phase_n = std::sin((double)cpost_host[(n_freq+n_freq-1)*L_cp + f]);
                s += mag_n * std::cos(M_PI * t + phase_n);
                s /= (double)n_fft;
                int oi = f * hop + t;
                if (oi < T_audio) {
                    audio[oi] += hann[t] * s;
                    norm_buf[oi] += hann[t] * hann[t];
                }
            }
        }
        for (int i = 0; i < T_audio; ++i) if (norm_buf[i] > 1e-10) audio[i] /= norm_buf[i];
    }

    // center=True trim + DC removal. NO loudness normalization: the reference
    // emits the raw float waveform (peak ~0.3); the old 0.8/p99 rescale (~4.3x)
    // hard-clipped ~0.36% of samples, which lays an audible high-pitched
    // distorted copy of the voice over the speech. Clamp is safety only.
    int trim = n_fft / 2;
    int T_trim = T_audio - 2 * trim;
    double dc = 0.0;
    for (int i = 0; i < T_trim; ++i) dc += audio[trim + i];
    dc /= (double)T_trim;
    out_pcm_int16.assign((size_t)T_trim, 0);
    for (int i = 0; i < T_trim; ++i) {
        double s = audio[trim + i] - dc;
        if (s > 1.0) s = 1.0; if (s < -1.0) s = -1.0;
        out_pcm_int16[i] = (int16_t)(s * 32767.0);
    }
    TICK("conv_post + host iSTFT done");
    NNOPT_CHECKPOINT("GPU FP32 GENERATOR done");
    #undef TICK
    return 0;
}

// GPU conv_post: weight_norm Conv1d(128 -> 2*n_freq, K=7, pad=3) in fp32 +
// exp/sin epilogue, writing the fp16 mag/phase buffers directly. Replaces the
// host fp64 version (~0.4s of blocking readback + CPU loops per inference).
extern "C" int gen_conv_post_gpu(OpenCLContext& cl_ctx, Weights& weights, cl_command_queue queue,
                                 cl_mem xs_fp16, cl_mem mag_fp16, cl_mem phase_fp16,
                                 int C_in, int T, int n_freq) {
    if (!ensure_built_gf(cl_ctx)) return -1;
    int N_in = C_in * T;
    cl_mem xs_f32 = alloc_fp32(cl_ctx, (size_t)N_in);
    if (!xs_f32) return -1;
    gen_fp16_to_fp32(cl_ctx, queue, xs_fp16, xs_f32, N_in);
    cl_mem out_f32 = alloc_fp32(cl_ctx, (size_t)(2 * n_freq) * T);
    if (!out_f32) { gen_release(xs_f32); return -1; }
    int L_out = 0, C_out = 0;
    if (conv1d_wn_fp32(cl_ctx, weights, queue, "decoder.module.generator.conv_post",
                       xs_f32, out_f32, T, 1, 3, 1, 1, &L_out, &C_out) != 0) {
        gen_release(xs_f32); gen_release(out_f32); return -1;
    }
    if (L_out != T || C_out != 2 * n_freq) {
        NNOPT_ERROR_FMT("conv_post_gpu: unexpected dims L_out=%d C_out=%d (want %d, %d)",
                        L_out, C_out, T, 2 * n_freq);
        gen_release(xs_f32); gen_release(out_f32); return -1;
    }
    clSetKernelArg(g_kf_conv_post_split, 0, sizeof(cl_mem), &out_f32);
    clSetKernelArg(g_kf_conv_post_split, 1, sizeof(cl_mem), &mag_fp16);
    clSetKernelArg(g_kf_conv_post_split, 2, sizeof(cl_mem), &phase_fp16);
    clSetKernelArg(g_kf_conv_post_split, 3, sizeof(int), &n_freq);
    clSetKernelArg(g_kf_conv_post_split, 4, sizeof(int), &T);
    size_t gws = (size_t)n_freq * T;
    nnopt_enqueue_profiled(queue, g_kf_conv_post_split, 1, nullptr, &gws, nullptr, 0, nullptr, nullptr);
    gen_release(xs_f32);
    gen_release(out_f32);
    return 0;
}

// GPU iSTFT entry: mag/phase fp16 device buffers -> host fp32 audio vector.
extern "C" int gen_istft_gpu(OpenCLContext& cl_ctx, cl_command_queue queue,
                             cl_mem mag_fp16, cl_mem phase_fp16,
                             int L_frames, int n_fft, int hop, int n_freq,
                             float* audio_out, int T_audio) {
    if (!ensure_built_gf(cl_ctx)) return -1;
    cl_mem audio_dev = alloc_fp32(cl_ctx, (size_t)T_audio);
    if (!audio_dev) return -1;
    clSetKernelArg(g_kf_istft, 0, sizeof(cl_mem), &mag_fp16);
    clSetKernelArg(g_kf_istft, 1, sizeof(cl_mem), &phase_fp16);
    clSetKernelArg(g_kf_istft, 2, sizeof(cl_mem), &audio_dev);
    clSetKernelArg(g_kf_istft, 3, sizeof(int), &L_frames);
    clSetKernelArg(g_kf_istft, 4, sizeof(int), &T_audio);
    clSetKernelArg(g_kf_istft, 5, sizeof(int), &n_fft);
    clSetKernelArg(g_kf_istft, 6, sizeof(int), &hop);
    clSetKernelArg(g_kf_istft, 7, sizeof(int), &n_freq);
    size_t gws = ((size_t)T_audio + 63) & ~(size_t)63;
    size_t lws = 64;
    nnopt_enqueue_profiled(queue, g_kf_istft, 1, nullptr, &gws, &lws, 0, nullptr, nullptr);
    cl_int err = clEnqueueReadBuffer(queue, audio_dev, CL_TRUE, 0,
                                     sizeof(float) * (size_t)T_audio, audio_out, 0, nullptr, nullptr);
    gen_release(audio_dev);
    return (err == CL_SUCCESS) ? 0 : -1;
}
