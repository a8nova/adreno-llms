// Reference: model_info/transformers_src/modeling_vits.py (see HifiGanResidualBlock.forward and
// VitsHifiGan.forward: leaky_relu, residual adds, final tanh)
//
// hifigan_residual_block — HiFi-GAN-style residual block, applied per
// upsample stage after the ConvTranspose1d.
//
// HiFi-GAN's "ResBlock1" (the most common; VITS / Piper use it):
//   for each of three dilation rates d in [1, 3, 5]:
//     y = x + conv1d(leaky_relu(conv1d(leaky_relu(x), d=d, pad=d, K=3)),
//                     d=1, pad=1, K=3)
//   sum three branches into output.
//
// We DON'T fuse those into one kernel — they're best expressed as a
// sequence of conv_1d kernel launches with different dilations, gated by
// leaky_relu launches. That keeps the kernel small and lets the host wrapper
// orchestrate. This file defines just the leaky_relu activation kernel —
// the rest is conv_1d calls in src/ops/vocoder.cpp host code.
//
// PyTorch leaky_relu default negative_slope is 0.1; pass at runtime so
// quantized fp16 paths can match.

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

// Branchless leaky_relu (§10.3.6: eliminate wave divergence).
__kernel void leaky_relu(
    __global       storage_t* x,
    const int N,
    const float negative_slope) {

    const int gid = get_global_id(0);
    if (gid >= N) return;
    const float v = (float)LOAD(x, gid);
    STORE(x, gid, fmax(v, 0.0f) + fmin(v, 0.0f) * negative_slope);
}

// elementwise add — for residual sums and the 3-branch ResBlock1 reduction.
__kernel void elementwise_add(
    __global const storage_t* a,
    __global const storage_t* b,
    __global       storage_t* out,
    const int N) {
    const int gid = get_global_id(0);
    if (gid >= N) return;
    STORE(out, gid, ((float)LOAD(a, gid) + (float)LOAD(b, gid)));
}

// final tanh — applied once after the post-conv1d in HiFi-GAN's generator.
// COMMON BUG (per tts.md): omitting this produces clipped/blown-out audio
// with cos ≥ 0.95 at the pre-tanh layer but audibly wrong output.
// tanh via native_exp: tanh(x) = 1 - 2/(exp(2x)+1).
// With -cl-fast-relaxed-math, exp() already uses native_exp.
__kernel void tanh_inplace(
    __global       storage_t* x,
    const int N) {
    const int gid = get_global_id(0);
    if (gid >= N) return;
    const float v = (float)LOAD(x, gid);
    const float e2 = native_exp(2.0f * v);
    STORE(x, gid, (e2 - 1.0f) * native_recip(e2 + 1.0f));
}

// In-place scale. Used for averaging the 3 parallel resblock outputs.
__kernel void scale_inplace(
    __global       storage_t* x,
    const int N,
    const float scale) {
    const int gid = get_global_id(0);
    if (gid >= N) return;
    STORE(x, gid, ((float)LOAD(x, gid)) * scale);
}

// Fused 3-branch reduction: out = (b0 + b1 + b2) * (1/3). Replaces
// 2 × elementwise_add + 1 × scale_inplace per resblock stage (4 → 1 per
// stage; saves 12 dispatches across the 4 vocoder upsample stages).
__kernel void branch_reduce_3(
    __global const storage_t* b0,
    __global const storage_t* b1,
    __global const storage_t* b2,
    __global       storage_t* out,
    const int N) {
    const int gid = get_global_id(0);
    if (gid >= N) return;
    const float s = (float)LOAD(b0, gid) + (float)LOAD(b1, gid) + (float)LOAD(b2, gid);
    STORE(out, gid, s * (1.0f / 3.0f));
}
