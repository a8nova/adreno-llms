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

__kernel void leaky_relu(
    __global       nnopt_storage_t* x,
    const int N,
    const float negative_slope) {

    const int gid = get_global_id(0);
    if (gid >= N) return;
    const float v = (float)x[gid];
    x[gid] = (nnopt_storage_t)(v >= 0.0f ? v : v * negative_slope);
}

// elementwise add — for residual sums and the 3-branch ResBlock1 reduction.
__kernel void elementwise_add(
    __global const nnopt_storage_t* a,
    __global const nnopt_storage_t* b,
    __global       nnopt_storage_t* out,
    const int N) {
    const int gid = get_global_id(0);
    if (gid >= N) return;
    out[gid] = (nnopt_storage_t)((float)a[gid] + (float)b[gid]);
}

// final tanh — applied once after the post-conv1d in HiFi-GAN's generator.
// COMMON BUG (per tts.md): omitting this produces clipped/blown-out audio
// with cos ≥ 0.95 at the pre-tanh layer but audibly wrong output.
__kernel void tanh_inplace(
    __global       nnopt_storage_t* x,
    const int N) {
    const int gid = get_global_id(0);
    if (gid >= N) return;
    x[gid] = (nnopt_storage_t)tanh((float)x[gid]);
}
