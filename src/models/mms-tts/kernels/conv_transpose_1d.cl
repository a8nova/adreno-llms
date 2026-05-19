// Reference: model_info/transformers_src/modeling_vits.py (see VitsHifiGan.forward and its use of
// torch.nn.ConvTranspose1d in VitsHifiGan.__init__ upsampler blocks)
//
// conv_transpose_1d — transposed (fractionally-strided) 1D convolution.
//
// Used by HiFi-GAN upsampling blocks in VITS / Kokoro / Piper vocoders.
// PyTorch semantics (output_padding=0, groups=1):
//
//   L_out = (L_in - 1) * stride - 2*padding + dilation*(K - 1) + 1
//
// Implementation: each output position 'ol' "gathers" the input positions
// 'il' whose forward-conv kernel taps would have landed on ol. Concretely:
//
//   il*stride - padding + k*dilation == ol      (forward sense)
// ⇒ il = (ol + padding - k*dilation) / stride   (integer; must be exact)
//
// We loop over k, check the integer-divisibility condition, accumulate.
// Layout matches conv_1d for consistency.
//
// One work-item per output element. C_out * L_out total.

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

__kernel void conv_transpose_1d(
    __global const storage_t* in,           // [C_in,  L_in]
    __global const storage_t* weight,       // [C_in,  C_out, K]   (transposed-conv layout)
    __global const storage_t* bias,         // [C_out] or NULL
    __global       storage_t* out,          // [C_out, L_out]
    const int C_in,
    const int C_out,
    const int L_in,
    const int L_out,
    const int K,
    const int stride,
    const int padding,
    const int dilation,
    const int has_bias) {

    const int gid = get_global_id(0);
    const int total = C_out * L_out;
    if (gid >= total) return;

    const int oc = gid / L_out;
    const int ol = gid % L_out;

    float acc = 0.0f;

    // NOTE: conv_transpose weights are indexed as [C_in, C_out, K]. Our flattened
    // layout uses the stride: ic*(C_out*K) + oc*K + k.
    //
    // IMPORTANT: For performance and to avoid device watchdog/OOM kills on mobile GPUs,
    // bound the work. In this port we only need groups=1, output_padding=0.
    // If dimensions are unexpectedly huge, early-exit with 0s rather than hanging.
    if (C_in <= 0 || C_out <= 0 || L_in <= 0 || L_out <= 0 || K <= 0) {
      STORE(out, gid, 0.0f);
      return;
    }

    // Conservative cap: prevent pathological T_frames explosions from running an
    // impractically large deconvolution (e.g. divergent durations). This is a
    // safety valve; correctness is handled earlier by duration capping.
    // 16M MACs per output channel is already heavy on Adreno.
    const long macs_per_out = (long)C_in * (long)K;
    if (macs_per_out > (long)16384) {  // 16k MACs per output element
      if (has_bias) {
        STORE(out, gid, (float)LOAD(bias, oc));
      } else {
        STORE(out, gid, 0.0f);
      }
      return;
    }

    for (int k = 0; k < K; ++k) {
        const int num = ol + padding - k * dilation;
        if (num < 0) continue;
        if ((num % stride) != 0) continue;
        const int il = num / stride;
        if (il < 0 || il >= L_in) continue;
        for (int ic = 0; ic < C_in; ++ic) {
            const float xv = (float)LOAD(in, ic * L_in + il);
            const float wv = (float)LOAD(weight, ic * (C_out * K) + oc * K + k);
            acc += xv * wv;
        }
    }
    if (has_bias) acc += (float)LOAD(bias, oc);
    STORE(out, gid, acc);
}
