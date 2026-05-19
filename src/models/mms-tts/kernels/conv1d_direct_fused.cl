// Direct fused conv1d for the HiFi-GAN resblock path.
//
// Replaces the (leaky_im2col + CLBlast HGemm + add_bias_broadcast[_resid])
// 3-dispatch chain that conv1d_gemm_fused() generates today with ONE dispatch.
//
// On Adreno 620 the per-dispatch host overhead is ~70 ms (kernel-busy fraction
// during the vocoder phase is 4.5%), so dispatch-count is the dominant cost,
// not raw kernel time. Collapsing 3→1 dispatches per conv removes ~140 host
// overhead ticks per inference.
//
// One workitem per output element (C_out * L_out total). fp32 accumulator,
// fp16 storage. Optional leaky_relu(slope=0.1) at the read, optional bias,
// optional residual add at the write — all gated by flags so a single kernel
// covers conv1 (leaky+bias) and conv2 (leaky+bias+resid).

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

__kernel void conv1d_direct_fused(
    __global const storage_t* in,           // [C_in, L_in]
    __global const storage_t* weight,       // [C_out, C_in, K]
    __global const storage_t* bias,         // [C_out] (read only if has_bias)
    __global const storage_t* resid,        // [C_out, L_out] (read only if has_resid)
    __global       storage_t* out,          // [C_out, L_out]
    const int C_in,
    const int C_out,
    const int L_in,
    const int L_out,
    const int K,
    const int stride,
    const int padding,
    const int dilation,
    const int leaky_in,         // 1 = leaky_relu(slope=0.1) on the read; 0 = pass-through
    const int has_bias,
    const int has_resid) {

    const int gid = get_global_id(0);
    const int total = C_out * L_out;
    if (gid >= total) return;

    const int oc = gid / L_out;
    const int ol = gid % L_out;

    float acc = 0.0f;
    for (int ic = 0; ic < C_in; ++ic) {
        for (int k = 0; k < K; ++k) {
            const int il = ol * stride + k * dilation - padding;
            if (il < 0 || il >= L_in) continue;
            float xv = (float)LOAD(in, ic * L_in + il);
            if (leaky_in && xv < 0.0f) xv *= 0.1f;
            const float wv = (float)LOAD(weight, oc * (C_in * K) + ic * K + k);
            acc += xv * wv;
        }
    }
    if (has_bias)  acc += (float)LOAD(bias, oc);
    if (has_resid) acc += (float)LOAD(resid, gid);
    STORE(out, gid, acc);
}
