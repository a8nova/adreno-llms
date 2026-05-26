// im2col for 1D convolution.
//
// Lifts a channels-first conv1d input [C_in, L_in] into a 2D matrix
// [C_in * K, L_out] so that conv1d → standard GEMM:
//
//   out[OC, L_out] = W[OC, C_in*K]  @  im2col[C_in*K, L_out]
//
// One workitem per im2col output element. Coordinates:
//   row = ic * K + k        (0 ≤ row < C_in*K)
//   col = ol                (0 ≤ col < L_out)
//   src = ol * stride + k * dilation - padding   (into in[ic, :])
//
// Out-of-range source positions write 0 (matches zero-padding semantics
// used by the reference conv1d kernel).

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

// Step C #9: 3D NDRange — eliminates four expensive integer divide/modulo
// operations per workitem (col=gid%L_out, row=gid/L_out, k=row%K, ic=row/K).
// Adreno guide §8.12 (p.69): "Avoid integer divide. Integer divide is very
// expensive in Adreno GPUs … Avoid integer modulo operation, which is
// expensive for Adreno GPUs." Host now dispatches with global=(L_out, K, C_in).
__kernel void im2col_1d(
    __global const storage_t* in,        // [C_in, L_in]
    __global       storage_t* out,       // [C_in * K, L_out]
    const int C_in,
    const int L_in,
    const int L_out,
    const int K,
    const int stride,
    const int padding,
    const int dilation) {

    const int col = get_global_id(0);
    const int k   = get_global_id(1);
    const int ic  = get_global_id(2);
    if (col >= L_out || k >= K || ic >= C_in) return;

    // Adreno §8.12: mul24 is one instruction vs multi-instruction full 32-bit mul.
    // ic*K+k < ~2^15, *L_out < ~2^25 — well within 24-bit signed range.
    const int row = mad24(ic, K, k);
    const int gid = mad24(row, L_out, col);

    const int src = mad24(col, stride, k * dilation) - padding;
    float v = 0.0f;
    if (src >= 0 && src < L_in) {
        v = (float)LOAD(in, mad24(ic, L_in, src));
    }
    STORE(out, gid, v);
}

// Fused leaky_relu(slope=0.1) + im2col. Slope is hardcoded — passing it as
// a float arg cost ~ms of dispatch overhead on Adreno. Same 3D NDRange as
// im2col_1d above for the same div/mod elimination win (§8.12).
// Vectorized 4-wide leaky_im2col per Qualcomm §10.3.3 + §10.3.6.
// Each workitem processes 4 consecutive L_out positions.
// Benefits:
//   - 4x fewer memory transactions (vload_half4 = 128-bit = 4 halfs)
//   - 4x fewer workitems → each runs longer → better gap hiding
//   - Branchless leaky (no wave divergence)
// gws[0] should be ceil(L_out/4), not L_out.
__kernel void leaky_im2col_1d(
    __global const storage_t* in,        // [C_in, L_in]
    __global       storage_t* out,       // [C_in * K, L_out]
    const int C_in,
    const int L_in,
    const int L_out,
    const int K,
    const int stride,
    const int padding,
    const int dilation) {

    const int col4 = get_global_id(0);   // processes cols [col4*4 .. col4*4+3]
    const int k    = get_global_id(1);
    const int ic   = get_global_id(2);
    const int col  = col4 * 4;
    if (col >= L_out || k >= K || ic >= C_in) return;

    const int row = mad24(ic, K, k);
    const int base_out = mad24(row, L_out, col);
    const int in_base = mul24(ic, L_in);
    const int k_dil = mul24(k, dilation);

    // Process up to 4 consecutive output columns
    const int n = min(4, L_out - col);

    // When stride=1 (always for resblock convs), the 4 input positions
    // are consecutive: src, src+1, src+2, src+3. Use vload_half4.
    if (stride == 1 && n == 4) {
        const int src0 = col + k_dil - padding;
        // Check all 4 positions are in bounds
        if (src0 >= 0 && src0 + 3 < L_in) {
#ifdef USE_FP16
            float4 v = vload_half4(0, (__global const half*)in + in_base + src0);
#else
            float4 v = vload4(0, (__global const float*)in + in_base + src0);
#endif
            // Branchless leaky on all 4 values
            v = fmax(v, (float4)0.0f) + fmin(v, (float4)0.0f) * 0.1f;
#ifdef USE_FP16
            vstore_half4(v, 0, (__global half*)out + base_out);
#else
            vstore4(v, 0, (__global float*)out + base_out);
#endif
            return;
        }
    }

    // Scalar fallback for boundary / non-stride-1 cases
    for (int i = 0; i < n; ++i) {
        const int c = col + i;
        const int src = mad24(c, stride, k_dil) - padding;
        float v = 0.0f;
        if (src >= 0 && src < L_in) {
            v = (float)LOAD(in, in_base + src);
            v = fmax(v, 0.0f) + fmin(v, 0.0f) * 0.1f;
        }
        STORE(out, base_out + i, v);
    }
}

// Zero-stuff for ConvTranspose1d → Conv1d equivalence.
//
//   x_in  [C_in, L_in]    →   x_out[C_in, (L_in-1)*stride + 1]
//
// x_out[ic, t] = x_in[ic, t/stride] if t % stride == 0 else 0.
// One workitem per output element. No branch divergence beyond the modulo check.
__kernel void zero_stuff_1d(
    __global const storage_t* x_in,        // [C_in, L_in]
    __global       storage_t* x_out,       // [C_in, L_exp]
    const int C_in,
    const int L_in,
    const int L_exp,
    const int stride) {

    const int gid = get_global_id(0);
    const int total = C_in * L_exp;
    if (gid >= total) return;
    const int ic = gid / L_exp;
    const int t  = gid - ic * L_exp;
    float v = 0.0f;
    if ((t % stride) == 0) {
        const int t_in = t / stride;
        if (t_in < L_in) v = (float)LOAD(x_in, ic * L_in + t_in);
    }
    STORE(x_out, gid, v);
}

// Reorder a ConvTranspose1d weight (layout [C_in, C_out, K])
// into the equivalent Conv1d weight (layout [C_out, C_in, K]) WITH the
// kernel axis reversed so that direct convolution on the zero-stuffed
// input matches PyTorch ConvTranspose1d output.
//
//   w_conv[oc, ic, k_conv] = w_convT[ic, oc, K-1-k_conv]
__kernel void reorder_convT_to_conv1d_weight(
    __global const storage_t* w_in,        // [C_in, C_out, K]
    __global       storage_t* w_out,       // [C_out, C_in, K]
    const int C_in,
    const int C_out,
    const int K) {

    const int gid = get_global_id(0);
    const int total = C_out * C_in * K;
    if (gid >= total) return;
    const int k  = gid % K;
    const int ic = (gid / K) % C_in;
    const int oc = gid / (K * C_in);

    const int k_in = K - 1 - k;
    const int src  = ic * (C_out * K) + oc * K + k_in;
    STORE(w_out, gid, (float)LOAD(w_in, src));
}

// Add a [C_out] bias broadcast across L_out to a [C_out, L_out] tensor in-place.
// Used after CLBlast HGemm because Gemm does not have a bias-add path.
// Step C.c: bias promoted to __constant memory (Adreno guide §6.4 p.46).
// "the buffer foo can be promoted to the fast on-chip constant memory if the
// compiler can determine that its size, as specified via the max_constant_size
// attribute, does not exceed the available constant memory. … the content in
// constant memory can broadcast into ALUs in no time for fast ALU computing.
// All other memories (global, local, and private) must go through the lengthy
// load/store path." Vocoder/flow biases are ≤ 1024 fp16 elements = 2048 bytes.
__kernel void add_bias_broadcast(
    __global       storage_t* y,         // [C_out, L_out]
    __constant     storage_t* bias       // [C_out]
        __attribute__((max_constant_size(2048))),
    const int C_out,
    const int L_out) {

    const int gid = get_global_id(0);
    const int total = C_out * L_out;
    if (gid >= total) return;
    const int oc = gid / L_out;
    const float v = (float)LOAD(y, gid) + (float)LOAD(bias, oc);
    STORE(y, gid, v);
}

// Fused bias broadcast + residual add. For the HiFi-GAN resblock final conv
// per kpair we always do bias + residual; one kernel replaces two
// (add_bias_broadcast + elementwise_add).
__kernel void add_bias_broadcast_resid(
    __global       storage_t* y,         // [C_out, L_out]
    __constant     storage_t* bias       // [C_out]
        __attribute__((max_constant_size(2048))),
    __global const storage_t* resid,     // [C_out, L_out] (too large for __constant)
    const int C_out,
    const int L_out) {

    const int gid = get_global_id(0);
    const int total = C_out * L_out;
    if (gid >= total) return;
    const int oc = gid / L_out;
    const float v = (float)LOAD(y, gid) + (float)LOAD(bias, oc) + (float)LOAD(resid, gid);
    STORE(y, gid, v);
}

// Step C — Fused im2col + bias-init for HGemm beta=1.
//
// Each conv1d_gemm call today does 3 dispatches: im2col → HGemm → bias_add.
// On Adreno 620 each dispatch costs ~70 ms of gap time, so the bias_add alone
// is ~5 s of wall time across 77 calls per inference. By having THIS kernel
// also pre-fill the HGemm output buffer C with bias_broadcast (+ optional
// residual), we can call HGemm with beta=1 and SKIP the trailing bias kernel.
// 3 dispatches per conv → 2.
//
// Workitem mapping: same 3D NDRange (L_out, K, C_in) as leaky_im2col_1d.
// The bias-init pass is done by workitems with k==0 and ic<C_out — one
// workitem per (oc=ic, col) cell of C, covering the full [C_out, L_out] grid
// exactly ONCE. **This requires C_in >= C_out** (caller falls back to the
// 3-dispatch path when that doesn't hold — currently only `conv_pre` where
// C_in=192 < C_out=512; conv_pre is called once per inference, irrelevant).
//
// Adreno guide §8.1 (p.63): "Fuse multiple kernels into one kernel … if data
// traffic can be reduced with good parallelization."
// Fused im2col + bias-init. Branchless leaky (§10.3.6).
// gws[0] = L_out (1 element per workitem — vectorization hurts on Adreno 620
// due to boundary-check divergence outweighing the bandwidth gain).
__kernel void im2col_1d_fused_bias(
    __global const storage_t* in,           // [C_in, L_in]
    __global       storage_t* out_im2col,   // [C_in * K, L_out]
    __global       storage_t* out_C,        // [C_out, L_out]
    __constant     storage_t* bias          // [C_out]
        __attribute__((max_constant_size(2048))),
    __global const storage_t* resid,        // [C_out, L_out]
    const int C_in,
    const int C_out,
    const int L_in,
    const int L_out,
    const int K,
    const int stride,
    const int padding,
    const int dilation,
    const int leaky_in,
    const int has_bias,
    const int has_resid) {

    const int col = get_global_id(0);
    const int k   = get_global_id(1);
    const int ic  = get_global_id(2);
    if (col >= L_out || k >= K || ic >= C_in) return;

    const int row = mad24(ic, K, k);
    const int gid_im = mad24(row, L_out, col);
    const int src = mad24(col, stride, mul24(k, dilation)) - padding;
    float v = 0.0f;
    if (src >= 0 && src < L_in) {
        v = (float)LOAD(in, mad24(ic, L_in, src));
        if (leaky_in) v = fmax(v, 0.0f) + fmin(v, 0.0f) * 0.1f;
    }
    STORE(out_im2col, gid_im, v);

    if (k == 0 && ic < C_out) {
        const int gid_C = mad24(ic, L_out, col);
        float c_val = 0.0f;
        if (has_bias)  c_val += (float)LOAD(bias, ic);
        if (has_resid) c_val += (float)LOAD(resid, gid_C);
        STORE(out_C, gid_C, c_val);
    }
}
