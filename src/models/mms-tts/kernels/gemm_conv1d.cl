// gemm_conv1d.cl — fused im2col + GEMM + optional leaky ReLU + optional bias
// + optional residual add, dispatched via clEnqueueNDRangeKernel.
// Replaces CLBlast's clblast::Gemm for the vocoder's conv1d path so the
// entire vocoder can be recorded with cl_qcom_recordable_queues.
//
// C[m, n] = bias[m] + sum_k A[m, k] * im2col(B, k, n)
// where im2col(B, ci*K+kk, t) = leaky_opt(B[ci, t + kk*dilation - padding])
//
// Layout: A = weights [M, K_full] row-major, B = input [C_in, L] channel-first
// K_full = C_in * K_conv
//
// One workitem per output element (m, n). Inner loop over K_full.
// For fp16: vload_half4 + fp32 accumulation → vstore_half result.

#ifdef USE_FP16
  #pragma OPENCL EXTENSION cl_khr_fp16 : enable
  typedef half storage_t;
  #define LOAD(p,i)   vload_half((i),(p))
  #define STORE(p,i,v) vstore_half((v),(i),(p))
  #define LOAD4(p,i)  vload_half4((i),(p))
#else
  typedef float storage_t;
  #define LOAD(p,i)   ((p)[(i)])
  #define STORE(p,i,v) ((p)[(i)] = (v))
  #define LOAD4(p,i)  vload4((i),(p))
#endif

#define LEAKY_SLOPE 0.1f

__kernel void gemm_conv1d_fused(
    __global const storage_t* W,        // [M, K_full] weights
    __global const storage_t* B,        // [C_in, L_in] input (channel-first)
    __global const storage_t* bias,     // [M] or dummy
    __global const storage_t* resid,    // [M, L_out] or dummy
    __global storage_t*       C_out,    // [M, L_out] output
    const int M,
    const int K_full,       // = C_in * K_conv
    const int N,            // = L_out
    const int C_in,
    const int K_conv,
    const int L_in,
    const int stride,
    const int padding,
    const int dilation,
    const int has_bias,
    const int has_leaky,
    const int has_resid)
{
    const int m = get_global_id(0);   // output channel
    const int n = get_global_id(1);   // output time step
    if (m >= M || n >= N) return;

    float acc = has_bias ? (float)LOAD(bias, m) : 0.0f;

    // W[m, k] for k in 0..K_full-1 is contiguous in memory at offset m*K_full.
    // im2col(B, k, n): k = ci*K_conv + kk → B[ci, n*stride + kk*dilation - padding]
    const int w_base = m * K_full;

    for (int ci = 0; ci < C_in; ++ci) {
        const int b_ci_base = ci * L_in;
        const int w_ci_base = w_base + ci * K_conv;
        for (int kk = 0; kk < K_conv; ++kk) {
            const int t_in = n * stride + kk * dilation - padding;
            float bv = 0.0f;
            if (t_in >= 0 && t_in < L_in) {
                bv = (float)LOAD(B, b_ci_base + t_in);
                if (has_leaky) bv = bv < 0.0f ? LEAKY_SLOPE * bv : bv;
            }
            acc += (float)LOAD(W, w_ci_base + kk) * bv;
        }
    }

    if (has_resid) {
        acc += (float)LOAD(resid, m * N + n);
    }

    STORE(C_out, m * N + n, acc);
}

// Simpler version without im2col for ConvTranspose1d post-GEMM or plain GEMM.
// C[m, n] = alpha * sum_k A[m, k] * B[k, n] + beta * C[m, n]
// A = [M, K] row-major, B = [K, N] row-major, C = [M, N] row-major
__kernel void gemm_nn(
    __global const storage_t* A,   // [M, K]
    __global const storage_t* B_mat, // [K, N]
    __global storage_t*       C,   // [M, N]
    const int M, const int N, const int K,
    const float alpha, const float beta)
{
    const int m = get_global_id(0);
    const int n = get_global_id(1);
    if (m >= M || n >= N) return;

    float acc = 0.0f;
    const int a_base = m * K;
    for (int k = 0; k < K; ++k) {
        acc += (float)LOAD(A, a_base + k) * (float)LOAD(B_mat, k * N + n);
    }

    float c_val = beta != 0.0f ? beta * (float)LOAD(C, m * N + n) : 0.0f;
    STORE(C, m * N + n, alpha * acc + c_val);
}
