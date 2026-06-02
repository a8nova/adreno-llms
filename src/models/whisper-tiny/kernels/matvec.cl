// Matrix-vector product for the M=1 decode path.   (opt: custom decode kernels)
// Reference: replaces CLBlast Gemm when M==1. CLBlast's GemmStridedBatched is
// tuned for the encoder's large M (1500); at M=1 each call is a tiny
// matrix-vector product whose compute is dwarfed by CLBlast's per-call fixed
// overhead (~1.4-2.9ms measured per M=1 GEMM). A direct matvec removes that.
//
// Semantics match pytorch_linear's CLBlast call exactly:
//   out[1,N] = x[1,K] @ W[N,K]^T   ->   y[n] = sum_k x[k] * W[n,k]
// W is the nn.Linear weight [N,K] row-major; x is [K]; y is [N].
// One work-item per output row n; fp32 accumulation (matches CLBlast Hgemm).

// Dtype-template preamble — DO NOT EDIT. Driven by host-side -D USE_FP16.
#ifdef USE_FP16
  #pragma OPENCL EXTENSION cl_khr_fp16 : enable
  typedef half storage_t;
  #define LOAD(p, i)    vload_half((i), (p))
  #define STORE(p, i, v) vstore_half((v), (i), (p))
#else
  typedef float storage_t;
  #define LOAD(p, i)    ((p)[(i)])
  #define STORE(p, i, v) ((p)[(i)] = (v))
#endif

__kernel void matvec_nk(
    __global const storage_t* x,   // [K]
    __global const storage_t* W,   // [N, K] row-major
    __global storage_t* y,         // [N]
    int N,
    int K)
{
    int n = (int)get_global_id(0);
    if (n >= N) return;
    int wb = n * K;
    float acc = 0.0f;
    int k = 0;
    for (; k + 8 <= K; k += 8) {
        acc += (float)LOAD(x, k + 0) * (float)LOAD(W, wb + k + 0)
             + (float)LOAD(x, k + 1) * (float)LOAD(W, wb + k + 1)
             + (float)LOAD(x, k + 2) * (float)LOAD(W, wb + k + 2)
             + (float)LOAD(x, k + 3) * (float)LOAD(W, wb + k + 3)
             + (float)LOAD(x, k + 4) * (float)LOAD(W, wb + k + 4)
             + (float)LOAD(x, k + 5) * (float)LOAD(W, wb + k + 5)
             + (float)LOAD(x, k + 6) * (float)LOAD(W, wb + k + 6)
             + (float)LOAD(x, k + 7) * (float)LOAD(W, wb + k + 7);
    }
    for (; k < K; ++k) acc += (float)LOAD(x, k) * (float)LOAD(W, wb + k);
    STORE(y, n, acc);
}
