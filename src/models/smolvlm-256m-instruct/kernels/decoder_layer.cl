// Reference: https://raw.githubusercontent.com/huggingface/transformers/main/src/transformers/models/llama/modeling_llama.py LlamaDecoderLayer.forward
// NOTE: Decoder layer heavy ops are dispatched in C++ via op_LlamaRMSNorm/op_LlamaSdpaAttention/op_LlamaMLP.
// This file provides only small glue kernels if needed by host wiring.

#ifdef USE_FP16
  #pragma OPENCL EXTENSION cl_khr_fp16 : enable
  typedef half storage_t;
  #define LOAD(p, i)     vload_half((i), (p))
  #define STORE(p, i, v) vstore_half((v), (i), (p))
#else
  typedef float storage_t;
  #define LOAD(p, i)     ((p)[(i)])
  #define STORE(p, i, v) ((p)[(i)] = (v))
#endif

__kernel void add_residual_inplace(
    __global storage_t* x,           // [n]
    __global const storage_t* res,   // [n]
    const int n) {
  const int i = (int)get_global_id(0);
  if (i >= n) return;
  const float a = (float)LOAD(x, i);
  const float b = (float)LOAD(res, i);
  STORE(x, i, (storage_t)(a + b));
}
