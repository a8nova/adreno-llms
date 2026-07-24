// Reference: kitten-tts-nano-0.1 ONNX graph /bert_encoder/* (Add bias after MatMul).
// Bias-add: out[r,c] = in[r,c] + bias[c], row-major [rows, cols].
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

__kernel void be_add_bias(
    __global const storage_t* in,
    __global const storage_t* bias,
    __global storage_t* out,
    const int rows,
    const int cols)
{
    int gid = get_global_id(0);
    int total = rows * cols;
    if (gid >= total) return;
    int c = gid % cols;
    float v = (float)LOAD(in, gid) + (float)LOAD(bias, c);
    STORE(out, gid, v);
}
