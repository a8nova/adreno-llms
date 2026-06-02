// Bias add kernel for [rows, cols] row-major.
// Reference: model_info/transformers_src/modeling_whisper.py (nn.Linear includes bias for q/v/out_proj and fc1/fc2)

#ifdef USE_FP16
  #pragma OPENCL EXTENSION cl_khr_fp16 : enable
  typedef half storage_t;
  #define LOAD(p,i)    vload_half((i),(p))
  #define STORE(p,i,v) vstore_half((v),(i),(p))
#else
  typedef float storage_t;
  #define LOAD(p,i)    ((p)[(i)])
  #define STORE(p,i,v) ((p)[(i)] = (v))
#endif

__kernel void bias_add_rows(
    __global storage_t* y,
    __global const storage_t* bias,
    const int rows,
    const int cols)
{
    int gid = (int)get_global_id(0);
    int total = rows * cols;
    if (gid >= total) return;
    int c = gid - (gid / cols) * cols;
    float v = (float)LOAD(y, gid) + (float)LOAD(bias, c);
    STORE(y, gid, v);
}
