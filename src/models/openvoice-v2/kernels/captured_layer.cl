// Reference: /Users/alazarshenkute/.nnopt/repos/OpenVoice/openvoice/models.py:Generator.forward / PosteriorEncoder.forward scaffolding placeholder
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

__kernel void copy_storage(__global const storage_t* input,
                           __global storage_t* output,
                           const int n) {
    int gid = get_global_id(0);
    if (gid >= n) return;
    STORE(output, gid, LOAD(input, gid));
}
