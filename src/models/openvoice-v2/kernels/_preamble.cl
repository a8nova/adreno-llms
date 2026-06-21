// Shared preamble — concatenated FIRST. Defines storage dtype + LOAD/STORE/WLOAD
// and the tiling constants. Driven by host -DSTORE_FP32. No kernels here.
// OpenVoice V2 ToneColorConverter — OpenCL kernels (manual port).
// Storage dtype is parametrized: default half (fp16); -DSTORE_FP32 → float.
// ALL accumulation is float (fp32) regardless. Layout NCHW-1d: [C,T] row-major, batch=1.
#pragma OPENCL EXTENSION cl_khr_fp16 : enable

#ifdef STORE_FP32
  #define REAL float
  #define LOAD(i, p)     ((p)[(i)])
  #define STORE(v, i, p) ((p)[(i)] = (v))
#else
  #define REAL half
  #define LOAD(i, p)     vload_half((i), (p))
  #define STORE(v, i, p) vstore_half((v), (i), (p))
#endif

// Weights are ALWAYS fp16 (true mixed precision: fp16 weights + REAL activations).
// Halves weight memory vs fp32 activations; matches the recommended nnopt config.
#define WLOAD(i, p) vload_half((i), (p))

// Tiling constants (kept in sync with host Engine::TILE_T / NC).
// NC is guarded so a -DNC=<n> build flag (host NNOPT_NC re-tune) takes precedence.
#define TILE_T 8
#ifndef NC
#define NC 2
#endif
