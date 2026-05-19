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

// OpenCL implementation of HuggingFace Transformers `PytorchGELUTanh`.
// Reference forward():
//   return nn.functional.gelu(input, approximate="tanh")
// (transformers/activations.py:28-46 in the user's reference snippet)
//
// This kernel computes the tanh-approximate GELU elementwise:
//   gelu_tanh(x) = 0.5 * x * (1 + tanh( sqrt(2/pi) * (x + 0.044715*x^3) ))
//
// Notes:
// - Accumulation is not applicable (pure pointwise op), but we still compute in float
//   for numerical stability even when storage is fp16.

__kernel void gelu_tanh_forward(
    __global const storage_t* x,
    __global storage_t* out,
    int n_elements)
{
    const int gid = (int)get_global_id(0);
    if (gid >= n_elements) return;

    const float xf = (float)LOAD(x, gid);

    // Constants from the standard tanh GELU approximation used by PyTorch.
    const float kAlpha = 0.7978845608028654f;   // sqrt(2/pi)
    const float kBeta  = 0.044715f;

    const float x3 = xf * xf * xf;
    const float inner = kAlpha * (xf + kBeta * x3);
    const float t = tanh(inner);
    const float yf = 0.5f * xf * (1.0f + t);

    STORE(out, gid, (storage_t)yf);
}
