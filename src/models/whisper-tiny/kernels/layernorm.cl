// LayerNorm kernel.
// Reference: model_info/transformers_src/modeling_whisper.py:~280-520 WhisperEncoderLayer/WhisperDecoderLayer use nn.LayerNorm
// Also matches torch.nn.functional.layer_norm.

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

__kernel void layernorm_forward(
    __global const storage_t* x,
    __global const storage_t* gamma,
    __global const storage_t* beta,
    __global storage_t* y,
    const int rows,
    const int cols,
    const float eps)
{
    int row = (int)get_global_id(0);
    if (row >= rows) return;

    // Welford variance for numerical stability (matches PyTorch behavior closely).
    float mean = 0.0f;
    float m2 = 0.0f;
    int count = 0;

    for (int c = 0; c < cols; ++c) {
        float v = (float)LOAD(x, row * cols + c);
        count += 1;
        float delta = v - mean;
        mean += delta / (float)count;
        float delta2 = v - mean;
        m2 += delta * delta2;
    }

    float var = m2 / (float)cols;
    float inv_std = rsqrt(var + eps);

    for (int c = 0; c < cols; ++c) {
        float xn = ((float)LOAD(x, row * cols + c) - mean) * inv_std;
        float g = (float)LOAD(gamma, c);
        float b = (float)LOAD(beta, c);
        STORE(y, row * cols + c, xn * g + b);
    }
}
