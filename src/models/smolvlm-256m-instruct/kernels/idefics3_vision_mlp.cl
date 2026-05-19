// Reference: model_info/transformers_src/modeling_idefics3.py:266-278 Idefics3VisionMLP
//   - __init__: activation_fn = ACT2FN[config.hidden_act]; fc1 = nn.Linear(...); fc2 = nn.Linear(...)
//   - forward: hidden_states = fc1(hidden_states); hidden_states = activation_fn(hidden_states); hidden_states = fc2(hidden_states)
// Reference: model_info/transformers_src/configuration_idefics3.py:55 Idefics3VisionConfig.hidden_act = "gelu_pytorch_tanh"
//
// This file emits ONLY the pointwise activation stage for the vision MLP.
// The fc1/fc2 matmuls (and their bias-adds) are dispatched elsewhere (CLBlast + bias kernels).

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

// GeLU tanh approximation ("gelu_pytorch_tanh") in fp32:
// y = 0.5 * x * (1 + tanh(sqrt(2/pi) * (x + 0.044715*x^3)))
__kernel void mlp_gelu(
    __global const storage_t* x,
    __global storage_t* out,
    const int total_elements) {
  const int i = (int)get_global_id(0);
  if (i >= total_elements) return;

  const float g = (float)LOAD(x, i);

  const float g2 = g * g;
  const float g3 = g2 * g;
  const float inner = 0.7978845608f * (g + 0.044715f * g3);
  const float y = 0.5f * g * (1.0f + tanh(inner));

  STORE(out, i, (storage_t)y);
}
