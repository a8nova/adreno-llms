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

// MLP activation kernel for this project.
//
// Reference: Idefics3VisionMLP.forward
//   model_info/transformers_src/modeling_idefics3.py:274-278
//     hidden_states = self.fc1(hidden_states)
//     hidden_states = self.activation_fn(hidden_states)   where activation_fn = ACT2FN[config.hidden_act]
//     hidden_states = self.fc2(hidden_states)
//
// For Idefics3VisionConfig, hidden_act defaults to "gelu_pytorch_tanh"
// (configuration_idefics3.py:55). This repository's kernels implement the
// tanh-approx GeLU variant.
//
// Bias-add note (MANDATORY audit):
//   - Idefics3VisionMLP.fc1: nn.Linear(...), bias argument omitted => bias=True in PyTorch.
//   - Idefics3VisionMLP.fc2: nn.Linear(...), bias argument omitted => bias=True in PyTorch.
//   Bias-add is performed outside this kernel (e.g., via a separate bias_add kernel
//   after the GEMM). This kernel is activation-only.

__kernel void mlp_gelu(
    __global const storage_t* x,
    __global storage_t* out,
    const int total_elements) {
  const int i = (int)get_global_id(0);
  if (i >= total_elements) return;

  // Compute activation in fp32 even when storage is fp16.
  const float g = (float)LOAD(x, i);

  // GeLU tanh approximation ("gelu_pytorch_tanh" / "gelu_new"):
  // y = 0.5*x*(1 + tanh(sqrt(2/pi)*(x + 0.044715*x^3)))
  const float g2 = g * g;
  const float g3 = g2 * g;
  const float inner = 0.7978845608f * (g + 0.044715f * g3);
  const float y = 0.5f * g * (1.0f + tanh(inner));

  STORE(out, i, (storage_t)y);
}
