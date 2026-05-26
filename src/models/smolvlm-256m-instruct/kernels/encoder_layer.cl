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

/*
Idefics3EncoderLayer (transformers/models/idefics3/modeling_idefics3.py)

forward():
  residual = hidden_states
  hidden_states = layer_norm1(hidden_states)
  hidden_states = self_attn(hidden_states, attention_mask)
  hidden_states = residual + hidden_states
  residual = hidden_states
  hidden_states = layer_norm2(hidden_states)
  hidden_states = mlp(hidden_states)
  hidden_states = residual + hidden_states

All heavy ops are handled by existing kernels / CLBlast in this project:
  - LayerNorm: dispatched by the project's LayerNorm op (not implemented here).
  - Self-attention: q/k/v/out projections are matmuls (CLBlast) and attention
    softmax/masking is handled by the project's attention op.
  - MLP: fc1/fc2 are matmuls (CLBlast) and activation is handled by the
    project's activation op.

This file intentionally defines no additional kernels for Idefics3EncoderLayer.
It exists so the build system always has a compilable OpenCL source at
kernels/encoder_layer.cl.
*/
