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
backbone.cl — LlamaModel top-level forward

The reference LlamaModel.forward (transformers/models/llama/modeling_llama.py) is a pure orchestration wrapper:
  - token embedding lookup (embed_tokens)
  - causal mask update (_update_causal_mask)
  - rotary position embeddings (rotary_emb)
  - loop over decoder layers (self.layers[i](...))
  - final RMSNorm (self.norm)

All compute-heavy ops are inside submodules (attention/MLP/norm/embedding) and are dispatched by the host via
CLBlast matmuls and the pre-characterized kernels for those layers. This file intentionally defines no custom
kernels for LlamaModel itself.

It must still compile as OpenCL 1.2 for build systems that always compile kernels/backbone.cl.
*/
