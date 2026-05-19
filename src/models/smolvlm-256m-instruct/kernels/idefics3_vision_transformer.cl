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
  Idefics3VisionTransformer (transformers/models/idefics3/modeling_idefics3.py)

  The forward() for this class is a thin orchestration wrapper:
    - builds a patch_attention_mask if missing (torch.ones + dtype/bool cast)
    - calls self.embeddings(...)
    - reshapes mask and calls create_bidirectional_mask(...)
    - calls self.encoder(...)
    - applies self.post_layernorm (nn.LayerNorm)

  In this OpenCL port, the heavy ops are handled elsewhere:
    - embeddings path has its own kernels (see kernels/idefics3_vision_embeddings.cl)
    - encoder layers (attention/MLP/norm) have their own kernels and CLBlast matmuls
      (see kernels/encoder_layer.cl, kernels/vision_attention.cl, kernels/layer_norm.cl, etc.)

  Therefore, no additional custom kernels are required specifically for
  Idefics3VisionTransformer beyond what the submodules already provide.

  This file intentionally contains only the required dtype-templating preamble
  so it compiles cleanly even though it defines no kernels.
*/
