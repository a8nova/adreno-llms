// kernels/idefics3_model.cl
//
// Idefics3Model forward() (transformers/models/idefics3/modeling_idefics3.py)
// is a high-level orchestration wrapper that:
//   - optionally computes image_hidden_states via vision_model + connector
//   - merges image_hidden_states into token embeddings via masked_scatter
//   - calls the text_model transformer
//
// In this OpenCL port, those computations are handled by other kernels / ops:
//   - vision_model path: kernels/idefics3_vision_transformer.cl (+ sub-kernels)
//   - connector path: kernels/idefics3_connector.cl
//   - token embedding: kernels/embedding.cl
//   - image token splicing: kernels/splice_image_tokens.cl (project-specific)
//   - text_model transformer: llama decoder layer kernels + CLBlast matmuls
//
// Therefore, Idefics3Model itself requires no additional custom kernels.

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

// No kernels emitted for this class.
