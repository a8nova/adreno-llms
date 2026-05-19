// Splice image tokens into the LM embedding sequence — VLM port load-bearing.
//
// HF VLM convention (Idefics3/SmolVLM, LLaVA-NeXT, Qwen-VL, PaliGemma):
// the processor PRE-EXPANDS the image placeholder into the input_ids — there
// are exactly N image_token_id positions in input_ids, and the projector
// emits exactly N image-feature vectors. Splice REPLACES the embedding at
// each placeholder position with the matching image-feature vector.
//
//   text_embeds[positions[i], :] = image_features[i, :]   for i in [0, N)
//
// Sequence length is UNCHANGED (T_out == T_in); we mutate in place.
//
// This is NOT the legacy LLaVA-1 "single <image> token expands to N features"
// semantics (which was T_out = T_in - 1 + N). That pattern only applies to
// processors whose chat template emits a single placeholder. The modern HF
// VLMs all pre-expand; assume in-place multi-position unless the model's
// reference_tokens.json::num_image_tokens == 1.
//
// Dtype-template preamble — DO NOT EDIT. Driven by host-side -DNNOPT_USE_FP16.
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


__kernel void splice_image_tokens(
    __global storage_t* text_embeds,           // [T_in, D] — modified in place
    __global const storage_t* image_features,  // [N, D]
    __global const int* positions,             // [N] — placeholder indices into text_embeds rows
    const int N,
    const int D) {

    int i = get_global_id(0);  // which image-feature row (0..N)
    int d = get_global_id(1);  // hidden dim
    if (i >= N || d >= D) return;

    int pos = positions[i];
    float v = LOAD(image_features, i * D + d);
    STORE(text_embeds, pos * D + d, v);
}
