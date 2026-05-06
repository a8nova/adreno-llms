// Reference: model_info/transformers_src/modeling_llama.py:375-428 LlamaModel.forward (embed_tokens)
//
// LLaMA-family embedding: token embedding lookup ONLY (no absolute position embeddings).
// Auto-generated scaffold originally added a wpe add; that is incorrect for RoPE-based models
// and breaks SxS at the very first layer.
//
// Embedding kernel: token embedding lookup.
// Each work-item handles one sequence position (relative to the dispatch).
//
// start_pos contract: the absolute position of input_ids[0] in the full
// sequence. Prefill passes 0; decode passes the count of tokens already
// processed (prompt_len + decode_step). The wpe lookup uses
// (start_pos + relative_pos); the output buffer is laid out per relative
// position. Without start_pos the decode pass reads wpe[0] for every
// generated token regardless of its absolute position — every post-prefill
// token is then wrong. Required for any model with absolute position
// embeddings (GPT-2, GPT-Neo, OPT, BERT, BART, ...). RoPE-based models
// (Llama, Mistral, Qwen) don't have wpe and apply position info in
// attention's RoPE step instead — for them the agent rewrites this kernel
// to drop the wpe lookup; the start_pos arg can stay for signature
// compatibility (and is harmless when unused).
//
// Uniform storage rule (fp16Standards.ts): wte and wpe are storage_t under
// the active build dtype (half under fp16, float under fp32). The kernel
// reads via LOAD, lifts to fp32 for the add, then STOREs as storage_t.
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

__kernel void embedding_forward(
    __global const int* token_ids,
    __global const storage_t* wte,
    __global const storage_t* wpe_unused,
    __global storage_t* output,
    int hidden_size,
    int start_pos) {
    (void)wpe_unused;
    (void)start_pos;

    const int rel_pos = (int)get_global_id(0);
    if (rel_pos < 0) return;

    const int tok = token_ids[rel_pos];
    if (tok < 0) return;

    const int base_tok = tok * hidden_size;
    const int base_out = rel_pos * hidden_size;

    for (int i = 0; i < hidden_size; i++) {
        float v = (float)LOAD(wte, base_tok + i);
        STORE(output, base_out + i, v);
    }
}
