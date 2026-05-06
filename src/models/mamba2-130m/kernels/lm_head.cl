// Reference: model_info/transformers_src/modeling_mamba2.py (Mamba2ForCausalLM.forward uses lm_head linear)
// Note: lm_head is a pure linear projection; this file exists only to satisfy
// the "kernels/*.cl first" convention in the porting workflow. No kernels.
