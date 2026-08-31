// SinkAttention.cpp — DeferredLocalDotProductSelfAttention core (single query frame).
// Reference: magenta_rt/mlx/attention.py LocalDotProductSelfAttention.__call__
//
// Computes attn_out [H*D] from input [in_dim]:
//   qkv = input @ qkv_proj.weight.T          (linear_f32_w16, out_dim = 3*H*D)
//   split q/k/v, per-head sink attention      (sink_attention_f32, kv = sink + self)
// Weights (weight_prefix = "...body.layers.1.inner"):
//   <wp>.qkv_proj.weight          [3*H*D, in_dim] fp16   (dequantized raw qkv_proj triplet)
//   <wp>._per_dim_scale           [D] fp16
//   <wp>.sink_key_embeddings      [1,H,D] fp16
//   <wp>.sink_value_embeddings    [1,H,D] fp16
// out_proj (EinsumDense) and the residual/post-norm are applied by the caller.

#include "../opencl_context.h"
#include "../weights.h"
#include "../debug_utils.h"
#include "../model_config.h"
#include "../utils.h"
#include <string>
#include <cmath>

extern "C" {
cl_mem SinkAttention_forward(
    OpenCLContext& cl_ctx,
    Weights& weights,
    cl_command_queue queue,
    cl_mem input,
    int seq_len,
    int layer_idx,
    int start_pos,
    cl_mem* k_cache_inout,
    cl_mem* v_cache_inout,
    cl_mem encoder_hidden_states,
    const char* weight_prefix)
{
    (void)layer_idx; (void)start_pos;
    (void)k_cache_inout; (void)v_cache_inout; (void)encoder_hidden_states;
    const std::string wp = weight_prefix ? std::string(weight_prefix) : std::string();
    if (wp.empty() || !input) { NNOPT_ERROR("SinkAttention: null wp/input"); return nullptr; }
    if (seq_len != 1) { NNOPT_ERROR_FMT("SinkAttention: seq_len=%d (only single-frame supported)", seq_len); return nullptr; }

    cl_mem pds   = weights.get_buffer(wp + "._per_dim_scale");
    // Sink is optional: depth-body self-attn has num_sink_embeddings=0.
    const bool has_sink = weights.has_tensor(wp + ".sink_key_embeddings");
    cl_mem sink_k = has_sink ? weights.get_buffer(wp + ".sink_key_embeddings") : nullptr;
    cl_mem sink_v = has_sink ? weights.get_buffer(wp + ".sink_value_embeddings") : nullptr;
    int qkv_out = 0, in_dim = 0;   // logical dims — a Q4 weight declares [N, K/2]
    if (!nnopt_weight_dims(weights, wp + ".qkv_proj.weight", &qkv_out, &in_dim)) return nullptr;
    int H, D;
    if (has_sink) {
        const std::vector<int> sksh = weights.get_shape(wp + ".sink_key_embeddings");  // [1,H,D]
        H = sksh[1]; D = sksh[2];
    } else {
        // derive heads from per_dim_scale length (= D); H = (qkv_out/3)/D
        const std::vector<int> psh = weights.get_shape(wp + "._per_dim_scale");
        D = psh.empty() ? 0 : psh.back();
        H = (D > 0) ? (qkv_out / 3) / D : 0;
    }
    if (D <= 0 || H <= 0 || qkv_out != 3 * H * D) { NNOPT_ERROR_FMT("SinkAttention: bad dims H=%d D=%d qkv_out=%d", H, D, qkv_out); return nullptr; }
    const float inv_sqrt_d = 1.0f / std::sqrt((float)D);

    cl_int err = CL_SUCCESS;
    // ── qkv = input @ qkv_w.T — fp16 or Q4 depending on the bundle (see utils.h) ──
    cl_mem qkv = nnopt_gemv(cl_ctx, weights, queue, input, wp + ".qkv_proj.weight",
                            1, in_dim, qkv_out, nullptr);
    if (!qkv) { NNOPT_ERROR("SinkAttention: qkv"); return nullptr; }

    // ── no-sink single position: attends only to itself ⇒ attn_out = v ──
    if (!has_sink) {
        cl_mem out = pool_alloc(cl_ctx.context(), CL_MEM_READ_WRITE, (size_t)H*D*sizeof(float), nullptr, &err);
        if (!out) { NNOPT_ERROR("SinkAttention: out alloc"); pool_free(qkv); return nullptr; }
        // Kernel copy — see the note in CrossAttention.cpp; a recording cannot capture a buffer copy.
        nnopt_copy_buffer(cl_ctx, qkv, out, 2*H*D, 0, H*D, wp, ".sink.v");
        
        pool_free(qkv);
        return out;
    }

    // ── sink attention core ──
    static cl_kernel sk = nullptr;
    if (!sk) {
        cl_program p = cl_ctx.build_program_from_file("kernels/sink_attention_f32.cl");
        if (!p) { NNOPT_ERROR("SinkAttention: build sink_attention_f32 failed"); pool_free(qkv); return nullptr; }
        sk = clCreateKernel(p, "sink_attention_f32", &err);
        if (err != CL_SUCCESS || !sk) { NNOPT_ERROR_FMT("SinkAttention: sink kernel (err=%d)", err); pool_free(qkv); return nullptr; }
    }
    cl_kernel ski = nnopt_kernel_instance(sk, wp, ".sink");
    cl_mem out = pool_alloc(cl_ctx.context(), CL_MEM_READ_WRITE, (size_t)H * D * sizeof(float), nullptr, &err);
    if (!out) { NNOPT_ERROR("SinkAttention: out alloc"); pool_free(qkv); return nullptr; }
    clSetKernelArg(ski, 0, sizeof(cl_mem), &qkv);
    clSetKernelArg(ski, 1, sizeof(cl_mem), &sink_k);
    clSetKernelArg(ski, 2, sizeof(cl_mem), &sink_v);
    clSetKernelArg(ski, 3, sizeof(cl_mem), &pds);
    clSetKernelArg(ski, 4, sizeof(cl_mem), &out);
    clSetKernelArg(ski, 5, sizeof(int), &H);
    clSetKernelArg(ski, 6, sizeof(int), &D);
    clSetKernelArg(ski, 7, sizeof(float), &inv_sqrt_d);
    size_t sg = (size_t)H;
    // profEnqueue, and the per-site INSTANCE: a raw clEnqueueNDRangeKernel on the shared `sk` both
    // skipped the profiler and re-armed a kernel object a recording may already reference.
    err = cl_ctx.profEnqueue(ski, 1, &sg, nullptr, "sink_attn");
    if (err != CL_SUCCESS) { NNOPT_ERROR_FMT("SinkAttention: sink enqueue (err=%d)", err); pool_free(qkv); pool_free(out); return nullptr; }
    
    pool_free(qkv);
    return out;
}
}
