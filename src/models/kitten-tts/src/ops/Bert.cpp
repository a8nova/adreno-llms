// Reference: kitten-tts-nano-0.1 ONNX graph /bert/* (ALBERT phoneme encoder).
// See .nnport/onnx_graph_spec.json. ALBERT structure (12 weight-shared layers):
//   emb   = word_emb[ids] + token_type_emb[0] + position_emb[0..T]
//   emb   = LayerNorm(emb, embeddings.LayerNorm)                          [T,128]
//   h0    = emb @ embedding_hidden_mapping_in.W + bias                    [T,768]
//   repeat 12x (weight-shared):
//     q = h@Wq+bq; k = h@Wk+bk; v = h@Wv+bv    (768->768)
//     ctx = MHSA(q,k,v, 12 heads, head_dim 64, non-causal)
//     a = ctx @ Wdense + b_dense                                          [T,768]
//     h = LayerNorm(h + a, attention.LayerNorm)                          (residual)
//     f = gelu_new(h @ Wffn + b_ffn)            (768->2048)
//     f = f @ Wffn_out + b_ffn_out              (2048->768)
//     h = LayerNorm(h + f, full_layer_layer_norm)                       (residual)
//   boundary dump = full_layer_layer_norm output of the 12th layer.       [T,768]
//
// Weights (ONNX MatMul layout [in,out] => pytorch_conv1d; nn-style [out]-bias):
//   kmodel.bert.embeddings.word_embeddings.weight        [178,128]
//   kmodel.bert.embeddings.token_type_embeddings.weight  [2,128]
//   kmodel.bert.embeddings.position_embeddings.weight    [512,128]
//   kmodel.bert.embeddings.LayerNorm.{weight,bias}       [128]
//   onnx_MatMul_7606_quantized (pre-dequantized fp) [128,768]  hidden_mapping W
//   kmodel.bert.encoder.embedding_hidden_mapping_in.bias [768]
//   onnx_MatMul_7607/7610/7613 (q/k/v)  [768,768]
//   kmodel.bert...attention.{query,key,value}.bias        [768]
//   onnx_MatMul_7617 (dense)  [768,768] ; attention.dense.bias [768]
//   kmodel.bert...attention.LayerNorm.{weight,bias}       [768]
//   onnx_MatMul_7618 (ffn)    [768,2048] ; ffn.bias [2048]
//   onnx_MatMul_7619 (ffn_out)[2048,768] ; ffn_output.bias [768]
//   kmodel.bert...full_layer_layer_norm.{weight,bias}     [768]

#include "../opencl_context.h"
#include "../weights.h"
#include "../nnopt_error.h"
#include "../model_config.h"
#include "../utils.h"
#include "../profiler.h"
#include <string>
#include <cmath>

namespace {
constexpr int H_EMB    = 128;   // embedding dim
constexpr int H        = 768;   // encoder hidden
constexpr int FFN      = 2048;
constexpr int N_HEADS  = 12;
constexpr int HEAD_DIM = 64;
constexpr int N_LAYERS = 12;
constexpr float LN_EPS = 1e-12f;  // ALBERT LayerNorm eps

cl_program g_prog = nullptr;
cl_kernel g_k_embed = nullptr;
cl_kernel g_k_ln = nullptr;
cl_kernel g_k_gelu = nullptr;
cl_kernel g_k_bias = nullptr;
cl_kernel g_k_resadd = nullptr;
cl_kernel g_k_attn = nullptr;

bool ensure_program(OpenCLContext& cl_ctx) {
    if (g_prog) return true;
    g_prog = cl_ctx.build_program_from_file("kernels/bert.cl");
    if (!g_prog) { NNOPT_ERROR("Bert: failed to build kernels/bert.cl"); return false; }
    cl_int e;
    g_k_embed  = clCreateKernel(g_prog, "bert_embed_sum", &e);   if (e!=CL_SUCCESS){NNOPT_ERROR_FMT("kernel bert_embed_sum %d",e);return false;}
    g_k_ln     = clCreateKernel(g_prog, "bert_layernorm", &e);   if (e!=CL_SUCCESS){NNOPT_ERROR_FMT("kernel bert_layernorm %d",e);return false;}
    g_k_gelu   = clCreateKernel(g_prog, "bert_gelu_new", &e);    if (e!=CL_SUCCESS){NNOPT_ERROR_FMT("kernel bert_gelu_new %d",e);return false;}
    g_k_bias   = clCreateKernel(g_prog, "bert_add_bias", &e);    if (e!=CL_SUCCESS){NNOPT_ERROR_FMT("kernel bert_add_bias %d",e);return false;}
    g_k_resadd = clCreateKernel(g_prog, "bert_residual_add", &e);if (e!=CL_SUCCESS){NNOPT_ERROR_FMT("kernel bert_residual_add %d",e);return false;}
    g_k_attn   = clCreateKernel(g_prog, "bert_attention", &e);   if (e!=CL_SUCCESS){NNOPT_ERROR_FMT("kernel bert_attention %d",e);return false;}
    return true;
}

cl_mem alloc(OpenCLContext& cl_ctx, size_t n) {
    cl_int e;
    cl_mem m = clCreateBuffer(cl_ctx.context(), CL_MEM_READ_WRITE, n * sizeof(nnopt_storage_t), nullptr, &e);
    if (e != CL_SUCCESS) { NNOPT_ERROR_FMT("Bert alloc %zu err %d", n, e); return nullptr; }
    return m;
}

// out[rows,cols] = in @ W + bias, W stored ONNX-MatMul [K,N] => pytorch_conv1d.
bool linear_bias(cl_command_queue q, cl_program utils, int rows, int N, int K,
                 cl_mem in, cl_mem W, cl_mem bias, cl_mem out, cl_kernel k_bias) {
    if (!pytorch_conv1d(q, rows, N, K, in, W, out)) return false;
    if (bias) {
        cl_int e = CL_SUCCESS;
        e |= clSetKernelArg(k_bias, 0, sizeof(cl_mem), &out);
        e |= clSetKernelArg(k_bias, 1, sizeof(cl_mem), &bias);
        e |= clSetKernelArg(k_bias, 2, sizeof(cl_mem), &out);
        e |= clSetKernelArg(k_bias, 3, sizeof(int), &rows);
        e |= clSetKernelArg(k_bias, 4, sizeof(int), &N);
        if (e != CL_SUCCESS) { NNOPT_ERROR_FMT("linear_bias setarg %d", e); return false; }
        size_t gws = (size_t)rows * N;
        e = clEnqueueNDRangeKernel(q, k_bias, 1, nullptr, &gws, nullptr, 0, nullptr, KernelProfiler::event_for("bert_add_bias"));
        if (e != CL_SUCCESS) { NNOPT_ERROR_FMT("linear_bias dispatch %d", e); return false; }
    }
    (void)utils;
    return true;
}

bool layernorm(cl_command_queue q, int rows, int cols, cl_mem in, cl_mem g, cl_mem b, cl_mem out, cl_kernel k) {
    float eps = LN_EPS;
    cl_int e = CL_SUCCESS;
    e |= clSetKernelArg(k, 0, sizeof(cl_mem), &in);
    e |= clSetKernelArg(k, 1, sizeof(cl_mem), &g);
    e |= clSetKernelArg(k, 2, sizeof(cl_mem), &b);
    e |= clSetKernelArg(k, 3, sizeof(cl_mem), &out);
    e |= clSetKernelArg(k, 4, sizeof(int), &rows);
    e |= clSetKernelArg(k, 5, sizeof(int), &cols);
    e |= clSetKernelArg(k, 6, sizeof(float), &eps);
    if (e != CL_SUCCESS) { NNOPT_ERROR_FMT("layernorm setarg %d", e); return false; }
    size_t gws = (size_t)rows;
    e = clEnqueueNDRangeKernel(q, k, 1, nullptr, &gws, nullptr, 0, nullptr, KernelProfiler::event_for("bert_layernorm"));
    if (e != CL_SUCCESS) { NNOPT_ERROR_FMT("layernorm dispatch %d", e); return false; }
    return true;
}
}  // namespace

extern "C" {
cl_mem Bert_forward(
    OpenCLContext& cl_ctx,
    Weights& weights,
    cl_command_queue queue,
    cl_mem input,          // int32 token ids [T]
    int seq_len,
    int layer_idx,
    int start_pos,
    cl_mem* k_cache_inout,
    cl_mem* v_cache_inout,
    cl_mem encoder_hidden_states,
    const char* weight_prefix)
{
    (void)layer_idx; (void)start_pos; (void)k_cache_inout; (void)v_cache_inout;
    (void)encoder_hidden_states; (void)weight_prefix;
    const int T = seq_len;
    if (!ensure_program(cl_ctx)) return nullptr;
    cl_program utils = cl_ctx.build_program_from_file("kernels/utils.cl"); // PROGRAM-INIT-OK (cached by ctx)

    // ── weights ──
    cl_mem w_word = weights.get_buffer("kmodel.bert.embeddings.word_embeddings.weight");
    cl_mem w_type = weights.get_buffer("kmodel.bert.embeddings.token_type_embeddings.weight");
    cl_mem w_pos  = weights.get_buffer("kmodel.bert.embeddings.position_embeddings.weight");
    cl_mem eln_w  = weights.get_buffer("kmodel.bert.embeddings.LayerNorm.weight");
    cl_mem eln_b  = weights.get_buffer("kmodel.bert.embeddings.LayerNorm.bias");
    cl_mem map_w  = weights.get_buffer("onnx_MatMul_7606_quantized");
    cl_mem map_b  = weights.get_buffer("kmodel.bert.encoder.embedding_hidden_mapping_in.bias");
    const std::string L = "kmodel.bert.encoder.albert_layer_groups.0.albert_layers.0.";
    cl_mem q_w = weights.get_buffer("onnx_MatMul_7607");
    cl_mem k_w = weights.get_buffer("onnx_MatMul_7610");
    cl_mem v_w = weights.get_buffer("onnx_MatMul_7613");
    cl_mem q_b = weights.get_buffer(L + "attention.query.bias");
    cl_mem k_b = weights.get_buffer(L + "attention.key.bias");
    cl_mem v_b = weights.get_buffer(L + "attention.value.bias");
    cl_mem d_w = weights.get_buffer("onnx_MatMul_7617");
    cl_mem d_b = weights.get_buffer(L + "attention.dense.bias");
    cl_mem aln_w = weights.get_buffer(L + "attention.LayerNorm.weight");
    cl_mem aln_b = weights.get_buffer(L + "attention.LayerNorm.bias");
    cl_mem ffn_w = weights.get_buffer("onnx_MatMul_7618");
    cl_mem ffn_b = weights.get_buffer(L + "ffn.bias");
    cl_mem fo_w  = weights.get_buffer("onnx_MatMul_7619");
    cl_mem fo_b  = weights.get_buffer(L + "ffn_output.bias");
    cl_mem fln_w = weights.get_buffer(L + "full_layer_layer_norm.weight");
    cl_mem fln_b = weights.get_buffer(L + "full_layer_layer_norm.bias");
    if (!w_word || !map_w || !q_w || !ffn_w || !fo_w) {
        NNOPT_ERROR("Bert: missing weights"); return nullptr;
    }

    // ── buffers ──
    cl_mem emb  = alloc(cl_ctx, (size_t)T * H_EMB);
    cl_mem embn = alloc(cl_ctx, (size_t)T * H_EMB);
    cl_mem h    = alloc(cl_ctx, (size_t)T * H);
    cl_mem q    = alloc(cl_ctx, (size_t)T * H);
    cl_mem k    = alloc(cl_ctx, (size_t)T * H);
    cl_mem v    = alloc(cl_ctx, (size_t)T * H);
    cl_mem ctx  = alloc(cl_ctx, (size_t)T * H);
    cl_mem attn = alloc(cl_ctx, (size_t)T * H);
    cl_mem res  = alloc(cl_ctx, (size_t)T * H);
    cl_mem ff   = alloc(cl_ctx, (size_t)T * FFN);
    cl_mem ffg  = alloc(cl_ctx, (size_t)T * FFN);
    cl_mem fout = alloc(cl_ctx, (size_t)T * H);
    auto cleanup = [&]() -> cl_mem {
        cl_mem all[] = {emb,embn,h,q,k,v,ctx,attn,res,ff,ffg,fout};
        for (cl_mem m : all) if (m) clReleaseMemObject(m);
        return nullptr;
    };
    if (!emb||!embn||!h||!q||!k||!v||!ctx||!attn||!res||!ff||!ffg||!fout) return cleanup();

    // ── embeddings: word + type[0] + pos ──
    {
        cl_int e = CL_SUCCESS;
        e |= clSetKernelArg(g_k_embed, 0, sizeof(cl_mem), &input);
        e |= clSetKernelArg(g_k_embed, 1, sizeof(cl_mem), &w_word);
        e |= clSetKernelArg(g_k_embed, 2, sizeof(cl_mem), &w_type);
        e |= clSetKernelArg(g_k_embed, 3, sizeof(cl_mem), &w_pos);
        e |= clSetKernelArg(g_k_embed, 4, sizeof(cl_mem), &emb);
        e |= clSetKernelArg(g_k_embed, 5, sizeof(int), &T);
        int he = H_EMB;
        e |= clSetKernelArg(g_k_embed, 6, sizeof(int), &he);
        if (e != CL_SUCCESS) { NNOPT_ERROR_FMT("embed setarg %d", e); return cleanup(); }
        size_t gws = (size_t)T * H_EMB;
        e = clEnqueueNDRangeKernel(queue, g_k_embed, 1, nullptr, &gws, nullptr, 0, nullptr, nullptr);
        if (e != CL_SUCCESS) { NNOPT_ERROR_FMT("embed dispatch %d", e); return cleanup(); }
    }
    if (!layernorm(queue, T, H_EMB, emb, eln_w, eln_b, embn, g_k_ln)) return cleanup();

    // ── embedding_hidden_mapping_in: 128 -> 768 ──
    if (!linear_bias(queue, utils, T, H, H_EMB, embn, map_w, map_b, h, g_k_bias)) return cleanup();

    // ── 12 weight-shared ALBERT layers ──
    for (int layer = 0; layer < N_LAYERS; layer++) {
        // q,k,v = h @ W + b
        if (!linear_bias(queue, utils, T, H, H, h, q_w, q_b, q, g_k_bias)) return cleanup();
        if (!linear_bias(queue, utils, T, H, H, h, k_w, k_b, k, g_k_bias)) return cleanup();
        if (!linear_bias(queue, utils, T, H, H, h, v_w, v_b, v, g_k_bias)) return cleanup();
        // attention
        {
            cl_int e = CL_SUCCESS;
            e |= clSetKernelArg(g_k_attn, 0, sizeof(cl_mem), &q);
            e |= clSetKernelArg(g_k_attn, 1, sizeof(cl_mem), &k);
            e |= clSetKernelArg(g_k_attn, 2, sizeof(cl_mem), &v);
            e |= clSetKernelArg(g_k_attn, 3, sizeof(cl_mem), &ctx);
            e |= clSetKernelArg(g_k_attn, 4, sizeof(int), &T);
            int nh = N_HEADS, hd = HEAD_DIM;
            e |= clSetKernelArg(g_k_attn, 5, sizeof(int), &nh);
            e |= clSetKernelArg(g_k_attn, 6, sizeof(int), &hd);
            if (e != CL_SUCCESS) { NNOPT_ERROR_FMT("attn setarg %d", e); return cleanup(); }
            size_t gws = (size_t)T * N_HEADS;
            e = clEnqueueNDRangeKernel(queue, g_k_attn, 1, nullptr, &gws, nullptr, 0, nullptr, KernelProfiler::event_for("bert_attention"));
            if (e != CL_SUCCESS) { NNOPT_ERROR_FMT("attn dispatch %d", e); return cleanup(); }
        }
        // dense: ctx @ Wdense + b
        if (!linear_bias(queue, utils, T, H, H, ctx, d_w, d_b, attn, g_k_bias)) return cleanup();
        // residual: res = h + attn
        {
            cl_int e = CL_SUCCESS; int n = T * H;
            e |= clSetKernelArg(g_k_resadd, 0, sizeof(cl_mem), &h);
            e |= clSetKernelArg(g_k_resadd, 1, sizeof(cl_mem), &attn);
            e |= clSetKernelArg(g_k_resadd, 2, sizeof(cl_mem), &res);
            e |= clSetKernelArg(g_k_resadd, 3, sizeof(int), &n);
            if (e != CL_SUCCESS) { NNOPT_ERROR_FMT("resadd setarg %d", e); return cleanup(); }
            size_t gws = (size_t)n;
            e = clEnqueueNDRangeKernel(queue, g_k_resadd, 1, nullptr, &gws, nullptr, 0, nullptr, KernelProfiler::event_for("bert_residual"));
            if (e != CL_SUCCESS) { NNOPT_ERROR_FMT("resadd dispatch %d", e); return cleanup(); }
        }
        // h = LayerNorm(res, attention.LayerNorm)
        if (!layernorm(queue, T, H, res, aln_w, aln_b, h, g_k_ln)) return cleanup();

        // ffn: gelu_new(h @ Wffn + b) -> ff [T,FFN]
        if (!linear_bias(queue, utils, T, FFN, H, h, ffn_w, ffn_b, ff, g_k_bias)) return cleanup();
        {
            cl_int e = CL_SUCCESS; int n = T * FFN;
            e |= clSetKernelArg(g_k_gelu, 0, sizeof(cl_mem), &ff);
            e |= clSetKernelArg(g_k_gelu, 1, sizeof(cl_mem), &ffg);
            e |= clSetKernelArg(g_k_gelu, 2, sizeof(int), &n);
            if (e != CL_SUCCESS) { NNOPT_ERROR_FMT("gelu setarg %d", e); return cleanup(); }
            size_t gws = (size_t)n;
            e = clEnqueueNDRangeKernel(queue, g_k_gelu, 1, nullptr, &gws, nullptr, 0, nullptr, KernelProfiler::event_for("bert_gelu"));
            if (e != CL_SUCCESS) { NNOPT_ERROR_FMT("gelu dispatch %d", e); return cleanup(); }
        }
        // ffn_output: ffg @ Wfo + b -> fout [T,H]
        if (!linear_bias(queue, utils, T, H, FFN, ffg, fo_w, fo_b, fout, g_k_bias)) return cleanup();
        // residual: res = h + fout
        {
            cl_int e = CL_SUCCESS; int n = T * H;
            e |= clSetKernelArg(g_k_resadd, 0, sizeof(cl_mem), &h);
            e |= clSetKernelArg(g_k_resadd, 1, sizeof(cl_mem), &fout);
            e |= clSetKernelArg(g_k_resadd, 2, sizeof(cl_mem), &res);
            e |= clSetKernelArg(g_k_resadd, 3, sizeof(int), &n);
            if (e != CL_SUCCESS) { NNOPT_ERROR_FMT("resadd2 setarg %d", e); return cleanup(); }
            size_t gws = (size_t)n;
            e = clEnqueueNDRangeKernel(queue, g_k_resadd, 1, nullptr, &gws, nullptr, 0, nullptr, KernelProfiler::event_for("bert_residual"));
            if (e != CL_SUCCESS) { NNOPT_ERROR_FMT("resadd2 dispatch %d", e); return cleanup(); }
        }
        // h = LayerNorm(res, full_layer_layer_norm)
        if (!layernorm(queue, T, H, res, fln_w, fln_b, h, g_k_ln)) return cleanup();
    }

    NNOPT_DEBUG_SYNC(queue);
    // return h [T,768]; free everything else
    cl_mem out = h; h = nullptr;
    cleanup();
    return out;
}
}
