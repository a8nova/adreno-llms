// Reference: kokoro/model.py:forward_with_tokens — bert_dur = self.bert(input_ids, attention_mask=(~text_mask).int())
// Reference: HuggingFace AlbertModel architecture with parameter sharing.
//
// Config (from model_info/config.json::plbert):
//   hidden_size=768, num_attention_heads=12 (head_dim=64),
//   intermediate_size=2048, max_position_embeddings=512,
//   num_hidden_layers=12 — but only 1 albert_layer_group with 1 albert_layer,
//   so all 12 transformer layers share the SAME weights.
//
// Albert's embedding factorization:
//   word_embeddings: [n_token=178, 128]   <-- LOW dim
//   position_embeddings: [512, 128]
//   token_type_embeddings: [2, 128]
//   embeddings.LayerNorm: [128]
//   encoder.embedding_hidden_mapping_in: Linear 128 -> 768
//
// Each Albert layer (shared):
//   attention.query/key/value/dense: Linear 768 -> 768 each
//   attention.LayerNorm: [768]
//   ffn:        Linear 768 -> 2048
//   ffn_output: Linear 2048 -> 768
//   full_layer_layer_norm: [768]
//
// Output: bert_dur [T, 768] storage_t. Validates against
//   reference/layers/bert_output.bin

#include "opencl_context.h"
#include "weights.h"
#include "debug_utils.h"
#include "profiler.h"
#include "model_config.h"
#include "utils.h"

#include <CL/cl.h>
#include <cstdint>
#include <string>
#include <vector>

extern "C" int prim_embedding_gather_nlc(OpenCLContext&, cl_command_queue,
                                         cl_mem, cl_mem, cl_mem, int, int, int);
extern "C" int prim_layernorm(OpenCLContext&, cl_command_queue,
                              cl_mem, cl_mem, cl_mem, cl_mem, int, int, float);

static cl_program g_bert_prog = nullptr;
static cl_kernel  g_k_add3 = nullptr;        // y = a+b+c
static cl_kernel  g_k_add_inplace = nullptr; // y += x
static cl_kernel  g_k_bias_add = nullptr;
static cl_kernel  g_k_softmax = nullptr;
static cl_kernel  g_k_gelu = nullptr;
static cl_kernel  g_k_scaled_attn = nullptr;
static cl_kernel  g_k_scaled_attn_ctx = nullptr;
static cl_kernel  g_k_pos_ids = nullptr;

static const char* k_bert_src = R"CLC(
#ifdef NNOPT_USE_FP16
  #pragma OPENCL EXTENSION cl_khr_fp16 : enable
  typedef half storage_t;
  #define LOAD(p,i)    vload_half((i), (__global const half*)(p))
  #define STORE(p,i,v) vstore_half((float)(v), (i), (__global half*)(p))
#else
  typedef float storage_t;
  #define LOAD(p,i)    ((p)[(i)])
  #define STORE(p,i,v) ((p)[(i)] = (v))
#endif

__kernel void k_add3(__global const storage_t* a,
                     __global const storage_t* b,
                     __global const storage_t* c,
                     __global storage_t* y, int N) {
    int i = get_global_id(0);
    if (i >= N) return;
    float v = (float)LOAD(a,i) + (float)LOAD(b,i) + (float)LOAD(c,i);
    STORE(y, i, v);
}

__kernel void k_add_inplace(__global storage_t* y,
                            __global const storage_t* x, int N) {
    int i = get_global_id(0);
    if (i >= N) return;
    float v = (float)LOAD(y,i) + (float)LOAD(x,i);
    STORE(y, i, v);
}

__kernel void k_bias_add_2d(__global storage_t* y,
                            __global const storage_t* b,
                            int M, int N) {
    int i = get_global_id(0);
    int total = M*N;
    if (i >= total) return;
    int n = i % N;
    float v = (float)LOAD(y,i) + (float)LOAD(b,n);
    STORE(y, i, v);
}

// Softmax over last dim. One thread per row.
__kernel void k_softmax_rowwise(__global storage_t* x, int M, int N) {
    int m = get_global_id(0);
    if (m >= M) return;
    int base = m*N;
    float mx = (float)LOAD(x, base);
    for (int n = 1; n < N; ++n) {
        float v = (float)LOAD(x, base+n);
        if (v > mx) mx = v;
    }
    float sum = 0.0f;
    for (int n = 0; n < N; ++n) {
        float v = exp((float)LOAD(x, base+n) - mx);
        STORE(x, base+n, v);
        sum += v;
    }
    float inv = 1.0f / sum;
    for (int n = 0; n < N; ++n) {
        float v = (float)LOAD(x, base+n) * inv;
        STORE(x, base+n, v);
    }
}

// Approximate GELU (tanh formulation, same as HF Albert).
__kernel void k_gelu(__global storage_t* y, int N) {
    int i = get_global_id(0);
    if (i >= N) return;
    float x = (float)LOAD(y, i);
    float c = 0.7978845608f * (x + 0.044715f * x*x*x); // sqrt(2/pi)*(x + 0.044715*x^3)
    float t = tanh(c);
    float v = 0.5f * x * (1.0f + t);
    STORE(y, i, v);
}

// Scaled dot-product attention per head:
//   scores[h, i, j] = (Q[h, i, :] . K[h, j, :]) / sqrt(d_head)
// Layout: Q, K, V are [T, H, D] (row-major: t * H*D + h*D + d).
// scores: [H, T, T] (row-major: h*T*T + i*T + j).
__kernel void k_scaled_attn_scores(__global const storage_t* Q,
                                   __global const storage_t* K,
                                   __global storage_t* scores,
                                   int T, int H, int D) {
    int h = get_global_id(0);
    int i = get_global_id(1);
    int j = get_global_id(2);
    if (h >= H || i >= T || j >= T) return;
    float acc = 0.0f;
    int qbase = i*H*D + h*D;
    int kbase = j*H*D + h*D;
    for (int d = 0; d < D; ++d) {
        acc += (float)LOAD(Q, qbase+d) * (float)LOAD(K, kbase+d);
    }
    acc *= rsqrt((float)D);
    STORE(scores, h*T*T + i*T + j, acc);
}

// ctx[i, h, d] = sum_j scores[h, i, j] * V[j, h, d]
__kernel void k_scaled_attn_ctx(__global const storage_t* scores,
                                __global const storage_t* V,
                                __global storage_t* ctx,
                                int T, int H, int D) {
    int i = get_global_id(0);
    int h = get_global_id(1);
    int d = get_global_id(2);
    if (i >= T || h >= H || d >= D) return;
    float acc = 0.0f;
    for (int j = 0; j < T; ++j) {
        acc += (float)LOAD(scores, h*T*T + i*T + j) * (float)LOAD(V, j*H*D + h*D + d);
    }
    STORE(ctx, i*H*D + h*D + d, acc);
}

__kernel void k_make_pos_ids(__global int* pos, __global int* type_ids, int T) {
    int i = get_global_id(0);
    if (i >= T) return;
    pos[i] = i;
    type_ids[i] = 0;
}
)CLC";

static bool ensure_built(OpenCLContext& cl_ctx) {
    if (g_k_add3) return true;
    cl_int err = CL_SUCCESS;
    const char* opts =
#ifdef NNOPT_USE_FP16
        "-DNNOPT_USE_FP16=1";
#else
        "";
#endif
    cl_device_id dev = cl_ctx.device();
    g_bert_prog = nnopt_build_program_cached(cl_ctx.context(), dev, k_bert_src, opts, "bert", &err);
    if (!g_bert_prog) return false;
    g_k_add3         = clCreateKernel(g_bert_prog, "k_add3", &err);
    g_k_add_inplace  = clCreateKernel(g_bert_prog, "k_add_inplace", &err);
    g_k_bias_add     = clCreateKernel(g_bert_prog, "k_bias_add_2d", &err);
    g_k_softmax      = clCreateKernel(g_bert_prog, "k_softmax_rowwise", &err);
    g_k_gelu         = clCreateKernel(g_bert_prog, "k_gelu", &err);
    g_k_scaled_attn  = clCreateKernel(g_bert_prog, "k_scaled_attn_scores", &err);
    g_k_scaled_attn_ctx = clCreateKernel(g_bert_prog, "k_scaled_attn_ctx", &err);
    g_k_pos_ids      = clCreateKernel(g_bert_prog, "k_make_pos_ids", &err);
    return g_k_add3 && g_k_add_inplace && g_k_bias_add && g_k_softmax && g_k_gelu && g_k_scaled_attn && g_k_scaled_attn_ctx && g_k_pos_ids;
}

static cl_mem alloc(OpenCLContext& cl_ctx, size_t bytes) {
    cl_int e=CL_SUCCESS;
    return clCreateBuffer(cl_ctx.context(), CL_MEM_READ_WRITE, bytes, nullptr, &e);
}

static int run1d(cl_command_queue q, cl_kernel k, size_t gws) {
    return nnopt_enqueue_profiled(q, k, 1, nullptr, &gws, nullptr, 0, nullptr, nullptr);
}

// Albert is parameter-shared: all 12 layers use the same weight prefix.
static const char* LAYER_PREFIX = "bert.module.encoder.albert_layer_groups.0.albert_layers.0";

// out: caller-allocated [T, 768] storage_t.
extern "C" int op_bert(OpenCLContext& cl_ctx, Weights& weights, cl_command_queue queue,
                       cl_mem input_ids_i32, cl_mem out, int T) {
    if (!ensure_built(cl_ctx)) return -1;
    const int H = 12, D = 64, Hdim = H*D, FF = 2048;
    const size_t sz_t = sizeof(nnopt_storage_t);

    // Weights
    cl_mem W_word = weights.get_buffer("bert.module.embeddings.word_embeddings.weight");
    cl_mem W_pos  = weights.get_buffer("bert.module.embeddings.position_embeddings.weight");
    cl_mem W_tt   = weights.get_buffer("bert.module.embeddings.token_type_embeddings.weight");
    cl_mem ln_g   = weights.get_buffer("bert.module.embeddings.LayerNorm.weight");
    cl_mem ln_b   = weights.get_buffer("bert.module.embeddings.LayerNorm.bias");
    cl_mem map_w  = weights.get_buffer("bert.module.encoder.embedding_hidden_mapping_in.weight");
    cl_mem map_b  = weights.get_buffer("bert.module.encoder.embedding_hidden_mapping_in.bias");
    if (!W_word||!W_pos||!W_tt||!ln_g||!ln_b||!map_w||!map_b) {
        NNOPT_ERROR("op_bert: missing embedding weights"); return -1;
    }

    auto W = [&](const char* suf){
        std::string k = std::string(LAYER_PREFIX) + "." + suf;
        cl_mem m = weights.get_buffer(k);
        if (!m) NNOPT_ERROR_FMT("op_bert: missing %s", k.c_str());
        return m;
    };

    cl_int err = CL_SUCCESS;
    // Position ids + token_type ids
    cl_mem pos_i = clCreateBuffer(cl_ctx.context(), CL_MEM_READ_WRITE, sizeof(int)*T, nullptr, &err);
    cl_mem tt_i  = clCreateBuffer(cl_ctx.context(), CL_MEM_READ_WRITE, sizeof(int)*T, nullptr, &err);
    clSetKernelArg(g_k_pos_ids, 0, sizeof(cl_mem), &pos_i);
    clSetKernelArg(g_k_pos_ids, 1, sizeof(cl_mem), &tt_i);
    clSetKernelArg(g_k_pos_ids, 2, sizeof(int), &T);
    run1d(queue, g_k_pos_ids, (size_t)T);

    // Three gather outputs at [T, 128]
    cl_mem emb_w  = alloc(cl_ctx, sz_t*T*128);
    cl_mem emb_p  = alloc(cl_ctx, sz_t*T*128);
    cl_mem emb_t  = alloc(cl_ctx, sz_t*T*128);
    cl_mem emb_sum= alloc(cl_ctx, sz_t*T*128);
    cl_mem emb_ln = alloc(cl_ctx, sz_t*T*128);
    cl_mem hidden = alloc(cl_ctx, sz_t*T*Hdim);  // [T, 768]
    prim_embedding_gather_nlc(cl_ctx, queue, input_ids_i32, W_word, emb_w, T, 128, 178);
    prim_embedding_gather_nlc(cl_ctx, queue, pos_i,  W_pos, emb_p, T, 128, 512);
    prim_embedding_gather_nlc(cl_ctx, queue, tt_i,   W_tt,  emb_t, T, 128, 2);
    // sum
    int N3 = T*128;
    clSetKernelArg(g_k_add3, 0, sizeof(cl_mem), &emb_w);
    clSetKernelArg(g_k_add3, 1, sizeof(cl_mem), &emb_p);
    clSetKernelArg(g_k_add3, 2, sizeof(cl_mem), &emb_t);
    clSetKernelArg(g_k_add3, 3, sizeof(cl_mem), &emb_sum);
    clSetKernelArg(g_k_add3, 4, sizeof(int), &N3);
    run1d(queue, g_k_add3, (size_t)N3);
    prim_layernorm(cl_ctx, queue, emb_sum, ln_g, ln_b, emb_ln, T, 128, 1e-12f);

    // embedding_hidden_mapping_in: Linear 128 -> 768
    pytorch_linear(queue, T, Hdim, 128, emb_ln, map_w, hidden);
    int Mh = T, Nh = Hdim;
    clSetKernelArg(g_k_bias_add, 0, sizeof(cl_mem), &hidden);
    clSetKernelArg(g_k_bias_add, 1, sizeof(cl_mem), &map_b);
    clSetKernelArg(g_k_bias_add, 2, sizeof(int), &Mh);
    clSetKernelArg(g_k_bias_add, 3, sizeof(int), &Nh);
    run1d(queue, g_k_bias_add, (size_t)T*Hdim);

    // Per-layer scratch (parameter shared — load weights once)
    cl_mem Wq = W("attention.query.weight"), Bq = W("attention.query.bias");
    cl_mem Wk = W("attention.key.weight"),   Bk = W("attention.key.bias");
    cl_mem Wv = W("attention.value.weight"), Bv = W("attention.value.bias");
    cl_mem Wd = W("attention.dense.weight"), Bd = W("attention.dense.bias");
    cl_mem an_g = W("attention.LayerNorm.weight"), an_b = W("attention.LayerNorm.bias");
    cl_mem Wf1 = W("ffn.weight"),        Bf1 = W("ffn.bias");
    cl_mem Wf2 = W("ffn_output.weight"), Bf2 = W("ffn_output.bias");
    cl_mem fn_g = W("full_layer_layer_norm.weight"), fn_b = W("full_layer_layer_norm.bias");
    if (!Wq||!Wk||!Wv||!Wd||!Wf1||!Wf2) { NNOPT_ERROR("op_bert: missing layer weights"); return -1; }

    cl_mem Q = alloc(cl_ctx, sz_t*T*Hdim);
    cl_mem K = alloc(cl_ctx, sz_t*T*Hdim);
    cl_mem V = alloc(cl_ctx, sz_t*T*Hdim);
    cl_mem ctx_a = alloc(cl_ctx, sz_t*T*Hdim);
    cl_mem proj = alloc(cl_ctx, sz_t*T*Hdim);
    cl_mem scores = alloc(cl_ctx, sz_t*H*T*T);
    cl_mem ff1 = alloc(cl_ctx, sz_t*T*FF);
    cl_mem ff2 = alloc(cl_ctx, sz_t*T*Hdim);

    int Nh1 = Hdim, Nff = FF;
    int THT = T*Hdim;

    for (int layer = 0; layer < 12; ++layer) {
        // ---- Attention ----
        // Q = hidden @ Wq^T + Bq
        pytorch_linear(queue, T, Hdim, Hdim, hidden, Wq, Q);
        clSetKernelArg(g_k_bias_add, 0, sizeof(cl_mem), &Q);
        clSetKernelArg(g_k_bias_add, 1, sizeof(cl_mem), &Bq);
        clSetKernelArg(g_k_bias_add, 2, sizeof(int), &Mh);
        clSetKernelArg(g_k_bias_add, 3, sizeof(int), &Nh1);
        run1d(queue, g_k_bias_add, (size_t)THT);
        pytorch_linear(queue, T, Hdim, Hdim, hidden, Wk, K);
        clSetKernelArg(g_k_bias_add, 0, sizeof(cl_mem), &K);
        clSetKernelArg(g_k_bias_add, 1, sizeof(cl_mem), &Bk);
        run1d(queue, g_k_bias_add, (size_t)THT);
        pytorch_linear(queue, T, Hdim, Hdim, hidden, Wv, V);
        clSetKernelArg(g_k_bias_add, 0, sizeof(cl_mem), &V);
        clSetKernelArg(g_k_bias_add, 1, sizeof(cl_mem), &Bv);
        run1d(queue, g_k_bias_add, (size_t)THT);

        // scores [H, T, T]
        clSetKernelArg(g_k_scaled_attn, 0, sizeof(cl_mem), &Q);
        clSetKernelArg(g_k_scaled_attn, 1, sizeof(cl_mem), &K);
        clSetKernelArg(g_k_scaled_attn, 2, sizeof(cl_mem), &scores);
        clSetKernelArg(g_k_scaled_attn, 3, sizeof(int), &T);
        clSetKernelArg(g_k_scaled_attn, 4, sizeof(int), (int*)&H);
        clSetKernelArg(g_k_scaled_attn, 5, sizeof(int), (int*)&D);
        size_t gws3[3] = {(size_t)H, (size_t)T, (size_t)T};
        nnopt_enqueue_profiled(queue, g_k_scaled_attn, 3, nullptr, gws3, nullptr, 0, nullptr, nullptr);
        // softmax over j for each [h, i, :]
        int Msm = H*T, Nsm = T;
        clSetKernelArg(g_k_softmax, 0, sizeof(cl_mem), &scores);
        clSetKernelArg(g_k_softmax, 1, sizeof(int), &Msm);
        clSetKernelArg(g_k_softmax, 2, sizeof(int), &Nsm);
        run1d(queue, g_k_softmax, (size_t)Msm);
        // ctx[i, h, d] = sum_j scores[h, i, j] * V[j, h, d]
        clSetKernelArg(g_k_scaled_attn_ctx, 0, sizeof(cl_mem), &scores);
        clSetKernelArg(g_k_scaled_attn_ctx, 1, sizeof(cl_mem), &V);
        clSetKernelArg(g_k_scaled_attn_ctx, 2, sizeof(cl_mem), &ctx_a);
        clSetKernelArg(g_k_scaled_attn_ctx, 3, sizeof(int), &T);
        clSetKernelArg(g_k_scaled_attn_ctx, 4, sizeof(int), (int*)&H);
        clSetKernelArg(g_k_scaled_attn_ctx, 5, sizeof(int), (int*)&D);
        size_t gws3c[3] = {(size_t)T, (size_t)H, (size_t)D};
        nnopt_enqueue_profiled(queue, g_k_scaled_attn_ctx, 3, nullptr, gws3c, nullptr, 0, nullptr, nullptr);

        // out proj: ctx_a -> proj
        pytorch_linear(queue, T, Hdim, Hdim, ctx_a, Wd, proj);
        clSetKernelArg(g_k_bias_add, 0, sizeof(cl_mem), &proj);
        clSetKernelArg(g_k_bias_add, 1, sizeof(cl_mem), &Bd);
        run1d(queue, g_k_bias_add, (size_t)THT);
        // residual + LN (attention.LayerNorm)
        clSetKernelArg(g_k_add_inplace, 0, sizeof(cl_mem), &proj);
        clSetKernelArg(g_k_add_inplace, 1, sizeof(cl_mem), &hidden);
        clSetKernelArg(g_k_add_inplace, 2, sizeof(int), &THT);
        run1d(queue, g_k_add_inplace, (size_t)THT);
        prim_layernorm(cl_ctx, queue, proj, an_g, an_b, hidden, T, Hdim, 1e-12f);

        // ---- FFN ----
        pytorch_linear(queue, T, FF, Hdim, hidden, Wf1, ff1);
        int Mff = T;
        clSetKernelArg(g_k_bias_add, 0, sizeof(cl_mem), &ff1);
        clSetKernelArg(g_k_bias_add, 1, sizeof(cl_mem), &Bf1);
        clSetKernelArg(g_k_bias_add, 2, sizeof(int), &Mff);
        clSetKernelArg(g_k_bias_add, 3, sizeof(int), &Nff);
        run1d(queue, g_k_bias_add, (size_t)T*FF);
        int Nff_g = T*FF;
        clSetKernelArg(g_k_gelu, 0, sizeof(cl_mem), &ff1);
        clSetKernelArg(g_k_gelu, 1, sizeof(int), &Nff_g);
        run1d(queue, g_k_gelu, (size_t)Nff_g);
        pytorch_linear(queue, T, Hdim, FF, ff1, Wf2, ff2);
        clSetKernelArg(g_k_bias_add, 0, sizeof(cl_mem), &ff2);
        clSetKernelArg(g_k_bias_add, 1, sizeof(cl_mem), &Bf2);
        clSetKernelArg(g_k_bias_add, 2, sizeof(int), &Mh);
        clSetKernelArg(g_k_bias_add, 3, sizeof(int), &Nh1);
        run1d(queue, g_k_bias_add, (size_t)THT);
        // residual + LN (full_layer_layer_norm)
        clSetKernelArg(g_k_add_inplace, 0, sizeof(cl_mem), &ff2);
        clSetKernelArg(g_k_add_inplace, 1, sizeof(cl_mem), &hidden);
        clSetKernelArg(g_k_add_inplace, 2, sizeof(int), &THT);
        run1d(queue, g_k_add_inplace, (size_t)THT);
        prim_layernorm(cl_ctx, queue, ff2, fn_g, fn_b, hidden, T, Hdim, 1e-12f);
    }

    // Copy hidden -> out
    clEnqueueCopyBuffer(queue, hidden, out, 0, 0, sz_t*T*Hdim, 0, nullptr, nullptr);

    // Release scratch
    for (cl_mem m : {pos_i, tt_i, emb_w, emb_p, emb_t, emb_sum, emb_ln, hidden,
                     Q, K, V, ctx_a, proj, scores, ff1, ff2}) {
        if (m) clReleaseMemObject(m);
    }
    NNOPT_LAYER_CHECK("bert", queue, out, (size_t)T*Hdim);
    return 0;
}
