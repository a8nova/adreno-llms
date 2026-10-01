// MusicCoCa text tower + RVQ on the GPU. See musiccoca.h for the architecture and for what the
// oracle proved about it (K/V/Q order, no 1/sqrt(d), learned-query pooler).
//
// Weight names are the roles scripts/extract_musiccoca.py resolves from the dataflow:
//   embed, pos, layer{L}.{ln1,ln2}.{scale,bias}, layer{L}.{q,k,v,o,ffn1,ffn2}.{weight,bias},
//   pooler.{ln,ln_final}.{scale,bias}, pooler.query, pooler.{k,v,out}.{weight,bias},
//   rvq.codebook{0..11}
//
// Every matmul is [rows, K] x [N, K]^T, matching the FULLY_CONNECTED convention the export uses
// (weights stored [out, in]) and the layout linear_f32_w16 / linear_bias_f32 expect.

#include "musiccoca.h"

#include "debug_utils.h"
#include "utils.h"

#include <CL/cl.h>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>

#ifdef USE_CLBLAST
#include <clblast_c.h>
#endif

namespace {

constexpr int D = kMusicCoCaDim;
constexpr int H = kMusicCoCaHeads;
constexpr int HD = D / H;                 // 64
constexpr int F = kMusicCoCaFFN;
constexpr int PH = kMusicCoCaPoolHeads;   // 12
constexpr int PHD = kMusicCoCaPoolHeadDim;  // 256

// The export applies 50*tanh(0.02*QK^T) between the matmul and the softmax. 0.02 is exactly 1/50,
// so this is a pure logit cap with no head-dim scaling — that is folded into the query weights.
constexpr float kLogitCap = 50.0f;

cl_kernel build(OpenCLContext& cl_ctx, const char* file, const char* entry) {
    cl_program p = cl_ctx.build_program_from_file(file);
    if (!p) { NNOPT_ERROR_FMT("musiccoca: build %s", file); return nullptr; }
    cl_int e = CL_SUCCESS;
    cl_kernel k = clCreateKernel(p, entry, &e);
    if (e != CL_SUCCESS || !k) { NNOPT_ERROR_FMT("musiccoca: clCreateKernel %s (%d)", entry, e); return nullptr; }
    return k;
}

// out[rows, N] = in[rows, K] @ W[N, K]^T + bias. W and bias are fp16, activations fp32.
cl_mem linear(OpenCLContext& cl_ctx, cl_mem in, cl_mem W, cl_mem bias, int rows, int K, int N) {
    static cl_kernel k_nb = nullptr, k_b = nullptr;
    if (!k_nb) k_nb = build(cl_ctx, "kernels/linear_f32_w16.cl", "linear_f32_w16");
    if (bias && !k_b) k_b = build(cl_ctx, "kernels/linear_bias_f32.cl", "linear_bias_f32");
    cl_kernel use = bias ? k_b : k_nb;
    if (!use || !W) { NNOPT_ERROR("musiccoca: linear missing kernel or weight"); return nullptr; }
    cl_int e = CL_SUCCESS;
    cl_mem out = pool_alloc(cl_ctx.context(), CL_MEM_READ_WRITE, (size_t)rows * N * sizeof(float), nullptr, &e);
    if (!out) { NNOPT_ERROR("musiccoca: linear alloc"); return nullptr; }
    int ai = 0;
    clSetKernelArg(use, ai++, sizeof(cl_mem), &in);
    clSetKernelArg(use, ai++, sizeof(cl_mem), &W);
    if (bias) clSetKernelArg(use, ai++, sizeof(cl_mem), &bias);
    clSetKernelArg(use, ai++, sizeof(cl_mem), &out);
    clSetKernelArg(use, ai++, sizeof(int), &K);
    clSetKernelArg(use, ai++, sizeof(int), &N);
    const size_t nwg = ((size_t)N + 7) / 8;
    const size_t gws[2] = {(size_t)rows, nwg * 64}, lws[2] = {1, 64};
    if (cl_ctx.profEnqueue(use, 2, gws, lws, "mc_linear") != CL_SUCCESS) {
        NNOPT_ERROR("musiccoca: linear enqueue"); pool_free(out); return nullptr;
    }
    return out;
}

cl_mem layernorm(OpenCLContext& cl_ctx, cl_mem x, cl_mem scale, cl_mem bias, int rows, int dim) {
    static cl_kernel k = nullptr;
    if (!k) k = build(cl_ctx, "kernels/layer_norm_f32.cl", "layer_norm_f32");
    if (!k || !scale || !bias) { NNOPT_ERROR("musiccoca: layernorm missing kernel or weights"); return nullptr; }
    cl_int e = CL_SUCCESS;
    cl_mem out = pool_alloc(cl_ctx.context(), CL_MEM_READ_WRITE, (size_t)rows * dim * sizeof(float), nullptr, &e);
    if (!out) { NNOPT_ERROR("musiccoca: layernorm alloc"); return nullptr; }
    int ai = 0;
    const float eps = 1e-6f;
    clSetKernelArg(k, ai++, sizeof(cl_mem), &x);
    clSetKernelArg(k, ai++, sizeof(cl_mem), &scale);
    clSetKernelArg(k, ai++, sizeof(cl_mem), &bias);
    clSetKernelArg(k, ai++, sizeof(cl_mem), &out);
    clSetKernelArg(k, ai++, sizeof(int), &dim);
    clSetKernelArg(k, ai++, sizeof(float), &eps);
    const size_t gws[1] = {(size_t)rows};
    if (cl_ctx.profEnqueue(k, 1, gws, nullptr, "mc_ln") != CL_SUCCESS) {
        NNOPT_ERROR("musiccoca: layernorm enqueue"); pool_free(out); return nullptr;
    }
    return out;
}

// dst += src, both fp32 [n]. Returns a new buffer; frees neither input.
cl_mem add_f32(OpenCLContext& cl_ctx, cl_mem a, cl_mem b, int n) {
    static cl_kernel k = nullptr;
    if (!k) k = build(cl_ctx, "kernels/add_f32.cl", "add_f32");
    if (!k) return nullptr;
    cl_int e = CL_SUCCESS;
    cl_mem out = pool_alloc(cl_ctx.context(), CL_MEM_READ_WRITE, (size_t)n * sizeof(float), nullptr, &e);
    if (!out) { NNOPT_ERROR("musiccoca: add alloc"); return nullptr; }
    clSetKernelArg(k, 0, sizeof(cl_mem), &a);
    clSetKernelArg(k, 1, sizeof(cl_mem), &b);
    clSetKernelArg(k, 2, sizeof(cl_mem), &out);
    clSetKernelArg(k, 3, sizeof(int), &n);
    size_t g = (size_t)n;
    if (cl_ctx.profEnqueue(k, 1, &g, nullptr, "mc_add") != CL_SUCCESS) {
        NNOPT_ERROR("musiccoca: add enqueue"); pool_free(out); return nullptr;
    }
    return out;
}

// Exact (erf) GELU — the export's activation, so this matches by construction. (The tanh
// approximation also held the style tokens on 10/10 measured prompts at cosine 0.99999997; there
// is simply no reason to take the risk when erf is already the cheaper-to-justify choice.)
bool gelu_inplace(OpenCLContext& cl_ctx, cl_mem x, int n) {
    static cl_kernel k = nullptr;
    if (!k) k = build(cl_ctx, "kernels/gelu_f32.cl", "gelu_f32");
    if (!k) return false;
    clSetKernelArg(k, 0, sizeof(cl_mem), &x);
    clSetKernelArg(k, 1, sizeof(cl_mem), &x);
    clSetKernelArg(k, 2, sizeof(int), &n);
    size_t g = (size_t)n;
    if (cl_ctx.profEnqueue(k, 1, &g, nullptr, "mc_gelu") != CL_SUCCESS) {
        NNOPT_ERROR("musiccoca: gelu enqueue"); return false;
    }
    return true;
}

struct Scoped {
    cl_mem m = nullptr;
    ~Scoped() { if (m) pool_free(m); }
    cl_mem release() { cl_mem t = m; m = nullptr; return t; }
    void reset(cl_mem n) { if (m) pool_free(m); m = n; }
};

}  // namespace

bool musiccoca_embed_text(OpenCLContext& cl_ctx, Weights& mc,
                          const std::vector<int32_t>& ids,
                          std::vector<float>& embedding_out) {
    const int n = (int)ids.size();
    if (n < 1 || n > kMusicCoCaMaxSeq) {
        NNOPT_ERROR_FMT("musiccoca: need 1..%d ids, got %d", kMusicCoCaMaxSeq, n);
        return false;
    }
#ifndef USE_CLBLAST
    NNOPT_ERROR("musiccoca: attention requires CLBlast");
    return false;
#else
    cl_command_queue q = cl_ctx.queue();
    cl_int e = CL_SUCCESS;

    // Every position is real: the caller passes only actual tokens, so nothing is masked and the
    // tower runs at n instead of the exported 128.
    std::vector<int32_t> ones(n, 1);
    Scoped ids_buf, valid_buf;
    ids_buf.m = pool_alloc(cl_ctx.context(), CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
                           ids.size() * sizeof(int32_t), const_cast<int32_t*>(ids.data()), &e);
    valid_buf.m = pool_alloc(cl_ctx.context(), CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
                             ones.size() * sizeof(int32_t), ones.data(), &e);
    if (!ids_buf.m || !valid_buf.m) { NNOPT_ERROR("musiccoca: id upload"); return false; }

    // ── embedding + learned positional ──────────────────────────────────────
    cl_mem emb_tab = mc.get_buffer("embed");
    cl_mem pos_tab = mc.get_buffer("pos");
    if (!emb_tab || !pos_tab) { NNOPT_ERROR("musiccoca: missing embed/pos"); return false; }
    const std::vector<int> emb_shape = mc.get_shape("embed");
    if (emb_shape.size() != 2 || emb_shape[1] != D) {
        NNOPT_ERROR("musiccoca: embed table is not [vocab, 768]"); return false;
    }
    const int vocab = emb_shape[0];

    static cl_kernel k_embed = nullptr;
    if (!k_embed) k_embed = build(cl_ctx, "kernels/embed_add_pos_f16.cl", "embed_add_pos_f16");
    if (!k_embed) return false;

    Scoped x;
    x.m = pool_alloc(cl_ctx.context(), CL_MEM_READ_WRITE, (size_t)n * D * sizeof(float), nullptr, &e);
    if (!x.m) { NNOPT_ERROR("musiccoca: x alloc"); return false; }
    {
        int ai = 0, dim = D, nn = n, vv = vocab;
        clSetKernelArg(k_embed, ai++, sizeof(cl_mem), &emb_tab);
        clSetKernelArg(k_embed, ai++, sizeof(cl_mem), &pos_tab);
        clSetKernelArg(k_embed, ai++, sizeof(cl_mem), &ids_buf.m);
        clSetKernelArg(k_embed, ai++, sizeof(cl_mem), &x.m);
        clSetKernelArg(k_embed, ai++, sizeof(int), &dim);
        clSetKernelArg(k_embed, ai++, sizeof(int), &nn);
        clSetKernelArg(k_embed, ai++, sizeof(int), &vv);
        size_t g[2] = {(size_t)n, (size_t)D};
        if (cl_ctx.profEnqueue(k_embed, 2, g, nullptr, "mc_embed") != CL_SUCCESS) {
            NNOPT_ERROR("musiccoca: embed enqueue"); return false;
        }
    }

    static cl_kernel k_softmax = nullptr;
    if (!k_softmax) k_softmax = build(cl_ctx, "kernels/softmax_mask_cap_f32.cl", "softmax_mask_cap_f32");
    if (!k_softmax) return false;

    for (int L = 0; L < kMusicCoCaLayers; ++L) {
        const std::string p = "layer" + std::to_string(L) + ".";
        Scoped h;
        h.m = layernorm(cl_ctx, x.m, mc.get_buffer(p + "ln1.scale"), mc.get_buffer(p + "ln1.bias"), n, D);
        if (!h.m) return false;

        Scoped Q, K, V;
        Q.m = linear(cl_ctx, h.m, mc.get_buffer(p + "q.weight"), mc.get_buffer(p + "q.bias"), n, D, D);
        K.m = linear(cl_ctx, h.m, mc.get_buffer(p + "k.weight"), mc.get_buffer(p + "k.bias"), n, D, D);
        V.m = linear(cl_ctx, h.m, mc.get_buffer(p + "v.weight"), mc.get_buffer(p + "v.bias"), n, D, D);
        if (!Q.m || !K.m || !V.m) { NNOPT_ERROR_FMT("musiccoca: qkv layer %d", L); return false; }

        Scoped logits, ctx;
        logits.m = pool_alloc(cl_ctx.context(), CL_MEM_READ_WRITE, (size_t)H * n * n * sizeof(float), nullptr, &e);
        ctx.m = pool_alloc(cl_ctx.context(), CL_MEM_READ_WRITE, (size_t)n * D * sizeof(float), nullptr, &e);
        if (!logits.m || !ctx.m) { NNOPT_ERROR("musiccoca: attention alloc"); return false; }

        for (int hh = 0; hh < H; ++hh) {
            // Q[n, HD] (row stride D, offset hh*HD) x K[n, HD]^T -> [n, n]
            CLBlastStatusCode st = CLBlastSgemm(
                CLBlastLayoutRowMajor, CLBlastTransposeNo, CLBlastTransposeYes,
                (size_t)n, (size_t)n, (size_t)HD, 1.0f,
                Q.m, (size_t)hh * HD, (size_t)D,
                K.m, (size_t)hh * HD, (size_t)D, 0.0f,
                logits.m, (size_t)hh * n * n, (size_t)n, &q, nullptr);
            if (st != CLBlastSuccess) { NNOPT_ERROR_FMT("musiccoca: QK^T st=%d", (int)st); return false; }
        }
        {
            // scale 1.0: the head-dim scaling lives in the query weights, the cap is the only
            // transform between the matmul and the softmax.
            int ai = 0, t = n;
            float cap = kLogitCap, sc = 1.0f;
            clSetKernelArg(k_softmax, ai++, sizeof(cl_mem), &logits.m);
            clSetKernelArg(k_softmax, ai++, sizeof(cl_mem), &valid_buf.m);
            clSetKernelArg(k_softmax, ai++, sizeof(int), &t);
            clSetKernelArg(k_softmax, ai++, sizeof(float), &cap);
            clSetKernelArg(k_softmax, ai++, sizeof(float), &sc);
            size_t gws[1] = {(size_t)H * n * 64}, lws[1] = {64};
            if (cl_ctx.profEnqueue(k_softmax, 1, gws, lws, "mc_softmax") != CL_SUCCESS) {
                NNOPT_ERROR("musiccoca: softmax enqueue"); return false;
            }
        }
        for (int hh = 0; hh < H; ++hh) {
            // P[n, n] x V[n, HD] -> ctx[n, HD] at column offset hh*HD
            CLBlastStatusCode st = CLBlastSgemm(
                CLBlastLayoutRowMajor, CLBlastTransposeNo, CLBlastTransposeNo,
                (size_t)n, (size_t)HD, (size_t)n, 1.0f,
                logits.m, (size_t)hh * n * n, (size_t)n,
                V.m, (size_t)hh * HD, (size_t)D, 0.0f,
                ctx.m, (size_t)hh * HD, (size_t)D, &q, nullptr);
            if (st != CLBlastSuccess) { NNOPT_ERROR_FMT("musiccoca: P·V st=%d", (int)st); return false; }
        }

        Scoped attn_out;
        attn_out.m = linear(cl_ctx, ctx.m, mc.get_buffer(p + "o.weight"), mc.get_buffer(p + "o.bias"), n, D, D);
        if (!attn_out.m) return false;
        cl_mem x1 = add_f32(cl_ctx, x.m, attn_out.m, n * D);
        if (!x1) return false;
        x.reset(x1);

        // ── FFN ──────────────────────────────────────────────────────────────
        Scoped h2;
        h2.m = layernorm(cl_ctx, x.m, mc.get_buffer(p + "ln2.scale"), mc.get_buffer(p + "ln2.bias"), n, D);
        if (!h2.m) return false;
        Scoped f1;
        f1.m = linear(cl_ctx, h2.m, mc.get_buffer(p + "ffn1.weight"), mc.get_buffer(p + "ffn1.bias"), n, D, F);
        if (!f1.m || !gelu_inplace(cl_ctx, f1.m, n * F)) return false;
        Scoped f2;
        f2.m = linear(cl_ctx, f1.m, mc.get_buffer(p + "ffn2.weight"), mc.get_buffer(p + "ffn2.bias"), n, F, D);
        if (!f2.m) return false;
        cl_mem x2 = add_f32(cl_ctx, x.m, f2.m, n * D);
        if (!x2) return false;
        x.reset(x2);
    }

    // ── learned-query attention pooling → one 768-d vector ───────────────────
    Scoped hp;
    hp.m = layernorm(cl_ctx, x.m, mc.get_buffer("pooler.ln.scale"), mc.get_buffer("pooler.ln.bias"), n, D);
    if (!hp.m) return false;
    Scoped pk, pv;
    pk.m = linear(cl_ctx, hp.m, mc.get_buffer("pooler.k.weight"), mc.get_buffer("pooler.k.bias"), n, D, PH * PHD);
    pv.m = linear(cl_ctx, hp.m, mc.get_buffer("pooler.v.weight"), mc.get_buffer("pooler.v.bias"), n, D, PH * PHD);
    cl_mem query = mc.get_buffer("pooler.query");
    if (!pk.m || !pv.m || !query) { NNOPT_ERROR("musiccoca: pooler projections"); return false; }

    static cl_kernel k_pool = nullptr;
    if (!k_pool) k_pool = build(cl_ctx, "kernels/pool_attn_f32.cl", "pool_attn_f32");
    if (!k_pool) return false;
    Scoped pooled;
    pooled.m = pool_alloc(cl_ctx.context(), CL_MEM_READ_WRITE, (size_t)PH * PHD * sizeof(float), nullptr, &e);
    if (!pooled.m) { NNOPT_ERROR("musiccoca: pooled alloc"); return false; }
    {
        int ai = 0, nn = n, hd = PHD, stride = PH * PHD;
        clSetKernelArg(k_pool, ai++, sizeof(cl_mem), &pk.m);
        clSetKernelArg(k_pool, ai++, sizeof(cl_mem), &pv.m);
        clSetKernelArg(k_pool, ai++, sizeof(cl_mem), &query);
        clSetKernelArg(k_pool, ai++, sizeof(cl_mem), &pooled.m);
        clSetKernelArg(k_pool, ai++, sizeof(int), &nn);
        clSetKernelArg(k_pool, ai++, sizeof(int), &hd);
        clSetKernelArg(k_pool, ai++, sizeof(int), &stride);
        size_t gws[1] = {(size_t)PH * 64}, lws[1] = {64};
        if (cl_ctx.profEnqueue(k_pool, 1, gws, lws, "mc_pool") != CL_SUCCESS) {
            NNOPT_ERROR("musiccoca: pooler enqueue"); return false;
        }
    }

    Scoped proj;
    proj.m = linear(cl_ctx, pooled.m, mc.get_buffer("pooler.out.weight"), mc.get_buffer("pooler.out.bias"),
                    1, PH * PHD, D);
    if (!proj.m) { NNOPT_ERROR("musiccoca: pooler output projection"); return false; }
    Scoped out;
    out.m = layernorm(cl_ctx, proj.m, mc.get_buffer("pooler.ln_final.scale"),
                      mc.get_buffer("pooler.ln_final.bias"), 1, D);
    if (!out.m) return false;

    embedding_out.resize(D);
    if (clEnqueueReadBuffer(q, out.m, CL_TRUE, 0, embedding_out.size() * sizeof(float),
                            embedding_out.data(), 0, nullptr, nullptr) != CL_SUCCESS) {
        NNOPT_ERROR("musiccoca: embedding readback"); return false;
    }
    return true;
#endif
}

bool musiccoca_quantize(OpenCLContext& cl_ctx, Weights& mc,
                        const std::vector<float>& embedding,
                        std::vector<int32_t>& tokens_out) {
    if ((int)embedding.size() != D) {
        NNOPT_ERROR_FMT("musiccoca_quantize: expected %d dims, got %zu", D, embedding.size());
        return false;
    }
    cl_command_queue q = cl_ctx.queue();
    cl_int e = CL_SUCCESS;
    static cl_kernel k = nullptr;
    if (!k) k = build(cl_ctx, "kernels/rvq_quantize_f32.cl", "rvq_quantize_f32");
    if (!k) return false;

    Scoped residual, idx_buf;
    residual.m = pool_alloc(cl_ctx.context(), CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR,
                            embedding.size() * sizeof(float),
                            const_cast<float*>(embedding.data()), &e);
    idx_buf.m = pool_alloc(cl_ctx.context(), CL_MEM_READ_WRITE, sizeof(int32_t), nullptr, &e);
    if (!residual.m || !idx_buf.m) { NNOPT_ERROR("musiccoca_quantize: alloc"); return false; }

    tokens_out.assign(kMusicCoCaStyleTokens, 0);
    for (int s = 0; s < kMusicCoCaStyleTokens; ++s) {
        const std::string key = "rvq.codebook" + std::to_string(s);
        cl_mem book = mc.get_buffer(key);
        if (!book) { NNOPT_ERROR_FMT("musiccoca_quantize: missing codebook %d", s); return false; }
        const std::vector<int> shape = mc.get_shape(key);
        if (shape.size() != 2 || shape[1] != D) {
            NNOPT_ERROR_FMT("musiccoca_quantize: codebook %d is not [N, %d]", s, D); return false;
        }
        cl_mem next = pool_alloc(cl_ctx.context(), CL_MEM_READ_WRITE, (size_t)D * sizeof(float), nullptr, &e);
        if (!next) { NNOPT_ERROR("musiccoca_quantize: residual alloc"); return false; }
        int ai = 0, nrows = shape[0], dim = D;
        clSetKernelArg(k, ai++, sizeof(cl_mem), &residual.m);
        clSetKernelArg(k, ai++, sizeof(cl_mem), &book);
        clSetKernelArg(k, ai++, sizeof(cl_mem), &idx_buf.m);
        clSetKernelArg(k, ai++, sizeof(cl_mem), &next);
        clSetKernelArg(k, ai++, sizeof(int), &nrows);
        clSetKernelArg(k, ai++, sizeof(int), &dim);
        size_t gws[1] = {256}, lws[1] = {256};
        if (cl_ctx.profEnqueue(k, 1, gws, lws, "mc_rvq") != CL_SUCCESS) {
            NNOPT_ERROR_FMT("musiccoca_quantize: stage %d enqueue", s); pool_free(next); return false;
        }
        int32_t token = 0;
        if (clEnqueueReadBuffer(q, idx_buf.m, CL_TRUE, 0, sizeof(int32_t), &token, 0, nullptr, nullptr)
            != CL_SUCCESS) {
            NNOPT_ERROR_FMT("musiccoca_quantize: stage %d readback", s); pool_free(next); return false;
        }
        tokens_out[s] = token;
        residual.reset(next);
    }
    return true;
}

static_assert(kMidiStyleTokens == kMusicCoCaStyleTokens, "style token count must agree with midi.h");
static_assert(kMidiConditioningTokens == kConditioningTokens, "conditioning layout must agree with midi.h");

// The block itself is built in midi.cpp — see midi_build_conditioning() for why. This wrapper
// exists so the MusicCoCa side keeps a name that says what it produces.
bool musiccoca_conditioning_tokens(const std::vector<int32_t>& style_tokens,
                                   std::vector<int32_t>& tokens_out,
                                   const MidiState* midi) {
    return midi_build_conditioning(style_tokens, midi, tokens_out);
}
