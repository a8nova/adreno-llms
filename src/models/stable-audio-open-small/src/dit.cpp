// dit.cpp — Stable Audio Open Small DiffusionTransformer single-denoise-step forward.
//
// Reference: stable_audio_tools/models/dit.py DiffusionTransformer._forward
//            stable_audio_tools/models/transformer.py ContinuousTransformer.forward,
//              TransformerBlock.forward, Attention.forward, FeedForward.forward,
//              GLU.forward, LayerNorm.forward, RotaryEmbedding, apply_rotary_pos_emb
//            stable_audio_tools/models/blocks.py FourierFeatures.forward
//
// Config: embed_dim=1024, depth=16, num_heads=8, dim_heads=128, ff_inner=4096,
//   qk_norm='ln' (LayerNorm affine eps 1e-6), norm_type='layer_norm' (eps 1e-5),
//   global_cond prepended as 1 token, cross_attend=True (no rope on cross-attn k),
//   rope over first rot_dim=64 dims of each head. batch=1.

#include "dit.h"
#include "profiler.h"
#include "opencl_context.h"
#include "weights.h"
#include "debug_utils.h"
#include "model_config.h"
#include "utils.h"
#include <clblast.h>
#include <cmath>
#include <string>
#include <vector>
#include <cstdint>

// ── ctor / dtor ──
DiT::DiT(OpenCLContext& cl_ctx, Weights& weights)
    : cl_ctx_(cl_ctx), weights_(weights) {}

DiT::~DiT() {
    if (prog_) clReleaseProgram(prog_);
    if (utils_) clReleaseProgram(utils_);
}

bool DiT::initialize() {
    prog_ = cl_ctx_.build_program_from_file("kernels/dit.cl");
    if (!prog_) { NNOPT_ERROR("build kernels/dit.cl failed"); return false; }
    utils_ = cl_ctx_.build_program_from_file("kernels/utils.cl");
    if (!utils_) { NNOPT_ERROR("build kernels/utils.cl failed"); return false; }
    ready_ = true;
    return true;
}

// ── small host helpers ──
static inline nnopt_storage_t enc(float v) {
#ifdef NNOPT_USE_FP16
    return nnopt_storage_t(nnopt_f32_to_f16(v));
#else
    return v;
#endif
}
static inline float dec(nnopt_storage_t v) {
#ifdef NNOPT_USE_FP16
    return nnopt_f16_to_f32(static_cast<uint16_t>(v));
#else
    return v;
#endif
}

cl_mem DiT::alloc(size_t nelem) {
    cl_int err;
    cl_mem b = nnopt_pool_alloc(cl_ctx_.context(), nelem * sizeof(nnopt_storage_t), &err);
    if (err != CL_SUCCESS) { NNOPT_ERROR_FMT("alloc %zu err=%d", nelem, err); return nullptr; }
    return b;
}

cl_mem DiT::upload(const std::vector<float>& host, size_t nelem) {
    std::vector<nnopt_storage_t> tmp(nelem);
    for (size_t i = 0; i < nelem; i++) tmp[i] = enc(i < host.size() ? host[i] : 0.0f);
    cl_mem b = alloc(nelem);
    if (!b) { NNOPT_ERROR("upload alloc failed"); return nullptr; }
    cl_int err = clEnqueueWriteBuffer(cl_ctx_.queue(), b, CL_TRUE, 0,
                                      nelem * sizeof(nnopt_storage_t), tmp.data(),
                                      0, nullptr, nullptr);
    if (err != CL_SUCCESS) { NNOPT_ERROR_FMT("upload err=%d", err); nnopt_pool_release(b); return nullptr; }
    return b;
}

void DiT::download(cl_mem buf, std::vector<float>& host, size_t nelem) {
    std::vector<nnopt_storage_t> tmp(nelem);
    clEnqueueReadBuffer(cl_ctx_.queue(), buf, CL_TRUE, 0,
                        nelem * sizeof(nnopt_storage_t), tmp.data(), 0, nullptr, nullptr);
    host.resize(nelem);
    for (size_t i = 0; i < nelem; i++) host[i] = dec(tmp[i]);
}

// out[M,N] = x[M,K] @ W[N,K]^T  (nn.Linear). Returns new buffer.
cl_mem DiT::linear(cl_mem x, const std::string& wkey, int M, int N, int K) {
    cl_mem W = weights_.get_buffer(wkey);
    if (!W) { NNOPT_ERROR_FMT("missing weight %s", wkey.c_str()); return nullptr; }
    cl_mem out = alloc((size_t)M * N);
    if (!out) return nullptr;
    if (!pytorch_linear(cl_ctx_.queue(), M, N, K, x, W, out)) {
        NNOPT_ERROR_FMT("linear %s failed", wkey.c_str());
        nnopt_pool_release(out); return nullptr;
    }
    return out;
}

// x += bias per column [cols]
static bool add_bias_inplace(cl_program prog, cl_command_queue q, cl_mem x, cl_mem bias,
                             int rows, int cols) {
    cl_int err; cl_kernel k = nnopt_cached_kernel(prog, "add_bias", &err);
    if (err != CL_SUCCESS) return false;
    bool ok = set_arg_checked(k,0,sizeof(cl_mem),&x,"x")
           && set_arg_checked(k,1,sizeof(cl_mem),&bias,"bias")
           && set_arg_checked(k,2,sizeof(int),&rows,"rows")
           && set_arg_checked(k,3,sizeof(int),&cols,"cols");
    if (ok) {
        size_t gws = (size_t)rows * cols;
        err = clEnqueueNDRangeKernel(q,k,1,nullptr,&gws,nullptr,0,nullptr,KernelProfiler::event_for("add_bias"));
        ok = (err == CL_SUCCESS);
    }
    nnopt_kernel_done(k);
    return ok;
}

void DiT::silu(cl_mem x, int n) {
    cl_int err; cl_kernel k = nnopt_cached_kernel(prog_, "silu_inplace", &err);
    if (err != CL_SUCCESS) { NNOPT_ERROR("silu kernel"); return; }
    set_arg_checked(k,0,sizeof(cl_mem),&x,"x");
    set_arg_checked(k,1,sizeof(int),&n,"n");
    size_t gws = (size_t)n;
    clEnqueueNDRangeKernel(cl_ctx_.queue(),k,1,nullptr,&gws,nullptr,0,nullptr,KernelProfiler::event_for("silu"));
    nnopt_kernel_done(k);
}

// LayerNorm over last dim D, affine (gamma,beta). Returns new buffer.
cl_mem DiT::layernorm(cl_mem x, const std::string& gkey, const std::string& bkey,
                      int rows, int D, float eps) {
    cl_mem gamma = weights_.get_buffer(gkey);
    cl_mem beta  = weights_.get_buffer(bkey);
    if (!gamma || !beta) { NNOPT_ERROR_FMT("missing norm %s/%s", gkey.c_str(), bkey.c_str()); return nullptr; }
    cl_mem out = alloc((size_t)rows * D);
    if (!out) return nullptr;
    cl_int err; cl_kernel k = nnopt_cached_kernel(prog_, "layernorm", &err);
    if (err != CL_SUCCESS) { NNOPT_ERROR("ln kernel"); nnopt_pool_release(out); return nullptr; }
    set_arg_checked(k,0,sizeof(cl_mem),&x,"x");
    set_arg_checked(k,1,sizeof(cl_mem),&gamma,"gamma");
    set_arg_checked(k,2,sizeof(cl_mem),&beta,"beta");
    set_arg_checked(k,3,sizeof(cl_mem),&out,"out");
    set_arg_checked(k,4,sizeof(int),&rows,"rows");
    set_arg_checked(k,5,sizeof(int),&D,"D");
    set_arg_checked(k,6,sizeof(float),&eps,"eps");
    size_t gws = (size_t)rows;
    clEnqueueNDRangeKernel(cl_ctx_.queue(),k,1,nullptr,&gws,nullptr,0,nullptr,KernelProfiler::event_for("layernorm"));
    nnopt_kernel_done(k);
    return out;
}

// Generic kernel dispatch helpers used by attention/block.
static bool run1(cl_program prog, cl_command_queue q, const char* name, size_t gws,
                 std::initializer_list<std::pair<size_t,const void*>> args) {
    cl_int err; cl_kernel k = nnopt_cached_kernel(prog, name, &err);
    if (err != CL_SUCCESS) { NNOPT_ERROR_FMT("kernel %s err=%d", name, err); return false; }
    unsigned idx = 0; bool ok = true;
    for (auto& a : args) { ok = ok && set_arg_checked(k, idx, a.first, a.second, name); idx++; }
    if (ok) { err = clEnqueueNDRangeKernel(q,k,1,nullptr,&gws,nullptr,0,nullptr,KernelProfiler::event_for(name)); ok = (err==CL_SUCCESS); }
    nnopt_kernel_done(k);
    if (!ok) NNOPT_ERROR_FMT("dispatch %s failed", name);
    return ok;
}

// One attention (self or cross). `normed` is the pre-normed input [Nq, D].
// For cross: context_flat is [Nk, D] (already to_cond_embed'd), is_cross=true.
// Returns attention output [Nq, D] (post to_out). freqs is [Nq,rot_dim] or null.
cl_mem DiT::attention(cl_mem normed, int layer_idx, int seq,
                      cl_mem cross_flat, int cross_seq, cl_mem freqs, bool is_cross) {
    const int D  = MODEL_CONFIG::DIT_EMBED_DIM;   // 1024
    const int H  = MODEL_CONFIG::DIT_NUM_HEADS;   // 8
    const int Dh = MODEL_CONFIG::DIT_DIM_HEADS;   // 128
    const int rot_dim = MODEL_CONFIG::DIT_ROPE_DIM; // 64
    const float qk_eps = MODEL_CONFIG::DIT_QK_NORM_EPS;
    const float scale = 1.0f / std::sqrt((float)Dh);
    cl_command_queue q = cl_ctx_.queue();
    char wp[128];
    snprintf(wp, sizeof(wp), "model.model.transformer.layers.%d.%s", layer_idx,
             is_cross ? "cross_attn" : "self_attn");

    int Nk = is_cross ? cross_seq : seq;
    cl_mem kv_src = is_cross ? cross_flat : normed;

    // ── project q,k,v ──
    // Fused projections stay ON the GPU: q/k/v are read straight out of the
    // packed GEMM output with strided kernels. The previous host-side
    // chunk(3,-1) split did a BLOCKING readback + 3 blocking writes here —
    // 256 full pipeline drains per run; it was most of the DiT's wall time.
    cl_mem qkv = nullptr;      // self: [seq, 3D]
    cl_mem q_flat = nullptr;   // cross: [seq, D]
    cl_mem kv = nullptr;       // cross: [Nk, 2D]
    int q_stride, q_off, kv_stride, k_off, v_off;
    cl_mem q_src, kv_src_buf;
    if (is_cross) {
        q_flat = linear(normed, std::string(wp)+".to_q.weight", seq, D, D);
        kv     = linear(kv_src, std::string(wp)+".to_kv.weight", Nk, 2*D, D);
        if (!q_flat || !kv) { if(q_flat)nnopt_pool_release(q_flat); if(kv)nnopt_pool_release(kv); return nullptr; }
        q_src = q_flat; q_stride = D;   q_off = 0;
        kv_src_buf = kv; kv_stride = 2*D; k_off = 0; v_off = D;
    } else {
        qkv = linear(normed, std::string(wp)+".to_qkv.weight", seq, 3*D, D);
        if (!qkv) return nullptr;
        q_src = qkv; q_stride = 3*D; q_off = 0;
        kv_src_buf = qkv; kv_stride = 3*D; k_off = D; v_off = 2*D;
    }

    // ── qk_norm='ln' per head + reshape to head-major [H,N,Dh] ──
    cl_mem q_hnd = alloc((size_t)H*seq*Dh);
    cl_mem k_hnd = alloc((size_t)H*Nk*Dh);
    cl_mem v_hnd = alloc((size_t)H*Nk*Dh);
    cl_mem qg = weights_.get_buffer(std::string(wp)+".q_norm.weight");
    cl_mem qbz= weights_.get_buffer(std::string(wp)+".q_norm.bias");
    cl_mem kg = weights_.get_buffer(std::string(wp)+".k_norm.weight");
    cl_mem kbz= weights_.get_buffer(std::string(wp)+".k_norm.bias");
    {
        run1(prog_,q,"qk_norm_to_heads",(size_t)H*seq,{
            {sizeof(cl_mem),&q_src},{sizeof(cl_mem),&qg},{sizeof(cl_mem),&qbz},
            {sizeof(cl_mem),&q_hnd},{sizeof(int),&seq},{sizeof(int),&H},{sizeof(int),&Dh},{sizeof(float),&qk_eps},
            {sizeof(int),&q_stride},{sizeof(int),&q_off}});
        run1(prog_,q,"qk_norm_to_heads",(size_t)H*Nk,{
            {sizeof(cl_mem),&kv_src_buf},{sizeof(cl_mem),&kg},{sizeof(cl_mem),&kbz},
            {sizeof(cl_mem),&k_hnd},{sizeof(int),&Nk},{sizeof(int),&H},{sizeof(int),&Dh},{sizeof(float),&qk_eps},
            {sizeof(int),&kv_stride},{sizeof(int),&k_off}});
        run1(prog_,q,"split_heads",(size_t)H*Nk*Dh,{
            {sizeof(cl_mem),&kv_src_buf},{sizeof(cl_mem),&v_hnd},{sizeof(int),&Nk},{sizeof(int),&H},{sizeof(int),&Dh},
            {sizeof(int),&kv_stride},{sizeof(int),&v_off}});
    }
    if (qkv) nnopt_pool_release(qkv);
    if (q_flat) nnopt_pool_release(q_flat);
    if (kv) nnopt_pool_release(kv);

    // ── RoPE (self-attn only; cross-attn has no rope on k in this config) ──
    if (!is_cross && freqs) {
        run1(prog_,q,"rope_apply",(size_t)H*seq,{
            {sizeof(cl_mem),&q_hnd},{sizeof(cl_mem),&freqs},
            {sizeof(int),&H},{sizeof(int),&seq},{sizeof(int),&Dh},{sizeof(int),&rot_dim}});
        run1(prog_,q,"rope_apply",(size_t)H*Nk,{
            {sizeof(cl_mem),&k_hnd},{sizeof(cl_mem),&freqs},
            {sizeof(int),&H},{sizeof(int),&Nk},{sizeof(int),&Dh},{sizeof(int),&rot_dim}});
    }

    // ── scores → softmax → AV ──
    cl_mem scores = alloc((size_t)H*seq*Nk);
    run1(prog_,q,"attn_scores",(size_t)H*seq*Nk,{
        {sizeof(cl_mem),&q_hnd},{sizeof(cl_mem),&k_hnd},{sizeof(cl_mem),&scores},
        {sizeof(int),&H},{sizeof(int),&seq},{sizeof(int),&Nk},{sizeof(int),&Dh},{sizeof(float),&scale}});
    int rows = H*seq;
    run1(prog_,q,"softmax_rows",(size_t)rows,{
        {sizeof(cl_mem),&scores},{sizeof(int),&rows},{sizeof(int),&Nk}});
    cl_mem out_hnd = alloc((size_t)H*seq*Dh);
    run1(prog_,q,"attn_av",(size_t)H*seq*Dh,{
        {sizeof(cl_mem),&scores},{sizeof(cl_mem),&v_hnd},{sizeof(cl_mem),&out_hnd},
        {sizeof(int),&H},{sizeof(int),&seq},{sizeof(int),&Nk},{sizeof(int),&Dh}});
    nnopt_pool_release(scores); nnopt_pool_release(q_hnd); nnopt_pool_release(k_hnd); nnopt_pool_release(v_hnd);

    // merge heads → [seq, D]
    cl_mem merged = alloc((size_t)seq*D);
    run1(prog_,q,"merge_heads",(size_t)H*seq*Dh,{
        {sizeof(cl_mem),&out_hnd},{sizeof(cl_mem),&merged},{sizeof(int),&seq},{sizeof(int),&H},{sizeof(int),&Dh}});
    nnopt_pool_release(out_hnd);

    // to_out
    cl_mem attn_out = linear(merged, std::string(wp)+".to_out.weight", seq, D, D);
    nnopt_pool_release(merged);
    return attn_out;
}

// One transformer block, in-place semantics on x_flat [seq,D]; returns NEW x buffer.
cl_mem DiT::block(cl_mem x_flat, int layer_idx, int seq,
                  cl_mem cross_flat, int cross_seq, cl_mem freqs) {
    const int D = MODEL_CONFIG::DIT_EMBED_DIM;
    const int inner = MODEL_CONFIG::DIT_FF_INNER; // 4096
    const float eps = MODEL_CONFIG::DIT_NORM_EPS;
    cl_command_queue q = cl_ctx_.queue();
    char pfx[96]; snprintf(pfx,sizeof(pfx),"model.model.transformer.layers.%d",layer_idx);

    // x = x + self_attn(pre_norm(x))
    cl_mem pn = layernorm(x_flat, std::string(pfx)+".pre_norm.gamma", std::string(pfx)+".pre_norm.beta", seq, D, eps);
    if (!pn) return nullptr;
    cl_mem sa = attention(pn, layer_idx, seq, nullptr, 0, freqs, false);
    nnopt_pool_release(pn);
    if (!sa) return nullptr;
    cl_mem x1 = element_add(q, utils_, x_flat, sa, (size_t)seq*D);
    nnopt_pool_release(sa); nnopt_pool_release(x_flat);

    // x = x + cross_attn(cross_attend_norm(x), context)
    cl_mem cn = layernorm(x1, std::string(pfx)+".cross_attend_norm.gamma", std::string(pfx)+".cross_attend_norm.beta", seq, D, eps);
    if (!cn) { nnopt_pool_release(x1); return nullptr; }
    cl_mem ca = attention(cn, layer_idx, seq, cross_flat, cross_seq, nullptr, true);
    nnopt_pool_release(cn);
    if (!ca) { nnopt_pool_release(x1); return nullptr; }
    cl_mem x2 = element_add(q, utils_, x1, ca, (size_t)seq*D);
    nnopt_pool_release(ca); nnopt_pool_release(x1);

    // x = x + ff(ff_norm(x))  — FeedForward: GLU(proj->2*inner, value*silu(gate)) then linear inner->D
    cl_mem fn = layernorm(x2, std::string(pfx)+".ff_norm.gamma", std::string(pfx)+".ff_norm.beta", seq, D, eps);
    if (!fn) { nnopt_pool_release(x2); return nullptr; }
    // proj = fn @ ff.0.proj.weight^T + bias  -> [seq, 2*inner]
    cl_mem proj = linear(fn, std::string(pfx)+".ff.ff.0.proj.weight", seq, 2*inner, D);
    nnopt_pool_release(fn);
    if (!proj) { nnopt_pool_release(x2); return nullptr; }
    cl_mem pbias = weights_.get_buffer(std::string(pfx)+".ff.ff.0.proj.bias");
    if (pbias) add_bias_inplace(prog_, q, proj, pbias, seq, 2*inner);
    cl_mem glu = alloc((size_t)seq*inner);
    run1(prog_,q,"swiglu",(size_t)seq*inner,{
        {sizeof(cl_mem),&proj},{sizeof(cl_mem),&glu},{sizeof(int),&seq},{sizeof(int),&inner}});
    nnopt_pool_release(proj);
    // ff.2: linear inner->D + bias
    cl_mem ffout = linear(glu, std::string(pfx)+".ff.ff.2.weight", seq, D, inner);
    nnopt_pool_release(glu);
    if (!ffout) { nnopt_pool_release(x2); return nullptr; }
    cl_mem obias = weights_.get_buffer(std::string(pfx)+".ff.ff.2.bias");
    if (obias) add_bias_inplace(prog_, q, ffout, obias, seq, D);
    cl_mem x3 = element_add(q, utils_, x2, ffout, (size_t)seq*D);
    nnopt_pool_release(ffout); nnopt_pool_release(x2);

    char dn[96]; snprintf(dn,sizeof(dn),"transformer_layers_%d",layer_idx);
    NNOPT_LAYER_CHECK(dn, q, x3, (size_t)seq*D);
    return x3;
}

bool DiT::forward_step(const std::vector<float>& x_latent,
                       float t,
                       const std::vector<float>& cross_cond, int cross_seq,
                       const std::vector<float>& global_emb,
                       std::vector<float>& out) {
    if (!ready_ && !initialize()) return false;
    cl_command_queue q = cl_ctx_.queue();
    const int D = MODEL_CONFIG::DIT_EMBED_DIM;      // 1024
    const int C = MODEL_CONFIG::DIT_LATENT_CHANNELS;// 64
    const int cond_dim = MODEL_CONFIG::DIT_COND_TOKEN_DIM; // 768
    const int gcond_dim = MODEL_CONFIG::DIT_GLOBAL_COND_DIM;//768
    const int ff_half = MODEL_CONFIG::DIT_TIMESTEP_FEAT/2;  // 128 (fourier half_out)
    const int tfeat = MODEL_CONFIG::DIT_TIMESTEP_FEAT;      // 256
    const float eps = MODEL_CONFIG::DIT_NORM_EPS;
    int T = (int)x_latent.size() / C;               // latent_len

    // ── to_cond_embed(cross_cond): Linear(768->1024,nobias) -> SiLU -> Linear(1024->1024,nobias) ──
    cl_mem cc = upload(cross_cond, (size_t)cross_seq*cond_dim);
    cl_mem c0 = linear(cc, "model.model.to_cond_embed.0.weight", cross_seq, D, cond_dim);
    nnopt_pool_release(cc);
    if (!c0) return false;
    silu(c0, cross_seq*D);
    cl_mem cross_flat = linear(c0, "model.model.to_cond_embed.2.weight", cross_seq, D, D);
    nnopt_pool_release(c0);
    if (!cross_flat) return false;
    NNOPT_LAYER_CHECK("to_cond_embed", q, cross_flat, (size_t)cross_seq*D);

    // ── to_global_embed(global_emb): Linear(768->1024) SiLU Linear(1024->1024) ──
    cl_mem ge = upload(global_emb, (size_t)gcond_dim);
    cl_mem g0 = linear(ge, "model.model.to_global_embed.0.weight", 1, D, gcond_dim);
    nnopt_pool_release(ge);
    if (!g0) { nnopt_pool_release(cross_flat); return false; }
    silu(g0, D);
    cl_mem global_e = linear(g0, "model.model.to_global_embed.2.weight", 1, D, D);
    nnopt_pool_release(g0);
    if (!global_e) { nnopt_pool_release(cross_flat); return false; }
    NNOPT_LAYER_CHECK("to_global_embed", q, global_e, (size_t)D);

    // ── timestep_embed = to_timestep_embed(fourier(t)) ──
    // FourierFeatures: out[256] = cat(cos(2pi t W), sin(2pi t W)), W=[128,1]
    std::vector<float> tvec(1, t);
    cl_mem tbuf = upload(tvec, 1);
    cl_mem fW = weights_.get_buffer("model.model.timestep_features.weight");
    cl_mem ff = alloc((size_t)tfeat);
    {
        int rows1 = 1;
        run1(prog_,q,"fourier_features",(size_t)ff_half,{
            {sizeof(cl_mem),&tbuf},{sizeof(cl_mem),&fW},{sizeof(cl_mem),&ff},
            {sizeof(int),&rows1},{sizeof(int),&ff_half}});
    }
    nnopt_pool_release(tbuf);
    NNOPT_LAYER_CHECK("timestep_features", q, ff, (size_t)tfeat);
    // to_timestep_embed.0: Linear(256->1024)+bias
    cl_mem te0 = linear(ff, "model.model.to_timestep_embed.0.weight", 1, D, tfeat);
    nnopt_pool_release(ff);
    if (!te0) { nnopt_pool_release(cross_flat); nnopt_pool_release(global_e); return false; }
    cl_mem te0b = weights_.get_buffer("model.model.to_timestep_embed.0.bias");
    if (te0b) add_bias_inplace(prog_,q,te0,te0b,1,D);
    silu(te0, D);
    cl_mem tstep = linear(te0, "model.model.to_timestep_embed.2.weight", 1, D, D);
    nnopt_pool_release(te0);
    if (!tstep) { nnopt_pool_release(cross_flat); nnopt_pool_release(global_e); return false; }
    cl_mem te2b = weights_.get_buffer("model.model.to_timestep_embed.2.bias");
    if (te2b) add_bias_inplace(prog_,q,tstep,te2b,1,D);
    NNOPT_LAYER_CHECK("to_timestep_embed", q, tstep, (size_t)D);

    // global_embed = global_embed + timestep_embed  → [D]
    cl_mem gsum = element_add(q, utils_, global_e, tstep, (size_t)D);
    nnopt_pool_release(global_e); nnopt_pool_release(tstep);

    // ── preprocess_conv(x)+x : Conv1d k=1 over [C,T]; conv1d k=1 == per-timestep linear W[C,C] ──
    // x_latent is [C,T]. preprocess_conv.weight [64,64,1]. out[c,t] = sum_ci W[c,ci] * x[ci,t].
    // Reuse pytorch_linear by treating time as rows: need x as [T,C]. Transpose first.
    cl_mem x_ct = upload(x_latent, (size_t)C*T);           // [C,T]
    cl_mem x_tc = alloc((size_t)T*C);
    run1(prog_,q,"transpose_ct",(size_t)C*T,{
        {sizeof(cl_mem),&x_ct},{sizeof(cl_mem),&x_tc},{sizeof(int),&C},{sizeof(int),&T}}); // [T,C]
    // preprocess_conv.weight [C_out,C_in,1] treated as [C,C] nn.Linear-like -> [T,C]
    cl_mem pre = linear(x_tc, "model.model.preprocess_conv.weight", T, C, C);
    if (!pre) { nnopt_pool_release(x_ct);nnopt_pool_release(x_tc);nnopt_pool_release(cross_flat);nnopt_pool_release(gsum); return false; }
    cl_mem x_pre = element_add(q, utils_, x_tc, pre, (size_t)T*C); // [T,C]  (x = conv(x)+x)
    nnopt_pool_release(pre); nnopt_pool_release(x_ct); nnopt_pool_release(x_tc);
    NNOPT_LAYER_CHECK("preprocess_conv", q, x_pre, (size_t)T*C);

    // ── transformer.project_in: Linear(64->1024,nobias) over [T,C] -> [T,D] ──
    cl_mem proj_in = linear(x_pre, "model.model.transformer.project_in.weight", T, D, C);
    nnopt_pool_release(x_pre);
    if (!proj_in) { nnopt_pool_release(cross_flat);nnopt_pool_release(gsum); return false; }
    NNOPT_LAYER_CHECK("transformer_project_in", q, proj_in, (size_t)T*D);

    // ── prepend global token: x = cat([global_embed[None], x]) -> [T+1, D] ──
    int seq = T + 1;
    cl_mem x_seq = alloc((size_t)seq*D);
    run1(prog_,q,"prepend_row",(size_t)seq*D,{
        {sizeof(cl_mem),&gsum},{sizeof(cl_mem),&proj_in},{sizeof(cl_mem),&x_seq},
        {sizeof(int),&T},{sizeof(int),&D}});
    nnopt_pool_release(proj_in); nnopt_pool_release(gsum);

    // ── rope freqs [seq, rot_dim] from inv_freq[32] ──
    const int rot_dim = MODEL_CONFIG::DIT_ROPE_DIM;   // 64
    std::vector<float> inv = weights_.get_host_vec("model.model.transformer.rotary_pos_emb.inv_freq"); // [32]
    int half = rot_dim/2; // 32
    std::vector<float> freqs_h((size_t)seq*rot_dim);
    for (int n=0;n<seq;n++){
        for(int i=0;i<half;i++){
            float f = (float)n * inv[i];
            freqs_h[(size_t)n*rot_dim + i] = f;
            freqs_h[(size_t)n*rot_dim + half + i] = f; // cat(freqs,freqs)
        }
    }
    cl_mem freqs = upload(freqs_h, (size_t)seq*rot_dim);

    // ── 16 transformer blocks ──
    cl_mem x = x_seq;
    for (int L=0; L<MODEL_CONFIG::DIT_DEPTH; L++) {
        x = block(x, L, seq, cross_flat, cross_seq, freqs);
        if (!x) { nnopt_pool_release(freqs); nnopt_pool_release(cross_flat); return false; }
    }
    nnopt_pool_release(freqs); nnopt_pool_release(cross_flat);

    // ── project_out: Linear(1024->64,nobias) [seq,D]->[seq,C] ──
    cl_mem projo = linear(x, "model.model.transformer.project_out.weight", seq, C, D);
    nnopt_pool_release(x);
    if (!projo) return false;
    NNOPT_LAYER_CHECK("transformer_project_out", q, projo, (size_t)seq*C);

    // ── drop prepend row: [seq,C] -> [T,C] ── then transpose to [C,T]
    cl_mem dropped = alloc((size_t)T*C);
    { int drop=1;
      run1(prog_,q,"drop_rows",(size_t)T*C,{
        {sizeof(cl_mem),&projo},{sizeof(cl_mem),&dropped},{sizeof(int),&T},{sizeof(int),&C},{sizeof(int),&drop}});
    }
    nnopt_pool_release(projo);
    cl_mem out_ct = alloc((size_t)C*T);
    run1(prog_,q,"transpose_ct",(size_t)T*C,{
        {sizeof(cl_mem),&dropped},{sizeof(cl_mem),&out_ct},{sizeof(int),&T},{sizeof(int),&C}}); // [T,C]->[C,T]
    nnopt_pool_release(dropped);

    // ── postprocess_conv(out)+out : conv1d k=1 [C,C] over [C,T] ──
    cl_mem out_tc = alloc((size_t)T*C);
    run1(prog_,q,"transpose_ct",(size_t)C*T,{
        {sizeof(cl_mem),&out_ct},{sizeof(cl_mem),&out_tc},{sizeof(int),&C},{sizeof(int),&T}}); // [C,T]->[T,C]
    cl_mem post = linear(out_tc, "model.model.postprocess_conv.weight", T, C, C);
    if (!post) { nnopt_pool_release(out_ct);nnopt_pool_release(out_tc); return false; }
    cl_mem final_tc = element_add(q, utils_, out_tc, post, (size_t)T*C);
    nnopt_pool_release(post); nnopt_pool_release(out_tc);
    // transpose back to [C,T]
    cl_mem final_ct = alloc((size_t)C*T);
    run1(prog_,q,"transpose_ct",(size_t)T*C,{
        {sizeof(cl_mem),&final_tc},{sizeof(cl_mem),&final_ct},{sizeof(int),&T},{sizeof(int),&C}});
    nnopt_pool_release(final_tc); nnopt_pool_release(out_ct);

    NNOPT_LAYER_CHECK("dit_output", q, final_ct, (size_t)C*T);
    download(final_ct, out, (size_t)C*T);
    nnopt_pool_release(final_ct);
    return true;
}
