// Hand-port of google/magenta-realtime-2 DepthFormer forward (OpenCL GPU, fp16).
#include "version.h"
#include <unistd.h>
#include <algorithm>
#include <cctype>
#include <cmath>
#include <deque>
#include <condition_variable>
#include <atomic>
#include <mutex>
#include <thread>
// Built op-by-op against reference/forward_graph.json + reference/layers/*.bin,
// using the dequantized fp16 weights in weights/model.fp16.bin (keyed by MLX
// module path, e.g. "depthformer.decoder.embedder.layers.0._embedding.weight").
//
// Precision plan: weights stored fp16 (vload_half in kernels); activations fp32.
//
// PROGRESS: T2 layer 0 (decoder embedder). Subsequent nodes appended in order.

#include "opencl_context.h"
#include "weights.h"
#include "debug_utils.h"
#include "model_config.h"
#include "utils.h"
#include "song.h"

#include <CL/cl.h>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <string>
#include <vector>
#include <chrono>
#include <sys/resource.h>
#ifdef USE_CLBLAST
#include <unordered_map>
#include <cstring>
#include <clblast_c.h>
#include <clblast.h>      // clblast::OverrideParameters — per-device GEMM tuning, see maybe_override_xgemm

// Defined below; declared here because conv_clblast dispatches to it.
#ifdef USE_CLBLAST
long nnopt_adreno_model(OpenCLContext& cl_ctx);
static cl_mem conv_clblast_fp16(OpenCLContext&, cl_command_queue, cl_mem, cl_mem, cl_mem,
                                int, int, int, int, int, int, int, int, int, int, int, int, int,
                                bool, int, int, int, bool, bool, bool*);
#endif

#endif

// Op implementations live in src/ops/*.cpp (extern "C").
extern "C" cl_mem QuantizedEmbedding_forward(
    OpenCLContext& cl_ctx, Weights& weights, cl_command_queue queue,
    cl_mem input, int seq_len, int layer_idx, int start_pos,
    cl_mem* k_cache_inout, cl_mem* v_cache_inout, cl_mem encoder_hidden_states,
    const char* weight_prefix);
extern "C" cl_mem RMSNorm_forward(
    OpenCLContext& cl_ctx, Weights& weights, cl_command_queue queue,
    cl_mem input, int seq_len, int layer_idx, int start_pos,
    cl_mem* k_cache_inout, cl_mem* v_cache_inout, cl_mem encoder_hidden_states,
    const char* weight_prefix);
extern "C" cl_mem QuantizedLinear_forward(
    OpenCLContext& cl_ctx, Weights& weights, cl_command_queue queue,
    cl_mem input, int seq_len, int layer_idx, int start_pos,
    cl_mem* k_cache_inout, cl_mem* v_cache_inout, cl_mem encoder_hidden_states,
    const char* weight_prefix);
extern "C" cl_mem SinkAttention_forward(
    OpenCLContext& cl_ctx, Weights& weights, cl_command_queue queue,
    cl_mem input, int seq_len, int layer_idx, int start_pos,
    cl_mem* k_cache_inout, cl_mem* v_cache_inout, cl_mem encoder_hidden_states,
    const char* weight_prefix);
extern "C" cl_mem MLP_forward(
    OpenCLContext& cl_ctx, Weights& weights, cl_command_queue queue,
    cl_mem input, int seq_len, int layer_idx, int start_pos,
    cl_mem* k_cache_inout, cl_mem* v_cache_inout, cl_mem encoder_hidden_states,
    const char* weight_prefix, const char* prenorm_prefix = nullptr);
extern "C" cl_mem CrossAttention_forward(
    OpenCLContext& cl_ctx, Weights& weights, cl_command_queue queue,
    cl_mem input, int seq_len, int layer_idx, int start_pos,
    cl_mem* k_cache_inout, cl_mem* v_cache_inout, cl_mem encoder_hidden_states,
    const char* weight_prefix, const char* prenorm_prefix = nullptr);
extern "C" cl_mem LayerNorm_forward(
    OpenCLContext& cl_ctx, Weights& weights, cl_command_queue queue,
    cl_mem input, int seq_len, int layer_idx, int start_pos,
    cl_mem* k_cache_inout, cl_mem* v_cache_inout, cl_mem encoder_hidden_states,
    const char* weight_prefix);
extern "C" cl_mem Encoder_forward(
    OpenCLContext& cl_ctx, Weights& weights, cl_command_queue queue,
    cl_mem tokens, int seq_len, int layer_idx, int start_pos,
    cl_mem* k_cache_inout, cl_mem* v_cache_inout, cl_mem encoder_hidden_states,
    const char* weight_prefix);
extern "C" cl_mem CachedSelfAttention_forward(
    OpenCLContext& cl_ctx, Weights& weights, cl_command_queue queue,
    cl_mem input, int seq_len, int layer_idx, int start_pos,
    cl_mem* k_cache_inout, cl_mem* v_cache_inout, cl_mem encoder_hidden_states,
    const char* weight_prefix, const char* prenorm_prefix = nullptr);

// ── residual add (a+b) into a fresh buffer ──
static cl_mem residual_add(OpenCLContext& cl_ctx, cl_command_queue queue, cl_mem a, cl_mem b, int n) {
    static cl_kernel k = nullptr; cl_int err = CL_SUCCESS;
    if (!k) { cl_program p = cl_ctx.build_program_from_file("kernels/add_f32.cl");
        if (!p) { NNOPT_ERROR("residual_add: build add_f32 failed"); return nullptr; }
        k = clCreateKernel(p, "add_f32", &err); if (!k) return nullptr; }
    cl_mem out = pool_alloc(cl_ctx.context(), CL_MEM_READ_WRITE, (size_t)n*sizeof(float), nullptr, &err);
    // Per-call-site kernel object, so a recorded frame's dispatches keep their own args.
    static const std::string kResidualAdd("residual_add");   // static: no per-dispatch allocation
    k = nnopt_kernel_instance(k, kResidualAdd);
    clSetKernelArg(k,0,sizeof(cl_mem),&a); clSetKernelArg(k,1,sizeof(cl_mem),&b);
    clSetKernelArg(k,2,sizeof(cl_mem),&out); clSetKernelArg(k,3,sizeof(int),&n);
    size_t g=(size_t)n; cl_ctx.profEnqueue(k,1,&g,nullptr,"add");
     return out;
}

// ── fused RMSNorm + residual-add (AR layer-fusion): out = residual + rmsnorm(x)*scale, ONE enqueue ──
// Replaces the per-block `post=RMSNorm(x); out=residual_add(resid,post)` (2 enqueues) — the AR is
// host-issue-bound so this directly cuts wall. wp is the rms_norm weight prefix; seq_len assumed 1 (AR).
static cl_mem rmsnorm_add(OpenCLContext& cl_ctx, Weights& weights, cl_command_queue queue,
                          cl_mem x, cl_mem residual, int dim, const std::string& wp) {
    (void)queue;
    // Serial (1 work item) vs workgroup-parallel. The serial kernel walks 768–1024 elements twice on
    // ONE lane of ONE compute unit — 384 us/call, 84 calls/frame — because Adreno cannot spread a
    // workgroup across compute units (80-NB295-11 §3.2.5), so a 1-item dispatch is 1/12th of the GPU
    // at 1/64th of a wave. NNOPT_RMS=serial restores the old kernel for A/B.
    static int par = -1, par_epoch = -1;
    if (par_epoch != nnopt_toggle_epoch()) {
        const char* r = std::getenv("NNOPT_RMS");
        par = (r && std::strcmp(r, "serial") == 0) ? 0 : 1;
        par_epoch = nnopt_toggle_epoch();
    }
    static cl_kernel k_ser=nullptr, k_par=nullptr; cl_int err=CL_SUCCESS;
    cl_kernel& slot = par ? k_par : k_ser;
    if(!slot){
        const char* file = par ? "kernels/rms_norm_add_wg_f32.cl" : "kernels/rms_norm_add_f32.cl";
        cl_program p=cl_ctx.build_program_from_file(file);
        if(!p){NNOPT_ERROR("rmsnorm_add: build failed");return nullptr;}
        slot=clCreateKernel(p, par ? "rms_norm_add_wg_f32" : "rms_norm_add_f32",&err);
        if(!slot){NNOPT_ERROR_FMT("rmsnorm_add: clCreateKernel (err=%d)",err);return nullptr;}
    }
    cl_kernel k = nnopt_kernel_instance(slot, wp, ".rms_add");
    cl_mem scale=weights.get_buffer(wp+".weight");
    if(!scale){ NNOPT_ERROR_FMT("rmsnorm_add: missing %s.weight", wp.c_str()); return nullptr; }
    cl_mem out=pool_alloc(cl_ctx.context(),CL_MEM_READ_WRITE,(size_t)dim*sizeof(float),nullptr,&err);
    const float eps=1e-6f;
    clSetKernelArg(k,0,sizeof(cl_mem),&x); clSetKernelArg(k,1,sizeof(cl_mem),&scale); clSetKernelArg(k,2,sizeof(cl_mem),&residual);
    clSetKernelArg(k,3,sizeof(cl_mem),&out); clSetKernelArg(k,4,sizeof(int),&dim); clSetKernelArg(k,5,sizeof(float),&eps);
    if (par) {
        const size_t g=(size_t)128, l=(size_t)128;   // one 128-wide workgroup per row (rows==1 in AR)
        cl_ctx.profEnqueue(k,1,&g,&l,"rms_add");
    } else {
        size_t g=(size_t)1; cl_ctx.profEnqueue(k,1,&g,nullptr,"rms_add");
    }
    return out;
}

// ── one attention Residual block: x + rms2(out_proj(attn(rms1(x)))) ──
// body.layers: 0=rms1, 1=attn(inner), 2=out_proj(EinsumDense), 4=rms2.
// is_cross: use CrossAttention(source) instead of SinkAttention.
static cl_mem attn_block(OpenCLContext& cl_ctx, Weights& weights, cl_command_queue queue,
                         cl_mem x, const std::string& body, int dim, bool is_cross, cl_mem source) {
    const std::string rms_pfx = body + ".layers.0._rms_norm";
    cl_mem rms1 = nullptr, attn = nullptr;
    // Cross-attention folds its pre-norm into q_proj; the sink path keeps the separate norm (its
    // projection is shared with a code path that has no key to hand down).
    if (is_cross && nnopt_fuse_level() >= 3) {
        attn = CrossAttention_forward(cl_ctx,weights,queue,x,1,0,0,nullptr,nullptr,source,
                                      (body+".layers.1.inner").c_str(), rms_pfx.c_str());
    } else {
        rms1 = RMSNorm_forward(cl_ctx,weights,queue,x,1,0,0,nullptr,nullptr,nullptr,rms_pfx.c_str());
        if (!rms1) return nullptr;
        attn = is_cross
            ? CrossAttention_forward(cl_ctx,weights,queue,rms1,1,0,0,nullptr,nullptr,source,(body+".layers.1.inner").c_str())
            : SinkAttention_forward (cl_ctx,weights,queue,rms1,1,0,0,nullptr,nullptr,nullptr,(body+".layers.1.inner").c_str());
        pool_free(rms1);
    }
    if (!attn) return nullptr;
    cl_mem proj = QuantizedLinear_forward(cl_ctx,weights,queue,attn,1,0,0,nullptr,nullptr,nullptr,(body+".layers.2").c_str());
    pool_free(attn);
    if (!proj) return nullptr;
    // fused: out = x + rms2(proj)  (was RMSNorm + residual_add — 2 enqueues → 1)
    cl_mem out = rmsnorm_add(cl_ctx,weights,queue,proj,x,dim,body+".layers.4._rms_norm");
    pool_free(proj);
    return out;
}

// ── one MLP Residual block: x + rms2(dense2(gelu(dense1(rms1(x))))) ──
// body.layers: 0=rms1, 1=dense1, 3=dense2, 5=rms2.
static cl_mem mlp_block(OpenCLContext& cl_ctx, Weights& weights, cl_command_queue queue,
                        cl_mem x, const std::string& body, int dim) {
    // The pre-norm is handed to MLP_forward as a KEY rather than materialised: dense1 applies it
    // inside its own GEMV, so rms + dense1 + gelu collapse from three dispatches into one.
    // NNOPT_FUSE=0 makes MLP_forward run them separately again, and this path materialises rms1 to
    // match, so the A/B compares like with like.
    const int fuse_blk = nnopt_fuse_level();
    const std::string rms_pfx = body + ".layers.0._rms_norm";
    cl_mem ff = nullptr;
    if (fuse_blk >= 2) {
        ff = MLP_forward(cl_ctx,weights,queue,x,1,0,0,nullptr,nullptr,nullptr,body.c_str(),rms_pfx.c_str());
    } else {
        cl_mem rms1 = RMSNorm_forward(cl_ctx,weights,queue,x,1,0,0,nullptr,nullptr,nullptr,rms_pfx.c_str());
        if (!rms1) return nullptr;
        ff = MLP_forward(cl_ctx,weights,queue,rms1,1,0,0,nullptr,nullptr,nullptr,body.c_str());
        pool_free(rms1);
    }
    if (!ff) return nullptr;
    // fused: out = x + rms2(ff)  (RMSNorm + residual_add → 1 enqueue)
    cl_mem out = rmsnorm_add(cl_ctx,weights,queue,ff,x,dim,body+".layers.5._rms_norm");
    pool_free(ff);
    return out;
}

// ── full temporal_body: 12 layers × [self-sink-attn, cross-attn(source), MLP] ──
static cl_mem run_temporal_body(OpenCLContext& cl_ctx, Weights& weights, cl_command_queue queue,
                                cl_mem x_in, cl_mem source, int dim, int num_layers) {
    const std::string base = "depthformer.decoder.temporal_body.layers.0.layers.";
    cl_mem x = pool_alloc(cl_ctx.context(), CL_MEM_READ_WRITE, (size_t)dim*sizeof(float), nullptr, nullptr);
    clEnqueueCopyBuffer(queue, x_in, x, 0,0,(size_t)dim*sizeof(float),0,nullptr,nullptr); 
    for (int k = 0; k < num_layers; ++k) {
        const std::string L = base + std::to_string(k) + ".layers.";
        cl_mem a = attn_block(cl_ctx,weights,queue,x,L+"0.body",dim,false,nullptr); if(!a){pool_free(x);return nullptr;}
        pool_free(x); x=a;
        cl_mem c = attn_block(cl_ctx,weights,queue,x,L+"1.body",dim,true,source);  if(!c){pool_free(x);return nullptr;}
        pool_free(x); x=c;
        cl_mem mm = mlp_block(cl_ctx,weights,queue,x,L+"2.body",dim);              if(!mm){pool_free(x);return nullptr;}
        pool_free(x); x=mm;
    }
    return x;
}

// ── depth_body: adapter(1024→768) → 2× [self-attn, MLP] (dim 768) → LayerNorm → logits head(768→12294) ──
// Returns logits [vocab]. Depth blocks = Serial[3] = [self-attn(.0), Identity(.1), MLP(.2)].
static cl_mem run_depth_body(OpenCLContext& cl_ctx, Weights& weights, cl_command_queue queue, cl_mem x_in) {
    const std::string D = "depthformer.decoder.depth_body.";
    cl_mem h = QuantizedLinear_forward(cl_ctx,weights,queue,x_in,1,0,0,nullptr,nullptr,nullptr,(D+"layers.0.inner._linear").c_str());
    if (!h) { NNOPT_ERROR("depth: adapter failed"); return nullptr; }
    const int dim = weights.get_shape(D+"layers.0.inner._linear.weight")[0];  // 768
    for (int k = 0; k < 2; ++k) {
        const std::string L = D + "layers.1.layers." + std::to_string(k) + ".layers.";
        cl_mem a = attn_block(cl_ctx,weights,queue,h,L+"0.body",dim,false,nullptr); if(!a){pool_free(h);return nullptr;}
        pool_free(h); h=a;
        cl_mem mm = mlp_block(cl_ctx,weights,queue,h,L+"2.body",dim);              if(!mm){pool_free(h);return nullptr;}
        pool_free(h); h=mm;
    }
    cl_mem ln = LayerNorm_forward(cl_ctx,weights,queue,h,1,0,0,nullptr,nullptr,nullptr,(D+"layers.2._layer_norm").c_str());
    pool_free(h);
    if (!ln) { NNOPT_ERROR("depth: layernorm failed"); return nullptr; }
    cl_mem logits = QuantizedLinear_forward(cl_ctx,weights,queue,ln,1,0,0,nullptr,nullptr,nullptr,(D+"layers.3.inner._linear").c_str());
    pool_free(ln);
    return logits;  // [12294]
}

// ── one attention Residual block with a CACHED self-attn (temporal across frames) ──
static cl_mem attn_block_cached(OpenCLContext& cl_ctx, Weights& weights, cl_command_queue queue,
                                cl_mem x, const std::string& body, int dim,
                                cl_mem* kc, cl_mem* vc, int pos) {
    // Pre-norm folded into qkv_proj at fuse>=2, exactly as the MLP block folds it into dense1 —
    // same kernel, same PRENORM path. One fewer dispatch per attention block, 36 per frame.
    const std::string rms_pfx = body + ".layers.0._rms_norm";
    cl_mem rms1 = nullptr, attn = nullptr;
    if (nnopt_fuse_level() >= 3) {
        attn = CachedSelfAttention_forward(cl_ctx,weights,queue,x,1,0,pos,kc,vc,nullptr,
                                           (body+".layers.1.inner").c_str(), rms_pfx.c_str());
    } else {
        rms1 = RMSNorm_forward(cl_ctx,weights,queue,x,1,0,0,nullptr,nullptr,nullptr,rms_pfx.c_str());
        if (!rms1) return nullptr;
        attn = CachedSelfAttention_forward(cl_ctx,weights,queue,rms1,1,0,pos,kc,vc,nullptr,
                                           (body+".layers.1.inner").c_str());
        pool_free(rms1);
    }
    if (!attn) return nullptr;
    cl_mem proj = QuantizedLinear_forward(cl_ctx,weights,queue,attn,1,0,0,nullptr,nullptr,nullptr,(body+".layers.2").c_str());
    pool_free(attn);
    // fused: out = x + rms2(proj)
    cl_mem out = rmsnorm_add(cl_ctx,weights,queue,proj,x,dim,body+".layers.4._rms_norm");
    pool_free(proj);
    return out;
}

// ── one temporal STEP (frame): 12 layers × [cached self-attn(pos), cross-attn(source), MLP] ──
static cl_mem run_temporal_step(OpenCLContext& cl_ctx, Weights& weights, cl_command_queue queue,
                                cl_mem x_in, cl_mem source, cl_mem* kc, cl_mem* vc, int dim, int pos) {
    const std::string base = "depthformer.decoder.temporal_body.layers.0.layers.";
    cl_mem x = pool_alloc(cl_ctx.context(), CL_MEM_READ_WRITE, (size_t)dim*sizeof(float), nullptr, nullptr);
    { static const std::string kSite("temporal.x_in");
      if (!nnopt_copy_buffer(cl_ctx, x_in, x, 0, 0, dim, kSite, nullptr)) { pool_free(x); return nullptr; } }
    for (int k=0; k<12; ++k) {
        const std::string L = base + std::to_string(k) + ".layers.";
        cl_mem a = attn_block_cached(cl_ctx,weights,queue,x,L+"0.body",dim,&kc[k],&vc[k],pos); if(!a){pool_free(x);return nullptr;}
        pool_free(x); x=a;
        cl_mem c = attn_block(cl_ctx,weights,queue,x,L+"1.body",dim,true,source);              if(!c){pool_free(x);return nullptr;}
        pool_free(x); x=c;
        cl_mem mm = mlp_block(cl_ctx,weights,queue,x,L+"2.body",dim);                           if(!mm){pool_free(x);return nullptr;}
        pool_free(x); x=mm;
    }
    return x;
}

// ── one depth STEP for codebook q: adapter → 2× [cached self-attn, MLP] → LayerNorm → logits ──
// kc/vc hold per-layer K/V caches (over codebooks). pos = q.
static cl_mem run_depth_step(OpenCLContext& cl_ctx, Weights& weights, cl_command_queue queue,
                             cl_mem depth_input, cl_mem* kc, cl_mem* vc, int q, int lo, int count) {
    const std::string D = "depthformer.decoder.depth_body.";
    cl_mem h = QuantizedLinear_forward(cl_ctx,weights,queue,depth_input,1,0,0,nullptr,nullptr,nullptr,(D+"layers.0.inner._linear").c_str());
    if (!h) return nullptr;
    const int dim = weights.get_shape(D+"layers.0.inner._linear.weight")[0]; // 768
    for (int L = 0; L < 2; ++L) {
        const std::string B = D + "layers.1.layers." + std::to_string(L) + ".layers.0.body";
        // The depth body runs 12x per frame, so a dispatch removed here is removed twelve times.
        const std::string dr = B + ".layers.0._rms_norm";
        cl_mem rms1 = nullptr, attn = nullptr;
        if (nnopt_fuse_level() >= 3) {
            attn = CachedSelfAttention_forward(cl_ctx,weights,queue,h,1,0,q,&kc[L],&vc[L],nullptr,
                                               (B+".layers.1.inner").c_str(), dr.c_str());
        } else {
            rms1 = RMSNorm_forward(cl_ctx,weights,queue,h,1,0,0,nullptr,nullptr,nullptr,dr.c_str());
            attn = CachedSelfAttention_forward(cl_ctx,weights,queue,rms1,1,0,q,&kc[L],&vc[L],nullptr,
                                               (B+".layers.1.inner").c_str());
        }
        pool_free(rms1);
        cl_mem proj = QuantizedLinear_forward(cl_ctx,weights,queue,attn,1,0,0,nullptr,nullptr,nullptr,(B+".layers.2").c_str());
        pool_free(attn);
        // fused: h = h + rms2(proj)
        cl_mem h2 = rmsnorm_add(cl_ctx,weights,queue,proj,h,dim,B+".layers.4._rms_norm"); pool_free(proj); pool_free(h); h=h2;
        cl_mem mm = mlp_block(cl_ctx,weights,queue,h,D+"layers.1.layers."+std::to_string(L)+".layers.2.body",dim);
        pool_free(h); h=mm;
    }
    cl_mem ln = LayerNorm_forward(cl_ctx,weights,queue,h,1,0,0,nullptr,nullptr,nullptr,(D+"layers.2._layer_norm").c_str());
    pool_free(h);
    // Logits head: only this codebook's [lo, lo+count) slice is ever consumed (argmax). The full
    // 12294-row head re-streams 18.9MB of weights/codebook for outputs we discard — slice it (12× less DRAM).
    static cl_kernel slk=nullptr;
    if(!slk){ cl_program p=cl_ctx.build_program_from_file("kernels/linear_slice_bias_f32.cl");
        if(!p){NNOPT_ERROR("depth: build linear_slice_bias_f32");pool_free(ln);return nullptr;}
        cl_int e; slk=clCreateKernel(p,"linear_slice_bias_f32",&e); }
    cl_mem lw=weights.get_buffer(D+"layers.3.inner._linear.weight");
    cl_mem lb=weights.get_buffer(D+"layers.3.inner._linear.bias");
    if(!lw||!lb){ NNOPT_ERROR("depth: missing logits weight/bias"); pool_free(ln); return nullptr; }
    const int lin=weights.get_shape(D+"layers.3.inner._linear.weight")[1]; // 768
    cl_int e4; cl_mem logits=pool_alloc(cl_ctx.context(),CL_MEM_READ_WRITE,(size_t)count*sizeof(float),nullptr,&e4);
    // Own kernel object per dispatch. `lo`/`count` are host ints, which a capture pins — that is
    // safe here ONLY because this site is reached once per RVQ level per frame with the same value
    // at the same point in the sequence, and the per-frame ordinal in the instance key keeps the 12
    // levels from sharing one object.
    cl_kernel slki = nnopt_kernel_instance(slk, D, "logit_slice");
    clSetKernelArg(slki,0,sizeof(cl_mem),&ln); clSetKernelArg(slki,1,sizeof(cl_mem),&lw); clSetKernelArg(slki,2,sizeof(cl_mem),&lb);
    clSetKernelArg(slki,3,sizeof(cl_mem),&logits); clSetKernelArg(slki,4,sizeof(int),&lin);
    clSetKernelArg(slki,5,sizeof(int),&lo); clSetKernelArg(slki,6,sizeof(int),&count);
    const size_t nwg=((size_t)count+7)/8; size_t g[2]={1,nwg*64}, lws[2]={1,64};
    cl_ctx.profEnqueue(slki,2,g,lws,"gemv_logit_slice");
    pool_free(ln);
    return logits;  // [count] — the slice, indices 0..count-1 map to global rows lo..lo+count-1
}

// ── embed a single token id → [embed_dim] via the decoder embedder ──
static cl_mem embed_token(OpenCLContext& cl_ctx, Weights& weights, cl_command_queue queue, int token) {
    cl_int e; int32_t t = token;
    cl_mem tb = pool_alloc(cl_ctx.context(), CL_MEM_READ_ONLY|CL_MEM_COPY_HOST_PTR, sizeof(int32_t), &t, &e);
    cl_mem r = QuantizedEmbedding_forward(cl_ctx,weights,queue,tb,1,0,0,nullptr,nullptr,nullptr,
                                          "depthformer.decoder.embedder.layers.0._embedding");
    pool_free(tb);
    return r;
}

// ── the current render's sampling parameters ────────────────────────────────────────────────────
// Set once by run_song before generation rather than threaded through run_generate,
// run_song_pipelined and the op-test harness, all of which would have to carry three values they do
// not otherwise care about. The FRAME index is a real argument, not a global, because it varies per
// call and is what keeps consecutive frames from drawing identical noise.
namespace {
float    g_sample_temp = 0.0f;   // 0 = greedy, which is what the op-tests assume
int      g_sample_topk = 0;
unsigned g_sample_seed = 0;
}
void nnopt_set_sampling(float temperature, int top_k, unsigned seed) {
    g_sample_temp = temperature; g_sample_topk = top_k; g_sample_seed = seed;
}

// ── depth AR loop: sample one token per codebook → 12 RVQ tokens ──
// num_reserved=6, codebook_size=1024, 12 codebooks.
//
// This was greedy argmax, with a note that the model's soft cap is monotonic and therefore cannot
// change an argmax. That was true, and it stopped being true the moment temperature arrived: the cap
// changes the SPACING between logits and so changes every probability. The cap is applied inside
// sample_range_f32 with soft_cap_logits = 30.0 (magenta_rt/jax/model.py).
// fixed_inputs (optional): host [NUM_CB*1024] reference depth inputs — if set, used instead of
// embedder feedback (isolates depth-step+cache correctness from the feedback/CFG loop).
// Returns the GPU tokbuf [12] int — NO host readback, so the AR stays a deep async queue (caller
// frees). The host-vector wrapper run_depth_loop() below reads it for op-tests.
static cl_mem run_depth_loop_gpu(OpenCLContext& cl_ctx, Weights& weights, cl_command_queue queue,
                                       cl_mem temporal_out, const std::vector<float>* fixed_inputs = nullptr,
                                       int frame_idx = 0) {
    const int NUM_CB = 12, RESERVED = 6, CBSIZE = 1024;
    const int HD = weights.get_shape(
        "depthformer.decoder.depth_body.layers.1.layers.0.layers.0.body.layers.1.inner.qkv_proj.weight")[0] / 3; // 768
    cl_mem kc[2], vc[2];
    for (int L=0; L<2; ++L) {
        kc[L]=pool_alloc(cl_ctx.context(),CL_MEM_READ_WRITE,(size_t)NUM_CB*HD*sizeof(float),nullptr,nullptr);
        vc[L]=pool_alloc(cl_ctx.context(),CL_MEM_READ_WRITE,(size_t)NUM_CB*HD*sizeof(float),nullptr,nullptr);
    }
    cl_int e2;
    auto fixed_buf = [&](int q)->cl_mem{
        return pool_alloc(cl_ctx.context(),CL_MEM_READ_ONLY|CL_MEM_COPY_HOST_PTR,(size_t)1024*sizeof(float),
                              const_cast<float*>(fixed_inputs->data()+(size_t)q*1024),&e2);
    };
    // GPU argmax + embedder chain — no per-codebook host sync (read tokbuf once at the end).
    cl_int e3=CL_SUCCESS;
    static cl_kernel amk=nullptr, eok=nullptr;
    if(!amk){ cl_program p=cl_ctx.build_program_from_file("kernels/argmax_range_f32.cl"); amk=clCreateKernel(p,"argmax_range_f32",&e3); }
    static cl_kernel smk=nullptr;
    if(!smk){ cl_program p=cl_ctx.build_program_from_file("kernels/sample_range_f32.cl"); if(p) smk=clCreateKernel(p,"sample_range_f32",&e3); }
    if(!eok){ cl_program p=cl_ctx.build_program_from_file("kernels/embed_one_f16.cl"); eok=clCreateKernel(p,"embed_one_f16",&e3); }
    cl_mem etab=weights.get_buffer("depthformer.decoder.embedder.layers.0._embedding.weight");
    const std::vector<int> esh=weights.get_shape("depthformer.decoder.embedder.layers.0._embedding.weight");
    const int evoc=esh[0], edim=esh[1];
    cl_mem tokbuf=pool_alloc(cl_ctx.context(),CL_MEM_READ_WRITE,(size_t)NUM_CB*sizeof(int),nullptr,&e3);
    cl_mem depth_input;
    if (fixed_inputs) depth_input = fixed_buf(0);
    else { depth_input = pool_alloc(cl_ctx.context(),CL_MEM_READ_WRITE,(size_t)1024*sizeof(float),nullptr,nullptr);
           static const std::string kSite("depth.input");
           nnopt_copy_buffer(cl_ctx, temporal_out, depth_input, 0, 0, 1024, kSite, nullptr); }
    for (int q=0; q<NUM_CB; ++q) {
        const int lo=RESERVED+q*CBSIZE;
        cl_mem logits = run_depth_step(cl_ctx,weights,queue,depth_input,kc,vc,q,lo,CBSIZE);
        if (!logits) { NNOPT_ERROR_FMT("depth loop: step %d failed", q); break; }
        // logits is the [0,CBSIZE) slice now → argmax over the whole slice, token = lo + argmax.
        int base=lo, cnt=CBSIZE;
        const size_t ag=64, al=64;   // one 64-wide workgroup, not one work item
        if (g_sample_temp > 0.0f && smk) {
            static const std::string kSample("depth.sample");
            cl_kernel smki = nnopt_kernel_instance(smk, kSample);
            const float cap = 30.0f;              // soft_cap_logits, magenta_rt/jax/model.py
            const unsigned sd = g_sample_seed, fr = (unsigned)frame_idx;
            clSetKernelArg(smki,0,sizeof(cl_mem),&logits); clSetKernelArg(smki,1,sizeof(cl_mem),&tokbuf);
            clSetKernelArg(smki,2,sizeof(int),&base);      clSetKernelArg(smki,3,sizeof(int),&cnt);
            clSetKernelArg(smki,4,sizeof(int),&q);         clSetKernelArg(smki,5,sizeof(float),&g_sample_temp);
            clSetKernelArg(smki,6,sizeof(int),&g_sample_topk); clSetKernelArg(smki,7,sizeof(float),&cap);
            clSetKernelArg(smki,8,sizeof(unsigned),&sd);   clSetKernelArg(smki,9,sizeof(unsigned),&fr);
            cl_ctx.profEnqueue(smki,1,&ag,&al,"sample");
        } else {
            static const std::string kArgmax("depth.argmax");
            cl_kernel amki = nnopt_kernel_instance(amk, kArgmax);
            clSetKernelArg(amki,0,sizeof(cl_mem),&logits); clSetKernelArg(amki,1,sizeof(cl_mem),&tokbuf);
            clSetKernelArg(amki,2,sizeof(int),&base); clSetKernelArg(amki,3,sizeof(int),&cnt); clSetKernelArg(amki,4,sizeof(int),&q);
            cl_ctx.profEnqueue(amki,1,&ag,&al,"argmax");
        }
        pool_free(logits);
        if (q < NUM_CB-1) {
            pool_free(depth_input);
            if (fixed_inputs) depth_input = fixed_buf(q+1);
            else { // embed tokbuf[q] (stays on GPU)
                depth_input=pool_alloc(cl_ctx.context(),CL_MEM_READ_WRITE,(size_t)edim*sizeof(float),nullptr,&e3);
                static const std::string kEmbedOne("depth.embed_one");
                cl_kernel eoki = nnopt_kernel_instance(eok, kEmbedOne);
                clSetKernelArg(eoki,0,sizeof(cl_mem),&etab); clSetKernelArg(eoki,1,sizeof(cl_mem),&tokbuf); clSetKernelArg(eoki,2,sizeof(int),&q);
                clSetKernelArg(eoki,3,sizeof(cl_mem),&depth_input); clSetKernelArg(eoki,4,sizeof(int),&edim); clSetKernelArg(eoki,5,sizeof(int),&evoc);
                size_t ed=(size_t)edim; cl_ctx.profEnqueue(eoki,1,&ed,nullptr,"embed1");
            }
        }
    }
    pool_free(depth_input);
    for (int L=0; L<2; ++L) { pool_free(kc[L]); pool_free(vc[L]); }
    return tokbuf;  // GPU [12] int — caller frees; no readback here
}

// Host-vector wrapper (op-tests / debugging): runs the GPU depth loop and reads the 12 tokens back.
static std::vector<int> run_depth_loop(OpenCLContext& cl_ctx, Weights& weights, cl_command_queue queue,
                                       cl_mem temporal_out, const std::vector<float>* fixed_inputs = nullptr) {
    cl_mem tokbuf = run_depth_loop_gpu(cl_ctx, weights, queue, temporal_out, fixed_inputs);
    if (!tokbuf) return {};
    std::vector<int32_t> tk(12);
    clEnqueueReadBuffer(queue,tokbuf,CL_TRUE,0,tk.size()*sizeof(int32_t),tk.data(),0,nullptr,nullptr);
    pool_free(tokbuf);
    return std::vector<int>(tk.begin(), tk.end());
}

// ── frame embedder: embed 12 RVQ tokens → [12,1024] → mean over codebooks → temporal_input [1024] ──
static cl_mem frame_embed_mean(OpenCLContext& cl_ctx, Weights& weights, cl_command_queue queue,
                               const std::vector<int>& toks) {
    std::vector<int32_t> ids(toks.begin(), toks.end());
    cl_int e; cl_mem tb = pool_alloc(cl_ctx.context(), CL_MEM_READ_ONLY|CL_MEM_COPY_HOST_PTR,
                                         ids.size()*sizeof(int32_t), ids.data(), &e);
    cl_mem g = QuantizedEmbedding_forward(cl_ctx,weights,queue,tb,(int)ids.size(),0,0,nullptr,nullptr,nullptr,
                                          "depthformer.decoder.embedder.layers.0._embedding");
    pool_free(tb);
    if (!g) return nullptr;
    const int Q = (int)ids.size(), dim = 1024;
    std::vector<float> host((size_t)Q*dim);
    clEnqueueReadBuffer(queue,g,CL_TRUE,0,host.size()*sizeof(float),host.data(),0,nullptr,nullptr);
    pool_free(g);
    std::vector<float> mean(dim,0.0f);
    for (int q=0;q<Q;q++) for (int d=0;d<dim;d++) mean[d]+=host[(size_t)q*dim+d];
    for (int d=0;d<dim;d++) mean[d]/=(float)Q;
    cl_mem out = pool_alloc(cl_ctx.context(), CL_MEM_READ_ONLY|CL_MEM_COPY_HOST_PTR, mean.size()*sizeof(float), mean.data(), &e);
    return out;
}

// OPT #5: on-GPU variant — mean-reduce [Q,dim] embeddings over codebooks without a host
// round-trip. Identical embedding values (same QuantizedEmbedding); only the reduction moves
// to the GPU, keeping the AR feedback on-device. NNOPT_NOGPUEMB=1 selects the host path above.
static cl_mem frame_embed_mean_gpu(OpenCLContext& cl_ctx, Weights& weights, cl_command_queue queue,
                                   const std::vector<int>& toks) {
    static int nogpuemb = -1;
    if (nogpuemb < 0) { const char* ev = std::getenv("NNOPT_NOGPUEMB"); nogpuemb = (ev && ev[0] != '0') ? 1 : 0; }
    if (nogpuemb) return frame_embed_mean(cl_ctx, weights, queue, toks);
    std::vector<int32_t> ids(toks.begin(), toks.end());
    cl_int e; cl_mem tb = pool_alloc(cl_ctx.context(), CL_MEM_READ_ONLY|CL_MEM_COPY_HOST_PTR,
                                     ids.size()*sizeof(int32_t), ids.data(), &e);
    cl_mem g = QuantizedEmbedding_forward(cl_ctx,weights,queue,tb,(int)ids.size(),0,0,nullptr,nullptr,nullptr,
                                          "depthformer.decoder.embedder.layers.0._embedding");
    pool_free(tb);
    if (!g) return nullptr;
    const int Q = (int)ids.size(), dim = 1024;
    static cl_kernel mk = nullptr;
    if (!mk) { cl_program p = cl_ctx.build_program_from_file("kernels/mean_rows_f32.cl");
               if (p) mk = clCreateKernel(p, "mean_rows_f32", &e); }
    if (!mk) { pool_free(g); return frame_embed_mean(cl_ctx, weights, queue, toks); }
    cl_mem out = pool_alloc(cl_ctx.context(), CL_MEM_READ_WRITE, (size_t)dim*sizeof(float), nullptr, &e);
    const float scale = 1.0f/(float)Q;
    clSetKernelArg(mk,0,sizeof(cl_mem),&g); clSetKernelArg(mk,1,sizeof(cl_mem),&out);
    clSetKernelArg(mk,2,sizeof(int),&Q); clSetKernelArg(mk,3,sizeof(int),&dim); clSetKernelArg(mk,4,sizeof(float),&scale);
    size_t gd=(size_t)dim; cl_ctx.profEnqueue(mk,1,&gd,nullptr,"frame_mean");
    pool_free(g);
    return out;
}

// OVERLAP/no-sync AR: frame-embed directly from the GPU tokbuf (12 global ids) — gather the 12 fp16
// embedding rows and mean — with NO host round-trip. Same math as frame_embed_mean (sum/12 of the same
// fp16 rows). Lets run_generate keep the whole AR on a deep async queue (one readback at the very end).
// dst != nullptr writes in place instead of allocating. The AR loop used to swap `ti` for a fresh
// buffer every frame; a recorded dispatch pins the handle it saw at capture, so the frame input has
// to live at ONE address for the life of the render.
static cl_mem frame_embed_from_tokbuf(OpenCLContext& cl_ctx, Weights& weights, cl_command_queue queue, cl_mem tokbuf, cl_mem dst = nullptr) {
    cl_int e=CL_SUCCESS;
    cl_mem etab=weights.get_buffer("depthformer.decoder.embedder.layers.0._embedding.weight");
    const std::vector<int> esh=weights.get_shape("depthformer.decoder.embedder.layers.0._embedding.weight");
    const int dim=esh[1], N=12;
    static cl_kernel gk=nullptr;
    if(!gk){ cl_program p=cl_ctx.build_program_from_file("kernels/gather_reduce_f16.cl"); if(!p) return nullptr; gk=clCreateKernel(p,"gather_reduce_f16",&e); }
    static cl_mem zeros=nullptr;  // offsets[N]=0 (tokbuf already holds global embedding ids)
    if(!zeros){ std::vector<int> z((size_t)N,0); zeros=clCreateBuffer(cl_ctx.context(),CL_MEM_READ_ONLY|CL_MEM_COPY_HOST_PTR,N*sizeof(int),z.data(),&e); }
    cl_mem out = dst ? dst : pool_alloc(cl_ctx.context(),CL_MEM_READ_WRITE,(size_t)dim*sizeof(float),nullptr,&e);
    // scale = sqrt(dim)/N : the embedder applies a sqrt(embedding_dim) scale (embedding_gather_f16,
    // sl.Embedding _scale) AND we mean over N=12 codebooks. Omitting sqrt(dim) silently breaks the
    // scale-sensitive depth adapter (cosine-invariant, so gates miss it).
    const int tok_start=0; const float scale=sqrtf((float)dim)/(float)N;
    // The last dispatch in the frame that still shared a kernel object — one per frame, which is
    // exactly what the dispatch/lookup reconciliation reported (519 vs 518).
    static const std::string kFrameEmbed("ar.frame_embed");
    cl_kernel gki = nnopt_kernel_instance(gk, kFrameEmbed);
    clSetKernelArg(gki,0,sizeof(cl_mem),&etab); clSetKernelArg(gki,1,sizeof(cl_mem),&tokbuf); clSetKernelArg(gki,2,sizeof(int),&tok_start);
    clSetKernelArg(gki,3,sizeof(cl_mem),&zeros); clSetKernelArg(gki,4,sizeof(cl_mem),&out); clSetKernelArg(gki,5,sizeof(int),&N);
    clSetKernelArg(gki,6,sizeof(int),&dim); clSetKernelArg(gki,7,sizeof(float),&scale);
    size_t gd=(size_t)dim; cl_ctx.profEnqueue(gki,1,&gd,nullptr,"frame_embed");
    return out;
}

// The cross-queue gate used by the AR∥codec pipeline. These are OpenCL 1.2 core, but an app can
// ship a libOpenCL that predates them, and because LD_LIBRARY_PATH puts the app's own lib ahead of
// /vendor/lib64, that stub shadows the real driver. Android resolves an executable's symbols
// eagerly, so an unresolved import kills the process at load — over a code path that is off by
// default. Declaring them weak makes them null-if-absent, and the pipeline falls back to a full
// queue drain, which is correct (just without overlap).
extern "C" __attribute__((weak)) cl_int clEnqueueMarkerWithWaitList(
    cl_command_queue, cl_uint, const cl_event*, cl_event*);
extern "C" __attribute__((weak)) cl_int clEnqueueBarrierWithWaitList(
    cl_command_queue, cl_uint, const cl_event*, cl_event*);

// Smallest time-chunk the SpectroStream cascade can decode. Its transpose-convs consume a few
// frames of context per stage, and every measurement this port has ever taken used 5-frame chunks
// — nothing smaller has ever been exercised, and T=1 segfaults inside the cascade.
static const int kMinCodecFrames = 5;
// Ceilings for the auto-chunk picker. A too-large codec buffer does not fail cleanly on Adreno —
// it resets the GPU and reboots the phone, so these are safety limits, not tuning knobs. Raise the
// chunk deliberately with `chunk=N` on a device you are watching, never automatically.
// (the old flat 384 MB working-set cap is gone — it bound on every device; see the auto-chunk block)
static const int    kCodecLiveBuffers   = 6;    // buffers of ~peak size the cascade holds at once
static const int    kCodecChunkCeiling  = 25;   // frames; beyond this we have no evidence it is safe
// The chunk that is actually FASTEST, on both GPUs measured. Auto-chunk never exceeds it; the
// memory budgets exist to shrink below it on a small device, not to grow past it on a big one.
static const int    kAutoChunkPreferred = 5;

// Peak single-buffer demand seen by the codec convs, in bytes. Reset per codec pass; read by the
// auto-chunk picker in run_song(). See conv_clblast for why one small chunk predicts a big one.
static size_t g_codec_peak_alloc = 0;

// ── fp32 codec weight store (the conv decoder needs fp32: its 10M dynamic range + phase-sensitive
// ISTFT can't tolerate fp16). Loads weights/model.codec.f32.{idx,bin} once; caches cl_mem per tensor. ──
#include <map>
#include <set>
#include <tuple>
#include <fstream>
#include <sstream>
struct CodecF32Store {
    std::vector<float> blob;
    std::map<std::string, std::pair<size_t,size_t>> idx; // name -> (offset_floats, num)
    std::map<std::string, std::vector<int>> shape;       // name -> dims, from the codec meta.json
    std::map<std::string, cl_mem> cache;
    bool loaded=false;
};
static CodecF32Store g_cf32;
static cl_mem get_codec_f32(OpenCLContext& cl_ctx, const std::string& name) {
    if (!g_cf32.loaded) {
        std::ifstream idxf("weights/model.codec.f32.idx");
        std::string nm; size_t off,num;
        while (idxf >> nm >> off >> num) g_cf32.idx[nm]={off,num};
        FILE* bf=std::fopen("weights/model.codec.f32.bin","rb");
        if (bf){ std::fseek(bf,0,SEEK_END); long sz=std::ftell(bf); std::fseek(bf,0,SEEK_SET);
            g_cf32.blob.resize(sz/sizeof(float)); size_t rd=std::fread(g_cf32.blob.data(),1,sz,bf); std::fclose(bf); (void)rd; }
        // Codec tensor SHAPES come from the codec's own meta.json. They used to be read from the
        // main fp16 blob's metadata, which happened to carry a full unused copy of the codec — so
        // stripping that copy turned every codec conv into an out-of-bounds read on an empty shape
        // vector (a silent SIGSEGV with no error and no allocation). The codec owns its metadata.
        { std::ifstream mf("weights/model.codec.f32.meta.json");
          std::string j((std::istreambuf_iterator<char>(mf)), std::istreambuf_iterator<char>());
          size_t pos = 0;
          while (true) {
              const size_t k0 = j.find("\"spectrostream", pos);
              if (k0 == std::string::npos) break;
              const size_t k1 = j.find('"', k0 + 1);
              if (k1 == std::string::npos) break;
              const std::string nm2 = j.substr(k0 + 1, k1 - k0 - 1);
              const size_t sh = j.find("\"shape\"", k1);
              const size_t lb = (sh == std::string::npos) ? std::string::npos : j.find('[', sh);
              const size_t rb = (lb == std::string::npos) ? std::string::npos : j.find(']', lb);
              if (rb == std::string::npos) break;
              std::vector<int> dims;
              std::istringstream ds(j.substr(lb + 1, rb - lb - 1));
              int d; char comma;
              while (ds >> d) { dims.push_back(d); ds >> comma; }
              g_cf32.shape[nm2] = dims;
              pos = rb;
          }
        }
        g_cf32.loaded=true;
    }
    auto c=g_cf32.cache.find(name); if (c!=g_cf32.cache.end()) return c->second;
    auto it=g_cf32.idx.find(name); if (it==g_cf32.idx.end()) { NNOPT_ERROR_FMT("codec f32: missing %s", name.c_str()); return nullptr; }
    cl_int e; cl_mem b=pool_alloc(cl_ctx.context(), CL_MEM_READ_ONLY|CL_MEM_COPY_HOST_PTR,
        it->second.second*sizeof(float), g_cf32.blob.data()+it->second.first, &e);
    g_cf32.cache[name]=b; return b;
}

// Shape of a codec tensor, from the codec's own metadata. Empty when absent — callers MUST check
// rather than index, which is the mistake that made a stripped bundle segfault instead of complain.
static const std::vector<int>& get_codec_shape(OpenCLContext& cl_ctx, const std::string& name) {
    static const std::vector<int> kEmpty;
    if (!g_cf32.loaded) get_codec_f32(cl_ctx, name);   // lazily loads idx + blob + shapes
    auto it = g_cf32.shape.find(name);
    return it == g_cf32.shape.end() ? kEmpty : it->second;
}

// Does the CODEC own this tensor? The cascade uses this to decide whether a residual block has a
// shortcut convolution. It used to ask the fp16 blob's tensor map, which only answered "yes"
// because that blob happened to carry a second, never-read fp16 copy of the whole codec. Dropping
// that dead copy silently turned every shortcut off — no error, because "no shortcut" is a legal
// answer — and the cascade produced full-scale noise. Ask the codec's own index.
static bool has_codec_tensor(OpenCLContext& cl_ctx, const std::string& name) {
    if (!g_cf32.loaded) get_codec_f32(cl_ctx, name);
    return g_cf32.idx.find(name) != g_cf32.idx.end();
}

// Forward decl: ELU op (defined below). Needed by conv2d_op for the ident-1×1 fallback,
// where there is no im2col gather to fuse the activation into (OPT #7).
static cl_mem elu_op(OpenCLContext& cl_ctx, cl_command_queue queue, cl_mem x, int n);

// OPT #7 toggle: fuse the pre-conv ELU into the im2col gather. NNOPT_NOFUSEELU=1 forces the
// old standalone-elu path for A/B measurement and as a correctness escape hatch.
static bool fuse_elu_enabled() {
    static int v = -1;
    if (v < 0) { const char* e = std::getenv("NNOPT_NOFUSEELU"); v = (e && e[0] != '0') ? 0 : 1; }
    return v != 0;
}

// dot8 feasibility probe: simulate int8 (qcom_dot8_acc) numerical precision in the fp32 codec convs
// (per-channel weights, per-row activations) to test whether the phase-sensitive codec survives int8
// BEFORE building the real int8/dot8 pipeline. NNOPT_FAKEQ8=1. Pure measurement path; never a default.
static bool fakeq8_enabled() {
    // Epoch-keyed, not latch-once: latching caches the value during the WARM-UP render,
    // before a serve request can set it, so the toggle silently never applies and the run
    // reports the default under the experiment's name.
    static int v = -1, v_epoch = -1;
    if (v_epoch != nnopt_toggle_epoch()) {
        const char* e = std::getenv("NNOPT_FAKEQ8");
        v = (e && e[0] != '0') ? 1 : 0;
        v_epoch = nnopt_toggle_epoch();
    }
    return v != 0;
}

#ifdef USE_CLBLAST
// ── Conv via im2col + CLBlast SGEMM. out[M,N] = col[M,K] @ W[N,K].T + bias. im2col_kern = conv2d or
// conv2dT variant (same arg layout). CLBlast tiles both operands in LDS → kills the weight-bandwidth wall.
// apply_elu (OPT #7): fuse the activation into the im2col gather (no-op on the ident-1×1 path).
// The two im2col gathers, published at file scope. conv_clblast is handed one of them and needs to
// know WHICH — the plain and transposed variants have completely different index math. Comparing the
// handle costs nothing and, unlike clGetKernelInfo, introduces no OpenCL symbol: the S26's loader
// could not resolve clGetSupportedImageFormats/clCreateImage/clGetKernelInfo and refused to start the
// engine at all. Only symbols already proven on device may appear in this file.
static cl_kernel g_im2col_plain = nullptr;
static cl_kernel g_im2col_transposed = nullptr;

static cl_mem conv_clblast(OpenCLContext& cl_ctx, cl_command_queue queue, cl_mem in, cl_mem Wbuf, cl_mem biasbuf,
                           int B, int T,int F,int Cin,int Tout,int Fout,int Cout,int kH,int kW,int sh,int sw,
                           int p1,int p2, cl_kernel im2col_kern, bool apply_elu=false) {
    const int M=B*Tout*Fout, N=Cout, K=kH*kW*Cin;
    // Largest single allocation any conv in this codec pass wanted. The codec's buffers scale
    // linearly with the time-chunk, so one small chunk measures what a big one would cost — that
    // is how the auto-chunk picker sizes itself to an unknown device instead of guessing.
    { const size_t want = (size_t)M * (size_t)(K > N ? K : N) * sizeof(float);
      if (want > g_codec_peak_alloc) g_codec_peak_alloc = want; }
    cl_int err=CL_SUCCESS;
    static cl_kernel bk=nullptr;
    if(!bk){ cl_program p=cl_ctx.build_program_from_file("kernels/broadcast_bias_f32.cl"); bk=clCreateKernel(p,"broadcast_bias_f32",&err); }
    // OPT #1: 1×1 / stride-1 / no-pad conv is an im2col IDENTITY — the materialized col
    // buffer is byte-for-byte the input ([B*T*F, Cin] == [M, K]). Skip the col alloc + the
    // im2col kernel entirely and feed `in` straight to SGEMM. im2col was ~30% of codec GPU.
    // NNOPT_NO1X1=1 forces the old (materialize) path for A/B measurement.
    static int no1x1 = -1;
    if (no1x1 < 0) { const char* e = std::getenv("NNOPT_NO1X1"); no1x1 = (e && e[0] != '0') ? 1 : 0; }
    const bool ident_1x1 = (kH==1 && kW==1 && sh==1 && sw==1 && p1==0 && p2==0 && !no1x1);
    // Pre-flight the two big allocations. At CHUNK≫5 the im2col col buffer ([M,K] fp32) is the
    // first thing to blow past the driver's single-allocation ceiling, and the historical failure
    // mode was a hard crash inside CLBlast's own allocator (`clReleaseMemObject:-38`) that killed
    // the whole run. Refuse with the numbers instead: the caller reports one failed chunk and the
    // sweep moves on to the next config.
    {
        static cl_ulong max_alloc = 0;
        if (!max_alloc)
            clGetDeviceInfo(cl_ctx.device(), CL_DEVICE_MAX_MEM_ALLOC_SIZE, sizeof(max_alloc), &max_alloc, nullptr);
        const size_t col_bytes = (size_t)M*(size_t)K*sizeof(float);
        const size_t out_bytes = (size_t)M*(size_t)N*sizeof(float);
        const size_t need = ident_1x1 ? out_bytes : (col_bytes > out_bytes ? col_bytes : out_bytes);
        if (max_alloc && need > (size_t)max_alloc) {
            NNOPT_ERROR_FMT("conv_clblast: needs a %zu MB buffer, device max alloc is %llu MB "
                            "(M=%d N=%d K=%d) — lower the codec chunk",
                            need >> 20, (unsigned long long)(max_alloc >> 20), M, N, K);
            return nullptr;
        }
    }
    // ── fp16 GEMM path (NNOPT_CODECFP16=1) ─────────────────────────────────────────────────────
    // The cascade is GEMM-bound at ~154 GFLOP/s against ~3.7 TFLOPS of fp32, so fp16 buys 2x ALU
    // and halves every operand. The blocker was never precision, it was RANGE: late blocks reach
    // |x| ~ 9.5e6 and fp16 stops at 65504. Convolution is linear, so we divide the input by a
    // POWER-OF-TWO scale (exact in binary floating point, no rounding introduced), run the whole
    // GEMM in fp16, and multiply back.
    //
    // The scale is measured per call rather than baked in, because the per-block magnitude varies
    // more than 10x between prompts (block4 measured 526k on one prompt and 9.5M on another) and a
    // static scale that fits one would overflow the other.
    //
    // NOTE CLBlast's HGEMM accumulates in fp16. With K up to 4608 that is the real risk here, not
    // the range — which is why this is a toggle gated on a waveform comparison, not the default.
    // DEFAULT ON for Adreno 8xx as of 2026-08-26. Measured on an 840: codec 1.474 s -> 0.962-1.078,
    // total 2.015 -> 1.504-1.638, RTF 1.008 -> 0.752-0.819. That is the difference between missing
    // real time and clearing it, and it comes from running the GEMM in half precision plus keeping
    // the scale selection off the host — not from any tile parameter (see BENCHMARK.md; twenty-one
    // candidates over two rounds, CLBlast's own choice won both).
    //
    // Gated on the device rather than switched on everywhere, because the fp16 path's numerics are
    // verified on the 840 alone: cosine 0.99886-0.99959 against the fp32 render. A 6xx has neither
    // been measured nor listened to, and it is the device this port validates correctness on, so it
    // keeps fp32. NNOPT_CODECFP16=0 forces fp32 anywhere; =1 forces fp16 anywhere.
    static int fp16_on = -1, fp16_epoch = -1;
    if (fp16_epoch != nnopt_toggle_epoch()) {
        const char* e2 = std::getenv("NNOPT_CODECFP16");
        // DEFAULT OFF as of 2026-08-28. The reference decodes the codec at full precision — the
        // `.mlxfn` path, `--bits=8` opt-in and not used — and an oracle run on Apple Silicon ranks
        // "fp16 GEMM with fp16 ACCUMULATION" as the #2 fidelity gap in this port, behind int8. The
        // accumulation is the risky half: CLBlast's HGEMM sums K terms in half precision with K up
        // to 4608, and the codec decoder sums many terms.
        //
        // We can afford it now. Before pipelining, fp32 codec meant 2.015 s per 2 s of audio (0.99x,
        // under real time). With AR and codec overlapped the wall is max(AR, codec) rather than the
        // sum, so a 1.474 s fp32 codec still lands near 1.36x — ahead of the 1.16-1.25x the
        // reference itself measures on a base M1. Speed was the only reason for fp16 here.
        // Back ON for Adreno 8xx, 2026-08-28. Reference precision (fp32 codec + fp16 AR) was tried
        // as a default and measured 0.85x on device — under real time, buffer draining — while the
        // artifact the change was aimed at was still audible. It bought nothing that could be heard
        // and cost the headroom that makes live mode work, so the shipping config goes back to what
        // is fast. Both axes stay reachable from the Precision and Codec chips for A/B.
        if (e2 && *e2) fp16_on = (e2[0] != '0') ? 1 : 0;
        else           fp16_on = (nnopt_adreno_model(cl_ctx) >= 800) ? 1 : 0;
        fp16_epoch = nnopt_toggle_epoch();
    }
    // codecab=1 decodes the same chunk down both paths to compare them, so for the duration of one
    // pass the choice comes from the caller rather than the environment. -1 = whatever the env said.
    const int forced = nnopt_codec_fp16_forced();
    const bool use_fp16 = (forced >= 0) ? (forced != 0) : (fp16_on != 0);

    if (use_fp16) {
        bool skipped = false;
        // The caller chose a gather; the fp16 path must honour the SAME choice.
        const bool is_transposed = (im2col_kern == g_im2col_transposed);
        cl_mem r = conv_clblast_fp16(cl_ctx, queue, in, Wbuf, biasbuf, B, T, F, Cin,
                                     Tout, Fout, Cout, kH, kW, sh, sw, p1, p2, apply_elu,
                                     M, N, K, ident_1x1, is_transposed, &skipped);
        // Count what actually ran in fp16 versus what fell back, and say so once per render. An
        // "fp16 run" that quietly executed fp32 is the failure mode this whole campaign keeps
        // hitting, so the split is reported rather than assumed.
        nnopt_codec_fp16_tally(r ? 1 : 0, (!r && skipped) ? 1 : 0, (!r && !skipped) ? 1 : 0);
        { int nf = 0, ns = 0, nx = 0;
          nnopt_codec_fp16_get(&nf, &ns, &nx);
          if (((nf + ns + nx) % 24) == 0)
              std::fprintf(stderr, "CODECFP16 fp16=%d skipped_1x1=%d failed=%d\n", nf, ns, nx); }
        if (r) return r;
        if (!skipped) {
            static bool warned = false;
            if (!warned) { warned = true; NNOPT_ERROR("codec fp16 path FAILED — falling back to fp32"); }
        }
    }

    cl_mem col = nullptr;
    if (!ident_1x1) {
        col=pool_alloc(cl_ctx.context(),CL_MEM_READ_WRITE,(size_t)M*K*sizeof(float),nullptr,&err);
        if(!col){ NNOPT_ERROR("conv_clblast: col alloc"); return nullptr; }
        int ai=0;
        clSetKernelArg(im2col_kern,ai++,sizeof(cl_mem),&in); clSetKernelArg(im2col_kern,ai++,sizeof(cl_mem),&col);
        clSetKernelArg(im2col_kern,ai++,sizeof(int),&B);
        clSetKernelArg(im2col_kern,ai++,sizeof(int),&T); clSetKernelArg(im2col_kern,ai++,sizeof(int),&F); clSetKernelArg(im2col_kern,ai++,sizeof(int),&Cin);
        clSetKernelArg(im2col_kern,ai++,sizeof(int),&Tout); clSetKernelArg(im2col_kern,ai++,sizeof(int),&Fout);
        clSetKernelArg(im2col_kern,ai++,sizeof(int),&kH); clSetKernelArg(im2col_kern,ai++,sizeof(int),&kW);
        clSetKernelArg(im2col_kern,ai++,sizeof(int),&sh); clSetKernelArg(im2col_kern,ai++,sizeof(int),&sw);
        clSetKernelArg(im2col_kern,ai++,sizeof(int),&p1); clSetKernelArg(im2col_kern,ai++,sizeof(int),&p2);
        const int elu_flag = apply_elu ? 1 : 0;
        clSetKernelArg(im2col_kern,ai++,sizeof(int),&elu_flag);
        size_t ig[2]={(size_t)K,(size_t)M};
        if (OpenCLContext::profWantEvent()) { clFinish(queue); cl_ctx.compTic(); }
        // NNOPT_IM4: same gather, four k per work item -> 128-bit loads and stores instead of 32-bit.
        // Requires Cin % 4 == 0 and K % 4 == 0, which every codec conv satisfies; anything else keeps
        // the scalar kernel.
        static int i4_on = -1, i4_epoch = -1;
        if (i4_epoch != nnopt_toggle_epoch()) {
            // ON by default: 3.46x on the gather (1771 -> 512 ms/85 calls) with byte-identical
            // audio, so there is no reason to make the user opt in. NNOPT_IM4=0 restores the scalar
            // kernel for A/B.
            const char* e4 = std::getenv("NNOPT_IM4");
            i4_on = (e4 && e4[0] == '0') ? 0 : 1;
            i4_epoch = nnopt_toggle_epoch();
        }
        static cl_kernel imk_v4 = nullptr, imk_v4T = nullptr;
        cl_kernel v4k = nullptr;
        if (i4_on && (Cin % 4) == 0 && (K % 4) == 0) {
            if (!imk_v4) {
                cl_program vp = cl_ctx.build_program_from_file("kernels/im2col_conv2d_v4.cl");
                if (vp) { imk_v4  = clCreateKernel(vp, "im2col_conv2d_f32_v4",  &err);
                          imk_v4T = clCreateKernel(vp, "im2col_conv2dT_f32_v4", &err); }
            }
            // Match the vectorized gather to the scalar one actually passed. Anything that is
            // neither (the f16 gather, which writes half) keeps its own kernel rather than being
            // silently mis-gathered — substituting the plain math for a transposed conv produces
            // wrong audio AND a fake speedup, because the bad reads fall out of range and skip work.
            if      (im2col_kern == g_im2col_transposed) v4k = imk_v4T;
            else if (im2col_kern == g_im2col_plain)      v4k = imk_v4;
        }
        if (v4k) {
            int a4 = 0;
            clSetKernelArg(v4k,a4++,sizeof(cl_mem),&in);  clSetKernelArg(v4k,a4++,sizeof(cl_mem),&col);
            clSetKernelArg(v4k,a4++,sizeof(int),&B);
            clSetKernelArg(v4k,a4++,sizeof(int),&T);    clSetKernelArg(v4k,a4++,sizeof(int),&F);
            clSetKernelArg(v4k,a4++,sizeof(int),&Cin);
            clSetKernelArg(v4k,a4++,sizeof(int),&Tout); clSetKernelArg(v4k,a4++,sizeof(int),&Fout);
            clSetKernelArg(v4k,a4++,sizeof(int),&kH);   clSetKernelArg(v4k,a4++,sizeof(int),&kW);
            clSetKernelArg(v4k,a4++,sizeof(int),&sh);   clSetKernelArg(v4k,a4++,sizeof(int),&sw);
            clSetKernelArg(v4k,a4++,sizeof(int),&p1);   clSetKernelArg(v4k,a4++,sizeof(int),&p2);
            const int elu4 = apply_elu ? 1 : 0;
            clSetKernelArg(v4k,a4++,sizeof(int),&elu4);
            size_t ig4[2] = {(size_t)(K/4), (size_t)M};
            cl_ctx.profEnqueue(v4k,2,ig4,nullptr,(v4k == imk_v4T) ? "im2col_v4T" : "im2col_v4");
        }
        else cl_ctx.profEnqueue(im2col_kern,2,ig,nullptr,"im2col");
        if (OpenCLContext::profWantEvent()) { clFinish(queue); cl_ctx.compAccum("im2col_total"); }
    } else {
        col = in;  // identity — caller still owns `in`; do NOT release it below.
    }
    cl_mem out=pool_alloc(cl_ctx.context(),CL_MEM_READ_WRITE,(size_t)M*N*sizeof(float),nullptr,&err);
    if(!out){ NNOPT_ERROR_FMT("conv_clblast: out alloc failed (M=%d N=%d, %zu MB)", M, N,
                              ((size_t)M*(size_t)N*sizeof(float))>>20);
              if (col != in) pool_free(col); return nullptr; }
    cl_command_queue q=queue;
    clSetKernelArg(bk,0,sizeof(cl_mem),&biasbuf); clSetKernelArg(bk,1,sizeof(cl_mem),&out);
    clSetKernelArg(bk,2,sizeof(int),&M); clSetKernelArg(bk,3,sizeof(int),&N);
    size_t bg=(size_t)M*N; cl_ctx.profEnqueue(bk,1,&bg,nullptr,"bias");
    // dot8 FEASIBILITY: fake-quantize col (per-row, M groups) and a temp W copy (per-channel, N groups)
    // to int8 grid, then run the SAME fp32 SGEMM — measures int8 precision loss without the dot8 kernel.
    // Skipped on the ident-1×1 path (col aliases the caller's `in`; quantizing it would corrupt state).
    cl_mem Wgemm = Wbuf;
    if (fakeq8_enabled() && col != in) {
        static cl_kernel fqk=nullptr; cl_int fe=CL_SUCCESS;
        if(!fqk){ cl_program p=cl_ctx.build_program_from_file("kernels/fakeq8_rows.cl"); if(p) fqk=clCreateKernel(p,"fakeq8_rows",&fe); }
        if (fqk) {
            const size_t LW=64;
            clSetKernelArg(fqk,0,sizeof(cl_mem),&col); clSetKernelArg(fqk,1,sizeof(int),&K);
            size_t gM[1]={(size_t)M*LW}, lw[1]={LW}; cl_ctx.profEnqueue(fqk,1,gM,lw,"fakeq8");
            Wgemm=pool_alloc(cl_ctx.context(),CL_MEM_READ_WRITE,(size_t)N*K*sizeof(float),nullptr,&fe);
            clEnqueueCopyBuffer(q,Wbuf,Wgemm,0,0,(size_t)N*K*sizeof(float),0,nullptr,nullptr);
            clSetKernelArg(fqk,0,sizeof(cl_mem),&Wgemm); clSetKernelArg(fqk,1,sizeof(int),&K);
            size_t gN[1]={(size_t)N*LW}; cl_ctx.profEnqueue(fqk,1,gN,lw,"fakeq8");
        }
    }
    // C = 1*col @ W^T + 1*C(=bias). fp32 SGEMM is required: full-fp16 HGEMM was tested and is non-viable —
    // the codec's deep upsampling cascade produces ~60000-magnitude intermediates that overflow fp16 (65504)
    // → NaN → silence. fp16 *weights* are safe (audio cos 1.0) but a mixed fp16-W/fp32-act GEMM would need a
    // custom kernel (CLBlast has no mixed-precision mode). See project_magenta_codec_fp16_weights_safe memory.
    // The GEMM goes through CLBlast, which does not use our profEnqueue wrapper — so the single
    // largest consumer in the codec has never appeared in a profile. Hand CLBlast an event and
    // register it, so "is the codec GEMM-bound or im2col-bound" becomes a measurement.
    // ── weights pre-transposed [N,K] -> [K,N], cached ───────────────────────────────────────────
    // CLBlast skips its temporary buffers only when the operand needs no transpose (NoTempBuffer,
    // xgemm.hpp). Passing TransposeYes therefore made it allocate a temp and transpose the whole
    // weight matrix EVERY call, for weights that never change. Do it once instead.
    // NNOPT_WT=0 restores the old TransposeYes path for A/B.
    static int wt_on = -1, wt_epoch = -1;
    if (wt_epoch != nnopt_toggle_epoch()) {
        const char* e2 = std::getenv("NNOPT_WT");
        wt_on = (e2 && e2[0] == '0') ? 0 : 1;
        wt_epoch = nnopt_toggle_epoch();
    }
    static std::map<cl_mem, cl_mem> s_wt_cache;
    cl_mem Wgemm_t = nullptr;
    if (wt_on) {
        auto it = s_wt_cache.find(Wgemm);
        if (it != s_wt_cache.end()) {
            Wgemm_t = it->second;
        } else {
            static cl_kernel tk = nullptr;
            if (!tk) { cl_program tp = cl_ctx.build_program_from_file("kernels/transpose_nk_to_kn_f32.cl");
                       if (tp) tk = clCreateKernel(tp, "transpose_nk_to_kn_f32", &err); }
            if (tk) {
                cl_mem wt = clCreateBuffer(cl_ctx.context(), CL_MEM_READ_WRITE,
                                           (size_t)N*K*sizeof(float), nullptr, &err);
                if (wt && err == CL_SUCCESS) {
                    clSetKernelArg(tk,0,sizeof(cl_mem),&Wgemm); clSetKernelArg(tk,1,sizeof(cl_mem),&wt);
                    clSetKernelArg(tk,2,sizeof(int),&N);        clSetKernelArg(tk,3,sizeof(int),&K);
                    size_t tg[2]={(size_t)K,(size_t)N};
                    if (cl_ctx.profEnqueue(tk,2,tg,nullptr,"wtranspose") == CL_SUCCESS) {
                        // Cached for the life of the process: these are model weights, not activations.
                        s_wt_cache[Wgemm] = wt; Wgemm_t = wt;
                    } else { clReleaseMemObject(wt); }
                }
            }
        }
    }

    // One line per DISTINCT shape: CLBlast needs M and K multiples of MWG/KWG for A, and M and N
    // multiples of MWG/NWG for C, or it allocates a temporary and copies. Which of our shapes miss
    // that is the difference between a 61 ms matmul and 10.9 s of copying.
    if (OpenCLContext::profWantEvent()) {
        static std::set<std::tuple<int,int,int>> seen;
        if (seen.insert(std::make_tuple(M,N,K)).second)
            std::fprintf(stderr, "GEMMSHAPE M=%d(%s32) N=%d(%s32) K=%d(%s32)\n",
                         M, (M%32)?"NOT x":"x", N, (N%32)?"NOT x":"x", K, (K%32)?"NOT x":"x");
    }
    // clFinish-bracketed wall around the WHOLE CLBlast call. The per-event number only covers the
    // last kernel CLBlast enqueues; the indirect path also pads/transposes A, B and C, and those
    // passes are what we cannot otherwise see. Serialising here distorts overlap, so it only runs
    // under NNOPT_PROFILE.
    const bool want_ev = OpenCLContext::profWantEvent();
    if (want_ev) { clFinish(q); cl_ctx.compTic(); }
    cl_event gemm_ev = nullptr;
    // With the pre-transposed weights B is [K,N] with ld=N and needs no transpose, which is the
    // condition CLBlast requires to use the caller's buffer directly instead of a temporary.
    CLBlastStatusCode st = Wgemm_t
        ? CLBlastSgemm(CLBlastLayoutRowMajor, CLBlastTransposeNo, CLBlastTransposeNo,
            (size_t)M,(size_t)N,(size_t)K, 1.0f, col,0,(size_t)K, Wgemm_t,0,(size_t)N, 1.0f, out,0,(size_t)N, &q,
            want_ev ? &gemm_ev : nullptr)
        : CLBlastSgemm(CLBlastLayoutRowMajor, CLBlastTransposeNo, CLBlastTransposeYes,
            (size_t)M,(size_t)N,(size_t)K, 1.0f, col,0,(size_t)K, Wgemm,0,(size_t)K, 1.0f, out,0,(size_t)N, &q,
            want_ev ? &gemm_ev : nullptr);
    if (want_ev) {
        OpenCLContext::profEventAdd("gemm", gemm_ev);
        clFinish(q); cl_ctx.compAccum("clblast_gemm_total");
    }
    if (col != in) pool_free(col);
    if (Wgemm != Wbuf) pool_free(Wgemm);
    if(st!=CLBlastSuccess){ NNOPT_ERROR_FMT("conv_clblast: CLBlastSgemm st=%d", (int)st); pool_free(out); return nullptr; }
    return out;
}
#endif

// OPT (codec fused conv): route codec convs through the direct register-tiled kernels
// (conv2d_f32w / conv2d_transpose_f32w) which compute the conv WITHOUT materializing an
// im2col col-buffer (no DRAM write+read of [M,K], and no per-chunk OOM ceiling). ELU is fused
// into the input gather. NNOPT_DIRECTCONV=1 selects this over the im2col+CLBlast SGEMM path.
static bool direct_conv_enabled() {
    // Epoch-keyed, not latch-once: latching caches the value during the WARM-UP render,
    // before a serve request can set it, so the toggle silently never applies and the run
    // reports the default under the experiment's name.
    static int v = -1, v_epoch = -1;
    if (v_epoch != nnopt_toggle_epoch()) {
        const char* e = std::getenv("NNOPT_DIRECTCONV");
        v = (e && e[0] != '0') ? 1 : 0;
        v_epoch = nnopt_toggle_epoch();
    }
    return v != 0;
}

// OPT (cropped/lazy l5 cascade): only the last 2*T_chunk frames of l5 survive the l6 crop. Blocks
// 1-5 + child6 are time-stride-1 with a fixed left receptive field (5×4 + child6's 6 = 26 frames),
// so feeding them only [keptStart-MARGIN, end) yields the kept frames bit-exact while skipping ~half
// the late (large-F) cascade compute at large CHUNK. NNOPT_CROPL5=1.
// ── CLBlast GEMM tuning override ────────────────────────────────────────────────────────────────
// CLBlast picks GEMM tile parameters from a database keyed on the device. It ships tuned entries for
// Adreno 640 / 730 / 740 — but NOT 840, so an 840 silently falls back to the GENERIC default, which
// is 32x32 tiles with local-memory staging ON. Every tuned Adreno entry wants the opposite: 64x128
// tiles with staging OFF. The codec is ~3 s of CLBlast SGEMM, so this is not a detail.
//
// NNOPT_XGEMM=a740|a640|def selects a parameter set at runtime, because which one wins is a
// per-device question and this device is not in anyone's database. Purely a scheduling change —
// the arithmetic is identical, so audio cannot move.
static cl_mem run_codec_decode(OpenCLContext& cl_ctx, Weights& weights, cl_command_queue queue,
                               const std::vector<int>& codes, int T, int* nwav_o, cl_mem codes_gpu);

// ── Have the codec's function-local statics been built yet? ──────────────────
// Every codec kernel, program and weight-transpose cache inside run_codec_decode
// is a `static` guarded by a plain null check, so the FIRST call in a process must
// not race a second thread. That is the only reason the pipelined path runs chunk 0
// inline on the AR thread — and it costs a full chunk of AR/codec overlap.
//
// In `--serve` the warm-up render (main.cpp, 5 frames) already builds all of them
// before any request arrives, so from request #1 onward chunk 0 can go straight to
// the worker. Set only on a SUCCESSFUL completed decode, and only ever by the first
// caller (no worker exists yet at that point), so this is a one-way latch.
static std::atomic<bool> g_codec_statics_warm{false};

// ── CLBlast GEMM parameters: a fixed per-device table ───────────────────────────────────────────
// This used to be an 8-candidate sweep run at startup against a real codec chunk, with the winner
// cached to disk. Three things were wrong with that, and only the first is obvious:
//
//   1. The cache never survived. It was written to the engine's working directory, which the app
//      re-stages on every model load, so every launch paid the full sweep again.
//   2. So every launch RE-DECIDED the parameters from a noisy one-chunk microbenchmark. Two runs of
//      the identical build could land on different tile sizes, which means no two measurements of
//      anything else were comparable either. A benchmark you cannot repeat is not a benchmark.
//   3. The question does not need deciding at runtime. The answer is a property of the GPU, it was
//      already measured, and a shipped app re-deriving it on a user's phone buys them nothing but a
//      slower launch.
//
// So: a table. Adreno 6xx takes a640 (measured on the 620: codec -23%). Adreno 7xx takes a740, the
// closest genuinely-tuned entry CLBlast ships. Adreno 8xx takes ldsA, which was MEASURED on an 840
// rather than borrowed: 1.474 s of codec against a740's 1.639 (-10%), in a sweep whose repeated
// control differed by 2%, so the margin is real. Anything else keeps CLBlast's own choice rather
// than guessing. NNOPT_XGEMM still overrides, for when a new device needs a set picking.
//
// Why ldsA and not a bigger tile: in conv_clblast the GEMM is C = col x W, so A is the im2col
// column matrix and B is the weights. Every input element appears in several columns of A, which is
// reuse worth a local-memory copy; each weight tile is read once and streams fine from global.
// Staging A alone won. Staging B alone measured 1.606 — nothing. Staging BOTH measured 1.611, 9%
// WORSE than A alone, because the second 8 KB of LDS costs occupancy and buys a reuse that is not
// there. The tile size was never the lever; which operand you stage is.
namespace {
// Order: GEMMK KREG KWG KWI MDIMA MDIMC MWG NDIMB NDIMC NWG SA SB STRM STRN VWM VWN
// CLBlast requires MWG%(MDIMC*VWM)==0, NWG%(NDIMC*VWN)==0, MWG%(MDIMA*VWM)==0,
// NWG%(NDIMB*VWN)==0, KWG%KWI==0, and MDIMC*NDIMC <= the device's max workgroup size.
const size_t kXgemmA740[16] = {0,1,32,2,16,16,64,8,8,128,0,0,1,0,2,4};
const size_t kXgemmA640[16] = {0,1,32,2,16,16,64,8,8, 64,0,0,0,0,4,4};
// Adreno 8xx. a640's 64x64 tile with the A operand — and ONLY the A operand — staged through local
// memory. Measured on an 840, 2026-08-26: 1.474 s of codec against 1.639 for the a740 entry it
// replaces, over a run whose repeated-control spread was 2%. See BENCHMARK.md "Round 2".
const size_t kXgemmLdsA[16] = {0,1,32,2,16,16,64,8,8, 64,1,0,0,0,4,4};
const char*  kXgemmNames[16] = {"GEMMK","KREG","KWG","KWI","MDIMA","MDIMC","MWG",
                                "NDIMB","NDIMC","NWG","SA","SB","STRM","STRN","VWM","VWN"};

bool apply_xgemm(OpenCLContext& cl_ctx, const size_t* v) {
    std::unordered_map<std::string,size_t> params;
    for (int i = 0; i < 16; ++i) params[kXgemmNames[i]] = v[i];
    return clblast::OverrideParameters(cl_ctx.device(), "Xgemm", clblast::Precision::kSingle, params)
           == clblast::StatusCode::kSuccess;
}

// ── Xgemm candidate set ──────────────────────────────────────────────────────
// CLBlast's SGEMM kernel is not one kernel: it is a template over tile sizes, register blocking,
// vector widths and whether the operands are staged through local memory. The right values are a
// property of the GPU (CU count, register file, LDS, memory system), which is why CLBlast ships a
// per-device database — and the Adreno 840 is not in it. `xgemm_for_device` therefore falls back to
// a table keyed on the model number, and 840 >= 700 lands on the 730/740 entry.
//
// Every set below satisfies CLBlast's five divisibility constraints
//   MWG % (MDIMC*VWM) == 0, NWG % (NDIMC*VWN) == 0, MWG % (MDIMA*VWM) == 0,
//   NWG % (NDIMB*VWN) == 0, KWG % KWI == 0
// and keeps MDIMC*NDIMC (the workgroup) within 1024 and any LDS staging within 32 KB.
struct XgemmCand { const char* name; size_t p[16]; };

// ── Derive a parameter set from what the DEVICE reports ──────────────────────
// Everything else in this table is a hand-me-down: "a640" is a tuned Adreno 640, "a740" a 730/740.
// An 840 has more CUs, more LDS and a bigger workgroup ceiling than either, so borrowing their tile
// sizes leaves capability on the floor by construction. This asks the driver what it actually has
// and sizes the tiles to it.
//
//   MDIMC*NDIMC  = the workgroup, capped by CL_DEVICE_MAX_WORK_GROUP_SIZE
//   KWG*(MWG+NWG)*4 = the LDS a staged (SA/SB) kernel needs, capped by CL_DEVICE_LOCAL_MEM_SIZE
//   CU count     = how many workgroups have to exist before the GPU is full, which is what argues
//                  for SMALLER tiles on the codec: it is many modest GEMMs, not one large one.
static void xgemm_derive(OpenCLContext& cl_ctx, size_t out[16], int rung) {
    size_t max_wg = 256; cl_uint cus = 8; cl_ulong lmem = 32768;
    clGetDeviceInfo(cl_ctx.device(), CL_DEVICE_MAX_WORK_GROUP_SIZE, sizeof(max_wg), &max_wg, nullptr);
    clGetDeviceInfo(cl_ctx.device(), CL_DEVICE_MAX_COMPUTE_UNITS,  sizeof(cus),    &cus,    nullptr);
    clGetDeviceInfo(cl_ctx.device(), CL_DEVICE_LOCAL_MEM_SIZE,     sizeof(lmem),   &lmem,   nullptr);
    (void)cus; (void)lmem;

    // MEASURED CONSTRAINT, not a guess: hw_b (wg 8x8, m32/n64) took the device down on first
    // execution, while a640 (wg 16x8, the SAME n64 tile) ran fine and measured 1752 ms. hw_a
    // (wg 8x8, m32/n32) also survived. So the 8x8 workgroup is not safe in general on this driver
    // and CLBlast's documented constraints do not exclude it — every one of them is satisfied by
    // hw_b. The workgroup therefore stays at 16x8 = 128 and the rungs vary the TILE instead.
    size_t mdimc = 16, ndimc = 8;
    while (mdimc * ndimc > max_wg && mdimc > 4) mdimc >>= 1;

    static const size_t kM[4]  = {32, 64, 64, 32};
    static const size_t kN[4]  = {64, 64, 128, 32};
    static const size_t kVM[4] = { 2,  4,   2,  2};
    const size_t vwm = kVM[rung], vwn = 4;

    const size_t mstep = mdimc * vwm, nstep = ndimc * vwn;
    size_t mwg = kM[rung], nwg = kN[rung];
    if (mwg % mstep) mwg = ((mwg + mstep - 1) / mstep) * mstep;
    if (nwg % nstep) nwg = ((nwg + nstep - 1) / nstep) * nstep;

    const size_t v[16] = {0,1,32,2,mdimc,mdimc,mwg,ndimc,ndimc,nwg,0,0,0,0,vwm,vwn};
    for (int i = 0; i < 16; ++i) out[i] = v[i];
}

// A candidate's FIRST execution is what kills the device, so make that first execution as small as
// possible: one 128x128x128 GEMM on scratch buffers instead of the whole codec cascade. It costs
// ~1 ms, it forces the driver to compile and run this parameter set, and if the set is going to
// fault it does so having touched nothing the codec owns. Returns false if CLBlast rejects it.
static bool xgemm_probe(OpenCLContext& cl_ctx, cl_command_queue queue) {
    const int N = 128;
    cl_int e = CL_SUCCESS;
    const size_t bytes = (size_t)N * N * sizeof(float);
    cl_mem a = pool_alloc(cl_ctx.context(), CL_MEM_READ_WRITE, bytes, nullptr, &e);
    cl_mem b = pool_alloc(cl_ctx.context(), CL_MEM_READ_WRITE, bytes, nullptr, &e);
    cl_mem c = pool_alloc(cl_ctx.context(), CL_MEM_READ_WRITE, bytes, nullptr, &e);
    if (!a || !b || !c) { if (a) pool_free(a); if (b) pool_free(b); if (c) pool_free(c); return false; }
    // CLBlastSgemm (the C entry point) is what the codec's conv path already calls, so this adds no
    // new dynamic import. Buffers are deliberately left uninitialised: this probe exists to make the
    // driver COMPILE AND RUN the parameter set, and the arithmetic result is discarded.
    CLBlastStatusCode st = CLBlastSgemm(CLBlastLayoutRowMajor, CLBlastTransposeNo, CLBlastTransposeYes,
                                        N, N, N, 1.0f, a, 0, N, b, 0, N, 0.0f, c, 0, N, &queue, nullptr);
    if (st == CLBlastSuccess) clFinish(queue);
    pool_free(a); pool_free(b); pool_free(c);
    return st == CLBlastSuccess;
}

//                                    GEMMK KREG KWG KWI MDIMA MDIMC  MWG NDIMB NDIMC  NWG SA SB STRM STRN VWM VWN
static const XgemmCand kXgemmCands[] = {
    {"a740",   {0,1,32,2,16,16, 64, 8, 8,128,0,0,1,0,2,4}},  // what the 840 uses today
    {"a640",   {0,1,32,2,16,16, 64, 8, 8, 64,0,0,0,0,4,4}},
    {"big",    {0,1,32,2,16,16,128,16,16,128,0,0,0,0,4,4}},   // 128x128 tiles, WG 256
    {"wide",   {0,1,32,2,16,16, 64, 8, 8,256,0,0,1,0,2,4}},
    {"def",    {0,1,32,2, 8, 8, 32, 8, 8, 32,1,1,0,0,4,4}},   // generic fallback, LDS staged
    {"small",  {0,1,32,2, 8, 8, 32, 8, 8, 32,0,0,0,0,4,4}},   // 32x32, WG 64 — most workgroups
    {"m64n32", {0,1,32,2,16,16, 64, 8, 8, 32,0,0,0,0,4,4}},
    {"m32n64", {0,1,32,2, 8, 8, 32, 8, 8, 64,0,0,0,0,4,4}},
    {"lds64",  {0,1,32,2,16,16, 64, 8, 8, 64,1,1,0,0,4,4}},   // a640 + LDS staging (16 KB)
    {"k16",    {0,1,16,2,16,16, 64, 8, 8, 64,0,0,0,0,4,4}},   // a640, shallower K
    {"k64",    {0,1,64,2,16,16, 64, 8, 8, 64,0,0,0,0,4,4}},   // a640, deeper K
    {"vw2",    {0,1,32,2,16,16, 64, 8, 8, 64,0,0,0,0,2,2}},   // a640, narrower vectors
    {"strm",   {0,1,32,2,16,16, 64, 8, 8, 64,0,0,1,0,4,4}},   // a640 + strided M access
    // ── Round 2: the staging axis ────────────────────────────────────────────
    // Round 1's finding, in one line: `lds64` (a640's 64x64 tile, staged through LDS) measured
    // 1.610 s against 1.889 for the SAME tile unstaged and 1.871 for the shipped a740 — the only
    // axis in thirteen candidates that moved the codec more than the run-to-run spread. It is also
    // the only staged set in that round with a tile worth staging: `def`, the other one, is 32x32.
    //
    // That spread matters and these sets exist to beat it. "default" and "a740" select IDENTICAL
    // parameters on this device (xgemm_for_device sends 840 >= 700 to the a740 table), so round 1
    // measured the same configuration twice by accident: 2.182 and 1.871, a 17% gap on nothing.
    // A single lds64 run is therefore suggestive, not decided, which is why "default" leads the
    // app-side list — it re-measures the incumbent immediately before the challenger, on the same
    // thermal state, every time the sweep is run.
    //
    // The four axes below ask what about staging pays, since 16 KB of LDS is a real cost in
    // occupancy — an Adreno CU that can hold four workgroups at 8 KB holds two at 16 KB:
    //   ldsA / ldsB  — stage ONE operand (8 KB). If only A (the weights) benefits, this is lds64's
    //                  win at half the LDS, which is a second win on top of it.
    //   lds128       — staged with a740's wider N tile (24 KB). a740's NWG=128 and staging were
    //                  the two things that ever beat the a640 baseline; this is the cross.
    //   ldsm128      — the same question on M (24 KB).
    //   lds64k16/k64 — staging depth. k64 placed 2nd unstaged, so K is live; staged it doubles LDS
    //                  to exactly 32 KB, which the driver may refuse. A refusal is reported as
    //                  probefail and costs nothing, so it goes last.
    {"ldsA",     {0,1,32,2,16,16, 64, 8, 8, 64,1,0,0,0,4,4}},  // stage A only (8 KB)
    {"ldsB",     {0,1,32,2,16,16, 64, 8, 8, 64,0,1,0,0,4,4}},  // stage B only (8 KB)
    {"lds128",   {0,1,32,2,16,16, 64, 8, 8,128,1,1,0,0,4,4}},  // staged, N tile 128 (24 KB)
    {"ldsm128",  {0,1,32,2,16,16,128, 8, 8, 64,1,1,0,0,4,4}},  // staged, M tile 128 (24 KB)
    {"lds64k16", {0,1,16,2,16,16, 64, 8, 8, 64,1,1,0,0,4,4}},  // staged, shallow K (8 KB)
    {"lds64k64", {0,1,64,2,16,16, 64, 8, 8, 64,1,1,0,0,4,4}},  // staged, deep K (32 KB — may reject)
};
}  // namespace

// ── Xgemm parameters for HALF precision ─────────────────────────────────────────────────────────
// CLBlast keeps a SEPARATE parameter set per precision, and `apply_xgemm` above overrides only
// kSingle. So every tile decision measured in rounds 1 and 2 applies to the fp32 codec and the fp16
// codec inherited none of it — HGEMM has been running CLBlast's generic fallback on a device with no
// database entry, which is the same configuration round 1 measured at 2.182 s against a tuned
// 1.639 s. That is most likely why fp16 returned 5.8% instead of the 2.41x HGEMM promises.
//
// The fp32 answer is also not expected to transfer, for two hardware reasons:
//
//   VECTOR WIDTH. VWM/VWN=4 moves 128 bits per access in fp32. Four HALVES is 64 bits — half the
//   bus width. The fp16 equivalent of VW=4 is VW=8, and CLBlast's constraint MWG %% (MDIMC*VWM) == 0
//   then forces MWG >= 128 at MDIMC=16. That is not an obstacle to work around: it pushes the tile
//   to the larger size fp16 can now afford.
//
//   LOCAL MEMORY. A staged tile costs KWG*(MWG+NWG)*sizeof(elem). Halving the element halves the
//   LDS, so the occupancy argument that made `ldsA` stage only the A operand in fp32 is weaker here
//   — staging both, or staging a bigger tile, may now pay.
//
// Every set below satisfies CLBlast's divisibility constraints and keeps MDIMC*NDIMC <= 1024.
static bool apply_xgemm_half(OpenCLContext& cl_ctx, const size_t* v) {
    std::unordered_map<std::string,size_t> params;
    for (int i = 0; i < 16; ++i) params[kXgemmNames[i]] = v[i];
    return clblast::OverrideParameters(cl_ctx.device(), "Xgemm", clblast::Precision::kHalf, params)
           == clblast::StatusCode::kSuccess;
}

//                                        GEMMK KREG KWG KWI MDIMA MDIMC  MWG NDIMB NDIMC  NWG SA SB STRM STRN VWM VWN
// ROUND 2, centred on the MEASURED default instead of a guess.
//
// RetrieveParameters says CLBlast runs kHalf at m64 n64 k32 wg8x8 vw4x4 sa1 sb1, which is
// acc64 — (MWG/MDIMC)*(NWG/NDIMC) — the HIGHEST accumulator count in round 1's table, and it beat
// every candidate. So the "smaller register footprint wins" story was wrong: h_ldsA at acc32
// measured 1.724 s against h_vw8 at acc64's 1.547 s. Round 1's data said so and I read it backwards.
//
// The one property the winner has that NONE of round 1's nine candidates had is a workgroup of
// 8x8 = 64 work items; every candidate used 128 or 256. The guide (§3.1.2) notes that a workgroup
// at or below the wave size runs as a SINGLE WAVE, which would make the barriers a staged GEMM needs
// free, where at 128+ they are real cross-wave synchronisation. h_wg64/h_wg128/h_wg256 below differ
// in nothing but that, so this round tests the hypothesis instead of retelling it — and the engine
// now prints the device's actual wave size, so it stops being a guess either way.
//
// h_wg64 is the default's exact shape stated explicitly. It should tie with `default`; if it does
// not, the gap is this sweep's noise floor and every margin below it is meaningless.
static const XgemmCand kXgemmHalfCands[] = {
    // ── the workgroup probe: identical 64x64 tile, three workgroup sizes ──
    {"h_wg64",   {0,1,32,2, 8, 8, 64, 8, 8, 64,1,1,0,0,4,4}},  // 64 threads  · acc 64 · = the default
    {"h_wg128",  {0,1,32,2,16,16, 64, 8, 8, 64,1,1,0,0,4,4}},  // 128 threads · acc 32
    {"h_wg256",  {0,1,32,2,16,16, 64,16,16, 64,1,1,0,0,4,4}},  // 256 threads · acc 16
    // ── everything below holds wg8x8 and varies one axis off the default ──
    {"h_k16",    {0,1,16,2, 8, 8, 64, 8, 8, 64,1,1,0,0,4,4}},  // shallower K
    {"h_k64",    {0,1,64,2, 8, 8, 64, 8, 8, 64,1,1,0,0,4,4}},  // deeper K
    {"h_sa",     {0,1,32,2, 8, 8, 64, 8, 8, 64,1,0,0,0,4,4}},  // stage A only — what won in fp32
    {"h_nostage",{0,1,32,2, 8, 8, 64, 8, 8, 64,0,0,0,0,4,4}},  // stage neither
    {"h_vw2",    {0,1,32,2, 8, 8, 64, 8, 8, 64,1,1,0,0,2,2}},  // narrower vectors
    {"h_m128",   {0,1,32,2, 8, 8,128, 8, 8, 64,1,1,0,0,4,4}},  // taller tile  · acc 128
    {"h_n128",   {0,1,32,2, 8, 8, 64, 8, 8,128,1,1,0,0,4,4}},  // wider tile   · acc 128
    {"h_m32n32", {0,1,32,2, 8, 8, 32, 8, 8, 32,1,1,0,0,4,4}},  // smaller tile · acc 16
};



// ── PHASE 1: what IS the default? ───────────────────────────────────────────────────────────────
// The first fp16 sweep found that CLBlast's own kHalf choice beat all nine of our candidates, which
// left us searching around a point whose coordinates we did not know. CLBlast can just tell us:
// RetrieveParameters returns the set actually in force. Printed once, so the next round is a search
// around a MEASURED baseline instead of a guess.
//
// acc/thread = (MWG/MDIMC) * (NWG/NDIMC) is the number of accumulators each work item holds in
// registers, and on Adreno that is the number that decides everything: the guide (80-NB295-11
// §3.1.2, §6.5) says a larger register footprint means fewer resident waves, and past a point the
// compiler spills to system RAM. Round 1's candidates ran 32-128; the winner is far smaller. That
// figure belongs on screen next to every candidate.
static void report_xgemm_half_active(OpenCLContext& cl_ctx) {
    std::unordered_map<std::string,size_t> p;
    if (clblast::RetrieveParameters(cl_ctx.device(), "Xgemm", clblast::Precision::kHalf, p)
        != clblast::StatusCode::kSuccess) {
        std::fprintf(stderr, "XGEMMH_ACTIVE retrieve failed\n"); std::fflush(stderr); return;
    }
    auto g = [&](const char* k)->size_t { auto it = p.find(k); return it == p.end() ? 0 : it->second; };
    const size_t mwg=g("MWG"), nwg=g("NWG"), mdimc=g("MDIMC"), ndimc=g("NDIMC");
    const size_t acc = (mdimc && ndimc) ? (mwg/mdimc)*(nwg/ndimc) : 0;
    std::fprintf(stderr,
        "XGEMMH_ACTIVE m%zu n%zu k%zu wg%zux%zu vw%zux%zu sa%zu sb%zu acc%zu\n",
        mwg, nwg, g("KWG"), mdimc, ndimc, g("VWM"), g("VWN"), g("SA"), g("SB"), acc);
    std::fflush(stderr);
}

// NNOPT_XGEMMH=<name> picks one, once, at process start. Same rule as the fp32 table: changing
// CLBlast parameters inside a live process leaves its cached kernels inconsistent with the work
// sizes derived from the new values, which on Adreno is a device reset. One config per PROCESS.
static void maybe_override_xgemm_half(OpenCLContext& cl_ctx) {
    static int applied_epoch = -1;
    if (applied_epoch == nnopt_toggle_epoch()) return;
    applied_epoch = nnopt_toggle_epoch();
    const char* sel = std::getenv("NNOPT_XGEMMH");
    if (!sel || !*sel) return;                       // untouched: CLBlast's own kHalf choice
    const size_t n = sizeof(kXgemmHalfCands) / sizeof(kXgemmHalfCands[0]);
    for (size_t i = 0; i < n; ++i) {
        if (std::strcmp(sel, kXgemmHalfCands[i].name) != 0) continue;
        const bool ok = apply_xgemm_half(cl_ctx, kXgemmHalfCands[i].p);
        std::fprintf(stderr, "XGEMMH_SET %s applied=%d\n", kXgemmHalfCands[i].name, (int)ok);
        std::fflush(stderr);
        return;
    }
    std::fprintf(stderr, "XGEMMH_SET %s UNKNOWN — keeping CLBlast's own kHalf choice\n", sel);
    std::fflush(stderr);
}

// Adreno model number from the device name. Parse the LAST run of digits, never substring-match:
// "Adreno (TM) 618" contains an '8' and a naive check hands a 6xx part the 8xx settings, silently,
// on a device nobody here is holding.
long nnopt_adreno_model(OpenCLContext& cl_ctx) {
    static long cached = -1;
    if (cached >= 0) return cached;
    const std::string dev = cl_ctx.device_name();
    long model = 0;
    for (size_t i = 0; i < dev.size(); ) {
        if (std::isdigit((unsigned char)dev[i])) { size_t j = i; long n = 0;
            while (j < dev.size() && std::isdigit((unsigned char)dev[j])) { n = n*10 + (dev[j]-'0'); ++j; }
            model = n; i = j;
        } else ++i;
    }
    cached = model;
    return cached;
}

// Pick by device, once. Deterministic: the same phone gets the same parameters on every launch, so
// a difference between two runs is a difference in what was measured, not in what was chosen.
static void xgemm_for_device(OpenCLContext& cl_ctx) {
    static int applied_epoch = -1;
    if (applied_epoch == nnopt_toggle_epoch()) return;
    applied_epoch = nnopt_toggle_epoch();
    if (std::getenv("NNOPT_XGEMM")) return;          // explicit override handled by the caller
    const std::string dev = cl_ctx.device_name();
    // Parse the MODEL NUMBER, do not substring-match digits. "Adreno (TM) 618" contains an '8' and
    // a naive check hands a 6xx part the 7xx/8xx parameters — silently, on a device nobody here is
    // holding. Take the last run of digits in the name and branch on its magnitude.
    const long model = nnopt_adreno_model(cl_ctx);
    const size_t* v = nullptr; const char* which = nullptr;
    if      (model >= 800) { v = kXgemmLdsA; which = "ldsA"; }
    else if (model >= 700) { v = kXgemmA740; which = "a740"; }
    else if (model >= 600) { v = kXgemmA640; which = "a640"; }
    if (!v) { std::fprintf(stderr, "XGEMM device=\"%s\" model=%ld — no table entry, keeping "
                                   "CLBlast's own choice\n", dev.c_str(), model);
              std::fflush(stderr); return; }
    const bool ok = apply_xgemm(cl_ctx, v);
    std::fprintf(stderr, "XGEMM device=\"%s\" model=%ld set=%s applied=%d\n",
                 dev.c_str(), model, which, (int)ok);
    std::fflush(stderr);
}


// ── On-device Xgemm sweep ────────────────────────────────────────────────────
// Times every candidate in kXgemmCands against a REAL codec chunk — the actual objective, not a
// synthetic GEMM — and returns a one-line summary the app can display. This exists because the
// device is a remote one on a farm: a tuner that is only reachable from a CLI is a tuner that never
// runs (handoff §6), and CLBlast's own offline tuner needs a shell on the phone.
//
// Correctness is gated BEFORE the clock is read. A parameter set that violates a constraint can
// still build and run while computing garbage, and "16.5% faster" has already once turned out to be
// a kernel quietly doing less work. Each candidate's output energy is compared against the first
// candidate's; anything that drifts is reported as BAD and its time is not comparable.
std::string nnopt_xgemm_sweep(OpenCLContext& cl_ctx, Weights& weights,
                              const std::vector<int>& grid, int n_frames, int chunk) {
    if (grid.empty() || n_frames <= 0) return "";
    // global token -> per-codebook RVQ code, the same transform run_song does before the codec.
    std::vector<int> codes(grid.size());
    for (size_t i = 0; i < grid.size(); ++i) {
        const int q = (int)(i % 12);
        codes[i] = ((grid[i] - 6 - q * 1024) % 1024 + 1024) % 1024;
    }
    if (chunk < kMinCodecFrames) chunk = kMinCodecFrames;
    cl_command_queue queue = cl_ctx.queue();
    using clk = std::chrono::steady_clock;

    // Hardware-derived sets, built from what THIS device reports, and measured FIRST — the sweep
    // heats the GPU as it runs, so whatever goes last is penalised by thermal drift. The borrowed
    // a640/a740 tables come after these precisely because they are the incumbents, not the
    // hypothesis. Sizes: 64 = one tile per CU pass on a many-small-GEMM workload, 128 = fewer,
    // larger tiles, staged = the same through local memory, which the 840 has 32 KB of and the
    // shipped tables never use.
    static bool s_derived_built = false;
    static XgemmCand s_derived[4];
    if (!s_derived_built) {
        s_derived_built = true;
        const char* nm[4] = {"hw_a", "hw_b", "hw_c", "hw_d"};
        for (int i = 0; i < 4; ++i) {
            s_derived[i].name = nm[i];
            xgemm_derive(cl_ctx, s_derived[i].p, i);
            const size_t* q = s_derived[i].p;
            // On screen, not just in a log: the device is remote and a hang ends the session, so a
            // parameter set that is only visible in stderr is a parameter set nobody ever sees.
            std::fprintf(stderr, "XGEMM_DERIVE %s wg=%zux%zu m=%zu n=%zu k=%zu vw=%zux%zu\n",
                         nm[i], q[5], q[8], q[6], q[9], q[2], q[14], q[15]);
        }
        {   // one combined line the app renders verbatim
            std::string d;
            for (int i = 0; i < 4; ++i) {
                const size_t* q = s_derived[i].p;
                char b[96];
                std::snprintf(b, sizeof(b), "%s=wg%zux%zu/m%zu/n%zu/vw%zux%zu ",
                              nm[i], q[5], q[8], q[6], q[9], q[14], q[15]);
                d += b;
            }
            std::fprintf(stderr, "XGEMM_DERIVED %s\n", d.c_str());
        }
        std::fflush(stderr);
    }

    // One pass over the chunked codec, exactly as production drives it. Returns wall ms, and
    // accumulates the output energy so a wrong-but-fast candidate cannot win.
    auto run_once = [&](double* energy_o) -> double {
        double energy = 0.0;
        clFinish(queue);
        const auto t0 = clk::now();
        for (int s = 0; s < n_frames; s += chunk) {
            const int cf = std::min(chunk, n_frames - s);
            if (cf < kMinCodecFrames) break;
            std::vector<int> slice(codes.begin() + (size_t)s * 12, codes.begin() + (size_t)(s + cf) * 12);
            int nw = 0;
            cl_mem w = run_codec_decode(cl_ctx, weights, queue, slice, cf, &nw, nullptr);
            if (!w) { clFinish(queue); return -1.0; }
            if (energy_o) {
                std::vector<float> ro((size_t)nw * 2);
                clEnqueueReadBuffer(queue, w, CL_TRUE, 0, ro.size() * sizeof(float), ro.data(), 0, nullptr, nullptr);
                for (float v : ro) energy += (double)v * (double)v;
            }
            pool_free(w);
        }
        clFinish(queue);
        const double ms = std::chrono::duration<double, std::milli>(clk::now() - t0).count();
        if (energy_o) *energy_o = energy;
        return ms;
    };

    // ── Resumable, time-budgeted ────────────────────────────────────────────
    // Every crash this port has produced lands after roughly 20-40 s of UNBROKEN GPU saturation,
    // across three unrelated workloads. A 13-candidate sweep is ~45 s of exactly that, so running
    // it in one press is a reliable way to kill the device before it finishes — which is how the
    // first attempt died at candidate 4 of 13 and got mistaken for a finished result.
    //
    // So each press measures a few candidates and stops well inside that window. State carries in
    // statics: press Generate again and it resumes where it left off, and the accumulated table is
    // reported every time. The GPU gets a full idle between presses, which is the one thing that
    // has never been true when it crashed.
    static size_t s_next = 0;
    static std::string s_acc;
    static double s_base_energy = -1.0;
    const size_t n_derived = 4;
    const size_t n_table = sizeof(kXgemmCands) / sizeof(kXgemmCands[0]);
    const size_t n_cands = n_derived + n_table;
    if (s_next >= n_cands) return s_acc + " (done)";

    const auto sweep_t0 = clk::now();
    std::string& out = s_acc;
    double& base_energy = s_base_energy;
    for (; s_next < n_cands; ++s_next) {
        // Stop BEFORE starting one that would run past the budget, not after.
        if (s_next > 0 &&
            std::chrono::duration<double, std::milli>(clk::now() - sweep_t0).count() > 5000.0) break;
        const XgemmCand& c = (s_next < n_derived) ? s_derived[s_next] : kXgemmCands[s_next - n_derived];
        // A crash here takes the whole session, so the stage has to be on screen BEFORE it happens.
        // "apply" = handing CLBlast the parameters, "jit" = the driver compiling a kernel for every
        // GEMM shape in the codec (a new set invalidates the program cache, so this is dozens of
        // clBuildProgram calls), "timed" = actually executing it. Whichever word is last on screen
        // is where it died.
        auto mark = [&](const char* stage) {
            std::fprintf(stderr, "XGEMM_NOW %s:%s [%zu/%zu]\n", c.name, stage, s_next + 1, n_cands);
            std::fflush(stderr);
        };
        mark("apply");
        if (!apply_xgemm(cl_ctx, c.p)) { out += std::string(c.name) + ":x:reject "; continue; }
        mark("probe");
        if (!xgemm_probe(cl_ctx, queue)) { out += std::string(c.name) + ":x:probefail "; continue; }
        mark("jit");
        // First pass is discarded: it pays this parameter set's CLBlast JIT.
        if (run_once(nullptr) < 0.0) { out += std::string(c.name) + ":x:fail "; continue; }
        mark("timed");
        double energy = 0.0;
        const double ms = run_once(&energy);
        if (ms < 0.0) { out += std::string(c.name) + ":x:fail "; continue; }
        if (base_energy < 0.0) base_energy = energy;
        const double rel = (base_energy > 0.0) ? std::fabs(energy - base_energy) / base_energy : 0.0;
        char buf[64];
        std::snprintf(buf, sizeof(buf), "%s:%.0f%s ", c.name, ms, rel > 1e-4 ? ":BAD" : "");
        out += buf;
        std::fprintf(stderr, "XGEMM_SWEEP %-8s %8.1f ms  energy_rel=%.2e%s\n",
                     c.name, ms, rel, rel > 1e-4 ? "  <-- OUTPUT DIFFERS, not comparable" : "");
        // Emit the RUNNING total after every candidate, not once at the end. The tuning device is a
        // remote session that dies with whatever killed it — there is no reading a log afterwards,
        // so a result that only exists at completion is a result that is lost the moment the sweep
        // is what crashes. Partial output is already on screen when it goes.
        std::fprintf(stderr, "XGEMM_SWEEP_RESULT %s\n", out.c_str());
        std::fflush(stderr);
        // Let the GPU idle briefly between candidates. 26 back-to-back full codec decodes is the
        // heaviest sustained load this engine can produce, and a driver reset takes the session
        // with it — this costs nothing measurable because only the clFinish-bracketed run is timed.
        usleep(250 * 1000);   // plain libc: std::this_thread::sleep_for pulls in a libc++ symbol
                              // this binary did not previously import, and an unresolved import is a
                              // CANNOT LINK EXECUTABLE at load, not a link error here.
    }
    {   // progress, so a partial table is never mistaken for a finished one
        char pb[48];
        std::snprintf(pb, sizeof(pb), " [%zu/%zu]", s_next, n_cands);
        std::fprintf(stderr, "XGEMM_SWEEP_RESULT %s%s\n", out.c_str(), pb);
        std::fflush(stderr);
    }
    // Leave the device on whatever the normal selection would pick, not on the last candidate tried.
    nnopt_toggle_bump();
    xgemm_for_device(cl_ctx);
    char pb[48];
    std::snprintf(pb, sizeof(pb), " [%zu/%zu]", s_next, n_cands);
    std::string r = out;
    while (!r.empty() && r.back() == ' ') r.pop_back();
    return r + pb;
}

static void maybe_override_xgemm(OpenCLContext& cl_ctx) {
    static int applied_epoch = -1;
    if (applied_epoch == nnopt_toggle_epoch()) return;
    applied_epoch = nnopt_toggle_epoch();
    const char* sel = std::getenv("NNOPT_XGEMM");
    if (!sel || !*sel) return;                       // untouched: CLBlast's own choice

    // Name -> parameter set, straight out of kXgemmCands. This is the ONLY supported way to change
    // Xgemm parameters, and it is read once per process at startup.
    //
    // There used to be an in-process sweep that called OverrideParameters repeatedly to time each
    // candidate. It reset the GPU — and with it the whole test session — on four separate builds,
    // on configs with no geometry in common (wide, auto64s, hw_b, m32/n64@16x8), while every one of
    // those same configs is fine when it is the set chosen at startup. Switching parameters inside a
    // live process leaves CLBlast's cached kernels inconsistent with the work sizes computed from
    // the new parameters, which is an out-of-bounds launch, which is a device reset. Do not
    // reintroduce a runtime sweep: measure one config per PROCESS.
    const size_t n = sizeof(kXgemmCands) / sizeof(kXgemmCands[0]);
    for (size_t i = 0; i < n; ++i) {
        if (std::strcmp(sel, kXgemmCands[i].name) != 0) continue;
        const bool ok = apply_xgemm(cl_ctx, kXgemmCands[i].p);
        std::fprintf(stderr, "XGEMM_SET %s applied=%d\n", kXgemmCands[i].name, (int)ok);
        std::fflush(stderr);
        return;
    }
    std::fprintf(stderr, "XGEMM_SET %s UNKNOWN — keeping the device default\n", sel);
    std::fflush(stderr);
}

// ── Chunk duration: 40 ms per RVQ frame, not 20 ms ───────────────────────────
// The SpectroStream cascade upsamples T RVQ frames to 4T iSTFT frames (l4 conv-T:
// T -> 2T, l5 block 0: 2T -> 4T). At hop 480 that is T*1920 samples = 40 ms per
// frame, which is exactly what the reference emits: magenta_rt/mlx/system.py
// documents `samples shape: [T*1920, 2]` and loads
// `spectrostream_40ms_generic_48khz_stereo_config` ("25 frames = 1 second"), and
// the MRT2 model card's ~20 s receptive field only works out at 40 ms/frame
// (512-frame KV cache = 20.48 s).
//
// l6 used to crop the cascade output to its last 2T frames, halving every chunk
// to 20 ms/frame and discarding the first half of the music it had just spent AR
// time generating. That is the unexplained "duration mismatch" BENCHMARK.md warns
// about ("never wav-vs-reference cosine"). Full length is now the default.
//
// NNOPT_HALFAUDIO=1 restores the old crop for an A/B.
static bool half_audio_enabled() {
    static int v = -1, v_epoch = -1;
    if (v_epoch != nnopt_toggle_epoch()) {
        const char* e = std::getenv("NNOPT_HALFAUDIO");
        v = (e && e[0] != '0') ? 1 : 0;
        v_epoch = nnopt_toggle_epoch();
    }
    return v != 0;
}

static bool crop_l5_enabled() {
    // Epoch-keyed, not latch-once: latching caches the value during the WARM-UP render,
    // before a serve request can set it, so the toggle silently never applies and the run
    // reports the default under the experiment's name.
    static int v = -1, v_epoch = -1;
    if (v_epoch != nnopt_toggle_epoch()) {
        // Only legal when l6 discards the leading frames. At full length every
        // frame l5 produces is audio we keep, so there is nothing to skip and a
        // crop here would silently truncate the START of the chunk.
        const char* e = std::getenv("NNOPT_CROPL5");
        v = (e && e[0] != '0' && half_audio_enabled()) ? 1 : 0;
        v_epoch = nnopt_toggle_epoch();
    }
    return v != 0;
}

// Direct (col-buffer-free) conv2d / conv2dT. in [B,T,F,Cin] → out [B,Tout,Fout,Cout]. fp32 weights.
// transpose=false → conv2d_f32w; true → conv2d_transpose_f32w. apply_elu fuses ELU into the gather.
static cl_mem conv_direct(OpenCLContext& cl_ctx, cl_command_queue queue, cl_mem in, cl_mem Wbuf, cl_mem biasbuf,
                          int B,int T,int F,int Cin,int Tout,int Fout,int Cout,int kH,int kW,int sh,int sw,
                          int p1,int p2, bool transpose, bool apply_elu) {
    cl_int err=CL_SUCCESS;
    static cl_kernel fwd=nullptr, tr=nullptr;
    cl_kernel k=nullptr;
    if (transpose) {
        if(!tr){ cl_program p=cl_ctx.build_program_from_file("kernels/conv2d_transpose_f32w.cl");
            if(!p){NNOPT_ERROR("conv_direct: build conv2d_transpose_f32w");return nullptr;} tr=clCreateKernel(p,"conv2d_transpose_f32w",&err); }
        k=tr;
    } else {
        if(!fwd){ cl_program p=cl_ctx.build_program_from_file("kernels/conv2d_f32w.cl");
            if(!p){NNOPT_ERROR("conv_direct: build conv2d_f32w");return nullptr;} fwd=clCreateKernel(p,"conv2d_f32w",&err); }
        k=fwd;
    }
    if(!k){ NNOPT_ERROR("conv_direct: kernel create"); return nullptr; }
    cl_mem out=pool_alloc(cl_ctx.context(),CL_MEM_READ_WRITE,(size_t)B*Tout*Fout*Cout*sizeof(float),nullptr,&err);
    if(!out){ NNOPT_ERROR("conv_direct: out alloc"); return nullptr; }
    const int elu = apply_elu ? 1 : 0;
    int ai=0;
    clSetKernelArg(k,ai++,sizeof(cl_mem),&in); clSetKernelArg(k,ai++,sizeof(cl_mem),&Wbuf); clSetKernelArg(k,ai++,sizeof(cl_mem),&biasbuf); clSetKernelArg(k,ai++,sizeof(cl_mem),&out);
    clSetKernelArg(k,ai++,sizeof(int),&T); clSetKernelArg(k,ai++,sizeof(int),&F); clSetKernelArg(k,ai++,sizeof(int),&Cin);
    clSetKernelArg(k,ai++,sizeof(int),&Tout); clSetKernelArg(k,ai++,sizeof(int),&Fout); clSetKernelArg(k,ai++,sizeof(int),&Cout);
    clSetKernelArg(k,ai++,sizeof(int),&kH); clSetKernelArg(k,ai++,sizeof(int),&kW);
    clSetKernelArg(k,ai++,sizeof(int),&sh); clSetKernelArg(k,ai++,sizeof(int),&sw);
    clSetKernelArg(k,ai++,sizeof(int),&p1); clSetKernelArg(k,ai++,sizeof(int),&p2);
    clSetKernelArg(k,ai++,sizeof(int),&B); clSetKernelArg(k,ai++,sizeof(int),&elu);
    size_t g[3]={(size_t)((Cout+3)/4),(size_t)((Fout+3)/4),(size_t)B*Tout};
    if(cl_ctx.profEnqueue(k,3,g,nullptr,transpose?"conv2dT_d":"conv2d_d")!=CL_SUCCESS){ NNOPT_ERROR("conv_direct: enqueue"); pool_free(out); return nullptr; }
    return out;
}

// ── Conv2D (channels-last, fp32 codec weights). in [T,F,Cin] → out [Tout,Fout,Cout]. wp has .kernel/.bias.
// time_pad: "semicausal" (pad kH-1 on the left) or "same"; freq "same". Returns out + dims via pointers.
// ── streaming left-context for the codec cascade ────────────────────────────────────────────────
// Every cascade conv pads time on the LEFT only (padT0 = kH-1), i.e. it is semicausal: output frame
// t depends on input frames [t-(kH-1), t]. Today that context is manufactured by running the whole
// cascade over 2x the frames and throwing half the result away (l6 crops the last 2T of 4T).
//
// Keeping the last kH-1 input frames per conv supplies the same context for a few frames of state
// instead of half the cascade — and it is MORE correct than the current scheme, which restarts the
// context at every chunk boundary (the known "causal-per-chunk" discontinuity in this port).
//
// State is keyed by weight prefix, which is unique per conv site, and must be cleared whenever the
// audio stream restarts or a chunk would be fed context from a different song.
namespace {
// `q` is the queue the state was last written on. In the pipelined path chunk 0 is decoded inline
// on the AR thread while chunks 1+ run on the codec worker, and buffer contents are only ordered
// within a single queue — so state written on one and read from another is undefined. Recorded here
// so a mismatch drops the context (one seam) instead of reading garbage.
struct CodecTail { cl_mem mem = nullptr; int frames = 0, B = 0, F = 0, C = 0; cl_command_queue q = nullptr; };
std::map<std::string, CodecTail> g_codec_tails;
// Two threads touch this map — the AR thread for chunk 0, the codec worker for the rest.
std::mutex g_codec_tails_mu;

// DEFAULT ON as of 2026-08-28, and it verifies itself before it is trusted. See the self-check in
// run_song: chunked and whole decoding of the same grid must agree, and if they do not this turns
// itself off for the process. That is the expiry the 2026-08-27 quarantine was waiting on.
//
// This is how the reference produces gapless audio: every layer in the decode path keeps its own
// streaming state, so a chunk boundary is a slicing decision and not a numerical one. Two real bugs
// were fixed to get here (conv2d_op handed CLBlast the un-extended buffer with the extended length;
// the transpose convs had no state at all), and with it ON live audio goes SILENT after a handful of
// chunks while the buffer stays healthy and AudioTrack reports zero underruns — i.e. the engine is
// emitting zeros, not failing to keep up.
//
// Silence that arrives after N chunks rather than immediately is the signature of state that
// compounds, and this is the only state in the codec that does. The maths has been re-derived
// against the reference's Conv2DTranspose and the input-left-context form used here is equivalent to
// their output overlap-add, so the error is in the implementation, not the approach.
//
// It is OFF rather than deleted for exactly one reason: `codecchunk` decides it. Chunked and whole
// decoding of the same grid must agree, and that test now runs per-request so it is reachable from
// the device. If it reports DIFFERS, the state is wrong and this comes out; if it reports IDENTICAL,
// the silence is somewhere else and this goes back on. Until one of those happens, live audio ships
// with the 400 ms seam and no silence, because working-with-a-flaw beats silent.
namespace { bool g_codec_stream_forced_off = false; }
void codec_stream_disable() { g_codec_stream_forced_off = true; }

bool codec_stream_enabled() {
    if (g_codec_stream_forced_off) return false;
    static int v = -1, v_epoch = -1;
    if (v_epoch != nnopt_toggle_epoch()) {
        const char* e = std::getenv("NNOPT_CODECSTREAM");
        v = (e && e[0] == '0') ? 0 : 1;   // DEFAULT ON — gated by the self-check in run_song
        v_epoch = nnopt_toggle_epoch();
    }
    return v != 0;
}
}  // namespace

// The iSTFT overlap tail is part of the same streaming state as the conv context: both describe
// what the previous chunk left behind, and both are meaningful only within one continuous piece.
// One flag, one reset, so they can never disagree about whether a stream is starting.
static bool g_istft_tail_valid = false;

void codec_stream_reset() {
    std::lock_guard<std::mutex> lk(g_codec_tails_mu);
    for (auto& kv : g_codec_tails) if (kv.second.mem) pool_free(kv.second.mem);
    g_codec_tails.clear();
    g_istft_tail_valid = false;
}

// Prepend this conv's saved context to `in`, and remember the new tail. Returns the extended buffer
// (caller frees) and sets *T_ext, or nullptr when there is no usable state yet — in which case the
// caller keeps its normal zero-padded path, which is exactly right for the first chunk.
static cl_mem codec_stream_extend(OpenCLContext& cl_ctx, cl_command_queue queue,
                                  const std::string& key, cl_mem in, int B, int T, int F, int C,
                                  int want, int* T_ext) {
    if (want <= 0 || T < want) return nullptr;
    const size_t frame = (size_t)F * C * sizeof(float);
    cl_int e = CL_SUCCESS;
    std::lock_guard<std::mutex> lk(g_codec_tails_mu);
    auto it = g_codec_tails.find(key);
    cl_mem ext = nullptr;
    // Shape AND queue must match. A queue change means the previous chunk's copy has no ordering
    // guarantee against this read, so the context is dropped rather than trusted.
    if (it != g_codec_tails.end() && it->second.mem && it->second.q == queue &&
        it->second.frames == want && it->second.B == B && it->second.F == F && it->second.C == C) {
        const int Te = T + want;
        ext = pool_alloc(cl_ctx.context(), CL_MEM_READ_WRITE, (size_t)B * Te * frame, nullptr, &e);
        if (!ext) return nullptr;
        for (int b = 0; b < B; ++b) {
            clEnqueueCopyBuffer(queue, it->second.mem, ext, (size_t)b * want * frame,
                                (size_t)b * Te * frame, (size_t)want * frame, 0, nullptr, nullptr);
            clEnqueueCopyBuffer(queue, in, ext, (size_t)b * T * frame,
                                ((size_t)b * Te + want) * frame, (size_t)T * frame, 0, nullptr, nullptr);
        }
        *T_ext = Te;
    }
    // Save this call's tail for the next chunk regardless of whether we could use one this time.
    CodecTail& t = g_codec_tails[key];
    if (t.mem && (t.frames != want || t.B != B || t.F != F || t.C != C)) { pool_free(t.mem); t.mem = nullptr; }
    if (!t.mem) {
        t.mem = pool_alloc(cl_ctx.context(), CL_MEM_READ_WRITE, (size_t)B * want * frame, nullptr, &e);
        t.frames = want; t.B = B; t.F = F; t.C = C;
    }
    t.q = queue;
    if (t.mem) {
        for (int b = 0; b < B; ++b)
            clEnqueueCopyBuffer(queue, in, t.mem, ((size_t)b * T + (T - want)) * frame,
                                (size_t)b * want * frame, (size_t)want * frame, 0, nullptr, nullptr);
    }
    return ext;
}

static cl_mem conv2d_op(OpenCLContext& cl_ctx, Weights& weights, cl_command_queue queue,
                        cl_mem in, int T, int F, int Cin, const std::string& wp,
                        int sh, int sw, int* Tout_o, int* Fout_o, int* Cout_o, int B=1,
                        bool apply_elu=false) {
    cl_mem K = get_codec_f32(cl_ctx, wp + ".kernel");
    cl_mem b = get_codec_f32(cl_ctx, wp + ".bias");
    if (!K || !b) { NNOPT_ERROR_FMT("conv2d: missing %s .kernel/.bias", wp.c_str()); return nullptr; }
    const std::vector<int>& ks = get_codec_shape(cl_ctx, wp + ".kernel"); // [Cout,kH,kW,Cin]
    if (ks.size() < 3) { NNOPT_ERROR_FMT("conv2d: no shape for %s.kernel in the codec metadata", wp.c_str()); return nullptr; }
    const int Cout=ks[0], kH=ks[1], kW=ks[2];
    // freq 'same': pad_total = max((kW-1)+1 - sw, 0); time 'semicausal': pad kH-1 on the left only.
    const int padF_total = (kW-1)+1 - sw > 0 ? (kW-1)+1 - sw : 0;
    const int padF0 = padF_total/2;
    int padT_total = kH-1, padT0 = kH-1;
    int Tin_eff = T;
    cl_mem in_eff = in, in_ext = nullptr;
    // Streaming: prepend the previous chunk's last kH-1 frames and drop the zero padding, so this
    // conv sees real context instead of zeros at the chunk boundary. Output length is unchanged:
    // (T + (kH-1) - kH)/1 + 1 == T, which is what the zero-padded form produces too.
    if (codec_stream_enabled() && kH > 1 && sh == 1) {
        int Te = 0;
        in_ext = codec_stream_extend(cl_ctx, queue, wp, in, B, T, F, Cin, kH-1, &Te);
        if (in_ext) { in_eff = in_ext; Tin_eff = Te; padT_total = 0; padT0 = 0; }
    }
    const int Tout = (Tin_eff + padT_total - kH)/sh + 1;
    const int Fout = (F + padF_total - kW)/sw + 1;
    cl_int err=CL_SUCCESS; (void)err;
    cl_mem out=nullptr;
    const bool ident = (kH==1&&kW==1&&sh==1&&sw==1&&padT0==0&&padF0==0); (void)ident;
    // Direct (col-buffer-free) path: kernel fuses ELU into the gather, handles batch. Used by the
    // non-CLBlast build always, and by the CLBlast build under NNOPT_DIRECTCONV.
#ifdef USE_CLBLAST
    if (direct_conv_enabled()) {
#endif
        out = conv_direct(cl_ctx,queue,in_eff,K,b,B,Tin_eff,F,Cin,Tout,Fout,Cout,kH,kW,sh,sw,padT0,padF0,
                          /*transpose=*/false, /*apply_elu=*/apply_elu);
        if(!out) return nullptr;
#ifdef USE_CLBLAST
    } else {
        // OPT #7: fuse the pre-conv ELU into im2col. The ident-1×1 path has no gather to fuse into,
        // so fall back to a standalone elu there (and when NNOPT_NOFUSEELU disables fusion).
        const bool fuse = apply_elu && fuse_elu_enabled() && !ident;
        // in_eff, NOT in. When streaming prepends context, in_eff is the extended buffer and Tin_eff
        // its length; passing the ORIGINAL buffer with the EXTENDED length told CLBlast to read
        // (kH-1)*F*Cin floats past the end of it. That is why NNOPT_CODECSTREAM has never worked on
        // the shipped build — the conv_direct branch above always used in_eff, so only the CLBlast
        // path was wrong, and the CLBlast path is the one that ships.
        //
        // The free below compares against in_eff for the same reason: `cin != in` would free the
        // extended buffer here and again at `if (in_ext)`, which is a double free.
        cl_mem cin = in_eff;
        if (apply_elu && !fuse) cin = elu_op(cl_ctx,queue,in_eff,B*Tin_eff*F*Cin);
        static cl_kernel imk=nullptr;
        if(!imk){ cl_program p=cl_ctx.build_program_from_file("kernels/im2col_conv2d_f32.cl"); imk=clCreateKernel(p,"im2col_conv2d_f32",&err); g_im2col_plain=imk; }
        out=conv_clblast(cl_ctx,queue,cin,K,b,B,Tin_eff,F,Cin,Tout,Fout,Cout,kH,kW,sh,sw,padT0,padF0,imk,fuse);
        if (cin != in_eff) pool_free(cin);
        if(!out) return nullptr;
    }
#endif
    if (in_ext) pool_free(in_ext);
    if(Tout_o)*Tout_o=Tout; if(Fout_o)*Fout_o=Fout; if(Cout_o)*Cout_o=Cout;
    return out;
}

static cl_mem elu_op(OpenCLContext& cl_ctx, cl_command_queue queue, cl_mem x, int n) {
    cl_int err=CL_SUCCESS; static cl_kernel ek=nullptr;
    if(!ek){ cl_program p=cl_ctx.build_program_from_file("kernels/elu_f32.cl"); ek=clCreateKernel(p,"elu_f32",&err); }
    cl_mem out=pool_alloc(cl_ctx.context(),CL_MEM_READ_WRITE,(size_t)n*sizeof(float),nullptr,&err);
    clSetKernelArg(ek,0,sizeof(cl_mem),&x); clSetKernelArg(ek,1,sizeof(cl_mem),&out); clSetKernelArg(ek,2,sizeof(int),&n);
    size_t g=(size_t)n; cl_ctx.profEnqueue(ek,1,&g,nullptr,"elu"); 
    return out;
}

// ── decoder layer 1 (Residual): out = conv_b(x) + conv_s2(Elu(conv_s1(x))) ── (validates Conv2D, 1x1)
static cl_mem run_codec_l1(OpenCLContext& cl_ctx, Weights& weights, cl_command_queue queue, cl_mem x, int T, int F, int Cin) {
    const std::string P="spectrostream.embeddings_to_waveform_layer.layers.0.layers.1.";
    int Tb,Fb,Cb;
    cl_mem body = conv2d_op(cl_ctx,weights,queue,x,T,F,Cin,P+"body.layers.0.layers.0.inner",1,1,&Tb,&Fb,&Cb);
    int Ts,Fs,Cs;
    cl_mem s1 = conv2d_op(cl_ctx,weights,queue,x,T,F,Cin,P+"shortcut.layers.0.layers.0.inner",1,1,&Ts,&Fs,&Cs);
    cl_mem s1e = elu_op(cl_ctx,queue,s1,Ts*Fs*Cs); pool_free(s1);
    cl_mem s2 = conv2d_op(cl_ctx,weights,queue,s1e,Ts,Fs,Cs,P+"shortcut.layers.2.layers.0.inner",1,1,&Ts,&Fs,&Cs); pool_free(s1e);
    cl_mem out = residual_add(cl_ctx,queue,body,s2,Tb*Fb*Cb);
    pool_free(body); pool_free(s2);
    return out;
}

// ── Conv2DTranspose (gather form). Tout=T*sh, Fout=F*sw; padT=0, padF=(kW-1)/2 ('same'/'causal'). ──
// Drop the first `cut` time frames of a [B,Tfull,F,C] buffer, returning a fresh [B,Tfull-cut,F,C].
// A transpose conv turns `want` prepended input frames into `want*sh` leading output frames that the
// caller never asked for; those frames are the ones the previous chunk already emitted.
static cl_mem trim_front_time(OpenCLContext& cl_ctx, cl_command_queue queue, cl_mem in,
                              int B, int Tfull, int cut, int F, int C) {
    if (cut <= 0 || cut >= Tfull) return nullptr;
    const size_t frame = (size_t)F * C * sizeof(float);
    const int Tkeep = Tfull - cut;
    cl_int e = CL_SUCCESS;
    cl_mem out = pool_alloc(cl_ctx.context(), CL_MEM_READ_WRITE, (size_t)B * Tkeep * frame, nullptr, &e);
    if (!out) return nullptr;
    for (int b = 0; b < B; ++b)
        clEnqueueCopyBuffer(queue, in, out, ((size_t)b * Tfull + cut) * frame,
                            (size_t)b * Tkeep * frame, (size_t)Tkeep * frame, 0, nullptr, nullptr);
    return out;
}

static cl_mem conv2dtranspose_op(OpenCLContext& cl_ctx, Weights& weights, cl_command_queue queue,
                                 cl_mem in, int T, int F, int Cin, const std::string& wp,
                                 int sh, int sw, int* Tout_o, int* Fout_o, int* Cout_o, int B=1,
                                 bool apply_elu=false) {
    cl_mem K=get_codec_f32(cl_ctx,wp+".kernel"); cl_mem b=get_codec_f32(cl_ctx,wp+".bias");
    if(!K||!b){NNOPT_ERROR_FMT("convT: missing %s",wp.c_str());return nullptr;}
    const std::vector<int>& ks=get_codec_shape(cl_ctx, wp+".kernel");
    if (ks.size() < 3) { NNOPT_ERROR_FMT("conv2dT: no shape for %s.kernel in the codec metadata", wp.c_str()); return nullptr; }
    const int Cout=ks[0],kH=ks[1],kW=ks[2];
    // freq 'same' transpose pad = (kW-sw)/2; time 'causal' transpose pad = 0.
    const int Tout=T*sh, Fout=F*sw, padT=0, padF=(kW-sw)/2;
    // ── streaming context for the TIME UPSAMPLERS ────────────────────────────────────────────────
    // These are l4 and the transpose inside all six cascade blocks — every convolution that grows
    // the time axis — and they had no cross-chunk state at all. The plain-conv branch has kept
    // context since it was written; this one never called codec_stream_extend, so the frames that
    // actually stretch a chunk in time always restarted from zero padding.
    //
    // Same left-context mechanism as the plain conv, with one extra step: a transpose conv turns
    // `want` prepended input frames into `want*sh` extra OUTPUT frames at the front, which belong to
    // the previous chunk and must be trimmed. Without that trim the output silently shifts.
    //
    // The reference does this differently — sequence_layers keeps an overlap-add accumulator on the
    // un-emitted tail of the OUTPUT instead of a buffer of past INPUT (see BENCHMARK/architecture
    // notes). The two are equivalent for a causal transpose conv; this one reuses machinery the port
    // already has. `codecchunk` is the arbiter: chunked and whole must agree.
    int Tin_eff = T; cl_mem in_eff = in, in_ext = nullptr; int cut = 0;
    if (codec_stream_enabled() && kH > 1) {
        int Te = 0;
        in_ext = codec_stream_extend(cl_ctx, queue, wp, in, B, T, F, Cin, kH-1, &Te);
        if (in_ext) { in_eff = in_ext; Tin_eff = Te; cut = (kH-1)*sh; }
    }
    const int Tout_full = Tin_eff*sh;
    cl_int err=CL_SUCCESS; (void)err;
    cl_mem out=nullptr;
    const bool ident = (kH==1&&kW==1&&sh==1&&sw==1&&padT==0&&padF==0); (void)ident;
    // Direct (col-buffer-free) transpose conv: kernel fuses ELU + handles batch. Non-CLBlast build
    // always; CLBlast build under NNOPT_DIRECTCONV.
#ifdef USE_CLBLAST
    if (direct_conv_enabled()) {
#endif
        out = conv_direct(cl_ctx,queue,in_eff,K,b,B,Tin_eff,F,Cin,Tout_full,Fout,Cout,kH,kW,sh,sw,padT,padF,
                          /*transpose=*/true, /*apply_elu=*/apply_elu);
        if(!out) { if (in_ext) pool_free(in_ext); return nullptr; }
#ifdef USE_CLBLAST
    } else {
        // OPT #7: fuse the pre-convT ELU into im2col.
        const bool fuse = apply_elu && fuse_elu_enabled() && !ident;
        cl_mem cin = in_eff;
        if (apply_elu && !fuse) cin = elu_op(cl_ctx,queue,in_eff,B*Tin_eff*F*Cin);
        static cl_kernel imkT=nullptr;
        if(!imkT){ cl_program p=cl_ctx.build_program_from_file("kernels/im2col_conv2dT_f32.cl"); imkT=clCreateKernel(p,"im2col_conv2dT_f32",&err); g_im2col_transposed=imkT; }
        out=conv_clblast(cl_ctx,queue,cin,K,b,B,Tin_eff,F,Cin,Tout_full,Fout,Cout,kH,kW,sh,sw,padT,padF,imkT,fuse);
        if (cin != in_eff) pool_free(cin);
        if(!out) { if (in_ext) pool_free(in_ext); return nullptr; }
    }
#endif
    if (in_ext) pool_free(in_ext);
    if (cut) {
        cl_mem trimmed = trim_front_time(cl_ctx, queue, out, B, Tout_full, cut, Fout, Cout);
        if (!trimmed) { NNOPT_ERROR("convT: front trim failed"); pool_free(out); return nullptr; }
        pool_free(out); out = trimmed;
    }
    if(Tout_o)*Tout_o=Tout; if(Fout_o)*Fout_o=Fout; if(Cout_o)*Cout_o=Cout; return out;
}

static cl_mem upsample_time_op(OpenCLContext& cl_ctx, cl_command_queue queue, cl_mem in, int T, int F, int C, int r) {
    cl_int err=CL_SUCCESS; static cl_kernel uk=nullptr;
    if(!uk){ cl_program p=cl_ctx.build_program_from_file("kernels/upsample_time_f32.cl"); uk=clCreateKernel(p,"upsample_time_f32",&err); }
    cl_mem out=pool_alloc(cl_ctx.context(),CL_MEM_READ_WRITE,(size_t)T*r*F*C*sizeof(float),nullptr,&err);
    clSetKernelArg(uk,0,sizeof(cl_mem),&in); clSetKernelArg(uk,1,sizeof(cl_mem),&out);
    clSetKernelArg(uk,2,sizeof(int),&T); clSetKernelArg(uk,3,sizeof(int),&F); clSetKernelArg(uk,4,sizeof(int),&C); clSetKernelArg(uk,5,sizeof(int),&r);
    size_t g[3]={(size_t)C,(size_t)F,(size_t)T*r}; cl_ctx.profEnqueue(uk,3,g,nullptr,"upsample"); 
    return out;
}

static cl_mem upsample2d_op(OpenCLContext& cl_ctx, cl_command_queue queue, cl_mem in, int T, int F, int C, int rt, int rf, int B=1) {
    cl_int err=CL_SUCCESS; static cl_kernel uk=nullptr;
    if(!uk){ cl_program p=cl_ctx.build_program_from_file("kernels/upsample2d_f32.cl"); uk=clCreateKernel(p,"upsample2d_f32",&err); }
    cl_mem out=pool_alloc(cl_ctx.context(),CL_MEM_READ_WRITE,(size_t)B*T*rt*F*rf*C*sizeof(float),nullptr,&err);
    clSetKernelArg(uk,0,sizeof(cl_mem),&in); clSetKernelArg(uk,1,sizeof(cl_mem),&out);
    clSetKernelArg(uk,2,sizeof(int),&B); clSetKernelArg(uk,3,sizeof(int),&T); clSetKernelArg(uk,4,sizeof(int),&F); clSetKernelArg(uk,5,sizeof(int),&C);
    clSetKernelArg(uk,6,sizeof(int),&rt); clSetKernelArg(uk,7,sizeof(int),&rf);
    size_t g[3]={(size_t)C,(size_t)F*rf,(size_t)B*T*rt}; cl_ctx.profEnqueue(uk,3,g,nullptr,"upsample");
    return out;
}

// Diagnostic (NNOPT_CODECMAG=1): read a codec activation buffer and print its max |value| — used to
// find the fp16-safe boundary (fp16 max = 65504) for partial-fp16 (HGEMM) codec acceleration.
static void dbg_maxabs(OpenCLContext& cl_ctx, cl_command_queue queue, cl_mem buf, size_t n, const char* name) {
    static int on=-1, on_epoch=-1;
    if (on_epoch != nnopt_toggle_epoch()) { on = std::getenv("NNOPT_CODECMAG")?1:0; on_epoch = nnopt_toggle_epoch(); }
    if(!on || !buf) return;
    std::vector<float> h(n); clEnqueueReadBuffer(queue,buf,CL_TRUE,0,n*sizeof(float),h.data(),0,nullptr,nullptr);
    float m=0.0f; for(float v : h) { float a=v<0?-v:v; if(a>m) m=a; }
    std::printf("  [codecmag] %-22s max|x|=%.1f %s\n", name, m, m>65504.0f?"<-- OVERFLOWS fp16":(m>32000.0f?"(fp16 tight)":"(fp16 OK)"));
    std::fflush(stdout);
}

// ── fp16 codec conv: im2col(fp16, scaled) -> CLBlast HGEMM -> unscale to fp32 ──────────────────
// Returns nullptr (never a partial result) if anything is unavailable, so the caller can fall back.
// Every operand is divided by the same power-of-two scale and the output multiplied back; the bias
// rides inside the GEMM as CLBlast's C operand, so it is scaled too.
#ifdef USE_CLBLAST
static cl_mem conv_clblast_fp16(OpenCLContext& cl_ctx, cl_command_queue queue, cl_mem in,
                                cl_mem Wbuf, cl_mem biasbuf, int B, int T, int F, int Cin,
                                int Tout, int Fout, int Cout, int kH, int kW, int sh, int sw,
                                int p1, int p2, bool apply_elu, int M, int N, int K,
                                bool ident_1x1, bool transposed, bool* skipped) {
    cl_int err = CL_SUCCESS;
    // TWO gathers, exactly as the fp32 path has. A conv-transpose needs different index arithmetic
    // from a plain conv, and running the plain one over a transposed conv does not fail loudly — it
    // returns wrong data fast, because the bad indices fall out of bounds and the work is skipped.
    // That is what made every earlier fp16 render silent.
    static cl_kernel k_im2col=nullptr, k_im2colT=nullptr, k_bias=nullptr, k_wt=nullptr,
                     k_unscale=nullptr, k_absmax=nullptr, k_scale=nullptr;
    if (!k_im2col) {
        cl_program p1p = cl_ctx.build_program_from_file("kernels/im2col_conv2d_f16.cl");
        cl_program p2p = cl_ctx.build_program_from_file("kernels/codec_fp16_helpers.cl");
        if (!p1p || !p2p) { NNOPT_ERROR("codec fp16: kernel build failed"); return nullptr; }
        k_im2col  = clCreateKernel(p1p, "im2col_conv2d_f16", &err);
        k_im2colT = clCreateKernel(p1p, "im2col_conv2dT_f16", &err);
        k_scale   = clCreateKernel(p2p, "scale_from_partials", &err);
        k_bias    = clCreateKernel(p2p, "bias_prefill_f16", &err);
        k_wt      = clCreateKernel(p2p, "transpose_nk_to_kn_f16", &err);
        k_unscale = clCreateKernel(p2p, "unscale_f16_to_f32", &err);
        k_absmax  = clCreateKernel(p2p, "absmax_f32_partial", &err);
        if (!k_im2col || !k_im2colT || !k_bias || !k_wt || !k_unscale || !k_absmax || !k_scale) {
            NNOPT_ERROR("codec fp16: clCreateKernel failed"); return nullptr; }
        // The wave size, straight from the driver rather than assumed. A workgroup at or below this
        // runs as one wave, which is the difference between a free barrier and a real one — and the
        // leading theory for why CLBlast's 64-thread default beat every 128- and 256-thread
        // candidate. The guide says the value varies by tier (8/16/32/64/128), so guessing it is
        // exactly the mistake that produced round 1.
        size_t wave = 0;
        clGetKernelWorkGroupInfo(k_im2col, cl_ctx.device(),
                                 CL_KERNEL_PREFERRED_WORK_GROUP_SIZE_MULTIPLE,
                                 sizeof(wave), &wave, nullptr);
        std::fprintf(stderr, "WAVE_SIZE %zu\n", wave);
        std::fflush(stderr);
    }
    // The identity-1x1 shortcut feeds `in` straight to the GEMM as fp32; converting it would cost
    // the very copy that optimisation exists to avoid, so leave that case on the fp32 path. This is
    // NOT a failure — the caller must not report it as one.
    if (ident_1x1) { if (skipped) *skipped = true; return nullptr; }

    // ── pick the scale, on the GPU, without ever stopping the host ───────────────────────────
    // Two stages: many workgroups reduce |x| into partials, then one work item combines them and
    // writes {scale, 1/scale} into a buffer the gather, the bias prefill and the unscale all read.
    //
    // The previous version reduced the whole tensor in ONE workgroup and then blocked on
    // clEnqueueReadBuffer to bring a single float back so the host could compute the scale. On a
    // late cascade block that is ~19.6M floats reduced on one compute unit while the rest of the
    // GPU idles, followed by a full pipeline stall — 85 times per render.
    //
    // The scale CENTRES the block at 2^kTarget in fp16's exponent window instead of pinning its
    // maximum to 2^0. Mapping the peak to 1.0 wastes the fifteen octaves above it and pays at the
    // bottom, where small values fall subnormal and then to zero. Powers of two are exact in binary
    // floating point, so dividing in and multiplying back introduces no rounding of its own.
    static int kTarget = -1;
    if (kTarget < 0) {
        const char* t = std::getenv("NNOPT_CODECFP16_T");
        kTarget = t ? std::atoi(t) : 6;
        if (kTarget < 0 || kTarget > 14) kTarget = 6;
    }
    const size_t in_n = (size_t)B * T * F * Cin;
    const int kAbsGroups = 64;
    cl_mem part = pool_alloc(cl_ctx.context(), CL_MEM_READ_WRITE, kAbsGroups*sizeof(float), nullptr, &err);
    cl_mem sc   = pool_alloc(cl_ctx.context(), CL_MEM_READ_WRITE, 2*sizeof(float), nullptr, &err);
    if (!part || !sc) { if (part) pool_free(part); if (sc) pool_free(sc); return nullptr; }
    { int n_i = (int)in_n;
      clSetKernelArg(k_absmax,0,sizeof(cl_mem),&in); clSetKernelArg(k_absmax,1,sizeof(cl_mem),&part);
      clSetKernelArg(k_absmax,2,sizeof(int),&n_i);
      size_t g = (size_t)kAbsGroups*256, l = 256;
      if (cl_ctx.profEnqueue(k_absmax,1,&g,&l,"absmax") != CL_SUCCESS) { pool_free(part); pool_free(sc); return nullptr; } }
    { int ng = kAbsGroups;
      clSetKernelArg(k_scale,0,sizeof(cl_mem),&part); clSetKernelArg(k_scale,1,sizeof(cl_mem),&sc);
      clSetKernelArg(k_scale,2,sizeof(int),&ng);      clSetKernelArg(k_scale,3,sizeof(int),&kTarget);
      size_t g = 1;
      if (cl_ctx.profEnqueue(k_scale,1,&g,nullptr,"scale16") != CL_SUCCESS) { pool_free(part); pool_free(sc); return nullptr; } }
    pool_free(part);

    // ── weights: fp32 [N,K] -> fp16 [K,N], transposed once and cached ──
    static std::map<cl_mem, cl_mem> s_wt16;
    cl_mem Wh = nullptr;
    { auto it = s_wt16.find(Wbuf);
      if (it != s_wt16.end()) Wh = it->second;
      else {
        cl_mem w = clCreateBuffer(cl_ctx.context(), CL_MEM_READ_WRITE, (size_t)N*K*sizeof(cl_half), nullptr, &err);
        if (!w || err != CL_SUCCESS) return nullptr;
        clSetKernelArg(k_wt,0,sizeof(cl_mem),&Wbuf); clSetKernelArg(k_wt,1,sizeof(cl_mem),&w);
        clSetKernelArg(k_wt,2,sizeof(int),&N);       clSetKernelArg(k_wt,3,sizeof(int),&K);
        size_t tg[2]={(size_t)K,(size_t)N};
        if (cl_ctx.profEnqueue(k_wt,2,tg,nullptr,"wt16") != CL_SUCCESS) { clReleaseMemObject(w); return nullptr; }
        s_wt16[Wbuf] = w; Wh = w;
      } }

    // ── col (fp16, scaled) ──
    cl_mem colh = pool_alloc(cl_ctx.context(), CL_MEM_READ_WRITE, (size_t)M*K*sizeof(cl_half), nullptr, &err);
    if (!colh) { NNOPT_ERROR("codec fp16: col alloc"); pool_free(sc); return nullptr; }
    cl_kernel k_gather = transposed ? k_im2colT : k_im2col;
    { int ai=0;
      clSetKernelArg(k_gather,ai++,sizeof(cl_mem),&in);   clSetKernelArg(k_gather,ai++,sizeof(cl_mem),&colh);
      clSetKernelArg(k_gather,ai++,sizeof(int),&B);
      clSetKernelArg(k_gather,ai++,sizeof(int),&T);       clSetKernelArg(k_gather,ai++,sizeof(int),&F);
      clSetKernelArg(k_gather,ai++,sizeof(int),&Cin);
      clSetKernelArg(k_gather,ai++,sizeof(int),&Tout);    clSetKernelArg(k_gather,ai++,sizeof(int),&Fout);
      clSetKernelArg(k_gather,ai++,sizeof(int),&kH);      clSetKernelArg(k_gather,ai++,sizeof(int),&kW);
      clSetKernelArg(k_gather,ai++,sizeof(int),&sh);      clSetKernelArg(k_gather,ai++,sizeof(int),&sw);
      clSetKernelArg(k_gather,ai++,sizeof(int),&p1);      clSetKernelArg(k_gather,ai++,sizeof(int),&p2);
      const int elu_flag = apply_elu ? 1 : 0;
      clSetKernelArg(k_gather,ai++,sizeof(int),&elu_flag);
      clSetKernelArg(k_gather,ai++,sizeof(cl_mem),&sc);
      size_t ig[2]={(size_t)K,(size_t)M};
      if (cl_ctx.profEnqueue(k_gather,2,ig,nullptr,transposed?"im2colT16":"im2col16") != CL_SUCCESS) { pool_free(colh); pool_free(sc); return nullptr; } }

    // ── C prefilled with bias/scale, HGEMM accumulates into it ──
    cl_mem ch = pool_alloc(cl_ctx.context(), CL_MEM_READ_WRITE, (size_t)M*N*sizeof(cl_half), nullptr, &err);
    if (!ch) { pool_free(colh); pool_free(sc); return nullptr; }
    { clSetKernelArg(k_bias,0,sizeof(cl_mem),&biasbuf); clSetKernelArg(k_bias,1,sizeof(cl_mem),&ch);
      clSetKernelArg(k_bias,2,sizeof(int),&M);          clSetKernelArg(k_bias,3,sizeof(int),&N);
      clSetKernelArg(k_bias,4,sizeof(cl_mem),&sc);
      size_t bg=(size_t)M*N;
      if (cl_ctx.profEnqueue(k_bias,1,&bg,nullptr,"bias16") != CL_SUCCESS) { pool_free(colh); pool_free(ch); pool_free(sc); return nullptr; } }

    // PHASE 3 probe. NNOPT_CODECFP16_NOGEMM=1 runs the whole fp16 path — absmax, scale, gather, bias
    // prefill, unscale — and SKIPS ONLY the matrix multiply. The audio is meaningless; the point is
    // that codec_sec then measures everything-except-the-GEMM, and the difference against a normal
    // run is the GEMM's share. Timing it any other way needs a profiling-enabled queue, which
    // distorts the wall it is trying to measure.
    static int nogemm = -1, nogemm_epoch = -1;
    if (nogemm_epoch != nnopt_toggle_epoch()) {
        const char* e3 = std::getenv("NNOPT_CODECFP16_NOGEMM");
        nogemm = (e3 && e3[0] != '0') ? 1 : 0;
        nogemm_epoch = nnopt_toggle_epoch();
    }
    cl_command_queue q = queue;
    cl_event gev = nullptr;
    const bool want_ev = OpenCLContext::profWantEvent();
    const cl_half one = 0x3C00;   // 1.0 in IEEE binary16
    CLBlastStatusCode st = nogemm ? CLBlastSuccess
      : CLBlastHgemm(CLBlastLayoutRowMajor, CLBlastTransposeNo, CLBlastTransposeNo,
        (size_t)M,(size_t)N,(size_t)K, one, colh,0,(size_t)K, Wh,0,(size_t)N, one, ch,0,(size_t)N, &q,
        want_ev ? &gev : nullptr);
    if (want_ev && gev) OpenCLContext::profEventAdd("hgemm", gev);
    pool_free(colh);
    if (st != CLBlastSuccess) { NNOPT_ERROR_FMT("codec fp16: CLBlastHgemm st=%d", (int)st); pool_free(ch); pool_free(sc); return nullptr; }

    // ── back to fp32, undoing the scale ──
    cl_mem out = pool_alloc(cl_ctx.context(), CL_MEM_READ_WRITE, (size_t)M*N*sizeof(float), nullptr, &err);
    if (!out) { pool_free(ch); pool_free(sc); return nullptr; }
    { int n_i = M*N;
      clSetKernelArg(k_unscale,0,sizeof(cl_mem),&ch); clSetKernelArg(k_unscale,1,sizeof(cl_mem),&out);
      clSetKernelArg(k_unscale,2,sizeof(int),&n_i);   clSetKernelArg(k_unscale,3,sizeof(cl_mem),&sc);
      size_t ug=(size_t)n_i;
      if (cl_ctx.profEnqueue(k_unscale,1,&ug,nullptr,"unscale16") != CL_SUCCESS) { pool_free(ch); pool_free(out); pool_free(sc); return nullptr; } }
    pool_free(ch); pool_free(sc);
    return out;
}
#endif

// ── general transpose-residual block: out = conv3x3(Elu(convT(Elu(x)))) + shortcut(x).
// shortcut = upsample2d(conv1x1(x)) if a shortcut conv exists, else upsample2d(x). dims via pointers. ──
static cl_mem transpose_res_block(OpenCLContext& cl_ctx, Weights& weights, cl_command_queue queue,
                                  cl_mem x, int T, int F, int C, const std::string& wp, int sh, int sw,
                                  int* To, int* Fo, int* Co, int B=1) {
    // OPT #7: ELU fused into the consuming convT/conv im2col (shortcut still reads raw x).
    int Tt,Ft,Ct;
    cl_mem t0=conv2dtranspose_op(cl_ctx,weights,queue,x,T,F,C,wp+".body.layers.0.layers.1.inner",sh,sw,&Tt,&Ft,&Ct,B,/*apply_elu=*/true);
    if(!t0) return nullptr;
    int Tb,Fb,Cb;
    cl_mem body=conv2d_op(cl_ctx,weights,queue,t0,Tt,Ft,Ct,wp+".body.layers.1.layers.1.inner",1,1,&Tb,&Fb,&Cb,B,/*apply_elu=*/true); pool_free(t0);
    cl_mem sc;
    if (has_codec_tensor(cl_ctx, wp+".shortcut.layers.0.layers.0.inner.kernel")) {
        int Ts,Fs,Cs;
        cl_mem scc=conv2d_op(cl_ctx,weights,queue,x,T,F,C,wp+".shortcut.layers.0.layers.0.inner",1,1,&Ts,&Fs,&Cs,B);
        sc=upsample2d_op(cl_ctx,queue,scc,Ts,Fs,Cs,sh,sw,B); pool_free(scc);
    } else {
        sc=upsample2d_op(cl_ctx,queue,x,T,F,C,sh,sw,B);
    }
    cl_mem out=residual_add(cl_ctx,queue,body,sc,B*Tb*Fb*Cb); pool_free(body); pool_free(sc);
    if(To)*To=Tb; if(Fo)*Fo=Fb; if(Co)*Co=Cb; return out;
}

// ── inverse STFT (on device): dec [T,Nbins=480,4] → waveform [T*hop,2].
// IRFFT(960) + Hann synthesis window per frame/channel, then windowed overlap-add (hop=480). ──
static cl_mem run_istft(OpenCLContext& cl_ctx, Weights& weights, cl_command_queue queue,
                        cl_mem dec, int T, int Nbins, int Cdec) {
    (void)weights;
    const int Nfft=960, hop=480, Cout_wav=2;
    // Synthesis window [960]. Read ONCE and kept on the device: this used to fopen+fread the file
    // and allocate a fresh device buffer on every call, i.e. 20 file reads per render sitting on
    // the codec's hot path for a constant that never changes.
    cl_int err=CL_SUCCESS;
    static cl_mem winb = nullptr;
    if (!winb) {
        std::vector<float> win(Nfft);
        FILE* wf=std::fopen("weights/istft_window.bin","rb");
        if (wf){ size_t rd=std::fread(win.data(),sizeof(float),win.size(),wf); std::fclose(wf); (void)rd; }
        else { NNOPT_ERROR("istft: missing weights/istft_window.bin"); return nullptr; }
        winb=clCreateBuffer(cl_ctx.context(),CL_MEM_READ_ONLY|CL_MEM_COPY_HOST_PTR,
                            win.size()*sizeof(float),win.data(),&err);
        if(!winb){ NNOPT_ERROR("istft: window buffer"); return nullptr; }
    }
    static cl_kernel fk=nullptr, ok=nullptr;
    if(!fk){ cl_program p=cl_ctx.build_program_from_file("kernels/irfft_window_f32.cl"); fk=clCreateKernel(p,"irfft_window_f32",&err); }
    if(!ok){ cl_program p=cl_ctx.build_program_from_file("kernels/overlap_add_f32.cl"); ok=clCreateKernel(p,"overlap_add_f32",&err); }
    cl_mem frames=pool_alloc(cl_ctx.context(),CL_MEM_READ_WRITE,(size_t)T*Nfft*2*sizeof(float),nullptr,&err);
    clSetKernelArg(fk,0,sizeof(cl_mem),&dec); clSetKernelArg(fk,1,sizeof(cl_mem),&winb); clSetKernelArg(fk,2,sizeof(cl_mem),&frames);
    clSetKernelArg(fk,3,sizeof(int),&T); clSetKernelArg(fk,4,sizeof(int),&Nbins); clSetKernelArg(fk,5,sizeof(int),&Cdec); clSetKernelArg(fk,6,sizeof(int),&Nfft);
    size_t gf[3]={(size_t)Nfft,2,(size_t)T};
    if(cl_ctx.profEnqueue(fk,3,gf,nullptr,"irfft")!=CL_SUCCESS){NNOPT_ERROR("istft: irfft enqueue");pool_free(frames);return nullptr;}
    cl_mem out=pool_alloc(cl_ctx.context(),CL_MEM_READ_WRITE,(size_t)T*hop*Cout_wav*sizeof(float),nullptr,&err);
    // ── the overlap tail that joins consecutive chunks ───────────────────────────────────────────
    // 480 samples x 2 channels, ping-ponged between two buffers. The reference keeps exactly this
    // as InverseSTFT layer state: _buffer_width = max(0, frame_length - frame_step) = 960 - 480.
    //
    // Two buffers, not one: tail_in and tail_out are separate kernel parameters and one is const, so
    // the compiler may assume they do not alias and hoist the store above the load — at which point
    // a chunk adds its own fresh tail to its own opening samples.
    const int lap = Nfft - hop;
    static cl_mem tailb[2] = {nullptr, nullptr};
    static int tail_cur = 0;
    for (int i = 0; i < 2; ++i) if (!tailb[i]) {
        tailb[i] = clCreateBuffer(cl_ctx.context(), CL_MEM_READ_WRITE,
                                  (size_t)lap*Cout_wav*sizeof(float), nullptr, &err);
        if (!tailb[i]) { NNOPT_ERROR("istft: tail buffer"); pool_free(frames); pool_free(out); return nullptr; }
    }
    cl_mem t_in = tailb[tail_cur], t_out = tailb[tail_cur ^ 1];
    tail_cur ^= 1;
    // Tied to the conv state, not independent. With stateless convs the two half-windows being summed
    // come from separate decodes and adding them at full amplitude is LOUDER and worse than the notch
    // it replaces — that was measured on device earlier today. These two are one feature.
    const int use_tail = (g_istft_tail_valid && codec_stream_enabled()) ? 1 : 0;
    clSetKernelArg(ok,0,sizeof(cl_mem),&frames); clSetKernelArg(ok,1,sizeof(cl_mem),&out);
    clSetKernelArg(ok,2,sizeof(int),&T); clSetKernelArg(ok,3,sizeof(int),&Nfft); clSetKernelArg(ok,4,sizeof(int),&hop);
    clSetKernelArg(ok,5,sizeof(cl_mem),&t_in); clSetKernelArg(ok,6,sizeof(cl_mem),&t_out);
    clSetKernelArg(ok,7,sizeof(int),&use_tail);
    g_istft_tail_valid = true;
    size_t go[2]={(size_t)T*hop,2};
    if(cl_ctx.profEnqueue(ok,2,go,nullptr,"ola")!=CL_SUCCESS){NNOPT_ERROR("istft: ola enqueue");pool_free(out);pool_free(frames);return nullptr;}
    
    pool_free(frames);
    return out;
}

// ── decoder layer 5 (ParallelChannels). 2 groups share weights → run the cascade ONCE as a batch B=2,
// so every conv is a single CLBlast GEMM over 2× the rows (vs two small GEMMs). GPU channel split/concat
// (no host round-trips). [Tin,5,1024] → [2*Tin,480,4]. ──
static cl_mem run_codec_l5(OpenCLContext& cl_ctx, Weights& weights, cl_command_queue queue, cl_mem x, int Tin,
                           int* Tout_o=nullptr) {
    const std::string CH="spectrostream.embeddings_to_waveform_layer.layers.0.layers.5.child.layers.";
    const int Fin=5,G=2,Cg=512;
    const int strides[6][2]={{2,2},{1,2},{1,2},{1,3},{1,2},{1,2}};
    cl_int e=CL_SUCCESS;
    static cl_kernel splk=nullptr, conk=nullptr;
    if(!splk){ cl_program p=cl_ctx.build_program_from_file("kernels/split_concat_ch_f32.cl"); splk=clCreateKernel(p,"split_ch_to_batch",&e); conk=clCreateKernel(p,"concat_batch_to_ch",&e); }
    // split x[Tin,5,1024] → xg[B=2, Tin, 5, 512]
    cl_mem xg=pool_alloc(cl_ctx.context(),CL_MEM_READ_WRITE,(size_t)G*Tin*Fin*Cg*sizeof(float),nullptr,&e);
    clSetKernelArg(splk,0,sizeof(cl_mem),&x); clSetKernelArg(splk,1,sizeof(cl_mem),&xg);
    clSetKernelArg(splk,2,sizeof(int),&Tin); clSetKernelArg(splk,3,sizeof(int),&Fin); clSetKernelArg(splk,4,sizeof(int),&G); clSetKernelArg(splk,5,sizeof(int),&Cg);
    size_t sg[3]={(size_t)Cg,(size_t)Fin,(size_t)G*Tin}; cl_ctx.profEnqueue(splk,3,sg,nullptr,"split");
    cl_mem y=xg; int T=Tin,F=Fin,C=Cg;
    // block 0 — the ONLY time-upsampling block (time stride 2): Tin → 2*Tin.
    { int To,Fo,Co;
        cl_mem yn=transpose_res_block(cl_ctx,weights,queue,y,T,F,C,CH+"0",strides[0][0],strides[0][1],&To,&Fo,&Co,G);
        pool_free(y); y=yn; T=To;F=Fo;C=Co; if(!y){NNOPT_ERROR("l5 block 0");return nullptr;}
        dbg_maxabs(cl_ctx,queue,y,(size_t)G*T*F*C,"l5.block0"); }
    // OPT (cropped cascade): the l6 crop keeps only the last `Tin` frames (= 2*T_chunk). Blocks 1-5
    // + child6 are time-stride-1 with a 26-frame left receptive field, so drop the leading frames
    // beyond [keptStart - MARGIN, T): the kept tail stays bit-exact, late large-F blocks do ~half work.
    if (crop_l5_enabled()) {
        // MARGIN is how many frames of left context blocks 1-5 + child6 keep beyond what l6 will
        // retain. 28 is the full receptive field (5x4 + child6's 6 = 26, plus slack) and is exact,
        // but on a 5-frame chunk there are only 10 leading frames, so the crop can never trigger and
        // this optimisation has been dead in every run. NNOPT_CROPMARGIN exposes it: smaller margin
        // = less of the cascade recomputed = faster, with the receptive-field truncation showing up
        // as a cosine drop. Cropping HERE (after block 0) is strictly better than cropping l5's
        // input, because block 0 still sees its full context either way.
        int MARGIN = 28;
        if (const char* mm = std::getenv("NNOPT_CROPMARGIN")) { MARGIN = std::atoi(mm); if (MARGIN < 0) MARGIN = 0; }
        const int keptLen=Tin;                      // final kept frames = last Tin of T(=2*Tin)
        const int keepStart=(T-keptLen)-MARGIN;
        if (keepStart>0) {
            const int newT=T-keepStart;
            cl_mem yc=pool_alloc(cl_ctx.context(),CL_MEM_READ_WRITE,(size_t)G*newT*F*C*sizeof(float),nullptr,&e);
            for (int g=0; g<G; g++)
                clEnqueueCopyBuffer(queue,y,yc,(size_t)(g*T+keepStart)*F*C*sizeof(float),
                                    (size_t)g*newT*F*C*sizeof(float),(size_t)newT*F*C*sizeof(float),0,nullptr,nullptr);
            pool_free(y); y=yc; T=newT;
        }
    }
    for (int i=1;i<6;i++){ int To,Fo,Co;
        cl_mem yn=transpose_res_block(cl_ctx,weights,queue,y,T,F,C,CH+std::to_string(i),strides[i][0],strides[i][1],&To,&Fo,&Co,G);
        pool_free(y); y=yn; T=To;F=Fo;C=Co; if(!y){NNOPT_ERROR_FMT("l5 block %d",i);return nullptr;}
        char nm[24]; std::snprintf(nm,sizeof(nm),"l5.block%d",i); dbg_maxabs(cl_ctx,queue,y,(size_t)G*T*F*C,nm); }
    // child[6]: Elu → conv 7x7 (C→2), batched — ELU fused into the conv im2col (OPT #7).
    int Tf,Ff,Cf;
    cl_mem fin=conv2d_op(cl_ctx,weights,queue,y,T,F,C,CH+"6.layers.1.layers.0.inner",1,1,&Tf,&Ff,&Cf,G,/*apply_elu=*/true); pool_free(y);
    // concat fin[2,Tf,Ff,Cf] → out[Tf,Ff,2*Cf]
    cl_mem out=pool_alloc(cl_ctx.context(),CL_MEM_READ_WRITE,(size_t)Tf*Ff*G*Cf*sizeof(float),nullptr,&e);
    clSetKernelArg(conk,0,sizeof(cl_mem),&fin); clSetKernelArg(conk,1,sizeof(cl_mem),&out);
    clSetKernelArg(conk,2,sizeof(int),&Tf); clSetKernelArg(conk,3,sizeof(int),&Ff); clSetKernelArg(conk,4,sizeof(int),&G); clSetKernelArg(conk,5,sizeof(int),&Cf);
    size_t cg[3]={(size_t)Cf,(size_t)Ff,(size_t)Tf}; cl_ctx.profEnqueue(conk,3,cg,nullptr,"concat");
    pool_free(fin);
    if (Tout_o) *Tout_o = Tf;   // actual l5 output time (= 4*T_chunk normally, or 2*T_chunk+MARGIN when cropped)
    return out;
}

// ── decoder layer 4 (Residual, conv-transpose time 2→4, 512→1024) ──
// body = conv3x3(Elu(convT(Elu(x)))); shortcut = upsample_time(conv1x1(x)); out = body + shortcut.
static cl_mem run_codec_l4(OpenCLContext& cl_ctx, Weights& weights, cl_command_queue queue, cl_mem x, int T, int F, int C) {
    const std::string P="spectrostream.embeddings_to_waveform_layer.layers.0.layers.4.";
    // OPT #7: ELU fused into the consuming convT/conv im2col (shortcut still reads raw x).
    int Tt,Ft,Ct;
    cl_mem t0=conv2dtranspose_op(cl_ctx,weights,queue,x,T,F,C,P+"body.layers.0.layers.1.inner",2,1,&Tt,&Ft,&Ct,1,/*apply_elu=*/true);
    int Tb,Fb,Cb;
    cl_mem body=conv2d_op(cl_ctx,weights,queue,t0,Tt,Ft,Ct,P+"body.layers.1.layers.1.inner",1,1,&Tb,&Fb,&Cb,1,/*apply_elu=*/true); pool_free(t0);
    int Ts,Fs,Cs;
    cl_mem sc_c=conv2d_op(cl_ctx,weights,queue,x,T,F,C,P+"shortcut.layers.0.layers.0.inner",1,1,&Ts,&Fs,&Cs);
    cl_mem sc=upsample_time_op(cl_ctx,queue,sc_c,Ts,Fs,Cs,2); pool_free(sc_c);
    cl_mem out=residual_add(cl_ctx,queue,body,sc,Tb*Fb*Cb); pool_free(body); pool_free(sc);
    return out;
}

// ── decoder layer 3 (Residual, 3x3, no dim change): out = x + conv2(Elu(conv1(Elu(x)))) ──
// Regular conv2d_residual_unit = pre-activated convs; shortcut = identity (dims match).
static cl_mem run_codec_l3(OpenCLContext& cl_ctx, Weights& weights, cl_command_queue queue, cl_mem x, int T, int F, int C) {
    const std::string P="spectrostream.embeddings_to_waveform_layer.layers.0.layers.3.body.";
    // OPT #7: ELU fused into each consuming conv im2col (residual still reads raw x).
    int Ta,Fa,Ca;
    cl_mem c0 = conv2d_op(cl_ctx,weights,queue,x,T,F,C,P+"layers.0.layers.1.inner",1,1,&Ta,&Fa,&Ca,1,/*apply_elu=*/true);
    cl_mem c1 = conv2d_op(cl_ctx,weights,queue,c0,Ta,Fa,Ca,P+"layers.1.layers.1.inner",1,1,&Ta,&Fa,&Ca,1,/*apply_elu=*/true); pool_free(c0);
    cl_mem out = residual_add(cl_ctx,queue,x,c1,T*F*C); pool_free(c1);
    return out;
}

// ── full SpectroStream codec decode: RVQ codes [T,12] → waveform [T*1920, 2] (on device). ──
// RVQ embed → l1(1x1) → reshape → l3(3x3) → l4(convT) → l5(cascade) → l6(crop last) → ISTFT.
static cl_mem run_codec_decode(OpenCLContext& cl_ctx, Weights& weights, cl_command_queue queue,
                               const std::vector<int>& codes, int T, int* nwav_o, cl_mem codes_gpu=nullptr) {
    const int NCB=12, CB=1024, dim=256;
    // The cascade's transpose-convs each need a few frames of input to produce a valid output
    // window; below kMinCodecFrames some intermediate dimension goes to zero and the run
    // segfaults deep inside a kernel launch. Every validated run of this port used 5-frame
    // chunks, so that is the floor we know holds. Fail here with a reason instead.
    if (T < kMinCodecFrames) {
        NNOPT_ERROR_FMT("run_codec_decode: %d frames is below the codec's %d-frame minimum",
                        T, kMinCodecFrames);
        return nullptr;
    }
    cl_int err=CL_SUCCESS;
    // codes_gpu (pipeline): already-unoffset RVQ codes [T*12] on the GPU — skip the host upload so the
    // codec can run on a 2nd queue overlapping the AR. Otherwise upload the host `codes`.
    const bool own_cb = (codes_gpu == nullptr);
    cl_mem cb;
    if (codes_gpu) { cb = codes_gpu; }
    else { std::vector<int32_t> ci(codes.begin(), codes.end());
           cb=pool_alloc(cl_ctx.context(),CL_MEM_READ_ONLY|CL_MEM_COPY_HOST_PTR,ci.size()*sizeof(int32_t),ci.data(),&err); }
    cl_mem tbl=weights.get_buffer("spectrostream.quantizer.embedding");
    cl_mem emb=pool_alloc(cl_ctx.context(),CL_MEM_READ_WRITE,(size_t)T*dim*sizeof(float),nullptr,&err);
    { static cl_kernel rk=nullptr; if(!rk){ cl_program p=cl_ctx.build_program_from_file("kernels/rvq_embed_f16.cl"); rk=clCreateKernel(p,"rvq_embed_f16",&err); }
      clSetKernelArg(rk,0,sizeof(cl_mem),&tbl); clSetKernelArg(rk,1,sizeof(cl_mem),&cb); clSetKernelArg(rk,2,sizeof(cl_mem),&emb);
      clSetKernelArg(rk,3,sizeof(int),&T); clSetKernelArg(rk,4,sizeof(int),&NCB); clSetKernelArg(rk,5,sizeof(int),&CB); clSetKernelArg(rk,6,sizeof(int),&dim);
      size_t g[2]={(size_t)T,(size_t)dim}; cl_ctx.profEnqueue(rk,2,g,nullptr,"rvq");  }
    if (own_cb) pool_free(cb);
    cl_ctx.compTic();
    NNOPT_CHECKPOINT("codec: rvq embed done, entering l1");
    cl_mem l1=run_codec_l1(cl_ctx,weights,queue,emb,T,1,256); pool_free(emb);     // [T,1,2560]
    cl_ctx.compAccum("codec_l1(1x1)"); dbg_maxabs(cl_ctx,queue,l1,(size_t)T*1*2560,"l1");
    NNOPT_CHECKPOINT("codec: l1 done, entering l3");
    cl_mem l3=run_codec_l3(cl_ctx,weights,queue,l1,T,5,512); pool_free(l1);        // reshape→[T,5,512]→[T,5,512]
    cl_ctx.compAccum("codec_l3(3x3)"); dbg_maxabs(cl_ctx,queue,l3,(size_t)T*5*512,"l3");
    NNOPT_CHECKPOINT("codec: l3 done, entering l4");
    cl_mem l4=run_codec_l4(cl_ctx,weights,queue,l3,T,5,512); pool_free(l3);        // [2T,5,1024]
    cl_ctx.compAccum("codec_l4(convT)"); dbg_maxabs(cl_ctx,queue,l4,(size_t)2*T*5*1024,"l4");
    NNOPT_CHECKPOINT("codec: l4 done, entering l5 cascade");
    // NNOPT_L5HALF=1 — probe: the cascade produces 4T time frames and l6 keeps only the LAST 2T,
    // so half of blocks 1-5 (the most expensive part of the codec) is computed and thrown away.
    // Feed l5 only the SECOND half of l4's output and it produces exactly the 2T frames that
    // survive. Whether that is equivalent depends on how far the discarded half reaches into the
    // kept one through the convs' left context — which is a measurement, not an argument, so this
    // is a toggle validated against the fp32 output rather than a change.
    static int l5half = -1, l5half_epoch = -1;
    if (l5half_epoch != nnopt_toggle_epoch()) {
        // Like NNOPT_CROPL5, this only holds while l6 discards l4's first half.
        // At full length that half IS the first 20 ms of every frame's audio.
        const char* e2 = std::getenv("NNOPT_L5HALF");
        l5half = (e2 && e2[0] != '0' && half_audio_enabled()) ? 1 : 0;
        l5half_epoch = nnopt_toggle_epoch();
    }
    // How many CONTEXT frames of l4's first half to keep. l6 keeps the last 2T of l5's output, and
    // those depend on l4's second half plus however far the convs' receptive field reaches back into
    // the first half. Feeding all of it (ctx=T, the original) is exact; feeding none (ctx=0) is
    // 1.9x faster at cosine 0.84. This exposes the knee between them: cost is (T+ctx)/2T of the
    // cascade, so ctx=2 is ~1.7x and ctx=4 ~1.4x on a 5-frame chunk.
    int l5ctx = 0;
    if (const char* c = std::getenv("NNOPT_L5CTX")) l5ctx = std::atoi(c);
    if (l5ctx < 0) l5ctx = 0;
    if (l5ctx > T) l5ctx = T;
    int T5=0; cl_mem l5=nullptr;
    if (l5half) {
        // l4 is [2T, 5, 1024]; hand l5 frames [T-ctx, 2T) — the surviving half plus `ctx` frames of
        // real left context, instead of the zeros it would otherwise convolve against.
        const size_t row = (size_t)5*1024*sizeof(float);
        const int Tin = T + l5ctx;
        cl_mem l4b = pool_alloc(cl_ctx.context(), CL_MEM_READ_WRITE, (size_t)Tin*row, nullptr, &err);
        if (l4b) {
            clEnqueueCopyBuffer(queue, l4, l4b, (size_t)(T - l5ctx)*row, 0, (size_t)Tin*row, 0, nullptr, nullptr);
            l5 = run_codec_l5(cl_ctx,weights,queue,l4b,Tin,&T5);
            pool_free(l4b);
        }
    }
    if (!l5) l5=run_codec_l5(cl_ctx,weights,queue,l4,2*T,&T5);
    pool_free(l4);   // [T5,480,4], T5=4T (or 2T when halved)
    cl_ctx.compAccum("codec_l5(cascade)");
    if (std::getenv("NNOPT_DUMP_CHAIN")) {
        std::vector<float> d5((size_t)T5*480*4);
        clEnqueueReadBuffer(queue,l5,CL_TRUE,0,d5.size()*sizeof(float),d5.data(),0,nullptr,nullptr);
        FILE* f=std::fopen("layer_dumps/chain_l5.bin","wb"); if(f){std::fwrite(d5.data(),sizeof(float),d5.size(),f);std::fclose(f);} }
    // l6: every frame l5 produced is audio (T5 = 4T = 40 ms per RVQ frame — see
    // half_audio_enabled()), so the whole buffer goes to the iSTFT untouched.
    // Under NNOPT_HALFAUDIO=1 the historical front-crop to the last 2T is restored.
    NNOPT_CHECKPOINT("codec: l5 done, entering l6");
    const int FC=480*4;
    const int T6 = half_audio_enabled() ? 2*T : T5;
    cl_mem l6;
    if (T6 == T5) {
        l6 = l5;                       // no crop: hand the cascade output straight on
    } else {
        l6=pool_alloc(cl_ctx.context(),CL_MEM_READ_WRITE,(size_t)T6*FC*sizeof(float),nullptr,&err);
        clEnqueueCopyBuffer(queue,l5,l6,(size_t)(T5-T6)*FC*sizeof(float),0,(size_t)T6*FC*sizeof(float),0,nullptr,nullptr);
        pool_free(l5);
    }
    cl_ctx.compAccum("codec_l6(crop)");
    NNOPT_CHECKPOINT("codec: l6 done, entering istft");
    cl_mem wav=run_istft(cl_ctx,weights,queue,l6,T6,480,4); pool_free(l6);          // [T6*480, 2]
    cl_ctx.compAccum("istft");
    // Every static above is now built. Release order matters: the worker thread that
    // later reads this flag must see the statics, not just the bool.
    if (wav) g_codec_statics_warm.store(true, std::memory_order_release);
    if(nwav_o)*nwav_o=T6*480;
    return wav;
}

// ── streaming state: the temporal KV cache + the next frame's embedding ──
// Carried across run_song() calls so chunk N+1 continues chunk N instead of
// restarting from the style embedding. MAXF caps the cache; past it the session
// resets (loudly) rather than indexing off the end — see kv-cache clamp rule.
static const int kARMaxFrames = 512;   // 512 frames = 20.48 s of attention history (40 ms/frame)
struct SongSession {
    cl_mem kc[12] = {nullptr}, vc[12] = {nullptr};
    cl_mem ti     = nullptr;   // temporal input for the NEXT frame [1024]
    int    pos    = 0;         // absolute frame index (attention position)
    bool   alive  = false;
};

SongSession* song_session_create() { return new SongSession(); }

static void song_session_release(SongSession* s) {
    if (!s) return;
    for (int k = 0; k < 12; ++k) {
        if (s->kc[k]) { pool_free(s->kc[k]); s->kc[k] = nullptr; }
        if (s->vc[k]) { pool_free(s->vc[k]); s->vc[k] = nullptr; }
    }
    if (s->ti) { pool_free(s->ti); s->ti = nullptr; }
    s->pos = 0;
    s->alive = false;
}

void song_session_destroy(SongSession* s) { song_session_release(s); delete s; }

// ── full autoregressive generation: temporal_input_0 + source → token grid [n_frames][12] ──
// Greedy (argmax-per-codebook). Temporal self-attn caches across frames (pos=frame). Returns flattened
// [n_frames*12] int tokens. `sess` (optional) carries the cache across calls; when null the state is
// allocated, used and freed here — the original one-shot behaviour, bit-identical.
static std::vector<int> run_generate(OpenCLContext& cl_ctx, Weights& weights, cl_command_queue queue,
                                     cl_mem temporal_input_0, cl_mem source, int n_frames,
                                     SongSession* sess = nullptr) {
    const int dim = 1024, MAXF = kARMaxFrames, NCB = 12;
    // OVERLAP/deep-queue: the AR is sequential, but the GPU was idle ~½ the AR wall because every
    // frame ended in a blocking tokbuf readback (drain → host turnaround → GPU stalls). Keep tokens
    // ON GPU (write each frame's 12 into grid_gpu, feed the next frame via frame_embed_from_tokbuf),
    // read the whole grid back ONCE. clFinish only every ARFLUSH frames — bounds the deferred-free
    // memory lookahead while letting the host run ahead and fill the per-op dispatch bubbles.
    // NNOPT_ARFLUSH=1 reproduces the old sync-every-frame behavior (A/B); 0 = sync only at the end.
    int flush_every = 8;
    if (const char* fe = std::getenv("NNOPT_ARFLUSH")) flush_every = std::atoi(fe);
    cl_int e=CL_SUCCESS;
    SongSession local;                       // one-shot state when the caller has no session
    SongSession* st = sess ? sess : &local;
    // ── the cache fills every 20.48 s; COMPACT it, do not throw the piece away ──────────────────
    // This used to release the session, which restarted the AR from the style embedding. In live
    // mode that lands on chunk 11 — 10 chunks x 50 frames = 500, and 500 + 50 > 512 — so a stream
    // simply became a different piece of music every twenty seconds, mid-playback.
    //
    // It never had to. Temporal self-attention windows to max_past = 41 frames
    // (ops/CachedSelfAttention.cpp:113), so nothing ever reads further back than that; the 512-frame
    // cache exists only because `pos` indexes it absolutely. Slide the last kKeepFrames entries down
    // to the front, set pos to match, and the piece continues with its attention history intact.
    //
    // Safe because this attention applies NO rotary encoding — the comment at the top of
    // CachedSelfAttention.cpp is explicit that k/v go into cache[pos] unmodified — so the cached
    // vectors do not depend on the absolute position they were written at, only on their order.
    // With rope this would be wrong: the relative offsets would silently shift.
    static const int kKeepFrames = 64;   // >= max_past(41) + 1, with margin; 4 MB of copies per fill
    if (st->alive && st->pos + n_frames > MAXF) {
        if (st->pos > 2 * kKeepFrames && n_frames + kKeepFrames <= MAXF) {
            const size_t row = (size_t)dim * sizeof(float);
            const int    src = st->pos - kKeepFrames;
            for (int k = 0; k < 12; ++k) {
                clEnqueueCopyBuffer(queue, st->kc[k], st->kc[k], (size_t)src*row, 0, (size_t)kKeepFrames*row, 0, nullptr, nullptr);
                clEnqueueCopyBuffer(queue, st->vc[k], st->vc[k], (size_t)src*row, 0, (size_t)kKeepFrames*row, 0, nullptr, nullptr);
            }
            st->pos = kKeepFrames;
            std::fprintf(stderr, "AR_CACHE_COMPACT kept=%d frames, pos=%d (piece continues)\n",
                         kKeepFrames, st->pos);
            std::fflush(stderr);
        } else {
            // Only reachable if a single request is itself longer than the cache, which is a caller
            // error rather than a streaming one. Restarting is still the honest answer there.
            NNOPT_ERROR_FMT("generate: request of %d frames cannot fit the %d-frame cache — restarting",
                            n_frames, MAXF);
            song_session_release(st);
        }
    }
    if (!st->alive) {
        for (int k=0;k<12;k++){ st->kc[k]=pool_alloc(cl_ctx.context(),CL_MEM_READ_WRITE,(size_t)MAXF*dim*sizeof(float),nullptr,nullptr);
                                st->vc[k]=pool_alloc(cl_ctx.context(),CL_MEM_READ_WRITE,(size_t)MAXF*dim*sizeof(float),nullptr,nullptr); }
        st->ti = pool_alloc(cl_ctx.context(),CL_MEM_READ_WRITE,(size_t)dim*sizeof(float),nullptr,nullptr);
        clEnqueueCopyBuffer(queue,temporal_input_0,st->ti,0,0,(size_t)dim*sizeof(float),0,nullptr,nullptr);
        st->pos = 0;
        st->alive = true;
    }
    cl_mem* kc = st->kc; cl_mem* vc = st->vc;
    cl_mem grid_gpu = pool_alloc(cl_ctx.context(),CL_MEM_READ_WRITE,(size_t)n_frames*NCB*sizeof(int),nullptr,&e);
    const int pos0 = st->pos;
    for (int f=0; f<n_frames; ++f) {
        cl_ctx.compTic();
        cl_mem tout = run_temporal_step(cl_ctx,weights,queue,st->ti,source,kc,vc,dim,pos0+f);
        cl_ctx.compAccum("temporal_body");
        if (!tout) { NNOPT_ERROR_FMT("generate: temporal frame %d failed", f); break; }
        cl_mem tokbuf = run_depth_loop_gpu(cl_ctx,weights,queue,tout,nullptr,pos0+f);
        cl_ctx.compAccum("depth_body");
        pool_free(tout);
        if (!tokbuf) { NNOPT_ERROR_FMT("generate: depth frame %d failed", f); break; }
        // stash this frame's 12 tokens into the GPU grid (async copy — no readback)
        clEnqueueCopyBuffer(queue,tokbuf,grid_gpu,0,(size_t)f*NCB*sizeof(int),(size_t)NCB*sizeof(int),0,nullptr,nullptr);
        // The last frame's embedding is only needed when the session continues into the next chunk.
        if (f < n_frames-1 || sess) { cl_mem nt = frame_embed_from_tokbuf(cl_ctx,weights,queue,tokbuf);
                                      pool_free(st->ti); st->ti = nt; cl_ctx.compAccum("frame_embed"); }
        pool_free(tokbuf);
        st->pos = pos0 + f + 1;
        if (flush_every > 0 && ((f+1) % flush_every == 0)) clFinish(queue);  // bound lookahead memory
        // Heartbeat. A render is otherwise silent for its whole wall, which reads as a hang to the
        // app's watchdog (and to a human staring at a device-farm screen).
        if (((f+1) % 10) == 0 || f == n_frames-1) {
            std::fprintf(stderr, "MRT_PROGRESS stage=ar frame=%d/%d\n", f+1, n_frames);
            std::fflush(stderr);
        }
    }
    std::vector<int32_t> gh((size_t)n_frames*NCB);
    clEnqueueReadBuffer(queue,grid_gpu,CL_TRUE,0,gh.size()*sizeof(int32_t),gh.data(),0,nullptr,nullptr);  // single readback
    pool_free(grid_gpu);
    if (!sess) song_session_release(st);      // one-shot: the state dies with the call
    return std::vector<int>(gh.begin(), gh.end());
}

// ── per-render GPU profiling: OFF unless asked for ─────────────────────────────────────────────
// AR frame 0 and the first codec chunk used to be profiled on EVERY render, with no flag. That
// costs three forced clFinish inside the render and ~519 live cl_events, and one of the drains
// stops queue 2 half-way through — which is precisely the AR/codec overlap this path exists to
// create. Instrumentation that changes the thing it measures is worse than no instrumentation:
// it is how the codec profile read "0 dispatches" this morning.
//
// NNOPT_ARPROF=1 (or NNOPT_PROFILE) turns it back on for a deliberate measuring run.
static bool arprof_enabled() {
#ifdef NNOPT_MEASURE_BUILD
    return true;   // measurement APK: profile frame 0 and the first codec chunk every render
#endif
    static int v = -1, ep = -1;
    if (ep != nnopt_toggle_epoch()) {
        const char* e = std::getenv("NNOPT_ARPROF");
        v = (e && e[0] != '0') ? 1 : (std::getenv("NNOPT_PROFILE") ? 1 : 0);
        ep = nnopt_toggle_epoch();
    }
    return v != 0;
}

// ── PIPELINE: overlap the (host-dispatch-bound) AR on queue1 with the (GPU-bound) codec on queue2 ──
// The AR leaves the GPU idle ~40% (tiny kernels waiting on host issue); the codec is GPU-heavy. Running
// them on two queues lets the codec's SGEMMs fill the AR's GPU bubbles. As each CHUNK of frames is
// generated, its tokens are un-offset to RVQ codes on-GPU and the codec for that chunk is launched on
// queue2 (gated by a cross-queue event), while the AR keeps generating on queue1. Returns interleaved
// stereo PCM (float). Bit-identical token path to run_generate (validated by cosine vs the sequential wav).
static std::vector<float> run_song_pipelined(OpenCLContext& cl_ctx, Weights& weights, cl_command_queue q1,
                                             cl_mem temporal_input_0, cl_mem source, int n_frames,
                                             int CHUNK, int* nwav_o,
                                             double* ar_ms_o=nullptr, double* codec_ms_o=nullptr,
                                             std::vector<int>* tokens_o=nullptr,
                                             SongSession* sess=nullptr) {
    const int dim=1024, MAXF=512, NCB=12;
    cl_int e=CL_SUCCESS;
    // PROFILING ON for the codec queue. The codec's GPU time has never been measured — q2 was
    // created with no profiling, which is why CODEC_CHUNK_GPU reported "0 dispatches", and the
    // codec is mostly CLBlast, which does not route through profEnqueue at all. So every
    // "GPU utilisation" figure so far counted only the AR's kernels while the codec ran beside them
    // on the same GPU. That is the difference between "the GPU is 39% busy" and "the GPU is the
    // wall", and every decision today depended on which it was.
    //
    // Cheap here in a way it is not on q1: a render issues ~48,900 AR dispatches against a few
    // thousand codec ones.
    cl_command_queue q2 = clCreateCommandQueue(cl_ctx.context(), cl_ctx.device(),
                                               CL_QUEUE_PROFILING_ENABLE, &e);
    if (!q2) { NNOPT_ERROR("pipeline: clCreateCommandQueue(q2) failed"); return {}; }
    static cl_kernel uok=nullptr;
    if(!uok){ cl_program p=cl_ctx.build_program_from_file("kernels/rvq_unoffset.cl"); if(p) uok=clCreateKernel(p,"rvq_unoffset",&e); }
    if(!uok){ NNOPT_ERROR("pipeline: rvq_unoffset build failed"); clReleaseCommandQueue(q2); return {}; }
    // ── streaming state ─────────────────────────────────────────────────────────────────────────
    // This path used to allocate its KV caches, its next-frame embedding and its frame position
    // locally, which is exactly why live mode could not use it: ProcessEngine strips `pipeline=1`
    // from every streaming request and run_song demoted a continued session to the sequential AR.
    // That cost live mode everything this path exists for — the codec on its own thread, and the
    // AR/codec overlap — and it is the mode that actually needs to keep up with playback.
    //
    // The session already holds precisely these three things for the sequential path. Borrow them
    // when one is supplied, allocate locally when it is not (one-shot), and free only what we own.
    cl_mem kc_local[12], vc_local[12];
    cl_mem* kc; cl_mem* vc; cl_mem ti;
    const bool owns_state = (sess == nullptr);
    if (!owns_state) {
        if (!sess->alive) {
            for (int k=0;k<12;k++){
                sess->kc[k]=pool_alloc(cl_ctx.context(),CL_MEM_READ_WRITE,(size_t)MAXF*dim*sizeof(float),nullptr,nullptr);
                sess->vc[k]=pool_alloc(cl_ctx.context(),CL_MEM_READ_WRITE,(size_t)MAXF*dim*sizeof(float),nullptr,nullptr); }
            sess->ti = pool_alloc(cl_ctx.context(),CL_MEM_READ_WRITE,(size_t)dim*sizeof(float),nullptr,nullptr);
            clEnqueueCopyBuffer(q1,temporal_input_0,sess->ti,0,0,(size_t)dim*sizeof(float),0,nullptr,nullptr);
            sess->pos = 0; sess->alive = true;
        }
        kc = sess->kc; vc = sess->vc; ti = sess->ti;
    } else {
        for (int k=0;k<12;k++){ kc_local[k]=pool_alloc(cl_ctx.context(),CL_MEM_READ_WRITE,(size_t)MAXF*dim*sizeof(float),nullptr,nullptr);
                                vc_local[k]=pool_alloc(cl_ctx.context(),CL_MEM_READ_WRITE,(size_t)MAXF*dim*sizeof(float),nullptr,nullptr); }
        kc = kc_local; vc = vc_local;
        ti = pool_alloc(cl_ctx.context(),CL_MEM_READ_WRITE,(size_t)dim*sizeof(float),nullptr,nullptr);
        clEnqueueCopyBuffer(q1,temporal_input_0,ti,0,0,(size_t)dim*sizeof(float),0,nullptr,nullptr);
    }
    // Absolute frame index of this request's first frame. The temporal KV cache is indexed by it,
    // so a continued request must carry on from where the last one stopped, not restart at 0.
    const int pos0 = owns_state ? 0 : sess->pos;
    cl_mem grid_gpu = pool_alloc(cl_ctx.context(),CL_MEM_READ_WRITE,(size_t)n_frames*NCB*sizeof(int),nullptr,&e);
    // ── the codec runs on its own HOST thread ───────────────────────────────────────────────────
    // Two queues were never the problem: PIPE_SYNC has reported "marker+barrier (overlapping)" all
    // along and the GPU could always run codec work on q2 beside AR work on q1. What did not
    // overlap was the HOST. run_codec_decode was called straight from the frame loop, so the AR
    // stopped generating for as long as the codec took to issue ~200 dispatches and block on queue
    // backpressure. The measurement said so plainly: AR and codec ADD to the total, every run
    // (2.044 + 2.213 vs 4.666), when genuine overlap would give roughly the larger of the two.
    //
    // So: a worker thread owns q2. The AR hands it a chunk plus a cross-queue event and goes
    // straight back to generating frames. Chunk 0 is deliberately run INLINE first — every codec
    // kernel, program and weight-transpose cache is a function-local `static` guarded by a plain
    // null check, so the first call must not race a second thread building the same ones.
    const int n_chunks = (n_frames + CHUNK - 1) / CHUNK;
    std::vector<cl_mem> chunk_wav((size_t)n_chunks, nullptr);
    std::vector<int>    chunk_nw((size_t)n_chunks, 0);
    // Markers bracket each chunk ON q2. CLBlast is invisible to profEnqueue, so the only way to see
    // its GPU span is to time the queue itself: marker A executes before the chunk's kernels, marker
    // B after, and B.END - A.END is the span the codec occupied the GPU for.
    std::vector<cl_event> mk_a((size_t)((n_frames + CHUNK - 1) / CHUNK), nullptr);
    std::vector<cl_event> mk_b(mk_a.size(), nullptr);
    struct CodecJob { cl_event gate; cl_mem codes; int cf; int idx; };
    std::mutex               job_mu;
    std::condition_variable  job_cv;
    std::deque<CodecJob>     jobs;
    bool                     jobs_closed = false;
    std::thread              codec_thread;
    int chunk_idx = 0;
    auto codec_worker = [&]() {
        OpenCLContext::setThreadQueue(q2);
        nnopt_pin_perf_cores("codec");   // affinity is per-THREAD on Linux
        for (;;) {
            CodecJob j;
            {
                std::unique_lock<std::mutex> lk(job_mu);
                job_cv.wait(lk, [&]{ return !jobs.empty() || jobs_closed; });
                if (jobs.empty()) break;             // closed and drained
                j = jobs.front(); jobs.pop_front();
            }
            if (j.gate) {                            // q2 waits for q1's codes without stalling us
                clEnqueueBarrierWithWaitList(q2, 1, &j.gate, nullptr);
                clReleaseEvent(j.gate);
            }
            int nw = 0;
            if (clEnqueueMarkerWithWaitList) clEnqueueMarkerWithWaitList(q2,0,nullptr,&mk_a[(size_t)j.idx]);
            cl_mem cw = run_codec_decode(cl_ctx, weights, q2, {}, j.cf, &nw, j.codes);
            if (clEnqueueMarkerWithWaitList) clEnqueueMarkerWithWaitList(q2,0,nullptr,&mk_b[(size_t)j.idx]);
            chunk_wav[(size_t)j.idx] = cw; chunk_nw[(size_t)j.idx] = nw;
            pool_free(j.codes);
        }
        OpenCLContext::setThreadQueue(nullptr);
    };
    int cstart=0;
    // Host-issue spans. In pipelined mode the two stages run on separate queues and OVERLAP, so
    // there is no wall to split — reporting 0.000 for both (which is what this did) hides the
    // breakdown exactly where it is hardest to reason about. These measure the time the host spends
    // issuing each stage; they legitimately sum to MORE than the total, and the BENCH line says so.
    double ar_ms = 0.0, codec_ms = 0.0;
    using pclk = std::chrono::steady_clock;
    cl_ctx.setActiveQueue(q1);
    nnopt_set_live_queue(q1);   // position writes must never land on the recordable queue
    nnopt_kinst_reset();        // this render's dispatch trace and generation start clean
    OpenCLContext::hostProfReset();
    nnopt_pin_perf_cores("AR");
    // One banner per render, so the newest one is findable without scrolling the whole log.
    { static int render_no = 0;
      std::fprintf(stderr, "\n========== RENDER #%d  frames=%d chunk=%d ==========\n",
                   ++render_no, n_frames, CHUNK);
      std::fflush(stderr); }
    // ── record/replay ───────────────────────────────────────────────────────────────────────────
    // The AR is host-issue-bound: AR_ISSUE measures ~97% of the span in the issue loop with the
    // process burning MORE CPU than that span, i.e. building the command stream, not waiting on the
    // GPU. cl_qcom_recordable_queues captures one frame's ~455 dispatches and replays them, so
    // every later frame costs one enqueue instead of 455. Everything the frame needs to vary now
    // lives in device buffers (position) or at fixed addresses (`ti`), so a single capture is valid
    // for all frames. NNOPT_RECORD=0 forces live dispatch.
    static int rec_on = -1, rec_epoch = -1;
    if (rec_epoch != nnopt_toggle_epoch()) {
        // DEFAULT ON, and self-disabling — the same contract as the other tuners: the engine proves
        // the path is correct on this device and drops it otherwise. Nobody types a flag.
        //
        // History: this reset the GPU on both the 620 and the 840, because the engine shared ONE
        // cl_kernel per kernel TYPE across every layer. A recording references the kernel rather
        // than snapshotting its arguments, so a replay ran all 192 gemv dispatches with the LAST
        // args set — wrong buffers, out-of-bounds reads, fault, reset. Every recorded dispatch now
        // owns its own kernel object (nnopt_kernel_instance), with args set once at capture.
        //
        // Three independent gates stand between this and a device reset, all evaluated on device:
        //   1. nnopt_record_safe()   — no dispatch baked a host-computed value into an argument,
        //                              and the frame's dispatch sequence is identical every frame.
        //   2. record_capture_executes() — whether the captured frame's work actually happened.
        //   3. the replay verification below — a replayed frame must produce the SAME 12 tokens as
        //                              the same frame computed live, or the recording is dropped.
        // NNOPT_RECORD=0 forces live dispatch.
        // DEFAULT ON. NNOPT_RECORD=0 forces live dispatch.
        //
        // What guards it, all evaluated on the device with no flag to type:
        //   1. nnopt_record_safe()          — nothing baked a host value into an arg, the frame's
        //                                     dispatch sequence is identical every frame, and every
        //                                     dispatch in the frame came through the per-site
        //                                     kernel pool (dispatch/lookup counts must reconcile).
        //   2. record_capture_executes()    — whether the captured frame's work actually happened.
        //   3. the replay verification      — a replayed frame must produce the same 12 tokens as
        //                                     the same frame computed live.
        //
        // Gate 3 catches a WRONG replay. It cannot catch a FAULTING one — a GPU fault takes the
        // device down before any comparison runs — so the lifetime rules matter more than the
        // check: drain before destroying a recording, drain before releasing a buffer a replay
        // reads, and never let the codec's allocator touch a captured buffer.
        const char* r = std::getenv("NNOPT_RECORD");
        rec_on = (r && r[0] == '0') ? 0 : 1;
        rec_epoch = nnopt_toggle_epoch();
    }
    cl_command_queue rq = nullptr;
    cl_recording_qcom rec = nullptr;
    int rec_frame = -1;              // the frame whose dispatch sequence was captured
    bool rec_verified = false;       // a replay has been proved to match a live frame token-for-token
    // Split the replay cost into ENQUEUE (what a recording is supposed to make nearly free) and
    // WAIT (the throttle's clFinish). If enqueue dominates, recordable queues do not pay on this
    // driver; if wait dominates, we are simply blocking on a GPU the codec is also using.
    double rep_enq_ms = 0.0, rep_wait_ms = 0.0; long rep_n = 0;
    bool   codec_profiled = false;   // one codec-chunk profile per RENDER
    // MEASURED: draining every 20 frames instead of 5 cut AR 3.119 -> 2.979 but cost the codec
    // 1.307 -> 1.608 and the total 4.589 -> 5.109. One GPU, two queues: a deep AR queue starves the
    // codec on q2 and the render loses more than the AR gains. Back to a chunk.
    const int kReplayDrainEvery = CHUNK;
    // Frame 0 runs live (it carries every one-time build, tuner and cache fill), frame 1 is
    // captured, frame 2 verifies the replay against a live recomputation, frames 3+ replay. Below
    // that there is nothing left to amortise, so do not record at all.
    const bool want_record = rec_on && cl_ctx.has_recordable_queues() && n_frames >= 4;
    if (want_record) {
        rq = cl_ctx.create_recordable_queue();
        if (!rq) std::fprintf(stderr, "RECORDQ no recordable queue — live dispatch\n");
    }

    // Scratch for the replay verification: `ti` is the only cross-frame state a frame both reads
    // and writes, so saving it is enough to run the same frame twice (live, then replayed) and
    // compare. The KV row and the grid slot are written identically by both passes.
    cl_mem ti_save = nullptr, ti_live = nullptr;
    if (want_record) {
        ti_save = pool_alloc(cl_ctx.context(),CL_MEM_READ_WRITE,(size_t)dim*sizeof(float),nullptr,&e);
        ti_live = pool_alloc(cl_ctx.context(),CL_MEM_READ_WRITE,(size_t)dim*sizeof(float),nullptr,&e);
    }

    static cl_kernel stk = nullptr;
    if (!stk) {
        cl_program sp = cl_ctx.build_program_from_file("kernels/store_tokens_f32.cl");
        if (sp) stk = clCreateKernel(sp, "store_tokens_f32", &e);
    }

    // Dropping a recording mid-render must also drop the pool PIN it required. The pin holds every
    // buffer the captured frame touched so a concurrent codec cannot take one from the pool and
    // overwrite what a replay reads — but with the recording gone nothing reads them, and leaving
    // it on would accumulate every later live frame's intermediates for the rest of the render
    // (n_frames can be 512). Drain first: an in-flight replay may still be reading them.
    auto drop_recording = [&](const char* why) {
        clFinish(q1);
        if (rec) { cl_ctx.release_recording(rec); rec = nullptr; }
        rec_verified = false;
        pool_set_pin(false);
        pool_release_pinned();   // safe: the clFinish above drained every replay that read them
        if (why) { std::fprintf(stderr, "RECORDQ %s — live dispatch for the rest of the render\n", why);
                   std::fflush(stderr); }
    };

    for (int f=0; f<n_frames; ++f) {
        const auto t_ar0 = pclk::now();
        // The position every recorded kernel reads. Written before the replay, on the live queue.
        nnopt_pos_buffer(cl_ctx, q1, pos0 + f);
        nnopt_kinst_frame_begin();   // restart the dispatch ordinal that keys per-site kernels
        // Frame 0 is live and on the profiling queue: profile it once to see the GPU cost of every
        // dispatch in a real frame. One clFinish per render, no flag, no second run.
        if (f == 0 && arprof_enabled()) cl_ctx.arProfBegin();
        const bool verify_frame = (rec && !rec_verified && f == rec_frame + 1 && ti_save && ti_live);
        if (verify_frame)
            clEnqueueCopyBuffer(q1, ti, ti_save, 0,0,(size_t)dim*sizeof(float),0,nullptr,nullptr);

        if (rec && rec_verified && f > rec_frame) {
            // Whole frame in one enqueue.
            if (cl_ctx.enqueue_recording(q1, rec, 0, nullptr) != CL_SUCCESS) {
                NNOPT_ERROR_FMT("pipeline: replay failed at frame %d", f);
                drop_recording("replay failed");
            } else {
                // THROTTLE. A replay is one enqueue, so the host can run the whole render ahead of
                // the device. The original bound was one CHUNK (5 frames) — chosen when a replay
                // could fault, i.e. when deep queueing was dangerous rather than merely deep. With
                // per-site kernel objects and correct buffer lifetimes it is just queueing, and a
                // clFinish here does NOT only wait for the AR: under pipeline=1 the codec is on the
                // other queue competing for the same GPU, so this wait charges codec GPU time to
                // the AR's host-issue span. That is a plausible reading of "recording works and
                // buys nothing", so the wait is now measured separately and bounded far less often.
                {
                    const auto t_enq = pclk::now();
                    rep_enq_ms += std::chrono::duration<double,std::milli>(t_enq - t_ar0).count();
                    ++rep_n;
                    if ((f - rec_frame) % kReplayDrainEvery == 0) {
                        clFinish(q1);
                        rep_wait_ms += std::chrono::duration<double,std::milli>(pclk::now()-t_enq).count();
                    }
                }
                nnopt_kinst_frame_end();   // replayed frame issues no per-site lookups; close anyway
                ar_ms += std::chrono::duration<double,std::milli>(pclk::now()-t_ar0).count();
                if ((f+1) % CHUNK == 0 || f == n_frames-1) goto ar_chunk_boundary;
                continue;
            }
        }

        {
        // Capture on frame 1: frame 0 warms every kernel build and one-time cache, which must not
        // land inside the recording. The capture pass executes the frame as well.
        // nnopt_record_safe() is false if any dispatch so far baked a host-computed value into an
        // argument (frame 0 exercises every path this render will take), or if the dispatch
        // sequence already diverged. Checking it HERE means a capture is never even attempted on a
        // frame shape that cannot be replayed safely.
        const bool capturing = (rq && !rec && f == 1 && nnopt_record_safe());
        if (capturing) {
            rec = cl_ctx.new_recording(rq);
            // Pin from the moment capture starts and keep it pinned for the whole render: every
            // buffer this frame touches must remain the AR's, since all later frames replay these
            // exact handles while the codec allocates concurrently from the same pool.
            if (rec) { cl_ctx.setActiveQueue(rq); rec_frame = f; pool_set_pin(true); }
        }
        cl_command_queue fq = capturing && rec ? rq : q1;

        cl_mem tout = run_temporal_step(cl_ctx,weights,fq,ti,source,kc,vc,dim,pos0 + f);
        if (!tout) { NNOPT_ERROR_FMT("pipeline: temporal frame %d", f); break; }
        cl_mem tokbuf = run_depth_loop_gpu(cl_ctx,weights,fq,tout,nullptr,pos0+f);
        pool_free(tout);
        if (!tokbuf) { NNOPT_ERROR_FMT("pipeline: depth frame %d", f); break; }
        if (stk) {
            cl_mem pb = nnopt_pos_buffer(cl_ctx, q1, pos0 + f);
            static const std::string kStoreTok("ar.store_tokens");
            cl_kernel stki = nnopt_kernel_instance(stk, kStoreTok);
            clSetKernelArg(stki,0,sizeof(cl_mem),&tokbuf); clSetKernelArg(stki,1,sizeof(cl_mem),&grid_gpu);
            clSetKernelArg(stki,2,sizeof(cl_mem),&pb);     clSetKernelArg(stki,3,sizeof(int),&NCB);
            clSetKernelArg(stki,4,sizeof(int),&pos0);      // grid row = absolute pos - pos0
            size_t gs=(size_t)NCB; cl_ctx.profEnqueue(stki,1,&gs,nullptr,"store_tokens");
        } else {
            clEnqueueCopyBuffer(fq,tokbuf,grid_gpu,0,(size_t)f*NCB*sizeof(int),(size_t)NCB*sizeof(int),0,nullptr,nullptr);
        }
        // Write the next frame's input INTO `ti` rather than swapping buffers.
        // `|| sess`: the LAST frame's embedding is what the next request starts from, so a
        // continuing session must advance it too. Without this the session resumed from the
        // second-to-last frame — chunk 0 was bit-identical to the sequential path and every chunk
        // after it was garbage (rms 212 against 2727).
        if (f < n_frames-1 || sess) frame_embed_from_tokbuf(cl_ctx,weights,fq,tokbuf,ti);
        pool_free(tokbuf);

        if (capturing && rec) {
            cl_ctx.setActiveQueue(q1);
            if (cl_ctx.end_recording(rec) != CL_SUCCESS) {
                drop_recording("end_recording failed");
            } else {
                std::fprintf(stderr, "RECORDQ captured AR frame %d (%zu kernel instances)\n",
                             f, nnopt_kinst_count());
                std::fflush(stderr);
                // A recordable queue records; whether it also EXECUTES is a device property, asked
                // once (never assumed). If it only records, this frame produced nothing: its KV
                // cache row and its 12 tokens are missing, which is silently wrong audio rather
                // than a crash. Replay once, with the position buffer still holding THIS frame, to
                // actually perform the work. If capture does execute, replaying would double-apply
                // it — frame_embed advances `ti`, so the frame is NOT idempotent — hence the probe.
                // From here on, any LIVE frame must vend a DISJOINT set of kernel objects. The
                // recording references the ones captured above and holds no copy of their args, so
                // a live frame reusing them would repoint the recording at its own pool buffers —
                // which is the failure this whole change exists to remove.
                nnopt_kinst_set_generation(1);
                // Capture is complete: stop intercepting frees. The buffers already retained stay
                // retained (pool_release_pinned() hands them back after the final drain) — this
                // only stops the CODEC's per-chunk frees from piling up for the whole render.
                pool_set_pin(false);
                if (cl_ctx.record_capture_executes() == 0) {
                    if (cl_ctx.enqueue_recording(q1, rec, 0, nullptr) != CL_SUCCESS)
                        drop_recording("replay of the captured frame failed");
                } else if (cl_ctx.record_capture_executes() < 0) {
                    drop_recording("capture-execution semantics undetermined");
                }
            }
        }
        }
        if (f == 0 && arprof_enabled()) cl_ctx.arProfEnd(q1);

        // ── replay verification ────────────────────────────────────────────────────────────────
        // The frame just computed LIVE is now recomputed by REPLAYING the recording, and the two
        // must agree on all 12 tokens. Tokens are the right thing to compare: the audio derives
        // from them, so identical tokens mean identical audio, and a stale-argument replay (the
        // failure that reset the GPU) cannot produce them by chance.
        //
        // This is what allows the feature to be ON by default with no device in hand and no flag
        // for anyone to type: a recording that cannot reproduce a live frame is dropped here, and
        // the render finishes on the live path at the old speed rather than on wrong audio.
        if (verify_frame) {
            std::vector<int32_t> tok_live(NCB, -1), tok_rep(NCB, -2);
            clFinish(q1);
            clEnqueueReadBuffer(q1, grid_gpu, CL_TRUE, (size_t)f*NCB*sizeof(int),
                                (size_t)NCB*sizeof(int), tok_live.data(), 0, nullptr, nullptr);
            clEnqueueCopyBuffer(q1, ti, ti_live, 0,0,(size_t)dim*sizeof(float),0,nullptr,nullptr);
            // Rewind the only state the frame both reads and writes, then replay the same frame.
            clEnqueueCopyBuffer(q1, ti_save, ti, 0,0,(size_t)dim*sizeof(float),0,nullptr,nullptr);
            clFinish(q1);

            bool ok = (cl_ctx.enqueue_recording(q1, rec, 0, nullptr) == CL_SUCCESS);
            if (ok) {
                clFinish(q1);
                clEnqueueReadBuffer(q1, grid_gpu, CL_TRUE, (size_t)f*NCB*sizeof(int),
                                    (size_t)NCB*sizeof(int), tok_rep.data(), 0, nullptr, nullptr);
                for (int i = 0; i < NCB; ++i) if (tok_live[i] != tok_rep[i]) {
                    std::fprintf(stderr,
                        "RECORDQ verify FAILED at codebook %d (live %d, replay %d) — dropping the "
                        "recording, continuing live\n", i, (int)tok_live[i], (int)tok_rep[i]);
                    ok = false;
                    break;
                }
            } else {
                std::fprintf(stderr, "RECORDQ verify replay enqueue failed — continuing live\n");
            }
            // Either way the LIVE result is the one that stands: it is what the rest of the render
            // continues from, and it is correct by construction.
            clEnqueueCopyBuffer(q1, ti_live, ti, 0,0,(size_t)dim*sizeof(float),0,nullptr,nullptr);
            clFinish(q1);

            if (ok && !nnopt_record_safe()) {
                std::fprintf(stderr, "RECORDQ verify passed but a later dispatch is unsafe to "
                                     "record — continuing live\n");
                ok = false;
            }
            if (ok) {
                rec_verified = true;
                std::fprintf(stderr, "RECORDQ verify ok — replaying frames %d..%d as one enqueue each\n",
                             f+1, n_frames-1);
            } else {
                drop_recording("replay did not reproduce the live frame");
            }
            std::fflush(stderr);
        }
        ar_ms += std::chrono::duration<double,std::milli>(pclk::now()-t_ar0).count();
        ar_chunk_boundary:;
        // The AR part of this frame is done. Close the ordinal window BEFORE the codec runs: the
        // codec shares helpers like residual_add, and counting its dispatches into this frame's
        // sequence is what made frame 4 diverge from frame 0.
        nnopt_kinst_frame_end();
        if ((f+1) % CHUNK == 0 || f == n_frames-1) {
            const auto t_cd0 = pclk::now();
            const int cf = f - cstart + 1, n = cf*NCB, off = cstart*NCB;
            cl_mem codes_chunk = pool_alloc(cl_ctx.context(),CL_MEM_READ_WRITE,(size_t)n*sizeof(int),nullptr,&e);
            clSetKernelArg(uok,0,sizeof(cl_mem),&grid_gpu); clSetKernelArg(uok,1,sizeof(cl_mem),&codes_chunk);
            clSetKernelArg(uok,2,sizeof(int),&n); clSetKernelArg(uok,3,sizeof(int),&off);
            size_t gu=(size_t)n; cl_ctx.profEnqueue(uok,1,&gu,nullptr,"rvq_unoffset");  // on active q1
            // cross-queue gate: codec on q2 must wait for q1 to finish this chunk's grid+codes
            // Which cross-queue gate we got decides whether pipelining overlaps AT ALL: the
            // marker/barrier pair lets q2 wait on q1's work without stalling the host, while the
            // clFinish fallback drains q1 completely and serialises the two stages by construction.
            // These are weak symbols, so this is a link-time outcome, not a device capability —
            // report it once rather than assume.
            static bool sync_logged = false;
            if (!sync_logged) {
                sync_logged = true;
                std::fprintf(stderr, "PIPE_SYNC gate=%s (marker=%p barrier=%p)\n",
                             (clEnqueueMarkerWithWaitList && clEnqueueBarrierWithWaitList)
                                 ? "marker+barrier (overlapping)" : "clFinish (NO OVERLAP)",
                             (void*)clEnqueueMarkerWithWaitList, (void*)clEnqueueBarrierWithWaitList);
                std::fflush(stderr);
            }
            // The gate is now handed to whoever decodes this chunk — the worker applies it on q2
            // so the barrier is enqueued from the thread that owns that queue.
            cl_event gate_ev = nullptr;
            if (clEnqueueMarkerWithWaitList && clEnqueueBarrierWithWaitList) {
                clEnqueueMarkerWithWaitList(q1, 0, nullptr, &gate_ev);
                clFlush(q1);    // the marker must be visible to the other thread's barrier
            } else {
                clFinish(q1);   // no event API here — drain instead (correct, no overlap)
            }
            const int this_idx = chunk_idx++;
            // Chunk 0 only needs the inline treatment when nothing has built the codec
            // statics yet. In --serve the warm-up render already did, so handing it to
            // the worker like every other chunk recovers a whole chunk of AR/codec
            // overlap — at frames=50 chunk=25 that was HALF the codec serialised behind
            // the AR ("codec (AR blocked)" in the engine report).
            const bool need_inline_warm = !g_codec_statics_warm.load(std::memory_order_acquire);
            if (this_idx == 0 && need_inline_warm) {
                // Inline, on this thread: warms every codec static before the worker exists.
                if (gate_ev) { clEnqueueBarrierWithWaitList(q2, 1, &gate_ev, nullptr); clReleaseEvent(gate_ev); }
                cl_ctx.setActiveQueue(q2);
                // Tag this chunk's pool buffers as q2's even though we are on the AR's thread. The
                // pool only recycles a buffer back to the queue that last used it, and these are
                // still in flight on q2 when they are freed — untagged, the AR could take one and
                // write into a buffer the codec is reading.
                OpenCLContext::setThreadQueue(q2);
                int nw = 0;
                if (clEnqueueMarkerWithWaitList) clEnqueueMarkerWithWaitList(q2,0,nullptr,&mk_a[0]);
                chunk_wav[0] = run_codec_decode(cl_ctx, weights, q2, {}, cf, &nw, codes_chunk);
                if (clEnqueueMarkerWithWaitList) clEnqueueMarkerWithWaitList(q2,0,nullptr,&mk_b[0]);
                chunk_nw[0] = nw;
                pool_free(codes_chunk);
                OpenCLContext::setThreadQueue(nullptr);
                cl_ctx.setActiveQueue(q1);
                codec_thread = std::thread(codec_worker);
            } else {
                // Hand it off and go straight back to generating. This is the whole change.
                // The worker used to be spawned by the inline chunk-0 branch; when that
                // branch is skipped it has to be created here instead. Only the AR thread
                // touches codec_thread, so this needs no lock.
                if (!codec_thread.joinable()) codec_thread = std::thread(codec_worker);
                { std::lock_guard<std::mutex> lk(job_mu);
                  jobs.push_back(CodecJob{gate_ev, codes_chunk, cf, this_idx}); }
                job_cv.notify_one();
            }
            clFlush(q1);
            codec_ms += std::chrono::duration<double,std::milli>(pclk::now()-t_cd0).count();
            cstart = f+1;
        }
    }
    // Split the AR's cost into HOST ISSUE vs GPU DRAIN. Everything above is enqueued
    // asynchronously, so `ar_ms` is the time the host spent building the command stream and this
    // final wait is what the GPU still owed. That ratio is the whole case for recordable queues:
    // recording removes host-side enqueue cost, so it can only help if issue is a real share of the
    // total. If the drain dominates, the GPU is the wall and a recording changes nothing.
    // DRAIN FIRST. Everything below destroys objects the queued work still references: the
    // recording, the recordable queue, and (via pool_set_pin(false)) every buffer the replays read
    // and write. With the replay throttle allowing a chunk of frames in flight, that left ~2600
    // dispatches referencing a destroyed recording and freed buffers — a GPU-side use-after-free,
    // which is a device reset, not an error code. Ordinary enqueued kernels are refcount-protected
    // by the runtime; a recording holds its references OUTSIDE that tracking, and this driver's
    // recording support is already known to be partial (cl_array_arg_qcom returns -59 here).
    // JOIN FIRST. Everything below releases the recording, the recordable queue and the pinned
    // pool buffers; a codec chunk still in flight on the worker allocates from that same pool.
    {
        { std::lock_guard<std::mutex> lk(job_mu); jobs_closed = true; }
        job_cv.notify_all();
        if (codec_thread.joinable()) codec_thread.join();
    }
    const auto t_drain0 = pclk::now();
    clFinish(q1); clFinish(q2);
    if (rec) cl_ctx.release_recording(rec);
    if (rq)  clReleaseCommandQueue(rq);
    if (ti_save) pool_free(ti_save);
    if (ti_live) pool_free(ti_live);
    nnopt_kinst_set_generation(0);
    pool_set_pin(false);
    pool_release_pinned();   // after the drain above, never before
    const double drain_ms = std::chrono::duration<double,std::milli>(pclk::now()-t_drain0).count();
    // Codec GPU span, summed over chunks. The queues are drained above, so the timestamps are final.
    double codec_gpu_ms = 0.0;
    for (size_t i = 0; i < mk_a.size(); ++i) {
        if (!mk_a[i] || !mk_b[i]) continue;
        cl_ulong a = 0, b = 0;
        clGetEventProfilingInfo(mk_a[i], CL_PROFILING_COMMAND_END, sizeof(a), &a, nullptr);
        clGetEventProfilingInfo(mk_b[i], CL_PROFILING_COMMAND_END, sizeof(b), &b, nullptr);
        if (b > a) codec_gpu_ms += (double)(b - a) / 1e6;
    }
    for (size_t i = 0; i < mk_a.size(); ++i) {
        if (mk_a[i]) clReleaseEvent(mk_a[i]);
        if (mk_b[i]) clReleaseEvent(mk_b[i]);
    }
    nnopt_set_codec_gpu_ms(codec_gpu_ms);
    // A high issue share is NOT by itself proof the host is the bottleneck: enqueue blocks once the
    // queue is full, so a host throttled by the GPU looks identical to a host doing real work. CPU
    // time separates them — if the process burned CPU for that span it was genuinely building
    // commands (recordable queues recover it); if it burned almost none it was parked on
    // backpressure and recording is worthless.
    if (rep_n > 0) {
        std::fprintf(stderr,
            "REPLAY n=%ld enqueue=%.0f ms wait=%.0f ms (enqueue dominates => recordable queues do "
            "not pay here; wait dominates => blocked on a GPU shared with the codec)\n",
            rep_n, rep_enq_ms, rep_wait_ms);
        std::fflush(stderr);
    }
    struct rusage ru{}; getrusage(RUSAGE_SELF, &ru);
    const double cpu_ms = (double)ru.ru_utime.tv_sec * 1e3 + ru.ru_utime.tv_usec / 1e3 +
                          (double)ru.ru_stime.tv_sec * 1e3 + ru.ru_stime.tv_usec / 1e3;
    OpenCLContext::hostProfReport(ar_ms, n_frames);
    char i8buf[128]; nnopt_int8_summary(i8buf, sizeof(i8buf));
    std::fprintf(stderr, "AR_ISSUE host_issue=%.0f ms drain=%.0f ms issue_share=%.0f%% | %s\n",
                 ar_ms, drain_ms, 100.0 * ar_ms / (ar_ms + drain_ms > 0 ? ar_ms + drain_ms : 1),
                 i8buf);
    (void)cpu_ms;   // process_cpu is cumulative for the PROCESS, so comparing it against one
                    // render's span produced numbers like "cpu/issue=769%" that mean nothing.
    std::fflush(stderr);
    cl_ctx.setActiveQueue(q1);
    // assemble waveform from the per-chunk codec outputs (single readback each, codec already done)
    std::vector<float> ro;
    for (size_t i=0;i<chunk_wav.size();++i){ if (!chunk_wav[i]) continue;
        size_t base=ro.size(); int nw=chunk_nw[i];
        ro.resize(base+(size_t)nw*2);
        clEnqueueReadBuffer(q1,chunk_wav[i],CL_TRUE,0,(size_t)nw*2*sizeof(float),ro.data()+base,0,nullptr,nullptr);
        pool_free(chunk_wav[i]); }
    if (tokens_o) {
        tokens_o->assign((size_t)n_frames*NCB, 0);
        clEnqueueReadBuffer(q1, grid_gpu, CL_TRUE, 0, tokens_o->size()*sizeof(int),
                            tokens_o->data(), 0, nullptr, nullptr);
    }
    if (ar_ms_o) *ar_ms_o = ar_ms;
    if (codec_ms_o) *codec_ms_o = codec_ms;
    if (nwav_o) *nwav_o = (int)(ro.size()/2);
    if (owns_state) pool_free(ti);
    pool_free(grid_gpu);
    if (owns_state) { for (int k=0;k<12;k++){ pool_free(kc[k]); pool_free(vc[k]); } }
    else            { sess->pos = pos0 + n_frames; }   // the session owns its buffers
    clReleaseCommandQueue(q2);
    return ro;
}

// ── THE PRODUCTION PATH ──────────────────────────────────────────────────────
// style conditioning → n_frames of AR tokens → codec → interleaved stereo PCM.
// Everything that generates music (main.cpp one-shot, --serve, the `song`
// op-test) goes through here, so there is one e2e path to optimize and to
// benchmark. Timing is split AR/codec only in the sequential path — when
// pipelined the two stages overlap by construction and only the total is real.
bool run_song(OpenCLContext& cl_ctx, Weights& weights,
              const SongConditioning& cond, const SongConfig& cfg,
              SongResult& out, SongSession* sess) {
    out = SongResult();
    if (cfg.n_frames <= 0) { NNOPT_ERROR_FMT("run_song: n_frames must be > 0 (got %d)", cfg.n_frames); return false; }
    nnopt_codec_stats_reset();   // nothing measured in a previous request may be reported in this one
    // Arm the sampler for this render. A seed of 0 means "vary per render": without it every chunk
    // of a live stream would draw the SAME noise for the same frame index, which is a different way
    // of getting a stuck loop than greedy decoding but sounds much the same.
    {
        unsigned sd = cfg.seed;
        if (sd == 0) { static unsigned counter = 0x1234567u; counter = counter * 1664525u + 1013904223u; sd = counter; }
        nnopt_set_sampling(cfg.temperature, cfg.top_k, sd);
        std::fprintf(stderr, "SAMPLING temperature=%.2f top_k=%d seed=%u%s\n",
                     cfg.temperature, cfg.top_k, sd, cfg.temperature > 0.0f ? "" : " (greedy)");
        std::fflush(stderr);
    }
    // A request shorter than the codec's minimum decodes to nothing (and used to segfault inside
    // the cascade). Round up rather than refuse — the floor is 100 ms of audio.
    int n_frames = cfg.n_frames;
    if (n_frames < kMinCodecFrames) {
        std::fprintf(stderr, "MRT_NOTE frames=%d raised to the codec minimum of %d\n", n_frames, kMinCodecFrames);
        n_frames = kMinCodecFrames;
    }
    if (cond.temporal_input.size() != 1024) {
        NNOPT_ERROR_FMT("run_song: temporal_input is %zu floats, expected 1024", cond.temporal_input.size());
        return false;
    }
    if (cond.style_tokens.empty() && cond.source.size() != 256) {
        NNOPT_ERROR_FMT("run_song: source is %zu floats, expected 256", cond.source.size());
        return false;
    }
    if (cfg.reset && sess) song_session_release(sess);

    cl_command_queue queue = cl_ctx.queue();
    cl_int e = CL_SUCCESS;
    cl_mem xb = pool_alloc(cl_ctx.context(), CL_MEM_READ_ONLY|CL_MEM_COPY_HOST_PTR,
                           cond.temporal_input.size()*sizeof(float),
                           const_cast<float*>(cond.temporal_input.data()), &e);
    // Conditioning: prefer computing it on device from style tokens (the real path), and fall back
    // to a supplied 256-float vector only when no tokens were given.
    cl_mem srcb = nullptr;
    if (cond.style_tokens.size() == 144) {
        cl_mem tokb = pool_alloc(cl_ctx.context(), CL_MEM_READ_ONLY|CL_MEM_COPY_HOST_PTR,
                                 cond.style_tokens.size()*sizeof(int32_t),
                                 const_cast<int32_t*>(cond.style_tokens.data()), &e);
        if (tokb) {
            srcb = Encoder_forward(cl_ctx, weights, queue, tokb, 1, 0, 0, nullptr, nullptr, nullptr, nullptr);
            pool_free(tokb);
        }
        if (!srcb) { NNOPT_ERROR("run_song: style encoder failed on the supplied tokens"); pool_free(xb); return false; }
        std::fprintf(stderr, "CONDITIONING computed on device from %zu style tokens\n", cond.style_tokens.size());
    } else {
        srcb = pool_alloc(cl_ctx.context(), CL_MEM_READ_ONLY|CL_MEM_COPY_HOST_PTR,
                          cond.source.size()*sizeof(float),
                          const_cast<float*>(cond.source.data()), &e);
    }
    if (!xb || !srcb) {
        NNOPT_ERROR("run_song: conditioning buffer alloc failed");
        if (xb) pool_free(xb); if (srcb) pool_free(srcb);
        return false;
    }

    // chunk <= 0 means "size it to this device". The codec's per-chunk buffers scale linearly with
    // the number of frames, and on a 4 GB phone anything past ~15 frames blew the allocator — but
    // that ceiling is the DEVICE's, not the model's, and a flagship has an order of magnitude more
    // room. Rather than hardcode a number that is wrong on one of them, run the first chunk small,
    // measure what the convs actually demanded, and scale up to fit the real budget.
    // GEMM parameters from the per-device table (see xgemm_for_device). No sweep, no cache file,
    // no startup cost, and the same phone gets the same parameters on every launch.
    //
    // The AR used to run its own sweep here too — real frames timed with the depth MLPs in int8 and
    // in fp16, keep the winner. int8 is the answer, on both devices and by ear, so the render no
    // longer re-measures it before every song.
    xgemm_for_device(cl_ctx);
    maybe_override_xgemm(cl_ctx);   // explicit NNOPT_XGEMM still wins, for bringing up a new device
    // Report BEFORE the override, so the line means "CLBlast's own kHalf choice" — which is what it
    // is labelled. Reporting after made it echo whichever sweep candidate was armed, and the card
    // showed three different "defaults" across three presses.
    report_xgemm_half_active(cl_ctx);
    maybe_override_xgemm_half(cl_ctx);   // and the same for the fp16 codec's HGEMM
    // Codec streaming context belongs to ONE continuous stream of audio, so it is cleared when a
    // piece STARTS and not otherwise. Carrying it into a NEW render would splice the tail of the
    // previous song into the head of this one. Unconditionally resetting here wiped it at every request —
    // and live mode is one request per 2 s, so the context could only ever bridge the five
    // sub-chunks inside a request and never the boundary between them.
    if (!sess || cfg.reset) codec_stream_reset();
    // ── streaming state proves itself, once, before it is allowed to touch your audio ───────────
    // This is the feature that shipped silence: with per-conv context carried across chunks the
    // render decayed to zero after a handful of chunks, and the failure was invisible until it had
    // already reached a listener. Reading the code did not find it, so the code no longer gets to
    // assert it is correct — it demonstrates it.
    //
    // codecchunk decodes a fixed synthetic grid whole and then in pieces. Chunking the codec is a
    // SLICING decision, so the two must agree; if they do not, the state is wrong by definition.
    // Runs once per process, costs one small decode, and on failure switches streaming OFF for the
    // rest of the process and says so. Worst case is the 400 ms seam — never silence.
    if (codec_stream_enabled()) {
        static int verified = -1;   // -1 not yet, 0 failed, 1 passed
        if (verified < 0) {
            verified = nnopt_codecchunk_check(cl_ctx, weights, queue, 20, 5) ? 1 : 0;
            std::fprintf(stderr, "CODECSTREAM self-check %s\n",
                         verified ? "PASSED — chunked decode matches whole"
                                  : "FAILED — chunked decode DIFFERS, streaming disabled for this process");
            std::fflush(stderr);
            if (!verified) codec_stream_disable();
            codec_stream_reset();   // the check left its own state behind
        }
    }
    const bool auto_chunk = (cfg.chunk <= 0);
    const int CHUNK = auto_chunk ? 5 : cfg.chunk;
    using clk = std::chrono::steady_clock;
    std::vector<float> ro; int nwav = 0;
    // Wall spent on the codecab reference pass. It is measurement scaffolding, not work the render
    // needs, so it comes back off BOTH codec_sec and total_sec — otherwise turning the quality check
    // on would wreck the RTF number that the quality check exists to protect.
    double ab_extra_s = 0.0;
    const auto t0 = clk::now();
    // The two-queue path owns its own AR state, so it cannot CONTINUE a streaming session. Now that
    // it is the default (it is the only path with record/replay), a live session must demote it to
    // the sequential path rather than the other way round: destroying the caller's session to honour
    // a default would silently render a chunk that does not follow the previous one.
    // The pipelined path CARRIES the session now (see run_song_pipelined). It used to allocate its
    // own KV caches and start every request at frame 0, so a continued stream had to be demoted to
    // the sequential AR — and that quietly cost live mode the codec thread and the AR/codec overlap,
    // i.e. everything the pipelined path exists for, in the one mode that has to keep up with
    // playback. Measured: one-shot RTF 1.38, live mode 2.26 on the same device and settings.
    //
    // ── the cache fills every 20.48 s; COMPACT it, do not throw the piece away ──────────────────
    // This guard runs BEFORE the pipeline/sequential branch, so it is the one that decides what
    // happens when the KV cache fills — and it used to release the session, restarting the AR from
    // the style embedding. In live mode that lands on chunk 11 (10 x 50 = 500 frames, and 500 + 50
    // > 512), so a stream became a different piece of music every twenty seconds, mid-playback.
    //
    // Compaction was added inside run_generate first, which was useless: this guard had already
    // released the session by the time run_generate saw it, so that code never ran. It lives here
    // now, where BOTH paths go through it — the pipelined path has no guard of its own at all.
    //
    // Temporal self-attention windows to 41 frames per layer (ops/CachedSelfAttention.cpp:113,
    // matching the reference's decoder_temporal_self_attention_max_past_horizon for mrt2_small), so
    // nothing reads further back than that. Keeping 64 preserves the entire reachable history; the
    // deep ~20 s receptive field lives in the already-computed K/V of the higher layers, not in how
    // many entries this buffer holds. Safe because no rotary is applied to cached k/v, so the
    // vectors do not depend on the absolute position they were written at, only their order.
    const bool continuing = (sess && sess->alive);
    if (sess && sess->alive && sess->pos + n_frames > kARMaxFrames) {
        const int keep = 64;
        if (sess->pos > 2 * keep && n_frames + keep <= kARMaxFrames) {
            const size_t row = 1024 * sizeof(float);
            const int    src = sess->pos - keep;
            for (int k = 0; k < 12; ++k) {
                clEnqueueCopyBuffer(queue, sess->kc[k], sess->kc[k], (size_t)src*row, 0, (size_t)keep*row, 0, nullptr, nullptr);
                clEnqueueCopyBuffer(queue, sess->vc[k], sess->vc[k], (size_t)src*row, 0, (size_t)keep*row, 0, nullptr, nullptr);
            }
            sess->pos = keep;
            std::fprintf(stderr, "AR_CACHE_COMPACT kept=%d frames, pos=%d (piece continues)\n", keep, sess->pos);
            std::fflush(stderr);
        } else {
            std::fprintf(stderr, "run_song: request of %d frames cannot fit the %d-frame cache — restarting\n",
                         n_frames, kARMaxFrames);
            std::fflush(stderr);
            song_session_release(sess);
        }
    }
    (void)continuing;
    if (cfg.pipeline) {
        int nw = 0;
        double p_ar_ms=0.0, p_codec_ms=0.0;
        ro = run_song_pipelined(cl_ctx, weights, queue, xb, srcb, n_frames, CHUNK, &nw,
                                &p_ar_ms, &p_codec_ms, &out.tokens, sess);
        // Host-issue spans, not disjoint wall: the stages overlap, so these sum to more than
        // total_s. Reported anyway because 0.000/0.000 told us nothing at all.
        out.ar_s = p_ar_ms/1000.0; out.codec_s = p_codec_ms/1000.0;
        nwav = nw;
    } else {
        std::vector<int> grid = run_generate(cl_ctx, weights, queue, xb, srcb, n_frames, sess);
        out.tokens = grid;          // see SongResult::tokens — the only comparable AR output
        const auto t1 = clk::now();
        // global token → per-codebook RVQ code: strip the 6 reserved ids and the codebook offset.
        std::vector<int> codes(grid.size());
        for (size_t i = 0; i < grid.size(); ++i) { const int q = (int)(i % 12); codes[i] = ((grid[i]-6 - q*1024)%1024+1024)%1024; }
        if (std::getenv("NNOPT_DUMP_TOKENS")) {
            FILE* tf = std::fopen("layer_dumps/run_song_grid.bin", "wb");
            if (tf) { std::vector<int32_t> g32(grid.begin(), grid.end());
                      std::fwrite(g32.data(), sizeof(int32_t), g32.size(), tf); std::fclose(tf); }
            FILE* cf2 = std::fopen("layer_dumps/run_song_codes.bin", "wb");
            if (cf2) { std::vector<int32_t> c32(codes.begin(), codes.end());
                       std::fwrite(c32.data(), sizeof(int32_t), c32.size(), cf2); std::fclose(cf2); }
        }
        // TWO different limits govern the chunk, and collapsing them into one number is what pinned
        // every device to the minimum:
        //   * the largest SINGLE buffer must clear the driver's per-allocation limit, and
        //   * the whole working set (~kCodecLiveBuffers buffers of that size) must fit in memory.
        // The old code took min(global, alloc), divided BOTH by kCodecLiveBuffers — applying the
        // working-set divisor to the single-buffer limit too — and then capped at a flat 384 MB.
        // That flat cap always bound: 384/6 = 64 MB against a measured ~46 MB/frame gives chunk 1,
        // clamped up to the floor of 5, so a 16 GB flagship decoded in the same 5-frame pieces as
        // the 4 GB phone the cap was written for. Both limits now scale off what the device
        // reports; the halve-and-retry path below is still the backstop if a chunk fails.
        size_t single_budget = 0, workingset_budget = 0;
        if (auto_chunk) {
            cl_ulong max_alloc = 0, gmem = 0;
            clGetDeviceInfo(cl_ctx.device(), CL_DEVICE_MAX_MEM_ALLOC_SIZE, sizeof(max_alloc), &max_alloc, nullptr);
            clGetDeviceInfo(cl_ctx.device(), CL_DEVICE_GLOBAL_MEM_SIZE, sizeof(gmem), &gmem, nullptr);
            // Half the driver's per-allocation ceiling: asking for all of it is what reset the GPU
            // (and rebooted the phone) on a flagship.
            single_budget     = (size_t)(0.50 * (double)max_alloc);
            workingset_budget = (size_t)(0.25 * (double)gmem) / (size_t)kCodecLiveBuffers;
        }
        int chunk = CHUNK;
        bool sized = !auto_chunk;
        int emitted = 0;
        // A/B state. `ab_fp16_selected` is what the REQUEST asked to hear, so the reference pass is
        // always the other one — the comparison is the same either way round, and this way turning
        // the A/B on never changes which audio comes out.
        const bool o_codec_ab = cfg.codec_ab;
        const char* ab_envp = std::getenv("NNOPT_CODECFP16");
        const bool ab_fp16_selected = (ab_envp && ab_envp[0] != '0');
        double ab_dot = 0.0, ab_na = 0.0, ab_nb = 0.0, ab_err = 0.0;
        int    ab_n = 0;
        for (int f0 = 0; f0 < n_frames; ) {
            int cf = std::min(chunk, n_frames - f0);
            // Absorb a remainder smaller than the codec minimum into this chunk instead of
            // emitting a final 1-4 frame chunk the cascade cannot decode.
            const int rest = n_frames - f0 - cf;
            if (rest > 0 && rest < kMinCodecFrames) cf += rest;
            std::fprintf(stderr, "MRT_PROGRESS stage=codec frame=%d/%d chunk=%d\n", f0, n_frames, cf);
            std::fflush(stderr);
            g_codec_peak_alloc = 0;
            std::vector<int> ccodes(codes.begin()+(size_t)f0*12, codes.begin()+(size_t)(f0+cf)*12);
            // ── codecab=1: decode this chunk BOTH ways and score the difference ──────────────
            // Runs the reference (fp32) pass first, reads it back, then falls through to the normal
            // decode below with whatever path the request actually selected. The listener hears the
            // selected one; this pass exists only to be compared against, so its wall time is
            // accumulated separately and taken back off codec_sec.
            std::vector<float> ab_ref;
            if (o_codec_ab) {
                const auto ab_t0 = clk::now();
                const int prev_force = nnopt_codec_fp16_forced();
                nnopt_codec_fp16_force(ab_fp16_selected ? 0 : 1);   // the path NOT being shipped
                int rnw = 0;
                cl_mem rw = run_codec_decode(cl_ctx, weights, queue, ccodes, cf, &rnw);
                if (rw) {
                    ab_ref.resize((size_t)rnw*2);
                    clEnqueueReadBuffer(queue, rw, CL_TRUE, 0, ab_ref.size()*sizeof(float),
                                        ab_ref.data(), 0, nullptr, nullptr);
                    pool_free(rw);
                } else {
                    NNOPT_ERROR("codecab: reference decode failed — no comparison for this chunk");
                }
                nnopt_codec_fp16_force(prev_force);
                ab_extra_s += std::chrono::duration<double>(clk::now() - ab_t0).count();
                g_codec_peak_alloc = 0;   // the timed pass measures its own peak, not the reference's
            }
            int nw = 0; cl_mem cw = run_codec_decode(cl_ctx, weights, queue, ccodes, cf, &nw);
            if (!cw) {
                // Too big for this device after all — halve and retry the same frames. Only a chunk
                // that fails at the floor is a real failure.
                if (chunk > 5) {
                    chunk = chunk / 2;
                    NNOPT_ERROR_FMT("run_song: codec chunk too large, retrying at %d frames", chunk);
                    continue;
                }
                NNOPT_ERROR_FMT("run_song: codec chunk at frame %d failed", f0);
                break;
            }
            const size_t off = ro.size();
            ro.resize(off + (size_t)nw*2);
            clEnqueueReadBuffer(queue, cw, CL_TRUE, 0, (size_t)nw*2*sizeof(float), ro.data()+off, 0, nullptr, nullptr);
            pool_free(cw);
            // Cosine over the WHOLE render, not per chunk averaged: accumulate the three sums and
            // divide once at the end. A per-chunk mean would let one badly-wrong chunk hide behind
            // several good ones, which is the opposite of what a regression gate is for. Peak error
            // is tracked beside it because cosine is scale-invariant and a uniform gain error — the
            // exact failure mode a mis-applied rescale would produce — leaves cosine at 1.000000.
            if (o_codec_ab && ab_ref.size() == (size_t)nw*2) {
                for (size_t i = 0; i < ab_ref.size(); ++i) {
                    const double a = (double)ab_ref[i], b = (double)ro[off + i];
                    ab_dot += a*b; ab_na += a*a; ab_nb += b*b;
                    const double d = std::fabs(a - b);
                    if (d > ab_err) ab_err = d;
                }
                ++ab_n;
            }
            f0 += cf;
            ++emitted;
            if (!sized && g_codec_peak_alloc > 0) {
                sized = true;
                const double per_frame = (double)g_codec_peak_alloc / (double)cf;
                const int by_single = (int)((double)single_budget / per_frame);
                const int by_set    = (int)((double)workingset_budget / per_frame);
                int want = by_single < by_set ? by_single : by_set;
                if (want > n_frames) want = n_frames;   // never larger than the request
                if (want > kCodecChunkCeiling) want = kCodecChunkCeiling;  // never gamble the GPU
                // MEASURED: a bigger chunk is not a win, it is a loss — and past a point it is a
                // crash. On Adreno 840, chunk 5 -> 10 made the codec 80% SLOWER (3.38 s -> 6.10 s)
                // and an explicit chunk 25 reset the GPU and rebooted the phone. On Adreno 620,
                // 5 -> 10 bought 2.7%, i.e. nothing. So the memory budgets above may only ever
                // SHRINK the chunk on a constrained device; they must never grow it past the size
                // that is actually fastest. Bigger stays reachable with an explicit `chunk=N`,
                // which is how those numbers were measured in the first place.
                if (want > kAutoChunkPreferred) want = kAutoChunkPreferred;
                if (want < kMinCodecFrames) want = kMinCodecFrames;        // the known-good floor
                chunk = want;
                // Both limits are printed because which one binds is the whole diagnosis: a device
                // held at the floor by `single` needs a different fix than one held by `set`.
                std::fprintf(stderr,
                             "MRT_AUTOCHUNK peak=%.1fMB/frame single=%.0fMB(%d) set=%.0fMB(%d) chunk=%d\n",
                             per_frame / 1048576.0,
                             (double)single_budget / 1048576.0, by_single,
                             (double)workingset_budget / 1048576.0, by_set, chunk);
                std::fflush(stderr);
            }
        }
        (void)emitted;
        nwav = (int)(ro.size()/2);
        out.ar_s    = std::chrono::duration<double>(t1-t0).count();
        out.codec_s = std::chrono::duration<double>(clk::now()-t1).count() - ab_extra_s;
        if (o_codec_ab) {
            const double denom = std::sqrt(ab_na) * std::sqrt(ab_nb);
            const double cosv  = (denom > 0.0) ? (ab_dot / denom) : 0.0;
            nnopt_set_codec_ab(cosv, ab_err, ab_n);
            std::fprintf(stderr,
                         "CODEC_AB chunks=%d cos=%.6f max_abs_err=%.6g  (%s was heard, "
                         "the other was the reference; reference pass cost %.3f s, excluded)\n",
                         ab_n, cosv, ab_err, ab_fp16_selected ? "fp16" : "fp32", ab_extra_s);
            std::fflush(stderr);
        }
    }
    out.total_s = std::chrono::duration<double>(clk::now()-t0).count() - ab_extra_s;
    pool_free(xb); pool_free(srcb);

    out.pcm = std::move(ro);
    out.n_samples = nwav;
    out.audio_s = (double)nwav / 48000.0;
    if (nwav <= 0) { NNOPT_ERROR("run_song: produced no audio"); return false; }
    return true;
}

// ── chunked-vs-whole codec comparison, callable from the serve loop ─────────────────────────────
// The same check the `codecchunk` op-test runs, exposed as a function because the op-test path
// executes once at startup and then RETURNS FROM main — so on a device with no shell (the whole
// 840 story) it was unreachable. A per-request key runs it in-process and keeps serving.
//
// It deliberately bypasses the AR: greedy sampling means one flipped token becomes different music,
// which drowns out exactly the small localised seam error this is looking for. A fixed-seed
// synthetic grid keeps the input constant so any difference is the codec's doing.
bool nnopt_codecchunk_check(OpenCLContext& cl_ctx, Weights& weights, cl_command_queue queue,
                            int T, int CK) {
    if (T <= 0) T = 20;
    if (CK <= 0) CK = 5;
    const int NCB = 12;
    std::vector<int> codes((size_t)T * NCB);
    uint32_t st = 0x9E3779B9u;   // fixed seed: the test must be byte-reproducible
    for (auto& c : codes) { st = st * 1664525u + 1013904223u; c = (int)((st >> 13) % 1024); }

    auto decode = [&](const std::vector<int>& cd, int nf) {
        int nw = 0;
        cl_mem w = run_codec_decode(cl_ctx, weights, queue, cd, nf, &nw, nullptr);
        std::vector<float> h;
        if (w) { h.resize((size_t)nw * 2);
                 clEnqueueReadBuffer(queue, w, CL_TRUE, 0, h.size()*sizeof(float), h.data(), 0, nullptr, nullptr);
                 pool_free(w); }
        return h;
    };

    codec_stream_reset();
    const std::vector<float> ref = decode(codes, T);          // one call, no seams
    if (ref.empty()) { NNOPT_ERROR("codecchunk: whole-grid decode failed (chunk too large for this device?)");
                       return false; }

    codec_stream_reset();
    std::vector<float> got;
    for (int f0 = 0; f0 < T; f0 += CK) {
        const int cf = std::min(CK, T - f0);
        std::vector<int> cc(codes.begin()+(size_t)f0*NCB, codes.begin()+(size_t)(f0+cf)*NCB);
        const std::vector<float> part = decode(cc, cf);
        if (part.empty()) { NNOPT_ERROR_FMT("codecchunk: chunk at frame %d failed", f0); break; }
        got.insert(got.end(), part.begin(), part.end());
    }

    const size_t n = std::min(ref.size(), got.size());
    double maxd = 0.0, sum = 0.0, refmax = 0.0;
    size_t argmax_i = 0;
    for (size_t i = 0; i < n; ++i) {
        const double d = std::fabs((double)ref[i] - (double)got[i]);
        if (d > maxd) { maxd = d; argmax_i = i; }
        sum += d;
        refmax = std::max(refmax, std::fabs((double)ref[i]));
    }
    // Samples per frame, so the worst sample can be named as "frame k", which is what says
    // whether the error sits on a chunk boundary or is spread through the signal.
    const size_t spf = (T > 0) ? (ref.size()/2) / (size_t)T : 0;
    std::fprintf(stderr,
        "CODECCHUNK T=%d chunk=%d stream=%d  ref=%zu got=%zu samples  "
        "max|diff|=%.6g (%.2f%% of peak) at sample %zu = frame %.2f  mean|diff|=%.6g  peak=%.6g  -> %s\n",
        T, CK, codec_stream_enabled() ? 1 : 0, ref.size(), got.size(),
        maxd, refmax > 0 ? 100.0*maxd/refmax : 0.0, argmax_i,
        spf ? (double)(argmax_i/2) / (double)spf : 0.0,
        n ? sum/(double)n : 0.0, refmax,
        (ref.size() == got.size() && maxd <= 1e-5 * (refmax > 0 ? refmax : 1.0)) ? "IDENTICAL" : "DIFFERS");
    std::fflush(stderr);
    return ref.size() == got.size() && maxd <= 1e-5 * (refmax > 0 ? refmax : 1.0);
}

std::vector<float> model_forward_graph(
    OpenCLContext& cl_ctx,
    Weights& weights,
    const std::vector<int32_t>& input_ids,
    int start_pos)
{
    cl_command_queue queue = cl_ctx.queue();

    // ── Op-test harness ── NNOPT_OPTEST="linear|<weight_prefix>|<in_dim>|<rows>"
    // Reads weights/optest_input.bin (fp32, rows*in_dim), runs the op, dumps
    // layer_dumps/optest_out.bin for host cosine-vs-reference. Validates any op
    // type standalone, decoupled from chain wiring.
    if (const char* ot = std::getenv("NNOPT_OPTEST")) {
        std::string s(ot); auto p1 = s.find('|'), p2 = s.find('|', p1+1), p3 = s.find('|', p2+1);
        std::string op = s.substr(0, p1), wp = s.substr(p1+1, p2-p1-1);
        int in_dim = std::atoi(s.substr(p2+1, p3-p2-1).c_str());
        int rows = (p3==std::string::npos) ? 1 : std::atoi(s.substr(p3+1).c_str());
        std::vector<float> xin((size_t)rows*in_dim);
        FILE* xf = std::fopen("weights/optest_input.bin", "rb");
        if (xf) { size_t rd = std::fread(xin.data(), sizeof(float), xin.size(), xf); std::fclose(xf); (void)rd; }
        cl_int e2; cl_mem xb = pool_alloc(cl_ctx.context(), CL_MEM_READ_ONLY|CL_MEM_COPY_HOST_PTR,
                                              xin.size()*sizeof(float), xin.data(), &e2);
        // rvqembed: codes (weights/optest_codes.bin, T*12 int32) → embeddings [T,256]. wp = T.
        if (op == "rvqembed") {
            int T = wp.empty() ? 2 : std::atoi(wp.c_str());
            const int NCB=12, CB=1024, dim=256;
            std::vector<int32_t> codes((size_t)T*NCB);
            FILE* cf=std::fopen("weights/optest_codes.bin","rb");
            if (cf){ size_t rd=std::fread(codes.data(),sizeof(int32_t),codes.size(),cf); std::fclose(cf); (void)rd; }
            cl_int e; cl_mem cb=pool_alloc(cl_ctx.context(),CL_MEM_READ_ONLY|CL_MEM_COPY_HOST_PTR,codes.size()*sizeof(int32_t),codes.data(),&e);
            cl_mem tbl=weights.get_buffer("spectrostream.quantizer.embedding");
            cl_mem ob=pool_alloc(cl_ctx.context(),CL_MEM_READ_WRITE,(size_t)T*dim*sizeof(float),nullptr,&e);
            static cl_kernel rk=nullptr;
            if(!rk){ cl_program p=cl_ctx.build_program_from_file("kernels/rvq_embed_f16.cl"); rk=clCreateKernel(p,"rvq_embed_f16",&e); }
            clSetKernelArg(rk,0,sizeof(cl_mem),&tbl); clSetKernelArg(rk,1,sizeof(cl_mem),&cb); clSetKernelArg(rk,2,sizeof(cl_mem),&ob);
            clSetKernelArg(rk,3,sizeof(int),&T); clSetKernelArg(rk,4,sizeof(int),&NCB); clSetKernelArg(rk,5,sizeof(int),&CB); clSetKernelArg(rk,6,sizeof(int),&dim);
            size_t g[2]={(size_t)T,(size_t)dim}; cl_ctx.profEnqueue(rk,2,g,nullptr,"rvq"); 
            std::vector<float> ro((size_t)T*dim); clEnqueueReadBuffer(queue,ob,CL_TRUE,0,ro.size()*sizeof(float),ro.data(),0,nullptr,nullptr);
            FILE* of=std::fopen("layer_dumps/optest_out.bin","wb"); if(of){std::fwrite(ro.data(),sizeof(float),ro.size(),of); std::fclose(of);}
            pool_free(cb); pool_free(ob); pool_free(xb);
            return std::vector<float>(1,0.0f);
        }
        // codecl1: decoder layer-1 residual block. input codec_dec_l0 [T=2,F=1,Cin=256] → [2,1,2560].
        if (op == "codecl1") {
            cl_mem out = run_codec_l1(cl_ctx, weights, queue, xb, 2, 1, 256);
            std::vector<float> ro((size_t)2*1*2560);
            if (out) clEnqueueReadBuffer(queue,out,CL_TRUE,0,ro.size()*sizeof(float),ro.data(),0,nullptr,nullptr);
            FILE* of=std::fopen("layer_dumps/optest_out.bin","wb"); if(of){std::fwrite(ro.data(),sizeof(float),ro.size(),of); std::fclose(of);}
            if (out) pool_free(out);
            pool_free(xb); return std::vector<float>(1,0.0f);
        }
        // song: full end-to-end through the PRODUCTION path (run_song). wp = N frames.
        // Conditioning is loaded by the style loader, NOT from the generic optest_input buffer:
        // a two-field OPTEST string ("song|100") leaves in_dim=0, so `xb` above is an EMPTY
        // buffer — the pre-2026-08 e2e runs were conditioned on an uninitialised temporal input.
        // Timings were unaffected (the work is data-independent) but the audio was not a real render.
        if (op == "song") {
            int N = wp.empty() ? 50 : std::atoi(wp.c_str());
            SongConfig cfg;
            cfg.n_frames = N;
            if (const char* ce = std::getenv("NNOPT_CHUNK")) { int v = std::atoi(ce); if (v > 0) cfg.chunk = v; }
            cfg.pipeline = std::getenv("NNOPT_PIPELINE") != nullptr;
            const char* style_env = std::getenv("NNOPT_STYLE");
            SongConditioning cond; std::string why;
            if (!song_load_conditioning(style_env ? style_env : "", cond, why)) {
                NNOPT_ERROR_FMT("song: conditioning load failed — %s", why.c_str());
                pool_free(xb); return std::vector<float>(1, 0.0f);
            }
            std::fprintf(stderr, "song: conditioning — %s\n", why.c_str());
            SongResult r;
            const bool ok = run_song(cl_ctx, weights, cond, cfg, r);
            if (ok) song_write_wav("layer_dumps/out.wav", r.pcm, r.n_samples);
            std::printf("\n=== SONG (%d frames%s) ===\n", N, cfg.pipeline?", PIPELINED AR∥codec":"");
            if (!cfg.pipeline && r.ar_s > 0.0) {
                std::printf("  AR generation: %.2fs  (%.1f frames/s, %.0f tokens/s)\n", r.ar_s, N/r.ar_s, N*12/r.ar_s);
                std::printf("  Codec decode : %.2fs\n", r.codec_s);
            }
            std::printf("  TOTAL        : %.2fs  -> %.2fs audio  (RTF=%.2f, %s)\n",
                        r.total_s, r.audio_s, r.rtf(), r.total_s < r.audio_s ? "REALTIME" : "slower-than-RT");
            std::fflush(stdout);
            cl_ctx.compReport();
            cl_ctx.profReport();
            pool_free(xb); return std::vector<float>(1, 0.0f);
        }
        // ── codecchunk|T|CHUNK ── the codec's chunking is supposed to be a SLICING decision, not a
        // numerical one: the same token grid decoded whole and decoded in pieces must produce the
        // same samples. It does not. This test is the proof, and the regression guard for the fix.
        //
        // It deliberately does NOT go through the AR. Sampling is greedy, so an AR-level change
        // flips a token and the waveform is different music — which drowns out exactly the kind of
        // small, localised error this is looking for. A fixed synthetic grid keeps the input
        // constant so any difference in the output is the codec's doing and nothing else.
        if (op == "codecchunk") {
            nnopt_codecchunk_check(cl_ctx, weights, queue,
                                   wp.empty() ? 20 : std::atoi(wp.c_str()), in_dim > 0 ? in_dim : 5);
            pool_free(xb); return std::vector<float>(1,0.0f);
        }


        // ── tokdiff|T ── int8 vs fp16 compared where it can actually be compared: the tokens.
        // Renders the same prompt twice, once at each end of NNOPT_INT8SCOPE, and diffs the grids.
        // Identical tokens => bit-identical audio, and int8 is proved safe rather than argued safe.
        // Different tokens => the count and the first divergence, instead of "the wav looks different".
        if (op == "tokdiff") {
            const int N = wp.empty() ? 50 : std::atoi(wp.c_str());
            SongConditioning cond; std::string why;
            const char* style_env = std::getenv("NNOPT_STYLE");
            if (!song_load_conditioning(style_env ? style_env : "", cond, why)) {
                NNOPT_ERROR_FMT("tokdiff: conditioning load failed — %s", why.c_str());
                pool_free(xb); return std::vector<float>(1,0.0f);
            }
            auto render = [&](int scope, std::vector<int>& tok) {
                char buf[8]; std::snprintf(buf, sizeof(buf), "%d", scope);
                setenv("NNOPT_INT8SCOPE", buf, 1);
                nnopt_toggle_bump();               // make every cached env lookup re-read
                SongConfig cfg; cfg.n_frames = N; cfg.pipeline = false;
                SongResult r;
                const bool ok = run_song(cl_ctx, weights, cond, cfg, r);
                tok = r.tokens;
                return ok;
            };
            std::vector<int> t16, t8;
            const bool ok16 = render(0, t16);
            const bool ok8  = render(4, t8);
            unsetenv("NNOPT_INT8SCOPE"); nnopt_toggle_bump();
            if (!ok16 || !ok8 || t16.empty() || t16.size() != t8.size()) {
                NNOPT_ERROR_FMT("tokdiff: render failed (fp16=%d int8=%d, %zu vs %zu tokens)",
                                (int)ok16, (int)ok8, t16.size(), t8.size());
                pool_free(xb); return std::vector<float>(1,0.0f);
            }
            size_t same = 0, first = t16.size();
            for (size_t i = 0; i < t16.size(); ++i) {
                if (t16[i] == t8[i]) ++same;
                else if (first == t16.size()) first = i;
            }
            std::fprintf(stderr,
                "TOKDIFF frames=%d tokens=%zu identical=%zu (%.2f%%) first_divergence=%s%zu (frame %zu, codebook %zu)\n",
                N, t16.size(), same, 100.0*(double)same/(double)t16.size(),
                first == t16.size() ? "none at " : "", first, first/12, first%12);
            std::fflush(stderr);
            pool_free(xb); return std::vector<float>(1, 0.0f);
        }

        // gemvbench: hammer the depth qkv GEMV (W[2304,768] fp16) N times; report GPU ms/call vs the
        // memory roofline (W bytes / ~10GB/s) to see if we're memory-saturated. NNOPT_OPTEST="gemvbench|N|1|1"
        if (op == "gemvbench") {
            int Nruns = wp.empty()?2000:std::atoi(wp.c_str());
            const std::string wk="depthformer.decoder.depth_body.layers.1.layers.0.layers.0.body.layers.1.inner.qkv_proj.weight";
            cl_mem W=weights.get_buffer(wk); const std::vector<int> sh=weights.get_shape(wk);
            const int outd=sh[0], ind=sh[1];  // 2304 x 768
            cl_int e; std::vector<float> xin(ind, 0.01f);
            cl_mem xb2=pool_alloc(cl_ctx.context(),CL_MEM_READ_ONLY|CL_MEM_COPY_HOST_PTR,ind*sizeof(float),xin.data(),&e);
            cl_mem ob=pool_alloc(cl_ctx.context(),CL_MEM_READ_WRITE,(size_t)outd*sizeof(float),nullptr,&e);
            static cl_kernel gk2=nullptr;
            if(!gk2){ cl_program p=cl_ctx.build_program_from_file("kernels/linear_f32_w16.cl"); gk2=clCreateKernel(p,"linear_f32_w16",&e); }
            clSetKernelArg(gk2,0,sizeof(cl_mem),&xb2); clSetKernelArg(gk2,1,sizeof(cl_mem),&W); clSetKernelArg(gk2,2,sizeof(cl_mem),&ob);
            clSetKernelArg(gk2,3,sizeof(int),&ind); clSetKernelArg(gk2,4,sizeof(int),&outd);
            const size_t nwg=((size_t)outd+3)/4; size_t g[2]={1,nwg*64}, lws[2]={1,64};
            clFinish(queue); using clk=std::chrono::steady_clock; auto t0=clk::now();
            for(int r=0;r<Nruns;r++) clEnqueueNDRangeKernel(queue,gk2,2,nullptr,g,lws,0,nullptr,nullptr);
            clFinish(queue); double ms=std::chrono::duration<double,std::milli>(clk::now()-t0).count()/Nruns;
            double wbytes=(double)outd*ind*2.0; double roof_ms=wbytes/10e9*1000.0;  // ~10 GB/s
            std::printf("\n=== GEMV BENCH (W[%d,%d] fp16, %d iters) ===\n", outd, ind, Nruns);
            std::printf("  [wave64x4 baseline] %.4f ms | util %.0f%%\n", ms, 100.0*roof_ms/ms);
            // sweep WGSZ x NOUT on the parameterized kernel
            const int cfgs[][2]={{64,8},{64,12},{64,16},{64,24},{64,32},{32,16},{128,16},{96,16}};
            for (auto& c : cfgs) {
                int W_=c[0], O_=c[1];
                char opt[64]; std::snprintf(opt,sizeof(opt),"-D WGSZ=%d -D NOUT=%d",W_,O_);
                cl_program p=cl_ctx.build_program_from_file("kernels/gemv_param_f16.cl",opt);
                if(!p) continue; cl_kernel pk=clCreateKernel(p,"gemv_param_f16",&e); if(!pk) continue;
                clSetKernelArg(pk,0,sizeof(cl_mem),&xb2); clSetKernelArg(pk,1,sizeof(cl_mem),&W); clSetKernelArg(pk,2,sizeof(cl_mem),&ob);
                clSetKernelArg(pk,3,sizeof(int),&ind); clSetKernelArg(pk,4,sizeof(int),&outd);
                size_t nwg2=((size_t)outd+O_-1)/O_; size_t gp[2]={1,nwg2*W_}, lp[2]={1,(size_t)W_};
                if(clEnqueueNDRangeKernel(queue,pk,2,nullptr,gp,lp,0,nullptr,nullptr)!=CL_SUCCESS){ std::printf("  [wg%dx%d] FAILED\n",W_,O_); clReleaseKernel(pk); clReleaseProgram(p); continue; }
                clFinish(queue); auto ta=clk::now();
                for(int r=0;r<Nruns;r++) clEnqueueNDRangeKernel(queue,pk,2,nullptr,gp,lp,0,nullptr,nullptr);
                clFinish(queue); double msp=std::chrono::duration<double,std::milli>(clk::now()-ta).count()/Nruns;
                std::printf("  [wg%-3d x %d out] %.4f ms | util %.0f%%\n", W_,O_,msp, 100.0*roof_ms/msp);
                clReleaseKernel(pk); clReleaseProgram(p);
            }
            // int8-weight gemv probe (values arbitrary — timing only; access pattern is data-independent).
            // Half the weight bytes of fp16. If int8 ms < fp16 ms ⇒ AR gemv is DRAM-bound ⇒ int8 AR worth
            // building. If int8 ms ≈ or > fp16 ⇒ issue/occupancy-bound ⇒ quant dead (reconfirmed).
            {
                std::vector<signed char> w8((size_t)outd*ind, 1);
                std::vector<float> rs(outd, 1.0f);
                cl_mem W8=pool_alloc(cl_ctx.context(),CL_MEM_READ_ONLY|CL_MEM_COPY_HOST_PTR,w8.size(),w8.data(),&e);
                cl_mem rsb=pool_alloc(cl_ctx.context(),CL_MEM_READ_ONLY|CL_MEM_COPY_HOST_PTR,rs.size()*sizeof(float),rs.data(),&e);
                cl_kernel gk8=nullptr; cl_program p8=cl_ctx.build_program_from_file("kernels/gemv_w8.cl");
                if(p8) gk8=clCreateKernel(p8,"gemv_w8",&e);
                if(gk8){
                    clSetKernelArg(gk8,0,sizeof(cl_mem),&xb2); clSetKernelArg(gk8,1,sizeof(cl_mem),&W8); clSetKernelArg(gk8,2,sizeof(cl_mem),&rsb);
                    clSetKernelArg(gk8,3,sizeof(cl_mem),&ob); clSetKernelArg(gk8,4,sizeof(int),&ind); clSetKernelArg(gk8,5,sizeof(int),&outd);
                    size_t nwg8=((size_t)outd+7)/8; size_t g8[2]={1,nwg8*64}, l8[2]={1,64};
                    clEnqueueNDRangeKernel(queue,gk8,2,nullptr,g8,l8,0,nullptr,nullptr); clFinish(queue);
                    auto t8=clk::now();
                    for(int r=0;r<Nruns;r++) clEnqueueNDRangeKernel(queue,gk8,2,nullptr,g8,l8,0,nullptr,nullptr);
                    clFinish(queue); double ms8=std::chrono::duration<double,std::milli>(clk::now()-t8).count()/Nruns;
                    double roof8=(double)outd*ind*1.0/10e9*1000.0;
                    std::printf("  [int8-weight wg64x8] %.4f ms | util %.0f%% | vs fp16 production: %.2fx (>1 = int8 faster)\n",
                                ms8, 100.0*roof8/ms8, ms/ms8);
                    clReleaseKernel(gk8);
                }
                if(p8) clReleaseProgram(p8);
                pool_free(W8); pool_free(rsb);
            }
            std::fflush(stdout);
            pool_free(xb2); pool_free(ob); pool_free(xb);
            return std::vector<float>(1,0.0f);
        }
        // gemmbench: CLBlast Sgemm (fp32) vs Hgemm (fp16) on a representative codec conv GEMM shape —
        // decides whether partial-fp16 codec (HGEMM on the fp16-safe shallow layers) is worth building.
        // NNOPT_OPTEST="gemmbench|<iters>". Values arbitrary (timing only; data-independent).
        if (op == "gemmbench") {
#ifdef USE_CLBLAST
            int Nr = wp.empty()?200:std::atoi(wp.c_str());
            const int M=8192, N=256, K=2304;   // ~ a mid/late l5 conv GEMM (M=B*Tout*Fout, N=Cout, K=kH*kW*Cin)
            cl_int e; cl_command_queue q=queue;
            cl_mem A=pool_alloc(cl_ctx.context(),CL_MEM_READ_WRITE,(size_t)M*K*sizeof(float),nullptr,&e);
            cl_mem Bb=pool_alloc(cl_ctx.context(),CL_MEM_READ_WRITE,(size_t)N*K*sizeof(float),nullptr,&e);
            cl_mem C=pool_alloc(cl_ctx.context(),CL_MEM_READ_WRITE,(size_t)M*N*sizeof(float),nullptr,&e);
            using clk=std::chrono::steady_clock;
            CLBlastSgemm(CLBlastLayoutRowMajor,CLBlastTransposeNo,CLBlastTransposeYes,M,N,K,1.0f,A,0,K,Bb,0,K,0.0f,C,0,N,&q,nullptr); clFinish(q);
            auto ts=clk::now();
            for(int r=0;r<Nr;r++) CLBlastSgemm(CLBlastLayoutRowMajor,CLBlastTransposeNo,CLBlastTransposeYes,M,N,K,1.0f,A,0,K,Bb,0,K,0.0f,C,0,N,&q,nullptr);
            clFinish(q); double ms_s=std::chrono::duration<double,std::milli>(clk::now()-ts).count()/Nr;
            cl_mem Ah=pool_alloc(cl_ctx.context(),CL_MEM_READ_WRITE,(size_t)M*K*sizeof(cl_half),nullptr,&e);
            cl_mem Bh=pool_alloc(cl_ctx.context(),CL_MEM_READ_WRITE,(size_t)N*K*sizeof(cl_half),nullptr,&e);
            cl_mem Ch=pool_alloc(cl_ctx.context(),CL_MEM_READ_WRITE,(size_t)M*N*sizeof(cl_half),nullptr,&e);
            const cl_half h1=0x3C00, h0=0x0000;  // fp16 1.0, 0.0
            CLBlastStatusCode st=CLBlastHgemm(CLBlastLayoutRowMajor,CLBlastTransposeNo,CLBlastTransposeYes,M,N,K,h1,Ah,0,K,Bh,0,K,h0,Ch,0,N,&q,nullptr);
            if(st!=CLBlastSuccess){ std::printf("\n=== GEMM BENCH === Hgemm unsupported (st=%d); fp32 Sgemm=%.3f ms\n",(int)st,ms_s); }
            else { clFinish(q); auto th=clk::now();
                for(int r=0;r<Nr;r++) CLBlastHgemm(CLBlastLayoutRowMajor,CLBlastTransposeNo,CLBlastTransposeYes,M,N,K,h1,Ah,0,K,Bh,0,K,h0,Ch,0,N,&q,nullptr);
                clFinish(q); double ms_h=std::chrono::duration<double,std::milli>(clk::now()-th).count()/Nr;
                std::printf("\n=== GEMM BENCH (M=%d N=%d K=%d, %d iters) ===\n  Sgemm fp32: %.3f ms\n  Hgemm fp16: %.3f ms  (%.2fx vs fp32)\n",M,N,K,Nr,ms_s,ms_h,ms_s/ms_h); }
            std::fflush(stdout);
            pool_free(A);pool_free(Bb);pool_free(C);pool_free(Ah);pool_free(Bh);pool_free(Ch);
#endif
            pool_free(xb); return std::vector<float>(1,0.0f);
        }
        // codecbench: run the codec N times on weights/optest_codes.bin (T frames). Reports per-run
        // component timing for runs 1 (cold, incl CLBlast JIT) vs 2..N (warm). NNOPT_OPTEST="codecbench|<T>|<N>|1"
        if (op == "codecbench") {
            int T = wp.empty()?2:std::atoi(wp.c_str());
            int Nruns = std::max(1, rows);
            std::vector<int32_t> codes((size_t)T*12);
            FILE* cf=std::fopen("weights/optest_codes.bin","rb"); if(cf){size_t rd=std::fread(codes.data(),sizeof(int32_t),codes.size(),cf);std::fclose(cf);(void)rd;}
            std::vector<int> cv(codes.begin(),codes.end());
            using clk=std::chrono::steady_clock;
            for (int r=0; r<Nruns; ++r) {
                clFinish(queue);
                auto t0=clk::now();
                int nw=0; cl_mem w=run_codec_decode(cl_ctx,weights,queue,cv,T,&nw);
                clFinish(queue);
                double ms=std::chrono::duration<double,std::milli>(clk::now()-t0).count();
                std::printf("  codec run %d (%s): %.1f ms\n", r, r==0?"cold+JIT":"warm", ms);
                if (w) pool_free(w);
            }
            std::fflush(stdout);
            cl_ctx.compReport();
            pool_free(xb); return std::vector<float>(1,0.0f);
        }
        // codecfull: RVQ codes (weights/optest_codes.bin, T*12 int32) → waveform → layer_dumps/out.wav.
        // wp = T (frames). Writes 48kHz stereo 16-bit WAV + raw f32 to optest_out.bin (vs codec_istft).
        if (op == "codecfull") {
            int T = wp.empty() ? 2 : std::atoi(wp.c_str());
            std::vector<int32_t> codes((size_t)T*12);
            FILE* cf=std::fopen("weights/optest_codes.bin","rb"); if(cf){size_t rd=std::fread(codes.data(),sizeof(int32_t),codes.size(),cf);std::fclose(cf);(void)rd;}
            std::vector<int> cv(codes.begin(),codes.end());
            int nwav=0; cl_mem wav=run_codec_decode(cl_ctx,weights,queue,cv,T,&nwav);
            std::vector<float> ro((size_t)nwav*2);
            if (wav) clEnqueueReadBuffer(queue,wav,CL_TRUE,0,ro.size()*sizeof(float),ro.data(),0,nullptr,nullptr);
            if (wav) pool_free(wav);
            FILE* of=std::fopen("layer_dumps/optest_out.bin","wb"); if(of){std::fwrite(ro.data(),sizeof(float),ro.size(),of); std::fclose(of);}
            // WAV (48kHz stereo s16le). i16 = round(32767.5*clip(0.5*x,-1,1) - 0.5).
            const int SR=48000, CH=2; const int nfr=nwav;
            std::vector<int16_t> pcm((size_t)nfr*CH);
            for (size_t i=0;i<ro.size();i++){ float v=0.5f*ro[i]; v=v>1.f?1.f:(v<-1.f?-1.f:v);
                long s=lround(32767.5f*v-0.5f); pcm[i]=(int16_t)(s>32767?32767:(s<-32768?-32768:s)); }
            FILE* wf=std::fopen("layer_dumps/out.wav","wb");
            if (wf){ const int dataBytes=(int)pcm.size()*2, byteRate=SR*CH*2; const int chunk=36+dataBytes;
                auto w32=[&](int v){ unsigned char b[4]={(unsigned char)(v&0xff),(unsigned char)((v>>8)&0xff),(unsigned char)((v>>16)&0xff),(unsigned char)((v>>24)&0xff)}; std::fwrite(b,1,4,wf); };
                auto w16=[&](int v){ unsigned char b[2]={(unsigned char)(v&0xff),(unsigned char)((v>>8)&0xff)}; std::fwrite(b,1,2,wf); };
                std::fwrite("RIFF",1,4,wf); w32(chunk); std::fwrite("WAVE",1,4,wf); std::fwrite("fmt ",1,4,wf);
                w32(16); w16(1); w16(CH); w32(SR); w32(byteRate); w16(CH*2); w16(16);
                std::fwrite("data",1,4,wf); w32(dataBytes); std::fwrite(pcm.data(),1,dataBytes,wf); std::fclose(wf); }
            pool_free(xb); return std::vector<float>(1,0.0f);
        }
        // codecistft: on-device inverse STFT. input codec_decout [T=4,480,4] → waveform [1920,2].
        if (op == "codecistft") {
            cl_mem out = run_istft(cl_ctx, weights, queue, xb, 4, 480, 4);
            std::vector<float> ro((size_t)1920*2);
            if (out) clEnqueueReadBuffer(queue,out,CL_TRUE,0,ro.size()*sizeof(float),ro.data(),0,nullptr,nullptr);
            FILE* of=std::fopen("layer_dumps/optest_out.bin","wb"); if(of){std::fwrite(ro.data(),sizeof(float),ro.size(),of); std::fclose(of);}
            if (out) pool_free(out);
            pool_free(xb); return std::vector<float>(1,0.0f);
        }
        // codecl5: decoder layer-5 ParallelChannels cascade. input codec_dec_l4 [4,5,1024] → [8,480,4].
        if (op == "codecl5") {
            cl_mem out = run_codec_l5(cl_ctx, weights, queue, xb, 4);
            std::vector<float> ro((size_t)8*480*4);
            if (out) clEnqueueReadBuffer(queue,out,CL_TRUE,0,ro.size()*sizeof(float),ro.data(),0,nullptr,nullptr);
            FILE* of=std::fopen("layer_dumps/optest_out.bin","wb"); if(of){std::fwrite(ro.data(),sizeof(float),ro.size(),of); std::fclose(of);}
            if (out) pool_free(out);
            pool_free(xb); return std::vector<float>(1,0.0f);
        }
        // codecl4: decoder layer-4 (conv-transpose). input codec_dec_l3 [2,5,512] → [4,5,1024].
        if (op == "codecl4") {
            cl_mem out = run_codec_l4(cl_ctx, weights, queue, xb, 2, 5, 512);
            std::vector<float> ro((size_t)4*5*1024);
            if (out) clEnqueueReadBuffer(queue,out,CL_TRUE,0,ro.size()*sizeof(float),ro.data(),0,nullptr,nullptr);
            FILE* of=std::fopen("layer_dumps/optest_out.bin","wb"); if(of){std::fwrite(ro.data(),sizeof(float),ro.size(),of); std::fclose(of);}
            if (out) pool_free(out);
            pool_free(xb); return std::vector<float>(1,0.0f);
        }
        // codecl3: decoder layer-3 residual (3x3). input codec_dec_l2 [T=2,F=5,C=512] → [2,5,512].
        if (op == "codecl3") {
            cl_mem out = run_codec_l3(cl_ctx, weights, queue, xb, 2, 5, 512);
            std::vector<float> ro((size_t)2*5*512);
            if (out) clEnqueueReadBuffer(queue,out,CL_TRUE,0,ro.size()*sizeof(float),ro.data(),0,nullptr,nullptr);
            FILE* of=std::fopen("layer_dumps/optest_out.bin","wb"); if(of){std::fwrite(ro.data(),sizeof(float),ro.size(),of); std::fclose(of);}
            if (out) pool_free(out);
            pool_free(xb); return std::vector<float>(1,0.0f);
        }
        // depthloop: input = temporal_out [1024]; runs the 12-codebook greedy AR loop → int32 tokens.
        // depthfixed: reads weights/optest_dinall.bin [12*1024] as fixed per-codebook inputs (no feedback).
        if (op == "depthloop" || op == "depthfixed") {
            std::vector<float> dinall; const std::vector<float>* fx = nullptr;
            if (op == "depthfixed") {
                dinall.resize((size_t)12*1024);
                FILE* df = std::fopen("weights/optest_dinall.bin","rb");
                if (df) { size_t rd=std::fread(dinall.data(),sizeof(float),dinall.size(),df); std::fclose(df); (void)rd; }
                fx = &dinall;
            }
            std::vector<int> toks = run_depth_loop(cl_ctx, weights, queue, xb, fx);
            FILE* of = std::fopen("layer_dumps/optest_out.bin", "wb");
            if (of) { for (int t : toks) { int32_t v = t; std::fwrite(&v, sizeof(int32_t), 1, of); } std::fclose(of); }
            pool_free(xb);
            return std::vector<float>(1, 0.0f);
        }
        // generate: temporal_input_0 (optest_input) + source (optest_source) → AR token grid.
        // NNOPT_OPTEST="generate|<n_frames>|1024|1"
        if (op == "generate") {
            int n_frames = wp.empty() ? 2 : std::atoi(wp.c_str());
            std::vector<float> src(256);
            FILE* sf = std::fopen("weights/optest_source.bin","rb");
            if (sf) { size_t rd=std::fread(src.data(),sizeof(float),src.size(),sf); std::fclose(sf); (void)rd; }
            cl_int es; cl_mem srcb = pool_alloc(cl_ctx.context(), CL_MEM_READ_ONLY|CL_MEM_COPY_HOST_PTR,
                                                    src.size()*sizeof(float), src.data(), &es);
            std::vector<int> grid = run_generate(cl_ctx, weights, queue, xb, srcb, n_frames);
            pool_free(srcb);
            FILE* of = std::fopen("layer_dumps/optest_out.bin", "wb");
            if (of) { for (int t : grid) { int32_t v = t; std::fwrite(&v, sizeof(int32_t), 1, of); } std::fclose(of); }
            pool_free(xb);
            return std::vector<float>(1, 0.0f);
        }
        cl_mem r = nullptr;
        if (op == "linear") r = QuantizedLinear_forward(cl_ctx, weights, queue, xb, rows, 0,0,nullptr,nullptr,nullptr, wp.c_str());
        else if (op == "rmsnorm") r = RMSNorm_forward(cl_ctx, weights, queue, xb, rows, 0,0,nullptr,nullptr,nullptr, wp.c_str());
        else if (op == "sinkattn") r = SinkAttention_forward(cl_ctx, weights, queue, xb, rows, 0,0,nullptr,nullptr,nullptr, wp.c_str());
        else if (op == "mlpcore") r = MLP_forward(cl_ctx, weights, queue, xb, rows, 0,0,nullptr,nullptr,nullptr, wp.c_str());
        else if (op == "temporal") {
            // wp field carries num_layers (e.g. "12"); source from weights/optest_source.bin (256 fp32)
            int num_layers = wp.empty() ? 12 : std::atoi(wp.c_str());
            std::vector<float> src(256);
            FILE* sf = std::fopen("weights/optest_source.bin","rb");
            if (sf) { size_t rd=std::fread(src.data(),sizeof(float),src.size(),sf); std::fclose(sf); (void)rd; }
            cl_int es; cl_mem srcb = pool_alloc(cl_ctx.context(), CL_MEM_READ_ONLY|CL_MEM_COPY_HOST_PTR,
                                                    src.size()*sizeof(float), src.data(), &es);
            r = run_temporal_body(cl_ctx, weights, queue, xb, srcb, in_dim, num_layers);
            pool_free(srcb);
        }
        else if (op == "depth") r = run_depth_body(cl_ctx, weights, queue, xb);
        else if (op == "layernorm") r = LayerNorm_forward(cl_ctx, weights, queue, xb, rows, 0,0,nullptr,nullptr,nullptr, wp.c_str());
        else if (op == "encoder") {
            std::vector<int32_t> tk(144);
            FILE* tf = std::fopen("weights/optest_tokens.bin","rb");
            if (tf) { size_t rd=std::fread(tk.data(),sizeof(int32_t),tk.size(),tf); std::fclose(tf); (void)rd; }
            cl_int et; cl_mem tb = pool_alloc(cl_ctx.context(), CL_MEM_READ_ONLY|CL_MEM_COPY_HOST_PTR,
                                                  tk.size()*sizeof(int32_t), tk.data(), &et);
            r = Encoder_forward(cl_ctx, weights, queue, tb, 1, 0,0,nullptr,nullptr,nullptr, nullptr);
            pool_free(tb);
        }
        if (r) {
            int od;
            if (op == "linear") od = weights.get_shape(wp + ".weight")[0];
            else if (op == "sinkattn") { const std::vector<int>& s = weights.get_shape(wp + ".sink_key_embeddings"); od = s[1]*s[2]; }
            else if (op == "mlpcore") od = weights.get_shape(wp + ".layers.3.inner._linear.weight")[0];
            else if (op == "depth") od = weights.get_shape("depthformer.decoder.depth_body.layers.3.inner._linear.weight")[0];
            else if (op == "encoder") od = weights.get_shape("depthformer.encoder.body.layers.3._layer_norm.weight").back();
            else od = in_dim;
            std::vector<float> ro((size_t)rows*od);
            clEnqueueReadBuffer(queue, r, CL_TRUE, 0, ro.size()*sizeof(float), ro.data(), 0,nullptr,nullptr);
            FILE* of = std::fopen("layer_dumps/optest_out.bin", "wb");
            if (of) { std::fwrite(ro.data(), sizeof(float), ro.size(), of); std::fclose(of); }
            pool_free(r);
        }
        pool_free(xb);
        return std::vector<float>(1, 0.0f);
    }

    const int seq_len = static_cast<int>(input_ids.size());
    if (seq_len <= 0) {
        NNOPT_ERROR("model_forward_graph: empty input_ids");
        return {};
    }

    cl_int err = CL_SUCCESS;
    cl_mem ids_buf = clCreateBuffer(
        cl_ctx.context(), CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
        static_cast<size_t>(seq_len) * sizeof(int32_t),
        const_cast<int32_t*>(input_ids.data()), &err);
    if (err != CL_SUCCESS || !ids_buf) {
        NNOPT_ERROR_FMT("model_forward_graph: ids buffer alloc failed (err=%d)", err);
        return {};
    }

    // ── Node 0: decoder embedder — QuantizedEmbedding (dense fp16 gather) ──
    // Reference: reference/forward_graph.json nodes[0]; out shape [12, 1024].
    cl_mem h = QuantizedEmbedding_forward(
        cl_ctx, weights, queue, ids_buf, seq_len, /*layer_idx=*/0, start_pos,
        nullptr, nullptr, nullptr,
        "depthformer.decoder.embedder.layers.0._embedding");
    if (!h) { pool_free(ids_buf); return {}; }

    const std::vector<int> wshape =
        weights.get_shape("depthformer.decoder.embedder.layers.0._embedding.weight");
    const int embed_dim = (wshape.size() == 2) ? wshape[1] : 1024;

    NNOPT_LAYER_CHECK(
        "sampler.depthformer.decoder.embedder.layers.0._embedding",
        queue, h, static_cast<size_t>(seq_len) * embed_dim);

    // For this incremental milestone the forward returns the embedder output
    // (fp32). Subsequent nodes will consume `h` and the return becomes lm_head.
    std::vector<float> out(static_cast<size_t>(seq_len) * embed_dim);
    clEnqueueReadBuffer(queue, h, CL_TRUE, 0,
                        out.size() * sizeof(float), out.data(), 0, nullptr, nullptr);

    // Hand-port debug: dump the embedder output (fp32) for host cosine-vs-reference.
    { FILE* _df = std::fopen("layer_dumps/embedder0.bin", "wb");
      if (_df) { std::fwrite(out.data(), sizeof(float), out.size(), _df); std::fclose(_df); } }

    // ── temporal_inputs = mean(embedded over Q) ──  embedded is [seq_len, embed_dim]
    // where seq_len here is the Q codebook axis of the single captured frame.
    std::vector<float> mean(embed_dim, 0.0f);
    for (int q = 0; q < seq_len; ++q)
        for (int d = 0; d < embed_dim; ++d) mean[d] += out[(size_t)q * embed_dim + d];
    for (int d = 0; d < embed_dim; ++d) mean[d] /= static_cast<float>(seq_len);
    cl_mem mean_buf = pool_alloc(cl_ctx.context(), CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
                                     mean.size() * sizeof(float), mean.data(), &err);

    // ── temporal_body block 0: RMSNorm ──
    cl_mem rn = RMSNorm_forward(
        cl_ctx, weights, queue, mean_buf, /*seq_len=*/1, 0, start_pos, nullptr, nullptr, nullptr,
        "depthformer.decoder.temporal_body.layers.0.layers.0.layers.0.body.layers.0._rms_norm");
    if (rn) {
        std::vector<float> rno(embed_dim);
        clEnqueueReadBuffer(queue, rn, CL_TRUE, 0, rno.size() * sizeof(float), rno.data(), 0, nullptr, nullptr);
        FILE* _rf = std::fopen("layer_dumps/rmsnorm0.bin", "wb");
        if (_rf) { std::fwrite(rno.data(), sizeof(float), rno.size(), _rf); std::fclose(_rf); }
        pool_free(rn);
    }
    pool_free(mean_buf);

    pool_free(h);
    pool_free(ids_buf);
    return out;
}
