// Encoder.cpp — DepthFormer conditioning encoder body (branched embedding → MEAN → LayerNorm).
// Reference: magenta_rt/mlx/model.py branch_config + musiccoca_embedder + MultiChannelEmbedding.
//
// Input: 144 int32 conditioning tokens. Output: [256] conditioning vector (== cross-attn source).
//   branch0 (mulan, 12 tokens): idx = tok[k] + k*1031; gather emb0[12372,768]; SUM over 12 → [768];
//                               dense 768→256 (no bias) → b0
//   branch1 (132 channels): idx = tok[12+c] + offset[c] (per-channel cumsum); gather emb1[1536,256];
//                           MEAN over 132 → b1
//   out = LayerNorm((b0+b1)/2)   — LayerNorm is scale-invariant, so the /2 is dropped (use b0+b1).
//   per-channel vocab sizes: pianoroll[11]*128, drum[9], cfg[47]*2, cfg_drums[15]  (sum 1526 ≤ 1536).

#include "../opencl_context.h"
#include "../weights.h"
#include "../debug_utils.h"
#include "../model_config.h"
#include "../utils.h"
#include <string>
#include <vector>
#include <numeric>

extern "C" cl_mem LayerNorm_forward(
    OpenCLContext&, Weights&, cl_command_queue, cl_mem, int, int, int,
    cl_mem*, cl_mem*, cl_mem, const char*);

extern "C" cl_mem Encoder_forward(
    OpenCLContext& cl_ctx, Weights& weights, cl_command_queue queue,
    cl_mem tokens /*int32 [144]*/, int seq_len, int layer_idx, int start_pos,
    cl_mem* k_cache_inout, cl_mem* v_cache_inout, cl_mem encoder_hidden_states,
    const char* weight_prefix)
{
    (void)seq_len;(void)layer_idx;(void)start_pos;(void)k_cache_inout;(void)v_cache_inout;
    (void)encoder_hidden_states;(void)weight_prefix;
    const std::string E = "depthformer.encoder.body.";
    cl_mem emb0 = weights.get_buffer(E+"layers.0.layers.0.layers.1.layers.1._embedding.weight"); // [12372,768]
    cl_mem Wd   = weights.get_buffer(E+"layers.0.layers.0.layers.1.layers.3.inner._linear.weight"); // [256,768]
    cl_mem emb1 = weights.get_buffer(E+"layers.0.layers.1.layers.1.embedding");                  // [1536,256]
    if (!emb0 || !Wd || !emb1) { NNOPT_ERROR("Encoder: missing embedding weights"); return nullptr; }
    const int dim0 = weights.get_shape(E+"layers.0.layers.0.layers.1.layers.1._embedding.weight")[1]; // 768
    const int out_dim = weights.get_shape(E+"layers.0.layers.0.layers.1.layers.3.inner._linear.weight")[0]; // 256
    const int dim1 = weights.get_shape(E+"layers.0.layers.1.layers.1.embedding")[1]; // 256

    // offsets — branch0: k*1031 (k=0..11); branch1: cumsum of per-channel vocab sizes.
    std::vector<int> off0(12); for (int k=0;k<12;k++) off0[k]=k*1031;
    std::vector<int> npc; for (int i=0;i<128;i++) npc.push_back(11); npc.push_back(9); npc.push_back(47); npc.push_back(47); npc.push_back(15);
    std::vector<int> off1(npc.size(), 0); for (size_t c=1;c<npc.size();c++) off1[c]=off1[c-1]+npc[c-1];
    const int N1 = (int)npc.size(); // 132

    cl_int err = CL_SUCCESS;
    auto upload_int = [&](const std::vector<int>& v)->cl_mem{
        return pool_alloc(cl_ctx.context(), CL_MEM_READ_ONLY|CL_MEM_COPY_HOST_PTR,
                              v.size()*sizeof(int), const_cast<int*>(v.data()), &err);
    };
    cl_mem off0b = upload_int(off0), off1b = upload_int(off1);

    static cl_kernel gr = nullptr, lk = nullptr, add = nullptr;
    if (!gr) { cl_program p=cl_ctx.build_program_from_file("kernels/gather_reduce_f16.cl");
        if(!p){NNOPT_ERROR("Encoder: build gather_reduce");return nullptr;} gr=clCreateKernel(p,"gather_reduce_f16",&err); }
    if (!lk) { cl_program p=cl_ctx.build_program_from_file("kernels/linear_f32_w16.cl");
        if(!p){NNOPT_ERROR("Encoder: build linear");return nullptr;} lk=clCreateKernel(p,"linear_f32_w16",&err); }
    if (!add){ cl_program p=cl_ctx.build_program_from_file("kernels/add_f32.cl");
        if(!p){NNOPT_ERROR("Encoder: build add");return nullptr;} add=clCreateKernel(p,"add_f32",&err); }
    if (!gr||!lk||!add) { NNOPT_ERROR("Encoder: kernel create"); return nullptr; }
    (void)Wd;   // presence-checked above; the dense now goes through nnopt_gemv

    auto gather = [&](cl_mem table, int tok_start, cl_mem offs, int N, int dim, float scale)->cl_mem{
        cl_mem o = pool_alloc(cl_ctx.context(), CL_MEM_READ_WRITE, (size_t)dim*sizeof(float), nullptr, &err);
        clSetKernelArg(gr,0,sizeof(cl_mem),&table); clSetKernelArg(gr,1,sizeof(cl_mem),&tokens);
        clSetKernelArg(gr,2,sizeof(int),&tok_start); clSetKernelArg(gr,3,sizeof(cl_mem),&offs);
        clSetKernelArg(gr,4,sizeof(cl_mem),&o); clSetKernelArg(gr,5,sizeof(int),&N);
        clSetKernelArg(gr,6,sizeof(int),&dim); clSetKernelArg(gr,7,sizeof(float),&scale);
        size_t g=(size_t)dim;
        if(clEnqueueNDRangeKernel(queue,gr,1,nullptr,&g,nullptr,0,nullptr,nullptr)!=CL_SUCCESS){pool_free(o);return nullptr;}
        return o;
    };
    // branch0: sum over 12 → e0[768]; dense → b0[256]
    cl_mem e0 = gather(emb0, 0, off0b, 12, dim0, 1.0f);
    if (!e0) { NNOPT_ERROR("Encoder: branch0 gather"); return nullptr; }
    // Through nnopt_gemv, NOT a hand-rolled fp16 dispatch. This site used to bind `Wd` straight to
    // linear_f32_w16, which is correct only while the bundle is fp16 — and the q4 bundle packs THIS
    // tensor (`..._linear.weight -> q4_packed [256,384]` with a `.weight.scale` sibling). The fp16
    // kernel then read 4-bit nibbles through vload_half AND ran off the end of every row, because
    // the packed row is 384 bytes where 768 halves were assumed. The result was a conditioning
    // vector of 256 NaNs, which propagated through cross-attention to NaN logits, and argmax over
    // NaN returns index 0 — so every codebook of every frame sampled code 0 and the model emitted a
    // fixed drone that ignored the prompt entirely. It reproduced identically on Adreno 840 and
    // Apple M1, which is what proved it was never a GPU bug.
    //
    // nnopt_gemv picks the kernel from the tensor's own `.scale` presence, so it is right for both
    // bundles and stays right for whatever the quantizer packs next.
    cl_mem b0 = nnopt_gemv(cl_ctx, weights, queue, e0,
                           E+"layers.0.layers.0.layers.1.layers.3.inner._linear.weight",
                           1, dim0, out_dim, nullptr);
    pool_free(e0);
    if (!b0) { NNOPT_ERROR("Encoder: branch0 dense"); return nullptr; }
    // branch1: mean over 132 → b1[256]
    cl_mem b1 = gather(emb1, 12, off1b, N1, dim1, 1.0f/(float)N1);
    if (!b1) { NNOPT_ERROR("Encoder: branch1 gather"); pool_free(b0); return nullptr; }
    // s = b0 + b1
    cl_mem s = pool_alloc(cl_ctx.context(), CL_MEM_READ_WRITE, (size_t)out_dim*sizeof(float), nullptr, &err);
    clSetKernelArg(add,0,sizeof(cl_mem),&b0); clSetKernelArg(add,1,sizeof(cl_mem),&b1);
    clSetKernelArg(add,2,sizeof(cl_mem),&s); clSetKernelArg(add,3,sizeof(int),&out_dim);
    size_t sg=(size_t)out_dim; clEnqueueNDRangeKernel(queue,add,1,nullptr,&sg,nullptr,0,nullptr,nullptr);
    
    pool_free(b0); pool_free(b1); pool_free(off0b); pool_free(off1b);
    // LayerNorm (scale-invariant ⇒ b0+b1 fine)
    cl_mem out = LayerNorm_forward(cl_ctx,weights,queue,s,1,0,0,nullptr,nullptr,nullptr,(E+"layers.3._layer_norm").c_str());
    pool_free(s);
    return out;
}
