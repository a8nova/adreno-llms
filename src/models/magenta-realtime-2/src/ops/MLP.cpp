// MLP.cpp — DenseDeferred FFN core of a transformer Residual block.
// Reference: magenta_rt depthformer temporal/depth body MLP sub-block.
//
// Empirically pinned (node 9 / node 10 reference dumps):
//   dense1 = x @ W1.T + b1   (1024 -> 4096, _use_bias=True)
//   h      = GELU_erf(dense1)
//   dense2 = h @ W2.T + b2    (4096 -> 1024, _use_bias=True)
// Caller applies the surrounding rms1/rms2 + residual (rms norms reuse RMSNorm op).
//
// weight_prefix = the block body prefix, e.g.
//   "...layers.2.body"  -> .layers.1.inner._linear (dense1), .layers.3.inner._linear (dense2)

#include "../opencl_context.h"
#include "../weights.h"
#include "../debug_utils.h"
#include "../model_config.h"
#include "../utils.h"
#include <string>
#include <cstdlib>

static cl_kernel g_lb = nullptr, g_gelu = nullptr;

static cl_mem dense_bias(OpenCLContext& cl_ctx, cl_command_queue queue, Weights& weights,
                         cl_mem x, int rows, const std::string& wp,
                         const std::string& rms_key = std::string(), bool apply_gelu = false) {
    cl_mem b = weights.get_buffer(wp + ".bias");
    if (!b) { NNOPT_ERROR_FMT("MLP: missing %s.bias", wp.c_str()); return nullptr; }
    int out_dim = 0, in_dim = 0;
    if (!nnopt_weight_dims(weights, wp + ".weight", &out_dim, &in_dim)) return nullptr;
    // The MLP denses are the AR's heaviest weights — nnopt_gemv picks fp16 or Q4 per bundle.
    // Always go through the fused entry point, even when nothing is folded in: it is also where the
    // int8 depth-MLP path lives, and dense2 needs it as much as dense1 (they are the same 4.72 MB
    // each). With no prenorm and no gelu it compiles to the same loop as the plain kernel.
    return nnopt_gemv_fused(cl_ctx, weights, queue, x, wp + ".weight", rows, in_dim, out_dim, b,
                            rms_key, apply_gelu);
}

extern "C" cl_mem MLP_forward(
    OpenCLContext& cl_ctx, Weights& weights, cl_command_queue queue,
    cl_mem input, int seq_len, int layer_idx, int start_pos,
    cl_mem* k_cache_inout, cl_mem* v_cache_inout, cl_mem encoder_hidden_states,
    const char* weight_prefix, const char* prenorm_prefix)
{
    (void)layer_idx;(void)start_pos;(void)k_cache_inout;(void)v_cache_inout;(void)encoder_hidden_states;
    const std::string wp = weight_prefix ? std::string(weight_prefix) : std::string();
    if (wp.empty() || !input) { NNOPT_ERROR("MLP: null wp/input"); return nullptr; }
    cl_int err = CL_SUCCESS;
    if (!g_lb) {
        cl_program p = cl_ctx.build_program_from_file("kernels/linear_bias_f32.cl");
        if (!p) { NNOPT_ERROR("MLP: build linear_bias_f32 failed"); return nullptr; }
        g_lb = clCreateKernel(p,"linear_bias_f32",&err);
        if (err!=CL_SUCCESS||!g_lb){ NNOPT_ERROR_FMT("MLP: lb kernel (err=%d)",err); return nullptr; }
    }
    if (!g_gelu) {
        cl_program p = cl_ctx.build_program_from_file("kernels/gelu_f32.cl");
        if (!p) { NNOPT_ERROR("MLP: build gelu_f32 failed"); return nullptr; }
        g_gelu = clCreateKernel(p,"gelu_f32",&err);
        if (err!=CL_SUCCESS||!g_gelu){ NNOPT_ERROR_FMT("MLP: gelu kernel (err=%d)",err); return nullptr; }
    }
    // dense1 (-> 4096), with the caller's RMS pre-norm and the GELU folded into the same dispatch.
    // NNOPT_FUSE=0 restores the separate rms / gemv / gelu kernels for A/B.
    // 0 = none, 1 = GELU epilogue only (default), 2 = GELU + RMS pre-norm. Read through
    // nnopt_fuse_level() so this can never disagree with mlp_block's copy of the decision — when it
    // did, mlp_block skipped materialising rms1 while this kept rms_key empty and the normalisation
    // was applied nowhere at all.
    const int fuse = nnopt_fuse_level();
    const std::string rms_key =
        (fuse >= 2 && prenorm_prefix) ? std::string(prenorm_prefix) + ".weight" : std::string();
    cl_mem d1 = dense_bias(cl_ctx, queue, weights, input, seq_len,
                           wp + ".layers.1.inner._linear", rms_key, fuse >= 1);
    if (!d1) return nullptr;
    const int hid = weights.get_shape(wp + ".layers.1.inner._linear.weight")[0];
    if (fuse < 1) {
        const int n = seq_len * hid;
        // Per-call-site kernel object — see the note in RMSNorm.cpp. (g_lb above needs none: it is
        // built but never dispatched, since dense_bias routes through nnopt_gemv_fused.)
        cl_kernel gk = nnopt_kernel_instance(g_gelu, wp, ".gelu");
        clSetKernelArg(gk,0,sizeof(cl_mem),&d1); clSetKernelArg(gk,1,sizeof(cl_mem),&d1);
        clSetKernelArg(gk,2,sizeof(int),&n);
        size_t gn = (size_t)n;
        err = cl_ctx.profEnqueue(gk,1,&gn,nullptr,"gelu");
        if (err!=CL_SUCCESS){ NNOPT_ERROR_FMT("MLP: gelu enqueue (err=%d)",err); pool_free(d1); return nullptr; }
    }
    // dense2 (-> 1024)
    cl_mem d2 = dense_bias(cl_ctx, queue, weights, d1, seq_len, wp + ".layers.3.inner._linear");
    pool_free(d1);
    if (!d2) return nullptr;
    
    return d2;
}
