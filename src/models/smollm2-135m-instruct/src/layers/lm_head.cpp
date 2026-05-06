// Reference: model_info/transformers_src/modeling_llama.py:445-501 LlamaForCausalLM.forward
// Implements lm_head projection (tied to embed_tokens weight).

#include "layers/lm_head.h"
#include "opencl_context.h"
#include "weights.h"
#include "debug_utils.h"
#include "utils.h"
#include "model_config.h"
#include "prof.h"
#include <clblast.h>
#include <string>

LmHead::LmHead(OpenCLContext& cl_ctx, Weights& weights)
    : cl_ctx_(cl_ctx), weights_(weights) {}

LmHead::~LmHead() {
    if (fused_lm_head_m1_) clReleaseKernel(fused_lm_head_m1_);
    if (block_fused_prog_) clReleaseProgram(block_fused_prog_);
}

bool LmHead::initialize() {
    // Tied embeddings: use model.embed_tokens.weight as lm_head weight.
    w_ = weights_.get_buffer("model.embed_tokens.weight");
    if (!w_) {
        NNOPT_ERROR("LmHead: missing weight model.embed_tokens.weight (tied embeddings)");
        return false;
    }

    // Decode fast-path GEMV kernel (fused_lm_head_gemv_m1 in block_fused.cl).
    block_fused_prog_ = cl_ctx_.build_program_from_file(
        "kernels/block_fused.cl",
#ifdef NNOPT_USE_FP16
        "-DNNOPT_USE_FP16=1 -DUSE_FP16=1"
#else
        ""
#endif
    );
    if (!block_fused_prog_) { NNOPT_ERROR("LmHead: failed to build block_fused.cl"); return false; }

    cl_int err = CL_SUCCESS;
    fused_lm_head_m1_ = clCreateKernel(block_fused_prog_, "fused_lm_head_gemv_m1", &err);
    if (err != CL_SUCCESS || !fused_lm_head_m1_) {
        NNOPT_ERROR_FMT("clCreateKernel fused_lm_head_gemv_m1 failed: %d", err);
        return false;
    }

    NNOPT_LAYER_INIT("lm_head");
    return true;
}

cl_mem LmHead::forward(cl_command_queue queue, cl_mem hidden, int M) {
    // hidden: [M, H] ; W: [V, H] ; out: [M, V]
    const int H = MODEL_CONFIG::HIDDEN_SIZE;
    const int V = MODEL_CONFIG::VOCAB_SIZE;

    cl_int err = CL_SUCCESS;
    cl_context ctx = cl_ctx_.context();
    cl_mem out = clCreateBuffer(ctx, CL_MEM_READ_WRITE,
                               (size_t)M * (size_t)V * sizeof(nnopt_storage_t),
                               nullptr, &err);
    if (err != CL_SUCCESS || !out) {
        NNOPT_ERROR_FMT("LmHead: alloc out failed: %d", err);
        return nullptr;
    }

    if (M == 1 && fused_lm_head_m1_) {
        // Decode fast path: GEMV — one workgroup per output token, 64 threads cooperative.
        auto sa = [&](cl_uint idx, size_t sz, const void* v, const char* n) -> bool {
            cl_int e = clSetKernelArg(fused_lm_head_m1_, idx, sz, v);
            if (e != CL_SUCCESS) { NNOPT_ERROR_FMT("lm_head arg %s: %d", n, e); return false; }
            return true;
        };
        if (!sa(0, sizeof(cl_mem), &hidden, "x") ||
            !sa(1, sizeof(cl_mem), &w_,     "W") ||
            !sa(2, sizeof(cl_mem), &out,    "logits") ||
            !sa(3, sizeof(int),    &H,      "H") ||
            !sa(4, sizeof(int),    &V,      "VOCAB")) {
            clReleaseMemObject(out);
            return nullptr;
        }
        const size_t WG = 64;
        size_t gws = (size_t)V * WG;
        size_t lws = WG;
        err = nnopt_prof::enqueue(queue, fused_lm_head_m1_, 1, nullptr, &gws, &lws, 0, nullptr, nullptr);
        if (err != CL_SUCCESS) {
            NNOPT_ERROR_FMT("fused_lm_head_gemv_m1 dispatch failed: %d", err);
            clReleaseMemObject(out);
            return nullptr;
        }
    } else {
        if (!pytorch_linear(queue, M, V, H, hidden, w_, out)) {
            clReleaseMemObject(out);
            return nullptr;
        }
    }

    NNOPT_LAYER_CHECK("lm_head", queue, out, (size_t)M * (size_t)V);
    return out;
}
