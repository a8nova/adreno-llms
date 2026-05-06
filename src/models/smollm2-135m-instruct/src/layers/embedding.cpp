// Reference: model_info/transformers_src/modeling_llama.py:375-428 LlamaModel.forward (embed_tokens)

#include "layers/embedding.h"
#include "opencl_context.h"
#include "weights.h"
#include "debug_utils.h"
#include "utils.h"
#include "model_config.h"
#include "prof.h"
#include <string>
#include <vector>

Embedding::Embedding(OpenCLContext& cl_ctx, Weights& weights)
    : cl_ctx_(cl_ctx), weights_(weights) {}

Embedding::~Embedding() {
    if (kernel_) clReleaseKernel(kernel_);
    if (program_) clReleaseProgram(program_);
}

static inline bool set_arg_checked(cl_kernel k, cl_uint idx, size_t sz, const void* val, const char* what) {
    cl_int err = clSetKernelArg(k, idx, sz, val);
    if (err != CL_SUCCESS) {
        NNOPT_ERROR_FMT("clSetKernelArg %s failed: %d", what, err);
        return false;
    }
    return true;
}

bool Embedding::initialize() {
    program_ = cl_ctx_.build_program_from_file(
        "kernels/embedding.cl",
#ifdef NNOPT_USE_FP16
        "-DNNOPT_USE_FP16=1 -DUSE_FP16=1"
#else
        ""
#endif
    );
    if (!program_) {
        NNOPT_ERROR("Embedding: failed to build kernels/embedding.cl");
        return false;
    }

    cl_int err = CL_SUCCESS;
    kernel_ = clCreateKernel(program_, "embedding_forward", &err);
    if (err != CL_SUCCESS || !kernel_) {
        NNOPT_ERROR_FMT("Embedding: clCreateKernel(embedding_forward) failed: %d", err);
        return false;
    }

    // Weight key must match Embedding contract ((model metadata))
    wte_ = weights_.get_buffer("model.embed_tokens.weight");
    if (!wte_) {
        NNOPT_ERROR("Embedding: missing weight model.embed_tokens.weight");
        return false;
    }

    NNOPT_LAYER_INIT("embedding");
    return true;
}

cl_mem Embedding::forward(cl_command_queue queue, cl_mem input_ids, int seq_len) {
    NNOPT_LAYER_FWD("embedding");
    NNOPT_LAYER_CHECK_INPUT_INT("embedding", queue, input_ids, (size_t)seq_len);

    cl_int err = CL_SUCCESS;
    cl_context ctx = cl_ctx_.context();

    cl_mem out = clCreateBuffer(ctx, CL_MEM_READ_WRITE,
                               (size_t)seq_len * (size_t)MODEL_CONFIG::HIDDEN_SIZE * sizeof(nnopt_storage_t),
                               nullptr, &err);
    if (err != CL_SUCCESS || !out) {
        NNOPT_ERROR_FMT("Embedding: alloc out failed: %d", err);
        return nullptr;
    }

    const int hidden = MODEL_CONFIG::HIDDEN_SIZE;

    // LLaMA-family (RoPE) has NO absolute position embeddings (no wpe).
    // kernels/embedding.cl is the wte-only variant; start_pos is unused but kept for signature compatibility.
    const int start_pos = 0;

    if (!set_arg_checked(kernel_, 0, sizeof(cl_mem), &input_ids, "token_ids")) { clReleaseMemObject(out); return nullptr; }
    if (!set_arg_checked(kernel_, 1, sizeof(cl_mem), &wte_, "wte")) { clReleaseMemObject(out); return nullptr; }
    if (!set_arg_checked(kernel_, 2, sizeof(cl_mem), &wte_, "wpe(dummy)")) { clReleaseMemObject(out); return nullptr; }
    if (!set_arg_checked(kernel_, 3, sizeof(cl_mem), &out, "output")) { clReleaseMemObject(out); return nullptr; }
    if (!set_arg_checked(kernel_, 4, sizeof(int), &hidden, "hidden_size")) { clReleaseMemObject(out); return nullptr; }
    if (!set_arg_checked(kernel_, 5, sizeof(int), &start_pos, "start_pos")) { clReleaseMemObject(out); return nullptr; }

    size_t gws[1] = { (size_t)seq_len };
    err = nnopt_prof::enqueue(queue, kernel_, 1, nullptr, gws, nullptr, 0, nullptr, nullptr);
    if (err != CL_SUCCESS) {
        NNOPT_ERROR_FMT("Embedding: dispatch failed: %d", err);
        clReleaseMemObject(out);
        return nullptr;
    }

    NNOPT_LAYER_CHECK("embedding", queue, out, (size_t)seq_len * (size_t)hidden);
    NNOPT_LAYER_FWD_DONE("embedding");
    return out;
}
