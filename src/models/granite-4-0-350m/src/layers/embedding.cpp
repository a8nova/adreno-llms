// Reference: model_info/transformers_src/modeling_granitemoehybrid.py:1151-1210 GraniteMoeHybridModel.forward (embed_tokens + embedding_multiplier)
// Reference: model_info/transformers_src/modeling_granitemoehybrid.py:779-843 GraniteMoeHybridRotaryEmbedding (position_ids construction; RoPE lives in attention)

#include "layers/embedding.h"
#include "opencl_context.h"
#include "weights.h"
#include "debug_utils.h"
#include "model_config.h"
#include "utils.h"  // nnopt_storage_t
#include "kernel_profiler.h"

#include <CL/cl.h>
#include <string>

Embedding::Embedding(OpenCLContext& cl_ctx, Weights& weights)
    : cl_ctx_(cl_ctx), weights_(weights) {}

Embedding::~Embedding() {
    if (kernel_) clReleaseKernel(kernel_);
    if (program_) clReleaseProgram(program_);
    if (rope_cos_) clReleaseMemObject(rope_cos_);
    if (rope_sin_) clReleaseMemObject(rope_sin_);
    // embed_tokens_weight_ is owned by Weights
}

bool Embedding::initialize() {
    cl_int err = CL_SUCCESS;

    program_ = cl_ctx_.build_program_from_file("kernels/embedding.cl");
    if (!program_) {
        NNOPT_ERROR("Embedding: build_program_from_file(kernels/embedding.cl) failed");
        return false;
    }

    kernel_ = clCreateKernel(program_, "embedding_forward", &err);
    if (err != CL_SUCCESS || !kernel_) {
        NNOPT_ERROR_FMT("Embedding: clCreateKernel(embedding_forward) failed (%d)", err);
        return false;
    }

    embed_tokens_weight_ = weights_.get_buffer("model.embed_tokens.weight");
    if (!embed_tokens_weight_) {
        NNOPT_ERROR("Embedding: missing weight model.embed_tokens.weight");
        return false;
    }

    // Granite uses RoPE, no absolute position embedding table.
    // embedding_forward currently expects wpe; we pass a dummy buffer.
    position_weight_ = embed_tokens_weight_;

    // Precompute RoPE tables on host then upload once.
    // Reference: modeling_granitemoehybrid.py::GraniteMoeHybridRotaryEmbedding
    const int max_seq = MODEL_CONFIG::MAX_SEQ_LEN;
    const int head_dim = MODEL_CONFIG::HEAD_DIM;
    const float base = (float)MODEL_CONFIG::ROPE_THETA;

    if (head_dim % 2 != 0) {
        NNOPT_ERROR_FMT("Embedding: HEAD_DIM must be even for RoPE (got %d)", head_dim);
        return false;
    }

    std::vector<float> cos_f((size_t)max_seq * (size_t)head_dim);
    std::vector<float> sin_f((size_t)max_seq * (size_t)head_dim);

    for (int pos = 0; pos < max_seq; pos++) {
        for (int i = 0; i < head_dim / 2; i++) {
            const float inv_freq = powf(base, -(float)i * 2.0f / (float)head_dim);
            const float theta = (float)pos * inv_freq;
            const float c = cosf(theta);
            const float s = sinf(theta);
            cos_f[(size_t)pos * (size_t)head_dim + (size_t)i] = c;
            sin_f[(size_t)pos * (size_t)head_dim + (size_t)i] = s;
            cos_f[(size_t)pos * (size_t)head_dim + (size_t)(i + head_dim / 2)] = c;
            sin_f[(size_t)pos * (size_t)head_dim + (size_t)(i + head_dim / 2)] = s;
        }
    }

    rope_cos_ = clCreateBuffer(cl_ctx_.context(), CL_MEM_READ_WRITE,
                              (size_t)max_seq * (size_t)head_dim * sizeof(nnopt_storage_t), nullptr, &err);
    if (err != CL_SUCCESS || !rope_cos_) {
        NNOPT_ERROR_FMT("Embedding: clCreateBuffer rope_cos failed (%d)", err);
        return false;
    }
    rope_sin_ = clCreateBuffer(cl_ctx_.context(), CL_MEM_READ_WRITE,
                              (size_t)max_seq * (size_t)head_dim * sizeof(nnopt_storage_t), nullptr, &err);
    if (err != CL_SUCCESS || !rope_sin_) {
        NNOPT_ERROR_FMT("Embedding: clCreateBuffer rope_sin failed (%d)", err);
        return false;
    }

#ifdef NNOPT_USE_FP16
    std::vector<nnopt_storage_t> cos_h((size_t)max_seq * (size_t)head_dim);
    std::vector<nnopt_storage_t> sin_h((size_t)max_seq * (size_t)head_dim);
    for (size_t i = 0; i < cos_h.size(); i++) {
        cos_h[i] = (nnopt_storage_t)nnopt_f32_to_f16(cos_f[i]);
        sin_h[i] = (nnopt_storage_t)nnopt_f32_to_f16(sin_f[i]);
    }
    err = clEnqueueWriteBuffer(cl_ctx_.queue(), rope_cos_, CL_TRUE, 0, cos_h.size() * sizeof(nnopt_storage_t), cos_h.data(), 0, nullptr, nullptr);
    if (err != CL_SUCCESS) { NNOPT_ERROR_FMT("Embedding: write rope_cos failed (%d)", err); return false; }
    err = clEnqueueWriteBuffer(cl_ctx_.queue(), rope_sin_, CL_TRUE, 0, sin_h.size() * sizeof(nnopt_storage_t), sin_h.data(), 0, nullptr, nullptr);
    if (err != CL_SUCCESS) { NNOPT_ERROR_FMT("Embedding: write rope_sin failed (%d)", err); return false; }
#else
    err = clEnqueueWriteBuffer(cl_ctx_.queue(), rope_cos_, CL_TRUE, 0, cos_f.size() * sizeof(float), cos_f.data(), 0, nullptr, nullptr);
    if (err != CL_SUCCESS) { NNOPT_ERROR_FMT("Embedding: write rope_cos failed (%d)", err); return false; }
    err = clEnqueueWriteBuffer(cl_ctx_.queue(), rope_sin_, CL_TRUE, 0, sin_f.size() * sizeof(float), sin_f.data(), 0, nullptr, nullptr);
    if (err != CL_SUCCESS) { NNOPT_ERROR_FMT("Embedding: write rope_sin failed (%d)", err); return false; }
#endif

    return true;
}

cl_mem Embedding::forward(cl_command_queue queue, cl_mem token_ids, int seq_len, int start_pos) {
    NNOPT_LAYER_FWD("embedding");
    NNOPT_LAYER_CHECK_INPUT_INT("embedding", queue, token_ids, (size_t)seq_len);

    cl_context ctx = cl_ctx_.context();
    const int hidden = MODEL_CONFIG::HIDDEN_SIZE;

    cl_int err = CL_SUCCESS;
    const size_t out_elems = (size_t)seq_len * (size_t)hidden;
    cl_mem out = clCreateBuffer(ctx, CL_MEM_READ_WRITE, out_elems * sizeof(nnopt_storage_t), nullptr, &err);
    if (err != CL_SUCCESS || !out) {
        NNOPT_ERROR_FMT("Embedding: clCreateBuffer out failed (%d)", err);
        return nullptr;
    }

    // embedding_forward signature includes wpe + start_pos for scaffold compatibility.
    // For this RoPE model, the kernel ignores wpe + start_pos.
    cl_mem wpe_buf = position_weight_;

    int arg = 0;
    err = clSetKernelArg(kernel_, arg++, sizeof(cl_mem), &token_ids);
    if (err != CL_SUCCESS) { NNOPT_ERROR_FMT("Embedding: set arg token_ids failed (%d)", err); clReleaseMemObject(out); return nullptr; }
    err = clSetKernelArg(kernel_, arg++, sizeof(cl_mem), &embed_tokens_weight_);
    if (err != CL_SUCCESS) { NNOPT_ERROR_FMT("Embedding: set arg wte failed (%d)", err); clReleaseMemObject(out); return nullptr; }
    err = clSetKernelArg(kernel_, arg++, sizeof(cl_mem), &wpe_buf);
    if (err != CL_SUCCESS) { NNOPT_ERROR_FMT("Embedding: set arg wpe failed (%d)", err); clReleaseMemObject(out); return nullptr; }
    err = clSetKernelArg(kernel_, arg++, sizeof(cl_mem), &out);
    if (err != CL_SUCCESS) { NNOPT_ERROR_FMT("Embedding: set arg out failed (%d)", err); clReleaseMemObject(out); return nullptr; }
    err = clSetKernelArg(kernel_, arg++, sizeof(int), &hidden);
    if (err != CL_SUCCESS) { NNOPT_ERROR_FMT("Embedding: set arg hidden failed (%d)", err); clReleaseMemObject(out); return nullptr; }
    err = clSetKernelArg(kernel_, arg++, sizeof(int), &start_pos);
    if (err != CL_SUCCESS) { NNOPT_ERROR_FMT("Embedding: set arg start_pos failed (%d)", err); clReleaseMemObject(out); return nullptr; }

    size_t gws = (size_t)seq_len;
    err = clEnqueueNDRangeKernel(queue, kernel_, 1, nullptr, &gws, nullptr, 0, nullptr, KernelProfiler::event_for("embedding_lookup"));
    if (err != CL_SUCCESS) {
        NNOPT_ERROR_FMT("Embedding: enqueue embedding_forward failed (%d)", err);
        clReleaseMemObject(out);
        return nullptr;
    }

    // Multiply by embedding_multiplier (scalar) using utils kernel? For now: do it in-place with a tiny kernel.
    // Granite multiplies embeddings by EMBEDDING_MULTIPLIER (12). We'll do this later in Model::forward using a scale kernel.

    NNOPT_LAYER_CHECK("embedding", queue, out, out_elems);
    NNOPT_LAYER_FWD_DONE("embedding");
    return out;
}
