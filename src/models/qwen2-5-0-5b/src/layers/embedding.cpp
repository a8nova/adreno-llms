// Reference: model_info/transformers_src/modeling_qwen2.py:176-214 Qwen2Model.forward (embed_tokens)
// Reference: model_info/transformers_src/modeling_qwen2.py:292-333 Qwen2RotaryEmbedding.forward (position_ids semantics)

#include "layers/embedding.h"

#include "debug_utils.h"
#include "opencl_context.h"
#include "utils.h"
#include "prof.h"
#include "weights.h"

#include <CL/cl.h>
#include <string>
#include <vector>

cl_mem Embedding::forward(cl_command_queue queue, const std::vector<int32_t>& input_ids, int start_pos) {
    return forward(queue, (const int*)input_ids.data(), (int)input_ids.size(), start_pos);
}


Embedding::Embedding(OpenCLContext& cl_ctx, Weights& weights)
    : cl_ctx_(cl_ctx), weights_(weights) {}

Embedding::~Embedding() {
    if (kernel_) clReleaseKernel(kernel_);
    if (program_) clReleaseProgram(program_);
    if (ids_buf_decode_) clReleaseMemObject(ids_buf_decode_);
}

// Persistent-id-buffer write — called between decode iterations on the LIVE
// queue. NOT inside the recording (clEnqueueWriteBuffer is not recordable).
bool Embedding::set_decode_token(cl_command_queue queue, int32_t token_id) {
    cl_int err = CL_SUCCESS;
    if (!ids_buf_decode_) {
        ids_buf_decode_ = clCreateBuffer(cl_ctx_.context(), CL_MEM_READ_ONLY,
                                          sizeof(int32_t), nullptr, &err);
        if (err != CL_SUCCESS || !ids_buf_decode_) {
            NNOPT_ERROR_FMT("Embedding::set_decode_token: alloc ids_buf failed: %d", (int)err);
            return false;
        }
    }
    err = clEnqueueWriteBuffer(queue, ids_buf_decode_, CL_FALSE, 0,
                                sizeof(int32_t), &token_id, 0, nullptr, nullptr);
    if (err != CL_SUCCESS) {
        NNOPT_ERROR_FMT("Embedding::set_decode_token: write failed: %d", (int)err);
        return false;
    }
    return true;
}

// Recordable kernel-only dispatch. Caller owns dest (Model's persistent
// hidden buffer). Same arg layout as forward(); no clCreateBuffer here.
bool Embedding::forward_into_decode(cl_command_queue queue, cl_mem dest, int start_pos) {
    if (!ids_buf_decode_) {
        NNOPT_ERROR("Embedding::forward_into_decode: set_decode_token not called yet");
        return false;
    }
    const int hidden = MODEL_CONFIG::HIDDEN_SIZE;
    cl_int err = CL_SUCCESS;
    err = clSetKernelArg(kernel_, 0, sizeof(cl_mem), &ids_buf_decode_);  if (err != CL_SUCCESS) return false;
    err = clSetKernelArg(kernel_, 1, sizeof(cl_mem), &wte_);              if (err != CL_SUCCESS) return false;
    err = clSetKernelArg(kernel_, 2, sizeof(cl_mem), &wpe_zero_);         if (err != CL_SUCCESS) return false;
    err = clSetKernelArg(kernel_, 3, sizeof(cl_mem), &dest);              if (err != CL_SUCCESS) return false;
    err = clSetKernelArg(kernel_, 4, sizeof(int),    &hidden);            if (err != CL_SUCCESS) return false;
    err = clSetKernelArg(kernel_, 5, sizeof(int),    &start_pos);         if (err != CL_SUCCESS) return false;

    size_t gws[1] = {1};
    err = nnopt_prof::enqueue(queue, kernel_, 1, nullptr, gws, nullptr, 0, nullptr, nullptr);
    if (err != CL_SUCCESS) {
        NNOPT_ERROR_FMT("Embedding::forward_into_decode: enqueue failed: %d", (int)err);
        return false;
    }
    return true;
}

bool Embedding::initialize() {
    // Qwen2 uses token embeddings only in PyTorch; RoPE applies position later.
    // Our scaffold kernel adds wpe as well; we pass a zero wpe buffer so output
    // equals wte.
    wte_ = weights_.get_buffer("model.embed_tokens.weight");
    if (!wte_) {
        NNOPT_ERROR("Embedding: missing required weight model.embed_tokens.weight");
        return false;
    }

    // Create zero position embedding table: [MAX_SEQ_LEN, HIDDEN_SIZE]
    const size_t numel = (size_t)MODEL_CONFIG::MAX_POSITION_EMBEDDINGS * (size_t)MODEL_CONFIG::HIDDEN_SIZE;
    cl_int err = CL_SUCCESS;
    wpe_zero_ = clCreateBuffer(cl_ctx_.context(), CL_MEM_READ_WRITE,
                               numel * sizeof(nnopt_storage_t), nullptr, &err);
    if (err != CL_SUCCESS || !wpe_zero_) {
        NNOPT_ERROR_FMT("Embedding: clCreateBuffer(wpe_zero) failed: %d", (int)err);
        return false;
    }
    // Zero fill
    nnopt_storage_t zero = (nnopt_storage_t)0;
    err = clEnqueueFillBuffer(cl_ctx_.queue(), wpe_zero_, &zero, sizeof(zero), 0,
                              numel * sizeof(nnopt_storage_t), 0, nullptr, nullptr);
    if (err != CL_SUCCESS) {
        NNOPT_ERROR_FMT("Embedding: clEnqueueFillBuffer(wpe_zero) failed: %d", (int)err);
        return false;
    }
    NNOPT_DEBUG_SYNC(cl_ctx_.queue());

    program_ = cl_ctx_.build_program_from_file("kernels/embedding.cl");
    if (!program_) return false;

    kernel_ = clCreateKernel(program_, "embedding_forward", &err);
    if (err != CL_SUCCESS || !kernel_) {
        NNOPT_ERROR_FMT("Embedding: clCreateKernel(embedding_forward) failed: %d", (int)err);
        return false;
    }

    NNOPT_LAYER_INIT("embedding");
    return true;
}

cl_mem Embedding::forward(cl_command_queue queue, const int* token_ids, int seq_len, int start_pos) {
    if (!token_ids || seq_len <= 0) return nullptr;

    cl_int err = CL_SUCCESS;

    // UPLOAD-OK: input_ids
    cl_mem ids_buf = clCreateBuffer(cl_ctx_.context(), CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
                                    (size_t)seq_len * sizeof(int), (void*)token_ids, &err);
    if (err != CL_SUCCESS || !ids_buf) {
        NNOPT_ERROR_FMT("Embedding: clCreateBuffer(ids) failed: %d", (int)err);
        return nullptr;
    }

    cl_mem out = clCreateBuffer(cl_ctx_.context(), CL_MEM_READ_WRITE,
                                (size_t)seq_len * (size_t)MODEL_CONFIG::HIDDEN_SIZE * sizeof(nnopt_storage_t),
                                nullptr, &err);
    if (err != CL_SUCCESS || !out) {
        NNOPT_ERROR_FMT("Embedding: clCreateBuffer(out) failed: %d", (int)err);
        clReleaseMemObject(ids_buf);
        return nullptr;
    }

    const int hidden = MODEL_CONFIG::HIDDEN_SIZE;

    int arg = 0;
    err = clSetKernelArg(kernel_, arg++, sizeof(cl_mem), &ids_buf);
    if (err != CL_SUCCESS) {
        NNOPT_ERROR_FMT("Embedding: setArg token_ids failed: %d", (int)err);
        clReleaseMemObject(ids_buf);
        clReleaseMemObject(out);
        return nullptr;
    }
    err = clSetKernelArg(kernel_, arg++, sizeof(cl_mem), &wte_);
    if (err != CL_SUCCESS) {
        NNOPT_ERROR_FMT("Embedding: setArg wte failed: %d", (int)err);
        clReleaseMemObject(ids_buf);
        clReleaseMemObject(out);
        return nullptr;
    }
    err = clSetKernelArg(kernel_, arg++, sizeof(cl_mem), &wpe_zero_);
    if (err != CL_SUCCESS) {
        NNOPT_ERROR_FMT("Embedding: setArg wpe failed: %d", (int)err);
        clReleaseMemObject(ids_buf);
        clReleaseMemObject(out);
        return nullptr;
    }
    err = clSetKernelArg(kernel_, arg++, sizeof(cl_mem), &out);
    if (err != CL_SUCCESS) {
        NNOPT_ERROR_FMT("Embedding: setArg out failed: %d", (int)err);
        clReleaseMemObject(ids_buf);
        clReleaseMemObject(out);
        return nullptr;
    }
    err = clSetKernelArg(kernel_, arg++, sizeof(int), &hidden);
    if (err != CL_SUCCESS) {
        NNOPT_ERROR_FMT("Embedding: setArg hidden failed: %d", (int)err);
        clReleaseMemObject(ids_buf);
        clReleaseMemObject(out);
        return nullptr;
    }
    err = clSetKernelArg(kernel_, arg++, sizeof(int), &start_pos);
    if (err != CL_SUCCESS) {
        NNOPT_ERROR_FMT("Embedding: setArg start_pos failed: %d", (int)err);
        clReleaseMemObject(ids_buf);
        clReleaseMemObject(out);
        return nullptr;
    }

    size_t gws[1] = {(size_t)seq_len};
    err = nnopt_prof::enqueue(queue, kernel_, 1, nullptr, gws, nullptr, 0, nullptr, nullptr);
    if (err != CL_SUCCESS) {
        NNOPT_ERROR_FMT("Embedding: enqueue failed: %d", (int)err);
        clReleaseMemObject(ids_buf);
        clReleaseMemObject(out);
        return nullptr;
    }
    NNOPT_DEBUG_SYNC(queue);

    clReleaseMemObject(ids_buf);

    return out;
}
