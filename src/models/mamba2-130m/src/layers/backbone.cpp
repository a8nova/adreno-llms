// Reference: model_info/transformers_src/modeling_mamba2.py (Mamba2Model.forward)

#include "layers/backbone.h"

#include "debug_utils.h"
#include "kernel_profiler.h"
#include "opencl_context.h"
#include "utils.h"
#include "weights.h"
#include "model_config.h"

#include <clblast.h>
#include <string>

Backbone::Backbone(OpenCLContext& cl_ctx, Weights& weights)
    : cl_ctx_(cl_ctx), weights_(weights) {
}

Backbone::~Backbone() {
    // Weight buffers are owned by Weights (lazy GPU buffer cache). Do NOT release.
    embed_w_ = nullptr;
    release_buffers_();
}

void Backbone::release_buffers_() {
    if (buf_ids_)       { clReleaseMemObject(buf_ids_);       buf_ids_ = nullptr; }
    if (buf_embed_out_) { clReleaseMemObject(buf_embed_out_); buf_embed_out_ = nullptr; }
    if (buf_norm_f_out_){ clReleaseMemObject(buf_norm_f_out_);buf_norm_f_out_ = nullptr; }
    buf_capacity_rows_ = 0;
}

bool Backbone::ensure_buffers_(int rows) {
    if (rows <= buf_capacity_rows_ && buf_ids_ && buf_embed_out_ && buf_norm_f_out_) return true;
    release_buffers_();
    cl_int err = CL_SUCCESS;
    const int hidden = MODEL_CONFIG::D_MODEL;
    buf_ids_ = clCreateBuffer(cl_ctx_.context(), CL_MEM_READ_ONLY,
                              (size_t)rows * sizeof(int32_t), nullptr, &err);
    if (err != CL_SUCCESS || !buf_ids_) {
        NNOPT_ERROR_FMT("Backbone::ensure_buffers_(ids) failed (%d)", (int)err);
        return false;
    }
    buf_embed_out_ = clCreateBuffer(cl_ctx_.context(), CL_MEM_READ_WRITE,
                                    (size_t)rows * (size_t)hidden * sizeof(nnopt_storage_t),
                                    nullptr, &err);
    if (err != CL_SUCCESS || !buf_embed_out_) {
        NNOPT_ERROR_FMT("Backbone::ensure_buffers_(embed_out) failed (%d)", (int)err);
        return false;
    }
    buf_norm_f_out_ = clCreateBuffer(cl_ctx_.context(), CL_MEM_READ_WRITE,
                                     (size_t)rows * (size_t)hidden * sizeof(nnopt_storage_t),
                                     nullptr, &err);
    if (err != CL_SUCCESS || !buf_norm_f_out_) {
        NNOPT_ERROR_FMT("Backbone::ensure_buffers_(norm_f_out) failed (%d)", (int)err);
        return false;
    }
    buf_capacity_rows_ = rows;
    return true;
}

bool Backbone::initialize() {
    // weights_.get_buffer returns an owned cl_mem (Weights owns lifetime).
    embed_w_ = weights_.get_buffer("backbone.embedding.weight");
    if (!embed_w_) {
        NNOPT_ERROR("Backbone: missing backbone.embedding.weight");
        return false;
    }

    norm_w_ = weights_.get_buffer("backbone.norm_f.weight");
    if (!norm_w_) {
        NNOPT_ERROR("Backbone: missing backbone.norm_f.weight");
        return false;
    }

    embed_program_ = cl_ctx_.build_program_from_file("kernels/embedding.cl");
    if (!embed_program_) {
        NNOPT_ERROR("Backbone: failed to build kernels/embedding.cl");
        return false;
    }

    cl_int err = CL_SUCCESS;
    embed_kernel_ = clCreateKernel(embed_program_, "embedding_forward", &err);
    if (err != CL_SUCCESS || !embed_kernel_) {
        NNOPT_ERROR_FMT("Backbone: clCreateKernel(embedding_forward) failed (%d)", (int)err);
        return false;
    }

    // Final RMS norm (Mamba2RMSNorm)
    rms_program_ = cl_ctx_.build_program_from_file("kernels/layer_norm.cl");
    if (!rms_program_) {
        NNOPT_ERROR("Backbone: failed to build kernels/layer_norm.cl");
        return false;
    }
    rms_kernel_ = clCreateKernel(rms_program_, "rms_norm", &err);
    if (err != CL_SUCCESS || !rms_kernel_) {
        NNOPT_ERROR_FMT("Backbone: clCreateKernel(rms_norm) failed (%d)", (int)err);
        return false;
    }

    return true;
}

cl_mem Backbone::embed(cl_command_queue queue, const int32_t* input_ids, int seq_len) {
    if (!embed_kernel_ || !embed_w_) {
        NNOPT_ERROR("Backbone::embed called before initialize() or missing weights");
        return nullptr;
    }

    const int hidden = MODEL_CONFIG::D_MODEL;

    // Lever 4: persistent ids/embed_out buffers. The host-side input_ids
    // payload still has to be uploaded each call (~50 µs), but we no longer
    // alloc/free a fresh GPU buffer per token.
    if (!ensure_buffers_(seq_len)) return nullptr;

    cl_int err = clEnqueueWriteBuffer(queue, buf_ids_, CL_FALSE, 0,
                                      (size_t)seq_len * sizeof(int32_t),
                                      input_ids, 0, nullptr, nullptr);
    if (err != CL_SUCCESS) {
        NNOPT_ERROR_FMT("Backbone: write ids_buf failed (%d)", (int)err);
        return nullptr;
    }
    cl_mem ids_buf = buf_ids_;
    cl_mem out     = buf_embed_out_;

    // Build a persistent zero WPE buffer on first use.
    static cl_mem zero_wpe = nullptr;
    static int zero_wpe_hidden = 0;
    static int zero_wpe_vocab = 0;
    if (!zero_wpe || zero_wpe_hidden != hidden || zero_wpe_vocab != MODEL_CONFIG::VOCAB_SIZE) {
        if (zero_wpe) {
            clReleaseMemObject(zero_wpe);
            zero_wpe = nullptr;
        }
        const size_t n = (size_t)MODEL_CONFIG::VOCAB_SIZE * (size_t)hidden;
        zero_wpe = clCreateBuffer(cl_ctx_.context(), CL_MEM_READ_WRITE, n * sizeof(nnopt_storage_t), nullptr, &err);
        if (err != CL_SUCCESS || !zero_wpe) {
            NNOPT_ERROR_FMT("Backbone: clCreateBuffer(zero_wpe) failed (%d)", (int)err);
            clReleaseMemObject(ids_buf);
            clReleaseMemObject(out);
            return nullptr;
        }
        // Fill with zeros.
        const nnopt_storage_t zero = (nnopt_storage_t)0;
        err = clEnqueueFillBuffer(queue, zero_wpe, &zero, sizeof(nnopt_storage_t), 0, n * sizeof(nnopt_storage_t), 0, nullptr, nullptr);
        if (err != CL_SUCCESS) {
            NNOPT_ERROR_FMT("Backbone: clEnqueueFillBuffer(zero_wpe) failed (%d)", (int)err);
            clReleaseMemObject(ids_buf);
            clReleaseMemObject(out);
            clReleaseMemObject(zero_wpe);
            zero_wpe = nullptr;
            return nullptr;
        }
        zero_wpe_hidden = hidden;
        zero_wpe_vocab = MODEL_CONFIG::VOCAB_SIZE;
    }

    int arg = 0;
    err = clSetKernelArg(embed_kernel_, arg++, sizeof(cl_mem), &ids_buf);
    err |= clSetKernelArg(embed_kernel_, arg++, sizeof(cl_mem), &embed_w_);
    err |= clSetKernelArg(embed_kernel_, arg++, sizeof(cl_mem), &zero_wpe);
    err |= clSetKernelArg(embed_kernel_, arg++, sizeof(cl_mem), &out);
    err |= clSetKernelArg(embed_kernel_, arg++, sizeof(int), &hidden);
    int start_pos = 0;
    err |= clSetKernelArg(embed_kernel_, arg++, sizeof(int), &start_pos);
    if (err != CL_SUCCESS) {
        NNOPT_ERROR_FMT("Backbone: setArg embedding_forward failed (%d)", (int)err);
        return nullptr;
    }

    size_t gws[1] = {(size_t)seq_len};
    cl_event* evt = KernelProfiler::event_for("embedding");
    err = clEnqueueNDRangeKernel(queue, embed_kernel_, 1, nullptr, gws, nullptr, 0, nullptr, evt);
    if (err != CL_SUCCESS) {
        NNOPT_ERROR_FMT("Backbone: enqueue embedding_forward failed (%d)", (int)err);
        return nullptr;
    }

    // out is the persistent buf_embed_out_; caller borrows, must NOT release.
    return out;
}

cl_mem Backbone::embed_gpu(cl_command_queue queue, cl_mem ids_buf, int seq_len) {
    // Lever D: GPU-side ids variant. Bypasses the host write to buf_ids_.
    // Used by the async decode pipeline so the embedding kernel reads directly
    // from the previous iteration's argmax output (which is on GPU already).
    if (!embed_kernel_ || !embed_w_) {
        NNOPT_ERROR("Backbone::embed_gpu called before initialize()");
        return nullptr;
    }
    if (!ids_buf) {
        NNOPT_ERROR("Backbone::embed_gpu: ids_buf is null");
        return nullptr;
    }

    const int hidden = MODEL_CONFIG::D_MODEL;
    if (!ensure_buffers_(seq_len)) return nullptr;
    cl_mem out = buf_embed_out_;

    // Per-function zero_wpe (separate from embed()'s; both lazy-init the same).
    static cl_mem zero_wpe_g = nullptr;
    static int zero_wpe_g_hidden = 0;
    static int zero_wpe_g_vocab = 0;
    cl_int err = CL_SUCCESS;
    if (!zero_wpe_g || zero_wpe_g_hidden != hidden || zero_wpe_g_vocab != MODEL_CONFIG::VOCAB_SIZE) {
        if (zero_wpe_g) clReleaseMemObject(zero_wpe_g);
        const size_t n = (size_t)MODEL_CONFIG::VOCAB_SIZE * (size_t)hidden;
        zero_wpe_g = clCreateBuffer(cl_ctx_.context(), CL_MEM_READ_WRITE,
                                    n * sizeof(nnopt_storage_t), nullptr, &err);
        if (err != CL_SUCCESS || !zero_wpe_g) {
            NNOPT_ERROR_FMT("Backbone::embed_gpu zero_wpe alloc failed (%d)", (int)err);
            return nullptr;
        }
        const nnopt_storage_t zero = (nnopt_storage_t)0;
        err = clEnqueueFillBuffer(queue, zero_wpe_g, &zero, sizeof(nnopt_storage_t),
                                  0, n * sizeof(nnopt_storage_t), 0, nullptr, nullptr);
        if (err != CL_SUCCESS) { NNOPT_ERROR_FMT("Backbone::embed_gpu zero_wpe fill (%d)", err); return nullptr; }
        zero_wpe_g_hidden = hidden;
        zero_wpe_g_vocab = MODEL_CONFIG::VOCAB_SIZE;
    }

    int arg = 0;
    err  = clSetKernelArg(embed_kernel_, arg++, sizeof(cl_mem), &ids_buf);
    err |= clSetKernelArg(embed_kernel_, arg++, sizeof(cl_mem), &embed_w_);
    err |= clSetKernelArg(embed_kernel_, arg++, sizeof(cl_mem), &zero_wpe_g);
    err |= clSetKernelArg(embed_kernel_, arg++, sizeof(cl_mem), &out);
    err |= clSetKernelArg(embed_kernel_, arg++, sizeof(int), &hidden);
    int start_pos = 0;
    err |= clSetKernelArg(embed_kernel_, arg++, sizeof(int), &start_pos);
    if (err != CL_SUCCESS) {
        NNOPT_ERROR_FMT("Backbone::embed_gpu setArg failed (%d)", (int)err);
        return nullptr;
    }

    size_t gws[1] = {(size_t)seq_len};
    cl_event* evt = KernelProfiler::event_for("embedding");
    err = clEnqueueNDRangeKernel(queue, embed_kernel_, 1, nullptr, gws, nullptr, 0, nullptr, evt);
    if (err != CL_SUCCESS) {
        NNOPT_ERROR_FMT("Backbone::embed_gpu enqueue failed (%d)", (int)err);
        return nullptr;
    }
    return out;
}

cl_mem Backbone::norm_f(cl_command_queue queue, cl_mem input, int seq_len) {
    if (!rms_kernel_ || !norm_w_) {
        NNOPT_ERROR("Backbone::norm_f called before initialize() or missing weights");
        return nullptr;
    }

    const int hidden = MODEL_CONFIG::D_MODEL;
    const float eps = 1e-5f;

    if (!ensure_buffers_(seq_len)) return nullptr;
    cl_mem out = buf_norm_f_out_;
    cl_int err = CL_SUCCESS;

    int arg = 0;
    err  = clSetKernelArg(rms_kernel_, arg++, sizeof(cl_mem), &input);
    err |= clSetKernelArg(rms_kernel_, arg++, sizeof(cl_mem), &norm_w_);
    err |= clSetKernelArg(rms_kernel_, arg++, sizeof(cl_mem), &out);
    err |= clSetKernelArg(rms_kernel_, arg++, sizeof(int), &hidden);
    err |= clSetKernelArg(rms_kernel_, arg++, sizeof(float), &eps);
    if (err != CL_SUCCESS) {
        NNOPT_ERROR_FMT("Backbone: setArg rms_norm failed (%d)", (int)err);
        return nullptr;
    }

    const size_t WG = 64;
    size_t gws[1] = {(size_t)seq_len * WG};
    size_t lws[1] = {WG};
    cl_event* evt = KernelProfiler::event_for("rms_norm_f");
    err = clEnqueueNDRangeKernel(queue, rms_kernel_, 1, nullptr, gws, lws, 0, nullptr, evt);
    if (err != CL_SUCCESS) {
        NNOPT_ERROR_FMT("Backbone: enqueue rms_norm failed (%d)", (int)err);
        return nullptr;
    }

    // out is the persistent buf_norm_f_out_; caller borrows, must NOT release.
    return out;
}
