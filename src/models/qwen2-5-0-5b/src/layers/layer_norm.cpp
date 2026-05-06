// Reference: model_info/transformers_src/modeling_qwen2.py:177-198 Qwen2RMSNorm.forward

#include "layers/layer_norm.h"

#include "debug_utils.h"
#include "opencl_context.h"
#include "utils.h"
#include "prof.h"
#include "weights.h"

#include <CL/cl.h>
#include <string>

LayerNorm::LayerNorm(OpenCLContext& cl_ctx, Weights& weights, const std::string& weight_key,
                     int hidden_size, float eps)
    : cl_ctx_(cl_ctx), weights_(weights), weight_key_(weight_key), hidden_size_(hidden_size), eps_(eps) {}

LayerNorm::~LayerNorm() {
    if (kernel_) clReleaseKernel(kernel_);
    if (program_) clReleaseProgram(program_);
    if (buf_out_) clReleaseMemObject(buf_out_);
}

bool LayerNorm::initialize() {
    weight_buf_ = weights_.get_buffer(weight_key_);
    if (!weight_buf_) {
        NNOPT_ERROR_FMT("RMSNorm missing weight: %s", weight_key_.c_str());
        return false;
    }

    program_ = cl_ctx_.build_program_from_file(
        "kernels/rmsnorm.cl",
#ifdef NNOPT_USE_FP16
        "-DNNOPT_USE_FP16=1 -DUSE_FP16=1"
#else
        ""
#endif
    );
    if (!program_) {
        NNOPT_ERROR("RMSNorm: failed to build kernels/rmsnorm.cl");
        return false;
    }

    cl_int err = CL_SUCCESS;
    kernel_ = clCreateKernel(program_, "rmsnorm_forward", &err);
    if (err != CL_SUCCESS || !kernel_) {
        NNOPT_ERROR_FMT("RMSNorm: clCreateKernel(rmsnorm_forward) failed: %d", (int)err);
        return false;
    }

    return true;
}

cl_mem LayerNorm::forward(cl_command_queue queue, cl_mem input, int seq_len) {
    if (!input || seq_len <= 0) return nullptr;

    const int rows = seq_len;
    const int cols = hidden_size_;

    cl_int err = CL_SUCCESS;
    cl_context ctx = cl_ctx_.context();

    // Persistent output buffer — lazy-alloc, grow-on-demand. Saves one
    // clCreateBuffer per forward (25 LayerNorms per token at decode).
    if (!buf_out_ || rows > buf_capacity_rows_) {
        if (buf_out_) { clReleaseMemObject(buf_out_); buf_out_ = nullptr; }
        const size_t bytes = (size_t)rows * (size_t)cols * sizeof(nnopt_storage_t);
        buf_out_ = clCreateBuffer(ctx, CL_MEM_READ_WRITE, bytes, nullptr, &err);
        if (err != CL_SUCCESS || !buf_out_) {
            NNOPT_ERROR_FMT("RMSNorm: ensure buf_out_ failed: %d", (int)err);
            return nullptr;
        }
        buf_capacity_rows_ = rows;
    }
    cl_mem output = buf_out_;

    // rmsnorm_forward(x, weight, out, rows, cols, eps)
    if (clSetKernelArg(kernel_, 0, sizeof(cl_mem), &input) != CL_SUCCESS) {
        NNOPT_ERROR("RMSNorm: setArg 0 (x) failed");
        return nullptr;
    }
    if (clSetKernelArg(kernel_, 1, sizeof(cl_mem), &weight_buf_) != CL_SUCCESS) {
        NNOPT_ERROR("RMSNorm: setArg 1 (weight) failed");
        return nullptr;
    }
    if (clSetKernelArg(kernel_, 2, sizeof(cl_mem), &output) != CL_SUCCESS) {
        NNOPT_ERROR("RMSNorm: setArg 2 (out) failed");
        return nullptr;
    }
    if (clSetKernelArg(kernel_, 3, sizeof(int), &rows) != CL_SUCCESS) {
        NNOPT_ERROR("RMSNorm: setArg 3 (rows) failed");
        return nullptr;
    }
    if (clSetKernelArg(kernel_, 4, sizeof(int), &cols) != CL_SUCCESS) {
        NNOPT_ERROR("RMSNorm: setArg 4 (cols) failed");
        return nullptr;
    }
    if (clSetKernelArg(kernel_, 5, sizeof(float), &eps_) != CL_SUCCESS) {
        NNOPT_ERROR("RMSNorm: setArg 5 (eps) failed");
        return nullptr;
    }

    // Cooperative kernel requires explicit lws=64.
    size_t lws[1] = {64};
    size_t gws[1] = {(size_t)rows * 64};
    err = nnopt_prof::enqueue(queue, kernel_, 1, nullptr, gws, lws, 0, nullptr, nullptr);
    if (err != CL_SUCCESS) {
        NNOPT_ERROR_FMT("RMSNorm: enqueue failed: %d", (int)err);
        return nullptr;
    }

    NNOPT_DEBUG_SYNC(queue);
    // Caller will clReleaseMemObject the returned buffer; retain so the
    // release decrements back to our owned reference.
    clRetainMemObject(output);
    return output;
}
