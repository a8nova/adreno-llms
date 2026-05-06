// Reference: model_info/transformers_src/modeling_mamba.py:255-283 MambaRMSNorm.forward
// Reference: model_info/config.json:rms_norm=true, layer_norm_epsilon

#include "layers/layer_norm.h"

#include "debug_utils.h"
#include "kernel_profiler.h"
#include "opencl_context.h"
#include "utils.h"
#include "weights.h"

#include <CL/cl.h>
#include <cmath>
#include <string>

LayerNorm::LayerNorm(OpenCLContext& cl_ctx, Weights& weights, const std::string& weight_key,
                     int hidden_size, float eps)
    : cl_ctx_(cl_ctx), weights_(weights), weight_key_(weight_key), hidden_size_(hidden_size), eps_(eps) {}

LayerNorm::~LayerNorm() {
    if (kernel_)  clReleaseKernel(kernel_);
    if (program_) clReleaseProgram(program_);
    if (output_buf_) clReleaseMemObject(output_buf_);
}

bool LayerNorm::initialize() {
    gamma_ = weights_.get_buffer(weight_key_);
    if (!gamma_) {
        NNOPT_ERROR_FMT("LayerNorm: missing weight: %s", weight_key_.c_str());
        return false;
    }

    program_ = cl_ctx_.build_program_from_file("kernels/layer_norm.cl");
    if (!program_) return false;

    cl_int err = CL_SUCCESS;
    kernel_ = clCreateKernel(program_, "rms_norm", &err);
    if (err != CL_SUCCESS) {
        NNOPT_ERROR_FMT("LayerNorm: clCreateKernel(rms_norm) failed: %d", (int)err);
        return false;
    }

    return true;
}

cl_mem LayerNorm::forward(cl_command_queue queue, cl_mem input, int seq_len) {
    if (!kernel_ || !gamma_ || !input) return nullptr;

    cl_context ctx = cl_ctx_.context();
    const size_t out_bytes = (size_t)seq_len * (size_t)hidden_size_ * sizeof(nnopt_storage_t);

    cl_int err = CL_SUCCESS;
    // Persistent output buffer — grown on demand, reused across calls.
    if (seq_len > output_capacity_M_ || !output_buf_) {
        if (output_buf_) { clReleaseMemObject(output_buf_); output_buf_ = nullptr; }
        output_buf_ = clCreateBuffer(ctx, CL_MEM_READ_WRITE, out_bytes, nullptr, &err);
        if (err != CL_SUCCESS) {
            NNOPT_ERROR_FMT("LayerNorm: clCreateBuffer failed: %d", (int)err);
            return nullptr;
        }
        output_capacity_M_ = seq_len;
    }
    cl_mem output = output_buf_;

    int arg = 0;
    err = clSetKernelArg(kernel_, arg++, sizeof(cl_mem), &input);
    if (err != CL_SUCCESS) {
        NNOPT_ERROR_FMT("LayerNorm: setArg %d failed: %d", arg - 1, (int)err);
        return nullptr;
    }
    err = clSetKernelArg(kernel_, arg++, sizeof(cl_mem), &gamma_);
    if (err != CL_SUCCESS) {
        NNOPT_ERROR_FMT("LayerNorm: setArg %d failed: %d", arg - 1, (int)err);
        return nullptr;
    }
    err = clSetKernelArg(kernel_, arg++, sizeof(cl_mem), &output);
    if (err != CL_SUCCESS) {
        NNOPT_ERROR_FMT("LayerNorm: setArg %d failed: %d", arg - 1, (int)err);
        return nullptr;
    }
    err = clSetKernelArg(kernel_, arg++, sizeof(int), &hidden_size_);
    if (err != CL_SUCCESS) {
        NNOPT_ERROR_FMT("LayerNorm: setArg %d failed: %d", arg - 1, (int)err);
        return nullptr;
    }
    err = clSetKernelArg(kernel_, arg++, sizeof(float), &eps_);
    if (err != CL_SUCCESS) {
        NNOPT_ERROR_FMT("LayerNorm: setArg %d failed: %d", arg - 1, (int)err);
        return nullptr;
    }

    // Cooperative dispatch: 1 WG per row, 64 threads cooperate over hidden.
    // Matches kernels/layer_norm.cl::rms_norm vec4 + tree-reduce template.
    const size_t WG = 64;
    const size_t gws[1] = {(size_t)seq_len * WG};
    const size_t lws[1] = {WG};
    cl_event* prof_evt = KernelProfiler::event_for("rmsnorm");
    err = clEnqueueNDRangeKernel(queue, kernel_, 1, nullptr, gws, lws, 0, nullptr, prof_evt);
    if (err != CL_SUCCESS) {
        NNOPT_ERROR_FMT("LayerNorm: enqueue failed: %d", (int)err);
        return nullptr;
    }

    // Caller owns one ref via clReleaseMemObject after consuming. We keep
    // our persistent ref by retaining once per return.
    clRetainMemObject(output);
    return output;
}
