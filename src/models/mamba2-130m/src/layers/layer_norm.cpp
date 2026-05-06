// Reference: model_info/transformers_src/modeling_mamba2.py (RMSNorm/LayerNorm used in Mamba2 block); scaffold contract (model metadata)

#include "layers/layer_norm.h"

#include "debug_utils.h"
#include "kernel_profiler.h"
#include "opencl_context.h"
#include "utils.h"
#include "weights.h"

#include <CL/cl.h>
#include <string>

LayerNorm::LayerNorm(OpenCLContext& cl_ctx,
                     Weights& weights,
                     const std::string& weight_key,
                     int hidden_size,
                     float eps,
                     int layer_idx)
    : cl_ctx_(cl_ctx),
      weights_(weights),
      weight_key_(weight_key),
      hidden_size_(hidden_size),
      // Mamba2 uses RMSNorm eps = config.layer_norm_epsilon (=1e-5 in this port),
      // but the PyTorch reference for MambaRMSNormGated/Mamba2RMSNorm uses float32
      // math and typically eps=1e-6.
      // Reference: model_info/transformers_src/modeling_mamba2.py:103-120 MambaRMSNormGated.forward
      // Reference: model_info/transformers_src/modeling_mamba2.py:503-521 Mamba2RMSNorm.forward
      // Use the contract-provided eps if it is non-sentinel; otherwise default to 1e-6.
      eps_((eps > 0.0f) ? eps : 1e-6f),
      layer_idx_(layer_idx) {}

LayerNorm::~LayerNorm() {
    if (kernel_) clReleaseKernel(kernel_);
    if (program_) clReleaseProgram(program_);
    if (buf_out_) clReleaseMemObject(buf_out_);
}

bool LayerNorm::ensure_out_(int rows) {
    if (rows <= buf_capacity_rows_ && buf_out_ != nullptr) return true;
    if (buf_out_) { clReleaseMemObject(buf_out_); buf_out_ = nullptr; buf_capacity_rows_ = 0; }
    cl_int err = CL_SUCCESS;
    const size_t bytes = (size_t)rows * (size_t)hidden_size_ * sizeof(nnopt_storage_t);
    buf_out_ = clCreateBuffer(cl_ctx_.context(), CL_MEM_READ_WRITE, bytes, nullptr, &err);
    if (err != CL_SUCCESS || !buf_out_) {
        NNOPT_ERROR_FMT("LayerNorm[%d]: ensure_out_ failed (%d)", layer_idx_, (int)err);
        return false;
    }
    buf_capacity_rows_ = rows;
    return true;
}

bool LayerNorm::initialize() {
    gamma_ = weights_.get_buffer(weight_key_);
    if (!gamma_) {
        NNOPT_ERROR_FMT("LayerNorm missing weight: %s", weight_key_.c_str());
        return false;
    }

    program_ = cl_ctx_.build_program_from_file("kernels/layer_norm.cl");
    if (!program_) {
        NNOPT_ERROR("LayerNorm failed to build program kernels/layer_norm.cl");
        return false;
    }

    cl_int err = CL_SUCCESS;
    kernel_ = clCreateKernel(program_, "rms_norm", &err);
    if (err != CL_SUCCESS || !kernel_) {
        NNOPT_ERROR_FMT("clCreateKernel(rms_norm) failed: %d", (int)err);
        return false;
    }
    return true;
}

cl_mem LayerNorm::forward(cl_command_queue queue, cl_mem input, int seq_len) {
    NNOPT_CHECKPOINT("LayerNorm::forward");

    // Per-instance check so we can see if/when activations collapse.
    // IMPORTANT: this is the INPUT to norm.
    NNOPT_LAYER_CHECK_FMT("norm_%d_input", layer_idx_, queue, input,
                         (size_t)seq_len * (size_t)hidden_size_);

    if (!ensure_out_(seq_len)) return nullptr;
    cl_mem output = buf_out_;

    cl_int err = CL_SUCCESS;
    int arg = 0;
    err  = clSetKernelArg(kernel_, arg++, sizeof(cl_mem), &input);
    err |= clSetKernelArg(kernel_, arg++, sizeof(cl_mem), &gamma_);
    err |= clSetKernelArg(kernel_, arg++, sizeof(cl_mem), &output);
    err |= clSetKernelArg(kernel_, arg++, sizeof(int),    &hidden_size_);
    err |= clSetKernelArg(kernel_, arg++, sizeof(float),  &eps_);
    if (err != CL_SUCCESS) {
        NNOPT_ERROR_FMT("LayerNorm: setKernelArg failed: %d", (int)err);
        return nullptr;
    }

    // Cooperative dispatch: 1 WG per row, WG=64 threads cooperate over hidden.
    const size_t WG = 64;
    size_t gws[1] = {(size_t)seq_len * WG};
    size_t lws[1] = {WG};
    cl_event* evt = KernelProfiler::event_for("rms_norm");
    err = clEnqueueNDRangeKernel(queue, kernel_, 1, nullptr, gws, lws, 0, nullptr, evt);
    if (err != CL_SUCCESS) {
        NNOPT_ERROR_FMT("LayerNorm: enqueue kernel failed: %d", (int)err);
        return nullptr;
    }

    // IMPORTANT: this is the OUTPUT of norm.
    NNOPT_LAYER_CHECK_FMT("norm_%d", layer_idx_, queue, output,
                         (size_t)seq_len * (size_t)hidden_size_);

    return output;
}
