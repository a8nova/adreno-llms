// Reference: https://raw.githubusercontent.com/huggingface/transformers/main/src/transformers/models/llama/modeling_llama.py
// LlamaRMSNorm.forward:
//   input_dtype = hidden_states.dtype
//   hidden_states = hidden_states.to(torch.float32)
//   variance = hidden_states.pow(2).mean(-1, keepdim=True)
//   hidden_states = hidden_states * torch.rsqrt(variance + eps)
//   return self.weight * hidden_states.to(input_dtype)
//
// This file implements RMSNorm on OpenCL, using kernels/rms_norm.cl.

#include "opencl_context.h"
#include "weights.h"
#include "debug_utils.h"
#include "utils.h"

#include <CL/cl.h>
#include <cstring>
#include <string>

namespace {
struct RmsNormState {
    bool initialized = false;
    cl_program program = nullptr;
    cl_kernel kernel = nullptr;
    cl_kernel kernel_residual = nullptr;
};

RmsNormState& state() { static RmsNormState s; return s; }

bool ensure_initialized(OpenCLContext& cl_ctx) {
    auto& s = state();
    if (s.initialized) return true;
    s.program = cl_ctx.build_program_from_file("kernels/rms_norm.cl");  // PROGRAM-INIT-OK
    if (!s.program) {
        NNOPT_ERROR("op_LlamaRMSNorm: build_program_from_file(rms_norm.cl) failed");
        return false;
    }
    cl_int err = CL_SUCCESS;
    s.kernel = clCreateKernel(s.program, "rms_norm_forward", &err);
    if (err != CL_SUCCESS || !s.kernel) {
        NNOPT_ERROR_FMT("op_LlamaRMSNorm: clCreateKernel failed: %d", (int)err);
        return false;
    }
    s.kernel_residual = clCreateKernel(s.program, "rms_norm_residual_forward", &err);
    if (err != CL_SUCCESS || !s.kernel_residual) {
        NNOPT_ERROR_FMT("op_LlamaRMSNorm: clCreateKernel(residual) failed: %d", (int)err);
        return false;
    }
    s.initialized = true;
    return true;
}
} // namespace

extern "C" cl_mem op_LlamaRMSNorm(OpenCLContext& cl_ctx,
                                 Weights& weights,
                                 cl_command_queue queue,
                                 cl_mem input,
                                 int rows,
                                 int cols,
                                 float eps,
                                 const char* weight_key_w) {
    if (!input) {
        NNOPT_ERROR("op_LlamaRMSNorm: input is null");
        return nullptr;
    }
    if (!ensure_initialized(cl_ctx)) return nullptr;

    cl_mem w = weights.get_buffer(std::string(weight_key_w));
    if (!w) {
        NNOPT_ERROR_FMT("op_LlamaRMSNorm: missing weight tensor: %s", weight_key_w);
        return nullptr;
    }

    cl_int err = CL_SUCCESS;
    cl_context ctx = cl_ctx.context();
    const size_t out_bytes = (size_t)rows * (size_t)cols * sizeof(nnopt_storage_t);
    cl_mem out = clCreateBuffer(ctx, CL_MEM_READ_WRITE, out_bytes, nullptr, &err);
    if (err != CL_SUCCESS || !out) {
        NNOPT_ERROR_FMT("op_LlamaRMSNorm: clCreateBuffer out failed: %d", (int)err);
        return nullptr;
    }

    cl_kernel k = state().kernel;
    if (!set_arg_checked(k, 0, sizeof(cl_mem), &input, "x")) { clReleaseMemObject(out); return nullptr; }
    if (!set_arg_checked(k, 1, sizeof(cl_mem), &w, "weight")) { clReleaseMemObject(out); return nullptr; }
    if (!set_arg_checked(k, 2, sizeof(cl_mem), &out, "out")) { clReleaseMemObject(out); return nullptr; }
    if (!set_arg_checked(k, 3, sizeof(int), &rows, "rows")) { clReleaseMemObject(out); return nullptr; }
    if (!set_arg_checked(k, 4, sizeof(int), &cols, "cols")) { clReleaseMemObject(out); return nullptr; }
    if (!set_arg_checked(k, 5, sizeof(float), &eps, "eps")) { clReleaseMemObject(out); return nullptr; }

    // reqd_work_group_size(WG_SIZE,1,1) where WG_SIZE=64 in rms_norm.cl.
    const size_t lws[1] = {64};
    const size_t gws[1] = {(size_t)rows * lws[0]};
    err = clEnqueueNDRangeKernel(queue, k, 1, nullptr, gws, lws, 0, nullptr, nullptr);
    if (err != CL_SUCCESS) {
        NNOPT_ERROR_FMT("op_LlamaRMSNorm: clEnqueueNDRangeKernel failed: %d", (int)err);
        clReleaseMemObject(out);
        out = nullptr;
    }
    return out;
}

// Fused: x += residual; out = rmsnorm(x); returns out (new buffer). `x` is
// mutated in place so callers reuse it as the next residual stream.
extern "C" cl_mem op_LlamaRMSNormWithResidual(OpenCLContext& cl_ctx,
                                              Weights& weights,
                                              cl_command_queue queue,
                                              cl_mem x_inout,
                                              cl_mem residual,
                                              int rows,
                                              int cols,
                                              float eps,
                                              const char* weight_key_w) {
    if (!x_inout || !residual) {
        NNOPT_ERROR("op_LlamaRMSNormWithResidual: null input/residual");
        return nullptr;
    }
    if (!ensure_initialized(cl_ctx)) return nullptr;
    cl_mem w = weights.get_buffer(std::string(weight_key_w));
    if (!w) {
        NNOPT_ERROR_FMT("op_LlamaRMSNormWithResidual: missing weight: %s", weight_key_w);
        return nullptr;
    }
    cl_int err = CL_SUCCESS;
    cl_context ctx = cl_ctx.context();
    const size_t out_bytes = (size_t)rows * (size_t)cols * sizeof(nnopt_storage_t);
    cl_mem out = clCreateBuffer(ctx, CL_MEM_READ_WRITE, out_bytes, nullptr, &err);
    if (err != CL_SUCCESS || !out) {
        NNOPT_ERROR("op_LlamaRMSNormWithResidual: alloc out failed");
        return nullptr;
    }
    cl_kernel k = state().kernel_residual;
    if (!set_arg_checked(k, 0, sizeof(cl_mem), &x_inout, "x")) { clReleaseMemObject(out); return nullptr; }
    if (!set_arg_checked(k, 1, sizeof(cl_mem), &residual, "residual")) { clReleaseMemObject(out); return nullptr; }
    if (!set_arg_checked(k, 2, sizeof(cl_mem), &w, "weight")) { clReleaseMemObject(out); return nullptr; }
    if (!set_arg_checked(k, 3, sizeof(cl_mem), &out, "out")) { clReleaseMemObject(out); return nullptr; }
    if (!set_arg_checked(k, 4, sizeof(int), &rows, "rows")) { clReleaseMemObject(out); return nullptr; }
    if (!set_arg_checked(k, 5, sizeof(int), &cols, "cols")) { clReleaseMemObject(out); return nullptr; }
    if (!set_arg_checked(k, 6, sizeof(float), &eps, "eps")) { clReleaseMemObject(out); return nullptr; }
    const size_t lws[1] = {64};
    const size_t gws[1] = {(size_t)rows * lws[0]};
    err = clEnqueueNDRangeKernel(queue, k, 1, nullptr, gws, lws, 0, nullptr, nullptr);
    if (err != CL_SUCCESS) {
        NNOPT_ERROR_FMT("op_LlamaRMSNormWithResidual: dispatch failed: %d", (int)err);
        clReleaseMemObject(out);
        return nullptr;
    }
    return out;
}
