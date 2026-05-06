// Reference: model_info/transformers_src/modeling_llama.py:53-72 LlamaRMSNorm.forward

#include "layers/layer_norm.h"
#include "opencl_context.h"
#include "weights.h"
#include "debug_utils.h"
#include "utils.h"
#include "model_config.h"
#include "prof.h"
#include <CL/cl.h>
#include <string>

static inline bool set_arg_checked(cl_kernel k, cl_uint idx, size_t sz, const void* val, const char* what) {
    cl_int err = clSetKernelArg(k, idx, sz, val);
    if (err != CL_SUCCESS) {
        NNOPT_ERROR_FMT("clSetKernelArg %s failed: %d", what, err);
        return false;
    }
    return true;
}

LayerNorm::LayerNorm(OpenCLContext& cl_ctx, Weights& weights, int layer_idx, bool is_post_attn)
    : cl_ctx_(cl_ctx), weights_(weights), layer_idx_(layer_idx), is_post_attn_(is_post_attn), is_final_norm_(false) {}

LayerNorm::LayerNorm(OpenCLContext& cl_ctx, Weights& weights, int layer_idx, bool is_post_attn, bool is_final_norm)
    : cl_ctx_(cl_ctx), weights_(weights), layer_idx_(layer_idx), is_post_attn_(is_post_attn), is_final_norm_(is_final_norm) {}

LayerNorm::~LayerNorm() {
    if (decode_out_buf_) clReleaseMemObject(decode_out_buf_);
    if (kernel_) clReleaseKernel(kernel_);
    if (program_) clReleaseProgram(program_);
}

bool LayerNorm::set_weights() {
    // Reference: (model metadata)::weight_keys
    // NOTE: final RMSNorm uses key "model.norm.weight" (no layer index).
    // For per-layer norms, keys are always "model.layers.<i>.(input|post_attention)_layernorm.weight".
    if (is_final_norm_) {
        weight_key_ = "model.norm.weight";
    } else {
        // Defensive: layer_idx_ must be a valid layer index for non-final norms.
        if (layer_idx_ < 0) {
            NNOPT_ERROR_FMT("LayerNorm: invalid layer_idx=%d for non-final norm", layer_idx_);
            return false;
        }
        if (is_post_attn_) {
            weight_key_ = "model.layers." + std::to_string(layer_idx_) + ".post_attention_layernorm.weight";
        } else {
            weight_key_ = "model.layers." + std::to_string(layer_idx_) + ".input_layernorm.weight";
        }
    }

    weight_ = weights_.get_buffer(weight_key_);
    if (!weight_) {
        NNOPT_ERROR_FMT("LayerNorm: missing weight buffer for key %s", weight_key_.c_str());
        return false;
    }
    return true;
}

bool LayerNorm::initialize() {
    program_ = cl_ctx_.build_program_from_file(
        "kernels/rmsnorm.cl",
#ifdef NNOPT_USE_FP16
        "-DNNOPT_USE_FP16=1 -DUSE_FP16=1"
#else
        ""
#endif
    );
    if (!program_) {
        NNOPT_ERROR("Failed to build kernels/rmsnorm.cl");
        return false;
    }

    cl_int err = CL_SUCCESS;
    kernel_ = clCreateKernel(program_, "rmsnorm_forward", &err);
    if (err != CL_SUCCESS || !kernel_) {
        NNOPT_ERROR_FMT("clCreateKernel rmsnorm_forward failed: %d", err);
        return false;
    }

    if (!set_weights()) return false;

    // Pre-allocate persistent decode output buffer (M=1 path).
    cl_int err2 = CL_SUCCESS;
    decode_out_buf_ = clCreateBuffer(cl_ctx_.context(), CL_MEM_READ_WRITE,
                                     (size_t)MODEL_CONFIG::HIDDEN_SIZE * sizeof(nnopt_storage_t),
                                     nullptr, &err2);
    if (err2 != CL_SUCCESS || !decode_out_buf_) {
        NNOPT_ERROR_FMT("LayerNorm: alloc decode_out_buf_ failed: %d", err2);
        return false;
    }

    return true;
}

cl_mem LayerNorm::forward(cl_command_queue queue, cl_mem input, int rows) {
    const std::string tag = is_final_norm_ ? "final_norm" : weight_key_;
    NNOPT_LAYER_CHECK_INPUT(tag.c_str(), queue, input, (size_t)rows * MODEL_CONFIG::HIDDEN_SIZE);

    cl_int err = CL_SUCCESS;
    cl_context ctx = cl_ctx_.context();
    const int cols = MODEL_CONFIG::HIDDEN_SIZE;

    cl_mem out = clCreateBuffer(ctx, CL_MEM_READ_WRITE, (size_t)rows * cols * sizeof(nnopt_storage_t), nullptr, &err);
    if (err != CL_SUCCESS || !out) {
        NNOPT_ERROR_FMT("LayerNorm alloc out failed: %d", err);
        return nullptr;
    }

    const float eps = (float)MODEL_CONFIG::RMS_NORM_EPS;

    if (!set_arg_checked(kernel_, 0, sizeof(cl_mem), &input, "input")) { clReleaseMemObject(out); return nullptr; }
    if (!set_arg_checked(kernel_, 1, sizeof(cl_mem), &weight_, "weight")) { clReleaseMemObject(out); return nullptr; }
    if (!set_arg_checked(kernel_, 2, sizeof(cl_mem), &out, "output")) { clReleaseMemObject(out); return nullptr; }
    if (!set_arg_checked(kernel_, 3, sizeof(int), &rows, "rows")) { clReleaseMemObject(out); return nullptr; }
    if (!set_arg_checked(kernel_, 4, sizeof(int), &cols, "cols")) { clReleaseMemObject(out); return nullptr; }
    if (!set_arg_checked(kernel_, 5, sizeof(float), &eps, "eps")) { clReleaseMemObject(out); return nullptr; }

    // kernels/rmsnorm.cl declares reqd_work_group_size(64,1,1) and uses get_group_id(0)
    // as the row index. Host launch MUST use gws=rows*64 and lws=64.
    const size_t lws = 64;
    const size_t gws = (size_t)rows * lws;
    err = nnopt_prof::enqueue(queue, kernel_, 1, nullptr, &gws, &lws, 0, nullptr, nullptr);
    if (err != CL_SUCCESS) {
        NNOPT_ERROR_FMT("rmsnorm_forward dispatch failed: %d", err);
        clReleaseMemObject(out);
        return nullptr;
    }

    NNOPT_LAYER_CHECK(tag.c_str(), queue, out, (size_t)rows * MODEL_CONFIG::HIDDEN_SIZE);
    return out;
}

// Decode fast path: writes into the persistent decode_out_buf_.
// Returns decode_out_buf_ — NOT owned by the caller. Do NOT clReleaseMemObject it.
cl_mem LayerNorm::forward_decode(cl_command_queue queue, cl_mem input) {
    const int rows = 1;
    const int cols = MODEL_CONFIG::HIDDEN_SIZE;
    const float eps = (float)MODEL_CONFIG::RMS_NORM_EPS;

    if (!set_arg_checked(kernel_, 0, sizeof(cl_mem), &input,          "input"))  return nullptr;
    if (!set_arg_checked(kernel_, 1, sizeof(cl_mem), &weight_,        "weight")) return nullptr;
    if (!set_arg_checked(kernel_, 2, sizeof(cl_mem), &decode_out_buf_,"output")) return nullptr;
    if (!set_arg_checked(kernel_, 3, sizeof(int),    &rows,           "rows"))   return nullptr;
    if (!set_arg_checked(kernel_, 4, sizeof(int),    &cols,           "cols"))   return nullptr;
    if (!set_arg_checked(kernel_, 5, sizeof(float),  &eps,            "eps"))    return nullptr;

    const size_t lws = 64;
    const size_t gws = lws;  // rows=1 → 1 workgroup
    cl_int err = nnopt_prof::enqueue(queue, kernel_, 1, nullptr, &gws, &lws, 0, nullptr, nullptr);
    if (err != CL_SUCCESS) {
        NNOPT_ERROR_FMT("LayerNorm::forward_decode dispatch failed: %d", err);
        return nullptr;
    }
    return decode_out_buf_;
}
