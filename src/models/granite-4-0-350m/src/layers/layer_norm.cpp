// Reference: model_info/transformers_src/modeling_granitemoehybrid.py:735-752 GraniteMoeHybridRMSNormGated.forward

#include "layers/layer_norm.h"
#include "opencl_context.h"
#include "weights.h"
#include "debug_utils.h"
#include "utils.h"
#include "model_config.h"
#include "kernel_profiler.h"

#include <clblast.h>
#include <cmath>
#include <cstdio>
#include <string>
#include <vector>

static std::string fmt_key(const std::string& templ, int i) {
    std::string out = templ;
    size_t pos = out.find("{i}");
    if (pos != std::string::npos) out.replace(pos, 3, std::to_string(i));
    return out;
}

LayerNorm::LayerNorm(OpenCLContext& cl_ctx, Weights& weights, int layer_idx, LayerNorm::Kind kind)
    : cl_ctx_(cl_ctx), weights_(weights), layer_idx_(layer_idx), kind_(kind) {}

LayerNorm::~LayerNorm() {
    if (kernel_) clReleaseKernel(kernel_);
    if (kernel_v2_) clReleaseKernel(kernel_v2_);
    if (program_) clReleaseProgram(program_);
}

bool LayerNorm::initialize() {
    // PROGRAM-INIT-OK
    program_ = cl_ctx_.build_program_from_file("kernels/layer_norm.cl");
    if (!program_) {
        NNOPT_ERROR("LayerNorm: failed to build kernels/layer_norm.cl");
        return false;
    }

    cl_int err = CL_SUCCESS;
    kernel_ = clCreateKernel(program_, "rmsnorm_forward", &err);
    if (err != CL_SUCCESS || !kernel_) {
        NNOPT_ERROR_FMT("LayerNorm: clCreateKernel(rmsnorm_forward) failed (%d)", err);
        return false;
    }
    // v2 kernel uses WG=64 cooperative reduction. Use it when hidden % 256 == 0
    // (true for HIDDEN_SIZE=1024). Falls back to v1 otherwise.
    kernel_v2_ = clCreateKernel(program_, "rmsnorm_forward_v2", &err);
    if (err != CL_SUCCESS || !kernel_v2_) {
        NNOPT_ERROR_FMT("LayerNorm: clCreateKernel(rmsnorm_forward_v2) failed (%d)", err);
        return false;
    }

    std::string wkey;
    if (kind_ == Kind::FinalNorm) {
        wkey = "model.norm.weight";
    } else if (kind_ == Kind::InputLayerNorm) {
        wkey = fmt_key("model.layers.{i}.input_layernorm.weight", layer_idx_);
    } else {
        wkey = fmt_key("model.layers.{i}.post_attention_layernorm.weight", layer_idx_);
    }

    weight_ = weights_.get_buffer(wkey);
    if (!weight_) {
        NNOPT_ERROR_FMT("LayerNorm: missing weight %s", wkey.c_str());
        return false;
    }

    return true;
}

cl_mem LayerNorm::forward(cl_command_queue queue, cl_mem input, int rows) {
    NNOPT_LAYER_FWD("layer_norm");
    if (!kernel_ || !weight_) {
        NNOPT_ERROR("LayerNorm::forward called before initialize()");
        return nullptr;
    }

    const int hidden = MODEL_CONFIG::HIDDEN_SIZE;
    const size_t n_elem = (size_t)rows * (size_t)hidden;

    // Per-instance dump name so SxS can match every layernorm to its
    // reference (input_layernorm_K / post_attention_layernorm_K / final_norm).
    // The legacy "layer_norm" name made all 56+ calls share one filename
    // (first-writer-wins), losing visibility into every layer past the first.
    char dump_name[64];
    if (kind_ == Kind::FinalNorm) {
        std::snprintf(dump_name, sizeof(dump_name), "final_norm");
    } else if (kind_ == Kind::InputLayerNorm) {
        std::snprintf(dump_name, sizeof(dump_name), "input_layernorm_%d", layer_idx_);
    } else {
        std::snprintf(dump_name, sizeof(dump_name), "post_attention_layernorm_%d", layer_idx_);
    }

    NNOPT_LAYER_CHECK_INPUT(dump_name, queue, input, n_elem);

    cl_int err = CL_SUCCESS;
    cl_context ctx = cl_ctx_.context();
    cl_mem out = clCreateBuffer(ctx, CL_MEM_READ_WRITE, n_elem * sizeof(nnopt_storage_t), nullptr, &err);
    if (err != CL_SUCCESS || !out) {
        NNOPT_ERROR_FMT("LayerNorm: clCreateBuffer out failed (%d)", err);
        return nullptr;
    }

    float eps = MODEL_CONFIG::RMS_NORM_EPS;
    const int RMSNORM_WG = 64;
    const bool use_v2 = (hidden % (RMSNORM_WG * 4) == 0);
    cl_kernel k = use_v2 ? kernel_v2_ : kernel_;
    const char* prof_label = use_v2 ? "rmsnorm_forward_v2" : "rmsnorm_forward";
    int arg = 0;
    err = clSetKernelArg(k, arg++, sizeof(cl_mem), &input);
    if (err != CL_SUCCESS) { NNOPT_ERROR_FMT("LayerNorm: set arg input failed (%d)", err); clReleaseMemObject(out); return nullptr; }
    err = clSetKernelArg(k, arg++, sizeof(cl_mem), &weight_);
    if (err != CL_SUCCESS) { NNOPT_ERROR_FMT("LayerNorm: set arg weight failed (%d)", err); clReleaseMemObject(out); return nullptr; }
    // rmsnorm_forward signature in kernels/layer_norm.cl has NO gate arg.
    err = clSetKernelArg(k, arg++, sizeof(cl_mem), &out);
    if (err != CL_SUCCESS) { NNOPT_ERROR_FMT("LayerNorm: set arg out failed (%d)", err); clReleaseMemObject(out); return nullptr; }
    err = clSetKernelArg(k, arg++, sizeof(int), &rows);
    if (err != CL_SUCCESS) { NNOPT_ERROR_FMT("LayerNorm: set arg rows failed (%d)", err); clReleaseMemObject(out); return nullptr; }
    err = clSetKernelArg(k, arg++, sizeof(int), &hidden);
    if (err != CL_SUCCESS) { NNOPT_ERROR_FMT("LayerNorm: set arg hidden failed (%d)", err); clReleaseMemObject(out); return nullptr; }
    err = clSetKernelArg(k, arg++, sizeof(float), &eps);
    if (err != CL_SUCCESS) { NNOPT_ERROR_FMT("LayerNorm: set arg eps failed (%d)", err); clReleaseMemObject(out); return nullptr; }

    if (use_v2) {
        size_t gws_v2[1] = { (size_t)rows * (size_t)RMSNORM_WG };
        size_t lws_v2[1] = { (size_t)RMSNORM_WG };
        err = clEnqueueNDRangeKernel(queue, k, 1, nullptr, gws_v2, lws_v2, 0, nullptr, KernelProfiler::event_for(prof_label));
    } else {
        size_t gws[1] = { (size_t)rows };
        err = clEnqueueNDRangeKernel(queue, k, 1, nullptr, gws, nullptr, 0, nullptr, KernelProfiler::event_for(prof_label));
    }
    if (err != CL_SUCCESS) {
        NNOPT_ERROR_FMT("LayerNorm: dispatch failed (%d)", err);
        clReleaseMemObject(out);
        return nullptr;
    }

    NNOPT_LAYER_CHECK(dump_name, queue, out, n_elem);
    return out;
}
