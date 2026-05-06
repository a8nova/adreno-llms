// Reference: model_info/transformers_src/modeling_lfm2.py:100-113 Lfm2RMSNorm.forward
#include "layers/operator_norm.h"
#include "debug_utils.h"
#include "kernel_profiler.h"
#include "utils.h"
#include "model_config.h"
#include <CL/cl.h>
#include <string>
#include <vector>

OperatorNorm::OperatorNorm(OpenCLContext& cl_ctx, Weights& weights, const std::string& prefix, int layer_idx)
    : cl_ctx_(cl_ctx), weights_(weights), prefix_(prefix), layer_idx_(layer_idx) {}

OperatorNorm::~OperatorNorm() {
  if (kernel_) clReleaseKernel(kernel_);
  if (program_) clReleaseProgram(program_);
  if (buf_out_) clReleaseMemObject(buf_out_);
}

bool OperatorNorm::initialize() {
  // Use scaffold rmsnorm kernel for correctness.
  program_ = cl_ctx_.build_program_from_file("kernels/rmsnorm.cl");
  if (!program_) return false;
  cl_int err = CL_SUCCESS;
  kernel_ = clCreateKernel(program_, "rmsnorm_forward", &err);
  if (err != CL_SUCCESS) {
    NNOPT_ERROR_FMT("OperatorNorm: clCreateKernel(rmsnorm_forward) failed: %d", (int)err);
    return false;
  }
  // Validate that the expected weight exists. The actual key comes from the contract:
  //   OperatorNorm: model.layers.{i}.operator_norm.weight
  //   Mlp (ffn_norm): model.layers.{i}.ffn_norm.weight
  // plus the embedding_norm case: model.embedding_norm.weight
  cl_mem w = weights_.get_buffer(prefix_ + ".weight");
  if (!w) {
    NNOPT_ERROR_FMT("OperatorNorm: missing weight %s.weight", prefix_.c_str());
    return false;
  }
  return true;
}

cl_mem OperatorNorm::forward(cl_command_queue queue, cl_mem input, int seq_len) {
  if (!kernel_) return nullptr;
  cl_int err = CL_SUCCESS;
  cl_context ctx = cl_ctx_.context();

  const int hidden = MODEL_CONFIG::HIDDEN_SIZE;
  const size_t out_bytes = (size_t)seq_len * (size_t)hidden * sizeof(nnopt_storage_t);
  // Persistent output buffer — lazy-grow.
  if (seq_len > buf_seq_capacity_ || !buf_out_) {
    if (buf_out_) { clReleaseMemObject(buf_out_); buf_out_ = nullptr; }
    buf_out_ = clCreateBuffer(ctx, CL_MEM_READ_WRITE, out_bytes, nullptr, &err);
    if (err != CL_SUCCESS) { NNOPT_ERROR_FMT("OperatorNorm: alloc buf_out: %d", err); return nullptr; }
    buf_seq_capacity_ = seq_len;
  }
  cl_mem out = buf_out_;

  cl_mem weight = weights_.get_buffer(prefix_ + ".weight");
  if (!weight) {
    NNOPT_ERROR_FMT("OperatorNorm: missing weight buffer: %s.weight", prefix_.c_str());
    return nullptr;
  }

  // rmsnorm_forward(__global const storage_t* x, __global const storage_t* w, __global storage_t* out,
  //                int rows, int cols, float eps)
  int rows = seq_len;
  int cols = hidden;
  // Reference: model_info/transformers_src/modeling_lfm2.py:105-110 Lfm2RMSNorm.forward
  // eps comes from config.norm_eps.
  const float eps = MODEL_CONFIG::NORM_EPS;

  // Dump naming: use explicit names that match reference dumps.
  // - ffn_norm_i   corresponds to model.layers.i.ffn_norm (RMSNorm hidden_size)
  // - operator_norm_i corresponds to model.layers.i.operator_norm (RMSNorm hidden_size)
  // - embedding_norm corresponds to model.embedding_norm
  std::string dump_name = prefix_;
  if (prefix_.find(".ffn_norm") != std::string::npos && layer_idx_ >= 0) {
    dump_name = "ffn_norm_" + std::to_string(layer_idx_);
  } else if (prefix_.find(".operator_norm") != std::string::npos && layer_idx_ >= 0) {
    dump_name = "operator_norm_" + std::to_string(layer_idx_);
  } else if (prefix_ == "model.embedding_norm") {
    dump_name = "embedding_norm";
  }

  // Dump sites: also emit the intermediate tensors requested by the OperatorNorm contract.
  // These are used by SxS to pinpoint failures inside the RMSNorm.
  // - block0_sub_lfm2rmsnorm_hidden_states  == input (fp16/fp32)
  // - block0_sub_lfm2rmsnorm_variance       == mean(x^2) per row (fp32)
  if (layer_idx_ == 0 && dump_name == "operator_norm_0") {
    NNOPT_LAYER_CHECK("block0_sub_lfm2rmsnorm_hidden_states", queue, input, (size_t)rows * (size_t)cols);

    std::vector<float> var(rows, 0.0f);
    // NOTE: this is debug-only; we read back storage and compute variance on CPU.
    // It is acceptable because NNOPT_LAYER_CHECK itself is debug-only.
#ifdef NNOPT_USE_FP16
    std::vector<cl_half> x_host((size_t)rows * (size_t)cols);
    err = clEnqueueReadBuffer(queue, input, CL_TRUE, 0, x_host.size() * sizeof(cl_half), x_host.data(), 0, nullptr, nullptr);
    if (err == CL_SUCCESS) {
      for (int r = 0; r < rows; ++r) {
        double acc = 0.0;
        const size_t base = (size_t)r * (size_t)cols;
        for (int c = 0; c < cols; ++c) {
          float v = nnopt_f16_to_f32((uint16_t)x_host[base + (size_t)c]);
          acc += (double)v * (double)v;
        }
        var[(size_t)r] = (float)(acc / (double)cols);
      }
    }
#else
    std::vector<float> x_host((size_t)rows * (size_t)cols);
    err = clEnqueueReadBuffer(queue, input, CL_TRUE, 0, x_host.size() * sizeof(float), x_host.data(), 0, nullptr, nullptr);
    if (err == CL_SUCCESS) {
      for (int r = 0; r < rows; ++r) {
        double acc = 0.0;
        const size_t base = (size_t)r * (size_t)cols;
        for (int c = 0; c < cols; ++c) {
          float v = x_host[base + (size_t)c];
          acc += (double)v * (double)v;
        }
        var[(size_t)r] = (float)(acc / (double)cols);
      }
    }
#endif

    // Upload var (fp32) so SxS can compare against reference variance.
    cl_mem var_buf = clCreateBuffer(ctx, CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR, var.size() * sizeof(float), var.data(), &err);
    if (err == CL_SUCCESS && var_buf) {
      NNOPT_LAYER_CHECK("block0_sub_lfm2rmsnorm_variance", queue, var_buf, (size_t)rows);
      clReleaseMemObject(var_buf);
    }
  }

  // IMPORTANT: this class implements RMSNorm (Lfm2RMSNorm). It expects cols == HIDDEN_SIZE.
  // If this instance is a per-head norm (q_layernorm/k_layernorm), it must NOT use this class.
  if (weights_.has_tensor(prefix_ + ".weight")) {
    const size_t w_elems = weights_.get_num_elements(prefix_ + ".weight");
    if ((int)w_elems != cols) {
      NNOPT_ERROR_FMT(
          "OperatorNorm: weight elems (%zu) != cols (%d) for %s.weight — likely wrong norm class/dims",
          w_elems,
          cols,
          prefix_.c_str());
     
      return nullptr;
    }
  }

  NNOPT_LAYER_CHECK_INPUT(dump_name.c_str(), queue, input, (size_t)rows * (size_t)cols);

  int arg = 0;
  err = clSetKernelArg(kernel_, arg++, sizeof(cl_mem), &input);
  if (err != CL_SUCCESS) { NNOPT_ERROR_FMT("OperatorNorm: setArg %d failed: %d", arg - 1, (int)err); return nullptr; }
  err = clSetKernelArg(kernel_, arg++, sizeof(cl_mem), &weight);
  if (err != CL_SUCCESS) { NNOPT_ERROR_FMT("OperatorNorm: setArg %d failed: %d", arg - 1, (int)err); return nullptr; }
  err = clSetKernelArg(kernel_, arg++, sizeof(cl_mem), &out);
  if (err != CL_SUCCESS) { NNOPT_ERROR_FMT("OperatorNorm: setArg %d failed: %d", arg - 1, (int)err); return nullptr; }
  err = clSetKernelArg(kernel_, arg++, sizeof(int), &rows);
  if (err != CL_SUCCESS) { NNOPT_ERROR_FMT("OperatorNorm: setArg %d failed: %d", arg - 1, (int)err); return nullptr; }
  err = clSetKernelArg(kernel_, arg++, sizeof(int), &cols);
  if (err != CL_SUCCESS) { NNOPT_ERROR_FMT("OperatorNorm: setArg %d failed: %d", arg - 1, (int)err); return nullptr; }
  err = clSetKernelArg(kernel_, arg++, sizeof(float), &eps);
  if (err != CL_SUCCESS) { NNOPT_ERROR_FMT("OperatorNorm: setArg %d failed: %d", arg - 1, (int)err); return nullptr; }

  // kernels/rmsnorm.cl is 1D: gws = rows * WG_SIZE, lws = WG_SIZE.
  // WG_SIZE is hard-coded to 64 in the kernel.
  const size_t wg_size = 64;
  size_t gws[1] = {(size_t)rows * wg_size};
  size_t lws[1] = {wg_size};
  err = clEnqueueNDRangeKernel(queue, kernel_, 1, nullptr, gws, lws, 0, nullptr, KernelProfiler::event_for("rmsnorm"));
  if (err != CL_SUCCESS) {
    NNOPT_ERROR_FMT("OperatorNorm: enqueue rmsnorm failed: %d", (int)err);
   
    return nullptr;
  }

  NNOPT_LAYER_CHECK(dump_name.c_str(), queue, out, (size_t)rows * (size_t)cols);
  return out;
}
