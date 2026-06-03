#pragma once

#include "opencl_context.h"
#include "weights.h"
#include <CL/cl.h>
#include <string>

cl_program lfm2_program(OpenCLContext& cl_ctx);
cl_mem lfm2_alloc(OpenCLContext& cl_ctx, size_t elems, const char* label);
void   lfm2_pool_reclaim();
void   lfm2_pool_set_active(bool active);  // enable pooling only during decode (seq_len=1)
void   lfm2_pool_clear();                  // fully release all pool memory (outstanding + free)
bool lfm2_kernel1(cl_command_queue queue, cl_kernel kernel, size_t global, const char* label);
bool lfm2_kernel1_lws(cl_command_queue queue, cl_kernel kernel, size_t global, size_t local, const char* label);

cl_mem lfm2_rms_norm(OpenCLContext& cl_ctx, Weights& weights, cl_command_queue queue,
                     cl_mem input, int rows, int cols, const std::string& weight_key);

// Fused residual_add + rms_norm. Computes out_sum=a+b and out_norm=rms_norm(out_sum)*weight
// in a single dispatch (saves ~32 kernel launches per decode token vs the
// element_add → rms_norm two-call sequence). Both outputs are caller-owned.
bool lfm2_rms_norm_add(OpenCLContext& cl_ctx, Weights& weights, cl_command_queue queue,
                      cl_mem a, cl_mem b, int rows, int cols,
                      const std::string& weight_key,
                      cl_mem* out_sum, cl_mem* out_norm);

cl_mem lfm2_mlp(OpenCLContext& cl_ctx, Weights& weights, cl_command_queue queue,
                cl_mem input, int rows, int hidden_size, int intermediate_size,
                const std::string& prefix, int layer_idx);

// Register the OpenCLContext for the pytorch_linear M==1 fast path. Called
// once from main.cpp after cl_ctx.initialize().
void nnopt_register_cl_ctx_for_gemv(OpenCLContext* ctx);
