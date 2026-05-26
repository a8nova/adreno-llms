// Shared GPU conv1d wrapper exposing the CLBlast-backed implementation so
// flow_inverse and text_encoder can reuse the path that vocoder already
// uses. The actual definition lives in conv1d_gpu.cpp.
#pragma once

#include "opencl_context.h"
#include <CL/cl.h>

// out[C_out, L_out] = conv1d(in[C_in, L_in], w[C_out, C_in, K], b[C_out])
// stride=1, padding via `padding`, dilation via `dilation`.
// `bias` may be nullptr when has_bias=false.
// Caller owns the returned cl_mem — release with clReleaseMemObject.
cl_mem conv1d_gpu(OpenCLContext& cl_ctx,
                  cl_command_queue queue,
                  cl_mem in,
                  cl_mem w,
                  cl_mem bias,
                  int C_in, int C_out, int L_in,
                  int K, int stride, int padding, int dilation,
                  bool has_bias,
                  const char* label);

// Apply gated activation: acts[C, T] = tanh(x[0:C, T]) * sigmoid(x[C:2C, T]).
// Reads x [2C, T], writes a fresh buffer [C, T].
cl_mem gated_activation_gpu(OpenCLContext& cl_ctx,
                            cl_command_queue queue,
                            cl_mem x_2c, int C, int T,
                            const char* label);

// in-place elementwise add: a[i] += b[i] for i in 0..N.
bool elem_add_inplace_gpu(OpenCLContext& cl_ctx,
                          cl_command_queue queue,
                          cl_mem a, cl_mem b, int N,
                          const char* label);
