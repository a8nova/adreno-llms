// Reference: /Users/alazarshenkute/.nnopt/ref_venvs/env_stable/lib/python3.12/site-packages/transformers/models/idefics3/modeling_idefics3.py:374-386 Idefics3VisionMLP.forward
// Implements:
//   hidden_states = fc1(hidden_states)
//   hidden_states = activation_fn(hidden_states)  # GELU tanh approx
//   hidden_states = fc2(hidden_states)
// using existing primitives: op_Linear + op_PytorchGELUTanh.

#include "opencl_context.h"
#include "weights.h"
#include "debug_utils.h"
#include "forward_dispatch.h"

#include <CL/cl.h>
#include <cstddef>
#include <string>

extern "C" cl_mem op_Idefics3VisionMLP(OpenCLContext& cl_ctx,
                                       Weights& weights,
                                       cl_command_queue queue,
                                       cl_mem hidden_states,
                                       int rows,
                                       int hidden_size,
                                       int intermediate_size,
                                       const char* fc1_w,
                                       const char* fc1_b,
                                       const char* fc2_w,
                                       const char* fc2_b) {
  if (!queue || !hidden_states) {
    NNOPT_ERROR("op_Idefics3VisionMLP: null arg");
    return nullptr;
  }

  // fc1
  cl_mem x1 = op_Linear(cl_ctx, weights, queue,
                        hidden_states,
                        /*rows=*/rows,
                        /*in_features=*/hidden_size,
                        /*out_features=*/intermediate_size,
                        fc1_w,
                        fc1_b);
  if (!x1) {
    NNOPT_ERROR("op_Idefics3VisionMLP: fc1 failed");
    return nullptr;
  }

  // activation
  const int act_elems = rows * intermediate_size;
  cl_mem x2 = op_PytorchGELUTanh(cl_ctx, weights, queue, x1, act_elems);
  clReleaseMemObject(x1);
  if (!x2) {
    NNOPT_ERROR("op_Idefics3VisionMLP: activation failed");
    return nullptr;
  }

  // fc2
  cl_mem out = op_Linear(cl_ctx, weights, queue,
                         x2,
                         /*rows=*/rows,
                         /*in_features=*/intermediate_size,
                         /*out_features=*/hidden_size,
                         fc2_w,
                         fc2_b);
  clReleaseMemObject(x2);
  if (!out) {
    NNOPT_ERROR("op_Idefics3VisionMLP: fc2 failed");
    return nullptr;
  }

  return out;
}
