// Reference: model_info/transformers_src/modeling_idefics3.py Idefics3Connector.forward
// Implements the connector projection:
//   return self.modality_projection(x)
// where modality_projection.proj is a Linear(input_size=vision_hidden*(scale_factor^2), output_size=text_hidden, bias=False)

#include "opencl_context.h"
#include "weights.h"
#include "debug_utils.h"
#include "forward_dispatch.h"

#include <CL/cl.h>
#include <string>

extern "C" cl_mem op_Idefics3Connector(OpenCLContext& cl_ctx,
                                       Weights& weights,
                                       cl_command_queue queue,
                                       cl_mem vision_hidden_states,
                                       int rows,
                                       int in_features,
                                       int out_features,
                                       const char* proj_w) {
  if (!queue || !vision_hidden_states || !proj_w) {
    NNOPT_ERROR("op_Idefics3Connector: null arg");
    return nullptr;
  }

  // This is a bias-free linear.
  return op_Linear(cl_ctx,
                   weights,
                   queue,
                   vision_hidden_states,
                   rows,
                   in_features,
                   out_features,
                   proj_w,
                   /*bias*/ "");
}
