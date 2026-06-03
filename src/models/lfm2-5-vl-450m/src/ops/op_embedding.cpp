// Reference: model_info/transformers_src/modeling_lfm2_vl.py:226-294 Lfm2VlModel.forward delegates inputs_embeds to language_model
// Reference: transformers/models/lfm2/modeling_lfm2.py Lfm2Model.forward
//   class Lfm2Model(...):
//       self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size, self.padding_idx)
//       ...
//       def forward(..., input_ids=None, inputs_embeds=None, ...):
//           if (input_ids is None) ^ (inputs_embeds is not None):
//               raise ValueError("You must specify exactly one of input_ids or inputs_embeds")
//           if inputs_embeds is None:
//               inputs_embeds = self.embed_tokens(input_ids)
//           hidden_states = inputs_embeds
// C++ translation: gather each token-id row from model.language_model.embed_tokens.weight.

#include "opencl_context.h"
#include "weights.h"
#include "debug_utils.h"
#include "utils.h"
#include "ops/lfm2_common.h"

#include <CL/cl.h>
#include <cstddef>
#include <vector>

namespace {
static bool set_arg_local(cl_kernel k, cl_uint idx, size_t sz, const void* v, const char* name) {
    cl_int err = clSetKernelArg(k, idx, sz, v);
    if (err != CL_SUCCESS) {
        NNOPT_ERROR_FMT("op_Embedding: clSetKernelArg(%u,%s) failed (%d)", (unsigned)idx, name, (int)err);
        return false;
    }
    return true;
}

static bool validate_embedding_shape(Weights& weights, const char* key, int hidden_size) {
    if (!key || key[0] == '\0') {
        NNOPT_ERROR("op_Embedding: empty weight key");
        return false;
    }
    const std::vector<int> shape = weights.get_shape(key);
    if (shape.size() != 2) {
        NNOPT_ERROR_FMT("op_Embedding: %s shape rank=%zu, expected [vocab,hidden]", key, shape.size());
        return false;
    }
    if (shape[1] != hidden_size) {
        NNOPT_ERROR_FMT("op_Embedding: %s hidden dim=%d, expected %d", key, shape[1], hidden_size);
        return false;
    }
    if (shape[0] <= 0) {
        NNOPT_ERROR_FMT("op_Embedding: %s invalid vocab dim=%d", key, shape[0]);
        return false;
    }
    return true;
}
}  // namespace

extern "C" cl_mem op_Embedding(OpenCLContext& cl_ctx,
                                Weights& weights,
                                cl_command_queue queue,
                                cl_mem input_ids_i32,
                                int num_tokens,
                                int hidden_size,
                                const char* weight_key_wte) {
    NNOPT_CHECKPOINT("op_Embedding entry");
    if (!queue || !input_ids_i32 || num_tokens <= 0 || hidden_size <= 0) {
        NNOPT_ERROR_FMT("op_Embedding: invalid args ids=%p tokens=%d hidden=%d", (void*)input_ids_i32, num_tokens, hidden_size);
        return nullptr;
    }
    if (!validate_embedding_shape(weights, weight_key_wte, hidden_size)) return nullptr;

    cl_mem table = weights.get_buffer(weight_key_wte);
    if (!table) {
        NNOPT_ERROR_FMT("op_Embedding: missing table %s", weight_key_wte ? weight_key_wte : "<null>");
        return nullptr;
    }

    const size_t out_elems = (size_t)num_tokens * (size_t)hidden_size;
    cl_mem out = lfm2_alloc(cl_ctx, out_elems, "embedding_out");
    if (!out) return nullptr;

    cl_program p = lfm2_program(cl_ctx);
    if (!p) { clReleaseMemObject(out); return nullptr; }

    // int8 path: when the embed table is per-row int8-quantized, dispatch the
    // int8 gather kernel which reads char + per-row fp16 scale. Mirrors the
    // fp16 gather kernel's layout/threading exactly.
    const bool table_is_int8 = weights.is_int8(weight_key_wte);
    cl_mem scales = table_is_int8 ? weights.get_scale_buffer(weight_key_wte) : nullptr;

    cl_int err = CL_SUCCESS;
    cl_kernel k = nullptr;
    bool ok = false;
    if (table_is_int8 && scales) {
        k = clCreateKernel(p, "gather_embedding_int8", &err);
        if (err != CL_SUCCESS || !k) {
            NNOPT_ERROR_FMT("op_Embedding: clCreateKernel(gather_embedding_int8) failed (%d)", (int)err);
            clReleaseMemObject(out);
            return nullptr;
        }
        ok = set_arg_local(k, 0, sizeof(cl_mem), &input_ids_i32, "ids") &&
             set_arg_local(k, 1, sizeof(cl_mem), &table, "table_int8") &&
             set_arg_local(k, 2, sizeof(cl_mem), &scales, "scales") &&
             set_arg_local(k, 3, sizeof(cl_mem), &out, "out") &&
             set_arg_local(k, 4, sizeof(int), &num_tokens, "rows") &&
             set_arg_local(k, 5, sizeof(int), &hidden_size, "hidden") &&
             lfm2_kernel1(queue, k, out_elems, "op_embedding_int8");
    } else {
        k = clCreateKernel(p, "gather_embedding", &err);
        if (err != CL_SUCCESS || !k) {
            NNOPT_ERROR_FMT("op_Embedding: clCreateKernel(gather_embedding) failed (%d)", (int)err);
            clReleaseMemObject(out);
            return nullptr;
        }
        ok = set_arg_local(k, 0, sizeof(cl_mem), &input_ids_i32, "ids") &&
             set_arg_local(k, 1, sizeof(cl_mem), &table, "table") &&
             set_arg_local(k, 2, sizeof(cl_mem), &out, "out") &&
             set_arg_local(k, 3, sizeof(int), &num_tokens, "rows") &&
             set_arg_local(k, 4, sizeof(int), &hidden_size, "hidden") &&
             lfm2_kernel1(queue, k, out_elems, "op_embedding");
    }
    clReleaseKernel(k);
    if (!ok) {
        clReleaseMemObject(out);
        return nullptr;
    }
    NNOPT_LAYER_CHECK("op_embedding_out", queue, out, out_elems);
    return out;
}
