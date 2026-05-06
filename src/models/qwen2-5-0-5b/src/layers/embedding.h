// Reference: model_info/transformers_src/modeling_qwen2.py:188-263 Qwen2Model.forward (embed_tokens)
#pragma once

#include "opencl_context.h"
#include "weights.h"
#include "model_config.h"

#include <CL/cl.h>
#include <string>
#include <vector>

class Embedding {
public:
    Embedding(OpenCLContext& cl_ctx, Weights& weights);
    ~Embedding();

    bool initialize();

    // input_ids: host token ids length=seq_len
    // start_pos: absolute position offset (0 for prefill; decode passes prompt_len + step)
    cl_mem forward(cl_command_queue queue, const std::vector<int32_t>& input_ids, int start_pos);

    // Low-level overload used by model.cpp to avoid an extra vector conversion.
    // UPLOAD-OK: input_ids
    cl_mem forward(cl_command_queue queue, const int* token_ids, int seq_len, int start_pos);

    // ── Recording-friendly decode path ──
    // set_decode_token writes a single int32 token id into the persistent
    // ids_buf_decode_ buffer via clEnqueueWriteBuffer. NOT recordable.
    // Call between decode iterations to update the input token.
    bool set_decode_token(cl_command_queue queue, int32_t token_id);

    // forward_into_decode dispatches the embedding kernel using the
    // persistent ids_buf_decode_ as input and the caller-provided dest as
    // output. The kernel is the only operation; safely recordable.
    // Caller must have called set_decode_token first.
    bool forward_into_decode(cl_command_queue queue, cl_mem dest, int start_pos);

    // Accessor for the embedding kernel handle (for recording arg overrides).
    cl_kernel kernel() const { return kernel_; }

private:
    OpenCLContext& cl_ctx_;
    Weights& weights_;

    cl_program program_ = nullptr;
    cl_kernel kernel_ = nullptr;

    cl_mem wte_ = nullptr;
    cl_mem wpe_zero_ = nullptr;

    // Persistent single-token input id buffer for decode replay path.
    // 1 × int32. Lazy-allocated on first set_decode_token call.
    cl_mem ids_buf_decode_ = nullptr;
};
