// Reference: model_info/transformers_src/modeling_mamba2.py (Mamba2ForCausalLM.forward lm_head projection)

#include "layers/lm_head.h"

#include "debug_utils.h"
#include "opencl_context.h"
#include "utils.h"   // pytorch_linear
#include "weights.h"

LmHead::LmHead(OpenCLContext& cl_ctx, Weights& weights,
               const std::string& weight_key,
               int hidden_size, int vocab_size)
    : cl_ctx_(cl_ctx), weights_(weights), weight_key_(weight_key),
      hidden_size_(hidden_size), vocab_size_(vocab_size) {}

LmHead::~LmHead() {
    if (buf_logits_) clReleaseMemObject(buf_logits_);
}

bool LmHead::ensure_logits_(int rows, int padded_vocab) {
    if (rows <= buf_capacity_rows_ && buf_logits_ != nullptr) return true;
    if (buf_logits_) { clReleaseMemObject(buf_logits_); buf_logits_ = nullptr; buf_capacity_rows_ = 0; }
    cl_int err = CL_SUCCESS;
    const size_t bytes = (size_t)rows * (size_t)padded_vocab * sizeof(nnopt_storage_t);
    buf_logits_ = clCreateBuffer(cl_ctx_.context(), CL_MEM_READ_WRITE, bytes, nullptr, &err);
    if (err != CL_SUCCESS || !buf_logits_) {
        NNOPT_ERROR_FMT("LmHead::ensure_logits_ failed (%d)", (int)err);
        return false;
    }
    buf_capacity_rows_ = rows;
    return true;
}

bool LmHead::initialize() {
    w_ = weights_.get_buffer(weight_key_);
    if (!w_) {
        NNOPT_ERROR_FMT("LmHead missing weight: %s", weight_key_.c_str());
        return false;
    }
    return true;
}

cl_mem LmHead::forward(cl_command_queue queue, cl_mem hidden, int seq_len) {
    NNOPT_CHECKPOINT("LmHead::forward");

    if (!w_) {
        NNOPT_ERROR("LmHead::forward called before initialize() or weight missing");
        return nullptr;
    }

    // Debug visibility: ensure hidden activations are not already all-zeros.
    NNOPT_LAYER_CHECK_INPUT("lm_head", queue, hidden, (size_t)seq_len * (size_t)hidden_size_);

    const int padded_vocab = (int)weights_.get_shape(weight_key_)[0];
    if (!ensure_logits_(seq_len, padded_vocab)) return nullptr;
    cl_mem logits = buf_logits_;

    // [M, hidden] x [vocab, hidden]^T -> [M, vocab]
    if (!pytorch_linear(queue,
                        /*M=*/seq_len,
                        /*N=*/padded_vocab,
                        /*K=*/hidden_size_,
                        hidden,
                        w_,
                        logits)) {
        NNOPT_ERROR("LmHead pytorch_linear failed");
        return nullptr;
    }

    // Dump the full padded logits to match the reference dump-spec expectations.
    NNOPT_LAYER_CHECK("lm_head", queue, logits, (size_t)seq_len * (size_t)padded_vocab);

    return logits;
}
