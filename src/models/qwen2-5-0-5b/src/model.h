// Reference: model_info/transformers_src/modeling_qwen2.py:1-260 Qwen2Model/Qwen2ForCausalLM forward
#pragma once

#include "opencl_context.h"
#include "sampler.h"
#include "weights.h"

#include "model_config.h"

#include "layers/attention.h"
#include "layers/embedding.h"
#include "layers/layer_norm.h"
#include "layers/mlp.h"

#include <CL/cl.h>
#include <cstdint>
#include <functional>
#include <vector>

// Streaming callback: invoked once per sampled token (BOTH first and
// subsequent decode steps). `new_token` is the token just produced;
// `all_ids` is the full ids buffer (prompt_ids + every token so far,
// including new_token). Caller can incrementally decode + print.
// Pass nullptr for non-streaming behaviour.
using TokenCallback = std::function<void(int32_t /*new_token*/,
                                         const std::vector<int32_t>& /*all_ids*/)>;

class Model {
public:
    Model(OpenCLContext& cl_ctx, Weights& weights);
    ~Model();

    std::vector<float> forward(const std::vector<int32_t>& input_ids);
    std::vector<float> forward(const std::vector<int32_t>& input_ids, int start_pos);

    std::vector<float> forward_decode(int32_t token_id, int start_pos);

    // Greedy fast path: runs the full forward but resolves the argmax on
    // the GPU and reads back just the int32 token id. Saves the 304 KB
    // logits readback + host max_element scan on every decode step.
    // Returns -1 on failure. Only valid when sampling is greedy
    // (temperature ≤ 0 AND repetition_penalty == 1.0).
    int32_t forward_greedy(const std::vector<int32_t>& input_ids, int start_pos);

    // Chained-decode enqueue (Step 17): same as forward_greedy(seq=1) but
    // (a) the embedding reads its token id directly from argmax_out_buf_ —
    // no host roundtrip — and (b) NO synchronous readback at the end. The
    // new token id is written to argmax_out_buf_ by argmax_finalize and
    // is consumed by the NEXT call's embedding. Caller is responsible for
    // (1) seeding argmax_out_buf_ with the first decode token before the
    // first call, and (2) reading argmax_out_buf_ asynchronously to
    // retrieve produced tokens. Returns false on enqueue error.
    bool forward_chained_enqueue(int start_pos);

    // Accessor for the device-side argmax output buffer. Used by the
    // chained-decode loop in main.cpp to async-readback produced tokens
    // and to seed the buffer with the first token.
    cl_mem argmax_result_buffer() const { return argmax_out_buf_; }

    std::vector<int32_t> generate(
        const std::vector<int32_t>& prompt_ids,
        int max_new_tokens = 64,
        SamplerConfig sampler_config = SamplerConfig{},
        TokenCallback on_token = nullptr
    );

private:
    OpenCLContext& cl_ctx_;
    Weights& weights_;

    bool ok_ = false;

    cl_program utils_program_ = nullptr;

    Embedding* embedding_ = nullptr;
    LayerNorm* pre_attn_norm_[MODEL_CONFIG::NUM_HIDDEN_LAYERS] = {nullptr};
    Attention* attn_[MODEL_CONFIG::NUM_HIDDEN_LAYERS] = {nullptr};
    LayerNorm* post_attn_norm_[MODEL_CONFIG::NUM_HIDDEN_LAYERS] = {nullptr};
    Mlp* mlp_[MODEL_CONFIG::NUM_HIDDEN_LAYERS] = {nullptr};
    LayerNorm* final_norm_ = nullptr;

    cl_mem w_lm_head_ = nullptr;  // tied to model.embed_tokens.weight

    // Greedy GPU-side argmax fast path — built lazily on first forward_greedy call.
    // 2-pass reduce: argmax_partial fills NUM_PARTIALS slots, argmax_finalize
    // reduces them to one int32. NUM_PARTIALS = 32.
    cl_program argmax_program_  = nullptr;
    cl_kernel  argmax_partial_  = nullptr;
    cl_kernel  argmax_finalize_ = nullptr;
    cl_mem     argmax_out_buf_  = nullptr;  // 1×int32
    cl_mem     argmax_pv_buf_   = nullptr;  // NUM_PARTIALS×float (partial vals)
    cl_mem     argmax_pi_buf_   = nullptr;  // NUM_PARTIALS×int   (partial idxs)

    // Persistent lm_head logits buffer — single decode step writes
    // 1 × 151936 × 2 = ~304 KB. Saves one clCreateBuffer per forward
    // (33 forwards per 32-token decode = 33 alloc/free saved).
    cl_mem     buf_logits_      = nullptr;
    int        buf_logits_rows_ = 0;

    // Persistent decode hidden state buffer [1, HIDDEN_SIZE]. Lazy-alloc
    // on first decode forward. Allows the embedding output to stay in a
    // fixed cl_mem handle across decode iterations — prerequisite for
    // cl_qcom_recordable_queues recording (CL kernel handles + arg
    // pointers must stay valid across replays).
    cl_mem     buf_decode_hidden_ = nullptr;

    // Counter buffer [0]=start_pos, [1]=seq_k. Written by host via
    // clEnqueueWriteBuffer before each decode step; kernels (rope, kv_write,
    // attn_scores, softmax, attn_out) read from it at execution time.
    cl_mem     buf_counter_  = nullptr;

    // ── cl_qcom_recordable_queues integration (decode replay path) ──
    // recordable_q_ is created with CL_QUEUE_RECORDABLE_QCOM (bit 30)
    // alone — no profiling/OOO bits. It's used ONLY to capture a
    // recording; commands enqueued on it are recorded, not executed.
    // rec_decode_ is the recording handle (void* opaque type). Replay
    // happens on cl_ctx_.queue() (the live in-order PROFILING queue).
    // fn_* are dlsym'd entry points (vendor symbols not surfaced by
    // the Android ICD via clGetExtensionFunctionAddressForPlatform).
    void*              recordable_q_ = nullptr;     // cl_command_queue
    void*              rec_decode_ = nullptr;       // cl_recording_qcom (opaque)
    void*              fn_new_     = nullptr;       // clNewRecordingQCOM
    void*              fn_end_     = nullptr;       // clEndRecordingQCOM
    void*              fn_release_ = nullptr;       // clReleaseRecordingQCOM
    void*              fn_enqueue_ = nullptr;       // clEnqueueRecordingQCOM
    int                decode_iter_count_ = 0;      // 0=cold, 1=record, 2+=replay
    bool               recording_built_ = false;    // rec_decode_ is valid + replayable
};
