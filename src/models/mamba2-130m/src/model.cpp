// Auto-generated model implementation for state-spaces/mamba2-130m (SSM/Mamba architecture)
// Reference: model_info/transformers_src/modeling_mamba2.py:763-831  (Mamba2Model.forward)
// Reference: model_info/transformers_src/modeling_mamba2.py:874-932  (Mamba2ForCausalLM.forward)

#include "model.h"
#include "sampler.h"

#include "benchmark.h"
#include "debug_utils.h"
#include "kernel_profiler.h"
#include "model_config.h"
#include "utils.h"

#include <cstring>
#include <iostream>

Model::Model(OpenCLContext& cl_ctx, Weights& weights)
    : cl_ctx_(cl_ctx), weights_(weights) {
    NNOPT_CHECKPOINT("Model constructor");
}

Model::~Model() {
    if (utils_program_) clReleaseProgram(utils_program_);
    if (argmax_kernel_) clReleaseKernel(argmax_kernel_);
    if (argmax_partial_kernel_) clReleaseKernel(argmax_partial_kernel_);
    if (argmax_final_kernel_) clReleaseKernel(argmax_final_kernel_);
    if (argmax_program_) clReleaseProgram(argmax_program_);
    if (argmax_idx_buf_) clReleaseMemObject(argmax_idx_buf_);
    if (argmax_scratch_v_) clReleaseMemObject(argmax_scratch_v_);
    if (argmax_scratch_i_) clReleaseMemObject(argmax_scratch_i_);
    if (ids_history_buf_) clReleaseMemObject(ids_history_buf_);
    if (cur_token_buf_) clReleaseMemObject(cur_token_buf_);
}

bool Model::ensure_ids_history_(int N) {
    if (N <= ids_history_capacity_ && ids_history_buf_ && cur_token_buf_) return true;
    if (ids_history_buf_) { clReleaseMemObject(ids_history_buf_); ids_history_buf_ = nullptr; }
    cl_int err = CL_SUCCESS;
    ids_history_buf_ = clCreateBuffer(cl_ctx_.context(), CL_MEM_READ_WRITE,
                                      (size_t)N * sizeof(int32_t), nullptr, &err);
    if (err != CL_SUCCESS || !ids_history_buf_) {
        NNOPT_ERROR_FMT("Model: ensure_ids_history_(%d) failed (%d)", N, (int)err);
        return false;
    }
    if (!cur_token_buf_) {
        cur_token_buf_ = clCreateBuffer(cl_ctx_.context(), CL_MEM_READ_WRITE,
                                        sizeof(int32_t), nullptr, &err);
        if (err != CL_SUCCESS || !cur_token_buf_) {
            NNOPT_ERROR_FMT("Model: cur_token_buf alloc failed (%d)", (int)err);
            return false;
        }
    }
    ids_history_capacity_ = N;
    return true;
}

bool Model::initialize() {
    backbone_ = std::make_unique<Backbone>(cl_ctx_, weights_);
    if (!backbone_->initialize()) {
        NNOPT_ERROR("backbone_->initialize() FAILED");
        return false;
    }
    NNOPT_LAYER_INIT("backbone");

    utils_program_ = cl_ctx_.build_program_from_file("kernels/utils.cl");
    if (!utils_program_) {
        NNOPT_ERROR("failed to build kernels/utils.cl");
        return false;
    }

    for (int i = 0; i < MODEL_CONFIG::NUM_LAYERS; i++) {
        const std::string norm_key = "backbone.layers." + std::to_string(i) + ".norm.weight";
        norm_[i] = std::make_unique<LayerNorm>(
            cl_ctx_, weights_, norm_key, MODEL_CONFIG::HIDDEN_SIZE, /*eps=*/1e-5f, i);
        if (!norm_[i]->initialize()) {
            NNOPT_ERROR_FMT("norm_[%d].initialize() FAILED", i);
            return false;
        }
        NNOPT_LAYER_INIT_FMT("norm_%d", i);

        const std::string prefix = "backbone.layers." + std::to_string(i) + ".mixer";
        mixer_[i] = std::make_unique<Ssm>(cl_ctx_, weights_, prefix, i);
        mixer_[i]->set_utils_program(utils_program_);
        if (!mixer_[i]->initialize()) {
            NNOPT_ERROR_FMT("mixer_[%d].initialize() FAILED", i);
            return false;
        }
        NNOPT_LAYER_INIT_FMT("mixer_%d", i);
    }

    lm_head_ = std::make_unique<LmHead>(
        cl_ctx_, weights_, "lm_head.weight", MODEL_CONFIG::HIDDEN_SIZE, MODEL_CONFIG::VOCAB_SIZE);
    if (!lm_head_->initialize()) {
        NNOPT_ERROR("lm_head_->initialize() FAILED");
        return false;
    }
    NNOPT_LAYER_INIT("lm_head");

    // Lever 1: build GPU argmax kernel + persistent 4-byte index buffer.
    argmax_program_ = cl_ctx_.build_program_from_file("kernels/argmax.cl");
    if (!argmax_program_) {
        NNOPT_ERROR("Model: failed to build kernels/argmax.cl");
        return false;
    }
    cl_int err = CL_SUCCESS;
    argmax_kernel_ = clCreateKernel(argmax_program_, "argmax_row", &err);
    if (err != CL_SUCCESS || !argmax_kernel_) {
        NNOPT_ERROR_FMT("Model: clCreateKernel(argmax_row) failed (%d)", (int)err);
        return false;
    }
    argmax_idx_buf_ = clCreateBuffer(cl_ctx_.context(), CL_MEM_READ_WRITE, sizeof(int32_t),
                                     nullptr, &err);
    if (err != CL_SUCCESS || !argmax_idx_buf_) {
        NNOPT_ERROR_FMT("Model: clCreateBuffer(argmax_idx) failed (%d)", (int)err);
        return false;
    }
    // Lever H: 2-pass argmax kernels + scratch.
    argmax_partial_kernel_ = clCreateKernel(argmax_program_, "argmax_partial", &err);
    if (err != CL_SUCCESS || !argmax_partial_kernel_) {
        NNOPT_ERROR_FMT("Model: clCreateKernel(argmax_partial) failed (%d) — Lever H disabled",
                        (int)err);
        argmax_partial_kernel_ = nullptr;
        err = CL_SUCCESS;
    }
    argmax_final_kernel_ = clCreateKernel(argmax_program_, "argmax_final", &err);
    if (err != CL_SUCCESS || !argmax_final_kernel_) {
        NNOPT_ERROR_FMT("Model: clCreateKernel(argmax_final) failed (%d) — Lever H disabled",
                        (int)err);
        argmax_final_kernel_ = nullptr;
        err = CL_SUCCESS;
    }
    argmax_scratch_v_ = clCreateBuffer(cl_ctx_.context(), CL_MEM_READ_WRITE,
                                       (size_t)kArgmaxNumWG * sizeof(float), nullptr, &err);
    argmax_scratch_i_ = clCreateBuffer(cl_ctx_.context(), CL_MEM_READ_WRITE,
                                       (size_t)kArgmaxNumWG * sizeof(int32_t), nullptr, &err);
    if (err != CL_SUCCESS || !argmax_scratch_v_ || !argmax_scratch_i_) {
        NNOPT_ERROR_FMT("Model: argmax_scratch alloc failed (%d) — Lever H disabled", (int)err);
        if (argmax_scratch_v_) { clReleaseMemObject(argmax_scratch_v_); argmax_scratch_v_ = nullptr; }
        if (argmax_scratch_i_) { clReleaseMemObject(argmax_scratch_i_); argmax_scratch_i_ = nullptr; }
        err = CL_SUCCESS;
    }

    NNOPT_CHECKPOINT("Model::initialize() complete");
    return true;
}

void Model::reset_state() {
    cl_command_queue queue = cl_ctx_.queue();
    for (int i = 0; i < MODEL_CONFIG::NUM_LAYERS; i++) {
        if (mixer_[i]) mixer_[i]->reset_state(queue);
    }
}

std::vector<float> Model::forward(const std::vector<int32_t>& input_ids) {
    // Legacy wrapper (start_pos=0). Kept for scaffold compatibility.
    return forward(input_ids, /*start_pos=*/0);
}

cl_mem Model::compute_logits_(const std::vector<int32_t>& input_ids, int start_pos) {
    cl_command_queue queue = cl_ctx_.queue();
    const int seq_len = (int)input_ids.size();
    if (seq_len <= 0) return nullptr;

    if (!backbone_ || !lm_head_) {
        NNOPT_ERROR("Model::compute_logits_ called before initialize()");
        return nullptr;
    }

    // Lever 4: backbone->embed returns a BORROWED handle to its persistent
    // buf_embed_out_; do NOT release. We need to keep this buffer's *contents*
    // alive across the inner block stack (the residual chain reads/writes it),
    // so we copy into a working buffer or just operate on it in place.
    //
    // Trick: use the borrowed embed_out as `hidden` directly. The block loop
    // mutates hidden in place (via element_add_inplace and Lever 2's radd).
    // At the end of the loop, hidden still points at buf_embed_out_, which is
    // safe to pass to norm_f (different persistent buffer).
    cl_mem hidden = backbone_->embed(queue, input_ids.data(), seq_len);
    if (!hidden) {
        NNOPT_ERROR("backbone_->embed returned nullptr");
        return nullptr;
    }
    NNOPT_LAYER_CHECK("backbone_embedding", queue, hidden,
                     (size_t)seq_len * (size_t)MODEL_CONFIG::HIDDEN_SIZE);

    for (int i = 0; i < MODEL_CONFIG::NUM_LAYERS; i++) {
        cl_mem normed = norm_[i]->forward(queue, hidden, seq_len);
        if (!normed) {
            NNOPT_ERROR_FMT("norm_[%d]->forward returned nullptr", i);
            return nullptr;
        }

        cl_mem mix_in = (seq_len == 1) ? hidden : nullptr;
        cl_mem mix = mixer_[i]->forward(queue, normed, seq_len, start_pos, mix_in);
        // Lever 4: LayerNorm now returns a borrowed handle to its persistent
        // buf_out_; do NOT release.
        if (!mix) {
            NNOPT_ERROR_FMT("mixer_[%d]->forward returned nullptr", i);
            return nullptr;
        }

        if (mix != hidden) {
            if (!element_add_inplace(queue, utils_program_, hidden, mix,
                                     (size_t)seq_len * (size_t)MODEL_CONFIG::HIDDEN_SIZE)) {
                NNOPT_ERROR_FMT("element_add_inplace failed at layer %d", i);
                return nullptr;
            }
        }

        NNOPT_LAYER_CHECK_FMT("layer_%d", i, queue, hidden,
                              (size_t)seq_len * (size_t)MODEL_CONFIG::HIDDEN_SIZE);
    }

    // backbone->norm_f reads `hidden` (borrowed buf_embed_out_) and writes
    // into its own persistent buf_norm_f_out_. No release of either.
    cl_mem final_hidden = backbone_->norm_f(queue, hidden, seq_len);
    if (!final_hidden) {
        NNOPT_ERROR("backbone_->norm_f returned nullptr");
        return nullptr;
    }

    cl_mem logits = lm_head_->forward(queue, final_hidden, seq_len);
    if (!logits) {
        NNOPT_ERROR("lm_head_->forward returned nullptr");
        return nullptr;
    }
    return logits;
}

cl_mem Model::compute_logits_gpu_input_(cl_mem ids_buf, int start_pos) {
    cl_command_queue queue = cl_ctx_.queue();
    if (!backbone_ || !lm_head_ || !ids_buf) {
        NNOPT_ERROR("Model::compute_logits_gpu_input_ invalid state");
        return nullptr;
    }
    const int seq_len = 1;  // decode-only path

    cl_mem hidden = backbone_->embed_gpu(queue, ids_buf, seq_len);
    if (!hidden) return nullptr;

    for (int i = 0; i < MODEL_CONFIG::NUM_LAYERS; i++) {
        cl_mem normed = norm_[i]->forward(queue, hidden, seq_len);
        if (!normed) return nullptr;

        cl_mem mix = mixer_[i]->forward(queue, normed, seq_len, start_pos, hidden);
        if (!mix) return nullptr;
        if (mix != hidden) {
            // Lever 2 radd unavailable — fall back to plain add.
            if (!element_add_inplace(queue, utils_program_, hidden, mix,
                                     (size_t)seq_len * (size_t)MODEL_CONFIG::HIDDEN_SIZE)) return nullptr;
        }
    }

    cl_mem final_hidden = backbone_->norm_f(queue, hidden, seq_len);
    if (!final_hidden) return nullptr;

    cl_mem logits = lm_head_->forward(queue, final_hidden, seq_len);
    return logits;
}

std::vector<float> Model::forward(const std::vector<int32_t>& input_ids, int start_pos) {
    NNOPT_CHECKPOINT("forward() started");

    cl_command_queue queue = cl_ctx_.queue();
    const int seq_len = (int)input_ids.size();
    cl_mem logits = compute_logits_(input_ids, start_pos);
    if (!logits) return {};

    // Gather last-row logits to host.
    const int padded_vocab = (int)weights_.get_shape("lm_head.weight")[0];
    if (padded_vocab <= 0 || padded_vocab < MODEL_CONFIG::VOCAB_SIZE) {
        NNOPT_ERROR_FMT("Invalid padded_vocab=%d (VOCAB_SIZE=%d)", padded_vocab, MODEL_CONFIG::VOCAB_SIZE);
        /*Lever4: borrowed*/
        return {};
    }

    std::vector<float> host_logits((size_t)MODEL_CONFIG::VOCAB_SIZE);
    const size_t row_bytes_padded = (size_t)padded_vocab * sizeof(nnopt_storage_t);
    const size_t offset_bytes = (size_t)(seq_len - 1) * row_bytes_padded;

#ifdef NNOPT_USE_FP16
    std::vector<nnopt_storage_t> tmp((size_t)padded_vocab);
    cl_int err = clEnqueueReadBuffer(queue, logits, CL_TRUE, offset_bytes,
                                     row_bytes_padded, tmp.data(), 0, nullptr, nullptr);
    if (err != CL_SUCCESS) {
        NNOPT_ERROR_FMT("Read logits failed (%d)", (int)err);
        /*Lever4: borrowed*/
        return {};
    }
    const uint16_t* tmp_u16 = reinterpret_cast<const uint16_t*>(tmp.data());
    for (int i = 0; i < MODEL_CONFIG::VOCAB_SIZE; i++) {
        host_logits[(size_t)i] = nnopt_f16_to_f32(tmp_u16[(size_t)i]);
    }
#else
    std::vector<float> tmp((size_t)padded_vocab);
    cl_int err = clEnqueueReadBuffer(queue, logits, CL_TRUE, offset_bytes,
                                     row_bytes_padded, tmp.data(), 0, nullptr, nullptr);
    if (err != CL_SUCCESS) {
        NNOPT_ERROR_FMT("Read logits failed (%d)", (int)err);
        /*Lever4: borrowed*/
        return {};
    }
    std::memcpy(host_logits.data(), tmp.data(), (size_t)MODEL_CONFIG::VOCAB_SIZE * sizeof(float));
#endif

    /*Lever4: borrowed*/

    NNOPT_CHECKPOINT("forward() complete");
    return host_logits;
}

int32_t Model::forward_argmax_greedy(const std::vector<int32_t>& input_ids, int start_pos) {
    NNOPT_CHECKPOINT("forward_argmax_greedy() started");

    cl_command_queue queue = cl_ctx_.queue();
    const int seq_len = (int)input_ids.size();
    cl_mem logits = compute_logits_(input_ids, start_pos);
    if (!logits) return -1;

    const int padded_vocab = (int)weights_.get_shape("lm_head.weight")[0];
    if (padded_vocab <= 0 || padded_vocab < MODEL_CONFIG::VOCAB_SIZE) {
        NNOPT_ERROR_FMT("Invalid padded_vocab=%d (VOCAB_SIZE=%d)", padded_vocab, MODEL_CONFIG::VOCAB_SIZE);
        /*Lever4: borrowed*/
        return -1;
    }

    // Dispatch GPU argmax against the LAST row of logits.
    // The kernel reads logits[(seq_len-1)*padded_vocab + j] for j in [0, valid_n).
    // We pass a sub-buffer-style offset by re-binding the kernel arg to a
    // computed offset via clCreateSubBuffer? Easier: allocate logits with
    // last-row offset baked in. Cleanest: pass full buffer and a row stride
    // via an extra arg. But our kernel only takes a single offset-zero buffer.
    // → Use clCreateSubBuffer for the last row.
    // Pass row offset as a kernel arg to skip clCreateSubBuffer per call.
    cl_int err = CL_SUCCESS;
    int n = padded_vocab;
    int valid_n = MODEL_CONFIG::VOCAB_SIZE;
    int row_off = (seq_len - 1) * padded_vocab;

    err  = clSetKernelArg(argmax_kernel_, 0, sizeof(cl_mem), &logits);
    err |= clSetKernelArg(argmax_kernel_, 1, sizeof(cl_mem), &argmax_idx_buf_);
    err |= clSetKernelArg(argmax_kernel_, 2, sizeof(int),    &n);
    err |= clSetKernelArg(argmax_kernel_, 3, sizeof(int),    &valid_n);
    err |= clSetKernelArg(argmax_kernel_, 4, sizeof(int),    &row_off);
    if (err != CL_SUCCESS) {
        NNOPT_ERROR_FMT("forward_argmax_greedy: setKernelArg failed (%d)", (int)err);
        return -1;
    }

    const size_t WG = 64;
    size_t gws = WG;
    size_t lws = WG;
    cl_event* evt = KernelProfiler::event_for("argmax_row");
    err = clEnqueueNDRangeKernel(queue, argmax_kernel_, 1, nullptr, &gws, &lws, 0, nullptr, evt);
    if (err != CL_SUCCESS) {
        NNOPT_ERROR_FMT("forward_argmax_greedy: enqueue argmax_row failed (%d)", (int)err);
        return -1;
    }

    int32_t out_idx = -1;
    err = clEnqueueReadBuffer(queue, argmax_idx_buf_, CL_TRUE, 0, sizeof(int32_t),
                              &out_idx, 0, nullptr, nullptr);
    if (err != CL_SUCCESS) {
        NNOPT_ERROR_FMT("forward_argmax_greedy: readback argmax failed (%d)", (int)err);
        return -1;
    }

    /*Lever4: borrowed*/
    NNOPT_CHECKPOINT("forward_argmax_greedy() complete");
    return out_idx;
}

std::vector<int32_t> Model::generate(const std::vector<int32_t>& prompt_ids,
                                    int max_new_tokens,
                                    SamplerConfig sampler_config,
                                    Model::TokenCallback on_token) {
    NNOPT_CHECKPOINT("generate() started");

    reset_state();
    Sampler sampler(sampler_config);
    cl_command_queue queue = cl_ctx_.queue();

    const bool greedy_fast =
        sampler_config.temperature <= 0.0f && sampler_config.repetition_penalty == 1.0f;

    // Prefill via host readback (one-time).
    std::vector<int32_t> ids = prompt_ids;
    std::vector<float> logits = forward(prompt_ids, /*start_pos=*/0);

    if (!greedy_fast) {
        // Standard sampling path (rep penalty / top-k / top-p / temperature).
        for (int i = 0; i < max_new_tokens; i++) {
            std::vector<int32_t> generated(ids.begin() + prompt_ids.size(), ids.end());
            int next_token = sampler.sample(logits, generated);
            ids.push_back(next_token);
            NNOPT_BENCH_FIRST_TOKEN();
            if (on_token) on_token(next_token);
            if (sampler_config.eos_token_id >= 0 && next_token == sampler_config.eos_token_id) break;
            const int start_pos = (int)prompt_ids.size() + i;
            logits = forward({next_token}, start_pos);
            if (logits.empty()) { NNOPT_ERROR("forward returned empty logits"); break; }
        }
        NNOPT_CHECKPOINT("generate() complete");
        return ids;
    }

    // Lever D: async greedy decode pipeline.
    // - First token comes from prefill logits (host argmax). Push immediately.
    // - For tokens 1..N-1: enqueue forward + argmax with GPU-side ids; argmax
    //   writes to ids_history_buf_[i-1]; clEnqueueCopyBuffer pipes back into
    //   buf_ids_ for next iteration. No host readback inside the loop.
    // - At end of loop: clFinish, batch readback all 32 ids.
    if (!ensure_ids_history_(max_new_tokens)) {
        NNOPT_ERROR("ensure_ids_history_ failed");
        return ids;
    }

    // First token from prefill logits.
    std::vector<int32_t> generated_so_far(ids.begin() + prompt_ids.size(), ids.end());
    int first_token = sampler.sample(logits, generated_so_far);
    ids.push_back(first_token);
    NNOPT_BENCH_FIRST_TOKEN();
    if (on_token) on_token(first_token);
    logits.clear();

    if (sampler_config.eos_token_id >= 0 && first_token == sampler_config.eos_token_id) {
        return ids;
    }
    if (max_new_tokens <= 1) {
        return ids;
    }

    // Decode loop. Tokens are indexed 1..max_new_tokens-1 in this chain (token 0
    // already pushed above). argmax for iteration i writes ids_history_buf_[i-1].
    // Subsequent iterations copy ids_history_buf_[i-1] → cur_token_buf_[0] and
    // call embed_gpu(cur_token_buf_, 1) so the embedding reads the new token.
    int last_iter_completed = 0;  // index of last token written into history (1-based count of decoded tokens via async = last_iter_completed)
    const bool use_argmax_2pass = (argmax_partial_kernel_ != nullptr) && (argmax_final_kernel_ != nullptr) &&
                                  (argmax_scratch_v_ != nullptr) && (argmax_scratch_i_ != nullptr);
    auto enqueue_argmax = [&](cl_mem logits_buf, int write_idx) -> bool {
        const int padded_vocab = (int)weights_.get_shape("lm_head.weight")[0];
        int n = padded_vocab;
        int valid_n = MODEL_CONFIG::VOCAB_SIZE;
        int row_off = 0;

        if (use_argmax_2pass) {
            // Lever H pass 1: NUM_WG WGs scan interleaved tiles in parallel.
            cl_int err = CL_SUCCESS;
            err  = clSetKernelArg(argmax_partial_kernel_, 0, sizeof(cl_mem), &logits_buf);
            err |= clSetKernelArg(argmax_partial_kernel_, 1, sizeof(cl_mem), &argmax_scratch_v_);
            err |= clSetKernelArg(argmax_partial_kernel_, 2, sizeof(cl_mem), &argmax_scratch_i_);
            err |= clSetKernelArg(argmax_partial_kernel_, 3, sizeof(int),    &n);
            err |= clSetKernelArg(argmax_partial_kernel_, 4, sizeof(int),    &valid_n);
            err |= clSetKernelArg(argmax_partial_kernel_, 5, sizeof(int),    &row_off);
            if (err != CL_SUCCESS) { NNOPT_ERROR_FMT("Lever H partial setArg (%d)", err); return false; }
            const size_t WG = 64; size_t lws = WG; size_t gws = WG * (size_t)kArgmaxNumWG;
            cl_event* evt1 = KernelProfiler::event_for("argmax_partial");
            err = clEnqueueNDRangeKernel(queue, argmax_partial_kernel_, 1, nullptr, &gws, &lws, 0, nullptr, evt1);
            if (err != CL_SUCCESS) { NNOPT_ERROR_FMT("Lever H partial enqueue (%d)", err); return false; }

            // Pass 2: 1 WG of NUM_WG threads reduces and writes both buffers.
            err  = clSetKernelArg(argmax_final_kernel_, 0, sizeof(cl_mem), &argmax_scratch_v_);
            err |= clSetKernelArg(argmax_final_kernel_, 1, sizeof(cl_mem), &argmax_scratch_i_);
            err |= clSetKernelArg(argmax_final_kernel_, 2, sizeof(cl_mem), &ids_history_buf_);
            err |= clSetKernelArg(argmax_final_kernel_, 3, sizeof(cl_mem), &cur_token_buf_);
            err |= clSetKernelArg(argmax_final_kernel_, 4, sizeof(int),    &write_idx);
            if (err != CL_SUCCESS) { NNOPT_ERROR_FMT("Lever H final setArg (%d)", err); return false; }
            const size_t fws = (size_t)kArgmaxNumWG;
            const size_t flws = (size_t)kArgmaxNumWG;
            cl_event* evt2 = KernelProfiler::event_for("argmax_final");
            err = clEnqueueNDRangeKernel(queue, argmax_final_kernel_, 1, nullptr, &fws, &flws, 0, nullptr, evt2);
            if (err != CL_SUCCESS) { NNOPT_ERROR_FMT("Lever H final enqueue (%d)", err); return false; }
            return true;
        }

        cl_int err = CL_SUCCESS;
        err  = clSetKernelArg(argmax_kernel_, 0, sizeof(cl_mem), &logits_buf);
        err |= clSetKernelArg(argmax_kernel_, 1, sizeof(cl_mem), &ids_history_buf_);
        err |= clSetKernelArg(argmax_kernel_, 2, sizeof(int),    &n);
        err |= clSetKernelArg(argmax_kernel_, 3, sizeof(int),    &valid_n);
        err |= clSetKernelArg(argmax_kernel_, 4, sizeof(int),    &row_off);
        err |= clSetKernelArg(argmax_kernel_, 5, sizeof(int),    &write_idx);
        if (err != CL_SUCCESS) { NNOPT_ERROR_FMT("Lever D argmax setArg (%d)", err); return false; }
        const size_t WG = 64; size_t gws = WG; size_t lws = WG;
        cl_event* evt = KernelProfiler::event_for("argmax_row");
        err = clEnqueueNDRangeKernel(queue, argmax_kernel_, 1, nullptr, &gws, &lws, 0, nullptr, evt);
        if (err != CL_SUCCESS) { NNOPT_ERROR_FMT("Lever D argmax enqueue (%d)", err); return false; }
        return true;
    };

    // Iteration 1: input = first_token (host int, via standard embed path).
    // Output: ids_history_buf_[0] holds the 2nd decoded token.
    {
        const int start_pos = (int)ids.size() - 1;  // first decode token's start_pos
        cl_mem logits_buf = compute_logits_({first_token}, start_pos);
        if (!logits_buf) { NNOPT_ERROR("Lever D iter1 forward failed"); goto drain; }
        if (!enqueue_argmax(logits_buf, /*write_idx=*/0)) { goto drain; }
        last_iter_completed = 1;  // wrote history[0] = the 2nd decoded token
    }

    // Iterations 2..max_new_tokens-1.
    for (int i = 2; i < max_new_tokens; i++) {
        // Lever H: when 2-pass argmax is in use, the previous iteration's
        // argmax_final already wrote cur_token_buf_, so the copy is redundant.
        // Otherwise copy ids_history_buf_[i-2] → cur_token_buf_[0].
        if (!use_argmax_2pass) {
            cl_int cerr = clEnqueueCopyBuffer(queue, ids_history_buf_, cur_token_buf_,
                                              (size_t)(i - 2) * sizeof(int32_t), 0,
                                              sizeof(int32_t), 0, nullptr, nullptr);
            if (cerr != CL_SUCCESS) {
                NNOPT_ERROR_FMT("Lever D copy iter %d failed (%d)", i, (int)cerr);
                goto drain;
            }
        }
        const int start_pos = (int)ids.size() + i - 2;  // ids has prompt + first_token; for iter i we're at decode step (i-1) = (i-2)+1 ahead
        cl_mem logits_buf = compute_logits_gpu_input_(cur_token_buf_, start_pos);
        if (!logits_buf) { NNOPT_ERROR_FMT("Lever D iter %d forward failed", i); goto drain; }
        if (!enqueue_argmax(logits_buf, /*write_idx=*/i - 1)) { goto drain; }
        last_iter_completed = i;
    }

drain:
    // Drain the queue and read back all decoded-via-async tokens.
    // last_iter_completed = number of tokens that landed in ids_history_buf_,
    // stored at indices 0 .. last_iter_completed-1.
    clFinish(queue);
    if (last_iter_completed > 0) {
        std::vector<int32_t> decoded((size_t)last_iter_completed);
        cl_int rerr = clEnqueueReadBuffer(queue, ids_history_buf_, CL_TRUE, 0,
                                          (size_t)last_iter_completed * sizeof(int32_t),
                                          decoded.data(), 0, nullptr, nullptr);
        if (rerr == CL_SUCCESS) {
            for (int j = 0; j < last_iter_completed; j++) {
                int tok = (int)decoded[(size_t)j];
                ids.push_back(tok);
                if (on_token) on_token(tok);
                if (sampler_config.eos_token_id >= 0 && tok == sampler_config.eos_token_id) break;
            }
        } else {
            NNOPT_ERROR_FMT("Lever D drain readback failed (%d)", (int)rerr);
        }
    }

    NNOPT_CHECKPOINT("generate() complete");
    return ids;
}
