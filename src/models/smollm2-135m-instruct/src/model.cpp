// Auto-generated model implementation stub for HuggingFaceTB/SmolLM2-135M-Instruct
// This will be filled in by the agent after layer implementation.

#include "model.h"
#include "sampler.h"
#include "debug_utils.h"
#include "utils.h"
#include "model_config.h"
#include "benchmark.h"
#include "prof.h"
#include <CL/cl.h>
#include <clblast.h>
#include <iostream>
#include <memory>
#include <string>

Model::Model(OpenCLContext& cl_ctx, Weights& weights)
    : cl_ctx_(cl_ctx), weights_(weights) {
    // Constructor: refs only. NO kernel/program work here. All per-layer
    // initialize() calls live in Model::initialize() so failures can return
    // false to main() (a void ctor cannot signal failure).
    NNOPT_CHECKPOINT("Model constructor");

    embed_tokens_ = std::make_unique<Embedding>(cl_ctx_, weights_);
    // Final RMSNorm is model.norm.weight (no layer index).
    // Use the 5-arg ctor to avoid accidentally treating it as a per-layer post-attn norm.
    final_norm_ = std::make_unique<LayerNorm>(cl_ctx_, weights_, /*layer_idx=*/0, /*is_post_attn=*/false, /*is_final_norm=*/true);
    lm_head_ = std::make_unique<LmHead>(cl_ctx_, weights_);

    for (int i = 0; i < MODEL_CONFIG::NUM_HIDDEN_LAYERS; i++) {
        input_layernorm_[i] = std::make_unique<LayerNorm>(cl_ctx_, weights_, i, /*is_post_attn=*/false);
        self_attn_[i] = std::make_unique<Attention>(cl_ctx_, weights_, i);
        post_attention_layernorm_[i] = std::make_unique<LayerNorm>(cl_ctx_, weights_, i, /*is_post_attn=*/true);
        mlp_[i] = std::make_unique<Mlp>(cl_ctx_, weights_, i);
    }
}

Model::~Model() {
    if (argmax_partial_)   clReleaseKernel(argmax_partial_);
    if (argmax_final_)     clReleaseKernel(argmax_final_);
    if (argmax_prog_)      clReleaseProgram(argmax_prog_);
    if (argmax_scratch_v_) clReleaseMemObject(argmax_scratch_v_);
    if (argmax_scratch_i_) clReleaseMemObject(argmax_scratch_i_);
    if (argmax_out_idx_)   clReleaseMemObject(argmax_out_idx_);
    if (argmax_cur_token_) clReleaseMemObject(argmax_cur_token_);
}

bool Model::ensure_argmax_program(cl_command_queue queue) {
    if (argmax_tried_) return argmax_ready_;
    argmax_tried_ = true;

    constexpr int NUM_WG = 32;

    cl_context ctx = cl_ctx_.context();
    cl_int err = CL_SUCCESS;

    argmax_prog_ = cl_ctx_.build_program_from_file(
        "kernels/argmax.cl",
#ifdef NNOPT_USE_FP16
        "-DNNOPT_USE_FP16=1 -DUSE_FP16=1 -DNUM_WG=32"
#else
        "-DNUM_WG=32"
#endif
    );
    if (!argmax_prog_) {
        NNOPT_ERROR("Failed to build kernels/argmax.cl");
        return false;
    }
    argmax_partial_ = clCreateKernel(argmax_prog_, "argmax_partial", &err);
    if (err != CL_SUCCESS) { NNOPT_ERROR_FMT("argmax_partial kernel failed: %d", err); return false; }
    argmax_final_ = clCreateKernel(argmax_prog_, "argmax_final", &err);
    if (err != CL_SUCCESS) { NNOPT_ERROR_FMT("argmax_final kernel failed: %d", err); return false; }

    argmax_scratch_v_ = clCreateBuffer(ctx, CL_MEM_READ_WRITE, NUM_WG * sizeof(float), nullptr, &err);
    if (err != CL_SUCCESS) return false;
    argmax_scratch_i_ = clCreateBuffer(ctx, CL_MEM_READ_WRITE, NUM_WG * sizeof(int),   nullptr, &err);
    if (err != CL_SUCCESS) return false;
    argmax_out_idx_   = clCreateBuffer(ctx, CL_MEM_READ_WRITE, sizeof(int), nullptr, &err);
    if (err != CL_SUCCESS) return false;
    argmax_cur_token_ = clCreateBuffer(ctx, CL_MEM_READ_WRITE, sizeof(int), nullptr, &err);
    if (err != CL_SUCCESS) return false;

    (void)queue;
    argmax_ready_ = true;
    return true;
}

int32_t Model::forward_decode_greedy(int32_t token_id, int start_pos) {
    cl_command_queue queue = cl_ctx_.queue();
    cl_context ctx = cl_ctx_.context();
    cl_int err = CL_SUCCESS;

    if (!ensure_argmax_program(queue)) return -1;

    // 1. Token embedding.
    const std::vector<int32_t> single_id = { token_id };
    cl_mem ids_buf = clCreateBuffer(ctx, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
                                    sizeof(int32_t), (void*)single_id.data(), &err);
    if (err != CL_SUCCESS || !ids_buf) return -1;
    cl_mem hidden = embed_tokens_->forward(queue, ids_buf, 1);
    clReleaseMemObject(ids_buf);
    if (!hidden) return -1;

    // 2. Transformer layers (30) — with optional cl_qcom_recordable_queues path.
    //    Recording mode threads layer chain through persistent decode_hidden_buf_
    //    so the same kernel-arg cl_mem handles get captured at record time and
    //    re-used on every replay. Embedding output is copied into the persistent
    //    buffer once per call; subsequent layer dispatches mutate it in place.
    cl_mem hidden_for_layers = hidden;
    if (record_enabled_) {
        err = clEnqueueCopyBuffer(queue, hidden, decode_hidden_buf_, 0, 0,
                                  (size_t)MODEL_CONFIG::HIDDEN_SIZE * sizeof(nnopt_storage_t),
                                  0, nullptr, nullptr);
        clReleaseMemObject(hidden);
        if (err != CL_SUCCESS) { NNOPT_ERROR_FMT("RecordQ: embed→persistent copy failed err=%d", err); return -1; }
        hidden_for_layers = decode_hidden_buf_;
    }

    bool replayed = false;
    if (record_enabled_ && recording_built_) {
        // Replay path: 30 layers + KV cache writes run via the recording.
        cur_start_pos_ = start_pos;
        cur_seq_k_     = start_pos + 1;
        std::vector<cl_array_arg_qcom> args;
        args.reserve(rec_start_pos_args_.size() + rec_seq_k_args_.size());
        for (auto& p : rec_start_pos_args_) {
            args.push_back({p.kernel, p.arg_indx, sizeof(int), &cur_start_pos_});
        }
        for (auto& p : rec_seq_k_args_) {
            args.push_back({p.kernel, p.arg_indx, sizeof(int), &cur_seq_k_});
        }
        // DEBUG: try 0-args first (LFM2 probe baseline) to verify replay API works.
        // If this succeeds we know the issue is in our args[] construction.
        const bool USE_EMPTY_ARGS = std::getenv("NNOPT_RECORD_EMPTY_ARGS") &&
                                    std::getenv("NNOPT_RECORD_EMPTY_ARGS")[0] == '1';
        cl_int rerr;
        if (USE_EMPTY_ARGS) {
            rerr = cl_ctx_.enqueue_recording(queue, recording_, 0, nullptr);
        } else {
            rerr = cl_ctx_.enqueue_recording(queue, recording_, args.size(), args.data());
        }
        if (rerr != CL_SUCCESS) {
            NNOPT_ERROR_FMT("RecordQ: enqueue_recording failed err=%d — disabling and re-dispatching live", rerr);
            record_enabled_ = false;
            // Fall through to live dispatch below.
        } else {
            replayed = true;
        }
    }

    if (!replayed) {
        for (int i = 0; i < MODEL_CONFIG::NUM_HIDDEN_LAYERS; i++) {
            cl_mem normed = input_layernorm_[i]->forward_decode(queue, hidden_for_layers);
            if (!normed) { if (hidden_for_layers != decode_hidden_buf_) clReleaseMemObject(hidden_for_layers); return -1; }
            if (!self_attn_[i]->forward_decode_into_residual(queue, normed, start_pos, hidden_for_layers)) {
                if (hidden_for_layers != decode_hidden_buf_) clReleaseMemObject(hidden_for_layers); return -1;
            }
            cl_mem normed2 = post_attention_layernorm_[i]->forward_decode(queue, hidden_for_layers);
            if (!normed2) { if (hidden_for_layers != decode_hidden_buf_) clReleaseMemObject(hidden_for_layers); return -1; }
            if (!mlp_[i]->forward_decode_into_residual(queue, normed2, hidden_for_layers)) {
                if (hidden_for_layers != decode_hidden_buf_) clReleaseMemObject(hidden_for_layers); return -1;
            }
        }
    }

    // First call after live dispatch (only when recording enabled): capture the
    // dispatch sequence into a recording on record_queue_ (no GPU work — host-
    // side bookkeeping only). End the recording so subsequent calls can replay.
    if (record_enabled_ && !recording_built_) {
        recording_ = cl_ctx_.new_recording(record_queue_);
        if (!recording_) {
            std::cerr << "RecordQ: new_recording failed; disabling" << std::endl;
            record_enabled_ = false;
        } else {
            bool rec_ok = true;
            for (int i = 0; i < MODEL_CONFIG::NUM_HIDDEN_LAYERS; i++) {
                cl_mem normed = input_layernorm_[i]->forward_decode(record_queue_, decode_hidden_buf_);
                if (!normed) { rec_ok = false; break; }
                if (!self_attn_[i]->forward_decode_into_residual(record_queue_, normed, start_pos, decode_hidden_buf_)) { rec_ok = false; break; }
                cl_mem normed2 = post_attention_layernorm_[i]->forward_decode(record_queue_, decode_hidden_buf_);
                if (!normed2) { rec_ok = false; break; }
                if (!mlp_[i]->forward_decode_into_residual(record_queue_, normed2, decode_hidden_buf_)) { rec_ok = false; break; }
            }
            cl_int eerr = cl_ctx_.end_recording(recording_);
            if (!rec_ok || eerr != CL_SUCCESS) {
                std::cerr << "RecordQ: recording build failed (ok=" << rec_ok << " end_err=" << eerr << "); disabling" << std::endl;
                cl_ctx_.release_recording(recording_);
                recording_ = nullptr;
                record_enabled_ = false;
            } else {
                recording_built_ = true;
                std::cerr << "RecordQ: recording built (30 layers, start_pos@record=" << start_pos << ")" << std::endl;
            }
        }
    }

    cl_mem final_hidden = final_norm_->forward_decode(queue, hidden_for_layers);
    if (hidden_for_layers != decode_hidden_buf_) clReleaseMemObject(hidden_for_layers);
    if (!final_hidden) return -1;

    cl_mem logits_buf = lm_head_->forward(queue, final_hidden, 1);
    if (!logits_buf) return -1;

    // 3. GPU argmax — two-pass cooperative reduce.
    constexpr int NUM_WG = 32;
    constexpr int WG_SIZE = 64;
    const int V = MODEL_CONFIG::VOCAB_SIZE;
    const int row_off = 0;
    const int write_off = 0;
    const int valid_n = V;

    err  = clSetKernelArg(argmax_partial_, 0, sizeof(cl_mem), &logits_buf);
    err |= clSetKernelArg(argmax_partial_, 1, sizeof(cl_mem), &argmax_scratch_v_);
    err |= clSetKernelArg(argmax_partial_, 2, sizeof(cl_mem), &argmax_scratch_i_);
    err |= clSetKernelArg(argmax_partial_, 3, sizeof(int),    &V);
    err |= clSetKernelArg(argmax_partial_, 4, sizeof(int),    &valid_n);
    err |= clSetKernelArg(argmax_partial_, 5, sizeof(int),    &row_off);
    if (err != CL_SUCCESS) { clReleaseMemObject(logits_buf); return -1; }
    {
        size_t gws = (size_t)NUM_WG * WG_SIZE;
        size_t lws = WG_SIZE;
        err = nnopt_prof::enqueue(queue, argmax_partial_, 1, nullptr, &gws, &lws, 0, nullptr, nullptr);
        if (err != CL_SUCCESS) { clReleaseMemObject(logits_buf); return -1; }
    }
    err  = clSetKernelArg(argmax_final_, 0, sizeof(cl_mem), &argmax_scratch_v_);
    err |= clSetKernelArg(argmax_final_, 1, sizeof(cl_mem), &argmax_scratch_i_);
    err |= clSetKernelArg(argmax_final_, 2, sizeof(cl_mem), &argmax_out_idx_);
    err |= clSetKernelArg(argmax_final_, 3, sizeof(cl_mem), &argmax_cur_token_);
    err |= clSetKernelArg(argmax_final_, 4, sizeof(int),    &write_off);
    if (err != CL_SUCCESS) { clReleaseMemObject(logits_buf); return -1; }
    {
        size_t gws = NUM_WG, lws = NUM_WG;
        err = nnopt_prof::enqueue(queue, argmax_final_, 1, nullptr, &gws, &lws, 0, nullptr, nullptr);
        if (err != CL_SUCCESS) { clReleaseMemObject(logits_buf); return -1; }
    }

    // 4. Read back single int (4 bytes) instead of 98 KB of logits.
    int32_t out = -1;
    err = clEnqueueReadBuffer(queue, argmax_out_idx_, CL_TRUE, 0, sizeof(int32_t), &out, 0, nullptr, nullptr);
    clReleaseMemObject(logits_buf);
    if (err != CL_SUCCESS) return -1;
    return out;
}

// Per-layer instance + kernel construction. Return false on the FIRST
// per-layer failure — main.cpp checks the return and exits non-zero so the
// process never reaches forward() with un-initialized layers (avoids
// "Forward dispatches on null cl_mem → SIGSEGV inside the driver").
//
// MANDATORY pattern: every per-layer initialize() return is checked and
// failures are logged via NNOPT_ERROR_FMT (NEVER fprintf(stderr) — NNOPT
// macros include file:line automatically). Successful inits emit
// NNOPT_LAYER_INIT_FMT for the deploy/run log to confirm coverage.
//
// Hybrid architectures (where only a subset of layer indices have a given
// sub-block) are dispatched via the layer_has_<class>(i) helpers in
// model.h — emitted automatically when instance_indices < NUM_LAYERS.
// Dense architectures don't have those helpers; the loop runs every i.
bool Model::initialize() {
    // Reference: model_info/transformers_src/modeling_llama.py:375-428 LlamaModel.forward
    // Initialize shared programs used by multiple layers.
    cl_program utils_program = cl_ctx_.build_program_from_file(
        "kernels/utils.cl",
#ifdef NNOPT_USE_FP16
        "-DNNOPT_USE_FP16=1 -DUSE_FP16=1"
#else
        ""
#endif
    );
    if (!utils_program) { NNOPT_ERROR("Failed to build kernels/utils.cl"); return false; }

    if (!embed_tokens_->initialize()) { NNOPT_ERROR_FMT("embed_tokens_.initialize() FAILED"); return false; }
    NNOPT_LAYER_INIT("embed_tokens");

    for (int i = 0; i < MODEL_CONFIG::NUM_HIDDEN_LAYERS; i++) {
        if (!input_layernorm_[i]->initialize()) { NNOPT_ERROR_FMT("input_layernorm_[%d].initialize() FAILED", i); return false; }
        NNOPT_LAYER_INIT_FMT("block%d_sub_input_layernorm", i);

        if (!self_attn_[i]->initialize()) { NNOPT_ERROR_FMT("self_attn_[%d].initialize() FAILED", i); return false; }
        NNOPT_LAYER_INIT_FMT("block%d_sub_attn", i);

        if (!post_attention_layernorm_[i]->initialize()) { NNOPT_ERROR_FMT("post_attention_layernorm_[%d].initialize() FAILED", i); return false; }
        NNOPT_LAYER_INIT_FMT("block%d_sub_post_attention_layernorm", i);

        if (!mlp_[i]->initialize()) { NNOPT_ERROR_FMT("mlp_[%d].initialize() FAILED", i); return false; }
        NNOPT_LAYER_INIT_FMT("block%d_sub_mlp", i);
    }

    if (!final_norm_->initialize()) { NNOPT_ERROR_FMT("final_norm_.initialize() FAILED"); return false; }
    NNOPT_LAYER_INIT("final_norm");

    if (!lm_head_->initialize()) { NNOPT_ERROR_FMT("lm_head_.initialize() FAILED"); return false; }
    NNOPT_LAYER_INIT("lm_head");

    clReleaseProgram(utils_program);

    // Pre-warm RoPE cos/sin tables to MAX_POSITION_EMBEDDINGS up-front for ALL
    // layers. Without this, ensure_rope_tables rebuilds the tables every decode
    // step (seq_k grows by 1 each call → fp32 host loop + clCreateBuffer pair
    // per layer per token). Pre-warming once at init saved +17% decode tok/s
    // on Adreno 619 v2 (Step 12 measurement). Cheap: each layer reuses the
    // same MAX_POSITION_EMBEDDINGS sized buffer.
    for (int i = 0; i < MODEL_CONFIG::NUM_HIDDEN_LAYERS; ++i) {
        if (!self_attn_[i]->prewarm_rope_tables(cl_ctx_.queue(),
                                                 MODEL_CONFIG::MAX_POSITION_EMBEDDINGS)) {
            NNOPT_ERROR_FMT("prewarm_rope_tables failed for layer %d", i);
            return false;
        }
    }
    NNOPT_CHECKPOINT("RoPE tables prewarmed for all layers");

    // ── cl_qcom_recordable_queues setup (opt-in via NNOPT_RECORD=1).
    if (const char* r = std::getenv("NNOPT_RECORD"); r && r[0] == '1') {
        if (!cl_ctx_.has_recordable_queues()) {
            std::cerr << "RecordQ: NNOPT_RECORD=1 set but extension not exposed; disabled" << std::endl;
        } else {
            record_queue_ = cl_ctx_.create_recordable_queue();
            if (record_queue_) {
                cl_int berr = CL_SUCCESS;
                decode_hidden_buf_ = clCreateBuffer(
                    cl_ctx_.context(), CL_MEM_READ_WRITE,
                    (size_t)MODEL_CONFIG::HIDDEN_SIZE * sizeof(nnopt_storage_t),
                    nullptr, &berr);
                if (berr == CL_SUCCESS && decode_hidden_buf_) {
                    // RoPE tables are already pre-warmed unconditionally above.
                    // Collect per-layer kernel handles for the arg-update array on replay.
                    //   fused_rope_kvwrite_m1: start_pos is arg 10 (see attention.cpp dispatch).
                    //   fused_decode_attn_m1:  seq_k is arg 9.
                    for (int i = 0; i < MODEL_CONFIG::NUM_HIDDEN_LAYERS; ++i) {
                        rec_start_pos_args_.push_back({self_attn_[i]->rope_kvwrite_kernel(), 10});
                        rec_seq_k_args_.push_back   ({self_attn_[i]->decode_attn_kernel(),  9});
                    }
                    record_enabled_ = true;
                    std::cerr << "RecordQ: enabled (record_queue=" << record_queue_
                              << ", H=" << MODEL_CONFIG::HIDDEN_SIZE
                              << ", " << rec_start_pos_args_.size() << " per-step args x 2)" << std::endl;
                } else {
                    std::cerr << "RecordQ: alloc decode_hidden_buf_ failed err=" << berr << std::endl;
                    clReleaseCommandQueue(record_queue_);
                    record_queue_ = nullptr;
                }
            }
        }
    }

    NNOPT_CHECKPOINT("Model::initialize() complete");
    return true;
}

// Legacy single-arg forward: a thin wrapper over forward(ids, 0). Kept for
// callers that don't track absolute position. New code should use the
// two-arg form below.
std::vector<float> Model::forward(const std::vector<int32_t>& input_ids) {
    return forward(input_ids, /*start_pos=*/0);
}

// Forward with explicit cache offset (Rules KV-01 / GEN-01).
//
//   start_pos == 0           -> prefill. Process the full prompt sequence
//                                and fill the per-layer KV cache at
//                                positions [0, input_ids.size()).
//   start_pos > 0            -> decode. Process input_ids at absolute
//                                positions [start_pos, start_pos + input_ids.size()).
//                                Attention reads cache rows [0, seq_k) where
//                                seq_k = start_pos + input_ids.size().
//
// Decode fast-path is opt-in. Once prefill is correct, you can OPTIONALLY
// dispatch single-token decodes (seq_len == 1, start_pos > 0) to a
// hand-written forward_decode() that calls the cooperative fused kernels in
// kernels/block_fused.cl (fused_qkv_gemv_m1, fused_oproj_residual_m1,
// fused_down_residual_m1, fused_lm_head_gemv_m1). Until forward_decode is
// wired, single-token decode just runs through the prefill path below — the
// scaffolded forward_decode() default body delegates back to this function.
// That keeps correctness verification independent of the optimization step.
//
// To activate the fast path, after forward_decode() is implemented, add at
// the top of this function:
//     if (seq_len == 1 && start_pos > 0) return forward_decode(input_ids[0], start_pos);
//
// Returns the logits for the LAST token in input_ids.
// Reference: model_info/transformers_src/modeling_llama.py:375-428  (LlamaModel.forward)
//   This is the BACKBONE dispatch graph. The C++ Model::forward below mirrors
//   this PyTorch method line-for-line.
// Reference: model_info/transformers_src/modeling_llama.py:445-501  (LlamaForCausalLM.forward)
//   The HEAD wraps the backbone — `outputs = self.model(input_ids); logits =
//   self.lm_head(outputs)`. lm_head goes AFTER Model::forward, in main.cpp's
//   sampling path or as a tail block of forward() — match what PyTorch does.
//
// HOW TO PORT — read PyTorch line by line, mirror it block by block:
//   1) Read the backbone forward at the cited file:line above. Walk its
//      statements top-to-bottom. Each PyTorch line maps to one C++ block.
//   2) For every layer call (`self.<name>(...)`), open
//      `(model metadata)` for the class behind that name.
//      The contract has the forward() file:line range for the layer class —
//      cite it in the layer's own .cpp file.
//   3) Embedding lookup and final norm placement come straight from the
//      backbone forward — DO NOT GUESS from weight names. The bug class is:
//      "weight is named model.<x>_norm so I'll wire it where weight names
//      suggest" — instead, follow the PyTorch source line order.
//   4) Emit NNOPT_LAYER_CHECK("<canonical_name>", queue, buf, numel) after
//      every sub-block output and after final logits. Use names from
//      `(model metadata)` (canonical mapping).
//   5) Streaming is wired: just call sub-block forward() methods in order.
//      Do NOT remove the on_token callback hook in generate() — that's how
//      the user sees tokens live.
//
// Until you implement this body, every Infer call returns the SENTINEL banner
// below and Build refuses Deploy/Infer with FORWARD_NOT_IMPLEMENTED.
std::vector<float> Model::forward(const std::vector<int32_t>& input_ids, int start_pos) {
    // Reference: model_info/transformers_src/modeling_llama.py:375-428 LlamaModel.forward
    // Reference: model_info/transformers_src/modeling_llama.py:445-501 LlamaForCausalLM.forward
    NNOPT_CHECKPOINT("forward() started");

    const int seq_len = (int)input_ids.size();
    if (seq_len <= 0) {
        NNOPT_ERROR("forward(): empty input_ids");
        return {};
    }

    // Dispatch single-token decode to the fast path (GEMV, not GEMM).
    if (seq_len == 1 && start_pos > 0) return forward_decode(input_ids[0], start_pos);

    // Int8 prefill fallback: CLBlast HGemm in pytorch_linear() does not speak int8,
    // so we route prefill through the decode kernels token-by-token. Slower
    // (O(seq_len) decode steps instead of one batched GEMM) but correctness-correct
    // for the int8 weight path. Only the decode tok/s metric matters for tuning.
    if (const char* q = std::getenv("NNOPT_QUANT"); q && std::string(q) == "int8") {
        std::vector<float> last_logits;
        for (int i = 0; i < seq_len; ++i) {
            last_logits = forward_decode(input_ids[i], start_pos + i);
            if (last_logits.empty()) return {};
        }
        return last_logits;
    }

    cl_command_queue queue = cl_ctx_.queue();

    // Programs
    // TODO(perf): hoist this to Model::initialize() as a member (Rule PROG-01).
    // PROGRAM-INIT-OK: temporary; correctness first.
    cl_program utils_program = cl_ctx_.build_program_from_file(
        "kernels/utils.cl",
#ifdef NNOPT_USE_FP16
        "-DNNOPT_USE_FP16=1 -DUSE_FP16=1"
#else
        ""
#endif
    );
    if (!utils_program) { NNOPT_ERROR("Failed to build utils.cl"); return {}; }

    // Upload token IDs. UPLOAD-OK: input_ids
    // (Build gate UPLOAD-01 requires this exact comment for CL_MEM_COPY_HOST_PTR in forward().)
    cl_int err = CL_SUCCESS;
    // UPLOAD-OK: input_ids
    cl_mem ids_buf = clCreateBuffer(cl_ctx_.context(), CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR,
                                    (size_t)seq_len * sizeof(int32_t), (void*)input_ids.data(), &err);
    if (err != CL_SUCCESS || !ids_buf) {
        NNOPT_ERROR_FMT("alloc ids_buf failed: %d", err);
        clReleaseProgram(utils_program);
        return {};
    }
    NNOPT_LAYER_CHECK_INPUT_INT("input_ids", queue, ids_buf, (size_t)seq_len);

    // embed_tokens
    cl_mem hidden = embed_tokens_->forward(queue, ids_buf, seq_len);
    clReleaseMemObject(ids_buf);
    if (!hidden) { clReleaseProgram(utils_program); return {}; }
    NNOPT_LAYER_CHECK("embed_tokens", queue, hidden, (size_t)seq_len * MODEL_CONFIG::HIDDEN_SIZE);

    // decoder layers
    for (int i = 0; i < MODEL_CONFIG::NUM_HIDDEN_LAYERS; i++) {
        // input_layernorm
        cl_mem normed = input_layernorm_[i]->forward(queue, hidden, seq_len);
        if (!normed) { clReleaseMemObject(hidden); clReleaseProgram(utils_program); return {}; }
        NNOPT_LAYER_CHECK_FMT("block%d_sub_input_layernorm", i, queue, normed, (size_t)seq_len * MODEL_CONFIG::HIDDEN_SIZE);
        if (!normed) { clReleaseMemObject(hidden); clReleaseProgram(utils_program); return {}; }

        // self_attn
        cl_mem attn_out = self_attn_[i]->forward(queue, normed,
                                                 /*cos=*/nullptr, /*sin=*/nullptr,
                                                 /*seq_q=*/seq_len, /*start_pos=*/start_pos);
        clReleaseMemObject(normed);
        if (!attn_out) { clReleaseMemObject(hidden); clReleaseProgram(utils_program); return {}; }

        cl_mem hidden2 = element_add(queue, utils_program, hidden, attn_out,
                                     (size_t)seq_len * MODEL_CONFIG::HIDDEN_SIZE);
        clReleaseMemObject(hidden);
        clReleaseMemObject(attn_out);
        if (!hidden2) { clReleaseProgram(utils_program); return {}; }
        hidden = hidden2;

        // post_attention_layernorm
        cl_mem normed2 = post_attention_layernorm_[i]->forward(queue, hidden, seq_len);
        if (!normed2) { clReleaseMemObject(hidden); clReleaseProgram(utils_program); return {}; }
        NNOPT_LAYER_CHECK_FMT("block%d_sub_post_attention_layernorm", i, queue, normed2, (size_t)seq_len * MODEL_CONFIG::HIDDEN_SIZE);
        if (!normed2) { clReleaseMemObject(hidden); clReleaseProgram(utils_program); return {}; }

        // mlp
        cl_mem mlp_out = mlp_[i]->forward(queue, normed2, seq_len);
        clReleaseMemObject(normed2);
        if (!mlp_out) { clReleaseMemObject(hidden); clReleaseProgram(utils_program); return {}; }

        cl_mem hidden3 = element_add(queue, utils_program, hidden, mlp_out,
                                     (size_t)seq_len * MODEL_CONFIG::HIDDEN_SIZE);
        clReleaseMemObject(hidden);
        clReleaseMemObject(mlp_out);
        if (!hidden3) { clReleaseProgram(utils_program); return {}; }
        hidden = hidden3;
    }

    // final norm
    cl_mem final_hidden = final_norm_->forward(queue, hidden, seq_len);
    clReleaseMemObject(hidden);
    if (!final_hidden) { clReleaseProgram(utils_program); return {}; }

    // lm_head over full sequence (LM-HEAD-01)
    cl_mem logits_buf = lm_head_->forward(queue, final_hidden, seq_len);
    clReleaseMemObject(final_hidden);
    if (!logits_buf) { clReleaseProgram(utils_program); return {}; }
    NNOPT_LAYER_CHECK("lm_head", queue, logits_buf, (size_t)seq_len * MODEL_CONFIG::VOCAB_SIZE);

    // Read last token logits back to host
    std::vector<float> logits((size_t)MODEL_CONFIG::VOCAB_SIZE);
    const size_t row_bytes = (size_t)MODEL_CONFIG::VOCAB_SIZE * sizeof(nnopt_storage_t);
    const size_t off_bytes = (size_t)(seq_len - 1) * row_bytes;

#ifdef NNOPT_USE_FP16
    std::vector<nnopt_storage_t> tmp((size_t)MODEL_CONFIG::VOCAB_SIZE);
    err = clEnqueueReadBuffer(queue, logits_buf, CL_TRUE, off_bytes, row_bytes, tmp.data(), 0, nullptr, nullptr);
    if (err != CL_SUCCESS) {
        NNOPT_ERROR_FMT("read logits fp16 failed: %d", err);
        clReleaseMemObject(logits_buf);
        clReleaseProgram(utils_program);
        return {};
    }
    for (int i = 0; i < MODEL_CONFIG::VOCAB_SIZE; i++) logits[i] = nnopt_f16_to_f32((uint16_t)tmp[(size_t)i]);
#else
    err = clEnqueueReadBuffer(queue, logits_buf, CL_TRUE, off_bytes, row_bytes, logits.data(), 0, nullptr, nullptr);
    if (err != CL_SUCCESS) {
        NNOPT_ERROR_FMT("read logits fp32 failed: %d", err);
        clReleaseMemObject(logits_buf);
        clReleaseProgram(utils_program);
        return {};
    }
#endif

    clReleaseMemObject(logits_buf);
    clReleaseProgram(utils_program);

    NNOPT_CHECKPOINT("forward() complete");
    return logits;
}

// ── Decode fast path (OPTIONAL — single-token forward) ──────────────────
// AGENT NOTE — wire this AFTER prefill correctness is verified. The default
// body delegates to forward() so an unwired decode path just runs (slowly)
// through prefill, which means you can ship a correct port without ever
// touching this function. Wiring it is a performance optimization, not a
// correctness step.
//
// Architecture-blind canonical wiring (~20 lines — uses ONLY scaffold-
// provided kernels in kernels/block_fused.cl, no new kernels to write for
// any HF transformer family):
//
//   cl_command_queue queue = cl_ctx_.queue();
//   cl_mem hidden = embedding_.forward(queue, {token_id}, /*seq_len=*/1);
//
//   for (int i = 0; i < <num_layers_const>; i++) {                         // <- from your model_config.h
//       cl_mem residual = hidden;                                          //    (e.g. NUM_HIDDEN_LAYERS, N_LAYER, ...)
//       cl_mem normed = pre_attn_norm_[i]->forward(queue, residual, 1);    // <- whatever you named the pre-attn norm in model.h
//
//       // Attention::forward_decode_into_residual fans out QKV via fused_qkv_gemv_m1
//       // (or fused_c_attn_gemv_m1 for Conv1D-c_attn architectures), runs
//       // RoPE+kvwrite, attention, and folds o_proj+residual back into
//       // 'residual' in one cooperative call sequence. Returns true on success.
//       if (!attn_[i]->forward_decode_into_residual(queue, normed,
//               /*cos=*/nullptr, /*sin=*/nullptr, start_pos, residual)) {
//           NNOPT_ERROR_FMT("attn forward_decode_into_residual failed at layer %d", i);
//           clReleaseMemObject(normed); clReleaseMemObject(residual);
//           return std::vector<float>(VOCAB_SIZE, 0.0f);
//       }
//       clReleaseMemObject(normed);
//
//       cl_mem normed2 = post_attn_norm_[i]->forward(queue, residual, 1);  // <- whatever you named the post-attn norm
//       if (!mlp_[i]->forward_decode_into_residual(queue, normed2, residual)) {
//           NNOPT_ERROR_FMT("mlp forward_decode_into_residual failed at layer %d", i);
//           clReleaseMemObject(normed2); clReleaseMemObject(residual);
//           return std::vector<float>(VOCAB_SIZE, 0.0f);
//       }
//       clReleaseMemObject(normed2);
//       hidden = residual;
//   }
//
//   hidden = final_norm_->forward(queue, hidden, 1);                       // <- whatever you named the final pre-lm_head norm
//   // lm_head_->forward must dispatch fused_lm_head_gemv_m1 internally on
//   // the M=1 path (scaffold-emitted lm_head.cpp wires this when present).
//   return lm_head_->forward(queue, hidden, 1);
//
// Member names above (pre_attn_norm_, post_attn_norm_, final_norm_) are
// PLACEHOLDERS — substitute the actual identifiers you used in model.h. For
// HF transformers the rotation is universal: norm → attention → residual →
// norm → mlp → residual, repeated num_layers times, then a final norm + lm_head.
//
// CORRECTNESS INVARIANT: prefill+decode must match prefill-only ID-for-ID
// at greedy decode (--temperature 0). Any drift is a bug in this function
// or in a forward_decode_into_residual.
std::vector<float> Model::forward_decode(int32_t token_id, int start_pos) {
    NNOPT_CHECKPOINT("forward_decode() started");

    cl_command_queue queue = cl_ctx_.queue();
    cl_context ctx = cl_ctx_.context();
    cl_int err = CL_SUCCESS;

    // 1. Token embedding — single-token lookup, produces hidden [H].
    const std::vector<int32_t> single_id = { token_id };
    const size_t id_bytes = sizeof(int32_t);
    cl_mem ids_buf = clCreateBuffer(ctx, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
                                    id_bytes, (void*)single_id.data(), &err);
    if (err != CL_SUCCESS || !ids_buf) {
        NNOPT_ERROR_FMT("forward_decode: alloc ids_buf failed: %d", err);
        return {};
    }

    cl_mem hidden = embed_tokens_->forward(queue, ids_buf, 1);
    clReleaseMemObject(ids_buf);
    if (!hidden) { NNOPT_ERROR("forward_decode: embedding failed"); return {}; }

    // 2. Transformer layers — each updates hidden in-place via residual stream.
    // NOTE: forward_decode() returns a persistent non-owned buffer — do NOT
    // call clReleaseMemObject on it. Ownership stays with the LayerNorm.
    for (int i = 0; i < MODEL_CONFIG::NUM_HIDDEN_LAYERS; i++) {
        // Pre-attention RMSNorm: normed = norm(hidden) — non-owned persistent buffer
        cl_mem normed = input_layernorm_[i]->forward_decode(queue, hidden);
        if (!normed) {
            clReleaseMemObject(hidden);
            NNOPT_ERROR_FMT("forward_decode: input_layernorm[%d] failed", i);
            return {};
        }

        // Attention GEMV path: attn(normed) + hidden -> hidden in-place
        if (!self_attn_[i]->forward_decode_into_residual(queue, normed, start_pos, hidden)) {
            clReleaseMemObject(hidden);
            NNOPT_ERROR_FMT("forward_decode: attn[%d] failed", i);
            return {};
        }
        // normed is non-owned — do NOT release.

        // Post-attention RMSNorm — non-owned persistent buffer
        cl_mem normed2 = post_attention_layernorm_[i]->forward_decode(queue, hidden);
        if (!normed2) {
            clReleaseMemObject(hidden);
            NNOPT_ERROR_FMT("forward_decode: post_attention_layernorm[%d] failed", i);
            return {};
        }

        // MLP GEMV path: mlp(normed2) + hidden -> hidden in-place
        if (!mlp_[i]->forward_decode_into_residual(queue, normed2, hidden)) {
            clReleaseMemObject(hidden);
            NNOPT_ERROR_FMT("forward_decode: mlp[%d] failed", i);
            return {};
        }
        // normed2 is non-owned — do NOT release.
    }

    // 3. Final RMSNorm + lm_head.
    // final_hidden is non-owned persistent buffer — do NOT release.
    cl_mem final_hidden = final_norm_->forward_decode(queue, hidden);
    clReleaseMemObject(hidden);
    if (!final_hidden) { NNOPT_ERROR("forward_decode: final_norm failed"); return {}; }

    // lm_head detects M==1 and dispatches fused_lm_head_gemv_m1 automatically.
    cl_mem logits_buf = lm_head_->forward(queue, final_hidden, 1);
    // final_hidden is non-owned — do NOT release.
    if (!logits_buf) { NNOPT_ERROR("forward_decode: lm_head failed"); return {}; }

    // 4. Readback logits to CPU.
    const size_t V = (size_t)MODEL_CONFIG::VOCAB_SIZE;
    std::vector<float> logits_fp32(V);

#ifdef NNOPT_USE_FP16
    std::vector<nnopt_storage_t> logits_raw(V);
    err = clEnqueueReadBuffer(queue, logits_buf, CL_TRUE, 0,
                              V * sizeof(nnopt_storage_t), logits_raw.data(), 0, nullptr, nullptr);
    if (err != CL_SUCCESS) {
        NNOPT_ERROR_FMT("forward_decode: readback logits failed: %d", err);
        clReleaseMemObject(logits_buf);
        return {};
    }
    for (size_t j = 0; j < V; ++j) logits_fp32[j] = nnopt_f16_to_f32(logits_raw[j]);
#else
    err = clEnqueueReadBuffer(queue, logits_buf, CL_TRUE, 0,
                              V * sizeof(float), logits_fp32.data(), 0, nullptr, nullptr);
    if (err != CL_SUCCESS) {
        NNOPT_ERROR_FMT("forward_decode: readback logits failed: %d", err);
        clReleaseMemObject(logits_buf);
        return {};
    }
#endif

    clReleaseMemObject(logits_buf);
    NNOPT_CHECKPOINT("forward_decode() complete");
    return logits_fp32;
}

// Auto-regressive generation. Prefill+decode shape (Rule GEN-01 / KV-01).
// Do NOT change this to call forward(ids) inside the loop — that's the
// quadratic regression we already shipped and had to retrofit. Decode time
// per token is O(1) in prefix length when the KV cache is properly used.
//
// The optional on_token callback fires once per newly-generated token (NOT
// for prompt tokens), in BOTH the first-token path AND the decode loop. It
// is the streaming hook main.cpp uses to print tokens as they arrive — do
// NOT remove the callback fires below or the user sees a 30-second pause
// followed by a wall of text instead of a live stream.
std::vector<int32_t> Model::generate(
    const std::vector<int32_t>& prompt_ids,
    int max_new_tokens,
    SamplerConfig sampler_config,
    Model::TokenCallback on_token
) {
    NNOPT_CHECKPOINT("generate() started");
    Sampler sampler(sampler_config);
    auto ids = prompt_ids;

    if (prompt_ids.empty()) {
        NNOPT_ERROR("generate(): empty prompt");
        return ids;
    }

    // Prefill: process the full prompt once. Fills KV cache at positions
    // [0, prompt_ids.size()). Returns logits for the last prompt token.
    auto logits = forward(prompt_ids, /*start_pos=*/0);
    if (logits.empty()) {
        NNOPT_ERROR("prefill forward() returned empty logits");
        return ids;
    }

    std::vector<int32_t> generated;
    int next_token = sampler.sample(logits, generated);
    ids.push_back(next_token);
    generated.push_back(next_token);
    if (on_token) on_token(next_token);
    NNOPT_BENCH_FIRST_TOKEN();
    if (sampler_config.eos_token_id >= 0 && next_token == sampler_config.eos_token_id) {
        NNOPT_CHECKPOINT("generate() complete (eos at first token)");
        return ids;
    }

    // Decode loop: each step processes ONE new token. start_pos points at the
    // absolute position of that new token. Attention reuses cached K/V for
    // all prior tokens — work per step is O(layers * (hidden + seq_k * head_dim)),
    // not O((P + i)^2 * ...) like a re-prefill loop.
    //
    // GPU-argmax fast path: when sampler is pure greedy (temp ≤ 0,
    // rep_penalty == 1), we run argmax on the GPU and read back a single int
    // instead of the full V=49152 fp16 logits buffer (98 KB blocking transfer).
    const bool greedy_fast_path =
        sampler_config.temperature <= 0.0f &&
        sampler_config.repetition_penalty == 1.0f;

    for (int i = 1; i < max_new_tokens; i++) {
        const int start_pos = (int)prompt_ids.size() + (i - 1);

        if (greedy_fast_path) {
            int32_t tok = forward_decode_greedy(next_token, start_pos);
            if (tok < 0) {
                NNOPT_ERROR("decode forward_decode_greedy failed; falling back to logits readback");
                std::vector<int32_t> single = { next_token };
                logits = forward(single, start_pos);
                if (logits.empty()) break;
                next_token = sampler.sample(logits, generated);
            } else {
                next_token = tok;
            }
        } else {
            std::vector<int32_t> single = { next_token };
            logits = forward(single, start_pos);
            if (logits.empty()) {
                NNOPT_ERROR("decode forward() returned empty logits");
                break;
            }
            next_token = sampler.sample(logits, generated);
        }
        ids.push_back(next_token);
        generated.push_back(next_token);
        if (on_token) on_token(next_token);

        if (sampler_config.eos_token_id >= 0 && next_token == sampler_config.eos_token_id) break;
    }

    NNOPT_CHECKPOINT("generate() complete");
    return ids;
}

// Legacy generate-loop body kept here as a comment so anyone reading the
// scaffold knows what NOT to write:
//
//   for (int i = 0; i < max_new_tokens; i++) {
//       auto logits = forward(ids);                         // ← BANNED: O((P+N)^2)
//       int next_token = sampler.sample(logits, generated);
//       ids.push_back(next_token);
//   }
//
// Build refuses any Model::generate body that contains forward(ids) inside
// a for loop (Rule GEN-01).
#if 0  // (kept for documentation; never compiled)
std::vector<int32_t> Model::generate_LEGACY_DO_NOT_USE(
    const std::vector<int32_t>& prompt_ids,
    int max_new_tokens,
    SamplerConfig sampler_config
) {
    Sampler sampler(sampler_config);
    auto ids = prompt_ids;
    for (int i = 0; i < max_new_tokens; i++) {
        auto logits = forward(ids);

        std::vector<int32_t> generated(ids.begin() + prompt_ids.size(), ids.end());
        int next_token = sampler.sample(logits, generated);
        ids.push_back(next_token);

        if (sampler_config.eos_token_id >= 0 && next_token == sampler_config.eos_token_id) break;
    }

    NNOPT_CHECKPOINT("generate() complete");
    return ids;
}
#endif  // legacy generate documentation block — DO NOT WRITE THIS SHAPE
