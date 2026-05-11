// Auto-generated model implementation stub for ibm-granite/granite-4.0-350m
// This will be filled in by the agent after layer implementation.

#include "model.h"
#include "sampler.h"
#include "debug_utils.h"
#include "model_config.h"
#include "utils.h"
#include "benchmark.h"
#include "kernel_profiler.h"

#include <chrono>
#include <cstdio>
#include <iostream>
#include <cstdio>
#include <cstring>

Model::Model(OpenCLContext& cl_ctx, Weights& weights)
    : cl_ctx_(cl_ctx), weights_(weights) {
    // Constructor: refs only. NO kernel/program work here. All per-layer
    // initialize() calls live in Model::initialize() so failures can return
    // false to main() (a void ctor cannot signal failure).
    NNOPT_CHECKPOINT("Model constructor");
}

Model::~Model() {
    if (logits_buf_) clReleaseMemObject(logits_buf_);
    if (argmax_kernel_) clReleaseKernel(argmax_kernel_);
    if (argmax_program_) clReleaseProgram(argmax_program_);
    if (argmax_result_) clReleaseMemObject(argmax_result_);
}

bool Model::ensure_argmax_resources_() {
    if (argmax_kernel_ && argmax_result_) return true;
    if (!argmax_program_) {
        argmax_program_ = cl_ctx_.build_program_from_file(
            "kernels/argmax.cl",
#ifdef NNOPT_USE_FP16
            "-DNNOPT_USE_FP16=1 -DUSE_FP16=1 -DWG_SIZE=256"
#else
            "-DWG_SIZE=256"
#endif
        );
        if (!argmax_program_) {
            NNOPT_ERROR("ensure_argmax_resources_: build_program_from_file(kernels/argmax.cl) failed");
            return false;
        }
    }
    if (!argmax_kernel_) {
        cl_int err = CL_SUCCESS;
        argmax_kernel_ = clCreateKernel(argmax_program_, "argmax_fp16", &err);
        if (err != CL_SUCCESS || !argmax_kernel_) {
            NNOPT_ERROR_FMT("ensure_argmax_resources_: clCreateKernel(argmax_fp16) failed (%d)", err);
            return false;
        }
    }
    if (!argmax_result_) {
        cl_int err = CL_SUCCESS;
        argmax_result_ = clCreateBuffer(cl_ctx_.context(), CL_MEM_READ_WRITE,
                                        sizeof(int32_t), nullptr, &err);
        if (err != CL_SUCCESS || !argmax_result_) {
            NNOPT_ERROR_FMT("ensure_argmax_resources_: clCreateBuffer argmax_result_ failed (%d)", err);
            return false;
        }
    }
    return true;
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
    NNOPT_CHECKPOINT("Model::initialize() start");

    embedding_ = std::make_unique<Embedding>(cl_ctx_, weights_);
    if (!embedding_->initialize()) { NNOPT_ERROR("embedding.initialize() FAILED"); return false; }
    NNOPT_LAYER_INIT("embedding");

    final_norm_ = std::make_unique<LayerNorm>(cl_ctx_, weights_, /*layer_idx=*/-1, LayerNorm::Kind::FinalNorm);
    if (!final_norm_->initialize()) { NNOPT_ERROR("final_norm.initialize() FAILED"); return false; }
    NNOPT_LAYER_INIT("final_norm");

    for (int i = 0; i < MODEL_CONFIG::NUM_LAYERS; i++) {
        input_layernorm_[i] = std::make_unique<LayerNorm>(cl_ctx_, weights_, i, LayerNorm::Kind::InputLayerNorm);
        if (!input_layernorm_[i]->initialize()) { NNOPT_ERROR_FMT("input_layernorm[%d].initialize() FAILED", i); return false; }
        NNOPT_LAYER_INIT_FMT("input_layernorm_%d", i);

        attn_[i] = std::make_unique<Attention>(cl_ctx_, weights_, i);
        if (!attn_[i]->initialize()) { NNOPT_ERROR_FMT("self_attn[%d].initialize() FAILED", i); return false; }
        NNOPT_LAYER_INIT_FMT("self_attn_%d", i);

        post_attention_layernorm_[i] = std::make_unique<LayerNorm>(cl_ctx_, weights_, i, LayerNorm::Kind::PostAttentionLayerNorm);
        if (!post_attention_layernorm_[i]->initialize()) { NNOPT_ERROR_FMT("post_attention_layernorm[%d].initialize() FAILED", i); return false; }
        NNOPT_LAYER_INIT_FMT("post_attention_layernorm_%d", i);

        shared_mlp_[i] = std::make_unique<SharedMlp>(cl_ctx_, weights_, i);
        if (!shared_mlp_[i]->initialize()) { NNOPT_ERROR_FMT("shared_mlp[%d].initialize() FAILED", i); return false; }
        NNOPT_LAYER_INIT_FMT("shared_mlp_%d", i);
    }

    NNOPT_CHECKPOINT("Model::initialize() complete");
    return true;

    // TODO: Replace this stub with real per-layer initialize() calls. The
    // shape varies per architecture — consult the per-layer contracts to know
    // what to instantiate. Example:
    //
    //   if (!embedding_.initialize()) { NNOPT_ERROR_FMT("embedding.initialize() FAILED"); return false; }
    //   for (int i = 0; i < NUM_LAYERS; i++) {
    //       if (!ln1_[i].initialize()) { NNOPT_ERROR_FMT("ln1_[%d].initialize() FAILED", i); return false; }
    //       NNOPT_LAYER_INIT_FMT("ln1_%d", i);
    //       if (!attn_[i].initialize()) { NNOPT_ERROR_FMT("attn_[%d].initialize() FAILED", i); return false; }
    //       NNOPT_LAYER_INIT_FMT("attn_%d", i);
    //       if (!mlp_[i].initialize()) { NNOPT_ERROR_FMT("mlp_[%d].initialize() FAILED", i); return false; }
    //       NNOPT_LAYER_INIT_FMT("mlp_%d", i);
    //   }
    //   if (!final_norm_.initialize()) { NNOPT_ERROR_FMT("final_norm.initialize() FAILED"); return false; }
    //   if (!lm_head_.initialize()) { NNOPT_ERROR_FMT("lm_head.initialize() FAILED"); return false; }
    //
    // For HYBRID architectures (where attention/conv/etc. live on different
    // layer index subsets — visible as static constexpr <CLASS>_LAYER_INDICES
    // arrays in model.h), gate per-class init on the helper:
    //
    //   for (int i = 0; i < NUM_LAYERS; i++) {
    //       if (layer_has_attention(i)) {
    //           if (!attention_[i].initialize()) { NNOPT_ERROR_FMT("attention_[%d].initialize() FAILED", i); return false; }
    //       } else if (layer_has_convolution(i)) {
    //           if (!convolution_[i].initialize()) { NNOPT_ERROR_FMT("convolution_[%d].initialize() FAILED", i); return false; }
    //       }
    //   }
    //
    // INIT-NULLPTR DISCIPLINE — when you write raw clCreateKernel / get_buffer
    // calls directly into Model::initialize() (e.g., for a final-norm or
    // lm_head kernel cached on the Model rather than wrapped in a sub-Layer),
    // you MUST guard every return for nullptr so a missing kernel name or
    // wrong weight key fails fast with a stderr log instead of crashing the
    // device on first dispatch. Pattern (mirrors src/layers/attention.cpp):
    //
    //   cl_int err = CL_SUCCESS;
    //   final_norm_kernel_ = clCreateKernel(rmsnorm_program_, "rmsnorm_forward", &err);
    //   if (err != CL_SUCCESS || !final_norm_kernel_) {
    //       NNOPT_ERROR_FMT("clCreateKernel("rmsnorm_forward") failed (%d) — check kernels/rmsnorm.cl defines this name", err);
    //       return false;
    //   }
    //   final_norm_weight_ = weights_.get_buffer("model.norm.weight");
    //   if (!final_norm_weight_) {
    //       NNOPT_ERROR("get_buffer("model.norm.weight") returned nullptr — check the key against weights/model.meta.json");
    //       return false;
    //   }
    //
    // Why this matters: clCreateKernel returning nullptr with status -46
    // (CL_INVALID_KERNEL_NAME) is silent unless you check; the next dispatch
    // SIGSEGVs inside the Adreno driver and the device reboots. Same story
    // for get_buffer returning nullptr on a typo'd weight key. Belt-and-
    // suspenders cheap; device-reboot debugging cycles expensive.
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
// Reference: model_info/transformers_src/modeling_granitemoehybrid.py:1151-1210  (GraniteMoeHybridModel.forward)
//   This is the BACKBONE dispatch graph. The C++ Model::forward below mirrors
//   this PyTorch method line-for-line.
// Reference: model_info/transformers_src/modeling_granitemoehybrid.py:1328-1413  (GraniteMoeHybridForCausalLM.forward)
//   The HEAD wraps the backbone — `outputs = self.model(input_ids); logits =
//   self.lm_head(outputs)`. lm_head goes AFTER Model::forward, in main.cpp's
//   sampling path or as a tail block of forward() — match what PyTorch does.
//
// HOW TO PORT — read PyTorch line by line, mirror it block by block:
//   1) Read the backbone forward at the cited file:line above. Walk its
//      statements top-to-bottom. Each PyTorch line maps to one C++ block.
//   2) For every layer call (`self.<name>(...)`), consult the layer contract
//      for the class behind that name. The contract has the forward()
//      file:line range for the layer class — cite it in the layer's .cpp.
//   3) Embedding lookup and final norm placement come straight from the
//      backbone forward — DO NOT GUESS from weight names. The bug class is:
//      "weight is named model.<x>_norm so I'll wire it where weight names
//      suggest" — instead, follow the PyTorch source line order.
//   4) Emit NNOPT_LAYER_CHECK("<canonical_name>", queue, buf, numel) after
//      every sub-block output and after final logits. Use names from the
//      canonical dump-name map.
//   5) Streaming is wired: just call sub-block forward() methods in order.
//      Do NOT remove the on_token callback hook in generate() — that's how
//      the user sees tokens live.
//
// Until you implement this body, every Infer call returns the SENTINEL banner
// below and Build refuses Deploy/Infer with FORWARD_NOT_IMPLEMENTED.
// Per-section host-time accumulators inside forward(). Internal — used by
// the host-breakdown print at the end of generate(). Reset per-process so
// they aggregate across all forward() calls of the run.
static double g_fw_layers_ms   = 0.0;
static double g_fw_lm_head_ms  = 0.0;
static double g_fw_logits_read_ms = 0.0;
static double g_fw_logits_cvt_ms  = 0.0;
static int    g_fw_calls       = 0;

std::vector<float> Model::forward(const std::vector<int32_t>& input_ids, int start_pos) {
    NNOPT_CHECKPOINT("forward() started");

    const int seq_len = (int)input_ids.size();
    if (seq_len <= 0) {
        NNOPT_ERROR("forward(): empty input_ids");
        return {};
    }

    cl_command_queue queue = cl_ctx_.queue();
    cl_context ctx = cl_ctx_.context();
    using clk = std::chrono::steady_clock;
    auto t_fw_start = clk::now();

    // Upload token ids (int32)
    cl_int err = CL_SUCCESS;
    cl_mem ids_buf = clCreateBuffer(ctx, CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR,
        (size_t)seq_len * sizeof(int32_t), (void*)input_ids.data(), &err);  // UPLOAD-OK: input_ids
    if (err != CL_SUCCESS || !ids_buf) {
        NNOPT_ERROR_FMT("forward(): clCreateBuffer ids_buf failed (%d)", err);
        return {};
    }

    // Embedding
    cl_mem hidden = embedding_->forward(queue, ids_buf, seq_len, start_pos);
    clReleaseMemObject(ids_buf);
    if (!hidden) {
        NNOPT_ERROR("forward(): embedding forward failed");
        return {};
    }

    // Decoder layers: residual + attn * residual_multiplier + ffn * residual_multiplier
    for (int i = 0; i < MODEL_CONFIG::NUM_LAYERS; i++) {
        cl_mem residual0 = hidden;

        cl_mem normed1 = input_layernorm_[i]->forward(queue, residual0, seq_len);
        if (!normed1) { NNOPT_ERROR_FMT("forward(): input_layernorm failed at layer %d", i); clReleaseMemObject(residual0); return {}; }

        cl_mem attn_out = attn_[i]->forward(queue, normed1, /*seq_len=*/seq_len,
                                            /*start_pos=*/start_pos,
                                            /*cos=*/embedding_->rope_cos(),
                                            /*sin=*/embedding_->rope_sin());
        clReleaseMemObject(normed1);
        if (!attn_out) { NNOPT_ERROR_FMT("forward(): attn failed at layer %d", i); clReleaseMemObject(residual0); return {}; }

        // element_add kernel lives in kernels/utils.cl.
        // PROGRAM-INIT-OK: build once per forward() (prefill+decode) to avoid per-layer compilation.
        static cl_program utils_program = nullptr;
        if (!utils_program) {
            utils_program = cl_ctx_.build_program_from_file("kernels/utils.cl");
            if (!utils_program) {
                NNOPT_ERROR("forward(): build_program_from_file(kernels/utils.cl) failed");
                clReleaseMemObject(attn_out);
                clReleaseMemObject(residual0);
                return {};
            }
        }
        // Step 8b: in-place residual add. residual0 is mutated to hold
        //   residual0 = residual0 + attn_out * RESIDUAL_MULTIPLIER
        // Saves the clCreateBuffer + clEnqueueCopyBuffer that the non-inplace
        // element_add does on every layer.
        if (!element_add_inplace(queue, utils_program, residual0, attn_out,
                                 (size_t)seq_len * MODEL_CONFIG::HIDDEN_SIZE,
                                 MODEL_CONFIG::RESIDUAL_MULTIPLIER)) {
            NNOPT_ERROR_FMT("forward(): residual add (attn) failed at layer %d", i);
            clReleaseMemObject(attn_out);
            clReleaseMemObject(residual0);
            return {};
        }
        clReleaseMemObject(attn_out);
        cl_mem new_resid0 = residual0;  // residual0 ownership flows forward

        cl_mem residual1 = new_resid0;
        cl_mem normed2 = post_attention_layernorm_[i]->forward(queue, residual1, seq_len);
        if (!normed2) { NNOPT_ERROR_FMT("forward(): post_attention_layernorm failed at layer %d", i); clReleaseMemObject(residual1); return {}; }

        // Step 8: shared_mlp now fuses the residual add. Returns
        //   new_hidden = residual1 + branch * RESIDUAL_MULTIPLIER
        // directly — no need for a separate element_add call.
        cl_mem new_hidden = shared_mlp_[i]->forward(queue, normed2, residual1,
                                                    MODEL_CONFIG::RESIDUAL_MULTIPLIER, seq_len);
        clReleaseMemObject(normed2);
        clReleaseMemObject(residual1);
        if (!new_hidden) { NNOPT_ERROR_FMT("forward(): shared_mlp failed at layer %d", i); return {}; }
        hidden = new_hidden;
    }

    // End of per-layer loop — record layers wall (includes GPU enqueue +
    // any implicit syncs caused by buffer alloc/copy).
    auto t_layers_end = clk::now();
    g_fw_layers_ms += std::chrono::duration<double, std::milli>(t_layers_end - t_fw_start).count();

    // Final norm
    cl_mem final_hidden = final_norm_->forward(queue, hidden, seq_len);
    clReleaseMemObject(hidden);
    if (!final_hidden) {
        NNOPT_ERROR("forward(): final_norm failed");
        return {};
    }

    // lm_head is tied to embed_tokens.weight; run full [seq_len, vocab] then read last row
    cl_mem wte = weights_.get_buffer("model.embed_tokens.weight");
    if (!wte) { NNOPT_ERROR("forward(): missing model.embed_tokens.weight for lm_head"); clReleaseMemObject(final_hidden); return {}; }

    const size_t logits_elems = (size_t)seq_len * (size_t)MODEL_CONFIG::VOCAB_SIZE;
    auto t_lm_start = clk::now();
    // Persistent logits scratch (Step 6) — geometric-growth alloc, freed in
    // ~Model. Was clCreateBuffer per forward call (200 KB for decode, 1.2 MB
    // for prefill at seq=6).
    const size_t logits_bytes = logits_elems * sizeof(nnopt_storage_t);
    if (!logits_buf_ || logits_buf_cap_bytes_ < logits_bytes) {
        if (logits_buf_) { clReleaseMemObject(logits_buf_); logits_buf_ = nullptr; }
        size_t new_cap = (logits_buf_cap_bytes_ == 0) ? logits_bytes : (logits_buf_cap_bytes_ * 2);
        while (new_cap < logits_bytes) new_cap *= 2;
        logits_buf_ = clCreateBuffer(ctx, CL_MEM_READ_WRITE, new_cap, nullptr, &err);
        if (err != CL_SUCCESS || !logits_buf_) {
            NNOPT_ERROR_FMT("forward(): clCreateBuffer logits (%zu B) failed (%d)", new_cap, err);
            clReleaseMemObject(final_hidden);
            logits_buf_cap_bytes_ = 0;
            return {};
        }
        logits_buf_cap_bytes_ = new_cap;
    }
    cl_mem logits_buf = logits_buf_;
    if (!pytorch_linear(queue, seq_len, MODEL_CONFIG::VOCAB_SIZE, MODEL_CONFIG::HIDDEN_SIZE, final_hidden, wte, logits_buf)) {
        NNOPT_ERROR("forward(): pytorch_linear lm_head failed");
        clReleaseMemObject(final_hidden);
        clReleaseMemObject(logits_buf);
        return {};
    }
    NNOPT_LAYER_CHECK("lm_head", queue, logits_buf, logits_elems);
    clReleaseMemObject(final_hidden);
    auto t_lm_end = clk::now();
    g_fw_lm_head_ms += std::chrono::duration<double, std::milli>(t_lm_end - t_lm_start).count();

    // Step 12: zero-copy logits read via clEnqueueMapBuffer. On integrated
    // GPUs (Adreno is one — LPDDR is shared) this can avoid the explicit
    // 200 KB host copy. Map is still BLOCKING (CL_TRUE) — it must wait for
    // GPU writes to complete — so the GPU drain time is unchanged, but the
    // memcpy-to-host-buffer step is gone.
    const size_t row_bytes = (size_t)MODEL_CONFIG::VOCAB_SIZE * sizeof(nnopt_storage_t);
    const size_t row_off = (size_t)(seq_len - 1) * row_bytes;
    auto t_read_start = clk::now();
    cl_int map_err = CL_SUCCESS;
    void* mapped = clEnqueueMapBuffer(queue, logits_buf, CL_TRUE, CL_MAP_READ,
                                       row_off, row_bytes,
                                       0, nullptr, nullptr, &map_err);
    auto t_read_end = clk::now();
    g_fw_logits_read_ms += std::chrono::duration<double, std::milli>(t_read_end - t_read_start).count();
    if (map_err != CL_SUCCESS || !mapped) {
        NNOPT_ERROR_FMT("forward(): clEnqueueMapBuffer logits failed (%d)", map_err);
        return {};
    }

    auto t_cvt_start = clk::now();
    std::vector<float> out((size_t)MODEL_CONFIG::VOCAB_SIZE);
#ifdef NNOPT_USE_FP16
    const uint16_t* mapped_h = static_cast<const uint16_t*>(mapped);
    for (int i = 0; i < MODEL_CONFIG::VOCAB_SIZE; i++) {
        out[(size_t)i] = nnopt_f16_to_f32(mapped_h[i]) / (float)MODEL_CONFIG::LOGITS_SCALING;
    }
#else
    const float* mapped_f = static_cast<const float*>(mapped);
    for (int i = 0; i < MODEL_CONFIG::VOCAB_SIZE; i++) {
        out[(size_t)i] = mapped_f[i] / (float)MODEL_CONFIG::LOGITS_SCALING;
    }
#endif
    cl_int unmap_err = clEnqueueUnmapMemObject(queue, logits_buf, mapped, 0, nullptr, nullptr);
    if (unmap_err != CL_SUCCESS) {
        NNOPT_ERROR_FMT("forward(): clEnqueueUnmapMemObject logits failed (%d)", unmap_err);
    }
    g_fw_logits_cvt_ms += std::chrono::duration<double, std::milli>(clk::now() - t_cvt_start).count();
    g_fw_calls++;

    NNOPT_CHECKPOINT("forward() complete");
    return out;
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
    NNOPT_TODO("Model::forward_decode (delegating to prefill path until wired)");
    // Default: delegate to forward() with a single-token sequence. Slow but
    // correct — same per-layer code path as prefill. Replace this body with
    // the cooperative fused-decode pipeline once prefill is correct.
    return forward(std::vector<int32_t>{token_id}, start_pos);
}

// Step #1: greedy GPU-argmax fast path. Mirrors forward(input_ids, start_pos)
// through the layers + final_norm + lm_head into the persistent logits buffer,
// then dispatches a single-WG argmax kernel and reads back ONE int32 token id.
// Replaces the 200 KB blocking host-side fp16 readback (~355 ms/step on
// Adreno 620) with a 4-byte readback. Used by generate() when temperature == 0.
int32_t Model::forward_argmax_greedy(const std::vector<int32_t>& input_ids, int start_pos) {
    NNOPT_CHECKPOINT("forward_argmax_greedy() started");
    const int seq_len = (int)input_ids.size();
    if (seq_len <= 0) {
        NNOPT_ERROR("forward_argmax_greedy(): empty input_ids");
        return -1;
    }
    if (!ensure_argmax_resources_()) return -1;

    cl_command_queue queue = cl_ctx_.queue();
    cl_context ctx = cl_ctx_.context();
    using clk = std::chrono::steady_clock;
    auto t_fw_start = clk::now();

    cl_int err = CL_SUCCESS;
    cl_mem ids_buf = clCreateBuffer(ctx, CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR,
        (size_t)seq_len * sizeof(int32_t), (void*)input_ids.data(), &err);
    if (err != CL_SUCCESS || !ids_buf) {
        NNOPT_ERROR_FMT("forward_argmax_greedy(): clCreateBuffer ids_buf failed (%d)", err);
        return -1;
    }

    cl_mem hidden = embedding_->forward(queue, ids_buf, seq_len, start_pos);
    clReleaseMemObject(ids_buf);
    if (!hidden) { NNOPT_ERROR("forward_argmax_greedy(): embedding forward failed"); return -1; }

    for (int i = 0; i < MODEL_CONFIG::NUM_LAYERS; i++) {
        cl_mem residual0 = hidden;

        cl_mem normed1 = input_layernorm_[i]->forward(queue, residual0, seq_len);
        if (!normed1) { NNOPT_ERROR_FMT("forward_argmax_greedy(): input_layernorm failed at layer %d", i); clReleaseMemObject(residual0); return -1; }

        cl_mem attn_out = attn_[i]->forward(queue, normed1, seq_len, start_pos,
                                            embedding_->rope_cos(), embedding_->rope_sin());
        clReleaseMemObject(normed1);
        if (!attn_out) { NNOPT_ERROR_FMT("forward_argmax_greedy(): attn failed at layer %d", i); clReleaseMemObject(residual0); return -1; }

        static cl_program utils_program = nullptr;
        if (!utils_program) {
            utils_program = cl_ctx_.build_program_from_file("kernels/utils.cl");
            if (!utils_program) {
                NNOPT_ERROR("forward_argmax_greedy(): build kernels/utils.cl failed");
                clReleaseMemObject(attn_out); clReleaseMemObject(residual0); return -1;
            }
        }
        if (!element_add_inplace(queue, utils_program, residual0, attn_out,
                                 (size_t)seq_len * MODEL_CONFIG::HIDDEN_SIZE,
                                 MODEL_CONFIG::RESIDUAL_MULTIPLIER)) {
            NNOPT_ERROR_FMT("forward_argmax_greedy(): residual add (attn) failed at layer %d", i);
            clReleaseMemObject(attn_out); clReleaseMemObject(residual0); return -1;
        }
        clReleaseMemObject(attn_out);
        cl_mem residual1 = residual0;

        cl_mem normed2 = post_attention_layernorm_[i]->forward(queue, residual1, seq_len);
        if (!normed2) { NNOPT_ERROR_FMT("forward_argmax_greedy(): post_attention_layernorm failed at layer %d", i); clReleaseMemObject(residual1); return -1; }

        cl_mem new_hidden = shared_mlp_[i]->forward(queue, normed2, residual1,
                                                    MODEL_CONFIG::RESIDUAL_MULTIPLIER, seq_len);
        clReleaseMemObject(normed2);
        clReleaseMemObject(residual1);
        if (!new_hidden) { NNOPT_ERROR_FMT("forward_argmax_greedy(): shared_mlp failed at layer %d", i); return -1; }
        hidden = new_hidden;
    }

    auto t_layers_end = clk::now();
    g_fw_layers_ms += std::chrono::duration<double, std::milli>(t_layers_end - t_fw_start).count();

    cl_mem final_hidden = final_norm_->forward(queue, hidden, seq_len);
    clReleaseMemObject(hidden);
    if (!final_hidden) { NNOPT_ERROR("forward_argmax_greedy(): final_norm failed"); return -1; }

    cl_mem wte = weights_.get_buffer("model.embed_tokens.weight");
    if (!wte) { NNOPT_ERROR("forward_argmax_greedy(): missing model.embed_tokens.weight"); clReleaseMemObject(final_hidden); return -1; }

    const size_t logits_elems = (size_t)seq_len * (size_t)MODEL_CONFIG::VOCAB_SIZE;
    auto t_lm_start = clk::now();
    const size_t logits_bytes = logits_elems * sizeof(nnopt_storage_t);
    if (!logits_buf_ || logits_buf_cap_bytes_ < logits_bytes) {
        if (logits_buf_) { clReleaseMemObject(logits_buf_); logits_buf_ = nullptr; }
        size_t new_cap = (logits_buf_cap_bytes_ == 0) ? logits_bytes : (logits_buf_cap_bytes_ * 2);
        while (new_cap < logits_bytes) new_cap *= 2;
        logits_buf_ = clCreateBuffer(ctx, CL_MEM_READ_WRITE, new_cap, nullptr, &err);
        if (err != CL_SUCCESS || !logits_buf_) {
            NNOPT_ERROR_FMT("forward_argmax_greedy(): clCreateBuffer logits failed (%d)", err);
            clReleaseMemObject(final_hidden); logits_buf_cap_bytes_ = 0; return -1;
        }
        logits_buf_cap_bytes_ = new_cap;
    }
    if (!pytorch_linear(queue, seq_len, MODEL_CONFIG::VOCAB_SIZE, MODEL_CONFIG::HIDDEN_SIZE,
                        final_hidden, wte, logits_buf_)) {
        NNOPT_ERROR("forward_argmax_greedy(): pytorch_linear lm_head failed");
        clReleaseMemObject(final_hidden); return -1;
    }
    clReleaseMemObject(final_hidden);
    auto t_lm_end = clk::now();
    g_fw_lm_head_ms += std::chrono::duration<double, std::milli>(t_lm_end - t_lm_start).count();

    // GPU argmax over the LAST row of logits (decode reads only the last token's logits).
    const int vocab = MODEL_CONFIG::VOCAB_SIZE;
    const int offset_elements = (seq_len - 1) * vocab;
    if (clSetKernelArg(argmax_kernel_, 0, sizeof(cl_mem), &logits_buf_) != CL_SUCCESS ||
        clSetKernelArg(argmax_kernel_, 1, sizeof(cl_mem), &argmax_result_) != CL_SUCCESS ||
        clSetKernelArg(argmax_kernel_, 2, sizeof(int),    &vocab) != CL_SUCCESS ||
        clSetKernelArg(argmax_kernel_, 3, sizeof(int),    &offset_elements) != CL_SUCCESS) {
        NNOPT_ERROR("forward_argmax_greedy(): argmax setKernelArg failed");
        return -1;
    }
    {
        const size_t lws = 256;
        const size_t gws = 256;
        err = clEnqueueNDRangeKernel(queue, argmax_kernel_, 1, nullptr, &gws, &lws, 0, nullptr,
                                     KernelProfiler::event_for("argmax_fp16"));
        if (err != CL_SUCCESS) {
            NNOPT_ERROR_FMT("forward_argmax_greedy(): argmax dispatch failed (%d)", err);
            return -1;
        }
    }

    int32_t result = -1;
    auto t_read_start = clk::now();
    err = clEnqueueReadBuffer(queue, argmax_result_, CL_TRUE, 0,
                              sizeof(int32_t), &result, 0, nullptr, nullptr);
    auto t_read_end = clk::now();
    g_fw_logits_read_ms += std::chrono::duration<double, std::milli>(t_read_end - t_read_start).count();
    if (err != CL_SUCCESS) {
        NNOPT_ERROR_FMT("forward_argmax_greedy(): argmax_result read failed (%d)", err);
        return -1;
    }
    g_fw_calls++;
    NNOPT_CHECKPOINT("forward_argmax_greedy() complete");
    return result;
}

// Step #3: chained-decode greedy. Embedding input is argmax_result_ (the
// device-side int32 buffer written by the prior forward_argmax_greedy call).
// Skips the host→device clCreateBuffer(COPY_HOST_PTR) and the per-iter
// blocking 4-byte read inside forward; instead issues async read of
// argmax_result_ into *host_slot with *out_event so the caller can overlap
// GPU work with host enqueue of the next iter (OpenELM Step 9 pattern, +1.38×).
bool Model::forward_argmax_greedy_chained(int start_pos,
                                          int32_t* host_slot,
                                          cl_event* out_event) {
    NNOPT_CHECKPOINT("forward_argmax_greedy_chained() started");
    if (!ensure_argmax_resources_()) return false;
    if (!argmax_result_) { NNOPT_ERROR("chained: argmax_result_ not allocated (call forward_argmax_greedy first)"); return false; }

    cl_command_queue queue = cl_ctx_.queue();
    cl_context ctx = cl_ctx_.context();
    using clk = std::chrono::steady_clock;
    auto t_fw_start = clk::now();
    cl_int err = CL_SUCCESS;
    const int seq_len = 1;

    // Embedding reads token id directly from argmax_result_ (which holds the
    // last token's id from the prior call). Granite's embedding kernel
    // signature already takes a cl_mem token_ids — no API change needed.
    cl_mem hidden = embedding_->forward(queue, argmax_result_, seq_len, start_pos);
    if (!hidden) { NNOPT_ERROR("chained: embedding forward failed"); return false; }

    // Step "layer-skip" experiment: env-controlled decode-only layer count.
    // NNOPT_DECODE_LAYERS=N (1..28) skips layers [N, 28). Prefill always uses
    // all 28. Reduces per-token weight reads linearly — true fp16 ceiling
    // break, with a quality cost proportional to skipped layers.
    static const int s_decode_layers = []() {
        const char* env = std::getenv("NNOPT_DECODE_LAYERS");
        if (!env || !*env) return MODEL_CONFIG::NUM_LAYERS;
        int v = std::atoi(env);
        if (v < 1) v = 1;
        if (v > MODEL_CONFIG::NUM_LAYERS) v = MODEL_CONFIG::NUM_LAYERS;
        fprintf(stderr, "NNOPT_DECODE_LAYERS=%d (truncating decode to %d of %d layers)\n",
                v, v, MODEL_CONFIG::NUM_LAYERS);
        return v;
    }();

    for (int i = 0; i < s_decode_layers; i++) {
        cl_mem residual0 = hidden;

        cl_mem normed1 = input_layernorm_[i]->forward(queue, residual0, seq_len);
        if (!normed1) { NNOPT_ERROR_FMT("chained: input_layernorm failed at layer %d", i); clReleaseMemObject(residual0); return false; }

        cl_mem attn_out = attn_[i]->forward(queue, normed1, seq_len, start_pos,
                                            embedding_->rope_cos(), embedding_->rope_sin());
        clReleaseMemObject(normed1);
        if (!attn_out) { NNOPT_ERROR_FMT("chained: attn failed at layer %d", i); clReleaseMemObject(residual0); return false; }

        static cl_program utils_program = nullptr;
        if (!utils_program) {
            utils_program = cl_ctx_.build_program_from_file("kernels/utils.cl");
            if (!utils_program) { NNOPT_ERROR("chained: build utils.cl failed"); clReleaseMemObject(attn_out); clReleaseMemObject(residual0); return false; }
        }
        if (!element_add_inplace(queue, utils_program, residual0, attn_out,
                                 (size_t)seq_len * MODEL_CONFIG::HIDDEN_SIZE,
                                 MODEL_CONFIG::RESIDUAL_MULTIPLIER)) {
            NNOPT_ERROR_FMT("chained: residual add (attn) failed at layer %d", i);
            clReleaseMemObject(attn_out); clReleaseMemObject(residual0); return false;
        }
        clReleaseMemObject(attn_out);
        cl_mem residual1 = residual0;

        cl_mem normed2 = post_attention_layernorm_[i]->forward(queue, residual1, seq_len);
        if (!normed2) { NNOPT_ERROR_FMT("chained: post_attention_layernorm failed at layer %d", i); clReleaseMemObject(residual1); return false; }

        cl_mem new_hidden = shared_mlp_[i]->forward(queue, normed2, residual1,
                                                    MODEL_CONFIG::RESIDUAL_MULTIPLIER, seq_len);
        clReleaseMemObject(normed2);
        clReleaseMemObject(residual1);
        if (!new_hidden) { NNOPT_ERROR_FMT("chained: shared_mlp failed at layer %d", i); return false; }
        hidden = new_hidden;
    }
    auto t_layers_end = clk::now();
    g_fw_layers_ms += std::chrono::duration<double, std::milli>(t_layers_end - t_fw_start).count();

    cl_mem final_hidden = final_norm_->forward(queue, hidden, seq_len);
    clReleaseMemObject(hidden);
    if (!final_hidden) { NNOPT_ERROR("chained: final_norm failed"); return false; }

    cl_mem wte = weights_.get_buffer("model.embed_tokens.weight");
    if (!wte) { NNOPT_ERROR("chained: missing model.embed_tokens.weight"); clReleaseMemObject(final_hidden); return false; }

    const size_t logits_elems = (size_t)seq_len * (size_t)MODEL_CONFIG::VOCAB_SIZE;
    auto t_lm_start = clk::now();
    const size_t logits_bytes = logits_elems * sizeof(nnopt_storage_t);
    if (!logits_buf_ || logits_buf_cap_bytes_ < logits_bytes) {
        if (logits_buf_) { clReleaseMemObject(logits_buf_); logits_buf_ = nullptr; }
        size_t new_cap = (logits_buf_cap_bytes_ == 0) ? logits_bytes : (logits_buf_cap_bytes_ * 2);
        while (new_cap < logits_bytes) new_cap *= 2;
        logits_buf_ = clCreateBuffer(ctx, CL_MEM_READ_WRITE, new_cap, nullptr, &err);
        if (err != CL_SUCCESS || !logits_buf_) {
            NNOPT_ERROR_FMT("chained: clCreateBuffer logits failed (%d)", err);
            clReleaseMemObject(final_hidden); logits_buf_cap_bytes_ = 0; return false;
        }
        logits_buf_cap_bytes_ = new_cap;
    }
    if (!pytorch_linear(queue, seq_len, MODEL_CONFIG::VOCAB_SIZE, MODEL_CONFIG::HIDDEN_SIZE,
                        final_hidden, wte, logits_buf_)) {
        NNOPT_ERROR("chained: pytorch_linear lm_head failed");
        clReleaseMemObject(final_hidden); return false;
    }
    clReleaseMemObject(final_hidden);
    auto t_lm_end = clk::now();
    g_fw_lm_head_ms += std::chrono::duration<double, std::milli>(t_lm_end - t_lm_start).count();

    const int vocab = MODEL_CONFIG::VOCAB_SIZE;
    const int offset_elements = 0;  // seq_len == 1; row 0 is the only row
    if (clSetKernelArg(argmax_kernel_, 0, sizeof(cl_mem), &logits_buf_) != CL_SUCCESS ||
        clSetKernelArg(argmax_kernel_, 1, sizeof(cl_mem), &argmax_result_) != CL_SUCCESS ||
        clSetKernelArg(argmax_kernel_, 2, sizeof(int),    &vocab) != CL_SUCCESS ||
        clSetKernelArg(argmax_kernel_, 3, sizeof(int),    &offset_elements) != CL_SUCCESS) {
        NNOPT_ERROR("chained: argmax setKernelArg failed");
        return false;
    }
    {
        const size_t lws = 256;
        const size_t gws = 256;
        err = clEnqueueNDRangeKernel(queue, argmax_kernel_, 1, nullptr, &gws, &lws, 0, nullptr,
                                     KernelProfiler::event_for("argmax_fp16"));
        if (err != CL_SUCCESS) { NNOPT_ERROR_FMT("chained: argmax dispatch failed (%d)", err); return false; }
    }

    // Async readback of argmax_result_ into the caller's host slot. Signals
    // *out_event when the read completes — caller waits on it RIGHT BEFORE
    // using the token id (e.g., the on_token callback). In the meantime the
    // caller can enqueue the NEXT iter's chained forward on this queue, which
    // (a) doesn't depend on host_slot at all (it consumes argmax_result_
    // directly device-side via embedding) and (b) will be in flight on the
    // GPU before we block on this readback.
    err = clEnqueueReadBuffer(queue, argmax_result_, CL_FALSE, 0,
                              sizeof(int32_t), host_slot, 0, nullptr, out_event);
    if (err != CL_SUCCESS) {
        NNOPT_ERROR_FMT("chained: async argmax_result_ read failed (%d)", err);
        return false;
    }
    clFlush(queue);  // kick the GPU work + the async read into flight
    g_fw_calls++;
    NNOPT_CHECKPOINT("forward_argmax_greedy_chained() complete");
    return true;
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

    // Step #1: greedy fast-path branch. When temperature == 0 we use
    // forward_argmax_greedy() which dispatches a GPU argmax over the logits
    // buffer and reads back ONE int32 token id (vs the 200 KB host-side
    // map+convert that the regular forward() does on every step). The greedy
    // sampler is bit-exact with this argmax so token IDs match.
    const bool greedy_path = (sampler_config.temperature <= 0.0f);

    using clk = std::chrono::steady_clock;
    auto t_now = []() { return clk::now(); };
    auto ms_between = [](clk::time_point a, clk::time_point b) {
        return std::chrono::duration<double, std::milli>(b - a).count();
    };
    double prefill_forward_ms = 0.0;
    double prefill_sample_ms  = 0.0;
    double decode_forward_ms  = 0.0;
    double decode_sample_ms   = 0.0;
    double decode_ontoken_ms  = 0.0;
    int    decode_steps       = 0;

    int next_token = -1;
    std::vector<int32_t> generated;

    auto t0 = t_now();
    if (greedy_path) {
        // Prefill straight to argmax — no host logits readback.
        next_token = forward_argmax_greedy(prompt_ids, /*start_pos=*/0);
        prefill_forward_ms = ms_between(t0, t_now());
        if (next_token < 0) {
            NNOPT_ERROR("prefill forward_argmax_greedy() returned -1");
            return ids;
        }
        // sample_total stays at 0 for greedy (argmax is on GPU; not in host sampler).
    } else {
        auto logits = forward(prompt_ids, /*start_pos=*/0);
        prefill_forward_ms = ms_between(t0, t_now());
        if (logits.empty()) {
            NNOPT_ERROR("prefill forward() returned empty logits");
            return ids;
        }
        auto t1 = t_now();
        next_token = sampler.sample(logits, generated);
        prefill_sample_ms = ms_between(t1, t_now());
    }

    ids.push_back(next_token);
    generated.push_back(next_token);
    if (on_token) on_token(next_token);
    NNOPT_BENCH_FIRST_TOKEN();
    if (sampler_config.eos_token_id >= 0 && next_token == sampler_config.eos_token_id) {
        NNOPT_CHECKPOINT("generate() complete (eos at first token)");
        return ids;
    }

    if (greedy_path) {
        // Step #3 chained-decode loop. Pattern (OpenELM-style):
        //   iter i: enqueue forward_chained → async read argmax_result_ into slot[i%2]
        //   iter i (continued): wait on slot[(i-1)%2] from prior iter, callback with that token
        //   final: drain the last slot
        // This way the GPU is enqueued for iter N+1 while the host is still
        // waiting for iter N's argmax readback. Wall is no longer paced by
        // the per-iter CL_TRUE blocking call inside forward.
        cl_event read_events[2] = { nullptr, nullptr };
        int32_t  host_slots[2]  = { 0, 0 };
        int32_t  pending_token  = -1;  // token whose readback is pending (the "i-1" iter)
        bool     have_pending   = false;
        int      pending_slot   = -1;

        for (int i = 1; i < max_new_tokens; i++) {
            const int start_pos = (int)prompt_ids.size() + (i - 1);
            const int slot = i % 2;
            // Release any old event in this slot before reuse.
            if (read_events[slot]) { clReleaseEvent(read_events[slot]); read_events[slot] = nullptr; }

            auto t_fw = t_now();
            cl_event evt = nullptr;
            if (!forward_argmax_greedy_chained(start_pos, &host_slots[slot], &evt)) {
                NNOPT_ERROR("decode forward_argmax_greedy_chained returned false");
                break;
            }
            read_events[slot] = evt;
            decode_forward_ms += ms_between(t_fw, t_now());

            // Drain the prior pending iter NOW (after we've enqueued this one,
            // so GPU is already busy on the NEXT lm_head/argmax).
            if (have_pending) {
                cl_event w = read_events[pending_slot];
                if (w) clWaitForEvents(1, &w);
                int32_t prev_token = host_slots[pending_slot];
                ids.push_back(prev_token);
                generated.push_back(prev_token);
                auto t_cb = t_now();
                if (on_token) on_token(prev_token);
                decode_ontoken_ms += ms_between(t_cb, t_now());
                decode_steps++;
                if (sampler_config.eos_token_id >= 0 && prev_token == sampler_config.eos_token_id) {
                    have_pending = false;
                    break;
                }
            }
            pending_slot = slot;
            pending_token = -1;  // value lives in host_slots[slot] once event fires
            have_pending = true;
        }
        // Drain final pending readback.
        if (have_pending) {
            cl_event w = read_events[pending_slot];
            if (w) clWaitForEvents(1, &w);
            int32_t prev_token = host_slots[pending_slot];
            ids.push_back(prev_token);
            generated.push_back(prev_token);
            auto t_cb = t_now();
            if (on_token) on_token(prev_token);
            decode_ontoken_ms += ms_between(t_cb, t_now());
            decode_steps++;
            (void)pending_token;
        }
        if (read_events[0]) clReleaseEvent(read_events[0]);
        if (read_events[1]) clReleaseEvent(read_events[1]);
        // For the eos check below in the non-greedy path's loop, leave
        // next_token at the last decoded value.
        if (!ids.empty()) next_token = ids.back();
    } else {
        for (int i = 1; i < max_new_tokens; i++) {
            const int start_pos = (int)prompt_ids.size() + (i - 1);
            std::vector<int32_t> single = { next_token };

            auto t_fw = t_now();
            auto logits = forward(single, start_pos);
            decode_forward_ms += ms_between(t_fw, t_now());
            if (logits.empty()) { NNOPT_ERROR("decode forward() returned empty logits"); break; }
            auto t_sm = t_now();
            next_token = sampler.sample(logits, generated);
            decode_sample_ms += ms_between(t_sm, t_now());

            ids.push_back(next_token);
            generated.push_back(next_token);
            auto t_cb = t_now();
            if (on_token) on_token(next_token);
            decode_ontoken_ms += ms_between(t_cb, t_now());
            decode_steps++;

            if (sampler_config.eos_token_id >= 0 && next_token == sampler_config.eos_token_id) break;
        }
    }

    // Print host-side breakdown so we can see where wall time goes vs the
    // GPU kernel profile (NNOPT_KERNEL_PROFILE=1). Always on — cost is one
    // chrono::now() pair per step (negligible).
    fprintf(stderr,
            "\n=== HOST WALL BREAKDOWN ===\n"
            "  prefill: forward=%.1f ms, sample=%.1f ms\n"
            "  decode (n=%d): forward_total=%.1f ms (%.2f ms/step), "
            "sample_total=%.1f ms (%.2f ms/step), "
            "on_token_total=%.1f ms (%.2f ms/step)\n"
            "  forward() internals (all %d calls combined):\n"
            "    layers_loop=%.1f ms   lm_head=%.1f ms   "
            "logits_read=%.1f ms (blocking)   fp16_to_fp32_cvt=%.1f ms\n"
            "===========================\n",
            prefill_forward_ms, prefill_sample_ms,
            decode_steps,
            decode_forward_ms, decode_steps ? decode_forward_ms / decode_steps : 0.0,
            decode_sample_ms,  decode_steps ? decode_sample_ms  / decode_steps : 0.0,
            decode_ontoken_ms, decode_steps ? decode_ontoken_ms / decode_steps : 0.0,
            g_fw_calls,
            g_fw_layers_ms, g_fw_lm_head_ms,
            g_fw_logits_read_ms, g_fw_logits_cvt_ms);

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
