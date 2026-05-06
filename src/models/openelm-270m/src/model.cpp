// Auto-generated model implementation stub for apple/OpenELM-270M
// This will be filled in by the agent after layer implementation.

#include "model.h"
#include "sampler.h"
#include "debug_utils.h"
#include "benchmark.h"
#include "model_config.h"
#include "utils.h"
#include <iostream>
#include <memory>
#include <string>

Model::Model(OpenCLContext& cl_ctx, Weights& weights)
    : cl_ctx_(cl_ctx), weights_(weights) {
    // Constructor: refs only. NO kernel/program work here. All per-layer
    // initialize() calls live in Model::initialize() so failures can return
    // false to main() (a void ctor cannot signal failure).
    NNOPT_CHECKPOINT("Model constructor");
}

Model::~Model() {
    if (argmax_kernel_)  clReleaseKernel(argmax_kernel_);
    if (argmax_program_) clReleaseProgram(argmax_program_);
    if (argmax_result_)  clReleaseMemObject(argmax_result_);
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
            NNOPT_ERROR("Failed to build kernels/argmax.cl");
            return false;
        }
    }
    if (!argmax_kernel_) {
        cl_int err = CL_SUCCESS;
        argmax_kernel_ = clCreateKernel(argmax_program_, "argmax_fp16", &err);
        if (err != CL_SUCCESS || !argmax_kernel_) {
            NNOPT_ERROR_FMT("clCreateKernel(argmax_fp16) failed: %d", (int)err);
            return false;
        }
    }
    if (!argmax_result_) {
        cl_int err = CL_SUCCESS;
        argmax_result_ = clCreateBuffer(cl_ctx_.context(), CL_MEM_READ_WRITE,
                                        sizeof(int32_t), nullptr, &err);
        if (err != CL_SUCCESS || !argmax_result_) {
            NNOPT_ERROR_FMT("alloc argmax_result_ failed: %d", (int)err);
            return false;
        }
    }
    return true;
}

static std::string dump_name(const char* stem, int layer_idx) {
    return std::string("block") + std::to_string(layer_idx) + "_sub_" + stem;
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
    NNOPT_CHECKPOINT("Model::initialize() started");

    embedding_ = std::make_unique<Embedding>(cl_ctx_, weights_);
    if (!embedding_->initialize()) {
        NNOPT_ERROR("embedding.initialize() FAILED");
        return false;
    }
    NNOPT_LAYER_INIT("embedding");

    for (int i = 0; i < MODEL_CONFIG::NUM_TRANSFORMER_LAYERS; i++) {
        pre_attn_norm_[i] = std::make_unique<LayerNorm>(cl_ctx_, weights_, /*layer_idx=*/i,
            /*weight_key=*/"transformer.layers." + std::to_string(i) + ".attn_norm.weight");
        if (!pre_attn_norm_[i]->initialize()) {
            NNOPT_ERROR_FMT("pre_attn_norm_[%d].initialize() FAILED", i);
            return false;
        }
        NNOPT_LAYER_INIT_FMT("block%d_sub_pre_attn_norm", i);

        attn_[i] = std::make_unique<Attention>(cl_ctx_, weights_, /*layer_idx=*/i);
        if (!attn_[i]->initialize()) {
            NNOPT_ERROR_FMT("attn_[%d].initialize() FAILED", i);
            return false;
        }
        NNOPT_LAYER_INIT_FMT("block%d_sub_attn", i);

        post_attn_norm_[i] = std::make_unique<LayerNorm>(cl_ctx_, weights_, /*layer_idx=*/i,
            /*weight_key=*/"transformer.layers." + std::to_string(i) + ".ffn_norm.weight");
        if (!post_attn_norm_[i]->initialize()) {
            NNOPT_ERROR_FMT("post_attn_norm_[%d].initialize() FAILED", i);
            return false;
        }
        NNOPT_LAYER_INIT_FMT("block%d_sub_post_attn_norm", i);

        mlp_[i] = std::make_unique<Mlp>(cl_ctx_, weights_, /*layer_idx=*/i);
        if (!mlp_[i]->initialize()) {
            NNOPT_ERROR_FMT("mlp_[%d].initialize() FAILED", i);
            return false;
        }
        NNOPT_LAYER_INIT_FMT("block%d_sub_mlp", i);
    }

    final_norm_ = std::make_unique<LayerNorm>(cl_ctx_, weights_, /*layer_idx=*/-1,
        /*weight_key=*/"transformer.norm.weight");
    if (!final_norm_->initialize()) {
        NNOPT_ERROR("final_norm.initialize() FAILED");
        return false;
    }
    NNOPT_LAYER_INIT("final_norm");

    // TODO: Replace this stub with real per-layer initialize() calls. The
    // shape varies per architecture — read (model metadata)
    // and the per-class JSON files to know what to instantiate. Example:
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
// Reference: model_info/modeling_openelm.py:620-747  (OpenELMModel.forward)
//   This is the BACKBONE dispatch graph. The C++ Model::forward below mirrors
//   this PyTorch method line-for-line.
// Reference: model_info/modeling_openelm.py:836-910  (OpenELMForCausalLM.forward)
//   The HEAD wraps the backbone — `outputs = self.transformer(input_ids); logits =
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
    NNOPT_CHECKPOINT("forward() started");
    const int seq_len = (int)input_ids.size();
    if (seq_len <= 0) {
        NNOPT_ERROR("forward(): empty input_ids");
        return {};
    }

    cl_command_queue queue = cl_ctx_.queue();

    // Embedding
    cl_mem hidden = embedding_->forward(queue, input_ids.data(), seq_len, start_pos);
    if (!hidden) {
        NNOPT_ERROR("embedding->forward returned nullptr");
        return {};
    }

    // Transformer blocks
    for (int i = 0; i < MODEL_CONFIG::NUM_TRANSFORMER_LAYERS; i++) {
        cl_mem residual = hidden;

        cl_mem normed = pre_attn_norm_[i]->forward(queue, residual, seq_len);
        if (!normed) {
            NNOPT_ERROR_FMT("pre_attn_norm forward nullptr at layer %d", i);
            clReleaseMemObject(residual);
            return {};
        }

        // Attention returns a BORROWED handle (owned by attn_[i]). DO NOT release.
        cl_mem attn_out = attn_[i]->forward(queue, normed, seq_len, start_pos);
        clReleaseMemObject(normed);
        if (!attn_out) {
            NNOPT_ERROR_FMT("attn forward nullptr at layer %d", i);
            clReleaseMemObject(residual);
            return {};
        }

        cl_mem after_attn = element_add(queue, cl_ctx_.utils_program(), residual, attn_out,
            (size_t)seq_len * MODEL_CONFIG::HIDDEN_SIZE);
        clReleaseMemObject(residual);
        // attn_out NOT released — borrowed.
        if (!after_attn) {
            NNOPT_ERROR_FMT("residual add failed at layer %d", i);
            return {};
        }

        cl_mem normed2 = post_attn_norm_[i]->forward(queue, after_attn, seq_len);
        if (!normed2) {
            NNOPT_ERROR_FMT("post_attn_norm forward nullptr at layer %d", i);
            clReleaseMemObject(after_attn);
            return {};
        }

        // Mlp returns a BORROWED handle (owned by mlp_[i]). DO NOT release.
        cl_mem mlp_out = mlp_[i]->forward(queue, normed2, seq_len);
        clReleaseMemObject(normed2);
        if (!mlp_out) {
            NNOPT_ERROR_FMT("mlp forward nullptr at layer %d", i);
            clReleaseMemObject(after_attn);
            return {};
        }

        cl_mem after_mlp = element_add(queue, cl_ctx_.utils_program(), after_attn, mlp_out,
            (size_t)seq_len * MODEL_CONFIG::HIDDEN_SIZE);
        clReleaseMemObject(after_attn);
        // mlp_out NOT released — borrowed.
        if (!after_mlp) {
            NNOPT_ERROR_FMT("mlp residual add failed at layer %d", i);
            return {};
        }

        hidden = after_mlp;
    }

    // Final norm
    cl_mem final_hidden = final_norm_->forward(queue, hidden, seq_len);
    clReleaseMemObject(hidden);
    if (!final_hidden) {
        NNOPT_ERROR("final_norm forward nullptr");
        return {};
    }

    // LM head (tied embeddings). Use embedding weights as output projection.
    // W shape is [vocab, hidden] in our exported weights.
    cl_mem lm_w = weights_.get_buffer("transformer.token_embeddings.weight");
    if (!lm_w) {
        NNOPT_ERROR("missing transformer.token_embeddings.weight");
        clReleaseMemObject(final_hidden);
        return {};
    }

    cl_context ctx = cl_ctx_.context();
    cl_int err = CL_SUCCESS;
    cl_mem logits = clCreateBuffer(ctx, CL_MEM_READ_WRITE,
        (size_t)seq_len * MODEL_CONFIG::VOCAB_SIZE * sizeof(nnopt_storage_t), nullptr, &err);
    if (err != CL_SUCCESS || !logits) {
        NNOPT_ERROR_FMT("clCreateBuffer logits failed (%d)", err);
        clReleaseMemObject(final_hidden);
        return {};
    }

    if (!pytorch_linear(queue, seq_len, MODEL_CONFIG::VOCAB_SIZE, MODEL_CONFIG::HIDDEN_SIZE,
            final_hidden, lm_w, logits)) {
        NNOPT_ERROR("pytorch_linear lm_head failed");
        clReleaseMemObject(final_hidden);
        clReleaseMemObject(logits);
        return {};
    }
    clReleaseMemObject(final_hidden);

    // Read back last-token logits only
    const size_t vocab = (size_t)MODEL_CONFIG::VOCAB_SIZE;
    std::vector<float> host_logits(vocab);
    const size_t offset_bytes = (size_t)(seq_len - 1) * vocab * sizeof(nnopt_storage_t);

#ifdef NNOPT_USE_FP16
    std::vector<nnopt_storage_t> tmp(vocab);
    err = clEnqueueReadBuffer(queue, logits, CL_TRUE, offset_bytes,
        vocab * sizeof(nnopt_storage_t), tmp.data(), 0, nullptr, nullptr);
    if (err != CL_SUCCESS) {
        NNOPT_ERROR_FMT("Read logits failed (%d)", err);
        clReleaseMemObject(logits);
        return {};
    }
    for (size_t i = 0; i < vocab; i++) host_logits[i] = nnopt_f16_to_f32(tmp[i]);
#else
    err = clEnqueueReadBuffer(queue, logits, CL_TRUE, offset_bytes,
        vocab * sizeof(float), host_logits.data(), 0, nullptr, nullptr);
    if (err != CL_SUCCESS) {
        NNOPT_ERROR_FMT("Read logits failed (%d)", err);
        clReleaseMemObject(logits);
        return {};
    }
#endif

    clReleaseMemObject(logits);
    NNOPT_CHECKPOINT("forward() complete");
    return host_logits;
}

// Step 3: greedy fast-path. Mirrors forward(input_ids, start_pos) up to the
// logits-on-device buffer, then dispatches a single-WG argmax kernel and
// reads back ONE int32 (vs. the ~100 KB host-side fp16 readback).
int32_t Model::forward_argmax_greedy(const std::vector<int32_t>& input_ids, int start_pos) {
    NNOPT_CHECKPOINT("forward_argmax_greedy() started");
    const int seq_len = (int)input_ids.size();
    if (seq_len <= 0) {
        NNOPT_ERROR("forward_argmax_greedy(): empty input_ids");
        return -1;
    }

    if (!ensure_argmax_resources_()) return -1;

    cl_command_queue queue = cl_ctx_.queue();

    // Embedding
    cl_mem hidden = embedding_->forward(queue, input_ids.data(), seq_len, start_pos);
    if (!hidden) { NNOPT_ERROR("embedding->forward returned nullptr"); return -1; }

    // Transformer blocks
    for (int i = 0; i < MODEL_CONFIG::NUM_TRANSFORMER_LAYERS; i++) {
        cl_mem residual = hidden;

        cl_mem normed = pre_attn_norm_[i]->forward(queue, residual, seq_len);
        if (!normed) {
            NNOPT_ERROR_FMT("pre_attn_norm forward nullptr at layer %d", i);
            clReleaseMemObject(residual);
            return -1;
        }

        cl_mem attn_out = attn_[i]->forward(queue, normed, seq_len, start_pos);
        clReleaseMemObject(normed);
        if (!attn_out) {
            NNOPT_ERROR_FMT("attn forward nullptr at layer %d", i);
            clReleaseMemObject(residual);
            return -1;
        }

        cl_mem after_attn = element_add(queue, cl_ctx_.utils_program(), residual, attn_out,
            (size_t)seq_len * MODEL_CONFIG::HIDDEN_SIZE);
        clReleaseMemObject(residual);
        if (!after_attn) { NNOPT_ERROR_FMT("residual add failed at layer %d", i); return -1; }

        cl_mem normed2 = post_attn_norm_[i]->forward(queue, after_attn, seq_len);
        if (!normed2) {
            NNOPT_ERROR_FMT("post_attn_norm forward nullptr at layer %d", i);
            clReleaseMemObject(after_attn);
            return -1;
        }

        cl_mem mlp_out = mlp_[i]->forward(queue, normed2, seq_len);
        clReleaseMemObject(normed2);
        if (!mlp_out) {
            NNOPT_ERROR_FMT("mlp forward nullptr at layer %d", i);
            clReleaseMemObject(after_attn);
            return -1;
        }

        cl_mem after_mlp = element_add(queue, cl_ctx_.utils_program(), after_attn, mlp_out,
            (size_t)seq_len * MODEL_CONFIG::HIDDEN_SIZE);
        clReleaseMemObject(after_attn);
        if (!after_mlp) { NNOPT_ERROR_FMT("mlp residual add failed at layer %d", i); return -1; }

        hidden = after_mlp;
    }

    // Final norm
    cl_mem final_hidden = final_norm_->forward(queue, hidden, seq_len);
    clReleaseMemObject(hidden);
    if (!final_hidden) { NNOPT_ERROR("final_norm forward nullptr"); return -1; }

    // LM head (tied embeddings)
    cl_mem lm_w = weights_.get_buffer("transformer.token_embeddings.weight");
    if (!lm_w) {
        NNOPT_ERROR("missing transformer.token_embeddings.weight");
        clReleaseMemObject(final_hidden);
        return -1;
    }

    cl_int err = CL_SUCCESS;
    cl_mem logits = clCreateBuffer(cl_ctx_.context(), CL_MEM_READ_WRITE,
        (size_t)seq_len * MODEL_CONFIG::VOCAB_SIZE * sizeof(nnopt_storage_t), nullptr, &err);
    if (err != CL_SUCCESS || !logits) {
        NNOPT_ERROR_FMT("clCreateBuffer logits failed (%d)", err);
        clReleaseMemObject(final_hidden);
        return -1;
    }

    if (!pytorch_linear(queue, seq_len, MODEL_CONFIG::VOCAB_SIZE, MODEL_CONFIG::HIDDEN_SIZE,
            final_hidden, lm_w, logits)) {
        NNOPT_ERROR("pytorch_linear lm_head failed");
        clReleaseMemObject(final_hidden);
        clReleaseMemObject(logits);
        return -1;
    }
    clReleaseMemObject(final_hidden);

    // GPU argmax over the LAST token's logits row only. Pass offset_elements
    // instead of clCreateSubBuffer (Adreno alignment quirks).
    const int vocab = MODEL_CONFIG::VOCAB_SIZE;
    const int offset_elements = (seq_len - 1) * vocab;

    if (clSetKernelArg(argmax_kernel_, 0, sizeof(cl_mem), &logits) != CL_SUCCESS ||
        clSetKernelArg(argmax_kernel_, 1, sizeof(cl_mem), &argmax_result_) != CL_SUCCESS ||
        clSetKernelArg(argmax_kernel_, 2, sizeof(int),    &vocab) != CL_SUCCESS ||
        clSetKernelArg(argmax_kernel_, 3, sizeof(int),    &offset_elements) != CL_SUCCESS) {
        NNOPT_ERROR("argmax setKernelArg failed");
        clReleaseMemObject(logits);
        return -1;
    }
    {
        const size_t lws = 256;
        const size_t gws = 256;
        err = clEnqueueNDRangeKernel(queue, argmax_kernel_, 1, nullptr, &gws, &lws, 0, nullptr, nullptr);
        if (err != CL_SUCCESS) {
            NNOPT_ERROR_FMT("argmax dispatch failed: %d", (int)err);
            clReleaseMemObject(logits);
            return -1;
        }
    }

    int32_t result = -1;
    err = clEnqueueReadBuffer(queue, argmax_result_, CL_TRUE, 0,
                              sizeof(int32_t), &result, 0, nullptr, nullptr);
    clReleaseMemObject(logits);
    if (err != CL_SUCCESS) {
        NNOPT_ERROR_FMT("Read argmax result failed (%d)", err);
        return -1;
    }
    NNOPT_CHECKPOINT("forward_argmax_greedy() complete");
    return result;
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

    // Step 3: GPU-argmax fast path. Active when sampling is greedy AND no
    // repetition penalty is needed. Skips per-token vocab×fp16 host readback.
    const bool greedy_fast =
        sampler_config.temperature <= 0.0f && sampler_config.repetition_penalty == 1.0f;

    std::vector<int32_t> generated;
    int next_token = -1;

    if (greedy_fast) {
        next_token = forward_argmax_greedy(prompt_ids, /*start_pos=*/0);
        if (next_token < 0) {
            NNOPT_ERROR("prefill forward_argmax_greedy() failed");
            return ids;
        }
    } else {
        auto logits = forward(prompt_ids, /*start_pos=*/0);
        if (logits.empty()) {
            NNOPT_ERROR("prefill forward() returned empty logits");
            return ids;
        }
        next_token = sampler.sample(logits, generated);
    }
    ids.push_back(next_token);
    generated.push_back(next_token);
    if (on_token) on_token(next_token);
    NNOPT_BENCH_FIRST_TOKEN();
    if (sampler_config.eos_token_id >= 0 && next_token == sampler_config.eos_token_id) {
        NNOPT_CHECKPOINT("generate() complete (eos at first token)");
        return ids;
    }

    // Decode loop: each step processes ONE new token.
    for (int i = 1; i < max_new_tokens; i++) {
        const int start_pos = (int)prompt_ids.size() + (i - 1);
        std::vector<int32_t> single = { next_token };

        if (greedy_fast) {
            next_token = forward_argmax_greedy(single, start_pos);
            if (next_token < 0) {
                NNOPT_ERROR("decode forward_argmax_greedy() failed");
                break;
            }
        } else {
            auto logits = forward(single, start_pos);
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
