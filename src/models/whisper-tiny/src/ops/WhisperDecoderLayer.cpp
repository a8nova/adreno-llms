// WhisperDecoderLayer.cpp — shared implementation for ALL 4 WhisperDecoderLayer node(s).
// Reference: model_info/transformers_src/modeling_whisper.py:378-474 WhisperDecoderLayer.forward
//
// Implements (inference path, dropout disabled):
//   residual = x
//   x = self_attn_layer_norm(x)
//   x = self_attn(x)
//   x = residual + x
//
//   if encoder_hidden_states:
//     residual = x
//     x = encoder_attn_layer_norm(x)
//     x = encoder_attn(x, key_value_states=encoder_hidden_states)
//     x = residual + x
//
//   residual = x
//   x = final_layer_norm(x)
//   x = gelu(fc1(x))
//   x = fc2(x)
//   x = residual + x

#include "../opencl_context.h"
#include "../weights.h"
#include "../debug_utils.h"
#include "../model_config.h"
#include "../utils.h"

#include <CL/cl.h>
#include <string>

// Forward declarations for sibling ops we call.
extern "C" cl_mem LayerNorm_forward(OpenCLContext& cl_ctx, Weights& weights, cl_command_queue queue,
                                    cl_mem input, int seq_len, int layer_idx, int start_pos,
                                    cl_mem* k_cache_inout, cl_mem* v_cache_inout, cl_mem encoder_hidden_states,
                                    const char* weight_prefix);
extern "C" cl_mem GELUActivation_forward(OpenCLContext& cl_ctx, Weights& weights, cl_command_queue queue,
                                        cl_mem input, int seq_len, int layer_idx, int start_pos,
                                        cl_mem* k_cache_inout, cl_mem* v_cache_inout, cl_mem encoder_hidden_states,
                                        const char* weight_prefix);
extern "C" cl_mem WhisperSdpaAttention_forward(OpenCLContext& cl_ctx, Weights& weights, cl_command_queue queue,
                                               cl_mem input, int seq_len, int layer_idx, int start_pos,
                                               cl_mem* k_cache_inout, cl_mem* v_cache_inout, cl_mem encoder_hidden_states,
                                               const char* weight_prefix);

// Minimal bias add (y[M,N] += bias[N]) helper using kernels/bias_add.cl.
// Reference: model_info/transformers_src/modeling_whisper.py:210-474 (nn.Linear + bias semantics)
static bool bias_add_rows_inplace(OpenCLContext& cl_ctx, cl_command_queue queue,
                                 cl_mem y, cl_mem bias, int rows, int cols) {
    if (!y || !bias) return false;
    static cl_program s_prog = nullptr;
    static cl_kernel s_k = nullptr;
    if (!s_prog) {
        // PROGRAM-INIT-OK: function-local static cache
        s_prog = cl_ctx.build_program_from_file("kernels/bias_add.cl");
        if (!s_prog) { NNOPT_ERROR("bias_add_rows_inplace: build kernels/bias_add.cl failed"); return false; }
        cl_int kerr = CL_SUCCESS;
        s_k = clCreateKernel(s_prog, "bias_add_rows", &kerr);
        if (kerr != CL_SUCCESS || !s_k) {
            NNOPT_ERROR_FMT("bias_add_rows_inplace: clCreateKernel failed %d", (int)kerr);
            clReleaseProgram(s_prog); s_prog = nullptr;
            return false;
        }
    }

    cl_kernel k = s_k;
    clRetainKernel(k);

    if (!set_arg_checked(k, 0, sizeof(cl_mem), &y, "y")) { clReleaseKernel(k); return false; }
    if (!set_arg_checked(k, 1, sizeof(cl_mem), &bias, "bias")) { clReleaseKernel(k); return false; }
    if (!set_arg_checked(k, 2, sizeof(int), &rows, "rows")) { clReleaseKernel(k); return false; }
    if (!set_arg_checked(k, 3, sizeof(int), &cols, "cols")) { clReleaseKernel(k); return false; }

    const size_t gws = (size_t)rows * (size_t)cols;
    const cl_int err = clEnqueueNDRangeKernel(queue, k, 1, nullptr, &gws, nullptr, 0, nullptr, nullptr);
    clReleaseKernel(k);
    if (err != CL_SUCCESS) {
        NNOPT_ERROR_FMT("bias_add_rows_inplace: dispatch failed %d", (int)err);
        return false;
    }
    return true;
}

extern "C" {
cl_mem WhisperDecoderLayer_forward(
    OpenCLContext& cl_ctx,
    Weights& weights,
    cl_command_queue queue,
    cl_mem input,
    int seq_len,
    int layer_idx,
    int start_pos,
    cl_mem* k_cache_inout,
    cl_mem* v_cache_inout,
    cl_mem encoder_hidden_states,
    const char* weight_prefix)
{
    // Whisper uses caching in attention layers; thread through for decode correctness.

    if (!input) {
        NNOPT_ERROR("WhisperDecoderLayer_forward: input is null");
        return nullptr;
    }
    if (!weight_prefix) {
        NNOPT_ERROR("WhisperDecoderLayer_forward: weight_prefix is null");
        return nullptr;
    }

    const std::string wp(weight_prefix);

    // Weight existence sanity check (actual weight reads happen in the sibling ops).
    // Contracts list these keys; we load a representative subset to fail early if prefix is wrong.
    if (!weights.has_tensor(wp + ".self_attn_layer_norm.weight") ||
        !weights.has_tensor(wp + ".self_attn_layer_norm.bias") ||
        !weights.has_tensor(wp + ".final_layer_norm.weight") ||
        !weights.has_tensor(wp + ".final_layer_norm.bias") ||
        !weights.has_tensor(wp + ".fc1.weight") ||
        !weights.has_tensor(wp + ".fc1.bias") ||
        !weights.has_tensor(wp + ".fc2.weight") ||
        !weights.has_tensor(wp + ".fc2.bias")) {
        NNOPT_ERROR_FMT("WhisperDecoderLayer_forward: missing expected weights for %s", wp.c_str());
        return nullptr;
    }

    const int hidden = (int)MODEL_CONFIG::HIDDEN_SIZE;
    const size_t n = (size_t)seq_len * (size_t)hidden;

    cl_int err = CL_SUCCESS;
    cl_mem x = nullptr;
    cl_mem residual = nullptr;
    cl_mem tmp = nullptr;

    auto cleanup = [&]() -> cl_mem {
        if (x) { clReleaseMemObject(x); x = nullptr; }
        if (residual) { clReleaseMemObject(residual); residual = nullptr; }
        if (tmp) { clReleaseMemObject(tmp); tmp = nullptr; }
        return nullptr;
    };

    // Own a copy of the input.
    x = clCreateBuffer(cl_ctx.context(), CL_MEM_READ_WRITE, n * sizeof(nnopt_storage_t), nullptr, &err);
    if (err != CL_SUCCESS || !x) { NNOPT_ERROR_FMT("clCreateBuffer(x) %d", (int)err); return cleanup(); }
    err = clEnqueueCopyBuffer(queue, input, x, 0, 0, n * sizeof(nnopt_storage_t), 0, nullptr, nullptr);
    if (err != CL_SUCCESS) { NNOPT_ERROR_FMT("copy input->x %d", (int)err); return cleanup(); }

    // utils.cl program for residual adds.
    static cl_program s_utils_prog = nullptr;
    if (!s_utils_prog) {
        // PROGRAM-INIT-OK: function-local static cache
        s_utils_prog = cl_ctx.build_program_from_file("kernels/utils.cl");
        if (!s_utils_prog) { NNOPT_ERROR("WhisperDecoderLayer_forward: build kernels/utils.cl failed"); return cleanup(); }
    }

    // ── Self-attention block ───────────────────────────────
    residual = clCreateBuffer(cl_ctx.context(), CL_MEM_READ_WRITE, n * sizeof(nnopt_storage_t), nullptr, &err);
    if (err != CL_SUCCESS || !residual) { NNOPT_ERROR_FMT("clCreateBuffer(residual) %d", (int)err); return cleanup(); }
    err = clEnqueueCopyBuffer(queue, x, residual, 0, 0, n * sizeof(nnopt_storage_t), 0, nullptr, nullptr);
    if (err != CL_SUCCESS) { NNOPT_ERROR_FMT("copy x->residual %d", (int)err); return cleanup(); }

    tmp = LayerNorm_forward(cl_ctx, weights, queue, x, seq_len, layer_idx, start_pos,
                            k_cache_inout, v_cache_inout, encoder_hidden_states,
                            (wp + ".self_attn_layer_norm").c_str());
    if (!tmp) { NNOPT_ERROR("WhisperDecoderLayer_forward: self_attn_layer_norm failed"); return cleanup(); }
    clReleaseMemObject(x); x = tmp; tmp = nullptr;

    // SELF-attention: K/V come from x itself, NOT the encoder. Pass nullptr for
    // encoder_hidden_states — WhisperSdpaAttention uses `is_cross = (encoder_hidden_states
    // != nullptr)`, so passing it here wrongly turns decoder self-attn into cross-attn.
    tmp = WhisperSdpaAttention_forward(cl_ctx, weights, queue, x, seq_len, layer_idx, start_pos,
                                       k_cache_inout, v_cache_inout, /*encoder_hidden_states=*/nullptr,
                                       (wp + ".self_attn").c_str());
    if (!tmp) { NNOPT_ERROR("WhisperDecoderLayer_forward: self_attn failed"); return cleanup(); }
    clReleaseMemObject(x); x = tmp; tmp = nullptr;
    NNOPT_LAYER_CHECK_FMT("decoder_self_attn_%d", layer_idx, queue, x, (size_t)seq_len * (size_t)MODEL_CONFIG::HIDDEN_SIZE);

    // x = residual + x
    if (!element_add_inplace(queue, s_utils_prog, residual, x, n)) {
        NNOPT_ERROR("WhisperDecoderLayer_forward: element_add_inplace(self_attn) failed");
        return cleanup();
    }
    clReleaseMemObject(x); x = residual; residual = nullptr;

    // ── Cross-attention block (optional) ───────────────────
    if (encoder_hidden_states) {
        residual = clCreateBuffer(cl_ctx.context(), CL_MEM_READ_WRITE, n * sizeof(nnopt_storage_t), nullptr, &err);
        if (err != CL_SUCCESS || !residual) { NNOPT_ERROR_FMT("clCreateBuffer(residual2) %d", (int)err); return cleanup(); }
        err = clEnqueueCopyBuffer(queue, x, residual, 0, 0, n * sizeof(nnopt_storage_t), 0, nullptr, nullptr);
        if (err != CL_SUCCESS) { NNOPT_ERROR_FMT("copy x->residual2 %d", (int)err); return cleanup(); }

        tmp = LayerNorm_forward(cl_ctx, weights, queue, x, seq_len, layer_idx, start_pos,
                                k_cache_inout, v_cache_inout, encoder_hidden_states,
                                (wp + ".encoder_attn_layer_norm").c_str());
        if (!tmp) { NNOPT_ERROR("WhisperDecoderLayer_forward: encoder_attn_layer_norm failed"); return cleanup(); }
        clReleaseMemObject(x); x = tmp; tmp = nullptr;

        tmp = WhisperSdpaAttention_forward(cl_ctx, weights, queue, x, seq_len, layer_idx, start_pos,
                                           k_cache_inout, v_cache_inout, encoder_hidden_states,
                                           (wp + ".encoder_attn").c_str());
        if (!tmp) { NNOPT_ERROR("WhisperDecoderLayer_forward: encoder_attn failed"); return cleanup(); }
        clReleaseMemObject(x); x = tmp; tmp = nullptr;
        NNOPT_LAYER_CHECK_FMT("decoder_encoder_attn_%d", layer_idx, queue, x, (size_t)seq_len * (size_t)MODEL_CONFIG::HIDDEN_SIZE);

        if (!element_add_inplace(queue, s_utils_prog, residual, x, n)) {
            NNOPT_ERROR("WhisperDecoderLayer_forward: element_add_inplace(cross_attn) failed");
            return cleanup();
        }
        clReleaseMemObject(x); x = residual; residual = nullptr;
    }

    // ── FFN block ─────────────────────────────────────────
    residual = clCreateBuffer(cl_ctx.context(), CL_MEM_READ_WRITE, n * sizeof(nnopt_storage_t), nullptr, &err);
    if (err != CL_SUCCESS || !residual) { NNOPT_ERROR_FMT("clCreateBuffer(residual3) %d", (int)err); return cleanup(); }
    err = clEnqueueCopyBuffer(queue, x, residual, 0, 0, n * sizeof(nnopt_storage_t), 0, nullptr, nullptr);
    if (err != CL_SUCCESS) { NNOPT_ERROR_FMT("copy x->residual3 %d", (int)err); return cleanup(); }

    tmp = LayerNorm_forward(cl_ctx, weights, queue, x, seq_len, layer_idx, start_pos,
                            k_cache_inout, v_cache_inout, encoder_hidden_states,
                            (wp + ".final_layer_norm").c_str());
    if (!tmp) { NNOPT_ERROR("WhisperDecoderLayer_forward: final_layer_norm failed"); return cleanup(); }
    clReleaseMemObject(x); x = tmp; tmp = nullptr;

    // fc1
    cl_mem fc1_out = clCreateBuffer(cl_ctx.context(), CL_MEM_READ_WRITE,
                                    (size_t)seq_len * (size_t)MODEL_CONFIG::DECODER_FFN_DIM * sizeof(nnopt_storage_t),
                                    nullptr, &err);
    if (err != CL_SUCCESS || !fc1_out) { NNOPT_ERROR_FMT("clCreateBuffer(fc1_out) %d", (int)err); return cleanup(); }
    if (!pytorch_linear(queue, seq_len, (int)MODEL_CONFIG::DECODER_FFN_DIM, hidden, x,
                        weights.get_buffer(wp + ".fc1.weight"), fc1_out, "gemm_dec_fc1")) {
        NNOPT_ERROR("WhisperDecoderLayer_forward: fc1 pytorch_linear failed");
        clReleaseMemObject(fc1_out);
        return cleanup();
    }
    if (!bias_add_rows_inplace(cl_ctx, queue, fc1_out, weights.get_buffer(wp + ".fc1.bias"),
                              seq_len, (int)MODEL_CONFIG::DECODER_FFN_DIM)) {
        NNOPT_ERROR("WhisperDecoderLayer_forward: fc1 bias add failed");
        clReleaseMemObject(fc1_out);
        return cleanup();
    }

    tmp = GELUActivation_forward(cl_ctx, weights, queue, fc1_out, seq_len, layer_idx, start_pos,
                                k_cache_inout, v_cache_inout, encoder_hidden_states,
                                "gelu");
    clReleaseMemObject(fc1_out);
    if (!tmp) { NNOPT_ERROR("WhisperDecoderLayer_forward: gelu failed"); return cleanup(); }
    fc1_out = tmp; tmp = nullptr;

    // fc2
    cl_mem fc2_out = clCreateBuffer(cl_ctx.context(), CL_MEM_READ_WRITE, n * sizeof(nnopt_storage_t), nullptr, &err);
    if (err != CL_SUCCESS || !fc2_out) { NNOPT_ERROR_FMT("clCreateBuffer(fc2_out) %d", (int)err); clReleaseMemObject(fc1_out); return cleanup(); }
    if (!pytorch_linear(queue, seq_len, hidden, (int)MODEL_CONFIG::DECODER_FFN_DIM, fc1_out,
                        weights.get_buffer(wp + ".fc2.weight"), fc2_out, "gemm_dec_fc2")) {
        NNOPT_ERROR("WhisperDecoderLayer_forward: fc2 pytorch_linear failed");
        clReleaseMemObject(fc1_out);
        clReleaseMemObject(fc2_out);
        return cleanup();
    }
    clReleaseMemObject(fc1_out);
    if (!bias_add_rows_inplace(cl_ctx, queue, fc2_out, weights.get_buffer(wp + ".fc2.bias"), seq_len, hidden)) {
        NNOPT_ERROR("WhisperDecoderLayer_forward: fc2 bias add failed");
        clReleaseMemObject(fc2_out);
        return cleanup();
    }

    if (!element_add_inplace(queue, s_utils_prog, residual, fc2_out, n)) {
        NNOPT_ERROR("WhisperDecoderLayer_forward: element_add_inplace(ffn) failed");
        clReleaseMemObject(fc2_out);
        return cleanup();
    }
    clReleaseMemObject(fc2_out);
    clReleaseMemObject(x);
    x = residual; residual = nullptr;

    return x;
}
} // extern "C"
