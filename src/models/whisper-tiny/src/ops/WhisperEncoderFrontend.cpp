// WhisperEncoderFrontend.cpp — auto-generated audio encoder frontend.
// Reference: model_info/transformers_src/modeling_whisper.py WhisperEncoder.forward
//
// Implements (in PyTorch terms):
//   x = gelu(conv1(input_features))    # [B, n_mels, T_mel] → [B, hidden, T_mel]
//   x = gelu(conv2(x))                  # stride=2 → [B, hidden, T_mel/2]
//   x = x.permute(0, 2, 1)              # → [B, T_mel/2, hidden]
//   x = x + embed_positions.weight      # add learned positional embeddings
//
// Signature matches the universal <Class>_forward pattern so backbone.cpp can
// call this op identically to all others.

#include "../opencl_context.h"
#include "../weights.h"
#include "../debug_utils.h"
#include "../model_config.h"
#include "../utils.h"
#include "../profiler.h"

#include <CL/cl.h>
#include <string>
#include <vector>

extern "C" {
cl_mem WhisperEncoderFrontend_forward(
    OpenCLContext& cl_ctx,
    Weights& weights,
    cl_command_queue queue,
    cl_mem input_features,
    int seq_len,
    int layer_idx,
    int start_pos,
    cl_mem* k_cache_inout,
    cl_mem* v_cache_inout,
    cl_mem encoder_hidden_states,
    const char* weight_prefix)
{
    (void)seq_len; (void)layer_idx; (void)start_pos;
    (void)k_cache_inout; (void)v_cache_inout;
    (void)encoder_hidden_states; (void)weight_prefix;

    if (!input_features) {
        NNOPT_ERROR("WhisperEncoderFrontend_forward: input_features is null");
        return nullptr;
    }

    const std::string p = "model.encoder";
    cl_mem conv1_w = weights.get_buffer(p + ".conv1.weight");
    cl_mem conv1_b = weights.get_buffer(p + ".conv1.bias");
    cl_mem conv2_w = weights.get_buffer(p + ".conv2.weight");
    cl_mem conv2_b = weights.get_buffer(p + ".conv2.bias");
    cl_mem pos_w   = weights.get_buffer(p + ".embed_positions.weight");
    if (!conv1_w || !conv1_b || !conv2_w || !conv2_b || !pos_w) {
        NNOPT_ERROR("WhisperEncoderFrontend_forward: missing encoder.conv1/conv2/embed_positions weights");
        return nullptr;
    }

    const int n_mels = MODEL_CONFIG::NUM_MEL_BINS;
    const int hidden = MODEL_CONFIG::HIDDEN_SIZE;
    const int T_in = (int)MODEL_CONFIG::MAX_SOURCE_POSITIONS * 2;  // 3000 for Whisper
    const int T_out = (int)MODEL_CONFIG::MAX_SOURCE_POSITIONS;     // 1500

    cl_int err = CL_SUCCESS;
    cl_mem conv1_out = nullptr;
    cl_mem conv2_out = nullptr;
    cl_mem out = nullptr;
    cl_program prog = nullptr;
    cl_kernel k_conv = nullptr;
    cl_kernel k_addpos = nullptr;

    auto cleanup = [&]() -> cl_mem {
        if (k_conv) { clReleaseKernel(k_conv); k_conv = nullptr; }
        if (k_addpos) { clReleaseKernel(k_addpos); k_addpos = nullptr; }
        if (prog) { clReleaseProgram(prog); prog = nullptr; }
        if (conv1_out) { clReleaseMemObject(conv1_out); conv1_out = nullptr; }
        if (conv2_out) { clReleaseMemObject(conv2_out); conv2_out = nullptr; }
        if (out) { clReleaseMemObject(out); out = nullptr; }
        return nullptr;
    };

    // Build program once per process.
    static cl_program s_prog = nullptr;
    static cl_kernel s_k_conv = nullptr;
    static cl_kernel s_k_addpos = nullptr;
    if (!s_prog) {
        s_prog = cl_ctx.build_program_from_file("kernels/whisper_encoder_frontend.cl");
        if (!s_prog) {
            NNOPT_ERROR("WhisperEncoderFrontend_forward: failed to build kernels/whisper_encoder_frontend.cl");
            return nullptr;
        }
        cl_int kerr = CL_SUCCESS;
        s_k_conv = clCreateKernel(s_prog, "conv1d_k3_p1", &kerr);
        if (kerr != CL_SUCCESS || !s_k_conv) {
            NNOPT_ERROR_FMT("WhisperEncoderFrontend_forward: clCreateKernel conv1d_k3_p1 failed %d", (int)kerr);
            clReleaseProgram(s_prog); s_prog = nullptr;
            return nullptr;
        }
        s_k_addpos = clCreateKernel(s_prog, "add_positional_embeddings", &kerr);
        if (kerr != CL_SUCCESS || !s_k_addpos) {
            NNOPT_ERROR_FMT("WhisperEncoderFrontend_forward: clCreateKernel add_positional_embeddings failed %d", (int)kerr);
            clReleaseKernel(s_k_conv); s_k_conv = nullptr;
            clReleaseProgram(s_prog); s_prog = nullptr;
            return nullptr;
        }
    }
    prog = s_prog; k_conv = s_k_conv; k_addpos = s_k_addpos;
    clRetainProgram(prog);
    clRetainKernel(k_conv);
    clRetainKernel(k_addpos);

    // Allocate intermediates.
    conv1_out = clCreateBuffer(cl_ctx.context(), CL_MEM_READ_WRITE,
                                (size_t)hidden * (size_t)T_in * sizeof(nnopt_storage_t),
                                nullptr, &err);
    if (err != CL_SUCCESS || !conv1_out) {
        NNOPT_ERROR_FMT("WhisperEncoderFrontend_forward: clCreateBuffer(conv1_out) %d", (int)err);
        return cleanup();
    }
    conv2_out = clCreateBuffer(cl_ctx.context(), CL_MEM_READ_WRITE,
                                (size_t)hidden * (size_t)T_out * sizeof(nnopt_storage_t),
                                nullptr, &err);
    if (err != CL_SUCCESS || !conv2_out) {
        NNOPT_ERROR_FMT("WhisperEncoderFrontend_forward: clCreateBuffer(conv2_out) %d", (int)err);
        return cleanup();
    }
    out = clCreateBuffer(cl_ctx.context(), CL_MEM_READ_WRITE,
                         (size_t)T_out * (size_t)hidden * sizeof(nnopt_storage_t),
                         nullptr, &err);
    if (err != CL_SUCCESS || !out) {
        NNOPT_ERROR_FMT("WhisperEncoderFrontend_forward: clCreateBuffer(out) %d", (int)err);
        return cleanup();
    }

    // conv1: stride=1, gelu fused
    {
        const int C_in = n_mels, C_out = hidden, T_in_local = T_in, T_out_local = T_in, stride = 1, apply_gelu = 1;
        if (!set_arg_checked(k_conv, 0, sizeof(cl_mem), &input_features, "x")) return cleanup();
        if (!set_arg_checked(k_conv, 1, sizeof(cl_mem), &conv1_w, "w")) return cleanup();
        if (!set_arg_checked(k_conv, 2, sizeof(cl_mem), &conv1_b, "b")) return cleanup();
        if (!set_arg_checked(k_conv, 3, sizeof(cl_mem), &conv1_out, "y")) return cleanup();
        if (!set_arg_checked(k_conv, 4, sizeof(int), &C_in, "C_in")) return cleanup();
        if (!set_arg_checked(k_conv, 5, sizeof(int), &C_out, "C_out")) return cleanup();
        if (!set_arg_checked(k_conv, 6, sizeof(int), &T_in_local, "T_in")) return cleanup();
        if (!set_arg_checked(k_conv, 7, sizeof(int), &T_out_local, "T_out")) return cleanup();
        if (!set_arg_checked(k_conv, 8, sizeof(int), &stride, "stride")) return cleanup();
        if (!set_arg_checked(k_conv, 9, sizeof(int), &apply_gelu, "apply_gelu")) return cleanup();
        // LDS-tiled conv (opt #14): one workgroup per output channel (dim 1, lws=1),
        // W time-steps per group (dim 0). Must match conv1d_k3_p1's LDS contract.
        const size_t W = 64;
        size_t lws[2] = { W, 1 };
        size_t gws[2] = { ((size_t)T_out_local + W - 1) / W * W, (size_t)C_out };
        err = clEnqueueNDRangeKernel(queue, k_conv, 2, nullptr, gws, lws, 0, nullptr,
                                     KernelProfiler::event_for("whisper_conv1_gelu"));
        if (err != CL_SUCCESS) {
            NNOPT_ERROR_FMT("WhisperEncoderFrontend_forward: dispatch conv1 %d", (int)err);
            return cleanup();
        }
    }

    // conv2: stride=2, gelu fused
    {
        const int C_in = hidden, C_out = hidden, T_in_local = T_in, T_out_local = T_out, stride = 2, apply_gelu = 1;
        if (!set_arg_checked(k_conv, 0, sizeof(cl_mem), &conv1_out, "x")) return cleanup();
        if (!set_arg_checked(k_conv, 1, sizeof(cl_mem), &conv2_w, "w")) return cleanup();
        if (!set_arg_checked(k_conv, 2, sizeof(cl_mem), &conv2_b, "b")) return cleanup();
        if (!set_arg_checked(k_conv, 3, sizeof(cl_mem), &conv2_out, "y")) return cleanup();
        if (!set_arg_checked(k_conv, 4, sizeof(int), &C_in, "C_in")) return cleanup();
        if (!set_arg_checked(k_conv, 5, sizeof(int), &C_out, "C_out")) return cleanup();
        if (!set_arg_checked(k_conv, 6, sizeof(int), &T_in_local, "T_in")) return cleanup();
        if (!set_arg_checked(k_conv, 7, sizeof(int), &T_out_local, "T_out")) return cleanup();
        if (!set_arg_checked(k_conv, 8, sizeof(int), &stride, "stride")) return cleanup();
        if (!set_arg_checked(k_conv, 9, sizeof(int), &apply_gelu, "apply_gelu")) return cleanup();
        // LDS-tiled conv (opt #14) — same 2D contract as conv1.
        const size_t W = 64;
        size_t lws[2] = { W, 1 };
        size_t gws[2] = { ((size_t)T_out_local + W - 1) / W * W, (size_t)C_out };
        err = clEnqueueNDRangeKernel(queue, k_conv, 2, nullptr, gws, lws, 0, nullptr,
                                     KernelProfiler::event_for("whisper_conv2_gelu"));
        if (err != CL_SUCCESS) {
            NNOPT_ERROR_FMT("WhisperEncoderFrontend_forward: dispatch conv2 %d", (int)err);
            return cleanup();
        }
    }

    // Permute [hidden, T_out] -> [T_out, hidden] and add learned positional embedding [T_out, hidden].
    {
        if (!set_arg_checked(k_addpos, 0, sizeof(cl_mem), &conv2_out, "x_HT")) return cleanup();
        if (!set_arg_checked(k_addpos, 1, sizeof(cl_mem), &pos_w, "pos_TH")) return cleanup();
        if (!set_arg_checked(k_addpos, 2, sizeof(cl_mem), &out, "y_TH")) return cleanup();
        if (!set_arg_checked(k_addpos, 3, sizeof(int), &T_out, "T")) return cleanup();
        if (!set_arg_checked(k_addpos, 4, sizeof(int), &hidden, "H")) return cleanup();
        size_t gws[2] = { (size_t)T_out, (size_t)hidden };
        err = clEnqueueNDRangeKernel(queue, k_addpos, 2, nullptr, gws, nullptr, 0, nullptr,
                                     KernelProfiler::event_for("whisper_add_positional"));
        if (err != CL_SUCCESS) {
            NNOPT_ERROR_FMT("WhisperEncoderFrontend_forward: dispatch add_pos %d", (int)err);
            return cleanup();
        }
    }

    clFinish(queue);  // SYNC-OK: caller consumes `out` immediately in next op

    // Validate the WHOLE frontend output (post-conv1/2, post-gelu, post-pos-embed)
    // against the reference "<encoder>_frontend_out" pre-hook node. Without this
    // the frontend is invisible to SxS/Evaluate and a wrong Conv1D only shows as
    // a degraded encoder-layer-0 cosine the agent can't isolate (Whisper 2026-06).
    NNOPT_LAYER_CHECK("model_encoder_frontend_out", queue, out,
                      (size_t)T_out * (size_t)hidden);

    if (conv1_out) { clReleaseMemObject(conv1_out); conv1_out = nullptr; }
    if (conv2_out) { clReleaseMemObject(conv2_out); conv2_out = nullptr; }
    if (k_conv) { clReleaseKernel(k_conv); k_conv = nullptr; }
    if (k_addpos) { clReleaseKernel(k_addpos); k_addpos = nullptr; }
    if (prog) { clReleaseProgram(prog); prog = nullptr; }
    return out;
}
}
