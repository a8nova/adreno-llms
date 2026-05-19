// Reference: model_info/transformers_src/modeling_vits.py (VITS length regulator / forward path)
// NOTE: This file is part of the NNOpt TTS (VITS) forward graph.

#include "opencl_context.h"
#include "weights.h"
#include "debug_utils.h"
#include "profiler.h"
#include "utils.h"  // nnopt_storage_t

#include <CL/cl.h>
#include <cstdint>
#include <vector>
#include <cmath>
#include <cstring>

// NOTE: RegResult is part of the public op ABI: backbone.cpp forward-declares the same
// struct and relies on identical layout. Do NOT hide it in an anonymous namespace.
extern "C" {
struct RegResult {
  cl_mem expanded_hidden = nullptr; // [T_audio, hidden]
  cl_mem expanded_mask = nullptr;   // [T_audio] (storage_t, 0/1)
  int T_audio = 0;
  int _pad = 0;  // ABI: pad to 8-byte alignment (matches backbone.cpp's forward-declared struct)
};
}  // extern "C"

// Length regulator: expands per-text-token hidden states to per-audio-frame hidden states.
// Input:
//  - hidden_text: [T_text, hidden]
//  - durations_i32: [T_text] number of frames per text token
// Output:
//  - expanded_hidden: [sum(d), hidden]
//  - expanded_mask: [sum(d)]
//
// Uses kernels/length_regulator.cl.
extern "C" RegResult op_length_regulator(OpenCLContext& cl_ctx,
                                        Weights& /*weights*/,
                                        cl_command_queue queue,
                                        cl_mem hidden_text,
                                        cl_mem durations_i32,
                                        int T_text,
                                        int hidden_size) {
  NNOPT_CHECKPOINT("op_length_regulator entry");
  RegResult res;

  if (!queue || !hidden_text || !durations_i32 || T_text <= 0 || hidden_size <= 0) {
    NNOPT_ERROR("op_length_regulator: bad args");
    return res;
  }

  // Build the program and kernels (lazy init) once per process.
  // PROGRAM-INIT-OK: this is a leaf op used in the TTS forward graph; we keep program
  // objects process-lifetime because some Android drivers have issues if released.
  static cl_program prog = nullptr;
  static cl_kernel k_gather = nullptr;
  static cl_kernel k_mask_fill = nullptr;

  if (!prog) {
    prog = cl_ctx.build_program_from_file("kernels/length_regulator.cl");
    if (!prog) {
      NNOPT_ERROR("op_length_regulator: build_program_from_file(kernels/length_regulator.cl) failed");
      return res;
    }

    cl_int kerr = CL_SUCCESS;
    k_gather = clCreateKernel(prog, "length_regulator", &kerr);
    if (kerr != CL_SUCCESS || !k_gather) {
      NNOPT_ERROR_FMT("op_length_regulator: clCreateKernel(length_regulator) failed %d", (int)kerr);
      return res;
    }

    k_mask_fill = clCreateKernel(prog, "fill_mask", &kerr);
    if (kerr != CL_SUCCESS || !k_mask_fill) {
      NNOPT_ERROR_FMT("op_length_regulator: clCreateKernel(fill_mask) failed %d", (int)kerr);
      return res;
    }
  }

  // SYNC-01: The queue is in-order; our blocking read below is an implicit sync.
  // Avoid an unconditional clFinish here — on some Android stacks this has been
  // observed to increase driver flakiness rather than reduce it.

  cl_int err = CL_SUCCESS;
  cl_context ctx = cl_ctx.context();

  // Read durations to host to compute T_audio (small; T_text is small).
  // IMPORTANT: The upstream graph uses int32 durations, but in failure modes we may
  // accidentally pass token_ids (int32) here. Those values are small (<= vocab),
  // so a naive sum can look "reasonable" but cause out-of-bounds gathers later.
  std::vector<int32_t> h_dur((size_t)T_text, 0);
  err = clEnqueueReadBuffer(queue,
                           durations_i32,
                           CL_TRUE,
                           0,
                           (size_t)T_text * sizeof(int32_t),
                           h_dur.data(),
                           0,
                           nullptr,
                           nullptr);
  if (err != CL_SUCCESS) {
    NNOPT_ERROR_FMT("op_length_regulator: read durations failed %d", (int)err);
    // Do NOT release static kernels/program here (process-lifetime).
    return res;
  }

  // Clamp durations defensively.
  // Reference: modeling_vits.py clamps durations to be >= 1 frame.
  // Without this, total can become 0 and downstream ops crash.
  static constexpr int kMaxFramesPerChar = 2048;  // safety guard; should never hit for normal TTS.
  int total = 0;
  for (int i = 0; i < T_text; ++i) {
    int d = (int)h_dur[(size_t)i];
    if (d < 1) d = 1;
    if (d > kMaxFramesPerChar) d = kMaxFramesPerChar;
    h_dur[(size_t)i] = (int32_t)d;
    total += d;
  }

  if (total <= 0) {
    NNOPT_ERROR("op_length_regulator: total duration is 0 — upstream duration computation is broken");
    // Do NOT release static kernels/program here (process-lifetime).
    return res;
  }

  // Also guard absurd totals to avoid OOM / driver faults.
  // NOTE: TTS can legitimately exceed a few thousand frames; don't hard-fail.
  // If this gets absurdly large, clamp per-char durations above already prevents OOM.
  static constexpr int kMaxTotalFrames = 131072;  // safety cap (~2.7s @ 48k/ hop=??), still generous.
  if (total > kMaxTotalFrames) {
    NNOPT_ERROR_FMT("op_length_regulator: total duration too large (%d) — clamping to %d", total, kMaxTotalFrames);
    total = kMaxTotalFrames;
  }

  res.T_audio = total;

  cl_mem char_idx_i32 = nullptr;
  cl_mem expanded_hidden = nullptr;
  cl_mem expanded_mask = nullptr;

  auto cleanup = [&]() -> RegResult {
    if (char_idx_i32) {
      clReleaseMemObject(char_idx_i32);
      char_idx_i32 = nullptr;
    }
    if (expanded_hidden) {
      clReleaseMemObject(expanded_hidden);
      expanded_hidden = nullptr;
    }
    if (expanded_mask) {
      clReleaseMemObject(expanded_mask);
      expanded_mask = nullptr;
    }

    return RegResult{};
  };

  // Host compute char_idx[total] for each output frame.
  // IMPORTANT: Always fully initialize; if we somehow under-fill, pad with last valid char.
  std::vector<int32_t> h_char_idx((size_t)total, 0);
  {
    int out_t = 0;
    int32_t last_i = 0;
    for (int i = 0; i < T_text; ++i) {
      const int d = h_dur[(size_t)i];
      if (d > 0) last_i = (int32_t)i;
      for (int k = 0; k < d; ++k) {
        if (out_t < total) h_char_idx[(size_t)out_t] = (int32_t)i;
        ++out_t;
      }
    }
    for (; out_t < total; ++out_t) h_char_idx[(size_t)out_t] = last_i;
  }

  // Allocate+upload char_idx as CL_MEM_COPY_HOST_PTR.
  // UPLOAD-OK: char_idx (small)
  char_idx_i32 = clCreateBuffer(ctx,
                               CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
                               (size_t)total * sizeof(int32_t),
                               h_char_idx.data(),
                               &err);
  if (err != CL_SUCCESS || !char_idx_i32) {
    NNOPT_ERROR_FMT("op_LengthRegulator: clCreateBuffer(COPY_HOST_PTR) char_idx failed %d", (int)err);
    return cleanup();
  }

  // Allocate expanded outputs (storage_t buffers)
  const size_t n_hidden = (size_t)total * (size_t)hidden_size;
  expanded_hidden = clCreateBuffer(ctx, CL_MEM_READ_WRITE,
                                  n_hidden * sizeof(nnopt_storage_t), nullptr, &err);
  if (err != CL_SUCCESS || !expanded_hidden) {
    NNOPT_ERROR_FMT("op_LengthRegulator: clCreateBuffer expanded_hidden failed %d", (int)err);
    return cleanup();
  }
  expanded_mask = clCreateBuffer(ctx, CL_MEM_READ_WRITE,
                                (size_t)total * sizeof(nnopt_storage_t), nullptr, &err);
  if (err != CL_SUCCESS || !expanded_mask) {
    NNOPT_ERROR_FMT("op_LengthRegulator: clCreateBuffer expanded_mask failed %d", (int)err);
    return cleanup();
  }

  // length_regulator gather: expanded_hidden[t, c] = hidden_text[char_idx[t], c]
  // NOTE: do NOT use `err |= clSetKernelArg(...)` — OpenCL returns negative codes and
  // bitwise-OR can turn an error into 0 (false success), leading to a later SIGSEGV.
  if (!set_arg_checked(k_gather, 0, sizeof(cl_mem), &hidden_text, "src")) return cleanup();
  if (!set_arg_checked(k_gather, 1, sizeof(cl_mem), &char_idx_i32, "char_idx")) return cleanup();
  if (!set_arg_checked(k_gather, 2, sizeof(cl_mem), &expanded_hidden, "out")) return cleanup();
  if (!set_arg_checked(k_gather, 3, sizeof(int), &total, "T_frames")) return cleanup();
  if (!set_arg_checked(k_gather, 4, sizeof(int), &hidden_size, "C")) return cleanup();
  if (!set_arg_checked(k_gather, 5, sizeof(int), &T_text, "T_chars")) return cleanup();

  // Dispatch one work-item per element.
  const size_t total_elems = n_hidden;

  NNOPT_CHECKPOINT_FMT(
      "op_length_regulator: dispatch gather total=%d hidden=%d elems=%zu",
      total, hidden_size, total_elems);

  // NOTE: Adreno can SIGSEGV with huge 1D GWS when lws is nullptr.
  // Use a conservative fixed lws and round up.
  const size_t lws1[1] = {256};
  const size_t gws1[1] = {((total_elems + lws1[0] - 1) / lws1[0]) * lws1[0]};

  err = clEnqueueNDRangeKernel(queue,
                              k_gather,
                              1,
                              nullptr,
                              gws1,
                              lws1,
                              0,
                              nullptr,
                              KernelProfiler::event_for("length_regulator"));
  if (err != CL_SUCCESS) {
    NNOPT_ERROR_FMT("op_LengthRegulator: dispatch length_regulator failed %d", (int)err);
    return cleanup();
  }

  // Keep sync debug-only; queue is in-order so downstream kernels see writes.
  // (Also helps attribute any driver fault to this dispatch.)
  NNOPT_DEBUG_SYNC(queue);

  // Mask is currently all-ones (every expanded frame is valid).
  // Fill on-device to avoid host-side half ABI pitfalls (some Android stacks SIGSEGV
  // on clEnqueueWriteBuffer with half-typed host pointers).
  // NOTE: If downstream later needs a true padding mask, implement it here.
  {
    if (!set_arg_checked(k_mask_fill, 0, sizeof(cl_mem), &expanded_mask, "mask")) return cleanup();
    if (!set_arg_checked(k_mask_fill, 1, sizeof(int), &total, "n")) return cleanup();

    const size_t lws2[1] = {256};
    const size_t gws2[1] = {(((size_t)total + lws2[0] - 1) / lws2[0]) * lws2[0]};

    err = clEnqueueNDRangeKernel(queue,
                                k_mask_fill,
                                1,
                                nullptr,
                                gws2,
                                lws2,
                                0,
                                nullptr,
                                KernelProfiler::event_for("length_regulator_mask"));
    if (err != CL_SUCCESS) {
      NNOPT_ERROR_FMT("op_LengthRegulator: dispatch fill_mask failed %d", (int)err);
      return cleanup();
    }
  }

  // Ownership handoff.
  res.expanded_hidden = expanded_hidden;
  res.expanded_mask = expanded_mask;

  // Release local refs not returned.
  clReleaseMemObject(char_idx_i32);
  char_idx_i32 = nullptr;

  return res;
}
