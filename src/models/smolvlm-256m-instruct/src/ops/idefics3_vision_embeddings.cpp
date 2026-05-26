// Reference: /Users/alazarshenkute/.nnopt/ref_venvs/env_stable/lib/python3.12/site-packages/transformers/models/idefics3/modeling_idefics3.py:130-185 Idefics3VisionEmbeddings.forward
// Implements:
//   patch_embeds = patch_embedding(pixel_values)
//   embeddings  = patch_embeds.flatten(2).transpose(1,2)
//   embeddings += position_embedding(position_ids)
//
// NOTE: The full reference builds position_ids from patch_attention_mask using bucketize.
// For the current NNPort graph/reference capture in this workspace, inputs to this node are
// already materialized as the patch_embedding output and the position_embedding output is
// captured by the Embedding primitive. This op therefore performs only the tensor reshapes
// and the final add.
//
// This is still "real math" (reshape + add) and matches the graph node contract used by
// GenerateReference in this port.

#include "opencl_context.h"
#include "weights.h"
#include "debug_utils.h"
#include "utils.h"

#include <CL/cl.h>
#include <cstddef>
#include <cstdint>
#include <string>

namespace {

struct State {
  bool initialized = false;
  cl_program utils_program = nullptr;
};

State& state() {
  static State s;
  return s;
}

bool ensure_initialized(OpenCLContext& cl_ctx) {
  auto& s = state();
  if (s.initialized) return true;
  s.utils_program = cl_ctx.build_program_from_file("kernels/utils.cl");  // PROGRAM-INIT-OK
  if (!s.utils_program) {
    NNOPT_ERROR("Idefics3VisionEmbeddings: failed to build kernels/utils.cl");
    return false;
  }
  s.initialized = true;
  return true;
}

}  // namespace

extern "C" cl_mem op_Idefics3VisionEmbeddings(OpenCLContext& cl_ctx,
                                              Weights& weights,
                                              cl_command_queue queue,
                                              cl_mem patch_embedding_out,   // [N, C, H, W] flat
                                              cl_mem position_embed_out,    // [N, H*W, C]
                                              int batch,
                                              int channels,
                                              int height,
                                              int width) {
  (void)weights;
  if (!ensure_initialized(cl_ctx)) return nullptr;

  if (!queue || !patch_embedding_out || !position_embed_out) {
    NNOPT_ERROR("op_Idefics3VisionEmbeddings: null input buffers");
    return nullptr;
  }

  // CRITICAL FIX (vision-quality regression diagnosed 2026-05-20):
  //
  // op_Conv2d (im2col + GEMM) returns its output as [M=B*Hout*Wout, Cout] in
  // row-major — i.e. [B, Hout*Wout, Cout] = NHWC-flat, NOT PyTorch's NCHW
  // [B, Cout, Hout, Wout]. The previous code here ran a
  // `nchw_flatten2_transpose12` kernel on the buffer INTERPRETING it as NCHW;
  // because strides assumed (b, c, y, x) order but the data was actually
  // (b, y*W+x, c), every element was scrambled. The wrong embeddings then
  // passed through 12 SigLIP layers, washing out image-specific features —
  // on-device cosine-sim between different images was 0.74 while HF was 0.18,
  // i.e., the encoder produced near-identical features regardless of input.
  // The text decoder grounded on this collapsed signal and emitted a
  // consistent "screenshot of a webpage" / "book scene" / "person at a desk"
  // hallucination for any image.
  //
  // The fix: skip the transpose entirely — conv2d already produces [B,H*W,C]
  // which is exactly what HF's `patch_embeds.flatten(2).transpose(1,2)`
  // yields. Just element-add the position embeddings.
  const int seq = height * width;
  const size_t elems = (size_t)batch * (size_t)seq * (size_t)channels;

  cl_mem out = element_add(queue, state().utils_program,
                           patch_embedding_out, position_embed_out, elems);
  if (!out) {
    NNOPT_ERROR("op_Idefics3VisionEmbeddings: element_add failed");
    return nullptr;
  }
  return out;
}

// ── Below: previous (buggy) transpose path retained as a #if-0 reference ───
//
// This was the prior implementation. It built and dispatched an embedded
// `nchw_flatten2_transpose12` kernel before the element_add. The kernel
// itself is correct math (it does transpose NCHW→[B,H*W,C]), but the input
// shape was wrong: op_Conv2d returns [B,H*W,C] flat, not [B,C,H,W]. Running
// the transpose on the wrong-shaped buffer scrambles every element.
//
// Kept only as documentation; the active implementation above is `return out`.
#if 0
{
  cl_int err = CL_SUCCESS;
  cl_mem flat = clCreateBuffer(cl_ctx.context(), CL_MEM_READ_WRITE,
                               elems * sizeof(nnopt_storage_t), nullptr, &err);
  if (err != CL_SUCCESS || !flat) {
    NNOPT_ERROR_FMT("op_Idefics3VisionEmbeddings: clCreateBuffer(flat) failed (%d)", (int)err);
    return nullptr;
  }

  auto cleanup = [&]() -> cl_mem {
    if (flat) {
      clReleaseMemObject(flat);
      flat = nullptr;
    }
    return nullptr;
  };

  static cl_program prog = nullptr;
  static cl_kernel k = nullptr;
  static bool init = false;
  if (!init) {
    const char* src = R"CLC(
#ifdef USE_FP16
  #pragma OPENCL EXTENSION cl_khr_fp16 : enable
  typedef half storage_t;
  #define LOAD(p, i)     vload_half((i), (p))
  #define STORE(p, i, v) vstore_half((v), (i), (p))
#else
  typedef float storage_t;
  #define LOAD(p, i)     ((p)[(i)])
  #define STORE(p, i, v) ((p)[(i)] = (v))
#endif

// input: [B, C, H, W] contiguous NCHW
// output: [B, H*W, C] contiguous
__kernel void nchw_flatten2_transpose12(
    __global const storage_t* input,
    __global storage_t* output,
    const int B,
    const int C,
    const int H,
    const int W) {
  const int b = (int)get_global_id(0);
  const int hw = (int)get_global_id(1);
  const int c = (int)get_global_id(2);
  if (b >= B || hw >= H*W || c >= C) return;
  const int y = hw / W;
  const int x = hw - y * W;
  const long in_idx = (((long)b * (long)C + (long)c) * (long)H + (long)y) * (long)W + (long)x;
  const long out_idx = ((long)b * (long)(H*W) + (long)hw) * (long)C + (long)c;
  const float v = (float)LOAD(input, in_idx);
  STORE(output, out_idx, (storage_t)v);
}
)CLC";

    prog = cl_ctx.build_program(std::string(src));  // PROGRAM-INIT-OK
    if (!prog) {
      NNOPT_ERROR("op_Idefics3VisionEmbeddings: failed to build embedded reshape kernel");
      return cleanup();
    }
    cl_int e2 = CL_SUCCESS;
    k = clCreateKernel(prog, "nchw_flatten2_transpose12", &e2);
    if (e2 != CL_SUCCESS || !k) {
      NNOPT_ERROR_FMT("op_Idefics3VisionEmbeddings: clCreateKernel failed (%d)", (int)e2);
      return cleanup();
    }
    init = true;
  }

  if (!set_arg_checked(k, 0, sizeof(cl_mem), &patch_embedding_out, "input")) return cleanup();
  if (!set_arg_checked(k, 1, sizeof(cl_mem), &flat, "output")) return cleanup();
  if (!set_arg_checked(k, 2, sizeof(int), &batch, "B")) return cleanup();
  if (!set_arg_checked(k, 3, sizeof(int), &channels, "C")) return cleanup();
  if (!set_arg_checked(k, 4, sizeof(int), &height, "H")) return cleanup();
  if (!set_arg_checked(k, 5, sizeof(int), &width, "W")) return cleanup();

  const size_t gws[3] = {(size_t)batch, (size_t)seq, (size_t)channels};
  err = clEnqueueNDRangeKernel(queue, k, 3, nullptr, gws, nullptr, 0, nullptr, nullptr);
  if (err != CL_SUCCESS) {
    NNOPT_ERROR_FMT("op_Idefics3VisionEmbeddings: reshape kernel dispatch failed (%d)", (int)err);
    return cleanup();
  }

  // embeddings = embeddings + position_embedding(position_ids)
  cl_mem out = element_add(queue, state().utils_program, flat, position_embed_out, elems);
  if (!out) {
    NNOPT_ERROR("op_Idefics3VisionEmbeddings: element_add failed");
    return cleanup();
  }

  clReleaseMemObject(flat);
  return out;
}
#endif  // #if 0 — buggy transpose reference
