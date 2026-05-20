// Reference: extensions/cli/prompts/system/modality/vlm.md (VLM vision pipeline contract)
// Orchestrates: raw RGB (HWC u8) → resize → normalize → patchify → vision encoder → projector.

#include "model.h"
#include "model_config.h"
#include "opencl_context.h"
#include "weights.h"
#include "debug_utils.h"
#include "utils.h"
#include "forward_dispatch.h"

#include <CL/cl.h>
#include <vector>
#include <cstdint>
#include <string>
#include <cstdlib>
#include <fstream>
#include <cmath>

namespace {

struct PipelineState {
  bool initialized = false;
  cl_program resize_prog = nullptr;
  cl_program norm_prog = nullptr;
  cl_program patch_prog = nullptr;
  cl_kernel k_resize = nullptr;
  cl_kernel k_norm = nullptr;
  cl_kernel k_patch = nullptr;
};

PipelineState& state() {
  static PipelineState s;
  return s;
}

bool ensure_kernels(OpenCLContext& cl_ctx) {
  auto& s = state();
  if (s.initialized) return true;

  s.resize_prog = cl_ctx.build_program_from_file("kernels/image_resize.cl");  // PROGRAM-INIT-OK
  if (!s.resize_prog) {
    NNOPT_ERROR("vision_pipeline_forward: failed to build kernels/image_resize.cl");
    return false;
  }
  s.norm_prog = cl_ctx.build_program_from_file("kernels/image_normalize.cl");  // PROGRAM-INIT-OK
  if (!s.norm_prog) {
    NNOPT_ERROR("vision_pipeline_forward: failed to build kernels/image_normalize.cl");
    return false;
  }
  s.patch_prog = cl_ctx.build_program_from_file("kernels/image_patchify.cl");  // PROGRAM-INIT-OK
  if (!s.patch_prog) {
    NNOPT_ERROR("vision_pipeline_forward: failed to build kernels/image_patchify.cl");
    return false;
  }

  cl_int err = CL_SUCCESS;
  s.k_resize = clCreateKernel(s.resize_prog, "image_resize", &err);
  if (err != CL_SUCCESS || !s.k_resize) {
    NNOPT_ERROR_FMT("vision_pipeline_forward: clCreateKernel(resize) failed (%d)", (int)err);
    return false;
  }

  err = CL_SUCCESS;
  s.k_norm = clCreateKernel(s.norm_prog, "image_normalize", &err);
  if (err != CL_SUCCESS || !s.k_norm) {
    NNOPT_ERROR_FMT("vision_pipeline_forward: clCreateKernel(normalize) failed (%d)", (int)err);
    return false;
  }

  err = CL_SUCCESS;
  s.k_patch = clCreateKernel(s.patch_prog, "image_patchify", &err);
  if (err != CL_SUCCESS || !s.k_patch) {
    NNOPT_ERROR_FMT("vision_pipeline_forward: clCreateKernel(patchify) failed (%d)", (int)err);
    return false;
  }

  s.initialized = true;
  return true;
}

}  // namespace

bool vision_pipeline_forward(
    OpenCLContext& cl_ctx,
    Weights& weights,
    const std::vector<uint8_t>& rgb_u8,
    int W, int H,
    std::vector<float>& image_features_out) {
  NNOPT_CHECKPOINT("vision_pipeline_forward: start");

  image_features_out.clear();

  if (rgb_u8.empty() || W <= 0 || H <= 0) {
    NNOPT_ERROR("vision_pipeline_forward: empty/invalid image input");
    return false;
  }

  if (!ensure_kernels(cl_ctx)) return false;

  cl_command_queue queue = cl_ctx.queue();
  cl_context ctx = cl_ctx.context();
  if (!queue || !ctx) {
    NNOPT_ERROR("vision_pipeline_forward: null OpenCL ctx/queue");
    return false;
  }

  cl_mem in_u8 = nullptr;
  cl_mem resized_u8 = nullptr;
  cl_mem resized_f32 = nullptr;
  cl_mem patches = nullptr;
  cl_mem vision_hidden = nullptr;
  cl_mem shuffled = nullptr;
  cl_mem projected = nullptr;

  auto cleanup = [&]() {
    if (projected) { clReleaseMemObject(projected); projected = nullptr; }
    if (shuffled) { clReleaseMemObject(shuffled); shuffled = nullptr; }
    if (vision_hidden) { clReleaseMemObject(vision_hidden); vision_hidden = nullptr; }
    if (patches) { clReleaseMemObject(patches); patches = nullptr; }
    if (resized_f32) { clReleaseMemObject(resized_f32); resized_f32 = nullptr; }
    if (resized_u8) { clReleaseMemObject(resized_u8); resized_u8 = nullptr; }
    if (in_u8) { clReleaseMemObject(in_u8); in_u8 = nullptr; }
  };

  // ── Upload raw RGB u8 (HWC) ──
  {
    cl_int err = CL_SUCCESS;
    // UPLOAD-OK: image bytes
    in_u8 = clCreateBuffer(ctx, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
                           rgb_u8.size() * sizeof(uint8_t),
                           const_cast<uint8_t*>(rgb_u8.data()), &err);
    if (err != CL_SUCCESS || !in_u8) {
      NNOPT_ERROR_FMT("vision_pipeline_forward: clCreateBuffer(in_u8) failed (%d)", (int)err);
      cleanup();
      return false;
    }
  }

  // ── Resize to model image size (u8 HWC → u8 HWC) ──
  const int outW = MODEL_CONFIG::IMAGE_SIZE;
  const int outH = MODEL_CONFIG::IMAGE_SIZE;
  const int C = MODEL_CONFIG::NUM_CHANNELS;
  {
    cl_int err = CL_SUCCESS;
    const size_t out_bytes = (size_t)outH * (size_t)outW * (size_t)C * sizeof(uint8_t);
    resized_u8 = clCreateBuffer(ctx, CL_MEM_READ_WRITE, out_bytes, nullptr, &err);
    if (err != CL_SUCCESS || !resized_u8) {
      NNOPT_ERROR_FMT("vision_pipeline_forward: clCreateBuffer(resized_u8) failed (%d)", (int)err);
      cleanup();
      return false;
    }

    auto& s = state();
    if (!set_arg_checked(s.k_resize, 0, sizeof(cl_mem), &in_u8, "src")) { cleanup(); return false; }
    if (!set_arg_checked(s.k_resize, 1, sizeof(cl_mem), &resized_u8, "dst")) { cleanup(); return false; }
    if (!set_arg_checked(s.k_resize, 2, sizeof(int), &H, "H_in")) { cleanup(); return false; }
    if (!set_arg_checked(s.k_resize, 3, sizeof(int), &W, "W_in")) { cleanup(); return false; }
    if (!set_arg_checked(s.k_resize, 4, sizeof(int), &outH, "H_out")) { cleanup(); return false; }
    if (!set_arg_checked(s.k_resize, 5, sizeof(int), &outW, "W_out")) { cleanup(); return false; }

    const size_t gws[2] = {(size_t)outW, (size_t)outH};
    err = clEnqueueNDRangeKernel(queue, s.k_resize, 2, nullptr, gws, nullptr, 0, nullptr, nullptr);
    if (err != CL_SUCCESS) {
      NNOPT_ERROR_FMT("vision_pipeline_forward: resize dispatch failed (%d)", (int)err);
      cleanup();
      return false;
    }
  }

  // ── Normalize (u8 HWC → storage_t CHW) ──
  // NOTE: image_normalize expects uint8 HWC as src, and writes CHW storage_t.
  {
    cl_int err = CL_SUCCESS;
    const size_t out_elems = (size_t)1 * (size_t)C * (size_t)outH * (size_t)outW;
    resized_f32 = clCreateBuffer(ctx, CL_MEM_READ_WRITE, out_elems * sizeof(nnopt_storage_t), nullptr, &err);
    if (err != CL_SUCCESS || !resized_f32) {
      NNOPT_ERROR_FMT("vision_pipeline_forward: clCreateBuffer(resized_f32) failed (%d)", (int)err);
      cleanup();
      return false;
    }
  }

  // SigLIP defaults (also common for Idefics): mean=std=0.5 => output in [-1,1].
  // If this diverges, load mean/std from preprocessor_config.json.
  // SigLIP defaults (also common for Idefics): mean=std=0.5 => output in [-1,1].
  // If this diverges, load mean/std from preprocessor_config.json.
  {
    const float mean_r = 0.5f, mean_g = 0.5f, mean_b = 0.5f;
    const float std_r = 0.5f, std_g = 0.5f, std_b = 0.5f;

    auto& s = state();
    if (!set_arg_checked(s.k_norm, 0, sizeof(cl_mem), &resized_u8, "src")) { cleanup(); return false; }
    if (!set_arg_checked(s.k_norm, 1, sizeof(cl_mem), &resized_f32, "dst")) { cleanup(); return false; }
    if (!set_arg_checked(s.k_norm, 2, sizeof(int), &outH, "H")) { cleanup(); return false; }
    if (!set_arg_checked(s.k_norm, 3, sizeof(int), &outW, "W")) { cleanup(); return false; }
    if (!set_arg_checked(s.k_norm, 4, sizeof(float), &mean_r, "mean_r")) { cleanup(); return false; }
    if (!set_arg_checked(s.k_norm, 5, sizeof(float), &mean_g, "mean_g")) { cleanup(); return false; }
    if (!set_arg_checked(s.k_norm, 6, sizeof(float), &mean_b, "mean_b")) { cleanup(); return false; }
    if (!set_arg_checked(s.k_norm, 7, sizeof(float), &std_r, "std_r")) { cleanup(); return false; }
    if (!set_arg_checked(s.k_norm, 8, sizeof(float), &std_g, "std_g")) { cleanup(); return false; }
    if (!set_arg_checked(s.k_norm, 9, sizeof(float), &std_b, "std_b")) { cleanup(); return false; }

    const size_t gws[2] = {(size_t)outW, (size_t)outH};
    cl_int err = clEnqueueNDRangeKernel(queue, s.k_norm, 2, nullptr, gws, nullptr, 0, nullptr, nullptr);
    if (err != CL_SUCCESS) {
      NNOPT_ERROR_FMT("vision_pipeline_forward: normalize dispatch failed (%d)", (int)err);
      cleanup();
      return false;
    }
  }

  // ── Patchify to [num_patches, patch_dim] storage_t ──
  const int patch = MODEL_CONFIG::PATCH_SIZE;
  const int grid = outW / patch;
  const int num_patches = grid * grid;
  const int patch_dim = C * patch * patch;
  {
    cl_int err = CL_SUCCESS;
    const size_t patch_elems = (size_t)num_patches * (size_t)patch_dim;
    patches = clCreateBuffer(ctx, CL_MEM_READ_WRITE, patch_elems * sizeof(nnopt_storage_t), nullptr, &err);
    if (err != CL_SUCCESS || !patches) {
      NNOPT_ERROR_FMT("vision_pipeline_forward: clCreateBuffer(patches) failed (%d)", (int)err);
      cleanup();
      return false;
    }

    auto& s = state();
    if (!set_arg_checked(s.k_patch, 0, sizeof(cl_mem), &resized_f32, "image")) { cleanup(); return false; }
    if (!set_arg_checked(s.k_patch, 1, sizeof(cl_mem), &patches, "patches")) { cleanup(); return false; }
    if (!set_arg_checked(s.k_patch, 2, sizeof(int), &outH, "H")) { cleanup(); return false; }
    if (!set_arg_checked(s.k_patch, 3, sizeof(int), &outW, "W")) { cleanup(); return false; }
    if (!set_arg_checked(s.k_patch, 4, sizeof(int), &patch, "patch_size")) { cleanup(); return false; }

    const size_t gws[2] = {(size_t)num_patches, (size_t)patch_dim};
    err = clEnqueueNDRangeKernel(queue, s.k_patch, 2, nullptr, gws, nullptr, 0, nullptr, nullptr);
    if (err != CL_SUCCESS) {
      NNOPT_ERROR_FMT("vision_pipeline_forward: patchify dispatch failed (%d)", (int)err);
      cleanup();
      return false;
    }
  }

  // ── BISECT: optionally swap in HF reference tensors ──
  // NNOPT_BISECT_PIXEL_VALUES=1 → load fixtures/sample_pixel_values_chw.bin
  //                               (3*512*512 fp32) into resized_f32, skipping
  //                               resize + normalize. Tests preprocessing.
  // NNOPT_BISECT_VISION_HIDDEN=1 → after computing vision_hidden, OVERWRITE
  //                                with fixtures/sample_vision_hidden.bin
  //                                (1024*768 fp32). Tests vision encoder.
  const char* bisect_pv = std::getenv("NNOPT_BISECT_PIXEL_VALUES");
  if (bisect_pv && bisect_pv[0] == '1') {
    const char* path = "fixtures/sample_pixel_values_chw.bin";
    std::ifstream f(path, std::ios::binary);
    if (f) {
      const size_t n = (size_t)C * (size_t)outH * (size_t)outW;
      std::vector<float> host(n);
      f.read(reinterpret_cast<char*>(host.data()), n * sizeof(float));
      f.close();
#ifdef NNOPT_USE_FP16
      std::vector<nnopt_storage_t> half_buf(n);
      for (size_t i = 0; i < n; ++i) half_buf[i] = (nnopt_storage_t)nnopt_f32_to_f16(host[i]);
      cl_int werr = clEnqueueWriteBuffer(queue, resized_f32, CL_TRUE, 0,
                                         n * sizeof(nnopt_storage_t),
                                         half_buf.data(), 0, nullptr, nullptr);
#else
      cl_int werr = clEnqueueWriteBuffer(queue, resized_f32, CL_TRUE, 0,
                                         n * sizeof(float),
                                         host.data(), 0, nullptr, nullptr);
#endif
      if (werr != CL_SUCCESS) {
        NNOPT_ERROR_FMT("BISECT pixel_values write failed (%d)", (int)werr);
      } else {
        std::fprintf(stderr, "[BISECT] loaded HF pixel_values from %s — skipping resize+normalize\n", path);
      }
    } else {
      std::fprintf(stderr, "[BISECT] NNOPT_BISECT_PIXEL_VALUES=1 but %s missing\n", path);
    }
  }

  // ── Vision tower (already ported) ──
  // Input must be pixel_values [B,C,H,W].
  vision_hidden = op_Idefics3VisionTransformer(cl_ctx,
                                               weights,
                                               queue,
                                               resized_f32,
                                               /*patch_attention_mask_opt=*/nullptr,
                                               /*B=*/1,
                                               /*C=*/C,
                                               /*H=*/outH,
                                               /*W=*/outW,
                                               /*patch_size=*/patch,
                                               /*patch_emb_w=*/"model.vision_model.embeddings.patch_embedding.weight",
                                               /*patch_emb_b=*/"model.vision_model.embeddings.patch_embedding.bias",
                                               /*pos_emb_w=*/"model.vision_model.embeddings.position_embedding.weight");
  if (!vision_hidden) {
    NNOPT_ERROR("vision_pipeline_forward: op_Idefics3VisionTransformer failed");
    cleanup();
    return false;
  }

  // NNOPT_DUMP_VISION_HIDDEN=<path> writes the on-device vision encoder output
  // (1024 × 768 fp32) so we can diff against HF for layer-by-layer bisect.
  const char* dump_vh = std::getenv("NNOPT_DUMP_VISION_HIDDEN");
  if (dump_vh && dump_vh[0]) {
    const size_t T_vh = (size_t)(outH / patch) * (size_t)(outW / patch);
    const size_t D_vh = (size_t)MODEL_CONFIG::VISION_CONFIG_HIDDEN_SIZE;
    const size_t n = T_vh * D_vh;
    std::vector<float> host(n, 0.0f);
#ifdef NNOPT_USE_FP16
    std::vector<nnopt_storage_t> half_buf(n);
    cl_int rerr = clEnqueueReadBuffer(queue, vision_hidden, CL_TRUE, 0,
                                      n * sizeof(nnopt_storage_t),
                                      half_buf.data(), 0, nullptr, nullptr);
    if (rerr == CL_SUCCESS) {
      for (size_t i = 0; i < n; ++i) host[i] = nnopt_f16_to_f32((uint16_t)half_buf[i]);
    }
#else
    cl_int rerr = clEnqueueReadBuffer(queue, vision_hidden, CL_TRUE, 0,
                                      n * sizeof(float),
                                      host.data(), 0, nullptr, nullptr);
#endif
    if (rerr == CL_SUCCESS) {
      std::ofstream f(dump_vh, std::ios::binary);
      f.write(reinterpret_cast<const char*>(host.data()), n * sizeof(float));
      f.close();
      // Print a few stats so we can sanity-check at a glance.
      float minv = host[0], maxv = host[0], sumv = 0.0f;
      for (float v : host) { if (v < minv) minv = v; if (v > maxv) maxv = v; sumv += v; }
      std::fprintf(stderr, "[DUMP] vision_hidden → %s  (n=%zu min=%.4f max=%.4f mean=%.4f)\n",
                   dump_vh, n, minv, maxv, sumv / (float)n);
    }
  }

  const char* bisect_vh = std::getenv("NNOPT_BISECT_VISION_HIDDEN");
  if (bisect_vh && bisect_vh[0] == '1') {
    const char* path = "fixtures/sample_vision_hidden.bin";
    std::ifstream f(path, std::ios::binary);
    if (f) {
      const size_t T_vh = (size_t)(outH / patch) * (size_t)(outW / patch);
      const size_t D_vh = (size_t)MODEL_CONFIG::VISION_CONFIG_HIDDEN_SIZE;
      const size_t n = T_vh * D_vh;
      std::vector<float> host(n);
      f.read(reinterpret_cast<char*>(host.data()), n * sizeof(float));
      f.close();
#ifdef NNOPT_USE_FP16
      std::vector<nnopt_storage_t> half_buf(n);
      for (size_t i = 0; i < n; ++i) half_buf[i] = (nnopt_storage_t)nnopt_f32_to_f16(host[i]);
      cl_int werr = clEnqueueWriteBuffer(queue, vision_hidden, CL_TRUE, 0,
                                         n * sizeof(nnopt_storage_t),
                                         half_buf.data(), 0, nullptr, nullptr);
#else
      cl_int werr = clEnqueueWriteBuffer(queue, vision_hidden, CL_TRUE, 0,
                                         n * sizeof(float),
                                         host.data(), 0, nullptr, nullptr);
#endif
      if (werr != CL_SUCCESS) {
        NNOPT_ERROR_FMT("BISECT vision_hidden write failed (%d)", (int)werr);
      } else {
        std::fprintf(stderr, "[BISECT] loaded HF vision_hidden from %s — overwriting on-device encoder output\n", path);
      }
    } else {
      std::fprintf(stderr, "[BISECT] NNOPT_BISECT_VISION_HIDDEN=1 but %s missing\n", path);
    }
  }

  // ── Pixel shuffle (scale_factor=4) ──
  // Compresses the [num_patches=1024, VISION_HIDDEN=768] patch grid into a
  // [64, 12288] block that feeds the connector's Linear(12288 -> 576). The
  // connector weight is [576, 12288] (vision_hidden * scale_factor^2 = 768*16),
  // confirming pixel_shuffle runs BEFORE the projection. The resulting 64-row
  // output matches the 64 IMAGE_TOKEN placeholders the on-device tokenizer
  // emits, so splice gets a 1:1 position-to-feature match.
  const int grid_in = outW / patch;           // 32
  const int scale = MODEL_CONFIG::SCALE_FACTOR; // 4
  const int shuffled_rows = (grid_in / scale) * (grid_in / scale);          // 64
  const int shuffled_in_features = MODEL_CONFIG::VISION_CONFIG_HIDDEN_SIZE * scale * scale; // 12288
  shuffled = op_PixelShuffle(cl_ctx,
                             queue,
                             vision_hidden,
                             /*IN_H=*/grid_in,
                             /*IN_W=*/grid_in,
                             /*IN_C=*/MODEL_CONFIG::VISION_CONFIG_HIDDEN_SIZE,
                             /*SCALE=*/scale);
  if (!shuffled) {
    NNOPT_ERROR("vision_pipeline_forward: op_PixelShuffle failed");
    cleanup();
    return false;
  }

  // ── Project to LM hidden via connector (already ported) ──
  // Connector is a single Linear (no bias) from 12288 -> 576.
  projected = op_Idefics3Connector(cl_ctx,
                                  weights,
                                  queue,
                                  shuffled,
                                  /*rows=*/shuffled_rows,
                                  /*in_features=*/shuffled_in_features,
                                  /*out_features=*/MODEL_CONFIG::HIDDEN_SIZE,
                                  "model.connector.modality_projection.proj.weight");
  if (!projected) {
    NNOPT_ERROR("vision_pipeline_forward: op_Idefics3Connector failed");
    cleanup();
    return false;
  }

  // ── Read back projected features to host float32 ──
  const size_t feat_elems = (size_t)shuffled_rows * (size_t)MODEL_CONFIG::HIDDEN_SIZE;
#ifdef NNOPT_USE_FP16
  std::vector<nnopt_storage_t> tmp_half(feat_elems);
  cl_int err = clEnqueueReadBuffer(queue, projected, CL_TRUE, 0,
                                  feat_elems * sizeof(nnopt_storage_t),
                                  tmp_half.data(), 0, nullptr, nullptr);
  if (err != CL_SUCCESS) {
    NNOPT_ERROR_FMT("vision_pipeline_forward: readback failed (%d)", (int)err);
    cleanup();
    return false;
  }
  image_features_out.resize(feat_elems);
  for (size_t i = 0; i < feat_elems; ++i) {
    image_features_out[i] = nnopt_f16_to_f32((uint16_t)tmp_half[i]);
  }
#else
  image_features_out.resize(feat_elems);
  cl_int err = clEnqueueReadBuffer(queue, projected, CL_TRUE, 0,
                                  feat_elems * sizeof(float),
                                  image_features_out.data(), 0, nullptr, nullptr);
  if (err != CL_SUCCESS) {
    NNOPT_ERROR_FMT("vision_pipeline_forward: readback failed (%d)", (int)err);
    cleanup();
    return false;
  }
#endif

  cleanup();
  return true;
}
