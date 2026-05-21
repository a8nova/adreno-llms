// Reference: model_info/transformers_src/modeling_vits.py:255-350 VitsHifiGan.forward
// Implements the HiFi-GAN decoder used by VITS.

#include "opencl_context.h"
#include "weights.h"
#include "debug_utils.h"
#include "model_config.h"
#include "profiler.h"
#include "utils.h"  // nnopt_storage_t, nnopt_f16_to_f32

#include <CL/cl.h>
#include <clblast.h>
#include <cstdint>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <cstdlib>
#include <string>
#include <unordered_map>
#include <vector>

namespace {

// Buffer pool: every transient buffer routes through here. We track the
// byte size of every cl_mem we hand out so callers can `pool_release(buf)`
// without needing to remember the size. On Adreno OpenCL clCreateBuffer is
// ~5–20ms (driver-side allocation, mapping, zero-init); reusing dramatically
// reduces vocoder host overhead.
static std::vector<std::pair<size_t, cl_mem>> g_buf_pool_free;
static std::vector<std::pair<cl_mem, size_t>> g_buf_pool_known;

static cl_mem pool_acquire(cl_context ctx, size_t bytes) {
  // Phase A #1 — buffer pool is now ON by default. clCreateBuffer is 5–20 ms
  // on Adreno (driver-side allocation, mapping, zero-init). With ~50 fresh
  // allocs per vocoder run that's 250 ms–1 s of pure host overhead. The
  // previously-noted "Adreno sync on reuse" issue is moot in the steady-
  // state vocoder path: every reused buffer goes through a write→read→
  // discard cycle that the driver already serializes via the in-order
  // queue. Set NNOPT_VOC_POOL=0 to opt back into per-call allocs for A/B.
  static int s_enabled = -1;
  if (s_enabled < 0) {
    const char* env = std::getenv("NNOPT_VOC_POOL");
    s_enabled = (env && env[0] == '0') ? 0 : 1;
  }
  if (s_enabled) {
    for (auto it = g_buf_pool_free.begin(); it != g_buf_pool_free.end(); ++it) {
      if (it->first == bytes) {
        cl_mem b = it->second;
        g_buf_pool_free.erase(it);
        return b;
      }
    }
  }
  cl_int err = CL_SUCCESS;
  cl_mem b = clCreateBuffer(ctx, CL_MEM_READ_WRITE, bytes, nullptr, &err);
  if (err != CL_SUCCESS || !b) return nullptr;
  if (s_enabled) g_buf_pool_known.push_back({b, bytes});
  return b;
}

static bool pool_release(cl_mem b) {
  if (!b) return false;
  for (const auto& kv : g_buf_pool_known) {
    if (kv.first == b) {
      g_buf_pool_free.push_back({kv.second, b});
      return true;
    }
  }
  // Not a pool buffer (probably a Weights cl_mem) — leave it alone.
  return false;
}

// Convenience: try pool first, else fall back to plain release. Lets us
// uniformly call this on any cl_mem at end-of-life.
static void pool_release_or_release(cl_mem b) {
  if (!b) return;
  if (!pool_release(b)) clReleaseMemObject(b);
}

static cl_program build_prog(OpenCLContext& cl_ctx, const char* path) {
  cl_program p = cl_ctx.build_program_from_file(path);
  if (!p) NNOPT_ERROR_FMT("Failed to build program %s", path);
  return p;
}

static bool run_leaky_relu(OpenCLContext& cl_ctx,
                           cl_command_queue queue,
                           cl_program prog,
                           cl_mem x,
                           int N,
                           float slope) {
  // Cached cl_kernel — clCreateKernel is ~ms-level on Adreno; this op is
  // called 4+1 times per vocoder run and many times per text_encoder run.
  cl_int err = CL_SUCCESS;
  static cl_kernel k = nullptr;
  if (!k) {
    k = clCreateKernel(prog, "leaky_relu", &err);
    if (err != CL_SUCCESS || !k) {
      NNOPT_ERROR_FMT("clCreateKernel(leaky_relu) failed (%d)", (int)err);
      return false;
    }
  }
  bool ok = true;
  ok = ok && set_arg_checked(k, 0, sizeof(cl_mem), &x, "x");
  ok = ok && set_arg_checked(k, 1, sizeof(int), &N, "N");
  ok = ok && set_arg_checked(k, 2, sizeof(float), &slope, "negative_slope");
  if (!ok) return false;
  const size_t gws[1] = {(size_t)N};
  err = clEnqueueNDRangeKernel(queue, k, 1, nullptr, gws, nullptr, 0, nullptr,
                               KernelProfiler::event_for("vocoder.leaky_relu"));
  if (err != CL_SUCCESS) {
    NNOPT_ERROR_FMT("dispatch leaky_relu failed (%d)", (int)err);
    return false;
  }
  return true;
}

static bool run_tanh_inplace(OpenCLContext& cl_ctx,
                             cl_command_queue queue,
                             cl_program prog,
                             cl_mem x,
                             int N) {
  cl_int err = CL_SUCCESS;
  cl_kernel k = clCreateKernel(prog, "tanh_inplace", &err);
  if (err != CL_SUCCESS || !k) {
    NNOPT_ERROR_FMT("clCreateKernel(tanh_inplace) failed (%d)", (int)err);
    return false;
  }
  bool ok = true;
  ok = ok && set_arg_checked(k, 0, sizeof(cl_mem), &x, "x");
  ok = ok && set_arg_checked(k, 1, sizeof(int), &N, "N");
  if (!ok) {
    clReleaseKernel(k);
    return false;
  }
  const size_t gws[1] = {(size_t)N};
  err = clEnqueueNDRangeKernel(queue, k, 1, nullptr, gws, nullptr, 0, nullptr,
                               KernelProfiler::event_for("vocoder.tanh"));
  clReleaseKernel(k);
  if (err != CL_SUCCESS) {
    NNOPT_ERROR_FMT("dispatch tanh_inplace failed (%d)", (int)err);
    return false;
  }
  return true;
}

// GEMM-based conv1d via im2col + CLBlast HGemm.
//
// Why: the naive conv_1d.cl kernel re-reads weights from global memory for
// every output element. Per the kernel profile, resblock convs dominate GPU
// time (73%). CLBlast HGemm is hand-tuned per Adreno tile shapes and gives
// 2–5× over naive. We pay a small im2col write but trade it for highly
// optimized inner-product accumulation.
//
// Math:
//   im2col(in[C_in, L_in])  →  M[C_in*K, L_out]
//   out[C_out, L_out]       =  W[C_out, C_in*K] @ M
//   + bias broadcast over L_out
//
// W is already laid out as [C_out, C_in, K] flat — bitwise identical to
// [C_out, C_in*K]. No weight rearrangement needed.
// ─── image2d-resident weight path (Path 1 from the OpenCL guide §6.2) ───
//
// Lazily pack each conv weight buffer into a half-RGBA image of dimensions
// (C_in*K, ceil(C_out/4)) and dispatch a fused conv1d kernel that reads weights
// via the Adreno texture engine + image L1 cache. The packing is one-time,
// amortised over thousands of forward passes; the per-call cost is just the
// compute kernel itself, no im2col write/read round trip.
//
// Cache key is the source weight cl_mem pointer (weight buffers live in the
// Weights container for the lifetime of the process — pointer identity is stable).

struct ImageWeight {
  cl_mem img;
  int C_out;
  int CinK;
};
static std::unordered_map<cl_mem, ImageWeight> g_weight_image_cache;

static cl_mem get_weight_image(OpenCLContext& cl_ctx,
                               cl_command_queue queue,
                               cl_mem w_buf,
                               int C_out, int C_in, int K) {
  const int CinK = C_in * K;
  auto it = g_weight_image_cache.find(w_buf);
  if (it != g_weight_image_cache.end() &&
      it->second.C_out == C_out && it->second.CinK == CinK) {
    return it->second.img;
  }

  cl_int err = CL_SUCCESS;
  cl_context ctx = cl_ctx.context();
  const int oc4 = (C_out + 3) / 4;

  cl_image_format fmt{};
  fmt.image_channel_order     = CL_RGBA;
  fmt.image_channel_data_type = CL_HALF_FLOAT;
  cl_image_desc desc{};
  desc.image_type    = CL_MEM_OBJECT_IMAGE2D;
  desc.image_width   = (size_t)CinK;
  desc.image_height  = (size_t)oc4;
  desc.image_depth   = 1;
  desc.image_row_pitch = 0;
  desc.num_mip_levels = 0;
  desc.num_samples   = 0;
  desc.buffer        = nullptr;
  // CL_MEM_READ_WRITE because the packing kernel writes via write_imageh
  // and the compute kernel reads via read_imageh.
  cl_mem img = clCreateImage(ctx, CL_MEM_READ_WRITE, &fmt, &desc, nullptr, &err);
  if (err != CL_SUCCESS || !img) {
    NNOPT_ERROR_FMT("get_weight_image: clCreateImage (CinK=%d oc4=%d) (%d)", CinK, oc4, (int)err);
    return nullptr;
  }

  cl_program prog = cl_ctx.get_program("kernels/conv1d_image.cl");
  if (!prog) prog = cl_ctx.build_program_from_file("kernels/conv1d_image.cl");
  if (!prog) { clReleaseMemObject(img); return nullptr; }
  static cl_kernel kpack = nullptr;
  if (!kpack) {
    kpack = clCreateKernel(prog, "pack_weight_h4", &err);
    if (err != CL_SUCCESS || !kpack) {
      NNOPT_ERROR_FMT("get_weight_image: clCreateKernel(pack_weight_h4) (%d)", (int)err);
      clReleaseMemObject(img);
      return nullptr;
    }
  }
  bool ok = true;
  ok = ok && set_arg_checked(kpack, 0, sizeof(cl_mem), &w_buf, "w_buf");
  ok = ok && set_arg_checked(kpack, 1, sizeof(cl_mem), &img, "w_img");
  ok = ok && set_arg_checked(kpack, 2, sizeof(int), &C_out, "C_out");
  ok = ok && set_arg_checked(kpack, 3, sizeof(int), &CinK, "CinK");
  if (!ok) { clReleaseMemObject(img); return nullptr; }
  const size_t gws[2] = {(size_t)CinK, (size_t)oc4};
  err = clEnqueueNDRangeKernel(queue, kpack, 2, nullptr, gws, nullptr, 0, nullptr, nullptr);
  if (err != CL_SUCCESS) {
    NNOPT_ERROR_FMT("get_weight_image: dispatch pack_weight_h4 (%d)", (int)err);
    clReleaseMemObject(img);
    return nullptr;
  }
  // Block until the pack kernel completes so subsequent reads see committed
  // image data (Adreno is generally fine without this on an in-order queue,
  // but the upload is one-shot per weight so we don't care about the cost).
  clFinish(queue);

  g_weight_image_cache[w_buf] = ImageWeight{img, C_out, CinK};
  return img;
}

static cl_mem conv1d_image(OpenCLContext& cl_ctx,
                           cl_command_queue queue,
                           cl_mem in, cl_mem w, cl_mem b,
                           int C_in, int C_out, int L_in,
                           int K, int padding, int dilation,
                           bool has_bias,
                           bool leaky_in,
                           cl_mem residual_add,
                           const char* label) {
  cl_int err = CL_SUCCESS;
  cl_context ctx = cl_ctx.context();

  // stride is implicit 1 on this path. Caller guarantees this.
  const int L_out = L_in + 2 * padding - dilation * (K - 1);
  if (L_out <= 0) {
    NNOPT_ERROR_FMT("conv1d_image(%s): L_out=%d", label, L_out);
    return nullptr;
  }

  cl_mem w_img = get_weight_image(cl_ctx, queue, w, C_out, C_in, K);
  if (!w_img) return nullptr;

  const size_t out_bytes = (size_t)C_out * (size_t)L_out * sizeof(nnopt_storage_t);
  cl_mem out = pool_acquire(ctx, out_bytes);
  if (!out) {
    NNOPT_ERROR_FMT("conv1d_image(%s): pool_acquire(%zu)", label, out_bytes);
    return nullptr;
  }

  cl_program prog = cl_ctx.get_program("kernels/conv1d_image.cl");
  if (!prog) prog = cl_ctx.build_program_from_file("kernels/conv1d_image.cl");
  if (!prog) { pool_release_or_release(out); return nullptr; }
  static cl_kernel kconv = nullptr;
  if (!kconv) {
    kconv = clCreateKernel(prog, "conv1d_image_tiled_h4", &err);
    if (err != CL_SUCCESS || !kconv) {
      NNOPT_ERROR_FMT("conv1d_image(%s): clCreateKernel(conv1d_image_tiled_h4) (%d)", label, (int)err);
      pool_release_or_release(out);
      return nullptr;
    }
  }

  // bias / residual can be nullptr — pass `in` itself as harmless placeholder;
  // the kernel won't touch it because the `has_bias`/`has_resid` flags gate
  // the load. (OpenCL forbids passing NULL for __global pointer args.)
  cl_mem bias_arg  = (has_bias  && b)             ? b             : in;
  cl_mem resid_arg = (residual_add)               ? residual_add  : in;
  const int has_bias_i  = (has_bias  && b)             ? 1 : 0;
  const int has_resid_i = (residual_add != nullptr)    ? 1 : 0;
  const int leaky_in_i  = leaky_in ? 1 : 0;

  bool ok = true;
  ok = ok && set_arg_checked(kconv,  0, sizeof(cl_mem), &in,       "in");
  ok = ok && set_arg_checked(kconv,  1, sizeof(cl_mem), &w_img,    "w_img");
  ok = ok && set_arg_checked(kconv,  2, sizeof(cl_mem), &bias_arg, "bias");
  ok = ok && set_arg_checked(kconv,  3, sizeof(cl_mem), &resid_arg,"resid");
  ok = ok && set_arg_checked(kconv,  4, sizeof(cl_mem), &out,      "out");
  ok = ok && set_arg_checked(kconv,  5, sizeof(int),    &C_in,     "C_in");
  ok = ok && set_arg_checked(kconv,  6, sizeof(int),    &C_out,    "C_out");
  ok = ok && set_arg_checked(kconv,  7, sizeof(int),    &L_in,     "L_in");
  ok = ok && set_arg_checked(kconv,  8, sizeof(int),    &L_out,    "L_out");
  ok = ok && set_arg_checked(kconv,  9, sizeof(int),    &K,        "K");
  ok = ok && set_arg_checked(kconv, 10, sizeof(int),    &padding,  "padding");
  ok = ok && set_arg_checked(kconv, 11, sizeof(int),    &dilation, "dilation");
  ok = ok && set_arg_checked(kconv, 12, sizeof(int),    &has_bias_i,  "has_bias");
  ok = ok && set_arg_checked(kconv, 13, sizeof(int),    &has_resid_i, "has_resid");
  ok = ok && set_arg_checked(kconv, 14, sizeof(int),    &leaky_in_i,  "leaky_in");
  if (!ok) { pool_release_or_release(out); return nullptr; }

  const int oc4 = (C_out + 3) / 4;
  // Tiled kernel: each workitem covers 8 output timesteps + 4 output channels.
  const int TN = 8;
  const int ol_tiles = (L_out + TN - 1) / TN;
  const size_t gws[2] = {(size_t)ol_tiles, (size_t)oc4};
  err = clEnqueueNDRangeKernel(queue, kconv, 2, nullptr, gws, nullptr, 0, nullptr,
                               KernelProfiler::event_for(label));
  if (err != CL_SUCCESS) {
    NNOPT_ERROR_FMT("conv1d_image(%s): dispatch (%d)", label, (int)err);
    pool_release_or_release(out);
    return nullptr;
  }
  return out;
}

// Forward-declared full-form for fusion (leaky on input, residual add on output).
static cl_mem conv1d_gemm_fused(OpenCLContext& cl_ctx,
                                cl_command_queue queue,
                                cl_mem in, cl_mem w, cl_mem b,
                                int C_in, int C_out, int L_in,
                                int K, int stride, int padding, int dilation,
                                bool has_bias,
                                bool leaky_in, float leaky_slope,
                                cl_mem residual_add,   // may be nullptr
                                const char* label);

static cl_mem conv1d_gemm(OpenCLContext& cl_ctx,
                          cl_command_queue queue,
                          cl_mem in,
                          cl_mem w,
                          cl_mem b,
                          int C_in,
                          int C_out,
                          int L_in,
                          int K,
                          int stride,
                          int padding,
                          int dilation,
                          bool has_bias,
                          const char* label) {
  return conv1d_gemm_fused(cl_ctx, queue, in, w, b, C_in, C_out, L_in,
                            K, stride, padding, dilation, has_bias,
                            /*leaky_in=*/false, 0.0f, /*residual=*/nullptr, label);
}

// Use CLBlast Convgemm (direct conv, no separate im2col dispatch).
// Set NNOPT_CONV_CONVGEMM=0 to fall back to im2col+HGemm path.
static cl_mem conv1d_via_convgemm(OpenCLContext& cl_ctx,
                                  cl_command_queue queue,
                                  cl_mem in, cl_mem w,
                                  int C_in, int C_out, int L_in,
                                  int K, int stride, int padding, int dilation,
                                  const char* label) {
  const int L_out = (L_in + 2 * padding - dilation * (K - 1) - 1) / stride + 1;
  if (L_out <= 0) {
    NNOPT_ERROR_FMT("conv1d_via_convgemm(%s): L_out=%d", label, L_out);
    return nullptr;
  }
  cl_int err = CL_SUCCESS;
  cl_context ctx = cl_ctx.context();
  const size_t n_out = (size_t)C_out * (size_t)L_out;
  cl_mem out = clCreateBuffer(ctx, CL_MEM_READ_WRITE,
                              n_out * sizeof(nnopt_storage_t), nullptr, &err);
  if (err != CL_SUCCESS || !out) {
    NNOPT_ERROR_FMT("conv1d_via_convgemm(%s): out (%d)", label, (int)err);
    return nullptr;
  }
  // 1D as 2D with height=1
#ifdef NNOPT_USE_FP16
  auto status = clblast::Convgemm<cl_half>(
      clblast::KernelMode::kCrossCorrelation,
      /*channels=*/(size_t)C_in,
      /*height=*/  1,
      /*width=*/   (size_t)L_in,
      /*kernel_h=*/1,
      /*kernel_w=*/(size_t)K,
      /*pad_h=*/   0,
      /*pad_w=*/   (size_t)padding,
      /*stride_h=*/1,
      /*stride_w=*/(size_t)stride,
      /*dilation_h=*/1,
      /*dilation_w=*/(size_t)dilation,
      /*num_kernels=*/(size_t)C_out,
      /*batch_count=*/1,
      in,  0,
      w,   0,
      out, 0,
      &queue,
      KernelProfiler::event_for("conv1d_convgemm.hgemm"));
#else
  auto status = clblast::Convgemm<float>(
      clblast::KernelMode::kCrossCorrelation,
      (size_t)C_in, 1, (size_t)L_in, 1, (size_t)K,
      0, (size_t)padding, 1, (size_t)stride, 1, (size_t)dilation,
      (size_t)C_out, 1,
      in, 0, w, 0, out, 0,
      &queue, KernelProfiler::event_for("conv1d_convgemm.sgemm"));
#endif
  if (status != clblast::StatusCode::kSuccess) {
    NNOPT_ERROR_FMT("conv1d_via_convgemm(%s): Convgemm status=%d", label, (int)status);
    clReleaseMemObject(out);
    return nullptr;
  }
  return out;
}

// Step C — Fused im2col + bias-init path.
// Eliminates the trailing add_bias / add_bias_broadcast_resid dispatch by
// pre-filling the HGemm output buffer C with bias_broadcast(+resid) inside
// the im2col kernel itself, then calling HGemm with beta=1 (so HGemm computes
// C = alpha*A*B + 1*C  where C had bias_init pre-baked).
//
// 3 dispatches per conv (im2col + HGemm + bias_add) → 2 (fused_im2col + HGemm).
// Across 77 conv1d_gemm calls per inference × ~70 ms per-dispatch gap on Adreno
// 620 (kernel-busy fraction 4.5 % baseline) ≈ 5 s wall savings.
//
// Adreno guide §8.1 (p.63): "Fuse multiple kernels into one kernel … if data
// traffic can be reduced with good parallelization."
//
// Returns out cl_mem on success; nullptr on any failure (caller may fall back
// to the slow path). Precondition (caller checks): has_bias && b && C_in >= C_out.
static cl_mem conv1d_gemm_fused_bias_path(OpenCLContext& cl_ctx,
                                          cl_command_queue queue,
                                          cl_mem in, cl_mem w, cl_mem b,
                                          int C_in, int C_out, int L_in, int L_out,
                                          int K, int stride, int padding, int dilation,
                                          bool leaky_in,
                                          cl_mem residual_add,
                                          const char* label) {
  cl_int err = CL_SUCCESS;
  cl_context ctx = cl_ctx.context();

  // Reuse the im2col workspace from the slow path (same static, same growth rule).
  const size_t im2col_n = (size_t)C_in * (size_t)K * (size_t)L_out;
  static cl_mem g_im2col_ws_fused = nullptr;
  static size_t g_im2col_ws_fused_n = 0;
  if (im2col_n > g_im2col_ws_fused_n) {
    if (g_im2col_ws_fused) { clReleaseMemObject(g_im2col_ws_fused); g_im2col_ws_fused = nullptr; }
    g_im2col_ws_fused = clCreateBuffer(ctx, CL_MEM_READ_WRITE,
                                       im2col_n * sizeof(nnopt_storage_t), nullptr, &err);
    if (err != CL_SUCCESS || !g_im2col_ws_fused) {
      NNOPT_ERROR_FMT("conv1d_gemm_fused(%s): im2col ws (%d)", label, (int)err);
      g_im2col_ws_fused_n = 0;
      return nullptr;
    }
    g_im2col_ws_fused_n = im2col_n;
  }
  cl_mem im2col = g_im2col_ws_fused;

  cl_program prog = cl_ctx.get_program("kernels/im2col.cl");
  if (!prog) prog = cl_ctx.build_program_from_file("kernels/im2col.cl");
  if (!prog) return nullptr;

  static cl_kernel kim_fused = nullptr;
  if (!kim_fused) {
    kim_fused = clCreateKernel(prog, "im2col_1d_fused_bias", &err);
    if (err != CL_SUCCESS || !kim_fused) {
      NNOPT_ERROR_FMT("conv1d_gemm_fused(%s): kernel im2col_1d_fused_bias (%d)", label, (int)err);
      return nullptr;
    }
  }
  // Dummy resid buffer used when residual_add is null. The kernel's has_resid
  // gate short-circuits before reading, but Adreno's driver is happier with
  // a non-null cl_mem arg than with NULL.
  static cl_mem g_dummy_resid = nullptr;
  if (!g_dummy_resid) {
    g_dummy_resid = clCreateBuffer(ctx, CL_MEM_READ_WRITE, 16, nullptr, &err);
    if (err != CL_SUCCESS || !g_dummy_resid) {
      NNOPT_ERROR_FMT("conv1d_gemm_fused(%s): dummy resid (%d)", label, (int)err);
      return nullptr;
    }
  }

  const size_t n_out = (size_t)C_out * (size_t)L_out;
  const size_t out_bytes = n_out * sizeof(nnopt_storage_t);
  cl_mem out = pool_acquire(ctx, out_bytes);
  if (!out) {
    NNOPT_ERROR_FMT("conv1d_gemm_fused(%s): pool_acquire(%zu)", label, out_bytes);
    return nullptr;
  }

  cl_mem resid_arg = residual_add ? residual_add : g_dummy_resid;
  const int leaky_flag    = leaky_in       ? 1 : 0;
  const int has_bias_flag = 1;
  const int has_resid_flag = residual_add  ? 1 : 0;

  bool ok = true;
  ok = ok && set_arg_checked(kim_fused,  0, sizeof(cl_mem), &in,             "in");
  ok = ok && set_arg_checked(kim_fused,  1, sizeof(cl_mem), &im2col,         "out_im2col");
  ok = ok && set_arg_checked(kim_fused,  2, sizeof(cl_mem), &out,            "out_C");
  ok = ok && set_arg_checked(kim_fused,  3, sizeof(cl_mem), &b,              "bias");
  ok = ok && set_arg_checked(kim_fused,  4, sizeof(cl_mem), &resid_arg,      "resid");
  ok = ok && set_arg_checked(kim_fused,  5, sizeof(int),    &C_in,           "C_in");
  ok = ok && set_arg_checked(kim_fused,  6, sizeof(int),    &C_out,          "C_out");
  ok = ok && set_arg_checked(kim_fused,  7, sizeof(int),    &L_in,           "L_in");
  ok = ok && set_arg_checked(kim_fused,  8, sizeof(int),    &L_out,          "L_out");
  ok = ok && set_arg_checked(kim_fused,  9, sizeof(int),    &K,              "K");
  ok = ok && set_arg_checked(kim_fused, 10, sizeof(int),    &stride,         "stride");
  ok = ok && set_arg_checked(kim_fused, 11, sizeof(int),    &padding,        "padding");
  ok = ok && set_arg_checked(kim_fused, 12, sizeof(int),    &dilation,       "dilation");
  ok = ok && set_arg_checked(kim_fused, 13, sizeof(int),    &leaky_flag,     "leaky_in");
  ok = ok && set_arg_checked(kim_fused, 14, sizeof(int),    &has_bias_flag,  "has_bias");
  ok = ok && set_arg_checked(kim_fused, 15, sizeof(int),    &has_resid_flag, "has_resid");
  if (!ok) { pool_release_or_release(out); return nullptr; }

  // Per Adreno guide §4.5.1: clFlush forces the driver to submit batched
  // kernels to the GPU instead of waiting for its internal queue threshold
  // (which adds ~75 ms QUEUED→START delay per call on Adreno 620). We flush
  // both before and after the im2col so the previous batch reaches the GPU
  // before HGemm is enqueued. NNOPT_VOC_FLUSH_FINE=0 disables.
  static int s_flush_fine = -1;
  if (s_flush_fine < 0) {
    const char* e = std::getenv("NNOPT_VOC_FLUSH_FINE");
    s_flush_fine = (e && e[0] == '1') ? 1 : 0;  // default OFF: measured slightly slower
  }
  if (s_flush_fine) clFlush(queue);

  const size_t gws_im[3] = {(size_t)L_out, (size_t)K, (size_t)C_in};
  err = clEnqueueNDRangeKernel(queue, kim_fused, 3, nullptr, gws_im, nullptr, 0, nullptr,
                               KernelProfiler::event_for(leaky_in ? "conv1d_gemm.leaky_im2col" : "conv1d_gemm.im2col"));
  if (err != CL_SUCCESS) {
    NNOPT_ERROR_FMT("conv1d_gemm_fused(%s): fused im2col dispatch (%d)", label, (int)err);
    pool_release_or_release(out);
    return nullptr;
  }
  if (s_flush_fine) clFlush(queue);

  // HGemm with beta=1: C = alpha*A*B + 1*C  where C had bias_broadcast(+resid)
  // pre-filled by the fused kernel above. Equivalent to running a separate
  // add_bias_broadcast(_resid) kernel post-GEMM, but in zero dispatches.
  const int gemm_M = C_out;
  const int gemm_K = C_in * K;
  const int gemm_N = L_out;
#ifdef NNOPT_USE_FP16
  cl_half h_one = static_cast<cl_half>(nnopt_f32_to_f16(1.0f));
  auto status = clblast::Gemm<cl_half>(
      clblast::Layout::kRowMajor,
      clblast::Transpose::kNo,
      clblast::Transpose::kNo,
      gemm_M, gemm_N, gemm_K,
      h_one,
      w,      0, gemm_K,
      im2col, 0, gemm_N,
      h_one,    // beta = 1 (preserves the bias-init pre-fill)
      out,    0, gemm_N,
      &queue,
      KernelProfiler::event_for("conv1d_gemm.hgemm"));
  (void)status;
#else
  auto status = clblast::Gemm<float>(
      clblast::Layout::kRowMajor,
      clblast::Transpose::kNo,
      clblast::Transpose::kNo,
      gemm_M, gemm_N, gemm_K,
      1.0f,
      w,      0, gemm_K,
      im2col, 0, gemm_N,
      1.0f,
      out,    0, gemm_N,
      &queue,
      KernelProfiler::event_for("conv1d_gemm.sgemm"));
#endif
  if (status != clblast::StatusCode::kSuccess) {
    NNOPT_ERROR_FMT("conv1d_gemm_fused(%s): clblast::Gemm beta=1 status=%d M=%d N=%d K=%d",
                    label, (int)status, gemm_M, gemm_N, gemm_K);
    pool_release_or_release(out);
    return nullptr;
  }
  return out;
}

static cl_mem conv1d_gemm_impl(OpenCLContext& cl_ctx,
                          cl_command_queue queue,
                          cl_mem in,
                          cl_mem w,
                          cl_mem b,
                          int C_in,
                          int C_out,
                          int L_in,
                          int K,
                          int stride,
                          int padding,
                          int dilation,
                          bool has_bias,
                          bool leaky_in,
                          float leaky_slope,
                          cl_mem residual_add,
                          const char* label) {
  // Path 1 — image2d-resident weights (Qualcomm OpenCL guide §6.2 / §9.3.1).
  // OPT-IN via NNOPT_CONV_IMAGE=1. Both the naive and the tiled image-conv
  // kernels we tried were SLOWER than CLBlast HGemm on Adreno 620 (naive:
  // ~7× slower; tiled TN=8: ~9× slower, register spills). CLBlast HGemm has
  // local-memory tile staging + Adreno-specific scheduling we can't trivially
  // match. The image-path infrastructure (weight packing, cache, dispatch) is
  // left in place so future work can build on it. Falls back to im2col+HGemm
  // on any failure path so production is never regressed.
  static int s_use_image = -1;
  if (s_use_image < 0) {
    const char* env = std::getenv("NNOPT_CONV_IMAGE");
    s_use_image = (env && env[0] == '1') ? 1 : 0;
  }
  if (s_use_image && stride == 1) {
    cl_mem y = conv1d_image(cl_ctx, queue, in, w, b,
                            C_in, C_out, L_in, K, padding, dilation,
                            has_bias, leaky_in, residual_add, label);
    if (y) return y;
    // Image path failed — fall through to im2col+HGemm.
  }

  // OFF by default: CLBlast Convgemm is ~100× slower than im2col+HGemm on
  // Adreno 620 (CLBlast 1.6.x has no Adreno-tuned conv kernels). The
  // im2col + HGemm path stays primary. Set NNOPT_CONV_CONVGEMM=1 to try.
  static int s_use_convgemm = -1;
  if (s_use_convgemm < 0) {
    const char* env = std::getenv("NNOPT_CONV_CONVGEMM");
    s_use_convgemm = (env && env[0] == '1') ? 1 : 0;
  }
  if (s_use_convgemm && !leaky_in && !residual_add && stride == 1) {
    cl_mem y = conv1d_via_convgemm(cl_ctx, queue, in, w,
                                   C_in, C_out, L_in, K, stride, padding, dilation, label);
    if (y) {
      // Optional bias add post-Convgemm. One dispatch instead of (im2col+HGemm+bias) = 3.
      if (has_bias && b) {
        cl_program prog = cl_ctx.get_program("kernels/im2col.cl");
        if (!prog) prog = cl_ctx.build_program_from_file("kernels/im2col.cl");
        if (!prog) { clReleaseMemObject(y); return nullptr; }
        static cl_kernel kb = nullptr;
        cl_int kerr = CL_SUCCESS;
        if (!kb) {
          kb = clCreateKernel(prog, "add_bias_broadcast", &kerr);
          if (kerr != CL_SUCCESS || !kb) { clReleaseMemObject(y); return nullptr; }
        }
        const int L_out = (L_in + 2 * padding - dilation * (K - 1) - 1) / stride + 1;
        const int n_out_total = C_out * L_out;
        bool ok = true;
        ok = ok && set_arg_checked(kb, 0, sizeof(cl_mem), &y, "y");
        ok = ok && set_arg_checked(kb, 1, sizeof(cl_mem), &b, "bias");
        ok = ok && set_arg_checked(kb, 2, sizeof(int), &C_out, "C_out");
        ok = ok && set_arg_checked(kb, 3, sizeof(int), &L_out, "L_out");
        if (!ok) { clReleaseMemObject(y); return nullptr; }
        const size_t gws[1] = {(size_t)n_out_total};
        kerr = clEnqueueNDRangeKernel(queue, kb, 1, nullptr, gws, nullptr, 0, nullptr,
                                      KernelProfiler::event_for("conv1d_convgemm.bias"));
        if (kerr != CL_SUCCESS) { clReleaseMemObject(y); return nullptr; }
      }
      return y;
    }
    // Convgemm failed — fall through to im2col path below.
  }
  const int L_out = (L_in + 2 * padding - dilation * (K - 1) - 1) / stride + 1;
  if (L_out <= 0) {
    NNOPT_ERROR_FMT("conv1d_gemm(%s): L_out=%d", label, L_out);
    return nullptr;
  }

  // Step C — fused im2col + bias-init path. When has_bias is true AND
  // C_in >= C_out (true for every resblock conv where C_in==C_out==C, and
  // for the upsample-output convs where C_in > C_out), we skip the separate
  // bias kernel by pre-filling the HGemm output buffer inside the im2col
  // kernel and using HGemm beta=1. Saves 1 dispatch per call × ~70 ms Adreno
  // dispatch gap. Set NNOPT_VOC_FUSED_BIAS=0 to revert to the slow path.
  static int s_fused_bias = -1;
  if (s_fused_bias < 0) {
    const char* env = std::getenv("NNOPT_VOC_FUSED_BIAS");
    s_fused_bias = (env && env[0] == '0') ? 0 : 1;
  }
  if (s_fused_bias && has_bias && b && C_in >= C_out) {
    cl_mem y = conv1d_gemm_fused_bias_path(cl_ctx, queue, in, w, b,
                                           C_in, C_out, L_in, L_out,
                                           K, stride, padding, dilation,
                                           leaky_in, residual_add, label);
    if (y) return y;
    // fused path returned nullptr — fall through to the slow path below
  }

  cl_int err = CL_SUCCESS;
  cl_context ctx = cl_ctx.context();

  // im2col buffer — reused across calls. Grown lazily.
  const size_t im2col_n = (size_t)C_in * (size_t)K * (size_t)L_out;
  static cl_mem g_im2col_ws = nullptr;
  static size_t g_im2col_ws_n = 0;
  if (im2col_n > g_im2col_ws_n) {
    if (g_im2col_ws) { clReleaseMemObject(g_im2col_ws); g_im2col_ws = nullptr; }
    g_im2col_ws = clCreateBuffer(ctx, CL_MEM_READ_WRITE,
                                 im2col_n * sizeof(nnopt_storage_t), nullptr, &err);
    if (err != CL_SUCCESS || !g_im2col_ws) {
      NNOPT_ERROR_FMT("conv1d_gemm(%s): im2col workspace (%d)", label, (int)err);
      g_im2col_ws_n = 0;
      return nullptr;
    }
    g_im2col_ws_n = im2col_n;
  }
  cl_mem im2col = g_im2col_ws;
  cl_program prog = cl_ctx.get_program("kernels/im2col.cl");
  if (!prog) prog = cl_ctx.build_program_from_file("kernels/im2col.cl");
  if (!prog) {
    NNOPT_ERROR_FMT("conv1d_gemm(%s): build im2col.cl failed", label);
    return nullptr;
  }

  // Pick im2col kernel: plain or fused-with-leaky-relu.
  static cl_kernel kim_plain = nullptr;
  static cl_kernel kim_leaky = nullptr;
  cl_kernel kim;
  if (leaky_in) {
    if (!kim_leaky) {
      kim_leaky = clCreateKernel(prog, "leaky_im2col_1d", &err);
      if (err != CL_SUCCESS || !kim_leaky) {
        NNOPT_ERROR_FMT("conv1d_gemm(%s): kernel leaky_im2col_1d (%d)", label, (int)err);
        return nullptr;
      }
    }
    kim = kim_leaky;
  } else {
    if (!kim_plain) {
      kim_plain = clCreateKernel(prog, "im2col_1d", &err);
      if (err != CL_SUCCESS || !kim_plain) {
        NNOPT_ERROR_FMT("conv1d_gemm(%s): kernel im2col_1d (%d)", label, (int)err);
        return nullptr;
      }
    }
    kim = kim_plain;
  }
  bool ok = true;
  ok = ok && set_arg_checked(kim, 0, sizeof(cl_mem), &in, "in");
  ok = ok && set_arg_checked(kim, 1, sizeof(cl_mem), &im2col, "out");
  ok = ok && set_arg_checked(kim, 2, sizeof(int), &C_in, "C_in");
  ok = ok && set_arg_checked(kim, 3, sizeof(int), &L_in, "L_in");
  ok = ok && set_arg_checked(kim, 4, sizeof(int), &L_out, "L_out");
  ok = ok && set_arg_checked(kim, 5, sizeof(int), &K, "K");
  ok = ok && set_arg_checked(kim, 6, sizeof(int), &stride, "stride");
  ok = ok && set_arg_checked(kim, 7, sizeof(int), &padding, "padding");
  ok = ok && set_arg_checked(kim, 8, sizeof(int), &dilation, "dilation");
  (void)leaky_slope;  // slope is now hardcoded inside leaky_im2col_1d kernel
  if (!ok) return nullptr;
  // Step C #9 + #15: 3D NDRange so kernel reads col/k/ic from get_global_id()
  // directly, eliminating 4 expensive integer divides/modulos per workitem
  // (Adreno guide §8.12 p.69). Same total workitem count = C_in × K × L_out.
  const size_t gws_im[3] = {(size_t)L_out, (size_t)K, (size_t)C_in};
  err = clEnqueueNDRangeKernel(queue, kim, 3, nullptr, gws_im, nullptr, 0, nullptr,
                               KernelProfiler::event_for(leaky_in ? "conv1d_gemm.leaky_im2col" : "conv1d_gemm.im2col"));
  if (err != CL_SUCCESS) {
    NNOPT_ERROR_FMT("conv1d_gemm(%s): im2col dispatch (%d)", label, (int)err);
    return nullptr;
  }

  // Output buffer from the recycling pool — avoids the per-call clCreateBuffer
  // round-trip through the Adreno driver. Caller is responsible for releasing
  // via pool_release_t (or clReleaseMemObject, but pool is faster).
  const size_t n_out = (size_t)C_out * (size_t)L_out;
  const size_t out_bytes = n_out * sizeof(nnopt_storage_t);
  cl_mem out = pool_acquire(ctx, out_bytes);
  if (!out) {
    NNOPT_ERROR_FMT("conv1d_gemm(%s): pool_acquire(%zu)", label, out_bytes);
    return nullptr;
  }

  // GEMM: out[C_out, L_out] = W[C_out, C_in*K] @ im2col[C_in*K, L_out]
  // CLBlast RowMajor:  C[M,N] = alpha * A[M,K] * B[K,N] + beta * C[M,N]
  //   M = C_out, K = C_in*K, N = L_out
  //   lda = C_in*K  (W stride between rows)
  //   ldb = L_out    (im2col stride between rows)
  //   ldc = L_out    (out stride between rows)
  const int gemm_M = C_out;
  const int gemm_K = C_in * K;
  const int gemm_N = L_out;

#ifdef NNOPT_USE_FP16
  cl_half h_one  = static_cast<cl_half>(nnopt_f32_to_f16(1.0f));
  cl_half h_zero = static_cast<cl_half>(nnopt_f32_to_f16(0.0f));
  auto status = clblast::Gemm<cl_half>(
      clblast::Layout::kRowMajor,
      clblast::Transpose::kNo,
      clblast::Transpose::kNo,
      gemm_M, gemm_N, gemm_K,
      h_one,
      w,      0, gemm_K,
      im2col, 0, gemm_N,
      h_zero,
      out,    0, gemm_N,
      &queue,
      KernelProfiler::event_for("conv1d_gemm.hgemm"));
  (void)status;
#else
  auto status = clblast::Gemm<float>(
      clblast::Layout::kRowMajor,
      clblast::Transpose::kNo,
      clblast::Transpose::kNo,
      gemm_M, gemm_N, gemm_K,
      1.0f,
      w,      0, gemm_K,
      im2col, 0, gemm_N,
      0.0f,
      out,    0, gemm_N,
      &queue,
      KernelProfiler::event_for("conv1d_gemm.sgemm"));
#endif
  if (status != clblast::StatusCode::kSuccess) {
    NNOPT_ERROR_FMT("conv1d_gemm(%s): clblast::Gemm status=%d M=%d N=%d K=%d",
                    label, (int)status, gemm_M, gemm_N, gemm_K);
    pool_release_or_release(out);
    return nullptr;
  }

  if (has_bias && b) {
    static cl_kernel kb_plain = nullptr;
    static cl_kernel kb_resid = nullptr;
    cl_kernel kb;
    const bool fused_resid = (residual_add != nullptr);
    if (fused_resid) {
      if (!kb_resid) {
        kb_resid = clCreateKernel(prog, "add_bias_broadcast_resid", &err);
        if (err != CL_SUCCESS || !kb_resid) {
          NNOPT_ERROR_FMT("conv1d_gemm(%s): kernel add_bias_broadcast_resid (%d)", label, (int)err);
          pool_release_or_release(out);
          return nullptr;
        }
      }
      kb = kb_resid;
    } else {
      if (!kb_plain) {
        kb_plain = clCreateKernel(prog, "add_bias_broadcast", &err);
        if (err != CL_SUCCESS || !kb_plain) {
          NNOPT_ERROR_FMT("conv1d_gemm(%s): kernel add_bias_broadcast (%d)", label, (int)err);
          pool_release_or_release(out);
          return nullptr;
        }
      }
      kb = kb_plain;
    }
    bool ok2 = true;
    ok2 = ok2 && set_arg_checked(kb, 0, sizeof(cl_mem), &out, "y");
    ok2 = ok2 && set_arg_checked(kb, 1, sizeof(cl_mem), &b, "bias");
    int idx = 2;
    if (fused_resid) {
      ok2 = ok2 && set_arg_checked(kb, idx++, sizeof(cl_mem), &residual_add, "resid");
    }
    ok2 = ok2 && set_arg_checked(kb, idx++, sizeof(int), &C_out, "C_out");
    ok2 = ok2 && set_arg_checked(kb, idx++, sizeof(int), &L_out, "L_out");
    if (!ok2) { pool_release_or_release(out); return nullptr; }
    const size_t gws_b[1] = {n_out};
    err = clEnqueueNDRangeKernel(queue, kb, 1, nullptr, gws_b, nullptr, 0, nullptr,
                                 KernelProfiler::event_for(fused_resid ? "conv1d_gemm.bias_resid" : "conv1d_gemm.bias"));
    if (err != CL_SUCCESS) {
      NNOPT_ERROR_FMT("conv1d_gemm(%s): bias dispatch (%d)", label, (int)err);
      pool_release_or_release(out);
      return nullptr;
    }
  } else if (residual_add) {
    // No bias but still need residual add. Use elementwise add via the
    // existing kernel.
    cl_program act_prog = cl_ctx.get_program("kernels/hifigan_residual_block.cl");
    if (!act_prog) act_prog = cl_ctx.build_program_from_file("kernels/hifigan_residual_block.cl");
    if (!act_prog) { pool_release_or_release(out); return nullptr; }
    static cl_kernel kadd = nullptr;
    if (!kadd) {
      kadd = clCreateKernel(act_prog, "elementwise_add", &err);
      if (err != CL_SUCCESS || !kadd) {
        pool_release_or_release(out);
        return nullptr;
      }
    }
    cl_mem y_in = out;
    cl_mem out2 = pool_acquire(ctx, out_bytes);
    if (!out2) { pool_release_or_release(out); return nullptr; }
    const int N = (int)n_out;
    bool ok2 = true;
    ok2 = ok2 && set_arg_checked(kadd, 0, sizeof(cl_mem), &y_in, "a");
    ok2 = ok2 && set_arg_checked(kadd, 1, sizeof(cl_mem), &residual_add, "b");
    ok2 = ok2 && set_arg_checked(kadd, 2, sizeof(cl_mem), &out2, "out");
    ok2 = ok2 && set_arg_checked(kadd, 3, sizeof(int), &N, "N");
    if (!ok2) { pool_release_or_release(out); pool_release_or_release(out2); return nullptr; }
    const size_t gws_b[1] = {n_out};
    err = clEnqueueNDRangeKernel(queue, kadd, 1, nullptr, gws_b, nullptr, 0, nullptr,
                                 KernelProfiler::event_for("conv1d_gemm.resid_only"));
    pool_release_or_release(out);
    out = out2;
    if (err != CL_SUCCESS) { pool_release_or_release(out); return nullptr; }
  }
  return out;
}

static cl_mem conv1d_gemm_fused(OpenCLContext& cl_ctx,
                                cl_command_queue queue,
                                cl_mem in, cl_mem w, cl_mem b,
                                int C_in, int C_out, int L_in,
                                int K, int stride, int padding, int dilation,
                                bool has_bias,
                                bool leaky_in, float leaky_slope,
                                cl_mem residual_add,
                                const char* label) {
  return conv1d_gemm_impl(cl_ctx, queue, in, w, b, C_in, C_out, L_in,
                          K, stride, padding, dilation, has_bias,
                          leaky_in, leaky_slope, residual_add, label);
}

// ---- Phase 1 mega-kernel pivot: direct fused conv1d ----
// Replaces the (leaky_im2col + HGemm + add_bias[_resid]) 3-dispatch chain with
// ONE dispatch. On Adreno 620 the per-dispatch host overhead is ~70 ms (kernel
// busy fraction during the vocoder phase was 4.5% in the Phase 0 baseline),
// so dispatch-count reduction dominates over raw kernel efficiency. Each
// resblock conv goes 3 dispatches → 1 dispatch.
static cl_mem conv1d_direct_fused(OpenCLContext& cl_ctx,
                                  cl_command_queue queue,
                                  cl_mem in, cl_mem w, cl_mem b,
                                  int C_in, int C_out, int L_in,
                                  int K, int stride, int padding, int dilation,
                                  bool has_bias,
                                  bool leaky_in, float /*leaky_slope*/,
                                  cl_mem residual_add,
                                  const char* label) {
  const int L_out = (L_in + 2 * padding - dilation * (K - 1) - 1) / stride + 1;
  if (L_out <= 0) {
    NNOPT_ERROR_FMT("conv1d_direct(%s): L_out=%d", label, L_out);
    return nullptr;
  }
  cl_program prog = cl_ctx.get_program("kernels/conv1d_direct_fused.cl");
  if (!prog) prog = cl_ctx.build_program_from_file("kernels/conv1d_direct_fused.cl");
  if (!prog) {
    NNOPT_ERROR_FMT("conv1d_direct(%s): build conv1d_direct_fused.cl failed", label);
    return nullptr;
  }
  cl_int err = CL_SUCCESS;
  static cl_kernel k_direct = nullptr;
  if (!k_direct) {
    k_direct = clCreateKernel(prog, "conv1d_direct_fused", &err);
    if (err != CL_SUCCESS || !k_direct) {
      NNOPT_ERROR_FMT("conv1d_direct(%s): kernel create (%d)", label, (int)err);
      return nullptr;
    }
  }
  // Dummy buffer for nullable args. The kernel never reads it (has_bias /
  // has_resid gates short-circuit at the LOAD), but Adreno's OpenCL driver
  // is happier with non-null cl_mem args than with NULL.
  static cl_mem g_dummy = nullptr;
  if (!g_dummy) {
    g_dummy = clCreateBuffer(cl_ctx.context(), CL_MEM_READ_WRITE, 16, nullptr, &err);
    if (err != CL_SUCCESS || !g_dummy) {
      NNOPT_ERROR_FMT("conv1d_direct(%s): dummy buffer (%d)", label, (int)err);
      return nullptr;
    }
  }
  cl_mem bias_arg  = (has_bias && b) ? b            : g_dummy;
  cl_mem resid_arg = residual_add    ? residual_add : g_dummy;

  const size_t out_bytes = (size_t)C_out * (size_t)L_out * sizeof(nnopt_storage_t);
  cl_mem out = pool_acquire(cl_ctx.context(), out_bytes);
  if (!out) {
    NNOPT_ERROR_FMT("conv1d_direct(%s): pool_acquire(%zu)", label, out_bytes);
    return nullptr;
  }

  const int leaky_flag    = leaky_in              ? 1 : 0;
  const int has_bias_flag = (has_bias && b)       ? 1 : 0;
  const int has_resid     = residual_add          ? 1 : 0;
  bool ok = true;
  ok = ok && set_arg_checked(k_direct,  0, sizeof(cl_mem), &in,            "in");
  ok = ok && set_arg_checked(k_direct,  1, sizeof(cl_mem), &w,             "weight");
  ok = ok && set_arg_checked(k_direct,  2, sizeof(cl_mem), &bias_arg,      "bias");
  ok = ok && set_arg_checked(k_direct,  3, sizeof(cl_mem), &resid_arg,     "resid");
  ok = ok && set_arg_checked(k_direct,  4, sizeof(cl_mem), &out,           "out");
  ok = ok && set_arg_checked(k_direct,  5, sizeof(int),    &C_in,          "C_in");
  ok = ok && set_arg_checked(k_direct,  6, sizeof(int),    &C_out,         "C_out");
  ok = ok && set_arg_checked(k_direct,  7, sizeof(int),    &L_in,          "L_in");
  ok = ok && set_arg_checked(k_direct,  8, sizeof(int),    &L_out,         "L_out");
  ok = ok && set_arg_checked(k_direct,  9, sizeof(int),    &K,             "K");
  ok = ok && set_arg_checked(k_direct, 10, sizeof(int),    &stride,        "stride");
  ok = ok && set_arg_checked(k_direct, 11, sizeof(int),    &padding,       "padding");
  ok = ok && set_arg_checked(k_direct, 12, sizeof(int),    &dilation,      "dilation");
  ok = ok && set_arg_checked(k_direct, 13, sizeof(int),    &leaky_flag,    "leaky_in");
  ok = ok && set_arg_checked(k_direct, 14, sizeof(int),    &has_bias_flag, "has_bias");
  ok = ok && set_arg_checked(k_direct, 15, sizeof(int),    &has_resid,     "has_resid");
  if (!ok) { pool_release_or_release(out); return nullptr; }

  const size_t gws[1] = {(size_t)C_out * (size_t)L_out};
  err = clEnqueueNDRangeKernel(queue, k_direct, 1, nullptr, gws, nullptr, 0, nullptr,
                               KernelProfiler::event_for(label));
  if (err != CL_SUCCESS) {
    NNOPT_ERROR_FMT("conv1d_direct(%s): dispatch (%d)", label, (int)err);
    pool_release_or_release(out);
    return nullptr;
  }
  return out;
}

// Dispatcher: picks conv1d_direct_fused (1 dispatch) or conv1d_gemm_fused
// (3 dispatches via im2col+HGemm+bias) based on NNOPT_VOC_DIRECT env var.
// Default is direct=ON since the Phase 0 baseline showed per-dispatch host
// overhead dominates wall time. Set NNOPT_VOC_DIRECT=0 to revert for A/B.
static cl_mem conv1d_resblock(OpenCLContext& cl_ctx,
                              cl_command_queue queue,
                              cl_mem in, cl_mem w, cl_mem b,
                              int C_in, int C_out, int L_in,
                              int K, int stride, int padding, int dilation,
                              bool has_bias,
                              bool leaky_in, float leaky_slope,
                              cl_mem residual_add,
                              const char* label) {
  static int s_mode = -1;
  if (s_mode < 0) {
    const char* env = std::getenv("NNOPT_VOC_DIRECT");
    // Default OFF: naive direct conv is ~24× slower per MAC than CLBlast HGemm
    // (achieves ~5 GFLOPS vs HGemm's ~120 GFLOPS on Adreno 620). Net loss vs
    // the GEMM path despite saving 2/3 of dispatches. Flip back ON only after
    // a tiled implementation with __local memory weight/input caching exists.
    s_mode = (env && env[0] == '1') ? 1 : 0;
  }
  if (s_mode) {
    return conv1d_direct_fused(cl_ctx, queue, in, w, b, C_in, C_out, L_in,
                               K, stride, padding, dilation, has_bias,
                               leaky_in, leaky_slope, residual_add, label);
  }
  // conv1d_image: single-dispatch fused path (leaky+conv+bias+resid in one kernel).
  // Cuts 3 dispatches/conv → 1 vs conv1d_gemm_fused, BUT empirically slower per
  // kernel for resblock shapes (small C, dilated, K=3/7/11) because the tiled
  // image kernel isn't tuned for these — CLBlast HGemm carries the compute
  // weight in the GEMM path. Default OFF here; flip ON with
  // NNOPT_VOC_RESBLOCK_IMAGE=1 if a future image-kernel rewrite catches up.
  static int s_use_image = -1;
  if (s_use_image < 0) {
    const char* env = std::getenv("NNOPT_VOC_RESBLOCK_IMAGE");
    s_use_image = (env && env[0] == '1') ? 1 : 0;
  }
  if (s_use_image && stride == 1) {
    return conv1d_image(cl_ctx, queue, in, w, b, C_in, C_out, L_in,
                        K, padding, dilation, has_bias,
                        leaky_in, residual_add, label);
  }
  cl_mem r = conv1d_gemm_fused(cl_ctx, queue, in, w, b, C_in, C_out, L_in,
                               K, stride, padding, dilation, has_bias,
                               leaky_in, leaky_slope, residual_add, label);
  // Per Qualcomm Adreno OpenCL guide §4.5.1 / Fig 4-1: "The OpenCL software
  // may queue the kernel first and submit it along with several following
  // kernels in the queue later … Developers may use the clFlush function to
  // speed up the submission." Our profile shows ~75 ms QUEUED→START on HGemm
  // calls — exactly this driver batching. clFlush once per logical conv
  // (= 3 underlying dispatches) forces the driver to kick the batch to the
  // GPU instead of waiting for its internal threshold. Default ON; disable
  // with NNOPT_VOC_FLUSH=0 for A/B.
  static int s_flush = -1;
  if (s_flush < 0) {
    const char* e = std::getenv("NNOPT_VOC_FLUSH");
    s_flush = (e && e[0] == '1') ? 1 : 0;  // default OFF: measured ~0 net effect
  }
  if (s_flush && r) clFlush(queue);
  return r;
}

static cl_mem conv1d(OpenCLContext& cl_ctx,
                     cl_command_queue queue,
                     cl_program prog,
                     cl_mem in,
                     cl_mem w,
                     cl_mem b,
                     int C_in,
                     int C_out,
                     int L_in,
                     int K,
                     int stride,
                     int padding,
                     int dilation,
                     bool has_bias,
                     const char* label) {
  // Route all vocoder convs through CLBlast GEMM when fp16 unless explicitly
  // disabled. The naive path is kept below as a fallback for debugging.
  static int s_use_gemm = -1;
  if (s_use_gemm < 0) {
    const char* env = std::getenv("NNOPT_CONV1D_NAIVE");
    s_use_gemm = (env && env[0] == '1') ? 0 : 1;
  }
  if (s_use_gemm) {
    cl_mem r = conv1d_gemm(cl_ctx, queue, in, w, b, C_in, C_out, L_in, K,
                            stride, padding, dilation, has_bias, label);
    if (r) return r;
    NNOPT_ERROR_FMT("conv1d(%s): GEMM path failed, falling back to naive", label);
  }
  const int L_out = (L_in + 2 * padding - dilation * (K - 1) - 1) / stride + 1;
  if (L_out <= 0) {
    NNOPT_ERROR_FMT("conv1d(%s): computed L_out=%d", label, L_out);
    return nullptr;
  }

  cl_int err = CL_SUCCESS;
  cl_context ctx = cl_ctx.context();
  const size_t n_out = (size_t)C_out * (size_t)L_out;
  cl_mem out = pool_acquire(ctx, n_out * sizeof(nnopt_storage_t));
  if (!out) {
    NNOPT_ERROR_FMT("conv1d(%s): pool_acquire(out) failed", label);
    return nullptr;
  }

  cl_kernel k = clCreateKernel(prog, "conv_1d", &err);
  if (err != CL_SUCCESS || !k) {
    NNOPT_ERROR_FMT("conv1d(%s): clCreateKernel(conv_1d) failed (%d)", label, (int)err);
    pool_release_or_release(out);
    return nullptr;
  }

  const int hb = has_bias ? 1 : 0;
  bool ok = true;
  ok = ok && set_arg_checked(k, 0, sizeof(cl_mem), &in, "in");
  ok = ok && set_arg_checked(k, 1, sizeof(cl_mem), &w, "weight");
  ok = ok && set_arg_checked(k, 2, sizeof(cl_mem), &b, "bias");
  ok = ok && set_arg_checked(k, 3, sizeof(cl_mem), &out, "out");
  ok = ok && set_arg_checked(k, 4, sizeof(int), &C_in, "C_in");
  ok = ok && set_arg_checked(k, 5, sizeof(int), &C_out, "C_out");
  ok = ok && set_arg_checked(k, 6, sizeof(int), &L_in, "L_in");
  ok = ok && set_arg_checked(k, 7, sizeof(int), &L_out, "L_out");
  ok = ok && set_arg_checked(k, 8, sizeof(int), &K, "K");
  ok = ok && set_arg_checked(k, 9, sizeof(int), &stride, "stride");
  ok = ok && set_arg_checked(k, 10, sizeof(int), &padding, "padding");
  ok = ok && set_arg_checked(k, 11, sizeof(int), &dilation, "dilation");
  ok = ok && set_arg_checked(k, 12, sizeof(int), &hb, "has_bias");
  if (!ok) {
    clReleaseKernel(k);
    pool_release_or_release(out);
    return nullptr;
  }

  const size_t gws[1] = {n_out};
  err = clEnqueueNDRangeKernel(queue, k, 1, nullptr, gws, nullptr, 0, nullptr,
                               KernelProfiler::event_for(label));
  clReleaseKernel(k);
  if (err != CL_SUCCESS) {
    NNOPT_ERROR_FMT("conv1d(%s): dispatch failed (%d)", label, (int)err);
    pool_release_or_release(out);
    return nullptr;
  }

  return out;
}

// Cache of reordered ConvTranspose1d weights: maps the ORIGINAL on-disk
// weight cl_mem (key) to a freshly-built Conv1d-layout weight (value).
// Lifetime is process lifetime — convT is called ~4 times per inference,
// always with the same upsampler weights.
static std::vector<std::pair<cl_mem, cl_mem>> g_convt_weight_cache;

static cl_mem get_or_make_convT_to_conv_weight(
    OpenCLContext& cl_ctx, cl_command_queue queue, cl_mem w_orig,
    int C_in, int C_out, int K) {
  for (const auto& kv : g_convt_weight_cache) {
    if (kv.first == w_orig) return kv.second;
  }
  cl_int err = CL_SUCCESS;
  const size_t nelem = (size_t)C_out * C_in * K;
  cl_mem w_new = clCreateBuffer(cl_ctx.context(), CL_MEM_READ_ONLY,
                                nelem * sizeof(nnopt_storage_t), nullptr, &err);
  if (err != CL_SUCCESS || !w_new) {
    NNOPT_ERROR_FMT("convT weight cache: clCreateBuffer failed (%d)", (int)err);
    return nullptr;
  }
  cl_program prog = cl_ctx.get_program("kernels/im2col.cl");
  if (!prog) prog = cl_ctx.build_program_from_file("kernels/im2col.cl");
  if (!prog) { pool_release_or_release(w_new); return nullptr; }
  cl_kernel k = clCreateKernel(prog, "reorder_convT_to_conv1d_weight", &err);
  if (err != CL_SUCCESS || !k) { pool_release_or_release(w_new); return nullptr; }
  bool ok = true;
  ok = ok && set_arg_checked(k, 0, sizeof(cl_mem), &w_orig, "w_in");
  ok = ok && set_arg_checked(k, 1, sizeof(cl_mem), &w_new,  "w_out");
  ok = ok && set_arg_checked(k, 2, sizeof(int), &C_in, "C_in");
  ok = ok && set_arg_checked(k, 3, sizeof(int), &C_out, "C_out");
  ok = ok && set_arg_checked(k, 4, sizeof(int), &K,    "K");
  if (!ok) { clReleaseKernel(k); pool_release_or_release(w_new); return nullptr; }
  const size_t gws[1] = {nelem};
  err = clEnqueueNDRangeKernel(queue, k, 1, nullptr, gws, nullptr, 0, nullptr, nullptr);
  clReleaseKernel(k);
  if (err != CL_SUCCESS) { pool_release_or_release(w_new); return nullptr; }
  g_convt_weight_cache.push_back({w_orig, w_new});
  return w_new;
}

// ConvTranspose1d via zero-stuff + standard Conv1d (CLBlast GEMM).
//
// PyTorch ConvTranspose1d (stride=s, kernel=K, padding=p) on x[C_in, L_in]
// is mathematically equivalent to:
//   1. zero-stuff x → x_exp[C_in, (L_in-1)*s + 1]
//      x_exp[ic, t*s] = x[ic, t]; other positions = 0
//   2. Conv1d on x_exp with stride=1, padding=K-1-p, dilation=1
//      using a weight reordered to [C_out, C_in, K] WITH kernel axis reversed.
//
// Output length: (L_in-1)*s + 1 + 2*(K-1-p) - (K-1) = (L_in-1)*s - 2p + K ✓.
//
// Why this is faster than the direct convT kernel:
//   - The direct convT has `num % stride != 0 → continue` branches that
//     serialize warp threads. After zero-stuff every output position is a
//     well-formed conv1d dot product.
//   - We can leverage the CLBlast HGemm code path that conv1d_gemm already
//     uses → tuned tile shapes, fp32 internal accumulation.
static cl_mem convt1d_gemm(OpenCLContext& cl_ctx,
                           cl_command_queue queue,
                           cl_mem in,
                           cl_mem w_orig,
                           cl_mem b,
                           int C_in,
                           int C_out,
                           int L_in,
                           int K,
                           int stride,
                           int padding,
                           int dilation,
                           bool has_bias,
                           const char* label,
                           int* out_L_out) {
  if (dilation != 1) {
    NNOPT_ERROR_FMT("convt1d_gemm(%s): dilation %d != 1 not supported", label, dilation);
    return nullptr;
  }
  const int L_exp  = (L_in - 1) * stride + 1;
  const int L_out  = (L_in - 1) * stride - 2 * padding + K;
  if (L_out <= 0) {
    NNOPT_ERROR_FMT("convt1d_gemm(%s): bad L_out=%d", label, L_out);
    return nullptr;
  }

  cl_int err = CL_SUCCESS;
  cl_context ctx = cl_ctx.context();

  // 1. Zero-stuff input — pool_acquire so the same buffer is reused across
  // the 4 ConvT1d calls instead of allocating each time.
  const size_t exp_n = (size_t)C_in * (size_t)L_exp;
  cl_mem x_exp = pool_acquire(ctx, exp_n * sizeof(nnopt_storage_t));
  if (!x_exp) {
    NNOPT_ERROR_FMT("convt1d_gemm(%s): pool_acquire(x_exp) failed", label);
    return nullptr;
  }
  cl_program prog = cl_ctx.get_program("kernels/im2col.cl");
  if (!prog) prog = cl_ctx.build_program_from_file("kernels/im2col.cl");
  if (!prog) { pool_release_or_release(x_exp); return nullptr; }
  cl_kernel kz = clCreateKernel(prog, "zero_stuff_1d", &err);
  if (err != CL_SUCCESS || !kz) {
    NNOPT_ERROR_FMT("convt1d_gemm(%s): kernel zero_stuff_1d (%d)", label, (int)err);
    pool_release_or_release(x_exp);
    return nullptr;
  }
  bool ok = true;
  ok = ok && set_arg_checked(kz, 0, sizeof(cl_mem), &in, "x_in");
  ok = ok && set_arg_checked(kz, 1, sizeof(cl_mem), &x_exp, "x_out");
  ok = ok && set_arg_checked(kz, 2, sizeof(int), &C_in, "C_in");
  ok = ok && set_arg_checked(kz, 3, sizeof(int), &L_in, "L_in");
  ok = ok && set_arg_checked(kz, 4, sizeof(int), &L_exp, "L_exp");
  ok = ok && set_arg_checked(kz, 5, sizeof(int), &stride, "stride");
  if (!ok) { clReleaseKernel(kz); pool_release_or_release(x_exp); return nullptr; }
  const size_t gws_z[1] = {exp_n};
  err = clEnqueueNDRangeKernel(queue, kz, 1, nullptr, gws_z, nullptr, 0, nullptr,
                               KernelProfiler::event_for("convt1d_gemm.zero_stuff"));
  clReleaseKernel(kz);
  if (err != CL_SUCCESS) {
    NNOPT_ERROR_FMT("convt1d_gemm(%s): zero_stuff dispatch (%d)", label, (int)err);
    pool_release_or_release(x_exp);
    return nullptr;
  }

  // 2. Reorder weight to Conv1d layout (cached per source weight pointer)
  cl_mem w_conv = get_or_make_convT_to_conv_weight(cl_ctx, queue, w_orig, C_in, C_out, K);
  if (!w_conv) { pool_release_or_release(x_exp); return nullptr; }

  // 3. Standard conv1d via CLBlast on x_exp with reordered weight
  const int conv_pad = (K - 1) - padding;
  cl_mem y = conv1d_gemm(cl_ctx, queue, x_exp, w_conv, b,
                         C_in, C_out, L_exp, K,
                         /*stride=*/1, conv_pad, /*dilation=*/1,
                         has_bias, label);
  pool_release_or_release(x_exp);
  if (!y) return nullptr;
  if (out_L_out) *out_L_out = L_out;
  return y;
}

static cl_mem convt1d(OpenCLContext& cl_ctx,
                      cl_command_queue queue,
                      cl_program prog,
                      cl_mem in,
                      cl_mem w,
                      cl_mem b,
                      int C_in,
                      int C_out,
                      int L_in,
                      int K,
                      int stride,
                      int padding,
                      int dilation,
                      bool has_bias,
                      const char* label,
                      int* out_L_out) {
  // GEMM path via zero-stuff + standard Conv1d. Falls back to the naive
  // direct convT kernel if env NNOPT_CONVT_NAIVE=1.
  static int s_use_gemm = -1;
  if (s_use_gemm < 0) {
    const char* env = std::getenv("NNOPT_CONVT_NAIVE");
    s_use_gemm = (env && env[0] == '1') ? 0 : 1;
  }
  if (s_use_gemm && dilation == 1) {
    int Lo = 0;
    cl_mem r = convt1d_gemm(cl_ctx, queue, in, w, b, C_in, C_out, L_in, K,
                            stride, padding, dilation, has_bias, label, &Lo);
    if (r) { if (out_L_out) *out_L_out = Lo; return r; }
    NNOPT_ERROR_FMT("convt1d(%s): GEMM path failed, falling back to naive", label);
  }
  const int L_out = (L_in - 1) * stride - 2 * padding + dilation * (K - 1) + 1;
  if (L_out <= 0) {
    NNOPT_ERROR_FMT("convt1d(%s): computed L_out=%d", label, L_out);
    return nullptr;
  }
  if (out_L_out) *out_L_out = L_out;

  cl_int err = CL_SUCCESS;
  cl_context ctx = cl_ctx.context();
  const size_t n_out = (size_t)C_out * (size_t)L_out;
  cl_mem out = pool_acquire(ctx, n_out * sizeof(nnopt_storage_t));
  if (!out) {
    NNOPT_ERROR_FMT("convt1d(%s): pool_acquire(out) failed", label);
    return nullptr;
  }

  cl_kernel k = clCreateKernel(prog, "conv_transpose_1d", &err);
  if (err != CL_SUCCESS || !k) {
    NNOPT_ERROR_FMT("convt1d(%s): clCreateKernel(conv_transpose_1d) failed (%d)", label, (int)err);
    pool_release_or_release(out);
    return nullptr;
  }

  const int hb = has_bias ? 1 : 0;
  bool ok = true;
  ok = ok && set_arg_checked(k, 0, sizeof(cl_mem), &in, "in");
  ok = ok && set_arg_checked(k, 1, sizeof(cl_mem), &w, "weight");
  ok = ok && set_arg_checked(k, 2, sizeof(cl_mem), &b, "bias");
  ok = ok && set_arg_checked(k, 3, sizeof(cl_mem), &out, "out");
  ok = ok && set_arg_checked(k, 4, sizeof(int), &C_in, "C_in");
  ok = ok && set_arg_checked(k, 5, sizeof(int), &C_out, "C_out");
  ok = ok && set_arg_checked(k, 6, sizeof(int), &L_in, "L_in");
  ok = ok && set_arg_checked(k, 7, sizeof(int), &L_out, "L_out");
  ok = ok && set_arg_checked(k, 8, sizeof(int), &K, "K");
  ok = ok && set_arg_checked(k, 9, sizeof(int), &stride, "stride");
  ok = ok && set_arg_checked(k, 10, sizeof(int), &padding, "padding");
  ok = ok && set_arg_checked(k, 11, sizeof(int), &dilation, "dilation");
  ok = ok && set_arg_checked(k, 12, sizeof(int), &hb, "has_bias");
  if (!ok) {
    clReleaseKernel(k);
    pool_release_or_release(out);
    return nullptr;
  }

  const size_t gws[1] = {n_out};
  err = clEnqueueNDRangeKernel(queue, k, 1, nullptr, gws, nullptr, 0, nullptr,
                               KernelProfiler::event_for(label));
  clReleaseKernel(k);
  if (err != CL_SUCCESS) {
    NNOPT_ERROR_FMT("convt1d(%s): dispatch failed (%d)", label, (int)err);
    pool_release_or_release(out);
    return nullptr;
  }

  return out;
}

// ── Phase A: single-dispatch fused ResBlock conv ──────────────────────────
// Each HiFi-GAN ResBlock kpair currently issues 4 dispatches (im2col_fused +
// HGemm for conv1, then again for conv2). Across 4 upsample stages × 3
// branches × 3 kpairs × 2 convs that's 144 conv calls × 2 dispatches = 288.
// Each pays a 75-226 ms QUEUED→SUBMIT gap on Adreno 620 (vocoder GPU-busy is
// 5% per the profiler — the rest is driver scheduling). Collapsing each conv
// to ONE dispatch via a custom kernel ports the flow_wn_fused.cl win to the
// vocoder. Gated by NNOPT_VOC_RESBLOCK_FUSED=1; legacy path stays live.

// Reordered weight cache for the fused path. Key = original weight name.
// Storage layout: [OC, K, IC] (vs disk [OC, IC, K]) so the kernel's inner
// ci-loop is stride-1 → vload_half4 → dot(float4).
struct VocKicEntry {
  cl_mem weight = nullptr;
  int OC = 0, IC = 0, K = 0;
};
static std::unordered_map<std::string, VocKicEntry> g_voc_kic_cache;

static cl_mem get_or_make_voc_kic_weight(const std::string& key,
                                         Weights& weights,
                                         OpenCLContext& cl_ctx,
                                         cl_command_queue queue,
                                         int OC, int IC, int K) {
  auto it = g_voc_kic_cache.find(key);
  if (it != g_voc_kic_cache.end()) return it->second.weight;

  std::vector<float> host = weights.get_host_vec(key);
  if (host.empty()) {
    NNOPT_ERROR_FMT("voc_kic: get_host_vec(%s) empty", key.c_str());
    return nullptr;
  }
  const size_t expected = (size_t)OC * (size_t)IC * (size_t)K;
  if (host.size() != expected) {
    NNOPT_ERROR_FMT("voc_kic: %s size mismatch (got %zu, expect %zu)",
                    key.c_str(), host.size(), expected);
    return nullptr;
  }

  // Reorder [OC, IC, K] → [OC, K, IC] on host. One-shot per weight; cached.
  std::vector<float> reordered(host.size());
  for (int oc = 0; oc < OC; ++oc) {
    for (int ic = 0; ic < IC; ++ic) {
      for (int k = 0; k < K; ++k) {
        reordered[(size_t)oc * K * IC + (size_t)k * IC + ic] =
            host[(size_t)oc * IC * K + (size_t)ic * K + k];
      }
    }
  }

  // Upload as storage_t (fp16 if NNOPT_USE_FP16 else fp32).
  cl_int err = CL_SUCCESS;
  const size_t nbytes = reordered.size() * sizeof(nnopt_storage_t);
  std::vector<uint8_t> buf(nbytes);
#ifdef NNOPT_USE_FP16
  for (size_t i = 0; i < reordered.size(); ++i) {
    uint16_t h = nnopt_f32_to_f16(reordered[i]);
    std::memcpy(buf.data() + i * 2, &h, 2);
  }
#else
  std::memcpy(buf.data(), reordered.data(), nbytes);
#endif
  cl_mem b = clCreateBuffer(cl_ctx.context(), CL_MEM_READ_ONLY, nbytes, nullptr, &err);
  if (err != CL_SUCCESS || !b) {
    NNOPT_ERROR_FMT("voc_kic: clCreateBuffer(%s) failed (%d)", key.c_str(), (int)err);
    return nullptr;
  }
  err = clEnqueueWriteBuffer(queue, b, CL_TRUE, 0, nbytes, buf.data(), 0, nullptr, nullptr);
  if (err != CL_SUCCESS) {
    NNOPT_ERROR_FMT("voc_kic: clEnqueueWriteBuffer(%s) failed (%d)", key.c_str(), (int)err);
    clReleaseMemObject(b);
    return nullptr;
  }
  VocKicEntry e{b, OC, IC, K};
  g_voc_kic_cache.emplace(key, e);
  return b;
}

// Single-dispatch fused replacement for conv1d_resblock. Computes
//   out[c, t] = (has_resid ? resid[c, t] : 0) + bias[c]
//              + sum_k sum_ci w[c, k, ci] * leaky_in_opt(in[ci, t - pad + k*d])
// using __local memory tile for the K-wide input window. Returns a freshly
// pool-acquired cl_mem (same ownership contract as conv1d_resblock).
static cl_mem conv1d_resblock_fused_dispatch(OpenCLContext& cl_ctx,
                                             cl_command_queue queue,
                                             Weights& weights,
                                             cl_mem in,
                                             const std::string& weight_key,
                                             cl_mem b_buf,
                                             int C, int L, int K,
                                             int padding, int dilation,
                                             bool leaky_in,
                                             cl_mem residual_add,
                                             const char* label) {
  cl_mem w_kic = get_or_make_voc_kic_weight(weight_key, weights, cl_ctx, queue, C, C, K);
  if (!w_kic) {
    NNOPT_ERROR_FMT("conv1d_resblock_fused(%s): w_kic missing", label);
    return nullptr;
  }
  if (K > 11 || C > 256) {
    NNOPT_ERROR_FMT("conv1d_resblock_fused(%s): K=%d/C=%d exceed VOC_K_MAX=11/VOC_C_MAX=256",
                    label, K, C);
    return nullptr;
  }

  cl_program prog = cl_ctx.get_program("kernels/voc_resblock_conv_fused.cl");
  if (!prog) prog = cl_ctx.build_program_from_file("kernels/voc_resblock_conv_fused.cl");
  if (!prog) {
    NNOPT_ERROR_FMT("conv1d_resblock_fused(%s): program build failed", label);
    return nullptr;
  }

  static cl_kernel kf = nullptr;
  if (!kf) {
    cl_int err = CL_SUCCESS;
    kf = clCreateKernel(prog, "voc_conv1d_resblock_fused", &err);
    if (err != CL_SUCCESS || !kf) {
      NNOPT_ERROR_FMT("conv1d_resblock_fused(%s): clCreateKernel (%d)", label, (int)err);
      return nullptr;
    }
  }

  cl_context ctx = cl_ctx.context();
  cl_int err = CL_SUCCESS;
  const size_t n_out = (size_t)C * (size_t)L;
  cl_mem out = pool_acquire(ctx, n_out * sizeof(nnopt_storage_t));
  if (!out) {
    NNOPT_ERROR_FMT("conv1d_resblock_fused(%s): pool_acquire(%zu)", label, n_out);
    return nullptr;
  }

  // Dummy resid buffer; Adreno is happier with non-null cl_mem args even when
  // the kernel never reads them (has_resid=0 path).
  static cl_mem g_dummy_resid = nullptr;
  if (!g_dummy_resid) {
    g_dummy_resid = clCreateBuffer(ctx, CL_MEM_READ_WRITE, 16, nullptr, &err);
    if (err != CL_SUCCESS || !g_dummy_resid) {
      NNOPT_ERROR_FMT("conv1d_resblock_fused(%s): dummy resid (%d)", label, (int)err);
      pool_release_or_release(out);
      return nullptr;
    }
  }
  cl_mem resid_arg = residual_add ? residual_add : g_dummy_resid;

  const int leaky_flag     = leaky_in       ? 1 : 0;
  const int has_resid_flag = residual_add   ? 1 : 0;

  bool ok = true;
  ok = ok && set_arg_checked(kf,  0, sizeof(cl_mem), &in,         "h_in");
  ok = ok && set_arg_checked(kf,  1, sizeof(cl_mem), &w_kic,      "w");
  ok = ok && set_arg_checked(kf,  2, sizeof(cl_mem), &b_buf,      "b");
  ok = ok && set_arg_checked(kf,  3, sizeof(cl_mem), &resid_arg,  "resid");
  ok = ok && set_arg_checked(kf,  4, sizeof(cl_mem), &out,        "h_out");
  ok = ok && set_arg_checked(kf,  5, sizeof(int),    &C,          "C");
  ok = ok && set_arg_checked(kf,  6, sizeof(int),    &L,          "L");
  ok = ok && set_arg_checked(kf,  7, sizeof(int),    &K,          "K");
  ok = ok && set_arg_checked(kf,  8, sizeof(int),    &dilation,   "dilation");
  ok = ok && set_arg_checked(kf,  9, sizeof(int),    &padding,    "pad");
  ok = ok && set_arg_checked(kf, 10, sizeof(int),    &leaky_flag, "leaky_in");
  ok = ok && set_arg_checked(kf, 11, sizeof(int),    &has_resid_flag, "has_resid");
  if (!ok) {
    pool_release_or_release(out);
    return nullptr;
  }

  // gws = (C, L) ; lws = (C, 1). One WG per output time-step. WG size = C
  // because each WI loads one channel row into the cooperative __local tile.
  const size_t gws[2] = {(size_t)C, (size_t)L};
  const size_t lws[2] = {(size_t)C, 1};
  err = clEnqueueNDRangeKernel(queue, kf, 2, nullptr, gws, lws, 0, nullptr,
                               KernelProfiler::event_for("vocoder.resblock_fused"));
  if (err != CL_SUCCESS) {
    NNOPT_ERROR_FMT("conv1d_resblock_fused(%s): dispatch (%d)", label, (int)err);
    pool_release_or_release(out);
    return nullptr;
  }
  return out;
}

// Env-gated selector. Returns true once when NNOPT_VOC_RESBLOCK_FUSED=1.
static bool voc_resblock_fused_enabled() {
  static int s = -1;
  if (s < 0) {
    const char* e = std::getenv("NNOPT_VOC_RESBLOCK_FUSED");
    s = (e && e[0] == '1') ? 1 : 0;
  }
  return s == 1;
}

}  // namespace

extern "C" int op_Vocoder(OpenCLContext& cl_ctx,
                           Weights& weights,
                           cl_command_queue queue,
                           cl_mem mel_in,
                           int B,
                           int C_mel,
                           int T_mel,
                           std::vector<int16_t>& out_pcm) {
  if (!queue || !mel_in) {
    NNOPT_ERROR("op_Vocoder: null queue or mel_in");
    return -1;
  }
  if (B != 1) {
    NNOPT_ERROR_FMT("op_Vocoder: only B=1 supported (got %d)", B);
    return -2;
  }

  // Programs
  // IMPORTANT: building OpenCL programs inside forward-time ops is risky on Android
  // (long compile times, extra RAM pressure, and some drivers don't like it).
  // We therefore build these programs ONCE during OpenCLContext initialization.
  // This op just grabs handles from the context.
  cl_program conv_prog  = cl_ctx.get_program("kernels/conv_1d.cl");
  cl_program convt_prog = cl_ctx.get_program("kernels/conv_transpose_1d.cl");
  cl_program act_prog   = cl_ctx.get_program("kernels/hifigan_residual_block.cl");
  if (!conv_prog || !convt_prog || !act_prog) {
    NNOPT_ERROR("op_Vocoder: missing required programs (conv/convT/act)");
    return -3;
  }

  // Weights
  cl_mem w_pre = weights.get_buffer("decoder.conv_pre.weight");
  cl_mem b_pre = weights.get_buffer("decoder.conv_pre.bias");
  cl_mem w_post = weights.get_buffer("decoder.conv_post.weight");
  // conv_post has bias=False in HF; no bias tensor.

  if (!w_pre || !b_pre || !w_post) {
    NNOPT_ERROR("op_Vocoder: missing decoder conv weights");
    return -4;
  }

  // ── sub-stage profiling (NNOPT_VOC_SUBSTAGE=1) ───────────────────────
  // Wraps each major vocoder phase with clFinish + chrono to expose where the
  // ~7.5s wall-time inside this op actually lives. clFinish forces serialization
  // so the wall numbers reflect real GPU-completion order, not just enqueue.
  // Off by default — only set when profiling.
  const bool VOC_SUB = []{ const char* e = std::getenv("NNOPT_VOC_SUBSTAGE"); return e && e[0]!='0'; }();
  auto sub_now = []{ return std::chrono::steady_clock::now(); };
  auto sub_dt_ms = [](std::chrono::steady_clock::time_point a,
                      std::chrono::steady_clock::time_point b) {
    return std::chrono::duration<double, std::milli>(b - a).count();
  };
  auto t_sub_pre = sub_now();
  if (VOC_SUB) clFinish(queue);
  auto t_sub_after_finish = sub_now();

  // (a) conv_pre: [flow_size=192] -> [upsample_initial_channel=512], K=7, stride=1, pad=3
  cl_mem x = conv1d(cl_ctx, queue, conv_prog, mel_in, w_pre, b_pre,
                    /*C_in=*/C_mel,
                    /*C_out=*/MODEL_CONFIG::UPSAMPLE_INITIAL_CHANNEL,
                    /*L_in=*/T_mel,
                    /*K=*/7,
                    /*stride=*/1,
                    /*padding=*/3,
                    /*dilation=*/1,
                    /*has_bias=*/true,
                    "vocoder.conv_pre");
  if (!x) {
    return -10;
  }

  int C = MODEL_CONFIG::UPSAMPLE_INITIAL_CHANNEL;
  int L = T_mel;
  NNOPT_LAYER_CHECK("vocoder_conv_pre_out", queue, x, (size_t)C * (size_t)L);
  if (VOC_SUB) {
    clFinish(queue);
    auto t = sub_now();
    fprintf(stderr, "    voc.sub conv_pre              %8.2f ms  (initial_sync %.2f ms)\n",
            sub_dt_ms(t_sub_after_finish, t), sub_dt_ms(t_sub_pre, t_sub_after_finish));
    fflush(stderr);
  }

  // (b) upsample stages
  // upsample_rates: [8,8,2,2], upsample_kernel_sizes: [16,16,4,4]
  // Channels halve each stage: 512->256->128->64->32
  const int up_rates[4] = {MODEL_CONFIG::UPSAMPLE_RATES[0], MODEL_CONFIG::UPSAMPLE_RATES[1],
                           MODEL_CONFIG::UPSAMPLE_RATES[2], MODEL_CONFIG::UPSAMPLE_RATES[3]};
  const int up_ks[4] = {MODEL_CONFIG::UPSAMPLE_KERNEL_SIZES[0], MODEL_CONFIG::UPSAMPLE_KERNEL_SIZES[1],
                        MODEL_CONFIG::UPSAMPLE_KERNEL_SIZES[2], MODEL_CONFIG::UPSAMPLE_KERNEL_SIZES[3]};

  for (int i = 0; i < 4; ++i) {
    auto t_stage_start = sub_now();
    auto t_after_up = t_stage_start;
    // leaky_relu
    if (!run_leaky_relu(cl_ctx, queue, act_prog, x, C * L, (float)MODEL_CONFIG::LEAKY_RELU_SLOPE)) {
      pool_release_or_release(x);
      // Programs are owned/cached by OpenCLContext; do not clReleaseProgram() here.
      return -20;
    }

    // conv transpose
    char w_key[64];
    char b_key[64];
    std::snprintf(w_key, sizeof(w_key), "decoder.upsampler.%d.weight", i);
    std::snprintf(b_key, sizeof(b_key), "decoder.upsampler.%d.bias", i);
    cl_mem w_up = weights.get_buffer(w_key);
    cl_mem b_up = weights.get_buffer(b_key);
    if (!w_up || !b_up) {
      NNOPT_ERROR_FMT("op_Vocoder: missing upsampler weights for stage %d", i);
      pool_release_or_release(x);
      // Programs are owned/cached by OpenCLContext; do not clReleaseProgram() here.
      return -21;
    }

    const int C_out = C / 2;
    const int K = up_ks[i];
    const int stride = up_rates[i];
    const int padding = (K - stride) / 2;
    int L_out = 0;
    cl_mem x_up = convt1d(cl_ctx, queue, convt_prog, x, w_up, b_up,
                          /*C_in=*/C,
                          /*C_out=*/C_out,
                          /*L_in=*/L,
                          /*K=*/K,
                          /*stride=*/stride,
                          /*padding=*/padding,
                          /*dilation=*/1,
                          /*has_bias=*/true,
                          "vocoder.upsample",
                          &L_out);
    pool_release_or_release(x);
    x = x_up;
    if (!x) {
      // Programs are owned/cached by OpenCLContext; do not clReleaseProgram() here.
      return -22;
    }

    C = C_out;
    L = L_out;

    if (VOC_SUB) { clFinish(queue); t_after_up = sub_now(); }

    // ResBlocks: 3 per stage, average their outputs.
    // resblock_kernel_sizes: [3,7,11]; dilation_sizes per kernel: (1,3,5)
    //
    // Step B #4: each j-branch aliases the stage's upsample output `x` for its
    // first kpair — no per-branch clEnqueueCopyBuffer dispatch. The fused
    // leaky_im2col_1d kernel reads its input via `__global const` and writes
    // only to its im2col output buffer, so all three branches can share x
    // safely. For kpair 0 the residual operand aliases x; we MUST NOT release
    // it locally (the outer `pool_release_or_release(x)` after branch_reduce_3
    // handles it). Ownership is tracked via h_owns_buffer / resid_owns_buffer.
    //
    // The 3 branch outputs are collected and reduced via fused
    // branch_reduce_3 (sum + /3 in one dispatch).
    cl_mem branch_outs[3] = {nullptr, nullptr, nullptr};
    for (int j = 0; j < 3; ++j) {
      // h aliases x for kpair 0 (no copy buffer dispatch — saves 12 × ~73 ms
      // dispatch gap across all 4 upsample stages × 3 branches). After kpair 0
      // completes, h points to a fresh conv2 output buffer (h_owns_buffer=true).
      cl_mem h = x;
      bool h_owns_buffer = false;

      const int ks = MODEL_CONFIG::RESBLOCK_KERNEL_SIZES[j];
      // HF config has resblock_dilation_sizes, but model_config.h does not currently emit it.
      // For mms-tts-eng the default dilations are (1, 3, 5) for all kernel sizes.
      const int dilations[3] = {1, 3, 5};

      const int resblock_idx = i * 3 + j;
      for (int kpair = 0; kpair < 3; ++kpair) {
        // residual_kpair = kpair INPUT (matches HiFi-GAN h_out = conv2(conv1(h_in)) + h_in).
        // For kpair 0 it aliases x (not owned); for kpair>=1 it is the previous
        // kpair's h2 (owned by this branch's local h variable).
        cl_mem residual_kpair = h;
        const bool resid_owns_buffer = h_owns_buffer;

        // convs1.kpair with dilation
        char w1[96], b1[96];
        std::snprintf(w1, sizeof(w1), "decoder.resblocks.%d.convs1.%d.weight", resblock_idx, kpair);
        std::snprintf(b1, sizeof(b1), "decoder.resblocks.%d.convs1.%d.bias", resblock_idx, kpair);
        cl_mem w1b = weights.get_buffer(w1);
        cl_mem b1b = weights.get_buffer(b1);
        if (!w1b || !b1b) {
          NNOPT_ERROR_FMT("missing %s or %s", w1, b1);
          if (h_owns_buffer) pool_release_or_release(h);
          for (int bi = 0; bi < 3; ++bi) if (branch_outs[bi]) pool_release_or_release(branch_outs[bi]);
          pool_release_or_release(x);
          return -31;
        }
        const int pad1 = (ks * dilations[kpair] - dilations[kpair]) / 2;
        // Fused leaky+im2col absorbs the standalone leaky_relu dispatch.
        cl_mem h1;
        if (voc_resblock_fused_enabled()) {
          // Single-dispatch fused path. The fused kernel does
          // leaky→conv→bias in one shot (no separate im2col).
          h1 = conv1d_resblock_fused_dispatch(cl_ctx, queue, weights, h,
                                              w1, b1b,
                                              C, L, ks, pad1, dilations[kpair],
                                              /*leaky_in=*/true,
                                              /*residual_add=*/nullptr,
                                              "vocoder.resblock.conv1");
        } else {
          h1 = conv1d_resblock(cl_ctx, queue, h, w1b, b1b,
                                    C, C, L, ks, 1, pad1, dilations[kpair],
                                    /*has_bias=*/true,
                                    /*leaky_in=*/true, 0.0f,
                                    /*resid=*/nullptr,
                                    "vocoder.resblock.conv1");
        }
        if (!h1) {
          if (h_owns_buffer) pool_release_or_release(h);
          for (int bi = 0; bi < 3; ++bi) if (branch_outs[bi]) pool_release_or_release(branch_outs[bi]);
          pool_release_or_release(x);
          return -32;
        }

        // convs2.kpair with dilation=1, pad=(ks-1)/2
        char w2[96], b2[96];
        std::snprintf(w2, sizeof(w2), "decoder.resblocks.%d.convs2.%d.weight", resblock_idx, kpair);
        std::snprintf(b2, sizeof(b2), "decoder.resblocks.%d.convs2.%d.bias", resblock_idx, kpair);
        cl_mem w2b = weights.get_buffer(w2);
        cl_mem b2b = weights.get_buffer(b2);
        if (!w2b || !b2b) {
          NNOPT_ERROR_FMT("missing %s or %s", w2, b2);
          pool_release_or_release(h1);
          if (h_owns_buffer) pool_release_or_release(h);
          for (int bi = 0; bi < 3; ++bi) if (branch_outs[bi]) pool_release_or_release(branch_outs[bi]);
          pool_release_or_release(x);
          return -34;
        }
        const int pad2 = (ks - 1) / 2;
        // Fused leaky+im2col on h1 → conv2; bias AND residual fused via the
        // existing add_bias_broadcast_resid kernel.
        cl_mem h2;
        if (voc_resblock_fused_enabled()) {
          h2 = conv1d_resblock_fused_dispatch(cl_ctx, queue, weights, h1,
                                              w2, b2b,
                                              C, L, ks, pad2, /*dilation=*/1,
                                              /*leaky_in=*/true,
                                              /*residual_add=*/residual_kpair,
                                              "vocoder.resblock.conv2");
        } else {
          h2 = conv1d_resblock(cl_ctx, queue, h1, w2b, b2b,
                                    C, C, L, ks, 1, pad2, 1,
                                    /*has_bias=*/true,
                                    /*leaky_in=*/true, 0.0f,
                                    /*resid=*/residual_kpair,
                                    "vocoder.resblock.conv2");
        }
        pool_release_or_release(h1);
        if (!h2) {
          if (resid_owns_buffer) pool_release_or_release(residual_kpair);
          for (int bi = 0; bi < 3; ++bi) if (branch_outs[bi]) pool_release_or_release(branch_outs[bi]);
          pool_release_or_release(x);
          return -35;
        }
        // residual_kpair == previous h. Release only if we owned it (kpair >= 1).
        // For kpair 0 it aliases x — outer cleanup handles x.
        if (resid_owns_buffer) pool_release_or_release(residual_kpair);
        h = h2;
        h_owns_buffer = true;
      }

      // Stash this branch's output for the fused reduce below.
      branch_outs[j] = h;
    }

    // Fused (b0+b1+b2)/3 in a single dispatch. Replaces 2× elementwise_add +
    // 1× scale_inplace per stage (3 dispatches → 1; 12 total saved across
    // the 4 vocoder upsample stages). Cached cl_kernel: clCreateKernel is
    // hit only once per process.
    {
      cl_int err = CL_SUCCESS;
      static cl_kernel kred = nullptr;
      if (!kred) {
        kred = clCreateKernel(act_prog, "branch_reduce_3", &err);
        if (err != CL_SUCCESS || !kred) {
          NNOPT_ERROR_FMT("clCreateKernel(branch_reduce_3) failed (%d)", (int)err);
          for (int bi = 0; bi < 3; ++bi) if (branch_outs[bi]) pool_release_or_release(branch_outs[bi]);
          pool_release_or_release(x);
          return -43;
        }
      }
      cl_mem out_reduce = pool_acquire(cl_ctx.context(),
                                       (size_t)C * (size_t)L * sizeof(nnopt_storage_t));
      if (!out_reduce) {
        NNOPT_ERROR_FMT("pool_acquire(branch_reduce out) failed");
        for (int bi = 0; bi < 3; ++bi) if (branch_outs[bi]) pool_release_or_release(branch_outs[bi]);
        pool_release_or_release(x);
        return -43;
      }
      const int N = C * L;
      bool ok = true;
      ok = ok && set_arg_checked(kred, 0, sizeof(cl_mem), &branch_outs[0], "b0");
      ok = ok && set_arg_checked(kred, 1, sizeof(cl_mem), &branch_outs[1], "b1");
      ok = ok && set_arg_checked(kred, 2, sizeof(cl_mem), &branch_outs[2], "b2");
      ok = ok && set_arg_checked(kred, 3, sizeof(cl_mem), &out_reduce, "out");
      ok = ok && set_arg_checked(kred, 4, sizeof(int), &N, "N");
      if (!ok) {
        pool_release_or_release(out_reduce);
        for (int bi = 0; bi < 3; ++bi) if (branch_outs[bi]) pool_release_or_release(branch_outs[bi]);
        pool_release_or_release(x);
        return -43;
      }
      const size_t gws[1] = {(size_t)N};
      err = clEnqueueNDRangeKernel(queue, kred, 1, nullptr, gws, nullptr, 0, nullptr,
                                   KernelProfiler::event_for("vocoder.branch_reduce"));
      // 3 branch buffers consumed by this kernel; release after enqueue.
      for (int bi = 0; bi < 3; ++bi) { pool_release_or_release(branch_outs[bi]); branch_outs[bi] = nullptr; }
      if (err != CL_SUCCESS) {
        NNOPT_ERROR_FMT("dispatch branch_reduce failed (%d)", (int)err);
        pool_release_or_release(out_reduce);
        pool_release_or_release(x);
        return -43;
      }
      pool_release_or_release(x);
      x = out_reduce;
    }
    if (VOC_SUB) {
      clFinish(queue);
      auto t_stage_end = sub_now();
      fprintf(stderr,
              "    voc.sub stage%d L_out=%d C=%d   leaky+convT %7.2f ms  resblocks %7.2f ms  total %7.2f ms\n",
              i, L, C,
              sub_dt_ms(t_stage_start, t_after_up),
              sub_dt_ms(t_after_up,   t_stage_end),
              sub_dt_ms(t_stage_start, t_stage_end));
      fflush(stderr);
    }
  }

  auto t_after_stages = sub_now();
  // (c) final leaky_relu then conv_post then tanh
  if (!run_leaky_relu(cl_ctx, queue, act_prog, x, C * L, (float)MODEL_CONFIG::LEAKY_RELU_SLOPE)) {
    pool_release_or_release(x);
    // Programs are owned/cached by OpenCLContext; do not clReleaseProgram() here.
    return -50;
  }

  cl_mem y = conv1d(cl_ctx, queue, conv_prog, x, w_post, /*b=*/nullptr,
                    /*C_in=*/C,
                    /*C_out=*/1,
                    /*L_in=*/L,
                    /*K=*/7,
                    /*stride=*/1,
                    /*padding=*/3,
                    /*dilation=*/1,
                    /*has_bias=*/false,
                    "vocoder.conv_post");
  pool_release_or_release(x);
  if (!y) {
    // Programs are owned/cached by OpenCLContext; do not clReleaseProgram() here.
    return -51;
  }

  NNOPT_LAYER_CHECK("waveform_pre_squeeze", queue, y, (size_t)L);
  if (!run_tanh_inplace(cl_ctx, queue, act_prog, y, /*N=*/L)) {
    pool_release_or_release(y);
    // Programs are owned/cached by OpenCLContext; do not clReleaseProgram() here.
    return -52;
  }

  // Read back waveform float (fp16/fp32 storage) -> int16 PCM
  // IMPORTANT: nnopt_storage_t is cl_half under fp16 builds, but cl_half is NOT a C++ scalar type.
  // Never store fp16 device data in std::vector<nnopt_storage_t> — that's a vector type on many toolchains.
  std::vector<uint16_t> y_u16;
  std::vector<float> y_f32;
#ifdef NNOPT_USE_FP16
  y_u16.resize((size_t)L);
  cl_int err = clEnqueueReadBuffer(queue, y, CL_TRUE, 0, (size_t)L * sizeof(uint16_t),
                                   y_u16.data(), 0, nullptr, nullptr);
#else
  y_f32.resize((size_t)L);
  cl_int err = clEnqueueReadBuffer(queue, y, CL_TRUE, 0, (size_t)L * sizeof(float),
                                   y_f32.data(), 0, nullptr, nullptr);
#endif
  pool_release_or_release(y);
  if (VOC_SUB) {
    auto t_post = sub_now();
    fprintf(stderr, "    voc.sub conv_post+tanh+read    %8.2f ms  L=%d\n",
            sub_dt_ms(t_after_stages, t_post), L);
    fflush(stderr);
  }
  if (err != CL_SUCCESS) {
    NNOPT_ERROR_FMT("op_Vocoder: read waveform failed (%d)", (int)err);
    // Programs are owned/cached by OpenCLContext; do not clReleaseProgram() here.
    return -60;
  }

  // Empirical post-vocoder gain calibration. The HiFi-GAN with this port's
  // current upstream (deterministic-stub duration + simplified flow_inverse)
  // produces tanh-bounded waveform values around ±0.05 instead of the
  // reference ±1.0. Suspected causes: cumulative magnitude loss across the
  // upsample/resblock stack when the input z_latent has wider distribution
  // than the trained vocoder expects. As a stop-gap, scale the waveform up
  // before int16 conversion; if scaled value exceeds tanh-saturation range
  // we clip. Proper fix is to match reference z_latent statistics upstream.
  // With the layout fix in backbone (channels-first means/logvars + matching
  // noise fixture) z_latent matches reference byte-for-byte. Gain back to 1.0.
  static constexpr float NNPORT_VOCODER_OUTPUT_GAIN = 1.0f;
  out_pcm.resize((size_t)L);
  for (int i = 0; i < L; ++i) {
    float f = 0.0f;
#ifdef NNOPT_USE_FP16
    f = nnopt_f16_to_f32(y_u16[(size_t)i]);
#else
    f = y_f32[(size_t)i];
#endif
    f *= NNPORT_VOCODER_OUTPUT_GAIN;
    if (f > 1.0f) f = 1.0f;
    if (f < -1.0f) f = -1.0f;
    const int v = (int)lrintf(f * 32767.0f);
    out_pcm[(size_t)i] = (int16_t)v;
  }

  // Intentionally do NOT clReleaseProgram() for cached programs.
  // They live for process lifetime and avoid repeated recompilation on device.
  return 0;
}
