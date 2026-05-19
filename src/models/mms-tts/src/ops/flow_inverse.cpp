// Reference: model_info/transformers_src/modeling_vits.py
//   VitsResidualCouplingBlock.forward(reverse=True)
//   VitsResidualCouplingLayer.forward(reverse=True)
//   VitsWaveNet.forward
//
// GPU-side implementation (fifth pass, 2026-05-18). The earlier host-side
// version was correct but spent ~5s of host wall time per inference at 11s
// audio — the per-frame conv loops are CPU-bound on a phone.
//
// This pass:
//   • Pre-computes effective conv weights (weight_norm: g * v / ||v||) on
//     host ONCE per process, uploads to GPU as cl_mem. Cached forever.
//   • Runs every conv via the shared conv1d_gpu helper (CLBlast HGemm).
//   • Reuses existing wavenet_gate kernel for the gated activation.
//   • Reuses split/concat helpers added to flow_affine_coupling.cl.
//
// The host fallback path remains and is selected by NNOPT_FLOW_HOST=1.
//
// Math is unchanged: 4 reverse coupling stages, each {flip → coup}, mean_only=True.

#include "opencl_context.h"
#include "weights.h"
#include "debug_utils.h"
#include "model_config.h"
#include "utils.h"
#include "profiler.h"
#include "conv1d_gpu.h"

#include <CL/cl.h>
#include <cstdint>
#include <cmath>
#include <cstring>
#include <functional>
#include <string>
#include <thread>
#include <unordered_map>
#include <vector>

namespace {

constexpr int kFlowChannels = 192;
constexpr int kHalfChannels = 96;
constexpr int kWnKernel     = 5;
constexpr int kWnLayers     = 4;
constexpr int kFlowStages   = 4;

// Debug instrumentation for the host vs GPU coupling-stage comparison
// (NNOPT_FLOW_DEBUG=1). Mirrors the LAYER_CHECK stat format so host and GPU
// rows can be diff'd directly. Stats over the first prefix_n elements.
static inline bool dbg_enabled() {
  static int s = -1;
  if (s < 0) {
    const char* e = std::getenv("NNOPT_FLOW_DEBUG");
    s = (e && e[0] == '1') ? 1 : 0;
  }
  return s == 1;
}
static void dbg_host_stats(const char* name, const std::vector<float>& v, size_t prefix_n = 256) {
  if (!dbg_enabled() || v.empty()) return;
  const size_t n = std::min(prefix_n, v.size());
  float mn = v[0], mx = v[0], sum = 0.0f, sq = 0.0f;
  for (size_t i = 0; i < n; ++i) {
    float x = v[i];
    if (x < mn) mn = x;
    if (x > mx) mx = x;
    sum += x; sq += x * x;
  }
  std::fprintf(stderr, "DBG_FLOW[host][%s]: n=%zu min=%g max=%g mean=%g rms=%g\n",
               name, n, mn, mx, sum / (float)n, std::sqrt(sq / (float)n));
}
static void dbg_gpu_stats(const char* name, cl_command_queue queue, cl_mem buf, size_t n) {
  if (!dbg_enabled()) return;
  NNOPT_LAYER_CHECK(name, queue, buf, n);
}

// ── Effective-weight cache on GPU ───────────────────────────────────────
// Each entry holds the g * v / ||v|| weight on GPU (no weight_norm runtime
// math needed) plus the bias buffer.
struct GpuW {
  cl_mem weight = nullptr;   // [OC, IC, K] in storage_t
  cl_mem bias   = nullptr;   // [OC] in storage_t
  int OC = 0, IC = 0, K = 0;
};
static std::unordered_map<std::string, GpuW> g_gpu_w_cache;

static std::vector<float> apply_weight_norm(const std::vector<float>& v,
                                            const std::vector<float>& g,
                                            int OC, int IC, int K) {
  std::vector<float> out(v.size());
  const size_t per_oc = (size_t)IC * (size_t)K;
  for (int oc = 0; oc < OC; ++oc) {
    float sumsq = 0.0f;
    const float* vrow = v.data() + (size_t)oc * per_oc;
    for (size_t i = 0; i < per_oc; ++i) sumsq += vrow[i] * vrow[i];
    const float norm = std::sqrt(sumsq);
    const float scale = (norm > 1e-8f) ? (g[oc] / norm) : 0.0f;
    float* orow = out.data() + (size_t)oc * per_oc;
    for (size_t i = 0; i < per_oc; ++i) orow[i] = vrow[i] * scale;
  }
  return out;
}

static cl_mem upload_storage(cl_context ctx, cl_command_queue queue,
                             const std::vector<float>& src) {
  cl_int err = CL_SUCCESS;
  const size_t n = src.size();
  std::vector<uint8_t> buf(n * sizeof(nnopt_storage_t));
#ifdef NNOPT_USE_FP16
  for (size_t i = 0; i < n; ++i) {
    uint16_t h = nnopt_f32_to_f16(src[i]);
    std::memcpy(buf.data() + i * 2, &h, 2);
  }
#else
  std::memcpy(buf.data(), src.data(), n * sizeof(float));
#endif
  cl_mem b = clCreateBuffer(ctx, CL_MEM_READ_ONLY, buf.size(), nullptr, &err);
  if (err != CL_SUCCESS || !b) return nullptr;
  if (clEnqueueWriteBuffer(queue, b, CL_TRUE, 0, buf.size(), buf.data(),
                           0, nullptr, nullptr) != CL_SUCCESS) {
    clReleaseMemObject(b);
    return nullptr;
  }
  return b;
}

// Reorder a weight tensor from [OC, IC, K] to [OC, K, IC] layout. The fused
// WN kernel reads K with an outer loop and IC with an inner stride-1 loop,
// which lets us vload_half4 over IC. Used only for the in_layers conv (K=5);
// the legacy 8-dispatch path still consumes the original [OC, IC, K] layout
// from the separate cache.
static std::vector<float> transpose_ic_k(const std::vector<float>& src,
                                         int OC, int IC, int K) {
  std::vector<float> out(src.size());
  for (int oc = 0; oc < OC; ++oc) {
    for (int ic = 0; ic < IC; ++ic) {
      for (int k = 0; k < K; ++k) {
        // src[(oc, ic, k)] = oc*IC*K + ic*K + k
        // dst[(oc, k, ic)] = oc*K*IC + k*IC + ic
        out[(size_t)oc * K * IC + (size_t)k * IC + ic] =
            src[(size_t)oc * IC * K + (size_t)ic * K + k];
      }
    }
  }
  return out;
}

// Reordered-layout cache for the fused WN path (key = original prefix + "@kic")
// so the legacy path keeps its [OC, IC, K] cache unchanged.
static const GpuW* get_or_make_wn_weights_kic(const std::string& prefix,
                                              Weights& weights,
                                              OpenCLContext& cl_ctx,
                                              cl_command_queue queue,
                                              int OC, int IC, int K) {
  const std::string cache_key = prefix + "@kic";
  auto it = g_gpu_w_cache.find(cache_key);
  if (it != g_gpu_w_cache.end()) return &it->second;
  auto v = weights.get_host_vec(prefix + ".weight_v");
  auto g = weights.get_host_vec(prefix + ".weight_g");
  auto b = weights.get_host_vec(prefix + ".bias");
  if (v.empty() || g.empty()) return nullptr;
  std::vector<float> W = apply_weight_norm(v, g, OC, IC, K);
  std::vector<float> W_kic = transpose_ic_k(W, OC, IC, K);
  GpuW e;
  e.OC = OC; e.IC = IC; e.K = K;
  e.weight = upload_storage(cl_ctx.context(), queue, W_kic);
  e.bias   = upload_storage(cl_ctx.context(), queue, b);
  if (!e.weight || !e.bias) {
    if (e.weight) clReleaseMemObject(e.weight);
    if (e.bias) clReleaseMemObject(e.bias);
    return nullptr;
  }
  auto ins = g_gpu_w_cache.emplace(cache_key, std::move(e));
  return &ins.first->second;
}

static const GpuW* get_or_make_wn_weights(const std::string& prefix,
                                          Weights& weights,
                                          OpenCLContext& cl_ctx,
                                          cl_command_queue queue,
                                          int OC, int IC, int K) {
  auto it = g_gpu_w_cache.find(prefix);
  if (it != g_gpu_w_cache.end()) return &it->second;
  auto v = weights.get_host_vec(prefix + ".weight_v");
  auto g = weights.get_host_vec(prefix + ".weight_g");
  auto b = weights.get_host_vec(prefix + ".bias");
  if (v.empty() || g.empty()) return nullptr;
  std::vector<float> W = apply_weight_norm(v, g, OC, IC, K);
  GpuW e;
  e.OC = OC; e.IC = IC; e.K = K;
  e.weight = upload_storage(cl_ctx.context(), queue, W);
  e.bias   = upload_storage(cl_ctx.context(), queue, b);
  if (!e.weight || !e.bias) {
    if (e.weight) clReleaseMemObject(e.weight);
    if (e.bias) clReleaseMemObject(e.bias);
    return nullptr;
  }
  auto ins = g_gpu_w_cache.emplace(prefix, std::move(e));
  return &ins.first->second;
}

static const GpuW* get_or_make_plain_weights(const std::string& prefix,
                                             Weights& weights,
                                             OpenCLContext& cl_ctx,
                                             cl_command_queue queue,
                                             int OC, int IC, int K) {
  auto it = g_gpu_w_cache.find(prefix);
  if (it != g_gpu_w_cache.end()) return &it->second;
  auto W = weights.get_host_vec(prefix + ".weight");
  auto b = weights.get_host_vec(prefix + ".bias");
  if (W.empty()) return nullptr;
  GpuW e;
  e.OC = OC; e.IC = IC; e.K = K;
  e.weight = upload_storage(cl_ctx.context(), queue, W);
  e.bias   = upload_storage(cl_ctx.context(), queue, b);
  if (!e.weight || !e.bias) {
    if (e.weight) clReleaseMemObject(e.weight);
    if (e.bias) clReleaseMemObject(e.bias);
    return nullptr;
  }
  auto ins = g_gpu_w_cache.emplace(prefix, std::move(e));
  return &ins.first->second;
}

// Dispatch a small kernel. Caches the kernel handle.
static bool dispatch_simple(OpenCLContext& cl_ctx, cl_command_queue queue,
                            const char* prog_path, const char* kname,
                            size_t gws, cl_kernel* cache,
                            std::function<bool(cl_kernel)> set_args,
                            const char* label) {
  cl_program prog = cl_ctx.get_program(prog_path);
  if (!prog) prog = cl_ctx.build_program_from_file(prog_path);
  if (!prog) return false;
  cl_int err = CL_SUCCESS;
  if (!*cache) {
    *cache = clCreateKernel(prog, kname, &err);
    if (err != CL_SUCCESS || !*cache) return false;
  }
  if (!set_args(*cache)) return false;
  const size_t g[1] = {gws};
  err = clEnqueueNDRangeKernel(queue, *cache, 1, nullptr, g, nullptr,
                               0, nullptr, KernelProfiler::event_for(label));
  return err == CL_SUCCESS;
}

// Split [2*kHalfChannels, T] → y_a [kHalfChannels, T], y_b [kHalfChannels, T].
static bool dispatch_split_half(OpenCLContext& cl_ctx, cl_command_queue queue,
                                cl_mem x, cl_mem y_a, cl_mem y_b, int T) {
  static cl_kernel k = nullptr;
  return dispatch_simple(cl_ctx, queue,
      "kernels/flow_affine_coupling.cl", "split_half_c",
      (size_t)kHalfChannels * (size_t)T, &k,
      [&](cl_kernel kn) {
        cl_int e = 0;
        e |= clSetKernelArg(kn, 0, sizeof(cl_mem), &x);
        e |= clSetKernelArg(kn, 1, sizeof(cl_mem), &y_a);
        e |= clSetKernelArg(kn, 2, sizeof(cl_mem), &y_b);
        e |= clSetKernelArg(kn, 3, sizeof(int), &(int&)(*const_cast<int*>(&kHalfChannels)));
        e |= clSetKernelArg(kn, 4, sizeof(int), &T);
        return e == 0;
      }, "flow.split_half");
}

static bool dispatch_concat_half(OpenCLContext& cl_ctx, cl_command_queue queue,
                                 cl_mem x, cl_mem y_a, cl_mem y_b, int T) {
  static cl_kernel k = nullptr;
  return dispatch_simple(cl_ctx, queue,
      "kernels/flow_affine_coupling.cl", "concat_half_c",
      (size_t)kHalfChannels * (size_t)T, &k,
      [&](cl_kernel kn) {
        cl_int e = 0;
        e |= clSetKernelArg(kn, 0, sizeof(cl_mem), &x);
        e |= clSetKernelArg(kn, 1, sizeof(cl_mem), &y_a);
        e |= clSetKernelArg(kn, 2, sizeof(cl_mem), &y_b);
        e |= clSetKernelArg(kn, 3, sizeof(int), &(int&)(*const_cast<int*>(&kHalfChannels)));
        e |= clSetKernelArg(kn, 4, sizeof(int), &T);
        return e == 0;
      }, "flow.concat_half");
}

// Step D fusion — split rs INTO (h, skip_sum) IN-PLACE in one dispatch.
// Replaces split_res_skip + 2× elem_add_inplace_a (3 dispatches → 1).
static bool dispatch_split_rs_fold(OpenCLContext& cl_ctx, cl_command_queue queue,
                                   cl_mem rs, cl_mem h, cl_mem skip_sum,
                                   int C, int T) {
  static cl_kernel k = nullptr;
  return dispatch_simple(cl_ctx, queue,
      "kernels/flow_affine_coupling.cl", "split_rs_fold_h_skip",
      (size_t)C * (size_t)T, &k,
      [&](cl_kernel kn) {
        cl_int e = 0;
        e |= clSetKernelArg(kn, 0, sizeof(cl_mem), &rs);
        e |= clSetKernelArg(kn, 1, sizeof(cl_mem), &h);
        e |= clSetKernelArg(kn, 2, sizeof(cl_mem), &skip_sum);
        e |= clSetKernelArg(kn, 3, sizeof(int), &C);
        e |= clSetKernelArg(kn, 4, sizeof(int), &T);
        return e == 0;
      }, "flow.split_rs_fold");
}

static bool dispatch_split_res_skip(OpenCLContext& cl_ctx, cl_command_queue queue,
                                    cl_mem rs, cl_mem res, cl_mem skip, int C, int T) {
  static cl_kernel k = nullptr;
  return dispatch_simple(cl_ctx, queue,
      "kernels/flow_affine_coupling.cl", "split_res_skip",
      (size_t)C * (size_t)T, &k,
      [&](cl_kernel kn) {
        cl_int e = 0;
        e |= clSetKernelArg(kn, 0, sizeof(cl_mem), &rs);
        e |= clSetKernelArg(kn, 1, sizeof(cl_mem), &res);
        e |= clSetKernelArg(kn, 2, sizeof(cl_mem), &skip);
        e |= clSetKernelArg(kn, 3, sizeof(int), &C);
        e |= clSetKernelArg(kn, 4, sizeof(int), &T);
        return e == 0;
      }, "flow.split_res_skip");
}

// Fused subtract+concat — combines (y_b -= t_mean) + concat(y_a, y_b) → x_inout
// into ONE dispatch. Saves 1 dispatch per stage × 4 stages × ~54 ms gap.
static bool dispatch_sub_then_concat(OpenCLContext& cl_ctx, cl_command_queue queue,
                                     cl_mem x_inout, cl_mem y_a, cl_mem y_b,
                                     cl_mem t_mean, int T) {
  static cl_kernel k = nullptr;
  return dispatch_simple(cl_ctx, queue,
      "kernels/flow_affine_coupling.cl", "sub_then_concat_half_c",
      (size_t)kHalfChannels * (size_t)T, &k,
      [&](cl_kernel kn) {
        cl_int e = 0;
        e |= clSetKernelArg(kn, 0, sizeof(cl_mem), &x_inout);
        e |= clSetKernelArg(kn, 1, sizeof(cl_mem), &y_a);
        e |= clSetKernelArg(kn, 2, sizeof(cl_mem), &y_b);
        e |= clSetKernelArg(kn, 3, sizeof(cl_mem), &t_mean);
        e |= clSetKernelArg(kn, 4, sizeof(int), &(int&)(*const_cast<int*>(&kHalfChannels)));
        e |= clSetKernelArg(kn, 5, sizeof(int), &T);
        return e == 0;
      }, "flow.sub_concat");
}

static bool dispatch_sub_inplace(OpenCLContext& cl_ctx, cl_command_queue queue,
                                 cl_mem a, cl_mem b, int N) {
  static cl_kernel k = nullptr;
  return dispatch_simple(cl_ctx, queue,
      "kernels/flow_affine_coupling.cl", "elem_sub_inplace_a",
      (size_t)N, &k,
      [&](cl_kernel kn) {
        cl_int e = 0;
        e |= clSetKernelArg(kn, 0, sizeof(cl_mem), &a);
        e |= clSetKernelArg(kn, 1, sizeof(cl_mem), &b);
        e |= clSetKernelArg(kn, 2, sizeof(int), &N);
        return e == 0;
      }, "flow.sub_inplace");
}

static bool dispatch_add_inplace(OpenCLContext& cl_ctx, cl_command_queue queue,
                                 cl_mem a, cl_mem b, int N) {
  static cl_kernel k = nullptr;
  return dispatch_simple(cl_ctx, queue,
      "kernels/flow_affine_coupling.cl", "elem_add_inplace_a",
      (size_t)N, &k,
      [&](cl_kernel kn) {
        cl_int e = 0;
        e |= clSetKernelArg(kn, 0, sizeof(cl_mem), &a);
        e |= clSetKernelArg(kn, 1, sizeof(cl_mem), &b);
        e |= clSetKernelArg(kn, 2, sizeof(int), &N);
        return e == 0;
      }, "flow.add_inplace");
}

static bool dispatch_channel_flip(OpenCLContext& cl_ctx, cl_command_queue queue,
                                  cl_mem in, cl_mem out, int C, int T) {
  static cl_kernel k = nullptr;
  return dispatch_simple(cl_ctx, queue,
      "kernels/flow_affine_coupling.cl", "channel_flip",
      (size_t)C * (size_t)T, &k,
      [&](cl_kernel kn) {
        cl_int e = 0;
        e |= clSetKernelArg(kn, 0, sizeof(cl_mem), &in);
        e |= clSetKernelArg(kn, 1, sizeof(cl_mem), &out);
        e |= clSetKernelArg(kn, 2, sizeof(int), &C);
        e |= clSetKernelArg(kn, 3, sizeof(int), &T);
        return e == 0;
      }, "flow.channel_flip");
}

// One reverse-coupling stage on GPU. Input/output is x in-place [192, T].
static bool coupling_inverse_stage_gpu(OpenCLContext& cl_ctx,
                                       cl_command_queue queue,
                                       Weights& weights,
                                       int stage_idx,
                                       cl_mem x_inout, int T) {
  cl_context ctx = cl_ctx.context();
  cl_int err = CL_SUCCESS;

  // Split y_a / y_b
  const size_t half_bytes = (size_t)kHalfChannels * (size_t)T * sizeof(nnopt_storage_t);
  cl_mem y_a = clCreateBuffer(ctx, CL_MEM_READ_WRITE, half_bytes, nullptr, &err);
  if (err != CL_SUCCESS || !y_a) return false;
  cl_mem y_b = clCreateBuffer(ctx, CL_MEM_READ_WRITE, half_bytes, nullptr, &err);
  if (err != CL_SUCCESS || !y_b) { clReleaseMemObject(y_a); return false; }
  if (!dispatch_split_half(cl_ctx, queue, x_inout, y_a, y_b, T)) {
    clReleaseMemObject(y_a); clReleaseMemObject(y_b); return false;
  }
  if (stage_idx == 3) {
    dbg_gpu_stats("st3.gpu.y_a",   queue, y_a,   (size_t)kHalfChannels * T);
    dbg_gpu_stats("st3.gpu.y_b",   queue, y_b,   (size_t)kHalfChannels * T);
  }

  char buf[256];
  std::snprintf(buf, sizeof(buf), "flow.flows.%d.conv_pre", stage_idx);
  const GpuW* w_pre = get_or_make_plain_weights(buf, weights, cl_ctx, queue,
                                                kFlowChannels, kHalfChannels, 1);
  std::snprintf(buf, sizeof(buf), "flow.flows.%d.conv_post", stage_idx);
  const GpuW* w_post = get_or_make_plain_weights(buf, weights, cl_ctx, queue,
                                                 kHalfChannels, kFlowChannels, 1);
  if (!w_pre || !w_post) {
    clReleaseMemObject(y_a); clReleaseMemObject(y_b);
    return false;
  }

  // h = conv_pre(y_a) → [192, T]
  cl_mem h = conv1d_gpu(cl_ctx, queue, y_a, w_pre->weight, w_pre->bias,
                        kHalfChannels, kFlowChannels, T,
                        /*K=*/1, /*stride=*/1, /*padding=*/0, /*dilation=*/1,
                        /*has_bias=*/true, "flow.conv_pre");
  if (!h) { clReleaseMemObject(y_a); clReleaseMemObject(y_b); return false; }
  if (stage_idx == 3) {
    dbg_gpu_stats("st3.gpu.h_pre", queue, h, (size_t)kFlowChannels * T);
  }

  // WaveNet: accumulate skip_sum across 4 layers
  const size_t flow_bytes = (size_t)kFlowChannels * (size_t)T * sizeof(nnopt_storage_t);
  cl_mem skip_sum = clCreateBuffer(ctx, CL_MEM_READ_WRITE, flow_bytes, nullptr, &err);
  if (err != CL_SUCCESS || !skip_sum) {
    clReleaseMemObject(h); clReleaseMemObject(y_a); clReleaseMemObject(y_b);
    return false;
  }
  {
    std::vector<uint8_t> zeros((size_t)kFlowChannels * T * sizeof(nnopt_storage_t), 0);
    if (clEnqueueWriteBuffer(queue, skip_sum, CL_TRUE, 0, zeros.size(), zeros.data(),
                             0, nullptr, nullptr) != CL_SUCCESS) {
      clReleaseMemObject(h); clReleaseMemObject(y_a); clReleaseMemObject(y_b);
      clReleaseMemObject(skip_sum);
      return false;
    }
  }

  // ── WaveNet residual loop ─────────────────────────────────────────────
  //
  // Two implementations selectable at runtime:
  //   (A) per-op default: 8 dispatches per layer (in conv + gated_act + rs
  //       conv + split_rs_fold) — see the branch under !wn_fused.
  //   (B) NNOPT_FLOW_WN_FUSED=1: one dispatch per layer via
  //       kernels/flow_wn_fused.cl::flow_wn_layer_fused. Cuts the WaveNet
  //       inner loop from 64 dispatches (8×16) to 16 — directly attacks the
  //       per-dispatch idle gap that dominates Adreno 620 flow_inverse wall.
  //
  // (B) needs a second h buffer because the K=5 in conv reads h positions
  // adjacent to the one each workitem writes — pinging across layers.
  static int s_wn_fused = -1;
  if (s_wn_fused < 0) {
    const char* e = std::getenv("NNOPT_FLOW_WN_FUSED");
    s_wn_fused = (e && e[0] == '1') ? 1 : 0;
  }

  // Allocate the ping-pong h buffer up front when fused path is on.
  cl_mem h_alt = nullptr;
  cl_kernel k_wn_fused = nullptr;
  if (s_wn_fused) {
    h_alt = clCreateBuffer(ctx, CL_MEM_READ_WRITE, flow_bytes, nullptr, &err);
    if (err != CL_SUCCESS || !h_alt) {
      NNOPT_ERROR_FMT("flow_wn_fused: alloc h_alt (%d)", (int)err);
      clReleaseMemObject(h); clReleaseMemObject(y_a); clReleaseMemObject(y_b);
      clReleaseMemObject(skip_sum); return false;
    }
    cl_program prog = cl_ctx.get_program("kernels/flow_wn_fused.cl");
    if (!prog) prog = cl_ctx.build_program_from_file("kernels/flow_wn_fused.cl");
    if (!prog) {
      NNOPT_ERROR("flow_wn_fused: program missing");
      clReleaseMemObject(h_alt);
      clReleaseMemObject(h); clReleaseMemObject(y_a); clReleaseMemObject(y_b);
      clReleaseMemObject(skip_sum); return false;
    }
    k_wn_fused = clCreateKernel(prog, "flow_wn_layer_fused", &err);
    if (err != CL_SUCCESS || !k_wn_fused) {
      NNOPT_ERROR_FMT("flow_wn_fused: clCreateKernel (%d)", (int)err);
      clReleaseMemObject(h_alt);
      clReleaseMemObject(h); clReleaseMemObject(y_a); clReleaseMemObject(y_b);
      clReleaseMemObject(skip_sum); return false;
    }
  }

  cl_mem h_in_buf  = h;       // ping
  cl_mem h_out_buf = h_alt;   // pong (used only on fused path)

  for (int li = 0; li < kWnLayers; ++li) {
    std::snprintf(buf, sizeof(buf), "flow.flows.%d.wavenet.in_layers.%d", stage_idx, li);
    const GpuW* w_in = get_or_make_wn_weights(buf, weights, cl_ctx, queue,
                                              2 * kFlowChannels, kFlowChannels, kWnKernel);
    std::snprintf(buf, sizeof(buf), "flow.flows.%d.wavenet.res_skip_layers.%d", stage_idx, li);
    const int rs_oc = (li < kWnLayers - 1) ? (2 * kFlowChannels) : kFlowChannels;
    const GpuW* w_rs = get_or_make_wn_weights(buf, weights, cl_ctx, queue,
                                              rs_oc, kFlowChannels, 1);
    if (!w_in || !w_rs) {
      if (h_alt) clReleaseMemObject(h_alt);
      if (k_wn_fused) clReleaseKernel(k_wn_fused);
      clReleaseMemObject(h); clReleaseMemObject(y_a); clReleaseMemObject(y_b);
      clReleaseMemObject(skip_sum); return false;
    }

    if (s_wn_fused) {
      // ── Fused single-dispatch layer ──────────────────────────────────
      // Re-fetch w_in from the reordered [OC, K, IC] cache (vectorized inner
      // loop reads contiguous IC). w_rs (K=1) stays in the default cache.
      std::snprintf(buf, sizeof(buf), "flow.flows.%d.wavenet.in_layers.%d", stage_idx, li);
      const GpuW* w_in_kic = get_or_make_wn_weights_kic(buf, weights, cl_ctx, queue,
                                                        2 * kFlowChannels, kFlowChannels, kWnKernel);
      if (!w_in_kic) {
        NNOPT_ERROR("flow_wn_fused: failed to build [OC,K,IC] weight cache");
        clReleaseKernel(k_wn_fused); clReleaseMemObject(h_alt);
        clReleaseMemObject(h); clReleaseMemObject(y_a); clReleaseMemObject(y_b);
        clReleaseMemObject(skip_sum); return false;
      }
      const int is_last = (li == kWnLayers - 1) ? 1 : 0;
      bool ok = true;
      ok = ok && (clSetKernelArg(k_wn_fused, 0, sizeof(cl_mem), &h_in_buf)  == CL_SUCCESS);
      ok = ok && (clSetKernelArg(k_wn_fused, 1, sizeof(cl_mem), &h_out_buf) == CL_SUCCESS);
      ok = ok && (clSetKernelArg(k_wn_fused, 2, sizeof(cl_mem), &w_in_kic->weight) == CL_SUCCESS);
      ok = ok && (clSetKernelArg(k_wn_fused, 3, sizeof(cl_mem), &w_in_kic->bias)   == CL_SUCCESS);
      ok = ok && (clSetKernelArg(k_wn_fused, 4, sizeof(cl_mem), &w_rs->weight) == CL_SUCCESS);
      ok = ok && (clSetKernelArg(k_wn_fused, 5, sizeof(cl_mem), &w_rs->bias)   == CL_SUCCESS);
      ok = ok && (clSetKernelArg(k_wn_fused, 6, sizeof(cl_mem), &skip_sum)     == CL_SUCCESS);
      ok = ok && (clSetKernelArg(k_wn_fused, 7, sizeof(int),    &T)            == CL_SUCCESS);
      ok = ok && (clSetKernelArg(k_wn_fused, 8, sizeof(int),    &is_last)      == CL_SUCCESS);
      if (!ok) {
        NNOPT_ERROR("flow_wn_fused: setKernelArg failed");
        clReleaseKernel(k_wn_fused); clReleaseMemObject(h_alt);
        clReleaseMemObject(h); clReleaseMemObject(y_a); clReleaseMemObject(y_b);
        clReleaseMemObject(skip_sum); return false;
      }
      const size_t gws[2] = { (size_t)kFlowChannels, (size_t)T };
      const size_t lws[2] = { (size_t)kFlowChannels, 1 };
      cl_int de = clEnqueueNDRangeKernel(queue, k_wn_fused, 2, nullptr, gws, lws,
                                         0, nullptr, KernelProfiler::event_for("flow.wn.fused"));
      if (de != CL_SUCCESS) {
        NNOPT_ERROR_FMT("flow_wn_fused: enqueue (%d)", (int)de);
        clReleaseKernel(k_wn_fused); clReleaseMemObject(h_alt);
        clReleaseMemObject(h); clReleaseMemObject(y_a); clReleaseMemObject(y_b);
        clReleaseMemObject(skip_sum); return false;
      }
      // Ping-pong h buffers for next layer (last layer doesn't write h_out).
      if (!is_last) {
        cl_mem tmp = h_in_buf; h_in_buf = h_out_buf; h_out_buf = tmp;
      }
      if (stage_idx == 3 && li == 0) {
        // peek skip_sum after layer 0 (acts/pre_acts no longer materialized)
        dbg_gpu_stats("st3.gpu.fused.l0.skip_sum_after",
                      queue, skip_sum, (size_t)kFlowChannels * T);
      }
      continue;
    }

    // ── Legacy 8-dispatch path ──────────────────────────────────────────
    // pre_acts = conv(h, in_layer, K=5, pad=2) → [2*C, T]
    cl_mem pre_acts = conv1d_gpu(cl_ctx, queue, h, w_in->weight, w_in->bias,
                                 kFlowChannels, 2 * kFlowChannels, T,
                                 /*K=*/kWnKernel, /*stride=*/1,
                                 /*padding=*/(kWnKernel - 1) / 2, /*dilation=*/1,
                                 /*has_bias=*/true, "flow.wn.in");
    if (!pre_acts) {
      clReleaseMemObject(h); clReleaseMemObject(y_a); clReleaseMemObject(y_b);
      clReleaseMemObject(skip_sum); return false;
    }
    if (stage_idx == 3 && li == 0) {
      dbg_gpu_stats("st3.gpu.l0.pre_acts", queue, pre_acts, (size_t)(2 * kFlowChannels) * T);
    }

    // acts = wavenet_gate(pre_acts) → [C, T]
    cl_mem acts = gated_activation_gpu(cl_ctx, queue, pre_acts, kFlowChannels, T,
                                       "flow.wn.gate");
    clReleaseMemObject(pre_acts);
    if (!acts) {
      clReleaseMemObject(h); clReleaseMemObject(y_a); clReleaseMemObject(y_b);
      clReleaseMemObject(skip_sum); return false;
    }
    if (stage_idx == 3 && li == 0) {
      dbg_gpu_stats("st3.gpu.l0.acts", queue, acts, (size_t)kFlowChannels * T);
    }

    // rs = conv(acts, res_skip, K=1) → [rs_oc, T]
    cl_mem rs = conv1d_gpu(cl_ctx, queue, acts, w_rs->weight, w_rs->bias,
                           kFlowChannels, rs_oc, T,
                           /*K=*/1, /*stride=*/1, /*padding=*/0, /*dilation=*/1,
                           /*has_bias=*/true, "flow.wn.rs");
    clReleaseMemObject(acts);
    if (stage_idx == 3 && li == 0 && rs) {
      dbg_gpu_stats("st3.gpu.l0.rs", queue, rs, (size_t)rs_oc * T);
    }
    if (!rs) {
      clReleaseMemObject(h); clReleaseMemObject(y_a); clReleaseMemObject(y_b);
      clReleaseMemObject(skip_sum); return false;
    }

    if (li < kWnLayers - 1) {
      if (!dispatch_split_rs_fold(cl_ctx, queue, rs, h, skip_sum, kFlowChannels, T)) {
        clReleaseMemObject(rs); clReleaseMemObject(h);
        clReleaseMemObject(y_a); clReleaseMemObject(y_b);
        clReleaseMemObject(skip_sum); return false;
      }
      clReleaseMemObject(rs);
    } else {
      if (!dispatch_add_inplace(cl_ctx, queue, skip_sum, rs, kFlowChannels * T)) {
        clReleaseMemObject(rs); clReleaseMemObject(h); clReleaseMemObject(y_a);
        clReleaseMemObject(y_b); clReleaseMemObject(skip_sum); return false;
      }
      clReleaseMemObject(rs);
    }
  }
  if (k_wn_fused) clReleaseKernel(k_wn_fused);
  if (h_alt) clReleaseMemObject(h_alt);
  clReleaseMemObject(h);

  if (stage_idx == 3) {
    dbg_gpu_stats("st3.gpu.skip_final", queue, skip_sum, (size_t)kFlowChannels * T);
  }
  // t = conv_post(skip_sum) → [half, T]
  cl_mem t_mean = conv1d_gpu(cl_ctx, queue, skip_sum, w_post->weight, w_post->bias,
                             kFlowChannels, kHalfChannels, T,
                             /*K=*/1, /*stride=*/1, /*padding=*/0, /*dilation=*/1,
                             /*has_bias=*/true, "flow.conv_post");
  clReleaseMemObject(skip_sum);
  if (!t_mean) {
    clReleaseMemObject(y_a); clReleaseMemObject(y_b); return false;
  }
  if (stage_idx == 3) {
    dbg_gpu_stats("st3.gpu.t_mean", queue, t_mean, (size_t)kHalfChannels * T);
  }

  // Fused: y_b' = y_b - t_mean, then concat(y_a, y_b') → x_inout. One dispatch
  // replaces (sub_inplace + concat_half), saving the per-dispatch gap.
  if (!dispatch_sub_then_concat(cl_ctx, queue, x_inout, y_a, y_b, t_mean, T)) {
    clReleaseMemObject(t_mean); clReleaseMemObject(y_a); clReleaseMemObject(y_b);
    return false;
  }
  clReleaseMemObject(t_mean);
  clReleaseMemObject(y_a);
  clReleaseMemObject(y_b);
  return true;
}

// ── Host fallback (the previous implementation) ─────────────────────────
// Lifted into a static so it can be selected via env var if the GPU path
// fails. Keeping it for safety.
static std::vector<float> get_w(Weights& w, const std::string& key) {
  return w.get_host_vec(key);
}
struct WnCacheEntry { std::vector<float> W; std::vector<float> b; };
static std::unordered_map<std::string, WnCacheEntry> g_wn_host_cache;

static std::vector<float> weight_norm_host(const std::vector<float>& v,
                                           const std::vector<float>& g,
                                           int OC, int IC, int K) {
  return apply_weight_norm(v, g, OC, IC, K);
}

static void conv1d_cf_host(const std::vector<float>& x, int IC, int T,
                           const std::vector<float>& W, const std::vector<float>& b,
                           int OC, int K, int dilation,
                           std::vector<float>& y) {
  y.assign((size_t)OC * T, 0.0f);
  const int pad = (K * dilation - dilation) / 2;
  const float* W_data = W.data();
  const float* b_data = b.empty() ? nullptr : b.data();
  // Parallelize over output channels with OpenMP. Use the threshold guard
  // because OpenMP's first call in a process spins up its thread pool
  // (~few ms) — only worth it when the loop is fat enough to amortize.
#ifdef NNOPT_HAS_OPENMP
  #pragma omp parallel for schedule(static) if(OC * T > 50000)
#endif
  for (int oc = 0; oc < OC; ++oc) {
    const float bias_v = b_data ? b_data[oc] : 0.0f;
    const float* w_oc = W_data + (size_t)oc * IC * K;
    float* y_oc = y.data() + (size_t)oc * T;
    for (int t = 0; t < T; ++t) {
      float s = bias_v;
      for (int ic = 0; ic < IC; ++ic) {
        const float* wic = w_oc + (size_t)ic * K;
        for (int k = 0; k < K; ++k) {
          int tt = t + k * dilation - pad;
          if (tt < 0 || tt >= T) continue;
          s += x[(size_t)ic * T + tt] * wic[k];
        }
      }
      y_oc[t] = s;
    }
  }
}

static bool read_storage_to_f32(cl_command_queue q, cl_mem src,
                                size_t n, std::vector<float>& out) {
  std::vector<uint8_t> buf(n * sizeof(nnopt_storage_t));
  if (clEnqueueReadBuffer(q, src, CL_TRUE, 0, buf.size(), buf.data(),
                          0, nullptr, nullptr) != CL_SUCCESS) return false;
  out.assign(n, 0.0f);
#ifdef NNOPT_USE_FP16
  for (size_t i = 0; i < n; ++i) {
    uint16_t h; std::memcpy(&h, buf.data() + i * 2, 2);
    out[i] = nnopt_f16_to_f32(h);
  }
#else
  std::memcpy(out.data(), buf.data(), n * sizeof(float));
#endif
  return true;
}

static bool write_f32_to_storage(cl_command_queue q, cl_mem dst,
                                 const std::vector<float>& src) {
  const size_t n = src.size();
  std::vector<uint8_t> buf(n * sizeof(nnopt_storage_t));
#ifdef NNOPT_USE_FP16
  for (size_t i = 0; i < n; ++i) {
    uint16_t h = nnopt_f32_to_f16(src[i]);
    std::memcpy(buf.data() + i * 2, &h, 2);
  }
#else
  std::memcpy(buf.data(), src.data(), n * sizeof(float));
#endif
  return clEnqueueWriteBuffer(q, dst, CL_TRUE, 0, buf.size(), buf.data(),
                              0, nullptr, nullptr) == CL_SUCCESS;
}

}  // namespace

extern "C" cl_mem op_FlowInverse(OpenCLContext& cl_ctx,
                                 Weights& weights,
                                 cl_command_queue queue,
                                 cl_mem z_prior,
                                 int B, int C, int T,
                                 const char* /*weight_prefix unused*/) {
  if (!queue || !z_prior || B != 1 || C != kFlowChannels || T <= 0) {
    NNOPT_ERROR_FMT("op_FlowInverse: bad args B=%d C=%d T=%d", B, C, T);
    return nullptr;
  }
  if (!weights.has_tensor("flow.flows.0.conv_pre.weight")) {
    NNOPT_ERROR("op_FlowInverse: missing flow weights");
    return nullptr;
  }

  // Default to HOST path: on Adreno 620 the GPU path has ~200 small kernel
  // dispatches at ~50ms each (driver overhead dominates), while the host
  // path's parallel-loop conv runs in ~0.75s for T=110 and ~5s for T=587.
  // GPU only wins for very long T_frames (compute exceeds 10s) — set
  // NNOPT_FLOW_GPU=1 for that case.
  static int s_host_path = -1;
  if (s_host_path < 0) {
    const char* env = std::getenv("NNOPT_FLOW_GPU");
    s_host_path = (env && env[0] == '1') ? 0 : 1;
  }
  if (s_host_path) {
    // Host-side fallback (slow but correctness-preserved).
    const size_t n = (size_t)C * (size_t)T;
    std::vector<float> x;
    if (!read_storage_to_f32(queue, z_prior, n, x)) return nullptr;

    auto coupling_host = [&](int stage_idx) {
      // Split
      std::vector<float> y_a((size_t)kHalfChannels * T), y_b((size_t)kHalfChannels * T);
      for (int c = 0; c < kHalfChannels; ++c) for (int t = 0; t < T; ++t) {
        y_a[(size_t)c * T + t] = x[(size_t)c * T + t];
        y_b[(size_t)c * T + t] = x[(size_t)(c + kHalfChannels) * T + t];
      }
      if (stage_idx == 3) {
        dbg_host_stats("st3.host.y_a", y_a);
        dbg_host_stats("st3.host.y_b", y_b);
      }
      char buf[256];
      std::snprintf(buf, sizeof(buf), "flow.flows.%d.conv_pre", stage_idx);
      auto pre_W = get_w(weights, std::string(buf) + ".weight");
      auto pre_b = get_w(weights, std::string(buf) + ".bias");
      std::snprintf(buf, sizeof(buf), "flow.flows.%d.conv_post", stage_idx);
      auto post_W = get_w(weights, std::string(buf) + ".weight");
      auto post_b = get_w(weights, std::string(buf) + ".bias");

      std::vector<float> h;
      conv1d_cf_host(y_a, kHalfChannels, T, pre_W, pre_b, kFlowChannels, 1, 1, h);
      if (stage_idx == 3) dbg_host_stats("st3.host.h_pre", h);
      std::vector<float> skip((size_t)kFlowChannels * T, 0.0f);
      for (int li = 0; li < kWnLayers; ++li) {
        std::snprintf(buf, sizeof(buf), "flow.flows.%d.wavenet.in_layers.%d", stage_idx, li);
        const std::string in_pref = buf;
        std::snprintf(buf, sizeof(buf), "flow.flows.%d.wavenet.res_skip_layers.%d", stage_idx, li);
        const std::string rs_pref = buf;
        WnCacheEntry* in_e = nullptr;
        {
          auto it = g_wn_host_cache.find(in_pref);
          if (it == g_wn_host_cache.end()) {
            auto v = get_w(weights, in_pref + ".weight_v");
            auto g = get_w(weights, in_pref + ".weight_g");
            WnCacheEntry e; e.W = weight_norm_host(v, g, 2*kFlowChannels, kFlowChannels, kWnKernel);
            e.b = get_w(weights, in_pref + ".bias");
            in_e = &(g_wn_host_cache[in_pref] = std::move(e));
          } else in_e = &it->second;
        }
        std::vector<float> pre_acts;
        conv1d_cf_host(h, kFlowChannels, T, in_e->W, in_e->b, 2*kFlowChannels, kWnKernel, 1, pre_acts);
        if (stage_idx == 3 && li == 0) dbg_host_stats("st3.host.l0.pre_acts", pre_acts);
        std::vector<float> acts((size_t)kFlowChannels * T);
        for (int c = 0; c < kFlowChannels; ++c) {
          const float* ta = pre_acts.data() + (size_t)c * T;
          const float* sb = pre_acts.data() + (size_t)(c + kFlowChannels) * T;
          float* dst = acts.data() + (size_t)c * T;
          for (int t = 0; t < T; ++t) {
            float sa = sb[t];
            if (sa > 15.0f) sa = 15.0f; else if (sa < -15.0f) sa = -15.0f;
            dst[t] = std::tanh(ta[t]) * (1.0f / (1.0f + std::exp(-sa)));
          }
        }
        if (stage_idx == 3 && li == 0) dbg_host_stats("st3.host.l0.acts", acts);
        const int rs_oc = (li < kWnLayers - 1) ? (2*kFlowChannels) : kFlowChannels;
        WnCacheEntry* rs_e = nullptr;
        {
          auto it = g_wn_host_cache.find(rs_pref);
          if (it == g_wn_host_cache.end()) {
            auto v = get_w(weights, rs_pref + ".weight_v");
            auto g = get_w(weights, rs_pref + ".weight_g");
            WnCacheEntry e; e.W = weight_norm_host(v, g, rs_oc, kFlowChannels, 1);
            e.b = get_w(weights, rs_pref + ".bias");
            rs_e = &(g_wn_host_cache[rs_pref] = std::move(e));
          } else rs_e = &it->second;
        }
        std::vector<float> rs_out;
        conv1d_cf_host(acts, kFlowChannels, T, rs_e->W, rs_e->b, rs_oc, 1, 1, rs_out);
        if (stage_idx == 3 && li == 0) dbg_host_stats("st3.host.l0.rs", rs_out);
        if (li < kWnLayers - 1) {
          for (int c = 0; c < kFlowChannels; ++c) for (int t = 0; t < T; ++t) {
            h[(size_t)c * T + t]    += rs_out[(size_t)c * T + t];
            skip[(size_t)c * T + t] += rs_out[(size_t)(c + kFlowChannels) * T + t];
          }
        } else {
          for (size_t i = 0; i < skip.size(); ++i) skip[i] += rs_out[i];
        }
      }
      if (stage_idx == 3) dbg_host_stats("st3.host.skip_final", skip);
      std::vector<float> t_mean;
      conv1d_cf_host(skip, kFlowChannels, T, post_W, post_b, kHalfChannels, 1, 1, t_mean);
      if (stage_idx == 3) dbg_host_stats("st3.host.t_mean", t_mean);
      for (int c = 0; c < kHalfChannels; ++c) for (int t = 0; t < T; ++t) {
        x[(size_t)c * T + t] = y_a[(size_t)c * T + t];
        x[(size_t)(c + kHalfChannels) * T + t] =
            y_b[(size_t)c * T + t] - t_mean[(size_t)c * T + t];
      }
    };

    for (int stage = kFlowStages - 1; stage >= 0; --stage) {
      // flip
      for (int c = 0; c < C / 2; ++c) {
        int cp = C - 1 - c;
        for (int t = 0; t < T; ++t)
          std::swap(x[(size_t)c * T + t], x[(size_t)cp * T + t]);
      }
      coupling_host(stage);
    }

    cl_int err = CL_SUCCESS;
    cl_mem out = clCreateBuffer(cl_ctx.context(), CL_MEM_READ_WRITE,
                                n * sizeof(nnopt_storage_t), nullptr, &err);
    if (err != CL_SUCCESS || !out) return nullptr;
    if (!write_f32_to_storage(queue, out, x)) { clReleaseMemObject(out); return nullptr; }
    NNOPT_LAYER_CHECK("flow_output_cf", queue, out, n);
    return out;
  }

  // ── GPU path ─────────────────────────────────────────────────────────
  cl_context ctx = cl_ctx.context();
  cl_int err = CL_SUCCESS;
  const size_t n = (size_t)C * (size_t)T;

  // Allocate the working buffer that holds x throughout the inverse pipeline.
  cl_mem x_buf = clCreateBuffer(ctx, CL_MEM_READ_WRITE,
                                n * sizeof(nnopt_storage_t), nullptr, &err);
  if (err != CL_SUCCESS || !x_buf) {
    NNOPT_ERROR_FMT("op_FlowInverse: x_buf alloc (%d)", (int)err);
    return nullptr;
  }
  // Copy z_prior → x_buf (z_prior is read-only relative to caller).
  if (clEnqueueCopyBuffer(queue, z_prior, x_buf, 0, 0,
                          n * sizeof(nnopt_storage_t), 0, nullptr, nullptr) != CL_SUCCESS) {
    clReleaseMemObject(x_buf);
    return nullptr;
  }

  // Allocate one extra scratch buffer for channel_flip's out-of-place output.
  cl_mem x_scratch = clCreateBuffer(ctx, CL_MEM_READ_WRITE,
                                    n * sizeof(nnopt_storage_t), nullptr, &err);
  if (err != CL_SUCCESS || !x_scratch) {
    clReleaseMemObject(x_buf);
    return nullptr;
  }

  for (int stage = kFlowStages - 1; stage >= 0; --stage) {
    if (!dispatch_channel_flip(cl_ctx, queue, x_buf, x_scratch, C, T)) {
      clReleaseMemObject(x_buf); clReleaseMemObject(x_scratch);
      return nullptr;
    }
    std::swap(x_buf, x_scratch);  // x_buf now holds the flipped data
    if (!coupling_inverse_stage_gpu(cl_ctx, queue, weights, stage, x_buf, T)) {
      clReleaseMemObject(x_buf); clReleaseMemObject(x_scratch);
      return nullptr;
    }
  }
  clReleaseMemObject(x_scratch);

  NNOPT_LAYER_CHECK("flow_output_cf", queue, x_buf, n);
  NNOPT_CHECKPOINT_FMT("op_FlowInverse GPU: T=%d stages=%d", T, kFlowStages);
  return x_buf;
}
