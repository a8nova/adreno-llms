// Linear.cpp — shared implementation for ALL Linear nodes (nn.Linear).
// Reference: model_info/transformers_src/modeling_moonshine.py — nn.Linear
//   used everywhere: q/k/v/o_proj, mlp.fc1/fc2, proj_out.
//   nn.Linear stores weight as [out_features, in_features]; forward is
//   y = x @ W^T (+ bias if present). We use pytorch_linear (Transpose::kYes).
//
// weight_prefix here is the FULL weight-key base, e.g.
//   "model.encoder.layers.0.self_attn.q_proj"  → loads ".weight" (+ optional ".bias")
// seq_len = number of rows (M). The input/output feature dims are derived
// from the weight shape [N, K].

#include "../opencl_context.h"
#include "../weights.h"
#include "../nnopt_error.h"
#include "../model_config.h"
#include "../utils.h"
#include "../profiler.h"
#include <cstdio>
#include <cstdlib>
#include <string>
#include <unordered_map>
#include <vector>

// OPT-6: process-lifetime kernel cache (moonshine_common.cpp) — do not release.
cl_kernel moonshine_kernel(OpenCLContext& cl_ctx, const char* name);

// OPT-4: transposed-weight cache for the huge-N GEMV (proj_out 32768×288).
// Keyed by the source weight buffer; entries live as long as the process
// (weight buffers are loaded once and never released mid-run). ~19MB extra
// for proj_out — buys a fully-coalesced K-major GEMV.
static std::unordered_map<cl_mem, cl_mem> g_transposed_weights;

// Non-static: Conv1d.cpp reuses this for its im2col GEMM weights (a conv
// weight [Cout, Cin, K] is bitwise a nn.Linear weight [Cout, Cin*K]).
cl_mem transposed_weight_for(OpenCLContext& cl_ctx, cl_command_queue queue,
                             cl_mem w_buf, int N, int K) {
    auto it = g_transposed_weights.find(w_buf);
    if (it != g_transposed_weights.end()) return it->second;
    cl_int err = CL_SUCCESS;
    cl_mem wt = clCreateBuffer(cl_ctx.context(), CL_MEM_READ_WRITE,
                               (size_t)N * K * sizeof(nnopt_storage_t), nullptr, &err);
    if (err != CL_SUCCESS || !wt) { NNOPT_ERROR_FMT("transpose alloc %d", err); return nullptr; }
    cl_kernel tk = moonshine_kernel(cl_ctx, "mat_transpose");
    if (!tk) { clReleaseMemObject(wt); return nullptr; }
    const size_t TR = 16;   // must match TR_TILE in kernels/gemm.cl
    bool ok = set_arg_checked(tk, 0, sizeof(cl_mem), &w_buf, "W") &&
              set_arg_checked(tk, 1, sizeof(cl_mem), &wt, "WT") &&
              set_arg_checked(tk, 2, sizeof(int), &N, "N") &&
              set_arg_checked(tk, 3, sizeof(int), &K, "K") &&
              clSetKernelArg(tk, 4, TR * (TR + 1) * sizeof(float), nullptr) == CL_SUCCESS;
    if (ok) {
        size_t gws[2] = { ((size_t)K + TR - 1) / TR * TR, ((size_t)N + TR - 1) / TR * TR };
        size_t lws[2] = { TR, TR };
        ok = clEnqueueNDRangeKernel(queue, tk, 2, nullptr, gws, lws, 0, nullptr,
                                    KernelProfiler::event_for("mat_transpose_once")) == CL_SUCCESS;
    }
    if (!ok) { clReleaseMemObject(wt); return nullptr; }
    g_transposed_weights[w_buf] = wt;
    return wt;
}

extern "C" {
cl_mem Linear_forward(
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
    (void)layer_idx; (void)start_pos;
    (void)k_cache_inout; (void)v_cache_inout; (void)encoder_hidden_states;

    const std::string wp = weight_prefix ? std::string(weight_prefix) : std::string();
    if (wp.empty()) { NNOPT_ERROR("Linear_forward: empty weight_prefix"); return nullptr; }

    std::vector<int> wshape = weights.get_shape(wp + ".weight");
    if (wshape.size() != 2) {
        NNOPT_ERROR_FMT("Linear: bad weight shape for %s (rank=%zu)", wp.c_str(), wshape.size());
        return nullptr;
    }
    const int N = wshape[0];  // out_features
    const int K = wshape[1];  // in_features
    const int M = seq_len;

    cl_mem w_buf = weights.get_buffer(wp + ".weight");
    if (!w_buf) { NNOPT_ERROR_FMT("Linear: missing %s.weight", wp.c_str()); return nullptr; }

    cl_int err = CL_SUCCESS;
    cl_mem out = clCreateBuffer(cl_ctx.context(), CL_MEM_READ_WRITE,
                               (size_t)M * N * sizeof(nnopt_storage_t), nullptr, &err);
    if (err != CL_SUCCESS || !out) { NNOPT_ERROR_FMT("Linear: alloc out %d", err); return nullptr; }

    // OPT-4 (r4): M=1 (decode step) never touches CLBlast — its Gemm at M=1
    // ran at ~1% of DRAM bandwidth on Adreno 620 (920µs per 288×288 call,
    // 58% of total GPU time in the r3 profile), and routing just proj_out
    // back to it cost 12 tk/s of decode (hidden pad/transpose pre-kernels
    // its event never showed). Two bespoke kernels instead:
    //   N ≤ 4096: workgroup-per-row + local reduction (linear_gemv)
    //   N > 4096: one-time weight transpose + coalesced K-major 4-out
    //             lanes (linear_gemv_t) — the row-scan shape underutilizes
    //             at 32768 rows (10.2ms/call measured).
    // M>1 (encoder / prefill) stays on CLBlast's tuned GEMM path.
    bool gemm_ok = false;
    if (M == 1) {
        // A/B toggle: NNOPT_GEMVT=0 forces the row-scan kernel for huge N too
        // (DVFS on this device makes cross-build wall comparisons unreliable —
        // same-binary A/B in one thermal window is the only fair test).
        static const bool gemvt_enabled = [] {
            const char* e = std::getenv("NNOPT_GEMVT");
            return !(e && e[0] == '0');
        }();
        // Default 0: EVERY M=1 linear goes through the transposed coalesced
        // GEMV. On-device interleaved A/B (2026-07-03, DVFS-controlled):
        // all-gemvt 37.6 tk/s > fc1+proj only 35.5 > proj only 25.0 >
        // row-scan everything 14.9 > CLBlast 10.4.
        static const int gemvt_min_n = [] {
            const char* e = std::getenv("NNOPT_GEMVT_MIN_N");
            return e ? std::atoi(e) : 0;
        }();
        const bool huge_n = gemvt_enabled && (N > gemvt_min_n) && (N % 4 == 0);
        cl_mem wt = huge_n ? transposed_weight_for(cl_ctx, queue, w_buf, N, K) : nullptr;
        // OPT-11 (r10): small-N shapes (N < 1024 → ≤256 lanes) starve the GPU
        // on the single-pass kernel (185µs for a 166KB read). K-split into S
        // fp32 partial chunks + reduce. NNOPT_GEMV_KSPLIT=0 reverts.
        static const bool ksplit_enabled = [] {
            const char* e = std::getenv("NNOPT_GEMV_KSPLIT");
            return !(e && e[0] == '0');
        }();
        const bool ksplit = ksplit_enabled && wt && (N < 1024);
        if (ksplit) {
            const int S = (K >= 1024) ? 8 : 4;
            // Reusable fp32 partial scratch (grown to max S*N seen; decode
            // shapes are fixed so this allocates twice per process).
            static cl_mem s_partial = nullptr;
            static size_t s_partial_floats = 0;
            const size_t need = (size_t)S * N;
            if (need > s_partial_floats) {
                if (s_partial) clReleaseMemObject(s_partial);
                s_partial = clCreateBuffer(cl_ctx.context(), CL_MEM_READ_WRITE,
                                           need * sizeof(float), nullptr, &err);
                s_partial_floats = s_partial ? need : 0;
            }
            cl_kernel kk = moonshine_kernel(cl_ctx, "linear_gemv_ks");
            cl_kernel rk = moonshine_kernel(cl_ctx, "gemv_ks_reduce");
            if (s_partial && kk && rk) {
                const size_t lws = 64;
                bool ok = set_arg_checked(kk, 0, sizeof(cl_mem), &input, "x") &&
                          set_arg_checked(kk, 1, sizeof(cl_mem), &wt, "WT") &&
                          set_arg_checked(kk, 2, sizeof(cl_mem), &s_partial, "partial") &&
                          set_arg_checked(kk, 3, sizeof(int), &N, "N") &&
                          set_arg_checked(kk, 4, sizeof(int), &K, "K") &&
                          set_arg_checked(kk, 5, sizeof(int), &S, "S");
                if (ok) {
                    char label[48];
                    snprintf(label, sizeof(label), "gemvks_n%d_k%d", N, K);
                    size_t gws[2] = { (((size_t)(N / 4) + lws - 1) / lws) * lws, (size_t)S };
                    size_t lws2[2] = { lws, 1 };
                    ok = clEnqueueNDRangeKernel(queue, kk, 2, nullptr, gws, lws2, 0, nullptr,
                                                KernelProfiler::event_for(label)) == CL_SUCCESS;
                }
                if (ok) {
                    ok = set_arg_checked(rk, 0, sizeof(cl_mem), &s_partial, "partial") &&
                         set_arg_checked(rk, 1, sizeof(cl_mem), &out, "out") &&
                         set_arg_checked(rk, 2, sizeof(int), &N, "N") &&
                         set_arg_checked(rk, 3, sizeof(int), &S, "S");
                    if (ok) {
                        size_t gws = (size_t)N;
                        ok = clEnqueueNDRangeKernel(queue, rk, 1, nullptr, &gws, nullptr, 0, nullptr,
                                                    KernelProfiler::event_for("gemvks_reduce")) == CL_SUCCESS;
                    }
                }
                gemm_ok = ok;
            }
        }
        const char* kname = (huge_n && wt) ? "linear_gemv_t" : "linear_gemv";
        cl_kernel gk = !gemm_ok ? moonshine_kernel(cl_ctx, kname) : nullptr;
        if (gk) {
            const size_t lws = 64;
            cl_mem wsrc = (huge_n && wt) ? wt : w_buf;
            const size_t local_bytes = (huge_n && wt) ? (size_t)K * sizeof(float)
                                                      : lws * sizeof(float);
            if (set_arg_checked(gk, 0, sizeof(cl_mem), &input, "x") &&
                set_arg_checked(gk, 1, sizeof(cl_mem), &wsrc, "W") &&
                set_arg_checked(gk, 2, sizeof(cl_mem), &out, "out") &&
                set_arg_checked(gk, 3, sizeof(int), &N, "N") &&
                set_arg_checked(gk, 4, sizeof(int), &K, "K") &&
                clSetKernelArg(gk, 5, local_bytes, nullptr) == CL_SUCCESS) {
                char label[48];
                snprintf(label, sizeof(label), "%s_n%d_k%d",
                         (huge_n && wt) ? "gemvt" : "gemv", N, K);
                size_t lanes = (huge_n && wt) ? (size_t)(N / 4) : (size_t)N * lws;
                size_t gws = ((lanes + lws - 1) / lws) * lws;
                gemm_ok = clEnqueueNDRangeKernel(queue, gk, 1, nullptr, &gws, &lws, 0, nullptr,
                                                 KernelProfiler::event_for(label)) == CL_SUCCESS;
            }
        }
        if (!gemm_ok) NNOPT_ERROR_FMT("Linear: M=1 GEMV dispatch failed for %s — falling back to CLBlast", wp.c_str());
    } else if (M > 1 && (N % 4 == 0)) {
        // OPT-6 (r6): M>1 (encoder / prefill) also bypasses CLBlast — not for
        // GPU time (its GEMM was fine) but because its lazy first-call kernel
        // compile burned ~1.4s of TTFT in EVERY process launch. Same
        // transposed coalesced walk as the GEMV, one WG row per (m, n-tile).
        cl_mem wt = transposed_weight_for(cl_ctx, queue, w_buf, N, K);
        cl_kernel gk = wt ? moonshine_kernel(cl_ctx, "linear_gemm_t4") : nullptr;
        if (gk) {
            const size_t lws = 64;
            const size_t MT = 4, KT = 256;   // must match GEMM_MT/GEMM_KT
            if (set_arg_checked(gk, 0, sizeof(cl_mem), &input, "x") &&
                set_arg_checked(gk, 1, sizeof(cl_mem), &wt, "WT") &&
                set_arg_checked(gk, 2, sizeof(cl_mem), &out, "out") &&
                set_arg_checked(gk, 3, sizeof(int), &M, "M") &&
                set_arg_checked(gk, 4, sizeof(int), &N, "N") &&
                set_arg_checked(gk, 5, sizeof(int), &K, "K") &&
                clSetKernelArg(gk, 6, MT * KT * sizeof(float), nullptr) == CL_SUCCESS) {
                char label[48];
                snprintf(label, sizeof(label), "gemmt_m%d_n%d_k%d", M, N, K);
                size_t n_lanes = (size_t)(N / 4);
                size_t gws[2] = { ((n_lanes + lws - 1) / lws) * lws, ((size_t)M + MT - 1) / MT };
                size_t lws2[2] = { lws, 1 };
                gemm_ok = clEnqueueNDRangeKernel(queue, gk, 2, nullptr, gws, lws2, 0, nullptr,
                                                 KernelProfiler::event_for(label)) == CL_SUCCESS;
            }
        }
        if (!gemm_ok) NNOPT_ERROR_FMT("Linear: linear_gemm_t4 dispatch failed for %s — falling back to CLBlast", wp.c_str());
    }
    if (!gemm_ok && !pytorch_linear(queue, M, N, K, input, w_buf, out)) {
        NNOPT_ERROR_FMT("Linear: pytorch_linear failed for %s", wp.c_str());
        clReleaseMemObject(out);
        return nullptr;
    }

    // Optional bias (encoder mlp.fc1/fc2, conv2/conv3 have bias; q/k/v/o_proj do not).
    if (weights.has_tensor(wp + ".bias")) {
        cl_mem b_buf = weights.get_buffer(wp + ".bias");
        cl_kernel k = b_buf ? moonshine_kernel(cl_ctx, "bias_add") : nullptr;
        if (k) {
            if (set_arg_checked(k, 0, sizeof(cl_mem), &out, "in") &&
                set_arg_checked(k, 1, sizeof(cl_mem), &b_buf, "bias") &&
                set_arg_checked(k, 2, sizeof(cl_mem), &out, "out") &&
                set_arg_checked(k, 3, sizeof(int), &M, "rows") &&
                set_arg_checked(k, 4, sizeof(int), &N, "cols")) {
                size_t gws = (size_t)M * N;
                clEnqueueNDRangeKernel(queue, k, 1, nullptr, &gws, nullptr, 0, nullptr,
                                       KernelProfiler::event_for("bias_add"));
            }
        }
    }
    return out;
}
}
