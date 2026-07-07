// Shared infrastructure for the Depth-Anything-V2 pipeline stages.
// Split from the former monolithic backbone.cpp (adreno-llms convention:
// one file per module under src/layers/, shared plumbing here).
//
// Reference: model_info/transformers_src/modeling_dinov2.py + modeling_depth_anything.py
#pragma once

#include "opencl_context.h"
#include "weights.h"
#include "debug_utils.h"
#include "model_config.h"
#include "utils.h"
#include "profiler.h"   // KernelProfiler — dormant unless NNOPT_PROFILE=1
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <map>
#include <string>
#include <vector>

// ─────────────────────────────────────────────────────────────────────
// fp16/fp32 storage helpers
// ─────────────────────────────────────────────────────────────────────
static inline nnopt_storage_t enc(float v) {
#ifdef NNOPT_USE_FP16
    return nnopt_storage_t(nnopt_f32_to_f16(v));
#else
    return v;
#endif
}
static inline float dec(nnopt_storage_t s) {
#ifdef NNOPT_USE_FP16
    return nnopt_f16_to_f32(static_cast<uint16_t>(s));
#else
    return s;
#endif
}

// A holder for the compiled depth kernels. Built ONCE per process.
struct DepthKernels {
    bool built = false;   // programs are per kernel FILE — see build_depth_kernels
    cl_kernel conv2d = nullptr;
    cl_kernel conv_transpose2d = nullptr;
    cl_kernel conv_transpose2d_sk = nullptr;   // stride==kernel fast path
    cl_kernel embeddings_assemble = nullptr;
    cl_kernel layernorm_rows = nullptr;
    cl_kernel gelu = nullptr;
    cl_kernel relu = nullptr;
    cl_kernel layer_scale = nullptr;
    cl_kernel add_bias_rows = nullptr;
    cl_kernel attn_scores = nullptr;
    cl_kernel attn_softmax = nullptr;
    cl_kernel attn_context = nullptr;
    cl_kernel to_head_major = nullptr;
    cl_kernel tokens_to_chw = nullptr;
    cl_kernel chw_to_tokens = nullptr;
    cl_kernel bilinear_interp = nullptr;
    cl_kernel add = nullptr;
    cl_kernel im2col2d = nullptr;
    cl_kernel im2col2d_v4 = nullptr;   // 4 output positions per WI
    cl_kernel bias_per_row = nullptr;
    cl_kernel attn_softmax_wg = nullptr;
};

extern cl_context g_ctx;
extern cl_command_queue g_q;
extern DepthKernels g_k;

// ── Activation buffer pool (campaign 2026-07-05). Exact-size free-list;
// pooling is correctness-critical (heap fragmentation → CL_OUT_OF_RESOURCES).
cl_mem alloc_buf(size_t n);
void pool_release(cl_mem b);

// Host transfer helpers (storage_t <-> fp32).
cl_mem upload_f32(const std::vector<float>& v);
std::vector<float> download_f32(cl_mem b, size_t n);

// Dispatch + set-arg helpers.
void run1d(cl_kernel k, size_t total);
bool sa(cl_kernel k, int idx, size_t sz, const void* p);

// Persistent grow-only im2col scratch (chunked under COL_CAP_ELEMS).
extern const size_t COL_CAP_ELEMS;
cl_mem col_scratch(size_t elems);

// ── Sync-profiler for CLBlast composite routines (NNOPT_SYNC_PROF=1).
// CLBlast's cl_event maps only to the LAST kernel of its internal chain, so
// the event ledger undercounts it ~10×. SyncScope brackets a CLBlast call in
// clFinish + host clock: true composite cost. clFinish serialization inflates
// absolute wall — attribution tool ONLY, never an A/B number.
bool sync_prof_on();
extern std::map<std::string, double> g_sync_ms;
extern std::map<std::string, int>    g_sync_calls;
struct SyncScope {
    const char* l; bool on; std::chrono::steady_clock::time_point t0;
    explicit SyncScope(const char* label) : l(label), on(sync_prof_on()) {
        if (on) { clFinish(g_q); t0 = std::chrono::steady_clock::now(); }
    }
    ~SyncScope() {
        if (on) {
            clFinish(g_q);
            g_sync_ms[l] += std::chrono::duration<double, std::milli>(
                std::chrono::steady_clock::now() - t0).count();
            g_sync_calls[l]++;
        }
    }
};
void sync_prof_dump();

// Row-major C = A·B via CLBlast (beta=0), SyncScope-instrumented.
bool gemm_nn(int M, int N, int K, cl_mem A, size_t a_off, int lda,
             cl_mem B, size_t b_off, int ldb,
             cl_mem C, size_t c_off, int ldc, const char* label);

// ── Op wrappers (each dispatches kernels from kernels/<op>.cl) ──
cl_mem conv2d(cl_mem in, int Cin, int Hin, int Win,
              cl_mem w, cl_mem bias, int Cout, int KH, int KW,
              int stride, int pad, int& Hout, int& Wout);
cl_mem conv_transpose2d(cl_mem in, int Cin, int Hin, int Win,
                        cl_mem w, cl_mem bias, int Cout, int KH, int KW,
                        int stride, int& Hout, int& Wout);
cl_mem bilinear(cl_mem in, int Cc, int Hin, int Win, int Hout, int Wout, int align);
cl_mem layernorm(cl_mem in, int rows, int D, cl_mem gamma, cl_mem beta);
cl_mem act_gelu(cl_mem in, size_t n);
cl_mem act_relu(cl_mem in, size_t n);
cl_mem elt_add(cl_mem x, cl_mem y, size_t n);
cl_mem layer_scale(cl_mem in, cl_mem lambda, int rows, int D);
void add_bias(cl_mem buf, cl_mem bias, int rows, int D);
// nn.Linear: out[rows,N] = in[rows,K] @ W[N,K]^T + bias. W is [N,K] row-major.
cl_mem linear(cl_mem in, int rows, int N, int K, cl_mem w, cl_mem bias);

// Build every kernel program ONCE (one cl_program per kernel file).
bool build_depth_kernels(OpenCLContext& cl_ctx);

// LAYER_CHECK helper: pull first `n` elems (capped) as storage_t and check.
// The reference-side capture truncates every dump to a leading slice of
// NNOPT_REF_DUMP_CAP floats; the C++ dump MUST write the SAME leading-prefix
// count or SxS reports a spurious size-mismatch.
constexpr size_t NNOPT_REF_DUMP_CAP = 262144;
#define DCHECK(name, buf, n) \
    NNOPT_LAYER_CHECK(name, g_q, buf, \
        ((size_t)(n) > NNOPT_REF_DUMP_CAP ? NNOPT_REF_DUMP_CAP : (size_t)(n)))

// Per-sub-module DCHECK dump with the encoder-layer-scoped name.
void layer_dump(int layer_idx, const char* sub, cl_mem buf, size_t n);

// ── Pipeline stages ──
// embeddings: pixel_values → assembled token matrix [M, D] (patch conv +
// bicubic-interpolated position embeddings + cls token).
cl_mem embeddings_stage(Weights& Wt, const std::vector<float>& pixel_values,
                        int C, int H, int W);
// one DINOv2 encoder layer (Dinov2Layer) on the token matrix.
cl_mem encoder_layer(Weights& W, cl_mem h, int M, int layer_idx);
// DPT neck (reassemble + fusion) + depth head. Consumes (releases) feats.
// Returns the final depth map as host fp32 (already ×max_depth).
std::vector<float> dpt_neck_head(Weights& Wt, cl_mem feats[4],
                                 int M, int D, int Hp, int Wp, int PS);
