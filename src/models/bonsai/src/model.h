// OpenCL device model — mirrors the token-exact P1 host forward 1:1.
// Weights uploaded ONCE at init (packed Q1 bytes verbatim; f32 norms).
// storage_t is fp32 first (P2 gate), -DUSE_FP16 in kernel builds is the
// P3 A/B. Kernel families live in SEPARATE programs (Adreno register-
// footprint isolation): q1_gemv.cl / rmsnorm.cl / rope.cl / attention.cl /
// mlp.cl / utils.cl.
//
// Class DECLARATION only. Method bodies are defined out-of-line across
// model.cpp and layers/{attention,mlp,embedding,layer_norm}.cpp.
#pragma once
#include <algorithm>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <map>
#include <string>
#include <thread>
#include <vector>

#include "rope.h"            // RopeYarn (token-exact host math)
#include "nnb.h"             // Nnb / Tensor / ModelMeta
#include "q1.h"              // half_to_float / float_to_half
#include "opencl_context.h"

// BONSAI_PROF=1: clFinish-bracketed wall time per op family (the honest
// profile — CL event times on CLBlast-style multi-kernel ops undercount).
struct OpProf {
    bool on = false;
    std::map<std::string, std::pair<double,int>> acc;
    std::chrono::steady_clock::time_point t0;
    cl_command_queue q = nullptr;
    void begin() { if (on) { clFinish(q); t0 = std::chrono::steady_clock::now(); } }
    void end(const char* k) {
        if (!on) return;
        clFinish(q);
        double ms = std::chrono::duration<double, std::milli>(
                        std::chrono::steady_clock::now() - t0).count();
        acc[k].first += ms; acc[k].second++;
    }
    void dump() {
        if (!on) return;
        double tot = 0;
        for (auto& kv : acc) tot += kv.second.first;
        fprintf(stderr, "=== DEVICE PROFILE (BONSAI_PROF=1) ===\n");
        for (auto& kv : acc)
            fprintf(stderr, "%-14s %10.1f ms %7.1f%% %8d calls %8.3f ms/call\n",
                    kv.first.c_str(), kv.second.first,
                    100.0 * kv.second.first / tot, kv.second.second,
                    kv.second.first / kv.second.second);
        fprintf(stderr, "TOTAL %10.1f ms\n", tot);
    }
};

#define CLCHECK(err, what)                                            \
    do {                                                              \
        cl_int e_ = (err);                                            \
        if (e_ != CL_SUCCESS) {                                       \
            fprintf(stderr, "FATAL CL %d at %s\n", e_, what);         \
            exit(4);                                                  \
        }                                                             \
    } while (0)

class DeviceModel {
  public:
    static constexpr int CTX_CAP = 2048;
    static constexpr int M_MAX = 8;
#ifdef BONSAI_FP16_STORAGE
    static constexpr size_t ES = 2;    // storage_t = half on device
#else
    static constexpr size_t ES = 4;    // storage_t = float
#endif

    DeviceModel(const Nnb& nnb, OpenCLContext& ocl, const std::string& kdir);
    ~DeviceModel();

    // Forward one token at pos. If want_logits, runs the output head and
    // device argmax; returns argmax id (1-int readback). Else returns -1
    // (prefill tokens skip the 87MB logits GEMV entirely).
    // token >= 0: explicit (prefill / first decode step).
    // token < 0: embedding gather reads the PREVIOUS argmax result from
    //            the device buffer — no host readback in the loop.
    int forward(int token, int pos, bool want_logits);

    // Batched prefill: process M prompt tokens (<= M_MAX) in one pass —
    // every weight is streamed ONCE for the whole chunk. Per-row GEMV math
    // is byte-identical to v4's, so KV contents keep the token-exact gate.
    void prefill(const int* tokens, int M, int pos0);

    int read_token();   // blocking read of the latest argmax result

    void print_topk();

  private:
    struct QW { cl_mem bits, scales, bits_img = nullptr; };   // split-stream 1-bit weight
    struct LayerW {
        QW wqkv, wo, wgu, wd;
        cl_mem attn_norm, ffn_norm, q_norm, k_norm;
        const uint8_t* up_host = nullptr;   // ffn_up  original Q1 (hybrid CPU rows)
        const uint8_t* down_host = nullptr; // ffn_down original Q1 (hybrid CPU rows)
    };

    void build_programs(const std::string& kdir);
    cl_mem make_bits_image(cl_mem buf, size_t texels);
    cl_mem upload(const void* p, size_t n, const char* what);
    cl_mem scratch(size_t n, const char* what);
    // Split-stream repack: 18-byte units -> aligned bits[] (16B/unit) +
    // scales[] (fp16/unit). Same bytes, still 1-bit packed — deinterleaved
    // so the kernel's loads are aligned vector ops. One-time host cost.
    // Fused upload: concatenate tensors along the OUTPUT dim (all share K)
    // into one K-major transposed pair — one GEMV launch for QKV (N=6144)
    // and gate+up (N=24576) instead of 3+2 small ones (k/v alone were 4
    // workgroups — the GPU sat idle).
    QW upload_q1_fused(const std::vector<std::string>& names);
    QW upload_q1(const std::string& n);
    void upload_weights();
    void make_scratch();
    void make_rope_tables();

    // ---- dispatch helpers ----
    void arg(cl_kernel k, int i, size_t sz, const void* v, const char* what) {
        CLCHECK(clSetKernelArg(k, i, sz, v), what);
    }
    void run1(cl_kernel k, size_t gws, size_t lws, const char* what) {
        CLCHECK(clEnqueueNDRangeKernel(ocl_.queue(), k, 1, nullptr, &gws,
                                       lws ? &lws : nullptr, 0, nullptr, nullptr),
                what);
    }
    void run2(cl_kernel k, size_t g0, size_t g1, const char* what) {
        size_t g[2] = {g0, g1};
        CLCHECK(clEnqueueNDRangeKernel(ocl_.queue(), k, 2, nullptr, g, nullptr,
                                       0, nullptr, nullptr), what);
    }
    void run_gemv_b(QW& W, cl_mem x, cl_mem out, int N, int K, int Ntot,
                    int row_off, int M);
    void run_xsum_b(cl_mem x, int K, int M);
    void run_gather_b(QW& W, cl_mem out, int M, int K);
    void run_rope_m(cl_mem qb, cl_mem kb, int NH, int KH, int D, int M);
    void run_scores_m(cl_mem qb, cl_mem kc, int NH, int KH, int D, int M,
                      int seq_k);
    void run_softmax_m(int NH, int M);
    void run_attnout_m(cl_mem vc, int NH, int KH, int D, int M);
    void run_kvq4(cl_mem src, cl_mem dst, int rows, int KH, int D, int dst_off);
    void run_scores4(cl_mem qb, cl_mem kc, int NH, int KH, int D, int seq_k);
    void run_attnout4(cl_mem vc, int NH, int KH, int D);
    void run_gather_dev(QW& W, cl_mem out, int K);
    void run_gather(QW& W, cl_mem out, int token, int K);
    void run_xsum(cl_mem x, int K);
    // n_dispatch (default N): compute only output rows [0, n_dispatch). The
    // kernel arg N stays the TRUE row count so the transposed weight stride
    // bits_t[u*N + row] is preserved — only the launch size shrinks. Used by the
    // hybrid to hand the top rows to the CPU. Callers align n_dispatch to 256.
    void run_gemv(QW& W, cl_mem x, cl_mem out, int N, int K, int out_off,
                  int n_dispatch = -1);
    // Hybrid GEMV: GPU computes rows [0, N-ncpu); CPU computes the top `ncpu`
    // rows straight from original-layout Q1 weights (cpu_w, starting at
    // cpu_src_row0) in parallel. x + xsum are read back FIRST so the in-order
    // queue lets the GPU GEMV and the CPU threads overlap. cpu_src_row0 lets the
    // fused gate+up map its top rows onto the standalone ffn_up tensor.
    void run_gemv_hybrid(QW& W, cl_mem x, cl_mem out, int N, int K, int out_off,
                         const uint8_t* cpu_w, int cpu_src_rows);
    void run_rms(cl_mem x, cl_mem w, cl_mem out, int rows, int cols);
    void run_rope(cl_mem qb, cl_mem kb, int NH, int KH, int D);
    void run_scores(cl_mem qb, cl_mem kc, int NH, int KH, int D, int seq_k);
    void run_softmax(int NH, int seq_k);
    void run_attnout(cl_mem vc, int NH, int KH, int D);
    void run_swiglu(cl_mem g, cl_mem u, int n);
    void run_add(cl_mem acc, cl_mem b, int n);
    void run_argmax(int N);

    OpProf prof_;
    const Nnb& nnb_;
    const ModelMeta& m_;
    OpenCLContext& ocl_;
    std::vector<LayerW> lw_;
    // CPU+GPU hybrid state
    bool hybrid_ = false;
    bool hybrid_readonly_ = false;
    float hybrid_frac_ = 0.22f;
    int hybrid_threads_ = 8;
    std::vector<float> hx_, hout_;   // host x readback + CPU output slice
    QW tok_embd_, w_out_;
    cl_mem out_norm_, cos_, sin_, counter_, amax_;
    cl_mem x_, xb_, xb2_, qkv_, qv_, kv_, vv_, att_, scores_, gu_, gate_, up_, logits_, xsum_;
    cl_mem xp_, xbp_, xb2p_, qp_, kp_, vp_, attp_, gatep_, upp_, xsump_, tokbuf_;
    std::vector<cl_mem> kcache_, vcache_;
    cl_kernel k_xsum_, k_gemv_, k_gemv7_, k_gather_, k_gather_dev_,
              k_gemv_b_, k_xsum_b_, k_gather_b_, k_argmax_, k_rms_, k_rope_,
              k_scores_, k_softmax_, k_attnout_, k_swiglu_, k_add_;
    cl_kernel k_kvq4_ = nullptr, k_scores4_ = nullptr, k_attnout4_ = nullptr;
    cl_kernel k_gemv_img_ = nullptr;
    bool use_img_ = false;
    bool no_xsum_ = false;
    bool kv4_ = false;
};
