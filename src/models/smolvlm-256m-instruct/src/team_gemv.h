// fp16 decode GEMVs for GPUs without a usable texture path (kernels/gemv_team.cl).
// Same kernels and policy as smollm2-135m-instruct/src/team_gemv.h; this copy is
// keyed on the command queue because SmolVLM's GEMV entry points (utils.cpp)
// only receive a queue.
//
// On by default exactly where texture_policy.h denies textures (PowerVR Rogue),
// i.e. where decode would otherwise use the buffer GEMVs / CLBlast at M=1.
//
//   NNOPT_TEAM_GEMV=1   force on (exercise the path on Adreno)
//   NNOPT_TEAM_GEMV=0   force off
//   NNOPT_TEAM_MODE=m   one LOADMODE for every shape (test hook)
//   NNOPT_TEAM_SWEEP=1  (or -DNNOPT_TEAM_SWEEP) time candidate geometries per
//                       decode shape on first use, print them, keep the fastest.
#pragma once

#include <CL/cl.h>

#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <map>
#include <sstream>
#include <string>
#include <tuple>
#include <vector>

#include "opencl_context.h"
#include "texture_policy.h"

namespace team_gemv {

enum Shape { QKV = 0, OPROJ, GATEUP, DOWN, LMHEAD, NUM_SHAPES };
struct Geo { int wg, lpr, rpl, mode, xl = 0, unr = 1; };   // LOADMODE, x in __local, chunk unroll

inline Geo& geo(Shape s) {
    static Geo g[NUM_SHAPES] = {
        {64,  8, 4, 1},   // QKV     N=960   K=576    0.68 ms (norm fused)  GE8320 sweep 2026-10-03
        {64, 16, 2, 1},   // OPROJ   N=576   K=576    0.35 ms
        {64, 16, 4, 1},   // GATEUP  N=1536  K=576    1.51 ms (norm fused, x2 matrices)
        {64, 16, 4, 1},   // DOWN    N=576   K=1536   0.75 ms
        {64, 16, 2, 1},   // LMHEAD  N=49280 K=576  19.7 ms unfused (fusing the final norm: 24.1 ms)
    };
    return g[s];
}
inline const char* shape_name(Shape s) {
    static const char* n[NUM_SHAPES] = {"qkv", "o_proj", "gate_up", "down", "lm_head"};
    return n[s];
}
// Plain/residual GEMVs are classified by shape.
inline Shape classify(int N, int K) {
    if (N >= 16384) return LMHEAD;
    if (K > N) return DOWN;
    return OPROJ;
}

struct Kernels { cl_program prog = nullptr; cl_kernel plain = nullptr, addres = nullptr,
                 swiglu = nullptr, qkv = nullptr; };

class Ctx {
  public:
    static Ctx& get(cl_command_queue q) { static Ctx c(q); return c; }

    bool enabled() const { return enabled_; }
    // Fold the decoder layer's input RMSNorm into the QKV GEMV.
    // NNOPT_TEAM_FUSE_NORM=0 disables.
    bool fuse_norm() const {
        const char* e = std::getenv("NNOPT_TEAM_FUSE_NORM");
        return enabled_ && !(e && e[0] == '0');
    }

    // gamma != nullptr: x is the raw residual and RMSNorm(gamma, eps) is fused in.
    bool gemv(cl_command_queue q, cl_mem x, cl_mem W, cl_mem out, int N, int K,
              const Geo* go = nullptr, cl_mem gamma = nullptr, float eps = 0.0f) {
        const Geo& g = go ? *go : geo(classify(N, K));
        const Kernels* k = kernels(g, gamma != nullptr);
        if (!k) return false;
        cl_kernel kk = k->plain;
        clSetKernelArg(kk, 0, sizeof(cl_mem), &x);
        clSetKernelArg(kk, 1, sizeof(cl_mem), &W);
        clSetKernelArg(kk, 2, sizeof(cl_mem), &out);
        clSetKernelArg(kk, 3, sizeof(int), &N);
        clSetKernelArg(kk, 4, sizeof(int), &K);
        if (gamma) {
            clSetKernelArg(kk, 5, sizeof(cl_mem), &gamma);
            clSetKernelArg(kk, 6, sizeof(float), &eps);
        }
        return launch(q, kk, g, N);
    }

    // out = res + W x
    bool gemv_addres(cl_command_queue q, cl_mem x, cl_mem W, cl_mem res, cl_mem out,
                     int N, int K, const Geo* go = nullptr) {
        const Geo& g = go ? *go : geo(classify(N, K));
        const Kernels* k = kernels(g);
        if (!k) return false;
        cl_kernel kk = k->addres;
        clSetKernelArg(kk, 0, sizeof(cl_mem), &x);
        clSetKernelArg(kk, 1, sizeof(cl_mem), &W);
        clSetKernelArg(kk, 2, sizeof(cl_mem), &res);
        clSetKernelArg(kk, 3, sizeof(cl_mem), &out);
        clSetKernelArg(kk, 4, sizeof(int), &N);
        clSetKernelArg(kk, 5, sizeof(int), &K);
        return launch(q, kk, g, N);
    }

    bool swiglu(cl_command_queue q, cl_mem x, cl_mem Wg, cl_mem Wu, cl_mem out, int N, int K,
                const Geo* go = nullptr, cl_mem gamma = nullptr, float eps = 0.0f) {
        const Geo& g = go ? *go : geo(GATEUP);
        const Kernels* k = kernels(g, gamma != nullptr);
        if (!k) return false;
        cl_kernel kk = k->swiglu;
        clSetKernelArg(kk, 0, sizeof(cl_mem), &x);
        clSetKernelArg(kk, 1, sizeof(cl_mem), &Wg);
        clSetKernelArg(kk, 2, sizeof(cl_mem), &Wu);
        clSetKernelArg(kk, 3, sizeof(cl_mem), &out);
        clSetKernelArg(kk, 4, sizeof(int), &N);
        clSetKernelArg(kk, 5, sizeof(int), &K);
        if (gamma) {
            clSetKernelArg(kk, 6, sizeof(cl_mem), &gamma);
            clSetKernelArg(kk, 7, sizeof(float), &eps);
        }
        return launch(q, kk, g, N);
    }

    // kernels/decode_team.cl (attn_rope_gqa, embed_row), built once.
    cl_kernel attn_rope_kernel() { decode_prog(); return attn_rope_; }
    cl_kernel embed_kernel()     { decode_prog(); return embed_; }
    cl_kernel rms_row_kernel()   { decode_prog(); return rms_row_; }

    bool launch_raw(cl_command_queue q, cl_kernel k, size_t gws, size_t lws) {
        cl_int e = clEnqueueNDRangeKernel(q, k, 1, nullptr, &gws, &lws, 0, nullptr, nullptr);
        if (e != CL_SUCCESS) std::fprintf(stderr, "NNOPT_TEAM: enqueue failed (%d)\n", e);
        return e == CL_SUCCESS;
    }

    // Compile every program decode will use now, not on the first token.
    void prebuild() {
        if (!enabled_) return;
        decode_prog();
        for (int s = 0; s < NUM_SHAPES; ++s) kernels(geo((Shape)s), false);
        if (fuse_norm()) {
            kernels(geo(QKV), true);
            kernels(geo(GATEUP), true);
            
        }
    }

    // gamma != nullptr: x is the raw residual and RMSNorm(gamma, eps) is fused in.
    bool qkv(cl_command_queue q, cl_mem x, cl_mem Wq, cl_mem Wk, cl_mem Wv,
             cl_mem qo, cl_mem ko, cl_mem vo, int QN, int KVN, int K, const Geo* go = nullptr,
             cl_mem gamma = nullptr, float eps = 0.0f) {
        const Geo& g = go ? *go : geo(QKV);
        const Kernels* k = kernels(g, gamma != nullptr);
        if (!k) return false;
        cl_kernel kk = k->qkv;
        clSetKernelArg(kk, 0, sizeof(cl_mem), &x);
        clSetKernelArg(kk, 1, sizeof(cl_mem), &Wq);
        clSetKernelArg(kk, 2, sizeof(cl_mem), &Wk);
        clSetKernelArg(kk, 3, sizeof(cl_mem), &Wv);
        clSetKernelArg(kk, 4, sizeof(cl_mem), &qo);
        clSetKernelArg(kk, 5, sizeof(cl_mem), &ko);
        clSetKernelArg(kk, 6, sizeof(cl_mem), &vo);
        clSetKernelArg(kk, 7, sizeof(int), &QN);
        clSetKernelArg(kk, 8, sizeof(int), &KVN);
        clSetKernelArg(kk, 9, sizeof(int), &K);
        if (gamma) {
            clSetKernelArg(kk, 10, sizeof(cl_mem), &gamma);
            clSetKernelArg(kk, 11, sizeof(float), &eps);
        }
        return launch(q, kk, g, QN + 2 * KVN);
    }

    // Once per process, before the first decode GEMV, when requested.
    void maybe_sweep(cl_command_queue q, int H, int I, int QN, int KVN, int V) {
        if (swept_) return;
        swept_ = true;
#ifdef NNOPT_TEAM_SWEEP
        const bool want = true;
#else
        const char* e = std::getenv("NNOPT_TEAM_SWEEP");
        const bool want = e && e[0] == '1';
#endif
        if (want) sweep(q, H, I, QN, KVN, V);
    }

  private:
    explicit Ctx(cl_command_queue q) {
        clGetCommandQueueInfo(q, CL_QUEUE_CONTEXT, sizeof(ctx_), &ctx_, nullptr);
        clGetCommandQueueInfo(q, CL_QUEUE_DEVICE, sizeof(dev_), &dev_, nullptr);
#ifdef NNOPT_USE_FP16
        const char* e = std::getenv("NNOPT_TEAM_GEMV");
        if (e && e[0] == '1') enabled_ = true;
        else if (e && e[0] == '0') enabled_ = false;
        else enabled_ = !nnopt_textures_allowed(dev_);
#endif
        if (const char* m = std::getenv("NNOPT_TEAM_MODE"))
            for (int s = 0; s < NUM_SHAPES; ++s) geo((Shape)s).mode = std::atoi(m);
        if (const char* xl = std::getenv("NNOPT_TEAM_XL"))     // test hook: stage x in __local
            for (int s = 0; s < NUM_SHAPES; ++s) geo((Shape)s).xl = std::atoi(xl);
        if (enabled_) {
            std::ifstream f("kernels/gemv_team.cl");
            std::stringstream ss; ss << f.rdbuf();
            src_ = ss.str();
            if (src_.empty()) {
                std::fprintf(stderr, "NNOPT_TEAM: kernels/gemv_team.cl missing — team GEMVs OFF\n");
                enabled_ = false;
            } else {
                std::fprintf(stderr, "NNOPT_TEAM: fp16 team GEMVs ON (kernels/gemv_team.cl)\n");
            }
        }
    }

    const Kernels* kernels(const Geo& g, bool fuse_norm = false) {
        auto key = std::make_tuple(g.wg, g.lpr, g.rpl, g.mode + (fuse_norm ? 100 : 0) + (g.xl ? 1000 : 0) + (g.unr == 2 ? 10000 : 0));
        auto it = cache_.find(key);
        if (it != cache_.end()) return it->second.prog ? &it->second : nullptr;
        Kernels& k = cache_[key];
        char opts[128];
        std::snprintf(opts, sizeof(opts), "-DWG=%d -DLPR=%d -DRPL=%d -DLOADMODE=%d%s",
                      g.wg, g.lpr, g.rpl, g.mode,
                      std::string(fuse_norm ? " -DFUSE_NORM=1 -DNORM_PRECISE_RSQRT=1" : "")
                          .append(g.xl ? " -DXLOCAL=1536" : "").append(g.unr == 2 ? " -DUNR=2" : "").c_str());
        k.prog = nnopt_build_program_cached(ctx_, dev_, src_, opts);
        if (!k.prog) return nullptr;
        cl_int e;
        k.plain  = clCreateKernel(k.prog, "gemv_t", &e);
        k.addres = clCreateKernel(k.prog, "gemv_addres_t", &e);
        k.swiglu = clCreateKernel(k.prog, "gemv_swiglu_t", &e);
        k.qkv    = clCreateKernel(k.prog, "gemv_qkv_t", &e);
        if (!k.plain || !k.addres || !k.swiglu || !k.qkv) { k.prog = nullptr; return nullptr; }
        for (cl_kernel kk : {k.plain, k.addres, k.swiglu, k.qkv}) {
            size_t m = 0;
            clGetKernelWorkGroupInfo(kk, dev_, CL_KERNEL_WORK_GROUP_SIZE, sizeof(m), &m, nullptr);
            if (m < (size_t)g.wg) { k.prog = nullptr; return nullptr; }
        }
        return &k;
    }

    bool launch(cl_command_queue q, cl_kernel k, const Geo& g, int N) {
        const int rows_per_wg = (g.wg / g.lpr) * g.rpl;
        size_t gw = (size_t)((N + rows_per_wg - 1) / rows_per_wg) * g.wg, lw = (size_t)g.wg;
        cl_int e = clEnqueueNDRangeKernel(q, k, 1, nullptr, &gw, &lw, 0, nullptr, nullptr);
        if (e != CL_SUCCESS) {
            std::fprintf(stderr, "NNOPT_TEAM: enqueue failed (%d) wg=%d lpr=%d rpl=%d N=%d\n",
                         e, g.wg, g.lpr, g.rpl, N);
            return false;
        }
        return true;
    }

    void sweep(cl_command_queue q, int H, int I, int QN, int KVN, int V) {
        std::vector<Geo> cands;
        // GE8320 sweep 2026-10-03: mode 0 2-3x slower, XLOCAL staging 1.3-2x
        // slower; neither is re-run. UNR=2 (two chunks per iteration) is new.
        for (int unr = 1; unr <= 2; ++unr)
            for (Geo g : {Geo{64, 8, 2, 1}, Geo{64, 8, 4, 1}, Geo{64, 16, 2, 1}, Geo{64, 16, 4, 1},
                          Geo{64, 32, 2, 1}, Geo{64, 4, 4, 1}, Geo{64, 8, 8, 1}, Geo{128, 16, 2, 1}}) {
                g.unr = unr;
                cands.push_back(g);
            }
#ifdef NNOPT_TEAM_ALLOW_HALF
        const int adopt_max_mode = 2;
#else
        const int adopt_max_mode = 1;   // mode 2 changes numerics: report, don't adopt
#endif
        cl_int e;
        auto mk = [&](size_t halves) {
            std::vector<cl_half> host(halves);
            for (size_t i = 0; i < halves; ++i) host[i] = (cl_half)(0x2000 + (i * 2654435761u >> 20) % 0x1000);
            return clCreateBuffer(ctx_, CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR, halves * 2, host.data(), &e);
        };
        const int KMAX = H > I ? H : I;
        cl_mem x = mk(KMAX), out = mk(V > I ? V : I), r1 = mk(H), r2 = mk(H);
        cl_mem Wbig = mk((size_t)V * H), Wa = mk((size_t)I * H), Wb = mk((size_t)I * H);
        cl_mem Wq = mk((size_t)QN * H), Wk = mk((size_t)KVN * H), Wv = mk((size_t)KVN * H);
        cl_mem ko = mk(KVN), vo = mk(KVN);
        cl_mem gam = fuse_norm() ? mk(KMAX) : nullptr;
        const float ep = 1e-5f;
        const double mb[NUM_SHAPES] = {
            (QN + 2.0 * KVN) * H * 2 / 1e6, (double)H * QN * 2 / 1e6, 2.0 * I * H * 2 / 1e6,
            (double)H * I * 2 / 1e6, (double)V * H * 2 / 1e6};
        for (int s = 0; s < NUM_SHAPES; ++s) {
            double best = 1e30; Geo bg = geo((Shape)s);
            for (const Geo& g : cands) {
                auto run = [&]() -> bool {
                    switch ((Shape)s) {
                        case QKV:    return qkv(q, x, Wq, Wk, Wv, out, ko, vo, QN, KVN, H, &g, gam, ep);
                        case OPROJ:  return gemv(q, x, Wa, r1, H, QN, &g);
                        case GATEUP: return swiglu(q, x, Wa, Wb, out, I, H, &g, gam, ep);
                        case DOWN:   return gemv_addres(q, x, Wa, r1, r2, H, I, &g);
                        default:     return gemv(q, x, Wbig, out, V, H, &g);
                    }
                };
                if (!run()) continue;
                clFinish(q);
                const int reps = s == LMHEAD ? 6 : 20;
                auto t0 = std::chrono::steady_clock::now();
                for (int i = 0; i < reps; ++i) run();
                clFinish(q);
                const double ms = std::chrono::duration<double, std::milli>(
                                      std::chrono::steady_clock::now() - t0).count() / reps;
                std::fprintf(stderr, "NNOPT_TEAM: %-7s m%d%s u%d %3d/%2d/%d %7.3f ms %5.2f GB/s\n",
                             shape_name((Shape)s), g.mode, g.xl ? "x" : " ", g.unr, g.wg, g.lpr, g.rpl,
                             ms, mb[s] / ms);
                if (ms < best && g.mode <= adopt_max_mode) { best = ms; bg = g; }
            }
            geo((Shape)s) = bg;
            std::fprintf(stderr, "NNOPT_TEAM_BEST: %-7s m%d%s u%d %d/%d/%d %.3f ms %.2f GB/s\n",
                         shape_name((Shape)s), bg.mode, bg.xl ? "x" : " ", bg.unr, bg.wg, bg.lpr, bg.rpl,
                         best, mb[s] / best);
        }
        for (cl_mem m : {x, out, r1, r2, Wbig, Wa, Wb, Wq, Wk, Wv, ko, vo, gam}) if (m) clReleaseMemObject(m);
        std::fflush(stderr);
    }

    cl_program decode_prog() {
        if (!decode_tried_) {
            decode_tried_ = true;
            std::ifstream f("kernels/decode_team.cl");
            std::stringstream ss; ss << f.rdbuf();
            if (!ss.str().empty()) decode_prog_ = nnopt_build_program_cached(ctx_, dev_, ss.str(), "");
            if (decode_prog_) {
                attn_rope_ = clCreateKernel(decode_prog_, "attn_rope_gqa_tree", nullptr);
                embed_ = clCreateKernel(decode_prog_, "embed_row", nullptr);
                rms_row_ = clCreateKernel(decode_prog_, "rms_row", nullptr);
            }
        }
        return decode_prog_;
    }

    bool decode_tried_ = false;
    cl_program decode_prog_ = nullptr;
    cl_kernel attn_rope_ = nullptr, embed_ = nullptr, rms_row_ = nullptr;
    cl_context ctx_ = nullptr;
    cl_device_id dev_ = nullptr;
    bool enabled_ = false, swept_ = false;
    std::string src_;
    std::map<std::tuple<int, int, int, int>, Kernels> cache_;
};

}  // namespace team_gemv
