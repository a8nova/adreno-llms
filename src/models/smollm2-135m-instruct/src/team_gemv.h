// fp16 decode GEMVs for GPUs without a usable texture path (kernels/gemv_team.cl).
//
// On by default exactly where texture_policy.h denies textures (PowerVR Rogue),
// i.e. where the decode would otherwise fall to the block_fused.cl buffer
// kernels, which stream ~0.3 GB/s there. Elsewhere nothing changes.
//
//   NNOPT_TEAM_GEMV=1   force on (exercise the path on Adreno)
//   NNOPT_TEAM_GEMV=0   force off
//   NNOPT_TEAM_SWEEP=1  (or build with -DNNOPT_TEAM_SWEEP) time every candidate
//                       geometry per decode shape after load, print the table,
//                       and use the fastest for this run.
#pragma once

#include <CL/cl.h>

#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <map>
#include <string>
#include <tuple>
#include <vector>

#include "opencl_context.h"
#include "prof.h"
#include "texture_policy.h"

namespace team_gemv {

enum Shape { QKV = 0, OPROJ, GATEUP, DOWN, LMHEAD, NUM_SHAPES };
struct Geo { int wg, lpr, rpl, mode, xl = 0, unr = 1; };   // LOADMODE, x in __local, chunk unroll

// Default geometry per decode shape. Measured on PowerVR GE8320 (Vivo Y21) with
// the NNOPT_TEAM_SWEEP table; re-run the sweep before trusting it elsewhere.
inline Geo& geo(Shape s) {
    static Geo g[NUM_SHAPES] = {
        {64,  8, 4, 1},   // QKV     N=960   K=576    0.68 ms (norm fused)  GE8320 sweep 2026-10-03
        {64, 16, 2, 1},   // OPROJ   N=576   K=576    0.35 ms
        {64, 16, 4, 1},   // GATEUP  N=1536  K=576    1.51 ms (norm fused, x2 matrices)
        {64, 16, 4, 1},   // DOWN    N=576   K=1536   0.75 ms
        {64, 16, 2, 1},   // LMHEAD  N=49152 K=576  19.7 ms unfused (fusing the final norm: 24.1 ms)
    };
    return g[s];
}
inline const char* shape_name(Shape s) {
    static const char* n[NUM_SHAPES] = {"qkv", "o_proj", "gate_up", "down", "lm_head"};
    return n[s];
}

struct Kernels { cl_program prog = nullptr; cl_kernel plain = nullptr, res = nullptr,
                 swiglu = nullptr, qkv = nullptr; size_t max_wg = 0; };

class Ctx {
  public:
    static Ctx& get(OpenCLContext& cl) { static Ctx c(cl); return c; }

    bool enabled() const { return enabled_; }
    // Fold each decoder layer's input/post-attention RMSNorm into the QKV and
    // gate_up GEMVs (one launch fewer per norm). NNOPT_TEAM_FUSE_NORM=0 disables.
    bool fuse_norm() const {
        const char* e = std::getenv("NNOPT_TEAM_FUSE_NORM");
        return enabled_ && !(e && e[0] == '0');
    }

    // Kernels for one geometry, built on first use (nullptr if the build failed).
    // fuse_norm: build with -DFUSE_NORM (qkv/swiglu apply the RMSNorm themselves).
    const Kernels* kernels(const Geo& g, bool fuse_norm = false) {
        auto key = std::make_tuple(g.wg, g.lpr, g.rpl, g.mode + (fuse_norm ? 100 : 0) + (g.xl ? 1000 : 0) + (g.unr == 2 ? 10000 : 0));
        auto it = cache_.find(key);
        if (it != cache_.end()) return it->second.prog ? &it->second : nullptr;
        Kernels& k = cache_[key];
        char opts[128];
        std::snprintf(opts, sizeof(opts), "-DWG=%d -DLPR=%d -DRPL=%d -DLOADMODE=%d%s",
                      g.wg, g.lpr, g.rpl, g.mode,
                      std::string(fuse_norm ? " -DFUSE_NORM=1" : "").append(g.xl ? " -DXLOCAL=1536" : "").append(g.unr == 2 ? " -DUNR=2" : "").c_str());
        k.prog = cl_.build_program_from_file("kernels/gemv_team.cl", opts);
        if (!k.prog) return nullptr;
        cl_int e;
        k.plain  = clCreateKernel(k.prog, "gemv_t", &e);
        k.res    = clCreateKernel(k.prog, "gemv_res_t", &e);
        k.swiglu = clCreateKernel(k.prog, "gemv_swiglu_t", &e);
        k.qkv    = clCreateKernel(k.prog, "gemv_qkv_t", &e);
        if (!k.plain || !k.res || !k.swiglu || !k.qkv) { k.prog = nullptr; return nullptr; }
        k.max_wg = SIZE_MAX;
        for (cl_kernel kk : {k.plain, k.res, k.swiglu, k.qkv}) {
            size_t m = 0;
            clGetKernelWorkGroupInfo(kk, cl_.device(), CL_KERNEL_WORK_GROUP_SIZE, sizeof(m), &m, nullptr);
            if (m < k.max_wg) k.max_wg = m;
        }
        if (k.max_wg < (size_t)g.wg) { k.prog = nullptr; return nullptr; }
        return &k;
    }

    // kernels/decode_team.cl, built once (nullptr if unavailable).
    cl_program decode_prog() {
        if (!decode_tried_) {
            decode_tried_ = true;
            decode_prog_ = cl_.build_program_from_file("kernels/decode_team.cl", "");
        }
        return decode_prog_;
    }
    // attn_tree: use the previous tree-reduction kernel (A/B; NNOPT_ATTN_TREE=1).
    bool attn_tree = true;   // tree kernel measured faster on Adreno; GE8320 A/B pending
    cl_kernel attn_rope_kernel() {
        if (!attn_rope_ && decode_prog()) {
            attn_rope_ = clCreateKernel(decode_prog_, "attn_rope_gqa", nullptr);
            attn_tree_ = clCreateKernel(decode_prog_, "attn_rope_gqa_tree", nullptr);
        }
        return attn_tree && attn_tree_ ? attn_tree_ : attn_rope_;
    }
    cl_kernel split_kernel() {
        if (!split_ && decode_prog()) split_ = clCreateKernel(decode_prog_, "attn_split", nullptr);
        return split_;
    }
    cl_kernel merge_kernel() {
        if (!merge_ && decode_prog()) merge_ = clCreateKernel(decode_prog_, "attn_merge", nullptr);
        return merge_;
    }
    // Context splits for decode attention (0 = single-pass attn_rope_gqa).
    // NNOPT_ATTN_SPLIT=S overrides; used when seq_k >= split_min_ctx.
    int attn_splits = 0;
    int split_min_ctx = 128;
    cl_mem split_scratch(size_t bytes) {
        if (bytes > scratch_bytes_) {
            if (scratch_) clReleaseMemObject(scratch_);
            cl_int e;
            scratch_ = clCreateBuffer(cl_.context(), CL_MEM_READ_WRITE, bytes, nullptr, &e);
            scratch_bytes_ = scratch_ ? bytes : 0;
        }
        return scratch_;
    }
    cl_kernel embed_kernel() {
        if (!embed_ && decode_prog()) embed_ = clCreateKernel(decode_prog_, "embed_row", nullptr);
        return embed_;
    }

    // Build every program decode will use, so the compile happens at load.
    void prebuild() {
        if (!enabled_) return;
        attn_rope_kernel();
        embed_kernel();
        for (int s = 0; s < NUM_SHAPES; ++s) {
            kernels(geo((Shape)s), false);
            if (fuse_norm() && (s == QKV || s == GATEUP)) kernels(geo((Shape)s), true);
        }
    }

    static size_t gws(const Geo& g, int N) {
        const int rows_per_wg = (g.wg / g.lpr) * g.rpl;
        return (size_t)((N + rows_per_wg - 1) / rows_per_wg) * g.wg;
    }

    // out[n] = W x            (res=false)   |   out[n] += W x   (res=true)
    // gamma != nullptr: x is the raw residual and RMSNorm(gamma, eps) is fused in.
    bool gemv(cl_command_queue q, Shape s, cl_mem x, cl_mem W, cl_mem out, int N, int K,
              bool res, const Geo* g_override = nullptr, cl_mem gamma = nullptr, float eps = 0.0f) {
        const Geo& g = g_override ? *g_override : geo(s);
        const Kernels* k = kernels(g, gamma != nullptr);
        if (!k) return false;
        cl_kernel kk = res ? k->res : k->plain;
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

    // gamma != nullptr: x is the raw residual and RMSNorm(gamma, eps) is fused in.
    bool swiglu(cl_command_queue q, cl_mem x, cl_mem Wg, cl_mem Wu, cl_mem out, int N, int K,
                const Geo* g_override = nullptr, cl_mem gamma = nullptr, float eps = 0.0f) {
        const Geo& g = g_override ? *g_override : geo(GATEUP);
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

    bool qkv(cl_command_queue q, cl_mem x, cl_mem Wq, cl_mem Wk, cl_mem Wv,
             cl_mem qo, cl_mem ko, cl_mem vo, int QN, int KVN, int K,
             const Geo* g_override = nullptr, cl_mem gamma = nullptr, float eps = 0.0f) {
        const Geo& g = g_override ? *g_override : geo(QKV);
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

    // Time every candidate geometry on each decode shape (synthetic buffers of
    // the real sizes), print the table, and adopt the fastest per shape.
    // H/I/QN/KVN/V are the model dims.
    void sweep(cl_command_queue q, int H, int I, int QN, int KVN, int V) {
        std::vector<Geo> cands;
        // Mode 0 (vload_half8) measured 2-3x slower on the GE8320; not re-run.
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
        cl_context ctx = cl_.context();
        cl_int e;
        auto mk = [&](size_t halves) {
            std::vector<cl_half> host(halves);
            for (size_t i = 0; i < halves; ++i) host[i] = (cl_half)(0x2000 + (i * 2654435761u >> 20) % 0x1000);
            return clCreateBuffer(ctx, CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR,
                                  halves * 2, host.data(), &e);
        };
        const int KMAX = H > I ? H : I;
        cl_mem x = mk(KMAX), out = mk(V > I ? V : I), r1 = mk(H);
        cl_mem Wbig = mk((size_t)V * H), Wa = mk((size_t)I * H), Wb = mk((size_t)I * H);
        cl_mem Wq = mk((size_t)QN * H), Wk = mk((size_t)KVN * H), Wv = mk((size_t)KVN * H);
        cl_mem ko = mk(KVN), vo = mk(KVN);
        // Time what decode runs: QKV / gate_up / lm_head with the RMSNorm fused.
        cl_mem gam = fuse_norm() ? mk(KMAX) : nullptr;
        const float ep = 1e-5f;
        {   // bare launch cost: one 8-wide row
            const Geo g{64, 1, 1, 1};
            if (gemv(q, LMHEAD, x, Wbig, out, 1, 8, false, &g)) {
                clFinish(q);
                auto t0 = std::chrono::steady_clock::now();
                for (int i = 0; i < 50; ++i) gemv(q, LMHEAD, x, Wbig, out, 1, 8, false, &g);
                clFinish(q);
                std::fprintf(stderr, "NNOPT_TEAM: launch overhead %.3f ms/launch\n",
                             std::chrono::duration<double, std::milli>(
                                 std::chrono::steady_clock::now() - t0).count() / 50);
            }
        }
        const double mb[NUM_SHAPES] = {
            (QN + 2.0 * KVN) * H * 2 / 1e6, (double)H * QN * 2 / 1e6, 2.0 * I * H * 2 / 1e6,
            (double)H * I * 2 / 1e6, (double)V * H * 2 / 1e6};
        for (int s = 0; s < NUM_SHAPES; ++s) {
            double best = 1e30; Geo bg = geo((Shape)s);
            for (const Geo& g : cands) {
                auto run = [&]() -> bool {
                    switch ((Shape)s) {
                        case QKV:    return qkv(q, x, Wq, Wk, Wv, out, ko, vo, QN, KVN, H, &g, gam, ep);
                        case OPROJ:  return gemv(q, OPROJ, x, Wa, r1, H, QN, true, &g);
                        case GATEUP: return swiglu(q, x, Wa, Wb, out, I, H, &g, gam, ep);
                        case DOWN:   return gemv(q, DOWN, x, Wa, r1, H, I, true, &g);
                        default:     return gemv(q, LMHEAD, x, Wbig, out, V, H, false, &g);
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
        for (cl_mem m : {x, out, r1, Wbig, Wa, Wb, Wq, Wk, Wv, ko, vo, gam}) if (m) clReleaseMemObject(m);
        std::fflush(stderr);
    }

    bool sweep_requested() const {
#ifdef NNOPT_TEAM_SWEEP
        return true;
#else
        const char* e = std::getenv("NNOPT_TEAM_SWEEP");
        return e && e[0] == '1';
#endif
    }

  private:
    explicit Ctx(OpenCLContext& cl) : cl_(cl) {
#ifdef NNOPT_USE_FP16
        const char* e = std::getenv("NNOPT_TEAM_GEMV");
        if (e && e[0] == '1') enabled_ = true;
        else if (e && e[0] == '0') enabled_ = false;
        else enabled_ = !nnopt_textures_allowed(cl.device());
#endif
        if (const char* m = std::getenv("NNOPT_TEAM_MODE"))   // test hook: one LOADMODE everywhere
            for (int s = 0; s < NUM_SHAPES; ++s) geo((Shape)s).mode = std::atoi(m);
        if (const char* xl = std::getenv("NNOPT_TEAM_XL"))     // test hook: stage x in __local
            for (int s = 0; s < NUM_SHAPES; ++s) geo((Shape)s).xl = std::atoi(xl);
        if (const char* sp = std::getenv("NNOPT_ATTN_SPLIT")) attn_splits = std::atoi(sp);
        if (const char* at = std::getenv("NNOPT_ATTN_TREE")) attn_tree = at[0] != '0';
        if (enabled_) std::fprintf(stderr, "NNOPT_TEAM: fp16 team GEMVs ON (kernels/gemv_team.cl)\n");
    }

    bool launch(cl_command_queue q, cl_kernel k, const Geo& g, int N) {
        size_t gw = gws(g, N), lw = (size_t)g.wg;
        cl_int e = nnopt_prof::enqueue(q, k, 1, nullptr, &gw, &lw, 0, nullptr, nullptr);
        if (e != CL_SUCCESS) {
            std::fprintf(stderr, "NNOPT_TEAM: enqueue failed (%d) wg=%d lpr=%d rpl=%d N=%d\n",
                         e, g.wg, g.lpr, g.rpl, N);
            return false;
        }
        return true;
    }

    OpenCLContext& cl_;
    bool enabled_ = false;
    bool decode_tried_ = false;
    cl_program decode_prog_ = nullptr;
    cl_kernel attn_rope_ = nullptr, attn_tree_ = nullptr, embed_ = nullptr, split_ = nullptr, merge_ = nullptr;
    cl_mem scratch_ = nullptr;
    size_t scratch_bytes_ = 0;
    std::map<std::tuple<int, int, int, int>, Kernels> cache_;
};

}  // namespace team_gemv
