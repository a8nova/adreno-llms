// gemm_bench — standalone micro-harness for the round-4 bespoke GEMM campaign.
//
// The e2e A/B loop costs ~90 s per data point; this binary measures a kernel
// candidate in seconds at the REAL DiT shapes, against the CLBlast TransB
// baseline (the production path), with cosine validation so a fast-but-wrong
// kernel can't sneak through.
//
// Usage (on device, from the deploy dir so kernels/ resolves):
//   ./gemm_bench [M N K]     # default: the four hot DiT shapes
//
// Output: one line per (shape, variant): ms/iter, GFLOPS, cos vs CLBlast.

#include "opencl_context.h"
#include "utils.h"
#include "profiler.h"
#include "debug_utils.h"

#include <clblast.h>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <random>
#include <string>
#include <vector>

namespace {

std::vector<uint16_t> random_f16(size_t n, uint32_t seed) {
    std::mt19937 rng(seed);
    std::normal_distribution<float> d(0.0f, 0.5f);
    std::vector<uint16_t> v(n);
    for (auto& x : v) x = nnopt_f32_to_f16(d(rng));
    return v;
}

std::vector<float> download_f16(cl_command_queue q, cl_mem buf, size_t n) {
    std::vector<uint16_t> tmp(n);
    clEnqueueReadBuffer(q, buf, CL_TRUE, 0, n * 2, tmp.data(), 0, nullptr, nullptr);
    std::vector<float> out(n);
    for (size_t i = 0; i < n; i++) out[i] = nnopt_f16_to_f32(tmp[i]);
    return out;
}

double cosine(const std::vector<float>& a, const std::vector<float>& b) {
    double dot = 0, na = 0, nb = 0;
    for (size_t i = 0; i < a.size(); i++) { dot += (double)a[i]*b[i]; na += (double)a[i]*a[i]; nb += (double)b[i]*b[i]; }
    return dot / (std::sqrt(na) * std::sqrt(nb) + 1e-12);
}

struct Shape { int M, N, K; const char* tag; };

} // namespace

int main(int argc, char** argv) {
    OpenCLContext cl;
    if (!cl.initialize()) { fprintf(stderr, "CL init failed\n"); return 1; }
    cl_context ctx = cl.context();
    cl_command_queue q = cl.queue();

    std::vector<Shape> shapes = {
        {257, 8192, 1024, "ffn1"},
        {257, 1024, 4096, "ffn2"},
        {257, 2048, 1024, "kv"},
        {257, 1024, 1024, "q/out"},
    };
    if (argc == 4) shapes = {{atoi(argv[1]), atoi(argv[2]), atoi(argv[3]), "cli"}};

    cl_program prog = cl.build_program_from_file("kernels/gemm_tex.cl");
    if (!prog) { fprintf(stderr, "gemm_tex.cl build failed\n"); return 1; }

    const int WARM = 2, ITERS = 6;
    for (const auto& s : shapes) {
        const size_t xs = (size_t)s.M * s.K, ws = (size_t)s.N * s.K, os = (size_t)s.M * s.N;
        auto xh = random_f16(xs, 1), wh = random_f16(ws, 2);
        cl_int err;
        cl_mem X = clCreateBuffer(ctx, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, xs*2, xh.data(), &err);
        cl_mem W = clCreateBuffer(ctx, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, ws*2, wh.data(), &err);
        cl_mem O = clCreateBuffer(ctx, CL_MEM_READ_WRITE, os*2, nullptr, &err);
        if (err != CL_SUCCESS) { fprintf(stderr, "alloc failed\n"); return 1; }
        const double gflop = 2.0 * s.M * s.N * s.K / 1e9;

        // ── baseline: CLBlast TransB (the production pytorch_linear path) ──
        std::vector<float> ref;
        {
            for (int i = 0; i < WARM; i++) pytorch_linear(q, s.M, s.N, s.K, X, W, O);
            clFinish(q);
            auto t0 = std::chrono::steady_clock::now();
            for (int i = 0; i < ITERS; i++) pytorch_linear(q, s.M, s.N, s.K, X, W, O);
            clFinish(q);
            double ms = std::chrono::duration<double,std::milli>(std::chrono::steady_clock::now()-t0).count()/ITERS;
            ref = download_f16(q, O, os);
            printf("%-6s M=%d N=%d K=%d  clblast_transb   %8.2f ms  %6.1f GFLOPS\n",
                   s.tag, s.M, s.N, s.K, ms, gflop/ms*1000.0);
        }

        // ── W as RGBA fp16 image2d: width K/4 texels, height N ──
        cl_image_format fmt{CL_RGBA, CL_HALF_FLOAT};
        cl_image_desc desc{};
        desc.image_type = CL_MEM_OBJECT_IMAGE2D;
        desc.image_width = (size_t)s.K / 4;
        desc.image_height = (size_t)s.N;
        cl_mem Wimg = clCreateImage(ctx, CL_MEM_READ_ONLY, &fmt, &desc, nullptr, &err);
        if (err == CL_SUCCESS) {
            size_t origin[3] = {0,0,0}, region[3] = {desc.image_width, desc.image_height, 1};
            err = clEnqueueCopyBufferToImage(q, W, Wimg, 0, origin, region, 0, nullptr, nullptr);
        }
        // Packed-buffer W4[k4][n][4] — the no-texture control for the v2 tiling.
        std::vector<uint16_t> w4h(ws);
        for (int n = 0; n < s.N; n++)
            for (int k4 = 0; k4 < s.K / 4; k4++)
                for (int j = 0; j < 4; j++)
                    w4h[((size_t)k4 * s.N + n) * 4 + j] = wh[(size_t)n * s.K + k4 * 4 + j];
        cl_mem W4 = clCreateBuffer(ctx, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, ws*2, w4h.data(), &err);

        if (err != CL_SUCCESS) { fprintf(stderr, "  image setup failed (%d) — skipping tex variants\n", err); }
        else {
            for (const char* kname : {"tex_local_m16", "p4_local_m16", "tex_local_16x16"}) {
                cl_kernel k = clCreateKernel(prog, kname, &err);
                if (err != CL_SUCCESS) { fprintf(stderr, "  %s missing\n", kname); continue; }
                const bool packed = std::string(kname).rfind("p4_", 0) == 0;
                clSetKernelArg(k, 0, sizeof(cl_mem), &X);
                clSetKernelArg(k, 1, sizeof(cl_mem), packed ? &W4 : &Wimg);
                clSetKernelArg(k, 2, sizeof(cl_mem), &O);
                clSetKernelArg(k, 3, sizeof(int), &s.M);
                clSetKernelArg(k, 4, sizeof(int), &s.N);
                clSetKernelArg(k, 5, sizeof(int), &s.K);
                const bool nx16 = std::string(kname) == "tex_local_16x16";
                size_t lws[2] = {64, 1};
                size_t gws[2] = {nx16 ? ((size_t)s.N + 15)/16*64 : ((size_t)s.N + 63)/64*64,
                                 ((size_t)s.M + 15)/16};
                auto run = [&]() { return clEnqueueNDRangeKernel(q, k, 2, nullptr, gws, lws, 0, nullptr, nullptr); };
                bool ok = true;
                for (int i = 0; i < WARM; i++) if (run() != CL_SUCCESS) { ok = false; break; }
                clFinish(q);
                if (!ok) { fprintf(stderr, "  %s dispatch failed\n", kname); clReleaseKernel(k); continue; }
                auto t0 = std::chrono::steady_clock::now();
                for (int i = 0; i < ITERS; i++) run();
                clFinish(q);
                double ms = std::chrono::duration<double,std::milli>(std::chrono::steady_clock::now()-t0).count()/ITERS;
                auto got = download_f16(q, O, os);
                printf("%-6s M=%d N=%d K=%d  %-16s %8.2f ms  %6.1f GFLOPS  cos=%.6f\n",
                       s.tag, s.M, s.N, s.K, kname, ms, gflop/ms*1000.0, cosine(ref, got));
                clReleaseKernel(k);
            }
            clReleaseMemObject(Wimg);
        }
        if (W4) clReleaseMemObject(W4);
        clReleaseMemObject(X); clReleaseMemObject(W); clReleaseMemObject(O);
    }
    return 0;
}
