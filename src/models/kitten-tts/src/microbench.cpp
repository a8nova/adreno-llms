// Device peak characterization: streaming bandwidth and FMA throughput.
// Run with --microbench. Grounds the roofline in measured numbers rather than
// vendor spec sheets.
#include "styletts_ops.h"
#include "nnopt_error.h"
#include <cstdio>
#include <vector>
#include <chrono>
#include <clblast.h>

void run_microbench(OpenCLContext& ctx) {
    {   // device limits that constrain tiling
        cl_device_id d = ctx.device();
        cl_ulong lds = 0; size_t wg = 0; cl_uint cu = 0, freq = 0; cl_ulong gcache = 0;
        clGetDeviceInfo(d, CL_DEVICE_LOCAL_MEM_SIZE, sizeof(lds), &lds, nullptr);
        clGetDeviceInfo(d, CL_DEVICE_MAX_WORK_GROUP_SIZE, sizeof(wg), &wg, nullptr);
        clGetDeviceInfo(d, CL_DEVICE_MAX_COMPUTE_UNITS, sizeof(cu), &cu, nullptr);
        clGetDeviceInfo(d, CL_DEVICE_MAX_CLOCK_FREQUENCY, sizeof(freq), &freq, nullptr);
        clGetDeviceInfo(d, CL_DEVICE_GLOBAL_MEM_CACHE_SIZE, sizeof(gcache), &gcache, nullptr);
        fprintf(stderr, "  local_mem=%llu B  max_wg=%zu  CUs=%u  clk=%u MHz  L2=%llu B\n",
                (unsigned long long)lds, wg, cu, freq, (unsigned long long)gcache);
    }
    cl_command_queue q = ctx.queue();
    cl_program p = ctx.build_program_from_file("kernels/styletts.cl");
    if (!p) { fprintf(stderr, "microbench: program build failed\n"); return; }
    cl_int e;
    cl_kernel kc  = clCreateKernel(p, "bench_copy",  &e);
    cl_kernel kc4 = clCreateKernel(p, "bench_copy4", &e);
    cl_kernel kf  = clCreateKernel(p, "bench_fma",   &e);

    const int N = 8 * 1024 * 1024;    // 32 MB per buffer
    cl_mem a = clCreateBuffer(ctx.context(), CL_MEM_READ_WRITE, (size_t)N * 4, nullptr, &e);
    cl_mem b = clCreateBuffer(ctx.context(), CL_MEM_READ_WRITE, (size_t)N * 4, nullptr, &e);
    if (!a || !b) { fprintf(stderr, "microbench: alloc failed\n"); return; }

    auto timeit = [&](cl_kernel k, size_t gws, int reps, const char* name, double work, const char* unit) {
        clFinish(q);
        auto t0 = std::chrono::high_resolution_clock::now();
        for (int r = 0; r < reps; r++)
            clEnqueueNDRangeKernel(q, k, 1, nullptr, &gws, nullptr, 0, nullptr, nullptr);
        clFinish(q);
        auto t1 = std::chrono::high_resolution_clock::now();
        double s = std::chrono::duration<double>(t1 - t0).count();
        fprintf(stderr, "  %-22s %8.3f s   %8.2f %s\n", name, s, work * reps / s, unit);
    };

    int n = N;
    clSetKernelArg(kc, 0, sizeof(cl_mem), &a); clSetKernelArg(kc, 1, sizeof(cl_mem), &b);
    clSetKernelArg(kc, 2, sizeof(int), &n);
    timeit(kc, (size_t)N, 20, "copy  scalar", (double)N * 8 / 1e9, "GB/s");

    int n4 = N / 4;
    clSetKernelArg(kc4, 0, sizeof(cl_mem), &a); clSetKernelArg(kc4, 1, sizeof(cl_mem), &b);
    clSetKernelArg(kc4, 2, sizeof(int), &n4);
    timeit(kc4, (size_t)n4, 20, "copy  float4", (double)N * 8 / 1e9, "GB/s");

    int iters = 2048;
    size_t threads = 32768;
    clSetKernelArg(kf, 0, sizeof(cl_mem), &b); clSetKernelArg(kf, 1, sizeof(int), &iters);
    // 8 independent float4 FMAs per loop iteration = 8*4*2 = 64 flops
    timeit(kf, threads, 10, "fma  fp32 (ILP8)", (double)threads * iters * 64 / 1e9, "GFLOP/s");

    clReleaseMemObject(a); clReleaseMemObject(b);   // free 128 MB before GEMM allocs

    // CLBlast SGEMM at shapes representative of the conv workload after im2col.
    // Decides hand-tiled conv vs im2col+GEMM for the 91% of GPU time in convs.
    // GATED OFF by default: on Adreno 6xx (tablet SM-X210 AND razr Adreno620v2)
    // CLBlast's first-call kernel JIT either OOM-kills or hangs indefinitely,
    // which also blocks the copy4/FMA numbers above from flushing. im2col+CLBlast
    // is already ruled out for this port; opt in with NNOPT_BENCH_GEMM=1 only to
    // re-confirm the JIT behavior on a new device.
    if (const char* g = getenv("NNOPT_BENCH_GEMM"); g && g[0] == '1') {
        struct S { int M, N, K; const char* what; };
        const S shapes[] = {
            {  64,   4096,  192, "resblk C=64  k3 (N=4096)" },
            {  64,   4096,  704, "noise_res.1 C=64 k11"     },
            { 128,   3180,  384, "resblk C=128 k3  T=3180"  },
            { 128,   3180,  896, "noise_res.0 C=128 k7"     },
        };
        for (const auto& sh : shapes) {
            fprintf(stderr, "  [gemm] alloc M=%d N=%d K=%d (%.1f MB)\n", sh.M, sh.N, sh.K,
                    ((double)sh.M*sh.K + (double)sh.K*sh.N + (double)sh.M*sh.N) * 4 / 1e6);
            fflush(stderr);
            size_t szA = (size_t)sh.M * sh.K, szB = (size_t)sh.K * sh.N, szC = (size_t)sh.M * sh.N;
            cl_mem A = clCreateBuffer(ctx.context(), CL_MEM_READ_WRITE, szA * 4, nullptr, &e);
            cl_mem B = clCreateBuffer(ctx.context(), CL_MEM_READ_WRITE, szB * 4, nullptr, &e);
            cl_mem C = clCreateBuffer(ctx.context(), CL_MEM_READ_WRITE, szC * 4, nullptr, &e);
            if (!A || !B || !C) { fprintf(stderr, "  gemm alloc failed\n"); continue; }
            fprintf(stderr, "  [gemm] warmup...\n"); fflush(stderr);
            // warmup (CLBlast compiles + caches its kernel on first call)
            clblast::Gemm<float>(clblast::Layout::kRowMajor, clblast::Transpose::kNo,
                                 clblast::Transpose::kNo, sh.M, sh.N, sh.K, 1.0f,
                                 A, 0, sh.K, B, 0, sh.N, 0.0f, C, 0, sh.N, &q, nullptr);
            clFinish(q);
            const int reps = 5;
            auto t0 = std::chrono::high_resolution_clock::now();
            for (int r = 0; r < reps; r++)
                clblast::Gemm<float>(clblast::Layout::kRowMajor, clblast::Transpose::kNo,
                                     clblast::Transpose::kNo, sh.M, sh.N, sh.K, 1.0f,
                                     A, 0, sh.K, B, 0, sh.N, 0.0f, C, 0, sh.N, &q, nullptr);
            clFinish(q);
            auto t1 = std::chrono::high_resolution_clock::now();
            double sec = std::chrono::duration<double>(t1 - t0).count();
            double gf = 2.0 * sh.M * sh.N * sh.K * reps / 1e9 / sec;
            fprintf(stderr, "  GEMM %-26s %8.2f GFLOP/s  (%.1f ms/call)\n",
                    sh.what, gf, sec * 1000 / reps);
            clReleaseMemObject(A); clReleaseMemObject(B); clReleaseMemObject(C);
        }
    }

}
