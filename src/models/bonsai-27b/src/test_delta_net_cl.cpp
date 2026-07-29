// Differential test for kernels/delta_net.cl — the OpenCL twin of
// src/test_delta_net.cpp.
//
// Reads the SAME fixture format, runs `delta_net_step` on the real device for T
// timesteps, and writes outputs in the SAME layout. scripts/test_kernel_delta.sh
// then diffs kernel output against gdn::step (which is itself verified against
// transformers). This closes the gap that let two bugs through: `delta_rule.h`
// was differential-tested, but `delta_net.cl` is a SEPARATE implementation of
// the same math and had only ever had a compile check.
//
//   fixture: i32 nv, nk, dk, dv, T
//            f32 a_neg[nv], dt_bias[nv]
//            T x { f32 q[nk*dk], k[nk*dk], v[nv*dv], a[nv], b[nv] }
//   output:  T x f32 out[nv*dv], then f32 S[nv*dk*dv]
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <string>
#include <vector>

#include "opencl_context.h"

#define CK(err, what)                                                    \
    do {                                                                 \
        cl_int e_ = (err);                                               \
        if (e_ != CL_SUCCESS) {                                          \
            fprintf(stderr, "FATAL CL %d at %s\n", e_, what); return 4;   \
        }                                                                \
    } while (0)

static void rd(FILE* f, void* p, size_t n) {
    if (fread(p, 1, n, f) != n) { fprintf(stderr, "short read\n"); exit(2); }
}

int main(int argc, char** argv) {
    if (argc < 4) {
        fprintf(stderr, "usage: %s <fixture.bin> <out.bin> <kernels_dir>\n", argv[0]);
        return 1;
    }
    FILE* f = fopen(argv[1], "rb");
    if (!f) { fprintf(stderr, "cannot open %s\n", argv[1]); return 1; }

    int32_t nv, nk, dk, dv, T;
    rd(f, &nv, 4); rd(f, &nk, 4); rd(f, &dk, 4); rd(f, &dv, 4); rd(f, &T, 4);
    fprintf(stderr, "[cl] nv=%d nk=%d dk=%d dv=%d T=%d\n", nv, nk, dk, dv, T);

    std::vector<float> a_neg(nv), dt_b(nv);
    rd(f, a_neg.data(), 4u * nv);
    rd(f, dt_b.data(), 4u * nv);

    const int key_dim = nk * dk, value_dim = nv * dv;
    const int qkv_dim = 2 * key_dim + value_dim;

    OpenCLContext ocl;
    if (!ocl.initialize()) { fprintf(stderr, "no OpenCL device\n"); return 1; }
    std::string kdir = argv[3];
    cl_program prog = ocl.build_program_from_file(kdir + "/delta_net.cl", "");
    if (!prog) { fprintf(stderr, "build failed\n"); return 4; }
    cl_int e;
    cl_kernel k = clCreateKernel(prog, "delta_net_step", &e);
    CK(e, "clCreateKernel");

    auto buf = [&](size_t n, cl_mem_flags fl) {
        cl_int err; cl_mem m = clCreateBuffer(ocl.context(), fl, n, nullptr, &err);
        if (err != CL_SUCCESS) { fprintf(stderr, "buf err %d\n", err); exit(4); }
        return m;
    };
    cl_mem d_S = buf(sizeof(float) * (size_t)nv * dk * dv, CL_MEM_READ_WRITE);
    cl_mem d_qkv = buf(sizeof(float) * qkv_dim, CL_MEM_READ_ONLY);
    cl_mem d_a = buf(sizeof(float) * nv, CL_MEM_READ_ONLY);
    cl_mem d_b = buf(sizeof(float) * nv, CL_MEM_READ_ONLY);
    cl_mem d_an = buf(sizeof(float) * nv, CL_MEM_READ_ONLY);
    cl_mem d_dt = buf(sizeof(float) * nv, CL_MEM_READ_ONLY);
    cl_mem d_out = buf(sizeof(float) * value_dim, CL_MEM_READ_WRITE);

    const float zero = 0.0f;
    CK(clEnqueueFillBuffer(ocl.queue(), d_S, &zero, 4, 0,
                           sizeof(float) * (size_t)nv * dk * dv, 0, nullptr, nullptr),
       "S zero");
    CK(clEnqueueWriteBuffer(ocl.queue(), d_an, CL_TRUE, 0, 4u * nv, a_neg.data(),
                            0, nullptr, nullptr), "a_neg");
    CK(clEnqueueWriteBuffer(ocl.queue(), d_dt, CL_TRUE, 0, 4u * nv, dt_b.data(),
                            0, nullptr, nullptr), "dt");

    int ai = 0;
    CK(clSetKernelArg(k, ai++, sizeof(cl_mem), &d_S), "S");
    CK(clSetKernelArg(k, ai++, sizeof(cl_mem), &d_qkv), "qkv");
    CK(clSetKernelArg(k, ai++, sizeof(cl_mem), &d_a), "a");
    CK(clSetKernelArg(k, ai++, sizeof(cl_mem), &d_b), "b");
    CK(clSetKernelArg(k, ai++, sizeof(cl_mem), &d_an), "a_neg");
    CK(clSetKernelArg(k, ai++, sizeof(cl_mem), &d_dt), "dt");
    CK(clSetKernelArg(k, ai++, sizeof(cl_mem), &d_out), "out");
    CK(clSetKernelArg(k, ai++, sizeof(int), &nv), "nv");
    CK(clSetKernelArg(k, ai++, sizeof(int), &nk), "nk");
    CK(clSetKernelArg(k, ai++, sizeof(int), &dk), "dk");
    CK(clSetKernelArg(k, ai++, sizeof(int), &dv), "dv");

    std::vector<float> q(key_dim), kk(key_dim), v(value_dim), a(nv), b(nv),
                       qkv(qkv_dim), out(value_dim);
    FILE* o = fopen(argv[2], "wb");
    if (!o) { fprintf(stderr, "cannot write %s\n", argv[2]); return 1; }

    for (int t = 0; t < T; ++t) {
        rd(f, q.data(), 4u * key_dim);
        rd(f, kk.data(), 4u * key_dim);
        rd(f, v.data(), 4u * value_dim);
        rd(f, a.data(), 4u * nv);
        rd(f, b.data(), 4u * nv);
        // the kernel consumes ONE fused buffer: q | k | v
        memcpy(qkv.data(), q.data(), 4u * key_dim);
        memcpy(qkv.data() + key_dim, kk.data(), 4u * key_dim);
        memcpy(qkv.data() + 2 * key_dim, v.data(), 4u * value_dim);
        CK(clEnqueueWriteBuffer(ocl.queue(), d_qkv, CL_FALSE, 0, 4u * qkv_dim,
                                qkv.data(), 0, nullptr, nullptr), "qkv");
        CK(clEnqueueWriteBuffer(ocl.queue(), d_a, CL_FALSE, 0, 4u * nv,
                                a.data(), 0, nullptr, nullptr), "a");
        CK(clEnqueueWriteBuffer(ocl.queue(), d_b, CL_FALSE, 0, 4u * nv,
                                b.data(), 0, nullptr, nullptr), "b");
        size_t gws = (size_t)nv * dv, lws = (size_t)dv;
        CK(clEnqueueNDRangeKernel(ocl.queue(), k, 1, nullptr, &gws, &lws,
                                  0, nullptr, nullptr), "delta_net_step");
        CK(clEnqueueReadBuffer(ocl.queue(), d_out, CL_TRUE, 0, 4u * value_dim,
                               out.data(), 0, nullptr, nullptr), "read out");
        fwrite(out.data(), 4, out.size(), o);
    }
    std::vector<float> S((size_t)nv * dk * dv);
    CK(clEnqueueReadBuffer(ocl.queue(), d_S, CL_TRUE, 0, 4 * S.size(), S.data(),
                           0, nullptr, nullptr), "read S");
    fwrite(S.data(), 4, S.size(), o);
    fclose(o); fclose(f);
    fprintf(stderr, "[cl] ok\n");
    return 0;
}
