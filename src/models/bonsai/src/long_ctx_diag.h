// Long-context decode diagnostics. Everything here is OFF unless its env var is set, so a shipped
// build behaves exactly as before.
//
//   BONSAI_KPROF=1   per-kernel GPU time from the hardware START/END timestamps
//                    (clGetEventProfilingInfo; the queue already has CL_QUEUE_PROFILING_ENABLE).
//                    Unlike BONSAI_PROF (clFinish around op groups), this does not serialise the
//                    queue, and it separates kernel execution from the GPU sitting idle between
//                    launches (host/driver overhead): idle = wall - sum(kernel time).
//   BONSAI_BW=1      device memory bandwidth, measured with a streaming read and a copy kernel, so
//                    "attention reaches N% of bandwidth" is against THIS phone, not a spec sheet.
//   BONSAI_ATTN2=... candidate attention kernels, a comma list of: sc (scores reading each K row
//                    once per GQA group), sm (softmax with one work-group per head instead of one
//                    work-item), out (attn-out split over keys, then a reduce). "1" or "all" = all.
//                    fp32 storage only; on an fp16 build it says so and keeps the shipped kernels.
//
// This file is shared verbatim by src/models/bonsai and src/models/bonsai-27b.
#pragma once

#include <CL/cl.h>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <map>
#include <string>
#include <vector>

namespace lcd {

// ---------------------------------------------------------------------------------------------
// Candidate kernels. Same layouts as kernels/attention.cl:
//   q [NH, D], k/v cache [S, KH, D], scores [NH, S], out [NH, D], counter[1] = S.
static const char* kAttn2Src = R"CLC(
__kernel void gqa_scores_grp(__global const float* q, __global const float* k,
                             __global float* scores, __global const int* counter,
                             const int NH, const int KH, const int D, const float scale) {
  const int tk = get_global_id(0), kvh = get_global_id(1);
  const int S = counter[1];
  if (tk >= S || kvh >= KH) return;
  const int nrep = NH / KH, D4 = D >> 2;
  const __global float4* kr = (const __global float4*)(k + ((long)tk * KH + kvh) * D);
  float acc[8] = {0, 0, 0, 0, 0, 0, 0, 0};
  for (int d = 0; d < D4; ++d) {
    const float4 kv = kr[d];
    for (int r = 0; r < nrep; ++r)
      acc[r] += dot(vload4(d, q + (kvh * nrep + r) * D), kv);
  }
  for (int r = 0; r < nrep; ++r) scores[(long)(kvh * nrep + r) * S + tk] = acc[r] * scale;
}

#define SMW 256
__kernel __attribute__((reqd_work_group_size(SMW, 1, 1)))
void gqa_softmax_wg(__global float* scores, __global const int* counter) {
  const int r = get_group_id(0), l = get_local_id(0), S = counter[1];
  __global float* row = scores + (long)r * S;
  __local float red[SMW];
  float m = -3.402823466e+38f;
  for (int c = l; c < S; c += SMW) m = fmax(m, row[c]);
  red[l] = m;
  barrier(CLK_LOCAL_MEM_FENCE);
  for (int s = SMW / 2; s > 0; s >>= 1) {
    if (l < s) red[l] = fmax(red[l], red[l + s]);
    barrier(CLK_LOCAL_MEM_FENCE);
  }
  m = red[0];
  barrier(CLK_LOCAL_MEM_FENCE);
  float sum = 0.0f;
  for (int c = l; c < S; c += SMW) { const float e = exp(row[c] - m); row[c] = e; sum += e; }
  red[l] = sum;
  barrier(CLK_LOCAL_MEM_FENCE);
  for (int s = SMW / 2; s > 0; s >>= 1) {
    if (l < s) red[l] += red[l + s];
    barrier(CLK_LOCAL_MEM_FENCE);
  }
  const float inv = 1.0f / (red[0] + 1e-20f);
  for (int c = l; c < S; c += SMW) row[c] *= inv;
}

// One work-group per (head, chunk of CH keys); lane = one float4 of head_dim, so a V row is read
// by consecutive lanes (coalesced). Partials [nsplit, NH, D] are summed by gqa_attn_out_reduce.
__kernel void gqa_attn_out_part(__global const float* scores, __global const float* v,
                                __global float4* part, __global const int* counter,
                                const int NH, const int KH, const int D, const int CH) {
  const int d4 = get_local_id(0), g = get_group_id(0), D4 = D >> 2;
  const int qh = g % NH, s = g / NH, S = counter[1];
  const int kvh = qh / (NH / KH);
  const int t0 = s * CH, t1 = min(S, t0 + CH);
  const __global float* sc = scores + (long)qh * S;
  const __global float4* v4 = (const __global float4*)v;
  float4 acc = (float4)(0.0f);
  for (int t = t0; t < t1; ++t) acc += sc[t] * v4[((long)t * KH + kvh) * D4 + d4];
  part[((long)s * NH + qh) * D4 + d4] = acc;
}

__kernel void gqa_attn_out_reduce(__global const float4* part, __global float* out,
                                  const int NH, const int D, const int nsplit) {
  const int i = get_global_id(0), D4 = D >> 2;
  if (i >= NH * D4) return;
  float4 a = (float4)(0.0f);
  for (int s = 0; s < nsplit; ++s) a += part[(long)s * NH * D4 + i];
  vstore4(a, i, out);
}

__kernel void bw_read(__global const float4* a, __global float* out, const int n4) {
  const int g = get_global_id(0), gs = get_global_size(0);
  float4 s = (float4)(0.0f);
  for (int i = g; i < n4; i += gs) s += a[i];
  out[g] = s.x + s.y + s.z + s.w;
}

__kernel void bw_copy(__global const float4* a, __global float4* b, const int n4) {
  const int g = get_global_id(0), gs = get_global_size(0);
  for (int i = g; i < n4; i += gs) b[i] = a[i];
}
)CLC";

static constexpr int kOutChunk = 256;   // keys per attn-out work-group

// ---------------------------------------------------------------------------------------------
struct KProf {
    bool on = getenv("BONSAI_KPROF") != nullptr;
    std::vector<std::pair<std::string, cl_event>> ev;

    // Event slot for the launch about to happen, or nullptr when off (zero cost).
    cl_event* next(cl_kernel k) {
        if (!on) return nullptr;
        char name[96] = {0};
        clGetKernelInfo(k, CL_KERNEL_FUNCTION_NAME, sizeof(name), name, nullptr);
        ev.emplace_back(name, nullptr);
        return &ev.back().second;
    }
    void reset() {
        for (auto& e : ev) if (e.second) clReleaseEvent(e.second);
        ev.clear();
    }
    // Call after clFinish. One line per kernel, then a summary line:
    //   KPROF <label> kernel=<name> calls_per_tok=<n> us_per_tok=<t> share=<%>
    //   KPROF <label> TOTAL gpu_ms_per_tok=<g> wall_ms_per_tok=<w> gpu_idle_pct=<i>
    void dump(const char* label, int ntok, double wall_s) {
        if (!on || ev.empty() || ntok <= 0) return;
        std::map<std::string, std::pair<double, long>> agg;   // ns, calls
        double tot = 0;
        for (auto& e : ev) {
            if (!e.second) continue;
            cl_ulong a = 0, b = 0;
            clGetEventProfilingInfo(e.second, CL_PROFILING_COMMAND_START, sizeof(a), &a, nullptr);
            clGetEventProfilingInfo(e.second, CL_PROFILING_COMMAND_END, sizeof(b), &b, nullptr);
            const double ns = b > a ? (double)(b - a) : 0.0;
            agg[e.first].first += ns;
            agg[e.first].second += 1;
            tot += ns;
        }
        std::vector<std::pair<std::string, std::pair<double, long>>> rows(agg.begin(), agg.end());
        std::sort(rows.begin(), rows.end(),
                  [](auto& x, auto& y) { return x.second.first > y.second.first; });
        for (auto& r : rows)
            fprintf(stderr, "KPROF %s kernel=%s calls_per_tok=%.1f us_per_tok=%.1f share=%.1f%%\n",
                    label, r.first.c_str(), (double)r.second.second / ntok,
                    r.second.first / 1e3 / ntok, 100.0 * r.second.first / tot);
        const double gpu_ms = tot / 1e6 / ntok, wall_ms = wall_s * 1e3 / ntok;
        fprintf(stderr, "KPROF %s TOTAL gpu_ms_per_tok=%.2f wall_ms_per_tok=%.2f gpu_idle_pct=%.1f\n",
                label, gpu_ms, wall_ms, wall_ms > 0 ? 100.0 * (1.0 - gpu_ms / wall_ms) : 0.0);
        fflush(stderr);
        reset();
    }
};

// ---------------------------------------------------------------------------------------------
struct Attn2 {
    bool sc = false, sm = false, out = false;
    cl_program prog = nullptr;
    cl_kernel k_sc = nullptr, k_sm = nullptr, k_part = nullptr, k_red = nullptr;
    cl_mem part = nullptr;

    bool any() const { return sc || sm || out; }

    void init(cl_context ctx, cl_device_id dev, size_t es, int NH, int D, int ctx_cap) {
        const char* e = getenv("BONSAI_ATTN2");
        if (!e || !*e) return;
        const std::string s = e;
        const bool all = s == "1" || s == "all";
        sc = all || s.find("sc") != std::string::npos;
        sm = all || s.find("sm") != std::string::npos;
        out = all || s.find("out") != std::string::npos;
        if (es != 4) {
            fprintf(stderr, "ATTN2 off: fp32 storage only (this build stores %zu-byte values)\n", es);
            sc = sm = out = false;
            return;
        }
        cl_int err;
        prog = clCreateProgramWithSource(ctx, 1, &kAttn2Src, nullptr, &err);
        if (err == CL_SUCCESS) err = clBuildProgram(prog, 1, &dev, "", nullptr, nullptr);
        if (err != CL_SUCCESS) {
            char log[4096] = {0};
            clGetProgramBuildInfo(prog, dev, CL_PROGRAM_BUILD_LOG, sizeof(log), log, nullptr);
            fprintf(stderr, "ATTN2 off: build failed (%d)\n%s\n", err, log);
            sc = sm = out = false;
            return;
        }
        k_sc = clCreateKernel(prog, "gqa_scores_grp", &err);
        k_sm = clCreateKernel(prog, "gqa_softmax_wg", &err);
        k_part = clCreateKernel(prog, "gqa_attn_out_part", &err);
        k_red = clCreateKernel(prog, "gqa_attn_out_reduce", &err);
        const int nsplit = (ctx_cap + kOutChunk - 1) / kOutChunk;
        part = clCreateBuffer(ctx, CL_MEM_READ_WRITE, (size_t)nsplit * NH * D * 4, nullptr, &err);
        fprintf(stderr, "ATTN2 on: scores=%s softmax=%s attn_out=%s (chunk %d keys)\n",
                sc ? "grp" : "orig", sm ? "wg" : "orig", out ? "split" : "orig", kOutChunk);
    }

    // Each returns false when the candidate is off, so the caller runs the shipped kernel.
    bool scores(cl_command_queue q, KProf& kp, cl_mem qb, cl_mem kc, cl_mem scr, cl_mem ctr,
                int NH, int KH, int D, int S) {
        if (!sc || NH / KH > 8) return false;
        const float scale = 1.0f / sqrtf((float)D);
        clSetKernelArg(k_sc, 0, sizeof(cl_mem), &qb);
        clSetKernelArg(k_sc, 1, sizeof(cl_mem), &kc);
        clSetKernelArg(k_sc, 2, sizeof(cl_mem), &scr);
        clSetKernelArg(k_sc, 3, sizeof(cl_mem), &ctr);
        clSetKernelArg(k_sc, 4, sizeof(int), &NH);
        clSetKernelArg(k_sc, 5, sizeof(int), &KH);
        clSetKernelArg(k_sc, 6, sizeof(int), &D);
        clSetKernelArg(k_sc, 7, sizeof(float), &scale);
        const size_t g[2] = {(size_t)S, (size_t)KH};
        return clEnqueueNDRangeKernel(q, k_sc, 2, nullptr, g, nullptr, 0, nullptr, kp.next(k_sc)) ==
               CL_SUCCESS;
    }
    bool softmax(cl_command_queue q, KProf& kp, cl_mem scr, cl_mem ctr, int rows) {
        if (!sm) return false;
        clSetKernelArg(k_sm, 0, sizeof(cl_mem), &scr);
        clSetKernelArg(k_sm, 1, sizeof(cl_mem), &ctr);
        const size_t g = (size_t)rows * 256, l = 256;
        return clEnqueueNDRangeKernel(q, k_sm, 1, nullptr, &g, &l, 0, nullptr, kp.next(k_sm)) ==
               CL_SUCCESS;
    }
    bool attn_out(cl_command_queue q, KProf& kp, cl_mem scr, cl_mem vc, cl_mem outb, cl_mem ctr,
                  int NH, int KH, int D, int S) {
        if (!out) return false;
        const int CH = kOutChunk, nsplit = (S + CH - 1) / CH;
        clSetKernelArg(k_part, 0, sizeof(cl_mem), &scr);
        clSetKernelArg(k_part, 1, sizeof(cl_mem), &vc);
        clSetKernelArg(k_part, 2, sizeof(cl_mem), &part);
        clSetKernelArg(k_part, 3, sizeof(cl_mem), &ctr);
        clSetKernelArg(k_part, 4, sizeof(int), &NH);
        clSetKernelArg(k_part, 5, sizeof(int), &KH);
        clSetKernelArg(k_part, 6, sizeof(int), &D);
        clSetKernelArg(k_part, 7, sizeof(int), &CH);
        const size_t l = (size_t)(D / 4), g = (size_t)NH * nsplit * l;
        if (clEnqueueNDRangeKernel(q, k_part, 1, nullptr, &g, &l, 0, nullptr, kp.next(k_part)) !=
            CL_SUCCESS)
            return false;
        clSetKernelArg(k_red, 0, sizeof(cl_mem), &part);
        clSetKernelArg(k_red, 1, sizeof(cl_mem), &outb);
        clSetKernelArg(k_red, 2, sizeof(int), &NH);
        clSetKernelArg(k_red, 3, sizeof(int), &D);
        clSetKernelArg(k_red, 4, sizeof(int), &nsplit);
        const size_t g2 = (size_t)NH * D / 4;
        return clEnqueueNDRangeKernel(q, k_red, 1, nullptr, &g2, nullptr, 0, nullptr,
                                      kp.next(k_red)) == CL_SUCCESS;
    }
};

// ---------------------------------------------------------------------------------------------
// BONSAI_BW=1: best of several launch shapes, 3 reps each, 256 MB buffers.
//   BWBENCH read_GBps=<r> copy_GBps=<c> (copy counts read + write bytes)
inline void bw_bench(cl_context ctx, cl_device_id dev, cl_command_queue q) {
    if (!getenv("BONSAI_BW")) return;
    cl_int err;
    cl_program p = clCreateProgramWithSource(ctx, 1, &kAttn2Src, nullptr, &err);
    if (err != CL_SUCCESS || clBuildProgram(p, 1, &dev, "", nullptr, nullptr) != CL_SUCCESS) {
        fprintf(stderr, "BWBENCH build failed\n");
        return;
    }
    cl_kernel kr = clCreateKernel(p, "bw_read", &err), kc = clCreateKernel(p, "bw_copy", &err);
    const size_t bytes = 256u << 20;
    const int n4 = (int)(bytes / 16);
    cl_mem a = clCreateBuffer(ctx, CL_MEM_READ_WRITE, bytes, nullptr, &err);
    cl_mem b = clCreateBuffer(ctx, CL_MEM_READ_WRITE, bytes, nullptr, &err);
    cl_mem o = clCreateBuffer(ctx, CL_MEM_READ_WRITE, (1u << 20) * 4, nullptr, &err);
    const float zero = 0.0f;
    clEnqueueFillBuffer(q, a, &zero, 4, 0, bytes, 0, nullptr, nullptr);
    clFinish(q);
    auto time_ns = [&](cl_kernel k, size_t g) {
        double best = 1e30;
        for (int rep = 0; rep < 3; ++rep) {
            cl_event e;
            const size_t l = 128;
            clEnqueueNDRangeKernel(q, k, 1, nullptr, &g, &l, 0, nullptr, &e);
            clWaitForEvents(1, &e);
            cl_ulong s0 = 0, s1 = 0;
            clGetEventProfilingInfo(e, CL_PROFILING_COMMAND_START, sizeof(s0), &s0, nullptr);
            clGetEventProfilingInfo(e, CL_PROFILING_COMMAND_END, sizeof(s1), &s1, nullptr);
            clReleaseEvent(e);
            best = std::min(best, (double)(s1 - s0));
        }
        return best;
    };
    double rd = 0, cp = 0;
    for (size_t g : {16384u, 65536u, 262144u, 1048576u}) {
        clSetKernelArg(kr, 0, sizeof(cl_mem), &a);
        clSetKernelArg(kr, 1, sizeof(cl_mem), &o);
        clSetKernelArg(kr, 2, sizeof(int), &n4);
        const double r = bytes / time_ns(kr, g);
        clSetKernelArg(kc, 0, sizeof(cl_mem), &a);
        clSetKernelArg(kc, 1, sizeof(cl_mem), &b);
        clSetKernelArg(kc, 2, sizeof(int), &n4);
        const double c = 2.0 * bytes / time_ns(kc, g);
        fprintf(stderr, "BWBENCH lanes=%zu read_GBps=%.1f copy_GBps=%.1f\n", g, r, c);
        rd = std::max(rd, r);
        cp = std::max(cp, c);
    }
    fprintf(stderr, "BWBENCH best read_GBps=%.1f copy_GBps=%.1f\n", rd, cp);
    fflush(stderr);
    clReleaseMemObject(a); clReleaseMemObject(b); clReleaseMemObject(o);
    clReleaseKernel(kr); clReleaseKernel(kc); clReleaseProgram(p);
}

// Bytes the SHIPPED attention kernels touch per decoded token at context length S, summed over
// the attention layers. Lower bounds: each K/V row counted once per q-head that reads it (the
// shipped scores/attn-out kernels re-read a GQA group's row once per head; whether the cache
// absorbs that is exactly what the measured GB/s shows).
//   BYTES S=<S> kv_read_MB=<a> scores_rw_MB=<b> per attention layer x <layers>
inline void print_bytes(int S, int layers, int NH, int KH, int D, size_t es) {
    const double kv_unique = 2.0 * S * KH * D * es;          // K + V rows, read once
    const double kv_headwise = 2.0 * S * NH * D * es;        // as the shipped kernels issue them
    const double scores = (double)NH * S * es * (1 /*w scores*/ + 3 /*softmax r*/ + 2 /*softmax w*/ + 1 /*attn-out r*/);
    fprintf(stderr,
            "BYTES S=%d layers=%d kv_unique_MB=%.2f kv_as_issued_MB=%.2f scores_rw_MB=%.2f "
            "per_token_total_unique_MB=%.1f\n",
            S, layers, kv_unique / 1e6, kv_headwise / 1e6, scores / 1e6,
            layers * (kv_unique + scores) / 1e6);
}

}  // namespace lcd
