// DeviceModel MLP dispatch: the 1-bit GEMV families (decode v4/v7/img,
// batched, and the CPU+GPU hybrid) plus SwiGLU. cpu_gemv_rows is the NEON
// host GEMV that feeds the hybrid path.
#include "model.h"

#if defined(__aarch64__)
#include <arm_neon.h>
#endif

// CPU+GPU HYBRID (BONSAI_HYBRID=1): the decode GEMVs are ALU-issue-bound on the
// 1-bit unpack and the Adreno uses only ~2.3 of ~11 GB/s, so the 2 A76 + 6 A55
// cores add INDEPENDENT unpack throughput with no bandwidth contention. The CPU
// computes the TOP `frac` rows of the two fat MLP GEMVs (gate+up, down) straight
// from the original mmap'd Q1 weights (zero extra RAM) while the GPU computes the
// rest. Same fp32 two-sum math (dot = s·(2·Σ_{bit=1}x − Σx)); tiny reduction-order
// diffs are argmax-safe. Microbench: CPU 8-thread ≈ 25% of GPU → ~1.25× decode.
static inline void cpu_gemv_rows(const uint8_t* w, int src_row0, int ncpu,
                                 int n_in, const float* x, float* out,
                                 int nthreads) {
    const int units = n_in / 128;
    std::vector<float> xsum(units);
    for (int u = 0; u < units; ++u) {
        float a = 0.f; const float* xu = x + u * 128;
        for (int i = 0; i < 128; ++i) a += xu[i];
        xsum[u] = a;
    }
    auto work = [&](int i0, int i1) {
#if defined(__aarch64__)
        const uint32x4_t sel = (uint32x4_t){1, 2, 4, 8};
#endif
        for (int i = i0; i < i1; ++i) {
            const uint8_t* p = w + (size_t)(src_row0 + i) * units * 18;
            float acc = 0.f;
            for (int u = 0; u < units; ++u, p += 18) {
                uint16_t sh; memcpy(&sh, p, 2);
                const float s = half_to_float(sh);
                const uint8_t* b = p + 2; const float* xu = x + u * 128;
#if defined(__aarch64__)
                float32x4_t a0 = vdupq_n_f32(0.f), a1 = vdupq_n_f32(0.f);
                for (int by = 0; by < 16; ++by) {
                    uint32_t wb = b[by];
                    uint32x4_t mL = vtstq_u32(vdupq_n_u32(wb), sel);
                    uint32x4_t mH = vtstq_u32(vdupq_n_u32(wb >> 4), sel);
                    float32x4_t xL = vld1q_f32(xu + by * 8);
                    float32x4_t xH = vld1q_f32(xu + by * 8 + 4);
                    a0 = vaddq_f32(a0, vreinterpretq_f32_u32(
                             vandq_u32(vreinterpretq_u32_f32(xL), mL)));
                    a1 = vaddq_f32(a1, vreinterpretq_f32_u32(
                             vandq_u32(vreinterpretq_u32_f32(xH), mH)));
                }
                float pos = vaddvq_f32(vaddq_f32(a0, a1));
#else
                float pos = 0.f;
                for (int by = 0; by < 16; ++by) {
                    uint8_t bb = b[by]; const float* xw = xu + by * 8;
                    for (int k = 0; k < 8; ++k) if (bb & (1 << k)) pos += xw[k];
                }
#endif
                acc += s * (2.f * pos - xsum[u]);
            }
            out[i] = acc;
        }
    };
    if (nthreads <= 1 || ncpu < 256) { work(0, ncpu); return; }
    std::vector<std::thread> th;
    int per = (ncpu + nthreads - 1) / nthreads;
    for (int t = 0; t < nthreads; ++t) {
        int a = t * per, b = std::min(ncpu, a + per);
        if (a >= b) break;
        th.emplace_back(work, a, b);
    }
    for (auto& t : th) t.join();
}

void DeviceModel::run_gemv_b(QW& W, cl_mem x, cl_mem out, int N, int K, int Ntot,
                int row_off, int M) {
    int a = 0;
    arg(k_gemv_b_, a++, sizeof(cl_mem), &W.bits, "gb.Wb");
    arg(k_gemv_b_, a++, sizeof(cl_mem), &W.scales, "gb.Ws");
    arg(k_gemv_b_, a++, sizeof(cl_mem), &x, "gb.x");
    arg(k_gemv_b_, a++, sizeof(cl_mem), &xsump_, "gb.xs");
    arg(k_gemv_b_, a++, sizeof(cl_mem), &out, "gb.o");
    arg(k_gemv_b_, a++, sizeof(int), &N, "gb.N");
    arg(k_gemv_b_, a++, sizeof(int), &K, "gb.K");
    arg(k_gemv_b_, a++, sizeof(int), &Ntot, "gb.Nt");
    arg(k_gemv_b_, a++, sizeof(int), &row_off, "gb.ro");
    arg(k_gemv_b_, a++, sizeof(int), &M, "gb.M");
    run1(k_gemv_b_, ((size_t)N + 63) / 64 * 64, 64, "gemv_b");
}

void DeviceModel::run_gemv(QW& W, cl_mem x, cl_mem out, int N, int K, int out_off,
              int n_dispatch) {
    if (n_dispatch < 0) n_dispatch = N;
    // Hybrid: fat GEMVs (gate/up, logits) take the LUT kernel (v7,
    // 2048 rows/WG amortizes the LUT build); narrow ones stay on v4.
    // v7 LUT measured 2.2x WORSE (lmem bank conflicts on random
    // per-lane indexing) — opt-in only for re-testing.
    if (use_img_) {
        int a = 0;
        arg(k_gemv_img_, a++, sizeof(cl_mem), &W.bits_img, "gi.Wimg");
        arg(k_gemv_img_, a++, sizeof(cl_mem), &W.scales, "gi.Ws");
        arg(k_gemv_img_, a++, sizeof(cl_mem), &x, "gi.x");
        arg(k_gemv_img_, a++, sizeof(cl_mem), &xsum_, "gi.xs");
        arg(k_gemv_img_, a++, sizeof(cl_mem), &out, "gi.o");
        arg(k_gemv_img_, a++, sizeof(int), &N, "gi.N");
        arg(k_gemv_img_, a++, sizeof(int), &K, "gi.K");
        arg(k_gemv_img_, a++, sizeof(int), &out_off, "gi.off");
        run1(k_gemv_img_, ((size_t)n_dispatch + 255) / 256 * 64, 64, "gemv_img");
        return;
    }
    const bool fat = N >= 16384 && getenv("BONSAI_LUT") != nullptr;
    cl_kernel k = fat ? k_gemv7_ : k_gemv_;
    int a = 0;
    arg(k, a++, sizeof(cl_mem), &W.bits, "gv.Wb");
    arg(k, a++, sizeof(cl_mem), &W.scales, "gv.Ws");
    arg(k, a++, sizeof(cl_mem), &x, "gv.x");
    arg(k, a++, sizeof(cl_mem), &xsum_, "gv.xs");
    arg(k, a++, sizeof(cl_mem), &out, "gv.o");
    arg(k, a++, sizeof(int), &N, "gv.N");
    arg(k, a++, sizeof(int), &K, "gv.K");
    arg(k, a++, sizeof(int), &out_off, "gv.off");
    if (fat) run1(k, ((size_t)n_dispatch + 2047) / 2048 * 128, 128, "gemv7");
    else     run1(k, ((size_t)n_dispatch + 255) / 256 * 64, 64, "gemv");
}

void DeviceModel::run_gemv_hybrid(QW& W, cl_mem x, cl_mem out, int N, int K, int out_off,
                     const uint8_t* cpu_w, int cpu_src_rows) {
    // Align the GPU/CPU split to 256 (the kernel's quad-row block) so the
    // GPU launch covers EXACTLY rows [0, ngpu) with no gap/overlap.
    int ngpu = ((N - (int)(N * hybrid_frac_)) / 256) * 256;
    int ncpu = N - ngpu;
    if (!hybrid_ || cpu_w == nullptr || ncpu < 256 || ncpu > cpu_src_rows) {
        run_gemv(W, x, out, N, K, out_off);
        return;
    }
    // Top `ncpu` output rows map onto the source tensor's last `ncpu` rows —
    // for the fused gate+up that is the standalone ffn_up tensor's tail.
    const int cpu_src_row0 = cpu_src_rows - ncpu;
    cl_command_queue q = ocl_.queue();
    // 1) read x to host (blocks on the prior RMSNorm; CPU needs it)
    CLCHECK(clEnqueueReadBuffer(q, x, CL_TRUE, 0, (size_t)K * 4,
                                hx_.data(), 0, nullptr, nullptr), "hyb.x");
    // PROBE: isolate the cost of the blocking readback alone — do the full
    // GPU op, no CPU/upload. If this regresses too, the drain is the wall.
    if (hybrid_readonly_) { run_gemv(W, x, out, N, K, out_off); return; }
    // 2) GPU computes the bottom rows — TRUE N (stride), launch only ngpu
    run_gemv(W, x, out, N, K, out_off, ngpu);
    clFlush(q);
    // 3) CPU computes the top `ncpu` rows from the original mmap weights,
    //    overlapping the GPU GEMV above
    cpu_gemv_rows(cpu_w, cpu_src_row0, ncpu, K, hx_.data(), hout_.data(),
                  hybrid_threads_);
    // 4) splice the CPU rows in — NON-blocking. The in-order queue keeps the
    //    GEMV → upload → swiglu order correct without draining the pipeline;
    //    hout_ stays valid until the next hybrid op's blocking readback.
    CLCHECK(clEnqueueWriteBuffer(q, out, CL_FALSE,
                                 (size_t)(out_off + ngpu) * 4,
                                 (size_t)ncpu * 4, hout_.data(), 0, nullptr,
                                 nullptr), "hyb.o");
}

void DeviceModel::run_swiglu(cl_mem g, cl_mem u, int n) {
    int a = 0;
    arg(k_swiglu_, a++, sizeof(cl_mem), &g, "sw.g");
    arg(k_swiglu_, a++, sizeof(cl_mem), &u, "sw.u");
    arg(k_swiglu_, a++, sizeof(int), &n, "sw.n");
    run1(k_swiglu_, n, 0, "swiglu");
}
