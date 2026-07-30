// Shared dispatch: the 1-bit GEMV + its x-sum pre-pass, embedding gather,
// RMSNorm, SwiGLU MLP, residual add, device argmax. Identical for both block
// kinds — these are the ops the dense port already proved out.
#include "model.h"

// SwiGLU MLP. gate+up are fused along the output dim (they share K), so one
// N=2*ffn launch instead of two.
void DeviceModel::mlp_block(int L, const BlockW& bw) {
    (void)L;
    const int H = m_.hidden, FF = m_.ffn;
    // gate + up as two GEMVs into the two halves of gu_.
    // ONE dispatch for gate and up. They read the same x and land in the two halves of gu_, which is
    // exactly the fused matrix's output layout — gate_ and up_ are sub-buffers of it.
    run_gemv(bw.wgu_up, xb_, gu_, bw.gu_off[1] + FF, H);
    if (const char* dd = getenv("BONSAI_DUMP_MLP")) if (L == atoi(dd)) {
        auto D2 = [&](const char* tag, cl_mem b, size_t nb) {
            std::vector<uint8_t> r(nb); clFinish(ocl_.queue());
            clEnqueueReadBuffer(ocl_.queue(), b, CL_TRUE, 0, nb, r.data(), 0, nullptr, nullptr);
            char pth[512]; snprintf(pth, sizeof(pth), "dump/m_%s.f32.bin", tag);
            if (FILE* f = fopen(pth, "wb")) { fwrite(r.data(),1,r.size(),f); fclose(f); }
        };
        D2("xb", xb_, (size_t)H * ES);
        D2("g",  gate_, (size_t)FF * ES);
        D2("u",  up_,   (size_t)FF * ES);
    }
    run_swiglu(gate_, up_, FF);
    run_xsum(gate_, FF);
    run_gemv(bw.wd, gate_, mix_, H, FF);
}

void DeviceModel::run_gemv(const QW& W, cl_mem x, cl_mem out, int N, int K, int out_off) {
    // Texture path when this tensor got an image view. Same maths, same weights — the only
    // difference is that the reads go through the texture engine's L1 instead of the buffer path.
    const bool img = use_img_ && W.bits_img != nullptr;
    int a = 0;
    arg(k_gemv_, a++, sizeof(cl_mem), &W.bits, "gv.Wb");
    arg(k_gemv_, a++, sizeof(cl_mem), &W.scales, "gv.Ws");
    arg(k_gemv_, a++, sizeof(cl_mem), &x, "gv.x");
    arg(k_gemv_, a++, sizeof(cl_mem), &xsum_, "gv.xs");
    arg(k_gemv_, a++, sizeof(cl_mem), &out, "gv.o");
    arg(k_gemv_, a++, sizeof(int), &N, "gv.N");
    arg(k_gemv_, a++, sizeof(int), &K, "gv.K");
    arg(k_gemv_, a++, sizeof(int), &out_off, "gv.off");
    const size_t per_group = (size_t)Q1_WG * Q1_ROWS;
    const size_t ng = ((size_t)N + per_group - 1) / per_group;   // row-blocks = workgroups if splits==1

    // Split-K when N alone can't fill the GPU. v4's parallelism is set entirely by N, and MEASURED on
    // the Adreno 840 that is the whole story: lm_head (N=248320, 485 groups) hits 40.3 GB/s while the
    // MLP down-projection (N=5120, 10 groups) crawls, for the SAME 12 MB of weights. Splitting the K
    // loop gives each row-block several workgroups, so occupancy stops depending on the matrix shape.
    const int U = K >> 7;
    int splits = 1;
    if (ng < SK_TARGET_GROUPS && U >= 2 * 2) {
        splits = (int)((SK_TARGET_GROUPS + ng - 1) / ng);
        if (splits > SK_MAX_SPLITS) splits = SK_MAX_SPLITS;
        while (splits > 1 && U / splits < 2) --splits;   // keep >=2 units of real work per slice
    }
    if (splits <= 1) {
        if (img) {
            int ai = 0;
            arg(k_gemv_img_, ai++, sizeof(cl_mem), &W.bits_img, "gvi.Wimg");
            arg(k_gemv_img_, ai++, sizeof(cl_mem), &W.scales, "gvi.Ws");
            arg(k_gemv_img_, ai++, sizeof(cl_mem), &x, "gvi.x");
            arg(k_gemv_img_, ai++, sizeof(cl_mem), &xsum_, "gvi.xs");
            arg(k_gemv_img_, ai++, sizeof(cl_mem), &out, "gvi.o");
            arg(k_gemv_img_, ai++, sizeof(int), &N, "gvi.N");
            arg(k_gemv_img_, ai++, sizeof(int), &K, "gvi.K");
            arg(k_gemv_img_, ai++, sizeof(int), &out_off, "gvi.off");
            run1(k_gemv_img_, ng * Q1_WG, Q1_WG, "gemv_img");
            return;
        }
        run1(k_gemv_, ng * Q1_WG, Q1_WG, "gemv");
        return;
    }
    if (img && use_fused_sk_) {
        // One dispatch instead of two. The reduce that used to follow was 27% of a decode token and
        // essentially all of it was launch cost, so folding it into the producer removes real time,
        // not just a line of code.
        cl_kernel kf = k_gemv_sk_fused_;
        int af = 0;
        arg(kf, af++, sizeof(cl_mem), &W.bits_img, "skf.Wimg");
        arg(kf, af++, sizeof(cl_mem), &W.scales, "skf.Ws");
        arg(kf, af++, sizeof(cl_mem), &x, "skf.x");
        arg(kf, af++, sizeof(cl_mem), &xsum_, "skf.xs");
        arg(kf, af++, sizeof(cl_mem), &sk_partial_, "skf.p");
        arg(kf, af++, sizeof(cl_mem), &sk_flags_, "skf.f");
        arg(kf, af++, sizeof(cl_mem), &out, "skf.o");
        arg(kf, af++, sizeof(int), &N, "skf.N");
        arg(kf, af++, sizeof(int), &K, "skf.K");
        arg(kf, af++, sizeof(int), &splits, "skf.s");
        arg(kf, af++, sizeof(int), &out_off, "skf.off");
        run2(kf, ng * Q1_WG, (size_t)splits, Q1_WG, "gemv_sk_fused");
        return;
    }
    if (img) {
        int ai = 0;
        arg(k_gemv_sk_img_, ai++, sizeof(cl_mem), &W.bits_img, "ski.Wimg");
        arg(k_gemv_sk_img_, ai++, sizeof(cl_mem), &W.scales, "ski.Ws");
        arg(k_gemv_sk_img_, ai++, sizeof(cl_mem), &x, "ski.x");
        arg(k_gemv_sk_img_, ai++, sizeof(cl_mem), &xsum_, "ski.xs");
        arg(k_gemv_sk_img_, ai++, sizeof(cl_mem), &sk_partial_, "ski.p");
        arg(k_gemv_sk_img_, ai++, sizeof(int), &N, "ski.N");
        arg(k_gemv_sk_img_, ai++, sizeof(int), &K, "ski.K");
        arg(k_gemv_sk_img_, ai++, sizeof(int), &splits, "ski.s");
        run1(k_gemv_sk_img_, ng * (size_t)splits * Q1_WG, Q1_WG, "gemv_sk_img");
        int a4 = 0;
        arg(k_sk_reduce_, a4++, sizeof(cl_mem), &sk_partial_, "skr.p");
        arg(k_sk_reduce_, a4++, sizeof(cl_mem), &out, "skr.o");
        arg(k_sk_reduce_, a4++, sizeof(int), &N, "skr.N");
        arg(k_sk_reduce_, a4++, sizeof(int), &splits, "skr.s");
        arg(k_sk_reduce_, a4++, sizeof(int), &out_off, "skr.off");
        run1(k_sk_reduce_, (size_t)N, 0, "sk_reduce");
        return;
    }
    int a2 = 0;
    arg(k_gemv_sk_, a2++, sizeof(cl_mem), &W.bits, "sk.Wb");
    arg(k_gemv_sk_, a2++, sizeof(cl_mem), &W.scales, "sk.Ws");
    arg(k_gemv_sk_, a2++, sizeof(cl_mem), &x, "sk.x");
    arg(k_gemv_sk_, a2++, sizeof(cl_mem), &xsum_, "sk.xs");
    arg(k_gemv_sk_, a2++, sizeof(cl_mem), &sk_partial_, "sk.p");
    arg(k_gemv_sk_, a2++, sizeof(int), &N, "sk.N");
    arg(k_gemv_sk_, a2++, sizeof(int), &K, "sk.K");
    arg(k_gemv_sk_, a2++, sizeof(int), &splits, "sk.s");
    run1(k_gemv_sk_, ng * (size_t)splits * Q1_WG, Q1_WG, "gemv_sk");

    int a3 = 0;
    arg(k_sk_reduce_, a3++, sizeof(cl_mem), &sk_partial_, "skr.p");
    arg(k_sk_reduce_, a3++, sizeof(cl_mem), &out, "skr.o");
    arg(k_sk_reduce_, a3++, sizeof(int), &N, "skr.N");
    arg(k_sk_reduce_, a3++, sizeof(int), &splits, "skr.s");
    arg(k_sk_reduce_, a3++, sizeof(int), &out_off, "skr.off");
    run1(k_sk_reduce_, (size_t)N, 0, "sk_reduce");
}


void DeviceModel::run_xsum(cl_mem x, int K) {
    int a = 0;
    arg(k_xsum_, a++, sizeof(cl_mem), &x, "xs.x");
    arg(k_xsum_, a++, sizeof(cl_mem), &xsum_, "xs.o");
    arg(k_xsum_, a++, sizeof(int), &K, "xs.K");
    const size_t items = (size_t)K / 64;   // one work-item per 64-lane half-unit
    run1(k_xsum_, (items + Q1_WG - 1) / Q1_WG * Q1_WG, Q1_WG, "xsum");
}

void DeviceModel::run_gather(const QW& W, cl_mem out, int token, int K) {
    int a = 0, V = m_.vocab;
    arg(k_gather_, a++, sizeof(cl_mem), &W.bits, "g.Wb");
    arg(k_gather_, a++, sizeof(cl_mem), &W.scales, "g.Ws");
    arg(k_gather_, a++, sizeof(cl_mem), &out, "g.out");
    arg(k_gather_, a++, sizeof(int), &token, "g.tok");
    arg(k_gather_, a++, sizeof(int), &V, "g.V");
    arg(k_gather_, a++, sizeof(int), &K, "g.K");
    run1(k_gather_, (size_t)K, 0, "gather");
}

void DeviceModel::run_rms(cl_mem x, cl_mem w, cl_mem out, int rows, int cols) {
    int a = 0;
    float eps = m_.rms_eps;
    arg(k_rms_, a++, sizeof(cl_mem), &x, "rm.x");
    arg(k_rms_, a++, sizeof(cl_mem), &w, "rm.w");
    arg(k_rms_, a++, sizeof(cl_mem), &out, "rm.o");
    arg(k_rms_, a++, sizeof(int), &rows, "rm.r");
    arg(k_rms_, a++, sizeof(int), &cols, "rm.c");
    arg(k_rms_, a++, sizeof(float), &eps, "rm.e");
    run1(k_rms_, (size_t)rows * 64, 64, "rms");
}

void DeviceModel::run_swiglu(cl_mem g, cl_mem u, int n) {
    int a = 0;
    arg(k_swiglu_, a++, sizeof(cl_mem), &g, "sw.g");
    arg(k_swiglu_, a++, sizeof(cl_mem), &u, "sw.u");
    arg(k_swiglu_, a++, sizeof(int), &n, "sw.n");
    run1(k_swiglu_, (size_t)n, 0, "swiglu");
}

void DeviceModel::run_add(cl_mem acc, cl_mem b, int n) {
    int a = 0;
    arg(k_add_, a++, sizeof(cl_mem), &acc, "ad.a");
    arg(k_add_, a++, sizeof(cl_mem), &b, "ad.b");
    arg(k_add_, a++, sizeof(int), &n, "ad.n");
    run1(k_add_, (size_t)n, 0, "add");
}

// Output head. Same maths as run_gemv, with two deliberate differences:
//   - the store is FP32 regardless of activation dtype (see q1_gemv4_img_f32: an argmax over 248320
//     half-precision logits ties on quantisation and returns the wrong token), and
//   - no split-K: at N=248320 the head already has 485 row-blocks, far past where splitting helps.
void DeviceModel::run_head(const QW& W, cl_mem x, cl_mem out, int N, int K) {
    cl_kernel k = (use_img_ && W.bits_img && k_gemv_head_) ? k_gemv_head_ : nullptr;
    if (!k) { run_gemv(W, x, out, N, K); return; }   // no image view -> fall back
    int a = 0, off = 0;
    arg(k, a++, sizeof(cl_mem), &W.bits_img, "hd.Wimg");
    arg(k, a++, sizeof(cl_mem), &W.scales, "hd.Ws");
    arg(k, a++, sizeof(cl_mem), &x, "hd.x");
    arg(k, a++, sizeof(cl_mem), &xsum_, "hd.xs");
    arg(k, a++, sizeof(cl_mem), &out, "hd.o");
    arg(k, a++, sizeof(int), &N, "hd.N");
    arg(k, a++, sizeof(int), &K, "hd.K");
    arg(k, a++, sizeof(int), &off, "hd.off");
    const size_t ng = ((size_t)N + (size_t)Q1_WG * Q1_ROWS - 1) / ((size_t)Q1_WG * Q1_ROWS);
    run1(k, ng * Q1_WG, Q1_WG, "gemv_head");
}

void DeviceModel::run_argmax(int N) {
    // Reads logits as fp32 — they are stored fp32 whatever the activation dtype, because an argmax
    // over 248320 half-precision values ties on quantisation and picks the wrong token.
    cl_kernel ka = k_argmax_f32_ ? k_argmax_f32_ : k_argmax_;
    int a = 0;
    arg(ka, a++, sizeof(cl_mem), &logits_, "am.l");
    arg(ka, a++, sizeof(cl_mem), &amax_, "am.o");
    arg(ka, a++, sizeof(int), &N, "am.n");
    run1(ka, 256, 256, "argmax");
}
