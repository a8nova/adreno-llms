// DeviceModel driver for the qwen35 hybrid stack: constructor, the
// heterogeneous forward loop, program build, weight upload, scratch + state
// allocation, RoPE tables. Per-op dispatch lives in src/layers/.
//
// Mirrors Model::forward() in reference_model.h. Where the dense bonsai port
// runs one uniform block N times, this runs a block whose kind is read from
// meta.layer_types ("LLLFLLLF..." — 48 'L' + 16 'F' on the 27B).
#include "model.h"


static const float kZero = 0.0f;

// ── Device-dependent tunables ───────────────────────────────────────────────
// Defaults are the values swept on Adreno, so on that hardware this function changes nothing. It
// exists because the two constants that decide GPU occupancy were compiled in, and the reasoning
// behind them ("12 CUs", "two waves of 64") describes one vendor's hardware. On a 6-CU PowerVR DXT
// with 32-wide warps they are the wrong shape, and there was no way to try another without a
// rebuild. Both are read once, here, before any kernel is compiled or any buffer is sized.
void DeviceModel::select_tunables() {
    auto env_int = [](const char* k, long def) -> long {
        const char* v = getenv(k);
        if (!v || !*v) return def;
        char* end = nullptr;
        const long n = strtol(v, &end, 10);
        if (end == v || *end != '\0' || n <= 0) {
            fprintf(stderr, "[tune] ignoring %s='%s' (not a positive integer)\n", k, v);
            return def;
        }
        return n;
    };

    // Asking for more than the device allows fails EVERY dispatch with CL_INVALID_WORK_GROUP_SIZE,
    // which reaches the user as "the model produced no output". Clamp rather than die.
    size_t dev_max_wg = 0;
    clGetDeviceInfo(ocl_.device(), CL_DEVICE_MAX_WORK_GROUP_SIZE,
                    sizeof(dev_max_wg), &dev_max_wg, nullptr);

    int wg = (int)env_int("BONSAI_Q1_WG", Q1_WG_DEFAULT);
    if (wg < 32 || (wg & (wg - 1)) != 0) {
        fprintf(stderr, "[tune] BONSAI_Q1_WG=%d is not a power of two >= 32 — keeping %d\n",
                wg, Q1_WG_DEFAULT);
        wg = Q1_WG_DEFAULT;
    }
    if (dev_max_wg > 0 && (size_t)wg > dev_max_wg) {
        fprintf(stderr, "[tune] BONSAI_Q1_WG=%d exceeds device max %zu — clamping\n", wg, dev_max_wg);
        wg = (int)dev_max_wg;
    }
    Q1_WG = wg;

    // sk_partial_ is sized from SK_MAX_N, a compile-time constant built from the DEFAULTS, so a
    // raised group target would write past the end of it. Lowering — the PowerVR direction — always
    // shrinks the product and is safe.
    size_t sk = (size_t)env_int("BONSAI_SK_GROUPS", (long)SK_TARGET_GROUPS_DEFAULT);
    const size_t slots = sk * (size_t)Q1_WG * (size_t)Q1_ROWS;
    if (slots > SK_MAX_N) {
        const size_t cap = SK_MAX_N / ((size_t)Q1_WG * (size_t)Q1_ROWS);
        fprintf(stderr, "[tune] BONSAI_SK_GROUPS=%zu needs %zu split-K slots (cap %zu) — clamping to %zu\n",
                sk, slots, SK_MAX_N, cap);
        sk = cap ? cap : 1;
    }
    SK_TARGET_GROUPS = sk;

    if (Q1_WG != Q1_WG_DEFAULT || SK_TARGET_GROUPS != SK_TARGET_GROUPS_DEFAULT)
        fprintf(stderr, "[tune] Q1_WG=%d (default %d)  SK_TARGET_GROUPS=%zu (default %zu)\n",
                Q1_WG, Q1_WG_DEFAULT, SK_TARGET_GROUPS, SK_TARGET_GROUPS_DEFAULT);
}


DeviceModel::DeviceModel(const Nnb& nnb, OpenCLContext& ocl, const std::string& kdir)
    : nnb_(nnb), m_(nnb.meta), ocl_(ocl) {
    // Before kernels are compiled (-DWG_SIZE comes from Q1_WG) and before scratch is sized.
    select_tunables();
    if (m_.arch != "qwen35") {
        fprintf(stderr, "FATAL: this runtime is qwen35-only, got '%s'\n",
                m_.arch.c_str());
        exit(5);
    }
    // RESIDENT-ONLY. This runtime targets devices where the whole model fits in
    // GPU memory (Adreno 8xx / ≥~5 GB: Snapdragon 8 Elite etc.). All weights are
    // uploaded once and stay resident — no per-token streaming. The Razr-era
    // streaming/hybrid-residency subsystem was deleted: on a device that fits,
    // it was pure overhead. Fail loudly and early if the weights don't fit,
    // instead of OOMing mid-forward.
    {
        cl_ulong gmem = 0, maxalloc = 0;
        clGetDeviceInfo(ocl_.device(), CL_DEVICE_GLOBAL_MEM_SIZE, sizeof(gmem), &gmem, nullptr);
        clGetDeviceInfo(ocl_.device(), CL_DEVICE_MAX_MEM_ALLOC_SIZE, sizeof(maxalloc), &maxalloc, nullptr);
        double want = 0;
        for (const auto& [n, t] : nnb.tensors()) want += (double)t.nbytes;
        fprintf(stderr, "[mem] device global %llu MB, max single alloc %llu MB\n",
                (unsigned long long)(gmem >> 20), (unsigned long long)(maxalloc >> 20));
        fprintf(stderr, "[mem] weights (resident) %.0f MB = %.2fx global\n",
                want / 1048576.0, want / (double)gmem);
        // ~1.2 GB headroom for KV + recurrent state + activations + logits.
        const double need = want + 1200.0 * 1048576.0;
        if (need > (double)gmem) {
            fprintf(stderr,
                    "FATAL: model needs ~%.0f MB but device global is %llu MB. "
                    "This build is resident-only (8xx-class). Use a device with "
                    "more GPU memory.\n",
                    need / 1048576.0, (unsigned long long)(gmem >> 20));
            exit(6);
        }
    }
    if ((int)m_.layer_types.size() != m_.layers) {
        fprintf(stderr, "FATAL: layer_types has %zu entries, expected %d\n",
                m_.layer_types.size(), m_.layers);
        exit(5);
    }
    // GGUF ssm.* keys are Mamba-inherited misnomers (PORT_CHECKLIST.md §D).
    nv_ = m_.ssm_time_step_rank;   // num_v_heads
    nk_ = m_.ssm_group_count;      // num_k_heads
    dk_ = m_.ssm_state_size;
    dv_ = dk_;
    if (nv_ <= 0 || nk_ <= 0 || nv_ % nk_) {
        fprintf(stderr, "FATAL: bad v/k head counts %d/%d\n", nv_, nk_);
        exit(5);
    }
    key_dim_ = nk_ * dk_;
    value_dim_ = nv_ * dv_;
    qkv_dim_ = 2 * key_dim_ + value_dim_;
    conv_k_ = m_.ssm_conv_kernel;
    rot_ = m_.rope_dim_count > 0 ? m_.rope_dim_count : m_.head_dim;

    build_programs(kdir);
    upload_weights();
    // Say whether the texture path is actually in use. It is gated per tensor on
    // CL_DEVICE_IMAGE_MAX_BUFFER_SIZE, so "we implemented image1d_buffer" and "this device is
    // running it" are different claims, and only the log can tell them apart.
    fprintf(stderr, "BENCHMARK image_weights: %s (%d tensors imaged, %d fell back, max %zu texels)\n",
            use_img_ ? (img_made_ ? "ON" : "ON-but-none") : "OFF",
            img_made_, img_skipped_, img_max_texels_);
    fflush(stderr);
    make_scratch();
    make_rope_tables();
    tune_prefill_path();
    prof_.on = getenv("BONSAI_PROF") != nullptr;
    prof_.q = ocl_.queue();
}

DeviceModel::~DeviceModel() { prof_.dump(); }

int DeviceModel::forward_embed(const float* emb, int pos, bool want_logits) {
    const int H = m_.hidden;
    // Upload straight into the residual stream, replacing the embedding gather. Everything after
    // this is identical to a text token, which is the whole point of the merger projecting to 5120.
#ifdef BONSAI_FP16_STORAGE
    std::vector<uint16_t> h(H);
    for (int i = 0; i < H; ++i) h[i] = float_to_half(emb[i]);
    CLCHECK(clEnqueueWriteBuffer(ocl_.queue(), x_, CL_FALSE, 0, (size_t)H * ES, h.data(),
                                 0, nullptr, nullptr), "vis emb upload");
#else
    CLCHECK(clEnqueueWriteBuffer(ocl_.queue(), x_, CL_FALSE, 0, (size_t)H * ES, emb,
                                 0, nullptr, nullptr), "vis emb upload");
#endif
    return forward_body(pos, want_logits);
}

void DeviceModel::tune_prefill_path() {
    if (const char* f = getenv("BONSAI_PREFILL")) {          // explicit override wins
        prefill_batched_ = (atoi(f) != 0);
        fprintf(stderr, "BENCHMARK prefill_path: %s (forced)\n",
                prefill_batched_ ? "batched" : "sequential");
        return;
    }
    // NOT AT LOAD. Measuring this honestly costs 2*PF_CHUNK forwards EACH WAY — ~17 s on an Adreno
    // 840 — and that is paid before the user's first word, on every cold start, forever. Shipping it
    // in the constructor took a 79 s turn to 253 s: the measurement cost more than the thing it was
    // choosing between could ever save. A tuner is a diagnostic, so it lives in /bench where it is
    // asked for, and the default is the path with the best measured end-to-end number (sequential
    // + image1d: 79.4 s total, 9 tok/s prefill).
    // Cheap, real, and load-bearing: time ONE batched GEMM against the GEMM_MT sequential GEMVs it
    // replaces, on a real weight matrix. ~2 ms, versus the 17 s the full 128-forward A/B cost when I
    // put that in the constructor and turned a 79 s turn into 253 s.
    //
    // Measuring only the GEMM is justified by the census: 93% of forward_batch's time is in the GEMM
    // stages (b.mlp 68.5%, b.L_proj 17.6%, b.L_out/b.F_* the rest); the recurrent core is 0.3%. So
    // whichever kernel wins here decides the whole path.
    const int K = m_.hidden, N = m_.ffn, R = GEMM_MT;
    const QW& W = blocks_[0].wgu_up;
    cl_mem x = scratch((size_t)R * K * ES, "tp.x");
    cl_mem xs = scratch((size_t)R * (K / 64) * 4, "tp.xs");
    cl_mem o = scratch((size_t)R * N * ES, "tp.o");
    for (int m = 0; m < R; ++m) {
        cl_buffer_region r{(size_t)m * K * ES, (size_t)K * ES};
        cl_int se; cl_mem xr = clCreateSubBuffer(x, CL_MEM_READ_WRITE, CL_BUFFER_CREATE_TYPE_REGION, &r, &se);
        if (se == CL_SUCCESS) { run_gather(tok_embd_, xr, 1000 + m * 37, K); clReleaseMemObject(xr); }
    }
    run_xsum_m(x, xs, K, R);
    clFinish(ocl_.queue());
    auto timeit = [&](const std::function<void()>& f) {
        f(); clFinish(ocl_.queue());
        auto t0 = std::chrono::steady_clock::now();
        for (int i = 0; i < 3; ++i) f();
        clFinish(ocl_.queue());
        return std::chrono::duration<double, std::milli>(
                   std::chrono::steady_clock::now() - t0).count() / 3;
    };
    const double tseq = timeit([&] {
        for (int m = 0; m < R; ++m) {
            cl_buffer_region r{(size_t)m * K * ES, (size_t)K * ES};
            cl_int se; cl_mem xr = clCreateSubBuffer(x, CL_MEM_READ_WRITE, CL_BUFFER_CREATE_TYPE_REGION, &r, &se);
            if (se != CL_SUCCESS) return;
            run_xsum(xr, K);
            run_gemv(W, xr, o, N, K);
            clReleaseMemObject(xr);
        }
    });
    const double tgem = timeit([&] { run_gemm(W, x, xs, o, N, K, R); });
    // Only take the batched path on a CLEAR win. At parity the sequential path is the safer default:
    // it is the one with an end-to-end number on real hardware.
    // FORCED SEQUENTIAL. The cheap GEMM-only probe is not a valid proxy and it has now cost a real
    // regression: it selected "batched" on one run and TTFT went 26.1 s -> 32.8 s. Every measurement
    // of the REAL 64-block forward disagrees with it — 0.64x, 0.87x, 0.89x across builds — because
    // batching only reduces WEIGHT TRAFFIC and this path is ALU-bound (the MT sweep is the proof:
    // MT=2 vs MT=8 is 4x less traffic for the same time). A ~2 ms probe on a device whose measured
    // peak bandwidth swings 43-62 GB/s between runs is deciding on noise.
    //
    // Keep the measurement for /bench, ignore it for the default. BONSAI_PREFILL=1 forces batched.
    prefill_batched_ = false;
    size_t pmem = 0;
    clGetKernelWorkGroupInfo(k_gemm_img_ ? k_gemm_img_ : k_gemm_, ocl_.device(),
                             CL_KERNEL_PRIVATE_MEM_SIZE, sizeof(pmem), &pmem, nullptr);
    fprintf(stderr, "BENCHMARK prefill_path: %s (%d tok: %d GEMVs %.2f ms vs 1 GEMM %.2f ms, %.2fx; "
                    "gemm private %zu B/wi)\n",
            prefill_batched_ ? "batched" : "sequential", R, R, tseq, tgem,
            tgem > 0 ? tseq / tgem : 0.0, pmem);
    fflush(stderr);
    for (cl_mem b : {x, xs, o}) clReleaseMemObject(b);
}

void DeviceModel::reset_state() {
    for (int L = 0; L < m_.layers; ++L) {
        if (!rec_[L]) continue;   // 'F' (attention) blocks have neither buffer
        clEnqueueFillBuffer(ocl_.queue(), rec_[L], &kZero, 4, 0,
                            (size_t)nv_ * dk_ * dv_ * 4, 0, nullptr, nullptr);
        clEnqueueFillBuffer(ocl_.queue(), convs_[L], &kZero, 4, 0,
                            (size_t)qkv_dim_ * conv_k_ * 4, 0, nullptr, nullptr);
    }
    clFinish(ocl_.queue());
}

void DeviceModel::make_prefill_scratch() {
    if (pf_ready_) return;
    const int H = m_.hidden, FF = m_.ffn, D = m_.head_dim, NH = m_.heads, KH = m_.kv_heads;
    const size_t es = ES, C = PF_CHUNK;
    pf_x_ = scratch(C * H * es, "pf.x");
    pf_n_ = scratch(C * H * es, "pf.n");
    pf_mix_ = scratch(C * H * es, "pf.mix");
    // One xsum buffer reused across every K in the block; size it for the widest.
    const size_t maxK = (size_t)std::max(std::max(FF, H), std::max(value_dim_, NH * D));
    pf_xsum_ = scratch(C * (maxK / 64) * 4, "pf.xsum");
    pf_gate_ = scratch(C * (size_t)FF * es, "pf.gate");
    pf_up_ = scratch(C * (size_t)FF * es, "pf.up");
    pf_qkv_ = scratch(C * (size_t)qkv_dim_ * es, "pf.qkv");
    pf_z_ = scratch(C * (size_t)value_dim_ * es, "pf.z");
    pf_a_ = scratch(C * (size_t)nv_ * es, "pf.a");
    pf_b_ = scratch(C * (size_t)nv_ * es, "pf.b");
    pf_core_ = scratch(C * (size_t)value_dim_ * 4, "pf.core");
    pf_qg_ = scratch(C * (size_t)NH * 2 * D * es, "pf.qg");
    pf_k_ = scratch(C * (size_t)KH * D * es, "pf.k");
    pf_v_ = scratch(C * (size_t)KH * D * es, "pf.v");
    pf_att_ = scratch(C * (size_t)NH * D * es, "pf.att");

    // Row views, once. Every stride here is a multiple of CL_DEVICE_MEM_BASE_ADDR_ALIGN (128 B on
    // Adreno) — the a/b vectors, whose nv_*ES == 192 B stride is NOT, are copied instead.
    auto views = [&](cl_mem base, size_t stride, std::vector<cl_mem>* out) {
        out->resize(C);
        for (size_t m = 0; m < C; ++m) {
            cl_buffer_region r{m * stride, stride};
            cl_int e;
            (*out)[m] = clCreateSubBuffer(base, CL_MEM_READ_WRITE,
                                          CL_BUFFER_CREATE_TYPE_REGION, &r, &e);
            CLCHECK(e, "pf row view");
        }
    };
    views(pf_qkv_, (size_t)qkv_dim_ * es, &pf_qkv_r_);
    views(pf_z_, (size_t)value_dim_ * es, &pf_z_r_);
    views(pf_core_, (size_t)value_dim_ * 4, &pf_core_r_);
    views(pf_qg_, (size_t)NH * 2 * D * es, &pf_qg_r_);
    views(pf_k_, (size_t)KH * D * es, &pf_k_r_);
    views(pf_v_, (size_t)KH * D * es, &pf_v_r_);
    views(pf_att_, (size_t)NH * D * es, &pf_att_r_);
    pf_ready_ = true;
}

void DeviceModel::run_xsum_m(cl_mem x, cl_mem xs, int K, int rows) {
    int a = 0;
    const int items = rows * (K >> 6);
    arg(k_xsum_m_, a++, sizeof(cl_mem), &x, "xm.x");
    arg(k_xsum_m_, a++, sizeof(cl_mem), &xs, "xm.o");
    arg(k_xsum_m_, a++, sizeof(int), &K, "xm.K");
    arg(k_xsum_m_, a++, sizeof(int), &rows, "xm.M");
    run1(k_xsum_m_, (size_t)((items + 63) / 64 * 64), 64, "xsum_m");
}

void DeviceModel::run_gemm(const QW& W, cl_mem x, cl_mem xs, cl_mem out, int N, int K, int rows,
                           int n_stride, int n0) {
    if (n_stride <= 0) n_stride = N;
    // ONE ROW PER WORK-ITEM. MEASURED on an Adreno 840, ffn_gate N=17408 K=5120, 32 tokens:
    //     q1_gemm      180.79 ms   0.6 GB/s   private 2944 B/wi
    //     q1_gemm_r1    15.36 ms   6.5 GB/s   private  672 B/wi
    //     32 GEMVs      25.6  ms  15.6 GB/s
    // The 4-row tile kept acc[4][MT] + 4 uint4 + 4 float4 live across a 128-deep unrolled body; the
    // register file gave up and the accumulators spilled to private memory, which on Adreno is global
    // memory. Cutting to one row is 11.8x, and it is the first version that beats the per-token path.
    // The texture variants are NOT used here: both measured 0.6 GB/s, because 8 tiles re-reading the
    // same 12.5 MB thrash the image L1. Images stay on the GEMV, where each weight is read once.
    cl_kernel k = k_gemm_r1_ ? k_gemm_r1_ : k_gemm_;
    const bool r1 = (k == k_gemm_r1_);
    int a = 0;
    arg(k, a++, sizeof(cl_mem), &W.bits, "gm.Wb");
    arg(k, a++, sizeof(cl_mem), &W.scales, "gm.Ws");
    arg(k, a++, sizeof(cl_mem), &x, "gm.x");
    arg(k, a++, sizeof(cl_mem), &xs, "gm.xs");
    arg(k, a++, sizeof(cl_mem), &out, "gm.o");
    arg(k, a++, sizeof(int), &N, "gm.N");
    arg(k, a++, sizeof(int), &K, "gm.K");
    arg(k, a++, sizeof(int), &rows, "gm.M");
    if (r1) {
        arg(k, a++, sizeof(int), &n_stride, "gm.NW");
        arg(k, a++, sizeof(int), &n0, "gm.n0");
    }
    // r1 covers WG_SIZE rows per group, not WG_SIZE*4.
    const size_t ng = r1 ? (size_t)((N + Q1_WG - 1) / Q1_WG)
                         : (size_t)((N + Q1_WG * 4 - 1) / (Q1_WG * 4));
    const size_t tiles = (size_t)((rows + GEMM_MT - 1) / GEMM_MT);
    run1(k, ng * tiles * Q1_WG, Q1_WG, "gemm");
}

void DeviceModel::forward_batch(const float* embs, int count, int pos0) {
    if (count <= 0) return;
    const int H = m_.hidden, FF = m_.ffn, D = m_.head_dim, NH = m_.heads, KH = m_.kv_heads;
    make_prefill_scratch();
    cl_command_queue q = ocl_.queue();

    for (int c0 = 0; c0 < count; c0 += PF_CHUNK) {
        const int C = std::min(PF_CHUNK, count - c0);
        // Upload this tile's embeddings into the residual stream.
#ifdef BONSAI_FP16_STORAGE
        std::vector<uint16_t> h((size_t)C * H);
        for (int i = 0; i < C * H; ++i) h[i] = float_to_half(embs[(size_t)(c0)*H + i]);
        CLCHECK(clEnqueueWriteBuffer(q, pf_x_, CL_TRUE, 0, (size_t)C * H * ES, h.data(),
                                     0, nullptr, nullptr), "pf emb");
#else
        CLCHECK(clEnqueueWriteBuffer(q, pf_x_, CL_TRUE, 0, (size_t)C * H * ES,
                                     embs + (size_t)c0 * H, 0, nullptr, nullptr), "pf emb");
#endif
        for (int L = 0; L < m_.layers; ++L) {
            const BlockW& bw = blocks_[L];
            prof_.begin();
            run_rms(pf_x_, bw.attn_norm, pf_n_, C, H);
            run_xsum_m(pf_n_, pf_xsum_, H, C);
            prof_.end("b.norm");

            if (bw.kind == 'L') {
                const LinearLayerW& w = lin_[bw.idx];
                prof_.begin();
                // Slices of the fused input projection. Prefill keeps four separate [M][N] outputs
                // because the batched result is row-major per token — a fused output would interleave
                // the four projections within each row, which nothing downstream expects.
                run_gemm(w.win, pf_n_, pf_xsum_, pf_qkv_, qkv_dim_,   H, C, w.in_n, w.in_off[0]);
                run_gemm(w.win, pf_n_, pf_xsum_, pf_z_,   value_dim_, H, C, w.in_n, w.in_off[1]);
                run_gemm(w.win, pf_n_, pf_xsum_, pf_a_,   nv_,        H, C, w.in_n, w.in_off[2]);
                run_gemm(w.win, pf_n_, pf_xsum_, pf_b_,   nv_,        H, C, w.in_n, w.in_off[3]);
                prof_.end("b.L_proj");
                prof_.begin();
                // Recurrent core: strictly sequential, but weight-free apart from conv_w/A_log/dt_b.
                const size_t nvb = (size_t)nv_ * ES;
                for (int m = 0; m < C; ++m) {
                    // a and b are COPIED, not viewed: their row stride is nv_*ES == 192 B, which is
                    // not a multiple of CL_DEVICE_MEM_BASE_ADDR_ALIGN (128 B on Adreno), so
                    // clCreateSubBuffer would fail on every odd row. Copy offsets have no such rule,
                    // and 192 B is nothing.
                    CLCHECK(clEnqueueCopyBuffer(q, pf_a_, avec_, m * nvb, 0, nvb,
                                                0, nullptr, nullptr), "pf a");
                    CLCHECK(clEnqueueCopyBuffer(q, pf_b_, bvec_, m * nvb, 0, nvb,
                                                0, nullptr, nullptr), "pf b");
                    run_conv1d(pf_qkv_r_[m], convs_[L], w.conv_w, qkv_dim_, conv_k_);
                    run_delta_net(rec_[L], pf_qkv_r_[m], avec_, bvec_, w.A_log, w.dt_b,
                                  pf_core_r_[m]);
                    run_norm_gate(pf_core_r_[m], pf_z_r_[m], w.ssm_norm);
                }
                prof_.end("b.L_recur");
                prof_.begin();
                run_xsum_m(pf_core_, pf_xsum_, value_dim_, C);
                run_gemm(w.wout, pf_core_, pf_xsum_, pf_mix_, H, value_dim_, C);
                prof_.end("b.L_out");
            } else {
                const FullLayerW& w = full_[bw.idx];
                prof_.begin();
                run_gemm(w.win, pf_n_, pf_xsum_, pf_qg_, NH * 2 * D, H, C, w.in_n, w.in_off[0]);
                run_gemm(w.win, pf_n_, pf_xsum_, pf_k_,  KH * D,     H, C, w.in_n, w.in_off[1]);
                run_gemm(w.win, pf_n_, pf_xsum_, pf_v_,  KH * D,     H, C, w.in_n, w.in_off[2]);
                prof_.end("b.F_proj");
                prof_.begin();
                const size_t kvb = (size_t)KH * D * ES;
                for (int m = 0; m < C; ++m) {
                    const int pos = pos0 + c0 + m;
                    // Blocking, for the same stack-lifetime reason as full_block().
                    int ctr[2] = {pos, pos + 1};
                    CLCHECK(clEnqueueWriteBuffer(q, counter_, CL_TRUE, 0, 8, ctr, 0,
                                                 nullptr, nullptr), "pf counter");
                    run_split_qg(pf_qg_r_[m], q_, agate_, NH, D);
                    run_rms(q_, w.q_norm, q_, NH, D);
                    run_rms(pf_k_r_[m], w.k_norm, pf_k_r_[m], KH, D);
                    run_rope_partial(q_, NH, D, pos);
                    run_rope_partial(pf_k_r_[m], KH, D, pos);
                    CLCHECK(clEnqueueCopyBuffer(q, pf_k_r_[m], kcache_[L], 0, pos * kvb, kvb,
                                                0, nullptr, nullptr), "pf kcpy");
                    CLCHECK(clEnqueueCopyBuffer(q, pf_v_r_[m], vcache_[L], 0, pos * kvb, kvb,
                                                0, nullptr, nullptr), "pf vcpy");
                    run_scores(q_, kcache_[L], NH, KH, D, pos + 1);
                    run_softmax(NH, pos + 1);
                    run_attnout(vcache_[L], NH, KH, D);
                    run_apply_gate(att_, agate_, NH * D);
                    CLCHECK(clEnqueueCopyBuffer(q, att_, pf_att_r_[m], 0, 0, (size_t)NH * D * ES,
                                                0, nullptr, nullptr), "pf attcpy");
                }
                prof_.end("b.F_attn");
                prof_.begin();
                run_xsum_m(pf_att_, pf_xsum_, NH * D, C);
                run_gemm(w.wo, pf_att_, pf_xsum_, pf_mix_, H, NH * D, C);
                prof_.end("b.F_out");
            }
            run_add(pf_x_, pf_mix_, C * H);

            prof_.begin();
            run_rms(pf_x_, bw.post_norm, pf_n_, C, H);
            run_xsum_m(pf_n_, pf_xsum_, H, C);
            // Two slices of the ONE fused matrix. Prefill keeps separate gate/up buffers because the
            // batched output is [M][N] row-major — the two halves of a fused output would interleave
            // per token, which no consumer expects.
            const int NW = bw.gu_off[1] + FF;
            run_gemm(bw.wgu_up, pf_n_, pf_xsum_, pf_gate_, FF, H, C, NW, bw.gu_off[0]);
            run_gemm(bw.wgu_up, pf_n_, pf_xsum_, pf_up_,   FF, H, C, NW, bw.gu_off[1]);
            run_swiglu(pf_gate_, pf_up_, C * FF);
            run_xsum_m(pf_gate_, pf_xsum_, FF, C);
            run_gemm(bw.wd, pf_gate_, pf_xsum_, pf_mix_, H, FF, C);
            run_add(pf_x_, pf_mix_, C * H);
            prof_.end("b.mlp");
        }
        clFinish(q);
    }
}

int DeviceModel::forward(int token, int pos, bool want_logits) {
    if (pos >= CTX_CAP) {
        fprintf(stderr, "FATAL: pos %d >= CTX_CAP %d\n", pos, CTX_CAP);
        exit(3);
    }
    const int H = m_.hidden;
    prof_.begin();
    run_gather(tok_embd_, x_, token, H);   // token_embd resident; gathers one row
    prof_.end("gather");
    return forward_body(pos, want_logits);
}

// Decode a step whose input token is already on the DEVICE.
//
// forward() + read_token() costs a blocking 4-byte clEnqueueReadBuffer per token, which on an
// in-order queue drains the entire pipeline — the GPU sits idle from the moment it finishes until
// the host has read four bytes and enqueued the next step. It is unavoidable only because the host
// needs the token to call run_gather(). It does not: q1_row_gather3_dev reads it from a buffer.
//
// With the token on the device the host can enqueue step i+1 BEFORE reading step i, so the readback
// waits on work that is already done while the GPU is busy on the next token.
int DeviceModel::forward_dev(int pos, bool want_logits) {
    if (pos >= CTX_CAP) { fprintf(stderr, "FATAL: pos %d >= CTX_CAP %d\n", pos, CTX_CAP); exit(3); }
    const int H = m_.hidden;
    prof_.begin();
    {
        int a = 0, V = m_.vocab;
        arg(k_gather_dev_, a++, sizeof(cl_mem), &tok_embd_.bits, "gd.Wb");
        arg(k_gather_dev_, a++, sizeof(cl_mem), &tok_embd_.scales, "gd.Ws");
        arg(k_gather_dev_, a++, sizeof(cl_mem), &x_, "gd.out");
        arg(k_gather_dev_, a++, sizeof(cl_mem), &amax2_[amax_slot_], "gd.tok");
        arg(k_gather_dev_, a++, sizeof(int), &V, "gd.V");
        arg(k_gather_dev_, a++, sizeof(int), &H, "gd.K");
        run1(k_gather_dev_, (size_t)H, 0, "gather_dev");
    }
    prof_.end("gather");
    // This step's argmax lands in the OTHER slot, leaving the one just consumed readable until the
    // step after next. Without that the read below would race the kernel already queued.
    amax_slot_ ^= 1;
    amax_ = amax2_[amax_slot_];
    forward_body(pos, want_logits);
    return amax_slot_;
}

int DeviceModel::read_token_slot(int slot) {
    int tk = 0;
    CLCHECK(clEnqueueReadBuffer(ocl_.queue(), amax2_[slot], CL_TRUE, 0, 4, &tk, 0, nullptr, nullptr),
            "read amax slot");
    return tk;
}

/** Put the token the first device-fed step will consume where its gather will look for it. */
void DeviceModel::seed_token(int token) {
    amax_slot_ = 0;
    CLCHECK(clEnqueueWriteBuffer(ocl_.queue(), amax2_[0], CL_TRUE, 0, 4, &token, 0, nullptr, nullptr),
            "seed token");
}

// Everything after the residual stream is seeded — shared by forward() (embedding gathered from the
// vocab) and forward_embed() (embedding supplied by the vision tower). Splitting it here is what
// makes an image token indistinguishable from a text token to the other 64 blocks.
int DeviceModel::forward_body(int pos, bool want_logits) {
    const int H = m_.hidden;
    if (const char* dd = getenv("DUMP_DIR")) {
        std::vector<uint8_t> raw((size_t)H * ES);
        clFinish(ocl_.queue());
        clEnqueueReadBuffer(ocl_.queue(), x_, CL_TRUE, 0, raw.size(), raw.data(),
                            0, nullptr, nullptr);
        char path[512];
        snprintf(path, sizeof(path), "%s/emb_pos%04d.f32.bin", dd, pos);
        if (FILE* f = fopen(path, "wb")) { fwrite(raw.data(), 1, raw.size(), f); fclose(f); }
    }

    for (int L = 0; L < m_.layers; ++L) {
        const BlockW& bw = blocks_[L];
        // ---- token mixer ----
        prof_.begin();
        run_rms(x_, bw.attn_norm, xb_, 1, H);
        run_xsum(xb_, H);
        prof_.end("norm");
        if (bw.kind == 'L') {
            prof_.begin();
            linear_block(L, bw, lin_[bw.idx]);
            prof_.end("linear_attn");
        } else {
            prof_.begin();
            full_block(L, bw, full_[bw.idx], pos);
            prof_.end("full_attn");
        }
        run_add(x_, mix_, H);
        if (const char* dd = getenv("BONSAI_DUMP_MLP")) if (L == atoi(dd)) {
            auto D3 = [&](const char* tag, cl_mem b, size_t nb) {
                std::vector<uint8_t> r(nb); clFinish(ocl_.queue());
                clEnqueueReadBuffer(ocl_.queue(), b, CL_TRUE, 0, nb, r.data(), 0, nullptr, nullptr);
                char pth[512]; snprintf(pth, sizeof(pth), "dump/a_%s.f32.bin", tag);
                if (FILE* f = fopen(pth, "wb")) { fwrite(r.data(),1,r.size(),f); fclose(f); }
            };
            D3("xafteradd", x_, (size_t)H * ES);
            D3("mix", mix_, (size_t)H * ES);
        }
        // ---- MLP (identical for both kinds) ----
        prof_.begin();
        run_rms(x_, bw.post_norm, xb_, 1, H);
        run_xsum(xb_, H);
        mlp_block(L, bw);
        run_add(x_, mix_, H);
        prof_.end("mlp");
        // DUMP_DIR: write the residual stream after each block, same filenames
        // the host reference uses, so scripts/diff_block.py-style comparison
        // localises a device/host divergence to ONE block.
        if (const char* dd = getenv("DUMP_DIR")) {
            std::vector<uint8_t> raw((size_t)H * ES);
            clFinish(ocl_.queue());
            clEnqueueReadBuffer(ocl_.queue(), x_, CL_TRUE, 0, raw.size(),
                                raw.data(), 0, nullptr, nullptr);
            char path[512];
            snprintf(path, sizeof(path), "%s/blk_%02d_pos%04d.f32.bin", dd, L, pos);
            if (FILE* f = fopen(path, "wb")) { fwrite(raw.data(), 1, raw.size(), f); fclose(f); }
        }
    }

    if (!want_logits) return -1;
    prof_.begin();
    run_rms(x_, out_norm_, xb_, 1, H);
    run_xsum(xb_, H);
    run_head(w_out_, xb_, logits_, m_.vocab, H);
    run_argmax(m_.vocab);
    prof_.end("logits");
    if (getenv("BONSAI_TOPK")) print_topk();
    return 0;
}

int DeviceModel::read_token() {
    int t = 0;
    CLCHECK(clEnqueueReadBuffer(ocl_.queue(), amax_, CL_TRUE, 0, 4, &t, 0,
                                nullptr, nullptr), "read amax");
    return t;
}

void DeviceModel::print_topk() {
    // logits_ is fp32 regardless of activation storage now, so read it straight.
    std::vector<float> lg(m_.vocab);
    CLCHECK(clEnqueueReadBuffer(ocl_.queue(), logits_, CL_TRUE, 0, lg.size() * 4,
                                lg.data(), 0, nullptr, nullptr), "read logits");
    for (int k = 0; k < 5; ++k) {
        int bi = 0;
        for (int i = 1; i < m_.vocab; ++i) if (lg[i] > lg[bi]) bi = i;
        fprintf(stderr, "  top%d id=%-6d logit=%.6f\n", k, bi, lg[bi]);
        lg[bi] = -1e30f;
    }
}

void DeviceModel::build_programs(const std::string& kdir) {
    const char* kopts_env = getenv("BONSAI_KERNEL_OPTS");
    std::string kopts = kopts_env ? kopts_env : "";
    // Kernel and host MUST agree on the workgroup size — reqd_work_group_size makes a mismatch a
    // hard CL_INVALID_WORK_GROUP_SIZE at every dispatch. One constant, passed through here.
    // -cl-fast-relaxed-math on the LM kernels. It was on the VISION kernels only; a grep of this
    // path returned zero. Guide 8.2/8.3 rates the conformant math forms in the two slowest classes,
    // and this path is ALU-bound (92% GPU-busy at 61% of peak bandwidth), so instruction count is
    // the thing that matters.
    //
    // It DOES relax numerics: it permits mad-contraction, which q1_gemv.cl deliberately disables
    // under BONSAI_NO_FMA because Adreno's fused multiply-add flips near-tie logits against the CPU
    // oracle. Greedy decode picks an argmax, so a flipped near-tie can change a token. Watch the
    // reply for coherence, not just the clock. BONSAI_NO_FASTMATH=1 reverts.
    if (!getenv("BONSAI_NO_FASTMATH")) kopts += " -cl-fast-relaxed-math";
    kopts += " -DWG_SIZE=" + std::to_string(Q1_WG);
    kopts_base_ = kopts;   // everything except -DGEMM_MT, for the /bench MT sweep
    kopts += " -DGEMM_MT=" + std::to_string(GEMM_MT);
#ifdef BONSAI_FP16_STORAGE
    kopts += " -DUSE_FP16=1";
#endif
    auto B = [&](const char* f) {
        cl_program p = ocl_.build_program_from_file(kdir + "/" + f, kopts);
        if (!p) { fprintf(stderr, "FATAL: build %s\n", f); exit(4); }
        return p;
    };
    // Same as B, plus extra build flags for one file. Falls back to the plain build if the device
    // rejects the flag, so a driver without CL2.0 still gets a working (two-dispatch) program.
    auto B2 = [&](const char* f, const char* extra) {
        cl_program p = ocl_.build_program_from_file(kdir + "/" + f, kopts + extra);
        if (!p) {
            fprintf(stderr, "[q1] %s rejected%s — falling back to default build\n", f, extra);
            p = ocl_.build_program_from_file(kdir + "/" + f, kopts);
            cl2_ok_ = false;
        }
        if (!p) { fprintf(stderr, "FATAL: build %s\n", f); exit(4); }
        return p;
    };
    auto K = [&](cl_program p, const char* n) {
        cl_int e; cl_kernel k = clCreateKernel(p, n, &e);
        if (e != CL_SUCCESS) { fprintf(stderr, "FATAL: kernel %s err %d\n", n, e); exit(4); }
        return k;
    };
    // -cl-std=CL2.0: q1_gemv4_sk_img_fused needs C11 atomics with memory_scope_device to make
    // one workgroup's partials visible to another. The 1.x mem_fence builtins cannot express it.
    cl_program pq = B2("q1_gemv.cl", " -cl-std=CL2.0");
    k_xsum_ = K(pq, "q1_xsum");
    k_gemv_ = K(pq, "q1_gemv4");
    k_gemv_sk_ = K(pq, "q1_gemv4_sk");
    k_gemv_img_ = K(pq, "q1_gemv4_img");
    k_gemv_head_ = K(pq, "q1_gemv4_img_f32");
    k_argmax_f32_ = K(pq, "q1_argmax_f32");
    k_gemv_sk_img_ = K(pq, "q1_gemv4_sk_img");
    k_gemv_sk_fused_ = cl2_ok_ ? K(pq, "q1_gemv4_sk_img_fused") : nullptr;
    use_fused_sk_ = cl2_ok_ && k_gemv_sk_fused_ && getenv("BONSAI_NO_FUSED_SK") == nullptr;
    use_img_ = getenv("BONSAI_NO_IMAGE_WEIGHTS") == nullptr;
    k_sk_reduce_ = K(pq, "q1_sk_reduce");
    k_gemm_ = K(pq, "q1_gemm");
    k_gemm_img_ = K(pq, "q1_gemm_img");
    k_gemm_r1_ = K(pq, "q1_gemm_r1");
    // OFF. MEASURED on an Adreno 840, and this is the whole story in two numbers:
    //     gemv_sk_fused  187.7 us/call    (mask-add, split-K, image weights)
    //     gemv_lut_sk   4210.1 us/call    (LUT, split-K)      -> 22x SLOWER
    // Decode fell to 0.5 tok/s and a turn went 33 s -> 234 s.
    //
    // The LUT idea was sound on paper and the standalone kernel is bit-exact and fast (0.60 ms vs
    // ~0.65 ms for the shipped GEMV). What kills it is the combination with split-K: the table read
    // is a byte-indexed GATHER, every lane hitting a different address in a 655 KB table, and split-K
    // multiplies the concurrent workgroups by 5-8x. All of them thrash the same table, so the cache
    // locality the LUT depends on collapses. Without split-K the LUT is roughly break-even; with it,
    // catastrophic — and split-K is worth 2x on every shape in this model, so it is not optional.
    //
    // Net: the LUT trades ALU for cache pressure, and this workload cannot pay in that currency.
    k_xsum_m_ = K(pq, "q1_xsum_m");
    k_gather_ = K(pq, "q1_row_gather3");
    k_gather_dev_ = K(pq, "q1_row_gather3_dev");
    k_argmax_ = K(pq, "q1_argmax");
    k_rms_ = K(B("rmsnorm.cl"), "rmsnorm_forward");
    cl_program pa = B("attention.cl");
    k_scores_ = K(pa, "gqa_attn_scores");
    k_softmax_ = K(pa, "gqa_softmax");
    k_attnout_ = K(pa, "gqa_attn_out");
    cl_program pg = B("gated_attention.cl");
    k_split_qg_ = K(pg, "attn_split_qg");
    k_rope_p_ = K(pg, "rope_partial");
    k_apply_gate_ = K(pg, "attn_apply_gate");
    cl_program pd = B("delta_net.cl");
    k_dn_step_ = K(pd, "delta_net_step");
    k_dn_norm_ = K(pd, "delta_net_norm_gate");
    k_dn_conv_ = K(pd, "delta_net_conv1d");
    k_swiglu_ = K(B("mlp.cl"), "swiglu_inplace");
    k_add_ = K(B("utils.cl"), "element_add");
}

// Wrap an existing bits buffer as a 1D image so the GEMV can read weights through the texture engine
// and its dedicated L1 (Qualcomm 80-NB295-11 §6.2/§7.1.5.3). This is a VIEW: cl_image_desc::buffer
// aliases the same allocation, so it adds a descriptor and not 3.5 GB.
//
// Returns null (caller falls back to the buffer kernel) when the tensor exceeds the device's
// image1d_buffer width. That is a real limit, not a formality: token_embd and the output head are
// U*N ~= 9.9M texels and are exactly the tensors most likely to be refused.
cl_mem DeviceModel::make_bits_image(cl_mem bits, size_t texels) {
    if (!img_max_texels_) {
        clGetDeviceInfo(ocl_.device(), CL_DEVICE_IMAGE_MAX_BUFFER_SIZE,
                        sizeof(img_max_texels_), &img_max_texels_, nullptr);
    }
    if (texels > img_max_texels_) { ++img_skipped_; return nullptr; }
    cl_image_format fmt{};
    fmt.image_channel_order = CL_RGBA;
    fmt.image_channel_data_type = CL_UNSIGNED_INT32;   // one uint4 texel == one 128-bit Q1 unit
    cl_image_desc desc{};
    desc.image_type = CL_MEM_OBJECT_IMAGE1D_BUFFER;
    desc.image_width = texels;
    desc.buffer = bits;
    cl_int err = CL_SUCCESS;
    cl_mem img = clCreateImage(ocl_.context(), CL_MEM_READ_ONLY, &fmt, &desc, nullptr, &err);
    if (err != CL_SUCCESS) { ++img_skipped_; return nullptr; }
    ++img_made_;
    return img;
}

cl_mem DeviceModel::upload(const void* p, size_t n, const char* what) {
    cl_int err;
    cl_mem m = clCreateBuffer(ocl_.context(),
                              CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, n,
                              (void*)p, &err);
    CLCHECK(err, what);
    weight_bytes_ += n;
    return m;
}

cl_mem DeviceModel::scratch(size_t n, const char* what) {
    cl_int err;
    cl_mem m = clCreateBuffer(ocl_.context(), CL_MEM_READ_WRITE, n, nullptr, &err);
    CLCHECK(err, what);
    return m;
}

// K-major split-stream repack at upload: bits_t[u][row] / scales_t[u][row], so
// adjacent GEMV lanes (rows) read adjacent words. Same 1.125 bit/weight, just
// deinterleaved. (Phase 4 moves this into the .nnb so weights can be streamed.)
// STREAMING: don't upload. Record where the bytes live in the mmap and how much
// staging space this tensor needs. Requires kind q1t (pre-repacked at conversion)
// so the copy is a straight memcpy with no CPU transform.
DeviceModel::QW DeviceModel::upload_q1_fused(const std::vector<std::string>& names,
                                             std::vector<int>* starts, int* n_total) {
    // Row alignment, in ROWS, so every consumer's sub-buffer origin is legal. Queried, not assumed:
    // the value is 1024 bits on the Adreno parts here but it is a device property.
    cl_uint align_bits = 1024;
    clGetDeviceInfo(ocl_.device(), CL_DEVICE_MEM_BASE_ADDR_ALIGN, sizeof(align_bits), &align_bits,
                    nullptr);
    const size_t align_rows = std::max<size_t>(1, (align_bits / 8) / ES);

    size_t U = 0, ntot = 0;
    starts->clear();
    for (const std::string& n : names) {
        const Tensor& t = nnb_.get(n);
        const size_t u = (size_t)t.ne(0) / 128, N = (size_t)t.ne(1);
        if (U == 0) U = u;
        if (u != U) { fprintf(stderr, "FATAL: fuse %s has U=%zu, group has %zu\n", n.c_str(), u, U); exit(4); }
        if (t.kind != Tensor::Q1T) { fprintf(stderr, "FATAL: fuse %s is not Q1T\n", n.c_str()); exit(4); }
        starts->push_back((int)ntot);
        ntot += ((N + align_rows - 1) / align_rows) * align_rows;   // pad up to the next legal origin
    }
    *n_total = (int)ntot;

    // Zero-filled, so padding rows contribute a defined (and discarded) result rather than whatever
    // was in the allocation.
    std::vector<uint8_t> bits(U * ntot * 16, 0), scales(U * ntot * 2, 0);
    for (size_t i = 0; i < names.size(); ++i) {
        const Tensor& t = nnb_.get(names[i]);
        const size_t N = (size_t)t.ne(1), off = (size_t)(*starts)[i];
        const uint8_t* sb = t.data;                 // [u][row][16]
        const uint8_t* ss = t.data + U * N * 16;    // [u][row][2]
        for (size_t u = 0; u < U; ++u) {
            memcpy(&bits[(u * ntot + off) * 16], sb + u * N * 16, N * 16);
            memcpy(&scales[(u * ntot + off) * 2], ss + u * N * 2, N * 2);
        }
        nnb_.release(t.data, t.nbytes);
    }
    QW w;
    const std::string tag = names[0] + "+fused";
    w.bits = upload(bits.data(), bits.size(), (tag + ".bits").c_str());
    w.scales = upload(scales.data(), scales.size(), (tag + ".scales").c_str());
    w.bits_img = make_bits_image(w.bits, U * ntot);
    return w;
}

DeviceModel::QW DeviceModel::upload_q1(const std::string& n) {
    const Tensor& t = nnb_.get(n);
    const size_t U = (size_t)t.ne(0) / 128, N = (size_t)t.ne(1);
    // Q1T is ALREADY K-major split-stream (repacked by --device-layout at
    // conversion). Repacking it again would scramble it — upload the two
    // streams straight through. This bites the tensors that stay RESIDENT even
    // in streaming mode (token_embd, output head): they take this path, not
    // stage_q1(), so they must honour the layout too.
    if (t.kind == Tensor::Q1T) {
        const size_t nbits = U * N * 16;
        QW w;
        w.bits = upload(t.data, nbits, (n + ".bits").c_str());
        w.scales = upload(t.data + nbits, U * N * 2, (n + ".scales").c_str());
        w.bits_img = make_bits_image(w.bits, U * N);
        // These bytes are now on the GPU and will never be read from the mmap again. Give the pages
        // back immediately — holding them doubles peak RSS on a shared-memory GPU. See Nnb::release.
        nnb_.release(t.data, t.nbytes);
        return w;
    }
    std::vector<uint8_t> bits(U * N * 16), scales(U * N * 2);
    const uint8_t* p = t.data;
    for (size_t row = 0; row < N; ++row)
        for (size_t u = 0; u < U; ++u) {
            const uint8_t* unit = p + (row * U + u) * 18;
            memcpy(&scales[(u * N + row) * 2], unit, 2);
            memcpy(&bits[(u * N + row) * 16], unit + 2, 16);
        }
    QW w;
    w.bits = upload(bits.data(), bits.size(), (n + ".bits").c_str());
    w.scales = upload(scales.data(), scales.size(), (n + ".scales").c_str());
    nnb_.release(t.data, t.nbytes);
    return w;
}

// Upload EVERY weight to the device, once, and build the per-block descriptors
// the forward loop indexes. Resident-only: there is no streaming counterpart —
// see the constructor's memory gate, which has already refused the run if the
// whole model plus headroom doesn't fit.
//
// This walks meta.layer_types so each block gets only the tensors its kind
// actually has: an 'L' block has no attn_q/k/v/output and an 'F' block has no
// ssm_* or attn_gate tensor at all. Asking the .nnb for the other kind's names
// would throw, so the branch is load-bearing, not tidiness.
void DeviceModel::upload_weights() {
    // f32 tensors (norm weights, conv taps, the per-head ssm scalars) go up
    // verbatim — nbytes is authoritative, so this stays correct if a dim moves.
    auto up_f32 = [&](const std::string& n) -> cl_mem {
        const Tensor& t = nnb_.get(n);
        cl_mem m = upload(t.data, t.nbytes, n.c_str());
        nnb_.release(t.data, t.nbytes);   // see Nnb::release — halves peak RSS
        return m;
    };

    tok_embd_ = upload_q1("token_embd.weight");
    w_out_    = upload_q1("output.weight");
    out_norm_ = up_f32("output_norm.weight");

    blocks_.resize(m_.layers);
    lin_.clear();
    full_.clear();
    for (int L = 0; L < m_.layers; ++L) {
        // HEARTBEAT. Uploading 3.6 GB is minutes of work on a cold run, and the host that spawned us
        // can only distinguish "slow" from "wedged" by whether we say anything. Going silent for the
        // whole upload got this process killed mid-load and reported as "the model produced no
        // output". Cheap to print, and it also localises a failure to a block index.
        if (L % 8 == 0) {
            // RSS is the number that actually decides whether we survive: on a shared-memory GPU the
            // device buffers count against this process, so a climb toward ~2x the weight size means
            // the mmap pages are NOT being released and an OOM kill is coming. See Nnb::release.
            long rss_kb = 0;
            if (FILE* f = fopen("/proc/self/status", "r")) {
                char ln[256];
                while (fgets(ln, sizeof(ln), f))
                    if (!strncmp(ln, "VmRSS:", 6)) { sscanf(ln + 6, "%ld", &rss_kb); break; }
                fclose(f);
            }
            fprintf(stderr, "[mem] uploading block %d/%d (rss %.0f MB)\n",
                    L, m_.layers, rss_kb / 1024.0);
            fflush(stderr);
        }
        const std::string p = "blk." + std::to_string(L) + ".";
        BlockW& bw = blocks_[L];
        bw.kind = m_.layer_types[L];
        // Shared by both kinds: the pre-mixer norm, the post-attention norm,
        // and the SwiGLU MLP. gate/up stay two tensors (two GEMVs into the
        // halves of one gu_ buffer) rather than one fused matrix.
        bw.attn_norm = up_f32(p + "attn_norm.weight");
        bw.post_norm = up_f32(p + "post_attention_norm.weight");
        {
            std::vector<int> offs; int ntot = 0;
            bw.wgu_up = upload_q1_fused({p + "ffn_gate.weight", p + "ffn_up.weight"}, &offs, &ntot);
            bw.gu_off[0] = offs[0]; bw.gu_off[1] = offs[1];
        }
        bw.wd  = upload_q1(p + "ffn_down.weight");

        if (bw.kind == 'L') {
            bw.idx = (int)lin_.size();
            LinearLayerW w;
            // "attn_gate" is the GGUF's misnomer for in_proj_z, the SSM output
            // gate — an 'F' block has no such tensor.
            {
                std::vector<int> offs;
                w.win = upload_q1_fused({p + "attn_qkv.weight", p + "attn_gate.weight",
                                         p + "ssm_alpha.weight", p + "ssm_beta.weight"},
                                        &offs, &w.in_n);
                for (int i = 0; i < 4; ++i) w.in_off[i] = offs[i];
            }
            w.wout   = upload_q1(p + "ssm_out.weight");
            w.conv_w = up_f32(p + "ssm_conv1d.weight");
            // ssm_a ALREADY holds -exp(A_log): the llama.cpp converter
            // pre-computes it. Applying exp() again shortens the state's
            // memory and stays fluent, so nothing crashes — it just stops
            // using its context. Upload raw; delta_net.cl multiplies directly.
            w.A_log  = up_f32(p + "ssm_a");
            w.dt_b   = up_f32(p + "ssm_dt.bias");
            w.ssm_norm = up_f32(p + "ssm_norm.weight");
            lin_.push_back(w);
        } else {
            bw.idx = (int)full_.size();
            FullLayerW w;
            // attn_q is [5120, 12288] = 24 heads x 512, q and the per-head
            // output gate INTERLEAVED — attn_split_qg de-interleaves it.
            {
                std::vector<int> offs;
                w.win = upload_q1_fused({p + "attn_q.weight", p + "attn_k.weight",
                                         p + "attn_v.weight"}, &offs, &w.in_n);
                for (int i = 0; i < 3; ++i) w.in_off[i] = offs[i];
            }
            w.wo = upload_q1(p + "attn_output.weight");
            w.q_norm = up_f32(p + "attn_q_norm.weight");
            w.k_norm = up_f32(p + "attn_k_norm.weight");
            full_.push_back(w);
        }
    }
    fprintf(stderr, "[mem] resident: %zu blocks (%zu 'L' + %zu 'F') uploaded\n",
            blocks_.size(), lin_.size(), full_.size());
}

// ── GEMV kernel sweep ───────────────────────────────────────────────────────
// Every variant below has the SAME signature (bits_t, scales_t, x, xsum, out, N, K, out_off) and
// differs only in how many output rows one work-item accumulates. Combined with -DWG_SIZE that is a
// 2-D occupancy/reuse sweep, which is exactly the axis the Adreno 620 tuning fixed at (4 rows, 64)
// and that a 12-CU part has every reason to want elsewhere.
// End-to-end bench, written to STDOUT so it lands in the chat bubble — the only channel we have on a
// device we cannot attach adb to. Answers one question the tok/s number cannot: is the model slow
// because the GPU is saturated, or because we are not feeding it?

void DeviceModel::make_scratch() {
    const int H = m_.hidden, D = m_.head_dim, NH = m_.heads, KH = m_.kv_heads;
    const size_t es = ES;
    x_ = scratch(H * es, "x");
    xb_ = scratch(H * es, "xb");
    mix_ = scratch(H * es, "mix");
    gu_ = scratch((size_t)2 * m_.ffn * es, "gu");
    cl_int serr;
    cl_buffer_region rg{0, (size_t)m_.ffn * es};
    cl_buffer_region ru{(size_t)m_.ffn * es, (size_t)m_.ffn * es};
    gate_ = clCreateSubBuffer(gu_, 0, CL_BUFFER_CREATE_TYPE_REGION, &rg, &serr);
    CLCHECK(serr, "sub gate");
    up_ = clCreateSubBuffer(gu_, 0, CL_BUFFER_CREATE_TYPE_REGION, &ru, &serr);
    CLCHECK(serr, "sub up");
    // FP32 ALWAYS, independent of activation storage. See q1_gemv4_img_f32: half-precision
    // logits collide on argmax across a 248320-entry vocabulary and produce output that tracks the
    // reference for a token or two then diverges. Costs 1 MB.
    logits_ = scratch((size_t)m_.vocab * 4, "logits");
    xsum_ = scratch((std::max(m_.ffn, qkv_dim_) / 64) * 4, "xsum");
    counter_ = scratch(8, "counter");
    amax_ = scratch(4, "amax");
    amax2_[0] = scratch(4, "amax0");
    amax2_[1] = scratch(4, "amax1");
    // Split-K partials: fp32 regardless of storage_t (they are summed before the narrowing store).
    // Sized for the widest matrix that can take the split path — ~0.8 MB, negligible next to 3.6 GB.
    sk_partial_ = scratch(SK_MAX_SPLITS * SK_MAX_N * 4, "sk_partial");
    // One counter per row-block. Zeroed ONCE here; the fused kernel resets each counter as
    // it consumes it, so it stays clean without a per-call clear (which would reinstate the
    // very dispatch this removes).
    sk_flags_ = scratch(65536 * 4, "sk_flags");
    CLCHECK(clEnqueueFillBuffer(ocl_.queue(), sk_flags_, &kZero, 4, 0, 65536 * 4,
                                0, nullptr, nullptr), "sk_flags zero");

    // 'L' scratch
    // ONE buffer for the whole 'L' input projection, with the four consumers as sub-buffers at the
    // offsets the fused weight was built with. The fused GEMV writes it in a single dispatch; nothing
    // downstream changes, because each consumer still sees a cl_mem holding exactly its own slice.
    if (!lin_.empty()) {
        lin_in_ = scratch((size_t)lin_[0].in_n * es, "lin_in");
        auto sub = [&](int off, int n, const char* tag) {
            cl_buffer_region r{(size_t)off * es, (size_t)n * es};
            cl_int se = CL_SUCCESS;
            cl_mem m = clCreateSubBuffer(lin_in_, CL_MEM_READ_WRITE, CL_BUFFER_CREATE_TYPE_REGION,
                                         &r, &se);
            CLCHECK(se, tag);
            return m;
        };
        qkv_  = sub(lin_[0].in_off[0], qkv_dim_,   "sub.qkv");
        z_    = sub(lin_[0].in_off[1], value_dim_, "sub.z");
        avec_ = sub(lin_[0].in_off[2], nv_,        "sub.a");
        bvec_ = sub(lin_[0].in_off[3], nv_,        "sub.b");
    } else {
        qkv_ = scratch((size_t)qkv_dim_ * es, "qkv");
        z_ = scratch((size_t)value_dim_ * es, "z");
        avec_ = scratch((size_t)nv_ * es, "a");
        bvec_ = scratch((size_t)nv_ * es, "b");
    }
    core_ = scratch((size_t)value_dim_ * 4, "core");   // fp32: feeds the fp32 state
    // 'F' scratch
    if (!full_.empty()) {
        full_in_ = scratch((size_t)full_[0].in_n * es, "full_in");
        auto subf = [&](int off, int n, const char* tag) {
            cl_buffer_region r{(size_t)off * es, (size_t)n * es};
            cl_int se = CL_SUCCESS;
            cl_mem m = clCreateSubBuffer(full_in_, CL_MEM_READ_WRITE, CL_BUFFER_CREATE_TYPE_REGION,
                                         &r, &se);
            CLCHECK(se, tag);
            return m;
        };
        qg_ = subf(full_[0].in_off[0], NH * 2 * D, "sub.qg");
    } else {
        qg_ = scratch((size_t)NH * 2 * D * es, "qg");
    }
    q_ = scratch((size_t)NH * D * es, "q");
    agate_ = scratch((size_t)NH * D * es, "agate");
    if (!full_.empty()) {
        cl_int se = CL_SUCCESS;
        cl_buffer_region rk{(size_t)full_[0].in_off[1] * es, (size_t)KH * D * es};
        k_ = clCreateSubBuffer(full_in_, CL_MEM_READ_WRITE, CL_BUFFER_CREATE_TYPE_REGION, &rk, &se);
        CLCHECK(se, "sub.k");
        cl_buffer_region rv{(size_t)full_[0].in_off[2] * es, (size_t)KH * D * es};
        v_ = clCreateSubBuffer(full_in_, CL_MEM_READ_WRITE, CL_BUFFER_CREATE_TYPE_REGION, &rv, &se);
        CLCHECK(se, "sub.v");
    } else {
        k_ = scratch((size_t)KH * D * es, "k");
        v_ = scratch((size_t)KH * D * es, "v");
    }
    att_ = scratch((size_t)NH * D * es, "att");
    scores_ = scratch((size_t)NH * CTX_CAP * es, "scores");

    // Per-block state. Only the slot matching the block kind is allocated —
    // 'L' blocks have no KV cache and 'F' blocks have no recurrent state, which
    // is the whole memory argument for this architecture.
    rec_.assign(m_.layers, nullptr);
    convs_.assign(m_.layers, nullptr);
    kcache_.assign(m_.layers, nullptr);
    vcache_.assign(m_.layers, nullptr);
    size_t rec_bytes = 0, kv_bytes = 0;
    for (int L = 0; L < m_.layers; ++L) {
        if (blocks_[L].kind == 'L') {
            // fp32, ALWAYS — half underflows/saturates within a few steps.
            const size_t rb = (size_t)nv_ * dk_ * dv_ * 4;
            rec_[L] = scratch(rb, "rec");
            convs_[L] = scratch((size_t)qkv_dim_ * conv_k_ * 4, "conv");
            CLCHECK(clEnqueueFillBuffer(ocl_.queue(), rec_[L], &kZero, 4, 0, rb,
                                        0, nullptr, nullptr), "rec zero");
            CLCHECK(clEnqueueFillBuffer(ocl_.queue(), convs_[L], &kZero, 4, 0,
                                        (size_t)qkv_dim_ * conv_k_ * 4,
                                        0, nullptr, nullptr), "conv zero");
            rec_bytes += rb;
        } else {
            const size_t kb = (size_t)CTX_CAP * KH * D * es;
            kcache_[L] = scratch(kb, "kcache");
            vcache_[L] = scratch(kb, "vcache");
            kv_bytes += 2 * kb;
        }
    }
    fprintf(stderr, "[state] recurrent %.1f MB (%d 'L' blocks) + KV %.1f MB "
                    "(%d 'F' blocks @ ctx %d)\n",
            rec_bytes / 1048576.0, (int)lin_.size(),
            kv_bytes / 1048576.0, (int)full_.size(), CTX_CAP);
}

// Partial RoPE table: rot/2 (cos,sin) pairs per position. Interleaved mRoPE
// collapses to plain RoPE for text-only input (the T/H/W position ids are
// identical), so one table suffices.

/**
 * Rebuild the RoPE tables from explicit (t, h, w) position ids — multimodal RoPE.
 *
 * mrope_section is [11, 11, 10] and sums to rot_/2 = 32, i.e. the 32 rotary PAIRS are split three
 * ways: pairs 0-10 rotate by the TEMPORAL id, 11-21 by the HEIGHT id, 22-31 by the WIDTH id. The
 * frequency index stays global across the sections (inv_freq is computed once for the full dim and
 * then sliced), which is what apply_multimodal_rotary_pos_emb does.
 *
 * For text-only input t == h == w == pos, and this reduces exactly to make_rope_tables(). That is
 * why the port worked for chat while being wrong the moment an image appeared: image patches need
 * h and w to vary per token, and feeding them sequential text positions destroys the 2D layout —
 * the model still receives the image FEATURES but cannot tell where anything is.
 */
void DeviceModel::set_mrope(const std::vector<std::array<int, 3>>& thw) {
    const int half = rot_ / 2;
    int sec[3] = {11, 11, 10};
    if (m_.rope_dim_count > 0 && sec[0] + sec[1] + sec[2] != half) {
        // Fall back to an even split rather than reading past the table if a future model differs.
        sec[0] = sec[1] = half / 3; sec[2] = half - 2 * (half / 3);
    }
    const int n = std::min((int)thw.size(), CTX_CAP);
    std::vector<float> ct((size_t)CTX_CAP * half, 0.0f), st((size_t)CTX_CAP * half, 0.0f);
    for (int pos = 0; pos < CTX_CAP; ++pos) {
        // Positions beyond the supplied ids (i.e. generated tokens) continue as plain text rope.
        const int tt = pos < n ? thw[pos][0] : pos;
        const int hh = pos < n ? thw[pos][1] : pos;
        const int ww = pos < n ? thw[pos][2] : pos;
        const int axis_pos[3] = {tt, hh, ww};
        int i = 0;
        for (int sIdx = 0; sIdx < 3; ++sIdx)
            for (int k = 0; k < sec[sIdx] && i < half; ++k, ++i) {
                const float inv = 1.0f / powf(m_.rope_theta, (float)(2 * i) / (float)rot_);
                const float th = (float)axis_pos[sIdx] * inv;
                ct[(size_t)pos * half + i] = cosf(th);
                st[(size_t)pos * half + i] = sinf(th);
            }
    }
    if (cos_) clReleaseMemObject(cos_);
    if (sin_) clReleaseMemObject(sin_);
    cos_ = upload(ct.data(), ct.size() * 4, "cos_mrope");
    sin_ = upload(st.data(), st.size() * 4, "sin_mrope");
}

void DeviceModel::make_rope_tables() {
    const int half = rot_ / 2;
    std::vector<float> ct((size_t)CTX_CAP * half), st((size_t)CTX_CAP * half);
    for (int pos = 0; pos < CTX_CAP; ++pos)
        for (int i = 0; i < half; ++i) {
            const float inv = 1.0f / powf(m_.rope_theta,
                                          (float)(2 * i) / (float)rot_);
            const float th = (float)pos * inv;
            ct[(size_t)pos * half + i] = cosf(th);
            st[(size_t)pos * half + i] = sinf(th);
        }
    cos_ = upload(ct.data(), ct.size() * 4, "cos");
    sin_ = upload(st.data(), st.size() * 4, "sin");
}
