// OpenCL device model — driver + setup (constructor, forward/prefill loops,
// program build, weight upload, scratch + rope-table allocation). The per-op
// kernel dispatch helpers live in layers/{attention,mlp,embedding,layer_norm}.cpp.
#include "model.h"

DeviceModel::DeviceModel(const Nnb& nnb, OpenCLContext& ocl, const std::string& kdir)
    : nnb_(nnb), m_(nnb.meta), ocl_(ocl) {
    build_programs(kdir);
    upload_weights();
    make_scratch();
    make_rope_tables();
    prof_.on = getenv("BONSAI_PROF") != nullptr;
    prof_.q = ocl_.queue();
    hybrid_ = getenv("BONSAI_HYBRID") != nullptr && ES == 4;  // fp32 out only
    hybrid_readonly_ = getenv("BONSAI_HYBRID_READONLY") != nullptr;
    if (const char* f = getenv("BONSAI_HYBRID_FRAC")) hybrid_frac_ = atof(f);
    if (const char* t = getenv("BONSAI_HYBRID_THREADS")) hybrid_threads_ = atoi(t);
    if (hybrid_) {
        const int maxK = std::max(m_.hidden, m_.ffn);
        hx_.resize(maxK);
        hout_.resize((size_t)(2 * m_.ffn * hybrid_frac_) + 256);
        fprintf(stderr, "[hybrid] CPU+GPU on: frac=%.2f threads=%d\n",
                hybrid_frac_, hybrid_threads_);
    }
}

DeviceModel::~DeviceModel() { prof_.dump(); }

int DeviceModel::forward(int token, int pos, bool want_logits) {
    if (pos >= CTX_CAP) {
        fprintf(stderr, "FATAL: pos %d >= KV cap %d\n", pos, CTX_CAP);
        exit(3);
    }
    const int H = m_.hidden, NH = m_.heads, KH = m_.kv_heads,
              D = m_.head_dim, FF = m_.ffn;
    cl_command_queue q = ocl_.queue();
    int ctr[2] = {pos, pos + 1};
    CLCHECK(clEnqueueWriteBuffer(q, counter_, CL_FALSE, 0, 8, ctr, 0,
                                 nullptr, nullptr), "counter");
    prof_.begin();
    if (token >= 0) run_gather(tok_embd_, x_, token, H);
    else            run_gather_dev(tok_embd_, x_, H);
    prof_.end("gather");
    for (int L = 0; L < m_.layers; ++L) {
        LayerW& w = lw_[L];
        // ---- attention ----
        prof_.begin();
        run_rms(x_, w.attn_norm, xb_, 1, H);
        run_xsum(xb_, H);
        prof_.end("norm+xsum");
        prof_.begin();
        run_gemv(w.wqkv, xb_, qkv_, (NH + 2 * KH) * D, H, 0);
        prof_.end("gemv_qkv");
        prof_.begin();
        run_rms(qv_, w.q_norm, qv_, NH, D);   // Qwen3 QK-norm
        run_rms(kv_, w.k_norm, kv_, KH, D);
        run_rope(qv_, kv_, NH, KH, D);
        prof_.end("qknorm+rope");
        prof_.begin();
        if (kv4_) {
            run_kvq4(kv_, kcache_[L], 1, KH, D, pos);
            run_kvq4(vv_, vcache_[L], 1, KH, D, pos);
            run_scores4(qv_, kcache_[L], NH, KH, D, pos + 1);
            run_softmax(NH, pos + 1);
            run_attnout4(vcache_[L], NH, KH, D);
        } else {
            const size_t kvb = (size_t)KH * D * ES;
            CLCHECK(clEnqueueCopyBuffer(q, kv_, kcache_[L], 0, pos * kvb,
                                        kvb, 0, nullptr, nullptr), "kcpy");
            CLCHECK(clEnqueueCopyBuffer(q, vv_, vcache_[L], 0, pos * kvb,
                                        kvb, 0, nullptr, nullptr), "vcpy");
            run_scores(qv_, kcache_[L], NH, KH, D, pos + 1);
            run_softmax(NH, pos + 1);
            run_attnout(vcache_[L], NH, KH, D);
        }
        prof_.end("attention");
        prof_.begin();
        run_xsum(att_, NH * D);
        run_gemv(w.wo, att_, xb2_, H, NH * D, 0);  // K = attn out dim (may != H)
        run_add(x_, xb2_, H);
        prof_.end("gemv_o");
        // ---- ffn ----
        prof_.begin();
        run_rms(x_, w.ffn_norm, xb_, 1, H);
        run_xsum(xb_, H);
        run_gemv_hybrid(w.wgu, xb_, gu_, 2 * FF, H, 0, w.up_host, FF);
        prof_.end("gemv_gateup");
        prof_.begin();
        run_swiglu(gate_, up_, FF);
        run_xsum(gate_, FF);
        run_gemv_hybrid(w.wd, gate_, xb2_, H, FF, 0, w.down_host, H);
        run_add(x_, xb2_, H);
        prof_.end("gemv_down");
    }
    if (!want_logits) return -1;   // in-order queue: no finish needed
    prof_.begin();
    run_rms(x_, out_norm_, xb_, 1, H);
    run_xsum(xb_, H);
    run_gemv(w_out_, xb_, logits_, m_.vocab, H, 0);
    run_argmax(m_.vocab);
    prof_.end("logits+amax");
    if (getenv("BONSAI_TOPK")) print_topk();
    return 0;   // token retrieved via read_token() (possibly lagged)
}

void DeviceModel::prefill(const int* tokens, int M, int pos0) {
    if (pos0 + M > CTX_CAP) {
        fprintf(stderr, "FATAL: prefill past KV cap\n");
        exit(3);
    }
    const int H = m_.hidden, NH = m_.heads, KH = m_.kv_heads,
              D = m_.head_dim, FF = m_.ffn;
    cl_command_queue q = ocl_.queue();
    int ctr[2] = {pos0, pos0 + M};
    CLCHECK(clEnqueueWriteBuffer(q, counter_, CL_FALSE, 0, 8, ctr, 0,
                                 nullptr, nullptr), "p.counter");
    CLCHECK(clEnqueueWriteBuffer(q, tokbuf_, CL_FALSE, 0, M * 4, tokens,
                                 0, nullptr, nullptr), "p.toks");
    run_gather_b(tok_embd_, xp_, M, H);
    for (int L = 0; L < m_.layers; ++L) {
        LayerW& w = lw_[L];
        run_rms(xp_, w.attn_norm, xbp_, M, H);
        run_xsum_b(xbp_, H, M);
        run_gemv_b(w.wqkv, xbp_, qp_, NH * D, H, (NH + 2 * KH) * D, 0, M);
        run_gemv_b(w.wqkv, xbp_, kp_, KH * D, H, (NH + 2 * KH) * D, NH * D, M);
        run_gemv_b(w.wqkv, xbp_, vp_, KH * D, H, (NH + 2 * KH) * D,
                   (NH + KH) * D, M);
        run_rms(qp_, w.q_norm, qp_, M * NH, D);
        run_rms(kp_, w.k_norm, kp_, M * KH, D);
        run_rope_m(qp_, kp_, NH, KH, D, M);
        const size_t kvrow = (size_t)KH * D * ES;
        CLCHECK(clEnqueueCopyBuffer(q, kp_, kcache_[L], 0, pos0 * kvrow,
                                    M * kvrow, 0, nullptr, nullptr), "p.k");
        CLCHECK(clEnqueueCopyBuffer(q, vp_, vcache_[L], 0, pos0 * kvrow,
                                    M * kvrow, 0, nullptr, nullptr), "p.v");
        run_scores_m(qp_, kcache_[L], NH, KH, D, M, pos0 + M);
        run_softmax_m(NH, M);
        run_attnout_m(vcache_[L], NH, KH, D, M);
        run_xsum_b(attp_, NH * D, M);
        run_gemv_b(w.wo, attp_, xb2p_, H, NH * D, H, 0, M);  // K = attn out dim
        run_add(xp_, xb2p_, M * H);
        run_rms(xp_, w.ffn_norm, xbp_, M, H);
        run_xsum_b(xbp_, H, M);
        run_gemv_b(w.wgu, xbp_, gatep_, FF, H, 2 * FF, 0, M);
        run_gemv_b(w.wgu, xbp_, upp_, FF, H, 2 * FF, FF, M);
        run_swiglu(gatep_, upp_, M * FF);
        run_xsum_b(gatep_, FF, M);
        run_gemv_b(w.wd, gatep_, xb2p_, H, FF, H, 0, M);
        run_add(xp_, xb2p_, M * H);
    }
    // hand the LAST token's hidden state to the M=1 decode state
    CLCHECK(clEnqueueCopyBuffer(q, xp_, x_, (size_t)(M - 1) * H * ES, 0,
                                (size_t)H * ES, 0, nullptr, nullptr), "p.x");
}

int DeviceModel::read_token() {   // blocking read of the latest argmax result
    int next = -1;
    CLCHECK(clEnqueueReadBuffer(ocl_.queue(), amax_, CL_TRUE, 0, 4, &next,
                                0, nullptr, nullptr), "amax read");
    return next;
}

void DeviceModel::print_topk() {
    std::vector<float> lg(m_.vocab);
#ifdef BONSAI_FP16_STORAGE
    std::vector<uint16_t> lh(m_.vocab);
    CLCHECK(clEnqueueReadBuffer(ocl_.queue(), logits_, CL_TRUE, 0,
                                (size_t)m_.vocab * 2, lh.data(), 0,
                                nullptr, nullptr), "logits read");
    for (int i = 0; i < m_.vocab; ++i) lg[i] = half_to_float(lh[i]);
#else
    CLCHECK(clEnqueueReadBuffer(ocl_.queue(), logits_, CL_TRUE, 0,
                                (size_t)m_.vocab * 4, lg.data(), 0,
                                nullptr, nullptr), "logits read");
#endif
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
#ifdef BONSAI_FP16_STORAGE
    kopts += " -DUSE_FP16=1";
#endif
    kopts += " -DMB=8";
    auto B = [&](const char* f) {
        cl_program p = ocl_.build_program_from_file(kdir + "/" + f, kopts);
        if (!p) { fprintf(stderr, "FATAL: build %s\n", f); exit(4); }
        return p;
    };
    cl_int err;
    auto K = [&](cl_program p, const char* name) {
        cl_kernel k = clCreateKernel(p, name, &err);
        CLCHECK(err, name);
        return k;
    };
    cl_program pq = B("q1_gemv.cl");
    k_xsum_ = K(pq, "q1_xsum");
    const char* gv = getenv("BONSAI_GEMV");
    k_gemv_ = K(pq, gv ? gv : "q1_gemv4");  // quad-row: x-reuse across 4 rows
    k_gemv_img_ = K(pq, "q1_gemv4_img");
    cl_kernel k_gemv_lx_ = 0;
    use_img_ = getenv("BONSAI_IMG") != nullptr;
    no_xsum_ = getenv("BONSAI_XOR") != nullptr;
    if (no_xsum_) k_gemv_ = K(pq, "q1_gemv4xor");
    k_gemv7_ = K(pq, "q1_gemv7");
    k_gather_dev_ = K(pq, "q1_row_gather3_dev");
    k_gemv_b_ = K(pq, "q1_gemv_b");
    k_xsum_b_ = K(pq, "q1_xsum_b");
    k_gather_b_ = K(pq, "q1_row_gather_b");
    k_gather_ = K(pq, "q1_row_gather3");
    k_argmax_ = K(pq, "q1_argmax");
    k_rms_ = K(B("rmsnorm.cl"), "rmsnorm_forward");
    k_rope_ = K(B("rope.cl"), "rope_apply_qk");
    cl_program pa = B("attention.cl");
    k_scores_ = K(pa, "gqa_attn_scores");
    k_softmax_ = K(pa, "gqa_softmax");
    k_attnout_ = K(pa, "gqa_attn_out");
    kv4_ = getenv("BONSAI_KV4") != nullptr;
    if (kv4_) {
        k_kvq4_ = K(pa, "kv_quant4");
        k_scores4_ = K(pa, "gqa_attn_scores_kv4");
        k_attnout4_ = K(pa, "gqa_attn_out_kv4");
    }
    k_swiglu_ = K(B("mlp.cl"), "swiglu_inplace");
    k_add_ = K(B("utils.cl"), "element_add");
}

cl_mem DeviceModel::make_bits_image(cl_mem buf, size_t texels) {
    cl_image_format fmt{CL_RGBA, CL_UNSIGNED_INT32};   // uint4 = one unit
    cl_image_desc d{};
    d.image_type = CL_MEM_OBJECT_IMAGE1D_BUFFER;
    d.image_width = texels;
    d.buffer = buf;
    cl_int err;
    cl_mem img = clCreateImage(ocl_.context(), CL_MEM_READ_ONLY, &fmt, &d,
                               nullptr, &err);
    if (err != CL_SUCCESS) { fprintf(stderr, "FATAL: bits image err %d\n", err); exit(4); }
    return img;
}

cl_mem DeviceModel::upload(const void* p, size_t n, const char* what) {
    cl_int err;
    cl_mem m = clCreateBuffer(ocl_.context(),
                              CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, n,
                              (void*)p, &err);
    CLCHECK(err, what);
    return m;
}

cl_mem DeviceModel::scratch(size_t n, const char* what) {
    cl_int err;
    cl_mem m = clCreateBuffer(ocl_.context(), CL_MEM_READ_WRITE, n, nullptr, &err);
    CLCHECK(err, what);
    return m;
}

DeviceModel::QW DeviceModel::upload_q1_fused(const std::vector<std::string>& names) {
    size_t U = 0, Ntot = 0;
    for (const auto& n : names) {
        const Tensor& t = nnb_.get(n);
        U = (size_t)t.ne(0) / 128;
        Ntot += (size_t)t.ne(1);
    }
    std::vector<uint8_t> bits(U * Ntot * 16);
    std::vector<uint8_t> scales(U * Ntot * 2);
    size_t row_base = 0;
    for (const auto& n : names) {
        const Tensor& t = nnb_.get(n);
        const size_t N = (size_t)t.ne(1);
        const uint8_t* p = t.data;
        for (size_t row = 0; row < N; ++row)
            for (size_t u = 0; u < U; ++u) {
                const uint8_t* unit = p + (row * U + u) * 18;
                const size_t dst = u * Ntot + row_base + row;
                memcpy(&scales[dst * 2], unit, 2);
                memcpy(&bits[dst * 16], unit + 2, 16);
            }
        row_base += N;
    }
    QW w;
    w.bits = upload(bits.data(), bits.size(), "fused.bits");
    w.scales = upload(scales.data(), scales.size(), "fused.scales");
    w.bits_img = make_bits_image(w.bits, bits.size() / 16);
    return w;
}

DeviceModel::QW DeviceModel::upload_q1(const std::string& n) {
    // K-MAJOR TRANSPOSE at upload: bits_t[u][row], scales_t[u][row] —
    // adjacent GEMV lanes (rows) read adjacent words: coalesced.
    const Tensor& t = nnb_.get(n);
    const size_t U = (size_t)t.ne(0) / 128, N = (size_t)t.ne(1);
    std::vector<uint8_t> bits(U * N * 16);
    std::vector<uint8_t> scales(U * N * 2);
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
    w.bits_img = make_bits_image(w.bits, bits.size() / 16);
    return w;
}

void DeviceModel::upload_weights() {
    auto Q = [&](const std::string& n) { return upload_q1(n); };
    auto F = [&](const std::string& n) {
        const Tensor& t = nnb_.get(n);
#ifdef BONSAI_FP16_STORAGE
        const size_t cnt = t.nbytes / 4;
        std::vector<uint16_t> h(cnt);
        const float* f = (const float*)t.data;
        for (size_t i = 0; i < cnt; ++i) h[i] = float_to_half(f[i]);
        return upload(h.data(), cnt * 2, n.c_str());
#else
        return upload(t.data, t.nbytes, n.c_str());
#endif
    };
    for (int L = 0; L < m_.layers; ++L) {
        const std::string p = "blk." + std::to_string(L) + ".";
        LayerW w;
        w.wqkv = upload_q1_fused({p + "attn_q.weight", p + "attn_k.weight",
                                  p + "attn_v.weight"});
        w.wo = Q(p + "attn_output.weight");
        w.wgu = upload_q1_fused({p + "ffn_gate.weight", p + "ffn_up.weight"});
        w.wd = Q(p + "ffn_down.weight");
        w.attn_norm = F(p + "attn_norm.weight");
        w.ffn_norm = F(p + "ffn_norm.weight");
        w.q_norm = F(p + "attn_q_norm.weight");
        w.k_norm = F(p + "attn_k_norm.weight");
        w.up_host = nnb_.get(p + "ffn_up.weight").data;      // hybrid CPU slice
        w.down_host = nnb_.get(p + "ffn_down.weight").data;  // hybrid CPU slice
        lw_.push_back(w);
    }
    tok_embd_ = Q("token_embd.weight");
    // Tied embeddings (e.g. Bonsai-4B): no output.weight → logits reuse the
    // token_embd matrix (same [hidden, vocab] Q1 layout).
    w_out_ = nnb_.has("output.weight") ? Q("output.weight") : tok_embd_;
    out_norm_ = F("output_norm.weight");
}

void DeviceModel::make_scratch() {
    const int H = m_.hidden, D = m_.head_dim;
    const size_t es = ES;
    x_ = scratch(H * es, "x");
    xb_ = scratch(H * es, "xb");
    xb2_ = scratch(H * es, "xb2");
    qkv_ = scratch((m_.heads + 2 * m_.kv_heads) * D * es, "qkv");
    cl_int serr;
    cl_buffer_region rq{0, (size_t)m_.heads * D * es};
    cl_buffer_region rk{(size_t)m_.heads * D * es, (size_t)m_.kv_heads * D * es};
    cl_buffer_region rv{(size_t)(m_.heads + m_.kv_heads) * D * es,
                        (size_t)m_.kv_heads * D * es};
    qv_ = clCreateSubBuffer(qkv_, 0, CL_BUFFER_CREATE_TYPE_REGION, &rq, &serr);
    CLCHECK(serr, "sub q");
    kv_ = clCreateSubBuffer(qkv_, 0, CL_BUFFER_CREATE_TYPE_REGION, &rk, &serr);
    CLCHECK(serr, "sub k");
    vv_ = clCreateSubBuffer(qkv_, 0, CL_BUFFER_CREATE_TYPE_REGION, &rv, &serr);
    CLCHECK(serr, "sub v");
    att_ = scratch(m_.heads * D * es, "att");
    scores_ = scratch((size_t)m_.heads * M_MAX * CTX_CAP * es, "scores");
    gu_ = scratch((size_t)2 * m_.ffn * es, "gu");
    cl_buffer_region rg{0, (size_t)m_.ffn * es};
    cl_buffer_region ru{(size_t)m_.ffn * es, (size_t)m_.ffn * es};
    gate_ = clCreateSubBuffer(gu_, 0, CL_BUFFER_CREATE_TYPE_REGION, &rg, &serr);
    CLCHECK(serr, "sub gate");
    up_ = clCreateSubBuffer(gu_, 0, CL_BUFFER_CREATE_TYPE_REGION, &ru, &serr);
    CLCHECK(serr, "sub up");
    logits_ = scratch((size_t)m_.vocab * es, "logits");
    xsum_ = scratch((m_.ffn / 64) * 4, "xsum");
    counter_ = scratch(8, "counter");
    amax_ = scratch(4, "amax");
    // batched-prefill scratch (M_MAX tokens per chunk)
    xp_ = scratch((size_t)M_MAX * H * es, "xp");
    xbp_ = scratch((size_t)M_MAX * H * es, "xbp");
    xb2p_ = scratch((size_t)M_MAX * H * es, "xb2p");
    qp_ = scratch((size_t)M_MAX * m_.heads * D * es, "qp");
    kp_ = scratch((size_t)M_MAX * m_.kv_heads * D * es, "kp");
    vp_ = scratch((size_t)M_MAX * m_.kv_heads * D * es, "vp");
    attp_ = scratch((size_t)M_MAX * m_.heads * D * es, "attp");
    gatep_ = scratch((size_t)M_MAX * m_.ffn * es, "gatep");
    upp_ = scratch((size_t)M_MAX * m_.ffn * es, "upp");
    xsump_ = scratch((size_t)M_MAX * (m_.ffn / 64) * 4, "xsump");
    tokbuf_ = scratch(M_MAX * 4, "tokbuf");
    // 4-bit KV: per (token,kv-head) group = 2B scale + head_dim/2 nibbles.
    const bool kv4 = getenv("BONSAI_KV4") != nullptr;
    const size_t grp = kv4 ? (2 + (size_t)m_.head_dim / 2) : (size_t)m_.head_dim * es;
    const size_t kvb = (size_t)CTX_CAP * m_.kv_heads * grp;
    for (int L = 0; L < m_.layers; ++L) {
        kcache_.push_back(scratch(kvb, "kcache"));
        vcache_.push_back(scratch(kvb, "vcache"));
    }
}

void DeviceModel::make_rope_tables() {
    // Host-side YaRN math (identical to the token-exact P1 RopeYarn).
    RopeYarn ry;
    ry.init(m_.rope_theta, m_.yarn_factor, m_.head_dim, m_.yarn_orig_ctx);
    const int D = m_.head_dim, half = D / 2;
    std::vector<float> ct((size_t)CTX_CAP * D), st((size_t)CTX_CAP * D);
    for (int pos = 0; pos < CTX_CAP; ++pos)
        for (int i = 0; i < half; ++i) {
            float c, s;
            ry.at(pos, i, &c, &s);
            ct[(size_t)pos * D + i] = c;
            st[(size_t)pos * D + i] = s;
            ct[(size_t)pos * D + i + half] = c;   // kernel reads i<half only
            st[(size_t)pos * D + i + half] = s;
        }
#ifdef BONSAI_FP16_STORAGE
    std::vector<uint16_t> ch(ct.size()), sh(st.size());
    for (size_t i = 0; i < ct.size(); ++i) { ch[i] = float_to_half(ct[i]); sh[i] = float_to_half(st[i]); }
    cos_ = upload(ch.data(), ch.size() * 2, "cos");
    sin_ = upload(sh.data(), sh.size() * 2, "sin");
#else
    cos_ = upload(ct.data(), ct.size() * 4, "cos");
    sin_ = upload(st.data(), st.size() * 4, "sin");
#endif
}
