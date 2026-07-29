// 'F' block — gated full attention.
//
// Mirrors Model::full_attn() in reference_model.h:
//   qg      = q_proj(x)                    [n_heads * 2 * head_dim]
//   q, gate = split_per_head(qg)           q and gate INTERLEAVE PER HEAD
//   k, v    = k_proj(x), v_proj(x)
//   q, k    = head_rmsnorm(q), head_rmsnorm(k)
//   q, k    = partial_rope(q), partial_rope(k)     first `rot` dims only
//   att     = GQA(q, kcache, vcache)               head_dim 256
//   att    *= sigmoid(gate)                        SIGMOID (the SSM gate is SILU)
//   out     = o_proj(att)
//
// The GQA core (gqa_attn_scores / gqa_softmax / gqa_attn_out) is reused
// unchanged from the dense port — those kernels already take head_dim as a
// runtime arg.
#include "model.h"

void DeviceModel::full_block(int L, const BlockW& bw, const FullLayerW& w, int pos) {
    const int H = m_.hidden, NH = m_.heads, KH = m_.kv_heads, D = m_.head_dim;
    (void)bw;
    cl_command_queue q = ocl_.queue();
    // BLOCKING (CL_TRUE) is required, not an optimisation choice. `ctr` is a
    // stack local; with a NON-blocking write the driver may read it after this
    // function returns and the frame has been reused, feeding garbage seq_k
    // into the attention kernels. That produced a wrong attention output in
    // every 'F' block while every 'L' block stayed correct (only full_block
    // writes counter_), and it vanished whenever a debug clFinish happened to
    // land in between — the classic "adding a print fixes it" signature.
    int ctr[2] = {pos, pos + 1};
    CLCHECK(clEnqueueWriteBuffer(q, counter_, CL_TRUE, 0, 8, ctr, 0,
                                 nullptr, nullptr), "counter");

    // q_proj emits NH * 2 * D; q and the gate interleave per head.
    auto DBG = [&](const char* tag, cl_mem b, size_t nb) {
        const char* dd = getenv("BONSAI_DUMP_QKV");
        if (!dd || L != 3) return;
        std::vector<uint8_t> r(nb);
        clFinish(ocl_.queue());
        clEnqueueReadBuffer(ocl_.queue(), b, CL_TRUE, 0, nb, r.data(), 0, nullptr, nullptr);
        char pth[512]; snprintf(pth, sizeof(pth), "%s/f_%s.f32.bin", dd, tag);
        if (FILE* f = fopen(pth, "wb")) { fwrite(r.data(),1,r.size(),f); fclose(f); }
    };
    // ONE dispatch for q + k + v — same x, and qg_/k_/v_ are slices of full_in_.
    run_gemv(w.win, xb_, full_in_, w.in_n, H);
    DBG("xb", xb_, (size_t)H * ES);
    DBG("qg", qg_, (size_t)NH * 2 * D * ES);
    run_split_qg(qg_, q_, agate_, NH, D);
    DBG("qsplit", q_, (size_t)NH * D * ES);
    DBG("gate", agate_, (size_t)NH * D * ES);
    // k and v already computed by the fused projection above.

    run_rms(q_, w.q_norm, q_, NH, D);
    run_rms(k_, w.k_norm, k_, KH, D);
    DBG("qnorm", q_, (size_t)NH * D * ES);
    DBG("knorm", k_, (size_t)KH * D * ES);
    DBG("v", v_, (size_t)KH * D * ES);
    run_rope_partial(q_, NH, D, pos);
    run_rope_partial(k_, KH, D, pos);

    const size_t kvb = (size_t)KH * D * ES;
    CLCHECK(clEnqueueCopyBuffer(q, k_, kcache_[L], 0, pos * kvb, kvb, 0,
                                nullptr, nullptr), "kcpy");
    CLCHECK(clEnqueueCopyBuffer(q, v_, vcache_[L], 0, pos * kvb, kvb, 0,
                                nullptr, nullptr), "vcpy");

    run_scores(q_, kcache_[L], NH, KH, D, pos + 1);
    run_softmax(NH, pos + 1);
    run_attnout(vcache_[L], NH, KH, D);

    DBG("att", att_, (size_t)NH * D * ES);
    run_apply_gate(att_, agate_, NH * D);
    DBG("attg", att_, (size_t)NH * D * ES);
    run_xsum(att_, NH * D);
    DBG("xsum", xsum_, (size_t)(NH * D / 64) * 4);
    run_gemv(w.wo, att_, mix_, H, NH * D);
    DBG("mix", mix_, (size_t)H * ES);
}

void DeviceModel::run_split_qg(cl_mem qg, cl_mem qout, cl_mem gate, int NH, int D) {
    int a = 0;
    arg(k_split_qg_, a++, sizeof(cl_mem), &qg, "sq.qg");
    arg(k_split_qg_, a++, sizeof(cl_mem), &qout, "sq.q");
    arg(k_split_qg_, a++, sizeof(cl_mem), &gate, "sq.g");
    arg(k_split_qg_, a++, sizeof(int), &NH, "sq.h");
    arg(k_split_qg_, a++, sizeof(int), &D, "sq.d");
    run1(k_split_qg_, (size_t)NH * D, 0, "split_qg");
}

// Rotate only the first rot_ of D dims, NeoX pairing (i, i + rot_/2).
void DeviceModel::run_rope_partial(cl_mem x, int n_heads, int D, int pos) {
    int a = 0;
    arg(k_rope_p_, a++, sizeof(cl_mem), &x, "rp.x");
    arg(k_rope_p_, a++, sizeof(cl_mem), &cos_, "rp.c");
    arg(k_rope_p_, a++, sizeof(cl_mem), &sin_, "rp.s");
    arg(k_rope_p_, a++, sizeof(int), &n_heads, "rp.h");
    arg(k_rope_p_, a++, sizeof(int), &D, "rp.d");
    arg(k_rope_p_, a++, sizeof(int), &rot_, "rp.r");
    arg(k_rope_p_, a++, sizeof(int), &pos, "rp.p");
    run1(k_rope_p_, (size_t)n_heads * (rot_ / 2), 0, "rope_partial");
}

void DeviceModel::run_apply_gate(cl_mem att, cl_mem gate, int n) {
    int a = 0;
    arg(k_apply_gate_, a++, sizeof(cl_mem), &att, "ag.a");
    arg(k_apply_gate_, a++, sizeof(cl_mem), &gate, "ag.g");
    arg(k_apply_gate_, a++, sizeof(int), &n, "ag.n");
    run1(k_apply_gate_, (size_t)n, 0, "apply_gate");
}

void DeviceModel::run_scores(cl_mem qb, cl_mem kc, int NH, int KH, int D, int seq_k) {
    int a = 0, seq_q = 1;
    float scale = 1.0f / sqrtf((float)D);
    arg(k_scores_, a++, sizeof(cl_mem), &qb, "sc.q");
    arg(k_scores_, a++, sizeof(cl_mem), &kc, "sc.k");
    arg(k_scores_, a++, sizeof(cl_mem), &scores_, "sc.s");
    arg(k_scores_, a++, sizeof(int), &seq_q, "sc.sq");
    arg(k_scores_, a++, sizeof(cl_mem), &counter_, "sc.ctr");
    arg(k_scores_, a++, sizeof(int), &NH, "sc.qh");
    arg(k_scores_, a++, sizeof(int), &KH, "sc.kh");
    arg(k_scores_, a++, sizeof(int), &D, "sc.d");
    arg(k_scores_, a++, sizeof(float), &scale, "sc.sc");
    size_t g[3] = {(size_t)NH, 1, (size_t)seq_k};
    CLCHECK(clEnqueueNDRangeKernel(ocl_.queue(), k_scores_, 3, nullptr, g,
                                   nullptr, 0, nullptr, nullptr), "scores");
}

void DeviceModel::run_softmax(int NH, int seq_k) {
    (void)seq_k;
    int a = 0, seq_q = 1, rows = NH;
    arg(k_softmax_, a++, sizeof(cl_mem), &scores_, "sm.s");
    arg(k_softmax_, a++, sizeof(int), &seq_q, "sm.sq");
    arg(k_softmax_, a++, sizeof(cl_mem), &counter_, "sm.ctr");
    arg(k_softmax_, a++, sizeof(int), &rows, "sm.r");
    run1(k_softmax_, (size_t)NH, 0, "softmax");
}

void DeviceModel::run_attnout(cl_mem vc, int NH, int KH, int D) {
    int a = 0, seq_q = 1;
    arg(k_attnout_, a++, sizeof(cl_mem), &scores_, "ao.s");
    arg(k_attnout_, a++, sizeof(cl_mem), &vc, "ao.v");
    arg(k_attnout_, a++, sizeof(cl_mem), &att_, "ao.o");
    arg(k_attnout_, a++, sizeof(int), &seq_q, "ao.sq");
    arg(k_attnout_, a++, sizeof(cl_mem), &counter_, "ao.ctr");
    arg(k_attnout_, a++, sizeof(int), &NH, "ao.qh");
    arg(k_attnout_, a++, sizeof(int), &KH, "ao.kh");
    arg(k_attnout_, a++, sizeof(int), &D, "ao.d");
    size_t g[3] = {(size_t)NH, 1, (size_t)(D / 4)};
    CLCHECK(clEnqueueNDRangeKernel(ocl_.queue(), k_attnout_, 3, nullptr, g,
                                   nullptr, 0, nullptr, nullptr), "attnout");
}
