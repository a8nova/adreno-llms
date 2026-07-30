// 'L' block — Gated DeltaNet linear attention.
//
// Mirrors Model::linear_attn() in reference_model.h:
//   qkv   = in_proj_qkv(x)                 [2*key_dim + value_dim]
//   z     = in_proj_z(x)                   [value_dim]   (GGUF: "attn_gate")
//   b, a  = in_proj_b(x), in_proj_a(x)     [n_v_heads]
//   qkv   = silu(causal_conv1d(qkv))       ALL qkv channels
//   core  = gated_delta_rule(q, k, v, a, b)
//   core  = gated_rmsnorm(core, z)         norm BEFORE gate
//   out   = out_proj(core)
//
// No KV cache: the recurrent state IS the history.
#include "model.h"

void DeviceModel::linear_block(int L, const BlockW& bw, const LinearLayerW& w) {
    const int H = m_.hidden;
    (void)bw;
    // xb_ already holds rmsnorm(x); xsum_ already computed by the caller.
    // ONE dispatch for qkv + z + alpha + beta. They all read this x, and their outputs are the four
    // slices of lin_in_ that qkv_/z_/avec_/bvec_ alias — so the three extra dispatches bought nothing
    // but launch cost, and two of them were N=48 on a 12-CU GPU.
    run_gemv(w.win, xb_, lin_in_, w.in_n, H);
    // BONSAI_DUMP_QKV: first use of a STAGED sub-buffer in the forward pass.
    // If this matches numpy, streaming+GEMV is fine and the divergence is later.
    if (const char* dd = getenv("BONSAI_DUMP_QKV")) {
        if (L == 0) {
            std::vector<uint8_t> raw((size_t)qkv_dim_ * ES);
            clFinish(ocl_.queue());
            clEnqueueReadBuffer(ocl_.queue(), qkv_, CL_TRUE, 0, raw.size(),
                                raw.data(), 0, nullptr, nullptr);
            char pth[512]; snprintf(pth, sizeof(pth), "%s/qkv0.f32.bin", dd);
            if (FILE* f = fopen(pth, "wb")) { fwrite(raw.data(),1,raw.size(),f); fclose(f); }
            // also dump the normed input so numpy can reproduce exactly
            std::vector<uint8_t> xr((size_t)H * ES);
            clEnqueueReadBuffer(ocl_.queue(), xb_, CL_TRUE, 0, xr.size(),
                                xr.data(), 0, nullptr, nullptr);
            snprintf(pth, sizeof(pth), "%s/xb0.f32.bin", dd);
            if (FILE* f = fopen(pth, "wb")) { fwrite(xr.data(),1,xr.size(),f); fclose(f); }
        }
    }
    // z / alpha / beta already computed by the fused projection above.
    auto DBG = [&](const char* tag, cl_mem b, size_t nbytes) {
        const char* dd = getenv("BONSAI_DUMP_QKV");
        if (!dd || L != 0) return;
        std::vector<uint8_t> r(nbytes);
        clFinish(ocl_.queue());
        clEnqueueReadBuffer(ocl_.queue(), b, CL_TRUE, 0, nbytes, r.data(), 0, nullptr, nullptr);
        char pth[512]; snprintf(pth, sizeof(pth), "%s/%s.f32.bin", dd, tag);
        if (FILE* f = fopen(pth, "wb")) { fwrite(r.data(),1,r.size(),f); fclose(f); }
    };
    DBG("z0",     z_,     (size_t)value_dim_ * ES);
    DBG("beta0",  bvec_,  (size_t)nv_ * ES);
    DBG("alpha0", avec_,  (size_t)nv_ * ES);
    run_conv1d(qkv_, convs_[L], w.conv_w, qkv_dim_, conv_k_);
    DBG("conv0",  qkv_,   (size_t)qkv_dim_ * ES);
    run_delta_net(rec_[L], qkv_, avec_, bvec_, w.A_log, w.dt_b, core_);
    DBG("core0",  core_,  (size_t)value_dim_ * 4);
    run_norm_gate(core_, z_, w.ssm_norm);
    DBG("gated0", core_,  (size_t)value_dim_ * 4);

    // core_ is fp32; the GEMV x-sum pre-pass expects the activation layout, so
    // recompute it over value_dim before the output projection.
    run_xsum(core_, value_dim_);
    run_gemv(w.wout, core_, mix_, H, value_dim_);
}

void DeviceModel::run_conv1d(cl_mem qkv, cl_mem state, cl_mem w,
                             int channels, int conv_k) {
    int a = 0;
    arg(k_dn_conv_, a++, sizeof(cl_mem), &qkv, "cv.x");
    arg(k_dn_conv_, a++, sizeof(cl_mem), &state, "cv.s");
    arg(k_dn_conv_, a++, sizeof(cl_mem), &w, "cv.w");
    arg(k_dn_conv_, a++, sizeof(int), &channels, "cv.c");
    arg(k_dn_conv_, a++, sizeof(int), &conv_k, "cv.k");
    run1(k_dn_conv_, (size_t)channels, 0, "conv1d");
}

// One workgroup per VALUE head, d_v work-items per group — work-item j owns
// column j of the [d_k x d_v] state. See kernels/delta_net.cl.
void DeviceModel::run_delta_net(cl_mem S, cl_mem qkv, cl_mem a_in, cl_mem b_in,
                                cl_mem A_log, cl_mem dt_b, cl_mem out) {
    int a = 0;
    arg(k_dn_step_, a++, sizeof(cl_mem), &S, "dn.S");
    arg(k_dn_step_, a++, sizeof(cl_mem), &qkv, "dn.qkv");
    arg(k_dn_step_, a++, sizeof(cl_mem), &a_in, "dn.a");
    arg(k_dn_step_, a++, sizeof(cl_mem), &b_in, "dn.b");
    arg(k_dn_step_, a++, sizeof(cl_mem), &A_log, "dn.A");
    arg(k_dn_step_, a++, sizeof(cl_mem), &dt_b, "dn.dt");
    arg(k_dn_step_, a++, sizeof(cl_mem), &out, "dn.o");
    arg(k_dn_step_, a++, sizeof(int), &nv_, "dn.nv");
    arg(k_dn_step_, a++, sizeof(int), &nk_, "dn.nk");
    arg(k_dn_step_, a++, sizeof(int), &dk_, "dn.dk");
    arg(k_dn_step_, a++, sizeof(int), &dv_, "dn.dv");
    run1(k_dn_step_, (size_t)nv_ * dv_, (size_t)dv_, "delta_net");
}

// norm BEFORE gate — mamba2's kernel normalises the gated product instead,
// which is a different function. See PORT_CHECKLIST.md §C3.
void DeviceModel::run_norm_gate(cl_mem x, cl_mem gate, cl_mem w) {
    int a = 0;
    float eps = m_.rms_eps;
    arg(k_dn_norm_, a++, sizeof(cl_mem), &x, "ng.x");
    arg(k_dn_norm_, a++, sizeof(cl_mem), &gate, "ng.g");
    arg(k_dn_norm_, a++, sizeof(cl_mem), &w, "ng.w");
    arg(k_dn_norm_, a++, sizeof(int), &nv_, "ng.nv");
    arg(k_dn_norm_, a++, sizeof(int), &dv_, "ng.dv");
    arg(k_dn_norm_, a++, sizeof(float), &eps, "ng.e");
    run1(k_dn_norm_, (size_t)nv_ * dv_, (size_t)dv_, "norm_gate");
}
