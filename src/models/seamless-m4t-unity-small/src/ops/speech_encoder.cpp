// Stage 3: w2v-BERT Conformer speech encoder + adaptor. fbank[N,80] -> [Tout,768].
#include "pipeline.h"
#include "opencl_context.h"
#include <CL/cl.h>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <string>

void dump_floats_dbg(const std::string& path, const std::vector<float>& v);

std::vector<float> Pipeline::encoder(const std::vector<float>& fb, int nframes, int& Tout) {
    const bool DBG = std::getenv("NNOPT_DUMP_ENC") != nullptr;
    const bool PROF = std::getenv("NNOPT_PROF_ENC") != nullptr;
    double pf_ffn = 0, pf_attn = 0, pf_conv = 0;
    auto sync_ms = [&]() { clFinish(ops_.cl().queue()); return std::chrono::steady_clock::now(); };
    auto add_ms = [&](double& slot, std::chrono::steady_clock::time_point t0) {
        if (!PROF) return; clFinish(ops_.cl().queue());
        slot += std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count();
    };
    const int ET = nframes / 2, T = ET;
    const std::string WE = "model.encoder.w2v_encoder.w2v_model.";
    const std::string AD = "model.encoder.adaptor.";
    const std::string ML = AD + "layers.0.";
    auto W = [&](const std::string& k) { return ops_.weight(k); };

    // front-end: pair-stack [nframes,80] -> [T,160] (contiguous), LN(160), proj 160->768.
    Tensor fbT = ops_.upload(fb);
    Tensor stacked{fbT.buf, T * 160};
    Tensor x = ops_.layernorm(stacked, T, 160, W(WE + "layer_norm.weight"), W(WE + "layer_norm.bias"));
    Tensor proj = ops_.linear(x, T, 160, W(WE + "post_extract_proj.weight"), W(WE + "post_extract_proj.bias"), Dm);

    // rel_pos w2v-BERT conformer: NO conv-based positional embedding. The scripted
    // ConformerEncoder.extract_features (pos_enc_type=="rel_pos", layer_norm_first=True)
    // feeds post_extract_proj straight into layer 0 — relative-position information
    // enters each layer through the sinusoid table `pe` (relpos_pos_emb) and the
    // per-layer pos_bias_u/v, not via an additive pos_conv. Adding a pos_conv here
    // was a ~6% structural error that the per-layer LayerNorms only partly absorbed
    // (tolerable on the 1s fixture, but it flipped decoded tokens on longer inputs).
    Tensor h = ops_.clone(proj);
    if (DBG) dump_floats_dbg("enc_conf_input.bin", ops_.download(h));

    Tensor pe = ops_.relpos_pos_emb(2 * T - 1, Dm, T);
    const int L = 2 * T - 1;

    auto enc_ffn = [&](const Tensor& in, int Tn, const std::string& pre) {
        Tensor a = ops_.layernorm(in, Tn, Dm, W(pre + "layer_norm.weight"), W(pre + "layer_norm.bias"));
        a = ops_.linear(a, Tn, Dm, W(pre + "w_1.weight"), W(pre + "w_1.bias"), 4096);
        ops_.act(a, ACT_SWISH);
        return ops_.linear(a, Tn, 4096, W(pre + "w_2.weight"), W(pre + "w_2.bias"), Dm);
    };
    auto conv_module = [&](const Tensor& in, int Tn, const std::string& pre) {
        Tensor ln = ops_.layernorm(in, Tn, Dm, W(pre + "layer_norm.weight"), W(pre + "layer_norm.bias"));
        Tensor a = ops_.linear(ln, Tn, Dm, W(pre + "pointwise_conv1.weight"), nullptr, 1536);
        Tensor gl = ops_.glu_tc(a, Tn, Dm);
        Tensor dc = ops_.conv1d_tc(gl, W(pre + "depthwise_conv.weight"), nullptr, Tn, Dm, Dm, Tn, 31, 1, 15, 1, Dm);
        Tensor bn = ops_.batchnorm_tc(dc, Tn, Dm, W(pre + "batch_norm.running_mean"), W(pre + "batch_norm.running_var"),
                                      W(pre + "batch_norm.weight"), W(pre + "batch_norm.bias"), 1e-5f);
        ops_.act(bn, ACT_SWISH);
        return ops_.linear(bn, Tn, Dm, W(pre + "pointwise_conv2.weight"), nullptr, Dm);
    };
    auto relpos_attn = [&](const Tensor& an, const std::string& pre) {
        Tensor q = ops_.linear(an, T, Dm, W(pre + "linear_q.weight"), W(pre + "linear_q.bias"), Dm);
        Tensor k = ops_.linear(an, T, Dm, W(pre + "linear_k.weight"), W(pre + "linear_k.bias"), Dm);
        Tensor v = ops_.linear(an, T, Dm, W(pre + "linear_v.weight"), W(pre + "linear_v.bias"), Dm);
        Tensor p = ops_.linear(pe, L, Dm, W(pre + "linear_pos.weight"), nullptr, Dm);
        Tensor ctx = ops_.relpos_attention(q, k, v, p, W(pre + "pos_bias_u"), W(pre + "pos_bias_v"), T, L, H, Dk);
        return ops_.linear(ctx, T, Dm, W(pre + "linear_out.weight"), W(pre + "linear_out.bias"), Dm);
    };
    auto std_attn = [&](const Tensor& an, int Tn, const std::string& pre) {
        Tensor q = ops_.linear(an, Tn, Dm, W(pre + "q_proj.weight"), W(pre + "q_proj.bias"), Dm);
        Tensor k = ops_.linear(an, Tn, Dm, W(pre + "k_proj.weight"), W(pre + "k_proj.bias"), Dm);
        Tensor v = ops_.linear(an, Tn, Dm, W(pre + "v_proj.weight"), W(pre + "v_proj.bias"), Dm);
        Tensor ctx = ops_.attention(q, k, v, Tn, Tn, H, Dk, false);
        return ops_.linear(ctx, Tn, Dm, W(pre + "out_proj.weight"), W(pre + "out_proj.bias"), Dm);
    };

    for (int Lyr = 0; Lyr < 8; ++Lyr) {
        std::string pre = WE + "encoder.layers." + std::to_string(Lyr) + ".";
        auto tp = PROF ? sync_ms() : std::chrono::steady_clock::now();
        Tensor f1 = enc_ffn(h, T, pre + "ffn1.");
        ops_.axpy(h, f1, 0.5f);
        add_ms(pf_ffn, tp); tp = std::chrono::steady_clock::now();
        Tensor an = ops_.layernorm(h, T, Dm, W(pre + "self_attn_layer_norm.weight"), W(pre + "self_attn_layer_norm.bias"));
        an = relpos_attn(an, pre + "self_attn.");
        ops_.axpy(h, an, 1.0f);
        add_ms(pf_attn, tp); tp = std::chrono::steady_clock::now();
        Tensor cv = conv_module(h, T, pre + "conv_module.");
        ops_.axpy(h, cv, 1.0f);
        add_ms(pf_conv, tp); tp = std::chrono::steady_clock::now();
        Tensor f2 = enc_ffn(h, T, pre + "ffn2.");
        ops_.axpy(h, f2, 0.5f);
        add_ms(pf_ffn, tp);
        h = ops_.layernorm(h, T, Dm, W(pre + "final_layer_norm.weight"), W(pre + "final_layer_norm.bias"));
        if (DBG) dump_floats_dbg("enc_conflayer" + std::to_string(Lyr) + ".bin", ops_.download(h));
    }
    h = ops_.layernorm(h, T, Dm, W(WE + "encoder.layer_norm.weight"), W(WE + "encoder.layer_norm.bias"));
    if (DBG) dump_floats_dbg("enc_conformer_out.bin", ops_.download(h));
    if (PROF) fprintf(stderr, "PROF_ENC(ms): ffn=%.0f attn=%.0f conv=%.0f (conformer only)\n", pf_ffn, pf_attn, pf_conv);

    // adaptor 49 -> 7
    Tensor pj = ops_.linear(h, T, Dm, W(AD + "proj.0.weight"), W(AD + "proj.0.bias"), 3072);
    ops_.act(pj, ACT_RELU);
    pj = ops_.linear(pj, T, 3072, W(AD + "proj.2.weight"), W(AD + "proj.2.bias"), Dm);
    ops_.axpy(h, pj, 0.5f);
    Tensor cl = ops_.layernorm(h, T, Dm, W(ML + "conv_ln.weight"), W(ML + "conv_ln.bias"));
    int T7 = (T + 2 * 4 - 8) / 8 + 1;
    Tensor cp = ops_.conv1d_tc(cl, W(ML + "conv_pool.1.weight"), W(ML + "conv_pool.1.bias"), T, Dm, 1536, T7, 8, 8, 4, 1, 1);
    Tensor g = ops_.glu_tc(cp, T7, Dm);
    Tensor f1 = enc_ffn(g, T7, ML + "ffn1.");
    ops_.axpy(g, f1, 0.5f);
    Tensor an = ops_.layernorm(g, T7, Dm, W(ML + "self_attn_layer_norm.weight"), W(ML + "self_attn_layer_norm.bias"));
    an = std_attn(an, T7, ML + "self_attn.");
    ops_.axpy(g, an, 1.0f);
    Tensor cv = conv_module(g, T7, ML + "conv_module.");
    ops_.axpy(g, cv, 1.0f);
    Tensor f2 = enc_ffn(g, T7, ML + "ffn2.");
    ops_.axpy(g, f2, 0.5f);
    g = ops_.layernorm(g, T7, Dm, W(ML + "final_layer_norm.weight"), W(ML + "final_layer_norm.bias"));
    g = ops_.layernorm(g, T7, Dm, W(AD + "out_ln.weight"), W(AD + "out_ln.bias"));
    Tout = T7;
    return ops_.download(g);
}
