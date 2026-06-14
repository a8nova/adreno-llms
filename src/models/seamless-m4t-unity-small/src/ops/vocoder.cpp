// Stage 1: CodeHiFiGAN vocoder. units -> waveform. Channel-major convs on GPU.
#include "pipeline.h"
#include <cmath>
#include <string>
#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <chrono>

std::vector<float> Pipeline::vocoder(const std::vector<float>& units) {
    // NNOPT_HWUTIL: per-vocoder-block GPU-busy via kgsl gpubusy (clFinish-bracketed).
    const bool hwutil = []{ const char* e = std::getenv("NNOPT_HWUTIL"); return e && *e && *e != '0'; }();
    auto vhw = [&](const char* name){ if(!hwutil) return; clFinish(ops_.cl().queue());
        std::ifstream f("/sys/class/kgsl/kgsl-3d0/gpubusy"); long long b=0,tot=0; if(f) f>>b>>tot;
        fprintf(stderr, "VHW %-10s busy=%lld total=%lld -> %.1f%%\n", name, b, tot, tot>0?100.0*(double)b/(double)tot:0.0); };
    if (hwutil) { clFinish(ops_.cl().queue()); std::ifstream f("/sys/class/kgsl/kgsl-3d0/gpubusy"); long long b,t; if(f) f>>b>>t; }
    // NNOPT_HOSTIDLE: time the host-side input prep (dict/spkr/lang load + channel-major
    // build + upload) — pure GPU-idle CPU work at the start of the vocoder stage.
    const bool hostidle = []{ const char* e = std::getenv("NNOPT_HOSTIDLE"); return e && *e && *e != '0'; }();
    auto _hp0 = std::chrono::steady_clock::now();
    const int T0 = (int)units.size();
    const int LE = 256, CE = 1280, SE = 256, Cin = LE + CE + SE;  // 1792
    const int lang_id = voc_lang_, spkr_id = voc_spkr_;  // target-language vocoder ids

    // Wrap unit ids on host (tiny, T0 ints) then build the channel-major input [Cin,T0] on the
    // GPU by gathering from the lang/dict/spkr WEIGHT buffers — replaces the old ~580ms host
    // build + 25MB dict page-in (pure GPU idle). dict/spkr/lang are already resident GPU weights.
    std::vector<int> code(T0);
    for (int t = 0; t < T0; ++t) { int u = (int)llround(units[t]); if (u < 0) u += 10000; code[t] = u; }
    cl_mem code_dev = ops_.upload_ints(code);
    Tensor x = ops_.vocoder_input_gather(code_dev,
                                         ops_.weight("vocoder.model.lang.weight"),
                                         ops_.weight("vocoder.model.dict.weight"),
                                         ops_.weight("vocoder.model.spkr.weight"),
                                         LE, CE, SE, T0, lang_id, spkr_id);
    if (hostidle) { clFinish(ops_.cl().queue());
        double ms = std::chrono::duration<double,std::milli>(std::chrono::steady_clock::now() - _hp0).count();
        fprintf(stderr, "HOSTIDLE vocoder_input_prep = %.1f ms (now GPU gather; was ~580ms host)\n", ms); }

    // conv_pre: 1792 -> 512, k7 p3
    x = ops_.conv1d_ct(x, ops_.weight("vocoder.model.conv_pre.weight"),
                       ops_.weight("vocoder.model.conv_pre.bias"), Cin, T0, 512, 7, 1, 3);
    int C = 512, T = T0;
    vhw("conv_pre");

    const int ups_k[5] = {11, 8, 8, 4, 4}, ups_s[5] = {5, 4, 4, 2, 2}, ups_out[5] = {256, 128, 64, 32, 16};
    const int rb_k[3] = {3, 7, 11}, rb_dil[3] = {1, 3, 5};

    auto resblock = [&](const Tensor& in, int ch, int Tn, int k, int rb_idx) -> Tensor {
        std::string p = "vocoder.model.resblocks." + std::to_string(rb_idx);
        Tensor cur = in;
        for (int j = 0; j < 3; ++j) {
            std::string s1 = p + ".convs1." + std::to_string(j);
            std::string s2 = p + ".convs2." + std::to_string(j);
            // lrelu FUSED into each conv's im2col (pre_act=ACT_LRELU01): removes the separate
            // clone+act passes (memory-bound full-tensor ops at huge T) — conv reads lrelu(input)
            // directly. conv1 reads cur (preserved for the residual); conv2 reads the conv1 output.
            Tensor xt = ops_.conv1d_ct(cur, ops_.weight(s1 + ".weight"), ops_.weight(s1 + ".bias"),
                                       ch, Tn, ch, k, rb_dil[j], get_padding(k, rb_dil[j]), ACT_LRELU01);
            xt = ops_.conv1d_ct(xt, ops_.weight(s2 + ".weight"), ops_.weight(s2 + ".bias"),
                                ch, Tn, ch, k, 1, get_padding(k, 1), ACT_LRELU01);
            ops_.axpy(xt, cur, 1.0f);  // residual: xt = xt + cur
            cur = xt;
        }
        return cur;
    };

    for (int i = 0; i < 5; ++i) {
        ops_.act_n(x, C * T, ACT_LRELU01);
        int Cout = ups_out[i], k = ups_k[i], s = ups_s[i], pad = (k - s) / 2;
        x = ops_.conv_transpose1d_ct(x, ops_.weight("vocoder.model.ups." + std::to_string(i) + ".weight"),
                                     ops_.weight("vocoder.model.ups." + std::to_string(i) + ".bias"),
                                     C, T, Cout, k, s, pad);
        C = Cout; T = x.n / C;
        Tensor acc = resblock(x, C, T, rb_k[0], i * 3 + 0);
        Tensor r1 = resblock(x, C, T, rb_k[1], i * 3 + 1);
        Tensor r2 = resblock(x, C, T, rb_k[2], i * 3 + 2);
        ops_.axpy(acc, r1, 1.0f);
        ops_.axpy(acc, r2, 1.0f);
        ops_.scale(acc, 1.0f / 3.0f);
        x = acc;
        if (hwutil) { char nm[16]; snprintf(nm, sizeof(nm), "ups%d(T=%d)", i, T); vhw(nm); }
    }
    ops_.act_n(x, C * T, ACT_LRELU001);
    x = ops_.conv1d_ct(x, ops_.weight("vocoder.model.conv_post.weight"),
                       ops_.weight("vocoder.model.conv_post.bias"), C, T, 1, 7, 1, 3);
    ops_.act_n(x, T, ACT_TANH);
    return ops_.download(x, T);
}
