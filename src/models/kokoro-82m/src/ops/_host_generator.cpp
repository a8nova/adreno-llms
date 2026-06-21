// Host-side fp64 implementation of the full iSTFTNet generator.
//   ups[0] → noise_res[0] add → resblocks[0..2] avg →
//   ups[1] → reflection_pad → noise_res[1] add → resblocks[3..5] avg →
//   leaky_relu → conv_post → exp/sin → iSTFT → PCM
//
// All computation in fp64. Reads input + weights from device via OpenCL,
// writes final int16 PCM to out_pcm_int16. Enabled via env var
// NNOPT_HOST_GENERATOR=1; the GPU path stays untouched for comparison.

#include "opencl_context.h"
#include "weights.h"
#include "debug_utils.h"
#include "model_config.h"
#include "utils.h"

#include <CL/cl.h>
#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <string>
#include <unordered_map>
#include <vector>

using vec64 = std::vector<double>;

namespace {

inline vec64 read_fp16_to_fp64(cl_command_queue queue, cl_mem buf, size_t n) {
    std::vector<uint16_t> h(n);
    clEnqueueReadBuffer(queue, buf, CL_TRUE, 0, n * sizeof(uint16_t), h.data(), 0, nullptr, nullptr);
    vec64 out(n);
    for (size_t i = 0; i < n; ++i) out[i] = (double)nnopt_f16_to_f32(h[i]);
    return out;
}

inline vec64 read_weight_fp64(Weights& weights, cl_command_queue queue, const std::string& key, size_t expected_n) {
    cl_mem buf = weights.get_buffer(key);
    if (!buf) {
        NNOPT_ERROR_FMT("host_gen: missing weight %s", key.c_str());
        return vec64();
    }
    return read_fp16_to_fp64(queue, buf, expected_n);
}

// Reconstruct weight_norm: W = (g / ||v||) * v. Returns flat W in fp64.
// v shape [C_out, C_in, K], g shape [C_out, 1, 1].
inline vec64 wn_recon(Weights& weights, cl_command_queue queue, const std::string& prefix,
                     int& C_out, int& C_in, int& K) {
    auto shape = weights.get_shape(prefix + ".weight_v");
    if (shape.size() < 3) { NNOPT_ERROR_FMT("wn: bad shape %s", prefix.c_str()); return vec64(); }
    C_out = shape[0]; C_in = shape[1]; K = shape[2];
    int per_oc = C_in * K;
    vec64 v = read_weight_fp64(weights, queue, prefix + ".weight_v", (size_t)C_out * per_oc);
    vec64 g = read_weight_fp64(weights, queue, prefix + ".weight_g", (size_t)C_out);
    vec64 W((size_t)C_out * per_oc);
    for (int oc = 0; oc < C_out; ++oc) {
        double sq = 0.0;
        for (int i = 0; i < per_oc; ++i) { double x = v[oc*per_oc + i]; sq += x*x; }
        double norm = std::sqrt(sq) + 1e-12;
        double scale = g[oc] / norm;
        for (int i = 0; i < per_oc; ++i) W[oc*per_oc + i] = v[oc*per_oc + i] * scale;
    }
    return W;
}

inline vec64 read_bias_fp64(Weights& weights, cl_command_queue queue, const std::string& key, int C, bool optional = true) {
    cl_mem buf = weights.get_buffer(key, optional);
    if (!buf) return vec64((size_t)C, 0.0);
    return read_fp16_to_fp64(queue, buf, (size_t)C);
}

// Conv1d (dilated) in fp64. in [C_in, L_in] -> out [C_out, L_out].
void conv1d(const vec64& in, int C_in, int L_in,
            const vec64& W, int C_out, int K,
            const vec64& bias,  // may be empty
            int stride, int padding, int dilation,
            vec64& out, int& L_out) {
    L_out = (L_in + 2*padding - dilation*(K-1) - 1) / stride + 1;
    out.assign((size_t)C_out * L_out, 0.0);
    int per_oc = C_in * K;
    for (int oc = 0; oc < C_out; ++oc) {
        for (int ol = 0; ol < L_out; ++ol) {
            double acc = 0.0;
            for (int k = 0; k < K; ++k) {
                int il = ol*stride - padding + k*dilation;
                if (il < 0 || il >= L_in) continue;
                for (int ic = 0; ic < C_in; ++ic) {
                    acc += in[ic*L_in + il] * W[oc*per_oc + ic*K + k];
                }
            }
            if (!bias.empty()) acc += bias[oc];
            out[oc*L_out + ol] = acc;
        }
    }
}

// ConvTranspose1d in fp64. in [C_in, L_in] -> out [C_out, L_out].
// Weight layout PyTorch ConvTranspose1d: [C_in, C_out/groups, K]
void convtr1d(const vec64& in, int C_in, int L_in,
              const vec64& W, int C_out, int K,
              const vec64& bias,
              int stride, int padding,
              vec64& out, int& L_out) {
    L_out = (L_in - 1)*stride - 2*padding + (K - 1) + 1;
    out.assign((size_t)C_out * L_out, 0.0);
    int per_ic = C_out * K;  // [C_in, C_out, K] flat: W[ic, oc, k] = W[ic*C_out*K + oc*K + k]
    for (int oc = 0; oc < C_out; ++oc) {
        for (int ol = 0; ol < L_out; ++ol) {
            double acc = 0.0;
            for (int k = 0; k < K; ++k) {
                int num = ol + padding - k;
                if (num < 0 || (num % stride) != 0) continue;
                int il = num / stride;
                if (il < 0 || il >= L_in) continue;
                for (int ic = 0; ic < C_in; ++ic) {
                    acc += in[ic*L_in + il] * W[ic*per_ic + oc*K + k];
                }
            }
            if (!bias.empty()) acc += bias[oc];
            out[oc*L_out + ol] = acc;
        }
    }
}

// Linear apply: y = W @ s + b. y[C_out], s[C_in], W[C_out, C_in], b[C_out].
vec64 linear_apply(const vec64& s, const vec64& W, const vec64& b, int C_in, int C_out) {
    vec64 y((size_t)C_out);
    for (int oc = 0; oc < C_out; ++oc) {
        double acc = b.empty() ? 0.0 : b[oc];
        for (int ic = 0; ic < C_in; ++ic) acc += W[oc*C_in + ic] * s[ic];
        y[oc] = acc;
    }
    return y;
}

// AdaIN1d: instance norm + (1+gamma)*xn + beta. x is [C, T].
vec64 adain1d(const vec64& x, int C, int T, const vec64& gamma, const vec64& beta) {
    vec64 y((size_t)C * T);
    for (int c = 0; c < C; ++c) {
        double mean = 0.0;
        for (int t = 0; t < T; ++t) mean += x[c*T + t];
        mean /= (double)T;
        double var = 0.0;
        for (int t = 0; t < T; ++t) { double d = x[c*T + t] - mean; var += d*d; }
        var /= (double)T;
        double inv = 1.0 / std::sqrt(var + 1e-5);
        for (int t = 0; t < T; ++t) {
            double xn = (x[c*T + t] - mean) * inv;
            y[c*T + t] = (1.0 + gamma[c]) * xn + beta[c];
        }
    }
    return y;
}

// Snake1D: y = x + (1/a)*sin²(a*x). alpha [C], x [C, T].
void snake_inplace(vec64& x, int C, int T, const vec64& alpha) {
    for (int c = 0; c < C; ++c) {
        double a = alpha[c];
        if (std::fabs(a) < 1e-6) a = (a >= 0 ? 1e-6 : -1e-6);
        for (int t = 0; t < T; ++t) {
            double s = std::sin(a * x[c*T + t]);
            x[c*T + t] = x[c*T + t] + (1.0 / a) * s * s;
        }
    }
}

void leaky_inplace(vec64& x, double slope) {
    for (auto& v : x) if (v < 0) v *= slope;
}

void add_inplace(vec64& a, const vec64& b) {
    size_t n = std::min(a.size(), b.size());
    for (size_t i = 0; i < n; ++i) a[i] += b[i];
}

// AdaINResBlock1: 3 iterations of (adain1 → snake → conv1[dil_i] → adain2 → snake → conv2[dil=1])
// Operates on x [C, T] in place.
void adain_resblock1(vec64& x, int C, int T, int K, const int* dilations,
                     const std::string& prefix, const vec64& style,
                     Weights& weights, cl_command_queue queue) {
    for (int i = 0; i < 3; ++i) {
        std::string si = std::to_string(i);
        // adain1
        vec64 fc1_w = read_weight_fp64(weights, queue, prefix + ".adain1." + si + ".fc.weight", (size_t)2*C*128);
        vec64 fc1_b = read_weight_fp64(weights, queue, prefix + ".adain1." + si + ".fc.bias", (size_t)2*C);
        vec64 h1 = linear_apply(style, fc1_w, fc1_b, 128, 2*C);
        vec64 gamma1(h1.begin(), h1.begin() + C);
        vec64 beta1(h1.begin() + C, h1.end());
        vec64 xt = adain1d(x, C, T, gamma1, beta1);
        // snake1
        vec64 a1 = read_weight_fp64(weights, queue, prefix + ".alpha1." + si, (size_t)C);
        snake_inplace(xt, C, T, a1);
        // conv1
        int Cout, Cin, Kw;
        vec64 W1 = wn_recon(weights, queue, prefix + ".convs1." + si, Cout, Cin, Kw);
        vec64 b1 = read_bias_fp64(weights, queue, prefix + ".convs1." + si + ".bias", Cout);
        int pad1 = (K * dilations[i] - dilations[i]) / 2;
        vec64 c1; int Lc1;
        conv1d(xt, Cin, T, W1, Cout, Kw, b1, 1, pad1, dilations[i], c1, Lc1);
        // adain2
        vec64 fc2_w = read_weight_fp64(weights, queue, prefix + ".adain2." + si + ".fc.weight", (size_t)2*C*128);
        vec64 fc2_b = read_weight_fp64(weights, queue, prefix + ".adain2." + si + ".fc.bias", (size_t)2*C);
        vec64 h2 = linear_apply(style, fc2_w, fc2_b, 128, 2*C);
        vec64 gamma2(h2.begin(), h2.begin() + C);
        vec64 beta2(h2.begin() + C, h2.end());
        xt = adain1d(c1, C, Lc1, gamma2, beta2);
        // snake2
        vec64 a2 = read_weight_fp64(weights, queue, prefix + ".alpha2." + si, (size_t)C);
        snake_inplace(xt, C, Lc1, a2);
        // conv2
        int Co2, Ci2, K2;
        vec64 W2 = wn_recon(weights, queue, prefix + ".convs2." + si, Co2, Ci2, K2);
        vec64 b2 = read_bias_fp64(weights, queue, prefix + ".convs2." + si + ".bias", Co2);
        int pad2 = (K - 1) / 2;
        vec64 c2; int Lc2;
        conv1d(xt, Ci2, Lc1, W2, Co2, K2, b2, 1, pad2, 1, c2, Lc2);
        // x += c2
        add_inplace(x, c2);
    }
}

void reflection_pad_left1(vec64& x, int C, int& T) {
    int T_new = T + 1;
    vec64 padded((size_t)C * T_new);
    for (int c = 0; c < C; ++c) {
        padded[c*T_new + 0] = x[c*T + 1];  // reflect index 1
        for (int t = 0; t < T; ++t) padded[c*T_new + 1 + t] = x[c*T + t];
    }
    x = std::move(padded);
    T = T_new;
}

// Forward STFT in fp64, center=True semantics
void stft(const vec64& signal, int T_audio, int n_fft, int hop, const vec64& window,
          vec64& mag, vec64& phase, int& T_frames) {
    T_frames = T_audio / hop + 1;
    int n_freq = n_fft / 2 + 1;
    mag.assign((size_t)n_freq * T_frames, 0.0);
    phase.assign((size_t)n_freq * T_frames, 0.0);
    for (int f = 0; f < T_frames; ++f) {
        int n0 = f * hop - n_fft / 2;
        for (int k = 0; k < n_freq; ++k) {
            double re = 0, im = 0;
            for (int t = 0; t < n_fft; ++t) {
                int n = n0 + t;
                if (n >= 0 && n < T_audio) {
                    double v = signal[n] * window[t];
                    double ang = -2.0 * M_PI * k * t / n_fft;
                    re += v * std::cos(ang);
                    im += v * std::sin(ang);
                }
            }
            mag[k*T_frames + f] = std::sqrt(re*re + im*im);
            phase[k*T_frames + f] = std::atan2(im, re);
        }
    }
}

}  // anonymous namespace

extern "C" int op_decoder_host(OpenCLContext& cl_ctx, Weights& weights, cl_command_queue queue,
                                cl_mem x_enc_fp16,        // [512, T_dec_final] decoder.decode chain output (after upsample by 2)
                                cl_mem F0_pred_fp16,      // [T_frames*2] F0 from F0Ntrain (predictor output for T_frames after F0_conv)
                                cl_mem N_pred_fp16,       // [T_frames*2]
                                cl_mem ref_s_dec_fp16,    // [256] style vector
                                int T_dec_final,          // = T_frames * 2 (since decode.3 upsamples by 2)
                                int T_frames,             // original T_frames before decode chain
                                std::vector<int16_t>& out_pcm_int16) {
    auto t_start = std::chrono::steady_clock::now();
    NNOPT_CHECKPOINT("HOST GENERATOR started (fp64)");
    (void)cl_ctx;

    // Read inputs to host fp64
    vec64 x = read_fp16_to_fp64(queue, x_enc_fp16, (size_t)512 * T_dec_final);
    vec64 F0 = read_fp16_to_fp64(queue, F0_pred_fp16, (size_t)T_frames * 2);
    vec64 N  = read_fp16_to_fp64(queue, N_pred_fp16,  (size_t)T_frames * 2);
    vec64 style_full = read_fp16_to_fp64(queue, ref_s_dec_fp16, 256);
    vec64 style(style_full.begin(), style_full.begin() + 128);

    // ─── SineGen + STFT for noise path ───
    const int n_fft = 20, hop = 5, n_freq = 11;
    const int upsample_to_audio_full = 300;
    int T_audio_full = T_dec_final * upsample_to_audio_full;

    // F0 upsample (linear) from T_frames*2 to T_audio_full
    int F0_T_in = T_frames * 2;
    vec64 f0_up((size_t)T_audio_full);
    for (int t = 0; t < T_audio_full; ++t) {
        double pos = (double)t * (double)(F0_T_in - 1) / (double)(T_audio_full - 1);
        int t0 = (int)pos; int t1 = std::min(t0 + 1, F0_T_in - 1);
        double frac = pos - t0;
        f0_up[t] = (1.0 - frac) * F0[t0] + frac * F0[t1];
    }
    // SineGen with cumsum (deterministic, rand_ini=0)
    int n_harm = 9;
    double samp_rate = 24000.0;
    double voiced_th = 10.0;
    double sine_amp = 0.1;
    vec64 cumphase((size_t)T_audio_full);
    double acc = 0.0;
    for (int t = 0; t < T_audio_full; ++t) {
        acc += f0_up[t] / samp_rate;
        cumphase[t] = acc;
    }
    vec64 sines((size_t)n_harm * T_audio_full);
    vec64 uv((size_t)T_audio_full);
    for (int t = 0; t < T_audio_full; ++t) {
        double u = f0_up[t] > voiced_th ? 1.0 : 0.0;
        uv[t] = u;
        for (int h = 0; h < n_harm; ++h) {
            double phi = 2.0 * M_PI * (double)(h + 1) * cumphase[t];
            sines[h*T_audio_full + t] = std::sin(phi) * sine_amp * u;
        }
    }
    // m_source.l_linear: Linear(9, 1) + tanh
    vec64 lm_w = read_weight_fp64(weights, queue, "decoder.module.generator.m_source.l_linear.weight", (size_t)1 * 9);
    vec64 lm_b = read_weight_fp64(weights, queue, "decoder.module.generator.m_source.l_linear.bias", (size_t)1);
    vec64 har_source((size_t)T_audio_full);
    for (int t = 0; t < T_audio_full; ++t) {
        double s = lm_b[0];
        for (int h = 0; h < n_harm; ++h) s += sines[h*T_audio_full + t] * lm_w[h];
        har_source[t] = std::tanh(s);
    }
    // STFT to get har_spec, har_phase
    vec64 hann((size_t)n_fft);
    for (int i = 0; i < n_fft; ++i) hann[i] = 0.5 * (1.0 - std::cos(2.0 * M_PI * i / n_fft));
    vec64 har_mag, har_phase; int T_high;
    stft(har_source, T_audio_full, n_fft, hop, hann, har_mag, har_phase, T_high);
    // Concat: har [22, T_high] = [har_mag(11), har_phase(11)]
    vec64 har((size_t)22 * T_high);
    for (int c = 0; c < n_freq; ++c) {
        for (int t = 0; t < T_high; ++t) {
            har[c*T_high + t] = har_mag[c*T_high + t];
            har[(n_freq+c)*T_high + t] = har_phase[c*T_high + t];
        }
    }

    // ─── Generator chain ───
    const int resblock_kernels[3] = {3, 7, 11};
    const int dilations[3] = {1, 3, 5};
    int gC = 512, gT = T_dec_final;

    // Level 0: leaky → ups[0] → noise_res[0] add → resblocks 0..2 avg
    leaky_inplace(x, 0.1);
    int u0_Cout, u0_Cin, u0_K;
    vec64 u0_W = wn_recon(weights, queue, "decoder.module.generator.ups.0", u0_Cin, u0_Cout, u0_K);  // [512, 256, 20]
    vec64 u0_b = read_bias_fp64(weights, queue, "decoder.module.generator.ups.0.bias", u0_Cout);
    vec64 u0_out; int L_u0;
    convtr1d(x, gC, gT, u0_W, u0_Cout, u0_K, u0_b, 10, 5, u0_out, L_u0);
    gC = 256; gT = L_u0;

    // noise_convs[0]: plain Conv1d(22, 256, k=12, stride=6, padding=3) on har
    vec64 nc0_W = read_weight_fp64(weights, queue, "decoder.module.generator.noise_convs.0.weight", (size_t)256 * 22 * 12);
    vec64 nc0_b = read_bias_fp64(weights, queue, "decoder.module.generator.noise_convs.0.bias", 256);
    vec64 nc0_out; int Lnc0;
    conv1d(har, 22, T_high, nc0_W, 256, 12, nc0_b, 6, 3, 1, nc0_out, Lnc0);
    // noise_res[0]: AdaINResBlock1(256, k=7, [1,3,5])
    if (Lnc0 == gT) {
        adain_resblock1(nc0_out, 256, gT, 7, dilations, "decoder.module.generator.noise_res.0", style, weights, queue);
        for (size_t i = 0; i < u0_out.size() && i < nc0_out.size(); ++i) u0_out[i] += nc0_out[i];
    }

    // resblocks 0..2: each AdaINResBlock1(256, k=resblock_kernels[j], dilations)
    vec64 xs0((size_t)gC * gT, 0.0);
    for (int j = 0; j < 3; ++j) {
        vec64 r = u0_out;  // copy
        adain_resblock1(r, gC, gT, resblock_kernels[j], dilations,
                        "decoder.module.generator.resblocks." + std::to_string(j),
                        style, weights, queue);
        for (size_t i = 0; i < xs0.size(); ++i) xs0[i] += r[i];
    }
    for (auto& v : xs0) v /= 3.0;

    // Level 1: leaky → ups[1] → reflection_pad → noise_res[1] add → resblocks 3..5 avg
    leaky_inplace(xs0, 0.1);
    int u1_Cin, u1_Cout, u1_K;
    vec64 u1_W = wn_recon(weights, queue, "decoder.module.generator.ups.1", u1_Cin, u1_Cout, u1_K);  // [256, 128, 12]
    vec64 u1_b = read_bias_fp64(weights, queue, "decoder.module.generator.ups.1.bias", u1_Cout);
    vec64 u1_out; int L_u1;
    convtr1d(xs0, gC, gT, u1_W, u1_Cout, u1_K, u1_b, 6, 3, u1_out, L_u1);
    gC = 128; gT = L_u1;
    reflection_pad_left1(u1_out, gC, gT);

    // noise_convs[1]: plain Conv1d(22, 128, k=1, stride=1, padding=0)
    vec64 nc1_W = read_weight_fp64(weights, queue, "decoder.module.generator.noise_convs.1.weight", (size_t)128 * 22 * 1);
    vec64 nc1_b = read_bias_fp64(weights, queue, "decoder.module.generator.noise_convs.1.bias", 128);
    vec64 nc1_out; int Lnc1;
    conv1d(har, 22, T_high, nc1_W, 128, 1, nc1_b, 1, 0, 1, nc1_out, Lnc1);
    if (Lnc1 == gT) {
        adain_resblock1(nc1_out, 128, gT, 11, dilations, "decoder.module.generator.noise_res.1", style, weights, queue);
        for (size_t i = 0; i < u1_out.size() && i < nc1_out.size(); ++i) u1_out[i] += nc1_out[i];
    }

    // resblocks 3..5
    vec64 xs1((size_t)gC * gT, 0.0);
    for (int j = 0; j < 3; ++j) {
        vec64 r = u1_out;
        adain_resblock1(r, gC, gT, resblock_kernels[j], dilations,
                        "decoder.module.generator.resblocks." + std::to_string(3 + j),
                        style, weights, queue);
        for (size_t i = 0; i < xs1.size(); ++i) xs1[i] += r[i];
    }
    for (auto& v : xs1) v /= 3.0;

    // Final leaky_relu (default 0.01) + conv_post
    leaky_inplace(xs1, 0.01);
    int cp_Cout, cp_Cin, cp_K;
    vec64 cp_W = wn_recon(weights, queue, "decoder.module.generator.conv_post", cp_Cout, cp_Cin, cp_K);  // [22, 128, 7]
    vec64 cp_b = read_bias_fp64(weights, queue, "decoder.module.generator.conv_post.bias", cp_Cout);
    vec64 cp_out; int Lcp;
    conv1d(xs1, cp_Cin, gT, cp_W, cp_Cout, cp_K, cp_b, 1, 3, 1, cp_out, Lcp);

    // exp(first n_freq) → mag; sin(next n_freq) → phase
    vec64 mag((size_t)n_freq * Lcp), phase_o((size_t)n_freq * Lcp);
    for (int c = 0; c < n_freq; ++c) {
        for (int t = 0; t < Lcp; ++t) {
            mag[c*Lcp + t] = std::exp(cp_out[c*Lcp + t]);
            phase_o[c*Lcp + t] = std::sin(cp_out[(n_freq+c)*Lcp + t]);
        }
    }

    // iSTFT: complex spec = mag * exp(j*phase). Use rfft inverse (Hermitian).
    int T_audio = (Lcp - 1) * hop + n_fft;
    vec64 audio((size_t)T_audio, 0.0), norm_buf((size_t)T_audio, 0.0);
    for (int f = 0; f < Lcp; ++f) {
        for (int t = 0; t < n_fft; ++t) {
            double s = mag[0*Lcp + f] * std::cos(phase_o[0*Lcp + f]);
            for (int k = 1; k < n_freq - 1; ++k) {
                double ang = 2.0 * M_PI * k * t / n_fft + phase_o[k*Lcp + f];
                s += 2.0 * mag[k*Lcp + f] * std::cos(ang);
            }
            double ang_n = M_PI * t + phase_o[(n_freq-1)*Lcp + f];
            s += mag[(n_freq-1)*Lcp + f] * std::cos(ang_n);
            s /= (double)n_fft;
            int oi = f * hop + t;
            if (oi < T_audio) {
                audio[oi] += hann[t] * s;
                norm_buf[oi] += hann[t] * hann[t];
            }
        }
    }
    for (int i = 0; i < T_audio; ++i) {
        if (norm_buf[i] > 1e-10) audio[i] /= norm_buf[i];
    }
    // center=True trim
    int trim = n_fft / 2;
    int T_trim = T_audio - 2 * trim;
    // DC removal only. NO loudness normalization: the reference emits the raw
    // float waveform (peak ~0.3); the old 0.8/p99 rescale (~4.3x) hard-clipped
    // ~0.36% of samples — an audible high-pitched distorted copy of the voice
    // laid over the speech. Clamp is safety only.
    double dc = 0.0;
    for (int i = 0; i < T_trim; ++i) dc += audio[trim + i];
    dc /= (double)T_trim;

    out_pcm_int16.assign((size_t)T_trim, 0);
    for (int i = 0; i < T_trim; ++i) {
        double s = audio[trim + i] - dc;
        if (s > 1.0) s = 1.0;
        if (s < -1.0) s = -1.0;
        out_pcm_int16[i] = (int16_t)(s * 32767.0);
    }
    auto t_end = std::chrono::steady_clock::now();
    double secs = std::chrono::duration_cast<std::chrono::duration<double>>(t_end - t_start).count();
    NNOPT_CHECKPOINT(("HOST GENERATOR done in " + std::to_string(secs) + "s").c_str());
    return 0;
}
