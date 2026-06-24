// extract — OpenVoice V2 tone-color extractor (ReferenceEncoder), CPU/fp32.
//
// Mirrors openvoice/models.py::ReferenceEncoder.forward and api.py::extract_se:
//   ref wav --22.05kHz--> spectrogram_torch [513,T] --transpose--> [T,513]
//     -> LayerNorm(513)
//     -> 6× Conv2d(k=3, stride=2, pad=1) + ReLU   (1→32→32→64→64→128→128, weight_norm)
//     -> [128, H6, 9] -> transpose/flatten -> GRU(1152 -> 128), take final hidden
//     -> Linear(128 -> 256) = g  (the 256-d tone-color embedding)
//   -> write 256 float32 to --out (the exact format load_g() reads, == assets/g_*.bin)
//
// Runs entirely on the CPU: the tensors are tiny and this is a one-time step per
// reference clip, so it is not worth a GPU kernel (cf. whisper-tiny's CPU mel
// front-end). fp32 throughout for accuracy.
#include "engine.h"          // Weights, OpenCLContext, boot()
#include "../ov_stft.h"      // ov_spectrogram (spectrogram_torch mirror)
#include "../load_wav.h"
#include <cmath>
#include <cstring>
#include <string>
#include <thread>
#include <algorithm>

namespace {

// weight_norm Conv2d weight: w[co,ci,kh,kw] = v * (g[co] / ||v_co||), norm over (ci,kh,kw).
std::vector<float> conv_weight(const Weights& W, const std::string& base, int Cout, int Cin) {
    std::vector<float> v = W.get_host_vec(base + ".weight_v");   // [Cout,Cin,3,3]
    std::vector<float> g = W.get_host_vec(base + ".weight_g");   // [Cout,1,1,1]
    const int rest = Cin * 9;
    std::vector<float> w(v.size());
    for (int co = 0; co < Cout; co++) {
        double nrm = 0.0;
        for (int i = 0; i < rest; i++) { double x = v[(size_t)co * rest + i]; nrm += x * x; }
        nrm = std::sqrt(nrm) + 1e-12;
        const float sc = (float)(g[co] / nrm);
        for (int i = 0; i < rest; i++) w[(size_t)co * rest + i] = v[(size_t)co * rest + i] * sc;
    }
    return w;
}

// Conv2d k=3 stride=2 pad=1 + ReLU. in/out are channel-major [C,H,W].
std::vector<float> conv2d_relu(const std::vector<float>& x, int Cin, int H, int Wd,
                               const std::vector<float>& w, const std::vector<float>& b,
                               int Cout, int& Ho, int& Wo) {
    const int K = 3, S = 2, P = 1;
    Ho = (H + 2 * P - K) / S + 1;
    Wo = (Wd + 2 * P - K) / S + 1;
    std::vector<float> out((size_t)Cout * Ho * Wo);
    auto worker = [&](int c0, int c1) {
        for (int co = c0; co < c1; co++) {
            for (int oh = 0; oh < Ho; oh++) {
                for (int ow = 0; ow < Wo; ow++) {
                    double acc = b[co];
                    for (int ci = 0; ci < Cin; ci++) {
                        const float* xc = &x[(size_t)ci * H * Wd];
                        const float* wc = &w[((size_t)co * Cin + ci) * 9];
                        for (int kh = 0; kh < K; kh++) {
                            int ih = oh * S - P + kh;
                            if (ih < 0 || ih >= H) continue;
                            for (int kw = 0; kw < K; kw++) {
                                int iw = ow * S - P + kw;
                                if (iw < 0 || iw >= Wd) continue;
                                acc += (double)xc[(size_t)ih * Wd + iw] * wc[kh * 3 + kw];
                            }
                        }
                    }
                    float v = (float)acc;
                    out[((size_t)co * Ho + oh) * Wo + ow] = v > 0.0f ? v : 0.0f;  // ReLU
                }
            }
        }
    };
    unsigned hw = std::thread::hardware_concurrency();
    int nt = (hw == 0) ? 4 : std::min<unsigned>(hw, 8);
    nt = std::min(nt, Cout);
    int per = (Cout + nt - 1) / nt;
    std::vector<std::thread> pool;
    for (int t = 0; t < nt; t++) {
        int a = t * per, c = std::min(Cout, a + per);
        if (a >= c) break;
        pool.emplace_back(worker, a, c);
    }
    for (auto& th : pool) th.join();
    return out;
}

inline float sigmoidf(float x) { return 1.0f / (1.0f + std::exp(-x)); }

}  // namespace

int run_extract(int argc, char** argv) {
    std::string in, out = "g.bin";
    for (int i = 2; i < argc; i++) {
        if (!std::strcmp(argv[i], "--in")  && i + 1 < argc) in = argv[++i];
        else if (!std::strcmp(argv[i], "--out") && i + 1 < argc) out = argv[++i];
    }
    if (in.empty()) { printf("extract: need --in ref.wav [--out g.bin]\n"); return 64; }

    OpenCLContext cl; Weights W; if (boot(cl, W)) return 1;   // GPU init only to load weights; compute is CPU

    // ── wav → spectrogram [513,T] (freq-major) → conv input [1,T,513] (time-major) ──
    std::vector<float> audio; nnopt::WavInfo wi;
    if (!nnopt::read_wav_mono_f32(in, audio, wi, 22050)) { printf("extract: cannot read --in %s\n", in.c_str()); return 1; }
    int T = 0;
    std::vector<float> spec = nnopt::ov_spectrogram(audio, &T);   // spec[f*T + t], f in [0,513)
    if (T <= 0) { printf("extract: --in too short (%zu samples)\n", audio.size()); return 1; }
    const int Wd0 = 513;
    printf("extract: --in %s | %d samples @ %dHz → spec [513,%d]\n", in.c_str(), (int)audio.size(), wi.sample_rate, T);

    // input image [C=1, H=T, W=513]: img[t,f] = spec[f*T + t]
    std::vector<float> img((size_t)T * Wd0);
    for (int t = 0; t < T; t++)
        for (int f = 0; f < Wd0; f++)
            img[(size_t)t * Wd0 + f] = spec[(size_t)f * T + t];

    // ── LayerNorm over last dim (513), per time row; eps=1e-5 ──
    {
        std::vector<float> ln_w = W.get_host_vec("ref_enc.layernorm.weight");
        std::vector<float> ln_b = W.get_host_vec("ref_enc.layernorm.bias");
        for (int t = 0; t < T; t++) {
            float* row = &img[(size_t)t * Wd0];
            double mean = 0.0; for (int f = 0; f < Wd0; f++) mean += row[f]; mean /= Wd0;
            double var = 0.0; for (int f = 0; f < Wd0; f++) { double d = row[f] - mean; var += d * d; } var /= Wd0;
            float inv = (float)(1.0 / std::sqrt(var + 1e-5));
            for (int f = 0; f < Wd0; f++) row[f] = ((float)(row[f] - mean) * inv) * ln_w[f] + ln_b[f];
        }
    }

    // ── 6× Conv2d(3,2,1)+ReLU: 1→32→32→64→64→128→128 ──
    const int chans[7] = {1, 32, 32, 64, 64, 128, 128};
    std::vector<float> x = img;          // [Cin=1, H=T, W=513]
    int H = T, Wd = Wd0, Cin = 1;
    for (int l = 0; l < 6; l++) {
        const int Cout = chans[l + 1];
        std::string base = "ref_enc.convs." + std::to_string(l);
        std::vector<float> w = conv_weight(W, base, Cout, Cin);
        std::vector<float> b = W.get_host_vec(base + ".bias");
        int Ho, Wo;
        x = conv2d_relu(x, Cin, H, Wd, w, b, Cout, Ho, Wo);
        Cin = Cout; H = Ho; Wd = Wo;
    }
    const int C6 = 128, H6 = H, W6 = Wd;   // expect W6 == 9
    printf("extract: after convs [%d,%d,%d] (GRU input %d, steps %d)\n", C6, H6, W6, C6 * W6, H6);

    // ── reshape [C=128, H6, W6] → GRU sequence [H6, 128*W6] : feat[t][c*W6 + w] ──
    const int FEAT = C6 * W6;             // 128*9 = 1152
    std::vector<float> seq((size_t)H6 * FEAT);
    for (int t = 0; t < H6; t++)
        for (int c = 0; c < C6; c++)
            for (int w = 0; w < W6; w++)
                seq[(size_t)t * FEAT + (c * W6 + w)] = x[((size_t)c * H6 + t) * W6 + w];

    // ── GRU(input=FEAT, hidden=128), gates ordered [r,z,n]; take final hidden ──
    const int HID = 128;
    std::vector<float> Wih = W.get_host_vec("ref_enc.gru.weight_ih_l0");  // [384, FEAT]
    std::vector<float> Whh = W.get_host_vec("ref_enc.gru.weight_hh_l0");  // [384, 128]
    std::vector<float> bih = W.get_host_vec("ref_enc.gru.bias_ih_l0");    // [384]
    std::vector<float> bhh = W.get_host_vec("ref_enc.gru.bias_hh_l0");    // [384]
    std::vector<float> h(HID, 0.0f);
    std::vector<float> gi(3 * HID), gh(3 * HID);
    for (int t = 0; t < H6; t++) {
        const float* xt = &seq[(size_t)t * FEAT];
        // gi = Wih·x + bih ; gh = Whh·h + bhh
        for (int o = 0; o < 3 * HID; o++) {
            const float* wr = &Wih[(size_t)o * FEAT];
            double a = bih[o];
            for (int k = 0; k < FEAT; k++) a += (double)wr[k] * xt[k];
            gi[o] = (float)a;
            const float* hr = &Whh[(size_t)o * HID];
            double c = bhh[o];
            for (int k = 0; k < HID; k++) c += (double)hr[k] * h[k];
            gh[o] = (float)c;
        }
        for (int j = 0; j < HID; j++) {
            float r = sigmoidf(gi[j]        + gh[j]);
            float z = sigmoidf(gi[HID + j]  + gh[HID + j]);
            float n = std::tanh(gi[2*HID + j] + r * gh[2*HID + j]);
            h[j] = (1.0f - z) * n + z * h[j];
        }
    }

    // ── proj: Linear(128→256) → g ──
    std::vector<float> pw = W.get_host_vec("ref_enc.proj.weight");  // [256,128]
    std::vector<float> pb = W.get_host_vec("ref_enc.proj.bias");    // [256]
    std::vector<float> g(256);
    for (int o = 0; o < 256; o++) {
        double a = pb[o];
        const float* wr = &pw[(size_t)o * HID];
        for (int k = 0; k < HID; k++) a += (double)wr[k] * h[k];
        g[o] = (float)a;
    }

    FILE* f = fopen(out.c_str(), "wb");
    if (!f) { printf("extract: cannot write %s\n", out.c_str()); return 1; }
    fwrite(g.data(), 4, 256, f); fclose(f);
    printf("extract: wrote %s (256-d tone color) | g[0..4]= %.4f %.4f %.4f %.4f %.4f\n",
           out.c_str(), g[0], g[1], g[2], g[3], g[4]);
    return 0;
}
