// =============================================================================
// encodec_host.cpp — CPU/fp32 EnCodec 32kHz decoder for musicgen-small.
// See encodec_host.h for the modeling_encodec.py line-range citations.
//
// Derived decoder layer map (from weights/codec_decoder.fp16.meta.json keys,
// reproducing EncodecDecoder.__init__'s module list, lines 310-334):
//
//   idx  module                                    weight keys
//   ----------------------------------------------------------------------
//    0   Conv1d(128->1024, k=7, s=1)               layers.0.conv.*
//    1   LSTM(1024, num_layers=2, residual)        layers.1.lstm.*
//    2   ELU
//    3   ConvTranspose1d(1024->512, k=16, s=8)     layers.3.conv.*
//    4   ResnetBlock(512): ELU,Conv(512->256,k=3), layers.4.block.{1,3}.conv.*
//        ELU,Conv(256->512,k=1); shortcut=Identity
//    5   ELU
//    6   ConvTranspose1d(512->256, k=10, s=5)      layers.6.conv.*
//    7   ResnetBlock(256): ELU,Conv(256->128,k=3), layers.7.block.{1,3}.conv.*
//        ELU,Conv(128->256,k=1); shortcut=Identity
//    8   ELU
//    9   ConvTranspose1d(256->128, k=8, s=4)       layers.9.conv.*
//   10   ResnetBlock(128): ELU,Conv(128->64,k=3),  layers.10.block.{1,3}.conv.*
//        ELU,Conv(64->128,k=1); shortcut=Identity
//   11   ELU
//   12   ConvTranspose1d(128->64, k=8, s=4)        layers.12.conv.*
//   13   ResnetBlock(64): ELU,Conv(64->32,k=3),    layers.13.block.{1,3}.conv.*
//        ELU,Conv(32->64,k=1); shortcut=Identity
//   14   ELU
//   15   Conv1d(64->1, k=7, s=1)                   layers.15.conv.*
//
// RVQ: audio_encoder.quantizer.layers.{0..3}.codebook.embed [2048,128].
//   decode = sum_k embed_k[ codes[k][t] ] -> latent[128][T]  (no projection;
//   EncodecVectorQuantization here has no project_in/out).
// =============================================================================

#include <map>
#include "encodec_host.h"
#include "opencl_context.h"
#include "profiler.h"
#include <vector>
#include <string>
#include <cmath>
#include <cstdio>
#include <cstdint>
#include <thread>
#include <algorithm>
#include <functional>
#include <chrono>

namespace {

// --- white-flash mitigation --------------------------------------------------
// The GPU EnCodec SEANet decode (encodec_decode_gpu) queues its whole conv/LSTM/
// upsample stack onto the in-order queue with no host sync until the final PCM
// readback. On the Adreno 620 (single CU) that unbroken burst starves
// SurfaceFlinger of every vsync → the entire screen goes WHITE for the duration
// (right after token decode completes). Draining the queue + a ~2 ms sleep
// between stages hands the GPU back so the compositor can grab a frame. Gated on
// NNOPT_GPU_YIELD (set by the Android app). Mirrors smolvlm's
// OpenCLContext::yield_for_compositor. Cost: a few clFinish stalls (~ms each).
void enc_yield(cl_command_queue q) {
    static const bool on = [](){
        const char* e = std::getenv("NNOPT_GPU_YIELD");
        return e != nullptr && e[0] != '0' && e[0] != '\0';
    }();
    if (!on || !q) return;
    clFinish(q);  // clFlush is non-blocking; clFinish blocks until the GPU truly idles
    std::this_thread::sleep_for(std::chrono::milliseconds(2));
}

// --- multi-core out-channel parallelism --------------------------------------
// The SEANet decode was 54 s of a 140 s 5-s-clip wall (0.35 ms/sample, ~11×
// slower than realtime) running these conv loops on ONE core. Both conv kinds
// are embarrassingly parallel over OUT channels: each thread owns a disjoint
// o-slice (disjoint output/acc rows), and within a fixed (o,t) the
// accumulation order (ic asc, then k asc, double accum) is IDENTICAL to the
// serial loop — output stays bit-identical. SD765G = 2 big + 6 LITTLE cores.
void parallel_over_channels(int n_ch, const std::function<void(int, int)>& fn) {
    unsigned hw = std::thread::hardware_concurrency();
    // NNOPT_ENC_THREADS caps the pool — the streaming worker runs WHILE the
    // GPU decodes and CPU memory traffic contends with the GPU's (shared
    // LPDDR4X): full 8 threads measured −0.5 tok/s decode. Tune the trade.
    static const int cap = [](){
        const char* e = std::getenv("NNOPT_ENC_THREADS");
        return e ? std::atoi(e) : 8;
    }();
    int nthreads = (int)std::min<unsigned>(hw ? hw : 4, (unsigned)std::max(1, cap));
    if (nthreads > n_ch) nthreads = n_ch;
    if (nthreads <= 1) { fn(0, n_ch); return; }
    std::vector<std::thread> pool;
    pool.reserve(nthreads - 1);
    int chunk = (n_ch + nthreads - 1) / nthreads;
    for (int i = 1; i < nthreads; ++i) {
        int lo = i * chunk, hi = std::min(n_ch, lo + chunk);
        if (lo >= hi) break;
        pool.emplace_back(fn, lo, hi);
    }
    fn(0, std::min(n_ch, chunk));
    for (auto& t : pool) t.join();
}

// A 1-D temporal signal: data laid out channel-major, channel c spans
// data[c*T .. c*T+T-1]. ch = number of channels, T = number of time steps.
struct Signal {
    int ch = 0;
    int T = 0;
    std::vector<float> data;  // size ch*T
    float& at(int c, int t) { return data[(size_t)c * T + t]; }
    float at(int c, int t) const { return data[(size_t)c * T + t]; }
};

bool g_error = false;

void fail(const char* what, const std::string& detail) {
    g_error = true;
    fprintf(stderr, "[encodec_host] ERROR %s: %s\n", what, detail.c_str());
}

// --- weight_norm effective weight (PyTorch dim=0, see header) ---------------
// raw weight axis-0 size = N (Conv1d: out, ConvTranspose1d: in). For each
// axis-0 slice s, effective[s,...] = g[s] * v[s,...] / l2norm(v[s,...]),
// where the l2 norm runs over the remaining axes (size = inner = prod of the
// other two dims). Returns the dequantized effective weight, flat row-major
// with the same [N, inner] layout as v.
bool weight_norm_effective(Weights& w, const std::string& prefix,
                           std::vector<float>& out, int& N, int& inner,
                           int& d1, int& d2) {
    const std::string vk = prefix + ".weight_v";
    const std::string gk = prefix + ".weight_g";
    if (!w.has_tensor(vk)) { fail("missing weight", vk); return false; }
    if (!w.has_tensor(gk)) { fail("missing weight", gk); return false; }
    std::vector<int> vs = w.get_shape(vk);
    std::vector<int> gs = w.get_shape(gk);
    if (vs.size() != 3) {
        fail("weight_v rank", vk + " expected rank 3 got rank " + std::to_string(vs.size()));
        return false;
    }
    N = vs[0]; d1 = vs[1]; d2 = vs[2];
    inner = d1 * d2;
    // g must be [N,1,1] (weight_norm dim=0).
    if (gs.empty() || gs[0] != N) {
        fail("weight_g shape", gk + " expected [" + std::to_string(N) +
             ",1,1], first dim mismatch");
        return false;
    }
    std::vector<float> v = w.get_host_vec(vk);
    std::vector<float> g = w.get_host_vec(gk);
    if ((int)v.size() != N * inner) {
        fail("weight_v elem count", vk + " got " + std::to_string(v.size()) +
             " expected " + std::to_string(N * inner));
        return false;
    }
    if ((int)g.size() < N) {
        fail("weight_g elem count", gk + " got " + std::to_string(g.size()) +
             " expected " + std::to_string(N));
        return false;
    }
    out.resize((size_t)N * inner);
    // Parallel over axis-0 slices (disjoint rows; per-slice math unchanged →
    // bit-identical). Host weight prep was 1.9 s/clip of the GPU-EnCodec path.
    parallel_over_channels(N, [&](int s_lo, int s_hi) {
        for (int s = s_lo; s < s_hi; ++s) {
            double sq = 0.0;
            const float* vp = &v[(size_t)s * inner];
            for (int j = 0; j < inner; ++j) sq += (double)vp[j] * (double)vp[j];
            double norm = std::sqrt(sq);
            double scale = (norm > 0.0) ? ((double)g[s] / norm) : 0.0;
            float* op = &out[(size_t)s * inner];
            for (int j = 0; j < inner; ++j) op[j] = (float)(scale * (double)vp[j]);
        }
    });
    return true;
}

// --- ELU(alpha=1.0), applied in place over the whole signal ------------------
void elu_inplace(Signal& s) {
    for (float& x : s.data)
        if (x <= 0.0f) x = std::expm1(x);  // exp(x)-1 for x<=0; identity for x>0
}

// --- reflect pad along time (mirror nn.functional.pad mode="reflect") --------
// Reflect padding does NOT repeat the border element: index maps via mirror
// about the first/last sample. _pad1d also inserts extra zero padding on the
// right first when length <= max(pad) (lines 144-148), then trims it off the
// reflected result. We replicate that exactly.
bool pad1d_reflect(const Signal& in, int pad_left, int pad_right, Signal& out) {
    if (pad_left < 0 || pad_right < 0) {
        fail("pad1d negative", "left=" + std::to_string(pad_left) +
             " right=" + std::to_string(pad_right));
        return false;
    }
    int length = in.T;
    int max_pad = pad_left > pad_right ? pad_left : pad_right;
    int extra_pad = 0;
    int work_len = length;
    if (length <= max_pad) {
        extra_pad = max_pad - length + 1;
        work_len = length + extra_pad;  // extra zeros appended on the right
    }
    int padded_len = pad_left + work_len + pad_right;
    Signal padded;
    padded.ch = in.ch;
    padded.T = padded_len;
    padded.data.assign((size_t)in.ch * padded_len, 0.0f);
    for (int c = 0; c < in.ch; ++c) {
        for (int o = 0; o < padded_len; ++o) {
            int src = o - pad_left;  // index into the work buffer (0..work_len-1)
            // reflect about [0, work_len-1] (mirror without repeating border)
            if (work_len > 1) {
                while (src < 0 || src >= work_len) {
                    if (src < 0) src = -src;
                    if (src >= work_len) src = 2 * (work_len - 1) - src;
                }
            } else {
                src = 0;
            }
            // work buffer = original samples [0..length-1] then extra_pad zeros
            float val = (src < length) ? in.at(c, src) : 0.0f;
            padded.at(c, o) = val;
        }
    }
    // Trim the extra_pad we inserted (lines 149-151: padded[..., :end]).
    int end = padded_len - extra_pad;
    out.ch = in.ch;
    out.T = end;
    out.data.assign((size_t)in.ch * end, 0.0f);
    for (int c = 0; c < in.ch; ++c)
        for (int o = 0; o < end; ++o) out.at(c, o) = padded.at(c, o);
    return true;
}

// --- EncodecConv1d.forward (non-causal, weight_norm) -------------------------
// dilation supported; stride is 1 for every decoder Conv1d but handled anyway.
bool encodec_conv1d(Weights& w, const std::string& prefix, const Signal& in,
                    int stride, int dilation, Signal& out) {
    std::vector<float> wt;
    int out_ch, inner, in_ch, k;
    if (!weight_norm_effective(w, prefix, wt, out_ch, inner, in_ch, k)) return false;
    // Conv1d weight is [out, in, k]; inner = in*k.
    if (in_ch != in.ch) {
        fail("conv1d in_channels", prefix + " weight in=" + std::to_string(in_ch) +
             " signal ch=" + std::to_string(in.ch));
        return false;
    }
    const std::string bk = prefix + ".bias";
    std::vector<float> bias;
    if (w.has_tensor(bk)) {
        bias = w.get_host_vec(bk);
        if ((int)bias.size() != out_ch) {
            fail("conv1d bias size", bk + " got " + std::to_string(bias.size()) +
                 " expected " + std::to_string(out_ch));
            return false;
        }
    } else {
        bias.assign(out_ch, 0.0f);
    }

    int eff_kernel = (k - 1) * dilation + 1;
    int padding_total = eff_kernel - stride;
    if (padding_total < 0) padding_total = 0;
    // _get_extra_padding_for_conv1d (lines 121-133)
    int length = in.T;
    double n_frames_f =
        (double)(length - eff_kernel + padding_total) / (double)stride + 1.0;
    long n_frames = (long)std::ceil(n_frames_f) - 1;
    long ideal_length = n_frames * stride + eff_kernel - padding_total;
    int extra_padding = (int)(ideal_length - length);
    if (extra_padding < 0) extra_padding = 0;
    // non-causal split (lines 159-163)
    int padding_right = padding_total / 2;
    int padding_left = padding_total - padding_right;

    Signal padded;
    if (!pad1d_reflect(in, padding_left, padding_right + extra_padding, padded))
        return false;

    int Tpad = padded.T;
    int Tout = (Tpad - eff_kernel) / stride + 1;
    if (Tout < 0) Tout = 0;
    out.ch = out_ch;
    out.T = Tout;
    out.data.assign((size_t)out_ch * Tout, 0.0f);
    parallel_over_channels(out_ch, [&](int o_lo, int o_hi) {
        for (int o = o_lo; o < o_hi; ++o) {
            const float* wp = &wt[(size_t)o * inner];  // [in, k] for this out chan
            for (int t = 0; t < Tout; ++t) {
                double acc = (double)bias[o];
                int base = t * stride;
                for (int ic = 0; ic < in_ch; ++ic) {
                    const float* wq = &wp[(size_t)ic * k];
                    for (int kk = 0; kk < k; ++kk) {
                        int ti = base + kk * dilation;
                        acc += (double)wq[kk] * (double)padded.at(ic, ti);
                    }
                }
                out.at(o, t) = (float)acc;
            }
        }
    });
    return true;
}

// --- EncodecConvTranspose1d.forward (non-causal, weight_norm) ----------------
bool encodec_conv_transpose1d(Weights& w, const std::string& prefix,
                              const Signal& in, int stride, Signal& out) {
    std::vector<float> wt;
    int in_ch, inner, out_ch, k;
    // ConvTranspose1d weight is [in, out, k]; weight_norm dim=0 -> axis0 = in.
    if (!weight_norm_effective(w, prefix, wt, in_ch, inner, out_ch, k)) return false;
    if (in_ch != in.ch) {
        fail("convT in_channels", prefix + " weight in=" + std::to_string(in_ch) +
             " signal ch=" + std::to_string(in.ch));
        return false;
    }
    const std::string bk = prefix + ".bias";
    std::vector<float> bias;
    if (w.has_tensor(bk)) {
        bias = w.get_host_vec(bk);
        if ((int)bias.size() != out_ch) {
            fail("convT bias size", bk + " got " + std::to_string(bias.size()) +
                 " expected " + std::to_string(out_ch));
            return false;
        }
    } else {
        bias.assign(out_ch, 0.0f);
    }

    // Raw transposed conv output length (no output_padding): (T-1)*stride + k.
    int Tin = in.T;
    int Traw = (Tin - 1) * stride + k;
    if (Traw < 0) Traw = 0;
    std::vector<double> acc((size_t)out_ch * Traw, 0.0);
    for (int o = 0; o < out_ch; ++o)
        for (int t = 0; t < Traw; ++t) acc[(size_t)o * Traw + t] = (double)bias[o];
    // weight[in, out, k]; scatter each input sample across the kernel window.
    // Parallel over o-slices: each thread owns disjoint acc rows. For a fixed
    // (o, t) the contribution order is ic asc, then ti asc, then kk — the SAME
    // order as the original ic-outer nest, so the double-accumulated output is
    // bit-identical to the serial version.
    parallel_over_channels(out_ch, [&](int o_lo, int o_hi) {
        for (int ic = 0; ic < in_ch; ++ic) {
            const float* wp = &wt[(size_t)ic * inner];  // [out, k]
            for (int ti = 0; ti < Tin; ++ti) {
                float xv = in.at(ic, ti);
                if (xv == 0.0f) continue;
                int base = ti * stride;
                for (int o = o_lo; o < o_hi; ++o) {
                    const float* wq = &wp[(size_t)o * k];
                    double* ap = &acc[(size_t)o * Traw];
                    for (int kk = 0; kk < k; ++kk)
                        ap[base + kk] += (double)wq[kk] * (double)xv;
                }
            }
        }
    });

    // Trim (lines 200-208): non-causal -> padding_right = padding_total//2,
    // padding_left = padding_total - padding_right; slice [left : len-right].
    int padding_total = k - stride;
    if (padding_total < 0) padding_total = 0;
    int padding_right = padding_total / 2;
    int padding_left = padding_total - padding_right;
    int end = Traw - padding_right;
    int Tout = end - padding_left;
    if (Tout < 0) Tout = 0;
    out.ch = out_ch;
    out.T = Tout;
    out.data.assign((size_t)out_ch * Tout, 0.0f);
    for (int o = 0; o < out_ch; ++o)
        for (int t = 0; t < Tout; ++t)
            out.at(o, t) = (float)acc[(size_t)o * Traw + (padding_left + t)];
    return true;
}

// --- EncodecLSTM.forward (output = lstm(x) + x, residual) --------------------
// PyTorch LSTM gate order in weight rows: [input(i), forget(f), cell(g),
// output(o)], each gate a contiguous block of `hidden` rows. 2 stacked layers.
// weight_ih_lN: [4*hidden, input], weight_hh_lN: [4*hidden, hidden],
// bias_ih_lN/bias_hh_lN: [4*hidden]. The module operates time-major after
// permute(2,0,1); we iterate time over the signal's T axis. dimension = ch.
bool encodec_lstm(Weights& w, const std::string& prefix, int num_layers,
                  Signal& sig) {
    const int H = sig.ch;     // dimension (input == hidden == 1024)
    const int T = sig.T;
    const int gates = 4 * H;
    // Keep the original input for the final residual add.
    Signal residual = sig;

    // Layer input starts as the signal; subsequent layers consume prior output.
    std::vector<float> layer_in(sig.data);  // [H][T] channel-major
    std::vector<float> layer_out((size_t)H * T, 0.0f);

    for (int l = 0; l < num_layers; ++l) {
        const std::string sl = std::to_string(l);
        std::string ihk = prefix + ".weight_ih_l" + sl;
        std::string hhk = prefix + ".weight_hh_l" + sl;
        std::string bik = prefix + ".bias_ih_l" + sl;
        std::string bhk = prefix + ".bias_hh_l" + sl;
        for (const std::string& kk : {ihk, hhk, bik, bhk})
            if (!w.has_tensor(kk)) { fail("missing lstm weight", kk); return false; }

        std::vector<int> ih_s = w.get_shape(ihk);
        std::vector<int> hh_s = w.get_shape(hhk);
        if (ih_s.size() != 2 || ih_s[0] != gates || ih_s[1] != H) {
            fail("lstm weight_ih shape", ihk + " expected [" +
                 std::to_string(gates) + "," + std::to_string(H) + "]");
            return false;
        }
        if (hh_s.size() != 2 || hh_s[0] != gates || hh_s[1] != H) {
            fail("lstm weight_hh shape", hhk + " expected [" +
                 std::to_string(gates) + "," + std::to_string(H) + "]");
            return false;
        }
        std::vector<float> Wih = w.get_host_vec(ihk);  // [gates, H]
        std::vector<float> Whh = w.get_host_vec(hhk);  // [gates, H]
        std::vector<float> bih = w.get_host_vec(bik);  // [gates]
        std::vector<float> bhh = w.get_host_vec(bhk);  // [gates]
        if ((int)bih.size() != gates || (int)bhh.size() != gates) {
            fail("lstm bias size", bik + "/" + bhk);
            return false;
        }

        std::vector<float> h(H, 0.0f), c(H, 0.0f);
        std::vector<float> g(gates, 0.0f);  // pre-activation gate accumulators
        // HOIST the input projection out of the time loop: gin[t][r] =
        // bih[r] + bhh[r] + Wih[r,:]·x[:,t] has NO recurrence dependency, so
        // it computes for ALL timesteps in one parallel pass. j-outer rank-1
        // form keeps the x row contiguous (layer_in is [H][T]); per-(t,r) the
        // accumulation order (biases first, then j ascending) and the DOUBLE
        // carrier are IDENTICAL to the fused loop → bit-identical output.
        // Halves the serial-critical per-step work (only Whh·h remains).
        std::vector<double> gin((size_t)T * gates);
        parallel_over_channels(gates, [&](int r_lo, int r_hi) {
            std::vector<double> accv((size_t)T);
            for (int r = r_lo; r < r_hi; ++r) {
                const double b0 = (double)bih[r] + (double)bhh[r];
                for (int t = 0; t < T; ++t) accv[t] = b0;
                const float* wr = &Wih[(size_t)r * H];
                for (int j = 0; j < H; ++j) {
                    const double wj = (double)wr[j];
                    const float* xrow = &layer_in[(size_t)j * T];
                    for (int t = 0; t < T; ++t) accv[t] += wj * (double)xrow[t];
                }
                for (int t = 0; t < T; ++t) gin[(size_t)t * gates + r] = accv[t];
            }
        });
        for (int t = 0; t < T; ++t) {
            // g = gin[t] + Whh * h_{t-1} — parallel over gate rows (disjoint
            // g[r]; order unchanged → bit-identical). The h/c recurrence
            // below stays serial — it needs the full gate vector of THIS step.
            parallel_over_channels(gates, [&](int r_lo, int r_hi) {
                for (int r = r_lo; r < r_hi; ++r) {
                    double acc = gin[(size_t)t * gates + r];
                    const float* ur = &Whh[(size_t)r * H];
                    for (int j = 0; j < H; ++j)
                        acc += (double)ur[j] * (double)h[j];
                    g[r] = (float)acc;
                }
            });
            // gate slices: i=[0,H) f=[H,2H) g_=[2H,3H) o=[3H,4H)
            for (int j = 0; j < H; ++j) {
                double ig = 1.0 / (1.0 + std::exp(-(double)g[j]));
                double fg = 1.0 / (1.0 + std::exp(-(double)g[H + j]));
                double gg = std::tanh((double)g[2 * H + j]);
                double og = 1.0 / (1.0 + std::exp(-(double)g[3 * H + j]));
                double cn = fg * (double)c[j] + ig * gg;
                c[j] = (float)cn;
                double hn = og * std::tanh(cn);
                h[j] = (float)hn;
                layer_out[(size_t)j * T + t] = (float)hn;
            }
        }
        layer_in = layer_out;  // feed this layer's output into the next layer
    }

    // residual add: output = lstm(x) + x
    for (int cc = 0; cc < H; ++cc)
        for (int t = 0; t < T; ++t)
            sig.at(cc, t) = layer_in[(size_t)cc * T + t] + residual.at(cc, t);
    return true;
}

// --- EncodecResnetBlock.forward (num_residual_layers=1, Identity shortcut) ---
// block = [ELU, Conv1d(dim->dim/2,k=3,dil=1), ELU, Conv1d(dim/2->dim,k=1,dil=1)]
// out = residual + block(x). block.1 and block.3 are the two convs.
bool encodec_resnet_block(Weights& w, const std::string& layer_prefix,
                          const Signal& in, Signal& out) {
    Signal residual = in;
    Signal h = in;
    elu_inplace(h);  // block.0 ELU
    Signal h1;
    if (!encodec_conv1d(w, layer_prefix + ".block.1.conv", h, /*stride*/ 1,
                        /*dilation*/ 1, h1))
        return false;
    elu_inplace(h1);  // block.2 ELU
    Signal h2;
    if (!encodec_conv1d(w, layer_prefix + ".block.3.conv", h1, /*stride*/ 1,
                        /*dilation*/ 1, h2))
        return false;
    // shortcut is Identity (use_conv_shortcut=False)
    if (h2.ch != residual.ch || h2.T != residual.T) {
        fail("resnet residual shape", layer_prefix + " conv out [" +
             std::to_string(h2.ch) + "," + std::to_string(h2.T) +
             "] vs residual [" + std::to_string(residual.ch) + "," +
             std::to_string(residual.T) + "]");
        return false;
    }
    out.ch = h2.ch;
    out.T = h2.T;
    out.data.assign(h2.data.size(), 0.0f);
    for (size_t i = 0; i < h2.data.size(); ++i)
        out.data[i] = residual.data[i] + h2.data[i];
    return true;
}

}  // namespace

std::vector<float> encodec_decode_host(
    Weights& weights, const std::vector<std::vector<int32_t>>& codes) {
    g_error = false;
    const int kCodebooks = 4;
    const int kCodebookSize = 2048;
    const int kCodebookDim = 128;  // == hidden_size

    if ((int)codes.size() != kCodebooks) {
        fail("codes num_codebooks", "expected " + std::to_string(kCodebooks) +
             " got " + std::to_string(codes.size()));
        return {};
    }
    int T = (int)codes[0].size();
    if (T <= 0) { fail("codes T_frames", "empty code stream"); return {}; }
    for (int k = 0; k < kCodebooks; ++k) {
        if ((int)codes[k].size() != T) {
            fail("codes ragged", "codebook " + std::to_string(k) + " has " +
                 std::to_string(codes[k].size()) + " frames, expected " +
                 std::to_string(T));
            return {};
        }
    }

    // -------------------------------------------------------------------------
    // RVQ decode: latent[128][T] = sum_k embed_k[ codes[k][t] ]  (lines 356-415)
    // -------------------------------------------------------------------------
    Signal latent;
    latent.ch = kCodebookDim;
    latent.T = T;
    latent.data.assign((size_t)kCodebookDim * T, 0.0f);
    for (int k = 0; k < kCodebooks; ++k) {
        std::string ek = "audio_encoder.quantizer.layers." + std::to_string(k) +
                         ".codebook.embed";
        if (!weights.has_tensor(ek)) { fail("missing codebook", ek); return {}; }
        std::vector<int> es = weights.get_shape(ek);
        if (es.size() != 2 || es[0] != kCodebookSize || es[1] != kCodebookDim) {
            fail("codebook shape", ek + " expected [" +
                 std::to_string(kCodebookSize) + "," +
                 std::to_string(kCodebookDim) + "]");
            return {};
        }
        std::vector<float> embed = weights.get_host_vec(ek);  // [2048,128]
        for (int t = 0; t < T; ++t) {
            int32_t id = codes[k][t];
            if (id < 0 || id >= kCodebookSize) {
                fail("code id range", "codebook " + std::to_string(k) +
                     " frame " + std::to_string(t) + " id=" +
                     std::to_string(id));
                return {};
            }
            const float* row = &embed[(size_t)id * kCodebookDim];
            for (int d = 0; d < kCodebookDim; ++d)
                latent.at(d, t) += row[d];  // permute(0,2,1) -> [dim][T]
        }
    }

    // -------------------------------------------------------------------------
    // Decoder module list (lines 310-334). See layer map at top of file.
    // -------------------------------------------------------------------------
    const std::string D = "audio_encoder.decoder.layers.";
    Signal x;

    // [0] Conv1d(128->1024, k=7, s=1)
    if (!encodec_conv1d(weights, D + "0.conv", latent, 1, 1, x)) return {};
    // [1] LSTM(1024, 2 layers, residual)
    if (!encodec_lstm(weights, D + "1.lstm", /*num_layers*/ 2, x)) return {};

    struct Stage { int convt_idx; int resnet_idx; int stride; };
    // ratios [8,5,4,4] -> strides; kernel = ratio*2 (derived inside convT).
    const Stage stages[4] = {{3, 4, 8}, {6, 7, 5}, {9, 10, 4}, {12, 13, 4}};
    for (const Stage& s : stages) {
        // ELU
        elu_inplace(x);
        // ConvTranspose1d
        Signal up;
        if (!encodec_conv_transpose1d(weights, D + std::to_string(s.convt_idx) +
                                      ".conv", x, s.stride, up))
            return {};
        // ResnetBlock
        Signal rb;
        if (!encodec_resnet_block(weights, D + std::to_string(s.resnet_idx), up,
                                  rb))
            return {};
        x = rb;
    }

    // [14] ELU
    elu_inplace(x);
    // [15] Conv1d(64->1, k=7, s=1)
    Signal out;
    if (!encodec_conv1d(weights, D + "15.conv", x, 1, 1, out)) return {};

    if (out.ch != 1) {
        fail("final channels", "expected 1 got " + std::to_string(out.ch));
        return {};
    }
    if (g_error) return {};

    std::vector<float> pcm(out.data.begin(), out.data.end());
    // Soft clamp to [-1,1] (waveform may slightly exceed due to fp).
    for (float& v : pcm) {
        if (v > 1.0f) v = 1.0f;
        else if (v < -1.0f) v = -1.0f;
    }
    return pcm;
}

// =============================================================================
// GPU SEANet decode (Phase A) — conv stack on the GPU via kernels/encodec.cl,
// LSTM on the CPU. Orchestration and padding/trim math REPLICATE the host
// functions above; the per-element accumulation is fp32 (host: fp64) and the
// convT is gather-form (host: scatter), so the PCM is validated by cosine
// against encodec_decode_host, not byte equality.
// =============================================================================

namespace {

struct EncGpu {
    cl_program prog = nullptr;
    cl_kernel k_pad = nullptr, k_conv = nullptr, k_convt = nullptr;
    cl_kernel k_conv_x = nullptr, k_convt_x = nullptr, k_xt = nullptr;   // transposed-x variants (NNOPT_ENC_TILE)
    cl_kernel k_elu = nullptr, k_elu_oop = nullptr, k_add = nullptr;
    cl_kernel k_lstm_gin = nullptr, k_lstm_gin_x = nullptr, k_lstm_gates = nullptr, k_lstm_cell = nullptr;
    bool ready = false;
};
EncGpu g_enc;

bool enc_gpu_init(OpenCLContext& cl_ctx) {
    if (g_enc.ready) return true;
    g_enc.prog = cl_ctx.build_program_from_file("kernels/encodec.cl");
    if (!g_enc.prog) { fail("enc-gpu", "build kernels/encodec.cl failed"); return false; }
    cl_int err = CL_SUCCESS;
    auto mk = [&](const char* nm) -> cl_kernel {
        cl_kernel k = clCreateKernel(g_enc.prog, nm, &err);
        if (err != CL_SUCCESS) { fail("enc-gpu kernel", nm); return nullptr; }
        return k;
    };
    g_enc.k_pad = mk("enc_pad_reflect"); g_enc.k_conv = mk("enc_conv1d");
    g_enc.k_convt = mk("enc_convt1d");   g_enc.k_elu = mk("enc_elu");
    g_enc.k_conv_x = mk("enc_conv1d_x4");
    g_enc.k_convt_x = mk("enc_convt1d_x4");
    g_enc.k_xt = mk("enc_xt");
    g_enc.k_elu_oop = mk("enc_elu_oop"); g_enc.k_add = mk("enc_add");
    g_enc.k_lstm_gin = mk("enc_lstm_gin");
    g_enc.k_lstm_gin_x = mk("enc_lstm_gin_x");
    g_enc.k_lstm_gates = mk("enc_lstm_gates");
    g_enc.k_lstm_cell = mk("enc_lstm_cell");
    g_enc.ready = g_enc.k_pad && g_enc.k_conv && g_enc.k_convt &&
                  g_enc.k_elu && g_enc.k_elu_oop && g_enc.k_add &&
                  g_enc.k_lstm_gin && g_enc.k_lstm_gates && g_enc.k_lstm_cell;
    return g_enc.ready;
}

// GPU-resident signal (fp32, channel-major like the host Signal).
struct GSig { cl_mem mem = nullptr; int ch = 0; int T = 0; };

// Size-keyed buffer pool: the SEANet stages allocate+release ~250 MB of
// signal buffers per generation; in serve mode that churn re-poisons the
// driver pool and the NEXT generation's first-step allocations stall again
// (the layers-10-13 pattern). Stage sizes are deterministic per clip length,
// so gen 2+ recycles every buffer exactly. Gated with the weight cache
// (NNOPT_ENC_WCACHE=0 → plain alloc/release).
static std::multimap<size_t, cl_mem> g_enc_pool;
static bool enc_wcache_on();   // fwd decl (defined with the weight cache below)

cl_mem enc_buf(OpenCLContext& cl_ctx, size_t floats) {
    const size_t bytes = floats * sizeof(float);
    if (enc_wcache_on()) {
        auto it = g_enc_pool.find(bytes);
        if (it != g_enc_pool.end()) {
            cl_mem m = it->second;
            g_enc_pool.erase(it);
            return m;
        }
    }
    cl_int err = CL_SUCCESS;
    cl_mem m = clCreateBuffer(cl_ctx.context(), CL_MEM_READ_WRITE, bytes, nullptr, &err);
    if (err != CL_SUCCESS) { fail("enc-gpu", "buffer alloc"); return nullptr; }
    return m;
}

// ── Persistent codec-weight cache (serve mode) ──────────────────────────────
// encodec_decode_gpu recomputed weight_norm on the HOST and re-uploaded
// ~118 MB of conv/LSTM weights EVERY generation, then released them — pure
// recompute cost plus driver-pool churn that degraded the NEXT generation's
// first-step allocations (measured: serve gen-2 first_step regressed 2.4 →
// 4.9 s). Entries are keyed by layer prefix and live for the process.
// NNOPT_ENC_WCACHE=0 disables (single-shot behavior unchanged either way).
struct EncWEntry { cl_mem w = nullptr, b = nullptr; int d0 = 0, d1 = 0, kk = 0; };
static std::map<std::string, EncWEntry> g_enc_wcache;
static bool enc_wcache_on() {
    static const bool on = [](){ const char* e = std::getenv("NNOPT_ENC_WCACHE"); return !(e && e[0]=='0'); }();
    return on;
}

cl_mem enc_upload(OpenCLContext& cl_ctx, const std::vector<float>& v, const char* what) {
    cl_int err = CL_SUCCESS;
    cl_mem m = clCreateBuffer(cl_ctx.context(), CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
                              v.size() * sizeof(float), (void*)v.data(), &err);
    if (err != CL_SUCCESS) { fail("enc-gpu upload", what); return nullptr; }
    return m;
}

void enc_release(cl_mem& m) {
    if (!m) return;
    if (enc_wcache_on()) {
        size_t bytes = 0;
        if (clGetMemObjectInfo(m, CL_MEM_SIZE, sizeof(bytes), &bytes, nullptr) == CL_SUCCESS && bytes) {
            g_enc_pool.insert({bytes, m});
            m = nullptr;
            return;
        }
    }
    clReleaseMemObject(m); m = nullptr;
}

bool enc_load_bias(Weights& w, const std::string& prefix, int out_ch, std::vector<float>& bias) {
    const std::string bk = prefix + ".bias";
    if (w.has_tensor(bk)) {
        bias = w.get_host_vec(bk);
        if ((int)bias.size() != out_ch) { fail("enc-gpu bias size", bk); return false; }
    } else {
        bias.assign(out_ch, 0.0f);
    }
    return true;
}

// GPU Conv1d with reflect pad — mirrors encodec_conv1d's padding math exactly.
bool enc_gpu_conv1d(OpenCLContext& cl_ctx, Weights& w, const std::string& prefix,
                    const GSig& in, int stride, int dilation, GSig& out) {
    std::vector<float> wt;
    int out_ch = 0, inner = 0, in_ch = 0, k = 0;
    cl_mem cw = nullptr, cb = nullptr; bool from_cache = false;
    if (enc_wcache_on()) {
        auto it = g_enc_wcache.find("c:" + prefix);
        if (it != g_enc_wcache.end()) {
            cw = it->second.w; cb = it->second.b;
            out_ch = it->second.d0; in_ch = it->second.d1; k = it->second.kk;
            from_cache = true;
        }
    }
    std::vector<float> bias;
    if (!from_cache) {
        if (!weight_norm_effective(w, prefix, wt, out_ch, inner, in_ch, k)) return false;
        if (!enc_load_bias(w, prefix, out_ch, bias)) return false;
    }
    if (in_ch != in.ch) { fail("enc-gpu conv1d in_channels", prefix); return false; }

    int eff_kernel = (k - 1) * dilation + 1;
    int padding_total = eff_kernel - stride;
    if (padding_total < 0) padding_total = 0;
    int length = in.T;
    double n_frames_f = (double)(length - eff_kernel + padding_total) / (double)stride + 1.0;
    long n_frames = (long)std::ceil(n_frames_f) - 1;
    long ideal_length = n_frames * stride + eff_kernel - padding_total;
    int extra_padding = (int)(ideal_length - length);
    if (extra_padding < 0) extra_padding = 0;
    int padding_right = padding_total / 2;
    int padding_left = padding_total - padding_right;
    // pad1d_reflect param math (extra-zero-pad quirk):
    int pl = padding_left, pr = padding_right + extra_padding;
    int max_pad = pl > pr ? pl : pr;
    int extra_pad = 0, work_len = length;
    if (length <= max_pad) { extra_pad = max_pad - length + 1; work_len = length + extra_pad; }
    int padded_len = pl + work_len + pr;
    int Tpad = padded_len - extra_pad;   // host trims the inserted zeros' tail

    cl_mem wbuf = from_cache ? cw : enc_upload(cl_ctx, wt, prefix.c_str());
    cl_mem bbuf = from_cache ? cb : enc_upload(cl_ctx, bias, prefix.c_str());
    cl_mem pad = enc_buf(cl_ctx, (size_t)in.ch * Tpad);
    if (!wbuf || !bbuf || !pad) return false;
    cl_command_queue q = cl_ctx.queue();
    cl_int err = CL_SUCCESS;
    err |= clSetKernelArg(g_enc.k_pad, 0, sizeof(cl_mem), &in.mem);
    err |= clSetKernelArg(g_enc.k_pad, 1, sizeof(cl_mem), &pad);
    err |= clSetKernelArg(g_enc.k_pad, 2, sizeof(int), &in.T);
    err |= clSetKernelArg(g_enc.k_pad, 3, sizeof(int), &Tpad);
    err |= clSetKernelArg(g_enc.k_pad, 4, sizeof(int), &pl);
    err |= clSetKernelArg(g_enc.k_pad, 5, sizeof(int), &work_len);
    err |= clSetKernelArg(g_enc.k_pad, 6, sizeof(int), &length);
    size_t gpad[2] = {(size_t)((Tpad + 63) / 64) * 64, (size_t)in.ch};
    size_t lpad[2] = {64, 1};
    if (err == CL_SUCCESS)
        err = clEnqueueNDRangeKernel(q, g_enc.k_pad, 2, nullptr, gpad, lpad, 0, nullptr, KernelProfiler::event_for("enc_pad"));
    if (err != CL_SUCCESS) { fail("enc-gpu pad", prefix); return false; }

    int Tout = (Tpad - eff_kernel) / stride + 1;
    if (Tout < 0) Tout = 0;
    out.ch = out_ch; out.T = Tout;
    out.mem = enc_buf(cl_ctx, (size_t)out_ch * Tout);
    if (!out.mem) return false;
    err  = clSetKernelArg(g_enc.k_conv, 0, sizeof(cl_mem), &pad);
    err |= clSetKernelArg(g_enc.k_conv, 1, sizeof(cl_mem), &wbuf);
    err |= clSetKernelArg(g_enc.k_conv, 2, sizeof(cl_mem), &bbuf);
    err |= clSetKernelArg(g_enc.k_conv, 3, sizeof(cl_mem), &out.mem);
    err |= clSetKernelArg(g_enc.k_conv, 4, sizeof(int), &in_ch);
    err |= clSetKernelArg(g_enc.k_conv, 5, sizeof(int), &Tpad);
    err |= clSetKernelArg(g_enc.k_conv, 6, sizeof(int), &Tout);
    err |= clSetKernelArg(g_enc.k_conv, 7, sizeof(int), &k);
    err |= clSetKernelArg(g_enc.k_conv, 8, sizeof(int), &stride);
    err |= clSetKernelArg(g_enc.k_conv, 9, sizeof(int), &dilation);
    // Transposed-x route (enc_conv1d_x): pad buffer transposed to [Tpad,in_ch]
    // + W transposed to [out_ch,k,in_ch] so every inner read is a contiguous
    // vload4 (the untiled kernel column-gathers x: one cache line per float).
    // ACCUMULATION ORDER CHANGES (kk-outer) -> gated on PCM cosine, not bytes.
    // NNOPT_ENC_TILE: 1/unset=both x-routes, 0=off, c=conv only, t=convT only.
    static const char tile_mode = [](){ const char* e = std::getenv("NNOPT_ENC_TILE"); return e ? e[0] : 't'; }();   // default 't': convT-x only (bit-identical, -0.5 s); conv-x measured SLOWER
    const bool xroute = (tile_mode == '1' || tile_mode == 'c') && g_enc.k_conv_x && g_enc.k_xt && (in_ch % 4) == 0 && !from_cache;
    cl_int errk = CL_SUCCESS;
    cl_mem padx = nullptr; cl_mem wtb = nullptr;
    if (xroute && err == CL_SUCCESS) {
        // W [out,in,k] -> [out,k,in]
        std::vector<float> wt_t((size_t)out_ch * k * in_ch);
        for (int o = 0; o < out_ch; ++o)
            for (int ic = 0; ic < in_ch; ++ic)
                for (int kk = 0; kk < k; ++kk)
                    wt_t[((size_t)o * k + kk) * in_ch + ic] = wt[((size_t)o * in_ch + ic) * k + kk];
        wtb = enc_upload(cl_ctx, wt_t, prefix.c_str());
        padx = enc_buf(cl_ctx, (size_t)Tpad * in_ch);
        if (wtb && padx) {
            errk  = clSetKernelArg(g_enc.k_xt, 0, sizeof(cl_mem), &pad);
            errk |= clSetKernelArg(g_enc.k_xt, 1, sizeof(cl_mem), &padx);
            errk |= clSetKernelArg(g_enc.k_xt, 2, sizeof(int), &in_ch);
            errk |= clSetKernelArg(g_enc.k_xt, 3, sizeof(int), &Tpad);
            size_t gx[2] = {(size_t)((Tpad + 63) / 64) * 64, (size_t)in_ch};
            size_t lx[2] = {64, 1};
            if (errk == CL_SUCCESS)
                errk = clEnqueueNDRangeKernel(q, g_enc.k_xt, 2, nullptr, gx, lx, 0, nullptr, KernelProfiler::event_for("enc_xt"));
            if (errk == CL_SUCCESS) {
                errk  = clSetKernelArg(g_enc.k_conv_x, 0, sizeof(cl_mem), &padx);
                errk |= clSetKernelArg(g_enc.k_conv_x, 1, sizeof(cl_mem), &wtb);
                errk |= clSetKernelArg(g_enc.k_conv_x, 2, sizeof(cl_mem), &bbuf);
                errk |= clSetKernelArg(g_enc.k_conv_x, 3, sizeof(cl_mem), &out.mem);
                errk |= clSetKernelArg(g_enc.k_conv_x, 4, sizeof(int), &in_ch);
                errk |= clSetKernelArg(g_enc.k_conv_x, 5, sizeof(int), &Tpad);
                errk |= clSetKernelArg(g_enc.k_conv_x, 6, sizeof(int), &Tout);
                errk |= clSetKernelArg(g_enc.k_conv_x, 7, sizeof(int), &k);
                errk |= clSetKernelArg(g_enc.k_conv_x, 8, sizeof(int), &stride);
                errk |= clSetKernelArg(g_enc.k_conv_x, 9, sizeof(int), &dilation);
                const size_t tb = ((size_t)Tout + 3) / 4;
                size_t gws[2] = {((tb + 63) / 64) * 64, (size_t)out_ch};
                size_t lws[2] = {64, 1};
                if (errk == CL_SUCCESS)
                    errk = clEnqueueNDRangeKernel(q, g_enc.k_conv_x, 2, nullptr, gws, lws, 0, nullptr, KernelProfiler::event_for("enc_conv1d"));
            }
        } else errk = CL_OUT_OF_RESOURCES;
    }
    if (!xroute || errk != CL_SUCCESS) {
        if (xroute) fprintf(stderr, "enc-gpu: conv1d_x failed (%d) - untiled fallback\n", (int)errk);
        size_t gws[2] = {(size_t)((Tout + 63) / 64) * 64, (size_t)out_ch};
        size_t lws[2] = {64, 1};
        if (err == CL_SUCCESS)
            err = clEnqueueNDRangeKernel(q, g_enc.k_conv, 2, nullptr, gws, lws, 0, nullptr, KernelProfiler::event_for("enc_conv1d"));
    }
    if (padx) enc_release(padx);
    if (wtb) enc_release(wtb);
    if (enc_wcache_on()) {
        if (!from_cache) g_enc_wcache["c:" + prefix] = {wbuf, bbuf, out_ch, in_ch, k};
    } else { enc_release(wbuf); enc_release(bbuf); }
    enc_release(pad);
    if (err != CL_SUCCESS) { fail("enc-gpu conv1d", prefix); return false; }
    return true;
}

// GPU ConvTranspose1d (gather + trim) — mirrors encodec_conv_transpose1d.
bool enc_gpu_convt1d(OpenCLContext& cl_ctx, Weights& w, const std::string& prefix,
                     const GSig& in, int stride, GSig& out) {
    std::vector<float> wt;
    int in_ch = 0, inner = 0, out_ch = 0, k = 0;
    cl_mem cw = nullptr, cb = nullptr; bool from_cache = false;
    if (enc_wcache_on()) {
        auto it = g_enc_wcache.find("t:" + prefix);
        if (it != g_enc_wcache.end()) {
            cw = it->second.w; cb = it->second.b;
            in_ch = it->second.d0; out_ch = it->second.d1; k = it->second.kk;
            from_cache = true;
        }
    }
    std::vector<float> bias;
    if (!from_cache) {
        if (!weight_norm_effective(w, prefix, wt, in_ch, inner, out_ch, k)) return false;
        if (!enc_load_bias(w, prefix, out_ch, bias)) return false;
    }
    if (in_ch != in.ch) { fail("enc-gpu convT in_channels", prefix); return false; }

    int Tin = in.T;
    int Traw = (Tin - 1) * stride + k;
    if (Traw < 0) Traw = 0;
    int padding_total = k - stride;
    if (padding_total < 0) padding_total = 0;
    int padding_right = padding_total / 2;
    int padding_left = padding_total - padding_right;
    int end = Traw - padding_right;
    int Tout = end - padding_left;
    if (Tout < 0) Tout = 0;

    // Transpose [in, out, k] → [out, k, in] so the kernel's inner ic loop is
    // contiguous (the PyTorch layout strode 32 KB between ic reads — 19 s/clip).
    cl_mem wbuf = cw, bbuf = cb;
    if (!from_cache) {
        std::vector<float> wt_t((size_t)out_ch * k * in_ch);
        parallel_over_channels(out_ch, [&](int o_lo, int o_hi) {
            for (int o = o_lo; o < o_hi; ++o)
                for (int ic = 0; ic < in_ch; ++ic)
                    for (int kk = 0; kk < k; ++kk)
                        wt_t[((size_t)o * k + kk) * in_ch + ic] = wt[((size_t)ic * out_ch + o) * k + kk];
        });
        wbuf = enc_upload(cl_ctx, wt_t, prefix.c_str());
        bbuf = enc_upload(cl_ctx, bias, prefix.c_str());
    }
    if (!wbuf || !bbuf) return false;
    out.ch = out_ch; out.T = Tout;
    out.mem = enc_buf(cl_ctx, (size_t)out_ch * Tout);
    if (!out.mem) return false;
    cl_int err = CL_SUCCESS;
    err  = clSetKernelArg(g_enc.k_convt, 0, sizeof(cl_mem), &in.mem);
    err |= clSetKernelArg(g_enc.k_convt, 1, sizeof(cl_mem), &wbuf);
    err |= clSetKernelArg(g_enc.k_convt, 2, sizeof(cl_mem), &bbuf);
    err |= clSetKernelArg(g_enc.k_convt, 3, sizeof(cl_mem), &out.mem);
    err |= clSetKernelArg(g_enc.k_convt, 4, sizeof(int), &in_ch);
    err |= clSetKernelArg(g_enc.k_convt, 5, sizeof(int), &Tin);
    err |= clSetKernelArg(g_enc.k_convt, 6, sizeof(int), &out_ch);
    err |= clSetKernelArg(g_enc.k_convt, 7, sizeof(int), &Tout);
    err |= clSetKernelArg(g_enc.k_convt, 8, sizeof(int), &k);
    err |= clSetKernelArg(g_enc.k_convt, 9, sizeof(int), &stride);
    err |= clSetKernelArg(g_enc.k_convt, 10, sizeof(int), &padding_left);
    static const char tile_mode_t = [](){ const char* e = std::getenv("NNOPT_ENC_TILE"); return e ? e[0] : 't'; }();
    // Transposed-x route (enc_convt1d_x): x transposed to [Tin,in_ch] once
    // (elementwise) so the a4 loop's x reads are contiguous vload4s. Loop
    // structure and accumulation order IDENTICAL to enc_convt1d -> output
    // bit-identical (same-seed wav byte gate applies).
    const bool xroute = (tile_mode_t == '1' || tile_mode_t == 't') && g_enc.k_convt_x && g_enc.k_xt && (in_ch % 4) == 0;
    cl_int errk = CL_SUCCESS;
    cl_mem xtb = nullptr;
    if (xroute) {
        xtb = enc_buf(cl_ctx, (size_t)Tin * in_ch);
        if (xtb) {
            errk  = clSetKernelArg(g_enc.k_xt, 0, sizeof(cl_mem), &in.mem);
            errk |= clSetKernelArg(g_enc.k_xt, 1, sizeof(cl_mem), &xtb);
            errk |= clSetKernelArg(g_enc.k_xt, 2, sizeof(int), &in_ch);
            errk |= clSetKernelArg(g_enc.k_xt, 3, sizeof(int), &Tin);
            size_t gx[2] = {(size_t)((Tin + 63) / 64) * 64, (size_t)in_ch};
            size_t lx[2] = {64, 1};
            if (errk == CL_SUCCESS)
                errk = clEnqueueNDRangeKernel(cl_ctx.queue(), g_enc.k_xt, 2, nullptr, gx, lx, 0, nullptr, KernelProfiler::event_for("enc_xt"));
            if (errk == CL_SUCCESS) {
                cl_kernel kt = g_enc.k_convt_x;
                errk  = clSetKernelArg(kt, 0, sizeof(cl_mem), &xtb);
                errk |= clSetKernelArg(kt, 1, sizeof(cl_mem), &wbuf);
                errk |= clSetKernelArg(kt, 2, sizeof(cl_mem), &bbuf);
                errk |= clSetKernelArg(kt, 3, sizeof(cl_mem), &out.mem);
                errk |= clSetKernelArg(kt, 4, sizeof(int), &in_ch);
                errk |= clSetKernelArg(kt, 5, sizeof(int), &Tin);
                errk |= clSetKernelArg(kt, 6, sizeof(int), &out_ch);
                errk |= clSetKernelArg(kt, 7, sizeof(int), &Tout);
                errk |= clSetKernelArg(kt, 8, sizeof(int), &k);
                errk |= clSetKernelArg(kt, 9, sizeof(int), &stride);
                errk |= clSetKernelArg(kt, 10, sizeof(int), &padding_left);
                const size_t tbt = ((size_t)Tout + 3) / 4;
                size_t gws[2] = {((tbt + 63) / 64) * 64, (size_t)out_ch};
                size_t lws[2] = {64, 1};
                if (errk == CL_SUCCESS)
                    errk = clEnqueueNDRangeKernel(cl_ctx.queue(), kt, 2, nullptr, gws, lws, 0, nullptr, KernelProfiler::event_for("enc_convt1d"));
            }
        } else errk = CL_OUT_OF_RESOURCES;
    }
    if (!xroute || errk != CL_SUCCESS) {
        if (xroute) fprintf(stderr, "enc-gpu: convt1d_x failed (%d) - untiled fallback\n", (int)errk);
        size_t gws[2] = {(size_t)((Tout + 63) / 64) * 64, (size_t)out_ch};
        size_t lws[2] = {64, 1};
        if (err == CL_SUCCESS)
            err = clEnqueueNDRangeKernel(cl_ctx.queue(), g_enc.k_convt, 2, nullptr, gws, lws, 0, nullptr, KernelProfiler::event_for("enc_convt1d"));
    }
    if (xtb) enc_release(xtb);
    if (enc_wcache_on()) {
        if (!from_cache) g_enc_wcache["t:" + prefix] = {wbuf, bbuf, in_ch, out_ch, k};
    } else { enc_release(wbuf); enc_release(bbuf); }
    if (err != CL_SUCCESS) { fail("enc-gpu convT", prefix); return false; }
    return true;
}

bool enc_gpu_elu(OpenCLContext& cl_ctx, GSig& s) {
    const int n = s.ch * s.T;
    cl_int err = CL_SUCCESS;
    err |= clSetKernelArg(g_enc.k_elu, 0, sizeof(cl_mem), &s.mem);
    err |= clSetKernelArg(g_enc.k_elu, 1, sizeof(int), &n);
    size_t gws[1] = {(size_t)((n + 63) / 64) * 64}; size_t lws[1] = {64};
    if (err == CL_SUCCESS)
        err = clEnqueueNDRangeKernel(cl_ctx.queue(), g_enc.k_elu, 1, nullptr, gws, lws, 0, nullptr, KernelProfiler::event_for("enc_elu"));
    if (err != CL_SUCCESS) { fail("enc-gpu", "elu"); return false; }
    return true;
}

// ResnetBlock: out = in + conv_k1(elu(conv_k3(elu(in)))). Mirrors
// encodec_resnet_block (Identity shortcut).
bool enc_gpu_resnet(OpenCLContext& cl_ctx, Weights& w, const std::string& layer_prefix,
                    GSig& x /* in-place result */) {
    const int n = x.ch * x.T;
    GSig h{enc_buf(cl_ctx, (size_t)n), x.ch, x.T};
    if (!h.mem) return false;
    cl_int err = CL_SUCCESS;
    err |= clSetKernelArg(g_enc.k_elu_oop, 0, sizeof(cl_mem), &x.mem);
    err |= clSetKernelArg(g_enc.k_elu_oop, 1, sizeof(cl_mem), &h.mem);
    err |= clSetKernelArg(g_enc.k_elu_oop, 2, sizeof(int), &n);
    size_t gws[1] = {(size_t)((n + 63) / 64) * 64}; size_t lws[1] = {64};
    if (err == CL_SUCCESS)
        err = clEnqueueNDRangeKernel(cl_ctx.queue(), g_enc.k_elu_oop, 1, nullptr, gws, lws, 0, nullptr, KernelProfiler::event_for("enc_elu_oop"));
    if (err != CL_SUCCESS) { fail("enc-gpu", "resnet elu0"); enc_release(h.mem); return false; }
    GSig h1;
    if (!enc_gpu_conv1d(cl_ctx, w, layer_prefix + ".block.1.conv", h, 1, 1, h1)) { enc_release(h.mem); return false; }
    enc_release(h.mem);
    if (!enc_gpu_elu(cl_ctx, h1)) { enc_release(h1.mem); return false; }
    GSig h2;
    if (!enc_gpu_conv1d(cl_ctx, w, layer_prefix + ".block.3.conv", h1, 1, 1, h2)) { enc_release(h1.mem); return false; }
    enc_release(h1.mem);
    if (h2.ch != x.ch || h2.T != x.T) { fail("enc-gpu resnet shape", layer_prefix); enc_release(h2.mem); return false; }
    const int n2 = h2.ch * h2.T;
    err  = clSetKernelArg(g_enc.k_add, 0, sizeof(cl_mem), &h2.mem);
    err |= clSetKernelArg(g_enc.k_add, 1, sizeof(cl_mem), &x.mem);
    err |= clSetKernelArg(g_enc.k_add, 2, sizeof(int), &n2);
    size_t gws2[1] = {(size_t)((n2 + 63) / 64) * 64};
    if (err == CL_SUCCESS)
        err = clEnqueueNDRangeKernel(cl_ctx.queue(), g_enc.k_add, 1, nullptr, gws2, lws, 0, nullptr, KernelProfiler::event_for("enc_add"));
    if (err != CL_SUCCESS) { fail("enc-gpu", "resnet add"); enc_release(h2.mem); return false; }
    enc_release(x.mem);
    x = h2;   // x := residual-added block output
    return true;
}

}  // namespace

std::vector<float> encodec_decode_gpu(OpenCLContext& cl_ctx, Weights& weights,
                                      const std::vector<std::vector<int32_t>>& codes) {
    g_error = false;
    const int kCodebooks = 4, kCodebookSize = 2048, kCodebookDim = 128;
    if ((int)codes.size() != kCodebooks) { fail("enc-gpu codes", "num_codebooks"); return {}; }
    int T = (int)codes[0].size();
    if (T <= 0) { fail("enc-gpu codes", "empty"); return {}; }
    for (int k = 0; k < kCodebooks; ++k)
        if ((int)codes[k].size() != T) { fail("enc-gpu codes", "ragged"); return {}; }
    const auto ph_t0 = std::chrono::steady_clock::now();
    auto ph_mark = [&](const char* nm) {
        static thread_local auto last = ph_t0;
        auto now = std::chrono::steady_clock::now();
        fprintf(stderr, "ENC_GPU_PHASE %s: %.3f s\n", nm,
                std::chrono::duration<double>(now - last).count());
        last = now;
    };
    if (!enc_gpu_init(cl_ctx)) return {};
    ph_mark("compile");
    cl_command_queue q = cl_ctx.queue();

    // RVQ on host (table lookups; trivial), then upload the latent.
    Signal latent;
    latent.ch = kCodebookDim; latent.T = T;
    latent.data.assign((size_t)kCodebookDim * T, 0.0f);
    for (int k = 0; k < kCodebooks; ++k) {
        std::string ek = "audio_encoder.quantizer.layers." + std::to_string(k) + ".codebook.embed";
        if (!weights.has_tensor(ek)) { fail("enc-gpu codebook", ek); return {}; }
        std::vector<float> embed = weights.get_host_vec(ek);
        for (int t = 0; t < T; ++t) {
            int32_t id = codes[k][t];
            if (id < 0 || id >= kCodebookSize) { fail("enc-gpu code id", ek); return {}; }
            const float* row = &embed[(size_t)id * kCodebookDim];
            for (int d = 0; d < kCodebookDim; ++d) latent.at(d, t) += row[d];
        }
    }
    GSig x{enc_upload(cl_ctx, latent.data, "latent"), latent.ch, latent.T};
    if (!x.mem) return {};
    ph_mark("rvq+upload");

    const std::string D = "audio_encoder.decoder.layers.";
    // [0] Conv1d(128->1024, k7) on GPU
    GSig x0;
    if (!enc_gpu_conv1d(cl_ctx, weights, D + "0.conv", x, 1, 1, x0)) { enc_release(x.mem); return {}; }
    enc_release(x.mem); x = x0;

    // [1] LSTM — Phase B: on the GPU (NNOPT_ENC_LSTM_GPU=0 reverts to the CPU
    // path). gin batched over all t in one dispatch; the recurrence is 2 small
    // async dispatches per step (gates GEMV + cell), serialized by the
    // in-order queue. No host sync until the final PCM readback.
    const bool lstm_gpu = [](){ const char* e = std::getenv("NNOPT_ENC_LSTM_GPU"); return !(e && e[0] == '0'); }();
    if (lstm_gpu) {
        const int H = x.ch, Tt = x.T;
        const int gates = 4 * H;
        GSig xres{enc_buf(cl_ctx, (size_t)H * Tt), H, Tt};       // residual copy
        cl_mem gin = enc_buf(cl_ctx, (size_t)Tt * gates);
        cl_mem gbuf = enc_buf(cl_ctx, (size_t)gates);
        cl_mem hbuf = enc_buf(cl_ctx, (size_t)H);
        cl_mem cbuf = enc_buf(cl_ctx, (size_t)H);
        cl_mem ybuf = enc_buf(cl_ctx, (size_t)H * Tt);            // layer output
        if (!xres.mem || !gin || !gbuf || !hbuf || !cbuf || !ybuf) { enc_release(x.mem); return {}; }
        cl_int err = clEnqueueCopyBuffer(q, x.mem, xres.mem, 0, 0, (size_t)H * Tt * sizeof(float), 0, nullptr, nullptr);
        if (err != CL_SUCCESS) { fail("enc-gpu", "lstm residual copy"); enc_release(x.mem); return {}; }
        const float fz = 0.0f;
        bool ok = true;
        for (int l = 0; l < 2 && ok; ++l) {
            const std::string sl = std::to_string(l);
            cl_mem wih = nullptr, whh = nullptr, bi = nullptr, bh = nullptr;
            bool lstm_cached = false;
            if (enc_wcache_on()) {
                auto i1 = g_enc_wcache.find("lw:" + sl);
                auto i2 = g_enc_wcache.find("lb:" + sl);
                if (i1 != g_enc_wcache.end() && i2 != g_enc_wcache.end()) {
                    wih = i1->second.w; whh = i1->second.b;
                    bi = i2->second.w; bh = i2->second.b;
                    lstm_cached = true;
                }
            }
            if (!lstm_cached) {
                std::vector<float> Wih = weights.get_host_vec(D + "1.lstm.weight_ih_l" + sl);
                std::vector<float> Whh = weights.get_host_vec(D + "1.lstm.weight_hh_l" + sl);
                std::vector<float> bih = weights.get_host_vec(D + "1.lstm.bias_ih_l" + sl);
                std::vector<float> bhh = weights.get_host_vec(D + "1.lstm.bias_hh_l" + sl);
                if ((int)Wih.size() != gates * H || (int)Whh.size() != gates * H ||
                    (int)bih.size() != gates || (int)bhh.size() != gates) {
                    fail("enc-gpu lstm weights", "shape l" + sl); ok = false; break;
                }
                wih = enc_upload(cl_ctx, Wih, "Wih");
                whh = enc_upload(cl_ctx, Whh, "Whh");
                bi = enc_upload(cl_ctx, bih, "bih");
                bh = enc_upload(cl_ctx, bhh, "bhh");
            }
            if (!wih || !whh || !bi || !bh) { ok = false; break; }
            err  = clEnqueueFillBuffer(q, hbuf, &fz, sizeof(float), 0, (size_t)H * sizeof(float), 0, nullptr, nullptr);
            err |= clEnqueueFillBuffer(q, cbuf, &fz, sizeof(float), 0, (size_t)H * sizeof(float), 0, nullptr, nullptr);
            // gin for ALL timesteps in one dispatch. Transposed-x route
            // (enc_lstm_gin_x, bit-identical a4 order): the plain kernel
            // gathers x[j*T+t] — four stride-T cache lines per iteration.
            // gin_x VERDICT (2026-06-05): bit-identical but SLOWER (583->806 ms).
            // x[j*T+t] across adjacent t-threads is per-WARP COALESCED — the
            // [ch,T] layout was already optimal; transposing only added the
            // enc_xt cost. Route disabled; kernel kept with this verdict.
            cl_mem xtg = nullptr;
            bool ginx = false;
            if (ginx) {
                err  = clSetKernelArg(g_enc.k_xt, 0, sizeof(cl_mem), &x.mem);
                err |= clSetKernelArg(g_enc.k_xt, 1, sizeof(cl_mem), &xtg);
                err |= clSetKernelArg(g_enc.k_xt, 2, sizeof(int), &H);
                err |= clSetKernelArg(g_enc.k_xt, 3, sizeof(int), &Tt);
                size_t gx[2] = {(size_t)((Tt + 63) / 64) * 64, (size_t)H};
                size_t lx[2] = {64, 1};
                if (err == CL_SUCCESS)
                    err = clEnqueueNDRangeKernel(q, g_enc.k_xt, 2, nullptr, gx, lx, 0, nullptr, KernelProfiler::event_for("enc_xt"));
                if (err == CL_SUCCESS) {
                    err  = clSetKernelArg(g_enc.k_lstm_gin_x, 0, sizeof(cl_mem), &wih);
                    err |= clSetKernelArg(g_enc.k_lstm_gin_x, 1, sizeof(cl_mem), &bi);
                    err |= clSetKernelArg(g_enc.k_lstm_gin_x, 2, sizeof(cl_mem), &bh);
                    err |= clSetKernelArg(g_enc.k_lstm_gin_x, 3, sizeof(cl_mem), &xtg);
                    err |= clSetKernelArg(g_enc.k_lstm_gin_x, 4, sizeof(cl_mem), &gin);
                    err |= clSetKernelArg(g_enc.k_lstm_gin_x, 5, sizeof(int), &H);
                    err |= clSetKernelArg(g_enc.k_lstm_gin_x, 6, sizeof(int), &Tt);
                    size_t ggin[2] = {(size_t)((Tt + 63) / 64) * 64, (size_t)gates};
                    size_t lgin[2] = {64, 1};
                    if (err == CL_SUCCESS)
                        err = clEnqueueNDRangeKernel(q, g_enc.k_lstm_gin_x, 2, nullptr, ggin, lgin, 0, nullptr,
                                                     KernelProfiler::event_for("enc_lstm_gin"));
                }
                if (err != CL_SUCCESS) ginx = false;   // fall through to plain gin
            }
            if (xtg) enc_release(xtg);
            if (!ginx) {
            err |= clSetKernelArg(g_enc.k_lstm_gin, 0, sizeof(cl_mem), &wih);
            err |= clSetKernelArg(g_enc.k_lstm_gin, 1, sizeof(cl_mem), &bi);
            err |= clSetKernelArg(g_enc.k_lstm_gin, 2, sizeof(cl_mem), &bh);
            err |= clSetKernelArg(g_enc.k_lstm_gin, 3, sizeof(cl_mem), &x.mem);
            err |= clSetKernelArg(g_enc.k_lstm_gin, 4, sizeof(cl_mem), &gin);
            err |= clSetKernelArg(g_enc.k_lstm_gin, 5, sizeof(int), &H);
            err |= clSetKernelArg(g_enc.k_lstm_gin, 6, sizeof(int), &Tt);
            size_t ggin[2] = {(size_t)((Tt + 63) / 64) * 64, (size_t)gates};
            size_t lgin[2] = {64, 1};
            err = (err != CL_SUCCESS) ? err : clEnqueueNDRangeKernel(q, g_enc.k_lstm_gin, 2, nullptr, ggin, lgin, 0, nullptr,
                                             KernelProfiler::event_for("enc_lstm_gin"));
            }
            if (err != CL_SUCCESS) { fail("enc-gpu", "lstm gin"); ok = false; }
            // recurrence: gates + cell per step (async; queue order = data order)
            const int rpw = 32;   // 4096/32 = 128 WGs
            for (int t = 0; t < Tt && ok; ++t) {
                if ((t & 63) == 0) enc_yield(q);  // drain every 64 steps so the compositor gets a vsync
                err  = clSetKernelArg(g_enc.k_lstm_gates, 0, sizeof(cl_mem), &whh);
                err |= clSetKernelArg(g_enc.k_lstm_gates, 1, sizeof(cl_mem), &gin);
                err |= clSetKernelArg(g_enc.k_lstm_gates, 2, sizeof(cl_mem), &hbuf);
                err |= clSetKernelArg(g_enc.k_lstm_gates, 3, sizeof(cl_mem), &gbuf);
                err |= clSetKernelArg(g_enc.k_lstm_gates, 4, sizeof(int), &H);
                err |= clSetKernelArg(g_enc.k_lstm_gates, 5, sizeof(int), &t);
                err |= clSetKernelArg(g_enc.k_lstm_gates, 6, sizeof(int), &rpw);
                size_t gg[1] = {(size_t)(gates / rpw) * 64}; size_t lg[1] = {64};
                if (err == CL_SUCCESS)
                    err = clEnqueueNDRangeKernel(q, g_enc.k_lstm_gates, 1, nullptr, gg, lg, 0, nullptr,
                                                 t == 0 ? KernelProfiler::event_for("enc_lstm_gates") : nullptr);
                err |= clSetKernelArg(g_enc.k_lstm_cell, 0, sizeof(cl_mem), &gbuf);
                err |= clSetKernelArg(g_enc.k_lstm_cell, 1, sizeof(cl_mem), &cbuf);
                err |= clSetKernelArg(g_enc.k_lstm_cell, 2, sizeof(cl_mem), &hbuf);
                err |= clSetKernelArg(g_enc.k_lstm_cell, 3, sizeof(cl_mem), &ybuf);
                err |= clSetKernelArg(g_enc.k_lstm_cell, 4, sizeof(int), &H);
                err |= clSetKernelArg(g_enc.k_lstm_cell, 5, sizeof(int), &Tt);
                err |= clSetKernelArg(g_enc.k_lstm_cell, 6, sizeof(int), &t);
                size_t gc[1] = {(size_t)H}; size_t lc[1] = {64};
                if (err == CL_SUCCESS)
                    err = clEnqueueNDRangeKernel(q, g_enc.k_lstm_cell, 1, nullptr, gc, lc, 0, nullptr, nullptr);
                if (err != CL_SUCCESS) { fail("enc-gpu", "lstm step"); ok = false; }
            }
            if (enc_wcache_on()) {
                if (!lstm_cached) {
                    g_enc_wcache["lw:" + sl] = {wih, whh, 0, 0, 0};
                    g_enc_wcache["lb:" + sl] = {bi, bh, 0, 0, 0};
                }
            } else { enc_release(wih); enc_release(whh); enc_release(bi); enc_release(bh); }
            // layer output becomes next layer's input (swap x ↔ y)
            cl_mem tmp = x.mem; x.mem = ybuf; ybuf = tmp;
        }
        if (ok) {
            // residual: x += xres
            const int n = H * Tt;
            err  = clSetKernelArg(g_enc.k_add, 0, sizeof(cl_mem), &x.mem);
            err |= clSetKernelArg(g_enc.k_add, 1, sizeof(cl_mem), &xres.mem);
            err |= clSetKernelArg(g_enc.k_add, 2, sizeof(int), &n);
            size_t ga[1] = {(size_t)((n + 63) / 64) * 64}; size_t la[1] = {64};
            if (err == CL_SUCCESS)
                err = clEnqueueNDRangeKernel(q, g_enc.k_add, 1, nullptr, ga, la, 0, nullptr, nullptr);
            if (err != CL_SUCCESS) { fail("enc-gpu", "lstm residual add"); ok = false; }
        }
        enc_release(xres.mem); enc_release(gin); enc_release(gbuf);
        enc_release(hbuf); enc_release(cbuf); enc_release(ybuf);
        if (!ok) { enc_release(x.mem); return {}; }
    } else {
        // Phase-A fallback: LSTM on CPU (download [1024, T] ~1 MB, run, upload).
        Signal sx; sx.ch = x.ch; sx.T = x.T;
        sx.data.resize((size_t)x.ch * x.T);
        cl_int err = clEnqueueReadBuffer(q, x.mem, CL_TRUE, 0, sx.data.size() * sizeof(float),
                                         sx.data.data(), 0, nullptr, nullptr);
        if (err != CL_SUCCESS) { fail("enc-gpu", "lstm readback"); enc_release(x.mem); return {}; }
        if (!encodec_lstm(weights, D + "1.lstm", 2, sx)) { enc_release(x.mem); return {}; }
        err = clEnqueueWriteBuffer(q, x.mem, CL_TRUE, 0, sx.data.size() * sizeof(float),
                                   sx.data.data(), 0, nullptr, nullptr);
        if (err != CL_SUCCESS) { fail("enc-gpu", "lstm upload"); enc_release(x.mem); return {}; }
    }
    ph_mark("conv0+lstm");
    enc_yield(q);  // yield after the LSTM block

    struct Stage { int convt_idx; int resnet_idx; int stride; };
    const Stage stages[4] = {{3, 4, 8}, {6, 7, 5}, {9, 10, 4}, {12, 13, 4}};
    for (const Stage& s : stages) {
        if (!enc_gpu_elu(cl_ctx, x)) { enc_release(x.mem); return {}; }
        GSig up;
        if (!enc_gpu_convt1d(cl_ctx, weights, D + std::to_string(s.convt_idx) + ".conv",
                             x, s.stride, up)) { enc_release(x.mem); return {}; }
        enc_release(x.mem); x = up;
        if (!enc_gpu_resnet(cl_ctx, weights, D + std::to_string(s.resnet_idx), x)) {
            enc_release(x.mem); return {};
        }
        enc_yield(q);  // yield between the (increasingly large) upsampling stages
    }
    if (!enc_gpu_elu(cl_ctx, x)) { enc_release(x.mem); return {}; }
    GSig wave;
    if (!enc_gpu_conv1d(cl_ctx, weights, D + "15.conv", x, 1, 1, wave)) { enc_release(x.mem); return {}; }
    enc_release(x.mem);
    if (wave.ch != 1) { fail("enc-gpu", "final channels"); enc_release(wave.mem); return {}; }

    ph_mark("conv_stack_enqueue");
    std::vector<float> pcm((size_t)wave.T);
    cl_int err = clEnqueueReadBuffer(q, wave.mem, CL_TRUE, 0, pcm.size() * sizeof(float),
                                     pcm.data(), 0, nullptr, nullptr);
    enc_release(wave.mem);
    if (err != CL_SUCCESS) { fail("enc-gpu", "pcm readback"); return {}; }
    ph_mark("gpu_drain+readback");
    if (g_error) return {};
    for (float& v : pcm) { if (v > 1.0f) v = 1.0f; else if (v < -1.0f) v = -1.0f; }
    return pcm;
}

// =============================================================================
// EncodecStream — chunked CPU SEANet decode that runs WHILE the GPU is still
// generating frames (pipeline mode leaves the CPU idle during decode).
//
// Exactness strategy (target: PCM byte-identical to encodec_decode_host):
//   conv0  — recomputed over latent[0..avail) per push (cheap, threaded);
//            outputs within 3 frames of the avail-edge are reflect-corrupted
//            → only frames <= avail-4 are consumed (all, when is_last).
//   LSTM   — strictly stateful (h/c carried across chunks; gin hoisted per
//            chunk with the same per-(t,r) accumulation order as the full run).
//   stages — each emitted segment [a,b) is computed on a window
//            [a-MARGIN, b+MARGIN] of the residual history; MARGIN exceeds the
//            upsample stack's receptive radius, so interior outputs see the
//            exact same inputs as a full run; clip edges (wa==0 / wb==F) use
//            the true reflect padding. Emitted samples = window slice
//            [(a-wa)*640, (b-wa)*640).
// =============================================================================

struct EncodecStream::Impl {
    Weights& w;
    int F = 0;                 // total frames
    bool failed = false;
    static constexpr int kMargin = 12;        // frames; > upsample RF (~6)
    static constexpr int kUp = 640;           // samples per frame (8*5*4*4)

    Signal latent;             // [128, F]
    Signal conv0_out;          // [1024, F] (recomputed per push; valid to usable)
    Signal resid_hist;         // [1024, F] = lstm2_out + conv0_out
    std::vector<float> pcm_out;

    int latent_done = 0;
    int lstm_done = 0;
    int emitted = 0;

    // cached weights
    std::vector<std::vector<float>> embeds;   // 4 × [2048*128]
    struct LstmLayer {
        std::vector<float> Wih, Whh, bih, bhh;
        std::vector<float> h, c;
        Signal out;            // [1024, F] layer output history
    };
    LstmLayer lstm[2];

    Impl(Weights& weights, int total_frames) : w(weights), F(total_frames) {
        latent.ch = 128; latent.T = F; latent.data.assign((size_t)128 * F, 0.0f);
        resid_hist.ch = 1024; resid_hist.T = F; resid_hist.data.assign((size_t)1024 * F, 0.0f);
        pcm_out.reserve((size_t)F * kUp);
        for (int k = 0; k < 4; ++k) {
            std::string ek = "audio_encoder.quantizer.layers." + std::to_string(k) + ".codebook.embed";
            embeds.push_back(w.get_host_vec(ek));
            if (embeds.back().size() != (size_t)2048 * 128) { fail("enc-stream codebook", ek); failed = true; }
        }
        const std::string L = "audio_encoder.decoder.layers.1.lstm.";
        for (int l = 0; l < 2 && !failed; ++l) {
            const std::string sl = std::to_string(l);
            lstm[l].Wih = w.get_host_vec(L + "weight_ih_l" + sl);
            lstm[l].Whh = w.get_host_vec(L + "weight_hh_l" + sl);
            lstm[l].bih = w.get_host_vec(L + "bias_ih_l" + sl);
            lstm[l].bhh = w.get_host_vec(L + "bias_hh_l" + sl);
            if (lstm[l].Wih.size() != (size_t)4096 * 1024 || lstm[l].bih.size() != 4096) {
                fail("enc-stream lstm weights", sl); failed = true;
            }
            lstm[l].h.assign(1024, 0.0f);
            lstm[l].c.assign(1024, 0.0f);
            lstm[l].out.ch = 1024; lstm[l].out.T = F;
            lstm[l].out.data.assign((size_t)1024 * F, 0.0f);
        }
    }

    // Incremental LSTM over input frames [lo, hi) of `in` (mirrors
    // encodec_lstm's math exactly: gin hoist + parallel gate rows + serial cell).
    void lstm_layer_advance(LstmLayer& L, const Signal& in, int lo, int hi) {
        const int H = 1024, gates = 4096, T = in.T;
        const int n = hi - lo;
        if (n <= 0) return;
        std::vector<double> gin((size_t)n * gates);
        parallel_over_channels(gates, [&](int r_lo, int r_hi) {
            std::vector<double> accv((size_t)n);
            for (int r = r_lo; r < r_hi; ++r) {
                const double b0 = (double)L.bih[r] + (double)L.bhh[r];
                for (int t = 0; t < n; ++t) accv[t] = b0;
                const float* wr = &L.Wih[(size_t)r * H];
                for (int j = 0; j < H; ++j) {
                    const double wj = (double)wr[j];
                    const float* xrow = &in.data[(size_t)j * T];
                    for (int t = 0; t < n; ++t) accv[t] += wj * (double)xrow[lo + t];
                }
                for (int t = 0; t < n; ++t) gin[(size_t)t * gates + r] = accv[t];
            }
        });
        std::vector<float> g(gates);
        for (int t = 0; t < n; ++t) {
            parallel_over_channels(gates, [&](int r_lo, int r_hi) {
                for (int r = r_lo; r < r_hi; ++r) {
                    double acc = gin[(size_t)t * gates + r];
                    const float* ur = &L.Whh[(size_t)r * H];
                    for (int j = 0; j < H; ++j) acc += (double)ur[j] * (double)L.h[j];
                    g[r] = (float)acc;
                }
            });
            for (int j = 0; j < H; ++j) {
                double ig = 1.0 / (1.0 + std::exp(-(double)g[j]));
                double fg = 1.0 / (1.0 + std::exp(-(double)g[H + j]));
                double gg = std::tanh((double)g[2 * H + j]);
                double og = 1.0 / (1.0 + std::exp(-(double)g[3 * H + j]));
                double cn = fg * (double)L.c[j] + ig * gg;
                L.c[j] = (float)cn;
                double hn = og * std::tanh(cn);
                L.h[j] = (float)hn;
                L.out.data[(size_t)j * T + (lo + t)] = (float)hn;
            }
        }
    }

    bool push(const std::vector<std::vector<int32_t>>& codes, int avail, bool is_last) {
        if (failed) return false;
        if (avail > F) avail = F;
        // 1) RVQ for new frames
        for (int f = latent_done; f < avail; ++f) {
            for (int k = 0; k < 4; ++k) {
                const int32_t id = codes[(size_t)k][(size_t)f];
                if (id < 0 || id >= 2048) { fail("enc-stream code id", std::to_string(id)); failed = true; return false; }
                const float* row = &embeds[(size_t)k][(size_t)id * 128];
                for (int d = 0; d < 128; ++d) latent.at(d, f) += row[d];
            }
        }
        latent_done = avail;
        // 2) conv0 over latent[0..avail) (recompute; threaded ~0.3 s for 250fr).
        //    Valid output frames: [0, usable) — reflect at the avail-edge
        //    corrupts the last 3 unless this is the true end.
        Signal lat_win;
        lat_win.ch = 128; lat_win.T = avail;
        lat_win.data.assign(latent.data.begin(), latent.data.begin() + (size_t)128 * 0);
        lat_win.data.resize((size_t)128 * avail);
        for (int c = 0; c < 128; ++c)
            std::memcpy(&lat_win.data[(size_t)c * avail], &latent.data[(size_t)c * F], (size_t)avail * sizeof(float));
        Signal c0;
        if (!encodec_conv1d(w, "audio_encoder.decoder.layers.0.conv", lat_win, 1, 1, c0)) { failed = true; return false; }
        const int usable = is_last ? avail : std::max(0, avail - 4);
        if (conv0_out.T != F) { conv0_out.ch = 1024; conv0_out.T = F; conv0_out.data.assign((size_t)1024 * F, 0.0f); }
        for (int c = 0; c < 1024; ++c)
            for (int f = lstm_done; f < usable; ++f)
                conv0_out.at(c, f) = c0.at(c, f);
        // 3) LSTM incremental over conv0 frames [lstm_done, usable)
        if (usable > lstm_done) {
            lstm_layer_advance(lstm[0], conv0_out, lstm_done, usable);
            lstm_layer_advance(lstm[1], lstm[0].out, lstm_done, usable);
            for (int c = 0; c < 1024; ++c)
                for (int f = lstm_done; f < usable; ++f)
                    resid_hist.at(c, f) = lstm[1].out.at(c, f) + conv0_out.at(c, f);
            lstm_done = usable;
        }
        // 4) emit segment [emitted, b) with window margins
        const int b = is_last ? F : std::max(emitted, lstm_done - kMargin);
        if (b <= emitted) return true;
        if (is_last && lstm_done != F) { fail("enc-stream", "lstm incomplete at last push"); failed = true; return false; }
        const int wa = std::max(0, emitted - kMargin);
        const int wb = is_last ? F : std::min(lstm_done, b + kMargin);
        Signal win;
        win.ch = 1024; win.T = wb - wa;
        win.data.resize((size_t)1024 * (wb - wa));
        for (int c = 0; c < 1024; ++c)
            std::memcpy(&win.data[(size_t)c * (wb - wa)], &resid_hist.data[(size_t)c * F + wa], (size_t)(wb - wa) * sizeof(float));
        // upsample stage chain (identical sequence to encodec_decode_host)
        const std::string D = "audio_encoder.decoder.layers.";
        struct Stage { int convt_idx; int resnet_idx; int stride; };
        const Stage stages[4] = {{3, 4, 8}, {6, 7, 5}, {9, 10, 4}, {12, 13, 4}};
        Signal x = win;
        for (const Stage& st : stages) {
            elu_inplace(x);
            Signal up;
            if (!encodec_conv_transpose1d(w, D + std::to_string(st.convt_idx) + ".conv", x, st.stride, up)) { failed = true; return false; }
            Signal rb;
            if (!encodec_resnet_block(w, D + std::to_string(st.resnet_idx), up, rb)) { failed = true; return false; }
            x = rb;
        }
        elu_inplace(x);
        Signal wave;
        if (!encodec_conv1d(w, D + "15.conv", x, 1, 1, wave)) { failed = true; return false; }
        if (wave.ch != 1 || wave.T != (wb - wa) * kUp) { fail("enc-stream", "window sample count"); failed = true; return false; }
        const size_t s0 = (size_t)(emitted - wa) * kUp;
        const size_t s1 = (size_t)(b - wa) * kUp;
        for (size_t i = s0; i < s1; ++i) {
            float v = wave.data[i];
            if (v > 1.0f) v = 1.0f; else if (v < -1.0f) v = -1.0f;
            pcm_out.push_back(v);
        }
        emitted = b;
        return true;
    }
};

EncodecStream::EncodecStream(Weights& weights, int total_frames)
    : impl_(new Impl(weights, total_frames)) {}
EncodecStream::~EncodecStream() { delete impl_; }
bool EncodecStream::push(const std::vector<std::vector<int32_t>>& codes, int frames_avail, bool is_last) {
    return impl_->push(codes, frames_avail, is_last);
}
const std::vector<float>& EncodecStream::pcm() const { return impl_->pcm_out; }
bool EncodecStream::ok() const { return !impl_->failed; }
