// HiFiGAN-style vocoder + harmonic source module + iSTFT head.
// kmodel.decoder.decoder.generator.*
//
//   decoded [256,2F] -> LeakyReLU(0.1) -> ups.0 (x10) -> +noise_res.0
//                    -> resblocks.0/1 -> avg -> LeakyReLU(0.1)
//                    -> ups.1 (x6) -> reflect-pad 1 -> +noise_res.1
//                    -> resblocks.2/3 -> avg -> LeakyReLU(0.01)
//                    -> conv_post (22ch) -> iSTFT -> waveform
//
//   source: F0 (RAW, pre-F0_conv) -> harmonic sines + noise -> tanh
//           -> edge-pad -> forward STFT -> [mag||phase] 22ch
//           -> noise_convs.{0,1} -> noise_res.{0,1} -> injected above
//
// The source module runs on the HOST: it is elementwise + a cumulative sum over
// 95400x9 values, the cumsum is strictly sequential, and its phase accumulator
// must stay fp32 (it reaches ~1.2e5 — in fp16 it becomes Inf, sin(Inf)=NaN, and
// the whole waveform goes silent). CPU cost is a few ms.
//
// The noise fixture is REPLAYED from assets, never regenerated: 14.8% of samples
// are unvoiced and are essentially pure noise, so a device RNG pins any cosine
// gate at ~0.85 permanently.

#include "../opencl_context.h"
#include "../weights.h"
#include "../nnopt_error.h"
#include "../styletts_ops.h"
#include "../debug_utils.h"
#include <string>
#include <vector>
#include <cmath>
#include <cstdint>
#include <cstdio>

namespace {
constexpr int   N_HARM   = 9;
constexpr float SR       = 24000.0f;
constexpr float UV_THR   = 10.0f;
constexpr float SINE_AMP = 0.1f;
constexpr float NOISE_V  = 0.003f;
constexpr float UV_DIV   = 3.0f;
constexpr int   UPP      = 300;      // samples per decoder frame
constexpr int   NFFT     = 20;
constexpr int   HOP      = 5;
constexpr int   NBIN     = 11;       // NFFT/2 + 1
constexpr float PI_F     = 3.1415927410125732f;
constexpr float MAG_EPS  = 1e-14f;

struct ResblkCfg { const char* name; int C; int K; };

// One AdaINResBlock1: 3 sub-blocks, convs1 dilated [1,3,5], convs2 dilation 1.
cl_mem gen_resblock(OpenCLContext& ctx, cl_command_queue q, Weights& w,
                    cl_mem x, int C, int T, int K, const std::string& blk,
                    cl_mem style_q) {
    const std::string WP = "kmodel.decoder.decoder.generator." + blk + ".";
    const std::string RP = "decoder.generator." + blk + ".";   // Reciprocal_* live here
    const int DIL[3] = {1, 3, 5};
    const char* RECIP[6] = {"Reciprocal_output_0",   "Reciprocal_1_output_0",
                            "Reciprocal_2_output_0", "Reciprocal_3_output_0",
                            "Reciprocal_4_output_0", "Reciprocal_5_output_0"};
    std::vector<cl_mem> owned;
    auto own = [&](cl_mem m) { if (m) owned.push_back(m); return m; };
    auto fail = [&]() -> cl_mem {
        for (cl_mem m : owned) if (m) clReleaseMemObject(m);
        return nullptr;
    };
    cl_mem params = own(st::alloc_f32(ctx, 2));
    cl_mem cur = x;
    bool cur_owned = false;

    for (int j = 0; j < 3; j++) {
        const std::string js = std::to_string(j);
        cl_mem a1  = w.get_buffer(WP + "alpha1." + js);
        cl_mem a2  = w.get_buffer(WP + "alpha2." + js);
        cl_mem r1  = w.get_buffer(RP + RECIP[2 * j]);       // 1/alpha1.j
        cl_mem r2  = w.get_buffer(RP + RECIP[2 * j + 1]);   // 1/alpha2.j
        cl_mem n1w = w.get_buffer(WP + "adain1." + js + ".norm.weight");
        cl_mem n1b = w.get_buffer(WP + "adain1." + js + ".norm.bias");
        cl_mem n2w = w.get_buffer(WP + "adain2." + js + ".norm.weight");
        cl_mem n2b = w.get_buffer(WP + "adain2." + js + ".norm.bias");
        cl_mem c1w = w.get_buffer(WP + "convs1." + js + ".weight_quantized");
        cl_mem c1b = w.get_buffer(WP + "convs1." + js + ".bias");
        cl_mem c2w = w.get_buffer(WP + "convs2." + js + ".weight_quantized");
        cl_mem c2b = w.get_buffer(WP + "convs2." + js + ".bias");
        if (!a1 || !a2 || !r1 || !r2 || !n1w || !n2w || !c1w || !c2w) {
            NNOPT_ERROR_FMT("gen_resblock %s.%d: missing weights", blk.c_str(), j);
            return fail();
        }
        cl_mem f1 = own(st::adain_fc(ctx, q, w, style_q,
                                     WP + "adain1." + js + ".fc.weight_quantized",
                                     WP + "adain1." + js + ".fc.bias", 2 * C));
        cl_mem f2 = own(st::adain_fc(ctx, q, w, style_q,
                                     WP + "adain2." + js + ".fc.weight_quantized",
                                     WP + "adain2." + js + ".fc.bias", 2 * C));
        if (!f1 || !f2) return fail();

        cl_mem scr = own(st::alloc(ctx, (size_t)C * T));
        cl_mem h   = own(st::alloc(ctx, (size_t)C * T));
        cl_mem hq  = own(st::alloc(ctx, (size_t)C * T));
        cl_mem h2  = own(st::alloc(ctx, (size_t)C * T));
        if (!scr || !h || !hq || !h2) return fail();

        const int dil = DIL[j];
        if (!st::adain(q, cur, n1w, n1b, f1, scr, h, C, T)) return fail();
        if (!st::snake(q, h, a1, r1, h, C, T)) return fail();
        if (!st::dql(q, h, params, hq, C * T)) return fail();
        if (!st::conv1d(q, hq, c1w, c1b, h2, C, C, T, T, K, 1, dil * (K - 1) / 2, 1, dil))
            return fail();
        if (!st::adain(q, h2, n2w, n2b, f2, scr, h, C, T)) return fail();
        if (!st::snake(q, h, a2, r2, h, C, T)) return fail();
        if (!st::dql(q, h, params, hq, C * T)) return fail();
        if (!st::conv1d(q, hq, c2w, c2b, h2, C, C, T, T, K, 1, (K - 1) / 2, 1, 1))
            return fail();

        cl_mem nxt = st::alloc(ctx, (size_t)C * T);
        if (!nxt || !st::add(q, h2, cur, nxt, C * T)) { if (nxt) clReleaseMemObject(nxt); return fail(); }
        if (cur_owned) clReleaseMemObject(cur);
        cur = nxt;
        cur_owned = true;
    }
    for (cl_mem m : owned) if (m) clReleaseMemObject(m);
    return cur;
}
}  // namespace

extern "C" cl_mem Generator_forward(
    OpenCLContext& cl_ctx,
    Weights& weights,
    cl_command_queue queue,
    cl_mem decoded,                       // [256, 2F]
    cl_mem f0_raw,                        // [1, 2F]  RAW F0, pre-F0_conv
    cl_mem style,                         // [256]
    const std::vector<float>& rng_normal, // [2F*UPP*9] replay fixture
    int F,
    int* out_samples)
{
    if (!st::init(cl_ctx)) return nullptr;
    const int T0 = 2 * F;                 // decoder frames (318)
    const int NS = T0 * UPP;              // waveform samples (95400)
    const int NP = NS + 2 * (NFFT / 2);   // edge-padded (95420)
    const int FR = (NP - NFFT) / HOP + 1; // STFT frames (19081)
    const int T1 = T0 * 10;               // after ups.0 (3180)

    std::vector<cl_mem> owned;
    auto own = [&](cl_mem m) { if (m) owned.push_back(m); return m; };
    auto fail = [&]() -> cl_mem {
        for (cl_mem m : owned) if (m) clReleaseMemObject(m);
        owned.clear();
        return nullptr;
    };

    cl_mem params = own(st::alloc_f32(cl_ctx, 2));
    cl_mem s_acou = own(st::alloc(cl_ctx, 128));
    cl_mem sa_q   = own(st::alloc(cl_ctx, 128));
    if (!params || !s_acou || !sa_q) return fail();
    if (clEnqueueCopyBuffer(queue, style, s_acou, 0, 0,
                            128 * sizeof(nnopt_storage_t), 0, nullptr, nullptr) != CL_SUCCESS)
        return fail();
    if (!st::dql(queue, s_acou, params, sa_q, 128)) return fail();

    // ── source module (host; fp32 phase is mandatory) ────────────────────
    std::vector<float> f0;
    if (!st::download_as_f32(queue, f0_raw, (size_t)T0, f0)) return fail();

    // The captured noise fixture is sized for the REFERENCE utterance
    // (318 frames -> 95,400 samples). For a longer new utterance there is no
    // oracle to match, so fall back to deterministic host noise (fixed seed,
    // reproducible run-to-run). Never a device RNG, and never silently: any
    // cosine comparison against reference/layers/ is meaningless in this mode.
    std::vector<float> noise_fallback;
    const float* randn = rng_normal.data();
    if ((int)rng_normal.size() < NS * N_HARM) {
        fprintf(stderr, "NOTE: rng fixture holds %zu floats but this utterance needs %d; "
                        "using deterministic seeded host noise. Oracle comparison is NOT "
                        "valid in this mode.\n", rng_normal.size(), NS * N_HARM);
        noise_fallback.resize((size_t)NS * N_HARM);
        uint32_t st8 = 0x9E3779B9u;
        for (size_t i = 0; i < noise_fallback.size(); i += 2) {
            // Box-Muller from a splitmix32 stream (full avalanche finalizer)
            auto nextf = [&]() {
                st8 += 0x9E3779B9u; uint32_t z = st8;
                z = (z ^ (z >> 16)) * 0x21F0AAADu;
                z = (z ^ (z >> 15)) * 0x735A2D97u;
                z =  z ^ (z >> 15);
                return (float)((z >> 8) + 1u) / 16777217.0f;
            };
            float u1 = nextf(), u2 = nextf();
            float r = std::sqrt(-2.0f * std::log(u1));
            noise_fallback[i] = r * std::cos(2.0f * PI_F * u2);
            if (i + 1 < noise_fallback.size())
                noise_fallback[i + 1] = r * std::sin(2.0f * PI_F * u2);
        }
        randn = noise_fallback.data();
    }

    std::vector<float> har(NS);
    {
        // f0_up: nearest/asymmetric/floor replicate by UPP
        // rad: FLOOR-mod (F0 dips negative, so fmod would break unvoiced frames)
        std::vector<float> rad_f((size_t)T0 * N_HARM, 0.0f);
        for (int i = 0; i < T0; i++) {
            // Resize down (linear, half_pixel, scale 1/UPP) reads samples
            // UPP*i+149 and UPP*i+150 with equal weights.
            for (int h = 0; h < N_HARM; h++) {
                float acc = 0.0f;
                for (int d = 149; d <= 150; d++) {
                    int j = i * UPP + d;
                    if (j >= NS) j = NS - 1;
                    float f = f0[j / UPP] * (float)(h + 1) / SR;
                    acc += 0.5f * (f - std::floor(f));
                }
                rad_f[(size_t)i * N_HARM + h] = acc;
            }
        }
        // phase_f = 2*pi * cumsum over frames, then scaled by UPP
        std::vector<float> phase_f((size_t)T0 * N_HARM, 0.0f);
        for (int h = 0; h < N_HARM; h++) {
            float run = 0.0f;
            for (int i = 0; i < T0; i++) {
                run += rad_f[(size_t)i * N_HARM + h];
                phase_f[(size_t)i * N_HARM + h] = 2.0f * PI_F * run * (float)UPP;
            }
        }
        std::vector<float> Wl = weights.get_host_vec("onnx_MatMul_8321");
        std::vector<float> bl = weights.get_host_vec("kmodel.decoder.decoder.generator.m_source.l_linear.bias");
        if ((int)Wl.size() < N_HARM || bl.empty()) {
            NNOPT_ERROR("Generator: missing m_source.l_linear weights"); return fail();
        }
        for (int j = 0; j < NS; j++) {
            float f0j = f0[j / UPP];
            float uv  = (f0j > UV_THR) ? 1.0f : 0.0f;
            float namp = uv * NOISE_V + (1.0f - uv) * (SINE_AMP / UV_DIV);
            // Resize up (linear, half_pixel, scale UPP), edge-clamped
            float src = ((float)j + 0.5f) / (float)UPP - 0.5f;
            if (src < 0.0f) src = 0.0f;
            if (src > (float)(T0 - 1)) src = (float)(T0 - 1);
            int i0 = (int)std::floor(src);
            int i1 = (i0 + 1 < T0) ? i0 + 1 : T0 - 1;
            float fr = src - (float)i0;
            float acc = bl[0];
            for (int h = 0; h < N_HARM; h++) {
                float p = phase_f[(size_t)i0 * N_HARM + h] * (1.0f - fr)
                        + phase_f[(size_t)i1 * N_HARM + h] * fr;
                float sig = uv * SINE_AMP * std::sin(p)
                          + namp * randn[(size_t)j * N_HARM + h];
                acc += sig * Wl[h];
            }
            har[j] = std::tanh(acc);
        }
    }

    // edge-pad 10 samples at each end, then upload
    std::vector<float> harp(NP);
    for (int i = 0; i < NP; i++) {
        int s = i - NFFT / 2;
        if (s < 0) s = 0;
        if (s >= NS) s = NS - 1;
        harp[i] = har[s];
    }
    cl_mem src_pad = own(st::upload_f32_as_storage(cl_ctx, harp));
    if (!src_pad) return fail();
    NNOPT_LAYER_CHECK("decoder.generator.Pad_output_0", queue, src_pad, (size_t)NP);

    // ── forward STFT of the source -> [mag || phase] 22ch ────────────────
    cl_mem fwr = weights.get_buffer("kmodel.decoder.decoder.generator.stft.weight_forward_real");
    cl_mem fwi = weights.get_buffer("kmodel.decoder.decoder.generator.stft.weight_forward_imag");
    if (!fwr || !fwi) { NNOPT_ERROR("Generator: missing forward stft basis"); return fail(); }
    cl_mem sre = own(st::alloc(cl_ctx, (size_t)NBIN * FR));
    cl_mem sim = own(st::alloc(cl_ctx, (size_t)NBIN * FR));
    cl_mem smg = own(st::alloc(cl_ctx, (size_t)NBIN * FR));
    cl_mem sph = own(st::alloc(cl_ctx, (size_t)NBIN * FR));
    cl_mem cat22 = own(st::alloc(cl_ctx, (size_t)2 * NBIN * FR));
    if (!sre || !sim || !smg || !sph || !cat22) return fail();
    if (!st::conv1d(queue, src_pad, fwr, nullptr, sre, 1, NBIN, NP, FR, NFFT, HOP, 0, 1)) return fail();
    if (!st::conv1d(queue, src_pad, fwi, nullptr, sim, 1, NBIN, NP, FR, NFFT, HOP, 0, 1)) return fail();
    NNOPT_LAYER_CHECK("decoder.generator.Conv_output_0", queue, sre, (size_t)NBIN * FR);
    NNOPT_LAYER_CHECK("decoder.generator.Conv_1_output_0", queue, sim, (size_t)NBIN * FR);
    if (!st::mag_phase(queue, sre, sim, smg, sph, NBIN * FR, MAG_EPS)) return fail();
    NNOPT_LAYER_CHECK("decoder.generator.Sqrt_output_0", queue, smg, (size_t)NBIN * FR);
    NNOPT_LAYER_CHECK("decoder.generator.Where_2_output_0", queue, sph, (size_t)NBIN * FR);
    if (!st::concat_ct(queue, {smg, sph}, {NBIN, NBIN}, cat22, FR)) return fail();
    NNOPT_LAYER_CHECK("decoder.generator.Concat_1_output_0", queue, cat22, (size_t)2 * NBIN * FR);

    // ── noise branches ───────────────────────────────────────────────────
    cl_mem catq = own(st::alloc(cl_ctx, (size_t)2 * NBIN * FR));
    if (!catq || !st::dql(queue, cat22, params, catq, 2 * NBIN * FR)) return fail();

    cl_mem nc0 = own(st::alloc(cl_ctx, (size_t)128 * T1));
    cl_mem nc1 = own(st::alloc(cl_ctx, (size_t)64 * FR));
    if (!nc0 || !nc1) return fail();
    {
        cl_mem w0 = weights.get_buffer("kmodel.decoder.decoder.generator.noise_convs.0.weight_quantized");
        cl_mem b0 = weights.get_buffer("kmodel.decoder.decoder.generator.noise_convs.0.bias");
        cl_mem w1 = weights.get_buffer("kmodel.decoder.decoder.generator.noise_convs.1.weight_quantized");
        cl_mem b1 = weights.get_buffer("kmodel.decoder.decoder.generator.noise_convs.1.bias");
        if (!w0 || !b0 || !w1 || !b1) { NNOPT_ERROR("Generator: missing noise_convs"); return fail(); }
        if (!st::conv1d(queue, catq, w0, b0, nc0, 22, 128, FR, T1, 12, 6, 3, 1)) return fail();
        if (!st::conv1d(queue, catq, w1, b1, nc1, 22, 64, FR, FR, 1, 1, 0, 1)) return fail();
    }
    NNOPT_LAYER_CHECK("decoder.generator.noise_convs.0.Conv_output_0", queue, nc0, (size_t)128 * T1);

    cl_mem nr0 = own(gen_resblock(cl_ctx, queue, weights, nc0, 128, T1, 7, "noise_res.0", sa_q));
    cl_mem nr1 = own(gen_resblock(cl_ctx, queue, weights, nc1, 64, FR, 11, "noise_res.1", sa_q));
    if (!nr0 || !nr1) { NNOPT_ERROR("Generator: noise_res failed"); return fail(); }
    NNOPT_LAYER_CHECK("decoder.generator.noise_res.0.Add_8_output_0", queue, nr0, (size_t)128 * T1);

    // ── trunk stage 0 ────────────────────────────────────────────────────
    cl_mem act = own(st::alloc(cl_ctx, (size_t)256 * T0));
    cl_mem u0  = own(st::alloc(cl_ctx, (size_t)128 * T1));
    cl_mem add3 = own(st::alloc(cl_ctx, (size_t)128 * T1));
    if (!act || !u0 || !add3) return fail();
    if (!st::leaky_relu_a(queue, decoded, act, 256 * T0, 0.1f)) return fail();
    {
        cl_mem w0 = weights.get_buffer("kmodel.decoder.decoder.generator.ups.0.weight");
        cl_mem b0 = weights.get_buffer("kmodel.decoder.decoder.generator.ups.0.bias");
        if (!w0 || !b0) { NNOPT_ERROR("Generator: missing ups.0"); return fail(); }
        if (!st::conv_transpose1d(queue, act, w0, b0, u0, 256, 128, T0, T1, 20, 10, 5)) return fail();
    }
    if (!st::add(queue, u0, nr0, add3, 128 * T1)) return fail();
    NNOPT_LAYER_CHECK("decoder.generator.Add_3_output_0", queue, add3, (size_t)128 * T1);

    cl_mem rb0 = own(gen_resblock(cl_ctx, queue, weights, add3, 128, T1, 3, "resblocks.0", sa_q));
    cl_mem rb1 = own(gen_resblock(cl_ctx, queue, weights, add3, 128, T1, 3, "resblocks.1", sa_q));
    if (!rb0 || !rb1) { NNOPT_ERROR("Generator: resblocks.0/1 failed"); return fail(); }
    cl_mem avg0 = own(st::alloc(cl_ctx, (size_t)128 * T1));
    if (!avg0 || !st::add_scaled(queue, rb0, rb1, avg0, 128 * T1, 0.5f)) return fail();
    NNOPT_LAYER_CHECK("decoder.generator.Div_1_output_0", queue, avg0, (size_t)128 * T1);

    // ── trunk stage 1 ────────────────────────────────────────────────────
    const int T1B = T1 * 6;          // after ups.1 (19080)
    cl_mem act1 = own(st::alloc(cl_ctx, (size_t)128 * T1));
    cl_mem u1   = own(st::alloc(cl_ctx, (size_t)64 * T1B));
    cl_mem u1p  = own(st::alloc(cl_ctx, (size_t)64 * FR));
    cl_mem add5 = own(st::alloc(cl_ctx, (size_t)64 * FR));
    if (!act1 || !u1 || !u1p || !add5) return fail();
    if (!st::leaky_relu_a(queue, avg0, act1, 128 * T1, 0.1f)) return fail();
    {
        cl_mem w1 = weights.get_buffer("kmodel.decoder.decoder.generator.ups.1.weight");
        cl_mem b1 = weights.get_buffer("kmodel.decoder.decoder.generator.ups.1.bias");
        if (!w1 || !b1) { NNOPT_ERROR("Generator: missing ups.1"); return fail(); }
        if (!st::conv_transpose1d(queue, act1, w1, b1, u1, 128, 64, T1, T1B, 12, 6, 3)) return fail();
    }
    // reflect-pad ONE sample at the FRONT of time; this is what makes the trunk
    // length match the source-STFT frame count (60L+1).
    if (!st::pad_front_reflect(queue, u1, u1p, 64, T1B, 1)) return fail();
    if (!st::add(queue, u1p, nr1, add5, 64 * FR)) return fail();
    NNOPT_LAYER_CHECK("decoder.generator.Add_5_output_0", queue, add5, (size_t)64 * FR);

    cl_mem rb2 = own(gen_resblock(cl_ctx, queue, weights, add5, 64, FR, 3, "resblocks.2", sa_q));
    cl_mem rb3 = own(gen_resblock(cl_ctx, queue, weights, add5, 64, FR, 3, "resblocks.3", sa_q));
    if (!rb2 || !rb3) { NNOPT_ERROR("Generator: resblocks.2/3 failed"); return fail(); }
    cl_mem avg1 = own(st::alloc(cl_ctx, (size_t)64 * FR));
    if (!avg1 || !st::add_scaled(queue, rb2, rb3, avg1, 64 * FR, 0.5f)) return fail();
    NNOPT_LAYER_CHECK("decoder.generator.Div_2_output_0", queue, avg1, (size_t)64 * FR);

    // ── conv_post + iSTFT ────────────────────────────────────────────────
    cl_mem act2 = own(st::alloc(cl_ctx, (size_t)64 * FR));
    cl_mem post = own(st::alloc(cl_ctx, (size_t)22 * FR));
    if (!act2 || !post) return fail();
    if (!st::leaky_relu_a(queue, avg1, act2, 64 * FR, 0.01f)) return fail();   // 0.01 here, not 0.1
    {
        cl_mem w = weights.get_buffer("kmodel.decoder.decoder.generator.conv_post.weight");
        cl_mem b = weights.get_buffer("kmodel.decoder.decoder.generator.conv_post.bias");
        if (!w || !b) { NNOPT_ERROR("Generator: missing conv_post"); return fail(); }
        if (!st::conv1d(queue, act2, w, b, post, 64, 22, FR, FR, 7, 1, 3, 1)) return fail();
    }
    NNOPT_LAYER_CHECK("decoder.generator.conv_post.Conv_output_0", queue, post, (size_t)22 * FR);

    // channels [0:11] = log-magnitude, [11:22] = raw phase (phase = sin(raw))
    cl_mem mag = own(st::alloc(cl_ctx, (size_t)NBIN * FR));
    cl_mem raw = own(st::alloc(cl_ctx, (size_t)NBIN * FR));
    cl_mem ore = own(st::alloc(cl_ctx, (size_t)NBIN * FR));
    cl_mem oim = own(st::alloc(cl_ctx, (size_t)NBIN * FR));
    if (!mag || !raw || !ore || !oim) return fail();
    size_t half = (size_t)NBIN * FR * sizeof(nnopt_storage_t);
    if (clEnqueueCopyBuffer(queue, post, mag, 0, 0, half, 0, nullptr, nullptr) != CL_SUCCESS ||
        clEnqueueCopyBuffer(queue, post, raw, half, 0, half, 0, nullptr, nullptr) != CL_SUCCESS)
        return fail();
    if (!st::exp_(queue, mag, mag, NBIN * FR)) return fail();
    NNOPT_LAYER_CHECK("decoder.generator.Exp_output_0", queue, mag, (size_t)NBIN * FR);
    if (!st::polar_to_cart(queue, mag, raw, ore, oim, NBIN * FR)) return fail();

    cl_mem bwr = weights.get_buffer("kmodel.decoder.decoder.generator.stft.weight_backward_real");
    cl_mem bwi = weights.get_buffer("kmodel.decoder.decoder.generator.stft.weight_backward_imag");
    if (!bwr || !bwi) { NNOPT_ERROR("Generator: missing backward stft basis"); return fail(); }
    cl_mem full = own(st::alloc(cl_ctx, (size_t)NP));
    if (!full) return fail();
    if (!st::istft_ola(queue, ore, oim, bwr, bwi, full, NBIN, FR, NFFT, HOP, NP)) return fail();
    NNOPT_LAYER_CHECK("decoder.generator.Sub_1_output_0", queue, full, (size_t)NP);

    // trim the 10 edge samples added by the source pad
    cl_mem wave = st::alloc(cl_ctx, (size_t)NS);
    if (!wave) return fail();
    if (clEnqueueCopyBuffer(queue, full, wave, (NFFT / 2) * sizeof(nnopt_storage_t), 0,
                            (size_t)NS * sizeof(nnopt_storage_t), 0, nullptr, nullptr) != CL_SUCCESS) {
        clReleaseMemObject(wave); return fail();
    }
    NNOPT_DEBUG_SYNC(queue);
    NNOPT_LAYER_CHECK("decoder.generator.Slice_3_output_0", queue, wave, (size_t)NS);

    for (cl_mem m : owned) if (m) clReleaseMemObject(m);
    owned.clear();
    *out_samples = NS;
    return wave;
}
