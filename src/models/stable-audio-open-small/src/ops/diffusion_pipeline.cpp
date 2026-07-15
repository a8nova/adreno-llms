// diffusion_pipeline.cpp — Stable Audio Open Small end-to-end driver.
//
// Reference: stable_audio_tools/inference/sampling.py sample_flow_pingpong
//              (pingpong sigma schedule, x - sigma*v denoise, resample step)
//            stable_audio_tools/models/dit.py DiffusionTransformer.forward
//              (one DiT forward per denoise step; cfg_scale=1.0)
//            stable_audio_tools/models/autoencoders.py OobleckDecoder.forward
//              (autoencoder / vae_decode of the final latent → stereo waveform)
//            reference/gen_e2e_reference.py (8-step pingpong, seed=0)
//
// This file is the diffusion pipeline: [1] conditioning is provided as fixed
// fixtures (cross-attn cond + global seconds embed), [2] a fixed-N pingpong
// denoise loop runs one DiT forward per sigma step using reference noise
// fixtures (NEVER device RNG), [3] the Oobleck autoencoder decodes the final
// latent to a stereo waveform. main.cpp calls run_diffusion_pipeline().

#include "opencl_context.h"
#include "weights.h"
#include "dit.h"
#include "decoder.h"
#include "debug_utils.h"
#include "model_config.h"
#include "benchmark.h"
#include "profiler.h"

#include <chrono>
#include <fstream>
#include <string>
#include <vector>
#include <cstdint>
#include <cstdio>
#include <cstdlib>

namespace {
bool read_bin(const std::string& path, std::vector<float>& out) {
    std::ifstream f(path, std::ios::binary);
    if (!f) { NNOPT_ERROR_FMT("cannot open %s", path.c_str()); return false; }
    f.seekg(0, std::ios::end);
    std::streamsize n = f.tellg();
    f.seekg(0, std::ios::beg);
    if (n <= 0 || (n % 4) != 0) { NNOPT_ERROR_FMT("bad size %s", path.c_str()); return false; }
    out.resize((size_t)n / 4);
    f.read(reinterpret_cast<char*>(out.data()), n);
    return true;
}

void write_bin(const std::string& path, const std::vector<float>& v) {
    std::ofstream f(path, std::ios::binary);
    f.write(reinterpret_cast<const char*>(v.data()), (std::streamsize)v.size() * 4);
}
} // namespace

// Runs the full diffusion → autoencoder pipeline. Fills `audio` with the
// decoded stereo waveform [channels * frames] (channel-major) and sets
// `frames_out`. Returns true on success.
bool run_diffusion_pipeline(OpenCLContext& cl_ctx,
                            DiT& dit, Decoder& decoder,
                            const std::vector<float>& cross_cond, int cross_seq,
                            const std::vector<float>& global_emb,
                            std::vector<float>& audio, int& frames_out,
                            int latent_len_override) {
    const int C = MODEL_CONFIG::DIT_LATENT_CHANNELS; // 64

    // ── pingpong sigma schedule + init noise (fixtures, deterministic) ──
    std::vector<float> sigmas;
    if (!read_bin("assets/sigmas.bin", sigmas)) return false;
    int steps = (int)sigmas.size() - 1;
    NNOPT_CHECKPOINT_FMT("diffusion steps=%d", steps);

    std::vector<float> x;
    if (!read_bin("assets/init_noise.bin", x)) return false;   // [1,64,latent_len]
    int T = (int)x.size() / C;
    // Fast-iteration truncation (--latent-len): keep the first N frames of
    // each channel. Layout is channel-major [C, T]. Perf iteration only —
    // output is not reference-comparable.
    if (latent_len_override > 0 && latent_len_override < T) {
        const int Tn = latent_len_override;
        std::vector<float> xs((size_t)C * Tn);
        for (int c = 0; c < C; c++)
            for (int t2 = 0; t2 < Tn; t2++)
                xs[(size_t)c * Tn + t2] = x[(size_t)c * T + t2];
        x.swap(xs);
        T = Tn;
        NNOPT_CHECKPOINT_FMT("latent truncated to T=%d (perf-iteration mode)", T);
    }
    for (float& v : x) v *= sigmas[0];  // x = init_noise * sigma_max

    const auto t_dit_start = std::chrono::steady_clock::now();

    // ── B4: device-resident denoise loop — x stays on-GPU for all steps, the
    // pingpong update is one fused kernel, noise buffers are preloaded, and
    // only the final latent is downloaded. Requires full-length noise (the
    // --latent-len fast-iteration mode falls back to the host loop below).
    // NNOPT_DENOISE_GPU=0 reverts.
    const bool denoise_gpu = [&] {
        const char* e = std::getenv("NNOPT_DENOISE_GPU");
        if (e && e[0] == '0') return false;
        if (getenv("NNOPT_DUMP_LAYERS")) return false;  // trajectory dumps need host x
        return true;
    }();
    if (denoise_gpu) {
        std::vector<cl_mem> noise_bufs;
        bool noise_ok = true;
        for (int i = 0; i < steps && noise_ok; i++) {
            std::vector<float> noise;
            char np[96];
            snprintf(np, sizeof(np), "assets/step_noise_%d.bin", i);
            if (!read_bin(std::string(np), noise) || (int)noise.size() / C != T) {
                noise_ok = false;  // missing or truncated-latent mismatch
                break;
            }
            cl_mem nb = dit.upload_buf(noise, noise.size());
            if (!nb) { noise_ok = false; break; }
            noise_bufs.push_back(nb);
        }
        if (noise_ok) {
            cl_mem x_dev = dit.upload_buf(x, x.size());
            if (!x_dev) { for (cl_mem b : noise_bufs) dit.release_buf(b); return false; }
            const int n = (int)x.size();
            bool ok = true;
            for (int i = 0; i < steps && ok; i++) {
                cl_mem v_dev = nullptr;
                ok = dit.forward_step_dev(x_dev, T, sigmas[i], cross_cond, cross_seq,
                                          global_emb, &v_dev);
                if (ok) ok = dit.denoise_resample(x_dev, v_dev, noise_bufs[i],
                                                  sigmas[i], sigmas[i + 1], n);
                if (v_dev) dit.release_buf(v_dev);
                if (i == 0) NNOPT_BENCH_FIRST_TOKEN();
                // Per-step progress marker for app UIs (same stderr channel as SERVE_*).
                if (ok) { fprintf(stderr, "SA_PROGRESS step=%d/%d\n", i + 1, steps); fflush(stderr); }
            }
            for (cl_mem b : noise_bufs) dit.release_buf(b);
            if (ok) dit.download_buf(x_dev, x, x.size());
            dit.release_buf(x_dev);
            if (!ok) { NNOPT_ERROR("GPU denoise loop failed"); return false; }
            write_bin("latent_final.bin", x);
            NNOPT_CHECKPOINT_FMT("LATENT_FINAL size=%zu (T=%d)", x.size(), T);
            const auto t_gpu_dit_end = std::chrono::steady_clock::now();
            if (!decoder.decode(x, T, audio)) { NNOPT_ERROR("autoencoder decode failed"); return false; }
            const auto t_gpu_dec_end = std::chrono::steady_clock::now();
            const double gdit_sec = std::chrono::duration<double>(t_gpu_dit_end - t_dit_start).count();
            const double gdec_sec = std::chrono::duration<double>(t_gpu_dec_end - t_gpu_dit_end).count();
            fprintf(stderr, "BENCHMARK dit_total_sec: %.3f\n", gdit_sec);
            fprintf(stderr, "BENCHMARK dit_step_avg_sec: %.3f\n", steps > 0 ? gdit_sec / steps : 0.0);
            fprintf(stderr, "BENCHMARK decoder_sec: %.3f\n", gdec_sec);
            fprintf(stderr, "BENCHMARK latent_len: %d\n", T);
            extern double g_dec_host_weight_sec, g_dec_host_download_sec;
            fprintf(stderr, "BENCHMARK dec_host_weight_fold_upload_sec: %.3f\n", g_dec_host_weight_sec);
            fprintf(stderr, "BENCHMARK dec_host_download_sec: %.3f\n", g_dec_host_download_sec);
            KernelProfiler::dump_summary();
            frames_out = (int)audio.size() / MODEL_CONFIG::AUDIO_CHANNELS;
            return true;
        }
        for (cl_mem b : noise_bufs) dit.release_buf(b);
        NNOPT_CHECKPOINT_FMT("denoise_gpu fallback to host loop (noise mismatch)%s", "");
    }

    // ── fixed-N pingpong denoise loop (one DiT forward per sigma) ──
    for (int i = 0; i < steps; i++) {
        float sc = sigmas[i];
        float sn = sigmas[i + 1];
        std::vector<float> model_out;
        if (!dit.forward_step(x, sc, cross_cond, cross_seq, global_emb, model_out)) {
            NNOPT_ERROR_FMT("DiT forward_step failed at step %d", i);
            return false;
        }
        if (i == 0) NNOPT_BENCH_FIRST_TOKEN();
        // Per-step progress marker for app UIs (same stderr channel as SERVE_*).
        fprintf(stderr, "SA_PROGRESS step=%d/%d\n", i + 1, steps); fflush(stderr);

        // denoised = x - sigma_curr * v
        std::vector<float> denoised(x.size());
        for (size_t j = 0; j < x.size(); j++) denoised[j] = x[j] - sc * model_out[j];

        // per-step trajectory dump (pairs with reference/trajectory/) —
        // validation artifact, only when dumps are requested
        if (getenv("NNOPT_DUMP_LAYERS")) {
            char tj[96];
            snprintf(tj, sizeof(tj), "trajectory_denoised_step_%d.bin", i);
            write_bin(std::string(tj), denoised);
        }

        // resample: x = (1 - sigma_next) * denoised + sigma_next * step_noise[i]
        std::vector<float> noise;
        char np[96];
        snprintf(np, sizeof(np), "assets/step_noise_%d.bin", i);
        if (!read_bin(std::string(np), noise)) return false;
        // Truncated-latent mode: noise files are full length; index per channel.
        const int Tfull = (int)noise.size() / C;
        if (Tfull != T) {
            for (int c = 0; c < C; c++)
                for (int t2 = 0; t2 < T; t2++)
                    x[(size_t)c * T + t2] = (1.0f - sn) * denoised[(size_t)c * T + t2]
                                          + sn * noise[(size_t)c * Tfull + t2];
        } else {
            for (size_t j = 0; j < x.size(); j++)
                x[j] = (1.0f - sn) * denoised[j] + sn * noise[j];
        }
    }
    write_bin("latent_final.bin", x);
    NNOPT_CHECKPOINT_FMT("LATENT_FINAL size=%zu (T=%d)", x.size(), T);
    const auto t_dit_end = std::chrono::steady_clock::now();

    // ── Oobleck autoencoder decode: latent [64,T] → stereo [2, T*2048] ──
    if (!decoder.decode(x, T, audio)) { NNOPT_ERROR("autoencoder decode failed"); return false; }
    const auto t_dec_end = std::chrono::steady_clock::now();

    const double dit_sec = std::chrono::duration<double>(t_dit_end - t_dit_start).count();
    const double dec_sec = std::chrono::duration<double>(t_dec_end - t_dit_end).count();
    fprintf(stderr, "BENCHMARK dit_total_sec: %.3f\n", dit_sec);
    fprintf(stderr, "BENCHMARK dit_step_avg_sec: %.3f\n", steps > 0 ? dit_sec / steps : 0.0);
    fprintf(stderr, "BENCHMARK decoder_sec: %.3f\n", dec_sec);
    fprintf(stderr, "BENCHMARK latent_len: %d\n", T);
    extern double g_dec_host_weight_sec, g_dec_host_download_sec;
    fprintf(stderr, "BENCHMARK dec_host_weight_fold_upload_sec: %.3f\n", g_dec_host_weight_sec);
    fprintf(stderr, "BENCHMARK dec_host_download_sec: %.3f\n", g_dec_host_download_sec);
    KernelProfiler::dump_summary();

    const int channels = MODEL_CONFIG::AUDIO_CHANNELS;   // 2
    frames_out = (int)audio.size() / channels;           // T * 2048
    return true;
}
