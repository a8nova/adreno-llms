// SeamlessM4T UnitY — on-device inference entry point.
// audio[16000] -> units -> waveform, all stages on OpenCL (fp16).
// Stage selector via argv[1]: "full" (default) | "vocoder" | "fbank" | "encoder".
// Outputs are written as float32 .bin files in the CWD (the device REMOTE_DIR),
// pulled and compared against reference/layers/e2e_out_{1,2} on the host.
#include "opencl_context.h"
#include "weights.h"
#include "gpu_ops.h"
#include "pipeline.h"
#include "wav_io.h"
#include "tokenizer.h"
#include "debug_utils.h"

#include <cstdio>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>
#include <fstream>
#include <iostream>   // --serve: std::cin / std::getline
#include <sstream>    // --serve: parse the request line

// Adreno hardware GPU-busy counter (kgsl): "<busy> <total>" cycles since last read.
// busy/total over an interval = TRUE GPU utilization, counting EVERY kernel (incl. CLBlast
// internals) — the ground truth with no profiler blind spot. Returns false if unreadable.
static bool read_gpubusy(long long& busy, long long& total) {
    std::ifstream f("/sys/class/kgsl/kgsl-3d0/gpubusy");
    if (!f) return false;
    busy = total = 0; f >> busy >> total; return true;
}

static std::vector<float> load_floats(const std::string& path) {
    std::vector<float> v;
    FILE* f = fopen(path.c_str(), "rb");
    if (!f) { fprintf(stderr, "MISSING %s\n", path.c_str()); return v; }
    fseek(f, 0, SEEK_END); long n = ftell(f); fseek(f, 0, SEEK_SET);
    v.resize(n / 4); size_t g = fread(v.data(), 4, v.size(), f); (void)g; fclose(f);
    return v;
}
static void write_floats(const std::string& path, const std::vector<float>& v) {
    FILE* f = fopen(path.c_str(), "wb");
    if (!f) { fprintf(stderr, "cannot write %s\n", path.c_str()); return; }
    fwrite(v.data(), 4, v.size(), f); fclose(f);
    fprintf(stderr, "wrote %s (%zu floats)\n", path.c_str(), v.size());
}

int main(int argc, char* argv[]) {
    nnopt_install_crash_handler();
    std::setvbuf(stdout, nullptr, _IONBF, 0);
    std::string mode = (argc > 1 && argv[1][0] != '-') ? argv[1] : "full";
    std::string in_wav = "assets/input.wav", out_wav = "output.wav", lang = "eng";
    bool in_given = false;
    bool serve_mode = false;   // --serve: stay resident, one translation per stdin request line
    for (int i = 2; i < argc; ++i) {
        if (!strcmp(argv[i], "--in") && i + 1 < argc) { in_wav = argv[++i]; in_given = true; }
        else if (!strcmp(argv[i], "--out") && i + 1 < argc) out_wav = argv[++i];
        else if (!strcmp(argv[i], "--lang") && i + 1 < argc) lang = argv[++i];
        else if (!strcmp(argv[i], "--serve")) serve_mode = true;
    }
    fprintf(stderr, "=== SeamlessM4T UnitY on-device (mode=%s) ===\n", mode.c_str());

    OpenCLContext cl;
    if (!cl.initialize()) { NNOPT_ERROR("Failed to initialize OpenCL"); return 1; }
    fprintf(stderr, "Device: %s\n", cl.device_name().c_str());

    Weights weights;
#ifdef NNOPT_USE_FP16
    const char* wbin = "weights/model.fp16.bin", *wmeta = "weights/model.fp16.meta.json";
#else
    const char* wbin = "weights/model.bin", *wmeta = "weights/model.meta.json";
#endif
    if (!weights.load(wbin, wmeta, cl.context())) { NNOPT_ERROR_FMT("Failed to load weights %s", wbin); return 1; }

    GpuOps ops(cl, weights);
    if (!ops.init("kernels")) { NNOPT_ERROR("GpuOps init failed"); return 1; }
    Pipeline pipe(ops, weights);

    // Target language control tokens (from the .ptl lang_tok_map / lang_tok_map_unit /
    // vocoder_lang_map / vocoder_spkr_map). This checkpoint supports eng/spa/por/hin/rus.
    struct LangCfg { const char* code; int text_prefix, unit_prefix, voc_lang, voc_spkr; };
    static const LangCfg LANGS[] = {
        {"eng", 20000, 10010,  8, 10}, {"spa", 20001, 10014, 25, 42},
        {"por", 20002, 10012, 21, 38}, {"hin", 20003, 10011, 12, 16},
        {"rus", 20004, 10013, 23, 40},
    };

    // ── Warm server (--serve): model loaded once above; one translation per stdin request line ──
    // Request: "<mode> <lang> <in_wav> <out_wav>"  (out_wav used only for s2s; "-" otherwise).
    // s2tt/asr → translated text on STDOUT; s2s → wav written to out_wav (blank stdout line). Each
    // reply ends with SEAMLESS_DONE on STDERR; SEAMLESS_READY (stderr) signals the model is warm.
    if (serve_mode) {
        Tokenizer stok; stok.load("weights/tokenizer_vocab.bin");
        fprintf(stderr, "SEAMLESS_READY\n"); std::fflush(stderr);
        std::string reqline;
        while (std::getline(std::cin, reqline)) {
            if (reqline.empty()) continue;
            std::istringstream iss(reqline);
            std::string rmode, rlang, rin, rout;
            iss >> rmode >> rlang >> rin >> rout;
            const LangCfg* L = nullptr;
            for (const auto& cfg : LANGS) if (rlang == cfg.code) { L = &cfg; break; }
            auto finish = []() { fprintf(stderr, "SEAMLESS_DONE\n"); std::fflush(stderr); };
            if (rmode.empty() || rin.empty() || !L) {
                fprintf(stderr, "seamless: bad request '%s'\n", reqline.c_str());
                std::fputc('\n', stdout); std::fflush(stdout); finish(); continue;
            }
            pipe.set_lang(L->text_prefix, L->unit_prefix, L->voc_lang, L->voc_spkr);
            WavData w;
            if (!read_wav(rin, w)) {
                fprintf(stderr, "seamless: failed to read %s\n", rin.c_str());
                std::fputc('\n', stdout); std::fflush(stdout); finish(); continue;
            }
            int nframes = (int)w.samples.size() / w.channels;
            Tensor mono = ops.audio_decode(ops.upload_s16(w.samples), nframes, w.channels);
            int n16 = nframes;
            if (w.sample_rate != 16000) { int nout = 0; mono = ops.audio_resample(mono, nframes, w.sample_rate, 16000, nout); n16 = nout; }
            auto audio = ops.download(mono, n16);
            if (rmode == "s2s") {
                std::vector<float> units, wav;
                pipe.run(audio, units, wav);
                Tensor wavT = ops.upload(wav);
                cl_mem s16out = ops.audio_encode_s16(wavT, (int)wav.size());
                auto pcm = ops.download_s16(s16out, (int)wav.size());
                write_wav(rout, pcm, 16000);
                std::fputc('\n', stdout); std::fflush(stdout);   // s2s has no text
            } else {  // s2tt / asr
                int nf = 0; auto fb = pipe.fbank(audio, nf);
                int Tenc = 0; auto enc = pipe.encoder(fb, nf, Tenc);
                auto hypo = pipe.text_beam_search(enc, Tenc);
                std::string text;
                if (hypo.size() >= 2) { std::vector<int32_t> ids(hypo.begin() + 1, hypo.end() - 1); text = stok.decode(ids); }
                fprintf(stdout, "%s\n", text.c_str()); std::fflush(stdout);
            }
            finish();
        }
        ops.free_all();
        return 0;
    }

    const LangCfg* lc = nullptr;
    for (const auto& L : LANGS) if (lang == L.code) { lc = &L; break; }
    if (!lc) {
        NNOPT_ERROR_FMT("unsupported --lang '%s' (this checkpoint supports: eng spa por hin rus)", lang.c_str());
        return 1;
    }
    pipe.set_lang(lc->text_prefix, lc->unit_prefix, lc->voc_lang, lc->voc_spkr);
    fprintf(stderr, "target language: %s (text=%d unit=%d voc_lang=%d voc_spkr=%d)\n",
            lc->code, lc->text_prefix, lc->unit_prefix, lc->voc_lang, lc->voc_spkr);

    if (mode == "vocoder") {
        // Known-units path: assets/e2e_out_1.bin (units, float32). Proves conv kernels.
        auto units = load_floats("assets/e2e_out_1.bin");
        if (units.empty()) { NNOPT_ERROR("missing assets/e2e_out_1.bin (units)"); return 1; }
        fprintf(stderr, "units: %zu\n", units.size());
        auto wav = pipe.vocoder(units);
        write_floats("waveform_out.bin", wav);
    } else if (mode == "fbank") {
        auto audio = load_floats("assets/test_audio_raw.bin");
        int nf = 0;
        auto fb = pipe.fbank(audio, nf);
        fprintf(stderr, "fbank: %d frames\n", nf);
        write_floats("fbank_out.bin", fb);
    } else if (mode == "encoder") {
        std::vector<float> audio;
        if (in_given) {
            WavData w;
            if (!read_wav(in_wav, w)) { NNOPT_ERROR_FMT("encoder: failed to read %s", in_wav.c_str()); return 1; }
            int nframes = (int)w.samples.size() / w.channels;
            Tensor mono = ops.audio_decode(ops.upload_s16(w.samples), nframes, w.channels);
            int n16 = nframes;
            if (w.sample_rate != 16000) { int nout = 0; mono = ops.audio_resample(mono, nframes, w.sample_rate, 16000, nout); n16 = nout; }
            audio = ops.download(mono, n16);
        } else {
            audio = load_floats("assets/test_audio_raw.bin");
        }
        int nf = 0;
        auto fb = pipe.fbank(audio, nf);
        int Tout = 0;
        auto enc = pipe.encoder(fb, nf, Tout);
        fprintf(stderr, "encoder: [%d, 768]\n", Tout);
        write_floats("encoder_out.bin", enc);
    } else if (mode == "s2s") {
        // Speech in -> speech out. WAV decode/resample + waveform encode on GPU.
        WavData w;
        if (!read_wav(in_wav, w)) { NNOPT_ERROR_FMT("s2s: failed to read %s", in_wav.c_str()); return 1; }
        int nframes = (int)w.samples.size() / w.channels;
        cl_mem s16 = ops.upload_s16(w.samples);
        Tensor mono = ops.audio_decode(s16, nframes, w.channels);
        int n16 = nframes;
        if (w.sample_rate != 16000) { int nout = 0; mono = ops.audio_resample(mono, nframes, w.sample_rate, 16000, nout); n16 = nout; }
        auto audio = ops.download(mono, n16);
        fprintf(stderr, "s2s: input %d frames @ %d Hz -> %d samples @ 16kHz\n", nframes, w.sample_rate, n16);
        // Deployment warmup: when int8 is on, the per-row weight quantization is a
        // one-time startup cost (a deployed model is loaded once, run many times).
        // Run the encoder once here (untimed) to populate the int8 cache so the
        // timed pipe.run() below reflects the real warm per-inference latency.
        if (std::getenv("NNOPT_INT8")) {
            int wnf = 0; auto wfb = pipe.fbank(audio, wnf); int wte = 0; pipe.encoder(wfb, wnf, wte);
            fprintf(stderr, "s2s: int8 encoder warmup done\n");
        }
        // NNOPT_WARMUP / NNOPT_INT8DEC: run one full untimed pass so that ALL
        // first-use costs (Adreno online kernel compilation, CLBlast program build,
        // and the one-time int8 per-row weight quantization) are excluded from the
        // timed run. Required for a fair warm steady-state comparison — without it
        // the timed run is dominated by compilation, not compute/bandwidth.
        if (std::getenv("NNOPT_WARMUP") || std::getenv("NNOPT_INT8DEC")) {
            std::vector<float> wu, ww; pipe.run(audio, wu, ww);
            fprintf(stderr, "s2s: warmup done (%zu units)\n", wu.size());
            ops.prof_reset();  // discard cold-compile profile; measure only the warm timed run
        }
        std::vector<float> units, wav;
        long long gb0 = 0, gt0 = 0; bool gbok = read_gpubusy(gb0, gt0);  // reset/start the HW busy window
        pipe.run(audio, units, wav);
        clFinish(ops.cl().queue());  // ensure all GPU work for the timed run is done before reading the counter
        long long gb1 = 0, gt1 = 0; gbok = gbok && read_gpubusy(gb1, gt1);
        if (gbok && (gt1 > 0)) {
            // gpubusy is a delta-since-last-read counter, so gb1/gt1 already cover only this window.
            fprintf(stderr, "GPU_HW_UTIL: busy=%lld total=%lld -> %.1f%% (true GPU-busy over the timed run, all kernels)\n",
                    gb1, gt1, 100.0 * (double)gb1 / (double)gt1);
        }
        Tensor wavT = ops.upload(wav);
        cl_mem s16out = ops.audio_encode_s16(wavT, (int)wav.size());
        auto pcm = ops.download_s16(s16out, (int)wav.size());
        write_wav(out_wav, pcm, 16000);
        write_floats("units_out.bin", units);
    } else if (mode == "s2tt" || mode == "asr") {
        // Speech -> text. Runs fbank+encoder+text-decoder beam search, then decodes the
        // hypothesis to text. Target language is fixed to English in this port, so for
        // English input s2tt (translate-to-English) and asr (transcribe) are the same path.
        std::vector<float> audio;
        if (in_given) {
            WavData w;
            if (!read_wav(in_wav, w)) { NNOPT_ERROR_FMT("s2tt: failed to read %s", in_wav.c_str()); return 1; }
            int nframes = (int)w.samples.size() / w.channels;
            Tensor mono = ops.audio_decode(ops.upload_s16(w.samples), nframes, w.channels);
            int n16 = nframes;
            if (w.sample_rate != 16000) { int nout = 0; mono = ops.audio_resample(mono, nframes, w.sample_rate, 16000, nout); n16 = nout; }
            audio = ops.download(mono, n16);
        } else {
            audio = load_floats("assets/test_audio_raw.bin");
        }
        if (audio.empty()) { NNOPT_ERROR("s2tt: no input audio"); return 1; }
        int nf = 0; auto fb = pipe.fbank(audio, nf);
        int Tenc = 0; auto enc = pipe.encoder(fb, nf, Tenc);
        auto hypo = pipe.text_beam_search(enc, Tenc);
        fprintf(stderr, "hypo:"); for (int t : hypo) fprintf(stderr, " %d", t); fprintf(stderr, "\n");
        Tokenizer tok;
        std::string text;
        if (tok.load("weights/tokenizer_vocab.bin") && hypo.size() >= 2) {
            std::vector<int32_t> ids(hypo.begin() + 1, hypo.end() - 1);  // drop lang prefix + eos
            text = tok.decode(ids);
        } else {
            NNOPT_ERROR("s2tt: tokenizer load failed or empty hypo");
        }
        fprintf(stdout, "%s\n", text.c_str());
        FILE* tf = fopen("text_out.txt", "w");
        if (tf) { fputs(text.c_str(), tf); fputs("\n", tf); fclose(tf); }
    } else if (mode == "s2tt_all") {
        // Speech -> text for ALL target languages in ONE process. The speech encoder is
        // language-independent, so compute fbank+encoder ONCE and reuse `enc` across langs;
        // only the text decoder's target-lang prefix changes. A warmup pass compiles the
        // CLBlast shapes so the timed numbers are warm steady-state.
        std::vector<float> audio;
        if (in_given) {
            WavData w;
            if (!read_wav(in_wav, w)) { NNOPT_ERROR_FMT("s2tt_all: failed to read %s", in_wav.c_str()); return 1; }
            int nframes = (int)w.samples.size() / w.channels;
            Tensor mono = ops.audio_decode(ops.upload_s16(w.samples), nframes, w.channels);
            int n16 = nframes;
            if (w.sample_rate != 16000) { int nout = 0; mono = ops.audio_resample(mono, nframes, w.sample_rate, 16000, nout); n16 = nout; }
            audio = ops.download(mono, n16);
        } else { audio = load_floats("assets/test_audio_raw.bin"); }
        if (audio.empty()) { NNOPT_ERROR("s2tt_all: no input audio"); return 1; }
        Tokenizer tok; tok.load("weights/tokenizer_vocab.bin");
        auto decode_lang = [&](const std::vector<float>& e, int Te, const LangCfg& L) -> std::string {
            pipe.set_lang(L.text_prefix, L.unit_prefix, L.voc_lang, L.voc_spkr);
            auto hypo = pipe.text_beam_search(e, Te);
            if (hypo.size() < 2) return std::string();
            std::vector<int32_t> ids(hypo.begin() + 1, hypo.end() - 1);
            return tok.decode(ids);
        };
        auto now = []{ return std::chrono::steady_clock::now(); };
        auto ms = [](std::chrono::steady_clock::time_point a, std::chrono::steady_clock::time_point b){
            return std::chrono::duration<double, std::milli>(b - a).count(); };
        // warmup (compile CLBlast/kernels) — untimed
        { int nf = 0; auto fb = pipe.fbank(audio, nf); int Te = 0; auto e = pipe.encoder(fb, nf, Te); decode_lang(e, Te, LANGS[0]); }
        // timed: encoder once, then all langs
        auto tA = now();
        int nf = 0; auto fb = pipe.fbank(audio, nf);
        int Tenc = 0; auto enc = pipe.encoder(fb, nf, Tenc);
        clFinish(ops.cl().queue());
        auto tEnc = now();
        fprintf(stderr, "s2tt_all: encoder %.0f ms (T_enc=%d), input %zu samples\n", ms(tA, tEnc), Tenc, audio.size());
        double total_txt = 0;
        for (const auto& L : LANGS) {
            auto tl = now();
            std::string text = decode_lang(enc, Tenc, L);
            clFinish(ops.cl().queue());
            double dt = ms(tl, now()); total_txt += dt;
            fprintf(stdout, "  %s (%.0f ms): %s\n", L.code, dt, text.c_str());
            fflush(stdout);
        }
        fprintf(stderr, "s2tt_all: TOTAL %.0f ms (encoder %.0f + 5×text %.0f)\n", ms(tA, now()), ms(tA, tEnc), total_txt);
    } else if (mode == "decenc") {
        // Debug: decode from a reference encoder output (assets/ref_enc.bin, float32 [T,768]).
        auto enc = load_floats("assets/ref_enc.bin");
        if (enc.empty()) { NNOPT_ERROR("decenc: missing assets/ref_enc.bin"); return 1; }
        int Tenc = (int)enc.size() / 768;
        fprintf(stderr, "decenc: injected encoder [%d, 768]\n", Tenc);
        std::vector<float> units, wav;
        pipe.run_from_encoder(enc, Tenc, units, wav);
        write_floats("units_out.bin", units);
    } else {  // full (S2ST on the bundled fixture)
        auto audio = load_floats("assets/test_audio_raw.bin");
        if (audio.empty()) { NNOPT_ERROR("missing assets/test_audio_raw.bin"); return 1; }
        std::vector<float> units, wav;
        pipe.run(audio, units, wav);
        write_floats("units_out.bin", units);
        write_floats("waveform_out.bin", wav);
    }

    ops.dump_gpu_prof();   // NNOPT_GPUPROF: per-kernel true GPU exec-time breakdown
    ops.free_all();
    std::fflush(stderr);
    fprintf(stderr, "NNOPT_EXIT_CLEAN exit_code=0\n");
    std::fflush(stderr);
    return 0;
}
