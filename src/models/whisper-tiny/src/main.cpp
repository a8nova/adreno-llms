#include <algorithm>
// Auto-generated graph-mode main.cpp for openai/whisper-tiny.
// Modality: input=audio, output=tokens
//
// In graph mode main.cpp is small: tokenize/load input, instantiate Model,
// loop a generate step (or single forward for non-autoregressive), write
// output. The interesting code lives in src/ops/*.cpp.
//
// API REFERENCE (DO NOT GUESS — these are the canonical signatures the
// scaffold emits; matching them avoids the iteration tax of inventing
// non-existent methods like load_from_disk / decode_one / sample_argmax):
//
//   class Weights {
//     Weights();                                 // default ctor only — no OpenCLContext arg
//     bool load(const std::string& bin_path,
//               const std::string& meta_path,
//               cl_context ctx);
//     cl_mem get_buffer(const std::string& key, bool optional=false);
//     // --- Introspection (use these when CLBlast returns kInsufficientMemoryB
//     //     or any dim-mismatch error: compare your op's expected dim against
//     //     the actual on-disk shape BEFORE editing model_config.h or kernels):
//     bool has_tensor(const std::string& key) const;
//     std::vector<int> get_shape(const std::string& key) const;       // [] if missing
//     size_t get_num_elements(const std::string& key) const;          // 0 if missing
//     size_t get_size_bytes(const std::string& key) const;            // 0 if missing
//     std::string get_dtype(const std::string& key) const;            // "" if missing
//   };
//   class Tokenizer {
//     bool load(const std::string& vocab_path); // singular path, NOT (bin, meta)
//     std::vector<int> encode(const std::string& text);
//     std::string decode(const std::vector<int32_t>& ids); // takes vector, NOT single int
//     int eos_token_id() const;
//   };
//   class Sampler {                              // class-based, NOT a free function
//     Sampler(const SamplerConfig& cfg);
//     int sample(std::vector<float>& logits,
//                const std::vector<int32_t>& generated_ids) const;
//   };

#include "model.h"
#include "model_config.h"
#include "opencl_context.h"
#include "weights.h"
#include "sampler.h"
#include "tokenizer.h"
#include "debug_utils.h"
#include "version.h"
#include "benchmark.h"  // BenchmarkTimer + NNOPT_BENCH_FIRST_TOKEN — see prefill/decode call sites below
#include "profiler.h"   // KernelProfiler::dump_summary — dormant unless NNOPT_PROFILE=1.



#include "load_bin.h"    // load_float_bin (mel fixture loader)
#include "forward_dispatch_audio.h"  // ForwardDispatch::set_input_features
#include "utils.h"             // nnopt_storage_t + nnopt_f32_to_f16 (fp16 conversion)
#include "mel_frontend.h"      // whisper_log_mel — on-device raw audio -> log-mel [80,3000]



#include <fstream>
#include <iostream>
#include <string>
#include <vector>
#include <cstdint>
#include <cstring>
#include <chrono>
#include <cmath>     // std::sqrt — streaming VAD frame energy
#include <cstdio>    // std::fread / stdin — streaming PCM input

// Read prompt input ids from a binary file (int32 little-endian) when the
// caller passes \`--token-ids /path/to/test_input_ids.bin\`. This path is
// the deterministic-evaluation contract — Evaluate compares the C++ output
// against the PyTorch reference for the SAME input ids, so the binary
// MUST consume the file rather than re-tokenizing the prompt string (which
// can produce different ids if the encoder isn't perfectly aligned with
// HF's tokenizer for this model family).
static bool read_input_ids_bin(const std::string& path, std::vector<int32_t>& out) {
    std::ifstream f(path, std::ios::binary);
    if (!f) return false;
    f.seekg(0, std::ios::end);
    const std::streamsize n_bytes = f.tellg();
    f.seekg(0, std::ios::beg);
    if (n_bytes <= 0 || (n_bytes % sizeof(int32_t)) != 0) return false;
    out.resize(static_cast<size_t>(n_bytes) / sizeof(int32_t));
    f.read(reinterpret_cast<char*>(out.data()), n_bytes);
    return f.good() || f.eof();
}

// ── Streaming mode (--stream): continuous 16kHz mono float32 PCM on stdin,
//    energy VAD + sliding window. Emits `PARTIAL: <text>` while a phrase is in
//    progress (re-transcribed every step) and `FINAL: <text>` when the VAD
//    detects end-of-speech (silence hangover) or the 30s Whisper window cap is
//    hit. Reuses the exact mel->encode->decode pipeline (model.forward) per
//    window; the only new model-side hook is the encoder-cache invalidator so a
//    changing window doesn't false-hit a stale encoder output.
//    NOTE: this duplicates the per-window decode core of the --audio path on
//    purpose — to keep the validated --audio-list benchmark path untouched. The
//    shared core belongs in the tool's asr scaffold; unify there, not by hand.
extern "C" void WhisperBackbone_invalidate_encoder_cache();

namespace {

// Transcribe one window of 16kHz mono float audio -> text.
std::string nnopt_transcribe_window(OpenCLContext& cl_ctx, Model& model, Tokenizer& tok,
                                    bool tokenizer_ok, Sampler& sampler,
                                    const SamplerConfig& scfg, int max_new_tokens,
                                    const std::vector<float>& wav,
                                    const std::vector<float>& mel_filters) {
    if (wav.empty()) return std::string();
    std::vector<float> mel = whisper_log_mel(wav, mel_filters);
    if (mel.empty()) return std::string();

    std::vector<nnopt_storage_t> mel_storage(mel.size());
#ifdef NNOPT_USE_FP16
    for (size_t i = 0; i < mel.size(); ++i) mel_storage[i] = nnopt_f32_to_f16(mel[i]);
#else
    for (size_t i = 0; i < mel.size(); ++i) mel_storage[i] = (nnopt_storage_t)mel[i];
#endif
    cl_int err = CL_SUCCESS;
    cl_mem feats = clCreateBuffer(cl_ctx.context(), CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR,
                                  mel_storage.size() * sizeof(nnopt_storage_t),
                                  mel_storage.data(), &err);
    if (err != CL_SUCCESS || !feats) { NNOPT_ERROR_FMT("stream: feats buffer alloc %d", (int)err); return std::string(); }

    WhisperBackbone_invalidate_encoder_cache();   // force recompute for THIS window's audio
    ForwardDispatch::set_input_features(feats);

    std::vector<int32_t> prompt_ids = {50258};    // <|startoftranscript|>
    const size_t prompt_len = prompt_ids.size();
    std::vector<float> logits = model.forward(prompt_ids, /*start_pos=*/0);
    int kv_pos = (int)prompt_ids.size();

    for (int step = 0; step < max_new_tokens; ++step) {
        std::vector<int32_t> gen_so_far(prompt_ids.begin() + prompt_len, prompt_ids.end());
        int next = sampler.sample(logits, gen_so_far);
        { static const int kForced[3] = {50259, 50359, 50363};   // <|en|><|transcribe|><|notimestamps|>
          const int gp = (int)prompt_ids.size() - (int)prompt_len;
          if (gp >= 0 && gp < 3) next = kForced[gp]; }
        prompt_ids.push_back(next);
        if (scfg.eos_token_id >= 0 && next == scfg.eos_token_id) break;
        if (step + 1 >= max_new_tokens) break;
        ForwardDispatch::set_input_features(feats);
        logits = model.forward(std::vector<int32_t>{next}, kv_pos);
        kv_pos++;
    }

    clReleaseMemObject(feats);
    WhisperBackbone_invalidate_encoder_cache();   // don't leave this window's encoder cached

    std::string text;
    if (prompt_ids.size() > prompt_len && tokenizer_ok) {
        const std::vector<int32_t> gen(prompt_ids.begin() + prompt_len, prompt_ids.end());
        text = tok.decode(gen);
    }
    return text;
}

int nnopt_run_stream(OpenCLContext& cl_ctx, Model& model, Tokenizer& tok, bool tokenizer_ok,
                     Sampler& sampler, const SamplerConfig& scfg, int max_new_tokens,
                     float vad_threshold, int step_ms, int hangover_ms) {
    if (!tokenizer_ok) { NNOPT_ERROR("stream: tokenizer required for streaming output"); return 4; }
    std::vector<float> mel_filters = load_float_bin("assets/mel_filters.bin");
    if (mel_filters.size() != 80u * 201u) {
        NNOPT_ERROR_FMT("stream: assets/mel_filters.bin wrong size %zu (need %u)", mel_filters.size(), 80u * 201u);
        return 4;
    }

    const int   SR              = 16000;
    const int   frame           = SR * 30 / 1000;             // 30ms VAD frame = 480 samples
    const size_t step_samples   = (size_t)SR * step_ms / 1000;
    const int   hangover_frames = (hangover_ms > 0 ? hangover_ms : 1) / 30;
    const int   trigger_frames  = 3;                          // consecutive voiced frames to start a phrase
    const float min_phrase_s    = 0.30f;                      // ignore sub-300ms blips
    const float max_phrase_s    = 28.0f;                      // force-commit before Whisper's 30s window
    const int   cap             = max_new_tokens > 16 ? max_new_tokens : 96;  // per-phrase token cap

    std::vector<float> window;     // accumulated samples of the in-progress phrase
    std::vector<float> framebuf;   // leftover samples between stdin reads (sub-frame)
    bool   in_speech = false;
    int    speech_run = 0, silence_run = 0;
    size_t samples_since_partial = 0;

    auto commit = [&](bool final_) {
        const float dur = (float)window.size() / (float)SR;
        if (window.empty() || dur < min_phrase_s) { if (final_) window.clear(); return; }
        const std::string txt = nnopt_transcribe_window(cl_ctx, model, tok, tokenizer_ok,
                                                         sampler, scfg, cap, window, mel_filters);
        std::cout << (final_ ? "FINAL: " : "PARTIAL: ") << txt << std::endl;
        if (final_) window.clear();
    };

    fprintf(stderr, "STREAM: ready — feed 16kHz mono float32 PCM on stdin "
                    "(vad_threshold=%.4f step=%dms hangover=%dms cap=%d tok)\n",
            vad_threshold, step_ms, hangover_ms, cap);
    fflush(stderr);

    std::vector<float> rd(4096);
    while (true) {
        const size_t got = std::fread(rd.data(), sizeof(float), rd.size(), stdin);
        if (got == 0) break;   // EOF / closed pipe
        framebuf.insert(framebuf.end(), rd.begin(), rd.begin() + got);

        size_t off = 0;
        while (framebuf.size() - off >= (size_t)frame) {
            double e = 0.0;
            for (int i = 0; i < frame; ++i) { const float s = framebuf[off + i]; e += (double)s * s; }
            const float rms = (float)std::sqrt(e / (double)frame);
            const bool voiced = rms >= vad_threshold;
            if (voiced) { ++speech_run; silence_run = 0; }
            else        { ++silence_run; speech_run = 0; }

            if (!in_speech && speech_run >= trigger_frames) in_speech = true;

            if (in_speech) {
                window.insert(window.end(), framebuf.begin() + off, framebuf.begin() + off + frame);
                samples_since_partial += frame;
                if (silence_run >= hangover_frames || (float)window.size() / (float)SR >= max_phrase_s) {
                    commit(/*final_=*/true);
                    in_speech = false; speech_run = silence_run = 0; samples_since_partial = 0;
                } else if (samples_since_partial >= step_samples) {
                    commit(/*final_=*/false);
                    samples_since_partial = 0;
                }
            }
            off += frame;
        }
        framebuf.erase(framebuf.begin(), framebuf.begin() + off);
    }

    if (in_speech && !window.empty()) commit(/*final_=*/true);   // flush trailing phrase on EOF
    fprintf(stderr, "STREAM: end (stdin EOF)\n");
    fflush(stderr);
    KernelProfiler::dump_summary();   // dormant unless NNOPT_PROFILE=1
    return 0;
}

}  // namespace

int main(int argc, char** argv) {
    // Arm the crash handler FIRST — on SIGSEGV/SIGABRT/SIGBUS it prints the
    // last NNOPT_CHECKPOINT, a backtrace, and the GPU-mem allocation log so a
    // device segfault reports WHERE it died instead of a bare "Segmentation
    // fault". Parity with main.cpp.tmpl:44 (graph-mode main dropped this in the
    // Phase 3 redesign, blinding every graph-mode port to crash locations).
    nnopt_install_crash_handler();
    // (No version banner — debug_utils does not define one, and emitting an
    // undefined macro here was breaking every fresh-port build.)
    // Argument parsing: positional "prompt" + optional flags.
    //   ./binary "<prompt>" [max_new_tokens] [--token-ids <file>]
    std::string prompt = "The teacher worked at the ";
    int max_new_tokens = 16;
    std::string token_ids_path;
    std::string audio_path;       // --audio: raw 16kHz mono float32 waveform .bin
    double audio_seconds = 0.0;   // --audio-seconds: clip duration (for RTF)
    std::string audio_list_path;  // --audio-list: TSV "wav_path<TAB>duration" per line —
                                  // transcribe MANY clips in ONE process so the per-process
                                  // JIT compile (CLBlast + .cl) is paid once, not per clip.
    bool   stream_mode        = false;   // --stream: continuous PCM-on-stdin + VAD + sliding window
    float  vad_threshold      = 0.01f;   // --vad-threshold: per-frame RMS speech threshold (normalized f32 audio)
    int    stream_step_ms     = 1500;    // --step-ms: re-transcribe the growing window every N ms (partials)
    int    stream_hangover_ms = 800;     // --hangover-ms: silence after speech that commits a phrase (FINAL)

    SamplerConfig sampler_config;
    sampler_config.temperature = 0.0f;          // greedy by default — matches PyTorch reference
    sampler_config.top_k = 1;
    sampler_config.top_p = 1.0f;
    sampler_config.repetition_penalty = 1.0f;
    sampler_config.seed = 42u;
    {
        int positional = 0;
        for (int i = 1; i < argc; i++) {
            const std::string a = argv[i];
            if (a == "--token-ids" && i + 1 < argc) {
                token_ids_path = argv[++i];
            } else if (a == "--audio" && i + 1 < argc) {
                audio_path = argv[++i];
            } else if (a == "--audio-seconds" && i + 1 < argc) {
                audio_seconds = std::stod(argv[++i]);
            } else if (a == "--audio-list" && i + 1 < argc) {
                audio_list_path = argv[++i];
            } else if (a == "--temperature" && i + 1 < argc) {
                sampler_config.temperature = std::stof(argv[++i]);
            } else if (a == "--top-k" && i + 1 < argc) {
                sampler_config.top_k = std::stoi(argv[++i]);
            } else if (a == "--top-p" && i + 1 < argc) {
                sampler_config.top_p = std::stof(argv[++i]);
            } else if (a == "--seed" && i + 1 < argc) {
                sampler_config.seed = static_cast<uint32_t>(std::stoul(argv[++i]));
            } else if (a == "--stream") {
                stream_mode = true;
            } else if (a == "--vad-threshold" && i + 1 < argc) {
                vad_threshold = std::stof(argv[++i]);
            } else if (a == "--step-ms" && i + 1 < argc) {
                stream_step_ms = std::stoi(argv[++i]);
            } else if (a == "--hangover-ms" && i + 1 < argc) {
                stream_hangover_ms = std::stoi(argv[++i]);
            } else if (a.rfind("--", 0) == 0) {
                // Unknown flag — skip the value too if present.
                if (i + 1 < argc && argv[i + 1][0] != '-') ++i;
            } else if (positional == 0) {
                prompt = a; positional++;
            } else if (positional == 1) {
                max_new_tokens = std::stoi(a); positional++;
            }
        }
    }

    // Benchmark instrumentation — emits BENCHMARK <key>: <value> lines on stderr,
    // parsed by runUtils.ts::parseInferenceMetrics. Five call sites total:
    //   1. mark_inference_start()        — here, immediately after arg parsing
    //   2. mark_prefill_start()          — immediately before the first model.forward()
    //   3. NNOPT_BENCH_FIRST_TOKEN()     — inside generate loop, right after first sampler.sample()
    //   4. mark_end()                    — after the decode loop exits
    //   5. print_summary(prompt_len, gen) — just before return 0
    // If you restructure prefill+decode (e.g. split into a single batched prefill
    // forward followed by per-token decode forwards), KEEP all five sites — they
    // partition wall-clock into ttft / prefill / decode correctly regardless of
    // loop shape. Removing any of them silently emits -1 for that metric.
    BenchmarkTimer& bench = BenchmarkTimer::instance();
    bench.mark_inference_start();

    OpenCLContext cl_ctx;
    if (!cl_ctx.initialize()) {
        NNOPT_ERROR("OpenCL init failed");
        return 1;
    }

    // Tokenizer first — its eos id feeds into sampler config below.
    Tokenizer tok;
    const bool tokenizer_ok = tok.load("weights/tokenizer_vocab.bin");
    if (!tokenizer_ok && token_ids_path.empty()) {
        NNOPT_ERROR("tokenizer load failed (use --token-ids to bypass for deterministic eval)");
        return 1;
    }
    if (tokenizer_ok) {
        sampler_config.eos_token_id = tok.eos_token_id();
    }

    // Weights — fp16 build pulls weights/model.fp16.bin; fp32 uses weights/model.bin.
    Weights weights;
#ifdef NNOPT_USE_FP16
    const char* nnopt_weights_bin  = "weights/model.fp16.bin";
    const char* nnopt_weights_meta = "weights/model.fp16.meta.json";
#else
    const char* nnopt_weights_bin  = "weights/model.bin";
    const char* nnopt_weights_meta = "weights/model.meta.json";
#endif
    if (!weights.load(nnopt_weights_bin, nnopt_weights_meta, cl_ctx.context())) {
        NNOPT_ERROR_FMT("weights load failed: %s", nnopt_weights_bin);
        return 1;
    }

    Sampler sampler(sampler_config);

    Model model(cl_ctx, weights);
    if (!model.initialize()) {
        NNOPT_ERROR("Model::initialize() failed — see prior NNOPT_ERROR for the layer that failed");
        return 1;
    }

    // Streaming mode: continuous PCM-on-stdin + VAD + sliding window. Separate
    // from the --audio/--audio-list (eval/benchmark) paths below; returns when
    // stdin closes.
    if (stream_mode) {
        return nnopt_run_stream(cl_ctx, model, tok, tokenizer_ok, sampler, sampler_config,
                                max_new_tokens, vad_threshold, stream_step_ms, stream_hangover_ms);
    }

    // Eagerly upload all weight buffers to the GPU now (opt #9), BEFORE the RTF
    // timer starts. Otherwise the first forward lazily uploads ~72 MB inside the
    // timed window — a one-time model-load cost that a real deployment pays once
    // and amortizes over many clips, not a per-clip processing cost.
    {
        const auto _w0 = std::chrono::high_resolution_clock::now();
        const size_t _nbuf = weights.preload_all();
        const double _wms = std::chrono::duration<double, std::milli>(
            std::chrono::high_resolution_clock::now() - _w0).count();
        if (std::getenv("NNOPT_PROFILE")) {
            std::cerr << "PRELOAD " << _nbuf << " weight buffers in " << _wms << " ms\n";
        }
    }




    // Prefer test_input_ids.bin (deterministic) over re-encoding the prompt.
    // The reference's "generated_text" field is computed from these EXACT
    // ids in PyTorch — comparing against a re-encoded prompt would mask
    // tokenizer drift as model error.
    const bool audio_mode = !audio_path.empty() || !audio_list_path.empty();

    // Build the clip work-list. --audio-list (TSV: "wav_path<TAB>duration") runs
    // MANY clips in ONE process so the per-process JIT compile (CLBlast + our .cl
    // kernels, ~4-5s) is paid once for clip 0, not re-paid every clip — matching a
    // real deployment that loads the model once and streams audio. --audio runs a
    // single clip. (lever #1)
    std::vector<std::pair<std::string, double>> _clips;
    if (!audio_list_path.empty()) {
        std::ifstream _lf(audio_list_path);
        std::string _ln;
        while (std::getline(_lf, _ln)) {
            if (_ln.empty()) continue;
            const size_t _tab = _ln.find('\t');
            std::string _p = (_tab == std::string::npos) ? _ln : _ln.substr(0, _tab);
            double _d = (_tab == std::string::npos) ? 0.0 : std::stod(_ln.substr(_tab + 1));
            _clips.push_back({_p, _d});
        }
        if (_clips.empty()) { NNOPT_ERROR_FMT("--audio-list %s is empty/unreadable", audio_list_path.c_str()); return 4; }
    } else {
        _clips.push_back({audio_path, audio_seconds});
    }

    // ── Per-clip loop. Within one process the model/weights/compiled kernels are
    //    reused; WhisperSdpaAttention_reset_caches() (called from the backbone on
    //    each encoder re-run) clears the per-clip KV caches between clips. ──
    // The encoder-output cache is keyed by the input-features buffer POINTER, so
    // each clip MUST get a distinct live pointer — otherwise freeing+reallocating
    // recycles the handle and clip N false-hits clip N-1's encoder output. Keep all
    // feats buffers alive until the run ends to guarantee distinct pointers.
    std::vector<cl_mem> _feats_keepalive;
    for (size_t _ci = 0; _ci < _clips.size(); ++_ci) {
    audio_path = _clips[_ci].first;
    audio_seconds = _clips[_ci].second;

    // Decoder prompt. Whisper's decoder starts from the SOT token (50258); the
    // next three positions are force-decoded (<|en|> <|transcribe|> <|notimestamps|>)
    // in the loop below. --token-ids overrides for deterministic eval.
    std::vector<int32_t> prompt_ids;
    if (!token_ids_path.empty() && read_input_ids_bin(token_ids_path, prompt_ids)) {
        // OK — authoritative ids from disk.
    } else if (audio_mode) {
        prompt_ids = {50258};  // <|startoftranscript|>
    } else {
        const std::vector<int> encoded = tok.encode(prompt);
        prompt_ids.assign(encoded.begin(), encoded.end());
    }
    const size_t prompt_len = prompt_ids.size();

    // RTF timer starts at the front of the pipeline (mel extraction), so the
    // reported processing time covers EVERYTHING the device does end-to-end:
    // raw audio -> log-mel, encode, decode. RTF = processing_seconds / audio_seconds.
    const auto t_proc_start = std::chrono::high_resolution_clock::now();
    double mel_ms = 0.0;

    // ── Input features: compute log-mel ON-DEVICE from a raw 16kHz waveform
    //    (--audio, true end-to-end) OR load a precomputed mel fixture (eval). ──
    std::vector<float> _mel_f32;
    if (audio_mode) {
        std::vector<float> wav = load_float_bin(audio_path);
        if (wav.empty()) { NNOPT_ERROR_FMT("failed to load --audio waveform: %s", audio_path.c_str()); return 4; }
        std::vector<float> mel_filters = load_float_bin("assets/mel_filters.bin");
        if (mel_filters.size() != 80u * 201u) {
            NNOPT_ERROR_FMT("assets/mel_filters.bin wrong size %zu (need %u)", mel_filters.size(), 80u * 201u);
            return 4;
        }
        const auto t_mel0 = std::chrono::high_resolution_clock::now();
        _mel_f32 = whisper_log_mel(wav, mel_filters);
        mel_ms = std::chrono::duration<double, std::milli>(
                     std::chrono::high_resolution_clock::now() - t_mel0).count();
        if (audio_seconds <= 0.0) audio_seconds = (double)wav.size() / 16000.0;
    } else {
        _mel_f32 = load_float_bin("assets/test_input_features.bin");
        if (_mel_f32.empty()) {
            NNOPT_ERROR("missing assets/test_input_features.bin (required for audio encoder-decoder; did GenerateReference run?)");
            return 4;
        }
    }
    std::vector<nnopt_storage_t> _mel_storage(_mel_f32.size());
#ifdef NNOPT_USE_FP16
    for (size_t i = 0; i < _mel_f32.size(); ++i) _mel_storage[i] = nnopt_f32_to_f16(_mel_f32[i]);
#else
    for (size_t i = 0; i < _mel_f32.size(); ++i) _mel_storage[i] = (nnopt_storage_t)_mel_f32[i];
#endif
    cl_int _feat_err = CL_SUCCESS;
    cl_mem _feats_buf = clCreateBuffer(cl_ctx.context(),
                                       CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR,
                                       _mel_storage.size() * sizeof(nnopt_storage_t),
                                       _mel_storage.data(),
                                       &_feat_err);
    if (_feat_err != CL_SUCCESS || !_feats_buf) {
        NNOPT_ERROR_FMT("clCreateBuffer(input_features) failed: %d", (int)_feat_err);
        return 4;
    }


    std::string streamed;  // text already emitted to stdout (audio streaming mode)
    if (audio_mode) std::cout << "TRANSCRIPT: " << std::flush;

    bench.mark_prefill_start();

    // ── KV-cached incremental decode (opt #12) ──────────────────────────────
    // PREFILL: run the prompt once (start_pos=0) to fill the encoder-output cache
    // and the decoder self-attn KV caches. Each subsequent step feeds ONLY the new
    // token at its absolute position, so the decoder processes 1 token/step instead
    // of reprocessing the whole growing prefix (O(N²) → O(N)). The KV caches inside
    // WhisperSdpaAttention supply the past-token context.
    ForwardDispatch::set_input_features(_feats_buf);
    const auto _pf0 = std::chrono::high_resolution_clock::now();
    std::vector<float> logits = model.forward(prompt_ids, /*start_pos=*/0);
    if (std::getenv("NNOPT_PROFILE")) {
        std::cerr << "PREFILL_FORWARD " << std::chrono::duration<double, std::milli>(
            std::chrono::high_resolution_clock::now() - _pf0).count() << " ms (encode + all first-use JIT compile)\n";
    }
    int kv_pos = (int)prompt_ids.size();   // absolute position of the next token fed

    for (int step = 0; step < max_new_tokens; ++step) {
        // Sample one token from the logits via the scaffold's Sampler class.
        std::vector<int32_t> generated_so_far(prompt_ids.begin() + prompt_len, prompt_ids.end());
        int next = sampler.sample(logits, generated_so_far);
        // Whisper forced_decoder_ids: after the start token (sot=50258), the first
        // three generated positions are FORCED to <|en|>, <|transcribe|>,
        // <|notimestamps|>. PyTorch's generate() applies these via a logits
        // processor; without it the model samples freely and never matches the
        // reference. gen_pos is the 0-based index of the token being generated.
        {
            static const int kWhisperForced[3] = {50259, 50359, 50363};
            const int gen_pos = (int)prompt_ids.size() - (int)prompt_len;
            if (gen_pos >= 0 && gen_pos < 3) next = kWhisperForced[gen_pos];
        }
        NNOPT_BENCH_FIRST_TOKEN();  // stamps TTFT on the first sample only
        prompt_ids.push_back(next);
        if (sampler_config.eos_token_id >= 0 && next == sampler_config.eos_token_id) break;
        // Real-time streaming (audio mode): decode the generated ids so far and
        // print only the newly-revealed suffix. Decoding the full prefix each
        // step (rather than per-token) keeps multi-byte BPE/UTF-8 pieces intact;
        // skip_special_tokens drops the forced <|...|> control ids automatically.
        if (audio_mode && tokenizer_ok) {
            const std::vector<int32_t> gen(prompt_ids.begin() + prompt_len, prompt_ids.end());
            const std::string full = tok.decode(gen);
            if (full.size() > streamed.size()) {
                std::cout << full.substr(streamed.size()) << std::flush;
                streamed = full;
            }
        }
        // Eval mode keeps a single stdout source (the post-loop GENERATED_TEXT
        // summary) so Evaluate's parser can't misclassify concatenated streams.

        if (step + 1 >= max_new_tokens) break;  // cap reached — skip the unused forward
        // DECODE: feed ONLY the new token at its absolute position; the KV caches
        // hold the prefix. The encoder is reused from its cache (same _feats_buf).
        ForwardDispatch::set_input_features(_feats_buf);
        const bool _time_dec = (step == 0) && std::getenv("NNOPT_PROFILE");
        const auto _dc0 = std::chrono::high_resolution_clock::now();
        logits = model.forward(std::vector<int32_t>{next}, kv_pos);
        if (_time_dec) {
            std::cerr << "DECODE_FORWARD[0] " << std::chrono::duration<double, std::milli>(
                std::chrono::high_resolution_clock::now() - _dc0).count() << " ms (steady-state, no compile)\n";
        }
        kv_pos++;
    }

    _feats_keepalive.push_back(_feats_buf);  // released after the loop (keep pointer distinct)

    bench.mark_end();
    std::cout << std::endl;
    // Emit a deterministic post-generation summary that matches Evaluate's
    // contract: ONLY the generated tokens (sliced after prompt_len), so
    // comparison apples-to-apples with reference_tokens.json::generated_text.
    if (prompt_ids.size() > prompt_len) {
        const std::vector<int32_t> generated_ids(prompt_ids.begin() + prompt_len, prompt_ids.end());
        if (tokenizer_ok) {
            std::cout << "GENERATED_TEXT: " << tok.decode(generated_ids) << std::endl;
        } else {
            std::cout << "GENERATED_IDS:";
            for (int32_t id : generated_ids) std::cout << " " << id;
            std::cout << std::endl;
        }
    }
    // Real-Time Factor: total on-device processing (audio->mel + encode + decode)
    // divided by the clip's audio duration. RTF < 1 == faster than real-time.
    if (audio_mode) {
        const double proc_s = std::chrono::duration<double>(
            std::chrono::high_resolution_clock::now() - t_proc_start).count();
        const double rtf = audio_seconds > 0.0 ? proc_s / audio_seconds : 0.0;
        char line[256];
        snprintf(line, sizeof(line),
                 "RTF: %.3f  (audio=%.2fs proc=%.2fs mel=%.3fs)",
                 rtf, audio_seconds, proc_s, mel_ms / 1000.0);
        std::cout << line << std::endl;
    }
    bench.print_summary((int)prompt_len, (int)(prompt_ids.size() - prompt_len));
    } // end per-clip loop (--audio-list)

    for (cl_mem _fb : _feats_keepalive) if (_fb) clReleaseMemObject(_fb);

    // Per-kernel GPU profile (env NNOPT_PROFILE=1). Dormant by default;
    // KEEP this call site even when restructuring prefill/decode (same
    // rule as the 5 benchmark sites — see vlm.md).
    KernelProfiler::dump_summary();
    return 0;

}
