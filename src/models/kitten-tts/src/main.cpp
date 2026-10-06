// Auto-generated graph-mode main.cpp for KittenML/kitten-tts-nano-0.1.
// Modality: input=tokens, output=waveform
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
// sampler.h intentionally NOT included — this is an audio-out (TTS) model with
// no token-sampling stage. The scaffold skips sampler generation for this
// modality (see model.h). The single-shot forward_graph path below produces a
// waveform directly; there is no logits->token sampling loop.
#include "tokenizer.h"
#include "debug_utils.h"
#include "version.h"
#include "benchmark.h"  // BenchmarkTimer + NNOPT_BENCH_FIRST_TOKEN — see prefill/decode call sites below
#include "profiler.h"   // KernelProfiler::dump_summary — dormant unless NNOPT_PROFILE=1.


#include "load_bin.h"    // load_int32_bin, load_float_bin (fixture loaders)
#include "write_wav.h"   // write_wav (RIFF int16 PCM, 16kHz mono)



#include <fstream>
#include <iostream>
#include <string>
#include <vector>
#include <cstdint>
#include <cstring>
#include <chrono>

#include "phonemizer.h"  // on-device G2P + chunk_text (impl compiled iff NNOPT_TTS_STREAMING)

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

void run_microbench(OpenCLContext& ctx);

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
    std::string voice_path = "assets/voices__expr-voice-2-m.bin";
    std::string out_path;
    bool stream_mode = false;        // --stream: on-device G2P + chunked PCM to stdout
    bool serve_stream_mode = false;  // --serve-stream: persistent --stream REPL (app drives this)
    std::string espeak_voice = "en-us";  // --espeak-voice: espeak G2P voice (KittenTTS ref = en-us)
    (void)max_new_tokens;  // TTS single-shot: no autoregressive token budget.

    {
        int positional = 0;
        for (int i = 1; i < argc; i++) {
            const std::string a = argv[i];
            if (a == "--token-ids" && i + 1 < argc) {
                token_ids_path = argv[++i];
            } else if (a == "--voice" && i + 1 < argc) {
                voice_path = argv[++i];
            } else if (a == "--out" && i + 1 < argc) {
                out_path = argv[++i];
            } else if (a == "--stream") {
                stream_mode = true;
            } else if (a == "--serve-stream") {
                serve_stream_mode = true;
            } else if (a == "--espeak-voice" && i + 1 < argc) {
                espeak_voice = argv[++i];
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
    {
        bool mb = false;
        for (int i = 1; i < argc; i++) if (std::string(argv[i]) == "--microbench") mb = true;
        if (mb) {
            OpenCLContext bctx;
            if (bctx.initialize()) { fprintf(stderr, "=== DEVICE PEAKS ===\n"); run_microbench(bctx); }
            return 0;
        }
    }

    OpenCLContext cl_ctx;
    if (!cl_ctx.initialize()) {
        NNOPT_ERROR("OpenCL init failed");
        return 1;
    }

        // Tokenizer — for this phoneme TTS model the deterministic eval path uses
    // pinned ids from assets/test_input_ids.bin (loaded below), so a tokenizer
    // load failure is non-fatal here. It is loaded only so the (P2) live-text
    // path has a vocab; the eos id is unused in the single-shot pipeline.
    Tokenizer tok;
    (void)tok.load("weights/tokenizer_vocab.bin");

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

        Model model(cl_ctx, weights);
    if (!model.initialize()) {
        NNOPT_ERROR("Model::initialize() failed — see prior NNOPT_ERROR for the layer that failed");
        return 1;
    }



    // ── On-device streaming (--stream / --serve-stream) ─────────────────
    // End-to-end in C++, no Python: text -> espeak-ng G2P -> chunk_text ->
    // per-chunk forward_graph synth -> raw int16 LE PCM flushed to STDOUT, so a
    // consumer plays chunk N while chunk N+1 synthesizes on the GPU.
    //
    //   --stream         one-shot: synthesize `prompt`, then exit.
    //   --serve-stream   persistent REPL: model + phonemizer load ONCE, then one
    //                    utterance per stdin line (this is what the app drives —
    //                    per-utterance cost is pure synthesis, no reload tax).
    //
    // Per-chunk framing (stdout carries ONLY raw int16 LE PCM @ 24kHz mono; all
    // diagnostics + the length prefix go to stderr, so `adb exec-out ... | ffplay
    // -f s16le -ar 24000 -ac 1 -nodisp -` plays it and the Kotlin reader can slurp
    // exactly n samples before the next line):
    //   stderr: KITTEN_PCM_BEGIN <n_samples> 24000   (flushed BEFORE the PCM)
    //   stdout: <n_samples> int16 LE samples          (flushed)
    //   ...one BEGIN+PCM pair per chunk...
    //   stderr: KITTEN_UTT_END                        (end of this utterance)
    if (stream_mode || serve_stream_mode) {
#ifdef NNOPT_TTS_STREAMING
        // Single fixed style/voice vector (KittenTTS is not length-indexed like
        // Kokoro's voice pack) — loaded once, reused for every chunk.
        auto style = load_float_bin(voice_path);
        if (style.empty()) {
            NNOPT_ERROR_FMT("voice/style vector missing: %s", voice_path.c_str());
            return 5;
        }
        nnopt_tts::Phonemizer ph;
        // espeak wants the PARENT of espeak-ng-data/; deploy pushes assets/* to
        // the run dir, so espeak-ng-data/ lives at assets/espeak-ng-data/.
        if (!ph.init(/*espeak_data_parent=*/"assets", "assets/phoneme_vocab.tsv", espeak_voice))
            return 5;

        // rng_uniform is provably dead (backbone drops it); passing an empty
        // rng_normal makes Generator auto-size deterministic seeded excitation
        // noise for THIS chunk's frame count. No oracle here — this is production
        // synthesis of arbitrary text, not the pinned reference utterance.
        const std::vector<float> no_rng;

        auto synth_chunks = [&](const std::string& text) -> int {
            const auto chunks = nnopt_tts::chunk_text(text);
            std::fprintf(stderr, "STREAM_CHUNKS %zu\n", chunks.size());
            std::fflush(stderr);
            for (size_t c = 0; c < chunks.size(); ++c) {
                std::vector<int32_t> ids = ph.phonemize(chunks[c]);
                if (ids.size() <= 2) {  // only bos/eos — nothing phonemized
                    std::fprintf(stderr, "STREAM_SKIP empty chunk %zu: \"%s\"\n",
                                 c, chunks[c].c_str());
                    continue;
                }
                std::vector<int16_t> chunk_pcm;
                const auto t0 = std::chrono::steady_clock::now();
                const int rc2 = model.forward_graph(ids, style, no_rng, no_rng, chunk_pcm);
                const double synth_s = std::chrono::duration<double>(
                    std::chrono::steady_clock::now() - t0).count();
                if (rc2 != 0) {
                    NNOPT_ERROR_FMT("forward_graph chunk %zu rc=%d", c, rc2);
                    return 7;
                }
                // Length prefix on stderr, THEN raw PCM on stdout.
                std::fprintf(stderr, "KITTEN_PCM_BEGIN %zu 24000\n", chunk_pcm.size());
                std::fflush(stderr);
                std::fwrite(chunk_pcm.data(), sizeof(int16_t), chunk_pcm.size(), stdout);
                std::fflush(stdout);

                const double audio_s = (double)chunk_pcm.size() / 24000.0;
                std::fprintf(stderr,
                             "STREAM_CHUNK %zu synth_s=%.3f samples=%zu audio_s=%.3f rtf=%.3f \"%s\"\n",
                             c, synth_s, chunk_pcm.size(), audio_s,
                             audio_s > 0 ? synth_s / audio_s : 0.0, chunks[c].c_str());
                std::fflush(stderr);
            }
            return 0;
        };

        std::fprintf(stderr, "STREAM_SAMPLE_RATE 24000\n");
        std::fflush(stderr);

        if (serve_stream_mode) {
            std::fprintf(stderr, "ready. send one utterance per line; blank line to quit.\n");
            std::fflush(stderr);
            std::string line;
            while (std::getline(std::cin, line)) {
                while (!line.empty() && (line.back() == '\r' || line.back() == '\n'))
                    line.pop_back();
                if (line.empty()) break;
                const int rc2 = synth_chunks(line);
                std::fprintf(stderr, "KITTEN_UTT_END\n");   // reader fires onComplete
                std::fflush(stderr);
                if (rc2 != 0) return rc2;
            }
        } else {
            const int rc2 = synth_chunks(prompt);
            std::fprintf(stderr, "KITTEN_UTT_END\n");
            std::fflush(stderr);
            if (rc2 != 0) return rc2;
        }
        std::fprintf(stderr, "NNOPT_EXIT_CLEAN exit_code=0\n");
        std::fflush(stderr);
        return 0;
#else
        NNOPT_ERROR("--stream/--serve-stream needs espeak-ng vendored at "
                    "src/third_party/espeak-ng/ and a rebuild (CMake auto-enables "
                    "NNOPT_TTS_STREAMING when present)");
        return 2;
#endif
    }

    // ── TTS single-shot pipeline ─────────────────────────────────────────
    // No autoregressive decode loop. We load the pinned input ids plus every
    // aux fixture reference/io_contract.json declares (style/speaker vectors,
    // RNG noise captures) and call Model::forward_graph(...) ONCE, then
    // write a WAV.
    //
    // The aux fixtures are captured by the reference run and SHIPPED inside
    // assets/ on the device. Loading them — never sampling RNG or inventing
    // conditioning vectors in C++ — is what makes per-layer cosine compare
    // against the reference meaningful. See tts.md axis D.
    // --token-ids / --voice override the pinned reference fixtures. The DEFAULTS
    // stay pinned so the oracle comparison remains reproducible; overriding them
    // is for generating new utterances, where no oracle exists.
    auto input_ids = load_int32_bin(token_ids_path.empty()
                                    ? std::string("assets/test_input_ids.bin")
                                    : token_ids_path);
    auto style = load_float_bin(voice_path);
    auto rng_decoder_generator_m_source_l_sin_gen_RandomUniformLike_output_0 = load_float_bin("assets/rng_decoder_generator_m_source_l_sin_gen_RandomUniformLike_output_0.bin");
    auto rng_decoder_generator_m_source_l_sin_gen_RandomNormalLike_output_0 = load_float_bin("assets/rng_decoder_generator_m_source_l_sin_gen_RandomNormalLike_output_0.bin");
    if (input_ids.empty() || style.empty() || rng_decoder_generator_m_source_l_sin_gen_RandomUniformLike_output_0.empty() || rng_decoder_generator_m_source_l_sin_gen_RandomNormalLike_output_0.empty()) {
        NNOPT_ERROR("required TTS fixture missing under assets/ "
                    "(did GenerateReference run? see tts.md)");
        return 5;
    }

    std::vector<int16_t> pcm;
    bench.mark_prefill_start();
    NNOPT_BENCH_FIRST_TOKEN();   // repurposed: marks first sample produced
    const int rc = model.forward_graph(input_ids, style, rng_decoder_generator_m_source_l_sin_gen_RandomUniformLike_output_0, rng_decoder_generator_m_source_l_sin_gen_RandomNormalLike_output_0, pcm);
    bench.mark_end();
    KernelProfiler::dump_summary();
    if (rc != 0) {
        NNOPT_ERROR_FMT("forward_graph returned %d", rc);
        return 7;
    }

    // WAV output — 16-bit signed mono. Sample rate is threaded in from
    // reference/io_contract.json (Phase 3E) so Kokoro/StyleTTS2 (24 kHz)
    // and other non-VITS TTS sources don't inherit the VITS 16 kHz default.
    const std::string out_wav = out_path.empty() ? std::string("output.wav") : out_path;
    if (!write_wav(out_wav, pcm.data(), (int)pcm.size(), 24000)) {
        NNOPT_ERROR_FMT("write_wav failed: %s", out_wav.c_str());
        return 8;
    }

    // Evaluate-parsed stderr contract.
    std::cerr << "TTS_OUTPUT_PCM_SAMPLES " << pcm.size() << std::endl;
    std::cerr << "TTS_OUTPUT_SAMPLE_RATE 24000" << std::endl;

    KernelProfiler::dump_summary();
    bench.print_summary((int)input_ids.size(), (int)pcm.size());
    return 0;


}
