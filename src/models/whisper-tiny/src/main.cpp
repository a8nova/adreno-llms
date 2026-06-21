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
#include <thread>    // streaming: non-blocking stdin reader thread
#include <mutex>     // streaming: guard the shared phrase buffer/queue
#include <atomic>    // streaming: eof flag
#include <deque>     // streaming: queue of completed phrases
#include <set>       // streaming: nnopt_text_loops 4-gram dedup
#include <cctype>    // streaming: nnopt_text_loops word normalization
#include <algorithm> // std::min — streaming partial token cap

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
//    energy VAD + sliding window. Emits `PARTIAL [t0-t1]: <text>` while a
//    phrase is in progress (re-transcribed every step) and `FINAL [t0-t1]:
//    <text>` when the VAD detects end-of-speech (silence hangover) or the 30s
//    Whisper window cap is hit. [t0-t1] = the phrase's position in the global
//    stdin stream, in seconds since the first sample — the host uses it to
//    fence off events from audio fed before a UI reset and to key phrases. Reuses the exact mel->encode->decode pipeline (model.forward) per
//    window; the only new model-side hook is the encoder-cache invalidator so a
//    changing window doesn't false-hit a stale encoder output.
//    NOTE: this duplicates the per-window decode core of the --audio path on
//    purpose — to keep the validated --audio-list benchmark path untouched. The
//    shared core belongs in the tool's asr scaffold; unify there, not by hand.
extern "C" void WhisperBackbone_invalidate_encoder_cache();

namespace {

// No-repeat n-gram blocking (HF `no_repeat_ngram_size`). Bans any token that would
// complete an n-gram already seen in `gen`, by setting its logit to -inf. This breaks
// the greedy repetition loops ("X. X. X. …") that the fast variable-length encode
// induces on short/live windows. Applied ONLY to PARTIALs — finals use the padded
// encode (which doesn't loop) and stay byte-exact with the benchmark.
static void ban_repeat_ngrams(std::vector<float>& logits,
                              const std::vector<int32_t>& gen, int n) {
    const int g = (int)gen.size();
    if (n < 2 || g < n) return;
    // Prefix = the last (n-1) generated tokens; ban whatever token followed any
    // earlier occurrence of that same prefix.
    for (int i = 0; i + n <= g; ++i) {
        bool match = true;
        for (int j = 0; j < n - 1; ++j) {
            if (gen[i + j] != gen[g - (n - 1) + j]) { match = false; break; }
        }
        if (match) {
            const int banned = gen[i + n - 1];
            if (banned >= 0 && banned < (int)logits.size()) logits[banned] = -INFINITY;
        }
    }
}

// Crude word-level loop detector: lowercase, strip punctuation, look for any
// repeated 4-gram. Normal speech doesn't repeat a 4-word sequence inside one
// ≤12s phrase; a decode that re-read padded silence repeats whole clauses
// verbatim. Used to trigger the 30s-pad redo on FINALs (see emit).
static bool nnopt_text_loops(const std::string& text) {
    std::vector<std::string> w;
    std::string cur;
    for (char c : text) {
        if (std::isalnum((unsigned char)c)) cur += (char)std::tolower((unsigned char)c);
        else if (!cur.empty()) { w.push_back(cur); cur.clear(); }
    }
    if (!cur.empty()) w.push_back(cur);
    if (w.size() < 8) return false;
    std::set<std::string> seen;
    for (size_t i = 0; i + 4 <= w.size(); ++i) {
        const std::string g = w[i] + " " + w[i+1] + " " + w[i+2] + " " + w[i+3];
        if (!seen.insert(g).second) return true;
    }
    return false;
}

// Transcribe one window of 16kHz mono float audio -> text.
// pad_30s: force the full fixed 30s padded encode (Whisper's training
// distribution — slowest but never loops). Used for the FINAL redo path.
std::string nnopt_transcribe_window(OpenCLContext& cl_ctx, Model& model, Tokenizer& tok,
                                    bool tokenizer_ok, Sampler& sampler,
                                    const SamplerConfig& scfg, int max_new_tokens,
                                    const std::vector<float>& wav,
                                    const std::vector<float>& mel_filters,
                                    bool full_quality, bool pad_30s = false) {
    if (wav.empty()) return std::string();
    const auto _t0 = std::chrono::high_resolution_clock::now();
    const double _win_s = (double)wav.size() / 16000.0;
    // Both PARTIAL and FINAL use the variable-length encode (encode only the actual
    // window, not a padded 30s) — the key to real-time. The difference is the context
    // floor: FINALs pad up to FINAL_FLOOR_FRAMES (~12s) so the committed decode has
    // enough zero-padded context that tiny doesn't loop; PARTIALs use no floor for the
    // cheapest possible live preview (and lean on no-repeat-ngram below). Padding only
    // to a ~12s floor — vs the old fixed 30s — removes most of the per-sentence FINAL
    // freeze while keeping greedy quality. A long FINAL (>12s) still encodes its full
    // length up to the 30s cap. (Batch/eval still uses the fixed 30s whisper_log_mel.)
    constexpr int FINAL_FLOOR_FRAMES = 1200;  // 12 s of context for committed phrases
    constexpr int WHISPER_MAX_FRAMES = 3000;  // 30 s @ hop 160 — encoder hard cap
    std::vector<float> mel =
        pad_30s      ? whisper_log_mel_n(wav, mel_filters, /*max_frames=*/WHISPER_MAX_FRAMES,
                                         /*min_frames=*/WHISPER_MAX_FRAMES)
        : full_quality ? whisper_log_mel_n(wav, mel_filters, /*max_frames=*/WHISPER_MAX_FRAMES,
                                           /*min_frames=*/FINAL_FLOOR_FRAMES)
                       : whisper_log_mel_n(wav, mel_filters);
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

    // Partials (variable encode) are loop-prone on short live windows → block repeated
    // 3-grams so the live preview self-corrects instead of spamming "X. X. X.". Finals
    // deliberately do NOT get the ban: the ~12s floor still loops occasionally (the
    // stream-replay harness caught an 8s phrase repeating verbatim 3x), and banning
    // n-grams only mutates the loop into near-duplicates ("Burtk"/"Fosters'") that
    // can't be detected. Instead finals keep the clean greedy path and the emit()
    // loop below detects verbatim repeats and redoes the decode with the full 30s
    // pad — Whisper's training distribution, which doesn't loop.
    const bool suppress_repeats = !full_quality;
    for (int step = 0; step < max_new_tokens; ++step) {
        std::vector<int32_t> gen_so_far(prompt_ids.begin() + prompt_len, prompt_ids.end());
        if (suppress_repeats) ban_repeat_ngrams(logits, gen_so_far, 3);
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

    // Per-window timing: proc time vs the audio it covered. rtf = proc/win. With the
    // variable-length encode (full_quality=false) the encoder runs on the ACTUAL
    // window, so a short window is cheap and proc scales with the window — partials
    // are bounded to a rolling slice (see nnopt_run_stream) to keep this near-constant.
    // Only full_quality=true pays the fixed 30s padded encode. Surfaced on stderr.
    const double _proc_s = std::chrono::duration<double>(
        std::chrono::high_resolution_clock::now() - _t0).count();
    const int _ntok = (int)(prompt_ids.size() - prompt_len);
    fprintf(stderr, "STREAM_TIMING win=%.2f proc=%.2f rtf=%.2f ntok=%d\n",
            _win_s, _proc_s, (_win_s > 0 ? _proc_s / _win_s : 0.0), _ntok);
    fflush(stderr);
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

    const int    SR               = 16000;
    const int    frame            = SR * 30 / 1000;            // 30ms VAD frame = 480 samples
    const size_t step_samples     = (size_t)SR * step_ms / 1000;
    const int    hangover_frames  = (hangover_ms > 0 ? hangover_ms : 1) / 30;
    const int    trigger_frames   = 3;                         // consecutive voiced frames to start a phrase
    const size_t min_phrase_samps = (size_t)(0.30f * SR);      // ignore sub-300ms blips
    const float  max_phrase_s     = 28.0f;                     // hard force-commit before Whisper's 30s window
    const int    cap              = max_new_tokens > 16 ? max_new_tokens : 96;  // per-phrase token cap
    // Long-speech segmentation, two-stage so a continuous talker never breaks:
    //   (a) soft_seg_s: once a phrase passes this, the FIRST short clause/breath pause
    //       (soft_silence_frames) commits it — the natural, clean cut point.
    //   (b) seg_force_s: if NO such pause appears (someone reciting without breaks), force
    //       a commit purely on duration so the open phrase can't outgrow the partial
    //       window. Without this the phrase grows toward the 28s hard cap, the window
    //       reverts to a rolling tail, and earlier words get deleted again (observed: a
    //       19s pause-less phrase). A forced cut may land mid-word, but tiny stitches the
    //       next segment fine and that's far better than losing 10s+ of visible text.
    // partial_window_s MUST be >= seg_force_s so the open phrase is always fully covered.
    const float  soft_seg_s          = 8.0f;
    const float  seg_force_s         = 12.0f;                  // commit by here even with no pause
    const int    soft_silence_frames = 420 / 30;               // ~0.42s clause-level pause
    // PARTIAL window. CRITICAL: this MUST be >= soft_seg_s. The on-screen transcript is
    // (committed finals) + (live partial); the partial re-transcribes only the last
    // partial_window_s of voiced audio. If that window is SHORTER than the open phrase,
    // words spoken earlier in the phrase are neither committed yet NOR in the window, so
    // they get overwritten/DELETED as the window slides — the "it eats my transcript"
    // bug. By keeping the window >= the soft-segment length, the partial always spans the
    // ENTIRE open phrase: it becomes a growing prefix (text accumulates / refines, never
    // deletes) and the phrase locks in every ~soft_seg_s. The cost is that a late partial
    // re-decodes the whole phrase from scratch (no decode-KV reuse across partials), so
    // partials refresh a bit slower — an acceptable trade vs. losing text. Measured decode
    // is ~0.05-0.06s/token (much cheaper than the old 0.24s estimate), so a full ~13s
    // window is only ~2-3s of proc. seg_force_s guarantees the open phrase never exceeds
    // this window, so partials never revert to a deleting rolling tail in practice.
    const float  partial_window_s     = 13.0f;  // >= seg_force_s (12s) + margin
    const size_t partial_window_samps = (size_t)(partial_window_s * SR);

    // Producer/consumer so transcription (slow) never blocks mic intake (real-time).
    // Reader thread: drains stdin continuously, runs VAD, appends to `cur`; on
    // end-of-speech it moves `cur` into the `finals` queue and starts fresh.
    // Consumer (this thread): transcribes completed phrases as FINAL, and the
    // latest `cur` snapshot as PARTIAL — always the most recent audio, so it
    // never falls behind (stale partials are simply skipped, not queued).
    std::mutex mtx;
    std::vector<float> cur;                  // in-progress phrase (guarded)
    size_t cur_voiced = 0;                   // samples in `cur` up to & incl. the last VOICED frame (guarded).
                                             // Partials anchor here, NOT cur.size(): trailing silence (the
                                             // hangover gap, background noise) must not enter the rolling
                                             // window or Whisper loops/repeats on near-silent context, and
                                             // partials must stop firing the instant speech stops.
    double cur_t0 = 0.0;                     // stream-time (s) of cur[0] — phrase start offset (guarded).
    // Every PARTIAL/FINAL is tagged with its phrase's [start-end] position in the
    // GLOBAL audio stream (seconds since first stdin sample). The host counts the
    // samples it has fed, so it can (a) drop events from audio fed before a UI
    // reset — transcription latency means a FINAL can land seconds after the user
    // started a new recording session (the "ghost text" bug) — and (b) key phrases
    // by start offset, making replace-vs-append decisions structural instead of
    // text-heuristic.
    struct Phrase { std::vector<float> pcm; double t0, t1; };
    std::deque<Phrase> finals;               // completed phrases awaiting transcription (guarded)
    std::atomic<bool> eof{false};

    std::thread reader([&]() {
        std::vector<float> rd(4096), framebuf;
        bool in_speech = false;
        int speech_run = 0, silence_run = 0;
        size_t stream_off = 0;               // total samples consumed from stdin (frame-granular)
        // Finite sentinel that never occurs in normalized [-1,1] PCM. The host writes it to flush the
        // in-progress phrase + reset streaming state WITHOUT closing stdin, so the model stays loaded
        // (warm) between dictation sessions. 1e30f has one exact IEEE-754 single representation, so the
        // host's Float(1e30f) bit-matches this — a plain value compare works (it's finite, not NaN).
        const float NNOPT_STREAM_RESET = 1.0e30f;
        // Process every COMPLETE frame currently buffered (VAD → phrase accumulation → FINAL on a pause).
        auto process_framebuf = [&]() {
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
                    std::lock_guard<std::mutex> lk(mtx);
                    if (cur.empty()) cur_t0 = (double)stream_off / (double)SR;  // phrase starts here
                    cur.insert(cur.end(), framebuf.begin() + off, framebuf.begin() + off + frame);
                    if (voiced) cur_voiced = cur.size();   // advance voiced extent ONLY on voiced frames
                    const float phrase_s = (float)cur.size() / (float)SR;
                    const bool hang_end  = silence_run >= hangover_frames;                            // real end-of-speech pause
                    const bool soft_end  = phrase_s >= soft_seg_s && silence_run >= soft_silence_frames; // long speech: clause pause
                    const bool force_end = phrase_s >= seg_force_s;                                   // long speech: no pause came → cut on duration
                    const bool hard_end  = phrase_s >= max_phrase_s;                                  // hard backstop
                    if (hang_end || soft_end || force_end || hard_end) {
                        if (cur.size() >= min_phrase_samps) {
                            // Surface long-speech segmentation so the host can SEE that a
                            // continuous talker is being chunked (not stalling/breaking).
                            if (!hang_end && soft_end)
                                fprintf(stderr, "STREAM_SEG soft phrase=%.1fs (long speech: clause cut)\n", phrase_s);
                            else if (!hang_end && !soft_end && (force_end || hard_end))
                                fprintf(stderr, "STREAM_SEG force phrase=%.1fs (long speech: duration cut)\n", phrase_s);
                            fflush(stderr);
                            const double t1 = cur_t0 + (double)cur.size() / (double)SR;
                            finals.push_back(Phrase{std::move(cur), cur_t0, t1});
                        }
                        cur.clear(); cur_voiced = 0;
                        in_speech = false; speech_run = 0; silence_run = 0;
                    }
                }
                off += frame;
                stream_off += (size_t)frame;
            }
            framebuf.erase(framebuf.begin(), framebuf.begin() + off);
        };
        // Flush the in-progress phrase as a FINAL and clear accumulation state, so the next dictation
        // session doesn't append to the last one's phrase. The model stays loaded; stream_off keeps
        // counting (the host ignores timestamps anyway).
        auto reset_stream = [&]() {
            std::lock_guard<std::mutex> lk(mtx);
            if (cur.size() >= min_phrase_samps) {
                const double t1 = cur_t0 + (double)cur.size() / (double)SR;
                finals.push_back(Phrase{std::move(cur), cur_t0, t1});
            }
            cur.clear(); cur_voiced = 0;
            framebuf.clear(); in_speech = false; speech_run = 0; silence_run = 0;
        };
        while (true) {
            const size_t got = std::fread(rd.data(), sizeof(float), rd.size(), stdin);
            if (got == 0) { eof.store(true); break; }
            size_t seg = 0;
            for (size_t k = 0; k < got; ++k) {
                if (rd[k] == NNOPT_STREAM_RESET) {
                    framebuf.insert(framebuf.end(), rd.begin() + seg, rd.begin() + k);
                    process_framebuf();   // drain audio fed before the marker
                    reset_stream();       // flush phrase + clear state (model stays warm)
                    seg = k + 1;          // skip the marker sample
                }
            }
            framebuf.insert(framebuf.end(), rd.begin() + seg, rd.begin() + got);
            process_framebuf();
        }
    });

    fprintf(stderr, "STREAM: ready — feed 16kHz mono float32 PCM on stdin "
                    "(vad_threshold=%.4f step=%dms hangover=%dms cap=%d tok)\n",
            vad_threshold, step_ms, hangover_ms, cap);
    fflush(stderr);

    auto emit = [&](const char* tag, const std::vector<float>& pcm, int tok_cap, bool full_quality,
                    double t0, double t1) {
        // PARTIALs (full_quality=false) use the cheapest variable-length encode for a
        // snappy live preview, guarded by no-repeat-ngram so short-context loops can't
        // run away. FINALs (full_quality=true) use the SAME variable encode but padded
        // up to a ~12s context floor (see nnopt_transcribe_window) and run greedy — enough
        // context for stable, correct committed text without paying the old fixed 30s
        // encode on every sentence boundary. Long finals encode their full length (≤30s).
        // [t0-t1] = the phrase's position in the global stdin stream (seconds) — see
        // the Phrase struct comment for why the host needs this.
        std::string txt = nnopt_transcribe_window(cl_ctx, model, tok, tokenizer_ok,
                                                  sampler, scfg, tok_cap, pcm, mel_filters,
                                                  full_quality);
        // FINAL loop guard: the ~12s-floor encode is out-of-distribution for
        // Whisper (trained on fixed 30s windows) and occasionally re-reads the
        // padded silence as a verbatim repeat of the phrase. Committed text
        // must be clean, so detect the repeat and redo with the full 30s pad —
        // slower (~RTF 1 on the phrase) but in-distribution and loop-free.
        // Clean decodes (the overwhelmingly common case) never pay this.
        if (full_quality && nnopt_text_loops(txt)) {
            fprintf(stderr, "STREAM_REDO loop detected in FINAL — re-decoding with 30s pad\n");
            fflush(stderr);
            txt = nnopt_transcribe_window(cl_ctx, model, tok, tokenizer_ok,
                                          sampler, scfg, tok_cap, pcm, mel_filters,
                                          /*full_quality=*/true, /*pad_30s=*/true);
        }
        char hdr[64];
        snprintf(hdr, sizeof(hdr), "%s [%.2f-%.2f]: ", tag, t0, t1);
        std::cout << hdr << txt << std::endl;
    };

    size_t last_partial_n = 0;
    while (true) {
        // 1. A completed phrase takes priority — commit it as FINAL. Finals encode the
        //    WHOLE phrase padded only to a ~12s floor (not a fixed 30s), so a short
        //    sentence commits in a fraction of the old cost while staying stable.
        Phrase fin;
        bool have_final = false;
        {
            std::lock_guard<std::mutex> lk(mtx);
            if (!finals.empty()) { fin = std::move(finals.front()); finals.pop_front(); have_final = true; }
        }
        if (have_final) {
            if (fin.pcm.size() >= min_phrase_samps)
                emit("FINAL", fin.pcm, cap, /*full_quality=*/true, fin.t0, fin.t1);
            last_partial_n = 0;
            continue;
        }
        // 2. Stream closed with nothing queued → flush the in-progress tail (FULL
        //    phrase, not the rolling window) as a closing FINAL.
        if (eof.load()) {
            std::vector<float> tail;
            double tail_t0 = 0.0;
            { std::lock_guard<std::mutex> lk(mtx); tail.swap(cur); tail_t0 = cur_t0; }
            if (tail.size() >= min_phrase_samps)
                emit("FINAL", tail, cap, /*full_quality=*/true,
                     tail_t0, tail_t0 + (double)tail.size() / (double)SR);
            break;
        }
        // 3. PARTIAL: re-transcribe the most recent ~partial_window_s of VOICED audio.
        //    The slice ends at cur_voiced (the last voiced sample), so trailing silence
        //    — the hangover gap or background noise — never enters the window (which
        //    would make Whisper loop/repeat on near-silent context). Cadence is tracked
        //    on cur_voiced too, so partials STOP the instant speech stops and resume
        //    only when real speech grows the phrase.
        std::vector<float> snap;
        size_t voiced_len = 0;
        double snap_t0 = 0.0, snap_t1 = 0.0;
        {
            std::lock_guard<std::mutex> lk(mtx);
            voiced_len = cur_voiced;
            const size_t from = voiced_len > partial_window_samps ? voiced_len - partial_window_samps : 0;
            snap.assign(cur.begin() + from, cur.begin() + voiced_len);
            snap_t0 = cur_t0 + (double)from / (double)SR;
            snap_t1 = cur_t0 + (double)voiced_len / (double)SR;
        }
        if (voiced_len >= last_partial_n + step_samples && snap.size() >= min_phrase_samps) {
            // Fast variable encode → live preview; no-repeat-ngram (inside the transcribe)
            // keeps it from looping. The clean text lands on the FINAL. The window now
            // spans the whole open phrase (~8-9s ≈ up to ~30 tokens of speech), so the cap
            // must be high enough to decode all of it — otherwise the partial would itself
            // truncate the phrase tail it's meant to show.
            emit("PARTIAL", snap, std::min(cap, 56), /*full_quality=*/false, snap_t0, snap_t1);
            last_partial_n = voiced_len;   // cadence on voiced extent — no partials during silence
        } else {
            std::this_thread::sleep_for(std::chrono::milliseconds(25));
        }
    }

    reader.join();
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
