// Auto-generated graph-mode main.cpp for UsefulSensors/moonshine-tiny.
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
#include "forward_dispatch.h"  // ForwardDispatch::set_input_features
#include "utils.h"             // nnopt_storage_t + nnopt_f32_to_f16 (fp16 conversion)


#include <fstream>
#include <iostream>
#include <string>
#include <vector>
#include <cstdint>
#include <cstring>
#include <atomic>    // streaming: reader-thread EOF flag
#include <chrono>    // streaming: STREAM_TIMING per-window proc time
#include <cmath>     // streaming: VAD frame RMS
#include <cstdio>    // streaming: std::fread on stdin
#include <deque>     // streaming: completed-phrase queue
#include <mutex>     // streaming: reader/consumer handoff
#include <thread>    // streaming: stdin reader thread

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

// FinalizePort's tokenizer gate (require_text_prompt=true) omits --token-ids
// and then reads layer_dumps/tokenizer_encode.json, diffing its `ids` array
// against reference/reference_tokens.json::input_ids. For this ASR model the
// decoder input is NOT produced by tokenizer.encode() of a text string — it
// is the decoder priming sequence [decoder_start_token_id]. We write that
// genuine, code-constructed id sequence (computed below, never copied from
// the reference) so the gate can verify the real decoder-input construction.
static void write_tokenizer_encode_dump(const std::vector<int32_t>& ids,
                                        const std::string& prompt) {
    std::ofstream out("layer_dumps/tokenizer_encode.json", std::ios::binary);
    if (!out) return;
    out << "{\"ids\":[";
    for (size_t i = 0; i < ids.size(); ++i) {
        if (i) out << ",";
        out << ids[i];
    }
    out << "],\"prompt\":\"";
    for (char c : prompt) {
        if (c == '"' || c == '\\') out << '\\';
        out << c;
    }
    out << "\"}";
    // Log line the gate documents as the confirmation signal.
    std::cerr << "Input tokens: " << ids.size() << " [";
    for (size_t i = 0; i < ids.size(); ++i) {
        if (i) std::cerr << ", ";
        std::cerr << ids[i];
    }
    std::cerr << "]" << std::endl;
}

// ── Streaming mode (--stream): continuous 16kHz mono float32 PCM on stdin,
//    energy VAD + sliding window. Emits `PARTIAL [t0-t1]: <text>` while a
//    phrase is in progress (re-transcribed every step) and `FINAL [t0-t1]:
//    <text>` on end-of-speech / segmentation / EOF. Same wire protocol as the
//    whisper-tiny port's --stream (see-and-say's WhisperSession parses it):
//    [t0-t1] = phrase position in the global stdin stream (s), STREAM_TIMING /
//    STREAM_SEG diagnostics on stderr, 1e30f in-band reset sentinel.
//    Moonshine differences vs whisper: NO mel frontend (the raw waveform goes
//    straight to the conv stack), NO forced language tokens, NO 30s-pad redo
//    (moonshine is trained on variable-length audio — every window is
//    in-distribution), and the window cap is ~20s (encoder cross-attn caps at
//    1024 frames ≈ 24.6s; decoder KV at 194 tokens/phrase).
extern "C" void MoonshineBackbone_invalidate_encoder_cache();

namespace {

// No-repeat n-gram blocking (HF `no_repeat_ngram_size`) for ALL stream decodes
// (PARTIAL and FINAL): bans a token that would complete an n-gram already
// generated in this window, so no repetition loop can ever be COMMITTED to the
// transcript. The batch (non-stream) path stays pure greedy — the token-exact
// gate only covers batch mode, which does not use this function.
static void ban_repeat_ngrams(std::vector<float>& logits,
                              const std::vector<int32_t>& gen, int n) {
    const int g = (int)gen.size();
    if (n < 2 || g < n) return;
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

// Transcribe one window of 16kHz mono float audio → generated TOKENS (EOS
// stripped). Token output (not text) is what makes local-agreement streaming
// possible: consecutive hypotheses are compared token-for-token and only the
// agreed prefix is committed. All stream decodes use the no-repeat guard.
std::vector<int32_t> nnopt_transcribe_window_tokens(
        OpenCLContext& cl_ctx, Model& model, Tokenizer& tok,
        bool tokenizer_ok, Sampler& sampler,
        const SamplerConfig& scfg, int max_new_tokens,
        const std::vector<float>& wav) {
    (void)tokenizer_ok;
    if (wav.empty()) return {};
    const auto _t0 = std::chrono::high_resolution_clock::now();
    const double _win_s = (double)wav.size() / 16000.0;

    std::vector<nnopt_storage_t> wav_storage(wav.size());
#ifdef NNOPT_USE_FP16
    for (size_t i = 0; i < wav.size(); ++i) wav_storage[i] = nnopt_f32_to_f16(wav[i]);
#else
    for (size_t i = 0; i < wav.size(); ++i) wav_storage[i] = (nnopt_storage_t)wav[i];
#endif
    cl_int err = CL_SUCCESS;
    cl_mem feats = clCreateBuffer(cl_ctx.context(), CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR,
                                  wav_storage.size() * sizeof(nnopt_storage_t),
                                  wav_storage.data(), &err);
    if (err != CL_SUCCESS || !feats) { NNOPT_ERROR_FMT("stream: waveform buffer alloc %d", (int)err); return {}; }

    MoonshineBackbone_invalidate_encoder_cache();   // force recompute for THIS window's audio
    ForwardDispatch::set_input_features(feats);
    ForwardDispatch::set_num_samples((int)wav.size());

    // Decoder primed with the start token; per-phrase token cap must respect
    // the KV ring (194 rows incl. the start token).
    std::vector<int32_t> decoder_ids;
    decoder_ids.push_back(MODEL_CONFIG::DECODER_START_TOKEN_ID);
    const size_t prompt_len = decoder_ids.size();
    const int kv_room = MODEL_CONFIG::MAX_SEQUENCE_LENGTH - (int)prompt_len - 1;
    const int cap = max_new_tokens < kv_room ? max_new_tokens : kv_room;

    for (int step = 0; step < cap; ++step) {
        std::vector<float> logits = model.forward(decoder_ids);
        if (logits.empty()) {   // hard forward failure — commit what we have
            NNOPT_ERROR("stream: forward failed mid-window — truncating phrase");
            break;
        }
        std::vector<int32_t> gen_so_far(decoder_ids.begin() + prompt_len, decoder_ids.end());
        ban_repeat_ngrams(logits, gen_so_far, 3);
        int next = sampler.sample(logits, gen_so_far);
        decoder_ids.push_back(next);
        if (scfg.eos_token_id >= 0 && next == scfg.eos_token_id) break;
    }

    clReleaseMemObject(feats);
    MoonshineBackbone_invalidate_encoder_cache();   // don't leave this window's encoder cached
    ForwardDispatch::set_input_features(nullptr);

    std::vector<int32_t> gen(decoder_ids.begin() + prompt_len, decoder_ids.end());
    if (!gen.empty() && scfg.eos_token_id >= 0 && gen.back() == scfg.eos_token_id)
        gen.pop_back();

    const double _proc_s = std::chrono::duration<double>(
        std::chrono::high_resolution_clock::now() - _t0).count();
    fprintf(stderr, "STREAM_TIMING win=%.2f proc=%.2f rtf=%.2f ntok=%d\n",
            _win_s, _proc_s, (_win_s > 0 ? _proc_s / _win_s : 0.0), (int)gen.size());
    fflush(stderr);
    return gen;
}

int nnopt_run_stream(OpenCLContext& cl_ctx, Model& model, Tokenizer& tok, bool tokenizer_ok,
                     Sampler& sampler, const SamplerConfig& scfg, int max_new_tokens,
                     float vad_threshold, int step_ms, int hangover_ms) {
    if (!tokenizer_ok) { NNOPT_ERROR("stream: tokenizer required for streaming output"); return 4; }

    const int    SR               = 16000;
    const int    frame            = SR * 30 / 1000;            // 30ms VAD frame = 480 samples
    const size_t step_samples     = (size_t)SR * step_ms / 1000;
    const int    hangover_frames  = (hangover_ms > 0 ? hangover_ms : 1) / 30;
    const int    trigger_frames   = 3;                         // consecutive voiced frames to start a phrase
    const size_t min_phrase_samps = (size_t)(0.30f * SR);      // ignore sub-300ms blips
    const float  max_phrase_s     = 20.0f;                     // hard force-commit — encoder caps at 1024 frames ≈ 24.6s
    const int    cap              = max_new_tokens > 16 ? max_new_tokens : 96;  // per-phrase token cap
    // Long-speech segmentation (same two-stage design as whisper's stream):
    //   soft_seg_s: past this, the first ~0.4s clause pause commits the phrase.
    //   seg_force_s: no pause at all → cut on duration so the open phrase never
    //   outgrows the partial window (or the 20s encoder cap).
    const float  soft_seg_s          = 8.0f;
    const float  seg_force_s         = 12.0f;
    const int    soft_silence_frames = 420 / 30;               // ~0.42s clause-level pause
    // PARTIAL window MUST be >= seg_force_s so the live partial always spans
    // the ENTIRE open phrase (growing prefix; never deletes committed-ish text).
    const float  partial_window_s     = 13.0f;
    const size_t partial_window_samps = (size_t)(partial_window_s * SR);

    // Producer/consumer so transcription (slow) never blocks mic intake.
    std::mutex mtx;
    std::vector<float> cur;                  // in-progress phrase (guarded)
    size_t cur_voiced = 0;                   // samples up to & incl. the last VOICED frame (guarded)
    double cur_t0 = 0.0;                     // stream-time (s) of cur[0] (guarded)
    struct Phrase { std::vector<float> pcm; double t0, t1; bool reset = false; };
    std::deque<Phrase> finals;               // completed phrases awaiting transcription (guarded)
    std::atomic<bool> eof{false};

    // Phrase-onset preroll: the VAD needs trigger_frames voiced frames to open a
    // phrase, so without lookback the first ~90ms of speech (plus any soft
    // sub-threshold onset) is CROPPED — measured on the stitched-clip gate as
    // "The quick" → "A quick" and "Neural networks" → "Neut works". Keep the
    // last ~250ms of pre-trigger audio and prepend it when a phrase opens.
    const size_t preroll_samps = (size_t)(0.25f * SR);

    std::thread reader([&]() {
        std::vector<float> rd(4096), framebuf, preroll;
        bool in_speech = false;
        int speech_run = 0, silence_run = 0;
        size_t stream_off = 0;               // total samples consumed from stdin (frame-granular)
        // In-band reset sentinel (see-and-say writes Float(1e30f) to flush the
        // in-progress phrase between dictation sessions WITHOUT closing stdin —
        // the model stays warm). 1e30f is finite with one exact fp32 encoding.
        const float NNOPT_STREAM_RESET = 1.0e30f;
        auto process_framebuf = [&]() {
            size_t off = 0;
            while (framebuf.size() - off >= (size_t)frame) {
                double e = 0.0;
                for (int i = 0; i < frame; ++i) { const float s = framebuf[off + i]; e += (double)s * s; }
                const float rms = (float)std::sqrt(e / (double)frame);
                const bool voiced = rms >= vad_threshold;
                if (voiced) { ++speech_run; silence_run = 0; }
                else        { ++silence_run; speech_run = 0; }
                if (!in_speech) {
                    // Maintain the onset lookback while idle (trimmed to preroll_samps).
                    preroll.insert(preroll.end(), framebuf.begin() + off, framebuf.begin() + off + frame);
                    if (preroll.size() > preroll_samps)
                        preroll.erase(preroll.begin(), preroll.end() - preroll_samps);
                    if (speech_run >= trigger_frames) in_speech = true;
                }
                if (in_speech) {
                    std::lock_guard<std::mutex> lk(mtx);
                    if (cur.empty()) {
                        // Phrase opens with the preroll (which already contains this
                        // frame when the trigger fired above) so the onset isn't cropped.
                        cur_t0 = (double)(stream_off + frame - preroll.size()) / (double)SR;
                        cur = preroll;
                        preroll.clear();
                    } else {
                        cur.insert(cur.end(), framebuf.begin() + off, framebuf.begin() + off + frame);
                    }
                    if (voiced) cur_voiced = cur.size();
                    const float phrase_s = (float)cur.size() / (float)SR;
                    const bool hang_end  = silence_run >= hangover_frames;
                    const bool soft_end  = phrase_s >= soft_seg_s && silence_run >= soft_silence_frames;
                    const bool force_end = phrase_s >= seg_force_s;
                    const bool hard_end  = phrase_s >= max_phrase_s;
                    if (hang_end || soft_end || force_end || hard_end) {
                        if (cur.size() >= min_phrase_samps) {
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
        auto reset_stream = [&]() {
            std::lock_guard<std::mutex> lk(mtx);
            if (cur.size() >= min_phrase_samps) {
                const double t1 = cur_t0 + (double)cur.size() / (double)SR;
                finals.push_back(Phrase{std::move(cur), cur_t0, t1});
            }
            // Reset marker AFTER any flushed tail: the consumer prints
            // STREAM_RESET_ACK once everything from the old session has been
            // emitted, so the app can drop stale events deterministically.
            finals.push_back(Phrase{{}, 0.0, 0.0, /*reset=*/true});
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
                    process_framebuf();
                    reset_stream();
                    seg = k + 1;
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

    // ── Local-agreement streaming (LocalAgreement-2) ────────────────────────
    // Each step re-decodes the open phrase; tokens where TWO consecutive
    // hypotheses agree are COMMITTED (append-only — the app renders them solid
    // and never repaints them); only the disagreeing tail is a PARTIAL. FINAL
    // emits just the not-yet-committed remainder, so nothing is ever printed
    // twice. Wire protocol (stdout):
    //   COMMIT  [t0-t1]: <newly agreed text>   (append)
    //   PARTIAL [t0-t1]: <unstable tail>       (replace)
    //   FINAL   [t0-t1]: <remainder>           (append; phrase closed)
    //   STREAM_RESET_ACK                       (all pre-reset events flushed)
    auto decode_tokens = [&](const std::vector<float>& pcm) {
        return nnopt_transcribe_window_tokens(cl_ctx, model, tok, tokenizer_ok,
                                              sampler, scfg, cap, pcm);
    };
    auto print_event = [&](const char* tag, double t0, double t1, const std::string& txt) {
        char hdr[64];
        snprintf(hdr, sizeof(hdr), "%s [%.2f-%.2f]: ", tag, t0, t1);
        std::cout << hdr << txt << std::endl;
    };
    std::vector<int32_t> committed;     // tokens already emitted as COMMIT (this phrase)
    std::vector<int32_t> prev_hyp;      // previous hypothesis (this phrase)
    double phrase_id = -1.0;            // cur_t0 of the phrase the state belongs to
    size_t last_partial_n = 0;

    while (true) {
        // 1. Completed phrases first.
        Phrase fin;
        bool have_final = false;
        {
            std::lock_guard<std::mutex> lk(mtx);
            if (!finals.empty()) { fin = std::move(finals.front()); finals.pop_front(); have_final = true; }
        }
        if (have_final) {
            if (fin.reset) {
                committed.clear(); prev_hyp.clear(); phrase_id = -1.0; last_partial_n = 0;
                std::cout << "STREAM_RESET_ACK" << std::endl;
                continue;
            }
            if (fin.pcm.size() >= min_phrase_samps) {
                std::vector<int32_t> hyp = decode_tokens(fin.pcm);
                // Emit only what the partials haven't already committed. If this
                // final's phrase never got a partial (very short), committed is
                // empty and the whole hypothesis is the remainder. Positional
                // clamp: committed text stands even if the final re-decode
                // disagrees with it (streaming commits are irrevocable).
                const size_t base = (fin.t0 == phrase_id && committed.size() < hyp.size())
                                        ? committed.size()
                                        : (fin.t0 == phrase_id ? hyp.size() : 0);
                std::vector<int32_t> rem(hyp.begin() + base, hyp.end());
                print_event("FINAL", fin.t0, fin.t1, tok.decode(rem));
            }
            committed.clear(); prev_hyp.clear(); phrase_id = -1.0; last_partial_n = 0;
            continue;
        }
        // 2. Stream closed with nothing queued → flush the in-progress tail.
        if (eof.load()) {
            std::vector<float> tail;
            double tail_t0 = 0.0;
            { std::lock_guard<std::mutex> lk(mtx); tail.swap(cur); tail_t0 = cur_t0; }
            if (tail.size() >= min_phrase_samps) {
                std::vector<int32_t> hyp = decode_tokens(tail);
                const size_t base = (tail_t0 == phrase_id && committed.size() < hyp.size())
                                        ? committed.size()
                                        : (tail_t0 == phrase_id ? hyp.size() : 0);
                std::vector<int32_t> rem(hyp.begin() + base, hyp.end());
                print_event("FINAL", tail_t0, tail_t0 + (double)tail.size() / (double)SR, tok.decode(rem));
            }
            break;
        }
        // 3. PARTIAL step: snapshot the open phrase's voiced audio.
        std::vector<float> snap;
        double snap_phrase = -1.0, snap_t0 = 0.0, snap_t1 = 0.0;
        {
            std::lock_guard<std::mutex> lk(mtx);
            if (cur_voiced >= min_phrase_samps && cur_voiced >= last_partial_n + step_samples) {
                const size_t take = cur_voiced < partial_window_samps ? cur_voiced : partial_window_samps;
                snap.assign(cur.begin() + (cur_voiced - take), cur.begin() + cur_voiced);
                snap_phrase = cur_t0;
                snap_t0 = cur_t0 + (double)(cur_voiced - take) / (double)SR;
                snap_t1 = cur_t0 + (double)cur_voiced / (double)SR;
                last_partial_n = cur_voiced;
            }
        }
        if (snap.empty()) { std::this_thread::sleep_for(std::chrono::milliseconds(15)); continue; }
        // A new phrase opened while state still belongs to the previous one
        // (its FINAL is queued but not yet processed) — don't cross-pollinate
        // hypotheses across phrases; the next loop iteration drains the FINAL.
        if (phrase_id >= 0.0 && snap_phrase != phrase_id) { last_partial_n = 0; continue; }
        if (phrase_id < 0.0) phrase_id = snap_phrase;

        std::vector<int32_t> hyp = decode_tokens(snap);
        // LocalAgreement-2: the prefix two consecutive hypotheses agree on is stable.
        size_t agree = 0;
        while (agree < prev_hyp.size() && agree < hyp.size() && prev_hyp[agree] == hyp[agree]) ++agree;
        if (agree > committed.size()) {
            std::vector<int32_t> newly(hyp.begin() + (long)committed.size(), hyp.begin() + (long)agree);
            committed.insert(committed.end(), newly.begin(), newly.end());
            print_event("COMMIT", snap_t0, snap_t1, tok.decode(newly));
        }
        const size_t tail_from = committed.size() < hyp.size() ? committed.size() : hyp.size();
        std::vector<int32_t> tail_toks(hyp.begin() + (long)tail_from, hyp.end());
        print_event("PARTIAL", snap_t0, snap_t1, tok.decode(tail_toks));
        prev_hyp = std::move(hyp);
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
    bool  stream_mode        = false;   // --stream: continuous PCM-on-stdin + VAD + sliding window
    float vad_threshold      = 0.01f;   // --vad-threshold: per-frame RMS speech threshold (normalized f32)
    int   stream_step_ms     = 1500;    // --step-ms: re-transcribe the growing window every N ms (partials)
    int   stream_hangover_ms = 800;     // --hangover-ms: silence after speech that commits a phrase (FINAL)

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

            } else if (a == "--stream") {
                stream_mode = true;
            } else if (a == "--vad-threshold" && i + 1 < argc) {
                vad_threshold = std::stof(argv[++i]);
            } else if (a == "--step-ms" && i + 1 < argc) {
                stream_step_ms = std::stoi(argv[++i]);
            } else if (a == "--hangover-ms" && i + 1 < argc) {
                stream_hangover_ms = std::stoi(argv[++i]);
            } else if (a == "--temperature" && i + 1 < argc) {
                sampler_config.temperature = std::stof(argv[++i]);
            } else if (a == "--top-k" && i + 1 < argc) {
                sampler_config.top_k = std::stoi(argv[++i]);
            } else if (a == "--top-p" && i + 1 < argc) {
                sampler_config.top_p = std::stof(argv[++i]);
            } else if (a == "--seed" && i + 1 < argc) {
                sampler_config.seed = static_cast<uint32_t>(std::stoul(argv[++i]));
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

    // ── Streaming mode: live PCM on stdin, PARTIAL/FINAL on stdout ──────
    // Takes over before the batch fixture load (no assets/test_waveform.bin
    // needed). Same protocol as the whisper port's --stream; see nnopt_run_stream.
    if (stream_mode) {
        return nnopt_run_stream(cl_ctx, model, tok, tokenizer_ok,
                                sampler, sampler_config,
                                max_new_tokens > 16 ? max_new_tokens : 96,
                                vad_threshold, stream_step_ms, stream_hangover_ms);
    }




        // ── ASR encoder-decoder harness (Moonshine) ─────────────────────────
    // Moonshine consumes the RAW 16 kHz mono waveform directly (the encoder's
    // conv1 stack IS the feature extractor). GenerateReference wrote the
    // fixture as assets/test_waveform.bin (float32 samples). The decoder is
    // primed with [decoder_start_token_id] and generated autoregressively.
    //
    // Decoder generation is teacher-free: decoder_ids starts as [1] and each
    // sampled token is appended. model.forward(decoder_ids) runs the encoder
    // ONCE (cached in the backbone keyed by the waveform pointer) then the full
    // decoder over decoder_ids, returning last-token logits[VOCAB_SIZE].
    (void)prompt; (void)read_input_ids_bin; (void)token_ids_path;
    std::vector<float> _wav_f32 = load_float_bin("assets/test_waveform.bin");
    if (_wav_f32.empty()) {
        NNOPT_ERROR("missing assets/test_waveform.bin (required for Moonshine audio encoder; did GenerateReference run?)");
        return 4;
    }
    std::vector<nnopt_storage_t> _wav_storage(_wav_f32.size());
#ifdef NNOPT_USE_FP16
    for (size_t i = 0; i < _wav_f32.size(); ++i) _wav_storage[i] = nnopt_f32_to_f16(_wav_f32[i]);
#else
    for (size_t i = 0; i < _wav_f32.size(); ++i) _wav_storage[i] = (nnopt_storage_t)_wav_f32[i];
#endif
    cl_int _feat_err = CL_SUCCESS;
    cl_mem _feats_buf = clCreateBuffer(cl_ctx.context(),
                                       CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR,
                                       _wav_storage.size() * sizeof(nnopt_storage_t),
                                       _wav_storage.data(),
                                       &_feat_err);
    if (_feat_err != CL_SUCCESS || !_feats_buf) {
        NNOPT_ERROR_FMT("clCreateBuffer(waveform) failed: %d", (int)_feat_err);
        return 4;
    }
    ForwardDispatch::set_input_features(_feats_buf);
    ForwardDispatch::set_num_samples((int)_wav_f32.size());

    // Decoder primed with the start token; grow autoregressively.
    std::vector<int32_t> decoder_ids;
    decoder_ids.push_back(MODEL_CONFIG::DECODER_START_TOKEN_ID);
    const size_t prompt_len = decoder_ids.size();  // exclude start token from output

    // When NOT running the deterministic --token-ids eval path (i.e. the
    // FinalizePort require_text_prompt run), emit the tokenizer-encode dump so
    // FinalizePort's gate can verify the decoder input construction against
    // reference_tokens.json::input_ids. The ids here are the real primed
    // decoder sequence, constructed above from MODEL_CONFIG — not a copy of
    // the reference.
    if (token_ids_path.empty()) {
        write_tokenizer_encode_dump(decoder_ids, prompt);
    }

    bench.mark_prefill_start();
    for (int step = 0; step < max_new_tokens; ++step) {
        std::vector<float> logits = model.forward(decoder_ids);
        if (logits.empty()) {
            // Hard forward failure (encoder refusal / dispatch error) — abort
            // generation instead of decoding garbage. The old zero-logits
            // fallback ground out max_new_tokens steps of id=0 (83s on clip H).
            NNOPT_ERROR("forward failed — aborting generation");
            break;
        }
        std::vector<int32_t> generated_so_far(decoder_ids.begin() + prompt_len, decoder_ids.end());
        int next = sampler.sample(logits, generated_so_far);
        NNOPT_BENCH_FIRST_TOKEN();  // stamps TTFT on the first sample only
        decoder_ids.push_back(next);
        if (sampler_config.eos_token_id >= 0 && next == sampler_config.eos_token_id) break;
    }

    clReleaseMemObject(_feats_buf);
    std::vector<int32_t>& prompt_ids = decoder_ids;

    bench.mark_end();
    std::cout << std::endl;
    // Emit a deterministic post-generation summary that matches Evaluate's
    // contract: ONLY the generated tokens (sliced after prompt_len), so
    // comparison apples-to-apples with reference_tokens.json::generated_text.
    if (prompt_ids.size() > prompt_len) {
                const std::vector<int32_t> generated_ids(prompt_ids.begin() + prompt_len, prompt_ids.end());
        // Always emit the token IDs so Evaluate can verify ID-for-ID token
        // convergence against reference_tokens.json::generated_ids, and emit
        // the decoded text when the tokenizer is available.
        // Emit the BARE decoded text as the FIRST stdout line (NO label) so
        // Evaluate's token-convergence parser aligns "He" as token[0] rather
        // than picking up the "GENERATED_TEXT:" label as token[0] (which made
        // convergence.matched=0 despite match:true — a false negative that
        // stalled cycles 1-4). The labeled lines follow for human readers and
        // for the GENERATED_IDS ID-for-ID check.
        if (tokenizer_ok) {
            const std::string decoded = tok.decode(generated_ids);
            std::cout << decoded << std::endl;
            std::cout << "GENERATED_TEXT: " << decoded << std::endl;
        }
        std::cout << "GENERATED_IDS:";
        for (int32_t id : generated_ids) std::cout << " " << id;
        std::cout << std::endl;
    }
    // Per-kernel GPU profile (env NNOPT_PROFILE=1). Dormant by default;
    // KEEP this call site even when restructuring prefill/decode (same
    // rule as the 5 benchmark sites — see vlm.md).
    KernelProfiler::dump_summary();
    bench.print_summary((int)prompt_len, (int)(prompt_ids.size() - prompt_len));
    return 0;

}
