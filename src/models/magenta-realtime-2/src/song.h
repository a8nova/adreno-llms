#pragma once
// ─────────────────────────────────────────────────────────────────────────────
// Magenta-RT-2 song generation — the PRODUCTION path.
//
// One call renders a chunk of music: style conditioning → temporal AR frames →
// depthformer RVQ tokens → SpectroStream codec → 48 kHz stereo PCM.
//
//   1 frame = 1920 samples = 40 ms @ 48 kHz   →   50 frames = 2.00 s
//
// This is the same pipeline the `song` op-test measures; the op-test is now a
// thin caller of run_song() so there is exactly one e2e code path to optimize
// (and every benchmark in BENCHMARK.md keeps measuring the shipped code).
//
// Streaming: pass a SongSession to carry the temporal KV cache and the next
// frame embedding across calls, so chunk N+1 continues chunk N musically
// instead of restarting from the style embedding.
// ─────────────────────────────────────────────────────────────────────────────

#include <string>
#include <vector>

#include "opencl_context.h"
#include "weights.h"

// Style conditioning for one generation. Loaded host-side (song_io.cpp) so
// run_song() never touches the filesystem.
struct SongConditioning {
    std::vector<float> temporal_input;   // [1024] initial frame embedding
    // Style tokens: 12 MusicCoCa RVQ codes ++ 132 control/MIDI channels. When present the engine
    // runs the ported style encoder ON DEVICE to produce `source`, which is the real conditioning
    // path — text → tokens → encoder → cross-attention. `source` below is only the fallback for
    // when nothing supplies tokens.
    std::vector<int32_t> style_tokens;   // [144] or empty
    std::vector<float> source;           // [256]  cross-attention conditioning (fallback)
    std::string        name;             // label for logs / bench lines
    bool               is_fixture = false;  // true = the shipped debug fixture, not a real style
};

// Sweep CLBlast's Xgemm parameter space on THIS device against a real codec chunk, and return a
// compact "name:ms name:ms ..." summary. `grid` is SongResult::tokens from a completed render, so
// the sweep measures the codec on the audio the caller just generated rather than on a fixture.
// Candidates whose output does not match the first candidate's are marked ":BAD" — a parameter set
// can build, run, and compute the wrong thing.
// Decode a fixed synthetic token grid whole, then in CK-frame pieces, and report how far apart the
// two waveforms are (CODECCHUNK line on stderr). Chunking the codec is supposed to be a slicing
// decision, not a numerical one; this says whether it is. Runs in-process so it is reachable from a
// device with no shell.
// Arm the depth loop's sampler for the next render. temperature <= 0 restores greedy argmax.
void nnopt_set_sampling(float temperature, int top_k, unsigned seed);

bool nnopt_codecchunk_check(OpenCLContext& cl_ctx, Weights& weights, cl_command_queue queue,
                            int T, int CK);

std::string nnopt_xgemm_sweep(OpenCLContext& cl_ctx, Weights& weights,
                              const std::vector<int>& grid, int n_frames, int chunk);

struct SongConfig {
    int  n_frames = 100;     // 100 frames = 2.00 s of audio
    int  chunk    = 0;       // codec time-chunk; 0 = measure the device and size it (see run_song)
    // DEFAULT ON. The AR/codec two-queue path is also the only path with record/replay, and a
    // feature the user must type a flag to reach is a feature that does not ship. Automatically
    // falls back to the sequential path when a streaming session has to be CONTINUED (see run_song)
    // — that is a correctness constraint, not a preference. pipeline=0 forces sequential.
    bool pipeline = true;    // run the AR and the codec on two queues
    bool reset    = false;   // drop any carried session state before generating
    // codecab=1 — decode every chunk TWICE, once with the fp32 codec and once with the fp16 one,
    // and report how far apart the two waveforms are. The point is that fp16 GEMM does not fail
    // loudly: CLBlast's HGEMM accumulates in fp16 with K up to 4608, so a bad result is not silence
    // or NaN, it is audio that is slightly worse in a way one listen does not reliably catch. This
    // turns "does it still sound right" into a number that is on screen next to the speed-up.
    //
    // The listener still hears whichever path NNOPT_CODECFP16 selected; the other one is decoded
    // only to compare against, and its time is subtracted from codec_sec so the timing stays honest.
    bool codec_ab = false;
    // ── sampling ────────────────────────────────────────────────────────────────────────────────
    // This port decoded with pure greedy argmax until 2026-08-27, which is the textbook degeneracy
    // mode for an audio LM: with nothing stochastic the model can reach a fixed point and never
    // leave, heard as a short figure repeating forever. Reference defaults, from
    // magenta_rt/mlx/system.py: temperature 1.3, top_k 40.
    //
    // temperature <= 0 restores exact greedy behaviour, including tie-breaking, so the old path
    // stays reachable for A/B and for the bit-reproducibility the op-tests rely on.
    float temperature = 1.3f;
    int   top_k       = 40;
    // Seed 0 means "vary per render" — the engine substitutes a per-request value so consecutive
    // chunks of a live stream do not draw identical noise. Any non-zero value is honoured exactly,
    // which makes a render reproducible.
    unsigned seed     = 0;
};

struct SongResult {
    std::vector<float> pcm;        // interleaved stereo float
    // The RVQ token grid the AR produced, [n_frames * 12]. Exposed because the waveform is the
    // WRONG place to compare two AR configurations: sampling is greedy, so one flipped token makes
    // the rest of the piece different music and every waveform metric reads as "totally different"
    // whether the cause was a rounding difference or a broken kernel. Tokens are the thing that can
    // actually be identical, so they are the thing to diff.
    std::vector<int>   tokens;
    int    n_samples = 0;          // per channel
    double ar_s      = 0.0;        // AR generation seconds (0 when pipelined — stages overlap)
    double codec_s   = 0.0;        // codec decode seconds  (0 when pipelined)
    double total_s   = 0.0;
    double audio_s   = 0.0;
    // RTF = T/L: seconds of audio produced per second of wall clock. HIGHER IS FASTER, and >= 1
    // means the render keeps up with playback.
    //
    // This was total_s / audio_s — the reciprocal — until 2026-08-27. Both conventions are in common
    // use, and "Live Music Models" (arXiv 2508.04651, the paper for this model) states the choice
    // explicitly in footnote 2: "RTF is commonly defined as both L/T and T/L. Here we use T/L, i.e.,
    // higher RTF means faster." We were printing the other one with a "x" suffix, so a run 12%
    // SLOWER than real time displayed as "1.13x" and read as 13% faster. Every RTF number in
    // BENCHMARK.md before that date is the old convention and has been converted.
    double rtf() const { return total_s > 0.0 ? audio_s / total_s : 0.0; }
};

// Opaque streaming state (temporal KV cache + next frame embedding + position).
struct SongSession;
SongSession* song_session_create();
void         song_session_destroy(SongSession* s);

// Render cfg.n_frames of audio. `sess` may be null (one-shot, fresh state).
bool run_song(OpenCLContext& cl_ctx, Weights& weights,
              const SongConditioning& cond, const SongConfig& cfg,
              SongResult& out, SongSession* sess = nullptr);

// ── host-side helpers (song_io.cpp) ──────────────────────────────────────────

// Resolve a style: a path to a .bin, or a name looked up as styles/<name>.bin
// (1280 fp32 = 1024 temporal ++ 256 source). Empty name → the shipped fixture
// pair (weights/optest_input.bin + weights/optest_source.bin).
// Returns false only when nothing loadable was found; `why` always explains.
bool song_load_conditioning(const std::string& style, SongConditioning& out, std::string& why);

// 48 kHz stereo, 16-bit. Matches the codec's validated output scaling
// (0.5 gain, round-half-down) — do not "simplify" this, the reference WAVs
// in BENCHMARK.md were produced with exactly this quantization.
bool song_write_wav(const std::string& path, const std::vector<float>& pcm_interleaved, int n_samples);
