// Magenta-RT-2 (google/magenta-realtime-2) — Android/OpenCL music generation.
//
// Modality: style conditioning → 48 kHz stereo music. NOT a text LM: there are
// no prompt tokens to decode and no logits to sample here — the temporal AR
// transformer emits RVQ codebook tokens which the SpectroStream codec turns
// into a waveform. The whole pipeline lives behind run_song() (src/song.h).
//
//   1 frame = 1920 samples = 40 ms @ 48 kHz  →  50 frames = 2.00 s
//
// Two modes:
//
//   one-shot   ./bin [style] [--seconds 2] [--out out.wav] [--chunk 5] [--pipeline]
//   serve      ./bin --serve      — persistent process, one generation per stdin
//                                   line, so the app pays the ~2 GB weight load
//                                   once. Markers on stderr:
//                                     SERVE_READY
//                                     SERVE_DONE idx=0 wav=... gen_sec=... rtf=...
//
// Every request line in serve mode may carry its own config as `key=value`
// tokens (frames/seconds/chunk/pipeline/reset/style/out, plus any NNOPT_* env
// toggle). That is deliberate: on a device farm the APK cannot be rebuilt per
// experiment, so the config space has to be reachable from the request itself.

#include "model.h"
#include "model_config.h"
#include "opencl_context.h"
#include "weights.h"
#include "song.h"
#include "musiccoca.h"
#include "spm_tokenizer.h"
#include "utils.h"          // nnopt_toggle_bump — per-request NNOPT_* refresh
#include "debug_utils.h"
#include "version.h"
#include <chrono>
#include "benchmark.h"
#include "profiler.h"

#include <algorithm>
#include <cctype>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <iostream>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>

namespace {

constexpr double kFramesPerSecond = 25.0;   // 1920 samples @ 48 kHz = 40 ms/frame

struct Options {
    SongConfig  cfg;
    // 12 MusicCoCa style tokens ++ 132 control channels. Supplied by the caller (the app runs the
    // MusicCoCa text tower); the engine then computes the conditioning on device instead of reading
    // a canned vector. Empty falls back to the shipped fixture.
    std::vector<int32_t> style_tokens;
    // xgemmsweep=1 — after rendering normally, re-run the CODEC once per CLBlast Xgemm candidate on
    // the tokens just produced and report the times. The device is remote, so this is the only way
    // the tuning measurement ever gets taken (handoff §6: a CLI-only measurement is never taken).
    bool xgemm_sweep = false;
    int  codecchunk_T = 0, codecchunk_CK = 0;   // codecchunk=T,CHUNK; 0 = do not run
    std::string style;
    std::string out = "output.wav";
    bool        serve = false;
    // --prompt / prompt=: a written style, resolved to conditioning ON DEVICE via SentencePiece +
    // the MusicCoCa text tower. Takes precedence over --style-tokens and over the shipped fixture.
    std::string prompt;
    // --blend / blend=: several written styles at once, each with a weight, blended in MusicCoCa's
    // EMBEDDING space before quantization — `c = sum(wi * M(ci)) / sum(wi)`, which is the same
    // weighted average the model was conditioned on upstream. Format is weight-first so a colon
    // inside a prompt is unambiguous: `0.6:minimal techno|0.4:harp`. Wins over `prompt=`.
    std::string blend;
    // steer=1 (default): a conditioning change does NOT restart the piece — the AR attention state
    // carries, so dragging on the app's 2-D surface morphs the music instead of producing a run of
    // disconnected 2-second clips. steer=0 restores "new conditioning = new piece". An explicit
    // reset=1 always resets, either way.
    bool        steer = true;
    // --musiccoca-ids: run the text tower standalone on already-tokenized ids and print what it
    // produced. This is the gate for the MusicCoCa port — the host compares the printed embedding
    // and style tokens against musiccoca/musiccoca_golden.json, which came from the TFLite
    // reference itself. It loads only the MusicCoCa blob, not the 493 MB LLM weights.
    std::vector<int32_t> mc_ids;
    bool        mc_test = false;
    // --musiccoca-text: the same gate driven from a raw string, so the SentencePiece port is
    // covered by the same comparison instead of being trusted.
    std::string mc_text;
    // midi= / --midi: the player's note state at the moment this request was issued. Held STATE,
    // not a stream of events — the model is conditioned on what is sounding now, so a request that
    // arrives mid-chord needs the whole chord, not the one note-on that happened to be newest.
    //
    // `midi.active` is what distinguishes "no MIDI in this request" (leave the conditioning block
    // exactly as the style resolver built it) from "MIDI is live and nothing is held" (all 128
    // pitches explicitly masked). Those two produce IDENTICAL tokens today, which is the point:
    // plugging in a controller and touching nothing must not change a single sample.
    MidiState   midi;
};

int frames_for_seconds(double seconds) {
    const int f = (int)lround(seconds * kFramesPerSecond);
    return f > 0 ? f : 1;
}

// Print what we are actually running on. On a device farm you do not get to
// choose the phone — a mis-booked device (wrong SoC, wrong GPU) has to be
// visible in the first line of the log, not discovered a week later.
void print_device_banner(OpenCLContext& cl_ctx) {
    char version[256] = {0};
    cl_ulong gmem = 0, alloc = 0;
    clGetDeviceInfo(cl_ctx.device(), CL_DEVICE_VERSION, sizeof(version), version, nullptr);
    clGetDeviceInfo(cl_ctx.device(), CL_DEVICE_GLOBAL_MEM_SIZE, sizeof(gmem), &gmem, nullptr);
    clGetDeviceInfo(cl_ctx.device(), CL_DEVICE_MAX_MEM_ALLOC_SIZE, sizeof(alloc), &alloc, nullptr);
    size_t ext_bytes = 0;
    clGetDeviceInfo(cl_ctx.device(), CL_DEVICE_EXTENSIONS, 0, nullptr, &ext_bytes);
    std::string ext(ext_bytes ? ext_bytes - 1 : 0, '\0');
    if (ext_bytes) clGetDeviceInfo(cl_ctx.device(), CL_DEVICE_EXTENSIONS, ext_bytes, &ext[0], nullptr);
    std::fprintf(stderr, "DEVICE name=\"%s\" cl_version=\"%s\" global_mem_mb=%llu max_alloc_mb=%llu\n",
                 cl_ctx.device_name().c_str(), version,
                 (unsigned long long)(gmem >> 20), (unsigned long long)(alloc >> 20));
    std::fprintf(stderr, "cl_device_extensions=%s\n", ext.c_str());

    // OPT_AVAIL — a one-line verdict on the levers that are worth engineering effort, because the
    // full extension string is long enough that it scrolls out of the app's log viewer and the
    // interesting entries get missed. Each of these changes what is worth building next:
    //   recordable_queues   record the AR's ~455-dispatch/frame sequence ONCE and replay it
    //   onchip_global_mem   place intermediates in the 840's 18 MB HPM explicitly
    //   ml_ops              Qualcomm's own tuned GEMM/conv, instead of ours or CLBlast's
    //   subgroups           wave-level reductions without local memory + barriers
    //   dot_product8        int8 MACs, if depth weights ever move to int8
    {
        struct { const char* name; const char* why; } probes[] = {
            {"cl_qcom_recordable_queues",     "record+replay dispatch sequence"},
            {"cl_qcom_onchip_global_memory",  "explicit on-chip (HPM) buffers"},
            {"cl_qcom_ml_ops",                "vendor-tuned ML kernels"},
            {"cl_khr_subgroups",              "barrier-free wave reductions"},
            {"cl_qcom_subgroup_shuffle",      "cross-lane exchange"},
            {"cl_qcom_reqd_sub_group_size",   "pin wave size"},
            {"cl_qcom_dot_product8",          "int8 dot products"},
            {"cl_khr_fp16",                   "half precision"},
            {"cl_khr_image2d_from_buffer",    "zero-copy texture views"},
        };
        std::string avail, missing;
        for (const auto& p : probes) {
            const bool have = ext.find(p.name) != std::string::npos;
            (have ? avail : missing) += std::string(" ") + p.name;
        }
        std::fprintf(stderr, "OPT_AVAIL  yes:%s\n", avail.empty()   ? " (none)" : avail.c_str());
        std::fprintf(stderr, "OPT_AVAIL  no: %s\n", missing.empty() ? " (none)" : missing.c_str());
        // Occupancy inputs — the guide's workgroup guidance is all relative to these two.
        size_t max_wg = 0; cl_uint cus = 0;
        clGetDeviceInfo(cl_ctx.device(), CL_DEVICE_MAX_WORK_GROUP_SIZE, sizeof(max_wg), &max_wg, nullptr);
        clGetDeviceInfo(cl_ctx.device(), CL_DEVICE_MAX_COMPUTE_UNITS, sizeof(cus), &cus, nullptr);
        cl_ulong lmem = 0;
        clGetDeviceInfo(cl_ctx.device(), CL_DEVICE_LOCAL_MEM_SIZE, sizeof(lmem), &lmem, nullptr);
        std::fprintf(stderr, "OPT_AVAIL  max_wg=%zu compute_units=%u local_mem_kb=%llu\n",
                     max_wg, (unsigned)cus, (unsigned long long)(lmem >> 10));
    }
    std::fflush(stderr);
}

// One machine-parsable line per generation — the only number the Lab screen and
// CI read. Wall time only (per the e2e-wall rule): internal spans lie about
// overlap, the process wall does not.
void print_bench_line(const std::string& device, const Options& o,
                      const SongConditioning& cond, const SongResult& r) {
    // The four numbers every optimisation decision today turned on: how many dispatches a frame
    // costs, what each costs the host, how much GPU work a frame actually contains, and therefore
    // how much of the time the GPU is doing nothing. gpu_ms is 0 unless a frame was profiled.
    long disp = 0; double us_disp = 0.0;
    OpenCLContext::hostProfTotals(o.cfg.n_frames, &disp, &us_disp);
    const double gpu_ms = OpenCLContext::arProfFrameMs();
    const double frame_ms = (o.cfg.n_frames > 0) ? (r.ar_s * 1000.0 / o.cfg.n_frames) : 0.0;
    const double gpu_util = (gpu_ms > 0.0 && frame_ms > 0.0) ? 100.0 * gpu_ms / frame_ms : 0.0;
    // EMIT ONLY WHEN REAL. gpu_ms needs a GPU-profiled frame, which a fast build does not do, and a
    // row that always reads 0.00 on the report card looks like a broken engine rather than a
    // disabled measurement.
    // Both queues. AR GPU per frame is only known when a frame was profiled; the codec's span comes
    // from queue markers and is always available. gpu_busy is the honest utilisation figure — every
    // one before this counted the AR's queue alone while the codec ran beside it.
    const double codec_gpu_ms = nnopt_codec_gpu_ms();
    const double ar_gpu_ms    = gpu_ms * (double)o.cfg.n_frames;      // 0 when unprofiled
    const double gpu_busy_pct = (r.total_s > 0.0)
                              ? 100.0 * (ar_gpu_ms + codec_gpu_ms) / (r.total_s * 1000.0) : 0.0;
    char gpubuf[64] = {0};
    if (gpu_ms > 0.0)
        std::snprintf(gpubuf, sizeof(gpubuf), " gpu_ms=%.2f gpu_util=%.0f", gpu_ms, gpu_util);
    // fp32-vs-fp16 codec verdict. EMITTED ONLY WHEN AN A/B RAN — a row that always reads 1.000000
    // would be indistinguishable from a comparison that never happened, and this particular number
    // exists to be trusted when it says the audio did not change.
    char abbuf[96] = {0};
    { double abc = 0.0, abe = 0.0; int abn = 0;
      nnopt_get_codec_ab(&abc, &abe, &abn);
      if (abn > 0) std::snprintf(abbuf, sizeof(abbuf), " codec_cos=%.6f codec_maxerr=%.6g", abc, abe); }
    // What the fp16 path actually did. Emitted whenever any conv took it, A/B or not.
    char f16buf[128] = {0};
    { int nf = 0, ns = 0, nx = 0;
      nnopt_codec_fp16_get(&nf, &ns, &nx);
      if (nf + nx > 0) std::snprintf(f16buf, sizeof(f16buf),
                                     " codec_fp16=%d codec_fp16_fail=%d", nf, nx); }
    std::fprintf(stderr,
                 "BENCH model=magenta-rt2 device=\"%s\" style=\"%s\" fixture=%d frames=%d chunk=%d "
                 "pipeline=%d ar_sec=%.3f codec_sec=%.3f total_sec=%.3f audio_sec=%.3f rtf=%.3f "
                 "disp=%ld us_disp=%.1f codec_gpu=%.0f gpu_busy=%.0f tok_s=%.0f%s%s%s\n",
                 device.c_str(), cond.name.c_str(), cond.is_fixture ? 1 : 0,
                 o.cfg.n_frames, o.cfg.chunk, o.cfg.pipeline ? 1 : 0,
                 r.ar_s, r.codec_s, r.total_s, r.audio_s, r.rtf(),
                 disp, us_disp, codec_gpu_ms, gpu_busy_pct,
                 // Token throughput, the paper's chunk-size-independent form of the same claim:
                 // "we reduce the bandwidth to 4kbps by generating only the first 16 RVQ levels,
                 // yielding a live throughput target of 400 tokens per second" = frame rate x RVQ
                 // depth. This port generates 12 levels at 25 Hz, so its target is 300 tok/s.
                 // Measured against total_s, not ar_s: the LM keeping up is not enough, the whole
                 // pipeline has to.
                 (r.total_s > 0.0) ? (double)(o.cfg.n_frames * 12) / r.total_s : 0.0,
                 gpubuf, abbuf, f16buf);
    std::fflush(stderr);
}

// `key=value` from a serve request or the command line. NNOPT_* keys are pushed
// into the environment so the existing toggle ledger (NNOPT_ARFLUSH, NNOPT_POOL,
// NNOPT_DIRECTCONV, …) is reachable without rebuilding. Toggles that latch into
// a static on first use only take effect on a fresh process — those are called
// out in BENCHMARK.md.
// "1,2,3,..." -> ints. Rejects the whole list on a bad element rather than conditioning the model
// on a half-parsed style.
bool parse_tokens(const std::string& csv, std::vector<int32_t>& out) {
    std::vector<int32_t> v;
    size_t i = 0;
    while (i < csv.size()) {
        size_t j = csv.find(',', i);
        if (j == std::string::npos) j = csv.size();
        const std::string tok = csv.substr(i, j - i);
        if (tok.empty()) return false;
        char* end = nullptr;
        const long val = std::strtol(tok.c_str(), &end, 10);
        if (end == tok.c_str() || *end != '\0') return false;
        v.push_back((int32_t)val);
        i = j + 1;
    }
    out.swap(v);
    return true;
}

// NNOPT_* keys set by the CURRENT request. setenv persists for the life of the process, so without
// clearing these a toggle set once stays set for every later request in the same warm process: tap
// "gemv v8" then "default" and you are still running v8, with the report labelling it default. That
// silently invalidated one of my own A/B runs before I noticed.
std::vector<std::string> g_request_env;

void clear_request_env() {
    for (const std::string& k : g_request_env) unsetenv(k.c_str());
    g_request_env.clear();
}

bool apply_kv(const std::string& key, const std::string& value, Options& o) {
    if (key == "frames")        { const int v = std::atoi(value.c_str()); if (v > 0) o.cfg.n_frames = v; return true; }
    if (key == "seconds")       { o.cfg.n_frames = frames_for_seconds(std::atof(value.c_str())); return true; }
    if (key == "chunk")         { const int v = std::atoi(value.c_str()); if (v > 0) o.cfg.chunk = v; return true; }
    if (key == "pipeline")      { o.cfg.pipeline = (value != "0"); return true; }
    if (key == "codecab")       { o.cfg.codec_ab = (value != "0"); return true; }
    if (key == "temperature")   { o.cfg.temperature = (float)std::atof(value.c_str()); return true; }
    if (key == "topk")          { o.cfg.top_k = std::atoi(value.c_str()); return true; }
    if (key == "seed")          { o.cfg.seed = (unsigned)std::strtoul(value.c_str(), nullptr, 10); return true; }
    // codecchunk=T,CHUNK — run the chunked-vs-whole codec comparison for this request and print
    // CODECCHUNK, then carry on serving. The op-test form runs once at startup and returns from
    // main, which on a device with no shell means it can never be taken.
    if (key == "codecchunk")    { const size_t c = value.find(',');
                                  o.codecchunk_T  = std::atoi(value.substr(0, c).c_str());
                                  o.codecchunk_CK = (c == std::string::npos) ? 0 : std::atoi(value.substr(c+1).c_str());
                                  return true; }
    if (key == "reset")         { o.cfg.reset = (value != "0"); return true; }
    if (key == "steer")         { o.steer    = (value != "0"); return true; }
    if (key == "style")         { o.style = value; return true; }
    if (key == "tokens") {
        if (!parse_tokens(value, o.style_tokens)) {
            std::fprintf(stderr, "SERVE_NOTE bad_tokens=1 (expected a comma-separated int list)\n");
            o.style_tokens.clear();
        } else if (o.style_tokens.size() != 144) {
            std::fprintf(stderr, "SERVE_NOTE tokens=%zu expected=144 — ignoring\n", o.style_tokens.size());
            o.style_tokens.clear();
        }
        return true;
    }
    if (key == "midi") {
        // A bad MIDI string must not silently condition the model on half a chord, and must not
        // kill a live performance either: fall back to "nothing held" and say so on stderr.
        if (!midi_parse(value, o.midi)) {
            std::fprintf(stderr, "SERVE_NOTE bad_midi=1 (expected pitch:state,... [|drum:v])\n");
            o.midi = MidiState();
            o.midi.active = true;
        }
        return true;
    }
    if (key == "out")           { o.out = value; return true; }
    if (key.rfind("NNOPT_", 0) == 0) {
        setenv(key.c_str(), value.c_str(), 1);
        g_request_env.push_back(key);
        return true;
    }
    return false;
}

// A serve request: an optional leading style (everything before the first
// `key=value` token, so multi-word prompts survive), then `key=value` tokens.
void parse_request(const std::string& line, Options& o) {
    clear_request_env();   // a toggle applies to the request that asked for it, and no others
    size_t i = 0;
    bool seen_kv = false;
    std::string style;
    const std::string style_in = o.style;
    while (i < line.size()) {
        while (i < line.size() && std::isspace((unsigned char)line[i])) ++i;
        const size_t start = i;
        while (i < line.size() && !std::isspace((unsigned char)line[i])) ++i;
        if (start == i) break;
        const std::string tok = line.substr(start, i - start);
        // `prompt=` takes the whole rest of the line: prompts are multi-word ("disco funk with a
        // slow build"), and splitting one on whitespace would silently condition on a fragment.
        // It therefore has to come last in a request.
        // `blend=` has the same rest-of-line rule as `prompt=` (weights carry multi-word prompts),
        // and wins over it: the app sends one or the other, never both.
        if (tok.rfind("blend=", 0) == 0) {
            o.blend = line.substr(start + 6);
            while (!o.blend.empty() && std::isspace((unsigned char)o.blend.back())) o.blend.pop_back();
            o.prompt.clear();
            seen_kv = true;
            break;
        }
        if (tok.rfind("prompt=", 0) == 0) {
            o.prompt = line.substr(start + 7);
            while (!o.prompt.empty() && std::isspace((unsigned char)o.prompt.back())) o.prompt.pop_back();
            seen_kv = true;
            break;
        }
        const size_t eq = tok.find('=');
        if (eq != std::string::npos && eq > 0) {
            seen_kv = true;
            const std::string key = tok.substr(0, eq), value = tok.substr(eq + 1);
            if (!apply_kv(key, value, o))
                std::fprintf(stderr, "SERVE_NOTE unknown_key=%s\n", key.c_str());
        } else if (!seen_kv) {
            if (!style.empty()) style += " ";
            style += tok;
        } else {
            std::fprintf(stderr, "SERVE_NOTE ignored_token=%s\n", tok.c_str());
        }
    }
    // An explicit `style=` anywhere in the line wins over the bare leading words.
    if (!style.empty() && o.style == style_in) o.style = style;
}

// A written prompt → the 144 conditioning tokens, entirely on device: SentencePiece → MusicCoCa
// text tower → residual VQ → conditioning block. Both the tokenizer and the tower are gated
// against the upstream reference (scripts/spm_device_test.py, scripts/musiccoca_device_test.py).
//
// The MusicCoCa weights are held in a static across calls. Re-prompting is the whole point of the
// steering feature, and a fresh load would charge ~1.6 s for the lazy 228 MB upload every time a
// user changes their mind; warm, the same call is ~150 ms.
SpmTokenizer* g_spm = nullptr;
Weights* g_musiccoca = nullptr;

// prompt text -> its 768-d MusicCoCa embedding. The text tower is ~150 ms warm, and a blend
// re-resolves on EVERY chunk while a finger is moving — but only the WEIGHTS change, not the
// prompts. Caching the per-prompt embedding turns a puck drag into a weighted average plus one
// RVQ (sub-millisecond) instead of N tower passes. This is the difference between a surface that
// steers in real time and one that costs 150 ms x N per 2 s of audio.
std::unordered_map<std::string, std::vector<float>> g_emb_cache;
constexpr size_t kEmbCacheMax = 32;   // the app's surface holds at most 8 prompts

void release_musiccoca() {
    delete g_musiccoca;
    g_musiccoca = nullptr;
    g_emb_cache.clear();
}

bool ensure_musiccoca(OpenCLContext& cl_ctx) {
    if (!g_spm) {
        g_spm = new SpmTokenizer();
        if (!g_spm->load("weights/musiccoca_spm.bin")) {
            delete g_spm;
            g_spm = nullptr;
            return false;
        }
    }
    if (!g_musiccoca) {
        g_musiccoca = new Weights();
        if (!g_musiccoca->load("weights/musiccoca.fp16.bin", "weights/musiccoca.fp16.meta.json",
                               cl_ctx.context())) {
            NNOPT_ERROR("prompt: weights/musiccoca.fp16.bin failed to load");
            delete g_musiccoca;
            g_musiccoca = nullptr;
            return false;
        }
    }
    return true;
}

// One prompt -> its 768-d embedding, through the cache. [n_ids_out] is the token count when this
// call actually ran the tower, and -1 on a cache hit (nothing was tokenized).
bool embed_prompt_cached(OpenCLContext& cl_ctx, const std::string& prompt,
                         const std::vector<float>** emb_out, int* n_ids_out) {
    *n_ids_out = -1;
    auto it = g_emb_cache.find(prompt);
    if (it != g_emb_cache.end()) { *emb_out = &it->second; return true; }
    if (!ensure_musiccoca(cl_ctx)) return false;

    std::vector<int32_t> ids;
    bool non_ascii = false;
    if (!g_spm->encode(prompt, kMusicCoCaMaxSeq, ids, &non_ascii)) return false;
    if (non_ascii) {
        std::fprintf(stderr, "PROMPT_NOTE non_ascii=1 (nmt_nfkc folding not applied)\n");
    }
    std::vector<float> emb;
    if (!musiccoca_embed_text(cl_ctx, *g_musiccoca, ids, emb)) return false;

    // Bounded: a user can type many prompts over a long session, and this holds GPU-derived floats.
    if (g_emb_cache.size() >= kEmbCacheMax) g_emb_cache.clear();
    it = g_emb_cache.emplace(prompt, std::move(emb)).first;
    *emb_out = &it->second;
    *n_ids_out = (int)ids.size();
    return true;
}

void log_style_tokens(const char* label, const std::string& what, int n_ids,
                      const std::vector<int32_t>& style) {
    std::fprintf(stderr, "%s \"%s\" ", label, what.c_str());
    if (n_ids >= 0) std::fprintf(stderr, "ids=%d ", n_ids); else std::fprintf(stderr, "cached ");
    std::fprintf(stderr, "style_tokens=");
    for (size_t i = 0; i < style.size(); ++i) std::fprintf(stderr, "%s%d", i ? "," : "", style[i]);
    std::fprintf(stderr, "\n");
    std::fflush(stderr);
}

// ONE line per request, however many notes are held.
//
// This runs on every chunk of a live performance, down the same stderr pipe the app reads line by
// line — and a host that cannot drain it fast enough blocks this process inside fprintf, mid-render
// and holding the GPU (the failure the serve loop's report suppression exists to prevent). So the
// per-request line is compact and bounded, and the full 144-token block — which is what §6's
// per-stage token diff against magenta_rt actually needs — is behind NNOPT_MIDI_DUMP, for a
// deliberate validation run rather than for every chunk of every performance.
void log_midi_state(const char* label, const MidiState& midi, const std::vector<int32_t>& block) {
    std::string held;
    int n = 0;
    for (int p = 0; p < kMidiNotes; ++p) {
        if (midi.notes[p] < 0) continue;
        ++n;
        if (n <= 12) {   // bounded: a forearm on a keyboard must not produce a 128-entry log line
            if (!held.empty()) held += ",";
            held += std::to_string(p) + ":" + std::to_string((int)midi.notes[p]);
        }
    }
    if (n > 12) held += ",+" + std::to_string(n - 12);
    std::fprintf(stderr, "%s held=%d drum=%d [%s]\n", label, n, (int)midi.drum, held.c_str());
    if (const char* d = std::getenv("NNOPT_MIDI_DUMP")) {
        if (d[0] != '0') {
            std::fprintf(stderr, "MIDI_BLOCK ");
            for (size_t i = 0; i < block.size(); ++i)
                std::fprintf(stderr, "%s%d", i ? "," : "", block[i]);
            std::fprintf(stderr, "\n");
        }
    }
    std::fflush(stderr);
}

// A 768-d embedding -> the 144 conditioning tokens.
bool quantize_to_conditioning(OpenCLContext& cl_ctx, const std::vector<float>& emb,
                              std::vector<int32_t>& cond_out, std::vector<int32_t>& style_out) {
    if (!musiccoca_quantize(cl_ctx, *g_musiccoca, emb, style_out)) return false;
    if (!musiccoca_conditioning_tokens(style_out, cond_out)) return false;
    return true;
}

bool resolve_prompt_tokens(OpenCLContext& cl_ctx, const std::string& prompt,
                           std::vector<int32_t>& cond_out) {
    const std::vector<float>* emb = nullptr;
    int n_ids = -1;
    if (!embed_prompt_cached(cl_ctx, prompt, &emb, &n_ids)) return false;
    std::vector<int32_t> style;
    if (!quantize_to_conditioning(cl_ctx, *emb, cond_out, style)) return false;
    log_style_tokens("PROMPT", prompt, n_ids, style);
    return true;
}

// "0.62:minimal techno|0.31:dub bassline" -> [(0.62,"minimal techno"), (0.31,"dub bassline")].
// Weight first, so a colon inside a prompt is never ambiguous. Zero/negative weights are dropped —
// a prompt the user has dragged out of range should not condition anything.
bool parse_blend(const std::string& s, std::vector<std::pair<float, std::string>>& out) {
    out.clear();
    size_t i = 0;
    while (i <= s.size()) {
        const size_t bar = s.find('|', i);
        std::string item = (bar == std::string::npos) ? s.substr(i) : s.substr(i, bar - i);
        const size_t colon = item.find(':');
        if (colon != std::string::npos) {
            const float w = (float)std::atof(item.substr(0, colon).c_str());
            std::string text = item.substr(colon + 1);
            size_t b = text.find_first_not_of(" \t");
            size_t e = text.find_last_not_of(" \t");
            text = (b == std::string::npos) ? "" : text.substr(b, e - b + 1);
            if (!text.empty() && w > 0.0f) out.emplace_back(w, text);
        }
        if (bar == std::string::npos) break;
        i = bar + 1;
    }
    return !out.empty();
}

// Weighted blend in EMBEDDING space, then quantize once: `c = sum(wi * M(ci)) / sum(wi)`, the same
// weighted average the model is conditioned on upstream. Blending the RVQ TOKENS instead would be
// meaningless — they are codebook indices, not a vector space — which is why this has to happen
// here in the engine rather than in the app.
//
// NNOPT_BLEND_NORM=1 L2-renormalizes the blended vector before quantizing. MusicCoCa is
// contrastive, so its per-prompt embeddings are ~unit-norm while a weighted average of them is
// not; whether upstream renormalizes is worth an on-device A/B rather than a guess. Default off =
// the plain weighted average the formula specifies.
bool resolve_blend_tokens(OpenCLContext& cl_ctx,
                          const std::vector<std::pair<float, std::string>>& pairs,
                          std::vector<int32_t>& cond_out) {
    if (pairs.empty()) return false;
    float wsum = 0.0f;
    for (const auto& p : pairs) wsum += p.first;
    if (wsum <= 0.0f) return false;

    std::vector<float> blend;
    int fresh = 0;
    for (const auto& p : pairs) {
        const std::vector<float>* emb = nullptr;
        int n_ids = -1;
        if (!embed_prompt_cached(cl_ctx, p.second, &emb, &n_ids)) return false;
        if (n_ids >= 0) ++fresh;
        if (blend.empty()) blend.assign(emb->size(), 0.0f);
        if (emb->size() != blend.size()) {
            NNOPT_ERROR("blend: embedding dim mismatch");
            return false;
        }
        const float w = p.first / wsum;
        for (size_t k = 0; k < blend.size(); ++k) blend[k] += (*emb)[k] * w;
    }

    const char* norm = std::getenv("NNOPT_BLEND_NORM");
    if (norm && std::strcmp(norm, "0") != 0) {
        double ss = 0.0;
        for (float v : blend) ss += (double)v * (double)v;
        if (ss > 0.0) {
            const float inv = (float)(1.0 / std::sqrt(ss));
            for (float& v : blend) v *= inv;
        }
    }

    std::vector<int32_t> style;
    if (!quantize_to_conditioning(cl_ctx, blend, cond_out, style)) return false;

    std::string label;
    for (const auto& p : pairs) {
        if (!label.empty()) label += " + ";
        char frac[16];
        std::snprintf(frac, sizeof(frac), "%.0f%% ", 100.0f * p.first / wsum);
        label += frac;
        label += p.second;
    }
    log_style_tokens("BLEND", label, fresh > 0 ? fresh : -1, style);
    return true;
}

// Render one chunk and write it. Returns false if the pipeline produced nothing.
bool generate_to_file(OpenCLContext& cl_ctx, Weights& weights, const Options& o,
                      SongSession* sess, SongResult& r, SongConditioning& cond) {
    std::string why;
    if (!song_load_conditioning(o.style, cond, why)) {
        NNOPT_ERROR_FMT("conditioning load failed — %s", why.c_str());
        return false;
    }
    // A serve request may have just set NNOPT_* toggles via its key=value tokens. Bump the epoch so
    // every cached env lookup (profiling, GEMV variant, …) re-reads instead of keeping the value it
    // latched during the warm-up render — which is what made per-request experiments silently
    // measure the default.
    nnopt_toggle_bump();
    OpenCLContext::profRefresh();
    std::fprintf(stderr, "CONDITIONING %s\n", why.c_str());
    if (!o.style_tokens.empty()) cond.style_tokens = o.style_tokens;   // on-device conditioning
    // Live MIDI overwrites the 129 note/drum channels of whatever block we ended up with. It runs
    // LAST and in place, because the style block above may have come straight out of the per-prompt
    // cache: baking MIDI into that cache would make every note-on look like a prompt change and pay
    // for a MusicCoCa text-tower pass that nothing asked for.
    if (o.midi.active && cond.style_tokens.size() == (size_t)kConditioningTokens) {
        if (!midi_apply_to_conditioning(o.midi, cond.style_tokens)) return false;
        log_midi_state("MIDI", o.midi, cond.style_tokens);
    } else if (o.midi.active) {
        // The fixture path has no 144-token block to patch, so MIDI would be silently dropped.
        std::fprintf(stderr, "SERVE_NOTE midi_ignored=1 (no style tokens — supply prompt=/blend=/tokens=)\n");
    }
    if (!o.prompt.empty()) {
        cond.name = o.prompt;
        cond.is_fixture = false;
    }
    if (!run_song(cl_ctx, weights, cond, o.cfg, r, sess)) return false;
    if (!song_write_wav(o.out, r.pcm, r.n_samples)) return false;
    return true;
}

}  // namespace

int main(int argc, char** argv) {
    // Arm the crash handler FIRST — on SIGSEGV/SIGABRT/SIGBUS it prints the last
    // NNOPT_CHECKPOINT, a backtrace and the GPU-mem allocation log, so a device
    // segfault reports WHERE it died instead of a bare "Segmentation fault".
    nnopt_install_crash_handler();

    Options opt;
    opt.cfg.n_frames = frames_for_seconds(2.0);   // one Magenta chunk
    for (int i = 1; i < argc; ++i) {
        const std::string a = argv[i];
        const bool has_next = (i + 1 < argc);
        if      (a == "--serve")                 opt.serve = true;
        else if (a == "--blend"    && has_next)  opt.blend = argv[++i];
        else if (a == "--no-steer")              opt.steer = false;
        else if (a == "--pipeline")              opt.cfg.pipeline = true;
        else if (a == "--seconds"  && has_next)  opt.cfg.n_frames = frames_for_seconds(std::atof(argv[++i]));
        else if (a == "--frames"   && has_next)  opt.cfg.n_frames = std::max(1, std::atoi(argv[++i]));
        else if (a == "--chunk"    && has_next)  opt.cfg.chunk    = std::max(1, std::atoi(argv[++i]));
        else if (a == "--midi"     && has_next)  {
            if (!midi_parse(argv[++i], opt.midi)) {
                NNOPT_ERROR("--midi: expected \"pitch:state,...[|drum:v]\"");
                return 1;
            }
        }
        else if (a == "--style"    && has_next)  opt.style        = argv[++i];
        else if (a == "--prompt"   && has_next)  opt.prompt       = argv[++i];
        else if (a == "--style-tokens" && has_next) {
            if (!parse_tokens(argv[++i], opt.style_tokens) || opt.style_tokens.size() != 144) {
                NNOPT_ERROR("--style-tokens: expected 144 comma-separated ints");
                return 1;
            }
        }
        else if (a == "--musiccoca-ids" && has_next) {
            if (!parse_tokens(argv[++i], opt.mc_ids) || opt.mc_ids.empty() ||
                (int)opt.mc_ids.size() > kMusicCoCaMaxSeq) {
                NNOPT_ERROR_FMT("--musiccoca-ids: expected 1..%d comma-separated ids", kMusicCoCaMaxSeq);
                return 1;
            }
            opt.mc_test = true;
        }
        else if (a == "--musiccoca-text" && has_next) { opt.mc_text = argv[++i]; opt.mc_test = true; }
        else if (a == "--out"      && has_next)  opt.out          = argv[++i];
        else if (a == "--token-ids" && has_next) ++i;   // accepted for op-test call compatibility
        else if (a.rfind("--", 0) == 0)          { if (has_next && argv[i + 1][0] != '-') ++i; }
        else if (a.find('=') != std::string::npos) {
            const size_t eq = a.find('=');
            apply_kv(a.substr(0, eq), a.substr(eq + 1), opt);
        }
        // Bare positionals are ignored: the op-test call shape is `./bin x 1 --token-ids …`.
    }

    BenchmarkTimer& bench = BenchmarkTimer::instance();
    bench.mark_inference_start();

    OpenCLContext cl_ctx;
    if (!cl_ctx.initialize()) { NNOPT_ERROR("OpenCL init failed"); return 1; }
    print_device_banner(cl_ctx);

    // Recordable-queue probe: report whether capture+replay works on THIS device, before anything
    // is built on top of it. Runs only with NNOPT_RECORDPROBE=1 so a normal launch pays nothing.
    if (const char* rp = std::getenv("NNOPT_RECORDPROBE")) {
        if (rp[0] && rp[0] != '0') {
            std::fprintf(stderr, "RECORDQ has_recordable_queues=%d\n", cl_ctx.has_recordable_queues() ? 1 : 0);
            cl_ctx.record_probe();
        }
    }

    // ── MusicCoCa gate ──────────────────────────────────────────────────────
    // Runs before the LLM weights load: the text tower is independent of them, and the whole point
    // of this path is to prove the tower on its own before anything downstream can mask a bug.
    if (opt.mc_test) {
        // Report load and compute separately: the tower itself is small, and a wall-clock number
        // that hides a 228 MB blob load behind it would misprice the whole feature.
        if (!opt.mc_text.empty()) {
            SpmTokenizer spm;
            if (!spm.load("weights/musiccoca_spm.bin")) return 1;
            bool non_ascii = false;
            if (!spm.encode(opt.mc_text, kMusicCoCaMaxSeq, opt.mc_ids, &non_ascii)) return 1;
            if (non_ascii) {
                std::fprintf(stderr, "MUSICCOCA_NOTE non_ascii=1 (nmt_nfkc folding not applied — "
                                     "ids may differ from the reference)\n");
            }
        }
        const auto t0 = std::chrono::steady_clock::now();
        Weights mc;
        if (!mc.load("weights/musiccoca.fp16.bin", "weights/musiccoca.fp16.meta.json", cl_ctx.context())) {
            NNOPT_ERROR("musiccoca: weights/musiccoca.fp16.bin failed to load");
            return 1;
        }
        const auto t1 = std::chrono::steady_clock::now();
        std::vector<float> emb;
        std::vector<int32_t> toks, cond;
        if (!musiccoca_embed_text(cl_ctx, mc, opt.mc_ids, emb)) return 1;
        const auto t2 = std::chrono::steady_clock::now();
        if (!musiccoca_quantize(cl_ctx, mc, emb, toks)) return 1;
        if (!musiccoca_conditioning_tokens(toks, cond)) return 1;
        const auto t3 = std::chrono::steady_clock::now();
        // Second pass: the first one pays for the lazy 228 MB weight upload and the kernel builds,
        // so it says nothing about what re-prompting costs. Steering re-embeds on every prompt
        // change inside one warm process, and THAT is the number the feature is priced on.
        std::vector<float> emb2;
        std::vector<int32_t> toks2;
        if (!musiccoca_embed_text(cl_ctx, mc, opt.mc_ids, emb2)) return 1;
        if (!musiccoca_quantize(cl_ctx, mc, emb2, toks2)) return 1;
        const auto t4 = std::chrono::steady_clock::now();
        if (toks2 != toks) { NNOPT_ERROR("musiccoca: repeat run disagreed with the first"); return 1; }
        const auto ms = [](auto a, auto b) {
            return std::chrono::duration_cast<std::chrono::milliseconds>(b - a).count();
        };
        std::fprintf(stderr,
                     "MUSICCOCA_BENCH open_ms=%lld cold_tower_ms=%lld cold_rvq_ms=%lld "
                     "warm_ms=%lld tokens=%zu\n",
                     (long long)ms(t0, t1), (long long)ms(t1, t2), (long long)ms(t2, t3),
                     (long long)ms(t3, t4), opt.mc_ids.size());

        std::printf("MUSICCOCA_IDS");
        for (int32_t v : opt.mc_ids) std::printf(" %d", v);
        std::printf("\nMUSICCOCA_TOKENS");
        for (int32_t v : toks) std::printf(" %d", v);
        std::printf("\nMUSICCOCA_COND");
        for (int32_t v : cond) std::printf(" %d", v);
        std::printf("\nMUSICCOCA_EMB");
        for (float v : emb) std::printf(" %.7g", v);
        std::printf("\n");
        std::fflush(stdout);
        return 0;
    }

    // Resolve a written prompt BEFORE the LLM weights are loaded. The tower's 228 MB and the LLM's
    // 493 MB coexisting cost ~10 s of generation time on a 4 GB phone (measured: 84.8 s vs 75.1 s
    // for the same 2 s of audio), so in one-shot mode the tower is loaded, used and dropped first.
    // Serve mode keeps it: there, re-prompting is the feature.
    if (!opt.blend.empty()) {
        std::vector<std::pair<float, std::string>> pairs;
        if (!parse_blend(opt.blend, pairs) ||
            !resolve_blend_tokens(cl_ctx, pairs, opt.style_tokens)) {
            NNOPT_ERROR_FMT("blend \"%s\" could not be resolved to conditioning", opt.blend.c_str());
            return 1;
        }
        if (!opt.serve) release_musiccoca();
    } else if (!opt.prompt.empty()) {
        if (!resolve_prompt_tokens(cl_ctx, opt.prompt, opt.style_tokens)) {
            NNOPT_ERROR_FMT("prompt \"%s\" could not be resolved to conditioning", opt.prompt.c_str());
            return 1;
        }
        if (!opt.serve) release_musiccoca();
    }

    Weights weights;
#ifdef NNOPT_USE_FP16
    std::string weights_bin  = "weights/model.fp16.bin";
    std::string weights_meta = "weights/model.fp16.meta.json";
#else
    std::string weights_bin  = "weights/model.bin";
    std::string weights_meta = "weights/model.meta.json";
#endif
    // NNOPT_QUANT=q4 selects the 4-bit AR bundle. The AR is DRAM-bound — it streams its weights
    // once per frame and the depth stack twelve times per frame — so weight precision, not kernel
    // tuning, sets the ceiling. Which blob is loaded is stated on stderr because it changes both
    // the speed and the sound, and a run whose precision you have to infer is a wasted measurement.
    if (const char* q = std::getenv("NNOPT_QUANT")) {
        if (std::strcmp(q, "q4") == 0) {
            weights_bin = "weights/model.q4.bin";
            weights_meta = "weights/model.q4.meta.json";
        } else if (std::strcmp(q, "fp16") != 0) {
            NNOPT_ERROR_FMT("NNOPT_QUANT=%s is not supported (use q4 or fp16) — loading fp16", q);
        }
    }
    std::fprintf(stderr, "WEIGHTS %s\n", weights_bin.c_str());
    if (!weights.load(weights_bin.c_str(), weights_meta.c_str(), cl_ctx.context())) {
        NNOPT_ERROR_FMT("weights load failed: %s", weights_bin.c_str());
        return 1;
    }

    Model model(cl_ctx, weights);
    if (!model.initialize()) {
        NNOPT_ERROR("Model::initialize() failed — see prior NNOPT_ERROR for the layer that failed");
        return 1;
    }

    // ── op-test harness (NNOPT_OPTEST) ──────────────────────────────────────
    // Every per-op and e2e test lives inside model_forward_graph; one forward
    // runs the selected test and returns. Keeps BENCHMARK.md reproducible.
    if (std::getenv("NNOPT_OPTEST")) {
        model.forward(std::vector<int32_t>(1, 0));
        bench.mark_end();
        bench.print_summary(0, 0);
        return 0;
    }

    bench.mark_prefill_start();

    if (!opt.serve) {
        SongResult r; SongConditioning cond;
        if (!generate_to_file(cl_ctx, weights, opt, /*sess=*/nullptr, r, cond)) return 1;
        bench.mark_end();
        print_bench_line(cl_ctx.device_name(), opt, cond, r);
        std::printf("WROTE %s  %.2fs audio in %.2fs (RTF %.2f)\n", opt.out.c_str(), r.audio_s, r.total_s, r.rtf());
        std::fflush(stdout);
        KernelProfiler::dump_summary();
        // Per-kernel GPU time for a REAL render. This used to be reachable only through the
        // NNOPT_OPTEST harness, which meant the only profile you could get was of a synthetic op
        // rather than of the thing we are actually trying to make faster.
        cl_ctx.profReport();
        // Per-STAGE wall (temporal_body / depth_body / frame_embed). The kernel profile above says
        // which kernel is hot; this says which part of the AR owns the wall, including the gaps
        // between kernels — and the gap is currently ~45% of it.
        cl_ctx.compReport();
        bench.print_summary(0, opt.cfg.n_frames);
        return 0;
    }

    // ── serve: persistent process, one generation per stdin line ────────────
    // The ~2 GB weight load and the OpenCL JIT are paid once. A warm-up render
    // of a single frame forces the codec blob load and every kernel build so
    // SERVE_READY means "the next request runs at steady-state speed".
    if (!std::getenv("NNOPT_PREWARM") || std::strcmp(std::getenv("NNOPT_PREWARM"), "0") != 0) {
        Options warm = opt;
        // The codec cannot decode fewer than 5 frames (its cascade underflows), and a warm-up
        // that crashes is worse than no warm-up — this is exactly what SIGSEGV'd at frames=1.
        warm.cfg.n_frames = 5;
        warm.out = "warmup.wav";
        SongResult wr; SongConditioning wc;
        // Sweep workgroup sizes during THIS render only — it re-runs kernels to time them, which is
        // safe here because the warm-up's audio is discarded, and unsafe anywhere else.
        OpenCLContext::setLwsTuningWindow(true);
        if (!generate_to_file(cl_ctx, weights, warm, /*sess=*/nullptr, wr, wc))
            std::fprintf(stderr, "SERVE_NOTE warmup_failed=1 (continuing — the first request pays the JIT)\n");
        OpenCLContext::setLwsTuningWindow(false);
        std::remove("warmup.wav");
    }

    SongSession* sess = song_session_create();
    std::fprintf(stderr, "SERVE_READY\n");
    std::fflush(stderr);

    std::string line;
    int idx = 0;
    std::string last_style = opt.style;
    std::string last_prompt = opt.prompt;
    std::vector<int32_t> last_prompt_tokens = opt.style_tokens;
    std::string last_blend = opt.blend;
    std::vector<int32_t> last_blend_tokens;
    while (std::getline(std::cin, line)) {
        if (line.find_first_not_of(" \t\r\n") == std::string::npos) continue;
        if (line == "quit" || line == "exit") break;

        Options req = opt;
        req.cfg.reset = false;
        parse_request(line, req);
        // What the CALLER explicitly asked for, before any conditioning-change policy below.
        const bool explicit_reset = req.cfg.reset;
        bool cond_changed = false;

        if (!req.blend.empty()) {
            // A weighted blend is steering. Re-resolving only when the blend STRING changed keeps a
            // held puck free; when it does change, the per-prompt embeddings are already cached, so
            // the cost is a weighted average plus one RVQ.
            if (req.blend != last_blend) {
                std::vector<std::pair<float, std::string>> pairs;
                if (!parse_blend(req.blend, pairs) ||
                    !resolve_blend_tokens(cl_ctx, pairs, req.style_tokens)) {
                    std::fprintf(stderr, "SERVE_ERROR idx=%d bad_blend=1\n", idx);
                    std::fflush(stderr);
                    ++idx;
                    continue;
                }
                last_blend = req.blend;
                last_blend_tokens = req.style_tokens;
                cond_changed = true;
            } else {
                req.style_tokens = last_blend_tokens;
            }
            last_prompt.clear();   // blend and prompt are alternatives, never both
        } else if (!req.prompt.empty() && req.prompt != last_prompt) {
            // A new prompt is steering: re-embed it (warm, ~150 ms) and treat it like a style change.
            // Re-resolving only when it actually changed is what keeps live steering cheap.
            if (!resolve_prompt_tokens(cl_ctx, req.prompt, req.style_tokens)) {
                std::fprintf(stderr, "SERVE_ERROR idx=%d bad_prompt=1\n", idx);
                std::fflush(stderr);
                ++idx;
                continue;
            }
            last_prompt = req.prompt;
            last_prompt_tokens = req.style_tokens;
            last_blend.clear();
            cond_changed = true;
        } else if (!req.prompt.empty()) {
            req.style_tokens = last_prompt_tokens;
        }
        if (req.style != last_style) { cond_changed = true; last_style = req.style; }

        // MIDI is deliberately absent from `cond_changed`. It changes every chunk by nature, so
        // counting it as a conditioning change would make steer=0 restart the piece on every note
        // — a player would hear a cut, not an instrument. A note-on must always bend.
        //
        // `midi=` is also NOT sticky across requests: a request that omits it renders unconditioned
        // rather than replaying the last chord. That is the safer failure mode for an instrument —
        // if the app stops sending, everything releases instead of a chord hanging forever.
        //
        // steer=1 (the default) keeps the AR attention state across a conditioning change, so the
        // music BENDS toward the new style instead of starting over — which is the whole point of
        // a surface you drag. steer=0 restores the old policy (new conditioning = new piece); an
        // explicit reset=1 from the caller always wins.
        req.cfg.reset = explicit_reset || (cond_changed && !req.steer);
        char wav[64];
        std::snprintf(wav, sizeof(wav), "output_serve_%d.wav", idx);
        req.out = wav;

        SongResult r; SongConditioning cond;
        if (!generate_to_file(cl_ctx, weights, req, sess, r, cond)) {
            std::fprintf(stderr, "SERVE_ERROR idx=%d\n", idx);
            std::fflush(stderr);
            ++idx;
            continue;
        }
        print_bench_line(cl_ctx.device_name(), req, cond, r);
        if (req.codecchunk_T > 0)
            nnopt_codecchunk_check(cl_ctx, weights, cl_ctx.queue(), req.codecchunk_T, req.codecchunk_CK);
        // Same reports the one-shot path prints. On a device farm the app IS the only way in, so a
        // measurement that is only reachable from the CLI is a measurement we never get to take.
        //
        // But they are OFF by default in serve mode. Live playback issues a request every ~2 s and
        // these three dump 20-40 lines EACH TIME, all of it down the same stderr pipe the host reads
        // line by line. When the host cannot drain that fast the 64 KB pipe fills and this process
        // blocks inside fprintf -- mid-render, holding the GPU -- so the app stops receiving
        // SERVE_DONE and live mode wedges with its stats frozen on the last chunk.
        // NNOPT_SERVE_REPORTS=1 restores them for a measurement run.
        // Everything a caller actually needs is already on the single BENCH line above.
        if (const char* sr = std::getenv("NNOPT_SERVE_REPORTS")) {
            if (sr[0] != '0') {
                KernelProfiler::dump_summary();
                cl_ctx.profReport();
                cl_ctx.compReport();
            }
        }
        std::fprintf(stderr,
                     "SERVE_DONE idx=%d wav=%s gen_sec=%.2f ar_sec=%.2f codec_sec=%.2f audio_sec=%.2f rtf=%.2f\n",
                     idx, wav, r.total_s, r.ar_s, r.codec_s, r.audio_s, r.rtf());
        std::fflush(stderr);
        ++idx;
    }

    song_session_destroy(sess);
    bench.mark_end();
    KernelProfiler::dump_summary();
    bench.print_summary(0, opt.cfg.n_frames);
    return 0;
}
