// bonsai_cli — host reference runtime for Bonsai-8B Q1_0.
// Usage (strict positional, invariant #7):
//   bonsai_cli <model.nnb> <tokenizer.json> encode  <prompt>
//   bonsai_cli <model.nnb> <tokenizer.json> gen     <prompt> <n_tokens>
//   bonsai_cli <model.nnb> <tokenizer.json> chat    <user_msg> <n_tokens>
// Env: DUMP_DIR (per-layer dumps), BONSAI_PLAIN_ROPE=1 (YaRN off A/B),
//      BONSAI_THREADS=N.
#include <algorithm>
#include <array>
#include <cmath>
#include <cstdio>
#include <iostream>
#include <memory>
#include <cstring>
#include <string>
#include <vector>

#include "nnb.h"
#ifdef BONSAI_VISION
#include "load_image.h"
#include "vision_model.h"
#endif
#include "reference_model.h"
#include "tokenizer.h"
#ifdef BONSAI_OPENCL
#include "model.h"
#endif

#include <sys/stat.h>
#include <dirent.h>
#include <chrono>

static bool file_exists(const char* p) {
    struct stat st{};
    return stat(p, &st) == 0 && S_ISREG(st.st_mode);
}

// First *.nnb in dir (one binary serves every Bonsai size). Falls back to a
// conventional name so a clear "file not found" surfaces if the dir is empty.
static std::string find_nnb(const char* dir) {
    if (DIR* d = opendir(dir)) {
        std::string hit;
        while (dirent* e = readdir(d)) {
            std::string n = e->d_name;
            if (n.size() > 4 && n.compare(n.size() - 4, 4, ".nnb") == 0) {
                hit = std::string(dir) + "/" + n;
                break;
            }
        }
        closedir(d);
        if (!hit.empty()) return hit;
    }
    return std::string(dir) + "/model.nnb";
}

#ifdef BONSAI_OPENCL
// One-time OpenCL device banner on stderr, in the exact key/value shape the
// Edgi ProcessEngine sniffs (`<key>  <value>`, keys in its OCL_KEYS set).
static void print_ocl_banner(OpenCLContext& ocl) {
    cl_device_id d = ocl.device();
    char buf[8192];
    size_t v; cl_uint u; cl_ulong ul; cl_bool ignore = 0; (void)ignore;
    auto S = [&](cl_device_info k) { buf[0] = 0; clGetDeviceInfo(d, k, sizeof(buf), buf, nullptr); return std::string(buf); };
    auto U = [&](cl_device_info k) { u = 0; clGetDeviceInfo(d, k, sizeof(u), &u, nullptr); return (int)u; };
    auto L = [&](cl_device_info k) { ul = 0; clGetDeviceInfo(d, k, sizeof(ul), &ul, nullptr); return (unsigned long long)ul; };
    (void)v;
    std::string plat;
    { cl_platform_id p; clGetDeviceInfo(d, CL_DEVICE_PLATFORM, sizeof(p), &p, nullptr);
      char pb[512]{}; clGetPlatformInfo(p, CL_PLATFORM_NAME, sizeof(pb), pb, nullptr); plat = pb; }
    std::string ext = S(CL_DEVICE_EXTENSIONS);
    auto has = [&](const char* e) { return ext.find(e) != std::string::npos ? "yes" : "no"; };
    fprintf(stderr, "── OpenCL device ──\n");
    fprintf(stderr, "platform  %s\n", plat.c_str());
    fprintf(stderr, "device  %s\n", S(CL_DEVICE_NAME).c_str());
    fprintf(stderr, "version  %s\n", S(CL_DEVICE_VERSION).c_str());
    fprintf(stderr, "driver  %s\n", S(CL_DRIVER_VERSION).c_str());
    fprintf(stderr, "compute_units  %d\n", U(CL_DEVICE_MAX_COMPUTE_UNITS));
    fprintf(stderr, "max_clock_MHz  %d\n", U(CL_DEVICE_MAX_CLOCK_FREQUENCY));
    fprintf(stderr, "max_workgroup  %d\n", (int)[&]{ size_t w=0; clGetDeviceInfo(d,CL_DEVICE_MAX_WORK_GROUP_SIZE,sizeof(w),&w,nullptr); return w; }());
    fprintf(stderr, "global_mem  %llu MB\n", L(CL_DEVICE_GLOBAL_MEM_SIZE) / (1024ULL*1024ULL));
    fprintf(stderr, "local_mem  %llu KB\n", L(CL_DEVICE_LOCAL_MEM_SIZE) / 1024ULL);
    fprintf(stderr, "cl_khr_fp16  %s\n", has("cl_khr_fp16"));
    fprintf(stderr, "qcom_perf_hint  %s\n", has("cl_qcom_perf_hint"));
    fprintf(stderr, "qcom_recordable_queues  %s\n", has("cl_qcom_recordable_queues"));
    fprintf(stderr, "qcom_dot_product8  %s\n", has("cl_qcom_dot_product8"));
    fprintf(stderr, "qcom_reqd_sub_group_size  %s\n", has("cl_qcom_reqd_sub_group_size"));
    fprintf(stderr, "cl_device_extensions  %s\n", ext.c_str());
    fflush(stderr);
}

// Edgi app subprocess protocol (matches ProcessEngine one-shot chat path):
//   argv: <prompt> <maxTokens> [--chat] [--system <s>] [--history <f>]
//         [--temperature f] [--top-k i] [--top-p f] [--repetition-penalty f]
//   weights from cwd weights/ ; reply tokens -> STDOUT ; BENCHMARK -> stderr.
// (We decode greedily; sampling params are parsed and currently ignored —
// deterministic output, which is exactly what the token-exact port produces.)
// Vision stage timings, filled by the last encode() so emit_bench can report them next to the LM
// numbers. A wall-clock total alone cannot say whether an image turn is slow because of the tower,
// the image prefill, or decode — and those three have completely different fixes.
struct VisBench { double tower_s = 0, upload_s = 0, compute_s = 0; int tokens = 0; bool used = false; };
static VisBench g_vis;
static double g_load_s = 0;   // model load seconds, for /bench

static void emit_bench(double prefill_s, int n_prompt, double decode_s, int produced) {
    if (g_vis.used) {
        fprintf(stderr, "\nBENCHMARK vision_tower_sec: %.3f\n", g_vis.tower_s);
        fprintf(stderr, "BENCHMARK vision_weight_upload_sec: %.3f\n", g_vis.upload_s);
        fprintf(stderr, "BENCHMARK vision_compute_sec: %.3f\n", g_vis.compute_s);
        fprintf(stderr, "BENCHMARK vision_tokens: %d\n", g_vis.tokens);
    }
    fprintf(stderr, "\nBENCHMARK prefill_tokens_per_sec: %.3f\n",
            prefill_s > 0 ? n_prompt / prefill_s : 0.0);
    fprintf(stderr, "BENCHMARK decode_tokens_per_sec: %.3f\n",
            decode_s > 0 ? produced / decode_s : 0.0);
    fprintf(stderr, "BENCHMARK time_to_first_token_sec: %.3f\n", prefill_s);
    FILE* f = fopen("/proc/self/status", "r"); char ln[256];
    while (f && fgets(ln, sizeof(ln), f)) if (!strncmp(ln, "VmHWM:", 6)) {
        long kb = 0; sscanf(ln + 6, "%ld", &kb);
        fprintf(stderr, "BENCHMARK peak_cpu_memory_mb: %.1f\n", kb / 1024.0); break; }
    if (f) fclose(f);
    fflush(stderr);
}

// Wrap a chat template around history + the current user turn. Bonsai is a
// reasoning model: greedy output starts with a <think>…</think> block, which
// the app filters — we keep it raw so nothing is silently dropped here.
static std::string chat_template(const std::string& system,
                                 const std::vector<std::pair<char, std::string>>& hist,
                                 const std::string& user) {
    std::string s;
    if (!system.empty()) s += "<|im_start|>system\n" + system + "<|im_end|>\n";
    for (auto& t : hist)
        s += std::string("<|im_start|>") + (t.first == 'U' ? "user" : "assistant") +
             "\n" + t.second + "<|im_end|>\n";
    s += "<|im_start|>user\n" + user + "<|im_end|>\n<|im_start|>assistant\n";
    // The generation prompt is NOT complete at "<|im_start|>assistant\n" — the
    // model's own Jinja template always appends a think-control suffix, and
    // which one decides whether it reasons:
    //     enable_thinking false -> "<think>\n\n</think>\n\n"   (pre-closed)
    //     otherwise             -> "<think>\n"                 (default)
    // We emit the pre-closed form, i.e. enable_thinking=false.
    //
    // Why non-thinking is the right default HERE: at ~6 tok/s a chain of thought
    // costs the user a minute before a single word of answer appears, and the
    // app hides <think> blocks, so a reply whose budget is consumed mid-thought
    // renders as an EMPTY BUBBLE. Asking "hi" and watching 128 tokens produce
    // nothing is the observed failure. Set BONSAI_THINK=1 for the reasoning form.
    s += getenv("BONSAI_THINK") ? "<think>\n" : "<think>\n\n</think>\n\n";
    return s;
}

static std::vector<std::pair<char, std::string>> read_history() {
    std::vector<std::pair<char, std::string>> h;
    FILE* f = fopen("history.bin", "rb");
    if (!f) return h;
    char role; long n;
    while (fscanf(f, " %c %ld\n", &role, &n) == 2 && n >= 0 && n < (1 << 24)) {
        std::string txt(n, '\0');
        if (fread(&txt[0], 1, n, f) != (size_t)n) break;
        int nl = fgetc(f); (void)nl;
        h.push_back({role == 'A' ? 'A' : 'U', txt});
    }
    fclose(f);
    return h;
}

// Prefill ids, greedy-decode up to n_gen tokens, stream decoded text to
// stdout. Each call resets to pos 0 (independent turn; history is in the
// prompt). Returns nothing; emits BENCHMARK on stderr.
static void run_reply(DeviceModel& model, Tokenizer& tok, const Nnb& nnb,
                      const std::vector<int>& ids, int n_gen) {
    using clk = std::chrono::steady_clock;
    // Independent turn: pos restarts at 0, so the recurrent state must restart too. See
    // DeviceModel::reset_state — the KV cache self-heals here, rec_/convs_ do not.
    model.reset_state();
    auto t0 = clk::now();
    int pos = 0, cur = ids.back();
    for (size_t i = 0; i + 1 < ids.size(); ++i) model.forward(ids[i], pos++, false);
    auto t1 = clk::now();
    int produced = 0;
    for (int i = 0; i < n_gen && pos < DeviceModel::CTX_CAP; ++i) {
        model.forward(cur, pos++, true);
        cur = model.read_token();
        if (cur == nnb.meta.eos) break;
        std::string piece = tok.decode({cur});
        fwrite(piece.data(), 1, piece.size(), stdout);
        fflush(stdout);
        ++produced;
    }
    auto t2 = clk::now();
    emit_bench(std::chrono::duration<double>(t1 - t0).count(), (int)ids.size(),
               std::chrono::duration<double>(t2 - t1).count(), produced);
}


#ifdef BONSAI_VISION
// Edgi's VLM protocol (ProcessEngine::describeImage), byte for byte:
//   spawn:  --interactive --temperature 0 --max-tokens N --image-size N
//   stdin:  optional "/newchat\n"          - start a NEW conversation (drops the KV cache)
//           optional "/hist U|A <nbytes>\n" + raw bytes + "\n"  - replay a prior turn into it
//           "/reset\n"                       - clear the pending image (sent every turn)
//           optional "/image <path>\n"
//           "<prompt>\n"
//   stdout: reply tokens, streamed
//   stderr: "✓ turn" ends the turn
// The prompt arrives collapsed to ONE line, so a single getline per field is correct.
//
// WITHOUT /newchat the process CONTINUES the conversation: turn 2 keeps the KV cache turn 1 built,
// so a follow-up about the same image costs only its own tokens instead of re-encoding the image
// and re-prefilling the transcript. The app decides the boundary because only it knows when the
// user started a new chat or opened an old one — an engine that guessed would eventually answer
// one conversation with another one's context.

// ── conversation session ────────────────────────────────────────────────────
// The VLM process is WARM — it survives across turns — so a follow-up question can CONTINUE from
// the KV cache the previous turn already built instead of re-prefilling the conversation. That is
// worth the bookkeeping below: prefill runs one full 3.6 GB forward pass per token, so re-reading a
// 128-token image plus the transcript costs ~40 s, while continuing costs only the new question.
//
// Four things must be carried across a turn boundary, and none is recoverable from `pos` alone:
//   pos  — next KV slot, and the length of the cache.
//   thw  — the mRoPE (t,h,w) triple of EVERY position so far. Image tokens carry grid row/column
//          instead of a running index, so after one image the text ids permanently stop equalling
//          pos and the table cannot be regenerated from a counter.
//   p    — the next PLAIN-TEXT rope id. After an image it trails pos and never catches up.
//   open — a reply sits in the cache with no terminator: decode stops ON the stop token WITHOUT
//          forwarding it, so the cache ends "...assistant\nHello" and nothing closes the turn. The
//          next turn must emit that <|im_end|> itself or the model reads one run-on assistant turn
//          and answers as if it were still mid-sentence.
struct VlmSession {
    int pos = 0;
    int p = 0;
    std::vector<std::array<int, 3>> thw;
    bool open_reply = false;
    // The stop token was already forwarded by the decode loop's lookahead step, so the next turn
    // must NOT emit <|im_end|> again — the cache would then hold two of them.
    bool eos_forwarded = false;
    std::vector<std::pair<char, std::string>> pending_hist;   // replayed transcript, see /hist
    void clear() { pos = 0; p = 0; thw.clear(); open_reply = false; eos_forwarded = false;
                   pending_hist.clear(); }
};

// Run one turn, appending to the session rather than restarting it.
//
// `emb` (n_img > 0) are vision embeddings the tower already projected into the text model's
// hidden space, so an image token is just a position whose embedding came from pixels instead of
// from token_embd. The turn is laid out as [ text before ][ embeddings ][ text after ], with pos
// advancing continuously across all three and across every previous turn.
static void run_turn(DeviceModel& model, Tokenizer& tok, const Nnb& nnb, VlmSession& s,
                     const std::string& user, const std::vector<float>& emb,
                     int n_img, int hidden, int n_gen, int grid_h, int grid_w) {
    using clk = std::chrono::steady_clock;
    const std::string think = getenv("BONSAI_THINK") ? "<think>\n" : "<think>\n\n</think>\n\n";

    std::vector<int> pre, post;
    auto assemble = [&]() {
        std::string head;
        if (s.open_reply) head += s.eos_forwarded ? "\n" : "<|im_end|>\n";   // close the reply
        for (auto& h : s.pending_hist)
            head += std::string("<|im_start|>") + (h.first == 'U' ? "user" : "assistant") + "\n" +
                    h.second + "<|im_end|>\n";
        head += "<|im_start|>user\n";
        if (n_img) head += "<|vision_start|>";
        const std::string tail = std::string(n_img ? "<|vision_end|>" : "") + user +
                                 "<|im_end|>\n<|im_start|>assistant\n" + think;
        pre = tok.encode(head);
        post = tok.encode(tail);
    };
    assemble();

    // Context is finite (CTX_CAP). When the transcript no longer fits, drop it and keep the CURRENT
    // question — answering the new question without history beats refusing, and silently letting pos
    // run past the cap would corrupt every later turn.
    const int need = (int)pre.size() + n_img + (int)post.size() + n_gen;
    if (s.pos + need > DeviceModel::CTX_CAP) {
        fprintf(stderr, "[vlm] context full (%d used + %d needed > %d) — dropping history\n",
                s.pos, need, DeviceModel::CTX_CAP);
        s.clear();
        model.reset_state();
        assemble();
    }
    s.pending_hist.clear();

    // ── multimodal position ids ────────────────────────────────────────────
    // Text tokens advance t == h == w together. IMAGE tokens share one temporal id and carry their
    // GRID ROW and COLUMN in h and w, which is the entire point of mrope_section [11,11,10]. Text
    // after the image resumes from max(t,h,w)+1, matching Qwen's get_rope_index.
    for (size_t i = 0; i < pre.size(); ++i) { s.thw.push_back({s.p, s.p, s.p}); ++s.p; }
    if (n_img) {
        const int base = s.p;
        for (int i = 0; i < n_img; ++i)
            s.thw.push_back({base, base + i / grid_w, base + i % grid_w});
        s.p = base + std::max(grid_h, grid_w);
    }
    for (size_t i = 0; i < post.size(); ++i) { s.thw.push_back({s.p, s.p, s.p}); ++s.p; }
    const int p_gen = s.p;
    for (int i = 0; i < n_gen; ++i) s.thw.push_back({p_gen + i, p_gen + i, p_gen + i});
    model.set_mrope(s.thw);

    auto t0 = clk::now();
    for (int id : pre) model.forward(id, s.pos++, false);
    const auto tp0 = clk::now();
    // Batched image prefill sweeps the weights once per chunk instead of once per token, but it is
    // OFF by default: it can only be judged on the target part, and on a 1-CU device the shipped
    // split-K GEMV is a 3.3x pessimisation, which inverts the comparison. Measuring against the
    // wrong baseline is exactly how GEMM_MT=16 shipped and made prefill 2x slower.
    if (n_img && model.prefill_batched()) {
        model.forward_batch(emb.data(), n_img, s.pos);
        s.pos += n_img;
    } else {
        for (int i = 0; i < n_img; ++i)
            model.forward_embed(&emb[(size_t)i * hidden], s.pos++, false);
    }
    if (n_img) {
        const double dt = std::chrono::duration<double>(clk::now() - tp0).count();
        fprintf(stderr, "BENCHMARK image_prefill_sec: %.3f (%d tokens, %.1f tok/s)\n",
                dt, n_img, dt > 0 ? n_img / dt : 0.0);
        fflush(stderr);
    }
    // Last token of `post` is the one we generate FROM, so stop one short.
    for (size_t i = 0; i + 1 < post.size(); ++i) model.forward(post[i], s.pos++, false);
    auto t1 = clk::now();

    // Device-fed decode with ONE step of lookahead.
    //
    // The token never round-trips through the host to be consumed: the gather reads it from the
    // buffer argmax wrote, so step i+1 can be enqueued before step i's token has been looked at.
    // The readback then waits on work that is already finished instead of draining the pipeline,
    // which is the whole difference between the bench number and what a reply actually gets.
    int cur = post.back(), produced = 0, steps = 0;
    model.seed_token(cur);
    int slot = -1;
    bool eos_fwd = false;
    for (int i = 0; i < n_gen && s.pos < DeviceModel::CTX_CAP; ++i) {
        if (slot < 0) { slot = model.forward_dev(s.pos++, true); ++steps; }
        // Enqueue the NEXT step first — it needs no host input — so the GPU stays fed across the
        // read below. Not on the last iteration: that one would consume a position and advance the
        // recurrent state with a token nobody asked for.
        int next = -1;
        if (i + 1 < n_gen && s.pos < DeviceModel::CTX_CAP) {
            next = model.forward_dev(s.pos++, true);
            ++steps;
        }
        cur = model.read_token_slot(slot);
        if (cur == nnb.meta.eos) {
            // The lookahead step already forwarded this token — and this token is <|im_end|>, which
            // is exactly what the next turn would otherwise have to prepend. So the speculative work
            // is not wasted and the state is not polluted; record it so the turn is not closed twice.
            eos_fwd = (next >= 0);
            break;
        }
        std::string piece = tok.decode({cur});
        fwrite(piece.data(), 1, piece.size(), stdout);
        fflush(stdout);
        ++produced;
        slot = next;
        if (slot < 0) break;
    }
    auto t2 = clk::now();

    // Generated token j occupied speculative slot j-1 (slot 0 belongs to post.back()), so the next
    // free plain-text id is p_gen + steps - 1. Trim the speculative tail that decode never reached.
    s.p = p_gen + (steps > 0 ? steps - 1 : 0);
    s.thw.resize(s.pos);
    s.open_reply = true;
    s.eos_forwarded = eos_fwd;

    emit_bench(std::chrono::duration<double>(t1 - t0).count(),
               (int)(pre.size() + n_img + post.size()),
               std::chrono::duration<double>(t2 - t1).count(), produced);
}

static int vlm_main(int argc, char** argv, Tokenizer& tok, const Nnb& nnb,
                    OpenCLContext& ocl, DeviceModel& model, const std::string& kdir) {
    int n_gen = 256;
    for (int i = 1; i < argc; ++i)
        if (!strcmp(argv[i], "--max-tokens") && i + 1 < argc) n_gen = atoi(argv[++i]);

    // The vision tower is loaded lazily, on the first image. A text-only turn in the VLM task must
    // not pay 630 MB and ~20 s for a tower it never uses.
    std::unique_ptr<VisionModel> vis;
    std::string vis_path = "weights/bonsai27b-vision.nnb";

    fprintf(stderr, "VLM_READY\n"); fflush(stderr);
    VlmSession sess;
    std::string line, image_path, prompt;
    while (std::getline(std::cin, line)) {
        if (line.rfind("/newchat", 0) == 0) {
            sess.clear();
            model.reset_state();
            image_path.clear();
            fprintf(stderr, "[vlm] new conversation\n"); fflush(stderr);
            continue;
        }
        // A prior turn to replay, so reopening a saved chat keeps its context. Length-prefixed
        // rather than one line per turn, because a stored reply contains newlines and a getline
        // protocol would silently truncate it at the first one.
        if (line.rfind("/hist ", 0) == 0) {
            char role = 'U'; long n = 0;
            if (sscanf(line.c_str() + 6, " %c %ld", &role, &n) == 2 && n >= 0 && n < (1 << 22)) {
                std::string txt((size_t)n, '\0');
                std::cin.read(&txt[0], n);
                std::cin.get();   // trailing newline
                sess.pending_hist.push_back({role == 'A' ? 'A' : 'U', txt});
            }
            continue;
        }
        if (line.rfind("/reset", 0) == 0) { image_path.clear(); continue; }
        if (line.rfind("/image ", 0) == 0) { image_path = line.substr(7); continue; }
        prompt = line;
        // One line per turn saying whether an image is in play. Without this, a text-only answer
        // is ambiguous between "the app sent no image" and "the tower failed".
        fprintf(stderr, "[vlm] turn: image=%s prompt=%zu chars\n",
                image_path.empty() ? "NONE" : image_path.c_str(), prompt.size());
        fflush(stderr);

        std::vector<float> vis_emb;
        int gh = 0, gw = 0;
        if (!image_path.empty()) {
            if (!vis) {
                try {
                    const auto tv0 = std::chrono::steady_clock::now();
                    vis.reset(new VisionModel(vis_path, ocl, kdir));
                    fprintf(stderr, "BENCHMARK vision_tower_load_sec: %.3f\n",
                            std::chrono::duration<double>(
                                std::chrono::steady_clock::now() - tv0).count());
                    fflush(stderr);
                } catch (const std::exception& e) {
                    fprintf(stderr, "vision unavailable: %s\n", e.what());
                    char m[512];
                    snprintf(m, sizeof(m), "[vision engine could not start: %s]", e.what());
                    fwrite(m, 1, strlen(m), stdout); fflush(stdout);
                    fprintf(stderr, "\xE2\x9C\x93 turn\n"); fflush(stderr);
                    continue;
                }
            }
            ImageBufferU8 img;
            if (!load_image_rgb_u8(image_path, img)) {
                fprintf(stderr, "image load failed: %s\n", img.error.c_str());
                char m[512];
                snprintf(m, sizeof(m), "[could not read the image: %s]", img.error.c_str());
                fwrite(m, 1, strlen(m), stdout); fflush(stdout);
                fprintf(stderr, "\xE2\x9C\x93 turn\n"); fflush(stderr);
                continue;
            }
            vis->encode(img.data.data(), img.H, img.W, &vis_emb, &gh, &gw);
            {
                const VisionModel::Timing& t = vis->timing();
                g_vis.tower_s = t.total; g_vis.upload_s = t.upload;
                g_vis.compute_s = t.compute; g_vis.tokens = gh * gw; g_vis.used = true;
            }
            // NaN/inf guard. The 'L' blocks carry recurrent state across the WHOLE sequence, so a
            // single bad value in the image embeddings poisons every token after it — every logit
            // becomes NaN and argmax returns token 0, which prints as an endless "!". That is what a
            // position-embedding buffer overrun produced, and it was invisible until the output was
            // read. Check here, where it is cheap and unambiguous.
            // NaN test by BIT PATTERN, not std::isfinite: this TU is built with -ffast-math, which
            // lets the compiler assume NaNs cannot occur and folds isfinite() to constant true. The
            // first version of this guard was silently compiled away — it reported "0 non-finite"
            // over an array that was entirely NaN, and printed lo/hi still at their initial values
            // because every NaN comparison is false. Bit inspection cannot be optimised out.
            auto nonfinite = [](float v) {
                uint32_t b; std::memcpy(&b, &v, 4);
                return ((b >> 23) & 0xFF) == 0xFF;          // exponent all ones => inf or nan
            };
            size_t bad = 0;
            double lo = 1e30, hi = -1e30;
            for (float v : vis_emb) {
                if (nonfinite(v)) { ++bad; continue; }
                lo = std::min(lo, (double)v); hi = std::max(hi, (double)v);
            }
            if (bad == vis_emb.size()) { lo = hi = 0.0; }   // don't print the sentinels as data
            fprintf(stderr, "[vis] %dx%d -> %d image tokens, emb range [%.3f, %.3f]%s\n",
                    img.W, img.H, gh * gw, lo, hi,
                    bad ? "  ** NON-FINITE VALUES **" : "");
            fflush(stderr);
            if (bad) {
                // Do NOT quietly answer text-only. The model then says "I'm a text-based AI" with
                // full confidence, which is indistinguishable from a correct answer and hides a
                // real engine failure. Report it as the engine's failure, not the model's opinion.
                fprintf(stderr, "vision produced %zu/%zu non-finite values\n", bad, vis_emb.size());
                const char* msg = "[vision engine error: the image encoder produced invalid values, "
                                  "so this answer would not be about your image]";
                fwrite(msg, 1, strlen(msg), stdout);
                fflush(stdout);
                fprintf(stderr, "\xE2\x9C\x93 turn\n"); fflush(stderr);
                image_path.clear();
                continue;
            }
        }

        run_turn(model, tok, nnb, sess, prompt, vis_emb, vis_emb.empty() ? 0 : gh * gw,
                 nnb.meta.hidden, n_gen, gh, gw);
        // The image belongs to the turn that carried it, not to the session: it is already in the KV
        // cache, so keeping it pending would re-encode the tower AND duplicate it in context on the
        // next question. Follow-ups see it through the cache.
        image_path.clear();
        fprintf(stderr, "\xE2\x9C\x93 turn\n"); fflush(stderr);
    }
    return 0;
}
#endif  // BONSAI_VISION

static int edgi_main(int argc, char** argv) {
    std::string prompt = argv[1];
    int n_gen = atoi(argv[2]);
    if (n_gen <= 0) n_gen = 128;
    bool chat = false, serve = false;
    std::string system;
    for (int i = 3; i < argc; ++i) {
        std::string a = argv[i];
        if (a == "--chat") chat = true;
        else if (a == "--serve") serve = true;
        else if (a == "--system" && i + 1 < argc) system = argv[++i];
        else if ((a == "--temperature" || a == "--top-k" || a == "--top-p" ||
                  a == "--repetition-penalty" || a == "--history") && i + 1 < argc) ++i;
    }
    Tokenizer tok;
    tok.load("weights/tokenizer.json");
    // One binary serves every Bonsai size (4B/8B/…): find the .nnb in weights/
    // rather than hardcoding a per-model filename.
    Nnb nnb(find_nnb("weights").c_str());
    OpenCLContext ocl;
    if (!ocl.initialize()) { fprintf(stderr, "FATAL: no OpenCL device\n"); return 4; }
    print_ocl_banner(ocl);
    const char* kd = getenv("BONSAI_KERNEL_DIR");
    // Time the load. The app reports one "warming up" number covering spawn + this + the vision
    // tower + image prefill + the first decode step, and those have four different fixes — without
    // this line the log cannot say which of them the user actually waited on.
    const auto t_load0 = std::chrono::steady_clock::now();
    DeviceModel model(nnb, ocl, kd ? kd : "kernels");
    g_load_s = std::chrono::duration<double>(std::chrono::steady_clock::now() - t_load0).count();
    fprintf(stderr, "BENCHMARK model_load_sec: %.3f\n", g_load_s);
    fflush(stderr);

#ifdef BONSAI_VISION
    for (int i = 1; i < argc; ++i)
        if (!strcmp(argv[i], "--interactive"))
            return vlm_main(argc, argv, tok, nnb, ocl, model, kd ? kd : "kernels");
#endif
    if (serve) {
        // Warm REPL (ProcessEngine tryGenerateWarm): stay resident across
        // queries so message 2+ skips the cold load. Request framing:
        //   header: "GEN <maxTok> <temp> <topK> <topP> <repPen> <sysBytes> <prBytes> <useHist>\n"
        //   then raw system bytes, then raw prompt bytes.
        // Reply: stream tokens to stdout, terminate with the 0x1E sentinel.
        fprintf(stderr, "SERVE_READY\n"); fflush(stderr);
        for (;;) {
            int maxTok, topK, useHist, sysN, prN;
            float temp, topP, rep;
            char verb[16];
            int m = scanf("%15s %d %f %d %f %f %d %d %d", verb, &maxTok, &temp,
                          &topK, &topP, &rep, &sysN, &prN, &useHist);
            if (m != 9 || strcmp(verb, "GEN") != 0) return 0;   // EOF / bad frame → exit
            getchar();   // consume the header newline
            std::string sys(sysN, '\0'), pr(prN, '\0');
            if (sysN && fread(&sys[0], 1, sysN, stdin) != (size_t)sysN) return 0;
            if (prN && fread(&pr[0], 1, prN, stdin) != (size_t)prN) return 0;
            auto hist = useHist ? read_history()
                                : std::vector<std::pair<char, std::string>>{};
            std::vector<int> ids = tok.encode(chat_template(sys, hist, pr));
            run_reply(model, tok, nnb, ids, maxTok > 0 ? maxTok : n_gen);
            fputc(0x1E, stdout); fflush(stdout);   // reply-complete sentinel
        }
    }

    // One-shot path.
    auto hist = read_history();
    std::string full = chat ? chat_template(system, hist, prompt) : prompt;
    std::vector<int> ids = tok.encode(full);
    run_reply(model, tok, nnb, ids, n_gen);
    return 0;
}
#endif


#ifdef BONSAI_VISION
// Standalone vision-tower check: loads ONLY the vision .nnb, runs one image, dumps the merged
// embeddings. Deliberately independent of the 3.6 GB text model so it runs on ANY Adreno — the
// tower is 922 MB in fp16 and streams a block at a time, so a Razr (Adreno 620, 3.7 GB) can host it
// even though the full 27B cannot. This is the differential harness the text port used to land
// token-exact, applied to vision: compare this dump against the numpy reference, op by op.
//   usage: bonsai27b_inference --vision-check <vision.nnb> <image> <out.bin> [kernels_dir]
static int vision_check(int argc, char** argv) {
    const char* nnb = argv[2];
    const char* imgp = argv[3];
    const char* outp = argv[4];
    const char* kdir = argc > 5 ? argv[5] : "kernels";
    OpenCLContext ocl;
    if (!ocl.initialize()) { fprintf(stderr, "FATAL: no OpenCL device\n"); return 4; }
    print_ocl_banner(ocl);
    VisionModel vis(nnb, ocl, kdir);
    ImageBufferU8 img;
    if (!load_image_rgb_u8(imgp, img)) { fprintf(stderr, "image: %s\n", img.error.c_str()); return 2; }
    std::vector<float> emb; int gh = 0, gw = 0;
    vis.encode(img.data.data(), img.H, img.W, &emb, &gh, &gw);
    FILE* f = fopen(outp, "wb");
    if (!f) { fprintf(stderr, "cannot write %s\n", outp); return 2; }
    const int32_t hdr[3] = {gh, gw, (int32_t)vis.cfg().out_hidden};
    fwrite(hdr, 4, 3, f);
    fwrite(emb.data(), 4, emb.size(), f);
    fclose(f);
    size_t bad = 0; double lo = 1e30, hi = -1e30;
    for (float v : emb) {
        uint32_t b; memcpy(&b, &v, 4);
        if (((b >> 23) & 0xFF) == 0xFF) { ++bad; continue; }
        lo = std::min(lo, (double)v); hi = std::max(hi, (double)v);
    }
    fprintf(stderr, "[check] grid %dx%d  tokens %d  dim %d  nonfinite %zu  range [%.5g, %.5g]\n",
            gh, gw, gh * gw, vis.cfg().out_hidden, bad, bad == emb.size() ? 0.0 : lo,
            bad == emb.size() ? 0.0 : hi);
    return 0;
}
#endif



int main(int argc, char** argv) {
#ifdef BONSAI_VISION
    // Before anything else: this path must NOT construct the text model.
    if (argc >= 5 && !strcmp(argv[1], "--vision-check")) return vision_check(argc, argv);
#endif
#ifdef BONSAI_OPENCL
    // Edgi subprocess mode: argv[1] is a PROMPT (not a readable .nnb path).
    if (argc >= 3 && !file_exists(argv[1])) return edgi_main(argc, argv);
#endif
    if (argc < 5) {
        fprintf(stderr, "usage: %s <model.nnb> <tokenizer.json> "
                        "encode|gen|chat <prompt> [n_tokens]\n", argv[0]);
        return 2;
    }
    const std::string mode = argv[3];
    std::string prompt = argv[4];
    // @file prompt transport: shell quoting mangles newlines/specials, and
    // $(...) strips trailing newlines — files are byte-exact.
    if (!prompt.empty() && prompt[0] == '@') {
        FILE* pf = fopen(prompt.c_str() + 1, "rb");
        if (!pf) { fprintf(stderr, "FATAL: cannot open prompt file %s\n", prompt.c_str() + 1); return 2; }
        std::string s; char buf[4096]; size_t n;
        while ((n = fread(buf, 1, sizeof(buf), pf)) > 0) s.append(buf, n);
        fclose(pf);
        prompt = s;
    }

    Tokenizer tok;
    tok.load(argv[2]);

    if (mode == "chat")
        prompt = "<|im_start|>user\n" + prompt + "<|im_end|>\n<|im_start|>assistant\n";

    std::vector<int> ids = tok.encode(prompt);
    printf("prompt_ids:");
    for (int id : ids) printf(" %d", id);
    printf("\n");
    if (mode == "encode") return 0;

    const int n_gen = argc > 5 ? atoi(argv[5]) : 64;
    Nnb nnb(argv[1]);
    std::vector<int> out_ids;
    std::string text;
    int pos = 0, cur = ids.back();
#ifdef BONSAI_OPENCL
    OpenCLContext ocl;
    if (!ocl.initialize()) { fprintf(stderr, "FATAL: no OpenCL device\n"); return 4; }
    fprintf(stderr, "opencl: %s\n", ocl.device_name().c_str());
    const char* kd = getenv("BONSAI_KERNEL_DIR");
    // Time the load. The app reports one "warming up" number covering spawn + this + the vision
    // tower + image prefill + the first decode step, and those have four different fixes — without
    // this line the log cannot say which of them the user actually waited on.
    const auto t_load0 = std::chrono::steady_clock::now();
    DeviceModel model(nnb, ocl, kd ? kd : "kernels");
    g_load_s = std::chrono::duration<double>(std::chrono::steady_clock::now() - t_load0).count();
    fprintf(stderr, "BENCHMARK model_load_sec: %.3f\n", g_load_s);
    fflush(stderr);
    // NOTE: batched prefill (M=8 GEMV) was built and measured a REGRESSION
    // (1.5 vs 0.51 s/tok): the 1-bit GEMV is ISSUE-bound, not memory-bound,
    // so batching (which only amortizes weight streaming) does the same
    // issue-bound op count with worse register pressure. A true GEMM tile
    // would win but hits the Adreno register cliff. Sequential prefill:
    for (size_t i = 0; i + 1 < ids.size(); ++i) model.forward(ids[i], pos++, false);
    // NOTE: enqueue-ahead with a lagged argmax read was tried and measured
    // a NO-OP — the token chain is GPU-serial (step i+1's gather needs
    // step i's argmax), so there is no cross-token overlap to reclaim.
    auto _t_dec = std::chrono::steady_clock::now();
    for (int i = 0; i < n_gen; ++i) {
        model.forward(cur, pos++, true);
        cur = model.read_token();
#else
    Model model(nnb);
    for (size_t i = 0; i + 1 < ids.size(); ++i) model.forward(ids[i], pos++);
    for (int i = 0; i < n_gen; ++i) {
        model.forward(cur, pos++);
        cur = model.argmax_logits();
        if (getenv("BONSAI_TOPK")) model.print_topk();
#endif
        if (cur == nnb.meta.eos) { fprintf(stderr, "[eos]\n"); break; }
        out_ids.push_back(cur);
        std::string piece = tok.decode({cur});
        text += piece;
        fprintf(stderr, "[tok %02d] id=%-6d '%s'\n", i, cur, piece.c_str());
        fflush(stderr);
    }
    fprintf(stderr, "\n");
#ifdef BONSAI_OPENCL
    {
        double s = std::chrono::duration<double>(
                       std::chrono::steady_clock::now() - _t_dec).count();
        int n = (int)out_ids.size();
        fprintf(stderr, "DECODE %.3f tok/s  (%.3f s/tok over %d tokens)\n",
                n > 0 ? n / s : 0.0, n > 0 ? s / n : 0.0, n);
    }
#endif
    printf("output_ids:");
    for (int id : out_ids) printf(" %d", id);
    printf("\noutput_text: %s\n", text.c_str());
    return 0;
}
