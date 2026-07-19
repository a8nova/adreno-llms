#include <algorithm>
// bonsai_cli — host reference runtime for Bonsai-8B Q1_0.
// Usage (strict positional, invariant #7):
//   bonsai_cli <model.nnb> <tokenizer.json> encode  <prompt>
//   bonsai_cli <model.nnb> <tokenizer.json> gen     <prompt> <n_tokens>
//   bonsai_cli <model.nnb> <tokenizer.json> chat    <user_msg> <n_tokens>
// Env: DUMP_DIR (per-layer dumps), BONSAI_PLAIN_ROPE=1 (YaRN off A/B),
//      BONSAI_THREADS=N.
#include <cstdio>
#include <cstring>
#include <string>
#include <vector>

#include "nnb.h"
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
static void emit_bench(double prefill_s, int n_prompt, double decode_s, int produced) {
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
    DeviceModel model(nnb, ocl, kd ? kd : "kernels");

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

int main(int argc, char** argv) {
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
    DeviceModel model(nnb, ocl, kd ? kd : "kernels");
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
