// Auto-generated inference entry point for apple/OpenELM-270M
// Model type: openelm

#include "model.h"
#include "tokenizer.h"
#include "sampler.h"
#include "opencl_context.h"
#include "weights.h"
#include "utils.h"
#include "debug_utils.h"
#include "benchmark.h"
#include "kernel_profiler.h"
#include <iostream>
#include <string>
#include <cstring>
#include <cstdio>
#include <cstdint>
#include <vector>

static std::vector<int> load_token_ids_from_file(const std::string& path) {
    std::vector<int> ids;
    FILE* f = fopen(path.c_str(), "rb");
    if (!f) return ids;
    fseek(f, 0, SEEK_END);
    long sz = ftell(f);
    fseek(f, 0, SEEK_SET);
    int count = sz / sizeof(int32_t);
    ids.resize(count);
    fread(ids.data(), sizeof(int32_t), count, f);
    fclose(f);
    return ids;
}

// A prior conversation turn (multi-turn memory), loaded from --history.
struct ChatTurn { char role; std::string content; };  // role: 'U' user, 'A' assistant

// History file format, one record per prior turn:
//   <role 'U'|'A'> <byte-length>\n<content bytes>\n
// Length-delimited so message content needs no escaping. Empty/unreadable ⇒ single-turn.
static std::vector<ChatTurn> load_history(const std::string& path) {
    std::vector<ChatTurn> turns;
    if (path.empty()) return turns;
    FILE* f = fopen(path.c_str(), "rb");
    if (!f) return turns;
    while (true) {
        int role = fgetc(f);
        while (role == '\n' || role == '\r' || role == ' ') role = fgetc(f);  // skip separators
        if (role == EOF) break;
        long len = -1;
        if (fscanf(f, "%ld", &len) != 1 || len < 0) break;
        fgetc(f);  // consume the single '\n' before the content bytes
        std::string content((size_t)len, '\0');
        if (len > 0 && fread(&content[0], 1, (size_t)len, f) != (size_t)len) break;
        turns.push_back({ (char)role, content });
    }
    fclose(f);
    return turns;
}

static void print_usage(const char* prog) {
    std::cerr << "Usage: " << prog
              << " <prompt> [max_tokens]"
              << " [--temperature T] [--top-k K] [--top-p P]"
              << " [--repetition-penalty R] [--seed S]"
              << " [--chat] [--system \"<text>\"]"
              << " [--token-ids <file>]"
              << std::endl;
}

int main(int argc, char* argv[]) {
    nnopt_install_crash_handler();
    NNOPT_CHECKPOINT("main() started");

    // Unbuffered stdout — adb shell line-buffers by default, which means
    // streaming tokens (lambda below) accumulate behind the buffer and the
    // user sees nothing until the run ends. Match the working mamba pattern:
    // stdout is unbuffered so each fflush actually reaches the user.
    std::setvbuf(stdout, nullptr, _IONBF, 0);

    if (argc < 2) {
        print_usage(argv[0]);
        return 1;
    }

    std::string prompt = argv[1];
    int max_tokens = 64;
    SamplerConfig sampler_config;
    std::string token_ids_file;
    bool chat_mode = false;
    std::string system_prompt;  // optional system instruction (chat mode)
    std::string history_file;   // optional prior-turns file (multi-turn memory); empty = none

    // Parse positional and optional args
    int arg_idx = 2;
    if (arg_idx < argc && argv[arg_idx][0] != '-') {
        max_tokens = std::stoi(argv[arg_idx++]);
    }
    while (arg_idx < argc) {
        std::string flag = argv[arg_idx++];
        if (flag == "--chat") { chat_mode = true; continue; }  // bare flag, no value
        if (arg_idx >= argc) { print_usage(argv[0]); return 1; }
        if (flag == "--system") { system_prompt = argv[arg_idx++]; continue; }
        if (flag == "--history") { history_file = argv[arg_idx++]; continue; }
        if (flag == "--temperature")       sampler_config.temperature = std::stof(argv[arg_idx++]);
        else if (flag == "--top-k")        sampler_config.top_k = std::stoi(argv[arg_idx++]);
        else if (flag == "--top-p")        sampler_config.top_p = std::stof(argv[arg_idx++]);
        else if (flag == "--repetition-penalty") sampler_config.repetition_penalty = std::stof(argv[arg_idx++]);
        else if (flag == "--seed")         sampler_config.seed = static_cast<uint32_t>(std::stoul(argv[arg_idx++]));
        else if (flag == "--token-ids")    token_ids_file = argv[arg_idx++];
        else { std::cerr << "Unknown flag: " << flag << std::endl; print_usage(argv[0]); return 1; }
    }

    // Initialize OpenCL
    NNOPT_CHECKPOINT("initializing OpenCL");
    OpenCLContext cl_ctx;
    if (!cl_ctx.initialize()) {
        std::cerr << "Failed to initialize OpenCL" << std::endl;
        return 1;
    }
    std::cerr << "Device: " << cl_ctx.device_description() << std::endl;
    NNOPT_CHECKPOINT("OpenCL initialized");

    // Load tokenizer (still needed for decoding output)
    NNOPT_CHECKPOINT("loading tokenizer");
    Tokenizer tokenizer;
    bool tokenizer_ok = tokenizer.load("weights/tokenizer_vocab.bin");
    if (!tokenizer_ok && token_ids_file.empty()) {
        std::cerr << "Failed to load tokenizer (use --token-ids to bypass)" << std::endl;
        return 1;
    }
    if (tokenizer_ok) {
        sampler_config.eos_token_id = tokenizer.eos_token_id();
    }
    NNOPT_CHECKPOINT("tokenizer loaded");

    // Load weights — fp16 build pulls weights/model.fp16.bin (smaller, half-precision
    // storage); fp32 build uses weights/model.bin. Both written side-by-side by
    // ConvertWeights so they coexist on device and the right one loads by binary suffix.
    NNOPT_CHECKPOINT("loading weights");
    Weights weights;
#ifdef NNOPT_USE_FP16
    const char* nnopt_weights_bin  = "weights/model.fp16.bin";
    const char* nnopt_weights_meta = "weights/model.fp16.meta.json";
#else
    const char* nnopt_weights_bin  = "weights/model.bin";
    const char* nnopt_weights_meta = "weights/model.meta.json";
#endif
    if (!weights.load(nnopt_weights_bin, nnopt_weights_meta, cl_ctx.context())) {
        std::cerr << "Failed to load weights from " << nnopt_weights_bin << std::endl;
        return 1;
    }
    NNOPT_CHECKPOINT("weights loaded");

    // Create model. Constructor stores refs only — kernel builds and per-layer
    // initialize() calls live in Model::initialize() so failures can return
    // false here. NEVER skip the initialize() return check: dispatching a
    // forward() pass on un-initialized layers crashes inside the OpenCL
    // driver with kernel-arg / null cl_mem errors.
    NNOPT_CHECKPOINT("creating model");
    Model model(cl_ctx, weights);
    NNOPT_CHECKPOINT("calling Model::initialize()");
    if (!model.initialize()) {
        NNOPT_ERROR("Model::initialize() FAILED — see prior NNOPT_ERROR for the layer that failed");
        return 1;
    }
    NNOPT_CHECKPOINT("model created");

    // Baseline benchmark instrumentation (emits BENCHMARK <key>: <value>
    // lines on stderr; parsed by runUtils.ts parseInferenceMetrics).
    //   inference_start = BEFORE tokenize  — TTFT numerator (matches vLLM / MLPerf / llama-bench:
    //                     TTFT is inclusive of tokenization, prefill, and first-token sampling;
    //                     exclusive of model load / OpenCL init, which are amortized setup).
    //   prefill_start   = right before model.generate() — bounds the forward-pass on the prompt
    //                     (llama-bench 'pp' convention: pure compute throughput, excludes tokenize).
    //   first_token     = stamped by NNOPT_BENCH_FIRST_TOKEN() inside generate(), immediately
    //                     after the first sampled token is appended — do NOT remove the macro
    //                     call from generate() or prefill/decode split collapses to -1.
    BenchmarkTimer& bench = BenchmarkTimer::instance();
    bench.mark_inference_start();

    // Get input token IDs: from file (bypass tokenizer) or by encoding prompt
    std::vector<int> input_ids;
    if (!token_ids_file.empty()) {
        input_ids = load_token_ids_from_file(token_ids_file);
        if (input_ids.empty()) {
            std::cerr << "Failed to load token IDs from " << token_ids_file << std::endl;
            return 1;
        }
        std::cerr << "Loaded " << input_ids.size() << " token IDs from file (tokenizer bypass)" << std::endl;
    } else if (chat_mode) {
        // OpenELM-270M-Instruct ships NO official chat template — Apple's model
        // card drives the instruct model with plain prompts (Llama-2 tokenizer,
        // add_bos_token=true). We honor that: --chat feeds the prompt as-is
        // (encode() already prepends BOS), optionally prefixing a system
        // instruction. This keeps the flag uniform across the APK's models
        // without inventing a template the 270M model was never tuned on.
        // No official template — concatenate plainly. System first, then prior turns
        // (multi-turn memory via --history) as "role: content" lines, then the new prompt.
        std::string text;
        if (!system_prompt.empty()) text += system_prompt + "\n\n";
        for (const auto& t : load_history(history_file)) {
            text += (t.role == 'A' ? "Assistant: " : "User: ") + t.content + "\n";
        }
        text += (history_file.empty() ? prompt : ("User: " + prompt + "\nAssistant:"));
        input_ids = tokenizer.encode(text);
    } else {
        input_ids = tokenizer.encode(prompt);
        // Dump encode result so FinalizePort can diff C++ tokenizer output
        // against Python's reference_tokens.json.input_ids. Piggybacks on the
        // layer_dumps/ pull plumbing — no extra infrastructure. Written only
        // on the encode path (never with --token-ids).
        {
#ifdef _WIN32
            (void)system("mkdir layer_dumps 2> NUL");
#else
            (void)system("mkdir -p layer_dumps");
#endif
            FILE* tfe = fopen("layer_dumps/tokenizer_encode.json", "w");
            if (tfe) {
                fputs("{\n  \"ids\": [", tfe);
                for (size_t i = 0; i < input_ids.size(); i++) {
                    if (i > 0) fputs(", ", tfe);
                    fprintf(tfe, "%d", input_ids[i]);
                }
                fputs("],\n  \"prompt\": \"", tfe);
                for (char c : prompt) {
                    switch (c) {
                        case '"':  fputs("\\\"", tfe); break;
                        case '\\': fputs("\\\\", tfe); break;
                        case '\n': fputs("\\n", tfe);  break;
                        case '\r': fputs("\\r", tfe);  break;
                        case '\t': fputs("\\t", tfe);  break;
                        default:
                            if ((unsigned char)c < 0x20) fprintf(tfe, "\\u%04x", c);
                            else fputc(c, tfe);
                    }
                }
                fputs("\"\n}\n", tfe);
                fclose(tfe);
            }
        }
    }
    std::cerr << "Input tokens: " << input_ids.size() << " [";
    for (size_t i = 0; i < std::min(input_ids.size(), (size_t)15); i++) {
        if (i > 0) std::cerr << ", ";
        std::cerr << input_ids[i];
    }
    if (input_ids.size() > 15) std::cerr << ", ...";
    std::cerr << "]" << std::endl;
    std::cerr << "Sampling: temp=" << sampler_config.temperature
              << " top_k=" << sampler_config.top_k
              << " top_p=" << sampler_config.top_p
              << " rep_penalty=" << sampler_config.repetition_penalty
              << " eos=" << sampler_config.eos_token_id << std::endl;

#ifdef NNOPT_DEBUG
    // EMBEDDING_VERIFY: host-side wte readback for input_ids[0]. Bypasses the
    // embedding kernel and the LAYER_CHECK readback path, so the printed values
    // come strictly from the weight upload. Compare to
    // reference/layers/embedding_wte_output.bin first 8 floats:
    //   - match  -> bug is in the embedding kernel or LAYER_CHECK dump path
    //   - differ -> bug is in weight selection/upload (wrong meta, wrong dtype,
    //               wrong row stride). Check meta.dtype, model.fp16.bin vs model.bin.
    if (!input_ids.empty()) {
        const char* embed_key_candidates[] = {
            "backbone.embeddings.weight",
            "transformer.wte.weight",
            "model.embed_tokens.weight",
            "embed_tokens.weight",
            "tok_embeddings.weight",
        };
        cl_mem wte_buf = nullptr;
        const char* wte_key = nullptr;
        for (const char* k : embed_key_candidates) {
            wte_buf = weights.get_buffer(k, true);
            if (wte_buf) { wte_key = k; break; }
        }
        if (wte_buf) {
            const int tok = input_ids[0];
            const int n = 8;
            const size_t hsz = (size_t)MODEL_CONFIG::HIDDEN_SIZE;
#ifdef NNOPT_USE_FP16
            const size_t row_off = (size_t)tok * hsz * 2;
            std::vector<uint16_t> raw(n);
            cl_int e = clEnqueueReadBuffer(cl_ctx.queue(), wte_buf, CL_TRUE,
                row_off, (size_t)n * 2, raw.data(), 0, nullptr, nullptr);
            if (e == CL_SUCCESS) {
                fprintf(stderr, "EMBEDDING_VERIFY key=%s token=%d first%d:", wte_key, tok, n);
                for (int i = 0; i < n; i++)
                    fprintf(stderr, " %.4f", _nnopt_dbg_f16_to_f32(raw[i]));
                fprintf(stderr, "\n");
            } else {
                fprintf(stderr, "EMBEDDING_VERIFY: clEnqueueReadBuffer failed err=%d\n", e);
            }
#else
            const size_t row_off = (size_t)tok * hsz * 4;
            std::vector<float> raw(n);
            cl_int e = clEnqueueReadBuffer(cl_ctx.queue(), wte_buf, CL_TRUE,
                row_off, (size_t)n * 4, raw.data(), 0, nullptr, nullptr);
            if (e == CL_SUCCESS) {
                fprintf(stderr, "EMBEDDING_VERIFY key=%s token=%d first%d:", wte_key, tok, n);
                for (int i = 0; i < n; i++) fprintf(stderr, " %.4f", raw[i]);
                fprintf(stderr, "\n");
            } else {
                fprintf(stderr, "EMBEDDING_VERIFY: clEnqueueReadBuffer failed err=%d\n", e);
            }
#endif
            fflush(stderr);
        } else {
            fprintf(stderr, "EMBEDDING_VERIFY: no embedding weight key matched any of the common candidates\n");
            fflush(stderr);
        }
    }
#endif

    // Generate
    NNOPT_CHECKPOINT("starting generation");

    // Prefill clock starts now — excludes tokenize (see comment block above the
    // mark_inference_start() call for the full metric contract).
    bench.mark_prefill_start();

    // Streaming: print each token as soon as it's produced. Disabled by setting
    // NNOPT_NO_STREAM=1 in the run env (benchmark runs — saves ~10 µs/token of
    // host work in the cumulative-decode path). When off, the full text is
    // printed once at the end (legacy behavior).
    //
    // The lambda decodes the cumulative buffer of generated tokens then prints
    // only the suffix — this is the universal pattern that handles both BPE
    // and SentencePiece tokenizers without per-token decoding artifacts (BPE
    // multi-byte sequences, SentencePiece "▁" prefixes, ChatML wrappers).
    const bool stream_enabled = []() {
        const char* e = std::getenv("NNOPT_NO_STREAM");
        return !(e && *e && *e != '0');
    }();
    std::vector<int32_t> stream_buf;
    size_t streamed_chars = 0;
    Model::TokenCallback on_token = nullptr;
    if (tokenizer_ok && stream_enabled) {
        on_token = [&](int32_t t) {
            stream_buf.push_back(t);
            std::string text = tokenizer.decode(stream_buf);
            if (text.size() > streamed_chars) {
                std::fputs(text.c_str() + streamed_chars, stdout);
                std::fflush(stdout);
                streamed_chars = text.size();
            }
        };
    }

    auto output_ids = model.generate(input_ids, max_tokens, sampler_config, on_token);
    if (on_token) {
        std::fputc('\n', stdout);
        std::fflush(stdout);
    }
    bench.mark_end();
    int gen_tokens = output_ids.size() - input_ids.size();
    NNOPT_CHECKPOINT("generation complete");

    // Per-token id stderr trace — gated so benchmark runs (≥10 tok/s) don't
    // pay the per-token syscall cost. Enable with NNOPT_DEBUG_LAYERS=1 when
    // diagnosing token-id mismatches against the reference run.
    {
        const char* env = std::getenv("NNOPT_DEBUG_LAYERS");
        if (env != nullptr && env[0] != '0') {
            for (size_t i = input_ids.size(); i < output_ids.size(); ++i) {
                fprintf(stderr, "Generated token: %d\n", output_ids[i]);
            }
        }
    }

    // Decode and print (skipped if we already streamed).
    if (tokenizer_ok && !on_token) {
        std::string output = tokenizer.decode(output_ids);
        std::cout << output << std::endl;
    } else if (!tokenizer_ok) {
        // No tokenizer -- print raw token IDs
        std::cout << "Token IDs: ";
        for (size_t i = 0; i < output_ids.size(); ++i) {
            if (i > 0) std::cout << " ";
            std::cout << output_ids[i];
        }
        std::cout << std::endl;
    }

    // Structured baseline metrics for FinalizePort / README generation.
    bench.print_summary((int)input_ids.size(), gen_tokens);

    // Per-kernel profile dump — only emits when NNOPT_KERNEL_PROFILE=1.
    KernelProfiler::dump_summary();

    // Clean-exit marker — Infer treats absence of this line in stderr as
    // evidence that the binary was killed mid-run (lowmemorykiller, SIGSEGV,
    // SIGKILL — adb shell exit code does NOT reflect on-device process
    // death). flush(stderr) so it actually reaches the host before exit.
    std::fflush(stderr);
    std::cerr << "NNOPT_EXIT_CLEAN exit_code=0" << std::endl;
    std::fflush(stderr);

    return 0;
}
