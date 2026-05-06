// Auto-generated inference entry point for LiquidAI/LFM2.5-350M-Base
// Model type: lfm2

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
#include <chrono>
#include <fstream>

// STREAM-style microbenchmark — measures the practical streaming-read
// ceiling on this device for both buffer-cache and texture-cache reads.
// Triggered via NNOPT_BW_PROBE=1, runs and exits before LLM work begins.
// Adapted from qwen2.5-0.5B/.../src/main.cpp:run_bw_probe.
static int run_bw_probe(OpenCLContext& cl_ctx) {
    using namespace std::chrono;
    cl_context  ctx = cl_ctx.context();
    cl_command_queue q = cl_ctx.queue();
    cl_int err = CL_SUCCESS;

    const size_t TOTAL_BYTES   = 256ull * 1024 * 1024;     // 256 MB ≫ L2
    const size_t TOTAL_HALVES  = TOTAL_BYTES / 2;
    const size_t TOTAL_VEC4    = TOTAL_HALVES / 4;          // # fp16x4 elements
    const size_t WG            = 64;
    const size_t WG_COUNT      = 256;
    const size_t TOTAL_THREADS = WG * WG_COUNT;

    cl_mem src = clCreateBuffer(ctx, CL_MEM_READ_ONLY,  TOTAL_BYTES,                    nullptr, &err);
    if (err != CL_SUCCESS) { std::cerr << "src alloc fail " << err << "\n"; return 1; }
    cl_mem dce = clCreateBuffer(ctx, CL_MEM_READ_WRITE, TOTAL_THREADS * sizeof(float),  nullptr, &err);
    if (err != CL_SUCCESS) { std::cerr << "dce alloc fail " << err << "\n"; return 1; }
    {
        const cl_uchar pat[2] = {0x00, 0x3c};   // fp16 1.0 little-endian
        clEnqueueFillBuffer(q, src, pat, 2, 0, TOTAL_BYTES, 0, nullptr, nullptr);
        clFinish(q);
    }

    cl_program prog = cl_ctx.build_program_from_file("kernels/gemv_m1.cl");
    if (!prog) { std::cerr << "build kernels/gemv_m1.cl fail\n"; return 1; }

    cl_kernel k_buf = clCreateKernel(prog, "gemv_stream_buf", &err);
    if (err != CL_SUCCESS) { std::cerr << "createKernel buf " << err << "\n"; return 1; }

    int iters_per_thread = (int)(TOTAL_VEC4 / TOTAL_THREADS);
    clSetKernelArg(k_buf, 0, sizeof(cl_mem), &src);
    clSetKernelArg(k_buf, 1, sizeof(cl_mem), &dce);
    clSetKernelArg(k_buf, 2, sizeof(int),    &iters_per_thread);

    double best_buf_gbs = 0.0;
    for (int trial = 0; trial < 5; ++trial) {
        size_t gws = TOTAL_THREADS, lws = WG;
        clEnqueueNDRangeKernel(q, k_buf, 1, nullptr, &gws, &lws, 0, nullptr, nullptr);
        clFinish(q);
        auto t0 = high_resolution_clock::now();
        clEnqueueNDRangeKernel(q, k_buf, 1, nullptr, &gws, &lws, 0, nullptr, nullptr);
        clFinish(q);
        auto t1 = high_resolution_clock::now();
        double secs = duration<double>(t1 - t0).count();
        double gbs  = (double)TOTAL_BYTES / secs / 1e9;
        if (gbs > best_buf_gbs) best_buf_gbs = gbs;
    }
    std::cerr << "STREAM[buf]: " << TOTAL_BYTES/(1024.0*1024.0) << " MB read "
              << "→ " << best_buf_gbs << " GB/s (best of 5)\n";

    cl_kernel k_img = clCreateKernel(prog, "gemv_stream_img", &err);
    if (err != CL_SUCCESS) { std::cerr << "createKernel img " << err << "\n"; return 1; }

    const int img_w = 4096;
    const int img_h = (int)(TOTAL_VEC4 / img_w);
    cl_image_format fmt = { CL_RGBA, CL_HALF_FLOAT };
    cl_image_desc desc = {};
    desc.image_type   = CL_MEM_OBJECT_IMAGE2D;
    desc.image_width  = img_w;
    desc.image_height = img_h;
    desc.buffer       = src;
    cl_mem img = clCreateImage(ctx, CL_MEM_READ_ONLY, &fmt, &desc, nullptr, &err);
    if (err != CL_SUCCESS) {
        std::cerr << "createImage err=" << err << " (h=" << img_h << ") — image-from-buffer not supported, skipping image probe\n";
        clReleaseKernel(k_img); clReleaseKernel(k_buf); clReleaseProgram(prog);
        clReleaseMemObject(dce); clReleaseMemObject(src);
        return 0;
    }

    int rows_per_wg = img_h / 256;
    if (rows_per_wg < 1) rows_per_wg = 1;
    int row_pixels = img_w;
    clSetKernelArg(k_img, 0, sizeof(cl_mem), &img);
    clSetKernelArg(k_img, 1, sizeof(cl_mem), &dce);
    clSetKernelArg(k_img, 2, sizeof(int),    &rows_per_wg);
    clSetKernelArg(k_img, 3, sizeof(int),    &row_pixels);

    double best_img_gbs = 0.0;
    for (int trial = 0; trial < 5; ++trial) {
        size_t wg_count = (size_t)(img_h / rows_per_wg);
        size_t lws = WG, gws = wg_count * lws;
        clEnqueueNDRangeKernel(q, k_img, 1, nullptr, &gws, &lws, 0, nullptr, nullptr);
        clFinish(q);
        auto t0 = high_resolution_clock::now();
        clEnqueueNDRangeKernel(q, k_img, 1, nullptr, &gws, &lws, 0, nullptr, nullptr);
        clFinish(q);
        auto t1 = high_resolution_clock::now();
        double secs = duration<double>(t1 - t0).count();
        double bytes = (double)img_h * img_w * 8.0;  // 4 fp16 / pixel = 8 B
        double gbs   = bytes / secs / 1e9;
        if (gbs > best_img_gbs) best_img_gbs = gbs;
    }
    std::cerr << "STREAM[img]: " << ((double)img_h*img_w*8.0)/(1024.0*1024.0) << " MB read via image2d "
              << "→ " << best_img_gbs << " GB/s (best of 5)\n";

    // Roofline summary using LFM2.5's per-token weight footprint.
    const double weight_mb = 676.0;
    std::cerr << "\n=== Practical roofline for LFM2.5-350M-Base (676 MB/token fp16) ===\n"
              << "  Buffer-cache ceiling: " << best_buf_gbs << " GB/s → max "
              <<  (best_buf_gbs * 1000.0 / weight_mb) << " tok/s\n"
              << "  Texture-cache ceiling: " << best_img_gbs << " GB/s → max "
              <<  (best_img_gbs * 1000.0 / weight_mb) << " tok/s\n";

    clReleaseMemObject(img);
    clReleaseKernel(k_img);
    clReleaseKernel(k_buf);
    clReleaseProgram(prog);
    clReleaseMemObject(dce);
    clReleaseMemObject(src);
    return 0;
}

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

static void print_usage(const char* prog) {
    std::cerr << "Usage: " << prog
              << " <prompt> [max_tokens]"
              << " [--temperature T] [--top-k K] [--top-p P]"
              << " [--repetition-penalty R] [--seed S]"
              << " [--token-ids <file>]"
              << std::endl;
}

int main(int argc, char* argv[]) {
    nnopt_install_crash_handler();
    NNOPT_CHECKPOINT("main() started");

    // Streaming UX: unbuffer stdout so per-token `<< std::flush` reaches the
    // user immediately even when stdout is a pipe (e.g. `adb shell`).
    std::setvbuf(stdout, nullptr, _IONBF, 0);
    std::cout.setf(std::ios::unitbuf);

    if (argc < 2) {
        print_usage(argv[0]);
        return 1;
    }

    std::string prompt = argv[1];
    int max_tokens = 64;
    SamplerConfig sampler_config;
    std::string token_ids_file;
    bool ignore_eos = false;

    // Parse positional and optional args
    int arg_idx = 2;
    if (arg_idx < argc && argv[arg_idx][0] != '-') {
        max_tokens = std::stoi(argv[arg_idx++]);
    }
    while (arg_idx < argc) {
        std::string flag = argv[arg_idx++];
        if (flag == "--ignore-eos" || flag == "--no-eos") { ignore_eos = true; continue; }
        if (arg_idx >= argc) { print_usage(argv[0]); return 1; }
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
    std::cerr << "Device: " << cl_ctx.device_name() << std::endl;
    NNOPT_CHECKPOINT("OpenCL initialized");

    // Bandwidth probe — exits before LLM work. Toggle with NNOPT_BW_PROBE=1.
    if (const char* bw = std::getenv("NNOPT_BW_PROBE"); bw && bw[0] == '1') {
        return run_bw_probe(cl_ctx);
    }

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
    if (ignore_eos) sampler_config.eos_token_id = -1;
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
                    fprintf(stderr, " %.4f", nnopt_f16_to_f32(raw[i]));
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
    Timer timer;
    timer.start();

    // Print the prompt prefix (decoded back through the tokenizer for
    // round-trip parity), then stream each new token's text as it's
    // produced by Model::generate via the on_token callback. We compute
    // the delta against the running emitted-byte count so multi-byte
    // (UTF-8 / CJK) tokens always print as complete characters — if a
    // partial UTF-8 sequence sits at the tail, decode().size() doesn't
    // grow until the next token completes the codepoint, and the
    // `text.size() > emitted_len` guard keeps us from printing garbage.
    std::string emitted_text;
    if (tokenizer_ok) {
        emitted_text = tokenizer.decode(input_ids);
        std::cout << emitted_text << std::flush;
    }

    // Per-token "Generated token: N" stderr noise was visible when decode
    // was 4 tok/s; at 10+ tok/s each fprintf is a measurable syscall in
    // the critical path. Gate behind NNOPT_DEBUG_LAYERS=1 (default off).
    const bool dbg_token_ids =
        []() { const char* d = std::getenv("NNOPT_DEBUG_LAYERS"); return d && d[0] != '0'; }();

    auto on_token = [&](int32_t new_tok, const std::vector<int32_t>& all_ids) {
        if (dbg_token_ids) {
            fprintf(stderr, "Generated token: %d\n", new_tok);
        }
        if (!tokenizer_ok) return;
        std::string full = tokenizer.decode(all_ids);
        if (full.size() > emitted_text.size()) {
            std::cout << full.substr(emitted_text.size()) << std::flush;
            emitted_text = std::move(full);
        }
    };

    // Prefill clock starts now — excludes tokenize (see comment block above the
    // mark_inference_start() call for the full metric contract).
    bench.mark_prefill_start();
    auto output_ids = model.generate(input_ids, max_tokens, sampler_config, on_token);
    bench.mark_end();
    double elapsed = timer.elapsed_ms();
    int gen_tokens = output_ids.size() - input_ids.size();
    NNOPT_CHECKPOINT("generation complete");

    // End the streaming line. If tokenizer wasn't loaded we never streamed
    // anything; print the raw IDs so callers (e.g. pipeline validators)
    // still get something deterministic on stdout.
    if (tokenizer_ok) {
        std::cout << std::endl;
    } else {
        std::cout << "Token IDs: ";
        for (size_t i = 0; i < output_ids.size(); ++i) {
            if (i > 0) std::cout << " ";
            std::cout << output_ids[i];
        }
        std::cout << std::endl;
    }

    // Stats (human-readable summary — kept for backward compat with old parsers)
    std::cerr << "Generated " << gen_tokens << " tokens in "
              << elapsed << " ms ("
              << (gen_tokens * 1000.0 / elapsed) << " tokens/sec)" << std::endl;

    // Structured baseline metrics for FinalizePort / README generation.
    bench.print_summary((int)input_ids.size(), gen_tokens);

    // Per-kernel timing breakdown (env NNOPT_KERNEL_PROFILE=1).
    KernelProfiler::dump_summary();

    return 0;
}
