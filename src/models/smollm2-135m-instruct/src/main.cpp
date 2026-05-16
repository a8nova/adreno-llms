// Auto-generated inference entry point for HuggingFaceTB/SmolLM2-135M-Instruct
// Model type: llama

#include "model.h"
#include "tokenizer.h"
#include "sampler.h"
#include "opencl_context.h"
#include "weights.h"
#include "utils.h"
#include "debug_utils.h"
#include "benchmark.h"
#include "prof.h"
#include <iostream>
#include <string>
#include <cstring>
#include <cstdio>
#include <cstdint>
#include <vector>
#include <fstream>
#include <sstream>
#include <chrono>
#include <dlfcn.h>

// ── cl_qcom_recordable_queues probe (PDF §9.1.3).
//   Mirrors the LFM2 probe (proven on Adreno 620, driver E031.37.12.07).
//   Confirms record/replay works on this device + measures the per-dispatch
//   speedup vs sequential clEnqueueNDRangeKernel. Triggered by NNOPT_RECORD_PROBE=1.
static int run_record_probe(OpenCLContext& cl_ctx) {
    using namespace std::chrono;
    cl_context  ctx = cl_ctx.context();
    cl_command_queue q = cl_ctx.queue();
    cl_int err = CL_SUCCESS;

    typedef void* cl_recording_qcom;
    typedef cl_recording_qcom (CL_API_CALL *clNewRecordingQCOM_fn)(cl_command_queue, cl_int*);
    typedef cl_int (CL_API_CALL *clEndRecordingQCOM_fn)(cl_recording_qcom);
    typedef cl_int (CL_API_CALL *clReleaseRecordingQCOM_fn)(cl_recording_qcom);
    struct cl_array_arg_qcom {
        cl_kernel    kernel;
        cl_uint      arg_indx;
        size_t       arg_size;
        const void*  arg_value;
    };
    struct cl_array_kernel_exec_info_qcom {
        cl_kernel       kernel;
        cl_uint         indx;
        size_t          param_value_size;
        const void*     param_value;
    };
    typedef cl_int (CL_API_CALL *clEnqueueRecordingQCOM_fn)(
        cl_command_queue queue,
        cl_recording_qcom recording,
        size_t num_args,
        const cl_array_arg_qcom* args,
        size_t num_global_offsets,
        const cl_array_kernel_exec_info_qcom* global_offsets,
        size_t num_global_work_sizes,
        const cl_array_kernel_exec_info_qcom* global_work_sizes,
        size_t num_local_work_sizes,
        const cl_array_kernel_exec_info_qcom* local_work_sizes,
        cl_uint num_events_in_wait_list,
        const cl_event* event_wait_list,
        cl_event* event);

    auto fnNew     = (clNewRecordingQCOM_fn)    dlsym(RTLD_DEFAULT, "clNewRecordingQCOM");
    auto fnEnd     = (clEndRecordingQCOM_fn)    dlsym(RTLD_DEFAULT, "clEndRecordingQCOM");
    auto fnRelease = (clReleaseRecordingQCOM_fn)dlsym(RTLD_DEFAULT, "clReleaseRecordingQCOM");
    auto fnEnqueue = (clEnqueueRecordingQCOM_fn)dlsym(RTLD_DEFAULT, "clEnqueueRecordingQCOM");
    if (!fnNew || !fnEnd || !fnRelease || !fnEnqueue) {
        std::cerr << "Record: missing one or more entry points\n";
        return 1;
    }

    cl_device_id dev = cl_ctx.device();
    constexpr cl_command_queue_properties RECORD_BIT = (cl_command_queue_properties)1 << 30;
    cl_int qerr = 0;
    cl_command_queue probe_q = clCreateCommandQueue(ctx, dev, RECORD_BIT, &qerr);
    if (qerr != CL_SUCCESS || !probe_q) {
        std::cerr << "Record: clCreateCommandQueue(RECORD_BIT alone) failed err=" << qerr << "\n";
        return 1;
    }
    std::cerr << "Record: recordable queue created OK\n";

    cl_command_queue live_q = q;

    // Build probe_noop kernel from block_fused.cl.
    std::ifstream f("kernels/block_fused.cl", std::ios::binary);
    std::string src_text((std::istreambuf_iterator<char>(f)), std::istreambuf_iterator<char>());
    cl_program prog = cl_ctx.build_program(src_text, "");
    if (!prog) { std::cerr << "Record: build kernels fail\n"; clReleaseCommandQueue(probe_q); return 1; }
    cl_kernel k = clCreateKernel(prog, "probe_noop", &err);
    if (err != CL_SUCCESS) { std::cerr << "Record: createKernel fail " << err << "\n"; clReleaseCommandQueue(probe_q); return 1; }

    cl_mem counter = clCreateBuffer(ctx, CL_MEM_READ_WRITE, sizeof(cl_int), nullptr, &err);
    if (err != CL_SUCCESS) { std::cerr << "Record: alloc counter fail\n"; return 1; }
    int incr = 1;
    clSetKernelArg(k, 0, sizeof(cl_mem), &counter);
    clSetKernelArg(k, 1, sizeof(int),    &incr);

    constexpr int N_DISPATCH_PER_RECORDING = 240;  // approx 1 forward_decode worth
    constexpr int N_REPLAYS                = 32;   // approx 1 benchmark worth of decode steps
    constexpr int N_BASELINE_DISPATCHES    = N_DISPATCH_PER_RECORDING * N_REPLAYS;

    auto reset_counter = [&]() {
        cl_int zero = 0;
        clEnqueueWriteBuffer(live_q, counter, CL_TRUE, 0, sizeof(int), &zero, 0, nullptr, nullptr);
    };
    auto read_counter = [&]() {
        cl_int v = 0;
        clEnqueueReadBuffer(live_q, counter, CL_TRUE, 0, sizeof(int), &v, 0, nullptr, nullptr);
        return v;
    };

    reset_counter();
    size_t gws = 1, lws = 1;
    auto t0 = high_resolution_clock::now();
    for (int i = 0; i < N_BASELINE_DISPATCHES; ++i) {
        clEnqueueNDRangeKernel(live_q, k, 1, nullptr, &gws, &lws, 0, nullptr, nullptr);
    }
    clFinish(live_q);
    auto t1 = high_resolution_clock::now();
    double base_ms = duration<double, std::milli>(t1 - t0).count();
    int base_val = read_counter();
    std::cerr << "Baseline: " << N_BASELINE_DISPATCHES << " sequential dispatches -> "
              << base_ms << " ms, counter=" << base_val << "\n";

    reset_counter();
    cl_int new_err = 0;
    cl_recording_qcom rec = fnNew(probe_q, &new_err);
    if (new_err != CL_SUCCESS || !rec) {
        std::cerr << "Record: clNewRecordingQCOM failed err=" << new_err << "\n";
        return 1;
    }
    for (int i = 0; i < N_DISPATCH_PER_RECORDING; ++i) {
        cl_int e = clEnqueueNDRangeKernel(probe_q, k, 1, nullptr, &gws, &lws, 0, nullptr, nullptr);
        if (e != CL_SUCCESS) { std::cerr << "Record: enqueue during record failed err=" << e << "\n"; fnRelease(rec); return 1; }
    }
    cl_int end_err = fnEnd(rec);
    if (end_err != CL_SUCCESS) { std::cerr << "Record: end failed err=" << end_err << "\n"; fnRelease(rec); return 1; }

    reset_counter();
    auto t2 = high_resolution_clock::now();
    for (int r = 0; r < N_REPLAYS; ++r) {
        cl_int e = fnEnqueue(live_q, rec, 0, nullptr, 0, nullptr, 0, nullptr, 0, nullptr,
                             0, nullptr, nullptr);
        if (e != CL_SUCCESS) { std::cerr << "Record: replay failed err=" << e << "\n"; fnRelease(rec); return 1; }
    }
    clFinish(live_q);
    auto t3 = high_resolution_clock::now();
    double replay_ms = duration<double, std::milli>(t3 - t2).count();
    int replay_val = read_counter();
    std::cerr << "Replay: " << N_BASELINE_DISPATCHES << " dispatches -> "
              << replay_ms << " ms, counter=" << replay_val << "\n";

    if (replay_val != base_val) {
        std::cerr << "Record: COUNTER MISMATCH (replay isn't equivalent)\n";
    } else {
        double speedup = base_ms / replay_ms;
        double per_baseline = base_ms / N_BASELINE_DISPATCHES * 1000.0;
        double per_replay   = replay_ms / N_BASELINE_DISPATCHES * 1000.0;
        std::cerr << "Record: speedup = " << speedup << "x ("
                  << per_baseline << " us/dispatch baseline -> "
                  << per_replay   << " us/dispatch replay)\n";
    }

    fnRelease(rec);
    clReleaseMemObject(counter);
    clReleaseKernel(k);
    clReleaseProgram(prog);
    clReleaseCommandQueue(probe_q);
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

    // Parse positional and optional args
    int arg_idx = 2;
    if (arg_idx < argc && argv[arg_idx][0] != '-') {
        max_tokens = std::stoi(argv[arg_idx++]);
    }
    while (arg_idx < argc) {
        std::string flag = argv[arg_idx++];
        if (flag == "--chat") { chat_mode = true; continue; }  // bare flag, no value
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
    std::cerr << "Device: " << cl_ctx.device_description() << std::endl;
    NNOPT_CHECKPOINT("OpenCL initialized");

    // cl_qcom_recordable_queues probe — exits early if NNOPT_RECORD_PROBE=1.
    if (const char* r = std::getenv("NNOPT_RECORD_PROBE"); r && r[0] == '1') {
        return run_record_probe(cl_ctx);
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
    // NNOPT_QUANT=int8 → mixed int8 weights (linear blocks int8, norms/lm_head fp16).
    // Quantize via scripts/quantize_weights.py. Triggers int8 image path in layers
    // when the corresponding weight tensor dtype is "int8".
    if (const char* q = std::getenv("NNOPT_QUANT"); q && std::string(q) == "int8") {
        nnopt_weights_bin  = "weights/model.int8.bin";
        nnopt_weights_meta = "weights/model.int8.meta.json";
        std::cerr << "NNOPT_QUANT=int8: using " << nnopt_weights_bin << std::endl;
    }
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
        NNOPT_ERROR_FMT("Model::initialize() FAILED — see prior NNOPT_ERROR for the layer that failed");
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
        // SmolLM2-Instruct chat template (matches the HF tokenizer's
        // chat_template field). Special tokens are inserted as raw IDs because
        // the BPE encoder treats them as substrings; only the user-visible
        // body and "user"/"assistant" role labels go through encode().
        //   <|im_start|>user
        //   {prompt}<|im_end|>
        //   <|im_start|>assistant
        //
        constexpr int IM_START = 1;  // <|im_start|>
        constexpr int IM_END   = 2;  // <|im_end|>
        auto user_ids = tokenizer.encode("user\n");
        auto body_ids = tokenizer.encode(prompt);
        auto assistant_ids = tokenizer.encode("\nassistant\n");
        input_ids.push_back(IM_START);
        input_ids.insert(input_ids.end(), user_ids.begin(), user_ids.end());
        input_ids.insert(input_ids.end(), body_ids.begin(), body_ids.end());
        input_ids.push_back(IM_END);
        input_ids.push_back(IM_START);
        input_ids.insert(input_ids.end(), assistant_ids.begin(), assistant_ids.end());
        // Stop generation at <|im_end|> (the assistant's turn-end marker).
        sampler_config.eos_token_id = IM_END;
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
    nnopt_prof::reset();  // start fresh — capture only generate() kernel time

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
        // Seed the stream buffer with the prompt so the user sees their input
        // before the first generated token appears. Decoding through the
        // tokenizer (rather than echoing argv[1]) gives round-trip parity and
        // covers the chat-mode path where input_ids include ChatML wrapper
        // tokens that aren't in the raw prompt string.
        stream_buf.assign(input_ids.begin(), input_ids.end());
        std::string text = tokenizer.decode(stream_buf);
        std::fputs(text.c_str(), stdout);
        std::fflush(stdout);
        streamed_chars = text.size();
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
    nnopt_prof::dump(cl_ctx.queue());  // no-op unless NNOPT_PROFILE=1
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

    return 0;
}
