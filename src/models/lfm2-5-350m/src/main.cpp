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
#include <algorithm>
#include <chrono>
#include <fstream>
#include <dlfcn.h>
#include <CL/cl.h>

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

// cl_qcom_recordable_queues probe — proves we can record the per-token
// decode dispatch sequence once and replay it cheaply, attacking the
// ~16 ms/token host gap. Triggered via NNOPT_RECORD_PROBE=1; exits before
// LLM work. Mirrors qwen2.5-0.5B/src/main.cpp:run_record_probe.
//
// On Razr 2020 / Adreno 620 (driver E031.37.12.07): bit 30
// (CL_QUEUE_RECORDABLE_QCOM = 0x40000000) on clCreateCommandQueue, ALONE
// (NOT combined with CL_QUEUE_PROFILING_ENABLE), creates a queue that
// clNewRecordingQCOM accepts. The probe iterates many candidates so the
// finding remains evidence-based on other devices.
static int run_record_probe(OpenCLContext& cl_ctx) {
    using namespace std::chrono;
    cl_context  ctx = cl_ctx.context();
    cl_command_queue q = cl_ctx.queue();
    cl_int err = CL_SUCCESS;

    typedef void* cl_recording_qcom;
    typedef cl_recording_qcom (CL_API_CALL *clNewRecordingQCOM_fn)(
        cl_command_queue, cl_int*);
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
    cl_command_queue probe_q = nullptr;
    cl_recording_qcom winning_rec = nullptr;
    int win_attempt = -1;
    cl_command_queue live_q = q;

    auto try_recording = [&](cl_command_queue qq, int attempt_id) -> bool {
        cl_int e = 0;
        cl_recording_qcom h = fnNew(qq, &e);
        std::cerr << "  attempt " << attempt_id << ": clNewRecordingQCOM err="
                  << e << " handle=" << h << "\n";
        if (e == CL_SUCCESS && h) {
            probe_q = qq;
            fnEnd(h);
            fnRelease(h);
            win_attempt = attempt_id;
            winning_rec = (cl_recording_qcom)1;
            return true;
        }
        return false;
    };

    {
        cl_command_queue_properties supp = 0;
        clGetDeviceInfo(dev, CL_DEVICE_QUEUE_PROPERTIES, sizeof(supp), &supp, nullptr);
        std::cerr << "  CL_DEVICE_QUEUE_PROPERTIES = 0x" << std::hex << supp << std::dec
                  << " (bit 30=CL_QUEUE_RECORDABLE_QCOM)\n";
    }

    // Direct candidates per Snapdragon Programming Guide §9.1.3 — clCreateCommandQueue.
    constexpr cl_command_queue_properties RECORD_BIT = (cl_command_queue_properties)1 << 30;
    cl_command_queue_properties direct_candidates[] = {
        RECORD_BIT,
        RECORD_BIT | CL_QUEUE_PROFILING_ENABLE,
        RECORD_BIT | CL_QUEUE_OUT_OF_ORDER_EXEC_MODE_ENABLE,
        RECORD_BIT | CL_QUEUE_PROFILING_ENABLE | CL_QUEUE_OUT_OF_ORDER_EXEC_MODE_ENABLE,
    };
    for (size_t i = 0; i < sizeof(direct_candidates)/sizeof(direct_candidates[0]); ++i) {
        if (winning_rec) break;
        cl_int qerr = 0;
        cl_command_queue qq = clCreateCommandQueue(ctx, dev, direct_candidates[i], &qerr);
        std::cerr << "  direct: props=0x" << std::hex << direct_candidates[i] << std::dec
                  << " qerr=" << qerr << " qq=" << qq << "\n";
        if (qerr == CL_SUCCESS && qq) {
            if (!try_recording(qq, 400 + (int)i)) clReleaseCommandQueue(qq);
        }
    }

    if (!winning_rec) {
        std::cerr << "Record: no candidate worked. Aborting.\n";
        return 1;
    }
    std::cerr << "Record: WIN attempt " << win_attempt
              << " (probe_q=" << probe_q << ")\n";

    // Build the probe kernel from the gemv_m1.cl program.
    std::ifstream f("kernels/gemv_m1.cl", std::ios::binary);
    std::string src_text((std::istreambuf_iterator<char>(f)), std::istreambuf_iterator<char>());
    cl_program prog = OpenCLContext::build_cached_program_from_queue(q, src_text, "");
    if (!prog) { std::cerr << "build kernels fail\n"; return 1; }
    cl_kernel k = clCreateKernel(prog, "probe_noop", &err);
    if (err != CL_SUCCESS) { std::cerr << "createKernel fail " << err << "\n"; return 1; }

    cl_mem counter = clCreateBuffer(ctx, CL_MEM_READ_WRITE, sizeof(cl_int), nullptr, &err);
    if (err != CL_SUCCESS) { std::cerr << "alloc counter fail\n"; return 1; }
    int incr = 1;
    clSetKernelArg(k, 0, sizeof(cl_mem), &counter);
    clSetKernelArg(k, 1, sizeof(int),    &incr);

    constexpr int N_DISPATCH_PER_RECORDING = 100;
    constexpr int N_REPLAYS                = 100;
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

    // Baseline: N_BASELINE_DISPATCHES sequential clEnqueueNDRangeKernel on live_q.
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
    std::cerr << "Baseline: " << N_BASELINE_DISPATCHES << " sequential dispatches → "
              << base_ms << " ms, counter=" << base_val << "\n";

    // Recording: capture N_DISPATCH_PER_RECORDING enqueues on probe_q.
    reset_counter();
    cl_int new_err = 0;
    cl_recording_qcom rec = fnNew(probe_q, &new_err);
    if (new_err != CL_SUCCESS || !rec) {
        std::cerr << "Record: clNewRecordingQCOM failed err=" << new_err << "\n";
        return 1;
    }
    for (int i = 0; i < N_DISPATCH_PER_RECORDING; ++i) {
        cl_int e = clEnqueueNDRangeKernel(probe_q, k, 1, nullptr, &gws, &lws, 0, nullptr, nullptr);
        if (e != CL_SUCCESS) {
            std::cerr << "Record: enqueue during record failed at i=" << i << " err=" << e << "\n";
            fnRelease(rec);
            return 1;
        }
    }
    cl_int end_err = fnEnd(rec);
    if (end_err != CL_SUCCESS) {
        std::cerr << "Record: end failed err=" << end_err << "\n";
        fnRelease(rec);
        return 1;
    }

    // Replay: N_REPLAYS × N_DISPATCH_PER_RECORDING dispatches on live_q.
    reset_counter();
    auto t2 = high_resolution_clock::now();
    for (int r = 0; r < N_REPLAYS; ++r) {
        cl_int e = fnEnqueue(live_q, rec,
            0, nullptr, 0, nullptr, 0, nullptr, 0, nullptr,
            0, nullptr, nullptr);
        if (e != CL_SUCCESS) {
            std::cerr << "Record: replay failed at r=" << r << " err=" << e << "\n";
            fnRelease(rec);
            return 1;
        }
    }
    clFinish(live_q);
    auto t3 = high_resolution_clock::now();
    double replay_ms = duration<double, std::milli>(t3 - t2).count();
    int replay_val = read_counter();
    std::cerr << "Replay: " << N_BASELINE_DISPATCHES << " dispatches → "
              << replay_ms << " ms, counter=" << replay_val << "\n";

    if (replay_val != base_val) {
        std::cerr << "Record: COUNTER MISMATCH (replay isn't equivalent)\n";
    } else {
        double speedup = base_ms / replay_ms;
        double per_baseline = base_ms / N_BASELINE_DISPATCHES * 1000.0;
        double per_replay   = replay_ms / N_BASELINE_DISPATCHES * 1000.0;
        std::cerr << "Record: speedup = " << speedup << "× ("
                  << per_baseline << " µs/dispatch baseline → "
                  << per_replay   << " µs/dispatch replay)\n";
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
    bool chat_mode = false;
    bool serve_mode = false;    // --serve: stay resident, stream replies per stdin request (warm model)
    std::string system_prompt;  // optional system turn (chat mode); empty = none
    std::string history_file;   // optional prior-turns file (multi-turn memory); empty = none

    // Parse positional and optional args
    int arg_idx = 2;
    if (arg_idx < argc && argv[arg_idx][0] != '-') {
        max_tokens = std::stoi(argv[arg_idx++]);
    }
    while (arg_idx < argc) {
        std::string flag = argv[arg_idx++];
        if (flag == "--ignore-eos" || flag == "--no-eos") { ignore_eos = true; continue; }
        if (flag == "--chat") { chat_mode = true; continue; }  // bare flag, no value
        if (flag == "--serve") { serve_mode = true; continue; } // bare flag, no value
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

    // Bandwidth probe — exits before LLM work. Toggle with NNOPT_BW_PROBE=1.
    if (const char* bw = std::getenv("NNOPT_BW_PROBE"); bw && bw[0] == '1') {
        return run_bw_probe(cl_ctx);
    }
    // cl_qcom_recordable_queues probe — exits before LLM work. Toggle with NNOPT_RECORD_PROBE=1.
    if (const char* rp = std::getenv("NNOPT_RECORD_PROBE"); rp && rp[0] == '1') {
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
    // NNOPT_QUANT picks the quantized weight bundle and triggers the matching
    // GEMV image-path registration.
    //   int8 → weights/model.int8.bin  (per-row symmetric int8 + fp16 scales)
    //   q4   → weights/model.q4.bin    (block-32 symmetric Q4 + fp16 scales)
    bool nnopt_quant_int8 = false;
    bool nnopt_quant_q4   = false;
    if (const char* q = std::getenv("NNOPT_QUANT")) {
        std::string mode(q);
        if (mode == "int8") {
            nnopt_weights_bin  = "weights/model.int8.bin";
            nnopt_weights_meta = "weights/model.int8.meta.json";
            nnopt_quant_int8 = true;
            std::cerr << "NNOPT_QUANT=int8: using " << nnopt_weights_bin << std::endl;
        } else if (mode == "q4") {
            nnopt_weights_bin  = "weights/model.q4.bin";
            nnopt_weights_meta = "weights/model.q4.meta.json";
            nnopt_quant_q4 = true;
            std::cerr << "NNOPT_QUANT=q4: using " << nnopt_weights_bin << std::endl;
        }
    }
    if (!weights.load(nnopt_weights_bin, nnopt_weights_meta, cl_ctx.context())) {
        std::cerr << "Failed to load weights from " << nnopt_weights_bin << std::endl;
        return 1;
    }
    NNOPT_CHECKPOINT("weights loaded");

    // Walk every quantized tensor + sibling `.scale` and hand both buffers to
    // utils.cpp. From here on, any pytorch_linear() call on these W buffers
    // takes the appropriate quantized image-path GEMV.
    if (nnopt_quant_int8) {
        int registered = 0;
        for (const std::string& name : weights.all_keys()) {
            if (weights.get_dtype(name) != "int8") continue;
            const std::string scale_name = name + ".scale";
            if (!weights.has_tensor(scale_name)) continue;
            const auto shape = weights.get_shape(name);
            if (shape.size() != 2) continue;
            cl_mem W = weights.get_buffer(name);
            cl_mem S = weights.get_buffer(scale_name);
            if (!W || !S) continue;
            if (nnopt_register_int8_weight(W, S, shape[0], shape[1])) {
                registered++;
            }
        }
        std::cerr << "NNOPT_QUANT=int8: registered " << registered << " int8 weights for GEMV fast path" << std::endl;
    }
    if (nnopt_quant_q4) {
        int registered = 0;
        for (const std::string& name : weights.all_keys()) {
            if (weights.get_dtype(name) != "q4_packed") continue;
            const std::string scale_name = name + ".scale";
            if (!weights.has_tensor(scale_name)) continue;
            const auto shape = weights.get_shape(name);
            if (shape.size() != 2) continue;
            // shape is [N, K/2] (packed bytes). The kernel needs the unpacked
            // K, which the scale rows give us: K = scale_cols * 32.
            const auto scale_shape = weights.get_shape(scale_name);
            if (scale_shape.size() != 2) continue;
            const int N = shape[0];
            const int K_unpacked = scale_shape[1] * 32;
            cl_mem W = weights.get_buffer(name);
            cl_mem S = weights.get_buffer(scale_name);
            if (!W || !S) continue;
            if (nnopt_register_q4_weight(W, S, N, K_unpacked)) {
                registered++;
            }
        }
        std::cerr << "NNOPT_QUANT=q4: registered " << registered << " q4 weights for GEMV fast path" << std::endl;
    }

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

    // ── Persistent serve mode (warm model) — port of the qwen --serve REPL ──────
    // OpenCL init + weight upload + kernel compile run ONCE; --serve then loops reading
    // requests from stdin so query 2+ skip the cold load. Protocol (identical to qwen):
    //   "GEN <max> <temp> <top_k> <top_p> <rep_pen> <sys_nbytes> <prompt_nbytes> <use_history>\n"
    //   then system + prompt bytes; use_history=1 ⇒ read prior turns from "history.bin".
    // Reply streams to stdout, ending with a single 0x1E byte; per-reply BENCHMARK lines on stderr.
    if (serve_mode) {
        std::cerr << "SERVE_READY" << std::endl;
        // Cross-request prefix cache: the KV and conv caches persist between requests, so a follow-up
        // whose prompt strictly extends the previous turn only prefills its NEW tokens (warm chat).
        // cached_ids mirrors exactly what's resident in those caches — the prior prompt + reply, minus
        // the final token, which is generated but never fed back so never entered the caches.
        std::vector<int> cached_ids;
        std::string header;
        while (std::getline(std::cin, header)) {
            if (header.empty()) continue;
            int rq_max = 256, rq_topk = 40, rq_usehist = 0;
            float rq_temp = 0.7f, rq_topp = 0.95f, rq_rep = 1.1f;
            unsigned long rq_sysn = 0, rq_promptn = 0;
            if (std::sscanf(header.c_str(), "GEN %d %f %d %f %f %lu %lu %d",
                            &rq_max, &rq_temp, &rq_topk, &rq_topp, &rq_rep,
                            &rq_sysn, &rq_promptn, &rq_usehist) != 8) {
                std::cerr << "serve: bad request header: " << header << std::endl;
                std::cout << '\x1e' << std::flush; continue;
            }
            std::string rq_system(rq_sysn, '\0');
            if (rq_sysn) std::cin.read(&rq_system[0], (std::streamsize)rq_sysn);
            std::string rq_prompt(rq_promptn, '\0');
            if (rq_promptn) std::cin.read(&rq_prompt[0], (std::streamsize)rq_promptn);
            if (!std::cin) break;

            // LFM2.5 ChatML template (BOS=1, IM_START=6, IM_END=7; encode add_bos=false) — same as --chat.
            std::vector<int> ids;
            constexpr int BOS = 1, IM_START = 6, IM_END = 7;
            auto add = [&](const std::string& s){ auto e = tokenizer.encode(s, /*add_bos=*/false); ids.insert(ids.end(), e.begin(), e.end()); };
            ids.push_back(BOS);
            if (!rq_system.empty()) { ids.push_back(IM_START); add("system\n" + rq_system); ids.push_back(IM_END); add("\n"); }
            if (rq_usehist) for (const auto& t : load_history("history.bin")) {
                ids.push_back(IM_START); add(std::string(t.role == 'A' ? "assistant" : "user") + "\n" + t.content); ids.push_back(IM_END); add("\n");
            }
            ids.push_back(IM_START); add("user\n" + rq_prompt); ids.push_back(IM_END); add("\n");
            ids.push_back(IM_START); add("assistant\n");

            SamplerConfig sc;
            sc.temperature = rq_temp; sc.top_k = rq_topk; sc.top_p = rq_topp;
            sc.repetition_penalty = rq_rep; sc.eos_token_id = IM_END;

            std::string emitted = tokenizer.decode(ids);  // delta baseline
            const size_t prompt_toks = ids.size();
            const auto t_req = std::chrono::steady_clock::now();
            std::chrono::steady_clock::time_point t_first; bool got_first = false; int gen_n = 0;
            auto on_token = [&](int32_t, const std::vector<int32_t>& all){
                if (!got_first) { got_first = true; t_first = std::chrono::steady_clock::now(); }
                ++gen_n;
                std::string full = tokenizer.decode(all);
                if (full.size() > emitted.size()) { std::cout << full.substr(emitted.size()) << std::flush; emitted = std::move(full); }
            };
            // Warm follow-up: if this prompt strictly extends what's already cached, prefill only the
            // new suffix (resume at the cached frontier where both KV and conv state are valid). Any
            // divergence — new chat, edited history, tokenizer round-trip mismatch — falls back to a
            // full prefill (n_past=0), which is always correct.
            int n_past = 0;
            if (!cached_ids.empty() && cached_ids.size() < ids.size() &&
                std::equal(cached_ids.begin(), cached_ids.end(), ids.begin())) {
                n_past = (int)cached_ids.size();
            }
            auto out_ids = model.generate(ids, rq_max, sc, on_token, n_past);
            // Mirror the caches: prompt + generated, minus the final (un-fed-back) token.
            cached_ids.assign(out_ids.begin(), out_ids.end());
            if (!cached_ids.empty()) cached_ids.pop_back();
            const auto t_end = std::chrono::steady_clock::now();
            auto secs = [](std::chrono::steady_clock::time_point a, std::chrono::steady_clock::time_point b){ return std::chrono::duration<double>(b - a).count(); };
            const double ttft = got_first ? secs(t_req, t_first) : 0.0;
            const double dec_s = got_first ? secs(t_first, t_end) : 0.0;
            if (ttft > 0.0) { std::cerr << "BENCHMARK time_to_first_token_sec: " << ttft << "\n"; std::cerr << "BENCHMARK prefill_tokens_per_sec: " << (prompt_toks / ttft) << "\n"; }
            if (dec_s > 0.0 && gen_n > 1) std::cerr << "BENCHMARK decode_tokens_per_sec: " << ((gen_n - 1) / dec_s) << "\n";
            std::cerr << std::flush;
            std::cout << '\x1e' << std::flush;  // end-of-reply sentinel
        }
        return 0;
    }

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
        // LFM2.5 ChatML template (LiquidAI/LFM2.5-350M chat_template.jinja). The
        // jinja prepends bos_token, then one ChatML block per turn:
        //   <|startoftext|>[<|im_start|>system\n{system}<|im_end|>\n]
        //   <|im_start|>user\n{prompt}<|im_end|>\n<|im_start|>assistant\n
        // The assistant turn is opened but not closed so the model continues it.
        // Special tokens go in as raw IDs; text segments use encode(..., add_bos=
        // false) so the single leading <|startoftext|> + prefix space aren't
        // repeated per turn. Stop at <|im_end|> (the chat EOS, not <|endoftext|>).
        constexpr int BOS      = 1;  // <|startoftext|>
        constexpr int IM_START = 6;  // <|im_start|>
        constexpr int IM_END   = 7;  // <|im_end|>
        auto append_text = [&](const std::string& s) {
            auto ids = tokenizer.encode(s, /*add_bos=*/false);
            input_ids.insert(input_ids.end(), ids.begin(), ids.end());
        };
        input_ids.push_back(BOS);
        if (!system_prompt.empty()) {
            input_ids.push_back(IM_START); append_text("system\n" + system_prompt);
            input_ids.push_back(IM_END);   append_text("\n");
        }
        // prior turns (multi-turn memory) from --history, between system and the new user turn.
        for (const auto& t : load_history(history_file)) {
            const char* role = (t.role == 'A') ? "assistant" : "user";
            input_ids.push_back(IM_START); append_text(std::string(role) + "\n" + t.content);
            input_ids.push_back(IM_END);   append_text("\n");
        }
        input_ids.push_back(IM_START); append_text("user\n" + prompt);
        input_ids.push_back(IM_END);   append_text("\n");
        input_ids.push_back(IM_START); append_text("assistant\n");
        if (!ignore_eos) sampler_config.eos_token_id = IM_END;
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
        // Don't echo the prompt template in chat mode — only the assistant reply reaches the user.
        emitted_text = tokenizer.decode(input_ids);
        if (!chat_mode) std::cout << emitted_text << std::flush;
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

    // Structured baseline metrics for FinalizePort / README generation.
    bench.print_summary((int)input_ids.size(), gen_tokens);

    // Per-kernel timing breakdown (env NNOPT_KERNEL_PROFILE=1).
    KernelProfiler::dump_summary();

    return 0;
}
