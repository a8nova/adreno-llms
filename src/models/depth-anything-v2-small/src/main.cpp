// main.cpp for depth-anything/Depth-Anything-V2-Small-hf.
// Modality: VISION depth estimation. Input = pixel_values [1,3,H,W] fp32
// (assets/test_pixel_values.bin). Output = predicted_depth [1,H,W] fp32.
//
// This is NOT a text model — there is no tokenizer, no sampler, no
// autoregressive loop. We load the preprocessed image tensor, run a single
// forward through the DINOv2 backbone -> DPT neck -> depth head, and write
// the resulting depth map to depth_out.bin (and print summary stats).

#include "model.h"
#include "model_config.h"
#include "opencl_context.h"
#include "weights.h"
#include "debug_utils.h"
#include "benchmark.h"
#include "profiler.h"
#include <chrono>   // KernelProfiler::dump_summary — dormant unless NNOPT_PROFILE=1

#include <fstream>
#include <iostream>
#include <string>
#include <vector>
#include <cstdint>
#include <cstring>
#include <cmath>
#include <cstdlib>    // getenv/atoi — NNOPT_REPEAT frame loop
#include <algorithm>

// Read a little-endian float32 binary file into a vector<float>.
static bool read_f32_bin(const std::string& path, std::vector<float>& out) {
    std::ifstream f(path, std::ios::binary);
    if (!f) return false;
    f.seekg(0, std::ios::end);
    const std::streamsize n_bytes = f.tellg();
    f.seekg(0, std::ios::beg);
    if (n_bytes <= 0 || (n_bytes % sizeof(float)) != 0) return false;
    out.resize(static_cast<size_t>(n_bytes) / sizeof(float));
    f.read(reinterpret_cast<char*>(out.data()), n_bytes);
    return f.good() || f.eof();
}

int main(int argc, char** argv) {
    nnopt_install_crash_handler();

    // Optional positional args are ignored (kept so run_android.sh's
    // "<prompt> <max_tokens>" invocation doesn't break). Real input is the
    // pixel_values fixture on disk.
    std::string pixel_path = "assets/test_pixel_values.bin";
    for (int i = 1; i < argc; i++) {
        const std::string a = argv[i];
        if (a == "--pixel-values" && i + 1 < argc) {
            pixel_path = argv[++i];
        }
    }

    BenchmarkTimer& bench = BenchmarkTimer::instance();
    bench.mark_inference_start();

    OpenCLContext cl_ctx;
    if (!cl_ctx.initialize()) {
        NNOPT_ERROR("OpenCL init failed");
        return 1;
    }

    // ── Load pixel_values fixture ──
    std::vector<float> pixel_values;
    if (!read_f32_bin(pixel_path, pixel_values)) {
        NNOPT_ERROR_FMT("failed to load pixel_values from %s", pixel_path.c_str());
        return 1;
    }
    const int C = MODEL_CONFIG::NUM_CHANNELS;
    // Infer H, W from element count: C*H*W. The fixture is [1,3,518,686].
    const int64_t total = (int64_t)pixel_values.size();
    if (total % C != 0) {
        NNOPT_ERROR_FMT("pixel_values size %lld not divisible by channels %d",
                        (long long)total, C);
        return 1;
    }
    // Known geometry from io_contract: 518 x 686.
    int H = 518, W = 686;
    if ((int64_t)C * H * W != total) {
        NNOPT_ERROR_FMT("pixel_values size %lld != %d*%d*%d", (long long)total, C, H, W);
        return 1;
    }
    std::cerr << "[main] loaded pixel_values " << pixel_path
              << " (" << C << "x" << H << "x" << W << ", " << total << " floats)" << std::endl;

    // ── Weights ──
    auto _t0 = std::chrono::steady_clock::now();
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

    auto _t1 = std::chrono::steady_clock::now();
    fprintf(stderr, "PHASE weights_load_sec: %.3f\n", std::chrono::duration<double>(_t1 - _t0).count());
    Model model(cl_ctx, weights);
    if (!model.initialize()) {
        NNOPT_ERROR("Model::initialize() failed");
        return 1;
    }

    bench.mark_prefill_start();
    NNOPT_BENCH_FIRST_TOKEN();  // stamps TTFT — single-pass model, first (only) forward

    // NNOPT_REPEAT=N runs the forward N times in-process. Frame 1 pays the
    // one-time costs (CLBlast per-shape kernel compiles, pool warm-up);
    // frames 2+ are the steady-state per-frame cost. Depth map from the
    // last frame is what gets written/validated.
    int repeat = 1;
    if (const char* r = std::getenv("NNOPT_REPEAT")) {
        repeat = std::max(1, atoi(r));
    }
    std::vector<float> depth;
    for (int frame = 1; frame <= repeat; frame++) {
        auto _t2 = std::chrono::steady_clock::now();
        depth = model.forward_depth(pixel_values, C, H, W);
        auto _t3 = std::chrono::steady_clock::now();
        fprintf(stderr, "PHASE forward_sec[frame %d/%d]: %.3f\n", frame, repeat,
                std::chrono::duration<double>(_t3 - _t2).count());
    }

    bench.mark_end();

    if (depth.empty()) {
        NNOPT_ERROR("forward_depth returned empty depth map");
        return 1;
    }

    // ── Write depth map ──
    {
        std::ofstream of("depth_out.bin", std::ios::binary);
        of.write(reinterpret_cast<const char*>(depth.data()),
                 (std::streamsize)(depth.size() * sizeof(float)));
    }

    // ── Summary stats to stdout so Evaluate has a signal ──
    double mn = 1e30, mx = -1e30, sum = 0.0;
    int nan_ct = 0;
    for (float v : depth) {
        if (std::isnan(v) || std::isinf(v)) { nan_ct++; continue; }
        if (v < mn) mn = v;
        if (v > mx) mx = v;
        sum += v;
    }
    const double mean = sum / (double)depth.size();
    std::cout << "DEPTH_MAP: elems=" << depth.size()
              << " min=" << mn << " max=" << mx << " mean=" << mean
              << " nan_inf=" << nan_ct << std::endl;
    std::cout << "GENERATED_TEXT: depth map " << H << "x" << W
              << " min=" << mn << " max=" << mx << " mean=" << mean << std::endl;

    bench.print_summary(/*prompt_len=*/1, /*gen=*/0);
    KernelProfiler::dump_summary();  // no-op unless NNOPT_PROFILE=1
    return 0;
}
