#pragma once
// Auto-generated benchmark utility — DO NOT EDIT.
// Single source of truth for baseline perf metrics emitted by the inference binary.
// Source template: extensions/cli/prompts/templates/scaffolds/opencl/benchmark.h.tmpl

#include <chrono>
#include <cstdio>
#include <cstdint>
#include <fstream>
#include <string>
#include <sys/resource.h>

#if defined(__linux__) || defined(__ANDROID__)
#include <unistd.h>
#endif

class BenchmarkTimer {
public:
    static BenchmarkTimer& instance() {
        static BenchmarkTimer t;
        return t;
    }

    void mark_inference_start() { t_inference_start_ = now(); }
    void mark_prefill_start()   { t_prefill_start_   = now(); }
    void mark_first_token()     { if (!first_token_set_) { t_first_token_ = now(); first_token_set_ = true; } }
    void mark_end()             { t_end_ = now(); }

    // Emit structured BENCHMARK <key>: <value> lines on stderr. These are parsed
    // by runUtils.ts parseInferenceMetrics in the TS CLI.
    void print_summary(int n_prompt_tokens, int n_generated_tokens) {
        using namespace std::chrono;
        double ttft_s       = sec_between(t_inference_start_, t_first_token_);
        double prefill_s    = sec_between(t_prefill_start_,   t_first_token_);
        double decode_s     = sec_between(t_first_token_,     t_end_);
        double total_s      = sec_between(t_inference_start_, t_end_);

        double prefill_tps  = (prefill_s > 0 && n_prompt_tokens > 0 && first_token_set_)
                              ? (double)n_prompt_tokens / prefill_s : -1.0;
        double decode_tps   = (decode_s  > 0 && n_generated_tokens > 1 && first_token_set_)
                              ? (double)(n_generated_tokens - 1) / decode_s : -1.0;
        double peak_mb      = peak_memory_mb();

        fprintf(stderr, "BENCHMARK prefill_tokens_per_sec: %.4f\n", prefill_tps);
        fprintf(stderr, "BENCHMARK decode_tokens_per_sec: %.4f\n",  decode_tps);
        fprintf(stderr, "BENCHMARK time_to_first_token_sec: %.4f\n", first_token_set_ ? ttft_s : -1.0);
        fprintf(stderr, "BENCHMARK peak_cpu_memory_mb: %.2f\n", peak_mb);
        fprintf(stderr, "BENCHMARK n_prompt_tokens: %d\n",     n_prompt_tokens);
        fprintf(stderr, "BENCHMARK n_generated_tokens: %d\n",  n_generated_tokens);
        fprintf(stderr, "BENCHMARK total_inference_sec: %.4f\n", total_s);
        fprintf(stderr, "BENCHMARK first_token_instrumented: %s\n",
                first_token_set_ ? "true" : "false");
    }

private:
    using clock      = std::chrono::steady_clock;
    using time_point = std::chrono::steady_clock::time_point;

    time_point t_inference_start_{};
    time_point t_prefill_start_{};
    time_point t_first_token_{};
    time_point t_end_{};
    bool first_token_set_ = false;

    static time_point now() { return clock::now(); }

    static double sec_between(time_point a, time_point b) {
        using namespace std::chrono;
        if (a.time_since_epoch().count() == 0 || b.time_since_epoch().count() == 0) return -1.0;
        auto ns = duration_cast<nanoseconds>(b - a).count();
        return (double)ns / 1e9;
    }

    static double peak_memory_mb() {
#if defined(__linux__) || defined(__ANDROID__)
        // /proc/self/status VmHWM is in kilobytes — peak resident set
        std::ifstream st("/proc/self/status");
        std::string line;
        while (std::getline(st, line)) {
            if (line.rfind("VmHWM:", 0) == 0) {
                long kb = 0;
                sscanf(line.c_str(), "VmHWM: %ld kB", &kb);
                return (double)kb / 1024.0;
            }
        }
        // Fallback to getrusage (Linux: ru_maxrss in kilobytes)
        struct rusage ru{};
        if (getrusage(RUSAGE_SELF, &ru) == 0) return (double)ru.ru_maxrss / 1024.0;
        return -1.0;
#elif defined(__APPLE__)
        // macOS: ru_maxrss is in bytes
        struct rusage ru{};
        if (getrusage(RUSAGE_SELF, &ru) == 0) return (double)ru.ru_maxrss / (1024.0 * 1024.0);
        return -1.0;
#else
        struct rusage ru{};
        if (getrusage(RUSAGE_SELF, &ru) == 0) return (double)ru.ru_maxrss / 1024.0;
        return -1.0;
#endif
    }
};

// Single call site inside generate(): immediately after the first sampled
// token is appended to the output vector. Safe to invoke multiple times —
// only the first call records a timestamp.
#define NNOPT_BENCH_FIRST_TOKEN() BenchmarkTimer::instance().mark_first_token()
