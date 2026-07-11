#include "profiler.h"

#include <algorithm>
#include <chrono>
#include <cstdio>
#include <cstdint>
#include <map>
#include <string>
#include <vector>

namespace KernelProfiler {

struct Profile {
    uint64_t total_ns = 0;
    int      count    = 0;
};

static std::map<std::string, Profile> g_profile;
static std::vector<std::pair<std::string, cl_event>> g_pending;
static std::map<std::string, Profile> g_host;  // host-wall accumulator

void host_add_ns(const char* label, uint64_t ns) {
    if (!enabled()) return;
    auto& p = g_host[label];
    p.total_ns += ns;
    p.count    += 1;
}

static unsigned long long now_ns() {
    return (unsigned long long)std::chrono::duration_cast<std::chrono::nanoseconds>(
        std::chrono::steady_clock::now().time_since_epoch()).count();
}

HostTimer::HostTimer(const char* label) : label_(label), t0_(enabled() ? now_ns() : 0ull) {}
HostTimer::~HostTimer() {
    if (!enabled()) return;
    host_add_ns(label_, now_ns() - t0_);
}

cl_event* event_for(const char* label) {
    if (!enabled()) return nullptr;
    g_pending.push_back({label, nullptr});
    return &g_pending.back().second;
}

void process_pending() {
    if (g_pending.empty()) return;
    for (auto& kv : g_pending) {
        if (kv.second == nullptr) continue;
        cl_ulong t_start = 0, t_end = 0;
        cl_int s1 = clGetEventProfilingInfo(kv.second, CL_PROFILING_COMMAND_START, sizeof(t_start), &t_start, nullptr);
        cl_int s2 = clGetEventProfilingInfo(kv.second, CL_PROFILING_COMMAND_END,   sizeof(t_end),   &t_end,   nullptr);
        if (s1 == CL_SUCCESS && s2 == CL_SUCCESS && t_end >= t_start) {
            auto& p = g_profile[kv.first];
            p.total_ns += (t_end - t_start);
            p.count    += 1;
        }
        clReleaseEvent(kv.second);
        kv.second = nullptr;
    }
    g_pending.clear();
}

void dump_summary() {
    process_pending();
    if (g_profile.empty()) return;

    std::vector<std::pair<std::string, Profile>> sorted(g_profile.begin(), g_profile.end());
    std::sort(sorted.begin(), sorted.end(),
              [](const std::pair<std::string, Profile>& a,
                 const std::pair<std::string, Profile>& b) {
                  return a.second.total_ns > b.second.total_ns;
              });

    uint64_t grand_total_ns = 0;
    for (const auto& kv : sorted) grand_total_ns += kv.second.total_ns;

    fprintf(stderr, "\n=== KERNEL PROFILE (env NNOPT_PROFILE=1) ===\n");
    fprintf(stderr, "%-32s %12s %8s %12s %8s\n", "label", "total_ms", "%total", "calls", "avg_us");
    fprintf(stderr, "%-32s %12s %8s %12s %8s\n", "--------------------------------",
            "------------", "--------", "------------", "--------");
    for (const auto& kv : sorted) {
        const std::string& name = kv.first;
        const Profile& p = kv.second;
        double total_ms = p.total_ns / 1.0e6;
        double pct      = grand_total_ns > 0 ? 100.0 * (double)p.total_ns / (double)grand_total_ns : 0.0;
        double avg_us   = p.count > 0 ? (p.total_ns / 1.0e3) / (double)p.count : 0.0;
        fprintf(stderr, "%-32s %12.3f %7.2f%% %12d %8.2f\n",
                name.c_str(), total_ms, pct, p.count, avg_us);
    }
    fprintf(stderr, "=== TOTAL GPU kernel time: %.3f ms ===\n\n", grand_total_ns / 1.0e6);

    if (!g_host.empty()) {
        std::vector<std::pair<std::string, Profile>> hs(g_host.begin(), g_host.end());
        std::sort(hs.begin(), hs.end(),
                  [](const std::pair<std::string, Profile>& a,
                     const std::pair<std::string, Profile>& b) {
                      return a.second.total_ns > b.second.total_ns;
                  });
        uint64_t htot = 0;
        for (const auto& kv : hs) htot += kv.second.total_ns;
        fprintf(stderr, "=== HOST-WALL PROFILE (NNOPT_PROFILE=1) ===\n");
        fprintf(stderr, "%-32s %12s %12s %8s\n", "label", "total_ms", "calls", "avg_us");
        for (const auto& kv : hs) {
            const Profile& p = kv.second;
            double total_ms = p.total_ns / 1.0e6;
            double avg_us   = p.count > 0 ? (p.total_ns / 1.0e3) / (double)p.count : 0.0;
            fprintf(stderr, "%-32s %12.3f %12d %8.2f\n",
                    kv.first.c_str(), total_ms, p.count, avg_us);
        }
        fprintf(stderr, "=== TOTAL measured host-wall: %.3f ms ===\n\n", htot / 1.0e6);
    }
}

void reset() {
    for (auto& kv : g_pending) {
        if (kv.second) clReleaseEvent(kv.second);
    }
    g_pending.clear();
    g_profile.clear();
}

}  // namespace KernelProfiler
