// Reference: src/profiler.h (scaffold utility; not model-specific)
#include "profiler.h"

#include <algorithm>
#include <cstdio>
#include <cstdint>
#include <map>
#include <vector>

namespace KernelProfiler {

struct Profile {
    uint64_t total_ns    = 0;
    uint64_t total_bytes = 0;
    int      count       = 0;
};

struct Pending {
    std::string label;
    uint64_t    bytes = 0;
    cl_event    event = nullptr;
};

static std::map<std::string, Profile> g_profile;
static std::vector<Pending> g_pending;
static bool g_suppress = false;

void suppress(bool on) { g_suppress = on; }

cl_event* event_for(const char* label, uint64_t bytes_touched) {
    if (!enabled() || g_suppress) return nullptr;
    g_pending.push_back({label, bytes_touched, nullptr});
    return &g_pending.back().event;
}

void process_pending() {
    if (g_pending.empty()) return;
    for (auto& kv : g_pending) {
        if (kv.event == nullptr) continue;
        cl_ulong t_start = 0, t_end = 0;
        cl_int s1 = clGetEventProfilingInfo(kv.event, CL_PROFILING_COMMAND_START, sizeof(t_start), &t_start, nullptr);
        cl_int s2 = clGetEventProfilingInfo(kv.event, CL_PROFILING_COMMAND_END,   sizeof(t_end),   &t_end,   nullptr);
        if (s1 == CL_SUCCESS && s2 == CL_SUCCESS && t_end >= t_start) {
            auto& p = g_profile[kv.label];
            p.total_ns    += (t_end - t_start);
            p.total_bytes += kv.bytes;
            p.count       += 1;
        }
        clReleaseEvent(kv.event);
        kv.event = nullptr;
    }
    g_pending.clear();
}

void dump_summary() {
    process_pending();
    if (g_profile.empty()) return;

    // Some Android builds can SIGKILL when printing huge summaries (logd/stdout buffer
    // pressure). Cap output to the top entries.
    const size_t kMaxRows = 32;

    std::vector<std::pair<std::string, Profile>> sorted(g_profile.begin(), g_profile.end());
    std::sort(sorted.begin(), sorted.end(),
              [](const std::pair<std::string, Profile>& a,
                 const std::pair<std::string, Profile>& b) {
                  return a.second.total_ns > b.second.total_ns;
              });

    uint64_t grand_total_ns = 0;
    for (const auto& kv : sorted) grand_total_ns += kv.second.total_ns;

    fprintf(stderr, "\n=== KERNEL PROFILE (env NNOPT_PROFILE=1) ===\n");
    fprintf(stderr, "%-32s %12s %8s %12s %8s %8s\n", "label", "total_ms", "%total", "calls", "avg_us", "GB/s");
    fprintf(stderr, "%-32s %12s %8s %12s %8s %8s\n", "--------------------------------",
            "------------", "--------", "------------", "--------", "--------");
    const size_t n_rows = std::min(sorted.size(), kMaxRows);
    for (size_t i = 0; i < n_rows; ++i) {
        const auto& kv = sorted[i];
        const std::string& name = kv.first;
        const Profile& p = kv.second;
        double total_ms = p.total_ns / 1.0e6;
        double pct      = grand_total_ns > 0 ? 100.0 * (double)p.total_ns / (double)grand_total_ns : 0.0;
        double avg_us   = p.count > 0 ? (p.total_ns / 1.0e3) / (double)p.count : 0.0;
        if (p.total_bytes > 0 && p.total_ns > 0) {
            // bytes/ns == GB/s; achieved bandwidth vs ~14 GB/s practical bus.
            double gbs = (double)p.total_bytes / (double)p.total_ns;
            fprintf(stderr, "%-32s %12.3f %7.2f%% %12d %8.2f %8.2f\n",
                    name.c_str(), total_ms, pct, p.count, avg_us, gbs);
        } else {
            fprintf(stderr, "%-32s %12.3f %7.2f%% %12d %8.2f %8s\n",
                    name.c_str(), total_ms, pct, p.count, avg_us, "-");
        }
    }
    fprintf(stderr, "=== TOTAL GPU kernel time: %.3f ms ===\n\n", grand_total_ns / 1.0e6);
}

void reset() {
    for (auto& kv : g_pending) {
        if (kv.event) clReleaseEvent(kv.event);
    }
    g_pending.clear();
    g_profile.clear();
}

}  // namespace KernelProfiler
