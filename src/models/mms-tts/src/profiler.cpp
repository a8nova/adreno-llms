// Reference: model_info/transformers_src/modeling_vits.py (profiling is infrastructure-only; VITS
// uses many Conv1d/attention/flow kernels and this collects GPU timings for optimization)
#include "profiler.h"

#include <algorithm>
#include <cstdio>
#include <cstdint>
#include <map>
#include <vector>

namespace KernelProfiler {

struct Profile {
    uint64_t total_ns = 0;   // START → END (pure kernel runtime)
    uint64_t queue_ns = 0;   // QUEUED → END (full pipeline including driver/queue wait)
    int      count    = 0;
};

static std::map<std::string, Profile> g_profile;
static std::vector<std::pair<std::string, cl_event>> g_pending;

cl_event* event_for(const char* label) {
    if (!enabled()) return nullptr;
    g_pending.push_back({label, nullptr});
    return &g_pending.back().second;
}

// Sorted timeline of (start_ns, end_ns) across every captured kernel so we
// can compute REAL inter-kernel gap times (end_n to start_n+1).
static std::vector<std::pair<cl_ulong, cl_ulong>> g_timeline;

void process_pending() {
    if (g_pending.empty()) return;
    for (auto& kv : g_pending) {
        if (kv.second == nullptr) continue;
        cl_ulong t_queued = 0, t_start = 0, t_end = 0;
        cl_int s0 = clGetEventProfilingInfo(kv.second, CL_PROFILING_COMMAND_QUEUED, sizeof(t_queued), &t_queued, nullptr);
        cl_int s1 = clGetEventProfilingInfo(kv.second, CL_PROFILING_COMMAND_START, sizeof(t_start), &t_start, nullptr);
        cl_int s2 = clGetEventProfilingInfo(kv.second, CL_PROFILING_COMMAND_END,   sizeof(t_end),   &t_end,   nullptr);
        if (s1 == CL_SUCCESS && s2 == CL_SUCCESS && t_end >= t_start) {
            auto& p = g_profile[kv.first];
            p.total_ns += (t_end - t_start);
            if (s0 == CL_SUCCESS && t_end >= t_queued) {
                p.queue_ns += (t_end - t_queued);
            }
            p.count    += 1;
            g_timeline.push_back({t_start, t_end});
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

    uint64_t grand_queue_ns = 0;
    for (const auto& kv : sorted) grand_queue_ns += kv.second.queue_ns;

    fprintf(stderr, "\n=== KERNEL PROFILE (env NNOPT_PROFILE=1) ===\n");
    fprintf(stderr, "%-32s %10s %10s %8s %10s\n",
            "label", "kern_ms", "queue_ms", "calls", "q-k_us");
    fprintf(stderr, "%-32s %10s %10s %8s %10s\n", "--------------------------------",
            "----------", "----------", "--------", "----------");
    for (const auto& kv : sorted) {
        const std::string& name = kv.first;
        const Profile& p = kv.second;
        double kern_ms  = p.total_ns / 1.0e6;
        double queue_ms = p.queue_ns / 1.0e6;
        double diff_us  = p.count > 0 ? ((p.queue_ns - p.total_ns) / 1.0e3) / (double)p.count : 0.0;
        fprintf(stderr, "%-32s %10.3f %10.3f %8d %10.2f\n",
                name.c_str(), kern_ms, queue_ms, p.count, diff_us);
    }
    fprintf(stderr, "=== TOTAL kernel runtime (start->end):    %.3f ms ===\n",
            grand_total_ns / 1.0e6);
    fprintf(stderr, "=== TOTAL pipeline latency (queued->end): %.3f ms ===\n",
            grand_queue_ns / 1.0e6);
    fprintf(stderr, "=== Per-dispatch driver/queue overhead ≈ %.0f us ===\n",
            grand_queue_ns > grand_total_ns
                ? (grand_queue_ns - grand_total_ns) / 1000.0 / 280.0
                : 0.0);

    // REAL inter-kernel gap analysis. Sort timeline by start_ns. The wall
    // span is (last_end - first_start), the actual kernel-busy fraction is
    // sum(end-start)/wall_span. The gaps are end_n -> start_n+1.
    if (!g_timeline.empty()) {
        std::sort(g_timeline.begin(), g_timeline.end(),
                  [](const std::pair<cl_ulong, cl_ulong>& a,
                     const std::pair<cl_ulong, cl_ulong>& b) {
                      return a.first < b.first;
                  });
        const cl_ulong wall_start = g_timeline.front().first;
        const cl_ulong wall_end   = g_timeline.back().second;
        const double   wall_ms    = (wall_end - wall_start) / 1.0e6;
        uint64_t gap_ns = 0;
        int gap_count = 0;
        uint64_t max_gap_ns = 0;
        for (size_t i = 1; i < g_timeline.size(); ++i) {
            const cl_ulong prev_end   = g_timeline[i-1].second;
            const cl_ulong this_start = g_timeline[i].first;
            if (this_start > prev_end) {
                const uint64_t g = this_start - prev_end;
                gap_ns += g;
                if (g > max_gap_ns) max_gap_ns = g;
                ++gap_count;
            }
        }
        fprintf(stderr, "=== GPU TIMELINE ===\n");
        fprintf(stderr, "    span (first kernel start -> last kernel end): %.3f ms\n", wall_ms);
        fprintf(stderr, "    kernel-busy fraction:                          %.1f%%\n",
                wall_ms > 0 ? 100.0 * grand_total_ns / (wall_end - wall_start) : 0.0);
        fprintf(stderr, "    total inter-kernel idle (sum of gaps):         %.3f ms across %d gaps\n",
                gap_ns / 1.0e6, gap_count);
        fprintf(stderr, "    avg gap:                                       %.3f ms\n",
                gap_count > 0 ? (gap_ns / 1.0e6) / gap_count : 0.0);
        fprintf(stderr, "    max gap:                                       %.3f ms\n",
                max_gap_ns / 1.0e6);
        fprintf(stderr, "\n");
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
