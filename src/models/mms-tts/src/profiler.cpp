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
    uint64_t submit_ns = 0;  // QUEUED → SUBMIT (CPU overhead: cache flush + driver setup)
    uint64_t wait_ns = 0;    // SUBMIT → START (GPU busy with prior work)
    int      count    = 0;
};

static std::map<std::string, Profile> g_profile;
static std::vector<std::pair<std::string, cl_event>> g_pending;

cl_event* event_for(const char* label) {
    if (!enabled()) return nullptr;
    g_pending.push_back({label, nullptr});
    return &g_pending.back().second;
}

// Sorted timeline of (start_ns, end_ns, label) across every captured kernel so we
// can compute REAL inter-kernel gap times (end_n to start_n+1) and report which
// two kernels bracket each gap.
struct TimelineEntry {
    cl_ulong start_ns;
    cl_ulong end_ns;
    std::string label;
};
static std::vector<TimelineEntry> g_timeline;

void process_pending() {
    if (g_pending.empty()) return;
    for (auto& kv : g_pending) {
        if (kv.second == nullptr) continue;
        cl_ulong t_queued = 0, t_submit = 0, t_start = 0, t_end = 0;
        cl_int s0 = clGetEventProfilingInfo(kv.second, CL_PROFILING_COMMAND_QUEUED, sizeof(t_queued), &t_queued, nullptr);
        cl_int s3 = clGetEventProfilingInfo(kv.second, CL_PROFILING_COMMAND_SUBMIT, sizeof(t_submit), &t_submit, nullptr);
        cl_int s1 = clGetEventProfilingInfo(kv.second, CL_PROFILING_COMMAND_START, sizeof(t_start), &t_start, nullptr);
        cl_int s2 = clGetEventProfilingInfo(kv.second, CL_PROFILING_COMMAND_END,   sizeof(t_end),   &t_end,   nullptr);
        if (s1 == CL_SUCCESS && s2 == CL_SUCCESS && t_end >= t_start) {
            auto& p = g_profile[kv.first];
            p.total_ns += (t_end - t_start);
            if (s0 == CL_SUCCESS && t_end >= t_queued) {
                p.queue_ns += (t_end - t_queued);
            }
            if (s0 == CL_SUCCESS && s3 == CL_SUCCESS && t_submit >= t_queued) {
                p.submit_ns += (t_submit - t_queued);
            }
            if (s3 == CL_SUCCESS && s1 == CL_SUCCESS && t_start >= t_submit) {
                p.wait_ns += (t_start - t_submit);
            }
            p.count    += 1;
            g_timeline.push_back(TimelineEntry{t_start, t_end, kv.first});
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
    // Per-kernel QUEUED→SUBMIT→START→END breakdown (Qualcomm guide §4.5.2)
    uint64_t grand_submit_ns = 0, grand_wait_ns = 0;
    for (const auto& kv : sorted) { grand_submit_ns += kv.second.submit_ns; grand_wait_ns += kv.second.wait_ns; }
    int total_dispatches = 0;
    for (const auto& kv : sorted) total_dispatches += kv.second.count;

    fprintf(stderr, "\n=== QUALCOMM §4.5 PROFILING BREAKDOWN ===\n");
    fprintf(stderr, "  QUEUED→SUBMIT (CPU overhead):  %10.1f ms  (cache flush + driver setup)\n", grand_submit_ns / 1.0e6);
    fprintf(stderr, "  SUBMIT→START  (GPU wait):      %10.1f ms  (GPU busy with prior work)\n", grand_wait_ns / 1.0e6);
    fprintf(stderr, "  START→END     (GPU kernel):    %10.1f ms  (actual compute)\n", grand_total_ns / 1.0e6);
    fprintf(stderr, "  QUEUED→END    (total):         %10.1f ms\n", grand_queue_ns / 1.0e6);
    fprintf(stderr, "  Dispatches:                    %10d\n", total_dispatches);
    fprintf(stderr, "  Avg QUEUED→SUBMIT per dispatch: %7.1f ms\n",
            total_dispatches > 0 ? grand_submit_ns / 1.0e6 / total_dispatches : 0.0);
    fprintf(stderr, "  Avg SUBMIT→START per dispatch:  %7.1f ms\n",
            total_dispatches > 0 ? grand_wait_ns / 1.0e6 / total_dispatches : 0.0);
    fprintf(stderr, "  Avg START→END per dispatch:     %7.1f ms\n",
            total_dispatches > 0 ? grand_total_ns / 1.0e6 / total_dispatches : 0.0);

    // REAL inter-kernel gap analysis. Sort timeline by start_ns. The wall
    // span is (last_end - first_start), the actual kernel-busy fraction is
    // sum(end-start)/wall_span. The gaps are end_n -> start_n+1.
    if (!g_timeline.empty()) {
        std::sort(g_timeline.begin(), g_timeline.end(),
                  [](const TimelineEntry& a, const TimelineEntry& b) {
                      return a.start_ns < b.start_ns;
                  });
        const cl_ulong wall_start = g_timeline.front().start_ns;
        const cl_ulong wall_end   = g_timeline.back().end_ns;
        const double   wall_ms    = (wall_end - wall_start) / 1.0e6;
        uint64_t gap_ns = 0;
        int gap_count = 0;
        uint64_t max_gap_ns = 0;
        // Collect (gap_ns, before_label, after_label, idx) so we can rank.
        struct GapEntry { uint64_t g; size_t i; };
        std::vector<GapEntry> gaps;
        gaps.reserve(g_timeline.size());
        for (size_t i = 1; i < g_timeline.size(); ++i) {
            const cl_ulong prev_end   = g_timeline[i-1].end_ns;
            const cl_ulong this_start = g_timeline[i].start_ns;
            if (this_start > prev_end) {
                const uint64_t g = this_start - prev_end;
                gap_ns += g;
                if (g > max_gap_ns) max_gap_ns = g;
                ++gap_count;
                gaps.push_back({g, i});
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

        // Rank the top-10 biggest gaps with the kernels that bracket them.
        std::sort(gaps.begin(), gaps.end(),
                  [](const GapEntry& a, const GapEntry& b) { return a.g > b.g; });
        const size_t top_n = std::min((size_t)10, gaps.size());
        if (top_n > 0) {
            fprintf(stderr, "    top %zu gaps (between which kernels):\n", top_n);
            for (size_t k = 0; k < top_n; ++k) {
                const size_t i = gaps[k].i;
                const double g_ms = gaps[k].g / 1.0e6;
                const double t_from_start_ms = (g_timeline[i-1].end_ns - wall_start) / 1.0e6;
                fprintf(stderr, "      %7.2f ms  @ %.2f s  after [%s]  before [%s]\n",
                        g_ms, t_from_start_ms / 1000.0,
                        g_timeline[i-1].label.c_str(), g_timeline[i].label.c_str());
            }
        }
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
