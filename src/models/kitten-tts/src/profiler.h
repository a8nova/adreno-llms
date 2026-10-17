// Lightweight OpenCL event-based kernel profiler.
//
// Activation: set env `NNOPT_PROFILE=1` at runtime. When unset (the
// default benchmark/release path) all helpers compile to nullptr / no-op.
//
// Usage at an enqueue site:
//   cl_event* evt = KernelProfiler::event_for("op_attention_scores");
//   clEnqueueNDRangeKernel(queue, k, 1, nullptr, &gws, &lws, 0, nullptr, evt);
//
// The profiler stashes the event with its label. After generate() finishes,
// call KernelProfiler::process_pending() to read CL_PROFILING_COMMAND_START
// and CL_PROFILING_COMMAND_END from each captured event and accumulate into
// per-label totals. Then dump_summary() prints a sorted breakdown to stderr.
//
// The queue must have CL_QUEUE_PROFILING_ENABLE set (the scaffold's
// OpenCLContext does — see opencl_context.cpp). Per-event overhead is
// ~10-30 µs of host work plus ~0 GPU work — fine for one-off profile
// runs, never on the benchmark path.

#pragma once

#include <CL/cl.h>
#include <cstdlib>
#include <string>

namespace KernelProfiler {

inline bool enabled() {
    static int e = -1;
    if (e == -1) {
        const char* env = std::getenv("NNOPT_PROFILE");
        e = (env && env[0] != '0') ? 1 : 0;
    }
    return e == 1;
}

// Returns a cl_event* to pass to clEnqueueNDRangeKernel (or CLBlast). When
// profiling is off returns nullptr and the enqueue call sees no event arg.
cl_event* event_for(const char* label);

// Drain all captured events: read profiling info, accumulate, release.
// Safe to call multiple times — drains whatever is pending. The next
// process_pending() picks up new events captured since.
void process_pending();

// Print the per-label summary to stderr (sorted by total time desc).
// Implicitly calls process_pending() first so any tail events are counted.
void dump_summary();

// Reset all accumulated state. Useful between prefill and decode if you want
// per-phase numbers; we don't do that today (single combined dump).
void reset();

}  // namespace KernelProfiler
