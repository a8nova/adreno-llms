#include "prof.h"

#include <algorithm>
#include <cstdlib>
#include <cstring>
#include <iomanip>
#include <iostream>
#include <map>
#include <string>
#include <vector>

namespace nnopt_prof {

namespace {
struct State {
    bool initialized = false;
    bool on = false;
    // (kernel_name, count, total_ns) accumulator. Events are released as we
    // query them in dump() so the running map stays small.
    std::vector<std::pair<std::string, cl_event>> pending;
};
State& state() {
    static State s;
    if (!s.initialized) {
        const char* e = std::getenv("NNOPT_PROFILE");
        s.on = (e && e[0] == '1');
        s.initialized = true;
    }
    return s;
}
} // namespace

bool enabled() { return state().on; }

cl_int enqueue(cl_command_queue queue,
               cl_kernel        kernel,
               cl_uint          work_dim,
               const size_t*    global_work_offset,
               const size_t*    global_work_size,
               const size_t*    local_work_size,
               cl_uint          num_events_in_wait_list,
               const cl_event*  event_wait_list,
               cl_event*        event_out) {
    State& s = state();
    if (!s.on) {
        return clEnqueueNDRangeKernel(queue, kernel, work_dim,
                                       global_work_offset, global_work_size,
                                       local_work_size,
                                       num_events_in_wait_list, event_wait_list,
                                       event_out);
    }
    cl_event prof_evt = nullptr;
    cl_int err = clEnqueueNDRangeKernel(queue, kernel, work_dim,
                                        global_work_offset, global_work_size,
                                        local_work_size,
                                        num_events_in_wait_list, event_wait_list,
                                        &prof_evt);
    if (err != CL_SUCCESS) return err;
    char name[128] = {0};
    clGetKernelInfo(kernel, CL_KERNEL_FUNCTION_NAME, sizeof(name), name, nullptr);
    s.pending.push_back({std::string(name), prof_evt});
    if (event_out) {
        clRetainEvent(prof_evt);
        *event_out = prof_evt;
    }
    return err;
}

void dump(cl_command_queue queue) {
    State& s = state();
    if (!s.on) return;
    if (s.pending.empty()) return;
    clFinish(queue);

    // Aggregate by name.
    struct Agg { uint64_t count = 0; uint64_t total_ns = 0; uint64_t max_ns = 0; };
    std::map<std::string, Agg> by_name;
    uint64_t grand_ns = 0;
    for (auto& kv : s.pending) {
        cl_ulong start = 0, end = 0;
        clGetEventProfilingInfo(kv.second, CL_PROFILING_COMMAND_START,
                                sizeof(start), &start, nullptr);
        clGetEventProfilingInfo(kv.second, CL_PROFILING_COMMAND_END,
                                sizeof(end), &end, nullptr);
        uint64_t dur = (end > start) ? (uint64_t)(end - start) : 0;
        Agg& a = by_name[kv.first];
        a.count += 1;
        a.total_ns += dur;
        if (dur > a.max_ns) a.max_ns = dur;
        grand_ns += dur;
        clReleaseEvent(kv.second);
    }
    s.pending.clear();

    // Sort by total time desc.
    std::vector<std::pair<std::string, Agg>> rows(by_name.begin(), by_name.end());
    std::sort(rows.begin(), rows.end(),
              [](const auto& a, const auto& b) { return a.second.total_ns > b.second.total_ns; });

    std::cerr << "\n=========== GPU per-kernel profile ===========\n";
    std::cerr << "Total recorded GPU time: " << (grand_ns / 1e6) << " ms across "
              << rows.size() << " distinct kernels\n";
    std::cerr << std::left
              << std::setw(36) << "kernel"
              << std::right
              << std::setw(8)  << "count"
              << std::setw(12) << "total_ms"
              << std::setw(10) << "avg_us"
              << std::setw(10) << "max_us"
              << std::setw(8)  << "% tot"
              << "\n";
    std::cerr << std::string(84, '-') << "\n";
    for (auto& r : rows) {
        const Agg& a = r.second;
        double total_ms = a.total_ns / 1e6;
        double avg_us   = (a.count > 0) ? (a.total_ns / 1e3 / (double)a.count) : 0.0;
        double max_us   = a.max_ns / 1e3;
        double pct      = (grand_ns > 0) ? (100.0 * a.total_ns / (double)grand_ns) : 0.0;
        std::cerr << std::left
                  << std::setw(36) << r.first
                  << std::right
                  << std::setw(8)  << a.count
                  << std::setw(12) << std::fixed << std::setprecision(3) << total_ms
                  << std::setw(10) << std::fixed << std::setprecision(1) << avg_us
                  << std::setw(10) << std::fixed << std::setprecision(1) << max_us
                  << std::setw(7)  << std::fixed << std::setprecision(1) << pct << "%"
                  << "\n";
    }
    std::cerr << std::string(84, '-') << "\n\n";
}

void reset() {
    State& s = state();
    for (auto& kv : s.pending) clReleaseEvent(kv.second);
    s.pending.clear();
}

} // namespace nnopt_prof
