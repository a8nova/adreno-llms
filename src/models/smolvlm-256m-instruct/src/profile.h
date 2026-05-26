#pragma once
// Lightweight host-side profiler for decode hotspots.
// Activated by env var NNOPT_PROFILE=1. When off, every macro is a no-op
// (no clFinish, no chrono, no map lookup).
//
// Usage:
//   NNOPT_PROFILE_BEGIN(queue, "label");
//   ... enqueue some kernels ...
//   NNOPT_PROFILE_END(queue, "label");
//
// At process exit, all labels are dumped to stderr as:
//   PROFILE label: total_ms=… calls=… mean_us=…
// sorted by descending total_ms.

#include <CL/cl.h>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <unordered_map>
#include <vector>
#include <algorithm>

namespace nnopt_profile {

struct Stat { uint64_t total_ns = 0; uint64_t calls = 0; };

inline std::unordered_map<std::string, Stat>& stats() {
    static std::unordered_map<std::string, Stat> m;
    return m;
}

inline bool enabled() {
    static int cached = -1;
    if (cached < 0) {
        const char* e = getenv("NNOPT_PROFILE");
        cached = (e && e[0] && e[0] != '0') ? 1 : 0;
        if (cached) {
            std::atexit([]() {
                auto& s = stats();
                std::vector<std::pair<std::string, Stat>> rows(s.begin(), s.end());
                std::sort(rows.begin(), rows.end(),
                          [](const auto& a, const auto& b) {
                              return a.second.total_ns > b.second.total_ns;
                          });
                fprintf(stderr, "\n=== PROFILE SUMMARY ===\n");
                uint64_t grand = 0;
                for (auto& r : rows) grand += r.second.total_ns;
                for (auto& r : rows) {
                    const double total_ms = (double)r.second.total_ns / 1e6;
                    const double mean_us  = r.second.calls
                        ? (double)r.second.total_ns / r.second.calls / 1e3 : 0.0;
                    const double pct = grand ? (100.0 * r.second.total_ns / grand) : 0.0;
                    fprintf(stderr,
                            "PROFILE %-28s total_ms=%10.3f  calls=%7llu  mean_us=%9.2f  pct=%5.1f%%\n",
                            r.first.c_str(), total_ms,
                            (unsigned long long)r.second.calls, mean_us, pct);
                }
                fprintf(stderr, "PROFILE %-28s total_ms=%10.3f\n", "(SUM)", (double)grand / 1e6);
                fflush(stderr);
            });
        }
    }
    return cached == 1;
}

inline std::string& phase_prefix() {
    static std::string p;  // e.g., "prefill_" or "decode_"
    return p;
}

inline void set_phase(const char* name) {
    if (!name || !*name) { phase_prefix().clear(); return; }
    phase_prefix() = std::string(name) + "_";
}

inline void record(const std::string& label, uint64_t ns) {
    auto& s = stats()[label];
    s.total_ns += ns;
    s.calls += 1;
}

// In-flight stack of started labels. Pair with their start timestamps.
struct InFlight {
    std::string label;
    std::chrono::steady_clock::time_point t0;
};

inline std::vector<InFlight>& stack() {
    static std::vector<InFlight> v;
    return v;
}

inline void begin(cl_command_queue queue, const char* label) {
    if (!enabled()) return;
    if (queue) clFinish(queue);  // make prior work observable before we time
    stack().push_back({phase_prefix() + label, std::chrono::steady_clock::now()});
}

inline void end(cl_command_queue queue, const char* label) {
    if (!enabled()) return;
    if (queue) clFinish(queue);
    auto& st = stack();
    if (st.empty()) return;
    const std::string full = phase_prefix() + label;
    // Pop matching label (defensive — labels should nest cleanly).
    auto it = st.end();
    for (auto rit = st.rbegin(); rit != st.rend(); ++rit) {
        if (rit->label == full) { it = (rit + 1).base(); break; }
    }
    if (it == st.end()) { st.pop_back(); return; }
    const auto t1 = std::chrono::steady_clock::now();
    const uint64_t ns = (uint64_t)std::chrono::duration_cast<std::chrono::nanoseconds>(t1 - it->t0).count();
    record(full, ns);
    st.erase(it);
}

}  // namespace nnopt_profile

#define NNOPT_PROFILE_BEGIN(queue, label) ::nnopt_profile::begin((queue), (label))
#define NNOPT_PROFILE_END(queue, label)   ::nnopt_profile::end((queue), (label))
