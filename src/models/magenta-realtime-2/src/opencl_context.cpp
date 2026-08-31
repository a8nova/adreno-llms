#include "version.h"
#include "opencl_context.h"
#include "profiler.h"
#include <dlfcn.h>
#include <vector>
#include "utils.h"   // nnopt_toggle_epoch — per-request toggle refresh
#include "debug_utils.h"   // NNOPT_ERROR_FMT — used in build_program / build_program_from_file below.

#include <fstream>
#include <sstream>
#include <iostream>
#include <cstring>

OpenCLContext::OpenCLContext() {}

OpenCLContext::~OpenCLContext() {
    if (queue_) clReleaseCommandQueue(queue_);
    if (context_) clReleaseContext(context_);
}

bool OpenCLContext::initialize(int platform_idx, int device_idx) {
    cl_uint num_platforms;
    clGetPlatformIDs(0, nullptr, &num_platforms);
    if (num_platforms == 0) return false;

    std::vector<cl_platform_id> platforms(num_platforms);
    clGetPlatformIDs(num_platforms, platforms.data(), nullptr);

    if (platform_idx >= (int)num_platforms) platform_idx = 0;
    platform_ = platforms[platform_idx];

    cl_uint num_devices;
    clGetDeviceIDs(platform_, CL_DEVICE_TYPE_ALL, 0, nullptr, &num_devices);
    if (num_devices == 0) return false;

    std::vector<cl_device_id> devices(num_devices);
    clGetDeviceIDs(platform_, CL_DEVICE_TYPE_ALL, num_devices, devices.data(), nullptr);

    if (device_idx >= (int)num_devices) device_idx = 0;
    device_ = devices[device_idx];

    cl_int err;
    // Ask the driver for the high performance / high priority operating point. Qualcomm exposes
    // these as CONTEXT PROPERTIES (cl_qcom_perf_hint, cl_qcom_priority_hint — both advertised by
    // the Adreno 840), and without them a compute context can be scheduled at a lower DVFS point
    // than the workload deserves. Costs nothing and cannot change a single computed value.
    //
    // The extension may be absent on other devices, so the hinted context is attempted first and we
    // fall back to a plain one rather than failing to start. NNOPT_PERFHINT=0 disables for A/B.
    static const cl_context_properties kPerfHintQcom     = 0x40C2;  // CL_CONTEXT_PERF_HINT_QCOM
    static const cl_context_properties kPerfHintHigh     = 0x40C3;  // CL_PERF_HINT_HIGH_QCOM
    static const cl_context_properties kPriorityHintQcom = 0x40C9;  // CL_CONTEXT_PRIORITY_HINT_QCOM
    static const cl_context_properties kPriorityHintHigh = 0x40CA;  // CL_PRIORITY_HINT_HIGH_QCOM
    const char* ph = std::getenv("NNOPT_PERFHINT");
    const bool want_hint = !(ph && ph[0] == '0');
    context_ = nullptr;
    if (want_hint) {
        // Try both hints together, then perf alone: a driver that rejects one property fails the
        // whole clCreateContext with CL_INVALID_PROPERTY, so an all-or-nothing attempt would throw
        // away the hint that IS supported.
        const cl_context_properties both[] = {
            kPerfHintQcom, kPerfHintHigh, kPriorityHintQcom, kPriorityHintHigh, 0 };
        const cl_context_properties perf_only[] = { kPerfHintQcom, kPerfHintHigh, 0 };
        const char* which = "none";
        context_ = clCreateContext(both, 1, &device_, nullptr, nullptr, &err);
        if (err == CL_SUCCESS && context_) which = "perf+priority";
        else {
            context_ = clCreateContext(perf_only, 1, &device_, nullptr, nullptr, &err);
            if (err == CL_SUCCESS && context_) which = "perf";
            else context_ = nullptr;
        }
        std::fprintf(stderr, "PERFHINT qcom hints=%s (err=%d)\n", which, (int)err);
        std::fflush(stderr);
    }
    if (!context_) context_ = clCreateContext(nullptr, 1, &device_, nullptr, nullptr, &err);
    if (err != CL_SUCCESS || !context_) return false;

    // CL_QUEUE_PROFILING_ENABLE only when something is actually going to read a timestamp.
    //
    // It was unconditional. Every dispatch on this queue therefore carried the driver's profiling
    // instrumentation — 48,300 of them per 2 s render — and the AR measures 94% host-bound at ~62 us
    // of host cost PER DISPATCH, which is roughly ten times what a plain enqueue should cost. A
    // queue that has to stamp queued/submit/start/end on every command, and give up whatever
    // batching that precludes, is the first thing to suspect for that.
    //
    // Nothing needs it by default: the workgroup-size tuner times with steady_clock, and both event
    // profilers (AR_FRAME_GPU, the component profile) are behind NNOPT_ARPROF / NNOPT_PROFILE. Ask
    // for either and the queue is created with profiling, so the measuring path is unchanged.
    // NNOPT_MEASURE_BUILD: profiling queue from startup. The env-var route cannot work for the
    // app — the queue is created before the first request, and the request is where the app can set
    // anything. This is a build flag so a measurement APK exists at all.
    const bool want_prof =
#ifdef NNOPT_MEASURE_BUILD
        true;
#else
        std::getenv("NNOPT_PROFILE") ||
                           (std::getenv("NNOPT_ARPROF") && std::getenv("NNOPT_ARPROF")[0] != '0');
#endif
    queue_ = clCreateCommandQueue(context_, device_,
                                  want_prof ? CL_QUEUE_PROFILING_ENABLE : 0, &err);
    if (err != CL_SUCCESS) return false;
    std::fprintf(stderr, "QUEUE profiling=%d (off is the fast path; NNOPT_ARPROF=1 turns it on)\n",
                 (int)want_prof);
    std::fflush(stderr);
    load_record_fns();
    return true;
}

cl_program OpenCLContext::build_program(const std::string& source, const std::string& options) {
    cl_int err;
    const char* src_ptr = source.c_str();
    size_t src_len = source.size();

    cl_program program = clCreateProgramWithSource(context_, 1, &src_ptr, &src_len, &err);
    if (err != CL_SUCCESS) {
        NNOPT_ERROR_FMT("clCreateProgramWithSource failed (err=%d)", (int)err);
        return nullptr;
    }

    // Forward host-side dtype to the kernel preamble. Without this, every
    // scaffold-emitted and agent-written kernel falls through to the fp32
    // path of `#ifdef USE_FP16` and reads garbage from cl_half buffers.
    std::string effective_options = options;
#ifdef NNOPT_USE_FP16
    if (effective_options.find("USE_FP16") == std::string::npos) {
        if (!effective_options.empty()) effective_options += " ";
        effective_options += "-D USE_FP16=1";
    }
#endif
    // Fast-math: enables mad fusion + native trig/exp (big win for the ISTFT cos/sin
    // and softplus/exp in attention). Safe for this audio port's precision budget.
    if (effective_options.find("fast-relaxed-math") == std::string::npos) {
        if (!effective_options.empty()) effective_options += " ";
        effective_options += "-cl-fast-relaxed-math -cl-mad-enable";
    }

    err = clBuildProgram(program, 1, &device_, effective_options.c_str(), nullptr, nullptr);
    if (err != CL_SUCCESS) {
        NNOPT_ERROR_FMT("clBuildProgram FAILED (err=%d)", (int)err);
        size_t log_size = 0;
        clGetProgramBuildInfo(program, device_, CL_PROGRAM_BUILD_LOG, 0, nullptr, &log_size);
        if (log_size > 0) {
            std::vector<char> log(log_size + 1, 0);
            clGetProgramBuildInfo(program, device_, CL_PROGRAM_BUILD_LOG, log_size, log.data(), nullptr);
            fprintf(stderr, "OpenCL Build Log: %s\n", log.data());
            fflush(stderr);
        }
        clReleaseProgram(program);
        return nullptr;
    }

    return program;
}

cl_program OpenCLContext::build_program_from_file(const std::string& path, const std::string& options) {
    std::ifstream file(path);
    if (!file.is_open()) {
        NNOPT_ERROR_FMT("Failed to open kernel file: %s", path.c_str());
        return nullptr;
    }

    std::stringstream buffer;
    buffer << file.rdbuf();
    cl_program prog = build_program(buffer.str(), options);
    if (!prog) {
        NNOPT_ERROR_FMT("OpenCL kernel compilation FAILED for: %s (file opened OK, but clBuildProgram returned error)", path.c_str());
    }
    return prog;
}

std::string OpenCLContext::device_name() const {
    char name[256];
    clGetDeviceInfo(device_, CL_DEVICE_NAME, sizeof(name), name, nullptr);
    return std::string(name);
}

size_t OpenCLContext::max_work_group_size() const {
    size_t size;
    clGetDeviceInfo(device_, CL_DEVICE_MAX_WORK_GROUP_SIZE, sizeof(size), &size, nullptr);
    return size;
}

size_t OpenCLContext::local_mem_size() const {
    cl_ulong size;
    clGetDeviceInfo(device_, CL_DEVICE_LOCAL_MEM_SIZE, sizeof(size), &size, nullptr);
    return (size_t)size;
}

// ── Qualcomm-guide cl_event GPU profiler ─────────────────────────────────────
#include <map>
#include <mutex>
#include <vector>
#include <algorithm>
#include <cstdio>
#include <cstdlib>
namespace {
struct ProfEvt { cl_event ev; const char* name; };
std::vector<ProfEvt> g_prof_events;
bool g_prof_init = false, g_prof_on = false;
}
void OpenCLContext::profRefresh() { g_prof_init = false; }
bool OpenCLContext::profWantEvent() { return profEnabled(); }
void OpenCLContext::profEventAdd(const char* label, cl_event ev) {
    if (!ev) return;
    if (!profEnabled()) { clReleaseEvent(ev); return; }
    g_prof_events.push_back({ev, label});
}
// ── one-frame AR GPU profile ────────────────────────────────────────────────────────────────────
// Always on, for exactly ONE live AR frame, so "where does the time go" never needs a flag or a
// second run. It answers the question that matters now that host issue is no longer the wall: of
// the ~519 dispatches in a frame, which ones actually cost GPU time, and how much of the frame is
// kernel execution versus per-dispatch launch overhead.
namespace { double g_arprof_frame_ms = 0.0; }
double OpenCLContext::arProfFrameMs() { return g_arprof_frame_ms; }
namespace { bool g_arprof_on = false; std::vector<ProfEvt> g_arprof_events;
           const char* g_arprof_tag = "AR_FRAME_GPU"; }

void OpenCLContext::arProfBegin(const char* tag) {
    // RELEASE, not just clear. A session that never reaches arProfEnd — any error path that breaks
    // out of the frame loop between the two — leaves its events sitting here, and clear() drops the
    // handles without releasing them. That is ~519 leaked cl_events per render, forever, on a
    // profile that runs on frame 0 of EVERY render rather than only under a flag.
    for (auto& pe : g_arprof_events) if (pe.ev) clReleaseEvent(pe.ev);
    g_arprof_events.clear();
    g_arprof_on = true; g_arprof_tag = tag ? tag : "AR_FRAME_GPU";
}

void OpenCLContext::arProfEnd(cl_command_queue q) {
    if (!g_arprof_on) return;
    g_arprof_on = false;
    if (g_arprof_events.empty()) return;
    clFinish(q);
    struct Agg { double ms = 0; long n = 0; };
    std::map<std::string, Agg> by;
    double total = 0.0;
    for (auto& pe : g_arprof_events) {
        cl_ulong t0 = 0, t1 = 0;
        clGetEventProfilingInfo(pe.ev, CL_PROFILING_COMMAND_START, sizeof(t0), &t0, nullptr);
        clGetEventProfilingInfo(pe.ev, CL_PROFILING_COMMAND_END,   sizeof(t1), &t1, nullptr);
        clReleaseEvent(pe.ev);
        if (t1 > t0) { const double ms = (double)(t1 - t0) / 1e6;
                       by[pe.name ? pe.name : "?"].ms += ms; by[pe.name ? pe.name : "?"].n += 1;
                       total += ms; }
    }
    g_arprof_events.clear();
    g_arprof_frame_ms = total;
    std::vector<std::pair<std::string, Agg>> v(by.begin(), by.end());
    std::sort(v.begin(), v.end(), [](const auto& a, const auto& b){ return a.second.ms > b.second.ms; });
    long ndisp = 0; for (auto& x : v) ndisp += x.second.n;
    std::fprintf(stderr, "%s: %ld dispatches, %.2f ms of kernel execution\n",
                 g_arprof_tag, ndisp, total);
    for (auto& x : v)
        std::fprintf(stderr, "%s   %-18s %7.3f ms  n=%-4ld  %5.1f%%  %6.1f us/call\n",
                     g_arprof_tag, x.first.c_str(), x.second.ms, x.second.n,
                     100.0 * x.second.ms / (total > 0 ? total : 1),
                     1000.0 * x.second.ms / (double)(x.second.n ? x.second.n : 1));
    std::fflush(stderr);
}

bool OpenCLContext::profEnabled() {
    // Keyed on the per-generation toggle epoch: latching this once per process meant a profile
    // requested by a serve request could never turn on, because the first call is the warm-up.
    static int seen_epoch = -1;
    if (!g_prof_init || seen_epoch != nnopt_toggle_epoch()) {
        g_prof_on = (std::getenv("NNOPT_PROFILE") != nullptr);
        g_prof_init = true;
        seen_epoch = nnopt_toggle_epoch();
    }
    return g_prof_on;
}
// ── automatic workgroup-size selection ──────────────────────────────────────────────────────────
// 23 of this engine's 34 dispatch sites pass lws=nullptr and let the driver choose. Guide
// 80-NB295-11 Rev C 6.1.5.1 is blunt about that: "the default workgroup size is unlikely to be
// optimal", and 6.1.5.2 says the best size and shape need trial and error per kernel. So do the
// trial and error on the device, once per (kernel, global-size), and cache the winner.
//
// SAFETY — a wrong workgroup size is a correctness bug, not just a slow one. Only kernels that
// cannot care about their group size are tuned:
//   * CL_KERNEL_COMPILE_WORK_GROUP_SIZE == 0    — no __attribute__((reqd_work_group_size))
//   * CL_KERNEL_LOCAL_MEM_SIZE          == 0    — no __local, therefore no barrier/tree reduction
//                                                 whose shape depends on the group size
// Anything else keeps whatever the caller passed. Candidates must also divide the global size
// exactly (OpenCL 1.2) and fit CL_KERNEL_WORK_GROUP_SIZE.
namespace {
struct LwsKey {
    const void* kernel; cl_uint dim; size_t g[3];
    bool operator<(const LwsKey& o) const {
        if (kernel != o.kernel) return kernel < o.kernel;
        if (dim != o.dim) return dim < o.dim;
        for (int i = 0; i < 3; ++i) if (g[i] != o.g[i]) return g[i] < o.g[i];
        return false;
    }
};
std::map<LwsKey, OpenCLContext::LwsChoice> g_lws_cache;

// NNOPT_LWSLOG=1 prints each choice; off by default so a normal run stays readable.
bool lws_verbose() {
    static int on = -1, epoch = -1;
    if (epoch != nnopt_toggle_epoch()) {
        const char* e = std::getenv("NNOPT_LWSLOG");
        on = (e && e[0] && e[0] != '0') ? 1 : 0;
        epoch = nnopt_toggle_epoch();
    }
    return on != 0;
}

// Epoch-keyed, NOT a plain static: a function-local static is evaluated during the WARM-UP render,
// before a serve request has set its env, so every later request silently measures the warm-up's
// value under the experiment's name. That trap has produced false verdicts in this port before.
bool g_lws_window = false;
std::mutex g_lws_mu;

bool lws_tuning_enabled() {
    static int on = -1, epoch = -1;
    if (epoch != nnopt_toggle_epoch()) {
        const char* e = std::getenv("NNOPT_LWS");
        on = (e && e[0] == '0') ? 0 : 1;
        epoch = nnopt_toggle_epoch();
    }
    return on != 0;
}
}  // namespace

void OpenCLContext::setLwsTuningWindow(bool open) { g_lws_window = open; }
bool OpenCLContext::lwsTuningWindowOpen() { return g_lws_window; }

// ── host-side cost of a dispatch ────────────────────────────────────────────────────────────────
// The AR measures 94% HOST-bound: 2.9 s building the command stream against 198 ms of GPU drain.
// A GPU profiler shows a mostly-empty timeline, which is accurate and useless. What was missing is
// where the HOST time goes, and the frame decomposes into exactly three parts:
//
//   driver     = time inside clEnqueueNDRangeKernel
//   wrapper    = the rest of profEnqueue (workgroup-size lookup, kernel bookkeeping)
//   call site  = frame time - profEnqueue time  (clSetKernelArg, string building, everything else)
//
// Keyed on the name POINTER, not the string: these are string literals with stable addresses, and
// hashing a std::string per dispatch would be spending the very resource this is here to measure.
namespace {
struct HostCost { const char* name; double total_us; double drv_us; long n; };
HostCost g_host[48];
int      g_host_n = 0;
double   g_host_total_us = 0, g_host_drv_us = 0;
inline void host_note(const char* name, double total_us, double drv_us) {
    g_host_total_us += total_us; g_host_drv_us += drv_us;
    for (int i = 0; i < g_host_n; ++i)
        if (g_host[i].name == name) { g_host[i].total_us += total_us; g_host[i].drv_us += drv_us; ++g_host[i].n; return; }
    if (g_host_n < 48) g_host[g_host_n++] = HostCost{name, total_us, drv_us, 1};
}
}  // namespace

void OpenCLContext::hostProfReset() { g_host_n = 0; g_host_total_us = g_host_drv_us = 0; }
// Totals for the render summary: dispatches per frame, and host us per dispatch. The Engine report
// card reads these, so the two numbers that decide every optimisation are on the front screen
// instead of buried in a log.
void OpenCLContext::hostProfTotals(int frames, long* disp_per_frame, double* us_per_dispatch) {
    long n = 0;
    for (int i = 0; i < g_host_n; ++i) n += g_host[i].n;
    if (disp_per_frame)   *disp_per_frame   = frames > 0 ? n / frames : 0;
    if (us_per_dispatch)  *us_per_dispatch  = n > 0 ? g_host_total_us / (double)n : 0.0;
}

void OpenCLContext::hostProfReport(double frame_ms, int frames) {
    if (g_host_n == 0 || frames <= 0) return;
    std::vector<HostCost> v(g_host, g_host + g_host_n);
    std::sort(v.begin(), v.end(), [](const HostCost& a, const HostCost& b){ return a.total_us > b.total_us; });
    const double enq_ms = g_host_total_us / 1000.0, drv_ms = g_host_drv_us / 1000.0;
    std::fprintf(stderr,
        "\nHOST_COST over %d frames: frame_host=%.0f ms | in profEnqueue=%.0f ms "
        "(driver=%.0f ms, wrapper=%.0f ms) | outside=%.0f ms (clSetKernelArg, key building, op glue)\n",
        frames, frame_ms, enq_ms, drv_ms, enq_ms - drv_ms, frame_ms - enq_ms);
    // Top 8 only. The tail is a long list of kernels worth under 1% each, and printing all of them
    // is what turns this report into something you have to scroll past.
    std::fprintf(stderr, "  %-18s %9s %8s %9s %7s\n", "kernel", "total_ms", "calls", "per_frame", "us/call");
    const int show = (int)v.size() < 8 ? (int)v.size() : 8;
    for (int i = 0; i < show; ++i) {
        const auto& x = v[i];
        std::fprintf(stderr, "  %-18s %9.1f %8ld %9.0f %7.1f\n",
                     x.name ? x.name : "?", x.total_us/1000.0, x.n,
                     (double)x.n/(double)frames, x.n ? x.total_us/(double)x.n : 0.0);
    }
    if ((int)v.size() > show) {
        double rest = 0; long restn = 0;
        for (size_t i = show; i < v.size(); ++i) { rest += v[i].total_us; restn += v[i].n; }
        std::fprintf(stderr, "  %-18s %9.1f %8ld\n", "(others)", rest/1000.0, restn);
    }
    std::fflush(stderr);
}

namespace { thread_local cl_command_queue t_thread_queue = nullptr; }
void OpenCLContext::setThreadQueue(cl_command_queue q) { t_thread_queue = q; }
cl_command_queue OpenCLContext::threadQueue() { return t_thread_queue; }
cl_command_queue OpenCLContext::activeQueue() const {
    if (t_thread_queue) return t_thread_queue;
    return active_queue_ ? active_queue_ : queue_;
}

cl_int OpenCLContext::profEnqueue(cl_kernel k, cl_uint dim, const size_t* gws, const size_t* lws, const char* name) {
    const auto t_prof_entry = std::chrono::steady_clock::now();
    cl_command_queue q = activeQueue();   // per-thread first, then the shared overlap target
    nnopt_kinst_note_dispatch();   // reconcile against the per-site kernel lookups (see utils.h)

    size_t tuned[3] = {0,0,0};
    if (!lws && dim >= 1 && dim <= 3 && lws_tuning_enabled()) {
        // Key on the PROTOTYPE. Per-site instances are distinct cl_kernel objects, so keying on the
        // instance turned one tuned entry into 519 misses and silently disabled workgroup tuning
        // for the entire AR — every site fell back to the driver default.
        LwsKey key{}; key.kernel = (const void*)nnopt_kinst_proto_for(k); key.dim = dim;
        for (cl_uint i = 0; i < dim; ++i) key.g[i] = gws[i];
        // Two host threads reach this map now (AR and codec). A std::map insert racing a lookup
        // corrupts the tree; the critical section is a pointer compare, so the lock costs nothing
        // next to the dispatch it guards.
        std::lock_guard<std::mutex> lk(g_lws_mu);
        auto it = g_lws_cache.find(key);
        if (it == g_lws_cache.end()) {
            // Only sweep inside the warm-up window; outside it, record "driver default" for this
            // shape so a real request never re-runs a live kernel to time it.
            LwsChoice v = g_lws_window ? tune_lws(k, dim, gws, name) : LwsChoice{};
            it = g_lws_cache.emplace(key, v).first;
        }
        if (it->second.use) { for (cl_uint i = 0; i < dim; ++i) tuned[i] = it->second.l[i]; lws = tuned; }
    }

    if (!profEnabled() && !g_arprof_on) {
        const auto h0 = std::chrono::steady_clock::now();
        const cl_int e = clEnqueueNDRangeKernel(q, k, dim, nullptr, gws, lws, 0, nullptr, nullptr);
        const auto h1 = std::chrono::steady_clock::now();
        host_note(name, std::chrono::duration<double,std::micro>(h1 - t_prof_entry).count(),
                        std::chrono::duration<double,std::micro>(h1 - h0).count());
        return e;
    }
    cl_event ev = nullptr;
    cl_int e = clEnqueueNDRangeKernel(q, k, dim, nullptr, gws, lws, 0, nullptr, &ev);
    if (e != CL_SUCCESS) {
        // Instrumentation must never change the result it measures. Asking for an event costs one
        // live cl_event per dispatch until the report is read, and a driver has a ceiling on those:
        // the 620 starts refusing at dispatch ~513 of the AR frame's 519, so the last few GEMVs of
        // frame 0 were simply NOT DISPATCHED — a null propagated into LayerNorm and the frame the
        // whole render conditions on was garbage. Retry unprofiled: the kernel runs, and it is
        // missing from the breakdown instead of missing from the model.
        static bool warned = false;
        if (!warned) {
            warned = true;
            std::fprintf(stderr, "PROF event-limited at '%s' (err=%d) — re-dispatching without an "
                                 "event; this kernel is absent from the profile, not from the run\n",
                         name ? name : "?", (int)e);
            std::fflush(stderr);
        }
        return clEnqueueNDRangeKernel(q, k, dim, nullptr, gws, lws, 0, nullptr, nullptr);
    }
    if (ev) {
        if (g_arprof_on) g_arprof_events.push_back({ev, name});
        else             g_prof_events.push_back({ev, name});
    }
    return e;
}

// Times each legal candidate on the real dispatch and keeps the fastest. Runs once per
// (kernel, global-size) and the result is cached for the process, so the cost is bounded by the
// number of distinct shapes, not by the number of calls.
OpenCLContext::LwsChoice OpenCLContext::tune_lws(cl_kernel k, cl_uint dim, const size_t* gws, const char* name) {
    LwsChoice out;   // .use == false ⇒ keep the driver default
    cl_command_queue q = active_queue_ ? active_queue_ : queue_;

    size_t reqd[3] = {0,0,0};
    clGetKernelWorkGroupInfo(k, device_, CL_KERNEL_COMPILE_WORK_GROUP_SIZE, sizeof(reqd), reqd, nullptr);
    if (reqd[0] || reqd[1] || reqd[2]) return out;         // kernel pins its own size
    cl_ulong lmem = 0;
    clGetKernelWorkGroupInfo(k, device_, CL_KERNEL_LOCAL_MEM_SIZE, sizeof(lmem), &lmem, nullptr);
    if (lmem != 0) return out;                              // uses __local ⇒ shape-dependent
    size_t kmax = 0;
    clGetKernelWorkGroupInfo(k, device_, CL_KERNEL_WORK_GROUP_SIZE, sizeof(kmax), &kmax, nullptr);
    if (kmax == 0) return out;

    // Shapes worth trying. 1D: plain widths. 2D: vary how the group is split between the fast and
    // slow axis, which 6.1.5.2 calls out as mattering as much as the total size.
    static const size_t cand1[][3] = {{64,1,1},{128,1,1},{256,1,1},{32,1,1}};
    static const size_t cand2[][3] = {{64,1,1},{32,2,1},{16,4,1},{8,8,1},{128,1,1},{256,1,1}};
    const size_t (*cands)[3] = (dim == 1) ? cand1 : cand2;
    const int ncand = (dim == 1) ? 4 : 6;

    double best_ms = 0.0; bool have_best = false;
    for (int c = 0; c < ncand; ++c) {
        size_t l[3] = {cands[c][0], cands[c][1], cands[c][2]};
        size_t total = 1; bool ok = true;
        for (cl_uint i = 0; i < dim; ++i) {
            if (l[i] == 0 || gws[i] % l[i] != 0) { ok = false; break; }
            total *= l[i];
        }
        if (!ok || total > kmax) continue;
        // 3 iterations, take the total: enough to see past a single noisy launch without turning
        // startup into a benchmark suite.
        if (clEnqueueNDRangeKernel(q, k, dim, nullptr, gws, l, 0, nullptr, nullptr) != CL_SUCCESS) continue;
        clFinish(q);
        const auto t0 = std::chrono::steady_clock::now();
        for (int r = 0; r < 3; ++r) clEnqueueNDRangeKernel(q, k, dim, nullptr, gws, l, 0, nullptr, nullptr);
        clFinish(q);
        const double ms = std::chrono::duration<double, std::milli>(
                              std::chrono::steady_clock::now() - t0).count();
        if (!have_best || ms < best_ms) {
            have_best = true; best_ms = ms;
            out.use = true; for (int i = 0; i < 3; ++i) out.l[i] = l[i];
        }
    }
    if (lws_verbose() && !out.use) {
        size_t reqd_dbg[3] = {0,0,0}; cl_ulong lmem_dbg = 0; size_t kmax_dbg = 0;
        clGetKernelWorkGroupInfo(k, device_, CL_KERNEL_COMPILE_WORK_GROUP_SIZE, sizeof(reqd_dbg), reqd_dbg, nullptr);
        clGetKernelWorkGroupInfo(k, device_, CL_KERNEL_LOCAL_MEM_SIZE, sizeof(lmem_dbg), &lmem_dbg, nullptr);
        clGetKernelWorkGroupInfo(k, device_, CL_KERNEL_WORK_GROUP_SIZE, sizeof(kmax_dbg), &kmax_dbg, nullptr);
        std::fprintf(stderr, "LWS_TUNE %-18s gws=[%zu,%zu] SKIPPED (reqd=%zu lmem=%llu kmax=%zu)\n",
                     name ? name : "?", gws[0], dim > 1 ? gws[1] : 1,
                     reqd_dbg[0], (unsigned long long)lmem_dbg, kmax_dbg);
        std::fflush(stderr);
    }
    if (out.use && lws_verbose()) {
        std::fprintf(stderr, "LWS_TUNE %-18s gws=[%zu,%zu] -> lws=[%zu,%zu] (%.3f ms/3)\n",
                     name ? name : "?", gws[0], dim > 1 ? gws[1] : 1,
                     out.l[0], dim > 1 ? out.l[1] : 1, best_ms);
        std::fflush(stderr);
    }
    return out;
}


// ── cl_qcom_recordable_queues ───────────────────────────────────────────────────────────────────
#define NNOPT_CL_QUEUE_RECORDABLE_QCOM ((cl_command_queue_properties)0x40000000)

void OpenCLContext::load_record_fns() {
    // Resolve through clGetExtensionFunctionAddressForPlatform, NOT dlsym. Edgi ships its own
    // libOpenCL forwarding shim ahead of the vendor driver on LD_LIBRARY_PATH, and that shim
    // exports only the functions it was written to forward — a dlsym for a QCOM entry point finds
    // nothing there. The extension-address query IS forwarded, and it returns a pointer straight
    // into the real driver. (pocket-tts uses dlsym because it runs as a bare executable.)
    auto get = [&](const char* n) -> void* {
        if (void* p = clGetExtensionFunctionAddressForPlatform(platform_, n)) return p;
        return dlsym(RTLD_DEFAULT, n);   // bare-executable fallback
    };
    fn_new_recording_     = (clNewRecordingQCOM_fn)     get("clNewRecordingQCOM");
    fn_end_recording_     = (clEndRecordingQCOM_fn)     get("clEndRecordingQCOM");
    fn_release_recording_ = (clReleaseRecordingQCOM_fn) get("clReleaseRecordingQCOM");
    fn_enqueue_recording_ = (clEnqueueRecordingQCOM_fn) get("clEnqueueRecordingQCOM");
    record_fns_loaded_ = fn_new_recording_ && fn_end_recording_ &&
                         fn_release_recording_ && fn_enqueue_recording_;
    std::fprintf(stderr, "RECORDQ entry points: %s (new=%p end=%p rel=%p enq=%p)\n",
                 record_fns_loaded_ ? "available" : "NOT available",
                 (void*)fn_new_recording_, (void*)fn_end_recording_,
                 (void*)fn_release_recording_, (void*)fn_enqueue_recording_);
    std::fflush(stderr);
}

cl_command_queue OpenCLContext::create_recordable_queue() {
    if (!record_fns_loaded_) return nullptr;
    cl_int err = CL_SUCCESS;
    // Bit 30 ALONE — combining it with PROFILING_ENABLE fails on this driver (probe-validated).
    cl_command_queue q = clCreateCommandQueue(context_, device_, NNOPT_CL_QUEUE_RECORDABLE_QCOM, &err);
    if (err != CL_SUCCESS) {
        std::fprintf(stderr, "RECORDQ create failed (err=%d)\n", (int)err);
        return nullptr;
    }
    return q;
}
cl_recording_qcom OpenCLContext::new_recording(cl_command_queue q) const {
    if (!record_fns_loaded_ || !q) return nullptr;
    cl_int err = CL_SUCCESS;
    cl_recording_qcom rec = fn_new_recording_(q, &err);
    if (err != CL_SUCCESS) { std::fprintf(stderr, "RECORDQ new_recording err=%d\n", (int)err); return nullptr; }
    return rec;
}
cl_int OpenCLContext::end_recording(cl_recording_qcom rec) const {
    if (!record_fns_loaded_ || !rec) return CL_INVALID_VALUE;
    return fn_end_recording_(rec);
}
cl_int OpenCLContext::release_recording(cl_recording_qcom rec) const {
    if (!record_fns_loaded_ || !rec) return CL_INVALID_VALUE;
    return fn_release_recording_(rec);
}
cl_int OpenCLContext::enqueue_recording(cl_command_queue live_q, cl_recording_qcom rec,
                                        size_t num_args, const cl_array_arg_qcom* args) const {
    if (!record_fns_loaded_ || !rec) return CL_INVALID_VALUE;
    return fn_enqueue_recording_(live_q, rec, num_args, args,
                                 0, nullptr, 0, nullptr, 0, nullptr, 0, nullptr, nullptr);
}

// Does a recordable queue EXECUTE what it records, or only record it? Everything downstream
// depends on the answer: if capture only records, the captured frame produced no results and the
// AR must replay once for that frame or leave a hole in its KV cache and token grid; if capture
// also executes, that extra replay would double-apply the frame (frame_embed advances `ti`, so the
// sequence is NOT idempotent). Both readings are defensible from the extension docs, so this asks
// the device instead of assuming. One dispatch, once per process.
int OpenCLContext::record_capture_executes() {
    static int cached = -1;
    if (cached >= 0) return cached;
    if (!record_fns_loaded_) return -1;

    const char* src =
        "__kernel void rec_exec_probe(__global int* out){ out[get_global_id(0)] = 4242; }\n";
    cl_program p = build_program(src);
    if (!p) return -1;
    cl_int err = CL_SUCCESS;
    cl_kernel k = clCreateKernel(p, "rec_exec_probe", &err);
    if (!k) return -1;

    const int sentinel = -1;
    cl_mem buf = clCreateBuffer(context_, CL_MEM_READ_WRITE, sizeof(int), nullptr, &err);
    cl_command_queue rq = create_recordable_queue();
    if (!buf || !rq) { if (buf) clReleaseMemObject(buf); if (rq) clReleaseCommandQueue(rq);
                       clReleaseKernel(k); return -1; }
    clEnqueueWriteBuffer(queue_, buf, CL_TRUE, 0, sizeof(int), &sentinel, 0, nullptr, nullptr);
    clSetKernelArg(k, 0, sizeof(cl_mem), &buf);

    int result = -1;
    cl_recording_qcom rec = new_recording(rq);
    if (rec) {
        const size_t g = 1;
        clEnqueueNDRangeKernel(rq, k, 1, nullptr, &g, nullptr, 0, nullptr, nullptr);
        if (end_recording(rec) == CL_SUCCESS) {
            // Read WITHOUT replaying. If the value changed, the capture executed. Drain BOTH
            // queues: finishing only queue_ would not wait for rq, so a capture that does execute
            // (asynchronously) could still read as "does not execute" and earn a duplicate replay.
            clFinish(rq);
            clFinish(queue_);
            int seen = sentinel;
            clEnqueueReadBuffer(queue_, buf, CL_TRUE, 0, sizeof(int), &seen, 0, nullptr, nullptr);
            result = (seen == 4242) ? 1 : 0;
            std::fprintf(stderr, "RECORDQ capture %s the work it records\n",
                         result ? "EXECUTES" : "does NOT execute (replay needed for the captured frame)");
            std::fflush(stderr);
        }
        release_recording(rec);
    }
    clReleaseCommandQueue(rq);
    clReleaseMemObject(buf);
    clReleaseKernel(k);
    if (result >= 0) cached = result;
    return result;
}

// Capture a trivial kernel, replay it, and check the replay produced the same bytes as a live
// dispatch — and that an argument override actually takes effect. Recording silently doing nothing
// (or replaying with stale args) would look like a speedup and corrupt the audio, so nothing gets
// built on top of this until the device has demonstrated both properties.
bool OpenCLContext::record_probe() {
    if (!record_fns_loaded_) return false;
    // `base` comes from a BUFFER, not a scalar arg. That is the mechanism a recording can actually
    // vary: the capture pins the buffer handle, and the value behind it is rewritten between
    // replays. Scalar arg-override via cl_array_arg_qcom returns CL_INVALID_OPERATION (-59) on this
    // driver, and pocket-tts — the working implementation in this repo — never uses it either; it
    // keeps its decode position in a one-int buffer for exactly this reason.
    const char* src =
        "__kernel void rec_probe(__global int* out, __global const int* base){\n"
        "  const int i = get_global_id(0);\n"
        "  out[i] = base[0] + i;\n"
        "}\n";
    cl_program p = build_program(src);
    if (!p) { std::fprintf(stderr, "RECORDQ probe: build failed\n"); return false; }
    cl_int err = CL_SUCCESS;
    cl_kernel k = clCreateKernel(p, "rec_probe", &err);
    if (!k) { std::fprintf(stderr, "RECORDQ probe: kernel failed\n"); return false; }

    const int N = 256;
    cl_mem buf = clCreateBuffer(context_, CL_MEM_READ_WRITE, N * sizeof(int), nullptr, &err);
    cl_command_queue rq = create_recordable_queue();
    if (!buf || !rq) { std::fprintf(stderr, "RECORDQ probe: setup failed\n"); return false; }

    int base = 1000;
    cl_mem baseb = clCreateBuffer(context_, CL_MEM_READ_WRITE, sizeof(int), nullptr, &err);
    if (!baseb) { std::fprintf(stderr, "RECORDQ probe: base buffer failed\n"); return false; }
    clEnqueueWriteBuffer(queue_, baseb, CL_TRUE, 0, sizeof(int), &base, 0, nullptr, nullptr);
    clSetKernelArg(k, 0, sizeof(cl_mem), &buf);
    clSetKernelArg(k, 1, sizeof(cl_mem), &baseb);
    const size_t g = (size_t)N;

    cl_recording_qcom rec = new_recording(rq);
    if (!rec) { clReleaseCommandQueue(rq); return false; }
    const cl_int e_cap = clEnqueueNDRangeKernel(rq, k, 1, nullptr, &g, nullptr, 0, nullptr, nullptr);
    std::fprintf(stderr, "RECORDQ probe: capture enqueue err=%d\n", (int)e_cap);
    if (end_recording(rec) != CL_SUCCESS) {
        std::fprintf(stderr, "RECORDQ probe: end_recording failed\n");
        clReleaseCommandQueue(rq); return false;
    }

    std::vector<int> host(N, -1);
    bool ok = true;

    // Replay unchanged: out[i] must be base + i.
    const cl_int e_rep = enqueue_recording(queue_, rec, 0, nullptr);
    std::fprintf(stderr, "RECORDQ probe: replay err=%d\n", (int)e_rep);
    if (e_rep != CL_SUCCESS) ok = false;
    clFinish(queue_);
    clEnqueueReadBuffer(queue_, buf, CL_TRUE, 0, N * sizeof(int), host.data(), 0, nullptr, nullptr);
    for (int i = 0; i < N && ok; ++i) if (host[i] != base + i) {
        std::fprintf(stderr, "RECORDQ probe: replay mismatch at %d (got %d want %d)\n", i, host[i], base + i);
        ok = false;
    }

    // Rewrite the buffer and replay again: the recorded dispatch must observe the NEW value. This
    // is the property the AR needs in order to advance its frame position across replays.
    const int base2 = 7000;
    if (ok) {
        clEnqueueWriteBuffer(queue_, baseb, CL_TRUE, 0, sizeof(int), &base2, 0, nullptr, nullptr);
        const cl_int e2 = enqueue_recording(queue_, rec, 0, nullptr);
        std::fprintf(stderr, "RECORDQ probe: replay-after-buffer-update err=%d\n", (int)e2);
        if (e2 != CL_SUCCESS) ok = false;
    }
    clFinish(queue_);
    clEnqueueReadBuffer(queue_, buf, CL_TRUE, 0, N * sizeof(int), host.data(), 0, nullptr, nullptr);
    for (int i = 0; i < N && ok; ++i) if (host[i] != base2 + i) {
        std::fprintf(stderr, "RECORDQ probe: buffer-update mismatch at %d (got %d want %d) — a replay "
                             "that cannot see updated buffer contents is useless for the AR\n",
                     i, host[i], base2 + i);
        ok = false;
    }

    std::fprintf(stderr, "RECORDQ probe: %s\n",
                 ok ? "capture+replay OK, and replay observes updated buffer contents" : "FAILED");
    std::fflush(stderr);
    release_recording(rec);
    clReleaseCommandQueue(rq);
    clReleaseMemObject(buf);
    clReleaseMemObject(baseb);
    return ok;
}

void OpenCLContext::profReport() {
    if (!profEnabled() || g_prof_events.empty()) return;
    std::map<std::string, std::pair<double, long>> agg;  // name -> {total_ms, calls}
    for (auto& pe : g_prof_events) {
        if (!pe.ev) continue;   // a slot the callee never filled (e.g. an early-out path)
        clWaitForEvents(1, &pe.ev);
        cl_ulong s = 0, en = 0;
        clGetEventProfilingInfo(pe.ev, CL_PROFILING_COMMAND_START, sizeof(s), &s, nullptr);
        clGetEventProfilingInfo(pe.ev, CL_PROFILING_COMMAND_END, sizeof(en), &en, nullptr);
        agg[pe.name].first += (double)(en - s) / 1e6;
        agg[pe.name].second += 1;
        clReleaseEvent(pe.ev);
    }
    g_prof_events.clear();
    std::vector<std::pair<std::string, std::pair<double, long>>> v(agg.begin(), agg.end());
    std::sort(v.begin(), v.end(), [](const auto& a, const auto& b){ return a.second.first > b.second.first; });
    double tot = 0; for (auto& x : v) tot += x.second.first;
    std::fprintf(stderr, "\n=== GPU PROFILE (cl_event, per-kernel) ===\n");
    std::fprintf(stderr, "  %-24s %10s %8s %8s\n", "kernel", "GPU_ms", "calls", "%");
    for (auto& x : v)
        std::fprintf(stderr, "  %-24s %10.1f %8ld %7.1f%%\n", x.first.c_str(), x.second.first, x.second.second,
                    100.0 * x.second.first / (tot > 0 ? tot : 1));
    std::fprintf(stderr, "  %-24s %10.1f\n", "TOTAL (GPU active)", tot);
    std::fflush(stdout);
}

// ── Component (pipeline-stage) wall-time profiler ────────────────────────────
#include <chrono>
namespace {
std::chrono::steady_clock::time_point g_comp_start;
std::map<std::string, std::pair<double,long>> g_comp;  // name -> {total_ms, count}
}
void OpenCLContext::compTic() {
    if (!profEnabled()) return;
    clFinish(queue_);
    g_comp_start = std::chrono::steady_clock::now();
}
void OpenCLContext::compAccum(const char* name) {
    if (!profEnabled()) return;
    clFinish(queue_);
    auto now = std::chrono::steady_clock::now();
    g_comp[name].first += std::chrono::duration<double, std::milli>(now - g_comp_start).count();
    g_comp[name].second += 1;
    g_comp_start = now;
}
// ── Buffer pool (OPT #2) ─────────────────────────────────────────────────────
#include <unordered_map>
namespace {
    bool g_pool_init = false, g_pool_on = false;
    // A recycled buffer carries the queue that last used it — see the note in pool_alloc.
    struct PoolBuf { cl_mem mem; const void* owner; };
    std::unordered_map<size_t, std::vector<PoolBuf>> g_pool_free;  // bucket bytes -> free buffers
    std::unordered_map<cl_mem, size_t> g_pool_live;              // live buffer -> bucket bytes
    inline size_t pool_bucket(size_t bytes) {
        // round up to next 4 KB page so repeating op sizes share a bucket with ~0 waste.
        const size_t G = 4096;
        return ((bytes + G - 1) / G) * G;
    }
    inline bool pool_enabled() {
        // Default ON now. It was measured "neutral" back when the AR was believed to be GPU-bound;
        // the host profile says otherwise — 851 ms of the AR's 3016 ms is spent OUTSIDE the driver
        // enqueue, and the largest thing there is building and destroying ~300 buffers per frame,
        // 30,000 per render, plus re-setting every kernel argument because each frame's buffers are
        // new objects. Recycling makes both cheap. NNOPT_POOL=0 restores per-frame allocation.
        // Back OFF. Turning it on measured FLAT on host cost — it moved 648 ms out of our
        // allocation code and straight into driver enqueue time, net zero (v21 851/2155 ms vs v22
        // 203/2802 ms, frame_host 3016 vs 3013). Flat is not free: it changes how the driver
        // handles memory on every dispatch, and the best total of the day (3.271 s) was measured
        // with it OFF. Nothing justifies keeping it.
        if (!g_pool_init) { const char* e = std::getenv("NNOPT_POOL"); g_pool_on = (e && e[0] == '1'); g_pool_init = true; }
        return g_pool_on;
    }
}
// NNOPT_ZEROBUF=1 hands back zero-filled scratch instead of whatever the driver had.
// A kernel that fails to write every element of its output then produces a STABLE wrong
// answer instead of one that depends on allocation history — which is how a latent
// uninitialised read is proven (and, if it must ship, contained).
static bool zerobuf_enabled() {
    static int v = -1;
    if (v < 0) { const char* e = std::getenv("NNOPT_ZEROBUF"); v = (e && e[0] != '0') ? 1 : 0; }
    return v != 0;
}

// Both host threads allocate and free, so the free-list and the live map need a lock. It guards
// the MAPS ONLY — never a driver call.
//
// The first version held it across clCreateBuffer, and since the pool is off by default that IS
// clCreateBuffer: every allocation on either thread serialised against the other, with the codec
// taking multi-megabyte allocations while the AR takes small ones constantly. The AR ended up
// waiting on the codec's driver allocations, which is the opposite of what running them on two
// threads was for.
static std::mutex g_pool_mu;

cl_mem pool_alloc(cl_context ctx, cl_mem_flags flags, size_t bytes, void* host_ptr, cl_int* err) {
    if (host_ptr == nullptr && bytes && zerobuf_enabled()) {
        std::vector<unsigned char> zeros(bytes, 0);
        cl_int e = CL_SUCCESS;
        cl_mem m = clCreateBuffer(ctx, (flags | CL_MEM_COPY_HOST_PTR), bytes, zeros.data(), &e);
        if (err) *err = e;
        return m;
    }
    // Only recycle plain scratch buffers; anything with a host ptr or special flags is forwarded.
    const bool poolable = pool_enabled() && host_ptr == nullptr && flags == CL_MEM_READ_WRITE;
    if (!poolable) {
        cl_int e = CL_SUCCESS; cl_mem m = clCreateBuffer(ctx, flags, bytes, host_ptr, &e);
        if (err) *err = e; return m;
    }
    // Recycle ONLY within the queue that last used the buffer. The codec runs on its own thread and
    // its own queue now: handing the AR a buffer the codec's in-flight kernels are still reading is
    // a data race the GPU resolves by corrupting audio, and an in-order queue reusing its OWN buffer
    // is the case that is safe by construction. Keyed on the thread's queue, which is exactly the
    // ownership boundary.
    const void* owner = (const void*)OpenCLContext::threadQueue();
    const size_t bucket = pool_bucket(bytes);
    {   // recycle: map work only
        std::lock_guard<std::mutex> lk(g_pool_mu);
        auto it = g_pool_free.find(bucket);
        if (it != g_pool_free.end()) {
            for (size_t i = it->second.size(); i-- > 0; ) {
                if (it->second[i].owner != owner) continue;
                cl_mem m = it->second[i].mem;
                it->second.erase(it->second.begin() + (long)i);
                g_pool_live[m] = bucket; if (err) *err = CL_SUCCESS; return m;
            }
        }
    }
    cl_int e = CL_SUCCESS;
    cl_mem m = clCreateBuffer(ctx, CL_MEM_READ_WRITE, bucket, nullptr, &e);   // OUTSIDE the lock
    if (err) *err = e;
    if (e == CL_SUCCESS && m) { std::lock_guard<std::mutex> lk(g_pool_mu); g_pool_live[m] = bucket; }
    return m;
}
// While a recording is being captured, the buffers its dispatches reference must stay exclusively
// the AR's for the life of the render: the recording pins HANDLES, and a freed buffer goes back to
// the pool where the codec — running concurrently on the other queue under pipeline=1 — will take
// it and overwrite data the replay still reads. Pinning leaks one frame's AR intermediates on
// purpose; they are a few hundred KB and they are reused by every replay.
static bool g_pool_pin = false;
static std::vector<cl_mem> g_pool_pinned;
// Interception is ON only while the frame is being CAPTURED. It used to stay on for the whole
// render, which meant every codec buffer freed on the other queue was retained too — the codec
// allocates per chunk, so a 100-frame render accumulated far more than "one frame's AR
// intermediates". Turning it off no longer releases anything: the captured frame's buffers must
// outlive every replay, so they are handed back by pool_release_pinned() after the final drain.
void pool_set_pin(bool on) { g_pool_pin = on; }

void pool_release_pinned() {
    for (cl_mem m : g_pool_pinned) pool_free(m);
    g_pool_pinned.clear();
}

void pool_free(cl_mem m) {
    if (!m) return;
    cl_mem to_release = nullptr;
    {
        std::lock_guard<std::mutex> lk(g_pool_mu);
        if (g_pool_pin) { g_pool_pinned.push_back(m); return; }
        auto it = g_pool_live.find(m);
        if (it == g_pool_live.end()) { to_release = m; }        // not ours → real release, below
        else { const size_t bucket = it->second; g_pool_live.erase(it);
               g_pool_free[bucket].push_back(PoolBuf{m, (const void*)OpenCLContext::threadQueue()}); }
    }
    if (to_release) clReleaseMemObject(to_release);             // OUTSIDE the lock
}

void OpenCLContext::compReport() {
    if (!profEnabled() || g_comp.empty()) return;
    std::vector<std::pair<std::string, std::pair<double,long>>> v(g_comp.begin(), g_comp.end());
    std::sort(v.begin(), v.end(), [](const auto& a, const auto& b){ return a.second.first > b.second.first; });
    double tot = 0; for (auto& x : v) tot += x.second.first;
    std::fprintf(stderr, "\n=== E2E COMPONENT PROFILE (wall ms, clFinish-bracketed) ===\n");
    std::fprintf(stderr, "  %-22s %10s %8s %8s\n", "component", "wall_ms", "calls", "%");
    for (auto& x : v)
        std::fprintf(stderr, "  %-22s %10.1f %8ld %7.1f%%\n", x.first.c_str(), x.second.first, x.second.second,
                    100.0 * x.second.first / (tot > 0 ? tot : 1));
    std::fprintf(stderr, "  %-22s %10.1f\n", "TOTAL", tot);
    std::fflush(stdout);
    g_comp.clear();
}
