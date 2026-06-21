#include <chrono>
#include "opencl_context.h"
#include "debug_utils.h"   // NNOPT_ERROR_FMT — used in build_program / build_program_from_file below.
#include <dlfcn.h>         // dlsym — cl_qcom_recordable_queues entry points.

#include <fstream>
#include <sstream>
#include <iostream>
#include <cstring>
#include <cstdio>
#include <cstdint>
#include <vector>
#include <sys/stat.h>

OpenCLContext::OpenCLContext() {}

OpenCLContext::~OpenCLContext() {
    if (utils_program_) { clReleaseProgram(utils_program_); utils_program_ = nullptr; }
    if (queue_) clReleaseCommandQueue(queue_);
    if (context_) clReleaseContext(context_);
}

cl_program OpenCLContext::get_utils_program() {
    if (utils_program_) return utils_program_;
    utils_program_ = build_program_from_file("kernels/utils.cl");
    if (!utils_program_) {
        NNOPT_ERROR("OpenCLContext::get_utils_program: failed to build kernels/utils.cl");
        return nullptr;
    }
    return utils_program_;
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
    // Campaign-2 #7: cl_qcom_perf_hint — ask the Adreno driver for max GPU
    // clocks for this context (DVFS otherwise downclocks mid-decode; the
    // measured 3.75→3.63 tok/s drift across back-to-back runs is the governor,
    // not the kernels). Gated by extension presence; NNOPT_PERF_HINT=0 disables.
    // Fallback: if clCreateContext rejects the property (-30/-59), retry plain.
    bool try_perf_hint = [](){
        const char* e = std::getenv("NNOPT_PERF_HINT");
        return !(e && e[0] == '0');
    }();
    if (try_perf_hint) {
        size_t ext_len = 0;
        clGetDeviceInfo(device_, CL_DEVICE_EXTENSIONS, 0, nullptr, &ext_len);
        std::string exts(ext_len, '\0');
        clGetDeviceInfo(device_, CL_DEVICE_EXTENSIONS, ext_len, &exts[0], nullptr);
        try_perf_hint = exts.find("cl_qcom_perf_hint") != std::string::npos;
        // NNOPT_PRINT_EXT=1: dump the full extension + version info once (used
        // to ground the Adreno-guide optimization audit in driver facts).
        if (const char* pe = std::getenv("NNOPT_PRINT_EXT"); pe && pe[0] == '1') {
            char ver[256] = {0};
            clGetDeviceInfo(device_, CL_DEVICE_VERSION, sizeof(ver) - 1, ver, nullptr);
            fprintf(stderr, "CL_VERSION: %s\nCL_EXTENSIONS: %s\n", ver, exts.c_str());
        }
    }
    if (try_perf_hint) {
        #define NNOPT_CL_CONTEXT_PERF_HINT_QCOM 0x40C2
        #define NNOPT_CL_PERF_HINT_HIGH_QCOM    0x40C3
        const cl_context_properties props[] = {
            NNOPT_CL_CONTEXT_PERF_HINT_QCOM, NNOPT_CL_PERF_HINT_HIGH_QCOM, 0
        };
        context_ = clCreateContext(props, 1, &device_, nullptr, nullptr, &err);
        if (err == CL_SUCCESS) {
            fprintf(stderr, "PERF_HINT: cl_qcom_perf_hint HIGH active\n");
        } else {
            fprintf(stderr, "PERF_HINT: clCreateContext with perf hint failed (%d), plain retry\n", (int)err);
            context_ = nullptr;
        }
    }
    if (!context_) {
        context_ = clCreateContext(nullptr, 1, &device_, nullptr, nullptr, &err);
        if (err != CL_SUCCESS) return false;
    }
    if (!context_) return false;

    queue_ = clCreateCommandQueue(context_, device_, CL_QUEUE_PROFILING_ENABLE, &err);
    if (err != CL_SUCCESS) return false;

    // ── One-time device banner (matches mms-tts format). Always shown — it's
    // small + useful. Debug builds additionally dump the full extensions list.
    {
        char platform_name[128] = {0};
        char device_name[128]   = {0};
        char device_version[128]= {0};
        char driver_version[256]= {0};
        char ext_buf[8192]      = {0};
        cl_uint cu = 0;
        size_t max_wg = 0;
        cl_ulong gmem = 0, lmem = 0;
        cl_uint clock_mhz = 0;
        cl_platform_id platform = nullptr;
        clGetDeviceInfo(device_, CL_DEVICE_PLATFORM, sizeof(platform), &platform, nullptr);
        if (platform) clGetPlatformInfo(platform, CL_PLATFORM_NAME, sizeof(platform_name), platform_name, nullptr);
        clGetDeviceInfo(device_, CL_DEVICE_NAME,                 sizeof(device_name),    device_name,    nullptr);
        clGetDeviceInfo(device_, CL_DEVICE_VERSION,              sizeof(device_version), device_version, nullptr);
        clGetDeviceInfo(device_, CL_DRIVER_VERSION,              sizeof(driver_version), driver_version, nullptr);
        clGetDeviceInfo(device_, CL_DEVICE_MAX_COMPUTE_UNITS,    sizeof(cu),       &cu,       nullptr);
        clGetDeviceInfo(device_, CL_DEVICE_MAX_WORK_GROUP_SIZE,  sizeof(max_wg),   &max_wg,   nullptr);
        clGetDeviceInfo(device_, CL_DEVICE_GLOBAL_MEM_SIZE,      sizeof(gmem),     &gmem,     nullptr);
        clGetDeviceInfo(device_, CL_DEVICE_LOCAL_MEM_SIZE,       sizeof(lmem),     &lmem,     nullptr);
        clGetDeviceInfo(device_, CL_DEVICE_MAX_CLOCK_FREQUENCY,  sizeof(clock_mhz),&clock_mhz,nullptr);
        clGetDeviceInfo(device_, CL_DEVICE_EXTENSIONS, sizeof(ext_buf) - 1, ext_buf, nullptr);
        const bool has_fp16    = std::strstr(ext_buf, "cl_khr_fp16")          != nullptr;
        const bool has_perfhnt = std::strstr(ext_buf, "cl_qcom_perf_hint")    != nullptr;
        const bool has_record  = std::strstr(ext_buf, "cl_qcom_recordable_queues") != nullptr;
        const bool has_dotp8   = std::strstr(ext_buf, "cl_qcom_dot_product8") != nullptr;
        // Adreno 619 (SM6375) ADVERTISES cl_qcom_reqd_sub_group_size in CL_DEVICE_EXTENSIONS but its
        // compiler rejects the pragma (clBuildProgram -11). Advertisement != usability — so compile-probe it.
        auto ext_compiles = [&](const char* src) -> bool {
            cl_int e; const char* s = src;
            cl_program p = clCreateProgramWithSource(context_, 1, &s, nullptr, &e);
            if (e != CL_SUCCESS || !p) return false;
            e = clBuildProgram(p, 1, &device_, "", nullptr, nullptr);
            clReleaseProgram(p);
            return e == CL_SUCCESS;
        };
        const bool has_reqdsg  = (std::strstr(ext_buf, "cl_qcom_reqd_sub_group_size") != nullptr) && ext_compiles(
            "#pragma OPENCL EXTENSION cl_qcom_reqd_sub_group_size : enable\n"
            "__attribute__((qcom_reqd_sub_group_size(\"full\"))) __kernel void p(){}\n");
        fprintf(stderr, "── OpenCL device ────────────────────────────────────────────\n");
        fprintf(stderr, "  platform        %s\n", platform_name);
        fprintf(stderr, "  device          %s\n", device_name);
        fprintf(stderr, "  version         %s\n", device_version);
        fprintf(stderr, "  driver          %s\n", driver_version);
        fprintf(stderr, "  compute_units   %u\n", (unsigned)cu);
        fprintf(stderr, "  max_clock_MHz   %u\n", (unsigned)clock_mhz);
        fprintf(stderr, "  max_workgroup   %zu\n", max_wg);
        fprintf(stderr, "  global_mem      %.0f MB\n", (double)gmem / (1024.0 * 1024.0));
        fprintf(stderr, "  local_mem       %.0f KB\n", (double)lmem / 1024.0);
        fprintf(stderr, "  cl_khr_fp16            %s\n", has_fp16    ? "yes" : "no");
        fprintf(stderr, "  qcom_perf_hint         %s\n", has_perfhnt ? "yes" : "no");
        fprintf(stderr, "  qcom_recordable_queues %s\n", has_record  ? "yes" : "no");
        fprintf(stderr, "  qcom_dot_product8      %s\n", has_dotp8   ? "yes" : "no");
        fprintf(stderr, "  qcom_reqd_sub_group_size %s\n", has_reqdsg ? "yes" : "no");
        fprintf(stderr, "─────────────────────────────────────────────────────────────\n");
        fprintf(stderr, "  cl_device_extensions   %s\n", ext_buf);
        fflush(stderr);
    }

    // cl_qcom_recordable_queues entry points (dlsym — not exported by the CL
    // headers). All four must resolve for the record/replay decode path; when
    // absent (non-Adreno) has_recordable_queues() stays false and callers use
    // live dispatch.
    fn_new_recording_     = (clNewRecordingQCOM_fn)    dlsym(RTLD_DEFAULT, "clNewRecordingQCOM");
    fn_end_recording_     = (clEndRecordingQCOM_fn)    dlsym(RTLD_DEFAULT, "clEndRecordingQCOM");
    fn_release_recording_ = (clReleaseRecordingQCOM_fn)dlsym(RTLD_DEFAULT, "clReleaseRecordingQCOM");
    fn_enqueue_recording_ = (clEnqueueRecordingQCOM_fn)dlsym(RTLD_DEFAULT, "clEnqueueRecordingQCOM");
    record_fns_loaded_ = fn_new_recording_ && fn_end_recording_ &&
                         fn_release_recording_ && fn_enqueue_recording_;
    return true;
}

// ── cl_qcom_recordable_queues helpers ────────────────────────────────────────
#define NNOPT_CL_QUEUE_RECORDABLE_QCOM ((cl_command_queue_properties)0x40000000)

cl_command_queue OpenCLContext::create_recordable_queue() {
    if (!record_fns_loaded_) return nullptr;
    cl_int err = CL_SUCCESS;
    // Bit 30 ALONE — combining with PROFILING_ENABLE fails on this driver
    // (probe-validated recipe).
    cl_command_queue q = clCreateCommandQueue(context_, device_,
                                              NNOPT_CL_QUEUE_RECORDABLE_QCOM, &err);
    if (err != CL_SUCCESS) {
        NNOPT_ERROR_FMT("recordable queue create failed: %d", (int)err);
        return nullptr;
    }
    return q;
}

cl_recording_qcom OpenCLContext::new_recording(cl_command_queue q) const {
    if (!record_fns_loaded_ || !q) return nullptr;
    cl_int err = CL_SUCCESS;
    cl_recording_qcom rec = fn_new_recording_(q, &err);
    if (err != CL_SUCCESS) { NNOPT_ERROR_FMT("clNewRecordingQCOM failed: %d", (int)err); return nullptr; }
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
    return fn_enqueue_recording_(live_q, rec,
                                 num_args, args,
                                 0, nullptr, 0, nullptr, 0, nullptr,
                                 0, nullptr, nullptr);
}

// FNV-1a 64-bit — cache key hashing for the program binary cache.
static uint64_t nnopt_fnv1a(const void* data, size_t n, uint64_t h) {
    const uint8_t* p = (const uint8_t*)data;
    for (size_t i = 0; i < n; ++i) { h ^= p[i]; h *= 1099511628211ULL; }
    return h;
}

cl_program OpenCLContext::build_program(const std::string& source, const std::string& options) {
    cl_int err;

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

    // Campaign-2 Opt #1: compiler optimization flags (Qualcomm 80-NB295-11).
    // -cl-mad-enable fuses the GEMV's a*b+c into single MADs; -cl-fast-relaxed-math
    // speeds exp/tanh/rsqrt in LN/softmax/GELU. NNOPT_CL_FASTMATH gates for A/B:
    //   unset or "1" → both flags (default)
    //   "mad"        → -cl-mad-enable only (fallback if fast-math breaks tf-depth)
    //   "0"          → none (baseline A/B leg)
    {
        static const char* fm = std::getenv("NNOPT_CL_FASTMATH");
        const std::string mode = fm ? fm : "1";
        if (mode == "1") {
            effective_options += " -cl-mad-enable -cl-fast-relaxed-math";
        } else if (mode == "mad") {
            effective_options += " -cl-mad-enable";
        }
    }

    // ── On-disk program binary cache (.clcache/<hash>.bin) ──────────────────
    // The Adreno driver's own shader cache needs an Android app cache dir —
    // raw CLI processes in /data/local/tmp don't have one, so EVERY run paid
    // full source compilation (the megakernel program alone is multiple
    // seconds of the prefill segment). Key = source + build options + device
    // + driver version (driver updates invalidate stale binaries). Any load
    // failure falls back to source compilation. NNOPT_CL_CACHE=0 disables.
    static const bool cache_on = [](){
        const char* e = std::getenv("NNOPT_CL_CACHE");
        return !(e && e[0] == '0');
    }();
    std::string cache_path;
    if (cache_on) {
        char devname[256] = {0}, drvver[256] = {0};
        clGetDeviceInfo(device_, CL_DEVICE_NAME, sizeof(devname) - 1, devname, nullptr);
        clGetDeviceInfo(device_, CL_DRIVER_VERSION, sizeof(drvver) - 1, drvver, nullptr);
        uint64_t h = nnopt_fnv1a(source.data(), source.size(), 1469598103934665603ULL);
        h = nnopt_fnv1a(effective_options.data(), effective_options.size(), h);
        h = nnopt_fnv1a(devname, strlen(devname), h);
        h = nnopt_fnv1a(drvver, strlen(drvver), h);
        char fn[64];
        snprintf(fn, sizeof(fn), ".clcache/%016llx.bin", (unsigned long long)h);
        cache_path = fn;
        if (FILE* f = fopen(cache_path.c_str(), "rb")) {
            fseek(f, 0, SEEK_END);
            long sz = ftell(f);
            fseek(f, 0, SEEK_SET);
            std::vector<unsigned char> bin((size_t)(sz > 0 ? sz : 0));
            bool read_ok = sz > 0 && fread(bin.data(), 1, bin.size(), f) == bin.size();
            fclose(f);
            if (read_ok) {
                const unsigned char* bp = bin.data();
                const size_t bs = bin.size();
                cl_int bin_status = CL_SUCCESS;
                cl_program p = clCreateProgramWithBinary(context_, 1, &device_, &bs, &bp,
                                                         &bin_status, &err);
                if (p && err == CL_SUCCESS && bin_status == CL_SUCCESS &&
                    clBuildProgram(p, 1, &device_, "", nullptr, nullptr) == CL_SUCCESS) {
                    return p;   // cache hit — no source compilation
                }
                if (p) clReleaseProgram(p);
                fprintf(stderr, "CL_CACHE: stale/unloadable binary %s — recompiling\n", cache_path.c_str());
            }
        }
    }

    const char* src_ptr = source.c_str();
    size_t src_len = source.size();
    cl_program program = clCreateProgramWithSource(context_, 1, &src_ptr, &src_len, &err);
    if (err != CL_SUCCESS) {
        NNOPT_ERROR_FMT("clCreateProgramWithSource failed (err=%d)", (int)err);
        return nullptr;
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

    // Save the compiled binary for next run (best-effort; failures are silent
    // beyond the note — the cache is an accelerator, never a dependency).
    if (cache_on && !cache_path.empty()) {
        size_t bin_size = 0;
        if (clGetProgramInfo(program, CL_PROGRAM_BINARY_SIZES, sizeof(bin_size), &bin_size, nullptr) == CL_SUCCESS && bin_size > 0) {
            std::vector<unsigned char> bin(bin_size);
            unsigned char* bp = bin.data();
            if (clGetProgramInfo(program, CL_PROGRAM_BINARIES, sizeof(bp), &bp, nullptr) == CL_SUCCESS) {
                mkdir(".clcache", 0755);   // EEXIST is fine
                if (FILE* f = fopen(cache_path.c_str(), "wb")) {
                    fwrite(bin.data(), 1, bin.size(), f);
                    fclose(f);
                }
            }
        }
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
    // TTFT breakdown: per-program build wall time (covers .clcache hits AND
    // misses — a "hit" that still costs hundreds of ms is a finding).
    const auto b0 = std::chrono::steady_clock::now();
    cl_program prog = build_program(buffer.str(), options);
    if (nnopt_ttft_trace_enabled()) fprintf(stderr, "TTFT_TRACE [%.0f] program_build %s %.0f ms\n", nnopt_uptime_ms(), path.c_str(),
            std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - b0).count());
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
