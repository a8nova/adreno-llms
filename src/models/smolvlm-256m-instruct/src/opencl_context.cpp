#include "opencl_context.h"
#include "debug_utils.h"   // NNOPT_ERROR_FMT — used in build_program / build_program_from_file below.

#include <fstream>
#include <sstream>
#include <iostream>
#include <cstring>
#include <cstdlib>
#include <cstdio>
#include <cstdint>
#include <sys/stat.h>   // mkdir — kernel-binary cache directory

// Qualcomm extension tokens (cl_qcom_perf_hint + cl_qcom_priority_hint).
// Defined inline because vanilla OpenCL headers don't ship the Adreno ext tokens.
// Values from the Qualcomm OpenCL SDK header `cl_ext_qcom.h`.
#ifndef CL_CONTEXT_PERF_HINT_QCOM
#define CL_CONTEXT_PERF_HINT_QCOM        0x40C2
#define CL_PERF_HINT_HIGH_QCOM           0x40C3
#define CL_PERF_HINT_NORMAL_QCOM         0x40C4
#define CL_PERF_HINT_LOW_QCOM            0x40C5
#endif
#ifndef CL_CONTEXT_PRIORITY_HINT_QCOM
#define CL_CONTEXT_PRIORITY_HINT_QCOM    0x40C9
#define CL_PRIORITY_HINT_HIGH_QCOM       0x40CA
#define CL_PRIORITY_HINT_NORMAL_QCOM     0x40CB
#define CL_PRIORITY_HINT_LOW_QCOM        0x40CC
#endif

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
    // Opt #13: request HIGH perf hint + HIGH priority hint on context (Adreno
    // guide §9.1.1, §9.1.2). HIGH perf is documented as the default but
    // setting explicitly keeps boost clocks engaged during bursty decode.
    cl_context_properties ctx_props[] = {
        CL_CONTEXT_PERF_HINT_QCOM,     CL_PERF_HINT_HIGH_QCOM,
        CL_CONTEXT_PRIORITY_HINT_QCOM, CL_PRIORITY_HINT_HIGH_QCOM,
        0
    };
    context_ = clCreateContext(ctx_props, 1, &device_, nullptr, nullptr, &err);
    if (err != CL_SUCCESS) {
        // Fall back to default context if the extension hints aren't accepted.
        fprintf(stderr, "[opencl_context] perf+priority hints rejected (err=%d), retrying without them\n", (int)err);
        context_ = clCreateContext(nullptr, 1, &device_, nullptr, nullptr, &err);
        if (err != CL_SUCCESS) return false;
    }

    // Note: keep CL_QUEUE_PROFILING_ENABLE always on. Empirically (3-run bench
    // 2026-05-16) disabling it REGRESSED decode 4.17 → 3.63 tok/s. Suspected
    // cause: the Adreno driver picks a different (slower) submission path
    // without profiling enabled, possibly because no-event queues drop natural
    // backpressure. Counter-intuitive but reproducible — leave it on.
    queue_ = clCreateCommandQueue(context_, device_, CL_QUEUE_PROFILING_ENABLE, &err);

    // ── Phase 0a: one-time device probe (opt #0a in benchmark.md Round 3 plan).
    // Print device name/version/extensions so we know which Qualcomm-specific
    // extensions are available on this Adreno (e.g. cl_qcom_recordable_queues,
    // cl_qcom_dot_product8, cl_qcom_ml_ops). Tag lines with "DEVICE_PROBE" so
    // benchmark logs can grep them.
    auto get_info_str = [this](cl_device_info param) -> std::string {
        size_t sz = 0;
        clGetDeviceInfo(device_, param, 0, nullptr, &sz);
        if (sz == 0) return "";
        std::vector<char> buf(sz + 1, 0);
        clGetDeviceInfo(device_, param, sz, buf.data(), nullptr);
        return std::string(buf.data());
    };
    auto get_info_uint = [this](cl_device_info param) -> cl_uint {
        cl_uint v = 0;
        clGetDeviceInfo(device_, param, sizeof(v), &v, nullptr);
        return v;
    };
    auto get_info_size = [this](cl_device_info param) -> size_t {
        size_t v = 0;
        clGetDeviceInfo(device_, param, sizeof(v), &v, nullptr);
        return v;
    };
    fprintf(stderr, "DEVICE_PROBE name: %s\n", get_info_str(CL_DEVICE_NAME).c_str());
    fprintf(stderr, "DEVICE_PROBE vendor: %s\n", get_info_str(CL_DEVICE_VENDOR).c_str());
    fprintf(stderr, "DEVICE_PROBE version: %s\n", get_info_str(CL_DEVICE_VERSION).c_str());
    fprintf(stderr, "DEVICE_PROBE driver_version: %s\n", get_info_str(CL_DRIVER_VERSION).c_str());
    fprintf(stderr, "DEVICE_PROBE max_compute_units: %u\n", get_info_uint(CL_DEVICE_MAX_COMPUTE_UNITS));
    fprintf(stderr, "DEVICE_PROBE max_work_group_size: %zu\n", get_info_size(CL_DEVICE_MAX_WORK_GROUP_SIZE));
    fprintf(stderr, "DEVICE_PROBE preferred_vec_width_half: %u\n", get_info_uint(CL_DEVICE_PREFERRED_VECTOR_WIDTH_HALF));
    fprintf(stderr, "DEVICE_PROBE local_mem_size: %zu bytes\n", (size_t)[&](){cl_ulong v=0; clGetDeviceInfo(device_, CL_DEVICE_LOCAL_MEM_SIZE, sizeof(v), &v, nullptr); return v;}());
    fprintf(stderr, "DEVICE_PROBE global_mem_size: %zu bytes\n", (size_t)[&](){cl_ulong v=0; clGetDeviceInfo(device_, CL_DEVICE_GLOBAL_MEM_SIZE, sizeof(v), &v, nullptr); return v;}());
    fprintf(stderr, "DEVICE_PROBE global_mem_cache_size: %zu bytes\n", (size_t)[&](){cl_ulong v=0; clGetDeviceInfo(device_, CL_DEVICE_GLOBAL_MEM_CACHE_SIZE, sizeof(v), &v, nullptr); return v;}());
    fprintf(stderr, "DEVICE_PROBE global_mem_cacheline_size: %u bytes\n", get_info_uint(CL_DEVICE_GLOBAL_MEM_CACHELINE_SIZE));
    // Extensions list can be long; emit on a single line, prefixed.
    fprintf(stderr, "DEVICE_PROBE extensions: %s\n", get_info_str(CL_DEVICE_EXTENSIONS).c_str());
    fflush(stderr);

    return err == CL_SUCCESS;
}

// ──────────────────────────────────────────────
// On-disk OpenCL program-binary cache.
// ──────────────────────────────────────────────
//
// First call with a given (source + options + device + driver) compiles from
// source AND writes the device binary to disk. Subsequent calls reload via
// `clCreateProgramWithBinary`, skipping the slow source-build path. Saves
// ~50–200ms × N kernel builds on cold start (rolls directly into TTFT for the
// VLM workload).
//
// Cache directory: `<cwd>/kernel_cache/`. Filename: 16-hex-digit FNV-1a hash
// of source + " | " + options + " | " + device_name + " | " + driver_version.
// On binary build failure (driver upgrade silently changed something) the
// loader falls through to source compilation and overwrites the bad entry.

static uint64_t nnopt_fnv1a64(const std::string& s) {
    uint64_t h = 14695981039346656037ULL;
    for (unsigned char c : s) { h ^= c; h *= 1099511628211ULL; }
    return h;
}

static std::string nnopt_cache_path_for(uint64_t key) {
    char buf[64];
    std::snprintf(buf, sizeof(buf), "kernel_cache/%016llx.bin", (unsigned long long)key);
    return std::string(buf);
}

cl_program nnopt_build_program_cached(cl_context ctx, cl_device_id dev,
                                      const std::string& source,
                                      const std::string& options) {
    // Probe device once.
    static std::string s_dev_name, s_drv_ver;
    if (s_dev_name.empty()) {
        char b[256] = {0};
        clGetDeviceInfo(dev, CL_DEVICE_NAME, sizeof(b), b, nullptr); s_dev_name = b;
        std::memset(b, 0, sizeof(b));
        clGetDeviceInfo(dev, CL_DRIVER_VERSION, sizeof(b), b, nullptr); s_drv_ver = b;
    }
    const std::string key_str = source + "|" + options + "|" + s_dev_name + "|" + s_drv_ver;
    const uint64_t key = nnopt_fnv1a64(key_str);
    const std::string cache_path = nnopt_cache_path_for(key);

    cl_int err = CL_SUCCESS;
    cl_program program = nullptr;

    // ── Cache HIT path ────────────────────────────────────────────────────
    {
        std::ifstream f(cache_path, std::ios::binary | std::ios::ate);
        if (f) {
            const std::streamsize sz_signed = f.tellg();
            f.seekg(0, std::ios::beg);
            if (sz_signed > 0) {
                const size_t sz = (size_t)sz_signed;
                std::vector<unsigned char> bin(sz);
                if (f.read(reinterpret_cast<char*>(bin.data()), sz)) {
                    cl_int binary_status = CL_SUCCESS;
                    const unsigned char* bp = bin.data();
                    program = clCreateProgramWithBinary(ctx, 1, &dev, &sz, &bp,
                                                        &binary_status, &err);
                    if (err == CL_SUCCESS && binary_status == CL_SUCCESS && program) {
                        err = clBuildProgram(program, 1, &dev, options.c_str(),
                                             nullptr, nullptr);
                        if (err == CL_SUCCESS) {
                            return program;
                        }
                        fprintf(stderr, "[kernel_cache] HIT but build-from-binary failed "
                                        "(err=%d); rebuilding from source: %s\n",
                                (int)err, cache_path.c_str());
                        clReleaseProgram(program);
                        program = nullptr;
                    } else if (program) {
                        clReleaseProgram(program);
                        program = nullptr;
                    }
                }
            }
        }
    }

    // ── Cache MISS path: build from source, then save binary ─────────────
    const char* src_ptr = source.c_str();
    size_t src_len = source.size();
    program = clCreateProgramWithSource(ctx, 1, &src_ptr, &src_len, &err);
    if (err != CL_SUCCESS || !program) {
        NNOPT_ERROR_FMT("clCreateProgramWithSource failed (err=%d)", (int)err);
        return nullptr;
    }

    err = clBuildProgram(program, 1, &dev, options.c_str(), nullptr, nullptr);
    if (err != CL_SUCCESS) {
        NNOPT_ERROR_FMT("clBuildProgram FAILED (err=%d)", (int)err);
        size_t log_size = 0;
        clGetProgramBuildInfo(program, dev, CL_PROGRAM_BUILD_LOG, 0, nullptr, &log_size);
        if (log_size > 0) {
            std::vector<char> log(log_size + 1, 0);
            clGetProgramBuildInfo(program, dev, CL_PROGRAM_BUILD_LOG, log_size,
                                  log.data(), nullptr);
            fprintf(stderr, "OpenCL Build Log: %s\n", log.data());
            fflush(stderr);
        }
        clReleaseProgram(program);
        return nullptr;
    }

    // Persist the device binary for next launch.
    size_t bin_sz = 0;
    if (clGetProgramInfo(program, CL_PROGRAM_BINARY_SIZES, sizeof(bin_sz),
                         &bin_sz, nullptr) == CL_SUCCESS && bin_sz > 0) {
        std::vector<unsigned char> bin(bin_sz);
        unsigned char* binp = bin.data();
        if (clGetProgramInfo(program, CL_PROGRAM_BINARIES, sizeof(binp),
                             &binp, nullptr) == CL_SUCCESS) {
            // Best-effort mkdir + write; failures are non-fatal (next run
            // just falls back to source again).
            mkdir("kernel_cache", 0755);
            std::ofstream out(cache_path, std::ios::binary | std::ios::trunc);
            if (out) out.write(reinterpret_cast<const char*>(bin.data()), bin.size());
        }
    }

    return program;
}

cl_program OpenCLContext::build_program(const std::string& source, const std::string& options) {
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
    // Opt #0b / #8: fast-relaxed-math. Adreno guide §8.2.
    if (effective_options.find("relaxed-math") == std::string::npos) {
        if (!effective_options.empty()) effective_options += " ";
        effective_options += "-cl-fast-relaxed-math";
    }
    // OpenCL 2.0 std needed for cl_khr_subgroups built-ins.
    if (effective_options.find("-cl-std=") == std::string::npos) {
        effective_options += " -cl-std=CL2.0";
    }

    return nnopt_build_program_cached(context_, device_, source, effective_options);
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
