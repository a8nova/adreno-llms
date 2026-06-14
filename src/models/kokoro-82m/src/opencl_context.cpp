#include "opencl_context.h"
#include "debug_utils.h"   // NNOPT_ERROR_FMT — used in build_program / build_program_from_file below.
#include <dlfcn.h>         // dlsym — cl_qcom_recordable_queues entry points.

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

    // cl_qcom_perf_hint: request highest GPU performance level. Vendor
    // extension on Adreno. Probe CL_DEVICE_EXTENSIONS first; if absent,
    // create the context without the hint (don't fail on non-Qualcomm).
    // Values per Qualcomm OpenCL programming guide §9.1.1.
    constexpr cl_context_properties CL_CONTEXT_PERF_HINT_QCOM = 0x40C2;
    constexpr cl_context_properties CL_PERF_HINT_HIGH_QCOM    = 0x40C3;

    size_t ext_len = 0;
    clGetDeviceInfo(device_, CL_DEVICE_EXTENSIONS, 0, nullptr, &ext_len);
    std::string ext_str(ext_len, '\0');
    if (ext_len > 0) clGetDeviceInfo(device_, CL_DEVICE_EXTENSIONS, ext_len, ext_str.data(), nullptr);
    bool has_perf_hint = ext_str.find("cl_qcom_perf_hint") != std::string::npos;

    // NNOPT_PRINT_EXTENSIONS=1: dump the device extension list once (drives
    // which vendor extensions — recordable queues, onchip global memory,
    // subgroup shuffle — the optimization plan can rely on).
    if (const char* pe = std::getenv("NNOPT_PRINT_EXTENSIONS")) {
        if (pe[0] == '1') fprintf(stderr, "CL_DEVICE_EXTENSIONS: %s\n", ext_str.c_str());
    }

    if (has_perf_hint) {
        cl_context_properties props[] = {
            CL_CONTEXT_PERF_HINT_QCOM, CL_PERF_HINT_HIGH_QCOM,
            0
        };
        context_ = clCreateContext(props, 1, &device_, nullptr, nullptr, &err);
        if (err != CL_SUCCESS) {
            // Some drivers list the extension but reject the property at
            // context creation. Fall back to no hint.
            context_ = clCreateContext(nullptr, 1, &device_, nullptr, nullptr, &err);
        }
    } else {
        context_ = clCreateContext(nullptr, 1, &device_, nullptr, nullptr, &err);
    }
    if (err != CL_SUCCESS) return false;

    queue_ = clCreateCommandQueue(context_, device_, CL_QUEUE_PROFILING_ENABLE, &err);
    if (err != CL_SUCCESS) return false;

    // cl_qcom_recordable_queues entry points (dlsym — not exported by the CL
    // headers). All four must resolve for the record/replay generator path;
    // when absent (non-Adreno) has_recordable_queues() stays false and callers
    // use live dispatch.
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
    // (probe-validated recipe from the musicgen port, same device).
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
    if (err != CL_SUCCESS) {
        NNOPT_ERROR_FMT("clNewRecordingQCOM failed: %d", (int)err);
        return nullptr;
    }
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

// ─── Kernel binary cache implementation ─────────────────────────────────
//
// We avoid linking a crypto library; the hash is a simple FNV-1a 64-bit
// digest of (source + options + device name). Collision-resistant enough
// for a per-program cache file name (any change to any input produces a
// different file → no risk of using a stale binary).

#include <cerrno>
#include <chrono>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <sys/stat.h>
#include <vector>

static uint64_t nnopt_fnv1a64(const char* data, size_t n) {
    uint64_t h = 1469598103934665603ULL;
    for (size_t i = 0; i < n; ++i) {
        h ^= (uint64_t)(uint8_t)data[i];
        h *= 1099511628211ULL;
    }
    return h;
}

static std::string nnopt_cache_dir() {
    if (const char* env = std::getenv("NNOPT_KERNEL_CACHE_DIR")) return env;
    return "kernel_cache";
}

static bool nnopt_ensure_dir(const std::string& path) {
    struct stat st;
    if (stat(path.c_str(), &st) == 0) return S_ISDIR(st.st_mode);
    if (mkdir(path.c_str(), 0755) == 0) return true;
    return (errno == EEXIST);
}

static std::vector<unsigned char> nnopt_read_file(const std::string& path) {
    std::vector<unsigned char> out;
    FILE* f = fopen(path.c_str(), "rb");
    if (!f) return out;
    fseek(f, 0, SEEK_END);
    long sz = ftell(f);
    fseek(f, 0, SEEK_SET);
    if (sz > 0) {
        out.resize((size_t)sz);
        size_t r = fread(out.data(), 1, out.size(), f);
        if (r != out.size()) out.clear();
    }
    fclose(f);
    return out;
}

static bool nnopt_write_file(const std::string& path,
                              const unsigned char* data, size_t n) {
    FILE* f = fopen(path.c_str(), "wb");
    if (!f) return false;
    size_t w = fwrite(data, 1, n, f);
    fclose(f);
    return w == n;
}

cl_program nnopt_build_program_cached(cl_context ctx,
                                       cl_device_id dev,
                                       const char* source,
                                       const char* options,
                                       const char* cache_key,
                                       cl_int* out_err) {
    if (!ctx || !dev || !source || !cache_key) {
        if (out_err) *out_err = CL_INVALID_VALUE;
        return nullptr;
    }
    const char* opts = options ? options : "";

    // Get device name for cache key uniqueness.
    char dev_name[256] = {0};
    clGetDeviceInfo(dev, CL_DEVICE_NAME, sizeof(dev_name), dev_name, nullptr);

    // Compute combined hash of source + options + device name.
    std::string blob;
    blob.append(source);
    blob.push_back('|');
    blob.append(opts);
    blob.push_back('|');
    blob.append(dev_name);
    uint64_t h = nnopt_fnv1a64(blob.data(), blob.size());

    std::string dir = nnopt_cache_dir();
    nnopt_ensure_dir(dir);
    char fname[128];
    snprintf(fname, sizeof(fname), "%s/%s.%016llx.bin",
             dir.c_str(), cache_key, (unsigned long long)h);
    std::string cache_path = fname;

    cl_int err = CL_SUCCESS;
    cl_program prog = nullptr;

    // Try loading the cached binary first.
    auto bin = nnopt_read_file(cache_path);
    if (!bin.empty()) {
        const unsigned char* binp = bin.data();
        size_t bin_sz = bin.size();
        cl_int binary_status = CL_SUCCESS;
        prog = clCreateProgramWithBinary(ctx, 1, &dev, &bin_sz, &binp,
                                          &binary_status, &err);
        if (err == CL_SUCCESS && prog && binary_status == CL_SUCCESS) {
            err = clBuildProgram(prog, 1, &dev, opts, nullptr, nullptr);
            if (err == CL_SUCCESS) {
                if (out_err) *out_err = CL_SUCCESS;
                return prog;
            }
            // Build with binary failed — fall through to source compile + refresh cache.
            clReleaseProgram(prog);
            prog = nullptr;
        } else if (prog) {
            clReleaseProgram(prog);
            prog = nullptr;
        }
    }

    // Compile from source.
    auto t0 = std::chrono::steady_clock::now();
    size_t src_len = strlen(source);
    prog = clCreateProgramWithSource(ctx, 1, &source, &src_len, &err);
    if (err != CL_SUCCESS || !prog) {
        if (out_err) *out_err = err;
        NNOPT_ERROR_FMT("nnopt_build_program_cached[%s]: clCreateProgramWithSource (%d)",
                        cache_key, (int)err);
        return nullptr;
    }
    err = clBuildProgram(prog, 1, &dev, opts, nullptr, nullptr);
    if (err != CL_SUCCESS) {
        size_t log_sz = 0;
        clGetProgramBuildInfo(prog, dev, CL_PROGRAM_BUILD_LOG, 0, nullptr, &log_sz);
        std::string log(log_sz, '\0');
        if (log_sz > 0) {
            clGetProgramBuildInfo(prog, dev, CL_PROGRAM_BUILD_LOG, log_sz, log.data(), nullptr);
        }
        NNOPT_ERROR_FMT("nnopt_build_program_cached[%s]: clBuildProgram (%d): %s",
                        cache_key, (int)err, log.c_str());
        clReleaseProgram(prog);
        if (out_err) *out_err = err;
        return nullptr;
    }
    auto t1 = std::chrono::steady_clock::now();
    double compile_s = std::chrono::duration<double>(t1 - t0).count();

    // Save the compiled binary for next time.
    size_t num_dev = 0;
    clGetProgramInfo(prog, CL_PROGRAM_NUM_DEVICES, sizeof(num_dev), &num_dev, nullptr);
    if (num_dev == 1) {
        size_t bin_sz = 0;
        clGetProgramInfo(prog, CL_PROGRAM_BINARY_SIZES, sizeof(bin_sz), &bin_sz, nullptr);
        if (bin_sz > 0) {
            std::vector<unsigned char> binbuf(bin_sz);
            unsigned char* binptrs[1] = { binbuf.data() };
            cl_int gerr = clGetProgramInfo(prog, CL_PROGRAM_BINARIES,
                                            sizeof(binptrs), binptrs, nullptr);
            if (gerr == CL_SUCCESS) {
                if (!nnopt_write_file(cache_path, binbuf.data(), binbuf.size())) {
                    // Non-fatal: cache write failed, next run will recompile.
                    NNOPT_ERROR_FMT("nnopt_build_program_cached[%s]: cache write to %s failed",
                                    cache_key, cache_path.c_str());
                }
            }
        }
    }
    NNOPT_CHECKPOINT_FMT("kernel_cache[%s]: compiled in %.3fs, cached at %s",
                         cache_key, compile_s, cache_path.c_str());
    if (out_err) *out_err = CL_SUCCESS;
    return prog;
}
