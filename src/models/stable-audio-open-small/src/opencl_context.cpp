#include "opencl_context.h"
#include "debug_utils.h"   // NNOPT_ERROR_FMT — used in build_program / build_program_from_file below.

#include <fstream>
#include <sstream>
#include <iostream>
#include <cstring>
#include <cstdlib>
#include <string>

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
    // B10: cl_qcom_perf_hint — request highest GPU clock policy (guide
    // §9.1.1; same pattern as the kokoro port). Probe the extension first;
    // fall back to a plain context on non-Qualcomm or rejecting drivers.
    // NNOPT_PERF_HINT=0 disables for A/B.
    constexpr cl_context_properties CL_CONTEXT_PERF_HINT_QCOM = 0x40C2;
    constexpr cl_context_properties CL_PERF_HINT_HIGH_QCOM    = 0x40C3;
    bool want_hint = true;
    if (const char* e = std::getenv("NNOPT_PERF_HINT")) want_hint = (e[0] != '0');
    context_ = nullptr;
    {
        size_t ext_len = 0;
        clGetDeviceInfo(device_, CL_DEVICE_EXTENSIONS, 0, nullptr, &ext_len);
        std::string ext_str(ext_len, '\0');
        if (ext_len > 0) clGetDeviceInfo(device_, CL_DEVICE_EXTENSIONS, ext_len, &ext_str[0], nullptr);
        // NNOPT_PRINT_EXTENSIONS=1: dump the extension list once — drives
        // which vendor extensions (recordable queues, onchip global memory)
        // future campaign rounds can rely on.
        if (const char* pe = std::getenv("NNOPT_PRINT_EXTENSIONS")) {
            if (pe[0] == '1') fprintf(stderr, "CL_DEVICE_EXTENSIONS: %s\n", ext_str.c_str());
        }
        if (want_hint && ext_str.find("cl_qcom_perf_hint") != std::string::npos) {
            cl_context_properties props[] = {
                CL_CONTEXT_PERF_HINT_QCOM, CL_PERF_HINT_HIGH_QCOM, 0
            };
            context_ = clCreateContext(props, 1, &device_, nullptr, nullptr, &err);
            if (err != CL_SUCCESS) context_ = nullptr;   // driver rejected the hint
        }
    }
    if (!context_) context_ = clCreateContext(nullptr, 1, &device_, nullptr, nullptr, &err);
    if (err != CL_SUCCESS) return false;

    queue_ = clCreateCommandQueue(context_, device_, CL_QUEUE_PROFILING_ENABLE, &err);
    return err == CL_SUCCESS;
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
    // Adreno quick win (opt #10): fast-relaxed-math + mad-enable for all
    // custom kernels (CLBlast builds its own programs — unaffected).
    // e2e waveform cosine must be re-validated after any change here.
    effective_options += " -cl-fast-relaxed-math -cl-mad-enable";

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

// ── OPT-5: on-disk program binary cache ──────────────────────────────────
// clBuildProgram from source is a multi-hundred-ms cost per .cl file per
// process, paid on EVERY one-shot run (guide §5.7.3: use
// clCreateProgramWithBinary). Cache file `<path>.bin` = [8-byte fnv1a key]
// [CL_PROGRAM_BINARIES bytes], keyed on source+options+device+driver+dtype so
// a driver update or kernel edit invalidates cleanly (falls back to source
// build and rewrites). Ported from the moonshine r6 implementation.
// Kill-switch: NNOPT_PROG_CACHE=0 (A/B).
static uint64_t nnopt_fnv1a64(const std::string& s, uint64_t h = 1469598103934665603ULL) {
    for (unsigned char c : s) { h ^= c; h *= 1099511628211ULL; }
    return h;
}

cl_program OpenCLContext::load_or_build_program(const std::string& path,
                                                const std::string& source,
                                                const std::string& options) {
    static const bool cache_on = [] {
        const char* e = std::getenv("NNOPT_PROG_CACHE");
        return !(e && e[0] == '0');
    }();
    if (!cache_on) return build_program(source, options);

    char driver[256] = {0};
    clGetDeviceInfo(device_, CL_DRIVER_VERSION, sizeof(driver), driver, nullptr);
    uint64_t key = nnopt_fnv1a64(source);
    key = nnopt_fnv1a64(options, key);
    // build_program appends fast-math/mad/USE_FP16 deterministically — fold a
    // constant marker so changing that suffix invalidates the cache too.
    key = nnopt_fnv1a64("-cl-fast-relaxed-math -cl-mad-enable", key);
    key = nnopt_fnv1a64(device_name(), key);
    key = nnopt_fnv1a64(driver, key);
#ifdef NNOPT_USE_FP16
    key = nnopt_fnv1a64("fp16", key);
#endif
    const std::string cache_path = path + ".bin";

    // Try the cache: [8-byte key][binary bytes].
    {
        std::ifstream in(cache_path, std::ios::binary);
        if (in.is_open()) {
            uint64_t stored_key = 0;
            in.read(reinterpret_cast<char*>(&stored_key), sizeof(stored_key));
            if (in.good() && stored_key == key) {
                std::vector<unsigned char> bin((std::istreambuf_iterator<char>(in)),
                                               std::istreambuf_iterator<char>());
                if (!bin.empty()) {
                    const unsigned char* bin_ptr = bin.data();
                    size_t bin_len = bin.size();
                    cl_int bin_status = CL_SUCCESS, err = CL_SUCCESS;
                    cl_program prog = clCreateProgramWithBinary(
                        context_, 1, &device_, &bin_len, &bin_ptr, &bin_status, &err);
                    if (prog && err == CL_SUCCESS && bin_status == CL_SUCCESS &&
                        clBuildProgram(prog, 1, &device_, nullptr, nullptr, nullptr) == CL_SUCCESS) {
                        return prog;
                    }
                    if (prog) clReleaseProgram(prog);
                    // Corrupt/incompatible cache — fall through to source build.
                }
            }
        }
    }

    cl_program prog = build_program(source, options);
    if (!prog) return nullptr;

    // Persist the binary (best-effort; failure to write is not an error).
    size_t bin_len = 0;
    if (clGetProgramInfo(prog, CL_PROGRAM_BINARY_SIZES, sizeof(bin_len), &bin_len, nullptr) == CL_SUCCESS
        && bin_len > 0) {
        std::vector<unsigned char> bin(bin_len);
        unsigned char* bin_ptr = bin.data();
        if (clGetProgramInfo(prog, CL_PROGRAM_BINARIES, sizeof(bin_ptr), &bin_ptr, nullptr) == CL_SUCCESS) {
            std::ofstream out(cache_path, std::ios::binary | std::ios::trunc);
            if (out.is_open()) {
                out.write(reinterpret_cast<const char*>(&key), sizeof(key));
                out.write(reinterpret_cast<const char*>(bin.data()), (std::streamsize)bin.size());
            }
        }
    }
    return prog;
}

cl_program OpenCLContext::build_program_from_file(const std::string& path, const std::string& options) {
    std::ifstream file(path);
    if (!file.is_open()) {
        NNOPT_ERROR_FMT("Failed to open kernel file: %s", path.c_str());
        return nullptr;
    }

    std::stringstream buffer;
    buffer << file.rdbuf();
    cl_program prog = load_or_build_program(path, buffer.str(), options);
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

// ── Kernel cache + buffer pool (see opencl_context.h for rationale) ─────────
#include <cstdlib>
#include <unordered_map>

static bool _cache_on(const char* env) {
    const char* e = getenv(env);
    return !(e && e[0] == '0');
}

cl_kernel nnopt_cached_kernel(cl_program prog, const char* name, cl_int* err_out) {
    static const bool on = _cache_on("NNOPT_KERNEL_CACHE");
    if (!on) {
        return clCreateKernel(prog, name, err_out);
    }
    static std::unordered_map<std::string, cl_kernel> cache;
    char key[192];
    snprintf(key, sizeof(key), "%p:%s", (void*)prog, name);
    auto it = cache.find(key);
    if (it != cache.end()) { if (err_out) *err_out = CL_SUCCESS; return it->second; }
    cl_kernel k = clCreateKernel(prog, name, err_out);
    if (k) cache.emplace(key, k);
    return k;
}

void nnopt_kernel_done(cl_kernel k) {
    static const bool on = _cache_on("NNOPT_KERNEL_CACHE");
    if (!on && k) clReleaseKernel(k);
}

// Exact-size free-list pool. live: pool-owned buffers currently handed out;
// free_list: pool-owned buffers awaiting reuse, keyed by byte size.
static std::unordered_map<cl_mem, size_t> g_pool_live;
static std::unordered_multimap<size_t, cl_mem> g_pool_free;

cl_mem nnopt_pool_alloc(cl_context ctx, size_t bytes, cl_int* err_out) {
    static const bool on = _cache_on("NNOPT_BUF_POOL");
    if (on) {
        auto it = g_pool_free.find(bytes);
        if (it != g_pool_free.end()) {
            cl_mem b = it->second;
            g_pool_free.erase(it);
            g_pool_live.emplace(b, bytes);
            if (err_out) *err_out = CL_SUCCESS;
            return b;
        }
    }
    cl_mem b = clCreateBuffer(ctx, CL_MEM_READ_WRITE, bytes, nullptr, err_out);
    if (on && b) g_pool_live.emplace(b, bytes);
    return b;
}

void nnopt_pool_release(cl_mem buf) {
    if (!buf) return;
    auto it = g_pool_live.find(buf);
    if (it == g_pool_live.end()) { clReleaseMemObject(buf); return; }  // foreign buffer
    g_pool_free.emplace(it->second, buf);
    g_pool_live.erase(it);
}
