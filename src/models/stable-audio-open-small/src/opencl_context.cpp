#include "opencl_context.h"
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
    context_ = clCreateContext(nullptr, 1, &device_, nullptr, nullptr, &err);
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
