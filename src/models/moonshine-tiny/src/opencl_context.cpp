#include "opencl_context.h"
#include "debug_utils.h"   // NNOPT_ERROR_FMT — used in build_program / build_program_from_file below.

#include <fstream>
#include <sstream>
#include <iostream>
#include <cstdint>
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

// OPT-6 (r6): device-side program binary cache. clBuildProgram of the full
// kernel file costs ~1s of every process launch (TTFT). Cache the compiled
// binary next to the .cl keyed by FNV-1a of (source, options, device name,
// driver version) — any change to source/flags/driver misses and recompiles.
static uint64_t nnopt_fnv1a64(const std::string& s, uint64_t h = 1469598103934665603ULL) {
    for (unsigned char c : s) { h ^= c; h *= 1099511628211ULL; }
    return h;
}

cl_program OpenCLContext::load_or_build_program(const std::string& path,
                                                const std::string& source,
                                                const std::string& options) {
    char driver[256] = {0};
    clGetDeviceInfo(device_, CL_DRIVER_VERSION, sizeof(driver), driver, nullptr);
    uint64_t key = nnopt_fnv1a64(source);
    key = nnopt_fnv1a64(options, key);
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
