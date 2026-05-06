#include "opencl_context.h"
#include "debug_utils.h"   // NNOPT_ERROR_FMT — used in build_program / build_program_from_file below.

#include <fstream>
#include <sstream>
#include <iostream>
#include <cstring>
#include <cstdint>
#include <sys/stat.h>
#include <dlfcn.h>

// ──────────────────────────────────────────────────────────────────────
// Program binary cache — fixes the cold-start TTFT problem (~30 s
// kernel-JIT compile on first-run, observed on Adreno 6xx). On first
// build we save the binary via clGetProgramInfo(CL_PROGRAM_BINARIES);
// on subsequent builds we load via clCreateProgramWithBinary, which
// the Adreno compiler treats as already-compiled — TTFT drops to <1s.
//
// Cache directory:
//   - On Android (cwd is the deploy dir under /data/local/tmp/...),
//     we write to ./kernel_cache/ which is writable.
//   - On host development we use the same relative path.
// Cache key: 64-bit FNV-1a hash of (source + "|" + options + "|"
//   + device_name + "|" + driver_version) so any change invalidates.
// ──────────────────────────────────────────────────────────────────────
namespace {

const char* kCacheDir = "kernel_cache";

uint64_t fnv1a64(const void* data, size_t len) {
    const uint8_t* p = (const uint8_t*)data;
    uint64_t h = 0xcbf29ce484222325ULL;
    for (size_t i = 0; i < len; i++) {
        h ^= p[i];
        h *= 0x100000001b3ULL;
    }
    return h;
}

bool ensure_cache_dir() {
    struct stat st;
    if (stat(kCacheDir, &st) == 0) return true;
    return mkdir(kCacheDir, 0755) == 0;
}

std::string cache_path(uint64_t key) {
    char buf[64];
    snprintf(buf, sizeof(buf), "%s/%016llx.bin", kCacheDir, (unsigned long long)key);
    return std::string(buf);
}

bool read_file(const std::string& path, std::vector<unsigned char>* out) {
    std::ifstream f(path, std::ios::binary);
    if (!f.is_open()) return false;
    f.seekg(0, std::ios::end);
    std::streamoff sz = f.tellg();
    if (sz <= 0) return false;
    out->resize((size_t)sz);
    f.seekg(0, std::ios::beg);
    f.read(reinterpret_cast<char*>(out->data()), sz);
    return f.good() || f.eof();
}

bool write_file(const std::string& path, const unsigned char* data, size_t len) {
    std::ofstream f(path, std::ios::binary | std::ios::trunc);
    if (!f.is_open()) return false;
    f.write(reinterpret_cast<const char*>(data), (std::streamsize)len);
    return f.good();
}

// Core: try cache → load binary; otherwise compile from source and cache.
// Used by both OpenCLContext::build_program and the static helper.
cl_program build_cached_core(cl_context ctx, cl_device_id dev,
                             const std::string& source,
                             const std::string& options) {
    char dev_name[256] = {0};
    char drv_ver [256] = {0};
    clGetDeviceInfo(dev, CL_DEVICE_NAME,    sizeof(dev_name), dev_name, nullptr);
    clGetDeviceInfo(dev, CL_DRIVER_VERSION, sizeof(drv_ver),  drv_ver,  nullptr);

    std::string key_str;
    key_str.reserve(source.size() + options.size() + 600);
    key_str += source;
    key_str += '|';
    key_str += options;
    key_str += '|';
    key_str += dev_name;
    key_str += '|';
    key_str += drv_ver;
    uint64_t cache_key = fnv1a64(key_str.data(), key_str.size());

    bool cache_dir_ok = ensure_cache_dir();
    std::string bin_path = cache_path(cache_key);
    cl_int err = CL_SUCCESS;

    if (cache_dir_ok) {
        std::vector<unsigned char> bin;
        if (read_file(bin_path, &bin) && !bin.empty()) {
            const unsigned char* bin_ptr = bin.data();
            size_t bin_len = bin.size();
            cl_int bin_status = CL_SUCCESS;
            cl_program prog = clCreateProgramWithBinary(
                ctx, 1, &dev, &bin_len, &bin_ptr, &bin_status, &err);
            if (err == CL_SUCCESS && bin_status == CL_SUCCESS && prog) {
                err = clBuildProgram(prog, 1, &dev, options.c_str(),
                                     nullptr, nullptr);
                if (err == CL_SUCCESS) {
                    return prog;
                }
                clReleaseProgram(prog);
            } else if (prog) {
                clReleaseProgram(prog);
            }
        }
    }

    const char* src_ptr = source.c_str();
    size_t src_len = source.size();
    cl_program program = clCreateProgramWithSource(ctx, 1, &src_ptr, &src_len, &err);
    if (err != CL_SUCCESS) {
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
            clGetProgramBuildInfo(program, dev, CL_PROGRAM_BUILD_LOG, log_size, log.data(), nullptr);
            fprintf(stderr, "OpenCL Build Log: %s\n", log.data());
            fflush(stderr);
        }
        clReleaseProgram(program);
        return nullptr;
    }

    if (cache_dir_ok) {
        cl_uint num_devs = 0;
        clGetProgramInfo(program, CL_PROGRAM_NUM_DEVICES, sizeof(num_devs), &num_devs, nullptr);
        if (num_devs == 1) {
            size_t bin_size = 0;
            clGetProgramInfo(program, CL_PROGRAM_BINARY_SIZES, sizeof(bin_size), &bin_size, nullptr);
            if (bin_size > 0) {
                std::vector<unsigned char> bin(bin_size);
                unsigned char* bin_ptr = bin.data();
                if (clGetProgramInfo(program, CL_PROGRAM_BINARIES,
                                     sizeof(unsigned char*), &bin_ptr, nullptr) == CL_SUCCESS) {
                    write_file(bin_path, bin.data(), bin_size);
                }
            }
        }
    }
    return program;
}

}  // namespace


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

    // CL_QUEUE_PROFILING_ENABLE forces the GPU to record start/end timestamps
    // on every dispatch — measurable overhead (~1–3% at our 240-disp/tok rate).
    // Only set it when the per-kernel profiler is requested via NNOPT_PROFILE=1.
    cl_command_queue_properties q_props = 0;
    {
        const char* e = std::getenv("NNOPT_PROFILE");
        if (e && e[0] == '1') q_props |= CL_QUEUE_PROFILING_ENABLE;
    }
    queue_ = clCreateCommandQueue(context_, device_, q_props, &err);
    if (err != CL_SUCCESS) return false;

    // cl_qcom_perf_hint — boost-clock hint to the Adreno KGSL driver.
    // Tells the dynamic-frequency governor to pin this context at the
    // highest available power level instead of waiting for sustained load
    // to detect a high-priority workload. Only Adreno GPUs respond.
    //
    // We load clSetPerfHintQCOM dynamically via clGetExtensionFunctionAddress
    // because vendor headers aren't part of the standard cl.h we include
    // (Qualcomm ships them with the Adreno SDK, not Khronos). Symbol name
    // and constants come from public Adreno OpenCL Programming Guide.
    //
    // Set NNOPT_NO_PERF_HINT=1 to disable (e.g. for thermal-tuning A/B).
    if (const char* off = std::getenv("NNOPT_NO_PERF_HINT"); !(off && off[0]=='1')) {
        typedef cl_int (CL_API_CALL *clSetPerfHintQCOM_fn)(cl_context, cl_uint);
        constexpr cl_uint CL_PERF_HINT_HIGH_QCOM = 0x40C3;

        auto fn = (clSetPerfHintQCOM_fn)clGetExtensionFunctionAddressForPlatform(
            platform_, "clSetPerfHintQCOM");

        // Try dlsym with RTLD_DEFAULT — searches all currently-loaded libs
        // for the symbol. Skips the ICD if the symbol is in libOpenCL.so
        // already mapped into the process. Avoids a fresh dlopen.
        if (!fn) {
            fn = (clSetPerfHintQCOM_fn)dlsym(RTLD_DEFAULT, "clSetPerfHintQCOM");
        }

        if (fn) {
            cl_int hint_err = fn(context_, CL_PERF_HINT_HIGH_QCOM);
            std::cerr << "PerfHint: clSetPerfHintQCOM(HIGH) -> err=" << hint_err << std::endl;
        } else {
            std::cerr << "PerfHint: clSetPerfHintQCOM not exposed (ICD blocks vendor symbols)" << std::endl;
        }
    }

    return true;
}

// Helper: append USE_FP16 + fast-relaxed-math to options if not present.
static std::string make_effective_options(const std::string& options) {
    std::string eff = options;
#ifdef NNOPT_USE_FP16
    if (eff.find("USE_FP16") == std::string::npos) {
        if (!eff.empty()) eff += " ";
        eff += "-D USE_FP16=1";
    }
#endif
    if (eff.find("-cl-fast-relaxed-math") == std::string::npos) {
        if (!eff.empty()) eff += " ";
        eff += "-cl-fast-relaxed-math";
    }
    return eff;
}

cl_program OpenCLContext::build_program(const std::string& source, const std::string& options) {
    return build_cached_core(context_, device_, source, make_effective_options(options));
}

cl_program OpenCLContext::build_cached_program_from_queue(
    cl_command_queue queue, const std::string& source, const std::string& options) {
    cl_context  ctx = nullptr;
    cl_device_id dev = nullptr;
    if (clGetCommandQueueInfo(queue, CL_QUEUE_CONTEXT, sizeof(ctx), &ctx, nullptr) != CL_SUCCESS ||
        clGetCommandQueueInfo(queue, CL_QUEUE_DEVICE,  sizeof(dev), &dev, nullptr) != CL_SUCCESS) {
        return nullptr;
    }
    return build_cached_core(ctx, dev, source, make_effective_options(options));
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
