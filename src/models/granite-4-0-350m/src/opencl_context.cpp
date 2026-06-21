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
