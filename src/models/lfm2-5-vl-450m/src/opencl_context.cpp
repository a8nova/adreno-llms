#include "opencl_context.h"
#include "debug_utils.h"   // NNOPT_ERROR_FMT — used in build_program / build_program_from_file below.

#include <dlfcn.h>
#include <fstream>
#include <sstream>
#include <iostream>
#include <cstring>
#include <cstdlib>
#include <thread>
#include <chrono>

OpenCLContext::OpenCLContext() {}

OpenCLContext::~OpenCLContext() {
    if (recordable_queue_) clReleaseCommandQueue(recordable_queue_);
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

    // Surface device identity + extension list at startup so we can verify
    // what's actually available (Qualcomm guide §9 — extensions vary across
    // Adreno generations + driver versions; assumption-driven optimisation
    // breaks when an extension doesn't exist on this device).
    {
        char buf[4096] = {0};
        clGetDeviceInfo(device_, CL_DEVICE_NAME, sizeof(buf), buf, nullptr);
        fprintf(stderr, "[opencl] DEVICE_NAME=%s\n", buf);
        memset(buf, 0, sizeof(buf));
        clGetDeviceInfo(device_, CL_DEVICE_VERSION, sizeof(buf), buf, nullptr);
        fprintf(stderr, "[opencl] DEVICE_VERSION=%s\n", buf);
        memset(buf, 0, sizeof(buf));
        clGetDeviceInfo(device_, CL_DRIVER_VERSION, sizeof(buf), buf, nullptr);
        fprintf(stderr, "[opencl] DRIVER_VERSION=%s\n", buf);
        size_t max_wgs = 0;
        clGetDeviceInfo(device_, CL_DEVICE_MAX_WORK_GROUP_SIZE, sizeof(max_wgs), &max_wgs, nullptr);
        fprintf(stderr, "[opencl] MAX_WORK_GROUP_SIZE=%zu\n", max_wgs);
        cl_uint cu = 0;
        clGetDeviceInfo(device_, CL_DEVICE_MAX_COMPUTE_UNITS, sizeof(cu), &cu, nullptr);
        fprintf(stderr, "[opencl] MAX_COMPUTE_UNITS=%u\n", cu);
        cl_ulong gmem = 0, lmem = 0, cmem = 0;
        clGetDeviceInfo(device_, CL_DEVICE_GLOBAL_MEM_SIZE, sizeof(gmem), &gmem, nullptr);
        clGetDeviceInfo(device_, CL_DEVICE_LOCAL_MEM_SIZE, sizeof(lmem), &lmem, nullptr);
        clGetDeviceInfo(device_, CL_DEVICE_MAX_CONSTANT_BUFFER_SIZE, sizeof(cmem), &cmem, nullptr);
        fprintf(stderr, "[opencl] GLOBAL_MEM=%lluMB LOCAL_MEM=%lluKB MAX_CONSTANT=%lluKB\n",
                (unsigned long long)(gmem/(1024*1024)),
                (unsigned long long)(lmem/1024),
                (unsigned long long)(cmem/1024));
        cl_uint addr_bits = 0;
        clGetDeviceInfo(device_, CL_DEVICE_MEM_BASE_ADDR_ALIGN, sizeof(addr_bits), &addr_bits, nullptr);
        fprintf(stderr, "[opencl] MEM_BASE_ADDR_ALIGN=%u bits (%u bytes)\n", addr_bits, addr_bits / 8);
        memset(buf, 0, sizeof(buf));
        clGetDeviceInfo(device_, CL_DEVICE_EXTENSIONS, sizeof(buf), buf, nullptr);
        fprintf(stderr, "[opencl] EXTENSIONS=%s\n", buf);
        fflush(stderr);
    }

    cl_int err;

    // Track 1 — request highest sustained performance via cl_qcom_perf_hint
    // (Qualcomm guide §9.1.1). Userspace request, no root needed; pushes
    // DVFS toward higher GPU clocks. Property IDs taken from cl_qcom_ext.h:
    //   CL_CONTEXT_PERF_HINT_QCOM = 0x40C2
    //   CL_PERF_HINT_HIGH_QCOM    = 0x40C3
    // Extension confirmed present in DEVICE_EXTENSIONS at startup.
    cl_context_properties props_perfhint[] = {
        CL_CONTEXT_PLATFORM, (cl_context_properties)platform_,
        (cl_context_properties)0x40C2, (cl_context_properties)0x40C3,
        0
    };
    context_ = clCreateContext(props_perfhint, 1, &device_, nullptr, nullptr, &err);
    if (err != CL_SUCCESS) {
        // Fall back to plain context if driver rejects the perf-hint property.
        fprintf(stderr, "[opencl] WARN clCreateContext(perf_hint=HIGH) failed (err=%d) — falling back to plain context\n", (int)err);
        fflush(stderr);
        context_ = clCreateContext(nullptr, 1, &device_, nullptr, nullptr, &err);
        if (err != CL_SUCCESS) return false;
    } else {
        fprintf(stderr, "[opencl] perf_hint=HIGH_QCOM enabled\n");
        fflush(stderr);
    }

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

    // Track 5 — cl_qcom_recordable_queues feature detection + setup.
    // Vendor extension; Adreno hides entry points from
    // clGetExtensionFunctionAddressForPlatform, so dlsym from RTLD_DEFAULT.
    // Gated behind NNOPT_RECORD env (default OFF) + presence of the extension
    // string in DEVICE_EXTENSIONS to avoid breaking devices without it.
    const char* rec_env = std::getenv("NNOPT_RECORD");
    const bool rec_requested = rec_env && rec_env[0] != '0' && rec_env[0] != '\0';
    if (rec_requested) {
        char ext_buf[8192] = {0};
        clGetDeviceInfo(device_, CL_DEVICE_EXTENSIONS, sizeof(ext_buf), ext_buf, nullptr);
        const bool has_ext = (strstr(ext_buf, "cl_qcom_recordable_queues") != nullptr);
        if (!has_ext) {
            fprintf(stderr, "[opencl] NNOPT_RECORD=1 but cl_qcom_recordable_queues NOT in EXTENSIONS — recording disabled\n");
            fflush(stderr);
        } else {
            rec_fn_new_     = (nnopt_qcom_rec::fn_new_t)     dlsym(RTLD_DEFAULT, "clNewRecordingQCOM");
            rec_fn_end_     = (nnopt_qcom_rec::fn_end_t)     dlsym(RTLD_DEFAULT, "clEndRecordingQCOM");
            rec_fn_release_ = (nnopt_qcom_rec::fn_release_t) dlsym(RTLD_DEFAULT, "clReleaseRecordingQCOM");
            rec_fn_enqueue_ = (nnopt_qcom_rec::fn_enqueue_t) dlsym(RTLD_DEFAULT, "clEnqueueRecordingQCOM");
            if (!rec_fn_new_ || !rec_fn_end_ || !rec_fn_release_ || !rec_fn_enqueue_) {
                fprintf(stderr, "[opencl] dlsym recordable_queues entry points: new=%p end=%p release=%p enqueue=%p — recording disabled\n",
                        (void*)rec_fn_new_, (void*)rec_fn_end_, (void*)rec_fn_release_, (void*)rec_fn_enqueue_);
                fflush(stderr);
                rec_fn_new_ = nullptr;
            } else {
                // Use OpenCL 2.0 clCreateCommandQueueWithProperties — the custom
                // RECORDABLE_BIT isn't accepted by legacy clCreateCommandQueue
                // (returns -30/CL_INVALID_VALUE). dlsym since CL 1.2 headers
                // don't expose the WithProperties variant.
                typedef cl_command_queue (CL_API_CALL *fn_create_q_t)(
                    cl_context, cl_device_id, const cl_ulong*, cl_int*);
                fn_create_q_t fn_create_q = (fn_create_q_t)
                    dlsym(RTLD_DEFAULT, "clCreateCommandQueueWithProperties");
                if (!fn_create_q) {
                    fprintf(stderr, "[opencl] dlsym(clCreateCommandQueueWithProperties)=null — recording disabled\n");
                    fflush(stderr);
                    rec_fn_new_ = nullptr;
                } else {
                    // CL_QUEUE_PROPERTIES = 0x1093 per OpenCL 2.0 spec. RECORDABLE
                    // is NOT compatible with PROFILING_ENABLE on Adreno — combining
                    // them returns -30 (CL_INVALID_VALUE). Use RECORD_BIT alone
                    // per validated Qwen impl.
                    cl_ulong props[] = {
                        0x1093, /*CL_QUEUE_PROPERTIES*/
                        (cl_ulong)nnopt_qcom_rec::RECORDABLE_BIT,
                        0
                    };
                    cl_int rerr = CL_SUCCESS;
                    recordable_queue_ = fn_create_q(context_, device_, props, &rerr);
                    if (rerr != CL_SUCCESS || !recordable_queue_) {
                        fprintf(stderr, "[opencl] clCreateCommandQueueWithProperties(RECORDABLE) failed (%d) — recording disabled\n", (int)rerr);
                        fflush(stderr);
                        recordable_queue_ = nullptr;
                        rec_fn_new_ = nullptr;
                    } else {
                        fprintf(stderr, "[opencl] cl_qcom_recordable_queues ENABLED (live_q + recordable_q created)\n");
                        fflush(stderr);
                    }
                }
            }
        }
    }

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

    // Track 2 — fast math. Qualcomm guide §8.2 / §8.3: enables relaxed-IEEE
    // for built-in transcendentals (exp/log/sqrt/division) and flush-denormals.
    // Numerical impact is gated by the 7/7 token greedy match; if any token
    // diverges, this flag is the prime suspect.
    if (effective_options.find("-cl-fast-relaxed-math") == std::string::npos) {
        if (!effective_options.empty()) effective_options += " ";
        effective_options += "-cl-fast-relaxed-math";
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

void OpenCLContext::yield_for_compositor(int sleep_ms) {
    // Cache the env check once: cheap to call per layer.
    static const bool enabled = ([]() {
        const char* e = std::getenv("NNOPT_GPU_YIELD");
        return e != nullptr && e[0] != '0' && e[0] != '\0';
    })();
    if (!enabled || !queue_) return;
    // clFinish (not clFlush) so the GPU actually drains and goes idle, letting
    // the compositor render a frame before we enqueue the next layer's kernels.
    clFinish(queue_);
    if (sleep_ms > 0) {
        std::this_thread::sleep_for(std::chrono::milliseconds(sleep_ms));
    }
}
