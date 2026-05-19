#include "opencl_context.h"
#include "debug_utils.h"   // NNOPT_ERROR_FMT — used in build_program / build_program_from_file below.

#include <fstream>
#include <sstream>
#include <iostream>
#include <cstdlib>
#include <cstring>
#include <dlfcn.h>

OpenCLContext::OpenCLContext() {}

OpenCLContext::~OpenCLContext() {
    for (auto& e : program_cache_) {
        if (e.program) {
            clReleaseProgram(e.program);
            e.program = nullptr;
        }
    }
    program_cache_.clear();

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
    // Phase A #4 — request high-performance GPU clock via cl_qcom_perf_hint.
    // Without this, the driver may DVFS the GPU down between dispatches,
    // adding 10–25% latency. Tokens taken from the public Qualcomm OpenCL
    // extension spec (also used by TFLite + MNN). If the extension isn't
    // supported on this device the create-with-properties call returns
    // CL_INVALID_PROPERTY (-64); we then fall back to no properties.
    #define NNOPT_CL_CONTEXT_PERF_HINT_QCOM 0x40C2
    #define NNOPT_CL_PERF_HINT_HIGH_QCOM    0x40C3
    {
        size_t ext_len = 0;
        clGetDeviceInfo(device_, CL_DEVICE_EXTENSIONS, 0, nullptr, &ext_len);
        std::vector<char> ext(ext_len + 1, 0);
        clGetDeviceInfo(device_, CL_DEVICE_EXTENSIONS, ext_len, ext.data(), nullptr);
        if (std::strstr(ext.data(), "cl_qcom_perf_hint") != nullptr) {
            cl_context_properties props[] = {
                NNOPT_CL_CONTEXT_PERF_HINT_QCOM, NNOPT_CL_PERF_HINT_HIGH_QCOM,
                0
            };
            context_ = clCreateContext(props, 1, &device_, nullptr, nullptr, &err);
            if (err == CL_SUCCESS && context_) {
                // Perf hint accepted — done.
            } else {
                context_ = nullptr;
            }
        }
    }
    if (!context_) {
        context_ = clCreateContext(nullptr, 1, &device_, nullptr, nullptr, &err);
        if (err != CL_SUCCESS) return false;
    }

    // Enable PROFILING only when explicitly requested (NNOPT_PROFILE=1).
    // CL_QUEUE_PROFILING_ENABLE adds event-bookkeeping overhead per dispatch
    // even when no event is requested — drop it from the steady-state hot
    // path. (Adreno does not support CL_QUEUE_OUT_OF_ORDER_EXEC_MODE_ENABLE.)
    cl_command_queue_properties qprops = 0;
    const char* prof_env = std::getenv("NNOPT_PROFILE");
    if (prof_env && prof_env[0] != '0') qprops |= CL_QUEUE_PROFILING_ENABLE;
    queue_ = clCreateCommandQueue(context_, device_, qprops, &err);
    if (err != CL_SUCCESS) return false;

    // ── cl_qcom_recordable_queues entry-point load (Adreno guide §9.1.3).
    //   dlsym from the already-loaded libOpenCL.so. Same pattern as
    //   lfm2-5-350m/src/opencl_context.cpp:244-254, which validated this on
    //   the same A6xx driver family. Symbols may also be ICD-blocked — fall
    //   back to clGetExtensionFunctionAddressForPlatform if dlsym misses.
    fn_new_recording_     = (clNewRecordingQCOM_fn)    dlsym(RTLD_DEFAULT, "clNewRecordingQCOM");
    fn_end_recording_     = (clEndRecordingQCOM_fn)    dlsym(RTLD_DEFAULT, "clEndRecordingQCOM");
    fn_release_recording_ = (clReleaseRecordingQCOM_fn)dlsym(RTLD_DEFAULT, "clReleaseRecordingQCOM");
    fn_enqueue_recording_ = (clEnqueueRecordingQCOM_fn)dlsym(RTLD_DEFAULT, "clEnqueueRecordingQCOM");
    if (!fn_new_recording_) {
        fn_new_recording_ = (clNewRecordingQCOM_fn)clGetExtensionFunctionAddressForPlatform(platform_, "clNewRecordingQCOM");
        fn_end_recording_ = (clEndRecordingQCOM_fn)clGetExtensionFunctionAddressForPlatform(platform_, "clEndRecordingQCOM");
        fn_release_recording_ = (clReleaseRecordingQCOM_fn)clGetExtensionFunctionAddressForPlatform(platform_, "clReleaseRecordingQCOM");
        fn_enqueue_recording_ = (clEnqueueRecordingQCOM_fn)clGetExtensionFunctionAddressForPlatform(platform_, "clEnqueueRecordingQCOM");
    }
    record_fns_loaded_ = (fn_new_recording_ && fn_end_recording_ &&
                          fn_release_recording_ && fn_enqueue_recording_);
    if (!std::getenv("NNOPT_QUIET")) {
        fprintf(stderr, "  recordable_qs   %s\n", record_fns_loaded_ ? "yes" : "no");
    }

    // One-shot device banner so the demo shows exactly which OpenCL stack is
    // running. Mirrors the info nnopt's last_probe.json captures host-side, but
    // queried live from the actual device this binary connected to. Suppress
    // with NNOPT_QUIET=1 (test harnesses, perf scripts).
    if (!std::getenv("NNOPT_QUIET")) {
        char platform_name[128] = {0};
        char device_name[128]   = {0};
        char device_version[128]= {0};
        char driver_version[256]= {0};
        char ext_buf[8192]      = {0};
        cl_uint cu = 0;
        size_t max_wg = 0;
        cl_ulong gmem = 0, lmem = 0;
        cl_uint clock_mhz = 0;
        clGetPlatformInfo(platform_, CL_PLATFORM_NAME, sizeof(platform_name), platform_name, nullptr);
        clGetDeviceInfo(device_, CL_DEVICE_NAME,                 sizeof(device_name),    device_name,    nullptr);
        clGetDeviceInfo(device_, CL_DEVICE_VERSION,              sizeof(device_version), device_version, nullptr);
        clGetDeviceInfo(device_, CL_DRIVER_VERSION,              sizeof(driver_version), driver_version, nullptr);
        clGetDeviceInfo(device_, CL_DEVICE_MAX_COMPUTE_UNITS,    sizeof(cu),       &cu,       nullptr);
        clGetDeviceInfo(device_, CL_DEVICE_MAX_WORK_GROUP_SIZE,  sizeof(max_wg),   &max_wg,   nullptr);
        clGetDeviceInfo(device_, CL_DEVICE_GLOBAL_MEM_SIZE,      sizeof(gmem),     &gmem,     nullptr);
        clGetDeviceInfo(device_, CL_DEVICE_LOCAL_MEM_SIZE,       sizeof(lmem),     &lmem,     nullptr);
        clGetDeviceInfo(device_, CL_DEVICE_MAX_CLOCK_FREQUENCY,  sizeof(clock_mhz),&clock_mhz,nullptr);
        clGetDeviceInfo(device_, CL_DEVICE_EXTENSIONS, sizeof(ext_buf) - 1, ext_buf, nullptr);
        const bool has_fp16    = std::strstr(ext_buf, "cl_khr_fp16")       != nullptr;
        const bool has_perfhnt = std::strstr(ext_buf, "cl_qcom_perf_hint") != nullptr;
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
        fprintf(stderr, "  cl_khr_fp16     %s\n", has_fp16    ? "yes" : "no");
        fprintf(stderr, "  qcom_perf_hint  %s\n", has_perfhnt ? "yes" : "no");
        fprintf(stderr, "─────────────────────────────────────────────────────────────\n");
        fflush(stderr);
    }

    // Build and cache commonly-used programs at init time.
    // This avoids long compile times and extra RAM pressure mid-inference.
    // NOTE: Many ops (esp. debug/graph wiring) assume kernels/utils.cl is available.
    if (!ensure_program_from_file("kernels/utils.cl")) return false;
    if (!ensure_program_from_file("kernels/conv_1d.cl")) return false;
    if (!ensure_program_from_file("kernels/conv_transpose_1d.cl")) return false;
    if (!ensure_program_from_file("kernels/hifigan_residual_block.cl")) return false;

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

    // Phase B #6 — fast-math + mad-enable. The vocoder/flow kernels use
    // tanh/exp/sigmoid which are precision-tolerant for audio output. The
    // ResBlock convs are fp16 anyway so we're already at rough precision.
    // Set NNOPT_NO_FAST_MATH=1 to disable for A/B (e.g. when chasing a
    // numerical regression).
    {
        static int s_fast_math = -1;
        if (s_fast_math < 0) {
            const char* env = std::getenv("NNOPT_NO_FAST_MATH");
            s_fast_math = (env && env[0] == '1') ? 0 : 1;
        }
        if (s_fast_math) {
            if (!effective_options.empty()) effective_options += " ";
            effective_options += "-cl-fast-relaxed-math -cl-mad-enable";
        }
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

bool OpenCLContext::ensure_program_from_file(const std::string& path, const std::string& options) {
    for (const auto& e : program_cache_) {
        if (e.path == path) return e.program != nullptr;
    }
    cl_program p = build_program_from_file(path, options);
    program_cache_.push_back(ProgramEntry{path, p});
    return p != nullptr;
}

cl_program OpenCLContext::get_program(const std::string& path) const {
    for (const auto& e : program_cache_) {
        if (e.path == path) return e.program;
    }
    return nullptr;
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

// ── cl_qcom_recordable_queues helpers ────────────────────────────────────────
// Lifted from lfm2-5-350m/src/opencl_context.cpp:259-297 — same A6xx driver
// family. The recordable bit is (cl_command_queue_properties)1 << 30.
cl_command_queue OpenCLContext::create_recordable_queue() {
    if (!record_fns_loaded_) return nullptr;
    constexpr cl_command_queue_properties RECORD_BIT = (cl_command_queue_properties)1 << 30;
    cl_int err = CL_SUCCESS;
    cl_command_queue q = clCreateCommandQueue(context_, device_, RECORD_BIT, &err);
    if (err != CL_SUCCESS || !q) {
        fprintf(stderr, "RecordQ: clCreateCommandQueue(RECORD_BIT) failed err=%d\n", (int)err);
        return nullptr;
    }
    return q;
}

cl_recording_qcom OpenCLContext::new_recording(cl_command_queue q) const {
    if (!fn_new_recording_) return nullptr;
    cl_int err = CL_SUCCESS;
    cl_recording_qcom rec = fn_new_recording_(q, &err);
    if (err != CL_SUCCESS) return nullptr;
    return rec;
}

cl_int OpenCLContext::end_recording(cl_recording_qcom rec) const {
    if (!fn_end_recording_) return CL_INVALID_OPERATION;
    return fn_end_recording_(rec);
}

cl_int OpenCLContext::release_recording(cl_recording_qcom rec) const {
    if (!fn_release_recording_) return CL_INVALID_OPERATION;
    return fn_release_recording_(rec);
}

cl_int OpenCLContext::enqueue_recording(cl_command_queue live_q,
                                        cl_recording_qcom rec,
                                        size_t num_args,
                                        const cl_array_arg_qcom* args) const {
    if (!fn_enqueue_recording_) return CL_INVALID_OPERATION;
    return fn_enqueue_recording_(live_q, rec, num_args, args,
                                 0, nullptr, 0, nullptr, 0, nullptr,
                                 0, nullptr, nullptr);
}
