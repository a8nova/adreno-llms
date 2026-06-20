#include "opencl_context.h"
#include "debug_utils.h"   // NNOPT_ERROR_FMT — used in build_program / build_program_from_file below.

#include <fstream>
#include <sstream>
#include <iostream>
#include <cstdlib>
#include <cstdio>      // FILE* + fopen/fwrite/fflush/fclose for fsync'd cache writes
#include <cstring>
#include <cstdint>     // uint64_t — FNV-1a hash key
#include <dlfcn.h>
#include <sys/stat.h>  // stat / mkdir for kernel_cache directory
#include <unistd.h>    // fsync — durably commit cache writes before process exit
#include <vector>

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

    // Phase A smoke test: when NNOPT_RECORD=1, attempt full create/new/end/release
    // cycle to verify the entire recordable-queue infrastructure works end-to-end.
    // Prints the canonical "ENABLED" line from the porting guide on success so
    // we have a hard go/no-go before doing the real refactor. No-op when unset.
    if (record_fns_loaded_) {
        const char* rec_env = std::getenv("NNOPT_RECORD");
        if (rec_env && rec_env[0] == '1') {
            char ext_buf[8192] = {0};
            clGetDeviceInfo(device_, CL_DEVICE_EXTENSIONS, sizeof(ext_buf), ext_buf, nullptr);
            const bool has_ext = (std::strstr(ext_buf, "cl_qcom_recordable_queues") != nullptr);
            if (!has_ext) {
                fprintf(stderr, "[opencl] NNOPT_RECORD=1 but cl_qcom_recordable_queues "
                                "NOT in EXTENSIONS — recording disabled\n");
            } else {
                cl_command_queue rec_q = create_recordable_queue();
                if (!rec_q) {
                    fprintf(stderr, "[opencl] recordable queue creation FAILED\n");
                } else {
                    cl_recording_qcom rec = new_recording(rec_q);
                    if (!rec) {
                        fprintf(stderr, "[opencl] clNewRecordingQCOM returned null\n");
                    } else {
                        cl_int e_end = end_recording(rec);
                        cl_int e_rel = release_recording(rec);
                        if (e_end == CL_SUCCESS && e_rel == CL_SUCCESS) {
                            fprintf(stderr, "[opencl] cl_qcom_recordable_queues ENABLED "
                                            "(live_q + recordable_q created)\n");
                        } else {
                            fprintf(stderr, "[opencl] end=%d release=%d on empty recording\n",
                                    (int)e_end, (int)e_rel);
                        }
                    }
                    clReleaseCommandQueue(rec_q);
                }
            }
        }
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
        fprintf(stderr, "  cl_device_extensions   %s\n", ext_buf);
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

// ──────────────────────────────────────────────
// On-disk OpenCL program-binary cache (lifted verbatim from
// smolvlm-256m-instruct/src/opencl_context.cpp:155–378). First call with a
// given (source + options + device + driver) compiles from source AND writes
// the device binary to `<cwd>/kernel_cache/<hash>.bin`. Subsequent launches
// reload via clCreateProgramWithBinary, skipping the ~50–200 ms × N kernel-
// build cost that otherwise hits every launch.
//
// Cache key: 16-hex-digit FNV-1a hash of "source|options|device_name|driver".
// Write path: fopen + fwrite + fflush + fsync + fclose, with std::remove
// on any failure so the next launch doesn't load a corrupt half-file.
// Read path: validates file size > 0 before passing to
// clCreateProgramWithBinary; falls through to source compile on any miss.
// Diagnostics: NNOPT_KCACHE_LOG=1 → one stderr line per HIT/MISS/SAVE.
// ──────────────────────────────────────────────

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

static bool nnopt_ensure_cache_dir() {
    struct stat st{};
    if (stat("kernel_cache", &st) == 0) {
        return S_ISDIR(st.st_mode);
    }
    if (mkdir("kernel_cache", 0755) == 0) {
        return true;
    }
    if (stat("kernel_cache", &st) == 0 && S_ISDIR(st.st_mode)) return true;
    return false;
}

static inline bool nnopt_kcache_log() {
    static int s = -1;
    if (s < 0) {
        const char* e = std::getenv("NNOPT_KCACHE_LOG");
        s = (e && e[0] == '1') ? 1 : 0;
    }
    return s == 1;
}

static cl_program nnopt_build_program_cached(cl_context ctx, cl_device_id dev,
                                             const std::string& source,
                                             const std::string& options_in) {
    // Inject -cl-fast-relaxed-math globally. Per Qualcomm's Adreno guide §8.2,
    // this enables fast math (native_exp, native_rsqrt, etc.) across all
    // kernels — 2-5x faster than IEEE-conformant math functions, and the
    // vocoder/flow LeakyReLU/tanh don't need IEEE precision.
    std::string options = options_in;
    if (options.find("fast-relaxed-math") == std::string::npos)
        options += " -cl-fast-relaxed-math";
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
        if (!f) {
            if (nnopt_kcache_log()) {
                fprintf(stderr, "[kernel_cache] MISS no-file %s\n", cache_path.c_str());
            }
        } else {
            const std::streamsize sz_signed = f.tellg();
            f.seekg(0, std::ios::beg);
            if (sz_signed <= 0) {
                if (nnopt_kcache_log()) {
                    fprintf(stderr, "[kernel_cache] MISS empty-file %s\n", cache_path.c_str());
                }
            } else {
                const size_t sz = (size_t)sz_signed;
                std::vector<unsigned char> bin(sz);
                if (!f.read(reinterpret_cast<char*>(bin.data()), sz)) {
                    if (nnopt_kcache_log()) {
                        fprintf(stderr, "[kernel_cache] MISS read-fail %s\n", cache_path.c_str());
                    }
                } else {
                    cl_int binary_status = CL_SUCCESS;
                    const unsigned char* bp = bin.data();
                    program = clCreateProgramWithBinary(ctx, 1, &dev, &sz, &bp,
                                                        &binary_status, &err);
                    if (err == CL_SUCCESS && binary_status == CL_SUCCESS && program) {
                        err = clBuildProgram(program, 1, &dev, options.c_str(),
                                             nullptr, nullptr);
                        if (err == CL_SUCCESS) {
                            if (nnopt_kcache_log()) {
                                fprintf(stderr, "[kernel_cache] HIT %s (%zu B)\n",
                                        cache_path.c_str(), sz);
                            }
                            return program;
                        }
                        fprintf(stderr, "[kernel_cache] HIT-but-rebuild-failed "
                                        "(err=%d); falling back to source: %s\n",
                                (int)err, cache_path.c_str());
                        clReleaseProgram(program);
                        program = nullptr;
                    } else {
                        if (nnopt_kcache_log()) {
                            fprintf(stderr, "[kernel_cache] MISS createProgramWithBinary "
                                            "err=%d binary_status=%d %s\n",
                                    (int)err, (int)binary_status, cache_path.c_str());
                        }
                        if (program) { clReleaseProgram(program); program = nullptr; }
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
            clGetProgramBuildInfo(program, dev, CL_PROGRAM_BUILD_LOG, log_size, log.data(), nullptr);
            fprintf(stderr, "OpenCL Build Log: %s\n", log.data());
            fflush(stderr);
        }
        clReleaseProgram(program);
        return nullptr;
    }

    // Persist the device binary for next launch. fopen/fwrite/fflush/fsync/
    // fclose with explicit error checks at every step — std::ofstream's
    // implicit dtor close doesn't flush to durable storage, leaving a 0-byte
    // file behind if the process exits before the OS commits the page.
    size_t bin_sz = 0;
    if (clGetProgramInfo(program, CL_PROGRAM_BINARY_SIZES, sizeof(bin_sz),
                         &bin_sz, nullptr) == CL_SUCCESS && bin_sz > 0) {
        std::vector<unsigned char> bin(bin_sz);
        unsigned char* binp = bin.data();
        if (clGetProgramInfo(program, CL_PROGRAM_BINARIES, sizeof(binp),
                             &binp, nullptr) == CL_SUCCESS) {
            if (!nnopt_ensure_cache_dir()) {
                if (nnopt_kcache_log()) {
                    fprintf(stderr, "[kernel_cache] SAVE skipped — cache dir unavailable\n");
                }
            } else {
                FILE* fp = std::fopen(cache_path.c_str(), "wb");
                bool ok = false;
                if (fp) {
                    const size_t wrote = std::fwrite(bin.data(), 1, bin.size(), fp);
                    if (wrote == bin.size()) {
                        if (std::fflush(fp) == 0) {
                            const int fd = fileno(fp);
                            if (fd >= 0 && fsync(fd) == 0) {
                                ok = true;
                            } else if (nnopt_kcache_log()) {
                                fprintf(stderr, "[kernel_cache] fsync failed: %s\n",
                                        cache_path.c_str());
                            }
                        } else if (nnopt_kcache_log()) {
                            fprintf(stderr, "[kernel_cache] fflush failed: %s\n",
                                    cache_path.c_str());
                        }
                    } else if (nnopt_kcache_log()) {
                        fprintf(stderr, "[kernel_cache] short-write %zu/%zu: %s\n",
                                wrote, bin.size(), cache_path.c_str());
                    }
                    std::fclose(fp);
                } else if (nnopt_kcache_log()) {
                    fprintf(stderr, "[kernel_cache] fopen-for-write failed: %s\n",
                            cache_path.c_str());
                }
                if (ok && nnopt_kcache_log()) {
                    fprintf(stderr, "[kernel_cache] SAVE %s (%zu B)\n",
                            cache_path.c_str(), bin.size());
                }
                if (!ok) {
                    std::remove(cache_path.c_str());
                }
            }
        }
    }

    return program;
}

cl_program OpenCLContext::build_program(const std::string& source, const std::string& options) {
    // Forward host-side dtype to the kernel preamble. Without this, every
    // kernel falls through to the fp32 path of `#ifdef USE_FP16` and reads
    // garbage from cl_half buffers.
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

    // The cache key must include `effective_options` (not the raw `options`)
    // so an fp16 build and an fp32 build don't share a cache entry — different
    // -D defines compile to different device binaries.
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
// Per Adreno OpenCL guide §9.1.3 and the cross-port porting guide
// (/Users/alazarshenkute/Downloads/adreno-recordable-queues-porting-guide.md):
//   • Legacy clCreateCommandQueue with RECORDABLE_BIT returns -30 on Adreno.
//     Must use the OpenCL 2.0 clCreateCommandQueueWithProperties API.
//   • That API isn't in our CL 1.2 headers — dlsym it from RTLD_DEFAULT.
//   • RECORDABLE_BIT must be passed ALONE; OR'ing CL_QUEUE_PROFILING_ENABLE
//     returns -30.
cl_command_queue OpenCLContext::create_recordable_queue() {
    if (!record_fns_loaded_) return nullptr;

    typedef cl_command_queue (CL_API_CALL *fn_create_q_t)(
        cl_context, cl_device_id, const cl_ulong*, cl_int*);
    static fn_create_q_t fn_create_q = (fn_create_q_t)
        dlsym(RTLD_DEFAULT, "clCreateCommandQueueWithProperties");
    if (!fn_create_q) {
        fprintf(stderr, "RecordQ: dlsym(clCreateCommandQueueWithProperties) returned null\n");
        return nullptr;
    }

    // {CL_QUEUE_PROPERTIES, RECORDABLE_BIT, 0} — terminator is a single 0.
    // PROFILING is intentionally absent (incompatible with RECORDABLE on Adreno).
    const cl_ulong props[] = {
        (cl_ulong)NNOPT_CL_QUEUE_PROPERTIES,
        (cl_ulong)NNOPT_CL_QUEUE_RECORDABLE_QCOM,
        0
    };
    cl_int err = CL_SUCCESS;
    cl_command_queue q = fn_create_q(context_, device_, props, &err);
    if (err != CL_SUCCESS || !q) {
        fprintf(stderr, "RecordQ: clCreateCommandQueueWithProperties(RECORDABLE) failed err=%d\n",
                (int)err);
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
