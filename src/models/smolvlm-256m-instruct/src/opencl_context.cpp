#include "opencl_context.h"
#include "debug_utils.h"   // NNOPT_ERROR_FMT — used in build_program / build_program_from_file below.

#include <fstream>
#include <sstream>
#include <iostream>
#include <cstring>
#include <cstdlib>
#include <cstdio>
#include <cstdint>
#include <chrono>       // std::this_thread::sleep_for — yield_for_compositor
#include <thread>       // std::this_thread::sleep_for — yield_for_compositor
#include <sys/stat.h>   // mkdir / stat — kernel-binary cache directory
#include <unistd.h>     // fsync — durably flush the cache write to disk before process exit

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
    //
    // The PRIORITY hint controls how the Adreno driver schedules our compute
    // against the system's GPU work — most importantly SurfaceFlinger, which
    // composites every frame on the same GPU. HIGH wins us a few percent of
    // throughput for benchmark runs but starves the compositor, producing
    // visible UI stutter and (on some Adreno paths) white-frame drops in
    // foreground apps. The NNOPT_QCOM_PRIORITY env var lets demo/foreground
    // builds opt into LOW so the UI thread always gets its frames in.
    //   unset / HIGH   → CL_PRIORITY_HINT_HIGH_QCOM   (benchmark default)
    //   NORMAL         → CL_PRIORITY_HINT_NORMAL_QCOM (driver default scheduling)
    //   LOW            → CL_PRIORITY_HINT_LOW_QCOM    (yield to compositor)
    cl_uint priority_hint = CL_PRIORITY_HINT_HIGH_QCOM;
    if (const char* pe = std::getenv("NNOPT_QCOM_PRIORITY")) {
        std::string s(pe);
        if (s == "LOW" || s == "low")         priority_hint = CL_PRIORITY_HINT_LOW_QCOM;
        else if (s == "NORMAL" || s == "normal") priority_hint = CL_PRIORITY_HINT_NORMAL_QCOM;
        fprintf(stderr, "[opencl_context] NNOPT_QCOM_PRIORITY=%s → priority_hint=0x%x\n", pe, priority_hint);
    }
    cl_context_properties ctx_props[] = {
        CL_CONTEXT_PERF_HINT_QCOM,     CL_PERF_HINT_HIGH_QCOM,
        CL_CONTEXT_PRIORITY_HINT_QCOM, (cl_context_properties)priority_hint,
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
        fprintf(stderr, "─────────────────────────────────────────────────────────────\n");
#ifdef NNOPT_DEBUG
        // Debug-only: full extensions dump for diagnostics.
        fprintf(stderr, "  extensions      %s\n", ext_buf);
#endif
        fflush(stderr);
    }

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

// stat()-check then mkdir(). Returns true iff the directory exists and is
// writable after the call. Older code did `mkdir(... , 0755)` and ignored
// the return — silently producing a never-saved cache when the parent
// path didn't exist or was non-writable (e.g. on a freshly-installed APK
// where filesDir/smolvlm/kernel_cache hasn't been created yet).
static bool nnopt_ensure_cache_dir() {
    struct stat st{};
    if (stat("kernel_cache", &st) == 0) {
        return S_ISDIR(st.st_mode);
    }
    if (mkdir("kernel_cache", 0755) == 0) {
        return true;
    }
    // EEXIST is benign (race with another caller).
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
    // Logs HIT / MISS / corrupt-file / build-failure outcomes when
    // NNOPT_KCACHE_LOG=1. Previously this path was silent on miss, so a
    // never-saved cache file was invisible — first-launch and every launch
    // looked identical (paid full JIT cost). Diagnostic logging gated to
    // avoid spamming production stderr.
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
            clGetProgramBuildInfo(program, dev, CL_PROGRAM_BUILD_LOG, log_size,
                                  log.data(), nullptr);
            fprintf(stderr, "OpenCL Build Log: %s\n", log.data());
            fflush(stderr);
        }
        clReleaseProgram(program);
        return nullptr;
    }

    // Persist the device binary for next launch. The previous version of
    // this block let `std::ofstream` go out of scope without explicit close,
    // no good() check, no fsync — on Android, if the app process exited
    // before the OS flushed the page cache, the file ended up
    // created-but-empty and every subsequent launch silently fell back to
    // JIT. Hence the careful write+close+fsync sequence below.
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
                // Use C-level FILE* so we can fsync() before close to durably
                // commit the bytes — std::ofstream offers no portable way to
                // get the underlying fd. Failures of any step are logged but
                // non-fatal (next run just re-JITs).
                FILE* fp = std::fopen(cache_path.c_str(), "wb");
                bool ok = false;
                if (fp) {
                    const size_t wrote = std::fwrite(bin.data(), 1, bin.size(), fp);
                    if (wrote == bin.size()) {
                        // Flush libc buffer to the kernel.
                        if (std::fflush(fp) == 0) {
                            // Push kernel buffer to durable storage so an
                            // app exit (intentional or crash) before the
                            // periodic writeback won't leave us with an
                            // empty file.
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
                    // Leave nothing behind that the next run would mistake
                    // for a valid cache file.
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

void OpenCLContext::yield_for_compositor(int sleep_ms) {
    // Cheap: env var lookup is a string compare against process env on first
    // call; the JVM-style "always check" is fine here since this fires at most
    // ~12 times per vision_pipe call (once per transformer layer).
    static const bool enabled = ([](){
        const char* e = std::getenv("NNOPT_GPU_YIELD");
        return e != nullptr && e[0] != '0' && e[0] != '\0';
    })();
    if (!enabled || !queue_) return;
    // Drain the queue so the GPU actually goes idle (clFlush is non-blocking
    // and the driver might keep batching dispatches). clFinish blocks until
    // every prior enqueued command has completed on the device.
    clFinish(queue_);
    if (sleep_ms > 0) {
        std::this_thread::sleep_for(std::chrono::milliseconds(sleep_ms));
    }
}
