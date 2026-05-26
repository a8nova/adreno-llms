#pragma once
#include <string>
#include <vector>

// Define OpenCL version BEFORE including headers to avoid version conflicts
#define CL_TARGET_OPENCL_VERSION 120

// Always use portable Khronos headers (works for cross-compilation to Android)
#include <CL/cl.h>

// On-disk OpenCL program binary cache. Compiles source on first call; subsequent
// calls with the same (source + options + device + driver) skip the slow source
// build path via `clCreateProgramWithBinary`. Cache lives at
// `<cwd>/kernel_cache/<fnv1a64-hex>.bin`. Cuts cold-start TTFT by skipping
// ~100ms × N kernel builds.
//
// On build-from-binary failure (binary stale or invalidated by driver upgrade),
// falls back to source compilation and overwrites the cache entry. No manual
// invalidation step needed — the hash key encodes everything that matters.
cl_program nnopt_build_program_cached(cl_context ctx, cl_device_id dev,
                                      const std::string& source,
                                      const std::string& options);

class OpenCLContext {
public:
    OpenCLContext();
    ~OpenCLContext();

    bool initialize(int platform_idx = 0, int device_idx = 0);
    cl_program build_program(const std::string& source, const std::string& options = "");
    cl_program build_program_from_file(const std::string& path, const std::string& options = "");

    cl_context context() const { return context_; }
    cl_command_queue queue() const { return queue_; }
    cl_device_id device() const { return device_; }

    // Flush the command queue and sleep briefly so SurfaceFlinger can grab a
    // vsync slot on the GPU. Useful between large kernel batches (e.g. between
    // vision transformer layers) so foreground UI doesn't go white during
    // bursty multi-second GPU compute. Gated by NNOPT_GPU_YIELD env var —
    // benchmark builds leave it unset for max throughput.
    //
    // Cost: blocks the calling thread for `sleep_ms` (default 20 ms). Call
    // sparingly — once per layer is plenty.
    void yield_for_compositor(int sleep_ms = 20);

    // Device info
    std::string device_name() const;
    size_t max_work_group_size() const;
    size_t local_mem_size() const;

private:
    cl_platform_id platform_ = nullptr;
    cl_device_id device_ = nullptr;
    cl_context context_ = nullptr;
    cl_command_queue queue_ = nullptr;
};
