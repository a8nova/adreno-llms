#pragma once
#include <string>
#include <vector>

// Define OpenCL version BEFORE including headers to avoid version conflicts
#define CL_TARGET_OPENCL_VERSION 120

// Always use portable Khronos headers (works for cross-compilation to Android)
#include <CL/cl.h>

class OpenCLContext {
public:
    OpenCLContext();
    ~OpenCLContext();

    bool initialize(int platform_idx = 0, int device_idx = 0);
    cl_program build_program(const std::string& source, const std::string& options = "");
    cl_program build_program_from_file(const std::string& path, const std::string& options = "");
    // OPT-5: `<path>.bin` on-disk program binary cache (guide §5.7.3) —
    // skips clBuildProgram-from-source on every run after the first.
    // NNOPT_PROG_CACHE=0 bypasses for A/B.
    cl_program load_or_build_program(const std::string& path,
                                     const std::string& source,
                                     const std::string& options = "");

    cl_context context() const { return context_; }
    cl_command_queue queue() const { return queue_; }
    cl_device_id device() const { return device_; }

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

// ── Dispatch-overhead killers (profiled: DiT wall was ~80% overhead) ────────
//
// nnopt_cached_kernel: clCreateKernel/clReleaseKernel per dispatch costs more
// than many of the kernels themselves on Adreno. Returns a process-lifetime
// cached kernel object per (program, name). Safe single-threaded: args are
// snapshotted by clEnqueueNDRangeKernel. NNOPT_KERNEL_CACHE=0 restores the
// create/release-per-dispatch behavior for A/B.
cl_kernel nnopt_cached_kernel(cl_program prog, const char* name, cl_int* err_out);
void nnopt_kernel_done(cl_kernel k);   // releases only when the cache is off

// nnopt_pool_alloc/release: per-op clCreateBuffer/clReleaseMemObject churn is
// slow AND fragments the CL heap (the intermittent 128 MB alloc failure).
// Exact-size free-list pool; activation shapes are fixed per graph position,
// so the hit rate is ~100% after the first step. Buffers arrive with
// UNDEFINED contents, same contract as a fresh clCreateBuffer. release() on
// a buffer the pool doesn't own does a real clReleaseMemObject, so it is
// always safe to route releases through the pool. NNOPT_BUF_POOL=0 for A/B.
cl_mem nnopt_pool_alloc(cl_context ctx, size_t bytes, cl_int* err_out);
void nnopt_pool_release(cl_mem buf);
