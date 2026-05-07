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

    cl_context context() const { return context_; }
    cl_command_queue queue() const { return queue_; }
    cl_device_id device() const { return device_; }

    // Lazy-built shared program for kernels/utils.cl (element_add, split_last_dim_2).
    // Layer code passes this to the free helpers in utils.h.
    cl_program utils_program();

    // Device info
    std::string device_name() const;
    std::string device_description() const;
    size_t max_work_group_size() const;
    size_t local_mem_size() const;

    // Build a program with the same on-device kernel-binary cache as
    // build_program(), but standalone — derives ctx + device from the queue
    // and applies the same key (source + options + device_name + driver).
    static cl_program build_cached_program_from_queue(
        cl_command_queue queue,
        const std::string& source,
        const std::string& options);

private:
    cl_platform_id platform_ = nullptr;
    cl_device_id device_ = nullptr;
    cl_context context_ = nullptr;
    cl_command_queue queue_ = nullptr;
    cl_program utils_program_ = nullptr;
};
