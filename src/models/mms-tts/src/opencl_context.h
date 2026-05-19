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

    // Program cache (build once, reuse everywhere)
    bool ensure_program_from_file(const std::string& path, const std::string& options = "");
    cl_program get_program(const std::string& path) const;

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

    struct ProgramEntry {
        std::string path;
        cl_program program;
    };
    std::vector<ProgramEntry> program_cache_;
};
