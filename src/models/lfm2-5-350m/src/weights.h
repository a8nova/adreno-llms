#pragma once
#include <string>
#include <unordered_map>
#include <vector>
#include <cstdint>

// Define OpenCL version BEFORE including headers to avoid version conflicts
#define CL_TARGET_OPENCL_VERSION 120

// Always use portable Khronos headers (works for cross-compilation to Android)
#include <CL/cl.h>

class Weights {
public:
    bool load(const std::string& bin_path, const std::string& meta_path, cl_context ctx);

    // Check if a tensor key exists in the weight file
    bool has_tensor(const std::string& key) const;

    // Get a weight tensor as an OpenCL buffer. Creates the buffer lazily on
    // first call (avoids double allocation if layers also call set_weights).
    //
    // If optional=true, a missing key returns nullptr silently (no stderr
    // spam). Use this for tensors that legitimately may not exist for a
    // given model (e.g. bias tensors on models that disable bias, or
    // tokenizer-specific padding tensors). Default (optional=false) logs
    // every miss loudly since those are real bugs.
    cl_mem get_buffer(const std::string& key, bool optional = false);

    // Get raw float pointer (host memory). Returns nullptr if key not found.
    const float* get_host(const std::string& key) const;

    // Get size in bytes for a tensor (useful for layers to verify buffer sizes)
    size_t get_size_bytes(const std::string& key) const;

    size_t get_num_elements(const std::string& key) const;
    std::vector<int> get_shape(const std::string& key) const;

    // Returns the tensor as a host-side std::vector<float> regardless of its
    // on-disk dtype — fp16/bfloat16 tensors are decoded transparently. Use
    // this for inspecting weights (sin/cos tables, embeddings, etc.) without
    // having to branch on dtype at every call site. Defined in weights.cpp.
    std::vector<float> get_host_vec(const std::string& key) const;

    // Returns the per-tensor dtype string from meta.json ("float32",
    // "float16", "bfloat16", "int8"). Empty string if key not found.
    std::string get_dtype(const std::string& key) const;

    // Enumerate all tensor names known to the weight file. Used by the int8
    // wiring in main.cpp to find pairs of `name` (int8) + `name.scale` (fp16).
    std::vector<std::string> all_keys() const;

    // Destructor — munmaps the weight file and releases any GPU buffers
    // created by get_buffer().
    ~Weights();

private:
    // Memory-mapped weight file. We use mmap(MAP_PRIVATE) instead of
    // reading the full file into a vector because:
    //   1. The OS kernel only faults in pages that are actually accessed,
    //      so a 1.3 GB model on a 2 GB-available phone has a working set
    //      of just the active layer's weights, not the whole file.
    //   2. Linux/Android can reclaim these pages under memory pressure
    //      without killing the process — a vector<uint8_t> of the same
    //      size is anonymous memory that triggers OOM-kill instead.
    //   3. No double-buffering during weight load — the OS doesn't have
    //      to copy file bytes into the heap.
    uint8_t* mapped_ = nullptr;
    size_t mapped_size_ = 0;
    int mapped_fd_ = -1;

    struct TensorMeta {
        size_t offset;
        size_t size_bytes;
        size_t num_elements;
        std::string dtype;       // "float32" / "float16" / "bfloat16" — from meta.json
        std::vector<int> shape;
        cl_mem buffer = nullptr;
    };

    std::unordered_map<std::string, TensorMeta> tensors_;
    cl_context ctx_ = nullptr;
};
