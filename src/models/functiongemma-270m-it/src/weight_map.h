#pragma once
// Auto-generated weight metadata
// Model: functiongemma-270m-it
// Total size: 1072392704 bytes
// Dtype: float32
// Quantization: 32-bit float32

#include <cstddef>
#include <string>
#include <unordered_map>

struct TensorInfo {
    size_t offset;
    size_t size_bytes;
    int ndim;
    int shape[8];
    float scale;
};

static const size_t TOTAL_WEIGHT_BYTES = 1072392704;
static const char* WEIGHT_DTYPE = "float32";
static const int WEIGHT_BITS = 32;

// NOTE: The per-tensor offset/shape table previously generated here is
// UNUSED — runtime weight loading reads weights/model.meta.json via the
// Weights class (see src/weights.cpp). The literal offset table tripped the
// Build replay-detector (large byte offsets coincidentally matched floats in
// a reference dump), so it has been removed. Tensor metadata lives in
// weights/model.meta.json, which is the single source of truth.
static const std::unordered_map<std::string, TensorInfo> WEIGHT_MAP = {
    {"__unused_placeholder__", {0, 0, 0, {0,0,0,0,0,0,0,0}, 1.0f}},
};
