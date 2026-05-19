#include "weights.h"
#include "debug_utils.h"
#include <fstream>
#include <iostream>
#include <cstdlib>   // std::getenv (weight roundtrip mode)
#include <cstring>
#include <sstream>
#include <algorithm>
// mmap dependencies — POSIX (Android, Linux, macOS).
#include <fcntl.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <unistd.h>

// Minimal JSON metadata parser for model.meta.json
// Expected format: {"tensors": {"name": {"offset": N, "size": N, "shape": [d1,d2,...], "dtype": "float32"}, ...}}
static std::string trim(const std::string& s) {
    size_t start = s.find_first_not_of(" \t\n\r\"");
    size_t end = s.find_last_not_of(" \t\n\r\"");
    return (start == std::string::npos) ? "" : s.substr(start, end - start + 1);
}

static bool parse_meta_json(const std::string& json,
                            std::unordered_map<std::string, std::tuple<size_t, size_t, std::vector<int>, std::string>>& out) {
    // Find "tensors" object
    size_t tensors_pos = json.find("\"tensors\"");
    if (tensors_pos == std::string::npos) {
        // Flat format: top-level keys are tensor names directly
        tensors_pos = 0;
    } else {
        tensors_pos = json.find('{', tensors_pos + 9);
        if (tensors_pos == std::string::npos) return false;
    }

    // Iterate through tensor entries by finding "key": { ... } patterns
    size_t pos = tensors_pos;
    while (pos < json.size()) {
        // Find next key
        size_t key_start = json.find('"', pos);
        if (key_start == std::string::npos) break;
        size_t key_end = json.find('"', key_start + 1);
        if (key_end == std::string::npos) break;
        std::string key = json.substr(key_start + 1, key_end - key_start - 1);

        // Skip non-tensor keys
        if (key == "tensors" || key == "format" || key == "metadata") {
            pos = key_end + 1;
            // Skip past the value
            size_t brace = json.find('{', pos);
            size_t next_quote = json.find('"', pos + 1);
            if (brace != std::string::npos && (next_quote == std::string::npos || brace < next_quote)) {
                int depth = 1;
                pos = brace + 1;
                while (pos < json.size() && depth > 0) {
                    if (json[pos] == '{') depth++;
                    else if (json[pos] == '}') depth--;
                    pos++;
                }
            }
            continue;
        }

        // Find opening brace for this tensor's metadata
        size_t obj_start = json.find('{', key_end);
        if (obj_start == std::string::npos) break;

        // Find matching closing brace
        int depth = 1;
        size_t obj_end = obj_start + 1;
        while (obj_end < json.size() && depth > 0) {
            if (json[obj_end] == '{') depth++;
            else if (json[obj_end] == '}') depth--;
            obj_end++;
        }

        std::string obj = json.substr(obj_start, obj_end - obj_start);

        // Parse offset
        size_t offset_val = 0;
        size_t offset_pos = obj.find("\"offset\"");
        if (offset_pos != std::string::npos) {
            size_t colon = obj.find(':', offset_pos);
            if (colon != std::string::npos) {
                offset_val = std::stoull(trim(obj.substr(colon + 1, obj.find_first_of(",}", colon) - colon - 1)));
            }
        }

        // Parse size
        size_t size_val = 0;
        size_t size_pos = obj.find("\"size\"");
        if (size_pos == std::string::npos) size_pos = obj.find("\"size_bytes\"");
        if (size_pos != std::string::npos) {
            size_t colon = obj.find(':', size_pos);
            if (colon != std::string::npos) {
                size_val = std::stoull(trim(obj.substr(colon + 1, obj.find_first_of(",}", colon) - colon - 1)));
            }
        }

        // Parse shape array
        std::vector<int> shape;
        size_t shape_pos = obj.find("\"shape\"");
        if (shape_pos != std::string::npos) {
            size_t arr_start = obj.find('[', shape_pos);
            size_t arr_end = obj.find(']', arr_start);
            if (arr_start != std::string::npos && arr_end != std::string::npos) {
                std::string arr = obj.substr(arr_start + 1, arr_end - arr_start - 1);
                std::istringstream ss(arr);
                std::string token;
                while (std::getline(ss, token, ',')) {
                    std::string t = trim(token);
                    if (!t.empty()) shape.push_back(std::stoi(t));
                }
            }
        }

        // Parse per-tensor dtype (defaults to "float32" — the meta.json format
        // emitted by convertWeightsTs always sets this; older meta.json files
        // without dtype are treated as fp32).
        std::string dtype_val = "float32";
        size_t dtype_pos = obj.find("\"dtype\"");
        if (dtype_pos != std::string::npos) {
            size_t colon = obj.find(':', dtype_pos);
            if (colon != std::string::npos) {
                size_t q1 = obj.find('"', colon);
                if (q1 != std::string::npos) {
                    size_t q2 = obj.find('"', q1 + 1);
                    if (q2 != std::string::npos) {
                        dtype_val = obj.substr(q1 + 1, q2 - q1 - 1);
                    }
                }
            }
        }

        if (size_val > 0) {
            out[key] = {offset_val, size_val, shape, dtype_val};
        }

        pos = obj_end;
    }
    return !out.empty();
}

// Bytes-per-element for the recognized dtype strings emitted by ConvertWeights.
static inline size_t _nnopt_bytes_per_element(const std::string& dtype) {
    if (dtype == "float16" || dtype == "bfloat16") return 2;
    return 4;  // float32 default
}

Weights::~Weights() {
    if (mapped_ != nullptr && mapped_ != MAP_FAILED) {
        munmap(mapped_, mapped_size_);
        mapped_ = nullptr;
    }
    if (mapped_fd_ >= 0) {
        close(mapped_fd_);
        mapped_fd_ = -1;
    }
    for (auto& [k, meta] : tensors_) {
        if (meta.buffer) {
            clReleaseMemObject(meta.buffer);
            meta.buffer = nullptr;
        }
    }
}

bool Weights::load(const std::string& bin_path, const std::string& meta_path, cl_context ctx) {
    ctx_ = ctx;
    NNOPT_CHECKPOINT("weights: opening binary file");

    // Memory-map the weight file. Crucial on memory-constrained devices:
    // a single 1.3GB allocation via std::vector::resize() will trigger the
    // Android lowmemorykiller; mmap(MAP_PRIVATE) lets the kernel page in
    // only the tensor regions actually accessed and reclaim them under
    // pressure without killing us.
    mapped_fd_ = open(bin_path.c_str(), O_RDONLY);
    if (mapped_fd_ < 0) {
        NNOPT_ERROR_FMT("weights: failed to open %s for mmap (errno=%d)", bin_path.c_str(), errno);
        return false;
    }

    struct stat st;
    if (fstat(mapped_fd_, &st) != 0 || st.st_size <= 0) {
        NNOPT_ERROR_FMT("weights: fstat failed for %s (errno=%d)", bin_path.c_str(), errno);
        close(mapped_fd_);
        mapped_fd_ = -1;
        return false;
    }
    mapped_size_ = (size_t)st.st_size;

    NNOPT_CHECKPOINT_FMT("weights: mmapping %zuMB from %s",
                         mapped_size_ / (1024*1024), bin_path.c_str());
    mapped_ = (uint8_t*)mmap(nullptr, mapped_size_, PROT_READ, MAP_PRIVATE, mapped_fd_, 0);
    if (mapped_ == MAP_FAILED) {
        NNOPT_ERROR_FMT("weights: mmap failed (errno=%d) — falling back to read", errno);
        mapped_ = nullptr;
        // Fallback: streamed read into a heap allocation. This is the old
        // behavior; OOM risk applies. Kept so the runtime works on any FS
        // that doesn't support mmap (rare).
        mapped_ = (uint8_t*)malloc(mapped_size_);
        if (!mapped_) {
            NNOPT_ERROR("weights: malloc fallback failed");
            close(mapped_fd_);
            mapped_fd_ = -1;
            return false;
        }
        std::ifstream bin_file(bin_path, std::ios::binary);
        bin_file.read(reinterpret_cast<char*>(mapped_), mapped_size_);
    } else {
        // Tensor reads are sparse and order-independent — tell the kernel.
        madvise(mapped_, mapped_size_, MADV_RANDOM);
    }
    NNOPT_CHECKPOINT("weights: mmap complete (resident set will grow on demand)");

    // Load and parse metadata JSON
    std::ifstream meta_file(meta_path);
    if (!meta_file.is_open()) {
        std::cerr << "Failed to open metadata: " << meta_path << std::endl;
        return false;
    }

    std::string meta_str((std::istreambuf_iterator<char>(meta_file)),
                          std::istreambuf_iterator<char>());

    std::unordered_map<std::string, std::tuple<size_t, size_t, std::vector<int>, std::string>> parsed;
    if (!parse_meta_json(meta_str, parsed)) {
        std::cerr << "Warning: Could not parse weight metadata from " << meta_path << std::endl;
        std::cerr << "Loaded " << mapped_size_ << " bytes of raw weights (no tensor mapping)" << std::endl;
        return true;
    }

    // Populate tensors_ map from parsed metadata (no GPU buffers yet).
    // num_elements derives from per-tensor dtype, not a hardcoded /4 — fp16
    // tensors store 2 bytes/elem, fp32 4 bytes/elem; mixed-precision ports
    // (e.g. fp32 embeddings + fp16 attention projections under fp16 build)
    // need this distinction.
    for (auto& [name, info] : parsed) {
        auto& [offset, size_bytes, shape, dtype] = info;
        TensorMeta meta;
        meta.offset = offset;
        meta.size_bytes = size_bytes;
        meta.dtype = dtype;
        meta.num_elements = size_bytes / _nnopt_bytes_per_element(dtype);
        meta.shape = shape;
        meta.buffer = nullptr;
        tensors_[name] = meta;
    }

    std::cerr << "Loaded " << mapped_size_ << " bytes, " << tensors_.size() << " tensors" << std::endl;
    NNOPT_CHECKPOINT_FMT("weights: parsed metadata for %zu tensors (no GPU buffers yet)",
                         tensors_.size());

    // GPU buffers are created lazily by get_buffer() to avoid double allocation.
    // Layers that call set_weights(get_host(...)) create their own buffers.
    // Layers that call get_buffer() get a buffer created on-demand here.

    return true;
}

bool Weights::has_tensor(const std::string& key) const {
    return tensors_.find(key) != tensors_.end();
}

// Weight-upload roundtrip verifier. Reads back a small sample of the GPU
// buffer (first 64 bytes) and compares byte-for-byte against the host mmap
// region. If they disagree the GPU upload is silently corrupting weights —
// an entire bug class that would otherwise look like a math error in the
// first kernel that consumes the tensor (yesterday's OpenELM trace had
// embedding_wte producing NaNs with no diagnosable kernel-level cause; a
// roundtrip check here would have flagged it as upload corruption instead).
//
// Runs ONCE per process (first get_buffer() call). If that one passes the
// upload path is healthy and we don't pay the cost on every subsequent
// tensor. Set NNOPT_VERIFY_WEIGHTS=0 to disable; set =1 (default) or =all
// to enable for every tensor (slow but useful when chasing a real bug).
static int _nnopt_verify_mode() {
    const char* env = std::getenv("NNOPT_VERIFY_WEIGHTS");
    if (!env) return 1; // default: verify first tensor only
    if (std::strcmp(env, "0") == 0 || std::strcmp(env, "off") == 0 || std::strcmp(env, "false") == 0) return 0;
    if (std::strcmp(env, "all") == 0 || std::strcmp(env, "2") == 0) return 2;
    return 1;
}

static bool _nnopt_roundtrip_verified_once = false;

static bool _nnopt_verify_buffer_roundtrip(
    cl_context ctx,
    cl_mem buffer,
    const uint8_t* host_bytes,
    size_t total_bytes,
    const char* key
) {
    if (!ctx || !buffer || !host_bytes || total_bytes == 0) return true;
    cl_device_id device = nullptr;
    cl_int err = clGetContextInfo(ctx, CL_CONTEXT_DEVICES, sizeof(device), &device, nullptr);
    if (err != CL_SUCCESS || !device) return true; // can't verify — be permissive
    cl_command_queue q = clCreateCommandQueue(ctx, device, 0, &err);
    if (err != CL_SUCCESS || !q) return true;

    // Sample windows: first 64 bytes + last 64 bytes (or up to total).
    const size_t SAMPLE = 64;
    const size_t front = std::min(SAMPLE, total_bytes);
    uint8_t front_gpu[SAMPLE] = {};
    err = clEnqueueReadBuffer(q, buffer, CL_TRUE, 0, front, front_gpu, 0, nullptr, nullptr);
    bool ok = (err == CL_SUCCESS);
    if (ok) {
        for (size_t i = 0; i < front; i++) {
            if (front_gpu[i] != host_bytes[i]) {
                NNOPT_ERROR_FMT(
                    "weight upload mismatch for '%s' at front offset %zu: gpu=0x%02x host=0x%02x. "
                    "GPU buffer does not match host mmap bytes — every kernel reading this tensor "
                    "will see corrupted data (looks like math errors, but is upload corruption). "
                    "Likely causes: dtype mismatch, wrong meta.json offset, byte-swap, OOB allocation.",
                    key, i, (unsigned)front_gpu[i], (unsigned)host_bytes[i]);
                ok = false;
                break;
            }
        }
    }
    if (ok && total_bytes > SAMPLE * 2) {
        const size_t tail_off = total_bytes - SAMPLE;
        uint8_t tail_gpu[SAMPLE] = {};
        err = clEnqueueReadBuffer(q, buffer, CL_TRUE, tail_off, SAMPLE, tail_gpu, 0, nullptr, nullptr);
        if (err == CL_SUCCESS) {
            for (size_t i = 0; i < SAMPLE; i++) {
                if (tail_gpu[i] != host_bytes[tail_off + i]) {
                    NNOPT_ERROR_FMT(
                        "weight upload mismatch for '%s' at tail offset %zu: gpu=0x%02x host=0x%02x.",
                        key, tail_off + i, (unsigned)tail_gpu[i], (unsigned)host_bytes[tail_off + i]);
                    ok = false;
                    break;
                }
            }
        }
    }

    clReleaseCommandQueue(q);
    return ok;
}

cl_mem Weights::get_buffer(const std::string& key, bool optional) {
    auto it = tensors_.find(key);
    if (it == tensors_.end()) {
        if (!optional) {
            NNOPT_ERROR_FMT("weight key not found: '%s'", key.c_str());
        }
        return nullptr;
    }

    auto& meta = it->second;
    if (meta.buffer != nullptr) return meta.buffer;

    // Lazy creation: first access creates the GPU buffer
    if (ctx_ == nullptr) return nullptr;
    if (meta.offset + meta.size_bytes > mapped_size_) {
        std::cerr << "Warning: tensor " << key << " exceeds file bounds" << std::endl;
        return nullptr;
    }

    cl_int err;
    meta.buffer = tracked_clCreateBuffer(
        ctx_, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
        meta.size_bytes,
        mapped_ + meta.offset,
        &err, key.c_str()
    );
    if (err != CL_SUCCESS) {
        std::cerr << "Failed to create buffer for " << key << " (err=" << err << ")" << std::endl;
        meta.buffer = nullptr;
        return nullptr;
    }

    // Roundtrip verification — see _nnopt_verify_mode() comment above.
    const int verify_mode = _nnopt_verify_mode();
    const bool should_verify =
        (verify_mode == 2) ||                                // verify all
        (verify_mode == 1 && !_nnopt_roundtrip_verified_once); // verify first only
    if (should_verify) {
        const bool ok = _nnopt_verify_buffer_roundtrip(
            ctx_, meta.buffer, mapped_ + meta.offset, meta.size_bytes, key.c_str()
        );
        if (!ok) {
            // Tear down so the corrupt buffer doesn't propagate — the agent gets
            // a hard error instead of cascading NaNs.
            clReleaseMemObject(meta.buffer);
            meta.buffer = nullptr;
            return nullptr;
        }
        _nnopt_roundtrip_verified_once = true;
    }

    return meta.buffer;
}

const float* Weights::get_host(const std::string& key) const {
    auto it = tensors_.find(key);
    if (it == tensors_.end()) return nullptr;
    // CRITICAL: this returns raw bytes cast to float*. Only safe when the
    // on-disk dtype IS float32. Under fp16 builds (model.fp16.bin), tensors
    // are stored as float16 — reinterpreting those bytes as float32 reads
    // pairs of fp16 values as one fp32 = garbage. Caller has TWO options:
    //   (1) call get_host_vec() instead — it dtype-decodes to float32.
    //   (2) only call get_host() when meta.dtype == "float32".
    // We refuse the unsafe path here so any latent caller fails loudly
    // instead of silently producing NaN-cascade-worth of corrupted weights.
    // Past incident: under fp16, A_log loaded via get_host() corrupted A,
    // which corrupted exp(dt*A), which produced NaN scan_output, which
    // cascaded to empty transcripts for hours. See get_host_vec() below.
    if (it->second.dtype != "float32") {
        std::cerr << "Weights::get_host(\"" << key << "\"): refusing to return raw float* "
                  << "for tensor of dtype \"" << it->second.dtype << "\". "
                  << "Use get_host_vec() instead — it decodes fp16/bf16 to float32." << std::endl;
        return nullptr;
    }
    return reinterpret_cast<const float*>(mapped_ + it->second.offset);
}

std::string Weights::get_dtype(const std::string& key) const {
    auto it = tensors_.find(key);
    if (it == tensors_.end()) return "";
    return it->second.dtype;
}

// IEEE 754 binary16 → float32. Bit-exact, branch-light. Same algorithm as
// utils.cpp::nnopt_f16_to_f32 — duplicated here so weights.cpp stays
// independent of utils.h.
static inline float _wts_f16_to_f32(uint16_t bits) {
    uint32_t sign = (uint32_t)(bits >> 15) & 0x1u;
    uint32_t exp  = (uint32_t)(bits >> 10) & 0x1Fu;
    uint32_t mant = (uint32_t)(bits      ) & 0x3FFu;
    uint32_t out_sign = sign << 31;
    uint32_t out;
    if (exp == 0) {
        if (mant == 0) { out = out_sign; }
        else {
            int e = -1; do { e++; mant <<= 1; } while ((mant & 0x400u) == 0);
            mant &= 0x3FFu; uint32_t out_exp = (uint32_t)(127 - 15 - e);
            out = out_sign | (out_exp << 23) | (mant << 13);
        }
    } else if (exp == 0x1F) { out = out_sign | 0x7F800000u | (mant << 13); }
    else { uint32_t out_exp = (uint32_t)(exp - 15 + 127);
           out = out_sign | (out_exp << 23) | (mant << 13); }
    float f; std::memcpy(&f, &out, sizeof(f)); return f;
}

// bfloat16 → float32 (just shift left 16 — top 16 bits of fp32 layout).
static inline float _wts_bf16_to_f32(uint16_t bits) {
    uint32_t b = ((uint32_t)bits) << 16;
    float f; std::memcpy(&f, &b, sizeof(f)); return f;
}

std::vector<float> Weights::get_host_vec(const std::string& key) const {
    auto it = tensors_.find(key);
    if (it == tensors_.end()) return {};
    const auto& meta = it->second;
        const uint8_t* base = mapped_ + meta.offset;

    std::vector<float> out(meta.num_elements);
    if (meta.dtype == "float16") {
        const uint16_t* p = reinterpret_cast<const uint16_t*>(base);
        for (size_t i = 0; i < meta.num_elements; i++) out[i] = _wts_f16_to_f32(p[i]);
    } else if (meta.dtype == "bfloat16") {
        const uint16_t* p = reinterpret_cast<const uint16_t*>(base);
        for (size_t i = 0; i < meta.num_elements; i++) out[i] = _wts_bf16_to_f32(p[i]);
    } else {
        // float32 (default)
        const float* p = reinterpret_cast<const float*>(base);
        std::memcpy(out.data(), p, meta.num_elements * sizeof(float));
    }
    return out;
}

size_t Weights::get_size_bytes(const std::string& key) const {
    auto it = tensors_.find(key);
    return (it != tensors_.end()) ? it->second.size_bytes : 0;
}

size_t Weights::get_num_elements(const std::string& key) const {
    auto it = tensors_.find(key);
    return (it != tensors_.end()) ? it->second.num_elements : 0;
}

std::vector<int> Weights::get_shape(const std::string& key) const {
    auto it = tensors_.find(key);
    return (it != tensors_.end()) ? it->second.shape : std::vector<int>{};
}
