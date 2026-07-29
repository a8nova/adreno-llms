// OpenCL vision tower for Bonsai-27B (Qwen3-VL, 27 blocks, hidden 1152).
//
// Runs ONCE per image, unlike the text tower which runs once per token. That asymmetry drives every
// design choice here: dense CLBlast GEMMs instead of the 1-bit path, weights dequantised to
// storage_t at upload instead of unpacked per use, and the whole tower freed after the image is
// encoded so text-only chat keeps its memory headroom.
//
// Structure is verified against the shipped mmproj, not assumed — see VISION_PORT.md for the tensor
// manifest and the dtype census (F32 224 / Q8_0 83 / F16 27). Numerics follow
// src/vision_reference.h, which is the CPU implementation this must agree with.
#pragma once
#ifdef BONSAI_VISION

#include <string>
#include <vector>

#include <cstdint>
#include <unordered_map>

#include "opencl_context.h"

/**
 * Loader for the vision .nnb (magic NNBVIS).
 *
 * Deliberately NOT the text Nnb: that one is NNB1BIT with kinds {Q1, F32, Q1T}, and the vision file
 * is a different container carrying {f32, f16, q8_0}. Sharing the type would mean widening an enum
 * the whole 1-bit path switches on, to describe formats it will never see.
 */
class VisionNnb {
  public:
    struct T { std::vector<int> dims; std::string kind; const uint8_t* data; size_t nbytes; };
    explicit VisionNnb(const std::string& path);
    ~VisionNnb();
    const T& get(const std::string& name) const;
    /** Drop the mmap pages for a tensor already copied to the device — see Nnb::release. */
    void release(const void* p, size_t n) const;
    std::unordered_map<std::string, std::string> meta;   // raw header meta, values as strings

  private:
    int fd_ = -1;
    const uint8_t* base_ = nullptr;
    size_t size_ = 0;
    std::unordered_map<std::string, T> tensors_;
};

/** Vision config, read from the .nnb header rather than hardcoded. */
struct VisionCfg {
    int depth = 27, hidden = 1152, ffn = 4304, heads = 16, head_dim = 72;
    int patch = 16, merge = 2, temporal_patch = 2, out_hidden = 5120, image_size = 768;
    float eps = 1e-6f;
    float mean[3] = {0.5f, 0.5f, 0.5f};
    float std[3] = {0.5f, 0.5f, 0.5f};
};

class VisionModel {
  public:
    VisionModel(const std::string& nnb_path, OpenCLContext& ocl, const std::string& kdir);
    ~VisionModel();

    const VisionCfg& cfg() const { return cfg_; }

    /**
     * Encode one RGB image into merged vision embeddings.
     *
     * `rgb` is HWC u8 at the ORIGINAL size; smart_resize + normalisation + patching happen inside.
     * Returns [n_tokens][out_hidden] in `out`, and the token grid in *gh/*gw so the caller can build
     * the mRoPE position ids — text-only mRoPE (identical T/H/W) stops being valid here.
     */
    void encode(const uint8_t* rgb, int h, int w, std::vector<float>* out, int* gh, int* gw);

    /** Tokens this image will produce, without running the tower — for prefill budgeting. */
    long token_count(int h, int w) const;

    /** Stage timings from the last encode(), for the benchmark line. */
    struct Timing { double total = 0, upload = 0, compute = 0, merger = 0; int patches = 0; };
    const Timing& timing() const { return timing_; }

  private:
    struct Mat { cl_mem w = nullptr, b = nullptr; };   // weight + optional bias, both storage_t
    struct Block { cl_mem ln1_w, ln1_b, ln2_w, ln2_b; Mat qkv, out, fc1, fc2; };

    // Dequantise a .nnb tensor (f32 / f16 / q8_0) to storage_t and upload. Doing it once at load
    // costs memory but removes the dequant from the inner loop entirely; the tower is only ~0.63 GB
    // quantised and runs once per image, so the tradeoff is clearly the right way round here.
    cl_mem upload_dequant(const std::string& name);
    cl_mem upload_dequant_sum2(const std::string& a, const std::string& b);
    Block load_block(int i);
    void free_block(Block& b);
    /** pixels -> [M, hidden]: smart_resize, normalise by mean/std, 3D patchify, project, +pos_embd. */
    void patch_embed(const uint8_t* rgb, int h, int w, int rh, int rw, int M, cl_mem out);
    /** post_ln -> 2x2 spatial merge -> mm.0 -> GELU(erf) -> mm.2 -> [n/4, out_hidden] fp32. */
    void merge_and_project(cl_mem hid, int M, std::vector<float>* out);
    /** Build cos/sin for `tokens` positions of the vision rope into cos_/sin_. */
    void make_rope(int gh, int gw);
    void probe(const char* tag, cl_mem buf, size_t n);

    VisionNnb nnb_;
    OpenCLContext& ocl_;
    VisionCfg cfg_;
    std::vector<Block> blocks_;
    cl_mem patch_w_ = nullptr, patch_b_ = nullptr;   // Conv3d folded to [1536 -> hidden]
    cl_mem pos_embd_ = nullptr;
    cl_mem post_ln_w_ = nullptr, post_ln_b_ = nullptr;
    Mat mm0_, mm2_;                                   // merger fc1 / fc2

    cl_mem cos_ = nullptr, sin_ = nullptr;   // vision rope tables, rebuilt per grid
    cl_kernel k_qkv_split_ = nullptr;
    cl_kernel k_scores_t_ = nullptr, k_softmax_t_ = nullptr, k_context_t_ = nullptr;
    cl_kernel k_attn_fused_ = nullptr;   // scores+softmax+context, probability row kept in local
    size_t local_mem_ = 0;               // CL_DEVICE_LOCAL_MEM_SIZE, gates the fused path
    Timing timing_;
    bool f16_ = true;   // fp16 tower (guide 7.2.4: 2x 16-bit ALU). BONSAI_VIS_FP32=1 reverts.
    size_t ES_ = 2;     // bytes per activation/weight element on device
    cl_kernel k_ln_ = nullptr, k_gelu_ = nullptr, k_gelu_tanh_ = nullptr,
              k_add_bias_ = nullptr, k_rope_ = nullptr, k_add_ = nullptr,
              k_scores_ = nullptr, k_softmax_ = nullptr, k_context_ = nullptr,
              k_head_major_ = nullptr;
};

#endif  // BONSAI_VISION
