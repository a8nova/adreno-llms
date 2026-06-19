// SigLIP vision encoder forward (LFM2-VL vision tower).
// Reference: model_info/transformers_src/modeling_lfm2_vl.py (Lfm2VlMultiModalProjector)
//            and transformers' modeling_siglip2.Siglip2VisionTransformer +
//            Siglip2EncoderLayer (referenced indirectly via the LFM2-VL
//            `model.vision_tower` submodule).
//
// Per-tile pipeline (single tile per call):
//   resize → fp16 normalize → patchify [N_patches, 3*P*P]
//   → patch_embedding (Linear 768→768 + bias)
//   → bilinear-resize position table 16x16 → spatial grid, add element-wise
//   → pad rows beyond num_valid filled with pos_resized[0]
//   → 12 × SigLIP encoder layers
//        residual + (out_proj ∘ attn ∘ ln1)
//        residual + (fc2 ∘ gelu_tanh ∘ fc1 ∘ ln2)
//   → post_layernorm
//   → strip padding (keep first num_valid rows)
//   → pixel_unshuffle_2d 2x2 (spatial_h, spatial_w, 768) → (spatial_h/2, spatial_w/2, 3072)
//   → linear_1 + bias → gelu_erf → linear_2 + bias
//
// Output is the projector's per-token fp32 features in flattened
// (spatial_h/2 * spatial_w/2, lm_hidden) order — what the caller will splice
// into the LM input embedding stream.

#include "opencl_context.h"
#include "weights.h"
#include "debug_utils.h"
#include "model_config.h"
#include "ops/lfm2_common.h"
#include "ops/lfm2_vl_image_processor.h"
#include <cstring>

// ── GPU-resident output hooks (set by Model::set_image before encoder calls).
// When set, the encoder writes fp16 features directly into the dest GPU buffer
// via clEnqueueCopyBuffer instead of reading back to host fp32 and reuploading.
cl_mem g_siglip_gpu_dest_buf = nullptr;
size_t g_siglip_gpu_dest_offset = 0;            // per-tile call
std::vector<size_t>* g_siglip_gpu_dest_offsets_batched = nullptr;  // batched call
#include "utils.h"
#include "forward_dispatch.h"

#include <CL/cl.h>
#include <vector>
#include <cstdint>
#include <cmath>
#include <string>

namespace {

static bool set_arg_local(cl_kernel k, cl_uint idx, size_t sz, const void* v, const char* name) {
    cl_int err = clSetKernelArg(k, idx, sz, v);
    if (err != CL_SUCCESS) {
        NNOPT_ERROR_FMT("siglip_encoder: clSetKernelArg(%u,%s) failed (%d)", (unsigned)idx, name, (int)err);
        return false;
    }
    return true;
}

static cl_kernel kernel(OpenCLContext& cl_ctx, const char* name) {
    cl_program p = lfm2_program(cl_ctx);
    if (!p) return nullptr;
    cl_int err = CL_SUCCESS;
    cl_kernel k = clCreateKernel(p, name, &err);
    if (err != CL_SUCCESS || !k) {
        NNOPT_ERROR_FMT("siglip_encoder: clCreateKernel(%s) failed (%d)", name, (int)err);
        return nullptr;
    }
    return k;
}

static cl_kernel image_kernel(OpenCLContext& cl_ctx, const char* file, const char* name) {
    cl_program p = cl_ctx.build_program_from_file(file);
    if (!p) {
        NNOPT_ERROR_FMT("siglip_encoder: build %s failed", file);
        return nullptr;
    }
    cl_int err = CL_SUCCESS;
    cl_kernel k = clCreateKernel(p, name, &err);
    clReleaseProgram(p);
    if (err != CL_SUCCESS || !k) {
        NNOPT_ERROR_FMT("siglip_encoder: clCreateKernel(%s) failed (%d)", name, (int)err);
        return nullptr;
    }
    return k;
}

static bool enqueue2(cl_command_queue q, cl_kernel k, size_t g0, size_t g1, const char* label) {
    const size_t gws[2] = {g0, g1};
    cl_int err = clEnqueueNDRangeKernel(q, k, 2, nullptr, gws, nullptr, 0, nullptr, nullptr);
    if (err != CL_SUCCESS) {
        NNOPT_ERROR_FMT("%s dispatch failed (%d)", label, (int)err);
        return false;
    }
    NNOPT_DEBUG_SYNC(q);
    return true;
}

// Layer-norm helper: WG=32 cooperative reduction, in/out [rows, cols], cols
// must be a multiple of 4 for the fp16 vec path (cols=768 satisfies this).
// Caller owns 'out'. Returns false on failure.
static bool ln_rows_bias(OpenCLContext& cl_ctx, cl_command_queue queue,
                         cl_mem in, cl_mem w, cl_mem b, cl_mem out,
                         int rows, int cols, float eps) {
    static cl_kernel k = nullptr;
    if (!k) k = kernel(cl_ctx, "layer_norm_rows_bias");
    if (!k) return false;
    if (!set_arg_local(k, 0, sizeof(cl_mem), &in, "input") ||
        !set_arg_local(k, 1, sizeof(cl_mem), &w,  "weight") ||
        !set_arg_local(k, 2, sizeof(cl_mem), &b,  "bias") ||
        !set_arg_local(k, 3, sizeof(cl_mem), &out, "out") ||
        !set_arg_local(k, 4, sizeof(int),     &rows, "rows") ||
        !set_arg_local(k, 5, sizeof(int),     &cols, "cols") ||
        !set_arg_local(k, 6, sizeof(float),   &eps,  "eps")) return false;
    return lfm2_kernel1_lws(queue, k, (size_t)rows * 32u, 32u, "siglip_layer_norm");
}

static bool bias_add(OpenCLContext& cl_ctx, cl_command_queue queue,
                     cl_mem x, cl_mem b, int rows, int cols) {
    static cl_kernel k = nullptr;
    if (!k) k = kernel(cl_ctx, "bias_add");
    if (!k) return false;
    if (!set_arg_local(k, 0, sizeof(cl_mem), &x, "x") ||
        !set_arg_local(k, 1, sizeof(cl_mem), &b, "b") ||
        !set_arg_local(k, 2, sizeof(int),    &rows, "rows") ||
        !set_arg_local(k, 3, sizeof(int),    &cols, "cols")) return false;
    return lfm2_kernel1(queue, k, (size_t)rows * (size_t)cols, "siglip_bias_add");
}

static bool seq_to_head_major(OpenCLContext& cl_ctx, cl_command_queue queue,
                              cl_mem src, cl_mem dst, int rows, int heads, int head_dim) {
    static cl_kernel k = nullptr;
    if (!k) k = kernel(cl_ctx, "seq_to_heads");
    if (!k) return false;
    if (!set_arg_local(k, 0, sizeof(cl_mem), &src, "src") ||
        !set_arg_local(k, 1, sizeof(cl_mem), &dst, "dst") ||
        !set_arg_local(k, 2, sizeof(int),    &rows, "rows") ||
        !set_arg_local(k, 3, sizeof(int),    &heads, "heads") ||
        !set_arg_local(k, 4, sizeof(int),    &head_dim, "head_dim")) return false;
    return lfm2_kernel1(queue, k, (size_t)rows * (size_t)heads * (size_t)head_dim, "siglip_seq_to_heads");
}

static bool head_major_to_seq(OpenCLContext& cl_ctx, cl_command_queue queue,
                              cl_mem src, cl_mem dst, int rows, int heads, int head_dim) {
    static cl_kernel k = nullptr;
    if (!k) k = kernel(cl_ctx, "heads_to_seq");
    if (!k) return false;
    if (!set_arg_local(k, 0, sizeof(cl_mem), &src, "src") ||
        !set_arg_local(k, 1, sizeof(cl_mem), &dst, "dst") ||
        !set_arg_local(k, 2, sizeof(int),    &rows, "rows") ||
        !set_arg_local(k, 3, sizeof(int),    &heads, "heads") ||
        !set_arg_local(k, 4, sizeof(int),    &head_dim, "head_dim")) return false;
    return lfm2_kernel1(queue, k, (size_t)rows * (size_t)heads * (size_t)head_dim, "siglip_heads_to_seq");
}

static bool element_add3_buf(OpenCLContext& cl_ctx, cl_command_queue queue,
                             cl_mem a, cl_mem b, cl_mem out, int n) {
    static cl_kernel k = nullptr;
    if (!k) k = kernel(cl_ctx, "element_add3");
    if (!k) return false;
    if (!set_arg_local(k, 0, sizeof(cl_mem), &a, "a") ||
        !set_arg_local(k, 1, sizeof(cl_mem), &b, "b") ||
        !set_arg_local(k, 2, sizeof(cl_mem), &out, "out") ||
        !set_arg_local(k, 3, sizeof(int),    &n, "n")) return false;
    return lfm2_kernel1(queue, k, (size_t)n, "siglip_element_add3");
}

static bool element_add_inplace_buf(OpenCLContext& cl_ctx, cl_command_queue queue,
                                    cl_mem a, cl_mem b, int n) {
    static cl_kernel k = nullptr;
    if (!k) k = kernel(cl_ctx, "element_add");
    if (!k) return false;
    if (!set_arg_local(k, 0, sizeof(cl_mem), &a, "a") ||
        !set_arg_local(k, 1, sizeof(cl_mem), &b, "b") ||
        !set_arg_local(k, 2, sizeof(int),    &n, "n")) return false;
    return lfm2_kernel1(queue, k, (size_t)n, "siglip_element_add");
}

static bool gelu_tanh(OpenCLContext& cl_ctx, cl_command_queue queue,
                      cl_mem in, cl_mem out, int n) {
    static cl_kernel k = nullptr;
    if (!k) k = kernel(cl_ctx, "gelu_tanh");
    if (!k) return false;
    if (!set_arg_local(k, 0, sizeof(cl_mem), &in,  "in") ||
        !set_arg_local(k, 1, sizeof(cl_mem), &out, "out") ||
        !set_arg_local(k, 2, sizeof(int),    &n,   "n")) return false;
    return lfm2_kernel1(queue, k, (size_t)n, "siglip_gelu_tanh");
}

static bool gelu_erf(OpenCLContext& cl_ctx, cl_command_queue queue,
                     cl_mem in, cl_mem out, int n) {
    static cl_kernel k = nullptr;
    if (!k) k = kernel(cl_ctx, "gelu_erf");
    if (!k) return false;
    if (!set_arg_local(k, 0, sizeof(cl_mem), &in,  "in") ||
        !set_arg_local(k, 1, sizeof(cl_mem), &out, "out") ||
        !set_arg_local(k, 2, sizeof(int),    &n,   "n")) return false;
    return lfm2_kernel1(queue, k, (size_t)n, "projector_gelu_erf");
}

static bool pixel_unshuffle_2d(OpenCLContext& cl_ctx, cl_command_queue queue,
                               cl_mem in, cl_mem out, int H, int W, int C) {
    static cl_kernel k = nullptr;
    if (!k) k = kernel(cl_ctx, "pixel_unshuffle_2d");
    if (!k) return false;
    if (!set_arg_local(k, 0, sizeof(cl_mem), &in,  "in") ||
        !set_arg_local(k, 1, sizeof(cl_mem), &out, "out") ||
        !set_arg_local(k, 2, sizeof(int),    &H,   "H") ||
        !set_arg_local(k, 3, sizeof(int),    &W,   "W") ||
        !set_arg_local(k, 4, sizeof(int),    &C,   "C")) return false;
    size_t total = (size_t)(H >> 1) * (size_t)(W >> 1) * (size_t)(C * 4);
    return lfm2_kernel1(queue, k, total, "siglip_pixel_unshuffle_2d");
}

static bool bilinear_position_resize(OpenCLContext& cl_ctx, cl_command_queue queue,
                                     cl_mem in, cl_mem out, int H_out, int W_out, int C) {
    static cl_kernel k = nullptr;
    if (!k) k = kernel(cl_ctx, "bilinear_position_resize");
    if (!k) return false;
    if (!set_arg_local(k, 0, sizeof(cl_mem), &in,    "in") ||
        !set_arg_local(k, 1, sizeof(cl_mem), &out,   "out") ||
        !set_arg_local(k, 2, sizeof(int),    &H_out, "H_out") ||
        !set_arg_local(k, 3, sizeof(int),    &W_out, "W_out") ||
        !set_arg_local(k, 4, sizeof(int),    &C,     "C")) return false;
    return lfm2_kernel1(queue, k, (size_t)H_out * (size_t)W_out * (size_t)C, "siglip_position_resize");
}

static bool siglip_fill_padding(OpenCLContext& cl_ctx, cl_command_queue queue,
                                cl_mem x, cl_mem pos_resized,
                                int num_valid, int N_seq, int hidden) {
    if (num_valid >= N_seq) return true;
    static cl_kernel k = nullptr;
    if (!k) k = kernel(cl_ctx, "siglip_fill_padding");
    if (!k) return false;
    int npad = (N_seq - num_valid) * hidden;
    if (!set_arg_local(k, 0, sizeof(cl_mem), &x, "x") ||
        !set_arg_local(k, 1, sizeof(cl_mem), &pos_resized, "pos_resized") ||
        !set_arg_local(k, 2, sizeof(int),    &num_valid, "num_valid") ||
        !set_arg_local(k, 3, sizeof(int),    &N_seq,     "N_seq") ||
        !set_arg_local(k, 4, sizeof(int),    &hidden,    "hidden")) return false;
    return lfm2_kernel1(queue, k, (size_t)npad, "siglip_fill_padding");
}

// Lazy loader for the separate cl_program containing the image2d-backed SigLIP
// attention kernel. Kept in its own program to avoid Adreno register-allocation
// interference with the buffer-path kernels in lfm2_ops.cl (ARTICLE.md).
static cl_kernel get_siglip_attn_img_kernel(OpenCLContext& cl_ctx) {
    static cl_program prog = nullptr;
    static cl_kernel  ker  = nullptr;
    if (ker) return ker;
    if (prog) return nullptr;
    prog = cl_ctx.build_program_from_file("kernels/siglip_attn_image.cl");
    if (!prog) { NNOPT_ERROR_FMT("siglip_attn_image: build failed%s", ""); return nullptr; }
    cl_int err = CL_SUCCESS;
    ker = clCreateKernel(prog, "siglip_flash_attn_prefill_img", &err);
    if (err != CL_SUCCESS || !ker) {
        NNOPT_ERROR_FMT("siglip_attn_image: clCreateKernel failed (%d)", (int)err);
        ker = nullptr;
        return nullptr;
    }
    return ker;
}

// Wrap a fp16 K/V buffer of shape [kv_heads, k_rows, head_dim=64] as an
// image2d_t view via cl_khr_image2d_from_buffer. Zero-copy: shares backing
// storage with the buffer. Caller owns the returned image and must release it.
// Image layout: CL_RGBA / CL_HALF_FLOAT, width = head_dim/4 = 16 pixels,
// height = kv_heads * k_rows (max 16384 on Adreno 620 — comfortably covers
// k_rows=1024 * 12 heads = 12288).
static cl_mem wrap_kv_as_image(OpenCLContext& cl_ctx, cl_mem buf,
                               int kv_heads, int k_rows, int head_dim) {
    if (!buf) return nullptr;
    cl_image_format fmt;
    fmt.image_channel_order = CL_RGBA;
    fmt.image_channel_data_type = CL_HALF_FLOAT;
    cl_image_desc desc; std::memset(&desc, 0, sizeof(desc));
    desc.image_type   = CL_MEM_OBJECT_IMAGE2D;
    desc.image_width  = (size_t)(head_dim / 4);
    desc.image_height = (size_t)kv_heads * (size_t)k_rows;
    desc.buffer       = buf;
    cl_int err = CL_SUCCESS;
    cl_mem img = clCreateImage(cl_ctx.context(), CL_MEM_READ_ONLY, &fmt, &desc, nullptr, &err);
    if (err != CL_SUCCESS || !img) {
        NNOPT_ERROR_FMT("wrap_kv_as_image: clCreateImage failed (%d)", (int)err);
        return nullptr;
    }
    return img;
}

static bool flash_attn(OpenCLContext& cl_ctx, cl_command_queue queue,
                       cl_mem Q, cl_mem K, cl_mem V, cl_mem pad_mask, cl_mem out,
                       int q_rows, int k_rows, int q_heads, int kv_heads,
                       int head_dim, float scale) {
    // Try the image2d fast path first.
    cl_kernel k_img = get_siglip_attn_img_kernel(cl_ctx);
    if (k_img && (head_dim % 4) == 0 && ((size_t)kv_heads * (size_t)k_rows) <= 16384u) {
        cl_mem K_img = wrap_kv_as_image(cl_ctx, K, kv_heads, k_rows, head_dim);
        cl_mem V_img = wrap_kv_as_image(cl_ctx, V, kv_heads, k_rows, head_dim);
        if (K_img && V_img) {
            const int fa_bq = 4;
            const int num_q_tiles = (q_rows + fa_bq - 1) / fa_bq;
            size_t global = (size_t)q_heads * (size_t)num_q_tiles * 64u;
            size_t local = 64u;
            bool ok = set_arg_local(k_img, 0, sizeof(cl_mem), &Q, "Q") &&
                      set_arg_local(k_img, 1, sizeof(cl_mem), &K_img, "K_img") &&
                      set_arg_local(k_img, 2, sizeof(cl_mem), &V_img, "V_img") &&
                      set_arg_local(k_img, 3, sizeof(cl_mem), &pad_mask, "pad_mask") &&
                      set_arg_local(k_img, 4, sizeof(cl_mem), &out, "out") &&
                      set_arg_local(k_img, 5, sizeof(int),    &q_rows, "q_rows") &&
                      set_arg_local(k_img, 6, sizeof(int),    &k_rows, "k_rows") &&
                      set_arg_local(k_img, 7, sizeof(int),    &q_heads, "q_heads") &&
                      set_arg_local(k_img, 8, sizeof(int),    &kv_heads, "kv_heads") &&
                      set_arg_local(k_img, 9, sizeof(int),    &head_dim, "head_dim") &&
                      set_arg_local(k_img, 10, sizeof(float), &scale, "scale") &&
                      lfm2_kernel1_lws(queue, k_img, global, local, "siglip_flash_attn_prefill_img");
            // Images are zero-copy views; releasing here is safe because the
            // kernel's clEnqueueNDRange has already taken its retain count.
            clReleaseMemObject(K_img);
            clReleaseMemObject(V_img);
            if (ok) return true;
        } else {
            if (K_img) clReleaseMemObject(K_img);
            if (V_img) clReleaseMemObject(V_img);
        }
        // Fall through to buffer-path on failure
    }

    // Buffer-path fallback.
    static cl_kernel k = nullptr;
    if (!k) k = kernel(cl_ctx, "siglip_flash_attn_prefill");
    if (!k) return false;
    const int fa_bq = 4;
    const int num_q_tiles = (q_rows + fa_bq - 1) / fa_bq;
    size_t global = (size_t)q_heads * (size_t)num_q_tiles * 64u;
    size_t local = 64u;
    if (!set_arg_local(k, 0, sizeof(cl_mem), &Q, "Q") ||
        !set_arg_local(k, 1, sizeof(cl_mem), &K, "K") ||
        !set_arg_local(k, 2, sizeof(cl_mem), &V, "V") ||
        !set_arg_local(k, 3, sizeof(cl_mem), &pad_mask, "pad_mask") ||
        !set_arg_local(k, 4, sizeof(cl_mem), &out, "out") ||
        !set_arg_local(k, 5, sizeof(int),    &q_rows, "q_rows") ||
        !set_arg_local(k, 6, sizeof(int),    &k_rows, "k_rows") ||
        !set_arg_local(k, 7, sizeof(int),    &q_heads, "q_heads") ||
        !set_arg_local(k, 8, sizeof(int),    &kv_heads, "kv_heads") ||
        !set_arg_local(k, 9, sizeof(int),    &head_dim, "head_dim") ||
        !set_arg_local(k, 10, sizeof(float), &scale, "scale")) return false;
    return lfm2_kernel1_lws(queue, k, global, local, "siglip_flash_attn_prefill");
}

}  // namespace

// ──────────────────────────────────────────────────────────────────────
// Public entry point.
// ──────────────────────────────────────────────────────────────────────
bool siglip_vision_forward_tile(
    OpenCLContext& cl_ctx,
    Weights& weights,
    const std::vector<uint8_t>& tile_rgb_u8,
    int H_px, int W_px,
    int spatial_h, int spatial_w,
    std::vector<float>& image_features_out) {

    NNOPT_CHECKPOINT("siglip_vision_forward_tile");
    image_features_out.clear();
    cl_command_queue queue = cl_ctx.queue();
    if (!queue || tile_rgb_u8.empty() || H_px <= 0 || W_px <= 0 || spatial_h <= 0 || spatial_w <= 0) {
        NNOPT_ERROR("siglip_vision_forward_tile: invalid input");
        return false;
    }
    if (spatial_h * 16 != H_px || spatial_w * 16 != W_px) {
        NNOPT_ERROR_FMT("siglip_vision_forward_tile: spatial (%dx%d) ≠ pixel (%dx%d)/16",
                        spatial_h, spatial_w, H_px, W_px);
        return false;
    }
    if ((spatial_h & 1) || (spatial_w & 1)) {
        NNOPT_ERROR_FMT("siglip_vision_forward_tile: spatial dims must be even (got %dx%d)",
                        spatial_h, spatial_w);
        return false;
    }

    const int patch = 16;
    const int patch_flat = 3 * patch * patch;                // 768
    const int hidden = MODEL_CONFIG::VISION_CONFIG_HIDDEN_SIZE;          // 768
    const int intermediate = MODEL_CONFIG::VISION_CONFIG_INTERMEDIATE_SIZE; // 3072
    const int num_heads = MODEL_CONFIG::VISION_CONFIG_NUM_ATTENTION_HEADS;  // 12
    const int head_dim = hidden / num_heads;                 // 64
    const int num_layers = MODEL_CONFIG::VISION_CONFIG_NUM_HIDDEN_LAYERS;   // 12
    const float ln_eps = MODEL_CONFIG::LAYER_NORM_EPS;       // 1e-6
    const int projector_in = hidden * MODEL_CONFIG::DOWNSAMPLE_FACTOR * MODEL_CONFIG::DOWNSAMPLE_FACTOR; // 3072
    const int projector_hidden = MODEL_CONFIG::PROJECTOR_HIDDEN_SIZE; // 2048
    const int lm_hidden = MODEL_CONFIG::HIDDEN_SIZE;         // 1024 (LM hidden)

    const int num_valid = spatial_h * spatial_w;
    // Reference pads non-thumbnail tiles to 1024 (32x32). The thumbnail tile
    // (e.g. 26x38 = 988) stays at its native num_valid count — no padding
    // beyond. We treat N_seq == num_valid when the tile already isn't a
    // padded "full" tile, otherwise pad to 1024.
    // The simple rule from the reference: full tiles use N_seq=1024
    // (32x32 = 1024 patches max), thumbnail tiles use their natural count.
    int N_seq;
    if (spatial_h == 32 && spatial_w == 32) {
        N_seq = 1024;
    } else {
        // Thumbnail / non-square: no padding, mask is all-valid.
        N_seq = num_valid;
    }

    cl_int err = CL_SUCCESS;

    // Required weights.
    cl_mem patch_w = weights.get_buffer("model.vision_tower.vision_model.embeddings.patch_embedding.weight");
    cl_mem patch_b = weights.get_buffer("model.vision_tower.vision_model.embeddings.patch_embedding.bias");
    cl_mem pos_w   = weights.get_buffer("model.vision_tower.vision_model.embeddings.position_embedding.weight");
    cl_mem post_ln_w = weights.get_buffer("model.vision_tower.vision_model.post_layernorm.weight");
    cl_mem post_ln_b = weights.get_buffer("model.vision_tower.vision_model.post_layernorm.bias");
    cl_mem p1_w = weights.get_buffer("model.multi_modal_projector.linear_1.weight");
    cl_mem p1_b = weights.get_buffer("model.multi_modal_projector.linear_1.bias");
    cl_mem p2_w = weights.get_buffer("model.multi_modal_projector.linear_2.weight");
    cl_mem p2_b = weights.get_buffer("model.multi_modal_projector.linear_2.bias");
    if (!patch_w || !patch_b || !pos_w || !post_ln_w || !post_ln_b ||
        !p1_w || !p1_b || !p2_w || !p2_b) {
        NNOPT_ERROR("siglip_vision_forward_tile: missing required vision/projector weight");
        return false;
    }

    // ── 1. Upload + (optional) resize + normalize ──
    cl_mem src = clCreateBuffer(cl_ctx.context(), CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
                                tile_rgb_u8.size(), const_cast<uint8_t*>(tile_rgb_u8.data()), &err);
    if (err != CL_SUCCESS || !src) {
        NNOPT_ERROR_FMT("siglip_encoder: src upload failed (%d)", (int)err);
        return false;
    }
    cl_mem norm = lfm2_alloc(cl_ctx, (size_t)H_px * W_px * 3u, "siglip_norm");
    cl_mem patches = lfm2_alloc(cl_ctx, (size_t)num_valid * patch_flat, "siglip_patches");
    cl_mem patch_emb = lfm2_alloc(cl_ctx, (size_t)num_valid * hidden, "siglip_patch_emb");
    cl_mem pos_resized = lfm2_alloc(cl_ctx, (size_t)spatial_h * spatial_w * hidden, "siglip_pos_resized");
    cl_mem x = lfm2_alloc(cl_ctx, (size_t)N_seq * hidden, "siglip_x");
    if (!norm || !patches || !patch_emb || !pos_resized || !x) {
        if (src) clReleaseMemObject(src);
        if (norm) clReleaseMemObject(norm);
        if (patches) clReleaseMemObject(patches);
        if (patch_emb) clReleaseMemObject(patch_emb);
        if (pos_resized) clReleaseMemObject(pos_resized);
        if (x) clReleaseMemObject(x);
        return false;
    }

    cl_kernel norm_k = image_kernel(cl_ctx, "kernels/image_normalize.cl", "image_normalize");
    cl_kernel patchify_k = image_kernel(cl_ctx, "kernels/image_patchify.cl", "image_patchify");
    auto cleanup_kernels = [&]() {
        if (norm_k) clReleaseKernel(norm_k);
        if (patchify_k) clReleaseKernel(patchify_k);
    };
    if (!norm_k || !patchify_k) {
        cleanup_kernels();
        clReleaseMemObject(src); clReleaseMemObject(norm); clReleaseMemObject(patches);
        clReleaseMemObject(patch_emb); clReleaseMemObject(pos_resized); clReleaseMemObject(x);
        return false;
    }

    // image_normalize: writes CHW fp16 normalized image.
    float mean = 0.5f, stdev = 0.5f;
    if (!set_arg_local(norm_k, 0, sizeof(cl_mem), &src, "src") ||
        !set_arg_local(norm_k, 1, sizeof(cl_mem), &norm, "dst") ||
        !set_arg_local(norm_k, 2, sizeof(int),    &H_px, "H") ||
        !set_arg_local(norm_k, 3, sizeof(int),    &W_px, "W") ||
        !set_arg_local(norm_k, 4, sizeof(float),  &mean, "mr") ||
        !set_arg_local(norm_k, 5, sizeof(float),  &mean, "mg") ||
        !set_arg_local(norm_k, 6, sizeof(float),  &mean, "mb") ||
        !set_arg_local(norm_k, 7, sizeof(float),  &stdev, "sr") ||
        !set_arg_local(norm_k, 8, sizeof(float),  &stdev, "sg") ||
        !set_arg_local(norm_k, 9, sizeof(float),  &stdev, "sb") ||
        !enqueue2(queue, norm_k, (size_t)W_px, (size_t)H_px, "siglip_image_normalize")) {
        cleanup_kernels();
        clReleaseMemObject(src); clReleaseMemObject(norm); clReleaseMemObject(patches);
        clReleaseMemObject(patch_emb); clReleaseMemObject(pos_resized); clReleaseMemObject(x);
        return false;
    }

    if (!set_arg_local(patchify_k, 0, sizeof(cl_mem), &norm, "image") ||
        !set_arg_local(patchify_k, 1, sizeof(cl_mem), &patches, "patches") ||
        !set_arg_local(patchify_k, 2, sizeof(int),    &H_px, "H") ||
        !set_arg_local(patchify_k, 3, sizeof(int),    &W_px, "W") ||
        !set_arg_local(patchify_k, 4, sizeof(int),    &patch, "patch") ||
        !enqueue2(queue, patchify_k, (size_t)num_valid, (size_t)patch_flat, "siglip_image_patchify")) {
        cleanup_kernels();
        clReleaseMemObject(src); clReleaseMemObject(norm); clReleaseMemObject(patches);
        clReleaseMemObject(patch_emb); clReleaseMemObject(pos_resized); clReleaseMemObject(x);
        return false;
    }
    cleanup_kernels();
    clReleaseMemObject(src);
    clReleaseMemObject(norm);

    // ── 2. patch_embedding: Linear 768→768 + bias ──
    if (!pytorch_linear(queue, num_valid, hidden, patch_flat, patches, patch_w, patch_emb)) {
        clReleaseMemObject(patches); clReleaseMemObject(patch_emb);
        clReleaseMemObject(pos_resized); clReleaseMemObject(x);
        return false;
    }
    clReleaseMemObject(patches);
    if (!bias_add(cl_ctx, queue, patch_emb, patch_b, num_valid, hidden)) {
        clReleaseMemObject(patch_emb); clReleaseMemObject(pos_resized); clReleaseMemObject(x);
        return false;
    }

    // ── 3. position_embedding: bilinear-resize [16,16,768] → [spatial_h, spatial_w, 768] ──
    if (!bilinear_position_resize(cl_ctx, queue, pos_w, pos_resized, spatial_h, spatial_w, hidden)) {
        clReleaseMemObject(patch_emb); clReleaseMemObject(pos_resized); clReleaseMemObject(x);
        return false;
    }

    // ── 4. Combine: x[0..num_valid) = patch_emb + pos_resized, then pad. ──
    if (!element_add3_buf(cl_ctx, queue, patch_emb, pos_resized, x, num_valid * hidden)) {
        clReleaseMemObject(patch_emb); clReleaseMemObject(pos_resized); clReleaseMemObject(x);
        return false;
    }
    // Fill any padded rows (num_valid..N_seq) with pos_resized[0,0,:].
    if (N_seq > num_valid) {
        if (!siglip_fill_padding(cl_ctx, queue, x, pos_resized, num_valid, N_seq, hidden)) {
            clReleaseMemObject(patch_emb); clReleaseMemObject(pos_resized); clReleaseMemObject(x);
            return false;
        }
    }
    clReleaseMemObject(patch_emb);
    clReleaseMemObject(pos_resized);

    // ── 5. Pad-mask buffer (int32[N_seq]) ──
    std::vector<int32_t> mask_host(N_seq, 1);
    for (int i = num_valid; i < N_seq; ++i) mask_host[i] = 0;
    cl_mem pad_mask = clCreateBuffer(cl_ctx.context(),
                                     CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
                                     (size_t)N_seq * sizeof(int32_t),
                                     mask_host.data(), &err);
    if (err != CL_SUCCESS || !pad_mask) {
        NNOPT_ERROR_FMT("siglip_encoder: pad_mask upload failed (%d)", (int)err);
        clReleaseMemObject(x);
        return false;
    }

    // ── 6. SigLIP encoder layers ──
    // Reused per-layer scratch buffers (allocated once outside the loop).
    cl_mem ln_out  = lfm2_alloc(cl_ctx, (size_t)N_seq * hidden, "siglip_ln_out");
    cl_mem q_seq   = lfm2_alloc(cl_ctx, (size_t)N_seq * hidden, "siglip_q_seq");
    cl_mem k_seq   = lfm2_alloc(cl_ctx, (size_t)N_seq * hidden, "siglip_k_seq");
    cl_mem v_seq   = lfm2_alloc(cl_ctx, (size_t)N_seq * hidden, "siglip_v_seq");
    cl_mem q_hm    = lfm2_alloc(cl_ctx, (size_t)N_seq * hidden, "siglip_q_hm");
    cl_mem k_hm    = lfm2_alloc(cl_ctx, (size_t)N_seq * hidden, "siglip_k_hm");
    cl_mem v_hm    = lfm2_alloc(cl_ctx, (size_t)N_seq * hidden, "siglip_v_hm");
    cl_mem attn_hm = lfm2_alloc(cl_ctx, (size_t)N_seq * hidden, "siglip_attn_hm");
    cl_mem attn_seq = lfm2_alloc(cl_ctx, (size_t)N_seq * hidden, "siglip_attn_seq");
    cl_mem attn_proj = lfm2_alloc(cl_ctx, (size_t)N_seq * hidden, "siglip_attn_proj");
    cl_mem fc1_out = lfm2_alloc(cl_ctx, (size_t)N_seq * intermediate, "siglip_fc1_out");
    cl_mem mlp_act = lfm2_alloc(cl_ctx, (size_t)N_seq * intermediate, "siglip_mlp_act");
    cl_mem fc2_out = lfm2_alloc(cl_ctx, (size_t)N_seq * hidden, "siglip_fc2_out");
    if (!ln_out || !q_seq || !k_seq || !v_seq || !q_hm || !k_hm || !v_hm ||
        !attn_hm || !attn_seq || !attn_proj || !fc1_out || !mlp_act || !fc2_out) {
        if (ln_out) clReleaseMemObject(ln_out);
        if (q_seq) clReleaseMemObject(q_seq);
        if (k_seq) clReleaseMemObject(k_seq);
        if (v_seq) clReleaseMemObject(v_seq);
        if (q_hm) clReleaseMemObject(q_hm);
        if (k_hm) clReleaseMemObject(k_hm);
        if (v_hm) clReleaseMemObject(v_hm);
        if (attn_hm) clReleaseMemObject(attn_hm);
        if (attn_seq) clReleaseMemObject(attn_seq);
        if (attn_proj) clReleaseMemObject(attn_proj);
        if (fc1_out) clReleaseMemObject(fc1_out);
        if (mlp_act) clReleaseMemObject(mlp_act);
        if (fc2_out) clReleaseMemObject(fc2_out);
        clReleaseMemObject(pad_mask); clReleaseMemObject(x);
        return false;
    }

    const float scale = 1.0f / std::sqrt((float)head_dim);

    auto cleanup_all = [&]() {
        clReleaseMemObject(ln_out); clReleaseMemObject(q_seq); clReleaseMemObject(k_seq);
        clReleaseMemObject(v_seq); clReleaseMemObject(q_hm); clReleaseMemObject(k_hm);
        clReleaseMemObject(v_hm); clReleaseMemObject(attn_hm); clReleaseMemObject(attn_seq);
        clReleaseMemObject(attn_proj); clReleaseMemObject(fc1_out); clReleaseMemObject(mlp_act);
        clReleaseMemObject(fc2_out); clReleaseMemObject(pad_mask);
    };

    for (int li = 0; li < num_layers; ++li) {
        cl_ctx.yield_for_compositor();   // yield GPU between vision layers so the UI doesn't ANR
        std::string p = "model.vision_tower.vision_model.encoder.layers." + std::to_string(li);
        cl_mem ln1_w = weights.get_buffer(p + ".layer_norm1.weight");
        cl_mem ln1_b = weights.get_buffer(p + ".layer_norm1.bias");
        cl_mem qw = weights.get_buffer(p + ".self_attn.q_proj.weight");
        cl_mem qb = weights.get_buffer(p + ".self_attn.q_proj.bias");
        cl_mem kw = weights.get_buffer(p + ".self_attn.k_proj.weight");
        cl_mem kb = weights.get_buffer(p + ".self_attn.k_proj.bias");
        cl_mem vw = weights.get_buffer(p + ".self_attn.v_proj.weight");
        cl_mem vb = weights.get_buffer(p + ".self_attn.v_proj.bias");
        cl_mem ow = weights.get_buffer(p + ".self_attn.out_proj.weight");
        cl_mem ob = weights.get_buffer(p + ".self_attn.out_proj.bias");
        cl_mem ln2_w = weights.get_buffer(p + ".layer_norm2.weight");
        cl_mem ln2_b = weights.get_buffer(p + ".layer_norm2.bias");
        cl_mem fc1_w = weights.get_buffer(p + ".mlp.fc1.weight");
        cl_mem fc1_b = weights.get_buffer(p + ".mlp.fc1.bias");
        cl_mem fc2_w = weights.get_buffer(p + ".mlp.fc2.weight");
        cl_mem fc2_b = weights.get_buffer(p + ".mlp.fc2.bias");
        if (!ln1_w || !ln1_b || !qw || !qb || !kw || !kb || !vw || !vb ||
            !ow || !ob || !ln2_w || !ln2_b || !fc1_w || !fc1_b || !fc2_w || !fc2_b) {
            NNOPT_ERROR_FMT("siglip_encoder: missing layer-%d weight", li);
            cleanup_all(); clReleaseMemObject(x);
            return false;
        }

        // ── self-attention block: pre-norm + residual ──
        if (!ln_rows_bias(cl_ctx, queue, x, ln1_w, ln1_b, ln_out, N_seq, hidden, ln_eps)) {
            cleanup_all(); clReleaseMemObject(x);
            return false;
        }
        if (!pytorch_linear(queue, N_seq, hidden, hidden, ln_out, qw, q_seq) ||
            !pytorch_linear(queue, N_seq, hidden, hidden, ln_out, kw, k_seq) ||
            !pytorch_linear(queue, N_seq, hidden, hidden, ln_out, vw, v_seq)) {
            cleanup_all(); clReleaseMemObject(x);
            return false;
        }
        if (!bias_add(cl_ctx, queue, q_seq, qb, N_seq, hidden) ||
            !bias_add(cl_ctx, queue, k_seq, kb, N_seq, hidden) ||
            !bias_add(cl_ctx, queue, v_seq, vb, N_seq, hidden)) {
            cleanup_all(); clReleaseMemObject(x);
            return false;
        }
        if (!seq_to_head_major(cl_ctx, queue, q_seq, q_hm, N_seq, num_heads, head_dim) ||
            !seq_to_head_major(cl_ctx, queue, k_seq, k_hm, N_seq, num_heads, head_dim) ||
            !seq_to_head_major(cl_ctx, queue, v_seq, v_hm, N_seq, num_heads, head_dim)) {
            cleanup_all(); clReleaseMemObject(x);
            return false;
        }
        if (!flash_attn(cl_ctx, queue, q_hm, k_hm, v_hm, pad_mask, attn_hm,
                        N_seq, N_seq, num_heads, num_heads, head_dim, scale)) {
            cleanup_all(); clReleaseMemObject(x);
            return false;
        }
        if (!head_major_to_seq(cl_ctx, queue, attn_hm, attn_seq, N_seq, num_heads, head_dim)) {
            cleanup_all(); clReleaseMemObject(x);
            return false;
        }
        if (!pytorch_linear(queue, N_seq, hidden, hidden, attn_seq, ow, attn_proj)) {
            cleanup_all(); clReleaseMemObject(x);
            return false;
        }
        if (!bias_add(cl_ctx, queue, attn_proj, ob, N_seq, hidden)) {
            cleanup_all(); clReleaseMemObject(x);
            return false;
        }
        // x = x + attn_proj
        if (!element_add_inplace_buf(cl_ctx, queue, x, attn_proj, N_seq * hidden)) {
            cleanup_all(); clReleaseMemObject(x);
            return false;
        }

        // ── MLP block: pre-norm + residual ──
        if (!ln_rows_bias(cl_ctx, queue, x, ln2_w, ln2_b, ln_out, N_seq, hidden, ln_eps)) {
            cleanup_all(); clReleaseMemObject(x);
            return false;
        }
        if (!pytorch_linear(queue, N_seq, intermediate, hidden, ln_out, fc1_w, fc1_out)) {
            cleanup_all(); clReleaseMemObject(x);
            return false;
        }
        if (!bias_add(cl_ctx, queue, fc1_out, fc1_b, N_seq, intermediate)) {
            cleanup_all(); clReleaseMemObject(x);
            return false;
        }
        if (!gelu_tanh(cl_ctx, queue, fc1_out, mlp_act, N_seq * intermediate)) {
            cleanup_all(); clReleaseMemObject(x);
            return false;
        }
        if (!pytorch_linear(queue, N_seq, hidden, intermediate, mlp_act, fc2_w, fc2_out)) {
            cleanup_all(); clReleaseMemObject(x);
            return false;
        }
        if (!bias_add(cl_ctx, queue, fc2_out, fc2_b, N_seq, hidden)) {
            cleanup_all(); clReleaseMemObject(x);
            return false;
        }
        if (!element_add_inplace_buf(cl_ctx, queue, x, fc2_out, N_seq * hidden)) {
            cleanup_all(); clReleaseMemObject(x);
            return false;
        }
    }

    // ── 7. Final post-LayerNorm ──
    if (!ln_rows_bias(cl_ctx, queue, x, post_ln_w, post_ln_b, ln_out, N_seq, hidden, ln_eps)) {
        cleanup_all(); clReleaseMemObject(x);
        return false;
    }

    // ── 8. Strip padding (keep first num_valid rows) → ln_out already has
    //       the full N_seq×hidden post-layernorm output; we copy only the
    //       first num_valid rows into a tightly-sized buffer. If num_valid
    //       == N_seq, just reuse ln_out via clRetain.
    cl_mem encoder_out = nullptr;
    if (num_valid == N_seq) {
        encoder_out = ln_out;
        clRetainMemObject(encoder_out);
    } else {
        encoder_out = lfm2_alloc(cl_ctx, (size_t)num_valid * hidden, "siglip_encoder_out_unpadded");
        if (!encoder_out) { cleanup_all(); clReleaseMemObject(x); return false; }
        cl_int re = clEnqueueCopyBuffer(queue, ln_out, encoder_out,
                                        0, 0,
                                        (size_t)num_valid * hidden * sizeof(nnopt_storage_t),
                                        0, nullptr, nullptr);
        if (re != CL_SUCCESS) {
            NNOPT_ERROR_FMT("siglip_encoder: strip-padding copy failed (%d)", (int)re);
            clReleaseMemObject(encoder_out); cleanup_all(); clReleaseMemObject(x);
            return false;
        }
    }

    // ── 9. Pixel-unshuffle (spatial_h, spatial_w, 768) → (spatial_h/2, spatial_w/2, 3072) ──
    const int H2 = spatial_h >> 1;
    const int W2 = spatial_w >> 1;
    const int num_out = H2 * W2;
    cl_mem proj_in = lfm2_alloc(cl_ctx, (size_t)num_out * projector_in, "projector_in");
    if (!proj_in) { clReleaseMemObject(encoder_out); cleanup_all(); clReleaseMemObject(x); return false; }
    if (!pixel_unshuffle_2d(cl_ctx, queue, encoder_out, proj_in, spatial_h, spatial_w, hidden)) {
        clReleaseMemObject(proj_in); clReleaseMemObject(encoder_out); cleanup_all(); clReleaseMemObject(x);
        return false;
    }
    clReleaseMemObject(encoder_out);

    // ── 10. Projector linear_1 + bias ──
    cl_mem proj_hidden = lfm2_alloc(cl_ctx, (size_t)num_out * projector_hidden, "projector_hidden");
    if (!proj_hidden) { clReleaseMemObject(proj_in); cleanup_all(); clReleaseMemObject(x); return false; }
    if (!pytorch_linear(queue, num_out, projector_hidden, projector_in, proj_in, p1_w, proj_hidden)) {
        clReleaseMemObject(proj_hidden); clReleaseMemObject(proj_in); cleanup_all(); clReleaseMemObject(x);
        return false;
    }
    clReleaseMemObject(proj_in);
    if (!bias_add(cl_ctx, queue, proj_hidden, p1_b, num_out, projector_hidden)) {
        clReleaseMemObject(proj_hidden); cleanup_all(); clReleaseMemObject(x); return false;
    }

    // ── 11. Projector GELU (erf-based) ──
    cl_mem proj_gelu = lfm2_alloc(cl_ctx, (size_t)num_out * projector_hidden, "projector_gelu");
    if (!proj_gelu) { clReleaseMemObject(proj_hidden); cleanup_all(); clReleaseMemObject(x); return false; }
    if (!gelu_erf(cl_ctx, queue, proj_hidden, proj_gelu, num_out * projector_hidden)) {
        clReleaseMemObject(proj_gelu); clReleaseMemObject(proj_hidden); cleanup_all(); clReleaseMemObject(x);
        return false;
    }
    clReleaseMemObject(proj_hidden);

    // ── 12. Projector linear_2 + bias ──
    cl_mem out = lfm2_alloc(cl_ctx, (size_t)num_out * lm_hidden, "projector_out");
    if (!out) { clReleaseMemObject(proj_gelu); cleanup_all(); clReleaseMemObject(x); return false; }
    if (!pytorch_linear(queue, num_out, lm_hidden, projector_hidden, proj_gelu, p2_w, out)) {
        clReleaseMemObject(out); clReleaseMemObject(proj_gelu); cleanup_all(); clReleaseMemObject(x);
        return false;
    }
    clReleaseMemObject(proj_gelu);
    if (!bias_add(cl_ctx, queue, out, p2_b, num_out, lm_hidden)) {
        clReleaseMemObject(out); cleanup_all(); clReleaseMemObject(x); return false;
    }

    // ── 13. Output: GPU-direct copy if a destination hook is registered,
    //         otherwise fall back to host fp32 readback. The GPU path saves
    //         the read + fp16→fp32 + fp32→fp16 + reupload triple round-trip.
    extern cl_mem g_siglip_gpu_dest_buf;
    extern size_t g_siglip_gpu_dest_offset;
    const size_t out_bytes = (size_t)num_out * lm_hidden * sizeof(nnopt_storage_t);
    if (g_siglip_gpu_dest_buf) {
        cl_int ce = clEnqueueCopyBuffer(queue, out, g_siglip_gpu_dest_buf,
                                        0, g_siglip_gpu_dest_offset,
                                        out_bytes, 0, nullptr, nullptr);
        if (ce != CL_SUCCESS) {
            NNOPT_ERROR_FMT("siglip_encoder: gpu copy failed (%d)", (int)ce);
            clReleaseMemObject(out); cleanup_all(); clReleaseMemObject(x);
            return false;
        }
        image_features_out.clear();  // signal no host data
    } else {
        std::vector<nnopt_storage_t> tmp((size_t)num_out * lm_hidden);
        err = clEnqueueReadBuffer(queue, out, CL_TRUE, 0,
                                  tmp.size() * sizeof(nnopt_storage_t), tmp.data(),
                                  0, nullptr, nullptr);
        if (err != CL_SUCCESS) {
            NNOPT_ERROR_FMT("siglip_encoder: read projector_out failed (%d)", (int)err);
            clReleaseMemObject(out); cleanup_all(); clReleaseMemObject(x);
            return false;
        }
        image_features_out.resize(tmp.size());
        for (size_t i = 0; i < tmp.size(); ++i) {
#ifdef NNOPT_USE_FP16
            image_features_out[i] = nnopt_f16_to_f32((uint16_t)tmp[i]);
#else
            image_features_out[i] = tmp[i];
#endif
        }
    }

    clReleaseMemObject(out);
    cleanup_all();
    clReleaseMemObject(x);
    NNOPT_CHECKPOINT("siglip_vision_forward_tile done");
    return true;
}

// ──────────────────────────────────────────────────────────────────────
// Batched entry point: N tiles with IDENTICAL spatial shape, processed
// through the encoder as one big M = N_tiles * patches_per_tile batch so
// the per-token Linear/LN/MLP ops collapse from N_tiles CLBlast calls
// into ONE. Attention stays per-tile (each tile is its own attention
// context). Projector stays per-tile (output is per-tile).
//
// Padding: only spatial 32x32 = 1024 patches per tile, no padding needed
// (each tile already at native count). Mask is all-1s. We require all
// tiles to share spatial_h and spatial_w — caller groups tiles ahead of
// time.
// ──────────────────────────────────────────────────────────────────────
bool siglip_vision_forward_batched(
    OpenCLContext& cl_ctx,
    Weights& weights,
    const std::vector<const Lfm2VlTile*>& tiles_ptrs,
    std::vector<std::vector<float>>& per_tile_features_out) {

    NNOPT_CHECKPOINT("siglip_vision_forward_batched");
    per_tile_features_out.clear();

    cl_command_queue queue = cl_ctx.queue();
    if (!queue || tiles_ptrs.empty()) {
        NNOPT_ERROR("siglip_vision_forward_batched: empty tiles or no queue");
        return false;
    }
    const int N_tiles = (int)tiles_ptrs.size();
    const int spatial_h = tiles_ptrs[0]->spatial_h;
    const int spatial_w = tiles_ptrs[0]->spatial_w;
    const int H_px = tiles_ptrs[0]->H_px;
    const int W_px = tiles_ptrs[0]->W_px;
    for (int t = 0; t < N_tiles; ++t) {
        if (!tiles_ptrs[t] ||
            tiles_ptrs[t]->spatial_h != spatial_h ||
            tiles_ptrs[t]->spatial_w != spatial_w ||
            tiles_ptrs[t]->H_px != H_px ||
            tiles_ptrs[t]->W_px != W_px) {
            NNOPT_ERROR_FMT("siglip_vision_forward_batched: tile %d shape mismatch", t);
            return false;
        }
    }
    if (spatial_h <= 0 || spatial_w <= 0 || (spatial_h & 1) || (spatial_w & 1)) {
        NNOPT_ERROR_FMT("siglip_vision_forward_batched: invalid spatial dims (%dx%d)", spatial_h, spatial_w);
        return false;
    }
    if (spatial_h * 16 != H_px || spatial_w * 16 != W_px) {
        NNOPT_ERROR_FMT("siglip_vision_forward_batched: spatial(%dx%d) != pixel(%dx%d)/16",
                        spatial_h, spatial_w, H_px, W_px);
        return false;
    }

    const int patch = 16;
    const int patch_flat = 3 * patch * patch;                                  // 768
    const int hidden = MODEL_CONFIG::VISION_CONFIG_HIDDEN_SIZE;                // 768
    const int intermediate = MODEL_CONFIG::VISION_CONFIG_INTERMEDIATE_SIZE;    // 3072
    const int num_heads = MODEL_CONFIG::VISION_CONFIG_NUM_ATTENTION_HEADS;     // 12
    const int head_dim = hidden / num_heads;                                   // 64
    const int num_layers = MODEL_CONFIG::VISION_CONFIG_NUM_HIDDEN_LAYERS;      // 12
    const float ln_eps = MODEL_CONFIG::LAYER_NORM_EPS;                         // 1e-6
    const int projector_in = hidden * MODEL_CONFIG::DOWNSAMPLE_FACTOR * MODEL_CONFIG::DOWNSAMPLE_FACTOR; // 3072
    const int projector_hidden = MODEL_CONFIG::PROJECTOR_HIDDEN_SIZE;          // 2048
    const int lm_hidden = MODEL_CONFIG::HIDDEN_SIZE;                           // 1024 (LM hidden)

    const int patches_per_tile = spatial_h * spatial_w;     // e.g. 1024
    const int M = N_tiles * patches_per_tile;               // e.g. 6144

    cl_int err = CL_SUCCESS;

    // Required weights.
    cl_mem patch_w   = weights.get_buffer("model.vision_tower.vision_model.embeddings.patch_embedding.weight");
    cl_mem patch_b   = weights.get_buffer("model.vision_tower.vision_model.embeddings.patch_embedding.bias");
    cl_mem pos_w     = weights.get_buffer("model.vision_tower.vision_model.embeddings.position_embedding.weight");
    cl_mem post_ln_w = weights.get_buffer("model.vision_tower.vision_model.post_layernorm.weight");
    cl_mem post_ln_b = weights.get_buffer("model.vision_tower.vision_model.post_layernorm.bias");
    cl_mem p1_w = weights.get_buffer("model.multi_modal_projector.linear_1.weight");
    cl_mem p1_b = weights.get_buffer("model.multi_modal_projector.linear_1.bias");
    cl_mem p2_w = weights.get_buffer("model.multi_modal_projector.linear_2.weight");
    cl_mem p2_b = weights.get_buffer("model.multi_modal_projector.linear_2.bias");
    if (!patch_w || !patch_b || !pos_w || !post_ln_w || !post_ln_b ||
        !p1_w || !p1_b || !p2_w || !p2_b) {
        NNOPT_ERROR("siglip_vision_forward_batched: missing required vision/projector weight");
        return false;
    }

    // ── Step 1: per-tile upload + normalize + patchify into the batched
    // 'patches' buffer (M, patch_flat=768). We can't run normalize on one
    // big CHW buffer because tiles aren't contiguous in CHW (each tile is
    // its own image plane), so do per-tile dispatch but write directly
    // into the shared 'patches' buffer at byte offsets.
    cl_mem patches  = lfm2_alloc(cl_ctx, (size_t)M * patch_flat, "siglip_patches_batched");
    cl_mem patch_emb = lfm2_alloc(cl_ctx, (size_t)M * hidden, "siglip_patch_emb_batched");
    cl_mem pos_resized = lfm2_alloc(cl_ctx, (size_t)spatial_h * spatial_w * hidden, "siglip_pos_resized_batched");
    cl_mem x        = lfm2_alloc(cl_ctx, (size_t)M * hidden, "siglip_x_batched");
    if (!patches || !patch_emb || !pos_resized || !x) {
        if (patches) clReleaseMemObject(patches);
        if (patch_emb) clReleaseMemObject(patch_emb);
        if (pos_resized) clReleaseMemObject(pos_resized);
        if (x) clReleaseMemObject(x);
        return false;
    }

    cl_kernel norm_k = image_kernel(cl_ctx, "kernels/image_normalize.cl", "image_normalize");
    cl_kernel patchify_k = image_kernel(cl_ctx, "kernels/image_patchify.cl", "image_patchify");
    if (!norm_k || !patchify_k) {
        if (norm_k) clReleaseKernel(norm_k);
        if (patchify_k) clReleaseKernel(patchify_k);
        clReleaseMemObject(patches); clReleaseMemObject(patch_emb);
        clReleaseMemObject(pos_resized); clReleaseMemObject(x);
        return false;
    }

    const size_t patches_slice_elems = (size_t)patches_per_tile * (size_t)patch_flat;
    const size_t patches_slice_bytes = patches_slice_elems * sizeof(nnopt_storage_t);

    for (int t = 0; t < N_tiles; ++t) {
        const auto& tile = *tiles_ptrs[t];

        // Per-tile normalized fp16 image (CHW). Local scratch (small enough).
        cl_mem src = clCreateBuffer(cl_ctx.context(), CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
                                    tile.rgb.size(), const_cast<uint8_t*>(tile.rgb.data()), &err);
        if (err != CL_SUCCESS || !src) {
            NNOPT_ERROR_FMT("siglip_encoder_batched: src upload (tile %d) failed (%d)", t, (int)err);
            clReleaseKernel(norm_k); clReleaseKernel(patchify_k);
            clReleaseMemObject(patches); clReleaseMemObject(patch_emb);
            clReleaseMemObject(pos_resized); clReleaseMemObject(x);
            return false;
        }
        cl_mem norm = lfm2_alloc(cl_ctx, (size_t)H_px * W_px * 3u, "siglip_norm_batched");
        if (!norm) {
            clReleaseMemObject(src);
            clReleaseKernel(norm_k); clReleaseKernel(patchify_k);
            clReleaseMemObject(patches); clReleaseMemObject(patch_emb);
            clReleaseMemObject(pos_resized); clReleaseMemObject(x);
            return false;
        }
        float mean = 0.5f, stdev = 0.5f;
        int Hpx = H_px, Wpx = W_px;
        if (!set_arg_local(norm_k, 0, sizeof(cl_mem), &src, "src") ||
            !set_arg_local(norm_k, 1, sizeof(cl_mem), &norm, "dst") ||
            !set_arg_local(norm_k, 2, sizeof(int),    &Hpx, "H") ||
            !set_arg_local(norm_k, 3, sizeof(int),    &Wpx, "W") ||
            !set_arg_local(norm_k, 4, sizeof(float),  &mean, "mr") ||
            !set_arg_local(norm_k, 5, sizeof(float),  &mean, "mg") ||
            !set_arg_local(norm_k, 6, sizeof(float),  &mean, "mb") ||
            !set_arg_local(norm_k, 7, sizeof(float),  &stdev, "sr") ||
            !set_arg_local(norm_k, 8, sizeof(float),  &stdev, "sg") ||
            !set_arg_local(norm_k, 9, sizeof(float),  &stdev, "sb") ||
            !enqueue2(queue, norm_k, (size_t)W_px, (size_t)H_px, "siglip_image_normalize_batched")) {
            clReleaseMemObject(norm); clReleaseMemObject(src);
            clReleaseKernel(norm_k); clReleaseKernel(patchify_k);
            clReleaseMemObject(patches); clReleaseMemObject(patch_emb);
            clReleaseMemObject(pos_resized); clReleaseMemObject(x);
            return false;
        }

        // Per-tile patches sub-buffer at byte offset t * patches_slice_bytes.
        cl_buffer_region region{ (size_t)t * patches_slice_bytes, patches_slice_bytes };
        cl_mem patches_sub = clCreateSubBuffer(patches, CL_MEM_WRITE_ONLY,
                                               CL_BUFFER_CREATE_TYPE_REGION, &region, &err);
        if (err != CL_SUCCESS || !patches_sub) {
            NNOPT_ERROR_FMT("siglip_encoder_batched: patches_sub create (tile %d) failed (%d)", t, (int)err);
            clReleaseMemObject(norm); clReleaseMemObject(src);
            clReleaseKernel(norm_k); clReleaseKernel(patchify_k);
            clReleaseMemObject(patches); clReleaseMemObject(patch_emb);
            clReleaseMemObject(pos_resized); clReleaseMemObject(x);
            return false;
        }
        int patch_const = patch;
        if (!set_arg_local(patchify_k, 0, sizeof(cl_mem), &norm, "image") ||
            !set_arg_local(patchify_k, 1, sizeof(cl_mem), &patches_sub, "patches") ||
            !set_arg_local(patchify_k, 2, sizeof(int),    &Hpx, "H") ||
            !set_arg_local(patchify_k, 3, sizeof(int),    &Wpx, "W") ||
            !set_arg_local(patchify_k, 4, sizeof(int),    &patch_const, "patch") ||
            !enqueue2(queue, patchify_k, (size_t)patches_per_tile, (size_t)patch_flat, "siglip_image_patchify_batched")) {
            clReleaseMemObject(patches_sub);
            clReleaseMemObject(norm); clReleaseMemObject(src);
            clReleaseKernel(norm_k); clReleaseKernel(patchify_k);
            clReleaseMemObject(patches); clReleaseMemObject(patch_emb);
            clReleaseMemObject(pos_resized); clReleaseMemObject(x);
            return false;
        }
        clReleaseMemObject(patches_sub);
        clReleaseMemObject(norm);
        clReleaseMemObject(src);
    }
    clReleaseKernel(norm_k);
    clReleaseKernel(patchify_k);

    // ── Step 2: patch_embedding — ONE CLBlast call on M=N_tiles*patches.
    if (!pytorch_linear(queue, M, hidden, patch_flat, patches, patch_w, patch_emb)) {
        clReleaseMemObject(patches); clReleaseMemObject(patch_emb);
        clReleaseMemObject(pos_resized); clReleaseMemObject(x);
        return false;
    }
    clReleaseMemObject(patches);
    if (!bias_add(cl_ctx, queue, patch_emb, patch_b, M, hidden)) {
        clReleaseMemObject(patch_emb); clReleaseMemObject(pos_resized); clReleaseMemObject(x);
        return false;
    }

    // ── Step 3: bilinear-resize position table ONCE (shared across tiles).
    if (!bilinear_position_resize(cl_ctx, queue, pos_w, pos_resized, spatial_h, spatial_w, hidden)) {
        clReleaseMemObject(patch_emb); clReleaseMemObject(pos_resized); clReleaseMemObject(x);
        return false;
    }

    // ── Step 4: x[tile_i] = patch_emb[tile_i] + pos_resized — per-tile add.
    const size_t tile_x_elems = (size_t)patches_per_tile * (size_t)hidden;
    const size_t tile_x_bytes = tile_x_elems * sizeof(nnopt_storage_t);
    for (int t = 0; t < N_tiles; ++t) {
        cl_buffer_region region_pe{ (size_t)t * tile_x_bytes, tile_x_bytes };
        cl_int er = CL_SUCCESS;
        cl_mem pe_sub = clCreateSubBuffer(patch_emb, CL_MEM_READ_ONLY,
                                          CL_BUFFER_CREATE_TYPE_REGION, &region_pe, &er);
        if (er != CL_SUCCESS || !pe_sub) {
            NNOPT_ERROR_FMT("siglip_encoder_batched: pe_sub create (tile %d) failed (%d)", t, (int)er);
            clReleaseMemObject(patch_emb); clReleaseMemObject(pos_resized); clReleaseMemObject(x);
            return false;
        }
        cl_mem x_sub = clCreateSubBuffer(x, CL_MEM_WRITE_ONLY,
                                         CL_BUFFER_CREATE_TYPE_REGION, &region_pe, &er);
        if (er != CL_SUCCESS || !x_sub) {
            NNOPT_ERROR_FMT("siglip_encoder_batched: x_sub create (tile %d) failed (%d)", t, (int)er);
            clReleaseMemObject(pe_sub);
            clReleaseMemObject(patch_emb); clReleaseMemObject(pos_resized); clReleaseMemObject(x);
            return false;
        }
        if (!element_add3_buf(cl_ctx, queue, pe_sub, pos_resized, x_sub, (int)tile_x_elems)) {
            clReleaseMemObject(pe_sub); clReleaseMemObject(x_sub);
            clReleaseMemObject(patch_emb); clReleaseMemObject(pos_resized); clReleaseMemObject(x);
            return false;
        }
        clReleaseMemObject(pe_sub);
        clReleaseMemObject(x_sub);
    }
    clReleaseMemObject(patch_emb);
    clReleaseMemObject(pos_resized);

    // ── Step 5: pad-mask (all-1s, shared across tiles). patches_per_tile entries.
    std::vector<int32_t> mask_host(patches_per_tile, 1);
    cl_mem pad_mask = clCreateBuffer(cl_ctx.context(),
                                     CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
                                     (size_t)patches_per_tile * sizeof(int32_t),
                                     mask_host.data(), &err);
    if (err != CL_SUCCESS || !pad_mask) {
        NNOPT_ERROR_FMT("siglip_encoder_batched: pad_mask upload failed (%d)", (int)err);
        clReleaseMemObject(x);
        return false;
    }

    // ── Step 6: per-layer scratch (batched M wide for seq-major buffers; same
    //   M wide for head-major buffers since seq_to_heads keeps element count).
    cl_mem ln_out   = lfm2_alloc(cl_ctx, (size_t)M * hidden, "siglip_ln_out_batched");
    cl_mem q_seq    = lfm2_alloc(cl_ctx, (size_t)M * hidden, "siglip_q_seq_batched");
    cl_mem k_seq    = lfm2_alloc(cl_ctx, (size_t)M * hidden, "siglip_k_seq_batched");
    cl_mem v_seq    = lfm2_alloc(cl_ctx, (size_t)M * hidden, "siglip_v_seq_batched");
    cl_mem q_hm     = lfm2_alloc(cl_ctx, (size_t)M * hidden, "siglip_q_hm_batched");
    cl_mem k_hm     = lfm2_alloc(cl_ctx, (size_t)M * hidden, "siglip_k_hm_batched");
    cl_mem v_hm     = lfm2_alloc(cl_ctx, (size_t)M * hidden, "siglip_v_hm_batched");
    cl_mem attn_hm  = lfm2_alloc(cl_ctx, (size_t)M * hidden, "siglip_attn_hm_batched");
    cl_mem attn_seq = lfm2_alloc(cl_ctx, (size_t)M * hidden, "siglip_attn_seq_batched");
    cl_mem attn_proj = lfm2_alloc(cl_ctx, (size_t)M * hidden, "siglip_attn_proj_batched");
    cl_mem fc1_out  = lfm2_alloc(cl_ctx, (size_t)M * intermediate, "siglip_fc1_out_batched");
    cl_mem mlp_act  = lfm2_alloc(cl_ctx, (size_t)M * intermediate, "siglip_mlp_act_batched");
    cl_mem fc2_out  = lfm2_alloc(cl_ctx, (size_t)M * hidden, "siglip_fc2_out_batched");
    if (!ln_out || !q_seq || !k_seq || !v_seq || !q_hm || !k_hm || !v_hm ||
        !attn_hm || !attn_seq || !attn_proj || !fc1_out || !mlp_act || !fc2_out) {
        if (ln_out) clReleaseMemObject(ln_out);
        if (q_seq) clReleaseMemObject(q_seq);
        if (k_seq) clReleaseMemObject(k_seq);
        if (v_seq) clReleaseMemObject(v_seq);
        if (q_hm) clReleaseMemObject(q_hm);
        if (k_hm) clReleaseMemObject(k_hm);
        if (v_hm) clReleaseMemObject(v_hm);
        if (attn_hm) clReleaseMemObject(attn_hm);
        if (attn_seq) clReleaseMemObject(attn_seq);
        if (attn_proj) clReleaseMemObject(attn_proj);
        if (fc1_out) clReleaseMemObject(fc1_out);
        if (mlp_act) clReleaseMemObject(mlp_act);
        if (fc2_out) clReleaseMemObject(fc2_out);
        clReleaseMemObject(pad_mask); clReleaseMemObject(x);
        return false;
    }

    auto cleanup_layer_scratch = [&]() {
        clReleaseMemObject(ln_out); clReleaseMemObject(q_seq); clReleaseMemObject(k_seq);
        clReleaseMemObject(v_seq); clReleaseMemObject(q_hm); clReleaseMemObject(k_hm);
        clReleaseMemObject(v_hm); clReleaseMemObject(attn_hm); clReleaseMemObject(attn_seq);
        clReleaseMemObject(attn_proj); clReleaseMemObject(fc1_out); clReleaseMemObject(mlp_act);
        clReleaseMemObject(fc2_out); clReleaseMemObject(pad_mask);
    };

    const float scale = 1.0f / std::sqrt((float)head_dim);

    // Per-tile byte offsets used for the attention sub-buffer dance.
    const size_t tile_seq_bytes = (size_t)patches_per_tile * (size_t)hidden * sizeof(nnopt_storage_t);
    // Head-major buffer for one tile is the same byte count: heads*rows*head_dim = hidden*rows.

    for (int li = 0; li < num_layers; ++li) {
        cl_ctx.yield_for_compositor();   // yield GPU between vision layers so the UI doesn't ANR
        std::string p = "model.vision_tower.vision_model.encoder.layers." + std::to_string(li);
        cl_mem ln1_w = weights.get_buffer(p + ".layer_norm1.weight");
        cl_mem ln1_b = weights.get_buffer(p + ".layer_norm1.bias");
        cl_mem qw = weights.get_buffer(p + ".self_attn.q_proj.weight");
        cl_mem qb = weights.get_buffer(p + ".self_attn.q_proj.bias");
        cl_mem kw = weights.get_buffer(p + ".self_attn.k_proj.weight");
        cl_mem kb = weights.get_buffer(p + ".self_attn.k_proj.bias");
        cl_mem vw = weights.get_buffer(p + ".self_attn.v_proj.weight");
        cl_mem vb = weights.get_buffer(p + ".self_attn.v_proj.bias");
        cl_mem ow = weights.get_buffer(p + ".self_attn.out_proj.weight");
        cl_mem ob = weights.get_buffer(p + ".self_attn.out_proj.bias");
        cl_mem ln2_w = weights.get_buffer(p + ".layer_norm2.weight");
        cl_mem ln2_b = weights.get_buffer(p + ".layer_norm2.bias");
        cl_mem fc1_w = weights.get_buffer(p + ".mlp.fc1.weight");
        cl_mem fc1_b = weights.get_buffer(p + ".mlp.fc1.bias");
        cl_mem fc2_w = weights.get_buffer(p + ".mlp.fc2.weight");
        cl_mem fc2_b = weights.get_buffer(p + ".mlp.fc2.bias");
        if (!ln1_w || !ln1_b || !qw || !qb || !kw || !kb || !vw || !vb ||
            !ow || !ob || !ln2_w || !ln2_b || !fc1_w || !fc1_b || !fc2_w || !fc2_b) {
            NNOPT_ERROR_FMT("siglip_encoder_batched: missing layer-%d weight", li);
            cleanup_layer_scratch(); clReleaseMemObject(x);
            return false;
        }

        // ── self-attention pre-norm (batched M).
        if (!ln_rows_bias(cl_ctx, queue, x, ln1_w, ln1_b, ln_out, M, hidden, ln_eps)) {
            cleanup_layer_scratch(); clReleaseMemObject(x);
            return false;
        }
        // Q, K, V projections — ONE CLBlast call each on M=N_tiles*patches.
        if (!pytorch_linear(queue, M, hidden, hidden, ln_out, qw, q_seq) ||
            !pytorch_linear(queue, M, hidden, hidden, ln_out, kw, k_seq) ||
            !pytorch_linear(queue, M, hidden, hidden, ln_out, vw, v_seq)) {
            cleanup_layer_scratch(); clReleaseMemObject(x);
            return false;
        }
        if (!bias_add(cl_ctx, queue, q_seq, qb, M, hidden) ||
            !bias_add(cl_ctx, queue, k_seq, kb, M, hidden) ||
            !bias_add(cl_ctx, queue, v_seq, vb, M, hidden)) {
            cleanup_layer_scratch(); clReleaseMemObject(x);
            return false;
        }

        // ── Per-tile attention: split each Q/K/V into per-tile slices,
        //    reshape to head-major, run flash_attn, reshape back to seq-major.
        for (int t = 0; t < N_tiles; ++t) {
            cl_buffer_region reg{ (size_t)t * tile_seq_bytes, tile_seq_bytes };
            cl_int er = CL_SUCCESS;
            cl_mem q_seq_sub  = clCreateSubBuffer(q_seq,   CL_MEM_READ_ONLY,  CL_BUFFER_CREATE_TYPE_REGION, &reg, &er);
            cl_mem k_seq_sub  = (er == CL_SUCCESS) ? clCreateSubBuffer(k_seq, CL_MEM_READ_ONLY, CL_BUFFER_CREATE_TYPE_REGION, &reg, &er) : nullptr;
            cl_mem v_seq_sub  = (er == CL_SUCCESS) ? clCreateSubBuffer(v_seq, CL_MEM_READ_ONLY, CL_BUFFER_CREATE_TYPE_REGION, &reg, &er) : nullptr;
            cl_mem q_hm_sub   = (er == CL_SUCCESS) ? clCreateSubBuffer(q_hm, CL_MEM_READ_WRITE, CL_BUFFER_CREATE_TYPE_REGION, &reg, &er) : nullptr;
            cl_mem k_hm_sub   = (er == CL_SUCCESS) ? clCreateSubBuffer(k_hm, CL_MEM_READ_WRITE, CL_BUFFER_CREATE_TYPE_REGION, &reg, &er) : nullptr;
            cl_mem v_hm_sub   = (er == CL_SUCCESS) ? clCreateSubBuffer(v_hm, CL_MEM_READ_WRITE, CL_BUFFER_CREATE_TYPE_REGION, &reg, &er) : nullptr;
            cl_mem attn_hm_sub  = (er == CL_SUCCESS) ? clCreateSubBuffer(attn_hm, CL_MEM_READ_WRITE, CL_BUFFER_CREATE_TYPE_REGION, &reg, &er) : nullptr;
            cl_mem attn_seq_sub = (er == CL_SUCCESS) ? clCreateSubBuffer(attn_seq, CL_MEM_WRITE_ONLY, CL_BUFFER_CREATE_TYPE_REGION, &reg, &er) : nullptr;
            if (er != CL_SUCCESS || !q_seq_sub || !k_seq_sub || !v_seq_sub ||
                !q_hm_sub || !k_hm_sub || !v_hm_sub || !attn_hm_sub || !attn_seq_sub) {
                NNOPT_ERROR_FMT("siglip_encoder_batched: attn sub-buffer (layer %d tile %d) failed (%d)", li, t, (int)er);
                if (q_seq_sub) clReleaseMemObject(q_seq_sub);
                if (k_seq_sub) clReleaseMemObject(k_seq_sub);
                if (v_seq_sub) clReleaseMemObject(v_seq_sub);
                if (q_hm_sub) clReleaseMemObject(q_hm_sub);
                if (k_hm_sub) clReleaseMemObject(k_hm_sub);
                if (v_hm_sub) clReleaseMemObject(v_hm_sub);
                if (attn_hm_sub) clReleaseMemObject(attn_hm_sub);
                if (attn_seq_sub) clReleaseMemObject(attn_seq_sub);
                cleanup_layer_scratch(); clReleaseMemObject(x);
                return false;
            }

            bool ok = seq_to_head_major(cl_ctx, queue, q_seq_sub, q_hm_sub, patches_per_tile, num_heads, head_dim) &&
                      seq_to_head_major(cl_ctx, queue, k_seq_sub, k_hm_sub, patches_per_tile, num_heads, head_dim) &&
                      seq_to_head_major(cl_ctx, queue, v_seq_sub, v_hm_sub, patches_per_tile, num_heads, head_dim) &&
                      flash_attn(cl_ctx, queue, q_hm_sub, k_hm_sub, v_hm_sub, pad_mask, attn_hm_sub,
                                 patches_per_tile, patches_per_tile, num_heads, num_heads, head_dim, scale) &&
                      head_major_to_seq(cl_ctx, queue, attn_hm_sub, attn_seq_sub, patches_per_tile, num_heads, head_dim);

            clReleaseMemObject(q_seq_sub); clReleaseMemObject(k_seq_sub); clReleaseMemObject(v_seq_sub);
            clReleaseMemObject(q_hm_sub); clReleaseMemObject(k_hm_sub); clReleaseMemObject(v_hm_sub);
            clReleaseMemObject(attn_hm_sub); clReleaseMemObject(attn_seq_sub);

            if (!ok) {
                cleanup_layer_scratch(); clReleaseMemObject(x);
                return false;
            }
        }

        // ── out_proj — ONE batched CLBlast call.
        if (!pytorch_linear(queue, M, hidden, hidden, attn_seq, ow, attn_proj)) {
            cleanup_layer_scratch(); clReleaseMemObject(x);
            return false;
        }
        if (!bias_add(cl_ctx, queue, attn_proj, ob, M, hidden)) {
            cleanup_layer_scratch(); clReleaseMemObject(x);
            return false;
        }
        // x = x + attn_proj  (full batched).
        if (!element_add_inplace_buf(cl_ctx, queue, x, attn_proj, M * hidden)) {
            cleanup_layer_scratch(); clReleaseMemObject(x);
            return false;
        }

        // ── MLP block (batched).
        if (!ln_rows_bias(cl_ctx, queue, x, ln2_w, ln2_b, ln_out, M, hidden, ln_eps)) {
            cleanup_layer_scratch(); clReleaseMemObject(x);
            return false;
        }
        if (!pytorch_linear(queue, M, intermediate, hidden, ln_out, fc1_w, fc1_out)) {
            cleanup_layer_scratch(); clReleaseMemObject(x);
            return false;
        }
        if (!bias_add(cl_ctx, queue, fc1_out, fc1_b, M, intermediate)) {
            cleanup_layer_scratch(); clReleaseMemObject(x);
            return false;
        }
        if (!gelu_tanh(cl_ctx, queue, fc1_out, mlp_act, M * intermediate)) {
            cleanup_layer_scratch(); clReleaseMemObject(x);
            return false;
        }
        if (!pytorch_linear(queue, M, hidden, intermediate, mlp_act, fc2_w, fc2_out)) {
            cleanup_layer_scratch(); clReleaseMemObject(x);
            return false;
        }
        if (!bias_add(cl_ctx, queue, fc2_out, fc2_b, M, hidden)) {
            cleanup_layer_scratch(); clReleaseMemObject(x);
            return false;
        }
        if (!element_add_inplace_buf(cl_ctx, queue, x, fc2_out, M * hidden)) {
            cleanup_layer_scratch(); clReleaseMemObject(x);
            return false;
        }
    }

    // ── Final post-LayerNorm (batched M).
    if (!ln_rows_bias(cl_ctx, queue, x, post_ln_w, post_ln_b, ln_out, M, hidden, ln_eps)) {
        cleanup_layer_scratch(); clReleaseMemObject(x);
        return false;
    }

    NNOPT_CHECKPOINT_FMT("siglip_vision_forward_batched: encoder done, M=%d tiles=%d", M, N_tiles);

    // ── Per-tile projector: pixel_unshuffle → linear_1 → GELU → linear_2.
    // We could batch these too (output is per-tile, M_proj = N_tiles * (H/2*W/2)),
    // but downstream readback is per-tile and output shape is per-tile-natural;
    // staying per-tile keeps memory low (proj_in is M_proj * 3072 fp16 ≈ ~9MB
    // for 6 full tiles which would otherwise be allocated all at once). Per-tile
    // also keeps numerics identical to the per-tile path.

    const int H2 = spatial_h >> 1;
    const int W2 = spatial_w >> 1;
    const int num_out = H2 * W2;
    per_tile_features_out.resize((size_t)N_tiles);

    cl_mem proj_in     = lfm2_alloc(cl_ctx, (size_t)num_out * projector_in, "projector_in_batched");
    cl_mem proj_hidden = lfm2_alloc(cl_ctx, (size_t)num_out * projector_hidden, "projector_hidden_batched");
    cl_mem proj_gelu   = lfm2_alloc(cl_ctx, (size_t)num_out * projector_hidden, "projector_gelu_batched");
    cl_mem proj_out    = lfm2_alloc(cl_ctx, (size_t)num_out * lm_hidden, "projector_out_batched");
    if (!proj_in || !proj_hidden || !proj_gelu || !proj_out) {
        if (proj_in) clReleaseMemObject(proj_in);
        if (proj_hidden) clReleaseMemObject(proj_hidden);
        if (proj_gelu) clReleaseMemObject(proj_gelu);
        if (proj_out) clReleaseMemObject(proj_out);
        cleanup_layer_scratch(); clReleaseMemObject(x);
        return false;
    }

    for (int t = 0; t < N_tiles; ++t) {
        cl_buffer_region reg{ (size_t)t * tile_seq_bytes, tile_seq_bytes };
        cl_int er = CL_SUCCESS;
        cl_mem enc_sub = clCreateSubBuffer(ln_out, CL_MEM_READ_ONLY,
                                            CL_BUFFER_CREATE_TYPE_REGION, &reg, &er);
        if (er != CL_SUCCESS || !enc_sub) {
            NNOPT_ERROR_FMT("siglip_encoder_batched: enc_sub create (tile %d) failed (%d)", t, (int)er);
            clReleaseMemObject(proj_in); clReleaseMemObject(proj_hidden);
            clReleaseMemObject(proj_gelu); clReleaseMemObject(proj_out);
            cleanup_layer_scratch(); clReleaseMemObject(x);
            return false;
        }

        bool ok = pixel_unshuffle_2d(cl_ctx, queue, enc_sub, proj_in, spatial_h, spatial_w, hidden) &&
                  pytorch_linear(queue, num_out, projector_hidden, projector_in, proj_in, p1_w, proj_hidden) &&
                  bias_add(cl_ctx, queue, proj_hidden, p1_b, num_out, projector_hidden) &&
                  gelu_erf(cl_ctx, queue, proj_hidden, proj_gelu, num_out * projector_hidden) &&
                  pytorch_linear(queue, num_out, lm_hidden, projector_hidden, proj_gelu, p2_w, proj_out) &&
                  bias_add(cl_ctx, queue, proj_out, p2_b, num_out, lm_hidden);
        clReleaseMemObject(enc_sub);
        if (!ok) {
            clReleaseMemObject(proj_in); clReleaseMemObject(proj_hidden);
            clReleaseMemObject(proj_gelu); clReleaseMemObject(proj_out);
            cleanup_layer_scratch(); clReleaseMemObject(x);
            return false;
        }

        // Output: GPU-direct copy if hook is registered, else host readback.
        extern cl_mem g_siglip_gpu_dest_buf;
        extern std::vector<size_t>* g_siglip_gpu_dest_offsets_batched;
        const size_t out_bytes = (size_t)num_out * lm_hidden * sizeof(nnopt_storage_t);
        if (g_siglip_gpu_dest_buf && g_siglip_gpu_dest_offsets_batched &&
            (size_t)t < g_siglip_gpu_dest_offsets_batched->size()) {
            const size_t doff = (*g_siglip_gpu_dest_offsets_batched)[(size_t)t];
            cl_int ce = clEnqueueCopyBuffer(queue, proj_out, g_siglip_gpu_dest_buf,
                                            0, doff, out_bytes, 0, nullptr, nullptr);
            if (ce != CL_SUCCESS) {
                NNOPT_ERROR_FMT("siglip_encoder_batched: gpu copy (tile %d) failed (%d)", t, (int)ce);
                clReleaseMemObject(proj_in); clReleaseMemObject(proj_hidden);
                clReleaseMemObject(proj_gelu); clReleaseMemObject(proj_out);
                cleanup_layer_scratch(); clReleaseMemObject(x);
                return false;
            }
            per_tile_features_out[t].clear();  // signal no host data
            NNOPT_CHECKPOINT_FMT("siglip_vision_forward_batched: tile %d projector done (gpu-direct)", t);
            continue;
        }
        std::vector<nnopt_storage_t> tmp((size_t)num_out * lm_hidden);
        cl_int re = clEnqueueReadBuffer(queue, proj_out, CL_TRUE, 0,
                                        tmp.size() * sizeof(nnopt_storage_t), tmp.data(),
                                        0, nullptr, nullptr);
        if (re != CL_SUCCESS) {
            NNOPT_ERROR_FMT("siglip_encoder_batched: read projector_out (tile %d) failed (%d)", t, (int)re);
            clReleaseMemObject(proj_in); clReleaseMemObject(proj_hidden);
            clReleaseMemObject(proj_gelu); clReleaseMemObject(proj_out);
            cleanup_layer_scratch(); clReleaseMemObject(x);
            return false;
        }
        per_tile_features_out[t].resize(tmp.size());
        for (size_t i = 0; i < tmp.size(); ++i) {
#ifdef NNOPT_USE_FP16
            per_tile_features_out[t][i] = nnopt_f16_to_f32((uint16_t)tmp[i]);
#else
            per_tile_features_out[t][i] = tmp[i];
#endif
        }
        NNOPT_CHECKPOINT_FMT("siglip_vision_forward_batched: tile %d projector done", t);
    }

    clReleaseMemObject(proj_in);
    clReleaseMemObject(proj_hidden);
    clReleaseMemObject(proj_gelu);
    clReleaseMemObject(proj_out);
    cleanup_layer_scratch();
    clReleaseMemObject(x);

    NNOPT_CHECKPOINT("siglip_vision_forward_batched done");
    return true;
}
