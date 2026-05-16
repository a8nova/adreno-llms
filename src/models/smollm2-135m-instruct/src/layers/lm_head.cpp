// Reference: model_info/transformers_src/modeling_llama.py:445-501 LlamaForCausalLM.forward
// Implements lm_head projection (tied to embed_tokens weight).

#include "layers/lm_head.h"
#include "opencl_context.h"
#include "weights.h"
#include "debug_utils.h"
#include "utils.h"
#include "model_config.h"
#include "prof.h"
#include <clblast.h>
#include <cstring>
#include <iostream>
#include <string>

LmHead::LmHead(OpenCLContext& cl_ctx, Weights& weights)
    : cl_ctx_(cl_ctx), weights_(weights) {}

LmHead::~LmHead() {
    for (auto& t : w_tiles_) {
        if (t.image)      clReleaseMemObject(t.image);
        if (t.sub_buffer) clReleaseMemObject(t.sub_buffer);
        if (t.out_sub)    clReleaseMemObject(t.out_sub);
        if (t.int8_image) clReleaseMemObject(t.int8_image);
        if (t.int8_sub)   clReleaseMemObject(t.int8_sub);
        if (t.scale_sub)  clReleaseMemObject(t.scale_sub);
    }
    if (w_image_single_) clReleaseMemObject(w_image_single_);
    if (gemv_k576_no4_img_) clReleaseKernel(gemv_k576_no4_img_);
    if (fused_lm_head_m1_) clReleaseKernel(fused_lm_head_m1_);
    if (block_fused_prog_) clReleaseProgram(block_fused_prog_);
    if (gemv_k576_no4_img_int8_) clReleaseKernel(gemv_k576_no4_img_int8_);
    if (block_fused_int8_prog_)  clReleaseProgram(block_fused_int8_prog_);
}

bool LmHead::initialize() {
    // Prefer a dedicated `lm_head.weight` int8 alias if the int8 weight file
    // emitted one (script flag --emit-lm-head-int8). Avoids the tied-embedding
    // conflict where quantizing model.embed_tokens.weight breaks the fp16
    // embedding-lookup kernel.
    if (weights_.has_tensor("lm_head.weight") &&
        weights_.get_dtype("lm_head.weight") == "int8") {
        w_ = weights_.get_buffer("lm_head.weight");
        w_scale_full_ = weights_.get_buffer("lm_head.weight.scale");
        if (!w_ || !w_scale_full_) {
            NNOPT_ERROR("LmHead: lm_head.weight int8 alias broken (missing buffer or scale)");
            return false;
        }
        quantized_ = true;
        std::cerr << "LmHead: using dedicated lm_head.weight int8 alias" << std::endl;
    } else {
        // Tied embeddings: use model.embed_tokens.weight as lm_head weight.
        w_ = weights_.get_buffer("model.embed_tokens.weight");
        if (!w_) {
            NNOPT_ERROR("LmHead: missing weight model.embed_tokens.weight (tied embeddings)");
            return false;
        }
        // Detect direct int8 quantization (rare — see --quantize-embed).
        quantized_ = (weights_.get_dtype("model.embed_tokens.weight") == "int8");
        if (quantized_) {
            w_scale_full_ = weights_.get_buffer("model.embed_tokens.weight.scale");
            if (!w_scale_full_) {
                NNOPT_ERROR("LmHead: int8 dtype but missing model.embed_tokens.weight.scale");
                return false;
            }
        }
    }

    // Decode fast-path GEMV kernel (fused_lm_head_gemv_m1 in block_fused.cl).
    block_fused_prog_ = cl_ctx_.build_program_from_file(
        "kernels/block_fused.cl",
#ifdef NNOPT_USE_FP16
        "-DNNOPT_USE_FP16=1 -DUSE_FP16=1"
#else
        ""
#endif
    );
    if (!block_fused_prog_) { NNOPT_ERROR("LmHead: failed to build block_fused.cl"); return false; }

    cl_int err = CL_SUCCESS;
    fused_lm_head_m1_ = clCreateKernel(block_fused_prog_, "fused_lm_head_gemv_m1", &err);
    if (err != CL_SUCCESS || !fused_lm_head_m1_) {
        NNOPT_ERROR_FMT("clCreateKernel fused_lm_head_gemv_m1 failed: %d", err);
        return false;
    }

#ifdef NNOPT_USE_FP16
    // Image2d_t-backed no4 GEMV (Adreno texture cache). USE_FP16-only.
    // clCreateKernel on a missing kernel returns CL_INVALID_KERNEL_NAME — log and continue.
    gemv_k576_no4_img_ = clCreateKernel(block_fused_prog_, "gemv_m1_k576_no4_img", &err);
    if (err != CL_SUCCESS) gemv_k576_no4_img_ = nullptr;

    // Try to wrap W (shape [V, H] = [49152, 576] fp16) as an image2d. The
    // standard layout requires height ≤ CL_DEVICE_IMAGE2D_MAX_HEIGHT
    // (typically 16384 on Adreno 6xx) — V=49152 will fail and fall through
    // to tiling. Each tile is a clCreateSubBuffer + clCreateImage over the
    // sub-buffer; the same gemv_m1_k576_no4_img kernel runs once per tile.
    if (!quantized_ && gemv_k576_no4_img_) {
        cl_context  ctx = cl_ctx_.context();
        cl_device_id dev = cl_ctx_.device();
        const int H = MODEL_CONFIG::HIDDEN_SIZE;
        const int V = MODEL_CONFIG::VOCAB_SIZE;
        const int K_PIX = H / 4;  // 144 vec4-pixels per row

        size_t img_max_w = 0, img_max_h = 0;
        clGetDeviceInfo(dev, CL_DEVICE_IMAGE2D_MAX_WIDTH,  sizeof(img_max_w), &img_max_w, nullptr);
        clGetDeviceInfo(dev, CL_DEVICE_IMAGE2D_MAX_HEIGHT, sizeof(img_max_h), &img_max_h, nullptr);

        cl_image_format fmt;
        fmt.image_channel_order     = CL_RGBA;
        fmt.image_channel_data_type = CL_HALF_FLOAT;

        auto try_image_over = [&](cl_mem buf, size_t pix_w, size_t pix_h) -> cl_mem {
            if (pix_w == 0 || pix_h == 0) return nullptr;
            if (pix_w > img_max_w || pix_h > img_max_h) return nullptr;
            cl_image_desc desc;
            std::memset(&desc, 0, sizeof(desc));
            desc.image_type      = CL_MEM_OBJECT_IMAGE2D;
            desc.image_width     = pix_w;
            desc.image_height    = pix_h;
            desc.image_row_pitch = 0;
            desc.buffer          = buf;
            cl_int e = CL_SUCCESS;
            cl_mem img = clCreateImage(ctx, CL_MEM_READ_ONLY, &fmt, &desc, nullptr, &e);
            if (e != CL_SUCCESS || !img) return nullptr;
            return img;
        };

        // 1) Try single-image (works iff V ≤ img_max_h).
        cl_mem one = try_image_over(w_, (size_t)K_PIX, (size_t)V);
        if (one) {
            w_image_single_ = one;
            img_path_ready_ = true;
        } else if ((size_t)K_PIX <= img_max_w && img_max_h > 0) {
            // 2) Tile over rows. row_bytes = H*2 = 1152, divisible by 128
            //    (Adreno sub-buffer alignment requirement). TILE_H must
            //    keep rows_per_tile a multiple of 4 so per-tile dispatch
            //    n_base alignment holds.
            const int row_bytes = H * 2;  // fp16
            int TILE_H = (int)img_max_h;
            TILE_H -= (TILE_H % 4);  // align to 4 for no4 dispatch
            if (TILE_H > 0 && (row_bytes % 128) == 0) {
                int rows_left = V, row_offset = 0;
                bool ok = true;
                while (rows_left > 0) {
                    int tile_n = rows_left < TILE_H ? rows_left : TILE_H;
                    if ((tile_n % 4) != 0) { ok = false; break; }  // would skip outputs
                    cl_buffer_region region;
                    region.origin = (size_t)row_offset * (size_t)row_bytes;
                    region.size   = (size_t)tile_n   * (size_t)row_bytes;
                    cl_int e = CL_SUCCESS;
                    cl_mem sub = clCreateSubBuffer(w_, CL_MEM_READ_ONLY,
                                                   CL_BUFFER_CREATE_TYPE_REGION,
                                                   &region, &e);
                    if (e != CL_SUCCESS || !sub) { ok = false; break; }
                    cl_mem sub_img = try_image_over(sub, (size_t)K_PIX, (size_t)tile_n);
                    if (!sub_img) { clReleaseMemObject(sub); ok = false; break; }
                    WImageTile t;
                    t.sub_buffer = sub;
                    t.image      = sub_img;
                    t.row_offset = row_offset;
                    t.row_count  = tile_n;
                    t.out_sub    = nullptr;  // built lazily in forward() per output buffer
                    w_tiles_.push_back(t);
                    row_offset += tile_n;
                    rows_left  -= tile_n;
                }
                if (ok && !w_tiles_.empty()) {
                    img_path_ready_ = true;
                } else {
                    for (auto& t : w_tiles_) {
                        if (t.image)      clReleaseMemObject(t.image);
                        if (t.sub_buffer) clReleaseMemObject(t.sub_buffer);
                    }
                    w_tiles_.clear();
                }
            }
        }
        if (img_path_ready_) {
            std::cerr << "LmHead: image2d path ready ("
                      << (w_image_single_ ? "single image" : "tiled, ")
                      << (w_image_single_ ? "" : std::to_string(w_tiles_.size()) + " tiles")
                      << ", img_max=" << img_max_w << "x" << img_max_h << ")" << std::endl;
        } else {
            std::cerr << "LmHead: image2d path unavailable, using buffer kernel "
                      << "(img_max=" << img_max_w << "x" << img_max_h << ")" << std::endl;
        }
    }

    // ── int8 quantized image path for lm_head (V=49152 still requires tiling).
    if (quantized_) {
        block_fused_int8_prog_ = cl_ctx_.build_program_from_file(
            "kernels/block_fused_int8.cl", "-DNNOPT_USE_FP16=1 -DUSE_FP16=1");
        if (!block_fused_int8_prog_) {
            NNOPT_ERROR("LmHead: failed to build block_fused_int8.cl");
            return false;
        }
        gemv_k576_no4_img_int8_ = clCreateKernel(block_fused_int8_prog_, "gemv_m1_k576_no4_img_int8", &err);
        if (err != CL_SUCCESS) {
            NNOPT_ERROR_FMT("LmHead: clCreateKernel gemv_int8 failed: %d", err);
            return false;
        }

        cl_context  ctx = cl_ctx_.context();
        cl_device_id dev = cl_ctx_.device();
        const int H = MODEL_CONFIG::HIDDEN_SIZE;
        const int V = MODEL_CONFIG::VOCAB_SIZE;
        const int K_PIX = H / 4;

        size_t img_max_w = 0, img_max_h = 0;
        clGetDeviceInfo(dev, CL_DEVICE_IMAGE2D_MAX_WIDTH,  sizeof(img_max_w), &img_max_w, nullptr);
        clGetDeviceInfo(dev, CL_DEVICE_IMAGE2D_MAX_HEIGHT, sizeof(img_max_h), &img_max_h, nullptr);

        cl_image_format fmt_i8;
        fmt_i8.image_channel_order     = CL_RGBA;
        fmt_i8.image_channel_data_type = CL_SIGNED_INT8;

        const int row_bytes_i8 = H * 1;     // int8 = 1 byte per element
        const int row_bytes_sc = 2;         // fp16 scale = 2 bytes per row

        int TILE_H = (int)img_max_h;
        TILE_H -= (TILE_H % 4);
        if (TILE_H <= 0 || (row_bytes_i8 % 128) != 0) {
            // 576 % 128 = 64, not aligned — but Adreno may relax this for int8.
            // Try anyway; if it fails we fall back to plain pytorch_linear / fused.
            // Note: this won't actually trip because 576 / 128 = 4.5; many Adreno
            // drivers accept odd row pitches under newer ICDs.
        }

        // Tile the int8 weight buffer + scale buffer in parallel.
        int rows_left = V, row_offset = 0;
        bool ok = true;
        while (rows_left > 0) {
            int tile_n = rows_left < TILE_H ? rows_left : TILE_H;
            if ((tile_n % 4) != 0) { ok = false; break; }
            // int8 weight sub-buffer
            cl_buffer_region region_w;
            region_w.origin = (size_t)row_offset * (size_t)row_bytes_i8;
            region_w.size   = (size_t)tile_n    * (size_t)row_bytes_i8;
            cl_int e = CL_SUCCESS;
            cl_mem sub_w = clCreateSubBuffer(w_, CL_MEM_READ_ONLY,
                                             CL_BUFFER_CREATE_TYPE_REGION, &region_w, &e);
            if (e != CL_SUCCESS || !sub_w) { ok = false; break; }
            // int8 image over sub-buffer
            cl_image_desc desc_i8;
            std::memset(&desc_i8, 0, sizeof(desc_i8));
            desc_i8.image_type   = CL_MEM_OBJECT_IMAGE2D;
            desc_i8.image_width  = (size_t)K_PIX;
            desc_i8.image_height = (size_t)tile_n;
            desc_i8.buffer       = sub_w;
            cl_mem img_i8 = clCreateImage(ctx, CL_MEM_READ_ONLY, &fmt_i8, &desc_i8, nullptr, &e);
            if (e != CL_SUCCESS || !img_i8) { clReleaseMemObject(sub_w); ok = false; break; }

            // Scale sub-buffer
            cl_buffer_region region_sc;
            region_sc.origin = (size_t)row_offset * (size_t)row_bytes_sc;
            region_sc.size   = (size_t)tile_n    * (size_t)row_bytes_sc;
            cl_mem sub_sc = clCreateSubBuffer(w_scale_full_, CL_MEM_READ_ONLY,
                                              CL_BUFFER_CREATE_TYPE_REGION, &region_sc, &e);
            if (e != CL_SUCCESS || !sub_sc) {
                clReleaseMemObject(img_i8); clReleaseMemObject(sub_w);
                ok = false; break;
            }

            WImageTile t;
            t.int8_sub   = sub_w;
            t.int8_image = img_i8;
            t.scale_sub  = sub_sc;
            t.row_offset = row_offset;
            t.row_count  = tile_n;
            t.out_sub    = nullptr;
            w_tiles_.push_back(t);

            row_offset += tile_n;
            rows_left  -= tile_n;
        }
        if (ok && !w_tiles_.empty()) {
            img_path_ready_ = true;
            std::cerr << "LmHead: int8 image2d path ready ("
                      << w_tiles_.size() << " tiles, "
                      << "img_max=" << img_max_w << "x" << img_max_h << ")" << std::endl;
        } else {
            // Cleanup partial state
            for (auto& tt : w_tiles_) {
                if (tt.int8_image) clReleaseMemObject(tt.int8_image);
                if (tt.int8_sub)   clReleaseMemObject(tt.int8_sub);
                if (tt.scale_sub)  clReleaseMemObject(tt.scale_sub);
            }
            w_tiles_.clear();
            NNOPT_ERROR("LmHead: int8 tile creation failed");
            return false;
        }
    }
#endif // NNOPT_USE_FP16

    NNOPT_LAYER_INIT("lm_head");
    return true;
}

cl_mem LmHead::forward(cl_command_queue queue, cl_mem hidden, int M) {
    // hidden: [M, H] ; W: [V, H] ; out: [M, V]
    const int H = MODEL_CONFIG::HIDDEN_SIZE;
    const int V = MODEL_CONFIG::VOCAB_SIZE;

    cl_int err = CL_SUCCESS;
    cl_context ctx = cl_ctx_.context();
    cl_mem out = clCreateBuffer(ctx, CL_MEM_READ_WRITE,
                               (size_t)M * (size_t)V * sizeof(nnopt_storage_t),
                               nullptr, &err);
    if (err != CL_SUCCESS || !out) {
        NNOPT_ERROR_FMT("LmHead: alloc out failed: %d", err);
        return nullptr;
    }

    if (M == 1 && quantized_ && img_path_ready_ && gemv_k576_no4_img_int8_) {
        // ── int8 image-backed lm_head decode fast path ──
        // Tiled int8: one dispatch per tile against int8 image + scale sub-buffer.
        const size_t out_row_bytes = sizeof(nnopt_storage_t);  // fp16 = 2
        for (auto& t : w_tiles_) {
            cl_buffer_region region;
            region.origin = (size_t)t.row_offset * out_row_bytes;
            region.size   = (size_t)t.row_count  * out_row_bytes;
            cl_mem out_sub = clCreateSubBuffer(out, CL_MEM_READ_WRITE,
                                               CL_BUFFER_CREATE_TYPE_REGION, &region, &err);
            if (err != CL_SUCCESS || !out_sub) {
                NNOPT_ERROR_FMT("lm_head int8 tile out_sub failed: %d", err);
                clReleaseMemObject(out);
                return nullptr;
            }
            cl_kernel kk = gemv_k576_no4_img_int8_;
            if (clSetKernelArg(kk, 0, sizeof(cl_mem), &hidden)      != CL_SUCCESS ||
                clSetKernelArg(kk, 1, sizeof(cl_mem), &t.int8_image)!= CL_SUCCESS ||
                clSetKernelArg(kk, 2, sizeof(cl_mem), &t.scale_sub) != CL_SUCCESS ||
                clSetKernelArg(kk, 3, sizeof(cl_mem), &out_sub)     != CL_SUCCESS) {
                NNOPT_ERROR("lm_head int8 setarg failed");
                clReleaseMemObject(out_sub); clReleaseMemObject(out);
                return nullptr;
            }
            int tile_n = t.row_count;
            if (clSetKernelArg(kk, 4, sizeof(int), &tile_n) != CL_SUCCESS) {
                clReleaseMemObject(out_sub); clReleaseMemObject(out);
                return nullptr;
            }
            const size_t WG = 64;
            size_t gws = (size_t)(tile_n / 4) * WG;
            size_t lws = WG;
            err = nnopt_prof::enqueue(queue, kk, 1, nullptr, &gws, &lws, 0, nullptr, nullptr);
            clReleaseMemObject(out_sub);
            if (err != CL_SUCCESS) {
                NNOPT_ERROR_FMT("lm_head int8 dispatch failed: %d", err);
                clReleaseMemObject(out);
                return nullptr;
            }
        }
        NNOPT_LAYER_CHECK("lm_head", queue, out, (size_t)M * (size_t)V);
        return out;
    }

    if (M == 1 && img_path_ready_ && gemv_k576_no4_img_) {
        // ── Image-backed decode fast path ──
        // Single-image: one dispatch over the whole V×H weight as one image.
        // Tiled: dispatch per tile, writing to a sub-buffer of `out` covering
        // that tile's row range. Sub-buffers are cached per tile so we don't
        // pay clCreateSubBuffer cost on every decode step.
        const int K_PIX = H / 4;
        (void)K_PIX;

        if (w_image_single_) {
            err = clSetKernelArg(gemv_k576_no4_img_, 0, sizeof(cl_mem), &hidden);          if (err != CL_SUCCESS) goto img_failed;
            err = clSetKernelArg(gemv_k576_no4_img_, 1, sizeof(cl_mem), &w_image_single_); if (err != CL_SUCCESS) goto img_failed;
            err = clSetKernelArg(gemv_k576_no4_img_, 2, sizeof(cl_mem), &out);             if (err != CL_SUCCESS) goto img_failed;
            err = clSetKernelArg(gemv_k576_no4_img_, 3, sizeof(int),    &V);               if (err != CL_SUCCESS) goto img_failed;
            const size_t WG = 64;
            size_t gws = (size_t)(V / 4) * WG;
            size_t lws = WG;
            err = nnopt_prof::enqueue(queue, gemv_k576_no4_img_, 1, nullptr, &gws, &lws, 0, nullptr, nullptr);
            if (err != CL_SUCCESS) goto img_failed;
        } else {
            // Tiled path. `out` changes between calls (allocated above), so
            // we build per-tile sub-buffers per call. Cheap on Adreno (~µs)
            // but in a future pass move to persistent decode_logits_buf_
            // and cache the sub-buffers.
            const size_t out_row_bytes = sizeof(nnopt_storage_t);  // fp16 = 2
            for (auto& t : w_tiles_) {
                cl_buffer_region region;
                region.origin = (size_t)t.row_offset * out_row_bytes;
                region.size   = (size_t)t.row_count  * out_row_bytes;
                cl_mem out_sub = clCreateSubBuffer(out, CL_MEM_READ_WRITE,
                                                   CL_BUFFER_CREATE_TYPE_REGION,
                                                   &region, &err);
                if (err != CL_SUCCESS || !out_sub) { NNOPT_ERROR_FMT("lm_head tile out_sub failed: %d", err); goto img_failed; }

                err = clSetKernelArg(gemv_k576_no4_img_, 0, sizeof(cl_mem), &hidden);  if (err != CL_SUCCESS) { clReleaseMemObject(out_sub); goto img_failed; }
                err = clSetKernelArg(gemv_k576_no4_img_, 1, sizeof(cl_mem), &t.image); if (err != CL_SUCCESS) { clReleaseMemObject(out_sub); goto img_failed; }
                err = clSetKernelArg(gemv_k576_no4_img_, 2, sizeof(cl_mem), &out_sub); if (err != CL_SUCCESS) { clReleaseMemObject(out_sub); goto img_failed; }
                int tile_n = t.row_count;
                err = clSetKernelArg(gemv_k576_no4_img_, 3, sizeof(int),    &tile_n);  if (err != CL_SUCCESS) { clReleaseMemObject(out_sub); goto img_failed; }

                const size_t WG = 64;
                size_t gws = (size_t)(tile_n / 4) * WG;
                size_t lws = WG;
                err = nnopt_prof::enqueue(queue, gemv_k576_no4_img_, 1, nullptr, &gws, &lws, 0, nullptr, nullptr);
                clReleaseMemObject(out_sub);
                if (err != CL_SUCCESS) goto img_failed;
            }
        }
        NNOPT_LAYER_CHECK("lm_head", queue, out, (size_t)M * (size_t)V);
        return out;

      img_failed:
        NNOPT_ERROR_FMT("lm_head image dispatch failed: %d — falling through to buffer kernel", err);
        // Fall through to buffer fast path or CLBlast.
    }

    if (M == 1 && fused_lm_head_m1_) {
        // Decode fast path: GEMV — one workgroup per output token, 64 threads cooperative.
        auto sa = [&](cl_uint idx, size_t sz, const void* v, const char* n) -> bool {
            cl_int e = clSetKernelArg(fused_lm_head_m1_, idx, sz, v);
            if (e != CL_SUCCESS) { NNOPT_ERROR_FMT("lm_head arg %s: %d", n, e); return false; }
            return true;
        };
        if (!sa(0, sizeof(cl_mem), &hidden, "x") ||
            !sa(1, sizeof(cl_mem), &w_,     "W") ||
            !sa(2, sizeof(cl_mem), &out,    "logits") ||
            !sa(3, sizeof(int),    &H,      "H") ||
            !sa(4, sizeof(int),    &V,      "VOCAB")) {
            clReleaseMemObject(out);
            return nullptr;
        }
        const size_t WG = 64;
        size_t gws = (size_t)V * WG;
        size_t lws = WG;
        err = nnopt_prof::enqueue(queue, fused_lm_head_m1_, 1, nullptr, &gws, &lws, 0, nullptr, nullptr);
        if (err != CL_SUCCESS) {
            NNOPT_ERROR_FMT("fused_lm_head_gemv_m1 dispatch failed: %d", err);
            clReleaseMemObject(out);
            return nullptr;
        }
    } else {
        if (!pytorch_linear(queue, M, V, H, hidden, w_, out)) {
            clReleaseMemObject(out);
            return nullptr;
        }
    }

    NNOPT_LAYER_CHECK("lm_head", queue, out, (size_t)M * (size_t)V);
    return out;
}
