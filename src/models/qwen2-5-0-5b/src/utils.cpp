#include "utils.h"
#include "prof.h"
#include "debug_utils.h"   // NNOPT_ERROR_FMT — used by element_add / pytorch_linear / etc.
#include "opencl_context.h" // OpenCLContext::build_cached_program_from_queue

#include <clblast.h>       // clblast::Gemm — used by pytorch_linear (dtype-templated dispatch).

#include <fstream>
#include <cstring>
#include <algorithm>
#include <numeric>
#include <cstdint>
#include <vector>
#include <unordered_map>

// ──────────────────────────────────────────────
// gemv_m1: custom cooperative GEMV for M=1 decode path.
// Lazy-built on first call so call sites (q/k/v/o/gate/up/down/lm_head) need
// no plumbing changes. Specialized kernels for K=896 and K=4864 — Qwen's
// hidden_size and intermediate_size — both hard-unroll the K loop.
// Generic K kernel handles other sizes when K%(WG*4)==0.
// ──────────────────────────────────────────────
namespace {

constexpr int GEMV_WG = 64;

struct GemvM1State {
    cl_program program = nullptr;
    cl_kernel  kernel_k896      = nullptr;  // single-output (small N or N%4!=0)
    cl_kernel  kernel_k4864     = nullptr;
    cl_kernel  kernel_k896_no4  = nullptr;  // 4-output-per-WG (large N, N%4==0)
    cl_kernel  kernel_k4864_no4 = nullptr;
    cl_kernel  kernel_k896_no4_img  = nullptr;  // image2d_t-backed weight (Adreno texture cache)
    cl_kernel  kernel_k4864_no4_img = nullptr;
    cl_kernel  kernel_k896_no8_img  = nullptr;  // 8-output-per-WG (N%8==0)
    cl_kernel  kernel_k4864_no8_img = nullptr;
    cl_kernel  kernel_k896_no4_img_add  = nullptr;  // fused + residual_add (compiler hint; not dispatched on Adreno 620)
    cl_kernel  kernel_k4864_no4_img_add = nullptr;
    cl_kernel  kernel_kv_write = nullptr;               // recordable replacement for clEnqueueCopyBuffer
    bool       tried = false;
    bool       ok    = false;

    // Per-buffer image2d_t view cache. Lazy-allocated when a weight is
    // first routed through gemv_m1's image path. Released by the global
    // dtor (atexit) — OpenCL will release on context destroy anyway.
    //
    // Standard entries hold one (image) and tiles is empty. Tiled
    // entries (only used for weights whose row count exceeds the
    // device's CL_DEVICE_IMAGE2D_MAX_HEIGHT, e.g. lm_head N=151936)
    // hold a vector of (sub_buffer, sub_image, row_offset, row_count)
    // and the kernel is dispatched once per tile, writing to a
    // sub-buffer of the output.
    struct WImageTile {
        cl_mem sub_buffer = nullptr;
        cl_mem image      = nullptr;
        int    row_offset = 0;   // first W-row index in this tile
        int    row_count  = 0;   // number of W-rows in this tile
    };
    struct WImageEntry {
        cl_mem image = nullptr;          // single-image case (most weights)
        std::vector<WImageTile> tiles;   // multi-tile case (lm_head)
    };
    std::unordered_map<cl_mem, WImageEntry> w_image_cache;
    // Negative cache: weights we've tried to image-wrap and failed for
    // (e.g., dimensions exceed device limits even after tiling).
    std::unordered_map<cl_mem, char> w_image_skip;

    // Per-device image2d max dims. Queried once on first use.
    bool   limits_known = false;
    size_t img_max_w    = 0;
    size_t img_max_h    = 0;
};

static GemvM1State& gemv_state() {
    static GemvM1State s;
    return s;
}

static bool ensure_gemv_m1(cl_command_queue queue) {
    GemvM1State& s = gemv_state();
    if (s.tried) return s.ok;
    s.tried = true;

    cl_context ctx = nullptr;
    cl_int err = clGetCommandQueueInfo(queue, CL_QUEUE_CONTEXT, sizeof(ctx), &ctx, nullptr);
    if (err != CL_SUCCESS || !ctx) {
        NNOPT_ERROR_FMT("ensure_gemv_m1: clGetCommandQueueInfo failed: %d", err);
        return false;
    }
    cl_device_id dev = nullptr;
    err = clGetCommandQueueInfo(queue, CL_QUEUE_DEVICE, sizeof(dev), &dev, nullptr);
    if (err != CL_SUCCESS || !dev) {
        NNOPT_ERROR_FMT("ensure_gemv_m1: clGetCommandQueueInfo(DEVICE) failed: %d", err);
        return false;
    }

    // Read kernels/gemv_m1.cl from cwd (same path the rest of the model uses).
    std::ifstream f("kernels/gemv_m1.cl", std::ios::binary);
    if (!f.is_open()) {
        NNOPT_ERROR("ensure_gemv_m1: cannot open kernels/gemv_m1.cl");
        return false;
    }
    std::string src((std::istreambuf_iterator<char>(f)), std::istreambuf_iterator<char>());

    // Route through the shared cached-program helper — gives us the
    // persistent disk binary cache (cold-start TTFT fix) and
    // -cl-fast-relaxed-math for free.
    s.program = OpenCLContext::build_cached_program_from_queue(queue, src, "");
    if (!s.program) {
        NNOPT_ERROR("ensure_gemv_m1: build_cached_program_from_queue failed");
        return false;
    }
    (void)ctx; (void)dev;  // kept above for future direct-API use

    s.kernel_k896 = clCreateKernel(s.program, "gemv_m1_k896", &err);
    if (err != CL_SUCCESS) { NNOPT_ERROR_FMT("ensure_gemv_m1: clCreateKernel(gemv_m1_k896) failed: %d", err); return false; }
    s.kernel_k4864 = clCreateKernel(s.program, "gemv_m1_k4864", &err);
    if (err != CL_SUCCESS) { NNOPT_ERROR_FMT("ensure_gemv_m1: clCreateKernel(gemv_m1_k4864) failed: %d", err); return false; }
    s.kernel_k896_no4 = clCreateKernel(s.program, "gemv_m1_k896_no4", &err);
    if (err != CL_SUCCESS) { NNOPT_ERROR_FMT("ensure_gemv_m1: clCreateKernel(gemv_m1_k896_no4) failed: %d", err); return false; }
    s.kernel_k4864_no4 = clCreateKernel(s.program, "gemv_m1_k4864_no4", &err);
    if (err != CL_SUCCESS) { NNOPT_ERROR_FMT("ensure_gemv_m1: clCreateKernel(gemv_m1_k4864_no4) failed: %d", err); return false; }

    // Image-backed variants are USE_FP16-only; clCreateKernel on an undefined
    // kernel returns CL_INVALID_KERNEL_NAME — log and continue without them
    // (caller falls back to the buffer-backed gemv).
    s.kernel_k896_no4_img  = clCreateKernel(s.program, "gemv_m1_k896_no4_img", &err);
    if (err != CL_SUCCESS) { s.kernel_k896_no4_img = nullptr; }
    s.kernel_k4864_no4_img = clCreateKernel(s.program, "gemv_m1_k4864_no4_img", &err);
    if (err != CL_SUCCESS) { s.kernel_k4864_no4_img = nullptr; }
    s.kernel_k896_no4_img_add = clCreateKernel(s.program, "gemv_m1_k896_no4_img_add", &err);
    if (err != CL_SUCCESS) { s.kernel_k896_no4_img_add = nullptr; }
    s.kernel_k4864_no4_img_add = clCreateKernel(s.program, "gemv_m1_k4864_no4_img_add", &err);
    if (err != CL_SUCCESS) { s.kernel_k4864_no4_img_add = nullptr; }
    s.kernel_kv_write = clCreateKernel(s.program, "kv_write", &err);
    if (err != CL_SUCCESS) { s.kernel_kv_write = nullptr; }

    // no8 kernels (gemv_m1_no8.cl) measured SLOWER than no4 on Adreno 620:
    // 8.23 vs 9.02 tok/s. 8 simultaneous W reads per iteration increases
    // register pressure enough to reduce effective WG occupancy beyond what
    // the 2× WG-count reduction saves. Disabled; kernel_k896/k4864_no8_img stay null.

    s.ok = true;
    return true;
}

// Look up (or create on first use) an image2d_t view of fp16 weight buffer W
// shaped [N, K]. Returns the cached entry — either {image, tiles=[]} for
// the standard single-image case, or {image=nullptr, tiles=[...]} when N
// exceeds CL_DEVICE_IMAGE2D_MAX_HEIGHT and we had to split into tiles.
// On failure returns an empty entry → caller falls back to buffer kernel.
//
// Why tile rather than pack: Adreno's texture cache is 2D-tiled. The no4
// kernel reads (pix, n_base+0..3) — 4 consecutive image rows at the same
// x — so the 4 reads hit the same texture tile. A "packed" layout where
// PACK W-rows live in one image-row destroys that locality (4 outputs land
// at far-apart x offsets), measured 1.75× regression. Tiling preserves the
// access pattern: each tile is a standard-layout image and we dispatch
// the existing gemv_m1_kK_no4_img kernel once per tile.
static GemvM1State::WImageEntry get_or_create_w_image(
    cl_command_queue queue, cl_mem W, int N, int K) {
    GemvM1State& s = gemv_state();
    GemvM1State::WImageEntry kEmpty;
    auto it = s.w_image_cache.find(W);
    if (it != s.w_image_cache.end()) return it->second;
    if (s.w_image_skip.count(W)) return kEmpty;

    cl_context  ctx = nullptr;
    cl_device_id dev = nullptr;
    if (clGetCommandQueueInfo(queue, CL_QUEUE_CONTEXT, sizeof(ctx), &ctx, nullptr) != CL_SUCCESS ||
        clGetCommandQueueInfo(queue, CL_QUEUE_DEVICE,  sizeof(dev), &dev, nullptr) != CL_SUCCESS) {
        s.w_image_skip[W] = 1;
        return kEmpty;
    }

    if (!s.limits_known) {
        clGetDeviceInfo(dev, CL_DEVICE_IMAGE2D_MAX_WIDTH,  sizeof(s.img_max_w), &s.img_max_w, nullptr);
        clGetDeviceInfo(dev, CL_DEVICE_IMAGE2D_MAX_HEIGHT, sizeof(s.img_max_h), &s.img_max_h, nullptr);
        s.limits_known = true;
    }

    cl_image_format fmt;
    fmt.image_channel_order     = CL_RGBA;
    fmt.image_channel_data_type = CL_HALF_FLOAT;

    auto try_create_image_over_buffer = [&](cl_mem buf, size_t pix_w, size_t pix_h) -> cl_mem {
        if (pix_w == 0 || pix_h == 0) return nullptr;
        if (pix_w > s.img_max_w || pix_h > s.img_max_h) return nullptr;

        cl_image_desc desc;
        std::memset(&desc, 0, sizeof(desc));
        desc.image_type      = CL_MEM_OBJECT_IMAGE2D;
        desc.image_width     = pix_w;
        desc.image_height    = pix_h;
        desc.image_row_pitch = 0;
        desc.buffer          = buf;

        cl_int err = CL_SUCCESS;
        cl_mem img = clCreateImage(ctx, CL_MEM_READ_ONLY, &fmt, &desc, nullptr, &err);
        if (err != CL_SUCCESS || !img) return nullptr;
        return img;
    };

    // 1) Try standard layout (single image covering the whole weight).
    {
        cl_mem img = try_create_image_over_buffer(W, (size_t)(K / 4), (size_t)N);
        if (img) {
            GemvM1State::WImageEntry e;
            e.image = img;
            s.w_image_cache[W] = e;
            return e;
        }
    }

    // 2) Tiling fallback for weights whose row count exceeds img_max_h.
    //    Carve W into chunks of TILE_H rows, create a sub-buffer per chunk,
    //    and create a standard-layout image2d over each sub-buffer.
    //    Sub-buffer alignment requires the byte offset to be a multiple of
    //    CL_DEVICE_MEM_BASE_ADDR_ALIGN (in bits). On Adreno that's 1024
    //    bits = 128 bytes. With K=896 fp16 (1792 bytes/row), any TILE_H
    //    that's a multiple of 1 satisfies (1792 already divisible by 128).
    if (K > 0 && (size_t)(K / 4) <= s.img_max_w) {
        const int TILE_H_MAX = (int)s.img_max_h;
        const int row_bytes  = K * 2;   // fp16
        if (row_bytes > 0 && (row_bytes % 128) == 0 && N > 0) {
            std::vector<GemvM1State::WImageTile> tiles;
            int rows_left   = N;
            int row_offset  = 0;
            bool ok         = true;
            while (rows_left > 0) {
                int tile_n = rows_left < TILE_H_MAX ? rows_left : TILE_H_MAX;
                cl_buffer_region region;
                region.origin = (size_t)row_offset * (size_t)row_bytes;
                region.size   = (size_t)tile_n   * (size_t)row_bytes;

                cl_int err = CL_SUCCESS;
                cl_mem sub = clCreateSubBuffer(W, CL_MEM_READ_ONLY,
                                               CL_BUFFER_CREATE_TYPE_REGION,
                                               &region, &err);
                if (err != CL_SUCCESS || !sub) { ok = false; break; }

                cl_mem sub_img = try_create_image_over_buffer(sub, (size_t)(K / 4), (size_t)tile_n);
                if (!sub_img) {
                    clReleaseMemObject(sub);
                    ok = false;
                    break;
                }
                GemvM1State::WImageTile t;
                t.sub_buffer = sub;
                t.image      = sub_img;
                t.row_offset = row_offset;
                t.row_count  = tile_n;
                tiles.push_back(t);

                row_offset += tile_n;
                rows_left  -= tile_n;
            }
            if (ok && !tiles.empty()) {
                GemvM1State::WImageEntry e;
                e.tiles = std::move(tiles);
                s.w_image_cache[W] = e;
                return e;
            }
            // Roll back any partially-built tiles on failure.
            for (auto& t : tiles) {
                if (t.image)      clReleaseMemObject(t.image);
                if (t.sub_buffer) clReleaseMemObject(t.sub_buffer);
            }
        }
    }

    // 3) No image route available — caller falls back to buffer kernel.
    s.w_image_skip[W] = 1;
    return kEmpty;
}

// Cache for sub-buffers of a given output cl_mem at (offset_bytes, size_bytes).
// Reused across decode steps if the same (out, offset, size) tuple recurs.
// Adreno's clCreateSubBuffer is cheap (~few µs) but pooling avoids per-call cost.
namespace {
struct OutSubKey {
    cl_mem buf;
    size_t off;
    size_t sz;
    bool operator==(const OutSubKey& o) const {
        return buf == o.buf && off == o.off && sz == o.sz;
    }
};
struct OutSubKeyHash {
    size_t operator()(const OutSubKey& k) const {
        size_t h = (size_t)k.buf;
        h = h * 0x9e3779b97f4a7c15ULL ^ k.off;
        h = h * 0x9e3779b97f4a7c15ULL ^ k.sz;
        return h;
    }
};
static std::unordered_map<OutSubKey, cl_mem, OutSubKeyHash> g_out_sub_cache;
}

static cl_mem get_or_create_out_sub(cl_mem out, size_t off_bytes, size_t sz_bytes) {
    OutSubKey key{out, off_bytes, sz_bytes};
    auto it = g_out_sub_cache.find(key);
    if (it != g_out_sub_cache.end()) return it->second;

    cl_buffer_region region{off_bytes, sz_bytes};
    cl_int err = CL_SUCCESS;
    cl_mem sub = clCreateSubBuffer(out, CL_MEM_READ_WRITE,
                                   CL_BUFFER_CREATE_TYPE_REGION, &region, &err);
    if (err != CL_SUCCESS || !sub) return nullptr;
    g_out_sub_cache[key] = sub;
    return sub;
}

// Returns true if the call site was handled by gemv_m1; false ⇒ fall through to CLBlast.
// Routing rules:
//   M == 1, K ∈ {896, 4864}:
//     - N % 4 == 0 AND N >= 8 AND image creation OK → image-backed no4 (texture cache)
//     - N % 4 == 0 AND N >= 8                       → buffer-backed no4
//     - else                                         → buffer-backed single-output
//   else                                             → CLBlast HGemm
static bool try_gemv_m1(cl_command_queue queue,
                        int M, int N, int K,
                        cl_mem x, cl_mem W, cl_mem out) {
    if (M != 1) return false;
    if (K != 896 && K != 4864) return false;

    if (!ensure_gemv_m1(queue)) return false;
    GemvM1State& s = gemv_state();

    const bool use_no4 = (N % 4 == 0) && (N >= 8);

    // Try image-backed (Adreno texture cache) first when no4 applies.
    // Earlier hypothesis: lm_head's tiled image (10 × 28.7 MB tiles) blows
    // out the L1 cache so the buffer kernel might be faster (PDF §7.1.5.3).
    // A/B test 2026-05-03 disproved it: per-token lm_head is 22.6 ms via
    // tiled image vs 30.3 ms via buffer (-25% slower with buffer). The
    // texture ENGINE on Adreno 620 is faster than buffer fetch independent
    // of L1 hits — the engine has dedicated hardware for streaming texel
    // fetches and format conversion that buffer reads can't match.
    //
    if (use_no4) {
        GemvM1State::WImageEntry img_entry = get_or_create_w_image(queue, W, N, K);

        cl_kernel k_img = (K == 896) ? s.kernel_k896_no4_img : s.kernel_k4864_no4_img;
        const int stride = 4;

        // Single-image path (the common case — most weights fit in one image).
        if (img_entry.image && k_img) {
            cl_int err = CL_SUCCESS;
            err = clSetKernelArg(k_img, 0, sizeof(cl_mem), &x);               if (err != CL_SUCCESS) goto fallback_buffer;
            err = clSetKernelArg(k_img, 1, sizeof(cl_mem), &img_entry.image); if (err != CL_SUCCESS) goto fallback_buffer;
            err = clSetKernelArg(k_img, 2, sizeof(cl_mem), &out);             if (err != CL_SUCCESS) goto fallback_buffer;
            err = clSetKernelArg(k_img, 3, sizeof(int),    &N);               if (err != CL_SUCCESS) goto fallback_buffer;
            size_t lws = (size_t)GEMV_WG;
            size_t gws = (size_t)(N / stride) * lws;
            err = nnopt_prof::enqueue(queue, k_img, 1, nullptr, &gws, &lws,
                                         0, nullptr, nullptr);
            if (err == CL_SUCCESS) {
                NNOPT_DEBUG_SYNC(queue);
                return true;
            }
            NNOPT_ERROR_FMT("gemv_m1 image dispatch failed (N=%d K=%d): %d — falling back to buffer kernel",
                            N, K, err);
        }

        // Tiled path: lm_head and any future weight whose row count exceeds
        // the device's image2d height limit. Each tile is a standard-layout
        // sub-image; we dispatch the same image kernel once per tile, with a
        // sub-buffer of `out` so the kernel writes to the correct row range.
        if (!img_entry.tiles.empty() && k_img) {
            const size_t out_row_bytes = sizeof(uint16_t);  // fp16 logits
            cl_int err = CL_SUCCESS;
            for (const auto& t : img_entry.tiles) {
                size_t off_bytes = (size_t)t.row_offset * out_row_bytes;
                size_t sz_bytes  = (size_t)t.row_count  * out_row_bytes;
                cl_mem out_sub = get_or_create_out_sub(out, off_bytes, sz_bytes);
                if (!out_sub) { err = CL_OUT_OF_RESOURCES; break; }

                err = clSetKernelArg(k_img, 0, sizeof(cl_mem), &x);       if (err != CL_SUCCESS) break;
                err = clSetKernelArg(k_img, 1, sizeof(cl_mem), &t.image); if (err != CL_SUCCESS) break;
                err = clSetKernelArg(k_img, 2, sizeof(cl_mem), &out_sub); if (err != CL_SUCCESS) break;
                int tile_n = t.row_count;
                err = clSetKernelArg(k_img, 3, sizeof(int),    &tile_n);  if (err != CL_SUCCESS) break;

                size_t lws = (size_t)GEMV_WG;
                size_t gws = (size_t)(tile_n / 4) * lws;
                err = nnopt_prof::enqueue(queue, k_img, 1, nullptr, &gws, &lws,
                                             0, nullptr, nullptr);
                if (err != CL_SUCCESS) break;
            }
            if (err == CL_SUCCESS) {
                NNOPT_DEBUG_SYNC(queue);
                return true;
            }
            NNOPT_ERROR_FMT("gemv_m1 tiled image dispatch failed (N=%d K=%d tiles=%zu): %d — falling back to buffer",
                            N, K, img_entry.tiles.size(), err);
        }
    }

fallback_buffer:
    cl_kernel k;
    size_t wg_count;
    if (use_no4) {
        k = (K == 896) ? s.kernel_k896_no4 : s.kernel_k4864_no4;
        wg_count = (size_t)(N / 4);
    } else {
        k = (K == 896) ? s.kernel_k896 : s.kernel_k4864;
        wg_count = (size_t)N;
    }
    // k4864 single-output is a noop stub; fall through to CLBlast.
    if (K == 4864 && !use_no4) return false;

    cl_int err = CL_SUCCESS;
    err = clSetKernelArg(k, 0, sizeof(cl_mem), &x);   if (err != CL_SUCCESS) return false;
    err = clSetKernelArg(k, 1, sizeof(cl_mem), &W);   if (err != CL_SUCCESS) return false;
    err = clSetKernelArg(k, 2, sizeof(cl_mem), &out); if (err != CL_SUCCESS) return false;
    err = clSetKernelArg(k, 3, sizeof(int),    &N);   if (err != CL_SUCCESS) return false;

    size_t lws = (size_t)GEMV_WG;
    size_t gws = wg_count * lws;
    err = nnopt_prof::enqueue(queue, k, 1, nullptr, &gws, &lws, 0, nullptr, nullptr);
    if (err != CL_SUCCESS) {
        NNOPT_ERROR_FMT("gemv_m1 dispatch failed (M=%d N=%d K=%d no4=%d): %d",
                        M, N, K, (int)use_no4, err);
        return false;
    }
    NNOPT_DEBUG_SYNC(queue);
    return true;
}

// Fused projection + residual_add for M=1 decode. Same routing rules as
// try_gemv_m1, but the image kernel reads `out[n]` and adds before writing.
// Caller passes the residual destination (= hidden state buffer) as `out`.
// Falls through false to let caller do the unfused two-launch path.
static bool try_gemv_m1_add(cl_command_queue queue,
                            int M, int N, int K,
                            cl_mem x, cl_mem W, cl_mem out) {
    if (M != 1) return false;
    if (K != 896 && K != 4864) return false;
    if ((N % 4) != 0 || N < 8) return false;
    if (!ensure_gemv_m1(queue)) return false;
    GemvM1State& s = gemv_state();
    cl_kernel k_img = (K == 896) ? s.kernel_k896_no4_img_add : s.kernel_k4864_no4_img_add;
    if (!k_img) return false;

    GemvM1State::WImageEntry img_entry = get_or_create_w_image(queue, W, N, K);
    if (!img_entry.image && img_entry.tiles.empty()) return false;

    // Single-image path. The fused-add variant is only wired for the
    // standard single-image case — o_proj (N=896) and down_proj (N=896)
    // are well below the 16384 image-height limit, so neither weight
    // ever needs the tiled path.
    if (img_entry.image) {
        cl_int err = CL_SUCCESS;
        err = clSetKernelArg(k_img, 0, sizeof(cl_mem), &x);               if (err != CL_SUCCESS) return false;
        err = clSetKernelArg(k_img, 1, sizeof(cl_mem), &img_entry.image); if (err != CL_SUCCESS) return false;
        err = clSetKernelArg(k_img, 2, sizeof(cl_mem), &out);             if (err != CL_SUCCESS) return false;
        err = clSetKernelArg(k_img, 3, sizeof(int),    &N);               if (err != CL_SUCCESS) return false;
        size_t lws = (size_t)GEMV_WG;
        size_t gws = (size_t)(N / 4) * lws;
        err = nnopt_prof::enqueue(queue, k_img, 1, nullptr, &gws, &lws,
                                     0, nullptr, nullptr);
        if (err != CL_SUCCESS) {
            NNOPT_ERROR_FMT("gemv_m1 fused-add image dispatch failed (N=%d K=%d): %d",
                            N, K, err);
            return false;
        }
        NNOPT_DEBUG_SYNC(queue);
        return true;
    }
    return false;
}

}  // namespace

// Public façade for the fused gate_proj + up_proj + silu*mul kernel.
// Returns false (caller must fall back) when the predicate doesn't hold.
bool fused_gate_up_silu_m1(cl_command_queue queue,
                           int N, int K,
                           cl_mem x, cl_mem Wg, cl_mem Wu, cl_mem out) {
    if (K != 896) return false;
    if ((N % 4) != 0 || N < 8) return false;
    if (!ensure_gemv_m1(queue)) return false;
    // Kernel removed (Step 7 regression — register spill on Adreno 620).
    (void)x; (void)Wg; (void)Wu; (void)out;
    return false;
}


// ──────────────────────────────────────────────
// IEEE 754 binary16 codec (host-side).
// Bit-exact: handles subnormals, Inf, NaN, saturating overflow on encode.
// Returns float32 on decode. Branch-light implementation, no compiler-half
// intrinsic dependence so it compiles identically across NDK / Linux hosts.
// ──────────────────────────────────────────────

float nnopt_f16_to_f32(uint16_t bits) {
    uint32_t sign = (uint32_t)(bits >> 15) & 0x1u;
    uint32_t exp  = (uint32_t)(bits >> 10) & 0x1Fu;
    uint32_t mant = (uint32_t)(bits      ) & 0x3FFu;
    uint32_t out_sign = sign << 31;
    uint32_t out;
    if (exp == 0) {
        if (mant == 0) {
            out = out_sign;                              // ±0
        } else {
            // Subnormal: normalize.
            int e = -1;
            do { e++; mant <<= 1; } while ((mant & 0x400u) == 0);
            mant &= 0x3FFu;
            uint32_t out_exp = (uint32_t)(127 - 15 - e);
            out = out_sign | (out_exp << 23) | (mant << 13);
        }
    } else if (exp == 0x1F) {
        // Inf or NaN.
        out = out_sign | 0x7F800000u | (mant << 13);
    } else {
        uint32_t out_exp = (uint32_t)(exp - 15 + 127);
        out = out_sign | (out_exp << 23) | (mant << 13);
    }
    float f;
    std::memcpy(&f, &out, sizeof(f));
    return f;
}

uint16_t nnopt_f32_to_f16(float v) {
    uint32_t bits;
    std::memcpy(&bits, &v, sizeof(bits));
    uint32_t sign = (bits >> 31) & 0x1u;
    int32_t  exp  = (int32_t)((bits >> 23) & 0xFFu) - 127 + 15;
    uint32_t mant = bits & 0x7FFFFFu;
    uint16_t out_sign = (uint16_t)(sign << 15);
    if (((bits >> 23) & 0xFFu) == 0xFFu) {
        // Inf / NaN
        uint16_t out_mant = mant ? (uint16_t)((mant >> 13) | 0x200u) : 0u; // preserve NaN-ness
        return (uint16_t)(out_sign | 0x7C00u | out_mant);
    }
    if (exp >= 0x1F) {
        // Saturating overflow → ±Inf
        return (uint16_t)(out_sign | 0x7C00u);
    }
    if (exp <= 0) {
        // Subnormal or underflow.
        if (exp < -10) return out_sign;                     // → ±0
        mant |= 0x800000u;                                  // restore implicit 1
        uint32_t shift = (uint32_t)(14 - exp);
        // Round to nearest even
        uint32_t round_bit = mant & (1u << (shift - 1));
        uint32_t sticky    = mant & ((1u << (shift - 1)) - 1u);
        uint16_t out_mant  = (uint16_t)(mant >> shift);
        if (round_bit && (sticky || (out_mant & 1u))) out_mant++;
        return (uint16_t)(out_sign | out_mant);
    }
    // Normal — round to nearest even.
    uint32_t round_bit = mant & 0x1000u;
    uint32_t sticky    = mant & 0x0FFFu;
    uint16_t out_mant  = (uint16_t)((mant >> 13) & 0x3FFu);
    uint16_t out_exp   = (uint16_t)(exp & 0x1Fu);
    uint16_t out       = (uint16_t)(out_sign | (out_exp << 10) | out_mant);
    if (round_bit && (sticky || (out_mant & 1u))) {
        out++;  // may carry into exp; that's ok per IEEE 754 round-half-to-even.
    }
    return out;
}

float compute_mse(const float* a, const float* b, size_t n) {
    double sum = 0.0;
    for (size_t i = 0; i < n; i++) {
        double diff = (double)a[i] - (double)b[i];
        sum += diff * diff;
    }
    return (float)(sum / n);
}

float compute_max_diff(const float* a, const float* b, size_t n) {
    float max_diff = 0.0f;
    for (size_t i = 0; i < n; i++) {
        float diff = std::abs(a[i] - b[i]);
        if (diff > max_diff) max_diff = diff;
    }
    return max_diff;
}

std::vector<float> load_npy_float32(const std::string& path) {
    std::ifstream file(path, std::ios::binary);
    if (!file.is_open()) return {};

    // Skip numpy header (simplified parser)
    char magic[6];
    file.read(magic, 6);
    uint8_t major, minor;
    file.read(reinterpret_cast<char*>(&major), 1);
    file.read(reinterpret_cast<char*>(&minor), 1);
    uint16_t header_len;
    file.read(reinterpret_cast<char*>(&header_len), 2);

    std::string header(header_len, '\0');
    file.read(&header[0], header_len);

    // Read remaining data as float32
    auto pos = file.tellg();
    file.seekg(0, std::ios::end);
    auto end_pos = file.tellg();
    file.seekg(pos);

    size_t num_bytes = end_pos - pos;
    size_t num_floats = num_bytes / sizeof(float);

    std::vector<float> data(num_floats);
    file.read(reinterpret_cast<char*>(data.data()), num_bytes);

    return data;
}

void save_npy_float32(const std::string& path, const float* data, const std::vector<size_t>& shape) {
    // Minimal .npy writer for float32
    std::ofstream file(path, std::ios::binary);

    // Magic
    file.write("\x93NUMPY", 6);
    uint8_t major = 1, minor = 0;
    file.write(reinterpret_cast<char*>(&major), 1);
    file.write(reinterpret_cast<char*>(&minor), 1);

    // Header
    std::string shape_str = "(";
    for (size_t i = 0; i < shape.size(); i++) {
        shape_str += std::to_string(shape[i]);
        if (i < shape.size() - 1) shape_str += ", ";
    }
    shape_str += ")";

    std::string header = "{'descr': '<f4', 'fortran_order': False, 'shape': " + shape_str + "}";
    // Pad to multiple of 64
    while ((10 + header.size() + 1) % 64 != 0) header += ' ';
    header += '\n';

    uint16_t header_len = (uint16_t)header.size();
    file.write(reinterpret_cast<char*>(&header_len), 2);
    file.write(header.c_str(), header.size());

    // Data
    size_t total = 1;
    for (auto s : shape) total *= s;
    file.write(reinterpret_cast<const char*>(data), total * sizeof(float));
}

// In-place add: a[i] += b[i]. Kernel object cached per program so repeat
// calls don't pay clCreateKernel. Use this for residual adds at decode
// to keep the M=1 hot path allocation-free (Rule FUSE-DECODE-01).
bool element_add_inplace(cl_command_queue queue, cl_program utils_program,
                         cl_mem a, cl_mem b, size_t n) {
    static cl_program s_cached_program = nullptr;
    static cl_kernel  s_cached_kernel  = nullptr;
    if (s_cached_program != utils_program) {
        if (s_cached_kernel) { clReleaseKernel(s_cached_kernel); s_cached_kernel = nullptr; }
        cl_int kerr = CL_SUCCESS;
        s_cached_kernel = clCreateKernel(utils_program, "element_add", &kerr);
        if (kerr != CL_SUCCESS || !s_cached_kernel) {
            NNOPT_ERROR_FMT("element_add_inplace: clCreateKernel failed (%d)", kerr);
            s_cached_kernel = nullptr;
            return false;
        }
        s_cached_program = utils_program;
    }
    int n_int = (int)n;
    cl_int err = clSetKernelArg(s_cached_kernel, 0, sizeof(cl_mem), &a);
    if (err != CL_SUCCESS) { NNOPT_ERROR_FMT("element_add_inplace arg0: %d", err); return false; }
    err = clSetKernelArg(s_cached_kernel, 1, sizeof(cl_mem), &b);
    if (err != CL_SUCCESS) { NNOPT_ERROR_FMT("element_add_inplace arg1: %d", err); return false; }
    err = clSetKernelArg(s_cached_kernel, 2, sizeof(int), &n_int);
    if (err != CL_SUCCESS) { NNOPT_ERROR_FMT("element_add_inplace arg2: %d", err); return false; }
    // Vec4 dispatch — kernel emits 4 fp16 per thread when n%4==0 (HIDDEN=896 ✓).
    // Even-rounded gws covers the tail when n is non-multiple-of-4 (rare).
    size_t gws = (n + 3) >> 2;
    err = nnopt_prof::enqueue(queue, s_cached_kernel, 1, nullptr, &gws, nullptr, 0, nullptr, nullptr);
    if (err != CL_SUCCESS) {
        NNOPT_ERROR_FMT("element_add_inplace: nnopt_prof::enqueue failed (%d)", err);
        return false;
    }
    return true;
}

cl_mem element_add(cl_command_queue queue, cl_program utils_program, cl_mem a, cl_mem b, size_t n) {
    cl_int err;
    cl_context ctx;
    clGetCommandQueueInfo(queue, CL_QUEUE_CONTEXT, sizeof(ctx), &ctx, nullptr);

    // Allocate output buffer (storage_t: cl_half under fp16, float under fp32).
    cl_mem out = clCreateBuffer(ctx, CL_MEM_READ_WRITE, n * sizeof(nnopt_storage_t), nullptr, &err);
    if (err != CL_SUCCESS) {
        NNOPT_ERROR_FMT("element_add: clCreateBuffer failed (%d)", err);
        return nullptr;
    }

    // Copy a into out
    err = clEnqueueCopyBuffer(queue, a, out, 0, 0, n * sizeof(nnopt_storage_t), 0, nullptr, nullptr);
    if (err != CL_SUCCESS) {
        NNOPT_ERROR_FMT("element_add: clEnqueueCopyBuffer failed (%d)", err);
        clReleaseMemObject(out);
        return nullptr;
    }

    // Dispatch element_add kernel: out[i] += b[i]
    cl_kernel kernel = clCreateKernel(utils_program, "element_add", &err);
    if (err != CL_SUCCESS) {
        NNOPT_ERROR_FMT("element_add: clCreateKernel failed (%d)", err);
        clReleaseMemObject(out);
        return nullptr;
    }

    int n_int = (int)n;
    clSetKernelArg(kernel, 0, sizeof(cl_mem), &out);
    clSetKernelArg(kernel, 1, sizeof(cl_mem), &b);
    clSetKernelArg(kernel, 2, sizeof(int), &n_int);

    size_t global_size = n;
    err = nnopt_prof::enqueue(queue, kernel, 1, nullptr, &global_size, nullptr, 0, nullptr, nullptr);
    if (err != CL_SUCCESS) {
        NNOPT_ERROR_FMT("element_add: nnopt_prof::enqueue failed (%d)", err);
    }

    clReleaseKernel(kernel);
    return out;
}

bool split_last_dim_2(cl_command_queue queue, cl_program utils_program,
                      cl_mem src, cl_mem first, cl_mem second,
                      int rows, int half_cols) {
    cl_int err;
    cl_kernel kernel = clCreateKernel(utils_program, "split_last_dim_2", &err);
    if (err != CL_SUCCESS) {
        NNOPT_ERROR_FMT("split_last_dim_2: clCreateKernel failed (%d)", err);
        return false;
    }

    clSetKernelArg(kernel, 0, sizeof(cl_mem), &src);
    clSetKernelArg(kernel, 1, sizeof(cl_mem), &first);
    clSetKernelArg(kernel, 2, sizeof(cl_mem), &second);
    clSetKernelArg(kernel, 3, sizeof(int), &rows);
    clSetKernelArg(kernel, 4, sizeof(int), &half_cols);

    size_t global_size = (size_t)rows * (size_t)half_cols;
    err = nnopt_prof::enqueue(queue, kernel, 1, nullptr, &global_size, nullptr, 0, nullptr, nullptr);
    if (err != CL_SUCCESS) {
        NNOPT_ERROR_FMT("split_last_dim_2: nnopt_prof::enqueue failed (%d)", err);
        clReleaseKernel(kernel);
        return false;
    }

    // SYNC-01: queue is in-order; downstream kernels see this output without
    // explicit sync. NNOPT_DEBUG_SYNC strips to no-op in release.
    NNOPT_DEBUG_SYNC(queue);
    clReleaseKernel(kernel);
    return true;
}

bool pytorch_linear(cl_command_queue queue,
                    int M, int N, int K,
                    cl_mem x, cl_mem W, cl_mem out) {
    // Decode fast path: cooperative GEMV at M=1 for the K sizes we ship a
    // specialization for. Falls through to CLBlast on any miss (prefill,
    // odd K, or build error).
    if (try_gemv_m1(queue, M, N, K, x, W, out)) return true;
    // out[M, N] = x[M, K] @ W[N, K]^T  where W is nn.Linear weight [N, K].
    //
    // CLBlast RowMajor GEMM signature:
    //   C[M,N] = alpha * op(A)[M,K] * op(B)[K,N] + beta * C[M,N]
    // With TransposeA=kNo, TransposeB=kYes, op(B) treats W[N,K] as B[K,N]^T.
    //
    // Leading dimensions for RowMajor:
    //   lda = K (A's stride between rows of A[M,K])
    //   ldb = K (B's stored stride between rows of W[N,K])  ← gotcha
    //   ldc = N (C's stride between rows of C[M,N])
    //
    // Dtype-templated dispatch: HGemm under fp16, SGemm under fp32. Internal
    // accumulation in CLBlast Hgemm is fp32 (verified — square sanity test).
#ifdef NNOPT_USE_FP16
    // Use the portable host-side IEEE 754 fp16 encoder defined above.
    // clblast::FloatToHalf is not portable across CLBlast builds (some
    // versions only expose it when cl_khr_fp16 was enabled at CLBlast
    // build time, leading to "no member named 'FloatToHalf'" link errors).
    cl_half h_one  = static_cast<cl_half>(nnopt_f32_to_f16(1.0f));
    cl_half h_zero = static_cast<cl_half>(nnopt_f32_to_f16(0.0f));
    auto status = clblast::Gemm<cl_half>(
        clblast::Layout::kRowMajor,
        clblast::Transpose::kNo,
        clblast::Transpose::kYes,
        M, N, K,
        h_one,
        x, 0, K,
        W, 0, K,
        h_zero,
        out, 0, N,
        &queue,
        nullptr);
#else
    auto status = clblast::Gemm<float>(
        clblast::Layout::kRowMajor,
        clblast::Transpose::kNo,
        clblast::Transpose::kYes,
        M, N, K,
        1.0f,
        x, 0, K,
        W, 0, K,
        0.0f,
        out, 0, N,
        &queue,
        nullptr);
#endif
    if (status != clblast::StatusCode::kSuccess) {
        NNOPT_ERROR_FMT("pytorch_linear: CLBlast Gemm failed status=%d (M=%d N=%d K=%d)",
                        (int)status, M, N, K);
        return false;
    }
    // SYNC-01: queue is in-order. Removing this clFinish (downstream kernels
    // see the GEMM output without explicit sync) is worth ~30% of decode
    // throughput on a 30-layer transformer at M=1 (measured 0.40 -> 1.71
    // tok/s on SmolLM2-135M, of which ~half came from removing this site
    // alone — pytorch_linear is called 7 times per layer per token).
    NNOPT_DEBUG_SYNC(queue);
    return true;
}

// Fused projection + residual add. On the M=1 image path, this saves the
// element_add_inplace launch that would otherwise follow pytorch_linear.
// Returns true on success (the fused kernel ran). Returns false if the
// fast path predicate doesn't hold (M!=1, K not in {896,4864}, no4
// disabled by N, or image creation failed) — caller must fall back to
// pytorch_linear + element_add_inplace.
// Public façade for the kv_write kernel (recordable replacement for
// clEnqueueCopyBuffer of K/V into the KV cache). Returns true on success.
// On miss (kernel not built), falls back false so caller can use copy.
bool kv_write_kernel(cl_command_queue queue,
                     cl_mem src, cl_mem cache,
                     cl_mem counter, int kv_dim) {
    if (!ensure_gemv_m1(queue)) return false;
    GemvM1State& s = gemv_state();
    if (!s.kernel_kv_write) return false;
    cl_int err = CL_SUCCESS;
    err = clSetKernelArg(s.kernel_kv_write, 0, sizeof(cl_mem), &src);     if (err != CL_SUCCESS) return false;
    err = clSetKernelArg(s.kernel_kv_write, 1, sizeof(cl_mem), &cache);   if (err != CL_SUCCESS) return false;
    err = clSetKernelArg(s.kernel_kv_write, 2, sizeof(cl_mem), &counter); if (err != CL_SUCCESS) return false;
    err = clSetKernelArg(s.kernel_kv_write, 3, sizeof(int),    &kv_dim);  if (err != CL_SUCCESS) return false;
    size_t lws = 64;
    size_t gws = ((kv_dim + lws - 1) / lws) * lws;
    err = nnopt_prof::enqueue(queue, s.kernel_kv_write, 1, nullptr, &gws, &lws, 0, nullptr, nullptr);
    if (err != CL_SUCCESS) {
        NNOPT_ERROR_FMT("kv_write_kernel: enqueue failed: %d", err);
        return false;
    }
    return true;
}

// Returns the cl_kernel for kv_write (used by recording integration to
// build per-replay arg overrides for start_pos).
cl_kernel get_kv_write_kernel(cl_command_queue queue) {
    if (!ensure_gemv_m1(queue)) return nullptr;
    return gemv_state().kernel_kv_write;
}

bool pytorch_linear_add(cl_command_queue queue,
                        int M, int N, int K,
                        cl_mem x, cl_mem W, cl_mem residual_inout) {
    // RUNTIME-DISABLED on Adreno 620 — measured 27-38% regression vs the
    // unfused two-launch path (8.32 → 5-6 tok/s). The fp16 read-modify-write
    // at the kernel epilogue (vload_half4 of `out`, add the partials, store
    // back) appears to introduce buffer-cache coherence overhead on Adreno
    // that exceeds the ~50µs/launch saved by collapsing residual_add into
    // o_proj/down_proj.
    //
    // Counterintuitively, the act of *defining* gemv_m1_kK_no4_img_add in
    // gemv_m1.cl (without dispatching it) drove the Adreno OpenCL compiler
    // to produce a faster compile of the OTHER (live) kernels — measured
    // +8.7% (8.32 → 9.04 tok/s) at Step 10b. So the kernels stay in source;
    // only the runtime dispatch is gated off here.
    //
    // If retrying on a different mobile GPU (Mali, PowerVR, Apple), flip
    // this back to `try_gemv_m1_add(...)` and re-measure.
    (void)queue; (void)M; (void)N; (void)K; (void)x; (void)W; (void)residual_inout;
    return false;
}

bool pytorch_conv1d(cl_command_queue queue,
                    int M, int N, int K,
                    cl_mem x, cl_mem W, cl_mem out) {
    // out[M, N] = x[M, K] @ W[K, N]  where W is HF Conv1D weight [K, N] = [in, out].
    //
    // HF Conv1D forward (transformers.pytorch_utils.Conv1D.forward):
    //   y = x @ self.weight + self.bias
    // No transpose. Weight is allocated as nn.Parameter(torch.empty(in, out)).
    // This is the OPPOSITE of nn.Linear, which stores [out, in] and forwards
    // as x @ W^T. The two GEMM wrappers differ by exactly one transpose flag
    // and one leading-dim convention — pick correctly per layer contract's
    // weight_key_parent_classes field.
    //
    // CLBlast RowMajor GEMM signature:
    //   C[M,N] = alpha * op(A)[M,K] * op(B)[K,N] + beta * C[M,N]
    // With TransposeA=kNo, TransposeB=kNo, op(B) reads W[K,N] directly.
    //
    // Leading dimensions for RowMajor:
    //   lda = K (A's stride between rows of A[M,K])
    //   ldb = N (B's stride between rows of W[K,N])  ← differs from pytorch_linear
    //   ldc = N (C's stride between rows of C[M,N])
#ifdef NNOPT_USE_FP16
    cl_half h_one  = static_cast<cl_half>(nnopt_f32_to_f16(1.0f));
    cl_half h_zero = static_cast<cl_half>(nnopt_f32_to_f16(0.0f));
    auto status = clblast::Gemm<cl_half>(
        clblast::Layout::kRowMajor,
        clblast::Transpose::kNo,
        clblast::Transpose::kNo,
        M, N, K,
        h_one,
        x, 0, K,
        W, 0, N,
        h_zero,
        out, 0, N,
        &queue,
        nullptr);
#else
    auto status = clblast::Gemm<float>(
        clblast::Layout::kRowMajor,
        clblast::Transpose::kNo,
        clblast::Transpose::kNo,
        M, N, K,
        1.0f,
        x, 0, K,
        W, 0, N,
        0.0f,
        out, 0, N,
        &queue,
        nullptr);
#endif
    if (status != clblast::StatusCode::kSuccess) {
        NNOPT_ERROR_FMT("pytorch_conv1d: CLBlast Gemm failed status=%d (M=%d N=%d K=%d)",
                        (int)status, M, N, K);
        return false;
    }
    NNOPT_DEBUG_SYNC(queue);
    return true;
}
