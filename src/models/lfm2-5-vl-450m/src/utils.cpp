#include "utils.h"
#include "debug_utils.h"   // NNOPT_ERROR_FMT — used by element_add / pytorch_linear / etc.
#include "profiler.h"      // KernelProfiler::event_for — dormant unless NNOPT_PROFILE=1.

#include <clblast.h>       // clblast::Gemm — used by pytorch_linear (dtype-templated dispatch).

#include <fstream>
#include <cstring>
#include <algorithm>
#include <numeric>
#include <cstdint>
#include <cstdio>
#include <unordered_map>
#include <vector>

// ─────────────────────────────────────────────────────────────────────────────
// Image2d-backed GEMV for the M==1 (decode) hot path.
//
// Ported from lfm2-5-350m: reads W through Adreno's texture cache (L1)
// instead of the buffer cache (L2). Measured on Razr 2020:
//   buffer streaming: 7.85 GB/s
//   image  streaming: 13.46 GB/s   <- 1.71x faster
//
// Layout: W[N,K] fp16 wrapped as image2d_t with CL_RGBA / CL_HALF_FLOAT
// (4 fp16 per pixel). image_width = K/4, image_height = N. Same backing
// memory as the buffer (cl_khr_image2d_from_buffer).
//
// The image kernels are compiled in a SEPARATE cl_program from lfm2_ops.cl
// to avoid register allocation interference (per ARTICLE.md finding).
// ─────────────────────────────────────────────────────────────────────────────
#ifdef NNOPT_USE_FP16

// Image GEMV program + kernel handles (compiled from kernels/gemv_m1_image.cl).
static cl_program s_gemv_m1_img_prog       = nullptr;
static cl_kernel  s_gemv_m1_k1024_no4_img  = nullptr;
static cl_kernel  s_gemv_m1_k1024_no8_img  = nullptr;
static cl_kernel  s_gemv_m1_k4608_no4_img  = nullptr;
static cl_program s_gemv_fused_img_prog        = nullptr;
static cl_kernel  s_gemv_fused3_k1024_no4_img = nullptr;
static cl_kernel  s_gemv_fused2_k1024_no4_img = nullptr;

// Per-buffer image2d_t view cache (lazy). Standard entries hold one image;
// tiled entries hold multiple sub-buffer/sub-image pairs for weights whose
// row count exceeds CL_DEVICE_IMAGE2D_MAX_HEIGHT (e.g. lm_head N=65536).
struct WImageTile { cl_mem sub_buffer = nullptr; cl_mem image = nullptr; int row_offset = 0; int row_count = 0; };
struct WImageEntry { cl_mem image = nullptr; std::vector<WImageTile> tiles; };
static std::unordered_map<cl_mem, WImageEntry> s_w_image_cache;
static std::unordered_map<cl_mem, char>        s_w_image_skip;
static bool   s_img_limits_known = false;
static size_t s_img_max_w = 0, s_img_max_h = 0;

static bool ensure_gemv_m1_image_program(cl_command_queue queue) {
    if (s_gemv_m1_img_prog) return true;

    cl_context   ctx    = nullptr;
    cl_device_id device = nullptr;
    clGetCommandQueueInfo(queue, CL_QUEUE_CONTEXT, sizeof(ctx),    &ctx,    nullptr);
    clGetCommandQueueInfo(queue, CL_QUEUE_DEVICE,  sizeof(device), &device, nullptr);
    if (!ctx || !device) {
        NNOPT_ERROR_FMT("gemv_m1_image: failed to get context/device%s", "");
        return false;
    }

    // Read kernels/gemv_m1_image.cl from cwd (build/deploy stages it next to
    // the binary on the device).
    std::ifstream f("kernels/gemv_m1_image.cl", std::ios::binary);
    if (!f.is_open()) {
        NNOPT_ERROR_FMT("gemv_m1_image: cannot open kernels/gemv_m1_image.cl%s", "");
        return false;
    }
    std::string src((std::istreambuf_iterator<char>(f)), std::istreambuf_iterator<char>());
    const char* src_cstr = src.c_str();
    size_t src_len = src.size();
    cl_int err;
    s_gemv_m1_img_prog = clCreateProgramWithSource(ctx, 1, &src_cstr, &src_len, &err);
    if (err != CL_SUCCESS) {
        NNOPT_ERROR_FMT("gemv_m1_image: clCreateProgramWithSource failed (%d)", (int)err);
        return false;
    }
    err = clBuildProgram(s_gemv_m1_img_prog, 1, &device, "-cl-fast-relaxed-math", nullptr, nullptr);
    if (err != CL_SUCCESS) {
        size_t log_size = 0;
        clGetProgramBuildInfo(s_gemv_m1_img_prog, device, CL_PROGRAM_BUILD_LOG, 0, nullptr, &log_size);
        if (log_size > 0) {
            std::vector<char> log(log_size + 1, 0);
            clGetProgramBuildInfo(s_gemv_m1_img_prog, device, CL_PROGRAM_BUILD_LOG, log_size, log.data(), nullptr);
            fprintf(stderr, "gemv_m1_image build log: %s\n", log.data());
        }
        clReleaseProgram(s_gemv_m1_img_prog);
        s_gemv_m1_img_prog = nullptr;
        return false;
    }
    // Image variants — optional (silently null on devices without image2d-from-buffer).
    s_gemv_m1_k1024_no4_img = clCreateKernel(s_gemv_m1_img_prog, "gemv_m1_k1024_no4_img", &err);
    if (err != CL_SUCCESS) { s_gemv_m1_k1024_no4_img = nullptr; }
    s_gemv_m1_k1024_no8_img = clCreateKernel(s_gemv_m1_img_prog, "gemv_m1_k1024_no8_img", &err);
    if (err != CL_SUCCESS) { s_gemv_m1_k1024_no8_img = nullptr; }
    s_gemv_m1_k4608_no4_img = clCreateKernel(s_gemv_m1_img_prog, "gemv_m1_k4608_no4_img", &err);
    if (err != CL_SUCCESS) { s_gemv_m1_k4608_no4_img = nullptr; }
    return s_gemv_m1_k1024_no4_img || s_gemv_m1_k1024_no8_img || s_gemv_m1_k4608_no4_img;
}

static bool ensure_gemv_fused_image_program(cl_command_queue queue) {
    if (s_gemv_fused_img_prog) return true;
    cl_context ctx = nullptr; cl_device_id dev = nullptr;
    clGetCommandQueueInfo(queue, CL_QUEUE_CONTEXT, sizeof(ctx), &ctx, nullptr);
    clGetCommandQueueInfo(queue, CL_QUEUE_DEVICE, sizeof(dev), &dev, nullptr);
    if (!ctx || !dev) return false;
    std::ifstream ff("kernels/gemv_fused_image.cl", std::ios::binary);
    if (!ff.is_open()) return false;
    std::string src((std::istreambuf_iterator<char>(ff)), std::istreambuf_iterator<char>());
    const char* s = src.c_str(); size_t len = src.size();
    cl_int err;
    s_gemv_fused_img_prog = clCreateProgramWithSource(ctx, 1, &s, &len, &err);
    if (err != CL_SUCCESS) return false;
    err = clBuildProgram(s_gemv_fused_img_prog, 1, &dev, "-cl-fast-relaxed-math", nullptr, nullptr);
    if (err != CL_SUCCESS) {
        size_t log_sz = 0;
        clGetProgramBuildInfo(s_gemv_fused_img_prog, dev, CL_PROGRAM_BUILD_LOG, 0, nullptr, &log_sz);
        if (log_sz > 0) { std::vector<char> log(log_sz+1, 0); clGetProgramBuildInfo(s_gemv_fused_img_prog, dev, CL_PROGRAM_BUILD_LOG, log_sz, log.data(), nullptr); fprintf(stderr, "gemv_fused_image build: %s\n", log.data()); }
        clReleaseProgram(s_gemv_fused_img_prog); s_gemv_fused_img_prog = nullptr; return false;
    }
    s_gemv_fused3_k1024_no4_img = clCreateKernel(s_gemv_fused_img_prog, "gemv_fused3_k1024_no4_img", &err);
    if (err != CL_SUCCESS) s_gemv_fused3_k1024_no4_img = nullptr;
    s_gemv_fused2_k1024_no4_img = clCreateKernel(s_gemv_fused_img_prog, "gemv_fused2_k1024_no4_img", &err);
    if (err != CL_SUCCESS) s_gemv_fused2_k1024_no4_img = nullptr;
    return s_gemv_fused3_k1024_no4_img || s_gemv_fused2_k1024_no4_img;
}

// Look up (or create on first use) an image2d_t view of fp16 weight buffer W
// shaped [N, K]. Standard layout: single image (K/4 wide, N tall). Tiled
// fallback when N > CL_DEVICE_IMAGE2D_MAX_HEIGHT -- used for lm_head N=65536
// on Adreno 620 (max image height = 16384).
static const WImageEntry* get_or_create_w_image(cl_command_queue queue, cl_mem W, int N, int K) {
    static const WImageEntry kEmpty;
    auto it = s_w_image_cache.find(W);
    if (it != s_w_image_cache.end()) return &it->second;
    if (s_w_image_skip.count(W)) return &kEmpty;

    cl_context   ctx = nullptr;
    cl_device_id dev = nullptr;
    if (clGetCommandQueueInfo(queue, CL_QUEUE_CONTEXT, sizeof(ctx), &ctx, nullptr) != CL_SUCCESS ||
        clGetCommandQueueInfo(queue, CL_QUEUE_DEVICE,  sizeof(dev), &dev, nullptr) != CL_SUCCESS) {
        s_w_image_skip[W] = 1; return &kEmpty;
    }
    if (!s_img_limits_known) {
        clGetDeviceInfo(dev, CL_DEVICE_IMAGE2D_MAX_WIDTH,  sizeof(s_img_max_w), &s_img_max_w, nullptr);
        clGetDeviceInfo(dev, CL_DEVICE_IMAGE2D_MAX_HEIGHT, sizeof(s_img_max_h), &s_img_max_h, nullptr);
        s_img_limits_known = true;
        if (const char* d = std::getenv("NNOPT_DEBUG_LAYERS"); d && d[0] != '0') {
            fprintf(stderr, "Adreno image2d limits: max_w=%zu max_h=%zu\n", s_img_max_w, s_img_max_h);
        }
    }

    cl_image_format fmt; fmt.image_channel_order = CL_RGBA; fmt.image_channel_data_type = CL_HALF_FLOAT;
    auto try_create = [&](cl_mem buf, size_t pix_w, size_t pix_h) -> cl_mem {
        if (pix_w == 0 || pix_h == 0) return nullptr;
        if (pix_w > s_img_max_w || pix_h > s_img_max_h) return nullptr;
        cl_image_desc desc; std::memset(&desc, 0, sizeof(desc));
        desc.image_type = CL_MEM_OBJECT_IMAGE2D;
        desc.image_width  = pix_w;
        desc.image_height = pix_h;
        desc.buffer = buf;
        cl_int e = CL_SUCCESS;
        cl_mem img = clCreateImage(ctx, CL_MEM_READ_ONLY, &fmt, &desc, nullptr, &e);
        if (e != CL_SUCCESS || !img) return nullptr;
        return img;
    };

    // Standard layout -- single image, fits in one shot.
    if (cl_mem img = try_create(W, (size_t)(K / 4), (size_t)N)) {
        WImageEntry e; e.image = img;
        s_w_image_cache[W] = e; return &s_w_image_cache[W];
    }

    // Tiled fallback for lm_head (N=65536 > 16384).
    if (K > 0 && (size_t)(K / 4) <= s_img_max_w) {
        const int TILE_H = (int)s_img_max_h;
        const int row_bytes = K * 2;  // fp16: 2 bytes per element
        if (row_bytes > 0 && (row_bytes % 128) == 0 && N > 0) {
            std::vector<WImageTile> tiles;
            int rows_left = N, row_off = 0;
            bool ok = true;
            while (rows_left > 0) {
                int tile_n = rows_left < TILE_H ? rows_left : TILE_H;
                cl_buffer_region region{ (size_t)row_off * (size_t)row_bytes, (size_t)tile_n * (size_t)row_bytes };
                cl_int e = CL_SUCCESS;
                cl_mem sub = clCreateSubBuffer(W, CL_MEM_READ_ONLY, CL_BUFFER_CREATE_TYPE_REGION, &region, &e);
                if (e != CL_SUCCESS || !sub) { ok = false; break; }
                cl_mem sub_img = try_create(sub, (size_t)(K / 4), (size_t)tile_n);
                if (!sub_img) { clReleaseMemObject(sub); ok = false; break; }
                tiles.push_back({sub, sub_img, row_off, tile_n});
                row_off += tile_n; rows_left -= tile_n;
            }
            if (ok && !tiles.empty()) {
                WImageEntry e; e.tiles = std::move(tiles);
                s_w_image_cache[W] = e; return &s_w_image_cache[W];
            }
            for (auto& t : tiles) { if (t.image) clReleaseMemObject(t.image); if (t.sub_buffer) clReleaseMemObject(t.sub_buffer); }
        }
    }
    s_w_image_skip[W] = 1;
    return &kEmpty;
}

// Image-backed dispatch. Returns false on miss/failure (caller falls through).
static bool run_gemv_m1_image(cl_command_queue queue, int N, int K, cl_mem W, cl_mem x, cl_mem out) {
    if (!ensure_gemv_m1_image_program(queue)) return false;
    cl_kernel k = nullptr;
    int stride = 4;  // outputs per WG
    // no8 wins for large N (more arithmetic density per thread hides texture
    // latency better) but loses for small N where the thread count drops
    // below the device's latency-hiding threshold. Profile (Adreno 620):
    //   N=4608/3072/65536-tile-16384: +10-19% with no8
    //   N=1024: -2%, N=512: -11% (fall back to no4).
    if (K == 1024 && N >= 2048 && (N % 8) == 0 && s_gemv_m1_k1024_no8_img) {
        k = s_gemv_m1_k1024_no8_img; stride = 8;
    } else if (K == 1024 && (N % 4) == 0 && s_gemv_m1_k1024_no4_img) {
        k = s_gemv_m1_k1024_no4_img; stride = 4;
    } else if (K == 4608 && (N % 4) == 0 && s_gemv_m1_k4608_no4_img) {
        k = s_gemv_m1_k4608_no4_img; stride = 4;
    } else {
        return false;
    }

    const WImageEntry* ent = get_or_create_w_image(queue, W, N, K);
    if (!ent || (!ent->image && ent->tiles.empty())) return false;

    char lbl[64]; snprintf(lbl, sizeof(lbl), "gemv_m1_K%d_N%d_no%d_img", K, N, stride);
    cl_int err = CL_SUCCESS;

    // Single-image path.
    if (ent->image) {
        clSetKernelArg(k, 0, sizeof(cl_mem), &x);
        clSetKernelArg(k, 1, sizeof(cl_mem), &ent->image);
        clSetKernelArg(k, 2, sizeof(cl_mem), &out);
        clSetKernelArg(k, 3, sizeof(int),    &N);
        const size_t WG = 64;
        size_t gws = (size_t)(N / stride) * WG;
        size_t lws = WG;
        cl_event* evt = KernelProfiler::event_for(lbl);
        err = clEnqueueNDRangeKernel(queue, k, 1, nullptr, &gws, &lws, 0, nullptr, evt);
        if (err != CL_SUCCESS) { NNOPT_ERROR_FMT("gemv_m1_image enqueue: %d (K=%d N=%d)", err, K, N); return false; }
        return true;
    }

    // Tiled path: dispatch once per tile, writing into out at the tile's row offset.
    // Output row size for fp16 logits is 2 bytes -- sub-buffer offsets must be 128 B aligned.
    // tile.row_offset x 2 bytes -- need (row_offset x 2) % 128 == 0 => row_offset % 64 == 0.
    // CL_DEVICE_IMAGE2D_MAX_HEIGHT is 16384 on Adreno 620 (multiple of 64) -- natural alignment.
    for (const auto& t : ent->tiles) {
        cl_buffer_region region{ (size_t)t.row_offset * sizeof(nnopt_storage_t), (size_t)t.row_count * sizeof(nnopt_storage_t) };
        cl_mem out_sub = clCreateSubBuffer(out, CL_MEM_READ_WRITE, CL_BUFFER_CREATE_TYPE_REGION, &region, &err);
        if (err != CL_SUCCESS) { NNOPT_ERROR_FMT("tile out_sub create: %d", err); return false; }

        clSetKernelArg(k, 0, sizeof(cl_mem), &x);
        clSetKernelArg(k, 1, sizeof(cl_mem), &t.image);
        clSetKernelArg(k, 2, sizeof(cl_mem), &out_sub);
        int tile_n = t.row_count;
        clSetKernelArg(k, 3, sizeof(int), &tile_n);
        const size_t WG = 64;
        size_t gws = (size_t)(tile_n / stride) * WG;
        size_t lws = WG;
        cl_event* evt = KernelProfiler::event_for(lbl);
        err = clEnqueueNDRangeKernel(queue, k, 1, nullptr, &gws, &lws, 0, nullptr, evt);
        clReleaseMemObject(out_sub);
        if (err != CL_SUCCESS) { NNOPT_ERROR_FMT("tile dispatch: %d", err); return false; }
    }
    return true;
}

// Fused 3-output M=1 GEMV (Q+K+V or B+C+X). K=1024 only, all N must be %4==0.
static bool run_gemv_fused3_image(cl_command_queue queue, int K,
                                  cl_mem Wa, int Na, cl_mem Wb, int Nb, cl_mem Wc, int Nc,
                                  cl_mem x, cl_mem ya, cl_mem yb, cl_mem yc) {
    if (K != 1024) return false;
    if ((Na % 4) || (Nb % 4) || (Nc % 4)) return false;
    if (!ensure_gemv_fused_image_program(queue)) return false;
    if (!ensure_gemv_m1_image_program(queue)) return false;
    if (!s_gemv_fused3_k1024_no4_img) return false;
    const WImageEntry* ea = get_or_create_w_image(queue, Wa, Na, K);
    const WImageEntry* eb = get_or_create_w_image(queue, Wb, Nb, K);
    const WImageEntry* ec = get_or_create_w_image(queue, Wc, Nc, K);
    if (!ea || !ea->image || !eb || !eb->image || !ec || !ec->image) return false;

    cl_kernel k = s_gemv_fused3_k1024_no4_img;
    clSetKernelArg(k, 0, sizeof(cl_mem), &x);
    clSetKernelArg(k, 1, sizeof(cl_mem), &ea->image);
    clSetKernelArg(k, 2, sizeof(cl_mem), &eb->image);
    clSetKernelArg(k, 3, sizeof(cl_mem), &ec->image);
    clSetKernelArg(k, 4, sizeof(cl_mem), &ya);
    clSetKernelArg(k, 5, sizeof(cl_mem), &yb);
    clSetKernelArg(k, 6, sizeof(cl_mem), &yc);
    clSetKernelArg(k, 7, sizeof(int), &Na);
    clSetKernelArg(k, 8, sizeof(int), &Nb);
    clSetKernelArg(k, 9, sizeof(int), &Nc);
    size_t gws = (size_t)((Na + Nb + Nc) / 4) * 64;
    size_t lws = 64;
    cl_int err = clEnqueueNDRangeKernel(queue, k, 1, nullptr, &gws, &lws, 0, nullptr,
                                         KernelProfiler::event_for("gemv_fused3_img"));
    return err == CL_SUCCESS;
}

// Fused 2-output M=1 GEMV (w1+w3). K=1024 only, both N must be %4==0.
static bool run_gemv_fused2_image(cl_command_queue queue, int K,
                                  cl_mem Wa, int Na, cl_mem Wb, int Nb,
                                  cl_mem x, cl_mem ya, cl_mem yb) {
    if (K != 1024) return false;
    if ((Na % 4) || (Nb % 4)) return false;
    if (!ensure_gemv_fused_image_program(queue)) return false;
    if (!ensure_gemv_m1_image_program(queue)) return false;
    if (!s_gemv_fused2_k1024_no4_img) return false;
    const WImageEntry* ea = get_or_create_w_image(queue, Wa, Na, K);
    const WImageEntry* eb = get_or_create_w_image(queue, Wb, Nb, K);
    if (!ea || !ea->image || !eb || !eb->image) return false;

    cl_kernel k = s_gemv_fused2_k1024_no4_img;
    clSetKernelArg(k, 0, sizeof(cl_mem), &x);
    clSetKernelArg(k, 1, sizeof(cl_mem), &ea->image);
    clSetKernelArg(k, 2, sizeof(cl_mem), &eb->image);
    clSetKernelArg(k, 3, sizeof(cl_mem), &ya);
    clSetKernelArg(k, 4, sizeof(cl_mem), &yb);
    clSetKernelArg(k, 5, sizeof(int), &Na);
    clSetKernelArg(k, 6, sizeof(int), &Nb);
    size_t gws = (size_t)((Na + Nb) / 4) * 64;
    size_t lws = 64;
    cl_int err = clEnqueueNDRangeKernel(queue, k, 1, nullptr, &gws, &lws, 0, nullptr,
                                         KernelProfiler::event_for("gemv_fused2_img"));
    return err == CL_SUCCESS;
}

// ─────────────────────────────────────────────────────────────────────────────
// int8 — image2d-backed GEMV for the M==1 (decode) hot path.
//
// Ported from lfm2-5-350m/src/utils.cpp `// ── int8 ──` sections. Halves the
// weight memory footprint vs fp16, and reads int8 weights through the texture
// L1 cache (CL_RGBA / CL_SIGNED_INT8). Per-row fp16 scale folded in at the
// reduction tail. The kernels live in kernels/gemv_m1_int8_image.cl and are
// compiled in a SEPARATE cl_program from the fp16 image kernels — register
// allocation interference (ARTICLE.md finding).
//
// Eligibility:
//   - W registered via nnopt_register_int8_weight (main.cpp walks `.scales`).
//   - K is 1024 or 4608 (the LFM2 backbone projection dims).
//   - N is divisible by 4 (or 8 for the no8 K=1024 fast path).
// On miss/failure the dispatcher returns false and pytorch_linear falls
// through to the fp16 image path / CLBlast.
//
// M>1 (prefill / SigLIP) is NOT served by these kernels — CLBlast cannot
// consume int8 buffers directly. Instead pytorch_linear dequantizes the int8
// weight into a cached fp16 scratch buffer and runs the existing HGemm.
// ─────────────────────────────────────────────────────────────────────────────

static cl_program s_gemv_m1_int8_prog            = nullptr;
static cl_kernel  s_gemv_m1_k1024_no4_img_int8   = nullptr;
static cl_kernel  s_gemv_m1_k1024_no8_img_int8   = nullptr;
static cl_kernel  s_gemv_m1_k4608_no4_img_int8   = nullptr;

// Per-W aux: the int8 weight buffer, its fp16 scale buffer, dims, and the
// image2d view (built lazily on first dispatch). Tiled path is unused here
// since none of the int8 weights in this port exceed CL_DEVICE_IMAGE2D_MAX_HEIGHT
// (max N = 65536 lm_head → 16384 rows; will tile via the same shape as fp16 path
// if needed).
struct Int8Tile {
    cl_mem sub_W     = nullptr;
    cl_mem image     = nullptr;
    cl_mem scale_sub = nullptr;
    int    row_offset = 0;
    int    row_count  = 0;
};
struct Int8Aux {
    cl_mem image = nullptr;          // single-image path (nullptr → tiled or pending)
    cl_mem scale = nullptr;          // single-image path: scale buffer to bind
    std::vector<Int8Tile> tiles;     // tiled path (N > image-height cap)
    int N = 0;
    int K = 0;
    cl_mem W_int8     = nullptr;     // root W buffer, kept for sub-buffer creation
    cl_mem scale_root = nullptr;     // root scale buffer, for sub-buffer creation
};
static std::unordered_map<cl_mem, Int8Aux> s_int8_aux;

static bool ensure_gemv_m1_int8_program(cl_command_queue queue) {
    if (s_gemv_m1_int8_prog) return true;

    cl_context   ctx    = nullptr;
    cl_device_id device = nullptr;
    clGetCommandQueueInfo(queue, CL_QUEUE_CONTEXT, sizeof(ctx),    &ctx,    nullptr);
    clGetCommandQueueInfo(queue, CL_QUEUE_DEVICE,  sizeof(device), &device, nullptr);
    if (!ctx || !device) return false;

    std::ifstream f("kernels/gemv_m1_int8_image.cl", std::ios::binary);
    if (!f.is_open()) {
        NNOPT_ERROR_FMT("gemv_m1_int8: cannot open kernels/gemv_m1_int8_image.cl%s", "");
        return false;
    }
    std::string src((std::istreambuf_iterator<char>(f)), std::istreambuf_iterator<char>());
    const char* src_cstr = src.c_str();
    size_t src_len = src.size();
    cl_int err;
    s_gemv_m1_int8_prog = clCreateProgramWithSource(ctx, 1, &src_cstr, &src_len, &err);
    if (err != CL_SUCCESS) {
        NNOPT_ERROR_FMT("gemv_m1_int8: clCreateProgramWithSource failed (%d)", (int)err);
        return false;
    }
    err = clBuildProgram(s_gemv_m1_int8_prog, 1, &device, "-cl-fast-relaxed-math", nullptr, nullptr);
    if (err != CL_SUCCESS) {
        size_t log_size = 0;
        clGetProgramBuildInfo(s_gemv_m1_int8_prog, device, CL_PROGRAM_BUILD_LOG, 0, nullptr, &log_size);
        if (log_size > 0) {
            std::vector<char> log(log_size + 1, 0);
            clGetProgramBuildInfo(s_gemv_m1_int8_prog, device, CL_PROGRAM_BUILD_LOG, log_size, log.data(), nullptr);
            fprintf(stderr, "gemv_m1_int8 build log: %s\n", log.data());
        }
        clReleaseProgram(s_gemv_m1_int8_prog);
        s_gemv_m1_int8_prog = nullptr;
        return false;
    }
    s_gemv_m1_k1024_no4_img_int8 = clCreateKernel(s_gemv_m1_int8_prog, "gemv_m1_k1024_no4_img_int8", &err);
    if (err != CL_SUCCESS) { s_gemv_m1_k1024_no4_img_int8 = nullptr; }
    s_gemv_m1_k1024_no8_img_int8 = clCreateKernel(s_gemv_m1_int8_prog, "gemv_m1_k1024_no8_img_int8", &err);
    if (err != CL_SUCCESS) { s_gemv_m1_k1024_no8_img_int8 = nullptr; }
    s_gemv_m1_k4608_no4_img_int8 = clCreateKernel(s_gemv_m1_int8_prog, "gemv_m1_k4608_no4_img_int8", &err);
    if (err != CL_SUCCESS) { s_gemv_m1_k4608_no4_img_int8 = nullptr; }
    return true;
}

// Called from main.cpp after Weights::load(). Records {W, scale, N, K}; the
// image2d view is built lazily on first dispatch (needs a queue).
extern "C" bool nnopt_register_int8_weight(cl_mem W_int8, cl_mem scale_fp16, int N, int K) {
    if (!W_int8 || !scale_fp16 || N <= 0 || K <= 0) return false;
    Int8Aux a;
    a.image = nullptr;
    a.scale = nullptr;
    a.N = N;
    a.K = K;
    a.W_int8     = W_int8;
    a.scale_root = scale_fp16;
    s_int8_aux[W_int8] = a;
    return true;
}

static bool prepare_int8_entry(cl_command_queue queue, cl_mem W) {
    auto it = s_int8_aux.find(W);
    if (it == s_int8_aux.end()) return false;
    Int8Aux& a = it->second;
    if (a.image || !a.tiles.empty()) return true;  // already prepared

    cl_context   ctx = nullptr;
    cl_device_id dev = nullptr;
    if (clGetCommandQueueInfo(queue, CL_QUEUE_CONTEXT, sizeof(ctx), &ctx, nullptr) != CL_SUCCESS ||
        clGetCommandQueueInfo(queue, CL_QUEUE_DEVICE,  sizeof(dev), &dev, nullptr) != CL_SUCCESS) return false;

    if (!s_img_limits_known) {
        clGetDeviceInfo(dev, CL_DEVICE_IMAGE2D_MAX_WIDTH,  sizeof(s_img_max_w), &s_img_max_w, nullptr);
        clGetDeviceInfo(dev, CL_DEVICE_IMAGE2D_MAX_HEIGHT, sizeof(s_img_max_h), &s_img_max_h, nullptr);
        s_img_limits_known = true;
    }
    if ((size_t)(a.K / 4) > s_img_max_w) {
        NNOPT_ERROR_FMT("int8 image too wide: K/4=%d > %zu", a.K/4, s_img_max_w);
        return false;
    }

    cl_image_format fmt;
    fmt.image_channel_order = CL_RGBA;
    fmt.image_channel_data_type = CL_SIGNED_INT8;

    if ((size_t)a.N <= s_img_max_h) {
        cl_image_desc desc; std::memset(&desc, 0, sizeof(desc));
        desc.image_type   = CL_MEM_OBJECT_IMAGE2D;
        desc.image_width  = (size_t)(a.K / 4);
        desc.image_height = (size_t)a.N;
        desc.buffer       = W;
        cl_int err = CL_SUCCESS;
        cl_mem img = clCreateImage(ctx, CL_MEM_READ_ONLY, &fmt, &desc, nullptr, &err);
        if (err != CL_SUCCESS || !img) {
            NNOPT_ERROR_FMT("int8 clCreateImage failed (%d) N=%d K=%d", (int)err, a.N, a.K);
            return false;
        }
        a.image = img;
        a.scale = a.scale_root;
        return true;
    }

    // Tiled path: split N into chunks of s_img_max_h rows.
    // W_int8 row stride = K bytes; aligned if K%128==0 (K=1024 ✓).
    // Scale row stride  = 2 bytes; aligned if row_offset%64==0
    //                    (TILE_H = 16384 on Adreno 620 → multiple of 64 ✓).
    const int TILE_H = (int)s_img_max_h;
    int rows_left = a.N, row_off = 0;
    while (rows_left > 0) {
        const int tile_n = rows_left < TILE_H ? rows_left : TILE_H;
        Int8Tile t; t.row_offset = row_off; t.row_count = tile_n;

        cl_buffer_region w_region  { (size_t)row_off * (size_t)a.K,             (size_t)tile_n * (size_t)a.K };
        cl_buffer_region s_region  { (size_t)row_off * sizeof(nnopt_storage_t), (size_t)tile_n * sizeof(nnopt_storage_t) };
        cl_int err = CL_SUCCESS;
        t.sub_W     = clCreateSubBuffer(W,              CL_MEM_READ_ONLY, CL_BUFFER_CREATE_TYPE_REGION, &w_region, &err);
        if (err != CL_SUCCESS || !t.sub_W)     { NNOPT_ERROR_FMT("int8 tile sub_W create: %d row=%d", err, row_off); return false; }
        t.scale_sub = clCreateSubBuffer(a.scale_root,   CL_MEM_READ_ONLY, CL_BUFFER_CREATE_TYPE_REGION, &s_region, &err);
        if (err != CL_SUCCESS || !t.scale_sub) { NNOPT_ERROR_FMT("int8 tile scale_sub create: %d row=%d", err, row_off); return false; }

        cl_image_desc desc; std::memset(&desc, 0, sizeof(desc));
        desc.image_type   = CL_MEM_OBJECT_IMAGE2D;
        desc.image_width  = (size_t)(a.K / 4);
        desc.image_height = (size_t)tile_n;
        desc.buffer       = t.sub_W;
        t.image = clCreateImage(ctx, CL_MEM_READ_ONLY, &fmt, &desc, nullptr, &err);
        if (err != CL_SUCCESS || !t.image)     { NNOPT_ERROR_FMT("int8 tile clCreateImage: %d row=%d", err, row_off); return false; }

        a.tiles.push_back(t);
        row_off   += tile_n;
        rows_left -= tile_n;
    }
    return true;
}

static bool run_gemv_m1_image_int8(cl_command_queue queue, int N, int K, cl_mem W, cl_mem x, cl_mem out) {
    auto it = s_int8_aux.find(W);
    if (it == s_int8_aux.end()) return false;
    if (!ensure_gemv_m1_int8_program(queue)) return false;
    if (!prepare_int8_entry(queue, W)) return false;

    auto pick_kernel = [&](int Nlocal) -> std::pair<cl_kernel,int> {
        if (K == 1024 && Nlocal >= 2048 && (Nlocal % 8) == 0 && s_gemv_m1_k1024_no8_img_int8) return {s_gemv_m1_k1024_no8_img_int8, 8};
        if (K == 1024 &&                   (Nlocal % 4) == 0 && s_gemv_m1_k1024_no4_img_int8) return {s_gemv_m1_k1024_no4_img_int8, 4};
        if (K == 4608 &&                   (Nlocal % 4) == 0 && s_gemv_m1_k4608_no4_img_int8) return {s_gemv_m1_k4608_no4_img_int8, 4};
        return {nullptr, 0};
    };

    Int8Aux& a = it->second;

    if (!a.tiles.empty()) {
        for (const auto& t : a.tiles) {
            auto [kt, stride_t] = pick_kernel(t.row_count);
            if (!kt) return false;
            cl_buffer_region out_region{ (size_t)t.row_offset * sizeof(nnopt_storage_t),
                                         (size_t)t.row_count  * sizeof(nnopt_storage_t) };
            cl_int err = CL_SUCCESS;
            cl_mem out_sub = clCreateSubBuffer(out, CL_MEM_READ_WRITE, CL_BUFFER_CREATE_TYPE_REGION, &out_region, &err);
            if (err != CL_SUCCESS || !out_sub) { NNOPT_ERROR_FMT("int8 tile out_sub: %d row=%d", err, t.row_offset); return false; }
            int tile_n = t.row_count;
            clSetKernelArg(kt, 0, sizeof(cl_mem), &x);
            clSetKernelArg(kt, 1, sizeof(cl_mem), &t.image);
            clSetKernelArg(kt, 2, sizeof(cl_mem), &t.scale_sub);
            clSetKernelArg(kt, 3, sizeof(cl_mem), &out_sub);
            clSetKernelArg(kt, 4, sizeof(int),    &tile_n);
            const size_t WG = 64;
            size_t gws = (size_t)(tile_n / stride_t) * WG;
            size_t lws = WG;
            char lbl[64]; snprintf(lbl, sizeof(lbl), "gemv_m1_K%d_N%d_no%d_img_int8_tile", K, tile_n, stride_t);
            cl_event* evt = KernelProfiler::event_for(lbl);
            err = clEnqueueNDRangeKernel(queue, kt, 1, nullptr, &gws, &lws, 0, nullptr, evt);
            clReleaseMemObject(out_sub);
            if (err != CL_SUCCESS) { NNOPT_ERROR_FMT("int8 tile enqueue: %d row=%d", err, t.row_offset); return false; }
        }
        return true;
    }

    auto [k, stride] = pick_kernel(N);
    if (!k || !a.image) return false;

    char lbl[64]; snprintf(lbl, sizeof(lbl), "gemv_m1_K%d_N%d_no%d_img_int8", K, N, stride);
    clSetKernelArg(k, 0, sizeof(cl_mem), &x);
    clSetKernelArg(k, 1, sizeof(cl_mem), &a.image);
    clSetKernelArg(k, 2, sizeof(cl_mem), &a.scale);
    clSetKernelArg(k, 3, sizeof(cl_mem), &out);
    clSetKernelArg(k, 4, sizeof(int),    &N);
    const size_t WG = 64;
    size_t gws = (size_t)(N / stride) * WG;
    size_t lws = WG;
    cl_event* evt = KernelProfiler::event_for(lbl);
    cl_int err = clEnqueueNDRangeKernel(queue, k, 1, nullptr, &gws, &lws, 0, nullptr, evt);
    if (err != CL_SUCCESS) {
        NNOPT_ERROR_FMT("gemv_m1_image_int8 enqueue: %d (K=%d N=%d)", err, K, N);
        return false;
    }
    return true;
}

// Has this W buffer been registered as int8?
static inline bool is_int8_registered(cl_mem W) {
    return s_int8_aux.find(W) != s_int8_aux.end();
}

// Defined in src/ops/lfm2_common.cpp — handles to the dequant kernel in
// kernels/lfm2_ops.cl (lazily clCreateKernel'd, process-lifetime).
extern "C" cl_kernel nnopt_get_dequant_int8_kernel(cl_command_queue queue);

// Materialize an int8 weight to a fresh fp16 cl_mem (caller takes ownership).
// Used by main.cpp at startup to unblock layers like op_lfm2_conv_block that
// split in_proj.weight into sub-buffers with hardcoded fp16 byte stride.
// Synchronous — blocks on clFinish so the returned buffer is immediately
// safe to bind to subsequent kernels on the same queue.
extern "C" cl_mem nnopt_dequant_int8_to_fp16_alloc(cl_command_queue queue,
                                                   cl_mem W_int8,
                                                   cl_mem scale_fp16,
                                                   int N, int K) {
    cl_context ctx = nullptr;
    clGetCommandQueueInfo(queue, CL_QUEUE_CONTEXT, sizeof(ctx), &ctx, nullptr);
    if (!ctx) return nullptr;
    cl_int err = CL_SUCCESS;
    cl_mem out = clCreateBuffer(ctx, CL_MEM_READ_WRITE,
                                (size_t)N * (size_t)K * sizeof(nnopt_storage_t),
                                nullptr, &err);
    if (err != CL_SUCCESS || !out) {
        NNOPT_ERROR_FMT("nnopt_dequant_int8_to_fp16_alloc: clCreateBuffer N=%d K=%d failed (%d)", N, K, (int)err);
        return nullptr;
    }
    cl_kernel dq = nnopt_get_dequant_int8_kernel(queue);
    if (!dq) { clReleaseMemObject(out); return nullptr; }
    cl_int e = CL_SUCCESS;
    e |= clSetKernelArg(dq, 0, sizeof(cl_mem), &W_int8);
    e |= clSetKernelArg(dq, 1, sizeof(cl_mem), &scale_fp16);
    e |= clSetKernelArg(dq, 2, sizeof(cl_mem), &out);
    e |= clSetKernelArg(dq, 3, sizeof(int),    &N);
    e |= clSetKernelArg(dq, 4, sizeof(int),    &K);
    if (e != CL_SUCCESS) {
        NNOPT_ERROR_FMT("nnopt_dequant_int8_to_fp16_alloc: setArg failed (%d)", (int)e);
        clReleaseMemObject(out);
        return nullptr;
    }
    size_t total = (size_t)N * (size_t)K;
    size_t lws = 256;
    while (lws > 1 && (total % lws) != 0) lws >>= 1;
    e = clEnqueueNDRangeKernel(queue, dq, 1, nullptr, &total, &lws, 0, nullptr, nullptr);
    if (e != CL_SUCCESS) {
        NNOPT_ERROR_FMT("nnopt_dequant_int8_to_fp16_alloc: enqueue failed (%d) N=%d K=%d", (int)e, N, K);
        clReleaseMemObject(out);
        return nullptr;
    }
    clFinish(queue);  // caller may bind immediately; ensure GPU has produced bytes.
    return out;
}

// ─────────────────────────────────────────────────────────────────────────────
// int8 — prefill (M>1) path: dequantize int8 W to fp16 scratch, then HGemm.
//
// CLBlast cannot consume int8 buffers directly. For M>1 calls (SigLIP encoder,
// LM prefill) we run kernels/lfm2_ops.cl::dequant_int8_to_fp16 once into a
// scratch fp16 buffer sized N*K, then dispatch the existing HGemm against
// that scratch. Scratch buffers are cached per (W cl_mem) so the cost is
// O(1 dequant per layer per forward), not per-call.
//
// The dequant kernel is `__kernel void dequant_int8_to_fp16(__global const
// char* w_int8, __global const half* scales, __global half* w_fp16, int N,
// int K)` and dispatches one WI per output element.
// ─────────────────────────────────────────────────────────────────────────────

// Cache: int8 W cl_mem → fp16 scratch cl_mem (lifetime = process). The scratch
// is dequant'd lazily on each forward; we currently dequant on every M>1 call
// (cheap relative to HGemm). A "scratch already valid this forward" bitmap
// would let us skip the redundant dequant — left for later if profile shows
// dequant overhead, since prefill is the dominant cost and dequant is ~1% of
// it. Per-shape sharing is unnecessary because each layer's weight is a
// distinct cl_mem.
struct DequantScratch {
    cl_mem fp16 = nullptr;
    int    N = 0;
    int    K = 0;
};
static std::unordered_map<cl_mem, DequantScratch> s_dequant_scratch;

static cl_mem dequant_int8_weight_to_fp16(cl_command_queue queue, cl_mem W_int8, cl_mem scale_fp16, int N, int K) {
    auto& sc = s_dequant_scratch[W_int8];
    cl_context ctx = nullptr;
    clGetCommandQueueInfo(queue, CL_QUEUE_CONTEXT, sizeof(ctx), &ctx, nullptr);
    if (!ctx) return nullptr;
    if (!sc.fp16) {
        cl_int err = CL_SUCCESS;
        sc.fp16 = clCreateBuffer(ctx, CL_MEM_READ_WRITE,
                                 (size_t)N * (size_t)K * sizeof(nnopt_storage_t),
                                 nullptr, &err);
        if (err != CL_SUCCESS || !sc.fp16) {
            NNOPT_ERROR_FMT("dequant scratch alloc failed: N=%d K=%d err=%d", N, K, (int)err);
            sc.fp16 = nullptr;
            return nullptr;
        }
        sc.N = N; sc.K = K;
    }
    cl_kernel dq = nnopt_get_dequant_int8_kernel(queue);
    if (!dq) return nullptr;
    cl_int e = CL_SUCCESS;
    e |= clSetKernelArg(dq, 0, sizeof(cl_mem), &W_int8);
    e |= clSetKernelArg(dq, 1, sizeof(cl_mem), &scale_fp16);
    e |= clSetKernelArg(dq, 2, sizeof(cl_mem), &sc.fp16);
    e |= clSetKernelArg(dq, 3, sizeof(int),    &N);
    e |= clSetKernelArg(dq, 4, sizeof(int),    &K);
    if (e != CL_SUCCESS) {
        NNOPT_ERROR_FMT("dequant_int8: setArg failed (%d)", (int)e);
        return nullptr;
    }
    size_t total = (size_t)N * (size_t)K;
    // Pick a local size that divides total; one-WI-per-element is the simplest
    // schedule and good enough since this runs once per layer per prefill.
    size_t lws = 256;
    while (lws > 1 && (total % lws) != 0) lws >>= 1;
    e = clEnqueueNDRangeKernel(queue, dq, 1, nullptr, &total, &lws, 0, nullptr,
                               KernelProfiler::event_for("dequant_int8_to_fp16"));
    if (e != CL_SUCCESS) {
        NNOPT_ERROR_FMT("dequant_int8: enqueue failed (%d) N=%d K=%d", (int)e, N, K);
        return nullptr;
    }
    return sc.fp16;
}

#endif // NNOPT_USE_FP16

// ──────────────────────────────────────────────
// set_arg_checked — public clSetKernelArg wrapper.
// One definition used by every op (src/ops/*.cpp). Returns false on failure
// after surfacing a descriptive NNOPT_ERROR_FMT (kernel name slot, arg index,
// CL error code). Model-agnostic.
// ──────────────────────────────────────────────
bool set_arg_checked(cl_kernel kernel,
                     unsigned int arg_index,
                     size_t arg_size,
                     const void* arg_value,
                     const char* arg_name) {
    const cl_int err = clSetKernelArg(kernel, arg_index, arg_size, arg_value);
    if (err != CL_SUCCESS) {
        NNOPT_ERROR_FMT("clSetKernelArg(%s) idx=%u failed: %d",
                        arg_name ? arg_name : "(unnamed)",
                        (unsigned)arg_index,
                        (int)err);
        return false;
    }
    return true;
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
    size_t gws = n;
    err = clEnqueueNDRangeKernel(queue, s_cached_kernel, 1, nullptr, &gws, nullptr, 0, nullptr,
                                 KernelProfiler::event_for("util_element_add_inplace"));
    if (err != CL_SUCCESS) {
        NNOPT_ERROR_FMT("element_add_inplace: clEnqueueNDRangeKernel failed (%d)", err);
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

    // Track — cached kernel handle (process-lifetime), 3-buffer fused add.
    // Was: clCreateBuffer + clEnqueueCopyBuffer(a→out) + clCreateKernel(element_add) +
    //      3× setArg + dispatch + clReleaseKernel  =  ~700µs of host work per call.
    // Now: clCreateBuffer + clCreateKernel-once + 4× setArg + dispatch — saves the
    // buffer copy step and the per-call kernel create/release. Called ~32× per decode
    // token (16 layers × 2 residuals).
    static cl_kernel g_element_add3_kernel = nullptr;
    if (!g_element_add3_kernel) {
        g_element_add3_kernel = clCreateKernel(utils_program, "element_add3", &err);
        if (err != CL_SUCCESS || !g_element_add3_kernel) {
            NNOPT_ERROR_FMT("element_add: clCreateKernel(\"element_add3\") failed (%d)", err);
            clReleaseMemObject(out);
            g_element_add3_kernel = nullptr;
            return nullptr;
        }
    }

    int n_int = (int)n;
    clSetKernelArg(g_element_add3_kernel, 0, sizeof(cl_mem), &a);
    clSetKernelArg(g_element_add3_kernel, 1, sizeof(cl_mem), &b);
    clSetKernelArg(g_element_add3_kernel, 2, sizeof(cl_mem), &out);
    clSetKernelArg(g_element_add3_kernel, 3, sizeof(int), &n_int);

    size_t global_size = n;
    err = clEnqueueNDRangeKernel(queue, g_element_add3_kernel, 1, nullptr, &global_size, nullptr, 0, nullptr,
                                 KernelProfiler::event_for("util_element_add"));
    if (err != CL_SUCCESS) {
        NNOPT_ERROR_FMT("element_add: clEnqueueNDRangeKernel failed (%d)", err);
    }

    return out;
}

bool split_last_dim_2(cl_command_queue queue, cl_program utils_program,
                      cl_mem src, cl_mem first, cl_mem second,
                      int rows, int half_cols) {
    cl_int err;
    cl_kernel kernel = clCreateKernel(utils_program, "split_last_dim_2", &err);
    if (err != CL_SUCCESS || !kernel) {
        NNOPT_ERROR_FMT("split_last_dim_2: clCreateKernel(\"split_last_dim_2\") failed (%d)", err);
        return false;
    }

    clSetKernelArg(kernel, 0, sizeof(cl_mem), &src);
    clSetKernelArg(kernel, 1, sizeof(cl_mem), &first);
    clSetKernelArg(kernel, 2, sizeof(cl_mem), &second);
    clSetKernelArg(kernel, 3, sizeof(int), &rows);
    clSetKernelArg(kernel, 4, sizeof(int), &half_cols);

    size_t global_size = (size_t)rows * (size_t)half_cols;
    err = clEnqueueNDRangeKernel(queue, kernel, 1, nullptr, &global_size, nullptr, 0, nullptr,
                                 KernelProfiler::event_for("util_split_last_dim_2"));
    if (err != CL_SUCCESS) {
        NNOPT_ERROR_FMT("split_last_dim_2: clEnqueueNDRangeKernel failed (%d)", err);
        clReleaseKernel(kernel);
        return false;
    }

    // SYNC-01: queue is in-order; downstream kernels see this output without
    // explicit sync. NNOPT_DEBUG_SYNC strips to no-op in release.
    NNOPT_DEBUG_SYNC(queue);
    clReleaseKernel(kernel);
    return true;
}

// CLBlast status code → human-readable name + meaning. Generated from
// /Users/<...>/.nnopt/deps/clblast/include/clblast.h:55-120 (kept in sync
// with the CLBlast header we link against). Keeps the failure message
// actionable: "status=-1010" tells nobody anything, but
// "kInsufficientMemoryB (Matrix B's OpenCL buffer is too small)" plus the
// requested M/N/K and actual buffer size lets the agent identify the
// caller-vs-weight dim mismatch on the FIRST failing cycle.
static const char* nnopt_clblast_status_name(int s) {
    switch (s) {
        case 0:     return "kSuccess";
        // OpenCL-passthrough errors (most useful: -38, -46, -54, -55).
        case -11:   return "kOpenCLBuildProgramFailure (OpenCL kernel compile failed)";
        case -30:   return "kInvalidValue (CL_INVALID_VALUE)";
        case -36:   return "kInvalidCommandQueue (CL_INVALID_COMMAND_QUEUE)";
        case -38:   return "kInvalidMemObject (CL_INVALID_MEM_OBJECT — buffer is null or released)";
        case -42:   return "kInvalidBinary";
        case -43:   return "kInvalidBuildOptions";
        case -44:   return "kInvalidProgram";
        case -45:   return "kInvalidProgramExecutable";
        case -46:   return "kInvalidKernelName (CL_INVALID_KERNEL_NAME — host name doesn't match a __kernel in the .cl)";
        case -47:   return "kInvalidKernelDefinition";
        case -48:   return "kInvalidKernel";
        case -49:   return "kInvalidArgIndex";
        case -50:   return "kInvalidArgValue";
        case -51:   return "kInvalidArgSize";
        case -52:   return "kInvalidKernelArgs (CL_INVALID_KERNEL_ARGS — some kernel arg unset)";
        case -53:   return "kInvalidLocalNumDimensions (CL_INVALID_WORK_DIMENSION)";
        case -54:   return "kInvalidLocalThreadsTotal (CL_INVALID_WORK_GROUP_SIZE — local size exceeds device limit)";
        case -55:   return "kInvalidLocalThreadsDim (CL_INVALID_WORK_ITEM_SIZE — per-dim local size exceeds device limit)";
        case -56:   return "kInvalidGlobalOffset";
        case -57:   return "kInvalidEventWaitList";
        case -58:   return "kInvalidEvent";
        case -59:   return "kInvalidOperation (CL_INVALID_OPERATION)";
        case -61:   return "kInvalidBufferSize";
        case -63:   return "kInvalidGlobalWorkSize";
        // CLBlast / clBLAS shared error codes — matrix/dim/buffer issues.
        case -1024: return "kNotImplemented";
        case -1022: return "kInvalidMatrixA (Matrix A is not a valid OpenCL buffer)";
        case -1021: return "kInvalidMatrixB (Matrix B is not a valid OpenCL buffer)";
        case -1020: return "kInvalidMatrixC (Matrix C is not a valid OpenCL buffer)";
        case -1019: return "kInvalidVectorX";
        case -1018: return "kInvalidVectorY";
        case -1017: return "kInvalidDimension (M, N, K must be > 0)";
        case -1016: return "kInvalidLeadDimA (lda smaller than A's first dimension)";
        case -1015: return "kInvalidLeadDimB (ldb smaller than B's first dimension)";
        case -1014: return "kInvalidLeadDimC (ldc smaller than C's first dimension)";
        case -1013: return "kInvalidIncrementX";
        case -1012: return "kInvalidIncrementY";
        case -1011: return "kInsufficientMemoryA (Matrix A's OpenCL buffer is too small for the requested GEMM)";
        case -1010: return "kInsufficientMemoryB (Matrix B's OpenCL buffer is too small for the requested GEMM — check that the WEIGHT shape on disk matches the (N, K) the caller is passing)";
        case -1009: return "kInsufficientMemoryC (Matrix C's OpenCL buffer is too small)";
        case -1008: return "kInsufficientMemoryX";
        case -1007: return "kInsufficientMemoryY";
        // CLBlast-specific.
        case -2050: return "kInsufficientMemoryTemp";
        case -2049: return "kInvalidBatchCount";
        case -2048: return "kInvalidOverrideKernel";
        case -2047: return "kMissingOverrideParameter";
        case -2046: return "kInvalidLocalMemUsage (not enough device local memory)";
        case -2045: return "kNoHalfPrecision (device doesn't support fp16)";
        case -2044: return "kNoDoublePrecision";
        case -2043: return "kInvalidVectorScalar";
        case -2042: return "kInsufficientMemoryScalar";
        case -2041: return "kDatabaseError";
        case -2040: return "kUnknownError";
        case -2039: return "kUnexpectedError";
        default:    return "<unrecognized CLBlast status — see clblast.h>";
    }
}

// Query the actual byte-size of an OpenCL buffer object. Used to enrich
// CLBlast failure messages with what the buffer ACTUALLY holds vs what the
// GEMM dimensions require — turns "status=-1010" into "buffer is X bytes,
// GEMM needs Y bytes (off by Z)" which the agent can act on.
static size_t nnopt_cl_mem_size_bytes(cl_mem m) {
    if (!m) return 0;
    size_t sz = 0;
    cl_int e = clGetMemObjectInfo(m, CL_MEM_SIZE, sizeof(sz), &sz, nullptr);
    return (e == CL_SUCCESS) ? sz : 0;
}

// Forward declarations resolved at link time; defined in src/ops/lfm2_common.cpp.
extern "C" {
    cl_kernel nnopt_get_gemv_kernel(cl_command_queue queue);
}

bool pytorch_linear(cl_command_queue queue,
                    int M, int N, int K,
                    cl_mem x, cl_mem W, cl_mem out) {
    // out[M, N] = x[M, K] @ W[N, K]^T  where W is nn.Linear weight [N, K].
    //
    // FAST PATH for M==1 (decode step): use a custom GEMV kernel instead of
    // CLBlast HGEMM. CLBlast adds ~25ms of per-call CPU overhead (kernel arg
    // setup + dispatch coordination). Decode does ~100 projections per token,
    // so ~2.5s/token is host overhead alone. The GEMV kernel takes a single
    // clEnqueueNDRangeKernel — orders of magnitude cheaper to dispatch.
    if (M == 1) {
#ifdef NNOPT_USE_FP16
        // int8 image path — halves weight bandwidth vs fp16, hits texture L1.
        // Only dispatched when nnopt_register_int8_weight() recorded W (from
        // main.cpp's all_keys() walk of model.int8.bin). On miss falls through
        // to the fp16 image path below.
        if (run_gemv_m1_image_int8(queue, N, K, W, x, out)) {
            NNOPT_DEBUG_SYNC(queue);
            return true;
        }
        // Phase 2: Adreno texture-cache image2d_t path (1.71x faster L1 vs L2 on Razr 2020,
        // measured 13.46 vs 7.85 GB/s). Tries no8 for large N (K=1024), no4 fallback,
        // tiles for lm_head N=65536. Compiled as separate cl_program to avoid register
        // allocation interference with lfm2_ops.cl kernels.
        if (run_gemv_m1_image(queue, N, K, W, x, out)) {
            NNOPT_DEBUG_SYNC(queue);
            return true;
        }
#endif
        // Buffer GEMV fallback: single-output, WG=64, runtime-K.
        cl_kernel gemv = nnopt_get_gemv_kernel(queue);
        if (gemv) {
            cl_int e1 = clSetKernelArg(gemv, 0, sizeof(cl_mem), &x);
            cl_int e2 = clSetKernelArg(gemv, 1, sizeof(cl_mem), &W);
            cl_int e3 = clSetKernelArg(gemv, 2, sizeof(cl_mem), &out);
            cl_int e4 = clSetKernelArg(gemv, 3, sizeof(int), &N);
            cl_int e5 = clSetKernelArg(gemv, 4, sizeof(int), &K);
            if (e1 == CL_SUCCESS && e2 == CL_SUCCESS && e3 == CL_SUCCESS && e4 == CL_SUCCESS && e5 == CL_SUCCESS) {
                // Kernel uses 64-WI workgroup-level reduction -> global = N*64, local = 64.
                size_t global = (size_t)N * 64u;
                size_t local = 64u;
                cl_int re = clEnqueueNDRangeKernel(queue, gemv, 1, nullptr, &global, &local,
                                                   0, nullptr, KernelProfiler::event_for("gemv_pytorch_linear"));
                if (re == CL_SUCCESS) return true;
                NNOPT_ERROR_FMT("pytorch_linear M=1: gemv_pytorch_linear dispatch failed (%d) -- falling back to CLBlast", (int)re);
            }
            // Else fall through to CLBlast.
        }
    }
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
    // M>1 with int8-registered W: CLBlast can't read int8 directly. Dequantize
    // once into a cached fp16 scratch buffer, then run HGemm on the scratch.
    // Scratch is reused per (W, N, K); dequant cost is amortized across
    // prefill rows since this kernel is bandwidth-bound and N*K dequant is a
    // small fraction of N*K*M HGemm work for prefill seq_len ≥ ~64.
    cl_mem W_for_gemm = W;
    if (is_int8_registered(W)) {
        auto it = s_int8_aux.find(W);
        if (it != s_int8_aux.end()) {
            cl_mem scratch = dequant_int8_weight_to_fp16(queue, W, it->second.scale_root, N, K);
            if (scratch) W_for_gemm = scratch;
        }
    }
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
        W_for_gemm, 0, K,
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
        // Enrich the failure message with: (1) the CLBlast status name,
        // (2) the requested M/N/K, (3) the actual buffer sizes for x/W/out
        // alongside what the GEMM dimensions imply they should be. This
        // turns an opaque "status=-1010" into a one-line diagnosis the
        // agent can act on without needing to look up the CLBlast header.
        const size_t elem = sizeof(nnopt_storage_t);
        const size_t x_bytes_actual = nnopt_cl_mem_size_bytes(x);
        const size_t W_bytes_actual = nnopt_cl_mem_size_bytes(W);
        const size_t out_bytes_actual = nnopt_cl_mem_size_bytes(out);
        const size_t x_bytes_need = (size_t)M * (size_t)K * elem;
        const size_t W_bytes_need = (size_t)N * (size_t)K * elem;
        const size_t out_bytes_need = (size_t)M * (size_t)N * elem;
        // Compute on-disk weight dim implied by the buffer (W is fp16; rows
        // count = bytes / (cols * elem). When cols == K, rows == intermediate
        // for nn.Linear). This lets the agent compare actual-vs-expected N at
        // a glance instead of dividing in their head.
        const long long W_rows_actual = (K > 0) ? (long long)(W_bytes_actual / ((size_t)K * elem)) : -1;
        NNOPT_ERROR_FMT(
            "pytorch_linear: CLBlast Gemm failed status=%d %s | M=%d N=%d K=%d | "
            "x=%zub need=%zub (M*K*elem) | "
            "W=%zub need=%zub (N*K*elem; weight stored as [N,K] for nn.Linear; rows-implied N=%lld) | "
            "out=%zub need=%zub. "
            "DIAGNOSTIC FLOW (no prescription — diagnose the root): "
            "(1) Read .nnport/model_info.json::tensor_shapes for this layer's weight key (whichever the caller passed to op_Linear). "
            "(2) Compare the on-disk shape against the M/N/K above to find which dim disagrees. "
            "(3) Consult .nnport/dimensions_audit.json for the MODEL_CONFIG constant feeding the disagreeing dim — "
            "if source=weights and consensus matches the on-disk shape, the bug is NOT the constant; the bug is "
            "wrong weight_key / wrong layer_idx / merged-projection assumed split (e.g. Phi-3 gate_up_proj has "
            "rows=2*intermediate) / transposed layout. "
            "(4) If audit source=config (no evidence keys matched), the family's weight-key layout is novel — "
            "add an entry to dimensionsAudit.ts::DIM_EVIDENCE and re-port.",
            (int)status, nnopt_clblast_status_name((int)status),
            M, N, K,
            x_bytes_actual, x_bytes_need,
            W_bytes_actual, W_bytes_need, W_rows_actual,
            out_bytes_actual, out_bytes_need);
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
        // Same enrichment pattern as pytorch_linear, with conv1d's weight
        // layout: HF Conv1D stores W as [K, N] (in, out), so W's required
        // size is K*N*elem (NOT N*K*elem like nn.Linear).
        const size_t elem = sizeof(nnopt_storage_t);
        const size_t x_bytes_actual = nnopt_cl_mem_size_bytes(x);
        const size_t W_bytes_actual = nnopt_cl_mem_size_bytes(W);
        const size_t out_bytes_actual = nnopt_cl_mem_size_bytes(out);
        const size_t x_bytes_need = (size_t)M * (size_t)K * elem;
        const size_t W_bytes_need = (size_t)K * (size_t)N * elem;
        const size_t out_bytes_need = (size_t)M * (size_t)N * elem;
        NNOPT_ERROR_FMT(
            "pytorch_conv1d: CLBlast Gemm failed status=%d %s | M=%d N=%d K=%d | "
            "x=%zub need=%zub (M*K*elem) | "
            "W=%zub need=%zub (K*N*elem; HF Conv1D stores [K,N] = [in,out]) | "
            "out=%zub need=%zub. "
            "If a buffer is SMALLER than 'need', the caller's dim disagrees with the actual weight shape — "
            "look up the weight's shape in weights/model.meta.json and fix the caller's MODEL_CONFIG value.",
            (int)status, nnopt_clblast_status_name((int)status),
            M, N, K,
            x_bytes_actual, x_bytes_need,
            W_bytes_actual, W_bytes_need,
            out_bytes_actual, out_bytes_need);
        return false;
    }
    NNOPT_DEBUG_SYNC(queue);
    return true;
}

bool pytorch_linear_fused3(cl_command_queue queue, int M, int K,
                           cl_mem x,
                           cl_mem Wa, int Na, cl_mem ya,
                           cl_mem Wb, int Nb, cl_mem yb,
                           cl_mem Wc, int Nc, cl_mem yc) {
#ifdef NNOPT_USE_FP16
    if (M == 1 && run_gemv_fused3_image(queue, K, Wa, Na, Wb, Nb, Wc, Nc, x, ya, yb, yc))
        return true;
#endif
    return pytorch_linear(queue, M, Na, K, x, Wa, ya) &&
           pytorch_linear(queue, M, Nb, K, x, Wb, yb) &&
           pytorch_linear(queue, M, Nc, K, x, Wc, yc);
}

bool pytorch_linear_fused2(cl_command_queue queue, int M, int K,
                           cl_mem x,
                           cl_mem Wa, int Na, cl_mem ya,
                           cl_mem Wb, int Nb, cl_mem yb) {
#ifdef NNOPT_USE_FP16
    if (M == 1 && run_gemv_fused2_image(queue, K, Wa, Na, Wb, Nb, x, ya, yb))
        return true;
#endif
    return pytorch_linear(queue, M, Na, K, x, Wa, ya) &&
           pytorch_linear(queue, M, Nb, K, x, Wb, yb);
}
