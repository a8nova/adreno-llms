#include "utils.h"
#include "debug_utils.h"   // NNOPT_ERROR_FMT — used by element_add / pytorch_linear / etc.
#include "kernel_profiler.h"

#include <clblast.h>       // clblast::Gemm — used by pytorch_linear (dtype-templated dispatch).
#include <cstdio>          // snprintf for per-shape profile labels
#include <cstring>         // memset for cl_image_desc
#include <unordered_map>

#include <fstream>
#include <cstring>
#include <algorithm>
#include <numeric>
#include <cstdint>
#include <vector>

// ──────────────────────────────────────────────────────────────────────────────
// fp32-accumulation GEMV for the M==1 (decode) hot path.
//
// PyTorch CPU fp16 always upcasts inputs to fp32 before GEMM. CLBlast Hgemm
// on Adreno 620 uses native fp16 arithmetic, which causes ~3% relative error
// per GEMM. For K=4608 (MLP projections), the accumulated error (n*eps ≈ 4.6)
// exceeds fp16 precision entirely, causing cos < 0.7 at deep MLP layers and
// flipping greedy-decode token rankings.
//
// Fix: for M==1, replace CLBlast Hgemm with a custom OpenCL GEMV that reads
// half weights but accumulates each dot product in float. One work group of
// GEMV_WG=128 threads per output row; float4/vload_half4 for bandwidth.
// ──────────────────────────────────────────────────────────────────────────────
#ifdef NNOPT_USE_FP16

// GEMV_WG sweep on Adreno 620: WG=128 baseline = 33% of ceiling at K=1024 N=4608.
// WG=64 (one wave per WG, no inter-wave barrier, 2× iterations per thread for
// better latency hiding) is worth testing — toggle here. Re-measure after change.
static const char* kGemvSrc = R"CL(
#pragma OPENCL EXTENSION cl_khr_fp16 : enable
#define GEMV_WG 64

// y[N] = W[N,K] @ x[K], fp32 accumulation, half I/O.
// Launch: global=(N*GEMV_WG,), local=(GEMV_WG,) => one WG per output row.
__attribute__((reqd_work_group_size(GEMV_WG, 1, 1)))
__kernel void gemv_rT_fp32acc(
    __global const half* W,
    __global const half* x,
    __global half* y,
    const int K)
{
    const int row = (int)get_group_id(0);
    const int lid = (int)get_local_id(0);

    __local float partial[GEMV_WG];
    float acc = 0.0f;
    const int W_base = row * K;
    const int K4 = K >> 2;

    for (int k4 = lid; k4 < K4; k4 += GEMV_WG) {
        float4 wv = vload_half4(k4, W + W_base);
        float4 xv = vload_half4(k4, x);
        acc += dot(wv, xv);
    }
    for (int k = K4 * 4 + lid; k < K; k += GEMV_WG) {
        acc += (float)W[W_base + k] * (float)x[k];
    }

    partial[lid] = acc;
    barrier(CLK_LOCAL_MEM_FENCE);
    for (int s = GEMV_WG/2; s > 0; s >>= 1) {
        if (lid < s) partial[lid] += partial[lid + s];
        barrier(CLK_LOCAL_MEM_FENCE);
    }
    if (lid == 0) vstore_half(partial[0], row, y);
}
)CL";

static cl_program s_gemv_prog   = nullptr;
static cl_kernel  s_gemv_kernel = nullptr;

static bool ensure_gemv_program(cl_command_queue queue) {
    if (s_gemv_prog) return true;

    cl_context    ctx    = nullptr;
    cl_device_id  device = nullptr;
    clGetCommandQueueInfo(queue, CL_QUEUE_CONTEXT, sizeof(ctx),    &ctx,    nullptr);
    clGetCommandQueueInfo(queue, CL_QUEUE_DEVICE,  sizeof(device), &device, nullptr);
    if (!ctx || !device) {
        NNOPT_ERROR_FMT("gemv_fp32acc: failed to get context/device from queue (err=unknown)%s", "");
        return false;
    }

    const char* src     = kGemvSrc;
    size_t      src_len = strlen(src);
    cl_int err;
    s_gemv_prog = clCreateProgramWithSource(ctx, 1, &src, &src_len, &err);
    if (err != CL_SUCCESS) {
        NNOPT_ERROR_FMT("gemv_fp32acc: clCreateProgramWithSource failed (%d)", (int)err);
        s_gemv_prog = nullptr;
        return false;
    }
    err = clBuildProgram(s_gemv_prog, 1, &device, "-cl-fast-relaxed-math", nullptr, nullptr);
    if (err != CL_SUCCESS) {
        size_t log_size = 0;
        clGetProgramBuildInfo(s_gemv_prog, device, CL_PROGRAM_BUILD_LOG, 0, nullptr, &log_size);
        if (log_size > 0) {
            std::vector<char> log(log_size + 1, 0);
            clGetProgramBuildInfo(s_gemv_prog, device, CL_PROGRAM_BUILD_LOG, log_size, log.data(), nullptr);
            fprintf(stderr, "gemv_fp32acc build log: %s\n", log.data());
        }
        clReleaseProgram(s_gemv_prog);
        s_gemv_prog = nullptr;
        return false;
    }
    s_gemv_kernel = clCreateKernel(s_gemv_prog, "gemv_rT_fp32acc", &err);
    if (err != CL_SUCCESS) {
        NNOPT_ERROR_FMT("gemv_fp32acc: clCreateKernel failed (%d)", (int)err);
        clReleaseProgram(s_gemv_prog);
        s_gemv_prog = nullptr;
        return false;
    }
    return true;
}

static bool run_gemv_rT_fp32acc(cl_command_queue queue, int N, int K, cl_mem W, cl_mem x, cl_mem y) {
    if (!ensure_gemv_program(queue)) return false;
    clSetKernelArg(s_gemv_kernel, 0, sizeof(cl_mem), &W);
    clSetKernelArg(s_gemv_kernel, 1, sizeof(cl_mem), &x);
    clSetKernelArg(s_gemv_kernel, 2, sizeof(cl_mem), &y);
    clSetKernelArg(s_gemv_kernel, 3, sizeof(int),    &K);
    size_t gws = (size_t)N * 64;
    size_t lws = 64;
    char lbl[64]; snprintf(lbl, sizeof(lbl), "gemv_fp32acc_K%d_N%d", K, N);
    cl_event* evt = KernelProfiler::event_for(lbl);
    cl_int err = clEnqueueNDRangeKernel(queue, s_gemv_kernel, 1, nullptr, &gws, &lws, 0, nullptr, evt);
    if (err != CL_SUCCESS) {
        NNOPT_ERROR_FMT("gemv_fp32acc: clEnqueueNDRangeKernel failed (%d)", (int)err);
        return false;
    }
    return true;
}

// ─────────────────────────────────────────────────────────────────────────────
// LFM2.5 cooperative + multi-output GEMV (kernels/gemv_m1.cl).
//
// Specializations match the K values that occur in this port:
//   K=1024 (most projections)  → gemv_m1_k1024_no4
//   K=4608 (mlp.w2)            → gemv_m1_k4608_no4
// Both require N % 4 == 0; every site in the model satisfies this. WG=64
// (one Adreno A6xx wave). 4 outputs per WG share x reads.
//
// Eligibility predicate: M==1 && N%4==0 && K∈{1024,4608}. Outside the
// predicate falls through to the WG=128 single-output gemv_rT_fp32acc above.
// ─────────────────────────────────────────────────────────────────────────────

static cl_program s_gemv_m1_prog       = nullptr;
static cl_kernel  s_gemv_m1_k1024      = nullptr;
static cl_kernel  s_gemv_m1_k1024_no4  = nullptr;
static cl_kernel  s_gemv_m1_k4608_no4  = nullptr;
// Image-backed variants (Adreno texture L1 cache, 1.71× faster than buffer L2 on Razr 2020).
static cl_kernel  s_gemv_m1_k1024_no8_img = nullptr;
static cl_kernel  s_gemv_m1_k1024_no4_img = nullptr;
static cl_kernel  s_gemv_m1_k1024_no2_img = nullptr;
static cl_kernel  s_gemv_m1_k1024_no8_silufused_img = nullptr;
static cl_kernel  s_gemv_m1_k4608_no4_img = nullptr;

// Int8 image-path variants (kernels/gemv_m1_int8.cl). Loaded by
// ensure_gemv_m1_int8_program(); each one is the int8 counterpart of the
// matching s_gemv_m1_k*_no*_img kernel above. Null if the build failed
// (treated as "int8 path unavailable" → caller falls through to fp16).
static cl_program s_gemv_m1_int8_prog            = nullptr;
static cl_kernel  s_gemv_m1_k1024_no4_img_int8   = nullptr;
static cl_kernel  s_gemv_m1_k1024_no8_img_int8   = nullptr;
static cl_kernel  s_gemv_m1_k4608_no4_img_int8   = nullptr;
static cl_kernel  s_gemv_m1_k4608_no8_img_int8   = nullptr;

// Block-32 symmetric Q4 image-path variants (kernels/gemv_m1_q4.cl). Same
// dispatch geometry as int8 but the W reads come back as uint4 (RGBA UINT8)
// and have to be nibble-unpacked + scale-multiplied per-block. Per-token
// weight footprint is ~half of int8.
static cl_program s_gemv_m1_q4_prog              = nullptr;
static cl_kernel  s_gemv_m1_k1024_no4_img_q4     = nullptr;
static cl_kernel  s_gemv_m1_k1024_no8_img_q4     = nullptr;
static cl_kernel  s_gemv_m1_k4608_no4_img_q4     = nullptr;

// Per-weight int8 metadata. Keyed by the int8 cl_mem buffer that the layer
// passes into pytorch_linear() as W. Holds:
//   image  — image2d_t view of W (CL_RGBA / CL_SIGNED_INT8, K/4 px × N rows)
//   scale  — fp16 per-row absolute-max scale buffer [N]
//   N, K   — explicit dims so the dispatch site doesn't have to re-derive
// Single-image entries store {image, scale} directly. Weights whose row count
// exceeds CL_DEVICE_IMAGE2D_MAX_HEIGHT (e.g. lm_head N=65536 > 16384 on
// Adreno 6xx) take the tiled path: row-major sub-buffers of W_int8 and
// scale_fp16, with a per-tile image2d_t view of the W sub-buffer.
struct Int8Tile {
    cl_mem sub_W     = nullptr;
    cl_mem image     = nullptr;
    cl_mem scale_sub = nullptr;
    int    row_offset = 0;
    int    row_count  = 0;
};
struct Int8Aux {
    cl_mem image = nullptr;      // single-image path (nullptr → tiled or pending)
    cl_mem scale = nullptr;      // single-image path
    std::vector<Int8Tile> tiles; // tiled path (V > image-height cap)
    int N = 0;
    int K = 0;
    cl_mem W_int8     = nullptr; // root W buffer, kept for sub-buffer creation
    cl_mem scale_root = nullptr; // root scale buffer, for sub-buffer creation
};
static std::unordered_map<cl_mem, Int8Aux> s_int8_aux;

// Q4 weight registry. Same shape as Int8Aux but the image is built from a
// uint8 buffer (K/2 bytes per row), and scales are per-block (K/32 fp16 per
// row). Block size is fixed at 32.
struct Q4Tile {
    cl_mem sub_W     = nullptr;
    cl_mem image     = nullptr;
    cl_mem scale_sub = nullptr;
    int    row_offset = 0;
    int    row_count  = 0;
};
struct Q4Aux {
    cl_mem image = nullptr;
    cl_mem scale = nullptr;
    std::vector<Q4Tile> tiles;
    int N = 0;
    int K = 0;
    cl_mem W_q4       = nullptr;
    cl_mem scale_root = nullptr;
};
static std::unordered_map<cl_mem, Q4Aux> s_q4_aux;

// Per-buffer image2d_t view cache (lazy). Standard entries hold one image;
// tiled entries hold multiple sub-buffer/sub-image pairs for weights whose
// row count exceeds CL_DEVICE_IMAGE2D_MAX_HEIGHT (e.g. lm_head N=65536).
struct WImageTile { cl_mem sub_buffer = nullptr; cl_mem image = nullptr; int row_offset = 0; int row_count = 0; };
struct WImageEntry { cl_mem image = nullptr; std::vector<WImageTile> tiles; };
static std::unordered_map<cl_mem, WImageEntry> s_w_image_cache;
static std::unordered_map<cl_mem, char>        s_w_image_skip;
static bool   s_img_limits_known = false;
static size_t s_img_max_w = 0, s_img_max_h = 0;

static bool ensure_gemv_m1_program(cl_command_queue queue) {
    if (s_gemv_m1_prog) return true;

    cl_context   ctx    = nullptr;
    cl_device_id device = nullptr;
    clGetCommandQueueInfo(queue, CL_QUEUE_CONTEXT, sizeof(ctx),    &ctx,    nullptr);
    clGetCommandQueueInfo(queue, CL_QUEUE_DEVICE,  sizeof(device), &device, nullptr);
    if (!ctx || !device) {
        NNOPT_ERROR_FMT("gemv_m1: failed to get context/device%s", "");
        return false;
    }

    // Read kernels/gemv_m1.cl from cwd (build/deploy stages it next to the
    // binary on the device).
    std::ifstream f("kernels/gemv_m1.cl", std::ios::binary);
    if (!f.is_open()) {
        NNOPT_ERROR_FMT("gemv_m1: cannot open kernels/gemv_m1.cl%s", "");
        return false;
    }
    std::string src((std::istreambuf_iterator<char>(f)), std::istreambuf_iterator<char>());
    const char* src_cstr = src.c_str();
    size_t src_len = src.size();
    cl_int err;
    s_gemv_m1_prog = clCreateProgramWithSource(ctx, 1, &src_cstr, &src_len, &err);
    if (err != CL_SUCCESS) {
        NNOPT_ERROR_FMT("gemv_m1: clCreateProgramWithSource failed (%d)", (int)err);
        return false;
    }
    err = clBuildProgram(s_gemv_m1_prog, 1, &device, "-DUSE_FP16=1 -cl-fast-relaxed-math", nullptr, nullptr);
    if (err != CL_SUCCESS) {
        size_t log_size = 0;
        clGetProgramBuildInfo(s_gemv_m1_prog, device, CL_PROGRAM_BUILD_LOG, 0, nullptr, &log_size);
        if (log_size > 0) {
            std::vector<char> log(log_size + 1, 0);
            clGetProgramBuildInfo(s_gemv_m1_prog, device, CL_PROGRAM_BUILD_LOG, log_size, log.data(), nullptr);
            fprintf(stderr, "gemv_m1 build log: %s\n", log.data());
        }
        clReleaseProgram(s_gemv_m1_prog);
        s_gemv_m1_prog = nullptr;
        return false;
    }
    s_gemv_m1_k1024 = clCreateKernel(s_gemv_m1_prog, "gemv_m1_k1024", &err);
    if (err != CL_SUCCESS) { NNOPT_ERROR_FMT("gemv_m1: k1024 create failed (%d)", err); }
    s_gemv_m1_k1024_no4 = clCreateKernel(s_gemv_m1_prog, "gemv_m1_k1024_no4", &err);
    if (err != CL_SUCCESS) { NNOPT_ERROR_FMT("gemv_m1: k1024_no4 create failed (%d)", err); }
    s_gemv_m1_k4608_no4 = clCreateKernel(s_gemv_m1_prog, "gemv_m1_k4608_no4", &err);
    if (err != CL_SUCCESS) { NNOPT_ERROR_FMT("gemv_m1: k4608_no4 create failed (%d)", err); }
    // Image variants — optional (silently null on devices without image2d-from-buffer).
    s_gemv_m1_k1024_no8_img = clCreateKernel(s_gemv_m1_prog, "gemv_m1_k1024_no8_img", &err);
    if (err != CL_SUCCESS) { s_gemv_m1_k1024_no8_img = nullptr; }
    s_gemv_m1_k1024_no4_img = clCreateKernel(s_gemv_m1_prog, "gemv_m1_k1024_no4_img", &err);
    if (err != CL_SUCCESS) { s_gemv_m1_k1024_no4_img = nullptr; }
    s_gemv_m1_k1024_no2_img = clCreateKernel(s_gemv_m1_prog, "gemv_m1_k1024_no2_img", &err);
    if (err != CL_SUCCESS) { s_gemv_m1_k1024_no2_img = nullptr; }
    s_gemv_m1_k4608_no4_img = clCreateKernel(s_gemv_m1_prog, "gemv_m1_k4608_no4_img", &err);
    if (err != CL_SUCCESS) { s_gemv_m1_k4608_no4_img = nullptr; }
    return s_gemv_m1_k1024 && s_gemv_m1_k4608_no4;
}

// Build kernels/gemv_m1_int8.cl. Same context/device as the fp16 program.
// Failures are silent — int8 path stays unavailable; callers fall through.
static bool ensure_gemv_m1_int8_program(cl_command_queue queue) {
    if (s_gemv_m1_int8_prog) return true;

    cl_context   ctx    = nullptr;
    cl_device_id device = nullptr;
    clGetCommandQueueInfo(queue, CL_QUEUE_CONTEXT, sizeof(ctx),    &ctx,    nullptr);
    clGetCommandQueueInfo(queue, CL_QUEUE_DEVICE,  sizeof(device), &device, nullptr);
    if (!ctx || !device) return false;

    std::ifstream f("kernels/gemv_m1_int8.cl", std::ios::binary);
    if (!f.is_open()) {
        NNOPT_ERROR_FMT("gemv_m1_int8: cannot open kernels/gemv_m1_int8.cl%s", "");
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
    s_gemv_m1_k4608_no8_img_int8 = clCreateKernel(s_gemv_m1_int8_prog, "gemv_m1_k4608_no8_img_int8", &err);
    if (err != CL_SUCCESS) { s_gemv_m1_k4608_no8_img_int8 = nullptr; }
    return true;
}

// Register an int8 weight buffer with utils so pytorch_linear() can pick it up
// at M=1. Called from main.cpp after Weights::load() under NNOPT_QUANT=int8.
// Builds the int8 image2d_t view lazily on first dispatch (needs a queue);
// the registration only records the {scale, N, K} sidecar for now.
bool nnopt_register_int8_weight(cl_mem W_int8, cl_mem scale_fp16, int N, int K) {
    if (!W_int8 || !scale_fp16 || N <= 0 || K <= 0) return false;
    Int8Aux a;
    a.image = nullptr;       // built lazily on first GEMV call
    a.scale = nullptr;       // single-image path: scale is just scale_fp16
    a.N = N;
    a.K = K;
    a.W_int8     = W_int8;
    a.scale_root = scale_fp16;
    s_int8_aux[W_int8] = a;
    return true;
}

// Lazily build the single image2d_t view or the per-tile views for an int8
// weight. Returns true if the entry is ready for dispatch. The choice
// between single-image and tiled depends on N vs the device's image-height
// cap (CL_DEVICE_IMAGE2D_MAX_HEIGHT). Adreno 6xx caps at 16384, so any
// weight with N > 16384 (only lm_head at V=65536) tiles into 4 chunks.
static bool prepare_int8_entry(cl_command_queue queue, cl_mem W) {
    auto it = s_int8_aux.find(W);
    if (it == s_int8_aux.end()) return false;
    Int8Aux& a = it->second;
    if (a.image || !a.tiles.empty()) return true;  // already prepared

    cl_context   ctx = nullptr;
    cl_device_id dev = nullptr;
    if (clGetCommandQueueInfo(queue, CL_QUEUE_CONTEXT, sizeof(ctx), &ctx, nullptr) != CL_SUCCESS ||
        clGetCommandQueueInfo(queue, CL_QUEUE_DEVICE,  sizeof(dev), &dev, nullptr) != CL_SUCCESS) return false;

    // Use the same image-limit globals the fp16 path probes once.
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

    // Single-image path: weight fits in one image2d_t.
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

    // Tiled path: split N into chunks of s_img_max_h rows. Each tile gets
    // a sub-buffer of W (int8) + a sub-buffer of scale (fp16) + an
    // image2d_t view of the W sub-buffer.
    //
    // Sub-buffer base offsets must be 128-byte aligned on Adreno. Check:
    //   W_int8 offset = row_offset * K bytes; aligned if K%128==0 — K=1024 ✓
    //   scale offset  = row_offset * 2 bytes; aligned if row_offset%64==0
    //                   — tile height 16384 is a multiple of 64 ✓
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

// Int8 image-path GEMV dispatch. Returns false on miss/failure (caller falls
// through to fp16). Eligibility:
//   - W is registered as int8 (via nnopt_register_int8_weight)
//   - K is 1024 or 4608, matches a built kernel
//   - N is divisible by 4 (or 8 for the no8 fast path on N≥2048)
static bool run_gemv_m1_image_int8(cl_command_queue queue, int N, int K, cl_mem W, cl_mem x, cl_mem out) {
    auto it = s_int8_aux.find(W);
    if (it == s_int8_aux.end()) return false;
    if (!ensure_gemv_m1_int8_program(queue)) return false;
    if (!prepare_int8_entry(queue, W)) return false;

    // Lambda picks kernel + stride from the active K/N. For single-image
    // path it runs once; for tiled it runs once per tile with the tile's
    // own row count as N.
    auto pick_kernel = [&](int Nlocal) -> std::pair<cl_kernel,int> {
        if (K == 1024 && Nlocal >= 2048 && (Nlocal % 8) == 0 && s_gemv_m1_k1024_no8_img_int8) return {s_gemv_m1_k1024_no8_img_int8, 8};
        if (K == 1024 &&                   (Nlocal % 4) == 0 && s_gemv_m1_k1024_no4_img_int8) return {s_gemv_m1_k1024_no4_img_int8, 4};
        // K=4608 only happens at w2 (down-proj). _no8 variant exists in the
        // kernel file (gemv_m1_k4608_no8_img_int8) but measured −6.5% decode
        // on Tab A9+: 8 fp32 acc + 18 K-iters × 8 W reads spilled registers.
        // Stick with no4 here. K=4608 was already at 85% of texture ceiling
        // per Razr 2020 BENCHMARK — there's no room to extract via wider tiles.
        if (K == 4608 &&                   (Nlocal % 4) == 0 && s_gemv_m1_k4608_no4_img_int8) return {s_gemv_m1_k4608_no4_img_int8, 4};
        return {nullptr, 0};
    };

    Int8Aux& a = it->second;

    // Tiled dispatch: loop over the prebuilt tiles, slice `out` by row.
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

    // Single-image dispatch.
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

// ─── Q4 program build + dispatch (mirrors the int8 path) ───
static bool ensure_gemv_m1_q4_program(cl_command_queue queue) {
    if (s_gemv_m1_q4_prog) return true;
    cl_context   ctx    = nullptr;
    cl_device_id device = nullptr;
    clGetCommandQueueInfo(queue, CL_QUEUE_CONTEXT, sizeof(ctx),    &ctx,    nullptr);
    clGetCommandQueueInfo(queue, CL_QUEUE_DEVICE,  sizeof(device), &device, nullptr);
    if (!ctx || !device) return false;

    std::ifstream f("kernels/gemv_m1_q4.cl", std::ios::binary);
    if (!f.is_open()) {
        NNOPT_ERROR_FMT("gemv_m1_q4: cannot open kernels/gemv_m1_q4.cl%s", "");
        return false;
    }
    std::string src((std::istreambuf_iterator<char>(f)), std::istreambuf_iterator<char>());
    const char* src_cstr = src.c_str();
    size_t src_len = src.size();
    cl_int err;
    s_gemv_m1_q4_prog = clCreateProgramWithSource(ctx, 1, &src_cstr, &src_len, &err);
    if (err != CL_SUCCESS) {
        NNOPT_ERROR_FMT("gemv_m1_q4: clCreateProgramWithSource failed (%d)", (int)err);
        return false;
    }
    err = clBuildProgram(s_gemv_m1_q4_prog, 1, &device, "-cl-fast-relaxed-math", nullptr, nullptr);
    if (err != CL_SUCCESS) {
        size_t log_size = 0;
        clGetProgramBuildInfo(s_gemv_m1_q4_prog, device, CL_PROGRAM_BUILD_LOG, 0, nullptr, &log_size);
        if (log_size > 0) {
            std::vector<char> log(log_size + 1, 0);
            clGetProgramBuildInfo(s_gemv_m1_q4_prog, device, CL_PROGRAM_BUILD_LOG, log_size, log.data(), nullptr);
            fprintf(stderr, "gemv_m1_q4 build log: %s\n", log.data());
        }
        clReleaseProgram(s_gemv_m1_q4_prog);
        s_gemv_m1_q4_prog = nullptr;
        return false;
    }
    s_gemv_m1_k1024_no4_img_q4 = clCreateKernel(s_gemv_m1_q4_prog, "gemv_m1_k1024_q4_no4_img", &err);
    if (err != CL_SUCCESS) { s_gemv_m1_k1024_no4_img_q4 = nullptr; }
    s_gemv_m1_k1024_no8_img_q4 = clCreateKernel(s_gemv_m1_q4_prog, "gemv_m1_k1024_q4_no8_img", &err);
    if (err != CL_SUCCESS) { s_gemv_m1_k1024_no8_img_q4 = nullptr; }
    s_gemv_m1_k4608_no4_img_q4 = clCreateKernel(s_gemv_m1_q4_prog, "gemv_m1_k4608_q4_no4_img", &err);
    if (err != CL_SUCCESS) { s_gemv_m1_k4608_no4_img_q4 = nullptr; }
    return true;
}

bool nnopt_register_q4_weight(cl_mem W_q4, cl_mem scale_fp16, int N, int K) {
    if (!W_q4 || !scale_fp16 || N <= 0 || K <= 0) return false;
    Q4Aux a;
    a.image = nullptr;
    a.scale = nullptr;
    a.N = N;
    a.K = K;
    a.W_q4       = W_q4;
    a.scale_root = scale_fp16;
    s_q4_aux[W_q4] = a;
    return true;
}

// Lazy build of the per-W image2d_t view(s). Q4 packs 2 weights/byte, so the
// image width is K/8 pixels (RGBA UINT8 = 4 bytes = 8 weights per pixel).
// Tiled path handles N > image-height cap (lm_head V=65536 > 16384).
static bool prepare_q4_entry(cl_command_queue queue, cl_mem W) {
    auto it = s_q4_aux.find(W);
    if (it == s_q4_aux.end()) return false;
    Q4Aux& a = it->second;
    if (a.image || !a.tiles.empty()) return true;

    cl_context   ctx = nullptr;
    cl_device_id dev = nullptr;
    if (clGetCommandQueueInfo(queue, CL_QUEUE_CONTEXT, sizeof(ctx), &ctx, nullptr) != CL_SUCCESS ||
        clGetCommandQueueInfo(queue, CL_QUEUE_DEVICE,  sizeof(dev), &dev, nullptr) != CL_SUCCESS) return false;
    if (!s_img_limits_known) {
        clGetDeviceInfo(dev, CL_DEVICE_IMAGE2D_MAX_WIDTH,  sizeof(s_img_max_w), &s_img_max_w, nullptr);
        clGetDeviceInfo(dev, CL_DEVICE_IMAGE2D_MAX_HEIGHT, sizeof(s_img_max_h), &s_img_max_h, nullptr);
        s_img_limits_known = true;
    }
    if ((size_t)(a.K / 8) > s_img_max_w) {
        NNOPT_ERROR_FMT("q4 image too wide: K/8=%d > %zu", a.K/8, s_img_max_w);
        return false;
    }

    cl_image_format fmt;
    fmt.image_channel_order = CL_RGBA;
    fmt.image_channel_data_type = CL_UNSIGNED_INT8;

    if ((size_t)a.N <= s_img_max_h) {
        cl_image_desc desc; std::memset(&desc, 0, sizeof(desc));
        desc.image_type   = CL_MEM_OBJECT_IMAGE2D;
        desc.image_width  = (size_t)(a.K / 8);
        desc.image_height = (size_t)a.N;
        desc.buffer       = W;
        cl_int err = CL_SUCCESS;
        cl_mem img = clCreateImage(ctx, CL_MEM_READ_ONLY, &fmt, &desc, nullptr, &err);
        if (err != CL_SUCCESS || !img) {
            NNOPT_ERROR_FMT("q4 clCreateImage failed (%d) N=%d K=%d", (int)err, a.N, a.K);
            return false;
        }
        a.image = img;
        a.scale = a.scale_root;
        return true;
    }

    // Tiled path. Each tile: sub-buffer of W (uint8, K/2 bytes/row),
    // sub-buffer of scale (fp16, K/32 elems/row), image view of the W sub-buffer.
    const int TILE_H = (int)s_img_max_h;
    const size_t row_bytes_W     = (size_t)(a.K / 2);              // 2 weights per byte
    const size_t row_bytes_scale = (size_t)(a.K / 32) * sizeof(nnopt_storage_t);
    int rows_left = a.N, row_off = 0;
    while (rows_left > 0) {
        const int tile_n = rows_left < TILE_H ? rows_left : TILE_H;
        Q4Tile t; t.row_offset = row_off; t.row_count = tile_n;

        cl_buffer_region w_region  { (size_t)row_off * row_bytes_W,     (size_t)tile_n * row_bytes_W };
        cl_buffer_region s_region  { (size_t)row_off * row_bytes_scale, (size_t)tile_n * row_bytes_scale };
        cl_int err = CL_SUCCESS;
        t.sub_W     = clCreateSubBuffer(W,              CL_MEM_READ_ONLY, CL_BUFFER_CREATE_TYPE_REGION, &w_region, &err);
        if (err != CL_SUCCESS || !t.sub_W)     { NNOPT_ERROR_FMT("q4 tile sub_W: %d row=%d", err, row_off); return false; }
        t.scale_sub = clCreateSubBuffer(a.scale_root,   CL_MEM_READ_ONLY, CL_BUFFER_CREATE_TYPE_REGION, &s_region, &err);
        if (err != CL_SUCCESS || !t.scale_sub) { NNOPT_ERROR_FMT("q4 tile scale_sub: %d row=%d", err, row_off); return false; }

        cl_image_desc desc; std::memset(&desc, 0, sizeof(desc));
        desc.image_type   = CL_MEM_OBJECT_IMAGE2D;
        desc.image_width  = (size_t)(a.K / 8);
        desc.image_height = (size_t)tile_n;
        desc.buffer       = t.sub_W;
        t.image = clCreateImage(ctx, CL_MEM_READ_ONLY, &fmt, &desc, nullptr, &err);
        if (err != CL_SUCCESS || !t.image)     { NNOPT_ERROR_FMT("q4 tile clCreateImage: %d row=%d", err, row_off); return false; }

        a.tiles.push_back(t);
        row_off   += tile_n;
        rows_left -= tile_n;
    }
    return true;
}

// Q4 image-path GEMV dispatch. Returns false on miss (caller falls through to
// int8 if registered there, otherwise to fp16).
static bool run_gemv_m1_image_q4(cl_command_queue queue, int N, int K, cl_mem W, cl_mem x, cl_mem out) {
    auto it = s_q4_aux.find(W);
    if (it == s_q4_aux.end()) return false;
    if (!ensure_gemv_m1_q4_program(queue)) return false;
    if (!prepare_q4_entry(queue, W)) return false;

    auto pick_kernel = [&](int Nlocal) -> std::pair<cl_kernel,int> {
        if (K == 1024 && (Nlocal % 4) == 0 && s_gemv_m1_k1024_no4_img_q4) return {s_gemv_m1_k1024_no4_img_q4, 4};
        // K=1024 _no8 variant exists in the kernel file (gemv_m1_k1024_q4_no8_img)
        // but measured −38% decode on Tab A9+: 8 fp32 acc + 8 unpacks of 8 weights
        // each spilled the register file hard on Adreno 619 v2. Stick with no4.
        // K=4608 _no8 also off-table — int8 no8 spilled at this K (T2 −6.5%);
        // Q4 has more ALU per byte, would be worse.
        if (K == 4608 && (Nlocal % 4) == 0 && s_gemv_m1_k4608_no4_img_q4) return {s_gemv_m1_k4608_no4_img_q4, 4};
        return {nullptr, 0};
    };

    Q4Aux& a = it->second;

    if (!a.tiles.empty()) {
        for (const auto& t : a.tiles) {
            auto [kt, stride_t] = pick_kernel(t.row_count);
            if (!kt) return false;
            cl_buffer_region out_region{ (size_t)t.row_offset * sizeof(nnopt_storage_t),
                                         (size_t)t.row_count  * sizeof(nnopt_storage_t) };
            cl_int err = CL_SUCCESS;
            cl_mem out_sub = clCreateSubBuffer(out, CL_MEM_READ_WRITE, CL_BUFFER_CREATE_TYPE_REGION, &out_region, &err);
            if (err != CL_SUCCESS || !out_sub) { NNOPT_ERROR_FMT("q4 tile out_sub: %d row=%d", err, t.row_offset); return false; }
            int tile_n = t.row_count;
            clSetKernelArg(kt, 0, sizeof(cl_mem), &x);
            clSetKernelArg(kt, 1, sizeof(cl_mem), &t.image);
            clSetKernelArg(kt, 2, sizeof(cl_mem), &t.scale_sub);
            clSetKernelArg(kt, 3, sizeof(cl_mem), &out_sub);
            clSetKernelArg(kt, 4, sizeof(int),    &tile_n);
            const size_t WG = 64;
            size_t gws = (size_t)(tile_n / stride_t) * WG;
            size_t lws = WG;
            char lbl[64]; snprintf(lbl, sizeof(lbl), "gemv_m1_K%d_N%d_no%d_img_q4_tile", K, tile_n, stride_t);
            cl_event* evt = KernelProfiler::event_for(lbl);
            err = clEnqueueNDRangeKernel(queue, kt, 1, nullptr, &gws, &lws, 0, nullptr, evt);
            clReleaseMemObject(out_sub);
            if (err != CL_SUCCESS) { NNOPT_ERROR_FMT("q4 tile enqueue: %d row=%d", err, t.row_offset); return false; }
        }
        return true;
    }

    auto [k, stride] = pick_kernel(N);
    if (!k || !a.image) return false;

    char lbl[64]; snprintf(lbl, sizeof(lbl), "gemv_m1_K%d_N%d_no%d_img_q4", K, N, stride);
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
    if (err != CL_SUCCESS) { NNOPT_ERROR_FMT("gemv_m1_image_q4 enqueue: %d K=%d N=%d", err, K, N); return false; }
    return true;
}

// Separate cl_program for the silufused kernel — keeping it in gemv_m1.cl
// caused Adreno's compiler to spill registers in the no8_img kernels,
// regressing decode by ~10×. Compiling as standalone isolates allocation.
static cl_program s_mlp_fused_prog = nullptr;

static bool ensure_mlp_fused_program(cl_command_queue queue) {
    if (s_mlp_fused_prog) return true;

    cl_context   ctx    = nullptr;
    cl_device_id device = nullptr;
    clGetCommandQueueInfo(queue, CL_QUEUE_CONTEXT, sizeof(ctx),    &ctx,    nullptr);
    clGetCommandQueueInfo(queue, CL_QUEUE_DEVICE,  sizeof(device), &device, nullptr);
    if (!ctx || !device) return false;

    std::ifstream f("kernels/mlp_fused.cl", std::ios::binary);
    if (!f.is_open()) {
        NNOPT_ERROR_FMT("mlp_fused: cannot open kernels/mlp_fused.cl%s", "");
        return false;
    }
    std::string src((std::istreambuf_iterator<char>(f)), std::istreambuf_iterator<char>());
    const char* src_cstr = src.c_str();
    size_t src_len = src.size();
    cl_int err;
    s_mlp_fused_prog = clCreateProgramWithSource(ctx, 1, &src_cstr, &src_len, &err);
    if (err != CL_SUCCESS) {
        NNOPT_ERROR_FMT("mlp_fused: clCreateProgramWithSource failed (%d)", (int)err);
        return false;
    }
    err = clBuildProgram(s_mlp_fused_prog, 1, &device, "-cl-fast-relaxed-math", nullptr, nullptr);
    if (err != CL_SUCCESS) {
        size_t log_size = 0;
        clGetProgramBuildInfo(s_mlp_fused_prog, device, CL_PROGRAM_BUILD_LOG, 0, nullptr, &log_size);
        if (log_size > 0) {
            std::vector<char> log(log_size + 1, 0);
            clGetProgramBuildInfo(s_mlp_fused_prog, device, CL_PROGRAM_BUILD_LOG, log_size, log.data(), nullptr);
            fprintf(stderr, "mlp_fused build log: %s\n", log.data());
        }
        clReleaseProgram(s_mlp_fused_prog);
        s_mlp_fused_prog = nullptr;
        return false;
    }
    s_gemv_m1_k1024_no8_silufused_img = clCreateKernel(s_mlp_fused_prog, "gemv_m1_k1024_no8_silufused_img", &err);
    if (err != CL_SUCCESS) { s_gemv_m1_k1024_no8_silufused_img = nullptr; }
    return s_gemv_m1_k1024_no8_silufused_img != nullptr;
}

// Look up (or create on first use) an image2d_t view of fp16 weight buffer W
// shaped [N, K]. Standard layout: single image (K/4 wide, N tall). Tiled
// fallback when N > CL_DEVICE_IMAGE2D_MAX_HEIGHT — used for lm_head N=65536
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
        // One-time init log gated behind NNOPT_DEBUG_LAYERS so it doesn't
        // interleave with the streamed token output on first decode call.
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

    // Standard layout — single image, fits in one shot.
    if (cl_mem img = try_create(W, (size_t)(K / 4), (size_t)N)) {
        WImageEntry e; e.image = img;
        s_w_image_cache[W] = e; return &s_w_image_cache[W];
    }

    // Tiled fallback for lm_head (N=65536 > 16384).
    if (K > 0 && (size_t)(K / 4) <= s_img_max_w) {
        const int TILE_H = (int)s_img_max_h;
        const int row_bytes = K * 2;
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
    if (!ensure_gemv_m1_program(queue)) return false;
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
    } else if (K == 1024 && (N % 2) == 0 && s_gemv_m1_k1024_no2_img) {
        k = s_gemv_m1_k1024_no2_img; stride = 2;
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
    // Output row size for fp16 logits is 2 bytes — sub-buffer offsets must be 128 B aligned.
    // tile.row_offset × 2 bytes — need (row_offset × 2) % 128 == 0 ⇒ row_offset % 64 == 0.
    // CL_DEVICE_IMAGE2D_MAX_HEIGHT is 16384 on Adreno 620 (multiple of 64) — natural alignment.
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

// K=1024 single-output specialization: uses gemv_m1_k1024 (fully unrolled,
// hardcoded K, WG=64). Same arg signature as gemv_fp32acc (W, x, y, K-implicit).
static bool run_gemv_m1_k1024(cl_command_queue queue, int N, cl_mem W, cl_mem x, cl_mem y) {
    if (!ensure_gemv_m1_program(queue)) return false;
    if (!s_gemv_m1_k1024) return false;
    clSetKernelArg(s_gemv_m1_k1024, 0, sizeof(cl_mem), &W);
    clSetKernelArg(s_gemv_m1_k1024, 1, sizeof(cl_mem), &x);
    clSetKernelArg(s_gemv_m1_k1024, 2, sizeof(cl_mem), &y);
    clSetKernelArg(s_gemv_m1_k1024, 3, sizeof(int),    &N);

    size_t gws = (size_t)N * 64;
    size_t lws = 64;
    char lbl[64]; snprintf(lbl, sizeof(lbl), "gemv_m1_K1024_N%d", N);
    cl_event* evt = KernelProfiler::event_for(lbl);
    cl_int err = clEnqueueNDRangeKernel(queue, s_gemv_m1_k1024, 1, nullptr, &gws, &lws, 0, nullptr, evt);
    if (err != CL_SUCCESS) {
        NNOPT_ERROR_FMT("gemv_m1_k1024 enqueue failed (N=%d err=%d)", N, err);
        return false;
    }
    return true;
}

// Dispatch the no4 specializations when eligible. Returns false on any error
// (caller falls back to gemv_rT_fp32acc / CLBlast).
static bool run_gemv_m1_no4(cl_command_queue queue, int N, int K, cl_mem W, cl_mem x, cl_mem y) {
    if (!ensure_gemv_m1_program(queue)) return false;
    cl_kernel k = nullptr;
    // K=1024 no4 was a ~1.6× regression on Adreno 620 in measurement (2790→4496 µs/call
    // for K=1024 N=4608) — register pressure from 4 acc + 4 W vec4 chains exceeds the
    // per-wave VGPR budget so the kernel spills, while the existing WG=128 single-output
    // baseline already saturates at 33% of ceiling. K=1024 sites stay on
    // gemv_rT_fp32acc until a different lever (image1d_buffer_t / no2 / subgroup) wins.
    if      (K == 4608 && (N % 4) == 0) k = s_gemv_m1_k4608_no4;
    else return false;

    clSetKernelArg(k, 0, sizeof(cl_mem), &x);
    clSetKernelArg(k, 1, sizeof(cl_mem), &W);
    clSetKernelArg(k, 2, sizeof(cl_mem), &y);
    clSetKernelArg(k, 3, sizeof(int),    &N);

    // K=1024 specialization uses WG=128 (matches the baseline thread count for
    // occupancy on Adreno 620). K=4608 uses WG=64 (one wave) — N is small at
    // that site (1024) so high arithmetic density per thread amortizes the
    // lower thread count.
    const size_t WG  = (K == 1024) ? 128 : 64;
    const int    n_wg = N / 4;
    size_t gws = (size_t)n_wg * WG;
    size_t lws = WG;

    char lbl[64]; snprintf(lbl, sizeof(lbl), "gemv_m1_K%d_N%d_no4", K, N);
    cl_event* evt = KernelProfiler::event_for(lbl);
    cl_int err = clEnqueueNDRangeKernel(queue, k, 1, nullptr, &gws, &lws, 0, nullptr, evt);
    if (err != CL_SUCCESS) {
        NNOPT_ERROR_FMT("gemv_m1_no4: enqueue failed (K=%d N=%d err=%d)", K, N, err);
        return false;
    }
    return true;
}
#endif // NNOPT_USE_FP16

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
    cl_event* evt = KernelProfiler::event_for("element_add_inplace");
    err = clEnqueueNDRangeKernel(queue, s_cached_kernel, 1, nullptr, &gws, nullptr, 0, nullptr, evt);
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
    cl_event* ea_evt = KernelProfiler::event_for("element_add");
    err = clEnqueueNDRangeKernel(queue, kernel, 1, nullptr, &global_size, nullptr, 0, nullptr, ea_evt);
    if (err != CL_SUCCESS) {
        NNOPT_ERROR_FMT("element_add: clEnqueueNDRangeKernel failed (%d)", err);
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
    cl_event* sl_evt = KernelProfiler::event_for("split_last_dim_2");
    err = clEnqueueNDRangeKernel(queue, kernel, 1, nullptr, &global_size, nullptr, 0, nullptr, sl_evt);
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

bool pytorch_linear(cl_command_queue queue,
                    int M, int N, int K,
                    cl_mem x, cl_mem W, cl_mem out) {
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
    // M==1 (decode) hot path: use fp32-accumulation GEMV instead of CLBlast
    // Hgemm. CLBlast Hgemm on Adreno 620 accumulates in native fp16, producing
    // ~3% relative error per call. For K=4608 (MLP projections) this gives
    // n*eps ≈ 4.6, causing systematic cosine drops in deep MLP layers that
    // flip greedy-decode token rankings. Our GEMV reads half weights but
    // accumulates via float4/dot → float, matching PyTorch CPU fp16 behavior.
    if (M == 1) {
        // Q4 image path: quarter the weight bytes vs fp16, half of int8.
        // Only dispatched if `nnopt_register_q4_weight` registered W.
        if (run_gemv_m1_image_q4(queue, N, K, W, x, out)) {
            NNOPT_DEBUG_SYNC(queue);
            return true;
        }
        // Int8 image path: half the weight bytes, hits the texture L1. Only
        // dispatched if `nnopt_register_int8_weight` registered W; otherwise
        // falls through to the fp16 image kernels below.
        if (run_gemv_m1_image_int8(queue, N, K, W, x, out)) {
            NNOPT_DEBUG_SYNC(queue);
            return true;
        }
        // Phase 2: Adreno texture-cache image2d_t path (1.71× faster L1 vs L2 on Razr 2020,
        // measured 13.46 vs 7.85 GB/s in --bw-probe). Tries no4 first, no2 fallback for
        // K=1024 (lower register pressure if no4 spills), tiles for lm_head N=65536.
        if (run_gemv_m1_image(queue, N, K, W, x, out)) {
            NNOPT_DEBUG_SYNC(queue);
            return true;
        }
        // K=4608: use the no4 multi-output specialization (Step 1, +1.6× per
        // call vs WG=128 baseline). N=1024 is small at this site so 4-output
        // amortization wins despite lower thread count.
        if (run_gemv_m1_no4(queue, N, K, W, x, out)) {
            NNOPT_DEBUG_SYNC(queue);
            return true;
        }
        // K=1024 hardcoded specialization REGRESSED on Adreno 620 (4.94 → 4.33
        // tok/s in measurement). The runtime-K loop in gemv_fp32acc generates
        // better Adreno IL than `#pragma unroll` over a hardcoded-K loop with
        // explicit `vload_half4(0, ptr+off)` form — likely a compiler quirk
        // around vload_half4 offset normalization. Kept gemv_m1_k1024 in
        // gemv_m1.cl for future re-evaluation under different drivers.
        if (run_gemv_rT_fp32acc(queue, N, K, W, x, out)) {
            NNOPT_DEBUG_SYNC(queue);
            return true;
        }
        // Fall through to CLBlast if GEMV fails (e.g., build error on device).
    }
    // Use the portable host-side IEEE 754 fp16 encoder defined above.
    // clblast::FloatToHalf is not portable across CLBlast builds (some
    // versions only expose it when cl_khr_fp16 was enabled at CLBlast
    // build time, leading to "no member named 'FloatToHalf'" link errors).
    cl_half h_one  = static_cast<cl_half>(nnopt_f32_to_f16(1.0f));
    cl_half h_zero = static_cast<cl_half>(nnopt_f32_to_f16(0.0f));
    char lin_lbl[48]; snprintf(lin_lbl, sizeof(lin_lbl), "linear_M%d_K%d_N%d", M, K, N);
    cl_event* lin_evt = KernelProfiler::event_for(lin_lbl);
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
        lin_evt);
#else
    char lin_lbl[48]; snprintf(lin_lbl, sizeof(lin_lbl), "linear_M%d_K%d_N%d", M, K, N);
    cl_event* lin_evt = KernelProfiler::event_for(lin_lbl);
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
        lin_evt);
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

#ifdef NNOPT_USE_FP16
bool pytorch_linear_silu_gate_fused(cl_command_queue queue,
                                    int N, int K,
                                    cl_mem x, cl_mem W3, cl_mem gate_inout) {
    // Eligibility: K=1024, N%8==0, no8 silufused kernel built, image2d view of W3 available.
    if (K != 1024 || (N % 8) != 0) return false;
    if (!ensure_gemv_m1_program(queue)) return false;
    if (!ensure_mlp_fused_program(queue)) return false;
    if (!s_gemv_m1_k1024_no8_silufused_img) return false;

    const WImageEntry* ent = get_or_create_w_image(queue, W3, N, K);
    if (!ent || !ent->image) return false;  // tiled path not supported here (lm_head only)

    cl_kernel k = s_gemv_m1_k1024_no8_silufused_img;
    clSetKernelArg(k, 0, sizeof(cl_mem), &x);
    clSetKernelArg(k, 1, sizeof(cl_mem), &ent->image);
    clSetKernelArg(k, 2, sizeof(cl_mem), &gate_inout);
    clSetKernelArg(k, 3, sizeof(int),    &N);

    const size_t WG = 64;
    size_t gws = (size_t)(N / 8) * WG;
    size_t lws = WG;
    cl_event* evt = KernelProfiler::event_for("gemv_m1_K1024_no8_silufused_img");
    cl_int err = clEnqueueNDRangeKernel(queue, k, 1, nullptr, &gws, &lws, 0, nullptr, evt);
    if (err != CL_SUCCESS) {
        NNOPT_ERROR_FMT("silufused enqueue: %d (N=%d)", err, N);
        return false;
    }
    NNOPT_DEBUG_SYNC(queue);
    return true;
}
#else
bool pytorch_linear_silu_gate_fused(cl_command_queue, int, int, cl_mem, cl_mem, cl_mem) {
    return false;  // fp32 build: no image path, fall back to host
}
#endif

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
    char c1d_lbl[48]; snprintf(c1d_lbl, sizeof(c1d_lbl), "conv1d_M%d_K%d_N%d", M, K, N);
    cl_event* c1d_evt = KernelProfiler::event_for(c1d_lbl);
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
        c1d_evt);
#else
    char c1d_lbl[48]; snprintf(c1d_lbl, sizeof(c1d_lbl), "conv1d_M%d_K%d_N%d", M, K, N);
    cl_event* c1d_evt = KernelProfiler::event_for(c1d_lbl);
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
        c1d_evt);
#endif
    if (status != clblast::StatusCode::kSuccess) {
        NNOPT_ERROR_FMT("pytorch_conv1d: CLBlast Gemm failed status=%d (M=%d N=%d K=%d)",
                        (int)status, M, N, K);
        return false;
    }
    NNOPT_DEBUG_SYNC(queue);
    return true;
}
