#include "utils.h"
#include "debug_utils.h"   // NNOPT_ERROR_FMT — used by element_add / pytorch_linear / etc.
#include <set>
#include <map>             // per-kernel work-group gate: one-time warn set
#include "kernel_profiler.h"
#include "opencl_context.h"   // build_cached_program_from_queue for the sweep

#include <clblast.h>       // clblast::Gemm — used by pytorch_linear (dtype-templated dispatch).
#include <cstdio>          // snprintf for per-shape profile labels
#include <cstring>      // strstr for the device denylist
#include <cctype>       // tolower         // memset for cl_image_desc
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
static cl_kernel  s_gemv_m1_k2560_no4_img_int8   = nullptr;  // LFM2.5-230M w2 (K=2560)

// Block-32 symmetric Q4 image-path variants (kernels/gemv_m1_q4.cl). Same
// dispatch geometry as int8 but the W reads come back as uint4 (RGBA UINT8)
// and have to be nibble-unpacked + scale-multiplied per-block. Per-token
// weight footprint is ~half of int8.
static cl_program s_gemv_m1_q4_prog              = nullptr;
static cl_kernel  s_gemv_m1_k1024_no4_img_q4     = nullptr;
static cl_kernel  s_gemv_m1_q4_no4_buf           = nullptr;  // buffer path (no image2d_t)
// Per-SHAPE tuned buffer kernels. Measured on PowerVR GE8320: the optimum work-group
// size depends on N, because this is a 1-CU part and what matters is how many groups are
// in flight, not the group size itself:
//   N=65536 (lm_head, 8192 groups at nout=8) -> wg=32: 76.3 ms, wg=64: 76.6 ms
//   N=3072  (MLP,      384 groups at nout=8) -> wg=64: 3.708 ms, wg=32: 4.368 ms (+15%)
// Large N already has plenty of groups so smaller groups schedule better; small N needs
// bigger groups or the device runs out of work. nout=8 wins at every N on this GPU.
// Adreno is the opposite: one wg=64/nout=4 kernel (nout=8 spills its register file).
struct Q4BufKernel {
    cl_kernel k = nullptr; size_t wg = 64; int nout = 4; int hm = 0; int bt = 1; int mode = 0;
    int rn = 1;   // ROW_NOUT: output rows per work-item (ROW mode only)
};

// MEASURED per-shape geometry. Every row came from the on-device sweep
// (NNOPT_SWEEP=1) against the real weights, with each result checked against the
// reference kernel — none of it is inferred.
//
// PowerVR Rogue GE8320 (Vivo Y21), LFM2.5-230M Q4_0:
//     N=65536 K=1024 -> wg=32 nout=8 hm=0   73.705 ms
//     N=3072  K=1024 -> wg=64 nout=8 hm=0    3.739 ms
//     N=2560  K=1024 -> wg=32 nout=8 hm=0    3.137 ms
//     N=1024  K=2560 -> wg=64 nout=8 hm=1    3.061 ms
//     N=1024  K=1024 -> wg=64 nout=8 hm=1    1.472 ms
//     N=512   K=1024 -> wg=32 nout=8 hm=1    0.772 ms
//
// Two things the data says that a heuristic would have got wrong:
//   * nout=8 wins at EVERY shape here and loses badly on Adreno (which spills).
//   * half-math (hm=1) wins only on the SMALL shapes. Rogue runs FP16 at twice
//     the FP32 rate (Imagination's Rogue OpenCL guide), but on the big shapes the
//     extra half<->float conversions cost more than the doubled ALU rate saves.
//   * work-group size does NOT follow a clean threshold in N — 32 wins at
//     N=65536/2560/512, 64 wins at N=3072/1024. Hence a table, not a formula.
//
// Adreno keeps a single wg=64/nout=4/hm=0 kernel; measured best at every shape.
// RE-RUN THE SWEEP before editing these, and add rows rather than guessing.
// mode: 0 = split-row reduction kernel, 1 = one-lane-per-output-row kernel.
struct Q4Geom { int N, K, wg, nout, hm, mode, rn; };
// MEASURED on PowerVR Rogue GE8320 (Vivo Y21), q4_0, 5 reps per config, verified
// against the reference at err/rms <= 0.001. Re-measured 2026-09-14 with the
// rows-per-lane (rn) axis added, which is the first change in three rounds to
// move the achieved-bandwidth number off the 0.38-0.58 GB/s plateau.
//
//   N      K     wg  rn    ms/call   GB/s    vs rn=1
//   65536  1024  64   4     58.320   0.65    -16.5%   <- lm_head, 27% of the budget
//    3072  1024  256  4      2.927   0.61     -8.8%
//    2560  1024  64   2      2.591   0.57     -8.4%   <- MLP w1/w3, 33% of the budget
//    1024  2560  512  1      2.526   0.59     -1.5%
//    1024  1024  128  4      1.030   0.57     -7.9%
//     512  1024  32   1      0.756   0.39     (rn=1 best)
//
// WHY rn>1 works here and the two obvious levers did not: this kernel is bound by
// the SERIAL dequant chain (mask->shift->convert->subtract->mul->add per weight),
// not by bandwidth (it runs at ~15% of a measured 3.4 GB/s streaming ceiling) and
// not by FLOPs (~0.9% of a measured 229 GFLOP/s fp16 rate). Widening the load to
// 16 B was wall-neutral and a lane-interleaved repack was shown unnecessary, but
// rn>1 gives each lane rn INDEPENDENT chains to interleave, which is what a
// dependency-stalled kernel actually needs. It also loads the activation block
// once for all rn rows.
//
// rn is NOT monotonic: it costs registers (private 136 B at rn=1, 240 B at rn=2,
// 292 B at rn=4) and the two smallest-work shapes prefer rn=1. Per-shape only.
// nout is unused in mode 1 (each work-item owns rn rows); left at 1 so the
// reduction-mode N%nout guard is trivially satisfied.
static const Q4Geom kQ4GeomRogue[] = {
    {65536, 1024,  64, 1, 0, 1, 4},
    { 3072, 1024, 256, 1, 0, 1, 4},
    { 2560, 1024,  64, 1, 0, 1, 2},
    { 1024, 2560, 512, 1, 0, 1, 1},
    { 1024, 1024, 128, 1, 0, 1, 4},
    {  512, 1024,  32, 1, 0, 1, 1},
};
// Fallback for a shape not in the table. rn=1 is the safe default: rn>1 is a win
// only where there is enough work per lane to pay for the extra registers.
static const Q4Geom kQ4GeomRogueDefault = {0, 0, 128, 1, 0, 1, 1};

static std::map<std::pair<int,int>, Q4BufKernel> s_q4_buf_by_shape;
static Q4BufKernel s_q4_buf_default;
static bool        s_q4_buf_is_rogue = false;
static std::string s_q4_src;                                 // kernel source, kept for retuning
static cl_kernel  s_gemv_m1_k1024_no8_img_q4     = nullptr;
static cl_kernel  s_gemv_m1_k4608_no4_img_q4     = nullptr;
static cl_kernel  s_gemv_m1_k2560_no4_img_q4     = nullptr;  // LFM2.5-230M w2 (K=2560)

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

// ── image2d-from-buffer viability gate ──────────────────────────────────────
// Every texture-path GEMV (q4/int8/fp16) builds a CL_MEM_OBJECT_IMAGE2D with
// desc.buffer set. That is cl_khr_image2d_from_buffer, and the spec requires the
// row pitch to respect CL_DEVICE_IMAGE_PITCH_ALIGNMENT (in pixels) and the base
// address to respect CL_DEVICE_IMAGE_BASE_ADDRESS_ALIGNMENT. Adreno's values are
// permissive so this was never checked. PowerVR Rogue (GE8320) is stricter:
// clCreateImage still SUCCEEDS, then the kernel faults on read --
//   NNOPT_TRACE: KERNEL DIED kernel='gemv_m1_k1024_q4_no4_img' clFinish=-14
// and every downstream fallback reinterprets the packed bytes as fp16, so the
// model emits reserved-token garbage instead of failing. Gate the texture path
// on the device actually supporting it.
//
// NNOPT_NO_IMAGES=1 forces the buffer path everywhere (useful for A/B on Adreno).
static bool   s_img_from_buffer_ok = false;
static size_t s_img_pitch_align = 0, s_img_base_align = 0;
static bool   s_img_validate_pending = false;

// Empirical backstop for the alignment gate above. The alignment rule PREDICTS
// which devices can do image2d-from-buffer; this CONFIRMS it. The first texture
// dispatch is followed by a clFinish, and if the kernel died the texture path is
// disabled process-wide so the caller falls through to the buffer GEMV. Costs
// exactly one clFinish per process. Needed because a device may report
// permissive alignment and still fault (PowerVR clCreateImage succeeds, then the
// read faults), which no static check can predict.
// ── Explicit local work size for 1-D elementwise dispatches ─────────────────
// These kernels used to pass local_work_size = nullptr ("driver, you pick").
// That is legal OpenCL, but under 1.2 work-groups must be UNIFORM: whatever the
// driver picks must divide the global size exactly, while respecting the device
// max. Adreno's heuristic copes; PowerVR Rogue GE8320 refuses the dispatch
// (*** NDRANGE_KERNEL executed abnormally ***). Handing a portability-critical
// decision to a driver we do not control is the bug.
//
// Instead: pick an explicit work-group size that divides the device max, round
// the global size UP to a multiple of it, and rely on the kernels' existing
// bounds guards (element_add: `if (gid < n)`, split_last_dim_2:
// `if (gid >= total) return;`) to discard the padding threads.


// ── Per-kernel death detection + blacklist ─────────────────────────────────
// clEnqueueNDRangeKernel returning CL_SUCCESS only means the kernel was QUEUED.
// A kernel that faults during execution is reported later, at the next blocking
// call — by which time run_gemv_m1_image() has already returned true and the
// caller has accepted its garbage output. That is why PowerVR produced text
// instead of an error: gemv_m1_k1024_no8_img died asynchronously and nothing
// noticed, so the (correct) buffer-GEMV fallback was never reached.
//
// Fix: the first time we dispatch a given kernel, clFinish and check. If it
// died, blacklist that cl_kernel permanently and return false so the caller
// falls through to a lighter/simpler path. Cost is ONE clFinish per distinct
// kernel per process, then zero — not per dispatch.
//
// This is deliberately cause-agnostic: it recovers whether the kernel dies from
// register pressure, an image-addressing quirk, or something else we have not
// identified on this driver.
static std::set<cl_kernel> s_kernel_ok;
static std::set<cl_kernel> s_kernel_dead;

static bool nnopt_kernel_is_dead(cl_kernel k) {
    return k && s_kernel_dead.find(k) != s_kernel_dead.end();
}

// mark_ok=false: check this dispatch but do NOT retire the kernel from checking.
// Needed because the lm_head is TILED — the same cl_kernel is dispatched once per
// tile (N=65536 over max_h=8192 => 8 tiles), each with a different sub-buffer and
// image. Retiring after tile 0 meant a death on tile 1..7 was never seen, which is
// exactly what happened: "first texture dispatch OK" followed by a dead kernel.
// A kernel is only retired once an ENTIRE tiled call has completed cleanly.
static bool nnopt_validate_kernel_dispatch(cl_command_queue queue, cl_kernel k,
                                           const char* who, bool mark_ok) {
    if (!k) return false;
    if (s_kernel_ok.find(k) != s_kernel_ok.end()) return true;
    if (s_kernel_dead.find(k) != s_kernel_dead.end()) return false;
    const cl_int e = clFinish(queue);
    if (e != CL_SUCCESS) {
        s_kernel_dead.insert(k);
        fprintf(stderr, "NNOPT_WG: kernel '%s' DIED (clFinish=%d) - blacklisted, falling back\n",
                who ? who : "?", e);
        fflush(stderr);
        return false;
    }
    if (mark_ok) s_kernel_ok.insert(k);
    return true;
}

static bool nnopt_validate_kernel_once(cl_command_queue queue, cl_kernel k, const char* who) {
    return nnopt_validate_kernel_dispatch(queue, k, who, /*mark_ok=*/true);
}

// ── Per-kernel work-group capability gate ──────────────────────────────────
// CL_KERNEL_WORK_GROUP_SIZE is a PER-KERNEL, PER-DEVICE limit the driver derives
// from that kernel's register + local-memory footprint. It can be far below
// CL_DEVICE_MAX_WORK_GROUP_SIZE. This port hardcoded WG=64 at every dispatch and
// never asked, which is fine on register-rich Adreno and fatal on a 1-CU PowerVR
// Rogue GE8320: gemv_m1_k1024_no8_img keeps 8 accumulators + 8 float4 weight
// vectors live and died with
//   NNOPT_TRACE: KERNEL DIED kernel='gemv_m1_k1024_no8_img' clFinish=-14 gws=[65536] lws=[64]
//
// NOTE: we SKIP such a kernel rather than shrinking the work-group. These kernels
// hardcode their reduction to 64 lanes (__local float partial[8][64], tree reduce
// from s=32), so running them narrower would leave partial[][] partly
// uninitialised — silently wrong numbers instead of a clean failure. Skipping
// lets the existing fallback chain pick a lighter kernel.

// Fetch a kernel's OpenCL function name for diagnostics (empty string on failure).
static const char* lbl_kernel_name(cl_kernel k) {
    static thread_local char buf[128];
    buf[0] = '\0';
    if (k) clGetKernelInfo(k, CL_KERNEL_FUNCTION_NAME, sizeof(buf) - 1, buf, nullptr);
    return buf;
}

static bool nnopt_kernel_supports_wg(cl_command_queue queue, cl_kernel k, size_t want_wg, const char* who) {
    if (!k) return false;
    cl_device_id dev = nullptr;
    if (clGetCommandQueueInfo(queue, CL_QUEUE_DEVICE, sizeof(dev), &dev, nullptr) != CL_SUCCESS || !dev) return true;
    size_t kmax = 0;
    if (clGetKernelWorkGroupInfo(k, dev, CL_KERNEL_WORK_GROUP_SIZE, sizeof(kmax), &kmax, nullptr) != CL_SUCCESS) return true;
    // Report every GEMV kernel's real on-device resource footprint ONCE. These are
    // the numbers that identify an Adreno-tuned kernel that does not fit elsewhere:
    //   LOCAL_MEM   vs CL_DEVICE_LOCAL_MEM_SIZE  (GE8320 has 4 KB; partial[8][64] = 2 KB)
    //   PRIVATE_MEM > 0 means the kernel SPILLED registers to global memory
    //   WORK_GROUP  is the per-kernel max, which can be far below the device max
    {
        static std::set<cl_kernel> reported;
        if (reported.insert(k).second) {
            cl_ulong lmem = 0, priv = 0;
            clGetKernelWorkGroupInfo(k, dev, CL_KERNEL_LOCAL_MEM_SIZE,   sizeof(lmem), &lmem, nullptr);
            clGetKernelWorkGroupInfo(k, dev, CL_KERNEL_PRIVATE_MEM_SIZE, sizeof(priv), &priv, nullptr);
            fprintf(stderr, "NNOPT_KRES: %-32s wg_max=%-4zu local=%llu B private=%llu B%s\n",
                    who ? who : "?", kmax,
                    (unsigned long long)lmem, (unsigned long long)priv,
                    priv > 0 ? "  <<<< SPILLING" : "");
            fflush(stderr);
        }
    }
    if (kmax >= want_wg) return true;
    static std::set<cl_kernel> warned;
    if (warned.insert(k).second) {
        cl_ulong priv = 0;
        clGetKernelWorkGroupInfo(k, dev, CL_KERNEL_PRIVATE_MEM_SIZE, sizeof(priv), &priv, nullptr);
        fprintf(stderr, "NNOPT_WG: skipping '%s' - kernel max work-group %zu < required %zu "
                        "(private mem %llu B); falling back to a lighter kernel\n",
                who ? who : "?", kmax, want_wg, (unsigned long long)priv);
        fflush(stderr);
    }
    return false;
}

static size_t nnopt_pick_lws(cl_command_queue queue) {
    static size_t cached = 0;
    if (cached) return cached;
    cl_device_id dev = nullptr;
    size_t dev_max = 0;
    if (clGetCommandQueueInfo(queue, CL_QUEUE_DEVICE, sizeof(dev), &dev, nullptr) == CL_SUCCESS && dev) {
        clGetDeviceInfo(dev, CL_DEVICE_MAX_WORK_GROUP_SIZE, sizeof(dev_max), &dev_max, nullptr);
    }
    if (dev_max == 0) dev_max = 64;          // conservative floor
    cached = dev_max >= 64 ? 64 : dev_max;   // 64 divides every real device max
    return cached;
}
static inline size_t nnopt_round_up(size_t n, size_t mult) {
    return ((n + mult - 1) / mult) * mult;
}

static bool nnopt_image_dispatch_ok(cl_command_queue queue) {
    if (!s_img_validate_pending) return s_img_from_buffer_ok;
    s_img_validate_pending = false;
    const cl_int e = clFinish(queue);
    if (e != CL_SUCCESS) {
        s_img_from_buffer_ok = false;
        fprintf(stderr, "NNOPT_IMAGES: first texture dispatch FAILED (clFinish=%d) "
                        "-> texture path DISABLED, using buffer GEMV\n", e);
        fflush(stderr);
        return false;
    }
    fprintf(stderr, "NNOPT_IMAGES: first texture dispatch OK — texture path confirmed\n");
    fflush(stderr);
    return true;
}

static bool nnopt_no_images_forced() {
    static int cached = -1;
    if (cached < 0) {
        const char* e = getenv("NNOPT_NO_IMAGES");
        cached = (e && e[0] != '0' && e[0] != '\0') ? 1 : 0;
    }
    return cached == 1;
}


// ── Texture-path device policy (ONE place; the probe block is duplicated 3x) ──
// Policy is a DENYLIST, not an allowlist. The texture GEMV kernels work on Adreno
// and on Mali; they are known to break only on PowerVR Rogue, where
// gemv_m1_k1024_no8_img faults mid-execution on a later lm_head tile (root cause
// still unknown). An allowlist would silently halve throughput on every GPU we
// simply have not tested — Mali included — so the default is ENABLED and we
// subtract only what is proven broken.
//
// A match can only DISABLE an optimisation: worst case is slower, never wrong.
// So match BROADLY (three identity sources, case-insensitive) rather than
// surgically, and FAIL OPEN when the queries give us nothing.
//
// Overrides (nothing sets these by default; they are a manual lever):
//   NNOPT_NO_IMAGES=1     force textures OFF (used over adb to validate the
//                         buffer path, and settable per-device from ProcessEngine)
//   NNOPT_FORCE_IMAGES=1  override the denylist if we ever deny a healthy device
static const char* kTextureDenyList[] = {
    // Rogue-era PowerVR only (Series6/6XT/7/8XE, e.g. GE8320/GE8300/GE6320).
    // Confirmed broken: Vivo Y21 / Helio P35 / PowerVR Rogue GE8320 — 4 KB local
    // vs Adreno's 32 KB, image max_h 8192 vs 16384, 1 CU. gemv_m1_k1024_no8_img
    // faults mid-execution on a later lm_head tile.
    //
    // Deliberately NOT "powervr" or "imagination": Google's Tensor G5 (Pixel 10 /
    // 10 Pro / Fold) ships PowerVR DXT-48-1536, a completely different, far larger
    // architecture that reports e.g. "PowerVR DXT-48-1536" — no "rogue" in it. A
    // broad vendor match would silently halve throughput on a current flagship we
    // have no evidence is broken. The runtime kernel-death check is the backstop
    // if DXT (or anything else) turns out to break too.
    "powervr ge",   // matches "PowerVR GE8320" / "GE8300" / "GE6320" (Rogue GE series).
                    // MEASURED: CL_DEVICE_NAME on the Vivo Y21 is exactly "PowerVR GE8320"
                    // (15 bytes incl. NUL) — note it does NOT contain "Rogue"; that word only
                    // appears in GL_RENDERER, which OpenCL never sees. Matching "rogue" here
                    // would silently never fire.
    "rogue",        // belt-and-braces for any driver that does put it in CL_DEVICE_NAME;
                    // cannot match DXT either.
};

static void nnopt_lower(char* p) { for (; *p; ++p) *p = (char)tolower((unsigned char)*p); }

static bool nnopt_texture_path_allowed(cl_device_id dev, cl_bool img_sup) {
    if (img_sup != CL_TRUE) return false;
    if (nnopt_no_images_forced()) {
        fprintf(stderr, "NNOPT_IMAGES: texture path FORCED OFF (NNOPT_NO_IMAGES)\n");
        fflush(stderr);
        return false;
    }
    char name[256] = {0}, vend[256] = {0}, plat[256] = {0};
    clGetDeviceInfo(dev, CL_DEVICE_NAME,   sizeof(name) - 1, name, nullptr);
    clGetDeviceInfo(dev, CL_DEVICE_VENDOR, sizeof(vend) - 1, vend, nullptr);
    cl_platform_id pid = nullptr;
    if (clGetDeviceInfo(dev, CL_DEVICE_PLATFORM, sizeof(pid), &pid, nullptr) == CL_SUCCESS && pid) {
        clGetPlatformInfo(pid, CL_PLATFORM_NAME, sizeof(plat) - 1, plat, nullptr);
    }
    char lname[256], lvend[256], lplat[256];
    snprintf(lname, sizeof(lname), "%s", name); nnopt_lower(lname);
    snprintf(lvend, sizeof(lvend), "%s", vend); nnopt_lower(lvend);
    snprintf(lplat, sizeof(lplat), "%s", plat); nnopt_lower(lplat);

    const char* hit = nullptr;
    for (const char* pat : kTextureDenyList) {
        if (strstr(lname, pat) || strstr(lvend, pat) || strstr(lplat, pat)) { hit = pat; break; }
    }
    const bool forced_on = [] {
        const char* e = getenv("NNOPT_FORCE_IMAGES");
#ifdef NNOPT_FORCE_IMAGES_DEFAULT_ON
        // Diagnostic build: override the denylist so a denied device still takes the
        // texture path, to find out WHICH shapes fault. The app cannot set env vars
        // for a measurement run, hence the compile-time door.
        return !(e && e[0] == '0');
#else
        return e && e[0] != '0' && e[0] != '\0';
#endif
    }();
    if (hit && forced_on) {
        fprintf(stderr, "NNOPT_IMAGES: name='%s' vendor='%s' platform='%s' -> denied by \"%s\" "
                        "but NNOPT_FORCE_IMAGES set, ENABLING anyway\n", name, vend, plat, hit);
        fflush(stderr);
        return true;
    }
    fprintf(stderr, "NNOPT_IMAGES: name='%s' vendor='%s' platform='%s' -> texture path %s\n",
            name, vend, plat,
            hit ? "DISABLED (matched deny entry, using buffer GEMV)" : "ENABLED");
    fflush(stderr);
    return hit == nullptr;   // fail open: unknown device keeps today's behaviour
}


// Same denylist as the texture gate, but usable before the image probe has run.
// The buffer-GEMV geometry choice needs it at program-build time.
static bool nnopt_device_is_denied_texture(cl_device_id dev) {
    if (nnopt_no_images_forced()) return true;
    char name[256] = {0}, vend[256] = {0};
    clGetDeviceInfo(dev, CL_DEVICE_NAME,   sizeof(name) - 1, name, nullptr);
    clGetDeviceInfo(dev, CL_DEVICE_VENDOR, sizeof(vend) - 1, vend, nullptr);
    char ln[256], lv[256];
    snprintf(ln, sizeof(ln), "%s", name); nnopt_lower(ln);
    snprintf(lv, sizeof(lv), "%s", vend); nnopt_lower(lv);
    for (const char* pat : kTextureDenyList) if (strstr(ln, pat) || strstr(lv, pat)) return true;
    return false;
}

// width_px: the image row width in PIXELS for the tensor about to be wrapped.
static bool nnopt_image_row_ok(size_t width_px) {
    if (nnopt_no_images_forced()) return false;
    if (!s_img_from_buffer_ok)    return false;
    // Pitch alignment is expressed in pixels; a row that is not a multiple of it
    // has an illegal pitch, which is exactly the undefined behaviour above.
    if (s_img_pitch_align && (width_px % s_img_pitch_align) != 0) return false;
    return true;
}

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
    s_q4_src = src;   // kept so the tuned/templated variant can be rebuilt with -D options
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
    s_gemv_m1_k2560_no4_img_int8 = clCreateKernel(s_gemv_m1_int8_prog, "gemv_m1_k2560_no4_img_int8", &err);
    if (err != CL_SUCCESS) { s_gemv_m1_k2560_no4_img_int8 = nullptr; }
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
        cl_bool img_sup = CL_FALSE;
        clGetDeviceInfo(dev, CL_DEVICE_IMAGE_SUPPORT, sizeof(img_sup), &img_sup, nullptr);
        cl_uint pa = 0, ba = 0;
        clGetDeviceInfo(dev, CL_DEVICE_IMAGE_PITCH_ALIGNMENT,        sizeof(pa), &pa, nullptr);
        clGetDeviceInfo(dev, CL_DEVICE_IMAGE_BASE_ADDRESS_ALIGNMENT, sizeof(ba), &ba, nullptr);
        s_img_pitch_align = pa; s_img_base_align = ba;
        s_img_from_buffer_ok = nnopt_texture_path_allowed(dev, img_sup);
        s_img_validate_pending = s_img_from_buffer_ok;
        fprintf(stderr,
                "NNOPT_IMAGES: support=%d max_w=%zu max_h=%zu pitch_align=%u base_align=%u -> texture path %s\n",
                (int)img_sup, s_img_max_w, s_img_max_h, pa, ba,
                nnopt_no_images_forced() ? "FORCED OFF" : (s_img_from_buffer_ok ? "eligible" : "DISABLED"));
        fflush(stderr);
        s_img_limits_known = true;
    }
    if (!nnopt_image_row_ok((size_t)(a.K / 4))) {
        // NOT a fallback. gemv_m1_int8.cl contains ONLY _img kernels -- there is no
        // int8 buffer variant -- so returning false here drops through to the fp16
        // image kernels in pytorch_linear(), which reinterpret the int8 bytes as
        // fp16 and emit reserved-token garbage. Measured on a PowerVR GE8320
        // (2026-09-14): int8 produced `<|reserved_29|>` output for exactly this
        // reason. q4 fails loudly in the same situation; int8 has no such guard.
        // int8 is therefore BROKEN on any texture-denied device.
        NNOPT_ERROR_FMT("int8 texture path unavailable (K/4=%d pitch_align=%zu) and there is "
                        "no int8 buffer kernel -- output WILL be garbage on this device",
                        a.K/4, s_img_pitch_align);
        return false;
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
        // K=2560 is the LFM2.5-230M w2 down-proj (intermediate_size=2560, no
        // auto-adjust). Same no4 texture path as K=4608, 10 K-iters.
        if (K == 2560 &&                   (Nlocal % 4) == 0 && s_gemv_m1_k2560_no4_img_int8) return {s_gemv_m1_k2560_no4_img_int8, 4};
        return {nullptr, 0};
    };

    Int8Aux& a = it->second;

    // Tiled dispatch: loop over the prebuilt tiles, slice `out` by row.
    if (!a.tiles.empty()) {
        cl_kernel retire_kernel = nullptr;   // retired only after ALL tiles pass
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
            // Skip this kernel if the device cannot run it at WG=64 (see nnopt_kernel_supports_wg).
            if (nnopt_kernel_is_dead(kt)) return false;
        if (!nnopt_kernel_supports_wg(queue, kt, 64, lbl_kernel_name(kt))) return false;
            const size_t WG = 64;
            size_t gws = (size_t)(tile_n / stride_t) * WG;
            size_t lws = WG;
            char lbl[64]; snprintf(lbl, sizeof(lbl), "gemv_m1_K%d_N%d_no%d_img_int8_tile", K, tile_n, stride_t);
            cl_event* evt = KernelProfiler::event_for(lbl);
            err = clEnqueueNDRangeKernel(queue, kt, 1, nullptr, &gws, &lws, 0, nullptr, evt);
            clReleaseMemObject(out_sub);
            if (err != CL_SUCCESS) { NNOPT_ERROR_FMT("int8 tile enqueue: %d row=%d", err, t.row_offset); return false; }
            if (!nnopt_validate_kernel_dispatch(queue, kt, lbl_kernel_name(kt), /*mark_ok=*/false)) return false;
            retire_kernel = kt;
        }
        // Whole tiled call survived — retire the kernel so we stop clFinish-ing.
        if (retire_kernel) s_kernel_ok.insert(retire_kernel);
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
    // Skip this kernel if the device cannot run it at WG=64 (see nnopt_kernel_supports_wg).
    if (nnopt_kernel_is_dead(k)) return false;
    if (!nnopt_kernel_supports_wg(queue, k, 64, lbl_kernel_name(k))) return false;
    const size_t WG = 64;
    size_t gws = (size_t)(N / stride) * WG;
    size_t lws = WG;
    cl_event* evt = KernelProfiler::event_for(lbl);
    cl_int err = clEnqueueNDRangeKernel(queue, k, 1, nullptr, &gws, &lws, 0, nullptr, evt);
    if (err != CL_SUCCESS) {
    if (!nnopt_validate_kernel_once(queue, k, lbl_kernel_name(k))) return false;
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
    s_gemv_m1_q4_no4_buf = clCreateKernel(s_gemv_m1_q4_prog, "gemv_m1_q4_no4_buf", &err);
    if (err != CL_SUCCESS) { s_gemv_m1_q4_no4_buf = nullptr; }

    // ── Per-device buffer-GEMV geometry (measured, not guessed) ─────────────
    // Built from a SELF-CONTAINED source, not the whole gemv_m1_q4.cl: redefining
    // WG_SIZE for the whole file broke the other kernels in it (clBuildProgram -6),
    // and compiling ~700 lines 1x per config is painfully slow on a Rogue compiler.
    //
    // The (WG_SIZE, Q4_NOUT) pair is what the on-device sweep (NNOPT_SWEEP=1)
    // actually measured fastest. The two GPU families want OPPOSITE geometry, so
    // no single hardcoded pair can serve both — lm_head N=65536 K=1024, ms/call:
    //
    //                    wg=64 nout=4   wg=32 nout=8
    //   Adreno 620            12.0        (nout=8 spills: 145 at wg=64)
    //   PowerVR GE8320        95.9         76.3   <- 20.5% faster
    //
    // A 1-CU Rogue part wants FEW threads per group and MORE outputs per thread;
    // Adreno wants the reverse. Re-run the sweep before changing these.
    {
        // Self-contained templated kernel, read from disk (a C++ string literal for
        // this was a needless escaping hazard, and got it wrong once).
        std::string tuned_src;
        {
            std::ifstream tf("kernels/gemv_m1_q4_tuned.cl", std::ios::binary);
            if (tf.is_open())
                tuned_src.assign((std::istreambuf_iterator<char>(tf)), std::istreambuf_iterator<char>());
        }
        const bool rogue = nnopt_device_is_denied_texture(device);
        s_q4_buf_is_rogue = rogue;

        auto build_variant = [&](int wg, int nout, int hm, int mode, int rn, Q4BufKernel& outk) {
            if (tuned_src.empty()) return;
            char topts[192];
            // Bit-trick defaults OFF (wrong on the GE8320 compiler, see dot8_bt).
            static const int bt = getenv("NNOPT_Q4_BITTRICK") ? 1 : 0;
            // ROW kernel load width: one uchar16 per 32-weight block instead of
            // four uchar4. Defaults OFF — MEASURED, and it does not pay.
            //
            // The NNOPT_STREAM_PAT probe made this look like the lever of the
            // whole project: reading the same bytes with the same thread count,
            // row-per-lane went 0.74 GB/s at a 4-byte load to 2.68 GB/s at a
            // 16-byte one (3.6x). In the real q4 GEMV it wins 2 of 6 shapes by
            // ~2% (inside run-to-run noise), loses the other 4, and RAISES
            // private memory from 136 B to 152 B.
            //
            // The gap is the whole finding: the probe's loop is ~0 ALU per byte,
            // so it exposes pure load throughput. The GEMV does 32 nibble
            // unpacks and 32 half MACs per 16 bytes, on a serial
            // mask->shift->convert->subtract->mul->add chain. It is bound by that
            // dependency chain, not by load width — which is why widening the
            // load buys nothing and why the coalescing tax (1.9x at 4 bytes) is
            // also not the binding constraint. Set NNOPT_ROW_VEC16=1 to re-A/B.
            static const int v16 = getenv("NNOPT_ROW_VEC16") ? 1 : 0;
            snprintf(topts, sizeof(topts),
                     "-DWG_SIZE=%d -DQ4_NOUT=%d -DQ4_HALF_MATH=%d -DQ4_BITTRICK=%d"
                     " -DROW_VEC16=%d -DROW_NOUT=%d -cl-fast-relaxed-math",
                     wg, nout, hm, bt, v16, rn);
            cl_program tp = OpenCLContext::build_cached_program_from_queue(queue, tuned_src, topts);
            if (!tp) return;
            cl_int te = CL_SUCCESS;
            cl_kernel kk = clCreateKernel(tp, mode ? "gemv_m1_q4_row_t" : "gemv_m1_q4_buf_t", &te);
            if (te == CL_SUCCESS && kk) { outk.k = kk; outk.wg = (size_t)wg; outk.nout = nout; outk.hm = hm; outk.bt = bt; outk.mode = mode; outk.rn = rn; }
        };

        if (rogue) {
            for (const Q4Geom& g : kQ4GeomRogue) {
                Q4BufKernel bk;
                build_variant(g.wg, g.nout, g.hm, g.mode, g.rn, bk);
                if (bk.k) s_q4_buf_by_shape[{g.N, g.K}] = bk;
            }
            build_variant(kQ4GeomRogueDefault.wg, kQ4GeomRogueDefault.nout,
                          kQ4GeomRogueDefault.hm, kQ4GeomRogueDefault.mode,
                          kQ4GeomRogueDefault.rn, s_q4_buf_default);
        } else {
            build_variant(64, 4, 0, 0, 1, s_q4_buf_default);   // Adreno: one geometry
        }
        fprintf(stderr, "NNOPT_Q4GEOM: %s | %zu per-shape kernels | default wg=%zu nout=%d hm=%d bt=%d mode=%d %s\n",
                rogue ? "Rogue/buffer" : "Adreno", s_q4_buf_by_shape.size(),
                s_q4_buf_default.wg, s_q4_buf_default.nout, s_q4_buf_default.hm,
                s_q4_buf_default.bt, s_q4_buf_default.mode, s_q4_buf_default.k ? "ok" : "FAILED");
        fflush(stderr);
    }    s_gemv_m1_k2560_no4_img_q4 = clCreateKernel(s_gemv_m1_q4_prog, "gemv_m1_k2560_q4_no4_img", &err);
    if (err != CL_SUCCESS) { s_gemv_m1_k2560_no4_img_q4 = nullptr; }
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
        cl_bool img_sup = CL_FALSE;
        clGetDeviceInfo(dev, CL_DEVICE_IMAGE_SUPPORT, sizeof(img_sup), &img_sup, nullptr);
        cl_uint pa = 0, ba = 0;
        clGetDeviceInfo(dev, CL_DEVICE_IMAGE_PITCH_ALIGNMENT,        sizeof(pa), &pa, nullptr);
        clGetDeviceInfo(dev, CL_DEVICE_IMAGE_BASE_ADDRESS_ALIGNMENT, sizeof(ba), &ba, nullptr);
        s_img_pitch_align = pa; s_img_base_align = ba;
        s_img_from_buffer_ok = nnopt_texture_path_allowed(dev, img_sup);
        s_img_validate_pending = s_img_from_buffer_ok;
        fprintf(stderr,
                "NNOPT_IMAGES: support=%d max_w=%zu max_h=%zu pitch_align=%u base_align=%u -> texture path %s\n",
                (int)img_sup, s_img_max_w, s_img_max_h, pa, ba,
                nnopt_no_images_forced() ? "FORCED OFF" : (s_img_from_buffer_ok ? "eligible" : "DISABLED"));
        fflush(stderr);
        s_img_limits_known = true;
    }
    if (!nnopt_image_row_ok((size_t)(a.K / 8))) {
        NNOPT_ERROR_FMT("q4 texture path unavailable (K/8=%d pitch_align=%zu) -> buffer path",
                        a.K/8, s_img_pitch_align);
        return false;
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


// ─────────────────────────────────────────────────────────────────────────────
// On-device Q4 GEMV sweep.
//
// A device iteration for the PowerVR work costs an APK upload plus a 270 MB model
// re-download (~5 min), so testing one kernel configuration per round trip is not
// viable. This builds gemv_m1_q4_buf_t once per (WG_SIZE, Q4_NOUT) pair, runs each
// against the REAL registered weights, times it, and checks its output against the
// shipped no4 kernel — so a single run measures the whole grid and says which
// configurations are both correct and fast on THIS hardware.
//
// Enable with NNOPT_SWEEP=1 (or -DNNOPT_SWEEP_DEFAULT_ON for an app build, which
// has no env plumbing of its own). Results go to stderr, which ProcessEngine
// mirrors into logcat under tag EdgiEngine.
static bool nnopt_sweep_enabled() {
    const char* e = getenv("NNOPT_SWEEP");
#ifdef NNOPT_SWEEP_DEFAULT_ON
    return !(e && e[0] == '0');
#else
    return e && e[0] != '0' && e[0] != '\0';
#endif
}


// Measure the device's real ALU ceiling for fp32 and fp16. See kernels/alu_probe.cl
// for why: on Rogue the q4 GEMV is ALU-bound, so the useful ceiling number is
// GFLOP/s, not the reported clock, and the fp16:fp32 ratio says whether the
// all-half dequant strategy is actually buying the 2x Imagination documents.
static void nnopt_alu_probe(cl_command_queue queue) {
    std::string src;
    { std::ifstream f("kernels/alu_probe.cl", std::ios::binary);
      if (!f.is_open()) { fprintf(stderr, "NNOPT_ALU: kernels/alu_probe.cl not found\n"); return; }
      src.assign((std::istreambuf_iterator<char>(f)), std::istreambuf_iterator<char>()); }

    cl_context ctx = nullptr; cl_device_id dev = nullptr;
    if (clGetCommandQueueInfo(queue, CL_QUEUE_CONTEXT, sizeof(ctx), &ctx, nullptr) != CL_SUCCESS) return;
    if (clGetCommandQueueInfo(queue, CL_QUEUE_DEVICE,  sizeof(dev), &dev, nullptr) != CL_SUCCESS) return;

    cl_uint mhz = 0, cus = 0;
    clGetDeviceInfo(dev, CL_DEVICE_MAX_CLOCK_FREQUENCY, sizeof(mhz), &mhz, nullptr);
    clGetDeviceInfo(dev, CL_DEVICE_MAX_COMPUTE_UNITS,   sizeof(cus), &cus, nullptr);

    cl_program prog = OpenCLContext::build_cached_program_from_queue(queue, src, "-cl-fast-relaxed-math");
    if (!prog) { fprintf(stderr, "NNOPT_ALU: probe build failed\n"); return; }

    const size_t GWS = 65536, LWS = 64;
    const int    ITERS = 512;
    const int    UNROLL = 8, CHAINS = 4;

    struct { const char* name; int lanes; const char* tag; } cases[] = {
        {"alu_probe_f32", 4, "fp32"},
        {"alu_probe_f16", 8, "fp16"},
    };
    double gf[2] = {0, 0};
    for (int ci = 0; ci < 2; ++ci) {
        cl_int e = CL_SUCCESS;
        cl_kernel k = clCreateKernel(prog, cases[ci].name, &e);
        if (!k || e != CL_SUCCESS) { fprintf(stderr, "NNOPT_ALU: %s create failed %d\n", cases[ci].tag, e); continue; }
        cl_mem o = clCreateBuffer(ctx, CL_MEM_READ_WRITE, GWS * 4, nullptr, &e);
        clSetKernelArg(k, 0, sizeof(cl_mem), &o);
        clSetKernelArg(k, 1, sizeof(int), &ITERS);
        // warm up, then time
        clEnqueueNDRangeKernel(queue, k, 1, nullptr, &GWS, &LWS, 0, nullptr, nullptr);
        if (clFinish(queue) != CL_SUCCESS) { fprintf(stderr, "NNOPT_ALU: %s dispatch failed\n", cases[ci].tag);
                                            clReleaseMemObject(o); clReleaseKernel(k); continue; }
        struct timespec t0, t1;
        clock_gettime(CLOCK_MONOTONIC, &t0);
        const int REPS = 5;
        for (int r = 0; r < REPS; ++r)
            clEnqueueNDRangeKernel(queue, k, 1, nullptr, &GWS, &LWS, 0, nullptr, nullptr);
        clFinish(queue);
        clock_gettime(CLOCK_MONOTONIC, &t1);
        const double sec = ((t1.tv_sec - t0.tv_sec) + (t1.tv_nsec - t0.tv_nsec) / 1e9) / REPS;
        // 2 flops per fma
        const double flops = (double)GWS * ITERS * UNROLL * CHAINS * cases[ci].lanes * 2.0;
        gf[ci] = flops / sec / 1e9;
        fprintf(stderr, "NNOPT_ALU: %s  %7.2f GFLOP/s  (%.3f ms/rep)\n", cases[ci].tag, gf[ci], sec * 1e3);
        clReleaseMemObject(o); clReleaseKernel(k);
    }
    fprintf(stderr, "NNOPT_ALU: reported clock %u MHz, %u CU | fp16:fp32 ratio %.2fx "
                    "(Rogue documents 2x; <1.3x means the all-half dequant is not paying off)\n",
            mhz, cus, gf[0] > 0 ? gf[1] / gf[0] : 0.0);
    fflush(stderr);
    clReleaseProgram(prog);
}

void nnopt_q4_sweep(cl_command_queue queue) {
    if (!nnopt_sweep_enabled()) return;
    nnopt_alu_probe(queue);
    if (s_q4_aux.empty()) { fprintf(stderr, "NNOPT_SWEEP: no q4 weights registered\n"); return; }
    // The q4 program is built lazily on first GEMV; the sweep runs before any
    // inference, so force it now or the reference kernel is null and every
    // configuration compares against zeros.
    if (!ensure_gemv_m1_q4_program(queue)) {
        fprintf(stderr, "NNOPT_SWEEP: q4 program build failed\n"); return;
    }

    cl_context ctx = nullptr; cl_device_id dev = nullptr;
    if (clGetCommandQueueInfo(queue, CL_QUEUE_CONTEXT, sizeof(ctx), &ctx, nullptr) != CL_SUCCESS) return;
    if (clGetCommandQueueInfo(queue, CL_QUEUE_DEVICE,  sizeof(dev), &dev, nullptr) != CL_SUCCESS) return;

    std::string src;
    {   // read the kernel source once
        FILE* f = fopen("kernels/gemv_m1_q4_tuned.cl", "rb");
        if (!f) { fprintf(stderr, "NNOPT_SWEEP: cannot open kernels/gemv_m1_q4_tuned.cl\n"); return; }
        char buf[8192]; size_t n;
        while ((n = fread(buf, 1, sizeof(buf), f)) > 0) src.append(buf, n);
        fclose(f);
    }

    // Sweep EVERY distinct (N,K) shape. Tuning only the lm_head left the MLP
    // projections running Adreno-derived geometry, and there is no reason the
    // optimum is the same at K=4608 as at K=1024 — different bytes-per-thread and
    // a different number of K-iterations per work-group.
    std::vector<const Q4Aux*> shapes;
    {
        std::set<std::pair<int,int>> seen;
        for (auto& kv : s_q4_aux) {
            const Q4Aux& a = kv.second;
            if (!a.W_q4 || !a.scale_root) continue;
            if (seen.insert({a.N, a.K}).second) shapes.push_back(&a);
        }
        std::sort(shapes.begin(), shapes.end(), [](const Q4Aux* x, const Q4Aux* y) {
            return (size_t)x->N * x->K > (size_t)y->N * y->K;
        });
    }
    if (shapes.empty()) return;

    {   // DVFS state at sweep time: the engine banner has reported both 400 and
        // 650 MHz on this device, which is a ~1.6x throughput swing before any
        // kernel work. Worth knowing which state a measurement was taken in.
        cl_uint mhz = 0, cus = 0;
        clGetDeviceInfo(dev, CL_DEVICE_MAX_CLOCK_FREQUENCY, sizeof(mhz), &mhz, nullptr);
        clGetDeviceInfo(dev, CL_DEVICE_MAX_COMPUTE_UNITS,   sizeof(cus), &cus, nullptr);
        fprintf(stderr, "NNOPT_SWEEP: device clock=%u MHz cus=%u shapes=%zu\n",
                mhz, cus, shapes.size());
        fflush(stderr);
    }

    for (const Q4Aux* big : shapes) {
    const int N = big->N, K = big->K;

    cl_int err = CL_SUCCESS;
    cl_mem xbuf = clCreateBuffer(ctx, CL_MEM_READ_WRITE, (size_t)K * 2, nullptr, &err);
    cl_mem obuf = clCreateBuffer(ctx, CL_MEM_READ_WRITE, (size_t)N * 2, nullptr, &err);
    cl_mem rbuf = clCreateBuffer(ctx, CL_MEM_READ_WRITE, (size_t)N * 2, nullptr, &err);
    if (!xbuf || !obuf || !rbuf) { fprintf(stderr, "NNOPT_SWEEP: scratch alloc failed\n"); return; }
    {   // deterministic activations so the correctness check is meaningful
        std::vector<uint16_t> h((size_t)K);
        for (int i = 0; i < K; ++i) h[i] = nnopt_f32_to_f16(((i % 17) - 8) * 0.05f);
        clEnqueueWriteBuffer(queue, xbuf, CL_TRUE, 0, h.size() * 2, h.data(), 0, nullptr, nullptr);
    }

    // Reference = the shipped kernel, already validated bit-identical to the image path.
    std::vector<uint16_t> ref((size_t)N, 0);
    if (s_gemv_m1_q4_no4_buf) {
        cl_kernel k = s_gemv_m1_q4_no4_buf;
        clSetKernelArg(k, 0, sizeof(cl_mem), &xbuf);
        clSetKernelArg(k, 1, sizeof(cl_mem), &big->W_q4);
        clSetKernelArg(k, 2, sizeof(cl_mem), &big->scale_root);
        clSetKernelArg(k, 3, sizeof(cl_mem), &rbuf);
        clSetKernelArg(k, 4, sizeof(int), &N);
        clSetKernelArg(k, 5, sizeof(int), &K);
        size_t g = (size_t)(N / 4) * 64, l = 64;
        if (clEnqueueNDRangeKernel(queue, k, 1, nullptr, &g, &l, 0, nullptr, nullptr) == CL_SUCCESS &&
            clFinish(queue) == CL_SUCCESS) {
            clEnqueueReadBuffer(queue, rbuf, CL_TRUE, 0, ref.size() * 2, ref.data(), 0, nullptr, nullptr);
        }
    }

    // image2d_t view over the SAME W bytes, for mode 2 (ROW over the texture path).
    // The probe measured 5.9 GB/s through image2d against 3.4 GB/s for buffers on
    // this part, and no ROW-mode kernel has ever been tried on it. Built here in
    // the sweep regardless of the texture denylist: the denylist is a decision
    // about the SHIPPING path, and the whole point is to measure whether that
    // decision is still right. Height is capped (8192 here), so a shape taller
    // than the cap gets no image and its mode-2 rows are skipped, not faked.
    cl_mem sweep_img = nullptr;
    {
        size_t max_h = 0, max_w = 0;
        clGetDeviceInfo(dev, CL_DEVICE_IMAGE2D_MAX_HEIGHT, sizeof(max_h), &max_h, nullptr);
        clGetDeviceInfo(dev, CL_DEVICE_IMAGE2D_MAX_WIDTH,  sizeof(max_w), &max_w, nullptr);
        if ((size_t)N <= max_h && (size_t)(K / 8) <= max_w) {
            cl_image_format fmt{};
            fmt.image_channel_order     = CL_RGBA;
            fmt.image_channel_data_type = CL_UNSIGNED_INT8;
            cl_image_desc d{};
            d.image_type   = CL_MEM_OBJECT_IMAGE2D;
            d.image_width  = (size_t)(K / 8);
            d.image_height = (size_t)N;
            d.buffer       = big->W_q4;
            cl_int ie = CL_SUCCESS;
            cl_context sctx = nullptr;
            clGetCommandQueueInfo(queue, CL_QUEUE_CONTEXT, sizeof(sctx), &sctx, nullptr);
            sweep_img = clCreateImage(sctx, CL_MEM_READ_ONLY, &fmt, &d, nullptr, &ie);
            if (ie != CL_SUCCESS) {
                fprintf(stderr, "NNOPT_SWEEP: image2d view unavailable (err=%d) - mode 2 skipped\n", ie);
                sweep_img = nullptr;
            }
        } else {
            fprintf(stderr, "NNOPT_SWEEP: N=%d exceeds image max_h=%zu - mode 2 skipped (needs tiling)\n",
                    N, max_h);
        }
    }

    // Bytes the kernel must actually move per call. NOTE: on Rogue this GB/s column
    // is NOT the binding constraint — the full-model budget lands at 0.47 GB/s against
    // a ~12.8 GB/s LPDDR4X peak (3.7%), so the kernel is ALU-bound in the dequant, not
    // bandwidth-bound. Adreno is the opposite. Keep the column as a ceiling reference.
    const double bytes_per_call =
        (double)N * (K / 2)          // packed q4 weights (2 per byte)
      + (double)N * (K / 32) * 2.0   // fp16 per-block scales
      + (double)K * 2.0;             // fp16 activations
    fprintf(stderr, "NNOPT_SWEEP: shape N=%d K=%d  (%.1f MB/call)\n",
            N, K, bytes_per_call / 1048576.0);
    fprintf(stderr, "NNOPT_SWEEP: MODE  WG  NOUT  HM  BT  RU V16  RN   local_B  priv_B    ms/call     GB/s  err/rms status\n");
    fflush(stderr);

    // Explicit config list rather than a full cross product: the Rogue compiler is
    // slow. NOTE the old claim here — "wg 128/256 lost by 3-10x at every shape" —
    // was measured against the REDUCTION kernel only and is false for ROW mode:
    // on 2026-09-14 wg=256 won (1024,1024) at 1.192 ms against wg=128's 1.651 ms.
    // Treat per-shape geometry as re-measurable, never as settled.
    //   mode 0 = split-row reduction kernel (gemv_m1_q4_buf_t)
    //   mode 1 = one-lane-per-output-row  (gemv_m1_q4_row_t), no barriers, no
    //            __local, and one scale op per 32 weights instead of per 8
    struct SweepCfg { int wg, nout, hm, bt, mode, ru, v16, rn; };
    std::vector<SweepCfg> cfgs;
    for (int wg : {32, 64}) for (int no : {4, 8}) cfgs.push_back({wg, no, 0, 0, 0, 4, 0, 1});  // reduction
    for (int wg : {64})                            cfgs.push_back({wg, 8, 0, 1, 0, 4, 0, 1});  // bit-trick A/B
    // ROW mode. wg=512 is in the grid because the 2026-09-14 re-measure had wg=256
    // winning at the EDGE of the old {32,64,128,256} grid for (2560,1024) and
    // (1024,1024) — an optimum at a grid boundary is not an optimum, it is an
    // untested direction. 512 is CL_DEVICE_MAX_WORK_GROUP_SIZE on this part, so
    // this closes the axis. (Kernels whose wg exceeds CL_KERNEL_WORK_GROUP_SIZE
    // are skipped with a WG>kernel_max row rather than run narrower — running a
    // hardcoded-lane kernel narrow gives WRONG numbers, not slow ones.)
    for (int wg : {32, 64, 128, 256, 512})        cfgs.push_back({wg, 1, 0, 0, 1, 4, 0, 1});  // ROW mode
    // Spill axis: same ROW kernel, less inner unroll. Only at the wg values that
    // actually won on 2026-09-14, so the grid stays cheap on a slow compiler.
// ROW_UNROLL closed 2026-09-14: ru=4 won everywhere (ru=1 cuts private 136->104 B
    // but runs 1.7x slower). Axis removed from the grid.
    // Load-width A/B — the lever the NNOPT_STREAM_PAT probe identified. Reading
    // an identical 256 MB with an identical thread count and only the per-lane
    // load width changed, row-per-lane went 0.74 GB/s at 4 bytes -> 1.34 at 8 ->
    // 2.68 at 16 (3.6x), and the strided-vs-coalesced penalty collapsed from
    // 1.94x to 0.94x. Swept across the full wg range because a wider load also
    // changes register pressure, and therefore the best group size.
    for (int wg : {32, 64, 128, 256, 512})
                                                   cfgs.push_back({wg, 1, 0, 0, 1, 4, 1, 1});  // ROW uchar16
    // Rows-per-lane A/B — the lever round 3's diagnosis points at. The kernel is
    // bound by the serial dequant chain, not by loads (widening the load to 16 B
    // was wall-neutral) and not by FLOPs (0.9% of the measured fp16 rate). R>1
    // gives each lane R INDEPENDENT chains to interleave and reuses one loaded
    // activation block across all R rows. Verified bit-exact against R=1 on a
    // host OpenCL device across all six live shapes plus N=513/1022/7 tails.
    for (int wg : {32, 64, 128, 256}) for (int rn : {2, 4})
                                                   cfgs.push_back({wg, 1, 0, 0, 1, 4, 0, rn});  // ROW multi-row
    // THE CROSS. vec16 was only ever measured at rn=1 and rn>1 only at vec16=0 —
    // and until now the rn>1 path had no wide-load variant at all, so the pair was
    // unimplementable, not just unmeasured. Separately: the wide load was
    // wall-neutral (one lane with one serial dequant chain has nothing to hide the
    // latency behind) and rn>1 won with narrow loads. Together the wide load
    // supplies the bandwidth the probe proved is there (0.74 -> 2.68 GB/s at 16 B)
    // and rn supplies the independent work to consume it: rn 16-byte loads in
    // flight per block, IR-verified. Bit-exact vs rn=1 on a host OpenCL device.
    for (int wg : {32, 64, 128, 256}) for (int rn : {2, 4})
                                                   cfgs.push_back({wg, 1, 0, 0, 1, 4, 1, rn});  // ROW multi-row x uchar16
// mode 2 (ROW over image2d_t) closed 2026-09-14: best 1.147 ms vs 0.764 ms for the
    // buffer path at N=512, correct but ~50% slower at every shape that fit the 8192
    // image height cap. The 5.9 GB/s image STREAM number does not survive a kernel
    // with a dependent dequant chain. Kernel kept; axis removed from the grid.
    double best_ms = 1e30; int best_wg = 0, best_no = 0, best_hm = 0, best_bt = 0, best_mode = 0, best_ru = 4, best_v16 = 0, best_rn = 1;

    for (const SweepCfg& cfg : cfgs) {
        const int wg = cfg.wg, no = cfg.nout, hm = cfg.hm, bt = cfg.bt, mode = cfg.mode, ru = cfg.ru, v16 = cfg.v16, rn = cfg.rn;
        if (mode == 0 && N % no != 0) continue;   // reduction mode tiles N by nout
        if (mode == 2 && !sweep_img) continue;    // no image view for this shape
        char opts[256];
        snprintf(opts, sizeof(opts),
                 "-DWG_SIZE=%d -DQ4_NOUT=%d -DQ4_HALF_MATH=%d -DQ4_BITTRICK=%d -DROW_UNROLL=%d"
                 " -DROW_VEC16=%d -DROW_NOUT=%d -cl-fast-relaxed-math",
                 wg, no, hm, bt, ru, v16, rn);
        cl_program prog = OpenCLContext::build_cached_program_from_queue(queue, src, opts);
        if (!prog) { fprintf(stderr, "NNOPT_SWEEP: %4s %4d %4d        -          -     -    BUILD FAIL\n", mode == 2 ? "IMG" : mode ? "ROW" : "RED", wg, no); continue; }
        const char* kname = (mode == 2) ? "gemv_m1_q4_rowimg_t"
                    : (mode == 1) ? "gemv_m1_q4_row_t" : "gemv_m1_q4_buf_t";
        cl_kernel k = clCreateKernel(prog, kname, &err);
        if (!k || err != CL_SUCCESS) { fprintf(stderr, "NNOPT_SWEEP: %4s %4d %4d        -          -     -    KERNEL FAIL\n", mode == 2 ? "IMG" : mode ? "ROW" : "RED", wg, no); clReleaseProgram(prog); continue; }

        cl_ulong lmem = 0;
        clGetKernelWorkGroupInfo(k, dev, CL_KERNEL_LOCAL_MEM_SIZE, sizeof(lmem), &lmem, nullptr);
        size_t kmax = 0;
        clGetKernelWorkGroupInfo(k, dev, CL_KERNEL_WORK_GROUP_SIZE, sizeof(kmax), &kmax, nullptr);
        if (kmax < (size_t)wg) {
            fprintf(stderr, "NNOPT_SWEEP: %4s %4d %4d  %8llu          -     -    WG>kernel_max(%zu)\n",
                    mode == 2 ? "IMG" : mode ? "ROW" : "RED", wg, no, (unsigned long long)lmem, kmax);
            clReleaseKernel(k); clReleaseProgram(prog); continue;
        }

        clSetKernelArg(k, 0, sizeof(cl_mem), &xbuf);
        // mode 2 reads the same bytes through an image2d_t view instead of the buffer.
        if (mode == 2) clSetKernelArg(k, 1, sizeof(cl_mem), &sweep_img);
        else           clSetKernelArg(k, 1, sizeof(cl_mem), &big->W_q4);
        clSetKernelArg(k, 2, sizeof(cl_mem), &big->scale_root);
        clSetKernelArg(k, 3, sizeof(cl_mem), &obuf);
        clSetKernelArg(k, 4, sizeof(int), &N);
        clSetKernelArg(k, 5, sizeof(int), &K);
        size_t gws = mode ? nnopt_round_up((size_t)((N + rn - 1) / rn), (size_t)wg)
                          : (size_t)(N / no) * (size_t)wg;   // modes 1 and 2 are both row-per-lane
        size_t lws = (size_t)wg;

        cl_int e = clEnqueueNDRangeKernel(queue, k, 1, nullptr, &gws, &lws, 0, nullptr, nullptr);
        if (e == CL_SUCCESS) e = clFinish(queue);
        if (e != CL_SUCCESS) {
            fprintf(stderr, "NNOPT_SWEEP: %4s %4d %4d  %8llu          -     -    DISPATCH FAIL %d\n",
                    mode == 2 ? "IMG" : mode ? "ROW" : "RED", wg, no, (unsigned long long)lmem, e);
            clReleaseKernel(k); clReleaseProgram(prog); continue;
        }

        // 5 reps: the slowest PowerVR configs take ~1.8 s/call, so 20 reps x 16 configs
        // x 6 shapes would run for many minutes on a device session that costs money.
        const int REPS = 5;
        struct timespec t0, t1;
        clock_gettime(CLOCK_MONOTONIC, &t0);
        for (int r = 0; r < REPS; ++r) clEnqueueNDRangeKernel(queue, k, 1, nullptr, &gws, &lws, 0, nullptr, nullptr);
        clFinish(queue);
        clock_gettime(CLOCK_MONOTONIC, &t1);
        const double ms = ((t1.tv_sec - t0.tv_sec) * 1e3 + (t1.tv_nsec - t0.tv_nsec) / 1e6) / REPS;

        std::vector<uint16_t> got((size_t)N, 0);
        clEnqueueReadBuffer(queue, obuf, CL_TRUE, 0, got.size() * 2, got.data(), 0, nullptr, nullptr);
        // Error normalised to the RMS of the reference, NOT per-element relative
        // error. A q4 GEMV row can land near zero, and there a rounding-level
        // absolute difference produces an unbounded RELATIVE one -- the old gate
        // reported rel=1.770 and rejected EVERY fp16 configuration, which would
        // have had the sweep hand back a geometry table that disables the
        // fp16 fast paths on correctness grounds that were an artefact of the
        // metric. Normalising by signal magnitude is the standard comparison for
        // a reduced-precision kernel and is what the accuracy claim actually means.
        double sumsq = 0.0, maxabs = 0.0;
        for (int i = 0; i < N; ++i) {
            const float a = nnopt_f16_to_f32(ref[i]), b = nnopt_f16_to_f32(got[i]);
            sumsq  += (double)a * (double)a;
            maxabs  = fmax(maxabs, (double)fabsf(a - b));
        }
        const double rms = sqrt(sumsq / (double)N);
        const double maxrel = rms > 0.0 ? maxabs / rms : 0.0;
        const bool ok = maxrel < 2e-2;
        if (ok && ms < best_ms) { best_ms = ms; best_wg = wg; best_no = no; best_hm = hm; best_bt = bt; best_mode = mode; best_ru = ru; best_v16 = v16; best_rn = rn; }
        const double gbs = bytes_per_call / (ms * 1e-3) / 1e9;
        cl_ulong pmem = 0;
        clGetKernelWorkGroupInfo(k, dev, CL_KERNEL_PRIVATE_MEM_SIZE, sizeof(pmem), &pmem, nullptr);
        fprintf(stderr, "NNOPT_SWEEP: %4s %4d %4d %3d %3d %3d %3d %3d  %8llu %7llu   %8.3f  %7.2f  %5.3f  %s\n",
                mode == 2 ? "IMG" : mode ? "ROW" : "RED", wg, no, hm, bt, ru, v16, rn, (unsigned long long)lmem,
                (unsigned long long)pmem, ms, gbs, maxrel, ok ? "ok" : "MISMATCH");
        fflush(stderr);
        clReleaseKernel(k); clReleaseProgram(prog);
    }

    if (best_wg) {
        fprintf(stderr, "NNOPT_SWEEP: BEST N=%d K=%d -> {%d, %d, %d, %d, %d, %d}  (N,K,wg,nout,hm,mode)"
                        "  bittrick=%d  unroll=%d  vec16=%d  rownout=%d  %.3f ms  %.2f GB/s\n",
                N, K, N, K, best_wg, best_no, best_hm, best_mode,
                best_bt, best_ru, best_v16, best_rn, best_ms, bytes_per_call / (best_ms * 1e-3) / 1e9);
    }
    fflush(stderr);
    if (sweep_img) clReleaseMemObject(sweep_img);
    clReleaseMemObject(xbuf); clReleaseMemObject(obuf); clReleaseMemObject(rbuf);
    }   // end per-shape loop
}

// Q4 image-path GEMV dispatch. Returns false on miss (caller falls through to
// int8 if registered there, otherwise to fp16).
static bool run_gemv_m1_image_q4(cl_command_queue queue, int N, int K, cl_mem W, cl_mem x, cl_mem out) {
    auto it = s_q4_aux.find(W);
    if (it == s_q4_aux.end()) return false;
    if (!ensure_gemv_m1_q4_program(queue)) return false;

    // ── Buffer path: no image2d_t at all ────────────────────────────────────
    // Taken when the texture path is unavailable for this device (PowerVR Rogue,
    // or NNOPT_NO_IMAGES). Without this there is NO q4 kernel on such a device —
    // every other kernel in gemv_m1_q4.cl is an _img variant — so q4 fell through
    // to code that reads W as fp16 and emitted garbage. Reads the same packed
    // buffer the image would have aliased, so the arithmetic is identical.
    if (!nnopt_image_row_ok((size_t)(K / 8))) {
        const Q4Aux& a = it->second;
        cl_mem Wq = a.W_q4 ? a.W_q4 : W;
        cl_mem sc = a.scale_root ? a.scale_root : a.scale;
        // Prefer the per-device tuned kernel (see NNOPT_Q4GEOM); fall back to the
        // fixed wg=64/nout=4 kernel if it failed to build.
        // Pick the geometry measured fastest for THIS shape (see Q4BufKernel above).
        auto shape_it = s_q4_buf_by_shape.find({N, K});
        const Q4BufKernel& pick =
            (shape_it != s_q4_buf_by_shape.end() && shape_it->second.k) ? shape_it->second
                                                                       : s_q4_buf_default;
        cl_kernel kbuf = pick.k ? pick.k : s_gemv_m1_q4_no4_buf;
        const size_t bwg  = pick.k ? pick.wg   : 64;
        const int    bnout= pick.k ? pick.nout : 4;
        if (!kbuf || !Wq || !sc || (K % 32) != 0) return false;
        if (!pick.mode && (N % bnout) != 0) return false;   // reduction mode tiles N by nout
        if (nnopt_kernel_is_dead(kbuf)) return false;
        if (!nnopt_kernel_supports_wg(queue, kbuf, bwg, lbl_kernel_name(kbuf))) return false;
        cl_kernel k = kbuf;
        clSetKernelArg(k, 0, sizeof(cl_mem), &x);
        clSetKernelArg(k, 1, sizeof(cl_mem), &Wq);
        clSetKernelArg(k, 2, sizeof(cl_mem), &sc);
        clSetKernelArg(k, 3, sizeof(cl_mem), &out);
        clSetKernelArg(k, 4, sizeof(int),    &N);
        clSetKernelArg(k, 5, sizeof(int),    &K);
        const size_t WG = bwg;
        // ROW mode: one work-item per output row, so the global size is N padded
        // up to a whole work-group (the kernel bounds-checks n >= N). Reduction
        // mode: one work-group per Q4_NOUT rows.
        size_t gws = pick.mode ? nnopt_round_up((size_t)((N + pick.rn - 1) / pick.rn), WG)
                               : (size_t)(N / bnout) * WG;
        size_t lws = WG;
        char lbl[80]; snprintf(lbl, sizeof(lbl), "gemv_m1_K%d_N%d_q4buf_wg%zu_no%d_hm%d_m%d",
                               K, N, bwg, bnout, pick.hm, pick.mode);
        cl_event* evt = KernelProfiler::event_for(lbl);
        cl_int err = clEnqueueNDRangeKernel(queue, k, 1, nullptr, &gws, &lws, 0, nullptr, evt);
        if (err != CL_SUCCESS) { NNOPT_ERROR_FMT("q4 buf dispatch: %d (N=%d K=%d)", err, N, K); return false; }
        if (!nnopt_validate_kernel_once(queue, k, lbl_kernel_name(k))) return false;
        return true;
    }

    if (!prepare_q4_entry(queue, W)) return false;

    auto pick_kernel = [&](int Nlocal) -> std::pair<cl_kernel,int> {
        if (K == 1024 && (Nlocal % 4) == 0 && s_gemv_m1_k1024_no4_img_q4) return {s_gemv_m1_k1024_no4_img_q4, 4};
        // K=1024 _no8 variant exists in the kernel file (gemv_m1_k1024_q4_no8_img)
        // but measured −38% decode on Tab A9+: 8 fp32 acc + 8 unpacks of 8 weights
        // each spilled the register file hard on Adreno 619 v2. Stick with no4.
        // K=4608 _no8 also off-table — int8 no8 spilled at this K (T2 −6.5%);
        // Q4 has more ALU per byte, would be worse.
        if (K == 4608 && (Nlocal % 4) == 0 && s_gemv_m1_k4608_no4_img_q4) return {s_gemv_m1_k4608_no4_img_q4, 4};
        // K=2560 is the LFM2.5-230M w2 down-proj (80 q4 blocks/row).
        if (K == 2560 && (Nlocal % 4) == 0 && s_gemv_m1_k2560_no4_img_q4) return {s_gemv_m1_k2560_no4_img_q4, 4};
        return {nullptr, 0};
    };

    Q4Aux& a = it->second;

    if (!a.tiles.empty()) {
        cl_kernel retire_kernel = nullptr;   // retired only after ALL tiles pass
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
            // Skip this kernel if the device cannot run it at WG=64 (see nnopt_kernel_supports_wg).
            if (nnopt_kernel_is_dead(kt)) return false;
        if (!nnopt_kernel_supports_wg(queue, kt, 64, lbl_kernel_name(kt))) return false;
            const size_t WG = 64;
            size_t gws = (size_t)(tile_n / stride_t) * WG;
            size_t lws = WG;
            char lbl[64]; snprintf(lbl, sizeof(lbl), "gemv_m1_K%d_N%d_no%d_img_q4_tile", K, tile_n, stride_t);
            cl_event* evt = KernelProfiler::event_for(lbl);
            err = clEnqueueNDRangeKernel(queue, kt, 1, nullptr, &gws, &lws, 0, nullptr, evt);
            clReleaseMemObject(out_sub);
            if (err != CL_SUCCESS) { NNOPT_ERROR_FMT("q4 tile enqueue: %d row=%d", err, t.row_offset); return false; }
            if (!nnopt_validate_kernel_dispatch(queue, kt, lbl_kernel_name(kt), /*mark_ok=*/false)) return false;
            retire_kernel = kt;
        }
        // Whole tiled call survived — retire the kernel so we stop clFinish-ing.
        if (retire_kernel) s_kernel_ok.insert(retire_kernel);
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
    // Skip this kernel if the device cannot run it at WG=64 (see nnopt_kernel_supports_wg).
    if (nnopt_kernel_is_dead(k)) return false;
    if (!nnopt_kernel_supports_wg(queue, k, 64, lbl_kernel_name(k))) return false;
    const size_t WG = 64;
    size_t gws = (size_t)(N / stride) * WG;
    size_t lws = WG;
    cl_event* evt = KernelProfiler::event_for(lbl);
    cl_int err = clEnqueueNDRangeKernel(queue, k, 1, nullptr, &gws, &lws, 0, nullptr, evt);
    if (err != CL_SUCCESS) { NNOPT_ERROR_FMT("gemv_m1_image_q4 enqueue: %d K=%d N=%d", err, K, N); return false; }
    if (!nnopt_validate_kernel_once(queue, k, lbl_kernel_name(k))) return false;
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
        cl_bool img_sup = CL_FALSE;
        clGetDeviceInfo(dev, CL_DEVICE_IMAGE_SUPPORT, sizeof(img_sup), &img_sup, nullptr);
        cl_uint pa = 0, ba = 0;
        clGetDeviceInfo(dev, CL_DEVICE_IMAGE_PITCH_ALIGNMENT,        sizeof(pa), &pa, nullptr);
        clGetDeviceInfo(dev, CL_DEVICE_IMAGE_BASE_ADDRESS_ALIGNMENT, sizeof(ba), &ba, nullptr);
        s_img_pitch_align = pa; s_img_base_align = ba;
        s_img_from_buffer_ok = nnopt_texture_path_allowed(dev, img_sup);
        s_img_validate_pending = s_img_from_buffer_ok;
        fprintf(stderr,
                "NNOPT_IMAGES: support=%d max_w=%zu max_h=%zu pitch_align=%u base_align=%u -> texture path %s\n",
                (int)img_sup, s_img_max_w, s_img_max_h, pa, ba,
                nnopt_no_images_forced() ? "FORCED OFF" : (s_img_from_buffer_ok ? "eligible" : "DISABLED"));
        fflush(stderr);
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
        // Device must actually support image2d-from-buffer at this row pitch;
        // otherwise creation succeeds and the kernel faults on read.
        if (!nnopt_image_row_ok(pix_w)) return nullptr;
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
        // Skip this kernel if the device cannot run it at WG=64 (see nnopt_kernel_supports_wg).
        if (nnopt_kernel_is_dead(k)) return false;
        if (nnopt_kernel_is_dead(k)) return false;
    if (!nnopt_kernel_supports_wg(queue, k, 64, lbl_kernel_name(k))) return false;
        const size_t WG = 64;
        size_t gws = (size_t)(N / stride) * WG;
        size_t lws = WG;
        cl_event* evt = KernelProfiler::event_for(lbl);
        err = clEnqueueNDRangeKernel(queue, k, 1, nullptr, &gws, &lws, 0, nullptr, evt);
        if (err != CL_SUCCESS) { NNOPT_ERROR_FMT("gemv_m1_image enqueue: %d (K=%d N=%d)", err, K, N); return false; }
        if (!nnopt_validate_kernel_dispatch(queue, k, lbl_kernel_name(k), /*mark_ok=*/false)) return false;
        if (k) s_kernel_ok.insert(k);
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
        // Skip this kernel if the device cannot run it at WG=64 (see nnopt_kernel_supports_wg).
        if (nnopt_kernel_is_dead(k)) return false;
        if (nnopt_kernel_is_dead(k)) return false;
    if (!nnopt_kernel_supports_wg(queue, k, 64, lbl_kernel_name(k))) return false;
        const size_t WG = 64;
        size_t gws = (size_t)(tile_n / stride) * WG;
        size_t lws = WG;
        cl_event* evt = KernelProfiler::event_for(lbl);
        err = clEnqueueNDRangeKernel(queue, k, 1, nullptr, &gws, &lws, 0, nullptr, evt);
        clReleaseMemObject(out_sub);
        if (err != CL_SUCCESS) { NNOPT_ERROR_FMT("tile dispatch: %d", err); return false; }
        if (!nnopt_validate_kernel_dispatch(queue, k, lbl_kernel_name(k), /*mark_ok=*/false)) return false;
    }
    // Whole tiled call survived — now it is safe to stop checking this kernel.
    if (k) s_kernel_ok.insert(k);
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
    const size_t lws_ea_ip = nnopt_pick_lws(queue);
    size_t gws = nnopt_round_up(n, lws_ea_ip);   // kernel guards with `if (gid < n)`
    cl_event* evt = KernelProfiler::event_for("element_add_inplace");
    err = clEnqueueNDRangeKernel(queue, s_cached_kernel, 1, nullptr, &gws, &lws_ea_ip, 0, nullptr, evt);
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

    const size_t lws_ea = nnopt_pick_lws(queue);
    size_t global_size = nnopt_round_up(n, lws_ea);   // kernel guards with `if (gid < n)`
    cl_event* ea_evt = KernelProfiler::event_for("element_add");
    err = clEnqueueNDRangeKernel(queue, kernel, 1, nullptr, &global_size, &lws_ea, 0, nullptr, ea_evt);
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

    const size_t lws_sl = nnopt_pick_lws(queue);
    size_t global_size = nnopt_round_up((size_t)rows * (size_t)half_cols, lws_sl);  // kernel guards with `if (gid >= total) return;`
    cl_event* sl_evt = KernelProfiler::event_for("split_last_dim_2");
    err = clEnqueueNDRangeKernel(queue, kernel, 1, nullptr, &global_size, &lws_sl, 0, nullptr, sl_evt);
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
        // NOTE: no nnopt_image_dispatch_ok() here — it returns the GLOBAL texture flag,
        // which is false whenever textures are off, and that discarded a successful
        // BUFFER dispatch. Per-kernel validation inside run_gemv_m1_image_q4() covers this.
        if (run_gemv_m1_image_q4(queue, N, K, W, x, out)) {
            NNOPT_DEBUG_SYNC(queue);
            return true;
        }
        // CORRECTNESS: every q4 GEMV in gemv_m1_q4.cl is an _img kernel, so if the
        // texture path is unavailable there is NO q4 kernel to fall back to. The
        // paths below all read W as fp16, which silently reinterprets q4-packed
        // bytes and emits reserved-token garbage (seen on PowerVR GE8320). Fail
        // loudly instead: a q4 weight has no non-texture implementation.
        if (s_q4_aux.find(W) != s_q4_aux.end()) {
            NNOPT_ERROR_FMT("q4 weight N=%d K=%d has no non-texture GEMV — refusing to "
                            "reinterpret q4 bytes as fp16. Use the fp16 model on this device.",
                            N, K);
            return false;
        }
        // Int8 image path: half the weight bytes, hits the texture L1. Only
        // dispatched if `nnopt_register_int8_weight` registered W; otherwise
        // falls through to the fp16 image kernels below.
        // NOTE: no nnopt_image_dispatch_ok() here — it returns the GLOBAL texture flag,
        // which is false whenever textures are off, and that discarded a successful
        // BUFFER dispatch. Per-kernel validation inside run_gemv_m1_image_int8() covers this.
        if (run_gemv_m1_image_int8(queue, N, K, W, x, out)) {
            NNOPT_DEBUG_SYNC(queue);
            return true;
        }
        // Phase 2: Adreno texture-cache image2d_t path (1.71× faster L1 vs L2 on Razr 2020,
        // measured 13.46 vs 7.85 GB/s in --bw-probe). Tries no4 first, no2 fallback for
        // K=1024 (lower register pressure if no4 spills), tiles for lm_head N=65536.
        // NOTE: no nnopt_image_dispatch_ok() here — it returns the GLOBAL texture flag,
        // which is false whenever textures are off, and that discarded a successful
        // BUFFER dispatch. Per-kernel validation inside run_gemv_m1_image() covers this.
        if (run_gemv_m1_image(queue, N, K, W, x, out)) {
            NNOPT_DEBUG_SYNC(queue);
            return true;
        }
        // If the texture dispatch just died, images are now disabled and we fall
        // through to run_gemv_m1_no4 / run_gemv_rT_fp32acc, which fully overwrite
        // `out` from the buffer — no partial texture result survives.
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

    // Skip this kernel if the device cannot run it at WG=64 (see nnopt_kernel_supports_wg).
    if (nnopt_kernel_is_dead(k)) return false;
    if (!nnopt_kernel_supports_wg(queue, k, 64, lbl_kernel_name(k))) return false;
    const size_t WG = 64;
    size_t gws = (size_t)(N / 8) * WG;
    size_t lws = WG;
    cl_event* evt = KernelProfiler::event_for("gemv_m1_K1024_no8_silufused_img");
    cl_int err = clEnqueueNDRangeKernel(queue, k, 1, nullptr, &gws, &lws, 0, nullptr, evt);
    if (err != CL_SUCCESS) {
    if (!nnopt_validate_kernel_once(queue, k, lbl_kernel_name(k))) return false;
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



// ── Kernel dispatch trace implementation (see debug_utils.h) ────────────────
// Defined with the macro undefined so the inner call reaches the real OpenCL
// entry point rather than recursing into this wrapper.
#undef clEnqueueNDRangeKernel
cl_int nnopt_enqueue_ndrange_traced(
    const char* file, int line,
    cl_command_queue queue, cl_kernel kernel, cl_uint work_dim,
    const size_t* global_offset, const size_t* global_size, const size_t* local_size,
    cl_uint num_wait, const cl_event* wait_list, cl_event* event) {

    const cl_int err = clEnqueueNDRangeKernel(queue, kernel, work_dim, global_offset,
                                              global_size, local_size, num_wait, wait_list, event);
    if (!nnopt_kernel_trace_enabled()) return err;

    // Announce once. Without this a clean log is ambiguous: "armed and nothing
    // failed" and "never armed" both look like silence.
    static bool announced = false;
    if (!announced) {
        announced = true;
        fprintf(stderr, "NNOPT_TRACE: armed — clFinish + error check after every dispatch\n");
        fflush(stderr);
    }

    char kname[128];
    kname[0] = '\0';
    clGetKernelInfo(kernel, CL_KERNEL_FUNCTION_NAME, sizeof(kname) - 1, kname, nullptr);

    // Report the work sizes too: an oversized local size is one of the ways a
    // dispatch that is fine on Adreno gets refused on a 512-max device.
    char dims[128];
    int n = snprintf(dims, sizeof(dims), "gws=[");
    for (cl_uint i = 0; i < work_dim && n < (int)sizeof(dims); ++i)
        n += snprintf(dims + n, sizeof(dims) - n, "%zu%s", global_size[i], i + 1 < work_dim ? "," : "");
    n += snprintf(dims + n, sizeof(dims) - n, "] lws=");
    if (local_size) {
        n += snprintf(dims + n, sizeof(dims) - n, "[");
        for (cl_uint i = 0; i < work_dim && n < (int)sizeof(dims); ++i)
            n += snprintf(dims + n, sizeof(dims) - n, "%zu%s", local_size[i], i + 1 < work_dim ? "," : "");
        snprintf(dims + n, sizeof(dims) - n, "]");
    } else {
        snprintf(dims + n, sizeof(dims) - n, "auto");
    }

    if (err != CL_SUCCESS) {
        fprintf(stderr, "NNOPT_TRACE: ENQUEUE FAILED kernel='%s' %s:%d err=%d %s\n",
                kname, file, line, err, dims);
        fflush(stderr);
        return err;
    }

    const cl_int fin = clFinish(queue);
    if (fin != CL_SUCCESS) {
        fprintf(stderr, "NNOPT_TRACE: KERNEL DIED kernel='%s' %s:%d clFinish=%d %s\n",
                kname, file, line, fin, dims);
        fflush(stderr);
    }
    return err;
}
