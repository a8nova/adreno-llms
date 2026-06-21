// Reference: model_info/transformers_src/modeling_musicgen.py:250-344 MusicgenDecoderLayer.forward
//
// Host orchestrator for the per-decoder-layer MEGAKERNEL (kernels/decoder_layer_mega.cl).
// Collapses ~24 dispatches/layer into ONE dispatch per CFG row (seq=1 decode),
// with fp32 accumulation throughout (AGENT_DIRECTIVE_FP32_ACCUM.md — the speed
// fix AND the audio-noise fix). Cross-attention K/V are PRECOMPUTED once per
// generation (decoder_cross_kv_precompute) so the per-step 11-token recompute is
// gone.
//
// Gating: NNOPT_MEGA_LAYERS=n routes layers [0, n) through this megakernel; the
// rest stay on the validated M=2 path. n is read once by the backbone.
//
// Weight layout (all fp16, row-major [out,in], bias-free except LN gamma/beta):
//   self_attn_layer_norm.{weight,bias}    self_attn.{q,k,v,out}_proj.weight
//   encoder_attn_layer_norm.{weight,bias} encoder_attn.{q,k,v,out}_proj.weight
//   final_layer_norm.{weight,bias}        fc1.weight  fc2.weight

#include <chrono>
#include "../opencl_context.h"
#include "../weights.h"
#include "../debug_utils.h"
#include "../model_config.h"
#include "../utils.h"
#include "../profiler.h"   // KernelProfiler::event_for — dormant unless NNOPT_PROFILE=1.

#include <CL/cl.h>
#include <string>
#include <cstdio>
#include <cmath>
#include <vector>
#include <numeric>
#include <unordered_map>   // KV-buffer → image2d-view cache (NNOPT_ATTN_TEX)
#include <algorithm>
#include <cstdint>

// ── Lever 1: workgroup size (occupancy + coalescing) ────────────────────────
// The megakernel runs ONE workgroup per CFG row. At MEGA_WG=256 it was both
// LOW-OCCUPANCY (2 WGs total can't saturate the Adreno memory bus) and the GEMV
// reads were stride-K UNCOALESCED. The kernel now uses a ROW-COOPERATIVE
// coalesced GEMV (MEGA_KGROUP threads per output row read consecutive K) and we
// raise the workgroup size so more threads stream weights in parallel.
// kMegaWG MUST equal the kernel's MEGA_WG (passed via -D MEGA_WG=N at build).
// NNOPT_MEGA_WG overrides at runtime (A/B different sizes on the same source).
static int mega_wg_size() {
    static int wg = [](){
        const char* e = std::getenv("NNOPT_MEGA_WG");
        int v = e ? std::atoi(e) : 512;
        if (v != 256 && v != 512 && v != 1024) v = 512;
        return v;
    }();
    return wg;
}
#define kMegaWG (mega_wg_size())

// MEGA_KGROUP = threads cooperating per output row in the coalesced GEMV.
// Larger KGROUP = wider coalesced read burst per row but a deeper lane reduction
// and fewer rows in parallel. KGROUP=1 = thread-per-row (original summation
// ORDER, bit-identical fp32 accum, but uncoalesced) — used to isolate whether a
// tf-depth change is accumulation-order vs a bug. Must divide MEGA_WG and be a
// power of two (for the lane tree-reduction). NNOPT_MEGA_KGROUP overrides.
static int mega_kgroup() {
    // Default 16 (was 8): KGROUP=8 FAILS the tf-depth gate on this kernel
    // (cb0 depth 16 < 40 — accumulation-order tie flips, isolated 2026-06-04
    // via KGROUP=1 order-identical run = depth 47). KGROUP=16 passes with
    // cb0=46 / cb1=36 / cb2=47 at ~2.26 tok/s (vs 2.39 for the failing 8).
    static int kg = [](){
        const char* e = std::getenv("NNOPT_MEGA_KGROUP");
        int v = e ? std::atoi(e) : 16;
        if (v != 1 && v != 2 && v != 4 && v != 8 && v != 16 && v != 32) v = 16;
        return v;
    }();
    return kg;
}

// ── Per-generation cross-attn K/V cache (precomputed once) ──────────────────
// Indexed [cfg_row*NUM_HIDDEN_LAYERS + layer_idx]. row0 = cond (g_enc_states),
// row1 = uncond (g_enc_zero). Each buffer is [enc_len, hidden] fp16.
static const int kMegaLayers = MODEL_CONFIG::NUM_HIDDEN_LAYERS;

// Campaign-2 gate metric: fp16 weight bytes ONE mega dispatch streams (M=2
// reads each weight once for both CFG rows). 8 GEMVs per layer:
// q/k/v/o + cross q/out (6 × H×H) and fc1/fc2 (2 × H×FFN). ≈ 29.4 MB.
// Passed to KernelProfiler::event_for so dump_summary derives achieved GB/s.
static const uint64_t kMegaWeightBytesPerDispatch =
    (uint64_t)2 /*fp16*/ *
    (6ull * MODEL_CONFIG::DECODER_HIDDEN_SIZE * MODEL_CONFIG::DECODER_HIDDEN_SIZE +
     2ull * MODEL_CONFIG::DECODER_HIDDEN_SIZE * MODEL_CONFIG::FFN_DIM);
// Attention-only stream (6 × H×H) — what the mega dispatch moves when the FFN
// runs externally (NNOPT_MWG_FFN=1; fc1/fc2 bytes are credited to their kernels).
static const uint64_t kMegaAttnWeightBytesPerDispatch =
    (uint64_t)2 * 6ull * MODEL_CONFIG::DECODER_HIDDEN_SIZE * MODEL_CONFIG::DECODER_HIDDEN_SIZE;
static cl_mem g_kcross[2 * 64] = {};
static cl_mem g_vcross[2 * 64] = {};
static int    g_kcross_enc_len = 0;
static bool   g_kcross_ready = false;

static cl_program s_mega_prog = nullptr;
static cl_kernel  s_k_mega = nullptr;
static cl_kernel  s_k_precompute = nullptr;

// ── Stage 2: persistent per-layer kernel clones + cached weight cl_mems ──────
// clSetKernelArg state is retained per cl_kernel object. With ONE shared kernel
// across 24 layers, every per-layer dispatch re-set all 28 args (and re-resolved
// 14 weight cl_mems via a string-map lookup + a clGetMemObjectInfo driver
// round-trip each). By cloning the kernel ONCE per layer (clCreateKernel returns
// an independent arg-state object from the same program), each clone holds its
// layer's LN/proj/FFN weight args + cross-K/V banks bound PERMANENTLY; per step
// only the I/O sub-buffers, KV caches, and scalars (start_pos/enc_len) change.
// This removes ~336 clGetMemObjectInfo round-trips + ~480 redundant setKernelArg
// calls per step from the host critical path. s_k_mega stays for the per-row
// A/B path (which rebinds anyway).
static const int kMegaMaxLayers = 64;
static cl_kernel s_k_mega_layer[kMegaMaxLayers] = {};
static bool      s_layer_weights_bound[kMegaMaxLayers] = {};

// ── Campaign-2 #2a: many-workgroup FFN (NNOPT_MWG_FFN) ───────────────────────
// The single-WG-per-row mega kernel caps occupancy at 2 resident WGs (the Step-0
// finding: 1.7-1.8 GB/s achieved = ~13% of the bus). With NNOPT_MWG_FFN=1 the
// mega program is built with -D MEGA_FFN_EXTERNAL=1: the mega kernel stops after
// final_layer_norm (writing fp32 normed/residual to scratch) and fc1/fc2 run as
// dedicated kernels with (FFN_DIM/rows_per_wg) × num_rows workgroups of 64
// threads — 32-128 WGs streaming the FFN's 16.8 MB/layer concurrently.
// Accumulation order matches the in-mega KGROUP=16 GEMV exactly (bit-identical
// per-row sums), so the tf-depth gate is preserved by construction.
static bool mwg_ffn_enabled() {
    // Default ON (validated 2026-06-04: guard byte-identical, tf-depth cb0=47,
    // fc1 3.9 GB/s / fc2 2.2 GB/s vs 1.55 in-mega). NNOPT_MWG_FFN=0 reverts.
    static const bool on = [](){
        const char* e = std::getenv("NNOPT_MWG_FFN");
        return !(e && e[0] == '0');
    }();
    return on;
}
static int mwg_rpw1() {   // fc1 output rows per WG (N=4096 → nwg = 4096/rpw)
    static const int v = [](){
        const char* e = std::getenv("NNOPT_MWG_RPW1");
        int r = e ? std::atoi(e) : 64;
        if (r < 4 || (MODEL_CONFIG::FFN_DIM % r) != 0) r = 64;
        return r;
    }();
    return v;
}
static int mwg_rpw2() {   // fc2 output rows per WG (N=1024 → nwg = 1024/rpw)
    static const int v = [](){
        const char* e = std::getenv("NNOPT_MWG_RPW2");
        int r = e ? std::atoi(e) : 32;
        // fc2 kernel constraints: rows_per_wg % ROWS_PAR(4) == 0 and
        // rows_per_wg/ROWS_PAR ≤ MEGA_MWG_MAX_ITERS(16) → r ≤ 64.
        if (r < 4 || r > 64 || (r % 4) != 0 || (MODEL_CONFIG::DECODER_HIDDEN_SIZE % r) != 0) r = 32;
        return r;
    }();
    return v;
}
static const int kMwgWG = 64;   // MUST equal the kernel's MEGA_MWG_WG
// Vector width defaults are 4 (the uint2/as_half4 64-bit fetch). Validated
// together 2026-06-04: tf-depth cb0=47 ✓ at VEC 4/4 (NOTE: 4/1 deterministically
// drops cb0 to 16 — these two interact through near-tied logits; change them
// TOGETHER or re-run the tf gate). Set =1 to A/B the scalar path.
static int mwg_vec() {   // 1 = scalar vload_half, 4 = uint2/as_half4 64-bit fetch
    static const int v = [](){
        const char* e = std::getenv("NNOPT_MWG_VEC");
        int r = e ? std::atoi(e) : 4;
        if (r != 1 && r != 4) r = 4;
        return r;
    }();
    return v;
}
static int mega_vec() {   // same, for the in-mega attention GEMVs
    static const int v = [](){
        const char* e = std::getenv("NNOPT_MEGA_VEC");
        int r = e ? std::atoi(e) : 4;
        if (r != 1 && r != 4) r = 4;
        return r;
    }();
    return v;
}
static cl_kernel s_k_fc1 = nullptr;
static cl_kernel s_k_fc2 = nullptr;
static cl_mem s_ffn_normed = nullptr;   // [2, hidden] fp32
static cl_mem s_ffn_resid  = nullptr;   // [2, hidden] fp32
static cl_mem s_ffn1       = nullptr;   // [2, ffn] fp32
// fc1/fc2 weight cl_mems cached per layer (resolved on the slow-path bind) so
// the persistent fast path can dispatch the FFN kernels without string lookups.
static cl_mem s_fc_w_cache[kMegaMaxLayers][2] = {};

// ── Campaign-2 #3: texture-path weights (NNOPT_TEX) ──────────────────────────
// Each GEMV weight [N,K] fp16 is mirrored ONCE into a CL_RGBA/CL_HALF_FLOAT
// image2d (width K/4 texels, height N) via an on-device CopyBufferToImage.
// read_imageh then fetches half4 through the dedicated texture pipe + L1 tex
// cache (dual-issues with the load/store pipe). fp16 builds only.
static bool mega_tex_enabled() {
    static const bool on = [](){
#ifndef NNOPT_USE_FP16
        return false;   // CL_HALF_FLOAT images require the fp16 storage build
#else
        const char* e = std::getenv("NNOPT_TEX");
        return !(e && e[0] == '0');   // default ON (validated: cb0=47, +16% decode)
#endif
    }();
    return on;
}
// fc2's texture path is gated separately (NNOPT_TEX_FC2), default ON. First
// measured a wash vs buffers (2.19 vs 2.17 GB/s) — but WITH #pragma unroll 4
// on the tex chunk loop it's 2.36 GB/s / 4.00 tok/s vs 2.08 / 3.81 buffer
// (2026-06-04): the unroll tightens the CFG row-pair lockstep enough that the
// texture L1 serves the twin row's weight fetches (same dedup that puts fc1 at
// 6.2 GB/s nominal, above DRAM peak). Costs the fc2 image mirror (8 MB/layer,
// ~192 MB over 24 layers). NNOPT_TEX_FC2=0 reverts to buffers.
static bool mega_tex_fc2_enabled() {
    static const bool on = [](){
        const char* e = std::getenv("NNOPT_TEX_FC2");
        return mega_tex_enabled() && !(e && e[0] == '0');
    }();
    return on;
}
// Stage-1 attn split (NNOPT_QKV_EXT → -D MEGA_QKV_EXTERNAL): LN1 + q/k/v leave
// the megakernel for standalone mega_ln_rows + mega_qkv dispatches (packed
// [3H, H/4] texture weights, multi-WG, unroll-4) — the in-mega register
// ceiling caps these GEMVs at 3.5 GB/s while standalone fc1 runs 6.2.
// Requires the texture path. Per-row sums are bit-identical (same KGROUP=16
// lane mapping + tree reduce as mega_gemv_tex). Default ON (validated
// 2026-06-04: mega_qkv 5.67 GB/s vs 3.52 in-mega, decode 3.98 → 4.13 tok/s,
// tf-depth cb0=47 ✓ argmax/logits byte-identical). NNOPT_QKV_EXT=0 reverts.
static bool mega_qkv_ext_enabled() {
    static const bool on = [](){
        const char* e = std::getenv("NNOPT_QKV_EXT");
        return mega_tex_enabled() && !(e && e[0] == '0');
    }();
    return on;
}
static cl_kernel s_k_ln_rows = nullptr;
static cl_kernel s_k_qkv = nullptr;
static cl_mem s_qkv_normed = nullptr;   // [2, hidden] fp32 (LN1 output)
static cl_mem s_q_ext = nullptr;        // [2, hidden] fp32 (q projection)
static cl_mem s_qkv_img[kMegaMaxLayers] = {};        // packed [3H, H/4] image
static cl_mem s_qkv_ln_w[kMegaMaxLayers] = {};       // cached self-attn LN weights
static cl_mem s_qkv_ln_b[kMegaMaxLayers] = {};

// Stage-2 attn split (NNOPT_ATTN_SPLIT): the megakernel is not dispatched at
// all — the whole layer becomes small kernels (ln→qkv→attn→o→ln→cq→attn→co→
// ln→fc1→fc2), each with its own register budget so the o/cq/co GEMVs ride
// the multi-WG texture + unroll-4 path (5.7-6.2 GB/s vs 3.5 in-mega).
// Requires qkv-ext + external FFN. Default ON (validated 2026-06-04:
// o/cq/co 3.5 → 5.45-5.71 GB/s, decode 4.07 → 4.20 tok/s, tf-depth cb0=47 ✓
// argmax/logits byte-identical). NNOPT_ATTN_SPLIT=0 reverts to the megakernel.
static bool mega_attn_split_enabled() {
    static const bool on = [](){
        const char* e = std::getenv("NNOPT_ATTN_SPLIT");
        return mega_qkv_ext_enabled() && mwg_ffn_enabled() && !(e && e[0] == '0');
    }();
    return on;
}
static cl_kernel s_k_attn_core = nullptr;
// Cross-attn CLONE of mega_attn_core (distinct cl_kernel object keeps the two
// dispatch streams' arg state independent).
static cl_kernel s_k_attn_core_x = nullptr;
// NNOPT_ATTN_TEX (default ON, fp16 builds): K reads in mega_attn_core go
// through an image2d VIEW over the KV cache buffer (the lm_heads zero-copy
// recipe) — texture L1 serves the t-strided rows the buffer path crawled at
// ~1.1 GB/s effective. =0 reverts to the vectorized-buffer path.
static bool mega_attn_tex_enabled() {
    static const bool on = [](){
        const char* e = std::getenv("NNOPT_ATTN_TEX");
        return sizeof(nnopt_storage_t) == 2 && !(e && e[0] == '0');
    }();
    return on;
}
static bool mega_int8o_enabled();   // fwd decl (defined below) — fuse-LN gate
// NNOPT_FUSE_LN=1 (default OFF — MEASURED REGRESSION): fuses the 3 standalone
// LayerNorm dispatches per layer (ln1/mega_ln_rows, ln2, ln3) into their
// consumer GEMVs' local staging (qkv, cq-proj, fc1). 12 → 9 dispatches/layer,
// gates pass (guard ✓, tf cb0=47) — but the LN stats are recomputed
// REDUNDANTLY by EVERY WG of the consumer (qkv 96 WGs/row, fc1 128): fc1
// 822→969 µs, qkv 666→841 µs, decode 10.78 → 9.86 tok/s (−8.5%, same-binary
// A/B 2026-06-05). The 30 µs standalone LN dispatch is the cheaper structure
// on multi-WG consumers. Kept env-gated for future single-WG-consumer shapes.
static bool mega_fuse_ln_enabled() {
    static const bool on = [](){
        const char* e = std::getenv("NNOPT_FUSE_LN");
        return mega_attn_split_enabled() && !mega_int8o_enabled() && (e && e[0] == '1');
    }();
    return on;
}
static cl_kernel s_k_proj = nullptr;
static cl_kernel s_k_ln_f32 = nullptr;

// fc2 split-K (NNOPT_FC2_SPLITK → kernels mega_ffn_fc2_sk + _reduce): each WG
// owns (row-block, K-segment) with a 4 KB staged x-segment — fc1's shape.
// CHANGES reduction order (not bit-identical) — gated on tf-depth and PASSED:
// cb0=44 ≥ 40 (deterministic ×2), argmax identical, logits ±1e-2 (2026-06-04).
// Default ON: fc2 2.22 → 5.78 GB/s, decode 4.11 → 5.30 tok/s (+29%).
// NNOPT_FC2_SPLITK=0 reverts to the chunked single-K kernel.
// Requires the fc2 texture path.
static const int kFc2KSeg = 4;          // MUST match kernel MEGA_FC2_KSEG
static bool mega_fc2_splitk_enabled() {
    static const bool on = [](){
        const char* e = std::getenv("NNOPT_FC2_SPLITK");
        return mega_tex_fc2_enabled() && !(e && e[0] == '0');
    }();
    return on;
}
static cl_kernel s_k_fc2_sk = nullptr;
static cl_kernel s_k_fc2_sk_red = nullptr;
static cl_mem s_fc2_part = nullptr;     // [2, KSEG, H] fp32 partials

// int8 + fp16-outlier FFN quantization (NNOPT_INT8O → kernels *_i8o).
// The Fix-12-verdict scheme: top ~1.5% |w| per row stay EXACT (fp32 outlier
// list applied post-reduce); the rest are int8 with per-128-group scales,
// streamed through the SAME image2d path (CFG-twin L1 dedup preserved).
// Gated on tf-depth cb0 >= 40 like every order-changing opt. Default OFF
// until the gate passes; requires tex + split-K + external FFN.
// mode: 0=off, 1=fc1 only ("f1"), 2=fc2 only ("f2"), 3=both ("1"),
//       4=fc1 via HARDWARE dot8 ("d1" — cl_qcom_dot_product8)
static int mega_int8o_mode() {
    static const int m = [](){
        const char* e = std::getenv("NNOPT_INT8O");
        if (!e || !mega_fc2_splitk_enabled() || !mwg_ffn_enabled()) return 0;
        if (e[0] == '1') return 3;
        if (e[0] == 'f' && e[1] == '1') return 1;
        if (e[0] == 'f' && e[1] == '2') return 2;
        if (e[0] == 'd' && e[1] == '1') return 4;
        if (e[0] == 'a') return 5;   // "auto": dot8 fc1 ONLY on single-row steps
        return 0;
    }();
    return m;
}
static const char* mega_dot8_fn() {
    static const char* fn = [](){
        const char* e = std::getenv("NNOPT_DOT8_FN");
        return e && e[0] ? e : "qcom_sdot8";
    }();
    return fn;
}
static bool mega_int8o_enabled() { return mega_int8o_mode() != 0; }
static cl_kernel s_k_fc1_i8o = nullptr;
static cl_kernel s_k_fc2_i8o = nullptr;
static cl_kernel s_k_fc1_i8d = nullptr;   // dot8 variant
static cl_kernel s_k_actq = nullptr;      // activation quantizer
static cl_mem s_actq8 = nullptr;          // [2, H] offset-binary int8 acts
static cl_mem s_actsc = nullptr;          // [2, H/128] fp32 act scales
static cl_mem s_actxs = nullptr;          // [2, H/128] int32 act group sums
static cl_mem s_i8o_wsum[kMegaMaxLayers][2] = {};   // [N, K/128] int32 weight sums
// per layer, slot 0=fc1 / 1=fc2: int8 image + scales + outlier idx/val buffers
static cl_mem s_i8o_img[kMegaMaxLayers][2] = {};
static cl_mem s_i8o_scale[kMegaMaxLayers][2] = {};
static cl_mem s_i8o_oidx[kMegaMaxLayers][2] = {};
static cl_mem s_i8o_oval[kMegaMaxLayers][2] = {};
static const int kI8oNoutFc1 = 16;   // 16/1024 = 1.6% outliers per row
static const int kI8oNoutFc2 = 64;   // 64/4096 = 1.6%
static cl_mem s_attn_scr = nullptr;   // [2, hidden] fp32 attention output
static cl_mem s_resid2 = nullptr;     // [2, hidden] fp32 residual after o-proj
static cl_mem s_ln2n = nullptr;       // [2, hidden] fp32 LN2 output
static cl_mem s_split_enc_ln_w[kMegaMaxLayers] = {};   // cross-attn LN weights
static cl_mem s_split_enc_ln_b[kMegaMaxLayers] = {};
static cl_mem s_split_fin_ln_w[kMegaMaxLayers] = {};   // final LN weights
static cl_mem s_split_fin_ln_b[kMegaMaxLayers] = {};

// CFG-row-fused fc2 (NNOPT_FC2_FUSE → -D MEGA_FC2_FUSE2): one WG computes BOTH
// CFG rows so each weight vector is fetched once (halves fc2 weight DRAM
// traffic — the (nwg,2) dispatch's drifting row-pairs defeat L2 dedup).
// Mutually exclusive with the fc2 texture path (buffer-only variant).
// CFG-twin row interleave (see kernels/decoder_layer_mega.cl MEGA_ROW_ILV
// comment): puts the CFG row on group-dim0 so twin WGs co-schedule and the
// texture L1 serves the second row's weight fetches. Byte-identical math.
// NNOPT_ROW_ILV=0 reverts to the (slices, rows) shape.
// TRUE CFG-row fusion (NNOPT_ROW_FUSE, default OFF until gated): when a GEMV
// runs with num_rows==2, dispatch the *_rf variant — one WG reads each weight
// texel once and accumulates both rows (see kernel comment). Byte-identical
// per row. Falls back to the unfused kernel on single-row steps.
static bool mega_row_fuse_enabled() {
    static const bool on = [](){
        const char* e = std::getenv("NNOPT_ROW_FUSE");
        return e && e[0] == '1';
    }();
    return on;
}
static cl_kernel s_k_fc1_rf = nullptr;

static bool mega_row_ilv_enabled() {
    static const bool on = [](){
        const char* e = std::getenv("NNOPT_ROW_ILV");
        return !(e && e[0] == '0');
    }();
    return on;
}

static bool mega_fc2_fuse_enabled() {
    static const bool on = [](){
        const char* e = std::getenv("NNOPT_FC2_FUSE");
        return e && e[0] == '1' && !mega_tex_fc2_enabled();
    }();
    return on;
}
// slot: 0=q 1=k 2=v 3=o 4=cq 5=co 6=fc1 7=fc2 (same convention as int8)
static cl_mem s_w_img[kMegaMaxLayers][8] = {};
static cl_mem mega_weight_image(OpenCLContext& cl_ctx, cl_command_queue queue,
                                cl_mem buf, int N, int K, int layer_idx, int slot) {
    if (s_w_img[layer_idx][slot]) return s_w_img[layer_idx][slot];
    if (!buf || (K % 4) != 0) return nullptr;
    cl_image_format fmt{};
    fmt.image_channel_order = CL_RGBA;
    fmt.image_channel_data_type = CL_HALF_FLOAT;
    cl_image_desc desc{};
    desc.image_type = CL_MEM_OBJECT_IMAGE2D;
    desc.image_width = (size_t)K / 4;
    desc.image_height = (size_t)N;
    cl_int err = CL_SUCCESS;
    // cl_khr_image2d_from_buffer (CONFIRMED on device): create the image as a
    // VIEW over the existing weight buffer — no CopyBufferToImage mirror
    // (192 MB of copies + duplicate memory across 24 layers). Rows are
    // contiguous (K/4 texels x 8 B). NNOPT_TEX_VIEW=0 or any failure falls
    // back to the mirror copy.
    static const bool view_on = [](){
        const char* e = std::getenv("NNOPT_TEX_VIEW");
        return !(e && e[0] == '0');
    }();
    if (view_on) {
        cl_image_desc vdesc = desc;
        vdesc.image_row_pitch = (size_t)K / 4 * 8;   // bytes; contiguous rows
        vdesc.buffer = buf;
        cl_mem img = clCreateImage(cl_ctx.context(), CL_MEM_READ_ONLY, &fmt, &vdesc, nullptr, &err);
        if (err == CL_SUCCESS && img) {
            s_w_img[layer_idx][slot] = img;
            return img;
        }
        static bool warned = false;
        if (!warned) { fprintf(stderr, "TEX_VIEW: image2d_from_buffer failed (%d) - mirror-copy fallback\n", (int)err); warned = true; }
    }
    cl_mem img = clCreateImage(cl_ctx.context(), CL_MEM_READ_ONLY, &fmt, &desc, nullptr, &err);
    if (err != CL_SUCCESS || !img) { NNOPT_ERROR_FMT("tex: clCreateImage %d (L=%d slot=%d)", (int)err, layer_idx, slot); return nullptr; }
    const size_t origin[3] = {0, 0, 0};
    const size_t region[3] = {(size_t)K / 4, (size_t)N, 1};
    err = clEnqueueCopyBufferToImage(queue, buf, img, 0, origin, region, 0, nullptr, nullptr);
    if (err != CL_SUCCESS) { NNOPT_ERROR_FMT("tex: CopyBufferToImage %d (L=%d slot=%d)", (int)err, layer_idx, slot); clReleaseMemObject(img); return nullptr; }
    s_w_img[layer_idx][slot] = img;
    return img;
}

static bool mega_ffn_ensure_scratch(OpenCLContext& cl_ctx) {
    if (s_ffn1) return true;
    const size_t H = MODEL_CONFIG::DECODER_HIDDEN_SIZE, F = MODEL_CONFIG::FFN_DIM;
    cl_int err = CL_SUCCESS;
    s_ffn_normed = clCreateBuffer(cl_ctx.context(), CL_MEM_READ_WRITE, 2 * H * sizeof(float), nullptr, &err);
    if (err != CL_SUCCESS) { NNOPT_ERROR_FMT("mwg-ffn: normed scratch alloc %d", (int)err); return false; }
    s_ffn_resid = clCreateBuffer(cl_ctx.context(), CL_MEM_READ_WRITE, 2 * H * sizeof(float), nullptr, &err);
    if (err != CL_SUCCESS) { NNOPT_ERROR_FMT("mwg-ffn: resid scratch alloc %d", (int)err); return false; }
    s_ffn1 = clCreateBuffer(cl_ctx.context(), CL_MEM_READ_WRITE, 2 * F * sizeof(float), nullptr, &err);
    if (err != CL_SUCCESS) { NNOPT_ERROR_FMT("mwg-ffn: ffn1 scratch alloc %d", (int)err); return false; }
    if (mega_fc2_splitk_enabled()) {
        s_fc2_part = clCreateBuffer(cl_ctx.context(), CL_MEM_READ_WRITE,
                                    (size_t)2 * kFc2KSeg * H * sizeof(float), nullptr, &err);
        if (err != CL_SUCCESS) { NNOPT_ERROR_FMT("fc2-sk: partials alloc %d", (int)err); return false; }
    }
    if (mega_int8o_mode() >= 4) {
        s_actq8 = clCreateBuffer(cl_ctx.context(), CL_MEM_READ_WRITE, 2 * H, nullptr, &err);
        if (err != CL_SUCCESS) { NNOPT_ERROR_FMT("dot8: actq8 alloc %d", (int)err); return false; }
        s_actsc = clCreateBuffer(cl_ctx.context(), CL_MEM_READ_WRITE, 2 * (H / 128) * sizeof(float), nullptr, &err);
        if (err != CL_SUCCESS) { NNOPT_ERROR_FMT("dot8: actsc alloc %d", (int)err); return false; }
        s_actxs = clCreateBuffer(cl_ctx.context(), CL_MEM_READ_WRITE, 2 * (H / 128) * sizeof(int32_t), nullptr, &err);
        if (err != CL_SUCCESS) { NNOPT_ERROR_FMT("dot8: actxs alloc %d", (int)err); return false; }
    }
    return true;
}

// Enqueue fc1 (+GELU) and fc2 (+residual+store) after the external-FFN mega
// dispatch. num_rows = the mega dispatch's num_wg (2 = both CFG rows).
static bool mega_ffn_dispatch(cl_command_queue queue, int layer_idx,
                              cl_mem x_out0, cl_mem x_out1, int num_rows) {
    const int H = MODEL_CONFIG::DECODER_HIDDEN_SIZE, F = MODEL_CONFIG::FFN_DIM;
    cl_mem fc1_w = s_fc_w_cache[layer_idx][0];
    cl_mem fc2_w = s_fc_w_cache[layer_idx][1];
    if (!fc1_w || !fc2_w) { NNOPT_ERROR_FMT("mwg-ffn: fc weights not cached (L=%d)", layer_idx); return false; }
    const uint64_t fc_bytes = (uint64_t)2 * (uint64_t)H * (uint64_t)F;   // fp16 weight stream per kernel
    const uint64_t fc_bytes_i8 = (uint64_t)H * (uint64_t)F;              // int8 stream
    cl_int err = CL_SUCCESS;
    // fc1: [F,H] @ normed → gelu → ffn1
    const int rpw1 = mwg_rpw1();
    const int i8mode = mega_int8o_mode();
    if (i8mode == 4 || (i8mode == 5 && num_rows == 1)) {
        // dynamic activation quant (tiny) then dot8 fc1. Mode 5 ("auto") uses
        // dot8-int8 ONLY on single-row (CFG-early) steps — the only regime
        // where int8 beats fp16-texture (no CFG twin to L1-dedup there).
        const int Hc = H;
        err  = clSetKernelArg(s_k_actq, 0, sizeof(cl_mem), &s_ffn_normed);
        err |= clSetKernelArg(s_k_actq, 1, sizeof(cl_mem), &s_actq8);
        err |= clSetKernelArg(s_k_actq, 2, sizeof(cl_mem), &s_actsc);
        err |= clSetKernelArg(s_k_actq, 3, sizeof(cl_mem), &s_actxs);
        err |= clSetKernelArg(s_k_actq, 4, sizeof(int), &Hc);
        if (err != CL_SUCCESS) { NNOPT_ERROR_FMT("dot8: actq args %d", (int)err); return false; }
        const size_t gq[2] = {(size_t)(H / 128 + 7) / 8 * 8, (size_t)num_rows};
        const size_t lq[2] = {8, 1};
        err = clEnqueueNDRangeKernel(queue, s_k_actq, 2, nullptr, gq, lq, 0, nullptr,
                                     KernelProfiler::event_for("i8o_actq"));
        if (err != CL_SUCCESS) { NNOPT_ERROR_FMT("dot8: actq dispatch %d", (int)err); return false; }
        const int rpw1d = mwg_rpw1();
        const int nout1 = kI8oNoutFc1;
        err  = clSetKernelArg(s_k_fc1_i8d, 0, sizeof(cl_mem), &s_i8o_img[layer_idx][0]);
        err |= clSetKernelArg(s_k_fc1_i8d, 1, sizeof(cl_mem), &s_i8o_scale[layer_idx][0]);
        err |= clSetKernelArg(s_k_fc1_i8d, 2, sizeof(cl_mem), &s_i8o_wsum[layer_idx][0]);
        err |= clSetKernelArg(s_k_fc1_i8d, 3, sizeof(cl_mem), &s_i8o_oidx[layer_idx][0]);
        err |= clSetKernelArg(s_k_fc1_i8d, 4, sizeof(cl_mem), &s_i8o_oval[layer_idx][0]);
        err |= clSetKernelArg(s_k_fc1_i8d, 5, sizeof(cl_mem), &s_ffn_normed);
        err |= clSetKernelArg(s_k_fc1_i8d, 6, sizeof(cl_mem), &s_actq8);
        err |= clSetKernelArg(s_k_fc1_i8d, 7, sizeof(cl_mem), &s_actsc);
        err |= clSetKernelArg(s_k_fc1_i8d, 8, sizeof(cl_mem), &s_actxs);
        err |= clSetKernelArg(s_k_fc1_i8d, 9, sizeof(cl_mem), &s_ffn1);
        err |= clSetKernelArg(s_k_fc1_i8d, 10, sizeof(int), &rpw1d);
        err |= clSetKernelArg(s_k_fc1_i8d, 11, sizeof(int), &nout1);
        if (err != CL_SUCCESS) { NNOPT_ERROR_FMT("dot8: fc1 args %d (L=%d)", (int)err, layer_idx); return false; }
        const size_t g1d[2] = {(size_t)(F / rpw1d) * kMwgWG, (size_t)num_rows};
        const size_t l1d[2] = {(size_t)kMwgWG, 1};
        err = clEnqueueNDRangeKernel(queue, s_k_fc1_i8d, 2, nullptr, g1d, l1d, 0, nullptr,
                                     KernelProfiler::event_for("mega_ffn_fc1_i8d", fc_bytes_i8));
        if (err != CL_SUCCESS) { NNOPT_ERROR_FMT("dot8: fc1 dispatch %d (L=%d)", (int)err, layer_idx); return false; }
    } else if (i8mode == 1 || i8mode == 3) {   // image-variant int8 (NOT mode 5: its fc1 data is a buffer)
        const int nout1 = kI8oNoutFc1;
        err  = clSetKernelArg(s_k_fc1_i8o, 0, sizeof(cl_mem), &s_i8o_img[layer_idx][0]);
        err |= clSetKernelArg(s_k_fc1_i8o, 1, sizeof(cl_mem), &s_i8o_scale[layer_idx][0]);
        err |= clSetKernelArg(s_k_fc1_i8o, 2, sizeof(cl_mem), &s_i8o_oidx[layer_idx][0]);
        err |= clSetKernelArg(s_k_fc1_i8o, 3, sizeof(cl_mem), &s_i8o_oval[layer_idx][0]);
        err |= clSetKernelArg(s_k_fc1_i8o, 4, sizeof(cl_mem), &s_ffn_normed);
        err |= clSetKernelArg(s_k_fc1_i8o, 5, sizeof(cl_mem), &s_ffn1);
        err |= clSetKernelArg(s_k_fc1_i8o, 6, sizeof(int), &rpw1);
        err |= clSetKernelArg(s_k_fc1_i8o, 7, sizeof(int), &nout1);
        if (err != CL_SUCCESS) { NNOPT_ERROR_FMT("i8o: fc1 args %d (L=%d)", (int)err, layer_idx); return false; }
        const size_t g1[2] = {(size_t)(F / rpw1) * kMwgWG, (size_t)num_rows};
        const size_t l1[2] = {(size_t)kMwgWG, 1};
        err = clEnqueueNDRangeKernel(queue, s_k_fc1_i8o, 2, nullptr, g1, l1, 0, nullptr,
                                     KernelProfiler::event_for("mega_ffn_fc1_i8o", fc_bytes_i8));
        if (err != CL_SUCCESS) { NNOPT_ERROR_FMT("i8o: fc1 dispatch %d (L=%d)", (int)err, layer_idx); return false; }
    } else if (s_k_fc1_rf && num_rows == 2 && !mega_fuse_ln_enabled()) {
    // TRUE row-fused fc1: (slices, 1) dispatch, both rows per weight read.
    err  = clSetKernelArg(s_k_fc1_rf, 0, sizeof(cl_mem), &fc1_w);
    err |= clSetKernelArg(s_k_fc1_rf, 1, sizeof(cl_mem), &s_ffn_normed);
    err |= clSetKernelArg(s_k_fc1_rf, 2, sizeof(cl_mem), &s_ffn1);
    err |= clSetKernelArg(s_k_fc1_rf, 3, sizeof(int), &rpw1);
    if (err != CL_SUCCESS) { NNOPT_ERROR_FMT("mwg-ffn: fc1_rf args %d (L=%d)", (int)err, layer_idx); return false; }
    const size_t gws1f[1] = {(size_t)(F / rpw1) * kMwgWG};
    const size_t lws1f[1] = {(size_t)kMwgWG};
    err = clEnqueueNDRangeKernel(queue, s_k_fc1_rf, 1, nullptr, gws1f, lws1f, 0, nullptr,
                                 KernelProfiler::event_for("mega_ffn_fc1", fc_bytes));
    if (err != CL_SUCCESS) { NNOPT_ERROR_FMT("mwg-ffn: fc1_rf dispatch %d (L=%d)", (int)err, layer_idx); return false; }
    } else {
    err |= clSetKernelArg(s_k_fc1, 0, sizeof(cl_mem), &fc1_w);
    if (mega_fuse_ln_enabled()) {
        // LN3 fused into fc1's staging: feed the RAW ffn_resid + the layer's
        // final-LN params (cached by the split-prep slow path).
        cl_mem fin_w = s_split_fin_ln_w[layer_idx], fin_b = s_split_fin_ln_b[layer_idx];
        if (!fin_w || !fin_b) { NNOPT_ERROR_FMT("mwg-ffn: fin LN not cached (L=%d)", layer_idx); return false; }
        const float eps_ln = MODEL_CONFIG::LAYER_NORM_EPS;
        err |= clSetKernelArg(s_k_fc1, 1, sizeof(cl_mem), &s_ffn_resid);
        err |= clSetKernelArg(s_k_fc1, 2, sizeof(cl_mem), &s_ffn1);
        err |= clSetKernelArg(s_k_fc1, 3, sizeof(int), &rpw1);
        err |= clSetKernelArg(s_k_fc1, 4, sizeof(cl_mem), &fin_w);
        err |= clSetKernelArg(s_k_fc1, 5, sizeof(cl_mem), &fin_b);
        err |= clSetKernelArg(s_k_fc1, 6, sizeof(float), &eps_ln);
    } else {
        err |= clSetKernelArg(s_k_fc1, 1, sizeof(cl_mem), &s_ffn_normed);
        err |= clSetKernelArg(s_k_fc1, 2, sizeof(cl_mem), &s_ffn1);
        err |= clSetKernelArg(s_k_fc1, 3, sizeof(int), &rpw1);
    }
    if (err != CL_SUCCESS) { NNOPT_ERROR_FMT("mwg-ffn: fc1 args %d (L=%d)", (int)err, layer_idx); return false; }
    const bool ilv1 = mega_row_ilv_enabled();
    const size_t gws1[2] = {ilv1 ? (size_t)num_rows * kMwgWG : (size_t)(F / rpw1) * kMwgWG,
                            ilv1 ? (size_t)(F / rpw1)        : (size_t)num_rows};
    const size_t lws1[2] = {(size_t)kMwgWG, 1};
    err = clEnqueueNDRangeKernel(queue, s_k_fc1, 2, nullptr, gws1, lws1, 0, nullptr,
                                 KernelProfiler::event_for("mega_ffn_fc1", fc_bytes));
    if (err != CL_SUCCESS) { NNOPT_ERROR_FMT("mwg-ffn: fc1 dispatch %d (L=%d)", (int)err, layer_idx); return false; }
    }
    if (mega_fc2_splitk_enabled()) {
        // split-K fc2: partial sums per (row, K-segment) + tiny reduce.
        const int rpw2 = mwg_rpw2();
        if (mega_int8o_mode() & 2) {
            const int nout2 = kI8oNoutFc2;
            err  = clSetKernelArg(s_k_fc2_i8o, 0, sizeof(cl_mem), &s_i8o_img[layer_idx][1]);
            err |= clSetKernelArg(s_k_fc2_i8o, 1, sizeof(cl_mem), &s_i8o_scale[layer_idx][1]);
            err |= clSetKernelArg(s_k_fc2_i8o, 2, sizeof(cl_mem), &s_i8o_oidx[layer_idx][1]);
            err |= clSetKernelArg(s_k_fc2_i8o, 3, sizeof(cl_mem), &s_i8o_oval[layer_idx][1]);
            err |= clSetKernelArg(s_k_fc2_i8o, 4, sizeof(cl_mem), &s_ffn1);
            err |= clSetKernelArg(s_k_fc2_i8o, 5, sizeof(cl_mem), &s_fc2_part);
            err |= clSetKernelArg(s_k_fc2_i8o, 6, sizeof(int), &rpw2);
            err |= clSetKernelArg(s_k_fc2_i8o, 7, sizeof(int), &nout2);
            if (err != CL_SUCCESS) { NNOPT_ERROR_FMT("i8o: fc2 args %d (L=%d)", (int)err, layer_idx); return false; }
            const size_t gsk[3] = {(size_t)(H / rpw2) * kMwgWG, (size_t)kFc2KSeg, (size_t)num_rows};
            const size_t lsk[3] = {(size_t)kMwgWG, 1, 1};
            err = clEnqueueNDRangeKernel(queue, s_k_fc2_i8o, 3, nullptr, gsk, lsk, 0, nullptr,
                                         KernelProfiler::event_for("mega_ffn_fc2_sk_i8o", fc_bytes_i8));
            if (err != CL_SUCCESS) { NNOPT_ERROR_FMT("i8o: fc2 dispatch %d (L=%d)", (int)err, layer_idx); return false; }
        } else {
        err  = clSetKernelArg(s_k_fc2_sk, 0, sizeof(cl_mem), &fc2_w);
        err |= clSetKernelArg(s_k_fc2_sk, 1, sizeof(cl_mem), &s_ffn1);
        err |= clSetKernelArg(s_k_fc2_sk, 2, sizeof(cl_mem), &s_fc2_part);
        err |= clSetKernelArg(s_k_fc2_sk, 3, sizeof(int), &rpw2);
        if (err != CL_SUCCESS) { NNOPT_ERROR_FMT("fc2-sk: args %d (L=%d)", (int)err, layer_idx); return false; }
        const bool ilvsk = mega_row_ilv_enabled();
        const size_t gsk[3] = {ilvsk ? (size_t)num_rows * kMwgWG : (size_t)(H / rpw2) * kMwgWG,
                               (size_t)kFc2KSeg,
                               ilvsk ? (size_t)(H / rpw2)        : (size_t)num_rows};
        const size_t lsk[3] = {(size_t)kMwgWG, 1, 1};
        err = clEnqueueNDRangeKernel(queue, s_k_fc2_sk, 3, nullptr, gsk, lsk, 0, nullptr,
                                     KernelProfiler::event_for("mega_ffn_fc2_sk", fc_bytes));
        if (err != CL_SUCCESS) { NNOPT_ERROR_FMT("fc2-sk: dispatch %d (L=%d)", (int)err, layer_idx); return false; }
        }
        err  = clSetKernelArg(s_k_fc2_sk_red, 0, sizeof(cl_mem), &s_fc2_part);
        err |= clSetKernelArg(s_k_fc2_sk_red, 1, sizeof(cl_mem), &s_ffn_resid);
        err |= clSetKernelArg(s_k_fc2_sk_red, 2, sizeof(cl_mem), &x_out0);
        err |= clSetKernelArg(s_k_fc2_sk_red, 3, sizeof(cl_mem), &x_out1);
        if (err != CL_SUCCESS) { NNOPT_ERROR_FMT("fc2-sk: reduce args %d (L=%d)", (int)err, layer_idx); return false; }
        const size_t gred[2] = {(size_t)H, (size_t)num_rows};
        const size_t lred[2] = {64, 1};
        err = clEnqueueNDRangeKernel(queue, s_k_fc2_sk_red, 2, nullptr, gred, lred, 0, nullptr,
                                     KernelProfiler::event_for("mega_ffn_fc2_sk_red", 0));
        if (err != CL_SUCCESS) { NNOPT_ERROR_FMT("fc2-sk: reduce dispatch %d (L=%d)", (int)err, layer_idx); return false; }
        return true;
    }
    // fc2: [H,F] @ ffn1 + resid → x_out (fp16)
    const int rpw2 = mwg_rpw2();
    err  = clSetKernelArg(s_k_fc2, 0, sizeof(cl_mem), &fc2_w);
    err |= clSetKernelArg(s_k_fc2, 1, sizeof(cl_mem), &s_ffn1);
    err |= clSetKernelArg(s_k_fc2, 2, sizeof(cl_mem), &s_ffn_resid);
    err |= clSetKernelArg(s_k_fc2, 3, sizeof(cl_mem), &x_out0);
    err |= clSetKernelArg(s_k_fc2, 4, sizeof(cl_mem), &x_out1);
    err |= clSetKernelArg(s_k_fc2, 5, sizeof(int), &rpw2);
    if (mega_fc2_fuse_enabled())   // fused variant takes num_rows as arg 6, runs (nwg, 1)
        err |= clSetKernelArg(s_k_fc2, 6, sizeof(int), &num_rows);
    if (err != CL_SUCCESS) { NNOPT_ERROR_FMT("mwg-ffn: fc2 args %d (L=%d)", (int)err, layer_idx); return false; }
    const size_t gws2[2] = {(size_t)(H / rpw2) * kMwgWG,
                            mega_fc2_fuse_enabled() ? (size_t)1 : (size_t)num_rows};
    const size_t lws2[2] = {(size_t)kMwgWG, 1};
    err = clEnqueueNDRangeKernel(queue, s_k_fc2, 2, nullptr, gws2, lws2, 0, nullptr,
                                 KernelProfiler::event_for("mega_ffn_fc2", fc_bytes));
    if (err != CL_SUCCESS) { NNOPT_ERROR_FMT("mwg-ffn: fc2 dispatch %d (L=%d)", (int)err, layer_idx); return false; }
    return true;
}

// ── Stage-1 attn split host side (NNOPT_QKV_EXT) ─────────────────────────────
static bool mega_qkv_ensure_scratch(OpenCLContext& cl_ctx) {
    if (s_q_ext) return true;
    const size_t H = MODEL_CONFIG::DECODER_HIDDEN_SIZE;
    cl_int err = CL_SUCCESS;
    s_qkv_normed = clCreateBuffer(cl_ctx.context(), CL_MEM_READ_WRITE, 2 * H * sizeof(float), nullptr, &err);
    if (err != CL_SUCCESS) { NNOPT_ERROR_FMT("qkv-ext: normed scratch alloc %d", (int)err); return false; }
    s_q_ext = clCreateBuffer(cl_ctx.context(), CL_MEM_READ_WRITE, 2 * H * sizeof(float), nullptr, &err);
    if (err != CL_SUCCESS) { NNOPT_ERROR_FMT("qkv-ext: q scratch alloc %d", (int)err); return false; }
    return true;
}

// Packed [3H, H/4] CL_RGBA/CL_HALF_FLOAT image: q rows at y[0,H), k at [H,2H),
// v at [2H,3H). Built once per layer from the three [H,H] fp16 weight buffers.
static cl_mem mega_qkv_image(OpenCLContext& cl_ctx, cl_command_queue queue,
                             cl_mem q_buf, cl_mem k_buf, cl_mem v_buf, int layer_idx) {
    if (s_qkv_img[layer_idx]) return s_qkv_img[layer_idx];
    const int H = MODEL_CONFIG::DECODER_HIDDEN_SIZE;
    if (!q_buf || !k_buf || !v_buf || (H % 4) != 0) return nullptr;
    cl_image_format fmt{};
    fmt.image_channel_order = CL_RGBA;
    fmt.image_channel_data_type = CL_HALF_FLOAT;
    cl_image_desc desc{};
    desc.image_type = CL_MEM_OBJECT_IMAGE2D;
    desc.image_width = (size_t)H / 4;
    desc.image_height = (size_t)3 * H;
    cl_int err = CL_SUCCESS;
    cl_mem img = clCreateImage(cl_ctx.context(), CL_MEM_READ_ONLY, &fmt, &desc, nullptr, &err);
    if (err != CL_SUCCESS || !img) { NNOPT_ERROR_FMT("qkv-ext: clCreateImage %d (L=%d)", (int)err, layer_idx); return nullptr; }
    const size_t region[3] = {(size_t)H / 4, (size_t)H, 1};
    const cl_mem bufs[3] = {q_buf, k_buf, v_buf};
    for (int s = 0; s < 3; ++s) {
        const size_t origin[3] = {0, (size_t)s * H, 0};
        err = clEnqueueCopyBufferToImage(queue, bufs[s], img, 0, origin, region, 0, nullptr, nullptr);
        if (err != CL_SUCCESS) { NNOPT_ERROR_FMT("qkv-ext: CopyBufferToImage %d (L=%d s=%d)", (int)err, layer_idx, s); clReleaseMemObject(img); return nullptr; }
    }
    s_qkv_img[layer_idx] = img;
    return img;
}

// Enqueue mega_ln_rows + mega_qkv BEFORE the mega dispatch. Uses the per-layer
// cached image + LN weights (populated by the slow-bind path on first use).
// ── Step-params buffers (recordable-replay support) ─────────────────────────
// s_sp_buf[0] = start_pos for the CURRENT step (FillBuffer'd once per step on
// the in-order queue: each enqueued step's kernels read the value the host
// wrote just before them — including replayed recordings, which is the whole
// point: no per-step kernel-arg overrides needed). s_enc_buf[0] = enc_len for
// the cross-attn seq_ptr (refilled when the cross-KV is re-precomputed).
static cl_mem s_sp_buf = nullptr;
static cl_mem s_enc_buf = nullptr;
static int    s_enc_filled = -1;

extern "C" bool mega_write_step_params(OpenCLContext& cl_ctx, cl_command_queue queue, int start_pos) {
    cl_int err = CL_SUCCESS;
    if (!s_sp_buf) {
        s_sp_buf = clCreateBuffer(cl_ctx.context(), CL_MEM_READ_ONLY, 2 * sizeof(int), nullptr, &err);
        if (err != CL_SUCCESS || !s_sp_buf) { NNOPT_ERROR_FMT("step-params alloc %d", (int)err); s_sp_buf = nullptr; return false; }
    }
    // FillBuffer: the pattern is COPIED at enqueue (no host-lifetime hazard,
    // unlike a non-blocking WriteBuffer), and it's a queue command, so each
    // step's kernels see their own value on the in-order queue.
    const int pat[2] = {start_pos, start_pos + 1};
    err = clEnqueueFillBuffer(queue, s_sp_buf, pat, sizeof(pat), 0, sizeof(pat), 0, nullptr, nullptr);
    if (err != CL_SUCCESS) { NNOPT_ERROR_FMT("step-params fill %d", (int)err); return false; }
    return true;
}

extern "C" cl_mem mega_step_params_buf() { return s_sp_buf; }

static cl_mem mega_enc_len_buf(OpenCLContext& cl_ctx, cl_command_queue queue, int enc_len) {
    cl_int err = CL_SUCCESS;
    if (!s_enc_buf) {
        s_enc_buf = clCreateBuffer(cl_ctx.context(), CL_MEM_READ_ONLY, 2 * sizeof(int), nullptr, &err);
        if (err != CL_SUCCESS || !s_enc_buf) { NNOPT_ERROR_FMT("enc-len buf alloc %d", (int)err); s_enc_buf = nullptr; return nullptr; }
    }
    if (s_enc_filled != enc_len) {
        const int pat[2] = {enc_len, enc_len};
        if (clEnqueueFillBuffer(queue, s_enc_buf, pat, sizeof(pat), 0, sizeof(pat), 0, nullptr, nullptr) != CL_SUCCESS)
            return nullptr;
        s_enc_filled = enc_len;
    }
    return s_enc_buf;
}

static bool mega_qkv_predispatch(cl_command_queue queue, int layer_idx,
                                 cl_mem x_in0, cl_mem x_in1,
                                 cl_mem* k_cache_r0, cl_mem* v_cache_r0,
                                 cl_mem* k_cache_r1, cl_mem* v_cache_r1,
                                 int start_pos, int num_wg) {
    const int H = MODEL_CONFIG::DECODER_HIDDEN_SIZE;
    const float eps = MODEL_CONFIG::LAYER_NORM_EPS;
    cl_mem img = s_qkv_img[layer_idx];
    cl_mem ln_w = s_qkv_ln_w[layer_idx], ln_b = s_qkv_ln_b[layer_idx];
    if (!img || !ln_w || !ln_b) { NNOPT_ERROR_FMT("qkv-ext: layer %d not prepared", layer_idx); return false; }
    cl_int err = CL_SUCCESS;
    const bool fuse_ln = mega_fuse_ln_enabled();
    if (!fuse_ln) {
        // Standalone LN1 dispatch (legacy; MEGA_FUSE_LN builds run LN1 inside
        // mega_qkv's local staging instead — one fewer dispatch per layer).
        err |= clSetKernelArg(s_k_ln_rows, 0, sizeof(cl_mem), &x_in0);
        err |= clSetKernelArg(s_k_ln_rows, 1, sizeof(cl_mem), &x_in1);
        err |= clSetKernelArg(s_k_ln_rows, 2, sizeof(cl_mem), &ln_w);
        err |= clSetKernelArg(s_k_ln_rows, 3, sizeof(cl_mem), &ln_b);
        err |= clSetKernelArg(s_k_ln_rows, 4, sizeof(cl_mem), &s_qkv_normed);
        err |= clSetKernelArg(s_k_ln_rows, 5, sizeof(float), &eps);
        if (err != CL_SUCCESS) { NNOPT_ERROR_FMT("qkv-ext: ln args %d (L=%d)", (int)err, layer_idx); return false; }
        const size_t gws_ln[1] = {(size_t)num_wg * (size_t)kMegaWG};
        const size_t lws_ln[1] = {(size_t)kMegaWG};
        err = clEnqueueNDRangeKernel(queue, s_k_ln_rows, 1, nullptr, gws_ln, lws_ln, 0, nullptr,
                                     KernelProfiler::event_for("mega_ln_rows", 0));
        if (err != CL_SUCCESS) { NNOPT_ERROR_FMT("qkv-ext: ln dispatch %d (L=%d)", (int)err, layer_idx); return false; }
    }
    const int rpw = 32;   // 3H/32 = 96 WGs/row
    err  = clSetKernelArg(s_k_qkv, 0, sizeof(cl_mem), &img);
    if (fuse_ln) err |= clSetKernelArg(s_k_qkv, 1, sizeof(cl_mem), &x_in0);   // raw residual row 0
    else         err |= clSetKernelArg(s_k_qkv, 1, sizeof(cl_mem), &s_qkv_normed);
    err |= clSetKernelArg(s_k_qkv, 2, sizeof(cl_mem), &s_q_ext);
    err |= clSetKernelArg(s_k_qkv, 3, sizeof(cl_mem), k_cache_r0);
    err |= clSetKernelArg(s_k_qkv, 4, sizeof(cl_mem), v_cache_r0);
    err |= clSetKernelArg(s_k_qkv, 5, sizeof(cl_mem), k_cache_r1);
    err |= clSetKernelArg(s_k_qkv, 6, sizeof(cl_mem), v_cache_r1);
    if (!s_sp_buf) { NNOPT_ERROR("qkv-ext: step-params buffer not written this step"); return false; }
    err |= clSetKernelArg(s_k_qkv, 7, sizeof(cl_mem), &s_sp_buf);   // sp[0]=start_pos
    err |= clSetKernelArg(s_k_qkv, 8, sizeof(int), &rpw);
    if (fuse_ln) {
        err |= clSetKernelArg(s_k_qkv,  9, sizeof(cl_mem), &x_in1);  // raw residual row 1
        err |= clSetKernelArg(s_k_qkv, 10, sizeof(cl_mem), &ln_w);
        err |= clSetKernelArg(s_k_qkv, 11, sizeof(cl_mem), &ln_b);
        err |= clSetKernelArg(s_k_qkv, 12, sizeof(float), &eps);
    }
    (void)start_pos;   // value now travels via the step-params buffer
    if (err != CL_SUCCESS) { NNOPT_ERROR_FMT("qkv-ext: qkv args %d (L=%d)", (int)err, layer_idx); return false; }
    const uint64_t qkv_bytes = (uint64_t)3 * H * H * 2;   // fp16 weight stream
    const bool ilvq = mega_row_ilv_enabled();
    const size_t gws_q[2] = {ilvq ? (size_t)num_wg * kMwgWG : (size_t)(3 * H / rpw) * kMwgWG,
                             ilvq ? (size_t)(3 * H / rpw)   : (size_t)num_wg};
    const size_t lws_q[2] = {(size_t)kMwgWG, 1};
    err = clEnqueueNDRangeKernel(queue, s_k_qkv, 2, nullptr, gws_q, lws_q, 0, nullptr,
                                 KernelProfiler::event_for("mega_qkv", qkv_bytes));
    if (err != CL_SUCCESS) { NNOPT_ERROR_FMT("qkv-ext: qkv dispatch %d (L=%d)", (int)err, layer_idx); return false; }
    return true;
}

// ── int8+outlier quantize/upload (once per layer per matrix) ────────────────
// w fp16→float from the blob. Per row: top-`nout` |w| become exact fp32
// outliers (zeroed in the int8 copy, indices SORTED ascending — fc2's
// split-K kernel range-filters per segment); the rest quantize with one
// fp32 scale per 128-wide K-group: scale = max|w_rest|/127.
static bool mega_i8o_prepare(OpenCLContext& cl_ctx, cl_command_queue queue,
                             Weights& weights, int layer_idx, int slot,
                             const std::string& key, int N, int K, int nout,
                             bool as_buffer) {
    if (s_i8o_img[layer_idx][slot]) return true;
    std::vector<float> w = weights.get_host_vec(key);
    if ((int)w.size() != N * K) { NNOPT_ERROR_FMT("i8o: %s size mismatch", key.c_str()); return false; }
    const int ngroups = K / 128;
    std::vector<int8_t> q((size_t)N * K, 0);
    std::vector<float> scales((size_t)N * ngroups, 0.0f);
    std::vector<uint16_t> oidx((size_t)N * nout, 0);
    std::vector<float> ovals((size_t)N * nout, 0.0f);
    std::vector<int> topk(nout);
    for (int n = 0; n < N; ++n) {
        float* row = &w[(size_t)n * K];
        // top-nout by |w| (partial selection over K)
        std::iota(topk.begin(), topk.end(), 0);
        std::vector<int> idxs(K);
        std::iota(idxs.begin(), idxs.end(), 0);
        std::partial_sort(idxs.begin(), idxs.begin() + nout, idxs.end(),
                          [&](int a, int b){ return std::fabs(row[a]) > std::fabs(row[b]); });
        std::sort(idxs.begin(), idxs.begin() + nout);   // ascending for the seg filter
        for (int j = 0; j < nout; ++j) {
            const int k = idxs[j];
            oidx[(size_t)n * nout + j] = (uint16_t)k;
            ovals[(size_t)n * nout + j] = row[k];
            row[k] = 0.0f;                              // excluded from int8
        }
        for (int g = 0; g < ngroups; ++g) {
            float m = 0.0f;
            for (int k = g * 128; k < (g + 1) * 128; ++k) m = std::max(m, std::fabs(row[k]));
            const float sc = (m > 0.0f) ? m / 127.0f : 0.0f;
            scales[(size_t)n * ngroups + g] = sc;
            const float inv = (sc > 0.0f) ? 1.0f / sc : 0.0f;
            for (int k = g * 128; k < (g + 1) * 128; ++k) {
                int v = (int)std::lround(row[k] * inv);
                q[(size_t)n * K + k] = (int8_t)std::min(127, std::max(-127, v));
            }
        }
    }
    if (as_buffer) {
        // dot8 path: OFFSET-BINARY uchar (q+128) + per-group signed sums for
        // the zero-point correction (the hardware dot is unsigned-byte only).
        // Integer group sums (the kernel corrects in INTEGER domain before
        // the float convert; a float-domain correction lost cb0 46→29 to
        // catastrophic cancellation).
        std::vector<int32_t> wsum((size_t)N * ngroups, 0);
        for (int n = 0; n < N; ++n)
            for (int g = 0; g < ngroups; ++g) {
                int32_t sum = 0;
                for (int k = g * 128; k < (g + 1) * 128; ++k) sum += q[(size_t)n * K + k];
                wsum[(size_t)n * ngroups + g] = sum;
            }
        // Weights stay PLAIN SIGNED int8 — probed semantics put the signed
        // operand on arg1 (only the activations need the +128 offset).
        cl_int berr = CL_SUCCESS;
        cl_mem wsb = clCreateBuffer(cl_ctx.context(), CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
                                    wsum.size() * sizeof(int32_t), wsum.data(), &berr);
        if (berr != CL_SUCCESS || !wsb) { NNOPT_ERROR_FMT("i8o: wsum %d (%s)", (int)berr, key.c_str()); return false; }
        s_i8o_wsum[layer_idx][slot] = wsb;
        cl_mem b = clCreateBuffer(cl_ctx.context(), CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
                                  q.size(), q.data(), &berr);
        if (berr != CL_SUCCESS || !b) { NNOPT_ERROR_FMT("i8o: buffer %d (%s)", (int)berr, key.c_str()); return false; }
        cl_mem sb2 = clCreateBuffer(cl_ctx.context(), CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
                                    scales.size() * sizeof(float), scales.data(), &berr);
        cl_mem ib2 = clCreateBuffer(cl_ctx.context(), CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
                                    oidx.size() * sizeof(uint16_t), oidx.data(), &berr);
        cl_mem vb2 = clCreateBuffer(cl_ctx.context(), CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
                                    ovals.size() * sizeof(float), ovals.data(), &berr);
        if (!sb2 || !ib2 || !vb2) { NNOPT_ERROR_FMT("i8o: aux upload (%s)", key.c_str()); return false; }
        s_i8o_img[layer_idx][slot] = b;
        s_i8o_scale[layer_idx][slot] = sb2;
        s_i8o_oidx[layer_idx][slot] = ib2;
        s_i8o_oval[layer_idx][slot] = vb2;
        (void)queue;
        return true;
    }
    // CL_RGBA + CL_UNSIGNED_INT32 image: one texel = 16 int8 weights. Rides
    // the texture L1 (CFG-twin dedup) with a NATIVE 128-bit texel format —
    // CL_SIGNED_INT8 texels were a 0.71 GB/s slow path, and plain buffers
    // lost the dedup (3.3× slower than fp16 across 4 variants).
    cl_image_format fmt{};
    fmt.image_channel_order = CL_RGBA;
    fmt.image_channel_data_type = CL_UNSIGNED_INT32;
    cl_image_desc desc{};
    desc.image_type = CL_MEM_OBJECT_IMAGE2D;
    desc.image_width = (size_t)K / 16;
    desc.image_height = (size_t)N;
    cl_int err = CL_SUCCESS;
    cl_mem img = clCreateImage(cl_ctx.context(), CL_MEM_READ_ONLY, &fmt, &desc, nullptr, &err);
    if (err != CL_SUCCESS || !img) { NNOPT_ERROR_FMT("i8o: clCreateImage %d (%s)", (int)err, key.c_str()); return false; }
    const size_t origin[3] = {0, 0, 0};
    const size_t region[3] = {(size_t)K / 16, (size_t)N, 1};
    err = clEnqueueWriteImage(queue, img, CL_TRUE, origin, region, (size_t)K, 0, q.data(), 0, nullptr, nullptr);
    if (err != CL_SUCCESS) { NNOPT_ERROR_FMT("i8o: WriteImage %d (%s)", (int)err, key.c_str()); clReleaseMemObject(img); return false; }
    auto up = [&](const void* d, size_t bytes, const char* nm) -> cl_mem {
        cl_mem m = clCreateBuffer(cl_ctx.context(), CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, bytes, (void*)d, &err);
        if (err != CL_SUCCESS) { NNOPT_ERROR_FMT("i8o: upload %s %d", nm, (int)err); return nullptr; }
        return m;
    };
    cl_mem sb = up(scales.data(), scales.size() * sizeof(float), "scales");
    cl_mem ib = up(oidx.data(), oidx.size() * sizeof(uint16_t), "oidx");
    cl_mem vb = up(ovals.data(), ovals.size() * sizeof(float), "ovals");
    if (!sb || !ib || !vb) { clReleaseMemObject(img); return false; }
    s_i8o_img[layer_idx][slot] = img;
    s_i8o_scale[layer_idx][slot] = sb;
    s_i8o_oidx[layer_idx][slot] = ib;
    s_i8o_oval[layer_idx][slot] = vb;
    return true;
}

// ── lm_heads own-GEMV dispatch (replaces the CLBlast GEMM) ───────────────────
// Lazily creates the kernel + an image2d VIEW over the heads_cat buffer
// ([N, H] fp16 → [N, H/4] RGBA-half texels; contiguous rows, no copy).
// Returns false on any failure so the caller can fall back to CLBlast.
static cl_kernel s_k_lmheads = nullptr;
static cl_mem s_lmheads_img = nullptr;
extern "C" bool mega_lmheads_dispatch(OpenCLContext& cl_ctx, cl_command_queue queue,
                                      cl_mem heads_cat, cl_mem x, cl_mem logits,
                                      int num_rows, int N) {
    if (!s_mega_prog || !mega_tex_enabled()) return false;
    cl_int err = CL_SUCCESS;
    if (!s_k_lmheads) {
        s_k_lmheads = clCreateKernel(s_mega_prog, "mega_lmheads", &err);
        if (err != CL_SUCCESS) { s_k_lmheads = nullptr; return false; }
    }
    if (!s_lmheads_img) {
        const int H = MODEL_CONFIG::DECODER_HIDDEN_SIZE;
        cl_image_format fmt{};
        fmt.image_channel_order = CL_RGBA;
        fmt.image_channel_data_type = CL_HALF_FLOAT;
        cl_image_desc desc{};
        desc.image_type = CL_MEM_OBJECT_IMAGE2D;
        desc.image_width = (size_t)H / 4;
        desc.image_height = (size_t)N;
        desc.image_row_pitch = (size_t)H / 4 * 8;
        desc.buffer = heads_cat;
        s_lmheads_img = clCreateImage(cl_ctx.context(), CL_MEM_READ_ONLY, &fmt, &desc, nullptr, &err);
        if (err != CL_SUCCESS || !s_lmheads_img) { s_lmheads_img = nullptr; return false; }
    }
    const int rpw = 32;   // 8192/32 = 256 WGs per row
    err  = clSetKernelArg(s_k_lmheads, 0, sizeof(cl_mem), &s_lmheads_img);
    err |= clSetKernelArg(s_k_lmheads, 1, sizeof(cl_mem), &x);
    err |= clSetKernelArg(s_k_lmheads, 2, sizeof(cl_mem), &logits);
    err |= clSetKernelArg(s_k_lmheads, 3, sizeof(int), &N);
    err |= clSetKernelArg(s_k_lmheads, 4, sizeof(int), &rpw);
    if (err != CL_SUCCESS) return false;
    const uint64_t bytes = (uint64_t)N * MODEL_CONFIG::DECODER_HIDDEN_SIZE * 2;
    const bool ilvh = mega_row_ilv_enabled();
    const size_t gws[2] = {ilvh ? (size_t)num_rows * kMwgWG : (size_t)(N / rpw) * kMwgWG,
                           ilvh ? (size_t)(N / rpw)         : (size_t)num_rows};
    const size_t lws[2] = {(size_t)kMwgWG, 1};
    return clEnqueueNDRangeKernel(queue, s_k_lmheads, 2, nullptr, gws, lws, 0, nullptr,
                                  KernelProfiler::event_for("mega_lmheads", bytes)) == CL_SUCCESS;
}

// ── Stage-2 split dispatch chain (replaces the decoder_layer_mega enqueue) ──
static bool mega_split_ensure_scratch(OpenCLContext& cl_ctx) {
    if (s_attn_scr) return true;
    const size_t H = MODEL_CONFIG::DECODER_HIDDEN_SIZE;
    cl_int err = CL_SUCCESS;
    auto mk = [&](const char* nm) -> cl_mem {
        cl_mem m = clCreateBuffer(cl_ctx.context(), CL_MEM_READ_WRITE, 2 * H * sizeof(float), nullptr, &err);
        if (err != CL_SUCCESS) { NNOPT_ERROR_FMT("attn-split: %s scratch alloc %d", nm, (int)err); return nullptr; }
        return m;
    };
    s_attn_scr = mk("attn"); s_resid2 = mk("resid2"); s_ln2n = mk("ln2n");
    return s_attn_scr && s_resid2 && s_ln2n;
}

// Cached image2d VIEWs over KV buffers ([rows, H] fp16 → [rows, H/4] RGBA half
// texels, zero-copy — the lm_heads recipe). Keyed by buffer handle: self-attn
// cache buffers are process-lifetime; cross-KV buffers are per-generation and
// their views are dropped in mega_reset_cross_kv.
static std::unordered_map<cl_mem, cl_mem> s_kv_img_views;
static cl_mem mega_kv_image_view(OpenCLContext& cl_ctx, cl_mem buf) {
    auto it = s_kv_img_views.find(buf);
    if (it != s_kv_img_views.end()) return it->second;
    size_t bytes = 0;
    if (clGetMemObjectInfo(buf, CL_MEM_SIZE, sizeof(bytes), &bytes, nullptr) != CL_SUCCESS || !bytes)
        return nullptr;
    const size_t row_pitch = (size_t)MODEL_CONFIG::DECODER_HIDDEN_SIZE * sizeof(nnopt_storage_t);
    cl_image_format fmt{};
    fmt.image_channel_order = CL_RGBA;
    fmt.image_channel_data_type = (sizeof(nnopt_storage_t) == 2) ? CL_HALF_FLOAT : CL_FLOAT;
    cl_image_desc desc{};
    desc.image_type = CL_MEM_OBJECT_IMAGE2D;
    desc.image_width = (size_t)MODEL_CONFIG::DECODER_HIDDEN_SIZE / 4;
    desc.image_height = bytes / row_pitch;
    desc.image_row_pitch = row_pitch;
    desc.buffer = buf;
    cl_int err = CL_SUCCESS;
    cl_mem img = clCreateImage(cl_ctx.context(), CL_MEM_READ_ONLY, &fmt, &desc, nullptr, &err);
    if (err != CL_SUCCESS || !img) { NNOPT_ERROR_FMT("attn-tex: kv view create %d", (int)err); return nullptr; }
    s_kv_img_views.emplace(buf, img);
    return img;
}

static bool mega_split_attn(cl_command_queue queue, const char* label, cl_kernel kk,
                            cl_mem q_g,
                            cl_mem k0, cl_mem v0, cl_mem k1, cl_mem v1,
                            cl_mem out_g, cl_mem seq_ptr, int seq_bias,
                            float scale, int num_wg) {
    cl_int err = CL_SUCCESS;
    err |= clSetKernelArg(kk, 0, sizeof(cl_mem), &q_g);
    err |= clSetKernelArg(kk, 1, sizeof(cl_mem), &k0);
    err |= clSetKernelArg(kk, 2, sizeof(cl_mem), &v0);
    err |= clSetKernelArg(kk, 3, sizeof(cl_mem), &k1);
    err |= clSetKernelArg(kk, 4, sizeof(cl_mem), &v1);
    err |= clSetKernelArg(kk, 5, sizeof(cl_mem), &out_g);
    err |= clSetKernelArg(kk, 6, sizeof(cl_mem), &seq_ptr);  // seq = seq_ptr[0]+seq_bias
    err |= clSetKernelArg(kk, 7, sizeof(int), &seq_bias);
    err |= clSetKernelArg(kk, 8, sizeof(float), &scale);
    if (err != CL_SUCCESS) { NNOPT_ERROR_FMT("attn-split: %s args %d", label, (int)err); return false; }
    // Head-parallel: one WG per (head, row) — 32 WGs (was 2 WGs looping 16
    // heads serially; that starved the GPU once the KV scan grew with t and
    // made attn_self the #1 decode kernel at 27.7% of a full clip).
    const size_t gws[2] = {(size_t)MODEL_CONFIG::NUM_ATTENTION_HEADS * 64, (size_t)num_wg};
    const size_t lws[2] = {64, 1};
    err = clEnqueueNDRangeKernel(queue, kk, 2, nullptr, gws, lws, 0, nullptr,
                                 KernelProfiler::event_for(label, 0));
    if (err != CL_SUCCESS) { NNOPT_ERROR_FMT("attn-split: %s dispatch %d", label, (int)err); return false; }
    return true;
}

static bool mega_split_proj(cl_command_queue queue, const char* label, cl_mem W_img,
                            cl_mem x_g, cl_mem y_g, cl_mem xin0, cl_mem xin1,
                            cl_mem resid_g, int mode, int num_wg,
                            // MEGA_FUSE_LN extras: do_ln=1 (cq) LayerNorms the
                            // staged x_g row in-kernel. ln bufs must be valid
                            // cl_mems even when do_ln=0 (kernel arg contract).
                            cl_mem ln_w, cl_mem ln_b, int do_ln) {
    const int H = MODEL_CONFIG::DECODER_HIDDEN_SIZE;
    const int rpw = 32;
    cl_int err = CL_SUCCESS;
    err |= clSetKernelArg(s_k_proj, 0, sizeof(cl_mem), &W_img);
    err |= clSetKernelArg(s_k_proj, 1, sizeof(cl_mem), &x_g);
    err |= clSetKernelArg(s_k_proj, 2, sizeof(cl_mem), &y_g);
    err |= clSetKernelArg(s_k_proj, 3, sizeof(cl_mem), &xin0);
    err |= clSetKernelArg(s_k_proj, 4, sizeof(cl_mem), &xin1);
    err |= clSetKernelArg(s_k_proj, 5, sizeof(cl_mem), &resid_g);
    err |= clSetKernelArg(s_k_proj, 6, sizeof(int), &mode);
    err |= clSetKernelArg(s_k_proj, 7, sizeof(int), &rpw);
    if (mega_fuse_ln_enabled()) {
        const float eps_ln = MODEL_CONFIG::LAYER_NORM_EPS;
        err |= clSetKernelArg(s_k_proj,  8, sizeof(cl_mem), &ln_w);
        err |= clSetKernelArg(s_k_proj,  9, sizeof(cl_mem), &ln_b);
        err |= clSetKernelArg(s_k_proj, 10, sizeof(float), &eps_ln);
        err |= clSetKernelArg(s_k_proj, 11, sizeof(int), &do_ln);
    }
    if (err != CL_SUCCESS) { NNOPT_ERROR_FMT("attn-split: %s args %d", label, (int)err); return false; }
    const uint64_t bytes = (uint64_t)H * H * 2;
    const bool ilvp = mega_row_ilv_enabled();
    const size_t gws[2] = {ilvp ? (size_t)num_wg * kMwgWG : (size_t)(H / rpw) * kMwgWG,
                           ilvp ? (size_t)(H / rpw)       : (size_t)num_wg};
    const size_t lws[2] = {(size_t)kMwgWG, 1};
    err = clEnqueueNDRangeKernel(queue, s_k_proj, 2, nullptr, gws, lws, 0, nullptr,
                                 KernelProfiler::event_for(label, bytes));
    if (err != CL_SUCCESS) { NNOPT_ERROR_FMT("attn-split: %s dispatch %d", label, (int)err); return false; }
    return true;
}

static bool mega_split_ln_f32(cl_command_queue queue, const char* label, cl_mem x_g,
                              cl_mem ln_w, cl_mem ln_b, cl_mem out_g, int num_wg) {
    const float eps = MODEL_CONFIG::LAYER_NORM_EPS;
    cl_int err = CL_SUCCESS;
    err |= clSetKernelArg(s_k_ln_f32, 0, sizeof(cl_mem), &x_g);
    err |= clSetKernelArg(s_k_ln_f32, 1, sizeof(cl_mem), &ln_w);
    err |= clSetKernelArg(s_k_ln_f32, 2, sizeof(cl_mem), &ln_b);
    err |= clSetKernelArg(s_k_ln_f32, 3, sizeof(cl_mem), &out_g);
    err |= clSetKernelArg(s_k_ln_f32, 4, sizeof(float), &eps);
    if (err != CL_SUCCESS) { NNOPT_ERROR_FMT("attn-split: %s args %d", label, (int)err); return false; }
    const size_t gws[1] = {(size_t)num_wg * (size_t)kMegaWG};
    const size_t lws[1] = {(size_t)kMegaWG};
    err = clEnqueueNDRangeKernel(queue, s_k_ln_f32, 1, nullptr, gws, lws, 0, nullptr,
                                 KernelProfiler::event_for(label, 0));
    if (err != CL_SUCCESS) { NNOPT_ERROR_FMT("attn-split: %s dispatch %d", label, (int)err); return false; }
    return true;
}

// The full per-layer split chain (steps 3-9; ln1+qkv ran in qkv_predispatch,
// fc1/fc2 follow via mega_ffn_dispatch). Writes s_ffn_normed + s_ffn_resid —
// exactly the megakernel's external-FFN handoff contract.
static bool mega_split_dispatch(OpenCLContext& cl_ctx, cl_command_queue queue, int layer_idx,
                                cl_mem x_in0, cl_mem x_in1,
                                cl_mem* k_cache_r0, cl_mem* v_cache_r0,
                                cl_mem* k_cache_r1, cl_mem* v_cache_r1,
                                int start_pos, int num_wg, int enc_len) {
    const float scale = 1.0f / std::sqrt((float)MODEL_CONFIG::HEAD_DIM);
    (void)start_pos;   // self-attn seq length now travels via the step-params buffer
    if (!s_sp_buf) { NNOPT_ERROR("attn-split: step-params buffer not written this step"); return false; }
    cl_mem enc_ptr = mega_enc_len_buf(cl_ctx, queue, enc_len);
    if (!enc_ptr) { NNOPT_ERROR("attn-split: enc-len buffer failed"); return false; }
    cl_mem o_img  = s_w_img[layer_idx][3];
    cl_mem cq_img = s_w_img[layer_idx][4];
    cl_mem co_img = s_w_img[layer_idx][5];
    cl_mem enc_w = s_split_enc_ln_w[layer_idx], enc_b = s_split_enc_ln_b[layer_idx];
    cl_mem fin_w = s_split_fin_ln_w[layer_idx], fin_b = s_split_fin_ln_b[layer_idx];
    if (!o_img || !cq_img || !co_img || !enc_w || !enc_b || !fin_w || !fin_b) {
        NNOPT_ERROR_FMT("attn-split: layer %d not prepared", layer_idx); return false;
    }
    cl_mem k_cross0 = g_kcross[0 * kMegaLayers + layer_idx];
    cl_mem v_cross0 = g_vcross[0 * kMegaLayers + layer_idx];
    cl_mem k_cross1 = g_kcross[1 * kMegaLayers + layer_idx];
    cl_mem v_cross1 = g_vcross[1 * kMegaLayers + layer_idx];
    // self-attn core over the caches (seq = sp[0]+1), q from mega_qkv.
    // NNOPT_ATTN_TEX: K goes in as an image2d view (texture L1); V stays buffer.
    cl_mem k_self0 = *k_cache_r0, k_self1 = *k_cache_r1;
    if (mega_attn_tex_enabled()) {
        k_self0 = mega_kv_image_view(cl_ctx, *k_cache_r0);
        k_self1 = mega_kv_image_view(cl_ctx, *k_cache_r1);
        if (!k_self0 || !k_self1) return false;
    }
    if (!mega_split_attn(queue, "mega_attn_self", s_k_attn_core, s_q_ext,
                         k_self0, *v_cache_r0, k_self1, *v_cache_r1,
                         s_attn_scr, s_sp_buf, /*seq_bias=*/1, scale, num_wg)) return false;
    const bool fuse_ln = mega_fuse_ln_enabled();
    // o-proj + residual(x_in) → resid2
    if (!mega_split_proj(queue, "mega_proj_o", o_img, s_attn_scr, s_resid2,
                         x_in0, x_in1, s_resid2, /*mode=*/1, num_wg,
                         enc_w, enc_b, /*do_ln=*/0)) return false;
    // LN2 → cq GEMV (q_ext reused as cq output — in-order queue, self-attn
    // already consumed it). MEGA_FUSE_LN: LN2 runs inside the cq GEMV's
    // staging (x_g = raw resid2, do_ln=1) — no standalone dispatch.
    if (!fuse_ln &&
        !mega_split_ln_f32(queue, "mega_ln2", s_resid2, enc_w, enc_b, s_ln2n, num_wg)) return false;
    if (!mega_split_proj(queue, "mega_proj_cq", cq_img,
                         fuse_ln ? s_resid2 : s_ln2n, s_q_ext,
                         x_in0, x_in1, s_resid2, /*mode=*/0, num_wg,
                         enc_w, enc_b, /*do_ln=*/fuse_ln ? 1 : 0)) return false;
    // cross-attn core over the precomputed cross K/V (seq = enc_len, via its
    // own constant buffer — distinct from the per-step sp buffer). The clone
    // kernel keeps the two dispatch streams' arg state independent. Cross-K is
    // read-only per generation — the safest texture-view candidate.
    cl_mem k_x0 = k_cross0, k_x1 = k_cross1;
    if (mega_attn_tex_enabled()) {
        k_x0 = mega_kv_image_view(cl_ctx, k_cross0);
        k_x1 = mega_kv_image_view(cl_ctx, k_cross1);
        if (!k_x0 || !k_x1) return false;
    }
    if (!mega_split_attn(queue, "mega_attn_cross", s_k_attn_core_x, s_q_ext,
                         k_x0, v_cross0, k_x1, v_cross1,
                         s_attn_scr, enc_ptr, /*seq_bias=*/0, scale, num_wg)) return false;
    // co-proj + residual(resid2) → ffn_resid (the FFN handoff buffer directly)
    if (!mega_split_proj(queue, "mega_proj_co", co_img, s_attn_scr, s_ffn_resid,
                         x_in0, x_in1, s_resid2, /*mode=*/2, num_wg,
                         fin_w, fin_b, /*do_ln=*/0)) return false;
    // LN3 → ffn_normed; fc1/fc2 take over from here. MEGA_FUSE_LN: LN3 runs
    // inside fc1's staging (mega_ffn_dispatch feeds it raw ffn_resid + fin LN).
    if (!fuse_ln &&
        !mega_split_ln_f32(queue, "mega_ln3", s_ffn_resid, fin_w, fin_b, s_ffn_normed, num_wg)) return false;
    return true;
}

// ── Stage 6: int8 weight quantization ────────────────────────────────────────
// NNOPT_INT8=1 builds the megakernel a SECOND time with -D MEGA_INT8=1 (the 8
// GEMV weights become int8 + per-output-row fp16 scale) and routes layers
// through the int8 kernel. Weights are quantized ONCE at first use from the fp16
// tensors: scale[n] = max(|W[n,:]|)/127, Wq[n,k] = round(W[n,k]/scale[n]). int8
// halves the per-token decoder weight traffic (~820→410 MB/token), the memory-
// bound wall (PERFORMANCE_ANALYSIS §3). fp32 accumulation in-kernel.
static bool      s_int8 = false;       // full int8 (all 8 GEMV weights)
static bool      s_int8_ffn = false;    // FFN-only int8 (fc1/fc2; attn stays fp16)
static cl_program s_mega_prog_i8 = nullptr;
static cl_kernel  s_k_mega_i8 = nullptr;
static cl_program s_mega_prog_i8f = nullptr;
static cl_kernel  s_k_mega_i8f = nullptr;
// Per-(layer, weight) int8 data + scale buffers. 8 GEMV weights/layer.
//   weight index: 0=q 1=k 2=v 3=o 4=cq 5=co 6=fc1 7=fc2
static cl_mem s_w_i8[64][8] = {};
static cl_mem s_w_scale[64][8] = {};
static bool   s_layer_i8_ready[64] = {};

static bool mega_build(OpenCLContext& cl_ctx) {
    if (s_mega_prog) return true;
    s_int8 = [](){ const char* e = std::getenv("NNOPT_INT8"); return e && e[0]=='1'; }();
    s_int8_ffn = [](){ const char* e = std::getenv("NNOPT_INT8_FFN"); return e && e[0]=='1'; }() && !s_int8;
    // Lever 1: build the kernel with -D MEGA_WG=<workgroup size> so the kernel's
    // reqd_work_group_size, local-memory tiling, and reduction match the host
    // dispatch's lws. Default 512 (2× the old 256 streaming threads + coalesced
    // reads). The megakernel's local budget is 3*1024 + 4096 + MEGA_WG floats;
    // clamp WG down if it would exceed the device's reported local-mem size or
    // max workgroup size. One-time diagnostic logs the device limits + achieved
    // budget so GB/s analysis has the occupancy context.
    {
        const size_t lmem = cl_ctx.local_mem_size();
        const size_t maxwg = cl_ctx.max_work_group_size();
        int wg = mega_wg_size();
        const size_t need = ((size_t)3*1024 + 4096 + (size_t)wg) * sizeof(float);
        // Direct stderr (works in release; CHECKPOINT macro is a no-op there).
        std::fprintf(stderr, "MEGA_INFO device='%s' local_mem=%zuB max_wg=%zu MEGA_WG=%d KGROUP=%d local_need=%zuB\n",
                     cl_ctx.device_name().c_str(), lmem, maxwg, wg, mega_kgroup(), need);
        if ((size_t)wg > maxwg || need > lmem) {
            NNOPT_ERROR_FMT("mega: MEGA_WG=%d infeasible (max_wg=%zu, local need=%zuB > %zuB); "
                            "set NNOPT_MEGA_WG to a smaller power of two", wg, maxwg, need, lmem);
            // Fall back to 256 (always fits: 3*1024+4096+256 = 7424 floats = 29.7KB).
        }
    }
    char wgflag[512]; std::snprintf(wgflag, sizeof(wgflag), "-D MEGA_WG=%d -D MEGA_KGROUP=%d%s -D MEGA_MWG_VEC=%d -D MEGA_VEC=%d%s%s%s%s -D MEGA_FC2_RPW=%d",
                                    mega_wg_size(), mega_kgroup(),
                                    mwg_ffn_enabled() ? " -D MEGA_FFN_EXTERNAL=1" : "",
                                    mwg_vec(), mega_vec(),
                                    mega_tex_enabled() ? " -D MEGA_TEX=1" : "",
                                    mega_tex_fc2_enabled() ? " -D MEGA_TEX_FC2=1" : "",
                                    mega_fc2_fuse_enabled() ? " -D MEGA_FC2_FUSE2=1" : "",
                                    mega_qkv_ext_enabled() ? " -D MEGA_QKV_EXTERNAL=1" : "",
                                    mwg_rpw2());
    if (mega_int8o_mode() >= 4) {
        char d8[64]; std::snprintf(d8, sizeof(d8), " -D MEGA_DOT8_FN=%s", mega_dot8_fn());
        std::strncat(wgflag, d8, sizeof(wgflag) - std::strlen(wgflag) - 1);
    }
    if (mega_attn_tex_enabled())
        std::strncat(wgflag, " -D MEGA_ATTN_TEX=1", sizeof(wgflag) - std::strlen(wgflag) - 1);
    if (mega_fuse_ln_enabled())
        std::strncat(wgflag, " -D MEGA_FUSE_LN=1", sizeof(wgflag) - std::strlen(wgflag) - 1);
    if (mega_row_ilv_enabled())
        std::strncat(wgflag, " -D MEGA_ROW_ILV=1", sizeof(wgflag) - std::strlen(wgflag) - 1);
    { const char* e = std::getenv("NNOPT_X4LDS");   // issue-bound probe: float4 LDS staging in fc1
      if (e && e[0] == '1')
        std::strncat(wgflag, " -D MEGA_X4LDS=1", sizeof(wgflag) - std::strlen(wgflag) - 1); }
    { const char* e = std::getenv("NNOPT_H4ACC");   // fp16-ALU probe: half-accumulate fc1 (tf-gated)
      if (e && e[0] == '1')
        std::strncat(wgflag, " -D MEGA_H4ACC=1", sizeof(wgflag) - std::strlen(wgflag) - 1); }
    s_mega_prog = cl_ctx.build_program_from_file("kernels/decoder_layer_mega.cl", wgflag); // PROGRAM-INIT-OK
    if (!s_mega_prog) { NNOPT_ERROR("mega: build decoder_layer_mega.cl failed"); return false; }
    cl_int err = CL_SUCCESS;
    s_k_mega = clCreateKernel(s_mega_prog, "decoder_layer_mega", &err);
    if (err != CL_SUCCESS) { NNOPT_ERROR_FMT("mega: clCreateKernel(decoder_layer_mega) %d", (int)err); return false; }
    s_k_precompute = clCreateKernel(s_mega_prog, "decoder_cross_kv_precompute", &err);
    if (err != CL_SUCCESS) { NNOPT_ERROR_FMT("mega: clCreateKernel(decoder_cross_kv_precompute) %d", (int)err); return false; }
    if (mwg_ffn_enabled()) {
        s_k_fc1 = clCreateKernel(s_mega_prog, "mega_ffn_fc1", &err);
        if (err != CL_SUCCESS) { NNOPT_ERROR_FMT("mega: clCreateKernel(mega_ffn_fc1) %d", (int)err); return false; }
        if (mega_tex_enabled() && mega_row_fuse_enabled()) {
            s_k_fc1_rf = clCreateKernel(s_mega_prog, "mega_ffn_fc1_rf", &err);
            if (err != CL_SUCCESS) { NNOPT_ERROR_FMT("mega: clCreateKernel(mega_ffn_fc1_rf) %d", (int)err); return false; }
        }
        s_k_fc2 = clCreateKernel(s_mega_prog, "mega_ffn_fc2", &err);
        if (err != CL_SUCCESS) { NNOPT_ERROR_FMT("mega: clCreateKernel(mega_ffn_fc2) %d", (int)err); return false; }
        std::fprintf(stderr, "MEGA_INFO mwg-ffn ENABLED: fc1 %d WGs/row (rpw=%d), fc2 %d WGs/row (rpw=%d), WG=%d\n",
                     MODEL_CONFIG::FFN_DIM / mwg_rpw1(), mwg_rpw1(),
                     MODEL_CONFIG::DECODER_HIDDEN_SIZE / mwg_rpw2(), mwg_rpw2(), kMwgWG);
    }
    if (mega_qkv_ext_enabled()) {
        s_k_ln_rows = clCreateKernel(s_mega_prog, "mega_ln_rows", &err);
        if (err != CL_SUCCESS) { NNOPT_ERROR_FMT("mega: clCreateKernel(mega_ln_rows) %d", (int)err); return false; }
        s_k_qkv = clCreateKernel(s_mega_prog, "mega_qkv", &err);
        if (err != CL_SUCCESS) { NNOPT_ERROR_FMT("mega: clCreateKernel(mega_qkv) %d", (int)err); return false; }
        std::fprintf(stderr, "MEGA_INFO qkv-ext ENABLED: %d WGs/row (rpw=32), packed [3H, H/4] tex\n",
                     3 * MODEL_CONFIG::DECODER_HIDDEN_SIZE / 32);
    }
    if (mega_attn_split_enabled()) {
        s_k_attn_core = clCreateKernel(s_mega_prog, "mega_attn_core", &err);
        if (err != CL_SUCCESS) { NNOPT_ERROR_FMT("mega: clCreateKernel(mega_attn_core) %d", (int)err); return false; }
        s_k_attn_core_x = clCreateKernel(s_mega_prog, "mega_attn_core", &err);  // cross-attn clone
        if (err != CL_SUCCESS) { NNOPT_ERROR_FMT("mega: clCreateKernel(mega_attn_core clone) %d", (int)err); return false; }
        s_k_proj = clCreateKernel(s_mega_prog, "mega_proj", &err);
        if (err != CL_SUCCESS) { NNOPT_ERROR_FMT("mega: clCreateKernel(mega_proj) %d", (int)err); return false; }
        s_k_ln_f32 = clCreateKernel(s_mega_prog, "mega_ln_rows_f32", &err);
        if (err != CL_SUCCESS) { NNOPT_ERROR_FMT("mega: clCreateKernel(mega_ln_rows_f32) %d", (int)err); return false; }
        std::fprintf(stderr, "MEGA_INFO attn-split ENABLED: megakernel dissolved into per-stage kernels\n");
    }
    if (mega_fc2_splitk_enabled()) {
        s_k_fc2_sk = clCreateKernel(s_mega_prog, "mega_ffn_fc2_sk", &err);
        if (err != CL_SUCCESS) { NNOPT_ERROR_FMT("mega: clCreateKernel(mega_ffn_fc2_sk) %d", (int)err); return false; }
        s_k_fc2_sk_red = clCreateKernel(s_mega_prog, "mega_ffn_fc2_sk_reduce", &err);
        if (err != CL_SUCCESS) { NNOPT_ERROR_FMT("mega: clCreateKernel(mega_ffn_fc2_sk_reduce) %d", (int)err); return false; }
        std::fprintf(stderr, "MEGA_INFO fc2-splitk ENABLED: KSEG=%d\n", kFc2KSeg);
    }
    if (mega_int8o_enabled() && mega_int8o_mode() != 4) {
        s_k_fc1_i8o = clCreateKernel(s_mega_prog, "mega_ffn_fc1_i8o", &err);
        if (err != CL_SUCCESS) { NNOPT_ERROR_FMT("mega: clCreateKernel(mega_ffn_fc1_i8o) %d", (int)err); return false; }
        s_k_fc2_i8o = clCreateKernel(s_mega_prog, "mega_ffn_fc2_sk_i8o", &err);
        if (err != CL_SUCCESS) { NNOPT_ERROR_FMT("mega: clCreateKernel(mega_ffn_fc2_sk_i8o) %d", (int)err); return false; }
        std::fprintf(stderr, "MEGA_INFO int8-outlier FFN ENABLED (nout fc1=%d fc2=%d)\n", kI8oNoutFc1, kI8oNoutFc2);
    }
    if (mega_int8o_mode() >= 4) {
        s_k_fc1_i8d = clCreateKernel(s_mega_prog, "mega_ffn_fc1_i8d", &err);
        if (err != CL_SUCCESS) { NNOPT_ERROR_FMT("mega: clCreateKernel(mega_ffn_fc1_i8d) %d", (int)err); return false; }
        s_k_actq = clCreateKernel(s_mega_prog, "i8o_actq", &err);
        if (err != CL_SUCCESS) { NNOPT_ERROR_FMT("mega: clCreateKernel(i8o_actq) %d", (int)err); return false; }
        std::fprintf(stderr, "MEGA_INFO dot8 fc1 ENABLED (fn=%s)\n", mega_dot8_fn());
    }
    if (s_int8) {
        char i8flag[80]; std::snprintf(i8flag, sizeof(i8flag), "-D MEGA_INT8=1 -D MEGA_WG=%d -D MEGA_KGROUP=%d", mega_wg_size(), mega_kgroup());
        s_mega_prog_i8 = cl_ctx.build_program_from_file("kernels/decoder_layer_mega.cl", i8flag);
        if (!s_mega_prog_i8) { NNOPT_ERROR("mega: build int8 variant failed"); return false; }
        s_k_mega_i8 = clCreateKernel(s_mega_prog_i8, "decoder_layer_mega", &err);
        if (err != CL_SUCCESS) { NNOPT_ERROR_FMT("mega: clCreateKernel(int8) %d", (int)err); return false; }
        NNOPT_CHECKPOINT("mega: int8 weight path ENABLED (NNOPT_INT8=1)");
    }
    if (s_int8_ffn) {
        char i8fflag[84]; std::snprintf(i8fflag, sizeof(i8fflag), "-D MEGA_INT8_FFN=1 -D MEGA_WG=%d -D MEGA_KGROUP=%d", mega_wg_size(), mega_kgroup());
        s_mega_prog_i8f = cl_ctx.build_program_from_file("kernels/decoder_layer_mega.cl", i8fflag);
        if (!s_mega_prog_i8f) { NNOPT_ERROR("mega: build int8-ffn variant failed"); return false; }
        s_k_mega_i8f = clCreateKernel(s_mega_prog_i8f, "decoder_layer_mega", &err);
        if (err != CL_SUCCESS) { NNOPT_ERROR_FMT("mega: clCreateKernel(int8-ffn) %d", (int)err); return false; }
        NNOPT_CHECKPOINT("mega: int8-FFN weight path ENABLED (NNOPT_INT8_FFN=1)");
    }
    return true;
}

// Quantize one [N,K] fp16 weight (by key) to int8 + per-row fp16 scale, cached
// per (layer, weight-slot). Returns false on failure.
static bool mega_quantize_weight(OpenCLContext& cl_ctx, Weights& weights,
                                 int layer_idx, int slot, const std::string& key) {
    if (s_w_i8[layer_idx][slot]) return true;   // already quantized
    const std::vector<int> sh = weights.get_shape(key);
    if (sh.size() != 2) { NNOPT_ERROR_FMT("int8: bad shape for %s", key.c_str()); return false; }
    const int N = sh[0], K = sh[1];
    const std::vector<float> w = weights.get_host_vec(key);
    if ((int)w.size() != N * K) { NNOPT_ERROR_FMT("int8: size mismatch %s", key.c_str()); return false; }
    // PER-GROUP int8: one scale per QGROUP-wide column block (must match the
    // kernel's MEGA_QGROUP). Far more accurate than per-row on a deep decoder.
    constexpr int QGROUP = 128;
    if (K % QGROUP != 0) { NNOPT_ERROR_FMT("int8: K=%d not %d-divisible (%s)", K, QGROUP, key.c_str()); return false; }
    const int ngroups = K / QGROUP;
    std::vector<signed char> wq((size_t)N * K);
    std::vector<nnopt_storage_t> sc((size_t)N * ngroups);
    for (int n = 0; n < N; ++n) {
        const float* row = &w[(size_t)n * K];
        for (int g = 0; g < ngroups; ++g) {
            float amax = 0.0f;
            const int kb = g * QGROUP;
            for (int j = 0; j < QGROUP; ++j) { float a = std::fabs(row[kb + j]); if (a > amax) amax = a; }
            float scale = (amax > 0.0f) ? (amax / 127.0f) : 1.0f;
            float inv = 1.0f / scale;
            for (int j = 0; j < QGROUP; ++j) {
                int q = (int)std::lrint(row[kb + j] * inv);
                if (q > 127) q = 127; else if (q < -127) q = -127;
                wq[(size_t)n * K + kb + j] = (signed char)q;
            }
#ifdef NNOPT_USE_FP16
            sc[(size_t)n * ngroups + g] = nnopt_f32_to_f16(scale);
#else
            sc[(size_t)n * ngroups + g] = scale;
#endif
        }
    }
    cl_int err = CL_SUCCESS;
    s_w_i8[layer_idx][slot] = clCreateBuffer(cl_ctx.context(), CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
                                             wq.size(), wq.data(), &err);
    if (err != CL_SUCCESS) { NNOPT_ERROR_FMT("int8: wq alloc %d (%s)", (int)err, key.c_str()); return false; }
    s_w_scale[layer_idx][slot] = clCreateBuffer(cl_ctx.context(), CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
                                                sc.size() * sizeof(nnopt_storage_t), sc.data(), &err);
    if (err != CL_SUCCESS) { NNOPT_ERROR_FMT("int8: scale alloc %d (%s)", (int)err, key.c_str()); return false; }
    return true;
}

static cl_kernel mega_layer_kernel(int layer_idx) {
    if (layer_idx < 0 || layer_idx >= kMegaMaxLayers) return s_k_mega;
    if (!s_k_mega_layer[layer_idx]) {
        cl_int err = CL_SUCCESS;
        s_k_mega_layer[layer_idx] = clCreateKernel(s_mega_prog, "decoder_layer_mega", &err);
        if (err != CL_SUCCESS) { NNOPT_ERROR_FMT("mega: clone kernel L=%d %d", layer_idx, (int)err);
                                 s_k_mega_layer[layer_idx] = nullptr; }
    }
    return s_k_mega_layer[layer_idx];
}

// Invalidate the per-layer persistent weight bindings — call when cross-K/V is
// re-precomputed (new generation), since the cross-K/V cl_mems are reallocated.
extern "C" void mega_reset_layer_kernels() {
    for (int i = 0; i < kMegaMaxLayers; ++i) s_layer_weights_bound[i] = false;
}

// Release the precomputed cross K/V (called from model_reset_decode_state on a
// new generation, since g_enc_states changes per prompt).
// Allocated capacity of g_kcross/g_vcross (enc_len they were sized for).
// Lets mega_precompute_cross_kv REUSE the buffers/views/kernel clones when a
// new generation has the same enc_len (values are overwritten by the kernel) —
// releasing+reallocating per generation was part of the measured ~4.8 s
// first-step allocation stall.
static int g_kcross_alloc_len = 0;

// Value-only invalidation: marks cross-K/V stale WITHOUT releasing buffers,
// texture views, or per-layer kernel clones. mega_precompute_cross_kv then
// overwrites the values in-place when enc_len matches the allocated capacity.
extern "C" void mega_invalidate_cross_kv() {
    g_kcross_ready = false;
}

extern "C" void mega_reset_cross_kv() {
    for (int i = 0; i < 2 * 64; ++i) {
        if (g_kcross[i]) {
            // Drop any texture view over this buffer BEFORE releasing it.
            auto it = s_kv_img_views.find(g_kcross[i]);
            if (it != s_kv_img_views.end()) { clReleaseMemObject(it->second); s_kv_img_views.erase(it); }
            clReleaseMemObject(g_kcross[i]); g_kcross[i] = nullptr;
        }
        if (g_vcross[i]) { clReleaseMemObject(g_vcross[i]); g_vcross[i] = nullptr; }
    }
    g_kcross_ready = false;
    g_kcross_enc_len = 0;
    g_kcross_alloc_len = 0;
    mega_reset_layer_kernels();   // cross-K/V cl_mems reallocated → rebind weights
}

// Precompute cross-attn K/V for every layer × CFG row from the encoder states.
// enc_row0 = g_enc_states (cond), enc_row1 = g_enc_zero (uncond). enc_len = T.
// Returns false on failure (caller should fall back to the M=2 path).
extern "C" bool mega_precompute_cross_kv(
    OpenCLContext& cl_ctx, Weights& weights, cl_command_queue queue,
    cl_mem enc_row0, cl_mem enc_row1, int enc_len) {
    if (g_kcross_ready && g_kcross_enc_len == enc_len) return true;
    if (!mega_build(cl_ctx)) return false;
    if (enc_len <= 0 || enc_len > 64) { NNOPT_ERROR_FMT("mega: enc_len %d out of range", enc_len); return false; }
    // Capacity-aware: same enc_len → REUSE buffers, texture views, and kernel
    // clones; the precompute kernel below overwrites every value. Only a
    // different enc_len forces the full release+realloc reset.
    if (g_kcross_alloc_len != enc_len) mega_reset_cross_kv();
    const auto ckv_t0 = std::chrono::steady_clock::now();   // TTFT breakdown

    const int hidden = MODEL_CONFIG::DECODER_HIDDEN_SIZE;
    const size_t kv_bytes = (size_t)enc_len * hidden * sizeof(nnopt_storage_t);
    cl_int err = CL_SUCCESS;

    for (int row = 0; row < 2; ++row) {
        cl_mem enc = (row == 0) ? enc_row0 : enc_row1;
        if (!enc) { NNOPT_ERROR_FMT("mega: enc states null for row %d", row); return false; }
        for (int L = 0; L < kMegaLayers; ++L) {
            char wp[96]; std::snprintf(wp, sizeof(wp), "decoder.model.decoder.layers.%d.encoder_attn", L);
            cl_mem kw = weights.get_buffer(std::string(wp) + ".k_proj.weight");
            cl_mem vw = weights.get_buffer(std::string(wp) + ".v_proj.weight");
            if (!kw || !vw) { NNOPT_ERROR_FMT("mega: missing cross k/v proj at layer %d", L); return false; }
            const int idx = row * kMegaLayers + L;
            if (!g_kcross[idx]) {
                g_kcross[idx] = clCreateBuffer(cl_ctx.context(), CL_MEM_READ_WRITE, kv_bytes, nullptr, &err);
                if (err != CL_SUCCESS || !g_kcross[idx]) { NNOPT_ERROR_FMT("mega: kcross alloc %d", (int)err); return false; }
            }
            if (!g_vcross[idx]) {
                g_vcross[idx] = clCreateBuffer(cl_ctx.context(), CL_MEM_READ_WRITE, kv_bytes, nullptr, &err);
                if (err != CL_SUCCESS || !g_vcross[idx]) { NNOPT_ERROR_FMT("mega: vcross alloc %d", (int)err); return false; }
            }

            int arg = 0;
            if (!set_arg_checked(s_k_precompute, arg++, sizeof(cl_mem), &enc, "enc")) return false;
            if (!set_arg_checked(s_k_precompute, arg++, sizeof(cl_mem), &kw, "k_w")) return false;
            if (!set_arg_checked(s_k_precompute, arg++, sizeof(cl_mem), &vw, "v_w")) return false;
            if (!set_arg_checked(s_k_precompute, arg++, sizeof(cl_mem), &g_kcross[idx], "k_cross")) return false;
            if (!set_arg_checked(s_k_precompute, arg++, sizeof(cl_mem), &g_vcross[idx], "v_cross")) return false;
            if (!set_arg_checked(s_k_precompute, arg++, sizeof(int), &enc_len, "enc_len")) return false;
            const size_t gws[1] = {(size_t)enc_len * kMegaWG};
            const size_t lws[1] = {(size_t)kMegaWG};
            err = clEnqueueNDRangeKernel(queue, s_k_precompute, 1, nullptr, gws, lws, 0, nullptr, nullptr);
            if (err != CL_SUCCESS) { NNOPT_ERROR_FMT("mega: precompute dispatch %d (L=%d row=%d)", (int)err, L, row); return false; }
        }
    }
    g_kcross_enc_len = enc_len;
    g_kcross_alloc_len = enc_len;
    g_kcross_ready = true;
    if (nnopt_ttft_trace_enabled()) fprintf(stderr, "TTFT_TRACE [%.0f] cross_kv_precompute %.0f ms\n", nnopt_uptime_ms(),
            std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - ckv_t0).count());
    return true;
}

// Run a decoder layer via the M=2-native megakernel.
//   x_in0/x_in1   : [hidden] residual-stream input, row0=cond / row1=uncond (fp16)
//   x_out0/x_out1 : [hidden] outputs (may alias x_in for in-place)
//   k/v_cache_r0/r1 : per-row per-layer self-attn KV cache (caller-owned)
//   num_wg : 2 → both CFG rows in one dispatch (Stage 1 default);
//            1 → single row (A/B per-row mode), with single_row∈{0,1} selecting
//                which CFG row's cross-K/V to bind to BOTH kernel slots.
// The kernel selects per-row state by get_group_id(0); gws={num_wg*256}.
extern "C" bool mega_decoder_layer_m2_n(
    OpenCLContext& cl_ctx, Weights& weights, cl_command_queue queue,
    cl_mem x_in0, cl_mem x_out0, cl_mem x_in1, cl_mem x_out1,
    int layer_idx, int start_pos,
    cl_mem* k_cache_r0, cl_mem* v_cache_r0,
    cl_mem* k_cache_r1, cl_mem* v_cache_r1,
    int num_wg, int single_row) {
    // TTFT breakdown: stamp the first pass through each layer (step 0 only).
    { static int s_t0_layers = 0;
      if (s_t0_layers < 25) {
          if (nnopt_ttft_trace_enabled()) fprintf(stderr, "TTFT_TRACE [%.0f] layer_enter %d\n", nnopt_uptime_ms(), s_t0_layers);
          ++s_t0_layers;
      } }

    if (!mega_build(cl_ctx)) return false;
    if (!g_kcross_ready) { NNOPT_ERROR("mega: cross K/V not precomputed"); return false; }
    if (mwg_ffn_enabled() && !mega_ffn_ensure_scratch(cl_ctx)) return false;
    if (mega_qkv_ext_enabled() && !mega_qkv_ensure_scratch(cl_ctx)) return false;
    if (mega_attn_split_enabled() && !mega_split_ensure_scratch(cl_ctx)) return false;

    const int hidden = MODEL_CONFIG::DECODER_HIDDEN_SIZE;
    const int head_dim = MODEL_CONFIG::HEAD_DIM;
    const float eps = MODEL_CONFIG::LAYER_NORM_EPS;
    const float scale = 1.0f / std::sqrt((float)head_dim);
    const int enc_len = g_kcross_enc_len;
    cl_int err = CL_SUCCESS;

    // Allocate each row's self-attn KV cache on first use (token-major
    // [kMaxK, hidden]). MUST equal the kernel's MEGA_MAXK. 2048 → ~40.9 s max
    // clip at the 50 Hz frame rate. KV total = kMaxK × hidden × 2B × 2(k,v) ×
    // 2(CFG banks) × 24 layers = kMaxK × 192 KB ⇒ 2048 → 384 MB (was 512 → 96 MB).
    // (main.cpp clamps max_new_tokens ≤ MAX_POSITION_EMBEDDINGS so this never trips.)
    constexpr int kMaxK = 2048;
    const size_t row_bytes = (size_t)hidden * sizeof(nnopt_storage_t);
    if (start_pos + 1 > kMaxK) { NNOPT_ERROR_FMT("mega: start_pos %d exceeds KV cap %d (clip too long)", start_pos, kMaxK); return false; }
    cl_mem* caches[4] = {k_cache_r0, v_cache_r0, k_cache_r1, v_cache_r1};
    for (int i = 0; i < 4; ++i) {
        if (!*caches[i]) {
            *caches[i] = clCreateBuffer(cl_ctx.context(), CL_MEM_READ_WRITE, (size_t)kMaxK * row_bytes, nullptr, &err);
            if (err != CL_SUCCESS || !*caches[i]) { NNOPT_ERROR_FMT("mega: kv cache alloc %d", (int)err); return false; }
        }
    }

    // ── Stage 2: persistent per-layer kernel (num_wg==2 only) ────────────────
    // Use a per-layer kernel clone so the weight/cross-KV/KV-cache args bind
    // ONCE per generation; per step only the I/O sub-buffers (0..3) + start_pos
    // (arg 26) change. The A/B per-row path (num_wg==1) keeps the full-rebind
    // s_k_mega route. NNOPT_MEGA_NOPERSIST=1 forces full rebind for A/B of the
    // persistent-args optimization on the same binary.
    static const bool no_persist = [](){ const char* e = std::getenv("NNOPT_MEGA_NOPERSIST"); return e && e[0]=='1'; }();
    const bool persist = (num_wg == 2) && !no_persist;

    // ── Stage 6: int8 path (separate kernel, full rebind each step) ──────────
    // Quantize the 8 GEMV weights once per layer; bind int8 data+scale + the
    // (fp16) LN/cache/cross args to the int8 kernel. Persistent-args fast-path is
    // skipped (Stage 2 proved it neutral) so the arg layout can differ freely.
    if (s_int8 && num_wg == 2) {
        char wpi[96]; std::snprintf(wpi, sizeof(wpi), "decoder.model.decoder.layers.%d", layer_idx);
        const std::string p(wpi);
        if (!s_layer_i8_ready[layer_idx]) {
            const char* keys[8] = {
                ".self_attn.q_proj.weight", ".self_attn.k_proj.weight",
                ".self_attn.v_proj.weight", ".self_attn.out_proj.weight",
                ".encoder_attn.q_proj.weight", ".encoder_attn.out_proj.weight",
                ".fc1.weight", ".fc2.weight" };
            for (int s = 0; s < 8; ++s)
                if (!mega_quantize_weight(cl_ctx, weights, layer_idx, s, p + keys[s])) return false;
            s_layer_i8_ready[layer_idx] = true;
        }
        auto Wf = [&](const char* suffix) -> cl_mem { return weights.get_buffer(p + suffix); };
        cl_mem self_ln_w = Wf(".self_attn_layer_norm.weight");
        cl_mem self_ln_b = Wf(".self_attn_layer_norm.bias");
        cl_mem enc_ln_w  = Wf(".encoder_attn_layer_norm.weight");
        cl_mem enc_ln_b  = Wf(".encoder_attn_layer_norm.bias");
        cl_mem fin_ln_w  = Wf(".final_layer_norm.weight");
        cl_mem fin_ln_b  = Wf(".final_layer_norm.bias");
        cl_mem k_cross0 = g_kcross[0 * kMegaLayers + layer_idx];
        cl_mem v_cross0 = g_vcross[0 * kMegaLayers + layer_idx];
        cl_mem k_cross1 = g_kcross[1 * kMegaLayers + layer_idx];
        cl_mem v_cross1 = g_vcross[1 * kMegaLayers + layer_idx];
        if (!self_ln_w || !self_ln_b || !enc_ln_w || !enc_ln_b || !fin_ln_w || !fin_ln_b
            || !k_cross0 || !v_cross0 || !k_cross1 || !v_cross1) {
            NNOPT_ERROR_FMT("int8: missing fp16 aux weight at layer %d", layer_idx); return false; }
        cl_kernel ik = s_k_mega_i8;
        cl_mem* W8 = s_w_i8[layer_idx];
        cl_mem* S8 = s_w_scale[layer_idx];
        int a = 0;
        auto sa  = [&](size_t sz, const void* v, const char* nm){ return set_arg_checked(ik, a++, sz, v, nm); };
        auto sm  = [&](cl_mem* m, const char* nm){ return set_arg_checked(ik, a++, sizeof(cl_mem), m, nm); };
        if (!sm(&x_in0,"x_in0")||!sm(&x_out0,"x_out0")||!sm(&x_in1,"x_in1")||!sm(&x_out1,"x_out1")) return false;
        if (!sm(&self_ln_w,"sln_w")||!sm(&self_ln_b,"sln_b")) return false;
        if (!sm(&W8[0],"qw")||!sm(&S8[0],"qs")||!sm(&W8[1],"kw")||!sm(&S8[1],"ks")
          ||!sm(&W8[2],"vw")||!sm(&S8[2],"vs")||!sm(&W8[3],"ow")||!sm(&S8[3],"os")) return false;
        if (!sm(k_cache_r0,"kc0")||!sm(v_cache_r0,"vc0")||!sm(k_cache_r1,"kc1")||!sm(v_cache_r1,"vc1")) return false;
        if (!sm(&enc_ln_w,"eln_w")||!sm(&enc_ln_b,"eln_b")) return false;
        if (!sm(&W8[4],"cqw")||!sm(&S8[4],"cqs")||!sm(&W8[5],"cow")||!sm(&S8[5],"cos")) return false;
        if (!sm(&k_cross0,"kx0")||!sm(&v_cross0,"vx0")||!sm(&k_cross1,"kx1")||!sm(&v_cross1,"vx1")) return false;
        if (!sm(&fin_ln_w,"fln_w")||!sm(&fin_ln_b,"fln_b")) return false;
        if (!sm(&W8[6],"f1w")||!sm(&S8[6],"f1s")||!sm(&W8[7],"f2w")||!sm(&S8[7],"f2s")) return false;
        const float eps_i = MODEL_CONFIG::LAYER_NORM_EPS;
        const float scale_i = 1.0f / std::sqrt((float)MODEL_CONFIG::HEAD_DIM);
        if (!sa(sizeof(int), &start_pos, "start_pos")||!sa(sizeof(int), &enc_len, "enc_len")
          ||!sa(sizeof(float), &eps_i, "eps")||!sa(sizeof(float), &scale_i, "scale")) return false;
        const size_t gws8[1] = {2 * (size_t)kMegaWG};
        const size_t lws8[1] = {(size_t)kMegaWG};
        cl_int e8 = clEnqueueNDRangeKernel(queue, ik, 1, nullptr, gws8, lws8, 0, nullptr, nullptr);
        if (e8 != CL_SUCCESS) { NNOPT_ERROR_FMT("int8: dispatch %d (L=%d)", (int)e8, layer_idx); return false; }
        return true;
    }

    // ── Stage 6b: FFN-only int8 (fc1/fc2 int8, attention stays fp16) ─────────
    // Recovers ~2/3 of the int8 bandwidth win (FFN = 8.4M of 12.4M params/layer)
    // while keeping the precision-sensitive attention projections in fp16 — the
    // attention logits feed near-tied lm_head outputs that full-int8 perturbed
    // (tf-depth cb0 46→20). GELU after fc1 also absorbs fc1 quant noise.
    if (s_int8_ffn && num_wg == 2) {
        char wpf[96]; std::snprintf(wpf, sizeof(wpf), "decoder.model.decoder.layers.%d", layer_idx);
        const std::string p(wpf);
        if (!s_layer_i8_ready[layer_idx]) {
            // slots 6=fc1, 7=fc2 (reuse the same cache arrays/indices).
            if (!mega_quantize_weight(cl_ctx, weights, layer_idx, 6, p + ".fc1.weight")) return false;
            if (!mega_quantize_weight(cl_ctx, weights, layer_idx, 7, p + ".fc2.weight")) return false;
            s_layer_i8_ready[layer_idx] = true;
        }
        auto Wf = [&](const char* suffix) -> cl_mem { return weights.get_buffer(p + suffix); };
        cl_mem self_ln_w = Wf(".self_attn_layer_norm.weight"), self_ln_b = Wf(".self_attn_layer_norm.bias");
        cl_mem q_w = Wf(".self_attn.q_proj.weight"), k_w = Wf(".self_attn.k_proj.weight");
        cl_mem v_w = Wf(".self_attn.v_proj.weight"), o_w = Wf(".self_attn.out_proj.weight");
        cl_mem enc_ln_w = Wf(".encoder_attn_layer_norm.weight"), enc_ln_b = Wf(".encoder_attn_layer_norm.bias");
        cl_mem cq_w = Wf(".encoder_attn.q_proj.weight"), co_w = Wf(".encoder_attn.out_proj.weight");
        cl_mem fin_ln_w = Wf(".final_layer_norm.weight"), fin_ln_b = Wf(".final_layer_norm.bias");
        cl_mem k_cross0 = g_kcross[0*kMegaLayers+layer_idx], v_cross0 = g_vcross[0*kMegaLayers+layer_idx];
        cl_mem k_cross1 = g_kcross[1*kMegaLayers+layer_idx], v_cross1 = g_vcross[1*kMegaLayers+layer_idx];
        if (!self_ln_w||!self_ln_b||!q_w||!k_w||!v_w||!o_w||!enc_ln_w||!enc_ln_b||!cq_w||!co_w
            ||!fin_ln_w||!fin_ln_b||!k_cross0||!v_cross0||!k_cross1||!v_cross1) {
            NNOPT_ERROR_FMT("int8-ffn: missing weight at layer %d", layer_idx); return false; }
        cl_kernel ik = s_k_mega_i8f;
        cl_mem* W8 = s_w_i8[layer_idx]; cl_mem* S8 = s_w_scale[layer_idx];
        int a = 0;
        auto sm = [&](cl_mem* m, const char* nm){ return set_arg_checked(ik, a++, sizeof(cl_mem), m, nm); };
        auto sv = [&](size_t sz, const void* v, const char* nm){ return set_arg_checked(ik, a++, sz, v, nm); };
        if (!sm(&x_in0,"x_in0")||!sm(&x_out0,"x_out0")||!sm(&x_in1,"x_in1")||!sm(&x_out1,"x_out1")) return false;
        if (!sm(&self_ln_w,"sln_w")||!sm(&self_ln_b,"sln_b")) return false;
        if (!sm(&q_w,"qw")||!sm(&k_w,"kw")||!sm(&v_w,"vw")||!sm(&o_w,"ow")) return false;   // fp16 attn
        if (!sm(k_cache_r0,"kc0")||!sm(v_cache_r0,"vc0")||!sm(k_cache_r1,"kc1")||!sm(v_cache_r1,"vc1")) return false;
        if (!sm(&enc_ln_w,"eln_w")||!sm(&enc_ln_b,"eln_b")) return false;
        if (!sm(&cq_w,"cqw")||!sm(&co_w,"cow")) return false;                                // fp16 cross
        if (!sm(&k_cross0,"kx0")||!sm(&v_cross0,"vx0")||!sm(&k_cross1,"kx1")||!sm(&v_cross1,"vx1")) return false;
        if (!sm(&fin_ln_w,"fln_w")||!sm(&fin_ln_b,"fln_b")) return false;
        if (!sm(&W8[6],"f1w")||!sm(&S8[6],"f1s")||!sm(&W8[7],"f2w")||!sm(&S8[7],"f2s")) return false;  // int8 FFN
        const float eps_i = MODEL_CONFIG::LAYER_NORM_EPS;
        const float scale_i = 1.0f / std::sqrt((float)MODEL_CONFIG::HEAD_DIM);
        if (!sv(sizeof(int),&start_pos,"start_pos")||!sv(sizeof(int),&enc_len,"enc_len")
          ||!sv(sizeof(float),&eps_i,"eps")||!sv(sizeof(float),&scale_i,"scale")) return false;
        const size_t gwsf[1] = {2*(size_t)kMegaWG}; const size_t lwsf[1] = {(size_t)kMegaWG};
        cl_int ef = clEnqueueNDRangeKernel(queue, ik, 1, nullptr, gwsf, lwsf, 0, nullptr, nullptr);
        if (ef != CL_SUCCESS) { NNOPT_ERROR_FMT("int8-ffn: dispatch %d (L=%d)", (int)ef, layer_idx); return false; }
        return true;
    }

    cl_kernel kk = persist ? mega_layer_kernel(layer_idx) : s_k_mega;
    if (!kk) return false;

    if (persist && s_layer_weights_bound[layer_idx]) {
        // Fast path: only the per-step-varying args. KV caches + cross-KV + all
        // weights + enc_len/eps/scale were bound at first use this generation.
        if (!set_arg_checked(kk, 0, sizeof(cl_mem), &x_in0, "x_in0")) return false;
        if (!set_arg_checked(kk, 1, sizeof(cl_mem), &x_out0, "x_out0")) return false;
        if (!set_arg_checked(kk, 2, sizeof(cl_mem), &x_in1, "x_in1")) return false;
        if (!set_arg_checked(kk, 3, sizeof(cl_mem), &x_out1, "x_out1")) return false;
        if (!set_arg_checked(kk, 26, sizeof(int), &start_pos, "start_pos")) return false;
        if (mega_qkv_ext_enabled() &&
            !mega_qkv_predispatch(queue, layer_idx, x_in0, x_in1,
                                  k_cache_r0, v_cache_r0, k_cache_r1, v_cache_r1,
                                  start_pos, num_wg)) return false;
        if (mega_attn_split_enabled()) {
            // Stage-2 split: the megakernel is not enqueued; the chain of
            // small kernels writes the same ffn_normed/ffn_resid handoff.
            if (!mega_split_dispatch(cl_ctx, queue, layer_idx, x_in0, x_in1,
                                     k_cache_r0, v_cache_r0, k_cache_r1, v_cache_r1,
                                     start_pos, num_wg, enc_len)) return false;
        } else {
            const size_t gws_f[1] = {(size_t)num_wg * (size_t)kMegaWG};
            const size_t lws_f[1] = {(size_t)kMegaWG};
            err = clEnqueueNDRangeKernel(queue, kk, 1, nullptr, gws_f, lws_f, 0, nullptr,
                                         KernelProfiler::event_for("decoder_layer_mega",
                                             mwg_ffn_enabled() ? kMegaAttnWeightBytesPerDispatch
                                                               : kMegaWeightBytesPerDispatch));
            if (err != CL_SUCCESS) { NNOPT_ERROR_FMT("mega: m2 fast dispatch %d (L=%d)", (int)err, layer_idx); return false; }
        }
        if (mwg_ffn_enabled() && !mega_ffn_dispatch(queue, layer_idx, x_out0, x_out1, num_wg)) return false;
        return true;
    }

    // Slow path (first use this generation, or A/B no-persist): resolve weights
    // and bind ALL args. Weight resolution (string-map lookup + integrity probe)
    // happens here only — once per layer per generation on the persistent path.
    char wp[96]; std::snprintf(wp, sizeof(wp), "decoder.model.decoder.layers.%d", layer_idx);
    auto W = [&](const char* suffix) -> cl_mem { return weights.get_buffer(std::string(wp) + suffix); };

    cl_mem self_ln_w = W(".self_attn_layer_norm.weight");
    cl_mem self_ln_b = W(".self_attn_layer_norm.bias");
    cl_mem q_w = W(".self_attn.q_proj.weight");
    cl_mem k_w = W(".self_attn.k_proj.weight");
    cl_mem v_w = W(".self_attn.v_proj.weight");
    cl_mem o_w = W(".self_attn.out_proj.weight");
    cl_mem enc_ln_w = W(".encoder_attn_layer_norm.weight");
    cl_mem enc_ln_b = W(".encoder_attn_layer_norm.bias");
    cl_mem cq_w = W(".encoder_attn.q_proj.weight");
    cl_mem co_w = W(".encoder_attn.out_proj.weight");
    cl_mem fin_ln_w = W(".final_layer_norm.weight");
    cl_mem fin_ln_b = W(".final_layer_norm.bias");
    cl_mem fc1_w = W(".fc1.weight");
    cl_mem fc2_w = W(".fc2.weight");

    // For num_wg==2: row0 slot → cond cross-KV, row1 slot → uncond cross-KV.
    // For num_wg==1: bind single_row's cross-KV into BOTH slots (only WG0 runs).
    const int cr0 = (num_wg == 2) ? 0 : single_row;
    const int cr1 = (num_wg == 2) ? 1 : single_row;
    cl_mem k_cross0 = g_kcross[cr0 * kMegaLayers + layer_idx];
    cl_mem v_cross0 = g_vcross[cr0 * kMegaLayers + layer_idx];
    cl_mem k_cross1 = g_kcross[cr1 * kMegaLayers + layer_idx];
    cl_mem v_cross1 = g_vcross[cr1 * kMegaLayers + layer_idx];

    if (!self_ln_w || !self_ln_b || !q_w || !k_w || !v_w || !o_w ||
        !enc_ln_w || !enc_ln_b || !cq_w || !co_w ||
        !fin_ln_w || !fin_ln_b || !fc1_w || !fc2_w ||
        !k_cross0 || !v_cross0 || !k_cross1 || !v_cross1) {
        NNOPT_ERROR_FMT("mega: missing weight/cross-kv at layer %d (m2)", layer_idx);
        return false;
    }

    if (mega_qkv_ext_enabled()) {
        // Stage-1 split prep: packed qkv image + LN-weight cache for the
        // external dispatches (the fast path reuses these every step).
        if (!mega_qkv_image(cl_ctx, queue, q_w, k_w, v_w, layer_idx)) {
            NNOPT_ERROR_FMT("mega: qkv image creation failed at layer %d", layer_idx);
            return false;
        }
        s_qkv_ln_w[layer_idx] = self_ln_w;
        s_qkv_ln_b[layer_idx] = self_ln_b;
    }
    if (mega_attn_split_enabled()) {
        // Stage-2 split prep: cache the cross/final LN weights for the
        // per-step chain (o/cq/co images come from the tex block below).
        s_split_enc_ln_w[layer_idx] = enc_ln_w;
        s_split_enc_ln_b[layer_idx] = enc_ln_b;
        s_split_fin_ln_w[layer_idx] = fin_ln_w;
        s_split_fin_ln_b[layer_idx] = fin_ln_b;
    }
    if (mega_int8o_enabled()) {
        const int H = MODEL_CONFIG::DECODER_HIDDEN_SIZE, F = MODEL_CONFIG::FFN_DIM;
        const int mode = mega_int8o_mode();
        if (mode & 1 || mode >= 4)
            if (!mega_i8o_prepare(cl_ctx, queue, weights, layer_idx, 0,
                                  std::string(wp) + ".fc1.weight", F, H, kI8oNoutFc1,
                                  /*as_buffer=*/mode >= 4)) return false;
        if (mode & 2)
            if (!mega_i8o_prepare(cl_ctx, queue, weights, layer_idx, 1,
                                  std::string(wp) + ".fc2.weight", H, F, kI8oNoutFc2,
                                  /*as_buffer=*/false)) return false;
    }

    // fc weight handles for the external-FFN kernels (image when tex, else buffer).
    cl_mem fc1_h = fc1_w, fc2_h = fc2_w;
    if (mega_tex_enabled()) {
        // #3 texture path: mirror the GEMV weights into image2d once per layer.
        // The 6 attention projections rebind into the mega kernel (built with
        // -D MEGA_TEX → image2d_t args); fc1/fc2 images go to the FFN kernels
        // ONLY — the mega kernel's fc args (24/25) stay buffers (unused when
        // MEGA_FFN_EXTERNAL) so the buffer handles keep binding cleanly.
        const int H = MODEL_CONFIG::DECODER_HIDDEN_SIZE, F = MODEL_CONFIG::FFN_DIM;
        q_w   = mega_weight_image(cl_ctx, queue, q_w,   H, H, layer_idx, 0);
        k_w   = mega_weight_image(cl_ctx, queue, k_w,   H, H, layer_idx, 1);
        v_w   = mega_weight_image(cl_ctx, queue, v_w,   H, H, layer_idx, 2);
        o_w   = mega_weight_image(cl_ctx, queue, o_w,   H, H, layer_idx, 3);
        cq_w  = mega_weight_image(cl_ctx, queue, cq_w,  H, H, layer_idx, 4);
        co_w  = mega_weight_image(cl_ctx, queue, co_w,  H, H, layer_idx, 5);
        fc1_h = mega_weight_image(cl_ctx, queue, fc1_w, F, H, layer_idx, 6);
        if (mega_tex_fc2_enabled())   // default OFF: fc2 buffers beat fc2 tex (2.46 vs 2.19 GB/s)
            fc2_h = mega_weight_image(cl_ctx, queue, fc2_w, H, F, layer_idx, 7);
        if (!q_w || !k_w || !v_w || !o_w || !cq_w || !co_w || !fc1_h || !fc2_h) {
            NNOPT_ERROR_FMT("mega: tex image creation failed at layer %d", layer_idx);
            return false;
        }
    }

    int a = 0;
    if (!set_arg_checked(kk, a++, sizeof(cl_mem), &x_in0, "x_in0")) return false;
    if (!set_arg_checked(kk, a++, sizeof(cl_mem), &x_out0, "x_out0")) return false;
    if (!set_arg_checked(kk, a++, sizeof(cl_mem), &x_in1, "x_in1")) return false;
    if (!set_arg_checked(kk, a++, sizeof(cl_mem), &x_out1, "x_out1")) return false;
    if (!set_arg_checked(kk, a++, sizeof(cl_mem), &self_ln_w, "self_ln_w")) return false;
    if (!set_arg_checked(kk, a++, sizeof(cl_mem), &self_ln_b, "self_ln_b")) return false;
    if (!set_arg_checked(kk, a++, sizeof(cl_mem), &q_w, "q_w")) return false;
    if (!set_arg_checked(kk, a++, sizeof(cl_mem), &k_w, "k_w")) return false;
    if (!set_arg_checked(kk, a++, sizeof(cl_mem), &v_w, "v_w")) return false;
    if (!set_arg_checked(kk, a++, sizeof(cl_mem), &o_w, "o_w")) return false;
    if (!set_arg_checked(kk, a++, sizeof(cl_mem), k_cache_r0, "k_cache0")) return false;
    if (!set_arg_checked(kk, a++, sizeof(cl_mem), v_cache_r0, "v_cache0")) return false;
    if (!set_arg_checked(kk, a++, sizeof(cl_mem), k_cache_r1, "k_cache1")) return false;
    if (!set_arg_checked(kk, a++, sizeof(cl_mem), v_cache_r1, "v_cache1")) return false;
    if (!set_arg_checked(kk, a++, sizeof(cl_mem), &enc_ln_w, "enc_ln_w")) return false;
    if (!set_arg_checked(kk, a++, sizeof(cl_mem), &enc_ln_b, "enc_ln_b")) return false;
    if (!set_arg_checked(kk, a++, sizeof(cl_mem), &cq_w, "cq_w")) return false;
    if (!set_arg_checked(kk, a++, sizeof(cl_mem), &co_w, "co_w")) return false;
    if (!set_arg_checked(kk, a++, sizeof(cl_mem), &k_cross0, "k_cross0")) return false;
    if (!set_arg_checked(kk, a++, sizeof(cl_mem), &v_cross0, "v_cross0")) return false;
    if (!set_arg_checked(kk, a++, sizeof(cl_mem), &k_cross1, "k_cross1")) return false;
    if (!set_arg_checked(kk, a++, sizeof(cl_mem), &v_cross1, "v_cross1")) return false;
    if (!set_arg_checked(kk, a++, sizeof(cl_mem), &fin_ln_w, "fin_ln_w")) return false;
    if (!set_arg_checked(kk, a++, sizeof(cl_mem), &fin_ln_b, "fin_ln_b")) return false;
    if (!set_arg_checked(kk, a++, sizeof(cl_mem), &fc1_w, "fc1_w")) return false;
    if (!set_arg_checked(kk, a++, sizeof(cl_mem), &fc2_w, "fc2_w")) return false;
    if (!set_arg_checked(kk, a++, sizeof(int), &start_pos, "start_pos")) return false;
    if (!set_arg_checked(kk, a++, sizeof(int), &enc_len, "enc_len")) return false;
    if (!set_arg_checked(kk, a++, sizeof(float), &eps, "eps")) return false;
    if (!set_arg_checked(kk, a++, sizeof(float), &scale, "scale")) return false;
    if (mwg_ffn_enabled()) {
        // External-FFN handoff buffers (args 30/31) + cache the fc weight
        // cl_mems so the persistent fast path can dispatch fc1/fc2 directly.
        if (!set_arg_checked(kk, a++, sizeof(cl_mem), &s_ffn_normed, "ffn_normed")) return false;
        if (!set_arg_checked(kk, a++, sizeof(cl_mem), &s_ffn_resid, "ffn_resid")) return false;
        s_fc_w_cache[layer_idx][0] = fc1_h;   // image when tex, else buffer
        s_fc_w_cache[layer_idx][1] = fc2_h;
    }
    if (mega_qkv_ext_enabled()) {
        // q arrives from the external mega_qkv dispatch (persistent scratch —
        // bound once; the fast path never rebinds it).
        if (!set_arg_checked(kk, a++, sizeof(cl_mem), &s_q_ext, "q_ext")) return false;
    }

    if (persist) s_layer_weights_bound[layer_idx] = true;

    if (mega_qkv_ext_enabled() &&
        !mega_qkv_predispatch(queue, layer_idx, x_in0, x_in1,
                              k_cache_r0, v_cache_r0, k_cache_r1, v_cache_r1,
                              start_pos, num_wg)) return false;

    if (mega_attn_split_enabled()) {
        // Stage-2 split: the chain of small kernels replaces the megakernel
        // and writes the same ffn_normed/ffn_resid handoff for fc1/fc2.
        if (!mega_split_dispatch(cl_ctx, queue, layer_idx, x_in0, x_in1,
                                 k_cache_r0, v_cache_r0, k_cache_r1, v_cache_r1,
                                 start_pos, num_wg, enc_len)) return false;
    } else {
        const size_t gws[1] = {(size_t)num_wg * (size_t)kMegaWG};   // num_wg WGs: WG0=row0, WG1=row1
        const size_t lws[1] = {(size_t)kMegaWG};
        err = clEnqueueNDRangeKernel(queue, kk, 1, nullptr, gws, lws, 0, nullptr,
                                     KernelProfiler::event_for("decoder_layer_mega",
                                         mwg_ffn_enabled() ? kMegaAttnWeightBytesPerDispatch
                                                           : kMegaWeightBytesPerDispatch));
        if (err != CL_SUCCESS) { NNOPT_ERROR_FMT("mega: m2 layer dispatch %d (L=%d)", (int)err, layer_idx); return false; }
    }
    if (mwg_ffn_enabled() && !mega_ffn_dispatch(queue, layer_idx, x_out0, x_out1, num_wg)) return false;
    return true;
}

// Stage-1 default entry: BOTH CFG rows in ONE dispatch (num_wg=2).
extern "C" bool mega_decoder_layer_m2(
    OpenCLContext& cl_ctx, Weights& weights, cl_command_queue queue,
    cl_mem x_in0, cl_mem x_out0, cl_mem x_in1, cl_mem x_out1,
    int layer_idx, int start_pos,
    cl_mem* k_cache_r0, cl_mem* v_cache_r0,
    cl_mem* k_cache_r1, cl_mem* v_cache_r1) {
    return mega_decoder_layer_m2_n(cl_ctx, weights, queue,
        x_in0, x_out0, x_in1, x_out1, layer_idx, start_pos,
        k_cache_r0, v_cache_r0, k_cache_r1, v_cache_r1, /*num_wg=*/2, /*single_row=*/0);
}

// ── Recordable-replay support (cl_qcom_recordable_queues) ────────────────────
// The backbone records one steady-state step per geometry and replays it.
// Per-step scalars travel via the step-params BUFFER (s_sp_buf — FillBuffer'd
// once per step on the in-order queue), NOT kernel args: this driver
// (E031.37.12.07) rejects clEnqueueRecordingQCOM arg overrides with -59
// (probe-proven; plain replay with num_args=0 works at 4-5×). So a captured
// recording is valid for every step of its geometry with ZERO per-replay
// updates. This checker gates capture to the shipped steady-state split
// config (the only dispatch stream the recording was designed against).
extern "C" int mega_record_supported() {
    return (mega_attn_split_enabled() && mega_qkv_ext_enabled() && mwg_ffn_enabled()
            && mega_fc2_splitk_enabled() && !s_int8 && !s_int8_ffn
            && !mega_int8o_enabled() && s_k_qkv && s_k_attn_core) ? 1 : 0;
}

// ── MULTI-TOKEN FEASIBILITY PROBE (NNOPT_MROWS_PROBE=1) ──────────────────────
// The gate for speculative/multi-token decode: does the texture-path GEMV
// amortize weight LOADS across rows (load-bound → batching wins) or scale
// linearly (MAD-bound → batching is futile, the ROW_FUSE failure mode)?
// Dispatches the real mega_proj [H->H] tex GEMV (ilv layout: CFG/candidate row
// on group-dim0) at M=1/2/4/8 rows over ONE shared weight image and reports
// GPU us/dispatch + us/row. Weight bytes/dispatch are constant in M; if
// us/dispatch is flat in M the loads fully amortize (ceiling ~M speedup);
// if us/dispatch ~doubles per row-doubling it is MAD/issue-bound (abort).
extern "C" void mega_mrows_probe(OpenCLContext& cl_ctx, cl_command_queue queue) {
    const int H = MODEL_CONFIG::HIDDEN_SIZE;   // 1024
    const int rpw = 32;                         // matches mega_split_proj
    const int Mmax = 8;
    cl_int err = CL_SUCCESS;

    // USE_FP16 is auto-added by build_program_from_file on the fp16 build.
    cl_program prog = cl_ctx.build_program_from_file(
        "kernels/decoder_layer_mega.cl",
        "-D MEGA_TEX=1 -D MEGA_ROW_ILV=1 -D MEGA_QKV_EXTERNAL=1");  // mega_proj guarded by QKV_EXTERNAL+TEX
    if (!prog) { fprintf(stderr, "MROWS_PROBE: program build failed\n"); return; }
    cl_kernel k = clCreateKernel(prog, "mega_proj", &err);
    if (err != CL_SUCCESS) { fprintf(stderr, "MROWS_PROBE: clCreateKernel mega_proj %d\n", (int)err); return; }

    cl_image_format fmt{}; fmt.image_channel_order = CL_RGBA; fmt.image_channel_data_type = CL_HALF_FLOAT;
    cl_image_desc desc{}; desc.image_type = CL_MEM_OBJECT_IMAGE2D;
    desc.image_width = (size_t)H / 4; desc.image_height = (size_t)H;
    cl_mem W = clCreateImage(cl_ctx.context(), CL_MEM_READ_ONLY, &fmt, &desc, nullptr, &err);
    if (err != CL_SUCCESS || !W) { fprintf(stderr, "MROWS_PROBE: clCreateImage %d\n", (int)err); return; }

    cl_mem x_g = clCreateBuffer(cl_ctx.context(), CL_MEM_READ_WRITE, (size_t)Mmax * H * sizeof(float), nullptr, &err);
    cl_mem y_g = clCreateBuffer(cl_ctx.context(), CL_MEM_READ_WRITE, (size_t)Mmax * H * sizeof(float), nullptr, &err);
    if (err != CL_SUCCESS || !x_g || !y_g) { fprintf(stderr, "MROWS_PROBE: buffers %d\n", (int)err); return; }
    { std::vector<float> z((size_t)Mmax * H, 0.0f);
      clEnqueueWriteBuffer(queue, x_g, CL_TRUE, 0, z.size() * sizeof(float), z.data(), 0, nullptr, nullptr); }

    const int mode = 0;  // no residual add
    const uint64_t wbytes = (uint64_t)H * H * 2;  // weight bytes read per dispatch (const in M)
    fprintf(stderr, "\n=== MROWS PROBE (mega_proj [%d->%d] fp16-tex, ilv, rpw=%d) ===\n", H, H, rpw);
    fprintf(stderr, "%4s %14s %12s %10s %10s\n", "M", "us/dispatch", "us/row", "amort", "GB/s(eff)");
    double us_row_m1 = 0.0;
    for (int M : {1, 2, 4, 8}) {
        err  = clSetKernelArg(k, 0, sizeof(cl_mem), &W);
        err |= clSetKernelArg(k, 1, sizeof(cl_mem), &x_g);
        err |= clSetKernelArg(k, 2, sizeof(cl_mem), &y_g);
        err |= clSetKernelArg(k, 3, sizeof(cl_mem), &x_g);  // xin0 (unused, mode 0)
        err |= clSetKernelArg(k, 4, sizeof(cl_mem), &x_g);  // xin1 (unused)
        err |= clSetKernelArg(k, 5, sizeof(cl_mem), &x_g);  // resid_g (unused)
        err |= clSetKernelArg(k, 6, sizeof(int), &mode);
        err |= clSetKernelArg(k, 7, sizeof(int), &rpw);
        if (err != CL_SUCCESS) { fprintf(stderr, "MROWS_PROBE: setarg M=%d %d\n", M, (int)err); break; }
        const size_t gws[2] = {(size_t)M * 64, (size_t)(H / rpw)};  // dim0=row*WG, dim1=output tiles
        const size_t lws[2] = {64, 1};
        for (int w = 0; w < 3; ++w) clEnqueueNDRangeKernel(queue, k, 2, nullptr, gws, lws, 0, nullptr, nullptr);
        clFinish(queue);
        const int ITER = 80;
        cl_ulong total = 0; int counted = 0;
        for (int it = 0; it < ITER; ++it) {
            cl_event ev;
            err = clEnqueueNDRangeKernel(queue, k, 2, nullptr, gws, lws, 0, nullptr, &ev);
            if (err != CL_SUCCESS) { fprintf(stderr, "MROWS_PROBE: dispatch M=%d %d\n", M, (int)err); break; }
            clWaitForEvents(1, &ev);
            cl_ulong s = 0, e = 0;
            clGetEventProfilingInfo(ev, CL_PROFILING_COMMAND_START, sizeof(s), &s, nullptr);
            clGetEventProfilingInfo(ev, CL_PROFILING_COMMAND_END,   sizeof(e), &e, nullptr);
            if (e >= s) { total += (e - s); ++counted; }
            clReleaseEvent(ev);
        }
        if (counted == 0) break;
        const double us_disp = (double)total / 1e3 / counted;
        const double us_row  = us_disp / M;
        if (M == 1) us_row_m1 = us_row;
        const double gbs = (double)wbytes / ((double)total / counted);  // bytes/ns = GB/s, per dispatch
        fprintf(stderr, "%4d %14.2f %12.2f %10.3f %10.2f\n", M, us_disp, us_row,
                us_row_m1 > 0 ? us_row_m1 / us_row : 1.0, gbs);
    }
    fprintf(stderr, "=== amort = (M=1 us/row)/(M us/row): ~M ideal (loads free), ~1 = MAD-bound ===\n\n");
    clReleaseMemObject(x_g); clReleaseMemObject(y_g); clReleaseMemObject(W);
    clReleaseKernel(k); clReleaseProgram(prog);
}

// ── INT4 vs fp16 GEMV head-to-head (NNOPT_INT4_PROBE=1) ──────────────────────
// Times mega_proj_int4 (nibble-packed int4, 32/texel — int4's densest fetch)
// against mega_proj (fp16-texture) at the real decode config (M=2, [H->H]).
// Decisive: if int4 isn't faster HERE it can't win in the pipeline.
extern "C" void mega_int4_probe(OpenCLContext& cl_ctx, cl_command_queue queue) {
    const int H = MODEL_CONFIG::HIDDEN_SIZE;   // 1024
    const int rpw = 32, M = 2;
    cl_int err = CL_SUCCESS;
    cl_program prog = cl_ctx.build_program_from_file(
        "kernels/decoder_layer_mega.cl",
        "-D MEGA_TEX=1 -D MEGA_ROW_ILV=1 -D MEGA_QKV_EXTERNAL=1 -D MEGA_INT4_PROBE=1");
    if (!prog) { fprintf(stderr, "INT4_PROBE: build failed\n"); return; }
    cl_kernel kf = clCreateKernel(prog, "mega_proj", &err);
    if (err != CL_SUCCESS) { fprintf(stderr, "INT4_PROBE: mega_proj %d\n", (int)err); return; }
    cl_kernel k4 = clCreateKernel(prog, "mega_proj_int4", &err);
    if (err != CL_SUCCESS) { fprintf(stderr, "INT4_PROBE: mega_proj_int4 %d\n", (int)err); return; }

    // fp16 weight image [N, K/4] RGBA half ; int4 weight image [N, K/32] RGBA uint32.
    cl_image_format ff{}; ff.image_channel_order = CL_RGBA; ff.image_channel_data_type = CL_HALF_FLOAT;
    cl_image_desc fd{}; fd.image_type = CL_MEM_OBJECT_IMAGE2D; fd.image_width = (size_t)H/4;  fd.image_height = (size_t)H;
    cl_mem Wf = clCreateImage(cl_ctx.context(), CL_MEM_READ_ONLY, &ff, &fd, nullptr, &err);
    cl_image_format i4{}; i4.image_channel_order = CL_RGBA; i4.image_channel_data_type = CL_UNSIGNED_INT32;
    cl_image_desc id{}; id.image_type = CL_MEM_OBJECT_IMAGE2D; id.image_width = (size_t)H/32; id.image_height = (size_t)H;
    cl_mem W4 = clCreateImage(cl_ctx.context(), CL_MEM_READ_ONLY, &i4, &id, nullptr, &err);
    if (err != CL_SUCCESS || !Wf || !W4) { fprintf(stderr, "INT4_PROBE: image %d\n", (int)err); return; }
    cl_mem scale = clCreateBuffer(cl_ctx.context(), CL_MEM_READ_WRITE, (size_t)H * (H/32) * sizeof(float), nullptr, &err);
    cl_mem x_g   = clCreateBuffer(cl_ctx.context(), CL_MEM_READ_WRITE, (size_t)M * H * sizeof(float), nullptr, &err);
    cl_mem y_g   = clCreateBuffer(cl_ctx.context(), CL_MEM_READ_WRITE, (size_t)M * H * sizeof(float), nullptr, &err);
    if (err != CL_SUCCESS) { fprintf(stderr, "INT4_PROBE: buffers %d\n", (int)err); return; }
    { std::vector<float> z((size_t)M * H, 0.0f);
      clEnqueueWriteBuffer(queue, x_g, CL_TRUE, 0, z.size()*sizeof(float), z.data(), 0, nullptr, nullptr); }

    const size_t gws[2] = {(size_t)M * 64, (size_t)(H / rpw)};
    const size_t lws[2] = {64, 1};
    auto timeit = [&](cl_kernel k, const char* lbl) {
        for (int w = 0; w < 3; ++w) clEnqueueNDRangeKernel(queue, k, 2, nullptr, gws, lws, 0, nullptr, nullptr);
        clFinish(queue);
        cl_ulong total = 0; int cnt = 0; const int IT = 100;
        for (int it = 0; it < IT; ++it) {
            cl_event ev;
            if (clEnqueueNDRangeKernel(queue, k, 2, nullptr, gws, lws, 0, nullptr, &ev) != CL_SUCCESS) break;
            clWaitForEvents(1, &ev);
            cl_ulong s=0,e=0;
            clGetEventProfilingInfo(ev, CL_PROFILING_COMMAND_START, sizeof(s), &s, nullptr);
            clGetEventProfilingInfo(ev, CL_PROFILING_COMMAND_END,   sizeof(e), &e, nullptr);
            if (e>=s){ total += (e-s); ++cnt; }
            clReleaseEvent(ev);
        }
        const double us = cnt ? (double)total/1e3/cnt : -1.0;
        fprintf(stderr, "INT4_PROBE %-14s %10.2f us/dispatch\n", lbl, us);
        return us;
    };
    int mode = 0;
    clSetKernelArg(kf,0,sizeof(cl_mem),&Wf); clSetKernelArg(kf,1,sizeof(cl_mem),&x_g);
    clSetKernelArg(kf,2,sizeof(cl_mem),&y_g); clSetKernelArg(kf,3,sizeof(cl_mem),&x_g);
    clSetKernelArg(kf,4,sizeof(cl_mem),&x_g); clSetKernelArg(kf,5,sizeof(cl_mem),&x_g);
    clSetKernelArg(kf,6,sizeof(int),&mode);  clSetKernelArg(kf,7,sizeof(int),&rpw);
    clSetKernelArg(k4,0,sizeof(cl_mem),&W4); clSetKernelArg(k4,1,sizeof(cl_mem),&scale);
    clSetKernelArg(k4,2,sizeof(cl_mem),&x_g); clSetKernelArg(k4,3,sizeof(cl_mem),&y_g);
    clSetKernelArg(k4,4,sizeof(int),&rpw);
    fprintf(stderr, "\n=== INT4 vs fp16 GEMV [%d->%d] M=%d (real decode config) ===\n", H, H, M);
    const double uf = timeit(kf, "fp16-texture");
    const double u4 = timeit(k4, "int4-nibble");
    if (uf > 0 && u4 > 0)
        fprintf(stderr, "INT4_PROBE speedup int4/fp16 = %.3fx (>1 = int4 faster)\n", uf / u4);
    fprintf(stderr, "===\n\n");
    clReleaseMemObject(scale); clReleaseMemObject(x_g); clReleaseMemObject(y_g);
    clReleaseMemObject(Wf); clReleaseMemObject(W4);
    clReleaseKernel(kf); clReleaseKernel(k4); clReleaseProgram(prog);
}
