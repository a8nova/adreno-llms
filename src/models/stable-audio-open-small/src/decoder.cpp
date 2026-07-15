// decoder.cpp — Oobleck autoencoder decoder for Stable Audio Open Small.
//
// Reference: stable_audio_tools/models/autoencoders.py OobleckDecoder.forward /
//              DecoderBlock.forward
//            stable_audio_tools/models/blocks.py ResidualUnit.forward /
//              SnakeBeta.forward / snake_beta / WNConv1d / WNConvTranspose1d
//            stable_audio_tools/models/pretransforms.py AutoencoderPretransform.decode
//              (scale=1.0 -> latent passed straight through)
//            stable_audio_tools/models/bottleneck.py VAEBottleneck.decode (identity)
//
// latent [64,T] -> waveform [2, T*2048]. Host orchestrates device conv kernels.

#include "decoder.h"
#include "profiler.h"
#include <chrono>
#include "opencl_context.h"
#include "weights.h"
#include "debug_utils.h"
#include "model_config.h"
#include "utils.h"
#include <cmath>
#include <cstdlib>
#include <string>
#include <vector>
#include <cstdint>
#include <future>
#include <fstream>       // OPT-4 folded-weight disk cache
#include <sys/stat.h>    // OPT-4 mkdir

// ── ctor / dtor ──
Decoder::Decoder(OpenCLContext& cl_ctx, Weights& weights)
    : cl_ctx_(cl_ctx), weights_(weights) {}

Decoder::~Decoder() {
    if (col_scratch_) nnopt_pool_release(col_scratch_);
    if (prog_)  clReleaseProgram(prog_);
    if (conv_)  clReleaseProgram(conv_);
    if (convt_) clReleaseProgram(convt_);
}

bool Decoder::initialize() {
    prog_ = cl_ctx_.build_program_from_file("kernels/decoder.cl");
    if (!prog_) { NNOPT_ERROR("build kernels/decoder.cl failed"); return false; }
    conv_ = cl_ctx_.build_program_from_file("kernels/conv_1d.cl");
    if (!conv_) { NNOPT_ERROR("build kernels/conv_1d.cl failed"); return false; }
    convt_ = cl_ctx_.build_program_from_file("kernels/conv_transpose_1d.cl");
    if (!convt_) { NNOPT_ERROR("build kernels/conv_transpose_1d.cl failed"); return false; }
    ready_ = true;
    return true;
}

// ── storage codec ──
static inline nnopt_storage_t enc(float v) {
#ifdef NNOPT_USE_FP16
    return nnopt_storage_t(nnopt_f32_to_f16(v));
#else
    return v;
#endif
}
static inline float dec(nnopt_storage_t v) {
#ifdef NNOPT_USE_FP16
    return nnopt_f16_to_f32(static_cast<uint16_t>(v));
#else
    return v;
#endif
}

cl_mem Decoder::alloc(size_t nelem) {
    cl_int err;
    // B8: route decoder activations through the shared buffer pool (the DiT
    // already does) — the VAE was the one component still doing raw
    // clCreateBuffer/Release per conv (guide §5.7.1; BENCHMARK.md notes the
    // churn also fragmented the Adreno heap). NNOPT_DEC_POOL=0 reverts.
    static const bool pool_on = [] {
        const char* e = std::getenv("NNOPT_DEC_POOL");
        return !(e && e[0] == '0');
    }();
    cl_mem b = pool_on
        ? nnopt_pool_alloc(cl_ctx_.context(), nelem * sizeof(nnopt_storage_t), &err)
        : clCreateBuffer(cl_ctx_.context(), CL_MEM_READ_WRITE,
                         nelem * sizeof(nnopt_storage_t), nullptr, &err);
    if (err != CL_SUCCESS) { NNOPT_ERROR_FMT("alloc %zu err=%d", nelem, err); return nullptr; }
    return b;
}

cl_mem Decoder::upload_f32(const std::vector<float>& host) {
    std::vector<nnopt_storage_t> tmp(host.size());
    // Parallel fp32->storage encode: weight uploads move hundreds of MB and
    // the scalar loop was ~1/3 of the host weight-prep cost.
    const size_t n = host.size();
    if (n > (1u << 16)) {
        const int nth = 4;
        std::vector<std::future<void>> fs;
        for (int t = 0; t < nth; t++) {
            const size_t lo = n * t / nth, hi = n * (t + 1) / nth;
            fs.emplace_back(std::async(std::launch::async, [&, lo, hi](){
                for (size_t i = lo; i < hi; i++) tmp[i] = enc(host[i]);
            }));
        }
        for (auto& f : fs) f.get();
    } else {
        for (size_t i = 0; i < n; i++) tmp[i] = enc(host[i]);
    }
    cl_int err;
    cl_mem b = clCreateBuffer(cl_ctx_.context(), CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR,
                              tmp.size() * sizeof(nnopt_storage_t), tmp.data(), &err);
    if (err != CL_SUCCESS) { NNOPT_ERROR_FMT("upload err=%d", err); return nullptr; }
    return b;
}

double g_dec_host_weight_sec = 0.0;
double g_dec_host_download_sec = 0.0;

void Decoder::download(cl_mem buf, std::vector<float>& host, size_t nelem) {
    const auto _t0 = std::chrono::steady_clock::now();
    struct _Acc { const std::chrono::steady_clock::time_point t0;
                  ~_Acc(){ g_dec_host_download_sec += std::chrono::duration<double>(std::chrono::steady_clock::now()-t0).count(); } } _acc{_t0};
    std::vector<nnopt_storage_t> tmp(nelem);
    clEnqueueReadBuffer(cl_ctx_.queue(), buf, CL_TRUE, 0,
                        nelem * sizeof(nnopt_storage_t), tmp.data(), 0, nullptr, nullptr);
    host.resize(nelem);
    for (size_t i = 0; i < nelem; i++) host[i] = dec(tmp[i]);
}

// Reconstruct weight_norm conv weight: W[o] = g[o] * v[o] / ||v[o]||_2 (dim=0).
// weight_g stored [Cout,1,1] -> Cout scalars. weight_v stored [Cout,Cin,K].
// (Same layout used for both Conv1d and ConvTranspose1d checkpoints; the
//  transpose-conv kernel indexes weight as [Cin,Cout,K] which is exactly how
//  PyTorch stores ConvTranspose1d weight — Cout here is the kernel's C_out arg.)
// ── OPT-4: folded-weight disk cache ─────────────────────────────────────
// load_wn_weight re-derived every conv weight from the raw fp32 checkpoint on
// EVERY decode: weight-norm fold over up to 156M params + repack + fp16
// re-encode ≈ 4+ s host wall per run (g_dec_host_weight_sec). The folded fp16
// bytes are deterministic per (weights file, layout variant), so persist them
// to weights/dec_fold_cache/<prefix>.<variant>.f16 and mmap-free reload on
// later runs. File = [magic u32][elem-count u64][fp16 payload]; a size or
// magic mismatch falls back to a fresh fold + rewrite.
// Kill-switch: NNOPT_DEC_FOLD_CACHE=0.
static bool dec_fold_cache_on() {
    static const bool on = [] {
        const char* e = std::getenv("NNOPT_DEC_FOLD_CACHE");
        return !(e && e[0] == '0');
    }();
    return on;
}
static std::string dec_fold_cache_path(const std::string& prefix, const char* variant) {
    std::string name = prefix;
    for (auto& c : name) if (c == '.' || c == '/') c = '_';
    return std::string("weights/dec_fold_cache/") + name + variant + ".f16";
}
// B7: 128-bit vectorized elementwise kernels (guide §6.3). NNOPT_VEC_KERNELS=0
// reverts to the scalar variants; non-8-aligned sizes fall back per call.
static bool vec_kernels_enabled() {
    static const bool on = [] {
        const char* e = std::getenv("NNOPT_VEC_KERNELS");
        return !(e && e[0] == '0');
    }();
    return on;
}
// OPT-B1: transpose-conv as GEMM + col2im gather. NNOPT_CONVT_GEMM=0 reverts
// to the register-tiled gather kernel (which measured ~8 GFLOPS — memory-bound
// on scalar half loads; the 5 upsample layers were 52% of ALL GPU time).
static bool convt_gemm_enabled() {
    static const bool on = [] {
        const char* e = std::getenv("NNOPT_CONVT_GEMM");
        return !(e && e[0] == '0');
    }();
    return on;
}
static constexpr uint32_t DEC_FOLD_MAGIC = 0x44464331u; // "DFC1"

cl_mem Decoder::load_wn_weight(const std::string& prefix, int Cout, int Cin, int K, int repack_mode) {
    const auto _t0 = std::chrono::steady_clock::now();
    struct _Acc { const std::chrono::steady_clock::time_point t0;
                  ~_Acc(){ g_dec_host_weight_sec += std::chrono::duration<double>(std::chrono::steady_clock::now()-t0).count(); } } _acc{_t0};

    // The repack variant is decidable from the caller's dims alone — needed
    // up front so the cache filename matches what the fold below would emit.
    const size_t dims_total = (Cout > 0 && Cin > 0 && K > 0)
                            ? (size_t)Cout * (size_t)Cin * (size_t)K : 0;
    const bool would_repack = repack_mode == 1 && Cout > 0 && (Cout % 4) == 0 && dims_total > 0;
    const bool would_tg     = repack_mode == 2 && dims_total > 0;
    const char* variant = would_tg ? ".tg" : (would_repack ? ".r4" : ".flat");

    // Fast path: cached folded fp16 bytes → straight to a device buffer.
    if (dec_fold_cache_on() && dims_total > 0) {
        const std::string cpath = dec_fold_cache_path(prefix, variant);
        std::ifstream in(cpath, std::ios::binary);
        if (in.is_open()) {
            uint32_t magic = 0; uint64_t count = 0;
            in.read(reinterpret_cast<char*>(&magic), sizeof(magic));
            in.read(reinterpret_cast<char*>(&count), sizeof(count));
            if (in.good() && magic == DEC_FOLD_MAGIC && count == dims_total) {
                std::vector<nnopt_storage_t> bytes(count);
                in.read(reinterpret_cast<char*>(bytes.data()),
                        (std::streamsize)(count * sizeof(nnopt_storage_t)));
                if (in.good()) {
                    cl_int err;
                    cl_mem b = clCreateBuffer(cl_ctx_.context(),
                                              CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR,
                                              bytes.size() * sizeof(nnopt_storage_t),
                                              bytes.data(), &err);
                    if (err == CL_SUCCESS) return b;
                }
            }
            // fall through: stale/corrupt cache → refold + rewrite
        }
    }

    std::vector<float> g = weights_.get_host_vec(prefix + ".weight_g");
    std::vector<float> v = weights_.get_host_vec(prefix + ".weight_v");
    if (g.empty() || v.empty()) {
        NNOPT_ERROR_FMT("missing weight_norm %s", prefix.c_str());
        return nullptr;
    }
    // For ConvTranspose1d PyTorch weight_norm default dim=0 as well; weight_g
    // shape is [Cin,1,1] for transpose (norm grouped by the FIRST stored dim).
    // We reconstruct per first-dim group so this works for both by computing
    // group count from g.size().
    const int groups = (int)g.size();          // first-dim size (Cout for conv, Cin for convT)
    const size_t total = v.size();
    const size_t per_group = total / (size_t)groups;
    std::vector<float> W(total);
    // Parallel weight-norm fold (was single-threaded over up to 156M params).
    {
        const int nth = 4;
        std::vector<std::future<void>> fs;
        for (int t = 0; t < nth; t++) {
            const int glo = groups * t / nth, ghi = groups * (t + 1) / nth;
            fs.emplace_back(std::async(std::launch::async, [&, glo, ghi](){
                for (int gi = glo; gi < ghi; gi++) {
                    double ss = 0.0;
                    const size_t base = (size_t)gi * per_group;
                    for (size_t j = 0; j < per_group; j++) { float x = v[base + j]; ss += (double)x * x; }
                    float norm = (float)std::sqrt(ss) + 1e-12f;
                    float scale = g[gi] / norm;
                    for (size_t j = 0; j < per_group; j++) W[base + j] = v[base + j] * scale;
                }
            }));
        }
        for (auto& f : fs) f.get();
    }
    // OPT-4: encode + persist + upload in one pass (falls back to plain
    // upload_f32 when the cache is off or dims don't describe the tensor).
    auto upload_and_persist = [&](const std::vector<float>& host, const char* var) -> cl_mem {
        if (!(dec_fold_cache_on() && dims_total == host.size())) return upload_f32(host);
        const size_t n = host.size();
        std::vector<nnopt_storage_t> tmp(n);
        {
            const int nth = 4;
            std::vector<std::future<void>> fs;
            for (int t = 0; t < nth; t++) {
                const size_t lo = n * t / nth, hi = n * (t + 1) / nth;
                fs.emplace_back(std::async(std::launch::async, [&, lo, hi](){
                    for (size_t i = lo; i < hi; i++) tmp[i] = enc(host[i]);
                }));
            }
            for (auto& f : fs) f.get();
        }
        // Persist (best-effort; a partial write fails the size check next run).
        ::mkdir("weights/dec_fold_cache", 0755);
        std::ofstream outf(dec_fold_cache_path(prefix, var),
                           std::ios::binary | std::ios::trunc);
        if (outf.is_open()) {
            uint32_t magic = DEC_FOLD_MAGIC; uint64_t count = n;
            outf.write(reinterpret_cast<const char*>(&magic), sizeof(magic));
            outf.write(reinterpret_cast<const char*>(&count), sizeof(count));
            outf.write(reinterpret_cast<const char*>(tmp.data()),
                       (std::streamsize)(n * sizeof(nnopt_storage_t)));
        }
        cl_int err;
        cl_mem b = clCreateBuffer(cl_ctx_.context(), CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR,
                                  tmp.size() * sizeof(nnopt_storage_t), tmp.data(), &err);
        if (err != CL_SUCCESS) { NNOPT_ERROR_FMT("upload err=%d", err); return nullptr; }
        return b;
    };

    // v2 repack, oc-tile-blocked: [Cout,Cin,K] -> [Cout/4][Cin*K][4]. For a
    // fixed oc-tile the (ic,k) walk is perfectly SEQUENTIAL vec4 loads (the
    // first [ick][oc] repack strided by Cout and was slower than v1).
    if (would_repack && dims_total == total) {
        const int ickn = Cin * K;
        std::vector<float> R(total);
        for (int oc = 0; oc < Cout; oc++)
            for (int ick = 0; ick < ickn; ick++)
                R[(size_t)(oc / 4) * ickn * 4 + (size_t)ick * 4 + (oc % 4)]
                    = W[(size_t)oc * ickn + ick];
        return upload_and_persist(R, ".r4");
    }
    // B1 repack for the convT-as-GEMM path. Stored convT layout is
    // [C1, C2, K] with C1 = the layer's Cin (param name Cout here — callers
    // pass the FIRST stored dim first). GEMM wants A = W2[(c2*K + k), c1] so
    // cols[(c2*K+k), il] = Σ_c1 W2 · x[c1, il] falls out of one row-major HGEMM.
    if (would_tg && dims_total == total) {
        const int C1 = Cout, C2 = Cin;
        std::vector<float> R(total);
        const int nth = 4;
        std::vector<std::future<void>> fs;
        for (int t = 0; t < nth; t++) {
            const int lo = C1 * t / nth, hi = C1 * (t + 1) / nth;
            fs.emplace_back(std::async(std::launch::async, [&, lo, hi](){
                for (int c1 = lo; c1 < hi; c1++)
                    for (int c2 = 0; c2 < C2; c2++)
                        for (int k = 0; k < K; k++)
                            R[((size_t)c2 * K + k) * C1 + c1]
                                = W[((size_t)c1 * C2 + c2) * K + k];
            }));
        }
        for (auto& f : fs) f.get();
        return upload_and_persist(R, ".tg");
    }
    (void)Cout; (void)Cin; (void)K;
    return upload_and_persist(W, ".flat");
}

cl_mem Decoder::load_vec(const std::string& key, int n) {
    std::vector<float> h = weights_.get_host_vec(key);
    if ((int)h.size() < n) { NNOPT_ERROR_FMT("missing vec %s", key.c_str()); return nullptr; }
    h.resize(n);
    return upload_f32(h);
}

bool Decoder::gemm_path_ok(int Cin, int K, int Lout) {
    static const bool on = [](){
        const char* e = getenv("NNOPT_CONV_GEMM");
        return !(e && e[0] == '0');
    }();
    if (!on) return false;
    // Chunked im2col bounds the scratch, so any stride==1 conv qualifies.
    (void)Cin; (void)K; (void)Lout;
    return true;
}

// im2col + CLBlast GEMM conv: out[Cout, Lout] = W[Cout, Cin*K] @ col[Cin*K, Lout].
// Beats the hand-tiled kernel because CLBlast's tuned GEMM achieves far higher
// arithmetic intensity than any per-output conv formulation on this GPU.
cl_mem Decoder::conv1d_gemm(cl_mem in, cl_mem w, cl_mem bias, int Cin, int Cout,
                            int Lin, int K, int padding, int dilation, int has_bias, int Lout) {
    const int rows = Cin * K;
    // Length-chunked: bound the col scratch to CAP and GEMM each chunk into
    // its column-slice of the output. Long stages (>128 MB unchunked) used to
    // bail to the slower tiled kernel — now everything runs through GEMM.
    // 48 MB col cap: CLBlast pads A+B+C into ONE internal temp allocation and
    // Adreno 620 rejects single allocations past ~128 MB (CL_OUT_OF_RESOURCES,
    // campaign-documented). col at 48 MB keeps the padded temp comfortably under.
    const size_t CAP_ELEMS = ((size_t)48 * 1024 * 1024) / sizeof(nnopt_storage_t);
    int Lc_max = (int)(CAP_ELEMS / (size_t)rows);
    Lc_max -= Lc_max % 4;                       // im2col WI covers 4 outputs
    if (Lc_max < 4) return nullptr;             // absurd rows — let caller fall back
    if (Lc_max > Lout) Lc_max = Lout;
    const size_t col_elems = (size_t)rows * Lc_max;
    if (col_elems > col_scratch_elems_) {
        if (col_scratch_) { nnopt_pool_release(col_scratch_); col_scratch_ = nullptr; }
        col_scratch_ = alloc(col_elems);
        col_scratch_elems_ = col_scratch_ ? col_elems : 0;
    }
    cl_mem col = col_scratch_;
    if (!col) return nullptr;
    cl_mem out = alloc((size_t)Cout * Lout);
    if (!out) return nullptr;
    for (int l0 = 0; l0 < Lout; l0 += Lc_max) {
        const int Lc = (Lout - l0 < Lc_max) ? (Lout - l0) : Lc_max;
        cl_int err;
        // B7 SPLIT: im2col's 8-wide fast path does UNALIGNED vload_half8
        // (il0 shifts by k*dilation-padding) — measured slower than the
        // scalar 4-wide kernel on this driver. Default OFF; NNOPT_IM2COL_V8=1
        // re-enables the experiment.
        static const bool im2col_v8_on = [] {
            const char* e = std::getenv("NNOPT_IM2COL_V8");
            return e && e[0] == '1';
        }();
        const bool im2col_v8 = im2col_v8_on;
        cl_kernel k = nnopt_cached_kernel(conv_, im2col_v8 ? "im2col_1d_v8" : "im2col_1d", &err);
        if (err != CL_SUCCESS) { NNOPT_ERROR("im2col kernel"); nnopt_pool_release(out); return nullptr; }
        bool ok = set_arg_checked(k, 0, sizeof(cl_mem), &in, "in")
               && set_arg_checked(k, 1, sizeof(cl_mem), &col, "col")
               && set_arg_checked(k, 2, sizeof(int), &Cin, "Cin")
               && set_arg_checked(k, 3, sizeof(int), &Lin, "Lin")
               && set_arg_checked(k, 4, sizeof(int), &Lout, "Lout")
               && set_arg_checked(k, 5, sizeof(int), &K, "K")
               && set_arg_checked(k, 6, sizeof(int), &padding, "padding")
               && set_arg_checked(k, 7, sizeof(int), &dilation, "dilation")
               && set_arg_checked(k, 8, sizeof(int), &l0, "l0")
               && set_arg_checked(k, 9, sizeof(int), &Lc, "Lc");
        if (ok) {
            size_t gws[2] = { im2col_v8 ? (size_t)((Lc + 7) / 8) : (size_t)((Lc + 3) / 4), (size_t)rows };
            cl_int e2 = clEnqueueNDRangeKernel(cl_ctx_.queue(), k, 2, nullptr, gws, nullptr, 0, nullptr, KernelProfiler::event_for(im2col_v8 ? "im2col_1d_v8" : "im2col_1d"));
            ok = (e2 == CL_SUCCESS);
        }
        nnopt_kernel_done(k);
        if (!ok) { NNOPT_ERROR("im2col dispatch"); nnopt_pool_release(out); return nullptr; }
        if (!gemm_ab_ld(cl_ctx_.queue(), Cout, Lc, rows, w, col, out, (size_t)l0, Lout)) {
            nnopt_pool_release(out); return nullptr;
        }
    }
    if (has_bias) {
        cl_int err;
        const bool v8 = vec_kernels_enabled() && (Lout % 8) == 0;
        cl_kernel k = nnopt_cached_kernel(prog_, v8 ? "bias_add_rows_v8" : "bias_add_rows", &err);
        if (err == CL_SUCCESS) {
            set_arg_checked(k, 0, sizeof(cl_mem), &out, "x");
            set_arg_checked(k, 1, sizeof(cl_mem), &bias, "bias");
            set_arg_checked(k, 2, sizeof(int), &Cout, "C");
            set_arg_checked(k, 3, sizeof(int), &Lout, "L");
            size_t gws = v8 ? (size_t)Cout * (Lout / 8) : (size_t)Cout * Lout;
            clEnqueueNDRangeKernel(cl_ctx_.queue(), k, 1, nullptr, &gws, nullptr, 0, nullptr, KernelProfiler::event_for(v8 ? "bias_add_rows_v8" : "bias_add_rows"));
            nnopt_kernel_done(k);
        }
    }
    return out;
}

// Conv1d: in [Cin,Lin] -> out [Cout,Lout]. Lout = (Lin+2p-dil*(K-1)-1)/stride+1
// stride==1 uses the register-tiled conv_1d_t4x4 (4 oc x 4 ol per WI) unless
// NNOPT_CONV_T4X4=0 forces the legacy kernel (on-device A/B toggle).
cl_mem Decoder::conv1d(cl_mem in, cl_mem w, cl_mem bias, int Cin, int Cout,
                       int Lin, int K, int stride, int padding, int dilation, int has_bias,
                       bool repacked, bool try_gemm) {
    static const bool use_t4x4 = [](){
        const char* e = getenv("NNOPT_CONV_T4X4");
        return !(e && e[0] == '0');
    }();
    const bool tiled = use_t4x4 && stride == 1;
    const bool v2 = tiled && repacked && (Cout % 4) == 0;
    int Lout = (Lin + 2 * padding - dilation * (K - 1) - 1) / stride + 1;
    if (try_gemm && !repacked && stride == 1 && gemm_path_ok(Cin, K, Lout)) {
        cl_mem r = conv1d_gemm(in, w, bias, Cin, Cout, Lin, K, padding, dilation, has_bias, Lout);
        if (r) return r;
        NNOPT_ERROR_FMT("conv GEMM path failed (Cin=%d K=%d Lout=%d) — tiled fallback", Cin, K, Lout);
    }
    cl_mem out = alloc((size_t)Cout * Lout);
    if (!out) return nullptr;
    cl_mem bias_use = bias;
    cl_mem zero_bias = nullptr;
    if (!has_bias) { zero_bias = alloc((size_t)Cout); bias_use = zero_bias; }
    cl_int err;
    if (tiled) {
        cl_kernel k = nnopt_cached_kernel(conv_, v2 ? "conv_1d_t4x4v2" : "conv_1d_t4x4", &err);
        if (err != CL_SUCCESS) { NNOPT_ERROR("conv_1d_t4x4 kernel"); if (zero_bias) nnopt_pool_release(zero_bias); nnopt_pool_release(out); return nullptr; }
        bool ok = set_arg_checked(k, 0, sizeof(cl_mem), &in, "in")
               && set_arg_checked(k, 1, sizeof(cl_mem), &w, "weight")
               && set_arg_checked(k, 2, sizeof(cl_mem), &bias_use, "bias")
               && set_arg_checked(k, 3, sizeof(cl_mem), &out, "out")
               && set_arg_checked(k, 4, sizeof(int), &Cin, "Cin")
               && set_arg_checked(k, 5, sizeof(int), &Cout, "Cout")
               && set_arg_checked(k, 6, sizeof(int), &Lin, "Lin")
               && set_arg_checked(k, 7, sizeof(int), &Lout, "Lout")
               && set_arg_checked(k, 8, sizeof(int), &K, "K")
               && set_arg_checked(k, 9, sizeof(int), &padding, "padding")
               && set_arg_checked(k, 10, sizeof(int), &dilation, "dilation")
               && set_arg_checked(k, 11, sizeof(int), &has_bias, "has_bias");
        if (ok) {
            size_t gws[2] = { (size_t)((Lout + 3) / 4), (size_t)((Cout + 3) / 4) };
            err = clEnqueueNDRangeKernel(cl_ctx_.queue(), k, 2, nullptr, gws, nullptr, 0, nullptr, KernelProfiler::event_for(v2 ? "conv_1d_t4x4v2" : "conv_1d_t4x4"));
            ok = (err == CL_SUCCESS);
        }
        nnopt_kernel_done(k);
        if (zero_bias) nnopt_pool_release(zero_bias);
        if (!ok) { NNOPT_ERROR("conv1d_t4x4 dispatch"); nnopt_pool_release(out); return nullptr; }
        return out;
    }
    cl_kernel k = nnopt_cached_kernel(conv_, "conv_1d", &err);
    if (err != CL_SUCCESS) { NNOPT_ERROR("conv_1d kernel"); if (zero_bias) nnopt_pool_release(zero_bias); nnopt_pool_release(out); return nullptr; }
    bool ok = set_arg_checked(k, 0, sizeof(cl_mem), &in, "in")
           && set_arg_checked(k, 1, sizeof(cl_mem), &w, "weight")
           && set_arg_checked(k, 2, sizeof(cl_mem), &bias_use, "bias")
           && set_arg_checked(k, 3, sizeof(cl_mem), &out, "out")
           && set_arg_checked(k, 4, sizeof(int), &Cin, "Cin")
           && set_arg_checked(k, 5, sizeof(int), &Cout, "Cout")
           && set_arg_checked(k, 6, sizeof(int), &Lin, "Lin")
           && set_arg_checked(k, 7, sizeof(int), &Lout, "Lout")
           && set_arg_checked(k, 8, sizeof(int), &K, "K")
           && set_arg_checked(k, 9, sizeof(int), &stride, "stride")
           && set_arg_checked(k, 10, sizeof(int), &padding, "padding")
           && set_arg_checked(k, 11, sizeof(int), &dilation, "dilation")
           && set_arg_checked(k, 12, sizeof(int), &has_bias, "has_bias");
    if (ok) {
        size_t gws = (size_t)Cout * Lout;
        err = clEnqueueNDRangeKernel(cl_ctx_.queue(), k, 1, nullptr, &gws, nullptr, 0, nullptr, KernelProfiler::event_for("conv_1d_legacy"));
        ok = (err == CL_SUCCESS);
    }
    nnopt_kernel_done(k);
    if (zero_bias) nnopt_pool_release(zero_bias);
    if (!ok) { NNOPT_ERROR("conv1d dispatch"); nnopt_pool_release(out); return nullptr; }
    return out;
}

// ConvTranspose1d: L_out = (Lin-1)*stride - 2*pad + (K-1) + 1
cl_mem Decoder::conv_transpose1d(cl_mem in, cl_mem w, cl_mem bias, int Cin, int Cout,
                                 int Lin, int K, int stride, int padding, int* Lout_out, int has_bias) {
    int Lout = (Lin - 1) * stride - 2 * padding + (K - 1) + 1;
    if (Lout_out) *Lout_out = Lout;
    cl_mem out = alloc((size_t)Cout * Lout);
    if (!out) return nullptr;
    cl_mem bias_use = bias;
    cl_mem zero_bias = nullptr;
    if (!has_bias) { zero_bias = alloc((size_t)Cout); bias_use = zero_bias; }
    int dilation = 1;
    cl_int err;

    // ── B1: convT as chunked HGEMM + 2-read col2im gather ──
    // cols[(oc*K+k), il-ia] = Σ_ic W2[(oc*K+k), ic] · in[ic, il] (one CLBlast
    // HGEMM per output chunk, B = column window of `in`, no input copy), then
    // convt_col2im gathers each output's ≤2 valid taps + bias. Requires the
    // .tg-repacked weight (decoder_block passes repack_mode=2 under the same
    // switch, so `w` and this path always agree). NNOPT_CONVT_GEMM=0 reverts.
    if (convt_gemm_enabled()) {
        const int M = Cout * K;
        const size_t max_cols_bytes = 64u << 20;   // cols scratch cap
        int nq_max = (int)(max_cols_bytes / ((size_t)M * sizeof(nnopt_storage_t)));
        if (nq_max < 8) nq_max = 8;
        if (nq_max > Lin) nq_max = Lin;
        // Input-window span for ow outputs is ow/stride + K/stride (+1 from
        // flooring) columns — with K = 2*stride that's ow/stride + 3, so a
        // 3-column margin keeps every chunk within the nq_max scratch.
        const int ow_max = ((nq_max - 3) * stride > 0) ? (nq_max - 3) * stride : stride;
        cl_kernel gk = nnopt_cached_kernel(convt_, "convt_col2im", &err);
        if (err != CL_SUCCESS) { NNOPT_ERROR("convt_col2im kernel"); if (zero_bias) nnopt_pool_release(zero_bias); nnopt_pool_release(out); return nullptr; }
        cl_mem cols = alloc((size_t)M * nq_max);   // one scratch for all chunks
        if (!cols) { if (zero_bias) nnopt_pool_release(zero_bias); nnopt_pool_release(out); return nullptr; }
        bool okg = true;
        for (int oa = 0; oa < Lout && okg; oa += ow_max) {
            const int ow = (Lout - oa < ow_max) ? (Lout - oa) : ow_max;
            int ia = (oa + padding - (K - 1)) / stride; if (ia < 0) ia = 0;
            int ib = (oa + ow - 1 + padding) / stride + 1; if (ib > Lin) ib = Lin;
            const int nq = ib - ia;
            if (!gemm_ab_bld(cl_ctx_.queue(), M, nq, Cin, w, in, (size_t)ia, Lin, cols)) {
                okg = false; break;
            }
            okg = set_arg_checked(gk, 0, sizeof(cl_mem), &cols, "cols")
               && set_arg_checked(gk, 1, sizeof(cl_mem), &bias_use, "bias")
               && set_arg_checked(gk, 2, sizeof(cl_mem), &out, "out")
               && set_arg_checked(gk, 3, sizeof(int), &Cout, "Cout")
               && set_arg_checked(gk, 4, sizeof(int), &K, "K")
               && set_arg_checked(gk, 5, sizeof(int), &stride, "stride")
               && set_arg_checked(gk, 6, sizeof(int), &padding, "padding")
               && set_arg_checked(gk, 7, sizeof(int), &Lin, "Lin")
               && set_arg_checked(gk, 8, sizeof(int), &Lout, "Lout")
               && set_arg_checked(gk, 9, sizeof(int), &ia, "ia")
               && set_arg_checked(gk, 10, sizeof(int), &nq, "nq")
               && set_arg_checked(gk, 11, sizeof(int), &oa, "oa")
               && set_arg_checked(gk, 12, sizeof(int), &ow, "ow")
               && set_arg_checked(gk, 13, sizeof(int), &has_bias, "has_bias");
            if (okg) {
                size_t gws = ((size_t)Cout * ow + 63) / 64 * 64;
                err = clEnqueueNDRangeKernel(cl_ctx_.queue(), gk, 1, nullptr, &gws, nullptr, 0, nullptr, KernelProfiler::event_for("convt_col2im"));
                okg = (err == CL_SUCCESS);
            }
        }
        nnopt_pool_release(cols);
        nnopt_kernel_done(gk);
        if (zero_bias) nnopt_pool_release(zero_bias);
        if (!okg) { NNOPT_ERROR("convT GEMM path failed"); nnopt_pool_release(out); return nullptr; }
        return out;
    }
    // Register-tiled path (4 oc x 4 same-phase ol per WI); NNOPT_CONV_T4X4=0
    // falls back to the scalar gather kernel for on-device A/B.
    static const bool use_t4x4_t = [](){
        const char* e = getenv("NNOPT_CONV_T4X4");
        return !(e && e[0] == '0');
    }();
    if (use_t4x4_t) {
        cl_kernel kt = nnopt_cached_kernel(convt_, "conv_transpose_1d_t4x4", &err);
        if (err != CL_SUCCESS) { NNOPT_ERROR("convT_t4x4 kernel"); if (zero_bias) nnopt_pool_release(zero_bias); nnopt_pool_release(out); return nullptr; }
        bool okt = set_arg_checked(kt, 0, sizeof(cl_mem), &in, "in")
                && set_arg_checked(kt, 1, sizeof(cl_mem), &w, "weight")
                && set_arg_checked(kt, 2, sizeof(cl_mem), &bias_use, "bias")
                && set_arg_checked(kt, 3, sizeof(cl_mem), &out, "out")
                && set_arg_checked(kt, 4, sizeof(int), &Cin, "Cin")
                && set_arg_checked(kt, 5, sizeof(int), &Cout, "Cout")
                && set_arg_checked(kt, 6, sizeof(int), &Lin, "Lin")
                && set_arg_checked(kt, 7, sizeof(int), &Lout, "Lout")
                && set_arg_checked(kt, 8, sizeof(int), &K, "K")
                && set_arg_checked(kt, 9, sizeof(int), &stride, "stride")
                && set_arg_checked(kt, 10, sizeof(int), &padding, "padding")
                && set_arg_checked(kt, 11, sizeof(int), &has_bias, "has_bias");
        if (okt) {
            const size_t blocks_per_phase = (((size_t)Lout + stride - 1) / stride + 3) / 4;
            size_t gws[2] = { (size_t)stride * blocks_per_phase, (size_t)((Cout + 3) / 4) };
            err = clEnqueueNDRangeKernel(cl_ctx_.queue(), kt, 2, nullptr, gws, nullptr, 0, nullptr, KernelProfiler::event_for("convT_t4x4"));
            okt = (err == CL_SUCCESS);
        }
        nnopt_kernel_done(kt);
        if (zero_bias) nnopt_pool_release(zero_bias);
        if (!okt) { NNOPT_ERROR("convT_t4x4 dispatch"); nnopt_pool_release(out); return nullptr; }
        return out;
    }
    cl_kernel k = nnopt_cached_kernel(convt_, "conv_transpose_1d", &err);
    if (err != CL_SUCCESS) { NNOPT_ERROR("convT kernel"); if (zero_bias) nnopt_pool_release(zero_bias); nnopt_pool_release(out); return nullptr; }
    bool ok = set_arg_checked(k, 0, sizeof(cl_mem), &in, "in")
           && set_arg_checked(k, 1, sizeof(cl_mem), &w, "weight")
           && set_arg_checked(k, 2, sizeof(cl_mem), &bias_use, "bias")
           && set_arg_checked(k, 3, sizeof(cl_mem), &out, "out")
           && set_arg_checked(k, 4, sizeof(int), &Cin, "Cin")
           && set_arg_checked(k, 5, sizeof(int), &Cout, "Cout")
           && set_arg_checked(k, 6, sizeof(int), &Lin, "Lin")
           && set_arg_checked(k, 7, sizeof(int), &Lout, "Lout")
           && set_arg_checked(k, 8, sizeof(int), &K, "K")
           && set_arg_checked(k, 9, sizeof(int), &stride, "stride")
           && set_arg_checked(k, 10, sizeof(int), &padding, "padding")
           && set_arg_checked(k, 11, sizeof(int), &dilation, "dilation")
           && set_arg_checked(k, 12, sizeof(int), &has_bias, "has_bias");
    if (ok) {
        size_t gws = (size_t)Cout * Lout;
        err = clEnqueueNDRangeKernel(cl_ctx_.queue(), k, 1, nullptr, &gws, nullptr, 0, nullptr, KernelProfiler::event_for("convT_legacy"));
        ok = (err == CL_SUCCESS);
    }
    nnopt_kernel_done(k);
    if (zero_bias) nnopt_pool_release(zero_bias);
    if (!ok) { NNOPT_ERROR("convT dispatch"); nnopt_pool_release(out); return nullptr; }
    return out;
}

void Decoder::snake(cl_mem x, cl_mem alpha, cl_mem beta, int C, int L) {
    cl_int err;
    // B7: half8 variant (one 128-bit load+store per chunk) when L is 8-aligned.
    const bool v8 = vec_kernels_enabled() && (L % 8) == 0;
    cl_kernel k = nnopt_cached_kernel(prog_, v8 ? "snake_beta_v8" : "snake_beta", &err);
    if (err != CL_SUCCESS) { NNOPT_ERROR("snake kernel"); return; }
    set_arg_checked(k, 0, sizeof(cl_mem), &x, "x");
    set_arg_checked(k, 1, sizeof(cl_mem), &alpha, "alpha");
    set_arg_checked(k, 2, sizeof(cl_mem), &beta, "beta");
    set_arg_checked(k, 3, sizeof(int), &C, "C");
    set_arg_checked(k, 4, sizeof(int), &L, "L");
    // Matches SNAKE_CHUNK=8 in kernels/decoder.cl: one WI per 8-element chunk.
    const size_t chunks_per_ch = ((size_t)L + 7) / 8;
    size_t gws = (size_t)C * chunks_per_ch;
    clEnqueueNDRangeKernel(cl_ctx_.queue(), k, 1, nullptr, &gws, nullptr, 0, nullptr, KernelProfiler::event_for(v8 ? "snake_beta_v8" : "snake_beta"));
    nnopt_kernel_done(k);
}

cl_mem Decoder::add(cl_mem a, cl_mem b, int n) {
    cl_mem out = alloc((size_t)n);
    if (!out) return nullptr;
    cl_int err;
    const bool v8 = vec_kernels_enabled() && (n % 8) == 0;
    cl_kernel k = nnopt_cached_kernel(prog_, v8 ? "add_cl_v8" : "add_cl", &err);
    if (err != CL_SUCCESS) { NNOPT_ERROR("add kernel"); nnopt_pool_release(out); return nullptr; }
    set_arg_checked(k, 0, sizeof(cl_mem), &a, "a");
    set_arg_checked(k, 1, sizeof(cl_mem), &b, "b");
    set_arg_checked(k, 2, sizeof(cl_mem), &out, "out");
    set_arg_checked(k, 3, sizeof(int), &n, "n");
    size_t gws = v8 ? (size_t)(n / 8) : (size_t)n;
    err = clEnqueueNDRangeKernel(cl_ctx_.queue(), k, 1, nullptr, &gws, nullptr, 0, nullptr, KernelProfiler::event_for(v8 ? "residual_add_v8" : "residual_add"));
    nnopt_kernel_done(k);
    if (err != CL_SUCCESS) { NNOPT_ERROR("add dispatch"); nnopt_pool_release(out); return nullptr; }
    return out;
}

// ResidualUnit: res = x; x = layers(x); return x + res.
//   layers.0 SnakeBeta(C)  (in-place on a copy)
//   layers.1 WNConv1d(C->C, k=7, dilation, pad=(dilation*6)//2)
//   layers.2 SnakeBeta(C)
//   layers.3 WNConv1d(C->C, k=1)
cl_mem Decoder::residual_unit(cl_mem x, const std::string& prefix, int C, int L, int dilation) {
    // copy x -> h (snake is in-place, must not clobber the residual x)
    cl_mem h = alloc((size_t)C * L);
    if (!h) return nullptr;
    clEnqueueCopyBuffer(cl_ctx_.queue(), x, h, 0, 0,
                        (size_t)C * L * sizeof(nnopt_storage_t), 0, nullptr, nullptr);

    // layers.0 snake
    cl_mem a0 = load_vec(prefix + ".layers.0.alpha", C);
    cl_mem b0 = load_vec(prefix + ".layers.0.beta", C);
    if (!a0 || !b0) { nnopt_pool_release(h); if (a0) nnopt_pool_release(a0); if (b0) nnopt_pool_release(b0); return nullptr; }
    snake(h, a0, b0, C, L);
    nnopt_pool_release(a0); nnopt_pool_release(b0);

    // layers.1 WNConv1d k=7 dilation, pad=(dilation*6)//2
    int pad = (dilation * 6) / 2;
    const bool g1 = gemm_path_ok(C, 7, L);
    const bool rp = !g1 && (C % 4) == 0;
    cl_mem w1 = load_wn_weight(prefix + ".layers.1", C, C, 7, rp);
    cl_mem bias1 = load_vec(prefix + ".layers.1.bias", C);
    if (!w1 || !bias1) { nnopt_pool_release(h); if (w1) nnopt_pool_release(w1); if (bias1) nnopt_pool_release(bias1); return nullptr; }
    cl_mem c1 = conv1d(h, w1, bias1, C, C, L, 7, 1, pad, dilation, 1, rp, g1);
    nnopt_pool_release(h); nnopt_pool_release(w1); nnopt_pool_release(bias1);
    if (!c1) return nullptr;

    // layers.2 snake
    cl_mem a2 = load_vec(prefix + ".layers.2.alpha", C);
    cl_mem b2 = load_vec(prefix + ".layers.2.beta", C);
    if (!a2 || !b2) { nnopt_pool_release(c1); if (a2) nnopt_pool_release(a2); if (b2) nnopt_pool_release(b2); return nullptr; }
    snake(c1, a2, b2, C, L);
    nnopt_pool_release(a2); nnopt_pool_release(b2);

    // layers.3 WNConv1d k=1
    const bool g3 = gemm_path_ok(C, 1, L);
    const bool rp3 = !g3 && (C % 4) == 0;
    cl_mem w3 = load_wn_weight(prefix + ".layers.3", C, C, 1, rp3);
    cl_mem bias3 = load_vec(prefix + ".layers.3.bias", C);
    if (!w3 || !bias3) { nnopt_pool_release(c1); if (w3) nnopt_pool_release(w3); if (bias3) nnopt_pool_release(bias3); return nullptr; }
    cl_mem c3 = conv1d(c1, w3, bias3, C, C, L, 1, 1, 0, 1, 1, rp3, g3);
    nnopt_pool_release(c1); nnopt_pool_release(w3); nnopt_pool_release(bias3);
    if (!c3) return nullptr;

    // return x + layers(x)
    cl_mem out = add(x, c3, C * L);
    nnopt_pool_release(c3);
    return out;
}

// DecoderBlock: in [Cin,Lin] -> [Cout, Lin*stride]
//   layers.0 SnakeBeta(Cin)
//   layers.1 WNConvTranspose1d(Cin->Cout, k=2*stride, stride, pad=ceil(stride/2))
//   layers.2 ResidualUnit(Cout, dilation=1)
//   layers.3 ResidualUnit(Cout, dilation=3)
//   layers.4 ResidualUnit(Cout, dilation=9)
cl_mem Decoder::decoder_block(cl_mem x, const std::string& prefix, int Cin, int Cout,
                              int Lin, int stride, int* Lout_out) {
    // layers.0 snake (in-place copy)
    cl_mem h = alloc((size_t)Cin * Lin);
    if (!h) return nullptr;
    clEnqueueCopyBuffer(cl_ctx_.queue(), x, h, 0, 0,
                        (size_t)Cin * Lin * sizeof(nnopt_storage_t), 0, nullptr, nullptr);
    cl_mem a0 = load_vec(prefix + ".layers.0.alpha", Cin);
    cl_mem b0 = load_vec(prefix + ".layers.0.beta", Cin);
    if (!a0 || !b0) { nnopt_pool_release(h); if (a0) nnopt_pool_release(a0); if (b0) nnopt_pool_release(b0); return nullptr; }
    snake(h, a0, b0, Cin, Lin);
    nnopt_pool_release(a0); nnopt_pool_release(b0);

    // layers.1 WNConvTranspose1d, k=2*stride, pad=ceil(stride/2)
    int K = 2 * stride;
    int pad = (stride + 1) / 2;  // ceil(stride/2)
    // ConvTranspose1d weight stored [Cin, Cout, K]; weight_g [Cin,1,1].
    cl_mem w1 = load_wn_weight(prefix + ".layers.1", Cin, Cout, K,
                               convt_gemm_enabled() ? 2 : 0);
    cl_mem bias1 = load_vec(prefix + ".layers.1.bias", Cout);
    if (!w1 || !bias1) { nnopt_pool_release(h); if (w1) nnopt_pool_release(w1); if (bias1) nnopt_pool_release(bias1); return nullptr; }
    int Lup = 0;
    NNOPT_CHECKPOINT_FMT("decoder_block %s: convT Cin=%d Cout=%d Lin=%d K=%d stride=%d", prefix.c_str(), Cin, Cout, Lin, K, stride);
    cl_mem up = conv_transpose1d(h, w1, bias1, Cin, Cout, Lin, K, stride, pad, &Lup, 1);
    nnopt_pool_release(h); nnopt_pool_release(w1); nnopt_pool_release(bias1);
    if (!up) return nullptr;
#ifdef NNOPT_DEBUG
    clFinish(cl_ctx_.queue());  // SYNC-OK: localize watchdog/fault to this dispatch
#endif
    NNOPT_CHECKPOINT_FMT("decoder_block %s: convT done Lup=%d", prefix.c_str(), Lup);

    // 3 residual units (dilation 1,3,9) on [Cout, Lup]
    cl_mem r = up;
    const int dils[3] = {1, 3, 9};
    for (int i = 0; i < 3; i++) {
        char rp[160];
        snprintf(rp, sizeof(rp), "%s.layers.%d", prefix.c_str(), i + 2);
        cl_mem nr = residual_unit(r, rp, Cout, Lup, dils[i]);
        nnopt_pool_release(r);
        if (!nr) return nullptr;
        r = nr;
#ifdef NNOPT_DEBUG
        clFinish(cl_ctx_.queue());  // SYNC-OK: localize watchdog/fault to this residual unit
#endif
        NNOPT_CHECKPOINT_FMT("decoder_block %s: residual %d done (dil=%d)", prefix.c_str(), i, dils[i]);
    }
    if (Lout_out) *Lout_out = Lup;
    return r;
}

bool Decoder::decode(const std::vector<float>& latent, int T, std::vector<float>& out) {
    if (!ready_ && !initialize()) return false;
    cl_command_queue q = cl_ctx_.queue();

    // ── B12: decoder-scoped CLBlast Xgemm params (2026-07-08 A/B). ──
    // The on-device-tuned set helps the decoder's huge-N conv GEMMs
    // (23.6 → 22.1 s) but hurts the DiT's M=257 linears (32.0 → 35.9 s), so
    // the override is applied ONLY for the decode stage and restored to this
    // device's stock database entry on exit (needed for serve mode, where the
    // next generation's DiT runs in the same process). Safe without cache
    // clears: CLBlast keys compiled programs by the param values.
    // NNOPT_DEC_TUNED=0 reverts.
    static const bool dec_tuned_on = [] {
        const char* e = std::getenv("NNOPT_DEC_TUNED");
        return !(e && e[0] == '0');
    }();
    // Tuned at m=256 n=8192 k=1024 fp16 (clblast_tuner_xgemm, this device).
    static const size_t XGEMM_DEC_TUNED[16] =
        {0, 1, 16, 2, 8, 16, 128, 16, 8, 64, 0, 1, 1, 1, 4, 4};
    // Stock = CLBlast's QUALCOMM Adreno wildcard entry (xgemm_16.hpp).
    static const size_t XGEMM_ADRENO_STOCK[16] =
        {0, 1, 32, 2, 8, 8, 64, 8, 8, 64, 1, 1, 0, 0, 4, 4};
    struct ScopedXgemm {
        cl_device_id dev; bool active;
        ~ScopedXgemm() { if (active) nnopt_xgemm_override(dev, XGEMM_ADRENO_STOCK); }
    } scoped{cl_ctx_.device(),
             dec_tuned_on && nnopt_xgemm_override(cl_ctx_.device(), XGEMM_DEC_TUNED)};
    const std::string P = "pretransform.model.decoder";
    const int latent_dim = MODEL_CONFIG::DIT_LATENT_CHANNELS;  // 64

    // upload latent [64, T]
    cl_mem x = upload_f32(latent);
    if (!x) return false;
    int L = T;

    // layers.0 WNConv1d(64 -> 2048, k=7, pad=3, bias=True)
    {
        const bool g0 = gemm_path_ok(latent_dim, 7, L);
        cl_mem w = load_wn_weight(P + ".layers.0", 2048, latent_dim, 7, !g0);
        cl_mem bias = load_vec(P + ".layers.0.bias", 2048);
        if (!w || !bias) { nnopt_pool_release(x); if (w) nnopt_pool_release(w); if (bias) nnopt_pool_release(bias); return false; }
        cl_mem c = conv1d(x, w, bias, latent_dim, 2048, L, 7, 1, 3, 1, 1, !g0, g0);
        nnopt_pool_release(x); nnopt_pool_release(w); nnopt_pool_release(bias);
        if (!c) return false;
        x = c;
    }
    int C = 2048;
    NNOPT_LAYER_CHECK("decoder_layers_0", q, x, (size_t)C * L);

    // DecoderBlocks 1..5:  (Cin, Cout, stride)
    const int blk_cin[5]    = {2048, 1024, 512, 256, 128};
    const int blk_cout[5]   = {1024,  512, 256, 128, 128};
    const int blk_stride[5] = {   8,    8,   4,   4,   2};
    for (int i = 0; i < 5; i++) {
        char bp[96];
        snprintf(bp, sizeof(bp), "%s.layers.%d", P.c_str(), i + 1);
        int Lout = 0;
        cl_mem nb = decoder_block(x, bp, blk_cin[i], blk_cout[i], L, blk_stride[i], &Lout);
        nnopt_pool_release(x);
        if (!nb) return false;
        x = nb; C = blk_cout[i]; L = Lout;
        char dn[64]; snprintf(dn, sizeof(dn), "decoder_block_%d", i + 1);
        NNOPT_LAYER_CHECK(dn, q, x, (size_t)C * L);
    }

    // layers.6 SnakeBeta(128)
    {
        cl_mem a = load_vec(P + ".layers.6.alpha", C);
        cl_mem b = load_vec(P + ".layers.6.beta", C);
        if (!a || !b) { nnopt_pool_release(x); if (a) nnopt_pool_release(a); if (b) nnopt_pool_release(b); return false; }
        snake(x, a, b, C, L);
        nnopt_pool_release(a); nnopt_pool_release(b);
    }

    // layers.7 WNConv1d(128 -> 2, k=7, pad=3, bias=False)
    {
        cl_mem w = load_wn_weight(P + ".layers.7", MODEL_CONFIG::AUDIO_CHANNELS, C, 7);
        if (!w) { nnopt_pool_release(x); return false; }
        cl_mem c = conv1d(x, w, nullptr, C, MODEL_CONFIG::AUDIO_CHANNELS, L, 7, 1, 3, 1, 0);
        nnopt_pool_release(x); nnopt_pool_release(w);
        if (!c) return false;
        x = c;
    }
    C = MODEL_CONFIG::AUDIO_CHANNELS;  // 2
    NNOPT_LAYER_CHECK("decoder_output", q, x, (size_t)C * L);

    // e2e audio gate: dump the FULL decoded waveform [2, L] so Evaluate scores
    // the decoded audio (cosine + rms_ratio vs reference/layers/waveform_output.bin)
    // instead of falling back to the text gate. Device-side dump (dump mode only).
    NNOPT_LAYER_CHECK("waveform", q, x, (size_t)C * L);

    // download [2, L] channel-major
    download(x, out, (size_t)C * L);
    nnopt_pool_release(x);
    return true;
}
