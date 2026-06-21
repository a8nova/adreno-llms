#include "gpu_ops.h"
#include "utils.h"        // nnopt_f16_to_f32 / nnopt_f32_to_f16
#include "debug_utils.h"  // NNOPT_ERROR_FMT, tracked_clCreateBuffer

#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <fstream>
#include <sstream>
#include <cstring>
#include <algorithm>
#include <chrono>

// Activation storage encode/decode. Under fp16 builds the GPU stores raw
// IEEE-754 binary16; under fp32 it stores plain floats.
#ifdef NNOPT_USE_FP16
typedef uint16_t sstore_t;
static inline sstore_t enc(float f) { return nnopt_f32_to_f16(f); }
static inline float dec(sstore_t s) { return nnopt_f16_to_f32(s); }
#else
typedef float sstore_t;
static inline sstore_t enc(float f) { return f; }
static inline float dec(sstore_t s) { return s; }
#endif

int GpuOps::storage_bytes() const { return (int)sizeof(sstore_t); }

GpuOps::~GpuOps() {
    free_all();
    for (auto& [name, k] : kernels_) if (k) clReleaseKernel(k);
    for (auto& [w, img] : img_cache_) if (img) clReleaseMemObject(img);
    for (auto& [w, wr] : wt_cache_) if (wr) clReleaseMemObject(wr);
    if (prog_) clReleaseProgram(prog_);
    if (img_prog_) clReleaseProgram(img_prog_);
    if (dummy_) clReleaseMemObject(dummy_);
    if (out_idx_) clReleaseMemObject(out_idx_);
    if (out_val_) clReleaseMemObject(out_val_);
    if (lsm_partial_) clReleaseMemObject(lsm_partial_);
    if (lsm_scalar_) clReleaseMemObject(lsm_scalar_);
}

bool GpuOps::init(const std::string& kernel_dir) {
    // Concatenate the per-category kernel files (preamble FIRST so storage_t /
    // LOAD / STORE / act_apply are defined before any kernel uses them), then
    // build as one program. Missing optional files (decode/audio) are skipped.
    const char* files[] = {
        "_preamble.cl", "elementwise.cl", "gemm.cl", "attention.cl",
        "conv.cl", "fbank.cl", "decode.cl", "audio.cl",
    };
    std::string src;
    for (const char* f : files) {
        std::ifstream in(kernel_dir + "/" + f);
        if (!in.is_open()) continue;
        std::stringstream ss; ss << in.rdbuf();
        src += "\n// ==== " + std::string(f) + " ====\n" + ss.str();
    }
    if (src.empty()) { NNOPT_ERROR_FMT("GpuOps: no kernel files found under %s", kernel_dir.c_str()); return false; }
    // Tier 2: relax math precision (faster erf/exp/log/sqrt/division). All exps in
    // this model operate on <=0 args (post-max-subtraction) and no inf/nan is
    // produced, so -cl-finite-math-only (implied) is safe. Verified to preserve
    // units-exact + waveform cos 0.999999.
    prog_ = cl_.build_program(src, "-cl-fast-relaxed-math");
    if (!prog_) { NNOPT_ERROR_FMT("GpuOps: failed to build kernels from %s", kernel_dir.c_str()); return false; }
    const char* names[] = {
        "act_inplace", "axpy_inplace", "scale_inplace", "bias_add", "bias_add_ct", "linear_forward", "linear_fast", "linear_gemm", "linear_gemm_m", "gemv_m", "conv_gemm_ct", "linear_gemm8", "linear_gemm8_int8", "gemm_coop_m", "linear_gemm_tiled", "linear_gemm_tiled_int8",
        "gemv_coop", "gemv_coop4", "gemv_coop_int8", "gemv_coop4_int8",
        "layernorm_forward", "layernorm_coop", "glu_tc", "batchnorm_tc", "attention_context",
        "relpos_attention_context", "relpos_attention_coop", "conv1d_tc", "conv1d_dw", "im2col_tc", "conv1d_ct", "im2col_ct", "conv1d_ct4", "conv1d_ct44", "conv_transpose1d_ct", "conv_transpose1d_ct4", "col2im_transpose", "vocoder_input_gather",
        "embed_scale_pos", "fbank_window", "fbank_power", "fbank_mel", "relpos_pos_emb",
        "logits_to_f32", "lsm_partial_max", "lsm_reduce_max", "lsm_partial_sum", "lsm_reduce_sum",
        "lsm_write", "mask_region", "add_scalar_region",
        "argmax_f32", "argmax_coop", "set_at_f32", "set_at_dev", "audio_decode_s16", "audio_resample_linear", "audio_encode_s16",
    };
    for (const char* n : names) {
        cl_int err;
        cl_kernel k = clCreateKernel(prog_, n, &err);
        if (err != CL_SUCCESS || !k) { NNOPT_ERROR_FMT("GpuOps: clCreateKernel('%s') err=%d", n, (int)err); return false; }
        kernels_[n] = k;
        kname_[k] = n;
    }
    // Occupancy probe (NNOPT_KINFO): max WG size + preferred multiple + local/private
    // mem per kernel — reveals register-pressure limits (low max-WG = few resident waves).
    if (std::getenv("NNOPT_KINFO")) {
        for (const char* n : {"linear_gemm8", "linear_gemm", "linear_gemm_tiled", "conv1d_ct4",
                              "conv_transpose1d_ct4", "relpos_attention_context", "gemv_img4"}) {
            cl_kernel k = K(n); if (!k) continue;
            size_t wg = 0, mult = 0; cl_ulong lmem = 0, pmem = 0;
            clGetKernelWorkGroupInfo(k, cl_.device(), CL_KERNEL_WORK_GROUP_SIZE, sizeof(wg), &wg, nullptr);
            clGetKernelWorkGroupInfo(k, cl_.device(), CL_KERNEL_PREFERRED_WORK_GROUP_SIZE_MULTIPLE, sizeof(mult), &mult, nullptr);
            clGetKernelWorkGroupInfo(k, cl_.device(), CL_KERNEL_LOCAL_MEM_SIZE, sizeof(lmem), &lmem, nullptr);
            clGetKernelWorkGroupInfo(k, cl_.device(), CL_KERNEL_PRIVATE_MEM_SIZE, sizeof(pmem), &pmem, nullptr);
            fprintf(stderr, "KINFO %-26s maxWG=%zu prefMult=%zu localMem=%lu privMem=%lu\n",
                    n, wg, mult, (unsigned long)lmem, (unsigned long)pmem);
        }
    }
    // Separate cl_program for the image GEMV kernels (texture-cache weights). Built
    // standalone so its register allocation is isolated from the buffer kernels
    // (mixing them in one program spills the image kernels ~10× on Adreno).
#ifdef NNOPT_USE_FP16
    {
        std::ifstream pin(kernel_dir + "/gemv_image.cl");
        if (pin.is_open()) {
            std::stringstream ss; ss << pin.rdbuf();
            img_prog_ = cl_.build_program(ss.str(), "-cl-fast-relaxed-math");
            if (img_prog_) {
                for (const char* n : {"gemv_img", "gemv_img4", "gemm_tiled_img", "gemm_reg_img4"}) {
                    cl_int e; cl_kernel k = clCreateKernel(img_prog_, n, &e);
                    if (e == CL_SUCCESS && k) { kernels_[n] = k; kname_[k] = n; }
                }
            }
        }
    }
#endif

    cl_int err;
    dummy_ = clCreateBuffer(cl_.context(), CL_MEM_READ_ONLY, sizeof(sstore_t) * 4, nullptr, &err);
    out_idx_ = clCreateBuffer(cl_.context(), CL_MEM_READ_WRITE, sizeof(int), nullptr, &err);
    out_val_ = clCreateBuffer(cl_.context(), CL_MEM_READ_WRITE, sizeof(float), nullptr, &err);
    lsm_partial_ = clCreateBuffer(cl_.context(), CL_MEM_READ_WRITE, sizeof(float) * 256, nullptr, &err);
    lsm_scalar_ = clCreateBuffer(cl_.context(), CL_MEM_READ_WRITE, sizeof(float) * 2, nullptr, &err);
    ok_ = (err == CL_SUCCESS);
    return ok_;
}

cl_kernel GpuOps::K(const std::string& name) {
    auto it = kernels_.find(name);
    return it == kernels_.end() ? nullptr : it->second;
}

bool GpuOps::gpuprof_on() {
    if (gpuprof_init_ == 0) { const char* e = std::getenv("NNOPT_GPUPROF"); gpuprof_ = (e && *e && *e != '0'); gpuprof_init_ = 1; }
    return gpuprof_;
}

// Read+accumulate+release every pending event. Called in batches (bounds live
// events) and once more at dump. The events have already been enqueued; this
// waits for them to finish, reads HW START/END, and folds into busy/span stats.
void GpuOps::prof_drain() {
    if (span_ev_.empty()) return;
    std::vector<cl_event> evs; evs.reserve(span_ev_.size());
    for (auto& p : span_ev_) evs.push_back(p.second);
    clWaitForEvents((cl_uint)evs.size(), evs.data());
    for (auto& [nm, ev] : span_ev_) {
        cl_ulong s = 0, e2 = 0;
        clGetEventProfilingInfo(ev, CL_PROFILING_COMMAND_START, sizeof(s), &s, nullptr);
        clGetEventProfilingInfo(ev, CL_PROFILING_COMMAND_END, sizeof(e2), &e2, nullptr);
        busy_ns_ += (double)(e2 - s);
        prof_ns_[nm] += (double)(e2 - s);
        prof_calls_[nm] += 1;
        if (!gset_) { gmin_ = s; gmax_ = e2; gset_ = true; }
        else { if (s < gmin_) gmin_ = s; if (e2 > gmax_) gmax_ = e2; }
        clReleaseEvent(ev);
    }
    span_ev_.clear();
}

void GpuOps::prof_register(const std::string& name, cl_event ev) {
    if (!ev) return;
    span_ev_.emplace_back(name, ev);
    if (span_ev_.size() >= 2048) prof_drain();   // bound live events; rare → negligible perturbation
}

void GpuOps::prof_reset() {
    if (!gpuprof_on()) return;
    for (auto& [nm, ev] : span_ev_) clReleaseEvent(ev);
    span_ev_.clear();
    prof_ns_.clear(); prof_calls_.clear();
    busy_ns_ = 0.0; gmin_ = 0; gmax_ = 0; gset_ = false;
    stage_marks_.clear();
}

void GpuOps::prof_mark(const std::string& stage) {
    if (!gpuprof_on()) return;
    prof_drain();   // ensure busy_ns_/gmax_ include everything up to this boundary
    stage_marks_.emplace_back(stage, busy_ns_, gmax_);
}

cl_int GpuOps::pEnq(cl_kernel k, cl_uint dim, const size_t* global, const size_t* local) {
    if (!gpuprof_on()) return clEnqueueNDRangeKernel(cl_.queue(), k, dim, nullptr, global, local, 0, nullptr, nullptr);
    cl_event ev = nullptr;
    cl_int err = clEnqueueNDRangeKernel(cl_.queue(), k, dim, nullptr, global, local, 0, nullptr, &ev);
    // NOTE: do NOT wait per-kernel — that serializes the queue and destroys the
    // pipelining we're trying to measure. Collect the event; drain in batches.
    if (err == CL_SUCCESS && ev) prof_register(kname_[k], ev);
    return err;
}

void GpuOps::run1d(cl_kernel k, size_t global) {
    if (global == 0) return;
    cl_int err = pEnq(k, 1, &global, nullptr);
    if (err != CL_SUCCESS) NNOPT_ERROR_FMT("GpuOps: enqueue kernel failed err=%d (global=%zu)", (int)err, global);
}

bool GpuOps::convprof_on() {
    if (convprof_ < 0) { const char* e = std::getenv("NNOPT_CONVPROF"); convprof_ = (e && *e && *e != '0') ? 1 : 0; }
    return convprof_ == 1;
}

void GpuOps::dump_gpu_prof() {
    if (convprof_on() && cp_calls_ > 0) {
        fprintf(stderr, "\n=== CONVPROF (clFinish-attributed host time over %ld conv1d_ct calls) ===\n", cp_calls_);
        fprintf(stderr, "im2col=%.0f ms  CLBlast_gemm=%.0f ms  bias=%.0f ms | per-call: im2col=%.1f gemm=%.1f bias=%.1f\n",
                cp_im2col_, cp_gemm_, cp_bias_, cp_im2col_/cp_calls_, cp_gemm_/cp_calls_, cp_bias_/cp_calls_);
    }
    if (!gpuprof_on()) return;
    prof_drain();   // flush any events still pending since the last batch drain
    // Sort by total GPU ns descending.
    std::vector<std::pair<std::string, double>> v(prof_ns_.begin(), prof_ns_.end());
    std::sort(v.begin(), v.end(), [](auto& a, auto& b) { return a.second > b.second; });
    double total = 0; for (auto& [n, ns] : v) total += ns;
    fprintf(stderr, "\n=== GPU PROFILE (clGetEventProfilingInfo START->END, true GPU exec time; CLBlast included) ===\n");
    fprintf(stderr, "%-26s %10s %8s %9s %6s\n", "kernel", "gpu_ms", "calls", "us/call", "%busy");
    for (auto& [n, ns] : v) {
        long c = prof_calls_[n];
        fprintf(stderr, "%-26s %10.1f %8ld %9.1f %6.1f\n", n.c_str(), ns / 1e6, c,
                ns / 1e3 / (c ? c : 1), total > 0 ? 100.0 * ns / total : 0.0);
    }
    long ncalls = 0; for (auto& [n, c] : prof_calls_) ncalls += c;
    double span_ms = gset_ ? (double)(gmax_ - gmin_) / 1e6 : 0.0;   // GPU wall: first START -> last END
    double busy_ms = busy_ns_ / 1e6;                                 // Σ kernel durations (in-order ⇒ no overlap)
    double util = span_ms > 0 ? 100.0 * busy_ms / span_ms : 0.0;
    fprintf(stderr, "%-26s %10.1f %8ld\n", "TOTAL_GPU_BUSY", busy_ms, ncalls);
    fprintf(stderr, "\n--- UTILIZATION (GPU-side wall, excludes host pre/post) ---\n");
    fprintf(stderr, "GPU_BUSY (kernels running) : %.1f ms\n", busy_ms);
    fprintf(stderr, "GPU_SPAN (first..last)     : %.1f ms\n", span_ms);
    fprintf(stderr, "GPU_IDLE (gaps/overhead)   : %.1f ms\n", span_ms - busy_ms);
    fprintf(stderr, "UTILIZATION (busy/span)    : %.1f %%\n", util);
    fprintf(stderr, "  ^ at 99%% util the GPU-side wall would be %.1f ms (busy/0.99)\n", busy_ms / 0.99);

    if (!stage_marks_.empty()) {
        fprintf(stderr, "\n--- PER-STAGE (GPU busy vs span; low util = idle/launch-bound) ---\n");
        fprintf(stderr, "%-12s %10s %10s %10s %7s\n", "stage", "busy_ms", "span_ms", "idle_ms", "util%");
        double pbusy = 0; unsigned long long pspan = gset_ ? gmin_ : 0;
        for (auto& [st, b, g] : stage_marks_) {
            double bd = (b - pbusy) / 1e6;
            double sd = (double)(g - pspan) / 1e6;
            fprintf(stderr, "%-12s %10.1f %10.1f %10.1f %7.1f\n", st.c_str(), bd, sd, sd - bd,
                    sd > 0 ? 100.0 * bd / sd : 0.0);
            pbusy = b; pspan = g;
        }
    }
}

bool GpuOps::bufpool_on() {
    if (bufpool_ < 0) { const char* e = std::getenv("NNOPT_BUFPOOL"); bufpool_ = (e && *e == '0') ? 0 : 1; }
    return bufpool_ == 1;
}

// Pooled uninitialized READ_WRITE buffer: reuse an exact-byte-size buffer from the free
// list if available (no driver alloc), else create one. Recorded in buf_bytes_ so release_to
// can return it to the pool. Always pushed to arena_ (the live-set), like the old allocators.
cl_mem GpuOps::pool_alloc(size_t bytes, const char* tag) {
    if (bufpool_on()) {
        auto it = free_pool_.find(bytes);
        if (it != free_pool_.end()) {
            cl_mem b = it->second;
            free_pool_.erase(it);
            arena_.push_back(b);
            return b;
        }
    }
    cl_int err;
    cl_mem b = tracked_clCreateBuffer(cl_.context(), CL_MEM_READ_WRITE, bytes, nullptr, &err, tag);
    if (err != CL_SUCCESS || !b) return nullptr;
    if (bufpool_on()) buf_bytes_[b] = bytes;
    arena_.push_back(b);
    return b;
}

Tensor GpuOps::alloc(int n) {
    cl_mem b = pool_alloc((size_t)n * sizeof(sstore_t), "act");
    if (!b) { NNOPT_ERROR_FMT("GpuOps::alloc(%d) failed", n); return {}; }
    return {b, n};
}

cl_mem GpuOps::alloc_f32(int n) {
    cl_mem b = pool_alloc((size_t)n * sizeof(float), "act_f32");
    if (!b) { NNOPT_ERROR_FMT("GpuOps::alloc_f32(%d) failed", n); return nullptr; }
    return b;
}

Tensor GpuOps::clone(const Tensor& t) {
    Tensor out = alloc(t.n);
    if (!out.valid()) return out;
    clEnqueueCopyBuffer(cl_.queue(), t.buf, out.buf, 0, 0, (size_t)t.n * sizeof(sstore_t), 0, nullptr, nullptr);
    return out;
}

Tensor GpuOps::upload(const std::vector<float>& host) {
    Tensor t = alloc((int)host.size());
    if (!t.valid()) return t;
    std::vector<sstore_t> tmp(host.size());
    for (size_t i = 0; i < host.size(); ++i) tmp[i] = enc(host[i]);
    clEnqueueWriteBuffer(cl_.queue(), t.buf, CL_TRUE, 0, tmp.size() * sizeof(sstore_t), tmp.data(), 0, nullptr, nullptr);
    return t;
}

std::vector<float> GpuOps::download(const Tensor& t) { return download(t, t.n); }

std::vector<float> GpuOps::download(const Tensor& t, int n) {
    std::vector<sstore_t> tmp(n);
    clEnqueueReadBuffer(cl_.queue(), t.buf, CL_TRUE, 0, (size_t)n * sizeof(sstore_t), tmp.data(), 0, nullptr, nullptr);
    std::vector<float> out(n);
    for (int i = 0; i < n; ++i) out[i] = dec(tmp[i]);
    return out;
}

cl_mem GpuOps::upload_ints(const std::vector<int>& ids) {
    cl_int err;
    cl_mem b = tracked_clCreateBuffer(cl_.context(), CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
                                      ids.size() * sizeof(int), (void*)ids.data(), &err, "ids");
    if (err != CL_SUCCESS || !b) { NNOPT_ERROR_FMT("GpuOps::upload_ints failed err=%d", (int)err); return nullptr; }
    arena_.push_back(b);
    return b;
}

cl_mem GpuOps::weight(const std::string& key, bool optional) { return w_.get_buffer(key, optional); }

Tensor GpuOps::copy_region(const Tensor& src, int off_elems, int n) {
    Tensor out = alloc(n);
    clEnqueueCopyBuffer(cl_.queue(), src.buf, out.buf, (size_t)off_elems * sizeof(sstore_t), 0,
                        (size_t)n * sizeof(sstore_t), 0, nullptr, nullptr);
    return out;
}

void GpuOps::copy_into(cl_mem dst, int dst_off_elems, const Tensor& src, int n) {
    clEnqueueCopyBuffer(cl_.queue(), src.buf, dst, 0, (size_t)dst_off_elems * sizeof(sstore_t),
                        (size_t)n * sizeof(sstore_t), 0, nullptr, nullptr);
}
void GpuOps::copy_into_off(cl_mem dst, int dst_off_elems, cl_mem src, int src_off_elems, int n) {
    clEnqueueCopyBuffer(cl_.queue(), src, dst, (size_t)src_off_elems * sizeof(sstore_t),
                        (size_t)dst_off_elems * sizeof(sstore_t), (size_t)n * sizeof(sstore_t),
                        0, nullptr, nullptr);
}

void GpuOps::free_all() {
    for (cl_mem b : arena_) if (b) clReleaseMemObject(b);
    arena_.clear();
    for (auto& [bytes, b] : free_pool_) if (b) clReleaseMemObject(b);  // release recycled buffers too
    free_pool_.clear();
    buf_bytes_.clear();
}

int GpuOps::mark() const { return (int)arena_.size(); }

void GpuOps::release_to(int m) {
    if (m < 0) m = 0;
    for (int i = (int)arena_.size() - 1; i >= m; --i) {
        cl_mem b = arena_[i];
        if (!b) continue;
        auto it = buf_bytes_.find(b);
        if (bufpool_on() && it != buf_bytes_.end()) free_pool_.emplace(it->second, b);  // recycle, don't release
        else clReleaseMemObject(b);                                                      // uploads / pool off
    }
    if (m < (int)arena_.size()) arena_.resize(m);
}

// ----------------------------------------------------------------- ops

bool GpuOps::int8_enabled() {
    if (int8_mode_ < 0) {
        const char* e = std::getenv("NNOPT_INT8");
        int8_mode_ = (e && *e && *e != '0') ? 1 : 0;
    }
    return int8_mode_ == 1;
}

// Lazily quantize an fp16 weight [N,K] to per-row symmetric int8 + fp16 scales.
// Reads the fp16 weight back from the GPU once, quantizes on host, uploads the
// int8 buffer + scale buffer. Cached by the source cl_mem so it's one-time.
const GpuOps::Int8W* GpuOps::get_int8(cl_mem w, int N, int K) {
    auto it = int8_cache_.find(w);
    if (it != int8_cache_.end()) return &it->second;
    std::vector<sstore_t> hw((size_t)N * K);
    if (clEnqueueReadBuffer(cl_.queue(), w, CL_TRUE, 0, hw.size() * sizeof(sstore_t),
                            hw.data(), 0, nullptr, nullptr) != CL_SUCCESS) return nullptr;
    std::vector<int8_t> q((size_t)N * K);
    std::vector<sstore_t> scale(N);
    for (int o = 0; o < N; ++o) {
        const size_t b = (size_t)o * K;
        float amax = 0.0f;
        for (int k = 0; k < K; ++k) { float v = dec(hw[b + k]); amax = fmaxf(amax, fabsf(v)); }
        float s = amax > 0.0f ? amax / 127.0f : 1.0f;
        float inv = 1.0f / s;
        for (int k = 0; k < K; ++k) {
            int qi = (int)lrintf(dec(hw[b + k]) * inv);
            q[b + k] = (int8_t)(qi < -127 ? -127 : (qi > 127 ? 127 : qi));
        }
        scale[o] = enc(s);
    }
    cl_int err;
    Int8W e;
    e.N = N; e.K = K;
    e.q = clCreateBuffer(cl_.context(), CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
                         q.size(), q.data(), &err);
    if (err != CL_SUCCESS) return nullptr;
    e.scale = clCreateBuffer(cl_.context(), CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
                             scale.size() * sizeof(sstore_t), scale.data(), &err);
    if (err != CL_SUCCESS) { clReleaseMemObject(e.q); return nullptr; }
    auto res = int8_cache_.emplace(w, e);
    return &res.first->second;
}

bool GpuOps::img_enabled() {
    if (img_mode_ < 0) {
        if (sizeof(sstore_t) != 2 || !img_prog_) { img_mode_ = 0; }     // fp16-only + program built
        else { const char* e = std::getenv("NNOPT_IMG"); img_mode_ = (e && *e == '0') ? 0 : 1; }  // default ON
    }
    return img_mode_ == 1;
}

// Concatenate q/k/v weights [N,K]->[3N,K] and biases [N]->[3N], cached by qw, for a
// fused QKV GEMV (one launch + one input read instead of three). One-time read+upload.
const GpuOps::QKV* GpuOps::get_qkv(cl_mem qw, cl_mem kw, cl_mem vw, cl_mem qb, cl_mem kb, cl_mem vb, int N, int K) {
    auto it = qkv_cache_.find(qw);
    if (it != qkv_cache_.end()) return &it->second;
    auto rd = [&](cl_mem b, size_t n) { std::vector<sstore_t> v(n); clEnqueueReadBuffer(cl_.queue(), b, CL_TRUE, 0, n * sizeof(sstore_t), v.data(), 0, nullptr, nullptr); return v; };
    std::vector<sstore_t> wcat((size_t)3 * N * K);
    auto wq = rd(qw, (size_t)N * K), wk = rd(kw, (size_t)N * K), wv = rd(vw, (size_t)N * K);
    std::copy(wq.begin(), wq.end(), wcat.begin());
    std::copy(wk.begin(), wk.end(), wcat.begin() + (size_t)N * K);
    std::copy(wv.begin(), wv.end(), wcat.begin() + (size_t)2 * N * K);
    cl_int e; QKV q;
    q.w = clCreateBuffer(cl_.context(), CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, wcat.size() * sizeof(sstore_t), wcat.data(), &e);
    if (qb && kb && vb) {
        std::vector<sstore_t> bcat((size_t)3 * N);
        auto bq = rd(qb, N), bk = rd(kb, N), bvv = rd(vb, N);
        std::copy(bq.begin(), bq.end(), bcat.begin());
        std::copy(bk.begin(), bk.end(), bcat.begin() + N);
        std::copy(bvv.begin(), bvv.end(), bcat.begin() + 2 * N);
        q.b = clCreateBuffer(cl_.context(), CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, bcat.size() * sizeof(sstore_t), bcat.data(), &e);
    }
    auto res = qkv_cache_.emplace(qw, q);
    return &res.first->second;
}

// Lazily repack transposed-conv weight w[Cin,Cout,K] -> Wr[Cout*K, Cin] (tap-major)
// so the upsample is a GEMM tmp[Cout*K,T]=Wr@in + col2im. Read back, repack on host,
// upload. Cached by w cl_mem (one-time).
cl_mem GpuOps::get_wt_transpose(cl_mem w, int Cin, int Cout, int K) {
    auto it = wt_cache_.find(w);
    if (it != wt_cache_.end()) return it->second;
    std::vector<sstore_t> hw((size_t)Cin * Cout * K);
    clEnqueueReadBuffer(cl_.queue(), w, CL_TRUE, 0, hw.size() * sizeof(sstore_t), hw.data(), 0, nullptr, nullptr);
    std::vector<sstore_t> wr((size_t)Cout * K * Cin);
    for (int ci = 0; ci < Cin; ++ci)
        for (int co = 0; co < Cout; ++co)
            for (int k = 0; k < K; ++k)
                wr[((size_t)co * K + k) * Cin + ci] = hw[((size_t)ci * Cout + co) * K + k];
    cl_int err;
    cl_mem b = clCreateBuffer(cl_.context(), CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
                              wr.size() * sizeof(sstore_t), wr.data(), &err);
    if (err != CL_SUCCESS) b = nullptr;
    wt_cache_.emplace(w, b);
    return b;
}

// Lazily create an image2d_t VIEW of the fp16 weight buffer [N,K] (CL_RGBA/
// CL_HALF_FLOAT, width K/4, height N) — same backing memory, texture-cache reads.
// Returns nullptr if ineligible (K%4, N over device height cap, or create fails).
cl_mem GpuOps::get_image(cl_mem w, int N, int K) {
    auto it = img_cache_.find(w);
    if (it != img_cache_.end()) return it->second;
    cl_mem img = nullptr;
    if ((K % 4) == 0) {
        cl_image_format fmt; fmt.image_channel_order = CL_RGBA; fmt.image_channel_data_type = CL_HALF_FLOAT;
        cl_image_desc desc; std::memset(&desc, 0, sizeof(desc));
        desc.image_type = CL_MEM_OBJECT_IMAGE2D;
        desc.image_width = (size_t)(K / 4);
        desc.image_height = (size_t)N;
        desc.image_row_pitch = (size_t)K * sizeof(sstore_t);  // tightly packed [N,K] fp16
        desc.buffer = w;                                      // image-from-buffer (shared backing; 1.2 field)
        cl_int e;
        img = clCreateImage(cl_.context(), CL_MEM_READ_ONLY, &fmt, &desc, nullptr, &e);
        if (e != CL_SUCCESS) img = nullptr;
    }
    img_cache_.emplace(w, img);
    return img;
}

// Register-blocking tile sizes — must match kernels/gemm.cl.
static const int LIN_TN = 4, GEMM_TM = 4, GEMM_TN = 4;
Tensor GpuOps::linear(const Tensor& x, int T, int Din, cl_mem w, cl_mem bias, int Dout) {
    Tensor out = alloc(T * Dout);
    int has_bias = bias ? 1 : 0;
    cl_mem bb = bias ? bias : dummy_;
    cl_int err;
    // int8 weight path (per-row symmetric): halves weight DRAM traffic. Applied to
    // the ENCODER GEMM (T>=32) only — its cosine stays high under per-row int8, so
    // units stay (near) exact. The AR decode (T==1) is NOT quantized: one flipped
    // token derails the whole sequence (measured 62/211). Quantization is done once
    // at weight-load (prequantize_int8), so it's outside the timed forward.
    if (int8_enabled() && T >= 32 && (Din % 8) == 0 && Dout >= 64) {
        const Int8W* q8 = get_int8(w, Dout, Din);
        if (q8) {
            const int TG_BM = 32, TG_BN = 32;
            cl_kernel k = K("linear_gemm_tiled_int8");
            setArg(k, 0, x.buf); setArg(k, 1, q8->q); setArg(k, 2, q8->scale); setArg(k, 3, bb); setArg(k, 4, out.buf);
            setArg(k, 5, T); setArg(k, 6, Din); setArg(k, 7, Dout); setArg(k, 8, has_bias);
            size_t global[2] = {(size_t)((Dout + TG_BN - 1) / TG_BN) * 8, (size_t)((T + TG_BM - 1) / TG_BM) * 8};
            size_t local[2] = {8, 8};
            err = pEnq(k, 2, global, local);
            if (err != CL_SUCCESS) NNOPT_ERROR_FMT("int8 linear enqueue err=%d (T=%d K=%d N=%d)", (int)err, T, Din, Dout);
            return out;
        }
    }
    // NOTE: image-weight tiled GEMM (gemm_tiled_img) was tried for the encoder and
    // REGRESSED (ffn 13.3→20.4 s) — the M=299 encoder GEMM is compute/occupancy-bound,
    // not weight-read-bound (texture cache + int8 both fail to help it). Kernel kept
    // in the image program for reference but not dispatched.
    // Auto-tuned CLBlast HGemm for LARGE-M GEMMs (encoder M≈38): 3.6× the hand kernel.
    // But CLBlast has heavy HOST-side per-call overhead (cache lookup, arg setup, several
    // sub-kernel dispatches) that dwarfs the GPU work of a TINY M GEMM — measured ~2 ms
    // host/call. The text-beam decode (M=5) does ~28 such GEMMs/step → ~400 ms/step of
    // pure host overhead, 91% of the stage (text_beam was 5.9% GPU-utilized). So route
    // small-M GEMMs to the single-launch hand `linear_gemm` (bias folded in, cheap host),
    // and keep CLBlast only for M >= NNOPT_CLBLAST_MINM. (encoder ≈38 stays on CLBlast.)
    // Default 8: text-beam decode (M=5, ~28 GEMMs/step × ~15 steps) → hand (CLBlast host
    // tax × hundreds of calls hurts); but mt_feat/synth (M≈12, ONE pass) → CLBlast (its
    // host tax is paid once, GPU efficiency wins). Measured: threshold 16 regressed
    // mt_feat +0.21s / synth +0.08s by wrongly handing their M=12 GEMMs to the slow kernel.
    static int clblast_minm = []{ const char* e = std::getenv("NNOPT_CLBLAST_MINM"); return e ? atoi(e) : 8; }();
    if (T >= clblast_minm) {
        cl_event gev = nullptr;
        pytorch_linear(cl_.queue(), T, Dout, Din, x.buf, w, out.buf, gpuprof_on() ? &gev : nullptr);
        if (gev) prof_register("clblast_gemm", gev);
        if (has_bias) {
            cl_kernel kb = K("bias_add");
            setArg(kb, 0, out.buf); setArg(kb, 1, bias); setArg(kb, 2, T); setArg(kb, 3, Dout);
            run1d(kb, (size_t)T * Dout);
        }
        return out;
    }
    if (T >= 2) {
        // gemv_m (one WI/output-channel, weight read once) is DEFAULT OFF: measured 2×
        // SLOWER than linear_gemm despite less weight traffic — a single thread's serial
        // K-loop has too little memory-level parallelism to hide DRAM latency on Adreno
        // (needs many concurrent threads). linear_gemm's register tile keeps more in flight
        // and wins. Kept behind NNOPT_GEMVM=1 for reference. (cosine was 1.0 — correct, just slow.)
        static int use_gemvm = []{ const char* e = std::getenv("NNOPT_GEMVM"); return (e && *e == '1') ? 1 : 0; }();
        if (use_gemvm && T <= 8 && (Din % 8) == 0) {
            // one WI per output channel, weight row read ONCE, reused across all M rows.
            cl_kernel k = K("gemv_m");
            setArg(k, 0, x.buf); setArg(k, 1, w); setArg(k, 2, bb); setArg(k, 3, out.buf);
            setArg(k, 4, T); setArg(k, 5, Din); setArg(k, 6, Dout); setArg(k, 7, has_bias);
            size_t global = (size_t)Dout;
            err = pEnq(k, 1, &global, nullptr);
            if (err != CL_SUCCESS) NNOPT_ERROR_FMT("gemv_m enqueue err=%d (T=%d K=%d N=%d)", (int)err, T, Din, Dout);
            return out;
        }
        // M-exact tile (NNOPT_LGM=1): weight read once/WI + no wasted M-tile, BUT measured marginally
        // SLOWER than linear_gemm (text_beam 4963→5145ms) — its runtime-M inner loop doesn't unroll like
        // linear_gemm's fully-static 4×4 tile, losing ILP. DEFAULT OFF; linear_gemm wins for M=5.
        static int use_lgm = []{ const char* e = std::getenv("NNOPT_LGM"); return (e && *e == '1') ? 1 : 0; }();
        if (use_lgm && T <= 8 && (Din % 8) == 0) {
            cl_kernel k = K("linear_gemm_m");
            setArg(k, 0, x.buf); setArg(k, 1, w); setArg(k, 2, bb); setArg(k, 3, out.buf);
            setArg(k, 4, T); setArg(k, 5, Din); setArg(k, 6, Dout); setArg(k, 7, has_bias);
            size_t global = (size_t)((Dout + 3) / 4);
            err = pEnq(k, 1, &global, nullptr);
            if (err != CL_SUCCESS) NNOPT_ERROR_FMT("linear_gemm_m enqueue err=%d (T=%d K=%d N=%d)", (int)err, T, Din, Dout);
            return out;
        }
        // Fallback (M>8 or K%8!=0 or disabled): small-M register GEMM, one launch, bias folded.
        cl_kernel k = K("linear_gemm");
        setArg(k, 0, x.buf); setArg(k, 1, w); setArg(k, 2, bb); setArg(k, 3, out.buf);
        setArg(k, 4, T); setArg(k, 5, Din); setArg(k, 6, Dout); setArg(k, 7, has_bias);
        size_t global[2] = {(size_t)((Dout + GEMM_TN - 1) / GEMM_TN), (size_t)((T + GEMM_TM - 1) / GEMM_TM)};
        err = pEnq(k, 2, global, nullptr);
        if (err != CL_SUCCESS) NNOPT_ERROR_FMT("linear_gemm enqueue err=%d (T=%d K=%d N=%d)", (int)err, T, Din, Dout);
        return out;
    }
    if (T >= 99999) {
        // Dead (CLBlast always handles T>=4 above); kept for reference: the
        // barrier-free 4×8 register GEMM that CLBlast superseded.
        auto r16 = [](size_t v) { return (v + 15) / 16 * 16; };
        size_t global[2] = {r16((Dout + 7) / 8), r16((T + 3) / 4)};
        size_t local[2] = {16, 16};
        const Int8W* q8 = (int8_enabled() && (Din % 8) == 0 && Dout >= 64) ? get_int8(w, Dout, Din) : nullptr;
        if (q8) {
            // int8 weights: the register GEMM is weight-BW-bound → ~2× BW. Small noise.
            cl_kernel k = K("linear_gemm8_int8");
            setArg(k, 0, x.buf); setArg(k, 1, q8->q); setArg(k, 2, q8->scale); setArg(k, 3, bb); setArg(k, 4, out.buf);
            setArg(k, 5, T); setArg(k, 6, Din); setArg(k, 7, Dout); setArg(k, 8, has_bias);
            err = pEnq(k, 2, global, local);
        } else {
            cl_kernel k = K("linear_gemm8");
            setArg(k, 0, x.buf); setArg(k, 1, w); setArg(k, 2, bb); setArg(k, 3, out.buf);
            setArg(k, 4, T); setArg(k, 5, Din); setArg(k, 6, Dout); setArg(k, 7, has_bias);
            err = pEnq(k, 2, global, local);
        }
    } else if (T >= 4) {
        // Text-beam M=5 / teacher-forced: 4×4 register GEMM (linear_gemm8's 4×8 tile
        // REGRESSED 3.6× at M=5 — wastes the small-M dimension + register pressure).
        cl_kernel k = K("linear_gemm");
        setArg(k, 0, x.buf); setArg(k, 1, w); setArg(k, 2, bb); setArg(k, 3, out.buf);
        setArg(k, 4, T); setArg(k, 5, Din); setArg(k, 6, Dout); setArg(k, 7, has_bias);
        size_t global[2] = {(size_t)((Dout + GEMM_TN - 1) / GEMM_TN), (size_t)((T + GEMM_TM - 1) / GEMM_TM)};
        err = pEnq(k, 2, global, nullptr);
    } else if (T >= 99999) {
        // Local-memory tiled GEMM (encoder T=299, adaptor T=38): stages x/w strips
        // into local for ~8× the arithmetic intensity of the register-only kernel.
        const int TG_BM = 32, TG_BN = 32;
        cl_kernel k = K("linear_gemm_tiled");
        setArg(k, 0, x.buf); setArg(k, 1, w); setArg(k, 2, bb); setArg(k, 3, out.buf);
        setArg(k, 4, T); setArg(k, 5, Din); setArg(k, 6, Dout); setArg(k, 7, has_bias);
        size_t global[2] = {(size_t)((Dout + TG_BN - 1) / TG_BN) * 8, (size_t)((T + TG_BM - 1) / TG_BM) * 8};
        size_t local[2] = {8, 8};
        err = pEnq(k, 2, global, local);
    } else if (T >= 4) {
        // M×N register-blocked GEMM (small-M forwards: text beam M=5, teacher-forced).
        // (gemm_coop_m — one wave/output-channel, M accumulators — was tried and
        // REGRESSED 2× at M=5: the per-WG cooperative reduce over N=20005 output
        // channels far outweighs the register tile's 2D x/w reuse.)
        cl_kernel k = K("linear_gemm");
        setArg(k, 0, x.buf); setArg(k, 1, w); setArg(k, 2, bb); setArg(k, 3, out.buf);
        setArg(k, 4, T); setArg(k, 5, Din); setArg(k, 6, Dout); setArg(k, 7, has_bias);
        size_t global[2] = {(size_t)((Dout + GEMM_TN - 1) / GEMM_TN), (size_t)((T + GEMM_TM - 1) / GEMM_TM)};
            err = pEnq(k, 2, global, nullptr);
    } else if (T == 1 && []{ static int e = getenv("NNOPT_INT8DEC") ? 1 : 0; return e; }()
               && (Din % 8) == 0 && Dout >= 64) {
        // EXPERIMENT (NNOPT_INT8DEC): int8 per-row weights on the AR-decode GEMV.
        // Decode is weight-BW-bound, so int8 (1 byte vs 2) should ~2× these GEMVs.
        // KNOWN RISK: a single flipped argmax derails the whole autoregressive
        // sequence (full-decode int8 earlier gave 62/211 units). Gated env so the
        // default fp16 path is untouched. Quantization (get_int8) is one-time at
        // first use, outside the timed forward.
        const Int8W* q8 = get_int8(w, Dout, Din);
        if (!q8) NNOPT_ERROR_FMT("int8 decode quantize failed (K=%d N=%d)", Din, Dout);
        const size_t COOP_WG = 64;
        const bool quad = (Dout % 4) == 0;
        cl_kernel k = K(quad ? "gemv_coop4_int8" : "gemv_coop_int8");
        setArg(k, 0, x.buf); setArg(k, 1, q8->q); setArg(k, 2, q8->scale); setArg(k, 3, bb); setArg(k, 4, out.buf);
        setArg(k, 5, T); setArg(k, 6, Din); setArg(k, 7, Dout); setArg(k, 8, has_bias);
        size_t ng = quad ? (size_t)((Dout + 3) / 4) : (size_t)Dout;
        size_t global = ng * COOP_WG, local = COOP_WG;
        err = pEnq(k, 1, &global, &local);
        if (err != CL_SUCCESS) NNOPT_ERROR_FMT("int8 decode GEMV enqueue err=%d (K=%d N=%d)", (int)err, Din, Dout);
        return out;
    } else if (T == 1 && img_enabled() && (Din % 4) == 0 && Dout >= 64 && get_image(w, Dout, Din)) {
        // image2d (L1 texture-cache) weight GEMV — LOSSLESS 1.71× BW on the
        // bandwidth-bound decode. Falls through to gemv_coop if image ineligible
        // (e.g. text lm_head N=20005 > device image-height cap → get_image nullptr).
        const size_t COOP_WG = 64;
        const bool quad = (Dout % 4) == 0;
        cl_kernel k = K(quad ? "gemv_img4" : "gemv_img");
        cl_mem img = get_image(w, Dout, Din);
        setArg(k, 0, img); setArg(k, 1, x.buf); setArg(k, 2, bb); setArg(k, 3, out.buf);
        setArg(k, 4, T); setArg(k, 5, Din); setArg(k, 6, Dout); setArg(k, 7, has_bias);
        size_t ng = quad ? (size_t)((Dout + 3) / 4) : (size_t)Dout;
        size_t global = ng * COOP_WG, local = COOP_WG;
        err = pEnq(k, 1, &global, &local);
        if (err != CL_SUCCESS) NNOPT_ERROR_FMT("gemv_img enqueue err=%d (K=%d N=%d)", (int)err, Din, Dout);
        return out;
    } else if (T == 1) {
        // Cooperative GEMV (single-token decode hot path): one WG (64 threads =
        // one Adreno wave) per output row, fp32 tree-reduce over K. ~6× the
        // occupancy of linear_fast at M=1 — saturates the memory bus.
        const size_t COOP_WG = 64;
        const bool quad = (Dout % 4) == 0;  // 4-outputs-per-WG when N divisible by 4
        cl_kernel k = K(quad ? "gemv_coop4" : "gemv_coop");
        setArg(k, 0, x.buf); setArg(k, 1, w); setArg(k, 2, bb); setArg(k, 3, out.buf);
        setArg(k, 4, T); setArg(k, 5, Din); setArg(k, 6, Dout); setArg(k, 7, has_bias);
        size_t ng = quad ? (size_t)((Dout + 3) / 4) : (size_t)Dout;
        size_t global = ng * COOP_WG, local = COOP_WG;
        err = pEnq(k, 1, &global, &local);
    } else {
        // N-register-blocked GEMV (rare T=2,3 forwards).
        cl_kernel k = K("linear_fast");
        setArg(k, 0, x.buf); setArg(k, 1, w); setArg(k, 2, bb); setArg(k, 3, out.buf);
        setArg(k, 4, T); setArg(k, 5, Din); setArg(k, 6, Dout); setArg(k, 7, has_bias);
        size_t global[2] = {(size_t)((Dout + LIN_TN - 1) / LIN_TN), (size_t)T};
            err = pEnq(k, 2, global, nullptr);
    }
    if (err != CL_SUCCESS) NNOPT_ERROR_FMT("linear enqueue err=%d (T=%d K=%d N=%d)", (int)err, T, Din, Dout);
    return out;
}

Tensor GpuOps::layernorm(const Tensor& x, int T, int D, cl_mem g, cl_mem b, float eps) {
    Tensor out = alloc(T * D);
    const bool coop = (D % 8) == 0;   // cooperative (one wave/row) when D vectorizable
    cl_kernel k = K(coop ? "layernorm_coop" : "layernorm_forward");
    setArg(k, 0, x.buf); setArg(k, 1, g); setArg(k, 2, b); setArg(k, 3, out.buf);
    setArg(k, 4, T); setArg(k, 5, D); setArg(k, 6, eps);
    if (coop) {
        size_t global = (size_t)T * 64, local = 64;
        cl_int err = pEnq(k, 1, &global, &local);
        if (err != CL_SUCCESS) NNOPT_ERROR_FMT("layernorm_coop enqueue err=%d", (int)err);
    } else {
        run1d(k, (size_t)T);
    }
    return out;
}

Tensor GpuOps::attention(const Tensor& q, const Tensor& k, const Tensor& v,
                         int Tq, int Tk, int H, int Dk, bool causal) {
    int Dm = H * Dk;
    Tensor out = alloc(Tq * Dm);
    cl_kernel kr = K("attention_context");
    int caus = causal ? 1 : 0;
    setArg(kr, 0, q.buf); setArg(kr, 1, k.buf); setArg(kr, 2, v.buf); setArg(kr, 3, out.buf);
    setArg(kr, 4, Tq); setArg(kr, 5, Tk); setArg(kr, 6, H); setArg(kr, 7, Dk); setArg(kr, 8, caus);
    run1d(kr, (size_t)H * Tq);
    return out;
}

Tensor GpuOps::relpos_attention(const Tensor& q, const Tensor& k, const Tensor& v,
                                const Tensor& p, cl_mem bu, cl_mem bv, int T, int L, int H, int Dk) {
    int Dm = H * Dk;
    Tensor out = alloc(T * Dm);
    // Single-WI-per-(head,query) with hoisted q+bias beats the cooperative
    // (8-thread flash-merge) variant on Adreno 620: 4784 work-items already hide
    // the serial-key latency, and the merge adds local-memory + idle-lane overhead.
    cl_kernel kr = K("relpos_attention_context");
    setArg(kr, 0, q.buf); setArg(kr, 1, k.buf); setArg(kr, 2, v.buf); setArg(kr, 3, p.buf);
    setArg(kr, 4, bu); setArg(kr, 5, bv); setArg(kr, 6, out.buf);
    setArg(kr, 7, T); setArg(kr, 8, L); setArg(kr, 9, H); setArg(kr, 10, Dk);
    run1d(kr, (size_t)H * T);
    return out;
}

Tensor GpuOps::conv1d_tc(const Tensor& in, cl_mem w, cl_mem bias, int T, int Cin, int Cout,
                         int Tout, int K_, int stride, int pad, int dil, int groups) {
    // groups=1 conv (adaptor conv_pool): reformulate as im2col + tiled GEMM. Weight
    // [Cout, Cin, K] is already contiguous as [Cout, Cin*K] = the GEMM weight. The
    // scalar conv1d_tc ran this at 0.2 GFLOP/s (1.78 s); the GEMM does it ~25× faster.
    if (groups == 1 && (Cin * K_) % 8 == 0) {
        Tensor col = alloc(Tout * Cin * K_);
        cl_kernel kc = K("im2col_tc");
        setArg(kc, 0, in.buf); setArg(kc, 1, col.buf);
        setArg(kc, 2, T); setArg(kc, 3, Cin); setArg(kc, 4, Tout);
        setArg(kc, 5, K_); setArg(kc, 6, stride); setArg(kc, 7, pad); setArg(kc, 8, dil);
        run1d(kc, (size_t)Tout * Cin * K_);
        return linear(col, Tout, Cin * K_, w, bias, Cout);
    }
    Tensor out = alloc(Tout * Cout);
    cl_kernel k = K((groups == Cout) ? "conv1d_dw" : "conv1d_tc");
    int has_bias = bias ? 1 : 0;
    cl_mem bb = bias ? bias : dummy_;
    setArg(k, 0, in.buf); setArg(k, 1, w); setArg(k, 2, bb); setArg(k, 3, out.buf);
    setArg(k, 4, T); setArg(k, 5, Cin); setArg(k, 6, Cout); setArg(k, 7, Tout);
    setArg(k, 8, K_); setArg(k, 9, stride); setArg(k, 10, pad); setArg(k, 11, dil);
    setArg(k, 12, groups); setArg(k, 13, has_bias);
    run1d(k, (size_t)Tout * Cout);
    return out;
}

cl_mem GpuOps::col_scratch(int n) {
    if (conv_scratch_mode_ < 0) { const char* e = std::getenv("NNOPT_CONV_SCRATCH"); conv_scratch_mode_ = (e && *e == '0') ? 0 : 1; }
    if (conv_scratch_mode_ == 0) return nullptr;
    if (col_scratch_ && col_scratch_n_ >= n) return col_scratch_;
    if (col_scratch_) clReleaseMemObject(col_scratch_);
    cl_int err;
    col_scratch_ = clCreateBuffer(cl_.context(), CL_MEM_READ_WRITE, (size_t)n * sizeof(sstore_t), nullptr, &err);
    if (err != CL_SUCCESS) { col_scratch_ = nullptr; col_scratch_n_ = 0; return nullptr; }
    col_scratch_n_ = n;
    return col_scratch_;
}

Tensor GpuOps::conv1d_ct(const Tensor& in, cl_mem w, cl_mem bias, int Cin, int T, int Cout,
                         int K_, int dil, int pad, int pre_act) {
    Tensor out = alloc(Cout * T);
    const bool quad = (Cout % 4) == 0;  // 4-output-channel tile when divisible
    // (conv1d_ct44 — also tile time ×4 to amortize weight reads — REGRESSED: 4× fewer
    // work-items under-occupies the late vocoder stages (small C, huge T). The conv is
    // occupancy-bound, not weight-BW-bound. Kept ct4.)
    int has_bias = bias ? 1 : 0;
    cl_mem bb = bias ? bias : dummy_;
    // Vocoder conv → im2col + CLBlast GEMM (out[Cout,T] = W[Cout,Cin*K] @ col[Cin*K,T],
    // stays channel-major). For big-C stages the GEMM wins; for small-C/huge-T stages
    // the im2col scratch + small-M GEMM is overhead, so use the direct conv1d_ct4 there.
    static int conv_thr = []{ const char* e = std::getenv("NNOPT_CONV_THRESH"); return e ? atoi(e) : 4; }();
    // Small-Cin DIRECT conv (NNOPT_DIRECT_CIN_MAX>0): for late vocoder stages the weight is TINY
    // (e.g. ups4 16×16×11 = 5.6 KB → stays in cache), so a direct conv's weight re-reads are cache
    // hits, while im2col still streams a ~24 MB col scratch through DRAM (K× write-expansion). Direct
    // wins there. (Early stages have huge weights that don't cache → keep im2col, hence the Cin gate.)
    // Default 16: measured sweep — Cin<=16 (ups4) direct = vocoder −1025ms; Cin=32 marginal; Cin=64
    // REGRESSED (weight too big to cache → im2col GEMM wins). So only the smallest-Cin stage goes direct.
    static int direct_cin_max = []{ const char* e = std::getenv("NNOPT_DIRECT_CIN_MAX"); return e ? atoi(e) : 16; }();
    if (direct_cin_max > 0 && Cin <= direct_cin_max && quad) {
        cl_kernel k = K("conv1d_ct4");
        setArg(k, 0, in.buf); setArg(k, 1, w); setArg(k, 2, bb); setArg(k, 3, out.buf);
        setArg(k, 4, Cin); setArg(k, 5, T); setArg(k, 6, Cout);
        setArg(k, 7, K_); setArg(k, 8, dil); setArg(k, 9, pad); setArg(k, 10, has_bias); setArg(k, 11, pre_act);
        size_t base = (size_t)(Cout / 4) * T; size_t lw = 128, gw = (base + lw - 1) / lw * lw;
        cl_int e2 = pEnq(k, 1, &gw, &lw);
        if (e2 != CL_SUCCESS) NNOPT_ERROR_FMT("conv1d_ct4(direct) enqueue err=%d", (int)e2);
        return out;
    }
    if (Cout >= conv_thr) {
        int CinK = Cin * K_;
        const bool cp = convprof_on();
        auto now = [&]{ return std::chrono::steady_clock::now(); };
        auto finn = [&]{ if (cp) clFinish(cl_.queue()); return now(); };
        cl_mem cols = col_scratch(CinK * T);          // reused persistent scratch (NNOPT_CONV_SCRATCH)
        Tensor col = cols ? Tensor{cols, CinK * T} : alloc(CinK * T);
        auto t0 = cp ? now() : std::chrono::steady_clock::time_point{};
        cl_kernel kc = K("im2col_ct");
        setArg(kc, 0, in.buf); setArg(kc, 1, col.buf);
        setArg(kc, 2, Cin); setArg(kc, 3, T); setArg(kc, 4, K_); setArg(kc, 5, dil); setArg(kc, 6, pad);
        setArg(kc, 7, pre_act);   // fuse the pre-conv activation into the gather (-1 = none)
        // 3D (t,k,ci), driver-chosen workgroup size (forcing local=(64,1,1) REGRESSED 2× — the
        // driver's default 3D WG shape beats a constrained 1D-along-T one for this kernel).
        { size_t g3[3] = {(size_t)T, (size_t)K_, (size_t)Cin}; pEnq(kc, 3, g3, nullptr); }
        auto t1 = finn();
        // Conv GEMM: CLBlast by DEFAULT. A hand conv_gemm_ct was tried (NNOPT_CONVGEMM=1) to dodge
        // CLBlast's apparent ~30ms/call "host overhead" — but it REGRESSED vocoder 8.9→14.9s. That
        // ~30ms is NOT reclaimable host overhead: it's CLBlast's helper kernels (pad/copy for
        // non-tile-aligned dims) which my single-event profiler doesn't capture (so it mislabeled
        // them as idle). CLBlast does the conv GEMM far more efficiently than the naive hand kernel
        // (whose col[k*T+t] access strides by T → cache-thrash). Kept off; correct (cos 0.998) but slow.
        static int conv_gemm_hand = []{ const char* e = std::getenv("NNOPT_CONVGEMM"); return (e && *e == '1') ? 1 : 0; }();
        if (conv_gemm_hand) {
            cl_kernel kg = K("conv_gemm_ct");
            setArg(kg, 0, w); setArg(kg, 1, col.buf); setArg(kg, 2, bb); setArg(kg, 3, out.buf);
            setArg(kg, 4, Cout); setArg(kg, 5, CinK); setArg(kg, 6, T); setArg(kg, 7, has_bias);
            size_t g2[2] = {(size_t)((T + 3) / 4), (size_t)((Cout + 3) / 4)};
            cl_int e2 = pEnq(kg, 2, g2, nullptr);
            if (e2 != CL_SUCCESS) NNOPT_ERROR_FMT("conv_gemm_ct enqueue err=%d (Cout=%d CinK=%d T=%d)", (int)e2, Cout, CinK, T);
            auto t2h = finn();
            if (cp) { cp_im2col_ += std::chrono::duration<double,std::milli>(t1 - t0).count();
                      cp_gemm_   += std::chrono::duration<double,std::milli>(t2h - t1).count(); cp_calls_++; }
            // bias folded into conv_gemm_ct → skip the separate bias_add_ct pass below.
            return out;
        }
        cl_event gev = nullptr;
        pytorch_conv1d(cl_.queue(), Cout, T, CinK, w, col.buf, out.buf, gpuprof_on() ? &gev : nullptr);   // out[Cout,T] = W @ col
        if (gev) prof_register("clblast_conv_gemm", gev);
        auto t2 = finn();
        if (has_bias) {
            // bias is per output channel (Cout), broadcast over T: out[co*T + t] += bias[co].
            cl_kernel kb = K("bias_add_ct");
            setArg(kb, 0, out.buf); setArg(kb, 1, bb); setArg(kb, 2, Cout); setArg(kb, 3, T);
            run1d(kb, (size_t)Cout * T);
        }
        auto t3 = finn();
        if (cp) {
            cp_im2col_ += std::chrono::duration<double,std::milli>(t1 - t0).count();
            cp_gemm_   += std::chrono::duration<double,std::milli>(t2 - t1).count();
            cp_bias_   += std::chrono::duration<double,std::milli>(t3 - t2).count();
            cp_calls_++;
        }
        return out;
    }
    // Fallback direct path (Cout < conv_thr, e.g. conv_post Cout=1). conv1d_ct4 supports pre_act;
    // the scalar conv1d_ct does not — guard only that case (conv_post uses pre_act=-1 anyway).
    if (pre_act >= 0 && !quad) NNOPT_ERROR_FMT("conv1d_ct scalar path can't fuse pre_act (Cout=%d)", Cout);
    cl_kernel k = K(quad ? "conv1d_ct4" : "conv1d_ct");
    setArg(k, 0, in.buf); setArg(k, 1, w); setArg(k, 2, bb); setArg(k, 3, out.buf);
    setArg(k, 4, Cin); setArg(k, 5, T); setArg(k, 6, Cout);
    setArg(k, 7, K_); setArg(k, 8, dil); setArg(k, 9, pad); setArg(k, 10, has_bias);
    if (quad) setArg(k, 11, pre_act);
    size_t base = (size_t)(quad ? (Cout / 4) : Cout) * T;
    size_t lw = 128, gw = (base + lw - 1) / lw * lw;
    pEnq(k, 1, &gw, &lw);
    return out;
}

Tensor GpuOps::conv_transpose1d_ct(const Tensor& in, cl_mem w, cl_mem bias, int Cin, int T,
                                   int Cout, int K_, int stride, int pad) {
    int Tout = (T - 1) * stride - 2 * pad + K_;
    Tensor out = alloc(Cout * Tout);
    int has_bias = bias ? 1 : 0;
    cl_mem bb = bias ? bias : dummy_;
    // Transposed conv → GEMM (tmp[Cout*K,T] = Wr[Cout*K,Cin] @ in[Cin,T] via CLBlast)
    // + col2im fold. Replaces the scalar conv_transpose1d_ct4. Wr repacked once.
    cl_mem wr = (Cout >= 4) ? get_wt_transpose(w, Cin, Cout, K_) : nullptr;
    if (wr) {
        Tensor tmp = alloc(Cout * K_ * T);
        cl_event gev = nullptr;
        pytorch_conv1d(cl_.queue(), Cout * K_, T, Cin, wr, in.buf, tmp.buf, gpuprof_on() ? &gev : nullptr);  // tmp[Cout*K,T]
        if (gev) prof_register("clblast_convT_gemm", gev);
        cl_kernel kf = K("col2im_transpose");
        setArg(kf, 0, tmp.buf); setArg(kf, 1, bb); setArg(kf, 2, out.buf);
        setArg(kf, 3, Cout); setArg(kf, 4, K_); setArg(kf, 5, T); setArg(kf, 6, Tout);
        setArg(kf, 7, stride); setArg(kf, 8, pad); setArg(kf, 9, has_bias);
        run1d(kf, (size_t)Cout * Tout);
        return out;
    }
    const bool quad = (Cout % 4) == 0;
    cl_kernel k = K(quad ? "conv_transpose1d_ct4" : "conv_transpose1d_ct");
    setArg(k, 0, in.buf); setArg(k, 1, w); setArg(k, 2, bb); setArg(k, 3, out.buf);
    setArg(k, 4, Cin); setArg(k, 5, T); setArg(k, 6, Cout); setArg(k, 7, Tout);
    setArg(k, 8, K_); setArg(k, 9, stride); setArg(k, 10, pad); setArg(k, 11, has_bias);
    run1d(k, (size_t)(quad ? (Cout / 4) : Cout) * Tout);
    return out;
}

Tensor GpuOps::glu_tc(const Tensor& in, int T, int C) {
    Tensor out = alloc(T * C);
    cl_kernel k = K("glu_tc");
    setArg(k, 0, in.buf); setArg(k, 1, out.buf); setArg(k, 2, T); setArg(k, 3, C);
    run1d(k, (size_t)T * C);
    return out;
}

Tensor GpuOps::batchnorm_tc(const Tensor& in, int T, int C, cl_mem mean, cl_mem var,
                            cl_mem g, cl_mem b, float eps) {
    Tensor out = alloc(T * C);
    cl_kernel k = K("batchnorm_tc");
    setArg(k, 0, in.buf); setArg(k, 1, out.buf); setArg(k, 2, mean); setArg(k, 3, var);
    setArg(k, 4, g); setArg(k, 5, b); setArg(k, 6, T); setArg(k, 7, C); setArg(k, 8, eps);
    run1d(k, (size_t)T * C);
    return out;
}

Tensor GpuOps::vocoder_input_gather(cl_mem code_ids, cl_mem lang, cl_mem dict, cl_mem spkr,
                                    int LE, int CE, int SE, int T0, int lang_id, int spkr_id) {
    const int Cin = LE + CE + SE;
    Tensor out = alloc(Cin * T0);
    cl_kernel k = K("vocoder_input_gather");
    setArg(k, 0, lang); setArg(k, 1, dict); setArg(k, 2, spkr); setArg(k, 3, code_ids);
    setArg(k, 4, out.buf); setArg(k, 5, LE); setArg(k, 6, CE); setArg(k, 7, SE);
    setArg(k, 8, T0); setArg(k, 9, lang_id); setArg(k, 10, spkr_id);
    size_t g2[2] = {(size_t)T0, (size_t)Cin};
    cl_int err = pEnq(k, 2, g2, nullptr);
    if (err != CL_SUCCESS) NNOPT_ERROR_FMT("vocoder_input_gather enqueue err=%d", (int)err);
    return out;
}

Tensor GpuOps::embed_scale_pos(cl_mem token_ids, cl_mem emb, int T, int Dm, float embed_scale,
                               int start, int pos_stride) {
    Tensor out = alloc(T * Dm);
    cl_kernel k = K("embed_scale_pos");
    setArg(k, 0, token_ids); setArg(k, 1, emb); setArg(k, 2, out.buf);
    setArg(k, 3, T); setArg(k, 4, Dm); setArg(k, 5, embed_scale); setArg(k, 6, start); setArg(k, 7, pos_stride);
    run1d(k, (size_t)T * (Dm / 2));
    return out;
}

Tensor GpuOps::fbank_window(const Tensor& audio, int nframes, int FL, int FS) {
    Tensor out = alloc(nframes * FL);
    cl_kernel k = K("fbank_window");
    setArg(k, 0, audio.buf); setArg(k, 1, out.buf); setArg(k, 2, nframes); setArg(k, 3, FL); setArg(k, 4, FS);
    run1d(k, (size_t)nframes);
    return out;
}

cl_mem GpuOps::fbank_power(const Tensor& win, int nframes, int FL, int NFFT, int NSPEC) {
    cl_mem power = alloc_f32(nframes * NSPEC);
    cl_kernel k = K("fbank_power");
    setArg(k, 0, win.buf); setArg(k, 1, power); setArg(k, 2, nframes);
    setArg(k, 3, FL); setArg(k, 4, NFFT); setArg(k, 5, NSPEC);
    run1d(k, (size_t)nframes * NSPEC);
    return power;
}

Tensor GpuOps::fbank_mel(cl_mem power, int nframes, int NB, int NSPEC, int SR, int NFFT, float eps) {
    Tensor out = alloc(nframes * NB);
    cl_kernel k = K("fbank_mel");
    setArg(k, 0, power); setArg(k, 1, out.buf); setArg(k, 2, nframes); setArg(k, 3, NB);
    setArg(k, 4, NSPEC); setArg(k, 5, SR); setArg(k, 6, NFFT); setArg(k, 7, eps);
    run1d(k, (size_t)nframes * NB);
    return out;
}

Tensor GpuOps::relpos_pos_emb(int L, int Dm, int ET) {
    Tensor out = alloc(L * Dm);
    cl_kernel k = K("relpos_pos_emb");
    setArg(k, 0, out.buf); setArg(k, 1, L); setArg(k, 2, Dm); setArg(k, 3, ET);
    run1d(k, (size_t)L * (Dm / 2));
    return out;
}

void GpuOps::act(const Tensor& t, int code) { act_n(t, t.n, code); }

void GpuOps::act_n(const Tensor& t, int n, int code) {
    cl_kernel k = K("act_inplace");
    setArg(k, 0, t.buf); setArg(k, 1, n); setArg(k, 2, code);
    run1d(k, (size_t)n);
}

void GpuOps::axpy(const Tensor& a, const Tensor& b, float alpha) {
    cl_kernel k = K("axpy_inplace");
    setArg(k, 0, a.buf); setArg(k, 1, b.buf); setArg(k, 2, alpha); setArg(k, 3, a.n);
    run1d(k, (size_t)a.n);
}

void GpuOps::scale(const Tensor& a, float alpha) {
    cl_kernel k = K("scale_inplace");
    setArg(k, 0, a.buf); setArg(k, 1, alpha); setArg(k, 2, a.n);
    run1d(k, (size_t)a.n);
}

void GpuOps::logits_to_f32(const Tensor& logits, cl_mem dst, int off, int n) {
    cl_kernel k = K("logits_to_f32");
    setArg(k, 0, logits.buf); setArg(k, 1, dst); setArg(k, 2, off); setArg(k, 3, n);
    run1d(k, (size_t)n);
}

void GpuOps::log_softmax_region(const Tensor& logits, cl_mem dst, int off, int vocab) {
    // No-barrier reduction sequence (see decode.cl). 256 work-items stride the vocab.
    cl_kernel k1 = K("lsm_partial_max");
    setArg(k1, 0, logits.buf); setArg(k1, 1, vocab); setArg(k1, 2, lsm_partial_);
    run1d(k1, 256);
    cl_kernel k2 = K("lsm_reduce_max");
    setArg(k2, 0, lsm_partial_); setArg(k2, 1, lsm_scalar_);
    run1d(k2, 1);
    cl_kernel k3 = K("lsm_partial_sum");
    setArg(k3, 0, logits.buf); setArg(k3, 1, vocab); setArg(k3, 2, lsm_scalar_); setArg(k3, 3, lsm_partial_);
    run1d(k3, 256);
    cl_kernel k4 = K("lsm_reduce_sum");
    setArg(k4, 0, lsm_partial_); setArg(k4, 1, lsm_scalar_);
    run1d(k4, 1);
    cl_kernel k5 = K("lsm_write");
    setArg(k5, 0, logits.buf); setArg(k5, 1, dst); setArg(k5, 2, off); setArg(k5, 3, vocab); setArg(k5, 4, lsm_scalar_);
    run1d(k5, (size_t)vocab);
}

void GpuOps::mask_region(cl_mem buf, int off, int vocab, int pad, int unk, int force_tok,
                         bool suppress_eos, int eos) {
    cl_kernel k = K("mask_region");
    int seos = suppress_eos ? 1 : 0;
    setArg(k, 0, buf); setArg(k, 1, off); setArg(k, 2, vocab); setArg(k, 3, pad);
    setArg(k, 4, unk); setArg(k, 5, force_tok); setArg(k, 6, seos); setArg(k, 7, eos);
    run1d(k, (size_t)vocab);
}

void GpuOps::add_scalar_region(cl_mem buf, int off, int n, float scalar) {
    cl_kernel k = K("add_scalar_region");
    setArg(k, 0, buf); setArg(k, 1, off); setArg(k, 2, n); setArg(k, 3, scalar);
    run1d(k, (size_t)n);
}

cl_mem GpuOps::alloc_ints(int n) {
    cl_mem b = pool_alloc((size_t)n * sizeof(int), "ints");
    if (!b) { NNOPT_ERROR_FMT("alloc_ints(%d) failed", n); return nullptr; }
    return b;
}
void GpuOps::argmax_dev(cl_mem buf, int n) {
    cl_kernel k = K("argmax_coop");
    setArg(k, 0, buf); setArg(k, 1, n); setArg(k, 2, out_idx_); setArg(k, 3, out_val_);
    size_t g = 64, l = 64; pEnq(k, 1, &g, &l);   // no host readback — token stays on GPU
}
void GpuOps::copy_ints(cl_mem dst, int dst_off, cl_mem src, int src_off, int n) {
    clEnqueueCopyBuffer(cl_.queue(), src, dst, (size_t)src_off * sizeof(int), (size_t)dst_off * sizeof(int),
                        (size_t)n * sizeof(int), 0, nullptr, nullptr);
}
void GpuOps::download_ints(cl_mem buf, int off, int n, int* host) {
    clEnqueueReadBuffer(cl_.queue(), buf, CL_TRUE, (size_t)off * sizeof(int), (size_t)n * sizeof(int), host, 0, nullptr, nullptr);
}

// Beam top-k entirely on-GPU: cand_size rounds of (argmax -> record -> mask), reading
// nothing back. The caller does ONE bulk read of cand_idx/cand_val afterward, replacing
// cand_size blocking argmax readbacks per beam step.
void GpuOps::topk_dev(cl_mem buf, int n, int cand_size, cl_mem cand_idx, cl_mem cand_val, float negval) {
    cl_kernel ka = K("argmax_coop");
    cl_kernel ks = K("set_at_dev");
    for (int c = 0; c < cand_size; ++c) {
        setArg(ka, 0, buf); setArg(ka, 1, n); setArg(ka, 2, out_idx_); setArg(ka, 3, out_val_);
        size_t g = 64, l = 64; pEnq(ka, 1, &g, &l);
        setArg(ks, 0, buf); setArg(ks, 1, out_idx_); setArg(ks, 2, out_val_); setArg(ks, 3, negval);
        setArg(ks, 4, cand_idx); setArg(ks, 5, cand_val); setArg(ks, 6, c);
        run1d(ks, 1);
    }
}

void GpuOps::argmax_f32(cl_mem buf, int n, int& idx, float& val) {
    cl_kernel k = K("argmax_coop");
    setArg(k, 0, buf); setArg(k, 1, n); setArg(k, 2, out_idx_); setArg(k, 3, out_val_);
    { size_t g = 64, l = 64; pEnq(k, 1, &g, &l); }
    clEnqueueReadBuffer(cl_.queue(), out_idx_, CL_TRUE, 0, sizeof(int), &idx, 0, nullptr, nullptr);
    clEnqueueReadBuffer(cl_.queue(), out_val_, CL_TRUE, 0, sizeof(float), &val, 0, nullptr, nullptr);
}

std::vector<float> GpuOps::download_f32(cl_mem buf, int n) {
    std::vector<float> out(n);
    clEnqueueReadBuffer(cl_.queue(), buf, CL_TRUE, 0, (size_t)n * sizeof(float), out.data(), 0, nullptr, nullptr);
    return out;
}

void GpuOps::set_at_f32(cl_mem buf, int idx, float val) {
    cl_kernel k = K("set_at_f32");
    setArg(k, 0, buf); setArg(k, 1, idx); setArg(k, 2, val);
    run1d(k, 1);
}

// ----------------------------------------------------------------- audio
cl_mem GpuOps::upload_s16(const std::vector<int16_t>& s) {
    cl_int err;
    cl_mem b = tracked_clCreateBuffer(cl_.context(), CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
                                      s.size() * sizeof(int16_t), (void*)s.data(), &err, "s16");
    if (err != CL_SUCCESS || !b) { NNOPT_ERROR_FMT("upload_s16 failed err=%d", (int)err); return nullptr; }
    arena_.push_back(b);
    return b;
}

cl_mem GpuOps::alloc_s16(int n) {
    cl_mem b = pool_alloc((size_t)n * sizeof(int16_t), "s16_out");
    if (!b) { NNOPT_ERROR("alloc_s16 failed"); return nullptr; }
    return b;
}

std::vector<int16_t> GpuOps::download_s16(cl_mem buf, int n) {
    std::vector<int16_t> out(n);
    clEnqueueReadBuffer(cl_.queue(), buf, CL_TRUE, 0, (size_t)n * sizeof(int16_t), out.data(), 0, nullptr, nullptr);
    return out;
}

Tensor GpuOps::audio_decode(cl_mem s16, int nframes, int nch) {
    Tensor out = alloc(nframes);
    cl_kernel k = K("audio_decode_s16");
    setArg(k, 0, s16); setArg(k, 1, out.buf); setArg(k, 2, nframes); setArg(k, 3, nch);
    run1d(k, (size_t)nframes);
    return out;
}

Tensor GpuOps::audio_resample(const Tensor& in, int nin, int rate_in, int rate_out, int& nout) {
    nout = (int)((long long)nin * rate_out / rate_in);
    float ratio = (float)rate_in / (float)rate_out;
    Tensor out = alloc(nout);
    cl_kernel k = K("audio_resample_linear");
    setArg(k, 0, in.buf); setArg(k, 1, out.buf); setArg(k, 2, nin); setArg(k, 3, nout); setArg(k, 4, ratio);
    run1d(k, (size_t)nout);
    return out;
}

cl_mem GpuOps::audio_encode_s16(const Tensor& wav, int n) {
    cl_mem out = alloc_s16(n);
    cl_kernel k = K("audio_encode_s16");
    setArg(k, 0, wav.buf); setArg(k, 1, out); setArg(k, 2, n);
    run1d(k, (size_t)n);
    return out;
}
