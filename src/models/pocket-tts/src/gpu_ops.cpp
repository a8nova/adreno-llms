// gpu_ops.cpp — implementations. See gpu_ops.h + .nnport/PORT_SPEC.md.
#include "gpu_ops.h"
#include "nnopt_error.h"
#include "profiler.h"
#include <clblast.h>      // OverrideParameters — force the recordable direct GEMM kernel
#include <cstring>

GpuOps::GpuOps(OpenCLContext& ctx, Weights& weights) : ctx_(ctx), weights_(weights) {}

GpuOps::~GpuOps() {
    if (offb_bb_) clReleaseMemObject(offb_bb_);
    if (offb_mm_) clReleaseMemObject(offb_mm_);
    if (clblast_temp_) clReleaseMemObject(clblast_temp_);
    if (cur_recording_) ctx_.release_recording(cur_recording_);
    if (rec_q_) clReleaseCommandQueue(rec_q_);
    for (cl_kernel k : rec_kernels_) if (k) clReleaseKernel(k);
    for (cl_mem b : pinned_) if (b) clReleaseMemObject(b);
    if (tc_cache_) clReleaseMemObject(tc_cache_);
    for (auto& kv : pool_) for (cl_mem b : kv.second) if (b) clReleaseMemObject(b);
    for (auto& kv : kcache_) if (kv.second) clReleaseKernel(kv.second);
    if (prog_) clReleaseProgram(prog_);
    if (utils_prog_) clReleaseProgram(utils_prog_);
}

bool GpuOps::init() {
    custom_gemm_ = getenv("NNOPT_CUSTOM_GEMM") && atoi(getenv("NNOPT_CUSTOM_GEMM"));
    if (getenv("NNOPT_GEMM_PARAMS")) {
        clblast::Precision prec = sizeof(nnopt_storage_t) == 2 ? clblast::Precision::kHalf : clblast::Precision::kSingle;
        for (const char* kn : {"Xgemm", "XgemmDirect", "GemmRoutine"}) {
            std::unordered_map<std::string,size_t> p;
            auto st = clblast::RetrieveParameters(ctx_.device(), kn, prec, p);
            fprintf(stderr, "PARAMS %s st=%d: ", kn, (int)st);
            for (auto& kv : p) fprintf(stderr, "%s=%zu ", kv.first.c_str(), kv.second);
            fprintf(stderr, "\n");
        }
    }
    // NNOPT_RECORD_MIMI: include mimi in the recorded frame. Transformer GEMMs use the
    // recordable ALIGNED path (tile-padded + pre-transposed → CLBlast's tuned indirect Xgemm
    // as a single recordable NDRange, see linear()); convs/convtr fall back to the recordable
    // tiled kernel. NO XGEMM_MIN_INDIRECT_SIZE override — keep the fast indirect kernel.
    rec_mimi_ = getenv("NNOPT_RECORD_MIMI") && atoi(getenv("NNOPT_RECORD_MIMI"));
    // build_program_from_file auto-injects -D USE_FP16=1 under NNOPT_USE_FP16.
    prog_ = ctx_.build_program_from_file("kernels/ops.cl");
    if (!prog_) { NNOPT_ERROR("GpuOps: failed to build kernels/ops.cl"); return false; }
    utils_prog_ = ctx_.build_program_from_file("kernels/utils.cl");
    if (!utils_prog_) { NNOPT_ERROR("GpuOps: failed to build kernels/utils.cl"); return false; }
    return true;
}

cl_kernel GpuOps::kernel(const std::string& name) {
    auto it = kcache_.find(name);
    if (it != kcache_.end()) return it->second;
    cl_int err = CL_SUCCESS;
    cl_kernel k = clCreateKernel(prog_, name.c_str(), &err);
    if (err != CL_SUCCESS || !k) {
        NNOPT_ERROR_FMT("GpuOps: clCreateKernel('%s') failed (%d)", name.c_str(), err);
        return nullptr;
    }
    kcache_[name] = k;
    return k;
}

cl_kernel GpuOps::rec_kernel(const std::string& kname) {
    // Recording captures each dispatch's args from its kernel object; a SHARED cached
    // kernel reused across layers would replay all dispatches with the LAST args. So
    // during recording every dispatch gets a FRESH kernel, and we collect the per-step
    // offset/Tkv arg slots (qkv_rope_cache.arg8, attention_wg.arg8/arg5) for override.
    cl_int err = CL_SUCCESS;
    cl_kernel k = clCreateKernel(prog_, kname.c_str(), &err);
    if (err != CL_SUCCESS || !k) { NNOPT_ERROR_FMT("rec_kernel('%s') failed (%d)", kname.c_str(), err); return nullptr; }
    rec_kernels_.push_back(k);
    if (kname == "qkv_rope_cache")      rec_offset_args_.push_back({k, 8u});
    else if (kname == "attention_wg") { rec_offset_args_.push_back({k, 8u}); rec_tkv_args_.push_back({k, 5u}); }
    return k;
}

bool GpuOps::run1d(const std::string& kname, size_t global,
                   std::vector<std::pair<size_t,const void*>> args) {
    KernelProfiler::HostTimer _ht(KernelProfiler::host_run1d_ms);
    cl_kernel k = recording_ ? rec_kernel(kname) : kernel(kname);
    if (!k) return false;
    for (unsigned i = 0; i < args.size(); ++i) {
        cl_int e = clSetKernelArg(k, i, args[i].first, args[i].second);
        if (e != CL_SUCCESS) { NNOPT_ERROR_FMT("%s: setarg %u failed (%d)", kname.c_str(), i, e); return false; }
    }
    size_t gws = global;
    cl_command_queue q = recording_ ? rec_q_ : ctx_.queue();
    cl_event* ev = recording_ ? nullptr : KernelProfiler::event_for(kname.c_str());
    cl_int e = clEnqueueNDRangeKernel(q, k, 1, nullptr, &gws, nullptr, 0, nullptr, ev);
    if (e != CL_SUCCESS) { NNOPT_ERROR_FMT("%s: enqueue failed (%d)", kname.c_str(), e); return false; }
    return true;
}

bool GpuOps::run1d_lws(const std::string& kname, size_t global, size_t local,
                       std::vector<std::pair<size_t,const void*>> args) {
    KernelProfiler::HostTimer _ht(KernelProfiler::host_run1d_ms);
    cl_kernel k = recording_ ? rec_kernel(kname) : kernel(kname);
    if (!k) return false;
    for (unsigned i = 0; i < args.size(); ++i) {
        cl_int e = clSetKernelArg(k, i, args[i].first, args[i].second);
        if (e != CL_SUCCESS) { NNOPT_ERROR_FMT("%s: setarg %u failed (%d)", kname.c_str(), i, e); return false; }
    }
    size_t gws = global, lws = local;
    cl_command_queue q = recording_ ? rec_q_ : ctx_.queue();
    cl_event* ev = recording_ ? nullptr : KernelProfiler::event_for(kname.c_str());
    cl_int e = clEnqueueNDRangeKernel(q, k, 1, nullptr, &gws, &lws, 0, nullptr, ev);
    if (e != CL_SUCCESS) { NNOPT_ERROR_FMT("%s: enqueue(lws) failed (%d)", kname.c_str(), e); return false; }
    return true;
}

bool GpuOps::run2d_lws(const std::string& kname, size_t g0, size_t g1, size_t l0, size_t l1,
                       std::vector<std::pair<size_t,const void*>> args) {
    KernelProfiler::HostTimer _ht(KernelProfiler::host_run1d_ms);
    cl_kernel k = recording_ ? rec_kernel(kname) : kernel(kname);
    if (!k) return false;
    for (unsigned i = 0; i < args.size(); ++i) {
        cl_int e = clSetKernelArg(k, i, args[i].first, args[i].second);
        if (e != CL_SUCCESS) { NNOPT_ERROR_FMT("%s: setarg %u failed (%d)", kname.c_str(), i, e); return false; }
    }
    size_t gws[2] = {g0, g1}, lws[2] = {l0, l1};
    cl_command_queue q = recording_ ? rec_q_ : ctx_.queue();
    cl_event* ev = recording_ ? nullptr : KernelProfiler::event_for(kname.c_str());
    cl_int e = clEnqueueNDRangeKernel(q, k, 2, nullptr, gws, lws, 0, nullptr, ev);
    if (e != CL_SUCCESS) { NNOPT_ERROR_FMT("%s: enqueue(2d) failed (%d)", kname.c_str(), e); return false; }
    return true;
}
bool GpuOps::gemm_tiled(int M, int N, int K, cl_mem A, cl_mem B, cl_mem C,
                        int transA, int transB, cl_mem bias, int has_bias) {
    const size_t TS = 16;
    cl_mem bb = has_bias ? bias : A;   // dummy when no bias
    size_t g0 = ((N + TS - 1) / TS) * TS;   // dim0 = columns (N)
    size_t g1 = ((M + TS - 1) / TS) * TS;   // dim1 = rows (M)
    return run2d_lws("gemm_tiled", g0, g1, TS, TS,
        {{sizeof(cl_mem),&A},{sizeof(cl_mem),&B},{sizeof(cl_mem),&C},
         {sizeof(int),&M},{sizeof(int),&N},{sizeof(int),&K},
         {sizeof(int),&transA},{sizeof(int),&transB},
         {sizeof(cl_mem),&bb},{sizeof(int),&has_bias}});
}

// ── record/replay (cl_qcom_recordable_queues) ────────────────────────────────
bool GpuOps::begin_recording() {
    if (!ctx_.has_recordable_queues()) return false;
    if (!rec_q_) rec_q_ = ctx_.create_recordable_queue();
    if (!rec_q_) return false;
    cur_recording_ = ctx_.new_recording(rec_q_);
    if (!cur_recording_) return false;
    recording_ = true;
    return true;
}
cl_recording_qcom GpuOps::end_recording() {
    ctx_.end_recording(cur_recording_);
    recording_ = false;
    return cur_recording_;
}
void GpuOps::replay(cl_recording_qcom rec, size_t n, const cl_array_arg_qcom* ov) {
    ctx_.enqueue_recording(ctx_.queue(), rec, n, ov);
}
cl_mem GpuOps::offset_buf() {
    cl_mem& slot = use_mimi_off_ ? offb_mm_ : offb_bb_;
    if (!slot) { cl_int err = CL_SUCCESS; slot = clCreateBuffer(ctx_.context(), CL_MEM_READ_WRITE, sizeof(int), nullptr, &err); }
    return slot;
}
void GpuOps::use_mimi_offset(bool m) { use_mimi_off_ = m; }
bool GpuOps::conv_gemm(int M, int N, int K, cl_mem x, cl_mem W, cl_mem out) {
    // out[M,N] = x[M,K] @ W[K,N]. CLBlast routes onto aq() (the recordable queue during
    // recording); XGEMM_MIN_INDIRECT_SIZE is overridden so it uses the DIRECT kernel
    // (single NDRange, recordable). custom_gemm_ forces the tiled fallback (debug only).
    if (recording_ || custom_gemm_)
        return gemm_tiled(M, N, K, x, W, out, /*transA=*/0, /*transB=*/0, nullptr, 0); // x[M,K]@W[K,N]
    return pytorch_conv1d(aq(), M, N, K, x, W, out, clblast_temp());
}
bool GpuOps::conv_gemm_atb(int M, int N, int Kc, cl_mem A, cl_mem B, cl_mem C) {
    // C[M,N] = A[Kc,M]^T @ B[Kc,N]. CLBlast direct on aq() (recordable); tiled fallback.
    if (recording_ || custom_gemm_)
        return gemm_tiled(M, N, Kc, A, B, C, /*transA=*/1, /*transB=*/0, nullptr, 0); // A[Kc,M]^T@B[Kc,N]
    return gemm_AtB(aq(), M, N, Kc, A, B, C, clblast_temp());
}
static void _set_int_live(cl_command_queue q, cl_kernel k, cl_mem buf, int val) {
    clSetKernelArg(k, 0, sizeof(cl_mem), &buf);
    clSetKernelArg(k, 1, sizeof(int), &val);
    size_t gws = 1;
    clEnqueueNDRangeKernel(q, k, 1, nullptr, &gws, nullptr, 0, nullptr, nullptr);
}
cl_mem GpuOps::transposed_weight(const std::string& w_key, int N, int K) {
    // [N,K] → [K,N], cached. MUST be called outside recording (transpose2d would be
    // captured otherwise) — the recording's aligned GEMM reads the cached buffer.
    auto it = wt_cache_.find(w_key);
    if (it != wt_cache_.end()) return it->second;
    cl_mem w = weights_.get_buffer(w_key);
    if (!w) { NNOPT_ERROR_FMT("transposed_weight: missing '%s'", w_key.c_str()); return nullptr; }
    cl_mem wt = transpose2d(w, N, K);          // [K,N]
    pinned_.push_back(wt);                      // keep alive for the whole run (never pooled)
    wt_cache_[w_key] = wt;
    return wt;
}
void GpuOps::prewarm_mimi_gemm() {
    for (const char* l : {"0", "1"}) {
        std::string p = std::string("mimi.decoder_transformer.transformer.layers.") + l;
        transposed_weight(p + ".self_attn.in_proj.weight", 1536, 512);
        transposed_weight(p + ".self_attn.out_proj.weight", 512, 512);
        transposed_weight(p + ".linear1.weight", 2048, 512);
        transposed_weight(p + ".linear2.weight", 512, 2048);
    }
}
cl_mem GpuOps::clblast_temp() {
    // Persistent temp buffer for CLBlast's indirect GEMM. Without it, CLBlast allocates
    // (clCreateBuffer) a fresh temp on EVERY call — ~per-call host cost we pay 19×/frame.
    // 16M fp16 elements covers the largest padded A+B+C among mimi GEMMs.
    if (!clblast_temp_) {
        cl_int err = CL_SUCCESS;
        clblast_temp_ = clCreateBuffer(ctx_.context(), CL_MEM_READ_WRITE,
                                       (size_t)16 * 1024 * 1024 * sizeof(nnopt_storage_t), nullptr, &err);
        if (err != CL_SUCCESS) { NNOPT_ERROR_FMT("clblast_temp alloc failed (%d)", err); return nullptr; }
    }
    return clblast_temp_;
}
void GpuOps::zero_buffer(cl_mem b, size_t numel) {
    // In-place zero on the LIVE queue (in-order). fp16/fp32 zero == all-zero bytes.
    unsigned char zero = 0;
    clEnqueueFillBuffer(ctx_.queue(), b, &zero, 1, 0, numel * sizeof(nnopt_storage_t), 0, nullptr, nullptr);
}
void GpuOps::write_offset(int off) {
    // set_int on the LIVE queue (never captured into a recording). Value copied at
    // enqueue time → race-free under async replay. Writes the ACTIVE offset buffer.
    _set_int_live(ctx_.queue(), kernel("set_int"), offset_buf(), off);
}
void GpuOps::write_replay_offsets(int bb, int mm) {
    bool save = use_mimi_off_;
    use_mimi_off_ = false; cl_mem b0 = offset_buf();
    use_mimi_off_ = true;  cl_mem b1 = offset_buf();
    use_mimi_off_ = save;
    cl_kernel k = kernel("set_int");
    _set_int_live(ctx_.queue(), k, b0, bb);
    _set_int_live(ctx_.queue(), k, b1, mm);
}

// ── buffers ──────────────────────────────────────────────────────────────
cl_mem GpuOps::alloc(size_t numel) {
    KernelProfiler::HostTimer _ht(KernelProfiler::host_alloc_ms);
    size_t bytes = numel * sizeof(nnopt_storage_t);
    if (!recording_) {                           // pool path (reuse freed buffers)
        auto it = pool_.find(bytes);
        if (it != pool_.end() && !it->second.empty()) {
            cl_mem b = it->second.back(); it->second.pop_back();
            return b;
        }
    }
    cl_int err = CL_SUCCESS;
    cl_mem b = clCreateBuffer(ctx_.context(), CL_MEM_READ_WRITE, bytes, nullptr, &err);
    if (err != CL_SUCCESS || !b) { NNOPT_ERROR_FMT("alloc(%zu) failed (%d)", numel, err); return nullptr; }
    if (recording_) pinned_.push_back(b);        // PIN: stays valid for all replays
    return b;
}

void GpuOps::release(cl_mem b) {
    KernelProfiler::HostTimer _ht(KernelProfiler::host_release_ms);
    if (recording_) return;                      // pinned during recording — never freed/reused
    // Return to the size-keyed pool instead of clReleaseMemObject. The in-order
    // queue guarantees any kernel still reading b runs before the next kernel that
    // re-allocates and writes it, so reuse is safe without an explicit sync.
    if (!b) return;
    size_t bytes = 0;
    if (clGetMemObjectInfo(b, CL_MEM_SIZE, sizeof(bytes), &bytes, nullptr) == CL_SUCCESS && bytes)
        pool_[bytes].push_back(b);
    else
        clReleaseMemObject(b);
}

cl_mem GpuOps::upload(const std::vector<float>& host) {
    KernelProfiler::HostTimer _ht(KernelProfiler::host_upload_ms);
    std::vector<nnopt_storage_t> tmp(host.size());
#ifdef NNOPT_USE_FP16
    for (size_t i = 0; i < host.size(); ++i) tmp[i] = nnopt_storage_t(nnopt_f32_to_f16(host[i]));
#else
    for (size_t i = 0; i < host.size(); ++i) tmp[i] = host[i];
#endif
    cl_int err = CL_SUCCESS;
    cl_mem b = clCreateBuffer(ctx_.context(), CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR,
                              tmp.size() * sizeof(nnopt_storage_t), tmp.data(), &err);
    if (err != CL_SUCCESS || !b) { NNOPT_ERROR_FMT("upload(%zu) failed (%d)", host.size(), err); return nullptr; }
    return b;
}

std::vector<float> GpuOps::download(cl_mem buf, size_t numel) {
    std::vector<nnopt_storage_t> tmp(numel);
    clEnqueueReadBuffer(ctx_.queue(), buf, CL_TRUE, 0, numel * sizeof(nnopt_storage_t), tmp.data(), 0, nullptr, nullptr);
    std::vector<float> out(numel);
#ifdef NNOPT_USE_FP16
    for (size_t i = 0; i < numel; ++i) out[i] = nnopt_f16_to_f32(static_cast<uint16_t>(tmp[i]));
#else
    for (size_t i = 0; i < numel; ++i) out[i] = tmp[i];
#endif
    return out;
}

// ── linear ───────────────────────────────────────────────────────────────
cl_mem GpuOps::linear(cl_mem x, int M, int N, int K,
                      const std::string& w_key, const std::string& bias_key) {
    cl_mem W = weights_.get_buffer(w_key);
    if (!W) { NNOPT_ERROR_FMT("linear: missing weight '%s'", w_key.c_str()); return nullptr; }
    cl_mem out = alloc((size_t)M * N);
    if (!out) return nullptr;
    if (M == 1) {
        // M=1 decode path: custom fp16 GEMV (CLBlast Gemm under-utilizes at M=1).
        // NOTE: a texture-weight variant (image1d_buffer view → Adreno texture
        // cache, gemv_fp16_img) REGRESSED here (backbone +28%): a pure M=1 GEMV
        // reads each weight exactly ONCE, so there is no reuse for the texture
        // cache to exploit — only sampling overhead. Texture weights would only
        // pay where weights are reused (GEMM M>1, or convs across time).
        // bias fused INTO the GEMV (saves a separate add_bias enqueue — we're host-bound).
        int has_bias = bias_key.empty() ? 0 : 1;
        cl_mem b = has_bias ? weights_.get_buffer(bias_key) : W;   // dummy when no bias
        bool ok;
        if (K >= 512 && K % 8 == 0) {
            // large-K: workgroup-per-row reduction GEMV (coalesced, high thread count).
            // lws=128 measured best on Adreno 620 (64 and 256 both slower).
            const size_t lws = 128;
            ok = run1d_lws("gemv_fp16_wg", (size_t)N * lws, lws,
                   {{sizeof(cl_mem),&W},{sizeof(cl_mem),&x},{sizeof(cl_mem),&b},{sizeof(cl_mem),&out},
                    {sizeof(int),&N},{sizeof(int),&K},{sizeof(int),&has_bias}});
        } else {
            ok = run1d("gemv_fp16", (size_t)N,
                   {{sizeof(cl_mem),&W},{sizeof(cl_mem),&x},{sizeof(cl_mem),&b},{sizeof(cl_mem),&out},
                    {sizeof(int),&N},{sizeof(int),&K},{sizeof(int),&has_bias}});
        }
        if (!ok) {
            NNOPT_ERROR_FMT("linear: gemv failed for '%s'", w_key.c_str());
            release(out); return nullptr;
        }
        return out;   // bias already applied inside the GEMV
    } else if (recording_ || custom_gemm_) {
        // Recordable tiled GEMM (bias fused). NOTE: CLBlast's tuned indirect Xgemm CANNOT be
        // recorded on this driver — it returns -59 even with tile-aligned, non-transposed inputs
        // (only the slow XgemmDirect records). So a fully-recorded mimi frame is stuck with this
        // slow tiled kernel; that's why mimi runs LIVE by default and rec_mimi_ is reference-only.
        int has_bias = bias_key.empty() ? 0 : 1;
        cl_mem bb = has_bias ? weights_.get_buffer(bias_key) : W;
        if (!gemm_tiled(M, N, K, x, W, out, /*transA=*/0, /*transB=*/1, bb, has_bias)) {
            NNOPT_ERROR_FMT("linear: rec GEMM failed for '%s'", w_key.c_str());
            release(out); return nullptr;
        }
        return out;   // bias fused
    } else if (!pytorch_linear(aq(), M, N, K, x, W, out, clblast_temp())) {
        // Persistent temp buffer avoids CLBlast's per-call temp alloc (the only host-side
        // win available: its ~6 ms/call is mostly intrinsic C++ orchestration on this device's
        // CPU, not the pad/copy kernels — aligning inputs to skip them was wall-neutral).
        NNOPT_ERROR_FMT("linear: GEMM failed for '%s'", w_key.c_str());
        release(out); return nullptr;
    }
    if (!bias_key.empty()) add_bias_inplace(out, M, N, bias_key);
    return out;
}

// ── elementwise / activations ──────────────────────────────────────────────
cl_mem GpuOps::copy(cl_mem in, size_t n) {
    cl_mem out = alloc(n); if (!out) return nullptr;
    int ni = (int)n;
    run1d("copy_buf", n, {{sizeof(cl_mem),&in},{sizeof(cl_mem),&out},{sizeof(int),&ni}});
    return out;
}
cl_mem GpuOps::elu(cl_mem in, size_t n) {
    cl_mem out = alloc(n); if (!out) return nullptr; int ni=(int)n;
    run1d("elu", n, {{sizeof(cl_mem),&in},{sizeof(cl_mem),&out},{sizeof(int),&ni}});
    return out;
}
cl_mem GpuOps::silu(cl_mem in, size_t n) {
    cl_mem out = alloc(n); if (!out) return nullptr; int ni=(int)n;
    run1d("silu", n, {{sizeof(cl_mem),&in},{sizeof(cl_mem),&out},{sizeof(int),&ni}});
    return out;
}
cl_mem GpuOps::gelu(cl_mem in, size_t n) {
    cl_mem out = alloc(n); if (!out) return nullptr; int ni=(int)n;
    run1d("gelu", n, {{sizeof(cl_mem),&in},{sizeof(cl_mem),&out},{sizeof(int),&ni}});
    return out;
}
cl_mem GpuOps::add(cl_mem a, cl_mem b, size_t n) {
    cl_mem out = alloc(n); if (!out) return nullptr; int ni=(int)n;
    run1d("add_buf", n, {{sizeof(cl_mem),&a},{sizeof(cl_mem),&b},{sizeof(cl_mem),&out},{sizeof(int),&ni}});
    return out;
}
void GpuOps::add_inplace(cl_mem a, cl_mem b, size_t n) {
    int ni=(int)n;
    run1d("add_inplace", n, {{sizeof(cl_mem),&a},{sizeof(cl_mem),&b},{sizeof(int),&ni}});
}
cl_mem GpuOps::mul(cl_mem a, cl_mem b, size_t n) {
    cl_mem out = alloc(n); if (!out) return nullptr; int ni=(int)n;
    run1d("mul_buf", n, {{sizeof(cl_mem),&a},{sizeof(cl_mem),&b},{sizeof(cl_mem),&out},{sizeof(int),&ni}});
    return out;
}
void GpuOps::add_bias_inplace(cl_mem x, int rows, int C, const std::string& bias_key) {
    cl_mem b = weights_.get_buffer(bias_key);
    if (!b) { NNOPT_ERROR_FMT("add_bias: missing '%s'", bias_key.c_str()); return; }
    run1d("add_bias", (size_t)rows*C, {{sizeof(cl_mem),&x},{sizeof(cl_mem),&b},{sizeof(int),&rows},{sizeof(int),&C}});
}
cl_mem GpuOps::scale(cl_mem in, float s, size_t n) {
    cl_mem out = alloc(n); if (!out) return nullptr; int ni=(int)n;
    run1d("scale_buf", n, {{sizeof(cl_mem),&in},{sizeof(cl_mem),&out},{sizeof(float),&s},{sizeof(int),&ni}});
    return out;
}
cl_mem GpuOps::mul_channel(cl_mem in, int rows, int C, const std::string& scale_key) {
    cl_mem sv = weights_.get_buffer(scale_key);
    if (!sv) { NNOPT_ERROR_FMT("mul_channel: missing '%s'", scale_key.c_str()); return nullptr; }
    cl_mem out = alloc((size_t)rows*C); if (!out) return nullptr;
    run1d("mul_channel", (size_t)rows*C, {{sizeof(cl_mem),&in},{sizeof(cl_mem),&sv},{sizeof(cl_mem),&out},{sizeof(int),&rows},{sizeof(int),&C}});
    return out;
}

// ── normalization ──────────────────────────────────────────────────────────
cl_mem GpuOps::layernorm(cl_mem x, int rows, int C, float eps,
                         const std::string& w_key, const std::string& b_key) {
    cl_mem out = alloc((size_t)rows*C); if (!out) return nullptr;
    int affine = (!w_key.empty()) ? 1 : 0;
    cl_mem w = affine ? weights_.get_buffer(w_key) : x;   // dummy buf when !affine
    cl_mem b = affine ? weights_.get_buffer(b_key) : x;
    if (affine && (!w || !b)) { NNOPT_ERROR_FMT("layernorm: missing '%s'/'%s'", w_key.c_str(), b_key.c_str()); release(out); return nullptr; }
    const size_t lws = 64;   // workgroup-per-row reduction (decode rows=1 → 1 thread otherwise)
    run1d_lws("layernorm_wg", (size_t)rows * lws, lws,
          {{sizeof(cl_mem),&x},{sizeof(cl_mem),&out},{sizeof(cl_mem),&w},{sizeof(cl_mem),&b},
           {sizeof(int),&rows},{sizeof(int),&C},{sizeof(float),&eps},{sizeof(int),&affine}});
    return out;
}
cl_mem GpuOps::rmsnorm(cl_mem x, int rows, int C, float eps, const std::string& alpha_key) {
    cl_mem a = weights_.get_buffer(alpha_key);
    if (!a) { NNOPT_ERROR_FMT("rmsnorm: missing '%s'", alpha_key.c_str()); return nullptr; }
    cl_mem out = alloc((size_t)rows*C); if (!out) return nullptr;
    run1d("rmsnorm_var", (size_t)rows,
          {{sizeof(cl_mem),&x},{sizeof(cl_mem),&out},{sizeof(cl_mem),&a},
           {sizeof(int),&rows},{sizeof(int),&C},{sizeof(float),&eps}});
    return out;
}
cl_mem GpuOps::modulate(cl_mem x, cl_mem shift, cl_mem scale, int rows, int C) {
    cl_mem out = alloc((size_t)rows*C); if (!out) return nullptr;
    run1d("modulate", (size_t)rows*C,
          {{sizeof(cl_mem),&x},{sizeof(cl_mem),&shift},{sizeof(cl_mem),&scale},{sizeof(cl_mem),&out},
           {sizeof(int),&rows},{sizeof(int),&C}});
    return out;
}
cl_mem GpuOps::modulate_packed(cl_mem x, cl_mem mod, int C, int shiftOff, int scaleOff) {
    cl_mem out = alloc((size_t)C); if (!out) return nullptr;
    run1d("modulate_packed", (size_t)C,
          {{sizeof(cl_mem),&x},{sizeof(cl_mem),&mod},{sizeof(cl_mem),&out},
           {sizeof(int),&C},{sizeof(int),&shiftOff},{sizeof(int),&scaleOff}});
    return out;
}
cl_mem GpuOps::mul_off(cl_mem a, int aOff, cl_mem b, int C) {
    cl_mem out = alloc((size_t)C); if (!out) return nullptr;
    run1d("mul_off", (size_t)C,
          {{sizeof(cl_mem),&a},{sizeof(cl_mem),&b},{sizeof(cl_mem),&out},{sizeof(int),&aOff},{sizeof(int),&C}});
    return out;
}
cl_mem GpuOps::denorm_latent(cl_mem x, int rows, int C,
                             const std::string& std_key, const std::string& mean_key) {
    cl_mem sv = weights_.get_buffer(std_key), mv = weights_.get_buffer(mean_key);
    if (!sv || !mv) { NNOPT_ERROR_FMT("denorm: missing '%s'/'%s'", std_key.c_str(), mean_key.c_str()); return nullptr; }
    cl_mem out = alloc((size_t)rows*C); if (!out) return nullptr;
    run1d("denorm_latent", (size_t)rows*C,
          {{sizeof(cl_mem),&x},{sizeof(cl_mem),&sv},{sizeof(cl_mem),&mv},{sizeof(cl_mem),&out},
           {sizeof(int),&rows},{sizeof(int),&C}});
    return out;
}

// ── convolutions ─────────────────────────────────────────────────────────
cl_mem GpuOps::conv1d_causal(cl_mem x, int Cin, int Cout, int T, int K, int dil,
                             const std::string& w_key, const std::string& bias_key) {
    cl_mem w = weights_.get_buffer(w_key);
    if (!w) { NNOPT_ERROR_FMT("conv1d: missing '%s'", w_key.c_str()); return nullptr; }
    int has_bias = !bias_key.empty();
    cl_mem b = has_bias ? weights_.get_buffer(bias_key) : x;
    if (has_bias && !b) { NNOPT_ERROR_FMT("conv1d: missing bias '%s'", bias_key.c_str()); return nullptr; }
    cl_mem out = alloc((size_t)Cout * T); if (!out) return nullptr;
    if (K == 1) {
        // pointwise conv = matmul out[Cout,T] = W[Cout,Cin] @ x[Cin,T]. T==1 → GEMV;
        // T>1 → CLBlast GEMM. (The naive conv1d_causal kernel under-uses the ALU.)
        int hb0 = 0;
        bool ok = (T == 1)
            ? run1d("gemv_fp16", (size_t)Cout,
                    {{sizeof(cl_mem),&w},{sizeof(cl_mem),&x},{sizeof(cl_mem),&x},{sizeof(cl_mem),&out},
                     {sizeof(int),&Cout},{sizeof(int),&Cin},{sizeof(int),&hb0}})
            : conv_gemm(Cout, T, Cin, w, x, out);
        if (!ok) { NNOPT_ERROR_FMT("conv1d_causal(K=1) failed '%s'", w_key.c_str()); release(out); return nullptr; }
        if (has_bias)
            run1d("add_bias_rows", (size_t)Cout * T,
                  {{sizeof(cl_mem),&out},{sizeof(cl_mem),&b},{sizeof(int),&Cout},{sizeof(int),&T}});
        return out;
    }
    run1d("conv1d_causal", (size_t)Cout * T,
          {{sizeof(cl_mem),&x},{sizeof(cl_mem),&w},{sizeof(cl_mem),&b},{sizeof(cl_mem),&out},
           {sizeof(int),&Cin},{sizeof(int),&Cout},{sizeof(int),&T},{sizeof(int),&K},
           {sizeof(int),&dil},{sizeof(int),&has_bias}});
    return out;
}
cl_mem GpuOps::convtranspose1d(cl_mem x, int Cin, int Cout, int T, int K, int S,
                               const std::string& w_key, const std::string& bias_key) {
    cl_mem w = weights_.get_buffer(w_key);
    if (!w) { NNOPT_ERROR_FMT("convtr: missing '%s'", w_key.c_str()); return nullptr; }
    int has_bias = !bias_key.empty();
    cl_mem b = has_bias ? weights_.get_buffer(bias_key) : x;
    if (has_bias && !b) { NNOPT_ERROR_FMT("convtr: missing bias '%s'", bias_key.c_str()); return nullptr; }
    int Tout = T * S;
    cl_mem out = alloc((size_t)Cout * Tout); if (!out) return nullptr;
    run1d("convtranspose1d", (size_t)Cout * Tout,
          {{sizeof(cl_mem),&x},{sizeof(cl_mem),&w},{sizeof(cl_mem),&b},{sizeof(cl_mem),&out},
           {sizeof(int),&Cin},{sizeof(int),&Cout},{sizeof(int),&T},{sizeof(int),&K},
           {sizeof(int),&S},{sizeof(int),&Tout},{sizeof(int),&has_bias}});
    return out;
}
cl_mem GpuOps::convtranspose1d_depthwise(cl_mem x, int C, int T, int K, int S,
                                         const std::string& w_key) {
    cl_mem w = weights_.get_buffer(w_key);
    if (!w) { NNOPT_ERROR_FMT("convtr_dw: missing '%s'", w_key.c_str()); return nullptr; }
    int Tout = T * S;
    cl_mem out = alloc((size_t)C * Tout); if (!out) return nullptr;
    run1d("convtranspose1d_depthwise", (size_t)C * Tout,
          {{sizeof(cl_mem),&x},{sizeof(cl_mem),&w},{sizeof(cl_mem),&out},
           {sizeof(int),&C},{sizeof(int),&T},{sizeof(int),&K},{sizeof(int),&S},{sizeof(int),&Tout}});
    return out;
}
// ── streaming conv variants ──────────────────────────────────────────────
cl_mem GpuOps::conv1d_streaming(cl_mem x, cl_mem leftctx, int Cin, int Cout, int T, int K, int dil,
                                const std::string& w_key, const std::string& bias_key) {
    cl_mem w = weights_.get_buffer(w_key);
    if (!w) { NNOPT_ERROR_FMT("conv1d_streaming: missing '%s'", w_key.c_str()); return nullptr; }
    int has_bias = !bias_key.empty();
    cl_mem b = has_bias ? weights_.get_buffer(bias_key) : x;
    int P = (K - 1) * dil;
    cl_mem out = alloc((size_t)Cout * T); if (!out) return nullptr;
    // GEMM-map (Opt #3): out[Cout,T] = W[Cout,Cin*K] @ xcol[Cin*K,T] via CLBlast.
    // (Measured ~13 ms/frame faster than the naive kernel once per-frame syncs were
    // removed — the GPU savings dominate CLBlast host overhead.) K=1 → xcol == x.
    cl_mem xcol = x;
    bool made_xcol = false;
    if (K > 1) {
        xcol = alloc((size_t)Cin * K * T);
        if (!xcol) { release(out); return nullptr; }
        made_xcol = true;
        run1d("conv1d_im2col", (size_t)Cin * K * T,
              {{sizeof(cl_mem),&x},{sizeof(cl_mem),&leftctx},{sizeof(cl_mem),&xcol},
               {sizeof(int),&Cin},{sizeof(int),&T},{sizeof(int),&K},{sizeof(int),&dil},{sizeof(int),&P}});
    }
    if (!conv_gemm(Cout, T, Cin * K, w, xcol, out)) {
        NNOPT_ERROR_FMT("conv1d_streaming: gemm failed '%s'", w_key.c_str());
        if (made_xcol) release(xcol); release(out); return nullptr;
    }
    if (made_xcol) release(xcol);
    if (has_bias)
        run1d("add_bias_rows", (size_t)Cout * T,
              {{sizeof(cl_mem),&out},{sizeof(cl_mem),&b},{sizeof(int),&Cout},{sizeof(int),&T}});
    if (P > 0)  // update left-context = input's last P columns
        run1d("extract_tail", (size_t)Cin * P,
              {{sizeof(cl_mem),&x},{sizeof(cl_mem),&leftctx},{sizeof(int),&Cin},{sizeof(int),&T},{sizeof(int),&P}});
    return out;
}
cl_mem GpuOps::convtranspose1d_streaming(cl_mem x, cl_mem partial, int Cin, int Cout, int T, int K, int S,
                                         const std::string& w_key, const std::string& bias_key) {
    cl_mem w = weights_.get_buffer(w_key);
    if (!w) { NNOPT_ERROR_FMT("convtr_streaming: missing '%s'", w_key.c_str()); return nullptr; }
    int has_bias = !bias_key.empty();
    cl_mem b = has_bias ? weights_.get_buffer(bias_key) : x;
    int PT = K - S, Tfull = (T - 1) * S + K, Tout = T * S;
    cl_mem full = alloc((size_t)Cout * Tfull); if (!full) return nullptr;
    // GEMM-map (Opt #1, the 43.6%-of-GPU kernel): cols[Cout*K, T] = W[Cin,Cout*K]^T
    // @ in[Cin,T] via CLBlast (coalesced Cin reduction), then a light col2im scatter.
    // Replaces the naive convtranspose1d_full (strided Cin loop, ~0.36 GFLOP/s).
    int MK = Cout * K;
    cl_mem cols = alloc((size_t)MK * T); if (!cols) { release(full); return nullptr; }
    if (!conv_gemm_atb(MK, T, Cin, w, x, cols)) {
        NNOPT_ERROR_FMT("convtr_streaming: gemm_AtB failed '%s'", w_key.c_str());
        release(cols); release(full); return nullptr;
    }
    run1d("convtr_col2im", (size_t)Cout * Tfull,
          {{sizeof(cl_mem),&cols},{sizeof(cl_mem),&b},{sizeof(cl_mem),&full},
           {sizeof(int),&Cout},{sizeof(int),&K},{sizeof(int),&S},{sizeof(int),&T},
           {sizeof(int),&Tfull},{sizeof(int),&has_bias}});
    release(cols);
    if (PT > 0)
        run1d("overlap_add", (size_t)Cout * PT,
              {{sizeof(cl_mem),&full},{sizeof(cl_mem),&partial},{sizeof(int),&Cout},{sizeof(int),&Tfull},{sizeof(int),&PT}});
    cl_mem out = slice_cols(full, Cout, Tfull, 0, Tout);
    if (PT > 0)  // new partial = full[:, Tout:Tout+PT] - bias
        run1d("extract_cols_sub_bias", (size_t)Cout * PT,
              {{sizeof(cl_mem),&full},{sizeof(cl_mem),&b},{sizeof(cl_mem),&partial},
               {sizeof(int),&Cout},{sizeof(int),&Tfull},{sizeof(int),&Tout},{sizeof(int),&PT},{sizeof(int),&has_bias}});
    release(full);
    return out;
}
cl_mem GpuOps::convtranspose1d_depthwise_streaming(cl_mem x, cl_mem partial, int C, int T, int K, int S,
                                                   const std::string& w_key) {
    cl_mem w = weights_.get_buffer(w_key);
    if (!w) { NNOPT_ERROR_FMT("convtr_dw_streaming: missing '%s'", w_key.c_str()); return nullptr; }
    int PT = K - S, Tfull = (T - 1) * S + K, Tout = T * S;
    cl_mem full = alloc((size_t)C * Tfull); if (!full) return nullptr;
    run1d("convtranspose1d_dw_full", (size_t)C * Tfull,
          {{sizeof(cl_mem),&x},{sizeof(cl_mem),&w},{sizeof(cl_mem),&full},
           {sizeof(int),&C},{sizeof(int),&T},{sizeof(int),&K},{sizeof(int),&S},{sizeof(int),&Tfull}});
    if (PT > 0)
        run1d("overlap_add", (size_t)C * PT,
              {{sizeof(cl_mem),&full},{sizeof(cl_mem),&partial},{sizeof(int),&C},{sizeof(int),&Tfull},{sizeof(int),&PT}});
    cl_mem out = slice_cols(full, C, Tfull, 0, Tout);
    int no_bias = 0;
    if (PT > 0)
        run1d("extract_cols_sub_bias", (size_t)C * PT,
              {{sizeof(cl_mem),&full},{sizeof(cl_mem),&full},{sizeof(cl_mem),&partial},
               {sizeof(int),&C},{sizeof(int),&Tfull},{sizeof(int),&Tout},{sizeof(int),&PT},{sizeof(int),&no_bias}});
    release(full);
    return out;
}
cl_mem GpuOps::seanet_resnet_block_streaming(cl_mem x, int dim, int T, const std::string& prefix, cl_mem b1_ctx) {
    int hid = dim / 2;
    cl_mem v1 = elu(x, (size_t)dim * T);
    cl_mem v2 = conv1d_streaming(v1, b1_ctx, dim, hid, T, 3, 1, prefix + ".block.1.conv.weight", prefix + ".block.1.conv.bias");
    cl_mem v3 = elu(v2, (size_t)hid * T);
    cl_mem v4 = conv1d_causal(v3, hid, dim, T, 1, 1, prefix + ".block.3.conv.weight", prefix + ".block.3.conv.bias");
    cl_mem out = add(x, v4, (size_t)dim * T);
    release(v1); release(v2); release(v3); release(v4);
    return out;
}

// SEANetResnetBlock: residual = ELU→conv(k3 dim→dim/2)→ELU→conv(k1 dim/2→dim); out = x + residual.
cl_mem GpuOps::seanet_resnet_block(cl_mem x, int dim, int T, const std::string& prefix) {
    int hid = dim / 2;
    cl_mem v1 = elu(x, (size_t)dim * T);
    cl_mem v2 = conv1d_causal(v1, dim, hid, T, 3, 1, prefix + ".block.1.conv.weight", prefix + ".block.1.conv.bias");
    cl_mem v3 = elu(v2, (size_t)hid * T);
    cl_mem v4 = conv1d_causal(v3, hid, dim, T, 1, 1, prefix + ".block.3.conv.weight", prefix + ".block.3.conv.bias");
    cl_mem out = add(x, v4, (size_t)dim * T);
    release(v1); release(v2); release(v3); release(v4);
    return out;
}

// ── transformer ────────────────────────────────────────────────────────────
cl_mem GpuOps::transpose2d(cl_mem in, int A, int B) {
    cl_mem out = alloc((size_t)A * B); if (!out) return nullptr;
    run1d("transpose2d", (size_t)A * B, {{sizeof(cl_mem),&in},{sizeof(cl_mem),&out},{sizeof(int),&A},{sizeof(int),&B}});
    return out;
}
cl_mem GpuOps::slice_cols(cl_mem in, int T, int full, int c0, int w) {
    cl_mem out = alloc((size_t)T * w); if (!out) return nullptr;
    run1d("slice_cols", (size_t)T * w,
          {{sizeof(cl_mem),&in},{sizeof(cl_mem),&out},{sizeof(int),&T},{sizeof(int),&full},
           {sizeof(int),&c0},{sizeof(int),&w}});
    return out;
}
void GpuOps::rope_inplace(cl_mem x, int T, int H, int D, int offset, float max_period) {
    run1d("rope_inplace", (size_t)T * H * (D / 2),
          {{sizeof(cl_mem),&x},{sizeof(int),&T},{sizeof(int),&H},{sizeof(int),&D},
           {sizeof(int),&offset},{sizeof(float),&max_period}});
}
cl_mem GpuOps::qkv_rope_cache(cl_mem qkv, int T, int d, int H, int Dh, int offset,
                              float max_period, cl_mem kc, cl_mem vc) {
    (void)offset;   // offset now comes from offset_buf() (set via write_offset), not a scalar
    cl_mem q_out = alloc((size_t)T * d); if (!q_out) return nullptr;
    cl_mem ob = offset_buf();
    run1d("qkv_rope_cache", (size_t)T * H * (Dh / 2),
          {{sizeof(cl_mem),&qkv},{sizeof(cl_mem),&q_out},{sizeof(cl_mem),&kc},{sizeof(cl_mem),&vc},
           {sizeof(int),&T},{sizeof(int),&d},{sizeof(int),&H},{sizeof(int),&Dh},
           {sizeof(cl_mem),&ob},{sizeof(float),&max_period}});
    return q_out;
}
cl_mem GpuOps::attention(cl_mem q, cl_mem kc, cl_mem vc, int Tq, int Tkv, int H, int D,
                         int offset, int context) {
    (void)Tkv; (void)offset;   // Tkv = offset+Tq computed in-kernel; offset from offset_buf()
    cl_mem out = alloc((size_t)Tq * H * D); if (!out) return nullptr;
    // workgroup-per-(tq,h): LSZ threads parallelize over KV positions (the old
    // per-item kernel ran only Tq*H = 16 work-items at decode).
    const size_t lws = 64;
    cl_mem ob = offset_buf();
    run1d_lws("attention_wg", (size_t)Tq * H * lws, lws,
          {{sizeof(cl_mem),&q},{sizeof(cl_mem),&kc},{sizeof(cl_mem),&vc},{sizeof(cl_mem),&out},
           {sizeof(int),&Tq},{sizeof(int),&H},{sizeof(int),&D},
           {sizeof(cl_mem),&ob},{sizeof(int),&context}});
    return out;
}
void GpuOps::cache_append(cl_mem cache, cl_mem block, int offset, int T, int d) {
    size_t es = sizeof(nnopt_storage_t);
    clEnqueueCopyBuffer(ctx_.queue(), block, cache, 0, (size_t)offset * d * es,
                        (size_t)T * d * es, 0, nullptr, nullptr);
}
cl_mem GpuOps::transformer_layer(cl_mem x, int T, int d, int H, int dff, int offset,
                                 int context, const std::string& prefix,
                                 bool layer_scale, cl_mem kc, cl_mem vc, float max_period) {
    int Dh = d / H;
    // Set the device-buffer offset for qkv_rope_cache/attention. On the LIVE path do it
    // here (per layer; all layers in a call share `offset`). During RECORDING we skip it
    // so the write isn't captured — the replay loop sets offset_buf per frame instead.
    if (!recording_) write_offset(offset);
    // self-attention block (pre-norm)
    cl_mem n1 = layernorm(x, T, d, 1e-5f, prefix + ".norm1.weight", prefix + ".norm1.bias");
    cl_mem qkv = linear(n1, T, 3 * d, d, prefix + ".self_attn.in_proj.weight");  // no bias
    // Fused: split qkv + RoPE(q,k) + write k,v into kc/vc at offset → roped q[T,d].
    // (Replaces 3×slice_cols + 2×rope_inplace + 2×cache_append = 7 launches.)
    cl_mem q = qkv_rope_cache(qkv, T, d, H, Dh, offset, max_period, kc, vc);
    cl_mem attn = attention(q, kc, vc, T, offset + T, H, Dh, offset, context);
    cl_mem ao = linear(attn, T, d, d, prefix + ".self_attn.out_proj.weight");    // no bias
    if (layer_scale) { cl_mem s = mul_channel(ao, T, d, prefix + ".layer_scale_1.scale"); release(ao); ao = s; }
    cl_mem x1 = add(x, ao, (size_t)T * d);
    release(n1); release(qkv); release(q); release(attn); release(ao);
    // FFN block (pre-norm, GELU)
    cl_mem n2 = layernorm(x1, T, d, 1e-5f, prefix + ".norm2.weight", prefix + ".norm2.bias");
    cl_mem h1 = linear(n2, T, dff, d, prefix + ".linear1.weight");   // no bias
    cl_mem h1g = gelu(h1, (size_t)T * dff);
    cl_mem h2 = linear(h1g, T, d, dff, prefix + ".linear2.weight");  // no bias
    if (layer_scale) { cl_mem s = mul_channel(h2, T, d, prefix + ".layer_scale_2.scale"); release(h2); h2 = s; }
    cl_mem out = add(x1, h2, (size_t)T * d);
    release(n2); release(h1); release(h1g); release(h2); release(x1);
    return out;
}

// ── flow-net denoiser (stateless) ───────────────────────────────────────────
// ResBlock: mod=Linear(SiLU(y)) → (shift,scale,gate); h=mlp(modulate(in_ln(x),shift,scale));
// return x + gate*h.  in_ln=LayerNorm(C,eps1e-6,affine); mlp=Linear,SiLU,Linear. PORT_SPEC §2e.
cl_mem GpuOps::flow_resblock(cl_mem x, cl_mem y, int C, const std::string& prefix) {
    cl_mem ys = silu(y, C);
    cl_mem mod = linear(ys, 1, 3 * C, C, prefix + ".adaLN_modulation.1.weight", prefix + ".adaLN_modulation.1.bias");
    cl_mem xn = layernorm(x, 1, C, 1e-6f, prefix + ".in_ln.weight", prefix + ".in_ln.bias");
    cl_mem hm = modulate_packed(xn, mod, C, 0, C);        // shift@0, scale@C — no slices
    cl_mem h0 = linear(hm, 1, C, C, prefix + ".mlp.0.weight", prefix + ".mlp.0.bias");
    cl_mem h0s = silu(h0, C);
    cl_mem h2 = linear(h0s, 1, C, C, prefix + ".mlp.2.weight", prefix + ".mlp.2.bias");
    cl_mem gh = mul_off(mod, 2 * C, h2, C);               // gate@2C * h2
    cl_mem out = add(x, gh, C);
    release(ys); release(mod);
    release(xn); release(hm); release(h0); release(h0s); release(h2); release(gh);
    return out;
}
// FinalLayer: mod=Linear(SiLU(y))→(shift,scale); linear(modulate(norm_final(x),shift,scale)).
// norm_final = LayerNorm(C, affine=False, eps1e-6). PORT_SPEC §2f.
cl_mem GpuOps::flow_final_layer(cl_mem x, cl_mem y, int C, int out_dim, const std::string& prefix) {
    cl_mem ys = silu(y, C);
    cl_mem mod = linear(ys, 1, 2 * C, C, prefix + ".adaLN_modulation.1.weight", prefix + ".adaLN_modulation.1.bias");
    cl_mem xn = layernorm(x, 1, C, 1e-6f, "", "");      // affine=False
    cl_mem hm = modulate_packed(xn, mod, C, 0, C);      // shift@0, scale@C — no slices
    cl_mem out = linear(hm, 1, out_dim, C, prefix + ".linear.weight", prefix + ".linear.bias");
    release(ys); release(mod); release(xn); release(hm);
    return out;
}
// flow_net(c[1024], s, t, x_noise[32]) → [32]. PORT_SPEC §2c.
cl_mem GpuOps::flow_net(cl_mem c, float s, float t, cl_mem x_noise) {
    const std::string P = "flow_lm.flow_net";
    cl_mem xp = linear(x_noise, 1, 512, 32, P + ".input_proj.weight", P + ".input_proj.bias");
    // The timestep contribution tc = (embed(s)+embed(t))/2 depends ONLY on (s,t),
    // which are constant (0,1) for every decode frame — cache it instead of
    // recomputing ~10 kernels + 2 host uploads each frame. (Host-bound win.)
    if (!tc_cache_ || tc_s_ != s || tc_t_ != t) {
        if (tc_cache_) { clReleaseMemObject(tc_cache_); tc_cache_ = nullptr; }
        cl_mem te0 = timestep_embedder(s, P + ".time_embed.0");
        cl_mem te1 = timestep_embedder(t, P + ".time_embed.1");
        cl_mem tsum = add(te0, te1, 512);
        tc_cache_ = scale(tsum, 0.5f, 512);             // (te0+te1)/2 — kept (NOT pooled)
        tc_s_ = s; tc_t_ = t;
        release(te0); release(te1); release(tsum);
    }
    cl_mem ce = linear(c, 1, 512, 1024, P + ".cond_embed.weight", P + ".cond_embed.bias");
    cl_mem y = add(tc_cache_, ce, 512);
    release(ce);
    for (int i = 0; i < 6; ++i) {
        cl_mem nx = flow_resblock(xp, y, 512, P + ".res_blocks." + std::to_string(i));
        release(xp); xp = nx;
    }
    cl_mem out = flow_final_layer(xp, y, 512, 32, P + ".final_layer");
    release(xp); release(y);
    return out;
}

cl_mem GpuOps::timestep_embedder(float t_val, const std::string& prefix) {
    // freqs[128] (buffer). args = t*freqs → emb = cat(cos(args), sin(args)) [256].
    std::vector<float> freqs = weights_.get_host_vec(prefix + ".freqs");
    if (freqs.empty()) { NNOPT_ERROR_FMT("timestep: missing '%s.freqs'", prefix.c_str()); return nullptr; }
    int half = (int)freqs.size();   // 128
    std::vector<float> emb(2 * half);
    for (int j = 0; j < half; ++j) {
        float a = t_val * freqs[j];
        emb[j] = std::cos(a); emb[half + j] = std::sin(a);
    }
    cl_mem x = upload(emb);                                   // [256]
    cl_mem h1 = linear(x, 1, 512, 2 * half, prefix + ".mlp.0.weight", prefix + ".mlp.0.bias");
    cl_mem h1s = silu(h1, 512);
    cl_mem h2 = linear(h1s, 1, 512, 512, prefix + ".mlp.2.weight", prefix + ".mlp.2.bias");
    cl_mem out = rmsnorm(h2, 1, 512, 1e-5f, prefix + ".mlp.3.alpha");
    release(x); release(h1); release(h1s); release(h2);
    return out;
}

// ── embedding ──────────────────────────────────────────────────────────────
cl_mem GpuOps::embedding(const std::vector<int>& ids, int dim, const std::string& w_key) {
    cl_mem table = weights_.get_buffer(w_key);
    if (!table) { NNOPT_ERROR_FMT("embedding: missing '%s'", w_key.c_str()); return nullptr; }
    cl_int err = CL_SUCCESS;
    cl_mem idbuf = clCreateBuffer(ctx_.context(), CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR,
                                  ids.size()*sizeof(int), (void*)ids.data(), &err);
    if (err != CL_SUCCESS) { NNOPT_ERROR_FMT("embedding: idbuf alloc (%d)", err); return nullptr; }
    int n_tok = (int)ids.size();
    cl_mem out = alloc((size_t)n_tok*dim);
    if (out) run1d("embedding_gather", (size_t)n_tok*dim,
                   {{sizeof(cl_mem),&table},{sizeof(cl_mem),&idbuf},{sizeof(cl_mem),&out},
                    {sizeof(int),&n_tok},{sizeof(int),&dim}});
    clReleaseMemObject(idbuf);
    return out;
}
