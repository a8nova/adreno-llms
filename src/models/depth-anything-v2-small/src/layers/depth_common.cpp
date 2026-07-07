// Shared plumbing for the Depth-Anything-V2 pipeline: buffer pool, dispatch
// helpers, CLBlast sync-profiler, op wrappers, kernel-program registry.
// Code moved verbatim from the former monolithic backbone.cpp.

#include "depth_common.h"
#include <clblast.h>

cl_context g_ctx = nullptr;
cl_command_queue g_q = nullptr;
DepthKernels g_k;

// ── Activation buffer pool (campaign 2026-07-05). ~80 create/release cycles
// per frame (some >100MB) cost seconds of driver time and fragment the heap
// (stable-audio: fragmentation caused intermittent CL_OUT_OF_RESOURCES —
// pooling is correctness-critical, not hygiene). Exact-size free-list:
// activations repeat shapes across the 12 encoder layers, so hit rate is
// high after layer 0. pool_release() returns a buffer to the pool; real
// clReleaseMemObject only happens at process exit (OS reclaims).
static std::multimap<size_t, cl_mem> g_buf_pool;      // elems -> free buffer
static std::map<cl_mem, size_t> g_buf_sizes;          // live buffer -> elems

cl_mem alloc_buf(size_t n) {
    auto it = g_buf_pool.find(n);
    if (it != g_buf_pool.end()) {
        cl_mem b = it->second;
        g_buf_pool.erase(it);
        return b;
    }
    cl_int err;
    cl_mem b = clCreateBuffer(g_ctx, CL_MEM_READ_WRITE, n * sizeof(nnopt_storage_t), nullptr, &err);
    if (err != CL_SUCCESS) { NNOPT_ERROR_FMT("alloc_buf(%zu) err=%d", n, err); return nullptr; }
    g_buf_sizes[b] = n;
    return b;
}

void pool_release(cl_mem b) {
    if (!b) return;
    auto it = g_buf_sizes.find(b);
    if (it == g_buf_sizes.end()) { clReleaseMemObject(b); return; } // not pooled
    g_buf_pool.insert({it->second, b});
}

// Upload a host fp32 vector to a fresh storage_t buffer.
cl_mem upload_f32(const std::vector<float>& v) {
    std::vector<nnopt_storage_t> s(v.size());
    for (size_t i = 0; i < v.size(); i++) s[i] = enc(v[i]);
    cl_int err;
    cl_mem b = clCreateBuffer(g_ctx, CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR,
                              s.size() * sizeof(nnopt_storage_t), s.data(), &err);
    if (err != CL_SUCCESS) { NNOPT_ERROR_FMT("upload_f32 err=%d", err); return nullptr; }
    return b;
}

// Read a storage_t buffer back to host fp32.
std::vector<float> download_f32(cl_mem b, size_t n) {
    std::vector<nnopt_storage_t> s(n);
    clEnqueueReadBuffer(g_q, b, CL_TRUE, 0, n * sizeof(nnopt_storage_t), s.data(), 0, nullptr, nullptr);
    std::vector<float> out(n);
    for (size_t i = 0; i < n; i++) out[i] = dec(s[i]);
    return out;
}

void run1d(cl_kernel k, size_t total) {
    size_t gws = ((total + 63) / 64) * 64;
    size_t lws = 64;
    // Profiler wiring (NNOPT_PROFILE=1): label events by the kernel's own
    // function name so every op dispatching through this helper lands in
    // the per-kernel ledger without per-callsite labels.
    cl_event* evt = nullptr;
    if (KernelProfiler::enabled()) {
        static char kname[128];
        kname[0] = '\0';
        clGetKernelInfo(k, CL_KERNEL_FUNCTION_NAME, sizeof(kname), kname, nullptr);
        evt = KernelProfiler::event_for(kname[0] ? kname : "unnamed_kernel");
    }
    cl_int err = clEnqueueNDRangeKernel(g_q, k, 1, nullptr, &gws, &lws, 0, nullptr, evt);
    if (err != CL_SUCCESS) NNOPT_ERROR_FMT("dispatch err=%d", err);
    NNOPT_DEBUG_SYNC(g_q);
}

// set-arg helper.
bool sa(cl_kernel k, int idx, size_t sz, const void* p) {
    cl_int err = clSetKernelArg(k, idx, sz, p);
    if (err != CL_SUCCESS) { NNOPT_ERROR_FMT("setarg %d err=%d", idx, err); return false; }
    return true;
}

// Persistent grow-only im2col scratch (per stable-audio: per-conv alloc/free
// fragments the heap and intermittently OOMs), chunked so the col matrix
// never exceeds COL_CAP elements.
static cl_mem g_col_scratch = nullptr;
static size_t g_col_scratch_elems = 0;
const size_t COL_CAP_ELEMS = 32u * 1024u * 1024u; // 64MB fp16 / 128MB fp32

cl_mem col_scratch(size_t elems) {
    if (elems <= g_col_scratch_elems && g_col_scratch) return g_col_scratch;
    if (g_col_scratch) pool_release(g_col_scratch);  // pooled — keep size map consistent
    g_col_scratch = alloc_buf(elems);
    g_col_scratch_elems = g_col_scratch ? elems : 0;
    return g_col_scratch;
}

// ── Sync-profiler state (see depth_common.h for the rationale) ──
std::map<std::string, double> g_sync_ms;
std::map<std::string, int>    g_sync_calls;
bool sync_prof_on() {
    static int v = -1;
    if (v < 0) { const char* e = std::getenv("NNOPT_SYNC_PROF"); v = (e && e[0]=='1') ? 1 : 0; }
    return v == 1;
}
void sync_prof_dump() {
    if (!sync_prof_on() || g_sync_ms.empty()) return;
    fprintf(stderr, "\n=== CLBLAST SYNC PROFILE (NNOPT_SYNC_PROF=1, clFinish-bracketed) ===\n");
    double total = 0.0;
    for (const auto& kv : g_sync_ms) {
        fprintf(stderr, "%-28s %10.1f ms  %5d calls\n", kv.first.c_str(), kv.second, g_sync_calls[kv.first]);
        total += kv.second;
    }
    fprintf(stderr, "=== TOTAL CLBlast composite: %.1f ms ===\n", total);
    g_sync_ms.clear(); g_sync_calls.clear();
}

bool gemm_nn(int M, int N, int K, cl_mem A, size_t a_off, int lda,
             cl_mem B, size_t b_off, int ldb,
             cl_mem C, size_t c_off, int ldc, const char* label) {
    SyncScope _sync(label);
#ifdef NNOPT_USE_FP16
    cl_half one  = static_cast<cl_half>(nnopt_f32_to_f16(1.0f));
    cl_half zero = static_cast<cl_half>(nnopt_f32_to_f16(0.0f));
    auto st = clblast::Gemm<cl_half>(
        clblast::Layout::kRowMajor, clblast::Transpose::kNo, clblast::Transpose::kNo,
        M, N, K, one, A, a_off, lda, B, b_off, ldb, zero, C, c_off, ldc,
        &g_q, KernelProfiler::event_for(label));
#else
    auto st = clblast::Gemm<float>(
        clblast::Layout::kRowMajor, clblast::Transpose::kNo, clblast::Transpose::kNo,
        M, N, K, 1.0f, A, a_off, lda, B, b_off, ldb, 0.0f, C, c_off, ldc,
        &g_q, KernelProfiler::event_for(label));
#endif
    if (st != clblast::StatusCode::kSuccess) {
        NNOPT_ERROR_FMT("%s: clblast gemm failed status=%d", label, (int)st);
        return false;
    }
    return true;
}

// ── conv2d wrapper. weight [Cout,Cin,KH,KW], bias [Cout] or nullptr. ──
//
// Default path: conv-as-GEMM via CLBlast (campaign 2026-07-05: the direct
// kernel was 58% of GPU time at ~2 GFLOPS; CLBlast GEMM is the proven fast
// path in this binary). 1×1 s1 p0 convs skip im2col entirely (the input IS
// the col matrix); K×K convs go through im2col into the persistent scratch.
// A/B: NNOPT_CONV_GEMM=0 reverts to the direct kernel.
cl_mem conv2d(cl_mem in, int Cin, int Hin, int Win,
              cl_mem w, cl_mem bias, int Cout, int KH, int KW,
              int stride, int pad, int& Hout, int& Wout) {
    Hout = (Hin + 2*pad - KH) / stride + 1;
    Wout = (Win + 2*pad - KW) / stride + 1;
    cl_mem out = alloc_buf((size_t)Cout*Hout*Wout);
    static int use_gemm = -1;
    if (use_gemm < 0) {
        const char* e = std::getenv("NNOPT_CONV_GEMM");
        use_gemm = (e && e[0] == '0') ? 0 : 1;
    }
    const size_t HW = (size_t)Hout * Wout;
    const int Kdim = Cin * KH * KW;
    if (use_gemm) {
        bool ok = true;
        if (KH == 1 && KW == 1 && stride == 1 && pad == 0) {
            // 1×1: out[Cout,HW] = W[Cout,Cin] · in[Cin,HW] — no im2col.
            ok = gemm_nn(Cout, (int)HW, Cin, w, 0, Cin, in, 0, (int)HW,
                         out, 0, (int)HW, "clblast_conv1x1");
        } else {
            // K×K: chunk output positions so col[Kdim, chunk] fits the cap.
            size_t chunk = COL_CAP_ELEMS / (size_t)Kdim;
            if (chunk < 1) chunk = 1;
            if (chunk > HW) chunk = HW;
            cl_mem col = col_scratch((size_t)Kdim * chunk);
            // 4-wide im2col by default (vload4 fast path on interior taps);
            // A/B: NNOPT_IM2COL_V4=0 reverts to the scalar kernel.
            static int use_v4 = -1;
            if (use_v4 < 0) {
                const char* e = std::getenv("NNOPT_IM2COL_V4");
                use_v4 = (e && e[0] == '0') ? 0 : 1;
            }
            for (size_t n0 = 0; n0 < HW && ok; n0 += chunk) {
                int ncols = (int)((n0 + chunk <= HW) ? chunk : (HW - n0));
                int in0 = (int)n0;
                cl_kernel k = use_v4 ? g_k.im2col2d_v4 : g_k.im2col2d; int a=0;
                sa(k,a++,sizeof(cl_mem),&in); sa(k,a++,sizeof(cl_mem),&col);
                sa(k,a++,sizeof(int),&Cin); sa(k,a++,sizeof(int),&Hin); sa(k,a++,sizeof(int),&Win);
                sa(k,a++,sizeof(int),&KH); sa(k,a++,sizeof(int),&KW);
                sa(k,a++,sizeof(int),&stride); sa(k,a++,sizeof(int),&pad);
                sa(k,a++,sizeof(int),&Wout); sa(k,a++,sizeof(int),&in0); sa(k,a++,sizeof(int),&ncols);
                run1d(k, use_v4 ? (size_t)Kdim * (((size_t)ncols + 3) / 4)
                                : (size_t)Kdim * ncols);
                ok = gemm_nn(Cout, ncols, Kdim, w, 0, Kdim, col, 0, ncols,
                             out, n0, (int)HW, "clblast_conv_gemm");
            }
        }
        if (ok && bias) {
            int total = (int)((size_t)Cout * HW), hw = (int)HW;
            cl_kernel k = g_k.bias_per_row; int a=0;
            sa(k,a++,sizeof(cl_mem),&out); sa(k,a++,sizeof(cl_mem),&bias);
            sa(k,a++,sizeof(int),&hw); sa(k,a++,sizeof(int),&total);
            run1d(k, (size_t)total);
        }
        if (ok) return out;
        // GEMM path failed — fall through to the direct kernel (loud, correct).
        NNOPT_ERROR_FMT("conv2d GEMM path failed (Cout=%d K=%d HW=%zu) — falling back to direct kernel", Cout, Kdim, HW);
    }
    int has_bias = bias ? 1 : 0;
    cl_mem bb = bias ? bias : in; // dummy non-null when unused
    cl_kernel k = g_k.conv2d; int a=0;
    sa(k,a++,sizeof(cl_mem),&in); sa(k,a++,sizeof(cl_mem),&w); sa(k,a++,sizeof(cl_mem),&bb);
    sa(k,a++,sizeof(cl_mem),&out);
    sa(k,a++,sizeof(int),&Cin); sa(k,a++,sizeof(int),&Hin); sa(k,a++,sizeof(int),&Win);
    sa(k,a++,sizeof(int),&Cout); sa(k,a++,sizeof(int),&KH); sa(k,a++,sizeof(int),&KW);
    sa(k,a++,sizeof(int),&stride); sa(k,a++,sizeof(int),&pad);
    sa(k,a++,sizeof(int),&Hout); sa(k,a++,sizeof(int),&Wout); sa(k,a++,sizeof(int),&has_bias);
    run1d(k, (size_t)Cout*Hout*Wout);
    return out;
}

// ── conv_transpose2d wrapper. weight [Cin,Cout,KH,KW], stride==kernel. ──
cl_mem conv_transpose2d(cl_mem in, int Cin, int Hin, int Win,
                        cl_mem w, cl_mem bias, int Cout, int KH, int KW,
                        int stride, int& Hout, int& Wout) {
    Hout = (Hin-1)*stride + KH;
    Wout = (Win-1)*stride + KW;
    cl_mem out = alloc_buf((size_t)Cout*Hout*Wout);
    int has_bias = bias ? 1 : 0;
    cl_mem bb = bias ? bias : in;
    // stride==kernel specialization: one weight tap per output, no window
    // loop (16× fewer inner iterations at K=4). A/B: NNOPT_CONVT_SK=0.
    static int use_sk = -1;
    if (use_sk < 0) {
        const char* e = std::getenv("NNOPT_CONVT_SK");
        use_sk = (e && e[0] == '0') ? 0 : 1;
    }
    if (use_sk && stride == KH && KH == KW) {
        cl_kernel k = g_k.conv_transpose2d_sk; int a=0;
        sa(k,a++,sizeof(cl_mem),&in); sa(k,a++,sizeof(cl_mem),&w); sa(k,a++,sizeof(cl_mem),&bb);
        sa(k,a++,sizeof(cl_mem),&out);
        sa(k,a++,sizeof(int),&Cin); sa(k,a++,sizeof(int),&Hin); sa(k,a++,sizeof(int),&Win);
        sa(k,a++,sizeof(int),&Cout); sa(k,a++,sizeof(int),&stride);
        sa(k,a++,sizeof(int),&Hout); sa(k,a++,sizeof(int),&Wout); sa(k,a++,sizeof(int),&has_bias);
        size_t wq = ((size_t)Wout + 3) / 4;
        run1d(k, (size_t)Cout*Hout*wq);
        return out;
    }
    cl_kernel k = g_k.conv_transpose2d; int a=0;
    sa(k,a++,sizeof(cl_mem),&in); sa(k,a++,sizeof(cl_mem),&w); sa(k,a++,sizeof(cl_mem),&bb);
    sa(k,a++,sizeof(cl_mem),&out);
    sa(k,a++,sizeof(int),&Cin); sa(k,a++,sizeof(int),&Hin); sa(k,a++,sizeof(int),&Win);
    sa(k,a++,sizeof(int),&Cout); sa(k,a++,sizeof(int),&KH); sa(k,a++,sizeof(int),&KW);
    sa(k,a++,sizeof(int),&stride);
    sa(k,a++,sizeof(int),&Hout); sa(k,a++,sizeof(int),&Wout); sa(k,a++,sizeof(int),&has_bias);
    run1d(k, (size_t)Cout*Hout*Wout);
    return out;
}

// ── bilinear interpolate wrapper ──
cl_mem bilinear(cl_mem in, int Cc, int Hin, int Win, int Hout, int Wout, int align) {
    cl_mem out = alloc_buf((size_t)Cc*Hout*Wout);
    cl_kernel k = g_k.bilinear_interp; int a=0;
    sa(k,a++,sizeof(cl_mem),&in); sa(k,a++,sizeof(cl_mem),&out);
    sa(k,a++,sizeof(int),&Cc); sa(k,a++,sizeof(int),&Hin); sa(k,a++,sizeof(int),&Win);
    sa(k,a++,sizeof(int),&Hout); sa(k,a++,sizeof(int),&Wout); sa(k,a++,sizeof(int),&align);
    run1d(k, (size_t)Cc*Hout*Wout);
    return out;
}

// ── layernorm over last dim wrapper ──
cl_mem layernorm(cl_mem in, int rows, int D, cl_mem gamma, cl_mem beta) {
    cl_mem out = alloc_buf((size_t)rows*D);
    float eps = MODEL_CONFIG::LAYER_NORM_EPS;
    cl_kernel k = g_k.layernorm_rows; int a=0;
    sa(k,a++,sizeof(cl_mem),&in); sa(k,a++,sizeof(cl_mem),&gamma); sa(k,a++,sizeof(cl_mem),&beta);
    sa(k,a++,sizeof(cl_mem),&out);
    sa(k,a++,sizeof(int),&rows); sa(k,a++,sizeof(int),&D); sa(k,a++,sizeof(float),&eps);
    run1d(k, (size_t)rows);
    return out;
}

cl_mem act_gelu(cl_mem in, size_t n) {
    cl_mem out = alloc_buf(n); int nn=(int)n; cl_kernel k=g_k.gelu; int a=0;
    sa(k,a++,sizeof(cl_mem),&in); sa(k,a++,sizeof(cl_mem),&out); sa(k,a++,sizeof(int),&nn);
    run1d(k, n); return out;
}
cl_mem act_relu(cl_mem in, size_t n) {
    cl_mem out = alloc_buf(n); int nn=(int)n; cl_kernel k=g_k.relu; int a=0;
    sa(k,a++,sizeof(cl_mem),&in); sa(k,a++,sizeof(cl_mem),&out); sa(k,a++,sizeof(int),&nn);
    run1d(k, n); return out;
}
cl_mem elt_add(cl_mem x, cl_mem y, size_t n) {
    cl_mem out = alloc_buf(n); int nn=(int)n; cl_kernel k=g_k.add; int a=0;
    sa(k,a++,sizeof(cl_mem),&x); sa(k,a++,sizeof(cl_mem),&y); sa(k,a++,sizeof(cl_mem),&out); sa(k,a++,sizeof(int),&nn);
    run1d(k, n); return out;
}
cl_mem layer_scale(cl_mem in, cl_mem lambda, int rows, int D) {
    cl_mem out = alloc_buf((size_t)rows*D); cl_kernel k=g_k.layer_scale; int a=0;
    sa(k,a++,sizeof(cl_mem),&in); sa(k,a++,sizeof(cl_mem),&lambda); sa(k,a++,sizeof(cl_mem),&out);
    sa(k,a++,sizeof(int),&rows); sa(k,a++,sizeof(int),&D);
    run1d(k, (size_t)rows*D); return out;
}
// add bias per-row in place (buf modified). rows x D, bias[D].
void add_bias(cl_mem buf, cl_mem bias, int rows, int D) {
    cl_kernel k=g_k.add_bias_rows; int a=0;
    sa(k,a++,sizeof(cl_mem),&buf); sa(k,a++,sizeof(cl_mem),&bias);
    sa(k,a++,sizeof(int),&rows); sa(k,a++,sizeof(int),&D);
    run1d(k, (size_t)rows*D);
}
// nn.Linear: out[rows,N] = in[rows,K] @ W[N,K]^T + bias. W is [N,K] row-major.
cl_mem linear(cl_mem in, int rows, int N, int K, cl_mem w, cl_mem bias) {
    cl_mem out = alloc_buf((size_t)rows*N);
    if (!pytorch_linear(g_q, rows, N, K, in, w, out)) { NNOPT_ERROR("pytorch_linear failed"); }
    if (bias) add_bias(out, bias, rows, N);
    return out;
}

// ─────────────────────────────────────────────────────────────────────
// Build the depth kernels ONCE.
// ─────────────────────────────────────────────────────────────────────
bool build_depth_kernels(OpenCLContext& cl_ctx) {
    if (g_k.built) return true;
    // One cl_program per kernel FILE (adreno-llms convention). This is also
    // load-bearing for performance: Adreno applies a program-wide worst-case
    // register footprint, so a fat kernel in a shared program cuts occupancy
    // for every kernel in it (measured: frame 6.6s → 10.5s from two
    // never-dispatched kernels). Op families stay isolated per file.
    struct { cl_kernel* slot; const char* name; const char* file; } tbl[] = {
        {&g_k.conv2d,              "conv2d",              "kernels/conv2d.cl"},
        {&g_k.conv_transpose2d,    "conv_transpose2d",    "kernels/conv_transpose.cl"},
        {&g_k.conv_transpose2d_sk, "conv_transpose2d_sk", "kernels/conv_transpose.cl"},
        {&g_k.im2col2d,            "im2col2d",            "kernels/im2col.cl"},
        {&g_k.im2col2d_v4,         "im2col2d_v4",         "kernels/im2col.cl"},
        {&g_k.bias_per_row,        "bias_per_row",        "kernels/im2col.cl"},
        {&g_k.attn_scores,         "attn_scores",         "kernels/attention.cl"},
        {&g_k.attn_context,        "attn_context",        "kernels/attention.cl"},
        {&g_k.to_head_major,       "to_head_major",       "kernels/attention.cl"},
        {&g_k.attn_softmax,        "attn_softmax",        "kernels/softmax.cl"},
        {&g_k.attn_softmax_wg,     "attn_softmax_wg",     "kernels/softmax.cl"},
        {&g_k.layernorm_rows,      "layernorm_rows",      "kernels/layernorm.cl"},
        {&g_k.gelu,                "gelu",                "kernels/elementwise.cl"},
        {&g_k.relu,                "relu",                "kernels/elementwise.cl"},
        {&g_k.layer_scale,         "layer_scale",         "kernels/elementwise.cl"},
        {&g_k.add_bias_rows,       "add_bias_rows",       "kernels/elementwise.cl"},
        {&g_k.add,                 "add",                 "kernels/elementwise.cl"},
        {&g_k.embeddings_assemble, "embeddings_assemble", "kernels/layout.cl"},
        {&g_k.tokens_to_chw,       "tokens_to_chw",       "kernels/layout.cl"},
        {&g_k.chw_to_tokens,       "chw_to_tokens",       "kernels/layout.cl"},
        {&g_k.bilinear_interp,     "bilinear_interp",     "kernels/layout.cl"},
    };
    std::map<std::string, cl_program> progs;
    cl_int e;
    for (auto& t : tbl) {
        auto it = progs.find(t.file);
        if (it == progs.end()) {
            cl_program p = cl_ctx.build_program_from_file(t.file);
            if (!p) { NNOPT_ERROR_FMT("build %s failed", t.file); return false; }
            it = progs.emplace(t.file, p).first;
        }
        *t.slot = clCreateKernel(it->second, t.name, &e);
        if (e != CL_SUCCESS || !*t.slot) {
            NNOPT_ERROR_FMT("clCreateKernel(\"%s\") from %s failed err=%d", t.name, t.file, e);
            return false;
        }
    }
    g_k.built = true;
    return true;
}

// Emit a per-sub-module DCHECK dump with the encoder-layer-scoped name so the
// SxS/Evaluate gate can pair each Dinov2Layer sub-module (norm1, attention,
// layer_scale1, drop_path, norm2, mlp, layer_scale2) against its reference hook.
// Reference: modeling_dinov2.py Dinov2Layer.forward — every named intermediate
// there corresponds to a captured module boundary.
void layer_dump(int layer_idx, const char* sub, cl_mem buf, size_t n) {
    char name[96];
    snprintf(name, sizeof(name), "backbone_encoder_layer_%d_%s", layer_idx, sub);
    NNOPT_LAYER_CHECK(name, g_q, buf,
        (n > NNOPT_REF_DUMP_CAP ? NNOPT_REF_DUMP_CAP : n));
}
