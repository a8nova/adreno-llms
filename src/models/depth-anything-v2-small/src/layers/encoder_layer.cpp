// One DINOv2 encoder layer (Dinov2Layer): LN1 -> attention (per-head
// CLBlast Gemm on token-major slices by default) -> layer_scale1 -> residual
// -> LN2 -> MLP(gelu) -> layer_scale2 -> residual.
// Code moved verbatim from the former monolithic backbone.cpp.

#include "depth_common.h"
#include <clblast.h>

// ── one DINOv2 encoder layer (Dinov2Layer), in place on token matrix ──
// h_n = LN1(h); a = attn(h_n); a *= lambda1; h = a + h
// o_n = LN2(h); m = mlp_gelu(o_n); m *= lambda2; out = m + h
cl_mem encoder_layer(Weights& W, cl_mem h, int M, int layer_idx) {
    const int D = MODEL_CONFIG::HIDDEN_SIZE;      // 384
    const int Hh = MODEL_CONFIG::NUM_ATTENTION_HEADS; // 6
    const int hd = MODEL_CONFIG::HEAD_DIM;        // 64
    const int I = MODEL_CONFIG::INTERMEDIATE_SIZE; // 1536
    const float scale = 1.0f / sqrtf((float)hd);
    std::string p = "backbone.encoder.layer." + std::to_string(layer_idx) + ".";

    // LN1
    cl_mem n1 = layernorm(h, M, D, W.get_buffer(p+"norm1.weight"), W.get_buffer(p+"norm1.bias"));
    layer_dump(layer_idx, "norm1", n1, (size_t)M*D);

    // Q/K/V projections -> token-major [M, D]
    cl_mem q = linear(n1, M, D, D, W.get_buffer(p+"attention.attention.query.weight"), W.get_buffer(p+"attention.attention.query.bias"));
    cl_mem k = linear(n1, M, D, D, W.get_buffer(p+"attention.attention.key.weight"),   W.get_buffer(p+"attention.attention.key.bias"));
    cl_mem v = linear(n1, M, D, D, W.get_buffer(p+"attention.attention.value.weight"), W.get_buffer(p+"attention.attention.value.bias"));
    pool_release(n1);

    // ── Attention layout strategy (campaign lever 5a, 2026-07-07).
    // HEADLOOP=1: per-head PLAIN Gemm calls read head slices straight out of
    // the token-major [M,D] projections (a_offset=h*hd, a_ld=D) and write
    // context token-major (c_offset=h*hd, c_ld=D) — no to_head_major copies
    // (36 calls, 206 ms/frame) and no batched-kernel variant (whose params
    // resisted both tuning attempts). A/B: NNOPT_ATTN_HEADLOOP=0 reverts to
    // GemmStridedBatched on head-major copies.
    static int use_headloop = -1;
    if (use_headloop < 0) {
        const char* e = std::getenv("NNOPT_ATTN_HEADLOOP");
        use_headloop = (e && e[0] == '0') ? 0 : 1;
        // The naive (NNOPT_ATTN_GEMM=0) kernels need head-major buffers.
        const char* ag = std::getenv("NNOPT_ATTN_GEMM");
        if (ag && ag[0] == '0') use_headloop = 0;
    }

    // to head-major [H, M, hd] — only for the batched / naive paths.
    cl_mem qh = nullptr, kh = nullptr, vh = nullptr;
    if (!use_headloop) {
        qh = alloc_buf((size_t)M*D); kh = alloc_buf((size_t)M*D); vh = alloc_buf((size_t)M*D);
        for (int pass = 0; pass < 3; pass++) {
            cl_mem in = pass==0?q:(pass==1?k:v);
            cl_mem out = pass==0?qh:(pass==1?kh:vh);
            cl_kernel kk=g_k.to_head_major; int a=0;
            sa(kk,a++,sizeof(cl_mem),&in); sa(kk,a++,sizeof(cl_mem),&out);
            sa(kk,a++,sizeof(int),&M); sa(kk,a++,sizeof(int),&Hh); sa(kk,a++,sizeof(int),&hd);
            run1d(kk, (size_t)M*D);
        }
        pool_release(q); pool_release(k); pool_release(v);
    }

    // ── Attention scores + context.
    // Default: CLBlast GemmStridedBatched — Q·Kᵀ and P·V are pure GEMMs at
    // M=1814, and the naive kernels were 77% of the whole frame (profile
    // 2026-07-04: attn_scores 18.2s + attn_context 7.4s + softmax 3.2s of
    // 37.3s GPU). CLBlast's own linears run in ≤2.6s total in this binary.
    // A/B: NNOPT_ATTN_GEMM=0 reverts to the original kernels (same build).
    static int use_attn_gemm = -1;
    if (use_attn_gemm < 0) {
        const char* e = std::getenv("NNOPT_ATTN_GEMM");
        use_attn_gemm = (e && e[0] == '0') ? 0 : 1;
    }
    cl_mem scores = alloc_buf((size_t)Hh*M*M);
    if (use_attn_gemm && use_headloop) {
        SyncScope _sync("clblast_attn_scores");
        // scores_h[M,M] = scale · Q[:,h·hd:+hd] · K[:,h·hd:+hd]ᵀ — plain Gemm
        // per head, reading token-major slices in place (offset h·hd, ld D).
        for (int hh = 0; hh < Hh; hh++) {
            const size_t off = (size_t)hh * hd;
#ifdef NNOPT_USE_FP16
            cl_half a_scale = static_cast<cl_half>(nnopt_f32_to_f16(scale));
            cl_half h_zero  = static_cast<cl_half>(nnopt_f32_to_f16(0.0f));
            auto st = clblast::Gemm<cl_half>(
                clblast::Layout::kRowMajor, clblast::Transpose::kNo, clblast::Transpose::kYes,
                M, M, hd, a_scale,
                q, off, D, k, off, D, h_zero,
                scores, (size_t)hh*M*M, M,
                &g_q, KernelProfiler::event_for("clblast_attn_scores"));
#else
            auto st = clblast::Gemm<float>(
                clblast::Layout::kRowMajor, clblast::Transpose::kNo, clblast::Transpose::kYes,
                M, M, hd, scale,
                q, off, D, k, off, D, 0.0f,
                scores, (size_t)hh*M*M, M,
                &g_q, KernelProfiler::event_for("clblast_attn_scores"));
#endif
            if (st != clblast::StatusCode::kSuccess) {
                NNOPT_ERROR_FMT("clblast attn scores (head %d) failed status=%d", hh, (int)st);
            }
        }
    } else if (use_attn_gemm) {
        SyncScope _sync("clblast_attn_scores");
        // scores_h[M,M] = scale · Q_h[M,hd] · K_h[M,hd]ᵀ, batched over heads.
#ifdef NNOPT_USE_FP16
        cl_half a_scale = static_cast<cl_half>(nnopt_f32_to_f16(scale));
        cl_half h_zero  = static_cast<cl_half>(nnopt_f32_to_f16(0.0f));
        auto st = clblast::GemmStridedBatched<cl_half>(
            clblast::Layout::kRowMajor, clblast::Transpose::kNo, clblast::Transpose::kYes,
            M, M, hd, a_scale,
            qh, 0, hd, (size_t)M*hd,
            kh, 0, hd, (size_t)M*hd,
            h_zero,
            scores, 0, M, (size_t)M*M,
            Hh, &g_q, KernelProfiler::event_for("clblast_attn_scores"));
#else
        auto st = clblast::GemmStridedBatched<float>(
            clblast::Layout::kRowMajor, clblast::Transpose::kNo, clblast::Transpose::kYes,
            M, M, hd, scale,
            qh, 0, hd, (size_t)M*hd,
            kh, 0, hd, (size_t)M*hd,
            0.0f,
            scores, 0, M, (size_t)M*M,
            Hh, &g_q, KernelProfiler::event_for("clblast_attn_scores"));
#endif
        if (st != clblast::StatusCode::kSuccess) {
            NNOPT_ERROR_FMT("clblast attn scores failed status=%d", (int)st);
        }
    } else {
        cl_kernel kk=g_k.attn_scores; int a=0;
        sa(kk,a++,sizeof(cl_mem),&qh); sa(kk,a++,sizeof(cl_mem),&kh); sa(kk,a++,sizeof(cl_mem),&scores);
        sa(kk,a++,sizeof(int),&Hh); sa(kk,a++,sizeof(int),&M); sa(kk,a++,sizeof(int),&hd); sa(kk,a++,sizeof(float),&scale);
        run1d(kk, (size_t)Hh*M*M);
    }
    if (use_headloop) { pool_release(q); pool_release(k); }
    else { pool_release(qh); pool_release(kh); }

    // softmax over last dim — workgroup-per-row cooperative version by
    // default (1-WI-per-row was 271 ms/call); NNOPT_SOFTMAX_WG=0 reverts.
    static int use_sm_wg = -1;
    if (use_sm_wg < 0) {
        const char* e = std::getenv("NNOPT_SOFTMAX_WG");
        use_sm_wg = (e && e[0] == '0') ? 0 : 1;
    }
    if (use_sm_wg) {
        cl_kernel kk = g_k.attn_softmax_wg; int a=0;
        sa(kk,a++,sizeof(cl_mem),&scores); sa(kk,a++,sizeof(int),&M);
        size_t gws = (size_t)Hh * M * 128, lws = 128;
        cl_event* evt = KernelProfiler::enabled()
            ? KernelProfiler::event_for("attn_softmax_wg") : nullptr;
        cl_int err = clEnqueueNDRangeKernel(g_q, kk, 1, nullptr, &gws, &lws, 0, nullptr, evt);
        if (err != CL_SUCCESS) NNOPT_ERROR_FMT("softmax_wg dispatch err=%d", err);
        NNOPT_DEBUG_SYNC(g_q);
    } else {
        cl_kernel kk=g_k.attn_softmax; int a=0;
        sa(kk,a++,sizeof(cl_mem),&scores); sa(kk,a++,sizeof(int),&Hh); sa(kk,a++,sizeof(int),&M);
        run1d(kk, (size_t)Hh*M);
    }

    // context -> token-major [M, D]
    cl_mem ctx = alloc_buf((size_t)M*D);
    if (use_attn_gemm && use_headloop) {
        SyncScope _sync("clblast_attn_context");
        // ctx[:, h·hd:+hd] = P_h[M,M] · V[:, h·hd:+hd] — plain Gemm per head;
        // B and C are token-major slices in place (offset h·hd, ld D).
        for (int hh = 0; hh < Hh; hh++) {
            const size_t off = (size_t)hh * hd;
#ifdef NNOPT_USE_FP16
            cl_half h_one  = static_cast<cl_half>(nnopt_f32_to_f16(1.0f));
            cl_half h_zero = static_cast<cl_half>(nnopt_f32_to_f16(0.0f));
            auto st = clblast::Gemm<cl_half>(
                clblast::Layout::kRowMajor, clblast::Transpose::kNo, clblast::Transpose::kNo,
                M, hd, M, h_one,
                scores, (size_t)hh*M*M, M,
                v, off, D, h_zero,
                ctx, off, D,
                &g_q, KernelProfiler::event_for("clblast_attn_context"));
#else
            auto st = clblast::Gemm<float>(
                clblast::Layout::kRowMajor, clblast::Transpose::kNo, clblast::Transpose::kNo,
                M, hd, M, 1.0f,
                scores, (size_t)hh*M*M, M,
                v, off, D, 0.0f,
                ctx, off, D,
                &g_q, KernelProfiler::event_for("clblast_attn_context"));
#endif
            if (st != clblast::StatusCode::kSuccess) {
                NNOPT_ERROR_FMT("clblast attn context (head %d) failed status=%d", hh, (int)st);
            }
        }
    } else if (use_attn_gemm) {
        SyncScope _sync("clblast_attn_context");
        // ctx[M, h*hd..h*hd+hd) = P_h[M,M] · V_h[M,hd] — the strided batch
        // writes each head's [M,hd] block straight into token-major columns
        // (c_ld = D, c_stride = hd), so no head→token transpose kernel runs.
#ifdef NNOPT_USE_FP16
        cl_half h_one  = static_cast<cl_half>(nnopt_f32_to_f16(1.0f));
        cl_half h_zero = static_cast<cl_half>(nnopt_f32_to_f16(0.0f));
        auto st = clblast::GemmStridedBatched<cl_half>(
            clblast::Layout::kRowMajor, clblast::Transpose::kNo, clblast::Transpose::kNo,
            M, hd, M, h_one,
            scores, 0, M, (size_t)M*M,
            vh, 0, hd, (size_t)M*hd,
            h_zero,
            ctx, 0, D, (size_t)hd,
            Hh, &g_q, KernelProfiler::event_for("clblast_attn_context"));
#else
        auto st = clblast::GemmStridedBatched<float>(
            clblast::Layout::kRowMajor, clblast::Transpose::kNo, clblast::Transpose::kNo,
            M, hd, M, 1.0f,
            scores, 0, M, (size_t)M*M,
            vh, 0, hd, (size_t)M*hd,
            0.0f,
            ctx, 0, D, (size_t)hd,
            Hh, &g_q, KernelProfiler::event_for("clblast_attn_context"));
#endif
        if (st != clblast::StatusCode::kSuccess) {
            NNOPT_ERROR_FMT("clblast attn context failed status=%d", (int)st);
        }
    } else {
        cl_kernel kk=g_k.attn_context; int a=0;
        sa(kk,a++,sizeof(cl_mem),&scores); sa(kk,a++,sizeof(cl_mem),&vh); sa(kk,a++,sizeof(cl_mem),&ctx);
        sa(kk,a++,sizeof(int),&Hh); sa(kk,a++,sizeof(int),&M); sa(kk,a++,sizeof(int),&hd);
        run1d(kk, (size_t)M*D);
    }
    pool_release(scores);
    if (use_headloop) pool_release(v); else pool_release(vh);

    // out_proj (attention.output.dense) — this IS Dinov2Attention.forward output
    cl_mem attn = linear(ctx, M, D, D, W.get_buffer(p+"attention.output.dense.weight"), W.get_buffer(p+"attention.output.dense.bias"));
    pool_release(ctx);
    layer_dump(layer_idx, "attention", attn, (size_t)M*D);

    // layer_scale1 + residual
    cl_mem a_scaled = layer_scale(attn, W.get_buffer(p+"layer_scale1.lambda1"), M, D);
    pool_release(attn);
    layer_dump(layer_idx, "layer_scale1", a_scaled, (size_t)M*D);
    cl_mem h1 = elt_add(a_scaled, h, (size_t)M*D);
    pool_release(a_scaled);

    // LN2 -> MLP (fc1 gelu fc2) -> layer_scale2 -> residual
    cl_mem n2 = layernorm(h1, M, D, W.get_buffer(p+"norm2.weight"), W.get_buffer(p+"norm2.bias"));
    layer_dump(layer_idx, "norm2", n2, (size_t)M*D);
    cl_mem f1 = linear(n2, M, I, D, W.get_buffer(p+"mlp.fc1.weight"), W.get_buffer(p+"mlp.fc1.bias"));
    pool_release(n2);
    cl_mem g = act_gelu(f1, (size_t)M*I);
    pool_release(f1);
    cl_mem f2 = linear(g, M, D, I, W.get_buffer(p+"mlp.fc2.weight"), W.get_buffer(p+"mlp.fc2.bias"));
    pool_release(g);
    layer_dump(layer_idx, "mlp", f2, (size_t)M*D);
    cl_mem m_scaled = layer_scale(f2, W.get_buffer(p+"layer_scale2.lambda1"), M, D);
    pool_release(f2);
    layer_dump(layer_idx, "layer_scale2", m_scaled, (size_t)M*D);
    // drop_path is nn.Identity (drop_path_rate==0), called TWICE per Dinov2Layer.
    // The reference module-hook on nn.Identity records only its LAST invocation,
    // which is drop_path(layer_output) applied to the layer_scale2 output. Verified
    // numerically: reference drop_path_output == layer_scale2_output (cos=1.0),
    // whereas the first-call value (layer_scale1 output) gives cos=-0.707. Dump the
    // second-call value (m_scaled) so SxS pairs the right tensor. Identity => value
    // unchanged; this is a dump-label fix, not a math change.
    layer_dump(layer_idx, "drop_path", m_scaled, (size_t)M*D);
    cl_mem out = elt_add(m_scaled, h1, (size_t)M*D);
    pool_release(m_scaled);
    pool_release(h1);
    // NOTE: the bare "backbone_encoder_layer_<L>" boundary is dumped by the
    // caller loop (with the correct un-suffixed name); do not dump here.
    return out;
}
