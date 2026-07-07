// Depth-Anything-V2-Small pipeline orchestrator: DINOv2 backbone -> DPT neck
// -> depth head. Stage implementations live in src/layers/ (adreno-llms
// convention): depth_common.{h,cpp} (pool/dispatch/op wrappers),
// embeddings.cpp, encoder_layer.cpp, dpt.cpp.
// Reference: model_info/transformers_src/modeling_dinov2.py + modeling_depth_anything.py

#include "layers/depth_common.h"
#include <clblast.h>
#include <unordered_map>

std::vector<float> depth_forward(
    OpenCLContext& cl_ctx,
    Weights& weights,
    const std::vector<float>& pixel_values,
    int C, int H, int W)
{
    g_ctx = cl_ctx.context();
    g_q   = cl_ctx.queue();

    // ── CLBlast Xgemm parameter override (campaign lever 2, 2026-07-07).
    // VERDICT: tuning is a CLOSED dead end on this workload — params tuned at
    // 1024³ were +0.9s/frame, params tuned at the REAL attn shape
    // (1792×1792×64, 64.9 GFLOPS in the tuner harness) were still +0.6s
    // in-model. Isolated tuner throughput does not transfer. Default OFF;
    // NNOPT_XGEMM_TUNED=1 re-enables the real-shape set for future re-tests.
    // (Other measured dead ends: forcing XgemmDirect = 4.3× slower;
    // XGEMM_MIN_INDIRECT_SIZE is CUBED internally — 1<<30 overflows to 2^26.)
    static bool s_gemm_override_done = false;
    if (!s_gemm_override_done) {
        s_gemm_override_done = true;
        const char* gt = std::getenv("NNOPT_XGEMM_TUNED");
        if (gt && gt[0] == '1') {
#ifdef NNOPT_USE_FP16
            const auto prec = clblast::Precision::kHalf;
#else
            const auto prec = clblast::Precision::kSingle;
#endif
            const auto ov = std::unordered_map<std::string, size_t>{
                {"GEMMK", 0}, {"KREG", 1}, {"KWG", 16}, {"KWI", 2},
                {"MDIMA", 8}, {"MDIMC", 16}, {"MWG", 128},
                {"NDIMB", 16}, {"NDIMC", 8}, {"NWG", 64},
                {"SA", 0}, {"SB", 1}, {"STRM", 1}, {"STRN", 1},
                {"VWM", 4}, {"VWN", 4}};
            const auto st = clblast::OverrideParameters(cl_ctx.device(), "Xgemm", prec, ov);
            if (st != clblast::StatusCode::kSuccess) {
                NNOPT_ERROR_FMT("OverrideParameters(Xgemm tuned) failed status=%d (continuing with stock)", (int)st);
            }
        }
    }

    if (!build_depth_kernels(cl_ctx)) return {};

    Weights& Wt = weights;
    const int D   = MODEL_CONFIG::HIDDEN_SIZE;       // 384
    const int PS  = MODEL_CONFIG::PATCH_SIZE;        // 14
    const int Hp  = H / PS;                          // 37
    const int Wp  = W / PS;                          // 49
    const int Np  = Hp * Wp;                         // 1813
    const int M   = Np + 1;                          // 1814 tokens
    NNOPT_CHECKPOINT("depth_forward: geometry computed");

    // ── Embeddings: patch conv + bicubic pos-embed + cls → tokens [M, D] ──
    cl_mem tokens = embeddings_stage(Wt, pixel_values, C, H, W);

    // ── Encoder: 12 layers; capture features after layers 3,6,9,12 ──
    cl_mem feats[4] = {nullptr,nullptr,nullptr,nullptr};
    const int fidx[4] = {3,6,9,12}; // hidden_states index -> layer output (layer k-1)
    cl_mem h = tokens;
    for (int L = 0; L < MODEL_CONFIG::NUM_HIDDEN_LAYERS; L++) {
        cl_mem next = encoder_layer(Wt, h, M, L);
        pool_release(h);
        h = next;
        // (encoder_layer already dumped "backbone_encoder_layer_<L>_"; keep the
        //  bare "backbone_encoder_layer_<L>" name too for the SxS spec.)
        {
            char nm[64];
            snprintf(nm, sizeof(nm), "backbone_encoder_layer_%d", L);
            DCHECK(nm, h, (size_t)M*D);
        }
        // hidden_states[k] = output of layer k-1; feature indices are 3,6,9,12
        int hs_index = L + 1; // after running layer L, this is hidden_states[L+1]
        for (int f = 0; f < 4; f++) {
            if (hs_index == fidx[f]) {
                // clone h into feats[f]
                cl_mem c = alloc_buf((size_t)M*D);
                clEnqueueCopyBuffer(g_q, h, c, 0, 0, (size_t)M*D*sizeof(nnopt_storage_t), 0, nullptr, nullptr);
                feats[f] = c;
            }
        }
    }
    // Dinov2Encoder.forward output == last_hidden_state (final layer output h).
    DCHECK("backbone_encoder", h, (size_t)M*D);
    pool_release(h);

    // ── backbone.layernorm applied to each of the 4 features (apply_layernorm=true) ──
    cl_mem gLN = Wt.get_buffer("backbone.layernorm.weight");
    cl_mem bLN = Wt.get_buffer("backbone.layernorm.bias");
    for (int f = 0; f < 4; f++) {
        cl_mem ln = layernorm(feats[f], M, D, gLN, bLN);
        pool_release(feats[f]);
        feats[f] = ln;
    }
    DCHECK("backbone_layernorm", feats[3], (size_t)M*D);

    // ── DPT neck (reassemble + fusion) + depth head ──
    std::vector<float> out = dpt_neck_head(Wt, feats, M, D, Hp, Wp, PS);

    sync_prof_dump();  // no-op unless NNOPT_SYNC_PROF=1
    NNOPT_CHECKPOINT("depth_forward: complete");
    return out;
}
