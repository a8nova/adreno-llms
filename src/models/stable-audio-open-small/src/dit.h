#pragma once
// Reference: stable_audio_tools/models/dit.py DiffusionTransformer._forward
//            + stable_audio_tools/models/transformer.py ContinuousTransformer.forward
//
// Bespoke DiT single-denoise-step forward for Stable Audio Open Small.
// The generic graph-mode backbone (vocab logits) does not fit a diffusion
// audio model, so the DiT is implemented directly here.
//
// forward_step:
//   x_latent   : host float [io_channels(64) * latent_len]  (channel-major [C,T])
//   t          : scalar timestep (sigma; rf_denoiser -> sigma=t)
//   cross_cond : host float [cross_seq(65) * 768]  (T5 cond, pre to_cond_embed)
//   global_emb : host float [768]                  (seconds cond, pre to_global_embed)
//   out        : host float [io_channels(64) * latent_len]  (raw model output v)
//
// Returns true on success. Emits NNOPT_LAYER_CHECK dumps matching the
// reference dump_names so SxSDebug/AutoBisect can localize divergence.

#include "opencl_context.h"
#include "weights.h"
#include <vector>
#include <string>
#include <unordered_map>

class DiT {
public:
    DiT(OpenCLContext& cl_ctx, Weights& weights);
    ~DiT();

    bool initialize();

    // Run one denoise step. latent_len (T) is derived from x_latent.size().
    bool forward_step(const std::vector<float>& x_latent,
                      float t,
                      const std::vector<float>& cross_cond, int cross_seq,
                      const std::vector<float>& global_emb,
                      std::vector<float>& out);

    // B4 (on-GPU denoise loop) device-resident API: x stays a cl_mem across
    // all 8 steps; only the final latent is downloaded. v_out receives a NEW
    // buffer with the raw model output [C,T] (caller releases).
    bool forward_step_dev(cl_mem x_ct, int T, float t,
                          const std::vector<float>& cross_cond, int cross_seq,
                          const std::vector<float>& global_emb,
                          cl_mem* v_out);
    // In-place pingpong update: x = (1-sn)*(x - sc*v) + sn*noise.
    bool denoise_resample(cl_mem x, cl_mem v, cl_mem noise,
                          float sc, float sn, int n);
    // Buffer plumbing for the pipeline driver (pool-backed).
    cl_mem upload_buf(const std::vector<float>& host, size_t nelem) { return upload(host, nelem); }
    void download_buf(cl_mem buf, std::vector<float>& host, size_t nelem) { download(buf, host, nelem); }
    void release_buf(cl_mem b);

private:
    OpenCLContext& cl_ctx_;
    Weights& weights_;
    cl_program prog_ = nullptr;   // kernels/dit.cl
    cl_program utils_ = nullptr;  // kernels/utils.cl (element_add)
    // B13 experiment program (kernels/linear_p8.cl) — built lazily ONLY when
    // NNOPT_BESPOKE_LINEAR=1: its register-spilling kernel must never share a
    // program with the hot dit.cl kernels (Adreno program-wide register trap;
    // measured DiT 32 → 44 s from mere co-residence).
    cl_program p8prog_ = nullptr;
    bool ready_ = false;

    // OPT-1: step-invariant cache. cross_flat (to_cond_embed output), global_e
    // (to_global_embed output, pre-timestep add) and the RoPE freq table depend
    // only on the conditioning / sequence length — not on t — yet the denoise
    // loop calls forward_step 8× with identical values. Cache keyed on a
    // fingerprint of the inputs; recomputed transparently when the prompt or
    // latent length changes. NNOPT_STEP_CACHE=0 reverts to per-step recompute.
    uint64_t  cache_sig_        = 0;
    cl_mem    cached_cross_flat_ = nullptr;   // [cross_seq, D]
    cl_mem    cached_global_e_   = nullptr;   // [D]
    cl_mem    cached_freqs_      = nullptr;   // [seq, rot_dim]

    // B11: per-weight transposed copy [K,N] of each nn.Linear weight [N,K],
    // built once on-GPU at first use so every DiT GEMM runs CLBlast's
    // NoTrans×NoTrans path (the TransB path re-pads/transposes the WEIGHT
    // matrix inside every call, every step, at stock params).
    // NNOPT_PRETRANS=0 reverts to pytorch_linear's TransB path.
    std::unordered_map<std::string, cl_mem> wt_cache_;

    // helpers
    cl_mem alloc(size_t nelem);
    cl_mem upload(const std::vector<float>& host, size_t nelem);
    void download(cl_mem buf, std::vector<float>& host, size_t nelem);
    // out = x @ W^T  (nn.Linear, W is [N,K]); returns new buffer [M,N]
    cl_mem linear(cl_mem x, const std::string& wkey, int M, int N, int K);
    void silu(cl_mem x, int n);
    cl_mem layernorm(cl_mem x, const std::string& gkey, const std::string& bkey,
                     int rows, int D, float eps);
    // one transformer block, in-place on x_flat [seq, D]; returns x_flat (same buf)
    cl_mem block(cl_mem x_flat, int layer_idx, int seq,
                 cl_mem cross_flat, int cross_seq, cl_mem freqs);
    cl_mem attention(cl_mem normed, int layer_idx, int seq,
                     cl_mem cross_flat, int cross_seq, cl_mem freqs, bool is_cross);
};
