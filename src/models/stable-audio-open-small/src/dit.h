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

private:
    OpenCLContext& cl_ctx_;
    Weights& weights_;
    cl_program prog_ = nullptr;   // kernels/dit.cl
    cl_program utils_ = nullptr;  // kernels/utils.cl (element_add)
    bool ready_ = false;

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
