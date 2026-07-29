// YaRN RoPE math (NEOX path) — shared by the host reference oracle
// (reference_model.h) and the OpenCL device model's rope-table builder.
#pragma once
#include <cmath>

struct RopeYarn {
    // Faithful port of ggml's rope_yarn (NEOX path) — llama.cpp applies YaRN
    // whenever the GGUF carries qwen3.rope.scaling.*, so we do too.
    // Constants: beta_fast=32, beta_slow=1, attn_factor=1, ext_factor=1.
    float theta_base, freq_scale;
    int n_dims, n_ctx_orig;
    float corr_low, corr_high, mscale;

    void init(float base, float yarn_factor, int head_dim, int orig_ctx) {
        theta_base = base;
        freq_scale = 1.0f / yarn_factor;
        n_dims = head_dim;
        n_ctx_orig = orig_ctx;
        auto corr_dim = [&](float n_beta) {
            return n_dims * logf(n_ctx_orig / (n_beta * 2.0f * (float)M_PI)) /
                   (2.0f * logf(theta_base));
        };
        corr_low = floorf(corr_dim(32.0f));
        corr_high = ceilf(corr_dim(1.0f));
        if (corr_low < 0) corr_low = 0;
        if (corr_high > n_dims - 1) corr_high = (float)(n_dims - 1);
        mscale = 1.0f + 0.1f * logf(1.0f / freq_scale);  // attn_factor = 1
    }
    // cos/sin for rotation pair index i0 (0..n_dims/2), position pos
    inline void at(int pos, int i0, float* c, float* s) const {
        const float theta_extrap =
            (float)pos * powf(theta_base, -2.0f * i0 / n_dims);
        const float theta_interp = freq_scale * theta_extrap;
        float ramp_y = (i0 - corr_low) / fmaxf(0.001f, corr_high - corr_low);
        float ramp = 1.0f - fminf(1.0f, fmaxf(0.0f, ramp_y));
        const float ext_factor = 1.0f;
        const float mix = ramp * ext_factor;
        const float theta = theta_interp * (1.0f - mix) + theta_extrap * mix;
        *c = cosf(theta) * mscale;
        *s = sinf(theta) * mscale;
    }
};
