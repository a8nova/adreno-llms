// Host (CPU, fp32) reference for the Bonsai-27B / Qwen3-VL vision tower.
//
// This exists for the same reason reference_model.h did for the text tower: it is the thing you can
// differential-test op by op against transformers WITHOUT a device, a GPU, or an OpenCL kernel. The
// text port found two silent, output-still-fluent bugs that way — the tile-vs-interleave head mapping
// and the double-exp on ssm_a — and neither would have been caught by "does it produce words".
//
// Transcribed from transformers/models/qwen3_vl/modeling_qwen3_vl.py (Qwen3VLVisionModel).
// Every deviation from that file is a bug here, not a design choice. See VISION_PORT.md for the
// config table and the list of ops that have no counterpart in the text tower.
#pragma once
#include <cmath>
#include <cstdint>
#include <cstring>
#include <string>
#include <vector>

#include "nnb.h"

struct VisionMeta {
    int depth = 27;
    int hidden = 1152;
    int ffn = 4304;
    int heads = 16;
    int head_dim = 72;          // 1152/16 — deliberately NOT a power of two
    int patch = 16;
    int temporal_patch = 2;
    int merge = 2;              // spatial_merge_size
    int in_channels = 3;
    int out_hidden = 5120;      // == text hidden; the transformers default 3584 is a smaller model
    float eps = 1e-6f;
    std::vector<int> deepstack_indexes{8, 16, 24};
};

// ── elementwise ────────────────────────────────────────────────────────────
// The block MLP uses gelu_pytorch_tanh; the mergers use nn.GELU() (erf). They differ, and the
// difference is visible in the output — do not unify them.
inline float gelu_tanh(float x) {
    const float c = 0.7978845608028654f;   // sqrt(2/pi)
    return 0.5f * x * (1.0f + std::tanh(c * (x + 0.044715f * x * x * x)));
}
inline float gelu_erf(float x) {
    return 0.5f * x * (1.0f + std::erf(x * 0.70710678118654752f));
}

/** LayerNorm — mean AND variance, with weight and bias. The text tower's RMSNorm has neither. */
inline void layer_norm(const float* x, const float* w, const float* b, float* out, int n, float eps) {
    float mean = 0.0f;
    for (int i = 0; i < n; ++i) mean += x[i];
    mean /= (float)n;
    float var = 0.0f;
    for (int i = 0; i < n; ++i) { const float d = x[i] - mean; var += d * d; }
    var /= (float)n;
    const float inv = 1.0f / std::sqrt(var + eps);
    for (int i = 0; i < n; ++i) out[i] = (x[i] - mean) * inv * w[i] + (b ? b[i] : 0.0f);
}

/** y[out] = W[out][in] . x[in] + b[out]. Row-major W, as torch stores nn.Linear.weight. */
inline void linear(const float* x, const float* W, const float* b, float* y, int in, int out) {
    for (int o = 0; o < out; ++o) {
        const float* row = W + (size_t)o * in;
        float acc = b ? b[o] : 0.0f;
        for (int i = 0; i < in; ++i) acc += row[i] * x[i];
        y[o] = acc;
    }
}

/**
 * Vision RoPE — the `rotate_half` convention, over the FULL head_dim.
 *
 * NOT the text tower's scheme, which is interleaved and PARTIAL (64 of 256 dims). Here
 * emb = cat(rot, rot) so cos/sin have head_dim entries, and the rotation pairs element i with
 * i + head_dim/2:
 *     out[i]      = q[i]      * cos[i]      - q[i + d/2] * sin[i]
 *     out[i+d/2]  = q[i+d/2]  * cos[i+d/2]  + q[i]       * sin[i+d/2]
 */
inline void rope_vision(float* v, const float* cos, const float* sin, int head_dim) {
    const int h = head_dim / 2;
    for (int i = 0; i < h; ++i) {
        const float a = v[i], b = v[i + h];
        v[i]     = a * cos[i]     - b * sin[i];
        v[i + h] = b * cos[i + h] + a * sin[i + h];
    }
}

/** One vision block's weights. Every Linear here has a bias — no text weight does. */
struct VisionBlockW {
    const float *norm1_w, *norm1_b;
    const float *qkv_w, *qkv_b;        // [3*hidden][hidden], [3*hidden]
    const float *proj_w, *proj_b;      // [hidden][hidden]
    const float *norm2_w, *norm2_b;
    const float *fc1_w, *fc1_b;        // [ffn][hidden]
    const float *fc2_w, *fc2_b;        // [hidden][ffn]
};

/**
 * One vision block over `n` patch tokens.
 *   h = h + attn(norm1(h))      full (NON-causal) attention, 16 heads, scale head_dim^-0.5
 *   h = h + mlp(norm2(h))       fc2(gelu_tanh(fc1(.)))
 *
 * Attention is non-causal because patches have no ordering — every patch attends to every patch.
 * That is the opposite of the text tower and the single easiest thing to get wrong by copying it.
 */
inline void vision_block(float* h, int n, const VisionMeta& m, const VisionBlockW& w) {
    const int H = m.hidden, D = m.head_dim, NH = m.heads;
    const float scale = 1.0f / std::sqrt((float)D);
    std::vector<float> nx((size_t)n * H), qkv((size_t)n * 3 * H), att((size_t)n * H), tmp(H);

    for (int t = 0; t < n; ++t) layer_norm(h + (size_t)t * H, w.norm1_w, w.norm1_b, &nx[(size_t)t * H], H, m.eps);
    for (int t = 0; t < n; ++t) linear(&nx[(size_t)t * H], w.qkv_w, w.qkv_b, &qkv[(size_t)t * 3 * H], H, 3 * H);
    // NOTE: rope is applied by the caller into qkv before this point in the real pipeline; kept
    // separate here so each stage can be diffed against transformers independently.

    std::vector<float> scores(n);
    for (int hh = 0; hh < NH; ++hh) {
        for (int i = 0; i < n; ++i) {
            const float* q = &qkv[(size_t)i * 3 * H + (size_t)hh * D];
            float mx = -1e30f;
            for (int j = 0; j < n; ++j) {
                const float* k = &qkv[(size_t)j * 3 * H + H + (size_t)hh * D];
                float s = 0.0f;
                for (int d = 0; d < D; ++d) s += q[d] * k[d];
                scores[j] = s * scale;
                mx = std::max(mx, scores[j]);
            }
            float sum = 0.0f;
            for (int j = 0; j < n; ++j) { scores[j] = std::exp(scores[j] - mx); sum += scores[j]; }
            const float inv = 1.0f / sum;
            float* o = &att[(size_t)i * H + (size_t)hh * D];
            for (int d = 0; d < D; ++d) o[d] = 0.0f;
            for (int j = 0; j < n; ++j) {
                const float p = scores[j] * inv;
                const float* v = &qkv[(size_t)j * 3 * H + 2 * H + (size_t)hh * D];
                for (int d = 0; d < D; ++d) o[d] += p * v[d];
            }
        }
    }
    for (int t = 0; t < n; ++t) {
        linear(&att[(size_t)t * H], w.proj_w, w.proj_b, tmp.data(), H, H);
        for (int i = 0; i < H; ++i) h[(size_t)t * H + i] += tmp[i];
    }

    std::vector<float> f(m.ffn);
    for (int t = 0; t < n; ++t) {
        layer_norm(h + (size_t)t * H, w.norm2_w, w.norm2_b, nx.data(), H, m.eps);
        linear(nx.data(), w.fc1_w, w.fc1_b, f.data(), H, m.ffn);
        for (int i = 0; i < m.ffn; ++i) f[i] = gelu_tanh(f[i]);
        linear(f.data(), w.fc2_w, w.fc2_b, tmp.data(), m.ffn, H);
        for (int i = 0; i < H; ++i) h[(size_t)t * H + i] += tmp[i];
    }
}

/** Patch merger, used for the final output AND for each deepstack tap. */
struct MergerW {
    const float *norm_w, *norm_b;
    const float *fc1_w, *fc1_b;   // [4*hidden][4*hidden]
    const float *fc2_w, *fc2_b;   // [out_hidden][4*hidden]
};

/**
 * LayerNorm over hidden, then group each 2x2 spatial neighbourhood into one 4*hidden vector, then
 * fc2(GELU_erf(fc1(.))) -> out_hidden. `n` must be a multiple of merge^2.
 *
 * The GELU here is the erf form (nn.GELU()), NOT the tanh approximation the block MLP uses.
 */
inline void patch_merger(const float* h, int n, const VisionMeta& m, const MergerW& w,
                         std::vector<float>* out) {
    const int H = m.hidden, G = m.merge * m.merge, HG = H * G;
    const int groups = n / G;
    out->assign((size_t)groups * m.out_hidden, 0.0f);
    std::vector<float> normed((size_t)HG), mid((size_t)HG);
    for (int g = 0; g < groups; ++g) {
        for (int s = 0; s < G; ++s)
            layer_norm(h + ((size_t)g * G + s) * H, w.norm_w, w.norm_b, &normed[(size_t)s * H], H, m.eps);
        linear(normed.data(), w.fc1_w, w.fc1_b, mid.data(), HG, HG);
        for (int i = 0; i < HG; ++i) mid[i] = gelu_erf(mid[i]);
        linear(mid.data(), w.fc2_w, w.fc2_b, &(*out)[(size_t)g * m.out_hidden], HG, m.out_hidden);
    }
}

/**
 * smart_resize — aspect-preserving, both dims a multiple of `factor`, total pixels clamped into
 * [min_pixels, max_pixels]. Ported from qwen2_vl/image_processing_qwen2_vl.py:76.
 *
 * factor is patch_size * spatial_merge_size = 32 here. This is NOT the fixed-square resize the
 * smolvlm / lfm2-vl SigLIP ports use, and it is why an image's token count varies.
 */
inline void smart_resize(int h, int w, int factor, long min_pixels, long max_pixels,
                         int* out_h, int* out_w) {
    auto round_by = [factor](double v) { return std::max(factor, (int)(std::round(v / factor) * factor)); };
    int hb = round_by(h), wb = round_by(w);
    if ((long)hb * wb > max_pixels) {
        const double beta = std::sqrt((double)h * w / (double)max_pixels);
        hb = std::max(factor, (int)(std::floor(h / beta / factor) * factor));
        wb = std::max(factor, (int)(std::floor(w / beta / factor) * factor));
    } else if ((long)hb * wb < min_pixels) {
        const double beta = std::sqrt((double)min_pixels / ((double)h * w));
        hb = (int)(std::ceil(h * beta / factor) * factor);
        wb = (int)(std::ceil(w * beta / factor) * factor);
    }
    *out_h = hb; *out_w = wb;
}

/** Tokens an image becomes after patching and the 2x2 merge — what prefill must process. */
inline long vision_token_count(int resized_h, int resized_w, const VisionMeta& m) {
    const int per = m.patch * m.merge;                 // 32 px per output token, each side
    return (long)(resized_h / per) * (resized_w / per);
}
