// Bonsai-8B (Qwen3) forward pass — host fp32 reference. Every reduction is
// fp32. KV cache sized to CTX_CAP from day one; append guarded; overflow
// clamps loudly (invariant #2).
#pragma once
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <string>
#include <vector>

#include "rope.h"
#include "nnb.h"
#include "q1.h"

class Model {
  public:
    static constexpr int CTX_CAP = 2048;

    explicit Model(const Nnb& nnb) : nnb_(nnb), m_(nnb.meta) {
        rope_.init(m_.rope_theta, m_.yarn_factor, m_.head_dim, m_.yarn_orig_ctx);
        const size_t kv_per_layer = (size_t)CTX_CAP * m_.kv_heads * m_.head_dim;
        kcache_.assign((size_t)m_.layers * kv_per_layer, 0.f);
        vcache_.assign((size_t)m_.layers * kv_per_layer, 0.f);
        // plain-RoPE toggle for the P0.4 YaRN A/B (default: YaRN on)
        plain_rope_ = getenv("BONSAI_PLAIN_ROPE") != nullptr;
        n_threads_ = 8;
        if (const char* t = getenv("BONSAI_THREADS")) n_threads_ = atoi(t);
        dump_dir_ = getenv("DUMP_DIR") ? getenv("DUMP_DIR") : "";
    }

    // Forward ONE token at position pos; returns logits [vocab].
    // (Prefill loops this; simple and exact. Speed is P3's problem.)
    const std::vector<float>& forward(int token, int pos) {
        if (pos >= CTX_CAP) {
            fprintf(stderr, "FATAL: pos %d >= KV cap %d — refusing to wrap\n",
                    pos, CTX_CAP);
            exit(3);
        }
        const int H = m_.hidden, NH = m_.heads, KH = m_.kv_heads,
                  D = m_.head_dim, FF = m_.ffn;
        x_.assign(H, 0.f);
        q1_row_decode(nnb_.get("token_embd.weight").data, token, H, x_.data());

        for (int L = 0; L < m_.layers; ++L) {
            const std::string p = "blk." + std::to_string(L) + ".";
            // ---- attention ----
            rmsnorm(x_, f32(p + "attn_norm.weight"), xb_);
            gemv(p + "attn_q.weight", xb_, q_);       // [NH*D]
            gemv(p + "attn_k.weight", xb_, k_);       // [KH*D]
            gemv(p + "attn_v.weight", xb_, v_);       // [KH*D]
            // Qwen3 per-head q/k RMSNorm over head_dim, then RoPE
            const float* qnw = f32(p + "attn_q_norm.weight");
            const float* knw = f32(p + "attn_k_norm.weight");
            for (int h = 0; h < NH; ++h) head_rmsnorm(&q_[h * D], qnw, D);
            for (int h = 0; h < KH; ++h) head_rmsnorm(&k_[h * D], knw, D);
            rope_apply(q_.data(), NH, D, pos);
            rope_apply(k_.data(), KH, D, pos);
            // KV append (guarded above)
            float* kc = kslot(L, pos);
            float* vc = vslot(L, pos);
            memcpy(kc, k_.data(), sizeof(float) * KH * D);
            memcpy(vc, v_.data(), sizeof(float) * KH * D);
            // GQA attention, fp32 softmax
            att_out_.assign(NH * D, 0.f);
            const int n_ctx = pos + 1;
            const float scale = 1.0f / sqrtf((float)D);
            for (int h = 0; h < NH; ++h) {
                const int kvh = h / (NH / KH);
                const float* qh = &q_[h * D];
                scores_.resize(n_ctx);
                float mx = -1e30f;
                for (int t = 0; t < n_ctx; ++t) {
                    const float* kt = kslot(L, t) + kvh * D;
                    float acc = 0.f;
                    for (int d = 0; d < D; ++d) acc += qh[d] * kt[d];
                    scores_[t] = acc * scale;
                    if (scores_[t] > mx) mx = scores_[t];
                }
                float sum = 0.f;
                for (int t = 0; t < n_ctx; ++t) {
                    scores_[t] = expf(scores_[t] - mx);
                    sum += scores_[t];
                }
                const float inv = 1.0f / sum;
                float* oh = &att_out_[h * D];
                for (int t = 0; t < n_ctx; ++t) {
                    const float w = scores_[t] * inv;
                    const float* vt = vslot(L, t) + kvh * D;
                    for (int d = 0; d < D; ++d) oh[d] += w * vt[d];
                }
            }
            gemv(p + "attn_output.weight", att_out_, xb2_);
            for (int i = 0; i < H; ++i) x_[i] += xb2_[i];
            // ---- ffn ----
            rmsnorm(x_, f32(p + "ffn_norm.weight"), xb_);
            gemv(p + "ffn_gate.weight", xb_, gate_);
            gemv(p + "ffn_up.weight", xb_, up_);
            for (int i = 0; i < FF; ++i) {
                const float g = gate_[i];
                gate_[i] = (g / (1.0f + expf(-g))) * up_[i];  // silu ⊙ up
            }
            gemv(p + "ffn_down.weight", gate_, xb2_);
            for (int i = 0; i < H; ++i) x_[i] += xb2_[i];

            if (!dump_dir_.empty()) dump_layer(L, pos);
        }
        rmsnorm(x_, f32("output_norm.weight"), xb_);
        logits_.resize(m_.vocab);
        gemv("output.weight", xb_, logits_);
        return logits_;
    }

    void print_topk() const {
        std::vector<float> lg = logits_;
        for (int k = 0; k < 5; ++k) {
            int bi = 0;
            for (size_t i = 1; i < lg.size(); ++i) if (lg[i] > lg[bi]) bi = (int)i;
            fprintf(stderr, "  top%d id=%-6d logit=%.6f\n", k, bi, lg[bi]);
            lg[bi] = -1e30f;
        }
    }
    int argmax_logits() const {
        int best = 0;
        for (int i = 1; i < (int)logits_.size(); ++i)
            if (logits_[i] > logits_[best]) best = i;
        return best;
    }
    const ModelMeta& meta() const { return m_; }

  private:
    const float* f32(const std::string& name) {
        const Tensor& t = nnb_.get(name);
        if (t.kind != Tensor::F32) {
            fprintf(stderr, "FATAL: %s expected f32\n", name.c_str());
            exit(3);
        }
        return (const float*)t.data;
    }
    void gemv(const std::string& name, const std::vector<float>& x,
              std::vector<float>& y) {
        const Tensor& t = nnb_.get(name);
        const int n_in = (int)t.ne(0), n_out = (int)t.ne(1);
        if ((int)x.size() != n_in) {
            fprintf(stderr, "FATAL: %s in-dim %d != x %zu\n", name.c_str(),
                    n_in, x.size());
            exit(3);
        }
        y.resize(n_out);
        q1_gemv(t.data, x.data(), y.data(), n_out, n_in, n_threads_);
    }
    void rmsnorm(const std::vector<float>& x, const float* w,
                 std::vector<float>& out) {
        double ss = 0;
        for (float v : x) ss += (double)v * v;
        const float inv = 1.0f / sqrtf((float)(ss / x.size()) + m_.rms_eps);
        out.resize(x.size());
        for (size_t i = 0; i < x.size(); ++i) out[i] = x[i] * inv * w[i];
    }
    void head_rmsnorm(float* h, const float* w, int D) {
        double ss = 0;
        for (int i = 0; i < D; ++i) ss += (double)h[i] * h[i];
        const float inv = 1.0f / sqrtf((float)(ss / D) + m_.rms_eps);
        for (int i = 0; i < D; ++i) h[i] = h[i] * inv * w[i];
    }
    void rope_apply(float* v, int n_heads, int D, int pos) {
        const int half = D / 2;
        for (int h = 0; h < n_heads; ++h) {
            float* p = v + h * D;
            for (int i = 0; i < half; ++i) {   // NEOX pairs (i, i+half)
                float c, s;
                if (plain_rope_) {
                    const float th =
                        (float)pos * powf(m_.rope_theta, -2.0f * i / D);
                    c = cosf(th); s = sinf(th);
                } else {
                    rope_.at(pos, i, &c, &s);
                }
                const float a = p[i], b = p[i + half];
                p[i] = a * c - b * s;
                p[i + half] = a * s + b * c;
            }
        }
    }
    float* kslot(int L, int t) {
        return &kcache_[((size_t)L * CTX_CAP + t) * m_.kv_heads * m_.head_dim];
    }
    float* vslot(int L, int t) {
        return &vcache_[((size_t)L * CTX_CAP + t) * m_.kv_heads * m_.head_dim];
    }
    void dump_layer(int L, int pos) {
        char path[512];
        snprintf(path, sizeof(path), "%s/blk_%02d_pos%04d.f32.bin",
                 dump_dir_.c_str(), L, pos);
        FILE* f = fopen(path, "wb");
        if (f) { fwrite(x_.data(), 4, x_.size(), f); fclose(f); }
    }

    const Nnb& nnb_;
    const ModelMeta& m_;
    RopeYarn rope_;
    bool plain_rope_;
    int n_threads_;
    std::string dump_dir_;
    std::vector<float> kcache_, vcache_;
    std::vector<float> x_, xb_, xb2_, q_, k_, v_, att_out_, scores_, gate_, up_, logits_;
};
