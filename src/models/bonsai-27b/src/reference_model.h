// Bonsai-27B (qwen35 hybrid) forward pass — host fp32 reference.
//
// This is the CORRECTNESS SPINE of the port: no OpenCL, no device. Build with
// scripts/build_host.sh and diff against the llama.cpp oracle before writing a
// single kernel.
//
// The stack alternates two block kinds (meta.layer_types, e.g. "LLLFLLLF..."):
//   'L' — Gated DeltaNet linear attention. Recurrent [d_k x d_v] fp32 state per
//         VALUE head; no KV cache. 48 of 64 blocks on the 27B.
//   'F' — Gated full attention. head_dim 256, GQA, sigmoid output gate, KV
//         cache, PARTIAL RoPE. 16 of 64 blocks.
// Both blocks share the SwiGLU MLP and the two RMSNorms.
//
// Math transcribed from transformers `qwen3_5/modular_qwen3_5.py` +
// `qwen3_next/modeling_qwen3_next.py`. Deviations that would silently produce
// plausible-but-wrong logits are called out inline — they are the traps.
#pragma once
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>

#include "delta_rule.h"
#include "nnb.h"
#include "q1.h"

class Model {
  public:
    static constexpr int CTX_CAP = 2048;

    explicit Model(const Nnb& nnb) : nnb_(nnb), m_(nnb.meta) {
        if (m_.arch != "qwen35") {
            fprintf(stderr, "FATAL: reference_model expects arch qwen35, got '%s'\n",
                    m_.arch.c_str());
            exit(3);
        }
        if ((int)m_.layer_types.size() != m_.layers) {
            fprintf(stderr, "FATAL: layer_types '%s' has %zu entries, expected %d\n",
                    m_.layer_types.c_str(), m_.layer_types.size(), m_.layers);
            exit(3);
        }
        // ---- dims (GGUF ssm.* keys are Mamba-inherited misnomers) ----
        nv_ = m_.ssm_time_step_rank;   // num_v_heads   (48 on 27B, 16 on 3.5-2B)
        nk_ = m_.ssm_group_count;      // num_k_heads   (16 on both)
        dk_ = m_.ssm_state_size;       // head_k_dim == head_v_dim (128)
        dv_ = dk_;
        if (nv_ <= 0 || nk_ <= 0 || nv_ % nk_) {
            fprintf(stderr, "FATAL: bad v/k head counts %d/%d\n", nv_, nk_);
            exit(3);
        }
        vk_rep_ = nv_ / nk_;           // 3 on the 27B, 1 on Qwen3.5-2B
        // (head mapping itself is TILE — see gdn::step in delta_rule.h)
        key_dim_ = nk_ * dk_;
        value_dim_ = nv_ * dv_;
        qkv_dim_ = 2 * key_dim_ + value_dim_;
        conv_k_ = m_.ssm_conv_kernel;
        rot_ = m_.rope_dim_count > 0 ? m_.rope_dim_count : m_.head_dim;

        n_threads_ = 8;
        if (const char* t = getenv("BONSAI_THREADS")) n_threads_ = atoi(t);
        dump_dir_ = getenv("DUMP_DIR") ? getenv("DUMP_DIR") : "";
        diag_ = getenv("BONSAI_DIAG") != nullptr;

        // ---- per-block state ----
        // 'L': recurrent [nv][dk][dv] fp32 + conv window [qkv_dim][conv_k] fp32.
        //      fp32 is NOT optional — as half these underflow/saturate within a
        //      few timesteps (the mamba2-130m lesson).
        // 'F': KV cache.
        rec_.resize(m_.layers);
        conv_.resize(m_.layers);
        kcache_.resize(m_.layers);
        vcache_.resize(m_.layers);
        for (int L = 0; L < m_.layers; ++L) {
            if (m_.layer_types[L] == 'L') {
                rec_[L].assign((size_t)nv_ * dk_ * dv_, 0.f);
                conv_[L].assign((size_t)qkv_dim_ * conv_k_, 0.f);
            } else {
                const size_t per = (size_t)CTX_CAP * m_.kv_heads * m_.head_dim;
                kcache_[L].assign(per, 0.f);
                vcache_[L].assign(per, 0.f);
            }
        }
        // inv_freq over the ROTARY dims only (partial_rotary_factor * head_dim).
        const int half = rot_ / 2;
        inv_freq_.resize(half);
        for (int i = 0; i < half; ++i)
            inv_freq_[i] = 1.0f / powf(m_.rope_theta, (float)(2 * i) / (float)rot_);
    }

    // Forward ONE token at position pos; returns logits [vocab].
    const std::vector<float>& forward(int token, int pos) {
        if (pos >= CTX_CAP) {
            fprintf(stderr, "FATAL: pos %d >= CTX_CAP %d\n", pos, CTX_CAP);
            exit(3);
        }
        const int H = m_.hidden;
        x_.assign(H, 0.f);
        q1_row_decode(nnb_.get("token_embd.weight").data, token, H, x_.data());
        if (!dump_dir_.empty()) dump_vec("emb", pos, x_);

        for (int L = 0; L < m_.layers; ++L) {
            const std::string p = "blk." + std::to_string(L) + ".";
            // ---- token mixer ----
            rmsnorm(x_, f32(p + "attn_norm.weight"), xb_);
            if (m_.layer_types[L] == 'L') linear_attn(L, p, xb_, mix_);
            else                          full_attn(L, p, xb_, mix_, pos);
            double mixer_rms = 0;
            if (diag_) {
                double s2 = 0;
                for (int i = 0; i < H; ++i) s2 += (double)mix_[i] * mix_[i];
                mixer_rms = sqrt(s2 / H);
            }
            if (!dump_dir_.empty()) dump_vec("mix" + std::to_string(L), pos, mix_);
            for (int i = 0; i < H; ++i) x_[i] += mix_[i];
            if (!dump_dir_.empty()) dump_vec("postmix" + std::to_string(L), pos, x_);
            // ---- MLP ----
            rmsnorm(x_, f32(p + "post_attention_norm.weight"), xb_);
            gemv(p + "ffn_gate.weight", xb_, gate_);
            gemv(p + "ffn_up.weight", xb_, up_);
            for (size_t i = 0; i < gate_.size(); ++i)
                gate_[i] = silu(gate_[i]) * up_[i];
            gemv(p + "ffn_down.weight", gate_, mix_);
            for (int i = 0; i < H; ++i) x_[i] += mix_[i];

            if (!dump_dir_.empty()) dump_layer(L, pos);
            // BONSAI_DIAG=1: per-block magnitudes. A block kind whose mixer
            // output is ~0 (or NaN/huge) relative to the residual stream is not
            // contributing, which localises a wiring bug to one branch without
            // needing per-tensor dumps.
            if (diag_) {
                double sx = 0, sm = 0;
                for (int i = 0; i < H; ++i) { sx += (double)x_[i] * x_[i]; sm += (double)mix_[i] * mix_[i]; }
                fprintf(stderr, "[diag] blk %2d %c  |x|=%9.4f  |mixer|=%9.4f  |mlp|=%9.4f%s\n",
                        L, m_.layer_types[L], sqrt(sx / H), mixer_rms, sqrt(sm / H),
                        (std::isnan(sx) || std::isnan(sm)) ? "  NaN!" : "");
            }
        }
        rmsnorm(x_, f32("output_norm.weight"), xb_);
        logits_.resize(m_.vocab);
        gemv(nnb_.has("output.weight") ? "output.weight" : "token_embd.weight",
             xb_, logits_);
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
    // ─────────────────────── 'L' block: Gated DeltaNet ───────────────────────
    void linear_attn(int L, const std::string& p,
                     const std::vector<float>& in, std::vector<float>& out) {
        gemv(p + "attn_qkv.weight", in, qkv_);      // [2*key_dim + value_dim]
        // NOTE: "attn_gate" on an 'L' block is in_proj_z — the SSM OUTPUT GATE,
        // not an attention gate.
        gemv(p + "attn_gate.weight", in, z_);       // [value_dim]
        gemv(p + "ssm_beta.weight", in, bvec_);     // [nv]
        gemv(p + "ssm_alpha.weight", in, avec_);    // [nv]

        // ---- causal depthwise conv over ALL qkv channels, then silu ----
        // conv1d.weight is [conv_k, qkv_dim] in GGUF order (ne0 = tap, fastest),
        // i.e. taps for channel c live at w[c*conv_k + k]. No bias tensor exists.
        const float* cw = f32(p + "ssm_conv1d.weight");
        float* cs = conv_[L].data();
        for (int c = 0; c < qkv_dim_; ++c) {
            float* win = cs + (size_t)c * conv_k_;
            for (int k = 0; k < conv_k_ - 1; ++k) win[k] = win[k + 1];  // roll
            win[conv_k_ - 1] = qkv_[c];
            float acc = 0.f;
            for (int k = 0; k < conv_k_; ++k) acc += win[k] * cw[c * conv_k_ + k];
            qkv_[c] = silu(acc);
        }

        const float* q = qkv_.data();
        const float* k = qkv_.data() + key_dim_;
        const float* v = qkv_.data() + 2 * key_dim_;
        const float* a_neg = f32(p + "ssm_a");        // [nv] ALREADY -exp(A_log)
        const float* dt_b = f32(p + "ssm_dt.bias");   // [nv]
        const float* nw = f32(p + "ssm_norm.weight"); // [dv]

        // The recurrence itself lives in delta_rule.h so that this reference,
        // the differential test (scripts/test_delta_net.py, verified against
        // transformers' torch_recurrent_gated_delta_rule) and the eventual
        // kernels/delta_net.cl all track ONE implementation.
        core_.assign((size_t)value_dim_, 0.f);
        gdn::step(rec_[L].data(), q, k, v, avec_.data(), bvec_.data(),
                  a_neg, dt_b, nv_, nk_, dk_, dv_, core_.data(), scratch_);

        // Gated RMSNorm per (value head) row of dv.
        // ORDER MATTERS: normalise the RAW input, then gate — NOT
        // norm(x * silu(gate)) as mamba2-130m's kernel does. transformers
        // comments this explicitly ("# Norm before gate").
        for (int hv = 0; hv < nv_; ++hv) {
            float* o = core_.data() + (size_t)hv * dv_;
            const float* zg = z_.data() + (size_t)hv * dv_;
            double ss = 0;
            for (int j = 0; j < dv_; ++j) ss += (double)o[j] * o[j];
            const float inv = 1.0f / sqrtf((float)(ss / dv_) + m_.rms_eps);
            for (int j = 0; j < dv_; ++j)
                o[j] = (o[j] * inv * nw[j]) * silu(zg[j]);
        }
        gemv(p + "ssm_out.weight", core_, out);
    }

    // ───────────────────── 'F' block: gated full attention ─────────────────────
    void full_attn(int L, const std::string& p, const std::vector<float>& in,
                   std::vector<float>& out, int pos) {
        const int NH = m_.heads, KH = m_.kv_heads, D = m_.head_dim;
        // q_proj emits n_heads * head_dim * 2, viewed as (-1, 2*D) then chunked
        // on the LAST dim — so q and gate INTERLEAVE PER HEAD: within each
        // head's 2*D outputs, [0,D) is q and [D,2*D) is the gate. Sizing this
        // as NH*D is off by 2x.
        gemv(p + "attn_q.weight", in, qg_);          // [NH * 2D]
        q_.resize((size_t)NH * D);
        agate_.resize((size_t)NH * D);
        for (int h = 0; h < NH; ++h) {
            const float* src = qg_.data() + (size_t)h * 2 * D;
            memcpy(q_.data() + (size_t)h * D, src, sizeof(float) * D);
            memcpy(agate_.data() + (size_t)h * D, src + D, sizeof(float) * D);
        }
        gemv(p + "attn_k.weight", in, k_);            // [KH * D]
        gemv(p + "attn_v.weight", in, v_);            // [KH * D]

        const float* qnw = f32(p + "attn_q_norm.weight");
        const float* knw = f32(p + "attn_k_norm.weight");
        for (int h = 0; h < NH; ++h) head_rmsnorm(&q_[(size_t)h * D], qnw, D);
        for (int h = 0; h < KH; ++h) head_rmsnorm(&k_[(size_t)h * D], knw, D);
        rope_partial(q_.data(), NH, D, pos);
        rope_partial(k_.data(), KH, D, pos);

        float* kc = &kcache_[L][(size_t)pos * KH * D];
        float* vc = &vcache_[L][(size_t)pos * KH * D];
        memcpy(kc, k_.data(), sizeof(float) * KH * D);
        memcpy(vc, v_.data(), sizeof(float) * KH * D);

        const int seq = pos + 1;
        const int grp = NH / KH;
        const float scale = 1.0f / sqrtf((float)D);
        att_.assign((size_t)NH * D, 0.f);
        scores_.resize(seq);
        for (int h = 0; h < NH; ++h) {
            const int kh = h / grp;
            const float* qh = q_.data() + (size_t)h * D;
            float mx = -1e30f;
            for (int t = 0; t < seq; ++t) {
                const float* kt = &kcache_[L][((size_t)t * KH + kh) * D];
                float dot = 0.f;
                for (int i = 0; i < D; ++i) dot += qh[i] * kt[i];
                scores_[t] = dot * scale;
                if (scores_[t] > mx) mx = scores_[t];
            }
            float sum = 0.f;
            for (int t = 0; t < seq; ++t) { scores_[t] = expf(scores_[t] - mx); sum += scores_[t]; }
            const float inv = 1.0f / sum;
            float* oh = att_.data() + (size_t)h * D;
            for (int t = 0; t < seq; ++t) {
                const float w = scores_[t] * inv;
                const float* vt = &vcache_[L][((size_t)t * KH + kh) * D];
                for (int i = 0; i < D; ++i) oh[i] += w * vt[i];
            }
        }
        // output gate is SIGMOID here (the SSM z-gate above is SILU)
        for (size_t i = 0; i < att_.size(); ++i) att_[i] *= sigmoidf(agate_[i]);
        gemv(p + "attn_output.weight", att_, out);
    }

    // ───────────────────────────── helpers ─────────────────────────────
    // activations shared with delta_rule.h / the kernels
    static float silu(float v)     { return gdn::silu(v); }
    static float sigmoidf(float v) { return gdn::sigmoidf(v); }
    // PARTIAL RoPE: only the first `rot_` dims of each head rotate (64 of 256
    // on this model); dims [rot_, D) pass through untouched. NeoX pairing
    // (i, i + rot_/2). Interleaved mRoPE collapses to plain RoPE for
    // text-only input, because the T/H/W position ids are all identical.
    void rope_partial(float* v, int n_heads, int D, int pos) {
        const int half = rot_ / 2;
        for (int h = 0; h < n_heads; ++h) {
            float* p = v + (size_t)h * D;
            for (int i = 0; i < half; ++i) {
                const float th = (float)pos * inv_freq_[i];
                const float c = cosf(th), s = sinf(th);
                const float a = p[i], b = p[i + half];
                p[i] = a * c - b * s;
                p[i + half] = a * s + b * c;
            }
        }
    }
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
    // Dump a named intermediate so scripts/diff_block.py can recompute the same
    // stage in numpy from the SAME .nnb weights and localise a divergence to one
    // operator instead of one hypothesis at a time.
    void dump_vec(const std::string& tag, int pos, const std::vector<float>& v) {
        char path[512];
        snprintf(path, sizeof(path), "%s/%s_pos%04d.f32.bin",
                 dump_dir_.c_str(), tag.c_str(), pos);
        FILE* f = fopen(path, "wb");
        if (f) { fwrite(v.data(), 4, v.size(), f); fclose(f); }
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
    int nv_ = 0, nk_ = 0, dk_ = 0, dv_ = 0, vk_rep_ = 1;
    int key_dim_ = 0, value_dim_ = 0, qkv_dim_ = 0, conv_k_ = 0, rot_ = 0;
    int n_threads_ = 8;
    bool diag_ = false;
    std::string dump_dir_;
    std::vector<float> inv_freq_;
    std::vector<std::vector<float>> rec_, conv_, kcache_, vcache_;
    std::vector<float> x_, xb_, mix_, qkv_, z_, bvec_, avec_, core_, scratch_,
                       qg_, q_, agate_, k_, v_, att_, scores_,
                       gate_, up_, logits_;
};
