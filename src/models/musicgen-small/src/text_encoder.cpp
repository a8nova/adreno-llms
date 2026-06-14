// Host-side T5 text encoder — see text_encoder.h for the contract.
// Reference: model_info/transformers_src/modeling_t5.py
//   T5LayerNorm.forward            (RMS norm, eps=1e-6, weight only)
//   T5Attention.forward            (no softmax scaling; additive position_bias)
//   T5Attention._relative_position_bucket (bidirectional=True, 32 buckets, max 128)
//   T5LayerSelfAttention / T5LayerFF (pre-norm + residual)
//   T5Stack.forward                (blocks → final_layer_norm)
// MusicGen bridge: modeling_musicgen.py MusicgenForConditionalGeneration —
//   encoder_hidden_states = enc_to_dec_proj(encoder_outputs[0])

#include "text_encoder.h"
#include "debug_utils.h"
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <thread>
#include <algorithm>
#include <functional>

namespace {

// Multi-core split over output rows (same pattern as encodec_host.cpp).
// Disjoint output slices; per-element math unchanged → bit-identical.
void t5_parallel(int n, const std::function<void(int, int)>& fn) {
    unsigned hw = std::thread::hardware_concurrency();
    int nthreads = (int)std::min<unsigned>(hw ? hw : 4, 8);
    if (nthreads > n) nthreads = n;
    if (nthreads <= 1) { fn(0, n); return; }
    std::vector<std::thread> pool;
    pool.reserve(nthreads - 1);
    int chunk = (n + nthreads - 1) / nthreads;
    for (int i = 1; i < nthreads; ++i) {
        int lo = i * chunk, hi = std::min(n, lo + chunk);
        if (lo >= hi) break;
        pool.emplace_back(fn, lo, hi);
    }
    fn(0, std::min(n, chunk));
    for (auto& t : pool) t.join();
}

constexpr int kDModel = 768;
constexpr int kHeads = 12;
constexpr int kDKv = 64;          // per-head dim; heads * d_kv == inner dim 768
constexpr int kDFf = 3072;
constexpr int kBlocks = 12;
constexpr int kRelBuckets = 32;
constexpr int kRelMaxDist = 128;
constexpr int kProjDim = 1024;
constexpr float kLnEps = 1e-6f;

// T5 RMS layer norm: x * w / sqrt(mean(x^2) + eps). No mean subtraction.
// modeling_t5.py::T5LayerNorm.forward
void t5_layer_norm(const std::vector<float>& w, std::vector<float>& x, int T) {
    for (int t = 0; t < T; ++t) {
        float* row = x.data() + (size_t)t * kDModel;
        double ss = 0.0;
        for (int i = 0; i < kDModel; ++i) ss += (double)row[i] * row[i];
        const float inv = 1.0f / std::sqrt((float)(ss / kDModel) + kLnEps);
        for (int i = 0; i < kDModel; ++i) row[i] = row[i] * inv * w[(size_t)i];
    }
}

// y[T,out] = x[T,in] @ W[out,in]^T  (PyTorch nn.Linear convention, no bias)
// Parallel over out rows (o-outer, t-inner so each thread streams its W rows
// once); per-(t,o) accumulation order unchanged → bit-identical. This was
// ~1 GMACs of single-core double-accum = most of the ~5 s "prefill" segment.
void linear_nobias(const std::vector<float>& W, int out_dim, int in_dim,
                   const std::vector<float>& x, int T, std::vector<float>& y) {
    y.assign((size_t)T * out_dim, 0.0f);
    t5_parallel(out_dim, [&](int o_lo, int o_hi) {
        for (int o = o_lo; o < o_hi; ++o) {
            const float* wr = W.data() + (size_t)o * in_dim;
            for (int t = 0; t < T; ++t) {
                const float* xr = x.data() + (size_t)t * in_dim;
                double acc = 0.0;
                for (int i = 0; i < in_dim; ++i) acc += (double)xr[i] * wr[i];
                y[(size_t)t * out_dim + o] = (float)acc;
            }
        }
    });
}

// modeling_t5.py::T5Attention._relative_position_bucket, bidirectional=True.
int rel_bucket(int relative_position) {
    int num_buckets = kRelBuckets;
    int ret = 0;
    num_buckets /= 2;                      // bidirectional split
    if (relative_position > 0) ret += num_buckets;
    int n = std::abs(relative_position);
    const int max_exact = num_buckets / 2; // 8
    if (n < max_exact) return ret + n;
    // log-spaced buckets for larger distances
    float v = max_exact +
        std::log((float)n / max_exact) / std::log((float)kRelMaxDist / max_exact) *
            (num_buckets - max_exact);
    int large = (int)v;
    if (large > num_buckets - 1) large = num_buckets - 1;
    return ret + large;
}

void maybe_dump_block(int block_idx, const std::vector<float>& x, int T) {
    const char* dump = std::getenv("NNOPT_DUMP_LAYERS");
    if (!dump || std::strcmp(dump, "1") != 0) return;
    char path[128];
    std::snprintf(path, sizeof(path), "layer_dumps/block_layer_%d.bin", block_idx);
    if (FILE* f = std::fopen(path, "wb")) {
        std::fwrite(x.data(), sizeof(float), (size_t)T * kDModel, f);
        std::fclose(f);
        char meta[160];
        std::snprintf(meta, sizeof(meta), "layer_dumps/block_layer_%d.bin.meta.json", block_idx);
        if (FILE* m = std::fopen(meta, "wb")) {
            std::fprintf(m,
                "{\"dtype\":\"float32\",\"num_elements\":%zu,\"bytes_per_element\":4,\"pass\":0}\n",
                (size_t)T * kDModel);
            std::fclose(m);
        }
    }
}

} // namespace

std::vector<float> t5_encode_host(Weights& weights, const std::vector<int32_t>& ids) {
    const int T = (int)ids.size();
    if (T <= 0) { NNOPT_ERROR("t5_encode_host: empty ids"); return {}; }

    // ── Embedding (text_encoder.shared) ─────────────────────────────────
    const std::vector<float> emb = weights.get_host_vec("text_encoder.shared.weight");
    if (emb.size() < (size_t)kDModel) { NNOPT_ERROR("t5_encode_host: missing text_encoder.shared.weight"); return {}; }
    const int vocab_rows = (int)(emb.size() / kDModel);
    std::vector<float> x((size_t)T * kDModel);
    for (int t = 0; t < T; ++t) {
        const int id = ids[(size_t)t];
        if (id < 0 || id >= vocab_rows) {
            NNOPT_ERROR_FMT("t5_encode_host: id %d out of range (vocab %d)", id, vocab_rows);
            return {};
        }
        std::memcpy(x.data() + (size_t)t * kDModel, emb.data() + (size_t)id * kDModel,
                    sizeof(float) * kDModel);
    }

    // ── Relative position bias (block 0, shared by every block) ─────────
    // bias[h][i][j] = rel_emb[bucket(j - i)][h]   (query i, key j)
    const std::vector<float> rel =
        weights.get_host_vec("text_encoder.encoder.block.0.layer.0.SelfAttention.relative_attention_bias.weight");
    if (rel.size() != (size_t)kRelBuckets * kHeads) {
        NNOPT_ERROR_FMT("t5_encode_host: rel bias size %zu != %d", rel.size(), kRelBuckets * kHeads);
        return {};
    }
    std::vector<float> pos_bias((size_t)kHeads * T * T);
    for (int i = 0; i < T; ++i) {
        for (int j = 0; j < T; ++j) {
            const int b = rel_bucket(j - i);   // memory_position - query_position
            for (int h = 0; h < kHeads; ++h) {
                pos_bias[((size_t)h * T + i) * T + j] = rel[(size_t)b * kHeads + h];
            }
        }
    }

    std::vector<float> normed, q, k, v, attn_out, ff_h, ff_o;
    std::vector<float> scores((size_t)T * T);

    for (int blk = 0; blk < kBlocks; ++blk) {
        char key[160];
        auto W = [&](const char* fmt) {
            std::snprintf(key, sizeof(key), fmt, blk);
            return weights.get_host_vec(key);
        };

        // ── layer.0: self-attention (pre-norm, residual) ─────────────────
        const std::vector<float> ln0 = W("text_encoder.encoder.block.%d.layer.0.layer_norm.weight");
        const std::vector<float> Wq = W("text_encoder.encoder.block.%d.layer.0.SelfAttention.q.weight");
        const std::vector<float> Wk = W("text_encoder.encoder.block.%d.layer.0.SelfAttention.k.weight");
        const std::vector<float> Wv = W("text_encoder.encoder.block.%d.layer.0.SelfAttention.v.weight");
        const std::vector<float> Wo = W("text_encoder.encoder.block.%d.layer.0.SelfAttention.o.weight");
        if (ln0.empty() || Wq.empty() || Wk.empty() || Wv.empty() || Wo.empty()) {
            NNOPT_ERROR_FMT("t5_encode_host: missing attn weights at block %d", blk);
            return {};
        }
        normed = x;
        t5_layer_norm(ln0, normed, T);
        linear_nobias(Wq, kDModel, kDModel, normed, T, q);
        linear_nobias(Wk, kDModel, kDModel, normed, T, k);
        linear_nobias(Wv, kDModel, kDModel, normed, T, v);

        attn_out.assign((size_t)T * kDModel, 0.0f);
        for (int h = 0; h < kHeads; ++h) {
            const int off = h * kDKv;
            // scores[i][j] = q_i · k_j + pos_bias[h][i][j]   (NO 1/sqrt(d) — T5)
            for (int i = 0; i < T; ++i) {
                const float* qi = q.data() + (size_t)i * kDModel + off;
                float mx = -1e30f;
                for (int j = 0; j < T; ++j) {
                    const float* kj = k.data() + (size_t)j * kDModel + off;
                    double s = 0.0;
                    for (int d = 0; d < kDKv; ++d) s += (double)qi[d] * kj[d];
                    const float sc = (float)s + pos_bias[((size_t)h * T + i) * T + j];
                    scores[(size_t)i * T + j] = sc;
                    if (sc > mx) mx = sc;
                }
                double denom = 0.0;
                for (int j = 0; j < T; ++j) {
                    const float e = std::exp(scores[(size_t)i * T + j] - mx);
                    scores[(size_t)i * T + j] = e;
                    denom += e;
                }
                float* out_i = attn_out.data() + (size_t)i * kDModel + off;
                for (int j = 0; j < T; ++j) {
                    const float wgt = (float)(scores[(size_t)i * T + j] / denom);
                    const float* vj = v.data() + (size_t)j * kDModel + off;
                    for (int d = 0; d < kDKv; ++d) out_i[d] += wgt * vj[d];
                }
            }
        }
        // output projection + residual
        linear_nobias(Wo, kDModel, kDModel, attn_out, T, normed);
        for (size_t i = 0; i < x.size(); ++i) x[i] += normed[i];

        // ── layer.1: feed-forward (pre-norm, ReLU, residual) ─────────────
        const std::vector<float> ln1 = W("text_encoder.encoder.block.%d.layer.1.layer_norm.weight");
        const std::vector<float> Wi = W("text_encoder.encoder.block.%d.layer.1.DenseReluDense.wi.weight");
        const std::vector<float> Wo2 = W("text_encoder.encoder.block.%d.layer.1.DenseReluDense.wo.weight");
        if (ln1.empty() || Wi.empty() || Wo2.empty()) {
            NNOPT_ERROR_FMT("t5_encode_host: missing FF weights at block %d", blk);
            return {};
        }
        normed = x;
        t5_layer_norm(ln1, normed, T);
        linear_nobias(Wi, kDFf, kDModel, normed, T, ff_h);
        for (auto& f : ff_h) f = f > 0.0f ? f : 0.0f;   // ReLU
        linear_nobias(Wo2, kDModel, kDFf, ff_h, T, ff_o);
        for (size_t i = 0; i < x.size(); ++i) x[i] += ff_o[i];

        maybe_dump_block(blk, x, T);
    }

    // ── final_layer_norm ─────────────────────────────────────────────────
    const std::vector<float> fln = weights.get_host_vec("text_encoder.encoder.final_layer_norm.weight");
    if (fln.empty()) { NNOPT_ERROR("t5_encode_host: missing final_layer_norm"); return {}; }
    t5_layer_norm(fln, x, T);

    // ── enc_to_dec_proj: 768 → 1024 with bias ───────────────────────────
    const std::vector<float> Pw = weights.get_host_vec("enc_to_dec_proj.weight");
    const std::vector<float> Pb = weights.get_host_vec("enc_to_dec_proj.bias");
    if (Pw.size() != (size_t)kProjDim * kDModel || Pb.size() != (size_t)kProjDim) {
        NNOPT_ERROR_FMT("t5_encode_host: enc_to_dec_proj shapes wrong (w=%zu b=%zu)", Pw.size(), Pb.size());
        return {};
    }
    std::vector<float> out;
    linear_nobias(Pw, kProjDim, kDModel, x, T, out);
    for (int t = 0; t < T; ++t) {
        float* r = out.data() + (size_t)t * kProjDim;
        for (int o = 0; o < kProjDim; ++o) r[o] += Pb[(size_t)o];
    }
    return out;   // [T, 1024]
}
