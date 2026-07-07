#include "t5_encoder.h"
#include "t5_tokenizer.h"
#include "weights.h"
#include "load_bin.h"

#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>

namespace {

constexpr int DIM = T5CondEncoder::DIM;        // 768
constexpr int S = T5CondEncoder::MAX_TOKENS;   // 64
constexpr int LAYERS = 12;
constexpr int HEADS = 12;
constexpr int DK = 64;                          // DIM / HEADS
constexpr int FF = 3072;
constexpr float LN_EPS = 1e-6f;

// y[M,N] = x[M,K] @ W[N,K]^T   (torch Linear layout, no bias anywhere in T5)
void matmul_wt(const float* x, const float* w, float* y, int M, int N, int K) {
    for (int i = 0; i < M; i++) {
        const float* xi = x + (size_t)i * K;
        float* yi = y + (size_t)i * N;
        for (int o = 0; o < N; o++) {
            const float* wo = w + (size_t)o * K;
            float acc = 0.0f;
            for (int k = 0; k < K; k++) acc += xi[k] * wo[k];
            yi[o] = acc;
        }
    }
}

// T5LayerNorm: x * rsqrt(mean(x^2) + eps) * weight — no mean subtract, no bias
void rms_norm(const float* x, const float* w, float* y, int rows, int dim) {
    for (int i = 0; i < rows; i++) {
        const float* xi = x + (size_t)i * dim;
        float ss = 0.0f;
        for (int d = 0; d < dim; d++) ss += xi[d] * xi[d];
        const float inv = 1.0f / std::sqrt(ss / dim + LN_EPS);
        float* yi = y + (size_t)i * dim;
        for (int d = 0; d < dim; d++) yi[d] = xi[d] * inv * w[d];
    }
}

// HF T5Attention._relative_position_bucket, bidirectional, 32 buckets, max 128
int rel_bucket(int rel /* = key_pos - query_pos */) {
    int bucket = 0;
    int nb = 16;                       // num_buckets / 2 (bidirectional)
    if (rel > 0) bucket += nb;
    int n = rel < 0 ? -rel : rel;
    const int max_exact = nb / 2;      // 8
    if (n < max_exact) return bucket + n;
    int large = max_exact +
        (int)(std::log((float)n / max_exact) / std::log(128.0f / max_exact) *
              (nb - max_exact));
    if (large > nb - 1) large = nb - 1;
    return bucket + large;
}

}  // namespace

struct T5CondEncoder::Impl {
    Weights weights;
    T5Tokenizer tokenizer;
    std::vector<float> seconds_table;   // [257 * 768] fp32
    std::vector<float> pos_bias;        // [HEADS * S * S], built once at load
};

bool T5CondEncoder::load(const std::string& weights_bin, const std::string& weights_meta,
                         const std::string& tokenizer_bin, const std::string& seconds_table_bin) {
    impl_ = new Impl();
    // null CL context: T5 stays on the CPU, Weights just mmaps + decodes
    if (!impl_->weights.load(weights_bin, weights_meta, nullptr)) return false;
    if (!impl_->tokenizer.load(tokenizer_bin)) return false;
    impl_->seconds_table = load_float_bin(seconds_table_bin);
    if (impl_->seconds_table.size() != 257u * DIM) {
        std::fprintf(stderr, "ERROR: seconds_table.bin wrong size %zu (want %d)\n",
                     impl_->seconds_table.size(), 257 * DIM);
        return false;
    }
    // Relative position bias for fixed S=64 — one table, shared by all blocks.
    std::vector<float> table = impl_->weights.get_host_vec(
        "encoder.block.0.layer.0.SelfAttention.relative_attention_bias.weight"); // [32, HEADS]
    if (table.size() != 32u * HEADS) {
        std::fprintf(stderr, "ERROR: relative_attention_bias missing/wrong size %zu\n", table.size());
        return false;
    }
    impl_->pos_bias.assign((size_t)HEADS * S * S, 0.0f);
    for (int i = 0; i < S; i++)
        for (int j = 0; j < S; j++) {
            const int b = rel_bucket(j - i);
            for (int h = 0; h < HEADS; h++)
                impl_->pos_bias[((size_t)h * S + i) * S + j] = table[(size_t)b * HEADS + h];
        }
    return true;
}

bool T5CondEncoder::compute(const std::string& prompt, int seconds_total,
                            std::vector<float>& cross_out, std::vector<float>& global_out,
                            int* n_real_tokens) {
    if (!impl_) return false;
    Weights& W = impl_->weights;

    int n_real = 0;
    std::vector<int32_t> ids = impl_->tokenizer.encode(prompt, S, &n_real);
    if (n_real_tokens) *n_real_tokens = n_real;

    // NNOPT_T5_DUMP_DIR=<dir>: write per-block hidden states for cosine
    // pairing against reference/t5_layers/ (module_path-keyed, per the
    // localizer identity rule).
    const char* dumpdir = std::getenv("NNOPT_T5_DUMP_DIR");
    auto dump = [&](const std::string& tag, const float* p, size_t n) {
        if (!dumpdir) return;
        std::string path = std::string(dumpdir) + "/" + tag + ".bin";
        FILE* fp = std::fopen(path.c_str(), "wb");
        if (fp) { std::fwrite(p, sizeof(float), n, fp); std::fclose(fp); }
    };
    if (dumpdir) {
        FILE* fp = std::fopen((std::string(dumpdir) + "/input_ids.bin").c_str(), "wb");
        if (fp) { std::fwrite(ids.data(), sizeof(int32_t), ids.size(), fp); std::fclose(fp); }
    }

    // embed
    std::vector<float> E = W.get_host_vec("shared.weight");   // [32128, DIM]
    if (E.empty()) return false;
    std::vector<float> x((size_t)S * DIM);
    for (int i = 0; i < S; i++)
        std::memcpy(&x[(size_t)i * DIM], &E[(size_t)ids[i] * DIM], DIM * sizeof(float));
    E.clear(); E.shrink_to_fit();
    dump("embed_tokens_out", x.data(), x.size());

    std::vector<float> ln((size_t)S * DIM), q((size_t)S * DIM), k((size_t)S * DIM),
        v((size_t)S * DIM), ctx((size_t)S * DIM), tmp((size_t)S * DIM),
        ff1((size_t)S * FF), scores((size_t)S * S);

    char name[160];
    auto tensor = [&](int layer, const char* suffix) {
        std::snprintf(name, sizeof(name), "encoder.block.%d.%s", layer, suffix);
        return W.get_host_vec(name);
    };

    for (int L = 0; L < LAYERS; L++) {
        // ── self-attention sublayer ──
        std::vector<float> lnw = tensor(L, "layer.0.layer_norm.weight");
        rms_norm(x.data(), lnw.data(), ln.data(), S, DIM);
        {
            std::vector<float> wq = tensor(L, "layer.0.SelfAttention.q.weight");
            matmul_wt(ln.data(), wq.data(), q.data(), S, DIM, DIM);
            std::vector<float> wk = tensor(L, "layer.0.SelfAttention.k.weight");
            matmul_wt(ln.data(), wk.data(), k.data(), S, DIM, DIM);
            std::vector<float> wv = tensor(L, "layer.0.SelfAttention.v.weight");
            matmul_wt(ln.data(), wv.data(), v.data(), S, DIM, DIM);
        }
        for (int h = 0; h < HEADS; h++) {
            const int off = h * DK;
            const float* bias_h = &impl_->pos_bias[(size_t)h * S * S];
            for (int i = 0; i < S; i++) {
                float* sc = &scores[(size_t)i * S];
                const float* qi = &q[(size_t)i * DIM + off];
                // NO 1/sqrt(dk) — T5 scores are unscaled by design
                float mx = -1e30f;
                for (int j = 0; j < S; j++) {
                    const float* kj = &k[(size_t)j * DIM + off];
                    float acc = 0.0f;
                    for (int d = 0; d < DK; d++) acc += qi[d] * kj[d];
                    acc += bias_h[(size_t)i * S + j];
                    if (j >= n_real) acc = -1e9f;   // mask padded KEYS only
                    sc[j] = acc;
                    if (acc > mx) mx = acc;
                }
                float sum = 0.0f;
                for (int j = 0; j < S; j++) { sc[j] = std::exp(sc[j] - mx); sum += sc[j]; }
                const float inv = 1.0f / sum;
                float* ci = &ctx[(size_t)i * DIM + off];
                for (int d = 0; d < DK; d++) ci[d] = 0.0f;
                for (int j = 0; j < S; j++) {
                    const float p = sc[j] * inv;
                    const float* vj = &v[(size_t)j * DIM + off];
                    for (int d = 0; d < DK; d++) ci[d] += p * vj[d];
                }
            }
        }
        {
            std::vector<float> wo = tensor(L, "layer.0.SelfAttention.o.weight");
            matmul_wt(ctx.data(), wo.data(), tmp.data(), S, DIM, DIM);
        }
        for (size_t i = 0; i < x.size(); i++) x[i] += tmp[i];

        // ── feed-forward sublayer (plain ReLU, wi/wo) ──
        lnw = tensor(L, "layer.1.layer_norm.weight");
        rms_norm(x.data(), lnw.data(), ln.data(), S, DIM);
        {
            std::vector<float> wi = tensor(L, "layer.1.DenseReluDense.wi.weight");
            matmul_wt(ln.data(), wi.data(), ff1.data(), S, FF, DIM);
        }
        for (size_t i = 0; i < ff1.size(); i++) if (ff1[i] < 0.0f) ff1[i] = 0.0f;
        {
            std::vector<float> wo2 = tensor(L, "layer.1.DenseReluDense.wo.weight");
            matmul_wt(ff1.data(), wo2.data(), tmp.data(), S, DIM, FF);
        }
        for (size_t i = 0; i < x.size(); i++) x[i] += tmp[i];
        {
            char tag[48];
            std::snprintf(tag, sizeof(tag), "encoder_block_%d_out", L);
            dump(tag, x.data(), x.size());
        }
    }

    // final norm, then the conditioner wrapper: zero pad rows, append seconds
    std::vector<float> fw = W.get_host_vec("encoder.final_layer_norm.weight");
    rms_norm(x.data(), fw.data(), ln.data(), S, DIM);
    dump("encoder_final_layer_norm_out", ln.data(), ln.size());
    for (int i = n_real; i < S; i++)
        std::memset(&ln[(size_t)i * DIM], 0, DIM * sizeof(float));

    if (seconds_total < 0) seconds_total = 0;
    if (seconds_total > 256) seconds_total = 256;
    const float* sec = &impl_->seconds_table[(size_t)seconds_total * DIM];

    cross_out.resize((size_t)(S + 1) * DIM);
    std::memcpy(cross_out.data(), ln.data(), (size_t)S * DIM * sizeof(float));
    std::memcpy(&cross_out[(size_t)S * DIM], sec, DIM * sizeof(float));
    global_out.assign(sec, sec + DIM);
    return true;
}
