// Pipeline shared helpers + the full audio->waveform driver. The individual
// stages live in src/ops/{fbank,speech_encoder,text_decoder,synth_encoder,
// unit_decoder,vocoder}.cpp. Math mirrors oracles/pipeline_ref.cpp.
#include "pipeline.h"
#include "debug_utils.h"

#include <cmath>
#include <cstdio>
#include <string>
#include <vector>
#include <chrono>
#include <fstream>

// Wall-clock since the given start, in ms. NNOPT_TIMING=1 prints a per-stage
// breakdown; otherwise stage timing is silent.
static double ms_since(std::chrono::steady_clock::time_point t0) {
    return std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count();
}

// Debug: write a float32 .bin (NNOPT_DUMP_ENC gates encoder-stage dumps).
void dump_floats_dbg(const std::string& path, const std::vector<float>& v) {
    FILE* f = fopen(path.c_str(), "wb");
    if (!f) return;
    fwrite(v.data(), 4, v.size(), f); fclose(f);
    fprintf(stderr, "DUMP %s (%zu floats)\n", path.c_str(), v.size());
}

// Load a float32 .bin from disk (debug fixtures / side files).
std::vector<float> load_file_floats(const std::string& path) {
    std::vector<float> v;
    FILE* f = fopen(path.c_str(), "rb");
    if (!f) { NNOPT_ERROR_FMT("MISSING %s", path.c_str()); return v; }
    fseek(f, 0, SEEK_END); long n = ftell(f); fseek(f, 0, SEEK_SET);
    v.resize(n / 4); size_t g = fread(v.data(), 4, v.size(), f); (void)g; fclose(f);
    return v;
}

// ===== shared decoder building blocks (text + synth + unit decoders) =====
// `pre` is the FULL device key prefix for the attention module (dir unused).
Tensor Pipeline::mha(const std::string&, const Tensor& xq, int Tq, const Tensor& xkv, int Tk,
                     const std::string& pre, bool causal) {
    auto W = [&](const std::string& k) { return ops_.weight(k); };
    Tensor q = ops_.linear(xq, Tq, Dm, W(pre + "q_proj.weight"), W(pre + "q_proj.bias"), Dm);
    Tensor k = ops_.linear(xkv, Tk, Dm, W(pre + "k_proj.weight"), W(pre + "k_proj.bias"), Dm);
    Tensor v = ops_.linear(xkv, Tk, Dm, W(pre + "v_proj.weight"), W(pre + "v_proj.bias"), Dm);
    Tensor ctx = ops_.attention(q, k, v, Tq, Tk, H, Dk, causal);
    return ops_.linear(ctx, Tq, Dm, W(pre + "out_proj.weight"), W(pre + "out_proj.bias"), Dm);
}

// Standard fairseq pre-norm decoder forward (GELU FFN, tied output). dir = full
// decoder key prefix (e.g. "model.target_letter_decoder."). Returns host [T*768].
Tensor Pipeline::decoder_features(const std::string& dir, const std::vector<int>& seq,
                                  const Tensor& mem, int Tmem, int nlayers) {
    auto W = [&](const std::string& k) { return ops_.weight(k); };
    int T = (int)seq.size();
    cl_mem ids = ops_.upload_ints(seq);
    Tensor x = ops_.embed_scale_pos(ids, W(dir + "embed_tokens.weight"), T, Dm, std::sqrt((float)Dm), 2);
    for (int L = 0; L < nlayers; ++L) {
        std::string pre = dir + "layers." + std::to_string(L) + ".";
        Tensor sn = ops_.layernorm(x, T, Dm, W(pre + "self_attn_layer_norm.weight"), W(pre + "self_attn_layer_norm.bias"));
        Tensor a1 = mha("", sn, T, sn, T, pre + "self_attn.", true);
        ops_.axpy(x, a1, 1.0f);
        Tensor cn = ops_.layernorm(x, T, Dm, W(pre + "encoder_attn_layer_norm.weight"), W(pre + "encoder_attn_layer_norm.bias"));
        Tensor a2 = mha("", cn, T, mem, Tmem, pre + "encoder_attn.", false);
        ops_.axpy(x, a2, 1.0f);
        Tensor fn = ops_.layernorm(x, T, Dm, W(pre + "final_layer_norm.weight"), W(pre + "final_layer_norm.bias"));
        Tensor h1 = ops_.linear(fn, T, Dm, W(pre + "fc1.weight"), W(pre + "fc1.bias"), 4096);
        ops_.act(h1, ACT_GELU);
        Tensor h2 = ops_.linear(h1, T, 4096, W(pre + "fc2.weight"), W(pre + "fc2.bias"), Dm);
        ops_.axpy(x, h2, 1.0f);
    }
    return ops_.layernorm(x, T, Dm, W(dir + "layer_norm.weight"), W(dir + "layer_norm.bias"));
}

// Logits [vocab] (fp16, on GPU) for the final position; output_projection tied
// to embed_tokens.
Tensor Pipeline::decoder_logits_last(const std::string& dir, const std::vector<int>& seq,
                                     const Tensor& mem, int Tmem, int nlayers, int vocab) {
    Tensor feat = decoder_features(dir, seq, mem, Tmem, nlayers);
    int T = (int)seq.size();
    Tensor last = ops_.copy_region(feat, (T - 1) * Dm, Dm);
    return ops_.linear(last, 1, Dm, ops_.weight(dir + "embed_tokens.weight"), nullptr, vocab);
}

// ===================== KV-cached incremental decode =====================
KVCache Pipeline::make_cache(const std::string& dir, const Tensor& mem, int Tmem, int nlayers,
                             int vocab, int max_len) {
    auto W = [&](const std::string& k) { return ops_.weight(k); };
    KVCache c;
    c.dir = dir; c.nlayers = nlayers; c.Tmem = Tmem; c.max_len = max_len; c.vocab = vocab; c.len = 0;
    for (int L = 0; L < nlayers; ++L) {
        std::string pre = dir + "layers." + std::to_string(L) + ".encoder_attn.";
        // cross-attn K/V depend only on the (fixed) memory — compute once.
        Tensor k = ops_.linear(mem, Tmem, Dm, W(pre + "k_proj.weight"), W(pre + "k_proj.bias"), Dm);
        Tensor v = ops_.linear(mem, Tmem, Dm, W(pre + "v_proj.weight"), W(pre + "v_proj.bias"), Dm);
        c.ck.push_back(k.buf); c.cv.push_back(v.buf);
        c.sk.push_back(ops_.alloc(max_len * Dm).buf);
        c.sv.push_back(ops_.alloc(max_len * Dm).buf);
    }
    return c;
}

// One incremental decode step. Returns fp16 logits [vocab] for predicting the
// token at position pos+1. Self K/V for `token` are appended to the cache at pos.
Tensor Pipeline::decode_step(KVCache& c, int token, int pos) {
    return decode_step_ids(c, ops_.upload_ints(std::vector<int>{token}), pos);
}

// Same as decode_step but the token comes from a device int buffer (ids[0]) — lets
// the greedy loop chain argmax→embed entirely on-GPU (no per-step host readback).
Tensor Pipeline::decode_step_ids(KVCache& c, cl_mem ids, int pos) {
    auto W = [&](const std::string& k) { return ops_.weight(k); };
    Tensor x = ops_.embed_scale_pos(ids, W(c.dir + "embed_tokens.weight"), 1, Dm, std::sqrt((float)Dm), 2 + pos);
    for (int L = 0; L < c.nlayers; ++L) {
        std::string pre = c.dir + "layers." + std::to_string(L) + ".";
        // self-attention (causal): new token attends to all cached positions.
        Tensor sn = ops_.layernorm(x, 1, Dm, W(pre + "self_attn_layer_norm.weight"), W(pre + "self_attn_layer_norm.bias"));
        // (Fused QKV — 3 GEMVs→1 + slice — was tried and is neutral/worse: the 3 slice
        // copies offset the saved launches, because the M=1 decode is weight-BW-bound,
        // not launch-bound. Kept the 3 separate projections.)
        Tensor q = ops_.linear(sn, 1, Dm, W(pre + "self_attn.q_proj.weight"), W(pre + "self_attn.q_proj.bias"), Dm);
        Tensor kn = ops_.linear(sn, 1, Dm, W(pre + "self_attn.k_proj.weight"), W(pre + "self_attn.k_proj.bias"), Dm);
        Tensor vn = ops_.linear(sn, 1, Dm, W(pre + "self_attn.v_proj.weight"), W(pre + "self_attn.v_proj.bias"), Dm);
        ops_.copy_into(c.sk[L], pos * Dm, kn, Dm);
        ops_.copy_into(c.sv[L], pos * Dm, vn, Dm);
        Tensor kc{c.sk[L], (pos + 1) * Dm}, vc{c.sv[L], (pos + 1) * Dm};
        Tensor a1 = ops_.attention(q, kc, vc, 1, pos + 1, H, Dk, false);
        a1 = ops_.linear(a1, 1, Dm, W(pre + "self_attn.out_proj.weight"), W(pre + "self_attn.out_proj.bias"), Dm);
        ops_.axpy(x, a1, 1.0f);
        // cross-attention to the precomputed memory K/V.
        Tensor cn = ops_.layernorm(x, 1, Dm, W(pre + "encoder_attn_layer_norm.weight"), W(pre + "encoder_attn_layer_norm.bias"));
        Tensor q2 = ops_.linear(cn, 1, Dm, W(pre + "encoder_attn.q_proj.weight"), W(pre + "encoder_attn.q_proj.bias"), Dm);
        Tensor kx{c.ck[L], c.Tmem * Dm}, vx{c.cv[L], c.Tmem * Dm};
        Tensor a2 = ops_.attention(q2, kx, vx, 1, c.Tmem, H, Dk, false);
        a2 = ops_.linear(a2, 1, Dm, W(pre + "encoder_attn.out_proj.weight"), W(pre + "encoder_attn.out_proj.bias"), Dm);
        ops_.axpy(x, a2, 1.0f);
        // FFN (GELU).
        Tensor fn = ops_.layernorm(x, 1, Dm, W(pre + "final_layer_norm.weight"), W(pre + "final_layer_norm.bias"));
        Tensor h1 = ops_.linear(fn, 1, Dm, W(pre + "fc1.weight"), W(pre + "fc1.bias"), 4096);
        ops_.act(h1, ACT_GELU);
        Tensor h2 = ops_.linear(h1, 1, 4096, W(pre + "fc2.weight"), W(pre + "fc2.bias"), Dm);
        ops_.axpy(x, h2, 1.0f);
    }
    c.len = pos + 1;
    Tensor xf = ops_.layernorm(x, 1, Dm, W(c.dir + "layer_norm.weight"), W(c.dir + "layer_norm.bias"));
    return ops_.linear(xf, 1, Dm, W(c.dir + "embed_tokens.weight"), nullptr, c.vocab);
}

// Batched beam decode: one M=B GEMM per projection, one cross-attention call
// (shared memory K/V), per-beam self-attention. Math is identical to B separate
// decode_step calls; only the work is batched. Returns logits [B*vocab].
Tensor Pipeline::decode_step_batch(const std::string& dir, int B, const std::vector<int>& toks,
                                   int step, int nlayers, int Tmem, int vocab,
                                   const std::vector<cl_mem>& ck, const std::vector<cl_mem>& cv,
                                   const std::vector<cl_mem>& sk, const std::vector<cl_mem>& sv) {
    auto W = [&](const std::string& k) { return ops_.weight(k); };
    cl_mem ids = ops_.upload_ints(toks);
    // All B beams are at the SAME position `step` (rows are beams, not positions).
    Tensor x = ops_.embed_scale_pos(ids, W(dir + "embed_tokens.weight"), B, Dm, std::sqrt((float)Dm), 2 + step, /*pos_stride=*/0);
    for (int L = 0; L < nlayers; ++L) {
        std::string pre = dir + "layers." + std::to_string(L) + ".";
        // self-attention: projections batched (M=B), attention per-beam (own cache).
        Tensor sn = ops_.layernorm(x, B, Dm, W(pre + "self_attn_layer_norm.weight"), W(pre + "self_attn_layer_norm.bias"));
        Tensor q = ops_.linear(sn, B, Dm, W(pre + "self_attn.q_proj.weight"), W(pre + "self_attn.q_proj.bias"), Dm);
        Tensor k = ops_.linear(sn, B, Dm, W(pre + "self_attn.k_proj.weight"), W(pre + "self_attn.k_proj.bias"), Dm);
        Tensor v = ops_.linear(sn, B, Dm, W(pre + "self_attn.v_proj.weight"), W(pre + "self_attn.v_proj.bias"), Dm);
        Tensor ctx = ops_.alloc(B * Dm);
        Tensor qb = ops_.alloc(Dm);  // reused across beams (in-order queue ⇒ safe); avoids per-beam alloc churn
        for (int b = 0; b < B; ++b) {
            int sidx = b * nlayers + L;
            // append beam b's K/V to its cache, and gather its q — all via offset copies, NO alloc.
            ops_.copy_into_off(sk[sidx], step * Dm, k.buf, b * Dm, Dm);
            ops_.copy_into_off(sv[sidx], step * Dm, v.buf, b * Dm, Dm);
            ops_.copy_into_off(qb.buf, 0, q.buf, b * Dm, Dm);
            Tensor kc{sk[sidx], (step + 1) * Dm}, vc{sv[sidx], (step + 1) * Dm};
            Tensor ab = ops_.attention(qb, kc, vc, 1, step + 1, H, Dk, false);
            ops_.copy_into(ctx.buf, b * Dm, ab, Dm);
        }
        ctx = ops_.linear(ctx, B, Dm, W(pre + "self_attn.out_proj.weight"), W(pre + "self_attn.out_proj.bias"), Dm);
        ops_.axpy(x, ctx, 1.0f);
        // cross-attention: all beams share the same memory K/V -> one batched call (Tq=B).
        Tensor cn = ops_.layernorm(x, B, Dm, W(pre + "encoder_attn_layer_norm.weight"), W(pre + "encoder_attn_layer_norm.bias"));
        Tensor q2 = ops_.linear(cn, B, Dm, W(pre + "encoder_attn.q_proj.weight"), W(pre + "encoder_attn.q_proj.bias"), Dm);
        Tensor kx{ck[L], Tmem * Dm}, vx{cv[L], Tmem * Dm};
        Tensor cc = ops_.attention(q2, kx, vx, B, Tmem, H, Dk, false);
        cc = ops_.linear(cc, B, Dm, W(pre + "encoder_attn.out_proj.weight"), W(pre + "encoder_attn.out_proj.bias"), Dm);
        ops_.axpy(x, cc, 1.0f);
        // FFN batched.
        Tensor fn = ops_.layernorm(x, B, Dm, W(pre + "final_layer_norm.weight"), W(pre + "final_layer_norm.bias"));
        Tensor h1 = ops_.linear(fn, B, Dm, W(pre + "fc1.weight"), W(pre + "fc1.bias"), 4096);
        ops_.act(h1, ACT_GELU);
        Tensor h2 = ops_.linear(h1, B, 4096, W(pre + "fc2.weight"), W(pre + "fc2.bias"), Dm);
        ops_.axpy(x, h2, 1.0f);
    }
    Tensor xf = ops_.layernorm(x, B, Dm, W(dir + "layer_norm.weight"), W(dir + "layer_norm.bias"));
    return ops_.linear(xf, B, Dm, W(dir + "embed_tokens.weight"), nullptr, vocab);  // [B, vocab]
}

// =============================== full pipeline ===============================
void Pipeline::run(const std::vector<float>& audio, std::vector<float>& units_out,
                   std::vector<float>& waveform_out) {
    const bool timing = []() { const char* e = std::getenv("NNOPT_TIMING"); return e && *e && *e != '0'; }();
    // NNOPT_HWUTIL: accurate per-stage GPU-busy via the Adreno kgsl gpubusy HW counter
    // (delta-since-last-read busy/total cycles). clFinish before each read so each stage's
    // window covers exactly its GPU work — no profiler blind spot (counts CLBlast helpers too).
    const bool hwutil = []{ const char* e = std::getenv("NNOPT_HWUTIL"); return e && *e && *e != '0'; }();
    auto rd_busy = [](long long& b, long long& tot){ std::ifstream f("/sys/class/kgsl/kgsl-3d0/gpubusy"); if(!f){b=tot=0;return;} b=tot=0; f>>b>>tot; };
    auto hw = [&](const char* name){ if(!hwutil) return; clFinish(ops_.cl().queue()); long long b=0,tot=0; rd_busy(b,tot);
        fprintf(stderr, "HWUTIL %-12s busy=%lld total=%lld -> %.1f%%\n", name, b, tot, tot>0?100.0*(double)b/(double)tot:0.0); };
    auto t = std::chrono::steady_clock::now();
    double t_fb = 0, t_enc = 0, t_text = 0, t_mt = 0, t_synth = 0, t_unit = 0, t_voc = 0;
    auto lap = [&](double& slot) { slot = ms_since(t); t = std::chrono::steady_clock::now(); };
    if (hwutil) { clFinish(ops_.cl().queue()); long long b=0,tot=0; rd_busy(b,tot); }  // reset window

    int nframes = 0;
    auto fb = fbank(audio, nframes);
    lap(t_fb); ops_.prof_mark("fbank"); hw("fbank");
    fprintf(stderr, "[2] fbank -> %d frames\n", nframes);
    int Tenc = 0;
    auto enc = encoder(fb, nframes, Tenc);
    lap(t_enc); ops_.prof_mark("encoder"); hw("encoder");
    fprintf(stderr, "[3] encoder -> [%d, 768]\n", Tenc);

    auto hypo = text_beam_search(enc, Tenc);
    lap(t_text); ops_.prof_mark("text_beam"); hw("text_beam");
    fprintf(stderr, "[4] text beam-5 -> hypo (%zu):", hypo.size());
    for (int t2 : hypo) fprintf(stderr, " %d", t2);
    fprintf(stderr, "\n");

    std::vector<int> prev_mt{2};
    for (size_t i = 1; i + 1 < hypo.size(); ++i) prev_mt.push_back(hypo[i]);
    int Tm = (int)prev_mt.size();

    Tensor encMem = ops_.upload(enc);
    auto mt_hidden = ops_.download(decoder_features("model.target_letter_decoder.", prev_mt, encMem, Tenc, 4));
    lap(t_mt); ops_.prof_mark("mt_feat"); hw("mt_feat");

    auto t2u_mem = synth_encoder(mt_hidden, Tm);
    lap(t_synth); ops_.prof_mark("synth"); hw("synth");
    fprintf(stderr, "[5a] synthesizer -> [%d, 768]\n", Tm);

    auto unit_toks = unit_greedy_decode(t2u_mem, Tm);
    lap(t_unit); ops_.prof_mark("unit_greedy"); hw("unit_greedy");
    units_out.clear();
    for (size_t i = 2; i < unit_toks.size(); ++i) units_out.push_back((float)(unit_toks[i] - 4));
    fprintf(stderr, "[5b] unit decode -> %zu units\n", units_out.size());

    waveform_out = vocoder(units_out);
    lap(t_voc); ops_.prof_mark("vocoder"); hw("vocoder");
    fprintf(stderr, "[1] vocoder -> %zu samples\n", waveform_out.size());

    if (timing) {
        double total = t_fb + t_enc + t_text + t_mt + t_synth + t_unit + t_voc;
        fprintf(stderr,
                "TIMING(ms): fbank=%.0f encoder=%.0f text_beam=%.0f mt_feat=%.0f synth=%.0f unit_greedy=%.0f vocoder=%.0f TOTAL=%.0f\n",
                t_fb, t_enc, t_text, t_mt, t_synth, t_unit, t_voc, total);
    }
}

void Pipeline::run_from_encoder(const std::vector<float>& enc, int Tenc,
                                std::vector<float>& units_out, std::vector<float>& waveform_out) {
    auto hypo = text_beam_search(enc, Tenc);
    fprintf(stderr, "[4] text beam-5 -> hypo (%zu):", hypo.size());
    for (int t2 : hypo) fprintf(stderr, " %d", t2);
    fprintf(stderr, "\n");
    std::vector<int> prev_mt{2};
    for (size_t i = 1; i + 1 < hypo.size(); ++i) prev_mt.push_back(hypo[i]);
    int Tm = (int)prev_mt.size();
    Tensor encMem = ops_.upload(enc);
    auto mt_hidden = ops_.download(decoder_features("model.target_letter_decoder.", prev_mt, encMem, Tenc, 4));
    auto t2u_mem = synth_encoder(mt_hidden, Tm);
    auto unit_toks = unit_greedy_decode(t2u_mem, Tm);
    units_out.clear();
    for (size_t i = 2; i < unit_toks.size(); ++i) units_out.push_back((float)(unit_toks[i] - 4));
    fprintf(stderr, "[5b] unit decode -> %zu units\n", units_out.size());
    waveform_out = vocoder(units_out);
}
