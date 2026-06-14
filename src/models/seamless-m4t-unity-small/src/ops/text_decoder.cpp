// Stage 4: text decoder generation. Faithful fairseq SequenceGenerator + BeamSearch
// (beam=5), now KV-cached. Cross-attn K/V are computed once (shared by all beams);
// each beam keeps its own self-attn K/V cache. On candidate selection the per-beam
// caches are REORDERED (child inherits parent's cache) via a double-buffered pool.
// Arithmetic on GPU: per-beam decode_step, log_softmax, masking, score accumulation,
// top-2*beam selection (iterative argmax). Hypothesis bookkeeping is scalar.
#include "pipeline.h"
#include <algorithm>
#include <string>
#include <vector>
#include <utility>
#include <chrono>
#include <cstdio>
#include <cstdlib>

std::vector<int> Pipeline::text_beam_search(const std::vector<float>& encv, int Tenc) {
    // NNOPT_TBPROF: attribute the per-step text-beam wall to its sub-phases.
    // clFinish() before each timestamp so async GPU work is charged to the right
    // phase (perturbs absolute timing slightly; reveals the idle culprit).
    const bool tbprof = []{ const char* e = std::getenv("NNOPT_TBPROF"); return e && *e && *e != '0'; }();
    double tp_decode = 0, tp_score = 0, tp_topk = 0, tp_kv = 0; int tp_steps = 0;
    auto clk = []{ return std::chrono::steady_clock::now(); };
    auto fin = [&]{ if (tbprof) clFinish(ops_.cl().queue()); return clk(); };
    auto add = [&](double& slot, std::chrono::steady_clock::time_point a) {
        if (tbprof) slot += std::chrono::duration<double, std::milli>(clk() - a).count();
    };
    const std::string dir = "model.target_letter_decoder.";
    const int nlayers = 4, vocab = 20005, beam = 5, eos = 2, pad = 1, min_len = 1, max_len = 200;
    const int CACHE_MAX = 200;  // bound on hypothesis length for the self-attn cache
    const std::vector<int> prefix{text_prefix_};  // target-language text lang token
    const float NEG = -1e30f;
    auto W = [&](const std::string& k) { return ops_.weight(k); };

    Tensor mem = ops_.upload(encv);
    // Shared cross-attn K/V (one set; every beam/step reuses it).
    std::vector<cl_mem> ck(nlayers), cv(nlayers);
    for (int L = 0; L < nlayers; ++L) {
        std::string pre = dir + "layers." + std::to_string(L) + ".encoder_attn.";
        ck[L] = ops_.linear(mem, Tenc, Dm, W(pre + "k_proj.weight"), W(pre + "k_proj.bias"), Dm).buf;
        cv[L] = ops_.linear(mem, Tenc, Dm, W(pre + "v_proj.weight"), W(pre + "v_proj.bias"), Dm).buf;
    }
    // Double-buffered self-attn K/V pools: sk[pool][beam*nlayers + L].
    std::vector<std::vector<cl_mem>> sk(2), sv(2);
    for (int p = 0; p < 2; ++p) {
        sk[p].resize(beam * nlayers); sv[p].resize(beam * nlayers);
        for (int i = 0; i < beam * nlayers; ++i) { sk[p][i] = ops_.alloc(CACHE_MAX * Dm).buf; sv[p][i] = ops_.alloc(CACHE_MAX * Dm).buf; }
    }
    int cur = 0;
    int base = ops_.mark();

    int cand_size = 2 * beam;
    std::vector<std::vector<int>> tokens(beam, std::vector<int>{eos});
    std::vector<std::vector<float>> shist(beam, std::vector<float>(max_len + 1, 0.f));
    std::vector<char> ignore(beam, 0);
    std::vector<std::pair<std::vector<int>, float>> finalized;

    for (int step = 0; step <= max_len; ++step) {
        int nb = (step == 0) ? 1 : (int)tokens.size();
        int force = (step >= max_len) ? eos : (step < (int)prefix.size() ? prefix[step] : -1);
        bool supeos = (step < max_len) && (step >= (int)prefix.size()) && (step < min_len);

        int m = ops_.mark();
        cl_mem scored = ops_.alloc_f32(nb * vocab);
        // All nb beams decoded in one batched pass (matmuls M=nb, one cross-attn call).
        std::vector<int> toks_b(nb);
        for (int b = 0; b < nb; ++b) toks_b[b] = tokens[b].back();
        auto _t0 = fin();
        Tensor logits = decode_step_batch(dir, nb, toks_b, step, nlayers, Tenc, vocab, ck, cv, sk[cur], sv[cur]);
        auto _t1 = fin(); add(tp_decode, _t0);
        for (int b = 0; b < nb; ++b) {
            Tensor lgb = ops_.copy_region(logits, b * vocab, vocab);
            ops_.log_softmax_region(lgb, scored, b * vocab, vocab);
            ops_.mask_region(scored, b * vocab, vocab, pad, /*unk*/ -1, force, supeos, eos);
            float basesc = (step == 0) ? 0.f : shist[b][step - 1];
            ops_.add_scalar_region(scored, b * vocab, vocab, basesc);
        }
        // GPU top-k: cand_size rounds of argmax+mask run entirely on-GPU, then ONE bulk
        // readback — replaces cand_size blocking argmax readbacks per beam step (the
        // dominant text-beam overhead: ~cand_size GPU drains/step → 1).
        auto _t2 = fin(); add(tp_score, _t1);
        cl_mem ci = ops_.alloc_ints(cand_size);
        cl_mem cv = ops_.alloc_f32(cand_size);
        ops_.topk_dev(scored, nb * vocab, cand_size, ci, cv, NEG);
        std::vector<int> craw(cand_size);
        ops_.download_ints(ci, 0, cand_size, craw.data());
        std::vector<float> cscore = ops_.download_f32(cv, cand_size);
        add(tp_topk, _t2); auto _t3 = clk();
        std::vector<int> cidx, cbeam;
        for (int c = 0; c < cand_size; ++c) { cbeam.push_back(craw[c] / vocab); cidx.push_back(craw[c] % vocab); }
        ops_.release_to(m);

        // ---- scalar hypothesis bookkeeping (control flow, not arithmetic) ----
        std::vector<char> emask(cand_size, 0);
        for (int i = 0; i < cand_size; ++i) emask[i] = (cidx[i] == eos && cscore[i] > NEG) ? 1 : 0;
        for (int i = 0; i < beam; ++i) if (ignore[i]) emask[i] = 0;
        for (int i = 0; i < beam; ++i)
            if (emask[i] && (int)finalized.size() < beam) {
                int b = cbeam[i];
                std::vector<int> seq(tokens[b].begin() + 1, tokens[b].end()); seq.push_back(eos);
                finalized.push_back({seq, cscore[i] / (float)(step + 1)});
            }
        if ((int)finalized.size() >= beam || step == max_len) break;
        std::vector<char> em(cand_size, 0);
        for (int i = 0; i < beam; ++i) em[i] = !((!ignore[i]) && (!emask[i]));
        for (int i = beam; i < cand_size; ++i) em[i] = emask[i];
        std::vector<int> order(cand_size); for (int i = 0; i < cand_size; ++i) order[i] = i;
        std::vector<long> am(cand_size);
        for (int i = 0; i < cand_size; ++i) am[i] = (long)(em[i] ? 1 : 0) * cand_size + i;
        std::partial_sort(order.begin(), order.begin() + beam, order.end(), [&](int a, int b) { return am[a] < am[b]; });

        // Reorder per-beam KV caches: new beam j inherits parent cbeam[order[j]]'s cache.
        int nxt = 1 - cur;
        int copy_n = (step + 1) * Dm;  // cache holds positions 0..step
        std::vector<std::vector<int>> nt(beam); std::vector<std::vector<float>> nh(beam); std::vector<char> nig(beam, 0);
        for (int j = 0; j < beam; ++j) {
            int c = order[j], b = cbeam[c];
            nig[j] = (am[c] >= cand_size) ? 1 : 0;
            nt[j] = tokens[b]; nt[j].push_back(cidx[c]);
            nh[j] = shist[b]; nh[j][step] = cscore[c];
            for (int L = 0; L < nlayers; ++L) {
                Tensor pk{sk[cur][b * nlayers + L], copy_n}; ops_.copy_into(sk[nxt][j * nlayers + L], 0, pk, copy_n);
                Tensor pv{sv[cur][b * nlayers + L], copy_n}; ops_.copy_into(sv[nxt][j * nlayers + L], 0, pv, copy_n);
            }
        }
        if (tbprof) { clFinish(ops_.cl().queue()); add(tp_kv, _t3); tp_steps++; }
        cur = nxt;
        tokens = std::move(nt); shist = std::move(nh); ignore = std::move(nig);
    }
    if (tbprof && tp_steps > 0)
        fprintf(stderr, "TBPROF(ms over %d steps): decode=%.0f score=%.0f topk+readback=%.0f kv_reorder=%.0f | per-step decode=%.1f score=%.1f topk=%.1f kv=%.1f\n",
                tp_steps, tp_decode, tp_score, tp_topk, tp_kv,
                tp_decode/tp_steps, tp_score/tp_steps, tp_topk/tp_steps, tp_kv/tp_steps);
    ops_.release_to(base);
    std::sort(finalized.begin(), finalized.end(), [](const std::pair<std::vector<int>,float>& a, const std::pair<std::vector<int>,float>& b) { return a.second > b.second; });
    return finalized.empty() ? std::vector<int>{} : finalized[0].first;
}
