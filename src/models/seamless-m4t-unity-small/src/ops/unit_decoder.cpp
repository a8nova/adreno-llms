// Stage 5b: T2U unit decoder generation. Greedy (beam=1), unk/pad suppressed,
// cross-attn to T2U memory. KV-cached incremental decode: cross-attn K/V computed
// once, self-attn K/V appended per step (each step processes ONE token, not the
// whole growing sequence). Logits/mask/argmax on GPU; only token append is scalar.
#include "pipeline.h"
#include <string>

std::vector<int> Pipeline::unit_greedy_decode(const std::vector<float>& memv, int Tmem) {
    const std::string dir = "model.decoder.";
    const int nlayers = 2, vocab = 10015, prefix = unit_prefix_, eos = 2, unk = 3, pad = 1, max_len = 300;
    Tensor mem = ops_.upload(memv);
    KVCache c = make_cache(dir, mem, Tmem, nlayers, vocab, max_len + 4);
    int base = ops_.mark();

    std::vector<int> toks{eos, prefix};
    decode_step(c, eos, 0);                 // prime position 0
    Tensor lg = decode_step(c, prefix, 1);  // position 1 -> logits predict first unit

    // CHAINED greedy decode: argmax writes the token to out_idx() (device); the next
    // step's embed reads it directly — NO per-step host readback (the in-order queue
    // guarantees embed reads before the next argmax overwrites). Each token is copied
    // to a device array and EOS is checked in batches of CHECK_EVERY, so the GPU
    // pipelines CHECK_EVERY steps between host syncs. (NNOPT_NOCHAIN = old per-step path.)
    if (!std::getenv("NNOPT_NOCHAIN")) {
        const int CHECK_EVERY = 8;
        cl_mem toks_dev = ops_.alloc_ints(max_len + 4);
        int base2 = ops_.mark();                 // step temps freed to here; toks_dev persists
        std::vector<int> htok(max_len + 4, -1);
        int pos = 2, eos_pos = -1, scanned = 2;
        while (pos < max_len) {
            cl_mem sc = ops_.alloc_f32(vocab);
            ops_.logits_to_f32(lg, sc, 0, vocab);
            ops_.mask_region(sc, 0, vocab, pad, unk, -1, false, eos);
            ops_.argmax_dev(sc, vocab);                          // token_pos -> out_idx() (no readback)
            ops_.copy_ints(toks_dev, pos, ops_.out_idx(), 0, 1); // record device-side
            if (((pos - 2) % CHECK_EVERY) == 0) {                // batched EOS check
                int upto = pos + 1;
                ops_.download_ints(toks_dev, scanned, upto - scanned, &htok[scanned]);
                for (int j = scanned; j < upto; ++j) if (htok[j] == eos) { eos_pos = j; break; }
                scanned = upto;
            }
            if (eos_pos >= 0) break;
            ops_.release_to(base2);                              // free sc + consumed lg + step temps
            lg = decode_step_ids(c, ops_.out_idx(), pos);        // chained embed (device token)
            ++pos;
        }
        if (pos > scanned) ops_.download_ints(toks_dev, scanned, pos + 1 - scanned, &htok[scanned]);
        for (int j = 2; j <= pos && j < max_len; ++j) { toks.push_back(htok[j]); if (htok[j] == eos) break; }
        ops_.release_to(base);
        return toks;
    }

    int pos = 2;
    while (true) {
        cl_mem sc = ops_.alloc_f32(vocab);
        ops_.logits_to_f32(lg, sc, 0, vocab);
        ops_.mask_region(sc, 0, vocab, pad, unk, /*force*/ -1, /*suppress_eos*/ false, eos);
        int best; float bv;
        ops_.argmax_f32(sc, vocab, best, bv);
        toks.push_back(best);
        if (best == eos || pos >= max_len) break;
        ops_.release_to(base);              // free prev logits + step temps; KV cache persists
        lg = decode_step(c, best, pos);
        ++pos;
    }
    ops_.release_to(base);
    return toks;
}
