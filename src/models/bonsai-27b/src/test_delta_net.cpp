// Differential test for the Gated DeltaNet recurrence.
//
// Reads a binary fixture written by scripts/test_delta_net.py, runs
// gdn::step() for T timesteps, writes the outputs back. The python side then
// diffs against transformers' `torch_recurrent_gated_delta_rule` verbatim.
//
// Exercised at BOTH v:k head ratios — 1:1 (Qwen3.5-2B) and 3:1 (Bonsai-27B) —
// because a kernel that assumes 1:1 passes every 2B test and is silently wrong
// on the 27B.
//
//   build: c++ -std=c++17 -O2 -o test_delta_net src/test_delta_net.cpp
//   run:   ./test_delta_net fixture.bin out.bin
//
// Fixture layout (little-endian):
//   i32 n_v_heads, n_k_heads, d_k, d_v, T
//   f32 A_log[n_v_heads], dt_bias[n_v_heads]
//   then T records of:
//     f32 q[n_k_heads*d_k], k[n_k_heads*d_k], v[n_v_heads*d_v],
//         a[n_v_heads], b[n_v_heads]
// Output: T records of f32 out[n_v_heads*d_v], then the final state
//         f32 S[n_v_heads*d_k*d_v].
#include <cstdint>
#include <cstdio>
#include <vector>

#include "delta_rule.h"

static void rd(FILE* f, void* p, size_t n) {
    if (fread(p, 1, n, f) != n) { fprintf(stderr, "short read\n"); exit(2); }
}

int main(int argc, char** argv) {
    if (argc < 3) { fprintf(stderr, "usage: %s <fixture.bin> <out.bin>\n", argv[0]); return 1; }
    FILE* f = fopen(argv[1], "rb");
    if (!f) { fprintf(stderr, "cannot open %s\n", argv[1]); return 1; }

    int32_t nv, nk, dk, dv, T;
    rd(f, &nv, 4); rd(f, &nk, 4); rd(f, &dk, 4); rd(f, &dv, 4); rd(f, &T, 4);
    fprintf(stderr, "n_v_heads=%d n_k_heads=%d (rep=%d) d_k=%d d_v=%d T=%d\n",
            nv, nk, nv / nk, dk, dv, T);

    std::vector<float> A_log(nv), dt_b(nv);
    rd(f, A_log.data(), sizeof(float) * nv);
    rd(f, dt_b.data(), sizeof(float) * nv);

    std::vector<float> S((size_t)nv * dk * dv, 0.f);
    std::vector<float> q((size_t)nk * dk), k((size_t)nk * dk),
                       v((size_t)nv * dv), a(nv), b(nv),
                       out((size_t)nv * dv), scratch;

    FILE* o = fopen(argv[2], "wb");
    if (!o) { fprintf(stderr, "cannot write %s\n", argv[2]); return 1; }
    for (int t = 0; t < T; ++t) {
        rd(f, q.data(), sizeof(float) * q.size());
        rd(f, k.data(), sizeof(float) * k.size());
        rd(f, v.data(), sizeof(float) * v.size());
        rd(f, a.data(), sizeof(float) * a.size());
        rd(f, b.data(), sizeof(float) * b.size());
        gdn::step(S.data(), q.data(), k.data(), v.data(), a.data(), b.data(),
                  A_log.data(), dt_b.data(), nv, nk, dk, dv, out.data(), scratch);
        fwrite(out.data(), sizeof(float), out.size(), o);
    }
    fwrite(S.data(), sizeof(float), S.size(), o);
    fclose(o);
    fclose(f);
    fprintf(stderr, "ok\n");
    return 0;
}
