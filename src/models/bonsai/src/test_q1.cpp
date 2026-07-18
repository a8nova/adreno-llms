// P1 unit test: q1_row_decode + q1_gemv vs the byte-exact F16 ground truth
// (reference/k_proj_unpacked_f16.bin, fetched from the unpacked repo).
#include <cmath>
#include <cstdio>
#include <fstream>

#include "nnb.h"
#include "q1.h"

int main(int argc, char** argv) {
    Nnb nnb(argv[1]);
    const Tensor& k = nnb.get("blk.0.attn_k.weight");
    const int n_in = (int)k.ne(0), n_out = (int)k.ne(1);
    printf("attn_k: [%d, %d] kind=%d\n", n_in, n_out, (int)k.kind);

    std::ifstream gt(argv[2], std::ios::binary);
    std::vector<uint16_t> gt16((size_t)n_in * n_out);
    gt.read((char*)gt16.data(), gt16.size() * 2);

    // 1) row decode vs ground truth
    std::vector<float> row(n_in);
    int mism = 0;
    for (int r = 0; r < n_out; ++r) {
        q1_row_decode(k.data, r, n_in, row.data());
        for (int i = 0; i < n_in; ++i)
            if (row[i] != half_to_float(gt16[(size_t)r * n_in + i])) ++mism;
    }
    printf("row-decode mismatches: %d / %lld\n", mism, (long long)n_in * n_out);

    // 2) gemv vs reference dot on decoded rows
    std::vector<float> x(n_in);
    for (int i = 0; i < n_in; ++i) x[i] = 0.01f * ((i * 2654435761u >> 16 & 1023) - 512);
    std::vector<float> y(n_out), yref(n_out);
    q1_gemv(k.data, x.data(), y.data(), n_out, n_in);
    double maxabs = 0, ymax = 0;
    for (int r = 0; r < n_out; ++r) {
        q1_row_decode(k.data, r, n_in, row.data());
        double acc = 0;
        for (int i = 0; i < n_in; ++i) acc += (double)row[i] * x[i];
        yref[r] = (float)acc;
        maxabs = std::max(maxabs, (double)fabs(y[r] - yref[r]));
        ymax = std::max(ymax, (double)fabs(yref[r]));
    }
    printf("gemv max abs err %.3e vs max |y| %.3e (normalized %.3e)\n",
           maxabs, ymax, maxabs / ymax);
    return (mism == 0 && maxabs / ymax < 1e-5) ? 0 : 1;
}
