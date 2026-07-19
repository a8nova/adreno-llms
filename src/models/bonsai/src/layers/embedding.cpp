// DeviceModel embedding dispatch: 1-bit token_embd row gather — host-supplied
// token (decode step 0), device-argmax token (subsequent decode), and batched
// (prefill) variants.
#include "model.h"

void DeviceModel::run_gather_b(QW& W, cl_mem out, int M, int K) {
    int a = 0, V = m_.vocab;
    arg(k_gather_b_, a++, sizeof(cl_mem), &W.bits, "gbB.Wb");
    arg(k_gather_b_, a++, sizeof(cl_mem), &W.scales, "gbB.Ws");
    arg(k_gather_b_, a++, sizeof(cl_mem), &out, "gbB.out");
    arg(k_gather_b_, a++, sizeof(cl_mem), &tokbuf_, "gbB.tok");
    arg(k_gather_b_, a++, sizeof(int), &V, "gbB.V");
    arg(k_gather_b_, a++, sizeof(int), &K, "gbB.K");
    arg(k_gather_b_, a++, sizeof(int), &M, "gbB.M");
    run2(k_gather_b_, (size_t)K, M, "gather_b");
}

void DeviceModel::run_gather_dev(QW& W, cl_mem out, int K) {
    int a = 0, V = m_.vocab;
    arg(k_gather_dev_, a++, sizeof(cl_mem), &W.bits, "gd.Wb");
    arg(k_gather_dev_, a++, sizeof(cl_mem), &W.scales, "gd.Ws");
    arg(k_gather_dev_, a++, sizeof(cl_mem), &out, "gd.out");
    arg(k_gather_dev_, a++, sizeof(cl_mem), &amax_, "gd.tok");
    arg(k_gather_dev_, a++, sizeof(int), &V, "gd.V");
    arg(k_gather_dev_, a++, sizeof(int), &K, "gd.K");
    run1(k_gather_dev_, K, 0, "gather_dev");
}

void DeviceModel::run_gather(QW& W, cl_mem out, int token, int K) {
    int a = 0, V = m_.vocab;
    arg(k_gather_, a++, sizeof(cl_mem), &W.bits, "g.Wb");
    arg(k_gather_, a++, sizeof(cl_mem), &W.scales, "g.Ws");
    arg(k_gather_, a++, sizeof(cl_mem), &out, "g.out");
    arg(k_gather_, a++, sizeof(int), &token, "g.tok");
    arg(k_gather_, a++, sizeof(int), &V, "g.V");
    arg(k_gather_, a++, sizeof(int), &K, "g.K");
    run1(k_gather_, K, 0, "gather");
}
