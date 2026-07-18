// DeviceModel norm / reduction / elementwise dispatch: RMSNorm, the 1-bit
// GEMV x-sum pre-pass (decode + batched), residual add, and device argmax.
#include "model.h"

void DeviceModel::run_xsum_b(cl_mem x, int K, int M) {
    int a = 0;
    arg(k_xsum_b_, a++, sizeof(cl_mem), &x, "xb.x");
    arg(k_xsum_b_, a++, sizeof(cl_mem), &xsump_, "xb.o");
    arg(k_xsum_b_, a++, sizeof(int), &K, "xb.K");
    arg(k_xsum_b_, a++, sizeof(int), &M, "xb.M");
    run2(k_xsum_b_, ((size_t)(K / 64) + 63) / 64 * 64, M, "xsum_b");
}

void DeviceModel::run_xsum(cl_mem x, int K) {
    if (no_xsum_) return;   // v4xor folds the sign; no pre-pass needed
    int a = 0;
    arg(k_xsum_, a++, sizeof(cl_mem), &x, "xs.x");
    arg(k_xsum_, a++, sizeof(cl_mem), &xsum_, "xs.o");
    arg(k_xsum_, a++, sizeof(int), &K, "xs.K");
    size_t items = K / 64;
    run1(k_xsum_, (items + 63) / 64 * 64, 64, "xsum");
}

void DeviceModel::run_rms(cl_mem x, cl_mem w, cl_mem out, int rows, int cols) {
    int a = 0;
    float eps = m_.rms_eps;
    arg(k_rms_, a++, sizeof(cl_mem), &x, "rm.x");
    arg(k_rms_, a++, sizeof(cl_mem), &w, "rm.w");
    arg(k_rms_, a++, sizeof(cl_mem), &out, "rm.o");
    arg(k_rms_, a++, sizeof(int), &rows, "rm.r");
    arg(k_rms_, a++, sizeof(int), &cols, "rm.c");
    arg(k_rms_, a++, sizeof(float), &eps, "rm.e");
    run1(k_rms_, (size_t)rows * 64, 64, "rms");
}

void DeviceModel::run_add(cl_mem acc, cl_mem b, int n) {
    int a = 0;
    arg(k_add_, a++, sizeof(cl_mem), &acc, "ad.a");
    arg(k_add_, a++, sizeof(cl_mem), &b, "ad.b");
    arg(k_add_, a++, sizeof(int), &n, "ad.n");
    run1(k_add_, n, 0, "add");
}

void DeviceModel::run_argmax(int N) {
    int a = 0;
    arg(k_argmax_, a++, sizeof(cl_mem), &logits_, "am.l");
    arg(k_argmax_, a++, sizeof(cl_mem), &amax_, "am.o");
    arg(k_argmax_, a++, sizeof(int), &N, "am.n");
    run1(k_argmax_, 256, 256, "argmax");
}
