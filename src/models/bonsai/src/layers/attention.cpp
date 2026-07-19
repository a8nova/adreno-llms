// DeviceModel attention dispatch: RoPE, GQA scores/softmax/attn-out, and the
// 4-bit-KV variants. Decode (M=1) and batched-prefill (M) paths.
#include "model.h"

void DeviceModel::run_rope_m(cl_mem qb, cl_mem kb, int NH, int KH, int D, int M) {
    int a = 0;
    arg(k_rope_, a++, sizeof(cl_mem), &qb, "rm.q");
    arg(k_rope_, a++, sizeof(cl_mem), &kb, "rm.k");
    arg(k_rope_, a++, sizeof(cl_mem), &cos_, "rm.c");
    arg(k_rope_, a++, sizeof(cl_mem), &sin_, "rm.s");
    arg(k_rope_, a++, sizeof(int), &M, "rm.sq");
    arg(k_rope_, a++, sizeof(int), &NH, "rm.qh");
    arg(k_rope_, a++, sizeof(int), &KH, "rm.kh");
    arg(k_rope_, a++, sizeof(int), &D, "rm.d");
    arg(k_rope_, a++, sizeof(cl_mem), &counter_, "rm.ctr");
    size_t pairs = (size_t)M * (NH + KH) * (D / 2);
    run1(k_rope_, (pairs + 63) / 64 * 64, 64, "rope_m");
}

void DeviceModel::run_scores_m(cl_mem qb, cl_mem kc, int NH, int KH, int D, int M,
                  int seq_k) {
    int a = 0;
    float scale = 1.0f / sqrtf((float)D);
    arg(k_scores_, a++, sizeof(cl_mem), &qb, "sm.q");
    arg(k_scores_, a++, sizeof(cl_mem), &kc, "sm.k");
    arg(k_scores_, a++, sizeof(cl_mem), &scores_, "sm.s");
    arg(k_scores_, a++, sizeof(int), &M, "sm.sq");
    arg(k_scores_, a++, sizeof(cl_mem), &counter_, "sm.ctr");
    arg(k_scores_, a++, sizeof(int), &NH, "sm.qh");
    arg(k_scores_, a++, sizeof(int), &KH, "sm.kh");
    arg(k_scores_, a++, sizeof(int), &D, "sm.d");
    arg(k_scores_, a++, sizeof(float), &scale, "sm.sc");
    size_t g[3] = {(size_t)NH, (size_t)M, (size_t)seq_k};
    CLCHECK(clEnqueueNDRangeKernel(ocl_.queue(), k_scores_, 3, nullptr, g,
                                   nullptr, 0, nullptr, nullptr), "scores_m");
}

void DeviceModel::run_softmax_m(int NH, int M) {
    int a = 0, rows = NH * M;
    arg(k_softmax_, a++, sizeof(cl_mem), &scores_, "smx.s");
    arg(k_softmax_, a++, sizeof(int), &M, "smx.sq");
    arg(k_softmax_, a++, sizeof(cl_mem), &counter_, "smx.ctr");
    arg(k_softmax_, a++, sizeof(int), &rows, "smx.r");
    run1(k_softmax_, rows, 0, "softmax_m");
}

void DeviceModel::run_attnout_m(cl_mem vc, int NH, int KH, int D, int M) {
    int a = 0;
    arg(k_attnout_, a++, sizeof(cl_mem), &scores_, "am.s");
    arg(k_attnout_, a++, sizeof(cl_mem), &vc, "am.v");
    arg(k_attnout_, a++, sizeof(cl_mem), &attp_, "am.o");
    arg(k_attnout_, a++, sizeof(int), &M, "am.sq");
    arg(k_attnout_, a++, sizeof(cl_mem), &counter_, "am.ctr");
    arg(k_attnout_, a++, sizeof(int), &NH, "am.qh");
    arg(k_attnout_, a++, sizeof(int), &KH, "am.kh");
    arg(k_attnout_, a++, sizeof(int), &D, "am.d");
    size_t g[3] = {(size_t)NH, (size_t)M, (size_t)(D / 4)};
    CLCHECK(clEnqueueNDRangeKernel(ocl_.queue(), k_attnout_, 3, nullptr, g,
                                   nullptr, 0, nullptr, nullptr), "attnout_m");
}

void DeviceModel::run_kvq4(cl_mem src, cl_mem dst, int rows, int KH, int D, int dst_off) {
    int a = 0;
    arg(k_kvq4_, a++, sizeof(cl_mem), &src, "q4.s");
    arg(k_kvq4_, a++, sizeof(cl_mem), &dst, "q4.d");
    arg(k_kvq4_, a++, sizeof(int), &rows, "q4.r");
    arg(k_kvq4_, a++, sizeof(int), &KH, "q4.kh");
    arg(k_kvq4_, a++, sizeof(int), &D, "q4.d");
    arg(k_kvq4_, a++, sizeof(int), &dst_off, "q4.off");
    run2(k_kvq4_, (size_t)rows, (size_t)KH, "kvq4");
}

void DeviceModel::run_scores4(cl_mem qb, cl_mem kc, int NH, int KH, int D, int seq_k) {
    int a = 0, seq_q = 1;
    float scale = 1.0f / sqrtf((float)D);
    arg(k_scores4_, a++, sizeof(cl_mem), &qb, "s4.q");
    arg(k_scores4_, a++, sizeof(cl_mem), &kc, "s4.k");
    arg(k_scores4_, a++, sizeof(cl_mem), &scores_, "s4.s");
    arg(k_scores4_, a++, sizeof(int), &seq_q, "s4.sq");
    arg(k_scores4_, a++, sizeof(cl_mem), &counter_, "s4.ctr");
    arg(k_scores4_, a++, sizeof(int), &NH, "s4.qh");
    arg(k_scores4_, a++, sizeof(int), &KH, "s4.kh");
    arg(k_scores4_, a++, sizeof(int), &D, "s4.d");
    arg(k_scores4_, a++, sizeof(float), &scale, "s4.sc");
    size_t g[3] = {(size_t)NH, 1, (size_t)seq_k};
    CLCHECK(clEnqueueNDRangeKernel(ocl_.queue(), k_scores4_, 3, nullptr, g,
                                   nullptr, 0, nullptr, nullptr), "scores4");
}

void DeviceModel::run_attnout4(cl_mem vc, int NH, int KH, int D) {
    int a = 0, seq_q = 1;
    arg(k_attnout4_, a++, sizeof(cl_mem), &scores_, "a4.s");
    arg(k_attnout4_, a++, sizeof(cl_mem), &vc, "a4.v");
    arg(k_attnout4_, a++, sizeof(cl_mem), &att_, "a4.o");
    arg(k_attnout4_, a++, sizeof(int), &seq_q, "a4.sq");
    arg(k_attnout4_, a++, sizeof(cl_mem), &counter_, "a4.ctr");
    arg(k_attnout4_, a++, sizeof(int), &NH, "a4.qh");
    arg(k_attnout4_, a++, sizeof(int), &KH, "a4.kh");
    arg(k_attnout4_, a++, sizeof(int), &D, "a4.d");
    size_t g[3] = {(size_t)NH, 1, (size_t)(D / 4)};
    CLCHECK(clEnqueueNDRangeKernel(ocl_.queue(), k_attnout4_, 3, nullptr, g,
                                   nullptr, 0, nullptr, nullptr), "attnout4");
}

void DeviceModel::run_rope(cl_mem qb, cl_mem kb, int NH, int KH, int D) {
    int a = 0, seq_q = 1;
    arg(k_rope_, a++, sizeof(cl_mem), &qb, "ro.q");
    arg(k_rope_, a++, sizeof(cl_mem), &kb, "ro.k");
    arg(k_rope_, a++, sizeof(cl_mem), &cos_, "ro.c");
    arg(k_rope_, a++, sizeof(cl_mem), &sin_, "ro.s");
    arg(k_rope_, a++, sizeof(int), &seq_q, "ro.sq");
    arg(k_rope_, a++, sizeof(int), &NH, "ro.qh");
    arg(k_rope_, a++, sizeof(int), &KH, "ro.kh");
    arg(k_rope_, a++, sizeof(int), &D, "ro.d");
    arg(k_rope_, a++, sizeof(cl_mem), &counter_, "ro.ctr");
    size_t pairs = (size_t)(NH + KH) * (D / 2);
    run1(k_rope_, (pairs + 63) / 64 * 64, 64, "rope");
}

void DeviceModel::run_scores(cl_mem qb, cl_mem kc, int NH, int KH, int D, int seq_k) {
    int a = 0, seq_q = 1;
    float scale = 1.0f / sqrtf((float)D);
    arg(k_scores_, a++, sizeof(cl_mem), &qb, "sc.q");
    arg(k_scores_, a++, sizeof(cl_mem), &kc, "sc.k");
    arg(k_scores_, a++, sizeof(cl_mem), &scores_, "sc.s");
    arg(k_scores_, a++, sizeof(int), &seq_q, "sc.sq");
    arg(k_scores_, a++, sizeof(cl_mem), &counter_, "sc.ctr");
    arg(k_scores_, a++, sizeof(int), &NH, "sc.qh");
    arg(k_scores_, a++, sizeof(int), &KH, "sc.kh");
    arg(k_scores_, a++, sizeof(int), &D, "sc.d");
    arg(k_scores_, a++, sizeof(float), &scale, "sc.sc");
    size_t g[3] = {(size_t)NH, 1, (size_t)seq_k};
    CLCHECK(clEnqueueNDRangeKernel(ocl_.queue(), k_scores_, 3, nullptr, g,
                                   nullptr, 0, nullptr, nullptr), "scores");
}

void DeviceModel::run_softmax(int NH, int seq_k) {
    (void)seq_k;
    int a = 0, seq_q = 1, rows = NH;
    arg(k_softmax_, a++, sizeof(cl_mem), &scores_, "sm.s");
    arg(k_softmax_, a++, sizeof(int), &seq_q, "sm.sq");
    arg(k_softmax_, a++, sizeof(cl_mem), &counter_, "sm.ctr");
    arg(k_softmax_, a++, sizeof(int), &rows, "sm.r");
    run1(k_softmax_, NH, 0, "softmax");
}

void DeviceModel::run_attnout(cl_mem vc, int NH, int KH, int D) {
    int a = 0, seq_q = 1;
    arg(k_attnout_, a++, sizeof(cl_mem), &scores_, "ao.s");
    arg(k_attnout_, a++, sizeof(cl_mem), &vc, "ao.v");
    arg(k_attnout_, a++, sizeof(cl_mem), &att_, "ao.o");
    arg(k_attnout_, a++, sizeof(int), &seq_q, "ao.sq");
    arg(k_attnout_, a++, sizeof(cl_mem), &counter_, "ao.ctr");
    arg(k_attnout_, a++, sizeof(int), &NH, "ao.qh");
    arg(k_attnout_, a++, sizeof(int), &KH, "ao.kh");
    arg(k_attnout_, a++, sizeof(int), &D, "ao.d");
    size_t g[3] = {(size_t)NH, 1, (size_t)(D / 4)};
    CLCHECK(clEnqueueNDRangeKernel(ocl_.queue(), k_attnout_, 3, nullptr, g,
                                   nullptr, 0, nullptr, nullptr), "attnout");
}
