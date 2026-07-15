// Conv1d.cpp — Moonshine encoder frontend Conv1d.
// Reference: model_info/transformers_src/modeling_moonshine.py:520-545 MoonshineEncoder.forward
//   self.conv1 = nn.Conv1d(1, 288, kernel_size=127, stride=64, bias=False)
//   self.conv2 = nn.Conv1d(288, 576, kernel_size=7, stride=3)     (bias=True)
//   self.conv3 = nn.Conv1d(576, 288, kernel_size=3, stride=2)     (bias=True)
// PyTorch Conv1d weight layout: [out_channels, in_channels, kernel_size].
// input [Cin, Lin] channel-major -> output [Cout, Lout], Lout=(Lin-K)/stride+1.
//
// Signature note: this op uses layer_idx to select which conv (0/1/2) and
// start_pos to carry Lin (input length). weight_prefix names the conv weight
// base (e.g. "model.encoder.conv1"). We derive shapes from the weight tensor.

#include "../opencl_context.h"
#include "../weights.h"
#include "../nnopt_error.h"
#include "../model_config.h"
#include "../utils.h"
#include "../profiler.h"
#include <cstdlib>
#include <string>

// OPT-6/7: process-lifetime kernel cache + transposed-weight cache
// (moonshine_common.cpp / Linear.cpp). Kernels are NOT released by callers.
cl_kernel moonshine_kernel(OpenCLContext& cl_ctx, const char* name);
cl_mem transposed_weight_for(OpenCLContext& cl_ctx, cl_command_queue queue,
                             cl_mem w_buf, int N, int K);

extern "C" {
// layer_idx: conv index (0,1,2) — used only for logging.
// start_pos: Lin (input length in samples/frames) — passed by caller.
cl_mem Conv1d_forward(
    OpenCLContext& cl_ctx,
    Weights& weights,
    cl_command_queue queue,
    cl_mem input,
    int seq_len,          // Lin (input length)
    int layer_idx,
    int start_pos,        // unused
    cl_mem* k_cache_inout,
    cl_mem* v_cache_inout,
    cl_mem encoder_hidden_states,
    const char* weight_prefix)
{
    (void)layer_idx; (void)start_pos;
    (void)k_cache_inout; (void)v_cache_inout; (void)encoder_hidden_states;
    const std::string wp = weight_prefix ? std::string(weight_prefix) : std::string();
    const int Lin = seq_len;

    std::vector<int> wshape = weights.get_shape(wp + ".weight");
    if (wshape.size() != 3) {
        NNOPT_ERROR_FMT("Conv1d: bad weight shape for %s (rank=%zu)", wp.c_str(), wshape.size());
        return nullptr;
    }
    const int Cout = wshape[0];
    const int Cin = wshape[1];
    const int K = wshape[2];
    const int stride = (K == 127) ? 64 : (K == 7 ? 3 : 2);
    const int Lout = (Lin - K) / stride + 1;

    cl_mem w_buf = weights.get_buffer(wp + ".weight");
    if (!w_buf) { NNOPT_ERROR_FMT("Conv1d: missing weight %s.weight", wp.c_str()); return nullptr; }
    const bool has_bias = weights.has_tensor(wp + ".bias");
    cl_mem b_buf = has_bias ? weights.get_buffer(wp + ".bias") : w_buf; // dummy if no bias

    cl_int err = CL_SUCCESS;
    cl_mem out = nullptr;
    cl_kernel kernel = nullptr;
    auto cleanup = [&]() -> cl_mem {
        if (out) { clReleaseMemObject(out); out = nullptr; }
        return nullptr;
    };

    out = clCreateBuffer(cl_ctx.context(), CL_MEM_READ_WRITE,
                         (size_t)Cout * Lout * sizeof(nnopt_storage_t), nullptr, &err);
    if (err != CL_SUCCESS || !out) { NNOPT_ERROR_FMT("Conv1d: alloc out %d", err); return cleanup(); }

    // OPT-7 (r7): im2col + linear_gemm_t + bias + permute. The naive kernel
    // (one MAC-chain per work item) ran at <1% ALU utilization: 165ms/clip
    // for ~320 MFLOP. As a GEMM the conv rides the same coalesced K-major
    // kernel as everything else. A/B toggle: NNOPT_CONV_IM2COL=0 reverts.
    static const bool im2col_enabled = [] {
        const char* e = std::getenv("NNOPT_CONV_IM2COL");
        return !(e && e[0] == '0');
    }();
    if (im2col_enabled && (Cout % 4 == 0)) {
        const int Kt = Cin * K;                      // GEMM depth
        cl_mem wt = transposed_weight_for(cl_ctx, queue, w_buf, Cout, Kt);
        cl_kernel ic_k = moonshine_kernel(cl_ctx, "im2col_1d");
        cl_kernel gk = moonshine_kernel(cl_ctx, "linear_gemm_t4");
        cl_kernel pk = moonshine_kernel(cl_ctx, "permute_cl_to_lc");
        cl_mem col = nullptr, lc = nullptr;
        bool ok = (wt && ic_k && gk && pk);
        if (ok) {
            col = clCreateBuffer(cl_ctx.context(), CL_MEM_READ_WRITE,
                                 (size_t)Lout * Kt * sizeof(nnopt_storage_t), nullptr, &err);
            lc = clCreateBuffer(cl_ctx.context(), CL_MEM_READ_WRITE,
                                (size_t)Lout * Cout * sizeof(nnopt_storage_t), nullptr, &err);
            ok = (col && lc);
        }
        if (ok) {                                    // im2col
            ok = set_arg_checked(ic_k, 0, sizeof(cl_mem), &input, "input") &&
                 set_arg_checked(ic_k, 1, sizeof(cl_mem), &col, "col") &&
                 set_arg_checked(ic_k, 2, sizeof(int), &Cin, "Cin") &&
                 set_arg_checked(ic_k, 3, sizeof(int), &Lin, "Lin") &&
                 set_arg_checked(ic_k, 4, sizeof(int), &K, "K") &&
                 set_arg_checked(ic_k, 5, sizeof(int), &stride, "stride") &&
                 set_arg_checked(ic_k, 6, sizeof(int), &Lout, "Lout");
            if (ok) {
                size_t gws = (size_t)Lout * Kt;
                ok = clEnqueueNDRangeKernel(queue, ic_k, 1, nullptr, &gws, nullptr, 0, nullptr,
                                            KernelProfiler::event_for("conv_im2col")) == CL_SUCCESS;
            }
        }
        if (ok) {                                    // GEMM: col[Lout,Kt] @ WT[Kt,Cout] → lc[Lout,Cout]
            const size_t lws = 64;
            const size_t MT = 4, KT = 256;           // must match GEMM_MT/GEMM_KT
            ok = set_arg_checked(gk, 0, sizeof(cl_mem), &col, "x") &&
                 set_arg_checked(gk, 1, sizeof(cl_mem), &wt, "WT") &&
                 set_arg_checked(gk, 2, sizeof(cl_mem), &lc, "out") &&
                 set_arg_checked(gk, 3, sizeof(int), &Lout, "M") &&
                 set_arg_checked(gk, 4, sizeof(int), &Cout, "N") &&
                 set_arg_checked(gk, 5, sizeof(int), &Kt, "K") &&
                 clSetKernelArg(gk, 6, MT * KT * sizeof(float), nullptr) == CL_SUCCESS;
            if (ok) {
                size_t n_lanes = (size_t)(Cout / 4);
                size_t gws2[2] = { ((n_lanes + lws - 1) / lws) * lws, ((size_t)Lout + MT - 1) / MT };
                size_t lws2[2] = { lws, 1 };
                ok = clEnqueueNDRangeKernel(queue, gk, 2, nullptr, gws2, lws2, 0, nullptr,
                                            KernelProfiler::event_for("conv_gemmt")) == CL_SUCCESS;
            }
        }
        if (ok && has_bias) {                        // bias over [Lout, Cout] (bias per output channel = per col)
            cl_kernel bk = moonshine_kernel(cl_ctx, "bias_add");
            ok = bk &&
                 set_arg_checked(bk, 0, sizeof(cl_mem), &lc, "in") &&
                 set_arg_checked(bk, 1, sizeof(cl_mem), &b_buf, "bias") &&
                 set_arg_checked(bk, 2, sizeof(cl_mem), &lc, "out") &&
                 set_arg_checked(bk, 3, sizeof(int), &Lout, "rows") &&
                 set_arg_checked(bk, 4, sizeof(int), &Cout, "cols");
            if (ok) {
                size_t gws = (size_t)Lout * Cout;
                ok = clEnqueueNDRangeKernel(queue, bk, 1, nullptr, &gws, nullptr, 0, nullptr,
                                            KernelProfiler::event_for("bias_add")) == CL_SUCCESS;
            }
        }
        if (ok) {                                    // [Lout, Cout] → [Cout, Lout]
            ok = set_arg_checked(pk, 0, sizeof(cl_mem), &lc, "in") &&
                 set_arg_checked(pk, 1, sizeof(cl_mem), &out, "out") &&
                 set_arg_checked(pk, 2, sizeof(int), &Lout, "C") &&
                 set_arg_checked(pk, 3, sizeof(int), &Cout, "L");
            if (ok) {
                size_t gws = (size_t)Lout * Cout;
                ok = clEnqueueNDRangeKernel(queue, pk, 1, nullptr, &gws, nullptr, 0, nullptr,
                                            KernelProfiler::event_for("permute_cl_to_lc")) == CL_SUCCESS;
            }
        }
        if (col) clReleaseMemObject(col);
        if (lc) clReleaseMemObject(lc);
        if (ok) return out;
        NNOPT_ERROR_FMT("Conv1d: im2col path failed for %s — falling back to naive kernel", wp.c_str());
    }

    kernel = moonshine_kernel(cl_ctx, "conv1d");
    if (!kernel) { NNOPT_ERROR("Conv1d: create kernel"); return cleanup(); }

    int has_bias_i = has_bias ? 1 : 0;
    if (!set_arg_checked(kernel, 0, sizeof(cl_mem), &input, "input")) return cleanup();
    if (!set_arg_checked(kernel, 1, sizeof(cl_mem), &w_buf, "weight")) return cleanup();
    if (!set_arg_checked(kernel, 2, sizeof(cl_mem), &b_buf, "bias")) return cleanup();
    if (!set_arg_checked(kernel, 3, sizeof(cl_mem), &out, "output")) return cleanup();
    if (!set_arg_checked(kernel, 4, sizeof(int), &Cin, "Cin")) return cleanup();
    if (!set_arg_checked(kernel, 5, sizeof(int), &Lin, "Lin")) return cleanup();
    if (!set_arg_checked(kernel, 6, sizeof(int), &Cout, "Cout")) return cleanup();
    if (!set_arg_checked(kernel, 7, sizeof(int), &K, "K")) return cleanup();
    if (!set_arg_checked(kernel, 8, sizeof(int), &stride, "stride")) return cleanup();
    if (!set_arg_checked(kernel, 9, sizeof(int), &Lout, "Lout")) return cleanup();
    if (!set_arg_checked(kernel, 10, sizeof(int), &has_bias_i, "has_bias")) return cleanup();

    size_t gws = (size_t)Cout * Lout;
    err = clEnqueueNDRangeKernel(queue, kernel, 1, nullptr, &gws, nullptr, 0, nullptr,
                                 KernelProfiler::event_for("op_conv1d"));
    if (err != CL_SUCCESS) { NNOPT_ERROR_FMT("Conv1d: dispatch %d", err); return cleanup(); }

    return out;
}
}
