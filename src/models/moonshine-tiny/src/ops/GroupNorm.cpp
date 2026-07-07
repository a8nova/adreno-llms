// GroupNorm.cpp — nn.GroupNorm(num_groups=1, num_channels=288, eps=1e-5).
// Reference: model_info/transformers_src/modeling_moonshine.py MoonshineEncoder
//   self.groupnorm = nn.GroupNorm(num_groups=1, num_channels=embed_dim, eps=1e-5)
//   applied to conv1 output [1, C, L]. num_groups=1 => stats over ALL C*L
//   elements (per batch), weight/bias are per-channel [C].
//
// Input/output are channel-major [C, L]. weight_prefix = "model.encoder.groupnorm".
// seq_len carries L (input length). C derived from weight shape.

#include "../opencl_context.h"
#include "../weights.h"
#include "../nnopt_error.h"
#include "../model_config.h"
#include "../utils.h"
#include "../profiler.h"
#include <string>
#include <vector>
#include <cmath>

cl_kernel moonshine_kernel(OpenCLContext& cl_ctx, const char* name);

extern "C" {
cl_mem GroupNorm_forward(
    OpenCLContext& cl_ctx,
    Weights& weights,
    cl_command_queue queue,
    cl_mem input,
    int seq_len,           // L
    int layer_idx,
    int start_pos,
    cl_mem* k_cache_inout,
    cl_mem* v_cache_inout,
    cl_mem encoder_hidden_states,
    const char* weight_prefix)
{
    (void)layer_idx; (void)start_pos;
    (void)k_cache_inout; (void)v_cache_inout; (void)encoder_hidden_states;

    const std::string wp = weight_prefix ? std::string(weight_prefix) : std::string();
    if (wp.empty()) { NNOPT_ERROR("GroupNorm: empty weight_prefix"); return nullptr; }

    std::vector<int> wshape = weights.get_shape(wp + ".weight");
    if (wshape.size() != 1) { NNOPT_ERROR_FMT("GroupNorm: bad weight shape %s", wp.c_str()); return nullptr; }
    const int C = wshape[0];
    const int L = seq_len;
    const int total = C * L;

    cl_mem w_buf = weights.get_buffer(wp + ".weight");
    cl_mem b_buf = weights.get_buffer(wp + ".bias");
    if (!w_buf || !b_buf) { NNOPT_ERROR_FMT("GroupNorm: missing weight/bias %s", wp.c_str()); return nullptr; }

    cl_int err = CL_SUCCESS;
    cl_mem out = clCreateBuffer(cl_ctx.context(), CL_MEM_READ_WRITE,
                               (size_t)total * sizeof(nnopt_storage_t), nullptr, &err);
    if (err != CL_SUCCESS || !out) { NNOPT_ERROR_FMT("GroupNorm: alloc %d", err); return nullptr; }

    // Stage 1: partial sums via workgroup reduction.
    const size_t local = 64;
    const size_t nwg = 32;              // fixed number of workgroups
    const size_t gws = nwg * local;
    std::vector<float> h_psum(nwg, 0.0f), h_pss(nwg, 0.0f);
    cl_mem d_psum = clCreateBuffer(cl_ctx.context(), CL_MEM_READ_WRITE, nwg * sizeof(float), nullptr, &err);
    cl_mem d_pss  = clCreateBuffer(cl_ctx.context(), CL_MEM_READ_WRITE, nwg * sizeof(float), nullptr, &err);

    cl_kernel kstats = moonshine_kernel(cl_ctx, "groupnorm_stats");
    if (!kstats) { NNOPT_ERROR("GroupNorm: stats kernel"); clReleaseMemObject(out); return nullptr; }

    set_arg_checked(kstats, 0, sizeof(cl_mem), &input, "input");
    set_arg_checked(kstats, 1, sizeof(cl_mem), &d_psum, "psum");
    set_arg_checked(kstats, 2, sizeof(cl_mem), &d_pss, "pss");
    set_arg_checked(kstats, 3, sizeof(int), &total, "total");
    clSetKernelArg(kstats, 4, local * sizeof(float), nullptr);   // lsum
    clSetKernelArg(kstats, 5, local * sizeof(float), nullptr);   // lsumsq
    err = clEnqueueNDRangeKernel(queue, kstats, 1, nullptr, &gws, &local, 0, nullptr,
                                 KernelProfiler::event_for("groupnorm_stats"));
    if (err != CL_SUCCESS) { NNOPT_ERROR_FMT("GroupNorm: stats dispatch %d", err); clReleaseMemObject(out); return nullptr; }
    clEnqueueReadBuffer(queue, d_psum, CL_TRUE, 0, nwg * sizeof(float), h_psum.data(), 0, nullptr, nullptr);
    clEnqueueReadBuffer(queue, d_pss, CL_TRUE, 0, nwg * sizeof(float), h_pss.data(), 0, nullptr, nullptr);
    clReleaseMemObject(d_psum); clReleaseMemObject(d_pss);

    double sum = 0.0, sumsq = 0.0;
    for (size_t i = 0; i < nwg; ++i) { sum += h_psum[i]; sumsq += h_pss[i]; }
    float mean = (float)(sum / (double)total);
    float var = (float)(sumsq / (double)total - (double)mean * mean);
    float inv_std = 1.0f / std::sqrt(var + 1e-5f);

    // Stage 2: apply.
    cl_kernel kapp = moonshine_kernel(cl_ctx, "groupnorm_apply");
    if (!kapp) { NNOPT_ERROR("GroupNorm: apply kernel"); clReleaseMemObject(out); return nullptr; }
    auto cleanup = [&]() -> cl_mem { clReleaseMemObject(out); return nullptr; };
    if (!set_arg_checked(kapp, 0, sizeof(cl_mem), &input, "input")) return cleanup();
    if (!set_arg_checked(kapp, 1, sizeof(cl_mem), &w_buf, "weight")) return cleanup();
    if (!set_arg_checked(kapp, 2, sizeof(cl_mem), &b_buf, "bias")) return cleanup();
    if (!set_arg_checked(kapp, 3, sizeof(cl_mem), &out, "output")) return cleanup();
    if (!set_arg_checked(kapp, 4, sizeof(float), &mean, "mean")) return cleanup();
    if (!set_arg_checked(kapp, 5, sizeof(float), &inv_std, "inv_std")) return cleanup();
    if (!set_arg_checked(kapp, 6, sizeof(int), &C, "C")) return cleanup();
    if (!set_arg_checked(kapp, 7, sizeof(int), &L, "L")) return cleanup();
    size_t gws2 = (size_t)total;
    err = clEnqueueNDRangeKernel(queue, kapp, 1, nullptr, &gws2, nullptr, 0, nullptr,
                                 KernelProfiler::event_for("groupnorm_apply"));
    if (err != CL_SUCCESS) { NNOPT_ERROR_FMT("GroupNorm: apply dispatch %d", err); return cleanup(); }
    return out;
}
}
