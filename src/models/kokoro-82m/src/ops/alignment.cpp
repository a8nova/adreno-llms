// Reference: kokoro/model.py:forward_with_tokens — steps 10..12, 15
//   indices = repeat_interleave(arange(T_chars), pred_dur)        # [T_frames]
//   pred_aln_trg [T_chars, T_frames] = one-hot at (indices[j], j)
//   en  = d.T  @ pred_aln_trg   -> [hidden, T_frames]   (length regulator for d)
//   asr = t_en @ pred_aln_trg   -> [hidden, T_frames]   (length regulator for t_en)
//
// Since pred_aln_trg is one-hot, "x @ pred_aln_trg" reduces to: for each
// output frame j, copy column indices[j] of x. That's a cheap gather, no
// matmul needed.

#include "opencl_context.h"
#include "weights.h"
#include "debug_utils.h"
#include "profiler.h"
#include "model_config.h"
#include "utils.h"

#include <CL/cl.h>
#include <cstdint>
#include <string>
#include <vector>

static cl_program g_aln_prog = nullptr;
static cl_kernel  g_k_aln_gather = nullptr;

static const char* k_aln_src = R"CLC(
#ifdef NNOPT_USE_FP16
  #pragma OPENCL EXTENSION cl_khr_fp16 : enable
  typedef half storage_t;
  #define LOAD(p,i)    vload_half((i), (__global const half*)(p))
  #define STORE(p,i,v) vstore_half((float)(v), (i), (__global half*)(p))
#else
  typedef float storage_t;
  #define LOAD(p,i)    ((p)[(i)])
  #define STORE(p,i,v) ((p)[(i)] = (v))
#endif

// x_NLC: [T_chars, C].  out_NCL: [C, T_frames] (NCL — matches reference layout).
// indices: [T_frames] int (source char per output frame).
__kernel void aln_gather_NCL(__global const storage_t* x_nlc,
                             __global const int* indices,
                             __global storage_t* out_ncl,
                             int T_chars, int T_frames, int C) {
    int c = get_global_id(0);
    int j = get_global_id(1);
    if (c >= C || j >= T_frames) return;
    int src = indices[j];
    if (src < 0) src = 0;
    if (src >= T_chars) src = T_chars - 1;
    float v = (float)LOAD(x_nlc, src*C + c);
    STORE(out_ncl, c*T_frames + j, v);
}
)CLC";

static bool ensure_built(OpenCLContext& cl_ctx) {
    if (g_k_aln_gather) return true;
    cl_int err = CL_SUCCESS;
    const char* opts =
#ifdef NNOPT_USE_FP16
        "-DNNOPT_USE_FP16=1";
#else
        "";
#endif
    cl_device_id dev = cl_ctx.device();
    g_aln_prog = nnopt_build_program_cached(cl_ctx.context(), dev, k_aln_src, opts, "alignment", &err);
    if (!g_aln_prog) return false;
    g_k_aln_gather = clCreateKernel(g_aln_prog, "aln_gather_NCL", &err);
    return g_k_aln_gather != nullptr;
}

// Build flat index list from per-char durations.
extern "C" int op_alignment_make_indices(const std::vector<int>& pred_dur,
                                         std::vector<int>& indices_out) {
    indices_out.clear();
    int total = 0;
    for (int d : pred_dur) total += (d > 0 ? d : 1);
    indices_out.reserve((size_t)total);
    for (int i = 0; i < (int)pred_dur.size(); ++i) {
        int d = pred_dur[i] > 0 ? pred_dur[i] : 1;
        for (int k = 0; k < d; ++k) indices_out.push_back(i);
    }
    return total;
}

// Gather along T axis using indices to expand x_nlc[T_chars, C] -> out_ncl[C, T_frames].
extern "C" int op_alignment_gather_NCL(OpenCLContext& cl_ctx,
                                       cl_command_queue queue,
                                       cl_mem x_nlc,
                                       cl_mem indices_dev,
                                       cl_mem out_ncl,
                                       int T_chars, int T_frames, int C) {
    if (!ensure_built(cl_ctx)) return -1;
    cl_int err = CL_SUCCESS;
    err |= clSetKernelArg(g_k_aln_gather, 0, sizeof(cl_mem), &x_nlc);
    err |= clSetKernelArg(g_k_aln_gather, 1, sizeof(cl_mem), &indices_dev);
    err |= clSetKernelArg(g_k_aln_gather, 2, sizeof(cl_mem), &out_ncl);
    err |= clSetKernelArg(g_k_aln_gather, 3, sizeof(int), &T_chars);
    err |= clSetKernelArg(g_k_aln_gather, 4, sizeof(int), &T_frames);
    err |= clSetKernelArg(g_k_aln_gather, 5, sizeof(int), &C);
    if (err != CL_SUCCESS) return -1;
    size_t gws[2] = {(size_t)C, (size_t)T_frames};
    return nnopt_enqueue_profiled(queue, g_k_aln_gather, 2, nullptr, gws, nullptr, 0, nullptr, nullptr);
}
