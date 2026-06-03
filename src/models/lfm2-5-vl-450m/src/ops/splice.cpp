// Reference: docs/MODALITY_VLM.md:splice_image_tokens contract; kernels/splice_image_tokens.cl
//
// Host wrapper for kernels/splice_image_tokens.cl.
// Allocates output buffer and dispatches the splice kernel.

#include "opencl_context.h"
#include "debug_utils.h"
#include "profiler.h"
#include "utils.h"

#include <CL/cl.h>

namespace {

static bool splice_set_arg_checked(cl_kernel k, cl_uint idx, size_t sz, const void* v, const char* name) {
    cl_int err = clSetKernelArg(k, idx, sz, v);
    if (err != CL_SUCCESS) {
        NNOPT_ERROR_FMT("splice_image_tokens: clSetKernelArg(%u,%s) failed (%d)", (unsigned)idx, name, (int)err);
        return false;
    }
    return true;
}

}  // namespace

bool splice_image_tokens(
    OpenCLContext& cl_ctx,
    cl_mem text_embeds, int T_in, int D,
    cl_mem image_features, int N,
    int placeholder_pos,
    cl_mem& out_spliced, int& T_out) {
    NNOPT_CHECKPOINT("splice_image_tokens");

    out_spliced = nullptr;
    T_out = T_in - 1 + N;

    if (!text_embeds || !image_features) {
        NNOPT_ERROR("splice_image_tokens: null input buffer");
        return false;
    }
    if (T_in <= 0 || N <= 0 || D <= 0) {
        NNOPT_ERROR_FMT("splice_image_tokens: invalid dims T_in=%d N=%d D=%d", T_in, N, D);
        return false;
    }
    if (placeholder_pos < 0 || placeholder_pos >= T_in) {
        NNOPT_ERROR_FMT("splice_image_tokens: placeholder_pos out of range (%d) for T_in=%d", placeholder_pos, T_in);
        return false;
    }

    cl_command_queue queue = cl_ctx.queue();
    if (!queue) {
        NNOPT_ERROR("splice_image_tokens: cl_ctx.queue() returned null");
        return false;
    }

    cl_int err = CL_SUCCESS;
    // PROGRAM-INIT-OK: small, called once per inference prefill in VLM path; TODO: hoist to persistent program in a class.
    cl_program program = cl_ctx.build_program_from_file("kernels/splice_image_tokens.cl");
    if (!program) {
        NNOPT_ERROR("splice_image_tokens: failed to build kernels/splice_image_tokens.cl");
        return false;
    }

    cl_kernel kernel = clCreateKernel(program, "splice_image_tokens", &err);
    if (err != CL_SUCCESS || !kernel) {
        NNOPT_ERROR_FMT("splice_image_tokens: clCreateKernel failed (%d)", (int)err);
        clReleaseProgram(program);
        return false;
    }

    const size_t out_elems = (size_t)T_out * (size_t)D;
    out_spliced = clCreateBuffer(cl_ctx.context(), CL_MEM_READ_WRITE,
                                 out_elems * sizeof(nnopt_storage_t), nullptr, &err);
    if (err != CL_SUCCESS || !out_spliced) {
        NNOPT_ERROR_FMT("splice_image_tokens: clCreateBuffer(out) failed (%d)", (int)err);
        clReleaseKernel(kernel);
        clReleaseProgram(program);
        return false;
    }

    auto fail = [&]() -> bool {
        if (out_spliced) { clReleaseMemObject(out_spliced); out_spliced = nullptr; }
        clReleaseKernel(kernel);
        clReleaseProgram(program);
        return false;
    };

    if (!splice_set_arg_checked(kernel, 0, sizeof(cl_mem), &text_embeds, "text_embeds")) return fail();
    if (!splice_set_arg_checked(kernel, 1, sizeof(cl_mem), &image_features, "image_features")) return fail();
    if (!splice_set_arg_checked(kernel, 2, sizeof(cl_mem), &out_spliced, "out")) return fail();
    if (!splice_set_arg_checked(kernel, 3, sizeof(int), &T_in, "T_in")) return fail();
    if (!splice_set_arg_checked(kernel, 4, sizeof(int), &N, "N")) return fail();
    if (!splice_set_arg_checked(kernel, 5, sizeof(int), &D, "D")) return fail();
    if (!splice_set_arg_checked(kernel, 6, sizeof(int), &placeholder_pos, "placeholder_pos")) return fail();

    {
        const size_t gws[2] = {(size_t)T_out, (size_t)D};
        err = clEnqueueNDRangeKernel(queue, kernel, 2, nullptr, gws, nullptr,
                                    0, nullptr,
                                    KernelProfiler::event_for("op_splice_image_tokens"));
        if (err != CL_SUCCESS) {
            NNOPT_ERROR_FMT("splice_image_tokens: dispatch failed (%d)", (int)err);
            return fail();
        }
    }

    clReleaseKernel(kernel);
    clReleaseProgram(program);
    return true;
}
