// Gemma3MLP — gate/up/down feedforward with gelu-tanh gating.
//
// Reference: model_info/transformers_src/modeling_gemma3.py:21-31 Gemma3MLP.forward
//   down_proj(act_fn(gate_proj(x)) * up_proj(x))
//   act_fn = gelu_pytorch_tanh
//
// weight_prefix is "model.layers.<i>.mlp"; weights:
//   gate_proj.weight [intermediate, hidden]
//   up_proj.weight   [intermediate, hidden]
//   down_proj.weight [hidden, intermediate]

#include "../opencl_context.h"
#include "../weights.h"
#include "../debug_utils.h"
#include "../model_config.h"
#include "../utils.h"
#include "../profiler.h"
#include <string>

// Defined in GELUTanh.cpp.
cl_mem gemma3_gelu_run(OpenCLContext& cl_ctx, cl_command_queue queue, cl_mem input, int n);

namespace {
cl_program g_prog = nullptr;
cl_kernel  g_mul = nullptr;
bool ensure_kernel(OpenCLContext& cl_ctx) {
    if (g_mul) return true;
    g_prog = cl_ctx.build_program_from_file("kernels/gemma3_ops.cl");  // PROGRAM-INIT-OK
    if (!g_prog) { NNOPT_ERROR("MLP: build gemma3_ops.cl failed"); return false; }
    cl_int err = CL_SUCCESS;
    g_mul = clCreateKernel(g_prog, "gemma3_mul", &err);
    if (err != CL_SUCCESS || !g_mul) { NNOPT_ERROR_FMT("MLP: clCreateKernel %d", err); return false; }
    return true;
}
} // namespace

extern "C" {
cl_mem Gemma3MLP_forward(
    OpenCLContext& cl_ctx,
    Weights& weights,
    cl_command_queue queue,
    cl_mem input,
    int seq_len,
    int layer_idx,
    int start_pos,
    cl_mem* k_cache_inout,
    cl_mem* v_cache_inout,
    cl_mem encoder_hidden_states,
    const char* weight_prefix)
{
    (void)layer_idx; (void)start_pos; (void)k_cache_inout; (void)v_cache_inout;
    (void)encoder_hidden_states;
    KernelProfiler::HostTimer _ht_mlp("host_mlp_total");
    if (!ensure_kernel(cl_ctx)) return nullptr;
    const std::string wp = weight_prefix ? std::string(weight_prefix) : std::string();

    cl_mem gate_w = weights.get_buffer(wp + ".gate_proj.weight");
    cl_mem up_w   = weights.get_buffer(wp + ".up_proj.weight");
    cl_mem down_w = weights.get_buffer(wp + ".down_proj.weight");
    if (!gate_w || !up_w || !down_w) {
        NNOPT_ERROR_FMT("MLP: missing weights for %s", wp.c_str());
        return nullptr;
    }

    const int M = seq_len;
    const int H = MODEL_CONFIG::HIDDEN_SIZE;
    const int I = MODEL_CONFIG::INTERMEDIATE_SIZE;

    cl_int err = CL_SUCCESS;
    cl_mem gate = nullptr, up = nullptr, act = nullptr, gated = nullptr, out = nullptr;
    auto cleanup = [&]() -> cl_mem {
        if (gate)  { nnopt_pool_free(gate);  gate  = nullptr; }
        if (up)    { nnopt_pool_free(up);    up    = nullptr; }
        if (act)   { nnopt_pool_free(act);   act   = nullptr; }
        if (gated) { nnopt_pool_free(gated); gated = nullptr; }
        if (out)   { nnopt_pool_free(out);   out   = nullptr; }
        return nullptr;
    };

    // gate = gate_proj(x), up = up_proj(x)  [M, I] each — fused into one GEMV
    // (both read the same x). Removes one dispatch per layer; decode is
    // dispatch-bound on these projections.
    gate = nnopt_pool_alloc(cl_ctx.context(), (size_t)M * I * sizeof(nnopt_storage_t), &err);
    if (err != CL_SUCCESS || !gate) { NNOPT_ERROR_FMT("MLP: alloc gate %d", err); return cleanup(); }
    up = nnopt_pool_alloc(cl_ctx.context(), (size_t)M * I * sizeof(nnopt_storage_t), &err);
    if (err != CL_SUCCESS || !up) { NNOPT_ERROR_FMT("MLP: alloc up %d", err); return cleanup(); }
    if (!pytorch_linear_fused2(queue, M, I, I, H, input, gate_w, up_w, gate, up)) {
        NNOPT_ERROR("MLP: gate/up proj failed"); return cleanup();
    }

    // act = gelu(gate)  [M, I]
    act = gemma3_gelu_run(cl_ctx, queue, gate, M * I);
    if (!act) { NNOPT_ERROR("MLP: gelu failed"); return cleanup(); }

    // gated = act * up  [M, I]
    gated = nnopt_pool_alloc(cl_ctx.context(), (size_t)M * I * sizeof(nnopt_storage_t), &err);
    if (err != CL_SUCCESS || !gated) { NNOPT_ERROR_FMT("MLP: alloc gated %d", err); return cleanup(); }
    {
        int n = M * I;
        if (!set_arg_checked(g_mul, 0, sizeof(cl_mem), &act,   "a"))   return cleanup();
        if (!set_arg_checked(g_mul, 1, sizeof(cl_mem), &up,    "b"))   return cleanup();
        if (!set_arg_checked(g_mul, 2, sizeof(cl_mem), &gated, "out")) return cleanup();
        if (!set_arg_checked(g_mul, 3, sizeof(int),    &n,     "n"))   return cleanup();
        size_t gws = (size_t)n;
        err = clEnqueueNDRangeKernel(queue, g_mul, 1, nullptr, &gws, nullptr,
                                     0, nullptr, KernelProfiler::event_for("op_gemma3_mul"));
        if (err != CL_SUCCESS) { NNOPT_ERROR_FMT("MLP: mul dispatch %d", err); return cleanup(); }
    }

    // out = down_proj(gated)  [M, H]
    out = nnopt_pool_alloc(cl_ctx.context(), (size_t)M * H * sizeof(nnopt_storage_t), &err);
    if (err != CL_SUCCESS || !out) { NNOPT_ERROR_FMT("MLP: alloc out %d", err); return cleanup(); }
    if (!pytorch_linear(queue, M, H, I, gated, down_w, out)) { NNOPT_ERROR("MLP: down_proj failed"); return cleanup(); }

    // Free intermediates (out is returned to caller).
    nnopt_pool_free(gate);  gate  = nullptr;
    nnopt_pool_free(up);    up    = nullptr;
    nnopt_pool_free(act);   act   = nullptr;
    nnopt_pool_free(gated); gated = nullptr;
    return out;
}
}
