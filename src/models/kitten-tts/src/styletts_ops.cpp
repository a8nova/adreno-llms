#include "styletts_ops.h"
#include "nnopt_error.h"
#include "debug_utils.h"
#include "profiler.h"
#include <cmath>
#include <cstring>
#include <algorithm>

namespace st {
namespace {

cl_program g_prog = nullptr;
struct K {
    cl_kernel dql_scale = nullptr, dql_apply = nullptr;
    cl_kernel conv1d = nullptr, conv1d_tiled = nullptr, convtr_dw = nullptr;
    cl_kernel instancenorm = nullptr, instancenorm_wg = nullptr, adain_combine = nullptr, layernorm = nullptr;
    cl_kernel leaky = nullptr, add = nullptr, add_scaled = nullptr;
    cl_kernel bias_ct = nullptr, bias_rows = nullptr;
    cl_kernel transpose = nullptr, resize = nullptr, embed = nullptr, align = nullptr;
    cl_kernel adaln_rows = nullptr, pack_style = nullptr;
    cl_kernel snake = nullptr, convtr = nullptr, convtr_tiled = nullptr, istft = nullptr, magphase = nullptr;
    cl_kernel expk = nullptr, polar = nullptr, scalek = nullptr, leaky_a = nullptr;
    cl_kernel padref = nullptr;
} g_k;

bool make(cl_program p, cl_kernel* dst, const char* name) {
    cl_int e;
    *dst = clCreateKernel(p, name, &e);
    if (e != CL_SUCCESS) { NNOPT_ERROR_FMT("styletts: kernel %s err %d", name, e); return false; }
    return true;
}

bool run1d(cl_command_queue q, cl_kernel k, size_t gws, const char* what) {
    cl_int e = clEnqueueNDRangeKernel(q, k, 1, nullptr, &gws, nullptr, 0, nullptr,
                                      KernelProfiler::event_for(what));
    if (e != CL_SUCCESS) { NNOPT_ERROR_FMT("styletts: dispatch %s err %d", what, e); return false; }
    return true;
}

}  // namespace

bool init(OpenCLContext& ctx) {
    if (g_prog) return true;
    g_prog = ctx.build_program_from_file("kernels/styletts.cl");
    if (!g_prog) { NNOPT_ERROR("styletts: failed to build kernels/styletts.cl"); return false; }
    return make(g_prog, &g_k.dql_scale,     "st_dql_scale")
        && make(g_prog, &g_k.dql_apply,     "st_dql_apply")
        && make(g_prog, &g_k.conv1d,        "st_conv1d")
        && make(g_prog, &g_k.conv1d_tiled,  "st_conv1d_tiled")
        && make(g_prog, &g_k.convtr_dw,     "st_convtranspose1d_dw")
        && make(g_prog, &g_k.instancenorm,  "st_instancenorm")
        && make(g_prog, &g_k.instancenorm_wg, "st_instancenorm_wg")
        && make(g_prog, &g_k.adain_combine, "st_adain_combine")
        && make(g_prog, &g_k.layernorm,     "st_layernorm_rows")
        && make(g_prog, &g_k.leaky,         "st_leaky_relu")
        && make(g_prog, &g_k.add,           "st_add")
        && make(g_prog, &g_k.add_scaled,    "st_add_scaled")
        && make(g_prog, &g_k.bias_ct,       "st_add_bias_ct")
        && make(g_prog, &g_k.bias_rows,     "st_add_bias_rows")
        && make(g_prog, &g_k.transpose,     "st_transpose")
        && make(g_prog, &g_k.resize,        "st_resize_nearest_t")
        && make(g_prog, &g_k.embed,         "st_embed_tc")
        && make(g_prog, &g_k.align,         "st_align_expand")
        && make(g_prog, &g_k.adaln_rows,    "st_adaln_rows")
        && make(g_prog, &g_k.pack_style,    "st_pack_style")
        && make(g_prog, &g_k.snake,         "st_snake")
        && make(g_prog, &g_k.convtr,        "st_convtranspose1d")
        && make(g_prog, &g_k.convtr_tiled,  "st_convtranspose1d_tiled")
        && make(g_prog, &g_k.istft,         "st_istft_ola")
        && make(g_prog, &g_k.magphase,      "st_mag_phase")
        && make(g_prog, &g_k.expk,          "st_exp")
        && make(g_prog, &g_k.polar,         "st_polar_to_cart")
        && make(g_prog, &g_k.scalek,        "st_scale")
        && make(g_prog, &g_k.leaky_a,       "st_leaky_relu_a")
        && make(g_prog, &g_k.padref,        "st_pad_front_reflect");
}

bool adaln_rows(cl_command_queue q, cl_mem n, cl_mem fc_out, cl_mem out,
                int rows, int cols) {
    if (!set_arg_checked(g_k.adaln_rows, 0, sizeof(cl_mem), &n, "n") ||
        !set_arg_checked(g_k.adaln_rows, 1, sizeof(cl_mem), &fc_out, "fc_out") ||
        !set_arg_checked(g_k.adaln_rows, 2, sizeof(cl_mem), &out, "out") ||
        !set_arg_checked(g_k.adaln_rows, 3, sizeof(int), &rows, "rows") ||
        !set_arg_checked(g_k.adaln_rows, 4, sizeof(int), &cols, "cols")) return false;
    return run1d(q, g_k.adaln_rows, (size_t)rows * cols, "adaln_rows");
}

bool pack_style(cl_command_queue q, cl_mem a, cl_mem s, cl_mem out,
                int T, int C1, int C2) {
    if (!set_arg_checked(g_k.pack_style, 0, sizeof(cl_mem), &a, "a") ||
        !set_arg_checked(g_k.pack_style, 1, sizeof(cl_mem), &s, "s") ||
        !set_arg_checked(g_k.pack_style, 2, sizeof(cl_mem), &out, "out") ||
        !set_arg_checked(g_k.pack_style, 3, sizeof(int), &T, "T") ||
        !set_arg_checked(g_k.pack_style, 4, sizeof(int), &C1, "C1") ||
        !set_arg_checked(g_k.pack_style, 5, sizeof(int), &C2, "C2")) return false;
    return run1d(q, g_k.pack_style, (size_t)T * (C1 + C2), "pack_style");
}

cl_mem alloc(OpenCLContext& ctx, size_t n) {
    cl_int e;
    cl_mem m = clCreateBuffer(ctx.context(), CL_MEM_READ_WRITE,
                              n * sizeof(nnopt_storage_t), nullptr, &e);
    if (e != CL_SUCCESS) { NNOPT_ERROR_FMT("styletts alloc %zu err %d", n, e); return nullptr; }
    return m;
}

cl_mem alloc_f32(OpenCLContext& ctx, size_t n) {
    cl_int e;
    cl_mem m = clCreateBuffer(ctx.context(), CL_MEM_READ_WRITE, n * sizeof(float), nullptr, &e);
    if (e != CL_SUCCESS) { NNOPT_ERROR_FMT("styletts alloc_f32 %zu err %d", n, e); return nullptr; }
    return m;
}

cl_mem upload_f32_as_storage(OpenCLContext& ctx, const std::vector<float>& v) {
    std::vector<nnopt_storage_t> h(v.size());
    for (size_t i = 0; i < v.size(); i++)
#ifdef NNOPT_USE_FP16
        h[i] = nnopt_f32_to_f16(v[i]);
#else
        h[i] = v[i];
#endif
    cl_int e;
    cl_mem m = clCreateBuffer(ctx.context(), CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR,
                              h.size() * sizeof(nnopt_storage_t), h.data(), &e);
    if (e != CL_SUCCESS) { NNOPT_ERROR_FMT("styletts upload err %d", e); return nullptr; }
    return m;
}

bool download_as_f32(cl_command_queue q, cl_mem buf, size_t n, std::vector<float>& out) {
    std::vector<nnopt_storage_t> h(n);
    cl_int e = clEnqueueReadBuffer(q, buf, CL_TRUE, 0, n * sizeof(nnopt_storage_t),
                                   h.data(), 0, nullptr, nullptr);
    if (e != CL_SUCCESS) { NNOPT_ERROR_FMT("styletts download err %d", e); return false; }
    out.resize(n);
    for (size_t i = 0; i < n; i++)
#ifdef NNOPT_USE_FP16
        out[i] = nnopt_f16_to_f32(static_cast<uint16_t>(h[i]));
#else
        out[i] = h[i];
#endif
    return true;
}

bool dql(cl_command_queue q, cl_mem x, cl_mem params, cl_mem out, int n) {
    if (!set_arg_checked(g_k.dql_scale, 0, sizeof(cl_mem), &x, "x") ||
        !set_arg_checked(g_k.dql_scale, 1, sizeof(cl_mem), &params, "params") ||
        !set_arg_checked(g_k.dql_scale, 2, sizeof(int), &n, "n")) return false;
    size_t gws = 256, lws = 256;   // single work-group reduction
    cl_int e = clEnqueueNDRangeKernel(q, g_k.dql_scale, 1, nullptr, &gws, &lws, 0, nullptr,
                                      KernelProfiler::event_for("dql_scale"));
    if (e != CL_SUCCESS) { NNOPT_ERROR_FMT("dql_scale dispatch %d", e); return false; }

    if (!set_arg_checked(g_k.dql_apply, 0, sizeof(cl_mem), &x, "x") ||
        !set_arg_checked(g_k.dql_apply, 1, sizeof(cl_mem), &params, "params") ||
        !set_arg_checked(g_k.dql_apply, 2, sizeof(cl_mem), &out, "out") ||
        !set_arg_checked(g_k.dql_apply, 3, sizeof(int), &n, "n")) return false;
    return run1d(q, g_k.dql_apply, (size_t)n, "dql_apply");
}

// OPT #1 tiled-conv tunables — MUST match kernels/styletts.cl.
// Workgroup covers (TILE_LTIME*TILE_TREG) output times × TILE_CO output channels.
namespace { constexpr int TILE_LTIME = 16, TILE_TREG = 4, TILE_CO = 16, TILE_SPAN_MAX = 128; }

bool conv1d(cl_command_queue q, cl_mem in, cl_mem w, cl_mem bias, cl_mem out,
            int Cin, int Cout, int T_in, int T_out,
            int K, int stride, int pad, int groups, int dilation) {
    int has_bias = bias ? 1 : 0;
    cl_mem bias_arg = bias ? bias : in;    // dummy; kernel ignores when has_bias==0

    // OPT #1: groups==1 fast path via LDS-tiled kernel. Gated on span fitting the
    // LDS tile; falls back to the naive kernel otherwise. NNOPT_CONV_TILED=0
    // forces the naive path for A/B benchmarking.
    static const bool tiled_on = [] {
        const char* e = getenv("NNOPT_CONV_TILED");
        return !(e && e[0] == '0');
    }();
    const int span = (TILE_LTIME * TILE_TREG - 1) * stride + (K - 1) * dilation + 1;
    if (tiled_on && groups == 1 && span <= TILE_SPAN_MAX) {
        cl_kernel k = g_k.conv1d_tiled;
        if (!set_arg_checked(k, 0, sizeof(cl_mem), &in, "in") ||
            !set_arg_checked(k, 1, sizeof(cl_mem), &w, "w") ||
            !set_arg_checked(k, 2, sizeof(cl_mem), &bias_arg, "bias") ||
            !set_arg_checked(k, 3, sizeof(cl_mem), &out, "out") ||
            !set_arg_checked(k, 4, sizeof(int), &Cin, "Cin") ||
            !set_arg_checked(k, 5, sizeof(int), &Cout, "Cout") ||
            !set_arg_checked(k, 6, sizeof(int), &T_in, "T_in") ||
            !set_arg_checked(k, 7, sizeof(int), &T_out, "T_out") ||
            !set_arg_checked(k, 8, sizeof(int), &K, "K") ||
            !set_arg_checked(k, 9, sizeof(int), &stride, "stride") ||
            !set_arg_checked(k, 10, sizeof(int), &pad, "pad") ||
            !set_arg_checked(k, 11, sizeof(int), &has_bias, "has_bias") ||
            !set_arg_checked(k, 12, sizeof(int), &dilation, "dilation")) return false;
        const int times_per_wg = TILE_LTIME * TILE_TREG;
        size_t local[2]  = { (size_t)TILE_LTIME, (size_t)TILE_CO };
        size_t global[2] = {
            (size_t)((T_out + times_per_wg - 1) / times_per_wg) * TILE_LTIME,
            (size_t)((Cout  + TILE_CO      - 1) / TILE_CO)      * TILE_CO,
        };
        cl_int e = clEnqueueNDRangeKernel(q, k, 2, nullptr, global, local, 0, nullptr,
                                          KernelProfiler::event_for("conv1d_tiled"));
        if (e != CL_SUCCESS) { NNOPT_ERROR_FMT("conv1d_tiled dispatch err %d", e); return false; }
        return true;
    }

    if (!set_arg_checked(g_k.conv1d, 0, sizeof(cl_mem), &in, "in") ||
        !set_arg_checked(g_k.conv1d, 1, sizeof(cl_mem), &w, "w") ||
        !set_arg_checked(g_k.conv1d, 2, sizeof(cl_mem), &bias_arg, "bias") ||
        !set_arg_checked(g_k.conv1d, 3, sizeof(cl_mem), &out, "out") ||
        !set_arg_checked(g_k.conv1d, 4, sizeof(int), &Cin, "Cin") ||
        !set_arg_checked(g_k.conv1d, 5, sizeof(int), &Cout, "Cout") ||
        !set_arg_checked(g_k.conv1d, 6, sizeof(int), &T_in, "T_in") ||
        !set_arg_checked(g_k.conv1d, 7, sizeof(int), &T_out, "T_out") ||
        !set_arg_checked(g_k.conv1d, 8, sizeof(int), &K, "K") ||
        !set_arg_checked(g_k.conv1d, 9, sizeof(int), &stride, "stride") ||
        !set_arg_checked(g_k.conv1d, 10, sizeof(int), &pad, "pad") ||
        !set_arg_checked(g_k.conv1d, 11, sizeof(int), &groups, "groups") ||
        !set_arg_checked(g_k.conv1d, 12, sizeof(int), &has_bias, "has_bias") ||
        !set_arg_checked(g_k.conv1d, 13, sizeof(int), &dilation, "dilation")) return false;
    return run1d(q, g_k.conv1d, (size_t)Cout * T_out, "conv1d_naive");
}

bool conv_transpose1d_dw(cl_command_queue q, cl_mem in, cl_mem w, cl_mem bias,
                         cl_mem out, int C, int T_in, int T_out,
                         int K, int stride, int pad) {
    int has_bias = bias ? 1 : 0;
    cl_mem bias_arg = bias ? bias : in;
    if (!set_arg_checked(g_k.convtr_dw, 0, sizeof(cl_mem), &in, "in") ||
        !set_arg_checked(g_k.convtr_dw, 1, sizeof(cl_mem), &w, "w") ||
        !set_arg_checked(g_k.convtr_dw, 2, sizeof(cl_mem), &bias_arg, "bias") ||
        !set_arg_checked(g_k.convtr_dw, 3, sizeof(cl_mem), &out, "out") ||
        !set_arg_checked(g_k.convtr_dw, 4, sizeof(int), &C, "C") ||
        !set_arg_checked(g_k.convtr_dw, 5, sizeof(int), &T_in, "T_in") ||
        !set_arg_checked(g_k.convtr_dw, 6, sizeof(int), &T_out, "T_out") ||
        !set_arg_checked(g_k.convtr_dw, 7, sizeof(int), &K, "K") ||
        !set_arg_checked(g_k.convtr_dw, 8, sizeof(int), &stride, "stride") ||
        !set_arg_checked(g_k.convtr_dw, 9, sizeof(int), &pad, "pad") ||
        !set_arg_checked(g_k.convtr_dw, 10, sizeof(int), &has_bias, "has_bias")) return false;
    return run1d(q, g_k.convtr_dw, (size_t)C * T_out, "convtranspose1d_dw");
}

bool instancenorm(cl_command_queue q, cl_mem in, cl_mem nw, cl_mem nb,
                  cl_mem out, int C, int T) {
    // OPT #7: workgroup-per-channel path (default). NNOPT_IN_WG=0 → naive 1-WI path.
    static const bool wg_on = [] {
        const char* e = getenv("NNOPT_IN_WG");
        return !(e && e[0] == '0');
    }();
    cl_kernel k = wg_on ? g_k.instancenorm_wg : g_k.instancenorm;
    if (!set_arg_checked(k, 0, sizeof(cl_mem), &in, "in") ||
        !set_arg_checked(k, 1, sizeof(cl_mem), &nw, "norm_w") ||
        !set_arg_checked(k, 2, sizeof(cl_mem), &nb, "norm_b") ||
        !set_arg_checked(k, 3, sizeof(cl_mem), &out, "out") ||
        !set_arg_checked(k, 4, sizeof(int), &C, "C") ||
        !set_arg_checked(k, 5, sizeof(int), &T, "T")) return false;
    if (wg_on) {
        constexpr int IN_WG = 128;
        size_t local = IN_WG, global = (size_t)C * IN_WG;
        cl_int e = clEnqueueNDRangeKernel(q, k, 1, nullptr, &global, &local, 0, nullptr,
                                          KernelProfiler::event_for("instancenorm"));
        if (e != CL_SUCCESS) { NNOPT_ERROR_FMT("instancenorm_wg dispatch err %d", e); return false; }
        return true;
    }
    return run1d(q, g_k.instancenorm, (size_t)C, "instancenorm");
}

bool layernorm_rows(cl_command_queue q, cl_mem in, cl_mem gamma, cl_mem beta,
                    cl_mem out, int rows, int cols, float eps) {
    if (!set_arg_checked(g_k.layernorm, 0, sizeof(cl_mem), &in, "in") ||
        !set_arg_checked(g_k.layernorm, 1, sizeof(cl_mem), &gamma, "gamma") ||
        !set_arg_checked(g_k.layernorm, 2, sizeof(cl_mem), &beta, "beta") ||
        !set_arg_checked(g_k.layernorm, 3, sizeof(cl_mem), &out, "out") ||
        !set_arg_checked(g_k.layernorm, 4, sizeof(int), &rows, "rows") ||
        !set_arg_checked(g_k.layernorm, 5, sizeof(int), &cols, "cols") ||
        !set_arg_checked(g_k.layernorm, 6, sizeof(float), &eps, "eps")) return false;
    return run1d(q, g_k.layernorm, (size_t)rows, "layernorm");
}

bool adain(cl_command_queue q, cl_mem x, cl_mem nw, cl_mem nb,
           cl_mem fc_out, cl_mem scratch, cl_mem out, int C, int T) {
    if (!instancenorm(q, x, nw, nb, scratch, C, T)) return false;
    if (!set_arg_checked(g_k.adain_combine, 0, sizeof(cl_mem), &scratch, "n") ||
        !set_arg_checked(g_k.adain_combine, 1, sizeof(cl_mem), &fc_out, "fc_out") ||
        !set_arg_checked(g_k.adain_combine, 2, sizeof(cl_mem), &out, "out") ||
        !set_arg_checked(g_k.adain_combine, 3, sizeof(int), &C, "C") ||
        !set_arg_checked(g_k.adain_combine, 4, sizeof(int), &T, "T")) return false;
    return run1d(q, g_k.adain_combine, (size_t)C * T, "adain_combine");
}

cl_mem adain_fc(OpenCLContext& ctx, cl_command_queue q, Weights& w,
                cl_mem style_q, const std::string& w_key,
                const std::string& b_key, int C_out2) {
    cl_mem W = w.get_buffer(w_key);
    cl_mem B = w.get_buffer(b_key);
    if (!W || !B) { NNOPT_ERROR_FMT("adain_fc: missing %s / %s", w_key.c_str(), b_key.c_str()); return nullptr; }
    cl_mem out = alloc(ctx, (size_t)C_out2);
    if (!out) return nullptr;
    // fc weight is stored [K=128, N=2C] — ONNX MatMul layout, already transposed
    // relative to nn.Linear. pytorch_conv1d is the [K,N] GEMM.
    if (!pytorch_conv1d(q, 1, C_out2, 128, style_q, W, out)) { clReleaseMemObject(out); return nullptr; }
    if (!add_bias_rows(q, out, B, out, 1, C_out2)) { clReleaseMemObject(out); return nullptr; }
    return out;
}

bool leaky_relu(cl_command_queue q, cl_mem in, cl_mem out, int n) {
    if (!set_arg_checked(g_k.leaky, 0, sizeof(cl_mem), &in, "in") ||
        !set_arg_checked(g_k.leaky, 1, sizeof(cl_mem), &out, "out") ||
        !set_arg_checked(g_k.leaky, 2, sizeof(int), &n, "n")) return false;
    return run1d(q, g_k.leaky, (size_t)n, "leaky_relu");
}

bool add(cl_command_queue q, cl_mem a, cl_mem b, cl_mem out, int n) {
    if (!set_arg_checked(g_k.add, 0, sizeof(cl_mem), &a, "a") ||
        !set_arg_checked(g_k.add, 1, sizeof(cl_mem), &b, "b") ||
        !set_arg_checked(g_k.add, 2, sizeof(cl_mem), &out, "out") ||
        !set_arg_checked(g_k.add, 3, sizeof(int), &n, "n")) return false;
    return run1d(q, g_k.add, (size_t)n, "add");
}

bool add_scaled(cl_command_queue q, cl_mem a, cl_mem b, cl_mem out, int n, float scale) {
    if (!set_arg_checked(g_k.add_scaled, 0, sizeof(cl_mem), &a, "a") ||
        !set_arg_checked(g_k.add_scaled, 1, sizeof(cl_mem), &b, "b") ||
        !set_arg_checked(g_k.add_scaled, 2, sizeof(cl_mem), &out, "out") ||
        !set_arg_checked(g_k.add_scaled, 3, sizeof(int), &n, "n") ||
        !set_arg_checked(g_k.add_scaled, 4, sizeof(float), &scale, "scale")) return false;
    return run1d(q, g_k.add_scaled, (size_t)n, "add_scaled");
}

bool add_bias_ct(cl_command_queue q, cl_mem in, cl_mem bias, cl_mem out, int C, int T) {
    if (!set_arg_checked(g_k.bias_ct, 0, sizeof(cl_mem), &in, "in") ||
        !set_arg_checked(g_k.bias_ct, 1, sizeof(cl_mem), &bias, "bias") ||
        !set_arg_checked(g_k.bias_ct, 2, sizeof(cl_mem), &out, "out") ||
        !set_arg_checked(g_k.bias_ct, 3, sizeof(int), &C, "C") ||
        !set_arg_checked(g_k.bias_ct, 4, sizeof(int), &T, "T")) return false;
    return run1d(q, g_k.bias_ct, (size_t)C * T, "add_bias_ct");
}

bool add_bias_rows(cl_command_queue q, cl_mem in, cl_mem bias, cl_mem out, int rows, int cols) {
    if (!set_arg_checked(g_k.bias_rows, 0, sizeof(cl_mem), &in, "in") ||
        !set_arg_checked(g_k.bias_rows, 1, sizeof(cl_mem), &bias, "bias") ||
        !set_arg_checked(g_k.bias_rows, 2, sizeof(cl_mem), &out, "out") ||
        !set_arg_checked(g_k.bias_rows, 3, sizeof(int), &rows, "rows") ||
        !set_arg_checked(g_k.bias_rows, 4, sizeof(int), &cols, "cols")) return false;
    return run1d(q, g_k.bias_rows, (size_t)rows * cols, "add_bias_rows");
}

bool transpose(cl_command_queue q, cl_mem in, cl_mem out, int A, int B) {
    if (!set_arg_checked(g_k.transpose, 0, sizeof(cl_mem), &in, "in") ||
        !set_arg_checked(g_k.transpose, 1, sizeof(cl_mem), &out, "out") ||
        !set_arg_checked(g_k.transpose, 2, sizeof(int), &A, "A") ||
        !set_arg_checked(g_k.transpose, 3, sizeof(int), &B, "B")) return false;
    return run1d(q, g_k.transpose, (size_t)A * B, "transpose");
}

bool resize_nearest_t(cl_command_queue q, cl_mem in, cl_mem out, int C, int T_in, int factor) {
    if (!set_arg_checked(g_k.resize, 0, sizeof(cl_mem), &in, "in") ||
        !set_arg_checked(g_k.resize, 1, sizeof(cl_mem), &out, "out") ||
        !set_arg_checked(g_k.resize, 2, sizeof(int), &C, "C") ||
        !set_arg_checked(g_k.resize, 3, sizeof(int), &T_in, "T_in") ||
        !set_arg_checked(g_k.resize, 4, sizeof(int), &factor, "factor")) return false;
    return run1d(q, g_k.resize, (size_t)C * T_in * factor, "resize_nearest_t");
}

bool embed_tc(cl_command_queue q, cl_mem ids, cl_mem emb, cl_mem out, int T, int C) {
    if (!set_arg_checked(g_k.embed, 0, sizeof(cl_mem), &ids, "ids") ||
        !set_arg_checked(g_k.embed, 1, sizeof(cl_mem), &emb, "emb") ||
        !set_arg_checked(g_k.embed, 2, sizeof(cl_mem), &out, "out") ||
        !set_arg_checked(g_k.embed, 3, sizeof(int), &T, "T") ||
        !set_arg_checked(g_k.embed, 4, sizeof(int), &C, "C")) return false;
    return run1d(q, g_k.embed, (size_t)T * C, "embed_tc");
}

bool align_expand(cl_command_queue q, cl_mem in, cl_mem frame_src, cl_mem out,
                  int C, int T_in, int F) {
    if (!set_arg_checked(g_k.align, 0, sizeof(cl_mem), &in, "in") ||
        !set_arg_checked(g_k.align, 1, sizeof(cl_mem), &frame_src, "frame_src") ||
        !set_arg_checked(g_k.align, 2, sizeof(cl_mem), &out, "out") ||
        !set_arg_checked(g_k.align, 3, sizeof(int), &C, "C") ||
        !set_arg_checked(g_k.align, 4, sizeof(int), &T_in, "T_in") ||
        !set_arg_checked(g_k.align, 5, sizeof(int), &F, "F")) return false;
    return run1d(q, g_k.align, (size_t)C * F, "align_expand");
}

bool concat_ct(cl_command_queue q, const std::vector<cl_mem>& parts,
               const std::vector<int>& channels, cl_mem out, int T) {
    size_t off = 0;
    for (size_t i = 0; i < parts.size(); i++) {
        size_t bytes = (size_t)channels[i] * T * sizeof(nnopt_storage_t);
        cl_int e = clEnqueueCopyBuffer(q, parts[i], out, 0, off, bytes, 0, nullptr, nullptr);
        if (e != CL_SUCCESS) { NNOPT_ERROR_FMT("concat_ct copy %zu err %d", i, e); return false; }
        off += bytes;
    }
    return true;
}

// ── quantized BiLSTM ─────────────────────────────────────────────────────

void dql_host(const float* x, int n, std::vector<float>& deq) {
    float xmin = 0.0f, xmax = 0.0f;      // clamped against 0 per ONNX spec
    for (int i = 0; i < n; i++) { xmin = std::min(xmin, x[i]); xmax = std::max(xmax, x[i]); }
    deq.resize(n);
    float scale = (xmax - xmin) / 255.0f;   // computed in float, matching ORT
    if (scale <= 0.0f) { std::fill(deq.begin(), deq.end(), 0.0f); return; }
    float zp = std::rint(-xmin / scale);     // round-half-to-EVEN
    zp = std::min(255.0f, std::max(0.0f, zp));
    for (int i = 0; i < n; i++) {
        float q = std::rint(x[i] / scale) + zp;
        q = std::min(255.0f, std::max(0.0f, q));
        deq[i] = (q - zp) * scale;
    }
}

bool bilstm_quant(cl_command_queue q, Weights& weights,
                  cl_mem x_tc, int T, int I, int H,
                  const std::string& w_key, const std::string& r_key,
                  const std::string& b_key, cl_mem out) {
    std::vector<float> X;
    if (!download_as_f32(q, x_tc, (size_t)T * I, X)) return false;

    std::vector<float> W = weights.get_host_vec(w_key);   // [2, I, 4H]
    std::vector<float> R = weights.get_host_vec(r_key);   // [2, H, 4H]
    std::vector<float> B = weights.get_host_vec(b_key);   // [2, 8H]
    const int G = 4 * H;
    if ((int)W.size() != 2 * I * G || (int)R.size() != 2 * H * G || (int)B.size() != 2 * 2 * G) {
        NNOPT_ERROR_FMT("bilstm %s: weight size mismatch W=%zu R=%zu B=%zu (I=%d H=%d)",
                        w_key.c_str(), W.size(), R.size(), B.size(), I, H);
        return false;
    }

    // X is DQL'd ONCE over the whole [T,I] tensor and shared by both directions.
    std::vector<float> Xq;
    dql_host(X.data(), T * I, Xq);

    std::vector<float> Y((size_t)T * 2 * H, 0.0f);   // [T, 2H]
    std::vector<float> h(H), c(H), hq(H), g(G);

    for (int d = 0; d < 2; d++) {
        std::fill(h.begin(), h.end(), 0.0f);
        std::fill(c.begin(), c.end(), 0.0f);
        const float* Wd = W.data() + (size_t)d * I * G;
        const float* Rd = R.data() + (size_t)d * H * G;
        const float* Wb = B.data() + (size_t)d * 2 * G;
        const float* Rb = Wb + G;

        for (int step = 0; step < T; step++) {
            int t = (d == 0) ? step : (T - 1 - step);
            // h_{t-1} is re-quantized EVERY timestep (c_t is not).
            dql_host(h.data(), H, hq);

            for (int n = 0; n < G; n++) g[n] = Wb[n] + Rb[n];
            const float* xt = Xq.data() + (size_t)t * I;
            for (int i = 0; i < I; i++) {
                float xv = xt[i];
                if (xv == 0.0f) continue;
                const float* wrow = Wd + (size_t)i * G;
                for (int n = 0; n < G; n++) g[n] += xv * wrow[n];
            }
            for (int k = 0; k < H; k++) {
                float hv = hq[k];
                if (hv == 0.0f) continue;
                const float* rrow = Rd + (size_t)k * G;
                for (int n = 0; n < G; n++) g[n] += hv * rrow[n];
            }
            // ONNX gate order: i, o, f, c
            for (int u = 0; u < H; u++) {
                float gi = 1.0f / (1.0f + std::exp(-g[0 * H + u]));
                float go = 1.0f / (1.0f + std::exp(-g[1 * H + u]));
                float gf = 1.0f / (1.0f + std::exp(-g[2 * H + u]));
                float gc = std::tanh(g[3 * H + u]);
                float cn = gf * c[u] + gi * gc;
                c[u] = cn;
                h[u] = go * std::tanh(cn);
            }
            // Y[t] = [h_fwd (H) || h_bwd (H)]
            float* yrow = Y.data() + (size_t)t * 2 * H + (size_t)d * H;
            for (int u = 0; u < H; u++) yrow[u] = h[u];
        }
    }

    std::vector<nnopt_storage_t> host(Y.size());
    for (size_t i = 0; i < Y.size(); i++)
#ifdef NNOPT_USE_FP16
        host[i] = nnopt_f32_to_f16(Y[i]);
#else
        host[i] = Y[i];
#endif
    cl_int e = clEnqueueWriteBuffer(q, out, CL_TRUE, 0,
                                    host.size() * sizeof(nnopt_storage_t),
                                    host.data(), 0, nullptr, nullptr);
    if (e != CL_SUCCESS) { NNOPT_ERROR_FMT("bilstm write err %d", e); return false; }
    return true;
}


bool conv_transpose1d(cl_command_queue q, cl_mem in, cl_mem w, cl_mem bias,
                      cl_mem out, int Cin, int Cout, int T_in, int T_out,
                      int K, int stride, int pad) {
    int has_bias = bias ? 1 : 0;
    cl_mem bias_arg = bias ? bias : in;

    // OPT #2: tiled ConvTranspose (groups==1). Gated on span_ti fitting the LDS
    // tile. NNOPT_CONVT_TILED=0 forces naive for A/B.
    static const bool ctt_on = [] {
        const char* e = getenv("NNOPT_CONVT_TILED");
        return !(e && e[0] == '0');
    }();
    constexpr int CT_OT = 32, CT_OTWI = 8, CT_CO = 8, CT_SPAN_MAX = 48;
    auto fdiv = [](int a, int b) { return (a >= 0) ? (a / b) : -(((-a) + b - 1) / b); };
    // Worst-case tile is the LAST workgroup row (largest o_base) — but span_ti is
    // independent of o_base for fixed stride/K/CT_OT, so evaluate at o_base=0.
    const int span_ti = fdiv(CT_OT - 1 + pad, stride) - fdiv(pad - (K - 1), stride) + 1;
    if (ctt_on && span_ti <= CT_SPAN_MAX) {
        cl_kernel k = g_k.convtr_tiled;
        if (!set_arg_checked(k, 0, sizeof(cl_mem), &in, "in") ||
            !set_arg_checked(k, 1, sizeof(cl_mem), &w, "w") ||
            !set_arg_checked(k, 2, sizeof(cl_mem), &bias_arg, "bias") ||
            !set_arg_checked(k, 3, sizeof(cl_mem), &out, "out") ||
            !set_arg_checked(k, 4, sizeof(int), &Cin, "Cin") ||
            !set_arg_checked(k, 5, sizeof(int), &Cout, "Cout") ||
            !set_arg_checked(k, 6, sizeof(int), &T_in, "T_in") ||
            !set_arg_checked(k, 7, sizeof(int), &T_out, "T_out") ||
            !set_arg_checked(k, 8, sizeof(int), &K, "K") ||
            !set_arg_checked(k, 9, sizeof(int), &stride, "stride") ||
            !set_arg_checked(k, 10, sizeof(int), &pad, "pad") ||
            !set_arg_checked(k, 11, sizeof(int), &has_bias, "has_bias")) return false;
        size_t local[2]  = { (size_t)CT_OTWI, (size_t)CT_CO };
        size_t global[2] = {
            (size_t)((T_out + CT_OT - 1) / CT_OT) * CT_OTWI,
            (size_t)((Cout  + CT_CO - 1) / CT_CO) * CT_CO,
        };
        cl_int e = clEnqueueNDRangeKernel(q, k, 2, nullptr, global, local, 0, nullptr,
                                          KernelProfiler::event_for("convtranspose1d_tiled"));
        if (e != CL_SUCCESS) { NNOPT_ERROR_FMT("convtr_tiled dispatch err %d", e); return false; }
        return true;
    }

    if (!set_arg_checked(g_k.convtr, 0, sizeof(cl_mem), &in, "in") ||
        !set_arg_checked(g_k.convtr, 1, sizeof(cl_mem), &w, "w") ||
        !set_arg_checked(g_k.convtr, 2, sizeof(cl_mem), &bias_arg, "bias") ||
        !set_arg_checked(g_k.convtr, 3, sizeof(cl_mem), &out, "out") ||
        !set_arg_checked(g_k.convtr, 4, sizeof(int), &Cin, "Cin") ||
        !set_arg_checked(g_k.convtr, 5, sizeof(int), &Cout, "Cout") ||
        !set_arg_checked(g_k.convtr, 6, sizeof(int), &T_in, "T_in") ||
        !set_arg_checked(g_k.convtr, 7, sizeof(int), &T_out, "T_out") ||
        !set_arg_checked(g_k.convtr, 8, sizeof(int), &K, "K") ||
        !set_arg_checked(g_k.convtr, 9, sizeof(int), &stride, "stride") ||
        !set_arg_checked(g_k.convtr, 10, sizeof(int), &pad, "pad") ||
        !set_arg_checked(g_k.convtr, 11, sizeof(int), &has_bias, "has_bias")) return false;
    return run1d(q, g_k.convtr, (size_t)Cout * T_out, "convtranspose1d");
}

bool snake(cl_command_queue q, cl_mem in, cl_mem alpha, cl_mem inv_alpha,
           cl_mem out, int C, int T) {
    if (!set_arg_checked(g_k.snake, 0, sizeof(cl_mem), &in, "in") ||
        !set_arg_checked(g_k.snake, 1, sizeof(cl_mem), &alpha, "alpha") ||
        !set_arg_checked(g_k.snake, 2, sizeof(cl_mem), &inv_alpha, "inv_alpha") ||
        !set_arg_checked(g_k.snake, 3, sizeof(cl_mem), &out, "out") ||
        !set_arg_checked(g_k.snake, 4, sizeof(int), &C, "C") ||
        !set_arg_checked(g_k.snake, 5, sizeof(int), &T, "T")) return false;
    return run1d(q, g_k.snake, (size_t)C * T, "snake");
}

bool leaky_relu_a(cl_command_queue q, cl_mem in, cl_mem out, int n, float alpha) {
    if (!set_arg_checked(g_k.leaky_a, 0, sizeof(cl_mem), &in, "in") ||
        !set_arg_checked(g_k.leaky_a, 1, sizeof(cl_mem), &out, "out") ||
        !set_arg_checked(g_k.leaky_a, 2, sizeof(int), &n, "n") ||
        !set_arg_checked(g_k.leaky_a, 3, sizeof(float), &alpha, "alpha")) return false;
    return run1d(q, g_k.leaky_a, (size_t)n, "leaky_relu_a");
}

bool scale(cl_command_queue q, cl_mem in, cl_mem out, int n, float s) {
    if (!set_arg_checked(g_k.scalek, 0, sizeof(cl_mem), &in, "in") ||
        !set_arg_checked(g_k.scalek, 1, sizeof(cl_mem), &out, "out") ||
        !set_arg_checked(g_k.scalek, 2, sizeof(int), &n, "n") ||
        !set_arg_checked(g_k.scalek, 3, sizeof(float), &s, "scale")) return false;
    return run1d(q, g_k.scalek, (size_t)n, "scale");
}

bool exp_(cl_command_queue q, cl_mem in, cl_mem out, int n) {
    if (!set_arg_checked(g_k.expk, 0, sizeof(cl_mem), &in, "in") ||
        !set_arg_checked(g_k.expk, 1, sizeof(cl_mem), &out, "out") ||
        !set_arg_checked(g_k.expk, 2, sizeof(int), &n, "n")) return false;
    return run1d(q, g_k.expk, (size_t)n, "exp");
}

bool mag_phase(cl_command_queue q, cl_mem re, cl_mem im, cl_mem mag, cl_mem phase,
               int n, float eps) {
    if (!set_arg_checked(g_k.magphase, 0, sizeof(cl_mem), &re, "re") ||
        !set_arg_checked(g_k.magphase, 1, sizeof(cl_mem), &im, "im") ||
        !set_arg_checked(g_k.magphase, 2, sizeof(cl_mem), &mag, "mag") ||
        !set_arg_checked(g_k.magphase, 3, sizeof(cl_mem), &phase, "phase") ||
        !set_arg_checked(g_k.magphase, 4, sizeof(int), &n, "n") ||
        !set_arg_checked(g_k.magphase, 5, sizeof(float), &eps, "eps")) return false;
    return run1d(q, g_k.magphase, (size_t)n, "mag_phase");
}

bool polar_to_cart(cl_command_queue q, cl_mem mag, cl_mem raw, cl_mem re, cl_mem im, int n) {
    if (!set_arg_checked(g_k.polar, 0, sizeof(cl_mem), &mag, "mag") ||
        !set_arg_checked(g_k.polar, 1, sizeof(cl_mem), &raw, "raw") ||
        !set_arg_checked(g_k.polar, 2, sizeof(cl_mem), &re, "re") ||
        !set_arg_checked(g_k.polar, 3, sizeof(cl_mem), &im, "im") ||
        !set_arg_checked(g_k.polar, 4, sizeof(int), &n, "n")) return false;
    return run1d(q, g_k.polar, (size_t)n, "polar_to_cart");
}

bool istft_ola(cl_command_queue q, cl_mem re, cl_mem im, cl_mem basis_r, cl_mem basis_i,
               cl_mem out, int K, int Fr, int N, int hop, int T_out) {
    if (!set_arg_checked(g_k.istft, 0, sizeof(cl_mem), &re, "re") ||
        !set_arg_checked(g_k.istft, 1, sizeof(cl_mem), &im, "im") ||
        !set_arg_checked(g_k.istft, 2, sizeof(cl_mem), &basis_r, "basis_r") ||
        !set_arg_checked(g_k.istft, 3, sizeof(cl_mem), &basis_i, "basis_i") ||
        !set_arg_checked(g_k.istft, 4, sizeof(cl_mem), &out, "out") ||
        !set_arg_checked(g_k.istft, 5, sizeof(int), &K, "K") ||
        !set_arg_checked(g_k.istft, 6, sizeof(int), &Fr, "Fr") ||
        !set_arg_checked(g_k.istft, 7, sizeof(int), &N, "N") ||
        !set_arg_checked(g_k.istft, 8, sizeof(int), &hop, "hop") ||
        !set_arg_checked(g_k.istft, 9, sizeof(int), &T_out, "T_out")) return false;
    return run1d(q, g_k.istft, (size_t)T_out, "istft_ola");
}

bool pad_front_reflect(cl_command_queue q, cl_mem in, cl_mem out, int C, int T_in, int pad_front) {
    if (!set_arg_checked(g_k.padref, 0, sizeof(cl_mem), &in, "in") ||
        !set_arg_checked(g_k.padref, 1, sizeof(cl_mem), &out, "out") ||
        !set_arg_checked(g_k.padref, 2, sizeof(int), &C, "C") ||
        !set_arg_checked(g_k.padref, 3, sizeof(int), &T_in, "T_in") ||
        !set_arg_checked(g_k.padref, 4, sizeof(int), &pad_front, "pad_front")) return false;
    return run1d(q, g_k.padref, (size_t)C * (T_in + pad_front), "pad_front_reflect");
}

}  // namespace st
