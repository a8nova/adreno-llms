// moonshine_common.cpp — shared OpenCL program cache + RoPE table builder
// for the Moonshine ASR port.
// Reference: model_info/transformers_src/modeling_moonshine.py MoonshineRotaryEmbedding.forward
//   compute_default_rope_parameters + forward(): inv_freq = 1/base^(arange(0,dim,2)/dim),
//   freqs = pos * inv_freq, emb = cat(freqs,freqs), cos/sin.
//   apply_rotary_pos_emb then takes cos[..., :dim//2].repeat_interleave(2)
//   → our host table stores the interleaved cos/sin of shape [max_pos, rotary_dim].
//
// This file is NOT a graph op (no <Class>_forward). It provides internal helpers
// used by the attention/conv ops. It contains real math (RoPE table generation).

#include "../opencl_context.h"
#include "../nnopt_error.h"
#include "../model_config.h"
#include "../utils.h"
#include "../profiler.h"
#include <string>
#include <unordered_map>
#include <vector>
#include <cmath>
#include <cstdlib>

// ── kernel → file routing (whisper-parity per-op kernel files) ──────────
// Each op owns a kernel file; this table is the single source of truth for
// where a kernel lives. Programs are built lazily per FILE (each with its
// own <file>.bin binary cache — see OpenCLContext::load_or_build_program).
static const char* moonshine_kernel_file(const char* name) {
    static const std::unordered_map<std::string, const char*> table = {
        {"attention",        "kernels/attn.cl"},
        {"linear_gemv",      "kernels/gemm.cl"},
        {"linear_gemv_t",    "kernels/gemm.cl"},
        {"linear_gemv_ks",   "kernels/gemm.cl"},
        {"gemv_ks_reduce",   "kernels/gemm.cl"},
        {"linear_gemm_t",    "kernels/gemm.cl"},
        {"linear_gemm_t4",   "kernels/gemm.cl"},
        {"mat_transpose",    "kernels/gemm.cl"},
        {"conv1d",           "kernels/conv.cl"},
        {"im2col_1d",        "kernels/conv.cl"},
        {"layernorm_nobias", "kernels/norms.cl"},
        {"groupnorm_stats",  "kernels/norms.cl"},
        {"groupnorm_apply",  "kernels/norms.cl"},
        {"gelu_act",         "kernels/activations.cl"},
        {"silu_act",         "kernels/activations.cl"},
        {"glu_silu",         "kernels/activations.cl"},
        {"tanh_act",         "kernels/activations.cl"},
        {"embed_gather",     "kernels/embedding.cl"},
        {"rope_apply",       "kernels/pack.cl"},
        {"split_heads",      "kernels/pack.cl"},
        {"merge_heads",      "kernels/pack.cl"},
        {"permute_cl_to_lc", "kernels/pack.cl"},
        {"bias_add",         "kernels/pack.cl"},
    };
    auto it = table.find(name);
    return it != table.end() ? it->second : nullptr;
}

// Build (once per file) and return a kernel file's program.
static cl_program moonshine_program_for(OpenCLContext& cl_ctx, const char* path) {
    static std::unordered_map<std::string, cl_program> cache;
    auto it = cache.find(path);
    if (it != cache.end()) return it->second;
    // OPT-10 (r8): -cl-mad-enable + -cl-fast-relaxed-math. Gated on the
    // 3-clip token-exact check; NNOPT_FAST_MATH=0 reverts at runtime.
    const char* fm = std::getenv("NNOPT_FAST_MATH");
    const std::string opts = (fm && fm[0] == '0')
        ? std::string()
        : std::string("-cl-mad-enable -cl-fast-relaxed-math");
    cl_program prog = cl_ctx.build_program_from_file(path, opts);  // PROGRAM-INIT-OK
    if (!prog) { NNOPT_ERROR_FMT("moonshine_program_for: build failed for %s", path); return nullptr; }
    cache.emplace(path, prog);
    return prog;
}

// OPT-6 (r6): kernel-object cache. clCreateKernel ran on EVERY dispatch
// (~2000 host round-trips per clip). cl_kernel objects are reusable — args
// are re-set fully before each enqueue and the host is single-threaded.
// Kernels live for the process lifetime, same as their programs.
cl_kernel moonshine_kernel(OpenCLContext& cl_ctx, const char* name) {
    static std::unordered_map<std::string, cl_kernel> cache;
    auto it = cache.find(name);
    if (it != cache.end()) return it->second;
    const char* path = moonshine_kernel_file(name);
    if (!path) { NNOPT_ERROR_FMT("moonshine_kernel: no kernel file mapped for '%s'", name); return nullptr; }
    cl_program prog = moonshine_program_for(cl_ctx, path);
    if (!prog) return nullptr;
    cl_int err = CL_SUCCESS;
    cl_kernel k = clCreateKernel(prog, name, &err);
    if (err != CL_SUCCESS || !k) {
        NNOPT_ERROR_FMT("moonshine_kernel: clCreateKernel(%s) from %s failed %d", name, path, err);
        return nullptr;
    }
    cache.emplace(name, k);
    return k;
}

// Build (once) interleaved RoPE cos/sin tables on device.
// rotary_dim = int(head_dim * partial_rotary_factor) rounded down; we store
// tables of shape [max_pos, rotary_dim] where entry [p, 2i] == entry [p, 2i+1]
// == cos(p * inv_freq[i]) (the repeat_interleave(2) layout apply_rotary uses).
// max_pos must cover the longest sequence (encoder ~207 frames > decoder).
// Returns cached buffers; caller must NOT release them.
bool moonshine_rope_tables(OpenCLContext& cl_ctx,
                           int max_pos,
                           int head_dim,
                           float theta_base,
                           cl_mem& cos_out,
                           cl_mem& sin_out) {
    static cl_mem cached_cos = nullptr;
    static cl_mem cached_sin = nullptr;
    static int cached_max_pos = 0;
    static int cached_rotary_dim = 0;

    const int rotary_dim = (int)((float)head_dim * MODEL_CONFIG::PARTIAL_ROTARY_FACTOR);
    // rotary_dim must be even for interleaved pairs.
    const int rdim = rotary_dim - (rotary_dim % 2);

    if (cached_cos && cached_max_pos >= max_pos && cached_rotary_dim == rdim) {
        cos_out = cached_cos; sin_out = cached_sin;
        return true;
    }
    if (cached_cos) { clReleaseMemObject(cached_cos); cached_cos = nullptr; }
    if (cached_sin) { clReleaseMemObject(cached_sin); cached_sin = nullptr; }

    // inv_freq[i] = 1 / base^(2i / rotary_dim_full)  where the PyTorch dim used
    // for inv_freq is rdim itself (dim = int(head_dim * partial_rotary_factor)).
    const int half = rdim / 2;
    std::vector<float> host_cos((size_t)max_pos * rdim);
    std::vector<float> host_sin((size_t)max_pos * rdim);
    for (int p = 0; p < max_pos; ++p) {
        for (int i = 0; i < half; ++i) {
            float inv_freq = 1.0f / std::pow(theta_base, (float)(2 * i) / (float)rdim);
            float ang = (float)p * inv_freq;
            float c = std::cos(ang);
            float s = std::sin(ang);
            // repeat_interleave(2): positions 2i and 2i+1 share the same value.
            host_cos[(size_t)p * rdim + 2 * i]     = c;
            host_cos[(size_t)p * rdim + 2 * i + 1] = c;
            host_sin[(size_t)p * rdim + 2 * i]     = s;
            host_sin[(size_t)p * rdim + 2 * i + 1] = s;
        }
    }

    std::vector<nnopt_storage_t> scos(host_cos.size());
    std::vector<nnopt_storage_t> ssin(host_sin.size());
#ifdef NNOPT_USE_FP16
    for (size_t i = 0; i < host_cos.size(); ++i) scos[i] = nnopt_f32_to_f16(host_cos[i]);
    for (size_t i = 0; i < host_sin.size(); ++i) ssin[i] = nnopt_f32_to_f16(host_sin[i]);
#else
    for (size_t i = 0; i < host_cos.size(); ++i) scos[i] = host_cos[i];
    for (size_t i = 0; i < host_sin.size(); ++i) ssin[i] = host_sin[i];
#endif

    cl_int err = CL_SUCCESS;
    cached_cos = clCreateBuffer(cl_ctx.context(), CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR,
                                scos.size() * sizeof(nnopt_storage_t), scos.data(), &err);
    if (err != CL_SUCCESS) { NNOPT_ERROR_FMT("rope cos alloc %d", err); return false; }
    cached_sin = clCreateBuffer(cl_ctx.context(), CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR,
                                ssin.size() * sizeof(nnopt_storage_t), ssin.data(), &err);
    if (err != CL_SUCCESS) { NNOPT_ERROR_FMT("rope sin alloc %d", err); return false; }

    cached_max_pos = max_pos;
    cached_rotary_dim = rdim;
    cos_out = cached_cos; sin_out = cached_sin;
    return true;
}

// Shared utils.cl program (element_add / split_last_dim_2). Built once.
cl_program moonshine_utils_program(OpenCLContext& cl_ctx) {
    static cl_program prog = nullptr;
    if (!prog) prog = cl_ctx.build_program_from_file("kernels/utils.cl");  // PROGRAM-INIT-OK
    return prog;
}

// Residual add: returns a NEW buffer a+b (n elems). Uses element_add (utils.cl).
cl_mem moonshine_resid_add(OpenCLContext& cl_ctx, cl_command_queue q,
                           cl_mem a, cl_mem b, int n) {
    return element_add(q, moonshine_utils_program(cl_ctx), a, b, (size_t)n);
}

// GELU: exact erf-based GELU over `n` elements. Returns a NEW buffer (caller owns).
cl_mem gelu_apply(OpenCLContext& cl_ctx, cl_command_queue queue, cl_mem input, int n) {
    cl_int err = CL_SUCCESS;
    cl_mem out = clCreateBuffer(cl_ctx.context(), CL_MEM_READ_WRITE,
                               (size_t)n * sizeof(nnopt_storage_t), nullptr, &err);
    if (err != CL_SUCCESS || !out) { NNOPT_ERROR_FMT("gelu_apply: alloc %d", err); return nullptr; }
    cl_kernel k = moonshine_kernel(cl_ctx, "gelu_act");
    if (!k) { NNOPT_ERROR("gelu_apply: kernel"); clReleaseMemObject(out); return nullptr; }
    auto cleanup = [&]() -> cl_mem { clReleaseMemObject(out); return nullptr; };
    if (!set_arg_checked(k, 0, sizeof(cl_mem), &input, "in")) return cleanup();
    if (!set_arg_checked(k, 1, sizeof(cl_mem), &out, "out")) return cleanup();
    if (!set_arg_checked(k, 2, sizeof(int), &n, "n")) return cleanup();
    size_t gws = (size_t)n;
    err = clEnqueueNDRangeKernel(queue, k, 1, nullptr, &gws, nullptr, 0, nullptr,
                                 KernelProfiler::event_for("gelu_act"));
    if (err != CL_SUCCESS) { NNOPT_ERROR_FMT("gelu_apply: dispatch %d", err); return cleanup(); }
    return out;
}
