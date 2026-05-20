// Reference: /Users/alazarshenkute/.nnopt/ref_venvs/env_stable/lib/python3.12/site-packages/transformers/models/idefics3/modeling_idefics3.py:189-265 Idefics3VisionAttention.forward
// Implements eager vision attention (non-causal):
//   Q = q_proj(x), K = k_proj(x), V = v_proj(x)
//   scores = (Q @ K^T) * scale
//   if attention_mask: scores += attention_mask
//   probs = softmax(scores)
//   ctx = probs @ V
//   out = out_proj(ctx)
// Returns out (attn_weights not returned in this inference port).

#include "opencl_context.h"
#include "weights.h"
#include "debug_utils.h"
#include "utils.h"
#include "model_config.h"
#include "profile.h"

#include <CL/cl.h>
#include <cstddef>
#include <cstdint>
#include <string>

namespace {

struct VisionAttnState {
  bool initialized = false;
  cl_program program = nullptr;
  cl_kernel k_scores = nullptr;
  cl_kernel k_scores_tiled = nullptr;   // tiled fp16 mha_scores — kept for split-K experiments
  cl_kernel k_scores_par = nullptr;     // mha_scores_par — 1 WG per (b,h,tq), 64-lane stride over tk
  cl_kernel k_scores_coal = nullptr;    // mha_scores_coalesced — 64-lane cooperative dot, coalesced K reads
  cl_kernel k_add_mask = nullptr;
  cl_kernel k_softmax = nullptr;
  cl_kernel k_softmax_par = nullptr;  // mha_softmax_parallel — WG per row, 64-lane reduce
  cl_kernel k_out = nullptr;
  cl_kernel k_out_par = nullptr;      // mha_out_par — WG per (b,h,tq), probs in __local
  cl_kernel k_out_coal = nullptr;     // mha_out_coalesced — 64-lane cooperative V read
  cl_kernel k_flash = nullptr;        // mha_flash_attn — fused scores+softmax+out, online softmax
};

VisionAttnState& state() {
  static VisionAttnState s;
  return s;
}

bool ensure_initialized(OpenCLContext& cl_ctx) {
  auto& s = state();
  if (s.initialized) return true;

  s.program = cl_ctx.build_program_from_file("kernels/vision_attention.cl");  // PROGRAM-INIT-OK
  if (!s.program) {
    NNOPT_ERROR("Idefics3VisionAttention: failed to build kernels/vision_attention.cl");
    return false;
  }

  cl_int err = CL_SUCCESS;
  s.k_scores = clCreateKernel(s.program, "mha_scores", &err);
  if (err != CL_SUCCESS || !s.k_scores) {
    NNOPT_ERROR_FMT("Idefics3VisionAttention: clCreateKernel(mha_scores) failed (%d)", (int)err);
    return false;
  }
  s.k_scores_tiled = clCreateKernel(s.program, "mha_scores_tiled", &err);
  if (err != CL_SUCCESS || !s.k_scores_tiled) {
    fprintf(stderr, "[vision_attn] mha_scores_tiled unavailable (%d) — falling back\n", (int)err);
    s.k_scores_tiled = nullptr;
  }
  s.k_scores_par = clCreateKernel(s.program, "mha_scores_par", &err);
  if (err != CL_SUCCESS || !s.k_scores_par) {
    fprintf(stderr, "[vision_attn] mha_scores_par unavailable (%d) — falling back to naive\n", (int)err);
    s.k_scores_par = nullptr;
  }
  s.k_scores_coal = clCreateKernel(s.program, "mha_scores_coalesced", &err);
  if (err != CL_SUCCESS || !s.k_scores_coal) {
    fprintf(stderr, "[vision_attn] mha_scores_coalesced unavailable (%d) — falling back\n", (int)err);
    s.k_scores_coal = nullptr;
  }
  s.k_add_mask = clCreateKernel(s.program, "mha_add_mask", &err);
  if (err != CL_SUCCESS || !s.k_add_mask) {
    NNOPT_ERROR_FMT("Idefics3VisionAttention: clCreateKernel(mha_add_mask) failed (%d)", (int)err);
    return false;
  }
  s.k_softmax = clCreateKernel(s.program, "mha_softmax", &err);
  if (err != CL_SUCCESS || !s.k_softmax) {
    NNOPT_ERROR_FMT("Idefics3VisionAttention: clCreateKernel(mha_softmax) failed (%d)", (int)err);
    return false;
  }
  s.k_softmax_par = clCreateKernel(s.program, "mha_softmax_parallel", &err);
  if (err != CL_SUCCESS || !s.k_softmax_par) {
    fprintf(stderr, "[vision_attn] mha_softmax_parallel unavailable (%d) — falling back to naive\n", (int)err);
    s.k_softmax_par = nullptr;
  }
  s.k_out = clCreateKernel(s.program, "mha_out", &err);
  if (err != CL_SUCCESS || !s.k_out) {
    NNOPT_ERROR_FMT("Idefics3VisionAttention: clCreateKernel(mha_out) failed (%d)", (int)err);
    return false;
  }
  s.k_out_par = clCreateKernel(s.program, "mha_out_par", &err);
  if (err != CL_SUCCESS || !s.k_out_par) {
    fprintf(stderr, "[vision_attn] mha_out_par unavailable (%d) — falling back to naive\n", (int)err);
    s.k_out_par = nullptr;
  }
  s.k_out_coal = clCreateKernel(s.program, "mha_out_coalesced", &err);
  if (err != CL_SUCCESS || !s.k_out_coal) {
    fprintf(stderr, "[vision_attn] mha_out_coalesced unavailable (%d) — falling back\n", (int)err);
    s.k_out_coal = nullptr;
  }
  s.k_flash = clCreateKernel(s.program, "mha_flash_attn", &err);
  if (err != CL_SUCCESS || !s.k_flash) {
    fprintf(stderr, "[vision_attn] mha_flash_attn unavailable (%d) — using separate scores/softmax/out\n", (int)err);
    s.k_flash = nullptr;
  }

  s.initialized = true;
  return true;
}

}  // namespace

extern "C" cl_mem op_Idefics3VisionAttention(OpenCLContext& cl_ctx,
                                             Weights& weights,
                                             cl_command_queue queue,
                                             cl_mem hidden_states,        // [B, T, C] flat
                                             cl_mem attention_mask_opt,   // [B, 1, T, T] OR null
                                             int B,
                                             int T,
                                             int C,
                                             int num_heads,
                                             const char* q_w,
                                             const char* q_b,
                                             const char* k_w,
                                             const char* k_b,
                                             const char* v_w,
                                             const char* v_b,
                                             const char* o_w,
                                             const char* o_b) {
  if (!ensure_initialized(cl_ctx)) return nullptr;

  if (!queue || !hidden_states) {
    NNOPT_ERROR("op_Idefics3VisionAttention: null queue/hidden_states");
    return nullptr;
  }

  const int head_dim = C / num_heads;
  if (head_dim * num_heads != C) {
    NNOPT_ERROR("op_Idefics3VisionAttention: C not divisible by num_heads");
    return nullptr;
  }

  // Flatten B*T rows for the projections.
  const int rows = B * T;

  cl_mem Wq = weights.get_buffer(std::string(q_w));
  cl_mem Wk = weights.get_buffer(std::string(k_w));
  cl_mem Wv = weights.get_buffer(std::string(v_w));
  cl_mem Wo = weights.get_buffer(std::string(o_w));
  if (!Wq || !Wk || !Wv || !Wo) {
    NNOPT_ERROR("op_Idefics3VisionAttention: missing projection weight");
    return nullptr;
  }

  cl_mem Bq = (q_b && q_b[0]) ? weights.get_buffer(std::string(q_b), true) : nullptr;
  cl_mem Bk = (k_b && k_b[0]) ? weights.get_buffer(std::string(k_b), true) : nullptr;
  cl_mem Bv = (v_b && v_b[0]) ? weights.get_buffer(std::string(v_b), true) : nullptr;
  cl_mem Bo = (o_b && o_b[0]) ? weights.get_buffer(std::string(o_b), true) : nullptr;

  // Allocate Q/K/V: [B,H,T,D] => elems = B*H*T*D = B*T*C
  const size_t qkv_elems = (size_t)rows * (size_t)C;
  cl_int err = CL_SUCCESS;
  cl_mem Q = clCreateBuffer(cl_ctx.context(), CL_MEM_READ_WRITE, qkv_elems * sizeof(nnopt_storage_t), nullptr, &err);
  cl_mem K = clCreateBuffer(cl_ctx.context(), CL_MEM_READ_WRITE, qkv_elems * sizeof(nnopt_storage_t), nullptr, &err);
  cl_mem V = clCreateBuffer(cl_ctx.context(), CL_MEM_READ_WRITE, qkv_elems * sizeof(nnopt_storage_t), nullptr, &err);
  if (!Q || !K || !V) {
    if (Q) clReleaseMemObject(Q);
    if (K) clReleaseMemObject(K);
    if (V) clReleaseMemObject(V);
    NNOPT_ERROR("op_Idefics3VisionAttention: failed to alloc Q/K/V");
    return nullptr;
  }

  // Projections (treat hidden_states as [rows,C])
  NNOPT_PROFILE_BEGIN(queue, "vis_qproj");
  if (!pytorch_linear(queue, rows, C, C, hidden_states, Wq, Q)) {
    NNOPT_PROFILE_END(queue, "vis_qproj");
    NNOPT_ERROR("op_Idefics3VisionAttention: q_proj GEMM failed");
    clReleaseMemObject(Q); clReleaseMemObject(K); clReleaseMemObject(V);
    return nullptr;
  }
  NNOPT_PROFILE_END(queue, "vis_qproj");
  if (Bq) {
    // reuse bias_add_2d (already in kernels/bias_add.cl)
    static cl_program bias_prog = nullptr;
    static cl_kernel bias_k = nullptr;
    static bool bias_init = false;
    if (!bias_init) {
      bias_prog = cl_ctx.build_program_from_file("kernels/bias_add.cl");  // PROGRAM-INIT-OK
      if (!bias_prog) {
        NNOPT_ERROR("op_Idefics3VisionAttention: build bias_add.cl failed");
        clReleaseMemObject(Q); clReleaseMemObject(K); clReleaseMemObject(V);
        return nullptr;
      }
      cl_int e2 = CL_SUCCESS;
      bias_k = clCreateKernel(bias_prog, "bias_add_2d", &e2);
      if (e2 != CL_SUCCESS || !bias_k) {
        NNOPT_ERROR_FMT("op_Idefics3VisionAttention: clCreateKernel(bias_add_2d) failed (%d)", (int)e2);
        clReleaseMemObject(Q); clReleaseMemObject(K); clReleaseMemObject(V);
        return nullptr;
      }
      bias_init = true;
    }
    if (!set_arg_checked(bias_k, 0, sizeof(cl_mem), &Q, "x")) { clReleaseMemObject(Q); clReleaseMemObject(K); clReleaseMemObject(V); return nullptr; }
    if (!set_arg_checked(bias_k, 1, sizeof(cl_mem), &Bq, "bias")) { clReleaseMemObject(Q); clReleaseMemObject(K); clReleaseMemObject(V); return nullptr; }
    if (!set_arg_checked(bias_k, 2, sizeof(int), &rows, "rows")) { clReleaseMemObject(Q); clReleaseMemObject(K); clReleaseMemObject(V); return nullptr; }
    if (!set_arg_checked(bias_k, 3, sizeof(int), &C, "cols")) { clReleaseMemObject(Q); clReleaseMemObject(K); clReleaseMemObject(V); return nullptr; }
    const size_t gwsb[2] = {(size_t)rows, (size_t)C};
    cl_int e3 = clEnqueueNDRangeKernel(queue, bias_k, 2, nullptr, gwsb, nullptr, 0, nullptr, nullptr);
    if (e3 != CL_SUCCESS) {
      NNOPT_ERROR_FMT("op_Idefics3VisionAttention: bias_add Q failed (%d)", (int)e3);
      clReleaseMemObject(Q); clReleaseMemObject(K); clReleaseMemObject(V);
      return nullptr;
    }
  }

  NNOPT_PROFILE_BEGIN(queue, "vis_kproj");
  if (!pytorch_linear(queue, rows, C, C, hidden_states, Wk, K)) {
    NNOPT_PROFILE_END(queue, "vis_kproj");
    NNOPT_ERROR("op_Idefics3VisionAttention: k_proj GEMM failed");
    clReleaseMemObject(Q); clReleaseMemObject(K); clReleaseMemObject(V);
    return nullptr;
  }
  NNOPT_PROFILE_END(queue, "vis_kproj");
  if (Bk) {
    // bias add for K
    static cl_program bias_prog2 = nullptr;
    static cl_kernel bias_k2 = nullptr;
    static bool bias_init2 = false;
    if (!bias_init2) {
      bias_prog2 = cl_ctx.build_program_from_file("kernels/bias_add.cl");  // PROGRAM-INIT-OK
      cl_int e2 = CL_SUCCESS;
      bias_k2 = clCreateKernel(bias_prog2, "bias_add_2d", &e2);
      if (e2 != CL_SUCCESS || !bias_k2) {
        NNOPT_ERROR_FMT("op_Idefics3VisionAttention: clCreateKernel(bias_add_2d) failed (%d)", (int)e2);
        clReleaseMemObject(Q); clReleaseMemObject(K); clReleaseMemObject(V);
        return nullptr;
      }
      bias_init2 = true;
    }
    if (!set_arg_checked(bias_k2, 0, sizeof(cl_mem), &K, "x")) { clReleaseMemObject(Q); clReleaseMemObject(K); clReleaseMemObject(V); return nullptr; }
    if (!set_arg_checked(bias_k2, 1, sizeof(cl_mem), &Bk, "bias")) { clReleaseMemObject(Q); clReleaseMemObject(K); clReleaseMemObject(V); return nullptr; }
    if (!set_arg_checked(bias_k2, 2, sizeof(int), &rows, "rows")) { clReleaseMemObject(Q); clReleaseMemObject(K); clReleaseMemObject(V); return nullptr; }
    if (!set_arg_checked(bias_k2, 3, sizeof(int), &C, "cols")) { clReleaseMemObject(Q); clReleaseMemObject(K); clReleaseMemObject(V); return nullptr; }
    const size_t gwsb[2] = {(size_t)rows, (size_t)C};
    cl_int e3 = clEnqueueNDRangeKernel(queue, bias_k2, 2, nullptr, gwsb, nullptr, 0, nullptr, nullptr);
    if (e3 != CL_SUCCESS) {
      NNOPT_ERROR_FMT("op_Idefics3VisionAttention: bias_add K failed (%d)", (int)e3);
      clReleaseMemObject(Q); clReleaseMemObject(K); clReleaseMemObject(V);
      return nullptr;
    }
  }

  NNOPT_PROFILE_BEGIN(queue, "vis_vproj");
  if (!pytorch_linear(queue, rows, C, C, hidden_states, Wv, V)) {
    NNOPT_PROFILE_END(queue, "vis_vproj");
    NNOPT_ERROR("op_Idefics3VisionAttention: v_proj GEMM failed");
    clReleaseMemObject(Q); clReleaseMemObject(K); clReleaseMemObject(V);
    return nullptr;
  }
  NNOPT_PROFILE_END(queue, "vis_vproj");
  if (Bv) {
    static cl_program bias_prog3 = nullptr;
    static cl_kernel bias_k3 = nullptr;
    static bool bias_init3 = false;
    if (!bias_init3) {
      bias_prog3 = cl_ctx.build_program_from_file("kernels/bias_add.cl");  // PROGRAM-INIT-OK
      cl_int e2 = CL_SUCCESS;
      bias_k3 = clCreateKernel(bias_prog3, "bias_add_2d", &e2);
      if (e2 != CL_SUCCESS || !bias_k3) {
        NNOPT_ERROR_FMT("op_Idefics3VisionAttention: clCreateKernel(bias_add_2d) failed (%d)", (int)e2);
        clReleaseMemObject(Q); clReleaseMemObject(K); clReleaseMemObject(V);
        return nullptr;
      }
      bias_init3 = true;
    }
    if (!set_arg_checked(bias_k3, 0, sizeof(cl_mem), &V, "x")) { clReleaseMemObject(Q); clReleaseMemObject(K); clReleaseMemObject(V); return nullptr; }
    if (!set_arg_checked(bias_k3, 1, sizeof(cl_mem), &Bv, "bias")) { clReleaseMemObject(Q); clReleaseMemObject(K); clReleaseMemObject(V); return nullptr; }
    if (!set_arg_checked(bias_k3, 2, sizeof(int), &rows, "rows")) { clReleaseMemObject(Q); clReleaseMemObject(K); clReleaseMemObject(V); return nullptr; }
    if (!set_arg_checked(bias_k3, 3, sizeof(int), &C, "cols")) { clReleaseMemObject(Q); clReleaseMemObject(K); clReleaseMemObject(V); return nullptr; }
    const size_t gwsb[2] = {(size_t)rows, (size_t)C};
    cl_int e3 = clEnqueueNDRangeKernel(queue, bias_k3, 2, nullptr, gwsb, nullptr, 0, nullptr, nullptr);
    if (e3 != CL_SUCCESS) {
      NNOPT_ERROR_FMT("op_Idefics3VisionAttention: bias_add V failed (%d)", (int)e3);
      clReleaseMemObject(Q); clReleaseMemObject(K); clReleaseMemObject(V);
      return nullptr;
    }
  }

  // BUG FIX (2026-05-20): The Q/K/V projections produce buffers in
  // [B, T, H*D] = [B, T, H, D] row-major flat. Previously this code claimed
  // "interpret as [B, H, T, D]" without an actual transpose, but the strides
  // differ: [B,H,T,D] addr = ((b*H+h)*T + t)*D + d  vs  [B,T,H,D] addr =
  // ((b*T+t)*H + h)*D + d. The mha_scores / mha_softmax / mha_out kernels
  // index Q/K/V with [B,H,T,D] strides, so without the transpose they read
  // scrambled data — every (h>0 or t>0) entry pulled from the wrong slot,
  // and attention essentially mixed unrelated query/key/value vectors. The
  // residual symptom: identical descriptions for different images, even after
  // the embedding-transpose fix landed earlier today.
  //
  // The fix: explicitly transpose Q/K/V from [B,T,H,D] to [B,H,T,D] post-bias.
  {
    static cl_program tr_qkv_prog = nullptr;
    static cl_kernel  tr_qkv_k    = nullptr;
    static bool       tr_qkv_init = false;
    if (!tr_qkv_init) {
      const char* tr_src = R"CLC(
#ifdef USE_FP16
  #pragma OPENCL EXTENSION cl_khr_fp16 : enable
  typedef half storage_t;
  #define LOAD(p, i)     vload_half((i), (p))
  #define STORE(p, i, v) vstore_half((v), (i), (p))
#else
  typedef float storage_t;
  #define LOAD(p, i)     ((p)[(i)])
  #define STORE(p, i, v) ((p)[(i)] = (v))
#endif

// in:  [B, T, H, D] row-major flat
// out: [B, H, T, D] row-major flat
__kernel void bthd_to_bhtd(
    __global const storage_t* in,
    __global storage_t* out,
    const int B, const int T, const int H, const int D) {
  const int b  = (int)get_global_id(0);
  const int h  = (int)get_global_id(1);
  const int td = (int)get_global_id(2);
  if (b >= B || h >= H || td >= T * D) return;
  const int t = td / D;
  const int d = td - t * D;
  const long in_idx  = (((long)b * (long)T + (long)t) * (long)H + (long)h) * (long)D + (long)d;
  const long out_idx = (((long)b * (long)H + (long)h) * (long)T + (long)t) * (long)D + (long)d;
  const float v = (float)LOAD(in, in_idx);
  STORE(out, out_idx, (storage_t)v);
}
)CLC";
      tr_qkv_prog = cl_ctx.build_program(std::string(tr_src));  // PROGRAM-INIT-OK
      if (tr_qkv_prog) {
        cl_int te = CL_SUCCESS;
        tr_qkv_k = clCreateKernel(tr_qkv_prog, "bthd_to_bhtd", &te);
        if (te == CL_SUCCESS && tr_qkv_k) tr_qkv_init = true;
      }
    }
    if (tr_qkv_init) {
      cl_int alloc_err = CL_SUCCESS;
      auto run_tr = [&](cl_mem src) -> cl_mem {
        cl_mem dst = clCreateBuffer(cl_ctx.context(), CL_MEM_READ_WRITE,
                                    qkv_elems * sizeof(nnopt_storage_t),
                                    nullptr, &alloc_err);
        if (alloc_err != CL_SUCCESS || !dst) return nullptr;
        if (!set_arg_checked(tr_qkv_k, 0, sizeof(cl_mem), &src, "in")) { clReleaseMemObject(dst); return nullptr; }
        if (!set_arg_checked(tr_qkv_k, 1, sizeof(cl_mem), &dst, "out")) { clReleaseMemObject(dst); return nullptr; }
        if (!set_arg_checked(tr_qkv_k, 2, sizeof(int), &B, "B")) { clReleaseMemObject(dst); return nullptr; }
        if (!set_arg_checked(tr_qkv_k, 3, sizeof(int), &T, "T")) { clReleaseMemObject(dst); return nullptr; }
        if (!set_arg_checked(tr_qkv_k, 4, sizeof(int), &num_heads, "H")) { clReleaseMemObject(dst); return nullptr; }
        if (!set_arg_checked(tr_qkv_k, 5, sizeof(int), &head_dim, "D")) { clReleaseMemObject(dst); return nullptr; }
        const size_t gws[3] = {(size_t)B, (size_t)num_heads, (size_t)(T * head_dim)};
        cl_int e = clEnqueueNDRangeKernel(queue, tr_qkv_k, 3, nullptr, gws, nullptr, 0, nullptr, nullptr);
        if (e != CL_SUCCESS) { clReleaseMemObject(dst); return nullptr; }
        return dst;
      };
      cl_mem Qt = run_tr(Q);
      cl_mem Kt = run_tr(K);
      cl_mem Vt = run_tr(V);
      if (!Qt || !Kt || !Vt) {
        NNOPT_ERROR("op_Idefics3VisionAttention: Q/K/V transpose failed");
        if (Qt) clReleaseMemObject(Qt);
        if (Kt) clReleaseMemObject(Kt);
        if (Vt) clReleaseMemObject(Vt);
        clReleaseMemObject(Q); clReleaseMemObject(K); clReleaseMemObject(V);
        return nullptr;
      }
      clReleaseMemObject(Q); clReleaseMemObject(K); clReleaseMemObject(V);
      Q = Qt; K = Kt; V = Vt;
    } else {
      NNOPT_ERROR("op_Idefics3VisionAttention: bthd_to_bhtd transpose unavailable — output will be scrambled");
    }
  }

  // R8: Flash Attention regressed (~6× slower per call than trio) on Adreno
  // 620's single CU — 16-21 KB __local/WG limits concurrency; 12,288 WGs ⇒
  // serialization dominates fused-kernel ALU savings. Kept the kernel in
  // source for future architectures/tuning, but route through the proven
  // scores → softmax → mha_out trio.
  const bool use_flash = false;

  // scale = head_dim^-0.5 (used by both paths)
  const float scale = 1.0f / sqrt((float)head_dim);

  cl_mem scores = nullptr;
  cl_mem out_heads = nullptr;
  if (use_flash) {
    out_heads = clCreateBuffer(cl_ctx.context(), CL_MEM_READ_WRITE,
                               qkv_elems * sizeof(nnopt_storage_t), nullptr, &err);
    if (!out_heads) {
      NNOPT_ERROR("op_Idefics3VisionAttention: alloc out_heads (flash) failed");
      clReleaseMemObject(Q); clReleaseMemObject(K); clReleaseMemObject(V);
      return nullptr;
    }
    NNOPT_PROFILE_BEGIN(queue, "vis_flash");
    cl_kernel kf = state().k_flash;
    if (!set_arg_checked(kf, 0, sizeof(cl_mem), &Q, "Q") ||
        !set_arg_checked(kf, 1, sizeof(cl_mem), &K, "K") ||
        !set_arg_checked(kf, 2, sizeof(cl_mem), &V, "V") ||
        !set_arg_checked(kf, 3, sizeof(cl_mem), &out_heads, "out") ||
        !set_arg_checked(kf, 4, sizeof(int), &B, "B") ||
        !set_arg_checked(kf, 5, sizeof(int), &num_heads, "H") ||
        !set_arg_checked(kf, 6, sizeof(int), &T, "T") ||
        !set_arg_checked(kf, 7, sizeof(int), &head_dim, "D") ||
        !set_arg_checked(kf, 8, sizeof(float), &scale, "scale")) {
      NNOPT_PROFILE_END(queue, "vis_flash");
      clReleaseMemObject(out_heads);
      clReleaseMemObject(Q); clReleaseMemObject(K); clReleaseMemObject(V);
      return nullptr;
    }
    const size_t lws[3] = {64, 1, 1};
    const size_t gws[3] = {(size_t)B * 64, (size_t)num_heads, (size_t)T};
    err = clEnqueueNDRangeKernel(queue, kf, 3, nullptr, gws, lws, 0, nullptr, nullptr);
    NNOPT_PROFILE_END(queue, "vis_flash");
    if (err != CL_SUCCESS) {
      NNOPT_ERROR_FMT("op_Idefics3VisionAttention: mha_flash_attn dispatch failed (%d)", (int)err);
      clReleaseMemObject(out_heads);
      clReleaseMemObject(Q); clReleaseMemObject(K); clReleaseMemObject(V);
      return nullptr;
    }
  } else {
    // ── Fallback path: separate scores → softmax → mha_out kernels. ───────
    const size_t scores_elems = (size_t)B * (size_t)num_heads * (size_t)T * (size_t)T;
    scores = clCreateBuffer(cl_ctx.context(), CL_MEM_READ_WRITE, scores_elems * sizeof(nnopt_storage_t), nullptr, &err);
    if (!scores) {
      NNOPT_ERROR("op_Idefics3VisionAttention: alloc scores failed");
      clReleaseMemObject(Q); clReleaseMemObject(K); clReleaseMemObject(V);
      return nullptr;
    }

  // 1) scores — DIAGNOSTIC bisect: disable parallel scores
  NNOPT_PROFILE_BEGIN(queue, "vis_scores");
  const bool use_coal_scores = false;
  const bool use_par_scores = false;
  cl_kernel ks = use_coal_scores ? state().k_scores_coal
                                 : (use_par_scores ? state().k_scores_par : state().k_scores);
  if (!set_arg_checked(ks, 0, sizeof(cl_mem), &Q, "Q")) return nullptr;
  if (!set_arg_checked(ks, 1, sizeof(cl_mem), &K, "K")) return nullptr;
  if (!set_arg_checked(ks, 2, sizeof(cl_mem), &scores, "scores")) return nullptr;
  if (!set_arg_checked(ks, 3, sizeof(int), &B, "B")) return nullptr;
  if (!set_arg_checked(ks, 4, sizeof(int), &num_heads, "H")) return nullptr;
  if (!set_arg_checked(ks, 5, sizeof(int), &T, "T")) return nullptr;
  if (!set_arg_checked(ks, 6, sizeof(int), &head_dim, "D")) return nullptr;
  if (!set_arg_checked(ks, 7, sizeof(float), &scale, "scale")) return nullptr;
  if (use_coal_scores || use_par_scores) {
    // gws (B*64, H, T), lws (64,1,1) → WG per (b,h,tq), 64-lane WG.
    const size_t lws[3] = {64, 1, 1};
    const size_t gws[3] = {(size_t)B * 64, (size_t)num_heads, (size_t)T};
    err = clEnqueueNDRangeKernel(queue, ks, 3, nullptr, gws, lws, 0, nullptr, nullptr);
  } else {
    const int TT = T * T;
    const size_t gws_scores[3] = {(size_t)B, (size_t)num_heads, (size_t)TT};
    err = clEnqueueNDRangeKernel(queue, ks, 3, nullptr, gws_scores, nullptr, 0, nullptr, nullptr);
  }
  if (err != CL_SUCCESS) {
    NNOPT_PROFILE_END(queue, "vis_scores");
    NNOPT_ERROR_FMT("op_Idefics3VisionAttention: mha_scores dispatch failed (%d)", (int)err);
    clReleaseMemObject(scores);
    clReleaseMemObject(Q); clReleaseMemObject(K); clReleaseMemObject(V);
    return nullptr;
  }
  NNOPT_PROFILE_END(queue, "vis_scores");

  // 2) optional mask add
  if (attention_mask_opt) {
    cl_kernel km = state().k_add_mask;
    if (!set_arg_checked(km, 0, sizeof(cl_mem), &scores, "scores")) return nullptr;
    if (!set_arg_checked(km, 1, sizeof(cl_mem), &attention_mask_opt, "mask")) return nullptr;
    if (!set_arg_checked(km, 2, sizeof(int), &B, "B")) return nullptr;
    if (!set_arg_checked(km, 3, sizeof(int), &num_heads, "H")) return nullptr;
    if (!set_arg_checked(km, 4, sizeof(int), &T, "T")) return nullptr;
    const int TT = T * T;
    const size_t gws_mask[3] = {(size_t)B, (size_t)num_heads, (size_t)TT};
    cl_int e2 = clEnqueueNDRangeKernel(queue, km, 3, nullptr, gws_mask, nullptr, 0, nullptr, nullptr);
    if (e2 != CL_SUCCESS) {
      NNOPT_ERROR_FMT("op_Idefics3VisionAttention: mha_add_mask dispatch failed (%d)", (int)e2);
      clReleaseMemObject(scores);
      clReleaseMemObject(Q); clReleaseMemObject(K); clReleaseMemObject(V);
      return nullptr;
    }
  }

  // 3) softmax in-place — DIAGNOSTIC bisect: disable parallel softmax
  NNOPT_PROFILE_BEGIN(queue, "vis_softmax");
  const bool use_par_sm = false;
  cl_kernel ksm = use_par_sm ? state().k_softmax_par : state().k_softmax;
  if (!set_arg_checked(ksm, 0, sizeof(cl_mem), &scores, "scores")) return nullptr;
  if (!set_arg_checked(ksm, 1, sizeof(int), &B, "B")) return nullptr;
  if (!set_arg_checked(ksm, 2, sizeof(int), &num_heads, "H")) return nullptr;
  if (!set_arg_checked(ksm, 3, sizeof(int), &T, "T")) return nullptr;
  const size_t gws_sm[3] = {(size_t)B, (size_t)num_heads, (size_t)T};
  if (use_par_sm) {
    const size_t lws_sm[3] = {64, 1, 1};
    const size_t gws_sm_par[3] = {(size_t)B * 64, (size_t)num_heads, (size_t)T};
    err = clEnqueueNDRangeKernel(queue, ksm, 3, nullptr, gws_sm_par, lws_sm, 0, nullptr, nullptr);
  } else {
    err = clEnqueueNDRangeKernel(queue, ksm, 3, nullptr, gws_sm, nullptr, 0, nullptr, nullptr);
  }
  if (err != CL_SUCCESS) {
    NNOPT_PROFILE_END(queue, "vis_softmax");
    NNOPT_ERROR_FMT("op_Idefics3VisionAttention: mha_softmax dispatch failed (%d)", (int)err);
    clReleaseMemObject(scores);
    clReleaseMemObject(Q); clReleaseMemObject(K); clReleaseMemObject(V);
    return nullptr;
  }
  NNOPT_PROFILE_END(queue, "vis_softmax");

  // 4) out heads: [B,H,T,D] (fallback path only; flash path allocated above)
  {
    const size_t out_heads_elems = qkv_elems;
    out_heads = clCreateBuffer(cl_ctx.context(), CL_MEM_READ_WRITE, out_heads_elems * sizeof(nnopt_storage_t), nullptr, &err);
    if (!out_heads) {
      NNOPT_ERROR("op_Idefics3VisionAttention: alloc out_heads failed");
      clReleaseMemObject(scores);
      clReleaseMemObject(Q); clReleaseMemObject(K); clReleaseMemObject(V);
      return nullptr;
    }
  }

  NNOPT_PROFILE_BEGIN(queue, "vis_mha_out");
  // Naive mha_out (256-WI WGs) outperforms both coalesced and parallel WG
  // variants on Adreno 620. Many tiny WIs let the scheduler hide memory
  // latency better than a small number of 64-lane WGs.
  const bool use_coal_out = false;
  const bool use_par_out = false;
  cl_kernel ko = use_coal_out ? state().k_out_coal
                              : (use_par_out ? state().k_out_par : state().k_out);
  if (!set_arg_checked(ko, 0, sizeof(cl_mem), &scores, "probs")) return nullptr;
  if (!set_arg_checked(ko, 1, sizeof(cl_mem), &V, "V")) return nullptr;
  if (!set_arg_checked(ko, 2, sizeof(cl_mem), &out_heads, "out")) return nullptr;
  if (!set_arg_checked(ko, 3, sizeof(int), &B, "B")) return nullptr;
  if (!set_arg_checked(ko, 4, sizeof(int), &num_heads, "H")) return nullptr;
  if (!set_arg_checked(ko, 5, sizeof(int), &T, "T")) return nullptr;
  if (!set_arg_checked(ko, 6, sizeof(int), &head_dim, "D")) return nullptr;
  const int D4 = head_dim >> 2;
  const int TD4 = T * D4;
  if (use_coal_out || use_par_out) {
    // gws (B*64, H, T), lws (64,1,1) → WG per (b,h,tq), 64-lane WG.
    const size_t lws[3] = {64, 1, 1};
    const size_t gws[3] = {(size_t)B * 64, (size_t)num_heads, (size_t)T};
    err = clEnqueueNDRangeKernel(queue, ko, 3, nullptr, gws, lws, 0, nullptr, nullptr);
  } else {
    const size_t gws_out[3] = {(size_t)B, (size_t)num_heads, (size_t)TD4};
    err = clEnqueueNDRangeKernel(queue, ko, 3, nullptr, gws_out, nullptr, 0, nullptr, nullptr);
  }
  if (err != CL_SUCCESS) {
    NNOPT_PROFILE_END(queue, "vis_mha_out");
    NNOPT_ERROR_FMT("op_Idefics3VisionAttention: mha_out dispatch failed (%d)", (int)err);
    clReleaseMemObject(out_heads);
    clReleaseMemObject(scores);
    clReleaseMemObject(Q); clReleaseMemObject(K); clReleaseMemObject(V);
    return nullptr;
  }
  NNOPT_PROFILE_END(queue, "vis_mha_out");
  }  // end of fallback (!use_flash) path

  // out_heads is [B,H,T,D]; we need ctx [B,T,C] contiguous for out_proj.
  // We'll transpose+reshape by a kernel: out_ctx[b,t,h,d] -> out_ctx_flat[(b*T+t), (h*D+d)].
  // PROGRAM-INIT-OK: build small embedded kernel once.
  static cl_program tr_prog = nullptr;
  static cl_kernel tr_k = nullptr;
  static bool tr_init = false;
  if (!tr_init) {
    const char* tr_src = R"CLC(
#ifdef USE_FP16
  #pragma OPENCL EXTENSION cl_khr_fp16 : enable
  typedef half storage_t;
  #define LOAD(p, i)     vload_half((i), (p))
  #define STORE(p, i, v) vstore_half((v), (i), (p))
#else
  typedef float storage_t;
  #define LOAD(p, i)     ((p)[(i)])
  #define STORE(p, i, v) ((p)[(i)] = (v))
#endif

// in:  [B,H,T,D] contiguous with D fastest
// out: [B,T,H*D] contiguous
__kernel void bhwd_to_bthd(
    __global const storage_t* in,
    __global storage_t* out,
    const int B,
    const int H,
    const int T,
    const int D) {
  const int b = (int)get_global_id(0);
  const int t = (int)get_global_id(1);
  const int hd = (int)get_global_id(2);
  if (b >= B || t >= T || hd >= H*D) return;
  const int h = hd / D;
  const int d = hd - h * D;
  const long in_idx = (((long)b * (long)H + (long)h) * (long)T + (long)t) * (long)D + (long)d;
  const long out_idx = (((long)b * (long)T + (long)t) * (long)(H*D)) + (long)hd;
  const float v = (float)LOAD(in, in_idx);
  STORE(out, out_idx, (storage_t)v);
}
)CLC";
    tr_prog = cl_ctx.build_program(std::string(tr_src));  // PROGRAM-INIT-OK
    if (!tr_prog) {
      NNOPT_ERROR("op_Idefics3VisionAttention: build transpose kernel failed");
      // fall through; will error below
    } else {
      cl_int te = CL_SUCCESS;
      tr_k = clCreateKernel(tr_prog, "bhwd_to_bthd", &te);
      if (te != CL_SUCCESS || !tr_k) {
        NNOPT_ERROR_FMT("op_Idefics3VisionAttention: clCreateKernel(bhwd_to_bthd) failed (%d)", (int)te);
      } else {
        tr_init = true;
      }
    }
  }
  if (!tr_init) {
    clReleaseMemObject(out_heads);
    clReleaseMemObject(scores);
    clReleaseMemObject(Q); clReleaseMemObject(K); clReleaseMemObject(V);
    return nullptr;
  }

  cl_mem ctx = clCreateBuffer(cl_ctx.context(), CL_MEM_READ_WRITE, qkv_elems * sizeof(nnopt_storage_t), nullptr, &err);
  if (!ctx) {
    NNOPT_ERROR("op_Idefics3VisionAttention: alloc ctx failed");
    clReleaseMemObject(out_heads);
    clReleaseMemObject(scores);
    clReleaseMemObject(Q); clReleaseMemObject(K); clReleaseMemObject(V);
    return nullptr;
  }

  if (!set_arg_checked(tr_k, 0, sizeof(cl_mem), &out_heads, "in")) return nullptr;
  if (!set_arg_checked(tr_k, 1, sizeof(cl_mem), &ctx, "out")) return nullptr;
  if (!set_arg_checked(tr_k, 2, sizeof(int), &B, "B")) return nullptr;
  if (!set_arg_checked(tr_k, 3, sizeof(int), &num_heads, "H")) return nullptr;
  if (!set_arg_checked(tr_k, 4, sizeof(int), &T, "T")) return nullptr;
  if (!set_arg_checked(tr_k, 5, sizeof(int), &head_dim, "D")) return nullptr;
  const size_t gws_tr[3] = {(size_t)B, (size_t)T, (size_t)C};
  err = clEnqueueNDRangeKernel(queue, tr_k, 3, nullptr, gws_tr, nullptr, 0, nullptr, nullptr);
  if (err != CL_SUCCESS) {
    NNOPT_ERROR_FMT("op_Idefics3VisionAttention: transpose dispatch failed (%d)", (int)err);
    clReleaseMemObject(ctx);
    clReleaseMemObject(out_heads);
    clReleaseMemObject(scores);
    clReleaseMemObject(Q); clReleaseMemObject(K); clReleaseMemObject(V);
    return nullptr;
  }

  // out_proj: [rows,C] -> [rows,C]
  cl_mem out = clCreateBuffer(cl_ctx.context(), CL_MEM_READ_WRITE, qkv_elems * sizeof(nnopt_storage_t), nullptr, &err);
  if (!out) {
    NNOPT_ERROR("op_Idefics3VisionAttention: alloc out failed");
    clReleaseMemObject(ctx);
    clReleaseMemObject(out_heads);
    clReleaseMemObject(scores);
    clReleaseMemObject(Q); clReleaseMemObject(K); clReleaseMemObject(V);
    return nullptr;
  }

  NNOPT_PROFILE_BEGIN(queue, "vis_oproj");
  if (!pytorch_linear(queue, rows, C, C, ctx, Wo, out)) {
    NNOPT_PROFILE_END(queue, "vis_oproj");
    NNOPT_ERROR("op_Idefics3VisionAttention: out_proj GEMM failed");
    clReleaseMemObject(out);
    clReleaseMemObject(ctx);
    clReleaseMemObject(out_heads);
    clReleaseMemObject(scores);
    clReleaseMemObject(Q); clReleaseMemObject(K); clReleaseMemObject(V);
    return nullptr;
  }
  NNOPT_PROFILE_END(queue, "vis_oproj");
  if (Bo) {
    // add bias to out
    static cl_program bias_prog4 = nullptr;
    static cl_kernel bias_k4 = nullptr;
    static bool bias_init4 = false;
    if (!bias_init4) {
      bias_prog4 = cl_ctx.build_program_from_file("kernels/bias_add.cl");  // PROGRAM-INIT-OK
      cl_int e2 = CL_SUCCESS;
      bias_k4 = clCreateKernel(bias_prog4, "bias_add_2d", &e2);
      if (e2 != CL_SUCCESS || !bias_k4) {
        NNOPT_ERROR_FMT("op_Idefics3VisionAttention: clCreateKernel(bias_add_2d) failed (%d)", (int)e2);
      } else {
        bias_init4 = true;
      }
    }
    if (bias_init4) {
      if (!set_arg_checked(bias_k4, 0, sizeof(cl_mem), &out, "x")) return nullptr;
      if (!set_arg_checked(bias_k4, 1, sizeof(cl_mem), &Bo, "bias")) return nullptr;
      if (!set_arg_checked(bias_k4, 2, sizeof(int), &rows, "rows")) return nullptr;
      if (!set_arg_checked(bias_k4, 3, sizeof(int), &C, "cols")) return nullptr;
      const size_t gwsb[2] = {(size_t)rows, (size_t)C};
      cl_int e3 = clEnqueueNDRangeKernel(queue, bias_k4, 2, nullptr, gwsb, nullptr, 0, nullptr, nullptr);
      if (e3 != CL_SUCCESS) {
        NNOPT_ERROR_FMT("op_Idefics3VisionAttention: bias_add out failed (%d)", (int)e3);
      }
    }
  }

  // Cleanup intermediates
  clReleaseMemObject(ctx);
  clReleaseMemObject(out_heads);
  clReleaseMemObject(scores);
  clReleaseMemObject(Q);
  clReleaseMemObject(K);
  clReleaseMemObject(V);

  return out;
}
