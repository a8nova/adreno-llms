#pragma once
#include <vector>
#include <string>
#include <cstdint>
#include <cmath>
#include <chrono>

// ──────────────────────────────────────────────
// Dtype-template typedefs (fp32 default, fp16 opt-in via -DNNOPT_USE_FP16=1).
// Scaffold-emitted; layer code uses these names instead of bare \`float\`.
// ──────────────────────────────────────────────
#include <CL/cl.h>
#ifdef NNOPT_USE_FP16
  // cl_half is uint16_t; the IEEE 754 binary16 codec below converts to/from float on the host.
  using nnopt_storage_t = cl_half;
  #define NNOPT_DTYPE_STR "float16"
#else
  using nnopt_storage_t = float;
  #define NNOPT_DTYPE_STR "float32"
#endif

// Host-side accumulator type — always float, regardless of storage dtype.
// fp16 storage with fp32 compute is the standard pattern. Layer code that
// reads weights to inspect on CPU should use \`get_host_vec()\` (returns
// std::vector<float>), which decodes fp16→float transparently.
using nnopt_compute_t = float;

// Legacy alias retained for backward compatibility with older code paths.
// Prefer nnopt_compute_t in new code.
using compute_t = float;
#define COMPUTE_DTYPE NNOPT_DTYPE_STR

// IEEE 754 binary16 codec (subnormals / Inf / NaN / saturating overflow).
// Bit-exact, branch-light, no host-half intrinsic dependence.
float    nnopt_f16_to_f32(uint16_t bits);
uint16_t nnopt_f32_to_f16(float v);

// ──────────────────────────────────────────────
// set_arg_checked — public clSetKernelArg wrapper with descriptive error.
// ──────────────────────────────────────────────
// Use this in EVERY op (src/ops/*.cpp) instead of bare clSetKernelArg.
// Returns false on failure after logging via NNOPT_ERROR_FMT. Pairs with
// the cleanup-lambda idiom in cpp-standards.md — the standard call site
// is:
//   if (!set_arg_checked(kernel, idx, sizeof(cl_mem), &buf, "buf")) return cleanup();
// Model-agnostic; no PyTorch semantics. Defined in utils.cpp so every op
// file links the same symbol (no per-file static copy required).
bool set_arg_checked(cl_kernel kernel,
                     unsigned int arg_index,
                     size_t arg_size,
                     const void* arg_value,
                     const char* arg_name);

// Timing utility
class Timer {
public:
    void start() { start_ = std::chrono::high_resolution_clock::now(); }
    double elapsed_ms() const {
        auto now = std::chrono::high_resolution_clock::now();
        return std::chrono::duration<double, std::milli>(now - start_).count();
    }
private:
    std::chrono::high_resolution_clock::time_point start_;
};

// Numerical comparison
float compute_mse(const float* a, const float* b, size_t n);
float compute_max_diff(const float* a, const float* b, size_t n);

// File I/O
std::vector<float> load_npy_float32(const std::string& path);
void save_npy_float32(const std::string& path, const float* data, const std::vector<size_t>& shape);

// Element-wise addition: out[i] = a[i] + b[i]
// Dispatches the element_add kernel from kernels/utils.cl.
// Returns a NEW buffer with the result (caller owns it).
#include <CL/cl.h>
cl_mem element_add(cl_command_queue queue, cl_program utils_program, cl_mem a, cl_mem b, size_t n);

// In-place element-wise addition: a[i] += b[i].
// Caller retains ownership of a; b is read-only. Saves the buffer-alloc
// + clEnqueueCopyBuffer pair that element_add() does on every residual.
// The kernel object is cached per-program in a function-local static so
// repeated calls don't pay clCreateKernel cost (Rule PROG-01).
//
// Use this for residual adds at decode (forward_decode_into_residual,
// Model::forward seq_len==1 branch). Pairs with the FUSE-DECODE-01
// fused kernels to keep the M=1 hot path allocation-free.
bool element_add_inplace(cl_command_queue queue, cl_program utils_program,
                         cl_mem a, cl_mem b, size_t n);

// Per-row split of a row-major [rows, 2*half_cols] buffer into two contiguous
// [rows, half_cols] buffers. Use for chunk(2, dim=-1) on GLU-style activations.
// CRITICAL: contiguous byte-offset sub-buffers (clCreateSubBuffer with origin
// 0 and origin=rows*half_cols*4) do NOT produce chunk(2, dim=-1) semantics —
// they split by ROWS, not by last dim. Always use this helper (or two sliced
// GEMMs).
//
// Caller owns 'first' and 'second' output buffers — allocate them as
// rows*half_cols*sizeof(nnopt_storage_t) CL_MEM_READ_WRITE.
//
// NOTE: parameter is half_cols, NOT half. The bare identifier 'half' is
// reserved by OpenCL's cl_khr_fp16 as the fp16 type token; using it as a
// variable name fails on Adreno's clang front-end with an opaque parse
// error. Same convention applies to every kernel + wrapper in this codebase.
bool split_last_dim_2(cl_command_queue queue, cl_program utils_program,
                      cl_mem src, cl_mem first, cl_mem second,
                      int rows, int half_cols);

// ──────────────────────────────────────────────
// pytorch_linear: dtype-templated GEMM wrapper for PyTorch nn.Linear layout
// ──────────────────────────────────────────────
// Computes: out[M, N] = x[M, K] @ W[N, K]^T
// where W is a PyTorch nn.Linear weight stored as [out_features, in_features].
//
// Hides CLBlast's counter-intuitive RowMajor+Transpose::kYes+ldb=K invocation
// so layer code never has to guess leading dimensions. Internally dispatches
// to clblast::Gemm<float> under fp32 and clblast::Gemm<cl_half> under fp16
// (with portable nnopt_f32_to_f16 alpha/beta). Use this for EVERY nn.Linear
// projection (q_proj, k_proj, v_proj, out_proj, gate_proj, up_proj,
// down_proj, lm_head, etc.). For non-nn.Linear weights (e.g. Conv1D [in,out]
// from GPT-2) call clblast::Gemm<nnopt_storage_t>(...) directly and document
// why at the call site. Raw clblast::Gemm<float> in layer code is mechanically
// refused by Build (cppStandards Rule 01a).
//
// M = batch*seq rows in x / out
// N = output feature dim (W.shape[0])
// K = input feature dim  (W.shape[1])
//
// Prints NNOPT_ERROR_FMT and returns false on CLBlast failure. On success,
// the OpenCL in-order queue ensures 'out' is observable to the next kernel
// enqueued on the same queue without an explicit sync (Rule SYNC-01).
bool pytorch_linear(cl_command_queue queue,
                    int M, int N, int K,
                    cl_mem x, cl_mem W, cl_mem out);

// GPU argmax for fp16 logits (size N). Writes one int32 to out_idx[0].
// Greedy-sampler fast path: avoids ~200 KB D2H per decode step.
bool argmax_fp16_dispatch(cl_command_queue queue, int N, cl_mem logits, cl_mem out_idx);

// Fused decode-only attention (seq_q=1 fp16 only).
// Online-softmax over seq_k, one workgroup per query head; writes the full
// attention output buffer in a single dispatch.
bool decode_attn_fp16_dispatch(cl_command_queue queue,
                                int num_q_heads, int num_kv_heads, int head_dim, int seq_k,
                                float scale,
                                cl_mem Q, cl_mem K_cache, cl_mem V_cache, cl_mem out);

// Fused Q/K/V + RoPE + KV-cache write (M=1 fp16 only). Q rows are rotated and
// written to `Y_q`; K rows are rotated and written directly into `K_cache` at
// the `start_pos` slot; V rows are written un-rotated into `V_cache`. Saves
// the separate RoPE dispatch and the `clEnqueueCopyBuffer` into the cache.
bool gemv_m1_qkv_fused_fp16_dispatch(cl_command_queue queue,
                                      int num_q_heads, int num_kv_heads, int head_dim, int K_hidden,
                                      int start_pos,
                                      cl_mem x,
                                      cl_mem W_q, cl_mem W_k, cl_mem W_v,
                                      cl_mem Y_q, cl_mem K_cache, cl_mem V_cache,
                                      cl_mem rope_cos, cl_mem rope_sin);

// Fused Q/K/V projection (M=1 fp16 only). Reads `x[K]` once and dispatches
// across (N_q + 2*N_kv) output rows in a single kernel.
bool gemv_m1_qkv_fp16_dispatch(cl_command_queue queue,
                                int N_q, int N_kv, int K,
                                cl_mem x,
                                cl_mem W_q, cl_mem W_k, cl_mem W_v,
                                cl_mem Y_q, cl_mem Y_k, cl_mem Y_v);

// Fused gate_proj + up_proj + SwiGLU (M=1 fp16 only). Computes
//   out[n] = silu((W_gate · x)[n]) * (W_up · x)[n]   for n in [0, N).
// Returns false if the build isn't fp16 or kernel init fails.
bool gemv_m1_swiglu_fp16_dispatch(cl_command_queue queue,
                                  int N, int K,
                                  cl_mem x, cl_mem W_gate, cl_mem W_up, cl_mem out);

// Fused GEMV + element-wise residual add (M=1 fp16 only). Computes
//   out[row] = (W · x)[row] + residual[row]   for row in [0, N).
// Used for down_proj on decode path so we can skip the separate
// add_residual_inplace dispatch (saves 1 dispatch per layer per token).
bool gemv_m1_residual_fp16_dispatch(cl_command_queue queue,
                                    int N, int K,
                                    cl_mem x, cl_mem W, cl_mem residual, cl_mem out);

// ──────────────────────────────────────────────
// Image2D-backed GEMV. Routes weight reads through Adreno's L1 texture cache
// (which only services image objects — buffer reads bypass it). Per Qualcomm
// guide §6.2 / §7.1.5 — biggest win when the weight matrix exceeds L2 (64 KB
// on Adreno 620v2).
// ──────────────────────────────────────────────

// Get-or-create a CL_RGBA+CL_HALF_FLOAT image2d view of an existing fp16 weight
// buffer. Width=K/4 (4 fp16 per RGBA pixel), height=N. Cached by source buffer
// — second call with the same `weight_buf` returns the existing image. Returns
// nullptr if K%4 != 0, N exceeds CL_DEVICE_IMAGE2D_MAX_HEIGHT, or the device
// rejects CL_RGBA+CL_HALF_FLOAT.
cl_mem get_or_create_weight_image(cl_context ctx, cl_command_queue queue,
                                  cl_mem weight_buf, int N, int K);

// R6.2: image2d_array variant for weights with N > CL_DEVICE_IMAGE2D_MAX_HEIGHT
// (lm_head: 49280 rows). slice_h = max_image_height; n_slices = ceil(N/slice_h).
// Returns {nullptr, 0, 0} on failure (K%4!=0, OOM, no array_size support).
struct WeightImageArray {
    cl_mem image = nullptr;
    int slice_h = 0;
    int n_slices = 0;
};
WeightImageArray get_or_create_weight_image_array(cl_context ctx,
                                                  cl_command_queue queue,
                                                  cl_mem weight_buf, int N, int K);

// Image-backed GEMV (M=1, fp16). Reads W via read_imageh through the L1
// texture cache. Returns false if the build isn't fp16 or kernel init fails.
bool gemv_m1_image_fp16_dispatch(cl_command_queue queue,
                                 int N, int K,
                                 cl_mem x, cl_mem W_image, cl_mem out);

// R6.2: image2d_array-backed GEMV (M=1, fp16). Same WG layout as
// gemv_m1_image_fp16_dispatch; per-row slice arithmetic in kernel.
bool gemv_m1_image_array_fp16_dispatch(cl_command_queue queue,
                                       int N, int K, int slice_h,
                                       cl_mem x, cl_mem W_image_array, cl_mem out);

// Image-backed fused GEMV + element-wise residual add (down_proj decode path).
bool gemv_m1_image_residual_fp16_dispatch(cl_command_queue queue,
                                          int N, int K,
                                          cl_mem x, cl_mem W_image,
                                          cl_mem residual, cl_mem out);

// Image-backed fused gate+up+SwiGLU GEMV (M=1 decode path; two image weights).
bool gemv_m1_image_swiglu_fp16_dispatch(cl_command_queue queue,
                                        int N, int K,
                                        cl_mem x,
                                        cl_mem W_gate_image, cl_mem W_up_image,
                                        cl_mem out);

// R6.4: fused rmsnorm + QKV image GEMV (M=1 decode). Skips the standalone
// rmsnorm dispatch by computing inv_rms inside each WG. eps from model_config.
bool gemv_m1_rmsnorm_qkv_image_fp16_dispatch(cl_command_queue queue,
                                             int N_q, int N_kv, int K, float eps,
                                             cl_mem x, cl_mem gamma,
                                             cl_mem W_q_image, cl_mem W_k_image, cl_mem W_v_image,
                                             cl_mem Y_q, cl_mem Y_k, cl_mem Y_v);

// R6.5: fused rmsnorm + residual-add + swiglu image GEMV (M=1 decode). Reads
// raw (attn_out, hidden_states); WG 0 writes their sum to sum_buf (residual2
// for the downstream fused down+residual). Eliminates the rmsnorm_post dispatch.
bool gemv_m1_rmsnorm_residual_image_swiglu_fp16_dispatch(
    cl_command_queue queue,
    int N, int K, float eps,
    cl_mem attn_out, cl_mem hidden_states, cl_mem gamma,
    cl_mem W_gate_image, cl_mem W_up_image,
    cl_mem y, cl_mem sum_buf);

// Image-backed fused QKV projection GEMV (M=1 decode path; three image weights).
bool gemv_m1_qkv_image_fp16_dispatch(cl_command_queue queue,
                                     int N_q, int N_kv, int K,
                                     cl_mem x,
                                     cl_mem W_q_image, cl_mem W_k_image, cl_mem W_v_image,
                                     cl_mem Y_q, cl_mem Y_k, cl_mem Y_v);

// TILE_N=2 variant — each workgroup handles 2 output rows. Halves WG count
// for high-N shapes (gate/up at N=1536) where per-WG dispatch overhead
// dominates the math.
bool gemv_m1_swiglu_tile2_fp16_dispatch(cl_command_queue queue,
                                         int N, int K,
                                         cl_mem x, cl_mem W_gate, cl_mem W_up, cl_mem out);

// ──────────────────────────────────────────────
// pytorch_conv1d: dtype-templated GEMM wrapper for HF Conv1D layout
// ──────────────────────────────────────────────
// Computes: out[M, N] = x[M, K] @ W[K, N]
// where W is a HuggingFace Conv1D weight stored as [in_features, out_features]
// — the OPPOSITE of nn.Linear. HF Conv1D forward is y = x @ W + b (NO
// transpose). Used by GPT-1, GPT-2, GPT-Neo, and any HF model whose
// modeling_*.py declares self.<attr> = Conv1D(...).
//
// Same arg ORDER as pytorch_linear, so call sites are interchangeable when
// you flip Conv1D ↔ Linear. The internal CLBlast call is:
//   Gemm(RowMajor, Transpose::kNo, Transpose::kNo, M, N, K,
//        alpha, x, 0, K, W, 0, N, beta, out, 0, N, ...)
// — TransposeB=kNo and ldb=N (out_features), in contrast to pytorch_linear's
// TransposeB=kYes and ldb=K.
//
// Layer dispatch: when the upstream HuggingFace module stores its weights
// under a Conv1D parent class, call this. When it's stored under Linear,
// call pytorch_linear. Calling pytorch_linear() on Conv1D-stored weights
// silently produces transposed garbage — be careful at the call sites.
//
// M = batch*seq rows in x / out
// N = output feature dim (W.shape[1] under Conv1D layout)
// K = input feature dim  (W.shape[0] under Conv1D layout)
bool pytorch_conv1d(cl_command_queue queue,
                    int M, int N, int K,
                    cl_mem x, cl_mem W, cl_mem out);
