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
#include <type_traits>
#ifdef NNOPT_USE_FP16
  // fp16 storage is RAW IEEE-754 binary16 BITS (cl_half == uint16_t). The #1
  // recurring port-killer was host code doing `(float)host_storage[i]` — a bare
  // cast reinterprets the BIT PATTERN as an integer (0..65535), NOT the half
  // value, so e.g. fp16(4.427) reads back as 17517 and argmax picks the most
  // negative token. To make that bug a COMPILE ERROR instead of silent garbage,
  // nnopt_storage_t is a 2-byte wrapper struct with NO numeric/float conversion:
  //   • `(float)x`  / `static_cast<float>(x)` / `x * 2.0f`  → won't compile.
  //   • the ONLY way to a float is `nnopt_f16_to_f32(static_cast<uint16_t>(x))`.
  // Encoding stays ergonomic: `storage = nnopt_f32_to_f16(f)` works because the
  // converting ctor from uint16_t is implicit. Layout is bit-identical to
  // cl_half (2 bytes, trivially copyable) so clEnqueueRead/WriteBuffer, memcpy,
  // sizeof, and std::vector<nnopt_storage_t> all behave exactly as before.
  struct nnopt_half_t {
      uint16_t bits;
      nnopt_half_t() = default;
      constexpr nnopt_half_t(uint16_t b) noexcept : bits(b) {}   // encode: storage = nnopt_f32_to_f16(f)
      constexpr explicit operator uint16_t() const noexcept { return bits; }  // decode: nnopt_f16_to_f32(static_cast<uint16_t>(x))
      // Intentionally NO `operator float`, NO arithmetic operators.
  };
  static_assert(sizeof(nnopt_half_t) == 2, "nnopt_half_t must be 2 bytes for GPU buffer layout");
  static_assert(std::is_trivially_copyable<nnopt_half_t>::value, "nnopt_half_t must be trivially copyable for buffer reads");
  using nnopt_storage_t = nnopt_half_t;
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

// Legacy alias retained so existing scaffold internals continue to compile
// during the dtype-template rollout. Prefer nnopt_compute_t in new code.
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
// error. Same convention applies to every scaffold/agent kernel + wrapper.
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
// from GPT-2) call clblast::Gemm<cl_half>(...) under fp16 / clblast::Gemm<float>
// under fp32 directly (NOT Gemm<nnopt_storage_t> — under fp16 nnopt_storage_t is
// the nnopt_half_t wrapper struct, which CLBlast cannot instantiate; use the raw
// cl_half element type) and document why at the call site. Raw clblast::Gemm<float> in layer code is mechanically
// refused by Build (cppStandards Rule 01a).
//
// M = batch*seq rows in x / out
// N = output feature dim (W.shape[0])
// K = input feature dim  (W.shape[1])
//
// Prints NNOPT_ERROR_FMT and returns false on CLBlast failure. On success,
// the OpenCL in-order queue ensures 'out' is observable to the next kernel
// enqueued on the same queue without an explicit sync (Rule SYNC-01).
// ──────────────────────────────────────────────
// Per-token buffer pool — see utils.cpp for rationale. Drop-in replacements for
// clCreateBuffer(ctx, CL_MEM_READ_WRITE, bytes, nullptr, &err) and
// clReleaseMemObject(mem) on PER-TOKEN scratch buffers. pool_free recycles pool
// buffers and real-releases non-pool ones, so it is safe on any cl_mem.
// Do NOT pool persistent buffers (weights, KV cache, cached transposed/concat
// weights) — only the intermediate scratch the op graph re-creates each token.
cl_mem nnopt_pool_alloc(cl_context ctx, size_t bytes, cl_int* err_out);
void   nnopt_pool_free(cl_mem mem);

bool pytorch_linear(cl_command_queue queue,
                    int M, int N, int K,
                    cl_mem x, cl_mem W, cl_mem out);

// ──────────────────────────────────────────────
// pytorch_linear_fused3 / _fused2: fuse 2-3 nn.Linear projections that share the
// same input x into ONE GEMV dispatch over a concatenated weight. q/k/v and
// gate/up qualify. Decode is dispatch-bound on the small projections (~480µs GPU
// launch overhead each); fusing removes those launches. Outputs route to their
// own buffers (out0/out1/out2), so call sites downstream are unchanged. The
// concatenated weight is built once on first use and cached (keyed by W0).
// fp32 build falls back to separate pytorch_linear calls. Same [N,K] (nn.Linear)
// weight layout and accumulation as pytorch_linear → bit-identical results.
bool pytorch_linear_fused3(cl_command_queue queue, int M,
                           int N0, int N1, int N2, int K,
                           cl_mem x, cl_mem W0, cl_mem W1, cl_mem W2,
                           cl_mem out0, cl_mem out1, cl_mem out2);
bool pytorch_linear_fused2(cl_command_queue queue, int M,
                           int N0, int N1, int K,
                           cl_mem x, cl_mem W0, cl_mem W1,
                           cl_mem out0, cl_mem out1);

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
// Layer-contract signal: when .nnport/layer_contracts/<Class>.json says
// weight_key_parent_classes.<field> == "Conv1D", call this. When it says
// "Linear", call pytorch_linear. Build mechanically refuses pytorch_linear()
// on Conv1D-stored weights (cppStandards Rule 01a / Build gate).
//
// M = batch*seq rows in x / out
// N = output feature dim (W.shape[1] under Conv1D layout)
// K = input feature dim  (W.shape[0] under Conv1D layout)
bool pytorch_conv1d(cl_command_queue queue,
                    int M, int N, int K,
                    cl_mem x, cl_mem W, cl_mem out);
