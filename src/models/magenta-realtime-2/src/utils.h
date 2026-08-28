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
bool pytorch_linear(cl_command_queue queue,
                    int M, int N, int K,
                    cl_mem x, cl_mem W, cl_mem out);

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

// ── One GEMV entry point for the whole AR stack ──────────────────────────────
// out[rows, out_dim] = x[rows, in_dim] @ W[out_dim, in_dim].T (+ bias).
//
// Picks the weight format from what the bundle actually shipped: if `<w_key>.scale`
// exists the weight is block-32 Q4 (kernels/gemv_q4_f32.cl), otherwise it is fp16
// (kernels/linear_f32_w16.cl — the tuned WG=64/NOUT=8 path). Every AR projection —
// attention QKV, cross-attention Q/KV, out-proj, MLP dense, the depth adapter —
// goes through here, so switching the model's precision is a weights swap and not
// a code change, and the two paths can never drift apart in dispatch shape.
//
// `bias` may be null. Returns a newly-allocated buffer the caller owns.
class OpenCLContext;
class Weights;
// GEMV with the surrounding elementwise ops folded in, so an RMSNorm-then-GEMV-then-GELU chain
// costs ONE dispatch instead of three. rms_prefix/gelu are opt-in; passing neither is the plain GEMV.
// The RMS constant is uniform over the row, so folding it is an algebraic identity, not an
// approximation — see kernels/linear_bias_fused_f32.cl.
// NNOPT_FUSE, read from ONE place. It is a CONTRACT between two files: backbone's mlp_block stops
// materialising rms1 at level >= 2 because MLP_forward promises to fold it into dense1's GEMV. They
// used to read the env separately with their own defaults, so changing one default silently dropped
// the normalisation from every MLP block — audio RMS fell 53x (3837 -> 72) with no error anywhere.
// 0 = nothing fused, 1 = GELU epilogue only, 2 = GELU + RMS pre-norm.
int nnopt_fuse_level();
void   nnopt_set_codec_gpu_ms(double ms);
double nnopt_codec_gpu_ms();

// Force the codec's fp16 GEMM path on (1) or off (0) regardless of NNOPT_CODECFP16, or hand control
// back to the environment (-1). The A/B needs both paths inside ONE render, and the alternative —
// setenv plus a toggle-epoch bump between passes — would also reset every other cached toggle
// mid-render. This switch touches exactly the one decision it names.
void nnopt_codec_fp16_force(int mode);
int  nnopt_codec_fp16_forced();

// Result of a codecab=1 render: cosine similarity and peak absolute error between the fp32 and
// fp16 codec waveforms, over every sample of every chunk. n_chunks == 0 means no A/B ran.
void nnopt_set_codec_ab(double cosine, double max_abs_err, int n_chunks);
void nnopt_get_codec_ab(double* cosine, double* max_abs_err, int* n_chunks);

// How many codec convolutions actually ran in fp16 this render, how many were 1x1 (no column
// buffer, so nothing to convert) and how many fell back. Reported because "ran fp16" silently
// executing fp32 is the failure mode this path keeps hitting — a speed-up with no quality change
// is exactly what a silent fallback looks like.
void nnopt_codec_fp16_tally(int fp16, int skipped, int failed);
void nnopt_codec_fp16_get(int* fp16, int* skipped, int* failed);


// Clear every per-render codec statistic. Called once at the top of run_song, because a value that
// survives into the NEXT request is reported as if it were measured there — which is precisely the
// bug that made codec_gpu read above 100% for weeks.
void nnopt_codec_stats_reset();


// Pin the CALLING THREAD to the performance cores and raise its priority. See utils.cpp — the AR is
// limited by CPU time inside the driver's enqueue, so which core runs it is a first-order question.
void nnopt_pin_perf_cores(const char* who);

// int8 conversion totals, reported once with the render summary rather than one line per tensor.
void nnopt_int8_note(double mb);
void nnopt_int8_summary(char* buf, size_t n);

// Device-resident AR position; see utils.cpp. `constant=true` returns an immutable per-value buffer
// (the depth body's RVQ level, which repeats identically every frame); false returns the shared
// buffer whose contents are rewritten between replays. Writes always go to the live queue.
cl_mem nnopt_pos_buffer(OpenCLContext& cl_ctx, cl_command_queue queue, int pos, bool constant = false);
void   nnopt_set_live_queue(cl_command_queue q);


// ── per-call-site kernel instances ──────────────────────────────────────────────────────────────
// Vend a cl_kernel unique to one dispatch SITE, cloned from `proto` (same program, same entry
// point). A recordable-queue capture references the kernel object rather than snapshotting its
// arguments, so a kernel shared across call sites replays every dispatch with the last args set.
// Args on an instance are set once per frame by exactly one site, which is what makes a capture
// valid. Falls back to `proto` if cloning fails, and is a no-op under NNOPT_KINST=0.
//
// `site_key` should be the weight prefix; the frame dispatch ordinal is appended internally,
// because run_depth_step reuses one prefix 12x per frame (once per RVQ codebook).
// `base` + `suffix` are kept separate and never concatenated on the dispatch path — building one
// key per dispatch would heap-allocate ~455 times a frame on the path this whole change exists to
// make cheaper. Pass the weight prefix as `base`; the frame dispatch ordinal is appended internally,
// because run_depth_step reuses one prefix 12x per frame (once per RVQ codebook).
cl_kernel nnopt_kernel_instance(cl_kernel proto, const std::string& base, const char* suffix = nullptr);

// Call at the top of every AR frame: resets the dispatch ordinal that keys the instances.
void   nnopt_kinst_frame_begin();
// Call when the AR part of the frame is done, BEFORE any codec work. Outside this window the
// ordinal does not advance and callers get the shared kernel object — the codec runs inside the
// AR's frame loop and would otherwise append its dispatches to the frame's sequence.
void   nnopt_kinst_frame_end();
// Called by profEnqueue for every NDRange dispatch, so the engine can prove that every dispatch in
// a recorded frame went through the per-site kernel pool rather than trusting an audit.
void   nnopt_kinst_note_dispatch();
// Map a per-site instance back to the kernel TYPE it was cloned from (identity for anything else).
cl_kernel nnopt_kinst_proto_for(cl_kernel k);
// False once the per-frame dispatch sequence has diverged from the first frame's — a capture taken
// on one frame is then not valid for the next, and recording must not be used.
bool   nnopt_kinst_stable();
void   nnopt_kinst_reset();
// Vend a DISJOINT set of kernel objects (generation != 0). Required for any live pass that runs
// after a capture: it would otherwise rewrite the args the recording depends on.
void   nnopt_kinst_set_generation(int g);
int    nnopt_kinst_generation();
size_t nnopt_kinst_count();

// Veto recording from anywhere that bakes a host-computed value into a kernel argument (such a
// value would freeze at the captured frame). Latches; nnopt_record_safe() also folds in
// nnopt_kinst_stable().
void nnopt_record_mark_unsafe(const char* why);

// Copy via an NDRange dispatch instead of clEnqueueCopyBuffer, which a recording cannot capture.
// Offsets are in ELEMENTS. Every copy on the AR frame path must use this.
bool nnopt_copy_buffer(OpenCLContext& cl_ctx, cl_mem src, cl_mem dst,
                       int src_off, int dst_off, int n,
                       const std::string& site, const char* suffix = nullptr);
bool nnopt_record_safe();

cl_mem nnopt_gemv_fused(OpenCLContext& cl_ctx, Weights& weights, cl_command_queue queue,
                        cl_mem x, const std::string& w_key,
                        int rows, int in_dim, int out_dim, cl_mem bias,
                        const std::string& rms_key, bool apply_gelu);

cl_mem nnopt_gemv(OpenCLContext& cl_ctx, Weights& weights, cl_command_queue queue,
                  cl_mem x, const std::string& w_key,
                  int rows, int in_dim, int out_dim, cl_mem bias);

// Logical [out_dim, in_dim] of a GEMV weight, independent of how it is packed.
// A q4 tensor declares shape [N, K/2] (two weights per byte) — every call site that
// derived in_dim straight from shape[1] would silently halve K and read garbage.
bool nnopt_weight_dims(Weights& weights, const std::string& w_key, int* out_dim, int* in_dim);

// ── per-request toggle epoch ────────────────────────────────────────────────────────────────────
// Every NNOPT_* switch used to be cached in a function-local `static` on first use. In serve mode
// "first use" is the WARM-UP render, which runs before a request's key=value tokens reach the
// environment — so per-request toggles silently did nothing and the run reported the DEFAULT's
// numbers under the experiment's name. That is worse than having no toggles at all: it manufactures
// confident wrong measurements. (It is how `gemv v8` read as "no change" on the 840, and why a
// requested profile never appeared.)
//
// Bump once per generation; any cached env lookup keyed on the epoch re-reads.
int  nnopt_toggle_epoch();
void nnopt_toggle_bump();
