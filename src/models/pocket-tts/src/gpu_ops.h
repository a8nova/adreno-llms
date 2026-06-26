#pragma once
// gpu_ops.h — functional fp16 GPU op layer for the pocket-tts hand port.
//
// Thin wrappers over cl_mem buffers (fp16 storage, fp32 compute) that mirror the
// PyTorch ops in .nnport/PORT_SPEC.md. Buffers are raw cl_mem; the caller tracks
// shapes (passed explicitly). Reuses the scaffold's OpenCLContext + Weights +
// CLBlast (pytorch_linear). All compute kernels live in kernels/ops.cl.
//
// Convention: ops that PRODUCE a tensor return a NEW cl_mem the caller owns
// (release with clReleaseMemObject). In-place variants mutate the first arg.
#include "opencl_context.h"
#include "weights.h"
#include "utils.h"
#include <string>
#include <vector>
#include <unordered_map>

class GpuOps {
public:
    GpuOps(OpenCLContext& ctx, Weights& weights);
    ~GpuOps();

    bool init();   // builds kernels/ops.cl + kernels/utils.cl; false on failure

    cl_command_queue q() const { return ctx_.queue(); }

    // ── buffer helpers ───────────────────────────────────────────────────
    cl_mem alloc(size_t numel);                       // CL_MEM_READ_WRITE, fp16 elems
    cl_mem upload(const std::vector<float>& host);    // f32 → fp16 device buffer
    std::vector<float> download(cl_mem buf, size_t numel);  // fp16 → f32 host
    void release(cl_mem b);   // returns the buffer to a size-keyed pool (no clReleaseMemObject churn)

    // ── linear / matmul (CLBlast) ────────────────────────────────────────
    // y[M,N] = x[M,K] @ W[N,K]^T (+ bias[N]).  W,bias loaded from weights by key.
    // bias_key empty ⇒ no bias.  Returns new [M,N].
    cl_mem linear(cl_mem x, int M, int N, int K,
                  const std::string& w_key, const std::string& bias_key = "");

    // ── elementwise / activations (return new buffer) ────────────────────
    cl_mem copy(cl_mem in, size_t n);
    cl_mem elu(cl_mem in, size_t n);
    cl_mem silu(cl_mem in, size_t n);
    cl_mem gelu(cl_mem in, size_t n);
    cl_mem add(cl_mem a, cl_mem b, size_t n);
    cl_mem mul(cl_mem a, cl_mem b, size_t n);   // elementwise a*b
    void   add_inplace(cl_mem a, cl_mem b, size_t n);
    void   add_bias_inplace(cl_mem x, int rows, int C, const std::string& bias_key);
    cl_mem scale(cl_mem in, float s, size_t n);
    cl_mem mul_channel(cl_mem in, int rows, int C, const std::string& scale_key); // LayerScale

    // ── normalization ────────────────────────────────────────────────────
    // affine when w_key non-empty. eps per spec (1e-5 transformer / 1e-6 flow).
    cl_mem layernorm(cl_mem x, int rows, int C, float eps,
                     const std::string& w_key = "", const std::string& b_key = "");
    cl_mem rmsnorm(cl_mem x, int rows, int C, float eps, const std::string& alpha_key);

    // adaLN: out = x*(1+scale) + shift   (shift,scale: [rows,C])
    cl_mem modulate(cl_mem x, cl_mem shift, cl_mem scale, int rows, int C);
    // adaLN reading shift/scale from a packed buffer at column offsets (rows=1) —
    // avoids slice_cols. mul_off: out = a[aOff+c]*b[c] (packed gate*h).
    cl_mem modulate_packed(cl_mem x, cl_mem mod, int C, int shiftOff, int scaleOff);
    cl_mem mul_off(cl_mem a, int aOff, cl_mem b, int C);

    // latent de-norm: out = x*std + mean   (std,mean loaded by key, [C])
    cl_mem denorm_latent(cl_mem x, int rows, int C,
                         const std::string& std_key, const std::string& mean_key);

    // ── convolutions (channels-first [C,T]) ──────────────────────────────
    // Causal Conv1d, stride 1, left-pad (K-1)*dil zeros. Returns [Cout,T].
    cl_mem conv1d_causal(cl_mem x, int Cin, int Cout, int T, int K, int dil,
                         const std::string& w_key, const std::string& bias_key = "");
    // ConvTranspose1d stride S, first-frame output [Cout, T*S].
    cl_mem convtranspose1d(cl_mem x, int Cin, int Cout, int T, int K, int S,
                           const std::string& w_key, const std::string& bias_key = "");
    // Depthwise (groups==C) ConvTranspose1d, output [C, T*S]. (mimi.upsample)
    cl_mem convtranspose1d_depthwise(cl_mem x, int C, int T, int K, int S,
                                     const std::string& w_key);
    // SEANetResnetBlock: ELU→conv(k3,dim→dim/2)→ELU→conv(k1,dim/2→dim)→ +residual.
    cl_mem seanet_resnet_block(cl_mem x, int dim, int T, const std::string& prefix);

    // ── streaming (cross-frame state) variants ───────────────────────────
    // Causal conv carrying left-context. leftctx is [Cin, (K-1)*dil] (caller-
    // owned, zero-init on frame 0); updated in place to this frame's tail.
    cl_mem conv1d_streaming(cl_mem x, cl_mem leftctx, int Cin, int Cout, int T, int K, int dil,
                            const std::string& w_key, const std::string& bias_key = "");
    // ConvTranspose with overlap-add. partial is [Cout, K-S] (zero-init f0),
    // updated in place. Returns [Cout, T*S].
    cl_mem convtranspose1d_streaming(cl_mem x, cl_mem partial, int Cin, int Cout, int T, int K, int S,
                                     const std::string& w_key, const std::string& bias_key = "");
    // Depthwise variant (mimi.upsample). partial [C, K-S]. Returns [C, T*S].
    cl_mem convtranspose1d_depthwise_streaming(cl_mem x, cl_mem partial, int C, int T, int K, int S,
                                               const std::string& w_key);
    // Streaming resnet: block.1 conv (k3) carries leftctx; block.3 (k1) stateless.
    cl_mem seanet_resnet_block_streaming(cl_mem x, int dim, int T, const std::string& prefix,
                                         cl_mem b1_ctx);

    // ── transformer ───────────────────────────────────────────────────────
    cl_mem transpose2d(cl_mem in, int A, int B);   // [A,B] → [B,A]
    cl_mem slice_cols(cl_mem in, int T, int full, int c0, int w);
    void   rope_inplace(cl_mem x, int T, int H, int D, int offset, float max_period);
    // Fused qkv→(rope q,k)→(write k,v into kc/vc at offset); returns roped q[T,d].
    cl_mem qkv_rope_cache(cl_mem qkv, int T, int d, int H, int Dh, int offset,
                          float max_period, cl_mem kc, cl_mem vc);
    // attention over KV cache: q[Tq,H,D], kc/vc[Tkv,H,D] → [Tq,H,D]=[Tq,d]
    cl_mem attention(cl_mem q, cl_mem kc, cl_mem vc, int Tq, int Tkv, int H, int D,
                     int offset, int context);
    // write [T,d] k/v block into a [cap,d] cache at row `offset`.
    void   cache_append(cl_mem cache, cl_mem block, int offset, int T, int d);
    // One StreamingTransformerLayer (pre-norm, RoPE self-attn, GELU FFN).
    // kc/vc are this layer's [cap,d] caches; appends T positions at `offset`.
    // layer_scale_prefix empty ⇒ flow backbone (Identity scales).
    cl_mem transformer_layer(cl_mem x, int T, int d, int H, int dff, int offset,
                             int context, const std::string& prefix,
                             bool layer_scale, cl_mem kc, cl_mem vc, float max_period);

    // ── flow-net denoiser (stateless) ────────────────────────────────────
    // ResBlock: y→adaLN(shift,scale,gate); x + gate*mlp(modulate(in_ln(x),shift,scale)).
    cl_mem flow_resblock(cl_mem x, cl_mem y, int C, const std::string& prefix);
    // FinalLayer: y→adaLN(shift,scale); linear(modulate(norm_final(x),shift,scale)) → [out].
    cl_mem flow_final_layer(cl_mem x, cl_mem y, int C, int out_dim, const std::string& prefix);
    // SimpleMLPAdaLN denoiser: flow_net(c[1024], s, t, x_noise[32]) → [32].
    cl_mem flow_net(cl_mem c, float s, float t, cl_mem x_noise);

    // TimestepEmbedder: scalar t → [512]. cat(cos,sin)(t*freqs) → Linear,SiLU,
    // Linear,RMSNorm. (flow_net time conditioning; stateless.)
    cl_mem timestep_embedder(float t_val, const std::string& prefix);

    // ── embedding ────────────────────────────────────────────────────────
    // ids: host int token ids → [n_tok, dim] gathered from weights[w_key].
    cl_mem embedding(const std::vector<int>& ids, int dim, const std::string& w_key);

    // raw access for kernels written elsewhere (pipeline conv/attn)
    OpenCLContext& ctx() { return ctx_; }
    Weights& weights() { return weights_; }
    cl_program prog() { return prog_; }
    cl_kernel kernel(const std::string& name);  // cached clCreateKernel from ops.cl

    // ── record/replay (cl_qcom_recordable_queues) ────────────────────────────
    // Between begin_recording()/end_recording(), run1d dispatches are CAPTURED onto
    // a recordable queue (not executed) and alloc() returns fresh PINNED buffers
    // (no pool reuse/release, so the recording's buffer args stay valid for replay).
    bool can_record() const { return ctx_.has_recordable_queues(); }
    bool begin_recording();                 // false if unsupported
    cl_recording_qcom end_recording();      // returns the recording (replay many times)
    void replay(cl_recording_qcom rec, size_t num_overrides, const cl_array_arg_qcom* ov);
    // Per-layer kernel instances + arg indices that carry the per-step `offset`/`Tkv`,
    // collected during recording (each layer gets a fresh kernel so args are distinct).
    const std::vector<std::pair<cl_kernel,cl_uint>>& rec_offset_args() const { return rec_offset_args_; }
    const std::vector<std::pair<cl_kernel,cl_uint>>& rec_tkv_args() const { return rec_tkv_args_; }
    // device-buffer offset for qkv_rope_cache/attention (record/replay-safe). Two buffers:
    // backbone decode position and mimi-transformer dt_off (used in the SAME recorded frame).
    // use_mimi_offset(true) switches which one subsequent attention/qkv dispatches read.
    cl_mem offset_buf();                       // the ACTIVE offset buffer
    void   use_mimi_offset(bool m);            // select backbone(false) vs mimi(true) buffer
    void   write_offset(int off);              // set the ACTIVE buffer (live; transformer_layer)
    void   zero_buffer(cl_mem b, size_t numel); // in-place zero on live queue (state reset)
    cl_mem transposed_weight(const std::string& w_key, int N, int K); // cache [N,K]→[K,N] (live; call before recording)
    void   prewarm_mimi_gemm();                 // pre-transpose mimi transformer weights for the recordable aligned GEMM
    void   write_replay_offsets(int bb, int mm); // set BOTH buffers per replay (live, in-order)
    cl_command_queue aq() { return recording_ ? rec_q_ : ctx_.queue(); }  // active queue (CLBlast)
    bool   conv_gemm(int M, int N, int K, cl_mem x, cl_mem W, cl_mem out); // recordable GEMM / CLBlast
    bool   conv_gemm_atb(int M, int N, int Kc, cl_mem A, cl_mem B, cl_mem C); // recordable A^T@B / CLBlast

private:
    OpenCLContext& ctx_;
    Weights& weights_;
    cl_program prog_ = nullptr;        // ops.cl
    cl_program utils_prog_ = nullptr;  // utils.cl
    std::unordered_map<std::string, cl_kernel> kcache_;
    std::unordered_map<size_t, std::vector<cl_mem>> pool_;   // size(bytes) → free buffers (alloc/release churn killer)
    cl_mem offb_bb_ = nullptr, offb_mm_ = nullptr;   // [1] int offsets: backbone / mimi
    bool   use_mimi_off_ = false;                    // which offset buffer is active
    bool   custom_gemm_ = false;                     // NNOPT_CUSTOM_GEMM: force custom GEMM on live path (debug)
public:
    bool   rec_mimi_ = false;                        // NNOPT_RECORD_MIMI: include mimi in recorded frame (slow; default off)
private:
    cl_command_queue rec_q_ = nullptr;       // recordable queue (lazy)
    cl_recording_qcom cur_recording_ = nullptr;
    bool recording_ = false;
    std::vector<cl_mem> pinned_;             // buffers pinned during recording (freed in dtor)
    std::vector<cl_kernel> rec_kernels_;     // fresh per-dispatch kernels during recording
    std::vector<std::pair<cl_kernel,cl_uint>> rec_offset_args_, rec_tkv_args_;
    std::unordered_map<std::string, cl_mem> wt_cache_;   // pre-transposed [K,N] mimi-transformer weights (recordable aligned GEMM)
    cl_mem clblast_temp_ = nullptr;      // persistent CLBlast indirect-GEMM temp (avoids per-call alloc)
    cl_mem clblast_temp();
    cl_mem tc_cache_ = nullptr;          // flow_net timestep embedding (frame-invariant; cached)
    float  tc_s_ = 1e30f, tc_t_ = 1e30f; // (s,t) the cache was built for
    bool run1d(const std::string& kname, size_t global, std::vector<std::pair<size_t,const void*>> args);
    bool run1d_lws(const std::string& kname, size_t global, size_t local, std::vector<std::pair<size_t,const void*>> args);
    bool run2d_lws(const std::string& kname, size_t g0, size_t g1, size_t l0, size_t l1, std::vector<std::pair<size_t,const void*>> args);
    // tiled GEMM dispatch: C[M,N] = op(A)@op(B) (+bias). recordable. transA/transB select layout.
    bool gemm_tiled(int M, int N, int K, cl_mem A, cl_mem B, cl_mem C, int transA, int transB, cl_mem bias, int has_bias);
    cl_kernel rec_kernel(const std::string& kname);   // fresh kernel + collect per-step args (recording)
};
