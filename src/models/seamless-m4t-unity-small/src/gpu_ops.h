#pragma once
// GPU op layer for the SeamlessM4T UnitY on-device port. Wraps kernels/ops.cl.
// Activations live as fp16 (half) cl_mem buffers; weights come fp16 from the
// packed file via Weights::get_buffer. All arithmetic accumulates in fp32
// inside the kernels. Mirrors the host helpers in oracles/pipeline_ref.cpp.
#include <string>
#include <vector>
#include <unordered_map>
#include <cstdint>

#define CL_TARGET_OPENCL_VERSION 120
#include <CL/cl.h>

#include "opencl_context.h"
#include "weights.h"

// An activation tensor: a flat fp16 GPU buffer of `n` elements. Shape is
// tracked by the caller (the pipeline knows [T, D] etc.).
struct Tensor {
    cl_mem buf = nullptr;
    int n = 0;
    bool valid() const { return buf != nullptr; }
};

class GpuOps {
public:
    GpuOps(OpenCLContext& cl, Weights& w) : cl_(cl), w_(w) {}
    ~GpuOps();

    bool init(const std::string& kernel_path);

    // ---- buffer management ----
    Tensor alloc(int n);                                  // uninitialized fp16 buffer
    cl_mem alloc_f32(int n);                              // uninitialized fp32 buffer (tracked)
    Tensor clone(const Tensor& t);                        // copy into a fresh buffer
    Tensor upload(const std::vector<float>& host);        // float -> fp16 buffer
    std::vector<float> download(const Tensor& t);         // fp16 buffer -> float
    std::vector<float> download(const Tensor& t, int n);  // first n elems
    cl_mem upload_ints(const std::vector<int>& ids);      // int32 buffer (tracked)
    cl_mem alloc_ints(int n);                             // uninitialized int32 buffer (tracked)
    // Chained-decode helpers: argmax that writes the token to a device buffer WITHOUT a
    // host readback (out_idx()), a device int copy, and a bulk int readback. These let
    // the greedy loop chain token→embed entirely on-GPU and only sync EOS in batches.
    void argmax_dev(cl_mem buf, int n);                   // argmax -> out_idx(), no readback
    cl_mem out_idx() const { return out_idx_; }
    void copy_ints(cl_mem dst, int dst_off, cl_mem src, int src_off, int n);
    void download_ints(cl_mem buf, int off, int n, int* host);
    void topk_dev(cl_mem buf, int n, int cand_size, cl_mem cand_idx, cl_mem cand_val, float negval);
    cl_mem weight(const std::string& key, bool optional = false);  // fp16, owned by Weights
    Tensor copy_region(const Tensor& src, int off_elems, int n);   // copy n elems from src offset
    void copy_into(cl_mem dst, int dst_off_elems, const Tensor& src, int n);  // src[0..n) -> dst[off..]
    // src[src_off..src_off+n) -> dst[dst_off..]; pure clEnqueueCopyBuffer, NO alloc.
    // For slicing a batched [B,D] buffer per beam without copy_region's buffer churn.
    void copy_into_off(cl_mem dst, int dst_off_elems, cl_mem src, int src_off_elems, int n);
    void free_all();                                      // release all tracked activation buffers
    int mark() const;                                     // arena checkpoint
    void release_to(int m);                               // free activation buffers past checkpoint

    int storage_bytes() const;

    // ---- ops (each mirrors a loop in pipeline_ref.cpp) ----
    Tensor linear(const Tensor& x, int T, int Din, cl_mem w, cl_mem bias, int Dout);
    Tensor layernorm(const Tensor& x, int T, int D, cl_mem g, cl_mem b, float eps = 1e-5f);
    Tensor attention(const Tensor& q, const Tensor& k, const Tensor& v,
                     int Tq, int Tk, int H, int Dk, bool causal);
    Tensor relpos_attention(const Tensor& q, const Tensor& k, const Tensor& v,
                            const Tensor& p, cl_mem bu, cl_mem bv, int T, int L, int H, int Dk);
    Tensor conv1d_tc(const Tensor& in, cl_mem w, cl_mem bias, int T, int Cin, int Cout,
                     int Tout, int K, int stride, int pad, int dil, int groups);
    Tensor conv1d_ct(const Tensor& in, cl_mem w, cl_mem bias, int Cin, int T, int Cout,
                     int K, int dil, int pad, int pre_act = -1);  // pre_act>=0: fuse activation into im2col
    Tensor conv_transpose1d_ct(const Tensor& in, cl_mem w, cl_mem bias, int Cin, int T,
                               int Cout, int K, int stride, int pad);
    Tensor glu_tc(const Tensor& in, int T, int C);
    Tensor batchnorm_tc(const Tensor& in, int T, int C, cl_mem mean, cl_mem var,
                        cl_mem g, cl_mem b, float eps = 1e-5f);
    Tensor embed_scale_pos(cl_mem token_ids, cl_mem emb, int T, int Dm, float embed_scale,
                           int start, int pos_stride = 1);
    // Build the vocoder's channel-major input [Cin,T0] on the GPU (gather from lang/dict/spkr
    // weight buffers by per-token code ids) — replaces a ~580ms host build + dict page-in.
    Tensor vocoder_input_gather(cl_mem code_ids, cl_mem lang, cl_mem dict, cl_mem spkr,
                                int LE, int CE, int SE, int T0, int lang_id, int spkr_id);
    Tensor relpos_pos_emb(int L, int Dm, int ET);   // Conformer rel-pos sinusoid table

    // fbank (stage 2)
    Tensor fbank_window(const Tensor& audio, int nframes, int FL, int FS);
    cl_mem fbank_power(const Tensor& win, int nframes, int FL, int NFFT, int NSPEC);
    Tensor fbank_mel(cl_mem power, int nframes, int NB, int NSPEC, int SR, int NFFT, float eps);

    void act(const Tensor& t, int code);                   // in-place activation
    void act_n(const Tensor& t, int n, int code);          // in-place over first n
    void axpy(const Tensor& a, const Tensor& b, float alpha);  // a += alpha*b
    void scale(const Tensor& a, float alpha);

    // GPU token-selection arithmetic (fp32). Sequence bookkeeping stays host-side.
    void logits_to_f32(const Tensor& logits, cl_mem dst, int off, int n);
    void log_softmax_region(const Tensor& logits, cl_mem dst, int off, int vocab);
    void mask_region(cl_mem buf, int off, int vocab, int pad, int unk, int force_tok,
                     bool suppress_eos, int eos);
    void add_scalar_region(cl_mem buf, int off, int n, float scalar);
    void argmax_f32(cl_mem buf, int n, int& idx, float& val);
    void set_at_f32(cl_mem buf, int idx, float val);
    std::vector<float> download_f32(cl_mem buf, int n);  // debug

    // Audio pre/post-processing (WAV decode -> mono 16kHz fp16; fp16 -> PCM16).
    cl_mem upload_s16(const std::vector<int16_t>& s);
    cl_mem alloc_s16(int n);
    std::vector<int16_t> download_s16(cl_mem buf, int n);
    Tensor audio_decode(cl_mem s16, int nframes, int nch);
    Tensor audio_resample(const Tensor& in, int nin, int rate_in, int rate_out, int& nout);
    cl_mem audio_encode_s16(const Tensor& wav, int n);

    OpenCLContext& cl() { return cl_; }

public:
    // Per-kernel GPU profiling (guide §4.5.2: clGetEventProfilingInfo START→END =
    // true GPU exec time). Gated by NNOPT_GPUPROF. dump_gpu_prof() prints the
    // breakdown by kernel name + total GPU-active time.
    void dump_gpu_prof();
    // Discard all accumulated profiling stats (drop pending events too). Call
    // after a warmup pass so the timed run's profile excludes cold compilation.
    void prof_reset();
    // Record a stage boundary (drains pending events first). dump_gpu_prof then
    // prints per-stage GPU busy / span / utilization deltas so we can see WHICH
    // stage is idle-bound vs compute-bound.
    void prof_mark(const std::string& stage);

private:
    cl_kernel K(const std::string& name);
    void run1d(cl_kernel k, size_t global);
    // Profiled enqueue: when NNOPT_GPUPROF is set, attaches an event and accumulates
    // START→END nanoseconds per kernel name (looked up via kname_).
    cl_int pEnq(cl_kernel k, cl_uint dim, const size_t* global, const size_t* local);
    bool gpuprof_ = false; int gpuprof_init_ = 0;
    std::unordered_map<cl_kernel, std::string> kname_;   // kernel handle -> name
    std::unordered_map<std::string, double> prof_ns_;    // total GPU ns per kernel
    std::unordered_map<std::string, long> prof_calls_;   // dispatch count per kernel
public:
    // True iff NNOPT_GPUPROF is set (lazily initialized). Lets the CLBlast call
    // sites (linear/conv wrappers) decide whether to capture the GEMM event.
    bool gpuprof_on();
    // Register a profiling event under a kernel name (used for CLBlast GEMMs,
    // which don't go through pEnq). Batches + drains to bound live events.
    void prof_register(const std::string& name, cl_event ev);
private:
    // Span-utilization tracking: every profiled enqueue's event is collected,
    // drained in batches. We accumulate total GPU-busy ns (sum of kernel
    // durations) AND the global GPU timeline [first START, last END]. The gap
    // (span - busy) is true GPU idle = overhead between kernels (host sync, CPU).
    std::vector<std::pair<std::string, cl_event>> span_ev_;  // pending (name,event)
    double busy_ns_ = 0.0;                 // Σ (END-START) over all kernels
    unsigned long long gmin_ = 0, gmax_ = 0;  // global first START / last END (ns)
    bool gset_ = false;
    void prof_drain();                     // read+accumulate+release pending events
    // (stage, cumulative busy_ns, cumulative gmax_ns) snapshots at each prof_mark.
    std::vector<std::tuple<std::string, double, unsigned long long>> stage_marks_;
    template <typename T> void setArg(cl_kernel k, cl_uint idx, const T& v) {
        clSetKernelArg(k, idx, sizeof(T), &v);
    }

    OpenCLContext& cl_;
    Weights& w_;
    cl_program prog_ = nullptr;
    std::unordered_map<std::string, cl_kernel> kernels_;
    std::vector<cl_mem> arena_;       // activation buffers we own

    // Buffer POOL: the arena previously clCreateBuffer'd on every alloc() and
    // clReleaseMemObject'd on every release_to() — thousands of create/release pairs per
    // run (decode steps, convs). The col-scratch fix proved that churn is expensive; this
    // generalizes it: release_to() returns buffers to a free-list (keyed by byte size) and
    // alloc() reuses an exact-size free buffer instead of recreating. In-order queue ⇒ a
    // reused buffer's new write executes after all prior readers (SYNC-01). NNOPT_BUFPOOL=0 off.
    std::unordered_multimap<size_t, cl_mem> free_pool_;  // freed buffers by byte size
    std::unordered_map<cl_mem, size_t> buf_bytes_;       // byte size of every pooled buffer
    int bufpool_ = -1;
    bool bufpool_on();
    cl_mem pool_alloc(size_t bytes, const char* tag);    // pooled uninitialized READ_WRITE buffer
    cl_mem dummy_ = nullptr;          // bound where a kernel has an unused bias arg
    cl_mem out_idx_ = nullptr;        // argmax result scratch (1 int)
    cl_mem out_val_ = nullptr;        // argmax result scratch (1 float)
    cl_mem lsm_partial_ = nullptr;    // log_softmax partial reduction scratch (256 floats)
    cl_mem lsm_scalar_ = nullptr;     // log_softmax [max, sumexp] scratch (2 floats)
    bool ok_ = false;

    // int8 weight cache (per-row symmetric quant), keyed by the fp16 weight cl_mem.
    // Built lazily on first use: read back fp16 weight, per-row max-abs scale,
    // quantize to int8, upload int8 buffer + fp16 scale buffer. NNOPT_INT8 gates it.
    struct Int8W { cl_mem q = nullptr; cl_mem scale = nullptr; int N = 0, K = 0; };
    std::unordered_map<cl_mem, Int8W> int8_cache_;
    int int8_mode_ = -1;              // -1 unknown, 0 off, 1 on (from NNOPT_INT8)
    bool int8_enabled();
    const Int8W* get_int8(cl_mem w, int N, int K);

    // image2d (L1 texture-cache) weight path for decode GEMV — LOSSLESS 1.71× BW.
    // Built in a SEPARATE cl_program (img_prog_) so it doesn't share register
    // allocation with the buffer kernels. Image-from-buffer view cached per weight.
    cl_program img_prog_ = nullptr;
    std::unordered_map<cl_mem, cl_mem> img_cache_;   // weight buffer -> image2d
    int img_mode_ = -1;               // -1 unknown, 0 off, 1 on (NNOPT_IMG, default on under fp16)
    bool img_enabled();
    cl_mem get_image(cl_mem w, int N, int K);  // nullptr if ineligible/unsupported

    // Transposed-conv weight repack cache: w[Cin,Cout,K] -> Wr[Cout*K, Cin] (tap-major),
    // so the upsample becomes a GEMM (tmp[Cout*K,T]=Wr@in) + col2im fold. Keyed by w cl_mem.
    std::unordered_map<cl_mem, cl_mem> wt_cache_;
    cl_mem get_wt_transpose(cl_mem w, int Cin, int Cout, int K);

    // Reused im2col scratch: the vocoder runs ~95 convs, each previously alloc'd a fresh
    // [Cin*K, T] col buffer (tens of MB for late upsamples) → clCreateBuffer churn + the
    // big im2col write. One growable persistent buffer reused across convs (in-order queue
    // ⇒ prior conv's GEMM finishes reading before next conv's im2col writes). NNOPT_CONV_SCRATCH=0 disables.
    cl_mem col_scratch_ = nullptr; int col_scratch_n_ = 0;
    int conv_scratch_mode_ = -1;     // -1 unknown, 0 off, 1 on
    cl_mem col_scratch(int n);       // persistent buffer with >= n fp16 elems (nullptr if disabled)

    // NNOPT_CONVPROF: clFinish-attributed host time per conv sub-phase (im2col / CLBlast
    // GEMM / bias), accumulated across all conv1d_ct calls, printed in dump_gpu_prof. Reveals
    // whether the vocoder idle is CLBlast host overhead vs the im2col write vs bias.
    int convprof_ = -1; double cp_im2col_ = 0, cp_gemm_ = 0, cp_bias_ = 0; long cp_calls_ = 0;
    bool convprof_on();

public:
    // Concatenate 3 [N,K] weight (and [N] bias) buffers into one [3N,K] / [3N] buffer,
    // cached by the q buffer — for a fused QKV projection (3 GEMVs → 1). nullptr bias ok.
    struct QKV { cl_mem w = nullptr; cl_mem b = nullptr; };
    const QKV* get_qkv(cl_mem qw, cl_mem kw, cl_mem vw, cl_mem qb, cl_mem kb, cl_mem vb, int N, int K);
private:
    std::unordered_map<cl_mem, QKV> qkv_cache_;
};
