// OpenCL device model for the qwen35 hybrid stack — mirrors the host forward
// in reference_model.h 1:1.
//
// The structural difference from the dense bonsai port: the block loop is
// HETEROGENEOUS. meta.layer_types is one char per block ('L' = Gated DeltaNet
// linear attention, 'F' = gated full attention), so weights, per-block state
// and dispatch all branch on kind:
//
//   'L'  attn_qkv -> conv1d+silu -> delta rule -> gated RMSNorm -> ssm_out
//        state: recurrent [n_v * d_k * d_v] fp32 + conv window [qkv * conv_k] fp32
//        NO KV cache
//   'F'  attn_q(q‖gate) / k / v -> q,k norm -> partial RoPE -> GQA
//        -> *sigmoid(gate) -> attn_output
//        state: KV cache
//
// Both kinds then run the same SwiGLU MLP. Kernel families live in separate
// programs (Adreno register-footprint isolation): q1_gemv.cl / rmsnorm.cl /
// rope.cl / attention.cl / gated_attention.cl / delta_net.cl / mlp.cl / utils.cl
#pragma once
#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <functional>
#include <map>
#include <string>
#include <vector>

#include "nnb.h"
#include "q1.h"
#include "opencl_context.h"

struct OpProf {
    bool on = false;
    std::map<std::string, std::pair<double, int>> acc;
    std::chrono::steady_clock::time_point t0;
    cl_command_queue q = nullptr;
    void begin() { if (on) { clFinish(q); t0 = std::chrono::steady_clock::now(); } }
    void end(const char* k) {
        if (!on) return;
        clFinish(q);
        double ms = std::chrono::duration<double, std::milli>(
                        std::chrono::steady_clock::now() - t0).count();
        acc[k].first += ms; acc[k].second++;
    }
    // `out` is a parameter so a caller can route this somewhere other than stderr; the
    // BONSAI_PROF=1 path uses the default.
    void dump(FILE* out = stderr) {
        if (!on && acc.empty()) return;
        double tot = 0;
        for (auto& kv : acc) tot += kv.second.first;
        if (tot <= 0) return;
        fprintf(out, "=== DEVICE PROFILE ===\n");
        for (auto& kv : acc)
            fprintf(out, "%-16s %10.1f ms %7.1f%% %8d calls %8.3f ms/call\n",
                    kv.first.c_str(), kv.second.first,
                    100.0 * kv.second.first / tot, kv.second.second,
                    kv.second.first / kv.second.second);
        fprintf(out, "TOTAL %10.1f ms\n", tot);
        fflush(out);
    }
};

#define CLCHECK(err, what)                                            \
    do {                                                              \
        cl_int e_ = (err);                                            \
        if (e_ != CL_SUCCESS) {                                       \
            fprintf(stderr, "FATAL CL %d at %s\n", e_, what);         \
            exit(4);                                                  \
        }                                                             \
    } while (0)

class DeviceModel {
  public:
    // Resident device has room for 2048 context.
    static constexpr int CTX_CAP = 2048;

    /**
     * Named tunable presets, reachable as "/bench N" from the chat box.
     *
     * Environment variables are the wrong interface for the only devices that matter here: the
     * phones under test run this engine from inside the app, where nobody can set one. A preset
     * index is something a person can type into a chat box with one thumb.
     *
     * The set is chosen from the PowerVR census — 1829 dispatches per token at 312 us each, 54% of
     * them split-K — so presets 1-3 all disable split-K, which is the hypothesis worth the most.
     */
    // `gemv` names the plain quad-row kernel to use for the non-split, non-texture path; nullptr
    // keeps q1_gemv4. `img`/`lut` override the texture and LUT gates (-1 = leave alone).
    //
    // These exist because the interesting decisions were unreachable from the device. Six kernels
    // were written against specific bottlenecks in the Qualcomm guide and never measured, and the LUT
    // path carries a comment claiming 1.30x over the shipped GEMV on an 840 yet ships disabled — none
    // of which could be settled from inside the app, which is the only place the 840 is reachable.
    // A variant preset also forces texture and split-K off: the variants take a plain buffer, so this
    // isolates the INNER LOOP against a same-configuration q1_gemv4 baseline (preset 1). Whichever
    // wins still has to be folded into the shipping split-K + texture kernel to count.


    /**
     * Zero the recurrent state — call before every independent turn.
     *
     * This is NOT symmetric with the KV cache, and the asymmetry is the whole point. A turn that
     * restarts at pos 0 rewrites kcache_/vcache_[0..len) before attention ever reads them, so the
     * KV side self-heals. rec_ (the Gated-DeltaNet state) and convs_ (the causal-conv window) are
     * ACCUMULATORS with no position index: nothing about starting over at pos 0 clears them, so
     * turn N decodes on top of everything turns 1..N-1 put there. On a 48-of-64-layers linear-
     * attention model that is most of the network reading corrupt state, which is why the symptom
     * is a fluent-looking degenerate loop rather than an obvious crash.
     */
    void reset_state();

    /**
     * Decode one step WITHOUT the host supplying the token.
     *
     * The ordinary path is forward(tok) -> read_token(), and read_token is a BLOCKING 4-byte
     * clEnqueueReadBuffer: it drains the whole pipeline once per token just to learn a number the
     * GPU already has. /bench never paid it — it times eight forward() calls back to back — which is
     * why the bench number reads ~15% above what the app reports (8.0 vs 6.9, 7.6 vs 6.2).
     *
     * q1_row_gather3_dev takes the token from a device buffer, so the next step can be enqueued
     * before the host has looked at anything. The argmax destination alternates between two slots so
     * the token being read back cannot be overwritten by the step already in flight.
     *
     * Returns the slot holding THIS step's token; pass it to read_token_slot() when convenient.
     */
    int forward_dev(int pos, bool want_logits);

    /** Read the token from a slot returned by forward_dev(). Blocking, but by then it is ready. */
    int read_token_slot(int slot);

    /** Seed the token the first forward_dev() will consume (the prompt's last token). */
    void seed_token(int token);

    /**
     * Workgroup size for q1_gemv.cl, compiled in as -DWG_SIZE and used by every dispatch of a kernel
     * in that file that declares reqd_work_group_size(WG_SIZE) — currently q1_gemv4 and q1_xsum.
     * (q1_argmax pins reqd(256) and q1_row_gather3 declares none, so both are independent of this.)
     *
     * 64, MEASURED on an Adreno 840 (12 CU), MLP shape K=5120 N=17408:
     *     wg 64 -> 0.32 ms (39.6 GB/s)   wg 128 -> 0.37 ms   =>  1.16x
     * which is ~1.10x end to end, since this shape is ~64% of all weight traffic.
     *
     * This REPLACES a 128 that the file called VALIDATED, quoting `wg 64 -> 0.41, wg 128 -> 0.36`.
     * That sweep could not have produced those numbers in this state of the tree: q1_gemv.cl gained
     * q1_gemm, the sweep's hand-rolled build options never gained -DGEMM_MT, and one kernel failing
     * to compile fails the whole program — so every row printed "program build FAILED". The old
     * numbers predate that regression, and predate the row-offset fix in seven other kernels.
     *
     * The direction is corroborated by two independent probes in the same run: split-K improves
     * monotonically with group count (10 groups 20.4 GB/s -> 80 groups 34.5 GB/s), and halving the
     * workgroup doubles the groups. This GPU is short of parallelism, not of per-group work.
     *
     * Only ever change this on a sweep where the winning row says `ok` — the check column is only
     * meaningful because x is now filled with a varied pattern and the output is poisoned first.
     *
     * The host's gws/lws and the kernel's reqd_work_group_size must agree or every dispatch fails.
     *
     * RUNTIME as of the PowerVR port, defaulting to the swept-and-validated Adreno value above, so
     * nothing changes on that hardware unless a device explicitly asks for something else.
     *
     * Why it had to stop being compile-time: 128 is two Adreno waves, and the tree reduce over
     * partial[WG_SIZE] is nearly free inside a wave. Imagination's scheduler assembles work into warps
     * of up to 32, so on a PowerVR DXT the same 128 spans four warps and every step of that reduction
     * becomes a real barrier. It also sets occupancy — at WG=128 x ROWS=4 one workgroup covers 512
     * output rows, so a 4096-row GEMV launches 8 groups: fine for a 12-CU Adreno, starvation on the
     * 6-CU DXT-48-1536, which wants ~512 work-items per USC to stay busy.
     *
     * Override with BONSAI_Q1_WG. Read once in select_tunables() and never changed after, because the
     * kernels are compiled with -DWG_SIZE from this value and declare reqd_work_group_size(WG_SIZE):
     * host and kernel must agree or every dispatch fails with CL_INVALID_WORK_GROUP_SIZE.
     */
    static constexpr int Q1_WG_DEFAULT = 64;
    int Q1_WG = Q1_WG_DEFAULT;

    /**
     * Output rows accumulated per work-item by q1_gemv4. Measured optimal at every workgroup size.
     *
     * NOT a knob, deliberately. The row count selects a different KERNEL, not a different constant:
     * q1_gemv9 is 1 row, q1_gemv_duo 2, q1_gemv4 4, q1_gemv_hex 6, q1_gemv5 8 — and only the 4-row
     * family has the split-K and image variants the decode path binds (q1_gemv4_sk, q1_gemv4_img,
     * q1_gemv4_sk_img). If a narrower row count wins on a given GPU
     * the sweep will say so — but the missing variants have to be written before it can ship.
     */
    static constexpr int Q1_ROWS = 4;

    /**
     * Split-K thresholds. Below SK_TARGET_GROUPS row-blocks the matrix cannot fill 12 CUs on its own,
     * so run_gemv() splits the K loop instead (see q1_gemv4_sk).
     *
     * MEASURED on the down-projection (K=17408 N=5120), where group count is the ONLY variable:
     *     no split   20 groups  0.60 ms  20.9 GB/s
     *     split=2    40 groups  0.42 ms  29.5 GB/s
     *     split=4    80 groups  0.34 ms  37.4 GB/s
     *     split=8   160 groups  0.29 ms  42.8 GB/s
     * Monotonic, and 42.8 GB/s beats anything the un-split kernel reaches at any workgroup size — so
     * the target is set high enough to drive even the wide matrices toward it, rather than the timid
     * 32 (~2.5 groups/CU) this started at. The reduce pass has not yet cost more than it buys at
     * SK_MAX_SPLITS=8; if a future sweep shows it does, lower this, don't lower the cap.
     *
     * Runtime for the same reason as Q1_WG: 160 groups is ~13 per CU on the 12-CU Adreno it was tuned
     * on, and this is the ONE lever already proven to buy 2x (20.9 -> 42.8 GB/s above) purely by
     * raising group count. A 6-CU part wants a different target. Override with BONSAI_SK_GROUPS.
     * Default is unchanged, so Adreno keeps the exact split behaviour it was tuned to.
     */
    // 320, from an INTERLEAVED A/B on the Adreno 840 (/ab 0 16): 122.7 vs 124.7 ms/token, +1.6%.
    // Marginal — under the 2% the tool calls a difference — but it was faster in 4 of 5 rounds and
    // it agrees with every other occupancy result on this part. 96 was measured WORSE (129.2 ms),
    // so this is a peak, not a slope: do not keep raising it without a fresh A/B.
    static constexpr size_t SK_TARGET_GROUPS_DEFAULT = 320;
    size_t SK_TARGET_GROUPS = SK_TARGET_GROUPS_DEFAULT;

    /**
     * Tokens per tile in the batched-prefill GEMM (q1_gemm), compiled in as -DGEMM_MT. Weight traffic
     * during prefill falls by this factor: a prompt of M tokens sweeps the 3617 MB of weights
     * ceil(M/GEMM_MT) times instead of M times. Bounded by registers (GEMM_MT accumulators per
     * work-item) and local memory (GEMM_MT*512 B of staged activations, against 32 KB).
     */
    /**
     * 2, MEASURED on the Adreno 840 that actually runs this model (ffn_gate N=17408 K=5120, 32
     * tokens), reading wall time — the column that decides prefill:
     *     r1 MT=2   10.56 ms   private  672 B/wi   (16 tiles)   <-- selected
     *     r1 MT=4   12.82 ms   private  672 B/wi   ( 8 tiles)       previous default
     *     r1 MT=8   14.30 ms   private  704 B/wi
     *     r1 MT=16  96.76 ms   private 2340 B/wi   (spilled)
     * against 14 ms for the 32 sequential GEMVs it replaces, so MT=2 is 1.33x where MT=4 was 1.09x.
     * Do not read the GB/s column to rank these: fewer tokens per tile means MORE weight traffic, so
     * a slower configuration can show a higher rate while taking longer.
     *
     * The previous default came from a table measured on an Adreno 620 — a 1-CU part that is not the
     * target and ranks these differently. That table is deleted rather than kept for reference: it
     * read as authoritative for long enough to ship the wrong constant, and a stale measurement from
     * the wrong device is worse than none. TUNE THIS ON THE 840 ONLY, via /bench.
     */
    static constexpr int GEMM_MT = 2;
    // 32. This was 8, and 8 was a CEILING the probe never got past rather than a measured optimum:
    // split-K improved monotonically all the way to it (10 groups 20.4 GB/s -> 80 groups 34.5 GB/s
    // on the Adreno 840) and then stopped because the cap stopped it.
    //
    // It binds hardest exactly where the GEMV is worst. attn_q_a/q_b are N=48: one row-block, so the
    // split is the ONLY source of parallelism and 8 groups on a 12-CU part leaves a third of the
    // machine idle, 96 times per token. attn_k/v (N=1024) want 40 splits and get 8.
    //
    // The `U / splits >= 2` guard in run_gemv still bounds this per shape (K=5120 -> U=40 -> at most
    // 20), so raising the cap widens the range the guard can choose from; it does not force it.
    // Costs sk_partial_: SK_MAX_SPLITS * SK_MAX_N * 4 B = 21 MB, against 5542 MB of device memory.
    static constexpr int SK_MAX_SPLITS = 32;
    /**
     * Widest N that can ever take the split path, for sizing sk_partial_.
     *
     * Deliberately built from the DEFAULTS, not the runtime values: the buffer is allocated once in
     * make_scratch() and this has to be a compile-time constant. select_tunables() therefore clamps
     * any override so that SK_TARGET_GROUPS * Q1_WG * Q1_ROWS never exceeds this. Lowering either
     * knob (the PowerVR direction) shrinks the product and is always safe; only raising one can
     * overflow, and that is what the clamp catches.
     */
    // Sized from the LARGEST workgroup any preset selects (256), not from Q1_WG_DEFAULT. Tying it to
    // the default meant lowering that default shrank the partials buffer, so every preset with a
    // bigger workgroup got its split target silently clamped — and a clamped comparison is not a
    // comparison. The buffer is fp32 partials, so this costs a few MB and removes the coupling.
    static constexpr size_t SK_MAX_N = 320 * (size_t)256 * Q1_ROWS;
#ifdef BONSAI_FP16_STORAGE
    // NOT USABLE YET — the fp16 build produces "!!!!" (all-NaN logits, argmax -> token 0).
    //
    // The activation dtype is not a single switch on this architecture. Several buffers are fp32 BY
    // DESIGN and must stay that way: core_ (the delta-net output, which feeds the fp32 recurrent
    // state), convs_ (the causal-conv window), and rec_ itself. Halving them destroys the state.
    // But every kernel that consumes them is typed `storage_t`, so under fp16 they reinterpret fp32
    // bits as pairs of halves and the network fills with NaN on the first 'L' block.
    //
    // Making fp16 work means typing the delta-net path explicitly `float` — mamba_rms_norm_gated,
    // causal_conv1d, delta_net, plus float-input variants of xsum and the ssm_out GEMV — while the
    // rest of the network moves to half. That is a real change and it needs a device that can load
    // the 27B to validate; the 620-class part here cannot (needs 4.8 GB, has 3.7 GB).
    static constexpr size_t ES = 2;
#else
    static constexpr size_t ES = 4;
#endif

    DeviceModel(const Nnb& nnb, OpenCLContext& ocl, const std::string& kdir);
    ~DeviceModel();

    int forward(int token, int pos, bool want_logits);

    /**
     * Same forward, but the residual stream starts from a PROVIDED embedding instead of a row of
     * token_embd. This is the splice point for vision: the tower's merged patch embeddings are
     * already in the text model's 5120-d space (mm.2 projects to out_hidden == hidden), so an image
     * token is just a position whose embedding came from pixels rather than from the vocab.
     * `emb` is `hidden` floats.
     */
    int forward_embed(const float* emb, int pos, bool want_logits);

    /**
     * Prefill `count` supplied embeddings at positions [pos0, pos0+count) in ONE pass.
     *
     * forward_embed() per token re-reads all 3.5 GB of weights for every token: a 760-token image
     * moves ~3 TB and takes ~160 s. Here the weight-bearing projections run as q1_gemm over a tile of
     * PF_CHUNK tokens, so the weights are swept ceil(count/PF_CHUNK) times instead of `count`.
     *
     * The recurrent cores (causal conv1d, the delta-net state update, and the attention KV append)
     * stay strictly per-token and in position order — they carry state, so they cannot be batched —
     * but they touch no large weights, which is why batching only the projections captures nearly all
     * the traffic. Produces no logits: prefill never needs them.
     */
    void forward_batch(const float* embs, int count, int pos0);

    /** True when the load-time measurement found the batched GEMM faster on this GPU. */
    bool prefill_batched() const { return prefill_batched_; }

    /** Tokens per prefill tile. Bounded by the [PF_CHUNK][ffn] staging buffers, not by the kernel. */
    static constexpr int PF_CHUNK = 32;

    /** Multimodal RoPE: rebuild the rope tables from per-position (t,h,w) ids. See the definition. */
    void set_mrope(const std::vector<std::array<int, 3>>& thw);
    int read_token();
    void print_topk();



  private:
    // A 1-bit weight — its own resident device buffers.
    struct QW {
        cl_mem bits = nullptr, scales = nullptr;
        // image1d_buffer VIEW of `bits`, or null when the tensor is wider than
        // CL_DEVICE_IMAGE_MAX_BUFFER_SIZE texels. Aliases the same allocation — no extra memory.
        cl_mem bits_img = nullptr;
    };

    // 'L' — Gated DeltaNet. `wqkv` is in_proj_qkv (q‖k‖v fused, already one
    // tensor upstream); `wz` is in_proj_z, which the GGUF confusingly names
    // "attn_gate" — it is the SSM output gate, not an attention gate.
    struct LinearLayerW {
        // qkv + z + alpha + beta FUSED: all four read the same post-norm x, and alpha/beta are N=48
        // each — a matrix so narrow that no split-K fills a 12-CU GPU, dispatched 96 times per token.
        QW win, wout;
        int in_off[4] = {0, 0, 0, 0};   // row offsets of qkv, z, alpha, beta within win
        int in_n = 0;                   // fused row count
        cl_mem conv_w;        // [channels][conv_k] f32
        cl_mem A_log, dt_b;   // [n_v] f32
        cl_mem ssm_norm;      // [d_v] f32
    };
    // 'F' — gated attention. `wq` emits q AND the output gate interleaved per
    // head (n_heads * head_dim * 2 columns).
    struct FullLayerW {
        // q + k + v FUSED — same x, and the three offsets happen to land already aligned.
        QW win, wo;
        int in_off[3] = {0, 0, 0};
        int in_n = 0;
        cl_mem q_norm, k_norm;
    };
    struct BlockW {
        char kind;                 // 'L' | 'F'
        int idx;                   // index into lin_ / full_
        // ffn gate+up FUSED into one [K, 2*ffn] matrix, plus down. Both halves read the same x and
        // write the two halves of gu_, which gate_/up_ already alias as sub-buffers — so fusing them
        // costs nothing on the output side and removes 64 dispatches per token.
        QW wgu_up, wd;
        int gu_off[2] = {0, 0};    // row offset of gate, up within wgu_up
        cl_mem attn_norm, post_norm;
    };

    int forward_body(int pos, bool want_logits);
    void build_programs(const std::string& kdir);
    cl_mem upload(const void* p, size_t n, const char* what);
    cl_mem scratch(size_t n, const char* what);
    QW upload_q1(const std::string& n);

    /**
     * Upload several Q1T tensors as ONE matrix, concatenated along N.
     *
     * Every projection in a group reads the SAME activation vector, so running them as separate
     * GEMVs pays a dispatch and a full kernel launch per projection to stream weights that could
     * have been one contiguous read. Worse, it makes the tiny ones tiny: ssm_alpha/ssm_beta are
     * N=48, one row-block, and there is no amount of split-K that fills a 12-CU GPU with a 34 KB
     * matrix — that shape ran 96 times per token.
     *
     * Q1T is [U][N] for both streams, so concatenating along N means, for each u, laying the
     * tensors' row-slices end to end. `starts` receives each tensor's row offset, PADDED so every
     * offset is a whole number of CL_DEVICE_MEM_BASE_ADDR_ALIGN bytes — the consumers read their
     * slice through clCreateSubBuffer, and that call rejects an unaligned origin. Padding rows are
     * zero-filled; their outputs are computed and discarded, which costs 32 of 16512 rows in the
     * 'L' group and nothing anywhere else.
     */
    QW upload_q1_fused(const std::vector<std::string>& names, std::vector<int>* starts,
                       int* n_total);
    void upload_weights();
    void make_scratch();
    void make_rope_tables();

    // ---- dispatch helpers ----
    void arg(cl_kernel k, int i, size_t sz, const void* v, const char* what) {
        CLCHECK(clSetKernelArg(k, i, sz, v), what);
    }
    void run1(cl_kernel k, size_t gws, size_t lws, const char* what) {
        CLCHECK(clEnqueueNDRangeKernel(ocl_.queue(), k, 1, nullptr, &gws,
                                       lws ? &lws : nullptr, 0, nullptr, nullptr),
                what);
    }

    /**
     * Hardware GPU timing, per Snapdragon guide 4.5.2. Four timestamps per dispatch:
     *   QUEUED -> SUBMIT   host/driver software overhead
     *   SUBMIT -> START    launch latency (the queue waiting to run it)
     *   START  -> END      actual GPU execution
     * A host-side clFinish timer collapses all three into one inflated number, which is why the
     * clFinish-based census could never answer "is this kernel slow, or just launched a lot".
     */

    /** 2D dispatch: {row-blocks x WG, splits}. Lets the hardware carry the split index so the kernel
     *  does not have to recover it with an integer divide+modulo (guide 8.12). */
    void run2(cl_kernel k, size_t gx, size_t gy, size_t lx, const char* what) {
        const size_t gws[2] = {gx, gy}, lws[2] = {lx, 1};
        CLCHECK(clEnqueueNDRangeKernel(ocl_.queue(), k, 2, nullptr, gws, lws, 0, nullptr, nullptr), what);
    }
    long disp_n_ = 0;
    std::map<std::string, long> disp_by_;

    /**
     * Pick the device-dependent tunables (Q1_WG, SK_TARGET_GROUPS) before anything is allocated or
     * compiled. Defaults are the Adreno-swept values, so this is a no-op on that hardware.
     */
    void select_tunables();

    // ---- block kinds (src/layers/) ----
    void linear_block(int L, const BlockW& bw, const LinearLayerW& w);
    void full_block(int L, const BlockW& bw, const FullLayerW& w, int pos);
    void mlp_block(int L, const BlockW& bw);

    // ---- ops ----
    void run_gemv(const QW& W, cl_mem x, cl_mem out, int N, int K, int out_off = 0);
    void run_xsum(cl_mem x, int K);
    void run_gather(const QW& W, cl_mem out, int token, int K);
    void run_rms(cl_mem x, cl_mem w, cl_mem out, int rows, int cols);
    void run_add(cl_mem acc, cl_mem b, int n);
    void run_argmax(int N);
    void run_swiglu(cl_mem g, cl_mem u, int n);
    // 'L'
    void run_conv1d(cl_mem qkv, cl_mem state, cl_mem w, int channels, int conv_k);
    void run_delta_net(cl_mem S, cl_mem qkv, cl_mem a, cl_mem b, cl_mem A_log,
                       cl_mem dt_b, cl_mem out);
    void run_norm_gate(cl_mem x, cl_mem gate, cl_mem w);
    // 'F'
    void run_split_qg(cl_mem qg, cl_mem q, cl_mem gate, int NH, int D);
    void run_rope_partial(cl_mem x, int n_heads, int D, int pos);
    void run_apply_gate(cl_mem att, cl_mem gate, int n);
    void run_scores(cl_mem qb, cl_mem kc, int NH, int KH, int D, int seq_k);
    void run_softmax(int NH, int seq_k);
    void run_attnout(cl_mem vc, int NH, int KH, int D);
    OpProf prof_;
    const Nnb& nnb_;
    const ModelMeta& m_;
    OpenCLContext& ocl_;

    // qwen35 dims (GGUF ssm.* keys are Mamba-inherited misnomers — see
    // PORT_CHECKLIST.md §D)
    int nv_ = 0;        // ssm.time_step_rank == num_v_heads
    int nk_ = 0;        // ssm.group_count    == num_k_heads
    int dk_ = 0, dv_ = 0;
    int key_dim_ = 0, value_dim_ = 0, qkv_dim_ = 0, conv_k_ = 0, rot_ = 0;

    std::vector<BlockW> blocks_;
    std::vector<LinearLayerW> lin_;
    std::vector<FullLayerW> full_;
    QW tok_embd_, w_out_;
    cl_mem out_norm_ = nullptr, cos_ = nullptr, sin_ = nullptr,
           counter_ = nullptr, amax_ = nullptr;
    // shared activations
    cl_mem x_, xb_, mix_, gu_, gate_, up_, logits_, xsum_;
    // 'L' scratch
    cl_mem qkv_, z_, avec_, bvec_, core_;
    // 'F' scratch
    cl_mem qg_, q_, agate_, k_, v_, att_, scores_;
    // per-block state (only the slot matching the block kind is allocated)
    std::vector<cl_mem> rec_, convs_, kcache_, vcache_;

    // ── batched-prefill scratch ────────────────────────────────────────────────────────────────
    // Row-major [PF_CHUNK][*] mirrors of the single-token buffers above. Allocated on first use, so
    // a text-only session never pays for them.
    cl_mem pf_x_ = nullptr, pf_n_ = nullptr, pf_mix_ = nullptr, pf_xsum_ = nullptr;
    cl_mem pf_gate_ = nullptr, pf_up_ = nullptr;
    cl_mem pf_qkv_ = nullptr, pf_z_ = nullptr, pf_a_ = nullptr, pf_b_ = nullptr, pf_core_ = nullptr;
    cl_mem pf_qg_ = nullptr, pf_k_ = nullptr, pf_v_ = nullptr, pf_att_ = nullptr;
    bool pf_ready_ = false;
    // Per-row VIEWS of the batch buffers, created ONCE. The recurrent cores are per-token and take
    // single-token buffers, so the batched path needs a handle to row m — but creating it inside the
    // token loop meant ~74k clCreateSubBuffer/clReleaseMemObject pairs per image, and Qualcomm's
    // guide (80-NB295-11 §5.7.1) calls out allocation between kernel calls as a top API-level
    // pitfall. Offsets are fixed, so these are hoisted to load time.
    std::vector<cl_mem> pf_qkv_r_, pf_z_r_, pf_core_r_, pf_qg_r_, pf_k_r_, pf_v_r_, pf_att_r_;
    void make_prefill_scratch();
    /** q1_gemm over `rows` tokens: out[rows][N] = W * x[rows][K]. Same weights as run_gemv. */
    // n_stride/n0 select a SLICE of a fused weight: n_stride is the fused matrix's row count,
    // n0 the slice's first row. Defaults reproduce the unfused case exactly.
    void run_gemm(const QW& W, cl_mem x, cl_mem xs, cl_mem out, int N, int K, int rows,
                  int n_stride = 0, int n0 = 0);
    /** Batched activation sums, [rows][K/64], the layout q1_gemm expects. */
    void run_xsum_m(cl_mem x, cl_mem xs, int K, int rows);

    cl_mem amax2_[2] = {nullptr, nullptr};   // double-buffered argmax, see forward_dev
    int amax_slot_ = 0;
    cl_kernel k_gather_dev_ = nullptr;
    cl_mem lin_in_ = nullptr, full_in_ = nullptr;   // fused input-projection outputs
    cl_mem sk_partial_ = nullptr;   // [SK_MAX_SPLITS][SK_MAX_N] fp32 split-K partials
    cl_kernel k_gemv_sk_ = nullptr, k_sk_reduce_ = nullptr;
    // Texture path. Enabled per tensor: a weight too wide for image1d_buffer keeps the buffer kernel,
    // so this is a per-call choice, not a global mode.
    cl_kernel k_gemv_img_ = nullptr, k_gemv_sk_img_ = nullptr;
    // Output head keeps fp32 logits whatever the activation dtype — see q1_gemv4_img_f32.
    cl_kernel k_gemv_head_ = nullptr, k_argmax_f32_ = nullptr;
    void run_head(const QW& W, cl_mem x, cl_mem out, int N, int K);
    // Split-K with the reduce fused in. sk_reduce was 27% of a decode token and 16.7 us of
    // that was its own launch (an empty kernel costs 16.8 us on the 840), so the split was
    // handing back everything it won. BONSAI_NO_FUSED_SK falls back to the two-dispatch path.
    cl_kernel k_gemv_sk_fused_ = nullptr;
    cl_mem sk_flags_ = nullptr;    // [ng] arrival counters, self-resetting
    bool use_fused_sk_ = true;
    bool cl2_ok_ = true;   // device accepted -cl-std=CL2.0 (needed for device-scope atomics)
    size_t img_max_texels_ = 0;
    size_t weight_bytes_ = 0;
    int img_made_ = 0, img_skipped_ = 0;
    bool use_img_ = true;
    cl_mem make_bits_image(cl_mem bits, size_t texels);

    /**
     * Which prefill path is faster ON THIS DEVICE, decided by running both once at load.
     *
     * This exists because I could not answer the question any other way. The 27B needs ~4.8 GB and
     * will not load on the only device I can attach to, and the small part I do have has 1 compute
     * unit — which INVERTS the ranking, because split-K exists to fill 12 CUs and on 1 CU it is a
     * 3.3x pessimisation. Guessing from it shipped GEMM_MT=16 and made prefill 2x slower. A 60 ms
     * measurement on the real GPU is worth more than any amount of extrapolation.
     */
    bool prefill_batched_ = false;
    void tune_prefill_path();
    /** The real A/B, run only from /bench — it costs ~17 s and must never be a startup cost. */
  public:
  private:
    cl_kernel k_gemm_ = nullptr, k_gemm_img_ = nullptr, k_xsum_m_ = nullptr;
    // One row per work-item: the 4-row tile spilled 2944 B/wi on an Adreno 840.
    cl_kernel k_gemm_r1_ = nullptr;
    // LUT path: precomputed partial sums replace the per-weight sign unpack.
    std::string kopts_base_;   // kernel build flags minus -DGEMM_MT (for the MT sweep)
    cl_kernel k_xsum_, k_gemv_, k_gather_, k_argmax_, k_rms_, k_add_, k_swiglu_,
              k_scores_, k_softmax_, k_attnout_,
              k_dn_step_, k_dn_norm_, k_dn_conv_,
              k_split_qg_, k_rope_p_, k_apply_gate_;
};
