#pragma once
// Auto-generated debug utilities for crash diagnosis.
// Included by scaffold-generated files (main.cpp, weights.cpp).

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cstdint>
#include <signal.h>
#include <unistd.h>
#include <sys/stat.h>
#include <vector>
#include <cmath>
#include <string>
#include <unordered_map>
#include <unordered_set>

#ifdef __ANDROID__
#include <unwind.h>
#include <dlfcn.h>
#else
#include <execinfo.h>
#endif

// Forward-declare the only OpenCL type we need so this header stays standalone.
// The actual <CL/cl.h> include comes from the translation unit.
// Ensure CL types are available for debug macros.
// For cross-compilation (Android NDK), CL headers are available at compile time
// even though the CL runtime is only on-device. If CL_SUCCESS is not defined,
// provide forward declarations so NNOPT_LAYER_CHECK always works.
#ifndef CL_SUCCESS
#include <CL/cl.h>
#endif

// ──────────────────────────────────────────────
// Build mode: NNOPT_DEBUG is defined by CMake for debug builds.
// Debug: full checkpoints, layer checks, GPU memory tracking.
// Release: errors still log (critical), everything else is no-op.
// ──────────────────────────────────────────────

// Basename helper and error macros are ALWAYS available (even in release).
// Strip path to basename for readability on device.
static inline const char* nnopt_basename(const char* path) {
    const char* last = path;
    for (const char* p = path; *p; p++) { if (*p == '/' || *p == '\\\\') last = p + 1; }
    return last;
}

// ──────────────────────────────────────────────
// Error logging — ALWAYS active (even in release). Errors are critical.
// ──────────────────────────────────────────────
#define NNOPT_ERROR(msg) do { \
    fprintf(stderr, "ERROR: %s (%s:%d)\\n", (msg), nnopt_basename(__FILE__), __LINE__); \
    fflush(stderr); \
} while(0)

#define NNOPT_ERROR_FMT(fmt, ...) do { \
    fprintf(stderr, "ERROR: " fmt " (%s:%d)\\n", ##__VA_ARGS__, nnopt_basename(__FILE__), __LINE__); \
    fflush(stderr); \
} while(0)


// NNOPT_TODO: scaffold-emitted stubs use this to mark unwired code paths.
// Logs once to stderr and continues — the stub's default body must produce
// correct (possibly slow) output. FinalizePort surfaces the unwired state via
// a "decode_tok_s ≤ prefill_tok_s" warning and the stderr line below appears
// in Infer logs. We deliberately do NOT abort: that forces the agent to wire
// optimizations before basic correctness is verifiable, which thrashes ports.
#define NNOPT_TODO(what) do { \
    static bool _nnopt_todo_logged = false; \
    if (!_nnopt_todo_logged) { \
        fprintf(stderr, "NNOPT_TODO: %s is not yet wired — using slow default path (%s:%d)\\n", \
                (what), nnopt_basename(__FILE__), __LINE__); \
        fflush(stderr); \
        _nnopt_todo_logged = true; \
    } \
} while(0)

// ──────────────────────────────────────────────
// GPU memory counters — ALWAYS declared (crash handler reads them in
// both debug and release). Only the debug-mode tracked_clCreateBuffer
// increments them; in release they stay at 0, which is the correct
// "unknown" value for the crash report.
// ──────────────────────────────────────────────
static size_t g_gpu_bytes_allocated = 0;
static int    g_gpu_alloc_count     = 0;

// ──────────────────────────────────────────────
// Sync discipline (cppStandards rule SYNC-01).
//
// The OpenCL command queue created by OpenCLContext is in-order
// (CL_QUEUE_PROFILING_ENABLE only — no out-of-order flag), so kernel-to-
// kernel data dependencies on the same queue are preserved without any
// explicit sync. Tag every clFinish that exists ONLY for debug visibility
// (around LAYER_CHECK calls, between layer kernels for error attribution)
// with NNOPT_DEBUG_SYNC(queue) — it compiles to a no-op in release.
//
// Use bare clFinish ONLY when:
//   - immediately before a host read of device data (and even then,
//     prefer blocking clEnqueueReadBuffer with CL_TRUE — implicit sync)
//   - inside one-time init (program builds, weight uploads)
//   - cross-queue / cross-context coordination
//
// Why: per-token clFinish residue measured to cost 4× decode throughput on
// SmolLM2-135M; stripping these in release mode is automatic with the macro.
#ifdef NNOPT_DEBUG
  #define NNOPT_DEBUG_SYNC(queue) clFinish(queue)
#else
  #define NNOPT_DEBUG_SYNC(queue) ((void)0)
#endif

#ifdef NNOPT_DEBUG
// ══════════════════════════════════════════════
// DEBUG MODE: full checkpoints, layer tracking, numerical validation
// ══════════════════════════════════════════════

static char g_last_checkpoint[512] = {0};
static int  g_checkpoint_count     = 0;

#define NNOPT_CHECKPOINT(msg) do { \
    snprintf(g_last_checkpoint, sizeof(g_last_checkpoint), "%s", (msg)); \
    fprintf(stderr, "CHECKPOINT[%d]: %s (%s:%d)\\n", ++g_checkpoint_count, (msg), nnopt_basename(__FILE__), __LINE__); \
    fflush(stderr); \
} while(0)

#define NNOPT_CHECKPOINT_FMT(fmt, ...) do { \
    snprintf(g_last_checkpoint, sizeof(g_last_checkpoint), fmt, __VA_ARGS__); \
    fprintf(stderr, "CHECKPOINT[%d]: " fmt " (%s:%d)\\n", ++g_checkpoint_count, __VA_ARGS__, nnopt_basename(__FILE__), __LINE__); \
    fflush(stderr); \
} while(0)

// Per-layer diagnostic logging
static char g_last_layer_op[512] = {0};

#define NNOPT_LAYER_INIT(name) do { \
    snprintf(g_last_layer_op, sizeof(g_last_layer_op), "INIT: %s", (name)); \
    fprintf(stderr, "LAYER_INIT: %s (%s:%d)\\n", (name), nnopt_basename(__FILE__), __LINE__); fflush(stderr); \
} while(0)

#define NNOPT_LAYER_INIT_FMT(fmt, idx) do { \
    char _li_name[256]; snprintf(_li_name, sizeof(_li_name), fmt, idx); \
    snprintf(g_last_layer_op, sizeof(g_last_layer_op), "INIT: %s", _li_name); \
    fprintf(stderr, "LAYER_INIT: %s (%s:%d)\\n", _li_name, nnopt_basename(__FILE__), __LINE__); fflush(stderr); \
} while(0)

#define NNOPT_LAYER_WEIGHTS(name) do { \
    snprintf(g_last_layer_op, sizeof(g_last_layer_op), "WEIGHTS: %s", (name)); \
    fprintf(stderr, "LAYER_WEIGHTS: %s (%s:%d)\\n", (name), nnopt_basename(__FILE__), __LINE__); fflush(stderr); \
} while(0)

#define NNOPT_LAYER_FWD(name) do { \
    snprintf(g_last_layer_op, sizeof(g_last_layer_op), "FWD: %s", (name)); \
    fprintf(stderr, "LAYER_FWD: %s (%s:%d)\\n", (name), nnopt_basename(__FILE__), __LINE__); fflush(stderr); \
} while(0)

#define NNOPT_LAYER_FWD_DONE(name) do { \
    snprintf(g_last_layer_op, sizeof(g_last_layer_op), "FWD_DONE: %s", (name)); \
    fprintf(stderr, "LAYER_FWD_DONE: %s (%s:%d)\\n", (name), nnopt_basename(__FILE__), __LINE__); fflush(stderr); \
} while(0)

// Per-layer numerical validation
#include <cmath>
#include <vector>

static inline bool nnopt_debug_layers_enabled() {
    static int cached = -1;
    if (cached < 0) {
        const char* env = getenv("NNOPT_DEBUG_LAYERS");
        cached = (env && (env[0] == '0' || env[0] == '\0')) ? 0 : 1;
    }
    return cached == 1;
}

// Forward declaration — defined after nnopt_layer_check_impl
static inline void nnopt_layer_dump_impl(
    const char* name, cl_command_queue q, cl_mem buf, size_t num_floats,
    int pass_index);

// IEEE 754 binary16 → float32 decoder for the layer-check sampler. Inlined
// here so debug_utils.h stays standalone (utils.h is not always included).
#ifdef NNOPT_USE_FP16
static inline float _nnopt_dbg_f16_to_f32(uint16_t bits) {
    uint32_t sign = ((uint32_t)(bits >> 15) & 0x1u);
    uint32_t exp  = ((uint32_t)(bits >> 10) & 0x1Fu);
    uint32_t mant = ((uint32_t)(bits      ) & 0x3FFu);
    uint32_t out_sign = sign << 31;
    uint32_t out;
    if (exp == 0) {
        if (mant == 0) { out = out_sign; }
        else {
            int e = -1; do { e++; mant <<= 1; } while ((mant & 0x400u) == 0);
            mant &= 0x3FFu; uint32_t out_exp = (uint32_t)(127 - 15 - e);
            out = out_sign | (out_exp << 23) | (mant << 13);
        }
    } else if (exp == 0x1F) { out = out_sign | 0x7F800000u | (mant << 13); }
    else { uint32_t out_exp = (uint32_t)(exp - 15 + 127);
           out = out_sign | (out_exp << 23) | (mant << 13); }
    float f; std::memcpy(&f, &out, sizeof(f)); return f;
}
#endif

static inline void nnopt_layer_check_impl(
    const char* name, cl_command_queue q, cl_mem buf, size_t num_floats)
{
    if (!nnopt_debug_layers_enabled() || !buf || num_floats == 0) return;

    // Sample 256 elements: 4 contiguous blocks of 64.
    //   Block 0: head    (offset 0)
    //   Block 1: ~33% in
    //   Block 2: ~66% in
    //   Block 3: TAIL    (last 64 elements, always)
    // The TAIL block is critical: for [seq_len, vocab] tensors the sampler
    // reads the LAST row, and for autoregressive caches the most recent
    // write lands at the tail. Pre-2026-05-03 sampling used b*stride
    // for all blocks — the last block landed at ~75% of the buffer and
    // missed sparse-NaN / late-row divergence. Confirmed regression on
    // LFM2.5: lm_head row 6 had argmax=730 (token-collapse) while the
    // LAYER_CHECK reading rows 0/2/4/5 showed valid ranges → "all OK".
    // Bytes per element follow the storage dtype.
#ifdef NNOPT_USE_FP16
    const size_t BYTES_PER = 2;
#else
    const size_t BYTES_PER = 4;
#endif
    const size_t BLOCK = 64;
    const size_t NUM_BLOCKS = 4;
    const size_t sample_total = BLOCK * NUM_BLOCKS;
    std::vector<float> samples(sample_total);

    size_t stride = (num_floats > sample_total) ? (num_floats / NUM_BLOCKS) : 0;
    size_t read_count = 0;
    for (size_t b = 0; b < NUM_BLOCKS; b++) {
        size_t offset;
        if (b == NUM_BLOCKS - 1 && num_floats > BLOCK) {
            // Tail block — always pin to the last BLOCK elements so the
            // row the sampler reads is always inspected.
            offset = num_floats - BLOCK;
        } else if (stride > 0) {
            offset = b * stride;
        } else {
            offset = b * BLOCK;
        }
        size_t count = BLOCK;
        if (offset + count > num_floats) count = (offset < num_floats) ? num_floats - offset : 0;
        if (count == 0) break;
#ifdef NNOPT_USE_FP16
        std::vector<uint16_t> raw(count);
        cl_int err = clEnqueueReadBuffer(q, buf, CL_TRUE,
            offset * BYTES_PER, count * BYTES_PER,
            raw.data(), 0, nullptr, nullptr);
        if (err != CL_SUCCESS) break;
        for (size_t i = 0; i < count; i++) {
            samples[read_count + i] = _nnopt_dbg_f16_to_f32(raw[i]);
        }
#else
        cl_int err = clEnqueueReadBuffer(q, buf, CL_TRUE,
            offset * BYTES_PER, count * BYTES_PER,
            samples.data() + read_count, 0, nullptr, nullptr);
        if (err != CL_SUCCESS) break;
#endif
        read_count += count;
    }
    if (read_count == 0) return;

    float vmin = samples[0], vmax = samples[0];
    double vsum = 0.0;
    int nan_count = 0, inf_count = 0, zero_count = 0;
    for (size_t i = 0; i < read_count; i++) {
        float v = samples[i];
        if (std::isnan(v)) { nan_count++; continue; }
        if (std::isinf(v)) { inf_count++; continue; }
        if (v == 0.0f) zero_count++;
        if (v < vmin) vmin = v;
        if (v > vmax) vmax = v;
        vsum += v;
    }
    float vmean = (read_count > (size_t)(nan_count + inf_count))
        ? (float)(vsum / (read_count - nan_count - inf_count)) : 0.0f;

    const char* status = "OK";
    if (nan_count > 0) status = "FAIL_NAN";
    else if (inf_count > 0) status = "FAIL_INF";
    else if (zero_count == (int)read_count) status = "FAIL_ZEROS";

    // Pass-index tag: per-name occurrence counter. First time this layer name
    // is seen in this process => pass=0 (prefill). Second time => pass=1
    // (first decode step). And so on. Lets Evaluate distinguish "weight load"
    // failures (zero on pass=0) from "decode-path / KV cache" failures
    // (non-zero on pass=0, zero on pass=1+).
    static std::unordered_map<std::string, int>* _pass_count = nullptr;
    if (!_pass_count) _pass_count = new std::unordered_map<std::string, int>();
    int _pass = (*_pass_count)[std::string(name)]++;

    fprintf(stderr, "LAYER_CHECK[pass=%d]: %s | n=%zu min=%.6g max=%.6g mean=%.6g nan=%d inf=%d zeros=%d | %s\n",
            _pass, name, read_count, vmin, vmax, vmean, nan_count, inf_count, zero_count, status);
    fflush(stderr);

    // Safety abort gate. When NNOPT_ABORT_ON_FAIL_LAYER=1, any FAIL_NAN /
    // FAIL_INF / FAIL_ZEROS status terminates the process BEFORE the next
    // op runs. Modality ports (TTS / VLM / ASR) iterate by implementing
    // one op at a time; the unimplemented ops upstream return zero buffers,
    // and feeding zero buffers into downstream OpenCL kernels (sample_prior,
    // flow, vocoder) has been observed to take Adreno GPU drivers down
    // hard (SIGSEGV / kernel panic, device wedged for minutes until reboot).
    // Each device wedge costs the user a USB reconnect and ~30s of inactive
    // device probes — recurrence makes the per-op iteration loop unviable.
    // With this gate the binary fails cleanly on host the moment an upstream
    // op produces obviously-broken output, and the next iteration can
    // proceed without a device wedge. Default off so the cosine-based SxS
    // path keeps working for production-ready ports.
    if (status[0] == 'F') {  // FAIL_NAN / FAIL_INF / FAIL_ZEROS
        const char* abort_env = getenv("NNOPT_ABORT_ON_FAIL_LAYER");
        if (abort_env && abort_env[0] == '1') {
            fprintf(stderr,
                "FATAL: layer '%s' status=%s — aborting before downstream ops "
                "to keep the GPU driver alive. Set NNOPT_ABORT_ON_FAIL_LAYER=0 "
                "to disable. Fix the upstream op then re-run.\n",
                name, status);
            fflush(stderr);
            std::exit(99);
        }
    }

    // If NNOPT_DUMP_LAYERS is set, also write the FULL buffer to a .bin file
    // for offline comparison with PyTorch reference captures.
    nnopt_layer_dump_impl(name, q, buf, num_floats, _pass);
}

// ──────────────────────────────────────────────
// Layer activation dump (for SxSDebug tool)
// ──────────────────────────────────────────────
// Writes the full layer output buffer to layer_dumps/<name>.bin when the
// NNOPT_DUMP_LAYERS=1 environment variable is set. Called automatically from
// nnopt_layer_check_impl — no model.cpp changes needed.
//
// Size-aware overwrite: lazy-loads dump_spec.txt (one "<name>\t<numel>\n" line
// per primary capture, written by Infer alongside the binary). For each call:
//   • If the name is known and num_floats matches the expected numel → write
//     (and lock so subsequent autoregressive smaller-tensor calls cannot clobber).
//   • If the name is known and num_floats does NOT match → emit a one-time
//     LAYER_DUMP_SHAPE_WARNING and skip the write. This is the autoregressive
//     single-token case AND the "main.cpp processes the prompt token-by-token"
//     case — in both, the smaller tensor must not corrupt the comparable dump.
//   • If the name is unknown (no dump_spec entry) → fall back to "first writer
//     wins" using an in-process set (NOT file existence; stale on-disk files
//     from prior runs must not pin the first dump).
static inline void nnopt_layer_dump_impl(
    const char* name, cl_command_queue q, cl_mem buf, size_t num_floats,
    int pass_index) {
    static int _dump_checked = 0;
    static int _dump_enabled = 0;
    static std::unordered_map<std::string, size_t>* _expected = nullptr;
    static std::unordered_set<std::string>* _written = nullptr;
    static std::unordered_set<std::string>* _warned = nullptr;
    if (!_dump_checked) {
        _dump_checked = 1;
        const char* env = getenv("NNOPT_DUMP_LAYERS");
        _dump_enabled = (env && env[0] != '0');
        if (_dump_enabled) {
#ifdef __ANDROID__
            mkdir("layer_dumps", 0755);
#else
            (void)system("mkdir -p layer_dumps");
#endif
                        _expected = new std::unordered_map<std::string, size_t>();
            _written  = new std::unordered_set<std::string>();
            _warned   = new std::unordered_set<std::string>();
            // Lazy-load dump_spec.txt from cwd.
            //
            // IMPORTANT: For non-transformer graphs (e.g. VITS TTS), the
            // reference may not emit per-module primary dump specs, so this
            // file can legitimately be empty or missing. In that case we must
            // NOT gate pass=0 writes on expected shapes — otherwise SxS gets
            // stuck in a permanent "alignment blocked" state due to a stale
            // spec.
            FILE* spec = fopen("dump_spec.txt", "r");
            if (spec) {
                char line[1024];
                while (fgets(line, sizeof(line), spec)) {
                    char nm[256]; size_t n = 0;
                    if (sscanf(line, "%255s %zu", nm, &n) == 2 && n > 0) {
                        (*_expected)[std::string(nm)] = n;
                    }
                }
                fclose(spec);
                fprintf(stderr, "LAYER_DUMP: loaded dump_spec.txt (%zu entries)\n",
                        _expected->size());
                fflush(stderr);
            } else {
                fprintf(stderr, "LAYER_DUMP: dump_spec.txt not found — falling back to first-writer-wins\n");
                fflush(stderr);
            }
        }
    }
    if (!_dump_enabled || !buf || num_floats == 0) return;

    std::string nm(name);

    // Pass-index aware write policy:
    //   pass=0 (prefill) — legacy behavior. Validate against dump_spec; if
    //     shape mismatches, emit one-time SHAPE_WARNING and skip (genuine
    //     instrumentation bug — pass=0 must be the SxS-comparable shape).
    //     Write to BOTH "<name>.bin" (legacy filename, kept for SxSDebug
    //     and inferTs dump-numel checks) AND "<name>__pass0.bin" (new).
    //   pass>=1 (decode) — write to "<name>__passN.bin" only, no shape gate.
    //     Decode tensors have a different (smaller) shape by design and are
    //     compared against pass-tagged references, not the prefill spec.
        if (pass_index == 0) {
        // If dump_spec.txt is present and contains an entry for this name,
        // validate the shape for pass=0 (prefill) to prevent clobbering the
        // comparable dump with a smaller tensor.
        // If dump_spec.txt is missing/empty OR the name is absent, fall back
        // to first-writer-wins so non-transformer graphs can still dump.
        if (_expected && !_expected->empty()) {
            auto it = _expected->find(nm);
            if (it != _expected->end()) {
                size_t expected = it->second;
                if (num_floats != expected) {
                    if (_warned->find(nm) == _warned->end()) {
                        _warned->insert(nm);
                        fprintf(stderr,
                            "LAYER_DUMP_SHAPE_WARNING: %s got num_floats=%zu, dump_spec expects %zu. "
                            "Either (a) NNOPT_LAYER_CHECK at this call site is being passed only the "
                            "last-token slice (pass the full [seq_len, ...] tensor instead), OR "
                            "(b) main.cpp is processing the prompt token-by-token so seq_len=1 on "
                            "every call (run the prompt as one batched forward, then autoregress).\n",
                            name, num_floats, expected);
                        fflush(stderr);
                    }
                    return;
                }
            } else {
                if (_written->find(nm) != _written->end()) return;
            }
        } else {
            if (_written->find(nm) != _written->end()) return;
        }
    }
    // pass>=1: always write, no validation, per-pass filename below.


    // Read raw bytes in the on-device storage dtype. fp16 → 2 bytes/elem;
    // fp32 → 4 bytes/elem. SxSDebug reads the sidecar meta.json to know how
    // to decode each dump back to float32 for cosine comparison.
#ifdef NNOPT_USE_FP16
    const size_t bytes_per = 2;
    const char* dump_dtype = "float16";
#else
    const size_t bytes_per = 4;
    const char* dump_dtype = "float32";
#endif
    std::vector<unsigned char> data(num_floats * bytes_per);
    cl_int err = clEnqueueReadBuffer(q, buf, CL_TRUE, 0,
                                     num_floats * bytes_per,
                                     data.data(), 0, nullptr, nullptr);
    if (err != CL_SUCCESS) {
        fprintf(stderr, "LAYER_DUMP: %s FAILED (err=%d)\n", name, err);
        fflush(stderr);
        return;
    }

    // Filenames written per pass:
    //   pass=0 → "layer_dumps/<name>.bin" (legacy, used by existing SxSDebug
    //            and inferTs numel checks) AND "layer_dumps/<name>__pass0.bin"
    //   pass>=1 → "layer_dumps/<name>__passN.bin" only
    auto write_one = [&](const char* path) -> bool {
        FILE* f = fopen(path, "wb");
        if (!f) {
            fprintf(stderr, "LAYER_DUMP: %s FAILED (fopen %s)\n", name, path);
            fflush(stderr);
            return false;
        }
        fwrite(data.data(), 1, data.size(), f);
        fclose(f);
        char meta_path[512];
        snprintf(meta_path, sizeof(meta_path), "%s.meta.json", path);
        FILE* mf = fopen(meta_path, "w");
        if (mf) {
            fprintf(mf,
                    "{\"dtype\":\"%s\",\"num_elements\":%zu,\"bytes_per_element\":%zu,\"pass\":%d}\n",
                    dump_dtype, num_floats, bytes_per, pass_index);
            fclose(mf);
        }
        return true;
    };

    char path_legacy[512];
    char path_pass[512];
    snprintf(path_pass, sizeof(path_pass), "layer_dumps/%s__pass%d.bin", name, pass_index);
    bool ok = write_one(path_pass);
    if (pass_index == 0) {
        snprintf(path_legacy, sizeof(path_legacy), "layer_dumps/%s.bin", name);
        ok = write_one(path_legacy) && ok;
    }
    if (ok) {
        if (pass_index == 0) _written->insert(nm);
        fprintf(stderr, "LAYER_DUMP: %s pass=%d -> %s (%zu elems, %zu bytes, %s)\n",
                name, pass_index, path_pass, num_floats, data.size(), dump_dtype);
        fflush(stderr);
    }
}

#define NNOPT_LAYER_CHECK(name, queue, buf, num_floats) \
    nnopt_layer_check_impl((name), (queue), (buf), (num_floats))

#define NNOPT_LAYER_CHECK_FMT(fmt, idx, queue, buf, num_floats) do { \
    char _lc_name[256]; \
    snprintf(_lc_name, sizeof(_lc_name), fmt, idx); \
    nnopt_layer_check_impl(_lc_name, (queue), (buf), (num_floats)); \
} while(0)

// ──────────────────────────────────────────────────────────────────────────
// Input-side check — call at the TOP of every layer forward() to verify the
// buffer received from the caller matches the PyTorch reference. Pairs with
// NNOPT_LAYER_CHECK at the bottom: SxS uses the pair to bisect data-flow
// bugs (wrong caller) from math bugs (wrong kernel).
//
// Filename: layer_dumps/<name>_input.bin (parallel to <name>.bin output).
// Reference: reference/layers/<name>_input.bin (already written by
// generateReferenceTs.ts via register_forward_pre_hook).
// ──────────────────────────────────────────────────────────────────────────
#define NNOPT_LAYER_CHECK_INPUT(name, queue, buf, num_floats) do { \
    char _lci_name[256]; \
    snprintf(_lci_name, sizeof(_lci_name), "%s_input", (name)); \
    nnopt_layer_check_impl(_lci_name, (queue), (buf), (num_floats)); \
} while(0)

// Int32 input variant for token-ID buffers (the embedding case). The ref
// stores embedding's input as float32-cast token IDs (np.array(t,
// dtype=np.float32)), so we cast int32 → float32 host-side here and write
// a float32 file. SxS's readF32Bin honors the meta.json's dtype so the
// comparison is correct.
static inline void nnopt_layer_check_input_int_impl(
    const char* name, cl_command_queue q, cl_mem buf, size_t num_ints)
{
    if (!buf || num_ints == 0) return;
    static int _ds_checked = 0;
    if (!_ds_checked) {
        _ds_checked = 1;
#ifdef __ANDROID__
        mkdir("layer_dumps", 0755);
#else
        (void)system("mkdir -p layer_dumps");
#endif
    }
    std::vector<int32_t> ints(num_ints);
    cl_int err = clEnqueueReadBuffer(q, buf, CL_TRUE, 0,
                                     num_ints * sizeof(int32_t),
                                     ints.data(), 0, nullptr, nullptr);
    if (err != CL_SUCCESS) {
        fprintf(stderr, "LAYER_DUMP_INPUT_INT: %s FAILED read (err=%d)\n", name, (int)err);
        fflush(stderr);
        return;
    }
    std::vector<float> floats(num_ints);
    for (size_t i = 0; i < num_ints; i++) floats[i] = (float)ints[i];
    char path[512];
    snprintf(path, sizeof(path), "layer_dumps/%s_input.bin", name);
    FILE* f = fopen(path, "wb");
    if (!f) {
        fprintf(stderr, "LAYER_DUMP_INPUT_INT: %s FAILED open %s\n", name, path);
        fflush(stderr);
        return;
    }
    fwrite(floats.data(), sizeof(float), num_ints, f);
    fclose(f);
    char meta_path[512];
    snprintf(meta_path, sizeof(meta_path), "%s.meta.json", path);
    FILE* mf = fopen(meta_path, "w");
    if (mf) {
        fprintf(mf,
                "{\"dtype\":\"float32\",\"num_elements\":%zu,\"bytes_per_element\":4,\"source_dtype\":\"int32\"}\n",
                num_ints);
        fclose(mf);
    }
    fprintf(stderr, "LAYER_DUMP_INPUT_INT: %s -> %s (%zu ints cast to f32)\n",
            name, path, num_ints);
    fflush(stderr);
}

#define NNOPT_LAYER_CHECK_INPUT_INT(name, queue, buf, num_ints) \
    nnopt_layer_check_input_int_impl((name), (queue), (buf), (num_ints))

// ──────────────────────────────────────────────
// GPU memory tracking (debug-mode allocator wrapper — counters are declared
// at top-level above so the release-mode crash handler can still read them)
// ──────────────────────────────────────────────

#ifndef NNOPT_DEBUG_NEEDS_CL_TYPES
static inline cl_mem tracked_clCreateBuffer(
    cl_context ctx, cl_mem_flags flags, size_t size,
    void* host_ptr, cl_int* errcode_ret, const char* label)
{
    cl_int err;
    cl_mem buf = clCreateBuffer(ctx, flags, size, host_ptr, &err);
    if (errcode_ret) *errcode_ret = err;

    if (err == CL_SUCCESS && buf != nullptr) {
        g_gpu_bytes_allocated += size;
        g_gpu_alloc_count++;
        fprintf(stderr, "GPU_MEM[%d]: +%zu bytes (%zuKB) for '%s' | total=%zuMB (%d allocs)\n",
                g_gpu_alloc_count, size, size / 1024, label,
                g_gpu_bytes_allocated / (1024 * 1024), g_gpu_alloc_count);
    } else {
        fprintf(stderr, "GPU_MEM: FAILED alloc %zu bytes (%zuMB) for '%s' err=%d | total=%zuMB\n",
                size, size / (1024 * 1024), label, err,
                g_gpu_bytes_allocated / (1024 * 1024));
    }
    fflush(stderr);
    return buf;
}

// ──────────────────────────────────────────────
// CLBuffer RAII wrapper
// ──────────────────────────────────────────────
// Provides default, move, and value constructors so LLM-generated code
// can declare CLBuffer members without resorting to dummy 0-byte allocations.
class CLBuffer {
public:
    // Default constructor — no allocation, safe for member init lists.
    CLBuffer() : mem_(nullptr) {}

    // Value constructor — wraps tracked_clCreateBuffer.
    CLBuffer(cl_context ctx, cl_mem_flags flags, size_t size,
             void* host_ptr = nullptr, const char* label = "unlabeled") : mem_(nullptr) {
        if (size == 0) return; // Skip 0-byte allocations silently
        cl_int err;
        mem_ = tracked_clCreateBuffer(ctx, flags, size, host_ptr, &err, label);
        if (err != CL_SUCCESS) mem_ = nullptr;
    }

    ~CLBuffer() { release(); }

    // Move constructor
    CLBuffer(CLBuffer&& o) noexcept : mem_(o.mem_) { o.mem_ = nullptr; }

    // Move assignment
    CLBuffer& operator=(CLBuffer&& o) noexcept {
        if (this != &o) { release(); mem_ = o.mem_; o.mem_ = nullptr; }
        return *this;
    }

    // No copies — would double-release
    CLBuffer(const CLBuffer&) = delete;
    CLBuffer& operator=(const CLBuffer&) = delete;

    cl_mem get() const { return mem_; }
    explicit operator bool() const { return mem_ != nullptr; }

private:
    void release() { if (mem_) { clReleaseMemObject(mem_); mem_ = nullptr; } }
    cl_mem mem_;
};
#endif

// ──────────────────────────────────────────────
// Crash signal handler
// ──────────────────────────────────────────────
#ifdef __ANDROID__
struct NnoptBacktraceState {
    void** current;
    void** end;
};

static _Unwind_Reason_Code nnopt_unwind_cb(struct _Unwind_Context* context, void* arg) {
    NnoptBacktraceState* state = static_cast<NnoptBacktraceState*>(arg);
    uintptr_t pc = _Unwind_GetIP(context);
    if (pc) {
        if (state->current == state->end) return _URC_END_OF_STACK;
        *state->current++ = reinterpret_cast<void*>(pc);
    }
    return _URC_NO_REASON;
}

static void nnopt_print_backtrace() {
    void* buffer[64];
    NnoptBacktraceState state = {buffer, buffer + 64};
    _Unwind_Backtrace(nnopt_unwind_cb, &state);
    size_t count = state.current - buffer;
    for (size_t i = 0; i < count; i++) {
        Dl_info info;
        if (dladdr(buffer[i], &info) && info.dli_sname) {
            fprintf(stderr, "  #%zu: %s (%s+0x%tx)\n", i, info.dli_sname,
                    info.dli_fname,
                    (ptrdiff_t)((char*)buffer[i] - (char*)info.dli_saddr));
        } else {
            fprintf(stderr, "  #%zu: %p\n", i, buffer[i]);
        }
    }
}
#endif

#else // !NNOPT_DEBUG — Release mode: no-op macros
// ══════════════════════════════════════════════
// RELEASE MODE: checkpoints and layer checks are compiled out.
// Errors still log. Crash handler still active.
// ══════════════════════════════════════════════
static char g_last_checkpoint[512] = {0};
static char g_last_layer_op[512] = {0};

#define NNOPT_CHECKPOINT(msg) ((void)0)
#define NNOPT_CHECKPOINT_FMT(fmt, ...) ((void)0)
#define NNOPT_LAYER_INIT(name) ((void)0)
#define NNOPT_LAYER_INIT_FMT(fmt, idx) ((void)0)
#define NNOPT_LAYER_WEIGHTS(name) ((void)0)
#define NNOPT_LAYER_FWD(name) ((void)0)
#define NNOPT_LAYER_FWD_DONE(name) ((void)0)
#define NNOPT_LAYER_CHECK(name, queue, buf, num_floats) ((void)0)
#define NNOPT_LAYER_CHECK_FMT(fmt, idx, queue, buf, num_floats) ((void)0)
#define NNOPT_LAYER_CHECK_INPUT(name, queue, buf, num_floats) ((void)0)
#define NNOPT_LAYER_CHECK_INPUT_INT(name, queue, buf, num_ints) ((void)0)

static inline cl_mem tracked_clCreateBuffer(cl_context ctx, cl_mem_flags flags, size_t size,
                                             void* host_ptr, cl_int* err, const char* /*label*/) {
    return clCreateBuffer(ctx, flags, size, host_ptr, err);
}

// Stub for crash handler (which is always active, even in release)
#ifdef __ANDROID__
static inline void nnopt_print_backtrace() {}
#endif

#endif // NNOPT_DEBUG

// ──────────────────────────────────────────────
// Crash handler — ALWAYS active (catches segfaults in both debug and release)
// ──────────────────────────────────────────────
static volatile sig_atomic_t g_crash_in_progress = 0;

static void nnopt_crash_handler(int sig) {
    if (g_crash_in_progress) _exit(128 + sig);
    g_crash_in_progress = 1;

    const char* sig_name = (sig == SIGSEGV) ? "SIGSEGV" :
                           (sig == SIGABRT) ? "SIGABRT" :
                           (sig == SIGBUS)  ? "SIGBUS"  : "UNKNOWN";

    fprintf(stderr, "\n===CRASH_REPORT===\n");
    fprintf(stderr, "Signal: %s (%d)\n", sig_name, sig);
    fprintf(stderr, "Last checkpoint: %s\n",
            g_last_checkpoint[0] ? g_last_checkpoint : "(none)");
    fprintf(stderr, "Last layer op: %s\n",
            g_last_layer_op[0] ? g_last_layer_op : "(none)");
    fprintf(stderr, "GPU memory at crash: %zuMB (%d allocations, %zu bytes)\n",
            g_gpu_bytes_allocated / (1024 * 1024), g_gpu_alloc_count,
            g_gpu_bytes_allocated);
    fprintf(stderr, "Backtrace:\n");

#ifdef __ANDROID__
    nnopt_print_backtrace();
#else
    void* frames[64];
    int n = backtrace(frames, 64);
    backtrace_symbols_fd(frames, n, fileno(stderr));
#endif

    fprintf(stderr, "===END_CRASH_REPORT===\n");
    fflush(stderr);
    _exit(128 + sig);
}

static inline void nnopt_install_crash_handler() {
    struct sigaction sa;
    memset(&sa, 0, sizeof(sa));
    sa.sa_handler = nnopt_crash_handler;
    sigemptyset(&sa.sa_mask);
    sa.sa_flags = SA_RESETHAND;

    sigaction(SIGSEGV, &sa, nullptr);
    sigaction(SIGABRT, &sa, nullptr);
    sigaction(SIGBUS,  &sa, nullptr);
}
