// Auto-generated graph-mode main.cpp for facebook/musicgen-small.
// Modality: input=text, output=audio
//
// In graph mode main.cpp is small: tokenize/load input, instantiate Model,
// loop a generate step (or single forward for non-autoregressive), write
// output. The interesting code lives in src/ops/*.cpp.
//
// API REFERENCE (DO NOT GUESS — these are the canonical signatures the
// scaffold emits; matching them avoids the iteration tax of inventing
// non-existent methods like load_from_disk / decode_one / sample_argmax):
//
//   class Weights {
//     Weights();                                 // default ctor only — no OpenCLContext arg
//     bool load(const std::string& bin_path,
//               const std::string& meta_path,
//               cl_context ctx);
//     cl_mem get_buffer(const std::string& key, bool optional=false);
//     // --- Introspection (use these when CLBlast returns kInsufficientMemoryB
//     //     or any dim-mismatch error: compare your op's expected dim against
//     //     the actual on-disk shape BEFORE editing model_config.h or kernels):
//     bool has_tensor(const std::string& key) const;
//     std::vector<int> get_shape(const std::string& key) const;       // [] if missing
//     size_t get_num_elements(const std::string& key) const;          // 0 if missing
//     size_t get_size_bytes(const std::string& key) const;            // 0 if missing
//     std::string get_dtype(const std::string& key) const;            // "" if missing
//   };
//   class Tokenizer {
//     bool load(const std::string& vocab_path); // singular path, NOT (bin, meta)
//     std::vector<int> encode(const std::string& text);
//     std::string decode(const std::vector<int32_t>& ids); // takes vector, NOT single int
//     int eos_token_id() const;
//   };
//   class Sampler {                              // class-based, NOT a free function
//     Sampler(const SamplerConfig& cfg);
//     int sample(std::vector<float>& logits,
//                const std::vector<int32_t>& generated_ids) const;
//   };

#include "model.h"
#include "text_encoder.h"   // t5_encode_host — prewarm overlap thread
#include "model_config.h"
#include "encodec_host.h"
#include "write_wav.h"
#include "opencl_context.h"
#include "weights.h"
#include "sampler.h"
#include "tokenizer.h"
#include "debug_utils.h"
#include "version.h"
#include "benchmark.h"  // BenchmarkTimer + NNOPT_BENCH_FIRST_TOKEN — see prefill/decode call sites below
#include "profiler.h"   // KernelProfiler::dump_summary — dormant unless NNOPT_PROFILE=1.




#include <fstream>
#include <iostream>
#include <string>
#include <vector>
#include <cstdint>
#include <cstring>
#include <cstdlib>
#include <dlfcn.h>
#include <unistd.h>
#include <thread>
#include <deque>
#include <mutex>
#include <condition_variable>
#include <memory>
#include <chrono>

// Read prompt input ids from a binary file (int32 little-endian) when the
// caller passes \`--token-ids /path/to/test_input_ids.bin\`. This path is
// the deterministic-evaluation contract — Evaluate compares the C++ output
// against the PyTorch reference for the SAME input ids, so the binary
// MUST consume the file rather than re-tokenizing the prompt string (which
// can produce different ids if the encoder isn't perfectly aligned with
// HF's tokenizer for this model family).
static bool read_input_ids_bin(const std::string& path, std::vector<int32_t>& out) {
    std::ifstream f(path, std::ios::binary);
    if (!f) return false;
    f.seekg(0, std::ios::end);
    const std::streamsize n_bytes = f.tellg();
    f.seekg(0, std::ios::beg);
    if (n_bytes <= 0 || (n_bytes % sizeof(int32_t)) != 0) return false;
    out.resize(static_cast<size_t>(n_bytes) / sizeof(int32_t));
    f.read(reinterpret_cast<char*>(out.data()), n_bytes);
    return f.good() || f.eof();
}

// Multi-token feasibility probe (defined in ops/MegaDecoderLayer.cpp).
extern "C" void mega_mrows_probe(OpenCLContext&, cl_command_queue);
extern "C" void mega_int4_probe(OpenCLContext&, cl_command_queue);

int main(int argc, char** argv) {
    // Arm the crash handler FIRST — on SIGSEGV/SIGABRT/SIGBUS it prints the
    // last NNOPT_CHECKPOINT, a backtrace, and the GPU-mem allocation log so a
    // device segfault reports WHERE it died instead of a bare "Segmentation
    // fault". Parity with main.cpp.tmpl:44 (graph-mode main dropped this in the
    // Phase 3 redesign, blinding every graph-mode port to crash locations).
    nnopt_install_crash_handler();
    // (No version banner — debug_utils does not define one, and emitting an
    // undefined macro here was breaking every fresh-port build.)
        // Argument parsing:
    //   --text "<prompt>"        (preferred)
    //   --token-ids <bin>         (deterministic fixture; bypasses tokenizer)
    //   --out-wav <path>          (output wav path)
    //   <max_new_tokens>          (positional, optional)
    std::string prompt = "Hello, my name is";
    int max_new_tokens = 64;
    std::string token_ids_path;
    std::string out_wav_path = "output.wav";
    float guidance_scale = 3.0f;                 // generation_config.guidance_scale (CFG)
    int cfg_steps_cli = 25;                      // --cfg-steps N: CFG-early switch point
                                                 // (25 = default; user-ear-validated 2026-06-13:
                                                 // ab_cfg25 == ab_cfg50 to the ear, −1.1 s/clip.
                                                 // 50 was the prior default; -1 = full CFG every step)
    std::string force_grid_path;                 // teacher-forcing: feed this grid, log argmax
    bool serve_mode = false;                     // --serve: persistent process, one
                                                 // generation per stdin line (the
                                                 // kept-allocations work makes gen 2+
                                                 // skip the driver-pool fill + weight
                                                 // upload: TTFT ~1.5 s instead of ~7.5)


    SamplerConfig sampler_config;
    sampler_config.temperature = 1.0f;          // MusicGen generation_config: do_sample=True
    sampler_config.top_k = 250;                  // top_k=250 (greedy collapses into token runs)
    sampler_config.top_p = 1.0f;
    sampler_config.repetition_penalty = 1.0f;
    sampler_config.seed = 42u;
    {
        int positional = 0;
        for (int i = 1; i < argc; i++) {
            const std::string a = argv[i];
                        if (a == "--token-ids" && i + 1 < argc) {
                token_ids_path = argv[++i];
            } else if (a == "--text" && i + 1 < argc) {
                prompt = argv[++i];
            } else if (a == "--out-wav" && i + 1 < argc) {
                out_wav_path = argv[++i];
            } else if (a == "--serve") {
                serve_mode = true;
            } else if (a == "--force-grid" && i + 1 < argc) {
                force_grid_path = argv[++i];
            } else if (a == "--guidance-scale" && i + 1 < argc) {
                guidance_scale = std::stof(argv[++i]);
            } else if (a == "--cfg-steps" && i + 1 < argc) {
                // CFG-early: full guidance for the first N steps, then single-
                // row (2.2× faster steps). -1 = guidance on every step.
                cfg_steps_cli = std::stoi(argv[++i]);
            } else if (a == "--temperature" && i + 1 < argc) {

                sampler_config.temperature = std::stof(argv[++i]);
            } else if (a == "--top-k" && i + 1 < argc) {
                sampler_config.top_k = std::stoi(argv[++i]);
            } else if (a == "--top-p" && i + 1 < argc) {
                sampler_config.top_p = std::stof(argv[++i]);
            } else if (a == "--seed" && i + 1 < argc) {
                sampler_config.seed = static_cast<uint32_t>(std::stoul(argv[++i]));
                        } else if (a.rfind("--", 0) == 0) {
                // Unknown flag — skip the value too if present.
                if (i + 1 < argc && argv[i + 1][0] != '-') ++i;
            } else if (positional == 0) {
                // Positional arg: max_new_tokens. Be defensive: when the runner injects
                // flags like "--temperature 0" some shells may pass an empty string or
                // other non-numeric token; don't crash the whole process.
                try {
                    max_new_tokens = std::stoi(a);
                    positional++;
                } catch (const std::exception&) {
                    // Non-numeric positional = the prompt text (run_android.sh
                    // passes it this way; --token-ids still takes precedence
                    // when present, so Infer's fixture contract is unchanged).
                    prompt = a;
                }
            }


        }
    }

    // Benchmark instrumentation — emits BENCHMARK <key>: <value> lines on stderr,
    // parsed by runUtils.ts::parseInferenceMetrics. Five call sites total:
    //   1. mark_inference_start()        — here, immediately after arg parsing
    //   2. mark_prefill_start()          — immediately before the first model.forward()
    //   3. NNOPT_BENCH_FIRST_TOKEN()     — inside generate loop, right after first sampler.sample()
    //   4. mark_end()                    — after the decode loop exits
    //   5. print_summary(prompt_len, gen) — just before return 0
    // If you restructure prefill+decode (e.g. split into a single batched prefill
    // forward followed by per-token decode forwards), KEEP all five sites — they
    // partition wall-clock into ttft / prefill / decode correctly regardless of
    // loop shape. Removing any of them silently emits -1 for that metric.
    BenchmarkTimer& bench = BenchmarkTimer::instance();
    bench.mark_inference_start();
    // True end-to-end wall anchor — printed as e2e_wall_sec just before exit.
    // total_inference_sec stops at mark_end() (decode-loop exit) and HID 7-13 s
    // of EnCodec/teardown (2026-06-05: reported 32.9 s vs 46.4 s real).
    const auto e2e_t0 = std::chrono::steady_clock::now();

    OpenCLContext cl_ctx;
    if (!cl_ctx.initialize()) {
        NNOPT_ERROR("OpenCL init failed");
        return 1;
    }

    // NNOPT_RECORD_PROBE=1: validate cl_qcom_recordable_queues on this device
    // (recipe from a8nova/adreno-llms: CL_QUEUE_RECORDABLE_QCOM=0x40000000
    // alone on clCreateCommandQueue; entry points via dlsym). Prints baseline
    // vs recorded-replay dispatch cost and exits.
    if (const char* rp = std::getenv("NNOPT_RECORD_PROBE"); rp && rp[0] == '1') {
        typedef void* cl_recording_qcom;
        typedef cl_recording_qcom (CL_API_CALL *NewFn)(cl_command_queue, cl_int*);
        typedef cl_int (CL_API_CALL *EndFn)(cl_recording_qcom);
        typedef cl_int (CL_API_CALL *RelFn)(cl_recording_qcom);
        typedef cl_int (CL_API_CALL *EnqFn)(cl_command_queue, cl_recording_qcom,
            size_t, const void*, size_t, const void*, size_t, const void*,
            size_t, const void*, cl_uint, const cl_event*, cl_event*);
        auto fnNew = (NewFn)dlsym(RTLD_DEFAULT, "clNewRecordingQCOM");
        auto fnEnd = (EndFn)dlsym(RTLD_DEFAULT, "clEndRecordingQCOM");
        auto fnRel = (RelFn)dlsym(RTLD_DEFAULT, "clReleaseRecordingQCOM");
        auto fnEnq = (EnqFn)dlsym(RTLD_DEFAULT, "clEnqueueRecordingQCOM");
        fprintf(stderr, "RECORD_PROBE: entry points %s\n",
                (fnNew && fnEnd && fnRel && fnEnq) ? "LOADED" : "MISSING");
        if (!(fnNew && fnEnd && fnRel && fnEnq)) return 1;
        cl_int perr = CL_SUCCESS;
        cl_program prog = cl_ctx.build_program_from_file("kernels/record_probe.cl");
        if (!prog) return 1;
        cl_kernel k = clCreateKernel(prog, "probe_noop", &perr);
        cl_mem buf = clCreateBuffer(cl_ctx.context(), CL_MEM_READ_WRITE, 64, nullptr, &perr);
        clSetKernelArg(k, 0, sizeof(cl_mem), &buf);
        const size_t g1 = 64, l1 = 64;
        // Baseline: N sequential dispatches on the normal queue.
        const int N = 10000;
        auto t0 = std::chrono::steady_clock::now();
        for (int i = 0; i < N; ++i)
            clEnqueueNDRangeKernel(cl_ctx.queue(), k, 1, nullptr, &g1, &l1, 0, nullptr, nullptr);
        clFinish(cl_ctx.queue());
        double base_ms = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count();
        // Recordable queue: bit 30 alone.
        cl_command_queue rq = clCreateCommandQueue(cl_ctx.context(), cl_ctx.device(),
                                                   (cl_command_queue_properties)0x40000000, &perr);
        fprintf(stderr, "RECORD_PROBE: recordable queue err=%d q=%p\n", (int)perr, (void*)rq);
        if (perr != CL_SUCCESS || !rq) return 1;
        cl_int e2 = CL_SUCCESS;
        cl_recording_qcom rec = fnNew(rq, &e2);
        fprintf(stderr, "RECORD_PROBE: clNewRecordingQCOM err=%d handle=%p\n", (int)e2, rec);
        if (e2 != CL_SUCCESS || !rec) return 1;
        for (int i = 0; i < 30; ++i)   // record a 30-dispatch step-like sequence
            clEnqueueNDRangeKernel(rq, k, 1, nullptr, &g1, &l1, 0, nullptr, nullptr);
        fnEnd(rec);
        auto t1 = std::chrono::steady_clock::now();
        for (int i = 0; i < N / 30; ++i)
            fnEnq(rq, rec, 0, nullptr, 0, nullptr, 0, nullptr, 0, nullptr, 0, nullptr, nullptr);
        clFinish(rq);
        double rep_ms = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t1).count();
        fprintf(stderr, "RECORD_PROBE: baseline %d dispatches = %.2f ms (%.2f us/dispatch)\n",
                N, base_ms, base_ms * 1000.0 / N);
        fprintf(stderr, "RECORD_PROBE: replay   %d dispatches = %.2f ms (%.2f us/dispatch) — %.2fx\n",
                (N / 30) * 30, rep_ms, rep_ms * 1000.0 / ((N / 30) * 30), base_ms / rep_ms);
        // Which queues accept a replay? The integration replays on the LIVE
        // queue (profiling-enabled) and got -59 — disambiguate driver rules.
        {
            cl_int e_live = fnEnq(cl_ctx.queue(), rec, 0, nullptr, 0, nullptr, 0, nullptr, 0, nullptr, 0, nullptr, nullptr);
            clFinish(cl_ctx.queue());
            cl_command_queue pq = clCreateCommandQueue(cl_ctx.context(), cl_ctx.device(), 0, &perr);
            cl_int e_plain = pq ? fnEnq(pq, rec, 0, nullptr, 0, nullptr, 0, nullptr, 0, nullptr, 0, nullptr, nullptr) : -999;
            if (pq) clFinish(pq);
            // arg-override on the recordable queue (the integration's other need:
            // 30 instances of the SAME kernel in this recording — does a single
            // (kernel,arg) override apply to all of them without error?)
            int v = 0;
            cl_array_arg_qcom upd[1] = {{k, 0, sizeof(cl_mem), &buf}};
            (void)v;
            cl_int e_args = fnEnq(rq, rec, 1, upd, 0, nullptr, 0, nullptr, 0, nullptr, 0, nullptr, nullptr);
            clFinish(rq);
            fprintf(stderr, "RECORD_PROBE: replay on live(profiling) q err=%d | plain q err=%d | rq+buf-override err=%d\n",
                    (int)e_live, (int)e_plain, (int)e_args);
            // SCALAR-override semantics: record probe_scalar ×3 (baked v=111),
            // replay with arg1 override v=222, read back slots 1..3.
            cl_kernel ks = clCreateKernel(prog, "probe_scalar", &perr);
            cl_mem sbuf = clCreateBuffer(cl_ctx.context(), CL_MEM_READ_WRITE, 64, nullptr, &perr);
            int zero[8] = {};
            clEnqueueWriteBuffer(cl_ctx.queue(), sbuf, CL_TRUE, 0, sizeof(zero), zero, 0, nullptr, nullptr);
            int v_bake = 111;
            clSetKernelArg(ks, 0, sizeof(cl_mem), &sbuf);
            clSetKernelArg(ks, 1, sizeof(int), &v_bake);
            cl_recording_qcom rec2 = fnNew(rq, &e2);
            const size_t g3 = 64, l3 = 64;
            // 3 instances of the SAME kernel, same args (like 24 layer dispatches)
            for (int i = 0; i < 3; ++i)
                clEnqueueNDRangeKernel(rq, ks, 1, nullptr, &g3, &l3, 0, nullptr, nullptr);
            fnEnd(rec2);
            int v_new = 222;
            cl_array_arg_qcom upd2[1] = {{ks, 1, sizeof(int), &v_new}};
            cl_int e_s = fnEnq(rq, rec2, 1, upd2, 0, nullptr, 0, nullptr, 0, nullptr, 0, nullptr, nullptr);
            clFinish(rq);
            int got[8] = {};
            clEnqueueReadBuffer(cl_ctx.queue(), sbuf, CL_TRUE, 0, sizeof(got), got, 0, nullptr, nullptr);
            fprintf(stderr, "RECORD_PROBE: scalar-override err=%d slots=[%d %d %d] (222,222,222 = applies to ALL instances)\n",
                    (int)e_s, got[1], got[2], got[3]);
            // replay rec2 on the LIVE queue WITH the override (the integration's
            // exact combination).
            v_new = 333;
            cl_int e_s_live = fnEnq(cl_ctx.queue(), rec2, 1, upd2, 0, nullptr, 0, nullptr, 0, nullptr, 0, nullptr, nullptr);
            clFinish(cl_ctx.queue());
            clEnqueueReadBuffer(cl_ctx.queue(), sbuf, CL_TRUE, 0, sizeof(got), got, 0, nullptr, nullptr);
            fprintf(stderr, "RECORD_PROBE: scalar-override on LIVE q err=%d slots=[%d %d %d]\n",
                    (int)e_s_live, got[1], got[2], got[3]);
            fnRel(rec2);
        }
        fnRel(rec);
        return 0;
    }

    // NNOPT_DOT8_PROBE=1: run the qcom_dot8_acc semantics probe and exit.
    if (const char* dp = std::getenv("NNOPT_DOT8_PROBE"); dp && dp[0] == '1') {
        cl_program prog = cl_ctx.build_program_from_file("kernels/dot8_probe.cl");
        if (!prog) { fprintf(stderr, "DOT8_PROBE: build failed\n"); return 1; }
        cl_int perr = CL_SUCCESS;
        cl_kernel k = clCreateKernel(prog, "dot8_probe", &perr);
        cl_mem buf = clCreateBuffer(cl_ctx.context(), CL_MEM_READ_WRITE, 8 * sizeof(int), nullptr, &perr);
        clSetKernelArg(k, 0, sizeof(cl_mem), &buf);
        size_t g1 = 1;
        clEnqueueNDRangeKernel(cl_ctx.queue(), k, 1, nullptr, &g1, nullptr, 0, nullptr, nullptr);
        int res[8] = {};
        clEnqueueReadBuffer(cl_ctx.queue(), buf, CL_TRUE, 0, sizeof(res), res, 0, nullptr, nullptr);
        for (int i = 0; i < 8; ++i) fprintf(stderr, "DOT8_PROBE[%d] = %d\n", i, res[i]);
        return 0;
    }

    // NNOPT_MROWS_PROBE=1: multi-token feasibility — does the tex GEMV amortize
    // weight loads across rows (load-bound → speculative decode wins) or scale
    // linearly (MAD-bound → abort)? Runs mega_proj at M=1/2/4/8 and exits.
    if (const char* mp = std::getenv("NNOPT_MROWS_PROBE"); mp && mp[0] == '1') {
        mega_mrows_probe(cl_ctx, cl_ctx.queue());
        return 0;
    }
    if (const char* i4 = std::getenv("NNOPT_INT4_PROBE"); i4 && i4[0] == '1') {
        mega_int4_probe(cl_ctx, cl_ctx.queue());
        return 0;
    }

    // Tokenizer first — its eos id feeds into sampler config below.
    Tokenizer tok;
    const bool tokenizer_ok = tok.load("weights/tokenizer_vocab.bin");
    if (!tokenizer_ok && token_ids_path.empty()) {
        NNOPT_ERROR("tokenizer load failed (use --token-ids to bypass for deterministic eval)");
        return 1;
    }
    if (tokenizer_ok) {
        sampler_config.eos_token_id = tok.eos_token_id();
    }

    // Weights — fp16 build pulls weights/model.fp16.bin; fp32 uses weights/model.bin.
    Weights weights;
#ifdef NNOPT_USE_FP16
    const char* nnopt_weights_bin  = "weights/model.fp16.bin";
    const char* nnopt_weights_meta = "weights/model.fp16.meta.json";
#else
    const char* nnopt_weights_bin  = "weights/model.bin";
    const char* nnopt_weights_meta = "weights/model.meta.json";
#endif
    if (!weights.load(nnopt_weights_bin, nnopt_weights_meta, cl_ctx.context())) {
        NNOPT_ERROR_FMT("weights load failed: %s", nnopt_weights_bin);
        return 1;
    }

    Sampler sampler(sampler_config);

    Model model(cl_ctx, weights);
    if (!model.initialize()) {
        NNOPT_ERROR("Model::initialize() failed — see prior NNOPT_ERROR for the layer that failed");
        return 1;
    }




    // ── SERVE LOOP ───────────────────────────────────────────────────────────
    // Single-shot mode runs this once. --serve loops: after each generation the
    // process prints SERVE_READY and reads the next prompt line from stdin
    // (empty line or EOF exits). All weights/KV/cross-KV/scratch/programs stay
    // resident across iterations (value-invalidation resets).
    std::vector<int32_t> generated_ids;          // last gen's codebook-0 stream
    size_t prompt_len = 0;
    int serve_gen = 0;
    while (true) {
    generated_ids.clear();
    const auto gen_t0 = std::chrono::steady_clock::now();
    // Prefer test_input_ids.bin (deterministic) over re-encoding the prompt.
    // The reference's "generated_text" field is computed from these EXACT
    // ids in PyTorch — comparing against a re-encoded prompt would mask
    // tokenizer drift as model error.
    std::vector<int32_t> prompt_ids;
    if (!token_ids_path.empty() && read_input_ids_bin(token_ids_path, prompt_ids)) {
        // OK — got authoritative ids from disk.
    } else {
        const std::vector<int> encoded = tok.encode(prompt);
        prompt_ids.assign(encoded.begin(), encoded.end());
        // Always log on-device-tokenized ids (parity check vs HF host ids).
        fprintf(stderr, "TOKENIZER_IDS[%zu]:", prompt_ids.size());
        for (size_t k = 0; k < prompt_ids.size() && k < 32; ++k)
            fprintf(stderr, " %d", prompt_ids[k]);
        fprintf(stderr, "\n");
    }
        prompt_len = prompt_ids.size();

    // NOTE: MusicGen is text->audio.
    //
    // This workspace's current graph trace covers ONLY the text-encoder/decoder
    // hidden-state path (no EnCodec/audio decoder yet). We still need the
    // language-model-style generate() loop to run so Evaluate can measure
    // prefill/decode behavior and produce non-empty output.

    // --- Prefill + (optional) decode ---
    // For now: run a minimal greedy decode loop over logits returned by
    // model.forward(), printing token IDs to stdout for deterministic eval.
    // This keeps the port testable even before audio decode is traced.

    bench.mark_prefill_start();

    // ════ Real MusicGen pipeline (modeling_musicgen.py) ════════════════
    // [1] T5 encoder + enc_to_dec_proj (host, once) → cross-attn states.
    //     prompt_ids here are the TEXT token ids (T5 tokenizer / fixture).
    // TTFT overlap: the bulk decoder/lm_heads weight upload to the GPU is
    // independent of the CPU T5 encode — run it in a worker thread while T5
    // computes (serially these cost ~3.0 s + ~2.0 s; overlapped ≈ max of the
    // two). Joined before the first forward needs the buffers.
    const auto t5_t0 = std::chrono::steady_clock::now();
    // Default OFF: measured 2026-06-05 — the upload is flash-page-fault-bound
    // and CONTENDS with T5 for memory bandwidth (A/B: overlap ~6.9 s TTFT vs
    // 5.2-7.3 noisy serial; no win). Kept for devices with faster storage.
    const bool overlap_on = [](){ const char* e = std::getenv("NNOPT_T5_OVERLAP"); return e && e[0]=='1'; }();
    std::thread uploader;
    if (overlap_on) uploader = std::thread([&weights]() { weights.upload_prefix("decoder."); });
    // ── TTFT prewarm (NNOPT_PREWARM, default ON) ─────────────────────────────
    // T5 is pure host math (no CL), so it runs on a worker thread while the
    // main thread absorbs ALL first-pass GPU resource creation via a zero-state
    // dummy step (the measured 4.8 s layers-10-13 driver stall). Join, then
    // apply the real encoder states (reset + re-precompute inside).
    // Default OFF: measured 2026-06-05 — the dummy step's driver-pool fill
    // contends host T5 to ~3x its solo time (1.8 -> 6.2 s), a net TTFT LOSS
    // (11.5-12 vs 7.5 s). Kept for experiments; the useful sibling (KV/cross-KV
    // allocations surviving reset) shipped unconditionally in backbone/mega.
    static const bool prewarm_on = [](){ const char* e = std::getenv("NNOPT_PREWARM"); return e && e[0]=='1'; }();
    bool t5_ok = false;
    if (prewarm_on) {
        std::vector<float> t5_states;
        std::thread t5_thr([&]() { t5_states = t5_encode_host(weights, prompt_ids); });
        const bool warm_ok = model.prewarm_decoder((int)prompt_ids.size());
        if (!warm_ok) NNOPT_ERROR("prewarm_decoder failed — continuing without prewarm");
        t5_thr.join();
        t5_ok = model.apply_encoder_states(t5_states);
    } else {
        t5_ok = model.encode_text(prompt_ids);
    }
    if (uploader.joinable()) uploader.join();
    if (nnopt_ttft_trace_enabled()) fprintf(stderr, "TTFT_TRACE [%.0f] t5_done\n", nnopt_uptime_ms());
    fprintf(stderr, "BENCHMARK t5_encode_sec: %.3f (overlapped with weight upload)\n",
            std::chrono::duration<double>(std::chrono::steady_clock::now() - t5_t0).count());
    if (!t5_ok) {
        NNOPT_ERROR("encode_text (T5 stage) failed");
        return 1;
    }

    // [2]+[3] Delay-pattern autoregressive decode.
    // build_delay_pattern_mask (modeling_musicgen.py:813): codebook k is
    // delayed k steps; column 0 is the BOS/pad row (decoder_start). At step t
    // we feed grid[:,t], the heads predict grid[:,t+1]; positions still inside
    // a codebook's delay window stay BOS instead of sampled.
    const int NCB = MODEL_CONFIG::NUM_CODEBOOKS;             // 4
    const int VOCAB = MODEL_CONFIG::VOCAB_SIZE;              // 2048
    const int32_t BOS = (int32_t)MODEL_CONFIG::BOS_TOKEN_ID; // 2048
    // Hard length cap = decoder max_position_embeddings (also the KV-cache cap
    // MEGA_MAXK). The delay pattern adds up to NCB-1 warmup frames, so leave a
    // small margin. Beyond this the model has no position embeddings AND the KV
    // cache would overflow → distortion (the pre-2026-06-13 ">10 s" bug). Clamp
    // loudly rather than silently corrupt.
    {
        const int kMaxLen = MODEL_CONFIG::MAX_POSITION_EMBEDDINGS - NCB;  // 2048-4 = 2044
        if (max_new_tokens > kMaxLen) {
            fprintf(stderr,
                    "WARNING: max_new_tokens=%d exceeds this model's limit of %d "
                    "(%.1f s @ 50 Hz); clamping to %d. MusicGen-small was trained to ~30 s.\n",
                    max_new_tokens, kMaxLen, (double)kMaxLen / 50.0, kMaxLen);
            max_new_tokens = kMaxLen;
        }
    }
    const int steps = max_new_tokens;
    std::vector<std::vector<int32_t>> grid(
        (size_t)NCB, std::vector<int32_t>((size_t)steps + 1, BOS));
    // Teacher forcing: preload the grid with a known-good (torch) grid so every
    // step's INPUT matches the reference exactly; the per-step argmax printed
    // below then isolates decode math from sampling. File: int32 [4][steps+1].
    bool force_grid = false;
    if (!force_grid_path.empty()) {
        std::ifstream fg(force_grid_path, std::ios::binary);
        if (fg) {
            for (int k = 0; k < NCB && fg; ++k)
                fg.read(reinterpret_cast<char*>(grid[(size_t)k].data()),
                        (std::streamsize)((steps + 1) * sizeof(int32_t)));
            force_grid = (bool)fg;
        }
        if (!force_grid) { NNOPT_ERROR_FMT("--force-grid %s unreadable", force_grid_path.c_str()); return 1; }
        fprintf(stderr, "TEACHER_FORCING from %s\n", force_grid_path.c_str());
    }
    std::vector<int32_t> col((size_t)NCB, BOS);
    // generated_ids declared above the serve loop (cleared per generation)
    std::vector<int32_t> no_history;                         // sampler: no rep-penalty across codebooks
    bool decode_ok = true;

    // Stage-3 FUSED READOUT: NNOPT_FUSED_READOUT=1 routes the per-step readout
    // through the single fused_readout dispatch (final LN + lm_head GEMV + CFG
    // blend + on-GPU sampling), returning 4 sampled ids — no logits cross the
    // host bus. NNOPT_FORCE_ARGMAX=1 makes the fused sampler pure argmax so the
    // greedy guard (66/1534/1513/1801) still applies. Only valid for CFG
    // (guidance>1) + free sampling (not --force-grid teacher forcing, which
    // needs logits-path argmax printed per fed column).
    // Default ON (Stage-3 validated, +2%: kills the per-step logits readback).
    // NNOPT_FUSED_READOUT=0 reverts to the host logits + host sampler path.
    const bool fused_readout = [](){ const char* e = std::getenv("NNOPT_FUSED_READOUT"); return !(e && e[0]=='0'); }()
                               && guidance_scale > 1.0f && !force_grid;
    const int  fused_force_argmax = [](){ const char* e = std::getenv("NNOPT_FORCE_ARGMAX"); return (e && e[0]=='1') ? 1 : 0; }();
    std::vector<int32_t> sampled_ids((size_t)NCB, BOS);

    // Stage-4 GPU-RESIDENT GRID: NNOPT_GPU_GRID=1 (requires fused_readout) keeps
    // the delay grid on the GPU. embed_prologue reads ids from grid[:,t] and
    // sample_grid writes grid[:,t+1] with delay forcing — the host uploads NO
    // per-step ids and the col read/write below is skipped. The whole grid is
    // read back ONCE after the loop. The per-step 16 B id readback (for the
    // STEP_ARGMAX log + cb0 display) remains but does not gate the dispatch
    // pipeline; it can be dropped, but is kept so the guard/Evaluate still see
    // a token stream without an extra end-of-loop reconstruction pass.
    // PIPELINE mode (default ON): GPU-resident grid + zero per-step readback —
    // the host enqueues the entire decode ahead; ids surface in the single
    // end-of-loop grid readback. NNOPT_PIPELINE=0 reverts to per-step reads.
    const bool pipeline_mode = [](){ const char* e = std::getenv("NNOPT_PIPELINE"); return !(e && e[0]=='0'); }();
    const bool gpu_grid = fused_readout
        && (pipeline_mode || [](){ const char* e = std::getenv("NNOPT_GPU_GRID"); return e && e[0]=='1'; }());
    void* grid_dev = nullptr;
    std::vector<int32_t> flat_grid;   // contiguous [NCB*(steps+1)] mirror for the GPU
    const int steps1 = steps + 1;
    if (gpu_grid) {
        flat_grid.assign((size_t)NCB * steps1, BOS);   // col0 + all delay cells = BOS
        grid_dev = model.gpu_grid_alloc(flat_grid.data(), NCB, steps1, BOS);
        if (!grid_dev) { NNOPT_ERROR("Stage4: gpu_grid_alloc failed"); decode_ok = false; }
    }

    // ── STREAMING ENCODEC (NNOPT_ENC_STREAM=1 opt-in; default OFF) ─────────
    // DEFAULT FLIPPED 2026-06-05 on PROCESS-WALL measurement: the CPU SEANet
    // worker looked free ("EnCodec wall disappears into the decode window",
    // tail 1.7 s) but `time` told the truth — 46.4 s real vs 38.2 s with the
    // GPU EnCodec running after decode. Two mechanisms: (a) the worker's
    // memory traffic robs the GPU GEMVs of ~9% decode throughput (9.87 vs
    // 10.88 tok/s); (b) the chunked path RECOMPUTES conv0 over all frames each
    // push (~158 CPU-seconds total vs ~8 for the whole GPU path). TOOL LESSON:
    // gate overlap optimizations on process wall-clock, never on an internal
    // benchmark that stops before teardown.
    const bool enc_stream_live = gpu_grid && pipeline_mode &&
        [](){ const char* e = std::getenv("NNOPT_ENC_STREAM"); return e && e[0]=='1'; }();
    const int stream_frames = steps - (NCB - 1);
    std::unique_ptr<EncodecStream> enc_stream;
    std::vector<std::vector<int32_t>> codes_live;
    std::thread enc_worker;
    std::mutex enc_mx;
    std::condition_variable enc_cv;
    std::deque<std::pair<void*, int>> enc_q;   // (cl_event, steps_done)
    bool enc_done_flag = false;
    bool enc_stream_bad = false;
    int enc_read_done = 1;   // grid cols [0, enc_read_done) already requested (col0 = BOS)
    if (enc_stream_live && stream_frames > 0) {
        enc_stream.reset(new EncodecStream(weights, stream_frames));
        codes_live.assign((size_t)NCB, std::vector<int32_t>((size_t)stream_frames, 0));
        enc_worker = std::thread([&]() {
            int frames_done = 0;
            while (true) {
                std::pair<void*, int> msg{nullptr, 0};
                {
                    std::unique_lock<std::mutex> lk(enc_mx);
                    enc_cv.wait(lk, [&]{ return !enc_q.empty() || enc_done_flag; });
                    if (enc_q.empty()) break;
                    msg = enc_q.front(); enc_q.pop_front();
                }
                cl_event ev = (cl_event)msg.first;
                clWaitForEvents(1, &ev);
                clReleaseEvent(ev);
                // un-delay the newly available frames; guard non-code ids.
                // frame f needs cols up to f+NCB (codebook 3's delay) — with
                // cols [0, msg.second) read, avail = msg.second - NCB.
                const int avail = std::min(stream_frames, msg.second - NCB);
                for (int f = frames_done; f < avail && !enc_stream_bad; ++f)
                    for (int k = 0; k < NCB; ++k) {
                        const int32_t id = flat_grid[(size_t)k * steps1 + f + k + 1];
                        if (id < 0 || id >= VOCAB) { enc_stream_bad = true; break; }
                        codes_live[(size_t)k][(size_t)f] = id;
                    }
                if (enc_stream_bad) break;
                if (avail > frames_done) {
                    if (!enc_stream->push(codes_live, avail, false)) { enc_stream_bad = true; break; }
                    frames_done = avail;
                }
            }
        });
    }

    const auto step0_t0 = std::chrono::steady_clock::now();
    if (nnopt_ttft_trace_enabled()) fprintf(stderr, "TTFT_TRACE [%.0f] decode_loop_enter\n", nnopt_uptime_ms());
    for (int t = 0; t < steps && decode_ok; ++t) {
        if (!gpu_grid)
            for (int k = 0; k < NCB; ++k) col[(size_t)k] = grid[(size_t)k][(size_t)t];

        // CFG-EARLY (NNOPT_CFG_STEPS=N): full CFG for the first N steps, then
        // single-row (cond only) — guidance matters most early; the uncond row
        // is never revisited so its KV bank simply stops growing. Unset = full
        // CFG every step (shipped default until the ear gate approves).
        // env NNOPT_CFG_STEPS overrides the --cfg-steps CLI value (A/B harness).
        static const int cfg_steps = [&](){ const char* e = std::getenv("NNOPT_CFG_STEPS"); return e ? std::atoi(e) : cfg_steps_cli; }();
        const float g_eff = (cfg_steps >= 0 && t >= cfg_steps) ? 1.0f : guidance_scale;

        if (gpu_grid) {
            // Pure GPU-resident step: forward reads grid[:,t], samples, writes
            // grid[:,t+1]. col is unused (embeddings source ids from the device
            // grid). sampled_ids returns the 4 ids for the log/display only.
            if (!model.forward_cfg_sampled(col, /*start_pos=*/t, g_eff,
                                           sampler_config.temperature, sampler_config.top_k,
                                           sampler_config.seed, fused_force_argmax,
                                           sampled_ids.data())) {
                NNOPT_ERROR_FMT("decode step %d: gpu-grid forward failed", t);
                decode_ok = false; break;
            }
            if (t == 0) {
                NNOPT_BENCH_FIRST_TOKEN();
                fprintf(stderr, "BENCHMARK first_step_sec: %.3f\n",
                        std::chrono::duration<double>(std::chrono::steady_clock::now() - step0_t0).count());
            }
            // streaming-EnCodec: hand a chunk of fresh grid columns to the
            // worker (non-blocking read; GPU keeps running). Cadence: 50-step
            // chunks mid-clip (amortizes the async grid read), tightening to 16
            // near the end — the tail the listener waits for after the last
            // token is then ≤16 frames + flush instead of a full 50-frame chunk
            // (measured 1.6 s tail). The worker consumes arbitrary column
            // increments, so cadence is a producer-only choice.
            const bool enc_hand_off = ((t + 1) % 50 == 0) ||
                                      (t + 1 > steps - 64 && (t + 1) % 16 == 0);
            if (enc_stream && !enc_stream_bad && enc_hand_off) {
                void* ev = nullptr;
                const int col1 = t + 2;   // cols written through step t
                if (model.gpu_grid_read_cols_async(grid_dev, flat_grid.data(), NCB, steps1,
                                                   enc_read_done, col1, &ev) && ev) {
                    enc_read_done = col1;
                    { std::lock_guard<std::mutex> lk(enc_mx); enc_q.emplace_back(ev, col1); }
                    enc_cv.notify_one();
                }
            }
            // Pipeline mode: ids are -1 sentinels (they live in the device
            // grid until the single end-of-loop readback) — nothing to print
            // per step. Non-pipeline grid mode has real ids; step logging is
            // debug-only (NNOPT_STEP_LOG=1) — these are SAMPLED ids, and the
            // guard protocol reads the host-logits path, not this one.
            static const bool step_log = [](){ const char* e = std::getenv("NNOPT_STEP_LOG"); return e && e[0]=='1'; }();
            if (sampled_ids[0] != -1) {
                if (step_log)
                    for (int k = 0; k < NCB; ++k)
                        fprintf(stderr, "STEP_SAMPLED t=%d cb=%d id=%d\n", t, k, sampled_ids[(size_t)k]);
                generated_ids.push_back(sampled_ids[0]);
            }
            continue;
        }

        if (fused_readout) {
            // One fused dispatch → 4 sampled ids (or argmax under FORCE_ARGMAX).
            if (!model.forward_cfg_sampled(col, /*start_pos=*/t, g_eff,
                                           sampler_config.temperature, sampler_config.top_k,
                                           sampler_config.seed, fused_force_argmax,
                                           sampled_ids.data())) {
                NNOPT_ERROR_FMT("decode step %d: fused readout failed", t);
                decode_ok = false; break;
            }
            if (t == 0) {
                NNOPT_BENCH_FIRST_TOKEN();
                fprintf(stderr, "BENCHMARK first_step_sec: %.3f\n",
                        std::chrono::duration<double>(std::chrono::steady_clock::now() - step0_t0).count());
            }
            if (sampled_ids[0] == -1) continue;   // pipeline mode: ids surface at end-of-loop grid read
            static const bool step_log_f = [](){ const char* e = std::getenv("NNOPT_STEP_LOG"); return e && e[0]=='1'; }();
            for (int k = 0; k < NCB; ++k) {
                // Step log is debug-only (these are SAMPLED ids; the guard
                // protocol uses the host-logits path's STEP_ARGMAX, which
                // stays unconditional there). Under FORCE_ARGMAX the fused id
                // IS the argmax — keep the historical label for that case.
                if (step_log_f)
                    fprintf(stderr, "STEP_ARGMAX t=%d cb=%d argmax=%d val=0.0000\n", t, k, sampled_ids[(size_t)k]);
                if (t + 1 <= k) { grid[(size_t)k][(size_t)t + 1] = BOS; continue; }  // delay window
                grid[(size_t)k][(size_t)t + 1] = sampled_ids[(size_t)k];
            }
            generated_ids.push_back(grid[0][(size_t)t + 1]);
            std::cout << grid[0][(size_t)t + 1];
            if (t + 1 < steps) std::cout << ' ';
            continue;
        }

        std::vector<float> logits = model.forward_cfg(col, /*start_pos=*/t, guidance_scale);
        if ((int)logits.size() != NCB * VOCAB) {
            NNOPT_ERROR_FMT("decode step %d: expected %d logits, got %zu", t, NCB * VOCAB, logits.size());
            decode_ok = false;
            break;
        }
        if (t == 0) NNOPT_BENCH_FIRST_TOKEN();
        for (int k = 0; k < NCB; ++k) {
            // Per-step per-codebook argmax (always logged — decode-math probe).
            int am = 0; float bv = logits[(size_t)k * VOCAB];
            for (int i = 1; i < VOCAB; ++i) {
                const float v = logits[(size_t)k * VOCAB + i];
                if (v > bv) { bv = v; am = i; }
            }
            fprintf(stderr, "STEP_ARGMAX t=%d cb=%d argmax=%d val=%.4f\n", t, k, am, bv);
            if (force_grid) continue;          // teacher forcing: keep preloaded grid
            if (t + 1 <= k) { grid[(size_t)k][(size_t)t + 1] = BOS; continue; }  // delay window
            std::vector<float> row(logits.begin() + (size_t)k * VOCAB,
                                   logits.begin() + (size_t)(k + 1) * VOCAB);
            no_history.clear();
            const int next = sampler.sample(row, no_history);
            grid[(size_t)k][(size_t)t + 1] = (int32_t)next;
        }
        generated_ids.push_back(grid[0][(size_t)t + 1]);
        std::cout << grid[0][(size_t)t + 1];
        if (t + 1 < steps) std::cout << ' ';
    }
    // Stage-4: ONE whole-grid readback at end-of-decode, then mirror into the
    // jagged host grid so the un-delay / EnCodec path below is unchanged.
    // NOTE: in pipeline mode the loop above only ENQUEUED the decode — this
    // blocking read is where the GPU actually finishes, so mark_end comes
    // after it (decode_tokens_per_sec stays honest).
    if (gpu_grid && grid_dev) {
        if (model.gpu_grid_read(grid_dev, flat_grid.data(), NCB, steps1)) {
            for (int k = 0; k < NCB; ++k)
                for (int c = 0; c < steps1; ++c)
                    grid[(size_t)k][(size_t)c] = flat_grid[(size_t)k * steps1 + c];
        } else {
            NNOPT_ERROR("Stage4: gpu_grid_read failed");
        }
        model.gpu_grid_free(grid_dev);
        grid_dev = nullptr;
    }
    bench.mark_end();   // after the blocking grid read — see pipeline note above
    // streaming-EnCodec: drain the worker, then run the FINAL chunk (true clip
    // edge) on this thread — only this residual tail costs wall time.
    std::vector<float> stream_pcm;
    if (enc_stream) {
        { std::lock_guard<std::mutex> lk(enc_mx); enc_done_flag = true; }
        enc_cv.notify_one();
        if (enc_worker.joinable()) enc_worker.join();
        if (!enc_stream_bad && decode_ok) {
            const auto tail_t0 = std::chrono::steady_clock::now();
            bool ok2 = true;
            for (int f = 0; f < stream_frames && ok2; ++f)
                for (int k = 0; k < NCB; ++k) {
                    const int32_t id = grid[(size_t)k][(size_t)(f + k + 1)];
                    if (id < 0 || id >= VOCAB) { ok2 = false; break; }
                    codes_live[(size_t)k][(size_t)f] = id;
                }
            if (ok2 && enc_stream->push(codes_live, stream_frames, true) && enc_stream->ok()) {
                stream_pcm = enc_stream->pcm();
                fprintf(stderr, "BENCHMARK encodec_stream_tail_sec: %.3f (rest overlapped with decode)\n",
                        std::chrono::duration<double>(std::chrono::steady_clock::now() - tail_t0).count());
            } else {
                fprintf(stderr, "ENC_STREAM: finalize failed - falling back to standard EnCodec\n");
            }
        } else if (enc_stream_bad) {
            fprintf(stderr, "ENC_STREAM: non-decodable ids (early EOS?) - standard EnCodec fallback\n");
        }
    }
    // Pipeline mode deferred the token stream — reconstruct it now from the
    // final grid (stdout keeps the Evaluate contract format).
    if (gpu_grid && pipeline_mode && generated_ids.empty()) {
        for (int t = 0; t < steps; ++t) {
            generated_ids.push_back(grid[0][(size_t)t + 1]);
            std::cout << grid[0][(size_t)t + 1];
            if (t + 1 < steps) std::cout << ' ';
        }
        std::cout << std::endl;
    }

    // [3] Un-delay: codes[k][f] = grid[k][f + k + 1]  (skip BOS col + k delay)
    const int frames = steps - (NCB - 1);
    if (decode_ok && frames > 0) {
        std::vector<std::vector<int32_t>> codes(
            (size_t)NCB, std::vector<int32_t>((size_t)frames, 0));
        for (int k = 0; k < NCB; ++k)
            for (int f = 0; f < frames; ++f)
                codes[(size_t)k][(size_t)f] = grid[(size_t)k][(size_t)(f + k + 1)];

        // Codes dump (always): lets host-side tooling A/B my EnCodec against
        // PyTorch EncodecModel on the IDENTICAL token grid — the decisive
        // stage-4 isolation test (FIX_LEDGER Fix 8 follow-up).
        if (FILE* cf = std::fopen("codes.bin", "wb")) {
            for (int k = 0; k < NCB; ++k)
                std::fwrite(codes[(size_t)k].data(), sizeof(int32_t), (size_t)frames, cf);
            std::fclose(cf);
            fprintf(stderr, "CODES_DUMP codes.bin [%d x %d]\n", NCB, frames);
        }

        // NNOPT_ENC_STREAM=2: offline validation of the streaming EnCodec —
        // feed codes in 50-frame chunks (as the live pipeline will) and
        // BYTE-COMPARE the streamed PCM against the full host decode.
        if (const char* es = std::getenv("NNOPT_ENC_STREAM"); es && es[0] == '2') {
            const int Fr = (int)codes[0].size();
            EncodecStream stream(weights, Fr);
            for (int a = 50; ; a += 50) {
                const bool last = a >= Fr;
                if (!stream.push(codes, last ? Fr : a, last)) break;
                if (last) break;
            }
            const std::vector<float> full = encodec_decode_host(weights, codes);
            const std::vector<float>& st = stream.pcm();
            size_t mism = 0, first = 0;
            const size_t n = std::min(full.size(), st.size());
            for (size_t i = 0; i < n; ++i)
                if (full[i] != st[i]) { if (!mism) first = i; ++mism; }
            fprintf(stderr, "ENC_STREAM_VALIDATE: sizes %zu/%zu mismatches=%zu first=%zu ok=%d\n",
                    full.size(), st.size(), mism, first,
                    (int)(stream.ok() && full.size() == st.size() && mism == 0));
            return 0;
        }
        // [4] EnCodec RVQ + SEANet decode → PCM → WAV + marker.
        // Default GPU (kernels/encodec.cl; LSTM stays host-side). Any GPU-path
        // error returns empty → automatic fallback to the host decoder.
        // NNOPT_ENCODEC_GPU=0 forces the host path (A/B + cosine validation).
        NNOPT_CHECKPOINT("encodec_decode start");
        const auto enc_t0 = std::chrono::steady_clock::now();
        const bool enc_gpu = [](){ const char* e = std::getenv("NNOPT_ENCODEC_GPU"); return !(e && e[0] == '0'); }();
        std::vector<float> pcm;
        const char* enc_path = "host";
        if (!stream_pcm.empty() && (int)stream_pcm.size() == frames * 640) {
            pcm.swap(stream_pcm);
            enc_path = "streamed";
        } else if (enc_gpu) {
            pcm = encodec_decode_gpu(cl_ctx, weights, codes);
            if (!pcm.empty()) enc_path = "gpu";
            else fprintf(stderr, "ENCODEC_GPU failed — falling back to host path\n");
        }
        if (pcm.empty()) pcm = encodec_decode_host(weights, codes);
        fprintf(stderr, "BENCHMARK encodec_decode_sec: %.4f (%s)\n",
                std::chrono::duration<double>(std::chrono::steady_clock::now() - enc_t0).count(),
                enc_path);
        if (pcm.empty()) {
            NNOPT_ERROR("encodec_decode_host returned no samples");
        } else if (!write_wav(out_wav_path, pcm.data(), pcm.size(), MODEL_CONFIG::SAMPLING_RATE)) {
            NNOPT_ERROR_FMT("write_wav(%s) failed", out_wav_path.c_str());
        } else {
            fprintf(stderr, "TTS_OUTPUT_PCM_SAMPLES %zu\n", pcm.size());
            char cwd_buf[512] = {0};
            const char* cw = getcwd(cwd_buf, sizeof(cwd_buf) - 1);
            fprintf(stderr, "\n=== AUDIO READY ===\n");
            fprintf(stderr, "  file:     %s/%s\n", cw ? cw : ".", out_wav_path.c_str());
            fprintf(stderr, "  duration: %.2f s @ %d Hz\n",
                    (double)pcm.size() / MODEL_CONFIG::SAMPLING_RATE, MODEL_CONFIG::SAMPLING_RATE);
            fprintf(stderr, "  prompt:   %s\n===================\n", prompt.c_str());
            NNOPT_CHECKPOINT("WAV written");
            // Host-side dump of the waveform so SxS can cosine it against
            // reference/layers/waveform_output.bin (pull via layer_dumps/).
            if (const char* d = std::getenv("NNOPT_DUMP_LAYERS"); d && std::strcmp(d, "1") == 0) {
                if (FILE* f = std::fopen("layer_dumps/waveform_output.bin", "wb")) {
                    std::fwrite(pcm.data(), sizeof(float), pcm.size(), f);
                    std::fclose(f);
                }
            }
        }
    } else if (decode_ok) {
        NNOPT_ERROR_FMT("max_new_tokens=%d too small for %d codebooks (need > %d)",
                        max_new_tokens, NCB, NCB - 1);
    }

    fprintf(stderr, "SERVE_GEN_DONE %d wall=%.3f s\n", serve_gen,
            std::chrono::duration<double>(std::chrono::steady_clock::now() - gen_t0).count());
    if (!serve_mode) break;
    fprintf(stderr, "SERVE_READY\n");
    fflush(stderr); fflush(stdout);
    {
        std::string line;
        if (!std::getline(std::cin, line)) break;
        while (!line.empty() && (line.back() == '\r' || line.back() == '\n')) line.pop_back();
        if (line.empty() || line == "quit") break;
        prompt = line;
        token_ids_path.clear();          // serve prompts always tokenize
        ++serve_gen;
        char wpath[64]; std::snprintf(wpath, sizeof(wpath), "serve_%03d.wav", serve_gen);
        out_wav_path = wpath;
    }
    }   // ── end serve loop ──

    // Per-kernel GPU profile (env NNOPT_PROFILE=1). Dormant by default;
    // KEEP this call site even when restructuring prefill/decode (same
    // rule as the 5 benchmark sites — see vlm.md).
    KernelProfiler::dump_summary();
    bench.print_summary((int)prompt_len, (int)generated_ids.size());
    fprintf(stderr, "BENCHMARK e2e_wall_sec: %.4f\n",
            std::chrono::duration<double>(std::chrono::steady_clock::now() - e2e_t0).count());
    return 0;

}
