// Auto-generated graph-mode main.cpp for functiongemma-270m-it.
// Modality: input=tokens, output=tokens
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
#include "model_config.h"
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
#include <chrono>

// Read prompt input ids from a binary file (int32 little-endian) when the
// caller passes \`--token-ids /path/to/test_input_ids.bin\`. This path is
// the deterministic-evaluation contract — Evaluate compares the C++ output
// against the PyTorch reference for the SAME input ids, so the binary
// MUST consume the file rather than re-tokenizing the prompt string (which
// can produce different ids if the encoder isn't perfectly aligned with
// HF's tokenizer for this model family).
// Greedy GPU-argmax fast-path (implemented in src/backbone.cpp).
extern "C" void nnopt_set_greedy_argmax(int on);
extern "C" int  nnopt_get_argmax_token();

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

int main(int argc, char** argv) {
    // Arm the crash handler FIRST — on SIGSEGV/SIGABRT/SIGBUS it prints the
    // last NNOPT_CHECKPOINT, a backtrace, and the GPU-mem allocation log so a
    // device segfault reports WHERE it died instead of a bare "Segmentation
    // fault". Parity with main.cpp.tmpl:44 (graph-mode main dropped this in the
    // Phase 3 redesign, blinding every graph-mode port to crash locations).
    nnopt_install_crash_handler();
    // (No version banner — debug_utils does not define one, and emitting an
    // undefined macro here was breaking every fresh-port build.)
    // Argument parsing: positional "prompt" + optional flags.
    //   ./binary "<prompt>" [max_new_tokens] [--token-ids <file>]
    std::string prompt = "The teacher worked at the ";
    int max_new_tokens = 16;
    std::string token_ids_path;

    SamplerConfig sampler_config;
    sampler_config.temperature = 0.0f;          // greedy by default — matches PyTorch reference
    sampler_config.top_k = 1;
    sampler_config.top_p = 1.0f;
    sampler_config.repetition_penalty = 1.0f;
    sampler_config.seed = 42u;
    {
        int positional = 0;
        for (int i = 1; i < argc; i++) {
            const std::string a = argv[i];
            if (a == "--token-ids" && i + 1 < argc) {
                token_ids_path = argv[++i];

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
                prompt = a; positional++;
            } else if (positional == 1) {
                max_new_tokens = std::stoi(a); positional++;
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

    OpenCLContext cl_ctx;
    if (!cl_ctx.initialize()) {
        NNOPT_ERROR("OpenCL init failed");
        return 1;
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

    // ── NNOPT_BWTEST=1: pure fp16 streaming-bandwidth microbenchmark ──
    // Reads the 335MB tied-embedding weight once per launch with the same
    // coalesced access pattern any matvec must use, summing into a dummy out.
    // This isolates the *device's* achievable fp16 streaming bandwidth — the
    // hard floor for decode — from any GEMV-kernel inefficiency. Prints GB/s
    // and the implied lm_head time, then exits before generation.
    if (const char* bw = std::getenv("NNOPT_BWTEST"); bw && bw[0] == '1') {
        {
            cl_device_id dev = cl_ctx.device();
            size_t esz = 0; clGetDeviceInfo(dev, CL_DEVICE_EXTENSIONS, 0, nullptr, &esz);
            std::vector<char> ext(esz + 1, 0);
            clGetDeviceInfo(dev, CL_DEVICE_EXTENSIONS, esz, ext.data(), nullptr);
            std::cerr << "DEVICE_EXTENSIONS: " << ext.data() << std::endl;
            char dv[256] = {0}; clGetDeviceInfo(dev, CL_DEVICE_VERSION, sizeof(dv), dv, nullptr);
            std::cerr << "DEVICE_VERSION: " << dv << std::endl;
        }
        cl_mem W = weights.get_buffer("model.embed_tokens.weight", true);
        if (!W) W = weights.get_buffer("lm_head.weight", true);
        if (!W) { NNOPT_ERROR("BWTEST: no embed/lm_head weight"); return 1; }
        const size_t n_elems = weights.get_num_elements("model.embed_tokens.weight");
        const size_t bytes = weights.get_size_bytes("model.embed_tokens.weight");
        std::string opts;
#ifdef NNOPT_USE_FP16
        opts = "-D USE_FP16=1";
#endif
        cl_program prog = cl_ctx.build_program_from_file("kernels/gemma3_ops.cl", opts);
        if (!prog) { NNOPT_ERROR("BWTEST: program build failed"); return 1; }
        cl_int err = CL_SUCCESS;
        cl_kernel k = clCreateKernel(prog, "gemma3_bw_test", &err);
        if (err != CL_SUCCESS || !k) { NNOPT_ERROR_FMT("BWTEST: clCreateKernel %d", (int)err); return 1; }
        // Sweep grid sizes — bandwidth on Adreno is sensitive to occupancy.
        const size_t grids[] = {16384, 65536, 262144, 1048576};
        const int n_elems_i = (int)n_elems;
        double best_gbps = 0.0; double best_ms = 0.0; size_t best_grid = 0;
        for (size_t gws : grids) {
            cl_mem out = clCreateBuffer(cl_ctx.context(), CL_MEM_WRITE_ONLY, gws * sizeof(float), nullptr, &err);
            if (err != CL_SUCCESS || !out) { NNOPT_ERROR_FMT("BWTEST: out alloc %d", (int)err); return 1; }
            clSetKernelArg(k, 0, sizeof(cl_mem), &W);
            clSetKernelArg(k, 1, sizeof(cl_mem), &out);
            clSetKernelArg(k, 2, sizeof(int), &n_elems_i);
            // Warm up (JIT + first-touch), then time 10 iterations.
            clEnqueueNDRangeKernel(cl_ctx.queue(), k, 1, nullptr, &gws, nullptr, 0, nullptr, nullptr);
            clFinish(cl_ctx.queue());
            const int iters = 10;
            auto t0 = std::chrono::high_resolution_clock::now();
            for (int it = 0; it < iters; ++it)
                clEnqueueNDRangeKernel(cl_ctx.queue(), k, 1, nullptr, &gws, nullptr, 0, nullptr, nullptr);
            clFinish(cl_ctx.queue());
            auto t1 = std::chrono::high_resolution_clock::now();
            double ms = std::chrono::duration<double, std::milli>(t1 - t0).count() / iters;
            double gbps = (double)bytes / (ms * 1e-3) / 1e9;
            std::cerr << "BWTEST grid=" << gws << "  " << ms << " ms/pass  "
                      << gbps << " GB/s" << std::endl;
            if (gbps > best_gbps) { best_gbps = gbps; best_ms = ms; best_grid = gws; }
            clReleaseMemObject(out);
        }
        // lm_head reads 168M MAC (= n_elems fp16 weights). At best bandwidth:
        std::cerr << "BWTEST best: " << best_gbps << " GB/s @ grid=" << best_grid
                  << "  (lm_head floor ~" << best_ms << " ms, "
                  << (n_elems / (best_ms * 1e-3) / 1e9) << " G MAC/s)" << std::endl;
        clReleaseKernel(k);
        clReleaseProgram(prog);
        return 0;
    }




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
    }
    const size_t prompt_len = prompt_ids.size();
    std::vector<float> wav_samples;
    std::vector<float> raw_out;



    bench.mark_prefill_start();
    // Greedy decode (temperature<=0) uses the GPU-argmax fast-path: the forward
    // computes argmax on-device and returns only the token id, skipping the
    // 262144-element logits readback + host convert + host argmax every token.
    // Sampling modes (temperature>0) keep the full logits path.
    const bool greedy = (sampler_config.temperature <= 0.0f);
    nnopt_set_greedy_argmax(greedy ? 1 : 0);
    // Incremental decode with KV cache: prefill processes the whole prompt
    // (start_pos=0) and populates the per-layer K/V cache; each subsequent
    // step feeds ONLY the freshly generated token at its absolute position so
    // the model attends over the cache instead of reprocessing the sequence.
    int cur_pos = 0;
    for (int step = 0; step < max_new_tokens; ++step) {

        std::vector<float> logits;
        if (step == 0) {
            logits = model.forward(prompt_ids, 0);
            cur_pos = (int)prompt_len;            // position of the token sampled below
        } else {
            std::vector<int32_t> one(1, prompt_ids.back());
            logits = model.forward(one, cur_pos);
            cur_pos++;
        }
        // Sample one token: greedy → GPU argmax id; otherwise the Sampler class.
        std::vector<int32_t> generated_so_far(prompt_ids.begin() + prompt_len, prompt_ids.end());
        int next = greedy ? nnopt_get_argmax_token()
                          : sampler.sample(logits, generated_so_far);
        NNOPT_BENCH_FIRST_TOKEN();  // stamps TTFT on the first sample only
        prompt_ids.push_back(next);
        if (sampler_config.eos_token_id >= 0 && next == sampler_config.eos_token_id) break;
        // Per-token streaming is intentionally OMITTED. Only the post-loop
        // "GENERATED_TEXT: ..." summary line emits to stdout. Rationale: dual
        // streams (per-token + summary) produced concatenated stdout that
        // Evaluate's parser misclassified — repeatedly. Single stdout source
        // → unambiguous parse → no duplication possible. If a user wants
        // live progress, they can run NNOPT_DEBUG=1 to log to stderr.
    }

    bench.mark_end();
    std::cout << std::endl;
    // Emit a deterministic post-generation summary that matches Evaluate's
    // contract: ONLY the generated tokens (sliced after prompt_len), so
    // comparison apples-to-apples with reference_tokens.json::generated_text.
    if (prompt_ids.size() > prompt_len) {
        const std::vector<int32_t> generated_ids(prompt_ids.begin() + prompt_len, prompt_ids.end());
        if (tokenizer_ok) {
            std::cout << "GENERATED_TEXT: " << tok.decode(generated_ids) << std::endl;
        } else {
            std::cout << "GENERATED_IDS:";
            for (int32_t id : generated_ids) std::cout << " " << id;
            std::cout << std::endl;
        }
    }
    // Per-kernel GPU profile (env NNOPT_PROFILE=1). Dormant by default;
    // KEEP this call site even when restructuring prefill/decode (same
    // rule as the 5 benchmark sites — see vlm.md).
    KernelProfiler::dump_summary();
    bench.print_summary((int)prompt_len, (int)(prompt_ids.size() - prompt_len));
    return 0;

}
