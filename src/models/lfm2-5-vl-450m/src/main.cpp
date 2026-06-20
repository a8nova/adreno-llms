// Auto-generated graph-mode main.cpp for LiquidAI/LFM2.5-VL-450M.
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
#include "ops/lfm2_common.h"  // nnopt_register_cl_ctx_for_gemv (M==1 GEMV fast path bootstrap)

// int8 fast-path registration (defined in src/utils.cpp). Forward-declared
// here so main.cpp doesn't need to pull all of utils.h's int8 plumbing.
extern "C" bool nnopt_register_int8_weight(cl_mem W_int8, cl_mem scale_fp16, int N, int K);
extern "C" cl_mem nnopt_dequant_int8_to_fp16_alloc(cl_command_queue queue,
                                                   cl_mem W_int8,
                                                   cl_mem scale_fp16,
                                                   int N, int K);

#include "load_image.h"  // load_image_rgb_u8 + ImageBufferU8 (stb_image)



#include <fstream>
#include <iostream>
#include <string>
#include <vector>
#include <cstdint>
#include <cstring>
#include <cstdlib>
#include <cstdio>
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

int main(int argc, char** argv) {
    // (No version banner — debug_utils does not define one, and emitting an
    // undefined macro here was breaking every fresh-port build.)
    // Argument parsing: positional "prompt" + optional flags.
    //   ./binary "<prompt>" [max_new_tokens] [--token-ids <file>]
    std::string prompt = "The teacher worked at the ";
    int max_new_tokens = 16;
    std::string token_ids_path;

    std::string image_path;  // --image <file.jpg|.png> for VLM inference
    bool interactive_mode = false;  // --interactive: stdin REPL with /image, /reset (host-app describeImage contract)

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

            } else if (a == "--image" && i + 1 < argc) {
                image_path = argv[++i];

            } else if (a == "--temperature" && i + 1 < argc) {
                sampler_config.temperature = std::stof(argv[++i]);
            } else if (a == "--top-k" && i + 1 < argc) {
                sampler_config.top_k = std::stoi(argv[++i]);
            } else if (a == "--top-p" && i + 1 < argc) {
                sampler_config.top_p = std::stof(argv[++i]);
            } else if (a == "--seed" && i + 1 < argc) {
                sampler_config.seed = static_cast<uint32_t>(std::stoul(argv[++i]));
            } else if (a == "--interactive") {
                interactive_mode = true;
            } else if (a == "--max-tokens" && i + 1 < argc) {
                max_new_tokens = std::stoi(argv[++i]);
            } else if (a == "--image-size" && i + 1 < argc) {
                ++i;  // accepted for app compatibility; LFM2-VL uses fixed TILE_SIZE (512)
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
        // Debug layer checks do synchronous device->host reads for every layer.
    // On this VLM prompt (1806 tokens) that alone can exceed the CLI timeout;
    // keep checks enabled only for explicit dump/debug runs.
    if (!std::getenv("NNOPT_DUMP_LAYERS") && !std::getenv("NNOPT_DEBUG_LAYERS")) {
        setenv("NNOPT_DEBUG_LAYERS", "0", 1);
    }

    BenchmarkTimer& bench = BenchmarkTimer::instance();
    bench.mark_inference_start();


    OpenCLContext cl_ctx;
    if (!cl_ctx.initialize()) {
        NNOPT_ERROR("OpenCL init failed");
        return 1;
    }
    // Register cl_ctx so pytorch_linear's M==1 GEMV fast path can build the
    // singleton kernel without threading the context through utils.cpp.
    nnopt_register_cl_ctx_for_gemv(&cl_ctx);

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

    // Weights — fp16 build prefers weights/model.int8.bin (per-row int8 +
    // fp16 scales; ~50% size reduction over model.fp16.bin and halved decode
    // bandwidth). Falls back to model.fp16.bin if the int8 bundle is missing
    // on device. fp32 build keeps original.
    Weights weights;
    bool nnopt_quant_int8 = false;
#ifdef NNOPT_USE_FP16
    const char* nnopt_weights_bin  = "weights/model.fp16.bin";
    const char* nnopt_weights_meta = "weights/model.fp16.meta.json";
    {
        // Probe int8 bundle: stat is cheap; if both files exist on device,
        // prefer them. fopen is robust to mmap fallback paths.
        FILE* probe_bin  = std::fopen("weights/model.int8.bin",       "rb");
        FILE* probe_meta = std::fopen("weights/model.int8.meta.json", "rb");
        if (probe_bin && probe_meta) {
            nnopt_weights_bin  = "weights/model.int8.bin";
            nnopt_weights_meta = "weights/model.int8.meta.json";
            nnopt_quant_int8   = true;
            std::cerr << "[main] using int8 weights: " << nnopt_weights_bin << std::endl;
        }
        if (probe_bin)  std::fclose(probe_bin);
        if (probe_meta) std::fclose(probe_meta);
    }
#else
    const char* nnopt_weights_bin  = "weights/model.bin";
    const char* nnopt_weights_meta = "weights/model.meta.json";
#endif
    if (!weights.load(nnopt_weights_bin, nnopt_weights_meta, cl_ctx.context())) {
        NNOPT_ERROR_FMT("weights load failed: %s", nnopt_weights_bin);
        return 1;
    }

#ifdef NNOPT_USE_FP16
    // Walk every int8 tensor + sibling `.scales` and hand both cl_mems to
    // utils.cpp. From here on, any pytorch_linear() call on these W buffers
    // takes the int8 image-path GEMV at M=1 and the dequant-then-HGemm path
    // at M>1.
    if (nnopt_quant_int8) {
        // Some op kernels create fixed-byte-stride sub-buffers off the raw W
        // buffer (e.g. op_lfm2_conv_block splits in_proj.weight into B/C/x
        // slices assuming fp16 byte layout). Those layers can't consume an
        // int8 buffer as-is — we leave their weights fp16 by not registering
        // them with the int8 fast path. The layer code will see is_int8()==
        // true via Weights but we just don't make it special-case anything;
        // the M>1 dequant path also won't fire (pytorch_linear only takes
        // that path when `is_int8_registered(W)` returns true).
        //
        // Skip suffix patterns:
        //   .conv.in_proj.weight — split into 3 sub-buffers of fp16 stride
        auto ends_with = [](const std::string& s, const char* suf) {
            const size_t L = std::strlen(suf);
            return s.size() >= L && std::memcmp(s.data() + s.size() - L, suf, L) == 0;
        };

        int registered = 0;
        int skipped = 0;
        for (const std::string& name : weights.all_keys()) {
            if (!weights.is_int8(name)) continue;
            const auto shape = weights.get_shape(name);
            if (shape.size() != 2) continue;
            // Layers below consume the weight buffer with hardcoded fp16
            // byte stride (sub-buffer split, direct half* indexed reads, etc.)
            // and cannot read int8 byte stride. Dequantize once to fp16, swap
            // the Weights-cached cl_mem so the layer sees an fp16 buffer, and
            // skip registering for the int8 fast path.
            //
            // - .conv.in_proj.weight: op_lfm2_conv_block.cpp splits into 3
            //   sub-buffers with fp16 byte stride.
            // - vision_tower position_embedding.weight: bilinear_position_resize
            //   kernel reads as `storage_t* = half*` directly (no scale).
            const bool needs_fp16_materialize =
                ends_with(name, ".conv.in_proj.weight") ||
                ends_with(name, "vision_tower.vision_model.embeddings.position_embedding.weight");
            if (needs_fp16_materialize) {
                const int N = shape[0];
                const int K = shape[1];
                cl_mem W_int8 = weights.get_buffer(name);
                cl_mem S      = weights.get_scale_buffer(name);
                if (W_int8 && S) {
                    cl_mem fp16 = nnopt_dequant_int8_to_fp16_alloc(cl_ctx.queue(), W_int8, S, N, K);
                    if (fp16) {
                        weights.replace_buffer_as_fp16(name, fp16);
                    }
                }
                ++skipped;
                continue;
            }
            cl_mem W = weights.get_buffer(name);
            cl_mem S = weights.get_scale_buffer(name);
            if (!W || !S) continue;
            if (nnopt_register_int8_weight(W, S, shape[0], shape[1])) {
                registered++;
            }
        }
        std::cerr << "[main] registered " << registered << " int8 weights for GEMV/GEMM fast path "
                  << "(skipped " << skipped << " sub-buffer-split-incompatible)" << std::endl;
    }
#endif

    Sampler sampler(sampler_config);

    Model model(cl_ctx, weights);
    if (!model.initialize()) {
        NNOPT_ERROR("Model::initialize() failed — see prior NNOPT_ERROR for the layer that failed");
        return 1;
    }

    // ── Interactive REPL ──
    // `--interactive` opens a stdin-driven loop that matches the host app's
    // describeImage contract (and SmolVLM's interactive mode):
    //   /image <path>   load image, run vision pipeline (set_image)
    //   /reset          drop the image (next prompt is text-only)
    //   /quit | EOF | blank line   exit
    // A plain text line is one user turn: encode_with_image() when an image is
    // loaded (else text-only encode), prefill at start_pos=0, stream decoded
    // tokens to STDOUT, stop at eos. Per-turn perf + a `✓ turn` marker go to
    // STDERR (the app reads stdout for the caption, stderr for perf). The caller
    // closes stdin after one turn, so getline() returns EOF and we exit cleanly.
    // NOTE: each turn re-prefills at start_pos=0 (no multi-turn KV reuse) — the
    // app does exactly one image+prompt per process, which this serves directly.
    if (interactive_mode) {
        if (!tokenizer_ok) {
            NNOPT_ERROR("--interactive needs weights/tokenizer_vocab.bin");
            return 1;
        }
        const int  MAX_DECODE_PER_TURN = (max_new_tokens > 0 ? max_new_tokens : 128);
        const bool greedy = (sampler_config.temperature <= 0.0f);
        // ChatML assistant turns terminate with <|im_end|> (id 7), NOT the tokenizer's
        // eos_id (2). Without stopping on it the decode runs to max-tokens and pads the
        // (otherwise correct) caption with repetition garbage. Stop on either.
        const int32_t im_end_id = tok.token_to_id("<|im_end|>");

        bool image_loaded = false;
        int  turn = 0;

        std::fprintf(stderr,
                     "ready. commands:\n"
                     "  /image <path>   load an image\n"
                     "  /reset          start fresh (text-only)\n"
                     "  /quit           exit (or Ctrl-D / blank line)\n");
        std::fflush(stderr);

        std::string line;
        while (true) {
            std::fprintf(stderr, "\n> "); std::fflush(stderr);
            if (!std::getline(std::cin, line)) break;   // EOF → exit
            while (!line.empty() && (line.back() == '\r' || line.back() == '\n')) line.pop_back();
            if (line.empty()) break;

            // /image <path>
            if (line.rfind("/image ", 0) == 0 || line.rfind("/image\t", 0) == 0) {
                std::string path = line.substr(7);
                while (!path.empty() && (path.front() == ' ' || path.front() == '\t')) path.erase(path.begin());
                while (!path.empty() && (path.back() == ' ' || path.back() == '\t')) path.pop_back();
                if (path.empty()) { std::fprintf(stderr, "usage: /image <path>\n"); continue; }
                ImageBufferU8 img;
                if (!load_image_rgb_u8(path, img)) {
                    std::fprintf(stderr, "failed to load %s: %s\n", path.c_str(), img.error.c_str());
                    continue;
                }
                if (!model.set_image(img.data, img.W, img.H)) {
                    std::fprintf(stderr, "set_image failed for %s\n", path.c_str());
                    image_loaded = false;
                    continue;
                }
                image_loaded = true;
                std::fprintf(stderr, "loaded %s (%dx%d)\n", path.c_str(), img.W, img.H);
                std::fflush(stderr);
                continue;
            }
            if (line == "/reset") { image_loaded = false; std::fprintf(stderr, "reset\n"); continue; }
            if (line == "/quit" || line == "/exit") break;

            // Build this turn's input ids (same machinery as the batch --image path).
            std::vector<int32_t> prompt_ids;
            if (image_loaded) {
                const int grid_h = model.image_grid_h();
                const int grid_w = model.image_grid_w();
                const int thumb_h = model.image_thumbnail_h();
                const int thumb_w = model.image_thumbnail_w();
                const bool use_thumbnail = (thumb_h > 0 && thumb_w > 0);
                const int tile_tokens = (MODEL_CONFIG::TILE_SIZE / MODEL_CONFIG::ENCODER_PATCH_SIZE
                                         / MODEL_CONFIG::DOWNSAMPLE_FACTOR);
                const int tile_tokens_sq = tile_tokens * tile_tokens;
                prompt_ids = tok.encode_with_image(line, grid_h, grid_w, thumb_h, thumb_w,
                                                   tile_tokens_sq, use_thumbnail,
                                                   /*add_generation_prompt=*/true);
            } else {
                const std::vector<int> enc = tok.encode(line);
                prompt_ids.assign(enc.begin(), enc.end());
            }
            if (prompt_ids.empty()) { std::fprintf(stderr, "tokenize produced no ids\n"); continue; }

            const size_t prompt_len = prompt_ids.size();
            const auto t_prefill_0 = std::chrono::steady_clock::now();
            std::vector<float> logits = model.forward(prompt_ids, /*start_pos=*/0);
            const auto t_first = std::chrono::steady_clock::now();

            std::vector<int32_t> generated;
            generated.reserve(MAX_DECODE_PER_TURN);
            for (int step = 0; step < MAX_DECODE_PER_TURN; ++step) {
                int next;
                if (!logits.empty()) { next = sampler.sample(logits, generated); logits.clear(); }
                else                 { next = model.read_greedy_result(); }
                if ((sampler_config.eos_token_id >= 0 && next == sampler_config.eos_token_id) ||
                    (im_end_id >= 0 && next == im_end_id)) break;
                generated.push_back(next);
                std::cout << tok.decode(std::vector<int32_t>{next}) << std::flush;
                if (step + 1 < MAX_DECODE_PER_TURN) {
                    const int decode_start_pos = (int)prompt_len + step;
                    if (greedy) model.forward_greedy(std::vector<int32_t>{next}, decode_start_pos);
                    else        logits = model.forward(std::vector<int32_t>{next}, decode_start_pos);
                }
            }
            const auto t_decode_1 = std::chrono::steady_clock::now();
            std::cout << std::endl;
            turn++;

            const double prefill_s = std::chrono::duration<double>(t_first    - t_prefill_0).count();
            const double decode_s  = std::chrono::duration<double>(t_decode_1 - t_first).count();
            const double prefill_tps = prefill_s > 0 ? (double)prompt_len        / prefill_s : 0.0;
            const double decode_tps  = decode_s  > 0 ? (double)generated.size()  / decode_s  : 0.0;
            std::fprintf(stderr,
                         "✓ turn %d  prefill %zu tok %.2fs (%.2f tok/s)  decode %zu tok %.2fs (%.2f tok/s)\n",
                         turn, prompt_len, prefill_s, prefill_tps, generated.size(), decode_s, decode_tps);
            // BENCHMARK lines: parsed by the app's PerfAccumulator for the tok/s strip.
            std::fprintf(stderr, "BENCHMARK prefill_tokens_per_sec: %.4f\n", prefill_tps);
            std::fprintf(stderr, "BENCHMARK decode_tokens_per_sec: %.4f\n", decode_tps);
            std::fprintf(stderr, "BENCHMARK time_to_first_token_sec: %.4f\n", prefill_s);
            std::fflush(stderr);
        }
        std::fprintf(stderr, "\nbye.\n");
        return 0;
    }

    // VLM input pipeline: when --image is provided, decode RGB8 via stb_image,
    // run the multi-tile image processor + SigLIP encoder + projector. The
    // tokenizer builds an input_ids sequence with the right number of <image>
    // placeholders to match the produced features. The masked_scatter splice
    // in backbone.cpp does per-position replacement.
    bool image_loaded = false;
    if (!image_path.empty()) {
        ImageBufferU8 img;
        if (!load_image_rgb_u8(image_path, img)) {
            NNOPT_ERROR_FMT("failed to load image %s: %s",
                            image_path.c_str(), img.error.c_str());
            return 1;
        }
        std::cerr << "[main] loaded image " << image_path
                  << " (" << img.W << "x" << img.H << ", "
                  << img.data.size() << " bytes RGB)" << std::endl;
        if (!model.set_image(img.data, img.W, img.H)) {
            NNOPT_ERROR("Model::set_image() failed");
            return 1;
        }
        image_loaded = true;
    }

    // Build input_ids.
    // - If --token-ids was passed: use those (deterministic eval against ref).
    // - Else if --image was passed: tokenize via encode_with_image() with
    //   the actual tile grid dims from the image processor.
    // - Else: plain prompt encode (text-only).
    std::vector<int32_t> prompt_ids;
    if (!token_ids_path.empty() && read_input_ids_bin(token_ids_path, prompt_ids)) {
        // OK — got authoritative ids from disk.
    } else if (image_loaded) {
        const int grid_h = model.image_grid_h();
        const int grid_w = model.image_grid_w();
        const int thumb_h = model.image_thumbnail_h();
        const int thumb_w = model.image_thumbnail_w();
        const bool use_thumbnail = (thumb_h > 0 && thumb_w > 0);
        // tile_tokens = (TILE_SIZE/PATCH_SIZE/DOWNSAMPLE_FACTOR)^2 = (512/16/2)^2 = 256
        const int tile_tokens = (MODEL_CONFIG::TILE_SIZE / MODEL_CONFIG::ENCODER_PATCH_SIZE
                                 / MODEL_CONFIG::DOWNSAMPLE_FACTOR);
        const int tile_tokens_sq = tile_tokens * tile_tokens;
        prompt_ids = tok.encode_with_image(prompt, grid_h, grid_w,
                                           thumb_h, thumb_w, tile_tokens_sq,
                                           use_thumbnail, /*add_generation_prompt=*/true);
        std::cerr << "[main] encode_with_image -> " << prompt_ids.size()
                  << " tokens (grid=" << grid_h << "x" << grid_w
                  << ", thumb=" << thumb_h << "x" << thumb_w << ")" << std::endl;
    } else {
        const std::vector<int> encoded = tok.encode(prompt);
        prompt_ids.assign(encoded.begin(), encoded.end());
    }
    const size_t prompt_len = prompt_ids.size();
    std::vector<float> wav_samples;
    std::vector<float> raw_out;

    // Single batched prefill, then per-token decode. GPU argmax for greedy
    // (temp=0): replaces 128KB host readback with 4-byte readback per step.
    const bool greedy = (sampler_config.temperature <= 0.0f);
    bench.mark_prefill_start();
    // Prefill always uses full logits path (needed for first sample + debug).
    std::vector<float> logits = model.forward(prompt_ids, /*start_pos=*/0);
    std::vector<int32_t> generated_so_far;
    for (int step = 0; step < max_new_tokens; ++step) {
        int next;
        if (!logits.empty()) {
            next = sampler.sample(logits, generated_so_far);
            logits.clear();
        } else {
            next = model.read_greedy_result();
            fprintf(stderr, "Sampler: max_logit=GPU_argmax at id=%d, logits_size=%d\n",
                    next, MODEL_CONFIG::VOCAB_SIZE);
        }
        NNOPT_BENCH_FIRST_TOKEN();
        generated_so_far.push_back(next);
        prompt_ids.push_back(next);
        if (sampler_config.eos_token_id >= 0 && next == sampler_config.eos_token_id) break;
        std::cout << tok.decode(std::vector<int32_t>{next}) << std::flush;
        if (step + 1 < max_new_tokens) {
            const int decode_start_pos = (int)prompt_len + step;
            if (greedy) {
                model.forward_greedy(std::vector<int32_t>{next}, decode_start_pos);
            } else {
                logits = model.forward(std::vector<int32_t>{next}, decode_start_pos);
            }
        }
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
