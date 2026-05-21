// Auto-generated graph-mode main.cpp for HuggingFaceTB/SmolVLM-256M-Instruct.
// Modality: input=tokens, output=tokens
//
// In graph mode main.cpp is small: tokenize/load input, instantiate Model,
// loop a generate step (or single forward for non-autoregressive), write
// output. The interesting code lives in src/ops/*.cpp.
//
// API REFERENCE (DO NOT GUESS — these are the canonical signatures used
// throughout this codebase; matching them avoids the iteration tax of
// inventing non-existent methods like load_from_disk / decode_one /
// sample_argmax):
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

#include "load_image.h"  // load_image_rgb_u8 + ImageBufferU8 (stb_image)
#include "benchmark.h"
#include "profile.h"


#include <fstream>
#include <iostream>
#include <string>
#include <vector>
#include <cstdint>
#include <cstring>
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
    // Argument parsing: positional "prompt" + optional flags.
    //   ./binary "<prompt>" [max_new_tokens] [--token-ids <file>]
    std::string prompt = "The teacher worked at the ";
    int max_new_tokens = 16;
    std::string token_ids_path;

    std::string image_path;  // --image <file.jpg|.png> for VLM inference
    bool interactive_mode = false;  // --interactive: REPL with /image, /reset, /quit + multi-turn KV reuse

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

            } else if (a == "--interactive") {
                interactive_mode = true;
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

    BenchmarkTimer& bench = BenchmarkTimer::instance();
    bench.mark_inference_start();
    const auto t_start = std::chrono::steady_clock::now();
    auto elapsed_ms = [&](const std::chrono::steady_clock::time_point& t) {
        return std::chrono::duration<double, std::milli>(t - t_start).count();
    };

    OpenCLContext cl_ctx;
    if (!cl_ctx.initialize()) {
        NNOPT_ERROR("OpenCL init failed");
        return 1;
    }
    const auto t_cl_init = std::chrono::steady_clock::now();
    std::fprintf(stderr, "  startup         opencl_init   %.0f ms\n", elapsed_ms(t_cl_init));

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
    const auto t_weights = std::chrono::steady_clock::now();
    std::fprintf(stderr, "  startup         weights_load  %.0f ms (cumulative %.0f ms)\n",
                 std::chrono::duration<double, std::milli>(t_weights - t_cl_init).count(),
                 elapsed_ms(t_weights));

    Sampler sampler(sampler_config);

    Model model(cl_ctx, weights);
    if (!model.initialize()) {
        NNOPT_ERROR("Model::initialize() failed — see prior NNOPT_ERROR for the layer that failed");
        return 1;
    }

    // ── Interactive REPL ──
    // `--interactive` opens a stdin-driven loop with three commands:
    //   /image <path>   load image, run vision pipeline, reset conversation
    //   /reset          drop image + KV state (logically — next prefill at start_pos=0 overwrites)
    //   /quit (or EOF or blank line)  exit
    // Plain text lines are tokenized as a follow-up user turn that reuses the
    // existing KV cache (KV_CACHE_MAX_LEN=2048 in llama_sdpa_attention.cpp).
    // Token-budget guard exits the turn early if total_pos approaches the cap.
    if (interactive_mode) {
        if (!tokenizer_ok) {
            NNOPT_ERROR("--interactive needs weights/tokenizer_vocab.bin");
            return 1;
        }

        // Prewarm: compile every decode kernel + warm L1 BEFORE the first turn
        // so first-prefill latency reflects steady-state, not cold compile.
        // Single <|im_start|> token at start_pos=0 — overwritten by turn 1.
        (void)model.forward(std::vector<int32_t>{1}, 0);
        model.reset_conversation();

        // Optional vision-tower prewarm. Gated by NNOPT_PREWARM_VISION=1 so
        // CLI bench runs / one-shot --image invocations don't pay the cost.
        // The see-and-say app sets this so app-launch eats the cold vision
        // pass once (~10-12s) instead of paying it on the first user query
        // and leaving the user staring at a spinner. Synthetic 512x512 gray
        // RGB exercises the same compute path as a real camera capture
        // (vision_pipeline resizes everything to 512x512 anyway).
        if (std::getenv("NNOPT_PREWARM_VISION")) {
            std::fprintf(stderr, "▶ prewarm: vision tower (synthetic 512x512)...\n");
            std::fflush(stderr);
            std::vector<uint8_t> dummy_rgb(3 * 512 * 512, 128);
            if (!model.set_image(dummy_rgb, 512, 512)) {
                NNOPT_ERROR("vision prewarm failed (continuing anyway)");
            }
            model.reset_conversation();
            std::fprintf(stderr, "▶ prewarm: vision done\n");
            std::fflush(stderr);
        }

        constexpr int32_t END_OF_UTTERANCE = 49279;
        constexpr int     MAX_DECODE_PER_TURN = 128;
        constexpr int     KV_CACHE_LIMIT = 2048;   // matches llama_sdpa_attention.cpp:43
        constexpr int     KV_HEADROOM = 256;

        int  total_pos = 0;
        bool image_loaded = false;
        int  turn = 0;

        std::fprintf(stderr,
                     "ready. commands:\n"
                     "  /image <path>   load an image (resets conversation)\n"
                     "  /reset          start a fresh conversation\n"
                     "  /quit           exit (or Ctrl-D / blank line)\n");
        std::fflush(stderr);

        std::string line;
        while (true) {
            std::fprintf(stderr, "\n> "); std::fflush(stderr);
            if (!std::getline(std::cin, line)) break;
            if (line.empty()) break;

            // /image <path>
            if (line.rfind("/image ", 0) == 0 || line.rfind("/image\t", 0) == 0) {
                std::string path = line.substr(7);
                while (!path.empty() &&
                       (path.front() == ' ' || path.front() == '\t')) path.erase(path.begin());
                while (!path.empty() &&
                       (path.back() == ' ' || path.back() == '\t' || path.back() == '\r')) path.pop_back();
                if (path.empty()) {
                    std::fprintf(stderr, "usage: /image <path>\n"); continue;
                }
                ImageBufferU8 img;
                if (!load_image_rgb_u8(path, img)) {
                    std::fprintf(stderr, "failed to load %s: %s\n",
                                 path.c_str(), img.error.c_str());
                    continue;
                }
                model.reset_conversation();
                total_pos = 0;
                image_loaded = false;
                if (!model.set_image(img.data, img.W, img.H)) {
                    std::fprintf(stderr, "set_image failed for %s\n", path.c_str());
                    continue;
                }
                image_loaded = true;
                std::fprintf(stderr, "▶ loaded %s (%dx%d)\n", path.c_str(), img.W, img.H);
                std::fflush(stderr);
                continue;
            }

            if (line == "/reset") {
                model.reset_conversation();
                total_pos = 0;
                image_loaded = false;
                std::fprintf(stderr, "▶ reset\n");
                continue;
            }
            if (line == "/quit" || line == "/exit") break;

            if (total_pos > KV_CACHE_LIMIT - KV_HEADROOM) {
                std::fprintf(stderr,
                             "context full (pos=%d, max=%d). /reset to continue.\n",
                             total_pos, KV_CACHE_LIMIT);
                continue;
            }

            std::vector<int32_t> ids =
                (total_pos == 0)
                ? tok.build_vlm_prompt(image_loaded, line)
                : tok.build_vlm_followup_user_turn(line);

            if (ids.empty()) {
                std::fprintf(stderr, "tokenize produced no ids — try different text\n");
                continue;
            }

            auto t_prefill_0 = std::chrono::steady_clock::now();
            std::vector<float> logits = model.forward(ids, total_pos);
            auto t_prefill_1 = std::chrono::steady_clock::now();

            std::vector<int32_t> generated;
            generated.reserve(MAX_DECODE_PER_TURN);

            auto t_decode_0 = std::chrono::steady_clock::now();
            for (int step = 0; step < MAX_DECODE_PER_TURN; ++step) {
                int next = sampler.sample(logits, generated);
                bool stop_eos = (sampler_config.eos_token_id >= 0 &&
                                 next == sampler_config.eos_token_id);
                bool stop_eou = (next == END_OF_UTTERANCE);
                if (stop_eos || stop_eou) break;
                generated.push_back(next);
                std::cout << tok.decode(std::vector<int32_t>{next}) << std::flush;
                int start_pos = total_pos + (int)ids.size() + step;
                logits = model.forward(std::vector<int32_t>{next}, start_pos);
            }
            auto t_decode_1 = std::chrono::steady_clock::now();
            std::cout << std::endl;

            total_pos += (int)ids.size() + (int)generated.size();
            turn++;

            double prefill_s = std::chrono::duration<double>(t_prefill_1 - t_prefill_0).count();
            double decode_s  = std::chrono::duration<double>(t_decode_1  - t_decode_0).count();
            double prefill_tps = prefill_s > 0 ? (double)ids.size()        / prefill_s : 0.0;
            double decode_tps  = decode_s  > 0 ? (double)generated.size()  / decode_s  : 0.0;
            std::fprintf(stderr,
                         "✓ turn %d  prefill %zu tok %.2fs (%.2f tok/s)  "
                         "decode %zu tok %.2fs (%.2f tok/s)  ctx=%d\n",
                         turn, ids.size(), prefill_s, prefill_tps,
                         generated.size(), decode_s, decode_tps, total_pos);
            std::fflush(stderr);
        }
        std::fprintf(stderr, "\nbye.\n");
        return 0;
    }

    // VLM input pipeline: when --image is provided, decode RGB8 bytes via
    // stb_image and hand them to the Model. The Model implementation (agent-
    // owned in graph mode) is expected to expose a set_image() entry point
    // that captures image bytes, runs preprocessing → vision_tower → projector
    // → splice, and stages the spliced embeddings so the subsequent token
    // loop forwards on the combined vision+text sequence.
    //
    // CONTRACT (agent must implement in src/model.cpp):
    //   bool Model::set_image(const std::vector<uint8_t>& rgb_u8,
    //                         int width, int height);
    // The runner is intentionally tiny: image load → set_image → unchanged
    // token loop. All multimodal work lives behind set_image.
    if (!image_path.empty()) {
        ImageBufferU8 img;
        if (!load_image_rgb_u8(image_path, img)) {
            NNOPT_ERROR_FMT("failed to load image %s: %s",
                            image_path.c_str(), img.error.c_str());
            return 1;
        }
        std::cerr << "  image           " << image_path
                  << " (" << img.W << "x" << img.H << ")" << std::endl;

        const auto t_before_set_image = std::chrono::steady_clock::now();
        if (!model.set_image(img.data, img.W, img.H)) {
            NNOPT_ERROR("Model::set_image failed");
            return 1;
        }
        const auto t_after_set_image = std::chrono::steady_clock::now();
        std::fprintf(stderr, "  startup         vision_pipe   %.0f ms (cumulative %.0f ms)\n",
                     std::chrono::duration<double, std::milli>(t_after_set_image - t_before_set_image).count(),
                     elapsed_ms(t_after_set_image));
    }


    // Prefer test_input_ids.bin (deterministic) over re-encoding the prompt.
    // The reference's "generated_text" field is computed from these EXACT
    // ids in PyTorch — comparing against a re-encoded prompt would mask
    // tokenizer drift as model error.
    //
    // VLM: when --image is provided and --token-ids is NOT, auto-load
    // reference/test_input_ids.bin if it exists. The processor's chat template
    // expands the image placeholder into 100s of tokens (e.g. SmolVLM emits
    // 832 image_token_id positions in a 874-token sequence). The on-device
    // tokenizer cannot reproduce that expansion — only the HF processor can.
    // Without this auto-fallback, prefill is wrong-shaped and splice has no
    // placeholders to replace.
    std::vector<int32_t> prompt_ids;
    if (!token_ids_path.empty() && read_input_ids_bin(token_ids_path, prompt_ids)) {
        // OK — explicit --token-ids override for deterministic-eval / debug parity.
    } else if (!image_path.empty()) {
        // VLM path: on-device chat template + single-tile image-placeholder block.
        // 64 IMAGE_TOKEN placeholders match the pixel_shuffle output rows so the
        // splice in backbone.cpp has the right number of positions.
        prompt_ids = tok.build_vlm_prompt(/*image_present=*/true, prompt);
    } else {
        const std::vector<int> encoded = tok.encode(prompt);
        prompt_ids.assign(encoded.begin(), encoded.end());
    }
    const size_t prompt_len = prompt_ids.size();
    // NNOPT_DUMP_PROMPT_IDS=1 prints the on-device tokenization for parity
    // checks against HF's processor (catches GPT-2 byte-level / chat-template
    // bugs that would otherwise present as model hallucinations).
    if (std::getenv("NNOPT_DUMP_PROMPT_IDS")) {
        std::fprintf(stderr, "PROMPT_IDS (%zu): [", prompt_len);
        for (size_t i = 0; i < prompt_len; ++i) {
            std::fprintf(stderr, "%d%s", prompt_ids[i], (i+1<prompt_len)?", ":"");
        }
        std::fprintf(stderr, "]\n");
    }
    std::vector<float> wav_samples;
    std::vector<float> raw_out;

        // IMPORTANT (prefill vs decode):
    // - Prefill must run ONE batched forward over the full prompt at start_pos=0.
    // - Decode must run per-token forward over ONLY the new token at start_pos=t.
    // This keeps dump shapes aligned with the PyTorch reference (prefill dumps are
    // [prompt_len, ...], decode dumps are [1, ...]).

    // Prefill (single call).
    // IMPORTANT: dump spec + PyTorch reference expect the prefill forward to run on the
    // full prompt in one shot (seq_len=prompt_len). Do NOT feed the prompt token-by-token.
    bench.mark_prefill_start();
    nnopt_profile::set_phase("prefill");
    std::vector<float> logits = model.forward(prompt_ids, /*start_pos=*/0);
    nnopt_profile::set_phase("decode");

    // Decode loop.
    std::vector<int32_t> generated_so_far; // grows with new tokens only
    generated_so_far.reserve(static_cast<size_t>(max_new_tokens));

    for (int step = 0; step < max_new_tokens; ++step) {
        int next = sampler.sample(logits, generated_so_far);

        // Benchmark hook (TTFT) — first generated token only.
        NNOPT_BENCH_FIRST_TOKEN();

        generated_so_far.push_back(next);

        if (sampler_config.eos_token_id >= 0 && next == sampler_config.eos_token_id) break;
        std::cout << tok.decode(std::vector<int32_t>{next}) << std::flush;

        // Next-step decode: feed ONLY the new token. start_pos is the absolute
        // position of this token in the full sequence (prompt_len + step).
        const int start_pos = static_cast<int>(prompt_len) + step;
        logits = model.forward(std::vector<int32_t>{next}, start_pos);
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
    bench.print_summary((int)prompt_len, (int)generated_so_far.size());
    return 0;
}
