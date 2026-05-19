// Auto-generated graph-mode main.cpp for HuggingFaceTB/SmolVLM-256M-Instruct.
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

#include "load_image.h"  // load_image_rgb_u8 + ImageBufferU8 (stb_image)
#include "benchmark.h"
#include "profile.h"


#include <fstream>
#include <iostream>
#include <string>
#include <vector>
#include <cstdint>
#include <cstring>

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
        // NNOPT_VERSION_BANNER does not exist in debug_utils.h for this scaffold.

    // Argument parsing: positional "prompt" + optional flags.
    //   ./binary "<prompt>" [max_new_tokens] [--token-ids <file>]
    std::string prompt = "The teacher worked at the ";
    int max_new_tokens = 16;
    std::string token_ids_path;

    std::string image_path;  // --image <file.jpg|.png> for VLM inference

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
                std::cerr << "[main] loaded image " << image_path
                  << " (" << img.W << "x" << img.H << ", "
                  << img.data.size() << " bytes RGB)" << std::endl;

        if (!model.set_image(img.data, img.W, img.H)) {
            NNOPT_ERROR("Model::set_image failed");
            return 1;
        }


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
