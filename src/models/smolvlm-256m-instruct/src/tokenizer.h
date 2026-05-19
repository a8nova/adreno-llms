#pragma once
// Reference: .nnport/tokenizer.json:model.vocab
// Reference: .nnport/tokenizer.json:model.merges
// Reference: .nnport/tokenizer.json:decoder
//
// Starting-point tokenizer header emitted by PortTokenizer. Iterate as needed
// for your model family — same workflow as layer files. Keep at least one
// "// Reference: .nnport/tokenizer.json:<section>" line in the first 40 lines
// (Build gate enforces this, same rule as NN layer code).

#include <cstdint>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>

class Tokenizer {
public:
    bool load(const std::string& vocab_path);

    std::vector<int32_t> encode(const std::string& text) const;
    std::string decode(const std::vector<int32_t>& token_ids) const;

    // SmolVLM single-tile chat-template + image-placeholder builder.
    // image_present=true emits 64 IMAGE_TOKEN_ID placeholders inside a global-img block
    // so the on-device splice has exactly as many positions as pixel_shuffle features.
    std::vector<int32_t> build_vlm_prompt(bool image_present,
                                         const std::string& user_text) const;

    int32_t bos_token_id() const { return bos_id_; }
    int32_t eos_token_id() const { return eos_id_; }
    int32_t pad_token_id() const { return pad_id_; }
    int32_t vocab_size() const { return static_cast<int32_t>(id_to_token_.size()); }

private:
    // Parsed vocab: index = token id, value = raw UTF-8 bytes for that token.
    std::vector<std::string> id_to_token_;
    std::unordered_map<std::string, int32_t> token_to_id_;
    // BPE merges (a, b) → ab. Order matters; first match in this list wins.
    std::vector<std::pair<std::string, std::string>> merges_;

    int32_t bos_id_ = -1;
    int32_t eos_id_ = -1;
    int32_t pad_id_ = -1;
};
