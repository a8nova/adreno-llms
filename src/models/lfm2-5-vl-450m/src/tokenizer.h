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

    // Look up a single special-token string (e.g. "<|im_start|>") to its id.
    // Returns -1 if not present. Used by encode_with_image and any other
    // chat-template machinery that needs exact special-token ids.
    int32_t token_to_id(const std::string& token) const;

    // Produce input_ids for a single user-turn prompt with one image.
    // Reference: capture_vision_dumps.py builds:
    //   [bos, im_start] + user_ids + newline + [img_token]*n_image_tokens
    //   + text_ids + [im_end] + newline + [im_start] + assistant_ids + newline
    // where n_image_tokens = sum((h/2)*(w/2) for h,w in spatial_shapes) over
    // all tiles + optional thumbnail. The image_start/image_end/img_row/
    // img_thumbnail special tokens are NOT used in this template.
    //
    // Args:
    //   prompt: user text (e.g. "Describe this image.")
    //   grid_h, grid_w: tile grid dims (number of tiles = grid_h * grid_w)
    //   thumbnail_spatial_h, thumbnail_spatial_w: thumbnail patch dims
    //     (0 if no thumbnail)
    //   tile_tokens: tokens per tile = (tile_size/patch/downsample)^2 = 256
    //   use_thumbnail: true if thumbnail is present
    //   add_generation_prompt: append "<|im_start|>assistant\n" trailer
    std::vector<int32_t> encode_with_image(
        const std::string& prompt,
        int grid_h, int grid_w,
        int thumbnail_spatial_h, int thumbnail_spatial_w,
        int tile_tokens,
        bool use_thumbnail,
        bool add_generation_prompt) const;

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
