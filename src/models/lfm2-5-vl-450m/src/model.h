#pragma once
// Auto-generated graph-mode model interface for LiquidAI/LFM2.5-VL-450M.
// Backbone class: lfm2_vl
//
// In graph mode the Model class is intentionally thin. The per-op code that
// implements the forward() data flow lives in src/ops/*.cpp, written by the
// agent one node at a time via PortNode. This file defines only the public
// interface main.cpp uses (constructor, generate-style forward).

#include "opencl_context.h"
#include "weights.h"
#include "sampler.h"
#include "tokenizer.h"
#include <vector>
#include <cstdint>

class Model {
public:
    Model(OpenCLContext& cl_ctx, Weights& weights);
    ~Model();

    // Initialize per-op state (cached programs, KV buffers, RoPE tables…).
    // Optional — graph-mode ops can lazy-init on first call. Returns true.
    bool initialize();

    // Backbone forward. Returns logits[VOCAB_SIZE] for the LAST token of
    // input_ids (matches PyTorch convention). start_pos is the absolute
    // KV-cache offset; 0 for prefill.
    std::vector<float> forward(const std::vector<int32_t>& input_ids, int start_pos);

    // Convenience wrapper used by main.cpp's generate loop.
    std::vector<float> forward(const std::vector<int32_t>& input_ids) {
        return forward(input_ids, 0);
    }

    // Greedy decode fast path: runs forward + GPU argmax, returns token ID.
    // Reads back 4 bytes instead of 128KB. Uses non-blocking readback so the
    // next iteration's enqueue overlaps with the DMA. Call read_greedy_result()
    // to wait for the previous step's result.
    bool forward_greedy(const std::vector<int32_t>& input_ids, int start_pos);
    int  read_greedy_result();
    bool init_argmax();

    bool set_image(const std::vector<uint8_t>& rgb_u8, int width, int height);
    bool has_image() const { return image_features_buf_ != nullptr && image_features_N_ > 0; }
    cl_mem image_features_buf() const { return image_features_buf_; }
    int image_features_N() const { return image_features_N_; }
    const std::vector<int32_t>& image_placeholder_positions() const { return image_placeholder_positions_; }
    const std::vector<float>& image_features() const { return image_features_; }
    int image_grid_h() const { return image_grid_h_; }
    int image_grid_w() const { return image_grid_w_; }
    int image_thumbnail_h() const { return image_thumbnail_h_; }
    int image_thumbnail_w() const { return image_thumbnail_w_; }

    // Diagnostic override: load PyTorch-computed merged text+vision
    // inputs_embeds from disk. When present, backbone.cpp uses these
    // for the first ref_inputs_embeds_N_ prefill positions instead of
    // running on-device op_Embedding + masked_scatter. Isolates LM-stack
    // bugs from vision-pipeline bugs end-to-end.
    bool load_reference_inputs_embeds(const std::string& path);
    bool has_reference_inputs_embeds() const { return ref_inputs_embeds_buf_ != nullptr && ref_inputs_embeds_N_ > 0; }
    cl_mem reference_inputs_embeds_buf() const { return ref_inputs_embeds_buf_; }
    int reference_inputs_embeds_N() const { return ref_inputs_embeds_N_; }

    // KV cache (full_attention layers) + short-conv state cache (conv layers).
    // Lazily allocated on the first prefill call; sized once for max_seq_len
    // positions (capped at 4096 by default — set higher via the bool overload
    // if a longer prompt is needed). conv_L_cache=3 → 2 pad rows cached per
    // conv layer.
    bool ensure_caches(int max_seq_len);
    cl_mem kv_K_cache(int layer_idx) const;
    cl_mem kv_V_cache(int layer_idx) const;
    cl_mem conv_bx_cache(int layer_idx) const;
    int max_kv_seq() const { return max_kv_seq_; }
    int conv_pad() const { return conv_pad_; }
    bool caches_ready() const { return max_kv_seq_ > 0; }

    // ── Track 5: counter buffer for cl_qcom_recordable_queues ──
    // Persistent 4-int buffer holding {start_pos, k_rows, _, _}. Kernels that
    // were previously taking these as scalar args now read from this buffer
    // so they're recordable (the buffer's cl_mem handle is stable across
    // replays — host just rewrites contents via clEnqueueWriteBuffer).
    // Allocated in ensure_caches; updated by update_counter() at the start of
    // every forward() call.
    cl_mem counter_buf() const { return counter_buf_; }
    bool update_counter(cl_command_queue queue, int start_pos, int k_rows);

private:
    OpenCLContext& cl_ctx_;
    Weights& weights_;
    std::vector<float> image_features_;
    cl_mem image_features_buf_ = nullptr;
    int image_features_N_ = 0;
    int image_grid_h_ = 0, image_grid_w_ = 0;
    int image_thumbnail_h_ = 0, image_thumbnail_w_ = 0;
    std::vector<int32_t> image_placeholder_positions_;
    cl_mem ref_inputs_embeds_buf_ = nullptr;
    int ref_inputs_embeds_N_ = 0;

    // KV / conv caches keyed by absolute layer index (0..NUM_HIDDEN_LAYERS-1).
    // Slots for layers of the wrong kind stay nullptr.
    std::vector<cl_mem> kv_K_cache_;
    std::vector<cl_mem> kv_V_cache_;
    std::vector<cl_mem> conv_bx_cache_;
    int max_kv_seq_ = 0;
    int conv_pad_ = 0;

    // Track 5 — counter buffer (4 ints). See counter_buf() above.
    cl_mem counter_buf_ = nullptr;

    // GPU argmax state
    cl_program argmax_prog_ = nullptr;
    cl_kernel  argmax_block_k_ = nullptr;
    cl_kernel  argmax_final_k_ = nullptr;
    cl_mem     argmax_partials_val_ = nullptr;
    cl_mem     argmax_partials_idx_ = nullptr;
    cl_mem     argmax_result_ = nullptr;
    cl_event   argmax_readback_evt_ = nullptr;
    int        argmax_result_host_ = -1;
};
