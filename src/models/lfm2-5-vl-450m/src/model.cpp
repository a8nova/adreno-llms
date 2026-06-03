// Auto-generated graph-mode model implementation for LiquidAI/LFM2.5-VL-450M.
// Backbone: lfm2_vl | total nodes captured: 0
//
// FRAMEWORK FILE — thin Model wrapper around graph-mode ops.

#include "model.h"
#include "model_config.h"
#include "debug_utils.h"
#include "forward_dispatch.h"
#include "utils.h"
#include "ops/lfm2_vl_image_processor.h"
#include "ops/lfm2_common.h"  // lfm2_pool_set_active / clear — vision-tower prefill churn

#include <CL/cl.h>
#include <vector>
#include <cstdint>
#include <fstream>
#include <string>

Model* g_active_model_for_vlm_splice = nullptr;

std::vector<float> model_forward_graph(
    OpenCLContext& cl_ctx,
    Weights& weights,
    const std::vector<int32_t>& input_ids,
    int start_pos);

bool vision_pipeline_forward(
    OpenCLContext& cl_ctx,
    Weights& weights,
    const std::vector<uint8_t>& rgb_u8,
    int W,
    int H,
    std::vector<float>& image_features_out);

namespace {

static void parse_placeholder_positions(const std::string& path,
                                        std::vector<int32_t>& out) {
    out.clear();
    std::ifstream f(path);
    if (!f) {
        // Reference: model_info/transformers_src/modeling_lfm2_vl.py:244-276
        // Placeholder positions are optional: on-device runs may not deploy the
        // JSON reference file, and backbone.cpp can derive the same mask from
        // input_ids == IMAGE_TOKEN_ID. Do not emit ERROR for this non-fatal path.
        NNOPT_CHECKPOINT_FMT("Model::initialize: optional %s unavailable; deriving image mask from input_ids", path.c_str());
        return;
    }
    const std::string s((std::istreambuf_iterator<char>(f)), std::istreambuf_iterator<char>());
    const std::string key = "\"image_placeholder_positions\"";
    size_t p = s.find(key);
    if (p == std::string::npos) return;
    p = s.find('[', p);
    if (p == std::string::npos) return;
    ++p;
    while (p < s.size()) {
        while (p < s.size() && (s[p] == ' ' || s[p] == '\n' || s[p] == '\r' || s[p] == '\t' || s[p] == ',')) ++p;
        if (p >= s.size() || s[p] == ']') break;
        bool neg = false;
        if (s[p] == '-') { neg = true; ++p; }
        int v = 0;
        bool any = false;
        while (p < s.size() && s[p] >= '0' && s[p] <= '9') {
            any = true;
            v = v * 10 + (s[p] - '0');
            ++p;
        }
        if (any) out.push_back((int32_t)(neg ? -v : v));
        while (p < s.size() && s[p] != ',' && s[p] != ']') ++p;
    }
}

}  // namespace

Model::Model(OpenCLContext& cl_ctx, Weights& weights)
    : cl_ctx_(cl_ctx), weights_(weights) {}

Model::~Model() {
    if (image_features_buf_) {
        clReleaseMemObject(image_features_buf_);
        image_features_buf_ = nullptr;
    }
    if (ref_inputs_embeds_buf_) {
        clReleaseMemObject(ref_inputs_embeds_buf_);
        ref_inputs_embeds_buf_ = nullptr;
    }
    for (cl_mem b : kv_K_cache_) if (b) clReleaseMemObject(b);
    for (cl_mem b : kv_V_cache_) if (b) clReleaseMemObject(b);
    for (cl_mem b : conv_bx_cache_) if (b) clReleaseMemObject(b);
    kv_K_cache_.clear();
    kv_V_cache_.clear();
    conv_bx_cache_.clear();
    if (counter_buf_) { clReleaseMemObject(counter_buf_); counter_buf_ = nullptr; }
    if (g_active_model_for_vlm_splice == this) g_active_model_for_vlm_splice = nullptr;
}

bool Model::ensure_caches(int max_seq_len) {
    if (max_kv_seq_ >= max_seq_len) return true;  // already big enough
    if (max_kv_seq_ > 0) {
        // Re-allocate at the larger size.
        for (cl_mem b : kv_K_cache_) if (b) clReleaseMemObject(b);
        for (cl_mem b : kv_V_cache_) if (b) clReleaseMemObject(b);
        for (cl_mem b : conv_bx_cache_) if (b) clReleaseMemObject(b);
        kv_K_cache_.clear();
        kv_V_cache_.clear();
        conv_bx_cache_.clear();
    }
    const int num_layers = MODEL_CONFIG::NUM_HIDDEN_LAYERS;
    const int hidden = MODEL_CONFIG::HIDDEN_SIZE;
    const int q_heads = MODEL_CONFIG::NUM_ATTENTION_HEADS;
    const int kv_heads = MODEL_CONFIG::NUM_KEY_VALUE_HEADS;
    const int head_dim = hidden / q_heads;
    const int ksize = MODEL_CONFIG::CONV_L_CACHE;  // 3 → pad=2
    const int pad = ksize - 1;
    // Layer kinds from contract: 0=conv, 1=full_attention. Mirror backbone.cpp.
    static const int LAYER_KINDS[16] = {0, 0, 1, 0, 0, 1, 0, 0, 1, 0, 1, 0, 1, 0, 1, 0};
    kv_K_cache_.assign((size_t)num_layers, nullptr);
    kv_V_cache_.assign((size_t)num_layers, nullptr);
    conv_bx_cache_.assign((size_t)num_layers, nullptr);
    cl_int err = CL_SUCCESS;
    for (int i = 0; i < num_layers; ++i) {
        if (LAYER_KINDS[i] == 1) {
            // [kv_heads, max_seq_len, head_dim] in nnopt_storage_t
            const size_t bytes = (size_t)kv_heads * (size_t)max_seq_len * (size_t)head_dim * sizeof(nnopt_storage_t);
            kv_K_cache_[i] = clCreateBuffer(cl_ctx_.context(), CL_MEM_READ_WRITE, bytes, nullptr, &err);
            if (err != CL_SUCCESS || !kv_K_cache_[i]) {
                NNOPT_ERROR_FMT("Model::ensure_caches: kv_K[%d] alloc (%zu bytes) failed (%d)", i, bytes, (int)err);
                return false;
            }
            kv_V_cache_[i] = clCreateBuffer(cl_ctx_.context(), CL_MEM_READ_WRITE, bytes, nullptr, &err);
            if (err != CL_SUCCESS || !kv_V_cache_[i]) {
                NNOPT_ERROR_FMT("Model::ensure_caches: kv_V[%d] alloc failed (%d)", i, (int)err);
                return false;
            }
        } else {
            // [pad, hidden] in nnopt_storage_t
            const size_t bytes = (size_t)pad * (size_t)hidden * sizeof(nnopt_storage_t);
            conv_bx_cache_[i] = clCreateBuffer(cl_ctx_.context(), CL_MEM_READ_WRITE, bytes, nullptr, &err);
            if (err != CL_SUCCESS || !conv_bx_cache_[i]) {
                NNOPT_ERROR_FMT("Model::ensure_caches: conv_bx[%d] alloc failed (%d)", i, (int)err);
                return false;
            }
        }
    }
    max_kv_seq_ = max_seq_len;
    conv_pad_ = pad;

    // Track 5 — allocate counter buffer (4 ints, persistent for process lifetime).
    if (!counter_buf_) {
        counter_buf_ = clCreateBuffer(cl_ctx_.context(), CL_MEM_READ_WRITE,
                                      4 * sizeof(int), nullptr, &err);
        if (err != CL_SUCCESS || !counter_buf_) {
            NNOPT_ERROR_FMT("Model::ensure_caches: counter_buf alloc failed (%d)", (int)err);
            return false;
        }
        int zero4[4] = {0, 0, 0, 0};
        clEnqueueWriteBuffer(cl_ctx_.queue(), counter_buf_, CL_TRUE, 0,
                             sizeof(zero4), zero4, 0, nullptr, nullptr);
    }

    NNOPT_CHECKPOINT_FMT("Model::ensure_caches: allocated KV cache (max_seq=%d, kv_dim=%d) + conv bx cache (pad=%d, hidden=%d)",
                         max_seq_len, kv_heads * head_dim, pad, hidden);
    return true;
}

bool Model::update_counter(cl_command_queue queue, int start_pos, int k_rows) {
    if (!counter_buf_) {
        NNOPT_ERROR("Model::update_counter: counter_buf_ not allocated (call ensure_caches first)");
        return false;
    }
    int data[4] = { start_pos, k_rows, 0, 0 };
    cl_int err = clEnqueueWriteBuffer(queue, counter_buf_, CL_FALSE, 0,
                                      sizeof(data), data, 0, nullptr, nullptr);
    if (err != CL_SUCCESS) {
        NNOPT_ERROR_FMT("Model::update_counter: clEnqueueWriteBuffer failed (%d)", (int)err);
        return false;
    }
    return true;
}

cl_mem Model::kv_K_cache(int layer_idx) const {
    if (layer_idx < 0 || (size_t)layer_idx >= kv_K_cache_.size()) return nullptr;
    return kv_K_cache_[(size_t)layer_idx];
}
cl_mem Model::kv_V_cache(int layer_idx) const {
    if (layer_idx < 0 || (size_t)layer_idx >= kv_V_cache_.size()) return nullptr;
    return kv_V_cache_[(size_t)layer_idx];
}
cl_mem Model::conv_bx_cache(int layer_idx) const {
    if (layer_idx < 0 || (size_t)layer_idx >= conv_bx_cache_.size()) return nullptr;
    return conv_bx_cache_[(size_t)layer_idx];
}

bool Model::load_reference_inputs_embeds(const std::string& path) {
    if (ref_inputs_embeds_buf_) {
        clReleaseMemObject(ref_inputs_embeds_buf_);
        ref_inputs_embeds_buf_ = nullptr;
        ref_inputs_embeds_N_ = 0;
    }
    std::ifstream f(path, std::ios::binary | std::ios::ate);
    if (!f) {
        NNOPT_CHECKPOINT_FMT("Model::load_reference_inputs_embeds: %s not present (on-device path)", path.c_str());
        return false;
    }
    const std::streamsize bytes = f.tellg();
    if (bytes <= 0) return false;
    f.seekg(0, std::ios::beg);
    const int hidden = MODEL_CONFIG::HIDDEN_SIZE;
    const size_t row_floats = (size_t)hidden;
    if ((size_t)bytes % (row_floats * sizeof(float)) != 0) {
        NNOPT_ERROR_FMT("Model::load_reference_inputs_embeds: size %lld not a multiple of hidden*fp32 (%zu)",
                        (long long)bytes, row_floats * sizeof(float));
        return false;
    }
    const int N = (int)((size_t)bytes / (row_floats * sizeof(float)));
    std::vector<float> host_fp32((size_t)N * row_floats);
    if (!f.read(reinterpret_cast<char*>(host_fp32.data()), bytes)) {
        NNOPT_ERROR_FMT("Model::load_reference_inputs_embeds: short read on %s", path.c_str());
        return false;
    }
    std::vector<nnopt_storage_t> storage(host_fp32.size());
    for (size_t i = 0; i < host_fp32.size(); ++i) {
#ifdef NNOPT_USE_FP16
        storage[i] = (nnopt_storage_t)nnopt_f32_to_f16(host_fp32[i]);
#else
        storage[i] = host_fp32[i];
#endif
    }
    cl_int err = CL_SUCCESS;
    ref_inputs_embeds_buf_ = clCreateBuffer(
        cl_ctx_.context(), CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
        storage.size() * sizeof(nnopt_storage_t), storage.data(), &err);
    if (err != CL_SUCCESS || !ref_inputs_embeds_buf_) {
        NNOPT_ERROR_FMT("Model::load_reference_inputs_embeds: clCreateBuffer failed (%d)", (int)err);
        ref_inputs_embeds_buf_ = nullptr;
        return false;
    }
    ref_inputs_embeds_N_ = N;
    NNOPT_CHECKPOINT_FMT("Model::load_reference_inputs_embeds: loaded %d rows from %s — vision pipeline + masked_scatter will be bypassed for first %d positions",
                         N, path.c_str(), N);
    return true;
}

bool Model::initialize() {
    NNOPT_CHECKPOINT("Model::initialize() — graph mode");
    parse_placeholder_positions("reference/reference_tokens.json", image_placeholder_positions_);
    g_active_model_for_vlm_splice = this;
    return true;
}

bool Model::set_image(const std::vector<uint8_t>& rgb_u8, int width, int height) {
    NNOPT_CHECKPOINT("Model::set_image() — VLM multi-tile pipeline");
    image_features_.clear();
    if (image_features_buf_) {
        clReleaseMemObject(image_features_buf_);
        image_features_buf_ = nullptr;
    }
    image_features_N_ = 0;
    image_grid_h_ = 0;
    image_grid_w_ = 0;
    image_thumbnail_h_ = 0;
    image_thumbnail_w_ = 0;

    // Enable the lfm2 buffer pool around the vision tower prefill. Each tile
    // through siglip_vision_forward_tile/batched allocates ~12 transient
    // buffers per layer × 12 layers ≈ ~145/tile (~1000 across all 7 tiles for
    // per-tile path; ~150 for batched + 1 thumbnail). Pool amortizes them.
    // We CLEAR the pool at the end — vision-sized buffers (~MB-class) would
    // OOM during LM prefill if left in the free list.
    lfm2_pool_set_active(true);
    auto vision_pool_guard = [&]() {
        lfm2_pool_set_active(false);
        lfm2_pool_clear();
    };

    // Step 1: multi-tile image processor (CPU-side resize + tile splitting)
    Lfm2VlImageProcessorOutput proc_out;
    if (!lfm2_vl_preprocess_image(rgb_u8, height, width, proc_out)) {
        NNOPT_ERROR("Model::set_image: image preprocessor failed");
        vision_pool_guard();
        return false;
    }
    image_grid_h_ = proc_out.grid_h;
    image_grid_w_ = proc_out.grid_w;
    image_thumbnail_h_ = proc_out.thumbnail_h / MODEL_CONFIG::ENCODER_PATCH_SIZE;
    image_thumbnail_w_ = proc_out.thumbnail_w / MODEL_CONFIG::ENCODER_PATCH_SIZE;
    NNOPT_CHECKPOINT_FMT("Model::set_image: %d tiles, grid=%dx%d, thumbnail=%dx%d",
                         (int)proc_out.tiles.size(), proc_out.grid_h, proc_out.grid_w,
                         proc_out.thumbnail_h, proc_out.thumbnail_w);

    // ── GPU-resident features: pre-allocate image_features_buf_ + compute per-tile
    // byte offsets. The encoder writes fp16 directly into this buffer; we skip
    // the per-tile fp16→fp32 readback + host→GPU fp32→fp16 reupload cycle.
    std::vector<size_t> tile_dest_offsets(proc_out.tiles.size());
    size_t cumulative_bytes = 0;
    for (size_t t = 0; t < proc_out.tiles.size(); ++t) {
        const auto& tile = proc_out.tiles[t];
        const size_t tokens = (size_t)(tile.spatial_h / 2) * (size_t)(tile.spatial_w / 2);
        tile_dest_offsets[t] = cumulative_bytes;
        cumulative_bytes += tokens * (size_t)MODEL_CONFIG::HIDDEN_SIZE * sizeof(nnopt_storage_t);
    }
    {
        cl_int alloc_err = CL_SUCCESS;
        image_features_buf_ = clCreateBuffer(cl_ctx_.context(), CL_MEM_READ_WRITE,
                                             cumulative_bytes, nullptr, &alloc_err);
        if (alloc_err != CL_SUCCESS || !image_features_buf_) {
            NNOPT_ERROR_FMT("Model::set_image: image_features_buf_ alloc (%zu B) failed (%d)",
                            cumulative_bytes, (int)alloc_err);
            image_features_buf_ = nullptr;
            vision_pool_guard(); return false;
        }
    }
    extern cl_mem g_siglip_gpu_dest_buf;
    extern size_t g_siglip_gpu_dest_offset;
    extern std::vector<size_t>* g_siglip_gpu_dest_offsets_batched;
    g_siglip_gpu_dest_buf = image_features_buf_;

    // Step 2: run SigLIP encoder + projector on the tiles. Group tiles into
    // a "full" set (all sharing a single spatial shape — the dominant case
    // is 6× 32x32 tiles) and process them in a single batched encoder call
    // so the per-token GEMMs collapse from N_tiles CLBlast invocations into
    // one. The thumbnail (or any tile with a different shape) keeps the
    // per-tile fallback. Feature concatenation order MUST match the original
    // PyTorch order (proc_out.tiles already row-major + thumbnail last).
    image_features_.clear();

    // Per-tile result staging — indexed in input order so we can concat in
    // PyTorch order regardless of which branch produced each tile.
    std::vector<std::vector<float>> tile_results(proc_out.tiles.size());

    // Identify the dominant "full" shape (the most common (spatial_h, spatial_w)
    // among non-thumbnail tiles). For the standard 32x32 case all main tiles
    // share that exact shape. Any tile whose spatial dims diverge falls back
    // to the per-tile path.
    int batch_spatial_h = 0, batch_spatial_w = 0;
    for (const auto& tile : proc_out.tiles) {
        if (!tile.is_thumbnail) {
            batch_spatial_h = tile.spatial_h;
            batch_spatial_w = tile.spatial_w;
            break;
        }
    }

    std::vector<const Lfm2VlTile*> batched_ptrs;
    std::vector<size_t> batched_indices;  // index into tile_results
    std::vector<size_t> fallback_indices;
    for (size_t t = 0; t < proc_out.tiles.size(); ++t) {
        const auto& tile = proc_out.tiles[t];
        if (!tile.is_thumbnail && tile.spatial_h == batch_spatial_h &&
            tile.spatial_w == batch_spatial_w && batch_spatial_h > 0) {
            batched_ptrs.push_back(&tile);
            batched_indices.push_back(t);
        } else {
            fallback_indices.push_back(t);
        }
    }

    if (batched_ptrs.size() >= 2) {
        NNOPT_CHECKPOINT_FMT("Model::set_image: batched encoder pass over %zu tiles (%dx%d)",
                             batched_ptrs.size(), batch_spatial_h, batch_spatial_w);
        std::vector<std::vector<float>> batched_results;
        // Set the GPU-direct offsets for the batched encoder.
        std::vector<size_t> batched_offsets(batched_indices.size());
        for (size_t i = 0; i < batched_indices.size(); ++i)
            batched_offsets[i] = tile_dest_offsets[batched_indices[i]];
        g_siglip_gpu_dest_offsets_batched = &batched_offsets;
        if (!siglip_vision_forward_batched(cl_ctx_, weights_, batched_ptrs, batched_results)) {
            g_siglip_gpu_dest_offsets_batched = nullptr;
            g_siglip_gpu_dest_buf = nullptr;
            NNOPT_ERROR("Model::set_image: batched SigLIP encoder failed");
            vision_pool_guard(); return false;
        }
        if (batched_results.size() != batched_indices.size()) {
            NNOPT_ERROR_FMT("Model::set_image: batched encoder returned %zu, expected %zu",
                            batched_results.size(), batched_indices.size());
            vision_pool_guard(); return false;
        }
        for (size_t i = 0; i < batched_indices.size(); ++i) {
            tile_results[batched_indices[i]] = std::move(batched_results[i]);
        }
    } else {
        // Fewer than 2 tiles in the "full" group — batching has no benefit;
        // route them through the per-tile fallback alongside the thumbnail.
        for (size_t idx : batched_indices) fallback_indices.push_back(idx);
        batched_indices.clear();
    }

    g_siglip_gpu_dest_offsets_batched = nullptr;  // batched path done
    for (size_t t : fallback_indices) {
        const auto& tile = proc_out.tiles[t];
        g_siglip_gpu_dest_offset = tile_dest_offsets[t];
        std::vector<float> tile_features;
        if (!siglip_vision_forward_tile(cl_ctx_, weights_, tile.rgb,
                                        tile.H_px, tile.W_px,
                                        tile.spatial_h, tile.spatial_w,
                                        tile_features)) {
            NNOPT_ERROR_FMT("Model::set_image: SigLIP encoder failed on tile %zu", t);
            g_siglip_gpu_dest_buf = nullptr;
            vision_pool_guard(); return false;
        }
        tile_results[t] = std::move(tile_features);
    }
    // Clear GPU output hooks — vision encoder is done writing to image_features_buf_.
    g_siglip_gpu_dest_buf = nullptr;
    g_siglip_gpu_dest_offsets_batched = nullptr;
    // Release pool before LM prefill (vision-sized buffers would OOM).
    vision_pool_guard();

    // tile_results may be empty for tiles that went via GPU-direct path;
    // skip the size check when image_features_buf_ was used directly.
    for (size_t t = 0; t < proc_out.tiles.size(); ++t) {
        const auto& tile = proc_out.tiles[t];
        const size_t tile_tokens = (size_t)(tile.spatial_h / 2) * (size_t)(tile.spatial_w / 2);
        const size_t expected = tile_tokens * (size_t)MODEL_CONFIG::HIDDEN_SIZE;
        if (!tile_results[t].empty() && tile_results[t].size() != expected) {
            NNOPT_ERROR_FMT("Model::set_image: tile %zu produced %zu features, expected %zu",
                            t, tile_results[t].size(), expected);
            return false;
        }
        NNOPT_CHECKPOINT_FMT("Model::set_image: tile %zu produced %zu tokens", t, tile_tokens);
        // image_features_ host vector is now empty if GPU-resident path was used.
        // Still accumulate for tiles that didn't use GPU path (none in current flow).
        if (!tile_results[t].empty()) {
            image_features_.insert(image_features_.end(),
                                   tile_results[t].begin(), tile_results[t].end());
        }
    }

    // Image features count = total bytes / (HIDDEN_SIZE * sizeof(storage_t))
    image_features_N_ = (int)(cumulative_bytes / ((size_t)MODEL_CONFIG::HIDDEN_SIZE * sizeof(nnopt_storage_t)));
    if (image_features_N_ <= 0) {
        NNOPT_ERROR("Model::set_image: zero features produced");
        return false;
    }
    NNOPT_CHECKPOINT_FMT("Model::set_image: %d image_features ready (GPU-resident, %zu bytes)",
                         image_features_N_, cumulative_bytes);
    return true;
}

std::vector<float> Model::forward(const std::vector<int32_t>& input_ids, int start_pos) {
    NNOPT_CHECKPOINT("Model::forward() — graph mode (delegating to model_forward_graph)");
    g_active_model_for_vlm_splice = this;
    return model_forward_graph(cl_ctx_, weights_, input_ids, start_pos);
}

// ── GPU argmax for greedy decode ──────────────────────────────────────────────

cl_mem model_forward_graph_logits_buf(
    OpenCLContext& cl_ctx,
    Weights& weights,
    const std::vector<int32_t>& input_ids,
    int start_pos);

bool Model::init_argmax() {
    if (argmax_prog_) return true;
    cl_context ctx = cl_ctx_.context();
    cl_device_id dev = nullptr;
    clGetCommandQueueInfo(cl_ctx_.queue(), CL_QUEUE_DEVICE, sizeof(dev), &dev, nullptr);

    std::ifstream f("kernels/argmax.cl", std::ios::binary);
    if (!f.is_open()) return false;
    std::string src((std::istreambuf_iterator<char>(f)), std::istreambuf_iterator<char>());
    const char* s = src.c_str();
    size_t len = src.size();
    cl_int err;
    argmax_prog_ = clCreateProgramWithSource(ctx, 1, &s, &len, &err);
    if (err != CL_SUCCESS) return false;
    err = clBuildProgram(argmax_prog_, 1, &dev, "-cl-fast-relaxed-math", nullptr, nullptr);
    if (err != CL_SUCCESS) {
        clReleaseProgram(argmax_prog_); argmax_prog_ = nullptr; return false;
    }
    argmax_block_k_ = clCreateKernel(argmax_prog_, "argmax_block", &err);
    if (err != CL_SUCCESS) { argmax_block_k_ = nullptr; return false; }
    argmax_final_k_ = clCreateKernel(argmax_prog_, "argmax_final", &err);
    if (err != CL_SUCCESS) { argmax_final_k_ = nullptr; return false; }

    const int V = MODEL_CONFIG::VOCAB_SIZE;
    const int CHUNK = 1024;
    const int num_wg = (V + CHUNK - 1) / CHUNK;
    argmax_partials_val_ = clCreateBuffer(ctx, CL_MEM_READ_WRITE, (size_t)num_wg * sizeof(float), nullptr, &err);
    argmax_partials_idx_ = clCreateBuffer(ctx, CL_MEM_READ_WRITE, (size_t)num_wg * sizeof(int), nullptr, &err);
    argmax_result_ = clCreateBuffer(ctx, CL_MEM_READ_WRITE, sizeof(int), nullptr, &err);
    return argmax_partials_val_ && argmax_partials_idx_ && argmax_result_;
}

bool Model::forward_greedy(const std::vector<int32_t>& input_ids, int start_pos) {
    g_active_model_for_vlm_splice = this;

    cl_mem logits_buf = model_forward_graph_logits_buf(cl_ctx_, weights_, input_ids, start_pos);
    if (!logits_buf) return false;

    if (!init_argmax()) {
        clReleaseMemObject(logits_buf);
        return false;
    }

    cl_command_queue queue = cl_ctx_.queue();
    const int V = MODEL_CONFIG::VOCAB_SIZE;
    const int CHUNK = 1024;
    const int num_wg = (V + CHUNK - 1) / CHUNK;

    clSetKernelArg(argmax_block_k_, 0, sizeof(cl_mem), &logits_buf);
    clSetKernelArg(argmax_block_k_, 1, sizeof(cl_mem), &argmax_partials_val_);
    clSetKernelArg(argmax_block_k_, 2, sizeof(cl_mem), &argmax_partials_idx_);
    clSetKernelArg(argmax_block_k_, 3, sizeof(int), &V);
    size_t gws1 = (size_t)num_wg * 64;
    size_t lws1 = 64;
    clEnqueueNDRangeKernel(queue, argmax_block_k_, 1, nullptr, &gws1, &lws1, 0, nullptr, nullptr);

    clSetKernelArg(argmax_final_k_, 0, sizeof(cl_mem), &argmax_partials_val_);
    clSetKernelArg(argmax_final_k_, 1, sizeof(cl_mem), &argmax_partials_idx_);
    clSetKernelArg(argmax_final_k_, 2, sizeof(cl_mem), &argmax_result_);
    clSetKernelArg(argmax_final_k_, 3, sizeof(int), &num_wg);
    size_t gws2 = 64;
    size_t lws2 = 64;
    clEnqueueNDRangeKernel(queue, argmax_final_k_, 1, nullptr, &gws2, &lws2, 0, nullptr, nullptr);

    if (argmax_readback_evt_) { clReleaseEvent(argmax_readback_evt_); argmax_readback_evt_ = nullptr; }
    clEnqueueReadBuffer(queue, argmax_result_, CL_FALSE, 0, sizeof(int),
                        &argmax_result_host_, 0, nullptr, &argmax_readback_evt_);
    clReleaseMemObject(logits_buf);
    return true;
}

int Model::read_greedy_result() {
    if (argmax_readback_evt_) {
        clWaitForEvents(1, &argmax_readback_evt_);
        clReleaseEvent(argmax_readback_evt_);
        argmax_readback_evt_ = nullptr;
    }
    return argmax_result_host_;
}
