// Auto-generated graph-mode model implementation for facebook/musicgen-small.
// Backbone: musicgen | total nodes captured: 0
//
// FRAMEWORK FILE — DO NOT EDIT (the agent restructures the encode/decode
// methods inside this file when wiring main.cpp for enc-dec models).
// Model::forward() delegates to model_forward_graph(...) which is provided
// by the agent in src/backbone.cpp.

#include "model.h"
#include "model_config.h"
#include "debug_utils.h"
#include "forward_dispatch.h"
#include "utils.h"
#include "profiler.h"

#include <vector>
#include <cstdint>
#include <cstdlib>

#include "text_encoder.h"

std::vector<float> model_forward_graph(
    OpenCLContext& cl_ctx,
    Weights& weights,
    const std::vector<int32_t>& input_ids,
    int start_pos);
extern "C" void model_forward_graph_logits_dev(
    OpenCLContext& cl_ctx,
    Weights& weights,
    const std::vector<int32_t>& input_ids,
    int start_pos,
    cl_mem* logits_dev_out);
extern "C" bool model_set_encoder_states(OpenCLContext& cl_ctx, const float* states, int T, int dim);
extern "C" void model_reset_decode_state();
extern "C" void model_set_cfg_branch(int uncond);
std::vector<float> model_forward_graph_cfg_m2(
    OpenCLContext& cl_ctx,
    Weights& weights,
    const std::vector<int32_t>& input_ids,
    int start_pos,
    float guidance);
extern "C" bool model_forward_graph_cfg_m2_sampled(
    OpenCLContext& cl_ctx, Weights& weights,
    const std::vector<int32_t>& input_ids, int start_pos, float guidance,
    float temperature, int top_k, uint32_t seed, int force_argmax,
    int32_t* out_ids);
extern "C" cl_mem mega_alloc_decode_grid(
    OpenCLContext& cl_ctx, const int32_t* host_grid, int num_codebooks, int steps1, int bos);
extern "C" bool mega_read_decode_grid(
    OpenCLContext& cl_ctx, cl_mem grid_dev, int32_t* host_grid, int num_codebooks, int steps1);
extern "C" void mega_free_decode_grid(cl_mem grid_dev);

Model::Model(OpenCLContext& cl_ctx, Weights& weights)
    : cl_ctx_(cl_ctx), weights_(weights) {}

Model::~Model() {
}

bool Model::initialize() {
    NNOPT_CHECKPOINT("Model::initialize() — graph mode");
    return true;
}

bool Model::encode_text(const std::vector<int32_t>& text_ids) {
    NNOPT_CHECKPOINT("Model::encode_text() — host T5 + enc_to_dec_proj");
    const std::vector<float> states = t5_encode_host(weights_, text_ids);
    return apply_encoder_states(states);
}

bool Model::apply_encoder_states(const std::vector<float>& states) {
    if (states.empty()) return false;
    model_reset_decode_state();   // new generation → fresh KV caches (also
                                  // wipes any prewarm_decoder() dummy state)
    const int dim = 1024;  // enc_to_dec_proj output (== DECODER_HIDDEN_SIZE)
    const int T = (int)(states.size() / dim);
    return model_set_encoder_states(cl_ctx_, states.data(), T, dim);
}

bool Model::prewarm_decoder(int enc_len) {
    NNOPT_CHECKPOINT("Model::prewarm_decoder() — zero-state dummy step");
    if (enc_len <= 0) return false;
    const int dim = 1024;
    std::vector<float> zeros((size_t)enc_len * dim, 0.0f);
    if (!model_set_encoder_states(cl_ctx_, zeros.data(), enc_len, dim)) return false;
    std::vector<int32_t> bos((size_t)MODEL_CONFIG::NUM_CODEBOOKS,
                             (int32_t)MODEL_CONFIG::BOS_TOKEN_ID);
    int32_t discard[8] = {0};
    // Params mirror a real step-0 call shape; outputs and all KV/grid writes
    // are discarded/overwritten by the real generation.
    return forward_cfg_sampled(bos, /*start_pos=*/0, /*guidance=*/3.0f,
                               /*temperature=*/1.0f, /*top_k=*/250,
                               /*seed=*/42u, /*force_argmax=*/0, discard);
}

std::vector<float> Model::forward_cfg(const std::vector<int32_t>& input_ids, int start_pos, float guidance) {
    if (guidance <= 1.0f) return forward(input_ids, start_pos);

    const int N = MODEL_CONFIG::NUM_CODEBOOKS * MODEL_CONFIG::VOCAB_SIZE;

    // ── Full M=2 CFG GEMM batching (BENCHMARKS Stage 1) ────────────────────
    // Runs BOTH CFG rows in ONE decoder pass over a [2,hidden] buffer: every
    // row-wise op (embeddings/pos, LayerNorm, fc1/GELU/fc2, residuals, lm_heads
    // GEMM) is batched M=2 — halving the non-attention dispatch overhead that
    // the profile proved is the sole bottleneck. Attention is per-row (own KV
    // bank + encoder states), math byte-identical. Falls back to the interleaved
    // route (and then the host two-pass) if the batched path reports an error.
    // NNOPT_NO_M2=1 forces the interleaved (non-batched) route — used ONLY for
    // controlled A/B benchmarking of M=2 vs interleaved under identical thermal
    // state on the same binary (DVFS makes cross-binary comparison unreliable).
    static const bool no_m2 = [](){ const char* e = std::getenv("NNOPT_NO_M2"); return e && e[0]=='1'; }();
    if (!no_m2) {
        std::vector<float> m2 = model_forward_graph_cfg_m2(cl_ctx_, weights_, input_ids, start_pos, guidance);
        if ((int)m2.size() == N) return m2;
        NNOPT_ERROR("forward_cfg: M=2 batched path failed — falling back to interleaved");
    }

    // ── Interleaved CFG (BENCHMARKS #2) ────────────────────────────────────
    // The two CFG passes were fully host-serialized: the cond pass enqueued its
    // ~600 dispatches AND blocked on a logits readback before the uncond pass
    // was even enqueued. With GPU busy only ~4% of wall, that mid-CFG drain is
    // pure dead time. Here we enqueue BOTH passes' dispatches into the in-order
    // queue (cond → its own logits buffer, uncond → its own buffer, no read
    // between), combine on-device (cfg_combine), and read the result ONCE. The
    // per-pass math is byte-identical to before — same kernels, same dual KV
    // banks (cond=bank0/g_enc_states, uncond=bank32/g_enc_zero).
    cl_mem cond_dev = nullptr, uncond_dev = nullptr;
    model_set_cfg_branch(0);
    model_forward_graph_logits_dev(cl_ctx_, weights_, input_ids, start_pos, &cond_dev);
    model_set_cfg_branch(1);
    model_forward_graph_logits_dev(cl_ctx_, weights_, input_ids, start_pos, &uncond_dev);
    model_set_cfg_branch(0);

    if (cond_dev && uncond_dev) {
        cl_command_queue queue = cl_ctx_.queue();
        cl_program utils_prog = cl_ctx_.get_utils_program();
        cl_int err = CL_SUCCESS;
        cl_mem comb = clCreateBuffer(cl_ctx_.context(), CL_MEM_READ_WRITE,
                                     (size_t)N * sizeof(nnopt_storage_t), nullptr, &err);
        if (err == CL_SUCCESS && comb && utils_prog) {
            static cl_kernel kc = nullptr;
            if (!kc) kc = clCreateKernel(utils_prog, "cfg_combine", &err);
            if (kc) {
                clSetKernelArg(kc, 0, sizeof(cl_mem), &cond_dev);
                clSetKernelArg(kc, 1, sizeof(cl_mem), &uncond_dev);
                clSetKernelArg(kc, 2, sizeof(cl_mem), &comb);
                clSetKernelArg(kc, 3, sizeof(float), &guidance);
                clSetKernelArg(kc, 4, sizeof(int), &N);
                size_t gws = (size_t)N;
                err = clEnqueueNDRangeKernel(queue, kc, 1, nullptr, &gws, nullptr, 0, nullptr,
                                             KernelProfiler::event_for("cfg_combine"));
            }
            std::vector<float> out((size_t)N, 0.0f);
            if (err == CL_SUCCESS) {
#ifdef NNOPT_USE_FP16
                std::vector<uint16_t> tmp((size_t)N);
                if (clEnqueueReadBuffer(queue, comb, CL_TRUE, 0, (size_t)N * sizeof(uint16_t), tmp.data(), 0, nullptr, nullptr) == CL_SUCCESS)
                    for (int i = 0; i < N; ++i) out[(size_t)i] = nnopt_f16_to_f32(tmp[(size_t)i]);
#else
                clEnqueueReadBuffer(queue, comb, CL_TRUE, 0, (size_t)N * sizeof(float), out.data(), 0, nullptr, nullptr);
#endif
            }
            clReleaseMemObject(comb);
            clReleaseMemObject(cond_dev);
            clReleaseMemObject(uncond_dev);
            return out;
        }
        if (comb) clReleaseMemObject(comb);
    }
    if (cond_dev) clReleaseMemObject(cond_dev);
    if (uncond_dev) clReleaseMemObject(uncond_dev);

    // Fallback: host two-pass (concat-GEMM unavailable). Math unchanged.
    model_set_cfg_branch(0);
    std::vector<float> cond = forward(input_ids, start_pos);
    model_set_cfg_branch(1);
    std::vector<float> uncond = forward(input_ids, start_pos);
    model_set_cfg_branch(0);
    if (cond.size() != uncond.size() || cond.empty()) return {};
    for (size_t i = 0; i < cond.size(); ++i)
        cond[i] = uncond[i] + guidance * (cond[i] - uncond[i]);
    return cond;
}

bool Model::forward_cfg_sampled(const std::vector<int32_t>& input_ids, int start_pos,
                                float guidance, float temperature, int top_k,
                                uint32_t seed, int force_argmax, int32_t* out_ids) {
    // guidance <= 1.0 now valid: CFG-early single-row mode (cond row only,
    // num_wg=1 through the same fast machinery).
    return model_forward_graph_cfg_m2_sampled(cl_ctx_, weights_, input_ids, start_pos,
                                              guidance, temperature, top_k, seed,
                                              force_argmax, out_ids);
}

void* Model::gpu_grid_alloc(const int32_t* host_grid, int num_codebooks, int steps1, int bos) {
    return (void*)mega_alloc_decode_grid(cl_ctx_, host_grid, num_codebooks, steps1, bos);
}
bool Model::gpu_grid_read(void* grid, int32_t* host_grid, int num_codebooks, int steps1) {
    return mega_read_decode_grid(cl_ctx_, (cl_mem)grid, host_grid, num_codebooks, steps1);
}
extern "C" bool mega_read_decode_grid_cols_async(
    OpenCLContext&, cl_mem, int32_t*, int, int, int, int, cl_event*);
bool Model::gpu_grid_read_cols_async(void* grid, int32_t* host_grid, int num_codebooks,
                                     int steps1, int col0, int col1, void** evt_out) {
    cl_event ev = nullptr;
    const bool ok = mega_read_decode_grid_cols_async(cl_ctx_, (cl_mem)grid, host_grid,
                                                     num_codebooks, steps1, col0, col1, &ev);
    *evt_out = (void*)ev;
    return ok;
}
void Model::gpu_grid_free(void* grid) {
    mega_free_decode_grid((cl_mem)grid);
}

std::vector<float> Model::forward(const std::vector<int32_t>& input_ids, int start_pos) {
    NNOPT_CHECKPOINT("Model::forward() — graph mode (delegating to model_forward_graph)");
    return model_forward_graph(cl_ctx_, weights_, input_ids, start_pos);
}
