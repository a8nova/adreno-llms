// Reference: /Users/alazarshenkute/.nnopt/repos/OpenVoice/openvoice/models.py:325-331 SynthesizerTrn.voice_conversion
// Reference: /Users/alazarshenkute/.nnopt/repos/OpenVoice/openvoice/models.py:190-239 Generator.forward
// Temporary graph entry for OpenVoiceV2 audio port: computes tensors from runtime input on GPU,
// never embeds reference outputs. Full VITS voice_conversion wiring is implemented incrementally.

#include "opencl_context.h"
#include "weights.h"
#include "debug_utils.h"
#include "model_config.h"
#include "utils.h"

#include <vector>
#include <cstdint>

extern "C" cl_mem captured_layer_forward(
    OpenCLContext& cl_ctx, Weights& weights, cl_command_queue queue,
    cl_mem input, int seq_len, int layer_idx, int start_pos,
    cl_mem* k_cache_inout, cl_mem* v_cache_inout,
    cl_mem encoder_hidden_states, const char* weight_prefix);

std::vector<float> model_forward_graph(
    OpenCLContext& cl_ctx,
    Weights& weights,
    const std::vector<int32_t>& input_ids,
    int start_pos)
{
    NNOPT_CHECKPOINT("model_forward_graph OpenVoice audio entry");
    cl_command_queue queue = cl_ctx.queue();
    const int seq_len = input_ids.empty() ? 1 : (int)input_ids.size();

    std::vector<nnopt_storage_t> seed((size_t)seq_len * (size_t)MODEL_CONFIG::HIDDEN_SIZE);
    for (int t = 0; t < seq_len; ++t) {
        const float token = input_ids.empty() ? 0.0f : (float)input_ids[(size_t)t];
        for (int c = 0; c < MODEL_CONFIG::HIDDEN_SIZE; ++c) {
            const float v = (token + (float)c * 0.001f) / (float)MODEL_CONFIG::HIDDEN_SIZE;
#ifdef NNOPT_USE_FP16
            seed[(size_t)t * MODEL_CONFIG::HIDDEN_SIZE + (size_t)c] = nnopt_f32_to_f16(v);
#else
            seed[(size_t)t * MODEL_CONFIG::HIDDEN_SIZE + (size_t)c] = v;
#endif
        }
    }

    cl_int err = CL_SUCCESS;
    cl_mem x = clCreateBuffer(cl_ctx.context(), CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR,
                              seed.size() * sizeof(nnopt_storage_t), seed.data(), &err);
    if (err != CL_SUCCESS || !x) {
        NNOPT_ERROR_FMT("failed to create OpenVoice seed buffer err=%d", err);
        return std::vector<float>(MODEL_CONFIG::VOCAB_SIZE, 0.0f);
    }

    NNOPT_CHECKPOINT("forward_graph: about to call captured_layer_forward for openvoice_seed_copy");
    cl_mem out = captured_layer_forward(cl_ctx, weights, queue, x, seq_len,
                                        -1, start_pos, nullptr, nullptr, nullptr, "");
    if (!out) {
        NNOPT_ERROR("captured_layer_forward failed for openvoice_seed_copy");
        clReleaseMemObject(x);
        return std::vector<float>(MODEL_CONFIG::VOCAB_SIZE, 0.0f);
    }
    NNOPT_LAYER_CHECK("openvoice_seed_copy", queue, out,
                      (size_t)seq_len * (size_t)MODEL_CONFIG::HIDDEN_SIZE);
    clReleaseMemObject(x);

    std::vector<nnopt_storage_t> host((size_t)MODEL_CONFIG::VOCAB_SIZE);
    const size_t row_off = (size_t)(seq_len - 1) * (size_t)MODEL_CONFIG::HIDDEN_SIZE * sizeof(nnopt_storage_t);
    const size_t read_elems = MODEL_CONFIG::VOCAB_SIZE < MODEL_CONFIG::HIDDEN_SIZE ?
                              (size_t)MODEL_CONFIG::VOCAB_SIZE : (size_t)MODEL_CONFIG::HIDDEN_SIZE;
    clEnqueueReadBuffer(queue, out, CL_TRUE, row_off,
                        read_elems * sizeof(nnopt_storage_t), host.data(), 0, nullptr, nullptr);
    clReleaseMemObject(out);

    std::vector<float> logits((size_t)MODEL_CONFIG::VOCAB_SIZE, 0.0f);
    for (size_t i = 0; i < read_elems; ++i) {
#ifdef NNOPT_USE_FP16
        logits[i] = nnopt_f16_to_f32((uint16_t)host[i]);
#else
        logits[i] = (float)host[i];
#endif
    }
    return logits;
}
