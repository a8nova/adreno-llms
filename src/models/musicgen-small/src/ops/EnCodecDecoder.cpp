// Reference: model_info/transformers_src/modeling_musicgen.py:~930-1060 MusicgenForConditionalGeneration.forward (audio_encoder usage)
// NOTE: EnCodec implementation is pending full reference source inclusion in model_info/transformers_src.
// This forward currently exists only to satisfy build-time pipeline wiring; it does NOT yet decode.
//
// The weights present in weights/model.meta.json indicate an EnCodec-style model:
//   - audio_encoder.decoder.layers.{i}.conv.weight_{g,v} (weight-norm conv)
//   - audio_encoder.decoder.layers.1.lstm.* (2-layer LSTM)
// This file will be expanded to match the exact HF EncodecModel.decode path.

#include "opencl_context.h"
#include "weights.h"
#include "debug_utils.h"
#include "model_config.h"

#include <cstddef>

extern "C" cl_mem EnCodecDecoder_forward(
    OpenCLContext& cl_ctx, Weights& weights, cl_command_queue queue,
    cl_mem input, int seq_len, int layer_idx, int start_pos,
    cl_mem* k_cache_inout, cl_mem* v_cache_inout,
    cl_mem encoder_hidden_states, const char* weight_prefix)
{
    (void)layer_idx;
    (void)start_pos;
    (void)k_cache_inout;
    (void)v_cache_inout;
    (void)encoder_hidden_states;

    // TODO: Implement full EnCodec decoder. For now, hard-fail loudly.
    // This prevents silent “success” while producing no audio.
    NNOPT_ERROR_FMT("EnCodecDecoder_forward missing implementation (prefix=%s)",
                    weight_prefix ? weight_prefix : "(null)");

    // Return nullptr signals to the caller that audio decode failed.
    (void)cl_ctx;
    (void)weights;
    (void)queue;
    (void)input;
    (void)seq_len;
    return nullptr;
}
