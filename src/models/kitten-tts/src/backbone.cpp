// Auto-generated backbone for KittenML/kitten-tts-nano-0.1.
//
// This file is intentionally small: the scaffold emits only TOP-LEVEL ops
// here. Composite ops (the per-layer wrapper) call their primitives
// internally, so backbone.cpp doesn't grow with NUM_LAYERS.
//
// Per-class trace counts (informational):
//      1× Bert
//      1× TextEncoder
//      1× DecoderGenerator
//      1× Predictor
//      1× DecoderDecoder
//      1× DecoderEncode
//      1× DecoderDecode
//      1× Lstm
//      1× BertEncoder
//      1× TextEncoder1
//
// ─── UNIVERSAL <Class>_forward SIGNATURE ───────────────────────────────
// Every <Class>_forward in src/ops/<Class>.cpp uses these 11 args:
//   (OpenCLContext&, Weights&, cl_command_queue,
//    cl_mem input, int seq_len, int layer_idx, int start_pos,
//    cl_mem* k_cache_inout, cl_mem* v_cache_inout,
//    cl_mem encoder_hidden_states,
//    const char* weight_prefix) → returns cl_mem output
//
// encoder_hidden_states is non-null only for cross-attention calls in
// encoder-decoder models (Whisper, T5, SeamlessM4T). Pass nullptr for
// every other call (causal LM ops, encoder self-attention, primitives).
//
// ─── THE AGENT'S JOB IN THIS FILE ──────────────────────────────────────
// This is an ONNX-source SUBMODEL graph (see .nnport/onnx_submodels.json).
// 1. Wire the submodel calls below by TENSOR DATAFLOW, not the raw ONNX node
//    order (ONNX topo-sort interleaves branches). Thread each submodel's real
//    input cl_mem(s) — some submodels take style/noise fixtures, not just x.
// 2. Each LAYER_CHECK is sized from the submodel's traced boundary shape and
//    named by its boundary dump_name — validate against reference/layers/.
// 3. The final readout returns the terminal submodel's output as float PCM
//    (the waveform). No logits / LM head. main.cpp writes it to output.wav.

#include "opencl_context.h"
#include "weights.h"
#include "debug_utils.h"
#include "model_config.h"
#include "utils.h"
#include "styletts_ops.h"
#include <cstdio>
#include <chrono>
#include <cstdlib>

extern "C" cl_mem Bert_forward(
    OpenCLContext& cl_ctx, Weights& weights, cl_command_queue queue,
    cl_mem input, int seq_len, int layer_idx, int start_pos,
    cl_mem* k_cache_inout, cl_mem* v_cache_inout,
    cl_mem encoder_hidden_states, const char* weight_prefix);
extern "C" cl_mem TextEncoder_forward(
    OpenCLContext& cl_ctx, Weights& weights, cl_command_queue queue,
    cl_mem input, int seq_len, int layer_idx, int start_pos,
    cl_mem* k_cache_inout, cl_mem* v_cache_inout,
    cl_mem encoder_hidden_states, const char* weight_prefix);
extern "C" cl_mem DecoderGenerator_forward(
    OpenCLContext& cl_ctx, Weights& weights, cl_command_queue queue,
    cl_mem input, int seq_len, int layer_idx, int start_pos,
    cl_mem* k_cache_inout, cl_mem* v_cache_inout,
    cl_mem encoder_hidden_states, const char* weight_prefix);
extern "C" cl_mem Predictor_forward(
    OpenCLContext& cl_ctx, Weights& weights, cl_command_queue queue,
    cl_mem input, int seq_len, int layer_idx, int start_pos,
    cl_mem* k_cache_inout, cl_mem* v_cache_inout,
    cl_mem encoder_hidden_states, const char* weight_prefix);
extern "C" cl_mem DecoderDecoder_forward(
    OpenCLContext& cl_ctx, Weights& weights, cl_command_queue queue,
    cl_mem input, int seq_len, int layer_idx, int start_pos,
    cl_mem* k_cache_inout, cl_mem* v_cache_inout,
    cl_mem encoder_hidden_states, const char* weight_prefix);
extern "C" cl_mem DecoderEncode_forward(
    OpenCLContext& cl_ctx, Weights& weights, cl_command_queue queue,
    cl_mem input, int seq_len, int layer_idx, int start_pos,
    cl_mem* k_cache_inout, cl_mem* v_cache_inout,
    cl_mem encoder_hidden_states, const char* weight_prefix);
extern "C" cl_mem DecoderDecode_forward(
    OpenCLContext& cl_ctx, Weights& weights, cl_command_queue queue,
    cl_mem input, int seq_len, int layer_idx, int start_pos,
    cl_mem* k_cache_inout, cl_mem* v_cache_inout,
    cl_mem encoder_hidden_states, const char* weight_prefix);
extern "C" cl_mem Lstm_forward(
    OpenCLContext& cl_ctx, Weights& weights, cl_command_queue queue,
    cl_mem input, int seq_len, int layer_idx, int start_pos,
    cl_mem* k_cache_inout, cl_mem* v_cache_inout,
    cl_mem encoder_hidden_states, const char* weight_prefix);
extern "C" cl_mem BertEncoder_forward(
    OpenCLContext& cl_ctx, Weights& weights, cl_command_queue queue,
    cl_mem input, int seq_len, int layer_idx, int start_pos,
    cl_mem* k_cache_inout, cl_mem* v_cache_inout,
    cl_mem encoder_hidden_states, const char* weight_prefix);
// Acoustic TextEncoder — a SEPARATE module from the collapsed "TextEncoder"
// stage (which is really the predictor's DurationEncoder). It has its own
// signature because it consumes input_ids directly, not the previous stage's
// activation. See PORT_JOURNAL.md TOOL GAP #5.
extern "C" cl_mem AcousticTextEncoder_forward(
    OpenCLContext& cl_ctx, Weights& weights, cl_command_queue queue,
    cl_mem input_ids, int seq_len);
// Prosody path: returns several outputs (prosody features, the shared alignment
// index vector, the aligned prosody tensor, and the DATA-DEPENDENT frame count).
// The scaffold's single-cl_mem return convention cannot express this.
extern "C" bool Prosody_forward(
    OpenCLContext& cl_ctx, Weights& weights, cl_command_queue queue,
    cl_mem d_en, cl_mem style, int T, float speed,
    cl_mem* out_prosody, cl_mem* out_frame_src, cl_mem* out_en, int* out_F);
extern "C" bool DecoderPath_forward(
    OpenCLContext& cl_ctx, Weights& weights, cl_command_queue queue,
    cl_mem prosody, cl_mem asr, cl_mem style, int F,
    cl_mem* out_decoded, cl_mem* out_f0_raw);
extern "C" cl_mem Generator_forward(
    OpenCLContext& cl_ctx, Weights& weights, cl_command_queue queue,
    cl_mem decoded, cl_mem f0_raw, cl_mem style,
    const std::vector<float>& rng_normal, int F, int* out_samples);
std::vector<float> model_forward_graph(
    OpenCLContext& cl_ctx,
    Weights& weights,
    const std::vector<int32_t>& input_ids,
    const std::vector<float>& style,
    const std::vector<float>& rng_uniform,
    const std::vector<float>& rng_normal,
    int start_pos)
{
    NNOPT_CHECKPOINT("model_forward_graph entry");
    // TRUE StyleTTS2 dataflow order (NOT the ONNX topo-sort / weight-prefix
    // collapse order the scaffold emitted). Derived from onnx_graph_spec.json
    // node-index tracing of each stage boundary tensor:
    //   Bert(ALBERT) -> BertEncoder(768->128) -> TextEncoder1(align/sort) ->
    //   TextEncoder(DurationEncoder CNN+BiLSTM+AdaLN) -> Lstm(duration) ->
    //   Predictor(F0/N) -> DecoderDecode(AdaIN) -> DecoderDecoder ->
    //   DecoderGenerator(HiFiGAN+iSTFT) -> waveform.
    // style feeds AdaLayerNorm/AdaIN; rng_* feed the source module noise.
    (void)rng_uniform;  // provably dead: the scatter it feeds is never read
    cl_command_queue queue = cl_ctx.queue();
    int seq_len = (int)input_ids.size();

    cl_int err = CL_SUCCESS;

    // Coarse per-stage wall timing (NNOPT_STAGE_TIMING=1). clFinish-bracketed so
    // it attributes GPU work the per-kernel profiler can't see (bert/lstm are not
    // wrapped in KernelProfiler). Zero cost when the env var is unset.
    const bool stage_timing = [] { const char* e = getenv("NNOPT_STAGE_TIMING"); return e && e[0] == '1'; }();
    auto stage_t0 = std::chrono::high_resolution_clock::now();
    auto stage_mark = [&](const char* name) {
        if (!stage_timing) return;
        clFinish(queue);
        auto now = std::chrono::high_resolution_clock::now();
        fprintf(stderr, "STAGE_MS %-22s %8.1f\n", name,
                std::chrono::duration<double, std::milli>(now - stage_t0).count());
        stage_t0 = now;
    };

    // ── Style/speaker vector upload (256 fp32) ─────────────────────────
    // StyleTTS2 splits the 256-d reference style: s[:128] = acoustic (decoder
    // AdaIN), s[128:] = prosodic (ProsodyPredictor DurationEncoder + F0/N).
    // The DurationEncoder's AdaLayerNorm fc (128->256) consumes the prosodic
    // half. We upload the FULL 256-d vector as nnopt_storage_t and let each
    // consumer slice the half it needs. Passed to ops via the
    // encoder_hidden_states arg slot (unused for this non-enc-dec model).
    cl_mem style_buf = nullptr;
    if (!style.empty()) {
        std::vector<nnopt_storage_t> style_h(style.size());
        for (size_t i = 0; i < style.size(); i++)
#ifdef NNOPT_USE_FP16
            style_h[i] = nnopt_f32_to_f16(style[i]);
#else
            style_h[i] = style[i];
#endif
        style_buf = clCreateBuffer(cl_ctx.context(),
            CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR,
            style_h.size() * sizeof(nnopt_storage_t), style_h.data(), &err);
        if (err != CL_SUCCESS || !style_buf) {
            NNOPT_ERROR_FMT("failed to upload style vector (err=%d)", err);
            return std::vector<float>();
        }
    }
    cl_mem ids_buf = clCreateBuffer(cl_ctx.context(),
        CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR,
        seq_len * sizeof(int32_t), (void*)input_ids.data(), &err);
    if (err != CL_SUCCESS || !ids_buf) {
        NNOPT_ERROR_FMT("failed to upload input_ids (err=%d)", err);
        return std::vector<float>();
    }

    // Halt helper: on a null op result, dump note + exit(0) after freeing bufs.
    auto halt = [&](const char* node, int order, std::vector<cl_mem> live) {
        fflush(stdout);
        fprintf(stderr, "\nNNOPT_FORWARD_HALTED node=%s order=%d : op not implemented yet (returned null). "
                        "The forward ran and dumped layer_dumps/ for every implemented op before it. "
                        "Implement this op next, in graph order, then rebuild.\n", node, order);
        fflush(stderr);
        for (cl_mem m : live) if (m) clReleaseMemObject(m);
        exit(0);
    };

    // ── [order 0] Bert (ALBERT phoneme encoder) : input_ids -> [T,768] ──
    cl_mem bert_out = Bert_forward(
        cl_ctx, weights, queue, ids_buf, seq_len, -1, start_pos,
        nullptr, nullptr, nullptr, "bert");
    if (!bert_out) halt("bert...full_layer_layer_norm_11", 0, {ids_buf});
    stage_mark("bert");
    NNOPT_LAYER_CHECK("bert.encoder.albert_layer_groups.0.albert_layers.0.full_layer_layer_norm_11.LayerNormalization_output_0_Cast_to_float16_input_0",
                      queue, bert_out, /*num_elems=*/(size_t)seq_len * 768);

    // ── [order 1] BertEncoder (Linear 768->128) : [T,768] -> [T,128] ──
    cl_mem enc_out = BertEncoder_forward(
        cl_ctx, weights, queue, bert_out, seq_len, -1, start_pos,
        nullptr, nullptr, nullptr, "bert_encoder");
    clReleaseMemObject(bert_out);
    if (!enc_out) halt("bert_encoder.Add_output_0", 1, {});
    stage_mark("bert_encoder");
    NNOPT_LAYER_CHECK("bert_encoder.Add_output_0", queue, enc_out,
                      /*num_elems=*/(size_t)seq_len * 128);

    // ── [order 2] text_encoder_1 (pack_padded_sequence / mask shape-metadata) ──
    // The entire text_encoder_1 ONNX subgraph (~200 Shape/Gather/Slice/Concat/
    // Where/TopK/ScatterND nodes) is pure control-flow that computes the
    // reshape-target shape vector [batch=1, channels=C, seq=T] for the
    // DurationEncoder's pack_padded_sequence. With batch=1 the length-sort is
    // the identity permutation, so it collapses to the constant [1, C, T].
    // It carries NO neural weights and its output is NOT consumed by any
    // downstream op (TextEncoder reads enc_out directly) — so we emit the shape
    // vector inline purely to validate against the boundary reference dump.
    // C = enc_out channel width (= kmodel.bert_encoder.bias length = 128).
    cl_mem te1_out = nullptr;
    {
        int channels = (int)weights.get_num_elements("kmodel.bert_encoder.bias");
        if (channels <= 0) halt("text_encoder_1.Where_24_output_0", 2, {enc_out});
        const int batch = 1;
        te1_out = clCreateBuffer(cl_ctx.context(), CL_MEM_READ_WRITE,
                                 3 * sizeof(nnopt_storage_t), nullptr, &err);
        if (err != CL_SUCCESS || !te1_out) halt("text_encoder_1.Where_24_output_0", 2, {enc_out});
        cl_program te1_prog = cl_ctx.build_program_from_file("kernels/text_encoder_1.cl");
        if (!te1_prog) halt("text_encoder_1.Where_24_output_0", 2, {enc_out, te1_out});
        cl_kernel te1_k = clCreateKernel(te1_prog, "te1_shape_vec", &err);
        if (err != CL_SUCCESS || !te1_k) halt("text_encoder_1.Where_24_output_0", 2, {enc_out, te1_out});
        if (!set_arg_checked(te1_k, 0, sizeof(cl_mem), &te1_out,  "out")      ||
            !set_arg_checked(te1_k, 1, sizeof(int),    &batch,    "batch")    ||
            !set_arg_checked(te1_k, 2, sizeof(int),    &channels, "channels") ||
            !set_arg_checked(te1_k, 3, sizeof(int),    &seq_len,  "seq"))
            halt("text_encoder_1.Where_24_output_0", 2, {enc_out, te1_out});
        size_t te1_gws = 3;
        err = clEnqueueNDRangeKernel(queue, te1_k, 1, nullptr, &te1_gws, nullptr, 0, nullptr, nullptr);
        if (err != CL_SUCCESS) halt("text_encoder_1.Where_24_output_0", 2, {enc_out, te1_out});
        NNOPT_DEBUG_SYNC(queue);
        clReleaseKernel(te1_k);
        clReleaseProgram(te1_prog);
    }
    NNOPT_LAYER_CHECK("text_encoder_1.Where_24_output_0", queue, te1_out, /*num_elems=*/3);

    // ── Acoustic TextEncoder : input_ids -> t_en [512, T] ──────────────
    // This is a PARALLEL branch of the DAG, not the next link in a chain: it
    // reads input_ids directly. Its output is later aligned (70 -> F frames) by
    // the duration model to produce the decoder's ASR features.
    cl_mem t_en = AcousticTextEncoder_forward(cl_ctx, weights, queue, ids_buf, seq_len);
    clReleaseMemObject(ids_buf);
    if (!t_en) halt("text_encoder.Transpose_6_output_0", 3, {enc_out, te1_out});
    stage_mark("acoustic_textenc");
    NNOPT_LAYER_CHECK("text_encoder.Transpose_6_output_0", queue, t_en,
                      /*num_elems=*/(size_t)512 * seq_len);

    // ── Prosody path : d_en + style -> durations, alignment, prosody features ──
    // This single call replaces the scaffold's linear TextEncoder -> Lstm ->
    // Predictor chain. It owns the DurationEncoder, the duration LSTM, length
    // regulation (T -> F frames), and the shared LSTM. F is DATA-DEPENDENT: it
    // is the sum of the predicted per-phoneme durations, not a constant.
    cl_mem prosody = nullptr, frame_src = nullptr, en = nullptr;
    int F = 0;
    const float speed = 1.0f;
    if (!Prosody_forward(cl_ctx, weights, queue, enc_out, style_buf, seq_len, speed,
                         &prosody, &frame_src, &en, &F))
        halt("shared.Reshape_output_0", 4, {enc_out, te1_out, t_en});

    // ── asr : the acoustic features under the SAME alignment ─────────────
    // asr[c][f] = t_en[c][frame_src[f]] — the identical shifted gather the
    // prosody branch used, which is why frame_src is published rather than
    // recomputed.
    cl_mem asr = st::alloc(cl_ctx, (size_t)512 * F);
    if (!asr || !st::align_expand(queue, t_en, frame_src, asr, 512, seq_len, F))
        halt("ScatterND_4_output_0", 5, {enc_out, te1_out, t_en, prosody, frame_src, en});
    stage_mark("prosody");
    NNOPT_LAYER_CHECK("ScatterND_4_output_0", queue, asr, (size_t)512 * F);

    // ── F0/N heads + decoder encode/decode -> [256, 2F] ──────────────────
    cl_mem decoded = nullptr, f0_raw = nullptr;
    if (!DecoderPath_forward(cl_ctx, weights, queue, prosody, asr, style_buf, F,
                             &decoded, &f0_raw))
        halt("decoder.decode.3.Mul_output_0", 6,
             {enc_out, te1_out, t_en, prosody, frame_src, en, asr});
    stage_mark("decoder_path");

    // ── generator: HiFiGAN + harmonic source + iSTFT -> waveform ─────────
    int n_samples = 0;
    cl_mem gen_out = Generator_forward(cl_ctx, weights, queue, decoded, f0_raw,
                                       style_buf, rng_normal, F, &n_samples);
    if (!gen_out) halt("decoder.generator.Slice_3_output_0", 7,
                       {enc_out, te1_out, t_en, prosody, frame_src, en, asr, decoded, f0_raw});
    stage_mark("generator");

    if (style_buf) clReleaseMemObject(style_buf);
    for (cl_mem m : {enc_out, te1_out, t_en, prosody, frame_src, en, asr, decoded, f0_raw})
        if (m) clReleaseMemObject(m);
    cl_mem x = gen_out;

    // ── Post-loop top-level ops (final norm, lm_head, …) ───────────────
    // (none)

    // ── Waveform readout (ONNX TTS submodel graph) ──────────────────────
    // The final submodel's output IS the audio (no LM head / logits). Read x
    // fully and return it as float PCM. _out_elems is the traced final-boundary
    // shape; if your real terminal op differs, adjust it. main.cpp writes this
    // vector to output.wav at io_contract.json::sample_rate_hz.
    // Sample count is DATA-DEPENDENT (sum of predicted durations x 300), not the
    // traced constant the scaffold emitted.
    const size_t _out_elems = (size_t)n_samples;
    std::vector<float> waveform(_out_elems, 0.0f);
    if (x && _out_elems > 0) {
        std::vector<nnopt_storage_t> host_storage(_out_elems);
        clEnqueueReadBuffer(queue, x, CL_TRUE, 0,
            _out_elems * sizeof(nnopt_storage_t),
            host_storage.data(), 0, nullptr, nullptr);
        for (size_t i = 0; i < _out_elems; i++)
#ifdef NNOPT_USE_FP16
            waveform[i] = nnopt_f16_to_f32(static_cast<uint16_t>(host_storage[i]));
#else
            waveform[i] = static_cast<float>(host_storage[i]);
#endif
        clReleaseMemObject(x);
    }
    return waveform;
}
