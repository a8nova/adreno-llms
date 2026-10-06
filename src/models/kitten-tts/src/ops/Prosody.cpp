// Prosody path — kmodel.predictor.*
//
//   DurationEncoder (6 x [BiLSTM -> AdaLayerNorm], style re-concatenated each round)
//     -> d [T,256]
//   duration LSTM -> duration_proj [T,50] -> sigmoid -> rowsum -> /speed
//     -> round-half-to-even -> clip(min=1) -> dur [T]
//   length regulation: F = sum(dur); frame_to_phon = repeat_interleave(arange(T), dur)
//   en[c][f] = d[frame_to_phon[max(f-1,0)]][c]     <- StyleTTS2's off-by-one shift
//   shared BiLSTM over en -> prosody [128, F]
//
// The alignment matrix pred_aln_trg [T,F] is strictly block-diagonal 0/1, so it
// is never materialized and no GEMM is run — it collapses to a gather. The
// shifted frame_src index vector is published so the acoustic branch can apply
// the identical alignment to t_en.
//
// Oracle gates: duration_output.bin (must be INTEGER-IDENTICAL — a +-1 drift on
// any phoneme changes F and desynchronizes everything downstream),
// ScatterND_2_output_0 (en), shared.Reshape_output_0, Transpose_1_output_0.

#include "../opencl_context.h"
#include "../weights.h"
#include "../nnopt_error.h"
#include "../styletts_ops.h"
#include "../debug_utils.h"
#include <string>
#include <vector>
#include <cmath>

namespace {
constexpr int C_EN   = 128;   // bert_encoder output width
constexpr int C_S    = 128;   // style half width
constexpr int C_X    = 256;   // DurationEncoder LSTM input (features + style)
constexpr int H      = 64;    // BiLSTM hidden per direction
constexpr int C_D    = 256;   // d width
constexpr int N_BINS = 50;    // duration_proj bins
constexpr float LN_EPS = 9.999999747378752e-06f;

// The six DurationEncoder BiLSTMs, then the duration LSTM, then the shared LSTM.
struct LstmKeys { const char* w; const char* r; const char* b; };
const LstmKeys DE_LSTM[6] = {
    {"onnx_LSTM_7817_quantized", "onnx_LSTM_7818_quantized", "onnx_LSTM_7816"},
    {"onnx_LSTM_7873_quantized", "onnx_LSTM_7874_quantized", "onnx_LSTM_7872"},
    {"onnx_LSTM_7929_quantized", "onnx_LSTM_7930_quantized", "onnx_LSTM_7928"},
    {"onnx_LSTM_7985_quantized", "onnx_LSTM_7986_quantized", "onnx_LSTM_7984"},
    {"onnx_LSTM_8041_quantized", "onnx_LSTM_8042_quantized", "onnx_LSTM_8040"},
    {"onnx_LSTM_8097_quantized", "onnx_LSTM_8098_quantized", "onnx_LSTM_8096"},
};
const LstmKeys DUR_LSTM    = {"onnx_LSTM_8152_quantized", "onnx_LSTM_8153_quantized", "onnx_LSTM_8151"};
const LstmKeys SHARED_LSTM = {"onnx_LSTM_8213_quantized", "onnx_LSTM_8214_quantized", "onnx_LSTM_8212"};
const int ADALN_IDX[6] = {1, 3, 5, 7, 9, 11};
}  // namespace

extern "C" bool Prosody_forward(
    OpenCLContext& cl_ctx,
    Weights& weights,
    cl_command_queue queue,
    cl_mem d_en,             // [T, 128] row-major (BertEncoder output)
    cl_mem style,            // [256] full style vector
    int T,
    float speed,
    cl_mem* out_prosody,     // [128, F]
    cl_mem* out_frame_src,   // int32 [F], ALREADY shifted
    cl_mem* out_en,          // [256, F]
    int* out_F)
{
    if (!st::init(cl_ctx)) return false;

    std::vector<cl_mem> owned;
    auto own = [&](cl_mem m) { if (m) owned.push_back(m); return m; };
    auto cleanup = [&]() -> bool {
        for (cl_mem m : owned) if (m) clReleaseMemObject(m);
        owned.clear();
        return false;
    };

    cl_int err = CL_SUCCESS;

    // ── prosody style half: style[128:256] ───────────────────────────────
    cl_mem s = own(st::alloc(cl_ctx, C_S));
    if (!s) return cleanup();
    err = clEnqueueCopyBuffer(queue, style, s, C_S * sizeof(nnopt_storage_t), 0,
                              C_S * sizeof(nnopt_storage_t), 0, nullptr, nullptr);
    if (err != CL_SUCCESS) { NNOPT_ERROR_FMT("Prosody: style slice err %d", err); return cleanup(); }

    cl_mem params = own(st::alloc_f32(cl_ctx, 2));
    cl_mem s_q    = own(st::alloc(cl_ctx, C_S));
    if (!params || !s_q) return cleanup();
    if (!st::dql(queue, s, params, s_q, C_S)) return cleanup();

    // ── AdaLayerNorm style fcs, hoisted (time-invariant) ─────────────────
    // lstms.{1,3,5,7,9} are int8 (MatMulInteger) and consume the DQL'd style;
    // lstms.11 is plain fp16 MatMul and consumes the unquantized style.
    cl_mem fc[6] = {nullptr};
    for (int i = 0; i < 6; i++) {
        const std::string base = "kmodel.predictor.text_encoder.lstms."
                               + std::to_string(ADALN_IDX[i]) + ".fc.";
        const bool quantized = (ADALN_IDX[i] != 11);
        fc[i] = own(st::adain_fc(cl_ctx, queue, weights,
                                 quantized ? s_q : s,
                                 base + (quantized ? "weight_quantized" : "weight"),
                                 base + "bias", 2 * C_EN));
        if (!fc[i]) { NNOPT_ERROR_FMT("Prosody: adaln fc %d failed", ADALN_IDX[i]); return cleanup(); }
    }

    // LayerNorm affine for the AdaLN blocks is identity (Constant_7 == 1,
    // Constant_8 == 0 — verified). Materialize once.
    cl_mem ln_ones  = own(st::upload_f32_as_storage(cl_ctx, std::vector<float>(C_EN, 1.0f)));
    cl_mem ln_zeros = own(st::upload_f32_as_storage(cl_ctx, std::vector<float>(C_EN, 0.0f)));
    if (!ln_ones || !ln_zeros) return cleanup();

    // ── DurationEncoder: 6 x [BiLSTM -> AdaLayerNorm -> re-concat style] ──
    cl_mem x  = own(st::alloc(cl_ctx, (size_t)T * C_X));   // [T,256]
    cl_mem y  = own(st::alloc(cl_ctx, (size_t)T * C_EN));  // [T,128]
    cl_mem nz = own(st::alloc(cl_ctx, (size_t)T * C_EN));
    cl_mem z  = own(st::alloc(cl_ctx, (size_t)T * C_EN));
    if (!x || !y || !nz || !z) return cleanup();

    if (!st::pack_style(queue, d_en, s, x, T, C_EN, C_S)) return cleanup();

    for (int i = 0; i < 6; i++) {
        if (!st::bilstm_quant(queue, weights, x, T, C_X, H,
                              DE_LSTM[i].w, DE_LSTM[i].r, DE_LSTM[i].b, y)) return cleanup();
        if (!st::layernorm_rows(queue, y, ln_ones, ln_zeros, nz, T, C_EN, LN_EPS)) return cleanup();
        if (!st::adaln_rows(queue, nz, fc[i], z, T, C_EN)) return cleanup();
        if (!st::pack_style(queue, z, s, x, T, C_EN, C_S)) return cleanup();
    }
    NNOPT_LAYER_CHECK("text_encoder_1.Where_25_output", queue, x, (size_t)T * C_X);
    // x is now d [T, 256]

    // ── duration LSTM -> duration_proj -> durations ──────────────────────
    cl_mem xd = own(st::alloc(cl_ctx, (size_t)T * C_EN));
    if (!xd) return cleanup();
    if (!st::bilstm_quant(queue, weights, x, T, C_X, H,
                          DUR_LSTM.w, DUR_LSTM.r, DUR_LSTM.b, xd)) return cleanup();
    NNOPT_LAYER_CHECK("lstm.Reshape_output_0", queue, xd, (size_t)T * C_EN);

    cl_mem dur_w = weights.get_buffer("onnx_MatMul_8154");                              // [128,50]
    cl_mem dur_b = weights.get_buffer("kmodel.predictor.duration_proj.linear_layer.bias");
    if (!dur_w || !dur_b) { NNOPT_ERROR("Prosody: missing duration_proj"); return cleanup(); }
    cl_mem logits = own(st::alloc(cl_ctx, (size_t)T * N_BINS));
    if (!logits) return cleanup();
    if (!pytorch_conv1d(queue, T, N_BINS, C_EN, xd, dur_w, logits)) {
        NNOPT_ERROR("Prosody: duration_proj GEMM failed"); return cleanup();
    }
    if (!st::add_bias_rows(queue, logits, dur_b, logits, T, N_BINS)) return cleanup();
    NNOPT_LAYER_CHECK("duration_proj.linear_layer.Add_output_0", queue, logits, (size_t)T * N_BINS);

    std::vector<float> lg;
    if (!st::download_as_f32(queue, logits, (size_t)T * N_BINS, lg)) return cleanup();

    // duration[i] = clip(round_half_to_even(sum_j sigmoid(logit[i][j]) / speed), min=1)
    // std::round is half-AWAY-from-zero and produces a different frame count on
    // an exact .5 — nearbyint under the default FE_TONEAREST is half-to-even.
    std::vector<int> dur(T);
    int F = 0;
    for (int i = 0; i < T; i++) {
        float sum = 0.0f;
        for (int j = 0; j < N_BINS; j++) sum += 1.0f / (1.0f + std::exp(-lg[(size_t)i * N_BINS + j]));
        float d = std::nearbyint(sum / speed);
        if (d < 1.0f) d = 1.0f;
        dur[i] = (int)d;
        F += dur[i];
    }
    if (F <= 0) { NNOPT_ERROR("Prosody: total duration is zero"); return cleanup(); }
    fprintf(stderr, "PROSODY: T=%d total_frames=%d (dur[0..7]=%d %d %d %d %d %d %d %d)\n",
            T, F, dur[0], dur[1], dur[2], dur[3], dur[4], dur[5], dur[6], dur[7]);

    // ── length regulation, fused with the StyleTTS2 shift ────────────────
    // frame_to_phon = repeat_interleave(arange(T), dur); then en reads
    // frame_to_phon[max(f-1,0)], which is the inference-time column shift.
    std::vector<int32_t> f2p(F);
    {
        int f = 0;
        for (int i = 0; i < T; i++)
            for (int k = 0; k < dur[i]; k++) f2p[f++] = i;
    }
    std::vector<int32_t> shifted(F);
    for (int f = 0; f < F; f++) shifted[f] = f2p[f == 0 ? 0 : f - 1];

    cl_mem frame_src = clCreateBuffer(cl_ctx.context(),
                                      CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR,
                                      (size_t)F * sizeof(int32_t), shifted.data(), &err);
    if (err != CL_SUCCESS || !frame_src) { NNOPT_ERROR_FMT("Prosody: frame_src err %d", err); return cleanup(); }

    // ── en[c][f] = d[shifted[f]][c] ──────────────────────────────────────
    cl_mem d_ct = own(st::alloc(cl_ctx, (size_t)C_D * T));
    cl_mem en   = st::alloc(cl_ctx, (size_t)C_D * F);
    if (!d_ct || !en) { clReleaseMemObject(frame_src); if (en) clReleaseMemObject(en); return cleanup(); }
    if (!st::transpose(queue, x, d_ct, T, C_D) ||
        !st::align_expand(queue, d_ct, frame_src, en, C_D, T, F)) {
        clReleaseMemObject(frame_src); clReleaseMemObject(en); return cleanup();
    }
    NNOPT_LAYER_CHECK("ScatterND_2_output_0", queue, en, (size_t)C_D * F);

    // ── shared BiLSTM over the aligned prosody features ──────────────────
    cl_mem en_fc = own(st::alloc(cl_ctx, (size_t)F * C_D));     // [F,256] time-major
    cl_mem hs    = own(st::alloc(cl_ctx, (size_t)F * C_EN));    // [F,128]
    cl_mem pros  = st::alloc(cl_ctx, (size_t)C_EN * F);         // [128,F]
    if (!en_fc || !hs || !pros) {
        clReleaseMemObject(frame_src); clReleaseMemObject(en);
        if (pros) clReleaseMemObject(pros);
        return cleanup();
    }
    if (!st::transpose(queue, en, en_fc, C_D, F) ||
        !st::bilstm_quant(queue, weights, en_fc, F, C_D, H,
                          SHARED_LSTM.w, SHARED_LSTM.r, SHARED_LSTM.b, hs)) {
        clReleaseMemObject(frame_src); clReleaseMemObject(en); clReleaseMemObject(pros);
        return cleanup();
    }
    NNOPT_LAYER_CHECK("shared.Reshape_output_0", queue, hs, (size_t)F * C_EN);
    if (!st::transpose(queue, hs, pros, F, C_EN)) {
        clReleaseMemObject(frame_src); clReleaseMemObject(en); clReleaseMemObject(pros);
        return cleanup();
    }
    NNOPT_DEBUG_SYNC(queue);
    NNOPT_LAYER_CHECK("Transpose_1_output_0", queue, pros, (size_t)C_EN * F);

    for (cl_mem m : owned) if (m) clReleaseMemObject(m);
    owned.clear();

    *out_prosody   = pros;
    *out_frame_src = frame_src;
    *out_en        = en;
    *out_F         = F;
    return true;
}
