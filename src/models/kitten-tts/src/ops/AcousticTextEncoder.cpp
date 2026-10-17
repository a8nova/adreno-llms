// Acoustic TextEncoder — kmodel.text_encoder.*  (ONNX nodes /text_encoder/embedding,
// /text_encoder/cnn.0..5, /text_encoder/lstm, /text_encoder/text_proj).
//
// NOT to be confused with kmodel.predictor.text_encoder.* (the DurationEncoder),
// which the ONNX exporter emitted under the SAME /text_encoder/ scope prefix.
// Select by weight prefix, never by node name. See PORT_JOURNAL.md TOOL GAP #5.
//
//   input_ids [T]
//     -> embedding [T,128] -> transpose -> [128,T]
//     -> 6 x { DQL, Conv1d(k=5,pad=2)+bias, LayerNorm(axis=-1,eps=1e-5), LeakyReLU(0.2) }
//     -> BiLSTM(in=128, hidden=64, bidirectional) -> [T,128]
//     -> text_proj MatMul [128,512] + bias -> [T,512]
//     -> transpose -> [512,T]                       <- stage output (asr source)
//
// Blocks 0-4 keep the CNN in channels-first [128,T] and transpose around the
// LayerNorm. Block 5 is ASYMMETRIC: the exporter folded away its transpose-back,
// so it ends channels-LAST [T,128] — which is exactly the layout the LSTM wants.
//
// Boundary oracle: reference/layers/text_encoder.Transpose_6_output_0_output.bin
//                  ([1,512,70] = 35840 f32)

#include "../opencl_context.h"
#include "../weights.h"
#include "../nnopt_error.h"
#include "../styletts_ops.h"
#include "../debug_utils.h"
#include <string>
#include <vector>

namespace {
constexpr int C      = 128;   // CNN channel width
constexpr int H      = 64;    // BiLSTM hidden per direction
constexpr int D      = 512;   // text_proj output
constexpr int K      = 5;     // CNN kernel
constexpr int PAD    = 2;
constexpr float LN_EPS = 9.999999747378752e-06f;
}  // namespace

extern "C" cl_mem AcousticTextEncoder_forward(
    OpenCLContext& cl_ctx,
    Weights& weights,
    cl_command_queue queue,
    cl_mem input_ids,      // int32 [T]
    int seq_len)
{
    const int T = seq_len;
    if (!st::init(cl_ctx)) return nullptr;

    std::vector<cl_mem> owned;
    auto own = [&](cl_mem m) { if (m) owned.push_back(m); return m; };
    auto cleanup = [&]() -> cl_mem {
        for (cl_mem m : owned) if (m) clReleaseMemObject(m);
        owned.clear();
        return nullptr;
    };

    cl_mem emb_w = weights.get_buffer("kmodel.text_encoder.embedding.weight");
    if (!emb_w) { NNOPT_ERROR("AcousticTextEncoder: missing embedding.weight"); return nullptr; }

    cl_mem params = own(st::alloc_f32(cl_ctx, 2));           // DQL scale/zp
    cl_mem tc     = own(st::alloc(cl_ctx, (size_t)T * C));   // [T,C] scratch
    cl_mem ct     = own(st::alloc(cl_ctx, (size_t)T * C));   // [C,T] scratch
    cl_mem ct2    = own(st::alloc(cl_ctx, (size_t)T * C));
    cl_mem tc2    = own(st::alloc(cl_ctx, (size_t)T * C));
    if (!params || !tc || !ct || !ct2 || !tc2) return cleanup();

    // ── embedding -> [T,128] -> transpose -> [128,T] ──
    if (!st::embed_tc(queue, input_ids, emb_w, tc, T, C)) return cleanup();
    if (!st::transpose(queue, tc, ct, T, C)) return cleanup();
    NNOPT_LAYER_CHECK("text_encoder.Transpose_output_0", queue, ct, (size_t)C * T);

    // ── 6 CNN blocks ──
    for (int n = 0; n < 6; n++) {
        const std::string p = "kmodel.text_encoder.cnn." + std::to_string(n);
        cl_mem cw = weights.get_buffer(p + ".0.weight_quantized");   // already dequantized
        cl_mem cb = weights.get_buffer(p + ".0.bias");
        cl_mem lg = weights.get_buffer(p + ".1.gamma");
        cl_mem lb = weights.get_buffer(p + ".1.beta");
        if (!cw || !cb || !lg || !lb) {
            NNOPT_ERROR_FMT("AcousticTextEncoder: missing weights for cnn.%d", n);
            return cleanup();
        }
        // DynamicQuantizeLinear on the block input (model semantics, not an opt).
        if (!st::dql(queue, ct, params, ct2, C * T)) return cleanup();
        // Conv1d k=5 pad=2 stride=1 group=1, + bias
        if (!st::conv1d(queue, ct2, cw, cb, ct, C, C, T, T, K, 1, PAD, 1)) return cleanup();
        // LayerNorm is over the 128 channels at each t — transpose to [T,C] first,
        // because in [C,T] those 128 elements are strided by T, not contiguous.
        if (!st::transpose(queue, ct, tc, C, T)) return cleanup();
        if (!st::layernorm_rows(queue, tc, lg, lb, tc2, T, C, LN_EPS)) return cleanup();
        if (!st::leaky_relu(queue, tc2, tc, T * C)) return cleanup();
        if (n < 5) {
            // blocks 0-4 go back to channels-first for the next conv
            if (!st::transpose(queue, tc, ct, T, C)) return cleanup();
        }
        // block 5 stays [T,C] — that IS the LSTM input layout
    }
    NNOPT_LAYER_CHECK("text_encoder.Where_6_output", queue, tc, (size_t)T * C);

    // ── BiLSTM: [T,128] time-major -> [T, 2*64] ──
    cl_mem lstm_out = own(st::alloc(cl_ctx, (size_t)T * 2 * H));
    if (!lstm_out) return cleanup();
    if (!st::bilstm_quant(queue, weights, tc, T, C, H,
                          "onnx_LSTM_7590_quantized",
                          "onnx_LSTM_7591_quantized",
                          "onnx_LSTM_7589", lstm_out)) return cleanup();
    NNOPT_LAYER_CHECK("text_encoder.Transpose_3_output_0", queue, lstm_out, (size_t)T * C);

    // ── text_proj: [T,128] @ [128,512] + bias -> [T,512] ──
    cl_mem proj_w = weights.get_buffer("onnx_MatMul_7598_quantized");   // ONNX [K,N]
    cl_mem proj_b = weights.get_buffer("kmodel.text_encoder.text_proj.bias");
    if (!proj_w || !proj_b) { NNOPT_ERROR("AcousticTextEncoder: missing text_proj"); return cleanup(); }

    cl_mem lq = own(st::alloc(cl_ctx, (size_t)T * C));
    cl_mem td = own(st::alloc(cl_ctx, (size_t)T * D));
    if (!lq || !td) return cleanup();
    if (!st::dql(queue, lstm_out, params, lq, T * C)) return cleanup();
    if (!pytorch_conv1d(queue, T, D, C, lq, proj_w, td)) {
        NNOPT_ERROR("AcousticTextEncoder: text_proj GEMM failed"); return cleanup();
    }
    if (!st::add_bias_rows(queue, td, proj_b, td, T, D)) return cleanup();

    // ── transpose -> [512,T] : the stage output ──
    cl_mem out = st::alloc(cl_ctx, (size_t)D * T);
    if (!out) return cleanup();
    if (!st::transpose(queue, td, out, T, D)) { clReleaseMemObject(out); return cleanup(); }
    NNOPT_DEBUG_SYNC(queue);
    NNOPT_LAYER_CHECK("text_encoder.Transpose_6_output_0", queue, out, (size_t)D * T);

    cleanup();
    return out;
}
