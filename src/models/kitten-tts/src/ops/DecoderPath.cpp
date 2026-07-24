// F0/N prosody heads + decoder encode/decode — kmodel.predictor.{F0,N}.* and
// kmodel.decoder.decoder.{encode,decode.0..3}.*
//
//   prosody [128,F] -> F0 chain -> F0 [1,2F] ; N chain -> N [1,2F]
//   F0/N downsampled by stride-2 convs back to F frames for the decoder trunk
//   asr_res = 1x1(asr) [64,F]  (computed once, reused by all four decode concats)
//   Concat[asr(512), F0d(1), Nd(1)]            -> encode   [256,F]
//   Concat[prev(256), asr_res(64), F0d, Nd]    -> decode.0/1/2 [256,F]
//   Concat[decode.2(256), asr_res, F0d, Nd]    -> decode.3 [256,2F]  (upsamples)
//
// Style is split: style[128:256] drives the 12 predictor AdaINs,
// style[0:128] drives the 10 decoder AdaINs. Both are hoisted — every fc is
// time-invariant, so the 22 GEMVs run once, outside any loop.
//
// The raw (un-downsampled) F0 is ALSO published: the generator's harmonic
// source module consumes it, NOT the F0_conv output.

#include "../opencl_context.h"
#include "../weights.h"
#include "../nnopt_error.h"
#include "../styletts_ops.h"
#include "../debug_utils.h"
#include <string>
#include <vector>

namespace {
constexpr int C_S = 128;

struct Blk { const char* prefix; int c_in, c_out; bool upsample, conv1x1; };
}  // namespace

extern "C" bool DecoderPath_forward(
    OpenCLContext& cl_ctx,
    Weights& weights,
    cl_command_queue queue,
    cl_mem prosody,      // [128, F]
    cl_mem asr,          // [512, F]
    cl_mem style,        // [256]
    int F,
    cl_mem* out_decoded, // [256, 2F]
    cl_mem* out_f0_raw)  // [1, 2F]  (for the generator source module)
{
    if (!st::init(cl_ctx)) return false;
    const int F2 = 2 * F;

    std::vector<cl_mem> owned;
    auto own = [&](cl_mem m) { if (m) owned.push_back(m); return m; };
    auto cleanup = [&]() -> bool {
        for (cl_mem m : owned) if (m) clReleaseMemObject(m);
        owned.clear();
        return false;
    };

    cl_int err = CL_SUCCESS;
    cl_mem params = own(st::alloc_f32(cl_ctx, 2));
    if (!params) return cleanup();

    // ── style halves ─────────────────────────────────────────────────────
    cl_mem s_pros = own(st::alloc(cl_ctx, C_S));   // style[128:256] -> predictor
    cl_mem s_acou = own(st::alloc(cl_ctx, C_S));   // style[0:128]   -> decoder
    if (!s_pros || !s_acou) return cleanup();
    err  = clEnqueueCopyBuffer(queue, style, s_pros, C_S * sizeof(nnopt_storage_t), 0,
                               C_S * sizeof(nnopt_storage_t), 0, nullptr, nullptr);
    err |= clEnqueueCopyBuffer(queue, style, s_acou, 0, 0,
                               C_S * sizeof(nnopt_storage_t), 0, nullptr, nullptr);
    if (err != CL_SUCCESS) { NNOPT_ERROR_FMT("DecoderPath: style slice %d", err); return cleanup(); }

    cl_mem sp_q = own(st::alloc(cl_ctx, C_S));
    cl_mem sa_q = own(st::alloc(cl_ctx, C_S));
    if (!sp_q || !sa_q) return cleanup();
    if (!st::dql(queue, s_pros, params, sp_q, C_S)) return cleanup();
    if (!st::dql(queue, s_acou, params, sa_q, C_S)) return cleanup();

    // Hoisted AdaIN style fcs. fc(norm1) is [2*C_in], fc(norm2) is [2*C_out].
    auto make_fc = [&](cl_mem sq, const std::string& p, int c2) -> cl_mem {
        return own(st::adain_fc(cl_ctx, queue, weights, sq,
                                p + ".fc.weight_quantized", p + ".fc.bias", c2));
    };

    // ── F0 / N chains ────────────────────────────────────────────────────
    const Blk PRED[3] = {
        {"0", 128, 128, false, false},   // T: F   -> F
        {"1", 128,  64, true,  true },   // T: F   -> 2F
        {"2",  64,  64, false, false},   // T: 2F  -> 2F
    };
    cl_mem chain_out[2] = {nullptr, nullptr};
    const char* CHAIN[2] = {"F0", "N"};

    for (int ci = 0; ci < 2; ci++) {
        cl_mem cur = prosody;
        int t_in = F;
        for (int b = 0; b < 3; b++) {
            const std::string pfx = std::string("kmodel.predictor.") + CHAIN[ci] + "." + PRED[b].prefix;
            const int t_out = PRED[b].upsample ? 2 * t_in : t_in;
            cl_mem f1 = make_fc(sp_q, pfx + ".norm1", 2 * PRED[b].c_in);
            cl_mem f2 = make_fc(sp_q, pfx + ".norm2", 2 * PRED[b].c_out);
            if (!f1 || !f2) { NNOPT_ERROR_FMT("DecoderPath: fc %s", pfx.c_str()); return cleanup(); }
            cl_mem nxt = st::adain_resblk(cl_ctx, queue, weights, cur,
                                          PRED[b].c_in, PRED[b].c_out, t_in, t_out,
                                          pfx, f1, f2, PRED[b].upsample, PRED[b].conv1x1);
            if (!nxt) { NNOPT_ERROR_FMT("DecoderPath: resblk %s", pfx.c_str()); return cleanup(); }
            if (cur != prosody) clReleaseMemObject(cur);
            cur = nxt;
            t_in = t_out;
        }
        chain_out[ci] = own(cur);   // [64, 2F]
    }
    NNOPT_LAYER_CHECK("F0.2.Mul_output_0", queue, chain_out[0], (size_t)64 * F2);
    NNOPT_LAYER_CHECK("N.2.Mul_output_0",  queue, chain_out[1], (size_t)64 * F2);

    // ── F0_proj / N_proj : plain float Conv(64->1, k=1) ──────────────────
    cl_mem f0 = own(st::alloc(cl_ctx, (size_t)F2));
    cl_mem nn = own(st::alloc(cl_ctx, (size_t)F2));
    if (!f0 || !nn) return cleanup();
    {
        cl_mem fw = weights.get_buffer("kmodel.predictor.F0_proj.weight");
        cl_mem fb = weights.get_buffer("kmodel.predictor.F0_proj.bias");
        cl_mem nw = weights.get_buffer("kmodel.predictor.N_proj.weight");
        cl_mem nb = weights.get_buffer("kmodel.predictor.N_proj.bias");
        if (!fw || !fb || !nw || !nb) { NNOPT_ERROR("DecoderPath: missing F0/N proj"); return cleanup(); }
        if (!st::conv1d(queue, chain_out[0], fw, fb, f0, 64, 1, F2, F2, 1, 1, 0, 1)) return cleanup();
        if (!st::conv1d(queue, chain_out[1], nw, nb, nn, 64, 1, F2, F2, 1, 1, 0, 1)) return cleanup();
    }
    NNOPT_LAYER_CHECK("F0_proj.Conv_output_0_Cast_to_float16_input_0", queue, f0, (size_t)F2);
    NNOPT_LAYER_CHECK("N_proj.Conv_output_0_Cast_to_float32_output_0", queue, nn, (size_t)F2);

    // ── F0_conv / N_conv : Conv(1->1, k=3, stride=2, pad=1) : 2F -> F ────
    cl_mem f0d = own(st::alloc(cl_ctx, (size_t)F));
    cl_mem nnd = own(st::alloc(cl_ctx, (size_t)F));
    if (!f0d || !nnd) return cleanup();
    {
        cl_mem fw = weights.get_buffer("kmodel.decoder.decoder.F0_conv.weight");
        cl_mem fb = weights.get_buffer("kmodel.decoder.decoder.F0_conv.bias");
        cl_mem nw = weights.get_buffer("kmodel.decoder.decoder.N_conv.weight");
        cl_mem nb = weights.get_buffer("kmodel.decoder.decoder.N_conv.bias");
        if (!fw || !fb || !nw || !nb) { NNOPT_ERROR("DecoderPath: missing F0/N_conv"); return cleanup(); }
        if (!st::conv1d(queue, f0, fw, fb, f0d, 1, 1, F2, F, 3, 2, 1, 1)) return cleanup();
        if (!st::conv1d(queue, nn, nw, nb, nnd, 1, 1, F2, F, 3, 2, 1, 1)) return cleanup();
    }
    NNOPT_LAYER_CHECK("decoder.F0_conv.Conv_output_0", queue, f0d, (size_t)F);
    NNOPT_LAYER_CHECK("decoder.N_conv.Conv_output_0",  queue, nnd, (size_t)F);

    // ── asr_res = 1x1(asr) -> [64,F], computed once ──────────────────────
    cl_mem asr_res = own(st::alloc(cl_ctx, (size_t)64 * F));
    cl_mem asr_q   = own(st::alloc(cl_ctx, (size_t)512 * F));
    if (!asr_res || !asr_q) return cleanup();
    {
        cl_mem rw = weights.get_buffer("kmodel.decoder.decoder.asr_res.0.weight_quantized");
        cl_mem rb = weights.get_buffer("kmodel.decoder.decoder.asr_res.0.bias");
        if (!rw || !rb) { NNOPT_ERROR("DecoderPath: missing asr_res"); return cleanup(); }
        if (!st::dql(queue, asr, params, asr_q, 512 * F)) return cleanup();
        if (!st::conv1d(queue, asr_q, rw, rb, asr_res, 512, 64, F, F, 1, 1, 0, 1)) return cleanup();
    }
    NNOPT_LAYER_CHECK("decoder.asr_res.asr_res.0.Conv_output_0", queue, asr_res, (size_t)64 * F);

    // ── encode : Concat[asr(512), F0d(1), Nd(1)] = 514 -> 256 ────────────
    cl_mem cat514 = own(st::alloc(cl_ctx, (size_t)514 * F));
    if (!cat514) return cleanup();
    if (!st::concat_ct(queue, {asr, f0d, nnd}, {512, 1, 1}, cat514, F)) return cleanup();
    NNOPT_LAYER_CHECK("decoder.Concat_output_0", queue, cat514, (size_t)514 * F);

    cl_mem cur = nullptr;
    {
        const std::string p = "kmodel.decoder.decoder.encode";
        cl_mem f1 = make_fc(sa_q, p + ".norm1", 2 * 514);
        cl_mem f2 = make_fc(sa_q, p + ".norm2", 2 * 256);
        if (!f1 || !f2) return cleanup();
        cur = own(st::adain_resblk(cl_ctx, queue, weights, cat514, 514, 256, F, F,
                                   p, f1, f2, /*upsample=*/false, /*conv1x1=*/true));
        if (!cur) { NNOPT_ERROR("DecoderPath: encode failed"); return cleanup(); }
    }
    NNOPT_LAYER_CHECK("decoder.encode.Mul_output_0", queue, cur, (size_t)256 * F);

    // ── decode.0..3 : Concat[prev(256), asr_res(64), F0d, Nd] = 322 ──────
    for (int i = 0; i < 4; i++) {
        const bool up = (i == 3);
        const int t_out = up ? F2 : F;
        cl_mem cat322 = own(st::alloc(cl_ctx, (size_t)322 * F));
        if (!cat322) return cleanup();
        if (!st::concat_ct(queue, {cur, asr_res, f0d, nnd}, {256, 64, 1, 1}, cat322, F))
            return cleanup();

        const std::string p = "kmodel.decoder.decoder.decode." + std::to_string(i);
        cl_mem f1 = make_fc(sa_q, p + ".norm1", 2 * 322);
        cl_mem f2 = make_fc(sa_q, p + ".norm2", 2 * 256);
        if (!f1 || !f2) return cleanup();
        cl_mem nxt = st::adain_resblk(cl_ctx, queue, weights, cat322, 322, 256, F, t_out,
                                      p, f1, f2, up, /*conv1x1=*/true);
        if (!nxt) { NNOPT_ERROR_FMT("DecoderPath: decode.%d failed", i); return cleanup(); }
        cur = own(nxt);
        NNOPT_LAYER_CHECK(("decoder.decode." + std::to_string(i) + ".Mul_output_0").c_str(),
                          queue, cur, (size_t)256 * t_out);
    }

    NNOPT_DEBUG_SYNC(queue);

    // Detach the two outputs from the owned list before releasing the rest.
    cl_mem decoded = cur, f0_raw = f0;
    for (cl_mem m : owned) if (m && m != decoded && m != f0_raw) clReleaseMemObject(m);
    owned.clear();

    *out_decoded = decoded;
    *out_f0_raw  = f0_raw;
    return true;
}
