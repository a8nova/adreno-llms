// backbone.cpp — Moonshine ASR encoder-decoder COMPOSER.
// Reference: model_info/transformers_src/modeling_moonshine.py
//   MoonshineEncoder.forward (:520-623), MoonshineDecoder.forward (:624-722).
//
// Whisper-parity structure (see port/whisper-tiny-.../src/backbone.cpp): this
// file holds ONLY graph composition + per-waveform cache bookkeeping. All
// layer math lives in src/ops/:
//   MoonshineEncoderFrontend.cpp  conv1→tanh→groupnorm→conv2→gelu→conv3→gelu→permute
//   MoonshineEncoderLayer.cpp     ln → self-attn → resid → ln → MLP → resid
//   MoonshineDecoderLayer.cpp     ln → cached self-attn → cross-attn → MLP (+resids)
//   MoonshineAttention.cpp        all attention modes + RoPE + KV state (OPT-1/2)
//   Moonshine{Encoder,Decoder}MLP.cpp, Linear/LayerNorm/GroupNorm/Conv1d/Embedding.cpp
//
// Harness shape (per main.cpp decode loop):
//   - Encoder runs ONCE (cached keyed by the waveform buffer pointer).
//   - Decoder is INCREMENTAL (OPT-1): each forward processes only the rows the
//     KV cache hasn't seen; returns last-row logits.

#include "opencl_context.h"
#include "weights.h"
#include "debug_utils.h"
#include "model_config.h"
#include "utils.h"
#include "forward_dispatch.h"
#include "profiler.h"
#include <cstdio>
#include <string>
#include <vector>

// ── ops (implemented in src/ops/*.cpp) ──────────────────────────────────
extern "C" cl_mem LayerNorm_forward(OpenCLContext&, Weights&, cl_command_queue,
    cl_mem, int, int, int, cl_mem*, cl_mem*, cl_mem, const char*);
extern "C" cl_mem Linear_forward(OpenCLContext&, Weights&, cl_command_queue,
    cl_mem, int, int, int, cl_mem*, cl_mem*, cl_mem, const char*);
extern "C" cl_mem Embedding_forward(OpenCLContext&, Weights&, cl_command_queue,
    cl_mem, int, int, int, cl_mem*, cl_mem*, cl_mem, const char*);
extern "C" cl_mem MoonshineEncoderFrontend_forward(OpenCLContext&, Weights&, cl_command_queue,
    cl_mem, int, int, int, cl_mem*, cl_mem*, cl_mem, const char*);
extern "C" cl_mem MoonshineEncoderLayer_forward(OpenCLContext&, Weights&, cl_command_queue,
    cl_mem, int, int, int, cl_mem*, cl_mem*, cl_mem, const char*);
extern "C" cl_mem MoonshineDecoderLayer_forward(OpenCLContext&, Weights&, cl_command_queue,
    cl_mem, int, int, int, cl_mem*, cl_mem*, cl_mem, const char*);
extern "C" int  moonshine_frontend_out_len(int num_samples);
extern "C" bool MoonshineAttention_build_decoder_caches(
    OpenCLContext&, Weights&, cl_command_queue, cl_mem enc, int enc_T);

// ── ForwardDispatch state (audio handoff from main.cpp) ─────────────────
namespace {
    cl_mem g_input_features = nullptr;
    int    g_num_samples = 0;
    // Encoder-output cache (encoder runs once per waveform).
    cl_mem g_encoder_out = nullptr;       // [enc_T, 288]
    int    g_encoder_T = 0;
    cl_mem g_encoder_cached_for = nullptr; // waveform pointer this cache belongs to
    // Decoder cache cursor (the K/V buffers themselves live in
    // MoonshineAttention.cpp; this file tracks how many rows are cached and
    // which waveform they belong to).
    constexpr int KV_CAP = MODEL_CONFIG::MAX_SEQUENCE_LENGTH; // 194 (max_position_embeddings)
    int    g_kv_len = 0;                    // tokens whose self K/V are cached
    cl_mem g_kv_cached_for = nullptr;       // waveform pointer the caches belong to
}
namespace ForwardDispatch {
    void set_input_features(cl_mem feats) { g_input_features = feats; }
    cl_mem get_input_features() { return g_input_features; }
    void set_num_samples(int n) { g_num_samples = n; }
    int get_num_samples() { return g_num_samples; }
}

// Streaming hook: drop the per-waveform caches so the NEXT forward recomputes
// the encoder + decoder K/V. Required between sliding windows — the caches key
// on the waveform cl_mem POINTER, and a freed window buffer can be reallocated
// at the same address, silently reusing the previous window's encoder output
// (same trap whisper hit — see its backbone.cpp invalidator). The batch path
// never calls this and is unaffected.
extern "C" void MoonshineBackbone_invalidate_encoder_cache() {
    if (g_encoder_out) { clReleaseMemObject(g_encoder_out); g_encoder_out = nullptr; }
    g_encoder_cached_for = nullptr;
    g_encoder_T = 0;
    g_kv_cached_for = nullptr;
    g_kv_len = 0;
}

// Hard-abort sentinel: an EMPTY logits vector means the forward FAILED (encoder
// refusal, OOM, dispatch error) and generation must stop. The old zero-logits
// fallback let the decode loop grind max_new_tokens steps of argmax(0)=id 0 —
// 83s of garbage on a >25s clip (benchmark.md, clip H).
static std::vector<float> forward_failed() {
    return std::vector<float>();
}

static constexpr int HID = MODEL_CONFIG::HIDDEN_SIZE;   // 288

// ── ENCODER (runs once per waveform) ────────────────────────────────────
// Returns [enc_T, HID]; sets *enc_T_out. Caller owns the buffer.
static cl_mem run_encoder(OpenCLContext& cl_ctx, Weights& w, cl_command_queue q,
                          cl_mem waveform, int num_samples, int* enc_T_out) {
    cl_mem x = MoonshineEncoderFrontend_forward(cl_ctx,w,q,waveform,num_samples,-1,0,
                                                nullptr,nullptr,nullptr,"model.encoder");
    if(!x){NNOPT_ERROR("encoder frontend");return nullptr;}
    const int T = moonshine_frontend_out_len(num_samples);

    // 6 encoder layers.
    for (int i=0;i<MODEL_CONFIG::ENCODER_NUM_HIDDEN_LAYERS;++i){
        std::string base = "model.encoder.layers." + std::to_string(i);
        cl_mem xn = MoonshineEncoderLayer_forward(cl_ctx,w,q,x,T,i,0,
                                                  nullptr,nullptr,nullptr,base.c_str());
        clReleaseMemObject(x);
        if(!xn){return nullptr;}
        x = xn;
        // Reference dump: model_encoder_layers_<i> (MoonshineEncoderLayer output).
        char enc_layer_name[64];
        snprintf(enc_layer_name,sizeof(enc_layer_name),"model_encoder_layers_%d",i);
        NNOPT_LAYER_CHECK(enc_layer_name, q, x, (size_t)T*HID);
    }
    // final layer_norm
    cl_mem xn = LayerNorm_forward(cl_ctx,w,q,x,T,-1,0,nullptr,nullptr,nullptr,"model.encoder.layer_norm");
    clReleaseMemObject(x);
    if(!xn){NNOPT_ERROR("enc final ln");return nullptr;}
    NNOPT_LAYER_CHECK("model_encoder_layer_norm", q, xn, (size_t)T*HID);
    NNOPT_LAYER_CHECK("model_encoder", q, xn, (size_t)T*HID);
    *enc_T_out = T;
    return xn;
}

// ── MAIN GRAPH ──────────────────────────────────────────────────────────
std::vector<float> model_forward_graph(
    OpenCLContext& cl_ctx,
    Weights& weights,
    const std::vector<int32_t>& input_ids,
    int start_pos)
{
    (void)start_pos;
    NNOPT_CHECKPOINT("model_forward_graph (moonshine enc-dec)");
    cl_command_queue queue = cl_ctx.queue();
    const int seq = (int)input_ids.size();

    // ── Encoder (cache once per waveform) ──
    cl_mem wav = ForwardDispatch::get_input_features();
    int num_samples = ForwardDispatch::get_num_samples();
    if(!wav){NNOPT_ERROR("no input features set");return forward_failed();}
    if(g_encoder_out == nullptr || g_encoder_cached_for != wav){
        if(g_encoder_out){clReleaseMemObject(g_encoder_out);g_encoder_out=nullptr;}
        int T=0;
        g_encoder_out = run_encoder(cl_ctx,weights,queue,wav,num_samples,&T);
        if(!g_encoder_out){NNOPT_ERROR("encoder failed");return forward_failed();}
        g_encoder_T = T;
        g_encoder_cached_for = wav;
    }
    cl_mem enc = g_encoder_out;
    const int enc_T = g_encoder_T;

    // ── Decoder (incremental, KV-cached — OPT-1) ──
    // Rebuild caches when the waveform changed or the requested sequence is
    // not a continuation of what's cached (fresh transcript / retry).
    if (g_kv_cached_for != wav || seq <= g_kv_len) {
        if (!MoonshineAttention_build_decoder_caches(cl_ctx, weights, queue, enc, enc_T))
            return forward_failed();
        g_kv_len = 0;
        g_kv_cached_for = wav;
    }
    if (seq > KV_CAP) {
        NNOPT_ERROR_FMT("decoder_ids length %d exceeds KV_CAP=%d (max_position_embeddings) — refusing to silently truncate",
                        seq, KV_CAP);
        return forward_failed();
    }
    const int past  = g_kv_len;
    const int t_new = seq - past;   // prefill: past=0 ⇒ t_new=seq; decode step: 1

    // embed ONLY the new tokens
    cl_int err=CL_SUCCESS;
    cl_mem ids_buf = clCreateBuffer(cl_ctx.context(),
        CL_MEM_READ_WRITE|CL_MEM_COPY_HOST_PTR,
        (size_t)t_new*sizeof(int32_t),(void*)(input_ids.data()+past),&err);  // UPLOAD-OK: input_ids
    if(!ids_buf){NNOPT_ERROR("ids alloc");return forward_failed();}
    cl_mem x = Embedding_forward(cl_ctx,weights,queue,ids_buf,t_new,-1,0,nullptr,nullptr,nullptr,"model.decoder.embed_tokens");
    clReleaseMemObject(ids_buf);
    if(!x){NNOPT_ERROR("embed failed");return forward_failed();}
    NNOPT_LAYER_CHECK("model_decoder_embed_tokens", queue, x, (size_t)t_new*HID);

    for(int i=0;i<MODEL_CONFIG::DECODER_NUM_HIDDEN_LAYERS;++i){
        std::string base = "model.decoder.layers." + std::to_string(i);
        cl_mem xn = MoonshineDecoderLayer_forward(cl_ctx,weights,queue,x,t_new,i,past,
                                                  nullptr,nullptr,enc,base.c_str());
        clReleaseMemObject(x);
        if(!xn){return forward_failed();}
        x = xn;
        // Reference dump: model_decoder_layers_<i> — NEW rows only (pass 0 has
        // t_new==seq so prefill dumps are unchanged vs the reference).
        char dec_layer_name[64];
        snprintf(dec_layer_name,sizeof(dec_layer_name),"model_decoder_layers_%d",i);
        NNOPT_LAYER_CHECK(dec_layer_name, queue, x, (size_t)t_new*HID);
    }
    // final decoder norm (new rows only)
    cl_mem xn = LayerNorm_forward(cl_ctx,weights,queue,x,t_new,-1,0,nullptr,nullptr,nullptr,"model.decoder.norm");
    clReleaseMemObject(x);
    if(!xn){return forward_failed();}
    NNOPT_LAYER_CHECK("model_decoder_norm", queue, xn, (size_t)t_new*HID);
    NNOPT_LAYER_CHECK("model_decoder", queue, xn, (size_t)t_new*HID);
    NNOPT_LAYER_CHECK("model", queue, xn, (size_t)t_new*HID);

    // Advance the cache cursor only after the full stack succeeded.
    g_kv_len = seq;

    // proj_out — tied to embed_tokens.weight [vocab, hid]; nn.Linear (y = x @ W^T).
    // Only the LAST row's logits are ever consumed, so project 1 row.
    cl_mem last_row = xn;
    cl_mem sub = nullptr;
    if (t_new > 1) {
        cl_buffer_region reg{(size_t)(t_new-1)*HID*sizeof(nnopt_storage_t), (size_t)HID*sizeof(nnopt_storage_t)};
        sub = clCreateSubBuffer(xn, CL_MEM_READ_ONLY, CL_BUFFER_CREATE_TYPE_REGION, &reg, &err);
        if (sub) last_row = sub;   // alignment failure → fall back to full projection below
    }
    const int proj_rows = (last_row == xn) ? t_new : 1;
    cl_mem logits = Linear_forward(cl_ctx,weights,queue,last_row,proj_rows,-1,0,nullptr,nullptr,nullptr,"model.decoder.embed_tokens");
    if (sub) clReleaseMemObject(sub);
    clReleaseMemObject(xn);
    if(!logits){NNOPT_ERROR("proj_out failed");return forward_failed();}
    NNOPT_LAYER_CHECK("proj_out", queue, logits, (size_t)proj_rows*MODEL_CONFIG::VOCAB_SIZE);

    // read last row logits.
    const int V = MODEL_CONFIG::VOCAB_SIZE;
    std::vector<nnopt_storage_t> last(V);
    size_t offset = (size_t)(proj_rows-1)*V*sizeof(nnopt_storage_t);
    clEnqueueReadBuffer(queue, logits, CL_TRUE, offset, V*sizeof(nnopt_storage_t), last.data(), 0, nullptr,
                        KernelProfiler::event_for("readback_logits"));
    clReleaseMemObject(logits);
    std::vector<float> out(V);
#ifdef NNOPT_USE_FP16
    for(int i=0;i<V;++i) out[i] = nnopt_f16_to_f32(static_cast<uint16_t>(last[i]));
#else
    for(int i=0;i<V;++i) out[i] = last[i];
#endif
    return out;
}
