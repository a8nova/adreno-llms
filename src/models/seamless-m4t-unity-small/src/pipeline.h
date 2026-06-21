#pragma once
// SeamlessM4T UnitY end-to-end pipeline, on-device. audio -> waveform.
// Math mirrors oracles/pipeline_ref.cpp; heavy compute runs on the GPU via
// GpuOps. Beam/greedy control flow is host scalar code (cheap).
#include <string>
#include <vector>

#include "gpu_ops.h"
#include "weights.h"

// Shared model dims + activation codes (match kernels/_preamble.cl::act_apply).
static const int Dm = 768, H = 16, Dk = 48;
enum { ACT_GELU = 0, ACT_SWISH = 1, ACT_RELU = 2, ACT_TANH = 3, ACT_SIGMOID = 4, ACT_LRELU01 = 5, ACT_LRELU001 = 6 };
inline int get_padding(int k, int d) { return (k * d - d) / 2; }
// Load a float32 .bin from disk (debug fixtures / side files). Defined in pipeline.cpp.
std::vector<float> load_file_floats(const std::string& path);

// Incremental-decode KV cache for one hypothesis. Cross-attn K/V are computed
// once (the memory is fixed across the decode); self-attn K/V grow per step.
struct KVCache {
    std::string dir;
    int nlayers = 0, Tmem = 0, max_len = 0, vocab = 0, len = 0;
    std::vector<cl_mem> ck, cv;   // cross-attn K/V per layer  [Tmem*Dm]   (shareable)
    std::vector<cl_mem> sk, sv;   // self-attn  K/V per layer  [max_len*Dm]
};

class Pipeline {
public:
    Pipeline(GpuOps& ops, Weights& w) : ops_(ops), w_(w) {}

    // Stage 1: CodeHiFiGAN vocoder. units (host, length T) -> waveform (length T*320).
    std::vector<float> vocoder(const std::vector<float>& units);

    // Stage 2: kaldi fbank. audio[16000] -> log-mel [nframes, 80] (host compute).
    std::vector<float> fbank(const std::vector<float>& audio, int& nframes);

    // Stage 3: speech encoder. fbank[Nframes,80] -> encoder_out [Tout,768] (host vec).
    std::vector<float> encoder(const std::vector<float>& fb, int nframes, int& Tout);

    // Stage 4/5: transformer decoders.
    std::vector<int> text_beam_search(const std::vector<float>& enc, int Tenc);
    std::vector<float> synth_encoder(const std::vector<float>& hidden, int T);
    std::vector<int> unit_greedy_decode(const std::vector<float>& mem, int Tmem);

    // Full pipeline. audio -> {units, waveform}.
    void run(const std::vector<float>& audio, std::vector<float>& units_out,
             std::vector<float>& waveform_out);

    // Decode-only path (debug bisect): given an encoder output [Tenc,768], run
    // text-beam -> synth -> unit-greedy -> vocoder. Lets us inject a reference
    // encoder to isolate decode-path bugs.
    void run_from_encoder(const std::vector<float>& enc, int Tenc,
                          std::vector<float>& units_out, std::vector<float>& waveform_out);

    // Target-language control tokens (from the .ptl's lang_tok_map / lang_tok_map_unit /
    // vocoder_lang_map / vocoder_spkr_map). Defaults to English; set per request.
    void set_lang(int text_prefix, int unit_prefix, int voc_lang, int voc_spkr) {
        text_prefix_ = text_prefix; unit_prefix_ = unit_prefix; voc_lang_ = voc_lang; voc_spkr_ = voc_spkr;
    }

private:
    GpuOps& ops_;
    Weights& w_;
    int text_prefix_ = 20000, unit_prefix_ = 10010, voc_lang_ = 8, voc_spkr_ = 10;  // eng default

    // shared decoder helpers (text + unit decoders); dir = device key prefix.
    // decoder_features returns the GPU hidden-states tensor [T*768]; logits_last
    // returns the fp16 logits tensor [vocab] for the final position (tied proj).
    Tensor decoder_features(const std::string& dir, const std::vector<int>& seq,
                            const Tensor& mem, int Tmem, int nlayers);
    Tensor decoder_logits_last(const std::string& dir, const std::vector<int>& seq,
                               const Tensor& mem, int Tmem, int nlayers, int vocab);
    Tensor mha(const std::string& dir, const Tensor& xq, int Tq, const Tensor& xkv, int Tk,
               const std::string& pre, bool causal);

    // Incremental (KV-cached) decode. make_cache precomputes cross-attn K/V and
    // allocates self-attn cache buffers; decode_step processes one token at
    // position `pos`, appends its self K/V, and returns the fp16 logits [vocab].
    KVCache make_cache(const std::string& dir, const Tensor& mem, int Tmem, int nlayers,
                       int vocab, int max_len);
    Tensor decode_step(KVCache& c, int token, int pos);
    Tensor decode_step_ids(KVCache& c, cl_mem tok_buf, int pos);  // token from device buffer (chained decode)

    // Batched incremental decode for beam search: processes B beams' tokens in one
    // pass — matmuls run as a single M=B GEMM, cross-attention batches into one call
    // (shared K/V), self-attention stays per-beam (each beam reads its own cache at
    // sk/sv[b*nlayers+L], appended at `step`). Returns logits [B, vocab].
    Tensor decode_step_batch(const std::string& dir, int B, const std::vector<int>& toks,
                             int step, int nlayers, int Tmem, int vocab,
                             const std::vector<cl_mem>& ck, const std::vector<cl_mem>& cv,
                             const std::vector<cl_mem>& sk, const std::vector<cl_mem>& sv);
};
