// pipeline.cpp — end-to-end pocket-tts decode. See pipeline.h / PORT_SPEC.md.
//
// v1 scope: backbone KV cache IS carried across the generate loop (mandatory —
// the model is autoregressive). The Mimi per-frame streaming state (causal-conv
// left-context, convtr overlap tails, decoder-transformer KV across frames) is
// NOT yet carried — each frame decodes with zero state, which produces audio
// with mild frame-boundary seams (PORT_SPEC §6.8). Backbone + flow path are full
// fidelity; Mimi cross-frame state is the remaining quality pass.
#include "pipeline.h"
#include "nnopt_error.h"
#include "profiler.h"
#include <string>
#include <vector>
#include <random>
#include <cmath>
#include <chrono>
#include <cstdio>
#include <cstdlib>

// Debug: dump a device buffer's stats (rms, range, zero-crossing rate) to stderr
// so we can see WHERE the signal becomes noise-like in the pipeline.
static void dbg_stats(GpuOps& g, const char* tag, cl_mem b, size_t n) {
    std::vector<float> v = g.download(b, n);
    double ss = 0; float mn = v[0], mx = v[0]; int zc = 0;
    for (size_t i = 0; i < n; ++i) { ss += (double)v[i]*v[i]; if (v[i]<mn) mn=v[i]; if (v[i]>mx) mx=v[i];
        if (i>0 && ((v[i]>0)!=(v[i-1]>0))) zc++; }
    fprintf(stderr, "  DBG %-22s n=%zu rms=%.4f range[%.3f,%.3f] zcr=%.3f\n",
            tag, n, std::sqrt(ss/n), mn, mx, (double)zc/n);
}

// Backbone: input [T,1024] → 6 StreamingTransformerLayers (RoPE, KV cache at
// `offset`, unbounded causal, no layer_scale). Returns [T,1024] (pre out_norm).
static cl_mem backbone_run(GpuOps& g, cl_mem x, int T, int offset, cl_mem* kc, cl_mem* vc) {
    cl_mem cur = x; bool owned = false;
    for (int i = 0; i < 6; ++i) {
        cl_mem nx = g.transformer_layer(cur, T, 1024, 16, 4096, offset, -1,
                        "flow_lm.transformer.layers." + std::to_string(i),
                        false, kc[i], vc[i], 10000.0f);
        if (owned) g.release(cur);
        cur = nx; owned = true;
    }
    return cur;
}

// Cross-frame Mimi streaming state (causal-conv left-contexts, convtr overlap
// tails, decoder-transformer KV). Zero-init on construction.
struct MimiState {
    GpuOps* g = nullptr;
    cl_mem up_partial = nullptr;            // upsample convtr [512, 16]
    // dt_off grows by 16/frame (absolute KV position, no ring-wrap in the kernel), so
    // the cache must hold the whole clip: 2048 ≈ 128 frames ≈ 10 s. (Attention still
    // only reads the last `context`=250 positions, but the WRITE index is absolute.)
    cl_mem dt_kc[2] = {0,0}, dt_vc[2] = {0,0}; int dt_off = 0; static const int DT_CAP = 2048;
    cl_mem c0_ctx = nullptr;                // model_0 conv k7 [512,6]
    cl_mem ct_part[3] = {0,0,0};            // model_2/5/8 convtr partials
    cl_mem rb_ctx[3] = {0,0,0};             // model_3/6/9 resnet block.1 k3 ctx
    cl_mem c11_ctx = nullptr;               // model_11 conv k3 [64,2]
    static cl_mem z(GpuOps& g, size_t n) { std::vector<float> zr(n,0.0f); return g.upload(zr); }
    void init(GpuOps& gg) {
        g = &gg;
        up_partial = z(gg, 512*16);
        for (int i=0;i<2;++i){ dt_kc[i]=z(gg,(size_t)DT_CAP*512); dt_vc[i]=z(gg,(size_t)DT_CAP*512); }
        c0_ctx = z(gg, 512*6);
        ct_part[0]=z(gg,256*6); ct_part[1]=z(gg,128*5); ct_part[2]=z(gg,64*4);
        rb_ctx[0]=z(gg,256*2); rb_ctx[1]=z(gg,128*2); rb_ctx[2]=z(gg,64*2);
        c11_ctx = z(gg, 64*2);
    }
    void free_() {
        cl_mem all[]={up_partial,dt_kc[0],dt_vc[0],dt_kc[1],dt_vc[1],c0_ctx,ct_part[0],ct_part[1],ct_part[2],rb_ctx[0],rb_ctx[1],rb_ctx[2],c11_ctx};
        for (cl_mem m: all) if(m) g->release(m);
    }
    // Zero all streaming context IN PLACE (same handles), keeping the buffers the
    // recording captured. The QCOM recording pass EXECUTES the captured frame once,
    // advancing streaming state; reset before replaying so step 0 starts fresh.
    void reset_zero() {
        struct BZ { cl_mem b; size_t n; };
        BZ bs[]={{up_partial,512*16},{dt_kc[0],(size_t)DT_CAP*512},{dt_vc[0],(size_t)DT_CAP*512},
                 {dt_kc[1],(size_t)DT_CAP*512},{dt_vc[1],(size_t)DT_CAP*512},{c0_ctx,512*6},
                 {ct_part[0],256*6},{ct_part[1],128*5},{ct_part[2],64*4},
                 {rb_ctx[0],256*2},{rb_ctx[1],128*2},{rb_ctx[2],64*2},{c11_ctx,64*2}};
        for (auto& z : bs) g->zero_buffer(z.b, z.n);
        dt_off = 0;
    }
};

// Mimi decode of one latent [32] → 1920 fp32 samples, carrying streaming state.
static cl_mem mimi_decode_frame(GpuOps& g, cl_mem latent32, MimiState& S,
                                double* xf_ms = nullptr, double* seanet_ms = nullptr) {
    auto _mt0 = std::chrono::steady_clock::now();
    cl_mem dn = g.denorm_latent(latent32, 1, 32, "flow_lm.emb_std", "flow_lm.emb_mean"); // [1,32]
    cl_mem x  = g.transpose2d(dn, 1, 32);                                                // [32,1]
    cl_mem q  = g.conv1d_causal(x, 32, 512, 1, 1, 1, "mimi.quantizer.output_proj.weight", ""); // [512,1]
    cl_mem up = g.convtranspose1d_depthwise_streaming(q, S.up_partial, 512, 1, 32, 16, "mimi.upsample.convtr.convtr.weight"); // [512,16]
    cl_mem ti = g.transpose2d(up, 512, 16);                                              // [16,512]
    // decoder_transformer (2 layers), PERSISTENT KV across frames at S.dt_off.
    cl_mem l0 = g.transformer_layer(ti, 16, 512, 8, 2048, S.dt_off, 250,
                    "mimi.decoder_transformer.transformer.layers.0", true, S.dt_kc[0], S.dt_vc[0], 10000.0f);
    cl_mem l1 = g.transformer_layer(l0, 16, 512, 8, 2048, S.dt_off, 250,
                    "mimi.decoder_transformer.transformer.layers.1", true, S.dt_kc[1], S.dt_vc[1], 10000.0f);
    S.dt_off += 16;
    cl_mem dt = g.transpose2d(l1, 16, 512);                                              // [512,16]
    if (xf_ms || seanet_ms) clFinish(g.q());
    auto _mt1 = std::chrono::steady_clock::now();
    if (xf_ms) *xf_ms = std::chrono::duration<double, std::milli>(_mt1 - _mt0).count();
    // SEANetDecoder model 0..11 with streaming conv/convtr state.
    const std::string P = "mimi.decoder.model";
    cl_mem c0  = g.conv1d_streaming(dt, S.c0_ctx, 512, 512, 16, 7, 1, P + ".0.conv.weight", P + ".0.conv.bias");
    cl_mem c1  = g.elu(c0, 512 * 16);
    cl_mem c2  = g.convtranspose1d_streaming(c1, S.ct_part[0], 512, 256, 16, 12, 6, P + ".2.convtr.weight", P + ".2.convtr.bias");
    cl_mem c3  = g.seanet_resnet_block_streaming(c2, 256, 96, P + ".3", S.rb_ctx[0]);
    cl_mem c4  = g.elu(c3, 256 * 96);
    cl_mem c5  = g.convtranspose1d_streaming(c4, S.ct_part[1], 256, 128, 96, 10, 5, P + ".5.convtr.weight", P + ".5.convtr.bias");
    cl_mem c6  = g.seanet_resnet_block_streaming(c5, 128, 480, P + ".6", S.rb_ctx[1]);
    cl_mem c7  = g.elu(c6, 128 * 480);
    cl_mem c8  = g.convtranspose1d_streaming(c7, S.ct_part[2], 128, 64, 480, 8, 4, P + ".8.convtr.weight", P + ".8.convtr.bias");
    cl_mem c9  = g.seanet_resnet_block_streaming(c8, 64, 1920, P + ".9", S.rb_ctx[2]);
    cl_mem c10 = g.elu(c9, 64 * 1920);
    cl_mem wave = g.conv1d_streaming(c10, S.c11_ctx, 64, 1, 1920, 3, 1, P + ".11.conv.weight", P + ".11.conv.bias"); // [1,1920]
    if (seanet_ms) { clFinish(g.q()); *seanet_ms = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - _mt1).count(); }
    cl_mem rel[] = {dn,x,q,up,ti,l0,l1,dt,c0,c1,c2,c3,c4,c5,c6,c7,c8,c9,c10};
    for (cl_mem m : rel) g.release(m);
    return wave;
}

std::vector<float> mimi_decode_one(GpuOps& g, const std::vector<float>& latent32) {
    MimiState S; S.init(g);
    cl_mem l = g.upload(latent32);
    cl_mem w = mimi_decode_frame(g, l, S);
    std::vector<float> out = g.download(w, 1920);
    g.release(l); g.release(w); S.free_();
    return out;
}

// Non-streaming decode: process ALL N latents as one [32,N] sequence. Must equal
// the streaming frame-by-frame result; any difference localizes a streaming bug.
static std::vector<float> mimi_decode_nonstream(GpuOps& g, const std::vector<float>& lats, int N) {
    cl_mem L = g.upload(lats);                       // [N,32]
    cl_mem dn = g.denorm_latent(L, N, 32, "flow_lm.emb_std", "flow_lm.emb_mean"); // [N,32]
    cl_mem x = g.transpose2d(dn, N, 32);             // [32,N]
    cl_mem q = g.conv1d_causal(x, 32, 512, N, 1, 1, "mimi.quantizer.output_proj.weight", ""); // [512,N]
    cl_mem up = g.convtranspose1d_depthwise(q, 512, N, 32, 16, "mimi.upsample.convtr.convtr.weight"); // [512,N*16]
    int T = N * 16;
    cl_mem ti = g.transpose2d(up, 512, T);           // [N*16,512]
    cl_mem kc0=g.alloc((size_t)T*512), vc0=g.alloc((size_t)T*512), kc1=g.alloc((size_t)T*512), vc1=g.alloc((size_t)T*512);
    cl_mem l0 = g.transformer_layer(ti, T, 512, 8, 2048, 0, 250, "mimi.decoder_transformer.transformer.layers.0", true, kc0, vc0, 10000.0f);
    cl_mem l1 = g.transformer_layer(l0, T, 512, 8, 2048, 0, 250, "mimi.decoder_transformer.transformer.layers.1", true, kc1, vc1, 10000.0f);
    cl_mem dt = g.transpose2d(l1, T, 512);           // [512,N*16]
    const std::string P = "mimi.decoder.model";
    cl_mem c0=g.conv1d_causal(dt,512,512,T,7,1,P+".0.conv.weight",P+".0.conv.bias");
    cl_mem c1=g.elu(c0,(size_t)512*T);
    cl_mem c2=g.convtranspose1d(c1,512,256,T,12,6,P+".2.convtr.weight",P+".2.convtr.bias"); int T2=T*6;
    cl_mem c3=g.seanet_resnet_block(c2,256,T2,P+".3");
    cl_mem c4=g.elu(c3,(size_t)256*T2);
    cl_mem c5=g.convtranspose1d(c4,256,128,T2,10,5,P+".5.convtr.weight",P+".5.convtr.bias"); int T5=T2*5;
    cl_mem c6=g.seanet_resnet_block(c5,128,T5,P+".6");
    cl_mem c7=g.elu(c6,(size_t)128*T5);
    cl_mem c8=g.convtranspose1d(c7,128,64,T5,8,4,P+".8.convtr.weight",P+".8.convtr.bias"); int T8=T5*4;
    cl_mem c9=g.seanet_resnet_block(c8,64,T8,P+".9");
    cl_mem c10=g.elu(c9,(size_t)64*T8);
    cl_mem wave=g.conv1d_causal(c10,64,1,T8,3,1,P+".11.conv.weight",P+".11.conv.bias"); // [1,N*1920]
    std::vector<float> out = g.download(wave, (size_t)T8);
    cl_mem rel[]={L,dn,x,q,up,ti,kc0,vc0,kc1,vc1,l0,l1,dt,c0,c1,c2,c3,c4,c5,c6,c7,c8,c9,c10,wave};
    for (cl_mem m: rel) g.release(m);
    return out;
}

std::vector<float> mimi_decode_seq(GpuOps& g, const std::vector<float>& latent32, int n_frames) {
    MimiState S; S.init(g);
    std::vector<float> wave;
    for (int i = 0; i < n_frames; ++i) {
        cl_mem l = g.upload(latent32);
        cl_mem w = mimi_decode_frame(g, l, S);
        std::vector<float> s = g.download(w, 1920);
        wave.insert(wave.end(), s.begin(), s.end());
        g.release(l); g.release(w);
    }
    S.free_();
    return wave;
}

std::vector<float> mimi_decode_seq_vec(GpuOps& g, const std::vector<float>& lats, int N) {
    MimiState S; S.init(g);
    std::vector<float> wave;
    for (int i = 0; i < N; ++i) {
        std::vector<float> one(lats.begin() + i * 32, lats.begin() + i * 32 + 32);
        cl_mem l = g.upload(one);
        cl_mem w = mimi_decode_frame(g, l, S);
        std::vector<float> s = g.download(w, 1920);
        wave.insert(wave.end(), s.begin(), s.end());
        g.release(l); g.release(w);
    }
    S.free_();
    return wave;
}
std::vector<float> mimi_decode_nonstream_pub(GpuOps& g, const std::vector<float>& lats, int N) {
    return mimi_decode_nonstream(g, lats, N);
}

std::vector<float> tts_generate(GpuOps& g, int n_frames, float noise_std,
                                const std::vector<int>& text_ids,
                                const std::vector<float>* noise_seq,
                                std::vector<float>* c_out, std::vector<float>* latent_out,
                                bool stop_on_eos, float eos_threshold) {
    const int CAP = 512;   // backbone KV capacity (prefill offset + decode frames); ~30s headroom
    cl_mem kc[6], vc[6];
    for (int i = 0; i < 6; ++i) { kc[i] = g.alloc((size_t)CAP * 1024); vc[i] = g.alloc((size_t)CAP * 1024); }

    using _clk = std::chrono::steady_clock;
    auto _t_prefill = _clk::now();

    // 1) prime KV with bos_before_voice ++ audio_prompt conditioning. The captured
    //    `audio_prompt` is the 125-frame conditioning WITHOUT the BOS marker
    //    (cos(ap[0],bos_before_voice)=0.09), so prepend bos_before_voice — the real
    //    get_state_for_audio_prompt does `cat([bos_before_voice, conditioning])`.
    std::vector<float> ap = g.weights().get_host_vec("audio_prompt");              // [125,1024]
    int prompt_len = 1 + (int)(ap.size() / 1024);                                  // bos + 125 = 126
    // The voice-prompt prime (T=126) processes the FIXED audio_prompt weight, so its KV
    // cache is identical every generation. Compute once, persist to weights/voice_kv.bin,
    // and LOAD it on later runs (skips ~1.3 s of prefill). Raw fp16 bytes.
    const size_t es = sizeof(nnopt_storage_t);
    const size_t kvbytes = (size_t)prompt_len * 1024 * es;
    const char* kvpath = "weights/voice_kv.bin";
    FILE* kf = fopen(kvpath, "rb");
    if (kf) {
        std::vector<unsigned char> buf(kvbytes);
        for (int i = 0; i < 6; ++i) {
            if (fread(buf.data(), 1, kvbytes, kf) == kvbytes)
                clEnqueueWriteBuffer(g.q(), kc[i], CL_FALSE, 0, kvbytes, buf.data(), 0, nullptr, nullptr);
            if (fread(buf.data(), 1, kvbytes, kf) == kvbytes)
                clEnqueueWriteBuffer(g.q(), vc[i], CL_FALSE, 0, kvbytes, buf.data(), 0, nullptr, nullptr);
        }
        fclose(kf);
        fprintf(stderr, "loaded cached voice KV (%d frames) — skipped prime\n", prompt_len);
    } else {
        std::vector<float> bbv = g.weights().get_host_vec("flow_lm.bos_before_voice"); // [1024]
        std::vector<float> prime; prime.reserve(bbv.size() + ap.size());
        prime.insert(prime.end(), bbv.begin(), bbv.end());
        prime.insert(prime.end(), ap.begin(), ap.end());
        cl_mem apb = g.upload(prime);
        cl_mem pr = backbone_run(g, apb, prompt_len, 0, kc, vc);
        g.release(pr); g.release(apb);
        clFinish(g.q());
        FILE* wf = fopen(kvpath, "wb");
        if (wf) {
            std::vector<unsigned char> buf(kvbytes);
            for (int i = 0; i < 6; ++i) {
                clEnqueueReadBuffer(g.q(), kc[i], CL_TRUE, 0, kvbytes, buf.data(), 0, nullptr, nullptr);
                fwrite(buf.data(), 1, kvbytes, wf);
                clEnqueueReadBuffer(g.q(), vc[i], CL_TRUE, 0, kvbytes, buf.data(), 0, nullptr, nullptr);
                fwrite(buf.data(), 1, kvbytes, wf);
            }
            fclose(wf);
            fprintf(stderr, "computed + cached voice KV → %s\n", kvpath);
        }
    }
    int offset = prompt_len;
    double _prime_host = std::chrono::duration<double, std::milli>(_clk::now() - _t_prefill).count();
    fprintf(stderr, "primed KV with %d frames (bos_before_voice + %d conditioning)\n", prompt_len, (int)(ap.size()/1024));

    // 2) prime text tokens.
    auto _t_text = _clk::now();
    for (int tok : text_ids) {
        cl_mem emb = g.embedding(std::vector<int>{tok}, 1024, "flow_lm.conditioner.embed.weight");
        cl_mem p = backbone_run(g, emb, 1, offset, kc, vc);
        g.release(p); g.release(emb); offset++;
    }
    double _text_host = std::chrono::duration<double, std::milli>(_clk::now() - _t_text).count();
    double _enq_pre = std::chrono::duration<double, std::milli>(_clk::now() - _t_prefill).count();

    clFinish(g.q());
    double _prefill_ms = std::chrono::duration<double, std::milli>(_clk::now() - _t_prefill).count();
    fprintf(stderr, "PHASE prefill_ms: %.1f  (%d positions)  [prime(126)_host=%.1f text(%zu)_host=%.1f host_enq=%.1f gpu_tail=%.1f]\n",
            _prefill_ms, offset, _prime_host, text_ids.size(), _text_host, _enq_pre, _prefill_ms - _enq_pre);
    fprintf(stderr, "PREFILL "); KernelProfiler::dump_host_profile();

    // 3b) ── RECORD/REPLAY decode path (cl_qcom_recordable_queues), NNOPT_RECORD=1 ──
    //     Record the pure-NDRange span [input_linear→backbone→layernorm→eos→flow→next]
    //     ONCE, then replay per frame (host issues 1 call instead of ~100), patching
    //     `offset`/`Tkv` via arg-override. mimi stays live (CLBlast not recordable).
    const char* _rec_env = getenv("NNOPT_RECORD");
    bool use_record = _rec_env && atoi(_rec_env) && g.can_record()
                      && !noise_seq && !c_out && !latent_out;
    if (use_record) {
        const size_t es = sizeof(nnopt_storage_t);
        MimiState S; S.init(g);
        std::mt19937 rng(1234);
        std::normal_distribution<float> gauss(0.0f, 1.0f);
        std::vector<float> bosv = g.weights().get_host_vec("flow_lm.bos_emb");
        cl_mem cur = g.upload(bosv);                       // [32] fixed latent (updated via copy)
        cl_mem noise_fixed = g.alloc(32);                  // [32] fixed noise (device-copied per frame)
        // all noise up-front → one upload; per-frame device copy avoids host writes.
        std::vector<float> all_noise((size_t)n_frames * 32);
        for (auto& v : all_noise) v = noise_std * gauss(rng);
        cl_mem noise_big = g.upload(all_noise);
        // warm tc_cache so the recording's flow_net reuses it (timestep not recorded)
        { cl_mem cdum = g.alloc(1024); cl_mem w = g.flow_net(cdum, 0.0f, 1.0f, noise_fixed);
          g.release(w); g.release(cdum); }
        clFinish(g.q());

        const bool rec_mimi = g.rec_mimi_;   // include mimi in the recording (default off; slow tiled GEMM)
        int rec_offset = offset;
        if (!g.begin_recording()) { fprintf(stderr, "begin_recording failed\n"); return {}; }
        // backbone uses offb_bb_ (decode position); mimi (if recorded) uses offb_mm_ (dt_off).
        g.use_mimi_offset(false);
        cl_mem inp  = g.linear(cur, 1, 1024, 32, "flow_lm.input_linear.weight");
        cl_mem h    = backbone_run(g, inp, 1, rec_offset, kc, vc);
        cl_mem c    = g.layernorm(h, 1, 1024, 1e-5f, "flow_lm.out_norm.weight", "flow_lm.out_norm.bias");
        cl_mem eosb = g.linear(c, 1, 1, 1024, "flow_lm.out_eos.weight", "flow_lm.out_eos.bias"); (void)eosb;
        cl_mem flow = g.flow_net(c, 0.0f, 1.0f, noise_fixed);
        cl_mem next = g.add(noise_fixed, flow, 32);        // pinned output latent
        cl_mem wave_rec = nullptr;
        if (rec_mimi) {
            g.use_mimi_offset(true);                        // mimi transformer → offb_mm_
            wave_rec = mimi_decode_frame(g, next, S);       // recorded (forces slow direct CLBlast)
            g.use_mimi_offset(false);
        }
        cl_recording_qcom rec = g.end_recording();
        // The recording pass EXECUTED the captured frame once (advancing mimi streaming
        // state + backbone KV). Reset mimi streaming so replay step 0 starts fresh; the
        // backbone KV slot at rec_offset is idempotently re-written by step 0.
        if (rec_mimi) S.reset_zero();
        clFinish(g.q());
        const int base_offset = offset;
        fprintf(stderr, "recorded decode span (offset=%d, mimi=%d), replaying %d frames\n",
                rec_offset, (int)rec_mimi, n_frames);
        std::vector<cl_mem> wave_bufs; wave_bufs.reserve(n_frames);
        KernelProfiler::reset_host_profile();
        auto _t_decode = _clk::now();
        for (int step = 0; step < n_frames; ++step) {
            clEnqueueCopyBuffer(g.q(), noise_big, noise_fixed, (size_t)step*32*es, 0, 32*es, 0, nullptr, nullptr);
            if (rec_mimi) g.write_replay_offsets(base_offset + step, step * 16);
            else          g.write_offset(base_offset + step);            // backbone offset only
            g.replay(rec, 0, nullptr);                                    // recorded span → next (this frame)
            cl_mem w;
            if (rec_mimi) {                                               // wave_rec reused each replay → copy out
                w = g.alloc(1920);
                clEnqueueCopyBuffer(g.q(), wave_rec, w, 0, 0, 1920*es, 0, nullptr, nullptr);
            } else {
                w = mimi_decode_frame(g, next, S);                       // mimi LIVE (fast indirect CLBlast)
            }
            wave_bufs.push_back(w);
            clEnqueueCopyBuffer(g.q(), next, cur, 0, 0, 32*es, 0, nullptr, nullptr);  // next→cur
        }
        clFinish(g.q());
        double _decode_ms = std::chrono::duration<double, std::milli>(_clk::now() - _t_decode).count();
        std::vector<float> wave; wave.reserve((size_t)n_frames * 1920);
        for (int step = 0; step < n_frames; ++step) {
            std::vector<float> s = g.download(wave_bufs[step], 1920);
            wave.insert(wave.end(), s.begin(), s.end());
            g.release(wave_bufs[step]);
        }
        S.free_();
        for (int i = 0; i < 6; ++i) { g.release(kc[i]); g.release(vc[i]); }
        double _audio_s = (double)n_frames * 1920.0 / 24000.0;
        double _compute_s = (_prefill_ms + _decode_ms) / 1000.0;
        fprintf(stderr, "PHASE decode_total_ms: %.1f  (%d frames, RECORD/REPLAY)\n", _decode_ms, n_frames);
        fprintf(stderr, "PHASE steady_per_frame_ms: %.1f  (decode/n_frames)\n", _decode_ms / n_frames);
        fprintf(stderr, "PHASE audio_s: %.3f  compute_s(prefill+decode): %.3f  RTF: %.4f  (%.1fx slower than real-time)\n",
                _audio_s, _compute_s, _audio_s / _compute_s, _compute_s / _audio_s);
        KernelProfiler::dump_host_profile();
        return wave;
    }

    // 3) flow-matching generation loop.
    MimiState S; S.init(g);                 // persistent mimi streaming state
    std::mt19937 rng(1234);                 // fixed seed → reproducible
    std::normal_distribution<float> gauss(0.0f, 1.0f);
    std::vector<float> bosv = g.weights().get_host_vec("flow_lm.bos_emb");
    cl_mem cur = g.upload(bosv);   // step-0 latent = bos_emb [32]
    // Keep eos + waveform ON-DEVICE and download AFTER the loop. The eos logit is
    // diagnostic only (the loop runs a fixed n_frames and never breaks on it) and
    // the waveform is a per-frame sink — so nothing in the loop needs a GPU→host
    // sync. The old per-frame blocking downloads forced the host to wait on the GPU
    // every frame, leaving the GPU idle ~62% of the wall. Deferring them lets the
    // host enqueue all frames ahead so the GPU runs back-to-back.
    std::vector<cl_mem> eos_bufs, wave_bufs;
    eos_bufs.reserve(n_frames); wave_bufs.reserve(n_frames);
    double _hb_ms = 0, _hf_ms = 0, _hm_ms = 0;   // host-enqueue time per section (frames 2..N)
    KernelProfiler::reset_host_profile();         // measure DECODE-only host cost
    auto _t_decode = _clk::now();
    for (int step = 0; step < n_frames; ++step) {
        auto _h0 = _clk::now();
        cl_mem inp = g.linear(cur, 1, 1024, 32, "flow_lm.input_linear.weight");     // [1024]
        cl_mem h   = backbone_run(g, inp, 1, offset, kc, vc);                        // [1024]
        cl_mem c   = g.layernorm(h, 1, 1024, 1e-5f, "flow_lm.out_norm.weight", "flow_lm.out_norm.bias");
        cl_mem eosb = g.linear(c, 1, 1, 1024, "flow_lm.out_eos.weight", "flow_lm.out_eos.bias");
        auto _h1 = _clk::now();   // backbone host-enqueue done
        std::vector<float> nz(32);
        if (noise_seq && (size_t)(step * 32 + 32) <= noise_seq->size())
            for (int j = 0; j < 32; ++j) nz[j] = (*noise_seq)[step * 32 + j];        // dumped reference noise
        else
            for (int j = 0; j < 32; ++j) nz[j] = noise_std * gauss(rng);             // N(0, std)
        cl_mem noise = g.upload(nz);
        cl_mem flow = g.flow_net(c, 0.0f, 1.0f, noise);                              // [32]
        cl_mem next = g.add(noise, flow, 32);                                        // next = noise + flow
        auto _h2 = _clk::now();   // flow host-enqueue done
        if (c_out)      { std::vector<float> cv = g.download(c, 1024); c_out->insert(c_out->end(), cv.begin(), cv.end()); }
        if (latent_out) { std::vector<float> lv = g.download(next, 32); latent_out->insert(latent_out->end(), lv.begin(), lv.end()); }
        cl_mem w = mimi_decode_frame(g, next, S);
        auto _h3 = _clk::now();   // mimi host-enqueue done
        if (step > 0) {
            _hb_ms += std::chrono::duration<double,std::milli>(_h1 - _h0).count();
            _hf_ms += std::chrono::duration<double,std::milli>(_h2 - _h1).count();
            _hm_ms += std::chrono::duration<double,std::milli>(_h3 - _h2).count();
        }
        eos_bufs.push_back(eosb);
        wave_bufs.push_back(w);
        g.release(inp); g.release(h); g.release(c); g.release(noise); g.release(flow);
        g.release(cur); cur = next;
        offset++;   // advance KV-cache write position + RoPE position each step
        // serve mode: stop when the model signals end-of-utterance. Downloads this frame's
        // eos logit (forces a per-frame sync — variable length over raw throughput).
        if (stop_on_eos && step + 1 >= 4) {
            float eos = g.download(eosb, 1)[0];
            if (eos > eos_threshold) break;
        }
    }
    const int gen_frames = (int)wave_bufs.size();   // actual frames produced (≤ n_frames)
    double _enq_ms = std::chrono::duration<double, std::milli>(_clk::now() - _t_decode).count();  // host enqueue time
    clFinish(g.q());   // single sync after the whole decode loop
    double _decode_ms = std::chrono::duration<double, std::milli>(_clk::now() - _t_decode).count();
    fprintf(stderr, "PHASE host_enqueue_ms: %.1f  gpu_tail_ms: %.1f  (if enqueue≈decode ⇒ HOST-bound)\n",
            _enq_ms, _decode_ms - _enq_ms);
    int _sn = gen_frames > 1 ? gen_frames - 1 : 1;
    fprintf(stderr, "PHASE host_section_ms/frame: backbone=%.2f flow=%.2f mimi=%.2f\n",
            _hb_ms/_sn, _hf_ms/_sn, _hm_ms/_sn);
    KernelProfiler::dump_host_profile();   // decode-only host cost by category
    g.release(cur); S.free_();
    for (int i = 0; i < 6; ++i) { g.release(kc[i]); g.release(vc[i]); }

    // deferred downloads — one batch, no per-frame sync
    std::vector<float> wave; wave.reserve((size_t)gen_frames * 1920);
    for (int step = 0; step < gen_frames; ++step) {
        std::vector<float> s = g.download(wave_bufs[step], 1920);
        wave.insert(wave.end(), s.begin(), s.end());
        if (!stop_on_eos) {   // eos already downloaded in-loop when stopping
            float eos = g.download(eos_bufs[step], 1)[0];
            fprintf(stderr, "frame %d/%d  eos_logit=%.3f  samples=%zu\n", step + 1, gen_frames, eos, wave.size());
        }
        g.release(wave_bufs[step]); g.release(eos_bufs[step]);
    }

    double _audio_s   = (double)gen_frames * 1920.0 / 24000.0;        // 0.08 s/frame @ 24kHz
    double _compute_s = (_prefill_ms + _decode_ms) / 1000.0;          // prefill + decode (excludes load/build)
    fprintf(stderr, "PHASE decode_total_ms: %.1f  (%d frames)\n", _decode_ms, gen_frames);
    fprintf(stderr, "PHASE steady_per_frame_ms: %.1f  (decode/n_frames)\n", _decode_ms / (gen_frames>0?gen_frames:1));
    fprintf(stderr, "PHASE audio_s: %.3f  compute_s(prefill+decode): %.3f  RTF: %.4f  (%.1fx slower than real-time)\n",
            _audio_s, _compute_s, _audio_s / _compute_s, _compute_s / _audio_s);
    // App-parseable perf (Edgi ProcessEngine: PerfAccumulator `*_per_sec:` lines + `rtf=` regex →
    // the same Perf telemetry kokoro/mms/musicgen surface). Audio "tokens" = 80 ms Mimi frames.
    double _decode_s = _decode_ms / 1000.0, _prefill_s = _prefill_ms / 1000.0;
    double _rtf = _audio_s > 0 ? _compute_s / _audio_s : 0.0;   // synth/audio (>1 = slower than real-time)
    fprintf(stderr, "BENCHMARK prefill_tokens_per_sec: %.4f\n", _prefill_s > 0 ? (double)text_ids.size() / _prefill_s : 0.0);
    fprintf(stderr, "BENCHMARK decode_tokens_per_sec: %.4f\n",  _decode_s  > 0 ? (double)gen_frames / _decode_s : 0.0);
    fprintf(stderr, "BENCHMARK time_to_first_token_sec: %.4f\n", _prefill_s);   // prefill = time to first audio frame
    fprintf(stderr, "POCKET done  audio=%.2f s  rtf=%.3f  synth=%.2f s  (%d frames, %d text tokens)\n",
            _audio_s, _rtf, _compute_s, gen_frames, (int)text_ids.size());
    return wave;
}
