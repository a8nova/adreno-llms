// dec — HiFiGAN Generator (vocoder): latent z_hat [192,T] → waveform [1, T*256].
// conv_pre + cond → 4× [lrelu, ConvTranspose upsample, MRF (3 ResBlocks averaged)]
// → lrelu(0.01) → conv_post → tanh. zero_g=True ⇒ g=ZEROS.
#include "engine.h"
#include "../ov_stft.h"     // CPU linear-spectrogram front-end (spectrogram_torch)
#include "../load_wav.h"    // 16-bit PCM WAV -> mono f32 [-1,1]
#include "../write_wav.h"   // f32 [-1,1] -> 16-bit PCM WAV
#include <random>
#include <string>
#include <cstring>

// ResBlock1: 3 (convs1,convs2) pairs, each indexed by d in dil[1,3,5].
// convs1 dilation = dil[d] (pad=dil*(K-1)/2); convs2 dilation 1 (pad=(K-1)/2).
// K is the resblock kernel size (3, 7 or 11). x[C,T]→[C,T].
Buf Engine::resblock(const Buf& x_in, const std::string& wkey, int K){
    const int dil[3]={1,3,5};
    Buf x=clone(x_in);
    for(int d=0;d<3;++d){
        Buf xt=clone(x); lrelu(xt,0.1f);
        Buf c1=conv1d(xt, wkey+".convs1."+std::to_string(d), x.C, K, 1, dil[d], dil[d]*(K-1)/2);
        rel(xt.mem);
        lrelu(c1,0.1f);
        Buf c2=conv1d(c1, wkey+".convs2."+std::to_string(d), x.C, K, 1, 1, (K-1)/2);
        rel(c1.mem);
        Buf s=addbuf(c2, x); rel(c2.mem); rel(x.mem); x=s;
    }
    return x;
}

// Generator / HiFiGAN. z_hat [192,T], g [256,1] → waveform [1, T*256].
Buf Engine::dec(const Buf& z, cl_mem g, int gin, bool dump_stages){
    Buf x=conv1d(z, "dec.conv_pre", 512, 7, 1, 1, 3);
    if(dump_stages) dump(x, "dec_conv_pre");
    Buf gbuf{g, gin, 1};
    Buf cond=conv1d(gbuf, "dec.cond", 512, 1, 1, 1, 0);   // [512,1] time-invariant
    x = add_cond_bcast(x, cond);                          // x[c,t] += cond[c,0]
    rel(cond.mem);
    const int up_k[4]={16,16,4,4}, up_s[4]={8,8,2,2}, up_pad[4]={4,4,1,1};
    for(int i=0;i<4;++i){
        lrelu(x,0.1f);
        Buf u=convT(x, "dec.ups."+std::to_string(i), 256>>i, up_k[i], up_s[i], up_pad[i]);
        rel(x.mem); x=u;
        if(dump_stages) dump(x, "dec_ups_"+std::to_string(i));
        const int rk[3]={3,7,11};
        Buf acc{0,0,0};
        for(int j=0;j<3;++j){
            Buf rb=resblock(x, "dec.resblocks."+std::to_string(i*3+j), rk[j]);
            if(j==0){ acc=rb; } else { Buf s=addbuf(acc,rb); rel(acc.mem); rel(rb.mem); acc=s; }
        }
        scale_(acc, 1.0f/3.0f);
        rel(x.mem); x=acc;
    }
    lrelu(x, 0.01f);                                 // final lrelu uses DEFAULT slope 0.01
    if(dump_stages) dump(x, "dec_pre_post");
    Buf o=conv1d(x, "dec.conv_post", 1, 7, 1, 1, 3); // bias=False
    rel(x.mem);
    if(dump_stages) dump(o, "dec_conv_post");
    tanh_(o);
    return o;
}

// dec: feed reference z_hat (flow_output) → dec(g=0) → dump dec.
int run_dec() {
    OpenCLContext cl; Weights W; if(boot(cl,W)) return 1; Engine E(cl,W);
    Buf z_hat = load_ref(E, "reference/layers/flow_output.bin", 192); if(!z_hat.mem) return 1;
    cl_mem g0=zero_g(E);                          // zero_g=True: dec g=0
    E.reset_gpu_timer();
    double t0=now_ms();
    Buf o = E.dec(z_hat, g0, 256, true);
    E.finish(); double t1=now_ms();
    E.dump(o, "dec");
    printf("dec done: waveform [%d,%d] | CPU(end-to-end) %.1f ms | peak %.1f MB\n", o.C, o.T, t1-t0, E.peak_mb());
    E.report_gpu();
    return 0;
}

// FAST localization: dec on the first N frames of z_hat (seconds, not minutes).
int run_dec_fast() {
    OpenCLContext cl; Weights W; if(boot(cl,W)) return 1; Engine E(cl,W);
    Buf full = load_ref(E, "reference/layers/flow_output.bin", 192); if(!full.mem) return 1;
    int N = 120; const char* e=getenv("NNOPT_DEC_FRAMES"); if(e) N=atoi(e);
    if(N>full.T) N=full.T;
    Buf z = E.slice_time(full, N);
    cl_mem g0=zero_g(E);
    if(getenv("NNOPT_WARMUP")) E.warmup();        // #17 ramp GPU clock before timing
    E.reset_gpu_timer(); double t0=now_ms();
    Buf o = E.dec(z, g0, 256, false);            // no stage dumps — pure perf
    E.finish(); double t1=now_ms();
    double audio_s = o.T/22050.0;
    printf("dec_fast: frames=%d waveform [%d,%d] = %.2fs audio | CPU(e2e) %.1f ms | GPU %.1f ms | %.1fx realtime | peak %.1f MB\n",
        N, o.C, o.T, audio_s, t1-t0, E.gpu_ms(), (t1-t0)/1000.0/audio_s, E.peak_mb());
    E.report_gpu();
    return 0;
}

// CLONE — the real end-to-end inference path: enc_q → flow → dec in ONE process,
// sharing the Engine (single OpenCL boot + kernel build + weight cache + arena), with
// NO stage dumps (dump=false). This is what ships; run_all/run_dec keep dumps for cos.
// flow input is the reference posterior sample (z) so flow/dec stay bit-exact verifiable;
// enc_q runs its real WaveNet forward for timing. zero_g=True (enc_q & dec g=0).
//
// Two input paths, selected by whether --src is given:
//   • --src ref.wav  → the REAL tool: STFT → enc_q.pre → WN → proj → reparam(z),
//                      then flow(g_src)→flow(g_tgt,rev)→dec → --out re-voiced WAV.
//   • no --src       → the legacy fused BENCHMARK: seeds enc_q input + z from the
//                      gitignored reference/layers/*.bin fixtures (bit-exact, no WAV).
// --g-src/--g-tgt override the baked assets/g_{src,tgt}.bin tone-color embeddings.
// tau (posterior noise scale) defaults to OpenVoice's convert() value 0.3; set
// NNOPT_TAU=0 for a deterministic (z=m) run when cosine-checking against upstream.
int run_clone(int argc, char** argv) {
    std::string src, out;
    const char* g_src_path = "assets/g_src.bin";
    const char* g_tgt_path = "assets/g_tgt.bin";
    for (int i = 2; i < argc; i++) {
        if (!std::strcmp(argv[i], "--src")   && i+1 < argc) src = argv[++i];
        else if (!std::strcmp(argv[i], "--out")   && i+1 < argc) out = argv[++i];
        else if (!std::strcmp(argv[i], "--g-src") && i+1 < argc) g_src_path = argv[++i];
        else if (!std::strcmp(argv[i], "--g-tgt") && i+1 < argc) g_tgt_path = argv[++i];
    }
    float tau = 0.3f;
    if (const char* e = getenv("NNOPT_TAU")) tau = (float)atof(e);

    OpenCLContext cl; Weights W; if(boot(cl,W)) return 1; Engine E(cl,W);
    const int H = 192;

    // ── enc_q content-encoder input: ex [192,T] ──────────────────────────────
    Buf ex; bool real_path = !src.empty();
    if (real_path) {
        // REAL front end: WAV → spectrogram_torch [513,T] → enc_q.pre (Conv1d 513→192).
        std::vector<float> audio; nnopt::WavInfo wi;
        if (!nnopt::read_wav_mono_f32(src, audio, wi, 22050)) {
            printf("clone: cannot read --src %s\n", src.c_str()); return 1;
        }
        int Tspec = 0;
        std::vector<float> spec = nnopt::ov_spectrogram(audio, &Tspec);   // [513, Tspec]
        if (Tspec <= 0) { printf("clone: --src too short (%zu samples)\n", audio.size()); return 1; }
        printf("clone: --src %s | %d samples @ %dHz → spec [513,%d]\n",
               src.c_str(), (int)audio.size(), wi.sample_rate, Tspec);
        Buf specbuf{E.alloc((size_t)513*Tspec), 513, Tspec}; E.upload(specbuf.mem, spec);
        ex = E.conv1d(specbuf, "enc_q.pre", H, 1, 1, 1, 0);               // [192,Tspec]
        E.rel(specbuf.mem);
    } else {
        // LEGACY bench: reference pre-output [192,T] from a cached fixture.
        FILE* f=fopen("reference/layers/enc_q_pre_output.bin","rb");
        if(!f){ printf("no enc_q_pre_output.bin (pass --src ref.wav for the real path)\n"); return 1; }
        fseek(f,0,SEEK_END); long sz=ftell(f); fseek(f,0,SEEK_SET);
        int n=sz/4; std::vector<float> ein(n); fread(ein.data(),4,n,f); fclose(f);
        int Tenc=n/H;
        ex = Buf{E.alloc((size_t)H*Tenc),H,Tenc}; E.upload(ex.mem,ein);
    }

    cl_mem g_src=load_g(E,g_src_path), g_tgt=load_g(E,g_tgt_path);
    cl_mem g0=zero_g(E);
    if(getenv("NNOPT_WARMUP")) E.warmup();

    // ── enc_q: WN → proj (m|logs) ────────────────────────────────────────────
    double t0=now_ms();
    Buf enc_out = E.wn(ex, g0, H, 256, 16, 5, "enc_q.enc", "", false);
    Buf stats   = E.conv1d(enc_out, "enc_q.proj", 384, 1, 1, 1, 0);   // [384,T] = m|logs
    E.finish(); double t1=now_ms(); E.rel(ex.mem); E.rel(enc_out.mem);

    // ── reparam: z = m + eps·tau·exp(logs) ───────────────────────────────────
    Buf z;
    if (real_path) {
        const int T = stats.T;
        std::vector<float> st = E.download(stats.mem, stats.n());      // [384,T] channel-major
        E.rel(stats.mem);
        std::vector<float> zf((size_t)H*T);
        std::mt19937 rng(1234); std::normal_distribution<float> nd(0.0f, 1.0f);
        for (int c = 0; c < H; c++) {
            const float* m  = &st[(size_t)c       * T];
            const float* lg = &st[(size_t)(c + H) * T];
            float* zc = &zf[(size_t)c * T];
            for (int t = 0; t < T; t++) {
                float eps = (tau > 0.0f) ? nd(rng) : 0.0f;
                zc[t] = m[t] + eps * tau * std::exp(lg[t]);
            }
        }
        z = Buf{E.alloc((size_t)H*T), H, T}; E.upload(z.mem, zf);
    } else {
        // bench: seed flow from the cached posterior sample so flow/dec stay bit-exact.
        E.rel(stats.mem);
        z = load_ref(E, "reference/layers/enc_q_output.bin", 192); if(!z.mem) return 1;
    }

    // ── flow (g_src fwd → g_tgt rev) → dec ───────────────────────────────────
    Buf z_p   = E.flow(z,   g_src, 256, false); E.rel(z.mem);
    Buf z_hat = E.flow(z_p, g_tgt, 256, true);
    E.finish(); double t2=now_ms(); E.rel(z_p.mem);
    Buf o     = E.dec(z_hat, g0, 256, false);                        // NO stage dumps
    E.finish(); double t3=now_ms(); E.rel(z_hat.mem);
    double audio_s = o.T/22050.0;

    // ── write re-voiced WAV ──────────────────────────────────────────────────
    if (!out.empty()) {
        std::vector<float> wav = E.download(o.mem, o.n());           // [1, T*256] in [-1,1]
        if (nnopt::write_wav(out, wav.data(), (int)wav.size(), 22050))
            printf("clone: wrote %s (%.2fs @ 22050Hz, tau=%.2f)\n", out.c_str(), audio_s, tau);
        else
            printf("clone: FAILED to write %s\n", out.c_str());
    }
    printf("CLONE (%s): waveform [%d,%d] = %.2fs audio | peak %.1f MB\n",
        real_path ? "real" : "bench", o.C, o.T, audio_s, E.peak_mb());
    printf("  enc_q %.1f ms | flow %.1f ms | dec %.1f ms | TOTAL %.1f ms | %.2fx realtime\n",
        t1-t0, t2-t1, t3-t2, t3-t0, (t3-t0)/1000.0/audio_s);
    return 0;
}

// full chain from reference z → waveform (flow fwd/rev with real g, dec with g=0).
int run_all() {
    OpenCLContext cl; Weights W; if(boot(cl,W)) return 1; Engine E(cl,W);
    Buf z = load_ref(E, "reference/layers/enc_q_output.bin", 192); if(!z.mem) return 1;
    cl_mem g_src=load_g(E,"assets/g_src.bin"), g_tgt=load_g(E,"assets/g_tgt.bin");
    cl_mem g0=zero_g(E);                          // zero_g=True: dec g=0 (flow keeps real g)
    E.reset_gpu_timer(); double t0=now_ms();
    Buf z_p   = E.flow(z,   g_src, 256, false);
    Buf z_hat = E.flow(z_p, g_tgt, 256, true);   E.finish(); double tflow=now_ms(); double gflow=E.gpu_ms();
    Buf o     = E.dec(z_hat, g0, 256, true);     E.finish(); double tdec=now_ms();  double gdec=E.gpu_ms()-gflow;
    E.dump(z_hat, "flow"); E.dump(o, "dec");
    auto wav=E.download(o.mem,o.n()); FILE* wf=fopen("layer_dumps/waveform_f32.bin","wb"); fwrite(wav.data(),4,wav.size(),wf); fclose(wf);
    printf("full chain: waveform [%d,%d] | peak %.1f MB\n", o.C, o.T, E.peak_mb());
    printf("  CPU(end-to-end): flow %.1f ms, dec %.1f ms, total %.1f ms\n", tflow-t0, tdec-tflow, tdec-t0);
    printf("  GPU-timer(kernel exec): flow %.1f ms, dec %.1f ms, total %.1f ms\n", gflow, gdec, E.gpu_ms());
    E.report_gpu();
    return 0;
}
