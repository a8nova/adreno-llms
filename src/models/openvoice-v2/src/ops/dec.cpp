// dec — HiFiGAN Generator (vocoder): latent z_hat [192,T] → waveform [1, T*256].
// conv_pre + cond → 4× [lrelu, ConvTranspose upsample, MRF (3 ResBlocks averaged)]
// → lrelu(0.01) → conv_post → tanh. zero_g=True ⇒ g=ZEROS.
#include "engine.h"

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
int run_clone() {
    OpenCLContext cl; Weights W; if(boot(cl,W)) return 1; Engine E(cl,W);
    // enc_q content-encoder input (reference pre-output [192,T])
    FILE* f=fopen("reference/layers/enc_q_pre_output.bin","rb");
    if(!f){ printf("no enc_q_pre_output.bin\n"); return 1; }
    fseek(f,0,SEEK_END); long sz=ftell(f); fseek(f,0,SEEK_SET);
    int n=sz/4; std::vector<float> ein(n); fread(ein.data(),4,n,f); fclose(f);
    int H=192, Tenc=n/H;
    Buf ex{E.alloc((size_t)H*Tenc),H,Tenc}; E.upload(ex.mem,ein);
    Buf z = load_ref(E, "reference/layers/enc_q_output.bin", 192); if(!z.mem) return 1;
    cl_mem g_src=load_g(E,"assets/g_src.bin"), g_tgt=load_g(E,"assets/g_tgt.bin");
    cl_mem g0=zero_g(E);
    if(getenv("NNOPT_WARMUP")) E.warmup();
    // ---- timed: single process, no dumps ----
    double t0=now_ms();
    Buf enc_out = E.wn(ex, g0, H, 256, 16, 5, "enc_q.enc", "", false);
    Buf stats   = E.conv1d(enc_out, "enc_q.proj", 384, 1, 1, 1, 0);   // m|logs (enc_q cost)
    E.finish(); double t1=now_ms(); E.rel(enc_out.mem); E.rel(stats.mem);
    Buf z_p   = E.flow(z,   g_src, 256, false);
    Buf z_hat = E.flow(z_p, g_tgt, 256, true);
    E.finish(); double t2=now_ms(); E.rel(z_p.mem);
    Buf o     = E.dec(z_hat, g0, 256, false);                        // NO stage dumps
    E.finish(); double t3=now_ms();
    double audio_s = o.T/22050.0;
    printf("CLONE (1 process, no dumps): waveform [%d,%d] = %.2fs audio | peak %.1f MB\n",
        o.C, o.T, audio_s, E.peak_mb());
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
