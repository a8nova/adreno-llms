// enc_q — PosteriorEncoder: pre (Conv1d 513→192) → WN (16-layer gated WaveNet)
// → proj (Conv1d 192→384). Content encoder. zero_g=True ⇒ g=ZEROS.
// (z = m + exp(logs)*eps*tau reparam is stochastic; we verify the deterministic
//  WN output and proj stats against the reference.)
#include "engine.h"

int run_enc_q_wn() {
    OpenCLContext cl; Weights W; if(boot(cl,W)) return 1; Engine E(cl, W);
    // input = reference enc_q_pre_output.bin (fp32) [H=192, T]
    FILE* f=fopen("reference/layers/enc_q_pre_output.bin","rb");
    if(!f){ printf("no enc_q_pre_output.bin\n"); return 1; }
    fseek(f,0,SEEK_END); long sz=ftell(f); fseek(f,0,SEEK_SET);
    int n=sz/4; std::vector<float> in(n); fread(in.data(),4,n,f); fclose(f);
    int H=192, T=n/H;
    printf("enc_q WN input: [%d,%d] (%d floats)\n", H, T, n);
    Buf x{E.alloc((size_t)H*T), H, T}; E.upload(x.mem, in);
    cl_mem g0=zero_g(E);                          // zero_g=True: enc_q g=0
    E.reset_gpu_timer();
    double t0=now_ms();
    Buf out = E.wn(x, g0, H, 256, 16, 5, "enc_q.enc", "enc_q_enc", true);
    Buf stats = E.conv1d(out, "enc_q.proj", 384, 1, 1, 1, 0);   // proj → m|logs
    E.finish(); double t1=now_ms();
    E.dump(out, "enc_q_enc"); E.dump(stats, "enc_q_proj");
    printf("enc_q done: out [%d,%d] stats [%d,%d] | CPU(end-to-end) %.1f ms | peak %.1f MB\n",
        out.C, out.T, stats.C, stats.T, t1-t0, E.peak_mb());
    E.report_gpu();
    return 0;
}
