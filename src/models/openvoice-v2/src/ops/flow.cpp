// flow — ResidualCouplingBlock: 4× [ ResidualCouplingLayer(mean_only) + Flip ].
// Forward (g_src) strips source identity; reverse (g_tgt) applies target identity.
// This is the ONLY component that uses the speaker embedding (zero_g doesn't touch flow).
#include "engine.h"

// ResidualCouplingLayer (mean_only=True). x [192,T]; g [256,1].
// forward: x1 = m + x1; reverse: x1 = x1 - m; out = cat([x0, x1]).
Buf Engine::coupling(const Buf& x, cl_mem g, int gin, const std::string& wkey, bool reverse){
    int half=x.C/2;                                  // 96
    Buf x0=slice(x,0,half), x1=slice(x,half,half);
    Buf h=conv1d(x0, wkey+".pre", 2*half, 1, 1, 1, 0);            // pre: Conv1d(96->192,k=1)
    Buf enc=wn(h, g, 2*half, gin, 4, 5, wkey+".enc", "", false);  // 4-layer WN, hidden=192
    Buf m=conv1d(enc, wkey+".post", half, 1, 1, 1, 0);           // post: Conv1d(192->96,k=1); mean_only
    if(reverse){ scale_(m,-1.0f); }                              // x1 := m + x1 (fwd) | x1 - m (rev)
    Buf x1n=addbuf(x1, m);
    Buf out=concat(x0, x1n);
    rel(x0.mem); rel(x1.mem); rel(h.mem); rel(enc.mem); rel(m.mem); rel(x1n.mem);
    return out;
}

// ResidualCouplingBlock. flows = [RCL0,Flip,RCL2,Flip,RCL4,Flip,RCL6,Flip].
// forward: in order; reverse: reversed order, each reverse=True.
Buf Engine::flow(const Buf& x_in, cl_mem g, int gin, bool reverse){
    Buf x=clone(x_in);
    const int rcl_idx[4]={0,2,4,6};
    if(!reverse){
        for(int f=0; f<4; ++f){
            Buf c=coupling(x, g, gin, "flow.flows."+std::to_string(rcl_idx[f]), false);
            rel(x.mem); x=c;
            Buf fl=flip(x); rel(x.mem); x=fl;
        }
    } else {
        for(int f=3; f>=0; --f){
            Buf fl=flip(x); rel(x.mem); x=fl;          // Flip (its own inverse)
            Buf c=coupling(x, g, gin, "flow.flows."+std::to_string(rcl_idx[f]), true);
            rel(x.mem); x=c;
        }
    }
    return x;
}

// flow: feed reference z (enc_q_output) → fwd(g_src) → rev(g_tgt) → dump flow_output.
int run_flow() {
    OpenCLContext cl; Weights W; if(boot(cl,W)) return 1; Engine E(cl,W);
    Buf z = load_ref(E, "reference/layers/enc_q_output.bin", 192); if(!z.mem) return 1;
    cl_mem g_src=load_g(E,"assets/g_src.bin"), g_tgt=load_g(E,"assets/g_tgt.bin");
    E.reset_gpu_timer();
    double t0=now_ms();
    Buf z_p   = E.flow(z,   g_src, 256, false);
    Buf z_hat = E.flow(z_p, g_tgt, 256, true);
    E.finish(); double t1=now_ms();
    E.dump(z_p, "flow_zp"); E.dump(z_hat, "flow");
    printf("flow done: z_hat [%d,%d] | CPU(end-to-end) %.1f ms | peak %.1f MB\n", z_hat.C, z_hat.T, t1-t0, E.peak_mb());
    E.report_gpu();
    return 0;
}
