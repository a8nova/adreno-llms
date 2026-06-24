// OpenVoice V2 ToneColorConverter — OpenCL kernels (split per functional unit).
// Built by concatenation (Engine reads _preamble.cl FIRST). DO NOT define macros here.

// ── hardware roofline probes ─────────────────────────────────────────────────
// Memory bandwidth: read 2 fp16 + write 1 fp16 per element (3 transactions).
__kernel void bw_copy(__global const half* a, __global const half* b, __global half* out, const int N){
    int i=get_global_id(0); if(i>=N) return;
    vstore_half(vload_half(i,a)+vload_half(i,b), i, out);
}
// Compute peak: register-resident FMA chains (no memory traffic in the loop).
// 16 independent chains → enough ILP to saturate the FMA units (latency hiding).
__kernel void compute_peak(__global float* out, const int iters){
    int i=get_global_id(0); float b=0.9999f, c=1e-3f;
    float8 x = (float8)((float)i*1e-6f) + (float8)(0,1,2,3,4,5,6,7);
    float8 y = x + (float8)(8.0f);
    for(int k=0;k<iters;++k){
        x=x*b+c; y=y*b+c; x=x*b+c; y=y*b+c;
        x=x*b+c; y=y*b+c; x=x*b+c; y=y*b+c;
    }
    if(i<0){ float8 s=x+y; out[i]=s.s0+s.s1+s.s2+s.s3+s.s4+s.s5+s.s6+s.s7; }
}
