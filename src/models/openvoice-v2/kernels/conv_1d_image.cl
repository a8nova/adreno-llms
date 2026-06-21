// OpenVoice V2 ToneColorConverter — OpenCL kernels (split per functional unit).
// Built by concatenation (Engine reads _preamble.cl FIRST). DO NOT define macros here.

// ── conv1d_img — read input via TEXTURE engine / L1 cache (Adreno, guide §6.2/7.1.5) ─
// Input is a read-only image2d [width=Tin, height=Cin], CL_HALF_FLOAT. Redundant input
// re-reads (across out-channels / overlapping taps) hit the dedicated texture L1 cache.
// CLK_ADDRESS_CLAMP returns 0 outside → free conv zero-padding (no bounds branch).
// Register-tiled NC channels × TILE_T times; fp32 accumulate.
__kernel void conv1d_img(
    __read_only image2d_t in_img, __global const half* w, __global const half* bias,
    __global half* out,
    const int Cin, const int Tin, const int Cout, const int Tout,
    const int K, const int stride, const int dilation, const int pad, const int has_bias)
{
    const sampler_t smp = CLK_NORMALIZED_COORDS_FALSE | CLK_ADDRESS_CLAMP | CLK_FILTER_NEAREST;
    int co0 = get_global_id(0)*NC;
    int to0 = get_global_id(1)*TILE_T;
    if (co0 >= Cout || to0 >= Tout) return;
    float acc[NC][TILE_T];
    #pragma unroll
    for(int n=0;n<NC;++n){ float b=(has_bias&&co0+n<Cout)?WLOAD(co0+n,bias):0.0f; for(int t=0;t<TILE_T;++t) acc[n][t]=b; }
    for(int ci=0; ci<Cin; ++ci){
        for(int k=0;k<K;++k){
            int base = k*dilation - pad;
            float iv[TILE_T];
            #pragma unroll
            for(int t=0;t<TILE_T;++t) iv[t]=read_imagef(in_img, smp, (int2)((to0+t)*stride+base, ci)).x;
            #pragma unroll
            for(int n=0;n<NC;++n){ float wv=WLOAD(k, w+((size_t)(co0+n)*Cin+ci)*K);
                for(int t=0;t<TILE_T;++t) acc[n][t]+=iv[t]*wv; }
        }
    }
    #pragma unroll
    for(int n=0;n<NC;++n) if(co0+n<Cout){
        #pragma unroll
        for(int t=0;t<TILE_T;++t){ int to=to0+t; if(to<Tout) vstore_half(acc[n][t], (size_t)(co0+n)*Tout+to, out); }
    }
}
