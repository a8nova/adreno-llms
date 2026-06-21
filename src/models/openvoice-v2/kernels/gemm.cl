// OpenVoice V2 ToneColorConverter — OpenCL kernels (split per functional unit).
// Built by concatenation (Engine reads _preamble.cl FIRST). DO NOT define macros here.

// ── im2col — lay out conv input as a [Cin*K, Tout] matrix for GEMM (#5) ───────
// col[(ci*K+k), to] = in[ci, to*stride - pad + k*dil]  (0 outside input). Row-major.
__kernel void im2col(__global const REAL* in, __global REAL* col,
    const int Cin, const int Tin, const int Tout,
    const int K, const int stride, const int dilation, const int pad)
{
    int row = get_global_id(0);   // 0..Cin*K-1
    int to  = get_global_id(1);   // 0..Tout-1
    if (row >= Cin*K || to >= Tout) return;
    int ci = row / K, k = row % K;
    int ti = to*stride - pad + k*dilation;
    float v = (ti>=0 && ti<Tin) ? (float)LOAD((size_t)ci*Tin + ti, in) : 0.0f;
    STORE(v, (size_t)row*Tout + to, col);
}

// ── gemm_tiled — 2-level-tiled implicit-GEMM microkernel (register × local) ───
// C[M,N] = A[M,Kd] · B[Kd,N], row-major. A=weights[Cout,Cin*K], B=im2col[Cin*K,Tout].
// Workgroup tile BM×BN; thread micro-tile MR×NR (registers); reduction streamed in
// BK chunks via __local so each loaded element is reused MR×NR times → high ALU AI.
// local size {BM/MR, BN/NR}=(8,8)=64 threads.
#define BM 32
#define BN 64
#define BK 16
#define MR 4
#define NR 8
__kernel void gemm_tiled(
    __global const half* A, __global const half* B, __global half* C,
    const int M, const int N, const int Kd)
{
    __local half As[BM*BK];
    __local half Bs[BK*BN];
    int lm=get_local_id(0), ln=get_local_id(1);     // 0..7, 0..7
    int gm=get_group_id(0)*BM, gn=get_group_id(1)*BN;
    int tid=lm*(BN/NR)+ln, nt=(BM/MR)*(BN/NR);       // 64
    float8 acc[MR];
    #pragma unroll
    for(int i=0;i<MR;++i) acc[i]=(float8)(0.0f);
    for(int k0=0;k0<Kd;k0+=BK){
        for(int idx=tid; idx<BM*BK; idx+=nt){ int r=idx/BK,c=idx%BK; int gr=gm+r,gc=k0+c;
            As[idx]=(gr<M&&gc<Kd)?A[(size_t)gr*Kd+gc]:(half)0; }
        for(int idx=tid; idx<BK*BN; idx+=nt){ int r=idx/BN,c=idx%BN; int gr=k0+r,gc=gn+c;
            Bs[idx]=(gr<Kd&&gc<N)?B[(size_t)gr*N+gc]:(half)0; }
        barrier(CLK_LOCAL_MEM_FENCE);
        #pragma unroll
        for(int kk=0;kk<BK;++kk){
            float8 rb=convert_float8(vload8(0, Bs+kk*BN+ln*NR));
            #pragma unroll
            for(int i=0;i<MR;++i) acc[i]+=(float8)((float)As[(lm*MR+i)*BK+kk])*rb;
        }
        barrier(CLK_LOCAL_MEM_FENCE);
    }
    #pragma unroll
    for(int i=0;i<MR;++i){ int gr=gm+lm*MR+i; if(gr<M && gn+ln*NR+NR<=N)
        vstore8(convert_half8(acc[i]), 0, C+(size_t)gr*N+gn+ln*NR); }
}

// ── add_bias — out[co,to] += bias[co]  (after GEMM, #5) ───────────────────────
__kernel void add_bias(__global REAL* out, __global const half* bias, const int Cout, const int Tout){
    int co=get_global_id(0), to=get_global_id(1);
    if(co>=Cout||to>=Tout) return;
    size_t i=(size_t)co*Tout+to; STORE(LOAD(i,out)+vload_half(co,bias), i, out);
}
