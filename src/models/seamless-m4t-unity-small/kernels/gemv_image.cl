// Image2d (L1 texture-cache) weight GEMV for the M=1 decode hot path. LOSSLESS:
// the weight is the SAME fp16 data, viewed as an image2d_t so reads go through
// Adreno's separate texture L1 cache (~1.71× the buffer-cache bandwidth on the
// Adreno 620, per the adreno-llms reference ports). Weight image: CL_RGBA /
// CL_HALF_FLOAT, width = K/4 pixels, height = N rows; read_imagef auto-converts
// fp16→float4. K must be a multiple of 4 (768/1536/3072/4096 all are).
//
// MUST be built in its OWN cl_program — mixing image + buffer kernels in one
// program makes the Adreno compiler's global register allocator spill the image
// kernels ~10× (documented across lfm2/qwen/smollm2).
#pragma OPENCL EXTENSION cl_khr_fp16 : enable

__constant sampler_t SMP = CLK_NORMALIZED_COORDS_FALSE | CLK_ADDRESS_CLAMP | CLK_FILTER_NEAREST;

// One work-group (64 threads = one wave) per output row.
__kernel __attribute__((reqd_work_group_size(64, 1, 1)))
void gemv_img(__read_only image2d_t W, __global const half* x, __global const half* bias,
              __global half* out, const int M, const int K, const int N, const int has_bias) {
    __local float ls[64];
    const int row = get_group_id(0), tid = get_local_id(0);
    if (row >= N) return;
    const int np = K >> 2;   // pixels per row
    float acc = 0.0f;
    for (int p = tid; p < np; p += 64) {
        float4 wv = read_imagef(W, SMP, (int2)(p, row));
        float4 xv = vload_half4(p, x);
        acc += dot(wv, xv);
    }
    ls[tid] = acc;
    barrier(CLK_LOCAL_MEM_FENCE);
    for (int s = 32; s > 0; s >>= 1) { if (tid < s) ls[tid] += ls[tid + s]; barrier(CLK_LOCAL_MEM_FENCE); }
    if (tid == 0) { float v = ls[0]; if (has_bias) v += vload_half(row, bias); vstore_half(v, row, out); }
}

// Image-weight tiled GEMM for the encoder (M>=32). Weight strip read from the
// texture L1 cache (each weight row is re-read ~M/32 times across M-tiles, so the
// cache helps); activations stay in buffers. 32×32 output block, BK=16, 4×4 scalar
// micro-tile, WG=8×8=64. Matches linear_gemm_tiled's math (lossless).
#define IG_BM 32
#define IG_BN 32
#define IG_BK 16
__kernel __attribute__((reqd_work_group_size(8, 8, 1)))
void gemm_tiled_img(__global const half* x, __read_only image2d_t W,
                    __global const half* bias, __global half* out,
                    const int M, const int K, const int N, const int has_bias) {
    __local float As[IG_BM][IG_BK];
    __local float Bs[IG_BK][IG_BN];
    const int tx = get_local_id(0), ty = get_local_id(1);
    const int bn0 = get_group_id(0) * IG_BN, bm0 = get_group_id(1) * IG_BM;
    const int tid = ty * 8 + tx;
    float acc[4][4];
    #pragma unroll
    for (int i = 0; i < 4; ++i) for (int j = 0; j < 4; ++j) acc[i][j] = 0.0f;
    for (int k0 = 0; k0 < K; k0 += IG_BK) {
        // As: 32×16 = 512 activation elems, 8 per thread (scalar, coalesced).
        #pragma unroll
        for (int i = 0; i < 8; ++i) {
            int idx = tid + i * 64, r = idx >> 4, kk = idx & 15, gm = bm0 + r;
            As[r][kk] = (gm < M) ? vload_half(gm * K + k0 + kk, x) : 0.0f;
        }
        // Bs: 16×32 from image. 32 cols × 4 pixels = 128 float4 reads, 2 per thread.
        #pragma unroll
        for (int i = 0; i < 2; ++i) {
            int idx = tid + i * 64, c = idx >> 2, pp = idx & 3, gn = bn0 + c;
            float4 wv = (gn < N) ? read_imagef(W, SMP, (int2)((k0 >> 2) + pp, gn)) : (float4)(0.0f);
            Bs[pp * 4 + 0][c] = wv.x; Bs[pp * 4 + 1][c] = wv.y;
            Bs[pp * 4 + 2][c] = wv.z; Bs[pp * 4 + 3][c] = wv.w;
        }
        barrier(CLK_LOCAL_MEM_FENCE);
        #pragma unroll
        for (int kk = 0; kk < IG_BK; ++kk) {
            float av[4], bv[4];
            #pragma unroll
            for (int i = 0; i < 4; ++i) av[i] = As[ty * 4 + i][kk];
            #pragma unroll
            for (int j = 0; j < 4; ++j) bv[j] = Bs[kk][tx * 4 + j];
            #pragma unroll
            for (int i = 0; i < 4; ++i) for (int j = 0; j < 4; ++j) acc[i][j] += av[i] * bv[j];
        }
        barrier(CLK_LOCAL_MEM_FENCE);
    }
    #pragma unroll
    for (int i = 0; i < 4; ++i) {
        int m = bm0 + ty * 4 + i;
        if (m >= M) continue;
        #pragma unroll
        for (int j = 0; j < 4; ++j) {
            int o = bn0 + tx * 4 + j;
            if (o < N) { float v = acc[i][j]; if (has_bias) v += vload_half(o, bias); vstore_half(v, m * N + o, out); }
        }
    }
}

// Register GEMM with IMAGE weights (encoder M>=32). The key combo for the
// latency-bound encoder GEMM: a SMALL 4×4 register tile keeps register pressure
// low (→ many resident waves → hides memory latency, guide §3.2.3), while the
// weight re-reads (across M-tiles) are served by the texture L1 cache instead of
// burning registers/local for reuse. x stays in a buffer (half4), W is image2d
// (CL_RGBA/half, width K/4, height N), read_imagef → float4. global=(ceil(N/4), ceil(M/4)).
__kernel void gemm_reg_img4(__global const half* x, __read_only image2d_t W,
                            __global const half* bias, __global half* out,
                            const int M, const int K, const int N, const int has_bias) {
    int row0 = get_global_id(1) * 4;
    int o0 = get_global_id(0) * 4;
    float acc[4][4];
    #pragma unroll
    for (int i = 0; i < 4; ++i) for (int j = 0; j < 4; ++j) acc[i][j] = 0.0f;
    int np = K >> 2;  // K/4 pixels per row
    for (int p = 0; p < np; ++p) {
        float4 xv0 = (row0 + 0 < M) ? vload_half4(p, x + (row0 + 0) * K) : (float4)(0.0f);
        float4 xv1 = (row0 + 1 < M) ? vload_half4(p, x + (row0 + 1) * K) : (float4)(0.0f);
        float4 xv2 = (row0 + 2 < M) ? vload_half4(p, x + (row0 + 2) * K) : (float4)(0.0f);
        float4 xv3 = (row0 + 3 < M) ? vload_half4(p, x + (row0 + 3) * K) : (float4)(0.0f);
        #pragma unroll
        for (int j = 0; j < 4; ++j) {
            int o = o0 + j;
            if (o < N) {
                float4 wv = read_imagef(W, SMP, (int2)(p, o));
                acc[0][j] += dot(xv0, wv); acc[1][j] += dot(xv1, wv);
                acc[2][j] += dot(xv2, wv); acc[3][j] += dot(xv3, wv);
            }
        }
    }
    #pragma unroll
    for (int i = 0; i < 4; ++i) { int m = row0 + i; if (m < M)
        #pragma unroll
        for (int j = 0; j < 4; ++j) { int o = o0 + j; if (o < N) {
            float v = acc[i][j]; if (has_bias) v += vload_half(o, bias);
            vstore_half(v, m * N + o, out); } } }
}

// 4 output rows per work-group: x pixel loaded once, 4 image rows, 4 ILP chains.
__kernel __attribute__((reqd_work_group_size(64, 1, 1)))
void gemv_img4(__read_only image2d_t W, __global const half* x, __global const half* bias,
               __global half* out, const int M, const int K, const int N, const int has_bias) {
    __local float ls[64 * 4];
    const int n0 = get_group_id(0) * 4, tid = get_local_id(0);
    if (n0 >= N) return;
    const int np = K >> 2;
    float a0 = 0.0f, a1 = 0.0f, a2 = 0.0f, a3 = 0.0f;
    for (int p = tid; p < np; p += 64) {
        float4 xv = vload_half4(p, x);
        a0 += dot(read_imagef(W, SMP, (int2)(p, n0)),     xv);
        a1 += dot(read_imagef(W, SMP, (int2)(p, n0 + 1)), xv);
        a2 += dot(read_imagef(W, SMP, (int2)(p, n0 + 2)), xv);
        a3 += dot(read_imagef(W, SMP, (int2)(p, n0 + 3)), xv);
    }
    ls[tid] = a0; ls[tid + 64] = a1; ls[tid + 128] = a2; ls[tid + 192] = a3;
    barrier(CLK_LOCAL_MEM_FENCE);
    for (int s = 32; s > 0; s >>= 1) {
        if (tid < s) { ls[tid] += ls[tid + s]; ls[tid + 64] += ls[tid + 64 + s];
                       ls[tid + 128] += ls[tid + 128 + s]; ls[tid + 192] += ls[tid + 192 + s]; }
        barrier(CLK_LOCAL_MEM_FENCE);
    }
    if (tid < 4 && (n0 + tid) < N) {
        float v = ls[tid * 64];
        if (has_bias) v += vload_half(n0 + tid, bias);
        vstore_half(v, n0 + tid, out);
    }
}
