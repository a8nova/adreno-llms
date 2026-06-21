// OpenVoice V2 ToneColorConverter — OpenCL kernels (split per functional unit).
// Built by concatenation (Engine reads _preamble.cl FIRST). DO NOT define macros here.

// ── fused_add_tanh_sigmoid_multiply (WaveNet gate) ─────────────────────────
// x_in, g_l: [2*H, T].  out: [H, T] = tanh(x_in[:H]+g_l[:H]) * sigmoid(x_in[H:]+g_l[H:])
__kernel void fused_tanh_sigmoid(
    __global const REAL* x_in, __global const REAL* g_l, __global REAL* out,
    const int H, const int T)
{
    int c = get_global_id(0);  // 0..H-1
    int t = get_global_id(1);
    if (c >= H || t >= T) return;
    float a = LOAD((size_t)c * T + t, x_in)       + LOAD((size_t)c * T + t, g_l);
    float b = LOAD((size_t)(c + H) * T + t, x_in) + LOAD((size_t)(c + H) * T + t, g_l);
    float ta = tanh(a);
    float si = 1.0f / (1.0f + exp(-b));
    STORE(ta * si, (size_t)c * T + t, out);
}

// ── fused gate with cond BROADCAST over time ───────────────────────────────
// x_in: [2H, T].  gcond: [2H*n_layers, 1] (time-invariant; k=1 conv of g).
// goff = i*2H channel offset for layer i.
// out[c,t] = tanh(x_in[c,t]+gcond[goff+c]) * sigmoid(x_in[H+c,t]+gcond[goff+H+c])
__kernel void fused_tanh_sigmoid_bcast(
    __global const REAL* x_in, __global const REAL* gcond, __global REAL* out,
    const int H, const int T, const int goff)
{
    int c = get_global_id(0);  // 0..H-1
    int t = get_global_id(1);
    if (c >= H || t >= T) return;
    float a = LOAD((size_t)c*T + t, x_in)       + LOAD((size_t)(goff + c),     gcond);
    float b = LOAD((size_t)(c+H)*T + t, x_in)   + LOAD((size_t)(goff + H + c), gcond);
    STORE(tanh(a) * (1.0f/(1.0f+exp(-b))), (size_t)c*T + t, out);
}

// ── dst[dst_choff + c, t] += src[src_choff + c, t]   (c in 0..H-1) ──────────
__kernel void add_channel_slice(
    __global REAL* dst, __global const REAL* src,
    const int dst_choff, const int src_choff, const int H, const int T)
{
    int c = get_global_id(0);
    int t = get_global_id(1);
    if (c >= H || t >= T) return;
    size_t di = (size_t)(dst_choff + c)*T + t;
    size_t si = (size_t)(src_choff + c)*T + t;
    STORE(LOAD(di, dst) + LOAD(si, src), di, dst);
}
