// OpenVoice V2 ToneColorConverter — OpenCL kernels (split per functional unit).
// Built by concatenation (Engine reads _preamble.cl FIRST). DO NOT define macros here.

// ── x[c,t] += cond[c,0]  (broadcast time-invariant cond over T) ─────────────
__kernel void add_cond_broadcast(__global REAL* x, __global const REAL* cond, const int C, const int T) {
    int c = get_global_id(0); int t = get_global_id(1);
    if (c >= C || t >= T) return;
    size_t i = (size_t)c*T + t;
    STORE(LOAD(i, x) + LOAD((size_t)c, cond), i, x);
}

// ── flip channel order: out[c,t] = in[C-1-c, t] ────────────────────────────
__kernel void flip_channels(__global const REAL* in, __global REAL* out, const int C, const int T) {
    int c = get_global_id(0); int t = get_global_id(1);
    if (c >= C || t >= T) return;
    STORE(LOAD((size_t)(C-1-c)*T + t, in), (size_t)c*T + t, out);
}

// ── elementwise: out = a + b (same shape N) ─────────────────────────────────
__kernel void add(__global const REAL* a, __global const REAL* b, __global REAL* out, const int N) {
    int i = get_global_id(0); if (i >= N) return;
    STORE(LOAD(i, a) + LOAD(i, b), i, out);
}

// ── leaky_relu in place (slope 0.1) ─────────────────────────────────────────
__kernel void leaky_relu(__global REAL* x, const int N, const float slope) {
    int i = get_global_id(0); if (i >= N) return;
    float v = LOAD(i, x);
    STORE(v >= 0.0f ? v : slope * v, i, x);
}

// ── tanh in place ───────────────────────────────────────────────────────────
__kernel void tanh_inplace(__global REAL* x, const int N) {
    int i = get_global_id(0); if (i >= N) return;
    STORE(tanh((float)LOAD(i, x)), i, x);
}

// ── scale in place: x *= s ──────────────────────────────────────────────────
__kernel void scale(__global REAL* x, const int N, const float s) {
    int i = get_global_id(0); if (i >= N) return;
    STORE(LOAD(i, x) * s, i, x);
}
