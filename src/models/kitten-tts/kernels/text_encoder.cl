// Reference: kitten-tts-nano-0.1 ONNX graph /text_encoder/* (StyleTTS2 acoustic
// TextEncoder). Kernels: embedding gather, causal-same conv1d, channel LayerNorm,
// LeakyReLU, bias-add, and per-gate bidirectional LSTM cell step.
// Dtype-template preamble — DO NOT EDIT. Driven by host-side -DNNOPT_USE_FP16.
#ifdef USE_FP16
  #pragma OPENCL EXTENSION cl_khr_fp16 : enable
  typedef half storage_t;
  #define LOAD(p, i)    vload_half((i), (p))
  #define STORE(p, i, v) vstore_half((v), (i), (p))
#else
  typedef float storage_t;
  #define LOAD(p, i)    ((p)[(i)])
  #define STORE(p, i, v) ((p)[(i)] = (v))
#endif

// Embedding gather, channels-first: out[c, t] = emb[ids[t], c]
// ids int32 [T]; emb [V, C]; out [C, T]
__kernel void te_embed_cf(
    __global const int*       ids,
    __global const storage_t* emb,
    __global storage_t*       out,
    const int T,
    const int C) {
    int gid = get_global_id(0);
    int total = C * T;
    if (gid >= total) return;
    int c = gid / T;
    int t = gid - c * T;
    int tok = ids[t];
    STORE(out, gid, LOAD(emb, tok * C + c));
}

// Conv1d, same padding (pad = (K-1)/2), stride 1, groups 1.
// in [Cin, T]; w [Cout, Cin, K]; bias [Cout]; out [Cout, T].
// One work-item per (cout, t).
__kernel void te_conv1d_same(
    __global const storage_t* in,
    __global const storage_t* w,
    __global const storage_t* bias,
    __global storage_t*       out,
    const int Cin,
    const int Cout,
    const int T,
    const int K) {
    int gid = get_global_id(0);
    int total = Cout * T;
    if (gid >= total) return;
    int co = gid / T;
    int t  = gid - co * T;
    int pad = (K - 1) / 2;
    float acc = LOAD(bias, co);
    for (int ci = 0; ci < Cin; ci++) {
        int wbase = (co * Cin + ci) * K;
        int ibase = ci * T;
        for (int k = 0; k < K; k++) {
            int ti = t + k - pad;
            if (ti < 0 || ti >= T) continue;
            acc += LOAD(w, wbase + k) * LOAD(in, ibase + ti);
        }
    }
    STORE(out, gid, acc);
}

// LayerNorm over channel dim for a [T, C] row-major tensor (one row = one time
// step, normalized across C). affine gamma/beta [C]. One work-item per row.
__kernel void te_layernorm_rows(
    __global const storage_t* in,
    __global const storage_t* gamma,
    __global const storage_t* beta,
    __global storage_t*       out,
    const int rows,
    const int C,
    const float eps) {
    int r = get_global_id(0);
    if (r >= rows) return;
    int base = r * C;
    float mean = 0.0f;
    for (int i = 0; i < C; i++) mean += LOAD(in, base + i);
    mean /= (float)C;
    float var = 0.0f;
    for (int i = 0; i < C; i++) { float v = LOAD(in, base + i) - mean; var += v * v; }
    var /= (float)C;
    float inv = rsqrt(var + eps);
    for (int i = 0; i < C; i++) {
        float v = (LOAD(in, base + i) - mean) * inv;
        STORE(out, base + i, v * LOAD(gamma, i) + LOAD(beta, i));
    }
}

// LeakyReLU, slope 0.2, in place-safe (in != out).
__kernel void te_leaky_relu(
    __global const storage_t* in,
    __global storage_t*       out,
    const int n,
    const float slope) {
    int gid = get_global_id(0);
    if (gid >= n) return;
    float x = LOAD(in, gid);
    STORE(out, gid, x >= 0.0f ? x : slope * x);
}

// Transpose [A, B] -> [B, A] row-major.
__kernel void te_transpose(
    __global const storage_t* in,
    __global storage_t*       out,
    const int A,
    const int B) {
    int gid = get_global_id(0);
    if (gid >= A * B) return;
    int a = gid / B;
    int b = gid - a * B;
    STORE(out, b * A + a, LOAD(in, a * B + b));
}

// Add bias broadcast over rows: out[r, c] = in[r, c] + bias[c]
__kernel void te_add_bias(
    __global const storage_t* in,
    __global const storage_t* bias,
    __global storage_t*       out,
    const int rows,
    const int cols) {
    int gid = get_global_id(0);
    if (gid >= rows * cols) return;
    int c = gid % cols;
    STORE(out, gid, LOAD(in, gid) + LOAD(bias, c));
}

// One time step of an LSTM for ALL hidden units of ONE direction.
// x_t [I] current input row; h_prev [Hd]; c_prev [Hd] (state, updated in place).
// W [4*Hd, I] input weights (gate order ONNX: i, o, f, g == input,output,forget,cell);
// R [4*Hd, Hd] recurrent weights; Wb [4*Hd] + Rb [4*Hd] biases.
// Writes h_out [Hd] and updates c_prev. One work-item per hidden unit.
// ONNX LSTM gate layout is [i, o, f, c(g)]; standard formulas.
__kernel void te_lstm_step(
    __global const storage_t* x_t,
    __global const storage_t* h_prev,
    __global storage_t*       c_state,
    __global const storage_t* W,
    __global const storage_t* R,
    __global const storage_t* Wb,
    __global const storage_t* Rb,
    __global storage_t*       h_out,
    const int I,
    const int Hd) {
    int j = get_global_id(0);
    if (j >= Hd) return;
    // gate base offsets in the 4*Hd packed weight rows (ONNX order i,o,f,g)
    int gi = 0 * Hd + j;
    int go = 1 * Hd + j;
    int gf = 2 * Hd + j;
    int gg = 3 * Hd + j;
    float ai = LOAD(Wb, gi) + LOAD(Rb, gi);
    float ao = LOAD(Wb, go) + LOAD(Rb, go);
    float af = LOAD(Wb, gf) + LOAD(Rb, gf);
    float ag = LOAD(Wb, gg) + LOAD(Rb, gg);
    for (int k = 0; k < I; k++) {
        float xv = LOAD(x_t, k);
        ai += LOAD(W, gi * I + k) * xv;
        ao += LOAD(W, go * I + k) * xv;
        af += LOAD(W, gf * I + k) * xv;
        ag += LOAD(W, gg * I + k) * xv;
    }
    for (int k = 0; k < Hd; k++) {
        float hv = LOAD(h_prev, k);
        ai += LOAD(R, gi * Hd + k) * hv;
        ao += LOAD(R, go * Hd + k) * hv;
        af += LOAD(R, gf * Hd + k) * hv;
        ag += LOAD(R, gg * Hd + k) * hv;
    }
    float ig = 1.0f / (1.0f + exp(-ai));
    float og = 1.0f / (1.0f + exp(-ao));
    float fg = 1.0f / (1.0f + exp(-af));
    float cg = tanh(ag);
    float c_new = fg * LOAD(c_state, j) + ig * cg;
    STORE(c_state, j, c_new);
    STORE(h_out, j, og * tanh(c_new));
}
