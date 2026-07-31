// Reference: kitten-tts-nano-0.1 ONNX graph /bert/* (ALBERT phoneme encoder)
// Kernels for the Bert submodel: embedding sum, layernorm, gelu, attention.
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

// Embedding sum: out[t, d] = word_emb[ids[t], d]
//                          + token_type_emb[0, d]
//                          + position_emb[t, d]
// ids: int32 [T]; word_emb [V, H]; type_emb [2, H]; pos_emb [Pmax, H]
__kernel void bert_embed_sum(
    __global const int*        ids,
    __global const storage_t*  word_emb,
    __global const storage_t*  type_emb,
    __global const storage_t*  pos_emb,
    __global storage_t*        out,
    const int T,
    const int H) {
    int gid = get_global_id(0);
    int total = T * H;
    if (gid >= total) return;
    int t = gid / H;
    int d = gid - t * H;
    int tok = ids[t];
    float w = LOAD(word_emb, tok * H + d);
    float ty = LOAD(type_emb, d);            // token_type id 0
    float p = LOAD(pos_emb, t * H + d);
    STORE(out, gid, w + ty + p);
}

// LayerNorm over last dim (H), affine gamma/beta. One work-item per row.
__kernel void bert_layernorm(
    __global const storage_t*  in,
    __global const storage_t*  gamma,
    __global const storage_t*  beta,
    __global storage_t*        out,
    const int rows,
    const int H,
    const float eps) {
    int r = get_global_id(0);
    if (r >= rows) return;
    int base = r * H;
    float mean = 0.0f;
    for (int i = 0; i < H; i++) mean += LOAD(in, base + i);
    mean /= (float)H;
    float var = 0.0f;
    for (int i = 0; i < H; i++) { float v = LOAD(in, base + i) - mean; var += v * v; }
    var /= (float)H;
    float inv = rsqrt(var + eps);
    for (int i = 0; i < H; i++) {
        float v = (LOAD(in, base + i) - mean) * inv;
        float g = LOAD(gamma, i);
        float b = LOAD(beta, i);
        STORE(out, base + i, v * g + b);
    }
}

// Exact GELU (erf form): 0.5*x*(1+erf(x/sqrt(2))). ALBERT uses gelu_new? No —
// HF ALBERT default hidden_act = "gelu_new" (tanh approx). Kitten ONNX traces
// the activation; we match the tanh approximation used by the graph.
// gelu_new(x) = 0.5*x*(1+tanh(sqrt(2/pi)*(x+0.044715*x^3)))
__kernel void bert_gelu_new(
    __global const storage_t* in,
    __global storage_t*       out,
    const int n) {
    int gid = get_global_id(0);
    if (gid >= n) return;
    float x = LOAD(in, gid);
    const float c = 0.7978845608028654f; // sqrt(2/pi)
    float inner = c * (x + 0.044715f * x * x * x);
    float y = 0.5f * x * (1.0f + tanh(inner));
    STORE(out, gid, y);
}

// Add bias broadcast over rows: out[r, c] = in[r, c] + bias[c]
__kernel void bert_add_bias(
    __global const storage_t* in,
    __global const storage_t* bias,
    __global storage_t*       out,
    const int rows,
    const int cols) {
    int gid = get_global_id(0);
    int total = rows * cols;
    if (gid >= total) return;
    int c = gid % cols;
    STORE(out, gid, LOAD(in, gid) + LOAD(bias, c));
}

// Residual add: out[i] = a[i] + b[i]
__kernel void bert_residual_add(
    __global const storage_t* a,
    __global const storage_t* b,
    __global storage_t*       out,
    const int n) {
    int gid = get_global_id(0);
    if (gid >= n) return;
    STORE(out, gid, LOAD(a, gid) + LOAD(b, gid));
}

// Multi-head self-attention for one ALBERT layer.
// Q,K,V are [T, H] row-major, H = num_heads*head_dim. Non-causal (full).
// Computes context[T, H] = softmax(Q Kt / sqrt(head_dim)) V, head-blocked so
// output stays token-major [T, H] (head h occupies cols [h*hd, (h+1)*hd)).
// One work-item per (query token t, head h). head_dim <= 64.
// ALBERT head_dim is 64 for this model; fixed at compile time so Q/acc live in
// registers and the inner loops unroll. OPT: flash-attention single pass —
// compute each Q·K dot ONCE with online softmax, instead of recomputing the dot
// head_dim×T times in a pass-3 triple loop (the old kernel was ~30× redundant:
// 944 ms/layer → the entire 11.3 s Bert bottleneck). K and V are each read once.
#define BERT_HEAD_DIM 64
__kernel void bert_attention(
    __global const storage_t* Q,
    __global const storage_t* K,
    __global const storage_t* V,
    __global storage_t*       out,
    const int T,
    const int num_heads,
    const int head_dim) {
    int gid = get_global_id(0);
    int total = T * num_heads;
    if (gid >= total) return;
    int t = gid / num_heads;
    int h = gid - t * num_heads;
    int H = num_heads * head_dim;
    int qbase = t * H + h * head_dim;
    int hoff  = h * head_dim;
    float scale = 1.0f / sqrt((float)head_dim);

#ifndef USE_FP16
    // fp32 float4 path: head_dim=64 → 16 float4 lanes. Vectorizes the dot and the
    // V-accumulate 4× (this kernel is issue-bound on 64-wide scalar loops).
    #define HD4 (BERT_HEAD_DIM / 4)
    float4 qreg[HD4];
    float4 acc[HD4];
    for (int d = 0; d < HD4; d++) { qreg[d] = vload4(0, Q + qbase + d * 4); acc[d] = (float4)(0.0f); }
    float m = -1e30f, l = 0.0f;
    for (int j = 0; j < T; j++) {
        int kb = j * H + hoff;
        // Sum in SCALAR lane order (d=0,1,2,3,…) — a 4-way tree reduction reorders
        // the sum by ~1e-6 and flips a duration rounding boundary (159→160 frames),
        // breaking oracle reproducibility. Vectorized LOADS are the win; keep the
        // add order identical to the scalar path.
        float dot = 0.0f;
        for (int d = 0; d < HD4; d++) {
            float4 q4 = qreg[d], k4 = vload4(0, K + kb + d * 4);
            dot += q4.x * k4.x; dot += q4.y * k4.y; dot += q4.z * k4.z; dot += q4.w * k4.w;
        }
        dot *= scale;
        float mnew = fmax(m, dot);
        float corr = exp(m - mnew);
        float wj   = exp(dot - mnew);
        l = l * corr + wj;
        int vb = j * H + hoff;
        for (int d = 0; d < HD4; d++) acc[d] = acc[d] * corr + wj * vload4(0, V + vb + d * 4);
        m = mnew;
    }
    float invl = 1.0f / l;
    for (int d = 0; d < HD4; d++) vstore4(acc[d] * invl, 0, out + qbase + d * 4);
#else
    float qreg[BERT_HEAD_DIM];
    float acc[BERT_HEAD_DIM];
    for (int d = 0; d < BERT_HEAD_DIM; d++) { qreg[d] = LOAD(Q, qbase + d); acc[d] = 0.0f; }
    float m = -1e30f, l = 0.0f;              // running max, running denom
    for (int j = 0; j < T; j++) {
        int kb = j * H + hoff;
        float dot = 0.0f;
        for (int d = 0; d < BERT_HEAD_DIM; d++) dot += qreg[d] * LOAD(K, kb + d);
        dot *= scale;
        float mnew = fmax(m, dot);
        float corr = exp(m - mnew);          // rescale prior accumulation
        float wj   = exp(dot - mnew);
        l = l * corr + wj;
        int vb = j * H + hoff;
        for (int d = 0; d < BERT_HEAD_DIM; d++) acc[d] = acc[d] * corr + wj * LOAD(V, vb + d);
        m = mnew;
    }
    float invl = 1.0f / l;
    for (int d = 0; d < BERT_HEAD_DIM; d++) STORE(out, qbase + d, acc[d] * invl);
#endif
}
