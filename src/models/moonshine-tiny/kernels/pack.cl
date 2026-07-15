// Layout/packing kernels: RoPE, head split/merge, permute, bias add.
// Owned by src/ops/MoonshineAttention.cpp / Linear.cpp / MoonshineEncoderFrontend.cpp.
// Split from the former kernels/moonshine.cl monolith (whisper-parity layout);
// kernel bodies are byte-identical to the monolith at the time of the split.
// Dtype-template preamble — DO NOT EDIT. Driven by host-side -DNNOPT_USE_FP16.
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

// ── permute [C, L] -> [L, C] ────────────────────────────────────────────
__kernel void permute_cl_to_lc(
    __global const storage_t* input,   // [C, L]
    __global storage_t* output,        // [L, C]
    const int C,
    const int L) {
    int gid = get_global_id(0);
    const int total = C * L;
    if (gid >= total) return;
    int c = gid / L;
    int l = gid - c * L;
    STORE(output, l * C + c, LOAD(input, c * L + l));
}

// ── RoPE apply (interleaved / Moonshine style, partial rotary) ──────────
// Applies rotary embedding to Q or K, shaped [heads, seq, head_dim].
// Only the first rotary_dim dims are rotated; the rest pass through.
// cos/sin tables: [seq, rotary_dim] where rotary_dim entries are the
// repeat_interleaved(2) cos of the base freqs (so cos[2i]==cos[2i+1]).
// rotate_half interleaved: for pair (2i, 2i+1):
//    out[2i]   = x[2i]*cos[2i]   - x[2i+1]*sin[2i]
//    out[2i+1] = x[2i+1]*cos[2i+1] + x[2i]*sin[2i+1]
// (cos[2i]==cos[2i+1], sin[2i]==sin[2i+1])
__kernel void rope_apply(
    __global storage_t* x,             // [heads, seq, head_dim] in/out
    __global const storage_t* cos_t,   // [seq, rotary_dim]
    __global const storage_t* sin_t,   // [seq, rotary_dim]
    const int heads,
    const int seq,
    const int head_dim,
    const int rotary_dim,
    const int pos_offset) {            // absolute position of row 0
    int gid = get_global_id(0);        // one item per (head, seq_pos)
    const int total = heads * seq;
    if (gid >= total) return;
    int h = gid / seq;
    int s = gid - h * seq;
    int base = (h * seq + s) * head_dim;
    int abs_pos = s + pos_offset;
    int cbase = abs_pos * rotary_dim;
    for (int i = 0; i < rotary_dim; i += 2) {
        float x0 = LOAD(x, base + i);
        float x1 = LOAD(x, base + i + 1);
        float c = LOAD(cos_t, cbase + i);
        float sn = LOAD(sin_t, cbase + i);
        STORE(x, base + i,     x0 * c - x1 * sn);
        STORE(x, base + i + 1, x1 * c + x0 * sn);
    }
    // dims [rotary_dim, head_dim) pass through unchanged.
}

// ── transpose [seq, heads*head_dim] -> [heads, seq, head_dim] ───────────
// input row-major [seq, H*D] where projections are laid out per-position as
// concatenated heads. output [heads, seq, head_dim].
__kernel void split_heads(
    __global const storage_t* input,   // [seq, H*D]
    __global storage_t* output,        // [H, seq, D]
    const int seq,
    const int H,
    const int D) {
    int gid = get_global_id(0);
    const int total = seq * H * D;
    if (gid >= total) return;
    // input index: s*(H*D) + h*D + d
    int s = gid / (H * D);
    int rem = gid - s * (H * D);
    int h = rem / D;
    int d = rem - h * D;
    STORE(output, (h * seq + s) * D + d, LOAD(input, gid));
}

// ── merge [heads, seq, head_dim] -> [seq, heads*head_dim] ───────────────
__kernel void merge_heads(
    __global const storage_t* input,   // [H, seq, D]
    __global storage_t* output,        // [seq, H*D]
    const int seq,
    const int H,
    const int D) {
    int gid = get_global_id(0);
    const int total = seq * H * D;
    if (gid >= total) return;
    int h = gid / (seq * D);
    int rem = gid - h * (seq * D);
    int s = rem / D;
    int d = rem - s * D;
    STORE(output, s * (H * D) + h * D + d, LOAD(input, gid));
}

// ── bias add (broadcast over rows): out[r,c] = in[r,c] + bias[c] ────────
__kernel void bias_add(
    __global const storage_t* in,
    __global const storage_t* bias,
    __global storage_t* out,
    const int rows,
    const int cols) {
    int gid = get_global_id(0);
    const int total = rows * cols;
    if (gid >= total) return;
    int c = gid % cols;
    STORE(out, gid, LOAD(in, gid) + LOAD(bias, c));
}
