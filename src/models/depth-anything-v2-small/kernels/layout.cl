// Depth-Anything-V2 (DINOv2 backbone + DPT neck/head) OpenCL kernels.
// One FILE = one cl_program on purpose: Adreno applies a program-wide
// worst-case register footprint, so a fat kernel in a shared program cuts
// wave occupancy for every kernel in it (measured 2026-07-07: frame
// 6.6s -> 10.5s from two never-dispatched kernels). Keep op families
// isolated. All data buffers are storage_t (fp16 or fp32); accumulators
// are float.

#ifdef USE_FP16
  #pragma OPENCL EXTENSION cl_khr_fp16 : enable
  typedef half storage_t;
  #define LOAD(p,i)   vload_half((i),(p))
  #define STORE(p,i,v) vstore_half((v),(i),(p))
  #define LOAD4(p,i)   vload_half4(0, (p) + (i))
  #define STORE4(p,i,v) vstore_half4((v), 0, (p) + (i))
#else
  typedef float storage_t;
  #define LOAD(p,i)   ((p)[(i)])
  #define STORE(p,i,v) ((p)[(i)] = (v))
  #define LOAD4(p,i)   vload4(0, (p) + (i))
  #define STORE4(p,i,v) vstore4((v), 0, (p) + (i))
#endif


// ── embeddings_assemble: build [1+Np, D] token matrix from patch conv output ──
// patch_conv: [D, Hp, Wp]  (output of patch projection, channel-major)
// cls_token:  [D]
// pos_embed:  [(1+Np), D]   (already interpolated, row-major per token)
// output:     [(1+Np), D]   row-major; token 0 = cls, tokens 1.. = patches
// Patch token t (0..Np-1) at grid (ph,pw) has feature d = patch_conv[d, ph, pw].
// PyTorch: projection(pixel).flatten(2).transpose(1,2) => token order row-major
// over (Hp,Wp), feature d. Then cat(cls, patches) + pos_embed.

__kernel void embeddings_assemble(
    __global const storage_t* patch_conv,
    __global const storage_t* cls_token,
    __global const storage_t* pos_embed,
    __global storage_t* output,
    const int D, const int Hp, const int Wp)
{
    int idx = get_global_id(0);
    int Np = Hp * Wp;
    int ntok = Np + 1;
    int total = ntok * D;
    if (idx >= total) return;

    int d = idx % D;
    int tok = idx / D;

    float val;
    if (tok == 0) {
        val = (float)LOAD(cls_token, d);
    } else {
        int p = tok - 1;            // patch index over (Hp,Wp) row-major
        // patch_conv is [D, Hp, Wp]; feature d at spatial p
        val = (float)LOAD(patch_conv, d * Np + p);
    }
    val += (float)LOAD(pos_embed, tok * D + d);
    STORE(output, idx, val);
}

// ── drop CLS + reshape [1+Np, D] -> CHW [D, Hp, Wp] ──
// tokens 1.. are patches row-major over (Hp,Wp). output channel-major.

__kernel void tokens_to_chw(
    __global const storage_t* tokens,  // [1+Np, D]
    __global storage_t* out_chw,       // [D, Hp, Wp]
    const int D, const int Hp, const int Wp)
{
    int idx = get_global_id(0);
    int Np = Hp * Wp;
    int total = D * Np;
    if (idx >= total) return;
    int p = idx % Np;          // spatial
    int d = idx / Np;          // channel
    // token = p+1, feature d
    float v = (float)LOAD(tokens, (p + 1) * D + d);
    STORE(out_chw, d * Np + p, v);
}

// ── chw_to_tokens: [D, Np] channel-major -> [Np, D] token-major ──
// Mirrors PyTorch Dinov2PatchEmbeddings.forward: projection(pixel).flatten(2)
// .transpose(1,2). Our conv output is [D, Hp*Wp] (channel-major); the reference
// hook captures the transposed [Np, D] token matrix. Used ONLY to produce a
// layout-matched dump for SxS — the live pipeline consumes CHW directly.

__kernel void chw_to_tokens(
    __global const storage_t* in_chw,   // [D, Np]
    __global storage_t* out_tok,        // [Np, D]
    const int D, const int Np)
{
    int idx = get_global_id(0);
    int total = D * Np;
    if (idx >= total) return;
    int p = idx % Np;   // spatial
    int d = idx / Np;   // channel
    // in_chw[d, p] -> out_tok[p, d]
    STORE(out_tok, p * D + d, (float)LOAD(in_chw, d * Np + p));
}

// ── bilinear interpolate NCHW [C,Hin,Win] -> [C,Hout,Wout] ──
// align_corners flag selects coordinate mapping. matches torch F.interpolate.

__kernel void bilinear_interp(
    __global const storage_t* input,
    __global storage_t* output,
    const int C, const int Hin, const int Win,
    const int Hout, const int Wout,
    const int align_corners)
{
    int idx = get_global_id(0);
    int total = C * Hout * Wout;
    if (idx >= total) return;
    int ow = idx % Wout;
    int oh = (idx / Wout) % Hout;
    int c = idx / (Wout * Hout);

    float in_y, in_x;
    if (align_corners) {
        in_y = (Hout > 1) ? (float)oh * (float)(Hin - 1) / (float)(Hout - 1) : 0.0f;
        in_x = (Wout > 1) ? (float)ow * (float)(Win - 1) / (float)(Wout - 1) : 0.0f;
    } else {
        in_y = ((float)oh + 0.5f) * (float)Hin / (float)Hout - 0.5f;
        in_x = ((float)ow + 0.5f) * (float)Win / (float)Wout - 0.5f;
    }
    int y0 = (int)floor(in_y);
    int x0 = (int)floor(in_x);
    float dy = in_y - (float)y0;
    float dx = in_x - (float)x0;
    int y1 = y0 + 1;
    int x1 = x0 + 1;
    // clamp
    int y0c = y0 < 0 ? 0 : (y0 >= Hin ? Hin - 1 : y0);
    int y1c = y1 < 0 ? 0 : (y1 >= Hin ? Hin - 1 : y1);
    int x0c = x0 < 0 ? 0 : (x0 >= Win ? Win - 1 : x0);
    int x1c = x1 < 0 ? 0 : (x1 >= Win ? Win - 1 : x1);
    int cb = c * Hin * Win;
    float v00 = (float)LOAD(input, cb + y0c * Win + x0c);
    float v01 = (float)LOAD(input, cb + y0c * Win + x1c);
    float v10 = (float)LOAD(input, cb + y1c * Win + x0c);
    float v11 = (float)LOAD(input, cb + y1c * Win + x1c);
    float top = v00 * (1.0f - dx) + v01 * dx;
    float bot = v10 * (1.0f - dx) + v11 * dx;
    float val = top * (1.0f - dy) + bot * dy;
    STORE(output, idx, val);
}
