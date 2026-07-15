// Dinov2Embeddings: patch-projection conv + bicubic-interpolated position
// embeddings + cls token → assembled token matrix [M, D].
// Code moved verbatim from the former monolithic backbone.cpp.

#include "depth_common.h"

// ─────────────────────────────────────────────────────────────────────
// Host-side bicubic interpolation of the DINOv2 patch position embeddings.
// PyTorch: reshape [1, sqrtN, sqrtN, D] -> permute [1,D,sqrtN,sqrtN] ->
//   F.interpolate(mode="bicubic", align_corners=False) -> [1,D,Hp,Wp].
// Computed in fp32 to match the reference (which upcasts to fp32).
// pos_patch: [Np_src, D] row-major (Np_src = src*src).  Returns [Hp*Wp, D].
// ─────────────────────────────────────────────────────────────────────
static inline float cubic_w(float t, float A) {
    t = fabsf(t);
    if (t <= 1.0f) return ((A + 2.0f) * t - (A + 3.0f)) * t * t + 1.0f;
    if (t < 2.0f)  return (((t - 5.0f) * t + 8.0f) * t - 4.0f) * A;
    return 0.0f;
}
static std::vector<float> bicubic_pos(const std::vector<float>& pos_patch,
                                      int src, int D, int Hp, int Wp) {
    const float A = -0.75f; // PyTorch bicubic constant
    std::vector<float> out((size_t)Hp * Wp * D);
    // align_corners=False coordinate mapping
    for (int oy = 0; oy < Hp; oy++) {
        float iy = ((float)oy + 0.5f) * (float)src / (float)Hp - 0.5f;
        int y0 = (int)floorf(iy);
        float fy = iy - (float)y0;
        float wy[4];
        for (int m = -1; m <= 2; m++) wy[m + 1] = cubic_w((float)m - fy, A);
        for (int ox = 0; ox < Wp; ox++) {
            float ix = ((float)ox + 0.5f) * (float)src / (float)Wp - 0.5f;
            int x0 = (int)floorf(ix);
            float fx = ix - (float)x0;
            float wx[4];
            for (int m = -1; m <= 2; m++) wx[m + 1] = cubic_w((float)m - fx, A);
            for (int d = 0; d < D; d++) {
                float acc = 0.0f;
                for (int m = -1; m <= 2; m++) {
                    int yy = y0 + m;
                    if (yy < 0) yy = 0; if (yy >= src) yy = src - 1;
                    for (int n = -1; n <= 2; n++) {
                        int xx = x0 + n;
                        if (xx < 0) xx = 0; if (xx >= src) xx = src - 1;
                        float v = pos_patch[(size_t)(yy * src + xx) * D + d];
                        acc += v * wy[m + 1] * wx[n + 1];
                    }
                }
                out[(size_t)(oy * Wp + ox) * D + d] = acc;
            }
        }
    }
    return out;
}

cl_mem embeddings_stage(Weights& Wt, const std::vector<float>& pixel_values,
                        int C, int H, int W) {
    const int D  = MODEL_CONFIG::HIDDEN_SIZE;   // 384
    const int PS = MODEL_CONFIG::PATCH_SIZE;    // 14
    const int Hp = H / PS;                      // 37
    const int Wp = W / PS;                      // 49
    const int M  = Hp * Wp + 1;                 // 1814 tokens

    // ── Patch embeddings: conv2d(pixel, proj.weight[D,3,14,14], proj.bias) stride14 ──
    cl_mem pix = upload_f32(pixel_values);
    int ph, pw;
    cl_mem patch = conv2d(pix, C, H, W,
                          Wt.get_buffer("backbone.embeddings.patch_embeddings.projection.weight"),
                          Wt.get_buffer("backbone.embeddings.patch_embeddings.projection.bias"),
                          D, PS, PS, PS, 0, ph, pw); // [D, 37, 49]
    pool_release(pix);
    // Reference dump boundaries (verified numerically against reference/layers/*):
    //  - `..._projection` reference == the Conv2d module output, CHANNEL-MAJOR
    //    [D, Np] (NCHW, BEFORE flatten+transpose). Dump the raw `patch` buffer,
    //    which is exactly [D, Np] channel-major. (Dumping the token-major copy
    //    here gives cos~0 because the ref-side dump-cap slices the first 144
    //    channels while a token-major dump slices the first 682 tokens — a
    //    layout+truncation artifact, NOT a math bug. Parent stage passes 0.9999.)
    //  - `..._patch_embeddings` reference == projection(pixel).flatten(2)
    //    .transpose(1,2) == [Np, D] token-major. Dump the token-major copy.
    {
        int Np_pe = ph * pw;
        DCHECK("backbone_embeddings_patch_embeddings_projection", patch, (size_t)D*Np_pe);
        cl_mem patch_tok = alloc_buf((size_t)Np_pe * D);
        cl_kernel kk = g_k.chw_to_tokens; int a = 0;
        sa(kk,a++,sizeof(cl_mem),&patch); sa(kk,a++,sizeof(cl_mem),&patch_tok);
        sa(kk,a++,sizeof(int),&D); sa(kk,a++,sizeof(int),&Np_pe);
        run1d(kk, (size_t)Np_pe * D);
        DCHECK("backbone_embeddings_patch_embeddings", patch_tok, (size_t)Np_pe*D);
        pool_release(patch_tok);
    }

    // ── Position embeddings: bicubic 37x37 -> 37x49 on host (fp32) ──
    // position_embeddings [1, 1370, D]; token0 = cls; tokens 1.. patch grid 37x37.
    std::vector<float> pos_all = Wt.get_host_vec("backbone.embeddings.position_embeddings"); // 1370*D
    int src = 37; // sqrt(1369)
    std::vector<float> pos_cls(pos_all.begin(), pos_all.begin() + D);
    std::vector<float> pos_patch_src(pos_all.begin() + D, pos_all.end()); // [1369, D]
    std::vector<float> pos_patch = bicubic_pos(pos_patch_src, src, D, Hp, Wp); // [Np, D]
    // assemble full pos_embed [M, D] = [cls; patches]
    std::vector<float> pos_full((size_t)M * D);
    for (int d = 0; d < D; d++) pos_full[d] = pos_cls[d];
    for (size_t i = 0; i < pos_patch.size(); i++) pos_full[D + i] = pos_patch[i];
    cl_mem pos_buf = upload_f32(pos_full);

    // ── embeddings_assemble: [M, D] tokens = cls/patch + pos ──
    cl_mem tokens = alloc_buf((size_t)M * D);
    { cl_kernel kk=g_k.embeddings_assemble; int a=0;
      cl_mem cls = Wt.get_buffer("backbone.embeddings.cls_token");
      sa(kk,a++,sizeof(cl_mem),&patch); sa(kk,a++,sizeof(cl_mem),&cls);
      sa(kk,a++,sizeof(cl_mem),&pos_buf); sa(kk,a++,sizeof(cl_mem),&tokens);
      sa(kk,a++,sizeof(int),&D); sa(kk,a++,sizeof(int),&Hp); sa(kk,a++,sizeof(int),&Wp);
      run1d(kk, (size_t)M*D); }
    pool_release(patch); pool_release(pos_buf);
    DCHECK("backbone_embeddings", tokens, (size_t)M*D);
    // Dinov2Embeddings.dropout is Identity at eval → output == embeddings.
    DCHECK("backbone_embeddings_dropout", tokens, (size_t)M*D);
    return tokens;
}
