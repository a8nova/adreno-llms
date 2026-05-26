// Reference: model_info/transformers_src/modeling_idefics3.py:90-167 Idefics3VisionEmbeddings.forward (position_ids bucketize + position_embedding add)

#ifdef USE_FP16
  #pragma OPENCL EXTENSION cl_khr_fp16 : enable
  typedef half storage_t;
  #define LOAD(p, i)     vload_half((i), (p))
  #define STORE(p, i, v) vstore_half((v), (i), (p))
#else
  typedef float storage_t;
  #define LOAD(p, i)     ((p)[(i)])
  #define STORE(p, i, v) ((p)[(i)] = (v))
#endif

// Idefics3VisionEmbeddings (transformers/models/idefics3/modeling_idefics3.py)
//
// forward() summary:
//   1) patch_embeds = Conv2d(pixel_values)  -> handled by host (im2col+GEMM or direct conv), no custom kernel here.
//   2) embeddings = patch_embeds.flatten(2).transpose(1,2) -> view/transpose only, host can fold.
//   3) position_ids computed from patch_attention_mask via arange/clamp/bucketize and masked scatter.
//   4) embeddings += position_embedding(position_ids) -> embedding gather + elementwise add.
//
// This file provides kernels for steps (3) and (4):
//   - compute_position_ids_bucketize: vectorized bucketize-based position id generation.
//   - embedding_gather_add: gather position embeddings and add to embeddings.

inline int clamp_int(int v, int lo, int hi) { return v < lo ? lo : (v > hi ? hi : v); }

__kernel void compute_position_ids_bucketize(
    __global const uchar* patch_attention_mask, // [B, max_patches_h, max_patches_w] values 0/1
    __global int* position_ids,                 // [B, max_patches_h*max_patches_w]
    const int batch_size,
    const int max_patches_h,
    const int max_patches_w,
    const int num_patches_per_side              // = image_size/patch_size
) {
    const int b = get_global_id(0);
    if (b >= batch_size) return;

    // Count valid patches along H and W:
    // nb_patches_h = sum over h of mask[h,0]
    // nb_patches_w = sum over w of mask[0,w]
    int nb_h = 0;
    for (int h = 0; h < max_patches_h; ++h) {
        const int idx = (b * max_patches_h + h) * max_patches_w + 0;
        nb_h += (patch_attention_mask[idx] != (uchar)0);
    }
    int nb_w = 0;
    for (int w = 0; w < max_patches_w; ++w) {
        const int idx = (b * max_patches_h + 0) * max_patches_w + w;
        nb_w += (patch_attention_mask[idx] != (uchar)0);
    }
    nb_h = nb_h > 0 ? nb_h : 1;
    nb_w = nb_w > 0 ? nb_w : 1;

    const float step_h = 1.0f / (float)nb_h;
    const float step_w = 1.0f / (float)nb_w;

    // For each (h,w):
    // fractional_coords_h = clamp(h * step_h, max=1-1e-6)
    // bucket_coords_h = floor(fractional_coords_h * num_patches_per_side)
    // similarly for w
    // pos_id = bucket_h * num_patches_per_side + bucket_w
    // position_ids[mask==1] = pos_id; else keep 0.
    const int hw = max_patches_h * max_patches_w;
    for (int i = 0; i < hw; ++i) {
        const int h = i / max_patches_w;
        const int w = i - h * max_patches_w;
        const int mask_idx = (b * max_patches_h + h) * max_patches_w + w;
        const int out_idx = b * hw + i;

        if (patch_attention_mask[mask_idx] == (uchar)0) {
            position_ids[out_idx] = 0;
            continue;
        }

        float fh = (float)h * step_h;
        float fw = (float)w * step_w;
        const float maxv = 1.0f - 1.0e-6f;
        fh = fh > maxv ? maxv : fh;
        fw = fw > maxv ? maxv : fw;

        int bucket_h = (int)floor(fh * (float)num_patches_per_side);
        int bucket_w = (int)floor(fw * (float)num_patches_per_side);
        bucket_h = clamp_int(bucket_h, 0, num_patches_per_side - 1);
        bucket_w = clamp_int(bucket_w, 0, num_patches_per_side - 1);

        position_ids[out_idx] = bucket_h * num_patches_per_side + bucket_w;
    }
}

__kernel void embedding_gather_add(
    __global const storage_t* embeddings,       // [B, seq, D]
    __global const storage_t* position_weight,  // [num_positions, D]
    __global const int* position_ids,           // [B, seq]
    __global storage_t* out,                    // [B, seq, D]
    const int batch_size,
    const int seq_len,
    const int embed_dim,
    const int num_positions
) {
    const int gid = get_global_id(0);
    const int total = batch_size * seq_len * embed_dim;
    if (gid >= total) return;

    const int d = gid % embed_dim;
    const int tmp = gid / embed_dim;
    const int s = tmp % seq_len;
    const int b = tmp / seq_len;

    int pid = position_ids[b * seq_len + s];
    pid = clamp_int(pid, 0, num_positions - 1);

    const int emb_idx = (b * seq_len + s) * embed_dim + d;
    const int pos_idx = pid * embed_dim + d;

    const float e = (float)LOAD(embeddings, emb_idx);
    const float p = (float)LOAD(position_weight, pos_idx);
    STORE(out, emb_idx, (storage_t)(e + p));
}
