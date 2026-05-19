// Reference: model_info/transformers_src/modeling_idefics3.py Idefics3Connector.pixel_shuffle
//
// Spatial pixel-shuffle with scale_factor=4 on the SmolVLM connector path.
// Compresses a (H=32, W=32) patch grid of dim=768 vectors into a (8, 8) grid
// of dim=12288 vectors by absorbing each 4x4 spatial block into the channel
// dim. Matches HF's reshape/permute chain bit-for-bit:
//
//   x[1, 32, 32, 768] -> view -> [1, 32, 8, 3072] -> perm(0,2,1,3) -> [1, 8, 32, 3072]
//   -> view -> [1, 8, 8, 12288] -> perm(0,2,1,3) -> [1, 8, 8, 12288] -> view -> [1, 64, 12288]
//
// Decomposes the output column c_out as: h_off*3072 + w_off*768 + c_in,
// where h_off,w_off ∈ [0,4). Reads input row (h_new*4 + h_off)*32 + (w_new*4 + w_off).
//
// In/out are row-major storage_t buffers; SCALE_FACTOR and inner dims are baked
// in to keep the kernel branchless. Dispatch shape: gws = {OUT_ROWS=64, OUT_COLS=12288}.

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

__kernel void pixel_shuffle(
    __global const storage_t* in_buf,    // [IN_H * IN_W, IN_C]
    __global       storage_t* out_buf,   // [OUT_ROWS, OUT_COLS]
    const int IN_H,                      // 32
    const int IN_W,                      // 32
    const int IN_C,                      // 768
    const int SCALE,                     // 4
    const int OUT_GRID,                  // IN_H / SCALE = 8
    const int OUT_COLS) {                // IN_C * SCALE * SCALE = 12288

    int out_row = get_global_id(0);
    int c_out = get_global_id(1);
    if (out_row >= OUT_GRID * OUT_GRID || c_out >= OUT_COLS) return;

    int h_new = out_row / OUT_GRID;
    int w_new = out_row - h_new * OUT_GRID;

    int per_h = SCALE * IN_C;   // 3072
    int h_off = c_out / per_h;
    int rem   = c_out - h_off * per_h;
    int w_off = rem / IN_C;
    int c_in  = rem - w_off * IN_C;

    int h_in = h_new * SCALE + h_off;
    int w_in = w_new * SCALE + w_off;

    int in_idx  = (h_in * IN_W + w_in) * IN_C + c_in;
    int out_idx = out_row * OUT_COLS + c_out;

    float v = LOAD(in_buf, in_idx);
    STORE(out_buf, out_idx, v);
}
