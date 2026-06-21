// Reference: model_info/transformers_src/modeling_musicgen.py (nn.LayerNorm usage)
// Single-workgroup-per-row LayerNorm with local reduction + vectorized half loads.
//
// Previous version launched gws={rows}: ONE work-item walked all `cols` (1024)
// columns serially → 1.24ms/call on Adreno for a microseconds-math op (95% of
// total GPU time per profile_summary.txt). This rewrite assigns one workgroup
// (LN_WG threads) per row, splits the column loop across threads, reduces mean
// and variance in local memory, then writes the affine output cooperatively.
// Math is IDENTICAL: mean/var over cols (population variance), rsqrt(var+eps),
// optional gamma (weight) and beta (bias). fp16 storage loaded/stored via
// vload_half/vstore_half (LOAD/STORE), accumulation in fp32.

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

#define LN_WG 256

__kernel __attribute__((reqd_work_group_size(LN_WG, 1, 1)))
void layernorm_simple(__global const storage_t* in,
                      __global storage_t* out,
                      const int rows,
                      const int cols,
                      const float eps,
                      __global const storage_t* gamma_opt,
                      __global const storage_t* beta_opt,
                      const int has_gamma,
                      const int has_beta) {
  const int row = get_group_id(0);
  if (row >= rows) return;
  const int lid = get_local_id(0);
  const int base = row * cols;

  __local float scratch[LN_WG];

  // ── Pass 1: sum for mean ──────────────────────────────────────────────
  float local_sum = 0.0f;
  for (int c = lid; c < cols; c += LN_WG) {
    local_sum += LOAD(in, base + c);
  }
  scratch[lid] = local_sum;
  barrier(CLK_LOCAL_MEM_FENCE);
  for (int off = LN_WG >> 1; off > 0; off >>= 1) {
    if (lid < off) scratch[lid] += scratch[lid + off];
    barrier(CLK_LOCAL_MEM_FENCE);
  }
  const float mean = scratch[0] / (float)cols;
  barrier(CLK_LOCAL_MEM_FENCE);

  // ── Pass 2: sum of squared deviations for variance ────────────────────
  float local_var = 0.0f;
  for (int c = lid; c < cols; c += LN_WG) {
    float d = LOAD(in, base + c) - mean;
    local_var += d * d;
  }
  scratch[lid] = local_var;
  barrier(CLK_LOCAL_MEM_FENCE);
  for (int off = LN_WG >> 1; off > 0; off >>= 1) {
    if (lid < off) scratch[lid] += scratch[lid + off];
    barrier(CLK_LOCAL_MEM_FENCE);
  }
  const float var = scratch[0] / (float)cols;
  const float inv_std = rsqrt(var + eps);

  // ── Affine output ─────────────────────────────────────────────────────
  for (int c = lid; c < cols; c += LN_WG) {
    float x = (LOAD(in, base + c) - mean) * inv_std;
    if (has_gamma) x *= LOAD(gamma_opt, c);
    if (has_beta)  x += LOAD(beta_opt, c);
    STORE(out, base + c, x);
  }
}
