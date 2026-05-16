// Reference: model_info/transformers_src/modeling_llama.py:20-38 LlamaRMSNorm.forward
// Implements RMSNorm: y = weight * (x * rsqrt(mean(x^2) + eps))
//
// Cooperative-WG variant: one workgroup per row, WG=64 threads cooperate
// over the column dimension via vec4 fp16 loads + __local-mem tree
// reduction. Host launches with gws=rows*64, lws=64. Bumping WG requires
// kernel rewrite (the kp / kp4 split below assumes WG=64).

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

#ifdef USE_SUBGROUP_REDUCE
  #pragma OPENCL EXTENSION cl_khr_subgroups : enable
#endif

#define WG_SIZE 64
#define MAX_SUBGROUPS 8   // WG_SIZE / smallest plausible Adreno wave (8)

__kernel void rmsnorm_forward(
    __global const storage_t* x,
    __global const storage_t* weight,
    __global storage_t* out,
    const int rows,
    const int cols,
    const float eps) {
  const int row = (int)get_group_id(0);
  const int lid = (int)get_local_id(0);
  if (row >= rows) return;

  const int base = row * cols;
#ifdef USE_SUBGROUP_REDUCE
  __local float subg_scratch[MAX_SUBGROUPS];
#else
  __local float partial[WG_SIZE];
#endif

#ifdef USE_FP16
  // Vec4 fp16 path — cols expected to be multiple of 4 (head_dim ⌃
  // hidden_size for every common transformer). Scalar tail handles
  // any leftover (unusual configs).
  const int C4 = cols >> 2;
  __global const half* xh = (__global const half*)x + base;
  __global const half* wh = (__global const half*)weight;

  // Pass 1: partial sum-of-squares over this thread's strided vec4 cols.
  float acc = 0.0f;
  for (int c4 = lid; c4 < C4; c4 += WG_SIZE) {
    float4 v = vload_half4(c4, xh);
    acc += dot(v, v);
  }
  for (int c = C4 * 4 + lid; c < cols; c += WG_SIZE) {
    float v = LOAD(x, base + c);
    acc += v * v;
  }

#ifdef USE_SUBGROUP_REDUCE
  // Subgroup reduce (PDF §6.5/§8.9/§9.2): hardware-accelerated, no __local round-trip.
  // Most Adreno 6xx kernels run as 1 full-wave subgroup at WG=64, in which case
  // the cross-subgroup roll-up below is a single-element no-op. The pattern stays
  // correct under half-wave (sub_group_size=32 → 2 subgroups) as well.
  float subg_sum = sub_group_reduce_add(acc);
  const uint sg_id  = get_sub_group_id();
  const uint sg_lid = get_sub_group_local_id();
  const uint nsg    = get_num_sub_groups();
  if (sg_lid == 0) subg_scratch[sg_id] = subg_sum;
  barrier(CLK_LOCAL_MEM_FENCE);
  float total;
  if (sg_id == 0) {
    float v = (sg_lid < nsg) ? subg_scratch[sg_lid] : 0.0f;
    total = sub_group_reduce_add(v);
    if (sg_lid == 0) subg_scratch[0] = total;
  }
  barrier(CLK_LOCAL_MEM_FENCE);
  const float inv_rms = native_rsqrt((subg_scratch[0] / (float)cols) + eps);
#else
  partial[lid] = acc;
  barrier(CLK_LOCAL_MEM_FENCE);

  for (int s = WG_SIZE >> 1; s > 0; s >>= 1) {
    if (lid < s) partial[lid] += partial[lid + s];
    barrier(CLK_LOCAL_MEM_FENCE);
  }
  const float inv_rms = native_rsqrt((partial[0] / (float)cols) + eps);
#endif

  // Pass 2: vec4 normalize+weight write.
  __global half* oh = (__global half*)out + base;
  for (int c4 = lid; c4 < C4; c4 += WG_SIZE) {
    float4 v = vload_half4(c4, xh);
    float4 w = vload_half4(c4, wh);
    vstore_half4(v * inv_rms * w, c4, oh);
  }
  for (int c = C4 * 4 + lid; c < cols; c += WG_SIZE) {
    float v = LOAD(x, base + c);
    float w = LOAD(weight, c);
    STORE(out, base + c, (v * inv_rms) * w);
  }
#else
  // Fp32 fall-back path — same cooperative structure, scalar reads.
  float acc = 0.0f;
  for (int c = lid; c < cols; c += WG_SIZE) {
    float v = LOAD(x, base + c);
    acc += v * v;
  }
  partial[lid] = acc;
  barrier(CLK_LOCAL_MEM_FENCE);
  for (int s = WG_SIZE >> 1; s > 0; s >>= 1) {
    if (lid < s) partial[lid] += partial[lid + s];
    barrier(CLK_LOCAL_MEM_FENCE);
  }
  const float inv_rms = native_rsqrt((partial[0] / (float)cols) + eps);
  for (int c = lid; c < cols; c += WG_SIZE) {
    float v = LOAD(x, base + c);
    float w = LOAD(weight, c);
    STORE(out, base + c, (v * inv_rms) * w);
  }
#endif
}
