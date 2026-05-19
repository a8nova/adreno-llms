// Reference: model_info/transformers_src/modeling_vits.py: VitsModel.forward (prior_latents sampling)
// prior_latents = prior_means + randn_like(prior_means) * exp(prior_log_variances) * noise_scale

// NOTE: On-device build uses OpenCLContext::build_program_from_file which does not set an include search path.
// So we must inline the utils.cl LOAD/STORE preamble instead of #include "utils.cl".

#ifdef USE_FP16
  #pragma OPENCL EXTENSION cl_khr_fp16 : enable
  typedef half storage_t;
  #define LOAD(p,i)   vload_half((i),(p))
  #define STORE(p,i,v) vstore_half((v),(i),(p))
#else
  typedef float storage_t;
  #define LOAD(p,i)   ((p)[(i)])
  #define STORE(p,i,v) ((p)[(i)] = (v))
#endif

__kernel void sample_prior_affine(
    __global const storage_t* prior_means,
    __global const storage_t* prior_log_variances,
    __global const storage_t* noise,
    __global storage_t* out,
    const float noise_scale,
    const int B,
    const int C,
    const int T) {
  const int gid = (int)get_global_id(0);
  const int n = B * C * T;
  if (gid >= n) return;

  // fp16 safety: even modestly-positive log-scale values overflow exp() quickly.
  // The combination sigma * noise * noise_scale must stay within fp16 range
  // (max ~65504). With noise = N(0,1) ranging up to ~4, noise_scale ~0.667,
  // and sigma = exp(logv): if logv = 3, sigma=20 → max product ~53. Tight enough.
  // Earlier clamp of [-10, 10] produced sigma up to 22000 and ±inf in z_prior.
  const float LOGV_MIN = -3.0f;
  const float LOGV_MAX =  3.0f;

  const float mu = LOAD(prior_means, gid);
  float logv = LOAD(prior_log_variances, gid);
  if (logv < LOGV_MIN) logv = LOGV_MIN;
  if (logv > LOGV_MAX) logv = LOGV_MAX;
  const float nz = LOAD(noise, gid);

  const float sigma = exp(logv);
  const float z = mu + nz * sigma * noise_scale;
  STORE(out, gid, z);
}
