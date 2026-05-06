// Reference: model_info/transformers_src/modeling_mamba.py (MambaMixer + selective scan usage)
// Note: Core kernels are scaffolded in kernels/selective_scan.cl and kernels/causal_conv1d.cl.
// This file contains only small glue kernels used by src/layers/ssm.cpp.

// NOTE: Android OpenCL compiler does not support relative includes in source strings.
// Inline the minimal dtype/template preamble (canonical form from kernels/utils.cl).
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

#ifndef WG_SIZE
#define WG_SIZE 64
#endif

// Split xz [rows, 2*cols] into x [rows, cols] and z [rows, cols].
// Each row is laid out as [x_row (cols), z_row (cols)].
//
// Cooperative dispatch: gws = rows * WG_SIZE, lws = WG_SIZE. One workgroup
// per row; threads cooperate over `cols` (= d_inner = 1536 for Mamba-130M)
// via vec4 fp16 loads/stores. cols / WG_SIZE = 24 fp16 / thread = 6 vec4.
//
// Replaces the old "1 thread per row, scalar 1536-iter inner loop" version
// which was the largest single contributor to decode wall time (40% per
// profile). Adreno's clang doesn't auto-SIMDize the scalar form.
__kernel __attribute__((reqd_work_group_size(WG_SIZE, 1, 1)))
void split_xz(__global const storage_t* xz,
              __global storage_t* x,
              __global storage_t* z,
              const int cols,
              const int rows) {
  const int r   = get_group_id(0);
  const int tid = get_local_id(0);
  if (r >= rows) return;

  const int base       = r * (2 * cols);
  const int out_base   = r * cols;
  const int per_thread = cols / WG_SIZE;       // 24 for cols=1536, WG=64
  const int c_start    = tid * per_thread;

#ifdef USE_FP16
  // Vec4 fast path. cols must be divisible by 4 (1536 ✓).
  int j = 0;
  #pragma unroll
  for (; j + 3 < per_thread; j += 4) {
    const int off = c_start + j;
    float4 xv = vload_half4(0, xz + base + off);
    float4 zv = vload_half4(0, xz + base + cols + off);
    vstore_half4(xv, 0, x + out_base + off);
    vstore_half4(zv, 0, z + out_base + off);
  }
  // Scalar tail (only fires if per_thread % 4 != 0).
  for (; j < per_thread; ++j) {
    STORE(x, out_base + c_start + j, LOAD(xz, base + c_start + j));
    STORE(z, out_base + c_start + j, LOAD(xz, base + cols + c_start + j));
  }
#else
  for (int j = 0; j < per_thread; ++j) {
    STORE(x, out_base + c_start + j, LOAD(xz, base + c_start + j));
    STORE(z, out_base + c_start + j, LOAD(xz, base + cols + c_start + j));
  }
#endif
}

// Split xproj [rows, dt_rank + 2*d_state] into dt_raw [rows, dt_rank], B [rows, d_state], C [rows, d_state].
// For Mamba-130M: dt_rank=48, d_state=16. row stride = 80, output sizes are
// 48, 16, 16 — all divisible by 4. Single WG per row, 16 threads cooperate
// over the row (one thread per output column on the largest segment).
//
// Smaller scale than split_xz so we use a smaller WG; the launch overhead
// from a too-large WG would dominate the actual work for these tiny rows.
__kernel __attribute__((reqd_work_group_size(16, 1, 1)))
void split_xproj(__global const storage_t* xproj,
                 __global storage_t* dt_raw,
                 __global storage_t* B,
                 __global storage_t* C,
                 const int dt_rank,
                 const int d_state,
                 const int rows) {
  const int r   = get_group_id(0);
  const int tid = get_local_id(0);
  if (r >= rows) return;

  const int row_stride = dt_rank + 2 * d_state;
  const int base       = r * row_stride;
  const int dt_base    = r * dt_rank;
  const int bc_base    = r * d_state;

  // dt_rank elements (48 / 16 = 3 fp16 / thread).
  const int dtp = dt_rank / 16;
  const int dt_off = tid * dtp;
  for (int i = 0; i < dtp; ++i) {
    STORE(dt_raw, dt_base + dt_off + i, LOAD(xproj, base + dt_off + i));
  }
  // d_state elements for B (16 / 16 = 1 fp16 / thread).
  if (tid < d_state) {
    STORE(B, bc_base + tid, LOAD(xproj, base + dt_rank + tid));
    STORE(C, bc_base + tid, LOAD(xproj, base + dt_rank + d_state + tid));
  }
}

// Broadcast-add bias: y[row, col] += bias[col] for y shape [rows, cols]
__kernel void bias_add_rows(__global storage_t* y,
                            __global const storage_t* bias,
                            const int rows,
                            const int cols) {
  const int gid = get_global_id(0);
  const int total = rows * cols;
  if (gid >= total) return;
  const int col = gid % cols;
  const float v = LOAD(y, gid) + LOAD(bias, col);
  STORE(y, gid, v);
}
