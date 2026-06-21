// Reference: docs/MODALITY_TTS.md: Length regulation — the one host-device round-trip
//
// length_regulator — gather text-encoder stats by per-character duration.
//
// VITS: after the duration predictor emits durations[T_chars], each frame
// of the output is taken from the encoder stats[char_idx] where char_idx
// is chosen such that cumsum(durations) first exceeds the frame index.
//
// We avoid recomputing the cumsum on device by precomputing char_idx[]
// on the HOST and uploading it as a small int32 buffer. This keeps the
// kernel a pure gather: out[i*C + c] = src[char_idx[i]*C + c].
//
// Activation: dormant unless NNOPT_PROFILE=1 (event attached for profiling).
//
// Inputs:
//   src       — [T_chars, C] storage_t (fp16/fp32, matches build dtype)
//   char_idx  — [T_frames]   int32, precomputed on host
//   T_frames  — int32, runtime-determined
//   C         — int32, channel width (e.g. 384 for VITS stats; could be 192 too)
// Output:
//   out       — [T_frames, C] storage_t
//
// One work-item per output element. Global work size = T_frames * C.

// Dtype-template preamble (LOAD/STORE) is REQUIRED for fp16 correctness on Adreno.
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

__kernel void length_regulator(
    __global const storage_t* src,
    __global const int*       char_idx,
    __global       storage_t* out,
    const int T_frames,
    const int C) {

    const int gid = get_global_id(0);
    const int total = T_frames * C;
    if (gid >= total) return;

    const int frame = gid / C;
    const int c     = gid % C;
    const int src_char = char_idx[frame];

    const float v = LOAD(src, src_char * C + c);
    STORE(out, gid, v);
}
