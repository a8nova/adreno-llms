// Reference: .nnport/onnx_graph_spec.json text_encoder_1.* (Shape/Gather/
// TopK/Where/ScatterND pack_padded_sequence metadata subgraph).
// Emits the boundary shape tensor text_encoder_1.Where_24_output_0 = [1, C, T]
// (batch=1, channels=C, seq=T). This whole ONNX subgraph is pure shape/mask/
// length-sort control flow that computes the [batch, channels, seq] reshape
// target for the downstream DurationEncoder; it carries NO neural weights.
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

// Write the 3-element shape vector [batch, channels, seq] into out.
__kernel void te1_shape_vec(__global storage_t* out,
                            const int batch,
                            const int channels,
                            const int seq)
{
    int i = get_global_id(0);
    if (i >= 3) return;
    float v = (i == 0) ? (float)batch : (i == 1) ? (float)channels : (float)seq;
    STORE(out, i, v);
}
