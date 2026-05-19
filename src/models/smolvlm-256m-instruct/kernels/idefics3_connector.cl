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

/*
Idefics3Connector (transformers/models/idefics3/modeling_idefics3.py)

Forward:
  image_hidden_states = pixel_shuffle(image_hidden_states, scale_factor)
  image_hidden_states = modality_projection(image_hidden_states)
  return image_hidden_states

The pixel_shuffle is purely a reshape/view/permute/reshape sequence (no arithmetic).
The modality_projection is an MLP implemented with Linear layers (matmuls) and
activations handled by existing ops / CLBlast dispatch.

Therefore, no custom OpenCL kernels are required specifically for this class.
This file exists to satisfy the build system's expectation of a kernel source.
*/

__kernel void idefics3_connector_noop(
    __global const storage_t* x,
    __global storage_t* out,
    int n_elements)
{
    // A real, compilable kernel body: simple copy.
    // Not used by the intended execution path, but safe if invoked.
    int gid = (int)get_global_id(0);
    if (gid < n_elements) {
        storage_t v = (storage_t)LOAD(x, gid);
        STORE(out, gid, v);
    }
}
