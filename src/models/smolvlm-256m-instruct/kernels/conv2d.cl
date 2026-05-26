// Reference: model_info/transformers_src/modeling_idefics3.py:47-88 Idefics3VisionEmbeddings.__init__ (patch_embedding Conv2d)
// Reference: model_info/transformers_src/modeling_idefics3.py:90-167 Idefics3VisionEmbeddings.forward (patch_embedding forward)

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
conv2d.cl

This model's Conv2d usage is the standard PyTorch nn.Conv2d forward:

  def forward(self, input):
      return self._conv_forward(input, self.weight, self.bias)

(see torch/nn/modules/conv.py:313-460 in the reference environment).

In the Idefics3 vision embeddings, patch_embedding is an nn.Conv2d with
kernel_size=stride=patch_size and padding="valid" (model_info/transformers_src/modeling_idefics3.py:121-127).

The OpenCL backend for this project dispatches Conv2d via existing host-side
implementations (e.g., im2col + GEMM/CLBlast or a generic conv path) in
src/ops/conv2d.cpp, so no custom OpenCL kernels are required for this class.

This file intentionally contains only the required dtype-templating preamble
so it compiles cleanly even when no kernels are needed.
*/
