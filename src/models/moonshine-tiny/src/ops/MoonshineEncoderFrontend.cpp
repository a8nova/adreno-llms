// MoonshineEncoderFrontend.cpp — encoder conv stack (mirrors WhisperEncoderFrontend.cpp).
// Reference: model_info/transformers_src/modeling_moonshine.py:520-545 MoonshineEncoder.forward
//   conv1(1→288, k127 s64, no bias) → tanh → groupnorm →
//   conv2(288→576, k7 s3, bias)     → gelu →
//   conv3(576→288, k3 s2, bias)     → gelu → permute [C,L]→[L,C].
//
// Standard signature: input = raw 16kHz waveform buffer [1, num_samples],
// seq_len = num_samples. Returns [enc_T, HID]; enc_T is recomputed by the
// caller from num_samples via moonshine_frontend_out_len() (the standard
// signature has no out-param for it).

#include "../opencl_context.h"
#include "../weights.h"
#include "../nnopt_error.h"
#include "../debug_utils.h"
#include "../model_config.h"
#include "../utils.h"
#include "../profiler.h"
#include <string>

cl_kernel moonshine_kernel(OpenCLContext& cl_ctx, const char* name);
cl_mem gelu_apply(OpenCLContext&, cl_command_queue, cl_mem, int);
extern "C" cl_mem Conv1d_forward(OpenCLContext&, Weights&, cl_command_queue,
    cl_mem, int, int, int, cl_mem*, cl_mem*, cl_mem, const char*);
extern "C" cl_mem GroupNorm_forward(OpenCLContext&, Weights&, cl_command_queue,
    cl_mem, int, int, int, cl_mem*, cl_mem*, cl_mem, const char*);

// ── conv stack config (derived from weight shapes / modeling_moonshine.py) ──
// conv1.weight [288,1,127], conv2.weight [576,288,7], conv3.weight [288,576,3].
// strides from MoonshineEncoder: 64, 3, 2. conv1 no bias; conv2/conv3 have bias.
static constexpr int HID          = MODEL_CONFIG::HIDDEN_SIZE;  // 288
static constexpr int CONV1_OUT    = HID;
static constexpr int CONV1_KERNEL = 127;
static constexpr int CONV1_STRIDE = 64;
static constexpr int CONV2_OUT    = 2 * HID;                    // 576
static constexpr int CONV2_KERNEL = 7;
static constexpr int CONV2_STRIDE = 3;
static constexpr int CONV3_OUT    = HID;
static constexpr int CONV3_KERNEL = 3;
static constexpr int CONV3_STRIDE = 2;

// Frames the conv stack produces for num_samples of input (callers size the
// encoder from this — keep in lockstep with the stack below).
extern "C" int moonshine_frontend_out_len(int num_samples) {
    int L1 = (num_samples - CONV1_KERNEL) / CONV1_STRIDE + 1;
    int L2 = (L1 - CONV2_KERNEL) / CONV2_STRIDE + 1;
    int L3 = (L2 - CONV3_KERNEL) / CONV3_STRIDE + 1;
    return L3;
}

// permute conv output [C, L] → [L, C]
static cl_mem permute_cl(OpenCLContext& cl_ctx, cl_command_queue q, cl_mem in, int C, int L) {
    cl_int err=CL_SUCCESS;
    cl_mem out = clCreateBuffer(cl_ctx.context(),CL_MEM_READ_WRITE,(size_t)C*L*sizeof(nnopt_storage_t),nullptr,&err);
    if(!out){NNOPT_ERROR("permute alloc");return nullptr;}
    cl_kernel k=moonshine_kernel(cl_ctx,"permute_cl_to_lc");
    if(!k){NNOPT_ERROR("permute kernel");clReleaseMemObject(out);return nullptr;}
    set_arg_checked(k,0,sizeof(cl_mem),&in,"in");
    set_arg_checked(k,1,sizeof(cl_mem),&out,"out");
    set_arg_checked(k,2,sizeof(int),&C,"C");
    set_arg_checked(k,3,sizeof(int),&L,"L");
    size_t gws=(size_t)C*L;
    err=clEnqueueNDRangeKernel(q,k,1,nullptr,&gws,nullptr,0,nullptr,
                               KernelProfiler::event_for("permute_cl_to_lc"));
    if(err!=CL_SUCCESS){NNOPT_ERROR_FMT("permute dispatch %d",err);clReleaseMemObject(out);return nullptr;}
    return out;
}

// tanh in-place-ish: returns new buffer tanh(x).
static cl_mem tanh_apply(OpenCLContext& cl_ctx, cl_command_queue q, cl_mem in, int n) {
    cl_int err=CL_SUCCESS;
    cl_mem out=clCreateBuffer(cl_ctx.context(),CL_MEM_READ_WRITE,(size_t)n*sizeof(nnopt_storage_t),nullptr,&err);
    if(!out){NNOPT_ERROR("tanh alloc");return nullptr;}
    cl_kernel k=moonshine_kernel(cl_ctx,"tanh_act");
    if(!k){NNOPT_ERROR("tanh kernel");clReleaseMemObject(out);return nullptr;}
    set_arg_checked(k,0,sizeof(cl_mem),&in,"in");
    set_arg_checked(k,1,sizeof(cl_mem),&out,"out");
    set_arg_checked(k,2,sizeof(int),&n,"n");
    size_t gws=(size_t)n;
    err=clEnqueueNDRangeKernel(q,k,1,nullptr,&gws,nullptr,0,nullptr,
                               KernelProfiler::event_for("tanh_act"));
    if(err!=CL_SUCCESS){NNOPT_ERROR_FMT("tanh dispatch %d",err);clReleaseMemObject(out);return nullptr;}
    return out;
}

extern "C" cl_mem MoonshineEncoderFrontend_forward(
    OpenCLContext& cl_ctx,
    Weights& w,
    cl_command_queue q,
    cl_mem input,          // raw waveform [1, num_samples]
    int seq_len,           // num_samples
    int layer_idx,
    int start_pos,
    cl_mem* k_cache_inout,
    cl_mem* v_cache_inout,
    cl_mem encoder_hidden_states,
    const char* weight_prefix)
{
    (void)layer_idx; (void)start_pos;
    (void)k_cache_inout; (void)v_cache_inout; (void)encoder_hidden_states; (void)weight_prefix;
    const int num_samples = seq_len;

    // conv1: [1, N] → [288, L1], k=127 s=64, no bias, then tanh.
    cl_mem c1 = Conv1d_forward(cl_ctx,w,q,input,num_samples,0,0,nullptr,nullptr,nullptr,"model.encoder.conv1");
    if(!c1){NNOPT_ERROR("conv1");return nullptr;}
    int L1 = (num_samples - CONV1_KERNEL)/CONV1_STRIDE + 1;
    // Reference dump: model_encoder_conv1 [1,288,L1]. tanh is applied AFTER the
    // conv module in MoonshineEncoder.forward, so the Conv1d module output is
    // the pre-tanh tensor.
    NNOPT_LAYER_CHECK("model_encoder_conv1", q, c1, (size_t)CONV1_OUT*L1);
    cl_mem c1t = tanh_apply(cl_ctx,q,c1,CONV1_OUT*L1);
    clReleaseMemObject(c1);
    if(!c1t){NNOPT_ERROR("conv1 tanh");return nullptr;}
    // groupnorm over [288, L1]
    cl_mem gn = GroupNorm_forward(cl_ctx,w,q,c1t,L1,0,0,nullptr,nullptr,nullptr,"model.encoder.groupnorm");
    clReleaseMemObject(c1t);
    if(!gn){NNOPT_ERROR("groupnorm");return nullptr;}
    NNOPT_LAYER_CHECK("model_encoder_groupnorm", q, gn, (size_t)CONV1_OUT*L1);
    // conv2: [288, L1] → [576, L2], k=7 s=3 bias, then gelu.
    cl_mem c2 = Conv1d_forward(cl_ctx,w,q,gn,L1,1,0,nullptr,nullptr,nullptr,"model.encoder.conv2");
    clReleaseMemObject(gn);
    if(!c2){NNOPT_ERROR("conv2");return nullptr;}
    int L2 = (L1 - CONV2_KERNEL)/CONV2_STRIDE + 1;
    NNOPT_LAYER_CHECK("model_encoder_conv2", q, c2, (size_t)CONV2_OUT*L2);
    cl_mem c2g = gelu_apply(cl_ctx,q,c2,CONV2_OUT*L2);
    clReleaseMemObject(c2);
    if(!c2g){NNOPT_ERROR("conv2 gelu");return nullptr;}
    // conv3: [576, L2] → [288, L3], k=3 s=2 bias, then gelu.
    cl_mem c3 = Conv1d_forward(cl_ctx,w,q,c2g,L2,2,0,nullptr,nullptr,nullptr,"model.encoder.conv3");
    clReleaseMemObject(c2g);
    if(!c3){NNOPT_ERROR("conv3");return nullptr;}
    int L3 = (L2 - CONV3_KERNEL)/CONV3_STRIDE + 1;
    NNOPT_LAYER_CHECK("model_encoder_conv3", q, c3, (size_t)CONV3_OUT*L3);
    cl_mem c3g = gelu_apply(cl_ctx,q,c3,CONV3_OUT*L3);
    clReleaseMemObject(c3);
    if(!c3g){NNOPT_ERROR("conv3 gelu");return nullptr;}
    // permute [288, L3] → [L3, 288]
    cl_mem x = permute_cl(cl_ctx,q,c3g,CONV3_OUT,L3);
    clReleaseMemObject(c3g);
    if(!x){NNOPT_ERROR("permute");return nullptr;}
    NNOPT_LAYER_CHECK("model_encoder_conv_out", q, x, (size_t)L3*HID);
    return x;
}
