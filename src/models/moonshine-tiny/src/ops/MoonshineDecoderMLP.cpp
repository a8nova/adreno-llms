// MoonshineDecoderMLP.cpp — shared implementation for all 6 MoonshineDecoderMLP nodes.
// Reference: model_info/transformers_src/modeling_moonshine.py MoonshineDecoderMLP.forward
//   fc1 → chunk(2, dim=-1) → silu(gate) * value → fc2. fc1 out = 2*1152 = 2304.
// weight_prefix = "model.decoder.layers.<i>.mlp"; seq_len = rows (t_new).

#include "../opencl_context.h"
#include "../weights.h"
#include "../nnopt_error.h"
#include "../model_config.h"
#include "../utils.h"
#include "../profiler.h"
#include <string>

cl_kernel moonshine_kernel(OpenCLContext& cl_ctx, const char* name);
cl_program moonshine_utils_program(OpenCLContext& cl_ctx);
extern "C" cl_mem Linear_forward(OpenCLContext&, Weights&, cl_command_queue,
    cl_mem, int, int, int, cl_mem*, cl_mem*, cl_mem, const char*);

extern "C" cl_mem MoonshineDecoderMLP_forward(
    OpenCLContext& cl_ctx,
    Weights& w,
    cl_command_queue q,
    cl_mem input,
    int seq_len,
    int layer_idx,
    int start_pos,
    cl_mem* k_cache_inout,
    cl_mem* v_cache_inout,
    cl_mem encoder_hidden_states,
    const char* weight_prefix)
{
    (void)layer_idx; (void)start_pos;
    (void)k_cache_inout; (void)v_cache_inout; (void)encoder_hidden_states;
    const std::string wp = weight_prefix ? std::string(weight_prefix) : std::string();
    if (wp.empty()) { NNOPT_ERROR("MoonshineDecoderMLP: empty weight_prefix"); return nullptr; }
    const int seq = seq_len;
    const int inter = MODEL_CONFIG::INTERMEDIATE_SIZE; // 1152
    cl_mem h1=nullptr,value=nullptr,gate=nullptr,gated=nullptr,out=nullptr;
    auto cleanup=[&]()->cl_mem{
        for(cl_mem* p:{&h1,&value,&gate,&gated}) if(*p){clReleaseMemObject(*p);*p=nullptr;}
        return nullptr;
    };
    h1 = Linear_forward(cl_ctx,w,q,input,seq,-1,0,nullptr,nullptr,nullptr,(wp+".fc1").c_str());
    if(!h1){NNOPT_ERROR("dec mlp fc1");return nullptr;}
    // chunk(2, dim=-1): hidden_states = first half (value), gate = second half.
    cl_int err=CL_SUCCESS;
    value = clCreateBuffer(cl_ctx.context(),CL_MEM_READ_WRITE,(size_t)seq*inter*sizeof(nnopt_storage_t),nullptr,&err);
    gate  = clCreateBuffer(cl_ctx.context(),CL_MEM_READ_WRITE,(size_t)seq*inter*sizeof(nnopt_storage_t),nullptr,&err);
    if(!value||!gate){NNOPT_ERROR("dec mlp chunk alloc");return cleanup();}
    if(!split_last_dim_2(q,moonshine_utils_program(cl_ctx),h1,value,gate,seq,inter)){NNOPT_ERROR("dec mlp split");return cleanup();}
    // gated = silu(gate) * value  (glu_silu kernel: value, gate → out)
    gated = clCreateBuffer(cl_ctx.context(),CL_MEM_READ_WRITE,(size_t)seq*inter*sizeof(nnopt_storage_t),nullptr,&err);
    if(!gated){NNOPT_ERROR("dec mlp gated alloc");return cleanup();}
    cl_kernel k = moonshine_kernel(cl_ctx,"glu_silu");
    if(!k){NNOPT_ERROR("glu_silu kernel");return cleanup();}
    int n = seq*inter;
    set_arg_checked(k,0,sizeof(cl_mem),&value,"value");
    set_arg_checked(k,1,sizeof(cl_mem),&gate,"gate");
    set_arg_checked(k,2,sizeof(cl_mem),&gated,"out");
    set_arg_checked(k,3,sizeof(int),&n,"n");
    size_t gws=(size_t)n;
    err=clEnqueueNDRangeKernel(q,k,1,nullptr,&gws,nullptr,0,nullptr,
                               KernelProfiler::event_for("glu_silu"));
    if(err!=CL_SUCCESS){NNOPT_ERROR_FMT("glu_silu dispatch %d",err);return cleanup();}
    out = Linear_forward(cl_ctx,w,q,gated,seq,-1,0,nullptr,nullptr,nullptr,(wp+".fc2").c_str());
    cleanup();
    if(!out){NNOPT_ERROR("dec mlp fc2");return nullptr;}
    return out;
}
