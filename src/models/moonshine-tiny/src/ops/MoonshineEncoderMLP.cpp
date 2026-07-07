// MoonshineEncoderMLP.cpp — shared implementation for all 6 MoonshineEncoderMLP nodes.
// Reference: model_info/transformers_src/modeling_moonshine.py MoonshineEncoderMLP.forward
//   fc2(gelu(fc1(x))), intermediate = 1152.
// weight_prefix = "model.encoder.layers.<i>.mlp"; seq_len = rows.

#include "../opencl_context.h"
#include "../weights.h"
#include "../nnopt_error.h"
#include "../model_config.h"
#include "../utils.h"
#include <string>

cl_mem gelu_apply(OpenCLContext&, cl_command_queue, cl_mem, int);
extern "C" cl_mem Linear_forward(OpenCLContext&, Weights&, cl_command_queue,
    cl_mem, int, int, int, cl_mem*, cl_mem*, cl_mem, const char*);

extern "C" cl_mem MoonshineEncoderMLP_forward(
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
    if (wp.empty()) { NNOPT_ERROR("MoonshineEncoderMLP: empty weight_prefix"); return nullptr; }
    const int seq = seq_len;
    cl_mem h1=nullptr,g=nullptr,out=nullptr;
    h1 = Linear_forward(cl_ctx,w,q,input,seq,-1,0,nullptr,nullptr,nullptr,(wp+".fc1").c_str());
    if(!h1){NNOPT_ERROR("enc mlp fc1");return nullptr;}
    g = gelu_apply(cl_ctx,q,h1,seq*MODEL_CONFIG::INTERMEDIATE_SIZE);
    clReleaseMemObject(h1);
    if(!g){NNOPT_ERROR("enc mlp gelu");return nullptr;}
    out = Linear_forward(cl_ctx,w,q,g,seq,-1,0,nullptr,nullptr,nullptr,(wp+".fc2").c_str());
    clReleaseMemObject(g);
    if(!out){NNOPT_ERROR("enc mlp fc2");return nullptr;}
    return out;
}
