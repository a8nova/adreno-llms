// MoonshineEncoderLayer.cpp — shared implementation for all 6 MoonshineEncoderLayer nodes.
// Reference: model_info/transformers_src/modeling_moonshine.py MoonshineEncoderLayer.forward
//   x = x + self_attn(input_layernorm(x))
//   x = x + mlp(post_attention_layernorm(x))
// weight_prefix = "model.encoder.layers.<i>"; seq_len = T (encoder frames).

#include "../opencl_context.h"
#include "../weights.h"
#include "../nnopt_error.h"
#include "../model_config.h"
#include "../utils.h"
#include <string>

cl_mem moonshine_resid_add(OpenCLContext&, cl_command_queue, cl_mem, cl_mem, int);
extern "C" cl_mem LayerNorm_forward(OpenCLContext&, Weights&, cl_command_queue,
    cl_mem, int, int, int, cl_mem*, cl_mem*, cl_mem, const char*);
extern "C" cl_mem MoonshineAttention_forward(OpenCLContext&, Weights&, cl_command_queue,
    cl_mem, int, int, int, cl_mem*, cl_mem*, cl_mem, const char*);
extern "C" cl_mem MoonshineEncoderMLP_forward(OpenCLContext&, Weights&, cl_command_queue,
    cl_mem, int, int, int, cl_mem*, cl_mem*, cl_mem, const char*);

extern "C" cl_mem MoonshineEncoderLayer_forward(
    OpenCLContext& cl_ctx,
    Weights& w,
    cl_command_queue q,
    cl_mem input,          // [T, HID] — NOT consumed (caller keeps ownership)
    int seq_len,           // T
    int layer_idx,
    int start_pos,
    cl_mem* k_cache_inout,
    cl_mem* v_cache_inout,
    cl_mem encoder_hidden_states,
    const char* weight_prefix)
{
    (void)start_pos; (void)k_cache_inout; (void)v_cache_inout; (void)encoder_hidden_states;
    const std::string base = weight_prefix && *weight_prefix
        ? std::string(weight_prefix)
        : "model.encoder.layers." + std::to_string(layer_idx);
    const int T = seq_len;
    const int HID = MODEL_CONFIG::HIDDEN_SIZE;

    // self-attn block: residual + attn(input_layernorm(x))
    cl_mem ln1 = LayerNorm_forward(cl_ctx,w,q,input,T,-1,0,nullptr,nullptr,nullptr,(base+".input_layernorm").c_str());
    if(!ln1){NNOPT_ERROR_FMT("enc ln1 %d",layer_idx);return nullptr;}
    cl_mem attn = MoonshineAttention_forward(cl_ctx,w,q,ln1,T,/*layer_idx=*/-1,0,
                                             nullptr,nullptr,nullptr,(base+".self_attn").c_str());
    clReleaseMemObject(ln1);
    if(!attn){return nullptr;}
    cl_mem x2 = moonshine_resid_add(cl_ctx,q,input,attn,T*HID);
    clReleaseMemObject(attn);
    if(!x2){NNOPT_ERROR_FMT("enc resid1 %d",layer_idx);return nullptr;}

    // mlp block: residual + mlp(post_attention_layernorm(x))
    cl_mem ln2 = LayerNorm_forward(cl_ctx,w,q,x2,T,-1,0,nullptr,nullptr,nullptr,(base+".post_attention_layernorm").c_str());
    if(!ln2){NNOPT_ERROR_FMT("enc ln2 %d",layer_idx);clReleaseMemObject(x2);return nullptr;}
    cl_mem mlp = MoonshineEncoderMLP_forward(cl_ctx,w,q,ln2,T,layer_idx,0,
                                             nullptr,nullptr,nullptr,(base+".mlp").c_str());
    clReleaseMemObject(ln2);
    if(!mlp){clReleaseMemObject(x2);return nullptr;}
    cl_mem x3 = moonshine_resid_add(cl_ctx,q,x2,mlp,T*HID);
    clReleaseMemObject(x2); clReleaseMemObject(mlp);
    if(!x3){NNOPT_ERROR_FMT("enc resid2 %d",layer_idx);return nullptr;}
    return x3;
}
