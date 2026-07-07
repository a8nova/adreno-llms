// MoonshineDecoderLayer.cpp — shared implementation for all 6 MoonshineDecoderLayer nodes.
// Reference: model_info/transformers_src/modeling_moonshine.py MoonshineDecoderLayer.forward
//   x = x + self_attn(input_layernorm(x))            (causal, RoPE, KV-cached)
//   x = x + encoder_attn(post_attention_layernorm(x)) (cross, cached enc K/V)
//   x = x + mlp(final_layernorm(x))                  (gated SiLU)
// weight_prefix = "model.decoder.layers.<i>"; seq_len = t_new (incremental
// rows, OPT-1); start_pos = past (absolute position of row 0). The KV state
// itself lives in MoonshineAttention.cpp, selected by layer_idx.

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
extern "C" cl_mem MoonshineDecoderMLP_forward(OpenCLContext&, Weights&, cl_command_queue,
    cl_mem, int, int, int, cl_mem*, cl_mem*, cl_mem, const char*);

extern "C" cl_mem MoonshineDecoderLayer_forward(
    OpenCLContext& cl_ctx,
    Weights& w,
    cl_command_queue q,
    cl_mem input,          // [t_new, HID] — NOT consumed (caller keeps ownership)
    int seq_len,           // t_new
    int layer_idx,
    int start_pos,         // past
    cl_mem* k_cache_inout,
    cl_mem* v_cache_inout,
    cl_mem encoder_hidden_states,
    const char* weight_prefix)
{
    (void)k_cache_inout; (void)v_cache_inout; (void)encoder_hidden_states;
    const std::string base = weight_prefix && *weight_prefix
        ? std::string(weight_prefix)
        : "model.decoder.layers." + std::to_string(layer_idx);
    const int t_new = seq_len;
    const int past = start_pos;
    const int HID = MODEL_CONFIG::HIDDEN_SIZE;

    // self-attn (causal + rope) over the persistent KV cache
    cl_mem ln1 = LayerNorm_forward(cl_ctx,w,q,input,t_new,-1,0,nullptr,nullptr,nullptr,(base+".input_layernorm").c_str());
    if(!ln1){return nullptr;}
    cl_mem sa = MoonshineAttention_forward(cl_ctx,w,q,ln1,t_new,layer_idx,past,
                                           nullptr,nullptr,nullptr,(base+".self_attn").c_str());
    clReleaseMemObject(ln1);
    if(!sa){return nullptr;}
    cl_mem x2 = moonshine_resid_add(cl_ctx,q,input,sa,t_new*HID);
    clReleaseMemObject(sa);
    if(!x2){return nullptr;}

    // cross-attn: per-waveform cached encoder K/V (no rope, no causal)
    cl_mem ln2 = LayerNorm_forward(cl_ctx,w,q,x2,t_new,-1,0,nullptr,nullptr,nullptr,(base+".post_attention_layernorm").c_str());
    if(!ln2){clReleaseMemObject(x2);return nullptr;}
    cl_mem ca = MoonshineAttention_forward(cl_ctx,w,q,ln2,t_new,layer_idx,past,
                                           nullptr,nullptr,nullptr,(base+".encoder_attn").c_str());
    clReleaseMemObject(ln2);
    if(!ca){clReleaseMemObject(x2);return nullptr;}
    cl_mem x3 = moonshine_resid_add(cl_ctx,q,x2,ca,t_new*HID);
    clReleaseMemObject(x2); clReleaseMemObject(ca);
    if(!x3){return nullptr;}

    // mlp (gated silu)
    cl_mem ln3 = LayerNorm_forward(cl_ctx,w,q,x3,t_new,-1,0,nullptr,nullptr,nullptr,(base+".final_layernorm").c_str());
    if(!ln3){clReleaseMemObject(x3);return nullptr;}
    cl_mem mlp = MoonshineDecoderMLP_forward(cl_ctx,w,q,ln3,t_new,layer_idx,0,
                                             nullptr,nullptr,nullptr,(base+".mlp").c_str());
    clReleaseMemObject(ln3);
    if(!mlp){clReleaseMemObject(x3);return nullptr;}
    cl_mem x4 = moonshine_resid_add(cl_ctx,q,x3,mlp,t_new*HID);
    clReleaseMemObject(x3); clReleaseMemObject(mlp);
    if(!x4){return nullptr;}
    return x4;
}
