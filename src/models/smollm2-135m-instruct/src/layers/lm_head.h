#pragma once
// Reference: model_info/transformers_src/modeling_llama.py:445-501 LlamaForCausalLM.forward
// lm_head is a simple linear projection from hidden_size -> vocab.

#include <CL/cl.h>
#include <vector>

class OpenCLContext;
class Weights;

class LmHead {
public:
    LmHead(OpenCLContext& cl_ctx, Weights& weights);
    ~LmHead();

    bool initialize();

    // hidden: [M, hidden]
    // returns: [M, vocab] (caller owns)
    cl_mem forward(cl_command_queue queue, cl_mem hidden, int M);

private:
    bool set_weights();

    OpenCLContext& cl_ctx_;
    Weights& weights_;

    cl_mem w_ = nullptr; // aliased from embed_tokens.weight if tied

    // Decode fast-path GEMV (M=1) — dispatched automatically from forward() when M==1.
    cl_program block_fused_prog_ = nullptr;
    cl_kernel fused_lm_head_m1_ = nullptr;

    // Image2d_t-backed no4 kernel + tiled views over W.
    // Tiling required because VOCAB=49152 typically exceeds Adreno's
    // CL_DEVICE_IMAGE2D_MAX_HEIGHT (16384). Each tile owns a sub-buffer
    // over W and an image2d created from that sub-buffer.
    cl_kernel gemv_k576_no4_img_ = nullptr;
    struct WImageTile {
        cl_mem sub_buffer = nullptr;
        cl_mem image      = nullptr;
        int    row_offset = 0;
        int    row_count  = 0;
        cl_mem out_sub    = nullptr;  // pre-created sub-buffer of decode_logits_buf_
    };
    std::vector<WImageTile> w_tiles_;
    cl_mem  w_image_single_ = nullptr;  // single-image fast path (if W fits)
    bool    img_path_ready_ = false;
};
