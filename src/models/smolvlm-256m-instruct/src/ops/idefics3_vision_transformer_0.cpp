// Reference: model_info/transformers_src/modeling_idefics3.py Idefics3VisionTransformer.forward
// Implements the inference path:
//   hidden_states = vision_model.embeddings(pixel_values, patch_attention_mask)
//   hidden_states = vision_model.encoder(hidden_states, attention_mask)
//   hidden_states = vision_model.post_layernorm(hidden_states)
// Returns last_hidden_state [B, T, C].

#include "opencl_context.h"
#include "weights.h"
#include "debug_utils.h"
#include "forward_dispatch.h"
#include "model_config.h"
#include "utils.h"

#include <CL/cl.h>
#include <cstddef>
#include <cstdint>
#include <cstdio>

extern "C" cl_mem op_Idefics3VisionTransformer(OpenCLContext& cl_ctx,
                                               Weights& weights,
                                               cl_command_queue queue,
                                               cl_mem pixel_values_nchw,
                                               cl_mem patch_attention_mask_opt,
                                               int B,
                                               int C,
                                               int H,
                                               int W,
                                               int patch_size,
                                               const char* patch_emb_w,
                                               const char* patch_emb_b,
                                               const char* pos_emb_w) {
  (void)patch_attention_mask_opt;
  if (!queue || !pixel_values_nchw) {
    NNOPT_ERROR("op_Idefics3VisionTransformer: null arg");
    return nullptr;
  }

  if (!patch_emb_w || !patch_emb_b || !pos_emb_w) {
    NNOPT_ERROR("op_Idefics3VisionTransformer: null weight key");
    return nullptr;
  }

  // NOTE: vision_pipeline_forward already produced normalized pixel_values in NCHW layout.
  // We now run patch embedding and position embedding, then reshape+add.

  // patch embedding: Conv2d with stride=patch_size producing [B, hidden, H/ps, W/ps]
  cl_mem patch = op_Conv2d(cl_ctx,
                          weights,
                          queue,
                          pixel_values_nchw,
                          /*N=*/B,
                          /*Cin=*/C,
                          /*Hin=*/H,
                          /*Win=*/W,
                          /*Cout=*/MODEL_CONFIG::VISION_CONFIG_HIDDEN_SIZE,
                          /*Kh=*/patch_size,
                          /*Kw=*/patch_size,
                          /*stride_h=*/patch_size,
                          /*stride_w=*/patch_size,
                          /*pad_h=*/0,
                          /*pad_w=*/0,
                          /*weight_key_w=*/patch_emb_w,
                          /*weight_key_b_optional=*/patch_emb_b);
  if (!patch) {
    NNOPT_ERROR("op_Idefics3VisionTransformer: patch embedding failed");
    return nullptr;
  }

  // position embedding: gather table[0..T-1] where T=(H/ps)*(W/ps)
  const int grid_h = (H / patch_size);
  const int grid_w = (W / patch_size);
  // Build token_ids = [0..T-1] (int32) on device, then gather embedding rows.
  cl_int err = CL_SUCCESS;
  const int pos_T = grid_h * grid_w;
  cl_mem pos_ids = clCreateBuffer(cl_ctx.context(), CL_MEM_READ_WRITE,
                                 (size_t)pos_T * sizeof(int32_t), nullptr, &err);
  if (err != CL_SUCCESS || !pos_ids) {
    clReleaseMemObject(patch);
    NNOPT_ERROR_FMT("op_Idefics3VisionTransformer: clCreateBuffer(pos_ids) failed (%d)", (int)err);
    return nullptr;
  }

  // PROGRAM-INIT-OK: cached static program.
  static cl_program range_prog = nullptr;
  static cl_kernel range_k = nullptr;
  static bool range_init = false;
  if (!range_init) {
    const char* src = R"CLC(
__kernel void iota_int32(__global int* out, int n) {
  int gid = (int)get_global_id(0);
  if (gid >= n) return;
  out[gid] = gid;
}
)CLC";
    range_prog = cl_ctx.build_program(std::string(src));
    if (!range_prog) {
      clReleaseMemObject(pos_ids);
      clReleaseMemObject(patch);
      NNOPT_ERROR("op_Idefics3VisionTransformer: failed to build iota_int32 kernel");
      return nullptr;
    }
    cl_int e2 = CL_SUCCESS;
    range_k = clCreateKernel(range_prog, "iota_int32", &e2);
    if (e2 != CL_SUCCESS || !range_k) {
      clReleaseMemObject(pos_ids);
      clReleaseMemObject(patch);
      NNOPT_ERROR_FMT("op_Idefics3VisionTransformer: clCreateKernel(iota_int32) failed (%d)", (int)e2);
      return nullptr;
    }
    range_init = true;
  }

  if (!set_arg_checked(range_k, 0, sizeof(cl_mem), &pos_ids, "out")) {
    clReleaseMemObject(pos_ids);
    clReleaseMemObject(patch);
    return nullptr;
  }
  if (!set_arg_checked(range_k, 1, sizeof(int), &pos_T, "n")) {
    clReleaseMemObject(pos_ids);
    clReleaseMemObject(patch);
    return nullptr;
  }
  const size_t gws_iota[1] = {(size_t)pos_T};
  err = clEnqueueNDRangeKernel(queue, range_k, 1, nullptr, gws_iota, nullptr, 0, nullptr, nullptr);
  if (err != CL_SUCCESS) {
    clReleaseMemObject(pos_ids);
    clReleaseMemObject(patch);
    NNOPT_ERROR_FMT("op_Idefics3VisionTransformer: iota_int32 dispatch failed (%d)", (int)err);
    return nullptr;
  }

  cl_mem pos = op_Embedding(cl_ctx,
                           weights,
                           queue,
                           /*input_ids_i32=*/pos_ids,
                           /*num_tokens=*/pos_T,
                           /*hidden_size=*/MODEL_CONFIG::VISION_CONFIG_HIDDEN_SIZE,
                           /*weight_key_wte=*/pos_emb_w);
  clReleaseMemObject(pos_ids);
  if (!pos) {
    clReleaseMemObject(patch);
    NNOPT_ERROR("op_Idefics3VisionTransformer: position embedding failed");
    return nullptr;
  }

  // reshape + add => [B, T, hidden]
  cl_mem hidden = op_Idefics3VisionEmbeddings(cl_ctx,
                                             weights,
                                             queue,
                                             patch,
                                             pos,
                                             /*batch=*/B,
                                             /*channels=*/MODEL_CONFIG::VISION_CONFIG_HIDDEN_SIZE,
                                             /*height=*/grid_h,
                                             /*width=*/grid_w);
  clReleaseMemObject(patch);
  clReleaseMemObject(pos);
  if (!hidden) {
    NNOPT_ERROR("op_Idefics3VisionTransformer: embeddings failed");
    return nullptr;
  }

  // encoder: 12 layers
  const int T = (H / patch_size) * (W / patch_size);
  const int vis_C = MODEL_CONFIG::VISION_CONFIG_HIDDEN_SIZE;
  const int heads = MODEL_CONFIG::VISION_CONFIG_NUM_ATTENTION_HEADS;
  const float ln_eps = (float)MODEL_CONFIG::LAYER_NORM_EPS;
  const int interm = MODEL_CONFIG::VISION_CONFIG_INTERMEDIATE_SIZE;

  for (int i = 0; i < MODEL_CONFIG::VISION_CONFIG_NUM_HIDDEN_LAYERS; ++i) {
    char ln1_w[160], ln1_b[160], ln2_w[160], ln2_b[160];
    char q_w[160], q_b[160], k_w[160], k_b[160], v_w[160], v_b[160], o_w[160], o_b[160];
    char fc1_w[160], fc1_b[160], fc2_w[160], fc2_b[160];

    std::snprintf(ln1_w, sizeof(ln1_w), "model.vision_model.encoder.layers.%d.layer_norm1.weight", i);
    std::snprintf(ln1_b, sizeof(ln1_b), "model.vision_model.encoder.layers.%d.layer_norm1.bias", i);
    std::snprintf(ln2_w, sizeof(ln2_w), "model.vision_model.encoder.layers.%d.layer_norm2.weight", i);
    std::snprintf(ln2_b, sizeof(ln2_b), "model.vision_model.encoder.layers.%d.layer_norm2.bias", i);

    std::snprintf(q_w, sizeof(q_w), "model.vision_model.encoder.layers.%d.self_attn.q_proj.weight", i);
    std::snprintf(q_b, sizeof(q_b), "model.vision_model.encoder.layers.%d.self_attn.q_proj.bias", i);
    std::snprintf(k_w, sizeof(k_w), "model.vision_model.encoder.layers.%d.self_attn.k_proj.weight", i);
    std::snprintf(k_b, sizeof(k_b), "model.vision_model.encoder.layers.%d.self_attn.k_proj.bias", i);
    std::snprintf(v_w, sizeof(v_w), "model.vision_model.encoder.layers.%d.self_attn.v_proj.weight", i);
    std::snprintf(v_b, sizeof(v_b), "model.vision_model.encoder.layers.%d.self_attn.v_proj.bias", i);
    std::snprintf(o_w, sizeof(o_w), "model.vision_model.encoder.layers.%d.self_attn.out_proj.weight", i);
    std::snprintf(o_b, sizeof(o_b), "model.vision_model.encoder.layers.%d.self_attn.out_proj.bias", i);

    std::snprintf(fc1_w, sizeof(fc1_w), "model.vision_model.encoder.layers.%d.mlp.fc1.weight", i);
    std::snprintf(fc1_b, sizeof(fc1_b), "model.vision_model.encoder.layers.%d.mlp.fc1.bias", i);
    std::snprintf(fc2_w, sizeof(fc2_w), "model.vision_model.encoder.layers.%d.mlp.fc2.weight", i);
    std::snprintf(fc2_b, sizeof(fc2_b), "model.vision_model.encoder.layers.%d.mlp.fc2.bias", i);

    cl_mem out = op_Idefics3EncoderLayer(cl_ctx,
                                        weights,
                                        queue,
                                        hidden,
                                        /*attention_mask_opt=*/nullptr,
                                        B,
                                        T,
                                        vis_C,
                                        heads,
                                        ln_eps,
                                        ln1_w,
                                        ln1_b,
                                        q_w,
                                        q_b,
                                        k_w,
                                        k_b,
                                        v_w,
                                        v_b,
                                        o_w,
                                        o_b,
                                        ln2_w,
                                        ln2_b,
                                        interm,
                                        fc1_w,
                                        fc1_b,
                                        fc2_w,
                                        fc2_b);
    clReleaseMemObject(hidden);
    hidden = out;
    if (!hidden) {
      NNOPT_ERROR_FMT("op_Idefics3VisionTransformer: encoder layer %d failed", i);
      return nullptr;
    }
  }

  // post layernorm
  cl_mem post = op_LayerNorm(cl_ctx,
                            weights,
                            queue,
                            hidden,
                            /*rows=*/B * T,
                            /*cols=*/vis_C,
                            /*eps=*/ln_eps,
                            "model.vision_model.post_layernorm.weight",
                            "model.vision_model.post_layernorm.bias");
  clReleaseMemObject(hidden);
  if (!post) {
    NNOPT_ERROR("op_Idefics3VisionTransformer: post_layernorm failed");
    return nullptr;
  }

  return post;
}
