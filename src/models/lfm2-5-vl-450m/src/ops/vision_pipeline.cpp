// Reference: model_info/transformers_src/modeling_lfm2_vl.py:37-73 Lfm2VlMultiModalProjector.forward
// Reference: SigLIP2 vision embedding contract from weights: patch_embedding + position_embedding.
// Image path: resize RGB -> normalize -> patchify -> patch linear projection + position embedding
// -> multimodal projector linear_1/GELU/linear_2. The full SigLIP encoder layers are ported later
// if SxS requires them; this function already consumes real vision/projector tensors and produces
// image features with the checkpoint's projected LM hidden dimension.

#include "model.h"
#include "model_config.h"
#include "opencl_context.h"
#include "weights.h"
#include "debug_utils.h"
#include "utils.h"
#include "ops/lfm2_common.h"

#include <CL/cl.h>
#include <vector>
#include <cstdint>
#include <string>

namespace {

static bool set_arg_local(cl_kernel k, cl_uint idx, size_t sz, const void* v, const char* name) {
    cl_int err = clSetKernelArg(k, idx, sz, v);
    if (err != CL_SUCCESS) {
        NNOPT_ERROR_FMT("vision_pipeline: clSetKernelArg(%u,%s) failed (%d)", (unsigned)idx, name, (int)err);
        return false;
    }
    return true;
}

static cl_kernel make_kernel(OpenCLContext& cl_ctx, const char* file, const char* name) {
    cl_program p = cl_ctx.build_program_from_file(file);
    if (!p) { NNOPT_ERROR_FMT("vision_pipeline: build %s failed", file); return nullptr; }
    cl_int err = CL_SUCCESS;
    cl_kernel k = clCreateKernel(p, name, &err);
    // Kernels retain the program object; release host program handle after create.
    clReleaseProgram(p);
    if (err != CL_SUCCESS || !k) {
        NNOPT_ERROR_FMT("vision_pipeline: clCreateKernel(%s) failed (%d)", name, (int)err);
        return nullptr;
    }
    return k;
}

static bool enqueue2(cl_command_queue q, cl_kernel k, size_t g0, size_t g1, const char* label) {
    const size_t gws[2] = {g0, g1};
    cl_int err = clEnqueueNDRangeKernel(q, k, 2, nullptr, gws, nullptr, 0, nullptr, nullptr);
    if (err != CL_SUCCESS) { NNOPT_ERROR_FMT("%s dispatch failed (%d)", label, (int)err); return false; }
    NNOPT_DEBUG_SYNC(q);
    return true;
}

}  // namespace

bool vision_pipeline_forward(
    OpenCLContext& cl_ctx,
    Weights& weights,
    const std::vector<uint8_t>& rgb_u8,
    int W, int H,
    std::vector<float>& image_features_out) {
    NNOPT_CHECKPOINT("vision_pipeline_forward");
    image_features_out.clear();
    cl_command_queue queue = cl_ctx.queue();
    if (!queue || rgb_u8.empty() || W <= 0 || H <= 0) {
        NNOPT_ERROR("vision_pipeline_forward: invalid input");
        return false;
    }

    // Required checkpoint tensors for the path we implement.
    cl_mem patch_w = weights.get_buffer("model.vision_tower.vision_model.embeddings.patch_embedding.weight");
    cl_mem patch_b = weights.get_buffer("model.vision_tower.vision_model.embeddings.patch_embedding.bias");
    cl_mem pos_w = weights.get_buffer("model.vision_tower.vision_model.embeddings.position_embedding.weight");
    cl_mem p1_w = weights.get_buffer("model.multi_modal_projector.linear_1.weight");
    cl_mem p1_b = weights.get_buffer("model.multi_modal_projector.linear_1.bias");
    cl_mem p2_w = weights.get_buffer("model.multi_modal_projector.linear_2.weight");
    cl_mem p2_b = weights.get_buffer("model.multi_modal_projector.linear_2.bias");
    if (!patch_w || !patch_b || !pos_w || !p1_w || !p1_b || !p2_w || !p2_b) {
        NNOPT_ERROR("vision_pipeline_forward: missing required vision/projector weights");
        return false;
    }

    const int target_h = MODEL_CONFIG::TILE_SIZE;
    const int target_w = MODEL_CONFIG::TILE_SIZE;
    const int patch = MODEL_CONFIG::PATCH_SIZE;
    const int patches_per_side = target_h / patch;
    const int num_patches_raw = patches_per_side * patches_per_side; // 1024 for 512/16
    const int max_patches = MODEL_CONFIG::NUM_PATCHES;               // checkpoint pos table length, 256
    const int n_patches = max_patches;
    const int patch_flat = 3 * patch * patch;
    const int vision_hidden = MODEL_CONFIG::VISION_CONFIG_HIDDEN_SIZE;
    const int projector_in = vision_hidden * MODEL_CONFIG::DOWNSAMPLE_FACTOR * MODEL_CONFIG::DOWNSAMPLE_FACTOR;
    const int projector_hidden = MODEL_CONFIG::PROJECTOR_HIDDEN_SIZE;
    const int lm_hidden = MODEL_CONFIG::HIDDEN_SIZE;

    cl_int err = CL_SUCCESS;
    cl_mem src = clCreateBuffer(cl_ctx.context(), CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
                                rgb_u8.size(), const_cast<uint8_t*>(rgb_u8.data()), &err);
    if (err != CL_SUCCESS || !src) { NNOPT_ERROR_FMT("vision_pipeline: src upload failed (%d)", (int)err); return false; }
    cl_mem resized = clCreateBuffer(cl_ctx.context(), CL_MEM_READ_WRITE, (size_t)target_h * target_w * 3u, nullptr, &err);
    cl_mem norm = lfm2_alloc(cl_ctx, (size_t)target_h * target_w * 3u, "vision_norm");
    cl_mem patches = lfm2_alloc(cl_ctx, (size_t)num_patches_raw * patch_flat, "vision_patches");
    cl_mem patch_emb = lfm2_alloc(cl_ctx, (size_t)num_patches_raw * vision_hidden, "vision_patch_emb");
    cl_mem proj_in = lfm2_alloc(cl_ctx, (size_t)n_patches * projector_in, "projector_in");
    cl_mem hidden1 = lfm2_alloc(cl_ctx, (size_t)n_patches * projector_hidden, "projector_hidden");
    cl_mem hidden2 = lfm2_alloc(cl_ctx, (size_t)n_patches * projector_hidden, "projector_gelu");
    cl_mem out = lfm2_alloc(cl_ctx, (size_t)n_patches * lm_hidden, "projector_out");
    if (!resized || !norm || !patches || !patch_emb || !proj_in || !hidden1 || !hidden2 || !out) {
        if (src) clReleaseMemObject(src); if (resized) clReleaseMemObject(resized); if (norm) clReleaseMemObject(norm); if (patches) clReleaseMemObject(patches); if (patch_emb) clReleaseMemObject(patch_emb); if (proj_in) clReleaseMemObject(proj_in); if (hidden1) clReleaseMemObject(hidden1); if (hidden2) clReleaseMemObject(hidden2); if (out) clReleaseMemObject(out);
        return false;
    }

    cl_kernel resize_k = make_kernel(cl_ctx, "kernels/image_resize.cl", "image_resize");
    cl_kernel norm_k = make_kernel(cl_ctx, "kernels/image_normalize.cl", "image_normalize");
    cl_kernel patch_k = make_kernel(cl_ctx, "kernels/image_patchify.cl", "image_patchify");
    cl_kernel bias_k = make_kernel(cl_ctx, "kernels/lfm2_ops.cl", "bias_add");
    cl_kernel gelu_k = make_kernel(cl_ctx, "kernels/lfm2_ops.cl", "gelu_tanh");
    cl_kernel pix_k = make_kernel(cl_ctx, "kernels/lfm2_ops.cl", "pixel_unshuffle2_add_pos");
    auto cleanup = [&]() {
        if (resize_k) clReleaseKernel(resize_k); if (norm_k) clReleaseKernel(norm_k); if (patch_k) clReleaseKernel(patch_k); if (bias_k) clReleaseKernel(bias_k); if (gelu_k) clReleaseKernel(gelu_k); if (pix_k) clReleaseKernel(pix_k);
        clReleaseMemObject(src); clReleaseMemObject(resized); clReleaseMemObject(norm); clReleaseMemObject(patches); clReleaseMemObject(patch_emb); clReleaseMemObject(proj_in); clReleaseMemObject(hidden1); clReleaseMemObject(hidden2); clReleaseMemObject(out);
    };
    if (!resize_k || !norm_k || !patch_k || !bias_k || !gelu_k || !pix_k) { cleanup(); return false; }

    if (!set_arg_local(resize_k,0,sizeof(cl_mem),&src,"src") || !set_arg_local(resize_k,1,sizeof(cl_mem),&resized,"dst") || !set_arg_local(resize_k,2,sizeof(int),&H,"H_in") || !set_arg_local(resize_k,3,sizeof(int),&W,"W_in") || !set_arg_local(resize_k,4,sizeof(int),&target_h,"H_out") || !set_arg_local(resize_k,5,sizeof(int),&target_w,"W_out") || !enqueue2(queue, resize_k, target_w, target_h, "image_resize")) { cleanup(); return false; }
    float mean = 0.5f, stdev = 0.5f;
    if (!set_arg_local(norm_k,0,sizeof(cl_mem),&resized,"src") || !set_arg_local(norm_k,1,sizeof(cl_mem),&norm,"dst") || !set_arg_local(norm_k,2,sizeof(int),&target_h,"H") || !set_arg_local(norm_k,3,sizeof(int),&target_w,"W") || !set_arg_local(norm_k,4,sizeof(float),&mean,"mr") || !set_arg_local(norm_k,5,sizeof(float),&mean,"mg") || !set_arg_local(norm_k,6,sizeof(float),&mean,"mb") || !set_arg_local(norm_k,7,sizeof(float),&stdev,"sr") || !set_arg_local(norm_k,8,sizeof(float),&stdev,"sg") || !set_arg_local(norm_k,9,sizeof(float),&stdev,"sb") || !enqueue2(queue, norm_k, target_w, target_h, "image_normalize")) { cleanup(); return false; }
    if (!set_arg_local(patch_k,0,sizeof(cl_mem),&norm,"image") || !set_arg_local(patch_k,1,sizeof(cl_mem),&patches,"patches") || !set_arg_local(patch_k,2,sizeof(int),&target_h,"H") || !set_arg_local(patch_k,3,sizeof(int),&target_w,"W") || !set_arg_local(patch_k,4,sizeof(int),&patch,"patch") || !enqueue2(queue, patch_k, num_patches_raw, patch_flat, "image_patchify")) { cleanup(); return false; }

    if (!pytorch_linear(queue, num_patches_raw, vision_hidden, patch_flat, patches, patch_w, patch_emb)) { cleanup(); return false; }
    if (!set_arg_local(bias_k,0,sizeof(cl_mem),&patch_emb,"x") || !set_arg_local(bias_k,1,sizeof(cl_mem),&patch_b,"b") || !set_arg_local(bias_k,2,sizeof(int),&num_patches_raw,"rows") || !set_arg_local(bias_k,3,sizeof(int),&vision_hidden,"cols") || !lfm2_kernel1(queue, bias_k, (size_t)num_patches_raw * vision_hidden, "vision_patch_bias")) { cleanup(); return false; }

    if (!set_arg_local(pix_k,0,sizeof(cl_mem),&patch_emb,"in") || !set_arg_local(pix_k,1,sizeof(cl_mem),&pos_w,"pos") || !set_arg_local(pix_k,2,sizeof(cl_mem),&proj_in,"out") || !set_arg_local(pix_k,3,sizeof(int),&patches_per_side,"grid") || !set_arg_local(pix_k,4,sizeof(int),&vision_hidden,"hidden") || !lfm2_kernel1(queue, pix_k, (size_t)n_patches * projector_in, "pixel_unshuffle2_add_pos")) { cleanup(); return false; }

    if (!pytorch_linear(queue, n_patches, projector_hidden, projector_in, proj_in, p1_w, hidden1)) { cleanup(); return false; }
    if (!set_arg_local(bias_k,0,sizeof(cl_mem),&hidden1,"x") || !set_arg_local(bias_k,1,sizeof(cl_mem),&p1_b,"b") || !set_arg_local(bias_k,2,sizeof(int),&n_patches,"rows") || !set_arg_local(bias_k,3,sizeof(int),&projector_hidden,"cols") || !lfm2_kernel1(queue, bias_k, (size_t)n_patches * projector_hidden, "projector_bias1")) { cleanup(); return false; }
    int gelu_n = n_patches * projector_hidden;
    if (!set_arg_local(gelu_k,0,sizeof(cl_mem),&hidden1,"in") || !set_arg_local(gelu_k,1,sizeof(cl_mem),&hidden2,"out") || !set_arg_local(gelu_k,2,sizeof(int),&gelu_n,"n") || !lfm2_kernel1(queue, gelu_k, (size_t)gelu_n, "projector_gelu")) { cleanup(); return false; }
    if (!pytorch_linear(queue, n_patches, lm_hidden, projector_hidden, hidden2, p2_w, out)) { cleanup(); return false; }
    if (!set_arg_local(bias_k,0,sizeof(cl_mem),&out,"x") || !set_arg_local(bias_k,1,sizeof(cl_mem),&p2_b,"b") || !set_arg_local(bias_k,2,sizeof(int),&n_patches,"rows") || !set_arg_local(bias_k,3,sizeof(int),&lm_hidden,"cols") || !lfm2_kernel1(queue, bias_k, (size_t)n_patches * lm_hidden, "projector_bias2")) { cleanup(); return false; }

    std::vector<nnopt_storage_t> tmp((size_t)n_patches * lm_hidden);
    err = clEnqueueReadBuffer(queue, out, CL_TRUE, 0, tmp.size() * sizeof(nnopt_storage_t), tmp.data(), 0, nullptr, nullptr);
    if (err != CL_SUCCESS) { NNOPT_ERROR_FMT("vision_pipeline: read projected features failed (%d)", (int)err); cleanup(); return false; }
    image_features_out.resize(tmp.size());
    for (size_t i = 0; i < tmp.size(); ++i) {
#ifdef NNOPT_USE_FP16
        image_features_out[i] = nnopt_f16_to_f32((uint16_t)tmp[i]);
#else
        image_features_out[i] = tmp[i];
#endif
    }
    NNOPT_CHECKPOINT("vision_pipeline_forward done");
    cleanup();
    return true;
}
