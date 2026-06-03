// Reference: transformers/models/lfm2/modeling_lfm2.py Lfm2DecoderLayer.forward + Lfm2ShortConv.slow_forward
// Lfm2DecoderLayer.forward:
//   residual = hidden_states
//   hidden_states = self.conv(hidden_states=self.operator_norm(hidden_states), ...)
//   hidden_states = hidden_states + residual
//   hidden_states = hidden_states + self.feed_forward(self.ffn_norm(hidden_states))
// Lfm2ShortConv.slow_forward:
//   BCx = self.in_proj(x).transpose(-1, -2); B, C, x = BCx.chunk(3, dim=-2)
//   Bx = B * x; conv_out = self.conv(Bx)[..., :seqlen]
//   y = C * conv_out; y = self.out_proj(y.transpose(-1, -2).contiguous())

#include "opencl_context.h"
#include "weights.h"
#include "debug_utils.h"
#include "model_config.h"
#include "forward_dispatch.h"
#include "model.h"
#include "ops/lfm2_common.h"
#include "utils.h"
#include <CL/cl.h>
#include <string>

extern Model* g_active_model_for_vlm_splice;

namespace {
static bool set_arg_local(cl_kernel k, cl_uint idx, size_t sz, const void* v, const char* name) {
    cl_int err = clSetKernelArg(k, idx, sz, v);
    if (err != CL_SUCCESS) { NNOPT_ERROR_FMT("lfm2_conv: clSetKernelArg(%u,%s) failed (%d)", (unsigned)idx, name, (int)err); return false; }
    return true;
}
static cl_kernel kernel(OpenCLContext& cl_ctx, const char* name) {
    cl_program p = lfm2_program(cl_ctx); if (!p) return nullptr;
    cl_int err = CL_SUCCESS; cl_kernel k = clCreateKernel(p, name, &err);
    if (err != CL_SUCCESS || !k) { NNOPT_ERROR_FMT("lfm2_conv: clCreateKernel(%s) failed (%d)", name, (int)err); return nullptr; }
    return k;
}
}

extern "C" cl_mem op_lfm2_conv_block(OpenCLContext& cl_ctx,
                                      Weights& weights,
                                      cl_command_queue queue,
                                      cl_mem hidden_in,
                                      int seq_len,
                                      int hidden_size,
                                      int layer_idx,
                                      int start_pos) {
    const std::string prefix = "model.language_model.layers." + std::to_string(layer_idx);
    cl_mem norm1 = lfm2_rms_norm(cl_ctx, weights, queue, hidden_in, seq_len, hidden_size, prefix + ".operator_norm.weight");
    if (!norm1) return nullptr;
    if (layer_idx == 0 && start_pos == 0) NNOPT_LAYER_CHECK("block0_sub_operator_norm", queue, norm1, (size_t)seq_len * (size_t)hidden_size);

    cl_mem in_w = weights.get_buffer(prefix + ".conv.in_proj.weight");
    cl_mem conv_w = weights.get_buffer(prefix + ".conv.conv.weight");
    cl_mem out_w = weights.get_buffer(prefix + ".conv.out_proj.weight");
    if (!in_w || !conv_w || !out_w) { clReleaseMemObject(norm1); NNOPT_ERROR_FMT("lfm2_conv: missing conv weights for layer %d", layer_idx); return nullptr; }

    // Reference: Lfm2ShortConv.slow_forward chunks in_proj(x) into B, C, x along dim=-2
    // after transpose. The weight is [3H,H] with three contiguous row slices; use three
    // H-wide GEMMs instead of one N=3H CLBlast call. This preserves PyTorch chunk
    // semantics and avoids Adreno/CLBlast instability on the very-wide 3072-column HGEMM.
    cl_mem bbuf = lfm2_alloc(cl_ctx, (size_t)seq_len * (size_t)hidden_size, "conv_B");
    cl_mem cbuf = lfm2_alloc(cl_ctx, (size_t)seq_len * (size_t)hidden_size, "conv_C");
    cl_mem xproj = lfm2_alloc(cl_ctx, (size_t)seq_len * (size_t)hidden_size, "conv_x");
    cl_mem bx = lfm2_alloc(cl_ctx, (size_t)seq_len * (size_t)hidden_size, "conv_bx");
    cl_mem conv_out = lfm2_alloc(cl_ctx, (size_t)seq_len * (size_t)hidden_size, "conv_out");
    cl_mem gated = lfm2_alloc(cl_ctx, (size_t)seq_len * (size_t)hidden_size, "conv_gated");
    cl_mem proj = lfm2_alloc(cl_ctx, (size_t)seq_len * (size_t)hidden_size, "conv_proj");
    cl_mem b_w = nullptr;
    cl_mem c_w = nullptr;
    cl_mem x_w = nullptr;
    auto cleanup = [&]() -> cl_mem {
        if (norm1) clReleaseMemObject(norm1);
        if (bbuf) clReleaseMemObject(bbuf);
        if (cbuf) clReleaseMemObject(cbuf);
        if (xproj) clReleaseMemObject(xproj);
        if (bx) clReleaseMemObject(bx);
        if (conv_out) clReleaseMemObject(conv_out);
        if (gated) clReleaseMemObject(gated);
        if (proj) clReleaseMemObject(proj);
        return nullptr;
    };
    if (!bbuf || !cbuf || !xproj || !bx || !conv_out || !gated || !proj) return cleanup();

    // Cache sub-buffers per layer to avoid 3 clCreateSubBuffer per conv per token.
    static cl_mem s_bw[16] = {}, s_cw[16] = {}, s_xw[16] = {};
    if (!s_bw[layer_idx]) {
        const size_t slice_bytes = (size_t)hidden_size * (size_t)hidden_size * sizeof(nnopt_storage_t);
        cl_buffer_region region_b{0u, slice_bytes};
        cl_buffer_region region_c{slice_bytes, slice_bytes};
        cl_buffer_region region_x{slice_bytes * 2u, slice_bytes};
        cl_int sub_err = CL_SUCCESS;
        s_bw[layer_idx] = clCreateSubBuffer(in_w, CL_MEM_READ_ONLY, CL_BUFFER_CREATE_TYPE_REGION, &region_b, &sub_err);
        if (sub_err != CL_SUCCESS) { NNOPT_ERROR_FMT("lfm2_conv: in_proj B subbuffer failed (%d)", (int)sub_err); return cleanup(); }
        s_cw[layer_idx] = clCreateSubBuffer(in_w, CL_MEM_READ_ONLY, CL_BUFFER_CREATE_TYPE_REGION, &region_c, &sub_err);
        if (sub_err != CL_SUCCESS) { NNOPT_ERROR_FMT("lfm2_conv: in_proj C subbuffer failed (%d)", (int)sub_err); return cleanup(); }
        s_xw[layer_idx] = clCreateSubBuffer(in_w, CL_MEM_READ_ONLY, CL_BUFFER_CREATE_TYPE_REGION, &region_x, &sub_err);
        if (sub_err != CL_SUCCESS) { NNOPT_ERROR_FMT("lfm2_conv: in_proj x subbuffer failed (%d)", (int)sub_err); return cleanup(); }
    }
    b_w = s_bw[layer_idx]; c_w = s_cw[layer_idx]; x_w = s_xw[layer_idx];

    // Decode fast path: fused B+C+x into one dispatch. Saves 2 kernel launches
    // per conv layer × 10 conv layers = 20 launches per decode step.
    if (!pytorch_linear(queue, seq_len, hidden_size, hidden_size, norm1, b_w, bbuf) ||
        !pytorch_linear(queue, seq_len, hidden_size, hidden_size, norm1, c_w, cbuf) ||
        !pytorch_linear(queue, seq_len, hidden_size, hidden_size, norm1, x_w, xproj)) {
        return cleanup();
    }
    if (layer_idx == 0 && start_pos == 0) NNOPT_LAYER_CHECK("block0_sub_conv_in_proj_out", queue, bbuf, (size_t)seq_len * (size_t)hidden_size);

    static cl_kernel conv_k = nullptr, mul_k = nullptr, cache_write_k = nullptr;
    if (!conv_k) conv_k = kernel(cl_ctx, "lfm2_depthwise_conv3");
    if (!mul_k) mul_k = kernel(cl_ctx, "mul_buffers");
    if (!cache_write_k) cache_write_k = kernel(cl_ctx, "kv_cache_write");
    if (!conv_k || !mul_k || !cache_write_k) return cleanup();
    int n = seq_len * hidden_size;
    if (!set_arg_local(mul_k,0,sizeof(cl_mem),&bbuf,"B") || !set_arg_local(mul_k,1,sizeof(cl_mem),&xproj,"x") || !set_arg_local(mul_k,2,sizeof(cl_mem),&bx,"bx") || !set_arg_local(mul_k,3,sizeof(int),&n,"n") || !lfm2_kernel1(queue, mul_k, (size_t)n, "lfm2_conv_Bx")) {
        return cleanup();
    }
    // bx is now the new-positions Bx; the depthwise conv kernel reads its left
    // padding from conv_bx_cache (per-layer) when start_pos > 0.
    Model* model = g_active_model_for_vlm_splice;
    cl_mem cache_bx = (model && model->caches_ready()) ? model->conv_bx_cache(layer_idx) : nullptr;
    const int pad = (model && model->caches_ready()) ? model->conv_pad() : 0;
    if (start_pos > 0 && !cache_bx) {
        NNOPT_ERROR_FMT("lfm2_conv: decode-step start_pos=%d but conv_bx_cache[%d] missing", start_pos, layer_idx);
        return cleanup();
    }
    int ksize = MODEL_CONFIG::CONV_L_CACHE;
    // Bind a dummy buffer for cache_bx on prefill (kernel ignores it when start_pos==0).
    cl_mem cache_arg = cache_bx ? cache_bx : bx;
    // Track 5 — start_pos now comes from counter[0] in the persistent buffer.
    cl_mem counter_buf = (model && model->caches_ready()) ? model->counter_buf() : nullptr;
    if (!counter_buf) {
        NNOPT_ERROR("lfm2_conv: counter_buf not allocated (Model::ensure_caches must be called first)");
        return cleanup();
    }
    if (!set_arg_local(conv_k,0,sizeof(cl_mem),&bx,"bx") ||
        !set_arg_local(conv_k,1,sizeof(cl_mem),&cache_arg,"cache_bx") ||
        !set_arg_local(conv_k,2,sizeof(cl_mem),&conv_w,"w") ||
        !set_arg_local(conv_k,3,sizeof(cl_mem),&conv_out,"out") ||
        !set_arg_local(conv_k,4,sizeof(int),&seq_len,"rows") ||
        !set_arg_local(conv_k,5,sizeof(int),&hidden_size,"hidden") ||
        !set_arg_local(conv_k,6,sizeof(int),&ksize,"ksize") ||
        !set_arg_local(conv_k,7,sizeof(cl_mem),&counter_buf,"counter") ||
        !lfm2_kernel1(queue, conv_k, (size_t)seq_len * (size_t)hidden_size, "lfm2_depthwise_conv3")) {
        return cleanup();
    }
    // After conv: cache the LAST `pad` rows of bx so next decode step has its
    // left-padding ready. If seq_len < pad (e.g. seq_len==1 during decode steps
    // past the first), prepend the previous cache to the new bx logical stream:
    //   new_cache[i] = (bx if (start_pos + seq_len - pad + i) >= start_pos
    //                   else cache_bx)[(start_pos + seq_len - pad + i) - start_pos
    //                                  + (pad if from cache else 0)]
    // Equivalent: take the last `pad` rows of the virtual concat(cache_bx_old, bx).
    // For seq_len >= pad (prefill or long prompts), copy bx[seq_len-pad : seq_len, :] → cache.
    // For seq_len < pad (single-token decode after the first), this needs a shift +
    // append. We implement the general case with a tiny custom kernel.
    if (cache_bx && pad > 0) {
        // Strategy: kv_cache_write supports arbitrary [heads, new_rows, head_dim] →
        // [heads, stride, head_dim] writes at dst_start. Treat cache_bx layout as
        // [1, pad, hidden]. Step 1: if seq_len >= pad → just copy bx tail to cache.
        // Step 2: if seq_len < pad → shift cache left by seq_len, then append bx.
        const int heads_one = 1;
        if (seq_len >= pad) {
            const int src_offset = (seq_len - pad);  // rows offset into bx
            // Use a sub-buffer view of bx starting at src_offset rows.
            cl_buffer_region region{(size_t)src_offset * (size_t)hidden_size * sizeof(nnopt_storage_t),
                                    (size_t)pad * (size_t)hidden_size * sizeof(nnopt_storage_t)};
            cl_int sub_err = CL_SUCCESS;
            cl_mem bx_tail = clCreateSubBuffer(bx, CL_MEM_READ_ONLY, CL_BUFFER_CREATE_TYPE_REGION, &region, &sub_err);
            if (sub_err != CL_SUCCESS || !bx_tail) {
                NNOPT_ERROR_FMT("lfm2_conv: bx tail sub-buffer failed (%d)", (int)sub_err);
                return cleanup();
            }
            int dst_start_0 = 0;
            if (!set_arg_local(cache_write_k,0,sizeof(cl_mem),&bx_tail,"src") ||
                !set_arg_local(cache_write_k,1,sizeof(cl_mem),&cache_bx,"cache") ||
                !set_arg_local(cache_write_k,2,sizeof(int),&pad,"new_rows") ||
                !set_arg_local(cache_write_k,3,sizeof(int),&heads_one,"heads") ||
                !set_arg_local(cache_write_k,4,sizeof(int),&hidden_size,"head_dim") ||
                !set_arg_local(cache_write_k,5,sizeof(int),&pad,"stride") ||
                !set_arg_local(cache_write_k,6,sizeof(int),&dst_start_0,"dst_start") ||
                !lfm2_kernel1(queue, cache_write_k, (size_t)pad * (size_t)hidden_size, "conv_bx_cache_write")) {
                clReleaseMemObject(bx_tail);
                return cleanup();
            }
            clReleaseMemObject(bx_tail);
        } else {
            // seq_len < pad — typically seq_len == 1 with pad == 2. Build new cache
            // = concat(old_cache[seq_len:], bx). Shift cache left (in-place is
            // unsafe; round-trip via a small temp). For pad=2, seq_len=1:
            //   new_cache[0] = old_cache[1]   (kept)
            //   new_cache[1] = bx[0]
            cl_mem cache_tmp = lfm2_alloc(cl_ctx, (size_t)pad * (size_t)hidden_size, "conv_bx_tmp");
            if (!cache_tmp) return cleanup();
            // tmp[0..pad-seq_len] = old_cache[seq_len..pad]
            cl_int copy_err = clEnqueueCopyBuffer(queue, cache_bx, cache_tmp,
                                                  (size_t)seq_len * (size_t)hidden_size * sizeof(nnopt_storage_t),
                                                  0,
                                                  (size_t)(pad - seq_len) * (size_t)hidden_size * sizeof(nnopt_storage_t),
                                                  0, nullptr, nullptr);
            if (copy_err != CL_SUCCESS) {
                NNOPT_ERROR_FMT("lfm2_conv: cache shift copy failed (%d)", (int)copy_err);
                clReleaseMemObject(cache_tmp);
                return cleanup();
            }
            // tmp[pad-seq_len..pad] = bx[0..seq_len]
            copy_err = clEnqueueCopyBuffer(queue, bx, cache_tmp,
                                           0,
                                           (size_t)(pad - seq_len) * (size_t)hidden_size * sizeof(nnopt_storage_t),
                                           (size_t)seq_len * (size_t)hidden_size * sizeof(nnopt_storage_t),
                                           0, nullptr, nullptr);
            if (copy_err != CL_SUCCESS) {
                NNOPT_ERROR_FMT("lfm2_conv: cache append copy failed (%d)", (int)copy_err);
                clReleaseMemObject(cache_tmp);
                return cleanup();
            }
            // cache_bx ← cache_tmp (entire buffer)
            copy_err = clEnqueueCopyBuffer(queue, cache_tmp, cache_bx, 0, 0,
                                           (size_t)pad * (size_t)hidden_size * sizeof(nnopt_storage_t),
                                           0, nullptr, nullptr);
            if (copy_err != CL_SUCCESS) {
                NNOPT_ERROR_FMT("lfm2_conv: cache writeback failed (%d)", (int)copy_err);
                clReleaseMemObject(cache_tmp);
                return cleanup();
            }
            clReleaseMemObject(cache_tmp);
        }
    }
    if (!set_arg_local(mul_k,0,sizeof(cl_mem),&cbuf,"c") || !set_arg_local(mul_k,1,sizeof(cl_mem),&conv_out,"conv_out") || !set_arg_local(mul_k,2,sizeof(cl_mem),&gated,"gated") || !set_arg_local(mul_k,3,sizeof(int),&n,"n") || !lfm2_kernel1(queue, mul_k, (size_t)n, "lfm2_conv_gate")) {
        return cleanup();
    }
    if (!pytorch_linear(queue, seq_len, hidden_size, hidden_size, gated, out_w, proj)) {
        return cleanup();
    }
    if (layer_idx == 0 && start_pos == 0) NNOPT_LAYER_CHECK("block0_sub_conv_out_proj_out", queue, proj, (size_t)n);
    // Fused: after_op = hidden_in + proj ; norm2 = rms_norm(after_op) * ffn_norm_weight.
    cl_mem after_op = nullptr, norm2 = nullptr;
    bool fused_ok = lfm2_rms_norm_add(cl_ctx, weights, queue, hidden_in, proj,
                                       seq_len, hidden_size, prefix + ".ffn_norm.weight",
                                       &after_op, &norm2);
    clReleaseMemObject(norm1); norm1 = nullptr;
    clReleaseMemObject(bbuf); bbuf = nullptr;
    clReleaseMemObject(cbuf); cbuf = nullptr;
    clReleaseMemObject(xproj); xproj = nullptr;
    clReleaseMemObject(bx); bx = nullptr;
    clReleaseMemObject(conv_out); conv_out = nullptr;
    clReleaseMemObject(gated); gated = nullptr;
    clReleaseMemObject(proj); proj = nullptr;
    // b_w, c_w, x_w are cached statics — do not release.
    if (!fused_ok) return nullptr;
    NNOPT_LAYER_CHECK_FMT("lfm2_operator_%d", layer_idx, queue, after_op, (size_t)n);
    if (layer_idx == 0 && start_pos == 0) NNOPT_LAYER_CHECK("block0_sub_ffn_norm", queue, norm2, (size_t)n);
    cl_mem mlp = lfm2_mlp(cl_ctx, weights, queue, norm2, seq_len, hidden_size, MODEL_CONFIG::INTERMEDIATE_SIZE, prefix, layer_idx);
    clReleaseMemObject(norm2);
    if (!mlp) { clReleaseMemObject(after_op); return nullptr; }
    if (!element_add_inplace(queue, lfm2_program(cl_ctx), after_op, mlp, (size_t)n)) {
        clReleaseMemObject(after_op); clReleaseMemObject(mlp); return nullptr;
    }
    clReleaseMemObject(mlp);
    NNOPT_LAYER_CHECK_FMT("lfm2_layer_%d", layer_idx, queue, after_op, (size_t)n);
    return after_op;
}
