// Reference: model_info/transformers_src/modeling_qwen2.py:1-260 Qwen2Model/Qwen2ForCausalLM forward

#include "model.h"

#include "layers/attention.h"
#include "layers/embedding.h"
#include "layers/layer_norm.h"
#include "layers/mlp.h"

#include "debug_utils.h"
#include "model_config.h"
#include "sampler.h"
#include "utils.h"
#include "prof.h"
#include "weights.h"
#include "benchmark.h"

#include <clblast.h>
#include <cstdint>
#include <dlfcn.h>
#include <iostream>
#include <string>
#include <vector>

// ── cl_qcom_recordable_queues vendor API ──
// These typedefs + struct layouts mirror the validated record_probe in
// main.cpp. Symbols are dlsym'd from RTLD_DEFAULT — Adreno Android ICD
// hides vendor entry points from clGetExtensionFunctionAddressForPlatform.
namespace nnopt_qcom_rec {
    typedef void* cl_recording_qcom;
    typedef cl_recording_qcom (CL_API_CALL *fn_new_t)(cl_command_queue, cl_int*);
    typedef cl_int (CL_API_CALL *fn_end_t)(cl_recording_qcom);
    typedef cl_int (CL_API_CALL *fn_release_t)(cl_recording_qcom);

    // Struct layout per cl_qcom_recordable_queues. The PDF documents the
    // semantics but not the C struct; this layout matches the validated
    // record_probe in main.cpp. Note: arg_indx uses cl_uint (4 bytes);
    // natural alignment puts size_t (8 bytes) at offset 16 with 4 bytes
    // of padding after arg_indx. Total sizeof = 32 on arm64.
    struct cl_array_arg_qcom {
        cl_kernel    kernel;
        cl_uint      arg_indx;
        size_t       arg_size;
        const void*  arg_value;
    };
    // Alternate layout: maybe the SDK uses size_t for arg_indx instead of
    // cl_uint, eliminating padding. Try if cl_uint variant fails.
    struct cl_array_arg_qcom_alt {
        cl_kernel    kernel;
        size_t       arg_indx;
        size_t       arg_size;
        const void*  arg_value;
    };
    struct cl_array_kernel_exec_info_qcom {
        cl_kernel       kernel;
        cl_uint         indx;
        size_t          param_value_size;
        const void*     param_value;
    };
    typedef cl_int (CL_API_CALL *fn_enqueue_t)(
        cl_command_queue queue,
        cl_recording_qcom recording,
        size_t num_args,
        const cl_array_arg_qcom* args,
        size_t num_global_offsets,
        const cl_array_kernel_exec_info_qcom* global_offsets,
        size_t num_global_work_sizes,
        const cl_array_kernel_exec_info_qcom* global_work_sizes,
        size_t num_local_work_sizes,
        const cl_array_kernel_exec_info_qcom* local_work_sizes,
        cl_uint num_events_in_wait_list,
        const cl_event* event_wait_list,
        cl_event* event);

    typedef cl_ulong cl_queue_properties_qcom;
    typedef cl_command_queue (CL_API_CALL *fn_create_q_t)(
        cl_context, cl_device_id,
        const cl_queue_properties_qcom*, cl_int*);

    constexpr cl_queue_properties_qcom CL_QUEUE_PROPERTIES_QCOM = 0x1093;
    constexpr cl_queue_properties_qcom RECORD_BIT = (cl_queue_properties_qcom)1 << 30;

}

Model::Model(OpenCLContext& cl_ctx, Weights& weights)
    : cl_ctx_(cl_ctx), weights_(weights) {
    NNOPT_CHECKPOINT("Model constructor");

    utils_program_ = cl_ctx_.build_program_from_file(
        "kernels/utils.cl",
#ifdef NNOPT_USE_FP16
        "-DNNOPT_USE_FP16=1 -DUSE_FP16=1"
#else
        ""
#endif
    );
    if (!utils_program_) {
        NNOPT_ERROR("Failed to build kernels/utils.cl");
        return;
    }

    embedding_ = new Embedding(cl_ctx_, weights_);
    if (!embedding_->initialize()) {
        NNOPT_ERROR("embedding.initialize() FAILED");
        return;
    }
    NNOPT_LAYER_INIT("embedding");

    for (int i = 0; i < MODEL_CONFIG::NUM_HIDDEN_LAYERS; ++i) {
        const std::string layer_prefix = "model.layers." + std::to_string(i);

        pre_attn_norm_[i] = new LayerNorm(cl_ctx_, weights_, layer_prefix + ".input_layernorm.weight",
                                          MODEL_CONFIG::HIDDEN_SIZE, MODEL_CONFIG::RMS_NORM_EPS);
        if (!pre_attn_norm_[i]->initialize()) {
            NNOPT_ERROR_FMT("pre_attn_norm_[%d].initialize() FAILED", i);
            return;
        }
        NNOPT_LAYER_INIT_FMT("pre_attn_norm_%d", i);

        attn_[i] = new Attention(cl_ctx_, weights_, i);
        if (!attn_[i]->initialize()) {
            NNOPT_ERROR_FMT("attn_[%d].initialize() FAILED", i);
            return;
        }
        NNOPT_LAYER_INIT_FMT("attn_%d", i);

        post_attn_norm_[i] = new LayerNorm(cl_ctx_, weights_, layer_prefix + ".post_attention_layernorm.weight",
                                           MODEL_CONFIG::HIDDEN_SIZE, MODEL_CONFIG::RMS_NORM_EPS);
        if (!post_attn_norm_[i]->initialize()) {
            NNOPT_ERROR_FMT("post_attn_norm_[%d].initialize() FAILED", i);
            return;
        }
        NNOPT_LAYER_INIT_FMT("post_attn_norm_%d", i);

        mlp_[i] = new Mlp(cl_ctx_, weights_, layer_prefix + ".mlp", i);
        if (!mlp_[i]->initialize()) {
            NNOPT_ERROR_FMT("mlp_[%d].initialize() FAILED", i);
            return;
        }
        NNOPT_LAYER_INIT_FMT("mlp_%d", i);
    }

    final_norm_ = new LayerNorm(cl_ctx_, weights_, "model.norm.weight",
                                MODEL_CONFIG::HIDDEN_SIZE, MODEL_CONFIG::RMS_NORM_EPS);
    if (!final_norm_->initialize()) {
        NNOPT_ERROR("final_norm_.initialize() FAILED");
        return;
    }
    NNOPT_LAYER_INIT("final_norm");

    // Tied embeddings: lm_head.weight is aliased to model.embed_tokens.weight.
    w_lm_head_ = weights_.get_buffer("model.embed_tokens.weight");
    if (!w_lm_head_) {
        NNOPT_ERROR("missing model.embed_tokens.weight for tied lm_head");
        return;
    }

    // ── cl_qcom_recordable_queues probe ──
    // OFF BY DEFAULT — set NNOPT_RECORD=1 to enable.
    // Mechanism is hardware-validated (probe achieves 2.64× speedup) and
    // the per-decode capture+replay produced +6.5% (8.9→9.48 tok/s) when
    // overrides were skipped, but on Adreno 620 the arg-override array
    // returns -59 (CL_INVALID_OPERATION) for both struct layouts I could
    // probe (cl_uint and size_t arg_indx). Without the actual cl_qcom_
    // recordable_queues SDK header, the correct struct layout can't be
    // pinned down. Re-enable + iterate once SDK access is available.
    const char* rec_env = std::getenv("NNOPT_RECORD");
    bool record_enabled = (rec_env && rec_env[0] == '1');
    using namespace nnopt_qcom_rec;
    fn_new_     = record_enabled ? (void*)dlsym(RTLD_DEFAULT, "clNewRecordingQCOM")     : nullptr;
    fn_end_     = record_enabled ? (void*)dlsym(RTLD_DEFAULT, "clEndRecordingQCOM")     : nullptr;
    fn_release_ = record_enabled ? (void*)dlsym(RTLD_DEFAULT, "clReleaseRecordingQCOM") : nullptr;
    fn_enqueue_ = record_enabled ? (void*)dlsym(RTLD_DEFAULT, "clEnqueueRecordingQCOM") : nullptr;
    auto fn_create_q = record_enabled ? (fn_create_q_t)dlsym(RTLD_DEFAULT, "clCreateCommandQueueWithProperties") : nullptr;

    if (fn_new_ && fn_end_ && fn_release_ && fn_enqueue_ && fn_create_q) {
        cl_queue_properties_qcom props[] = {
            CL_QUEUE_PROPERTIES_QCOM, RECORD_BIT,
            0
        };
        cl_int qerr = CL_SUCCESS;
        cl_command_queue rq = fn_create_q(cl_ctx_.context(), cl_ctx_.device(),
                                          props, &qerr);
        if (qerr == CL_SUCCESS && rq) {
            recordable_q_ = rq;
            std::cerr << "Record: recordable queue created (rq=" << rq << ")\n";
        } else {
            std::cerr << "Record: clCreateCommandQueueWithProperties failed (err="
                      << qerr << "); decode replay disabled\n";
            fn_new_ = fn_end_ = fn_release_ = fn_enqueue_ = nullptr;
        }
    } else {
        std::cerr << "Record: missing one or more vendor symbols; decode replay disabled\n";
        fn_new_ = fn_end_ = fn_release_ = fn_enqueue_ = nullptr;
    }

    // Allocate counter buffer [0]=start_pos, [1]=seq_k.
    {
        cl_int cerr = CL_SUCCESS;
        buf_counter_ = clCreateBuffer(cl_ctx_.context(),
                                      CL_MEM_READ_WRITE,
                                      2 * sizeof(int), nullptr, &cerr);
        if (cerr != CL_SUCCESS || !buf_counter_) {
            NNOPT_ERROR_FMT("alloc buf_counter_ failed: %d", (int)cerr);
            return;
        }
    }

    ok_ = true;
}

Model::~Model() {
    if (rec_decode_ && fn_release_) {
        ((nnopt_qcom_rec::fn_release_t)fn_release_)((nnopt_qcom_rec::cl_recording_qcom)rec_decode_);
    }
    if (recordable_q_) clReleaseCommandQueue((cl_command_queue)recordable_q_);
    if (buf_counter_)  clReleaseMemObject(buf_counter_);
    if (argmax_partial_)  clReleaseKernel(argmax_partial_);
    if (argmax_finalize_) clReleaseKernel(argmax_finalize_);
    if (argmax_program_)  clReleaseProgram(argmax_program_);
    if (argmax_out_buf_)  clReleaseMemObject(argmax_out_buf_);
    if (argmax_pv_buf_)   clReleaseMemObject(argmax_pv_buf_);
    if (argmax_pi_buf_)   clReleaseMemObject(argmax_pi_buf_);
    if (buf_logits_)      clReleaseMemObject(buf_logits_);
    if (buf_decode_hidden_) clReleaseMemObject(buf_decode_hidden_);

    if (utils_program_) clReleaseProgram(utils_program_);

    if (embedding_) delete embedding_;
    for (int i = 0; i < MODEL_CONFIG::NUM_HIDDEN_LAYERS; ++i) {
        if (pre_attn_norm_[i]) delete pre_attn_norm_[i];
        if (attn_[i]) delete attn_[i];
        if (post_attn_norm_[i]) delete post_attn_norm_[i];
        if (mlp_[i]) delete mlp_[i];
    }
    if (final_norm_) delete final_norm_;
}

std::vector<float> Model::forward(const std::vector<int32_t>& input_ids) {
    return forward(input_ids, /*start_pos=*/0);
}

std::vector<float> Model::forward(const std::vector<int32_t>& input_ids, int start_pos) {
    if (!ok_) {
        NNOPT_ERROR("Model not initialized (ok_==false)");
        return {};
    }
    if (input_ids.empty()) {
        NNOPT_ERROR("forward(): empty input_ids");
        return {};
    }

    NNOPT_CHECKPOINT("forward() started");

    cl_command_queue queue = cl_ctx_.queue();
    const int seq_len = (int)input_ids.size();

    // Embedding: Qwen2Model.embed_tokens (RoPE models have no absolute wpe).
    cl_mem hidden = embedding_->forward(queue, input_ids, /*start_pos=*/start_pos);
    if (!hidden) return {};
    NNOPT_LAYER_CHECK("embedding", queue, hidden, (size_t)seq_len * MODEL_CONFIG::HIDDEN_SIZE);

    // Write counter so attention kernels see start_pos / seq_k at execution.
    if (buf_counter_) {
        const int seq_k_init = start_pos + seq_len;
        int cdata[2] = { start_pos, seq_k_init };
        clEnqueueWriteBuffer(queue, buf_counter_, CL_FALSE, 0, 2 * sizeof(int), cdata, 0, nullptr, nullptr);
    }

    for (int i = 0; i < MODEL_CONFIG::NUM_HIDDEN_LAYERS; ++i) {
        // input_layernorm
        cl_mem norm1 = pre_attn_norm_[i]->forward(queue, hidden, seq_len);
        if (!norm1) { clReleaseMemObject(hidden); return {}; }
        NNOPT_LAYER_CHECK_FMT("input_layernorm_%d", i, queue, norm1, (size_t)seq_len * MODEL_CONFIG::HIDDEN_SIZE);

        // self_attn — at decode (seq_len==1), pass hidden as residual_dest
        // so o_proj fuses with the residual_add. Returned cl_mem == hidden
        // when fusion engaged; otherwise it's a separate buffer that needs
        // a follow-up element_add_inplace.
        cl_mem attn_out = attn_[i]->forward(queue, norm1, /*cos=*/nullptr, /*sin=*/nullptr,
                                            /*seq_q=*/seq_len, start_pos,
                                            /*counter=*/buf_counter_,
                                            /*residual_dest=*/(seq_len == 1 ? hidden : nullptr));
        clReleaseMemObject(norm1);
        if (!attn_out) { clReleaseMemObject(hidden); return {}; }
        NNOPT_LAYER_CHECK_FMT("self_attn_%d", i, queue, attn_out, (size_t)seq_len * MODEL_CONFIG::HIDDEN_SIZE);

        // residual: hidden += attn_out. Skip when fusion already applied
        // (attn returned hidden retained — residual_add was folded into
        // the o_proj kernel).
        if (attn_out != hidden) {
            if (!element_add_inplace(queue, utils_program_, hidden, attn_out,
                                     (size_t)seq_len * MODEL_CONFIG::HIDDEN_SIZE)) {
                clReleaseMemObject(hidden);
                clReleaseMemObject(attn_out);
                return {};
            }
        }
        clReleaseMemObject(attn_out);  // matched against the retain in attn_->forward (or buf_proj_'s retain)
        NNOPT_LAYER_CHECK_FMT("resid_attn_%d", i, queue, hidden, (size_t)seq_len * MODEL_CONFIG::HIDDEN_SIZE);

        // post_attention_layernorm
        cl_mem norm2 = post_attn_norm_[i]->forward(queue, hidden, seq_len);
        if (!norm2) { clReleaseMemObject(hidden); return {}; }
        NNOPT_LAYER_CHECK_FMT("post_attention_layernorm_%d", i, queue, norm2, (size_t)seq_len * MODEL_CONFIG::HIDDEN_SIZE);

        // mlp — same fused-residual pattern at decode.
        cl_mem mlp_out = mlp_[i]->forward(queue, norm2, seq_len,
                                          /*residual_dest=*/(seq_len == 1 ? hidden : nullptr));
        clReleaseMemObject(norm2);
        if (!mlp_out) { clReleaseMemObject(hidden); return {}; }
        NNOPT_LAYER_CHECK_FMT("mlp_%d", i, queue, mlp_out, (size_t)seq_len * MODEL_CONFIG::HIDDEN_SIZE);

        if (mlp_out != hidden) {
            if (!element_add_inplace(queue, utils_program_, hidden, mlp_out,
                                     (size_t)seq_len * MODEL_CONFIG::HIDDEN_SIZE)) {
                clReleaseMemObject(hidden);
                clReleaseMemObject(mlp_out);
                return {};
            }
        }
        clReleaseMemObject(mlp_out);
        NNOPT_LAYER_CHECK_FMT("resid_mlp_%d", i, queue, hidden, (size_t)seq_len * MODEL_CONFIG::HIDDEN_SIZE);
    }

    cl_mem norm_f = final_norm_->forward(queue, hidden, seq_len);
    clReleaseMemObject(hidden);
    if (!norm_f) return {};
    NNOPT_LAYER_CHECK("final_norm", queue, norm_f, (size_t)seq_len * MODEL_CONFIG::HIDDEN_SIZE);

    // lm_head (tied): logits = hidden @ W^T, where W is [vocab, hidden].
    // Persistent buf_logits_ — saves one clCreateBuffer per forward.
    cl_int err = CL_SUCCESS;
    if (!buf_logits_ || seq_len > buf_logits_rows_) {
        if (buf_logits_) { clReleaseMemObject(buf_logits_); buf_logits_ = nullptr; }
        const size_t bytes = (size_t)seq_len * MODEL_CONFIG::VOCAB_SIZE * sizeof(nnopt_storage_t);
        buf_logits_ = clCreateBuffer(cl_ctx_.context(), CL_MEM_READ_WRITE, bytes, nullptr, &err);
        if (err != CL_SUCCESS || !buf_logits_) {
            NNOPT_ERROR_FMT("alloc buf_logits_ failed: %d", err);
            clReleaseMemObject(norm_f);
            return {};
        }
        buf_logits_rows_ = seq_len;
    }
    cl_mem logits = buf_logits_;

    if (!pytorch_linear(queue,
                        /*M=*/seq_len,
                        /*N=*/MODEL_CONFIG::VOCAB_SIZE,
                        /*K=*/MODEL_CONFIG::HIDDEN_SIZE,
                        /*A=*/norm_f,
                        /*W=*/w_lm_head_,
                        /*C=*/logits)) {
        clReleaseMemObject(norm_f);
        return {};
    }
    clReleaseMemObject(norm_f);

    NNOPT_LAYER_CHECK("lm_head", queue, logits, (size_t)seq_len * MODEL_CONFIG::VOCAB_SIZE);

    // Return last token logits as float vector.
    std::vector<float> out((size_t)MODEL_CONFIG::VOCAB_SIZE);
    const size_t row_bytes = (size_t)MODEL_CONFIG::VOCAB_SIZE * sizeof(nnopt_storage_t);
    const size_t last_off = (size_t)(seq_len - 1) * row_bytes;

#ifdef NNOPT_USE_FP16
    std::vector<nnopt_storage_t> tmp((size_t)MODEL_CONFIG::VOCAB_SIZE);
    clEnqueueReadBuffer(queue, logits, CL_TRUE, last_off, row_bytes, tmp.data(), 0, nullptr, nullptr);
    for (int i = 0; i < MODEL_CONFIG::VOCAB_SIZE; ++i) out[i] = nnopt_f16_to_f32(tmp[(size_t)i]);
#else
    clEnqueueReadBuffer(queue, logits, CL_TRUE, last_off, row_bytes, out.data(), 0, nullptr, nullptr);
#endif

    NNOPT_CHECKPOINT("forward() complete");
    return out;
}

std::vector<float> Model::forward_decode(int32_t token_id, int start_pos) {
    NNOPT_CHECKPOINT("forward_decode() started");
    return forward(std::vector<int32_t>{token_id}, start_pos);
}

// Greedy fast path: same forward as above, but resolves argmax on the GPU
// and reads back just one int32. Saves the 304 KB fp16 logits readback +
// host fp16→fp32 conversion + std::max_element scan over Qwen's 152K vocab.
// Builds the argmax kernel lazily on first call.
int32_t Model::forward_greedy(const std::vector<int32_t>& input_ids, int start_pos) {
    if (!ok_ || input_ids.empty()) return -1;

    constexpr int NUM_PARTIALS = 32;

    // Lazy-init GPU argmax kernels + scratch (kernels/gemv_m1.cl).
    if (!argmax_partial_) {
        argmax_program_ = cl_ctx_.build_program_from_file(
            "kernels/gemv_m1.cl",
#ifdef NNOPT_USE_FP16
            "-DNNOPT_USE_FP16=1 -DUSE_FP16=1"
#else
            ""
#endif
        );
        if (!argmax_program_) {
            NNOPT_ERROR("forward_greedy: build kernels/gemv_m1.cl for argmax failed");
            return -1;
        }
        cl_int err = CL_SUCCESS;
        argmax_partial_ = clCreateKernel(argmax_program_, "argmax_partial", &err);
        if (err != CL_SUCCESS || !argmax_partial_) {
            NNOPT_ERROR_FMT("forward_greedy: clCreateKernel(argmax_partial) failed: %d", err);
            return -1;
        }
        argmax_finalize_ = clCreateKernel(argmax_program_, "argmax_finalize", &err);
        if (err != CL_SUCCESS || !argmax_finalize_) {
            NNOPT_ERROR_FMT("forward_greedy: clCreateKernel(argmax_finalize) failed: %d", err);
            return -1;
        }
        argmax_out_buf_ = clCreateBuffer(cl_ctx_.context(), CL_MEM_READ_WRITE,
                                         sizeof(int32_t), nullptr, &err);
        if (err != CL_SUCCESS || !argmax_out_buf_) return -1;
        argmax_pv_buf_  = clCreateBuffer(cl_ctx_.context(), CL_MEM_READ_WRITE,
                                         (size_t)NUM_PARTIALS * sizeof(float), nullptr, &err);
        if (err != CL_SUCCESS || !argmax_pv_buf_) return -1;
        argmax_pi_buf_  = clCreateBuffer(cl_ctx_.context(), CL_MEM_READ_WRITE,
                                         (size_t)NUM_PARTIALS * sizeof(int32_t), nullptr, &err);
        if (err != CL_SUCCESS || !argmax_pi_buf_) return -1;
    }

    cl_command_queue live_q = cl_ctx_.queue();
    const int seq_len = (int)input_ids.size();

    // ── Helper: dispatch lm_head + argmax on live_q, read result. ──
    // lm_head and argmax intentionally run on live_q (NOT recorded). The
    // tiled image GEMV for N=151936 K=896 returns -59 (CL_INVALID_OPERATION)
    // when dispatched on a recordable queue — sub-buffer/image dispatches
    // appear to be incompatible with cl_qcom_recordable_queues.
    auto dispatch_lm_head_argmax = [&](cl_mem norm_f) -> int32_t {
        cl_int err = CL_SUCCESS;
        if (!buf_logits_) {
            const size_t bytes = (size_t)MODEL_CONFIG::VOCAB_SIZE * sizeof(nnopt_storage_t);
            buf_logits_ = clCreateBuffer(cl_ctx_.context(), CL_MEM_READ_WRITE, bytes, nullptr, &err);
            if (err != CL_SUCCESS || !buf_logits_) {
                NNOPT_ERROR_FMT("forward_greedy: alloc buf_logits_ failed: %d", err);
                return -1;
            }
            buf_logits_rows_ = 1;
        }
        if (!pytorch_linear(live_q, /*M=*/1, /*N=*/MODEL_CONFIG::VOCAB_SIZE,
                            /*K=*/MODEL_CONFIG::HIDDEN_SIZE,
                            norm_f, w_lm_head_, buf_logits_)) {
            return -1;
        }
        int N = MODEL_CONFIG::VOCAB_SIZE;
        int num_partials = NUM_PARTIALS;
        err = clSetKernelArg(argmax_partial_, 0, sizeof(cl_mem), &buf_logits_);
        if (err == CL_SUCCESS) err = clSetKernelArg(argmax_partial_, 1, sizeof(cl_mem), &argmax_pv_buf_);
        if (err == CL_SUCCESS) err = clSetKernelArg(argmax_partial_, 2, sizeof(cl_mem), &argmax_pi_buf_);
        if (err == CL_SUCCESS) err = clSetKernelArg(argmax_partial_, 3, sizeof(int), &N);
        if (err == CL_SUCCESS) err = clSetKernelArg(argmax_partial_, 4, sizeof(int), &num_partials);
        if (err != CL_SUCCESS) return -1;
        {
            size_t lws = 64;
            size_t gws = (size_t)NUM_PARTIALS * lws;
            err = nnopt_prof::enqueue(live_q, argmax_partial_, 1, nullptr, &gws, &lws, 0, nullptr, nullptr);
            if (err != CL_SUCCESS) {
                NNOPT_ERROR_FMT("forward_greedy: argmax_partial dispatch failed: %d", err);
                return -1;
            }
        }
        err = clSetKernelArg(argmax_finalize_, 0, sizeof(cl_mem), &argmax_pv_buf_);
        if (err == CL_SUCCESS) err = clSetKernelArg(argmax_finalize_, 1, sizeof(cl_mem), &argmax_pi_buf_);
        if (err == CL_SUCCESS) err = clSetKernelArg(argmax_finalize_, 2, sizeof(cl_mem), &argmax_out_buf_);
        if (err == CL_SUCCESS) err = clSetKernelArg(argmax_finalize_, 3, sizeof(int), &num_partials);
        if (err != CL_SUCCESS) return -1;
        {
            size_t lws = 64;
            size_t gws = 64;
            err = nnopt_prof::enqueue(live_q, argmax_finalize_, 1, nullptr, &gws, &lws, 0, nullptr, nullptr);
            if (err != CL_SUCCESS) {
                NNOPT_ERROR_FMT("forward_greedy: argmax_finalize dispatch failed: %d", err);
                return -1;
            }
        }
        int32_t out = -1;
        err = clEnqueueReadBuffer(live_q, argmax_out_buf_, CL_TRUE, 0, sizeof(int32_t), &out, 0, nullptr, nullptr);
        if (err != CL_SUCCESS) {
            NNOPT_ERROR_FMT("forward_greedy: argmax readback failed: %d", err);
            return -1;
        }
        return out;
    };

    // ── Recording-replay fast path ──
    // Recording covers embedding + 24 layers + final_norm. lm_head + argmax
    // are NOT recorded (sub-buffer/image incompatibility). After the replay
    // executes, final_norm_'s persistent output buffer holds the right
    // data; we dispatch lm_head + argmax against it on live_q.
    //
    // Counter-buffer approach: write {start_pos, seq_k} to buf_counter_ on
    // live_q BEFORE replay. Kernels (rope, kv_write, attn_scores, softmax,
    // attn_out) read from buf_counter_ at execution time. Buffer address is
    // stable across replays — no clSetKernelArg or arg-override array needed,
    // both of which return -59 (CL_INVALID_OPERATION) on Adreno 620.
    if (recording_built_ && seq_len == 1 && rec_decode_ && fn_enqueue_) {
        using namespace nnopt_qcom_rec;
        if (!embedding_->set_decode_token(live_q, input_ids[0])) return -1;

        const int seq_k = start_pos + 1;
        // Update counter before replay.
        {
            int cdata[2] = { start_pos, seq_k };
            clEnqueueWriteBuffer(live_q, buf_counter_, CL_TRUE, 0, 2 * sizeof(int), cdata, 0, nullptr, nullptr);
        }

        cl_int rerr = ((fn_enqueue_t)fn_enqueue_)(
            live_q, (cl_recording_qcom)rec_decode_,
            0, nullptr, 0, nullptr, 0, nullptr, 0, nullptr,
            0, nullptr, nullptr);
        if (rerr != CL_SUCCESS) {
            NNOPT_ERROR_FMT("forward_greedy: recording replay failed: %d (start_pos=%d)",
                            (int)rerr, start_pos);
            return -1;
        }

        cl_mem norm_f = final_norm_->buf_out_value();
        if (!norm_f) {
            NNOPT_ERROR("forward_greedy(replay): final_norm buf_out_ is null");
            return -1;
        }
        int32_t out_id = dispatch_lm_head_argmax(norm_f);
        if (out_id < 0) return -1;
        decode_iter_count_++;
        return out_id;
    }

    // ── Recording capture iter (iter 1) ──
    // dispatch_q = recordable_q_ → kernels are recorded, NOT executed.
    // After end_recording, we replay on live_q to actually compute iter 1's
    // result (and then read argmax). buffer ops (set_decode_token,
    // readback) always run on live_q.
    bool capture_recording = false;
    cl_command_queue dispatch_q = live_q;
    if (seq_len == 1 && recordable_q_ && fn_new_ && !rec_decode_ &&
        decode_iter_count_ >= 1) {
        // Pre-allocate buf_scores_ at MAX capacity so the buffer address is
        // stable across all replay iterations (it won't be reallocated as
        // seq_k grows during the decode loop).
        for (int i = 0; i < MODEL_CONFIG::NUM_HIDDEN_LAYERS; ++i) {
            attn_[i]->preallocate_decode_buffers_max();
        }
        using namespace nnopt_qcom_rec;
        cl_int rerr = CL_SUCCESS;
        rec_decode_ = ((fn_new_t)fn_new_)((cl_command_queue)recordable_q_, &rerr);
        if (rerr == CL_SUCCESS && rec_decode_) {
            capture_recording = true;
            dispatch_q = (cl_command_queue)recordable_q_;
        } else {
            NNOPT_ERROR_FMT("forward_greedy: clNewRecordingQCOM failed: %d (skipping recording)",
                            (int)rerr);
            rec_decode_ = nullptr;
        }
    }

    cl_command_queue queue = dispatch_q;

    // Persistent decode hidden buffer — allows the same cl_mem handle to be
    // reused across decode iterations, which is a prerequisite for
    // cl_qcom_recordable_queues replay (recorded kernels must reference
    // buffers that remain valid across replays). Only used at seq_len==1.
    cl_mem hidden = nullptr;
    const bool use_persistent_hidden = (seq_len == 1);
    if (use_persistent_hidden) {
        if (!buf_decode_hidden_) {
            cl_int berr = CL_SUCCESS;
            const size_t bytes = (size_t)MODEL_CONFIG::HIDDEN_SIZE * sizeof(nnopt_storage_t);
            buf_decode_hidden_ = clCreateBuffer(cl_ctx_.context(), CL_MEM_READ_WRITE,
                                                bytes, nullptr, &berr);
            if (berr != CL_SUCCESS || !buf_decode_hidden_) {
                NNOPT_ERROR_FMT("forward_greedy: alloc buf_decode_hidden_ failed: %d", (int)berr);
                return -1;
            }
        }
        // set_decode_token writes to ids_buf via clEnqueueWriteBuffer —
        // NOT recordable; must run on live_q. Embedding kernel reads
        // whatever is in ids_buf at execution (record iter does NOT
        // execute; replay reads buffer state at replay time).
        if (!embedding_->set_decode_token(live_q, input_ids[0])) return -1;
        // Write counter before capture/live forward.
        if (buf_counter_) {
            const int seq_k_val = start_pos + 1;
            int cdata[2] = { start_pos, seq_k_val };
            clEnqueueWriteBuffer(live_q, buf_counter_, CL_FALSE, 0, 2 * sizeof(int), cdata, 0, nullptr, nullptr);
        }
        if (!embedding_->forward_into_decode(queue, buf_decode_hidden_, start_pos)) return -1;
        hidden = buf_decode_hidden_;
    } else {
        hidden = embedding_->forward(queue, input_ids, /*start_pos=*/start_pos);
        if (!hidden) return -1;
    }

    auto release_hidden = [&]() {
        if (!use_persistent_hidden && hidden) clReleaseMemObject(hidden);
    };

    for (int i = 0; i < MODEL_CONFIG::NUM_HIDDEN_LAYERS; ++i) {
        cl_mem norm1 = pre_attn_norm_[i]->forward(queue, hidden, seq_len);
        if (!norm1) { release_hidden(); return -1; }

        cl_mem attn_out = attn_[i]->forward(queue, norm1, /*cos=*/nullptr, /*sin=*/nullptr,
                                            /*seq_q=*/seq_len, start_pos,
                                            /*counter=*/buf_counter_,
                                            /*residual_dest=*/(seq_len == 1 ? hidden : nullptr));
        clReleaseMemObject(norm1);
        if (!attn_out) { release_hidden(); return -1; }

        if (attn_out != hidden) {
            if (!element_add_inplace(queue, utils_program_, hidden, attn_out,
                                     (size_t)seq_len * MODEL_CONFIG::HIDDEN_SIZE)) {
                release_hidden();
                clReleaseMemObject(attn_out);
                return -1;
            }
        }
        clReleaseMemObject(attn_out);

        cl_mem norm2 = post_attn_norm_[i]->forward(queue, hidden, seq_len);
        if (!norm2) { release_hidden(); return -1; }

        cl_mem mlp_out = mlp_[i]->forward(queue, norm2, seq_len,
                                          /*residual_dest=*/(seq_len == 1 ? hidden : nullptr));
        clReleaseMemObject(norm2);
        if (!mlp_out) { release_hidden(); return -1; }

        if (mlp_out != hidden) {
            if (!element_add_inplace(queue, utils_program_, hidden, mlp_out,
                                     (size_t)seq_len * MODEL_CONFIG::HIDDEN_SIZE)) {
                release_hidden();
                clReleaseMemObject(mlp_out);
                return -1;
            }
        }
        clReleaseMemObject(mlp_out);
    }

    cl_mem norm_f = final_norm_->forward(queue, hidden, seq_len);
    release_hidden();
    if (!norm_f) return -1;

    // ── End the recording here (BEFORE lm_head/argmax). ──
    // lm_head's tiled-image GEMV + sub-buffer dispatches are incompatible
    // with cl_qcom_recordable_queues (return -59 / CL_INVALID_OPERATION
    // when enqueued on a recordable queue). They run on live_q after the
    // replay completes.
    if (capture_recording && rec_decode_) {
        using namespace nnopt_qcom_rec;
        cl_int e_end = ((fn_end_t)fn_end_)((cl_recording_qcom)rec_decode_);
        if (e_end != CL_SUCCESS) {
            NNOPT_ERROR_FMT("forward_greedy: clEndRecordingQCOM failed: %d", e_end);
            ((fn_release_t)fn_release_)((cl_recording_qcom)rec_decode_);
            rec_decode_ = nullptr;
            clReleaseMemObject(norm_f);
            return -1;
        }
        // Replay on live_q with no overrides — same start_pos as recording.
        // This actually executes embedding + layers + final_norm on live_q.
        cl_int e_rep = ((fn_enqueue_t)fn_enqueue_)(
            live_q, (cl_recording_qcom)rec_decode_,
            0, nullptr, 0, nullptr, 0, nullptr, 0, nullptr,
            0, nullptr, nullptr);
        if (e_rep != CL_SUCCESS) {
            NNOPT_ERROR_FMT("forward_greedy: initial replay after capture failed: %d", e_rep);
            clReleaseMemObject(norm_f);
            return -1;
        }
        recording_built_ = true;
        std::cerr << "Record: decode recording captured + replayed via counter-buffer (start_pos="
                  << start_pos << ")\n";
        // norm_f from final_norm_->forward(recordable_q,...) was a kernel
        // dispatched but NOT executed (recording-only). After the replay,
        // final_norm_'s persistent buf_out_ holds the right data. The
        // returned cl_mem from forward() is just a retain of buf_out_, so
        // it's still valid; release the local handle here, use the
        // accessor below.
        clReleaseMemObject(norm_f);
        norm_f = final_norm_->buf_out_value();
        clRetainMemObject(norm_f);  // match the release at the end of this fn
    }

    int32_t out_id = dispatch_lm_head_argmax(norm_f);
    clReleaseMemObject(norm_f);
    if (out_id < 0) return -1;
    decode_iter_count_++;
    return out_id;
}

// Chained-decode enqueue (Step 17). Mirror of forward_greedy's plain
// non-recording path, with two differences:
//   (a) the embedding reads its token id directly from argmax_out_buf_
//       (written by the previous step's argmax_finalize) instead of from
//       a host int passed via input_ids — saves one clEnqueueWriteBuffer
//       and the implicit host barrier it carries.
//   (b) the final clEnqueueReadBuffer of the produced int32 is OMITTED.
//       The new token sits in argmax_out_buf_ until the NEXT call's
//       embedding picks it up. The caller (generate's chained loop) does
//       async readback in parallel with the next iteration's enqueue.
//
// All allocations + lazy-init are shared with forward_greedy so the first
// chained call must follow at least one forward_greedy call (which seeds
// argmax_out_buf_ with the first decode token). generate() handles that.
bool Model::forward_chained_enqueue(int start_pos) {
    if (!ok_) return false;
    if (!argmax_partial_ || !argmax_out_buf_) {
        NNOPT_ERROR("forward_chained_enqueue: argmax kernels not initialized — must call forward_greedy first");
        return false;
    }

    constexpr int NUM_PARTIALS = 32;

    cl_command_queue queue = cl_ctx_.queue();
    const int seq_len = 1;
    cl_int err = CL_SUCCESS;

    // Allocate persistent decode hidden buffer if needed (shared with the
    // recording path; it's nullptr at first chained call when recording
    // is disabled).
    if (!buf_decode_hidden_) {
        const size_t bytes = (size_t)MODEL_CONFIG::HIDDEN_SIZE * sizeof(nnopt_storage_t);
        buf_decode_hidden_ = clCreateBuffer(cl_ctx_.context(), CL_MEM_READ_WRITE, bytes, nullptr, &err);
        if (err != CL_SUCCESS || !buf_decode_hidden_) {
            NNOPT_ERROR_FMT("forward_chained_enqueue: alloc buf_decode_hidden_ failed: %d", (int)err);
            return false;
        }
    }

    // 1) Update buf_counter_ before the embedding (read by attention's
    // RoPE / KV-write / scores / softmax / attn_out kernels at execution
    // time on this in-order queue).
    if (buf_counter_) {
        const int seq_k_val = start_pos + 1;
        int cdata[2] = { start_pos, seq_k_val };
        err = clEnqueueWriteBuffer(queue, buf_counter_, CL_FALSE, 0,
                                    2 * sizeof(int), cdata, 0, nullptr, nullptr);
        if (err != CL_SUCCESS) {
            NNOPT_ERROR_FMT("forward_chained_enqueue: counter write failed: %d", (int)err);
            return false;
        }
    }

    // 2) Embedding from device token (argmax_out_buf_ from previous step).
    if (!embedding_->forward_into_decode_from_device_token(
            queue, argmax_out_buf_, /*dev_offset_bytes=*/0,
            buf_decode_hidden_, start_pos)) {
        return false;
    }
    cl_mem hidden = buf_decode_hidden_;

    // 3) Layer loop (24 layers). Same pattern as forward_greedy non-recording.
    for (int i = 0; i < MODEL_CONFIG::NUM_HIDDEN_LAYERS; ++i) {
        cl_mem norm1 = pre_attn_norm_[i]->forward(queue, hidden, seq_len);
        if (!norm1) return false;

        cl_mem attn_out = attn_[i]->forward(queue, norm1, /*cos=*/nullptr, /*sin=*/nullptr,
                                            /*seq_q=*/seq_len, start_pos,
                                            /*counter=*/buf_counter_,
                                            /*residual_dest=*/hidden);
        clReleaseMemObject(norm1);
        if (!attn_out) return false;
        if (attn_out != hidden) {
            if (!element_add_inplace(queue, utils_program_, hidden, attn_out,
                                     (size_t)seq_len * MODEL_CONFIG::HIDDEN_SIZE)) {
                clReleaseMemObject(attn_out);
                return false;
            }
        }
        clReleaseMemObject(attn_out);

        cl_mem norm2 = post_attn_norm_[i]->forward(queue, hidden, seq_len);
        if (!norm2) return false;

        cl_mem mlp_out = mlp_[i]->forward(queue, norm2, seq_len, /*residual_dest=*/hidden);
        clReleaseMemObject(norm2);
        if (!mlp_out) return false;
        if (mlp_out != hidden) {
            if (!element_add_inplace(queue, utils_program_, hidden, mlp_out,
                                     (size_t)seq_len * MODEL_CONFIG::HIDDEN_SIZE)) {
                clReleaseMemObject(mlp_out);
                return false;
            }
        }
        clReleaseMemObject(mlp_out);
    }

    // 4) Final norm.
    cl_mem norm_f = final_norm_->forward(queue, hidden, seq_len);
    if (!norm_f) return false;

    // 5) lm_head: pytorch_linear → buf_logits_.
    if (!buf_logits_) {
        const size_t bytes = (size_t)MODEL_CONFIG::VOCAB_SIZE * sizeof(nnopt_storage_t);
        buf_logits_ = clCreateBuffer(cl_ctx_.context(), CL_MEM_READ_WRITE, bytes, nullptr, &err);
        if (err != CL_SUCCESS || !buf_logits_) {
            NNOPT_ERROR_FMT("forward_chained_enqueue: alloc buf_logits_ failed: %d", (int)err);
            clReleaseMemObject(norm_f);
            return false;
        }
        buf_logits_rows_ = 1;
    }
    if (!pytorch_linear(queue, /*M=*/1,
                        /*N=*/MODEL_CONFIG::VOCAB_SIZE,
                        /*K=*/MODEL_CONFIG::HIDDEN_SIZE,
                        norm_f, w_lm_head_, buf_logits_)) {
        clReleaseMemObject(norm_f);
        return false;
    }
    clReleaseMemObject(norm_f);

    // 6) Argmax. Writes int32 → argmax_out_buf_. NO host readback.
    int N = MODEL_CONFIG::VOCAB_SIZE;
    int num_partials = NUM_PARTIALS;
    err = clSetKernelArg(argmax_partial_, 0, sizeof(cl_mem), &buf_logits_);
    if (err == CL_SUCCESS) err = clSetKernelArg(argmax_partial_, 1, sizeof(cl_mem), &argmax_pv_buf_);
    if (err == CL_SUCCESS) err = clSetKernelArg(argmax_partial_, 2, sizeof(cl_mem), &argmax_pi_buf_);
    if (err == CL_SUCCESS) err = clSetKernelArg(argmax_partial_, 3, sizeof(int), &N);
    if (err == CL_SUCCESS) err = clSetKernelArg(argmax_partial_, 4, sizeof(int), &num_partials);
    if (err != CL_SUCCESS) {
        NNOPT_ERROR_FMT("forward_chained_enqueue: argmax_partial setArg failed: %d", err);
        return false;
    }
    {
        size_t lws = 64;
        size_t gws = (size_t)NUM_PARTIALS * lws;
        err = nnopt_prof::enqueue(queue, argmax_partial_, 1, nullptr, &gws, &lws, 0, nullptr, nullptr);
        if (err != CL_SUCCESS) {
            NNOPT_ERROR_FMT("forward_chained_enqueue: argmax_partial enqueue failed: %d", err);
            return false;
        }
    }
    err = clSetKernelArg(argmax_finalize_, 0, sizeof(cl_mem), &argmax_pv_buf_);
    if (err == CL_SUCCESS) err = clSetKernelArg(argmax_finalize_, 1, sizeof(cl_mem), &argmax_pi_buf_);
    if (err == CL_SUCCESS) err = clSetKernelArg(argmax_finalize_, 2, sizeof(cl_mem), &argmax_out_buf_);
    if (err == CL_SUCCESS) err = clSetKernelArg(argmax_finalize_, 3, sizeof(int), &num_partials);
    if (err != CL_SUCCESS) {
        NNOPT_ERROR_FMT("forward_chained_enqueue: argmax_finalize setArg failed: %d", err);
        return false;
    }
    {
        size_t lws = 64;
        size_t gws = 64;
        err = nnopt_prof::enqueue(queue, argmax_finalize_, 1, nullptr, &gws, &lws, 0, nullptr, nullptr);
        if (err != CL_SUCCESS) {
            NNOPT_ERROR_FMT("forward_chained_enqueue: argmax_finalize enqueue failed: %d", err);
            return false;
        }
    }
    return true;
}

std::vector<int32_t> Model::generate(
    const std::vector<int32_t>& prompt_ids,
    int max_new_tokens,
    SamplerConfig sampler_config,
    TokenCallback on_token
) {
    NNOPT_CHECKPOINT("generate() started");
    Sampler sampler(sampler_config);
    auto ids = prompt_ids;

    if (prompt_ids.empty()) {
        NNOPT_ERROR("generate(): empty prompt");
        return ids;
    }

    auto logits = forward(prompt_ids, /*start_pos=*/0);
    if (logits.empty()) {
        NNOPT_ERROR("prefill forward() returned empty logits");
        return ids;
    }

    std::vector<int32_t> generated;
    int next_token = sampler.sample(logits, generated);
    ids.push_back(next_token);
    generated.push_back(next_token);
    NNOPT_BENCH_FIRST_TOKEN();

    // Drop prefill events from the profile — we want a decode-only attribution.
    nnopt_prof::reset();

    // First-token streaming callback. Fire BEFORE the eos check so the
    // user sees the eos token text (if any) even if generation stops here.
    if (on_token) on_token(next_token, ids);

    if (sampler_config.eos_token_id >= 0 && next_token == sampler_config.eos_token_id) {
        NNOPT_CHECKPOINT("generate() complete (eos at first token)");
        return ids;
    }

    // Greedy fast path: temperature ≤ 0 with no repetition penalty means the
    // sampler reduces to plain argmax. forward_greedy resolves the argmax on
    // the GPU and returns just the int32 token id — saves the 304 KB fp16
    // logits readback + host max_element scan over Qwen's 152K vocab on every
    // decode step. Same kernels otherwise; same token IDs guaranteed.
    const bool greedy_path = (sampler_config.temperature <= 0.0f) &&
                             (sampler_config.repetition_penalty == 1.0f);

    // ── Chained-decode fast path (Step 17) ──
    // When greedy AND we have ≥2 decode steps to do, use the chained
    // forward_chained_enqueue + ping-pong async readback pattern. The
    // token id flows from the GPU argmax buffer directly into the next
    // step's embedding (no host roundtrip), and host readback is
    // overlapped with the next forward's enqueue.
    //
    // The first chained call needs argmax_out_buf_ already seeded with
    // the first decode token. forward_greedy below does that — its sync
    // readback also gives us the value to push into ids/generated.
    //
    // Disabled if NNOPT_RECORD=1 (recording path uses its own dispatch).
    const char* nochain_env = std::getenv("NNOPT_NO_CHAIN");
    const bool chain_enabled = greedy_path && (max_new_tokens > 2)
                               && !(nochain_env && nochain_env[0] == '1')
                               && !recordable_q_;

    if (chain_enabled) {
        // Step 1: seed argmax_out_buf_ with the first decode token by
        // running forward_greedy once. This both produces the
        // generated[0] token AND populates argmax_out_buf_ for the next
        // call's embedding.
        const int pos0 = (int)prompt_ids.size();
        int32_t tok0 = forward_greedy(std::vector<int32_t>{next_token}, pos0);
        if (tok0 < 0) {
            NNOPT_ERROR("chained decode: seed forward_greedy failed");
            return ids;
        }
        next_token = tok0;
        ids.push_back(next_token);
        generated.push_back(next_token);
        if (on_token) on_token(next_token, ids);
        bool eos_seen =
            (sampler_config.eos_token_id >= 0 && next_token == sampler_config.eos_token_id);

        // Step 2+: chained loop with 2-slot ping-pong async readback.
        cl_command_queue queue = cl_ctx_.queue();
        int32_t host_int[2] = {0, 0};
        cl_event readback_event[2] = {nullptr, nullptr};
        bool slot_pending[2] = {false, false};
        int prev_slot = -1;

        for (int i = 2; i < max_new_tokens && !eos_seen; i++) {
            const int pos = (int)prompt_ids.size() + (i - 1);
            const int cur_slot = i & 1;

            if (!forward_chained_enqueue(pos)) {
                NNOPT_ERROR("chained decode forward failed");
                break;
            }
            cl_int rerr = clEnqueueReadBuffer(queue, argmax_out_buf_, CL_FALSE, 0,
                                               sizeof(int32_t), &host_int[cur_slot],
                                               0, nullptr, &readback_event[cur_slot]);
            if (rerr != CL_SUCCESS) {
                NNOPT_ERROR_FMT("chained async readback enqueue failed: %d", (int)rerr);
                break;
            }
            slot_pending[cur_slot] = true;
            clFlush(queue);

            // Drain the previous iteration's readback (the token from one step ago).
            if (prev_slot >= 0 && slot_pending[prev_slot]) {
                clWaitForEvents(1, &readback_event[prev_slot]);
                clReleaseEvent(readback_event[prev_slot]);
                readback_event[prev_slot] = nullptr;
                slot_pending[prev_slot] = false;
                int32_t t = host_int[prev_slot];
                ids.push_back(t);
                generated.push_back(t);
                if (on_token) on_token(t, ids);
                if (sampler_config.eos_token_id >= 0 && t == sampler_config.eos_token_id) {
                    eos_seen = true;
                }
            }
            prev_slot = cur_slot;
        }
        // Drain the in-flight final readback.
        if (prev_slot >= 0 && slot_pending[prev_slot]) {
            clWaitForEvents(1, &readback_event[prev_slot]);
            clReleaseEvent(readback_event[prev_slot]);
            readback_event[prev_slot] = nullptr;
            slot_pending[prev_slot] = false;
            if (!eos_seen) {
                int32_t t = host_int[prev_slot];
                ids.push_back(t);
                generated.push_back(t);
                if (on_token) on_token(t, ids);
            }
        }
        NNOPT_CHECKPOINT("generate() complete (chained)");
        nnopt_prof::dump(cl_ctx_.queue());
        return ids;
    }

    for (int i = 1; i < max_new_tokens; i++) {
        const int pos = (int)prompt_ids.size() + (i - 1);

        if (greedy_path) {
            int32_t tok = forward_greedy(std::vector<int32_t>{next_token}, pos);
            if (tok < 0) {
                NNOPT_ERROR("decode forward_greedy() failed");
                break;
            }
            next_token = tok;
        } else {
            logits = forward(std::vector<int32_t>{next_token}, pos);
            if (logits.empty()) {
                NNOPT_ERROR("decode forward() returned empty logits");
                break;
            }
            next_token = sampler.sample(logits, generated);
        }
        ids.push_back(next_token);
        generated.push_back(next_token);

        // Streaming callback after each new token. Fires BEFORE the eos
        // check so the user can see what the model decided to stop on.
        if (on_token) on_token(next_token, ids);

        if (sampler_config.eos_token_id >= 0 && next_token == sampler_config.eos_token_id) break;
    }

    NNOPT_CHECKPOINT("generate() complete");
    nnopt_prof::dump(cl_ctx_.queue());
    return ids;
}
