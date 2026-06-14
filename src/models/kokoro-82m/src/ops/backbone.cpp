// Reference: kokoro/model.py:KModel.forward_with_tokens (lines ~84-116)
// Graph wiring (matches the actual reference, NOT the VITS template):
//
//   1.  bert_dur  = bert(input_ids, mask)              [T, 768]
//   2.  d_en      = bert_encoder(bert_dur)             [T, 512]   (NLC; reference .T is layout-only)
//   3.  pred_dur  = predictor_durations(d_en, ref_s)   vector<int>[T]
//   4.  indices   = repeat_interleave(arange(T), pred_dur)  [T_frames]
//   5.  en        = gather_NCL(d_en, indices)          [512, T_frames]
//   6.  F0, N     = predictor_F0N(en, ref_s)           [T_frames] each
//   7.  t_en      = text_encoder(input_ids)            [T, 512]
//   8.  asr       = gather_NCL(t_en, indices)          [512, T_frames]
//   9.  audio     = decoder(asr, F0, N, ref_s[:, :128])  -> PCM int16

#include "opencl_context.h"
#include "weights.h"
#include "debug_utils.h"
#include "profiler.h"
#include "model_config.h"
#include "utils.h"

#include <CL/cl.h>
#include <chrono>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <string>
#include <vector>

// Forward decls for the ops.
extern "C" int op_bert(OpenCLContext&, Weights&, cl_command_queue,
                       cl_mem input_ids_i32, cl_mem out, int T);
extern "C" int op_bert_encoder(OpenCLContext&, Weights&, cl_command_queue,
                               cl_mem bert_h, cl_mem out, int T);
extern "C" int op_text_encoder_modules(OpenCLContext&, Weights&, cl_command_queue,
                                       cl_mem input_ids_i32, cl_mem out, int T);
extern "C" int op_predictor_durations(OpenCLContext&, Weights&, cl_command_queue,
                                      cl_mem d_en, cl_mem ref_s, int T,
                                      std::vector<int>& pred_dur);
extern "C" int op_predictor_F0N(OpenCLContext&, Weights&, cl_command_queue,
                                cl_mem en, cl_mem ref_s,
                                cl_mem F0_out, cl_mem N_out, int T_frames);
extern "C" int op_predictor_build_en640(OpenCLContext&, cl_command_queue,
                                         cl_mem d_en_gathered_512, cl_mem ref_s,
                                         cl_mem en640_out, int T_frames);
extern "C" int op_predictor_duration_encoder(OpenCLContext&, Weights&, cl_command_queue,
                                              cl_mem d_en_NLC, cl_mem ref_s,
                                              cl_mem d_out_NLC, int T_chars);
extern "C" int op_predictor_durations_real(OpenCLContext&, Weights&, cl_command_queue,
                                            cl_mem d_NLC, int T, std::vector<int>& pred_dur);
extern "C" int op_alignment_make_indices(const std::vector<int>& pred_dur,
                                         std::vector<int>& indices_out);
extern "C" int op_alignment_gather_NCL(OpenCLContext&, cl_command_queue,
                                       cl_mem x_nlc, cl_mem indices_dev, cl_mem out_ncl,
                                       int T_chars, int T_frames, int C);
extern "C" int op_decoder(OpenCLContext&, Weights&, cl_command_queue,
                          cl_mem asr, cl_mem F0_pred, cl_mem N_pred, cl_mem ref_s_dec,
                          int T_frames, std::vector<int16_t>& out_pcm_int16);

static cl_mem alloc_rw(OpenCLContext& cl_ctx, size_t bytes) {
    cl_int err = CL_SUCCESS;
    return clCreateBuffer(cl_ctx.context(), CL_MEM_READ_WRITE, bytes, nullptr, &err);
}

static cl_mem upload_i32(OpenCLContext& cl_ctx, cl_command_queue queue,
                         const std::vector<int32_t>& v) {
    cl_int err = CL_SUCCESS;
    cl_mem buf = clCreateBuffer(cl_ctx.context(),
                                CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
                                v.size() * sizeof(int32_t),
                                const_cast<int32_t*>(v.data()), &err);
    (void)queue;
    return (err == CL_SUCCESS) ? buf : nullptr;
}

// Upload fp32 host vector as nnopt_storage_t to device.
static cl_mem upload_floats_as_storage(OpenCLContext& cl_ctx, cl_command_queue queue,
                                       const std::vector<float>& v) {
    size_t n = v.size();
    size_t bytes = n * sizeof(nnopt_storage_t);
    cl_int err = CL_SUCCESS;
    cl_mem buf = clCreateBuffer(cl_ctx.context(), CL_MEM_READ_WRITE, bytes, nullptr, &err);
    if (err != CL_SUCCESS || !buf) return nullptr;
#ifdef NNOPT_USE_FP16
    std::vector<uint16_t> half(n);
    for (size_t i = 0; i < n; ++i) half[i] = nnopt_f32_to_f16(v[i]);
    clEnqueueWriteBuffer(queue, buf, CL_TRUE, 0, bytes, half.data(), 0, nullptr, nullptr);
#else
    clEnqueueWriteBuffer(queue, buf, CL_TRUE, 0, bytes, v.data(), 0, nullptr, nullptr);
#endif
    return buf;
}

int model_forward_graph_tts(OpenCLContext& cl_ctx,
                          Weights& weights,
                          const std::vector<int32_t>& input_ids,
                          const std::vector<float>& ref_s,
                          std::vector<int16_t>& out_pcm_int16) {
    cl_command_queue queue = cl_ctx.queue();
    if (!queue) { NNOPT_ERROR("backbone: null queue"); return -1; }

    const int T = (int)input_ids.size();
    if (T <= 0) { NNOPT_ERROR("backbone: empty input_ids"); return -1; }
    if (ref_s.size() != 256) {
        NNOPT_ERROR_FMT("backbone: ref_s must be 256 floats (got %zu)", ref_s.size());
        return -1;
    }

    const int Hbert = 768, Hmodel = 512;
    const size_t sz_t = sizeof(nnopt_storage_t);

    // Per-op timing for non-generator path (gated by env to keep production overhead 0)
    bool nong_prof = false;
    if (const char* p = std::getenv("NNOPT_NONGEN_PROFILE")) nong_prof = (p[0] == '1');
    auto t_start = std::chrono::steady_clock::now();
    auto t_prev = t_start;
    #define NONG_TICK(label) do { if (nong_prof) { clFinish(queue); auto _t = std::chrono::steady_clock::now(); double _s = std::chrono::duration<double>(_t - t_prev).count(); fprintf(stderr, "[nongen] %s @ %.3fs\n", (label), _s); fflush(stderr); t_prev = _t; } } while (0)

    cl_mem ids_dev   = upload_i32(cl_ctx, queue, input_ids);
    cl_mem ref_s_dev = upload_floats_as_storage(cl_ctx, queue, ref_s);
    if (!ids_dev || !ref_s_dev) { NNOPT_ERROR("backbone: input upload failed"); return -1; }
    NONG_TICK("upload_inputs");

    cl_mem bert_h = alloc_rw(cl_ctx, sz_t * T * Hbert);
    if (op_bert(cl_ctx, weights, queue, ids_dev, bert_h, T) != 0) return -1;
    NONG_TICK("op_bert");

    cl_mem d_en = alloc_rw(cl_ctx, sz_t * T * Hmodel);
    if (op_bert_encoder(cl_ctx, weights, queue, bert_h, d_en, T) != 0) return -1;
    NONG_TICK("op_bert_encoder");

    // Run DurationEncoder first to get d [T, 640]
    cl_mem d_de = alloc_rw(cl_ctx, sz_t * T * 640);
    if (op_predictor_duration_encoder(cl_ctx, weights, queue, d_en, ref_s_dev, d_de, T) != 0) return -1;
    NONG_TICK("op_predictor_duration_encoder");

    // Real durations from d
    std::vector<int> pred_dur;
    if (op_predictor_durations_real(cl_ctx, weights, queue, d_de, T, pred_dur) != 0) {
        // fall back to stub if real path fails
        if (op_predictor_durations(cl_ctx, weights, queue, d_en, ref_s_dev, T, pred_dur) != 0) return -1;
    }
    NONG_TICK("op_predictor_durations_real");
    {
        std::string s = "pred_dur: ";
        for (size_t i = 0; i < pred_dur.size(); ++i) { s += std::to_string(pred_dur[i]); s += " "; }
        NNOPT_CHECKPOINT(s.c_str());
    }

    std::vector<int> indices;
    int T_frames = op_alignment_make_indices(pred_dur, indices);
    if (T_frames <= 0) { NNOPT_ERROR("backbone: empty alignment"); return -1; }
    cl_int err = CL_SUCCESS;
    cl_mem indices_dev = clCreateBuffer(cl_ctx.context(),
                                        CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
                                        indices.size() * sizeof(int),
                                        indices.data(), &err);

    // Gather d_en (NLC [T, 512]) along indices to get en_512 [512, T_frames] (NCL) — kept for
    // potential downstream use (not used by F0Ntrain).
    cl_mem en = alloc_rw(cl_ctx, sz_t * Hmodel * T_frames);
    if (op_alignment_gather_NCL(cl_ctx, queue, d_en, indices_dev, en, T, T_frames, Hmodel) != 0) return -1;

    // Gather d_de via alignment indices: input is NLC [T, 640], output [640, T_frames] (NCL).
    cl_mem en640 = alloc_rw(cl_ctx, sz_t * 640 * T_frames);
    if (op_alignment_gather_NCL(cl_ctx, queue, d_de, indices_dev, en640, T, T_frames, 640) != 0) return -1;
    clReleaseMemObject(d_de);
    // F0/N at T_frames*2 (predictor middle block upsamples by 2).
    cl_mem F0_out = alloc_rw(cl_ctx, sz_t * T_frames * 2);
    cl_mem N_out  = alloc_rw(cl_ctx, sz_t * T_frames * 2);
    if (op_predictor_F0N(cl_ctx, weights, queue, en640, ref_s_dev, F0_out, N_out, T_frames) != 0) return -1;
    clReleaseMemObject(en640);

    NONG_TICK("alignment+gather+F0N");

    cl_mem t_en = alloc_rw(cl_ctx, sz_t * T * Hmodel);
    if (op_text_encoder_modules(cl_ctx, weights, queue, ids_dev, t_en, T) != 0) return -1;
    NONG_TICK("op_text_encoder_modules");

    cl_mem asr = alloc_rw(cl_ctx, sz_t * Hmodel * T_frames);
    if (op_alignment_gather_NCL(cl_ctx, queue, t_en, indices_dev, asr, T, T_frames, Hmodel) != 0) return -1;
    NONG_TICK("asr_gather");

    cl_mem ref_s_dec = ref_s_dev;
    int rc = op_decoder(cl_ctx, weights, queue, asr, F0_out, N_out, ref_s_dec,
                        T_frames, out_pcm_int16);
    NONG_TICK("op_decoder_total");

    for (cl_mem m : {ids_dev, ref_s_dev, bert_h, d_en, indices_dev, en, F0_out, N_out, t_en, asr}) {
        if (m) clReleaseMemObject(m);
    }
    return rc;
}
