// Reference: kitten-tts-nano-0.1 ONNX graph /bert_encoder/* (nodes 1947-1949).
// StyleTTS2 bert_encoder: a single Linear projecting ALBERT hidden 768 -> 128.
//   MatMul: /bert_encoder/MatMul_output_0 = bert_out[T,768] @ onnx::MatMul_7763[768,128]
//   Add:    /bert_encoder/Add_output_0    = kmodel.bert_encoder.bias[128] + MatMul_out
// Output boundary dump: bert_encoder.Add_output_0  shape [1,70,128].
//
// Weights (ONNX MatMul layout [in,out]=[768,128] => pytorch_conv1d; nn-style bias):
//   onnx_MatMul_7763          [768,128]  projection weight (pre-dequantized fp)
//   kmodel.bert_encoder.bias  [128]      projection bias
//
// Input: the Bert stage output h [T,768] (nnopt_storage_t).

#include "../opencl_context.h"
#include "../weights.h"
#include "../nnopt_error.h"
#include "../model_config.h"
#include "../utils.h"
#include <string>

namespace {
constexpr int H_IN  = 768;   // ALBERT encoder hidden
constexpr int H_OUT = 128;   // bert_encoder projection dim

cl_program g_prog = nullptr;
cl_kernel  g_k_bias = nullptr;

bool ensure_program(OpenCLContext& cl_ctx) {
    if (g_prog) return true;
    g_prog = cl_ctx.build_program_from_file("kernels/bert_encoder.cl");
    if (!g_prog) { NNOPT_ERROR("BertEncoder: failed to build kernels/bert_encoder.cl"); return false; }
    cl_int e;
    g_k_bias = clCreateKernel(g_prog, "be_add_bias", &e);
    if (e != CL_SUCCESS) { NNOPT_ERROR_FMT("kernel be_add_bias %d", e); return false; }
    return true;
}
}  // namespace

extern "C" {
cl_mem BertEncoder_forward(
    OpenCLContext& cl_ctx,
    Weights& weights,
    cl_command_queue queue,
    cl_mem input,          // Bert output h [T,768]
    int seq_len,
    int layer_idx,
    int start_pos,
    cl_mem* k_cache_inout,
    cl_mem* v_cache_inout,
    cl_mem encoder_hidden_states,
    const char* weight_prefix)
{
    (void)layer_idx; (void)start_pos; (void)k_cache_inout; (void)v_cache_inout;
    (void)encoder_hidden_states; (void)weight_prefix;
    const int T = seq_len;
    if (!ensure_program(cl_ctx)) return nullptr;

    cl_mem W = weights.get_buffer("onnx_MatMul_7763");     // [768,128]
    cl_mem b = weights.get_buffer("kmodel.bert_encoder.bias"); // [128]
    if (!W || !b) { NNOPT_ERROR("BertEncoder: missing weights"); return nullptr; }

    cl_int err = CL_SUCCESS;
    cl_mem mm  = clCreateBuffer(cl_ctx.context(), CL_MEM_READ_WRITE,
                                (size_t)T * H_OUT * sizeof(nnopt_storage_t), nullptr, &err);
    cl_mem out = clCreateBuffer(cl_ctx.context(), CL_MEM_READ_WRITE,
                                (size_t)T * H_OUT * sizeof(nnopt_storage_t), nullptr, &err);
    auto cleanup = [&]() -> cl_mem {
        if (mm)  { clReleaseMemObject(mm);  mm  = nullptr; }
        if (out) { clReleaseMemObject(out); out = nullptr; }
        return nullptr;
    };
    if (!mm || !out) { NNOPT_ERROR("BertEncoder: alloc failed"); return cleanup(); }

    // MatMul: out[T,128] = in[T,768] @ W[768,128]  (ONNX [in,out] layout => conv1d).
    if (!pytorch_conv1d(queue, T, H_OUT, H_IN, input, W, mm)) {
        NNOPT_ERROR("BertEncoder: matmul failed"); return cleanup();
    }

    // Add bias broadcast over columns.
    {
        int rows = T, cols = H_OUT;
        cl_int e = CL_SUCCESS;
        e |= clSetKernelArg(g_k_bias, 0, sizeof(cl_mem), &mm);
        e |= clSetKernelArg(g_k_bias, 1, sizeof(cl_mem), &b);
        e |= clSetKernelArg(g_k_bias, 2, sizeof(cl_mem), &out);
        e |= clSetKernelArg(g_k_bias, 3, sizeof(int), &rows);
        e |= clSetKernelArg(g_k_bias, 4, sizeof(int), &cols);
        if (e != CL_SUCCESS) { NNOPT_ERROR_FMT("BertEncoder bias setarg %d", e); return cleanup(); }
        size_t gws = (size_t)rows * cols;
        e = clEnqueueNDRangeKernel(queue, g_k_bias, 1, nullptr, &gws, nullptr, 0, nullptr, nullptr);
        if (e != CL_SUCCESS) { NNOPT_ERROR_FMT("BertEncoder bias dispatch %d", e); return cleanup(); }
    }

    NNOPT_DEBUG_SYNC(queue);
    clReleaseMemObject(mm); mm = nullptr;
    cl_mem result = out; out = nullptr;
    return result;
}
}
