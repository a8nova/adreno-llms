// Shared biLSTM primitive used by predictor + text_encoder.
//
// Reference: PyTorch nn.LSTM(input_size, hidden_size, bidirectional=True, batch_first=True)
//
// Weight layout (per direction):
//   weight_ih_l0[_reverse] : [4*H, in_size]   gate order: i, f, g, o
//   weight_hh_l0[_reverse] : [4*H, H]
//   bias_ih_l0[_reverse]   : [4*H]
//   bias_hh_l0[_reverse]   : [4*H]
//
// Forward equations (per timestep):
//   gates = W_ih x + b_ih + W_hh h_{t-1} + b_hh         # [4H]
//   i = sigmoid(gates[0:H])
//   f = sigmoid(gates[H:2H])
//   g = tanh(gates[2H:3H])
//   o = sigmoid(gates[3H:4H])
//   c_t = f * c_{t-1} + i * g
//   h_t = o * tanh(c_t)
//
// biLSTM output: concat([h_forward, h_backward], axis=-1)  -> [T, 2*H]

#include "opencl_context.h"
#include "weights.h"
#include "debug_utils.h"
#include "profiler.h"
#include "model_config.h"
#include "utils.h"

#include <CL/cl.h>
#include <cstdint>
#include <cstring>
#include <string>
#include <vector>

static cl_program g_lstm_prog = nullptr;
static cl_kernel  g_k_lstm_step = nullptr;
static cl_kernel  g_k_lstm_step_off = nullptr;
static cl_kernel  g_k_lstm_seq = nullptr;
static cl_kernel  g_k_zero = nullptr;
static cl_kernel  g_k_concat_lstm = nullptr;

static const char* k_lstm_src = R"CLC(
#ifdef NNOPT_USE_FP16
  #pragma OPENCL EXTENSION cl_khr_fp16 : enable
  typedef half storage_t;
  #define LOAD(p,i)    vload_half((i), (__global const half*)(p))
  #define STORE(p,i,v) vstore_half((float)(v), (i), (__global half*)(p))
#else
  typedef float storage_t;
  #define LOAD(p,i)    ((p)[(i)])
  #define STORE(p,i,v) ((p)[(i)] = (v))
#endif

// One LSTM step. Reads h_in, writes h_out and c_out.
// Wx_row: [4H] precomputed (W_ih @ x_t + b_ih) for this timestep.
__kernel void lstm_step(__global const storage_t* Wx_row,     // [4H]
                        __global const storage_t* W_hh,       // [4H, H]
                        __global const storage_t* b_hh,       // [4H]
                        __global const storage_t* h_in,       // [H]
                        __global const storage_t* c_in,       // [H]
                        __global storage_t* h_out,            // [H]
                        __global storage_t* c_out,            // [H]
                        int H) {
    int j = get_global_id(0);
    if (j >= H) return;
    // For each gate row j_gate = gate*H + j, compute:
    //   sum = Wx_row[row] + b_hh[row] + sum_k W_hh[row, k] * h_in[k]
    float pre[4];
    for (int gate = 0; gate < 4; ++gate) {
        int row = gate * H + j;
        float s = (float)LOAD(Wx_row, row) + (float)LOAD(b_hh, row);
        int wbase = row * H;
        for (int k = 0; k < H; ++k) s += (float)LOAD(h_in, k) * (float)LOAD(W_hh, wbase + k);
        pre[gate] = s;
    }
    float i_g = 1.0f / (1.0f + exp(-pre[0]));
    float f_g = 1.0f / (1.0f + exp(-pre[1]));
    float g_g = tanh(pre[2]);
    float o_g = 1.0f / (1.0f + exp(-pre[3]));
    float c_new = f_g * (float)LOAD(c_in, j) + i_g * g_g;
    float h_new = o_g * tanh(c_new);
    STORE(c_out, j, c_new);
    STORE(h_out, j, h_new);
}

// Offset-arg variant: Wx row and h output are addressed inside the big
// buffers directly — no per-step clCreateSubBuffer, no per-step copy.
__kernel void lstm_step_off(__global const storage_t* Wx,        // [T, 4H]
                            int wx_off,
                            __global const storage_t* W_hh,      // [4H, H]
                            __global const storage_t* b_hh,      // [4H]
                            __global const storage_t* h_prev,    // [T, H] or zeros
                            int hprev_off,
                            __global const storage_t* c_in,      // [H]
                            __global storage_t* h_out,           // [T, H]
                            int hout_off,
                            __global storage_t* c_out,           // [H]
                            int H) {
    int j = get_global_id(0);
    if (j >= H) return;
    float pre[4];
    for (int gate = 0; gate < 4; ++gate) {
        int row = gate * H + j;
        float s = (float)LOAD(Wx, wx_off + row) + (float)LOAD(b_hh, row);
        int wbase = row * H;
        for (int k = 0; k < H; ++k) s += (float)LOAD(h_prev, hprev_off + k) * (float)LOAD(W_hh, wbase + k);
        pre[gate] = s;
    }
    float i_g = 1.0f / (1.0f + exp(-pre[0]));
    float f_g = 1.0f / (1.0f + exp(-pre[1]));
    float g_g = tanh(pre[2]);
    float o_g = 1.0f / (1.0f + exp(-pre[3]));
    float c_new = f_g * (float)LOAD(c_in, j) + i_g * g_g;
    float h_new = o_g * tanh(c_new);
    STORE(c_out, j, c_new);
    STORE(h_out, hout_off + j, h_new);
}

// Whole-recurrence kernel: ONE launch runs all T steps with the hidden state
// in LDS (fp32) and the cell state in a register. Replaces T sequential
// launches (~0.5ms round-trip each). One workgroup of H=256 lanes; W_hh rows
// are read with 4-wide native-half loads (the scalar vload_half path is ~16x
// slower on this driver).
__kernel void lstm_seq(__global const storage_t* Wx,      // [T, 4H] (b_ih+b_hh folded separately)
                       __global const storage_t* W_hh,    // [4H, H]
                       __global const storage_t* b,       // [4H]
                       __global storage_t* h_out,         // [T, H]
                       int T, int H, int reverse) {
    int j = (int)get_local_id(0);
    __local float h_lds[256];
    float c = 0.0f;
    if (j < H) h_lds[j] = 0.0f;
    barrier(CLK_LOCAL_MEM_FENCE);
    for (int step = 0; step < T; ++step) {
        int t = reverse ? (T - 1 - step) : step;
        int wx_off = mul24(t, H << 2);
        float h_new = 0.0f;
        if (j < H) {
            float pre[4];
            for (int gate = 0; gate < 4; ++gate) {
                int row = mad24(gate, H, j);
                float sgate = (float)LOAD(Wx, wx_off + row) + (float)LOAD(b, row);
                int wb = mul24(row, H);
                for (int k = 0; k < H; k += 4) {
#ifdef NNOPT_USE_FP16
                    float4 wv = convert_float4(vload4(0, (__global const half*)W_hh + wb + k));
#else
                    float4 wv = vload4(0, (__global const float*)W_hh + wb + k);
#endif
                    float4 hv = vload4(0, h_lds + k);
                    sgate += dot(wv, hv);
                }
                pre[gate] = sgate;
            }
            float i_g = 1.0f / (1.0f + native_exp(-pre[0]));
            float f_g = 1.0f / (1.0f + native_exp(-pre[1]));
            float g_g = tanh(pre[2]);
            float o_g = 1.0f / (1.0f + native_exp(-pre[3]));
            c = f_g * c + i_g * g_g;
            h_new = o_g * tanh(c);
        }
        barrier(CLK_LOCAL_MEM_FENCE);   // everyone done reading old h
        if (j < H) {
            h_lds[j] = h_new;
            STORE(h_out, mad24(t, H, j), h_new);
        }
        barrier(CLK_LOCAL_MEM_FENCE);   // new h visible before next step
    }
}

__kernel void zero_buf(__global storage_t* y, int N) {
    int i = get_global_id(0); if (i >= N) return;
    STORE(y, i, 0.0f);
}

// Concatenate forward+backward outputs along last dim.
// fwd: [T, H], bwd: [T, H] (already time-reversed back to natural order). out: [T, 2H].
__kernel void concat_bi(__global const storage_t* fwd,
                        __global const storage_t* bwd,
                        __global storage_t* y, int T, int H) {
    int t = get_global_id(0);
    int j = get_global_id(1);
    if (t >= T || j >= 2*H) return;
    float v;
    if (j < H) v = (float)LOAD(fwd, t*H + j);
    else v = (float)LOAD(bwd, t*H + (j - H));
    STORE(y, t*2*H + j, v);
}
)CLC";

static bool ensure_built(OpenCLContext& cl_ctx) {
    if (g_k_lstm_step) return true;
    cl_int err = CL_SUCCESS;
    const char* opts =
#ifdef NNOPT_USE_FP16
        "-DNNOPT_USE_FP16=1";
#else
        "";
#endif
    cl_device_id dev = cl_ctx.device();
    g_lstm_prog = nnopt_build_program_cached(cl_ctx.context(), dev, k_lstm_src, opts, "lstm", &err);
    if (!g_lstm_prog) return false;
    g_k_lstm_step = clCreateKernel(g_lstm_prog, "lstm_step", &err);
    g_k_lstm_step_off = clCreateKernel(g_lstm_prog, "lstm_step_off", &err);
    g_k_lstm_seq = clCreateKernel(g_lstm_prog, "lstm_seq", &err);
    g_k_zero      = clCreateKernel(g_lstm_prog, "zero_buf", &err);
    g_k_concat_lstm = clCreateKernel(g_lstm_prog, "concat_bi", &err);
    return g_k_lstm_step && g_k_lstm_step_off && g_k_lstm_seq && g_k_zero && g_k_concat_lstm;
}

static cl_mem alloc(OpenCLContext& cl_ctx, size_t bytes) {
    cl_int e=CL_SUCCESS;
    return clCreateBuffer(cl_ctx.context(), CL_MEM_READ_WRITE, bytes, nullptr, &e);
}

static void zero_buffer(cl_command_queue queue, cl_mem buf, int N) {
    clSetKernelArg(g_k_zero, 0, sizeof(cl_mem), &buf);
    clSetKernelArg(g_k_zero, 1, sizeof(int), &N);
    size_t gws = (size_t)N;
    nnopt_enqueue_profiled(queue, g_k_zero, 1, nullptr, &gws, nullptr, 0, nullptr, nullptr);
}

// Apply one direction of LSTM. x_nlc is [T, in_size]. h_out is [T, H].
// reverse=true processes time in reverse and writes out in original order
// (so h_out[t] corresponds to backward-direction state at timestep t).
static int run_lstm_direction(OpenCLContext& cl_ctx, Weights& weights, cl_command_queue queue,
                              const std::string& wprefix, const std::string& suffix,
                              cl_mem x_nlc, int T, int in_size, int H,
                              cl_mem h_out, bool reverse) {
    cl_mem W_ih = weights.get_buffer(wprefix + ".weight_ih_l0" + suffix);
    cl_mem W_hh = weights.get_buffer(wprefix + ".weight_hh_l0" + suffix);
    cl_mem b_ih = weights.get_buffer(wprefix + ".bias_ih_l0" + suffix);
    cl_mem b_hh = weights.get_buffer(wprefix + ".bias_hh_l0" + suffix);
    if (!W_ih || !W_hh || !b_ih || !b_hh) {
        NNOPT_ERROR_FMT("lstm: missing weights for %s%s", wprefix.c_str(), suffix.c_str());
        return -1;
    }
    int four_H = 4 * H;

    // Precompute Wx[t, :] = W_ih @ x_t + b_ih for all t. Use pytorch_linear (treats x as [M=T, K=in_size]).
    cl_mem Wx = alloc(cl_ctx, sizeof(nnopt_storage_t) * T * four_H);
    if (!pytorch_linear(queue, T, four_H, in_size, x_nlc, W_ih, Wx)) {
        NNOPT_ERROR("lstm: Wx pytorch_linear failed");
        clReleaseMemObject(Wx);
        return -1;
    }
    // Add b_ih to every row of Wx. (We can fold b_ih into the per-step gate sum, but cheaper to apply once here.)
    // Reuse: a tiny inline kernel could do this; for simplicity write a host loop via a small kernel.
    // Actually we already need to add b_hh inside lstm_step, so just add b_ih here via a broadcast.
    // We'll use the concat kernel-launch infra; reuse zero+ a small bias add. To save effort, fold:
    // pass b_ih separately into lstm_step. But our lstm_step already adds b_hh. Easier: precompute
    // Wx_with_b = Wx + b_ih (broadcast). Reuse zero_buf? No. Quick host loop via a tiny kernel.
    // For maximum simplicity here, we'll let lstm_step add b_ih: pass it as the "b_hh" arg and skip
    // b_hh. But that breaks if we want both. Workaround: precompute b_combined = b_ih + b_hh on host
    // and pass that as "b_hh" to lstm_step. PyTorch always sums the two biases per step, so the
    // distinction is semantic only.
    cl_mem b_combined = alloc(cl_ctx, sizeof(nnopt_storage_t) * four_H);
    {
        // Read b_ih and b_hh to host, sum, write back.
        std::vector<float> bi(four_H), bh(four_H);
#ifdef NNOPT_USE_FP16
        std::vector<uint16_t> bi_h(four_H), bh_h(four_H), bs_h(four_H);
        clEnqueueReadBuffer(queue, b_ih, CL_TRUE, 0, sizeof(uint16_t)*four_H, bi_h.data(), 0, nullptr, nullptr);
        clEnqueueReadBuffer(queue, b_hh, CL_TRUE, 0, sizeof(uint16_t)*four_H, bh_h.data(), 0, nullptr, nullptr);
        for (int i = 0; i < four_H; ++i) bs_h[i] = nnopt_f32_to_f16(nnopt_f16_to_f32(bi_h[i]) + nnopt_f16_to_f32(bh_h[i]));
        clEnqueueWriteBuffer(queue, b_combined, CL_TRUE, 0, sizeof(uint16_t)*four_H, bs_h.data(), 0, nullptr, nullptr);
#else
        clEnqueueReadBuffer(queue, b_ih, CL_TRUE, 0, sizeof(float)*four_H, bi.data(), 0, nullptr, nullptr);
        clEnqueueReadBuffer(queue, b_hh, CL_TRUE, 0, sizeof(float)*four_H, bh.data(), 0, nullptr, nullptr);
        for (int i = 0; i < four_H; ++i) bi[i] += bh[i];
        clEnqueueWriteBuffer(queue, b_combined, CL_TRUE, 0, sizeof(float)*four_H, bi.data(), 0, nullptr, nullptr);
#endif
    }

    // h, c double-buffered.
    cl_mem h_a = alloc(cl_ctx, sizeof(nnopt_storage_t) * H);
    cl_mem h_b = alloc(cl_ctx, sizeof(nnopt_storage_t) * H);
    cl_mem c_a = alloc(cl_ctx, sizeof(nnopt_storage_t) * H);
    cl_mem c_b = alloc(cl_ctx, sizeof(nnopt_storage_t) * H);
    zero_buffer(queue, h_a, H);
    zero_buffer(queue, c_a, H);

    cl_mem h_in = h_a, h_outk = h_b, c_in = c_a, c_outk = c_b;

    // Whole recurrence in ONE kernel launch when the workgroup can hold H
    // lanes; falls back to the per-step kernel otherwise.
    if (H <= 256 && (H % 4) == 0) {
        int rev = reverse ? 1 : 0;
        clSetKernelArg(g_k_lstm_seq, 0, sizeof(cl_mem), &Wx);
        clSetKernelArg(g_k_lstm_seq, 1, sizeof(cl_mem), &W_hh);
        clSetKernelArg(g_k_lstm_seq, 2, sizeof(cl_mem), &b_combined);
        clSetKernelArg(g_k_lstm_seq, 3, sizeof(cl_mem), &h_out);
        clSetKernelArg(g_k_lstm_seq, 4, sizeof(int), &T);
        clSetKernelArg(g_k_lstm_seq, 5, sizeof(int), &H);
        clSetKernelArg(g_k_lstm_seq, 6, sizeof(int), &rev);
        size_t gws = 256, lws = 256;
        nnopt_enqueue_profiled(queue, g_k_lstm_seq, 1, nullptr, &gws, &lws, 0, nullptr, nullptr);
    } else {
    for (int step = 0; step < T; ++step) {
        int t = reverse ? (T - 1 - step) : step;
        int wx_off = t * four_H;
        int hout_off = t * H;
        int t_prev = reverse ? (t + 1) : (t - 1);
        cl_mem h_prev_buf = (step == 0) ? h_a /* zeros */ : h_out;
        int hprev_off = (step == 0) ? 0 : t_prev * H;
        clSetKernelArg(g_k_lstm_step_off, 0, sizeof(cl_mem), &Wx);
        clSetKernelArg(g_k_lstm_step_off, 1, sizeof(int), &wx_off);
        clSetKernelArg(g_k_lstm_step_off, 2, sizeof(cl_mem), &W_hh);
        clSetKernelArg(g_k_lstm_step_off, 3, sizeof(cl_mem), &b_combined);
        clSetKernelArg(g_k_lstm_step_off, 4, sizeof(cl_mem), &h_prev_buf);
        clSetKernelArg(g_k_lstm_step_off, 5, sizeof(int), &hprev_off);
        clSetKernelArg(g_k_lstm_step_off, 6, sizeof(cl_mem), &c_in);
        clSetKernelArg(g_k_lstm_step_off, 7, sizeof(cl_mem), &h_out);
        clSetKernelArg(g_k_lstm_step_off, 8, sizeof(int), &hout_off);
        clSetKernelArg(g_k_lstm_step_off, 9, sizeof(cl_mem), &c_outk);
        clSetKernelArg(g_k_lstm_step_off, 10, sizeof(int), &H);
        size_t gws = (size_t)H;
        nnopt_enqueue_profiled(queue, g_k_lstm_step_off, 1, nullptr, &gws, nullptr, 0, nullptr, nullptr);
        cl_mem tmp = c_in; c_in = c_outk; c_outk = tmp;
    }
    }

    clReleaseMemObject(Wx);
    clReleaseMemObject(b_combined);
    clReleaseMemObject(h_a); clReleaseMemObject(h_b);
    clReleaseMemObject(c_a); clReleaseMemObject(c_b);
    return 0;
}

// Public: run a PyTorch nn.LSTM(in_size, hidden_size, bidirectional=True) on x [T, in_size].
// out: caller-allocated [T, 2*H].
extern "C" int prim_bilstm(OpenCLContext& cl_ctx, Weights& weights, cl_command_queue queue,
                           const std::string& wprefix,   // e.g. "predictor.module.lstm"
                           cl_mem x_nlc, cl_mem out_nlc,
                           int T, int in_size, int H) {
    if (!ensure_built(cl_ctx)) return -1;
    cl_mem fwd = alloc(cl_ctx, sizeof(nnopt_storage_t) * T * H);
    cl_mem bwd = alloc(cl_ctx, sizeof(nnopt_storage_t) * T * H);
    if (run_lstm_direction(cl_ctx, weights, queue, wprefix, "", x_nlc, T, in_size, H, fwd, false) != 0) {
        clReleaseMemObject(fwd); clReleaseMemObject(bwd); return -1;
    }
    if (run_lstm_direction(cl_ctx, weights, queue, wprefix, "_reverse", x_nlc, T, in_size, H, bwd, true) != 0) {
        clReleaseMemObject(fwd); clReleaseMemObject(bwd); return -1;
    }
    // Concat -> [T, 2H]
    clSetKernelArg(g_k_concat_lstm, 0, sizeof(cl_mem), &fwd);
    clSetKernelArg(g_k_concat_lstm, 1, sizeof(cl_mem), &bwd);
    clSetKernelArg(g_k_concat_lstm, 2, sizeof(cl_mem), &out_nlc);
    clSetKernelArg(g_k_concat_lstm, 3, sizeof(int), &T);
    clSetKernelArg(g_k_concat_lstm, 4, sizeof(int), &H);
    size_t gws[2] = {(size_t)T, (size_t)(2*H)};
    nnopt_enqueue_profiled(queue, g_k_concat_lstm, 2, nullptr, gws, nullptr, 0, nullptr, nullptr);
    clReleaseMemObject(fwd);
    clReleaseMemObject(bwd);
    return 0;
}
