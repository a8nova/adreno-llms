// Shared StyleTTS2 primitive layer for kitten-tts-nano-0.1.
//
// Wraps kernels/styletts.cl and provides the composite ops every remaining
// stage is built from. See PORT_JOURNAL.md for the oracle-verified invariants
// each of these encodes.
#pragma once

#include "opencl_context.h"
#include "weights.h"
#include "utils.h"
#include <string>
#include <vector>

namespace st {

// Lazily builds kernels/styletts.cl and caches every kernel handle.
// Returns false (and NNOPT_ERRORs) if the program or any kernel fails to build.
bool init(OpenCLContext& ctx);

// ── buffers ──────────────────────────────────────────────────────────────
cl_mem alloc(OpenCLContext& ctx, size_t n_elems);          // storage_t buffer
cl_mem alloc_f32(OpenCLContext& ctx, size_t n_elems);      // float buffer
cl_mem upload_f32_as_storage(OpenCLContext& ctx, const std::vector<float>& v);
bool   download_as_f32(cl_command_queue q, cl_mem buf, size_t n, std::vector<float>& out);

// ── DynamicQuantizeLinear (model semantics, not an optimization) ─────────
// Computes per-tensor scale/zp then quantize→dequantizes in place-safe form.
// `params` must be a 2-float buffer from alloc_f32(ctx, 2).
bool dql(cl_command_queue q, cl_mem x, cl_mem params, cl_mem out, int n);

// ── conv ─────────────────────────────────────────────────────────────────
// in [Cin,T_in] -> out [Cout,T_out]; w [Cout, Cin/groups, K]; bias may be null.
bool conv1d(cl_command_queue q, cl_mem in, cl_mem w, cl_mem bias, cl_mem out,
            int Cin, int Cout, int T_in, int T_out,
            int K, int stride, int pad, int groups, int dilation = 1);

// General (non-depthwise) ConvTranspose1d. ONNX weight layout is [in, out, K].
bool conv_transpose1d(cl_command_queue q, cl_mem in, cl_mem w, cl_mem bias,
                      cl_mem out, int Cin, int Cout, int T_in, int T_out,
                      int K, int stride, int pad);

bool snake(cl_command_queue q, cl_mem in, cl_mem alpha, cl_mem inv_alpha,
           cl_mem out, int C, int T);
bool leaky_relu_a(cl_command_queue q, cl_mem in, cl_mem out, int n, float alpha);
bool scale(cl_command_queue q, cl_mem in, cl_mem out, int n, float s);
bool exp_(cl_command_queue q, cl_mem in, cl_mem out, int n);
bool mag_phase(cl_command_queue q, cl_mem re, cl_mem im, cl_mem mag, cl_mem phase,
               int n, float eps);
bool polar_to_cart(cl_command_queue q, cl_mem mag, cl_mem raw, cl_mem re, cl_mem im, int n);
bool istft_ola(cl_command_queue q, cl_mem re, cl_mem im, cl_mem basis_r, cl_mem basis_i,
               cl_mem out, int K, int Fr, int N, int hop, int T_out);
bool pad_front_reflect(cl_command_queue q, cl_mem in, cl_mem out, int C, int T_in, int pad_front);

// Depthwise ConvTranspose1d (groups == C). T_out is caller-computed (2*T_in for
// the stride-2 pad-1 output_padding-1 K-3 "pool" layers).
bool conv_transpose1d_dw(cl_command_queue q, cl_mem in, cl_mem w, cl_mem bias,
                         cl_mem out, int C, int T_in, int T_out,
                         int K, int stride, int pad);

// ── norms ────────────────────────────────────────────────────────────────
bool instancenorm(cl_command_queue q, cl_mem in, cl_mem norm_w, cl_mem norm_b,
                  cl_mem out, int C, int T);
bool layernorm_rows(cl_command_queue q, cl_mem in, cl_mem gamma, cl_mem beta,
                    cl_mem out, int rows, int cols, float eps);

// AdaIN: InstanceNorm(x) then (1+gamma)*n + beta, with gamma/beta taken as the
// two halves of a precomputed style-fc output [2C].
bool adain(cl_command_queue q, cl_mem x, cl_mem norm_w, cl_mem norm_b,
           cl_mem fc_out, cl_mem scratch, cl_mem out, int C, int T);

// Computes the AdaIN style-fc once: fc_out[2C] = style_q[1,128] @ W[128,2C] + b.
// `style_q` must ALREADY be the DQL'd style half (all fcs in a group share it).
cl_mem adain_fc(OpenCLContext& ctx, cl_command_queue q, Weights& w,
                cl_mem style_q, const std::string& w_key,
                const std::string& b_key, int C_out2);

// ── elementwise / shape ──────────────────────────────────────────────────
bool leaky_relu(cl_command_queue q, cl_mem in, cl_mem out, int n);
bool add(cl_command_queue q, cl_mem a, cl_mem b, cl_mem out, int n);
bool add_scaled(cl_command_queue q, cl_mem a, cl_mem b, cl_mem out, int n, float scale);
bool add_bias_ct(cl_command_queue q, cl_mem in, cl_mem bias, cl_mem out, int C, int T);
bool add_bias_rows(cl_command_queue q, cl_mem in, cl_mem bias, cl_mem out, int rows, int cols);
bool transpose(cl_command_queue q, cl_mem in, cl_mem out, int A, int B);

// AdaLayerNorm combine on a [rows, cols] row-major tensor (feature axis last).
bool adaln_rows(cl_command_queue q, cl_mem n, cl_mem fc_out, cl_mem out,
                int rows, int cols);

// out[t, 0:C1] = a[t,:] ; out[t, C1:C1+C2] = s[:]
bool pack_style(cl_command_queue q, cl_mem a, cl_mem s, cl_mem out,
                int T, int C1, int C2);
bool resize_nearest_t(cl_command_queue q, cl_mem in, cl_mem out, int C, int T_in, int factor);
bool embed_tc(cl_command_queue q, cl_mem ids, cl_mem emb, cl_mem out, int T, int C);
bool align_expand(cl_command_queue q, cl_mem in, cl_mem frame_src, cl_mem out,
                  int C, int T_in, int F);

// Concat on the channel axis of [C_i, T] buffers — contiguous block copies.
bool concat_ct(cl_command_queue q, const std::vector<cl_mem>& parts,
               const std::vector<int>& channels, cl_mem out, int T);

// ── quantized bidirectional LSTM (com.microsoft::DynamicQuantizeLSTM) ────
// Runs on the HOST: the recurrence is strictly sequential and the matrices are
// tiny (H=64), so a GPU dispatch per timestep would be slower than the CPU
// loop. Nine instances of this op exist in the model.
//
// x_tc:  [T, I] storage_t, time-major, batch 1
// out:   [T, 2H] storage_t, per-timestep concat [h_fwd(H) || h_bwd(H)]
// W key: [2, I, 4H]   R key: [2, H, 4H]   B key: [2, 8H]  (ALL pre-dequantized)
//
// Gate order is ONNX i,o,f,c. X is DQL'd ONCE over the whole tensor; h_{t-1} is
// re-DQL'd EVERY timestep (omitting that costs 1.7e-3 and drifts along t);
// c_t is never quantized.
bool bilstm_quant(cl_command_queue q, Weights& weights,
                  cl_mem x_tc, int T, int I, int H,
                  const std::string& w_key, const std::string& r_key,
                  const std::string& b_key, cl_mem out);

// Host-side DynamicQuantizeLinear over a float vector (used by bilstm_quant).
void dql_host(const float* x, int n, std::vector<float>& deq);

// ── AdainResBlk1d ────────────────────────────────────────────────────────
// The repeating StyleTTS2 block. Three shapes, selected by which weights the
// prefix owns:
//   A  (no conv1x1, no pool)   identity shortcut          F0.0/N.0, F0.2/N.2
//   B' (conv1x1, no pool)      1x1 shortcut               encode, decode.0/1/2
//   B  (conv1x1 + pool)        Resize-2x shortcut,        F0.1/N.1, decode.3
//                              pool upsamples the residual path
//
//   residual: AdaIN(norm1,x) -> LeakyReLU -> [pool] -> conv1 -> AdaIN(norm2)
//             -> LeakyReLU -> conv2
//   out = (residual + shortcut) * 1/sqrt(2)
//
// TRAP: norm1 sees the UN-upsampled x. Only the shortcut is Resize'd; the
// residual path is upsampled later, by pool.
//
// fc_norm1 / fc_norm2 are precomputed style-fc outputs ([2*C_in] and [2*C_out]).
// Returns a newly allocated [C_out, T_out] buffer, or nullptr.
cl_mem adain_resblk(OpenCLContext& ctx, cl_command_queue q, Weights& w,
                    cl_mem x, int C_in, int C_out, int T_in, int T_out,
                    const std::string& prefix,
                    cl_mem fc_norm1, cl_mem fc_norm2,
                    bool upsample, bool has_conv1x1);

}  // namespace st
