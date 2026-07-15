#pragma once
// Reference: stable_audio_tools/models/autoencoders.py OobleckDecoder.forward
//            + DecoderBlock.forward + blocks.py ResidualUnit.forward + SnakeBeta.forward
//            + AutoencoderPretransform.decode (scale=1.0) + VAEBottleneck.decode (identity)
//
// Oobleck convolutional autoencoder decoder for Stable Audio Open Small.
// Input:  latent [64, T]  (channel-major, T = latent_len = 256)
// Output: audio  [2, T*2048]  (stereo, deinterleaved [C, samples])
//
// Structure (from the checkpoint weight keys):
//   layers.0  WNConv1d(64 -> 2048, k=7, pad=3)
//   layers.1  DecoderBlock stride=8  (2048 -> 1024)
//   layers.2  DecoderBlock stride=8  (1024 -> 512)
//   layers.3  DecoderBlock stride=4  (512  -> 256)
//   layers.4  DecoderBlock stride=4  (256  -> 128)
//   layers.5  DecoderBlock stride=2  (128  -> 128)
//   layers.6  SnakeBeta(128)
//   layers.7  WNConv1d(128 -> 2, k=7, pad=3, bias=False)  (final_tanh=False -> Identity)
//
//   DecoderBlock: SnakeBeta -> WNConvTranspose1d(k=2*stride,stride,pad=ceil(stride/2))
//                 -> ResidualUnit(d=1) -> ResidualUnit(d=3) -> ResidualUnit(d=9)
//   ResidualUnit: x + [ SnakeBeta -> WNConv1d(k=7,dilation,pad=(dil*6)//2)
//                       -> SnakeBeta -> WNConv1d(k=1) ](x)
//
// weight_norm: stored weight_g [Cout,1,1] + weight_v [Cout,Cin,K]; reconstructed
//   W[o] = g[o] * v[o] / ||v[o]||_2   (norm over (Cin,K), dim=0). Done host-side.

#include "opencl_context.h"
#include "weights.h"
#include <vector>
#include <string>

class Decoder {
public:
    Decoder(OpenCLContext& cl_ctx, Weights& weights);
    ~Decoder();

    bool initialize();

    // latent: host float [64 * T] channel-major. out: host float [2 * (T*2048)]
    // channel-major (out[0..N) = left, out[N..2N) = right).
    bool decode(const std::vector<float>& latent, int T, std::vector<float>& out);

private:
    OpenCLContext& cl_ctx_;
    Weights& weights_;
    cl_program prog_ = nullptr;   // kernels/decoder.cl  (snake_beta, add_cl)
    cl_program conv_ = nullptr;   // kernels/conv_1d.cl
    cl_program convt_ = nullptr;  // kernels/conv_transpose_1d.cl
    bool ready_ = false;
    // Persistent grow-only im2col scratch: per-conv alloc/free of ~100MB
    // buffers fragments the CL heap and starves later large allocs (block-5
    // convT output failed with CL_OUT_OF_RESOURCES).
    cl_mem col_scratch_ = nullptr;
    size_t col_scratch_elems_ = 0;

    // helpers ---------------------------------------------------------------
    cl_mem alloc(size_t nelem);
    cl_mem upload_f32(const std::vector<float>& host);
    void   download(cl_mem buf, std::vector<float>& host, size_t nelem);

    // Reconstruct weight_norm conv weight (host) and upload as a device buffer.
    // Returns a NEW cl_mem [Cout*Cin*K]; caller owns.
    // repack_mode: 0 = stored layout, 1 = conv t4x4 oc-tile repack (.r4),
    //              2 = convT-as-GEMM repack [C2*K, C1] (.tg — B1 campaign).
    cl_mem load_wn_weight(const std::string& prefix, int Cout, int Cin, int K, int repack_mode = 0);
    // Upload a plain weight/param vector [n] as a device buffer.
    cl_mem load_vec(const std::string& key, int n);

    // Conv1d (dilation-aware). in [Cin,Lin] -> out [Cout,Lout]. Returns new buf.
    cl_mem conv1d(cl_mem in, cl_mem w, cl_mem bias, int Cin, int Cout,
                  int Lin, int K, int stride, int padding, int dilation, int has_bias,
                  bool repacked = false, bool try_gemm = false);
    // im2col + CLBlast GEMM conv path (stride==1, plain folded weight layout).
    cl_mem conv1d_gemm(cl_mem in, cl_mem w, cl_mem bias, int Cin, int Cout,
                       int Lin, int K, int padding, int dilation, int has_bias, int Lout);
    // GEMM-path selector: env NNOPT_CONV_GEMM!=0 and im2col buffer fits.
    static bool gemm_path_ok(int Cin, int K, int Lout);
    // ConvTranspose1d. in [Cin,Lin] -> out [Cout,Lout]. Returns new buf. Lout set via ref.
    cl_mem conv_transpose1d(cl_mem in, cl_mem w, cl_mem bias, int Cin, int Cout,
                            int Lin, int K, int stride, int padding, int* Lout_out, int has_bias);
    // SnakeBeta in-place on x [C,L] with alpha,beta [C].
    void snake(cl_mem x, cl_mem alpha, cl_mem beta, int C, int L);
    // element add: out = a + b [n]. Returns new buf.
    cl_mem add(cl_mem a, cl_mem b, int n);

    // ResidualUnit: returns NEW buffer (x + layers(x)), same [C,L].
    cl_mem residual_unit(cl_mem x, const std::string& prefix, int C, int L, int dilation);
    // DecoderBlock: in [Cin,Lin] -> out [Cout, Lin*stride]. Returns new buf; sets Lout.
    cl_mem decoder_block(cl_mem x, const std::string& prefix, int Cin, int Cout,
                         int Lin, int stride, int* Lout_out);
};
