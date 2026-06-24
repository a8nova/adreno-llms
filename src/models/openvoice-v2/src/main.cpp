// OpenVoice V2 tone-color converter — on-device inference entry point.
//
// Audio-to-audio: source waveform -> posterior encoder (enc_q) -> normalizing
// flow -> HiFi-GAN decoder -> re-voiced waveform. This is NOT an autoregressive
// token model, so there is no single Model::forward() / generate loop. Instead,
// like the seamless-m4t cascade, the entry point selects a stage by argv[1] and
// the real work lives in src/ops/*.cpp (run_clone is the shippable fused path;
// the others are per-stage cosine-verification harnesses).
//
//   clone     (default) — enc_q -> flow -> dec; --src WAV in, --out WAV out (real path)
//   extract             — ref_enc tone-color extractor: --in ref.wav --out g.bin
//   enc_q_wn            — posterior encoder (WaveNet) stage, cos check
//   flow               — normalizing flow stage, cos check
//   dec | decfast      — HiFi-GAN decoder stage, cos check
//   all                — every stage, dumping each intermediate for verification
//   bench | hwprobe    — microbenchmarks / hardware probe
//
// Each run_*() sets up its own OpenCLContext + Weights and emits its own
// BENCHMARK lines; this file is just the selector.

#include "debug_utils.h"   // nnopt_install_crash_handler

#include <cstdio>
#include <cstring>

extern int run_clone(int argc, char** argv);
extern int run_extract(int argc, char** argv);
extern int run_enc_q_wn();
extern int run_flow();
extern int run_dec();
extern int run_dec_fast();
extern int run_all();
extern int run_bench();
extern int run_hwprobe();

int main(int argc, char** argv) {
    // Arm the crash handler FIRST — on SIGSEGV/SIGABRT/SIGBUS it prints the last
    // NNOPT_CHECKPOINT, a backtrace, and the GPU-mem allocation log so a device
    // segfault reports WHERE it died instead of a bare "Segmentation fault".
    nnopt_install_crash_handler();
    std::setvbuf(stdout, nullptr, _IONBF, 0);

    const char* mode = (argc > 1 && argv[1][0] != '-') ? argv[1] : "clone";
    if (std::strcmp(mode, "clone")    == 0) return run_clone(argc, argv);
    if (std::strcmp(mode, "extract")  == 0) return run_extract(argc, argv);
    if (std::strcmp(mode, "enc_q_wn") == 0) return run_enc_q_wn();
    if (std::strcmp(mode, "flow")     == 0) return run_flow();
    if (std::strcmp(mode, "dec")      == 0) return run_dec();
    if (std::strcmp(mode, "decfast")  == 0) return run_dec_fast();
    if (std::strcmp(mode, "all")      == 0) return run_all();
    if (std::strcmp(mode, "bench")    == 0) return run_bench();
    if (std::strcmp(mode, "hwprobe")  == 0) return run_hwprobe();

    std::fprintf(stderr, "unknown mode: %s (expected: clone|extract|enc_q_wn|flow|dec|decfast|all|bench|hwprobe)\n", mode);
    return 64;
}
