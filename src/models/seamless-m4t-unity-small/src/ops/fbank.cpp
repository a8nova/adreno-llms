// Stage 2: kaldi-povey fbank. audio[16000] -> log-mel [nframes, 80]. GPU kernels.
#include "pipeline.h"
#include <limits>

std::vector<float> Pipeline::fbank(const std::vector<float>& audio, int& nframes) {
    const int SR = 16000, FL = 400, FS = 160, NB = 80, NFFT = 512, NSPEC = NFFT / 2 + 1;
    int ns = (int)audio.size();
    nframes = 1 + (ns - FL) / FS;
    Tensor a = ops_.upload(audio);
    Tensor win = ops_.fbank_window(a, nframes, FL, FS);
    cl_mem power = ops_.fbank_power(win, nframes, FL, NFFT, NSPEC);
    Tensor mel = ops_.fbank_mel(power, nframes, NB, NSPEC, SR, NFFT, std::numeric_limits<float>::epsilon());
    return ops_.download(mel);
}
