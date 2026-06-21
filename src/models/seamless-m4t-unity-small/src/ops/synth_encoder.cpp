// Stage 5a: synthesizer_encoder. text-decoder hidden -> T2U memory. 2-layer
// pre-norm transformer encoder (self-attn + GELU FFN, no cross-attn, no pos emb).
#include "pipeline.h"
#include <string>

std::vector<float> Pipeline::synth_encoder(const std::vector<float>& hidden, int T) {
    const std::string dir = "model.synthesizer_encoder.";
    auto W = [&](const std::string& k) { return ops_.weight(k); };
    Tensor x = ops_.upload(hidden);
    for (int L = 0; L < 2; ++L) {
        std::string pre = dir + "layers." + std::to_string(L) + ".";
        Tensor sn = ops_.layernorm(x, T, Dm, W(pre + "self_attn_layer_norm.weight"), W(pre + "self_attn_layer_norm.bias"));
        Tensor a = mha("", sn, T, sn, T, pre + "self_attn.", false);
        ops_.axpy(x, a, 1.0f);
        Tensor fn = ops_.layernorm(x, T, Dm, W(pre + "final_layer_norm.weight"), W(pre + "final_layer_norm.bias"));
        Tensor h1 = ops_.linear(fn, T, Dm, W(pre + "fc1.weight"), W(pre + "fc1.bias"), 4096);
        ops_.act(h1, ACT_GELU);
        Tensor h2 = ops_.linear(h1, T, 4096, W(pre + "fc2.weight"), W(pre + "fc2.bias"), Dm);
        ops_.axpy(x, h2, 1.0f);
    }
    Tensor out = ops_.layernorm(x, T, Dm, W(dir + "layer_norm.weight"), W(dir + "layer_norm.bias"));
    return ops_.download(out);
}
