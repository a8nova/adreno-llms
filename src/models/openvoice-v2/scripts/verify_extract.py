#!/usr/bin/env python3
"""E2 validation: device ref_enc extractor vs a torch reference built from the
SAME adreno weight bin (no upstream checkpoint needed — the adreno .bin IS the
converted converter checkpoint). Implements ReferenceEncoder.forward with torch's
own Conv2d/GRU/LayerNorm so only the architecture (read from models.py) is trusted.

  ~/.nnopt/ref_venvs/env_latest/bin/python scripts/verify_extract.py <src.wav> <device_g.bin>
"""
import sys, json
import numpy as np, torch, torch.nn.functional as F, librosa
from openvoice.mel_processing import spectrogram_torch

N_FFT, HOP, WIN, SR = 1024, 256, 1024, 22050
BIN = "weights/model.fp16.bin"
META = "weights/model.fp16.meta.json"


def load_weights():
    meta = json.load(open(META))["tensors"]
    raw = open(BIN, "rb").read()
    out = {}
    for k, t in meta.items():
        if not k.startswith("ref_enc"):
            continue
        off, nb = t["offset"], t["size_bytes"]
        a = np.frombuffer(raw[off:off + nb], dtype=np.float16).astype(np.float32)
        out[k] = torch.from_numpy(a.reshape(t["shape"]).copy())
    return out


def conv_w(W, base):  # weight_norm reconstruct: v * g/||v|| (norm over dims 1,2,3)
    v, g = W[base + ".weight_v"], W[base + ".weight_g"]
    nrm = v.flatten(1).norm(dim=1).view(-1, 1, 1, 1)
    return v * (g / (nrm + 1e-12))


def torch_ref_enc(spec, W):  # spec: [513, T]
    x = spec.transpose(0, 1).reshape(1, 1, -1, 513)  # [1,1,T,513]
    x = F.layer_norm(x, [513], W["ref_enc.layernorm.weight"], W["ref_enc.layernorm.bias"], 1e-5)
    chans = [1, 32, 32, 64, 64, 128, 128]
    for l in range(6):
        base = f"ref_enc.convs.{l}"
        x = F.conv2d(x, conv_w(W, base), W[base + ".bias"], stride=2, padding=1)
        x = F.relu(x)
    # [1,128,H6,9] -> [1,H6,128*9]
    x = x.transpose(1, 2).contiguous().view(1, x.size(2), -1)
    gru = torch.nn.GRU(input_size=x.size(2), hidden_size=128, batch_first=True)
    sd = {
        "weight_ih_l0": W["ref_enc.gru.weight_ih_l0"], "weight_hh_l0": W["ref_enc.gru.weight_hh_l0"],
        "bias_ih_l0": W["ref_enc.gru.bias_ih_l0"], "bias_hh_l0": W["ref_enc.gru.bias_hh_l0"],
    }
    gru.load_state_dict(sd)
    _, h_n = gru(x)                      # h_n: [1,1,128]
    h = h_n.squeeze(0)                   # [1,128]
    g = F.linear(h, W["ref_enc.proj.weight"], W["ref_enc.proj.bias"])  # [1,256]
    return g.squeeze(0).detach().numpy()


def main():
    wav, gbin = sys.argv[1], sys.argv[2]
    W = load_weights()
    audio, _ = librosa.load(wav, sr=SR, mono=True)
    pcm = (np.clip(np.round(audio * 32768.0), -32768, 32767).astype(np.int16)).astype(np.float32) / 32768.0
    spec = spectrogram_torch(torch.FloatTensor(pcm).unsqueeze(0), N_FFT, SR, HOP, WIN, center=False)[0]
    g_ref = torch_ref_enc(spec, W)
    g_dev = np.fromfile(gbin, dtype=np.float32)
    n = min(len(g_ref), len(g_dev)); a, b = g_ref[:n], g_dev[:n]
    cos = float(a @ b / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-12))
    mae = float(np.mean(np.abs(a - b)))
    print(f"g_ref[0..4] = {np.round(g_ref[:5],4)}")
    print(f"g_dev[0..4] = {np.round(g_dev[:5],4)}")
    print(f"ref_enc cosine(torch, device) = {cos:.6f} | MAE = {mae:.3e} | dims {n}")
    print("RESULT:", "PASS" if cos > 0.999 else ("CLOSE" if cos > 0.99 else "FAIL"))


if __name__ == "__main__":
    main()
