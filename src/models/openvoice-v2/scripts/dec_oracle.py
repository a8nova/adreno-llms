#!/usr/bin/env python3
"""Numpy reference of the HiFiGAN decoder (fp32) — localizes the C++ device bug.
Loads fp16 weights from weights/model.fp16.bin via meta offsets, folds weight_norm,
runs dec on reference/layers/flow_output.bin, dumps every stage at FULL resolution
to layer_dumps/oracle_*.npy, and prints cosine vs reference + vs device dumps.
"""
import json, struct, math, sys
import numpy as np

ROOT = "."
META = json.load(open(f"{ROOT}/weights/model.fp16.meta.json"))
TS = META["tensors"]
BIN = open(f"{ROOT}/weights/model.fp16.bin", "rb").read()

def w(key):
    t = TS[key]; off = t["offset"]; n = t["num_elements"]
    a = np.frombuffer(BIN[off:off + n*2], dtype=np.float16).astype(np.float32)
    return a.reshape(t["shape"])

def has(key): return key in TS

def weight(base):
    if has(base + ".weight"):
        return w(base + ".weight")
    v = w(base + ".weight_v"); g = w(base + ".weight_g")  # weight_norm, dim=0
    d0 = v.shape[0]
    norm = np.sqrt((v.reshape(d0, -1)**2).sum(1)) + 1e-12
    scale = (g.reshape(d0) / norm).reshape([d0] + [1]*(v.ndim-1))
    return v * scale

def bias(base):
    return w(base + ".bias") if has(base + ".bias") else None

def conv1d(x, base, stride=1, dil=1, pad=0):
    # x [Cin, T]; weight [Cout, Cin, K]
    W = weight(base); b = bias(base)
    Cout, Cin, K = W.shape
    T = x.shape[1]
    xp = np.pad(x, ((0,0),(pad,pad)))
    Tout = (xp.shape[1] - dil*(K-1) - 1)//stride + 1
    out = np.zeros((Cout, Tout), np.float32)
    for k in range(K):
        xs = xp[:, k*dil : k*dil + (Tout-1)*stride + 1 : stride]  # [Cin, Tout]
        out += W[:, :, k] @ xs
    if b is not None: out += b.reshape(-1,1)
    return out

def conv_transpose1d(x, base, stride, pad):
    # x [Cin,T]; weight [Cin,Cout,K]
    W = weight(base); b = bias(base)
    Cin, Cout, K = W.shape
    T = x.shape[1]
    Tout = (T-1)*stride - 2*pad + K
    full = np.zeros((Cout, (T-1)*stride + K), np.float32)
    for k in range(K):
        contrib = W[:, :, k].T @ x          # [Cout, T]
        # place at positions k + i*stride
        full[:, k : k + (T-1)*stride + 1 : stride] += contrib
    out = full[:, pad:pad+Tout]
    if b is not None: out += b.reshape(-1,1)
    return out

def lrelu(x, s): return np.where(x >= 0, x, s*x)

def resblock(x, base, K):
    dil = [1,3,5]
    for d in range(3):
        xt = lrelu(x, 0.1)
        xt = conv1d(xt, f"{base}.convs1.{d}", dil=dil[d], pad=dil[d]*(K-1)//2)
        xt = lrelu(xt, 0.1)
        xt = conv1d(xt, f"{base}.convs2.{d}", dil=1, pad=(K-1)//2)
        x = xt + x
    return x

def dec(z, g):
    stages = {}
    x = conv1d(z, "dec.conv_pre", pad=3); stages["dec_conv_pre"] = x.copy()
    cond = conv1d(g.reshape(-1,1), "dec.cond")   # [512,1]
    x = x + cond
    up_k=[16,16,4,4]; up_s=[8,8,2,2]; up_pad=[4,4,1,1]; rk=[3,7,11]
    for i in range(4):
        x = lrelu(x, 0.1)
        x = conv_transpose1d(x, f"dec.ups.{i}", up_s[i], up_pad[i]); stages[f"dec_ups_{i}"]=x.copy()
        acc = None
        for j in range(3):
            rb = resblock(x, f"dec.resblocks.{i*3+j}", rk[j])
            acc = rb if acc is None else acc + rb
        x = acc/3
    x = lrelu(x, 0.01); stages["dec_pre_post"]=x.copy()
    x = conv1d(x, "dec.conv_post", pad=3); stages["dec_conv_post"]=x.copy()
    x = np.tanh(x); stages["dec"]=x.copy()
    return stages

def read_ref(name, dt='float32'):
    p = f"{ROOT}/reference/layers/{name}_output.bin"
    d = open(p,'rb').read()
    return np.frombuffer(d, dtype=np.float16 if dt=='float16' else np.float32).astype(np.float32)

def read_dev(name):
    p = f"{ROOT}/layer_dumps/{name}__pass0.bin"
    meta = json.load(open(p+".meta.json"))
    d = open(p,'rb').read()
    dt = np.float16 if meta["dtype"]=="float16" else np.float32
    return np.frombuffer(d, dtype=dt).astype(np.float32)

def cos(a, b):
    n = min(len(a), len(b)); a=a[:n]; b=b[:n]
    return float(a@b / (np.linalg.norm(a)*np.linalg.norm(b) + 1e-12))

if __name__ == "__main__":
    z = read_ref("flow").reshape(192, -1)
    g = read_ref("flow")  # placeholder; load g_tgt below
    g = np.fromfile(f"{ROOT}/assets/g_tgt.bin", dtype=np.float32)
    import os
    if os.environ.get("ZERO_G", "1") == "1":   # converter config has zero_g=True
        g = np.zeros_like(g); print("ZERO_G: dec g set to zeros")
    print(f"z_hat {z.shape}, g {g.shape}")
    st = dec(z, g)
    print(f"{'stage':16s} {'oracle_vs_ref':>14s} {'oracle_vs_device':>18s}")
    for name, arr in st.items():
        flat = arr.reshape(-1)
        np.save(f"{ROOT}/layer_dumps/oracle_{name}.npy", flat)
        try: cr = cos(flat, read_ref(name))
        except Exception as e: cr = float('nan')
        try: cd = cos(flat, read_dev(name))
        except Exception as e: cd = float('nan')
        print(f"{name:16s} {cr:14.4f} {cd:18.4f}")
