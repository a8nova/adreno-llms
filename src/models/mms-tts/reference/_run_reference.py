#!/usr/bin/env python3
"""TTS reference template for VITS-family models (MMS-TTS).

Emits deterministic fixtures + per-op tensors for the 7-step VITS graph.

Reference: model_info/transformers_src/modeling_vits.py and docs/MODALITY_TTS.md
"""

import os, sys, json

def _progress(msg: str):
    sys.stderr.write(f"PROGRESS: {msg}\n")
    sys.stderr.flush()

def _write_bin(path: str, arr):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    arr.tofile(path)


def main():
    model_id = sys.argv[1]
    project_dir = sys.argv[2]

    ref_dir = os.path.join(project_dir, "reference")
    layers_dir = os.path.join(ref_dir, "layers")
    assets_dir = os.path.join(project_dir, "assets")
    os.makedirs(layers_dir, exist_ok=True)
    os.makedirs(assets_dir, exist_ok=True)

    try:
        import numpy as np
        import torch
        from transformers import AutoTokenizer, VitsModel
    except Exception as e:
        print(json.dumps({"success": False, "error": f"Import failed: {e}"}))
        return

    # Prompt from contract
    prompt = "Hello, my name is"
    try:
        cpath = os.path.join(project_dir, ".nnport", "contract.json")
        if os.path.exists(cpath):
            c = json.load(open(cpath))
            prompt = c.get("reference_protocol", {}).get("input_fixture", {}).get("prompt", prompt)
    except Exception:
        pass

    _progress(f"Loading VITS model: {model_id}")
    model = VitsModel.from_pretrained(model_id, torch_dtype=torch.float32)
    model.eval()
    tok = AutoTokenizer.from_pretrained(model_id)

    tok_out = tok(prompt, return_tensors="pt")
    input_ids = tok_out["input_ids"]
    # VITS' VitsTextEncoder uses padding_mask as a MULTIPLICATIVE mask
    # against hidden_states, shape [B, T, 1] (float), where 1.0 marks a
    # valid token and 0.0 marks padding. This matches the same orientation
    # as the tokenizer's attention_mask, NOT its negation. The unsqueeze(-1)
    # gives the broadcast shape for `hidden_states * padding_mask` where
    # hidden_states is [B, T, hidden]. Works on every transformers version
    # that requires padding_mask (older versions ignore the unused kwarg).
    _attn = tok_out.get("attention_mask", torch.ones_like(input_ids))
    _padding_mask = _attn.unsqueeze(-1).to(torch.float32)
    input_ids_np = input_ids[0].to(torch.int32).cpu().numpy()
    _write_bin(os.path.join(assets_dir, "test_input_ids.bin"), input_ids_np)

    B = 1
    T_chars = int(input_ids.shape[1])
    hidden = int(model.config.hidden_size)

    # Deterministic RNG fixtures
    torch.manual_seed(0)
    duration_noise = torch.randn((B, 2, T_chars), dtype=torch.float32)
    _write_bin(os.path.join(assets_dir, "duration_noise.bin"), duration_noise.cpu().numpy().astype("float32"))

    # Step 1: text encoder. Pass padding_mask only if the model's signature
    # requires it (newer transformers); older versions silently ignore the
    # kwarg via **kwargs but some intermediates raise TypeError.
    import inspect as _inspect
    _te_sig = _inspect.signature(model.text_encoder.forward)
    with torch.no_grad():
        if "padding_mask" in _te_sig.parameters:
            enc_out = model.text_encoder(input_ids=input_ids, padding_mask=_padding_mask)
        else:
            enc_out = model.text_encoder(input_ids=input_ids)
        encoder_hidden = enc_out.last_hidden_state  # [B,T,H]
        stats = model.text_encoder.proj(encoder_hidden)  # [B,T,2H]

    _write_bin(os.path.join(layers_dir, "text_encoder_out_output.bin"), encoder_hidden[0].cpu().numpy().astype("float32"))
    _write_bin(os.path.join(layers_dir, "text_encoder_stats_output.bin"), stats[0].cpu().numpy().astype("float32"))

    # Step 2: duration predictor
    with torch.no_grad():
        log_durations = model.duration_predictor(encoder_hidden, duration_noise)
    logdur_np = log_durations[0].cpu().numpy().astype("float32")
    _write_bin(os.path.join(layers_dir, "log_durations_output.bin"), logdur_np)

    # Step 3: host durations + char_idx
    speaking_rate = float(getattr(model.config, "speaking_rate", 1.0))
    durations = np.ceil(np.exp(logdur_np) * speaking_rate).astype("int32")
    durations[durations < 0] = 0
    T_frames = int(durations.sum())
    char_idx = np.zeros((T_frames,), dtype="int32")
    cur = 0
    for i in range(T_chars):
        d = int(durations[i])
        if d <= 0:
            continue
        char_idx[cur : cur + d] = i
        cur += d

    # Step 4: length regulator (reference-side gather)
    stats_np = stats[0].cpu().numpy().astype("float32")
    expanded_stats = np.zeros((B, T_frames, stats_np.shape[-1]), dtype="float32")
    expanded_stats[0, :, :] = stats_np[char_idx, :]
    _write_bin(os.path.join(layers_dir, "expanded_stats_output.bin"), expanded_stats[0])

    # Step 5: sample prior
    torch.manual_seed(1)
    prior_noise = torch.randn((B, hidden, T_frames), dtype=torch.float32)
    _write_bin(os.path.join(assets_dir, "prior_noise.bin"), prior_noise.cpu().numpy().astype("float32"))

    mean = torch.from_numpy(expanded_stats).to(torch.float32)[..., :hidden]
    log_scale = torch.from_numpy(expanded_stats).to(torch.float32)[..., hidden:]
    noise_scale = float(getattr(model.config, "noise_scale", 0.667))
    z_prior = (mean + torch.exp(log_scale) * prior_noise.permute(0, 2, 1) * noise_scale).permute(0, 2, 1)
    _write_bin(os.path.join(layers_dir, "z_prior_output.bin"), z_prior[0].cpu().numpy().astype("float32"))

    # Step 6: flow inverse
    with torch.no_grad():
        z_latent = model.flow(z_prior, reverse=True)
    _write_bin(os.path.join(layers_dir, "z_latent_output.bin"), z_latent[0].cpu().numpy().astype("float32"))

    # Step 7: vocoder
    with torch.no_grad():
        wav = model.decoder(z_latent)  # [B,T_audio]
    wav_np = wav[0].cpu().numpy().astype("float32")
    _write_bin(os.path.join(layers_dir, "waveform_output.bin"), wav_np)

    # Write wav using soundfile if available
    sr = int(getattr(model.config, "sampling_rate", 16000))
    try:
        import soundfile as sf
        os.makedirs(ref_dir, exist_ok=True)
        sf.write(os.path.join(ref_dir, "output.wav"), wav_np, sr)
    except Exception as e:
        _progress(f"WARNING: soundfile unavailable ({e}); wrote waveform_output.bin only")

    # Reference manifest (minimal)
    manifest = {
        "_nnport_capture_version": "mms_tts_v1",
        "_captured_layers": [
            "text_encoder_out",
            "text_encoder_stats",
            "log_durations",
            "expanded_stats",
            "z_prior",
            "z_latent",
            "waveform",
        ],
        "layers": {
            "text_encoder_out": {"output_shape": [B, T_chars, hidden], "output_dtype": "float32"},
            "text_encoder_stats": {"output_shape": [B, T_chars, hidden * 2], "output_dtype": "float32"},
            "log_durations": {"output_shape": [B, T_chars], "output_dtype": "float32"},
            "expanded_stats": {"output_shape": [B, T_frames, hidden * 2], "output_dtype": "float32"},
            "z_prior": {"output_shape": [B, hidden, T_frames], "output_dtype": "float32"},
            "z_latent": {"output_shape": [B, hidden, T_frames], "output_dtype": "float32"},
            "waveform": {"output_shape": [B, int(wav_np.shape[0])], "output_dtype": "float32"},
        },
    }
    with open(os.path.join(layers_dir, "manifest.json"), "w") as f:
        json.dump(manifest, f, indent=2)

    # Also write reference_text/tokens placeholders (not used by TTS Evaluate)
    with open(os.path.join(ref_dir, "reference_text.txt"), "w") as f:
        f.write("(tts)\n")
    with open(os.path.join(ref_dir, "reference_tokens.json"), "w") as f:
        json.dump({"produced_by": "_run_reference.py", "prompt": prompt, "input_ids": input_ids_np.tolist(), "generated_ids": []}, f)

    print(json.dumps({"success": True, "T_chars": T_chars, "T_frames": T_frames, "T_audio": int(wav_np.shape[0])}))


if __name__ == "__main__":
    main()
