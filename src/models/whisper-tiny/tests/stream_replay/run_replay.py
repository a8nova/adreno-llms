#!/usr/bin/env python3
"""Deterministic streaming replay harness for whisper-tiny `--stream`.

Reproduces EXACTLY what the see-and-say app does to the engine — paced
16 kHz mono float32 PCM on stdin (100 ms chunks, same as MicCapture), same
CLI flags as WhisperSession.start() — but from canned LibriSpeech clips, so
streaming behaviour is testable without a human talking into a phone.

Per scenario it captures a TIMESTAMPED event log (PARTIAL / FINAL /
STREAM_TIMING / STREAM_SEG / RESET markers) as JSONL. Those logs are:
  1. checked here for engine-level invariants (see check_events.py), and
  2. replayed through the app's TranscriptReducer in JVM unit tests
     (examples/see-and-say/app/src/test) to catch UI-fold bugs —
     duplication, ghosting across stop/start, partial churn.

Usage:
  python3 run_replay.py                 # run all scenarios
  python3 run_replay.py short pauses    # run a subset
  python3 run_replay.py --list

Outputs: out/events_<scenario>.jsonl  +  out/batch_ref_<clip>.txt
"""
import json
import os
import re
import subprocess
import sys
import threading
import time

# PARTIAL [0.00-5.10]: text  /  FINAL [0.00-6.60]: text — [t0-t1] is the
# phrase's position in the global stdin stream (seconds since first sample).
EVENT_RX = re.compile(r"^(PARTIAL|FINAL) \[([0-9.]+)-([0-9.]+)\]: ?(.*)$")

HERE = os.path.dirname(os.path.abspath(__file__))
AUDIO = os.path.join(HERE, "audio")
OUT = os.path.join(HERE, "out")

REMOTE_DIR = "/data/local/tmp/whisper_tiny_inference"
BINARY = "whisper_tiny_inference_fp16"   # same fp16 build the app ships as libwhisper.so

SR = 16000
CHUNK_SAMPLES = 1600          # 100 ms — matches MicCapture.CHUNK
CHUNK_BYTES = CHUNK_SAMPLES * 4

# Flags must mirror WhisperSession.start() defaults — we are testing the app's config.
STREAM_ARGS = "transcribe 96 --stream --vad-threshold 0.006 --step-ms 700 --hangover-ms 1400"

# Scenario = list of segments:
#   ("file", name)       paced 1x real-time from audio/<name>
#   ("sil", seconds)     paced silence (gap in speech while mic is open)
#   ("burst_sil", secs)  silence written in ONE write — mimics
#                        WhisperSession.flushSilence() after Stop is tapped
#   ("reset", None)      marker only: the app cleared its transcript state here
#                        (start of a new recording session). Engine sees nothing.
SCENARIOS = {
    # Single short utterance then silence: 1 phrase, hangover-committed FINAL.
    "short": [
        ("file", "sample_00_audio.bin"),       # 5.9 s
        ("sil", 2.5),
    ],
    # Two utterances split by a natural pause: multi-FINAL boundaries.
    "pauses": [
        ("file", "sample_00_audio.bin"),       # 5.9 s
        ("sil", 1.8),
        ("file", "sample_01_audio.bin"),       # 4.8 s
        ("sil", 2.5),
    ],
    # ~29 s continuous read speech: soft/force segmentation, long partials.
    "long": [
        ("file", "sample_04_audio.bin"),       # 29.4 s
        ("sil", 2.5),
    ],
    # Stop/start race: Stop (burst silence flush) then IMMEDIATELY a new
    # session. The closing FINAL of utterance 1 is still being transcribed
    # when the app resets — this is the ghosting repro.
    "stopstart": [
        ("file", "sample_01_audio.bin"),       # 4.8 s
        ("burst_sil", 1.7),                    # flushSilence(): hangover+300ms, one write
        ("reset", None),                       # user tapped Start again right away
        ("file", "sample_02_audio.bin"),       # 12.5 s
        ("sil", 2.5),
    ],
}

# Engine batch reference (--audio path) for FINAL-vs-batch comparison.
BATCH_CLIPS = {
    "sample_00_audio.bin": 5.855,
    "sample_01_audio.bin": 4.815,
    "sample_02_audio.bin": 12.485,
    "sample_04_audio.bin": 29.4,
}


def wait_device_idle(poll_s=20, timeout_s=3600):
    """The phone is SHARED across sessions (other model ports run on it too).
    Two GPU processes contend for the Adreno and can wedge the driver — never
    start while anything else is running; poll until the device is idle."""
    deadline = time.monotonic() + timeout_s
    while True:
        out = subprocess.run(
            ["adb", "shell", "ps -A | grep -iE 'inference|musicgen' | grep -v grep"],
            capture_output=True,
        ).stdout.decode("utf-8", "replace").strip()
        if not out:
            return
        if time.monotonic() > deadline:
            raise RuntimeError(f"device still busy after {timeout_s}s:\n{out}")
        print(f"[busy] another process is on the device — waiting {poll_s}s:\n  "
              + out.splitlines()[0])
        time.sleep(poll_s)


def adb_stream_cmd():
    env = f"LD_LIBRARY_PATH={REMOTE_DIR}/lib:/system/vendor/lib64:$LD_LIBRARY_PATH NNOPT_DEBUG_LAYERS=0"
    # 2>&1 so STREAM:/STREAM_TIMING/STREAM_SEG (stderr) land in the same
    # ordered timeline as PARTIAL:/FINAL: (stdout).
    return ["adb", "shell", "-T",
            f"cd {REMOTE_DIR} && {env} ./{BINARY} {STREAM_ARGS} 2>&1"]


def run_scenario(name, segments):
    wait_device_idle()
    os.makedirs(OUT, exist_ok=True)
    events = []
    ready = threading.Event()
    t0 = [None]  # set when feeding starts

    proc = subprocess.Popen(adb_stream_cmd(), stdin=subprocess.PIPE,
                            stdout=subprocess.PIPE, bufsize=0)

    def reader():
        for raw in proc.stdout:
            line = raw.decode("utf-8", "replace").rstrip("\n")
            now = time.monotonic()
            t_ms = int((now - t0[0]) * 1000) if t0[0] is not None else -1
            ev = {"t_ms": t_ms, "raw": line}
            m = EVENT_RX.match(line)
            if m:
                ev.update(type=m.group(1), t0=float(m.group(2)),
                          t1=float(m.group(3)), text=m.group(4).strip())
            elif line.startswith("STREAM_TIMING"):
                ev.update(type="TIMING")
            elif line.startswith("STREAM_SEG"):
                ev.update(type="SEG")
            elif line.startswith("STREAM: ready"):
                ev.update(type="READY")
                ready.set()
            else:
                ev.update(type="LOG")
            events.append(ev)
            tag = ev["type"]
            if tag in ("PARTIAL", "FINAL", "SEG", "READY"):
                print(f"  [{t_ms:>7}ms] {line[:110]}")

    th = threading.Thread(target=reader, daemon=True)
    th.start()

    print(f"[{name}] waiting for STREAM: ready (first run pays the JIT)...")
    if not ready.wait(timeout=120):
        proc.kill()
        raise RuntimeError("engine never printed 'STREAM: ready'")

    t0[0] = time.monotonic()
    next_deadline = t0[0]
    fed_samples = 0   # mirrors WhisperSession's fed-sample counter (the fence basis)

    def pace_chunks(data):
        """Write `data` in 100 ms chunks at 1x real time with drift correction."""
        nonlocal next_deadline, fed_samples
        for off in range(0, len(data), CHUNK_BYTES):
            chunk = data[off:off + CHUNK_BYTES]
            proc.stdin.write(chunk)
            proc.stdin.flush()
            fed_samples += len(chunk) // 4
            next_deadline += 0.100
            delay = next_deadline - time.monotonic()
            if delay > 0:
                time.sleep(delay)

    for kind, arg in segments:
        t_ms = int((time.monotonic() - t0[0]) * 1000)
        if kind == "file":
            with open(os.path.join(AUDIO, arg), "rb") as f:
                data = f.read()
            print(f"[{name}] {t_ms}ms feeding {arg} ({len(data)//4/SR:.1f}s paced)")
            pace_chunks(data)
        elif kind == "sil":
            print(f"[{name}] {t_ms}ms feeding {arg}s paced silence")
            pace_chunks(b"\x00" * int(arg * SR) * 4)
        elif kind == "burst_sil":
            print(f"[{name}] {t_ms}ms BURST silence {arg}s (flushSilence)")
            proc.stdin.write(b"\x00" * int(arg * SR) * 4)
            proc.stdin.flush()
            fed_samples += int(arg * SR)
            next_deadline = time.monotonic()  # re-anchor pacing after the burst
        elif kind == "reset":
            fed_s = fed_samples / SR
            print(f"[{name}] {t_ms}ms RESET marker (app cleared transcript, fence={fed_s:.2f}s)")
            events.append({"t_ms": t_ms, "type": "RESET", "fed_s": round(fed_s, 2),
                           "raw": "<app reset>"})
        else:
            raise ValueError(kind)

    # EOF -> engine flushes any tail as a closing FINAL and exits.
    proc.stdin.close()
    proc.wait(timeout=120)
    th.join(timeout=10)

    path = os.path.join(OUT, f"events_{name}.jsonl")
    with open(path, "w") as f:
        for ev in sorted(events, key=lambda e: e["t_ms"]):
            f.write(json.dumps(ev) + "\n")
    n_p = sum(1 for e in events if e["type"] == "PARTIAL")
    n_f = sum(1 for e in events if e["type"] == "FINAL")
    print(f"[{name}] done: {n_p} partials, {n_f} finals -> {path}\n")
    return path


def batch_reference(clip, dur):
    """Run the engine's validated batch path on the same clip — the transcript
    the streaming FINALs should converge to (same model, same weights)."""
    wait_device_idle()
    os.makedirs(OUT, exist_ok=True)
    out_path = os.path.join(OUT, f"batch_ref_{clip.replace('_audio.bin', '')}.txt")
    env = f"LD_LIBRARY_PATH={REMOTE_DIR}/lib:/system/vendor/lib64:$LD_LIBRARY_PATH NNOPT_DEBUG_LAYERS=0"
    cmd = ["adb", "shell", "-T",
           f"cd {REMOTE_DIR} && {env} ./{BINARY} transcribe 128 --audio bench/{clip} "
           f"--audio-seconds {dur} 2>/dev/null"]
    txt = subprocess.run(cmd, capture_output=True, timeout=300).stdout.decode("utf-8", "replace").strip()
    with open(out_path, "w") as f:
        f.write(txt + "\n")
    print(f"[batch] {clip}: {txt[:100]}")
    return out_path


if __name__ == "__main__":
    args = [a for a in sys.argv[1:] if not a.startswith("-")]
    if "--list" in sys.argv:
        print("\n".join(SCENARIOS))
        sys.exit(0)
    names = args or list(SCENARIOS)
    for clip, dur in BATCH_CLIPS.items():
        batch_reference(clip, dur)
    for n in names:
        run_scenario(n, SCENARIOS[n])
