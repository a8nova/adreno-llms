#!/usr/bin/env python
"""Streaming TTS driver: chunk text -> persistent device server -> pipelined playback.

Chunk N plays on the host while chunk N+1 synthesizes on the device.
Accepts MULTIPLE texts in one invocation — the phonemizer import and the
device server boot are paid once (like a real app), so per-text TTFA is the
steady-state number.

Usage: stream.py "First utterance." "Second utterance." [--no-play]
"""
import json, os, re, struct, subprocess, sys, threading, time
from pathlib import Path

PORT_DIR = Path(__file__).resolve().parent.parent
REMOTE = "/data/local/tmp/Kokoro_82M_inference"
SR = 24000

def chunk_text(text, first_max_words=5, max_words=14):
    words = text.strip().split()
    # First chunk: hard-split after first_max_words (or earlier at punctuation)
    # — time-to-first-audio is chunk-0 synth time, so keep it tiny.
    first_n = len(words)
    for i, w in enumerate(words[:first_max_words]):
        if w.endswith((',', '.', '!', '?', ';', ':')):
            first_n = i + 1
            break
    else:
        first_n = min(first_max_words, len(words))
    chunks = [" ".join(words[:first_n])]
    rest = " ".join(words[first_n:])
    if not rest:
        return chunks
    # Remaining text: clause/sentence units merged up to max_words.
    # A punctuation-free span longer than max_words is hard-split at word
    # boundaries — one oversized chunk at the tail is an underrun bomb (its
    # synth time exceeds all the audio buffered ahead of it at RTF ~1.0).
    parts = []
    for p in re.split(r'(?<=[.!?;:,])\s+', rest):
        p = p.strip()
        if not p:
            continue
        pw = p.split()
        if len(pw) > max_words:
            n_pieces = -(-len(pw) // max_words)          # ceil
            per = -(-len(pw) // n_pieces)                # balanced pieces
            parts.extend(" ".join(pw[i:i+per]) for i in range(0, len(pw), per))
        else:
            parts.append(p)
    cur = ""
    for p in parts:
        # Graduated cap: early chunks stay short so the playback buffer builds
        # faster than the schedule consumes it (chunk N+1 longer than chunk N
        # is what forces jitter-buffer holds at RTF ~1.0).
        cap = min(max_words, 4 + 4 * len(chunks))
        cand = (cur + " " + p).strip()
        if cur and len(cand.split()) > cap:
            chunks.append(cur); cur = p
        else:
            cur = cand
        if cur.endswith(('.', '!', '?')) and len(cur.split()) >= 4:
            chunks.append(cur); cur = ""
    if cur: chunks.append(cur)
    return chunks

class Player(threading.Thread):
    def __init__(self):
        super().__init__(daemon=True)
        self.q, self.cv = [], threading.Condition()
        self.done = False
        self.first_play_t = None
        self.underrun = 0.0
        self._last_end = None
        # Jitter buffer: playback of the FIRST wav holds until this monotonic
        # time. A short hold up front is inaudible; the same deficit as
        # mid-speech silence is jarring. Set via hold_until() after chunk 0's
        # duration calibrates the schedule.
        self._start_at = 0.0
    def hold_until(self, t_mono):
        with self.cv:
            self._start_at = max(self._start_at, t_mono); self.cv.notify()
    def add(self, path):
        with self.cv:
            self.q.append(path); self.cv.notify()
    def finish(self):
        with self.cv:
            self.done = True; self.cv.notify()
    def run(self):
        while True:
            with self.cv:
                while not self.q and not self.done: self.cv.wait()
                if not self.q and self.done: return
                path = self.q.pop(0)
            if self.first_play_t is None:
                dt = self._start_at - time.monotonic()
                if dt > 0: time.sleep(dt)
            t = time.monotonic()
            if self.first_play_t is None: self.first_play_t = t
            elif self._last_end is not None and t > self._last_end + 0.02:
                self.underrun += t - self._last_end
            subprocess.run(["afplay", str(path)])
            self._last_end = time.monotonic()

def main():
    texts = [a for a in sys.argv[1:] if not a.startswith("--")] or [
        "The teacher worked at the school for many years. She loved teaching children to read, and every morning she arrived early to prepare her classroom."]
    play = "--no-play" not in sys.argv
    t_start = time.monotonic()

    # One warm pipeline for ALL texts: misaki import + server boot are paid
    # once (like a real app); per-text TTFA below is the steady-state number.
    from misaki import en
    g2p = en.G2P()
    cfg = json.load(open(PORT_DIR / "model_info/config.json"))
    vocab = cfg["vocab"]
    print(f"[stream] phonemizer warm @ {time.monotonic() - t_start:.2f}s (one-time)")

    # Forward host NNOPT_* env to the device server so A/B toggles
    # (NNOPT_DOT8=..., NNOPT_HT_WAVE=..., ...) work through the streamer.
    nnopt_env = " ".join(f"{k}={v}" for k, v in sorted(os.environ.items())
                         if k.startswith("NNOPT_"))
    srv = subprocess.Popen(
        ["adb", "shell",
         f"cd {REMOTE} && NNOPT_GPU_FP32_GENERATOR=1 {nnopt_env} "
         f"LD_LIBRARY_PATH={REMOTE}/lib:/system/vendor/lib64:$LD_LIBRARY_PATH "
         f"./Kokoro_82M_inference_fp16 --serve"],
        stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL,
        text=True, bufsize=1)
    assert srv.stdout.readline().strip() == "SERVER_READY"
    print(f"[stream] server ready @ {time.monotonic() - t_start:.2f}s (one-time)")
    tmp = Path("/tmp/kokoro_stream"); tmp.mkdir(exist_ok=True)

    seq = 0
    for ti, text in enumerate(texts):
        t_text = time.monotonic()
        chunks = chunk_text(text)
        print(f"\n[stream] ▶ text {ti}: {len(chunks)} chunks: {chunks}")
        ids_list = []
        for c in chunks:
            ph, _ = g2p(c)
            ids_list.append([0] + [vocab[p] for p in ph if p in vocab] + [0])

        player = Player(); player.start()
        stats, ttfa = [], None
        for i, ids in enumerate(ids_list):
            p = tmp / f"ids_{seq}.bin"
            p.write_bytes(b"".join(struct.pack("<i", x) for x in ids))
            subprocess.run(["adb", "push", str(p), f"{REMOTE}/s_ids_{seq}.bin"],
                           capture_output=True, check=True)
            srv.stdin.write(f"SAY s_ids_{seq}.bin s_out_{seq}.wav\n"); srv.stdin.flush()
            line = srv.stdout.readline().strip()
            if not line.startswith("AUDIO_READY"):
                print(f"[stream] FAILED chunk {i}: {line}"); break
            _, rpath, n, synth = line.split()
            local = tmp / f"out_{seq}.wav"
            subprocess.run(["adb", "pull", f"{REMOTE}/{rpath}", str(local)], capture_output=True, check=True)
            if ttfa is None: ttfa = time.monotonic() - t_text
            audio_s = int(n) / SR
            stats.append((i, float(synth), audio_s))
            print(f"[stream]   chunk {i}: synth {float(synth):.2f}s for {audio_s:.2f}s audio "
                  f"(RTF {float(synth)/audio_s:.2f}) — \"{chunks[i]}\"")
            if i == 0 and play and len(ids_list) > 1:
                # Schedule playback start so no later chunk underruns:
                # chunk 0's measured duration calibrates sec/phoneme-id, every
                # remaining chunk's duration is estimated from its id count,
                # and synth wall is estimated at RTF_EST (measured marginal
                # ~1.0 + adb push/pull overhead). The worst cumulative
                # deficit max_k(Σ synth_est(1..k) − Σ audio_est(0..k-1))
                # becomes a hold BEFORE the first sample instead of silence
                # in the middle of speech.
                RTF_EST, MARGIN = 1.05, 0.25   # marginal RTF measures 0.96-1.01; 1.05 covers adb pull overhead
                spi = audio_s / max(len(ids_list[0]), 1)
                est = [audio_s] + [spi * len(x) for x in ids_list[1:]]
                deficit, synth_acc, audio_acc = 0.0, 0.0, 0.0
                for k in range(1, len(est)):
                    synth_acc += RTF_EST * est[k]
                    audio_acc += est[k - 1]
                    deficit = max(deficit, synth_acc - audio_acc)
                if deficit > 0:
                    print(f"[stream]   jitter-buffer hold: +{deficit + MARGIN:.2f}s before first sample")
                    player.hold_until(time.monotonic() + deficit + MARGIN)
            if play: player.add(local)
            seq += 1
        player.finish()
        if play: player.join()
        tot_audio = sum(a for _, _, a in stats)
        tot_synth = sum(s for _, s, a in stats)
        if play and player.first_play_t is not None:
            ttfa = player.first_play_t - t_text   # actual first sample (incl. jitter hold)
        print(f"[stream] text {ti}: TTFA {ttfa:.2f}s · audio {tot_audio:.2f}s · synth {tot_synth:.2f}s "
              f"· sustained RTF {tot_synth/max(tot_audio,1e-9):.2f}"
              + (f" · underrun {player.underrun:.2f}s" if play else ""))

    srv.stdin.write("QUIT\n"); srv.stdin.flush(); srv.wait(timeout=10)

if __name__ == "__main__":
    main()
