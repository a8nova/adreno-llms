#!/usr/bin/env bash
# run_device_reference.sh <model-key> [--capture] [--no-deploy] [--no-build]
#
# On-device regression test against an fp16 GOLDEN snapshot (the current known-good
# device output). Compares the model binary's output on the attached Adreno device
# to tests/device_golden/<key>.json and fails on any deviation — so a future engine
# change that alters a model's output is caught in CI.
#
#   --capture   mint/refresh the golden from the current device output (run once,
#               on a known-good build, then commit tests/device_golden/<key>.json)
#   --no-deploy skip adb deploy (binary+weights already on device)
#   --no-build  skip the fp16 release rebuild
#
# Exit 0 = match (or captured), 1 = mismatch/error, 2 = modality not yet implemented.
# Determinism: text/vlm/asr feed fixed input + greedy (--top-k 1). The PyTorch
# fp32 reference_tokens.json is kept as correctness documentation, NOT the gate
# (fp16 device output legitimately diverges from fp32 — see docs/TESTING.md).
set -euo pipefail

KEY="${1:?usage: run_device_reference.sh <model-key> [--capture] [--no-deploy] [--no-build]}"; shift || true
CAPTURE=0; DO_DEPLOY=1; DO_BUILD=1
for a in "$@"; do case "$a" in
  --capture) CAPTURE=1;; --no-deploy) DO_DEPLOY=0;; --no-build) DO_BUILD=0;;
esac; done

ROOT="$(cd "$(dirname "$0")/../.." && pwd)"

# Resolve a model's on-device REMOTE_DIR: prefer scripts/remote_dir.env, else the
# inline default in run_android.sh.
adb_remote_dir() {
  local d="$ROOT/src/models/$1"
  if [ -f "$d/scripts/remote_dir.env" ]; then
    ( NNOPT_DTYPE=fp16; . "$d/scripts/remote_dir.env" >/dev/null 2>&1; printf '%s' "$REMOTE_DIR" )
  else
    grep -E 'REMOTE_DIR="?\$\{REMOTE_DIR:-' "$d/scripts/run_android.sh" 2>/dev/null | head -1 \
      | sed -E 's/.*REMOTE_DIR:-([^}"]*)\}?"?.*/\1/'
  fi
}
MD="$ROOT/src/models/$KEY"
REF="$MD/reference"
GOLDEN_DIR="$ROOT/tests/device_golden"
GOLDEN="$GOLDEN_DIR/$KEY.json"
[ -d "$MD" ] || { echo "FAIL: no model dir $MD"; exit 1; }
mkdir -p "$GOLDEN_DIR"
export ANDROID_NDK="${ANDROID_NDK:-$HOME/Library/Android/sdk/ndk/android-ndk-r26d}"
export NNOPT_DTYPE=fp16

case "$KEY" in
  granite-4-0-350m|lfm2-5-350m|mamba-130m|mamba2-130m|openelm-270m|qwen2-5-0-5b|smollm2-135m-instruct) MODALITY=text ;;
  lfm2-5-vl-450m|smolvlm-256m-instruct) MODALITY=vlm ;;
  whisper-tiny|seamless-m4t-unity-small) MODALITY=asr ;;
  pocket-tts|kokoro-82m|mms-tts|openvoice-v2) MODALITY=tts ;;
  musicgen-small) MODALITY=music ;;
  *) echo "FAIL: unknown model key $KEY"; exit 1 ;;
esac
echo "== $KEY (modality=$MODALITY) $([ "$CAPTURE" = 1 ] && echo "[capture]") =="

adb get-state >/dev/null 2>&1 || { echo "FAIL: no adb device attached"; exit 1; }

if [ "$DO_BUILD" = 1 ]; then
  echo "-- build (fp16 release) --"; ( cd "$MD" && ./scripts/build.sh --release ) >/tmp/ref_build_$KEY.log 2>&1 \
    || { echo "FAIL: build (see /tmp/ref_build_$KEY.log)"; tail -5 /tmp/ref_build_$KEY.log; exit 1; }
fi
if [ "$DO_DEPLOY" = 1 ]; then
  echo "-- deploy --"; ( cd "$MD" && ./scripts/deploy_android.sh ) >/tmp/ref_deploy_$KEY.log 2>&1 \
    || { echo "FAIL: deploy (see /tmp/ref_deploy_$KEY.log)"; tail -5 /tmp/ref_deploy_$KEY.log; exit 1; }
fi

case "$MODALITY" in
  text)
    TOKBIN="$REF/test_input_ids.bin"; RJSON="$REF/reference_tokens.json"
    [ -f "$TOKBIN" ] && [ -f "$RJSON" ] || { echo "SKIP: missing $TOKBIN or $RJSON"; exit 2; }
    MAXNEW=$(python3 -c "import json;d=json.load(open('$RJSON'));print(len(d['generated_ids'])-len(d['input_ids']))")
    LOG=/tmp/ref_run_$KEY.log
    echo "-- run (greedy, --token-ids, max_new=$MAXNEW) --"
    NNOPT_DEBUG_LAYERS=1 "$MD/scripts/run_android.sh" "ref" "$MAXNEW" --token-ids "$TOKBIN" --top-k 1 >"$LOG" 2>&1 || true
    CAPTURE=$CAPTURE GOLDEN="$GOLDEN" python3 "$ROOT/scripts/ci/_cmp_tokens.py" "$LOG" text
    ;;

  asr)
    # ASR: encoder input features ship with the model (deployed test_input_features.bin);
    # prime the decoder with reference input_ids and greedily decode, then golden the
    # generated token ids.
    TOKBIN="$REF/test_input_ids.bin"
    [ -f "$TOKBIN" ] || { echo "SKIP: missing $TOKBIN"; exit 2; }
    MAXNEW="${ASR_MAX_TOKENS:-16}"
    LOG=/tmp/ref_run_$KEY.log
    echo "-- run (asr, --token-ids, greedy, max_new=$MAXNEW) --"
    NNOPT_DEBUG_LAYERS=1 "$MD/scripts/run_android.sh" "ref" "$MAXNEW" --token-ids "$TOKBIN" --top-k 1 >"$LOG" 2>&1 || true
    CAPTURE=$CAPTURE GOLDEN="$GOLDEN" python3 "$ROOT/scripts/ci/_cmp_text.py" "$LOG" asr
    ;;

  vlm)
    # VLM: feed a committed image fixture + greedy decode; golden the streamed caption text.
    IMG="$MD/fixtures/sample.jpg"
    [ -f "$IMG" ] || { echo "SKIP: missing image fixture $IMG"; exit 2; }
    MAXNEW="${VLM_MAX_TOKENS:-16}"
    LOG=/tmp/ref_run_$KEY.log
    # Not every VLM run script auto-uploads --image (lfm2-vl doesn't), so push the
    # fixture to the device ourselves and pass a device-relative path (cwd=REMOTE_DIR).
    RD=$(adb_remote_dir "$KEY"); IMGBASE=$(basename "$IMG")
    adb push "$IMG" "$RD/$IMGBASE" >/dev/null 2>&1 || { echo "FAIL: could not push image to $RD"; exit 1; }
    echo "-- run (vlm, --image $IMGBASE, greedy, max_new=$MAXNEW) --"
    NNOPT_DEBUG_LAYERS=1 "$MD/scripts/run_android.sh" "Describe the image." "$MAXNEW" --image "$IMGBASE" --top-k 1 >"$LOG" 2>&1 || true
    CAPTURE=$CAPTURE GOLDEN="$GOLDEN" python3 "$ROOT/scripts/ci/_cmp_vlm.py" "$LOG"
    ;;
  tts|music)
    # Capture the device's audio output to a local raw float file, then golden it
    # by a windowed-RMS fingerprint (tolerant of fp jitter, catches real changes).
    RD=$(adb_remote_dir "$KEY")
    LOCAL=/tmp/dev_audio_$KEY.f32
    rm -f "$LOCAL"
    echo "-- run (audio capture) --"
    case "$KEY" in
      pocket-tts)
        # writes out.bin (float32 LE) on device; deterministic with fixed frames+noise
        ( cd "$MD" && NNOPT_DTYPE=fp16 ./scripts/run_android.sh 60 0.7 ) >/tmp/ref_run_$KEY.log 2>&1 || true
        adb pull "$RD/out.bin" "$LOCAL" >/dev/null 2>&1 || { echo "FAIL: could not pull $RD/out.bin"; exit 1; }
        AUDIO_FMT=f32 ;;
      mms-tts|musicgen-small)
        # write output.wav on device; run script pulls it; convert wav->f32 below
        ( cd "$MD" && NNOPT_DTYPE=fp16 ./scripts/run_android.sh "reference test" ) >/tmp/ref_run_$KEY.log 2>&1 || true
        adb pull "$RD/output.wav" "/tmp/dev_audio_$KEY.wav" >/dev/null 2>&1 || { echo "FAIL: could not pull $RD/output.wav"; exit 1; }
        AUDIO_FMT=wav; LOCAL=/tmp/dev_audio_$KEY.wav ;;
      *)
        echo "SKIP: audio capture for $KEY not wired yet (streams stdout — needs exec-out path)"; exit 2 ;;
    esac
    CAPTURE=$CAPTURE GOLDEN="$GOLDEN" python3 - "$LOCAL" "$AUDIO_FMT" <<'PY'
import sys,os,json,struct,array,math
path,fmt=sys.argv[1],sys.argv[2]
raw=open(path,'rb').read()
if fmt=='wav':
    # minimal PCM WAV parse: find 'data' chunk, assume 16-bit mono LE
    i=raw.find(b'data'); n=struct.unpack('<I',raw[i+4:i+8])[0]; pcm=raw[i+8:i+8+n]
    s=array.array('h'); s.frombytes(pcm[:len(pcm)//2*2]); x=[v/32768.0 for v in s]
else:
    a=array.array('f'); a.frombytes(raw[:len(raw)//4*4]); x=list(a)
N=len(x)
W=32  # windows
def fp(x):
    if not x: return []
    step=max(1,len(x)//W); out=[]
    for w in range(W):
        seg=x[w*step:(w+1)*step] if w<W-1 else x[(W-1)*step:]
        out.append(math.sqrt(sum(v*v for v in seg)/max(1,len(seg))))
    return out
windows=fp(x)
gp=os.environ["GOLDEN"]
if os.environ["CAPTURE"]=="1":
    json.dump({"modality":"audio","n_samples":N,"rms_windows":windows},open(gp,"w"),indent=2)
    print(f"CAPTURED golden: {N} samples, {W}-window RMS fingerprint -> {gp}"); sys.exit(0)
if not os.path.exists(gp): print(f"FAIL: no golden at {gp} — run --capture first."); sys.exit(1)
g=json.load(open(gp)); gn=g["n_samples"]; gw=g["rms_windows"]
if abs(N-gn) > max(2, gn*0.001):
    print(f"FAIL: sample count {N} vs golden {gn}"); sys.exit(1)
maxrel=max(abs(a-b)/(b+1e-9) for a,b in zip(windows,gw)) if gw else 1.0
TOL=1e-2
if maxrel<=TOL: print(f"PASS: audio matches golden (n={N}, max window RMS rel-err {maxrel:.2e} <= {TOL})"); sys.exit(0)
print(f"FAIL: audio deviates from golden (max window RMS rel-err {maxrel:.2e} > {TOL})"); sys.exit(1)
PY
    ;;
esac
