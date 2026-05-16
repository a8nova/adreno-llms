#!/usr/bin/env bash
# Pin Adreno GPU + CPU into a stable peak-performance mode for benchmarking.
#
# Sources the routines from Qualcomm Snapdragon Mobile Platform OpenCL General
# Programming and Optimization (80-NB295-11 Rev C, Appendix A). Picks the right
# command sequence by Adreno tier auto-detected from CL_DEVICE_NAME. Requires
# `adb root` (i.e. a userdebug/rooted device); without root the /sys/class/kgsl
# writes silently no-op.
#
# Usage:
#     ./scripts/perf_mode.sh            # auto-detect Adreno tier, apply, verify
#     ./scripts/perf_mode.sh --check    # only verify current state (no writes)
#     ./scripts/perf_mode.sh --tier a6x # force tier (a3x|a4x|a5x|a6x|a7x)
#     ./scripts/perf_mode.sh --restore  # try to restore default governors
#
# Re-run if the device reboots.
set -uo pipefail

MODE="apply"
FORCED_TIER=""

while [[ $# -gt 0 ]]; do
  case "$1" in
    --check)   MODE="check"; shift ;;
    --restore) MODE="restore"; shift ;;
    --tier)    FORCED_TIER="$2"; shift 2 ;;
    -h|--help)
      sed -n '1,/^set -uo pipefail/p' "$0" | sed '$d' | sed 's/^# \?//' ; exit 0 ;;
    *) echo "unknown arg: $1" >&2; exit 1 ;;
  esac
done

# ── adb prep ─────────────────────────────────────────────────────────────────
require_adb() {
  if ! command -v adb >/dev/null 2>&1; then
    echo "adb not found in PATH" >&2; exit 1
  fi
  adb wait-for-device
  adb root >/dev/null 2>&1 || true
  adb wait-for-device
  adb remount >/dev/null 2>&1 || true
  adb wait-for-device
}

# ── tier detect ──────────────────────────────────────────────────────────────
detect_tier() {
  if [[ -n "$FORCED_TIER" ]]; then
    echo "$FORCED_TIER"; return
  fi
  # Try CL_DEVICE_NAME via a tiny clinfo-style probe if present, else fall back
  # to /sys discovery and ro.board.platform heuristics.
  local model name plat
  model=$(adb shell getprop ro.product.model 2>/dev/null | tr -d '\r')
  plat=$(adb shell getprop ro.board.platform 2>/dev/null | tr -d '\r')
  name=$(adb shell sh -c 'cat /sys/class/kgsl/kgsl-3d0/gpu_model 2>/dev/null' 2>/dev/null | tr -d '\r')
  if [[ -z "$name" ]]; then
    # gpu_model not exposed on all kernels — fall back to a guess from platform.
    name="$plat"
  fi
  # Map common Adreno generations.
  local lower="${name,,}"
  case "$lower" in
    *a3*|*adreno_3*|*adreno3*) echo "a3x" ;;
    *a4*|*adreno_4*|*adreno4*) echo "a4x" ;;
    *a5*|*adreno_5*|*adreno5*) echo "a5x" ;;
    *a6*|*adreno_6*|*adreno6*|*sm6125*|*sm6150*|*sm7125*|*sm7150*|*sm7225*|*sm7250*|*sm8150*|*sm8250*|*sm8350*) echo "a6x" ;;
    *a7*|*adreno_7*|*adreno7*|*sm8450*|*sm8475*|*sm8550*|*sm8650*) echo "a7x" ;;
    *) echo "a6x" ;;  # safe default for the SmolLM2 deployment target band
  esac
}

# ── apply commands per tier (mirrors PDF §A.1–A.3) ──────────────────────────
apply_a3x() {
  adb shell "stop mpdecision" || true
  adb shell "echo performance > /sys/devices/system/cpu/cpu0/cpufreq/scaling_governor"
  adb shell "echo 1 > /sys/devices/system/cpu/cpu1/online" || true
  adb shell "echo performance > /sys/devices/system/cpu/cpu1/cpufreq/scaling_governor" || true
  adb shell "echo performance > /sys/class/kgsl/kgsl-3d0/devfreq/governor" || true
  adb shell "echo none > /sys/class/kgsl/kgsl-3d0/pwrscale/policy" || true
  adb shell "echo 1 > /sys/class/kgsl/kgsl-3d0/force_clk_on" || true
  adb shell "echo 1 > /sys/class/kgsl/kgsl-3d0/force_bus_on" || true
  adb shell "echo 1 > /sys/class/kgsl/kgsl-3d0/force_rail_on" || true
  adb shell "echo 10000000 > /sys/class/kgsl/kgsl-3d0/idle_timer" || true
}

apply_a4x_or_a5x() {
  adb shell "stop thermald" || true
  adb shell "stop mpdecision" || true
  for c in 0 1 2 3 4 5 6 7; do
    adb shell "echo 1 > /sys/devices/system/cpu/cpu${c}/online" >/dev/null 2>&1 || true
    adb shell "echo performance > /sys/devices/system/cpu/cpu${c}/cpufreq/scaling_governor" >/dev/null 2>&1 || true
  done
  adb shell "echo 0 > /sys/class/kgsl/kgsl-3d0/min_pwrlevel" || true
  adb shell "echo 0 > /sys/class/kgsl/kgsl-3d0/max_pwrlevel" || true
  adb shell "echo performance > /sys/class/kgsl/kgsl-3d0/devfreq/governor" || true
  adb shell "echo performance > /sys/class/devfreq/qcom,cpubw.29/governor" >/dev/null 2>&1 || true
  adb shell "echo 1 > /sys/class/kgsl/kgsl-3d0/force_clk_on" || true
  adb shell "echo 1000000 > /sys/class/kgsl/kgsl-3d0/idle_timer" || true
}

apply_a6x_or_a7x() {
  adb shell "stop thermal-engine" || true
  for c in 0 1 2 3 4 5 6 7; do
    adb shell "echo 1 > /sys/devices/system/cpu/cpu${c}/online" >/dev/null 2>&1 || true
    adb shell "echo performance > /sys/devices/system/cpu/cpu${c}/cpufreq/scaling_governor" >/dev/null 2>&1 || true
  done
  adb shell "echo -n disable > /sys/devices/soc/soc:qcom,bcl/mode" >/dev/null 2>&1 || true
  adb shell "echo performance > /sys/class/devfreq/soc:qcom,cpubw/governor" >/dev/null 2>&1 || true
  adb shell "echo performance > /sys/class/devfreq/soc:qcom,gpubw/governor" >/dev/null 2>&1 || true
  adb shell "echo performance > /sys/class/kgsl/kgsl-3d0/devfreq/governor" || true
  adb shell "echo 0 > /sys/class/kgsl/kgsl-3d0/min_pwrlevel" || true
  adb shell "echo 0 > /sys/class/kgsl/kgsl-3d0/max_pwrlevel" || true
  adb shell "echo 100000000 > /sys/class/kgsl/kgsl-3d0/idle_timer" || true
}

# ── verify ──────────────────────────────────────────────────────────────────
verify() {
  echo "──── verify ────"
  adb shell "cat /sys/class/kgsl/kgsl-3d0/devfreq/governor" 2>/dev/null || true
  adb shell "cat /sys/class/kgsl/kgsl-3d0/devfreq/cur_freq" 2>/dev/null || true
  adb shell "cat /sys/devices/system/cpu/cpu0/cpufreq/scaling_governor" 2>/dev/null || true
  adb shell "cat /sys/devices/system/cpu/cpu0/cpufreq/scaling_cur_freq" 2>/dev/null || true
  adb shell "cat /sys/class/devfreq/soc:qcom,gpubw/governor" 2>/dev/null || true
}

restore() {
  for c in 0 1 2 3 4 5 6 7; do
    adb shell "echo schedutil > /sys/devices/system/cpu/cpu${c}/cpufreq/scaling_governor" >/dev/null 2>&1 || true
  done
  adb shell "echo msm-adreno-tz > /sys/class/kgsl/kgsl-3d0/devfreq/governor" >/dev/null 2>&1 || true
  echo "restored default-ish governors (best-effort)"
}

require_adb
TIER=$(detect_tier)
echo "Adreno tier: $TIER  (mode: $MODE)"

case "$MODE" in
  check)   verify ;;
  restore) restore; verify ;;
  apply)
    case "$TIER" in
      a3x) apply_a3x ;;
      a4x|a5x) apply_a4x_or_a5x ;;
      a6x|a7x) apply_a6x_or_a7x ;;
      *) echo "unknown tier '$TIER'" >&2; exit 1 ;;
    esac
    verify
    ;;
esac
