#!/bin/bash
# Build, package, and launch the See & Say APK on the connected device.
#
# Flags:
#   --skip-prepare   Don't rerun prepare_assets.sh (use cached assets/jniLibs).
#   --release        Build and install the release APK (default: debug).

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
APP_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"

SKIP_PREPARE=0
VARIANT=debug
for arg in "$@"; do
    case "$arg" in
        --skip-prepare) SKIP_PREPARE=1 ;;
        --release) VARIANT=release ;;
        *) echo "unknown arg: $arg" >&2; exit 2 ;;
    esac
done

if [[ "$SKIP_PREPARE" -eq 0 ]]; then
    "$SCRIPT_DIR/prepare_assets.sh"
fi

cd "$APP_ROOT"

if [[ -z "${JAVA_HOME:-}" ]]; then
    if [[ -d "/Applications/Android Studio.app/Contents/jbr/Contents/Home" ]]; then
        export JAVA_HOME="/Applications/Android Studio.app/Contents/jbr/Contents/Home"
    fi
fi

case "$VARIANT" in
    debug)   GRADLE_TASK=":app:assembleDebug";   APK_PATH="app/build/outputs/apk/debug/app-debug.apk" ;;
    release) GRADLE_TASK=":app:assembleRelease"; APK_PATH="app/build/outputs/apk/release/app-release-unsigned.apk" ;;
esac

echo "[install] gradle $GRADLE_TASK …"
./gradlew $GRADLE_TASK

if [[ ! -f "$APK_PATH" ]]; then
    echo "[install] APK not found at $APK_PATH" >&2; exit 1
fi

ADB="${ADB:-adb}"
if ! command -v "$ADB" >/dev/null 2>&1; then
    # Fallback: try the canonical Android SDK location under $ANDROID_HOME
    # (or $HOME/Library/Android/sdk on macOS, $HOME/Android/Sdk on Linux).
    for cand in \
        "${ANDROID_HOME:-}/platform-tools/adb" \
        "$HOME/Library/Android/sdk/platform-tools/adb" \
        "$HOME/Android/Sdk/platform-tools/adb"; do
        if [[ -x "$cand" ]]; then ADB="$cand"; break; fi
    done
fi

echo "[install] APK size: $(du -sh "$APK_PATH" | awk '{print $1}')"
echo "[install] adb install -r --no-incremental …"
"$ADB" install -r --no-incremental "$APK_PATH"

echo "[install] launching MainActivity …"
"$ADB" shell am start -n com.adreno.seeandsay/.MainActivity

echo "[install] done."
