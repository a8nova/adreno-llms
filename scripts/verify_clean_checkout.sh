#!/usr/bin/env bash
# Pre-publish verifier for adreno-llms.
#
# Run from anywhere inside the repo. Scans the would-be-shipped file set
# (tracked + untracked-but-not-gitignored) for the 8 problem classes that
# block a public release. The script EXCLUDES ITSELF and .gitignore from
# pattern scans so its own grep patterns / gitignore patterns aren't
# treated as leaks.
#
# Categories:
#   1. Local-path leaks            user homes, internal cache dirs, internal repo names
#   2. User identity               user-name leaks (a8nova@gmail.com IS allowed)
#   3. NNOpt tool internals        internal scaffolding files
#   4. Build artifacts             regenerable build output
#   5. Huge files (>50 MB)         would blow up git clone
#   6. Symlinks                    break on fresh clone
#   7. Credentials                 API keys, secrets, private keys, HF tokens
#   8. Required-file manifest      LICENSE, README, scripts, per-model files
#
# Plus a SOFT pass: broken relative markdown links (warn, don't fail).
#
# Exit 0 = clean; Exit 1 = at least one hard failure.

set -uo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "$REPO"

# ANSI colors
if [ -t 1 ]; then
  RED=$'\033[31m'; GRN=$'\033[32m'; YEL=$'\033[33m'; DIM=$'\033[2m'; OFF=$'\033[0m'
else
  RED=""; GRN=""; YEL=""; DIM=""; OFF=""
fi

PASS=0; FAIL=0; WARN=0
fail()  { echo "${RED}x${OFF} $*"; FAIL=$((FAIL+1)); }
pass()  { echo "${GRN}+${OFF} $*"; PASS=$((PASS+1)); }
warn()  { echo "${YEL}!${OFF} $*"; WARN=$((WARN+1)); }
info()  { echo "${DIM}  $*${OFF}"; }
section() { echo ""; echo "-- $1 --"; }

# ── Files to EXCLUDE from pattern scans ──────────────────────────────────
# These contain literal copies of the patterns we're looking for as part of
# their job (the verifier's own regexes; gitignore's exclusion patterns).
EXCLUDE_FROM_SCAN=(
  "scripts/verify_clean_checkout.sh"
  ".gitignore"
)

# Build a single regex like '^a$|^b$' for excluding from grep.
EXCLUDE_REGEX=""
for f in "${EXCLUDE_FROM_SCAN[@]}"; do
  EXCLUDE_REGEX="${EXCLUDE_REGEX}${EXCLUDE_REGEX:+|}^${f}\$"
done

# ── Build the would-be-shipped file list ─────────────────────────────────
SHIP_FILES=()
while IFS= read -r line; do
  [ -n "$line" ] && SHIP_FILES+=("$line")
done < <(git ls-files --cached --others --exclude-standard 2>/dev/null || true)

# Files included in pattern scans (excludes self + .gitignore).
SCAN_FILES=()
while IFS= read -r line; do
  [ -n "$line" ] && SCAN_FILES+=("$line")
done < <(printf '%s\n' "${SHIP_FILES[@]}" | grep -vE "$EXCLUDE_REGEX" || true)

TOTAL=${#SHIP_FILES[@]}
SCAN_TOTAL=${#SCAN_FILES[@]}
echo "Verifying ${TOTAL} files (${SCAN_TOTAL} scanned for content patterns)"

# Helper: grep a regex across SCAN_FILES only.
scan_grep() {
  local pattern="$1"
  if [ "$SCAN_TOTAL" -eq 0 ]; then return 1; fi
  printf '%s\n' "${SCAN_FILES[@]}" | xargs -I{} -- grep -lE "$pattern" "{}" 2>/dev/null
}

# ── 1. Local-path leaks ──────────────────────────────────────────────────
# Patterns assembled at runtime to avoid stray sed sweeps over this script.
DOT_NNPORT=$(printf '\\.nnp%s/' "ort")
DOT_NNOPT=$(printf '\\.nn%s/' "opt")
NNOPT_CONV=$(printf 'nnopt-%s' "converted")
LEAK_PAT="/Users/[a-zA-Z]+|/home/[a-zA-Z]+|/Volumes/|${NNOPT_CONV}|${DOT_NNPORT}"

section "1. Local-path leaks"
LEAKS=$(scan_grep "$LEAK_PAT" || true)
if [ -n "$LEAKS" ]; then
  fail "Found path leaks in:"
  echo "$LEAKS" | sed 's/^/    /'
else
  pass "no /Users/, /home/, /Volumes/, nnopt-converted, or nnopt internal-dir references"
fi

# ── 2. User identity ─────────────────────────────────────────────────────
section "2. User identity"
USER_PAT=$(printf '%s|%s' "alaz""ar" "shen""kute")
IDENT=$(scan_grep "$USER_PAT" || true)
if [ -n "$IDENT" ]; then
  fail "Found user identity leaks in:"
  echo "$IDENT" | sed 's/^/    /'
else
  pass "no user-name references"
fi

# ── 3. NNOpt tool internals ──────────────────────────────────────────────
section "3. NNOpt tool internals (must NOT ship)"
INTERNAL_NAMES=(
  "$(printf '.nnp%s' "ort")"
  "port_state.json"
  "port_timeline.jsonl"
  "port_invariants.json"
  "build_history.json"
  "IMPLEMENTATION_DESIGN.md"
  "SOURCE_ANALYSIS.md"
  "architecture_facts.md"
  "_port_invariants_probe.py"
  "nnport_config.yaml"
)
found_internal=""
for f in "${INTERNAL_NAMES[@]}"; do
  hits=$(printf '%s\n' "${SHIP_FILES[@]}" | grep -E "(^|/)${f}(/|\$)" || true)
  if [ -n "$hits" ]; then
    found_internal+="$hits"$'\n'
  fi
done
if [ -n "$found_internal" ]; then
  fail "NNOpt tool-internal files would ship:"
  echo "$found_internal" | sed 's/^/    /'
else
  pass "no internal scaffolding files (nnport dir, port_*, IMPLEMENTATION_DESIGN.md, etc.)"
fi

# ── 4. Build artifacts ───────────────────────────────────────────────────
section "4. Build artifacts (regenerable, should not ship)"
ARTIFACT_PATTERNS='(^|/)build($|/)|(^|/)build_tests($|/)|(^|/)__pycache__($|/)|\.(o|a|so|dylib|pyc|pyo)$|CMakeCache\.txt$|cmake_install\.cmake$|compile_commands\.json$|\.cache($|/)'
ARTIFACTS=$(printf '%s\n' "${SHIP_FILES[@]}" | grep -E "$ARTIFACT_PATTERNS" || true)
if [ -n "$ARTIFACTS" ]; then
  fail "Build artifacts found:"
  echo "$ARTIFACTS" | sed 's/^/    /' | head -20
else
  pass "no build/, *.o, *.so, *.a, __pycache__/, CMakeCache, compile_commands"
fi

# ── 5. Huge files (>50 MB) ───────────────────────────────────────────────
section "5. Huge files (>50 MB)"
HUGE=""
for f in "${SHIP_FILES[@]}"; do
  if [ -f "$f" ]; then
    size=$(stat -f%z "$f" 2>/dev/null || stat -c%s "$f" 2>/dev/null || echo 0)
    if [ "$size" -gt 52428800 ]; then
      HUGE+="$(printf '%6sM %s' "$((size / 1048576))" "$f")"$'\n'
    fi
  fi
done
if [ -n "$HUGE" ]; then
  fail "Files larger than 50 MB:"
  echo "$HUGE" | sed 's/^/    /'
else
  pass "no file exceeds 50 MB"
fi

# ── 6. Symlinks ──────────────────────────────────────────────────────────
section "6. Tracked symlinks"
LINKS=""
for f in "${SHIP_FILES[@]}"; do
  if [ -L "$f" ]; then
    LINKS+="$(printf '%s -> %s' "$f" "$(readlink "$f")")"$'\n'
  fi
done
if [ -n "$LINKS" ]; then
  fail "Tracked symlinks:"
  echo "$LINKS" | sed 's/^/    /'
else
  pass "no symlinks"
fi

# ── 7. Credentials / secrets ─────────────────────────────────────────────
section "7. Credentials and secrets"
SECRET_PAT='([A-Z_]*_)?API[_-]?KEY[ "=:]|SECRET[_-]?KEY[ "=:]|PASSWORD[ "=:]|BEGIN (RSA|OPENSSH|EC|DSA|PRIVATE) KEY|hf_[A-Za-z0-9]{30,}|aws_secret_access|gh[ps]_[A-Za-z0-9_]{30,}'
SECRETS=$(scan_grep "$SECRET_PAT" || true)
CRED_FILES=$(printf '%s\n' "${SHIP_FILES[@]}" | grep -E '\.env$|\.pem$|\.key$|\.p12$|id_rsa|id_dsa|id_ecdsa|id_ed25519' || true)
if [ -n "$SECRETS" ] || [ -n "$CRED_FILES" ]; then
  fail "Possible credentials in:"
  [ -n "$SECRETS" ] && echo "$SECRETS" | sed 's/^/    [content match] /'
  [ -n "$CRED_FILES" ] && echo "$CRED_FILES" | sed 's/^/    [filename] /'
else
  pass "no credential patterns or sensitive filenames"
fi

# ── 8. Required-file manifest ────────────────────────────────────────────
section "8. Required files"
required=(
  "README.md"
  "LICENSE"
  ".gitignore"
  "scripts/fetch_weights.sh"
  "scripts/fetch_openelm_weights.sh"
  "scripts/convert_openelm_weights.py"
  "scripts/upload_weights_to_hf.sh"
  "scripts/hf_repo_README.md"
  "scripts/verify_clean_checkout.sh"
)
MODELS=(granite-4-0-350m lfm2-5-350m mamba-130m mamba2-130m openelm-270m qwen2-5-0-5b smollm2-135m-instruct)
for m in "${MODELS[@]}"; do
  required+=(
    "src/models/$m/README.md"
    "src/models/$m/BENCHMARK.md"
    "src/models/$m/CMakeLists.txt"
    "src/models/$m/scripts/build.sh"
    "src/models/$m/scripts/deploy_android.sh"
    "src/models/$m/scripts/run_android.sh"
    "src/models/$m/weights/.gitkeep"
  )
done
missing=""
for r in "${required[@]}"; do
  [ -f "$r" ] || missing+="    $r"$'\n'
done
if [ -n "$missing" ]; then
  fail "Required files missing:"
  echo "$missing"
else
  pass "all ${#required[@]} required files present"
fi

# ── License sanity ───────────────────────────────────────────────────────
section "License sanity"
if [ -f LICENSE ] && grep -q 'Apache License' LICENSE && grep -q 'Version 2.0' LICENSE; then
  pass "LICENSE is Apache 2.0"
else
  fail "LICENSE missing or not Apache 2.0"
fi

# ── 9. Soft: broken markdown relative links ──────────────────────────────
section "9. Markdown relative links (soft check)"
broken=""
MD_FILES=()
while IFS= read -r line; do
  [ -n "$line" ] && MD_FILES+=("$line")
done < <(printf '%s\n' "${SHIP_FILES[@]}" | grep '\.md$' || true)
for md in "${MD_FILES[@]:-}"; do
  [ -z "$md" ] && continue
  while IFS= read -r link; do
    case "$link" in http*|mailto*|\#*) continue;; esac
    link_clean="${link%%#*}"
    link_clean="${link_clean%%\?*}"
    [ -z "$link_clean" ] && continue
    dir=$(dirname "$md")
    target="$dir/$link_clean"
    if [ ! -e "$target" ] && [ ! -d "$target" ]; then
      broken+="    $md -> $link"$'\n'
    fi
  done < <(grep -oE '\]\(([^)]+)\)' "$md" 2>/dev/null | sed -E 's/^\]\(//; s/\)$//' || true)
done
if [ -n "$broken" ]; then
  warn "Broken-looking relative links (review):"
  echo "$broken" | head -15
else
  pass "no broken relative markdown links"
fi

# ── Summary ──────────────────────────────────────────────────────────────
echo ""
echo "============================================="
echo " Summary: ${GRN}${PASS} pass${OFF}  ${RED}${FAIL} fail${OFF}  ${YEL}${WARN} warn${OFF}"
echo "============================================="

if [ "$FAIL" -gt 0 ]; then
  echo "${RED}Repo is NOT clean. Fix the failures above before pushing.${OFF}"
  exit 1
fi
if [ "$WARN" -gt 0 ]; then
  echo "${YEL}Clean enough to push, but review warnings.${OFF}"
fi
echo "${GRN}Ready to push.${OFF}"
exit 0
