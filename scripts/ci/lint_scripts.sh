#!/usr/bin/env bash
# Static invariant checks for every model's build/deploy/run scripts.
# Runs device-free in CI (and locally). Catches the regression classes we have
# actually hit: missing libclblast.so push, fp32-as-default, non-executable
# scripts, and shell syntax errors. Exit non-zero on any violation.
set -uo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
MODELS_DIR="${REPO_ROOT}/src/models"
fail=0
note() { printf '  %s\n' "$*"; }
bad()  { printf '✗ %s\n' "$*"; fail=1; }
ok()   { printf '✓ %s\n' "$*"; }

# Discover real model dirs (skip snapshot/junk dirs v2/ v3/ and build dirs).
models=()
for d in "${MODELS_DIR}"/*/; do
  name="$(basename "$d")"
  case "$name" in v2|v3) continue;; esac
  [ -d "${d}scripts" ] || continue
  models+=("$name")
done
echo "Linting ${#models[@]} models: ${models[*]}"
echo ""

for m in "${models[@]}"; do
  sdir="${MODELS_DIR}/${m}/scripts"

  # 1. required scripts exist
  for s in build.sh deploy_android.sh run_android.sh; do
    if [ ! -f "${sdir}/${s}" ]; then bad "${m}: missing scripts/${s}"; continue; fi
    # 2. executable bit (deploy is invoked as ./scripts/deploy_android.sh)
    [ -x "${sdir}/${s}" ] || bad "${m}: scripts/${s} is not executable (chmod +x)"
    # 3. shell syntax
    bash -n "${sdir}/${s}" 2>/dev/null || bad "${m}: scripts/${s} has a shell syntax error"
  done

  # 4. no fp32 default anywhere in the model's scripts (we only ship fp16)
  if grep -rqE 'NNOPT_DTYPE:-fp32' "${sdir}" 2>/dev/null; then
    bad "${m}: a script still defaults NNOPT_DTYPE to fp32 (should be fp16)"
  fi

  # 5. deploy must push libclblast.so (it is dynamically linked at launch).
  #    Allowed exception: a script that explicitly documents static linking.
  dep="${sdir}/deploy_android.sh"
  if [ -f "$dep" ] && ! grep -qiE 'clblast' "$dep"; then
    bad "${m}: deploy_android.sh never references libclblast.so (clean deploy will fail to link)"
  fi
done

# 6. shellcheck if available (warnings don't fail; errors do)
if command -v shellcheck >/dev/null 2>&1; then
  echo ""
  echo "Running shellcheck (errors fail the build)..."
  while IFS= read -r f; do
    shellcheck --severity=error "$f" || fail=1
  done < <(find "${MODELS_DIR}" -path '*/scripts/*.sh' ! -path '*/v2/*' ! -path '*/v3/*')
else
  note "shellcheck not installed — skipping (install in CI for deeper checks)"
fi

echo ""
if [ "$fail" -eq 0 ]; then echo "✓ all script invariants hold"; else echo "✗ lint failed"; fi
exit "$fail"
