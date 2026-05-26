#!/usr/bin/env bash
# Fetch the isi-nlp/uroman romanization tables into assets/uroman/.
#
# Required for non-Latin-script languages (amh, ara, khm, tha, …) since the
# on-device uroman pass needs both romanization-table.txt and the much
# larger romanization-auto-table.txt to transliterate Ge'ez / Arabic /
# Devanagari / etc. Latin-script languages (eng, deu, fra, …) run in
# passthrough mode and don't need these tables — running this script is
# still harmless for those.
#
# Idempotent: skips clone if assets/uroman/romanization-table.txt already
# exists. Override the source repo with UROMAN_REPO, the branch/tag with
# UROMAN_REF, or point UROMAN_DIR at a pre-existing local clone.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PORT_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"
DST="${PORT_DIR}/assets/uroman"

UROMAN_REPO="${UROMAN_REPO:-https://github.com/isi-nlp/uroman.git}"
UROMAN_REF="${UROMAN_REF:-master}"

mkdir -p "${DST}"

if [ -s "${DST}/romanization-table.txt" ] && [ -s "${DST}/romanization-auto-table.txt" ]; then
    echo "Uroman tables already present at ${DST} — nothing to do."
    exit 0
fi

# If the user pointed us at a local checkout, copy from there.
SRC_DATA=""
if [ -n "${UROMAN_DIR:-}" ]; then
    if [ -f "${UROMAN_DIR}/data/romanization-table.txt" ]; then
        SRC_DATA="${UROMAN_DIR}/data"
    elif [ -f "${UROMAN_DIR}/romanization-table.txt" ]; then
        SRC_DATA="${UROMAN_DIR}"
    else
        echo "ERROR: UROMAN_DIR=${UROMAN_DIR} doesn't contain romanization-table.txt" >&2
        exit 1
    fi
else
    # Shallow-clone into a temp dir so the repo doesn't pollute the cache.
    TMP="$(mktemp -d)"
    trap 'rm -rf "${TMP}"' EXIT
    echo ">>> Cloning ${UROMAN_REPO}@${UROMAN_REF} (shallow)"
    git clone --depth 1 --branch "${UROMAN_REF}" "${UROMAN_REPO}" "${TMP}/uroman"
    # The repo layout has moved over time — current upstream is
    # uroman/uroman/data/. Find the directory that actually contains
    # romanization-table.txt instead of hard-coding a path.
    TABLE_HIT="$(find "${TMP}/uroman" -maxdepth 5 -type f -name 'romanization-table.txt' -print -quit 2>/dev/null || true)"
    if [ -z "${TABLE_HIT}" ]; then
        echo "ERROR: romanization-table.txt not found inside the cloned repo" >&2
        echo "       (looked under ${TMP}/uroman). Upstream layout may have changed." >&2
        exit 1
    fi
    SRC_DATA="$(dirname "${TABLE_HIT}")"
fi

for f in romanization-table.txt romanization-auto-table.txt chars-to-delete.txt; do
    if [ -f "${SRC_DATA}/${f}" ]; then
        cp "${SRC_DATA}/${f}" "${DST}/${f}"
        echo "    ${f} ($(wc -c < "${DST}/${f}") bytes)"
    else
        echo "    note: ${f} not found in source — skipping" >&2
    fi
done

# Sanity: at minimum we need romanization-table.txt for the loader to succeed.
if [ ! -s "${DST}/romanization-table.txt" ]; then
    echo "ERROR: ${DST}/romanization-table.txt is empty or missing" >&2
    exit 1
fi

echo ""
echo "Uroman tables installed at ${DST}"
echo "Latin-script languages don't need these; non-Latin scripts now work."
