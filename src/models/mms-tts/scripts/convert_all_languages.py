#!/usr/bin/env python3
"""Batch-convert every MMS-TTS language pack to the on-device fp16 layout, then
zip into per-language packs ready for upload to HuggingFace.

For each language code (from `mms_tts_languages.json` next to this script):
  1. Run scripts/prep_lang.py <code> (re-uses the existing single-language
     converter that writes weights/<code>/ + assets/<code>/).
  2. Zip the 6 runtime files into packs/mms-tts-<code>.zip:
       weights/<code>/{model.fp16.bin, model.fp16.meta.json, tokenizer_vocab.bin}
       assets/<code>/{test_input_ids.bin, duration_noise.bin, prior_noise.bin}
  3. Delete weights/<code>/ + assets/<code>/ + the HF safetensors cache
     entry to keep disk usage bounded at ~150 MB regardless of how many
     languages we've processed.

Skips any language whose pack already exists (resumable).
Writes packs/languages.json — display-name registry consumed by the
in-app Language Picker. Names come from the `langcodes` library when
available; falls back to the raw ISO code.

Usage:
  python3 scripts/convert_all_languages.py                     # all 1140
  python3 scripts/convert_all_languages.py --limit 5           # first 5 only
  python3 scripts/convert_all_languages.py --codes eng spa hin # subset
  python3 scripts/convert_all_languages.py --resume-from amh   # restart from this code
"""
from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
import time
import zipfile
from pathlib import Path


HERE = Path(__file__).resolve().parent
PORT = HERE.parent
LANG_LIST = HERE / "mms_tts_languages.json"
PREP_LANG = HERE / "prep_lang.py"
WEIGHTS_DIR = PORT / "weights"
ASSETS_DIR = PORT / "assets"
PACKS_DIR = PORT / "packs"

# Files the on-device binary actually needs at runtime. The interactive REPL
# generates its own GaussianRng noise per utterance, so the prep_lang.py
# fixtures (test_input_ids / duration_noise / prior_noise) under assets/
# are NOT included — they exist only for deterministic-eval mode which the
# Android app never uses. Skipping them also removes the PyTorch ≥ 2.4
# dependency from prep_lang.py.
RUNTIME_FILES = [
    ("weights/{code}/model.fp16.bin",        True),
    ("weights/{code}/model.fp16.meta.json",  True),
    ("weights/{code}/tokenizer_vocab.bin",   True),
]


def _try_lang_meta(code: str) -> dict:
    """Look up display name + native name + script via the `langcodes` library
    if installed; otherwise return the raw code as both names. We DON'T fail
    when langcodes is missing — the language pack still works on-device, it
    just shows the ISO code in the picker."""
    base = code.split("-script_", 1)[0]
    script_suffix = code.split("-script_", 1)[1] if "-script_" in code else None
    try:
        import langcodes
        lang = langcodes.Language.get(base)
        name = lang.display_name()
        try:
            autonym = lang.autonym()
        except Exception:
            autonym = name
        script = (script_suffix.capitalize() if script_suffix
                  else (lang.script_name() if lang.script else
                        (lang.maximize().script_name() if lang.maximize().script else "")))
        if script_suffix:
            name = f"{name} ({script_suffix.capitalize()} script)"
    except ImportError:
        name = code
        autonym = code
        script = script_suffix.capitalize() if script_suffix else ""
    except Exception:
        name = code
        autonym = code
        script = script_suffix.capitalize() if script_suffix else ""
    return {"name": name, "native_name": autonym, "script": script}


def _zip_one(code: str) -> Path | None:
    """Pack the 6 runtime files for `code` into packs/mms-tts-<code>.zip.
    Returns the zip path on success, None if any required file is missing."""
    out_zip = PACKS_DIR / f"mms-tts-{code}.zip"
    PACKS_DIR.mkdir(parents=True, exist_ok=True)
    tmp_zip = out_zip.with_suffix(".zip.tmp")
    try:
        with zipfile.ZipFile(tmp_zip, "w", compression=zipfile.ZIP_STORED) as zf:
            for tmpl, required in RUNTIME_FILES:
                rel = tmpl.format(code=code)
                src = PORT / rel
                if not src.exists():
                    if required:
                        print(f"    MISSING required file: {rel}", file=sys.stderr)
                        return None
                    continue
                zf.write(src, arcname=rel)
        tmp_zip.rename(out_zip)
        return out_zip
    except Exception as e:
        if tmp_zip.exists():
            tmp_zip.unlink(missing_ok=True)
        print(f"    zip error for {code}: {e}", file=sys.stderr)
        return None


def _cleanup_one(code: str) -> None:
    """Delete intermediate per-language dirs so disk stays bounded.
    Idempotent — safe to call before or after prep_lang.py succeeds."""
    for parent in (WEIGHTS_DIR, ASSETS_DIR):
        d = parent / code
        if d.exists():
            shutil.rmtree(d, ignore_errors=True)


def _process_one(code: str, force: bool, log_dir: Path) -> tuple[bool, str]:
    """Convert + zip one language. Returns (ok, status_str)."""
    out_zip = PACKS_DIR / f"mms-tts-{code}.zip"
    if out_zip.exists() and not force:
        return True, f"skipped (zip already exists, {out_zip.stat().st_size // 1024 // 1024} MB)"

    # prep_lang.py writes weights/<code>/* and assets/<code>/* in-place.
    log_file = log_dir / f"{code}.log"
    t0 = time.time()
    # NNOPT_SKIP_FIXTURES=1 tells prep_lang.py to skip the make_fixtures
    # step. Fixtures (a) aren't needed at runtime for interactive synthesis
    # and (b) require PyTorch ≥ 2.4 which not every environment has.
    import os as _os
    env = {**_os.environ, "NNOPT_SKIP_FIXTURES": "1"}
    with open(log_file, "w") as logf:
        rc = subprocess.run(
            [sys.executable, str(PREP_LANG), code],
            cwd=str(PORT),
            stdout=logf,
            stderr=subprocess.STDOUT,
            env=env,
        ).returncode
    if rc != 0:
        # Don't cleanup — keep partial state for debugging. log_file has details.
        return False, f"prep_lang rc={rc} (see {log_file})"
    elapsed = time.time() - t0

    zp = _zip_one(code)
    _cleanup_one(code)
    if zp is None:
        return False, "zip failed (see log)"
    size_mb = zp.stat().st_size // 1024 // 1024
    return True, f"OK · {size_mb} MB · prep {elapsed:.0f}s"


def _build_languages_json(codes: list[str]) -> Path:
    """Walk packs/ for every successfully-built zip and emit a registry the
    in-app Language Picker can consume."""
    out = PACKS_DIR / "languages.json"
    entries = []
    for code in codes:
        zp = PACKS_DIR / f"mms-tts-{code}.zip"
        if not zp.exists():
            continue
        meta = _try_lang_meta(code)
        entries.append({
            "code": code,
            "name": meta["name"],
            "native_name": meta["native_name"],
            "script": meta["script"],
            "size_bytes": zp.stat().st_size,
        })
    # Sort by English name for a stable, alphabetical default ordering.
    entries.sort(key=lambda e: (e["name"].lower(), e["code"]))
    out.write_text(json.dumps(entries, ensure_ascii=False, indent=2))
    return out


def _write_upload_md() -> Path:
    """One-pager telling the user exactly how to push packs/ to HF."""
    out = PACKS_DIR / "UPLOAD.md"
    out.write_text(
        "# Upload to HuggingFace\n\n"
        "After convert_all_languages.py finishes, push the packs/ directory\n"
        "to your HF dataset repo:\n\n"
        "```\n"
        "huggingface-cli login   # one-time\n"
        "huggingface-cli upload a8nova/mms-tts-language-packs packs/ \\\n"
        "    . --repo-type dataset\n"
        "```\n\n"
        "Note the trailing `. --repo-type dataset` — `.` is the path inside\n"
        "the repo (root), and `--repo-type dataset` selects the dataset\n"
        "namespace (HF defaults to model repos). After upload, the in-app\n"
        "downloader fetches from:\n"
        "  https://huggingface.co/datasets/a8nova/mms-tts-language-packs/resolve/main/<file>\n"
    )
    return out


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--limit", type=int, default=None,
                   help="process only the first N codes (smoke test)")
    p.add_argument("--codes", nargs="+", default=None,
                   help="explicit list of codes to process (overrides --limit)")
    p.add_argument("--resume-from", default=None,
                   help="restart from this code (skip earlier codes)")
    p.add_argument("--force", action="store_true",
                   help="rebuild zips that already exist")
    p.add_argument("--batch-size", type=int, default=50,
                   help="print a header + summary every N languages (default 50)")
    args = p.parse_args()

    if not LANG_LIST.exists():
        print(f"ERROR: {LANG_LIST} not found", file=sys.stderr)
        return 1
    all_codes = json.loads(LANG_LIST.read_text())

    if args.codes:
        codes = args.codes
    else:
        codes = list(all_codes)
        if args.resume_from:
            try:
                idx = codes.index(args.resume_from)
                codes = codes[idx:]
            except ValueError:
                print(f"ERROR: --resume-from {args.resume_from} not in list", file=sys.stderr)
                return 1
        if args.limit:
            codes = codes[: args.limit]

    PACKS_DIR.mkdir(parents=True, exist_ok=True)
    log_dir = PACKS_DIR / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)

    print(f">>> {len(codes)} languages to process (of {len(all_codes)} total)")
    print(f">>> packs landing in {PACKS_DIR}")

    failed: list[tuple[str, str]] = []
    ok_count = 0
    started = time.time()
    batch_size = max(1, args.batch_size)
    total = len(codes)
    n_batches = (total + batch_size - 1) // batch_size
    batch_ok = 0
    batch_fail = 0
    batch_started = started
    for i, code in enumerate(codes, 1):
        if (i - 1) % batch_size == 0:
            batch_num = (i - 1) // batch_size + 1
            lo = i
            hi = min(i + batch_size - 1, total)
            elapsed_min = (time.time() - started) / 60.0
            bar = "=" * 60
            print(bar)
            print(f" Batch {batch_num}/{n_batches}  ·  languages {lo}-{hi} of {total}"
                  f"  ·  elapsed {elapsed_min:.1f} min")
            print(bar)
            batch_ok = 0
            batch_fail = 0
            batch_started = time.time()

        prefix = f"[{i:4d}/{total}] {code:30s}"
        ok, status = _process_one(code, force=args.force, log_dir=log_dir)
        print(f"{prefix} {status}")
        if ok:
            ok_count += 1
            batch_ok += 1
        else:
            failed.append((code, status))
            batch_fail += 1

        if i % batch_size == 0 or i == total:
            batch_num = (i - 1) // batch_size + 1
            batch_elapsed_min = (time.time() - batch_started) / 60.0
            print(f"--- batch {batch_num} done: {batch_ok} OK, {batch_fail} failed"
                  f" · {batch_elapsed_min:.1f} min"
                  f" · running totals {ok_count} OK / {len(failed)} failed ---")

    # Final registry + upload helper, regardless of partial failures.
    lang_json = _build_languages_json(all_codes)
    upload_md = _write_upload_md()
    elapsed = time.time() - started
    print()
    print(f"Done in {elapsed/60:.1f} min · {ok_count}/{len(codes)} OK · {len(failed)} failed")
    print(f"Registry: {lang_json} ({len(json.loads(lang_json.read_text()))} entries)")
    print(f"Upload instructions: {upload_md}")
    if failed:
        print()
        print("Failures:")
        for code, status in failed[:20]:
            print(f"  {code}: {status}")
        if len(failed) > 20:
            print(f"  … and {len(failed) - 20} more (see packs/logs/<code>.log)")
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
