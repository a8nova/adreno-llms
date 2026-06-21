#!/usr/bin/env python3
"""Run this in Google Colab (or any fast-internet machine) to batch-convert
MMS-TTS language packs and upload them to HuggingFace.

Quick start (Colab cell):
  !pip install -q huggingface_hub transformers safetensors numpy langcodes
  !huggingface-cli login --token YOUR_HF_TOKEN
  !python colab_batch_convert.py --limit 500 --resume-after bcl

It skips languages whose zip is already in packs/ (resumable after crashes).
After conversion, it uploads everything to a8nova/adreno-llms-weights/mms-tts/.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import struct
import sys
import time
import zipfile
from pathlib import Path

import numpy as np

HF_REPO = "a8nova/adreno-llms-weights"
HF_PREFIX = "mms-tts"

VOCAB_MAGIC = 0x564D5456
VOCAB_VERSION = 1

ALL_LANGS = [
    "abi","abp","aca","acd","ace","acf","ach","acn","acr","acu","ade","adh",
    "adj","adx","aeu","agd","agg","agn","agr","agu","agx","aha","ahk","aia",
    "aka","akb","ake","akp","alj","alp","alt","alz","ame","amf","amh","ami",
    "amk","ann","any","aoz","apb","apr","ara","arl","asa","asg","asm","ata",
    "atb","atg","ati","atq","ava","avn","avu","awa","awb","ayo","ayr","ayz",
    "azb","azg","azj-script_cyrillic","azj-script_latin","azz","bak","bam",
    "ban","bao","bav","bba","bbb","bbc","bbo","bcc-script_arabic",
    "bcc-script_latin","bcl","bcw","bdg","bdh","bdp","bdq","bdu","bdv","beh",
    "bel","bem","ben","bep","bex","bfa","bfo","bfy","bgc","bgq","bgr","bgt",
    "bgw","bha","bht","bhz","bib","bim","bis","bjn","bjr","bjz","bkd","bkv",
    "blh","blt","blx","blz","bmq","bmr","bmu","bmv","bng","bni","bnp","boa",
    "bod","boj","bom","bor","bos","box","bpr","bps","bqc","bqj","bqp","bre",
    "bru","bsc","bsn","bss","btd","bts","btt","btx","bud","bul","bus","bwq",
    "bwu","byr","bzh","bzi","bzj","caa","cab","cac","cak","cao","car","cas",
    "cax","cbc","cbi","cbr","cbs","cbt","cbu","cce","ccp","ces","cfm","cgc",
    "chv","cja","cje","cjk","cjp","ckb","cko","ckt","cla","cle","cly","cme-script_latin",
    "cmn","cnh","cni","cnl","cnt","cob","cod","cof","cos","cot","cou","cpb",
    "crh","crk-script_latin","crk-script_syllabics","crn","crx","csk","ctd",
    "ctg","cto","ctu","cuc","cui","cuk","cul","cwa","cwe","cwt","cya","daa",
    "dab","dad","daf","dag","dah","dai","dak","dan","dar","dbj","dbq","ddn",
    "ded","dee","deu","dgk","dgo","dgr","dhi","did","dig","dik","dip","div",
    "djk","dnj-script_arabic","dnj-script_latin","dnw","dob","doi","dos","dsh",
    "dso","dtp","dts","dug","duo","dwr","dyi","dyo","dyu","efi","eka","ell",
    "enb","eng","enx","epo","eri","ese","esk","ess","eus","eve","ewe","eza",
    "faa","fai","fan","fas","fij","fil","fin","fmu","fon","fra","frd","fub",
    "fue","fuf","fuh","fuq","fur","fuv","fvr","gaa","gag","gan","gax","gaz",
    "gbk","gbm","gbo","gbs","gby","gde","gdn","gdx","geb","gej","gfk","ghs",
    "glg","glk","gmv","gna","gnd","gng","gof","gog","gor","gqr","grc","grt",
    "gso","gub","guc","gud","guh","guk","gum","gun","guo","guq","guu","gux",
    "guz","gvl","gwr","gym","gyr","had","hag","hak","ham","hao","har","hat",
    "hau","hay","heb","hig","hil","hin","hla","hlb","hlt","hne","hni","hnn",
    "hnj","hns","hoc","hoy","hrv","hsb","hto","hub","hui","hun","hus","huu",
    "huv","hvn","ian","iba","ibg","ibo","icr","idd","ifa","ifb","ife","ifk",
    "ifu","ify","igb","ige","ijc","ikk","ikw","ilo","imo","inb","ind","ino",
    "iou","ipi","iqw","iri","irk","isl","ita","itv","ixl","iyo","izr","izz",
    "jac","jam","jav","jbu","jen","jic","jiv","jmc","jmx","jpn","jun","juy",
    "jvn","kaa","kab","kac","kad","kai","kaj","kam","kan","kao","kaq","kat",
    "kay","kaz","kbo","kbp","kbq","kbr","kby","kca","kcg","kdc","kde","kdh",
    "kdi","kdj","kdl","kdn","kdr","kdt","kek","ken","ker","kes","keu","key",
    "kez","kfb","kff-script_latin","kfr","kfy","kha","khb","khg","khm","khq",
    "khs","khu","khw","khz","kia","kij","kik","kin","kir","kix","kjb","kje",
    "kjg","kjh","kki","kkj","kle","klu","klv","klw","kma","kmd","kme","kmg",
    "kmh","kmk","kmo","kmr-script_arabic","kmr-script_latin","kms","kmu","kmz",
    "knb","kne","knf","knj","knk","kno","kog","kor","kos","koz","kpg","kpl",
    "kpm","kpo","kpq","kps","kpv","kpx","kpz","kqe","kqp","kqr","kqw","kqy",
    "krc","kri","krj","krl","krs","kru","ksb","ksr","kss","ktb","ktj","ktu",
    "ktx","ktz","kub","kud","kue","kum","kun","kup","kus","kvn","kvw","kwd",
    "kwf","kwi","kxc","kxf","kxm","kxv","kyb","kyc","kyf","kyg","kyq","kyu",
    "kyz","kzf","lac","laj","lam","lao","las","lat","lav","law","lbj","lbw",
    "lcp","lee","lem","lep","leu","lew","lex","lfn","lgg","lgr","lhu","lia",
    "lid","lif","lin","lip","lis","lit","lje","ljp","llb","lln","lme","lmk",
    "lmo","lmp","lms","lns","lob","lok","lom","lon","loq","los","lox","loz",
    "lrc","lsi","lsm","ltz","lua","luc","lud","lue","lug","lun","luo","lus",
    "luy","luz","lwl","lzz","mad","mae","mag","mah","mai","mak","mal","mam",
    "mar","mau","mav","maw","max","maz","mbb","mbc","mbh","mbi","mbj","mbt",
    "mbu","mbz","mca","mcb","mcd","mce","mcf","mci","mco","mcp","mcq","mcr",
    "mcu","mda","mdf","mdh","mdj","mdp","mdr","mds","mdy","med","mee","mej",
    "men","meq","met","meu","mev","mey","mfe","mfh","mfi","mfk","mfq","mfy",
    "mfz","mgd","mge","mgh","mgo","mhi","mhl","mhm","mhr","mhu","mhx","mhy",
    "mib","mie","mif","mih","mil","mim","min","mio","mip","miq","mit","miy",
    "miz","mjl","mjv","mkd","mkl","mkn","mks","mkz","mlg","mlt","mmg","mnb",
    "mnf","mnk","mnw","mnx","moa","moc","mog","mon","mop","mor","mos","mox",
    "moz","mpg","mpm","mpp","mpx","mqb","mqf","mqj","mqn","mrn","mro","mrt",
    "mrw","msy","mtd","mtg","mti","mtj","mto","mtt","mtu","mtv","mua","mub",
    "muh","muy","mva","mvn","mwm","mwn","mwq","mxb","mxq","mxt","mxv","myb",
    "myk","myl","myv","myw","myx","myy","mza","mzb","mzi","mzj","mzk","mzm",
    "mzn","mzw","mzy","nab","nag","nan","nas","naw","nba","nbc","nbh","nbi",
    "nbq","nbu","nca","nch","ncj","ncl","ncu","ndj","ndp","ndr","nds","ndy",
    "ndz","neb","new","nfa","nfr","nga","ngb","nge","ngp","ngu","nhe","nhi",
    "nhu","nhw","nhx","nhy","nia","nif","nim","nin","nit","niy","njm","njy",
    "nko","nla","nlc","nld","nle","nlg","nlk","nmz","nnb","nng","nnq","nnw",
    "noa","nob","nod","noe","nog","nor","nos","npi","npl","npy","nqo","nse",
    "nsk","nsn","nso","nss","nst","nsu","ntm","ntp","ntr","ntu","nus","nuy",
    "nwb","nxd","nxl","nyf","nyn","nyo","nyy","nza","nzi","obo","oci","ojb-script_latin",
    "ojb-script_syllabics","oku","old","omw","onb","ood","opm","or_","ory",
    "oss","ote","otq","ozm","pab","pad","pag","pam","pan","pao","pap","pau",
    "pbb","pbc","pbi","pbt","pby","pcm","peg","pes","pez","pib","pil","pir",
    "piw","pjt","pkb","pls","plt","pma","pmp","pms","pnb","pne","pnz","poc",
    "poh","poi","pol","pon","por","poy","pps","prf","prk","prs","pse","pss",
    "ptu","pui","puo","qub","quc","quf","quh","qul","qup","quw","quy","quz",
    "qva","qvc","qve","qvh","qvi","qvm","qvn","qvo","qvs","qvw","qvz","qwh",
    "qxh","qxl","qxn","qxo","qxr","rah","rai","rap","rav","raw","rej","rel",
    "rgu","rhg","rkb","rmn","rmo","rmy-script_cyrillic","rmy-script_latin",
    "rng","rnl","rob","roc","rod","roe","rom","ron","roo","rop","row","rro",
    "rub","ruf","rug","run","rus","rwo","sab","sag","sah","saj","san","saq",
    "sat","sau","sba","sbd","sbl","sbp","sch","sck","scn","sda","sea","seh",
    "ses","sey","sgb","sgj","sgw","shk","shn","shp","sid","sig","sil","sim",
    "sin","sja","sjp","skg","skr","slk","slv","sml","smk","snf","snk","snp",
    "snw","sny","sof","sok","som","soq","sot","soy","spa","spp","spy","sqi",
    "srm","srn","srp","srx","ssd","ssg","sst","ssx","stk","stn","stp","sua",
    "sue","suk","sun","sur","sus","suz","swa","swc","swe","swh","sxb","sxn",
    "syb","syl","syr","szl","tac","taj","tam","tao","tap","taq","tat","tav",
    "taw","tbc","tbf","tbg","tbk","tbl","tbo","tbz","tca","tcc","tcs","tcz",
    "tdj","ted","tel","tem","teo","ter","tes","tew","tex","tfr","tgk","tgl",
    "tgo","tgp","tha","thf","thk","thl","tif","tim","tio","tiv","tiy","tkr",
    "tlb","tlj","tly","tmc","tmf","tna","tng","tnk","tnn","tnp","tnt","tob",
    "toc","tod","tof","tog","toj","ton","too","top","tos","tpa","tpi","tpm",
    "tpp","tpt","tpz","trc","trn","tro","trq","trs","tsb","tsc","tsg","tsj",
    "tso","tsz","ttc","tte","ttj","ttq-script_latin","tuc","tue","tuf","tui",
    "tuk","tul","tuo","tur","tuv","tvw","twd","twu","txa","txq","txu","tye",
    "tzh","tzj","tzo","ubl","ubu","udm","udu","uig-script_arabic",
    "uig-script_latin","ukr","umb","und","unr","upv","ura","urb","urd","urk",
    "urt","ury","usp","uzb-script_cyrillic","uzb-script_latin","vag","vai",
    "vec","ven","vie","vif","vmw","vot","vun","vut","wal","war","wbm","wbr",
    "wed","wer","wes","wlx","wmw","wnk","wnu","wob","wol","wos","wrk","wrs",
    "wsg","wwa","xal","xav","xbi","xed","xer","xho","xla","xmm","xnj","xnr",
    "xog","xon","xpe","xrb","xsb","xsm","xsr","xsu","xtd","xte","xtm","yaa",
    "yad","yal","yam","yao","yap","yas","yat","yaz","ybe","ycl","ycn","yea",
    "yka","yon","yor","yrb","yre","yss","yua","yue","yuj","yut","yuw","zaa",
    "zab","zac","zad","zae","zai","zam","zao","zaq","zar","zas","zat","zav",
    "zaw","zca","zga","zhi","zhx","zia","ziw","zlm","zne","zos","zpc","zpg",
    "zpi","zpl","zpm","zpn","zpo","zpt","zpz","ztq","zty","zyb","zyp","zza",
]


def find_blank_id(vocab):
    for k in ("_", "<blank>", "BLANK"):
        if k in vocab:
            return vocab[k]
    if 0 in vocab.values():
        return 0
    return -1


def find_pad_id(vocab, tok_config):
    pt = tok_config.get("pad_token") if isinstance(tok_config, dict) else None
    if pt and pt in vocab:
        return vocab[pt]
    for k in ("<pad>", "[PAD]", "PAD"):
        if k in vocab:
            return vocab[k]
    return -1


def vocab_is_latin_only(vocab):
    for tok in vocab.keys():
        for ch in tok:
            if ord(ch) > 0x7F:
                return False
    return True


def write_vocab_bin(path, vocab, tok_config, add_blank, is_uroman):
    blank_id = find_blank_id(vocab)
    pad_id = find_pad_id(vocab, tok_config)
    unk_id = vocab.get("<unk>", -1)
    with path.open("wb") as f:
        f.write(struct.pack("<I", VOCAB_MAGIC))
        f.write(struct.pack("<I", VOCAB_VERSION))
        f.write(struct.pack("<I", len(vocab)))
        f.write(struct.pack("<i", pad_id))
        f.write(struct.pack("<i", unk_id))
        f.write(struct.pack("<i", blank_id))
        f.write(struct.pack("<B", 1 if add_blank else 0))
        f.write(struct.pack("<B", 1 if is_uroman else 0))
        f.write(struct.pack("<H", 0))
        for tok, tid in sorted(vocab.items(), key=lambda kv: kv[1]):
            utf8 = tok.encode("utf-8")
            f.write(struct.pack("<iI", tid, len(utf8)))
            f.write(utf8)


def convert_one(code, work_dir, packs_dir):
    from huggingface_hub import snapshot_download
    from safetensors import safe_open

    hf_id = f"facebook/mms-tts-{code}"
    weights_dir = work_dir / "weights" / code
    weights_dir.mkdir(parents=True, exist_ok=True)

    local = Path(snapshot_download(repo_id=hf_id, allow_patterns=[
        "*.safetensors", "pytorch_model.bin", "config.json",
        "vocab.json", "tokenizer_config.json", "special_tokens_map.json",
    ], max_workers=1))

    config = json.loads((local / "config.json").read_text())
    vocab = json.loads((local / "vocab.json").read_text())
    tok_config_path = local / "tokenizer_config.json"
    tok_config = json.loads(tok_config_path.read_text()) if tok_config_path.exists() else {}

    bin_path = weights_dir / "model.bin"
    tensors = {}
    offset = 0
    sha = hashlib.sha256()
    safetensors_files = sorted(local.glob("*.safetensors"))

    with bin_path.open("wb") as bf:
        if safetensors_files:
            for st in safetensors_files:
                with safe_open(st, framework="pt") as f:
                    for name in f.keys():
                        t = f.get_tensor(name)
                        if not t.dtype.is_floating_point:
                            continue
                        import torch
                        arr = t.detach().to("cpu").float().numpy()
                        raw = arr.tobytes(order="C")
                        bf.write(raw)
                        sha.update(raw)
                        tensors[name] = {
                            "offset": offset, "shape": list(arr.shape),
                            "dtype": "float32",
                            "num_elements": int(arr.size),
                            "size_bytes": len(raw),
                        }
                        offset += len(raw)
        else:
            import torch
            sd = torch.load(local / "pytorch_model.bin", map_location="cpu", weights_only=True)
            for name, t in sd.items():
                if not t.is_floating_point():
                    continue
                arr = t.float().numpy()
                raw = arr.tobytes(order="C")
                bf.write(raw)
                sha.update(raw)
                tensors[name] = {
                    "offset": offset, "shape": list(arr.shape),
                    "dtype": "float32",
                    "num_elements": int(arr.size),
                    "size_bytes": len(raw),
                }
                offset += len(raw)

    fp16_bin = weights_dir / "model.fp16.bin"
    fp16_tensors = {}
    cursor = 0
    sha16 = hashlib.sha256()
    with bin_path.open("rb") as src, fp16_bin.open("wb") as dst:
        ordered = sorted(tensors.items(), key=lambda kv: kv[1]["offset"])
        for name, info in ordered:
            src.seek(info["offset"])
            raw32 = src.read(info["size_bytes"])
            arr32 = np.frombuffer(raw32, dtype=np.float32).reshape(info["shape"]) if info["shape"] else np.frombuffer(raw32, dtype=np.float32)
            arr16 = arr32.astype(np.float16)
            raw16 = arr16.tobytes(order="C")
            dst.write(raw16)
            sha16.update(raw16)
            fp16_tensors[name] = {
                "offset": cursor, "shape": info["shape"],
                "dtype": "float16",
                "num_elements": info["num_elements"],
                "size_bytes": len(raw16),
            }
            cursor += len(raw16)

    meta16 = {
        "model_id": hf_id, "format": "binary", "layout": "row_major",
        "dtype": "float16",
        "quantization": {"enabled": True, "bits": 16, "method": "float16"},
        "bin_sha256": sha16.hexdigest(),
        "bin_size_bytes": cursor, "tensors": fp16_tensors, "total_bytes": cursor,
    }
    (weights_dir / "model.fp16.meta.json").write_text(json.dumps(meta16, indent=2))

    is_latin = vocab_is_latin_only(vocab)
    write_vocab_bin(weights_dir / "tokenizer_vocab.bin", vocab, tok_config,
                    add_blank=bool(tok_config.get("add_blank", True)),
                    is_uroman=bool(tok_config.get("is_uroman", is_latin)))

    # Delete fp32 model.bin immediately to save disk
    bin_path.unlink(missing_ok=True)
    (weights_dir / "model.meta.json").unlink(missing_ok=True)

    # Zip the runtime files
    out_zip = packs_dir / f"mms-tts-{code}.zip"
    tmp_zip = out_zip.with_suffix(".zip.tmp")
    runtime_files = [
        weights_dir / "model.fp16.bin",
        weights_dir / "model.fp16.meta.json",
        weights_dir / "tokenizer_vocab.bin",
    ]
    with zipfile.ZipFile(tmp_zip, "w", compression=zipfile.ZIP_STORED) as zf:
        for f in runtime_files:
            if f.exists():
                zf.write(f, arcname=f"weights/{code}/{f.name}")
    tmp_zip.rename(out_zip)

    # Cleanup
    shutil.rmtree(weights_dir, ignore_errors=True)
    return out_zip.stat().st_size


def try_lang_meta(code):
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


def build_languages_json(codes, packs_dir):
    entries = []
    for code in codes:
        zp = packs_dir / f"mms-tts-{code}.zip"
        if not zp.exists():
            continue
        meta = try_lang_meta(code)
        entries.append({
            "code": code,
            "name": meta["name"],
            "native_name": meta["native_name"],
            "script": meta["script"],
            "size_bytes": zp.stat().st_size,
        })
    entries.sort(key=lambda e: (e["name"].lower(), e["code"]))
    out = packs_dir / "languages.json"
    out.write_text(json.dumps(entries, ensure_ascii=False, indent=2))
    return out, len(entries)


def fetch_already_uploaded():
    """Check which zips are already on HuggingFace so we can skip them."""
    try:
        from huggingface_hub import HfApi
        api = HfApi()
        files = api.list_repo_files(HF_REPO)
        uploaded = set()
        for f in files:
            if f.startswith("mms-tts/mms-tts-") and f.endswith(".zip"):
                code = f.replace("mms-tts/mms-tts-", "").replace(".zip", "")
                uploaded.add(code)
        return uploaded
    except Exception as e:
        print(f"Warning: could not fetch HF file list: {e}")
        return set()


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--limit", type=int, default=None,
                   help="process at most N languages")
    p.add_argument("--codes", nargs="+", default=None,
                   help="explicit list of codes to process")
    p.add_argument("--resume-after", default=None,
                   help="start after this code (skip it and everything before)")
    p.add_argument("--skip-uploaded", action="store_true", default=True,
                   help="skip languages already on HuggingFace (default: true)")
    p.add_argument("--no-skip-uploaded", action="store_false", dest="skip_uploaded")
    p.add_argument("--upload", action="store_true", default=True,
                   help="upload packs to HF after conversion (default: true)")
    p.add_argument("--no-upload", action="store_false", dest="upload")
    p.add_argument("--upload-every", type=int, default=50,
                   help="upload intermediate batch every N languages (default 50)")
    p.add_argument("--work-dir", type=str, default="./mms_tts_work",
                   help="working directory for intermediate files")
    args, _ = p.parse_known_args()

    work_dir = Path(args.work_dir).resolve()
    packs_dir = work_dir / "packs"
    packs_dir.mkdir(parents=True, exist_ok=True)

    if args.codes:
        codes = args.codes
    else:
        codes = list(ALL_LANGS)
        if args.resume_after:
            try:
                idx = codes.index(args.resume_after)
                codes = codes[idx + 1:]
            except ValueError:
                print(f"ERROR: --resume-after {args.resume_after} not in list")
                return 1

    already_uploaded = set()
    if args.skip_uploaded:
        print("Checking HuggingFace for already-uploaded packs...")
        already_uploaded = fetch_already_uploaded()
        print(f"  {len(already_uploaded)} already on HF")

    # Filter out already-done
    todo = []
    for c in codes:
        if c in already_uploaded:
            continue
        if (packs_dir / f"mms-tts-{c}.zip").exists():
            continue
        todo.append(c)

    if args.limit:
        todo = todo[:args.limit]

    print(f"\n{'='*60}")
    print(f" {len(todo)} languages to convert")
    print(f" {len(already_uploaded)} already on HF (skipped)")
    print(f" packs → {packs_dir}")
    print(f"{'='*60}\n")

    if not todo:
        print("Nothing to do!")
        return 0

    failed = []
    ok_count = 0
    started = time.time()

    for i, code in enumerate(todo, 1):
        prefix = f"[{i:4d}/{len(todo)}] {code:30s}"
        try:
            t0 = time.time()
            size = convert_one(code, work_dir, packs_dir)
            elapsed = time.time() - t0
            size_mb = size / 1024 / 1024
            ok_count += 1
            eta_min = (time.time() - started) / i * (len(todo) - i) / 60
            print(f"{prefix} OK · {size_mb:.1f} MB · {elapsed:.0f}s · ETA {eta_min:.0f} min")
        except Exception as e:
            failed.append((code, str(e)))
            print(f"{prefix} FAILED: {e}")

        # Intermediate upload
        if args.upload and i % args.upload_every == 0:
            print(f"\n--- uploading batch ({ok_count} packs so far) ---")
            _do_upload(packs_dir)
            print("--- upload done ---\n")

    # Build languages.json covering ALL languages (uploaded + new)
    all_for_registry = list(ALL_LANGS)
    # Download existing languages.json from HF and merge
    lang_json, n_entries = build_languages_json(all_for_registry, packs_dir)
    print(f"\nlanguages.json: {n_entries} entries (local packs only)")

    if args.upload:
        print("\nFinal upload...")
        _do_upload(packs_dir)

    elapsed = time.time() - started
    print(f"\nDone in {elapsed/60:.1f} min · {ok_count}/{len(todo)} OK · {len(failed)} failed")
    if failed:
        print("\nFailures:")
        for code, err in failed[:20]:
            print(f"  {code}: {err}")
        if len(failed) > 20:
            print(f"  ... and {len(failed) - 20} more")
    return 1 if failed else 0


def _do_upload(packs_dir):
    import subprocess
    result = subprocess.run(
        ["huggingface-cli", "upload", HF_REPO, str(packs_dir), HF_PREFIX + "/"],
        capture_output=True, text=True,
    )
    if result.returncode != 0:
        print(f"Upload error: {result.stderr}")
    else:
        print(f"Upload OK: {result.stdout.strip()[:200]}")


if __name__ == "__main__":
    sys.exit(main())
