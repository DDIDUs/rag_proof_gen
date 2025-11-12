#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
import json
import re
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Iterable, Tuple, Optional

CODE_FENCE_RE = re.compile(r"```isabelle\s*(.*?)```", re.S)
LEMMA_NAME_RE = re.compile(r"\blemma\s+([A-Za-z0-9_']+)")
# Matches from "lemma <name>" up to next "lemma <something>" OR "end" at start of line.
LEMMA_BLOCK_RE_TMPL = r"(?s)(^|\n)(lemma\s+{name}\b.*?)(?=\nlemma\s|\nend\s*$)"

# -------- Progress utilities --------
class Progress:
    def __init__(self, total: int, mode: str = "auto", label: str = "items"):
        self.total = total
        self.count = 0
        self.mode = mode
        self.label = label
        self._t0 = time.time()
        self._tqdm = None
        if mode == "auto":
            try:
                from tqdm import tqdm  # type: ignore
                self._tqdm = tqdm(total=total, unit=label)
                self.mode = "tqdm"
            except Exception:
                self.mode = "simple"
        elif mode == "tqdm":
            from importlib import util
            if util.find_spec("tqdm") is None:
                self.mode = "simple"
            else:
                from tqdm import tqdm  # type: ignore
                self._tqdm = tqdm(total=total, unit=label)
        elif mode == "none":
            self.mode = "none"
        else:
            self.mode = "simple"

    def update_line(self, msg: str):
        if self.mode == "tqdm":
            assert self._tqdm is not None
            self._tqdm.set_description_str(msg, refresh=True)
        elif self.mode == "simple":
            # carriage return single-line update
            elapsed = time.time() - self._t0
            pct = (self.count / self.total * 100) if self.total else 0
            line = f"[{self.count}/{self.total} {pct:5.1f}% +{elapsed:5.1f}s] {msg}"
            sys.stderr.write("\r" + line[:200] + " " * 10)
            sys.stderr.flush()

    def step(self):
        self.count += 1
        if self.mode == "tqdm":
            assert self._tqdm is not None
            self._tqdm.update(1)
        elif self.mode == "simple":
            # force newline after finishing the step line
            sys.stderr.write("\n")
            sys.stderr.flush()

    def close(self):
        if self.mode == "tqdm" and self._tqdm is not None:
            self._tqdm.close()
        elif self.mode == "simple":
            sys.stderr.write("\n")
            sys.stderr.flush()

# -------- Core helpers --------
def strip_isabelle_fence(s: str) -> str:
    if not s:
        return ""
    m = CODE_FENCE_RE.search(s)
    return (m.group(1) if m else s).strip()

def lemma_name_from_code(code: str) -> Optional[str]:
    m = LEMMA_NAME_RE.search(code)
    return m.group(1) if m else None

def lemma_name_from_input(inp: str) -> Optional[str]:
    if not inp:
        return None
    m = LEMMA_NAME_RE.search(inp)
    return m.group(1) if m else None

def replace_lemma_block(thy_text: str, lemma_name: str, new_block: str) -> Tuple[str, bool]:
    pattern = re.compile(LEMMA_BLOCK_RE_TMPL.format(name=re.escape(lemma_name)))
    m = pattern.search(thy_text)
    if not m:
        return thy_text, False
    prefix = thy_text[:m.start(2)]
    suffix = thy_text[m.end(2):]
    if not new_block.endswith("\n"):
        new_block = new_block + "\n"
    return prefix + new_block + suffix, True

def run_isabelle_build(root: Path, session: str, extra_args: Optional[list] = None, timeout: int = 1800):
    cmd = ["isabelle", "build", "-d", str(root), "-b", session]
    if extra_args:
        cmd.extend(extra_args)
    proc = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout)
    return proc.returncode, proc.stdout, proc.stderr

def tail(s: str, n: int = 2000) -> str:
    return s if len(s) <= n else s[-n:]

def load_items(jsonl_path: Path) -> Iterable[dict]:
    text = jsonl_path.read_text(encoding="utf-8").strip()
    # Try JSON array
    try:
        data = json.loads(text)
        if isinstance(data, list):
            for obj in data:
                if isinstance(obj, dict):
                    yield obj
            return
    except Exception:
        pass
    # Fallback JSONL
    for line in text.splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            obj = json.loads(line)
            if isinstance(obj, dict):
                yield obj
        except Exception:
            yield {"_raw": line, "error": "invalid jsonl line"}

def count_items(jsonl_path: Path) -> int:
    text = jsonl_path.read_text(encoding="utf-8").strip()
    try:
        data = json.loads(text)
        if isinstance(data, list):
            return sum(1 for x in data if isinstance(x, dict))
    except Exception:
        pass
    # JSONL count
    c = 0
    for line in text.splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            obj = json.loads(line)
            if isinstance(obj, dict):
                c += 1
        except Exception:
            c += 1
    return c

# -------- Main --------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--jsonl", required=True, help="Input JSON/JSONL path")
    ap.add_argument("--thy", required=True, help="Target .thy file to patch")
    ap.add_argument("--session", required=True, help="Isabelle session name, e.g., CorresK")
    ap.add_argument("--root", default=".", help="Isabelle project root for -d (must contain ROOT/ROOTS)")
    ap.add_argument("--out", default="./build_report.jsonl", help="Output JSONL report path")
    ap.add_argument("--timeout", type=int, default=1800, help="Build timeout in seconds")
    ap.add_argument("--stop_on_success", action="store_true", help="Stop after first success per item")
    ap.add_argument("--dry_run", action="store_true", help="Do not run build; only report planned replacements")
    ap.add_argument("--progress", default="auto", choices=["auto", "tqdm", "simple", "none"], help="Progress display mode")
    args = ap.parse_args()

    jsonl_path = Path(args.jsonl)
    thy_path = Path(args.thy)
    root_path = Path(args.root).resolve()
    out_path = Path(args.out)

    thy_orig = thy_path.read_text(encoding="utf-8")

    total = count_items(jsonl_path)
    prog = Progress(total=total, mode=args.progress, label="items")

    with out_path.open("w", encoding="utf-8") as fout:
        for idx, item in enumerate(load_items(jsonl_path), 1):
            # Progress header per item
            prog.update_line(f"start idx={idx}")

            if "_raw" in item and item.get("error"):
                fout.write(json.dumps({
                    "index": idx,
                    "error": item["error"],
                    "raw": item["_raw"],
                }, ensure_ascii=False) + "\n")
                prog.update_line(f"invalid jsonl idx={idx}")
                prog.step()
                continue

            inp_raw = item.get("input", "") or ""
            prf_raw = item.get("proof", "") or ""

            inp = strip_isabelle_fence(inp_raw)
            prf = strip_isabelle_fence(prf_raw)

            if not inp.strip():
                fout.write(json.dumps({"index": idx, "error": "missing_input"}, ensure_ascii=False) + "\n")
                prog.update_line(f"skip idx={idx} missing_input")
                prog.step()
                continue
            if not prf.strip():
                fout.write(json.dumps({
                    "index": idx,
                    "input_lemma_guess": lemma_name_from_input(inp),
                    "result": "no_proof",
                }, ensure_ascii=False) + "\n")
                prog.update_line(f"skip idx={idx} no_proof")
                prog.step()
                continue

            full_block = inp.strip()
            if not full_block.endswith("\n"):
                full_block += "\n"
            full_block += "\t" + prf.strip()

            lemma_name = lemma_name_from_input(full_block)
            if not lemma_name:
                fout.write(json.dumps({"index": idx, "error": "lemma_name_not_found"}, ensure_ascii=False) + "\n")
                prog.update_line(f"fail idx={idx} lemma_name_not_found")
                prog.step()
                continue

            new_thy_text, ok = replace_lemma_block(thy_orig, lemma_name, full_block)
            if not ok:
                fout.write(json.dumps({
                    "index": idx,
                    "lemma": lemma_name,
                    "error": f"lemma_block_not_found_in_file: {thy_path.name}",
                }, ensure_ascii=False) + "\n")
                prog.update_line(f"fail idx={idx} block_not_found {lemma_name}")
                prog.step()
                continue

            try:
                backup_path = thy_path.with_suffix(".thy.bak_tmp")
                backup_path.write_text(thy_orig, encoding="utf-8")

                thy_path.write_text(new_thy_text, encoding="utf-8")

                if args.dry_run:
                    result = {
                        "time": datetime.utcnow().isoformat() + "Z",
                        "index": idx,
                        "lemma": lemma_name,
                        "status": "dry_run",
                        "thy": str(thy_path),
                        "session": args.session,
                    }
                    fout.write(json.dumps(result, ensure_ascii=False) + "\n")
                    thy_path.write_text(thy_orig, encoding="utf-8")
                    if backup_path.exists():
                        backup_path.unlink(missing_ok=True)
                    prog.update_line(f"dry idx={idx} {lemma_name}")
                    prog.step()
                    continue

                prog.update_line(f"build idx={idx} {lemma_name}")
                rc, out, err = run_isabelle_build(root=root_path, session=args.session, timeout=args.timeout)
                success = (rc == 0) and (f"Finished {args.session}" in out)

                result = {
                    "time": datetime.utcnow().isoformat() + "Z",
                    "index": idx,
                    "lemma": lemma_name,
                    "returncode": rc,
                    "stdout_tail": tail(out, 4000),
                    "stderr_tail": tail(err, 4000),
                    "success": success,
                    "thy": str(thy_path),
                    "session": args.session,
                }
                fout.write(json.dumps(result, ensure_ascii=False) + "\n")
                fout.flush()

                prog.update_line(f"{'ok' if success else 'fail'} idx={idx} rc={rc} {lemma_name}")

            except subprocess.TimeoutExpired as te:
                thy_path.write_text(thy_orig, encoding="utf-8")
                fout.write(json.dumps({
                    "time": datetime.utcnow().isoformat() + "Z",
                    "index": idx,
                    "lemma": lemma_name,
                    "error": f"timeout: {te}",
                    "thy": str(thy_path),
                    "session": args.session,
                }, ensure_ascii=False) + "\n")
                prog.update_line(f"timeout idx={idx} {lemma_name}")
            except Exception as e:
                try:
                    thy_path.write_text(thy_orig, encoding="utf-8")
                except Exception:
                    pass
                fout.write(json.dumps({
                    "time": datetime.utcnow().isoformat() + "Z",
                    "index": idx,
                    "lemma": lemma_name,
                    "error": f"{type(e).__name__}: {e}",
                    "thy": str(thy_path),
                    "session": args.session,
                }, ensure_ascii=False) + "\n")
                prog.update_line(f"error idx={idx} {type(e).__name__}")
            finally:
                try:
                    thy_path.write_text(thy_orig, encoding="utf-8")
                except Exception:
                    pass
                backup = thy_path.with_suffix(".thy.bak_tmp")
                if backup.exists():
                    backup.unlink(missing_ok=True)
                prog.step()

    prog.close()

if __name__ == "__main__":
    main()
