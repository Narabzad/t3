#!/usr/bin/env python3
"""
Build and push the T3 RAG-hints dataset to HuggingFace.

Schema (one row per trajectory):
  question     str    — original math question
  answer       str    — correct answer (empty for proof problems)
  subject      str    — math subject
  level        int    — difficulty level
  cot_type     str    — e.g. "math"
  trace        str    — original reasoning trace
  p_cheatsheet list[str]  — cheatsheet-style hints (1+ per trace)
  p_contrastive list[str] — contrastive hint (1 per trace)
  p_multipass  list[str]  — multi-pass hint (1 per trace)

Usage:
    python create_hf_dataset.py \\
        --input   trajectories_with_questions_58k.jsonl \\
        --outdir  outputs/ \\
        --repo    narabzad/t3-rag \\
        [--push]
"""

import argparse
import ast
import json
from pathlib import Path

from datasets import Dataset


def _clean(v) -> str:
    """Stringify and strip; return '' for None/null/None-string."""
    s = str(v).strip() if v is not None else ""
    return "" if s.lower() in ("none", "null", "") else s


def extract_answer(meta: dict) -> str:
    """Extract the best available answer string from a metadata dict."""
    # Standard S1K-style: direct answer key
    v = _clean(meta.get("answer"))
    if v:
        return v
    # GPQA-style: 'Correct Answer' (or revised variant)
    v = _clean(meta.get("Extra Revised Correct Answer") or meta.get("Correct Answer"))
    if v:
        return v
    # Multiple-choice with label index into options list
    if "label" in meta:
        opts = meta.get("options", [])
        if isinstance(opts, list) and opts:
            try:
                v = _clean(opts[int(meta["label"])])
                if v:
                    return v
            except (ValueError, IndexError):
                pass
        v = _clean(meta.get("label"))
        if v:
            return v
    # OlympiadBench-style: final_answer
    v = _clean(meta.get("final_answer"))
    if v:
        return v
    # Proof / open-ended problems (aops_forum messages, USACO, etc.) have no
    # single answer — return empty string intentionally.
    return ""


def parse_metadata(raw) -> dict:
    if isinstance(raw, dict):
        return raw
    if isinstance(raw, str):
        try:
            return ast.literal_eval(raw)
        except Exception:
            pass
        try:
            return json.loads(raw)
        except Exception:
            pass
    return {}


def load_passages(path: Path) -> dict[int, list[str]]:
    """Return {_idx: [passage, ...]} sorted by _passage_idx."""
    idx_map: dict[int, list[tuple[int, str]]] = {}
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            obj = json.loads(line)
            i = obj["_idx"]
            p_i = obj.get("_passage_idx", 0)
            passage = obj["passage"]
            idx_map.setdefault(i, []).append((p_i, passage))
    return {i: [p for _, p in sorted(v)] for i, v in idx_map.items()}


def build(input_path: Path, outdir: Path) -> Dataset:
    originals = []
    with open(input_path) as f:
        for line in f:
            line = line.strip()
            if line:
                originals.append(json.loads(line))

    cheatsheet  = load_passages(outdir / "p_cheatsheet.jsonl")
    contrastive = load_passages(outdir / "p_contrastive.jsonl")
    multipass   = load_passages(outdir / "p_multipass.jsonl")

    rows = []
    for idx, rec in enumerate(originals):
        meta = parse_metadata(rec.get("metadata", {}))

        rows.append({
            "question":      rec.get("question", ""),
            "answer":        extract_answer(meta),
            "subject":       _clean(meta.get("subject", "")),
            "level":         int(meta["level"]) if meta.get("level") is not None else 0,
            "cot_type":      rec.get("cot_type", ""),
            "trace":         rec.get("text", ""),
            "p_cheatsheet":  cheatsheet.get(idx, []),
            "p_contrastive": contrastive.get(idx, []),
            "p_multipass":   multipass.get(idx, []),
        })

    return Dataset.from_list(rows)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input",  default="trajectories_with_questions_58k.jsonl")
    parser.add_argument("--outdir", default="outputs")
    parser.add_argument("--repo",   default="narabzad/t3-rag")
    parser.add_argument("--push",   action="store_true", help="Push to HuggingFace Hub")
    args = parser.parse_args()

    input_path = Path(args.input)
    outdir = Path(args.outdir)

    print(f"Building dataset from {input_path} + {outdir}/p_*.jsonl …")
    ds = build(input_path, outdir)

    # Quick sanity check
    n_with_answer = sum(1 for r in ds if r["answer"])
    print(f"\nDataset: {ds}")
    print(f"Records with non-empty answer: {n_with_answer}/{len(ds)} "
          f"({100*n_with_answer/len(ds):.1f}%)")
    print(f"(Remaining are proof/open-ended problems without a single answer)\n")

    print("First example:")
    ex = ds[0]
    for k, v in ex.items():
        if isinstance(v, list):
            print(f"  {k}: list({len(v)}) → {str(v[0])[:80]}...")
        elif isinstance(v, str) and len(v) > 80:
            print(f"  {k}: {v[:80]}...")
        else:
            print(f"  {k}: {v}")

    if args.push:
        print(f"\nPushing to {args.repo} …")
        ds.push_to_hub(args.repo, private=False)
        print("Done.")
    else:
        print("\nDry run complete (pass --push to upload to HuggingFace).")


if __name__ == "__main__":
    main()
