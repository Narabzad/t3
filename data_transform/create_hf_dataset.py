#!/usr/bin/env python3
"""
Build and push the T3 RAG-hints dataset to HuggingFace.

Schema (one row per trajectory):
  unique_id    str    — from metadata.unique_id
  question     str    — original math question
  answer       str    — correct answer
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
        --repo    narabzad/t3-rag-hints \\
        [--push]
"""

import argparse
import json
from pathlib import Path

from datasets import Dataset


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
    # Load original trajectories
    originals = []
    with open(input_path) as f:
        for line in f:
            line = line.strip()
            if line:
                originals.append(json.loads(line))

    # Load transformed passages
    cheatsheet  = load_passages(outdir / "p_cheatsheet.jsonl")
    contrastive = load_passages(outdir / "p_contrastive.jsonl")
    multipass   = load_passages(outdir / "p_multipass.jsonl")

    rows = []
    for idx, rec in enumerate(originals):
        meta = rec.get("metadata", {})
        if isinstance(meta, str):
            try:
                meta = json.loads(meta.replace("'", '"'))
            except Exception:
                meta = {}

        rows.append({
            "unique_id":    meta.get("unique_id", ""),
            "question":     rec.get("question", ""),
            "answer":       str(meta.get("answer", "")),
            "subject":      meta.get("subject", ""),
            "level":        int(meta.get("level", 0)) if meta.get("level") is not None else 0,
            "cot_type":     rec.get("cot_type", ""),
            "trace":        rec.get("text", ""),
            "p_cheatsheet":  cheatsheet.get(idx, []),
            "p_contrastive": contrastive.get(idx, []),
            "p_multipass":   multipass.get(idx, []),
        })

    return Dataset.from_list(rows)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input",  default="trajectories_with_questions_58k.jsonl")
    parser.add_argument("--outdir", default="outputs")
    parser.add_argument("--repo",   default="narabzad/t3-rag-hints")
    parser.add_argument("--push",   action="store_true", help="Push to HuggingFace Hub")
    args = parser.parse_args()

    input_path = Path(args.input)
    outdir = Path(args.outdir)

    print(f"Building dataset from {input_path} + {outdir}/p_*.jsonl …")
    ds = build(input_path, outdir)

    print(f"\nDataset: {ds}")
    print(f"Features: {ds.features}")
    print(f"\nFirst example:")
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
