"""
Run all prompts on a JSONL file efficiently using OpenAI GPT models.
- Records loaded once
- All API calls share a single semaphore (one rate limit pool)
- One live tqdm bar per prompt so you can watch all progress simultaneously
- Per-prompt output files, resume-safe

Usage:
    python run_all_prompts_gpt.py \
        --input  /path/to/trajectories.jsonl \
        --outdir /path/to/outputs/ \
        [--prompts p_summary p_cheatsheet p_multipass p_concept_strategy p_query_reasoning p_contrastive] \
        [--model  gpt-5-nano] \
        [--concurrency 50]

Set OPENAI_API_KEY env var or pass --api-key.
"""

import argparse
import asyncio
import json
import os
import re
from pathlib import Path

from openai import AsyncOpenAI
from tqdm import tqdm

PROMPTS_DIR = Path(__file__).parent / "prompts"
ALL_PROMPTS = [
    "p_summary",
    "p_cheatsheet",
    "p_multipass",
    "p_concept_strategy",
    "p_query_reasoning",
    "p_contrastive",
]
PASSAGE_SEP = "[PASSAGE]"


def load_prompt(name: str) -> str:
    path = PROMPTS_DIR / name
    if not path.exists():
        raise FileNotFoundError(f"Prompt not found: {path}")
    return path.read_text()


def split_passages(text: str) -> list[str]:
    parts = re.split(r"\[PASSAGE\]", text)
    return [p.strip() for p in parts if p.strip()]


async def call_api(client: AsyncOpenAI, model: str, prompt_text: str, semaphore: asyncio.Semaphore, retries: int = 5) -> str:
    for attempt in range(retries):
        try:
            async with semaphore:
                response = await client.chat.completions.create(
                    model=model,
                    messages=[{"role": "user", "content": prompt_text}],
                )
            return response.choices[0].message.content or ""
        except Exception as e:
            if attempt == retries - 1:
                return f"ERROR: {e}"
            await asyncio.sleep(2 ** attempt)


async def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input",       required=True)
    parser.add_argument("--outdir",      required=True, help="Directory for per-prompt output files")
    parser.add_argument("--prompts",     nargs="+", default=ALL_PROMPTS)
    parser.add_argument("--model",       default="gpt-5-nano")
    parser.add_argument("--concurrency", type=int, default=50, help="Total concurrent API calls across all prompts")
    parser.add_argument("--limit",       type=int, default=None, help="Process only first N records (for testing)")
    parser.add_argument("--api-key",     default=None, help="OpenAI API key (overrides OPENAI_API_KEY env var)")
    args = parser.parse_args()

    api_key = args.api_key or os.environ.get("OPENAI_API_KEY")
    if not api_key:
        raise ValueError("Provide --api-key or set OPENAI_API_KEY environment variable")

    input_path = Path(args.input)
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    # Load records once
    records = [json.loads(l) for l in input_path.read_text().splitlines() if l.strip()]
    if args.limit:
        records = records[:args.limit]
    print(f"Loaded {len(records)} records | model={args.model} | concurrency={args.concurrency}")
    print(f"Prompts: {args.prompts}\n")

    # Load prompt templates
    templates = {p: load_prompt(p) for p in args.prompts}

    # Resume: find already-done indices per prompt
    done: dict[str, set[int]] = {}
    for prompt in args.prompts:
        out_path = outdir / f"{prompt}.jsonl"
        done[prompt] = set()
        if out_path.exists():
            with open(out_path) as f:
                for line in f:
                    if line.strip():
                        try:
                            obj = json.loads(line)
                            if "_idx" in obj:
                                done[prompt].add(obj["_idx"])
                        except json.JSONDecodeError:
                            pass

    # Report resume state
    for prompt in args.prompts:
        remaining = len(records) - len(done[prompt])
        print(f"  {prompt}: {len(done[prompt])} done, {remaining} remaining")
    print()

    client = AsyncOpenAI(api_key=api_key)
    semaphore = asyncio.Semaphore(args.concurrency)

    # One write lock and tqdm bar per prompt
    write_locks = {p: asyncio.Lock() for p in args.prompts}
    out_files   = {p: open(outdir / f"{p}.jsonl", "a") for p in args.prompts}
    pbars       = {
        p: tqdm(
            total=len(records) - len(done[p]),
            desc=f"{p:<22}",
            position=i,
            leave=True,
        )
        for i, p in enumerate(args.prompts)
    }
    error_counts = {p: 0 for p in args.prompts}

    async def process(idx: int, prompt: str):
        record = records[idx]
        filled = templates[prompt].replace("{trace}", record.get("text", ""))
        raw = await call_api(client, args.model, filled, semaphore)
        passages = split_passages(raw)

        lines = [
            json.dumps({**record, "passage": passage, "prompt": prompt, "_idx": idx, "_passage_idx": p_idx})
            for p_idx, passage in enumerate(passages)
        ]

        if raw.startswith("ERROR:"):
            error_counts[prompt] += 1

        async with write_locks[prompt]:
            for line in lines:
                out_files[prompt].write(line + "\n")
            out_files[prompt].flush()

        pbars[prompt].update(1)

    # Build all tasks — interleave by record so all prompts get fair semaphore access
    tasks = [
        process(idx, prompt)
        for idx in range(len(records))
        for prompt in args.prompts
        if idx not in done[prompt]
    ]

    print(f"Total API calls to make: {len(tasks)}\n")
    await asyncio.gather(*tasks)

    # Cleanup
    for p in args.prompts:
        pbars[p].close()
        out_files[p].close()

    print("\n\nAll done!")
    for prompt in args.prompts:
        total_lines = sum(1 for _ in open(outdir / f"{prompt}.jsonl"))
        errs = error_counts[prompt]
        print(f"  {prompt}: {total_lines} passages written" + (f" | {errs} errors" if errs else ""))


if __name__ == "__main__":
    asyncio.run(main())
