# T3: Thinking-Trace Retrieval for Reasoning Benchmarks

<p align="center">
  <a href="https://huggingface.co/datasets/narabzad/t3-rag"><img src="https://img.shields.io/badge/🤗%20Dataset-narabzad/t3--rag-yellow" alt="HuggingFace Dataset"></a>
  &nbsp;
  <a href="#"><img src="https://img.shields.io/badge/📄%20Paper-coming%20soon-lightgrey" alt="Paper"></a>
</p>

This repository contains data, retrieval results, evaluation code, and evaluated results for studying retrieval-augmented generation (RAG) on reasoning benchmarks.

## Repository Structure

```
t3/
├── data/
│   ├── queries/                      # Benchmark question sets
│   │   ├── aime_2025_2026_queries.jsonl
│   │   ├── gpqa_queries.jsonl
│   │   └── lcb_v4_minus_v2_queries.jsonl
│   └── retrieved_results/            # Top-3 retrieved passages per question per method
│       ├── aime_2025_2026/           # AIME 2025 & 2026 (60 problems)
│       ├── gpqa/                     # GPQA Diamond
│       └── lcb_v4/                   # LiveCodeBench v4
├── eval/
│   ├── tasks/                        # lm-evaluation-harness task configs
│   │   ├── aime/
│   │   ├── gpqa/
│   │   └── lcb/
│   └── scripts/
│       ├── run_eval.sh               # Main eval runner (MODEL/BENCH/PART env vars)
│       ├── launch_all.sh             # Launches all 18 eval sessions in screen
│       └── gemini_proxy_server.py    # Proxy server for Gemini API
└── results/
    ├── aime_2025/
    │   ├── no_retrieval/             # Baseline (no retrieved context)
    │   └── with_retrieval/           # RAG results per method
    ├── aime_2026/
    ├── gpqa/
    └── lcb/
```

## Benchmarks

| Benchmark | Description | Problems | Eval setting |
|-----------|-------------|----------|--------------|
| AIME 2025-2026 | AMC/AIME math competitions | 60 total | 8 samples/question (agg@8) |
| GPQA Diamond | Graduate-level science questions | 198 | 4 samples/question (agg@4) |
| LiveCodeBench v4 | Competitive programming (2024-04 → 2024-09) | ~120 | 4 samples/question (agg@4) |

## Retrieval Methods

Each method name in `data/retrieved_results/` encodes the corpus and retriever:

| Prefix | Corpus |
|--------|--------|
| `compactds_arxiv_e5base_512` | CompactDS arXiv subset, e5-base, chunk=512 |
| `compactds_dpr_wiki_e5base_512` | CompactDS DPR-Wikipedia, e5-base, chunk=512 |
| `compactds_github_e5base_512` | CompactDS GitHub, e5-base, chunk=512 |
| `compactds_rpj_wiki_e5base_512` | CompactDS RPJ-Wikipedia, e5-base, chunk=512 |
| `openwebmath_e5base_512/full` | OpenWebMath, e5-base, chunk=512 or full |
| `stackexchange_e5base_512/full` | StackExchange, e5-base, chunk=512 or full |
| `s1k_59k_attempt_e5base_512/full` | S1K 59K attempt traces, e5-base |
| `s1k_deepseek_attempt/thinking_e5base` | S1.1K DeepSeek attempt/thinking traces, e5-base |
| `s1k_gemini_attempt/thinking_e5base` | S1.1K Gemini attempt/thinking traces, e5-base |
| `searchengine_parallel/tavily_decontam` | Live search engine results (decontaminated) |
| `p_cheatsheet/contrastive/multipass_e5base_full` | Prompt-augmented retrieval variants |
| `trajectories_qwq32b_e5base_256/512/full` | QwQ-32B reasoning traces, e5-base |
| `trajectories_gptoss120b_e5base_256/512/full` | GPT-OSS-120B reasoning traces, e5-base |

All retrieval uses **top-3** passages.

## Models Evaluated

| Model | Provider | Name in results/ |
|-------|----------|-----------------|
| GPT-5 (2025-08-07) | OpenAI | `gpt5` |
| Gemini 2.5 Flash | Google | `gemini-2.5-flash` |
| GPT-OSS-120B (DeepInfra BF16) | OpenRouter | `gpt-oss-120b` |

## Results Format

Each `results/json` file is the raw output from lm-evaluation-harness, containing:
- `results`: metric scores (exact_match, cov@8, maj@8, etc.)
- `config`: eval configuration (model, temperature, max_gen_toks, etc.)

## Running Evaluations

### Prerequisites

```bash
# Install lm-evaluation-harness
pip install -e git+https://github.com/EleutherAI/lm-evaluation-harness#egg=lm-eval

# Set API keys
export OPENAI_API_KEY=...
export LMEVAL_API_KEY=...    # OpenRouter or Gemini
```

### With retrieval

```bash
cd eval/scripts
MODEL=gpt5 BENCH=aime PART=1 bash run_eval.sh
```

- `MODEL`: `gpt5` | `gemini` | `oss120b`
- `BENCH`: `aime` | `lcb` | `gpqa`
- `PART`: `1` (first half of files) | `2` (second half)
- `POOL`: parallelism (default 10, auto-reduces on rate limits)

### Without retrieval

Set `RETRIEVAL_FILE_PATH` to empty or omit; run the non-retrieval task variant (e.g., `aime25_nofigures_agg8`).

## Data Format

Query files (`.jsonl`): each line is a JSON object with fields:
- `id`: unique question identifier
- `problem`: question text
- `answer`: ground-truth answer
- `query`: retrieval query string

Retrieved results files (`.jsonl`): same fields plus:
- `ctxs`: list of retrieved passages, each with `id`, `title`, `text`, `score`

## Transformation Code

`data_transform/` contains the scripts to reproduce the `p_cheatsheet`, `p_contrastive`, and `p_multipass` passages from raw reasoning traces:

```bash
# Install deps
pip install openai tqdm

# Run all three transformations on a trajectories file
python data_transform/run_all_prompts_gpt.py \
    --input  trajectories_with_questions_58k.jsonl \
    --outdir outputs/ \
    --prompts p_cheatsheet p_contrastive p_multipass \
    --model  gpt-5-nano \
    --concurrency 50
```

The script is resume-safe: re-running picks up where it left off.

### Prompt descriptions

| Prompt | Description |
|--------|-------------|
| `p_cheatsheet` | Step-by-step cheatsheet per solution approach (can produce multiple per trace) |
| `p_contrastive` | Compares a correct approach with a common mistake |
| `p_multipass` | Multi-pass structured hint building up solution incrementally |

## HuggingFace Dataset

The full dataset (58,071 trajectories with all three transformations) is on HuggingFace:

```python
from datasets import load_dataset
ds = load_dataset("narabzad/t3-rag")
# Columns: question, answer, subject, level, cot_type,
#          trace, p_cheatsheet (list), p_contrastive (list), p_multipass (list)
```

To rebuild and re-push:
```bash
cd data_transform
HF_TOKEN=<your_token> python create_hf_dataset.py \
    --input ../trajectories_with_questions_58k.jsonl \
    --outdir ../outputs/ \
    --push
```

## Simple Single-Run Eval

`eval/scripts/run_single_eval.sh` runs one benchmark with or without retrieval:

```bash
# With retrieval
bash eval/scripts/run_single_eval.sh \
    --model gpt5 \
    --bench gpqa \
    --retrieval data/retrieved_results/gpqa/p_cheatsheet_e5base_full.jsonl

# No-RAG baseline
bash eval/scripts/run_single_eval.sh \
    --model gpt5 \
    --bench aime \
    --no-rag

# Gemini with retrieval
bash eval/scripts/run_single_eval.sh \
    --model gemini \
    --bench lcb \
    --retrieval data/retrieved_results/lcb_v4/p_cheatsheet_e5base_full.jsonl
```

Required env vars: `OPENAI_API_KEY` (gpt5), `OPENROUTER_API_KEY` (oss120b), `GEMINI_API_KEY` (gemini).
