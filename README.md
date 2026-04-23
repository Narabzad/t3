# T3: Thinking-Trace Retrieval for Reasoning Benchmarks

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
