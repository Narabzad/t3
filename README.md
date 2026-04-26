# T3: Transformation of Thinking Traces

<p align="center">
  <a href="https://huggingface.co/datasets/narabzad/t3-rag"><img src="https://img.shields.io/badge/🤗%20Dataset-narabzad/t3--rag-yellow" alt="HuggingFace Dataset"></a>
  &nbsp;
  <a href="#"><img src="https://img.shields.io/badge/📄%20Paper-coming%20soon-lightgrey" alt="Paper"></a>
</p>

RAG is widely believed to offer limited benefit for reasoning-intensive tasks like math and code. We challenge this assumption: **the limitation is the corpus, not the approach**. We show that retrieving *thinking traces* — intermediate reasoning trajectories from strong models — consistently improves performance across frontier models and benchmarks. We further introduce **T3**, an offline method that transforms raw traces into structured, retrieval-friendly representations, unlocking even stronger gains.

> **Key results on AIME 2025–2026:** RAG with Gemini-2-thinking traces improves Gemini-2.5-Flash by +50.1%, GPT-OSS-120B by +8.6%, and GPT-5 by +5.8%. T3 transformations also reduce inference cost by up to 15%.

![T3 Overview](figures/overview.png)

## How It Works

**Offline:** A strong reasoning model (e.g., Gemini-2-thinking, QwQ-32B) solves an auxiliary problem set and produces raw thinking traces. A smaller model (e.g., Gemini-2-Flash-Lite) then rewrites them into structured retrieval-friendly forms using T3.

**At inference time:** A new query is matched against this corpus. The top-3 retrieved passages are prepended to the prompt, and the solver model generates an answer — no fine-tuning required.

### T3 Transformations

| File prefix | Paper name | Description |
|-------------|------------|-------------|
| `t3_struct` | Structural Normalization | Rewrites traces into clean step-by-step procedural scaffolds; one trace can produce multiple passages |
| `t3_reflect` | Reflection | Contrastive form highlighting common mistakes, misleading paths, and how to avoid them |
| `t3_semantic` | Semantic Distillation | Multi-level abstraction; compresses traces to their core reasoning idea |

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
│       ├── gpqa/                     # GPQA Diamond (198 problems)
│       └── lcb_v4/                   # LiveCodeBench v4 (202 problems)
├── eval/
│   ├── tasks/                        # lm-evaluation-harness task configs
│   │   ├── aime/
│   │   ├── gpqa/
│   │   └── lcb/
│   └── scripts/
│       ├── run_single_eval.sh        # Eval runner
│       └── gemini_proxy_server.py    # Proxy server for Gemini API
└── data_transform/
    ├── README.md                     # How to apply T3 transformations
    ├── run_all_prompts_gpt.py        # Runs all three T3 transformations
    └── prompts/                      # Prompt templates for each transformation
```

## Benchmarks

| Benchmark | Description | Problems | Eval setting |
|-----------|-------------|----------|--------------|
| AIME 2025–2026 | AMC/AIME math competitions | 60 total | 8 samples/question (agg@8) |
| GPQA Diamond | Graduate-level science questions | 198 | 4 samples/question (agg@4) |
| LiveCodeBench v4 | Competitive programming (2024-04 → 2024-09) | 202 | 4 samples/question (agg@4) |

## Retrieval Corpora

Each subdirectory under `data/retrieved_results/` corresponds to one corpus+retriever combination:

| File prefix | Corpus |
|-------------|--------|
| `t3_struct_e5base_full` | T3 Structural Normalization, e5-base, full-doc |
| `t3_reflect_e5base_full` | T3 Reflection, e5-base, full-doc |
| `t3_semantic_e5base_full` | T3 Semantic Distillation, e5-base, full-doc |
| `trajectories_gemini2thinking_e5base_512` | Raw Gemini-2-thinking traces, e5-base, chunk=512 |
| `trajectories_qwq32b_e5base_{512,full}` | Raw QwQ-32B thinking traces, e5-base |
| `trajectories_gptoss120b_e5base_{512,full}` | Raw GPT-OSS-120B thinking traces, e5-base |
| `s1k_59k_attempt_e5base_{512,full}` | S1K 59K attempt traces, e5-base |
| `openwebmath_e5base_{512,full}` | OpenWebMath, e5-base |
| `stackexchange_e5base_{512,full}` | StackExchange, e5-base |
| `compactds_arxiv_e5base_512` | CompactDS arXiv subset, e5-base, chunk=512 |
| `compactds_dpr_wiki_e5base_512` | CompactDS DPR-Wikipedia, e5-base |
| `compactds_github_e5base_512` | CompactDS GitHub, e5-base |
| `compactds_rpj_wiki_e5base_512` | CompactDS RPJ-Wikipedia, e5-base |
| `searchengine_tavily_decontam` | Live web search results (Tavily, decontaminated) |

All retrieval uses **top-3** passages.

## Models Evaluated

| Model | Provider |
|-------|----------|
| GPT-5 (2025-08-07) | OpenAI |
| Gemini 2.5 Flash | Google |
| GPT-OSS-120B (DeepInfra BF16) | OpenRouter |

## Running Evaluations

### Prerequisites

```bash
# Install lm-evaluation-harness
pip install -e git+https://github.com/EleutherAI/lm-evaluation-harness#egg=lm-eval

# Set API keys
export OPENAI_API_KEY=...
export OPENROUTER_API_KEY=...
export GEMINI_API_KEY=...
```

### Run eval

```bash
# With retrieval
bash eval/scripts/run_single_eval.sh \
    --model gpt5 \
    --bench gpqa \
    --retrieval data/retrieved_results/gpqa/t3_struct_e5base_full.jsonl

# No-RAG baseline
bash eval/scripts/run_single_eval.sh \
    --model gpt5 \
    --bench aime \
    --no-rag

# Gemini with retrieval
bash eval/scripts/run_single_eval.sh \
    --model gemini \
    --bench lcb \
    --retrieval data/retrieved_results/lcb_v4/t3_reflect_e5base_full.jsonl
```

- `--model`: `gpt5` | `gemini` | `oss120b`
- `--bench`: `aime` | `lcb` | `gpqa`
- `--retrieval FILE` or `--no-rag`

## Data Format

**Query files** (`.jsonl`): each line has `id`, `problem`, `answer`, `query`.

**Retrieved results files** (`.jsonl`): same fields plus `ctxs` — a list of top-3 passages, each with `id`, `title`, `text`, `score`.

## T3 Transformation Code

See [`data_transform/README.md`](data_transform/README.md) for how to apply T3 transformations to your own thinking traces.

## HuggingFace Dataset

The full dataset (58,071 trajectories with all three T3 transformations) is on HuggingFace:

```python
from datasets import load_dataset
ds = load_dataset("narabzad/t3-rag")
# Columns: question, answer, subject, level, cot_type,
#          trace, t3_struct (list), t3_reflect (list), t3_semantic (list)
```
