# T3: Transformation of Thinking Traces

<p align="center">
  <a href="https://huggingface.co/narabzad"><img src="https://img.shields.io/badge/🤗%20Datasets-narabzad-yellow" alt="HuggingFace Datasets"></a>
  &nbsp;
  <a href="#"><img src="https://img.shields.io/badge/📄%20Paper-coming%20soon-lightgrey" alt="Paper"></a>
</p>

RAG is widely believed to offer limited benefit for reasoning-intensive tasks like math and code. We challenge this assumption: **the limitation is the corpus, not the approach**. We show that retrieving *thinking traces* — intermediate reasoning trajectories from strong models — consistently improves performance across frontier models and benchmarks. We further introduce **T3**, an offline method that transforms raw traces into structured, retrieval-friendly representations, unlocking even stronger gains.

> **Key results on AIME 2025–2026:** RAG with Gemini-2-thinking traces improves Gemini-2.5-Flash by +50.1%, GPT-OSS-120B by +8.6%, and GPT-5 by +5.8%. T3 transformations also reduce inference cost by up to 15%.

![T3 Overview](figures/overview.png)

## How It Works

**Offline:** A strong reasoning model (e.g., Gemini-2-thinking, QwQ-32B) solves an auxiliary problem set and produces raw thinking traces. A smaller model (e.g., Gemini-2-Flash-Lite) then rewrites them into structured retrieval-friendly forms using T3.

**At inference time:** A previously unseen query is retrieved against this corpus; the top-*k* passages (e.g., k=3) are returned and used as context to perform RAG, enabling a downstream solver model to generate the final answer — no training or fine-tuning required.

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
│   └── traces/
│       ├── raw/                      # Raw thinking traces (114K OpenThoughts + 59K s1k)
│       │                             # → HuggingFace: narabzad/t3-traces-*
│       └── transformed/              # T3-transformed corpora
│                                     # → HuggingFace: narabzad/t3-struct-*, t3-reflect-*, t3-semantic-*
├── eval/
│   ├── tasks/                        # lm-evaluation-harness task configs
│   │   ├── aime/
│   │   ├── gpqa/
│   │   └── lcb/
│   └── scripts/
│       └── run_single_eval.sh        # Eval runner (all models via OpenRouter)
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

## HuggingFace Datasets

All datasets are published on [HuggingFace](https://huggingface.co/narabzad).

### Raw Thinking Traces

| Dataset | Model | Traces | Columns |
|---------|-------|--------|---------|
| [t3-traces-gemini2thinking](https://huggingface.co/datasets/narabzad/t3-traces-gemini2thinking) | Gemini-2-thinking | 58K | `question`, `trace` |
| [t3-traces-gptoss120b](https://huggingface.co/datasets/narabzad/t3-traces-gptoss120b) | GPT-OSS-120B | 57K | `question`, `trace` |
| [t3-traces-qwq32b](https://huggingface.co/datasets/narabzad/t3-traces-qwq32b) | QwQ-32B | 57K | `question`, `trace` |

### T3-Transformed Corpora

| Dataset | Transformation | Source | Passages | Columns |
|---------|---------------|--------|----------|---------|
| [t3-struct-gemini2thinking](https://huggingface.co/datasets/narabzad/t3-struct-gemini2thinking) | Structural Normalization | Gemini-2-thinking 58K | 78K | `question`, `trace`, `transformed_traces` |
| [t3-reflect-gemini2thinking](https://huggingface.co/datasets/narabzad/t3-reflect-gemini2thinking) | Reflection | Gemini-2-thinking 58K | 58K | `question`, `trace`, `transformed_traces` |
| [t3-semantic-gemini2thinking](https://huggingface.co/datasets/narabzad/t3-semantic-gemini2thinking) | Semantic Distillation | Gemini-2-thinking 58K | 58K | `question`, `trace`, `transformed_traces` |
| [t3-struct-qwq32b](https://huggingface.co/datasets/narabzad/t3-struct-qwq32b) | Structural Normalization | OpenThoughts QwQ-32B 114K | 155K | `question`, `trace`, `transformed_traces` |
| [t3-reflect-qwq32b](https://huggingface.co/datasets/narabzad/t3-reflect-qwq32b) | Reflection | OpenThoughts QwQ-32B 114K | 114K | `question`, `trace`, `transformed_traces` |
| [t3-semantic-qwq32b](https://huggingface.co/datasets/narabzad/t3-semantic-qwq32b) | Semantic Distillation | OpenThoughts QwQ-32B 114K | 114K | `question`, `trace`, `transformed_traces` |

```python
from datasets import load_dataset

# Raw thinking traces
ds = load_dataset("narabzad/t3-traces-gemini2thinking")
# Columns: question, trace

# T3-transformed passages
ds = load_dataset("narabzad/t3-struct-gemini2thinking")
# Columns: question, trace, transformed_traces (list)
```

## Retrieved Results

Pre-computed top-3 retrieved passages for each benchmark question are in `data/retrieved_results/`.

| File prefix | Corpus |
|-------------|--------|
| `t3_struct_e5base_full` | T3 Structural Normalization, e5-base, full-doc |
| `t3_reflect_e5base_full` | T3 Reflection, e5-base, full-doc |
| `t3_semantic_e5base_full` | T3 Semantic Distillation, e5-base, full-doc |
| `trajectories_gemini2thinking_e5base_512` | Raw Gemini-2-thinking traces, e5-base, chunk=512 |
| `trajectories_qwq32b_e5base_{512,full}` | Raw QwQ-32B thinking traces, e5-base |
| `trajectories_gptoss120b_e5base_{512,full}` | Raw GPT-OSS-120B thinking traces, e5-base |

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

## RAG Pipeline

You can run the full retrieve-then-generate pipeline on your own questions using any of the HuggingFace datasets above as the retrieval corpus.

### 1. Retrieve passages

```python
from datasets import load_dataset
from sentence_transformers import SentenceTransformer
import faiss, numpy as np

# Load a T3 corpus
corpus = load_dataset("narabzad/t3-struct-gemini2thinking", split="train")
docs   = [p for row in corpus for p in row["passages"]]

# Build FAISS index
model = SentenceTransformer("intfloat/e5-base-v2")
vecs  = model.encode([f"passage: {d}" for d in docs], batch_size=256, show_progress_bar=True)
index = faiss.IndexFlatIP(vecs.shape[1])
faiss.normalize_L2(vecs)
index.add(vecs)

def retrieve(query: str, top_k: int = 3) -> list[str]:
    q_vec = model.encode([f"query: {query}"])
    faiss.normalize_L2(q_vec)
    _, ids = index.search(q_vec, top_k)
    return [docs[i] for i in ids[0]]
```

### 2. Generate with retrieved context

```python
# ── OpenAI ──────────────────────────────────────────────────────────────────
from openai import OpenAI
client = OpenAI()  # uses OPENAI_API_KEY

def rag_openai(question: str, model: str = "gpt-4o") -> str:
    ctx = "\n\n".join(retrieve(question))
    prompt = f"Use the following examples to help solve the problem.\n\n{ctx}\n\nProblem: {question}"
    resp = client.chat.completions.create(
        model=model,
        messages=[{"role": "user", "content": prompt}],
    )
    return resp.choices[0].message.content

# ── Google Gemini ────────────────────────────────────────────────────────────
import google.generativeai as genai
genai.configure(api_key=os.environ["GEMINI_API_KEY"])

def rag_gemini(question: str, model: str = "gemini-2.5-flash") -> str:
    ctx = "\n\n".join(retrieve(question))
    prompt = f"Use the following examples to help solve the problem.\n\n{ctx}\n\nProblem: {question}"
    resp = genai.GenerativeModel(model).generate_content(prompt)
    return resp.text

# ── OpenRouter (open-source models) ─────────────────────────────────────────
from openai import OpenAI as OpenRouterClient
router = OpenRouterClient(
    base_url="https://openrouter.ai/api/v1",
    api_key=os.environ["OPENROUTER_API_KEY"],
)

def rag_openrouter(question: str, model: str = "qwen/qwq-32b") -> str:
    ctx = "\n\n".join(retrieve(question))
    prompt = f"Use the following examples to help solve the problem.\n\n{ctx}\n\nProblem: {question}"
    resp = router.chat.completions.create(
        model=model,
        messages=[{"role": "user", "content": prompt}],
    )
    return resp.choices[0].message.content
```

## Data Format

**Query files** (`.jsonl`): each line has `id`, `problem`, `answer`, `query`.

**Retrieved results files** (`.jsonl`): same fields plus `ctxs` — a list of top-3 passages, each with `id`, `title`, `text`, `score`.

## T3 Transformation Code

See [`data_transform/README.md`](data_transform/README.md) for how to apply T3 transformations to your own thinking traces.

```bash
python data_transform/run_all_prompts_gpt.py \
    --input  your_thinking_traces.jsonl \
    --outdir outputs/ \
    --prompts t3_struct t3_reflect t3_semantic \
    --model  gpt-4o-mini \
    --concurrency 50
```
