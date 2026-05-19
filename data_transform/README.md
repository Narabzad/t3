# T3 Transformation Code

Given a JSONL file of thinking traces, apply any of the three T3 transformations using `transform.py`.

## Input format

Each line of the input JSONL must include a non-empty thinking trace in one of these fields:
- `trace` — the published raw-trace field name.
- `text` — a legacy field name, still accepted for backward compatibility.

Rows may also include `question`, the original problem. It is recommended and preserved in output when present.

## Usage

```bash
pip install openai tqdm

python transform.py \
    --input  your_thinking_traces.jsonl \
    --outdir outputs/ \
    --prompts t3_struct t3_reflect t3_semantic \
    --model  gpt-4o-mini \
    --concurrency 50
```

The script is **resume-safe**: re-running picks up where it left off.

## Transformations

| Prompt | Paper name | Description |
|--------|------------|-------------|
| `t3_struct` | Structural Normalization | Step-by-step procedural scaffold; one trace may produce multiple passages |
| `t3_reflect` | Reflection | Contrastive form highlighting common mistakes and how to avoid them |
| `t3_semantic` | Semantic Distillation | Core reasoning idea; multi-pass compression of the trace |

Prompt templates are in `prompts/`.

## Output format

Each output file (`outputs/t3_struct.jsonl`, etc.) has one line per passage:
- `_idx` — index of the source trace
- `_passage_idx` — passage index within that trace (for t3_struct, multiple passages per trace are possible)
- `passage` — the transformed text
