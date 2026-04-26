#!/bin/bash
# Run T3-RAG evaluation via OpenRouter (or any OpenAI-compatible endpoint).
#
# Usage:
#   # With retrieval
#   bash eval/scripts/run_eval.sh \
#       --model qwen/qwq-32b \
#       --bench aime \
#       --retrieval data/retrieved_results/aime_2025_2026/t3_struct_e5base_full.jsonl
#
#   # No-RAG baseline
#   bash eval/scripts/run_eval.sh --model openai/gpt-4o --bench gpqa --no-rag
#
# Required:
#   OPENROUTER_API_KEY   — set via: export OPENROUTER_API_KEY=...
#
# Optional env overrides:
#   BASE_URL             — API endpoint (default: OpenRouter)
#   CONCURRENCY          — parallel requests (default: 8)
#   TEMPERATURE          — sampling temperature (default: 0.6)
#   TOP_K                — passages to prepend (default: 3)

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
TASKS_DIR="${REPO_ROOT}/eval/tasks"

MODEL=""
BENCH=""
RETRIEVAL_FILE=""
NO_RAG=0
OUTPUT_DIR=""
SEED=2025
TEMPERATURE="${TEMPERATURE:-0.6}"
TOP_K="${TOP_K:-3}"
BASE_URL="${BASE_URL:-https://openrouter.ai/api/v1/chat/completions}"
CONCURRENCY="${CONCURRENCY:-8}"

usage() {
  cat <<EOF
Usage: $0 --model MODEL --bench BENCH [--retrieval FILE | --no-rag] [OPTIONS]

Required:
  --model MODEL     OpenRouter model ID, e.g.:
                      qwen/qwq-32b
                      openai/gpt-4o
                      google/gemini-2.5-flash
                      openai/gpt-oss-120b:deepinfra-bf16
  --bench BENCH     aime | gpqa | lcb

Retrieval (pick one):
  --retrieval FILE  Path to a retrieved results JSONL
  --no-rag          Run without retrieval (baseline)

Options:
  --output DIR      Output directory (auto-generated if omitted)
  --seed N          Random seed (default: 2025)
  --base-url URL    API base URL (default: OpenRouter)
EOF
  exit 1
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --model)     MODEL="$2";          shift 2 ;;
    --bench)     BENCH="$2";          shift 2 ;;
    --retrieval) RETRIEVAL_FILE="$2"; shift 2 ;;
    --no-rag)    NO_RAG=1;            shift   ;;
    --output)    OUTPUT_DIR="$2";     shift 2 ;;
    --seed)      SEED="$2";           shift 2 ;;
    --base-url)  BASE_URL="$2";       shift 2 ;;
    -h|--help)   usage ;;
    *) echo "Unknown option: $1"; usage ;;
  esac
done

[[ -z "$MODEL" || -z "$BENCH" ]] && usage
[[ $NO_RAG -eq 0 && -z "$RETRIEVAL_FILE" ]] && { echo "ERROR: provide --retrieval FILE or --no-rag"; exit 1; }
[[ $NO_RAG -eq 0 && ! -f "$RETRIEVAL_FILE" ]] && { echo "ERROR: retrieval file not found: $RETRIEVAL_FILE"; exit 1; }

OR_KEY="${OPENROUTER_API_KEY:?Set OPENROUTER_API_KEY}"
MODEL_ARGS="model=${MODEL},base_url=${BASE_URL},num_concurrent=${CONCURRENCY},max_length=128000,tokenizer_backend=None"

# ── Benchmark → task YAML ─────────────────────────────────────────────────────
case "$BENCH" in
  aime)
    INCLUDE="${TASKS_DIR}/aime"
    if [[ $NO_RAG -eq 1 ]]; then
      TASKS="aime25_nofigures_agg8 aime26_nofigures_agg8"
    else
      TASKS="aime25_nofigures_retrieval_agg8 aime26_nofigures_retrieval_agg8"
    fi
    ;;
  gpqa)
    INCLUDE="${TASKS_DIR}/gpqa"
    if [[ $NO_RAG -eq 1 ]]; then
      TASKS="gpqa_diamond_openai_agg4"
    else
      TASKS="gpqa_diamond_openai_retrieval_agg4"
    fi
    ;;
  lcb)
    INCLUDE="${TASKS_DIR}/lcb"
    if [[ $NO_RAG -eq 1 ]]; then
      TASKS="lcb_v4mv2_agg4"
    else
      TASKS="lcb_v4mv2_retrieval_agg4"
    fi
    ;;
  *) echo "Unknown bench: $BENCH (choose: aime | gpqa | lcb)"; exit 1 ;;
esac

# ── Output path ───────────────────────────────────────────────────────────────
MODEL_SLUG="${MODEL//\//_}"
if [[ -z "$OUTPUT_DIR" ]]; then
  if [[ $NO_RAG -eq 1 ]]; then
    LABEL="no_rag"
  else
    LABEL="$(basename "$RETRIEVAL_FILE" .jsonl)_top${TOP_K}"
  fi
  OUTPUT_DIR="${REPO_ROOT}/results/${BENCH}/${MODEL_SLUG}/${LABEL}"
fi

# ── Run ───────────────────────────────────────────────────────────────────────
echo "============================================================"
echo "  Model  : ${MODEL}"
echo "  Bench  : ${BENCH}  (${TASKS})"
if [[ $NO_RAG -eq 1 ]]; then
  echo "  Mode   : no-RAG baseline"
else
  echo "  Mode   : RAG  (top-${TOP_K})"
  echo "  File   : ${RETRIEVAL_FILE}"
fi
echo "  Output : ${OUTPUT_DIR}"
echo "============================================================"

RAG_ENV=""
if [[ $NO_RAG -eq 0 ]]; then
  RAG_ENV="RETRIEVAL_FILE_PATH=${RETRIEVAL_FILE} RETRIEVAL_TOP_K=${TOP_K} RETRIEVAL_OFFSET=0"
fi

env LMEVAL_API_KEY="${OR_KEY}" OPENAI_API_KEY="${OR_KEY}" $RAG_ENV \
  python -m lm_eval \
    --model openai-chat-completions \
    --model_args "${MODEL_ARGS}" \
    --tasks ${TASKS} \
    --include_path "${INCLUDE}" \
    --batch_size 1 \
    --apply_chat_template \
    --output_path "${OUTPUT_DIR}" \
    --log_samples \
    --seed "${SEED}" \
    --gen_kwargs "max_gen_toks=16384,temperature=${TEMPERATURE}"

echo ""
echo "Done → ${OUTPUT_DIR}"
