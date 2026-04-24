#!/bin/bash
# Simple wrapper: pick a benchmark, a retrieval file (or --no-rag), and run eval.
#
# Usage:
#   bash run_single_eval.sh --model gpt5 --bench aime --retrieval /path/to/file.jsonl
#   bash run_single_eval.sh --model gpt5 --bench gpqa --no-rag
#   bash run_single_eval.sh --model gemini --bench lcb  --retrieval /path/to/file.jsonl
#
# Required env vars (for API keys):
#   OPENAI_API_KEY      (for gpt5)
#   OPENROUTER_API_KEY  (for oss120b)
#   GEMINI_API_KEY      (for gemini)

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
LM_EVAL_DIR="${SCRIPT_DIR}/../.."       # repo root / eval/
# If running from inside the repo, lm-evaluation-harness lives alongside tasks/
LM_EVAL_HARNESS="${SCRIPT_DIR}/lm-evaluation-harness"

# ── Parse args ────────────────────────────────────────────────────────────────
MODEL=""
BENCH=""
RETRIEVAL_FILE=""
NO_RAG=0
OUTPUT_DIR=""
SEED=2025
TEMPERATURE=0.6
TOP_K=3

usage() {
  echo "Usage: $0 --model gpt5|gemini|oss120b --bench aime|gpqa|lcb [--retrieval FILE | --no-rag] [--output DIR]"
  exit 1
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --model)       MODEL="$2";          shift 2 ;;
    --bench)       BENCH="$2";          shift 2 ;;
    --retrieval)   RETRIEVAL_FILE="$2"; shift 2 ;;
    --no-rag)      NO_RAG=1;            shift   ;;
    --output)      OUTPUT_DIR="$2";     shift 2 ;;
    --seed)        SEED="$2";           shift 2 ;;
    *) usage ;;
  esac
done

[[ -z "$MODEL" || -z "$BENCH" ]] && usage
[[ $NO_RAG -eq 0 && -z "$RETRIEVAL_FILE" ]] && { echo "ERROR: provide --retrieval FILE or --no-rag"; exit 1; }
[[ $NO_RAG -eq 0 && ! -f "$RETRIEVAL_FILE" ]] && { echo "ERROR: retrieval file not found: $RETRIEVAL_FILE"; exit 1; }

# ── Model config ──────────────────────────────────────────────────────────────
HF_CACHE="${HF_DATASETS_CACHE:-$HOME/.cache/huggingface/datasets}"

case "$MODEL" in
  gpt5)
    MODEL_NAME="gpt-5-2025-08-07"
    OAI_KEY="${OPENAI_API_KEY:?Set OPENAI_API_KEY}"
    MODEL_ARGS="model=${MODEL_NAME},base_url=https://api.openai.com/v1/chat/completions,num_concurrent=32,max_length=128000,tokenizer_backend=None"
    RUN_ENV="PYTHONNOUSERSITE=1 HF_DATASETS_CACHE=$HF_CACHE OPENAI_API_KEY=$OAI_KEY PROCESSOR=gpt-4o-mini"
    ;;
  gemini)
    MODEL_NAME="gemini-2.5-flash"
    GEMINI_KEY="${GEMINI_API_KEY:?Set GEMINI_API_KEY}"
    OAI_KEY="${OPENAI_API_KEY:-dummy}"
    PROXY_PORT=8081
    if ! curl -sf "http://127.0.0.1:${PROXY_PORT}/health" > /dev/null 2>&1; then
      echo "Starting Gemini proxy on port $PROXY_PORT…"
      GOOGLE_GENAI_USE_VERTEXAI=true GEMINI_API_KEY="$GEMINI_KEY" \
        python "${SCRIPT_DIR}/gemini_proxy_server.py" --port "$PROXY_PORT" &
      sleep 5
    fi
    MODEL_ARGS="model=${MODEL_NAME},base_url=http://127.0.0.1:${PROXY_PORT}/v1/chat/completions,num_concurrent=8,max_length=1048576,tokenizer_backend=None"
    RUN_ENV="PYTHONNOUSERSITE=1 HF_DATASETS_CACHE=$HF_CACHE LMEVAL_API_KEY=$GEMINI_KEY OPENAI_API_KEY=$OAI_KEY GOOGLE_GENAI_USE_VERTEXAI=true GEMINI_API_KEY=$GEMINI_KEY PROCESSOR=gpt-4o-mini"
    ;;
  oss120b)
    MODEL_NAME="gpt-oss-120b-deepinfra-bf16"
    OR_KEY="${OPENROUTER_API_KEY:?Set OPENROUTER_API_KEY}"
    OAI_KEY="${OPENAI_API_KEY:-dummy}"
    MODEL_ARGS="model=openai/gpt-oss-120b:deepinfra-bf16,base_url=https://openrouter.ai/api/v1/chat/completions,num_concurrent=16,max_length=128000,tokenizer_backend=None"
    RUN_ENV="PYTHONNOUSERSITE=1 HF_DATASETS_CACHE=$HF_CACHE LMEVAL_API_KEY=$OR_KEY OPENAI_API_KEY=$OAI_KEY PROCESSOR=gpt-4o-mini"
    ;;
  *) echo "Unknown model: $MODEL"; usage ;;
esac

# ── Benchmark config ──────────────────────────────────────────────────────────
case "$BENCH" in
  aime)
    if [[ $NO_RAG -eq 1 ]]; then
      TASKS="aime25_nofigures_agg8 aime26_nofigures_agg8"
      INCLUDE="${SCRIPT_DIR}/../tasks/aime"
      GEN_KWARGS="max_gen_toks=65536,temperature=${TEMPERATURE}"
    else
      TASKS="aime25_nofigures_retrieval_agg8 aime26_nofigures_retrieval_agg8"
      INCLUDE="${SCRIPT_DIR}/../tasks/aime"
      GEN_KWARGS="max_gen_toks=16384,temperature=${TEMPERATURE}"
    fi
    ;;
  gpqa)
    if [[ $NO_RAG -eq 1 ]]; then
      TASKS="gpqa_diamond_openai_agg4"
      INCLUDE="${SCRIPT_DIR}/../tasks/gpqa"
      GEN_KWARGS="max_gen_toks=65536,temperature=${TEMPERATURE}"
    else
      TASKS="gpqa_diamond_openai_retrieval_agg4"
      INCLUDE="${SCRIPT_DIR}/../tasks/gpqa"
      GEN_KWARGS="max_gen_toks=16384,temperature=${TEMPERATURE}"
    fi
    ;;
  lcb)
    if [[ $NO_RAG -eq 1 ]]; then
      TASKS="lcb_v4mv2_agg4"
      INCLUDE="${SCRIPT_DIR}/../tasks/lcb"
      GEN_KWARGS="max_gen_toks=65536,temperature=${TEMPERATURE}"
    else
      TASKS="lcb_v4mv2_retrieval_agg4"
      INCLUDE="${SCRIPT_DIR}/../tasks/lcb"
      GEN_KWARGS="max_gen_toks=16384,temperature=${TEMPERATURE}"
    fi
    ;;
  *) echo "Unknown bench: $BENCH"; usage ;;
esac

# ── Output path ───────────────────────────────────────────────────────────────
if [[ -z "$OUTPUT_DIR" ]]; then
  if [[ $NO_RAG -eq 1 ]]; then
    LABEL="no_rag"
  else
    LABEL="$(basename "$RETRIEVAL_FILE" .jsonl)_top${TOP_K}"
  fi
  OUTPUT_DIR="${BENCH}_eval/${MODEL_NAME}/temp${TEMPERATURE}/${LABEL}"
fi

# ── Run ───────────────────────────────────────────────────────────────────────
echo "============================================================"
echo "  Model  : $MODEL_NAME"
echo "  Bench  : $BENCH  (tasks: $TASKS)"
if [[ $NO_RAG -eq 1 ]]; then
  echo "  Mode   : NO-RAG baseline"
else
  echo "  Mode   : RAG  (top-$TOP_K)"
  echo "  File   : $RETRIEVAL_FILE"
fi
echo "  Output : $OUTPUT_DIR"
echo "  Seed   : $SEED"
echo "============================================================"

RAG_ENV=""
if [[ $NO_RAG -eq 0 ]]; then
  RAG_ENV="RETRIEVAL_FILE_PATH=$RETRIEVAL_FILE RETRIEVAL_TOP_K=$TOP_K RETRIEVAL_OFFSET=0"
fi

cd "$LM_EVAL_HARNESS" || { echo "ERROR: lm-evaluation-harness dir not found at $LM_EVAL_HARNESS"; exit 1; }

env $RUN_ENV $RAG_ENV \
  python -m lm_eval \
    --model openai-chat-completions \
    --model_args "$MODEL_ARGS" \
    --tasks $TASKS \
    --include_path "$INCLUDE" \
    --batch_size 1 \
    --apply_chat_template \
    --output_path "$OUTPUT_DIR" \
    --log_samples \
    --seed "$SEED" \
    --gen_kwargs "$GEN_KWARGS"

echo ""
echo "Done → $OUTPUT_DIR"
