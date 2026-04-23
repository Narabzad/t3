#!/bin/bash
# Generic paper eval runner with parallel pool + rate-limit detection + per-run verification
# MODEL: gpt5 | gemini | oss120b
# BENCH: aime | lcb | gpqa
# PART:  1 (first half of files) | 2 (second half)
# POOL:  initial parallelism (default 10)

MODEL="${MODEL:-gpt5}"
BENCH="${BENCH:-aime}"
PART="${PART:-1}"
POOL="${POOL:-10}"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
LM_EVAL_DIR="${SCRIPT_DIR}/lm-evaluation-harness"
cd "$LM_EVAL_DIR" || exit 1
REPO_ROOT="$(cd "$SCRIPT_DIR/../../.." && pwd)"
RDIR="${REPO_ROOT}/retriever_results/s1_data_transform"

RETRIEVAL_TOP_K=3
TEMPERATURE=0.6
SEED=2025

OAI_KEY="${OPENAI_API_KEY:?Set OPENAI_API_KEY}"
OR_KEY="${OPENROUTER_API_KEY:?Set OPENROUTER_API_KEY}"
GEMINI_KEY="${GEMINI_API_KEY:?Set GEMINI_API_KEY}"

# ── Model config ──────────────────────────────────────────────────────────────
HF_CACHE="${HF_DATASETS_CACHE:-$HOME/.cache/huggingface/datasets}"
mkdir -p "$HF_CACHE"

if [[ "$MODEL" == "gpt5" ]]; then
  MODEL_NAME="gpt-5-2025-08-07"
  NUM_CONCURRENT=64
  MODEL_ARGS="model=${MODEL_NAME},base_url=https://api.openai.com/v1/chat/completions,num_concurrent=${NUM_CONCURRENT},max_length=128000,tokenizer_backend=None"
  CACHE_DIR="lm_eval_cache/${MODEL_NAME}_paper_${BENCH}_p${PART}"
  RUN_ENV="PYTHONNOUSERSITE=1 HF_DATASETS_CACHE=$HF_CACHE OPENAI_API_KEY=$OAI_KEY PROCESSOR=gpt-4o-mini"
elif [[ "$MODEL" == "gemini" ]]; then
  MODEL_NAME="gemini-2.5-flash"
  NUM_CONCURRENT=8
  PROXY_PORT=8081
  if ! curl -sf "http://127.0.0.1:${PROXY_PORT}/health" > /dev/null 2>&1; then
    GOOGLE_GENAI_USE_VERTEXAI=true GEMINI_API_KEY="$GEMINI_KEY" \
      python "${SCRIPT_DIR}/gemini_proxy_server.py" --port ${PROXY_PORT} &
    sleep 5
  fi
  MODEL_ARGS="model=${MODEL_NAME},base_url=http://127.0.0.1:${PROXY_PORT}/v1/chat/completions,num_concurrent=${NUM_CONCURRENT},max_length=1048576,tokenizer_backend=None"
  CACHE_DIR="lm_eval_cache/${MODEL_NAME}_paper_${BENCH}_p${PART}"
  RUN_ENV="PYTHONNOUSERSITE=1 HF_DATASETS_CACHE=$HF_CACHE LMEVAL_API_KEY=$GEMINI_KEY OPENAI_API_KEY=$OAI_KEY GOOGLE_GENAI_USE_VERTEXAI=true GEMINI_API_KEY=$GEMINI_KEY PROCESSOR=gpt-4o-mini"
elif [[ "$MODEL" == "oss120b" ]]; then
  MODEL_NAME="gpt-oss-120b-deepinfra-bf16"
  NUM_CONCURRENT=16
  MODEL_ARGS="model=openai/gpt-oss-120b:deepinfra-bf16,base_url=https://openrouter.ai/api/v1/chat/completions,num_concurrent=${NUM_CONCURRENT},max_length=128000,tokenizer_backend=None"
  CACHE_DIR="lm_eval_cache/${MODEL_NAME}_paper_${BENCH}_p${PART}"
  RUN_ENV="PYTHONNOUSERSITE=1 HF_DATASETS_CACHE=$HF_CACHE LMEVAL_API_KEY=$OR_KEY OPENAI_API_KEY=$OAI_KEY PROCESSOR=gpt-4o-mini"
else
  echo "Unknown MODEL=$MODEL"; exit 1
fi

# ── Shared state for pool management ─────────────────────────────────────────
POOL_SIZE=$POOL
declare -a PIDS=()
declare -A PID_OUT=()     # pid -> log file
declare -A PID_OUT_PATH=() # pid -> output_path
RATE_LIMIT_FILE="/tmp/rate_limit_${MODEL}_${BENCH}_p${PART}"
rm -f "$RATE_LIMIT_FILE"

# ── Helpers ───────────────────────────────────────────────────────────────────
check_results_exist() {
  local p="${LM_EVAL_DIR}/$1"
  [[ -d "$p" ]] && find "$p" -name "results_*.json" -type f 2>/dev/null | grep -q .
}

# Verify a completed result dir: check temp, max_gen_toks, empty rate
verify_result() {
  local OUT_PATH="$1"
  local FULL_PATH="${LM_EVAL_DIR}/${OUT_PATH}"
  [[ ! -d "$FULL_PATH" ]] && return 1

  local results_file; results_file=$(find "$FULL_PATH" -name "results_*.json" | head -1)
  [[ -z "$results_file" ]] && return 1

  # Check gen_kwargs
  local gen_kwargs; gen_kwargs=$(python3 -c "
import json
d=json.load(open('$results_file'))
print(d.get('config',{}).get('gen_kwargs',''))
" 2>/dev/null)

  local fail=0
  if [[ "$gen_kwargs" != *"0.6"* ]]; then
    echo "  ❌ VERIFY FAIL wrong temp in $OUT_PATH: $gen_kwargs"; fail=1
  fi
  if [[ "$gen_kwargs" != *"16384"* ]]; then
    echo "  ❌ VERIFY FAIL wrong max_gen_toks in $OUT_PATH: $gen_kwargs"; fail=1
  fi

  # Check empty response rate
  local samples_file; samples_file=$(find "$FULL_PATH" -name "samples_*.jsonl" | head -1)
  if [[ -n "$samples_file" ]]; then
    local empty_pct; empty_pct=$(python3 -c "
import json
total=0; empty=0
with open('$samples_file') as f:
  for line in f:
    try:
      d=json.loads(line)
      for r in d.get('resps',[[]])[0]:
        total+=1
        if not str(r).strip(): empty+=1
    except: pass
print(f'{100*empty/total:.1f}' if total>0 else '0')
" 2>/dev/null)
    if (( $(echo "${empty_pct:-0} > 5" | bc -l 2>/dev/null) )); then
      echo "  ❌ VERIFY FAIL ${empty_pct}% empty in $OUT_PATH"
      fail=1
    elif (( $(echo "${empty_pct:-0} > 0" | bc -l 2>/dev/null) )); then
      echo "  ⚠  ${empty_pct}% empty in $OUT_PATH"
    fi
  fi

  [[ $fail -eq 0 ]] && echo "  ✓ OK: $OUT_PATH"
  return $fail
}

# Launch eval directly in current shell (NO command substitution — avoids PID orphan bug)
# Appends to global PIDS array
launch_eval() {
  local FPATH="$1" OUT="$2" TASKS="$3" INCLUDE="$4" OFFSET="${5:-0}"
  local RFN; RFN=$(basename "$FPATH" .jsonl | sed 's/_/-/g')
  local LOGFILE="/tmp/eval_${MODEL}_${BENCH}_p${PART}_${RFN}.log"

  env $RUN_ENV \
    RETRIEVAL_FILE_PATH="$FPATH" RETRIEVAL_TOP_K=$RETRIEVAL_TOP_K RETRIEVAL_OFFSET=$OFFSET \
    python -m lm_eval --model openai-chat-completions --model_args "$MODEL_ARGS" \
      --tasks $TASKS --include_path $INCLUDE --batch_size 1 --apply_chat_template \
      --output_path "$OUT" --log_samples --seed $SEED --use_cache "$CACHE_DIR" \
      --gen_kwargs "max_gen_toks=16384,temperature=${TEMPERATURE}" \
    > "$LOGFILE" 2>&1 &

  local pid=$!
  PIDS+=("$pid")
  PID_OUT[$pid]="$LOGFILE"
  PID_OUT_PATH[$pid]="$OUT"
}

# Detect rate limit from a log file
detect_rate_limit() {
  local logfile="$1"
  grep -qi "429\|rate.limit\|too many requests\|RateLimitError" "$logfile" 2>/dev/null
}

# Pool management: wait for a slot, detect rate limits
wait_for_slot() {
  while true; do
    local NEW_PIDS=()
    for pid in "${PIDS[@]}"; do
      if kill -0 "$pid" 2>/dev/null; then
        # Still running — check for rate limit
        if detect_rate_limit "${PID_OUT[$pid]:-/dev/null}"; then
          touch "$RATE_LIMIT_FILE"
        fi
        NEW_PIDS+=("$pid")
      else
        # Finished — verify result
        wait "$pid"
        local exit_code=$?
        local logfile="${PID_OUT[$pid]}"
        local out_path="${PID_OUT_PATH[$pid]}"

        # Check rate limit in completed job log
        if detect_rate_limit "$logfile"; then
          touch "$RATE_LIMIT_FILE"
        fi

        if [[ -f "$RATE_LIMIT_FILE" ]] && [[ $POOL_SIZE -gt 3 ]]; then
          if [[ $POOL_SIZE -ge 10 ]]; then POOL_SIZE=7
          elif [[ $POOL_SIZE -ge 7 ]]; then POOL_SIZE=5
          elif [[ $POOL_SIZE -ge 5 ]]; then POOL_SIZE=3
          fi
          echo "  ⚠ Rate limit detected — reducing pool to $POOL_SIZE"
          rm -f "$RATE_LIMIT_FILE"
        fi

        # Verify result (log only — never delete)
        if [[ -n "$out_path" ]] && check_results_exist "$out_path"; then
          verify_result "$out_path"
        fi

        unset PID_OUT[$pid]
        unset PID_OUT_PATH[$pid]
      fi
    done
    PIDS=("${NEW_PIDS[@]}")
    [[ ${#PIDS[@]} -lt $POOL_SIZE ]] && break
    sleep 3
  done
}

wait_all() {
  while [[ ${#PIDS[@]} -gt 0 ]]; do
    wait_for_slot
    sleep 3
  done
}

# Submit a job to the pool (skip if done)
submit() {
  local FPATH="$1" OUT="$2" TASKS="$3" INCLUDE="$4" OFFSET="${5:-0}"
  [[ ! -f "$FPATH" ]] && echo "  ⚠ Missing: $FPATH" && return
  check_results_exist "$OUT" && echo "  ✓ Already done: $(basename "$FPATH" .jsonl)" && return
  wait_for_slot
  local RFN; RFN=$(basename "$FPATH" .jsonl | sed 's/_/-/g')
  echo "→ Launching: $RFN"
  launch_eval "$FPATH" "$OUT" "$TASKS" "$INCLUDE" "$OFFSET"
}

# Skip files matching these patterns (user instruction)
should_skip() {
  local fname; fname=$(basename "$1")
  [[ "$fname" == *merged* ]] && return 0
  [[ "$fname" == *gepa* ]] && return 0
  [[ "$fname" == *random* ]] && return 0
  [[ "$fname" == *qwen* ]] && return 0
  [[ "$fname" == *gemini3flash* ]] && return 0
  return 1
}

aime_submit() {
  local FPATH="$1"
  should_skip "$FPATH" && echo "  ⊘ Skipping: $(basename "$FPATH" .jsonl)" && return
  local RFN; RFN=$(basename "$FPATH" .jsonl | sed 's/_/-/g')
  for YEAR in 25 26; do
    local OUT="aime${YEAR}_eval/no_budget_forcing_retrieval_agg8_64k/${MODEL_NAME}/temp${TEMPERATURE}/${RFN}_top${RETRIEVAL_TOP_K}"
    submit "$FPATH" "$OUT" "aime${YEAR}_nofigures_retrieval_agg8" "lm_eval/tasks/aime" 0
  done
}

lcb_submit() {
  local FPATH="$1"
  should_skip "$FPATH" && echo "  ⊘ Skipping: $(basename "$FPATH" .jsonl)" && return
  local RFN; RFN=$(basename "$FPATH" .jsonl | sed 's/_/-/g')
  local OUT="lcb_eval/retrieval_agg4_64k/${MODEL_NAME}/temp${TEMPERATURE}/${RFN}_top${RETRIEVAL_TOP_K}"
  submit "$FPATH" "$OUT" "lcb_v4mv2_retrieval_agg4" "lm_eval/tasks/lcb" 0
}

gpqa_submit() {
  local FPATH="$1"
  should_skip "$FPATH" && echo "  ⊘ Skipping: $(basename "$FPATH" .jsonl)" && return
  local RFN; RFN=$(basename "$FPATH" .jsonl | sed 's/_/-/g')
  local OUT="gpqa_diamond_eval/no_budget_forcing_retrieval_agg4_64k/${MODEL_NAME}/temp${TEMPERATURE}/${RFN}_top${RETRIEVAL_TOP_K}"
  submit "$FPATH" "$OUT" "gpqa_diamond_openai_retrieval_agg4" "lm_eval/tasks/gpqa/openai" 0
}

# Split files: part 1 = first half, part 2 = second half
get_part_files() {
  local -n _OUT=$1
  local DIR="$2" GLOB="${3:-*.jsonl}"
  mapfile -t ALL < <(ls "${DIR}"/${GLOB} 2>/dev/null | sort)
  local TOTAL=${#ALL[@]} HALF=$(( (${#ALL[@]} + 1) / 2 ))
  if [[ "$PART" == "1" ]]; then _OUT=("${ALL[@]:0:$HALF}")
  else _OUT=("${ALL[@]:$HALF}"); fi
}

echo "=== MODEL=${MODEL} BENCH=${BENCH} PART=${PART} pool=${POOL_SIZE} ==="

# ── AIME ─────────────────────────────────────────────────────────────────────
if [[ "$BENCH" == "aime" ]]; then
  declare -a FILES
  get_part_files FILES "${RDIR}/aime_2025_2026"
  for f in "${FILES[@]}"; do aime_submit "$f"; done
fi

# ── LCB v4 only ──────────────────────────────────────────────────────────────
if [[ "$BENCH" == "lcb" ]]; then
  declare -a FILES
  get_part_files FILES "${RDIR}/lcb" "*lcb_v4*.jsonl"
  for f in "${FILES[@]}"; do lcb_submit "$f"; done
fi

# ── GPQA ─────────────────────────────────────────────────────────────────────
if [[ "$BENCH" == "gpqa" ]]; then
  declare -a FILES
  get_part_files FILES "${RDIR}/gpqa"
  for f in "${FILES[@]}"; do gpqa_submit "$f"; done
fi

wait_all
echo "=== Done: MODEL=${MODEL} BENCH=${BENCH} PART=${PART} ==="
