#!/bin/bash
# Launch 18 parallel eval sessions: 3 models × 3 benchmarks × 2 parts
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
GENERIC="${SCRIPT_DIR}/run_eval.sh"
chmod +x "$GENERIC"

for MODEL in gpt5 gemini oss120b; do
  for BENCH in aime lcb gpqa; do
    for PART in 1 2; do
      SESSION="${MODEL}_${BENCH}_p${PART}"
      # Kill if already running
      screen -S "$SESSION" -X quit 2>/dev/null
      sleep 0.3
      screen -dmS "$SESSION" bash -c "MODEL=$MODEL BENCH=$BENCH PART=$PART bash '$GENERIC' 2>&1 | tee '${SCRIPT_DIR}/${SESSION}.log'"
      echo "Started: $SESSION"
    done
  done
done

echo ""
echo "=== 18 sessions launched ==="
screen -ls | grep -E "gpt5|gemini|oss120b" | grep -E "aime|lcb|gpqa"
