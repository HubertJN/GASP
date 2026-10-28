#!/usr/bin/env bash
set -euo pipefail

PARAM_FILE="h_beta_dn.csv"
TASKS_PER_PAIR=10

# pass task id as first argument, default 0
TASK_ID="${1:-0}"

PAIR_INDEX=$(( TASK_ID / TASKS_PER_PAIR ))
LOCAL_TASK=$(( TASK_ID % TASKS_PER_PAIR ))

LINE=$(sed -n "$((PAIR_INDEX+2))p" "$PARAM_FILE")
IFS=',' read -r H BETA DN <<< "$LINE"

if [[ -z "${H:-}" || -z "${BETA:-}" || -z "${DN:-}" ]]; then
  echo "Failed to read params for PAIR_INDEX=$PAIR_INDEX from $PARAM_FILE (got: '$LINE')" >&2
  exit 1
fi

echo "[Task $TASK_ID] Pair $PAIR_INDEX local $LOCAL_TASK (h=$H, beta=$BETA, dn=$DN) on $(hostname) at $(date)"

python -m tools.slurm_committors --h "$H" --beta "$BETA" --dn "$DN"

echo "[Task $TASK_ID] Finished at $(date)"
