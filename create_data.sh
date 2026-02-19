#!/bin/bash
#SBATCH --account=su007-gpulowpri
#SBATCH --partition=gpulowpri
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --mem-per-cpu=3850
#SBATCH --gres=gpu:lovelace_l40:1
#SBATCH --time=24:00:00
#SBATCH --job-name=committors
#SBATCH --output=logs/out_%A_%a.txt
#SBATCH --error=logs/err_%A_%a.txt
#SBATCH --array=0-69%30

set -euo pipefail

module purge
module load GCC/13.2.0 OpenMPI/4.1.6 CUDA Python HDF5

cd /home/p/phunsc/GPU_2DIsing
source .venv/bin/activate

PARAM_FILE="h_beta_dn.csv"
TASKS_PER_PAIR=10

PAIR_INDEX=$(( SLURM_ARRAY_TASK_ID / TASKS_PER_PAIR ))
LOCAL_TASK=$(( SLURM_ARRAY_TASK_ID % TASKS_PER_PAIR ))

# header is line 1, so data starts at line 2
LINE=$(sed -n "$((PAIR_INDEX+2))p" "$PARAM_FILE")
IFS=',' read -r H BETA DN <<< "$LINE"

if [[ -z "${H:-}" || -z "${BETA:-}" || -z "${DN:-}" ]]; then
  echo "Failed to read params for PAIR_INDEX=$PAIR_INDEX from $PARAM_FILE (got: '$LINE')" >&2
  exit 1
fi

echo "[Task $SLURM_ARRAY_TASK_ID] Pair $PAIR_INDEX local $LOCAL_TASK (h=$H, beta=$BETA, dn=$DN) on $(hostname) at $(date)"

python -m tools.slurm_committors --h "$H" --beta "$BETA" --dn "$DN"

echo "[Task $SLURM_ARRAY_TASK_ID] Finished at $(date)"
