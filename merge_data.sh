#!/bin/bash
#SBATCH --account=su007-gpulowpri
#SBATCH --partition=gpulowpri
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --mem-per-cpu=3850
#SBATCH --gres=gpu:lovelace_l40:1
#SBATCH --time=02:00:00
#SBATCH --job-name=merge_committors
#SBATCH --output=logs/merge_out_%A.txt
#SBATCH --error=logs/merge_err_%A.txt

set -euo pipefail

module purge
module load GCC/13.2.0 OpenMPI/4.1.6 Python HDF5

cd /home/p/phunsc/GPU_2DIsing
source .venv/bin/activate

PARAM_FILE="h_beta_dn.csv"
PAIR_INDEX=${SLURM_ARRAY_TASK_ID}

LINE=$(sed -n "$((PAIR_INDEX+2))p" "$PARAM_FILE")  # skip header
IFS=',' read -r H BETA DN <<< "$LINE"

if [[ -z "${H:-}" || -z "${BETA:-}" ]]; then
  echo "Failed to read h,beta for PAIR_INDEX=$PAIR_INDEX from $PARAM_FILE (got: '$LINE')" >&2
  exit 1
fi

echo "[Merge $PAIR_INDEX] (h=$H, beta=$BETA) on $(hostname) at $(date)"

python -m tools.merge_files --h "$H" --beta "$BETA"

echo "[Merge $PAIR_INDEX] Finished at $(date)"
