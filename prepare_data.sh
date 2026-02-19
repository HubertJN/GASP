#!/bin/bash
#SBATCH --account=su007-gpulowpri
#SBATCH --partition=gpulowpri
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --mem-per-cpu=3850
#SBATCH --gres=gpu:lovelace_l40:1
#SBATCH --time=24:00:00
#SBATCH --job-name=training_set
#SBATCH --output=logs/out_%A_%a.txt
#SBATCH --error=logs/err_%A_%a.txt
#SBATCH --array=0-6%30  # <-- set 0-(NPAIRS-1) to match rows in h_beta.csv (excluding header)

set -euo pipefail

module purge
module load GCC/13.2.0 OpenMPI/4.1.6 CUDA Python HDF5

cd /home/p/phunsc/GPU_2DIsing
source .venv/bin/activate

PARAM_FILE="h_beta.csv"   # header: h,beta

PAIR_INDEX=${SLURM_ARRAY_TASK_ID}

# Read (PAIR_INDEX+2)-th line (skip header on line 1)
LINE=$(sed -n "$((PAIR_INDEX+2))p" "$PARAM_FILE")
IFS=',' read -r H BETA <<< "$LINE"

if [[ -z "${H:-}" || -z "${BETA:-}" ]]; then
  echo "Failed to read h,beta for PAIR_INDEX=$PAIR_INDEX from $PARAM_FILE (got: '$LINE')" >&2
  exit 1
fi

echo "[Task $SLURM_ARRAY_TASK_ID] (h=$H, beta=$BETA) on $(hostname) at $(date)"

# 1) Generate trajectories
python -m tools.generate_trajectories --h "$H" --beta "$BETA"

# 2) Analyse trajectories -> dn (assumes last line printed is numeric dn)
DN=$(python -m tools.analyse_trajectories --h "$H" --beta "$BETA" | tail -n 1)

num_re='^-?[0-9]+([.][0-9]+)?([eE]-?[0-9]+)?$'
if ! [[ "$DN" =~ $num_re ]]; then
  echo "Failed to parse dn from analyse_trajectories (got: '$DN')" >&2
  exit 1
fi

echo "[Task $SLURM_ARRAY_TASK_ID] dn=$DN"

# 3) Sigmoid committor -> target LCS05 (assumes last line printed is numeric target)
# If your sigmoid script uses --lower instead of --dn, change the flag here.
TARGET=$(python -m tools.sigmoid_committor --h "$H" --beta "$BETA" --dn "$DN" | tail -n 1)

if ! [[ "$TARGET" =~ $num_re ]]; then
  echo "Failed to parse target from sigmoid_committor (got: '$TARGET')" >&2
  exit 1
fi

echo "[Task $SLURM_ARRAY_TASK_ID] target=$TARGET"

# 4) Create training set
# If create_training_set expects --lower instead of --dn, change the flag here.
python -m tools.create_training_set --h "$H" --beta "$BETA" --target "$TARGET"

echo "[Task $SLURM_ARRAY_TASK_ID] Finished at $(date)"
