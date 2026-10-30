#!/usr/bin/env bash
# Generate MSM trajectory files for every condition with a training file in data/.
# Safe to rerun: finished conditions are skipped run by run inside the Python tool.
set -euo pipefail

for f in data/gridstates_training_*.hdf5; do
  tag=${f#data/gridstates_training_}; tag=${tag%.hdf5}
  beta=${tag%_*}; h=${tag#*_}
  python -m tools.generate_msm_trajectories --beta "$beta" --h "$h" "$@"
done
