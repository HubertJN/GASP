#!/bin/bash
set -euo pipefail

JOB1=$(sbatch --parsable training_set.sh)     # creates training set + writes h_beta_dn.csv
echo "Submitted training_set: $JOB1"

JOB2=$(sbatch --parsable --dependency=afterok:$JOB1 committors.sh)  # array 0-69%30
echo "Submitted committors: $JOB2"

JOB3=$(sbatch --parsable --dependency=afterok:$JOB2 merge_committors.sh)  # array 0-6
echo "Submitted merge: $JOB3"
