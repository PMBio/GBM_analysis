#!/bin/bash
#BSUB -J tf_state_testing
#BSUB -o logs/state_testing.out
#BSUB -e logs/state_testing.err
#BSUB -R "rusage[mem=150GB]"
#BSUB -q long
#BSUB -W 36:00

# Step 4 of the TF screen pipeline.
# 16-strategy grid of state-, topic-, and proportion-tests.
# Heaviest of the analysis jobs (~24 h, ~150 GB peak).

set -euo pipefail

# Adapt for your cluster:
source ~/.bashrc
conda activate tf_screen

mkdir -p logs
cd "$(dirname "$0")/.."   # cd into tf_screen/

python state_testing/state_testing.py
python state_testing/topic_testing.py
python state_testing/proportion_testing.py
