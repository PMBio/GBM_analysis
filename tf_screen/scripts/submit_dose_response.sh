#!/bin/bash
#BSUB -J tf_dose_response
#BSUB -o logs/dose_response.out
#BSUB -e logs/dose_response.err
#BSUB -R "rusage[mem=64GB]"
#BSUB -q long
#BSUB -W 12:00

# Step 5: build metacells + run correlation analysis.

set -euo pipefail
source ~/.bashrc
conda activate tf_screen

mkdir -p logs
cd "$(dirname "$0")/.."

python dose_response/build_metacells.py
python dose_response/correlation_analysis.py
