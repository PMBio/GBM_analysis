#!/bin/bash
#BSUB -J tf_preprocessing
#BSUB -o logs/preprocessing.out
#BSUB -e logs/preprocessing.err
#BSUB -R "rusage[mem=200GB]"
#BSUB -q long
#BSUB -W 12:00

# Step 1: build joint AnnData + run Harmony.
# Harmony is the memory-heaviest step in the pipeline (~150 GB peak).

set -euo pipefail
source ~/.bashrc
conda activate tf_screen

mkdir -p logs
cd "$(dirname "$0")/.."

python preprocessing/build_joint_anndata.py
python preprocessing/run_harmony.py
python preprocessing/qc_figures.py
