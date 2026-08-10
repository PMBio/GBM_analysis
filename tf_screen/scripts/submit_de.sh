#!/bin/bash
#BSUB -J tf_wilcoxon_de
#BSUB -o logs/wilcoxon_de.out
#BSUB -e logs/wilcoxon_de.err
#BSUB -R "rusage[mem=64GB]"
#BSUB -q long
#BSUB -W 24:00

# Step 6 of the TF screen pipeline.
# Per-cell-line Wilcoxon DE; runs all four strategies by default.
#
# To run only the primary (Fig. 3) strategy:
#   python differential_expression/wilcoxon_de.py all_all
#
# Writes ~50 GB of per-(TF, line) CSVs across all four strategies;
# the combined all_DE_results.csv per strategy contains the same data.

set -euo pipefail

source ~/.bashrc
conda activate tf_screen

mkdir -p logs
cd "$(dirname "$0")/.."

python differential_expression/wilcoxon_de.py
