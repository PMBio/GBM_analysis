#!/bin/bash
#BSUB -J tf_classifiers
#BSUB -o logs/classifiers.out
#BSUB -e logs/classifiers.err
#BSUB -R "rusage[mem=64GB]"
#BSUB -q long
#BSUB -W 6:00

# Step 2: train state + topic classifiers.

set -euo pipefail
source ~/.bashrc
conda activate tf_screen

mkdir -p logs
cd "$(dirname "$0")/.."

python classifiers/train_state_classifier.py
python classifiers/train_topic_classifier.py
