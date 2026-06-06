#!/bin/bash
# ============================================================
# CRANE — single-run SLURM submission script
# ============================================================
# Usage:
#   1. Fill in the <...> placeholders below with your cluster settings
#   2. Submit: sbatch run_crane.sh
# ============================================================
#SBATCH -o crane.%j.out
#SBATCH -e crane.%j.err
#SBATCH -J crane
#SBATCH -A <your-account>
#SBATCH -p <your-gpu-partition>
#SBATCH -q <your-qos>
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:1
#SBATCH --chdir=<your-project-dir>

# ============================================================
# Environment
# ============================================================
module load anaconda3/2022
conda activate data_analysis

# Optional: offline mode for compute nodes without internet
# export HF_HUB_OFFLINE=1
# export TRANSFORMERS_OFFLINE=1
# export HF_HOME="$HOME/.cache/huggingface"

# Optional: custom NLTK data path
# export NLTK_DATA=<your-nltk-data-dir>

python run.py
