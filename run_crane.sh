#!/bin/bash
# ============================================================
# CRANE — single-run SLURM submission script
# Configure the SBATCH directives below for your HPC cluster.
# ============================================================
#SBATCH -o crane.%j.out
#SBATCH -e crane.%j.err
#SBATCH -J crane
#SBATCH -A <your-account>          # <-- configure
#SBATCH -p <your-gpu-partition>    # <-- configure
#SBATCH -q <your-qos>              # <-- configure
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:1
#SBATCH --chdir=<your-project-dir> # <-- configure

# ============================================================
# Environment — adjust to your cluster setup
# ============================================================
module load anaconda3/2022       # <-- configure
conda activate data_analysis     # <-- configure

# Optional: offline mode (set if no internet on compute nodes)
# export HF_HUB_OFFLINE=1
# export TRANSFORMERS_OFFLINE=1
# export HF_HOME="$HOME/.cache/huggingface"

# NLTK data path (set if using custom location)
# export NLTK_DATA=<your-nltk-data-dir>

python run.py
