#!/bin/bash
# ============================================================
# CRANE — 4-seed ensemble training SLURM array script
# Submit: sbatch run_ensemble.sh
# Configure the SBATCH directives below for your HPC cluster.
# ============================================================
#SBATCH -o crane_ensemble.%A_%a.out
#SBATCH -e crane_ensemble.%A_%a.err
#SBATCH -J crane-ens
#SBATCH -A <your-account>          # <-- configure
#SBATCH -p <your-gpu-partition>    # <-- configure
#SBATCH -q <your-qos>              # <-- configure
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:1
#SBATCH --array=0-3
#SBATCH --chdir=<your-project-dir> # <-- configure

# ============================================================
# Environment — adjust to your cluster setup
# ============================================================
module load anaconda3/2022       # <-- configure
conda activate data_analysis     # <-- configure

# export NLTK_DATA=<your-nltk-data-dir>

SEEDS=(42 123 456 789)
SEED=${SEEDS[$SLURM_ARRAY_TASK_ID]}

SAVE_DIR="./saved_ensemble/seed${SEED}"
mkdir -p "$SAVE_DIR"

echo "Training ensemble member with seed=$SEED → $SAVE_DIR"

python run.py \
    --seed $SEED \
    --batch_size 8 \
    --lr 5e-6 \
    --fusion_method v2 \
    --dataset mosi \
    --model_save_path "$SAVE_DIR"

echo "Ensemble member seed=$SEED done"
