#!/bin/bash
#SBATCH -J fepred
#SBATCH -N 1
#SBATCH -p savio3_gpu
#SBATCH -A co_armada2
#SBATCH -q armada2_gpu3_normal
#SBATCH --gres=gpu:A40:1
#SBATCH --time=72:00:00
#SBATCH --cpus-per-task 8
#SBATCH --mem=200G

echo "Job started on $(hostname) at $(date)"
module load anaconda3/2024.02-1-11.4     # load conda module (check version with `module avail anaconda`)

# Activate your conda environment
eval "$(conda shell.bash hook)"
conda activate ucbbind
echo "Conda environment activated"

# Run your python script
# python -m residual_predictor.revised_scripts.rr_feature_extract
python FEpred.py
echo "Job finished at $(date)"
