#!/bin/sh
#SBATCH --job-name=small_lm_train
#SBATCH --output=small_lm.log
#SBATCH --cpus-per-task=2
#SBATCH --mem=16G
#SBATCH --partition=gpu
#SBATCH --gres=gpu:4


module purge
module load CUDA/9.0.176-GCC-6.4.0-2.28
module load cuDNN/7.1.4.18-fosscuda-2018b

python3 random_feature_embedder.py
