#!/bin/bash
#SBATCH --job-name=si_mace
#SBATCH --partition=gpu          # change to your cluster's GPU partition
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=04:00:00
#SBATCH --output=slurm-%j.out

# Runs training, evaluation, and MD validation back to back.
# Submit from this folder after step 1:  sbatch slurm_gpu.sh

module purge
# module load cuda/12.1          # if your cluster needs it
source ~/miniconda3/etc/profile.d/conda.sh
conda activate mlip

set -euo pipefail
nvidia-smi

bash 03_train.sh
python 02_evaluate.py data/test.xyz --models mp:small models/si_mace_fast.model
python 04_md_validate.py --models models/si_mace_fast.model mp:small \
                         --ref-xyz data/train.xyz
