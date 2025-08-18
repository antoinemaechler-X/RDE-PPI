#!/bin/bash
#SBATCH --job-name=wt_test_rde
#SBATCH --output=wt_rde%j.out
#SBATCH --error=wt_rde%j.err
#SBATCH --time=12:00:00
#SBATCH --mem=96G
#SBATCH --partition=gpu_test
#SBATCH --gres=gpu:1
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=antoinemaechler@fas.harvard.edu
#SBATCH --nice=10000

# Load your environment
source /n/sw/Mambaforge-23.11.0-0/etc/profile.d/conda.sh
conda activate RDE

# Use old numpy just for this job
export PYTHONPATH=/n/netscratch/shakhnovich_lab/Lab/amaechler/RDE-PPI/old_numpy:$PYTHONPATH

# Go to your project directory
cd /n/netscratch/shakhnovich_lab/Lab/amaechler/RDE-PPI

# Run training
python train_rde_network_skempi_wildtype_only.py \
    configs/train/rde_ddg_skempi_wildtype_only.yml \
    --num_cvfolds 10 \
    --threshold 45 \
    --device cuda \
    --num_workers 1 \
    --tag "wildtype_only_v1" 