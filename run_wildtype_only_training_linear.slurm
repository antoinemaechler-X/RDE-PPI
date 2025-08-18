#!/bin/bash
#SBATCH --job-name=wt_test_rde_linear
#SBATCH --output=wt_rde_linear%j.out
#SBATCH --error=wt_rde_linear%j.err
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
#python -m rde.linear.entropy --ckpt trained_models/RDE.pt

python -m rde.linear.calibrate_wildtype_only RDE_skempi_entropy.pkl --folds_dir ./cross_validation_folds_final --threshold 99 --num_folds 10