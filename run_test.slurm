#!/bin/bash
#SBATCH --job-name=test_rde
#SBATCH --output=rde%j.out
#SBATCH --error=rde%j.err
#SBATCH --time=12:00:00
#SBATCH --mem=16G
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

# Run your script
#python test_rde_network_skempi.py --ckpt ./trained_models/DDG_RDE_Network_30k.pt
#python test_rde_linear_skempi.py --ckpt ./trained_models/RDE.pt
# Training
python train_rde_network_skempi_grouped.py configs/train/rde_ddg_skempi_grouped.yml --num_workers 1

# Testing 
#python test_rde_network_skempi_grouped.py configs/train/rde_ddg_skempi_grouped.yml --checkpoint <path_to_model>

#training only regression
#python -m rde.linear.entropy

#python -m rde.linear.calibrate_grouped RDE_skempi_entropy.pkl --grouped_csv_path ./data/complex_sequences_grouped_99.csv --num_folds 10

