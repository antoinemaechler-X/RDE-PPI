#!/bin/bash
#SBATCH --job-name=separate_structures_rde
#SBATCH --output=separate_structures_rde%j.out
#SBATCH --error=separate_structures_rde%j.err
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

# Run separate structures training
# Default: 60% threshold, 10 folds
python train_rde_network_skempi_separate_structures.py configs/train/rde_ddg_skempi_separate_structures.yml --num_workers 1

# Alternative thresholds (uncomment to use):
# python train_rde_network_skempi_separate_structures.py configs/train/rde_ddg_skempi_separate_structures.yml --threshold 40 --num_workers 1
# python train_rde_network_skempi_separate_structures.py configs/train/rde_ddg_skempi_separate_structures.yml --threshold 80 --num_workers 1
# python train_rde_network_skempi_separate_structures.py configs/train/rde_ddg_skempi_separate_structures.yml --threshold 95 --num_workers 1
# python train_rde_network_skempi_separate_structures.py configs/train/rde_ddg_skempi_separate_structures.yml --threshold 99 --num_workers 1
# python train_rde_network_skempi_separate_structures.py configs/train/rde_ddg_skempi_separate_structures.yml --threshold 100 --num_workers 1

# Testing command (uncomment after training completes):
# python test_rde_network_skempi_separate_structures.py configs/train/rde_ddg_skempi_separate_structures.yml --checkpoint logs_skempi_separate_structures/checkpoints/30000.pt --num_workers 1 