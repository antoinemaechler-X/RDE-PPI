#!/bin/bash
#SBATCH --job-name=test_rde
#SBATCH --output=rde%j.out
#SBATCH --error=rde%j.err
#SBATCH --time=12:00:00
#SBATCH --mem=24G
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
#python train_rde_network_skempi_grouped.py configs/train/rde_ddg_skempi_grouped.yml --num_workers 1

#training only regression
#python -m rde.linear.entropy

#python -m rde.linear.calibrate_grouped RDE_skempi_entropy.pkl --grouped_csv_path ./data/complex_sequences_grouped_99.csv --num_folds 10

#python extract_rde_linear_per_residue.py RDE_skempi_entropy.pkl --grouped_csv_path ./data/complex_sequences_grouped_99.csv

# python -m rde.linear.extract_embeddings \
#   --ckpt ./trained_models/RDE.pt \
#   --skempi_dir ./data/SKEMPI_v2/PDBs \
#   --skempi_cache_path ./data/SKEMPI_v2_cache.pkl \
#   --skempi_csv_path ./data/SKEMPI_v2/skempi_v2.csv \
#   --output_dir complex_embeddings \
#   --device cuda

# python extract_ddg_network_embeddings.py \
#   --ckpt logs_skempi_grouped/rde_ddg_skempi_grouped_new_60/checkpoints/30000.pt \
#   --config configs/train/rde_ddg_skempi_grouped.yml \
#   --output_dir ddg_network_embeddings \
#   --device cuda

python extract_ddg_network_embeddings.py --ckpt logs_skempi_wildtype_only/rde_ddg_skempi_wildtype_only_60/checkpoints/30000.pt --config logs_skempi_wildtype_only/rde_ddg_skempi_wildtype_only_60/rde_ddg_skempi_wildtype_only.yml --output_dir ddg_network_embeddings_wildtype_only --device cuda