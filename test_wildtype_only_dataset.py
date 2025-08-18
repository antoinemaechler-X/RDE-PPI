#!/usr/bin/env python3
"""
Test script for the wildtype-only dataset to ensure it works correctly.
"""

import os
import sys
import argparse
from rde.datasets.skempi_wildtype_only import SkempiWildtypeOnlyDataset


def test_dataset():
    """Test the dataset creation and basic functionality."""
    print("Testing wildtype-only dataset...")
    
    # Create train dataset
    train_dataset = SkempiWildtypeOnlyDataset(
        csv_path='./data/SKEMPI_filtered.csv',
        wildtype_dir='./data/wildtype',
        cache_dir='./data/SKEMPI_v2_cache_wildtype_only_test',
        folds_dir='./cross_validation_folds_final',
        threshold=60,
        split='train',
        cvfold_index=0,
        reset=True
    )
    
    # Create validation dataset
    val_dataset = SkempiWildtypeOnlyDataset(
        csv_path='./data/SKEMPI_filtered.csv',
        wildtype_dir='./data/wildtype',
        cache_dir='./data/SKEMPI_v2_cache_wildtype_only_test',
        folds_dir='./cross_validation_folds_final',
        threshold=60,
        split='val',
        cvfold_index=0,
        reset=False
    )
    
    print(f"Train dataset size: {len(train_dataset)}")
    print(f"Validation dataset size: {len(val_dataset)}")
    
    # Check for data leakage
    train_complexes = set([e['complex'] for e in train_dataset.entries])
    val_complexes = set([e['complex'] for e in val_dataset.entries])
    leakage = train_complexes.intersection(val_complexes)
    
    print(f"Train complexes: {len(train_complexes)}")
    print(f"Val complexes: {len(val_complexes)}")
    print(f"Data leakage: {len(leakage)} complexes")
    
    if len(leakage) > 0:
        print("WARNING: Data leakage detected!")
        print("Leaking complexes:", list(leakage)[:5])
    else:
        print("✓ No data leakage detected")
    
    # Test a few samples
    print(f"\nTesting sample loading...")
    for i in range(min(3, len(train_dataset))):
        try:
            sample = train_dataset[i]
            print(f"  Sample {i}: {sample['complex']} - {sample['mutstr']} -> {sample['ddG']:.3f}")
            print(f"    Shape: {sample['aa'].shape}, Mutations: {sample['mut_flag'].sum().item()}")
        except Exception as e:
            print(f"  Error loading sample {i}: {e}")
    
    # Test validation samples
    print(f"\nTesting validation samples...")
    for i in range(min(3, len(val_dataset))):
        try:
            sample = val_dataset[i]
            print(f"  Val sample {i}: {sample['complex']} - {sample['mutstr']} -> {sample['ddG']:.3f}")
        except Exception as e:
            print(f"  Error loading val sample {i}: {e}")
    
    # Check ddG distribution
    train_ddgs = [e['ddG'] for e in train_dataset.entries]
    val_ddgs = [e['ddG'] for e in val_dataset.entries]
    
    print(f"\nDDG distribution:")
    print(f"  Train: {len([d for d in train_ddgs if d > 0])} positive, {len([d for d in train_ddgs if d < 0])} negative")
    print(f"  Val: {len([d for d in val_ddgs if d > 0])} positive, {len([d for d in val_ddgs if d < 0])} negative")
    
    print("\n✓ Dataset test completed successfully!")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Test the wildtype-only dataset")
    parser.add_argument('--csv_path', type=str, default='./data/SKEMPI_filtered.csv')
    parser.add_argument('--wildtype_dir', type=str, default='./data/wildtype')
    parser.add_argument('--cache_dir', type=str, default='./data/SKEMPI_v2_cache_wildtype_only_test')
    parser.add_argument('--folds_dir', type=str, default='./cross_validation_folds_final')
    parser.add_argument('--threshold', type=int, default=60)
    args = parser.parse_args()
    
    # Check if required files exist
    if not os.path.exists(args.csv_path):
        print(f"Error: CSV file not found: {args.csv_path}")
        sys.exit(1)
    
    if not os.path.exists(args.wildtype_dir):
        print(f"Error: Wildtype directory not found: {args.wildtype_dir}")
        sys.exit(1)
    
    if not os.path.exists(args.folds_dir):
        print(f"Error: Folds directory not found: {args.folds_dir}")
        sys.exit(1)
    
    test_dataset() 