#!/usr/bin/env python3
"""
Test script to verify the custom folds SKEMPI dataset works correctly.
"""

import pandas as pd
from rde.datasets.skempi_custom_folds import SkempiCustomFoldsDataset, load_custom_fold_splits

def test_fold_loading():
    """Test loading fold splits from files."""
    print("Testing fold split loading...")
    
    # Test loading fold 0 (fold_1 in files)
    train_complexes, test_complexes = load_custom_fold_splits(
        './cross_validation_folds_final', 60, 0
    )
    
    print(f"Fold 0: {len(train_complexes)} train PDB codes, {len(test_complexes)} test PDB codes")
    print(f"Train examples: {list(train_complexes)[:5]}")
    print(f"Test examples: {list(test_complexes)[:5]}")
    
    # Check for overlap
    overlap = train_complexes.intersection(test_complexes)
    if len(overlap) > 0:
        print(f"WARNING: Overlap between train and test: {overlap}")
    else:
        print("✓ No overlap between train and test sets")
    
    # Test another fold
    train_complexes2, test_complexes2 = load_custom_fold_splits(
        './cross_validation_folds_final', 60, 1
    )
    print(f"Fold 1: {len(train_complexes2)} train PDB codes, {len(test_complexes2)} test PDB codes")
    
    # Check that folds are different
    train_overlap = train_complexes.intersection(train_complexes2)
    test_overlap = test_complexes.intersection(test_complexes2)
    print(f"Train overlap between folds: {len(train_overlap)}")
    print(f"Test overlap between folds: {len(test_overlap)}")

def test_dataset():
    """Test the dataset creation and splitting."""
    print("\nTesting dataset creation...")
    
    # Create train dataset
    train_dataset = SkempiCustomFoldsDataset(
        csv_path='./data/SKEMPI_v2/skempi_v2.csv',
        pdb_dir='./data/SKEMPI_v2/PDBs',
        cache_dir='./data/SKEMPI_v2_cache_custom_folds_test',
        folds_dir='./cross_validation_folds_final',
        threshold=60,
        split='train',
        cvfold_index=0,
        reset=True
    )
    
    # Create validation dataset
    val_dataset = SkempiCustomFoldsDataset(
        csv_path='./data/SKEMPI_v2/skempi_v2.csv',
        pdb_dir='./data/SKEMPI_v2/PDBs',
        cache_dir='./data/SKEMPI_v2_cache_custom_folds_test',
        folds_dir='./cross_validation_folds_final',
        threshold=60,
        split='val',
        cvfold_index=0,
        reset=False
    )
    
    print(f"Train dataset size: {len(train_dataset)}")
    print(f"Validation dataset size: {len(val_dataset)}")
    
    # Check for data leakage
    train_pdb_codes = set([e['pdbcode'] for e in train_dataset.entries])
    val_pdb_codes = set([e['pdbcode'] for e in val_dataset.entries])
    leakage = train_pdb_codes.intersection(val_pdb_codes)
    
    print(f"Train PDB codes: {len(train_pdb_codes)}")
    print(f"Val PDB codes: {len(val_pdb_codes)}")
    print(f"Data leakage: {len(leakage)} PDB codes")
    
    if len(leakage) > 0:
        print("WARNING: Data leakage detected!")
        print("Leaking complexes:", list(leakage)[:5])
    else:
        print("✓ No data leakage detected")
    
    # Show some examples from each dataset
    print(f"\nTrain dataset examples:")
    for i in range(min(3, len(train_dataset.entries))):
        entry = train_dataset.entries[i]
        print(f"  {entry['complex']}: {entry['mutstr']} -> {entry['ddG']:.3f}")
    
    print(f"\nValidation dataset examples:")
    for i in range(min(3, len(val_dataset.entries))):
        entry = val_dataset.entries[i]
        print(f"  {entry['complex']}: {entry['mutstr']} -> {entry['ddG']:.3f}")

def test_different_thresholds():
    """Test loading different threshold percentages."""
    print("\nTesting different thresholds...")
    
    thresholds = [40, 60, 80, 95, 99, 100]
    
    for threshold in thresholds:
        try:
            train_complexes, test_complexes = load_custom_fold_splits(
                './cross_validation_folds_final', threshold, 0
            )
            print(f"Threshold {threshold}%: {len(train_complexes)} train, {len(test_complexes)} test")
        except FileNotFoundError as e:
            print(f"Threshold {threshold}%: Not available ({e})")

def test_dataset_manager():
    """Test the dataset manager."""
    print("\nTesting dataset manager...")
    
    # Create a simple config-like object
    class Config:
        class data:
            csv_path = './data/SKEMPI_v2/skempi_v2.csv'
            pdb_dir = './data/SKEMPI_v2/PDBs'
            cache_dir = './data/SKEMPI_v2_cache_custom_folds_test'
            folds_dir = './cross_validation_folds_final'
            transform = []
        
        class train:
            batch_size = 4
    
    config = Config()
    
    from rde.utils.skempi_custom_folds import SkempiCustomFoldsDatasetManager
    from rde.utils.misc import BlackHole
    
    dataset_mgr = SkempiCustomFoldsDatasetManager(
        config, 
        num_cvfolds=2,  # Just test with 2 folds
        threshold=60,
        num_workers=0,  # No multiprocessing for testing
        logger=BlackHole(),
    )
    
    print("Dataset manager created successfully")
    print(f"Number of train iterators: {len(dataset_mgr.train_iterators)}")
    print(f"Number of val loaders: {len(dataset_mgr.val_loaders)}")
    
    # Test getting a batch
    try:
        train_iterator = dataset_mgr.get_train_iterator(0)
        batch = next(train_iterator)
        print(f"Successfully got training batch with {batch['size']} samples")
    except Exception as e:
        print(f"Error getting training batch: {e}")

if __name__ == '__main__':
    test_fold_loading()
    test_different_thresholds()
    test_dataset()
    test_dataset_manager()
    print("\nAll tests completed!") 