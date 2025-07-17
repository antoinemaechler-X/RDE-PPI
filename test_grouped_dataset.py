#!/usr/bin/env python3
"""
Test script to verify the grouped SKEMPI dataset works correctly.
"""

import pandas as pd
from rde.datasets.skempi_grouped import SkempiGroupedDataset, create_complex_to_group_mapping

def test_mapping():
    """Test the complex to group mapping."""
    print("Testing complex to group mapping...")
    
    # Load the grouped CSV
    grouped_df = pd.read_csv('./data/complex_sequences_grouped_60.csv')
    
    # Create mapping
    pdb_to_group = create_complex_to_group_mapping('./data/complex_sequences_grouped_60.csv')
    
    print(f"Found {len(pdb_to_group)} unique PDB codes in grouped file")
    print(f"Number of unique groups: {len(set(pdb_to_group.values()))}")
    
    # Show some examples
    print("\nExample mappings:")
    for i, (pdb, group) in enumerate(list(pdb_to_group.items())[:10]):
        print(f"  {pdb} -> Group {group}")
    
    # Check group distribution
    group_counts = {}
    for group in pdb_to_group.values():
        group_counts[group] = group_counts.get(group, 0) + 1
    
    print(f"\nGroup distribution (showing first 10 groups):")
    for group in sorted(group_counts.keys())[:10]:
        print(f"  Group {group}: {group_counts[group]} complexes")

def test_dataset():
    """Test the dataset creation and splitting."""
    print("\nTesting dataset creation...")
    
    # Create train dataset
    train_dataset = SkempiGroupedDataset(
        csv_path='./data/SKEMPI_v2/skempi_v2.csv',
        pdb_dir='./data/SKEMPI_v2/PDBs',
        cache_dir='./data/SKEMPI_v2_cache_grouped_test',
        grouped_csv_path='./data/complex_sequences_grouped_60.csv',
        split='train',
        num_cvfolds=10,
        cvfold_index=0,
        reset=True
    )
    
    # Create validation dataset
    val_dataset = SkempiGroupedDataset(
        csv_path='./data/SKEMPI_v2/skempi_v2.csv',
        pdb_dir='./data/SKEMPI_v2/PDBs',
        cache_dir='./data/SKEMPI_v2_cache_grouped_test',
        grouped_csv_path='./data/complex_sequences_grouped_60.csv',
        split='val',
        num_cvfolds=10,
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
    
    # Show some examples from each dataset
    print(f"\nTrain dataset examples:")
    for i in range(min(3, len(train_dataset.entries))):
        entry = train_dataset.entries[i]
        print(f"  {entry['complex']}: {entry['mutstr']} -> {entry['ddG']:.3f}")
    
    print(f"\nValidation dataset examples:")
    for i in range(min(3, len(val_dataset.entries))):
        entry = val_dataset.entries[i]
        print(f"  {entry['complex']}: {entry['mutstr']} -> {entry['ddG']:.3f}")

def test_different_folds():
    """Test that different folds have different data."""
    print("\nTesting different folds...")
    
    fold_complexes = []
    for fold in range(3):  # Test first 3 folds
        dataset = SkempiGroupedDataset(
            csv_path='./data/SKEMPI_v2/skempi_v2.csv',
            pdb_dir='./data/SKEMPI_v2/PDBs',
            cache_dir='./data/SKEMPI_v2_cache_grouped_test',
            grouped_csv_path='./data/complex_sequences_grouped_60.csv',
            split='val',
            num_cvfolds=10,
            cvfold_index=fold,
            reset=False
        )
        complexes = set([e['complex'] for e in dataset.entries])
        fold_complexes.append(complexes)
        print(f"Fold {fold}: {len(complexes)} complexes")
    
    # Check overlap between folds
    for i in range(len(fold_complexes)):
        for j in range(i+1, len(fold_complexes)):
            overlap = fold_complexes[i].intersection(fold_complexes[j])
            print(f"Overlap between fold {i} and {j}: {len(overlap)} complexes")
            if len(overlap) > 0:
                print("WARNING: Overlap detected between folds!")

if __name__ == '__main__':
    test_mapping()
    test_dataset()
    test_different_folds()
    print("\n✓ All tests completed!") 