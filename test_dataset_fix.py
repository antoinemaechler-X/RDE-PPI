#!/usr/bin/env python3

import os
import sys
sys.path.append('.')

from rde.datasets.skempi_separate_structures import SkempiSeparateStructuresDataset

def test_dataset():
    print("Testing updated SKEMPI dataset...")
    
    # Test dataset loading
    dataset = SkempiSeparateStructuresDataset(
        csv_path='./data/SKEMPI2/SKEMPI_filtered.csv',
        wildtype_dir='./data/wildtype',
        optimized_dir='./data/optimized',
        cache_dir='./data/SKEMPI_v2_cache_separate_structures',
        folds_dir='./cross_validation_folds_final',
        threshold=60,
        split='train',
        cvfold_index=0,
        reset=True  # Force reload
    )
    
    print(f"Dataset size: {len(dataset)}")
    
    # Check a few entries to see if they have unique complex IDs
    complex_ids = set()
    pdb_codes = set()
    
    for i in range(min(100, len(dataset))):
        entry = dataset.entries[i]
        complex_ids.add(entry['complex'])
        pdb_codes.add(entry['pdbcode'])
        print(f"Entry {i}: complex={entry['complex']}, pdbcode={entry['pdbcode']}, mutstr={entry['mutstr']}, ddG={entry['ddG']}")
    
    print(f"\nUnique complex IDs: {len(complex_ids)}")
    print(f"Unique PDB codes: {len(pdb_codes)}")
    
    # Check if we have the same number of complex IDs as entries (should be true for unique mutations)
    if len(complex_ids) == len(dataset.entries):
        print("✓ SUCCESS: Each entry has a unique complex ID!")
    else:
        print("✗ FAILURE: Some entries share complex IDs")
    
    # Test loading a few samples
    print("\nTesting sample loading...")
    for i in range(min(5, len(dataset))):
        try:
            sample = dataset[i]
            print(f"Sample {i}: complex={sample['wildtype']['complex']}, mutstr={sample['wildtype']['mutstr']}")
        except Exception as e:
            print(f"Error loading sample {i}: {e}")
    
    print("\nTest completed!")

if __name__ == '__main__':
    test_dataset() 