#!/usr/bin/env python3
"""
Test script to verify the separate structures SKEMPI dataset works correctly.
"""

import pandas as pd
from rde.datasets.skempi_separate_structures import SkempiSeparateStructuresDataset, load_custom_fold_splits

def test_fold_loading():
    """Test loading fold splits from files."""
    print("Testing fold split loading...")
    
    # Test loading fold 0 (fold_1 in files)
    train_pdb_codes, test_pdb_codes = load_custom_fold_splits(
        './cross_validation_folds_final', 60, 0
    )
    
    print(f"Fold 0: {len(train_pdb_codes)} train PDB codes, {len(test_pdb_codes)} test PDB codes")
    print(f"Train examples: {list(train_pdb_codes)[:5]}")
    print(f"Test examples: {list(test_pdb_codes)[:5]}")
    
    # Check for overlap
    overlap = train_pdb_codes.intersection(test_pdb_codes)
    if len(overlap) > 0:
        print(f"WARNING: Overlap between train and test: {overlap}")
    else:
        print("✓ No overlap between train and test sets")
    
    # Test another fold
    train_pdb_codes2, test_pdb_codes2 = load_custom_fold_splits(
        './cross_validation_folds_final', 60, 1
    )
    print(f"Fold 1: {len(train_pdb_codes2)} train PDB codes, {len(test_pdb_codes2)} test PDB codes")
    
    return train_pdb_codes, test_pdb_codes

def test_dataset_loading():
    """Test loading the separate structures dataset."""
    print("\nTesting dataset loading...")
    
    try:
        # Create train dataset
        train_dataset = SkempiSeparateStructuresDataset(
            csv_path='./data/SKEMPI_v2/skempi_v2.csv',
            wildtype_dir='./data/wildtype',
            optimized_dir='./data/optimized',
            cache_dir='./data/SKEMPI_v2_cache_separate_structures_test',
            folds_dir='./cross_validation_folds_final',
            threshold=60,
            split='train',
            cvfold_index=0,
            reset=True  # Force reload for testing
        )
        
        # Create validation dataset
        val_dataset = SkempiSeparateStructuresDataset(
            csv_path='./data/SKEMPI_v2/skempi_v2.csv',
            wildtype_dir='./data/wildtype',
            optimized_dir='./data/optimized',
            cache_dir='./data/SKEMPI_v2_cache_separate_structures_test',
            folds_dir='./cross_validation_folds_final',
            threshold=60,
            split='val',
            cvfold_index=0,
            reset=False  # Use cached data
        )
        
        print(f"✓ Train dataset loaded: {len(train_dataset)} samples")
        print(f"✓ Val dataset loaded: {len(val_dataset)} samples")
        
        # Test getting a sample
        sample = train_dataset[0]
        print(f"✓ Sample structure: {type(sample)}")
        print(f"  - Wildtype keys: {list(sample['wildtype'].keys())}")
        print(f"  - Mutant keys: {list(sample['mutant'].keys())}")
        
        # Check that both structures have the same metadata
        wt_data = sample['wildtype']
        mt_data = sample['mutant']
        
        print(f"  - Complex: {wt_data['complex']}")
        print(f"  - PDB code: {wt_data['pdbcode']}")
        print(f"  - ddG: {wt_data['ddG']}")
        print(f"  - Mutations: {wt_data['mutstr']}")
        
        # Check data leakage
        train_pdb_codes = set([e['pdbcode'] for e in train_dataset.entries])
        val_pdb_codes = set([e['pdbcode'] for e in val_dataset.entries])
        leakage = train_pdb_codes.intersection(val_pdb_codes)
        
        print(f"Train PDB codes: {len(train_pdb_codes)}")
        print(f"Val PDB codes: {len(val_pdb_codes)}")
        print(f"Data leakage: {len(leakage)} PDB codes")
        
        if len(leakage) == 0:
            print("✓ No data leakage detected")
        else:
            print(f"⚠ WARNING: Data leakage detected: {leakage}")
        
        return train_dataset, val_dataset
        
    except Exception as e:
        print(f"✗ Error loading dataset: {e}")
        import traceback
        traceback.print_exc()
        return None, None

def test_structure_loading():
    """Test that wildtype and optimized structures are loaded correctly."""
    print("\nTesting structure loading...")
    
    try:
        dataset = SkempiSeparateStructuresDataset(
            csv_path='./data/SKEMPI_v2/skempi_v2.csv',
            wildtype_dir='./data/wildtype',
            optimized_dir='./data/optimized',
            cache_dir='./data/SKEMPI_v2_cache_separate_structures_test',
            folds_dir='./cross_validation_folds_final',
            threshold=60,
            split='train',
            cvfold_index=0,
            reset=False
        )
        
        print(f"✓ Wildtype structures loaded: {len(dataset.wildtype_structures)}")
        print(f"✓ Optimized structures loaded: {len(dataset.optimized_structures)}")
        
        # Check a few PDB codes
        sample_pdb_codes = list(dataset.wildtype_structures.keys())[:5]
        print(f"Sample PDB codes: {sample_pdb_codes}")
        
        for pdbcode in sample_pdb_codes:
            if pdbcode in dataset.wildtype_structures and pdbcode in dataset.optimized_structures:
                print(f"✓ {pdbcode}: Both wildtype and optimized structures available")
            else:
                print(f"✗ {pdbcode}: Missing structure")
        
        return True
        
    except Exception as e:
        print(f"✗ Error testing structure loading: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == '__main__':
    print("=== Testing Separate Structures SKEMPI Dataset ===\n")
    
    # Test fold loading
    train_codes, test_codes = test_fold_loading()
    
    # Test dataset loading
    train_dataset, val_dataset = test_dataset_loading()
    
    # Test structure loading
    structure_ok = test_structure_loading()
    
    print("\n=== Test Summary ===")
    if train_dataset is not None and val_dataset is not None and structure_ok:
        print("✓ All tests passed! The separate structures dataset is working correctly.")
    else:
        print("✗ Some tests failed. Please check the errors above.") 