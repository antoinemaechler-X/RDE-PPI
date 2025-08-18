#!/usr/bin/env python3
"""
Test script to verify complex ID loading and PDB verification.
Run this before the main extraction script to ensure everything is set up correctly.
"""

import os
from extract_ddg_network_embeddings import load_expected_complex_ids, verify_pdb_files_exist

def test_complex_id_loading():
    """Test loading complex IDs from txt files."""
    folds_dir = "cross_validation_folds_final/60_percent"
    
    print("Testing complex ID loading...")
    print("="*50)
    
    # Test a few folds
    for fold in [1, 2, 3]:
        print(f"\nFold {fold}:")
        
        # Test train split
        try:
            train_ids = load_expected_complex_ids(folds_dir, 60, fold, "train")
            print(f"  Train: {len(train_ids)} complexes")
            print(f"  Sample IDs: {list(train_ids)[:5]}")
        except Exception as e:
            print(f"  Train: ERROR - {e}")
        
        # Test test split
        try:
            test_ids = load_expected_complex_ids(folds_dir, 60, fold, "test")
            print(f"  Test: {len(test_ids)} complexes")
            print(f"  Sample IDs: {list(test_ids)[:5]}")
        except Exception as e:
            print(f"  Test: ERROR - {e}")
    
    print("\n" + "="*50)
    print("Complex ID loading test completed!")

def test_pdb_verification():
    """Test PDB file verification."""
    wildtype_dir = "data/wildtype"
    
    print("\nTesting PDB file verification...")
    print("="*50)
    
    # Test with a few complex IDs
    test_complex_ids = ["692_1N8O", "675_1N8O", "10_1ACB", "11_1ACB"]
    
    print(f"Testing PDB existence for: {test_complex_ids}")
    print(f"Wildtype directory: {wildtype_dir}")
    
    if os.path.exists(wildtype_dir):
        verify_pdb_files_exist(test_complex_ids, wildtype_dir)
    else:
        print(f"Warning: Wildtype directory {wildtype_dir} does not exist!")
    
    print("\n" + "="*50)
    print("PDB verification test completed!")

if __name__ == "__main__":
    test_complex_id_loading()
    test_pdb_verification() 