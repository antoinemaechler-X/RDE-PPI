#!/usr/bin/env python3
"""
Validation script to verify that extracted embeddings match the ground truth txt files.
Run this after the main extraction script to ensure everything is correct.
"""

import os
import glob
from extract_ddg_network_embeddings import load_expected_complex_ids

def validate_fold_embeddings(output_dir, folds_dir, threshold, fold_num):
    """Validate embeddings for a specific fold."""
    print(f"\nValidating fold {fold_num}...")
    print("-" * 40)
    
    fold_dir = os.path.join(output_dir, f"fold_{fold_num}")
    if not os.path.exists(fold_dir):
        print(f"  ERROR: Fold directory {fold_dir} does not exist!")
        return False
    
    # Check train and test splits
    for split in ['train', 'test']:
        split_dir = os.path.join(fold_dir, split)
        if not os.path.exists(split_dir):
            print(f"  ERROR: Split directory {split_dir} does not exist!")
            continue
        
        print(f"  {split.capitalize()} split:")
        
        # Load expected complex IDs
        try:
            expected_ids = load_expected_complex_ids(folds_dir, threshold, fold_num, split)
        except Exception as e:
            print(f"    ERROR loading expected IDs: {e}")
            continue
        
        # Check what files exist
        npy_files = glob.glob(os.path.join(split_dir, "*.npy"))
        pkl_files = glob.glob(os.path.join(split_dir, "*.pkl"))
        
        print(f"    Expected: {len(expected_ids)} complexes")
        print(f"    Found: {len(npy_files)} .npy files, {len(pkl_files)} .pkl files")
        
        # Check each expected complex ID
        missing_files = []
        for complex_id in expected_ids:
            expected_files = [
                f"{complex_id}_wt.npy",
                f"{complex_id}_mt.npy", 
                f"{complex_id}_wt_resmap.pkl",
                f"{complex_id}_mt_resmap.pkl"
            ]
            
            for filename in expected_files:
                filepath = os.path.join(split_dir, filename)
                if not os.path.exists(filepath):
                    missing_files.append(filename)
        
        if missing_files:
            print(f"    ERROR: Missing files: {missing_files[:5]}...")  # Show first 5
            return False
        else:
            print(f"    ✓ All files present for {len(expected_ids)} complexes")
    
    return True

def main():
    """Main validation function."""
    import argparse
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--output_dir', type=str, required=True, help='Output directory with extracted embeddings')
    parser.add_argument('--folds_dir', type=str, default='cross_validation_folds_final/60_percent', help='Directory with fold txt files')
    parser.add_argument('--threshold', type=int, default=60, help='Threshold percentage for folds')
    args = parser.parse_args()
    
    print("EMBEDDING VALIDATION")
    print("=" * 50)
    print(f"Output directory: {args.output_dir}")
    print(f"Folds directory: {args.folds_dir}")
    print(f"Threshold: {args.threshold}%")
    
    if not os.path.exists(args.output_dir):
        print(f"ERROR: Output directory {args.output_dir} does not exist!")
        return
    
    # Validate each fold
    all_valid = True
    for fold_num in range(1, 11):  # Folds 1-10
        if not validate_fold_embeddings(args.output_dir, args.folds_dir, args.threshold, fold_num):
            all_valid = False
    
    print("\n" + "=" * 50)
    if all_valid:
        print("✓ ALL VALIDATIONS PASSED!")
        print("All embeddings match the ground truth txt files.")
    else:
        print("✗ SOME VALIDATIONS FAILED!")
        print("Check the errors above and fix the extraction process.")
    
    print("=" * 50)

if __name__ == "__main__":
    main() 