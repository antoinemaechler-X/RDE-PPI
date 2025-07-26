#!/usr/bin/env python3
"""
Script to generate cluster folds using the exact same code as the user's project
"""

import os
import numpy as np
import pandas as pd
from sklearn.model_selection import KFold
from tqdm.auto import tqdm

def create_complex_to_group_mapping(grouped_csv_path):
    """Create mapping from SKEMPI complex names to group IDs."""
    df = pd.read_csv(grouped_csv_path)
    
    # Create mapping from PDB code to complex_group_id
    pdb_to_group = {}
    for _, row in df.iterrows():
        pdbcode = row['complex_name']
        group_id = row['complex_group_id']
        if pdbcode not in pdb_to_group:
            pdb_to_group[pdbcode] = group_id
    
    return pdb_to_group

def create_cluster_folds(complex_names, complex_to_group, n_folds=10, random_state=42):
    """Create 10 folds based on complex clusters."""
    if complex_to_group is None:
        # Fallback to original splitting method
        unique_ids = np.unique(complex_names)
        kf = KFold(n_splits=n_folds, shuffle=True, random_state=random_state)
        return kf.split(unique_ids)

    # Get unique complex groups
    complex_groups = []
    missing_complexes = []
    for complex_name in complex_names:
        if complex_name in complex_to_group:
            complex_groups.append(complex_to_group[complex_name])
        else:
            # If complex not in grouping, assign to a unique group
            complex_groups.append(f"ungrouped_{complex_name}")
            missing_complexes.append(complex_name)

    complex_groups = np.array(complex_groups)
    unique_groups = np.unique(complex_groups)

    print(f"Creating {n_folds} folds from {len(unique_groups)} complex groups")
    print(f"Average groups per fold: {len(unique_groups) / n_folds:.1f}")

    if missing_complexes:
        print(f"Warning: {len(missing_complexes)} complexes not found in grouping file:")
        print(f"Missing complexes: {missing_complexes[:10]}...")  # Show first 10

    # Check what complexes we have in our dataset
    unique_complexes_in_data = np.unique(complex_names)
    print(f"Total unique complexes in dataset: {len(unique_complexes_in_data)}")

    # Create KFold splitter for groups
    kf = KFold(n_splits=n_folds, shuffle=True, random_state=random_state)

    # Return fold indices for the complex_names array
    fold_indices = []
    for train_group_idx, test_group_idx in kf.split(unique_groups):
        train_groups = unique_groups[train_group_idx]
        test_groups = unique_groups[test_group_idx]

        # Create masks for this fold
        train_mask = np.isin(complex_groups, train_groups)
        test_mask = np.isin(complex_groups, test_groups)

        # Get indices for train and test
        train_indices = np.where(train_mask)[0]
        test_indices = np.where(test_mask)[0]

        fold_indices.append((train_indices, test_indices))

        print(f"Fold: {len(train_groups)} train groups, {len(test_groups)} test groups")
        print(f"      {len(train_indices)} train samples, {len(test_indices)} test samples")

    return fold_indices

def main():
    """Main function to generate splits using SKEMPI data."""
    
    # Load the grouped CSV mapping
    print("Loading complex to group mapping...")
    complex_to_group = create_complex_to_group_mapping('data/complex_sequences_grouped_60.csv')
    print(f"Loaded mapping for {len(complex_to_group)} complexes")
    
    # Load SKEMPI data to get complex names and count mutations
    print("\nLoading SKEMPI data...")
    df = pd.read_csv('data/SKEMPI_v2/skempi_v2.csv', sep=';')
    print(f"Loaded {len(df)} rows from SKEMPI CSV")
    
    # Extract complex names and filter to only those in grouped CSV
    complex_names = []
    for _, r in df.iterrows():
        current_complex_name = str(r['#Pdb'])
        if '_' in current_complex_name:
            processed_complex_name = current_complex_name.split('_', 1)[0]  # Take first part (PDB code)
        else:
            processed_complex_name = current_complex_name
        
        # Only include if it's in the grouped CSV
        if processed_complex_name in complex_to_group:
            complex_names.append(processed_complex_name)
    
    complex_names = np.array(complex_names)
    print(f"Loaded {len(complex_names)} complex names from SKEMPI data (filtered to grouped CSV)")
    print(f"Unique complexes: {len(np.unique(complex_names))}")
    
    # Generate folds
    print("\nGenerating cluster folds...")
    fold_indices = create_cluster_folds(complex_names, complex_to_group, n_folds=10, random_state=42)
    
    print(f"\nGenerated {len(fold_indices)} folds")
    return fold_indices

if __name__ == "__main__":
    main() 