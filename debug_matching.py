#!/usr/bin/env python3
"""
Debug script to check complex name matching between SKEMPI and grouped CSV
"""

import pandas as pd
import numpy as np

def debug_matching():
    print("=== Debugging Complex Name Matching ===")
    
    # Load grouped CSV
    print("\n1. Loading grouped CSV...")
    grouped_df = pd.read_csv('data/complex_sequences_grouped_60.csv')
    grouped_complexes = set(grouped_df['complex_name'].values)
    print(f"Grouped CSV has {len(grouped_complexes)} unique complexes")
    print("First 10 grouped complexes:")
    for i, name in enumerate(sorted(list(grouped_complexes))[:10]):
        print(f"  {name}")
    
    # Load SKEMPI CSV
    print("\n2. Loading SKEMPI CSV...")
    skempi_df = pd.read_csv('data/SKEMPI_v2/skempi_v2.csv', sep=';')
    print(f"SKEMPI CSV has {len(skempi_df)} rows")
    
    # Check what's in the #Pdb column
    print("\n3. Checking #Pdb column in SKEMPI...")
    pdb_values = skempi_df['#Pdb'].values
    unique_pdb = set(pdb_values)
    print(f"Unique #Pdb values: {len(unique_pdb)}")
    print("First 10 #Pdb values:")
    for i, name in enumerate(sorted(list(unique_pdb))[:10]):
        print(f"  {name}")
    
    # Process complex names like the script does
    print("\n4. Processing complex names like the script...")
    processed_complexes = []
    for pdb_val in pdb_values:
        current_complex_name = str(pdb_val)
        if '_' in current_complex_name:
            processed_complex_name = current_complex_name.split('_', 1)[1]
        else:
            processed_complex_name = current_complex_name
        processed_complexes.append(processed_complex_name)
    
    unique_processed = set(processed_complexes)
    print(f"Unique processed complexes: {len(unique_processed)}")
    print("First 10 processed complexes:")
    for i, name in enumerate(sorted(list(unique_processed))[:10]):
        print(f"  {name}")
    
    # Check overlap
    print("\n5. Checking overlap...")
    overlap = unique_processed.intersection(grouped_complexes)
    missing_in_grouped = unique_processed - grouped_complexes
    missing_in_skempi = grouped_complexes - unique_processed
    
    print(f"Complexes in both: {len(overlap)}")
    print(f"Complexes only in SKEMPI (processed): {len(missing_in_grouped)}")
    print(f"Complexes only in grouped CSV: {len(missing_in_skempi)}")
    
    if missing_in_grouped:
        print("\nFirst 10 complexes only in SKEMPI:")
        for name in sorted(list(missing_in_grouped))[:10]:
            print(f"  {name}")
    
    if missing_in_skempi:
        print("\nFirst 10 complexes only in grouped CSV:")
        for name in sorted(list(missing_in_skempi))[:10]:
            print(f"  {name}")
    
    if overlap:
        print("\nFirst 10 overlapping complexes:")
        for name in sorted(list(overlap))[:10]:
            print(f"  {name}")

if __name__ == "__main__":
    debug_matching() 