#!/usr/bin/env python3
"""
Debug script to test SKEMPI data loading
"""

import pandas as pd
import numpy as np

def test_affinity_parsing():
    """Test parsing affinity values from CSV."""
    df = pd.read_csv('./data/SKEMPI_v2/skempi_v2.csv', sep=';')
    print(f"Loaded CSV with {len(df)} rows")
    
    # Check first few rows
    print("\nFirst 5 rows:")
    for i in range(min(5, len(df))):
        row = df.iloc[i]
        pdb = row['#Pdb']
        affinity_mut = row.get('Affinity_mut_parsed', row.get('Affinity_mut (M)'))
        affinity_wt = row.get('Affinity_wt_parsed', row.get('Affinity_wt (M)'))
        print(f"  {pdb}: mut={affinity_mut}, wt={affinity_wt}")
    
    # Test ddG calculation
    R = 1.987  # cal/(mol·K)
    temp = 298
    
    print("\nTesting ddG calculation:")
    for i in range(min(5, len(df))):
        row = df.iloc[i]
        affinity_mut = row.get('Affinity_mut_parsed', row.get('Affinity_mut (M)'))
        affinity_wt = row.get('Affinity_wt_parsed', row.get('Affinity_wt (M)'))
        
        try:
            # Convert to float
            if isinstance(affinity_mut, str):
                affinity_mut = float(affinity_mut)
            if isinstance(affinity_wt, str):
                affinity_wt = float(affinity_wt)
            
            # Calculate ΔG values
            dg_mut = R * temp * np.log(affinity_mut) / 1000
            dg_wt = R * temp * np.log(affinity_wt) / 1000
            ddg = dg_mut - dg_wt
            
            print(f"  Row {i}: mut={affinity_mut}, wt={affinity_wt}, ddG={ddg:.3f}")
            
        except Exception as e:
            print(f"  Row {i}: Error - {e}")

def test_pdb_files():
    """Test PDB file existence."""
    df = pd.read_csv('./data/SKEMPI_v2/skempi_v2.csv', sep=';')
    
    print("\nTesting PDB files:")
    missing_pdbs = 0
    found_pdbs = 0
    
    for i in range(min(10, len(df))):
        row = df.iloc[i]
        pdbcode = row['#Pdb'].split('_')[0]
        pdb_path = f"./data/SKEMPI_v2/PDBs/{pdbcode.upper()}.pdb"
        
        import os
        if os.path.exists(pdb_path):
            found_pdbs += 1
            print(f"  {pdbcode}: ✓")
        else:
            missing_pdbs += 1
            print(f"  {pdbcode}: ✗")
    
    print(f"Found: {found_pdbs}, Missing: {missing_pdbs}")

if __name__ == '__main__':
    test_affinity_parsing()
    test_pdb_files() 