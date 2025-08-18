#!/usr/bin/env python3

import pandas as pd
import os

def test_sequence_mapping():
    # Load the CSV and check a few entries
    df = pd.read_csv('./data/SKEMPI2/SKEMPI_filtered.csv')
    
    print("Testing sequence mapping...")
    
    # Check a few mutations
    test_mutations = df['Mutation(s)_cleaned'].head(5).tolist()
    
    for mut_str in test_mutations:
        print(f"\nMutation: {mut_str}")
        
        # Parse like the dataset does
        if ',' in mut_str:
            muts = mut_str.split(',')
        else:
            muts = [mut_str]
        
        for m in muts:
            m = m.strip()
            if len(m) >= 4:
                wt_type = m[0]  # L in LI38G
                mutchain = m[1]  # I in LI38G  
                mt_type = m[-1]  # G in LI38G
                mutseq = int(m[2:-1])  # 38 in LI38G
                
                print(f"  Parsed: WT={wt_type}, Chain={mutchain}, ResSeq={mutseq}, MT={mt_type}")
                
                # Check if structure file exists
                complex_id = df[df['Mutation(s)_cleaned'] == mut_str]['#Pdb'].iloc[0]
                wildtype_file = f"{complex_id}.pdb"
                optimized_file = f"{complex_id}.pdb"
                
                wt_exists = os.path.exists(f"./data/wildtype/{wildtype_file}")
                opt_exists = os.path.exists(f"./data/optimized/{optimized_file}")
                
                print(f"  Structure files: WT={wt_exists}, OPT={opt_exists}")

if __name__ == "__main__":
    test_sequence_mapping() 