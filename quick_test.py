#!/usr/bin/env python3

import pandas as pd

def quick_test():
    # Load the CSV and check a few entries
    df = pd.read_csv('./data/SKEMPI2/SKEMPI_filtered.csv')
    
    print("First 5 entries:")
    for i, row in df.head().iterrows():
        print(f"Entry {i}: {row['#Pdb']} - {row['Mutation(s)_cleaned']} - ddG: {row['ddG']}")
    
    # Check if mutations are being parsed correctly
    print("\nChecking mutation parsing...")
    
    # Parse a few mutations manually
    mutations = df['Mutation(s)_cleaned'].head(10).tolist()
    for mut in mutations:
        print(f"Mutation: {mut}")
        # Parse like the dataset does
        if ',' in mut:
            muts = mut.split(',')
        else:
            muts = [mut]
        
        for m in muts:
            m = m.strip()
            if len(m) >= 4:
                chain = m[0]
                resseq = int(m[1:-3])
                wt = m[-3]
                mt = m[-1]
                print(f"  Chain: {chain}, ResSeq: {resseq}, WT: {wt}, MT: {mt}")

if __name__ == "__main__":
    quick_test() 