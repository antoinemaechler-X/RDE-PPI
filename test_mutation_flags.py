#!/usr/bin/env python3

import sys
sys.path.append('.')

def test_mutation_flags():
    # Test the mutation parsing logic
    def parse_mutation(mut_str):
        # Parse like the dataset does
        if ',' in mut_str:
            muts = mut_str.split(',')
        else:
            muts = [mut_str]
        
        mutations = []
        for m in muts:
            m = m.strip()
            if len(m) >= 4:
                wt_type = m[0]  # L in LI38G
                mutchain = m[1]  # I in LI38G  
                mt_type = m[-1]  # G in LI38G
                mutseq = int(m[2:-1])  # 38 in LI38G
                
                mutations.append({
                    'wt': wt_type,
                    'mt': mt_type,
                    'chain': mutchain,
                    'resseq': mutseq,
                    'icode': ' ',
                    'name': m
                })
        
        return mutations
    
    # Test a few mutations
    test_mutations = ['LI38G', 'LI38S', 'LI38P', 'LI38I', 'LI38D']
    
    print("Testing mutation parsing:")
    for mut_str in test_mutations:
        mutations = parse_mutation(mut_str)
        print(f"  {mut_str}: {mutations}")
    
    print("\nTesting mutation flag logic:")
    print("For separate structures approach:")
    print("  Wildtype structure:")
    print("    - aa = wildtype amino acids")
    print("    - aa_mut = wildtype amino acids (same as aa)")
    print("    - mut_flag = False everywhere")
    print("  Mutant structure:")
    print("    - aa = mutant amino acids (already applied)")
    print("    - aa_mut = wildtype amino acids (for reference)")
    print("    - mut_flag = True at mutation sites")
    
    print("\nFor custom folds approach:")
    print("  Single structure:")
    print("    - aa = wildtype amino acids")
    print("    - aa_mut = mutant amino acids")
    print("    - mut_flag = True where aa != aa_mut")

if __name__ == "__main__":
    test_mutation_flags() 