import pickle

cache_path = 'data/SKEMPI_v2_cache.pkl'
pdbcode = '1dan'  # Lowercase as in cache

# 1-based canonical mapping
map1 = {
    1: 'ALA', 2: 'ARG', 3: 'ASN', 4: 'ASP', 5: 'CYS', 6: 'GLN', 7: 'GLY', 8: 'GLU', 9: 'HIS', 10: 'ILE',
    11: 'LEU', 12: 'LYS', 13: 'MET', 14: 'PHE', 15: 'PRO', 16: 'SER', 17: 'THR', 18: 'TRP', 19: 'TYR', 20: 'VAL'
}
# 0-based canonical mapping
map0 = {
    0: 'ALA', 1: 'ARG', 2: 'ASN', 3: 'ASP', 4: 'CYS', 5: 'GLN', 6: 'GLY', 7: 'GLU', 8: 'HIS', 9: 'ILE',
    10: 'LEU', 11: 'LYS', 12: 'MET', 13: 'PHE', 14: 'PRO', 15: 'SER', 16: 'THR', 17: 'TRP', 18: 'TYR', 19: 'VAL'
}
# BioPython mapping (index: 0-19, order: A, R, N, D, C, Q, E, G, H, I, L, K, M, F, P, S, T, W, Y, V)
map_biopython = {
    0: 'ALA', 1: 'ARG', 2: 'ASN', 3: 'ASP', 4: 'CYS', 5: 'GLN', 6: 'GLU', 7: 'GLY', 8: 'HIS', 9: 'ILE',
    10: 'LEU', 11: 'LYS', 12: 'MET', 13: 'PHE', 14: 'PRO', 15: 'SER', 16: 'THR', 17: 'TRP', 18: 'TYR', 19: 'VAL'
}
# Fallback: just print the code

def get_all_translations(code):
    return {
        '1-based canonical': map1.get(code, f'UNK({code})'),
        '0-based canonical': map0.get(code, f'UNK({code})'),
        'BioPython': map_biopython.get(code, f'UNK({code})'),
        'Raw code': code
    }

with open(cache_path, 'rb') as f:
    cache = pickle.load(f)

if pdbcode not in cache:
    print(f"{pdbcode} not found in cache.")
    exit(1)

data = cache[pdbcode][0]

# Print aa at positions 33-40 in chain L
chain_ids = data['chain_id']
resseqs = data['resseq']
aas = data['aa']

positions = list(range(33, 41))
found_any = False
for i in range(len(chain_ids)):
    chain = chain_ids[i]
    resseq = resseqs[i].item() if hasattr(resseqs[i], 'item') else resseqs[i]
    if chain == 'L' and resseq in positions:
        aa_code = aas[i].item() if hasattr(aas[i], 'item') else aas[i]
        translations = get_all_translations(aa_code)
        print(f"Chain L, residue {resseq}: ")
        for k, v in translations.items():
            print(f"  {k}: {v}")
        found_any = True
if not found_any:
    print("No residues 33-40 found in chain L.") 