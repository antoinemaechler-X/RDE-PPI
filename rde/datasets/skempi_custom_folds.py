import os
import copy
import pickle
import warnings
import torch
import numpy as np
import pandas as pd
from torch.utils.data import Dataset
from tqdm.auto import tqdm
from Bio.PDB.PDBParser import PDBParser
from Bio.PDB.Polypeptide import one_to_index

# Suppress Biopython deprecation warnings
warnings.filterwarnings("ignore", category=DeprecationWarning, module="Bio.*")
warnings.filterwarnings("ignore", message=".*three_to_one.*", category=FutureWarning)

from rde.utils.protein.parsers import parse_biopython_structure


def load_skempi_entries(csv_path, pdb_dir, block_list={'1KBH'}):
    df = pd.read_csv(csv_path, sep=';')
    df['dG_wt'] =  (8.314/4184)*(273.15 + 25.0) * np.log(df['Affinity_wt_parsed'])
    df['dG_mut'] =  (8.314/4184)*(273.15 + 25.0) * np.log(df['Affinity_mut_parsed'])
    df['ddG'] = df['dG_mut'] - df['dG_wt']

    def _parse_mut(mut_name):
        wt_type, mutchain, mt_type = mut_name[0], mut_name[1], mut_name[-1]
        mutseq = int(mut_name[2:-1])
        return {
            'wt': wt_type,
            'mt': mt_type,
            'chain': mutchain,
            'resseq': mutseq,
            'icode': ' ',
            'name': mut_name
        }

    entries = []
    for i, row in df.iterrows():
        pdbcode, group1, group2 = row['#Pdb'].split('_')
        if pdbcode in block_list:
            continue
        mut_str = row['Mutation(s)_cleaned']
        muts = list(map(_parse_mut, row['Mutation(s)_cleaned'].split(',')))
        if muts[0]['chain'] in group1:
            group_ligand, group_receptor = group1, group2
        else:
            group_ligand, group_receptor = group2, group1

        pdb_path = os.path.join(pdb_dir, '{}.pdb'.format(pdbcode.upper()))
        if not os.path.exists(pdb_path):
            continue

        if not np.isfinite(row['ddG']):
            continue

        entry = {
            'id': i,
            'complex': row['#Pdb'],
            'mutstr': mut_str,
            'num_muts': len(muts),
            'pdbcode': pdbcode,
            'group_ligand': list(group_ligand),
            'group_receptor': list(group_receptor),
            'mutations': muts,
            'ddG': np.float32(row['ddG']),
            'pdb_path': pdb_path,
        }
        entries.append(entry)

    return entries


def load_custom_fold_splits(folds_dir, threshold, fold_index):
    """
    Load custom fold splits from the cross_validation_folds_final directory.
    
    Args:
        folds_dir: Path to cross_validation_folds_final directory
        threshold: Percentage threshold (e.g., 60 for 60_percent)
        fold_index: Which fold to load (0-indexed)
    
    Returns:
        train_complexes: Set of complex names for training
        test_complexes: Set of complex names for testing
    """
    fold_dir = os.path.join(folds_dir, f"{threshold}_percent", f"fold_{fold_index + 1}")
    
    # Load train complexes
    train_file = os.path.join(fold_dir, "train_complex_ids.txt")
    with open(train_file, 'r') as f:
        train_complex_ids = set(line.strip() for line in f if line.strip())
    
    # Load test complexes
    test_file = os.path.join(fold_dir, "test_complex_ids.txt")
    with open(test_file, 'r') as f:
        test_complex_ids = set(line.strip() for line in f if line.strip())
    
    # Extract PDB codes from complex IDs (format: "ID_PDBCODE" -> "PDBCODE")
    def extract_pdb_code(complex_id):
        # Split by underscore and take the last part (the PDB code)
        parts = complex_id.split('_')
        if len(parts) >= 2:
            return parts[-1]  # Take the last part as PDB code
        else:
            return complex_id  # If no underscore, return as is
    
    train_complexes = set(extract_pdb_code(cid) for cid in train_complex_ids)
    test_complexes = set(extract_pdb_code(cid) for cid in test_complex_ids)
    
    return train_complexes, test_complexes


class SkempiCustomFoldsDataset(Dataset):

    def __init__(
        self, 
        csv_path, 
        pdb_dir, 
        cache_dir,
        folds_dir,
        threshold,
        cvfold_index=0, 
        split='train', 
        transform=None, 
        blocklist=frozenset({'1KBH'}), 
        reset=False
    ):
        super().__init__()
        self.csv_path = csv_path
        self.pdb_dir = pdb_dir
        self.cache_dir = cache_dir
        self.folds_dir = folds_dir
        self.threshold = threshold
        os.makedirs(cache_dir, exist_ok=True)
        self.blocklist = blocklist
        self.transform = transform
        self.cvfold_index = cvfold_index
        assert split in ('train', 'val')
        self.split = split

        self.entries_cache = os.path.join(cache_dir, f'entries_custom_folds_{threshold}.pkl')
        self.entries = None
        self.entries_full = None
        self._load_entries(reset)

        # Load structures after entries are finalized
        self.structures_cache = os.path.join(cache_dir, f'structures_custom_folds_{threshold}.pkl')
        self.structures = None
        self._load_structures(reset)

    def _load_entries(self, reset):
        if not os.path.exists(self.entries_cache) or reset:
            self.entries_full = self._preprocess_entries()
        else:
            with open(self.entries_cache, 'rb') as f:
                self.entries_full = pickle.load(f)

        # Load custom fold splits
        train_pdb_codes, test_pdb_codes = load_custom_fold_splits(
            self.folds_dir, self.threshold, self.cvfold_index
        )
        
        # Filter entries based on split using PDB codes
        if self.split == 'val':
            target_pdb_codes = test_pdb_codes
        else:
            target_pdb_codes = train_pdb_codes

        # Filter entries to only include complexes whose PDB code is in the target split
        self.entries = [
            entry for entry in self.entries_full 
            if entry['pdbcode'] in target_pdb_codes
        ]
        
    def _preprocess_entries(self):
        entries = load_skempi_entries(self.csv_path, self.pdb_dir, self.blocklist)
        with open(self.entries_cache, 'wb') as f:
            pickle.dump(entries, f)
        return entries

    def _load_structures(self, reset):
        # Check if we need to reload structures based on the final entries
        if not os.path.exists(self.structures_cache) or reset:
            self.structures = self._preprocess_structures()
        else:
            with open(self.structures_cache, 'rb') as f:
                self.structures = pickle.load(f)
            
            # Check if we have all the PDB codes we need
            needed_pdbcodes = set([e['pdbcode'] for e in self.entries])
            available_pdbcodes = set(self.structures.keys())
            missing_pdbcodes = needed_pdbcodes - available_pdbcodes
            
            if missing_pdbcodes:
                print(f"Missing structures for PDB codes: {missing_pdbcodes}")
                # Load missing structures
                for pdbcode in tqdm(missing_pdbcodes, desc='Loading missing structures'):
                    parser = PDBParser(QUIET=True)
                    pdb_path = os.path.join(self.pdb_dir, '{}.pdb'.format(pdbcode.upper()))
                    model = parser.get_structure(None, pdb_path)[0]
                    data, seq_map = parse_biopython_structure(model)
                    self.structures[pdbcode] = (data, seq_map)
                
                # Save updated structures cache
                with open(self.structures_cache, 'wb') as f:
                    pickle.dump(self.structures, f)

    def _preprocess_structures(self):
        structures = {}
        # Get all PDB codes that are actually used in the final dataset
        pdbcodes = list(set([e['pdbcode'] for e in self.entries]))
        for pdbcode in tqdm(pdbcodes, desc='Structures'):
            parser = PDBParser(QUIET=True)
            pdb_path = os.path.join(self.pdb_dir, '{}.pdb'.format(pdbcode.upper()))
            model = parser.get_structure(None, pdb_path)[0]
            data, seq_map = parse_biopython_structure(model)
            structures[pdbcode] = (data, seq_map)
        with open(self.structures_cache, 'wb') as f:
            pickle.dump(structures, f)
        return structures

    def __len__(self):
        return len(self.entries)

    def __getitem__(self, index):
        entry = self.entries[index]
        data, seq_map = copy.deepcopy( self.structures[entry['pdbcode']] )
        keys = {'id', 'complex', 'mutstr', 'num_muts', 'pdbcode', 'ddG'}
        for k in keys:
            data[k] = entry[k]

        group_id = []
        for ch in data['chain_id']:
            if ch in entry['group_ligand']:
                group_id.append(1)
            elif ch in entry['group_receptor']:
                group_id.append(2)
            else:
                group_id.append(0)
        data['group_id'] = torch.LongTensor(group_id)

        aa_mut = data['aa'].clone()
        for mut in entry['mutations']:
            ch_rs_ic = (mut['chain'], mut['resseq'], mut['icode'])
            if ch_rs_ic not in seq_map: continue
            aa_mut[seq_map[ch_rs_ic]] = one_to_index(mut['mt'])
        data['aa_mut'] = aa_mut
        data['mut_flag'] = (data['aa'] != data['aa_mut'])

        if self.transform is not None:
            data = self.transform(data)

        return data


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--csv_path', type=str, default='./data/SKEMPI_v2/skempi_v2.csv')
    parser.add_argument('--pdb_dir', type=str, default='./data/SKEMPI_v2/PDBs')
    parser.add_argument('--cache_dir', type=str, default='./data/SKEMPI_v2_cache_custom_folds')
    parser.add_argument('--folds_dir', type=str, default='./cross_validation_folds_final')
    parser.add_argument('--threshold', type=int, default=60)
    parser.add_argument('--reset', action='store_true', default=False)
    args = parser.parse_args()

    dataset = SkempiCustomFoldsDataset(
        csv_path = args.csv_path,
        pdb_dir = args.pdb_dir,
        cache_dir = args.cache_dir,
        folds_dir = args.folds_dir,
        threshold = args.threshold,
        split='train',
        cvfold_index=0,
        reset=args.reset
    )
    print(f"Dataset size: {len(dataset)}")
    print(f"First entry: {dataset[0]}") 