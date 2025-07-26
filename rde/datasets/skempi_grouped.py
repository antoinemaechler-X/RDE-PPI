import os
import copy
import random
import pickle
import math
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


class SkempiGroupedDataset(Dataset):

    def __init__(
        self, 
        csv_path, 
        pdb_dir, 
        cache_dir,
        grouped_csv_path,
        cvfold_index=0, 
        num_cvfolds=10, 
        split='train', 
        split_seed=42,
        transform=None, 
        blocklist=frozenset({'1KBH'}), 
        reset=False
    ):
        super().__init__()
        self.csv_path = csv_path
        self.pdb_dir = pdb_dir
        self.cache_dir = cache_dir
        self.grouped_csv_path = grouped_csv_path
        os.makedirs(cache_dir, exist_ok=True)
        self.blocklist = blocklist
        self.transform = transform
        self.cvfold_index = cvfold_index
        self.num_cvfolds = num_cvfolds
        assert split in ('train', 'val')
        self.split = split
        self.split_seed = split_seed

        self.entries_cache = os.path.join(cache_dir, 'entries_grouped.pkl')
        self.entries = None
        self.entries_full = None
        self._load_entries(reset)

        # Load structures after entries are finalized
        self.structures_cache = os.path.join(cache_dir, 'structures_grouped.pkl')
        self.structures = None
        self._load_structures(reset)

    def _load_entries(self, reset):
        if not os.path.exists(self.entries_cache) or reset:
            self.entries_full = self._preprocess_entries()
        else:
            with open(self.entries_cache, 'rb') as f:
                self.entries_full = pickle.load(f)

        # Create mapping from complex names to group IDs
        pdb_to_group = create_complex_to_group_mapping(self.grouped_csv_path)
        
        # Group entries by complex group
        group_to_entries = {}
        missing_complexes = []
        for e in self.entries_full:
            pdbcode = e['pdbcode']
            if pdbcode not in pdb_to_group:
                # If complex not in grouping, assign to a unique group (like your code)
                group_id = f"ungrouped_{pdbcode}"
                missing_complexes.append(pdbcode)
            else:
                group_id = pdb_to_group[pdbcode]
            
            if group_id not in group_to_entries:
                group_to_entries[group_id] = []
            group_to_entries[group_id].append(e)

        # Create list of group IDs and use sklearn KFold for consistent splitting
        from sklearn.model_selection import KFold
        # Convert all group IDs to strings to avoid type comparison issues
        group_list = sorted([str(g) for g in group_to_entries.keys()])
        
        # Use KFold with same random_state as your code
        kf = KFold(n_splits=self.num_cvfolds, shuffle=True, random_state=42)
        group_splits = list(kf.split(group_list))
        
        # Get train and val groups for this fold
        train_group_indices, val_group_indices = group_splits[self.cvfold_index]
        train_split = [group_list[i] for i in train_group_indices]
        val_split = [group_list[i] for i in val_group_indices]
        
        # Convert back to original types for group_to_entries lookup
        train_split = [int(g) if g.isdigit() else g for g in train_split]
        val_split = [int(g) if g.isdigit() else g for g in val_split]
        
        if self.split == 'val':
            groups_this = val_split
        else:
            groups_this = train_split

        # Collect all entries from the selected groups
        entries = []
        for group_id in groups_this:
            entries += group_to_entries[group_id]
        self.entries = entries
        
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
    parser.add_argument('--cache_dir', type=str, default='./data/SKEMPI_v2_cache_grouped')
    parser.add_argument('--grouped_csv_path', type=str, default='./data/complex_sequences_grouped_60.csv')
    parser.add_argument('--reset', action='store_true', default=False)
    args = parser.parse_args()

    dataset = SkempiGroupedDataset(
        csv_path = args.csv_path,
        pdb_dir = args.pdb_dir,
        cache_dir = args.cache_dir,
        grouped_csv_path = args.grouped_csv_path,
        split = 'val',
        num_cvfolds=10,
        cvfold_index=2,
        reset=args.reset,
    )
    print(dataset[0])
    print(len(dataset)) 