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


def load_skempi_entries(csv_path, wildtype_dir, optimized_dir, block_list={'1KBH'}):
    df = pd.read_csv(csv_path, sep=',')
    
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
        complex_id = row['#Pdb']  # This is the unique ID like "0_1CSE"
        pdbcode = row['#Pdb_origin']  # This is the PDB code like "1CSE"
        partner1, partner2 = row['Partner1'], row['Partner2']
        
        if pdbcode in block_list:
            continue
            
        mut_str = row['Mutation(s)_cleaned']
        muts = list(map(_parse_mut, row['Mutation(s)_cleaned'].split(',')))
        
        # Determine ligand and receptor based on mutation chain
        if muts[0]['chain'] in partner1:
            group_ligand, group_receptor = partner1, partner2
        else:
            group_ligand, group_receptor = partner2, partner1

        # Check if both wildtype and optimized structures exist
        wildtype_file = find_pdb_file(wildtype_dir, complex_id)
        optimized_file = find_pdb_file(optimized_dir, complex_id)
        
        if wildtype_file is None or optimized_file is None:
            continue

        if not np.isfinite(row['ddG']):
            continue

        entry = {
            'id': i,
            'complex': pdbcode,  # Use the PDB code for grouping (like custom folds)
            'complex_id': complex_id,  # Keep the unique complex ID for structure loading
            'complex_origin': pdbcode,  # Keep the original PDB code for reference
            'mutstr': mut_str,
            'num_muts': len(muts),
            'pdbcode': pdbcode,  # Keep for backward compatibility
            'group_ligand': list(group_ligand),
            'group_receptor': list(group_receptor),
            'mutations': muts,
            'ddG': np.float32(row['ddG']),
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
        train_complexes: Set of complex IDs for training
        test_complexes: Set of complex IDs for testing
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
    
    # Return the full complex IDs (no need to extract PDB codes anymore)
    return train_complex_ids, test_complex_ids


def find_pdb_file(pdb_dir, complex_id):
    """
    Find a PDB file in the directory that matches the given complex ID.
    Returns the full filename if found, None otherwise.
    """
    if not os.path.exists(pdb_dir):
        return None
    
    # Look for exact match first
    exact_filename = f"{complex_id}.pdb"
    exact_path = os.path.join(pdb_dir, exact_filename)
    if os.path.exists(exact_path):
        return exact_filename
    
    # Fallback: look for files containing the complex ID
    for filename in os.listdir(pdb_dir):
        if filename.endswith('.pdb') and complex_id in filename:
            return filename
    return None


class SkempiSeparateStructuresDataset(Dataset):

    def __init__(
        self, 
        csv_path, 
        wildtype_dir,
        optimized_dir,
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
        self.wildtype_dir = wildtype_dir
        self.optimized_dir = optimized_dir
        self.cache_dir = cache_dir
        self.folds_dir = folds_dir
        self.threshold = threshold
        os.makedirs(cache_dir, exist_ok=True)
        self.blocklist = blocklist
        self.transform = transform
        self.cvfold_index = cvfold_index
        assert split in ('train', 'val')
        self.split = split

        self.entries_cache = os.path.join(cache_dir, f'entries_separate_structures_{threshold}.pkl')
        self.entries = None
        self.entries_full = None
        self._load_entries(reset)

        # Load structures after entries are finalized
        self.wildtype_structures_cache = os.path.join(cache_dir, f'wildtype_structures_{threshold}.pkl')
        self.optimized_structures_cache = os.path.join(cache_dir, f'optimized_structures_{threshold}.pkl')
        self.wildtype_structures = None
        self.optimized_structures = None
        self._load_structures(reset)

    def _load_entries(self, reset):
        # Force reset to ensure we have the new structure with complex_id field
        if not os.path.exists(self.entries_cache) or reset or True:  # Always regenerate for now
            self.entries_full = self._preprocess_entries()
        else:
            with open(self.entries_cache, 'rb') as f:
                self.entries_full = pickle.load(f)

        # Load custom fold splits
        train_complex_ids, test_complex_ids = load_custom_fold_splits(
            self.folds_dir, self.threshold, self.cvfold_index
        )
        
        # Extract PDB codes from complex IDs (like custom folds approach)
        def extract_pdb_code(complex_id):
            parts = complex_id.split('_')
            if len(parts) >= 2:
                return parts[-1]  # Take the last part as PDB code
            else:
                return complex_id  # If no underscore, return as is
        
        train_pdb_codes = set(extract_pdb_code(cid) for cid in train_complex_ids)
        test_pdb_codes = set(extract_pdb_code(cid) for cid in test_complex_ids)
        
        # Filter entries based on split using PDB codes
        if self.split == 'val':
            target_pdb_codes = test_pdb_codes
        else:
            target_pdb_codes = train_pdb_codes

        # Filter entries to only include complexes whose PDB code is in the target split
        self.entries = [
            entry for entry in self.entries_full 
            if entry['complex'] in target_pdb_codes
        ]
        
    def _preprocess_entries(self):
        entries = load_skempi_entries(self.csv_path, self.wildtype_dir, self.optimized_dir, self.blocklist)
        with open(self.entries_cache, 'wb') as f:
            pickle.dump(entries, f)
        return entries

    def _load_structures(self, reset):
        # Load wildtype structures
        if not os.path.exists(self.wildtype_structures_cache) or reset:
            self.wildtype_structures = self._preprocess_wildtype_structures()
        else:
            with open(self.wildtype_structures_cache, 'rb') as f:
                self.wildtype_structures = pickle.load(f)
            
            # Check if we have all the complex IDs we need
            needed_complex_ids = set([e['complex_id'] for e in self.entries])
            available_complex_ids = set(self.wildtype_structures.keys())
            missing_complex_ids = needed_complex_ids - available_complex_ids
            
            if missing_complex_ids:
                print(f"Missing wildtype structures for complex IDs: {missing_complex_ids}")
                # Load missing structures
                for complex_id in tqdm(missing_complex_ids, desc='Loading missing wildtype structures'):
                    filename = find_pdb_file(self.wildtype_dir, complex_id)
                    if filename:
                        parser = PDBParser(QUIET=True)
                        pdb_path = os.path.join(self.wildtype_dir, filename)
                        model = parser.get_structure(None, pdb_path)[0]
                        data, seq_map = parse_biopython_structure(model)
                        self.wildtype_structures[complex_id] = (data, seq_map)
                
                # Save updated structures cache
                with open(self.wildtype_structures_cache, 'wb') as f:
                    pickle.dump(self.wildtype_structures, f)

        # Load optimized structures
        if not os.path.exists(self.optimized_structures_cache) or reset:
            self.optimized_structures = self._preprocess_optimized_structures()
        else:
            with open(self.optimized_structures_cache, 'rb') as f:
                self.optimized_structures = pickle.load(f)
            
            # Check if we have all the complex IDs we need
            needed_complex_ids = set([e['complex_id'] for e in self.entries])
            available_complex_ids = set(self.optimized_structures.keys())
            missing_complex_ids = needed_complex_ids - available_complex_ids
            
            if missing_complex_ids:
                print(f"Missing optimized structures for complex IDs: {missing_complex_ids}")
                # Load missing structures
                for complex_id in tqdm(missing_complex_ids, desc='Loading missing optimized structures'):
                    filename = find_pdb_file(self.optimized_dir, complex_id)
                    if filename:
                        parser = PDBParser(QUIET=True)
                        pdb_path = os.path.join(self.optimized_dir, filename)
                        model = parser.get_structure(None, pdb_path)[0]
                        data, seq_map = parse_biopython_structure(model)
                        self.optimized_structures[complex_id] = (data, seq_map)
                
                # Save updated structures cache
                with open(self.optimized_structures_cache, 'wb') as f:
                    pickle.dump(self.optimized_structures, f)

    def _preprocess_wildtype_structures(self):
        structures = {}
        # Get all complex IDs that are actually used in the final dataset
        complex_ids = list(set([e['complex_id'] for e in self.entries]))
        for complex_id in tqdm(complex_ids, desc='Wildtype Structures'):
            filename = find_pdb_file(self.wildtype_dir, complex_id)
            if filename:
                parser = PDBParser(QUIET=True)
                pdb_path = os.path.join(self.wildtype_dir, filename)
                model = parser.get_structure(None, pdb_path)[0]
                data, seq_map = parse_biopython_structure(model)
                structures[complex_id] = (data, seq_map)
            else:
                print(f"Warning: No wildtype structure found for {complex_id}")
        with open(self.wildtype_structures_cache, 'wb') as f:
            pickle.dump(structures, f)
        return structures

    def _preprocess_optimized_structures(self):
        structures = {}
        # Get all complex IDs that are actually used in the final dataset
        complex_ids = list(set([e['complex_id'] for e in self.entries]))
        for complex_id in tqdm(complex_ids, desc='Optimized Structures'):
            filename = find_pdb_file(self.optimized_dir, complex_id)
            if filename:
                parser = PDBParser(QUIET=True)
                pdb_path = os.path.join(self.optimized_dir, filename)
                model = parser.get_structure(None, pdb_path)[0]
                data, seq_map = parse_biopython_structure(model)
                structures[complex_id] = (data, seq_map)
            else:
                print(f"Warning: No optimized structure found for {complex_id}")
        with open(self.optimized_structures_cache, 'wb') as f:
            pickle.dump(structures, f)
        return structures

    def __len__(self):
        return len(self.entries)

    def __getitem__(self, index):
        entry = self.entries[index]
        
        # Load wildtype structure
        if entry['complex_id'] not in self.wildtype_structures:
            raise ValueError(f"Wildtype structure not found for {entry['complex_id']}")
        data_wt, seq_map_wt = copy.deepcopy(self.wildtype_structures[entry['complex_id']])
        
        # Load optimized structure
        if entry['complex_id'] not in self.optimized_structures:
            raise ValueError(f"Optimized structure not found for {entry['complex_id']}")
        data_mt, seq_map_mt = copy.deepcopy(self.optimized_structures[entry['complex_id']])
        
        # Add entry metadata to both structures
        keys = {'id', 'complex', 'mutstr', 'num_muts', 'pdbcode', 'ddG'}
        for k in keys:
            data_wt[k] = entry[k]
            data_mt[k] = entry[k]

        # Add group information to wildtype structure
        group_id_wt = []
        for ch in data_wt['chain_id']:
            if ch in entry['group_ligand']:
                group_id_wt.append(1)
            elif ch in entry['group_receptor']:
                group_id_wt.append(2)
            else:
                group_id_wt.append(0)
        data_wt['group_id'] = torch.LongTensor(group_id_wt)

        # Add group information to optimized structure
        group_id_mt = []
        for ch in data_mt['chain_id']:
            if ch in entry['group_ligand']:
                group_id_mt.append(1)
            elif ch in entry['group_receptor']:
                group_id_mt.append(2)
            else:
                group_id_mt.append(0)
        data_mt['group_id'] = torch.LongTensor(group_id_mt)

        # Create mutation flags for both structures - separate structures approach
        # For wildtype: no mutations (this is the original structure)
        data_wt['mut_flag'] = torch.zeros_like(data_wt['aa'], dtype=torch.bool)
        data_wt['aa_mut'] = data_wt['aa'].clone()
        
        # For optimized: mark mutations based on the entry
        # The optimized structure already has the mutations applied, so we need to:
        # 1. Set aa_mut to the wildtype amino acids (for reference)
        # 2. Set mut_flag to True at mutation sites
        data_mt['mut_flag'] = torch.zeros_like(data_mt['aa'], dtype=torch.bool)
        data_mt['aa_mut'] = data_mt['aa'].clone()  # Start with current amino acids
        
        # Set aa_mut to wildtype amino acids at mutation sites (for reference)
        for mut in entry['mutations']:
            ch_rs_ic = (mut['chain'], mut['resseq'], mut['icode'])
            if ch_rs_ic in seq_map_mt:
                seq_idx = seq_map_mt[ch_rs_ic]
                # Set aa_mut to the wildtype amino acid (for reference)
                data_mt['aa_mut'][seq_idx] = one_to_index(mut['wt'])
                # Mark this position as mutated
                data_mt['mut_flag'][seq_idx] = True

        # Apply transforms if specified
        if self.transform is not None:
            data_wt = self.transform(data_wt)
            data_mt = self.transform(data_mt)

        # Return both structures
        return {
            'wildtype': data_wt,
            'mutant': data_mt
        }


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--csv_path', type=str, default='./data/SKEMPI_v2/skempi_v2.csv')
    parser.add_argument('--wildtype_dir', type=str, default='./data/wildtype')
    parser.add_argument('--optimized_dir', type=str, default='./data/optimized')
    parser.add_argument('--cache_dir', type=str, default='./data/SKEMPI_v2_cache_separate_structures')
    parser.add_argument('--folds_dir', type=str, default='./cross_validation_folds_final')
    parser.add_argument('--threshold', type=int, default=60)
    parser.add_argument('--reset', action='store_true', default=False)
    args = parser.parse_args()

    dataset = SkempiSeparateStructuresDataset(
        csv_path = args.csv_path,
        wildtype_dir = args.wildtype_dir,
        optimized_dir = args.optimized_dir,
        cache_dir = args.cache_dir,
        folds_dir = args.folds_dir,
        threshold = args.threshold,
        split='train',
        cvfold_index=0,
        reset=args.reset
    )
    print(f"Dataset size: {len(dataset)}")
    print(f"First entry: {dataset[0]}") 