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


def load_skempi_entries(csv_path, wildtype_dir, block_list={'1KBH'}):
    """Load SKEMPI entries from the filtered CSV file."""
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

        # Check if wildtype structure exists
        wildtype_file = find_pdb_file(wildtype_dir, complex_id)
        
        if wildtype_file is None:
            continue

        if not np.isfinite(row['ddG']):
            continue

        entry = {
            'id': i,
            'complex': pdbcode,  # Use the PDB code for grouping
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
    
    return train_complex_ids, test_complex_ids


def find_pdb_file(pdb_dir, complex_id):
    """Find PDB file for a given complex ID."""
    # Try different possible file extensions and naming conventions
    possible_files = [
        os.path.join(pdb_dir, f"{complex_id}.pdb"),
        os.path.join(pdb_dir, f"{complex_id}.PDB"),
        os.path.join(pdb_dir, f"{complex_id.upper()}.pdb"),
        os.path.join(pdb_dir, f"{complex_id.upper()}.PDB"),
    ]
    
    for file_path in possible_files:
        if os.path.exists(file_path):
            return file_path
    
    return None


class SkempiWildtypeOnlyDataset(Dataset):

    def __init__(
        self, 
        csv_path, 
        wildtype_dir,
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
        self.cache_dir = cache_dir
        self.folds_dir = folds_dir
        self.threshold = threshold
        os.makedirs(cache_dir, exist_ok=True)
        self.blocklist = blocklist
        self.transform = transform
        self.cvfold_index = cvfold_index
        assert split in ('train', 'val')
        self.split = split

        self.entries_cache = os.path.join(cache_dir, 'entries_wildtype_only.pkl')
        self.entries = None
        self.entries_full = None
        self._load_entries(reset)

        # Load structures after entries are finalized
        self.structures_cache = os.path.join(cache_dir, 'structures_wildtype_only.pkl')
        self.structures = None
        self._load_structures(reset)

    def _load_entries(self, reset):
        if not os.path.exists(self.entries_cache) or reset:
            self.entries_full = self._preprocess_entries()
        else:
            with open(self.entries_cache, 'rb') as f:
                self.entries_full = pickle.load(f)

        # Load fold splits
        train_complex_ids, test_complex_ids = load_custom_fold_splits(
            self.folds_dir, self.threshold, self.cvfold_index
        )
        
        # Filter entries based on split
        if self.split == 'val':
            target_complex_ids = test_complex_ids
        else:
            target_complex_ids = train_complex_ids
        
        # Filter entries to only include those in the target split
        self.entries = [
            entry for entry in self.entries_full 
            if entry['complex_id'] in target_complex_ids
        ]

    def _preprocess_entries(self):
        entries = load_skempi_entries(self.csv_path, self.wildtype_dir, self.blocklist)
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
            
            # Check if we have all the complex IDs we need
            needed_complex_ids = set([e['complex_id'] for e in self.entries])
            available_complex_ids = set(self.structures.keys())
            missing_complex_ids = needed_complex_ids - available_complex_ids
            
            if missing_complex_ids:
                print(f"Missing structures for complex IDs: {missing_complex_ids}")
                # Load missing structures
                for complex_id in tqdm(missing_complex_ids, desc='Loading missing structures'):
                    pdb_path = find_pdb_file(self.wildtype_dir, complex_id)
                    if pdb_path is None:
                        print(f"Warning: No wildtype structure found for {complex_id}")
                        continue
                    
                    parser = PDBParser(QUIET=True)
                    model = parser.get_structure(None, pdb_path)[0]
                    data, seq_map = parse_biopython_structure(model)
                    self.structures[complex_id] = (data, seq_map)
                
                # Save updated structures cache
                with open(self.structures_cache, 'wb') as f:
                    pickle.dump(self.structures, f)

    def _preprocess_structures(self):
        structures = {}
        # Get all complex IDs that are actually used in the final dataset
        complex_ids = list(set([e['complex_id'] for e in self.entries]))
        for complex_id in tqdm(complex_ids, desc='Structures'):
            pdb_path = find_pdb_file(self.wildtype_dir, complex_id)
            if pdb_path is None:
                print(f"Warning: No wildtype structure found for {complex_id}")
                continue
                
            parser = PDBParser(QUIET=True)
            model = parser.get_structure(None, pdb_path)[0]
            data, seq_map = parse_biopython_structure(model)
            structures[complex_id] = (data, seq_map)
        with open(self.structures_cache, 'wb') as f:
            pickle.dump(structures, f)
        return structures

    def __len__(self):
        return len(self.entries)

    def __getitem__(self, index):
        entry = self.entries[index]
        
        # Load wildtype structure
        if entry['complex_id'] not in self.structures:
            raise ValueError(f"Wildtype structure not found for {entry['complex_id']}")
        data, seq_map = copy.deepcopy(self.structures[entry['complex_id']])
        
        # Add entry metadata
        keys = {'id', 'complex', 'mutstr', 'num_muts', 'pdbcode', 'ddG'}
        for k in keys:
            data[k] = entry[k]

        # Add group information
        group_id = []
        for ch in data['chain_id']:
            if ch in entry['group_ligand']:
                group_id.append(1)
            elif ch in entry['group_receptor']:
                group_id.append(2)
            else:
                group_id.append(0)
        data['group_id'] = torch.LongTensor(group_id)

        # Create mutation flags - original masking approach
        # Start with wildtype amino acids
        data['aa_mut'] = data['aa'].clone()
        data['mut_flag'] = torch.zeros_like(data['aa'], dtype=torch.bool)
        
        # Apply mutations by changing amino acid codes at mutation sites
        for mut in entry['mutations']:
            ch_rs_ic = (mut['chain'], mut['resseq'], mut['icode'])
            if ch_rs_ic in seq_map:
                seq_idx = seq_map[ch_rs_ic]
                # Set the mutated amino acid
                data['aa_mut'][seq_idx] = one_to_index(mut['mt'])
                # Mark this position as mutated
                data['mut_flag'][seq_idx] = True

        # Apply transforms if specified
        if self.transform is not None:
            data = self.transform(data)

        return data


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--csv_path', type=str, default='./data/SKEMPI_filtered.csv')
    parser.add_argument('--wildtype_dir', type=str, default='./data/wildtype')
    parser.add_argument('--cache_dir', type=str, default='./data/SKEMPI_v2_cache_wildtype_only')
    parser.add_argument('--folds_dir', type=str, default='./cross_validation_folds_final')
    parser.add_argument('--threshold', type=int, default=60)
    parser.add_argument('--reset', action='store_true', default=False)
    args = parser.parse_args()

    dataset = SkempiWildtypeOnlyDataset(
        csv_path = args.csv_path,
        wildtype_dir = args.wildtype_dir,
        cache_dir = args.cache_dir,
        folds_dir = args.folds_dir,
        threshold = args.threshold,
        split='train',
        cvfold_index=0,
        reset=args.reset
    )
    print(f"Dataset size: {len(dataset)}")
    print(f"First entry: {dataset[0]}") 