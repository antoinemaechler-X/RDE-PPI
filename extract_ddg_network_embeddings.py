import os
import argparse
import numpy as np
import torch
import pandas as pd
from tqdm import tqdm
from rde.utils.misc import load_config
import pickle
import copy

from rde.models.rde_ddg import DDG_RDE_Network
from rde.datasets.skempi_wildtype_only import SkempiWildtypeOnlyDataset
from rde.utils.transforms import get_transform


def load_expected_complex_ids(folds_dir, threshold, fold, split):
    """Load the expected complex IDs from the txt files to ensure exact matching."""
    # Construct the correct path including threshold percentage
    threshold_dir = f"{threshold}_percent"
    txt_file = os.path.join(folds_dir, threshold_dir, f"fold_{fold}", f"{split}_complex_ids.txt")
    if not os.path.exists(txt_file):
        raise FileNotFoundError(f"Expected complex IDs file not found: {txt_file}")
    
    with open(txt_file, 'r') as f:
        complex_ids = [line.strip() for line in f if line.strip()]
    
    print(f"Loaded {len(complex_ids)} expected complex IDs from {txt_file}")
    return complex_ids


def load_mutation_data(csv_path, complex_id):
    """Load mutation data for a specific complex ID from the CSV file."""
    # Read the CSV file
    df = pd.read_csv(csv_path)
    
    # Find rows where #Pdb column matches the complex_id
    matching_rows = df[df['#Pdb'] == complex_id]
    
    if len(matching_rows) == 0:
        raise ValueError(f"No mutation data found for complex {complex_id} in {csv_path}")
    
    # For now, take the first mutation (we can extend this later if needed)
    mutation_row = matching_rows.iloc[0]
    
    # Extract mutation information
    mutation = mutation_row['Mutation(s)_cleaned']
    ddg = mutation_row['ddG']
    
    print(f"Found mutation data for {complex_id}: {mutation} (ddG: {ddg})")
    return mutation, ddg


def process_single_complex(
    model, 
    complex_id, 
    pdb_file_path, 
    mutation_data, 
    device, 
    output_dir,
    transform
):
    """Process a single complex: load PDB, apply mutation, compute embeddings, save files."""
    
    print(f"\n{'='*60}")
    print(f"PROCESSING COMPLEX: {complex_id}")
    print(f"{'='*60}")
    
    # Check if PDB file exists
    if not os.path.exists(pdb_file_path):
        raise FileNotFoundError(f"PDB file not found: {pdb_file_path}")
    
    # Parse mutation data
    mutation_str, ddg = mutation_data
    print(f"Mutation string: '{mutation_str}'")
    print(f"ddG value: {ddg}")
    
    # Parse the mutation string (e.g., "ME80L" -> wt="M", chain="E", resseq=80, mt="L")
    def _parse_mut(mut_name):
        wt_type, mutchain, mutseq, mt_type = mut_name[0], mut_name[1], int(mut_name[2:-1]), mut_name[-1]
        return {
            'wt': wt_type,
            'mt': mt_type,
            'chain': mutchain,
            'resseq': mutseq,
            'icode': ' ',
            'name': mut_name
        }
    
    mutations = list(map(_parse_mut, mutation_str.split(',')))
    print(f"Parsed mutations: {mutations}")
    
    # Load PDB structure using BioPython
    from Bio.PDB.PDBParser import PDBParser
    from rde.utils.protein.parsers import parse_biopython_structure
    from Bio.PDB.Polypeptide import one_to_index
    
    parser = PDBParser(QUIET=True)
    structure = parser.get_structure(None, pdb_file_path)
    data, seq_map = parse_biopython_structure(structure[0])
    
    print(f"Loaded PDB structure:")
    print(f"  - Total residues: {len(data['aa'])}")
    print(f"  - Sequence map keys: {list(seq_map.keys())[:10]}... (showing first 10)")
    print(f"  - Available data keys: {list(data.keys())}")
    
    # Add missing fields that the model expects (same as in the dataset)
    # Add entry metadata (dummy values since we don't have the full entry)
    data['id'] = complex_id
    data['complex'] = complex_id.split('_')[1] if '_' in complex_id else complex_id
    data['mutstr'] = mutation_str
    data['num_muts'] = len(mutations)
    data['pdbcode'] = complex_id.split('_')[1] if '_' in complex_id else complex_id
    data['ddG'] = ddg
    
    # Add group_id field (default to 0 for all residues)
    data['group_id'] = torch.zeros(len(data['aa']), dtype=torch.long)
    
    # Create mutation flags - start with wildtype amino acids
    data['aa_mut'] = data['aa'].clone()
    data['mut_flag'] = torch.zeros_like(data['aa'], dtype=torch.bool)
    
    print(f"Initial data setup:")
    print(f"  - aa shape: {data['aa'].shape}")
    print(f"  - mut_flag shape: {data['mut_flag'].shape}")
    print(f"  - mut_flag sum: {data['mut_flag'].sum()}")
    
    # Create a copy for mutant
    data_mut = copy.deepcopy(data)
    
    # Apply mutations to create mutant data
    aa_mut = data['aa'].clone()
    mut_flag = torch.zeros_like(data['aa'], dtype=torch.bool)
    
    print(f"\nApplying mutations:")
    for i, mut in enumerate(mutations):
        ch_rs_ic = (mut['chain'], mut['resseq'], mut['icode'])
        print(f"  Mutation {i+1}: {mut['name']} (chain={mut['chain']}, resseq={mut['resseq']}, icode='{mut['icode']}')")
        print(f"    - Looking for key: {ch_rs_ic}")
        print(f"    - Available keys in seq_map: {list(seq_map.keys())[:20]}... (showing first 20)")
        
        if ch_rs_ic in seq_map:
            seq_idx = seq_map[ch_rs_ic]
            print(f"    - ✓ Found in seq_map at index: {seq_idx}")
            print(f"    - Original amino acid: {data['aa'][seq_idx]}")
            print(f"    - Mutating to: {one_to_index(mut['mt'])}")
            
            # Set the mutated amino acid
            aa_mut[seq_idx] = one_to_index(mut['mt'])
            # Mark this position as mutated
            mut_flag[seq_idx] = True
            print(f"    - ✓ Mutation applied successfully")
        else:
            print(f"    - ✗ NOT FOUND in seq_map!")
            print(f"    - This is a critical error - mutation position not found")
            raise ValueError(f"Mutation position {ch_rs_ic} not found in sequence map for {complex_id}")
    
    data_mut['aa'] = aa_mut
    data_mut['mut_flag'] = mut_flag
    data_mut['aa_mut'] = aa_mut
    
    print(f"\nMutation application summary:")
    print(f"  - Total mutations applied: {mut_flag.sum()}")
    print(f"  - Mutation positions: {torch.where(mut_flag)[0].tolist()}")
    print(f"  - mut_flag content: {mut_flag}")
    
    # Ensure mutant data has the same metadata fields
    data_mut['id'] = data['id']
    data_mut['complex'] = data['complex']
    data_mut['mutstr'] = data['mutstr']
    data_mut['num_muts'] = data['num_muts']
    data_mut['pdbcode'] = data['pdbcode']
    data_mut['ddG'] = data['ddG']
    
    # CRITICAL FIX: Deterministically select the same residue patch for WT and MT using the patch
    if transform is not None:
        print(f"\n{'='*40}")
        print(f"TRANSFORM APPLICATION (DETERMINISTIC, NO FALLBACKS)")
        print(f"{'='*40}")

        # Keep a pristine copy of WT and MT before any selection
        original_wt_data = copy.deepcopy(data)
        original_mt_data = copy.deepcopy(data_mut)

        # Find a patch transform in the pipeline and split transforms into pre/post relative to it
        patch_cfg = None
        pre_transforms = []
        post_transforms = []
        if hasattr(transform, 'transforms') and len(transform.transforms) > 0:
            found = False
            for t in transform.transforms:
                if not found and hasattr(t, 'select_attr') and hasattr(t, 'patch_size'):
                    patch_cfg = {
                        'select_attr': getattr(t, 'select_attr'),
                        'patch_size': int(getattr(t, 'patch_size')),
                        'type': type(t).__name__
                    }
                    found = True
                    continue
                if not found:
                    pre_transforms.append(t)
                else:
                    post_transforms.append(t)
        if patch_cfg is None:
            raise RuntimeError("No patch transform with 'select_attr' and 'patch_size' found in pipeline. Abort.")

        print(f"Patch transform detected: type={patch_cfg['type']}, select_attr={patch_cfg['select_attr']}, patch_size={patch_cfg['patch_size']}")

        # 1) Apply pre-patch transforms to WT and MT to mirror training order (e.g., select_atom)
        for t in pre_transforms:
            t_name = type(t).__name__
            print(f"Applying pre-patch transform on WT/MT: {t_name}")
            data = t(data)
            data_mut = t(data_mut)

        # Validate mutation flags after pre-transforms
        if 'mut_flag' not in data_mut:
            raise RuntimeError("mut_flag missing in mutant data after pre-transforms. Abort.")
        if data_mut['mut_flag'].sum() == 0:
            raise RuntimeError("Mutation flag has zero active positions after pre-transforms. Abort.")
        mut_positions = torch.where(data_mut['mut_flag'])[0]
        print(f"Mutation positions after pre-transforms (indices in current space): {mut_positions.tolist()}")

        # Build position tensors for distance computation using current (pre-transformed) data
        if 'pos_atoms' in data and 'mask_atoms' in data:
            pos_tensor = data['pos_atoms']
            mask_tensor = data['mask_atoms']
            print("Using pos_atoms/mask_atoms for CB positions (post pre-transforms)")
        elif 'pos_heavyatom' in data and 'mask_heavyatom' in data:
            pos_tensor = data['pos_heavyatom']
            mask_tensor = data['mask_heavyatom']
            print("Using pos_heavyatom/mask_heavyatom for CB positions (post pre-transforms)")
        else:
            raise RuntimeError("Position tensors not found after pre-transforms. Abort.")

        from rde.utils.transforms._base import _get_CB_positions, _index_select_data
        pos_CB = _get_CB_positions(pos_tensor, mask_tensor)
        if pos_CB.dim() != 2 or pos_CB.size(0) != data['aa'].size(0):
            raise RuntimeError("CB position tensor has unexpected shape after pre-transforms. Abort.")

        # 2) Compute patch indices around mutation sites using current space
        patch_size = patch_cfg['patch_size']
        pos_mut = pos_CB[mut_positions]
        dist_from_mut = torch.cdist(pos_CB, pos_mut).min(dim=1)[0]
        patch_idx = torch.argsort(dist_from_mut)[:patch_size]
        patch_idx = patch_idx.sort()[0]
        print(f"Selected patch size: {patch_idx.numel()} (requested {patch_size})")

        # Ensure all mutation sites are included in the patch
        included_mask = torch.isin(mut_positions, patch_idx)
        if not bool(included_mask.all()):
            missing = mut_positions[~included_mask].tolist()
            raise RuntimeError(f"Patch selection does not include all mutation sites (after pre-transforms). Missing indices: {missing}. Abort.")

        # 3) Apply identical selection to WT and MT
        data = _index_select_data(data, patch_idx)
        data_mut = _index_select_data(data_mut, patch_idx)
        print(f"WT/MT patch after selection: {len(data['aa'])} residues (first 10 idx = {patch_idx[:10].tolist()})")

        # 4) Apply post-patch transforms (if any) to match training order
        for t in post_transforms:
            t_name = type(t).__name__
            print(f"Applying post-patch transform on WT/MT: {t_name}")
            data = t(data)
            data_mut = t(data_mut)

        # Final sanity check: WT and MT must remain aligned
        if len(data['aa']) != len(data_mut['aa']):
            raise RuntimeError(f"Transforms changed WT/MT lengths differently: WT={len(data['aa'])}, MT={len(data_mut['aa'])}. Abort.")
    else:
        raise RuntimeError("Transform pipeline is required for patch selection but was None. Abort.")
    
    # Prepare batches for model
    print(f"\nPreparing model batches:")
    print(f"  - Wildtype data shape: {len(data['aa'])} residues")
    print(f"  - Mutant data shape: {len(data_mut['aa'])} residues")
    
    batch_wt = {k: (v.unsqueeze(0).to(device) if isinstance(v, torch.Tensor) else v) for k, v in data.items()}
    batch_mt = {k: (v.unsqueeze(0).to(device) if isinstance(v, torch.Tensor) else v) for k, v in data_mut.items()}
    
    # Extract per-residue embeddings before maxpool/MLP
    print(f"Extracting embeddings...")
    emb_wt = model.encode(batch_wt).squeeze(0).cpu().numpy()  # (L, D)
    emb_mt = model.encode(batch_mt).squeeze(0).cpu().numpy()  # (L, D)
    
    print(f"  - Wildtype embedding shape: {emb_wt.shape}")
    print(f"  - Mutant embedding shape: {emb_mt.shape}")
    
    # Extract residue mapping from the PROCESSED data (after transforms)
    # This ensures we map to the exact residues that the model processed
    chain_ids = data['chain_id']
    resseqs = data['resseq']
    icodes = data['icode']
    
    # Convert to lists if they're tensors
    if hasattr(chain_ids, 'tolist'): 
        chain_ids = chain_ids.tolist()
    if hasattr(resseqs, 'tolist'): 
        resseqs = resseqs.tolist()
    if hasattr(icodes, 'tolist'): 
        icodes = icodes.tolist()
    
    # Create residue mapping for the processed patch
    residue_ids = list(zip(chain_ids, resseqs, icodes))
    
    print(f"Residue mapping:")
    print(f"  - Total residues in mapping: {len(residue_ids)}")
    print(f"  - First 10 residues: {residue_ids[:10]}")
    print(f"  - Last 10 residues: {residue_ids[-10:]}")
    print(f"  - Should match embedding dimension: {emb_wt.shape[0]}")
    
    # Save with the exact complex_id from the txt file
    base_path = os.path.join(output_dir, complex_id)
    
    # Save wildtype
    np.save(base_path + "_wt.npy", emb_wt)
    with open(base_path + "_wt_resmap.pkl", 'wb') as f:
        pickle.dump(residue_ids, f)
    
    # Save mutant
    np.save(base_path + "_mt.npy", emb_mt)
    with open(base_path + "_mt_resmap.pkl", 'wb') as f:
        pickle.dump(residue_ids, f)
    
    print(f"✓ Successfully processed {complex_id}: saved 4 files")
    print(f"  - {base_path}_wt.npy")
    print(f"  - {base_path}_wt_resmap.pkl")
    print(f"  - {base_path}_mt.npy")
    print(f"  - {base_path}_mt_resmap.pkl")


def extract_embeddings_for_fold(
    model,
    device,
    output_dir,
    expected_complex_ids,  # List of complex IDs from txt files
    wildtype_dir,  # Directory containing PDB files
    csv_path,  # Path to SKEMPI CSV file
    transform,  # Data transform
):
    """Extract embeddings for each complex ID from the txt file."""
    os.makedirs(output_dir, exist_ok=True)
    model.eval()
    
    print(f"Processing {len(expected_complex_ids)} complexes from txt file...")
    
    processed_complexes = 0
    
    with torch.no_grad():
        for complex_id in tqdm(expected_complex_ids, desc=f'Extracting embeddings to {output_dir}'):
            try:
                # Construct PDB file path
                pdb_file_path = os.path.join(wildtype_dir, f"{complex_id}.pdb")
                
                # Load mutation data from CSV
                mutation_data = load_mutation_data(csv_path, complex_id)
                
                # Process the complex
                process_single_complex(
                    model, complex_id, pdb_file_path, mutation_data, 
                    device, output_dir, transform
                )
                
                processed_complexes += 1
                
            except Exception as e:
                error_msg = f"Failed to process complex {complex_id}: {e}"
                print(f"ERROR: {error_msg}")
                raise RuntimeError(f"Stopping entire process due to error: {error_msg}")
    
    # Summary
    print(f"\nProcessing Summary:")
    print(f"  Total complexes in txt file: {len(expected_complex_ids)}")
    print(f"  Successfully processed: {processed_complexes}")
    
    return True  # If we get here, all complexes were processed successfully


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--ckpt', type=str, required=True, help='Path to 10-fold DDG checkpoint (e.g. 30000.pt)')
    parser.add_argument('--config', type=str, required=True, help='Path to config YAML used for training')
    parser.add_argument('--output_dir', type=str, required=True, help='Output directory for embeddings')
    parser.add_argument('--device', type=str, default='cuda')
    args = parser.parse_args()

    config, _ = load_config(args.config)
    device = args.device

    # Load 10-fold checkpoint
    ckpt = torch.load(args.ckpt, map_location='cpu')
    model_states = ckpt['model']['models']  # List of 10 state_dicts

    # Prepare transforms
    transform = get_transform(config.data.get('transform', []))

    # Extract threshold from config or default to 60
    threshold = 60  # Default for 60_percent folds
    
    # CSV path for mutation data
    csv_path = os.path.join(config.data.csv_path) if hasattr(config.data, 'csv_path') else "data/SKEMPI2/SKEMPI_filtered.csv"

    # For each fold
    for fold in range(10):
        fold_num = fold + 1  # Convert 0-9 to 1-10 for directory naming
        print(f"\n{'='*50}")
        print(f"Processing fold {fold_num}")
        print(f"{'='*50}")
        
        # Build model and load weights for this fold
        model = DDG_RDE_Network(config.model)
        model.load_state_dict(model_states[fold], strict=False)
        model = model.to(device)

        # Prepare datasets for this fold
        for split in ['train', 'val']:
            # Map 'val' to 'test' for txt file naming
            txt_split = 'test' if split == 'val' else split
            
            print(f"\nProcessing {split} split...")
            
            # Load expected complex IDs from txt files first
            expected_complex_ids = load_expected_complex_ids(
                config.data.folds_dir, threshold, fold_num, txt_split
            )
            
            split_dir = os.path.join(args.output_dir, f'fold_{fold_num}', 'train' if split == 'train' else 'test')
            
            # Process each complex ID from the txt file
            success = extract_embeddings_for_fold(
                model, device, split_dir, expected_complex_ids, 
                config.data.wildtype_dir, csv_path, transform
            )
            
            if not success:
                print(f"WARNING: Not all complexes in fold {fold_num} {split} were processed successfully")
    
    print("\n" + "="*50)
    print("EMBEDDING EXTRACTION COMPLETED")
    print("="*50)
    print(f"Output directory: {args.output_dir}")
    print("Check the output directories for the extracted embeddings:")
    print("- Each complex should have 4 files: _wt.npy, _mt.npy, _wt_resmap.pkl, _mt_resmap.pkl")
    print("- File names should exactly match the complex IDs from the txt files")

if __name__ == '__main__':
    main() 