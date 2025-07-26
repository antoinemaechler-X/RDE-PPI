import os
import argparse
import pickle
import numpy as np
import torch
from tqdm import tqdm
import gc

from rde.models.rde import CircularSplineRotamerDensityEstimator
from rde.linear.entropy import SkempiStructureRepo, SkempiMutationDataset
from rde.utils.transforms import Compose, SelectAtom


def get_full_complex_data(dataset, idx, state):
    # This is a copy of SkempiMutationDataset.get, but skips patching/subsetting
    entry = dataset.entries[idx]
    data, seq_map = dataset.repo[entry['pdbcode']]
    data = dataset.pre_transform(data)

    from Bio.PDB.Polypeptide import one_to_index, index_to_one
    from rde.utils.protein.constants import num_chi_angles
    mutation_flag = torch.zeros((data['aa'].shape[0]), dtype=torch.bool)
    chi_corrupt = data['chi'].clone()
    mut_beta_positions = []
    for mutation in entry['mutations']:
        position = (mutation['chain'], mutation['resseq'], mutation['icode'])
        seq_idx = seq_map[position]
        mutation_flag[seq_idx] = True
        chi_corrupt[seq_idx] = 0.0

        # Mutate the protein
        if state == 'mt':
            mtype = one_to_index(mutation['mt'])
            data['aa'][seq_idx] = mtype
            data['chi'][seq_idx] = 0.0
            data['chi_alt'][seq_idx] = 0.0
            data['chi_mask'][seq_idx] = False
            data['chi_mask'][seq_idx, :num_chi_angles[mtype]] = True

        pos_atom = data['pos_heavyatom'][seq_idx, :5]   # (5, 3)
        msk_atom = data['mask_heavyatom'][seq_idx, :5]  # (5,)
        beta_pos = pos_atom[4] if msk_atom[4].item() else pos_atom[1]
        mut_beta_positions.append(beta_pos)
    mut_beta_positions = torch.stack(mut_beta_positions)    # (M, 3)
    data['chi_masked_flag'] = mutation_flag
    data['chi_corrupt'] = chi_corrupt

    # For each residue, compute the distance to the closest mutated residue
    def _get_Cbeta_positions(pos_atoms, mask_atoms):
        from rde.utils.protein.constants import BBHeavyAtom
        L = pos_atoms.size(0)
        pos_CA = pos_atoms[:, BBHeavyAtom.CA]   # (L, 3)
        if pos_atoms.size(1) < 5:
            return pos_CA
        pos_CB = pos_atoms[:, BBHeavyAtom.CB]
        mask_CB = mask_atoms[:, BBHeavyAtom.CB, None].expand(L, 3)
        return torch.where(mask_CB, pos_CB, pos_CA)
    beta_pos = _get_Cbeta_positions(data['pos_heavyatom'], data['mask_heavyatom'])
    pw_dist = torch.cdist(beta_pos, mut_beta_positions) # (N, M)
    dist_to_mut = pw_dist.min(dim=1)[0] # (N, )
    data['dist_to_mut'] = dist_to_mut

    # Flags
    receptor_flag = torch.BoolTensor([
        (c in entry['group_receptor']) for c in data['chain_id']
    ])
    ligand_flag = torch.BoolTensor([
        (c in entry['group_ligand']) for c in data['chain_id']
    ])
    data['receptor_flag'] = receptor_flag
    data['ligand_flag'] = ligand_flag

    # Add the information of closest residues in the receptor
    receptors = []
    rec_idx = torch.logical_and(
        dist_to_mut <= 8.0,
        receptor_flag
    ).nonzero().flatten()
    for idx2 in rec_idx:
        receptors.append({
            'chain': data['chain_id'][idx2],
            'resseq': data['resseq'][idx2].item(),
            'icode': data['icode'][idx2],
            'type': index_to_one(data['aa'][idx2].item()),
            'distance': dist_to_mut[idx2].item(),
        })
    entry['receptors'] = receptors

    # Add the information of closest residues in the ligand
    lignbrs = []
    lig_idx = torch.logical_and(
        dist_to_mut <= 8.0,
        ligand_flag
    ).nonzero().flatten()
    for idx2 in lig_idx:
        lignbrs.append({
            'chain': data['chain_id'][idx2],
            'resseq': data['resseq'][idx2].item(),
            'icode': data['icode'][idx2],
            'type': index_to_one(data['aa'][idx2].item()),
            'distance': dist_to_mut[idx2].item(),
        })
    entry['lignbrs'] = lignbrs

    # Select the chain group (for full complex, keep all residues)
    # No masking or subsetting here!

    # Add tags
    data['entry'] = entry
    data['group'] = 'complex'
    data['state'] = state
    return data


def extract_embeddings(
    ckpt_path,
    skempi_dir,
    skempi_cache_path,
    skempi_csv_path,
    output_dir,
    device='cuda',
):
    os.makedirs(output_dir, exist_ok=True)
    # Load model
    ckpt = torch.load(ckpt_path, map_location='cpu')
    model = CircularSplineRotamerDensityEstimator(ckpt['config']['model']).to(device)
    model.load_state_dict(ckpt['model'])
    model.eval()
    torch.set_grad_enabled(False)

    # Load dataset
    structure_repo = SkempiStructureRepo(
        root=skempi_dir,
        cache_path=skempi_cache_path,
    )
    dataset = SkempiMutationDataset(
        structure_repo,
        csv_path=skempi_csv_path,
        entry_filters=[],
        patch_size=4096,  # ignored in our get_full_complex_data
    )

    # For each entry, extract embeddings for wt and mt
    for idx in tqdm(range(len(dataset)), desc='Extracting embeddings'):
        entry = dataset.entries[idx]
        complex_name = entry['complex']
        for state in ['wt', 'mt']:
            out_path = os.path.join(output_dir, f"{idx}_{complex_name}_{state}.npy")
            if os.path.exists(out_path):
                continue  # Resume logic: skip if already computed
            try:
                batch = get_full_complex_data(dataset, idx, state)
                batch_device = {k: (v.to(device) if isinstance(v, torch.Tensor) else v) for k, v in batch.items()}
                for k in batch_device:
                    if isinstance(batch_device[k], torch.Tensor) and batch_device[k].dtype == torch.bool:
                        batch_device[k] = batch_device[k].to(torch.uint8)
                for k in batch_device:
                    if isinstance(batch_device[k], torch.Tensor):
                        batch_device[k] = batch_device[k].unsqueeze(0)
                with torch.no_grad():
                    emb = model.encode(batch_device)
                emb = emb.squeeze(0).cpu().numpy()
                if emb.ndim == 2:
                    padded_emb = emb
                else:
                    max_dim = max(e.shape[0] for e in emb)
                    padded_emb = np.zeros((len(emb), max_dim), dtype=np.float32)
                    for i, e in enumerate(emb):
                        padded_emb[i, :e.shape[0]] = e
                np.save(out_path, padded_emb)
            except RuntimeError as e:
                print(f"[WARNING] Skipping idx={idx}, complex={complex_name}, state={state} due to error: {e}")
                continue
            finally:
                if 'emb' in locals():
                    del emb
                if 'padded_emb' in locals():
                    del padded_emb
                if 'batch_device' in locals():
                    del batch_device
                gc.collect()
                torch.cuda.empty_cache()

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--ckpt', type=str, required=True, help='Path to RDE checkpoint')
    parser.add_argument('--skempi_dir', type=str, required=True, help='Path to SKEMPI PDBs')
    parser.add_argument('--skempi_cache_path', type=str, required=True, help='Path to SKEMPI cache')
    parser.add_argument('--skempi_csv_path', type=str, required=True, help='Path to SKEMPI CSV')
    parser.add_argument('--output_dir', type=str, default='complex_embeddings', help='Output folder')
    parser.add_argument('--device', type=str, default='cuda')
    args = parser.parse_args()

    extract_embeddings(
        ckpt_path=args.ckpt,
        skempi_dir=args.skempi_dir,
        skempi_cache_path=args.skempi_cache_path,
        skempi_csv_path=args.skempi_csv_path,
        output_dir=args.output_dir,
        device=args.device,
    ) 