import os
import argparse
import numpy as np
import torch
from tqdm import tqdm
from rde.utils.misc import load_config
import pickle

from rde.models.rde_ddg import DDG_RDE_Network
from rde.datasets.skempi_grouped import SkempiGroupedDataset
from rde.utils.transforms import get_transform


def extract_embeddings_for_fold(
    model,
    dataset,
    device,
    output_dir,
):
    os.makedirs(output_dir, exist_ok=True)
    model.eval()
    with torch.no_grad():
        for idx in tqdm(range(len(dataset)), desc=f'Extracting embeddings to {output_dir}'):
            data = dataset[idx]
            # Prepare wildtype batch
            batch_wt = {k: (v.unsqueeze(0).to(device) if isinstance(v, torch.Tensor) else v) for k, v in data.items()}
            # Prepare mutant batch (copy and set aa to aa_mut)
            batch_mt = batch_wt.copy()
            batch_mt['aa'] = data['aa_mut'].unsqueeze(0).to(device)
            # Extract per-residue embeddings before maxpool/MLP
            emb_wt = model.encode(batch_wt).squeeze(0).cpu().numpy()  # (L, D)
            emb_mt = model.encode(batch_mt).squeeze(0).cpu().numpy()  # (L, D)
            # Extract residue mapping
            chain_ids = data['chain_id']
            resseqs = data['resseq']
            icodes = data['icode']
            if hasattr(chain_ids, 'tolist'): chain_ids = chain_ids.tolist()
            if hasattr(resseqs, 'tolist'): resseqs = resseqs.tolist()
            if hasattr(icodes, 'tolist'): icodes = icodes.tolist()
            residue_ids = list(zip(chain_ids, resseqs, icodes))
            # Save with informative name
            complex_name = data.get('complex', f'idx{idx}')
            mutstr = data.get('mutstr', 'NA')
            base_path = os.path.join(output_dir, f"{idx}_{complex_name}_{mutstr}")
            # Save wildtype
            np.save(base_path + "_wt.npy", emb_wt)
            with open(base_path + "_wt_resmap.pkl", 'wb') as f:
                pickle.dump(residue_ids, f)
            # Save mutant
            np.save(base_path + "_mt.npy", emb_mt)
            with open(base_path + "_mt_resmap.pkl", 'wb') as f:
                pickle.dump(residue_ids, f)


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

    # For each fold
    for fold in range(10):
        print(f"Processing fold {fold}")
        # Build model and load weights for this fold
        model = DDG_RDE_Network(config.model)
        model.load_state_dict(model_states[fold], strict=False)
        model = model.to(device)

        # Prepare datasets for this fold
        for split in ['train', 'val']:
            dataset = SkempiGroupedDataset(
                csv_path=config.data.csv_path,
                pdb_dir=config.data.pdb_dir,
                cache_dir=config.data.cache_dir,
                grouped_csv_path=config.data.grouped_csv_path,
                split=split,
                num_cvfolds=10,
                cvfold_index=fold,
                transform=transform,
            )
            split_dir = os.path.join(args.output_dir, f'fold_{fold}', 'train' if split == 'train' else 'test')
            extract_embeddings_for_fold(model, dataset, device, split_dir)

if __name__ == '__main__':
    main() 