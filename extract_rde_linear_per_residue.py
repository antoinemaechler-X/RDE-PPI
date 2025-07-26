#!/usr/bin/env python3
"""
Extract per-residue embeddings from RDE Linear pipeline.
Exactly like calibrate_grouped but stops before summing/regression.
"""

import os
import argparse
import pickle
import numpy as np
import pandas as pd
import torch
from rde.linear.calibrate_grouped import (
    convert_results_to_table, 
    TableToArrays, 
    Regression,
    run_calibration
)

def extract_embeddings(
    result, 
    grouped_csv_path,
    output_dir='./rde_linear_per_residue_embeddings',
    device='cpu'
):
    """Extract per-residue embeddings for all entries."""
    os.makedirs(output_dir, exist_ok=True)
    
    # Same as in run_calibration
    block_list = {'1KBH'}
    result = {
        k: v 
        for k, v in result.items() 
        if v['pdbcode'] not in block_list
    }
    
    # Same table conversion
    table = convert_results_to_table(result, single=False)
    
    # Same array creation
    arrayer = TableToArrays(table)
    Hs, Ts, ddG, labels, groups, grp_tensor = arrayer.to_tensor(device)
    
    # Same model creation (for add_ref)
    model = Regression(num_terms=len(labels), labels=labels).to(device)
    
    # Apply add_ref (same as in Regression.forward)
    Hs_ref = [model.add_ref(H, T) for H, T in zip(Hs, Ts)]

    # For each complex, build a (num_residues_total, num_terms_total) embedding matrix
    for idx in range(len(groups)):
        complex_name = groups[idx]

        # Hs_ref[0]: ligand, shape (4, num_ligand_residues, num_data)
        # Hs_ref[1]: receptor, shape (3, num_receptor_residues, num_data)
        ligand = Hs_ref[0][:, :, idx].detach().numpy()  # (4, num_ligand_residues)
        receptor = Hs_ref[1][:, :, idx].detach().numpy()  # (3, num_receptor_residues)

        # Remove padded residues: find which columns are all zero (these are padding)
        # For ligand
        ligand_mask = ~(np.all(ligand == 0, axis=0))  # shape (num_ligand_residues,)
        ligand = ligand[:, ligand_mask]  # (4, num_real_ligand_residues)
        # For receptor
        receptor_mask = ~(np.all(receptor == 0, axis=0))  # shape (num_receptor_residues,)
        receptor = receptor[:, receptor_mask]  # (3, num_real_receptor_residues)

        # Transpose to (num_real_ligand_residues, 4) and (num_real_receptor_residues, 3)
        ligand = ligand.T
        receptor = receptor.T

        # Concatenate along residue axis: ligand residues first, then receptor residues
        all_embeddings = np.concatenate([
            np.concatenate([ligand, np.zeros((ligand.shape[0], 3))], axis=1),
            np.concatenate([np.zeros((receptor.shape[0], 4)), receptor], axis=1)
        ], axis=0)  # (num_real_ligand_residues + num_real_receptor_residues, 7)

        out_name = f"{idx}_{complex_name}.npy"
        out_path = os.path.join(output_dir, out_name)
        np.save(out_path, all_embeddings)

        # Save residue indices for traceability (optional, but recommended)
        # If you have access to chain_id, resseq, icode, save them here
        # For now, just save the index order
        csv_path = os.path.join(output_dir, f"{idx}_{complex_name}_residues.csv")
        with open(csv_path, 'w') as f:
            f.write('residue_index,group\n')
            for i in range(ligand.shape[0]):
                f.write(f'{i},ligand\n')
            for i in range(receptor.shape[0]):
                f.write(f'{i},receptor\n')

        if idx % 100 == 0:
            print(f"Saved {out_name} with shape {all_embeddings.shape}")

    print(f"Done. Saved {len(groups)} per-residue embedding files to {output_dir}")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('result', type=str, help='Path to RDE_skempi_entropy.pkl')
    parser.add_argument('--grouped_csv_path', type=str, default='./data/complex_sequences_grouped_99.csv')
    parser.add_argument('-o', '--output_dir', type=str, default='./rde_linear_per_residue_embeddings')
    parser.add_argument('-d', '--device', type=str, default='cpu')
    args = parser.parse_args()

    with open(args.result, 'rb') as f:
        result = pickle.load(f)
    
    extract_embeddings(
        result,
        args.grouped_csv_path,
        output_dir=args.output_dir,
        device=args.device
    )

if __name__ == '__main__':
    main() 