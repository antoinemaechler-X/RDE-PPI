import os
import argparse
import pandas as pd
import torch
from tqdm.auto import tqdm

from rde.utils.misc import load_config, seed_all, get_logger
from rde.utils.train import *
from rde.models.rde_ddg_separate_structures import DDG_RDE_Network_SeparateStructures
from rde.utils.skempi_separate_structures import SkempiSeparateStructuresDatasetManager, per_complex_corr


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('config', type=str)
    parser.add_argument('--checkpoint', type=str, required=True)
    parser.add_argument('--num_cvfolds', type=int, default=10)
    parser.add_argument('--threshold', type=int, default=60, help='Percentage threshold for fold splits (e.g., 60 for 60_percent)')
    parser.add_argument('--device', type=str, default='cuda')
    parser.add_argument('--num_workers', type=int, default=4)
    parser.add_argument('--output', type=str, default='skempi_separate_structures_results.csv')
    args = parser.parse_args()
    
    logger = get_logger('test', None)

    # Load checkpoint
    ckpt = torch.load(args.checkpoint, map_location=args.device)
    config = ckpt['config']
    num_cvfolds = len(ckpt['model']['models'])

    # Data
    logger.info('Loading datasets...')
    dataset_mgr = SkempiSeparateStructuresDatasetManager(
        config, 
        num_cvfolds=num_cvfolds,
        threshold=args.threshold,
        num_workers=args.num_workers,
        logger=logger,
    )

    # Model
    logger.info('Building model...')
    cv_mgr = CrossValidation(
        model_factory=DDG_RDE_Network_SeparateStructures,
        config=config, 
        num_cvfolds=num_cvfolds
    ).to(args.device)
    logger.info('Loading state dict...')
    cv_mgr.load_state_dict(ckpt['model'])

    # Evaluation
    scalar_accum = ScalarMetricAccumulator()
    results = []
    with torch.no_grad():
        for fold in range(num_cvfolds):
            model, _, _ = cv_mgr.get(fold)
            for i, batch in enumerate(tqdm(dataset_mgr.get_val_loader(fold), desc=f'Fold {fold+1}/{num_cvfolds}', dynamic_ncols=True)):
                # Prepare data
                batch = recursive_to(batch, args.device)

                # Forward pass
                loss_dict, output_dict = model(batch)
                loss = sum_weighted_losses(loss_dict, config.train.loss_weights)
                scalar_accum.add(name='loss', value=loss, batchsize=batch['wildtype']['size'], mode='mean')

                # Extract complex and mutation info from wildtype batch
                for complex, mutstr, ddg_true, ddg_pred in zip(
                    batch['wildtype']['complex'], 
                    batch['wildtype']['mutstr'], 
                    output_dict['ddG_true'], 
                    output_dict['ddG_pred']
                ):
                    results.append({
                        'complex': complex,
                        'mutstr': mutstr,
                        'num_muts': len(mutstr.split(',')),
                        'ddG': ddg_true.item(),
                        'ddG_pred': ddg_pred.item()
                    })
    
    results = pd.DataFrame(results)
    results['method'] = 'RDE_SeparateStructures'
    results.to_csv(args.output, index=False)
    
    # Calculate metrics
    pearson_all = results[['ddG', 'ddG_pred']].corr('pearson').iloc[0, 1]
    spearman_all = results[['ddG', 'ddG_pred']].corr('spearman').iloc[0, 1]
    pearson_pc, spearman_pc = per_complex_corr(results)
    
    print(f'Results saved to {args.output}')
    print(f'[All] Pearson {pearson_all:.6f} Spearman {spearman_all:.6f}')
    print(f'[PC]  Pearson {pearson_pc:.6f} Spearman {spearman_pc:.6f}')
    
    # Save metrics
    metrics_df = pd.DataFrame([{
        'method': 'RDE_SeparateStructures',
        'threshold': args.threshold,
        'all_pearson': pearson_all,
        'all_spearman': spearman_all,
        'pc_pearson': pearson_pc,
        'pc_spearman': spearman_pc,
    }])
    metrics_df.to_csv(args.output.replace('.csv', '_metrics.csv'), index=False)
    print(f'Metrics saved to {args.output.replace(".csv", "_metrics.csv")}') 