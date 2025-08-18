import os
import argparse
import pandas as pd
import torch
from tqdm.auto import tqdm

from rde.utils.misc import load_config, seed_all, get_logger
from rde.utils.train import *
from rde.models.rde_ddg import DDG_RDE_Network
from rde.utils.skempi_wildtype_only import SkempiWildtypeOnlyDatasetManager, per_complex_corr


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('config', type=str)
    parser.add_argument('--checkpoint', type=str, required=True)
    parser.add_argument('--num_cvfolds', type=int, default=10)
    parser.add_argument('--threshold', type=int, default=60, help='Percentage threshold for fold splits (e.g., 60 for 60_percent)')
    parser.add_argument('--device', type=str, default='cuda')
    parser.add_argument('--num_workers', type=int, default=4)
    parser.add_argument('--output', type=str, default='./results_skempi_wildtype_only.csv')
    args = parser.parse_args()

    # Load configs
    config, config_name = load_config(args.config)
    seed_all(config.train.seed)

    # Data
    logger = get_logger('test', None)
    logger.info('Loading datasets...')
    dataset_mgr = SkempiWildtypeOnlyDatasetManager(
        config, 
        num_cvfolds=args.num_cvfolds,
        threshold=args.threshold,
        num_workers=args.num_workers,
        logger=logger,
    )

    # Model
    logger.info('Building model...')
    cv_mgr = CrossValidation(
        model_factory=DDG_RDE_Network,
        config=config, 
        num_cvfolds=args.num_cvfolds
    ).to(args.device)

    # Load checkpoint
    logger.info('Loading checkpoint: %s' % args.checkpoint)
    ckpt = torch.load(args.checkpoint, map_location=args.device)
    cv_mgr.load_state_dict(ckpt['model'])

    # Test
    logger.info('Testing...')
    results = []
    with torch.no_grad():
        for fold in range(args.num_cvfolds):
            model, optimizer, scheduler = cv_mgr.get(fold)
            model.eval()
            for i, batch in enumerate(tqdm(dataset_mgr.get_val_loader(fold), desc=f'Fold {fold+1}/{args.num_cvfolds}', dynamic_ncols=True)):
                # Prepare data
                batch = recursive_to(batch, args.device)

                # Forward pass
                loss_dict, output_dict = model(batch)

                for complex, mutstr, ddg_true, ddg_pred in zip(batch['complex'], batch['mutstr'], output_dict['ddG_true'], output_dict['ddG_pred']):
                    results.append({
                        'complex': complex,
                        'mutstr': mutstr,
                        'num_muts': len(mutstr.split(',')),
                        'ddG': ddg_true.item(),
                        'ddG_pred': ddg_pred.item()
                    })
    
    results = pd.DataFrame(results)
    results.to_csv(args.output, index=False)
    
    # Calculate metrics
    pearson_all = results[['ddG', 'ddG_pred']].corr('pearson').iloc[0, 1]
    spearman_all = results[['ddG', 'ddG_pred']].corr('spearman').iloc[0, 1]
    pearson_pc, spearman_pc = per_complex_corr(results)
    
    logger.info(f'[All] Pearson {pearson_all:.6f} Spearman {spearman_all:.6f}')
    logger.info(f'[PC]  Pearson {pearson_pc:.6f} Spearman {spearman_pc:.6f}')
    
    # Save metrics
    metrics_df = pd.DataFrame([{
        'metric': 'pearson_all',
        'value': pearson_all
    }, {
        'metric': 'spearman_all', 
        'value': spearman_all
    }, {
        'metric': 'pearson_pc',
        'value': pearson_pc
    }, {
        'metric': 'spearman_pc',
        'value': spearman_pc
    }])
    metrics_df.to_csv(args.output.replace('.csv', '_metrics.csv'), index=False)
    
    print(f"Results saved to {args.output}")
    print(f"Metrics saved to {args.output.replace('.csv', '_metrics.csv')}") 