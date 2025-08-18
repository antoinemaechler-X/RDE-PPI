import functools
import pandas as pd
from torch.utils.data import DataLoader
import numpy as np
from sklearn.metrics import roc_auc_score
from sklearn.linear_model import LinearRegression
from tqdm.auto import tqdm


from rde.utils.misc import inf_iterator, BlackHole
from rde.utils.data_separate_structures import create_separate_structures_dataloader
from rde.utils.transforms import get_transform
from rde.datasets.skempi_separate_structures import SkempiSeparateStructuresDataset


def per_complex_corr(df, pred_attr='ddG_pred', limit=10):
    corr_table = []
    for cplx in df['complex'].unique():
        df_cplx = df.query(f'complex == "{cplx}"')
        if len(df_cplx) < limit: 
            continue
        corr_table.append({
            'complex': cplx,
            'pearson': df_cplx[['ddG', pred_attr]].corr('pearson').iloc[0,1],
            'spearman': df_cplx[['ddG', pred_attr]].corr('spearman').iloc[0,1],
        })
    corr_table = pd.DataFrame(corr_table)
    
    # Handle case where no complexes meet the threshold
    if len(corr_table) == 0:
        return 0.0, 0.0
    
    avg = corr_table[['pearson', 'spearman']].mean()
    return avg['pearson'] , avg['spearman']


class SkempiSeparateStructuresDatasetManager(object):

    def __init__(self, config, num_cvfolds, threshold, num_workers=4, logger=BlackHole()):
        super().__init__()
        self.config = config
        self.num_cvfolds = num_cvfolds
        self.threshold = threshold
        self.train_iterators = []
        self.val_loaders = []
        self.logger = logger
        self.num_workers = num_workers
        for fold in range(num_cvfolds):
            train_iterator, val_loader = self.init_loaders(fold)
            self.train_iterators.append(train_iterator)
            self.val_loaders.append(val_loader)

    def init_loaders(self, fold):
        config = self.config
        dataset_ = functools.partial(
            SkempiSeparateStructuresDataset,
            csv_path = config.data.csv_path,
            wildtype_dir = config.data.wildtype_dir,
            optimized_dir = config.data.optimized_dir,
            cache_dir = config.data.cache_dir,
            folds_dir = config.data.folds_dir,
            threshold = self.threshold,
            cvfold_index = fold,
            transform = get_transform(config.data.transform)
        )
        train_dataset = dataset_(split='train')
        val_dataset = dataset_(split='val')
        
        # Count samples and complexes
        train_samples = len(train_dataset)
        val_samples = len(val_dataset)
        
        # Get unique PDB codes for logging
        train_pdb_codes = set()
        val_pdb_codes = set()
        
        for entry in train_dataset.entries:
            train_pdb_codes.add(entry['pdbcode'])
        for entry in val_dataset.entries:
            val_pdb_codes.add(entry['pdbcode'])
        
        # Check for data leakage
        leakage = train_pdb_codes.intersection(val_pdb_codes)
        assert len(leakage) == 0, f'data leakage {leakage}'

        # Create dataloaders using the custom collate function
        train_loader = create_separate_structures_dataloader(
            train_dataset, 
            batch_size=config.train.batch_size, 
            shuffle=True, 
            num_workers=self.num_workers
        )
        train_iterator = inf_iterator(train_loader)
        
        val_loader = create_separate_structures_dataloader(
            val_dataset, 
            batch_size=config.train.batch_size, 
            shuffle=False, 
            num_workers=self.num_workers
        )
        
        self.logger.info('Fold %d: %d train samples, %d test samples' % (fold, train_samples, val_samples))
        self.logger.info('      %d train PDB codes, %d test PDB codes' % (len(train_pdb_codes), len(val_pdb_codes)))
        return train_iterator, val_loader

    def get_train_iterator(self, fold):
        return self.train_iterators[fold]

    def get_val_loader(self, fold):
        return self.val_loaders[fold]


def overall_correlations(df):
    pearson = df[['ddG', 'ddG_pred']].corr('pearson').iloc[0,1]
    spearman = df[['ddG', 'ddG_pred']].corr('spearman').iloc[0,1]
    return {
        'overall_pearson': pearson, 
        'overall_spearman': spearman,
    }


def percomplex_correlations(df, return_details=False):
    corr_table = []
    for cplx in df['complex'].unique():
        df_cplx = df.query(f'complex == "{cplx}"')
        if len(df_cplx) < 10: 
            continue
        corr_table.append({
            'complex': cplx,
            'pearson': df_cplx[['ddG', 'ddG_pred']].corr('pearson').iloc[0,1],
            'spearman': df_cplx[['ddG', 'ddG_pred']].corr('spearman').iloc[0,1],
        })
    corr_table = pd.DataFrame(corr_table)
    average = corr_table[['pearson', 'spearman']].mean()
    out = {
        'percomplex_pearson': average['pearson'],
        'percomplex_spearman': average['spearman'],
    }
    if return_details:
        return out, corr_table
    else:
        return out


def overall_auroc(df):
    score = roc_auc_score(
        (df['ddG'] > 0).to_numpy(),
        df['ddG_pred'].to_numpy()
    )
    return {
        'auroc': score,
    }


def overall_rmse_mae(df):
    true = df['ddG'].to_numpy()
    pred = df['ddG_pred'].to_numpy()[:, None]
    reg = LinearRegression().fit(pred, true)
    pred_corrected = reg.predict(pred)
    rmse = np.sqrt( ((true - pred_corrected) ** 2).mean() )
    mae = np.abs(true - pred_corrected).mean()
    return {
        'rmse': rmse,
        'mae': mae,
    }


def analyze_all_results(df):
    methods = df['method'].unique()
    funcs = [
        overall_correlations,
        overall_rmse_mae,
        overall_auroc,
        percomplex_correlations,
    ]
    analysis = []
    for method in tqdm(methods):
        df_this = df[df['method'] == method]
        result = {
            'method': method,
        }
        for f in funcs:
            result.update(f(df_this))
        analysis.append(result)
    analysis = pd.DataFrame(analysis)
    return analysis


def analyze_all_percomplex_correlations(df):
    methods = df['method'].unique()
    df_corr = []
    for method in tqdm(methods):
        df_this = df[df['method'] == method]
        _, df_corr_this = percomplex_correlations(df_this, return_details=True)
        df_corr_this['method'] = method
        df_corr.append(df_corr_this)
    df_corr = pd.concat(df_corr).reset_index()
    return df_corr


def eval_skempi(df_items, mode, ddg_cutoff=None):
    assert mode in ('all', 'single', 'multiple')
    if mode == 'single':
        df_items = df_items.query('num_muts == 1')
    elif mode == 'multiple':
        df_items = df_items.query('num_muts > 1')

    if ddg_cutoff is not None:
        df_items = df_items.query(f"ddG >= {-ddg_cutoff} and ddG <= {ddg_cutoff}")

    df_metrics = analyze_all_results(df_items)
    df_corr = analyze_all_percomplex_correlations(df_items)
    df_metrics['mode'] = mode
    return df_metrics


def eval_skempi_three_modes(results, ddg_cutoff=None):
    df_all = eval_skempi(results, mode='all', ddg_cutoff=ddg_cutoff)
    df_single = eval_skempi(results, mode='single', ddg_cutoff=ddg_cutoff)
    df_multiple = eval_skempi(results, mode='multiple', ddg_cutoff=ddg_cutoff)
    
    return pd.concat([df_all, df_single, df_multiple], ignore_index=True) 