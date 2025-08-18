import functools
import pandas as pd
from torch.utils.data import DataLoader
import numpy as np
from sklearn.metrics import roc_auc_score
from sklearn.linear_model import LinearRegression
from tqdm.auto import tqdm


from rde.utils.misc import inf_iterator, BlackHole
from rde.utils.data import PaddingCollate
from rde.utils.transforms import get_transform
from rde.datasets.skempi_wildtype_only import SkempiWildtypeOnlyDataset


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


class SkempiWildtypeOnlyDatasetManager(object):

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
            SkempiWildtypeOnlyDataset,
            csv_path = config.data.csv_path,
            wildtype_dir = config.data.wildtype_dir,
            cache_dir = config.data.cache_dir,
            folds_dir = config.data.folds_dir,
            threshold = self.threshold,
            cvfold_index = fold,
            transform = get_transform(config.data.transform)
        )
        train_dataset = dataset_(split='train')
        val_dataset = dataset_(split='val')
        
        train_cplx = set([e['complex'] for e in train_dataset.entries])
        val_cplx = set([e['complex'] for e in val_dataset.entries])
        leakage = train_cplx.intersection(val_cplx)
        assert len(leakage) == 0, f'data leakage {leakage}'

        train_loader = DataLoader(
            train_dataset, 
            batch_size=config.train.batch_size, 
            shuffle=True, 
            collate_fn=PaddingCollate(), 
            num_workers=self.num_workers
        )
        train_iterator = inf_iterator(train_loader)
        val_loader = DataLoader(
            val_dataset, 
            batch_size=config.train.batch_size, 
            shuffle=False, 
            collate_fn=PaddingCollate(), 
            num_workers=self.num_workers
        )
        
        self.logger.info('Fold %d: %d train samples, %d test samples' % (fold, len(train_dataset), len(val_dataset)))
        self.logger.info('      %d train complexes, %d test complexes' % (len(train_cplx), len(val_cplx)))
        
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
            'n_samples': len(df_cplx),
        })
    corr_table = pd.DataFrame(corr_table)
    
    if return_details:
        return corr_table
    else:
        avg = corr_table[['pearson', 'spearman']].mean()
        return {
            'percomplex_pearson': avg['pearson'], 
            'percomplex_spearman': avg['spearman'],
        }


def overall_auroc(df):
    # Convert to binary classification: positive ddG = 1, negative ddG = 0
    df_binary = df.copy()
    df_binary['ddG_binary'] = (df_binary['ddG'] > 0).astype(int)
    
    try:
        auroc = roc_auc_score(df_binary['ddG_binary'], df_binary['ddG_pred'])
        return {'overall_auroc': auroc}
    except ValueError:
        return {'overall_auroc': np.nan}


def overall_rmse_mae(df):
    rmse = np.sqrt(np.mean((df['ddG'] - df['ddG_pred'])**2))
    mae = np.mean(np.abs(df['ddG'] - df['ddG_pred']))
    return {
        'overall_rmse': rmse, 
        'overall_mae': mae,
    }


def analyze_all_results(df):
    results = {}
    results.update(overall_correlations(df))
    results.update(percomplex_correlations(df))
    results.update(overall_auroc(df))
    results.update(overall_rmse_mae(df))
    return results


def analyze_all_percomplex_correlations(df):
    return percomplex_correlations(df, return_details=True)


def eval_skempi(df_items, mode, ddg_cutoff=None):
    if ddg_cutoff is not None:
        df_items = df_items[df_items['ddG'] > ddg_cutoff]
    
    if mode == 'all':
        return analyze_all_results(df_items)
    elif mode == 'percomplex':
        return analyze_all_percomplex_correlations(df_items)
    else:
        raise ValueError(f'Unknown mode: {mode}')


def eval_skempi_three_modes(results, ddg_cutoff=None):
    modes = ['all', 'percomplex']
    results_dict = {}
    
    for mode in modes:
        results_dict[mode] = eval_skempi(results, mode, ddg_cutoff)
    
    return results_dict 