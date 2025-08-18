import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats

def analyze_results(file_path, approach_name):
    """Analyze results from a specific approach."""
    df = pd.read_csv(file_path)
    
    # Calculate correlations
    pearson_corr = df[['ddG', 'ddG_pred']].corr('pearson').iloc[0, 1]
    spearman_corr = df[['ddG', 'ddG_pred']].corr('spearman').iloc[0, 1]
    
    # Calculate RMSE
    rmse = np.sqrt(np.mean((df['ddG'] - df['ddG_pred'])**2))
    
    # Analyze by ddG sign
    positive_mask = df['ddG'] > 0
    negative_mask = df['ddG'] < 0
    
    positive_pearson = df[positive_mask][['ddG', 'ddG_pred']].corr('pearson').iloc[0, 1] if positive_mask.sum() > 0 else np.nan
    negative_pearson = df[negative_mask][['ddG', 'ddG_pred']].corr('pearson').iloc[0, 1] if negative_mask.sum() > 0 else np.nan
    
    positive_rmse = np.sqrt(np.mean((df[positive_mask]['ddG'] - df[positive_mask]['ddG_pred'])**2)) if positive_mask.sum() > 0 else np.nan
    negative_rmse = np.sqrt(np.mean((df[negative_mask]['ddG'] - df[negative_mask]['ddG_pred'])**2)) if negative_mask.sum() > 0 else np.nan
    
    # Count samples
    n_total = len(df)
    n_positive = positive_mask.sum()
    n_negative = negative_mask.sum()
    
    return {
        'approach': approach_name,
        'n_total': n_total,
        'n_positive': n_positive,
        'n_negative': n_negative,
        'pearson_all': pearson_corr,
        'spearman_all': spearman_corr,
        'rmse_all': rmse,
        'pearson_positive': positive_pearson,
        'pearson_negative': negative_pearson,
        'rmse_positive': positive_rmse,
        'rmse_negative': negative_rmse,
        'df': df
    }

def plot_comparison(results_list):
    """Create comparison plots."""
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    
    # 1. Overall correlations
    approaches = [r['approach'] for r in results_list]
    pearson_all = [r['pearson_all'] for r in results_list]
    spearman_all = [r['spearman_all'] for r in results_list]
    
    x = np.arange(len(approaches))
    width = 0.35
    
    axes[0, 0].bar(x - width/2, pearson_all, width, label='Pearson')
    axes[0, 0].bar(x + width/2, spearman_all, width, label='Spearman')
    axes[0, 0].set_xlabel('Approach')
    axes[0, 0].set_ylabel('Correlation')
    axes[0, 0].set_title('Overall Correlations')
    axes[0, 0].set_xticks(x)
    axes[0, 0].set_xticklabels(approaches)
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)
    
    # 2. RMSE comparison
    rmse_all = [r['rmse_all'] for r in results_list]
    axes[0, 1].bar(approaches, rmse_all)
    axes[0, 1].set_xlabel('Approach')
    axes[0, 1].set_ylabel('RMSE')
    axes[0, 1].set_title('Overall RMSE')
    axes[0, 1].grid(True, alpha=0.3)
    
    # 3. Positive vs Negative correlations
    pearson_pos = [r['pearson_positive'] for r in results_list]
    pearson_neg = [r['pearson_negative'] for r in results_list]
    
    axes[0, 2].bar(x - width/2, pearson_pos, width, label='Positive ddG')
    axes[0, 2].bar(x + width/2, pearson_neg, width, label='Negative ddG')
    axes[0, 2].set_xlabel('Approach')
    axes[0, 2].set_ylabel('Pearson Correlation')
    axes[0, 2].set_title('Correlation by ddG Sign')
    axes[0, 2].set_xticks(x)
    axes[0, 2].set_xticklabels(approaches)
    axes[0, 2].legend()
    axes[0, 2].grid(True, alpha=0.3)
    
    # 4. Scatter plots for each approach
    for i, result in enumerate(results_list):
        row = i // 3
        col = i % 3
        if row < 2:  # Only plot in first two rows
            df = result['df']
            axes[row, col].scatter(df['ddG'], df['ddG_pred'], alpha=0.5, s=1)
            axes[row, col].plot([df['ddG'].min(), df['ddG'].max()], [df['ddG'].min(), df['ddG'].max()], 'r--', alpha=0.8)
            axes[row, col].set_xlabel('True ddG')
            axes[row, col].set_ylabel('Predicted ddG')
            axes[row, col].set_title(f'{result["approach"]}\nPearson: {result["pearson_all"]:.3f}')
            axes[row, col].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('performance_comparison.png', dpi=300, bbox_inches='tight')
    plt.show()

def analyze_negative_predictions(results_list):
    """Detailed analysis of negative ddG predictions."""
    print("\n=== DETAILED ANALYSIS OF NEGATIVE ddG PREDICTIONS ===\n")
    
    for result in results_list:
        df = result['df']
        negative_mask = df['ddG'] < 0
        
        if negative_mask.sum() == 0:
            print(f"{result['approach']}: No negative ddG samples")
            continue
            
        df_neg = df[negative_mask]
        
        print(f"\n{result['approach']}:")
        print(f"  Total negative samples: {negative_mask.sum()}")
        print(f"  Pearson correlation: {result['pearson_negative']:.3f}")
        print(f"  RMSE: {result['rmse_negative']:.3f}")
        
        # Analyze prediction bias
        bias = np.mean(df_neg['ddG_pred'] - df_neg['ddG'])
        print(f"  Mean bias (pred - true): {bias:.3f}")
        
        # Analyze extreme negative predictions
        extreme_neg = df_neg[df_neg['ddG'] < -2]
        if len(extreme_neg) > 0:
            extreme_pearson = extreme_neg[['ddG', 'ddG_pred']].corr('pearson').iloc[0, 1]
            extreme_rmse = np.sqrt(np.mean((extreme_neg['ddG'] - extreme_neg['ddG_pred'])**2))
            print(f"  Extreme negative (< -2): {len(extreme_neg)} samples")
            print(f"    Pearson: {extreme_pearson:.3f}")
            print(f"    RMSE: {extreme_rmse:.3f}")
        
        # Check if predictions are systematically overestimated for negative ddG
        underestimated = (df_neg['ddG_pred'] < df_neg['ddG']).sum()
        overestimated = (df_neg['ddG_pred'] > df_neg['ddG']).sum()
        print(f"  Underestimated: {underestimated} ({underestimated/len(df_neg)*100:.1f}%)")
        print(f"  Overestimated: {overestimated} ({overestimated/len(df_neg)*100:.1f}%)")

def main():
    # Define file paths
    separate_structures_file = "logs_skempi_separate_structures/rde_ddg_skempi_separate_structures(10-60)_2025_07_29__02_39_13/checkpoints/results_30000.csv"
    custom_folds_file = "logs_skempi_custom_folds/rde_ddg_skempi_custom_folds(10-60)_2025_07_27__19_14_23/checkpoints/results_25000.csv"
    
    # Analyze results
    results = []
    
    try:
        separate_results = analyze_results(separate_structures_file, "Separate Structures")
        results.append(separate_results)
        print(f"Loaded separate structures results: {separate_results['n_total']} samples")
    except Exception as e:
        print(f"Error loading separate structures results: {e}")
    
    try:
        custom_folds_results = analyze_results(custom_folds_file, "Custom Folds")
        results.append(custom_folds_results)
        print(f"Loaded custom folds results: {custom_folds_results['n_total']} samples")
    except Exception as e:
        print(f"Error loading custom folds results: {e}")
    
    if len(results) < 2:
        print("Need at least 2 approaches to compare")
        return
    
    # Print summary table
    print("\n=== PERFORMANCE SUMMARY ===")
    summary_df = pd.DataFrame([
        {
            'Approach': r['approach'],
            'Total Samples': r['n_total'],
            'Positive ddG': r['n_positive'],
            'Negative ddG': r['n_negative'],
            'Pearson All': f"{r['pearson_all']:.3f}",
            'Spearman All': f"{r['spearman_all']:.3f}",
            'RMSE All': f"{r['rmse_all']:.3f}",
            'Pearson Positive': f"{r['pearson_positive']:.3f}",
            'Pearson Negative': f"{r['pearson_negative']:.3f}",
            'RMSE Positive': f"{r['rmse_positive']:.3f}",
            'RMSE Negative': f"{r['rmse_negative']:.3f}",
        }
        for r in results
    ])
    print(summary_df.to_string(index=False))
    
    # Detailed analysis of negative predictions
    analyze_negative_predictions(results)
    
    # Create comparison plots
    plot_comparison(results)
    
    # Save summary to CSV
    summary_df.to_csv('performance_comparison_summary.csv', index=False)
    print("\nSummary saved to 'performance_comparison_summary.csv'")

if __name__ == "__main__":
    main() 