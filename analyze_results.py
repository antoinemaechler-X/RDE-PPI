import pandas as pd
import numpy as np
from scipy import stats
import matplotlib.pyplot as plt
import seaborn as sns

def analyze_results(csv_path):
    # Read the results
    df = pd.read_csv(csv_path)
    print(f"\nResults dataframe shape: {df.shape}")

    # Determine column names for true and predicted ddG
    if {'ddG', 'ddG_pred'}.issubset(df.columns):
        true_col = 'ddG'
        pred_col = 'ddG_pred'
    elif {'true_ddG', 'predicted_ddG'}.issubset(df.columns):
        true_col = 'true_ddG'
        pred_col = 'predicted_ddG'
    else:
        raise ValueError("Could not find appropriate ddG columns in the CSV.")

    # Calculate overall correlations
    pearson_corr = df[true_col].corr(df[pred_col], method='pearson')
    spearman_corr = df[true_col].corr(df[pred_col], method='spearman')
    rmse = np.sqrt(np.mean((df[true_col] - df[pred_col])**2))
    mae = np.mean(np.abs(df[true_col] - df[pred_col]))
    
    # Compute AUROC assuming positive class is ddG < 0 (stabilizing)
    # Uses -predicted ddG as score so higher scores correspond to positive class
    auroc = np.nan
    try:
        from sklearn.metrics import roc_auc_score
        y_true_binary = (df[true_col] < 0).astype(int)
        # AUROC requires both classes to be present
        if y_true_binary.nunique() == 2:
            y_score = -df[pred_col]
            auroc = float(roc_auc_score(y_true_binary, y_score))
        else:
            print("Warning: AUROC not computed because only one class is present in the true labels (based on ddG < 0 threshold).")
    except Exception as e:
        print(f"Warning: Could not compute AUROC ({e}).")

    # Print overall metrics
    print(f"\nAnalysis of {csv_path}")
    print("-" * 50)
    print(f"Number of predictions: {len(df)}")
    print(f"Overall Pearson correlation: {pearson_corr:.4f}")
    print(f"Overall Spearman correlation: {spearman_corr:.4f}")
    print(f"Overall AUROC: {auroc:.4f}")
    print(f"Overall RMSE: {rmse:.4f}")
    print(f"Overall MAE: {mae:.4f}")

    # Save overall metrics to CSV next to the input CSV
    metrics_path = csv_path.replace('.csv', '_metrics.csv')
    metrics_df = pd.DataFrame([{
        'pearson': pearson_corr,
        'spearman': spearman_corr,
        'auroc': auroc,
        'rmse': rmse,
        'mae': mae
    }])
    metrics_df.to_csv(metrics_path, index=False)
    print(f"\nMetrics CSV saved to: {metrics_path}")

    # If 'method' column exists, print per-method metrics
    if 'method' in df.columns:
        print(f"\nPer-method results:")
        print("-" * 50)
        for method in sorted(df['method'].unique()):
            method_data = df[df['method'] == method]
            m_pearson = method_data[true_col].corr(method_data[pred_col], method='pearson')
            m_spearman = method_data[true_col].corr(method_data[pred_col], method='spearman')
            m_rmse = np.sqrt(np.mean((method_data[true_col] - method_data[pred_col])**2))
            m_mae = np.mean(np.abs(method_data[true_col] - method_data[pred_col]))
            print(f"Method {method}: Pearson = {m_pearson:.4f}, Spearman = {m_spearman:.4f}, RMSE = {m_rmse:.4f}, MAE = {m_mae:.4f}, n = {len(method_data)}")

    # If 'fold' column exists, print fold-by-fold results
    if 'fold' in df.columns:
        print(f"\nFold-by-fold results:")
        print("-" * 50)
        for fold in sorted(df['fold'].unique()):
            fold_data = df[df['fold'] == fold]
            fold_pearson = fold_data[true_col].corr(fold_data[pred_col], method='pearson')
            fold_spearman = fold_data[true_col].corr(fold_data[pred_col], method='spearman')
            print(f"Fold {fold}: Pearson = {fold_pearson:.4f}, Spearman = {fold_spearman:.4f}, n = {len(fold_data)}")

    # Create simple scatter plot
    plt.figure(figsize=(8, 6))
    plt.scatter(df[true_col], df[pred_col], alpha=0.6)

    # Add diagonal line
    min_val = min(df[true_col].min(), df[pred_col].min())
    max_val = max(df[true_col].max(), df[pred_col].max())
    plt.plot([min_val, max_val], [min_val, max_val], 'r--', alpha=0.8)

    plt.xlabel('True ddG (kcal/mol)')
    plt.ylabel('Predicted ddG (kcal/mol)')
    plt.title(f'Prediction vs True ddG\nPearson r = {pearson_corr:.4f}, Spearman  = {spearman_corr:.4f}')
    plt.grid(True, alpha=0.3)

    # Add correlation coefficients text box
    plt.text(0.05, 0.95, f'Pearson r = {pearson_corr:.4f}\nSpearman  = {spearman_corr:.4f}', 
            transform=plt.gca().transAxes, fontsize=12,
            verticalalignment='top', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))

    plt.tight_layout()

    # Save plot
    plot_path = csv_path.replace('.csv', '_correlation.png')
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"\nCorrelation plot saved to: {plot_path}")

if __name__ == "__main__":
    # Analyze the cross-validation results file
    csv_path = "RDE_linear_skempi_wildtype_only_60/results.csv"
    analyze_results(csv_path) 