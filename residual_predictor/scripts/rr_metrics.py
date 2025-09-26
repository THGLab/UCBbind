import pandas as pd
from sklearn.metrics import root_mean_squared_error, mean_absolute_error
from scipy.stats import pearsonr

def evaluate_module_y_predictions(csv_path):
    df = pd.read_csv(csv_path)

    total = len(df)
    module_counts = df['Module'].value_counts().to_dict()

    results = {}

    for mod in ['y-joint', 'y-ligand']:
        subset = df[df['Module'] == mod]
        n = len(subset)

        # Drop rows with NaNs in either column
        subset = subset.dropna(subset=['Module Y Pred', 'Actual Free Energy'])
        if subset.empty:
            results[mod] = {
                'count': n,
                'RMSE': np.nan,
                'MAE': np.nan,
                'Pearson': np.nan
            }
            continue

        y_pred = subset['Module Y Pred']
        y_true = subset['Actual Free Energy']

        rmse = root_mean_squared_error(y_true, y_pred)
        mae = mean_absolute_error(y_true, y_pred)
        pearson, _ = pearsonr(y_true, y_pred)

        results[mod] = {
            'count': n,
            'RMSE': rmse,
            'MAE': mae,
            'Pearson': pearson
        }

    # Count summary
    print(f"\nTotal predictions: {total}")
    for mod in ['x', 'y-joint', 'y-ligand']:
        print(f"Module '{mod}': {module_counts.get(mod, 0)}")

    # Metrics summary
    print("\nPerformance (Module Y only):")
    for mod in ['y-joint', 'y-ligand']:
        r = results[mod]
        print(f"\n{mod}:")
        print(f"  Count   : {r['count']}")
        print(f"  RMSE    : {r['RMSE'] if pd.notna(r['RMSE']) else 'nan'}")
        print(f"  MAE     : {r['MAE'] if pd.notna(r['MAE']) else 'nan'}")
        print(f"  Pearson : {r['Pearson'] if pd.notna(r['Pearson']) else 'nan'}")

if __name__ == '__main__':
    
    import pandas as pd
    from sklearn.metrics import mean_absolute_error, root_mean_squared_error
    from scipy.stats import pearsonr

    results_csv = '../training_data/revised_rr_predictions.csv'
    evaluate_module_y_predictions(results_csv)
