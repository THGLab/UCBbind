import numpy as np
import pandas as pd

def calculate_metrics(df, true_col, pred_col):
    """Calculate MSE, RMSE, MAE, and Pearson correlation."""
    valid = df[[true_col, pred_col]].dropna()
    if valid.empty:
        return np.nan, np.nan, np.nan, np.nan
    mse = np.mean((valid[true_col] - valid[pred_col]) ** 2)
    rmse = np.sqrt(mse)
    mae = np.mean(abs(valid[true_col] - valid[pred_col]))
    pearson_corr = valid[true_col].corr(valid[pred_col])
    return round(mse, 4), round(rmse, 4), round(mae, 4), round(pearson_corr, 4)


def display_metrics_single(df, split_name=None):
    """Display Module X, global prediction, Module Y comparisons, and Binder/Nonbinder breakdowns."""

    def _display_for_subset(subset, label):
        if subset.empty:
            print(f"\n→ No data for {label}")
            return

        print(f"\n→ Metrics for: {label}")
        print("-" * 40)

        # Global Hybrid Prediction
        if 'Predicted Free Energy' in subset.columns:
            print("Global Prediction (Final Output):")
            mse, rmse, mae, corr = calculate_metrics(subset, 'Actual Free Energy', 'Predicted Free Energy')
            print(f"{'MSE:':<25}{mse}")
            print(f"{'RMSE:':<25}{rmse}")
            print(f"{'MAE:':<25}{mae}")
            print(f"{'Pearson Correlation:':<25}{corr}")
            print("-" * 40)

        # Module X (global)
        if 'Module X Pred' in subset.columns:
            print("Module X Performance (All Data):")
            mse, rmse, mae, corr = calculate_metrics(subset, 'Actual Free Energy', 'Module X Pred')
            print(f"{'MSE:':<25}{mse}")
            print(f"{'RMSE:':<25}{rmse}")
            print(f"{'MAE:':<25}{mae}")
            print(f"{'Pearson Correlation:':<25}{corr}")
            print("-" * 40)

        # Module X only pairs
        df_x_only = subset[subset['Module'] == 'x']
        if not df_x_only.empty:
            print("Module X Performance (Module X Pairs Only):")
            mse, rmse, mae, corr = calculate_metrics(df_x_only, 'Actual Free Energy', 'Module X Pred')
            print(f"{'MSE:':<25}{mse}")
            print(f"{'RMSE:':<25}{rmse}")
            print(f"{'MAE:':<25}{mae}")
            print(f"{'Pearson Correlation:':<25}{corr}")
            print("-" * 40)
        else:
            print("No pairs routed to Module X for Module X only metrics.")
            print("-" * 40)

        # Module Y combined
        required_cols = {'Module X Pred', 'Module Y Pred', 'Weighted Mean'}
        if required_cols.issubset(subset.columns):
            y_modules = ['y-ligand', 'y-joint']
            y_subset = subset[subset['Module'].isin(y_modules)][
                ['Actual Free Energy', 'Module X Pred', 'Module Y Pred', 'Weighted Mean']
            ].dropna()
            if y_subset.empty:
                print("No valid combined Module Y rows with all predictions for comparison.")
            else:
                print("→ Comparing Module X vs Combined Module Y vs Weighted Mean (on All Module Y Pairs)")
                mse_x, rmse_x, mae_x, corr_x = calculate_metrics(y_subset, 'Actual Free Energy', 'Module X Pred')
                mse_y, rmse_y, mae_y, corr_y = calculate_metrics(y_subset, 'Actual Free Energy', 'Module Y Pred')
                mse_w, rmse_w, mae_w, corr_w = calculate_metrics(y_subset, 'Actual Free Energy', 'Weighted Mean')
                print(f"{'':25s} {'Module X':>12s} {'Combined Module Y':>20s} {'Weighted Mean':>17s}")
                print(f"{'MSE:':25s} {mse_x:12.4f} {mse_y:20.4f} {mse_w:17.4f}")
                print(f"{'RMSE:':25s} {rmse_x:12.4f} {rmse_y:20.4f} {rmse_w:17.4f}")
                print(f"{'MAE:':25s} {mae_x:12.4f} {mae_y:20.4f} {mae_w:17.4f}")
                print(f"{'Pearson Correlation:':25s} {corr_x:12.4f} {corr_y:20.4f} {corr_w:17.4f}")
            print("-" * 40)

        # Individual Y breakdowns
        for mod_type, mod_label in [('y-joint', 'Module Y Joint'), ('y-ligand', 'Module Y Ligand')]:
            subset_mod = subset[subset['Module'] == mod_type][
                ['Actual Free Energy', 'Module X Pred', 'Module Y Pred', 'Weighted Mean']
            ].dropna()
            if subset_mod.empty:
                print(f"No valid {mod_label} rows with all predictions for comparison.")
            else:
                print(f"→ Comparing Module X vs {mod_label} vs Weighted Mean (on {mod_label} Pairs Only)")
                mse_x, rmse_x, mae_x, corr_x = calculate_metrics(subset_mod, 'Actual Free Energy', 'Module X Pred')
                mse_y, rmse_y, mae_y, corr_y = calculate_metrics(subset_mod, 'Actual Free Energy', 'Module Y Pred')
                mse_w, rmse_w, mae_w, corr_w = calculate_metrics(subset_mod, 'Actual Free Energy', 'Weighted Mean')
                print(f"{'':25s} {'Module X':>12s} {mod_label:>20s} {'Weighted Mean':>17s}")
                print(f"{'MSE:':25s} {mse_x:12.4f} {mse_y:20.4f} {mse_w:17.4f}")
                print(f"{'RMSE:':25s} {rmse_x:12.4f} {rmse_y:20.4f} {rmse_w:17.4f}")
                print(f"{'MAE:':25s} {mae_x:12.4f} {mae_y:20.4f} {mae_w:17.4f}")
                print(f"{'Pearson Correlation:':25s} {corr_x:12.4f} {corr_y:20.4f} {corr_w:17.4f}")
            print("-" * 40)

        # Module usage counts
        if 'Module' in subset.columns:
            total_count = len(subset)
            count_x = (subset['Module'] == 'x').sum()
            count_y_joint = (subset['Module'] == 'y-joint').sum()
            count_y_ligand = (subset['Module'] == 'y-ligand').sum()
            count_y = count_y_joint + count_y_ligand

            print(f"Module 'X': {count_x}/{total_count} pairs")
            print(f"Module 'Y': {count_y}/{total_count} pairs")
            print(f"  - Y-JOINT: {count_y_joint} pairs")
            print(f"  - Y-LIGAND: {count_y_ligand} pairs")
            print("=" * 40)

    # ---- Run for overall ----
    name = split_name if split_name else "Overall"
    _display_for_subset(df, name)

    # ---- Run for binders/nonbinders ----
    if 'Binder Status' in df.columns:
        for status in ['Binder', 'Nonbinder']:
            sub = df[df['Binder Status'] == status]
            _display_for_subset(sub, f"{name} ({status})")


def display_metrics(dfs):
    """Display metrics for each DataFrame in the dict."""
    for filename, df in dfs.items():
        print(f"\n{'='*40}\nEvaluating file: {filename}\n{'='*40}")
        display_metrics_single(df)


def load_data(folder):
    """Load all CSV files in a folder into a dict of DataFrames."""
    import os
    dfs = {}
    for file in os.listdir(folder):
        if file.endswith(".csv"):
            dfs[file] = pd.read_csv(os.path.join(folder, file))
    return dfs


def main():
    folder = 'predictions'  # Change to your folder path
    dfs = load_data(folder)
    display_metrics(dfs)


if __name__ == "__main__":
    main()

