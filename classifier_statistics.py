import pandas as pd
import numpy as np

# --- User settings ---
csv_file = 'predictions/moonshot_predictions.csv'  # path to your CSV
threshold = 6.817  # binder threshold
pred_col = 'Predicted Free Energy'
status_col = 'Binder Status'

# --- Load data ---
df = pd.read_csv(csv_file)

# --- Check required columns ---
if pred_col not in df.columns or status_col not in df.columns:
    raise ValueError(f"CSV must contain columns '{pred_col}' and '{status_col}'")

# --- Calculate accuracy for binders ---
binders = df[df[status_col] == 'Binder']
if not binders.empty:
    binder_correct = (binders[pred_col] >= threshold).sum()
    binder_acc = binder_correct / len(binders)
else:
    binder_acc = np.nan

# --- Calculate accuracy for nonbinders ---
nonbinders = df[df[status_col] == 'Nonbinder']
if not nonbinders.empty:
    nonbinder_correct = (nonbinders[pred_col] < threshold).sum()
    nonbinder_acc = nonbinder_correct / len(nonbinders)
else:
    nonbinder_acc = np.nan

# --- Print results ---
print(f"Classification Accuracy (Binder):     {binder_acc:.4f}")
print(f"Classification Accuracy (Nonbinder):  {nonbinder_acc:.4f}")

