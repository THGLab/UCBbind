import pandas as pd
import numpy as np

# --- User settings ---
csv_file = 'predictions/predictions.csv'  # path to your CSV
threshold = 5  # binder threshold
pred_col = 'Predicted Free Energy'
true_col = 'Actual Free Energy'
module_x_col = 'Module X Pred'  # column for Module X predictions

# --- Load data ---
df = pd.read_csv(csv_file)

# --- Create ground truth classes ---
df['True_Class'] = np.where(df[true_col] >= threshold, 'Binder', 'Nonbinder')

# --- Predicted classes (main model) ---
df['Pred_Class'] = np.where(df[pred_col] >= threshold, 'Binder', 'Nonbinder')

# --- Predicted classes (Module X) ---
df['ModuleX_Class'] = np.where(df[module_x_col] >= threshold, 'Binder', 'Nonbinder')

# --- Helper function for accuracy + counts ---
def accuracy_with_counts(sub_df, class_col):
    correct = (sub_df['True_Class'] == sub_df[class_col]).sum()
    total = len(sub_df)
    acc = correct / total if total > 0 else np.nan
    return acc, correct, total

# --- Split data ---
binders = df[df['True_Class'] == 'Binder']
nonbinders = df[df['True_Class'] == 'Nonbinder']

# --- Main prediction accuracies ---
binder_acc, binder_correct, binder_total = accuracy_with_counts(binders, 'Pred_Class')
nonbinder_acc, nonbinder_correct, nonbinder_total = accuracy_with_counts(nonbinders, 'Pred_Class')
overall_acc = (df['True_Class'] == df['Pred_Class']).mean()
overall_correct = (df['True_Class'] == df['Pred_Class']).sum()

# --- Module X accuracies ---
binder_acc_x, binder_correct_x, binder_total_x = accuracy_with_counts(binders, 'ModuleX_Class')
nonbinder_acc_x, nonbinder_correct_x, nonbinder_total_x = accuracy_with_counts(nonbinders, 'ModuleX_Class')
overall_acc_x = (df['True_Class'] == df['ModuleX_Class']).mean()
overall_correct_x = (df['True_Class'] == df['ModuleX_Class']).sum()

# --- Print results ---
print("=== UCBbind Prediction ===")
print(f"Classification Accuracy (Binder):     {binder_acc:.4f} ({binder_correct}/{binder_total})")
print(f"Classification Accuracy (Nonbinder):  {nonbinder_acc:.4f} ({nonbinder_correct}/{nonbinder_total})")
print(f"Overall Accuracy:                     {overall_acc:.4f} ({overall_correct}/{len(df)})")
print()

print("=== Module X Only Prediction ===")
print(f"Classification Accuracy (Binder):     {binder_acc_x:.4f} ({binder_correct_x}/{binder_total_x})")
print(f"Classification Accuracy (Nonbinder):  {nonbinder_acc_x:.4f} ({nonbinder_correct_x}/{nonbinder_total_x})")
print(f"Overall Accuracy:                     {overall_acc_x:.4f} ({overall_correct_x}/{len(df)})")

