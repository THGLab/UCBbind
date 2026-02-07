import os
import time
import tempfile
from concurrent.futures import ThreadPoolExecutor
from functools import partial
import argparse
import pickle

import pandas as pd
import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm

from module_y.mmseqs_config import MMseqsConfig
from module_y.sequencesearch import get_similar_proteins
from module_y.smilessearch import get_similar_ligands
from metrics import load_data, display_metrics

from module_x.dataset import label_smiles, label_sequence, CHARISOSMISET, CHARPROTSET
from module_x.model import AttentionDTA

# -----------------------------
# Load datasets
# -----------------------------
ref_fp = 'datasets/BindingDB.csv'
ref_df = pd.read_csv(ref_fp)
orig_size = ref_df.shape[0]

test_fp = '/global/scratch/users/justinpurnomo/cache/ucbbind_gnina_boltz_on_del_hits/ucbbind_validity_check/PGK2_1k_chembl.csv'
base = os.path.basename(test_fp)
stem = os.path.splitext(base)[0]

test_df = pd.read_csv(test_fp)

# test_df = test_df[test_df['Split']=='Test'].reset_index(drop=True)
test_size = test_df.shape[0]

# Remove test pairs from reference
# test_pairs = set(zip(test_df['Sequence'], test_df['SMILES']))
# ref_pairs = pd.Series(list(zip(ref_df['Sequence'], ref_df['SMILES'])))
# ref_df = ref_df[~ref_pairs.isin(test_pairs)]
# removed_count = orig_size - len(ref_df)
# print(f"Dropped {removed_count} rows from reference set")
# print(f"New reference size: {len(ref_df)}")

# -----------------------------
# Load fingerprints and residual model
# -----------------------------
alpha = 3
beta = 3
min_fraction = 0.60

with open('module_y/ref_fingerprints.pkl', "rb") as f:
    ref_fingerprints = pickle.load(f)

with open("residual_predictor/models/joint_ridge_model.pkl", "rb") as f:
    joint_ridge_model = pickle.load(f)

with open("residual_predictor/models/joint_scaler.pkl", "rb") as f:
    joint_scaler = pickle.load(f)

# -----------------------------
# Module Y: weighted transfer
# -----------------------------
def joint_transfer(sim_sequences, sim_smiles, ref_df):
    if len(sim_smiles) == 0 or sum(sim_smiles.values()) == 0:
        return np.nan, np.nan, np.nan, np.nan

    sim_smiles = {k: v / sum(sim_smiles.values()) for k, v in sim_smiles.items()}
    sim_sequences = {k: v / sum(sim_sequences.values()) for k, v in sim_sequences.items()}

    filtered_df = ref_df[
        ref_df['Sequence'].isin(sim_sequences) &
        ref_df['SMILES'].isin(sim_smiles)
    ].copy()

    if filtered_df.empty:
        return np.nan, np.nan, np.nan, np.nan

    filtered_df['seq_weight'] = filtered_df['Sequence'].map(sim_sequences)
    filtered_df['smi_weight'] = filtered_df['SMILES'].map(sim_smiles)
    filtered_df['pair_weight'] = filtered_df['seq_weight'] ** alpha * filtered_df['smi_weight'] ** beta

    max_pair_weight = filtered_df['pair_weight'].max()
    filtered_df = filtered_df[filtered_df['pair_weight'] >= min_fraction * max_pair_weight]

    total_weight = filtered_df['pair_weight'].sum()
    if total_weight < 1e-12:
        return np.nan, np.nan, np.nan, np.nan

    weighted_mean = (filtered_df['pair_weight'] * filtered_df['Value']).sum() / total_weight
    weighted_var = (filtered_df['pair_weight'] * (filtered_df['Value'] - weighted_mean)**2).sum() / total_weight
    weighted_std = np.sqrt(weighted_var)

    pw = filtered_df['pair_weight'].values
    pw_probs = pw / pw.sum()
    dominance_ratio = max_pair_weight / total_weight
    effective_n = 1 / np.sum(pw_probs ** 2) if pw_probs.size > 0 else 0.0

    return weighted_mean, weighted_std, dominance_ratio, effective_n

# -----------------------------
# Module X: batch predictions
# -----------------------------
def batch_modx_predictions(model, sequences, smiles, device, max_seq_len, max_smi_len, batch_size=64):
    prot_encoded = torch.from_numpy(np.stack(
        [label_sequence(seq, CHARPROTSET, max_seq_len) for seq in sequences])
    )
    smi_encoded = torch.from_numpy(np.stack(
        [label_smiles(smi, CHARISOSMISET, max_smi_len) for smi in smiles])
    )

    dataset = TensorDataset(smi_encoded, prot_encoded)
    loader = DataLoader(dataset, batch_size=batch_size)
    preds = []

    model.eval()
    with torch.no_grad():
        for smi_batch, prot_batch in loader:
            smi_batch, prot_batch = smi_batch.to(device), prot_batch.to(device)
            batch_pred = model(smi_batch, prot_batch)
            preds.extend(batch_pred.cpu().numpy().flatten())

    return np.array(preds)

# -----------------------------
# Per-row processing
# -----------------------------
def process_row(row, module_x_pred, blast_score, ligand_similarity, module_x_only, has_ground_truth):
    query_seq = row['Sequence']
    query_smiles = row['SMILES']
    true_value = row['Value'] if has_ground_truth else None

    with tempfile.TemporaryDirectory() as tmp_dir:

        config = MMseqsConfig(query_dir=tmp_dir)

        sim_protein_count, sim_sequences = get_similar_proteins(query_seq, ref_df, config, blast_score)
        sim_ligand_count, sim_smiles = get_similar_ligands(query_smiles, ref_fingerprints, ligand_similarity)

        modx_pred = module_x_pred[row.name]

        filtered_df = ref_df[
            ref_df['Sequence'].isin(sim_sequences) &
            ref_df['SMILES'].isin(sim_smiles)
        ]
        pair_count = len(filtered_df)

        if not filtered_df.empty and not module_x_only:
            weighted_mean, uncertainty, dominance_ratio, effective_n = joint_transfer(sim_sequences, sim_smiles, ref_df)
            y_feats = pd.DataFrame([{
                'Joint Uncertainty': uncertainty,
                'Joint Dominance Ratio': dominance_ratio,
                'Joint Effective Neighbors': effective_n
            }])

            X_scaled = joint_scaler.transform(y_feats)
            residual = joint_ridge_model.predict(X_scaled)[0]
            residual = np.tanh(residual)
            avg_fe = weighted_mean + residual
            valid_pair = 1
            mod = 'y-joint'
        else:
            avg_fe = modx_pred
            weighted_mean = np.nan
            valid_pair = 0
            mod = 'x'

    result = {
        'Sequence': query_seq,
        'SMILES': query_smiles,
        'Weighted Mean': weighted_mean,
        'Module Y Pred': avg_fe if valid_pair else None,
        'Module X Pred': modx_pred,
        'Predicted Free Energy': avg_fe,
        'Module': mod,
        'Pair Count': pair_count,
        'Protein Count': sim_protein_count,
        'Ligand Count': sim_ligand_count
    }
    
    # Only add ground truth if it exists
    if has_ground_truth:
        result['Actual Free Energy'] = true_value
    
    return result, valid_pair

# -----------------------------
# Main
# -----------------------------
def main(blast_score_values, ligand_similarity_values, module_x_only, output_file, model, device,
         MAX_SEQ_LEN, MAX_SMI_LEN, batch_size=64):

    # Check if ground truth column exists
    has_ground_truth = 'Value' in test_df.columns
    if has_ground_truth:
        print("Ground truth 'Value' column detected. Evaluation metrics will be computed.")
    else:
        print("No 'Value' column detected. Predictions will be generated without evaluation metrics.")

    # Batch Module X predictions
    print("Batch predicting Module X...")
    module_x_preds = batch_modx_predictions(model, test_df['Sequence'], test_df['SMILES'],
                                            device, MAX_SEQ_LEN, MAX_SMI_LEN, batch_size)


    for blast_score in blast_score_values:
        for ligand_similarity in ligand_similarity_values:
            total_valid_pairs = 0
            print(f"Blast Score: {blast_score}, Ligand Similarity: {ligand_similarity}")

            with ThreadPoolExecutor() as executor:
                process_partial = partial(process_row,
                                        module_x_pred=module_x_preds,
                                        blast_score=blast_score,
                                        ligand_similarity=ligand_similarity,
                                        module_x_only=module_x_only,
                                        has_ground_truth=has_ground_truth)

                results_valid = list(tqdm(executor.map(process_partial, [row for _, row in test_df.iterrows()]),
                                            total=len(test_df), desc="Processing", ncols=100))
                results, valid_pairs = zip(*results_valid)
                final_results = pd.DataFrame(results)
                os.makedirs('predictions', exist_ok=True)
                filename = output_file if output_file else f'predictions/{stem}_predictions.csv'
                final_results.to_csv(filename, index=False)

                total_valid_pairs += sum(valid_pairs)
                print(f"Total valid pairs so far: {total_valid_pairs}/{test_size}")
    

    # Compute metrics only if ground truth is available
    if has_ground_truth:
        print("\nComputing evaluation metrics...")
        dfs = load_data('predictions')
        display_metrics(dfs)
    else:
        print("\nSkipping evaluation metrics (no ground truth 'Value' column found).")

# -----------------------------
# Entry point
# -----------------------------
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='UCBBind batch Module X + Module Y predictions')
    parser.add_argument('--blast_score_values', nargs='+', type=int, default=[0.30])
    parser.add_argument('--ligand_similarity_values', nargs='+', type=float, default=[0.60])
    parser.add_argument('--module_x_only', action='store_true')
    parser.add_argument('--output_file', type=str, default=None)
    parser.add_argument('--batch_size', type=int, default=64)
    args = parser.parse_args()

    MAX_SMI_LEN = 100
    MAX_SEQ_LEN = 1200

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = AttentionDTA().to(device)
    model.load_state_dict(torch.load('module_x/training/valid_best_checkpoint.pth', weights_only=True))
    model.eval()

    main(args.blast_score_values, args.ligand_similarity_values, args.module_x_only, args.output_file,
         model, device, MAX_SEQ_LEN, MAX_SMI_LEN, batch_size=args.batch_size)

