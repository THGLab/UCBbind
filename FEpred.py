import os
import time
import tempfile
import concurrent.futures 
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor
import argparse
import contextlib
import sys
import re
import pickle

from functools import partial
from rdkit import RDLogger
from tqdm import tqdm

RDLogger.DisableLog('rdApp.*')

import pandas as pd
import numpy as np

from module_y.mmseqs_config import MMseqsConfig
from module_y.sequencesearch import get_similar_proteins
from module_y.smilessearch import get_similar_ligands
from metrics import load_data, calculate_metrics, display_metrics

from module_x.dataset import label_smiles, label_sequence, CHARISOSMISET, CHARPROTSET
from module_x.model import AttentionDTA
import torch, json

ref_fp = 'datasets/BindingDB.csv'
ref_df = pd.read_csv(ref_fp)
orig_size = ref_df.shape[0]

test_fp = 'datasets/Moonshot_new.csv'
test_df = pd.read_csv(test_fp)
test_size = test_df.shape[0]

test_pairs = set(zip(test_df['Sequence'], test_df['SMILES']))
ref_pairs = pd.Series(list(zip(ref_df['Sequence'], ref_df['SMILES'])))
ref_df = ref_df[~ref_pairs.isin(test_pairs)]
removed_count = orig_size - len(ref_df)
print(f"Dropped {removed_count} rows from reference set")
print(f"New reference size: {len(ref_df)}")

ref_pkl = 'module_y/ref_fingerprints.pkl'

with open(ref_pkl, "rb") as f:
    ref_fingerprints = pickle.load(f)

with open("residual_predictor/models/joint_ridge_model.pkl", "rb") as f:
    joint_ridge_model = pickle.load(f)

with open("residual_predictor/models/joint_scaler.pkl", "rb") as f:
    joint_scaler = pickle.load(f)

def joint_transfer(sim_sequences, sim_smiles, ref_df):

    if len(sim_smiles) == 0 or sum(sim_smiles.values()) == 0:
        raise ValueError("SMILES similarity dictionary is empty or sums to zero")

    # Normalize similarities
    sim_smiles = {k: v / sum(sim_smiles.values()) for k, v in sim_smiles.items()}
    sim_sequences = {k: v / sum(sim_sequences.values()) for k, v in sim_sequences.items()}

    # Filter ref_df to only relevant rows
    filtered_df = ref_df[
        ref_df['Sequence'].isin(sim_sequences) &
        ref_df['SMILES'].isin(sim_smiles)
    ].copy()

    if filtered_df.empty:
        return np.nan, np.nan, np.nan, np.nan  # Add placeholders for new metrics

    # Add similarity weights
    filtered_df['seq_weight'] = filtered_df['Sequence'].map(sim_sequences)
    filtered_df['smi_weight'] = filtered_df['SMILES'].map(sim_smiles)
    filtered_df['pair_weight'] = filtered_df['seq_weight'] + filtered_df['smi_weight']

    # Compute weighted average
    total_weight = filtered_df['pair_weight'].sum()
    weighted_mean = (filtered_df['pair_weight'] * filtered_df['Value']).sum() / total_weight if total_weight > 0 else np.nan

    # Weighted variance (uncertainty)
    weighted_var = (filtered_df['pair_weight'] * (filtered_df['Value'] - weighted_mean)**2).sum() / total_weight if total_weight > 0 else np.nan
    weighted_std = np.sqrt(weighted_var)

    pw = filtered_df['pair_weight'].values
    pw_probs = pw / pw.sum()

    max_pair_weight = filtered_df['pair_weight'].max()
    dominance_ratio = max_pair_weight / total_weight
    effective_n = 1 / np.sum(pw_probs ** 2) if pw_probs.size > 0 else 0.0

    return (
        weighted_mean,
        weighted_std,
        dominance_ratio,
        effective_n
    )

def modx(sequence, smiles, model, device):
    # Process the protein and ligand for model input
    compound_encoded = torch.from_numpy(label_smiles(smiles, CHARISOSMISET, MAX_SMI_LEN)).unsqueeze(0).to(device)
    protein_encoded = torch.from_numpy(label_sequence(sequence, CHARPROTSET, MAX_SEQ_LEN)).unsqueeze(0).to(device)

    with torch.no_grad():
        y_pred = model(compound_encoded, protein_encoded)

    return y_pred.item()

def ligand_transfer(sim_smiles, ref_df):
    sim_series = pd.Series(sim_smiles)
    ligand_mean_similarity = sim_series.mean()
    ligand_dominance_ratio = sim_series.max()/sim_series.sum()
    return ligand_mean_similarity, ligand_dominance_ratio

# Function to process each protein-ligand pair
def process_row(row, blast_score, ligand_similarity, k, module_x_only, model, device):

    query_seq = row['Sequence']
    query_smiles = row['SMILES']
    true_value = row['Value']
    binder_status = row['Binder_Status']

    dominance_ratio = 0

    with tempfile.TemporaryDirectory() as tmp_dir:
        
        config = MMseqsConfig(query_dir=tmp_dir)
        sim_protein_count, sim_sequences = get_similar_proteins(query_seq, ref_df, config, blast_score)
        sim_ligand_count, sim_smiles = get_similar_ligands(query_smiles, ref_fingerprints, ligand_similarity)

        modx_pred = modx(query_seq, query_smiles, model, device)
        filtered_df = ref_df[
            ref_df['Sequence'].isin(sim_sequences) &
            ref_df['SMILES'].isin(sim_smiles)
        ]
        
        if not filtered_df.empty and not module_x_only:
            weighted_mean, uncertainty, dominance_ratio, effective_n = joint_transfer(sim_sequences, sim_smiles, ref_df)
            y_feats = pd.DataFrame([{
                'Joint Uncertainty': uncertainty,
                'Joint Dominance Ratio': dominance_ratio,
                'Joint Effective Neighbors': effective_n
                }])

            X_scaled = joint_scaler.transform(y_feats)
            residual = joint_ridge_model.predict(X_scaled)[0]
            residual = 3.1*np.tanh(0.22*residual)
            avg_fe = weighted_mean + residual
            valid_pair = 1
            mod = 'y-joint'

        else:
            if sim_smiles:
                sim_series = pd.Series(sim_smiles)
                dominance_ratio = sim_series.max()/sim_series.sum()
                if dominance_ratio < 0.2:
                    df_sim_smi = ref_df[ref_df['SMILES'].isin(sim_smiles)]
                    df_sim_smi = df_sim_smi.copy()
                    df_sim_smi['smi_weight'] = df_sim_smi['SMILES'].map(sim_smiles)
                    total_weight = df_sim_smi['smi_weight'].sum()
                    weighted_mean = (df_sim_smi['Value'] * df_sim_smi['smi_weight']).sum() / total_weight
                    avg_fe = weighted_mean
                    valid_pair = 1
                    mod = 'y-ligand'
                else: 
                    weighted_mean = np.nan
                    avg_fe = np.nan
                    valid_pair = 0
                    mod = 'x'
            else: 
                weighted_mean = np.nan
                avg_fe = np.nan
                valid_pair = 0
                mod = 'x'
    
    data = {
        'Sequence': [query_seq],
        'SMILES': [query_smiles],
        'Weighted Mean': [weighted_mean] if valid_pair else None,
        'Module Y Pred': [avg_fe] if valid_pair else None,
        'Module X Pred': [modx_pred],
        'Actual Free Energy': [true_value],
        'Predicted Free Energy': [avg_fe] if valid_pair else [modx_pred],
        'Module': [mod],
        'Binder Status': [binder_status]
    }

    return pd.DataFrame(data), valid_pair

def write_to_csv(df, blast_score, ligand_similarity, blast_score_values, ligand_similarity_values, output_file):
    os.makedirs('predictions', exist_ok=True)
    if len(blast_score_values)== 1 and len(ligand_similarity_values)==1:
        filename = output_file if output_file else f'predictions/new_predictions.csv'
    else:
        filename = output_file if output_file else f'predictions/predictions_P{blast_score}L{int(ligand_similarity * 100)}.csv'
    df.to_csv(filename, mode='w', header=True, index=False)

def main(blast_score_values, ligand_similarity_values, k, module_x_only, output_file, model, device):

    for blast_score in blast_score_values:
        for ligand_similarity in ligand_similarity_values:
            total_valid_pairs = 0
            print(f"Blast Score: {blast_score}, Ligand Similarity: {ligand_similarity}")


            with ThreadPoolExecutor() as executor:
                process_row_partial = partial(process_row, blast_score=blast_score, ligand_similarity=ligand_similarity, 
                        k=k, module_x_only=module_x_only, model=model, device=device)
                results, valid_pair_count = zip(*tqdm(executor.map(process_row_partial, [row for _, row in test_df.iterrows()]),
                                            total=len(test_df),
                                            desc = "Progress",
                                            ncols=100))            
            final_results = pd.concat(results, ignore_index=True)
            write_to_csv(final_results, blast_score, ligand_similarity, blast_score_values, ligand_similarity_values, output_file)
            total_valid_pairs += sum(valid_pair_count)
            print(f"Total valid pairs so far: {total_valid_pairs}/{test_size}")
    folder = 'predictions'
    dfs = load_data(folder)
    display_metrics(dfs)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Process data with varying MMseqs2 bit scores and ligand similarity values')
    parser.add_argument('--blast_score_values', nargs='+', type=int, default=[0.40], help = 'list of MMseqs2 bit scores')
    parser.add_argument('--ligand_similarity_values', nargs='+', type=float, default=[0.80], help='list of ligand similarity values')
    parser.add_argument('--num_nearest_neighbors', type=int, default=30, help='maximum number of ligand nearest neighbors')
    parser.add_argument('--module_x_only', action='store_true', help='If true, bypass similarity checks and use module_x only')
    parser.add_argument('--output_file', type=str, default=None, help='custom output file name')
    args = parser.parse_args()

    start_time = time.time()

    # convert the smiles to one-hot encoding; CHANGE TO YOUR OWN PATH OF YOUR BEST MODEL
    smilelen, seqlen = 200, 2000
    MAX_SMI_LEN = 100
    MAX_SEQ_LEN = 1200
    # load model
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = AttentionDTA().cuda()
    model.load_state_dict(torch.load('module_x/training/valid_best_checkpoint.pth', weights_only=True))
    model.eval()
    
    valid_pair_count = 0 
    main(args.blast_score_values, args.ligand_similarity_values, args.num_nearest_neighbors, args.module_x_only, args.output_file, model, device)

    end_time = time.time()
    elapsed_time = end_time - start_time
    print(f"Elapsed time: {elapsed_time:.4f} seconds")
