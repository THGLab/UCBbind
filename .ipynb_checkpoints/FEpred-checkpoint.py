import os
import time
import tempfile
import concurrent.futures 
from concurrent.futures import ProcessPoolExecutor
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

from mmseqs_config import MMseqsConfig
from sequencesearch import get_similar_proteins
from smilessearch import get_similar_ligands
from metrics import load_data, calculate_metrics, display_metrics


from model import DeepDTA
import torch, json

file_path = '/global/scratch/users/justinpurnomo/ucbbind/datasets/BindingDB_FINAL.csv'
df = pd.read_csv(file_path)

#test_df = df[df["split"] == 'test']
test_df = pd.read_csv('../../datasets/moonshot_binders.csv')
test_df = test_df[test_df['split']=='test']
#test_df = test_df[:5]
test_size = test_df.shape[0]

ref_df = df[df["Split"].isin(['Train', 'Val'])]

df['Value'] = pd.to_numeric(df['Value'], errors='coerce')

ref_pkl = 'ref_fingerprints.pkl'

with open(ref_pkl, "rb") as f:
    ref_fingerprints = pickle.load(f)

def fe_prediction(sim_sequences, sim_smiles, ref_df):

    # Normalize similarities
    sim_smiles = {k: v / sum(sim_smiles.values()) for k, v in sim_smiles.items()}
    sim_sequences = {k: v / sum(sim_sequences.values()) for k, v in sim_sequences.items()}

    # Filter ref_df to only relevant rows
    filtered_df = ref_df[
        ref_df['Sequence'].isin(sim_sequences) &
        ref_df['SMILES'].isin(sim_smiles)
    ].copy()

    if filtered_df.empty:
        return np.nan

    # Add similarity weights
    filtered_df['seq_weight'] = filtered_df['Sequence'].map(sim_sequences)
    filtered_df['smi_weight'] = filtered_df['SMILES'].map(sim_smiles)
    filtered_df['pair_weight'] = filtered_df['seq_weight'] + filtered_df['smi_weight']

    # Compute weighted average
    total_value = (filtered_df['pair_weight'] * filtered_df['Value']).sum()
    total_weight = filtered_df['pair_weight'].sum()

    return total_value / total_weight if total_weight > 0 else np.nan


def modx(protein, ligand):
    # Process the protein and ligand for model input
    protein = [protein_dict.get(x, protein_dict['dummy']) for x in protein] + [protein_dict['dummy']] * (seqlen - len(protein))
    ligand = [ligand_dict[x] for x in ligand] + [ligand_dict['dummy']] * (smilelen - len(ligand))

    # Convert to torch tensors
    ligand = torch.tensor(ligand).unsqueeze(0)
    protein = torch.tensor(protein).unsqueeze(0)

    # Perform model inference
    with torch.no_grad():
        result = model(protein, ligand).item()
    return result

# Function to process each protein-ligand pair
def process_row(row, blast_score, ligand_similarity, k, module_x_only):
    # Ensure row contains valid data
    if pd.isna(row['Sequence']) or pd.isna(row['SMILES']) or pd.isna(row['Value']):
        print(f"Skipping row due to missing data: {row}")
        return None  # Return None if we want to skip this row

    #print('Row being processed!')

    #pdb = row['PDB']
    identifier = row['Unnamed: 0']
    query_seq = row['Sequence']
    query_smiles = row['SMILES']
    true_value = row['Value']
    
    with tempfile.TemporaryDirectory() as tmp_dir:
        
        config = MMseqsConfig(query_dir=tmp_dir)
        # Extract the necessary values from the config object
        sim_protein_count, sim_sequences = get_similar_proteins(query_seq, ref_df, config)
        sim_ligand_count, sim_smiles = get_similar_ligands(query_smiles, ref_fingerprints, ligand_similarity)
        #print(f'sim_protein_count: {sim_protein_count}')
        #print(f'sim_ligand_count: {sim_ligand_count}')
        if (sim_protein_count > 0 and sim_ligand_count > 0) and not module_x_only:
            avg_fe = fe_prediction(sim_sequences, sim_smiles, ref_df)
            if pd.isna(avg_fe):
                avg_fe = modx(query_seq, query_smiles)
                valid_pair = 0
                mod = 'x'
            else:
                valid_pair = 1
                mod = 'y'
        else: 
            avg_fe = modx(query_seq, query_smiles)
            valid_pair = 0
            mod = 'x'

        true_value = row['Value']  # Actual free energy
        relative_diff = None
        if true_value != 0:
            relative_diff = abs(avg_fe - true_value) / true_value
    
    data = {
        'identifier': [identifier],
        'sequence': [query_seq],
        'smiles': [query_smiles],
        'Predicted Free Energy': [avg_fe],
        'Actual Free Energy': [true_value],
        'Module': [mod]
        #'Number of Proteins Used to Average': [sim_protein_count],  # Placeholder for count of proteins used
        #'Number of Ligands Used to Average': [sim_ligand_count]   # Placeholder for count of ligands used
    }

    return pd.DataFrame(data), valid_pair

def write_to_csv(df, blast_score, ligand_similarity, blast_score_values, ligand_similarity_values, output_file):
    os.makedirs('predictions', exist_ok=True)
    if len(blast_score_values)== 1 and len(ligand_similarity_values)==1:
        filename = output_file if output_file else f'predictions/predictions.csv'
    else:
        filename = output_file if output_file else f'predictions/predictions_P{blast_score}L{int(ligand_similarity * 100)}.csv'
    df.to_csv(filename, mode='w', header=True, index=False)

def main(blast_score_values, ligand_similarity_values, k, module_x_only, output_file):

    for blast_score in blast_score_values:
        for ligand_similarity in ligand_similarity_values:
            total_valid_pairs = 0
            print(f"Blast Score: {blast_score}, Ligand Similarity: {ligand_similarity}")


            with concurrent.futures.ProcessPoolExecutor() as executor:
                process_row_partial = partial(process_row, blast_score=blast_score, ligand_similarity=ligand_similarity, k=k, module_x_only=module_x_only)
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
    parser = argparse.ArgumentParser(description='Process data with varying BLAST score and ligand similarity values')
    parser.add_argument('--blast_score_values', nargs='+', type=int, default=[250], help = 'list of BLAST score values')
    parser.add_argument('--ligand_similarity_values', nargs='+', type=float, default=[0.80], help='list of ligand similarity values')
    parser.add_argument('--num_nearest_neighbors', type=int, default=30, help='maximum number of ligand nearest neighbors')
    parser.add_argument('--module_x_only', action='store_true', help='If true, bypass similarity checks and use module_x only')
    parser.add_argument('--output_file', type=str, default=None, help='custom output file name')
    args = parser.parse_args()

    start_time = time.time()

    # convert the smiles to one-hot encoding; CHANGE TO YOUR OWN PATH OF YOUR BEST MODEL
    ligand_dict = json.load(open('module_x/ligand_dict-prk12-ldk8.json'))
    protein_dict = json.load(open('module_x/protein_dict-prk12-ldk8.json'))
    smilelen, seqlen = 200, 2000

    # load model
    model = DeepDTA(len(protein_dict)+1, len(ligand_dict)+1, 32, 12, 8)
    model.load_state_dict(torch.load('module_x/deepdta_retrain-prk12-ldk8.pt', weights_only = True))
    model.eval()
    valid_pair_count = 0 
    main(args.blast_score_values, args.ligand_similarity_values, args.num_nearest_neighbors, args.module_x_only, args.output_file)

    end_time = time.time()
    elapsed_time = end_time - start_time
    print(f"Elapsed time: {elapsed_time:.4f} seconds")
