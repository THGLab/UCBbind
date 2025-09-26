import pandas as pd
import subprocess
import os
import time
import pickle
import shutil
from rdkit import Chem
from rdkit.Chem import AllChem, DataStructs
import numpy as np
from rdkit import RDLogger

def make_protein_db(df, output_fasta_file, db_name, tmp):

    fasta_entries = []
    processed_uniprots = {}

    # Generate the FASTA file content

    for _, row in df.iterrows():
        
        uniprot_id = row['Uniprot ID']
        sequence = row['Sequence']

        if uniprot_id not in processed_uniprots or len(sequence) > len(processed_uniprots[uniprot_id]):
            processed_uniprots[uniprot_id] = sequence
    
    for uniprot_id, sequence in processed_uniprots.items():
        fasta_entry = f">{uniprot_id}\n{sequence}"
        fasta_entries.append(fasta_entry)
    
    with open(output_fasta_file, 'w') as fasta_file:
        fasta_file.write('\n'.join(fasta_entries))

    print(f"FASTA file created: {output_fasta_file}")

    os.makedirs(os.path.dirname(db_name), exist_ok=True)

    # Create MMseqs database from the generated FASTA file

    subprocess.run(["mmseqs", "createdb", output_fasta_file, db_name], check=True, stdout=subprocess.DEVNULL)
    print(f"Protein database {db_name} has been created and contains {len(processed_uniprots)} proteins")
    os.makedirs(tmp, exist_ok=True)
    subprocess.run(["mmseqs", "createindex", db_name, tmp], check=True,  stdout=subprocess.DEVNULL)
    os.remove(output_fasta_file)

def make_ligand_db(df, pkl_file, radius=2, n_bits=1024):

    fingerprints = {}
    unique_smiles = df['SMILES'].unique()

    for smiles in unique_smiles:
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            continue

        fp = AllChem.GetMorganFingerprintAsBitVect(mol, radius, nBits=n_bits)
        fingerprints[smiles] = fp

    with open(pkl_file, "wb") as f:
        pickle.dump(fingerprints, f, protocol=pickle.HIGHEST_PROTOCOL)

    print(f"Fingerprints saved to {pkl_file}")
    
    return fingerprints

if __name__ == '__main__':

    start_time = time.time()

    df = pd.read_csv('datasets/BindingDB.csv')
    ref_df = df[df["Split"].isin(['Train', 'Val'])]
    ref_pairs = pd.Series(list(zip(ref_df['Sequence'], ref_df['SMILES'])))
    orig_size = len(ref_df)

    test_df = pd.read_csv('datasets/Moonshot.csv')
    test_pairs = set(zip(test_df['Sequence'], test_df['SMILES']))
    ref_df = ref_df[~ref_pairs.isin(test_pairs)]

    removed_count = orig_size - len(ref_df)
    print(f"Dropped {removed_count} rows from reference set")
    print(f"New reference size: {len(ref_df)}")

    fasta_file = "ref_database.fasta"
    db_name = "module_y/mmseqs/refDB/refDB"
    tmp = "module_y/mmseqs/tmp"
    pkl_file = "module_y/ref_fingerprints.pkl"
    RDLogger.DisableLog('rdApp.*') 

    folder = "module_y/mmseqs"
    if os.path.exists(folder):
        shutil.rmtree(folder)
        print(f"Deleted existing protein database {folder}")

    make_protein_db(ref_df, fasta_file, db_name, tmp)
    make_ligand_db(ref_df, pkl_file)

    end_time = time.time()
    elapsed_time = end_time - start_time
    print(f'{elapsed_time:.2f} seconds to create the reference protein and ligand databases')
