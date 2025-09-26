import os
import pickle
import numpy as np
from rdkit import Chem, DataStructs, RDLogger
from rdkit.Chem import AllChem
from collections import defaultdict

RDLogger.DisableLog('rdApp.*')

def get_query_fingerprint(smiles, radius=2, n_bits=1024):
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None
    return AllChem.GetMorganFingerprintAsBitVect(mol, radius, nBits=n_bits)

def get_similar_ligands(query_smiles, ref_fingerprints, threshold=0.7):
    query_fp = get_query_fingerprint(query_smiles)
    if query_fp is None:
        return {}
    similar_ligands = {}
    
    # Extract SMILES and fingerprints separately to maintain mapping
    smiles_list = list(ref_fingerprints.keys())
    fps_list = list(ref_fingerprints.values())

    # Use BulkTanimotoSimilarity
    similarities = DataStructs.BulkTanimotoSimilarity(query_fp, fps_list)

    # Collect only those above the threshold
    similar_ligands = {
        smiles: sim
        for smiles, sim in zip(smiles_list, similarities)
        if sim >= threshold
    }
    return len(similar_ligands), similar_ligands

if __name__ == '__main__':
    
    ref_pkl = 'ref_fingerprints.pkl'

    with open(ref_pkl, "rb") as f:
        ref_fingerprints = pickle.load(f)

    query_smiles = 'Cc1ccc(NC(=O)c2ccc(C[N@H+]3CC[N@H+](C)CC3)cc2)cc1Nc1nccc(-c2cccnc2)n1'
    csv_file = '../data/moonshot_binders_binding_data.csv'

    count, sim_smiles = get_similar_ligands(query_smiles, ref_fingerprints, threshold=0.8)
    print('sim_smiles:', sim_smiles)
    print("Count:", count)
