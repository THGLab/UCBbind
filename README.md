# UCBbind

UCBbind is a program for predicting binding affinities for protein-ligand pairs. The program implements two modules:

1. **Module Y (Transfer prediction module):** Uses sequence alignment and Tanimoto similarity to select reference protein-ligand pairs and replicate their binding free energies.
2. **Module X (Deep learning module):** Takes features extracted from protein sequences and ligand SMILES strings to predict binding affinities.

## Authors

Justin Purnomo, Caitlin Kim, Kunyang Sun, Yingze Wang, and Teresa Head-Gordon

## Getting Started

This environment can be built via: ``conda env create -f env.yml``

## Training

To train Module X, run ``python X_prep.py``
To train Module Y, run ``python Y_prep.py``

Note: The trained Module X has already been provided. Module Y requires large `.idx` and `.pkl` files and the BindingDB dataset, which are not included in the repo due to size. You can download the cleaned BindingDB dataset here:

- [Figshare](https://figshare.com/articles/dataset/BindingDB/30234775?file=60150230)

After downloading ``BindingDB.csv`` and placing in the ``datasets`` folder, you can train Module Y. 

## Prediction

Predictions can be run using `python FEpred.py`. 

The script expects a CSV file with the following columns: `Sequence`, `SMILES`, and `Value`. These describe the protein sequence of the query, the ligand SMILES of the query, and the experimental binding free energy in positive kcal/mol.

The default test set used in `FEpred.py` is: `test_fp = 'datasets/PDBbind.csv'. Note that in ``Y_prep.py``, rows in BindingDB that are present in the test set are filtered out for reproducibility. Users do not need to do this for normal predictions.

## Binder v Nonbinder Classification Accuracy

To assess classification accuracy, you can run ``python classifier_statistics.py``

This script calculates the binder v nonbinder classification accuracy based on a pIC50 threshold of ``5`` for the binding affinity.





