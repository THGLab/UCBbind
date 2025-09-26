import subprocess
from Bio.Seq import Seq
from Bio.SeqRecord import SeqRecord
import pandas as pd
import os
import glob
from mmseqs_config import MMseqsConfig

# ---------------- Functions ----------------

def sequence_to_database(sequence, fasta_output, queryDB):
    os.makedirs(os.path.dirname(queryDB), exist_ok=True)
    seq = Seq(sequence)
    record = SeqRecord(seq, id="query protein", description="converted from input sequence")

    with open(fasta_output, "w") as f:
        f.write(record.format("fasta"))

    create_db_cmd = ['mmseqs', 'createdb', fasta_output, queryDB]
    subprocess.run(create_db_cmd, check=True, stdout=subprocess.DEVNULL)
    os.remove(fasta_output)

def remove_existing_files(pattern_prefix):
    files_to_remove = glob.glob(pattern_prefix + "*")
    for file in files_to_remove:
        if os.path.exists(file):
            os.remove(file)

def similar_protein_search(queryDB, refDB, resultDB, tmp, output_m8):
    remove_existing_files(resultDB)
    os.makedirs(os.path.dirname(resultDB), exist_ok=True)

    search_cmd = ['mmseqs', 'search', queryDB, refDB, resultDB, tmp]
    subprocess.run(search_cmd, check=True) #, stdout=subprocess.DEVNULL)

    format_cmd = ['mmseqs', 'convertalis', queryDB, refDB, resultDB, output_m8]
    subprocess.run(format_cmd, check=True) #, stdout=subprocess.DEVNULL)

def parse_results(output_file, threshold, ref_set):
    print("Parsing .m8 file at:", output_file)
    results = {}
    with open(output_file, 'r') as file:
        for line in file:
            columns = line.split()
            uniprot = columns[1]
            seq_identity = float(columns[2])
            if seq_identity >= threshold:
                results[uniprot] = seq_identity

    sim_sequences = {}
    ref_dict = dict(zip(ref_set["Uniprot ID"], ref_set['Sequence']))
    for uniprot in results.keys():
        if uniprot in ref_dict:
            sim_sequences[ref_dict[uniprot]] = results[uniprot]
    return len(sim_sequences), sim_sequences

def get_similar_proteins(query_seq, ref_df, config, threshold, output_m8):
    sequence_to_database(query_seq, config.fasta_file, config.queryDB)
    similar_protein_search(config.queryDB, config.refDB, config.resultDB, config.tmp, output_m8)
    count, sim_sequences = parse_results(output_m8, threshold, ref_df)
    return count, sim_sequences

# ---------------- Main workflow ----------------

if __name__ == '__main__':
    query_seq = 'GSGFRKMAFPSGKVEGCMVQVTCGTTTLNGLWLDDVVYCPRHVICTSEDMLNPNYEDLLIRKSNHNFLVQAGNVQLRVIGHSMQNCVLKLKVDTANPKTPKYKFVRIQPGQTFSVLACYNGSPSGVYQCAMRPNFTIKGSFLNGSCGSVGFNIDYDCVSFCYMHHMELPTGVHAGTDLEGNFYGPFVDRQTAQAAGTDTTITVNVLAWLYAAVINGDRWFLNRFTTTLNDFNLVAMKYNYEPLTQDHVDILGPLSAQTGIAVLDMCASLKELLQNGMNGRTILGSALLEDEFTPFDVVRQCSGVTFQ'

    df = pd.read_csv('/global/scratch/users/justinpurnomo/ucbbind/datasets/BindingDB_FINAL.csv')
    ref_df = df[df['Split'].isin(['Train', 'Val'])]

    # Persistent directory to store all MMseqs files
    persistent_dir = "mmseqs_output"
    os.makedirs(persistent_dir, exist_ok=True)

    # Configure all MMseqs paths to live in this folder
    config = MMseqsConfig(query_dir=persistent_dir)

    # Full path for .m8 output
    output_m8 = os.path.join(persistent_dir, "results.m8")

    count, sim_sequences = get_similar_proteins(query_seq, ref_df, config, threshold=0.4, output_m8=output_m8)
    print(count, sim_sequences)

