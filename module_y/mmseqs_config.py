import os

class MMseqsConfig:
    def __init__(self, query_dir, base_dir="module_y/mmseqs"):
        self.query_dir = query_dir  # Use provided temp directory

        self.fasta_file = os.path.join(self.query_dir, "output.fasta")
        self.queryDB = os.path.join(self.query_dir, "queryDB")
        self.resultDB = os.path.join(self.query_dir, "resultDB")
        self.tmp = os.path.join(self.query_dir, "tmp")
        self.output_file = f"{self.resultDB}.m8"

        self.refDB = os.path.join(base_dir, "refDB", "refDB")

