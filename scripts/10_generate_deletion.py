"""
Step 10: generate the deletion attack.

Reads the clean embeddings (out/*.csv), the targets (out/targets.tsv) and
the clean training triples, decides which triples to delete, and writes
the exact list to out/del_deletion.tsv.
"""
import os, sys
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
from deletion_attack import generate_deletion_triples


def load_embeddings(path):
    df = pd.read_csv(path, index_col=0)          # first column = name
    return {name: row.to_numpy(dtype=float) for name, row in df.iterrows()}

def load_triples(path):
    out = []
    with open(path) as f:
        for line in f:
            line = line.rstrip("\n")
            if line:
                out.append(tuple(line.split("\t")))
    return out


ROOT = os.path.join(os.path.dirname(__file__), "..")

ent_emb = load_embeddings(os.path.join(ROOT, "out/clean_entity_embeddings.csv"))
rel_emb = load_embeddings(os.path.join(ROOT, "out/clean_relation_embeddings.csv"))
targets = load_triples(os.path.join(ROOT, "out/targets.tsv"))
train   = load_triples(os.path.join(ROOT, "data/UMLS_clean/train.txt"))

deleted = generate_deletion_triples(targets, train, ent_emb, rel_emb, per_target=1)

out_path = os.path.join(ROOT, "out/del_deletion.tsv")
with open(out_path, "w") as f:
    for (s, r, o) in deleted:
        f.write(f"{s}\t{r}\t{o}\n")

print(f"Deletion attack: {len(deleted)} triples marked for deletion.")
print(f"Written to {out_path}")
