"""
Step 11: build the deleted dataset.

Removes the triples in out/del_deletion.tsv from the clean train.txt and
writes data/UMLS_deleted/train.txt. valid.txt and test.txt are copied
unchanged.
"""
import os, shutil

ROOT = os.path.join(os.path.dirname(__file__), "..")
SRC  = os.path.join(ROOT, "data/UMLS_clean")
DST  = os.path.join(ROOT, "data/UMLS_deleted")
os.makedirs(DST, exist_ok=True)

def load_lines(path):
    with open(path) as f:
        return [l.rstrip("\n") for l in f if l.strip()]

original = load_lines(f"{SRC}/train.txt")
deleted  = set(load_lines(os.path.join(ROOT, "out/del_deletion.tsv")))

kept = [l for l in original if l not in deleted]

with open(f"{DST}/train.txt", "w") as f:
    f.write("\n".join(kept) + "\n")

for extra in ("valid.txt", "test.txt"):
    src = f"{SRC}/{extra}"
    if os.path.exists(src):
        shutil.copy(src, DST)

print(f"original train : {len(original)}")
print(f"deleted        : {len(deleted)}")
print(f"new train      : {len(kept)}")
print(f"written to     : {DST}/train.txt")
