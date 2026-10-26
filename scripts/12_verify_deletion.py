"""
Step 12: prove the deletion really happened.
"""
import os, sys

ROOT = os.path.join(os.path.dirname(__file__), "..")

def load_set(path):
    with open(path) as f:
        return set(l.rstrip("\n") for l in f if l.strip())

original = load_set(os.path.join(ROOT, "data/UMLS_clean/train.txt"))
new      = load_set(os.path.join(ROOT, "data/UMLS_deleted/train.txt"))
deleted  = load_set(os.path.join(ROOT, "out/del_deletion.tsv"))

ok = True
for t in deleted:
    if t in new:
        print("ERROR: claimed deleted but still present:", t); ok = False
    if t not in original:
        print("ERROR: claimed deleted but never existed:", t); ok = False

expected = len(original) - len(deleted)
print(f"original={len(original)}  deleted={len(deleted)}  new={len(new)}  expected={expected}")
if len(new) != expected:
    print("ERROR: size mismatch"); ok = False

print("ALL CHECKS PASSED" if ok else "CHECKS FAILED")
sys.exit(0 if ok else 1)
