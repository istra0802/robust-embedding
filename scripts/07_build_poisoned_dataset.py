"""
Step 6 - build the poisoned dataset: clean train.txt + adversarial triples,
deduplicated, with any triple that already exists in train filtered out
(paper Section 4, "Implementation"). valid.txt / test.txt are copied
unchanged -- attacks never touch validation/test data.

Usage:
    python scripts/07_build_poisoned_dataset.py data/UMLS_clean \
        out/adv_symmetry_truth.tsv data/UMLS_poisoned_symmetry_truth
"""
import sys
sys.path.insert(0, "src")
import os
import argparse
from kge_utils import read_triples, write_triples


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("clean_dir")
    ap.add_argument("adv_file")
    ap.add_argument("out_dir")
    args = ap.parse_args()

    train = read_triples(f"{args.clean_dir}/train.txt")
    adv = read_triples(args.adv_file)

    train_set = set(train)
    new_edits = [t for t in adv if t not in train_set]
    poisoned_train = train + new_edits

    os.makedirs(args.out_dir, exist_ok=True)
    write_triples(f"{args.out_dir}/train.txt", poisoned_train)
    for split in ("valid.txt", "test.txt"):
        with open(f"{args.clean_dir}/{split}") as fin, open(f"{args.out_dir}/{split}", "w") as fout:
            fout.write(fin.read())

    print(f"Clean train: {len(train)} triples")
    print(f"Adversarial triples added: {len(new_edits)} (of {len(adv)} generated, "
          f"{len(adv) - len(new_edits)} already existed and were skipped)")
    print(f"Poisoned train: {len(poisoned_train)} triples -> {args.out_dir}/train.txt")


if __name__ == "__main__":
    main()
