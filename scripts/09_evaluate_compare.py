"""
Step 8 - the final comparison: for the SAME target triples, compute MRR /
Hits@1 / Hits@10 under the clean model and under each poisoned model, and
report the relative degradation, exactly as in Tables 4/5 of the paper.

Usage:
    python scripts/09_evaluate_compare.py \
        --targets out/targets.tsv \
        --clean_emb out/clean --clean_dir data/UMLS_clean \
        --runs symmetry_truth:out/poisoned_symmetry_truth:data/UMLS_poisoned_symmetry_truth \
               inversion_truth:out/poisoned_inversion_truth:data/UMLS_poisoned_inversion_truth \
        --out_csv results/comparison.csv --out_png results/comparison.png
"""
import sys
sys.path.insert(0, "src")
import argparse
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from kge_utils import read_triples, load_embeddings, build_known_index, evaluate_triples


def eval_run(emb_prefix, dataset_dir, targets):
    ent_emb, rel_emb, ent2id, rel2id = load_embeddings(
        f"{emb_prefix}_entity_embeddings.csv", f"{emb_prefix}_relation_embeddings.csv"
    )
    train = read_triples(f"{dataset_dir}/train.txt")
    valid = read_triples(f"{dataset_dir}/valid.txt")
    test = read_triples(f"{dataset_dir}/test.txt")

    def to_ids(triples):
        return [(ent2id[s], rel2id[r], ent2id[o]) for s, r, o in triples
                if s in ent2id and r in rel2id and o in ent2id]

    all_ids = to_ids(train) + to_ids(valid) + to_ids(test)
    sr2o, ro2s = build_known_index(all_ids)
    target_ids = to_ids(targets)
    metrics, _ = evaluate_triples(target_ids, ent_emb, rel_emb, sr2o, ro2s)
    return metrics


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--targets", required=True)
    ap.add_argument("--clean_emb", required=True)
    ap.add_argument("--clean_dir", required=True)
    ap.add_argument("--runs", nargs="+", required=True,
                     help="label:emb_prefix:dataset_dir triples, e.g. "
                          "symmetry_truth:out/poisoned_symmetry_truth:data/UMLS_poisoned_symmetry_truth")
    ap.add_argument("--out_csv", default="results/comparison.csv")
    ap.add_argument("--out_png", default="results/comparison.png")
    args = ap.parse_args()

    targets = read_triples(args.targets)

    rows = []
    clean_metrics = eval_run(args.clean_emb, args.clean_dir, targets)
    rows.append({"run": "Original (clean)", **clean_metrics})

    for spec in args.runs:
        label, emb_prefix, dataset_dir = spec.split(":")
        m = eval_run(emb_prefix, dataset_dir, targets)
        m["MRR_drop_%"] = 100.0 * (clean_metrics["MRR"] - m["MRR"]) / clean_metrics["MRR"]
        rows.append({"run": label, **m})

    df = pd.DataFrame(rows)
    df.to_csv(args.out_csv, index=False)
    print(df.to_string(index=False))

    fig, ax = plt.subplots(figsize=(8, 4.5))
    ax.bar(df["run"], df["MRR"], color=["#4c72b0"] + ["#c44e52"] * (len(df) - 1))
    ax.set_ylabel("MRR on target triples")
    ax.set_title("Effect of poisoning attacks on target-triple MRR (TransE / UMLS)")
    plt.xticks(rotation=30, ha="right")
    plt.tight_layout()
    plt.savefig(args.out_png, dpi=150)
    print(f"\nSaved table -> {args.out_csv}")
    print(f"Saved chart -> {args.out_png}")


if __name__ == "__main__":
    main()
