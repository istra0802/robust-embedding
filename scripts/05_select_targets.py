"""
Step 4 - select target triples: the subset of the test set that the clean
model already ranks <= 10 (both sides averaged), i.e. facts the model is
confident about and that an attacker wants to degrade. Mirrors paper
Section 4 ("Evaluation") / Table 3.

Usage:
    python scripts/05_select_targets.py out/clean data/UMLS_clean out/targets.tsv --max_rank 10
"""
import sys
sys.path.insert(0, "src")
import argparse
from kge_utils import read_triples, load_embeddings, build_known_index, filtered_rank_object, filtered_rank_subject


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("emb_prefix")       # out/clean
    ap.add_argument("dataset_dir")      # data/UMLS_clean
    ap.add_argument("out_file")         # out/targets.tsv
    ap.add_argument("--max_rank", type=int, default=10)
    args = ap.parse_args()

    ent_emb, rel_emb, ent2id, rel2id = load_embeddings(
        f"{args.emb_prefix}_entity_embeddings.csv", f"{args.emb_prefix}_relation_embeddings.csv"
    )
    train = read_triples(f"{args.dataset_dir}/train.txt")
    valid = read_triples(f"{args.dataset_dir}/valid.txt")
    test = read_triples(f"{args.dataset_dir}/test.txt")

    def to_ids(triples):
        out = []
        for s, r, o in triples:
            if s in ent2id and r in rel2id and o in ent2id:
                out.append((ent2id[s], rel2id[r], ent2id[o]))
        return out

    all_ids = to_ids(train) + to_ids(valid) + to_ids(test)
    sr2o, ro2s = build_known_index(all_ids)

    test_ids = to_ids(test)
    id2ent = {v: k for k, v in ent2id.items()}
    id2rel = {v: k for k, v in rel2id.items()}

    kept = 0
    with open(args.out_file, "w") as f:
        for s, r, o in test_ids:
            rs = filtered_rank_subject(s, r, o, ent_emb, rel_emb, ro2s.get((r, o), set()))
            ro = filtered_rank_object(s, r, o, ent_emb, rel_emb, sr2o.get((s, r), set()))
            if rs <= args.max_rank and ro <= args.max_rank:
                f.write(f"{id2ent[s]}\t{id2rel[r]}\t{id2ent[o]}\n")
                kept += 1

    print(f"Selected {kept}/{len(test_ids)} target triples (rank<={args.max_rank} both sides) -> {args.out_file}")


if __name__ == "__main__":
    main()
