"""
Task 1 - Plausibility scores of triples, before vs after attack.

The professor asked for the plausibility score computed with dicee's own
prediction function (model.triple_score / model.predict), for triples from
BOTH the test set and the training set, and shown as an intermediate result
for 10 records each.

This script:
  * loads each trained dicee model (clean + each attack)
  * scores 10 TEST target triples and 10 TRAIN triples with every model
  * prints both tables (the 10-record intermediate result)
  * writes the full tables to out/plausibility_test.csv and
    out/plausibility_train.csv

"Plausibility score" = how strongly the model believes a triple is true.
  - clean model  -> score BEFORE any attack
  - attacked model -> score AFTER that attack
Comparing the columns shows how each attack changed the model's belief.

Usage (from the repo root, with .venv active):

    python scripts/13_plausibility_scores.py
"""
import os
import csv
from dicee import KGE


# ---- which trained models to score -------------------------------------- #
MODELS = {
    "clean":             "Experiments/UMLS_TransE_clean",
    "symmetry_truth":    "Experiments/UMLS_TransE_poisoned_symmetry_truth",
    "inversion_truth":   "Experiments/UMLS_TransE_poisoned_inversion_truth",
    "composition_truth": "Experiments/UMLS_TransE_poisoned_composition_truth",
    "deletion":          "Experiments/UMLS_TransE_deleted",
}

TEST_TARGETS_FILE = "out/targets.tsv"          # test triples the clean model ranks <=10
TRAIN_FILE        = "data/UMLS_clean/train.txt"  # the training triples
N_SHOW            = 10                          # how many records of each set


def load_triples(path):
    """Read tab-separated (s, r, o) triples into a list of tuples."""
    triples = []
    with open(path) as f:
        for line in f:
            line = line.rstrip("\n")
            if line:
                parts = line.split("\t")
                if len(parts) == 3:
                    triples.append(tuple(parts))
    return triples


def score_one(model, s, r, o):
    """
    Ask a dicee model for the PLAUSIBILITY score of (s, r, o).

    Uses the KGE.predict method ("kge predict") with logits=False, so the
    score is a probability between 0 and 1 (1 = model fully believes the
    triple). This is dicee's own prediction function, as the professor asked.
    """
    val = model.predict(h=s, r=r, t=o, logits=False)
    try:
        return float(val)            # usually a scalar tensor
    except TypeError:
        return float(val[0])         # just in case a 1-element tensor comes back


def score_set(models, triples, model_names):
    """Return a list of rows: {subject,relation,object, <model>:score,...}."""
    rows = []
    for (s, r, o) in triples:
        row = {"subject": s, "relation": r, "object": o}
        for name in model_names:
            row[name] = score_one(models[name], s, r, o)
        rows.append(row)
    return rows


def print_table(title, rows, model_names):
    print(f"\n=== {title} ===")
    print("(plausibility between 0 and 1; higher = model believes the triple more)\n")
    header = f"{'triple':<55}" + "".join(f"{n:>18}" for n in model_names)
    print(header)
    print("-" * len(header))
    for row in rows:
        triple = f"({row['subject']}, {row['relation']}, {row['object']})"
        line = f"{triple:<55}" + "".join(f"{row[n]:>18.4f}" for n in model_names)
        print(line)


def save_csv(path, rows, model_names):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    fieldnames = ["subject", "relation", "object"] + model_names
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    print(f"  saved {len(rows)} rows -> {path}")


def main():
    # load the two sets, take the first 10 of each
    test_triples  = load_triples(TEST_TARGETS_FILE)[:N_SHOW]
    train_triples = load_triples(TRAIN_FILE)[:N_SHOW]
    print(f"Test  set: {len(test_triples)} triples (from {TEST_TARGETS_FILE})")
    print(f"Train set: {len(train_triples)} triples (from {TRAIN_FILE})")

    # load every model once
    models = {}
    for name, path in MODELS.items():
        if os.path.isdir(path):
            print(f"Loading model '{name}' from {path}")
            models[name] = KGE(path=path)
        else:
            print(f"SKIP '{name}': folder not found ({path})")
    model_names = list(models.keys())

    # score both sets
    test_rows  = score_set(models, test_triples, model_names)
    train_rows = score_set(models, train_triples, model_names)

    # print the intermediate results (10 records each)
    print_table("TEST set - plausibility of 10 target triples (before vs after attack)",
                test_rows, model_names)
    print_table("TRAIN set - plausibility of 10 training triples (before vs after attack)",
                train_rows, model_names)

    # save full tables
    print("\n=== Saving full tables ===")
    save_csv("out/plausibility_test.csv",  test_rows,  model_names)
    save_csv("out/plausibility_train.csv", train_rows, model_names)


if __name__ == "__main__":
    main()
