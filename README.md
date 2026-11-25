# Poisoning KGE via Relation Inference Patterns — dice-embeddings reproduction

Implements the data-poisoning attack from **Bhardwaj, Kelleher, Costabello,
O'Sullivan (2021), "Poisoning Knowledge Graph Embeddings via Relation
Inference Patterns"** (arXiv:2111.06345), applied to a **TransE** model
trained with the [`dicee`](https://github.com/dice-group/dice-embeddings)
(DICE Embeddings) framework on the **UMLS** knowledge graph.

TransE is an *additive* scoring model (`f(s,r,o) = -||e_s + e_r - e_o||`),
so we use the paper's additive-model formulas throughout (Section 3.1):

- **Symmetry**: no relation search needed, only decoy-entity selection.
- **Inversion**: `argmin_ri || e_r + e_ri ||`
- **Composition**: `argmin_(r1,r2) || e_r1 + e_r2 - e_r ||`

Decoy-entity selection implements all three heuristics from the paper
(Section 3.2): soft truth score (`truth`), KGE rank (`rank`), cosine
distance (`cos`).

## Two attack types

This repo covers **both** families of data-poisoning edits:

1. **Addition attack (the paper).** Adds adversarial triples that make the
   model believe a *decoy* fact through symmetry, inversion, or composition,
   which pushes the true target fact down in the ranking.

2. **Deletion attack (our extension).** For each target triple, it removes
   the training triples that most strongly *support* the target (the highest
   TransE score among the target's neighbours), so the model loses the
   evidence it relied on. Every deleted triple is recorded in
   `out/del_deletion.tsv` and verified by `12_verify_deletion.py` (confirms
   each deleted triple is gone from the new train.txt and existed in the
   original, and that the counts match).

For both attacks we always know exactly which triples changed: the added
triples are in `out/adv_*.tsv`, and the deleted triples are in
`out/del_deletion.tsv`.

## Pipeline overview

```
train.txt (clean) --[03]--> clean TransE model --[04]--> embeddings.csv
                                     |
                                    [05] select target triples (rank<=10)
                                     |
          ADDITION:                  |          DELETION:
          [06] generate adversarial  |          [10] choose triples to delete
               triples               |               (most-supportive neighbours)
          [07] merge into poisoned   |          [11] remove them -> UMLS_deleted
               train.txt             |          [12] verify deletion
                                     |
                                    [08] retrain TransE (same hyperparams)
                                     |
                                    [04] extract poisoned/deleted embeddings
                                     |
                                    [09] compare clean vs attacked MRR/Hits@1/Hits@10
                                          on the SAME target triples -> results/
```

## Repository layout

```
robust-embedding/
├── README.md
├── requirements.txt
├── scripts/
│   ├── 01_install_dicee.sh          # env setup
│   ├── 02_download_data.sh          # downloads KGs.zip, stages UMLS
│   ├── 03_train_clean.py            # trains clean TransE via dicee
│   ├── 04_extract_embeddings.py     # dumps entity/relation embeddings to CSV
│   ├── 05_select_targets.py         # picks rank<=10 target triples (paper Sec.4)
│   ├── 06_generate_attack.py        # ADDITION: generate adversarial triples (Sec.3)
│   ├── 07_build_poisoned_dataset.py # ADDITION: merge adversarial triples into train.txt
│   ├── 08_train_poisoned.py         # retrains TransE on poisoned/deleted data
│   ├── 09_evaluate_compare.py       # clean vs attacked comparison table+chart
│   ├── 10_generate_deletion.py      # DELETION: choose triples to delete + record them
│   ├── 11_build_deleted_dataset.py  # DELETION: remove them -> data/UMLS_deleted
│   ├── 12_verify_deletion.py        # DELETION: prove the deletion really happened
│   └── run_pipeline.sh              # runs addition + deletion end-to-end
├── src/
│   ├── kge_utils.py                 # TransE scoring + filtered ranking (paper Sec.2)
│   ├── attacks.py                   # ADDITION: symmetry/inversion/composition (Sec.3)
│   └── deletion_attack.py           # DELETION: most-supportive-neighbour logic
├── tests/
│   └── test_offline_sanity.py       # numpy-only smoke test, no dicee/network needed
└── results/                         # comparison.csv / comparison.png land here
```

## Step-by-step: run it yourself

All commands below assume you're at the repo root, on a machine **with
internet access** (needed only for steps 1–2; everything after that is
local).

### 1. Environment

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install dicee --extra-index-url https://download.pytorch.org/whl/cpu
pip install scikit-learn pandas numpy matplotlib
```

(or just `bash scripts/01_install_dicee.sh`, which does the same thing and
also installs `python3-venv` via `apt` if it's missing — common on fresh
WSL/Ubuntu installs)

**Every new terminal session**, re-activate before running anything else:
```bash
source .venv/bin/activate
```

### 2. Data

```bash
bash scripts/02_download_data.sh
```

This downloads `KGs.zip` from the dice-research servers and stages a
working copy of UMLS under `data/UMLS_clean/` (untouched baseline).

### 3. Sanity-check the attack code (optional but recommended)

```bash
python tests/test_offline_sanity.py
```

Runs the full symmetry/inversion/composition attack logic against a tiny
synthetic KG — no `dicee`, no GPU, no internet. Confirms the math/shape
logic before you spend real training time.

### 4. Train the clean model

```bash
python scripts/03_train_clean.py
python scripts/04_extract_embeddings.py Experiments/UMLS_TransE_clean out/clean
```

### 5. Select target triples

```bash
python scripts/05_select_targets.py out/clean data/UMLS_clean out/targets.tsv --max_rank 10
```

### 6. Addition attack (example: symmetry pattern, soft-truth-score heuristic)

```bash
python scripts/06_generate_attack.py out/clean data/UMLS_clean out/targets.tsv \
    symmetry truth out/adv_symmetry_truth.tsv

python scripts/07_build_poisoned_dataset.py data/UMLS_clean \
    out/adv_symmetry_truth.tsv data/UMLS_poisoned_symmetry_truth

python scripts/08_train_poisoned.py data/UMLS_poisoned_symmetry_truth \
    Experiments/UMLS_TransE_poisoned_symmetry_truth

python scripts/04_extract_embeddings.py \
    Experiments/UMLS_TransE_poisoned_symmetry_truth out/poisoned_symmetry_truth
```

Repeat with `inversion` / `composition` and `truth` / `rank` / `cos` for
the other 8 combinations, or just run everything at once (section 8).

The triples added by each attack are saved in `out/adv_<pattern>_<heuristic>.tsv`.

### 7. Deletion attack

```bash
python scripts/10_generate_deletion.py       # writes out/del_deletion.tsv (exact list)
python scripts/11_build_deleted_dataset.py   # writes data/UMLS_deleted/train.txt
python scripts/12_verify_deletion.py         # must print ALL CHECKS PASSED
python scripts/08_train_poisoned.py data/UMLS_deleted Experiments/UMLS_TransE_deleted
python scripts/04_extract_embeddings.py Experiments/UMLS_TransE_deleted out/deleted
```

The file `out/del_deletion.tsv` lists exactly which triples were removed.
`12_verify_deletion.py` confirms each deleted triple is gone from the new
`train.txt`, existed in the original, and that
`new size = original size − number deleted`.

### 8. Run the whole pipeline (addition + deletion) + comparison

```bash
bash scripts/run_pipeline.sh
```

This runs the three addition attacks and the deletion attack end-to-end and
writes the full comparison. Edit the `PATTERN_HEUR_PAIRS` array at the top of
`run_pipeline.sh` to add more pattern/heuristic combinations (9 total are
implemented).

### 9. Results

```bash
cat results/comparison.csv
open results/comparison.png   # or just view it
```

Example result on UMLS (TransE, dim 32, 128 epochs, truth heuristic):

| run                | MRR   | Hits@1 | Hits@10 | MRR_drop_% |
|--------------------|-------|--------|---------|------------|
| Original (clean)   | 0.649 | 0.432  | 1.000   | —          |
| symmetry_truth     | 0.719 | 0.542  | 0.988   | −10.7%     |
| inversion_truth    | 0.689 | 0.499  | 0.986   | −6.1%      |
| composition_truth  | 0.599 | 0.397  | 0.948   | +7.8%      |
| deletion           | 0.604 | 0.365  | 0.997   | +6.9%      |

A **positive** `MRR_drop_%` means the attack degraded the model on the target
triples (the attack worked); a **negative** value means target MRR went up.
On this TransE/UMLS setting, composition and deletion degrade the model,
while symmetry and inversion raise overall target MRR — a side effect the
original paper documents for TransE (Appendix C): injecting a genuine
relational pattern the dataset was missing can improve general performance
while the intended per-target damage is harder to see in overall MRR.

> **Note on fair comparison.** The addition attacks add ~1–2 triples per
> target, while the deletion attack here removed 76 triples total. For a
> strict head-to-head, match the number of edits (e.g. cap the deletion
> budget). This is a good next refinement.

## Notes / things to double check when you actually run this

- **`scripts/04_extract_embeddings.py`** auto-detects the entity/relation
  embedding tensors inside the trained `dicee` model by matching tensor row
  counts to the number of entities/relations. If `dicee`'s internal module
  names ever change and auto-detection fails, the script prints the full
  `state_dict` so you can hard-code `EXPECTED_ENTITY_KEY` /
  `EXPECTED_RELATION_KEY` at the top of the file.
- The paper attacks **WN18RR / FB15k-237** at full scale; those are much
  bigger than UMLS, so runtimes will be materially longer if you swap
  `KGs/UMLS` for `KGs/WN18RR` or `KGs/FB15k-237` in
  `03_train_clean.py` / `08_train_poisoned.py` — same scripts work, just
  budget more time/epochs.
- `06_generate_attack.py` restricts the decoy/adversarial-entity search to a
  random subsample of non-neighbour candidates on larger graphs for
  tractability; use exhaustive search on small KGs like UMLS.

## Collaborating (two people, one repo)

Before you start working each time:
```bash
git pull
```
After you finish:
```bash
git add -A
git commit -m "describe what you did"
git push
```
Large/regenerable folders (`Experiments/`, `KGs/`, `data/`, `out/`) are
gitignored, so only code and `results/` are tracked — your teammate can
regenerate the rest by running `bash scripts/run_pipeline.sh`.

## Reference

Bhardwaj, P., Kelleher, J., Costabello, L., O'Sullivan, D. (2021).
*Poisoning Knowledge Graph Embeddings via Relation Inference Patterns.*
arXiv:2111.06345. https://arxiv.org/abs/2111.06345
