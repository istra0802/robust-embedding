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

## Pipeline overview

```
train.txt (clean) --[03]--> clean TransE model --[04]--> embeddings.csv
                                     |
                                    [05] select target triples (rank<=10)
                                     |
                                    [06] generate adversarial triples
                                     |                (symmetry/inversion/composition)
                                    [07] merge into poisoned train.txt
                                     |
                                    [08] retrain TransE (same hyperparams)
                                     |
                                    [04] extract poisoned embeddings
                                     |
                                    [09] compare clean vs poisoned MRR/Hits@1/Hits@10
                                          on the SAME target triples -> results/
```

## Repository layout

```
kge-poisoning-attack/
├── README.md
├── requirements.txt
├── scripts/
│   ├── 01_install_dicee.sh          # env setup
│   ├── 02_download_data.sh          # downloads KGs.zip, stages UMLS
│   ├── 03_train_clean.py            # trains clean TransE via dicee
│   ├── 04_extract_embeddings.py     # dumps entity/relation embeddings to CSV
│   ├── 05_select_targets.py         # picks rank<=10 target triples (paper Sec.4)
│   ├── 06_generate_attack.py        # runs the paper's attack (Sec.3)
│   ├── 07_build_poisoned_dataset.py # merges adversarial triples into train.txt
│   ├── 08_train_poisoned.py         # retrains TransE on poisoned data
│   ├── 09_evaluate_compare.py       # clean vs poisoned comparison table+chart
│   └── run_pipeline.sh              # runs all of the above end-to-end
├── src/
│   ├── kge_utils.py                 # TransE scoring + filtered ranking (paper Sec.2)
│   └── attacks.py                   # symmetry/inversion/composition attacks (paper Sec.3)
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
working copy of UMLS under `data/UMLS_clean/` (untouched baseline) and
`data/UMLS_poisoned/` (scratch copy, overwritten per attack run).

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

### 6. Run one attack (example: symmetry pattern, soft-truth-score heuristic)

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
the other 8 combinations, or just run everything at once:

### 7. Run the whole pipeline (3 attacks by default) + comparison

```bash
bash scripts/run_pipeline.sh
```

Edit the `PATTERN_HEUR_PAIRS` array at the top of `run_pipeline.sh` to add
more pattern/heuristic combinations (9 total are implemented).

### 8. Results

```bash
cat results/comparison.csv
open results/comparison.png   # or just view it
```

You'll get a table like (numbers illustrative):

| run                | MRR   | Hits@1 | Hits@10 | MRR_drop_% |
|--------------------|-------|--------|---------|------------|
| Original (clean)   | 0.90  | 0.85   | 0.99    | —          |
| symmetry_truth     | 0.55  | 0.30   | 0.85    | 39%        |
| inversion_truth    | 0.80  | 0.70   | 0.95    | 11%        |
| composition_truth  | 0.78  | 0.68   | 0.94    | 13%        |

This is the exact comparison the paper reports in Tables 4/5 (reduction in
MRR/Hits@1 of target triples due to the attack) — just reproduced on UMLS
with `dicee` instead of the paper's own PyTorch codebase.

## Notes / things to double check when you actually run this

- **`scripts/04_extract_embeddings.py`** auto-detects the entity/relation
  embedding tensors inside the trained `dicee` model by matching tensor row
  counts to the number of entities/relations. If `dicee`'s internal module
  names ever change and auto-detection fails, the script prints the full
  `state_dict` so you can hard-code `EXPECTED_ENTITY_KEY` /
  `EXPECTED_RELATION_KEY` at the top of the file.
- The paper attacks **WN18RR / FB15k-237** at full scale; those are much
  bigger than UMLS, so runtimes will be materially longer if you swap
  `--dataset_dir "KGs/UMLS"` for `"KGs/WN18RR"` or `"KGs/FB15k-237"` in
  `03_train_clean.py` / `08_train_poisoned.py` — same scripts work, just
  budget more time/epochs.
- `06_generate_attack.py` restricts the decoy/adversarial-entity search to
  a random subsample of up to 200 non-neighbour candidates on larger graphs
  for tractability; drop `max_candidates` for exhaustive search on small
  KGs like UMLS.

## Pushing this to your own GitHub

```bash
cd kge-poisoning-attack
git init
git add .
git commit -m "Initial pipeline: KGE poisoning attack (inference patterns) with dice-embeddings"
git branch -M main
git remote add origin https://github.com/<your-username>/<your-repo-name>.git
git push -u origin main
```

(Create the empty repo on GitHub first via the web UI or `gh repo create
<your-repo-name> --public --source=. --remote=origin`.)

Because `Experiments/`, `KGs/`, `data/UMLS_*`, and `out/` are gitignored
(they're large/regenerable), only push `results/comparison.csv` and
`results/comparison.png` after a run if you want the numeric results
visible in the repo itself — everything else your professor can regenerate
by running `bash scripts/run_pipeline.sh`.

## Reference

Bhardwaj, P., Kelleher, J., Costabello, L., O'Sullivan, D. (2021).
*Poisoning Knowledge Graph Embeddings via Relation Inference Patterns.*
arXiv:2111.06345. https://arxiv.org/abs/2111.06345
