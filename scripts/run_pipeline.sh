#!/usr/bin/env bash
# Full end-to-end pipeline: clean train -> target selection -> attack ->
# poison -> retrain -> compare. Run from the repo root, inside the `dice`
# conda env, on a machine with internet (for steps 01/02 only).
#
# Usage: bash scripts/run_pipeline.sh
set -e

PATTERN_HEUR_PAIRS=("symmetry truth" "inversion truth" "composition truth")

echo "== Step 0: install + data (skip if already done) =="
# bash scripts/01_install_dicee.sh
# bash scripts/02_download_data.sh

echo "== Step 2: train clean model =="
python scripts/03_train_clean.py

echo "== Step 3: extract clean embeddings =="
python scripts/04_extract_embeddings.py Experiments/UMLS_TransE_clean out/clean

echo "== Step 4: select target triples (rank<=10 under clean model) =="
mkdir -p out results
python scripts/05_select_targets.py out/clean data/UMLS_clean out/targets.tsv --max_rank 10

RUN_ARGS=()
for pair in "${PATTERN_HEUR_PAIRS[@]}"; do
    read -r pattern heuristic <<< "$pair"
    tag="${pattern}_${heuristic}"

    echo "== Attack: $tag =="
    python scripts/06_generate_attack.py out/clean data/UMLS_clean out/targets.tsv \
        "$pattern" "$heuristic" "out/adv_${tag}.tsv"

    python scripts/07_build_poisoned_dataset.py data/UMLS_clean \
        "out/adv_${tag}.tsv" "data/UMLS_poisoned_${tag}"

    python scripts/08_train_poisoned.py "data/UMLS_poisoned_${tag}" \
        "Experiments/UMLS_TransE_poisoned_${tag}"

    python scripts/04_extract_embeddings.py "Experiments/UMLS_TransE_poisoned_${tag}" "out/poisoned_${tag}"

    RUN_ARGS+=("${tag}:out/poisoned_${tag}:data/UMLS_poisoned_${tag}")
done

echo "== Final comparison =="
python scripts/09_evaluate_compare.py \
    --targets out/targets.tsv \
    --clean_emb out/clean --clean_dir data/UMLS_clean \
    --runs "${RUN_ARGS[@]}" \
    --out_csv results/comparison.csv --out_png results/comparison.png

echo "Pipeline complete. See results/comparison.csv and results/comparison.png"
