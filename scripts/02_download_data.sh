#!/usr/bin/env bash
# Step 1 - download the benchmark knowledge graphs bundled by dice-embeddings
# and stage a working copy of UMLS that we will poison.
set -e

wget https://files.dice-research.org/datasets/dice-embeddings/KGs.zip --no-check-certificate
unzip -o KGs.zip

# sanity check: UMLS should have train/valid/test triples files
head -3 KGs/UMLS/train.txt
wc -l KGs/UMLS/train.txt KGs/UMLS/valid.txt KGs/UMLS/test.txt

# Make a clean working copy we can safely edit (the attack scripts write here)
mkdir -p data/UMLS_clean data/UMLS_poisoned
cp KGs/UMLS/train.txt KGs/UMLS/valid.txt KGs/UMLS/test.txt data/UMLS_clean/
cp KGs/UMLS/train.txt KGs/UMLS/valid.txt KGs/UMLS/test.txt data/UMLS_poisoned/

echo "Data staged in data/UMLS_clean and data/UMLS_poisoned"
