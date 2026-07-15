"""
Offline sanity check: builds a tiny synthetic KG + random embeddings and
runs the full symmetry/inversion/composition attack + evaluation code path,
to catch shape/logic bugs before you spend GPU time with dicee.
Run: python tests/test_offline_sanity.py
"""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
import numpy as np

from kge_utils import (build_id_maps, build_known_index, evaluate_triples)
import attacks

rng = np.random.default_rng(0)
n_entities, n_relations, dim = 20, 5, 8
ent_emb = rng.normal(size=(n_entities, dim))
rel_emb = rng.normal(size=(n_relations, dim))

# make relation 0 the target relation, and construct a synthetic "inverse" and
# "composition" so find_inverse_relation / find_composition_relations have signal
rel_emb[1] = -rel_emb[0] + rng.normal(scale=0.01, size=dim)          # rel 1 ~ inverse of rel 0
rel_emb[3] = rel_emb[0] * 0                                          # dummy
rel_emb[2] = rel_emb[0] - rel_emb[4] + rng.normal(scale=0.01, size=dim)  # rel0 ~ rel2 + rel4 -> composition test

train_ids = [(rng.integers(0, n_entities), rng.integers(0, n_relations), rng.integers(0, n_entities))
             for _ in range(60)]

# --- Step 1 ---
ri = attacks.find_inverse_relation(0, rel_emb, exclude=0)
print("Inverse of relation 0 ->", ri, "(expect 1)")
r1, r2 = attacks.find_composition_relations(0, rel_emb)
print("Composition of relation 0 ->", (r1, r2))

# --- Step 2/3 attacks, all patterns/heuristics ---
s_id, r_id, o_id = 2, 0, 7
candidates = set(range(n_entities)) - {s_id, o_id}

for heuristic in ["truth", "rank", "cos"]:
    o_prime = attacks.decoy_symmetry(s_id, r_id, o_id, ent_emb, rel_emb, heuristic, candidates)
    assert o_prime is not None
    print(f"[symmetry/{heuristic}] decoy o' =", o_prime)

for heuristic in ["truth", "rank", "cos"]:
    o_prime = attacks.decoy_inversion(s_id, r_id, o_id, ri, ent_emb, rel_emb, heuristic, candidates)
    assert o_prime is not None
    print(f"[inversion/{heuristic}] decoy o' =", o_prime)

for heuristic in ["truth", "rank", "cos"]:
    o_prime = attacks.decoy_composition(s_id, r_id, o_id, r1, r2, ent_emb, rel_emb, heuristic, list(candidates), n_clusters=4)
    assert o_prime is not None
    o_double = attacks.adversarial_entity_composition(s_id, r_id, o_prime, r1, r2, ent_emb, rel_emb, list(candidates))
    assert o_double is not None
    print(f"[composition/{heuristic}] decoy o' = {o_prime}, adversarial o'' = {o_double}")

# --- evaluation utilities ---
sr2o, ro2s = build_known_index(train_ids)
metrics, ranks = evaluate_triples(train_ids[:10], ent_emb, rel_emb, sr2o, ro2s)
print("Sample metrics on 10 synthetic triples:", metrics)

print("\nALL OFFLINE SANITY CHECKS PASSED")
