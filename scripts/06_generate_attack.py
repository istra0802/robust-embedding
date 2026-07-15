"""
Step 5 - generate adversarial (poisoning) triples for every target triple,
using one inference pattern (symmetry / inversion / composition) and one
decoy-selection heuristic (truth / rank / cos), following Section 3 of the
paper. Restricted to the neighbourhood of the target's entities, and to at
most 1 decoy triple per target entity (paper's "Threat Model", Section 2).

Usage:
    python scripts/06_generate_attack.py out/clean data/UMLS_clean \
        out/targets.tsv symmetry truth out/adv_symmetry_truth.tsv
"""
import sys
sys.path.insert(0, "src")
import argparse
import numpy as np
from kge_utils import read_triples, load_embeddings
import attacks


def neighbourhood_candidates(entity_id, train_ids, n_entities, max_candidates=None):
    """
    Restrict decoy/adversarial-entity search to entities that are NOT
    already directly connected to entity_id in training data plus the
    entity itself (paper restricts edits to the neighbourhood of target
    entities, but the decoy entity itself should be a plausible *missing*
    link, i.e. not already an existing neighbour). For small KGs like UMLS
    we simply search the full entity set minus direct neighbours.
    """
    neighbours = set()
    for s, r, o in train_ids:
        if s == entity_id:
            neighbours.add(o)
        if o == entity_id:
            neighbours.add(s)
    neighbours.add(entity_id)
    candidates = [e for e in range(n_entities) if e not in neighbours]
    if max_candidates and len(candidates) > max_candidates:
        rng = np.random.default_rng(0)
        candidates = list(rng.choice(candidates, size=max_candidates, replace=False))
    return candidates


def attack_one_side(s_id, r_id, o_id, pattern, heuristic, ent_emb, rel_emb, train_ids, n_entities, n_relations):
    """Generates the adversarial body triples for the decoy (s, r, o')."""
    candidates = neighbourhood_candidates(o_id, train_ids, n_entities, max_candidates=200)
    edits = []

    if pattern == "symmetry":
        o_prime = attacks.decoy_symmetry(s_id, r_id, o_id, ent_emb, rel_emb, heuristic, set(candidates))
        if o_prime is None:
            return []
        edits.append((o_prime, r_id, s_id))  # body: (o', r, s)

    elif pattern == "inversion":
        ri_id = attacks.find_inverse_relation(r_id, rel_emb, exclude=r_id)
        o_prime = attacks.decoy_inversion(s_id, r_id, o_id, ri_id, ent_emb, rel_emb, heuristic, set(candidates))
        if o_prime is None:
            return []
        edits.append((o_prime, ri_id, s_id))  # body: (o', ri, s)

    elif pattern == "composition":
        r1_id, r2_id = attacks.find_composition_relations(r_id, rel_emb)
        o_prime = attacks.decoy_composition(s_id, r_id, o_id, r1_id, r2_id, ent_emb, rel_emb, heuristic, candidates)
        if o_prime is None:
            return []
        o_double_candidates = neighbourhood_candidates(s_id, train_ids, n_entities, max_candidates=200)
        o_double = attacks.adversarial_entity_composition(
            s_id, r_id, o_prime, r1_id, r2_id, ent_emb, rel_emb, o_double_candidates
        )
        if o_double is None:
            return []
        edits.append((s_id, r1_id, o_double))   # body triple 1: (s, r1, o'')
        edits.append((o_double, r2_id, o_prime))  # body triple 2: (o'', r2, o')

    else:
        raise ValueError(pattern)

    return edits


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("emb_prefix")
    ap.add_argument("dataset_dir")
    ap.add_argument("targets_file")
    ap.add_argument("pattern", choices=["symmetry", "inversion", "composition"])
    ap.add_argument("heuristic", choices=["truth", "rank", "cos"])
    ap.add_argument("out_file")
    args = ap.parse_args()

    ent_emb, rel_emb, ent2id, rel2id = load_embeddings(
        f"{args.emb_prefix}_entity_embeddings.csv", f"{args.emb_prefix}_relation_embeddings.csv"
    )
    id2ent = {v: k for k, v in ent2id.items()}
    id2rel = {v: k for k, v in rel2id.items()}

    train = read_triples(f"{args.dataset_dir}/train.txt")
    train_ids = [(ent2id[s], rel2id[r], ent2id[o]) for s, r, o in train
                 if s in ent2id and r in rel2id and o in ent2id]
    train_set = set(train_ids)

    targets = read_triples(args.targets_file)
    n_entities, n_relations = ent_emb.shape[0], rel_emb.shape[0]

    all_edits = set()
    for s, r, o in targets:
        if s not in ent2id or r not in rel2id or o not in ent2id:
            continue
        s_id, r_id, o_id = ent2id[s], rel2id[r], ent2id[o]

        # object-side decoy: (s, r, o')  -> attack using s and o roles as-is
        edits_obj = attack_one_side(s_id, r_id, o_id, args.pattern, args.heuristic,
                                     ent_emb, rel_emb, train_ids, n_entities, n_relations)
        # subject-side decoy: (s', r, o) -> swap head/tail roles for TransE symmetry-style search
        edits_sub = attack_one_side(o_id, r_id, s_id, args.pattern, args.heuristic,
                                     ent_emb, rel_emb, train_ids, n_entities, n_relations)

        for (a, rr, b) in (edits_obj + edits_sub):
            if (a, rr, b) not in train_set:
                all_edits.add((a, rr, b))

    with open(args.out_file, "w") as f:
        for a, rr, b in all_edits:
            f.write(f"{id2ent[a]}\t{id2rel[rr]}\t{id2ent[b]}\n")

    print(f"Generated {len(all_edits)} adversarial triples ({args.pattern}/{args.heuristic}) -> {args.out_file}")


if __name__ == "__main__":
    main()
