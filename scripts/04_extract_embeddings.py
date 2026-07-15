"""
Step 3 - export the trained entity/relation embeddings + id-maps to plain
CSV files that our numpy-only attack code (src/attacks.py, src/kge_utils.py)
can consume, independently of dicee's internal API.

Usage:
    python scripts/04_extract_embeddings.py Experiments/UMLS_TransE_clean out/clean

Why this script looks the way it does:
dicee's internal module/attribute names have changed across releases, so
instead of hard-coding one path (e.g. `model.entity_embeddings.weight`) we
search the loaded model's state_dict for the two weight matrices whose row
count matches len(entity_to_idx) and len(relation_to_idx) respectively.
This is robust to minor internal refactors. If it ever fails, the script
prints the full state_dict so you can see the actual tensor name and hard
code it in EXPECTED_ENTITY_KEY / EXPECTED_RELATION_KEY below.
"""
import sys
import os
import json
import numpy as np
import pandas as pd
from dicee import KGE

# If auto-detection below ever fails, set these explicitly, e.g.:
# EXPECTED_ENTITY_KEY = "entity_embeddings.weight"
# EXPECTED_RELATION_KEY = "relation_embeddings.weight"
EXPECTED_ENTITY_KEY = None
EXPECTED_RELATION_KEY = None


def find_embedding_matrices(kge, n_entities, n_relations):
    state = kge.model.state_dict()
    ent_key = EXPECTED_ENTITY_KEY
    rel_key = EXPECTED_RELATION_KEY

    if ent_key is None or rel_key is None:
        for k, v in state.items():
            if v.ndim != 2:
                continue
            if v.shape[0] == n_entities and ent_key is None:
                ent_key = k
            if v.shape[0] == n_relations and rel_key is None:
                rel_key = k

    if ent_key is None or rel_key is None:
        print("!! Could not auto-detect embedding tensors. Full state_dict shapes:")
        for k, v in state.items():
            print(f"   {k}: {tuple(v.shape)}")
        raise RuntimeError(
            "Set EXPECTED_ENTITY_KEY / EXPECTED_RELATION_KEY at the top of this "
            "script using the tensor names printed above, then re-run."
        )

    print(f"Using entity tensor:   {ent_key}  shape={tuple(state[ent_key].shape)}")
    print(f"Using relation tensor: {rel_key}  shape={tuple(state[rel_key].shape)}")
    return state[ent_key].detach().cpu().numpy(), state[rel_key].detach().cpu().numpy()


def main(experiment_path, out_prefix):
    kge = KGE(path=experiment_path)

    entity_to_idx = kge.entity_to_idx
    relation_to_idx = kge.relation_to_idx
    if hasattr(entity_to_idx, "to_dict"):
        entity_to_idx = {k: int(v) for k, v in entity_to_idx["entity"].to_dict().items()} \
            if hasattr(entity_to_idx, "columns") else entity_to_idx
    # entity_to_idx / relation_to_idx are usually plain dicts already
    entity_to_idx = dict(entity_to_idx)
    relation_to_idx = dict(relation_to_idx)

    ent_emb, rel_emb = find_embedding_matrices(kge, len(entity_to_idx), len(relation_to_idx))

    # order rows by id so row i == embedding of the entity/relation with id i
    ent_names = [None] * len(entity_to_idx)
    for name, idx in entity_to_idx.items():
        ent_names[int(idx)] = name
    rel_names = [None] * len(relation_to_idx)
    for name, idx in relation_to_idx.items():
        rel_names[int(idx)] = name

    os.makedirs(os.path.dirname(out_prefix) or ".", exist_ok=True)
    pd.DataFrame(ent_emb, index=ent_names).to_csv(f"{out_prefix}_entity_embeddings.csv")
    pd.DataFrame(rel_emb, index=rel_names).to_csv(f"{out_prefix}_relation_embeddings.csv")
    print(f"Wrote {out_prefix}_entity_embeddings.csv ({ent_emb.shape})")
    print(f"Wrote {out_prefix}_relation_embeddings.csv ({rel_emb.shape})")


if __name__ == "__main__":
    experiment_path = sys.argv[1]   # e.g. Experiments/UMLS_TransE_clean
    out_prefix = sys.argv[2]        # e.g. out/clean
    main(experiment_path, out_prefix)
