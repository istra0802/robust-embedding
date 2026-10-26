"""
Deletion attack against a TransE KGE model.

Opposite of the addition attack: for each target (s, r, o), find the
TRAINING triples that most strongly SUPPORT the target (highest TransE
score among the target's neighbours) and delete them, so the model loses
the evidence it used to rank the target highly.

All triples are (subject, relation, object) tuples of NAMES, matching
targets.tsv and train.txt. Embeddings are looked up by name.
"""
import numpy as np


def transe_score(es, er, eo):
    """TransE score f = -|| es + er - eo ||  (higher = model believes more)."""
    return -np.linalg.norm(es + er - eo)


def _neighbours(train_triples, s, o):
    """Training triples that contain s or o in any slot."""
    return [(ts, tr, to) for (ts, tr, to) in train_triples
            if ts == s or to == s or ts == o or to == o]


def generate_deletion_triples(targets, train_triples,
                              ent_emb, rel_emb, per_target=1):
    """
    ent_emb, rel_emb : dicts  name -> numpy vector
    Returns a de-duplicated list of (s, r, o) name-triples to delete.
    """
    train_set = set(train_triples)
    to_delete = set()

    for (s, r, o) in targets:
        scored = []
        for (ts, tr, to) in _neighbours(train_triples, s, o):
            if (ts, tr, to) == (s, r, o):
                continue                          # never delete the target
            score = transe_score(ent_emb[ts], rel_emb[tr], ent_emb[to])
            scored.append((score, (ts, tr, to)))

        scored.sort(key=lambda x: x[0], reverse=True)   # strongest support first

        for _, triple in scored[:per_target]:
            if triple in train_set:
                to_delete.add(triple)

    return list(to_delete)
