"""
Core utilities: TransE scoring, filtered ranking/evaluation, I/O helpers.

Implements the additive scoring function from the paper
(Bhardwaj et al. 2021, "Poisoning Knowledge Graph Embeddings via Relation
Inference Patterns", Table 1):

    f(s, r, o) = - || e_s + e_r - e_o ||_2

and the filtered ranking protocol described in Section 2 ("Link Prediction").
"""
import os
import numpy as np
import pandas as pd


# --------------------------------------------------------------------------- #
# I/O
# --------------------------------------------------------------------------- #
def read_triples(path):
    """Reads a whitespace/tab separated (s, r, o) triples file."""
    df = pd.read_csv(path, sep=r"\s+", header=None, names=["s", "r", "o"], dtype=str)
    return list(df.itertuples(index=False, name=None))


def write_triples(path, triples):
    with open(path, "w") as f:
        for s, r, o in triples:
            f.write(f"{s}\t{r}\t{o}\n")


def build_id_maps(*triple_lists):
    entities, relations = set(), set()
    for triples in triple_lists:
        for s, r, o in triples:
            entities.add(s)
            entities.add(o)
            relations.add(r)
    ent2id = {e: i for i, e in enumerate(sorted(entities))}
    rel2id = {r: i for i, r in enumerate(sorted(relations))}
    return ent2id, rel2id


def load_embeddings(entity_csv, relation_csv):
    """
    Loads entity/relation embeddings previously exported to CSV
    (see scripts/04_extract_embeddings.py). First column = entity/relation
    name (index), remaining columns = embedding vector.
    """
    ent_df = pd.read_csv(entity_csv, index_col=0)
    rel_df = pd.read_csv(relation_csv, index_col=0)
    ent2id = {name: i for i, name in enumerate(ent_df.index)}
    rel2id = {name: i for i, name in enumerate(rel_df.index)}
    return ent_df.values.astype(np.float64), rel_df.values.astype(np.float64), ent2id, rel2id


# --------------------------------------------------------------------------- #
# TransE scoring (additive model, Table 1 of the paper)
# --------------------------------------------------------------------------- #
def transe_score(es, er, eo):
    """Vectorised TransE score: -||es + er - eo||_2 . Higher = more plausible."""
    return -np.linalg.norm(es + er - eo, axis=-1)


def score_all_objects(s_id, r_id, ent_emb, rel_emb):
    """Score triple (s, r, ?) against every candidate object entity."""
    es = ent_emb[s_id]
    er = rel_emb[r_id]
    return transe_score(es[None, :], er[None, :], ent_emb)


def score_all_subjects(r_id, o_id, ent_emb, rel_emb):
    """Score triple (?, r, o) against every candidate subject entity."""
    er = rel_emb[r_id]
    eo = ent_emb[o_id]
    return transe_score(ent_emb, er[None, :], eo[None, :])


def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x))


def soft_truth(s_id, r_id, o_id, ent_emb, rel_emb):
    """phi(s, r, o) = sigmoid(f_sro)  -- Section 3.2 of the paper."""
    f = transe_score(ent_emb[s_id], rel_emb[r_id], ent_emb[o_id])
    return float(sigmoid(f))


# --------------------------------------------------------------------------- #
# Filtered ranking protocol (Section 2)
# --------------------------------------------------------------------------- #
def filtered_rank_object(s_id, r_id, o_id, ent_emb, rel_emb, known_os):
    """
    Rank of the true object o against all corrupted objects, filtering out
    other known-true objects for (s, r, *). known_os: set of entity ids that
    are true objects for (s, r, *) (excluding the target o itself).
    """
    scores = score_all_objects(s_id, r_id, ent_emb, rel_emb)
    target_score = scores[o_id]
    scores = scores.copy()
    for e in known_os:
        if e != o_id:
            scores[e] = -np.inf
    # rank = 1 + number of candidates scoring strictly higher than target
    rank = 1 + int(np.sum(scores > target_score))
    return rank


def filtered_rank_subject(s_id, r_id, o_id, ent_emb, rel_emb, known_ss):
    scores = score_all_subjects(r_id, o_id, ent_emb, rel_emb)
    target_score = scores[s_id]
    scores = scores.copy()
    for e in known_ss:
        if e != s_id:
            scores[e] = -np.inf
    rank = 1 + int(np.sum(scores > target_score))
    return rank


def build_known_index(all_triples_ids):
    """
    For filtered evaluation: maps (s, r) -> set of true o's, and (r, o) -> set
    of true s's, over ALL known triples (train + valid + test).
    """
    sr2o, ro2s = {}, {}
    for s, r, o in all_triples_ids:
        sr2o.setdefault((s, r), set()).add(o)
        ro2s.setdefault((r, o), set()).add(s)
    return sr2o, ro2s


def evaluate_triples(triples_ids, ent_emb, rel_emb, sr2o, ro2s):
    """
    Computes MRR / Hits@1 / Hits@10 over a list of (s,r,o) id-triples using
    the standard "both sides averaged" filtered protocol, and returns the
    per-triple ranks too (needed to pick target triples with rank <= 10).
    """
    ranks = []
    for s, r, o in triples_ids:
        r_obj = filtered_rank_object(s, r, o, ent_emb, rel_emb, sr2o.get((s, r), set()))
        r_sub = filtered_rank_subject(s, r, o, ent_emb, rel_emb, ro2s.get((r, o), set()))
        ranks.append((r_sub, r_obj))
    ranks_flat = np.array(ranks, dtype=np.float64).flatten()
    mrr = float(np.mean(1.0 / ranks_flat))
    hits1 = float(np.mean(ranks_flat <= 1))
    hits10 = float(np.mean(ranks_flat <= 10))
    return {"MRR": mrr, "Hits@1": hits1, "Hits@10": hits10}, ranks
