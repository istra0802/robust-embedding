"""
Implements the three inference-pattern poisoning attacks from
Bhardwaj et al. 2021 (arXiv:2111.06345) for the TransE model, which the
paper classifies as an *additive* model (Section 2, "Multiplicative vs
Additive Interactions").

For additive models the paper's algebraic relation-inference formulas are:
    inversion:    minimise || e_r + e_ri ||           (Section 3.1)
    composition:  minimise || e_r1 + e_r2 - e_r ||     (Section 3.1)

Step 2 (decoy entity selection) implements all three heuristics from the
paper: soft truth score (Sft), KGE rank (Rnk), cosine distance (Cos).
Step 3 (adversarial entity for composition) uses the soft-truth-score
maximisation described in Section 3.3.

Only OBJECT-side decoy triples (s, r, o') are generated below; the same
functions are called a second time with roles swapped for the SUBJECT-side
decoy (s', r, o), exactly as prescribed in Section 2 ("For a target triple
(s, r, o), the decoy triple for degrading the rank on object side would be
(s, r, o') and ... on subject side would be (s', r, o)").
"""
import numpy as np
from sklearn.cluster import KMeans

from kge_utils import transe_score, sigmoid, soft_truth, score_all_objects


# --------------------------------------------------------------------------- #
# Step 1: determine adversarial relation(s)
# --------------------------------------------------------------------------- #
def find_inverse_relation(r_id, rel_emb, exclude=None):
    """argmin_ri || e_r + e_ri ||  (additive model, Section 3.1)."""
    diffs = rel_emb + rel_emb[r_id][None, :]
    norms = np.linalg.norm(diffs, axis=1)
    if exclude is not None:
        norms = norms.copy()
        norms[exclude] = np.inf
    return int(np.argmin(norms))


def find_composition_relations(r_id, rel_emb):
    """
    argmin_(r1,r2) || e_r1 + e_r2 - e_r ||  (additive model, Section 3.1).
    O(R^2) — fine for small/medium relation sets (UMLS has 46 relations).
    """
    R = rel_emb.shape[0]
    er = rel_emb[r_id]
    best = None
    best_score = np.inf
    # vectorised over r2 for each r1
    for r1 in range(R):
        composed = rel_emb[r1][None, :] + rel_emb  # (R, dim)
        norms = np.linalg.norm(composed - er[None, :], axis=1)
        r2 = int(np.argmin(norms))
        if norms[r2] < best_score:
            best_score = norms[r2]
            best = (r1, r2)
    return best  # (r1_id, r2_id)


# --------------------------------------------------------------------------- #
# Step 2: determine decoy entity
# --------------------------------------------------------------------------- #
def decoy_symmetry(s_id, r_id, o_id, ent_emb, rel_emb, heuristic,
                    candidate_ids, known_sr2o=None):
    """
    Grounding: (o', r, s) => (s, r, o').  Head atom (s, r, o') is the decoy.
    We search over candidate object entities o' and pick the one that
    minimises phi(G_s) [truth], or has the KGE-rank-adjacent negative rank,
    or maximises cosine distance from o, per Section 3.2.
    """
    if heuristic == "truth":
        best_o, best_score = None, np.inf
        for o_prime in candidate_ids:
            phi_body = soft_truth(o_prime, r_id, s_id, ent_emb, rel_emb)   # (o', r, s)
            phi_head = soft_truth(s_id, r_id, o_prime, ent_emb, rel_emb)   # (s, r, o')
            phi_g = phi_body * phi_head - phi_body + 1.0
            if phi_g < best_score:
                best_score, best_o = phi_g, o_prime
        return best_o

    elif heuristic == "rank":
        scores = score_all_objects(s_id, r_id, ent_emb, rel_emb)
        order = np.argsort(-scores)
        target_pos = int(np.where(order == o_id)[0][0])
        # the negative ranked just below the target
        for pos in range(target_pos + 1, len(order)):
            cand = int(order[pos])
            if cand in candidate_ids:
                return cand
        return int(order[min(target_pos + 1, len(order) - 1)])

    elif heuristic == "cos":
        eo = ent_emb[o_id]
        norms = np.linalg.norm(ent_emb, axis=1) * np.linalg.norm(eo) + 1e-12
        cos_sim = (ent_emb @ eo) / norms
        cos_dist = 1.0 - cos_sim
        cos_dist_masked = np.full_like(cos_dist, -np.inf)
        for c in candidate_ids:
            cos_dist_masked[c] = cos_dist[c]
        return int(np.argmax(cos_dist_masked))
    else:
        raise ValueError(heuristic)


def decoy_inversion(s_id, r_id, o_id, ri_id, ent_emb, rel_emb, heuristic, candidate_ids):
    """Grounding: (o', ri, s) => (s, r, o'). Same three heuristics as symmetry."""
    if heuristic == "truth":
        best_o, best_score = None, np.inf
        for o_prime in candidate_ids:
            phi_body = soft_truth(o_prime, ri_id, s_id, ent_emb, rel_emb)
            phi_head = soft_truth(s_id, r_id, o_prime, ent_emb, rel_emb)
            phi_g = phi_body * phi_head - phi_body + 1.0
            if phi_g < best_score:
                best_score, best_o = phi_g, o_prime
        return best_o
    elif heuristic == "rank":
        return decoy_symmetry(s_id, r_id, o_id, ent_emb, rel_emb, "rank", candidate_ids)
    elif heuristic == "cos":
        return decoy_symmetry(s_id, r_id, o_id, ent_emb, rel_emb, "cos", candidate_ids)
    else:
        raise ValueError(heuristic)


def decoy_composition(s_id, r_id, o_id, r1_id, r2_id, ent_emb, rel_emb,
                       heuristic, candidate_ids, n_clusters=10):
    """
    Grounding: (s, r1, o'') ^ (o'', r2, o') => (s, r, o'). Decoy triple has
    TWO unknowns (o', o''). Step 2 picks o' (greedy, K-means clustering of
    o'' as in Section 3.2); Step 3 (separate function below) then picks o''.
    """
    candidate_ids = list(candidate_ids)
    if heuristic == "rank":
        return decoy_symmetry(s_id, r_id, o_id, ent_emb, rel_emb, "rank", set(candidate_ids))
    if heuristic == "cos":
        return decoy_symmetry(s_id, r_id, o_id, ent_emb, rel_emb, "cos", set(candidate_ids))

    # heuristic == "truth": K-means over o'' candidates, min soft-truth per cluster,
    # then min across clusters (Section 3.2)
    k = min(n_clusters, len(candidate_ids))
    if k < 2:
        o_double_candidates = candidate_ids
    else:
        km = KMeans(n_clusters=k, n_init=5, random_state=0)
        labels = km.fit_predict(ent_emb[candidate_ids])
        o_double_candidates = []
        for cl in range(k):
            members = [candidate_ids[i] for i in range(len(candidate_ids)) if labels[i] == cl]
            if not members:
                continue
            # representative: member with min soft truth for (s,r1,o'') given some o' guess
            # simplified per-cluster representative = medoid (closest to cluster mean)
            member_embs = ent_emb[members]
            centroid = member_embs.mean(axis=0)
            rep = members[int(np.argmin(np.linalg.norm(member_embs - centroid, axis=1)))]
            o_double_candidates.append(rep)

    best_o_prime, best_score = None, np.inf
    for o_prime in candidate_ids:
        # for this o', evaluate against the representative o'' set and take the min
        scores = []
        for o_double in o_double_candidates:
            phi1 = soft_truth(s_id, r1_id, o_double, ent_emb, rel_emb)
            phi2 = soft_truth(o_double, r2_id, o_prime, ent_emb, rel_emb)
            phi_head = soft_truth(s_id, r_id, o_prime, ent_emb, rel_emb)
            phi_g = phi1 * phi2 * phi_head - phi1 * phi2 + 1.0
            scores.append(phi_g)
        min_score = min(scores) if scores else np.inf
        if min_score < best_score:
            best_score, best_o_prime = min_score, o_prime
    return best_o_prime


# --------------------------------------------------------------------------- #
# Step 3: determine adversarial entity for composition (Section 3.3)
# --------------------------------------------------------------------------- #
def adversarial_entity_composition(s_id, r_id, o_prime_id, r1_id, r2_id,
                                    ent_emb, rel_emb, candidate_ids):
    """argmax_{o''} phi(G_t)  -- Section 3.3."""
    best_o_double, best_score = None, -np.inf
    for o_double in candidate_ids:
        phi1 = soft_truth(s_id, r1_id, o_double, ent_emb, rel_emb)
        phi2 = soft_truth(o_double, r2_id, o_prime_id, ent_emb, rel_emb)
        phi_head = soft_truth(s_id, r_id, o_prime_id, ent_emb, rel_emb)
        phi_g = phi1 * phi2 * phi_head - phi1 * phi2 + 1.0
        if phi_g > best_score:
            best_score, best_o_double = phi_g, o_double
    return best_o_double


# --------------------------------------------------------------------------- #
# Baseline: random neighbourhood edits (Section 4, "Random n")
# --------------------------------------------------------------------------- #
def random_n_edits(entity_id, n_relations, n_entities, rng, n_edits=1):
    edits = []
    for _ in range(n_edits):
        r = int(rng.integers(0, n_relations))
        e = int(rng.integers(0, n_entities))
        edits.append((entity_id, r, e))
    return edits
