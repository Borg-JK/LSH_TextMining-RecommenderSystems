"""Transitivity step after the logistic regression (Sect. 3.5).

Pairs predicted as duplicates are merged with union-find in order of
decreasing probability (ties: in pair order). A merge is rejected if the
cluster would contain two products of one shop or, when brands are given, two
different known brands. All pairs inside a cluster are predicted duplicates.
"""
from itertools import combinations

import numpy as np


def transitive_clusters(n: int, ii, jj, prob, thresholds, shop, brand=None, labels=None):
    """Greedy transitive closure of the pairs with prob >= threshold.

    n       : number of products; ii, jj: the two products of each candidate pair
    prob    : predicted probability per pair
    thresholds : probability cut-offs; the state after all pairs with
              prob >= t is scored for every t
    shop    : shop code per product (two products of one shop never share a cluster)
    brand   : brand code per product (-1 = unknown) or None (no brand rule)
    labels  : gold cluster label per product (training only) or None

    Returns (scores, clusters): scores[t] = (true pairs, predicted pairs) after
    the pairs with prob >= t (true pairs 0 if labels is None), and the clusters
    (lists of products, size >= 2) for the lowest threshold.
    """
    desc = sorted(set(float(t) for t in thresholds), reverse=True)
    prob = np.asarray(prob, float)
    keep = np.flatnonzero(prob >= desc[-1])
    order = keep[np.argsort(-prob[keep], kind="stable")]
    ii_l, jj_l, p_l = (np.asarray(ii)[order].tolist(), np.asarray(jj)[order].tolist(),
                       prob[order].tolist())
    root = list(range(n))
    members = {}
    shop_mask = [1 << int(s) for s in shop]
    known = [int(b) for b in brand] if brand is not None else None

    scores, tp, n_pred, e = {}, 0, 0, 0
    for a, b, p in zip(ii_l, jj_l, p_l):
        while p < desc[e]:
            scores[desc[e]] = (tp, n_pred)
            e += 1
        ra, rb = root[a], root[b]
        if ra == rb:
            continue
        if shop_mask[ra] & shop_mask[rb]:
            continue                                  # two products of one shop
        if known is not None and known[ra] >= 0 and known[rb] >= 0 and known[ra] != known[rb]:
            continue                                  # two different known brands
        ca, cb = members.get(ra, [ra]), members.get(rb, [rb])
        if labels is not None:
            tp += sum(labels[x] == labels[y] for x in ca for y in cb)
        n_pred += len(ca) * len(cb)
        for y in cb:
            root[y] = ra
        members[ra] = ca + cb
        members.pop(rb, None)
        shop_mask[ra] |= shop_mask[rb]
        if known is not None and known[ra] < 0:
            known[ra] = known[rb]
    for t in desc[e:]:
        scores[t] = (tp, n_pred)
    return scores, list(members.values())


def cluster_pairs(clusters: list, n: int) -> np.ndarray:
    """Encoded pairs (i * n + j, i < j) inside the clusters, sorted."""
    return np.array(sorted(a * n + b for c in clusters for a, b in combinations(sorted(c), 2)),
                    dtype=np.int64)
