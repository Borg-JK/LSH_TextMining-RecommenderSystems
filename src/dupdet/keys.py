"""Key-value tokens learned on training data (Sect. 3.2, Appendix A).

1. clean key names and normalise values;
2. merge keys of different shops whose values agree for duplicates far more
   often than for non-duplicates (connected components);
3. per canonical key, m = P(agree | duplicate), u = P(agree | non-duplicate)
   and the Fellegi-Sunter weights log(m/u) and log((1-m)/(1-u));
4. select the K keys with the highest log(m/u).
"""
import math
import re
from collections import Counter, defaultdict
from dataclasses import dataclass, field

import numpy as np

from .config import (ALIGN_LIFT_MIN, ALIGN_M_MIN, MIN_PAIRS, N_NONDUP_ALIGN, N_NONDUP_WEIGHTS,
                     NUMERIC_SHARE, TOP_K_KEYS)

_KEY_JUNK = re.compile(r"[^a-z0-9]+")
_NUMBER = re.compile(r"\d+(?:\.\d+)?")
_NOT_ALNUM = re.compile(r"[^a-z0-9]")


def clean_key(key: str) -> str:
    """'Brand Name:' -> 'brand name'."""
    return _KEY_JUNK.sub(" ", key.lower()).strip()


def normalise_value(value: str, numeric: bool):
    """Numeric keys: first number, rounded. Other keys: lowercase alphanumerics."""
    v = value.lower()
    if numeric:
        m = _NUMBER.search(v)
        return str(round(float(m.group()))) if m else None
    v = _NOT_ALNUM.sub("", v)
    return v or None


def _smoothed(agree: int, n: int) -> float:
    return (agree + 1) / (n + 2)


@dataclass
class KeyModel:
    numeric_keys: set
    key_to_canon: dict                 # cleaned key -> canonical key
    member_rank: dict                  # cleaned key -> priority inside its canonical key
    weights: dict                      # canon -> dict(m, u, w_agree, w_disagree)
    selected: list                     # top-K canonical keys by log(m/u)
    informative: list                  # keys used in S+ and S-
    merged_groups: list                # the learned synonym groups
    _codes: dict = field(default_factory=dict)

    def canonical_values(self, features: dict) -> dict:
        """{canonical key: normalised value} for one product."""
        best = {}
        for raw_key, raw_val in features.items():
            k = clean_key(raw_key)
            canon = self.key_to_canon.get(k)
            if canon is None:
                continue
            val = normalise_value(raw_val, k in self.numeric_keys)
            if val is None:
                continue
            rank = self.member_rank[k]
            if canon not in best or rank < best[canon][0]:
                best[canon] = (rank, val)
        return {c: v for c, (_, v) in best.items()}

    def value_matrix(self, products: list, keys: list) -> np.ndarray:
        """Integer-coded values [n_products, n_keys]; -1 where the key is missing."""
        col = {c: j for j, c in enumerate(keys)}
        mat = np.full((len(products), len(keys)), -1, dtype=np.int64)
        for i, p in enumerate(products):
            for c, v in self.canonical_values(p.features).items():
                j = col.get(c)
                if j is not None:
                    mat[i, j] = self._codes.setdefault((c, v), len(self._codes))
        return mat


def _sample_nondup_pairs(n: int, labels: list, n_pairs: int, rng) -> np.ndarray:
    a = rng.integers(0, n, size=3 * n_pairs)
    b = rng.integers(0, n, size=3 * n_pairs)
    lab = np.asarray(labels)
    keep = (a != b) & (lab[a] != lab[b])
    return np.stack([a[keep], b[keep]], axis=1)[:n_pairs]


def learn_key_model(products: list, dup_pairs: list, rng) -> KeyModel:
    """Key alignment, weights and key selection on the training products;
    dup_pairs are the (i, j) positions of the training duplicates."""
    n = len(products)
    labels = [p.model_id for p in products]
    shops = [p.shop for p in products]

    # 1. numeric keys and normalised values
    has_digit, total = Counter(), Counter()
    for p in products:
        for rk, rv in p.features.items():
            k = clean_key(rk)
            total[k] += 1
            has_digit[k] += bool(re.search(r"\d", rv))
    numeric = {k for k in total if has_digit[k] / total[k] >= NUMERIC_SHARE}

    feats = []
    for p in products:
        f = {}
        for rk, rv in p.features.items():
            k = clean_key(rk)
            v = normalise_value(rv, k in numeric)
            if v is not None:
                f[k] = v
        feats.append(f)

    # 2. key alignment across shops
    co, ag = Counter(), Counter()
    for i, j in dup_pairs:
        if shops[i] == shops[j]:
            continue                       # one shop uses one set of key names
        for k1, v1 in feats[i].items():
            for k2, v2 in feats[j].items():
                if k1 == k2:
                    continue
                pair = (k1, k2) if k1 < k2 else (k2, k1)
                co[pair] += 1
                ag[pair] += v1 == v2
    promising = {pr for pr, c in co.items()
                 if c >= MIN_PAIRS and _smoothed(ag[pr], c) >= ALIGN_M_MIN}
    partners = defaultdict(set)
    for k1, k2 in promising:
        partners[k1].add(k2)
        partners[k2].add(k1)
    co_n, ag_n = Counter(), Counter()
    nondup = _sample_nondup_pairs(n, labels, N_NONDUP_ALIGN, rng)
    for i, j in nondup:
        if shops[i] == shops[j]:
            continue
        fi, fj = feats[i], feats[j]
        for k1, v1 in fi.items():
            for k2 in partners.get(k1, ()):
                v2 = fj.get(k2)
                if v2 is not None:
                    pair = (k1, k2) if k1 < k2 else (k2, k1)
                    co_n[pair] += 1
                    ag_n[pair] += v1 == v2
    parent = {k: k for k in total}

    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    for pr in sorted(promising):
        m = _smoothed(ag[pr], co[pr])
        u = _smoothed(ag_n[pr], co_n[pr])
        if math.log(m / u) >= ALIGN_LIFT_MIN:
            parent[find(pr[0])] = find(pr[1])

    groups = defaultdict(list)
    for k in total:
        groups[find(k)].append(k)
    key_to_canon, member_rank, merged = {}, {}, []
    for members in groups.values():
        members.sort(key=lambda k: (-total[k], k))       # most frequent key names the group
        canon = members[0]
        for rank, k in enumerate(members):
            key_to_canon[k], member_rank[k] = canon, rank
        if len(members) > 1:
            merged.append(members)

    model = KeyModel(numeric, key_to_canon, member_rank, {}, [], [], sorted(merged))

    # 3. m / u weights per canonical key
    canon_keys = sorted(set(key_to_canon.values()))
    vals = model.value_matrix(products, canon_keys)

    def agree_counts(pairs):
        pairs = np.asarray(pairs, dtype=np.int64).reshape(-1, 2)
        a, b = vals[pairs[:, 0]], vals[pairs[:, 1]]
        both = (a >= 0) & (b >= 0)
        return (both & (a == b)).sum(axis=0), both.sum(axis=0)

    ag_d, n_d = agree_counts(dup_pairs)
    ag_u, n_u = agree_counts(_sample_nondup_pairs(n, labels, N_NONDUP_WEIGHTS, rng))
    for j, c in enumerate(canon_keys):
        if n_d[j] < MIN_PAIRS or n_u[j] < MIN_PAIRS:
            continue                                     # too little evidence
        m, u = float(_smoothed(ag_d[j], n_d[j])), float(_smoothed(ag_u[j], n_u[j]))
        model.weights[c] = dict(m=m, u=u, w_agree=math.log(m / u),
                                w_disagree=math.log((1 - m) / (1 - u)))

    # 4. key selection
    by_weight = sorted(model.weights, key=lambda c: -model.weights[c]["w_agree"])
    model.selected = [c for c in by_weight if model.weights[c]["w_agree"] > 0][:TOP_K_KEYS]
    model.informative = [c for c in by_weight if model.weights[c]["w_agree"] > 0
                         and model.weights[c]["w_disagree"] < 0]
    return model
