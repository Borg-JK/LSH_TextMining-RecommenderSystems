"""MSM benchmark (van Bezu et al., 2015) with its adapted single-linkage clustering.

msmp_msm decides the candidates of MSMP's title model-word LSH (title_mw_plain)
with MSM; msm_all applies MSM to all pairs. Configuration: MSM preprocessing
with punctuation replaced by spaces, value model words as in MSMP+ (Hartveld
et al., 2018), Jaccard share of model words, greedy one-to-one key matching,
q = 3, beta = 0, full grid for alpha, gamma, mu and epsilon.

The scalar functions (qgram_sim, kvp_part, title_sim, hsim) compute one pair
and serve as reference in the tests; MSMIndex computes the same for many
pairs at once.
"""
import re
from collections import Counter
from itertools import combinations

import numpy as np
from rapidfuzz.distance import Levenshtein
from rapidfuzz.process import cdist
from scipy import sparse

from .config import MSM_BETA, MSM_GRID, MSM_Q
from .evaluate import lsh_metrics, prediction_metrics
from .representation import _MODEL_WORD

_PUNCT = re.compile(r"[()\[\],;:!?+&*/]")
_SPACE = re.compile(r"\s+")
_PAD = "\x00"                      # padding character for q-grams
_COS_DECIMALS = 12                 # the title cosine is rounded before comparing with alpha
_VALUE_MW = re.compile(r"(\d+(?:\.\d+)?)[a-zA-Z]*")   # ^\d+(\.\d+)?[a-zA-Z]+$ or ^\d+(\.\d+)?$


# ---------------------------------------------------------------------------
# one pair at a time (reference)
# ---------------------------------------------------------------------------
def preprocess(s: str) -> str:
    """Lowercase; ( ) [ ] , ; : ! ? + & * / become a space; whitespace collapsed."""
    return _SPACE.sub(" ", _PUNCT.sub(" ", str(s).lower())).strip()


def qgram_counts(s: str, q: int = MSM_Q) -> Counter:
    """q-grams of s padded with q-1 dummy characters at both ends."""
    t = _PAD * (q - 1) + s + _PAD * (q - 1)
    return Counter(t[k:k + q] for k in range(len(t) - q + 1))


def qgram_sim(s1: str, s2: str, q: int = MSM_Q) -> float:
    """(n1 + n2 - qGramDistance) / (n1 + n2), qGramDistance = sum_g |c1(g) - c2(g)|."""
    c1, c2 = qgram_counts(s1, q), qgram_counts(s2, q)
    n = sum(c1.values()) + sum(c2.values())
    dist = sum(abs(c1[g] - c2[g]) for g in c1.keys() | c2.keys())
    return (n - dist) / n


def norm_lv(x: str, y: str) -> float:
    """Levenshtein distance / max(len(x), len(y))."""
    return Levenshtein.normalized_distance(x, y)


def split_model_word(w: str) -> tuple[str, str]:
    """(numeric part: digits and dots, non-numeric part: everything else)."""
    num = "".join(c for c in w if c.isdigit() or c == ".")
    return num, "".join(c for c in w if not (c.isdigit() or c == "."))


def title_model_words(title_pp: str) -> list:
    """Model words of a preprocessed title, in title order, without repeats."""
    return list(dict.fromkeys(_MODEL_WORD.findall(title_pp)))


def value_model_words(value_pp: str) -> set:
    """Numeric part of the tokens matching ^\\d+(\\.\\d+)?[a-zA-Z]+$ or ^\\d+(\\.\\d+)?$."""
    return {m.group(1) for t in value_pp.split() if (m := _VALUE_MW.fullmatch(t))}


def mw_share(a: set, b: set) -> float:
    """mwPerc = |A n B| / |A u B|; 0 if both sets are empty."""
    if not a and not b:
        return 0.0
    return len(a & b) / len(a | b)


def kvp_part(kv_i: list, kv_j: list, gamma: float, q: int = MSM_Q) -> tuple:
    """(m, avgSim, mwPerc, minFeatures) for products given as lists of
    preprocessed (key, value). Every KVP of i in order is matched with the
    first unmatched KVP of j whose key similarity exceeds gamma."""
    sim = w = 0.0
    m = 0
    unmatched_i, unmatched_j = set(range(len(kv_i))), set(range(len(kv_j)))
    for a, (key_q, val_q) in enumerate(kv_i):
        for r in range(len(kv_j)):
            if r not in unmatched_j:
                continue
            key_sim = qgram_sim(key_q, kv_j[r][0], q)
            if key_sim > gamma:
                sim += key_sim * qgram_sim(val_q, kv_j[r][1], q)
                w += key_sim
                m += 1
                unmatched_i.discard(a)
                unmatched_j.discard(r)
                break
    avg_sim = sim / w if w > 0 else 0.0
    mw_i = set().union(*(value_model_words(kv_i[a][1]) for a in unmatched_i))
    mw_j = set().union(*(value_model_words(kv_j[r][1]) for r in unmatched_j))
    return m, avg_sim, mw_share(mw_i, mw_j), min(len(kv_i), len(kv_j))


def avg_lv_sim(pairs: list) -> float:
    """sum (1 - normLv(x, y)) * (|x| + |y|) / sum (|x| + |y|)."""
    num = den = 0.0
    for x, y in pairs:
        size = len(x) + len(y)
        num += (1 - norm_lv(x, y)) * size
        den += size
    return num / den if den > 0 else 0.0


def title_sim(t_i: str, t_j: str, alpha: float, beta: float = MSM_BETA) -> float:
    """TMWM title similarity of two preprocessed titles; -1 = no title evidence."""
    wi, wj = t_i.split(), t_j.split()
    ci, cj = Counter(wi), Counter(wj)
    norm = np.sqrt(sum(v * v for v in ci.values()) * sum(v * v for v in cj.values()))
    cos = sum(ci[t] * cj[t] for t in ci.keys() & cj.keys()) / norm if norm else 0.0
    cos = round(float(cos), _COS_DECIMALS)
    if cos > alpha:
        return 1.0
    matched = []
    for x in title_model_words(t_i):
        num_x, non_x = split_model_word(x)
        for y in title_model_words(t_j):
            num_y, non_y = split_model_word(y)
            if norm_lv(non_x, non_y) < 0.5:
                if num_x != num_y:
                    return -1.0
                matched.append((x, y))
    words_i, words_j = list(dict.fromkeys(wi)), list(dict.fromkeys(wj))
    final = 0.5 * cos + 0.5 * avg_lv_sim([(x, y) for x in words_i for y in words_j])
    if matched:
        final = 0.5 * avg_lv_sim(matched) + 0.5 * final
    return final if final > beta else -1.0


def hsim(m, avg_sim, mw_perc, min_features, t_sim, mu):
    """MSM similarity from the key-value part and the title similarity
    (scalars or arrays)."""
    m, avg_sim, mw_perc = np.asarray(m, float), np.asarray(avg_sim, float), np.asarray(mw_perc, float)
    min_features, t_sim = np.asarray(min_features, float), np.asarray(t_sim, float)
    share = np.divide(m, min_features, out=np.zeros_like(m), where=min_features > 0)
    no_title = share * avg_sim + (1 - share) * mw_perc
    theta1 = (1 - mu) * share
    with_title = theta1 * avg_sim + (1 - mu - theta1) * mw_perc + mu * t_sim
    out = np.where(t_sim == -1, no_title, with_title)
    return float(out) if out.ndim == 0 else out


# ---------------------------------------------------------------------------
# many pairs at once
# ---------------------------------------------------------------------------
def _qgram_matrix(strings: list, q: int):
    """Sparse q-gram count matrix (strings x q-grams) and the q-gram totals."""
    vocab, rows, cols, vals = {}, [], [], []
    for k, s in enumerate(strings):
        for g, c in qgram_counts(s, q).items():
            rows.append(k)
            cols.append(vocab.setdefault(g, len(vocab)))
            vals.append(c)
    x = sparse.csr_matrix((vals, (rows, cols)), shape=(len(strings), len(vocab)), dtype=np.float64)
    return x, np.asarray(x.sum(axis=1)).ravel()


def _qgram_pair_sim(x, n, a: np.ndarray, b: np.ndarray, chunk: int = 200_000) -> np.ndarray:
    """qGramSim for string-id pairs (a, b): 2 * sum_g min(c1, c2) / (n1 + n2)."""
    out = np.empty(len(a))
    for s in range(0, len(a), chunk):
        aa, bb = a[s:s + chunk], b[s:s + chunk]
        common = np.asarray(x[aa].minimum(x[bb]).sum(axis=1)).ravel()
        out[s:s + chunk] = 2 * common / (n[aa] + n[bb])
    return out


def _common_counts(x) -> np.ndarray:
    """sum_g min(c1(g), c2(g)) for every pair of rows of x, as a dense uint16
    matrix, using min(a, b) = sum_k [a >= k][b >= k]."""
    n = x.shape[0]
    out = np.zeros((n, n), np.uint16)
    for k in range(1, int(x.max()) + 1 if x.nnz else 1):
        bk = (x >= k).astype(np.float32).tocsr()
        for s in range(0, n, 1000):
            out[s:s + 1000] += (bk[s:s + 1000] @ bk.T).toarray().astype(np.uint16)
    return out


class Parts:
    """Parameter-free MSM components of a list of pairs: min_features, cos and
    rest (titleSim when cos <= alpha) per pair; m, avg_sim and mw_perc per
    pair and gamma."""

    def __init__(self, table: np.ndarray, n_gamma: int):
        self.min_features, self.cos, self.rest = table[:, 0], table[:, 1], table[:, 2]
        g = n_gamma
        self.m = table[:, 3:3 + g].T
        self.avg_sim = table[:, 3 + g:3 + 2 * g].T
        self.mw_perc = table[:, 3 + 2 * g:3 + 3 * g].T

    def title_sim(self, alpha: float) -> np.ndarray:
        return np.where(self.cos > alpha, 1.0, self.rest)

    def similarity(self, alpha: float, g: int, mu: float) -> np.ndarray:
        return hsim(self.m[g], self.avg_sim[g], self.mw_perc[g], self.min_features,
                    self.title_sim(alpha), mu)


class MSMIndex:
    """Preprocessed products and cached string similarities of the whole data
    set. Products are addressed by Product.idx; in a pair the product with the
    lower index plays the role of i in the greedy key matching."""

    def __init__(self, products: list):
        self.gammas = MSM_GRID
        self.n = n = len(products)
        assert all(p.idx == k for k, p in enumerate(products))

        # key-value pairs
        kv = [[(preprocess(k), preprocess(v)) for k, v in p.features.items()] for p in products]
        keys = sorted({k for kvs in kv for k, _ in kvs})
        vals = sorted({v for kvs in kv for _, v in kvs})
        key_id = {k: c for c, k in enumerate(keys)}
        val_id = {v: c for c, v in enumerate(vals)}
        self.n_kvp = np.array([len(kvs) for kvs in kv])
        width = int(self.n_kvp.max())
        pad = len(keys)                                   # padding key: never matches
        self.key_pad = np.full((n, width), pad, np.int64)
        self.val_pad = np.zeros((n, width), np.int64)
        for p, kvs in enumerate(kv):
            self.key_pad[p, :len(kvs)] = [key_id[k] for k, _ in kvs]
            self.val_pad[p, :len(kvs)] = [val_id[v] for _, v in kvs]
        # value model words per KVP; only KVPs that have any are kept
        self.kvp_mw = [[(a, frozenset(value_model_words(v)))
                        for a, (_, v) in enumerate(kvs) if value_model_words(v)]
                       for kvs in kv]

        kx, kn = _qgram_matrix(keys, MSM_Q)
        a, b = np.triu_indices(len(keys))
        key_sim = np.full((len(keys) + 1, len(keys) + 1), -1.0)
        key_sim[a, b] = key_sim[b, a] = _qgram_pair_sim(kx, kn, a, b)
        self.key_sim = key_sim
        vx, self.val_n = _qgram_matrix(vals, MSM_Q)
        self.val_common = _common_counts(vx)

        # titles
        titles = [preprocess(p.title) for p in products]
        words = [t.split() for t in titles]
        mws = [title_model_words(t) for t in titles]
        strings = sorted({w for ws in words for w in ws} | {w for ms in mws for w in ms})
        sid = {s: c for c, s in enumerate(strings)}
        length = np.array([len(s) for s in strings], float)
        self.lv = cdist(strings, strings, scorer=Levenshtein.normalized_distance,
                        dtype=np.float64, workers=1)
        self.length = length

        counts = sparse.lil_matrix((n, len(strings)))
        for p, ws in enumerate(words):
            for w, c in Counter(ws).items():
                counts[p, sid[w]] = c
        counts = counts.tocsr()
        norms = np.sqrt(np.asarray(counts.multiply(counts).sum(axis=1)).ravel())
        self.unit_counts = sparse.diags(np.divide(1, norms, out=np.zeros(n), where=norms > 0)) @ counts
        self.word_set = (counts > 0).astype(np.float64).tocsr()     # distinct words
        self.n_words = np.asarray(self.word_set.sum(axis=1)).ravel()
        self.word_len = self.word_set @ length
        weighted = self.lv * (length[:, None] + length[None, :])
        self.word_lv = np.asarray(self.word_set @ weighted)          # n x strings

        nonnum = sorted({split_model_word(w)[1] for ms in mws for w in ms})
        nid = {s: c for c, s in enumerate(nonnum)}
        nonnum_lv = cdist(nonnum, nonnum, scorer=Levenshtein.normalized_distance,
                          dtype=np.float64, workers=1)
        self.nonnum_close = (nonnum_lv < 0.5).tolist()
        self.mws = [[(sid[w], nid[split_model_word(w)[1]], split_model_word(w)[0]) for w in ms]
                    for ms in mws]

        self._keys = np.empty(0, np.int64)                 # cache of pair components
        self._table = np.empty((0, 3 + 3 * len(self.gammas)))

    def parts(self, gi: np.ndarray, gj: np.ndarray) -> Parts:
        """Components for the index pairs (gi[k], gj[k]); computed once per pair."""
        lo, hi = np.minimum(gi, gj), np.maximum(gi, gj)
        code = lo.astype(np.int64) * self.n + hi
        pos = np.searchsorted(self._keys, code)
        have = np.zeros(len(code), bool)
        if len(self._keys):
            have = self._keys[np.minimum(pos, len(self._keys) - 1)] == code
        if not have.all():
            new = np.unique(code[~have])
            table = self._compute(new // self.n, new % self.n)
            keys = np.concatenate([self._keys, new])
            order = np.argsort(keys, kind="stable")
            self._keys = keys[order]
            self._table = np.concatenate([self._table, table])[order]
            pos = np.searchsorted(self._keys, code)
        return Parts(self._table[pos], len(self.gammas))

    def _compute(self, gi: np.ndarray, gj: np.ndarray, chunk: int = 1000) -> np.ndarray:
        g = len(self.gammas)
        out = np.empty((len(gi), 3 + 3 * g))
        for s in range(0, len(gi), chunk):
            i, j = gi[s:s + chunk], gj[s:s + chunk]
            out[s:s + chunk, 0] = np.minimum(self.n_kvp[i], self.n_kvp[j])
            out[s:s + chunk, 1:3] = self._title_parts(i, j)
            out[s:s + chunk, 3:] = self._kvp_parts(i, j)
        return out

    def _title_parts(self, i: np.ndarray, j: np.ndarray) -> np.ndarray:
        """cos and rest (titleSim when cos <= alpha) for each pair."""
        cos = np.asarray(self.unit_counts[i].multiply(self.unit_counts[j]).sum(axis=1)).ravel()
        cos = np.round(cos, _COS_DECIMALS)
        lv_sum = np.asarray(self.word_set[j].multiply(self.word_lv[i]).sum(axis=1)).ravel()
        den = self.n_words[j] * self.word_len[i] + self.n_words[i] * self.word_len[j]
        avg_words = np.divide(den - lv_sum, den, out=np.zeros(len(i)), where=den > 0)
        lv, length, close = self.lv, self.length, self.nonnum_close
        rest = np.empty(len(i))
        for k, (a, b) in enumerate(zip(i.tolist(), j.tolist())):
            matched, conflict = [], False
            for xs, xn, xnum in self.mws[a]:
                for ys, yn, ynum in self.mws[b]:
                    if close[xn][yn]:
                        if xnum != ynum:
                            conflict = True
                            break
                        matched.append((xs, ys))
                if conflict:
                    break
            if conflict:
                rest[k] = -1.0
                continue
            final = 0.5 * cos[k] + 0.5 * avg_words[k]
            if matched:
                num = den_mw = 0.0
                for xs, ys in matched:
                    size = length[xs] + length[ys]
                    num += (1 - lv[xs, ys]) * size
                    den_mw += size
                final = 0.5 * (num / den_mw if den_mw > 0 else 0.0) + 0.5 * final
            rest[k] = final if final > MSM_BETA else -1.0
        return np.column_stack([cos, rest])

    def _kvp_parts(self, i: np.ndarray, j: np.ndarray) -> np.ndarray:
        """m, avgSim and mwPerc for every gamma (greedy matching, vectorised)."""
        c, g = len(i), len(self.gammas)
        wa, wb = int(self.n_kvp[i].max()), int(self.n_kvp[j].max())
        ki, kj = self.key_pad[i, :wa], self.key_pad[j, :wb]
        vi, vj = self.val_pad[i, :wa], self.val_pad[j, :wb]
        s = self.key_sim[ki[:, :, None], kj[:, None, :]]          # c x wa x wb, padding -1
        vs = 2 * self.val_common[vi[:, :, None], vj[:, None, :]] / \
            (self.val_n[vi][:, :, None] + self.val_n[vj][:, None, :])   # value qGramSim
        rows = np.arange(c)
        m, avg, mwp = np.empty((c, g)), np.empty((c, g)), np.empty((c, g))
        for gk, gamma in enumerate(self.gammas):
            above = s > gamma
            taken = np.zeros((c, wb), bool)
            match = np.full((c, wa), -1)
            for a in range(wa):
                avail = above[:, a, :] & ~taken
                has = avail.any(axis=1)
                if not has.any():
                    continue
                r = avail.argmax(axis=1)                          # first free KVP of j
                hr = rows[has]
                taken[hr, r[has]] = True
                match[hr, a] = r[has]
            p, a = np.nonzero(match >= 0)
            r = match[p, a]
            key_s = s[p, a, r]
            w = np.bincount(p, key_s, minlength=c)
            sim = np.bincount(p, key_s * vs[p, a, r], minlength=c)
            m[:, gk] = np.bincount(p, minlength=c)
            avg[:, gk] = np.divide(sim, w, out=np.zeros(c), where=w > 0)
            mi_l, mj_l = (match >= 0).tolist(), taken.tolist()
            for k, (pa, pb) in enumerate(zip(i.tolist(), j.tolist())):
                mi, mj = mi_l[k], mj_l[k]
                mw_i = set().union(*[mw for q, mw in self.kvp_mw[pa] if not mi[q]])
                mw_j = set().union(*[mw for q, mw in self.kvp_mw[pb] if not mj[q]])
                mwp[k, gk] = mw_share(mw_i, mw_j)
        return np.hstack([m, avg, mwp])


_INDEX = {}


def get_index(products: list) -> MSMIndex:
    """One MSMIndex per process and data set, shared by all its bootstraps."""
    key = tuple((p.shop, p.title, tuple(p.features.items())) for p in products)
    if key not in _INDEX:
        _INDEX.clear()
        _INDEX[key] = MSMIndex(products)
    return _INDEX[key]


# ---------------------------------------------------------------------------
# clustering
# ---------------------------------------------------------------------------
def cluster(n: int, ii, jj, dist, eps_grid, allowed, labels=None, shop=None, brand=None):
    """Adapted single linkage, Kruskal style.

    Edges with finite distance are processed in ascending order (ties: given
    order); two clusters merge unless some pair across them is not
    ``allowed``. For each eps in the ascending eps_grid the state after all
    edges with distance <= eps is scored. shop and brand (optional) only
    reject merges early that ``allowed`` would reject as well.

    Returns (scores, clusters): scores[k] = (true pairs, predicted pairs) at
    eps_grid[k] (true pairs 0 if labels is None) and the clusters at the last eps.
    """
    dist = np.asarray(dist, float)
    keep = np.nonzero(dist <= eps_grid[-1])[0]
    order = keep[np.argsort(dist[keep], kind="stable")]
    ii_l, jj_l, d_l = (np.asarray(ii)[order].tolist(), np.asarray(jj)[order].tolist(),
                       dist[order].tolist())
    root = list(range(n))
    members = {}
    shop_mask = [1 << int(s) for s in shop] if shop is not None else None
    known_brand = [int(b) for b in brand] if brand is not None else None

    scores, tp, n_pred, e = [], 0, 0, 0
    for a, b, d in zip(ii_l, jj_l, d_l):
        while d > eps_grid[e]:
            scores.append((tp, n_pred))
            e += 1
        ra, rb = root[a], root[b]
        if ra == rb:
            continue
        if shop_mask is not None and shop_mask[ra] & shop_mask[rb]:
            continue
        if known_brand is not None:
            ba, bb = known_brand[ra], known_brand[rb]
            if ba >= 0 and bb >= 0 and ba != bb:
                continue
        ca, cb = members.get(ra, [ra]), members.get(rb, [rb])
        if not all(allowed(x, y) for x in ca for y in cb):
            continue
        if labels is not None:
            tp += sum(labels[x] == labels[y] for x in ca for y in cb)
        n_pred += len(ca) * len(cb)
        for y in cb:
            root[y] = ra
        members[ra] = ca + cb
        members.pop(rb, None)
        if shop_mask is not None:
            shop_mask[ra] |= shop_mask[rb]
        if known_brand is not None and known_brand[ra] < 0:
            known_brand[ra] = known_brand[rb]
    while len(scores) < len(eps_grid):
        scores.append((tp, n_pred))
    return scores, list(members.values())


def cluster_pairs(clusters: list, n: int) -> np.ndarray:
    """Encoded pairs (i * n + j, i < j) inside the clusters, sorted."""
    return np.array(sorted(a * n + b for c in clusters for a, b in combinations(sorted(c), 2)),
                    dtype=np.int64)


# ---------------------------------------------------------------------------
# tuning and evaluation
# ---------------------------------------------------------------------------
class _SetView:
    """What MSM needs from a train or test ProductSet."""

    def __init__(self, ps):
        self.n = ps.n
        self.gidx = np.array([p.idx for p in ps.products])
        shop_code = {s: k for k, s in enumerate(sorted({p.shop for p in ps.products}))}
        self.shop = np.array([shop_code[p.shop] for p in ps.products])
        self.brand = np.asarray(ps.brand)
        model_code = {}
        self.labels = [model_code.setdefault(p.model_id, len(model_code)) for p in ps.products]
        self.n_gold = len(ps.gold)
        self.gold = ps.gold

    def finite(self, pairs: np.ndarray) -> np.ndarray:
        """Pairs with a finite distance: different shops, brands not known to differ."""
        i, j = pairs // self.n, pairs % self.n
        bi, bj = self.brand[i], self.brand[j]
        ok = (self.shop[i] != self.shop[j]) & ((bi < 0) | (bj < 0) | (bi == bj))
        return pairs[ok]

    def all_finite(self) -> np.ndarray:
        i, j = np.triu_indices(self.n, k=1)
        return self.finite(i.astype(np.int64) * self.n + j)

    def allowed(self, candidates=None):
        """allowed(a, b): False when the MSM distance of (a, b) is infinite."""
        shop, brand, n = self.shop.tolist(), self.brand.tolist(), self.n
        cand = None if candidates is None else set(candidates.tolist())

        def ok(a, b):
            if shop[a] == shop[b]:
                return False
            if brand[a] >= 0 and brand[b] >= 0 and brand[a] != brand[b]:
                return False
            return cand is None or (min(a, b) * n + max(a, b)) in cand
        return ok


def _tune(parts: Parts, ii, jj, view: _SetView, allowed) -> tuple:
    """(alpha, gamma index, mu, epsilon) with the highest training F1 (training
    duplicates outside the pairs are false negatives). Ties: first in grid order."""
    best_f1, best = -1.0, None
    for alpha in MSM_GRID:
        for g in range(len(MSM_GRID)):
            for mu in MSM_GRID:
                dist = 1 - parts.similarity(alpha, g, mu) if len(ii) else np.empty(0)
                scores, _ = cluster(view.n, ii, jj, dist, MSM_GRID, allowed, view.labels,
                                    view.shop, view.brand)
                for eps, (tp, n_pred) in zip(MSM_GRID, scores):
                    den = n_pred + view.n_gold
                    f1 = 2 * tp / den if den else 0.0
                    if f1 > best_f1:
                        best_f1, best = f1, (alpha, g, mu, eps)
    return best


def _predict(parts: Parts, ii, jj, view: _SetView, allowed, params) -> np.ndarray:
    alpha, g, mu, eps = params
    dist = 1 - parts.similarity(alpha, g, mu) if len(ii) else np.empty(0)
    _, clusters = cluster(view.n, ii, jj, dist, [eps], allowed, None, view.shop, view.brand)
    return cluster_pairs(clusters, view.n)


def _parts_for(index: MSMIndex, view: _SetView, pairs: np.ndarray):
    ii, jj = pairs // view.n, pairs % view.n
    return index.parts(view.gidx[ii], view.gidx[jj]), ii, jj


def _evaluate(index, train, test, tr_pairs, te_pairs, restrict: bool) -> tuple:
    """Tune on the training pairs, predict and score on the test pairs.
    restrict: only the given pairs have a finite distance (msmp_msm).
    Returns (test row, chosen parameters)."""
    tr_edges, te_edges = train.finite(tr_pairs), test.finite(te_pairs)
    tr_parts, tr_i, tr_j = _parts_for(index, train, tr_edges)
    params = _tune(tr_parts, tr_i, tr_j, train, train.allowed(tr_edges if restrict else None))
    te_parts, te_i, te_j = _parts_for(index, test, te_edges)
    predicted = _predict(te_parts, te_i, te_j, test,
                         test.allowed(te_edges if restrict else None), params)
    row = lsh_metrics(te_pairs, test.gold, test.n)
    row.update(prediction_metrics(predicted, test.gold))
    alpha, g, mu, eps = params
    return row, dict(alpha=alpha, gamma=MSM_GRID[g], mu=mu, epsilon=eps)


def run_msm(products: list, train_set, test_set, candidates: dict, grid: list) -> tuple:
    """Rows of msmp_msm (every (b, r)) and msm_all (bands = rows = 0), and the
    chosen MSM parameters per setting. candidates[split, (b, r)] are the
    (unfiltered) LSH candidates of title_mw_plain."""
    index = get_index(products)
    train, test = _SetView(train_set), _SetView(test_set)
    rows, params = [], {"msmp_msm": {}, "msm_all": {}}
    for b, r in grid:
        row, par = _evaluate(index, train, test, candidates["train", (b, r)],
                             candidates["test", (b, r)], restrict=True)
        rows.append(dict(method="msmp_msm", bands=b, rows=r, **row))
        params["msmp_msm"][f"{b}x{r}"] = par
    # msm_all: the pairs with a finite distance are its comparisons
    row, par = _evaluate(index, train, test, train.all_finite(), test.all_finite(), restrict=False)
    rows.append(dict(method="msm_all", bands=0, rows=0, **row))
    params["msm_all"]["all pairs"] = par
    return rows, params
