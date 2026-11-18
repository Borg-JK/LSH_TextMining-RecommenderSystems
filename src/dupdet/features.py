"""Pair features of the logistic regression (Sect. 3.5), vectorised."""
import math
from collections import Counter

import numpy as np
from scipy import sparse

from .lsh import decode_pairs

FEATURE_NAMES = ["title_token_jaccard", "title_3gram_jaccard", "title_tfidf_cosine",
                 "key_agreement_S_plus", "key_conflict_S_minus", "same_shop"]


def _binary_matrix(sets: list) -> sparse.csr_matrix:
    vocab, rows, cols = {}, [], []
    for i, s in enumerate(sets):
        for t in s:
            rows.append(i)
            cols.append(vocab.setdefault(t, len(vocab)))
    data = np.ones(len(rows), dtype=np.float32)
    return sparse.csr_matrix((data, (rows, cols)), shape=(len(sets), max(len(vocab), 1)))


def _jaccard(mat, sizes, i, j):
    inter = np.asarray(mat[i].multiply(mat[j]).sum(axis=1)).ravel()
    union = sizes[i] + sizes[j] - inter
    return np.divide(inter, union, out=np.zeros_like(inter, dtype=float), where=union > 0)


def _tfidf_matrix(token_lists: list) -> sparse.csr_matrix:
    """L2-normalised idf-weighted rows (idf on the collection itself), so a
    row product is the cosine similarity."""
    n = len(token_lists)
    df = Counter(t for toks in token_lists for t in toks)
    vocab = {t: k for k, t in enumerate(df)}
    rows, cols, data = [], [], []
    for i, toks in enumerate(token_lists):
        w = [math.log(n / df[t]) for t in toks]
        norm = math.sqrt(sum(x * x for x in w)) or 1.0
        for t, x in zip(toks, w):
            rows.append(i)
            cols.append(vocab[t])
            data.append(x / norm)
    return sparse.csr_matrix((data, (rows, cols)), shape=(n, max(len(vocab), 1)))


def char_ngrams(tokens: list, q: int = 3) -> set:
    """q-grams of the tokens glued together in title order."""
    s = "".join(tokens)
    return {s[k:k + q] for k in range(len(s) - q + 1)} if len(s) >= q else ({s} if s else set())


class PairFeatures:
    """Per-product data of one product set; compute() gives the six features
    (same_shop is dropped for our filtered methods)."""

    def __init__(self, products, all_tokens, filtered_tokens, key_model):
        self.n = len(products)
        self.tfidf = _tfidf_matrix(all_tokens)
        self.tok = _binary_matrix([set(t) for t in filtered_tokens])
        self.gram = _binary_matrix([char_ngrams(t) for t in filtered_tokens])
        self.tok_size = np.asarray(self.tok.sum(axis=1)).ravel()
        self.gram_size = np.asarray(self.gram.sum(axis=1)).ravel()
        keys = key_model.informative
        self.vals = key_model.value_matrix(products, keys)
        self.w_agree = np.array([key_model.weights[c]["w_agree"] for c in keys])
        self.w_dis = np.array([-key_model.weights[c]["w_disagree"] for c in keys])
        shop_code = {s: k for k, s in enumerate(sorted({p.shop for p in products}))}
        self.shop = np.array([shop_code[p.shop] for p in products])

    def compute(self, pairs: np.ndarray, chunk: int = 200_000) -> np.ndarray:
        out = np.zeros((len(pairs), len(FEATURE_NAMES)))
        for s in range(0, len(pairs), chunk):
            i, j = decode_pairs(pairs[s:s + chunk], self.n)
            a, b = self.vals[i], self.vals[j]
            both = (a >= 0) & (b >= 0)
            agree = both & (a == b)
            block = out[s:s + chunk]
            block[:, 0] = _jaccard(self.tok, self.tok_size, i, j)
            block[:, 1] = _jaccard(self.gram, self.gram_size, i, j)
            block[:, 2] = np.asarray(self.tfidf[i].multiply(self.tfidf[j]).sum(axis=1)).ravel()
            block[:, 3] = agree @ self.w_agree
            block[:, 4] = (both & ~agree) @ self.w_dis
            block[:, 5] = self.shop[i] == self.shop[j]
        return out
