"""Label-free product representations: title tokens, graded copies, brands,
MSMP model words."""
import math
import re
from collections import Counter

from .keys import clean_key


def filter_by_df(token_lists: list, cutoff: float, min_df: int) -> list:
    """Keep tokens with min_df <= df <= cutoff * N (Sect. 3.1, Eq. 1); df on the
    given collection itself. Token order is preserved."""
    n = len(token_lists)
    df = Counter(t for toks in token_lists for t in toks)
    hi = cutoff * n
    return [[t for t in toks if min_df <= df[t] <= hi] for toks in token_lists]


def idf_weights(token_lists: list) -> dict:
    """idf(t) = log(N / df(t)) on the given collection itself."""
    n = len(token_lists)
    df = Counter(t for toks in token_lists for t in toks)
    return {t: math.log(n / d) for t, d in df.items()}


def graded_copies(weight: float) -> int:
    """max(1, round(weight)) copies of a token (Sect. 3.3, Eq. 3)."""
    return max(1, round(weight))


def expand_weighted(tokens, weights: dict) -> set:
    """Token t becomes the distinct copies t#1, ..., t#c with c =
    graded_copies(weights[t]); the plain Jaccard of two expanded sets is the
    weighted Jaccard (Sect. 3.3, Eq. 2)."""
    return {f"{t}#{k}" for t in tokens for k in range(1, graded_copies(weights[t]) + 1)}


def combine_title_key(title_sets: list, key_sets: list) -> list:
    """One token set per product: title tokens prefixed 't:', key tokens 'k:'
    (Sect. 3.4)."""
    return [{f"t:{t}" for t in ts} | {f"k:{k}" for k in ks}
            for ts, ks in zip(title_sets, key_sets)]


# TV brands that occur only in titles of listings without a brand key
TV_BRANDS = frozenset({"avue", "azend", "contex", "curtisyoung", "dynex", "gpx", "hiteker",
                       "insignia", "tcl", "venturer", "viore"})


def extract_brands(products: list, token_lists: list, extra_brands=TV_BRANDS) -> list:
    """One brand per product or None (Sect. 3.5).

    Vocabulary: the first word of every value of a key whose name contains
    'brand', plus extra_brands. Brand: the first title token in the
    vocabulary, else the first word of the product's own brand value.
    """
    own, vocab = [], set(extra_brands)
    for p in products:
        b = None
        for rk, rv in p.features.items():
            if "brand" in clean_key(rk):
                words = clean_key(rv).split()
                if words:
                    b = words[0]
                    vocab.add(b)
                break
        own.append(b)
    out = []
    for toks, b in zip(token_lists, own):
        hit = next((t for t in toks if t in vocab), None)
        out.append(hit if hit is not None else b)
    return out


# title model-word regex of MSMP (van Dam et al., 2016)
_MODEL_WORD = re.compile(r"[a-zA-Z0-9]*(?:(?:[0-9]+[^0-9, ]+)|(?:[^0-9, ]+[0-9]+))[a-zA-Z0-9]*")


def model_words(title: str) -> list:
    """MSMP title model words of the raw title, in title order, without repeats."""
    seen, out = set(), []
    for mw in _MODEL_WORD.findall(title):
        if mw not in seen:
            seen.add(mw)
            out.append(mw)
    return out
