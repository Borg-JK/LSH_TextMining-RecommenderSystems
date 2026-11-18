"""Checks of the self-written LSH against theory.  Run: python tests/test_lsh.py"""
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from dupdet.lsh import MinHasher, lsh_candidates, candidate_probability  # noqa: E402


def make_pair(jaccard: float, size: int, tag: str):
    """Two sets of equal size with (approximately) the requested Jaccard."""
    shared = round(2 * size * jaccard / (1 + jaccard))
    a = {f"{tag}s{i}" for i in range(shared)} | {f"{tag}a{i}" for i in range(size - shared)}
    b = {f"{tag}s{i}" for i in range(shared)} | {f"{tag}b{i}" for i in range(size - shared)}
    return a, b, len(a & b) / len(a | b)


def test_minhash_estimates_jaccard():
    mh = MinHasher(2000, seed=1)
    for target in (0.1, 0.3, 0.5, 0.8):
        a, b, true_j = make_pair(target, 60, "x")
        sig, _ = mh.signatures([a, b])
        est = (sig[0] == sig[1]).mean()
        assert abs(est - true_j) < 0.04, (true_j, est)


def test_identical_and_empty_sets():
    mh = MinHasher(60, seed=2)
    sig, ok = mh.signatures([{"a", "b"}, {"b", "a"}, set(), {"zzz"}])
    assert (sig[0] == sig[1]).all() and not ok[2]
    cand = lsh_candidates(sig, ok, bands=3, rows=20)
    assert cand.tolist() == [0 * 4 + 1]          # only (0, 1); the empty set never matches


def test_banding_matches_s_curve():
    # b * r = 100 < n = 103: only the first 100 signature values are used,
    # as for the grid settings with n // r bands (e.g. (166, 3) with n = 500)
    bands, rows, n_pairs = 20, 5, 1500
    mh = MinHasher(bands * rows + 3, seed=3)
    assert bands * rows < mh.n_hashes
    for target in (0.3, 0.5, 0.7):
        sets, true_j = [], None
        for p in range(n_pairs):
            a, b, true_j = make_pair(target, 40, f"p{p}_")
            sets += [a, b]
        sig, ok = mh.signatures(sets)
        cand = set(lsh_candidates(sig, ok, bands, rows).tolist())
        # the unused last columns have no influence on the candidates
        scrambled = sig.copy()
        scrambled[:, bands * rows:] = np.arange(len(sets))[:, None]
        assert set(lsh_candidates(scrambled, ok, bands, rows).tolist()) == cand
        n = len(sets)
        hit = sum((2 * p) * n + 2 * p + 1 in cand for p in range(n_pairs)) / n_pairs
        theory = float(candidate_probability(true_j, bands, rows))
        assert abs(hit - theory) < 0.05, (true_j, hit, theory)


if __name__ == "__main__":
    for name, fn in list(globals().items()):
        if name.startswith("test_"):
            fn()
            print("ok ", name)
