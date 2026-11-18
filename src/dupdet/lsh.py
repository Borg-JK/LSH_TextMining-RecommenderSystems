"""MinHash and LSH banding, implemented from scratch (Sect. 3.4).

A candidate pair (i, j), i < j, is stored as the integer i * n + j, so
candidate sets are sorted integer arrays.
"""
import zlib

import numpy as np

PRIME = 2_147_483_647          # 2^31 - 1, a Mersenne prime


def token_hash(token: str) -> int:
    """Stable integer in [0, PRIME) for a string token."""
    return zlib.crc32(token.encode("utf-8")) % PRIME


class MinHasher:
    """n hash functions h_j(x) = (a_j x + b_j) mod (2^31 - 1) on CRC32 token ids."""

    def __init__(self, n_hashes: int, seed: int):
        rng = np.random.default_rng(seed)
        self.n_hashes = n_hashes
        self.a = rng.integers(1, PRIME, size=n_hashes, dtype=np.int64)
        self.b = rng.integers(0, PRIME, size=n_hashes, dtype=np.int64)

    def signatures(self, sets: list) -> tuple[np.ndarray, np.ndarray]:
        """Return (signature matrix [n_sets, n_hashes], non_empty mask).

        Row i holds min over tokens x in set i of h_k(x), for every k.
        Empty sets get a row of -1 and are flagged in the mask.
        """
        n = len(sets)
        sig = np.full((n, self.n_hashes), -1, dtype=np.int64)
        non_empty = np.array([len(s) > 0 for s in sets], dtype=bool)
        if not non_empty.any():
            return sig, non_empty
        ids, starts, pos = [], [], 0
        for s in sets:
            if s:
                starts.append(pos)
                ids.extend(token_hash(t) for t in s)
                pos += len(s)
        ids = np.asarray(ids, dtype=np.int64)
        # a < 2^31 and x < 2^31, so a * x + b < 2^63: no int64 overflow.
        hashed = (self.a[:, None] * ids[None, :] + self.b[:, None]) % PRIME
        mins = np.minimum.reduceat(hashed, np.asarray(starts), axis=1)
        sig[non_empty] = mins.T
        return sig, non_empty


def lsh_candidates(sig: np.ndarray, non_empty: np.ndarray, bands: int, rows: int,
                   chunk: int = 4_000_000) -> np.ndarray:
    """Banding. Returns the sorted, unique encoded candidate pairs.

    Uses the first bands * rows signature columns.  The bucket of a band is
    the exact tuple of its `rows` MinHash values, so two products share a
    bucket only if they agree on the whole band.
    """
    n = sig.shape[0]
    assert bands * rows <= sig.shape[1], "signature too short for this (b, r)"
    idx = np.flatnonzero(non_empty)
    if len(idx) < 2:
        return np.empty(0, dtype=np.int64)
    parts, pending, result = [], 0, np.empty(0, dtype=np.int64)
    for k in range(bands):
        band = sig[idx, k * rows:(k + 1) * rows]
        _, inverse, counts = np.unique(band, axis=0, return_inverse=True,
                                       return_counts=True)
        inverse = inverse.ravel()
        order = np.argsort(inverse, kind="stable")   # members stay ascending
        ends = np.cumsum(counts)
        for g in np.flatnonzero(counts >= 2):
            members = idx[order[ends[g] - counts[g]:ends[g]]]
            a, b = np.triu_indices(len(members), k=1)
            parts.append(members[a] * n + members[b])
            pending += len(a)
        if pending >= chunk:                         # keep memory bounded
            result = np.union1d(result, np.concatenate(parts))
            parts, pending = [], 0
    if parts:
        result = np.union1d(result, np.concatenate(parts))
    return result


def decode_pairs(pairs: np.ndarray, n: int) -> tuple[np.ndarray, np.ndarray]:
    return pairs // n, pairs % n


def candidate_probability(s, bands: int, rows: int):
    """Theoretical S-curve: P(candidate) = 1 - (1 - s^r)^b."""
    return 1.0 - (1.0 - np.asarray(s, dtype=float) ** rows) ** bands
