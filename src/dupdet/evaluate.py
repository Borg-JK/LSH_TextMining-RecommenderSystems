"""Evaluation measures (Sect. 3.6)."""
import numpy as np


def gold_pairs(model_ids: list) -> np.ndarray:
    """Encoded pairs (i * n + j, i < j) of products with the same modelID."""
    n, groups = len(model_ids), {}
    for i, m in enumerate(model_ids):
        groups.setdefault(m, []).append(i)
    out = [g[a] * n + g[b] for g in groups.values()
           for a in range(len(g)) for b in range(a + 1, len(g))]
    return np.array(sorted(out), dtype=np.int64)


def _harmonic(a: float, b: float) -> float:
    return 2 * a * b / (a + b) if a + b > 0 else 0.0


def lsh_metrics(candidates: np.ndarray, gold: np.ndarray, n: int) -> dict:
    """Comparisons, fraction of comparisons, PQ, PC and F1* of the candidates."""
    found = int(np.isin(candidates, gold, assume_unique=True).sum())
    n_comp, n_all = len(candidates), n * (n - 1) // 2
    pq = found / n_comp if n_comp else 0.0
    pc = found / len(gold) if len(gold) else 0.0
    return dict(n_comparisons=n_comp, fraction=n_comp / n_all, dup_found=found,
                pair_quality=pq, pair_completeness=pc, f1_star=_harmonic(pq, pc))


def prediction_metrics(predicted: np.ndarray, gold: np.ndarray) -> dict:
    """Precision, recall and F1 of the predicted pairs; every gold pair that is
    not predicted is a false negative, also when LSH never proposed it."""
    tp = int(np.isin(predicted, gold, assume_unique=True).sum())
    prec = tp / len(predicted) if len(predicted) else 0.0
    rec = tp / len(gold) if len(gold) else 0.0
    return dict(precision=prec, recall=rec, f1=_harmonic(prec, rec))
