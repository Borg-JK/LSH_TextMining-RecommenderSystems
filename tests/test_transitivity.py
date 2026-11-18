"""Checks of the transitivity step.  Run: python tests/test_transitivity.py"""
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from dupdet.classifier import PairClassifier  # noqa: E402
from dupdet.config import PROB_GRID  # noqa: E402
from dupdet.transitivity import cluster_pairs, transitive_clusters  # noqa: E402

# six products: shops a b c d a b, brands 0 0 0 1 -1 0 (-1 = unknown)
SHOP = [0, 1, 2, 3, 0, 1]
BRAND = [0, 0, 0, 1, -1, 0]


def run(pairs, prob, thresholds=(0.5,), brand=None, labels=None):
    ii, jj = zip(*pairs)
    return transitive_clusters(6, ii, jj, prob, thresholds, SHOP, brand, labels)


def test_closure_adds_pairs_that_were_never_compared():
    # (0,1) and (1,2) predicted, (0,2) was never a candidate: it is added
    _, clusters = run([(0, 1), (1, 2)], [0.9, 0.8])
    assert cluster_pairs(clusters, 6).tolist() == [0 * 6 + 1, 0 * 6 + 2, 1 * 6 + 2]


def test_same_shop_merge_rejected():
    # 0 and 4 are both from shop a: the second merge would join them
    _, clusters = run([(0, 1), (1, 4)], [0.9, 0.8])
    assert sorted(map(sorted, clusters)) == [[0, 1]]
    # a predicted pair within one shop is never a cluster
    _, clusters = run([(0, 4)], [0.9])
    assert clusters == []


def test_brand_rule_only_when_given():
    pairs, prob = [(0, 1), (1, 3)], [0.9, 0.8]       # 3 has brand 1, 0 and 1 brand 0
    _, with_rule = run(pairs, prob, brand=BRAND)
    _, without = run(pairs, prob)
    assert sorted(map(sorted, with_rule)) == [[0, 1]]
    assert sorted(map(sorted, without)) == [[0, 1, 3]]
    # an unknown brand (product 4) does not block
    _, clusters = run([(1, 4)], [0.9], brand=BRAND)
    assert sorted(map(sorted, clusters)) == [[1, 4]]


def test_higher_probability_wins_and_at_most_four():
    # (1,4) and (0,1) conflict through shop a; the more probable pair is kept
    _, clusters = run([(0, 1), (1, 4)], [0.6, 0.9])
    assert sorted(map(sorted, clusters)) == [[1, 4]]
    pairs = [(a, b) for a in range(6) for b in range(a + 1, 6)]
    _, clusters = run(pairs, [1.0] * len(pairs))
    assert all(len(c) <= 4 and len({SHOP[x] for x in c}) == len(c) for c in clusters)


def test_scores_per_threshold():
    labels = [0, 0, 0, 1, 2, 3]                       # 0, 1, 2 are one product
    scores, _ = run([(0, 1), (1, 2), (2, 3)], [0.9, 0.7, 0.3], (0.8, 0.5, 0.2), labels=labels)
    assert scores[0.8] == (1, 1)                      # only (0,1)
    assert scores[0.5] == (3, 3)                      # cluster {0,1,2}
    assert scores[0.2] == (3, 6)                      # + product 3: 3 more pairs, all wrong


def test_classifier_tunes_threshold_with_closure():
    rng = np.random.default_rng(0)
    X = rng.normal(size=(200, 2))
    y = (X[:, 0] > 0.8).astype(int)
    seen = {}

    def closure(prob, thresholds):
        seen["called"] = True
        pred = {t: int((prob >= t).sum()) for t in thresholds}
        return {float(t): (int(((prob >= t) & (y == 1)).sum()), n) for t, n in pred.items()}, []
    clf = PairClassifier().fit(X, y, int(y.sum()), closure)
    prob = clf.predict_proba(X)
    f1 = {t: 2 * int(((prob >= t) & (y == 1)).sum()) / (int((prob >= t).sum()) + int(y.sum()))
          for t in PROB_GRID}
    assert seen["called"] and clf.threshold == max(PROB_GRID, key=f1.get)   # first maximum


if __name__ == "__main__":
    for name, fn in list(globals().items()):
        if name.startswith("test_"):
            fn()
            print("ok ", name)
