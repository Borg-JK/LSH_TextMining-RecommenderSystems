"""Checks of the bootstrap split and of the data.  Run: python tests/test_bootstrap.py"""
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from dupdet.config import N_BOOTSTRAPS, SEED  # noqa: E402
from dupdet.data import load_products  # noqa: E402
from dupdet.pipeline import bootstrap_split  # noqa: E402

N = 1624


def splits():
    for boot in range(N_BOOTSTRAPS):              # the generators of the main run
        yield bootstrap_split(N, np.random.default_rng([SEED, boot]))


def test_no_repeated_products():
    for train, test in splits():
        assert len(np.unique(train)) == len(train) and len(np.unique(test)) == len(test)


def test_train_and_test_disjoint_and_complete():
    for train, test in splits():
        assert len(np.intersect1d(train, test)) == 0
        assert np.array_equal(np.union1d(train, test), np.arange(N))


def test_training_share_about_63_percent():
    shares = [len(train) / N for train, _ in splits()]
    assert all(0.60 < s < 0.66 for s in shares), shares        # 1 - 1/e = 0.632
    assert abs(np.mean(shares) - (1 - np.exp(-1))) < 0.01


def test_identical_listings_and_within_shop_duplicates():
    with open(ROOT / "data" / "TVs-all-merged.json", encoding="utf-8") as fh:
        items = [p for group in json.load(fh).values() for p in group]
    # listings with the same shop, title and features: 4 groups (10 listings),
    # every listing of a group has a different modelID, so no gold pair
    # consists of two identical listings
    groups = defaultdict(list)
    for p in items:
        groups[p["shop"], p["title"], json.dumps(p["featuresMap"], sort_keys=True)].append(p["modelID"])
    same = [g for g in groups.values() if len(g) > 1]
    assert len(same) == 4 and sum(len(g) for g in same) == 10
    assert all(len(set(g)) == len(g) for g in same)
    # 12 gold pairs within one shop (11 modelIDs); they stay in the gold standard
    products = load_products(str(ROOT / "data" / "TVs-all-merged.json"))
    by_model = defaultdict(list)
    for p in products:
        by_model[p.model_id].append(p.shop)
    within = Counter(s for shops in by_model.values()
                     for k, s in enumerate(shops) for t in shops[k + 1:] if s == t)
    assert sum(within.values()) == 12 and within == Counter({"newegg.com": 8, "bestbuy.com": 4})
    assert sum(len(s) * (len(s) - 1) // 2 for s in by_model.values()) == 399


if __name__ == "__main__":
    for name, fn in list(globals().items()):
        if name.startswith("test_"):
            fn()
            print("ok ", name)
