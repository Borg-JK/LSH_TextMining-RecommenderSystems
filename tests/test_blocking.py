"""Checks of the same-shop and brand filter after LSH and of the brand rule.
Run: python tests/test_blocking.py"""
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from dupdet.config import BR_GRID  # noqa: E402
from dupdet.data import Product, clean_title, load_products  # noqa: E402
from dupdet.evaluate import lsh_metrics  # noqa: E402
from dupdet.lsh import MinHasher, decode_pairs, lsh_candidates  # noqa: E402
from dupdet.pipeline import FILTERED_METHODS, LR_METHODS, ProductSet  # noqa: E402
from dupdet.representation import extract_brands  # noqa: E402


def test_filter_only_for_our_methods():
    assert set(FILTERED_METHODS) == {"title_df_graded", "keys_graded", "combined_graded"}
    assert "title_mw_plain" in LR_METHODS and "title_mw_plain" not in FILTERED_METHODS


def test_filter_drops_only_different_known_brands():
    def tv(i, shop, title, brand=None):
        return Product(i, shop, title, {"Brand": brand} if brand else {}, f"m{i}")
    products = [tv(0, "a", "Samsung 46 TV", "Samsung"), tv(1, "b", "Samsung 46 HDTV"),
                tv(2, "c", "LG 46 TV", "LG"), tv(3, "d", "Unknown 46 TV")]
    ps = ProductSet(products, MinHasher(10, seed=0))
    pairs = np.array([a * 4 + b for a in range(4) for b in range(a + 1, 4)])
    kept = set(ps.filter_candidates(pairs).tolist())
    assert kept == {0 * 4 + 1, 0 * 4 + 3, 1 * 4 + 3, 2 * 4 + 3}     # (0,2) and (1,2) differ
    assert len(ps.all_pairs()) == 6                                  # no_lsh: not filtered


def test_no_same_shop_candidate_and_benchmarks_unchanged():
    products = load_products(str(ROOT / "data" / "TVs-all-merged.json"))[::4]
    ps = ProductSet(products, MinHasher(500, seed=1))
    same_shop = 0
    for b, r in BR_GRID:
        raw = lsh_candidates(*ps.hasher.signatures([set(t) for t in ps.tokens]), b, r)
        kept = ps.candidates("title", ps.tokens, b, r, filtered=True)
        i, j = decode_pairs(kept, ps.n)
        assert (ps.shop[i] != ps.shop[j]).all()                       # filtered method
        assert np.array_equal(ps.candidates("title", ps.tokens, b, r), raw)   # benchmark
        assert set(kept.tolist()) <= set(raw.tolist())
        i, j = decode_pairs(raw, ps.n)
        same_shop += int((ps.shop[i] == ps.shop[j]).sum())
    assert same_shop > 0                                               # the filter had work


def test_pc_never_counts_within_shop_gold_pairs():
    # products 0, 1 (shop a) and 2 (shop b) are one TV; 3 is another
    def tv(i, shop, model):
        return Product(i, shop, "Samsung 46 TV", {"Brand": "Samsung"}, model)
    products = [tv(0, "a", "m1"), tv(1, "a", "m1"), tv(2, "b", "m1"), tv(3, "c", "m2")]
    ps = ProductSet(products, MinHasher(10, seed=0))
    kept = ps.filter_candidates(ps.all_pairs())
    assert 0 * 4 + 1 not in set(kept.tolist())                         # within-shop pair dropped
    m = lsh_metrics(kept, ps.gold, ps.n)
    within = 1                                                         # gold: (0,1), (0,2), (1,2)
    assert m["dup_found"] == 2 and abs(m["pair_completeness"] - (1 - within / len(ps.gold))) < 1e-12


def test_brand_list_coverage_and_no_wrong_block():
    products = load_products(str(ROOT / "data" / "TVs-all-merged.json"))
    tokens = [clean_title(p.title, p.shop) for p in products]
    before = extract_brands(products, tokens, extra_brands=())
    after = extract_brands(products, tokens)
    assert sum(b is None for b in before) == 82 and sum(b is None for b in after) == 2
    by_model = defaultdict(list)
    for k, p in enumerate(products):
        by_model[p.model_id].append(k)
    for members in by_model.values():                # no duplicate pair has two known brands
        known = {after[k] for k in members if after[k]}
        assert len(known) <= 1, known


if __name__ == "__main__":
    for name, fn in list(globals().items()):
        if name.startswith("test_"):
            fn()
            print("ok ", name)
