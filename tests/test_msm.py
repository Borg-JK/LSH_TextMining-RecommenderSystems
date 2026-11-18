"""Checks of the MSM benchmark.  Run: python tests/test_msm.py"""
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from dupdet.config import MSM_BETA, MSM_GRID, MSM_Q  # noqa: E402
from dupdet.data import Product, load_products  # noqa: E402
from dupdet.msm import (MSMIndex, cluster, hsim, kvp_part, mw_share, preprocess,  # noqa: E402
                        qgram_sim, title_model_words, title_sim, value_model_words)


def test_qgram_sim_identical_is_one():
    for s in ("", "a", "screen size", "UN46EH5000", "1920 x 1080"):
        assert qgram_sim(s, s) == 1.0
    assert 0 < qgram_sim("screen size", "screen size class") < 1
    assert qgram_sim("abc", "xyz") == 0.0


def test_title_sim_conflicting_model_words():
    t1, t2 = preprocess("Samsung 46inch UN46"), preprocess("Samsung 55inch UN55")
    for alpha in (0.5, 0.7, 0.9):
        assert title_sim(t1, t2, alpha) == -1.0


def test_cluster_never_merges_same_shop():
    # 0 and 2 are from the same shop; the chain 0 - 1 - 2 must not join them
    shop = ["a", "b", "a", "c"]

    def allowed(x, y):
        return shop[x] != shop[y]
    ii, jj, d = [0, 1, 1], [1, 2, 3], [0.1, 0.1, 0.2]
    _, clusters = cluster(4, ii, jj, d, [0.5], allowed)
    for c in clusters:
        assert len({shop[x] for x in c}) == len(c), clusters
    assert sorted(map(sorted, clusters)) == [[0, 1, 3]]

    # the same on real products: all cross-shop pairs of 300 products, loose settings
    products = load_products(str(ROOT / "data" / "TVs-all-merged.json"))
    index = MSMIndex(products)
    sample = np.random.default_rng(0).choice(len(products), 300, replace=False)
    sample.sort()
    shops = [products[k].shop for k in sample]
    i, j = np.triu_indices(len(sample), k=1)
    ok = np.array([shops[a] != shops[b] for a, b in zip(i, j)])
    i, j = i[ok], j[ok]
    parts = index.parts(sample[i], sample[j])
    dist = 1 - parts.similarity(0.5, 0, 0.5)
    _, clusters = cluster(len(sample), i, j, dist, [0.7], lambda x, y: shops[x] != shops[y])
    assert clusters
    for c in clusters:
        assert len({shops[x] for x in c}) == len(c)


def test_four_products_known_clusters():
    # two TVs, each listed by shops a and b; no brand information
    def tv(idx, shop, title, size, model, hz, keys):
        return Product(idx, shop, title, dict(zip(keys, [size, model, hz])), model)
    k1 = ["Screen Size", "Model", "Refresh Rate"]
    k2 = ["Screen Size Class", "Model Number", "Refresh Rate"]
    products = [
        tv(0, "a", "Samsung 46inch UN46EH5000 LED HDTV", "46 in", "UN46EH5000", "60Hz", k1),
        tv(1, "b", "Samsung UN46EH5000 46inch LED TV", "46 in", "UN46EH5000", "60Hz", k2),
        tv(2, "a", "LG 55inch 55LM4600 LED HDTV", "55 in", "55LM4600", "120Hz", k1),
        tv(3, "b", "LG 55LM4600 55inch LED TV", "55 in", "55LM4600", "120Hz", k2),
    ]
    index = MSMIndex(products)
    i, j = np.array([0, 0, 1, 2]), np.array([1, 3, 2, 3])     # pairs across shops
    parts = index.parts(i, j)
    dist = 1 - parts.similarity(0.7, 6, 0.6)                   # alpha, gamma = 0.6, mu
    _, clusters = cluster(4, i, j, dist, [0.5], lambda x, y: products[x].shop != products[y].shop)
    assert sorted(map(sorted, clusters)) == [[0, 1], [2, 3]], (dist, clusters)


def test_preprocessing():
    # punctuation becomes a space, so '(2012' and '(40' are no model words with
    # the non-numeric part '('
    assert preprocess("Sony Bravia (2012) KDL40") == "sony bravia 2012 kdl40"
    assert title_model_words(preprocess('Samsung 46" Class (45.9" Diag.) (2012)')) == ['46"', '45.9']
    assert preprocess('49-1/2"') == '49-1 2"'
    assert preprocess("A [b]; c: d! e? f+g & h*i") == "a b c d e f g h i"
    a, b = preprocess("Sony Bravia (2012) KDL40"), preprocess('Sony Bravia KDL40 (40" Diag.)')
    assert title_sim(a, b, alpha=1.0) != -1.0


def test_value_model_words_and_share():
    assert value_model_words("46inch 120hz 1920x1080 60 2.5w") == {"46", "120", "60", "2.5"}
    assert mw_share({"46", "120", "60"}, {"46", "60"}) == 2 / 3
    assert mw_share(set(), set()) == 0.0
    kv_i = [("screen size", "46"), ("screen size class", "46")]
    kv_j = [("screen size", "46")]
    assert kvp_part(kv_i, kv_j, 0.5)[0] == 1                   # one-to-one matching


def test_vectorised_matches_reference():
    products = load_products(str(ROOT / "data" / "TVs-all-merged.json"))
    index = MSMIndex(products)
    rng = np.random.default_rng(1)
    gi, gj = rng.integers(0, len(products), 150), rng.integers(0, len(products), 150)
    keep = gi != gj
    gi, gj = np.minimum(gi, gj)[keep], np.maximum(gi, gj)[keep]
    parts = index.parts(gi, gj)
    kv = {p.idx: [(preprocess(k), preprocess(v)) for k, v in p.features.items()] for p in products}
    for k, (a, b) in enumerate(zip(gi.tolist(), gj.tolist())):
        ta, tb = preprocess(products[a].title), preprocess(products[b].title)
        for alpha in MSM_GRID:
            assert abs(parts.title_sim(alpha)[k] - title_sim(ta, tb, alpha, MSM_BETA)) < 1e-9
        for g, gamma in enumerate(MSM_GRID):
            m, avg, mwp, minf = kvp_part(kv[a], kv[b], gamma, MSM_Q)
            assert parts.m[g][k] == m and parts.min_features[k] == minf
            assert abs(parts.avg_sim[g][k] - avg) < 1e-9 and abs(parts.mw_perc[g][k] - mwp) < 1e-12
            for mu in MSM_GRID:
                ref = hsim(m, avg, mwp, minf, title_sim(ta, tb, 0.7, MSM_BETA), mu)
                assert abs(parts.similarity(0.7, g, mu)[k] - ref) < 1e-9


if __name__ == "__main__":
    for name, fn in list(globals().items()):
        if name.startswith("test_"):
            fn()
            print("ok ", name)
