"""One bootstrap: learn on the training set, evaluate on the test set."""
import numpy as np

from .classifier import PairClassifier
from .config import BR_GRID, DF_CUTOFFS, GRADE_SCALES, MIN_DF, N_HASHES, SEED
from .data import clean_title
from .evaluate import gold_pairs, lsh_metrics, prediction_metrics
from .features import FEATURE_NAMES, PairFeatures
from .keys import learn_key_model
from .lsh import MinHasher, decode_pairs, lsh_candidates
from .msm import run_msm
from .representation import (combine_title_key, expand_weighted, extract_brands, filter_by_df,
                             idf_weights, model_words)
from .transitivity import cluster_pairs, transitive_clusters

# LSH methods classified by the logistic regression: title index, key index,
# one index (title + keys) and MSMP
LR_METHODS = ("title_df_graded", "keys_graded", "combined_graded", "title_mw_plain")
# our methods: same-shop and brand filter after LSH (Sect. 3.5); the
# benchmarks title_mw_plain, msmp_msm, no_lsh and msm_all are not filtered
FILTERED_METHODS = ("title_df_graded", "keys_graded", "combined_graded")
METRICS = ["n_comparisons", "fraction", "dup_found", "pair_completeness", "pair_quality",
           "f1_star", "precision", "recall", "f1"]


def bootstrap_split(n: int, rng) -> tuple[np.ndarray, np.ndarray]:
    """n draws with replacement; training set = the distinct drawn products
    (np.unique: no product twice), test set = the products never drawn."""
    drawn = np.unique(rng.integers(0, n, size=n))
    return drawn, np.setdiff1d(np.arange(n), drawn)


class ProductSet:
    """A training or test set with everything that can be built without labels."""

    def __init__(self, products, hasher):
        self.products, self.hasher = products, hasher
        self.n = len(products)
        self.tokens = [clean_title(p.title, p.shop) for p in products]
        self.gold = gold_pairs([p.model_id for p in products])   # labels: training and evaluation
        brands = extract_brands(products, self.tokens)
        code = {b: k for k, b in enumerate(sorted({b for b in brands if b}))}
        self.brand = np.array([code.get(b, -1) for b in brands])
        shop_code = {s: k for k, s in enumerate(sorted({p.shop for p in products}))}
        self.shop = np.array([shop_code[p.shop] for p in products])
        model_code = {}
        self.labels = [model_code.setdefault(p.model_id, len(model_code)) for p in products]
        self._sig = {}

    def filter_candidates(self, pairs: np.ndarray) -> np.ndarray:
        """Drop pairs of one shop and pairs of two different known brands (Sect. 3.5)."""
        if len(pairs) == 0:
            return pairs
        i, j = decode_pairs(pairs, self.n)
        bi, bj = self.brand[i], self.brand[j]
        return pairs[(self.shop[i] != self.shop[j]) & ((bi < 0) | (bj < 0) | (bi == bj))]

    def brand_filter(self, pairs: np.ndarray) -> np.ndarray:
        """Only the brand rule (to choose the benchmarks' cut-off c)."""
        if len(pairs) == 0:
            return pairs
        i, j = decode_pairs(pairs, self.n)
        bi, bj = self.brand[i], self.brand[j]
        return pairs[(bi < 0) | (bj < 0) | (bi == bj)]

    def all_pairs(self) -> np.ndarray:
        i, j = np.triu_indices(self.n, k=1)
        return i.astype(np.int64) * self.n + j

    def title_sets(self, cutoff: float) -> list:
        """Title tokens after the df filter (Sect. 3.1)."""
        return filter_by_df(self.tokens, cutoff, MIN_DF)

    def graded_title_sets(self, cutoff: float, scale: float) -> list:
        """df-filtered title tokens with max(1, round(scale * idf)) copies (Sect. 3.3)."""
        idf = idf_weights(self.tokens)
        return [expand_weighted(toks, {t: scale * idf[t] for t in toks})
                for toks in self.title_sets(cutoff)]

    def candidates(self, name: str, sets, bands: int, rows: int, filtered: bool = False) -> np.ndarray:
        """LSH candidates of one representation (signatures cached by name);
        filtered after LSH if ``filtered``."""
        if name not in self._sig:
            self._sig[name] = self.hasher.signatures([set(s) for s in sets])
        sig, ok = self._sig[name]
        pairs = lsh_candidates(sig, ok, bands, rows)
        return self.filter_candidates(pairs) if filtered else pairs

    def f1_star(self, pairs: np.ndarray) -> float:
        return lsh_metrics(pairs, self.gold, self.n)["f1_star"]


def split_sets(products: list, boot: int):
    """The bootstrap's random generator, training set and test set."""
    rng = np.random.default_rng([SEED, boot])
    tr_idx, te_idx = bootstrap_split(len(products), rng)
    hasher = MinHasher(N_HASHES, seed=int(rng.integers(0, 2**31)))
    train = ProductSet([products[i] for i in tr_idx], hasher)
    test = ProductSet([products[i] for i in te_idx], hasher)
    return rng, train, test


def run_bootstrap(products: list, boot: int) -> tuple[list, dict]:
    """Test rows of every method and setting, and the training choices."""
    rng, train, test = split_sets(products, boot)

    # cut-off c (Sect. 3.1): best training F1* over all (b, r) of the df-filtered
    # title tokens; for our methods on the filtered candidates, for the
    # benchmarks' classifier features with the brand rule only
    c_scores, bench_c_scores = {}, {}
    for c in DF_CUTOFFS:
        sets = train.title_sets(c)
        c_scores[c] = max(train.f1_star(train.candidates(f"c{c}", sets, b, r, True))
                          for b, r in BR_GRID)
        bench_c_scores[c] = max(train.f1_star(train.brand_filter(train.candidates(
                                    f"c{c}", sets, b, r))) for b, r in BR_GRID)
    best_c = max(c_scores, key=c_scores.get)
    bench_c = max(bench_c_scores, key=bench_c_scores.get)

    # key model (Sect. 3.2)
    i, j = decode_pairs(train.gold, train.n)
    km = learn_key_model(train.products, list(zip(i.tolist(), j.tolist())), rng)

    def graded_key_sets(ps, scale):
        """key=value tokens of the selected keys with max(1, round(scale * w+)) copies."""
        out = []
        for p in ps.products:
            vals = km.canonical_values(p.features)
            toks = {f"{c}={vals[c]}": scale * km.weights[c]["w_agree"]
                    for c in km.selected if c in vals}
            out.append(expand_weighted(toks, toks))
        return out

    def combined_sets(ps, s, sk):
        return combine_title_key(ps.graded_title_sets(best_c, s), graded_key_sets(ps, sk))

    # scales (Sect. 3.3): best training F1* over all (b, r) on the filtered
    # candidates; the one index (title + keys) tunes (s, s_k) jointly (ties: first in grid order)
    def best_training_f1_star(name, sets):
        return max(train.f1_star(train.candidates(name, sets, b, r, True)) for b, r in BR_GRID)

    s_scores = {s: best_training_f1_star(f"title_s{s}", train.graded_title_sets(best_c, s))
                for s in GRADE_SCALES}
    best_s = max(s_scores, key=s_scores.get)
    sk_scores = {s: best_training_f1_star(f"keys_s{s}", graded_key_sets(train, s))
                 for s in GRADE_SCALES}
    best_sk = max(sk_scores, key=sk_scores.get)
    combo_scores = {(s, sk): best_training_f1_star(f"combined_s{s}_sk{sk}",
                                                   combined_sets(train, s, sk))
                    for s in GRADE_SCALES for sk in GRADE_SCALES}
    best_combined = max(combo_scores, key=combo_scores.get)

    # LSH candidates of every method and setting
    def representations(ps):
        return {"title_mw_plain": [model_words(p.title) for p in ps.products],
                "title_df_graded": ps.graded_title_sets(best_c, best_s),
                "keys_graded": graded_key_sets(ps, best_sk),
                "combined_graded": combined_sets(ps, *best_combined)}

    cand = {}
    for split, ps in (("train", train), ("test", test)):
        for method, sets in representations(ps).items():
            for b, r in BR_GRID:
                cand[split, method, (b, r)] = ps.candidates(method, sets, b, r,
                                                            method in FILTERED_METHODS)

    # classifier features: c for our methods, the benchmarks' c otherwise
    feats = {"bench": {s: PairFeatures(ps.products, ps.tokens, ps.title_sets(bench_c), km)
                       for s, ps in (("train", train), ("test", test))},
             "ours": {s: PairFeatures(ps.products, ps.tokens, ps.title_sets(best_c), km)
                      for s, ps in (("train", train), ("test", test))}}
    keep_cols = [k for k, f in enumerate(FEATURE_NAMES) if f != "same_shop"]

    def classify_and_score(method, train_pairs, test_pairs):
        """Logistic regression and transitivity step (Sect. 3.5); our filtered
        methods drop same_shop (always 0) and also use the brand rule."""
        ours = method in FILTERED_METHODS

        def features(split, pairs):
            X = feats["ours" if ours else "bench"][split].compute(pairs)
            return X[:, keep_cols] if ours else X

        def closure(ps, pairs, labels):
            a, b = decode_pairs(pairs, ps.n)
            return lambda prob, thresholds: transitive_clusters(
                ps.n, a, b, prob, thresholds, ps.shop, ps.brand if ours else None, labels)

        y = np.isin(train_pairs, train.gold, assume_unique=True).astype(int)
        clf = PairClassifier().fit(features("train", train_pairs), y, len(train.gold),
                                   closure(train, train_pairs, train.labels))
        prob = clf.predict_proba(features("test", test_pairs))
        _, clusters = closure(test, test_pairs, None)(prob, [clf.threshold])
        row = lsh_metrics(test_pairs, test.gold, test.n)
        row.update(prediction_metrics(cluster_pairs(clusters, test.n), test.gold))
        return row, float(clf.threshold)

    rows, thresholds = [], {}
    for method in LR_METHODS:
        thresholds[method] = {}
        for b, r in BR_GRID:
            row, thr = classify_and_score(method, cand["train", method, (b, r)],
                                          cand["test", method, (b, r)])
            rows.append(dict(bootstrap=boot, method=method, bands=b, rows=r, **row))
            thresholds[method][f"{b}x{r}"] = thr
    row, thr = classify_and_score("no_lsh", train.all_pairs(), test.all_pairs())
    rows.append(dict(bootstrap=boot, method="no_lsh", bands=0, rows=0, **row))
    thresholds["no_lsh"] = {"all pairs": thr}

    # MSM on MSMP's candidates and on all pairs
    msm_cand = {(split, br): cand[split, "title_mw_plain", br]
                for split in ("train", "test") for br in BR_GRID}
    msm_rows, msm_params = run_msm(products, train, test, msm_cand, BR_GRID)
    rows += [dict(bootstrap=boot, **row) for row in msm_rows]

    info = dict(bootstrap=boot, df_cutoff=best_c, df_cutoff_benchmarks=bench_c,
                title_scale=best_s, key_scale=best_sk,
                combined_scales=dict(s=best_combined[0], s_k=best_combined[1]),
                selected_keys=km.selected, key_groups=km.merged_groups,
                thresholds=thresholds, msm_parameters=msm_params)
    return [{k: row[k] for k in ["bootstrap", "method", "bands", "rows"] + METRICS}
            for row in rows], info
