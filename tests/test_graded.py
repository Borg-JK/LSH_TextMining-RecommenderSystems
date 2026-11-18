"""Checks of the graded-weight expansion.  Run: python tests/test_graded.py"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from dupdet.representation import combine_title_key, expand_weighted, graded_copies  # noqa: E402


def jaccard(a: set, b: set) -> float:
    return len(a & b) / len(a | b) if a | b else 0.0


def weighted_jaccard(wa: dict, wb: dict) -> float:
    keys = wa.keys() | wb.keys()
    return (sum(min(wa.get(t, 0), wb.get(t, 0)) for t in keys)
            / sum(max(wa.get(t, 0), wb.get(t, 0)) for t in keys))


def test_expanded_jaccard_equals_weighted_jaccard():
    cases = [({"samsung": 3, "46inch": 2, "led": 1}, {"samsung": 1, "46inch": 2, "lcd": 4}),
             ({"a": 5, "b": 1}, {"a": 2, "b": 3, "c": 1}),
             ({"x": 2}, {"x": 2})]
    for wa, wb in cases:
        ea, eb = expand_weighted(wa, wa), expand_weighted(wb, wb)
        assert abs(jaccard(ea, eb) - weighted_jaccard(wa, wb)) < 1e-12, (wa, wb)


def test_unit_weights_keep_jaccard():
    a, b = {"samsung", "46inch", "led", "hdtv"}, {"samsung", "55inch", "led"}
    ea = expand_weighted(a, {t: 1 for t in a})
    eb = expand_weighted(b, {t: 1 for t in b})
    assert len(ea) == len(a) and len(eb) == len(b)
    assert jaccard(ea, eb) == jaccard(a, b)


def test_every_token_gets_a_copy():
    weights = {"zero": 0.0, "negative": -2.3, "small": 0.4, "half": 0.5, "big": 3.6}
    expanded = expand_weighted(weights, weights)
    for t, w in weights.items():
        n = sum(x.startswith(f"{t}#") for x in expanded)
        assert n == graded_copies(w) >= 1, (t, w, n)
    assert graded_copies(3.6) == 4


def test_combined_prefixes_keep_title_and_keys_apart():
    # the same string as a title token and as a key token must stay two elements
    title = [expand_weighted({"46inch"}, {"46inch": 2})]
    keys = [expand_weighted({"46inch"}, {"46inch": 1})]
    combined = combine_title_key(title, keys)[0]
    assert combined == {"t:46inch#1", "t:46inch#2", "k:46inch#1"}


def test_combined_without_key_tokens():
    title = [expand_weighted({"samsung", "un46"}, {"samsung": 1, "un46": 3})]
    combined = combine_title_key(title, [set()])[0]
    assert combined == {"t:samsung#1", "t:un46#1", "t:un46#2", "t:un46#3"}


def test_combined_jaccard_by_hand():
    # product A: title {samsung: 1, un46: 2}, keys {brand=samsung: 3}
    # product B: title {samsung: 1, un55: 2}, keys {brand=samsung: 2}
    def sets(title_w, key_w):
        return combine_title_key([expand_weighted(title_w, title_w)],
                                 [expand_weighted(key_w, key_w)])[0]
    a = sets({"samsung": 1, "un46": 2}, {"brand=samsung": 3})
    b = sets({"samsung": 1, "un55": 2}, {"brand=samsung": 2})
    # shared: t:samsung#1, k:brand=samsung#1..2 -> 3; union: 1 + 2 + 2 + 3 = 8
    assert jaccard(a, b) == 3 / 8
    assert len(a) == 6 and len(b) == 5


if __name__ == "__main__":
    for name, fn in list(globals().items()):
        if name.startswith("test_"):
            fn()
            print("ok ", name)
