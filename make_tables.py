"""Tables of the paper from results/raw_results.csv.

    python make_tables.py

Writes results/table1.csv (Table 1), table2.csv and table3.csv (Appendix B)
and f1_at_fractions.csv (F1 and PC at fixed fractions of comparisons, Sect. 4).
All values are means over the 10 test sets.
"""
from pathlib import Path

import numpy as np
import pandas as pd

RESULTS = Path(__file__).resolve().parent / "results"
COLS = {"n_comparisons": "candidates", "pair_completeness": "PC", "pair_quality": "PQ",
        "f1_star": "F1*", "f1": "F1"}
FRACTIONS = (0.001, 0.005, 0.01, 0.02)
CURVES = {"title_df_graded": "Title index", "keys_graded": "Key index",
          "combined_graded": "One index (title + keys)", "title_mw_plain": "MSMP"}


def setting_means(raw: pd.DataFrame, method: str) -> pd.DataFrame:
    """Means per (bands, rows), from the strictest (25 x 20) to the loosest setting."""
    t = raw[raw.method == method].groupby(["bands", "rows"])[list(COLS)].mean().reset_index()
    t.insert(0, "setting", [f"{b}x{r}" for b, r in zip(t.bands, t.rows)])
    return t.sort_values("rows", ascending=False).reset_index(drop=True)


def tidy(t: pd.DataFrame, prefix: str = "") -> pd.DataFrame:
    t = t[list(COLS)].rename(columns=COLS).copy()
    t["candidates"] = t["candidates"].round().astype(int)
    t = t.round(3)
    return t.add_prefix(prefix)


def table1(raw: pd.DataFrame) -> pd.DataFrame:
    """Per method the strictest and loosest setting and the settings with the
    highest F1* and F1; all pairs with the classifier and with MSM."""
    def picks(method, block):
        t = setting_means(raw, method)
        roles = {}
        for role, k in (("strictest", t.index[0]), ("best F1*", t.f1_star.idxmax()),
                        ("best F1", t.f1.idxmax()), ("loosest", t.index[-1])):
            roles.setdefault(k, []).append(role)
        out = tidy(t.loc[sorted(roles)])
        out.insert(0, "role", [", ".join(roles[k]) for k in sorted(roles)])
        out.insert(0, "setting", t.setting[sorted(roles)].to_numpy())
        out.insert(0, "block", block)
        return out

    # all pairs: both rows report every pair of the test set (the candidates,
    # PC, PQ and F1* of the classifier row); only F1 differs
    ref = []
    all_pairs = raw[raw.method == "no_lsh"][list(COLS)].mean()
    for method, name in (("no_lsh", "Classifier"), ("msm_all", "MSM")):
        r = all_pairs.copy()
        r["f1"] = raw[raw.method == method].f1.mean()
        r = tidy(r.to_frame().T)
        r.insert(0, "role", "all pairs")
        r.insert(0, "setting", name)
        r.insert(0, "block", "All pairs, no LSH")
        ref.append(r)
    return pd.concat([picks("combined_graded", "One index (title + keys)"), *ref,
                      picks("title_mw_plain", "MSMP with our classifier"),
                      picks("msmp_msm", "MSMP with MSM")], ignore_index=True)


def table2(raw: pd.DataFrame) -> pd.DataFrame:
    """Title index and key index at all settings."""
    title, keys = setting_means(raw, "title_df_graded"), setting_means(raw, "keys_graded")
    return pd.concat([title[["setting"]], tidy(title, "title_"), tidy(keys, "key_")], axis=1)


def table3(raw: pd.DataFrame) -> pd.DataFrame:
    """One index and MSMP at all settings; MSMP's candidates with our
    classifier and with MSM."""
    one, mw, msm = (setting_means(raw, m) for m in ("combined_graded", "title_mw_plain", "msmp_msm"))
    msmp = tidy(mw, "msmp_").rename(columns={"msmp_F1": "msmp_F1_classifier"})
    msmp["msmp_F1_MSM"] = msm.f1.round(3)
    return pd.concat([one[["setting"]], tidy(one, "one_index_"), msmp], axis=1)


def at_fraction(d: pd.DataFrame, metric: str, target: float) -> float:
    """metric at the target fraction, linear in log fraction between the
    settings of one bootstrap; NaN outside the range of the curve."""
    d = d.sort_values("fraction")
    x = np.log(d.fraction.to_numpy())
    if not (x.min() <= np.log(target) <= x.max()):
        return np.nan
    return float(np.interp(np.log(target), x, d[metric].to_numpy()))


def f1_at_fractions(raw: pd.DataFrame) -> pd.DataFrame:
    """F1 and PC per bootstrap at FRACTIONS; mean and SD over the bootstraps
    whose curve reaches the fraction (n_bootstraps)."""
    out = []
    for method, name in CURVES.items():
        d = raw[(raw.method == method) & (raw.fraction > 0)]
        for t in FRACTIONS:
            row = dict(method=method, name=name, fraction=t)
            for col, short in (("f1", "f1"), ("pair_completeness", "pc")):
                vals = np.array([at_fraction(g, col, t) for _, g in d.groupby("bootstrap")])
                ok = vals[~np.isnan(vals)]
                row["n_bootstraps"] = len(ok)
                row[f"{short}_mean"] = ok.mean() if len(ok) else np.nan
                row[f"{short}_sd"] = ok.std(ddof=1) if len(ok) > 1 else np.nan
            out.append(row)
    cols = ["method", "name", "fraction", "n_bootstraps", "f1_mean", "f1_sd", "pc_mean", "pc_sd"]
    return pd.DataFrame(out)[cols].round(4)


def main():
    raw = pd.read_csv(RESULTS / "raw_results.csv")
    for name, table in (("table1", table1(raw)), ("table2", table2(raw)),
                        ("table3", table3(raw)), ("f1_at_fractions", f1_at_fractions(raw))):
        table.to_csv(RESULTS / f"{name}.csv", index=False)
        print(f"\n{name}.csv\n{table.to_string(index=False)}")


if __name__ == "__main__":
    main()
