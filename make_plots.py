"""Figure 1 of the paper from results/raw_results.csv.

    python make_plots.py

Writes paper/figures/results_curves.pdf and results/results_curves.png:
(a) F1* and (b) F1 against the fraction of comparisons; one point per
(bands, rows) setting (mean over the test sets), band = mean +/- 1 SD.
"""
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

ROOT = Path(__file__).resolve().parent
RESULTS = ROOT / "results"
FIGURES = ROOT / "paper" / "figures"

# (method, label, colour, line style, marker), in legend order
CURVES = [
    ("title_df_graded", "Title index", "#2a78d6", "-", "o"),
    ("keys_graded", "Key index", "#eb6834", "-", "s"),
    ("combined_graded", "One index (title + keys)", "#c2185b", "-", "v"),
    ("title_mw_plain", "MSMP (title model words)", "#6250d6", "--", "x"),
]
XLIM = (0.0005, 150)                 # % of comparisons
XTICKS = [0.001, 0.01, 0.1, 1, 10, 100]
MIN_BOOTSTRAPS = 5                   # points reached by fewer bootstraps are dropped
GREY = "#52514e"


def curve_points(raw: pd.DataFrame, method: str) -> pd.DataFrame:
    """Per (bands, rows): mean fraction, mean and SD of F1* and F1 over the test sets."""
    t = raw[raw.method == method].groupby(["bands", "rows"]).agg(
        fraction=("fraction", "mean"), n=("bootstrap", "nunique"),
        f1_star=("f1_star", "mean"), f1_star_sd=("f1_star", "std"),
        f1=("f1", "mean"), f1_sd=("f1", "std")).reset_index()
    return t[t.n >= MIN_BOOTSTRAPS].sort_values("fraction")


def main():
    raw = pd.read_csv(RESULTS / "raw_results.csv")
    plt.rcParams.update({"font.size": 7.5, "font.family": "serif", "axes.titlesize": 8})
    fig, axes = plt.subplots(1, 2, figsize=(4.8, 2.3))
    for ax, (metric, title) in zip(axes, [("f1_star", "(a) $F_1^*$"), ("f1", "(b) $F_1$")]):
        for method, label, colour, ls, marker in CURVES:
            cv = curve_points(raw, method)
            x, y, sd = cv.fraction * 100, cv[metric], cv[f"{metric}_sd"].fillna(0)
            ax.fill_between(x, np.clip(y - sd, 0, 1), np.clip(y + sd, 0, 1), color=colour,
                            alpha=0.15, lw=0)
            ax.plot(x, y, ls, color=colour, marker=marker, ms=2.5, lw=1.1, label=label)
        if metric == "f1":                       # classifier on all pairs, without LSH
            ref = raw[raw.method == "no_lsh"].f1
            ax.errorbar([100], [ref.mean()], yerr=[ref.std()], fmt="s", color=GREY, ms=3,
                        capsize=2, lw=0.9)
            ax.annotate("all pairs", (100, ref.mean() - ref.std()), xytext=(0, -8),
                        textcoords="offset points", ha="center", fontsize=6, color=GREY)
        ax.set_xscale("log")
        ax.set_xlim(*XLIM)
        ax.set_xticks(XTICKS)
        ax.set_xticklabels([f"{t:g}" for t in XTICKS])
        ax.minorticks_off()
        ax.set_ylim(bottom=0)
        ax.set_title(title, loc="left")
        ax.set_ylabel("$F_1^*$" if metric == "f1_star" else "$F_1$")
        ax.set_xlabel("Fraction of comparisons (%)")
        ax.grid(True, color="#e4e3df", lw=0.5)
        ax.set_axisbelow(True)
        for side in ("top", "right"):
            ax.spines[side].set_visible(False)
        ax.tick_params(length=2)
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=4, frameon=False, fontsize=6.5)
    fig.tight_layout(rect=(0, 0.08, 1, 1))
    FIGURES.mkdir(parents=True, exist_ok=True)
    fig.savefig(FIGURES / "results_curves.pdf", bbox_inches="tight", pad_inches=0.02)
    fig.savefig(RESULTS / "results_curves.png", dpi=300, bbox_inches="tight", pad_inches=0.02)
    plt.close(fig)
    print(f"wrote {FIGURES / 'results_curves.pdf'} and {RESULTS / 'results_curves.png'}")


if __name__ == "__main__":
    main()
