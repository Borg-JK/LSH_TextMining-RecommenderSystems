"""Run the 10 bootstraps and write results/raw_results.csv, results/summary.csv
and results/learned_per_bootstrap.json.

    python run_experiment.py --jobs 4
"""
import argparse
import json
import sys
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "src"))

import pandas as pd  # noqa: E402

from dupdet.config import N_BOOTSTRAPS  # noqa: E402
from dupdet.data import load_products  # noqa: E402
from dupdet.pipeline import METRICS, run_bootstrap  # noqa: E402

DATA = ROOT / "data" / "TVs-all-merged.json"
OUT = ROOT / "results"


def _one(boot: int):
    t0 = time.time()
    rows, info = run_bootstrap(load_products(str(DATA)), boot)
    return rows, info, time.time() - t0


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--jobs", type=int, default=1, help="parallel processes")
    jobs = ap.parse_args().jobs

    OUT.mkdir(exist_ok=True)
    rows, infos = [], []
    with ProcessPoolExecutor(max_workers=jobs) as pool:
        for r, info, seconds in pool.map(_one, range(N_BOOTSTRAPS)):
            rows += r
            infos.append(info)
            print(f"bootstrap {info['bootstrap']}: {seconds:.0f} s", flush=True)

    raw = pd.DataFrame(rows)
    raw.to_csv(OUT / "raw_results.csv", index=False)
    g = raw.groupby(["method", "bands", "rows"], sort=False)[METRICS]
    summary = g.mean().join(g.std().add_suffix("_sd")).reset_index()
    summary.to_csv(OUT / "summary.csv", index=False)
    with open(OUT / "learned_per_bootstrap.json", "w") as fh:
        json.dump(infos, fh, indent=1)
    print(f"wrote {OUT}/raw_results.csv, summary.csv, learned_per_bootstrap.json")


if __name__ == "__main__":
    main()
