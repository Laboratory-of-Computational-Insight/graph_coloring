#!/usr/bin/env python3
"""
Sweep the classical (non-GNN) baseline (classical_baseline.py) over planted
3-colorable graphs, using the exact same generator/methodology as
sweep_planted.py so results are directly comparable to the GNN sweeps.
"""
import argparse
import json
import os
import time

import numpy as np
from tqdm import tqdm

from random_planted import create_planted_3col
from classical_baseline import classical_baseline_color
from spectral_coloring import alon_kahale_3color


def parse_args():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--n", type=int, default=2000)
    p.add_argument("--c-min", type=int, default=1)
    p.add_argument("--c-max", type=int, default=30)
    p.add_argument("--n-inst", type=int, default=500)
    p.add_argument("--dsatur-restarts", type=int, default=30)
    p.add_argument("--results-file", type=str, default=None)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--method", choices=["peel_dsatur", "spectral"], default="peel_dsatur",
                    help="peel_dsatur: classical_baseline_color (Task 0a). "
                         "spectral: full Alon-Kahale algorithm (spectral init + propagation + cleanup, DSATUR fallback).")
    return p.parse_args()


def run_sweep(args):
    results_file = args.results_file or f"../results/classical_baseline_n{args.n}.json"
    os.makedirs(os.path.dirname(results_file), exist_ok=True)

    results = {}
    if os.path.exists(results_file):
        with open(results_file) as f:
            results = json.load(f)
        print(f"Resuming — {len(results)} c-values already done.")

    rng = np.random.default_rng(args.seed)

    for c in range(args.c_min, args.c_max + 1):
        key = str(c)
        if key in results:
            continue

        successes = 0
        method_counts = {}
        t0 = time.time()
        for _ in tqdm(range(args.n_inst), desc=f"c={c:2d}", leave=False):
            _, adj = create_planted_3col(args.n, c)
            if args.method == "spectral":
                out = alon_kahale_3color(adj)
            else:
                out = classical_baseline_color(adj, dsatur_restarts=args.dsatur_restarts, rng=rng)
            if out["success"]:
                successes += 1
                method_counts[out["method"]] = method_counts.get(out["method"], 0) + 1
        dt = time.time() - t0

        rate = successes / args.n_inst
        results[key] = {
            "successes": successes,
            "n_inst": args.n_inst,
            "rate": rate,
            "method_counts": method_counts,
            "time_sec": dt,
        }
        print(f"  c={c:2d}  success={rate:.3f}  {method_counts}  time={dt:.1f}s")

        with open(results_file, "w") as f:
            json.dump(results, f, indent=2)

    return results


if __name__ == "__main__":
    args = parse_args()
    print(f"n={args.n}  c={args.c_min}..{args.c_max}  {args.n_inst} instances/c  dsatur_restarts={args.dsatur_restarts}")
    run_sweep(args)
