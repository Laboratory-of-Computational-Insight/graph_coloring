"""
For each c-band (easy-below-shattering, hard [4, 4.69], easy-again-above),
run BP with the "whatever works first" OR-strategy: try plain BP (rho=0)
first (a handful of restarts), fall back to BP+reinforcement (rho=0.003) if
plain doesn't find a zero-conflict coloring, and report the typical distance
of the resulting coloring from the PLANTED ground truth (not the nearest
legal coloring -- the actual hidden partition used to generate the graph).

Distance = 1 - overlap, where overlap is computed under the best of the 6
color-relabelings (colors are unlabeled / arbitrary, so raw label-by-label
Hamming distance would be meaningless without this).

0.0 = perfect recovery of the planted partition.
0.667 = random guessing baseline for 3 balanced classes (2/3 expected
mismatch if colors were assigned uniformly at random).
"""
import argparse
import itertools
import json

import numpy as np
import torch

from random_planted import create_planted_3col
from belief_propagation_coloring import bp_reinforced_coloring
from repair_utils import count_conflicts

N = 1000
MAX_RESTARTS_PER_METHOD = 5
RESULTS_DIR = "G:/graph_col/graph_coloring/results"

# the three named regimes, plus a couple of extra points inside the hard
# band and at the boundaries for a clearer trend
C_VALUES = [2, 3, 4, 4.3, 4.5, 4.69, 5, 6, 8]

PERMS = list(itertools.permutations(range(3)))


def distance_from_planted(colors, planted):
    """1 - best-permutation overlap fraction."""
    colors = np.asarray(colors)
    planted = np.asarray(planted)
    best_match = 0
    for perm in PERMS:
        remapped = np.array(perm)[colors]
        match = (remapped == planted).sum()
        best_match = max(best_match, match)
    return 1.0 - best_match / len(planted)


def try_bp(A, rho, seed):
    belief, _ = bp_reinforced_coloring(A, rho=rho, max_iter=3000, seed=seed)
    colors = belief.argmax(axis=1)
    conf = count_conflicts(A, colors)
    return colors, conf


def solve_whatever_works_first(A, seed_base):
    """Try plain BP restarts first, then reinforced BP restarts, OR-combined
    on the first zero-conflict result. If neither succeeds, return the best
    (lowest-conflict) attempt seen across both, tagged as a failure."""
    best_colors, best_conf, best_method = None, None, None

    for method, rho in [("plain", 0.0), ("reinforced", 0.003)]:
        for r in range(MAX_RESTARTS_PER_METHOD):
            seed = seed_base * 1000 + r
            colors, conf = try_bp(A, rho, seed)
            if best_conf is None or conf < best_conf:
                best_colors, best_conf, best_method = colors, conf, method
            if conf == 0:
                return colors, method, r + 1, True
    return best_colors, best_method, MAX_RESTARTS_PER_METHOD, False


def parse_args():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--c-values", type=float, nargs="+", default=C_VALUES)
    p.add_argument("--n-inst", type=int, default=20)
    p.add_argument("--out", default=f"{RESULTS_DIR}/bp_distance_from_planted.json")
    return p.parse_args()


def main():
    args = parse_args()
    torch.manual_seed(0)
    results = {}
    print(f"{'c':>6}{'n_success':>12}{'mean_dist_success':>20}{'mean_dist_ALL':>16}{'method_used(plain/reinf)':>28}")

    for c in args.c_values:
        c_disp = int(c) if c == int(c) else round(c, 4)
        dists_success = []
        dists_all = []
        methods_used = {"plain": 0, "reinforced": 0}
        n_success = 0
        per_inst = []

        for i in range(args.n_inst):
            planted, adj = create_planted_3col(N, c)
            A = adj.numpy()
            planted_np = planted.numpy()

            colors, method, restarts_used, success = solve_whatever_works_first(A, seed_base=i)
            dist = distance_from_planted(colors, planted_np)

            dists_all.append(dist)
            if success:
                dists_success.append(dist)
                methods_used[method] += 1
                n_success += 1

            per_inst.append({"success": success, "method": method, "restarts": restarts_used, "distance": dist})

        mean_success = np.mean(dists_success) if dists_success else float("nan")
        mean_all = np.mean(dists_all)
        method_str = f"{methods_used['plain']}/{methods_used['reinforced']}"
        print(f"{c_disp!s:>6}{f'{n_success}/{args.n_inst}':>12}{mean_success:>20.4f}{mean_all:>16.4f}{method_str:>28}",
              flush=True)

        results[str(c_disp)] = {
            "n_success": n_success,
            "mean_distance_success_only": None if dists_success == [] else float(mean_success),
            "mean_distance_all": float(mean_all),
            "methods_used": methods_used,
            "per_instance": per_inst,
        }
        with open(args.out, "w") as f:
            json.dump(results, f, indent=2)

    print(f"\nsaved to {args.out}")


if __name__ == "__main__":
    main()
