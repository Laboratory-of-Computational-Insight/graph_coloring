#!/usr/bin/env python3
"""
Diagnostic (2026-07-15): characterize EXACTLY how/where plain adjacency
spectral fails at c=5-9, to inform a baseline-conditioned model design
(warm-start from the classical phase-1+2 coloring, support-score-weighted
training, or a learned resolver replacing the brute-force/DSATUR fallback).

For each instance: run phase 1 (spectral init) + phase 2 (propagation), then
inspect the per-vertex support-score distribution and the size of the
residual "ambiguous" (low-support) vertex set BEFORE cleanup decides how to
handle it -- this is what determines whether cleanup succeeds via pure
brute-force, falls back to DSATUR, or fails outright.
"""
import numpy as np

from random_planted import create_planted_3col
from spectral_coloring import _spectral_init, _propagate, _support, _cleanup

for c in [5, 6, 7, 8, 9, 10]:
    n_inst = 8
    residual_fracs = []
    max_component_sizes = []
    outcomes = {"spectral": 0, "spectral+dsatur_fallback_ok": 0, "spectral+dsatur_fallback_fail": 0}
    support_hist = []

    for _ in range(n_inst):
        n = 1000
        assignment, adj = create_planted_3col(n, c)
        A = adj.numpy().astype(np.float64)

        colors = _spectral_init(A, n)
        propagation_iterations = max(1, int(np.ceil(np.log(max(n, 2)))))
        colors = _propagate(A, colors, propagation_iterations)

        supports = np.array([_support(A, colors, v) for v in range(n)])
        support_hist.append(supports)

        # residual set at threshold i=1 (roughly what cleanup starts probing)
        low_support_frac = float((supports < 1).mean())
        residual_fracs.append(low_support_frac)

        # accuracy of phase-1+2 coloring against the TRUE planted partition
        # (up to color permutation -- pick the permutation matching best)
        from itertools import permutations
        true_assign = assignment.numpy()
        best_acc = 0.0
        for perm in permutations(range(3)):
            remapped = np.array([perm[c_] for c_ in colors])
            acc = (remapped == true_assign).mean()
            best_acc = max(best_acc, acc)

        ok, final_colors, method = _cleanup(A, colors.copy(), max_component_size=18)
        if method == "spectral":
            outcomes["spectral"] += 1
        elif ok:
            outcomes["spectral+dsatur_fallback_ok"] += 1
        else:
            outcomes["spectral+dsatur_fallback_fail"] += 1

        print(f"c={c}  low_support(<1)_frac={low_support_frac:.3f}  "
              f"phase1+2_acc_vs_true={best_acc:.3f}  final_method={method if ok else method+'(FAILED)'}")

    print(f"--- c={c} summary over {n_inst} instances: "
          f"mean_low_support_frac={np.mean(residual_fracs):.3f}  outcomes={outcomes} ---\n")
