"""
For successful BP colorings in the hard band (c=4.4-4.69), look at the set
of vertices that disagree with the planted partition (after best-permutation
alignment) and ask: how many of those disagreeing vertices can be flipped
to their planted color WITHOUT creating a new conflict against the rest of
the (unmodified) current coloring?

Two measures:
  - one-shot: check each disagreeing vertex's flip independently against the
    CURRENT coloring (no cascade) -- the strict "is this single vertex's
    wrongness free-standing" test.
  - greedy closure: repeatedly scan the disagreement set and flip whichever
    vertices are currently safe, updating the working coloring as we go,
    until no more flips are safe -- since flipping one vertex can free up a
    neighbor that wasn't independently flippable before.

If most disagreeing vertices flip freely: BP's solution sits in the same
basin as the planted one, just with a lot of locally-arbitrary micro-choices
layered on top. If few flip (even after the greedy closure): BP's solution
is genuinely a different, structurally incompatible coloring, not a "noisy"
version of the planted one -- real cluster-level disagreement, not
individually-fixable errors.
"""
import itertools
import multiprocessing as mp

import numpy as np
import torch

from random_planted import create_planted_3col
from belief_propagation_coloring import bp_reinforced_coloring
from repair_utils import count_conflicts, greedy_repair

N = 1000
N_INST = 5
C_VALUES = [4.4, 4.5, 4.6, 4.69]
CAP = 100
N_WORKERS = 16
PERMS = list(itertools.permutations(range(3)))

RHO_VALUES = [0.001, 0.003, 0.01, 0.03, 0.1]
DAMPING_VALUES = [0.3, 0.5, 0.7]
HYPER_COMBOS = [(r, d) for r in RHO_VALUES for d in DAMPING_VALUES]


def best_perm_align(colors, planted, q=3):
    """Returns colors relabeled so they best match planted (not just the
    distance -- we need the actual aligned array for the flip analysis)."""
    best_perm, best_match = None, -1
    for perm in PERMS:
        remapped = np.array(perm)[colors]
        match = (remapped == planted).sum()
        if match > best_match:
            best_match, best_perm = match, perm
    return np.array(best_perm)[colors], best_match / len(planted)


def _run_one(args):
    A, rho, damping, seed = args
    belief, _ = bp_reinforced_coloring(A, rho=rho, damping=damping, max_iter=3000, seed=seed)
    colors = belief.argmax(axis=1)
    conf = count_conflicts(A, colors)
    if conf > 0:
        colors = greedy_repair(A, colors)
        conf = count_conflicts(A, colors)
    return seed, colors, conf


def solve_escalation(A, seed_base, pool, cap_total=CAP):
    per_rung = cap_total // 3
    plans = ([(0.0, 0.5, seed_base * 100000 + i) for i in range(per_rung)] +
             [(0.003, 0.5, seed_base * 100000 + per_rung + i) for i in range(per_rung)] +
             [(HYPER_COMBOS[i % len(HYPER_COMBOS)][0], HYPER_COMBOS[i % len(HYPER_COMBOS)][1],
               seed_base * 100000 + 2 * per_rung + i) for i in range(cap_total - 2 * per_rung)])
    args = [(A, rho, damping, seed) for (rho, damping, seed) in plans]
    for seed, colors, conf in pool.imap_unordered(_run_one, args, chunksize=1):
        if conf == 0:
            return colors, True
    return None, False


def flip_analysis(A, colors_aligned, planted, n):
    """colors_aligned: BP's coloring, already best-permutation-aligned to
    planted's labeling (so 'flip to planted color' is well-defined)."""
    neighbors = [np.where(A[v] != 0)[0] for v in range(n)]
    disagree = np.where(colors_aligned != planted)[0]

    # one-shot: check each disagreeing vertex independently against the
    # UNMODIFIED current coloring
    working = colors_aligned.copy()
    one_shot_safe = []
    for v in disagree:
        target = planted[v]
        neighbor_colors = working[neighbors[v]]
        if target not in neighbor_colors:
            one_shot_safe.append(v)

    # greedy closure: repeatedly flip whatever's currently safe
    working2 = colors_aligned.copy()
    remaining = set(disagree.tolist())
    flipped_closure = []
    changed = True
    while changed and remaining:
        changed = False
        for v in list(remaining):
            target = planted[v]
            neighbor_colors = working2[neighbors[v]]
            if target not in neighbor_colors:
                working2[v] = target
                flipped_closure.append(v)
                remaining.discard(v)
                changed = True

    return len(disagree), len(one_shot_safe), len(flipped_closure)


def main():
    torch.manual_seed(0)
    print(f"{'c':>6}{'inst':>6}{'n_disagree':>12}{'one_shot_safe':>15}{'greedy_closure':>16}"
          f"{'frac_oneshot':>14}{'frac_closure':>14}")

    with mp.Pool(N_WORKERS) as pool:
        for c in C_VALUES:
            for i in range(N_INST):
                planted, adj = create_planted_3col(N, c)
                A = adj.numpy()
                planted_np = planted.numpy()

                colors, success = solve_escalation(A, seed_base=i, pool=pool)
                if not success:
                    print(f"{c!s:>6}{i:>6}   FAILED to find a legal coloring within cap={CAP}")
                    continue

                colors_aligned, overlap = best_perm_align(colors, planted_np)
                n_disagree, n_oneshot, n_closure = flip_analysis(A, colors_aligned, planted_np, N)

                frac_one = n_oneshot / n_disagree if n_disagree else float("nan")
                frac_clo = n_closure / n_disagree if n_disagree else float("nan")
                print(f"{c!s:>6}{i:>6}{n_disagree:>12}{n_oneshot:>15}{n_closure:>16}"
                      f"{frac_one:>14.3f}{frac_clo:>14.3f}", flush=True)

    print("\ndone")


if __name__ == "__main__":
    mp.set_start_method("spawn", force=True)
    main()
