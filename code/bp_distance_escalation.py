"""
Same distance-from-planted measurement as bp_distance_from_planted.py, but
with the full 3-rung escalation ladder and a much bigger combined budget,
parallelized across cores (sequential was far too slow for a 100-attempt
cap x 16 c-values x 10 instances):
  1. plain BP (rho=0)
  2. BP+reinforcement (rho=0.003, default damping=0.5)
  3. hyperparameter-diverse BP (rho in {0.001,0.003,0.01,0.03,0.1} x
     damping in {0.3,0.5,0.7}, cycled)
OR-combined, first zero-conflict result wins, total attempts capped at
--cap (default 100) across all three rungs combined. BP itself still runs
max_iter=3000 internally (unchanged from the standard config).
"""
import argparse
import itertools
import json
import multiprocessing as mp

import numpy as np
import torch

from random_planted import create_planted_3col
from belief_propagation_coloring import bp_reinforced_coloring
from repair_utils import count_conflicts

N = 1000
RESULTS_DIR = "G:/graph_col/graph_coloring/results"
N_WORKERS = 16
PERMS = list(itertools.permutations(range(3)))

RHO_VALUES = [0.001, 0.003, 0.01, 0.03, 0.1]
DAMPING_VALUES = [0.3, 0.5, 0.7]
HYPER_COMBOS = [(r, d) for r in RHO_VALUES for d in DAMPING_VALUES]


def distance_from_planted(colors, planted):
    colors = np.asarray(colors)
    planted = np.asarray(planted)
    best_match = 0
    for perm in PERMS:
        remapped = np.array(perm)[colors]
        match = (remapped == planted).sum()
        best_match = max(best_match, match)
    return 1.0 - best_match / len(planted)


def _run_one(args):
    """One BP attempt. rho/damping fixed per call (worker picks them from
    the task tuple so plain/reinforced/hyperdiverse all share this one
    picklable top-level function)."""
    A, rho, damping, seed = args
    belief, _ = bp_reinforced_coloring(A, rho=rho, damping=damping, max_iter=3000, seed=seed)
    colors = belief.argmax(axis=1)
    conf = count_conflicts(A, colors)
    return seed, colors, conf


def run_rung_parallel(A, tasks, pool):
    """tasks: list of (rho, damping, seed). Returns (colors, conf, n_tried,
    success) -- n_tried = attempts DISPATCHED before (and including) the
    first success found via imap_unordered, or len(tasks) if none succeeded."""
    args = [(A, rho, damping, seed) for (rho, damping, seed) in tasks]
    best_colors, best_conf = None, None
    n_seen = 0
    for seed, colors, conf in pool.imap_unordered(_run_one, args, chunksize=1):
        n_seen += 1
        if best_conf is None or conf < best_conf:
            best_colors, best_conf = colors, conf
        if conf == 0:
            return colors, 0, n_seen, True
    return best_colors, best_conf, n_seen, False


def solve_escalation(A, seed_base, cap_total, pool):
    per_rung = cap_total // 3
    attempts_so_far = 0

    # rung 1: plain
    tasks = [(0.0, 0.5, seed_base * 100000 + i) for i in range(per_rung)]
    colors, conf, n_tried, success = run_rung_parallel(A, tasks, pool)
    attempts_so_far += n_tried
    if success:
        return colors, "plain", attempts_so_far, True
    best_colors, best_conf, best_method = colors, conf, "plain"

    # rung 2: reinforced
    tasks = [(0.003, 0.5, seed_base * 100000 + per_rung + i) for i in range(per_rung)]
    colors, conf, n_tried, success = run_rung_parallel(A, tasks, pool)
    attempts_so_far += n_tried
    if success:
        return colors, "reinforced", attempts_so_far, True
    if conf is not None and conf < best_conf:
        best_colors, best_conf, best_method = colors, conf, "reinforced"

    # rung 3: hyperparameter-diverse (takes remaining budget)
    remaining = cap_total - attempts_so_far
    tasks = [(HYPER_COMBOS[i % len(HYPER_COMBOS)][0], HYPER_COMBOS[i % len(HYPER_COMBOS)][1],
              seed_base * 100000 + 2 * per_rung + i) for i in range(remaining)]
    colors, conf, n_tried, success = run_rung_parallel(A, tasks, pool)
    attempts_so_far += n_tried
    if success:
        return colors, "hyperdiverse", attempts_so_far, True
    if conf is not None and conf < best_conf:
        best_colors, best_conf, best_method = colors, conf, "hyperdiverse"

    return best_colors, best_method, attempts_so_far, False


def parse_args():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--c-values", type=float, nargs="+",
                    default=[4.40, 4.42, 4.44, 4.46, 4.48, 4.50, 4.52, 4.54,
                             4.56, 4.58, 4.60, 4.62, 4.64, 4.66, 4.68, 4.69])
    p.add_argument("--n-inst", type=int, default=10)
    p.add_argument("--cap", type=int, default=100)
    p.add_argument("--n-workers", type=int, default=N_WORKERS)
    p.add_argument("--out", default=f"{RESULTS_DIR}/bp_distance_escalation.json")
    return p.parse_args()


def main():
    args = parse_args()
    torch.manual_seed(0)
    results = {}
    print(f"{'c':>6}{'n_success':>11}{'mean_retries':>14}{'max_retries':>13}"
          f"{'mean_dist(succ)':>18}{'method(plain/reinf/hyper)':>28}")

    with mp.Pool(args.n_workers) as pool:
        for c in args.c_values:
            c_disp = round(c, 4)
            dists_success = []
            retries_success = []
            methods_used = {"plain": 0, "reinforced": 0, "hyperdiverse": 0}
            n_success = 0
            per_inst = []

            for i in range(args.n_inst):
                planted, adj = create_planted_3col(N, c)
                A = adj.numpy()
                planted_np = planted.numpy()

                colors, method, attempts_used, success = solve_escalation(A, seed_base=i, cap_total=args.cap, pool=pool)
                dist = distance_from_planted(colors, planted_np)

                if success:
                    dists_success.append(dist)
                    retries_success.append(attempts_used)
                    methods_used[method] += 1
                    n_success += 1

                per_inst.append({"success": success, "method": method, "retries": attempts_used, "distance": dist})
                print(f"    c={c_disp} inst={i}: {'SUCCESS' if success else 'FAIL'} "
                      f"method={method} retries={attempts_used} dist={dist:.4f}", flush=True)

            mean_r = np.mean(retries_success) if retries_success else float("nan")
            max_r = max(retries_success) if retries_success else 0
            mean_d = np.mean(dists_success) if dists_success else float("nan")
            method_str = f"{methods_used['plain']}/{methods_used['reinforced']}/{methods_used['hyperdiverse']}"
            print(f"{c_disp!s:>6}{f'{n_success}/{args.n_inst}':>11}{mean_r:>14.2f}{max_r:>13}"
                  f"{mean_d:>18.4f}{method_str:>28}", flush=True)

            results[str(c_disp)] = {
                "n_success": n_success,
                "n_inst": args.n_inst,
                "mean_retries_success": None if not retries_success else float(mean_r),
                "max_retries_success": max_r,
                "mean_distance_success": None if not dists_success else float(mean_d),
                "methods_used": methods_used,
                "per_instance": per_inst,
            }
            with open(args.out, "w") as f:
                json.dump(results, f, indent=2)

    print(f"\nsaved to {args.out}")


if __name__ == "__main__":
    mp.set_start_method("spawn", force=True)
    main()
