"""
Parallelized (multiprocessing, one restart per core) BP restart search, for
running on elad-laca (16 cores) to speed up the doubled-restart budget test
for c=4.69 at n=2000 that's slow on the single-threaded Windows machine.
Same algorithm as bp_reinforced_coloring + greedy_repair, just fanned out
across a process pool instead of run sequentially.
"""
import sys
import time
import json
import multiprocessing as mp

import numpy as np
import torch

from random_planted import create_planted_3col
from belief_propagation_coloring import bp_reinforced_coloring
from repair_utils import count_conflicts, greedy_repair

RESULTS_DIR = "/home/elad/Documents/kcol/graph_coloring/results"
N = 2000
N_INST = 10
BP_RESTARTS = 100
HYPER_TRIALS = 60

RHO_VALUES = [0.001, 0.003, 0.01, 0.03, 0.1]
DAMPING_VALUES = [0.3, 0.5, 0.7]
HYPER_COMBOS = [(r, d) for r in RHO_VALUES for d in DAMPING_VALUES]


def _try_one(args):
    A, seed, rho, damping = args
    belief, _ = bp_reinforced_coloring(A, seed=seed, rho=rho, damping=damping, max_iter=3000)
    colors = belief.argmax(axis=1)
    conf = count_conflicts(A, colors)
    if conf > 0:
        colors = greedy_repair(A, colors)
        conf = count_conflicts(A, colors)
    return seed, rho, damping, conf == 0


def parallel_search(A, configs, n_workers=16):
    with mp.Pool(n_workers) as pool:
        args = [(A, *cfg) for cfg in configs]
        for seed, rho, damping, success in pool.imap_unordered(_try_one, args, chunksize=1):
            if success:
                pool.terminate()
                return True, (seed, rho, damping)
    return False, None


if __name__ == "__main__":
    mp.set_start_method("spawn", force=True)
    torch.manual_seed(0)  # same seed as the original n=2000 c=4.69 test -> same 10 instances

    results = {}
    for i in range(N_INST):
        _, adj = create_planted_3col(N, 4.69)
        A = adj.numpy()

        t0 = time.time()
        plain_configs = [(seed, 0.003, 0.5) for seed in range(BP_RESTARTS)]
        success, winner = parallel_search(A, plain_configs, n_workers=16)

        if not success:
            hyper_configs = [(1000 + t, HYPER_COMBOS[t % len(HYPER_COMBOS)][0], HYPER_COMBOS[t % len(HYPER_COMBOS)][1])
                              for t in range(HYPER_TRIALS)]
            success, winner = parallel_search(A, hyper_configs, n_workers=16)

        dt = time.time() - t0
        results[i] = {"success": success, "winner": winner, "time_s": dt}
        print(f"inst={i}: {'SUCCESS' if success else 'FAIL'} winner={winner} ({dt:.0f}s)", flush=True)

    n_solved = sum(r["success"] for r in results.values())
    print(f"\n=== c=4.69, n=2000, parallel BP100+hyper60 (16 cores): {n_solved}/{N_INST} ({100.0*n_solved/N_INST:.0f}%) ===")

    with open(f"{RESULTS_DIR}/n2000_c469_parallel_remote.json", "w") as f:
        json.dump(results, f, indent=2)
