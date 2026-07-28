"""
Hyperparameter-diverse BP fallback for instances that exhaust the plain-
restart budget. Must be a proper importable module (not `python -c`) for
spawn-based multiprocessing to correctly pickle the worker function.
"""
import sys
import time
import multiprocessing as mp

import torch

from random_planted import create_planted_3col
from belief_propagation_coloring import bp_reinforced_coloring
from repair_utils import count_conflicts, greedy_repair

RHO_VALUES = [0.001, 0.003, 0.01, 0.03, 0.1]
DAMPING_VALUES = [0.3, 0.5, 0.7]
HYPER_COMBOS = [(r, d) for r in RHO_VALUES for d in DAMPING_VALUES]


def _try_one(args):
    A, seed = args
    rho, damping = HYPER_COMBOS[seed % len(HYPER_COMBOS)]
    belief, _ = bp_reinforced_coloring(A, seed=1000 + seed, rho=rho, damping=damping, max_iter=5000)
    colors = belief.argmax(axis=1)
    conf = count_conflicts(A, colors)
    if conf > 0:
        colors = greedy_repair(A, colors)
        conf = count_conflicts(A, colors)
    return seed, conf == 0


def run_hyper_parallel(A, n_restarts=1000, n_workers=16):
    with mp.Pool(n_workers) as pool:
        args = [(A, seed) for seed in range(n_restarts)]
        for seed, success in pool.imap_unordered(_try_one, args, chunksize=1):
            if success:
                pool.terminate()
                return True, seed + 1
    return False, n_restarts


if __name__ == "__main__":
    mp.set_start_method("spawn", force=True)
    c = float(sys.argv[1])
    n = int(sys.argv[2])
    indices = [int(x) for x in sys.argv[3:]]

    torch.manual_seed(0)
    n_gen = max(indices) + 1
    target_adj = {}
    for i in range(n_gen):
        _, adj = create_planted_3col(n, c)
        if i in indices:
            target_adj[i] = adj

    for idx in indices:
        A = target_adj[idx].numpy()
        t0 = time.time()
        ok, used = run_hyper_parallel(A)
        print(f"inst={idx}: hyperparam-diverse {'SUCCESS' if ok else 'FAIL'} (restarts={used}) time={time.time()-t0:.0f}s", flush=True)
