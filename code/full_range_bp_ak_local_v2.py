"""
v2: parallelized (multiprocessing across local cores) version of
full_range_bp_ak_local.py -- the sequential version left a single hard
instance able to cost up to ~3.5 hours (1000 restarts x ~12.6s each) with no
interim visibility. This version fans restarts out across all local cores
(matching the approach already working well on elad-laca) and prints
per-instance progress.
"""
import sys
import time
import json
import multiprocessing as mp

import numpy as np
import torch

from random_planted import create_planted_3col
from spectral_coloring import alon_kahale_3color
from belief_propagation_coloring import bp_reinforced_coloring
from repair_utils import count_conflicts, greedy_repair

RESULTS_DIR = "G:/graph_col/graph_coloring/results"
N = 2000
N_INST = 20
BP_RESTARTS = 1000
BP_MAX_ITER = 5000
N_WORKERS = 16  # leave headroom rather than using all 32


def run_alon_kahale(adj, A):
    result = alon_kahale_3color(adj)
    colors = result["colors"]
    conf = count_conflicts(A, colors)
    if conf > 0:
        colors = greedy_repair(A, colors)
        conf = count_conflicts(A, colors)
    return conf == 0


def _try_one(args):
    A, seed = args
    belief, _ = bp_reinforced_coloring(A, seed=seed, max_iter=BP_MAX_ITER)
    colors = belief.argmax(axis=1)
    conf = count_conflicts(A, colors)
    if conf > 0:
        colors = greedy_repair(A, colors)
        conf = count_conflicts(A, colors)
    return seed, conf == 0


def run_bp_parallel(A, n_restarts=BP_RESTARTS, n_workers=N_WORKERS):
    with mp.Pool(n_workers) as pool:
        args = [(A, seed) for seed in range(n_restarts)]
        for seed, success in pool.imap_unordered(_try_one, args, chunksize=1):
            if success:
                pool.terminate()
                return True, seed + 1
    return False, n_restarts


if __name__ == "__main__":
    mp.set_start_method("spawn", force=True)
    c_list = [float(x) for x in sys.argv[1:]] if len(sys.argv) > 1 else [5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15]
    all_results = {}

    for c in c_list:
        c_disp = int(c) if c == int(c) else c
        torch.manual_seed(0)
        n_solved = 0
        restarts_used = []
        per_inst = {}
        t0 = time.time()
        for i in range(N_INST):
            _, adj = create_planted_3col(N, c)
            A = adj.numpy()

            ak_ok = run_alon_kahale(adj, A)
            if ak_ok:
                per_inst[i] = {"method": "alon_kahale", "restarts": 0}
                n_solved += 1
                print(f"  c={c_disp} inst={i}: alon_kahale SUCCESS", flush=True)
                continue

            bp_ok, used = run_bp_parallel(A)
            restarts_used.append(used)
            per_inst[i] = {"method": "bp" if bp_ok else "none", "restarts": used}
            n_solved += bp_ok
            print(f"  c={c_disp} inst={i}: bp {'SUCCESS' if bp_ok else 'FAIL'} (restarts={used})", flush=True)

        dt = time.time() - t0
        mean_r = np.mean(restarts_used) if restarts_used else 0
        max_r = max(restarts_used) if restarts_used else 0
        print(f"[LOCAL] c={c_disp}, n={N}: {n_solved}/{N_INST} ({100.0*n_solved/N_INST:.0f}%) "
              f"mean_bp_restarts={mean_r:.1f} max={max_r} time={dt:.0f}s", flush=True)
        all_results[str(c_disp)] = per_inst

    with open(f"{RESULTS_DIR}/full_range_bp_ak_local_v2.json", "w") as f:
        json.dump(all_results, f, indent=2)
    print("=== LOCAL PORTION COMPLETE ===")
