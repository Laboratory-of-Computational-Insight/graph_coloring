"""
SP with independent-restart OR-combination, mirroring bp_multi_restart_eval.py
exactly for a fair, apples-to-apples comparison against BP. No OR-Tools/CP-SAT
-- diagnostic-only, never counted.
"""
import sys
import json
import numpy as np
import torch

from survey_propagation_coloring import sp_reinforced_coloring
from repair_utils import count_conflicts, greedy_repair
from gc_utils import is_k_color


if __name__ == "__main__":
    RESULTS_DIR = "G:/graph_col/graph_coloring/results"
    c_list = [int(x) for x in sys.argv[1:-1]] if len(sys.argv) > 2 else [5]
    n_restarts = int(sys.argv[-1]) if len(sys.argv) > 1 else 10
    n_inst = 20

    all_results = {}
    for c in c_list:
        per_inst = {}
        n_solved = 0
        for i in range(n_inst):
            d = torch.load(f"{RESULTS_DIR}/shared_test_instances/c{c}_inst{i}.pt")
            adj = d["adj"]
            A = adj.numpy()

            restart_outcomes = []
            for seed in range(n_restarts):
                belief, _ = sp_reinforced_coloring(A, seed=seed, rho=0.003)
                colors = belief.argmax(axis=1)
                conf = count_conflicts(A, colors)
                if conf > 0:
                    colors = greedy_repair(A, colors)
                    conf = count_conflicts(A, colors)
                restart_outcomes.append(conf == 0)
                if conf == 0:
                    break

            success = any(restart_outcomes)
            per_inst[str(i)] = {"success": bool(success), "n_restarts_used": len(restart_outcomes)}
            n_solved += success
            print(f"c={c} inst={i}: {'SUCCESS' if success else 'fail'} (restarts_used={len(restart_outcomes)}/{n_restarts})")

        all_results[str(c)] = per_inst
        print(f"=== c={c}: SP x{n_restarts} restarts + repair, OR-combined: {n_solved}/{n_inst} ({100.0*n_solved/n_inst:.1f}%) ===\n")

    with open(f"{RESULTS_DIR}/shared_test_sp_multi_restart_results.json", "w") as f:
        json.dump(all_results, f, indent=2)
