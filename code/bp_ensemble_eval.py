"""
BP (belief propagation + reinforcement) as a standalone algorithm, + the
standard greedy_repair post-processing (per the project's universal
post-processing rule) -- NO OR-Tools/CP-SAT anywhere in this file, since
CP-SAT is diagnostic-only and must never contribute to reported accuracy.
Reports BP+repair's own success rate on the shared c=4,5,6 test instances,
for folding into the real cross-architecture ensemble.
"""
import sys
import json
import numpy as np
import torch

from belief_propagation_coloring import bp_best_of_restarts
from repair_utils import count_conflicts, greedy_repair
from gc_utils import is_k_color


if __name__ == "__main__":
    RESULTS_DIR = "G:/graph_col/graph_coloring/results"
    c_list = [int(x) for x in sys.argv[1:]] if len(sys.argv) > 1 else [4, 5, 6]
    n_inst = 20

    all_results = {}
    for c in c_list:
        per_inst = {}
        n_solved = 0
        for i in range(n_inst):
            d = torch.load(f"{RESULTS_DIR}/shared_test_instances/c{c}_inst{i}.pt")
            adj = d["adj"]
            A = adj.numpy()

            belief = bp_best_of_restarts(A, n_restarts=3, max_iter=3000)
            colors = belief.argmax(axis=1)
            conf = count_conflicts(A, colors)
            if conf > 0:
                colors = greedy_repair(A, colors)
                conf = count_conflicts(A, colors)

            ok, nconf, _ = is_k_color(adj.clone(), torch.tensor(colors))
            assert (nconf == 0) == (conf == 0)
            success = conf == 0
            per_inst[str(i)] = bool(success)
            n_solved += success
            print(f"c={c} inst={i}: {'SUCCESS' if success else f'fail (conflicts={conf})'}")

        all_results[str(c)] = per_inst
        print(f"=== c={c}: BP+repair {n_solved}/{n_inst} ({100.0*n_solved/n_inst:.1f}%) ===\n")

    with open(f"{RESULTS_DIR}/shared_test_bp_repair_results.json", "w") as f:
        json.dump(all_results, f, indent=2)
