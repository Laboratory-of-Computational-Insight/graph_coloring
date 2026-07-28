"""
Idea #1 (2026-07-18): does c=4.69 respond to more BP restarts the way c=5 did
(10->50 restarts: 75%->95%)? Reuses the saved c=4.69 shared instances and
plain BP+reinforcement, just with a much higher restart budget.
"""
import sys
import json
import numpy as np
import torch

from belief_propagation_coloring import bp_reinforced_coloring
from repair_utils import count_conflicts, greedy_repair

RESULTS_DIR = "G:/graph_col/graph_coloring/results"

if __name__ == "__main__":
    n_restarts = int(sys.argv[1]) if len(sys.argv) > 1 else 200
    n_inst = 20

    per_inst = {}
    n_solved = 0
    restarts_used_list = []
    for i in range(n_inst):
        d = torch.load(f"{RESULTS_DIR}/shared_test_instances/c4.69_inst{i}.pt")
        adj = d["adj"]
        A = adj.numpy()

        success = False
        n_used = n_restarts
        for seed in range(n_restarts):
            belief, _ = bp_reinforced_coloring(A, seed=seed, max_iter=3000)
            colors = belief.argmax(axis=1)
            conf = count_conflicts(A, colors)
            if conf > 0:
                colors = greedy_repair(A, colors)
                conf = count_conflicts(A, colors)
            if conf == 0:
                success = True
                n_used = seed + 1
                break

        per_inst[str(i)] = {"success": success, "restarts_used": n_used}
        restarts_used_list.append(n_used)
        n_solved += success
        print(f"inst={i}: {'SUCCESS' if success else 'fail'} (restarts_used={n_used}/{n_restarts})")

    print(f"\n=== c=4.69, BP x{n_restarts} restarts + repair, OR-combined: {n_solved}/{n_inst} ({100.0*n_solved/n_inst:.1f}%) ===")
    print(f"restarts used (mean/max): {np.mean(restarts_used_list):.1f} / {max(restarts_used_list)}")

    with open(f"{RESULTS_DIR}/phase_transition_bp_{n_restarts}restarts.json", "w") as f:
        json.dump(per_inst, f, indent=2)
