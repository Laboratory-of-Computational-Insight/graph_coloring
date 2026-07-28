"""
User's request (2026-07-17): run BP (with reinforcement, the standard trick to
make it work in the hard/clustered phase) on c=6 planted-coloring instances.
Rank vertices by final belief confidence (max_s belief_i(s)) -- the "most sure
nodes". For a given k, fix the top-k most-confident vertices to BP's
argmax-predicted color and ask OR-Tools CP-SAT whether a legal 3-coloring of
the WHOLE graph exists extending that partial fixing (the other n-k vertices
free). Feasibility is monotonically non-increasing in k (fixing more vertices
only shrinks the feasible region), so binary search finds the largest feasible
k exactly. Report mean/std of that k across instances.
"""
import sys
import numpy as np
import torch
from ortools.sat.python import cp_model

from belief_propagation_coloring import bp_best_of_restarts
from gc_utils import is_k_color


def largest_feasible_k(A, order, pred_colors, time_limit_s=15.0, workers=8):
    n = A.shape[0]
    edges = np.argwhere(np.triu(A.astype(bool), k=1))

    def feasible(k):
        if k == 0:
            return True  # graph is planted 3-colorable by construction, no need to verify via CP-SAT
        fixed_set = set(order[:k].tolist())
        model = cp_model.CpModel()
        var = {}
        for v in range(n):
            if v not in fixed_set:
                var[v] = model.NewIntVar(0, 2, f"c{v}")
        for u, v in edges:
            u, v = int(u), int(v)
            u_free, v_free = u in var, v in var
            if u_free and v_free:
                model.Add(var[u] != var[v])
            elif u_free:
                model.Add(var[u] != int(pred_colors[v]))
            elif v_free:
                model.Add(var[v] != int(pred_colors[u]))
            else:
                if pred_colors[u] == pred_colors[v]:
                    return False  # two fixed neighbors share a color -- immediately infeasible
        for v, x in var.items():
            model.AddHint(x, int(pred_colors[v]))
        solver = cp_model.CpSolver()
        solver.parameters.max_time_in_seconds = time_limit_s
        solver.parameters.num_search_workers = workers
        status = solver.Solve(model)
        return status in (cp_model.OPTIMAL, cp_model.FEASIBLE)

    lo, hi = 0, n
    while lo < hi:
        mid = (lo + hi + 1) // 2
        if feasible(mid):
            lo = mid
        else:
            hi = mid - 1
    return lo


if __name__ == "__main__":
    RESULTS_DIR = "G:/graph_col/graph_coloring/results"
    c_list = [int(x) for x in sys.argv[1:]] if len(sys.argv) > 1 else [6]
    n_inst = 20

    for c in c_list:
        ks = []
        print(f"\n########## c={c} ##########")
        for i in range(n_inst):
            d = torch.load(f"{RESULTS_DIR}/shared_test_instances/c{c}_inst{i}.pt")
            adj = d["adj"]
            A = adj.numpy()
            n = A.shape[0]

            belief = bp_best_of_restarts(A, n_restarts=3, max_iter=3000)
            confidence = belief.max(axis=1)
            pred_colors = belief.argmax(axis=1)
            order = np.argsort(-confidence)  # most confident first

            k_star = largest_feasible_k(A, order, pred_colors)
            ks.append(k_star)
            print(f"inst={i}: k*={k_star} ({100.0*k_star/n:.1f}% of n={n}), mean_maxbelief={confidence.mean():.3f}")

        ks = np.array(ks)
        print(f"\n=== c={c}: k* mean={ks.mean():.1f} std={ks.std():.1f} (n={n}, {n_inst} instances) ===\n")
