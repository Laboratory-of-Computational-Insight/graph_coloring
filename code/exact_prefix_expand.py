"""
Follow-up to exact_repair_conflicts.py (2026-07-17): freeing ONLY the conflict
vertices was proven INFEASIBLE (exact, CP-SAT) for every c=5/c=6 shared
instance. Next question: how far out do we need to expand the free set before
a legal coloring becomes reachable? Grow the free set in BFS hops outward from
the conflict-vertex core (a "prefix" of the expansion order) and re-run CP-SAT
at each radius until SAT or the free set gets too large to be a meaningful fix.
"""
import sys
import numpy as np
import torch
from ortools.sat.python import cp_model

from gc_utils import is_k_color
from repair_utils import count_conflicts, greedy_repair
from pairwise_nn import PairwiseEmbedder, pairwise_nn_3color
from multistage_pairwise_nn import MultiStagePairwiseEmbedder, multistage_pairwise_nn_3color
from exact_repair_conflicts import find_conflict_vertices


def bfs_expand(A, seed_verts, n_hops):
    n = A.shape[0]
    frontier = set(seed_verts)
    visited = set(seed_verts)
    for _ in range(n_hops):
        new_frontier = set()
        for v in frontier:
            neighbors = np.where(A[v] == 1)[0]
            for u in neighbors:
                if u not in visited:
                    new_frontier.add(int(u))
        visited |= new_frontier
        frontier = new_frontier
        if not frontier:
            break
    return visited


def solve_with_free_set(A, colors, free_verts, time_limit_s=30.0, workers=8):
    model = cp_model.CpModel()
    var = {v: model.NewIntVar(0, 2, f"c{v}") for v in free_verts}
    edges = np.argwhere(np.triu(A.astype(bool), k=1))
    for u, v in edges:
        u, v = int(u), int(v)
        u_free, v_free = u in var, v in var
        if not u_free and not v_free:
            continue
        if u_free and v_free:
            model.Add(var[u] != var[v])
        elif u_free:
            model.Add(var[u] != int(colors[v]))
        else:
            model.Add(var[v] != int(colors[u]))
    solver = cp_model.CpSolver()
    solver.parameters.max_time_in_seconds = time_limit_s
    solver.parameters.num_search_workers = workers
    status = solver.Solve(model)
    if status in (cp_model.OPTIMAL, cp_model.FEASIBLE):
        new_colors = colors.copy()
        for v, x in var.items():
            new_colors[v] = solver.Value(x)
        return "SAT", new_colors
    elif status == cp_model.INFEASIBLE:
        return "INFEASIBLE", None
    else:
        return "UNKNOWN(timeout)", None


if __name__ == "__main__":
    RESULTS_DIR = "G:/graph_col/graph_coloring/results"
    m_pw = PairwiseEmbedder(hidden_dim=32, embed_dim=8, rounds=32, reinject_degree_signal=True,
                             use_spectral_coords=True, num_spectral_eigvecs=4, head="distance")
    m_pw.load_state_dict(torch.load(f"{RESULTS_DIR}/pairwise_nn_medium_ablation_eig4_only.pt"))
    m_pw.eval()
    m_ms = MultiStagePairwiseEmbedder(hidden_dim=32, embed_dim=8, num_stages=3, inner_rounds=6,
                                       outer_iters=4, pos_enc_dim=16, reinject_degree_signal=True,
                                       use_spectral_coords=True, num_spectral_eigvecs=4, head="distance")
    m_ms.load_state_dict(torch.load(f"{RESULTS_DIR}/multistage_pairwise_nn_medium_n300_eig4.pt"))
    m_ms.eval()

    c_list = [int(x) for x in sys.argv[1:]] if len(sys.argv) > 1 else [5, 6]
    max_hops = 3
    per_solve_time = 20.0
    n_inst_test = 5  # keep it cheap first; expand once we see the pattern

    for c in c_list:
        print(f"\n########## c={c} ##########")
        for i in range(n_inst_test):
            d = torch.load(f"{RESULTS_DIR}/shared_test_instances/c{c}_inst{i}.pt")
            adj = d["adj"]
            A = adj.numpy()
            r_pw = pairwise_nn_3color(adj, m_pw, device="cpu")
            r_ms = multistage_pairwise_nn_3color(adj, m_ms, device="cpu")
            c_pw = greedy_repair(A, r_pw["colors"]) if not r_pw["success"] else r_pw["colors"]
            c_ms = greedy_repair(A, r_ms["colors"]) if not r_ms["success"] else r_ms["colors"]
            conf_pw, conf_ms = count_conflicts(A, c_pw), count_conflicts(A, c_ms)
            colors, conf = (c_pw, conf_pw) if conf_pw <= conf_ms else (c_ms, conf_ms)
            if conf == 0:
                print(f"inst={i}: already legal after repair, skip")
                continue

            conflict_verts, _ = find_conflict_vertices(A, colors)
            n = A.shape[0]
            found = False
            for hop in range(0, max_hops + 1):
                free_verts = bfs_expand(A, conflict_verts, hop) if hop > 0 else set(conflict_verts)
                status, new_colors = solve_with_free_set(A, colors, free_verts, time_limit_s=per_solve_time)
                pct = 100.0 * len(free_verts) / n
                print(f"inst={i} hop={hop}: |free|={len(free_verts)} ({pct:.1f}% of n) -> {status}")
                if status == "SAT":
                    ok, nconf, _ = is_k_color(adj.clone(), torch.tensor(new_colors))
                    assert ok and nconf == 0
                    found = True
                    break
                if status == "UNKNOWN(timeout)":
                    print(f"inst={i}: hit timeout at hop={hop}, stopping expansion (too expensive to keep growing)")
                    break
            if not found:
                print(f"inst={i}: NOT solved within hop<= {max_hops} (or hit timeout)")
