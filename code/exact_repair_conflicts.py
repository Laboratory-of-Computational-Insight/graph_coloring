"""
Exact check (2026-07-17, user's request): after greedy_repair still leaves some
monochromatic edges, does there EXIST a legal 3-coloring reachable by changing
ONLY the conflicting vertices (those touching at least one monochromatic edge),
holding every other vertex's color fixed at its repaired value? Uses OR-Tools
CP-SAT (exact, not heuristic) on just that vertex subset.

This is strictly stronger than greedy_repair: CP-SAT explores the full joint
assignment space of the conflicting vertices (with their fixed non-conflicting
neighbors as constants), not just single-vertex flips.
"""
import numpy as np
from ortools.sat.python import cp_model

from repair_utils import count_conflicts, greedy_repair


def find_conflict_vertices(A, colors):
    same = colors[:, None] == colors[None, :]
    conflict_edges = np.argwhere(np.triu(A.astype(bool) & same, k=1))
    return set(conflict_edges.flatten().tolist()), conflict_edges


def exact_fix_conflicts(A, colors, time_limit_s=30.0, workers=8):
    """Try to find a legal coloring by only changing colors of vertices that
    touch a monochromatic edge. Returns (feasible, new_colors_or_None, n_free_vars, status_name).
    status_name distinguishes proven INFEASIBLE from UNKNOWN (timeout, inconclusive)."""
    conflict_verts, conflict_edges = find_conflict_vertices(A, colors)
    if not conflict_verts:
        return True, colors.copy(), 0, "TRIVIAL"

    n = A.shape[0]
    model = cp_model.CpModel()
    var = {}
    for v in conflict_verts:
        var[v] = model.NewIntVar(0, 2, f"c{v}")

    edges = np.argwhere(np.triu(A.astype(bool), k=1))
    for u, v in edges:
        u, v = int(u), int(v)
        u_free = u in var
        v_free = v in var
        if not u_free and not v_free:
            continue  # both fixed, and if it were a conflict it'd be in conflict_verts
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
        return True, new_colors, len(conflict_verts), solver.StatusName(status)
    return False, None, len(conflict_verts), solver.StatusName(status)


if __name__ == "__main__":
    import sys
    import torch
    from gc_utils import is_k_color
    from pairwise_nn import PairwiseEmbedder, pairwise_nn_3color
    from multistage_pairwise_nn import MultiStagePairwiseEmbedder, multistage_pairwise_nn_3color

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
    time_limit = 30.0

    for c in c_list:
        n_solved_exact = 0
        n_total = 0
        free_var_counts = []
        for i in range(20):
            d = torch.load(f"{RESULTS_DIR}/shared_test_instances/c{c}_inst{i}.pt")
            adj = d["adj"]
            A = adj.numpy()

            r_pw = pairwise_nn_3color(adj, m_pw, device="cpu")
            r_ms = multistage_pairwise_nn_3color(adj, m_ms, device="cpu")
            colors_pw = greedy_repair(A, r_pw["colors"]) if not r_pw["success"] else r_pw["colors"]
            colors_ms = greedy_repair(A, r_ms["colors"]) if not r_ms["success"] else r_ms["colors"]
            conf_pw = count_conflicts(A, colors_pw)
            conf_ms = count_conflicts(A, colors_ms)
            colors, conf = (colors_pw, conf_pw) if conf_pw <= conf_ms else (colors_ms, conf_ms)

            n_total += 1
            if conf == 0:
                n_solved_exact += 1
                print(f"c={c} inst={i}: already legal after repair (skip exact solve)")
                continue

            feasible, new_colors, n_free, status_name = exact_fix_conflicts(A, colors, time_limit_s=time_limit)
            free_var_counts.append(n_free)
            if feasible:
                ok, nconf, _ = is_k_color(adj.clone(), torch.tensor(new_colors))
                assert ok and nconf == 0, "CP-SAT claimed feasible but is_k_color disagrees!"
                n_solved_exact += 1
                print(f"c={c} inst={i}: EXACT FIX FOUND (n_conflict_verts={n_free}, remaining_conflicts_before={conf}) -> legal")
            else:
                print(f"c={c} inst={i}: {status_name} (n_conflict_verts={n_free}, remaining_conflicts_before={conf}, time_limit={time_limit}s)")

        avg_free = np.mean(free_var_counts) if free_var_counts else 0
        print(f"\n=== c={c}: {n_solved_exact}/{n_total} legal after exact-fix-conflicts (repair+CP-SAT); avg free vars where CP-SAT ran = {avg_free:.1f} ===\n")
