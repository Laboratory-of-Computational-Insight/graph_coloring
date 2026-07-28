"""
Exact check on the RAW (pre-repair) model output (2026-07-17, user clarification):
before greedy_repair runs at all, take the raw NN coloring, find its conflicting
vertices (touching >=1 monochromatic edge), and ask via OR-Tools CP-SAT whether
a legal coloring exists by recoloring ONLY those vertices, holding everything
else fixed at the raw NN's choice. Compares directly against
exact_repair_conflicts.py's post-repair version to see whether greedy_repair
helps or hurts the exact-solvability of the remaining conflict region.
"""
import sys
import numpy as np
import torch
from ortools.sat.python import cp_model

from gc_utils import is_k_color
from repair_utils import count_conflicts
from pairwise_nn import PairwiseEmbedder, pairwise_nn_3color
from multistage_pairwise_nn import MultiStagePairwiseEmbedder, multistage_pairwise_nn_3color
from exact_repair_conflicts import find_conflict_vertices, exact_fix_conflicts


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
    time_limit = 30.0

    for c in c_list:
        print(f"\n########## c={c} (RAW, pre-repair) ##########")
        n_solved = 0
        n_total = 0
        for i in range(20):
            d = torch.load(f"{RESULTS_DIR}/shared_test_instances/c{c}_inst{i}.pt")
            adj = d["adj"]
            A = adj.numpy()

            r_pw = pairwise_nn_3color(adj, m_pw, device="cpu")
            r_ms = multistage_pairwise_nn_3color(adj, m_ms, device="cpu")
            conf_pw = count_conflicts(A, r_pw["colors"])
            conf_ms = count_conflicts(A, r_ms["colors"])
            colors, conf, src = (r_pw["colors"], conf_pw, "pairwise") if conf_pw <= conf_ms else (r_ms["colors"], conf_ms, "multistage")

            n_total += 1
            if conf == 0:
                n_solved += 1
                print(f"inst={i}: raw {src} already legal (skip)")
                continue

            feasible, new_colors, n_free, status_name = exact_fix_conflicts(A, colors, time_limit_s=time_limit)
            if feasible:
                ok, nconf, _ = is_k_color(adj.clone(), torch.tensor(new_colors))
                assert ok and nconf == 0
                n_solved += 1
                print(f"inst={i}: RAW {src} exact-fix SAT (n_conflict_verts={n_free}, raw_conflicts={conf}) -> legal")
            else:
                print(f"inst={i}: RAW {src} exact-fix status={status_name} (n_conflict_verts={n_free}, raw_conflicts={conf})")

        print(f"\n=== c={c} RAW: {n_solved}/{n_total} legal via exact-fix-on-raw-conflicts ===\n")
