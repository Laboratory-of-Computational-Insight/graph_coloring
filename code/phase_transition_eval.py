"""
Test the full ensemble at c=4.69, the theoretical phase-transition point for
this project's planted 3-coloring distribution (already cited throughout
EXPERIMENTS.md as the classical-baseline dip point). Fresh instances (n=1000,
20 of them), not the pre-saved shared set (which only covers integer c=4-9).
No OR-Tools/CP-SAT -- diagnostic-only, never counted.
"""
import json
import numpy as np
import torch

from random_planted import create_planted_3col
from gc_utils import is_k_color
from repair_utils import count_conflicts, greedy_repair
from pairwise_nn import PairwiseEmbedder, pairwise_nn_3color
from multistage_pairwise_nn import MultiStagePairwiseEmbedder, multistage_pairwise_nn_3color
from spectral_coloring import alon_kahale_3color
from classical_baseline import dsatur_3color, peel_3color
from belief_propagation_coloring import bp_reinforced_coloring

RESULTS_DIR = "G:/graph_col/graph_coloring/results"
C = 4.69
N = 1000
N_INST = 20
BP_RESTARTS = 50


def run_classical(fn, adj, A):
    result = fn(adj)
    if isinstance(result, dict):
        colors = result["colors"]
    elif isinstance(result, tuple):
        _, colors = result
    else:
        colors = result
    if colors is None:
        return False
    if torch.is_tensor(colors):
        colors = colors.numpy()
    colors = np.asarray(colors)
    conf = count_conflicts(A, colors)
    if conf > 0:
        colors = greedy_repair(A, colors)
        conf = count_conflicts(A, colors)
    return conf == 0


def run_bp_multi_restart(A, n_restarts=BP_RESTARTS):
    for seed in range(n_restarts):
        belief, _ = bp_reinforced_coloring(A, seed=seed, max_iter=3000)
        colors = belief.argmax(axis=1)
        conf = count_conflicts(A, colors)
        if conf > 0:
            colors = greedy_repair(A, colors)
            conf = count_conflicts(A, colors)
        if conf == 0:
            return True, seed + 1
    return False, n_restarts


if __name__ == "__main__":
    m_pw = PairwiseEmbedder(hidden_dim=32, embed_dim=8, rounds=32, reinject_degree_signal=True,
                             use_spectral_coords=True, num_spectral_eigvecs=4, head="distance")
    m_pw.load_state_dict(torch.load(f"{RESULTS_DIR}/pairwise_nn_medium_ablation_eig4_only.pt"))
    m_pw.eval()
    m_ms = MultiStagePairwiseEmbedder(hidden_dim=32, embed_dim=8, num_stages=3, inner_rounds=6,
                                       outer_iters=4, pos_enc_dim=16, reinject_degree_signal=True,
                                       use_spectral_coords=True, num_spectral_eigvecs=4, head="distance")
    m_ms.load_state_dict(torch.load(f"{RESULTS_DIR}/multistage_pairwise_nn_medium_n300_eig4.pt"))
    m_ms.eval()

    torch.manual_seed(0)
    per_inst = {}
    counts = {"pairwise": 0, "multistage": 0, "alon_kahale": 0, "dsatur": 0, "peel": 0, "bp": 0, "FULL_OR": 0}
    bp_restarts_used = []

    for i in range(N_INST):
        _, adj = create_planted_3col(N, C)
        A = adj.numpy()

        r_pw = run_classical(lambda a: pairwise_nn_3color(a, m_pw, device="cpu"), adj, A)
        r_ms = run_classical(lambda a: multistage_pairwise_nn_3color(a, m_ms, device="cpu"), adj, A)
        r_ak = run_classical(alon_kahale_3color, adj, A)
        r_ds = run_classical(dsatur_3color, adj, A)
        r_pl = run_classical(peel_3color, adj, A)
        r_bp, n_used = run_bp_multi_restart(A)
        bp_restarts_used.append(n_used)

        vals = {"pairwise": r_pw, "multistage": r_ms, "alon_kahale": r_ak,
                "dsatur": r_ds, "peel": r_pl, "bp": r_bp}
        for k, v in vals.items():
            counts[k] += v
        full_or = any(vals.values())
        counts["FULL_OR"] += full_or
        per_inst[str(i)] = {**vals, "bp_restarts_used": n_used, "FULL_OR": full_or}

        print(f"inst={i}: pairwise={r_pw} multistage={r_ms} alon_kahale={r_ak} dsatur={r_ds} "
              f"peel={r_pl} bp={r_bp}(restarts={n_used}) -> OR={full_or}")

    print(f"\n=== c={C}, n={N}, {N_INST} fresh instances ===")
    for k in ["pairwise", "multistage", "alon_kahale", "dsatur", "peel", "bp"]:
        print(f"{k}: {counts[k]}/{N_INST} ({100.0*counts[k]/N_INST:.1f}%)")
    print(f"FULL OR: {counts['FULL_OR']}/{N_INST} ({100.0*counts['FULL_OR']/N_INST:.1f}%)")
    print(f"BP restarts used (mean/max): {np.mean(bp_restarts_used):.1f} / {max(bp_restarts_used)}")

    with open(f"{RESULTS_DIR}/phase_transition_c4.69_results.json", "w") as f:
        json.dump({"c": C, "n": N, "per_instance": per_inst, "summary": counts}, f, indent=2)
