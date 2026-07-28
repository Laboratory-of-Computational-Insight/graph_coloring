"""
User's request (2026-07-18): retry c=4.69 at n=2000 with the FULL ensemble
(every algorithm, not just BP) and every restart count DOUBLED from what was
the standard/acceptable budget at n=1000: BP 50->100, hyperparameter-diverse
30->60. Per-instance short-circuit (cheapest/classical first) still applies.
"""
import json
import time
import torch

from random_planted import create_planted_3col
from repair_utils import count_conflicts, greedy_repair
from pairwise_nn import PairwiseEmbedder, pairwise_nn_3color
from multistage_pairwise_nn import MultiStagePairwiseEmbedder, multistage_pairwise_nn_3color
from spectral_coloring import alon_kahale_3color
from classical_baseline import dsatur_3color, peel_3color
from belief_propagation_coloring import bp_reinforced_coloring

RESULTS_DIR = "G:/graph_col/graph_coloring/results"
N = 2000
N_INST = 10
BP_RESTARTS = 100
HYPER_TRIALS = 60


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
    import numpy as np
    colors = np.asarray(colors)
    conf = count_conflicts(A, colors)
    if conf > 0:
        colors = greedy_repair(A, colors)
        conf = count_conflicts(A, colors)
    return conf == 0


def run_bp(A, n_restarts):
    for seed in range(n_restarts):
        belief, _ = bp_reinforced_coloring(A, seed=seed, max_iter=3000)
        colors = belief.argmax(axis=1)
        conf = count_conflicts(A, colors)
        if conf > 0:
            colors = greedy_repair(A, colors)
            conf = count_conflicts(A, colors)
        if conf == 0:
            return True
    return False


def run_bp_hyperparam(A, n_trials):
    rho_values = [0.001, 0.003, 0.01, 0.03, 0.1]
    damping_values = [0.3, 0.5, 0.7]
    combos = [(r, d) for r in rho_values for d in damping_values]
    for t in range(n_trials):
        rho, damping = combos[t % len(combos)]
        belief, _ = bp_reinforced_coloring(A, seed=1000 + t, rho=rho, damping=damping, max_iter=3000)
        colors = belief.argmax(axis=1)
        conf = count_conflicts(A, colors)
        if conf > 0:
            colors = greedy_repair(A, colors)
            conf = count_conflicts(A, colors)
        if conf == 0:
            return True
    return False


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

    torch.manual_seed(0)  # same seed as the original n=2000 c=4.69 test -> same 10 instances
    results = {}
    for i in range(N_INST):
        _, adj = create_planted_3col(N, 4.69)
        A = adj.numpy()
        torch.save({"assignment": None, "adj": adj},
                   f"{RESULTS_DIR}/n2000_test_instances/c4.69_inst{i}.pt")

        t0 = time.time()
        vals = {}
        solved = False
        for name, fn in [("peel", lambda: run_classical(peel_3color, adj, A)),
                          ("dsatur", lambda: run_classical(dsatur_3color, adj, A)),
                          ("alon_kahale", lambda: run_classical(alon_kahale_3color, adj, A)),
                          ("pairwise", lambda: run_classical(lambda a: pairwise_nn_3color(a, m_pw, device="cpu"), adj, A)),
                          ("multistage", lambda: run_classical(lambda a: multistage_pairwise_nn_3color(a, m_ms, device="cpu"), adj, A)),
                          (f"bp{BP_RESTARTS}", lambda: run_bp(A, BP_RESTARTS)),
                          (f"bp_hyper{HYPER_TRIALS}", lambda: run_bp_hyperparam(A, HYPER_TRIALS))]:
            if solved:
                vals[name] = None
                continue
            vals[name] = fn()
            if vals[name]:
                solved = True

        results[i] = {"success": solved, "per_method": vals}
        print(f"inst={i}: {'SUCCESS' if solved else 'FAIL'} ({time.time()-t0:.0f}s) -- {vals}")

    n_solved = sum(r["success"] for r in results.values())
    print(f"\n=== c=4.69, n=2000, FULL ensemble + doubled restarts (bp{BP_RESTARTS}, hyper{HYPER_TRIALS}): "
          f"{n_solved}/{N_INST} ({100.0*n_solved/N_INST:.0f}%) ===")

    with open(f"{RESULTS_DIR}/n2000_c469_full_ensemble_doubled.json", "w") as f:
        json.dump(results, f, indent=2)
