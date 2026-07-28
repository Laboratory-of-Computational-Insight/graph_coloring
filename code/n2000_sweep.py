"""
User's request (2026-07-18): test every c value at n=2000 (double the
project's standard n=1000), using EXISTING checkpoints/algorithms only -- no
retraining. Order: c<=3 first, then c>=7 in decreasing order, then
c=6,4,5,4.69 in that specific sequence. Uses the established winning
combination for each range (from the algorithm-per-c table), just at the
larger scale, as a generalization/scaling check.
"""
import sys
import time
import numpy as np
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


def load_nn_models():
    m_pw = PairwiseEmbedder(hidden_dim=32, embed_dim=8, rounds=32, reinject_degree_signal=True,
                             use_spectral_coords=True, num_spectral_eigvecs=4, head="distance")
    m_pw.load_state_dict(torch.load(f"{RESULTS_DIR}/pairwise_nn_medium_ablation_eig4_only.pt"))
    m_pw.eval()
    m_ms = MultiStagePairwiseEmbedder(hidden_dim=32, embed_dim=8, num_stages=3, inner_rounds=6,
                                       outer_iters=4, pos_enc_dim=16, reinject_degree_signal=True,
                                       use_spectral_coords=True, num_spectral_eigvecs=4, head="distance")
    m_ms.load_state_dict(torch.load(f"{RESULTS_DIR}/multistage_pairwise_nn_medium_n300_eig4.pt"))
    m_ms.eval()
    return m_pw, m_ms


def test_c(c, methods, m_pw=None, m_ms=None, n_inst=N_INST):
    """Cheapest/most-classical methods run first; the moment one succeeds for
    a given instance, skip the rest for that instance (per-instance
    short-circuit -- OR is already guaranteed true, no need to spend more
    compute on that instance)."""
    counts = {k: 0 for k in methods}
    not_run = {k: 0 for k in methods}
    counts["OR"] = 0
    torch.manual_seed(0)
    t0 = time.time()

    method_order = [m for m in
                    ["peel", "dsatur", "alon_kahale", "pairwise", "multistage", "bp10", "bp50", "bp_hyper30"]
                    if m in methods]

    for i in range(n_inst):
        _, adj = create_planted_3col(N, c)
        A = adj.numpy()
        vals = {}
        solved = False
        for m in method_order:
            if solved:
                not_run[m] += 1
                continue
            if m == "peel":
                vals[m] = run_classical(peel_3color, adj, A)
            elif m == "dsatur":
                vals[m] = run_classical(dsatur_3color, adj, A)
            elif m == "alon_kahale":
                vals[m] = run_classical(alon_kahale_3color, adj, A)
            elif m == "pairwise":
                vals[m] = run_classical(lambda a: pairwise_nn_3color(a, m_pw, device="cpu"), adj, A)
            elif m == "multistage":
                vals[m] = run_classical(lambda a: multistage_pairwise_nn_3color(a, m_ms, device="cpu"), adj, A)
            elif m == "bp10":
                vals[m] = run_bp(A, 10)
            elif m == "bp50":
                vals[m] = run_bp(A, 50)
            elif m == "bp_hyper30":
                vals[m] = run_bp_hyperparam(A, 30)
            if vals[m]:
                solved = True
        for k, v in vals.items():
            counts[k] += v
        counts["OR"] += any(vals.values())
    dt = time.time() - t0
    line = (f"c={c} (n={N}, {n_inst} inst, {dt:.0f}s): "
            + ", ".join(f"{k}={v}/{n_inst - not_run[k]}(skipped {not_run[k]})" for k, v in counts.items() if k != "OR")
            + f", OR={counts['OR']}/{n_inst}")
    print(line)
    return counts


if __name__ == "__main__":
    m_pw, m_ms = load_nn_models()

    print("########## PHASE 1: c<=3 (already done: 100% all methods, all c) ##########")

    print("\n########## PHASE 2: c>=7, decreasing (resuming from c=8; "
          "20/18/16/14/12 already done: 100% each; c=10 was 80% (8/10); c=9 was 100% via bp10) ##########")
    for c in [8, 7]:
        test_c(c, ["peel", "dsatur", "alon_kahale", "pairwise", "multistage", "bp10"], m_pw, m_ms)

    print("\n########## PHASE 3: c=6,4,5,4.69 ##########")
    print("--- c=6 ---")
    test_c(6, ["pairwise", "multistage", "alon_kahale", "dsatur", "bp10"], m_pw, m_ms)
    print("--- c=4 ---")
    test_c(4, ["pairwise", "multistage", "alon_kahale", "dsatur"], m_pw, m_ms)
    print("--- c=5 ---")
    test_c(5, ["pairwise", "multistage", "alon_kahale", "dsatur", "bp50"], m_pw, m_ms)
    print("--- c=4.69 ---")
    test_c(4.69, ["pairwise", "multistage", "alon_kahale", "dsatur", "bp50", "bp_hyper30"], m_pw, m_ms)

    print("\n=== n=2000 sweep complete ===")
