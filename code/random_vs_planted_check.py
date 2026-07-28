"""
User's request (2026-07-18): for every c range EXCEPT c in [4,5] (already
covered by the phase-transition investigation), check whether the winning
algorithm(s) for that range also work on genuine RANDOM (non-planted)
Erdos-Renyi G(n,p) graphs at the same average degree c -- using
random_planted.py's create_gnp, not create_planted_3col. Just a report, not
a fix attempt: a random graph at c above the general 3-colorability
threshold (~4.03-4.69) is very likely not even 3-colorable at all, so
failure there is expected and not a bug in any method.

QueryOptGNN_MP isn't available in this codebase, so this only tests the
local methods (BP, spectral-NN family, classical). Representative c values
chosen per range rather than exhaustively every integer.
"""
import numpy as np
import torch

from random_planted import create_gnp
from gc_utils import is_k_color
from repair_utils import count_conflicts, greedy_repair
from pairwise_nn import PairwiseEmbedder, pairwise_nn_3color
from multistage_pairwise_nn import MultiStagePairwiseEmbedder, multistage_pairwise_nn_3color
from spectral_coloring import alon_kahale_3color
from classical_baseline import dsatur_3color, peel_3color
from belief_propagation_coloring import bp_reinforced_coloring

RESULTS_DIR = "G:/graph_col/graph_coloring/results"
N = 1000
N_INST = 10
BP_RESTARTS = 10


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


def run_bp(A, n_restarts=BP_RESTARTS):
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

    c_values = [1, 2, 3, 6, 7, 8, 9, 10, 12, 14, 16, 20]

    for c in c_values:
        counts = {"pairwise": 0, "multistage": 0, "alon_kahale": 0, "dsatur": 0, "peel": 0, "bp": 0}
        for i in range(N_INST):
            _, adj, _, _ = create_gnp(N, 3, c=float(c))
            A = adj.numpy()

            counts["pairwise"] += run_classical(lambda a: pairwise_nn_3color(a, m_pw, device="cpu"), adj, A)
            counts["multistage"] += run_classical(lambda a: multistage_pairwise_nn_3color(a, m_ms, device="cpu"), adj, A)
            counts["alon_kahale"] += run_classical(alon_kahale_3color, adj, A)
            counts["dsatur"] += run_classical(dsatur_3color, adj, A)
            counts["peel"] += run_classical(peel_3color, adj, A)
            counts["bp"] += run_bp(A)

        line = f"c={c} (random, non-planted): " + ", ".join(f"{k}={v}/{N_INST}" for k, v in counts.items())
        print(line)
