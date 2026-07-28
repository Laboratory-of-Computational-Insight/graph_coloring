"""
For each neural model family (at c=10, n=1000), find a successful instance,
extract the raw embedding (the model's forward() output, [n, embed_dim] --
the "last layer" before any spectral/clustering readout), run PCA on it, and
report the explained variance of the first 4 principal components. Also
plots the first 2 PCs, colored by the final coloring assigned to each node.

Restart-vs-embedding note: only the `_cleanup` fallback (DSATUR) is
unseeded/random; the embedder's forward() itself is deterministic in eval()
mode, so the embedding used for PCA does not depend on which retry
succeeded -- only the coloring (used for the color-coding) does.
"""
import json

import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from random_planted import create_planted_3col
from gc_utils import is_k_color
from repair_utils import count_conflicts, greedy_repair

from spectral_nn import SpectralEmbedder, spectral_nn_3color
from pairwise_nn import PairwiseEmbedder, pairwise_nn_3color
from multistage_nn import MultiStageEmbedder, multistage_nn_3color
from multihead_gnn import MultiHeadGNNEmbedder, multihead_gnn_3color
from multistage_pairwise_nn import MultiStagePairwiseEmbedder, multistage_pairwise_nn_3color
from multihead_pairwise_nn import MultiHeadPairwiseEmbedder, multihead_pairwise_nn_3color

RESULTS_DIR = "G:/graph_col/graph_coloring/results"
N = 1000
C = 10
MAX_TRIES = 60  # retries to land a success (cleanup fallback is unseeded)

# 3-color categorical palette (dataviz skill reference palette, slots 1/3/6)
COLOR_HEX = ["#2a78d6", "#eda100", "#e34948"]

MANIFEST = {
    "spectral_nn": ("spectral_nn_medium_n300_r64_fisher_reinject_speccoords.pt",
        SpectralEmbedder, spectral_nn_3color,
        dict(hidden_dim=32, embed_dim=8, rounds=64, reinject_degree_signal=True, use_spectral_coords=True)),
    "pairwise_nn": ("pairwise_nn_medium_ablation_eig4_only.pt", PairwiseEmbedder, pairwise_nn_3color,
        dict(hidden_dim=32, embed_dim=8, rounds=32, reinject_degree_signal=True,
             use_spectral_coords=True, num_spectral_eigvecs=4, head="distance")),
    "multistage_nn": ("multistage_nn_medium_n300_s3_i6_o4_pos16_fisher_reinject.pt",
        MultiStageEmbedder, multistage_nn_3color,
        dict(hidden_dim=32, embed_dim=8, num_stages=3, inner_rounds=6, outer_iters=4,
             pos_enc_dim=16, reinject_degree_signal=True)),
    "multihead_gnn": ("multihead_gnn_medium_n300_h16_nh4_r64_fisher_reinject.pt",
        MultiHeadGNNEmbedder, multihead_gnn_3color,
        dict(head_dim=16, embed_dim=8, num_heads=4, rounds=64, reinject_degree_signal=True)),
    "multistage_pairwise_nn": ("multistage_pairwise_nn_medium_n300_eig4.pt",
        MultiStagePairwiseEmbedder, multistage_pairwise_nn_3color,
        dict(hidden_dim=32, embed_dim=8, num_stages=3, inner_rounds=6, outer_iters=4, pos_enc_dim=16,
             reinject_degree_signal=True, use_spectral_coords=True, num_spectral_eigvecs=4, head="distance")),
    "multihead_pairwise_nn": ("multihead_pairwise_nn_medium_n300_eig4.pt",
        MultiHeadPairwiseEmbedder, multihead_pairwise_nn_3color,
        dict(head_dim=16, embed_dim=8, num_heads=4, rounds=32, reinject_degree_signal=True,
             use_spectral_coords=True, num_spectral_eigvecs=4, head="distance")),
}

# first-success instance index per model, from neural_all_models_sweep.json at c=10
FIRST_SUCCESS_INST = {
    "spectral_nn": 2,
    "pairwise_nn": 0,
    "multistage_nn": 0,
    "multihead_gnn": 0,
    "multistage_pairwise_nn": 0,
    "multihead_pairwise_nn": 0,
}


PRECEDING_C_VALUES = [5, 6, 7, 8, 9]  # order + counts must match neural_all_models_sweep.py's
N_INST_PER_C = 20                      # RNG consumption, so instance indices line up


def get_instance(idx):
    """Replays the exact RNG sequence neural_all_models_sweep.py used (single
    manual_seed(0) at the top, c-values processed in order 5,6,7,8,9,10,...)
    so that "instance idx at c=10" is the identical graph that produced the
    recorded success/failure in neural_all_models_sweep.json."""
    torch.manual_seed(0)
    for c in PRECEDING_C_VALUES:
        for _ in range(N_INST_PER_C):
            create_planted_3col(N, c)
    adj = None
    for i in range(idx + 1):
        _, adj = create_planted_3col(N, C)
    return adj


def pca_explained_variance(X, n_components=4):
    Xc = X - X.mean(axis=0, keepdims=True)
    U, S, Vt = np.linalg.svd(Xc, full_matrices=False)
    var = (S ** 2) / (X.shape[0] - 1)
    total_var = var.sum()
    explained_ratio = var / total_var
    proj = Xc @ Vt[:n_components].T
    return explained_ratio[:n_components], proj


def main():
    results = {}
    fig, axes = plt.subplots(2, 3, figsize=(16, 10))
    axes = axes.flatten()

    for ax_idx, (name, (fname, cls, run_fn, kwargs)) in enumerate(MANIFEST.items()):
        m = cls(**kwargs)
        m.load_state_dict(torch.load(f"{RESULTS_DIR}/{fname}", map_location="cpu"))
        m.eval()

        adj = get_instance(FIRST_SUCCESS_INST[name])
        A = adj.numpy()

        success = False
        colors = None
        for attempt in range(1, MAX_TRIES + 1):
            r = run_fn(adj, m, device="cpu")
            c_ = r["colors"]
            conf = count_conflicts(A, c_)
            if conf > 0:
                c_ = greedy_repair(A, c_)
                conf = count_conflicts(A, c_)
            if conf == 0:
                success = True
                colors = c_
                break
        if not success:
            print(f"{name}: FAILED to find a successful run in {MAX_TRIES} tries -- skipping")
            continue
        ok, nconf, _ = is_k_color(adj.clone(), colors.copy())
        assert ok, f"{name} illegal, nconf={nconf}"

        with torch.no_grad():
            adj_t = adj if isinstance(adj, torch.Tensor) else torch.as_tensor(adj, dtype=torch.float32)
            embed = m(adj_t.float()).numpy()

        explained, proj2 = pca_explained_variance(embed, n_components=4)
        results[name] = {
            "attempt_used": attempt,
            "explained_variance_ratio_pc1to4": explained.tolist(),
        }
        print(f"{name}: succeeded on attempt {attempt}/{MAX_TRIES}, "
              f"explained variance (PC1-4) = {[f'{v:.3f}' for v in explained]}", flush=True)

        ax = axes[ax_idx]
        for k in range(3):
            mask = colors == k
            ax.scatter(proj2[mask, 0], proj2[mask, 1], s=8, alpha=0.7,
                       color=COLOR_HEX[k], label=f"color {k}")
        ax.set_title(f"{name}\nEV: PC1={explained[0]:.2f} PC2={explained[1]:.2f} "
                      f"PC3={explained[2]:.2f} PC4={explained[3]:.2f}", fontsize=10)
        ax.set_xlabel("PC1")
        ax.set_ylabel("PC2")
        ax.legend(fontsize=7, loc="best")

    plt.tight_layout()
    out_png = f"{RESULTS_DIR}/embedding_pca_c10.png"
    plt.savefig(out_png, dpi=140)
    print(f"\nsaved plot to {out_png}")

    with open(f"{RESULTS_DIR}/embedding_pca_c10.json", "w") as f:
        json.dump(results, f, indent=2)
    print(f"saved explained-variance data to {RESULTS_DIR}/embedding_pca_c10.json")


if __name__ == "__main__":
    main()
