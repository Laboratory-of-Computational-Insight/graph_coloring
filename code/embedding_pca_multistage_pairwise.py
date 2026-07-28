"""
Multiple-instance PCA check for multistage_pairwise_nn specifically (c=10,
n=1000) -- same embedding-before-readout PCA as embedding_pca_analysis.py,
but across several successful instances of ONE model, to see whether the
genuine 3-way separation seen on instance 0 is consistent or a fluke.
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
from multistage_pairwise_nn import MultiStagePairwiseEmbedder, multistage_pairwise_nn_3color

RESULTS_DIR = "G:/graph_col/graph_coloring/results"
N = 1000
C = 10
MAX_TRIES = 30
COLOR_HEX = ["#2a78d6", "#eda100", "#e34948"]

# instances confirmed successful for multistage_pairwise_nn at c=10 in
# neural_all_models_sweep.json (instance 0 already shown earlier)
INSTANCES = [7, 8, 10, 11, 12, 13]

PRECEDING_C_VALUES = [5, 6, 7, 8, 9]
N_INST_PER_C = 20


def get_instance(idx):
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
    explained_ratio = var / var.sum()
    proj = Xc @ Vt[:n_components].T
    return explained_ratio[:n_components], proj


def main():
    kwargs = dict(hidden_dim=32, embed_dim=8, num_stages=3, inner_rounds=6, outer_iters=4, pos_enc_dim=16,
                  reinject_degree_signal=True, use_spectral_coords=True, num_spectral_eigvecs=4, head="distance")
    m = MultiStagePairwiseEmbedder(**kwargs)
    m.load_state_dict(torch.load(f"{RESULTS_DIR}/multistage_pairwise_nn_medium_n300_eig4.pt", map_location="cpu"))
    m.eval()

    n_panels = len(INSTANCES)
    ncols = 3
    nrows = (n_panels + ncols - 1) // ncols
    fig, axes = plt.subplots(nrows, ncols, figsize=(16, 5 * nrows))
    axes = np.array(axes).flatten()

    results = {}
    for ax_idx, inst_idx in enumerate(INSTANCES):
        adj = get_instance(inst_idx)
        A = adj.numpy()

        success = False
        colors = None
        for attempt in range(1, MAX_TRIES + 1):
            r = multistage_pairwise_nn_3color(adj, m, device="cpu")
            c_ = r["colors"]
            if count_conflicts(A, c_) > 0:
                c_ = greedy_repair(A, c_)
            if count_conflicts(A, c_) == 0:
                success, colors = True, c_
                break
        ax = axes[ax_idx]
        if not success:
            print(f"instance {inst_idx}: FAILED in {MAX_TRIES} tries -- skipping")
            ax.set_title(f"instance {inst_idx}: no success in {MAX_TRIES} tries")
            continue

        ok, nconf, _ = is_k_color(adj.clone(), colors.copy())
        assert ok, f"illegal at instance {inst_idx}, nconf={nconf}"

        with torch.no_grad():
            embed = m(adj.float()).numpy()
        explained, proj2 = pca_explained_variance(embed, n_components=4)
        results[inst_idx] = {"attempt_used": attempt, "explained_variance_ratio_pc1to4": explained.tolist()}
        print(f"instance {inst_idx}: succeeded on attempt {attempt}/{MAX_TRIES}, "
              f"EV(PC1-4)={[f'{v:.3f}' for v in explained]}", flush=True)

        for k in range(3):
            mask = colors == k
            ax.scatter(proj2[mask, 0], proj2[mask, 1], s=8, alpha=0.7, color=COLOR_HEX[k], label=f"color {k}")
        ax.set_title(f"instance {inst_idx}\nEV: PC1={explained[0]:.2f} PC2={explained[1]:.2f} "
                     f"PC3={explained[2]:.2f} PC4={explained[3]:.2f}", fontsize=10)
        ax.set_xlabel("PC1")
        ax.set_ylabel("PC2")
        ax.legend(fontsize=7, loc="best")

    for j in range(n_panels, len(axes)):
        axes[j].axis("off")

    plt.tight_layout()
    out_png = f"{RESULTS_DIR}/embedding_pca_multistage_pairwise_c10_more2.png"
    plt.savefig(out_png, dpi=140)
    print(f"\nsaved plot to {out_png}")

    with open(f"{RESULTS_DIR}/embedding_pca_multistage_pairwise_c10_more2.json", "w") as f:
        json.dump(results, f, indent=2)


if __name__ == "__main__":
    main()
