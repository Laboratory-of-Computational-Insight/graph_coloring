#!/usr/bin/env python3
"""
Evaluation for the two new backbone x pairwise-head combinations
(multistage_pairwise_nn, multihead_pairwise_nn) and the multihead+speccoords
PCA-readout retrain, compared against plain spectral, the original
single-cell pairwise checkpoints, the confirmed-winning medium_eig4_only
(single-cell + 4 eigvecs), and the original PCA multistage/multihead
checkpoints. Targeted at c=7,8,9. Cross-checks every success against
gc_utils.is_k_color.
"""
import argparse
import time

import torch

from random_planted import create_planted_3col
from spectral_coloring import alon_kahale_3color
from gc_utils import is_k_color

from pairwise_nn import PairwiseEmbedder, pairwise_nn_3color
from multistage_pairwise_nn import MultiStagePairwiseEmbedder, multistage_pairwise_nn_3color
from multihead_pairwise_nn import MultiHeadPairwiseEmbedder, multihead_pairwise_nn_3color
from multistage_nn import MultiStageEmbedder, multistage_nn_3color
from multihead_gnn import MultiHeadGNNEmbedder, multihead_gnn_3color
from gps_pairwise_nn import GPSPairwiseEmbedder, gps_pairwise_nn_3color

RESULTS_DIR = "G:/graph_col/graph_coloring/results"

MANIFEST = {
    # new combos
    "small_multistage_pairwise":  ("multistage_pairwise_nn_small_n300_eig4.pt",
        MultiStagePairwiseEmbedder, multistage_pairwise_nn_3color,
        dict(hidden_dim=32, embed_dim=8, num_stages=3, inner_rounds=6, outer_iters=4, pos_enc_dim=16,
             reinject_degree_signal=True, use_spectral_coords=True, num_spectral_eigvecs=4, head="distance")),
    "medium_multistage_pairwise": ("multistage_pairwise_nn_medium_n300_eig4.pt",
        MultiStagePairwiseEmbedder, multistage_pairwise_nn_3color,
        dict(hidden_dim=32, embed_dim=8, num_stages=3, inner_rounds=6, outer_iters=4, pos_enc_dim=16,
             reinject_degree_signal=True, use_spectral_coords=True, num_spectral_eigvecs=4, head="distance")),
    "small_multihead_pairwise":  ("multihead_pairwise_nn_small_n300_eig4.pt",
        MultiHeadPairwiseEmbedder, multihead_pairwise_nn_3color,
        dict(head_dim=16, embed_dim=8, num_heads=4, rounds=32, reinject_degree_signal=True,
             use_spectral_coords=True, num_spectral_eigvecs=4, head="distance")),
    "medium_multihead_pairwise": ("multihead_pairwise_nn_medium_n300_eig4.pt",
        MultiHeadPairwiseEmbedder, multihead_pairwise_nn_3color,
        dict(head_dim=16, embed_dim=8, num_heads=4, rounds=32, reinject_degree_signal=True,
             use_spectral_coords=True, num_spectral_eigvecs=4, head="distance")),
    "small_multihead_speccoords":  ("multihead_gnn_small_n300_h16_nh4_r64_fisher_reinject_speccoords.pt",
        MultiHeadGNNEmbedder, multihead_gnn_3color,
        dict(head_dim=16, embed_dim=8, num_heads=4, rounds=64, reinject_degree_signal=True, use_spectral_coords=True)),
    "medium_multihead_speccoords": ("multihead_gnn_medium_n300_h16_nh4_r64_fisher_reinject_speccoords.pt",
        MultiHeadGNNEmbedder, multihead_gnn_3color,
        dict(head_dim=16, embed_dim=8, num_heads=4, rounds=64, reinject_degree_signal=True, use_spectral_coords=True)),
    "small_gps_pairwise": ("gps_pairwise_nn_small_n300_eig4.pt", GPSPairwiseEmbedder, gps_pairwise_nn_3color,
        dict(hidden_dim=32, embed_dim=8, num_layers=4, heads=4, use_spectral_coords=True,
             num_spectral_eigvecs=4, head="distance")),
    "medium_gps_pairwise": ("gps_pairwise_nn_medium_n300_eig4.pt", GPSPairwiseEmbedder, gps_pairwise_nn_3color,
        dict(hidden_dim=32, embed_dim=8, num_layers=4, heads=4, use_spectral_coords=True,
             num_spectral_eigvecs=4, head="distance")),
    # references
    "medium_eig4_only": ("pairwise_nn_medium_ablation_eig4_only.pt", PairwiseEmbedder, pairwise_nn_3color,
        dict(hidden_dim=32, embed_dim=8, rounds=32, reinject_degree_signal=True,
             use_spectral_coords=True, num_spectral_eigvecs=4, head="distance")),
    "small_pairwise": ("pairwise_nn_small_n300_r32_bce_fisher2_speccoords.pt", PairwiseEmbedder, pairwise_nn_3color,
        dict(hidden_dim=32, embed_dim=8, rounds=32, reinject_degree_signal=True,
             use_spectral_coords=True, head="distance")),
    "medium_pairwise": ("pairwise_nn_medium_n300_r32_bce_fisher2_speccoords.pt", PairwiseEmbedder, pairwise_nn_3color,
        dict(hidden_dim=32, embed_dim=8, rounds=32, reinject_degree_signal=True,
             use_spectral_coords=True, head="distance")),
    "small_multistage": ("multistage_nn_small_n300_s3_i6_o4_pos16_fisher_reinject.pt",
        MultiStageEmbedder, multistage_nn_3color,
        dict(hidden_dim=32, embed_dim=8, num_stages=3, inner_rounds=6, outer_iters=4,
             pos_enc_dim=16, reinject_degree_signal=True)),
    "medium_multistage": ("multistage_nn_medium_n300_s3_i6_o4_pos16_fisher_reinject.pt",
        MultiStageEmbedder, multistage_nn_3color,
        dict(hidden_dim=32, embed_dim=8, num_stages=3, inner_rounds=6, outer_iters=4,
             pos_enc_dim=16, reinject_degree_signal=True)),
    "small_multihead": ("multihead_gnn_small_n300_h16_nh4_r64_fisher_reinject.pt",
        MultiHeadGNNEmbedder, multihead_gnn_3color,
        dict(head_dim=16, embed_dim=8, num_heads=4, rounds=64, reinject_degree_signal=True)),
    "medium_multihead": ("multihead_gnn_medium_n300_h16_nh4_r64_fisher_reinject.pt",
        MultiHeadGNNEmbedder, multihead_gnn_3color,
        dict(head_dim=16, embed_dim=8, num_heads=4, rounds=64, reinject_degree_signal=True)),
}


def load_model(name):
    fname, cls, run_fn, kwargs = MANIFEST[name]
    m = cls(**kwargs)
    m.load_state_dict(torch.load(f"{RESULTS_DIR}/{fname}"))
    m.eval()
    return m, run_fn


def parse_args():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--n", type=int, default=1000)
    p.add_argument("--c-values", type=int, nargs="+", default=[7, 8, 9])
    p.add_argument("--n-inst", type=int, default=20)
    p.add_argument("--variants", nargs="+", default=None)
    return p.parse_args()


def main():
    args = parse_args()
    torch.manual_seed(0)

    variant_names = args.variants or list(MANIFEST.keys())
    models = {}
    for name in variant_names:
        try:
            models[name] = load_model(name)
        except FileNotFoundError:
            print(f"  (skipping {name}: checkpoint not found yet)")

    header = "c".rjust(4) + "plain".rjust(8)
    for name in models:
        header += name.rjust(max(24, len(name) + 2))
    print(header)

    t0 = time.time()
    for c in args.c_values:
        counts = {name: 0 for name in models}
        plain_count = 0
        for _ in range(args.n_inst):
            _, adj = create_planted_3col(args.n, c)
            r_plain = alon_kahale_3color(adj, variant="adjacency")
            plain_count += r_plain["success"]
            for name, (m, run_fn) in models.items():
                r = run_fn(adj, m, device="cpu")
                if r["success"]:
                    ok, nconf, _ = is_k_color(adj.clone(), r["colors"].copy())
                    assert ok, f"{name} illegal at c={c}, nconf={nconf}"
                    counts[name] += 1

        row = f"{c:4d}" + f"{plain_count}/{args.n_inst}".rjust(8)
        for name in models:
            row += f"{counts[name]}/{args.n_inst}".rjust(max(24, len(name) + 2))
        print(row)

    print(f"\n({time.time()-t0:.0f}s elapsed)")


if __name__ == "__main__":
    main()
