#!/usr/bin/env python3
"""
Evaluation matrix for pairwise_nn.py checkpoints, matching the style of
spectral_nn_matrix_eval.py / multistage_matrix_eval.py: rows = c values,
columns = plain adjacency spectral (reference) + one column per
band-specialized PairwiseEmbedder checkpoint. Cross-checks every reported
"success" independently against gc_utils.is_k_color.

Never run at n < 1000 for reporting (small n blurs the phase transition).
"""
import argparse
import time

import torch

from random_planted import create_planted_3col
from pairwise_nn import PairwiseEmbedder, pairwise_nn_3color
from spectral_coloring import alon_kahale_3color
from gc_utils import is_k_color

RESULTS_DIR = "G:/graph_col/graph_coloring/results"

# name -> (checkpoint filename, PairwiseEmbedder kwargs)
MANIFEST = {
    "small_pairwise":  ("pairwise_nn_small_n300_r32_bce_fisher2_speccoords.pt",
                     dict(hidden_dim=32, embed_dim=8, rounds=32, reinject_degree_signal=True,
                          use_spectral_coords=True, head="distance")),
    "medium_pairwise": ("pairwise_nn_medium_n300_r32_bce_fisher2_speccoords.pt",
                     dict(hidden_dim=32, embed_dim=8, rounds=32, reinject_degree_signal=True,
                          use_spectral_coords=True, head="distance")),
    "large_pairwise":  ("pairwise_nn_large_n300_r32_bce_fisher2_speccoords.pt",
                     dict(hidden_dim=32, embed_dim=8, rounds=32, reinject_degree_signal=True,
                          use_spectral_coords=True, head="distance")),
    "small_pairwise_dualres":  ("pairwise_nn_small_n300_r32_bce_fisher2_speccoords_dualcoords_residualcoords.pt",
                     dict(hidden_dim=32, embed_dim=8, rounds=32, reinject_degree_signal=True,
                          use_spectral_coords=False, use_dual_spectral_coords=True,
                          residual_spectral_coords=True, head="distance")),
    "medium_pairwise_dualres": ("pairwise_nn_medium_n300_r32_bce_fisher2_speccoords_dualcoords_residualcoords.pt",
                     dict(hidden_dim=32, embed_dim=8, rounds=32, reinject_degree_signal=True,
                          use_spectral_coords=False, use_dual_spectral_coords=True,
                          residual_spectral_coords=True, head="distance")),
    "small_pairwise_dualres_h128":  ("pairwise_nn_small_n300_r32_bce_fisher2_speccoords_dualcoords_residualcoords_h128.pt",
                     dict(hidden_dim=128, embed_dim=8, rounds=32, reinject_degree_signal=True,
                          use_spectral_coords=False, use_dual_spectral_coords=True,
                          residual_spectral_coords=True, head="distance")),
    "medium_pairwise_dualres_h128": ("pairwise_nn_medium_n300_r32_bce_fisher2_speccoords_dualcoords_residualcoords_h128.pt",
                     dict(hidden_dim=128, embed_dim=8, rounds=32, reinject_degree_signal=True,
                          use_spectral_coords=False, use_dual_spectral_coords=True,
                          residual_spectral_coords=True, head="distance")),
}


def load_model(name):
    fname, kwargs = MANIFEST[name]
    m = PairwiseEmbedder(**kwargs)
    m.load_state_dict(torch.load(f"{RESULTS_DIR}/{fname}"))
    m.eval()
    return m


def parse_args():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--n", type=int, default=1000)
    p.add_argument("--c-values", type=int, nargs="+", default=[5, 6, 7, 8, 9, 10, 11, 12, 13, 14])
    p.add_argument("--n-inst", type=int, default=10, help="Short/fast by default -- bump for a deeper look")
    p.add_argument("--variants", nargs="+", default=None,
                    help="Subset of MANIFEST keys to run (default: all available checkpoints)")
    p.add_argument("--skip-plain", action="store_true", help="Skip the plain-adjacency-spectral reference column")
    p.add_argument("--ensemble", action="store_true",
                    help="Add an 'ANY_pairwise' column: OR across just the pairwise checkpoints listed "
                         "(not plain) -- answers 'does the pairwise family alone cover this c'.")
    p.add_argument("--seed", type=int, default=0)
    return p.parse_args()


def main():
    args = parse_args()
    if args.n < 1000:
        print(f"WARNING: n={args.n} < 1000 -- results at this n are NOT valid for reporting "
              f"(finite-size effects blur the phase transition). Proceeding anyway since requested.")

    torch.manual_seed(args.seed)

    variant_names = args.variants or list(MANIFEST.keys())
    models = {}
    for name in variant_names:
        try:
            models[name] = load_model(name)
        except FileNotFoundError:
            print(f"  (skipping {name}: checkpoint not found yet)")

    header = "c".rjust(4) + ("plain".rjust(8) if not args.skip_plain else "")
    for name in models:
        header += name.rjust(max(12, len(name) + 2))
    if args.ensemble:
        header += "ANY_pairwise".rjust(14)
    print(header)

    t0 = time.time()
    for c in args.c_values:
        row_counts = {name: 0 for name in models}
        plain_count = 0
        any_count = 0
        for _ in range(args.n_inst):
            _, adj = create_planted_3col(args.n, c)
            instance_any = False
            if not args.skip_plain:
                r_plain = alon_kahale_3color(adj, variant="adjacency")
                if r_plain["success"]:
                    ok, nconf, _ = is_k_color(adj.clone(), r_plain["colors"].copy())
                    assert ok, f"plain spectral produced an illegal coloring at c={c}, nconf={nconf}"
                plain_count += r_plain["success"]
            for name, m in models.items():
                r = pairwise_nn_3color(adj, m, device="cpu")
                if r["success"]:
                    ok, nconf, _ = is_k_color(adj.clone(), r["colors"].copy())
                    assert ok, f"{name} produced an illegal coloring at c={c}, nconf={nconf}"
                    row_counts[name] += 1
                    instance_any = True
            if instance_any:
                any_count += 1

        row = f"{c:4d}"
        if not args.skip_plain:
            row += f"{plain_count}/{args.n_inst}".rjust(8)
        for name in models:
            row += f"{row_counts[name]}/{args.n_inst}".rjust(max(12, len(name) + 2))
        if args.ensemble:
            row += f"{any_count}/{args.n_inst}".rjust(14)
        print(row)

    print(f"\n({time.time()-t0:.0f}s elapsed)")


if __name__ == "__main__":
    main()
