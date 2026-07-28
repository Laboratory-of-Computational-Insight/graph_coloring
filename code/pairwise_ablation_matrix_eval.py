#!/usr/bin/env python3
"""
Evaluation for train_pairwise_ablation_matrix.py's checkpoints, targeted at
c=7,8,9 (the thin-signal zone). Compares every ablation config against the
original small_pairwise/medium_pairwise checkpoints and plain adjacency
spectral. Cross-checks every reported success against gc_utils.is_k_color.
"""
import argparse
import time

import torch

from random_planted import create_planted_3col
from pairwise_nn import PairwiseEmbedder, pairwise_nn_3color
from spectral_coloring import alon_kahale_3color
from gc_utils import is_k_color
from train_pairwise_ablation_matrix import CONFIGS, BANDS

RESULTS_DIR = "G:/graph_col/graph_coloring/results"

MANIFEST = {}
for cfg_name, kwargs in CONFIGS.items():
    for band_name in BANDS:
        full_kwargs = dict(hidden_dim=32, embed_dim=8, rounds=32, reinject_degree_signal=True,
                            head="distance")
        full_kwargs.update(kwargs)
        MANIFEST[f"{band_name}_{cfg_name}"] = (
            f"pairwise_nn_{band_name}_ablation_{cfg_name}.pt", full_kwargs)

# Originals for comparison
MANIFEST["small_pairwise"] = ("pairwise_nn_small_n300_r32_bce_fisher2_speccoords.pt",
    dict(hidden_dim=32, embed_dim=8, rounds=32, reinject_degree_signal=True,
         use_spectral_coords=True, head="distance"))
MANIFEST["medium_pairwise"] = ("pairwise_nn_medium_n300_r32_bce_fisher2_speccoords.pt",
    dict(hidden_dim=32, embed_dim=8, rounds=32, reinject_degree_signal=True,
         use_spectral_coords=True, head="distance"))


def load_model(name):
    fname, kwargs = MANIFEST[name]
    m = PairwiseEmbedder(**kwargs)
    m.load_state_dict(torch.load(f"{RESULTS_DIR}/{fname}"))
    m.eval()
    return m


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
        header += name.rjust(max(22, len(name) + 2))
    print(header)

    t0 = time.time()
    for c in args.c_values:
        counts = {name: 0 for name in models}
        plain_count = 0
        for _ in range(args.n_inst):
            _, adj = create_planted_3col(args.n, c)
            r_plain = alon_kahale_3color(adj, variant="adjacency")
            plain_count += r_plain["success"]
            for name, m in models.items():
                r = pairwise_nn_3color(adj, m, device="cpu")
                if r["success"]:
                    ok, nconf, _ = is_k_color(adj.clone(), r["colors"].copy())
                    assert ok, f"{name} illegal at c={c}, nconf={nconf}"
                    counts[name] += 1

        row = f"{c:4d}" + f"{plain_count}/{args.n_inst}".rjust(8)
        for name in models:
            row += f"{counts[name]}/{args.n_inst}".rjust(max(22, len(name) + 2))
        print(row)

    print(f"\n({time.time()-t0:.0f}s elapsed)")


if __name__ == "__main__":
    main()
