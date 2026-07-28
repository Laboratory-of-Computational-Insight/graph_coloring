#!/usr/bin/env python3
"""
Fast, all-variants evaluation matrix for spectral_nn.py checkpoints.

Every time a new mechanism/flag is tried, add its checkpoint + exact config
here (never guessed from the filename -- a manifest avoids ambiguity) and
re-run. Defaults to a short instance count for quick screening; bump
--n-inst for a deeper look at whichever variant looks promising.
"""
import argparse
import time

import torch

from random_planted import create_planted_3col
from spectral_nn import SpectralEmbedder, spectral_nn_3color
from spectral_coloring import alon_kahale_3color
from gc_utils import is_k_color

RESULTS_DIR = "G:/graph_col/graph_coloring/results"

# name -> (checkpoint filename, SpectralEmbedder kwargs)
MANIFEST = {
    "small_n300":  ("spectral_nn_small_n300_r64_fisher_reinject.pt",
                     dict(hidden_dim=32, embed_dim=8, rounds=64, reinject_degree_signal=True)),
    "medium_n300": ("spectral_nn_medium_n300_r64_fisher_reinject.pt",
                     dict(hidden_dim=32, embed_dim=8, rounds=64, reinject_degree_signal=True)),
    "large_n300":  ("spectral_nn_large_n300_r64_fisher_reinject.pt",
                     dict(hidden_dim=32, embed_dim=8, rounds=64, reinject_degree_signal=True)),
    "universal_n300_stale": ("spectral_nn_universal_n300_r8_contrastive_noreinject.pt",
                     dict(hidden_dim=32, embed_dim=8, rounds=8, reinject_degree_signal=False)),
    "small_n1000":  ("spectral_nn_small_n1000.pt",
                     dict(hidden_dim=32, embed_dim=8, rounds=64, reinject_degree_signal=True)),
    "medium_n1000": ("spectral_nn_medium_n1000.pt",
                     dict(hidden_dim=32, embed_dim=8, rounds=64, reinject_degree_signal=True)),
    "small_speccoords":  ("spectral_nn_small_n300_r64_fisher_reinject_speccoords.pt",
                     dict(hidden_dim=32, embed_dim=8, rounds=64, reinject_degree_signal=True, use_spectral_coords=True)),
    "medium_speccoords": ("spectral_nn_medium_n300_r64_fisher_reinject_speccoords.pt",
                     dict(hidden_dim=32, embed_dim=8, rounds=64, reinject_degree_signal=True, use_spectral_coords=True)),
    "large_speccoords":  ("spectral_nn_large_n300_r64_fisher_reinject_speccoords.pt",
                     dict(hidden_dim=32, embed_dim=8, rounds=64, reinject_degree_signal=True, use_spectral_coords=True)),
}


def load_model(name):
    fname, kwargs = MANIFEST[name]
    m = SpectralEmbedder(**kwargs)
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
                    help="Add an 'ANY' column: succeeds if plain OR any listed variant succeeds on that instance "
                         "(cheap combine-techniques check using already-trained checkpoints, no new training)")
    return p.parse_args()


def main():
    args = parse_args()
    variant_names = args.variants or list(MANIFEST.keys())

    models = {}
    for name in variant_names:
        try:
            models[name] = load_model(name)
        except FileNotFoundError:
            print(f"  (skipping {name}: checkpoint not found yet)")

    header = "c".rjust(4) + ("plain".rjust(8) if not args.skip_plain else "")
    for name in models:
        header += name.rjust(max(10, len(name) + 2))
    if args.ensemble:
        header += "ANY".rjust(10)
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
                plain_count += r_plain["success"]
                instance_any = instance_any or bool(r_plain["success"])
            for name, m in models.items():
                r = spectral_nn_3color(adj, m, device="cpu")
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
            row += f"{row_counts[name]}/{args.n_inst}".rjust(max(10, len(name) + 2))
        if args.ensemble:
            row += f"{any_count}/{args.n_inst}".rjust(10)
        print(row)

    print(f"\n({time.time()-t0:.0f}s elapsed)")


if __name__ == "__main__":
    main()
