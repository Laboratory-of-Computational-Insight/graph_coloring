#!/usr/bin/env python3
"""
Test-time round scaling (2026-07-15) -- the D1 mechanism that gave
QueryOptGNN_MP a real, free win (rounds 32->300, avg 74.4%->83.0%, NO
retraining) applied to tonight's spectral-family architectures for the first
time. `rounds` is a plain Python attribute on SpectralEmbedder/PairwiseEmbedder/
MultiHeadGNNEmbedder (not part of the state_dict), and all three reinject their
degree signal every round -- structurally sound for the same reason D1 worked:
no absolute round-index dependency baked into the weights, so running the
SAME trained checkpoint for more rounds than it was trained with is well
defined, not extrapolation into undefined behavior.

Focused on c=8-10 (the still-weak zone) since c=11-14 is already solved and
c=5-7 is the separately-established floor. Never run at n<1000 for reporting.
"""
import argparse
import time

import torch

from random_planted import create_planted_3col
from gc_utils import is_k_color

from spectral_nn import SpectralEmbedder, spectral_nn_3color
from pairwise_nn import PairwiseEmbedder, pairwise_nn_3color
from multihead_gnn import MultiHeadGNNEmbedder, multihead_gnn_3color

RESULTS_DIR = "G:/graph_col/graph_coloring/results"

# name -> (loader, run_fn, trained_rounds)
CANDIDATES = {
    "small_speccoords": (
        lambda: (SpectralEmbedder(hidden_dim=32, embed_dim=8, rounds=64, reinject_degree_signal=True,
                                   use_spectral_coords=True),
                 "spectral_nn_small_n300_r64_fisher_reinject_speccoords.pt"),
        spectral_nn_3color, 64),
    "universal_n300_stale": (
        lambda: (SpectralEmbedder(hidden_dim=32, embed_dim=8, rounds=8, reinject_degree_signal=False),
                 "spectral_nn_universal_n300_r8_contrastive_noreinject.pt"),
        spectral_nn_3color, 8),
    "large_pairwise": (
        lambda: (PairwiseEmbedder(hidden_dim=32, embed_dim=8, rounds=32, reinject_degree_signal=True,
                                   use_spectral_coords=True, head="distance"),
                 "pairwise_nn_large_n300_r32_bce_fisher2_speccoords.pt"),
        pairwise_nn_3color, 32),
    "medium_pairwise": (
        lambda: (PairwiseEmbedder(hidden_dim=32, embed_dim=8, rounds=32, reinject_degree_signal=True,
                                   use_spectral_coords=True, head="distance"),
                 "pairwise_nn_medium_n300_r32_bce_fisher2_speccoords.pt"),
        pairwise_nn_3color, 32),
    "small_multihead": (
        lambda: (MultiHeadGNNEmbedder(head_dim=16, embed_dim=8, num_heads=4, rounds=64, reinject_degree_signal=True),
                 "multihead_gnn_small_n300_h16_nh4_r64_fisher_reinject.pt"),
        multihead_gnn_3color, 64),
}


def load(name):
    build_fn, run_fn, trained_rounds = CANDIDATES[name]
    model, fname = build_fn()
    model.load_state_dict(torch.load(f"{RESULTS_DIR}/{fname}"))
    model.eval()
    return model, run_fn, trained_rounds


def parse_args():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--n", type=int, default=1000)
    p.add_argument("--c-values", type=int, nargs="+", default=[8, 9, 10])
    p.add_argument("--n-inst", type=int, default=15)
    p.add_argument("--multipliers", type=float, nargs="+", default=[1.0, 2.0, 4.0])
    p.add_argument("--variants", nargs="+", default=list(CANDIDATES.keys()))
    return p.parse_args()


def main():
    args = parse_args()
    torch.manual_seed(0)

    loaded = {name: load(name) for name in args.variants}

    for name, (model, run_fn, trained_rounds) in loaded.items():
        print(f"\n=== {name} (trained rounds={trained_rounds}) ===")
        header = "c".rjust(4) + "".join(f"x{m:g}(r={int(trained_rounds*m)})".rjust(16) for m in args.multipliers)
        print(header)

        t0 = time.time()
        for c in args.c_values:
            counts = {m: 0 for m in args.multipliers}
            for _ in range(args.n_inst):
                _, adj = create_planted_3col(args.n, c)
                for m in args.multipliers:
                    model.rounds = max(1, int(round(trained_rounds * m)))
                    r = run_fn(adj, model, device="cpu")
                    if r["success"]:
                        ok, nconf, _ = is_k_color(adj.clone(), r["colors"].copy())
                        assert ok, f"{name} x{m} illegal at c={c}, nconf={nconf}"
                        counts[m] += 1
            model.rounds = trained_rounds  # restore
            row = f"{c:4d}" + "".join(f"{counts[m]}/{args.n_inst}".rjust(16) for m in args.multipliers)
            print(row)
        print(f"({time.time()-t0:.0f}s elapsed for {name})")


if __name__ == "__main__":
    main()
