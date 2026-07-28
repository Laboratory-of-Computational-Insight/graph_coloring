#!/usr/bin/env python3
"""
Cross-family ensemble check (2026-07-15): OR-combine every trained checkpoint
across ALL THREE architectures tried tonight -- SpectralEmbedder (PCA readout),
PairwiseEmbedder (same/different classifier + assortative spectral readout),
MultiStageEmbedder (nested-loop PCA readout) -- plus plain adjacency spectral,
on the SAME instances. No new training: this only tests whether combining
already-trained models (cheapest possible "combine techniques" move) closes
more of the gap than any single family's own ensemble did.

Never run at n < 1000 for reporting.
"""
import argparse
import time

import torch

from random_planted import create_planted_3col
from spectral_coloring import alon_kahale_3color
from gc_utils import is_k_color

from spectral_nn import SpectralEmbedder, spectral_nn_3color
from pairwise_nn import PairwiseEmbedder, pairwise_nn_3color
from multistage_nn import MultiStageEmbedder, multistage_nn_3color
from multihead_gnn import MultiHeadGNNEmbedder, multihead_gnn_3color

RESULTS_DIR = "G:/graph_col/graph_coloring/results"

# (family_label, loader_fn, checkpoint, kwargs)
SPECTRAL_MANIFEST = {
    "small_speccoords":  ("spectral_nn_small_n300_r64_fisher_reinject_speccoords.pt",
                     dict(hidden_dim=32, embed_dim=8, rounds=64, reinject_degree_signal=True, use_spectral_coords=True)),
    "medium_speccoords": ("spectral_nn_medium_n300_r64_fisher_reinject_speccoords.pt",
                     dict(hidden_dim=32, embed_dim=8, rounds=64, reinject_degree_signal=True, use_spectral_coords=True)),
    "large_speccoords":  ("spectral_nn_large_n300_r64_fisher_reinject_speccoords.pt",
                     dict(hidden_dim=32, embed_dim=8, rounds=64, reinject_degree_signal=True, use_spectral_coords=True)),
    "universal_n300_stale": ("spectral_nn_universal_n300_r8_contrastive_noreinject.pt",
                     dict(hidden_dim=32, embed_dim=8, rounds=8, reinject_degree_signal=False)),
}

PAIRWISE_MANIFEST = {
    "small_pairwise":  ("pairwise_nn_small_n300_r32_bce_fisher2_speccoords.pt",
                     dict(hidden_dim=32, embed_dim=8, rounds=32, reinject_degree_signal=True,
                          use_spectral_coords=True, head="distance")),
    "medium_pairwise": ("pairwise_nn_medium_n300_r32_bce_fisher2_speccoords.pt",
                     dict(hidden_dim=32, embed_dim=8, rounds=32, reinject_degree_signal=True,
                          use_spectral_coords=True, head="distance")),
    "large_pairwise":  ("pairwise_nn_large_n300_r32_bce_fisher2_speccoords.pt",
                     dict(hidden_dim=32, embed_dim=8, rounds=32, reinject_degree_signal=True,
                          use_spectral_coords=True, head="distance")),
}

MULTISTAGE_MANIFEST = {
    "small_multistage":  ("multistage_nn_small_n300_s3_i6_o4_pos16_fisher_reinject.pt",
                     dict(hidden_dim=32, embed_dim=8, num_stages=3, inner_rounds=6, outer_iters=4,
                          pos_enc_dim=16, reinject_degree_signal=True)),
    "medium_multistage": ("multistage_nn_medium_n300_s3_i6_o4_pos16_fisher_reinject.pt",
                     dict(hidden_dim=32, embed_dim=8, num_stages=3, inner_rounds=6, outer_iters=4,
                          pos_enc_dim=16, reinject_degree_signal=True)),
    "large_multistage":  ("multistage_nn_large_n300_s3_i6_o4_pos16_fisher_reinject.pt",
                     dict(hidden_dim=32, embed_dim=8, num_stages=3, inner_rounds=6, outer_iters=4,
                          pos_enc_dim=16, reinject_degree_signal=True)),
}

MULTIHEAD_MANIFEST = {
    "small_multihead":  ("multihead_gnn_small_n300_h16_nh4_r64_fisher_reinject.pt",
                     dict(head_dim=16, embed_dim=8, num_heads=4, rounds=64, reinject_degree_signal=True)),
    "medium_multihead": ("multihead_gnn_medium_n300_h16_nh4_r64_fisher_reinject.pt",
                     dict(head_dim=16, embed_dim=8, num_heads=4, rounds=64, reinject_degree_signal=True)),
    "large_multihead":  ("multihead_gnn_large_n300_h16_nh4_r64_fisher_reinject.pt",
                     dict(head_dim=16, embed_dim=8, num_heads=4, rounds=64, reinject_degree_signal=True)),
}


def load_all():
    models = []  # list of (name, run_fn)
    for name, (fname, kwargs) in SPECTRAL_MANIFEST.items():
        m = SpectralEmbedder(**kwargs)
        m.load_state_dict(torch.load(f"{RESULTS_DIR}/{fname}"))
        m.eval()
        models.append((name, lambda adj, m=m: spectral_nn_3color(adj, m, device="cpu")))
    for name, (fname, kwargs) in PAIRWISE_MANIFEST.items():
        m = PairwiseEmbedder(**kwargs)
        m.load_state_dict(torch.load(f"{RESULTS_DIR}/{fname}"))
        m.eval()
        models.append((name, lambda adj, m=m: pairwise_nn_3color(adj, m, device="cpu")))
    for name, (fname, kwargs) in MULTISTAGE_MANIFEST.items():
        m = MultiStageEmbedder(**kwargs)
        m.load_state_dict(torch.load(f"{RESULTS_DIR}/{fname}"))
        m.eval()
        models.append((name, lambda adj, m=m: multistage_nn_3color(adj, m, device="cpu")))
    for name, (fname, kwargs) in MULTIHEAD_MANIFEST.items():
        m = MultiHeadGNNEmbedder(**kwargs)
        m.load_state_dict(torch.load(f"{RESULTS_DIR}/{fname}"))
        m.eval()
        models.append((name, lambda adj, m=m: multihead_gnn_3color(adj, m, device="cpu")))
    return models


def parse_args():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--n", type=int, default=1000)
    p.add_argument("--c-values", type=int, nargs="+", default=[5, 6, 7, 8, 9, 10, 11, 12, 13, 14])
    p.add_argument("--n-inst", type=int, default=20)
    p.add_argument("--seed", type=int, default=0)
    return p.parse_args()


def main():
    args = parse_args()
    torch.manual_seed(args.seed)
    models = load_all()
    print(f"loaded {len(models)} checkpoints: {[n for n, _ in models]}")

    # Full per-model breakdown -- not just the best single model's name --
    # so it's possible to see exactly which checkpoints contributed successes
    # to the OR at each c, not just who "won."
    col_names = ["plain"] + [name for name, _ in models] + ["ANY"]
    header = "c".rjust(4) + "".join(n.rjust(max(12, len(n) + 2)) for n in col_names)
    print(header)

    t0 = time.time()
    for c in args.c_values:
        plain_count = 0
        any_count = 0
        per_model_counts = {name: 0 for name, _ in models}
        # per-instance solved-by set, so we can also report which models were
        # the UNIQUE solver on at least one instance (contributed something
        # no other model got) vs. redundant with a stronger model.
        per_model_unique = {name: 0 for name, _ in models}

        for _ in range(args.n_inst):
            _, adj = create_planted_3col(args.n, c)
            solved_by = []

            r_plain = alon_kahale_3color(adj, variant="adjacency")
            if r_plain["success"]:
                ok, nconf, _ = is_k_color(adj.clone(), r_plain["colors"].copy())
                assert ok, f"plain spectral illegal at c={c}, nconf={nconf}"
                solved_by.append("plain")
            plain_count += r_plain["success"]

            for name, run_fn in models:
                r = run_fn(adj)
                if r["success"]:
                    ok, nconf, _ = is_k_color(adj.clone(), r["colors"].copy())
                    assert ok, f"{name} illegal at c={c}, nconf={nconf}"
                    per_model_counts[name] += 1
                    solved_by.append(name)

            if solved_by:
                any_count += 1
                if len(solved_by) == 1 and solved_by[0] != "plain":
                    per_model_unique[solved_by[0]] += 1

        row = f"{c:4d}" + f"{plain_count}/{args.n_inst}".rjust(max(12, len('plain') + 2))
        for name, _ in models:
            cell = f"{per_model_counts[name]}/{args.n_inst}"
            if per_model_unique[name] > 0:
                cell += f"*{per_model_unique[name]}"
            row += cell.rjust(max(12, len(name) + 2))
        row += f"{any_count}/{args.n_inst}".rjust(max(12, len('ANY') + 2))
        print(row)

    print("\n(*N after a model's count = it was the ONLY model that solved N of those "
          "instances -- a genuine, non-redundant contribution to the OR, not just tied "
          "with a stronger model)")
    print(f"({time.time()-t0:.0f}s elapsed)")


if __name__ == "__main__":
    main()
