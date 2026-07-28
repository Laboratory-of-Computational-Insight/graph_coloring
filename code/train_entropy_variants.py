#!/usr/bin/env python3
"""
Entropy-penalty variant (2026-07-17, user's request): direct symmetry-breaking
regularizer added to ALL FOUR pairwise-head architectures (single-cell,
multi-stage, multi-head, GraphGPS). Targeted specifically at c=5-9 (the
still-unsolved zone) since c>=10 is already closed -- small+medium bands only,
skip large.
"""
import time

import torch

RESULTS_DIR = "G:/graph_col/graph_coloring/results"
BANDS = {"small": (1, 8), "medium": (6, 18)}
ENTROPY_WEIGHT = 0.5


def main():
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("--device", default="cuda")
    p.add_argument("--steps", type=int, default=800)
    args = p.parse_args()

    from pairwise_nn import train_pairwise_nn
    from multistage_pairwise_nn import train_multistage_pairwise_nn
    from multihead_pairwise_nn import train_multihead_pairwise_nn
    from gps_pairwise_nn import train_gps_pairwise_nn

    for seed, (band, c_range) in enumerate(BANDS.items()):
        print(f"\n=== pairwise + entropy / {band} c_range={c_range} ===")
        t0 = time.time()
        m = train_pairwise_nn(n_train=300, c_range=c_range, steps=args.steps, hidden_dim=32, embed_dim=8,
            rounds=32, reinject_degree_signal=True, use_spectral_coords=True, num_spectral_eigvecs=4,
            head="distance", fisher_weight=2.0, entropy_weight=ENTROPY_WEIGHT,
            device=args.device, log_every=200, seed=seed)
        torch.save(m.state_dict(), f"{RESULTS_DIR}/pairwise_nn_{band}_eig4_entropy.pt")
        print(f"saved ({time.time()-t0:.0f}s)")

    for seed, (band, c_range) in enumerate(BANDS.items()):
        print(f"\n=== multistage_pairwise + entropy / {band} c_range={c_range} ===")
        t0 = time.time()
        m = train_multistage_pairwise_nn(n_train=300, c_range=c_range, steps=args.steps, hidden_dim=32, embed_dim=8,
            num_stages=3, inner_rounds=6, outer_iters=4, pos_enc_dim=16,
            reinject_degree_signal=True, use_spectral_coords=True, num_spectral_eigvecs=4,
            head="distance", fisher_weight=2.0, entropy_weight=ENTROPY_WEIGHT,
            device=args.device, log_every=200, seed=seed)
        torch.save(m.state_dict(), f"{RESULTS_DIR}/multistage_pairwise_nn_{band}_eig4_entropy.pt")
        print(f"saved ({time.time()-t0:.0f}s)")

    for seed, (band, c_range) in enumerate(BANDS.items()):
        print(f"\n=== multihead_pairwise + entropy / {band} c_range={c_range} ===")
        t0 = time.time()
        m = train_multihead_pairwise_nn(n_train=300, c_range=c_range, steps=args.steps, head_dim=16, embed_dim=8,
            num_heads=4, rounds=32, reinject_degree_signal=True, use_spectral_coords=True, num_spectral_eigvecs=4,
            head="distance", fisher_weight=2.0, entropy_weight=ENTROPY_WEIGHT,
            device=args.device, log_every=200, seed=seed)
        torch.save(m.state_dict(), f"{RESULTS_DIR}/multihead_pairwise_nn_{band}_eig4_entropy.pt")
        print(f"saved ({time.time()-t0:.0f}s)")

    for seed, (band, c_range) in enumerate(BANDS.items()):
        print(f"\n=== gps_pairwise + entropy / {band} c_range={c_range} ===")
        t0 = time.time()
        m = train_gps_pairwise_nn(n_train=300, c_range=c_range, steps=args.steps, hidden_dim=32, embed_dim=8,
            num_layers=4, heads=4, use_spectral_coords=True, num_spectral_eigvecs=4,
            head="distance", fisher_weight=2.0, entropy_weight=ENTROPY_WEIGHT,
            device=args.device, log_every=200, seed=seed)
        torch.save(m.state_dict(), f"{RESULTS_DIR}/gps_pairwise_nn_{band}_eig4_entropy.pt")
        print(f"saved ({time.time()-t0:.0f}s)")


if __name__ == "__main__":
    main()
