#!/usr/bin/env python3
"""Train GPSPairwiseEmbedder on small+medium bands. Same eig4 default as the
other combos tonight (the one mechanism confirmed to actually help at c=9)."""
import time

import torch

from gps_pairwise_nn import train_gps_pairwise_nn

RESULTS_DIR = "G:/graph_col/graph_coloring/results"
BANDS = {"small": (1, 8), "medium": (6, 18)}


def main():
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("--device", default="cuda")
    p.add_argument("--steps", type=int, default=800)
    args = p.parse_args()

    for seed, (band_name, c_range) in enumerate(BANDS.items()):
        print(f"\n=== gps_pairwise / {band_name} c_range={c_range} ===")
        t0 = time.time()
        m = train_gps_pairwise_nn(
            n_train=300, c_range=c_range, steps=args.steps, hidden_dim=32, embed_dim=8,
            num_layers=4, heads=4, use_spectral_coords=True, num_spectral_eigvecs=4,
            head="distance", fisher_weight=2.0, device=args.device, log_every=200, seed=seed,
        )
        fname = f"{RESULTS_DIR}/gps_pairwise_nn_{band_name}_n300_eig4.pt"
        torch.save(m.state_dict(), fname)
        print(f"saved {fname} ({time.time()-t0:.0f}s)")


if __name__ == "__main__":
    main()
