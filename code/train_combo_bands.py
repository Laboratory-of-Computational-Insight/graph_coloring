#!/usr/bin/env python3
"""
Train the two missing backbone x pairwise-head combinations
(multistage_pairwise_nn.py, multihead_pairwise_nn.py), plus a retrain of
plain multihead_gnn.py WITH use_spectral_coords=True (that flag existed on
MultiHeadGNNEmbedder since it was built but was never actually turned on in
train_multihead_bands.py -- a real gap, not a deliberate ablation).

Uses num_spectral_eigvecs=4, the one verified real win from the pairwise
mechanism sweep (c=9: 27.5% vs 5% plain) -- carried over as the default here
rather than re-litigating eigvecs=2 vs 4 from scratch on every new
architecture.
"""
import time

import torch

RESULTS_DIR = "G:/graph_col/graph_coloring/results"
BANDS = {"small": (1, 8), "medium": (6, 18)}


def train_all(device="cuda", steps=800):
    from multistage_pairwise_nn import train_multistage_pairwise_nn
    from multihead_pairwise_nn import train_multihead_pairwise_nn
    from multihead_gnn import train_multihead_gnn, check_collapse as check_collapse_mh

    for seed, (band_name, c_range) in enumerate(BANDS.items()):
        print(f"\n=== multistage_pairwise / {band_name} c_range={c_range} ===")
        t0 = time.time()
        m = train_multistage_pairwise_nn(
            n_train=300, c_range=c_range, steps=steps, hidden_dim=32, embed_dim=8,
            num_stages=3, inner_rounds=6, outer_iters=4, pos_enc_dim=16,
            reinject_degree_signal=True, use_spectral_coords=True,
            num_spectral_eigvecs=4, head="distance", fisher_weight=2.0,
            device=device, log_every=200, seed=seed,
        )
        fname = f"{RESULTS_DIR}/multistage_pairwise_nn_{band_name}_n300_eig4.pt"
        torch.save(m.state_dict(), fname)
        print(f"saved {fname} ({time.time()-t0:.0f}s)")

    for seed, (band_name, c_range) in enumerate(BANDS.items()):
        print(f"\n=== multihead_pairwise / {band_name} c_range={c_range} ===")
        t0 = time.time()
        m = train_multihead_pairwise_nn(
            n_train=300, c_range=c_range, steps=steps, head_dim=16, embed_dim=8,
            num_heads=4, rounds=32, reinject_degree_signal=True, use_spectral_coords=True,
            num_spectral_eigvecs=4, head="distance", fisher_weight=2.0,
            device=device, log_every=200, seed=seed,
        )
        fname = f"{RESULTS_DIR}/multihead_pairwise_nn_{band_name}_n300_eig4.pt"
        torch.save(m.state_dict(), fname)
        print(f"saved {fname} ({time.time()-t0:.0f}s)")

    for seed, (band_name, c_range) in enumerate(BANDS.items()):
        print(f"\n=== multihead_gnn (PCA readout) + speccoords / {band_name} c_range={c_range} ===")
        t0 = time.time()
        m = train_multihead_gnn(
            n_train=300, c_range=c_range, hard_window_prob=0.0, steps=steps,
            head_dim=16, embed_dim=8, num_heads=4, rounds=64,
            reinject_degree_signal=True, use_spectral_coords=True,
            loss_fn="fisher", device=device, log_every=200, seed=seed,
        )
        print(f"--- collapse check, {band_name} ---")
        check_collapse_mh(m.to("cpu"), n=300, c=9, seed=1)
        m = m.to(device)
        fname = f"{RESULTS_DIR}/multihead_gnn_{band_name}_n300_h16_nh4_r64_fisher_reinject_speccoords.pt"
        torch.save(m.state_dict(), fname)
        print(f"saved {fname} ({time.time()-t0:.0f}s)")


if __name__ == "__main__":
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("--device", default="cuda")
    p.add_argument("--steps", type=int, default=800)
    args = p.parse_args()
    train_all(device=args.device, steps=args.steps)
