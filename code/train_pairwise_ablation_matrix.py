#!/usr/bin/env python3
"""
Systematic combination sweep (2026-07-15) for the new PairwiseEmbedder
mechanisms (per_dim_scale, use_common_neighbors, num_spectral_eigvecs,
use_dual_spectral_coords + residual_spectral_coords) -- targeted at c=7,8,9,
the thin-signal zone identified by the representation-level diagnostics.

Not a full 2^5 combinatorial explosion (too expensive) -- prioritized configs
that test specific hypotheses:
  - baseline: reproduces the existing small_pairwise/medium_pairwise recipe
  - dualres: the ORIGINAL negative result (dual coords + residual, no fix)
  - dualres_pds: same, + per_dim_scale -- the direct fix for the diagnosed
    "uniform distance can't downweight noisy dims" root cause
  - pds_only: per_dim_scale alone, no dual/residual -- does it help even
    without the mechanism it was designed to fix?
  - cn_only: common-neighbor feature alone
  - eig4_only: more eigenvectors (4 instead of 2) alone
  - dualres_pds_cn: the fix + common-neighbors combined
  - dualres_pds_eig4: the fix + more eigenvectors combined
  - kitchen_sink: all mechanisms combined

Trained on GPU, hidden_dim=32 (capacity already ruled out as the bottleneck
tonight), small+medium bands (covers c=1-8 and c=6-18, both include c=7-9).
"""
import time

import torch

from pairwise_nn import train_pairwise_nn

RESULTS_DIR = "G:/graph_col/graph_coloring/results"
BANDS = {"small": (1, 8), "medium": (6, 18)}

CONFIGS = {
    "dualres_pds": dict(use_spectral_coords=False, use_dual_spectral_coords=True,
                         residual_spectral_coords=True, per_dim_scale=True),
    "pds_only": dict(use_spectral_coords=True, per_dim_scale=True),
    "cn_only": dict(use_spectral_coords=True, use_common_neighbors=True),
    "eig4_only": dict(use_spectral_coords=True, num_spectral_eigvecs=4),
    "dualres_pds_cn": dict(use_spectral_coords=False, use_dual_spectral_coords=True,
                            residual_spectral_coords=True, per_dim_scale=True,
                            use_common_neighbors=True),
    "dualres_pds_eig4": dict(use_spectral_coords=False, use_dual_spectral_coords=True,
                              residual_spectral_coords=True, per_dim_scale=True,
                              num_spectral_eigvecs=4),
    "kitchen_sink": dict(use_spectral_coords=False, use_dual_spectral_coords=True,
                          residual_spectral_coords=True, per_dim_scale=True,
                          use_common_neighbors=True, num_spectral_eigvecs=4),
}


def main():
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("--configs", nargs="+", default=list(CONFIGS.keys()))
    p.add_argument("--bands", nargs="+", default=list(BANDS.keys()))
    p.add_argument("--steps", type=int, default=800)
    p.add_argument("--device", default="cuda")
    args = p.parse_args()

    for cfg_name in args.configs:
        extra_kwargs = CONFIGS[cfg_name]
        for seed, band_name in enumerate(args.bands):
            c_range = BANDS[band_name]
            print(f"\n=== {cfg_name} / {band_name} c_range={c_range} ===")
            t0 = time.time()
            model = train_pairwise_nn(
                n_train=300, c_range=c_range, steps=args.steps, lr=1e-3,
                hidden_dim=32, embed_dim=8, rounds=32, reinject_degree_signal=True,
                head="distance", n_pairs=4000, fisher_weight=2.0,
                device=args.device, log_every=200, seed=seed,
                **extra_kwargs,
            )
            fname = f"{RESULTS_DIR}/pairwise_nn_{band_name}_ablation_{cfg_name}.pt"
            torch.save(model.state_dict(), fname)
            print(f"saved {fname}  ({time.time()-t0:.0f}s)")


if __name__ == "__main__":
    main()
