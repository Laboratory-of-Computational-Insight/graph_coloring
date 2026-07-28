#!/usr/bin/env python3
"""
Train PairwiseEmbedder (pairwise_nn.py) on the three established c-bands
(small/medium/large), mirroring the band structure already used for
spectral_nn.py's checkpoints tonight. Training is at n_train=300 (cheap);
evaluation/reporting must be done separately at n=1000 (see
pairwise_nn_matrix_eval.py) -- never report success rates from n=300.

Config (see pairwise_nn.py's pairwise_bce_loss / train_pairwise_nn
docstrings for why): use_spectral_coords=True + fisher_weight=2.0 is not
optional cosmetic tuning -- plain BCE from a cold start reliably collapses to
"always predict different" (verified via check_collapse(), see dev notes in
pairwise_nn.py). rounds=32 is a speed/quality compromise (spectral_nn.py
uses 64; 32 already showed real above-baseline separation in a quick check
and roughly halves training time for this same-day screening pass).

Checkpoint filenames fully encode the config -- never reused for a different
config (project-wide convention).
"""
import time

import torch

from pairwise_nn import train_pairwise_nn

RESULTS_DIR = "G:/graph_col/graph_coloring/results"

BANDS = {
    "small": (1, 8),
    "medium": (6, 18),
    "large": (14, 30),
}

CKPT_SUFFIX = "n300_r32_bce_fisher2_speccoords"


def main():
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("--dual-coords", action="store_true",
                    help="Feed both adjacency + Bethe-Hessian raw coords at init (4 dims, not 2)")
    p.add_argument("--residual-coords", action="store_true",
                    help="Skip-connect raw spectral coords onto the final embedding before readout")
    p.add_argument("--bands", nargs="+", default=list(BANDS.keys()))
    p.add_argument("--steps", type=int, default=800)
    p.add_argument("--hidden-dim", type=int, default=32)
    p.add_argument("--device", default="cpu", choices=["cpu", "cuda"])
    args = p.parse_args()

    suffix = CKPT_SUFFIX
    if args.dual_coords:
        suffix += "_dualcoords"
    if args.residual_coords:
        suffix += "_residualcoords"
    if args.hidden_dim != 32:
        suffix += f"_h{args.hidden_dim}"

    for seed, name in enumerate(args.bands):
        c_range = BANDS[name]
        print(f"\n=== training band '{name}', c_range={c_range} hidden_dim={args.hidden_dim} "
              f"device={args.device} (dual_coords={args.dual_coords}, residual_coords={args.residual_coords}) ===")
        t0 = time.time()
        model = train_pairwise_nn(
            n_train=300,
            c_range=c_range,
            steps=args.steps,
            lr=1e-3,
            hidden_dim=args.hidden_dim,
            embed_dim=8,
            rounds=32,
            reinject_degree_signal=True,
            use_spectral_coords=not args.dual_coords,  # dual mode supersedes single-variant mode
            spectral_coords_variant="adjacency",
            use_dual_spectral_coords=args.dual_coords,
            residual_spectral_coords=args.residual_coords,
            head="distance",
            n_pairs=4000,
            fisher_weight=2.0,
            device=args.device,
            log_every=100,
            seed=seed,
        )
        fname = f"{RESULTS_DIR}/pairwise_nn_{name}_{suffix}.pt"
        torch.save(model.state_dict(), fname)
        print(f"saved {fname}  ({time.time()-t0:.0f}s)")


if __name__ == "__main__":
    main()
