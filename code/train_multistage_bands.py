#!/usr/bin/env python3
"""
Train MultiStageEmbedder (multistage_nn.py) on the three c-bands, one model
per band, matching the band definitions used elsewhere tonight (small=1-8,
medium=6-18, large=14-30 -- see pairwise_nn.py's train_pairwise_nn for the
same band convention). c is sampled uniformly within the band every step (no
extra hard-window resampling bias), per this task's spec.

Checkpoint filenames fully encode the config (band, n_train, stage/inner/outer
loop shape, positional-encoding width, loss, reinject flag) so nothing here
can ever be confused with a different config later.
"""
import argparse
import time

import torch

from multistage_nn import train_multistage_nn, check_collapse

BANDS = {
    "small": (1, 8),
    "medium": (6, 18),
    "large": (14, 30),
}

RESULTS_DIR = "G:/graph_col/graph_coloring/results"


def ckpt_name(band, n_train, num_stages, inner_rounds, outer_iters, pos_enc_dim, loss_fn, reinject):
    reinject_tag = "reinject" if reinject else "noreinject"
    return (f"multistage_nn_{band}_n{n_train}_s{num_stages}_i{inner_rounds}_o{outer_iters}"
            f"_pos{pos_enc_dim}_{loss_fn}_{reinject_tag}.pt")


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--n-train", type=int, default=300)
    p.add_argument("--steps", type=int, default=1500)
    p.add_argument("--num-stages", type=int, default=3)
    p.add_argument("--inner-rounds", type=int, default=6)
    p.add_argument("--outer-iters", type=int, default=4)
    p.add_argument("--pos-enc-dim", type=int, default=16)
    p.add_argument("--loss-fn", default="fisher")
    p.add_argument("--bands", nargs="+", default=list(BANDS.keys()))
    args = p.parse_args()

    for band in args.bands:
        c_range = BANDS[band]
        print(f"\n=== training band={band} c_range={c_range} steps={args.steps} ===")
        t0 = time.time()
        model = train_multistage_nn(
            n_train=args.n_train,
            c_range=c_range,
            hard_window_prob=0.0,  # pure uniform sampling within the band, per task spec
            steps=args.steps,
            hidden_dim=32,
            embed_dim=8,
            num_stages=args.num_stages,
            inner_rounds=args.inner_rounds,
            outer_iters=args.outer_iters,
            reinject_degree_signal=True,
            pos_enc_dim=args.pos_enc_dim,
            loss_fn=args.loss_fn,
            log_every=250,
            seed=0,
        )
        print(f"band={band} training took {time.time()-t0:.0f}s")

        # Mandatory empirical collapse check before trusting the trained model
        # (per this task's instructions -- loss curves alone can hide it).
        print(f"--- collapse check, band={band} ---")
        ok6 = check_collapse(model, n=args.n_train, c=6, seed=1)
        ok12 = check_collapse(model, n=args.n_train, c=12, seed=1)
        if not (ok6 and ok12):
            print(f"*** WARNING: band={band} model shows signs of collapse -- inspect before using ***")

        fname = ckpt_name(band, args.n_train, args.num_stages, args.inner_rounds,
                           args.outer_iters, args.pos_enc_dim, args.loss_fn, True)
        path = f"{RESULTS_DIR}/{fname}"
        torch.save(model.state_dict(), path)
        print(f"saved {path}")


if __name__ == "__main__":
    main()
