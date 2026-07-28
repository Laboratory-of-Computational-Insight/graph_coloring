#!/usr/bin/env python3
"""
Train MultiHeadGNNEmbedder (multihead_gnn.py) on the three established c-bands,
one model per band, matching the convention used for spectral_nn.py/
multistage_nn.py/pairwise_nn.py tonight (small=1-8, medium=6-18, large=14-30).

Checkpoint filenames fully encode the config (band, n_train, head_dim,
num_heads, rounds, loss, reinject flag) so nothing here can ever be confused
with a different config later.
"""
import argparse
import time

import torch

from multihead_gnn import train_multihead_gnn, check_collapse

BANDS = {
    "small": (1, 8),
    "medium": (6, 18),
    "large": (14, 30),
}

RESULTS_DIR = "G:/graph_col/graph_coloring/results"


def ckpt_name(band, n_train, head_dim, num_heads, rounds, loss_fn, reinject):
    reinject_tag = "reinject" if reinject else "noreinject"
    return (f"multihead_gnn_{band}_n{n_train}_h{head_dim}_nh{num_heads}_r{rounds}"
            f"_{loss_fn}_{reinject_tag}.pt")


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--n-train", type=int, default=300)
    p.add_argument("--steps", type=int, default=1500)
    p.add_argument("--head-dim", type=int, default=16)
    p.add_argument("--num-heads", type=int, default=4)
    p.add_argument("--rounds", type=int, default=64)
    p.add_argument("--loss-fn", default="fisher")
    p.add_argument("--bands", nargs="+", default=list(BANDS.keys()))
    args = p.parse_args()

    for band in args.bands:
        c_range = BANDS[band]
        print(f"\n=== training band={band} c_range={c_range} steps={args.steps} ===")
        t0 = time.time()
        model = train_multihead_gnn(
            n_train=args.n_train,
            c_range=c_range,
            hard_window_prob=0.0,  # pure uniform sampling within the band
            steps=args.steps,
            head_dim=args.head_dim,
            embed_dim=8,
            num_heads=args.num_heads,
            rounds=args.rounds,
            reinject_degree_signal=True,
            loss_fn=args.loss_fn,
            log_every=250,
            seed=0,
        )
        print(f"band={band} training took {time.time()-t0:.0f}s")

        print(f"--- collapse check, band={band} ---")
        ok6 = check_collapse(model, n=args.n_train, c=6, seed=1)
        ok12 = check_collapse(model, n=args.n_train, c=12, seed=1)
        if not (ok6 and ok12):
            print(f"*** WARNING: band={band} model shows signs of collapse -- inspect before using ***")

        fname = ckpt_name(band, args.n_train, args.head_dim, args.num_heads,
                           args.rounds, args.loss_fn, True)
        path = f"{RESULTS_DIR}/{fname}"
        torch.save(model.state_dict(), path)
        print(f"saved {path}")


if __name__ == "__main__":
    main()
