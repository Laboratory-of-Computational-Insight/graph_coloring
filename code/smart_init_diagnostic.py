"""
Cheap first check for "can we build a smart BP init": before touching
bp_reinforced_coloring's init mechanism at all, just measure whether the
classical spectral guesses (plain adjacency-spectral, Bethe-Hessian) --
zero BP iterations, no reinforcement, nothing -- are already better-than-
random w.r.t. the PLANTED partition at the hard c-band. If they're not,
seeding BP from them can't help (biasing toward noise is still noise). If
they ARE better than random, that's a concrete, cheap signal worth wiring
into BP's init.
"""
import itertools

import numpy as np
import torch

from random_planted import create_planted_3col
from spectral_coloring import spectral_init_coloring

N = 1000
N_INST = 10
C_VALUES = [3, 4, 4.3, 4.5, 4.69, 5, 6]
PERMS = list(itertools.permutations(range(3)))


def distance_from_planted(colors, planted):
    colors = np.asarray(colors)
    planted = np.asarray(planted)
    best_match = 0
    for perm in PERMS:
        remapped = np.array(perm)[colors]
        match = (remapped == planted).sum()
        best_match = max(best_match, match)
    return 1.0 - best_match / len(planted)


def main():
    torch.manual_seed(0)
    print(f"{'c':>6}{'adjacency-spectral':>22}{'bethe-hessian':>18}")
    for c in C_VALUES:
        c_disp = int(c) if c == int(c) else c
        dist_adj, dist_bh = [], []
        for _ in range(N_INST):
            planted, adj = create_planted_3col(N, c)
            planted_np = planted.numpy()

            colors_adj = spectral_init_coloring(adj, variant="adjacency")
            colors_bh = spectral_init_coloring(adj, variant="bethe_hessian")

            dist_adj.append(distance_from_planted(colors_adj, planted_np))
            dist_bh.append(distance_from_planted(colors_bh, planted_np))

        print(f"{c_disp!s:>6}{np.mean(dist_adj):>22.4f}{np.mean(dist_bh):>18.4f}", flush=True)

    print("\n(0.0 = perfect recovery, 0.667 = random-guess baseline for 3 balanced classes)")


if __name__ == "__main__":
    main()
