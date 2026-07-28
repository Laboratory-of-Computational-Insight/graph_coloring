"""
Characterize what actually differs between a random-seed BP init that ends
up succeeding vs. one that ends up failing, on the SAME graph instance (so
any difference found is about the seed's noise, not instance difficulty).

Key attribute checked: the "iteration-0" implied per-vertex guess -- what
color each vertex's incident messages would vote for using ONLY the raw
random init psi, before a single message-passing update -- and its distance
from the planted partition. If successful seeds' iteration-0 guess is
already closer to planted than failing seeds', the init noise itself
carries recoverable signal (arguing FOR smart-init style approaches). If
there's no difference, success is coming from chaotic amplification of
noise through the iterations/reinforcement, not from anything present in
the init itself (arguing AGAINST smart-init, FOR just doing more restarts).
"""
import itertools

import numpy as np
import torch

from random_planted import create_planted_3col
from belief_propagation_coloring import bp_reinforced_coloring, _build_directed_edges
from repair_utils import count_conflicts, greedy_repair

N = 1000
N_SEEDS = 200
C_VALUES = [4.3, 4.5, 4.69]
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


def iteration0_guess(A, seed, q=3):
    """Replicates bp_reinforced_coloring's exact init, then reads off what
    the per-vertex belief would be from that raw psi alone (0 message-passing
    updates) -- i.e. purely the random noise's own implied vote."""
    rng = np.random.default_rng(seed)
    n = A.shape[0]
    src, dst, reverse = _build_directed_edges(A)
    m2 = len(src)
    psi = rng.uniform(0.9, 1.1, size=(m2, q)) * (1.0 / q)
    psi += rng.normal(scale=0.02, size=(m2, q))
    psi = np.clip(psi, 1e-6, None)
    psi /= psi.sum(axis=1, keepdims=True)

    logterm = np.log(np.clip(1.0 - psi, 1e-12, None))
    S = np.zeros((n, q))
    np.add.at(S, dst, logterm)
    return S.argmax(axis=1)


def main():
    torch.manual_seed(0)
    for c in C_VALUES:
        planted, adj = create_planted_3col(N, c)
        A = adj.numpy()
        planted_np = planted.numpy()

        records = []
        for seed in range(N_SEEDS):
            init_colors = iteration0_guess(A, seed)
            init_dist = distance_from_planted(init_colors, planted_np)

            belief, converged = bp_reinforced_coloring(A, rho=0.0, damping=0.5, max_iter=3000, seed=seed)
            final_colors = belief.argmax(axis=1)
            conf = count_conflicts(A, final_colors)
            if conf > 0:
                final_colors = greedy_repair(A, final_colors)
                conf = count_conflicts(A, final_colors)
            success = conf == 0
            final_dist = distance_from_planted(final_colors, planted_np)
            records.append({"seed": seed, "init_dist": init_dist, "success": success,
                             "final_dist": final_dist, "converged": converged})

        succ = [r for r in records if r["success"]]
        fail = [r for r in records if not r["success"]]

        print(f"\n=== c={c} ({len(succ)}/{N_SEEDS} succeeded) ===")
        print(f"  iteration-0 guess distance-from-planted (0.667=random, 0.0=perfect):")
        print(f"    successful seeds: mean={np.mean([r['init_dist'] for r in succ]):.4f}"
              if succ else "    successful seeds: n/a")
        print(f"    failing seeds:    mean={np.mean([r['init_dist'] for r in fail]):.4f}"
              if fail else "    failing seeds:    n/a")
        print(f"  converged (reached tol before max_iter):")
        print(f"    successful seeds: {sum(r['converged'] for r in succ)}/{len(succ)}" if succ else "    n/a")
        print(f"    failing seeds:    {sum(r['converged'] for r in fail)}/{len(fail)}" if fail else "    n/a")

    print("\ndone")


if __name__ == "__main__":
    main()
