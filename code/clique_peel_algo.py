#!/usr/bin/env python3
"""
User's proposed classical algorithm (2026-07-16):

1. Remove all nodes with degree 0 or 1 (trivially resolvable later -- a
   degree<=1 vertex always has a free color among 3 once its <=1 neighbor is
   colored, same logic as classical_baseline.py's peel_3color, just a lighter
   threshold here per the user's exact spec: "no neighbors or 1 neighbor",
   not <=2).
2. Take the complement of the residual graph. Each true color class (an
   independent set in the original graph) is a CLIQUE in the complement.
3. Find a large clique in the complement via LEAST DEGREE REMOVAL (Feige-Ron):
   start with the FULL candidate set, repeatedly remove the vertex with the
   LOWEST degree within the current candidate-induced subgraph, until what
   remains is a genuine clique (every remaining pair connected). This is the
   reverse of a greedy-construction heuristic (which grows a clique by
   repeatedly adding high-degree vertices) -- LDR instead shrinks the full
   set down to a clique by stripping away whichever vertex is currently
   least-connected to the rest of the candidate pool.
4. Remove that clique from the residual, repeat twice more (3 cliques total
   -- hopefully approximating the 3 color classes).
5. Report: how many nodes are left unaccounted for, and how "pure" each
   found clique is (what fraction of its members share the same TRUE
   planted color -- using ground truth only for diagnostic purposes, not
   fed into the algorithm itself, which is unsupervised).
"""
import time

import numpy as np

from random_planted import create_planted_3col


def prune_low_degree(A):
    """Remove nodes with degree 0 or 1. Returns (keep_mask, A_pruned_full_size)
    -- A_pruned_full_size has pruned rows/cols zeroed out (keeps original
    indexing, simpler for reporting against ground truth)."""
    n = A.shape[0]
    alive = np.ones(n, dtype=bool)
    degree = A.sum(axis=1)
    changed = True
    while changed:
        changed = False
        low = np.where(alive & (degree <= 1))[0]
        for v in low:
            if not alive[v]:
                continue
            alive[v] = False
            changed = True
            neighbors = np.where(A[v] == 1)[0]
            for u in neighbors:
                if alive[u]:
                    degree[u] -= 1
    return alive


def least_degree_removal_clique(comp_adj, candidate_mask):
    """Feige-Ron least-degree-removal heuristic: start with the FULL
    candidate set, repeatedly remove the vertex with the LOWEST degree
    within the current candidate-induced subgraph, until the remaining set
    is a genuine clique (every remaining vertex has degree == len(remaining)-1
    within the subgraph, i.e. connected to everyone else remaining)."""
    candidates = np.where(candidate_mask)[0]
    while len(candidates) > 1:
        sub = comp_adj[np.ix_(candidates, candidates)]
        m = len(candidates)
        deg_in_sub = sub.sum(axis=1)
        if np.all(deg_in_sub == m - 1):
            break  # already a clique -- every vertex connected to all others
        worst_local = np.argmin(deg_in_sub)
        candidates = np.delete(candidates, worst_local)
    return candidates.tolist()


def run_algorithm(n, c, seed=None):
    assignment, adj = create_planted_3col(n, c)
    A = adj.numpy().astype(np.int8)
    true_colors = assignment.numpy()

    alive = prune_low_degree(A)
    n_pruned = int((~alive).sum())
    residual_idx = np.where(alive)[0]

    A_res = A[np.ix_(residual_idx, residual_idx)]
    comp = 1 - A_res
    np.fill_diagonal(comp, 0)

    cliques = []
    remaining_mask = np.ones(len(residual_idx), dtype=bool)
    for _ in range(3):
        clique_local = least_degree_removal_clique(comp, remaining_mask)
        cliques.append([int(residual_idx[i]) for i in clique_local])
        for i in clique_local:
            remaining_mask[i] = False

    accounted = n_pruned + sum(len(c_) for c_ in cliques)
    unaccounted = n - accounted

    purities = []
    for clique in cliques:
        if len(clique) == 0:
            purities.append((0, 0, 0.0, {}))
            continue
        colors_in_clique = true_colors[clique]
        vals, counts = np.unique(colors_in_clique, return_counts=True)
        majority_count = counts.max()
        breakdown = dict(zip(vals.tolist(), counts.tolist()))
        purities.append((len(clique), int(majority_count), majority_count / len(clique), breakdown))

    return {
        "n_pruned": n_pruned,
        "clique_sizes": [len(c_) for c_ in cliques],
        "unaccounted": unaccounted,
        "purities": purities,
    }


if __name__ == "__main__":
    n, c, n_inst = 1000, 7, 10
    t0 = time.time()
    for i in range(n_inst):
        result = run_algorithm(n, c, seed=i)
        print(f"\n--- instance {i+1}/{n_inst} ---")
        print(f"  pruned (degree<=1): {result['n_pruned']}")
        print(f"  clique sizes: {result['clique_sizes']}")
        print(f"  unaccounted (n - pruned - sum(cliques)): {result['unaccounted']}")
        for j, (size, maj_count, purity, breakdown) in enumerate(result["purities"]):
            print(f"  clique {j+1}: size={size}  majority_color_count={maj_count}  "
                  f"purity={purity:.3f}  color_breakdown={breakdown}")
    print(f"\n({time.time()-t0:.0f}s elapsed for {n_inst} instances)")
