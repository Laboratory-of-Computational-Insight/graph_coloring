#!/usr/bin/env python3
"""
Odd Cycle Transversal (OCT) based 3-coloring (2026-07-16, user's idea): the
"focal point" nodes causing the odd-cycle obstruction to 2-coloring are
exactly a (minimum) odd cycle transversal -- the smallest vertex set whose
removal makes the graph bipartite. Finding the true minimum is NP-hard, but
a standard greedy heuristic is cheap and gives a real, usable (not
necessarily minimum) transversal:

1. BFS-color the graph by parity (0/1) from an arbitrary root per connected
   component -- the standard bipartiteness check.
2. Any edge connecting two same-parity vertices is a "conflict edge" -- it
   closes an odd cycle relative to the BFS tree.
3. Greedily pick the vertex touching the most remaining conflict edges, add
   it to the OCT set, remove its conflict edges, repeat until none remain
   (a greedy vertex-cover heuristic on the conflict-edge graph).
4. G minus the OCT set is now exactly bipartite -- colored 0/1 via the
   already-computed BFS parities, no further work needed there.
5. The OCT vertices are the real "focal points that need addressing": solve
   them as a small residual coloring problem via exact backtracking
   (reusing spectral_coloring.py's _solve_component -- same machinery
   already used for the cleanup phase's small residual components),
   respecting their fixed neighbors' colors in the bipartite remainder plus
   any edges among OCT vertices themselves.
"""
import time

import numpy as np
import torch

from random_planted import create_planted_3col
from spectral_coloring import _solve_component
from gc_utils import is_k_color


def find_oct_greedy(A):
    """Returns (colors01, oct_set) where colors01 is an (n,) array with 0/1
    for non-OCT vertices and -1 for OCT vertices (still to be resolved),
    and oct_set is the list of OCT vertex indices."""
    n = A.shape[0]
    parity = -np.ones(n, dtype=np.int64)
    visited = np.zeros(n, dtype=bool)

    for start in range(n):
        if visited[start]:
            continue
        visited[start] = True
        parity[start] = 0
        queue = [start]
        qi = 0
        while qi < len(queue):
            v = queue[qi]
            qi += 1
            for u in np.where(A[v] == 1)[0]:
                if not visited[u]:
                    visited[u] = True
                    parity[u] = 1 - parity[v]
                    queue.append(u)

    # Conflict edges: same-parity endpoints.
    conflict_adj = {}  # vertex -> set of vertices it conflicts with
    for v in range(n):
        neighbors = np.where(A[v] == 1)[0]
        for u in neighbors:
            if u > v and parity[u] == parity[v]:
                conflict_adj.setdefault(v, set()).add(u)
                conflict_adj.setdefault(u, set()).add(v)

    oct_set = set()
    # Greedy vertex cover on the conflict-edge graph.
    while conflict_adj:
        worst = max(conflict_adj, key=lambda v: len(conflict_adj[v]))
        oct_set.add(worst)
        for u in list(conflict_adj[worst]):
            conflict_adj[u].discard(worst)
            if not conflict_adj[u]:
                del conflict_adj[u]
        del conflict_adj[worst]

    colors01 = parity.copy()
    for v in oct_set:
        colors01[v] = -1
    return colors01, sorted(oct_set)


def oct_3color(adj, max_component_size=200, node_budget=200_000):
    """Full pipeline: greedy OCT + exact bipartite coloring on the rest +
    backtracking on the OCT vertices. Returns dict: success, colors, oct_size, conflicts."""
    A = adj.detach().cpu().numpy() if isinstance(adj, torch.Tensor) else np.asarray(adj)
    n = A.shape[0]

    colors, oct_set = find_oct_greedy(A)
    fixed_mask = colors >= 0

    if len(oct_set) == 0:
        ok = True
    elif len(oct_set) > max_component_size:
        ok = False  # too large to brute-force here; report size, don't attempt
    else:
        ok = _solve_component(oct_set, A, colors, fixed_mask, node_budget=node_budget)

    same = colors[:, None] == colors[None, :]
    valid_mask = (colors[:, None] >= 0) & (colors[None, :] >= 0)
    conflicts = int((A.astype(bool) & same & valid_mask).sum() // 2)

    success = ok and (colors >= 0).all() and conflicts == 0
    return {"success": success, "colors": colors, "oct_size": len(oct_set), "conflicts": conflicts,
            "solve_attempted": len(oct_set) <= max_component_size}


if __name__ == "__main__":
    for c in [5, 6, 7, 8, 9]:
        print(f"\n========== c={c} ==========")
        t0 = time.time()
        for i in range(10):
            _, adj = create_planted_3col(1000, c)
            r = oct_3color(adj)
            if r["success"]:
                ok, nconf, _ = is_k_color(adj.clone(), r["colors"].copy())
                assert ok, f"illegal coloring reported as success! nconf={nconf}"
            print(f"  inst {i+1}: oct_size={r['oct_size']}  solve_attempted={r['solve_attempted']}  "
                  f"success={r['success']}  remaining_conflicts={r['conflicts']}")
        print(f"  ({time.time()-t0:.0f}s for 10 instances)")
