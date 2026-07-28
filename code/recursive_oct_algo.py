#!/usr/bin/env python3
"""
Recursive soft-core (OCT) 3-coloring (2026-07-16/17, user's idea): the
single-level soft-core is too large to backtrack on directly (34-51% of the
graph at c=5-9), but the soft-core OF the soft-core shrinks dramatically
(empirically, e.g. c=9: 1000 -> 515 -> 182 -> 16). This file implements the
FULL recursive reduction correctly -- not just measuring induced-subgraph
OCT sizes level by level (which ignores boundary constraints to
already-colored vertices), but actually constructing a valid global coloring:

At each level, given a "remaining" (not-yet-colored) vertex set:
1. Find the OCT of the subgraph induced on `remaining` (ignoring, at this
   point, its as-yet-unresolved connections to the next-deeper level -- but
   NOT ignoring connections to already-colored ancestors, which matter).
2. `remaining - OCT` is bipartite. Split it into connected components (within
   the bipartite part). For EACH component independently, choose which 2 of
   the 3 real colors correspond to its two parity classes (and which
   orientation) -- trying all valid pairings and picking one consistent with
   any already-fixed neighbor colors from ancestor levels (a component with
   no fixed neighbors yet just picks colors 0/1 by convention).
3. Recurse into the OCT set (now a smaller `remaining`), which -- critically
   -- now has the just-colored `remaining - OCT` vertices as FIXED
   neighbors, exactly like spectral_coloring.py's existing cleanup/backtrack
   convention.
4. Base case: once `remaining` is small enough, hand off to exact
   backtracking (_solve_component, already used elsewhere in this project).
"""
import time

import numpy as np
import torch

from random_planted import create_planted_3col
from spectral_coloring import _solve_component
from gc_utils import is_k_color
from oct_algo import find_oct_greedy


def _color_bipartite_part(A, colors, comp_vertices, parity_lookup):
    """
    comp_vertices: list of global vertex indices forming ONE connected
    component of the (already known bipartite) `remaining - OCT` part.
    parity_lookup: dict global_vertex -> 0/1 (its parity class within this
    component's bipartition).
    Tries all 6 (3 unordered color pairs x 2 orientations) ways to map
    {parity 0, parity 1} -> two distinct real colors, picking the first that
    doesn't conflict with any already-fixed (colors >= 0) neighbor. Mutates
    `colors` in place. Returns True if a valid assignment was found.
    """
    n = A.shape[0]
    color_pairs = [(0, 1), (0, 2), (1, 2), (1, 0), (2, 0), (2, 1)]
    comp_set = set(comp_vertices)

    for c_for_0, c_for_1 in color_pairs:
        ok = True
        for v in comp_vertices:
            my_color = c_for_0 if parity_lookup[v] == 0 else c_for_1
            neighbors = np.where(A[v] == 1)[0]
            for u in neighbors:
                if u not in comp_set and colors[u] == my_color:
                    ok = False
                    break
            if not ok:
                break
        if ok:
            for v in comp_vertices:
                colors[v] = c_for_0 if parity_lookup[v] == 0 else c_for_1
            return True
    return False


def recursive_oct_solve(A, colors, remaining, max_final_component=60, max_depth=15, _depth=0):
    """Mutates `colors` in place. Returns True on success."""
    if len(remaining) == 0:
        return True
    if len(remaining) <= max_final_component or _depth >= max_depth:
        fixed_mask = colors >= 0
        return _solve_component(list(remaining), A, colors, fixed_mask)

    A_sub = A[np.ix_(remaining, remaining)]
    parity_sub, oct_local = find_oct_greedy(A_sub)
    oct_local_set = set(oct_local)
    non_oct_local = [i for i in range(len(remaining)) if i not in oct_local_set]

    if len(non_oct_local) == 0:
        # Entire remaining set is its own OCT (degenerate) -- fall back to backtracking.
        fixed_mask = colors >= 0
        return _solve_component(list(remaining), A, colors, fixed_mask)

    # Build the bipartite-part-only adjacency (within `remaining`, excluding OCT vertices)
    # to find connected components.
    non_oct_global = [remaining[i] for i in non_oct_local]
    parity_lookup_local = {remaining[i]: int(parity_sub[i]) for i in non_oct_local}

    A_bip = A[np.ix_(non_oct_global, non_oct_global)]
    n_bip = len(non_oct_global)
    visited = np.zeros(n_bip, dtype=bool)
    components = []
    for start in range(n_bip):
        if visited[start]:
            continue
        visited[start] = True
        stack = [start]
        comp = [start]
        while stack:
            v = stack.pop()
            for u in np.where(A_bip[v] == 1)[0]:
                if not visited[u]:
                    visited[u] = True
                    comp.append(u)
                    stack.append(u)
        components.append([non_oct_global[i] for i in comp])

    for comp in components:
        success = _color_bipartite_part(A, colors, comp, parity_lookup_local)
        if not success:
            return False  # no valid 2-color-to-3-color mapping for this component

    oct_global = np.array([remaining[i] for i in oct_local])
    return recursive_oct_solve(A, colors, oct_global, max_final_component=max_final_component,
                                max_depth=max_depth, _depth=_depth + 1)


def recursive_oct_3color(adj, max_final_component=60):
    A = adj.detach().cpu().numpy() if isinstance(adj, torch.Tensor) else np.asarray(adj)
    n = A.shape[0]
    colors = -np.ones(n, dtype=np.int64)
    remaining = np.arange(n)
    ok = recursive_oct_solve(A, colors, remaining, max_final_component=max_final_component)

    same = colors[:, None] == colors[None, :]
    valid_mask = (colors[:, None] >= 0) & (colors[None, :] >= 0)
    conflicts = int((A.astype(bool) & same & valid_mask).sum() // 2)
    success = ok and (colors >= 0).all() and conflicts == 0
    return {"success": success, "colors": colors, "conflicts": conflicts, "all_colored": bool((colors >= 0).all())}


if __name__ == "__main__":
    for c in [5, 6, 7, 8, 9]:
        print(f"\n========== c={c} ==========")
        t0 = time.time()
        successes = 0
        for i in range(10):
            _, adj = create_planted_3col(1000, c)
            r = recursive_oct_3color(adj)
            if r["success"]:
                ok, nconf, _ = is_k_color(adj.clone(), r["colors"].copy())
                assert ok, f"illegal coloring reported as success! nconf={nconf}"
                successes += 1
            print(f"  inst {i+1}: success={r['success']}  all_colored={r['all_colored']}  conflicts={r['conflicts']}")
        print(f"  ({successes}/10 successes, {time.time()-t0:.0f}s for 10 instances)")
