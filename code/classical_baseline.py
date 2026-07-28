"""
Classical (non-GNN) baseline for legal 3-coloring of planted graphs, matched to
create_planted_3col(n, c).

Two methods, applied in order:

1. peel_3color: repeatedly remove any vertex of current degree <= 2. If the whole
   graph peels away (i.e. the 3-core is empty), colors can always be assigned by
   processing removals in reverse: each vertex had at most 2 already-colored
   neighbors when removed, so one of 3 colors is always free. This is exactly the
   sparse-regime argument from the planted-3-coloring paper excerpt (Task 0a):
   for p <= c/n almost surely there is no subgraph with min degree >= 3, so this
   method succeeds almost surely and is a *guaranteed* legal 3-coloring whenever
   it completes.

2. dsatur_3color: DSATUR-ordered greedy restricted to a 3-color palette, with
   random restarts. Used as a fallback when the 3-core is non-empty (denser
   regime, near/above the phase transition), where no cheap guaranteed method is
   implemented here. This is a standard, well-established heuristic -- not a
   proof-backed algorithm like (1), and is reported separately so the two
   regimes aren't conflated.
"""
import numpy as np
import torch


def _to_numpy_adj(adj):
    if isinstance(adj, torch.Tensor):
        return adj.detach().cpu().numpy().astype(np.int8)
    return np.asarray(adj, dtype=np.int8)


def peel_3color(adj):
    """
    Returns (success, colors) where colors is an (n,) int array in {0,1,2} if
    success is True, else (False, None). Guaranteed-correct whenever it succeeds.
    """
    A = _to_numpy_adj(adj)
    n = A.shape[0]
    degree = A.sum(axis=1).astype(np.int64)
    alive = np.ones(n, dtype=bool)
    removal_order = []

    remaining = n
    while remaining > 0:
        candidates = np.where(alive & (degree <= 2))[0]
        if len(candidates) == 0:
            return False, None
        for v in candidates:
            if not alive[v]:
                continue
            alive[v] = False
            removal_order.append(v)
            remaining -= 1
            neighbors = np.where(A[v] == 1)[0]
            for u in neighbors:
                if alive[u]:
                    degree[u] -= 1

    colors = -np.ones(n, dtype=np.int64)
    for v in reversed(removal_order):
        neighbors = np.where(A[v] == 1)[0]
        used = set(colors[u] for u in neighbors if colors[u] >= 0)
        for c in range(3):
            if c not in used:
                colors[v] = c
                break
        else:
            return False, None  # should not happen given peel invariant

    return True, colors


def dsatur_3color(adj, restarts=20, rng=None):
    """
    DSATUR-ordered greedy coloring restricted to a 3-color palette, with random
    restarts (random tie-breaking + random restart order). Returns (success,
    colors) for the first restart that finds a legal 3-coloring, else
    (False, best_colors_by_min_conflicts).

    The hot loop is plain Python (lists/ints/bitmasks), not numpy: profiling a
    real slowdown (n=300, c=5: 4.5s/instance, almost entirely
    `saturation[v].sum()` and `degree[v]` numpy calls -- 1.35M calls for one
    instance) showed per-call numpy overhead, not asymptotic complexity, was
    the actual cost at this scale. Saturation is tracked as a 3-bit mask per
    vertex with an incrementally-maintained popcount, so no array is summed on
    every vertex-selection step.
    """
    A = _to_numpy_adj(adj)
    n = A.shape[0]
    if rng is None:
        rng = np.random.default_rng()

    neighbors_list = [np.where(A[i] == 1)[0].tolist() for i in range(n)]
    degree_list = [len(nb) for nb in neighbors_list]

    best_colors = None
    best_conflicts = None

    for _ in range(restarts):
        colors = [-1] * n
        sat_mask = [0] * n   # bit c set => color c forbidden by a colored neighbor
        sat_count = [0] * n  # popcount of sat_mask, maintained incrementally
        uncolored = set(range(n))

        while uncolored:
            # DSATUR: pick highest saturation degree, tie-break by degree, then random
            order = list(uncolored)
            rng.shuffle(order)
            best_v, best_sat, best_deg = -1, -1, -1
            for v in order:
                sat = sat_count[v]
                deg = degree_list[v]
                if sat > best_sat or (sat == best_sat and deg > best_deg):
                    best_v, best_sat, best_deg = v, sat, deg

            mask = sat_mask[best_v]
            available = [c for c in range(3) if not (mask & (1 << c))]
            if available:
                chosen = available[rng.integers(len(available))]
            else:
                # forced conflict: pick least-used color among neighbors
                nbr_colors = [colors[u] for u in neighbors_list[best_v]]
                counts = [nbr_colors.count(c) for c in range(3)]
                chosen = counts.index(min(counts))

            colors[best_v] = chosen
            uncolored.remove(best_v)
            bit = 1 << chosen
            for u in neighbors_list[best_v]:
                if u in uncolored and not (sat_mask[u] & bit):
                    sat_mask[u] |= bit
                    sat_count[u] += 1

        colors_arr = np.array(colors, dtype=np.int64)
        same = colors_arr[:, None] == colors_arr[None, :]
        conflicts = int((A.astype(bool) & same).sum() // 2)

        if conflicts == 0:
            return True, colors_arr
        if best_conflicts is None or conflicts < best_conflicts:
            best_conflicts = conflicts
            best_colors = colors_arr.copy()

    return False, best_colors


def classical_baseline_color(adj, dsatur_restarts=20, rng=None):
    """
    Try the guaranteed peel method first; fall back to DSATUR+restarts.
    Returns dict: success, colors, method ('peel' | 'dsatur'), conflicts.
    """
    ok, colors = peel_3color(adj)
    if ok:
        return {"success": True, "colors": colors, "method": "peel", "conflicts": 0}

    ok, colors = dsatur_3color(adj, restarts=dsatur_restarts, rng=rng)
    A = _to_numpy_adj(adj)
    if colors is not None:
        same = colors[:, None] == colors[None, :]
        conflicts = int((A.astype(bool) & same).sum() // 2)
    else:
        conflicts = -1
    return {"success": ok, "colors": colors, "method": "dsatur", "conflicts": conflicts}
