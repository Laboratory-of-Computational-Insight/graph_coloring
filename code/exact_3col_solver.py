"""
Exact 3-coloring solver (backtracking + forward checking + MRV + color-symmetry
breaking). Used to *verify* colorability/uncolorability rather than just trust a
single broken partition -- see TASKS.md note on generating literature-correct
adversarial training pairs (a single edge added to a planted-colorable graph
breaks *that* partition but does not by itself prove the graph has no other
3-coloring; this solver proves it either way, or reports UNKNOWN under budget).

Standard CSP backtracking:
  - MRV (minimum remaining values): branch on the uncolored vertex with the
    smallest current domain, tie-break on highest degree.
  - Forward checking: when a vertex is colored, remove that color from
    uncolored neighbors' domains; fail immediately if any domain empties.
  - Color-symmetry breaking: only allow introducing a brand-new color (one not
    yet used anywhere in the partial assignment) if it is the smallest unused
    color index. Cuts the branching factor by up to 3! without affecting the
    yes/no answer.
"""
import time

import numpy as np


class Budget:
    def __init__(self, node_budget=None, time_budget=None):
        self.node_budget = node_budget
        self.time_budget = time_budget
        self.nodes = 0
        self.t0 = time.time()

    def tick(self):
        self.nodes += 1
        if self.node_budget is not None and self.nodes > self.node_budget:
            return False
        if self.time_budget is not None and (time.time() - self.t0) > self.time_budget:
            return False
        return True


def solve_3coloring(adj, k=3, node_budget=200_000, time_budget=10.0):
    """
    Returns:
      - np.ndarray of shape (n,) with a legal k-coloring, if SAT
      - None if proven UNSAT (no k-coloring exists)
      - 'UNKNOWN' if the node/time budget was exhausted before resolving
    """
    if isinstance(adj, np.ndarray):
        A = adj
    else:
        A = adj.detach().cpu().numpy() if hasattr(adj, "detach") else np.asarray(adj)
    n = A.shape[0]
    neighbors = [np.where(A[i] != 0)[0].tolist() for i in range(n)]

    colors = [-1] * n
    domains = [set(range(k)) for _ in range(n)]
    num_used_colors = 0
    budget = Budget(node_budget, time_budget)

    def pick_vertex():
        best_v, best_domain_size, best_deg = -1, k + 1, -1
        for v in range(n):
            if colors[v] != -1:
                continue
            dsize = len(domains[v])
            deg = len(neighbors[v])
            if dsize < best_domain_size or (dsize == best_domain_size and deg > best_deg):
                best_v, best_domain_size, best_deg = v, dsize, deg
        return best_v

    def backtrack(num_colored):
        nonlocal num_used_colors
        if num_colored == n:
            return True
        if not budget.tick():
            return "UNKNOWN"

        v = pick_vertex()
        if len(domains[v]) == 0:
            return False

        candidate_colors = sorted(domains[v])
        for c in candidate_colors:
            if c >= num_used_colors + 1:
                # would introduce a new color that isn't the smallest unused one
                continue

            colors[v] = c
            introduced_new = c == num_used_colors
            if introduced_new:
                num_used_colors += 1

            removed = []
            consistent = True
            for u in neighbors[v]:
                if colors[u] == -1 and c in domains[u]:
                    domains[u].discard(c)
                    removed.append(u)
                    if len(domains[u]) == 0:
                        consistent = False
                        break

            result = False
            if consistent:
                result = backtrack(num_colored + 1)

            if result is True:
                # Success: leave colors[v] = c in place all the way up the stack.
                for u in removed:
                    domains[u].add(c)
                return True

            for u in removed:
                domains[u].add(c)
            colors[v] = -1
            if introduced_new:
                num_used_colors -= 1

            if result == "UNKNOWN":
                return "UNKNOWN"

        return False

    result = backtrack(0)
    if result is True:
        return np.array(colors, dtype=np.int64)
    if result == "UNKNOWN":
        return "UNKNOWN"
    return None
