"""
Idea #5 (2026-07-18): the FULL, exact subset-tracking SP for q=3 coloring --
fixes the specific approximation diagnosed in survey_propagation_coloring.py
(which assumed per-color independence of veto events across s=1,2,3). That
assumption is wrong: a single neighbor can only be frozen to ONE color, so
"color 1 is vetoed" and "color 2 is vetoed" are correlated through shared
neighbor identity, not independent.

Correct combination: each neighbor k contributes a 4-state categorical
distribution over {frozen-1, frozen-2, frozen-3, joker}. To combine N
independent neighbors into the joint distribution over "which subset of
colors got vetoed by at least one neighbor" (8 possible veto-patterns, since
each color is independently vetoed-or-not at the PATTERN level, even though
per-neighbor contributions are correlated), do a "sticky-OR" convolution:
maintain a distribution over the 8 veto-patterns, and for each neighbor,
update by OR-ing in whichever single color bit that neighbor's frozen-state
contributes (or no change if joker). This is an exact, not approximate,
combination -- no independence assumption needed across colors.

For the leave-one-out (cavity) message to each neighbor j, use prefix/suffix
convolution over the neighbor list (standard trick, O(degree) not
O(degree^2) per vertex) since sticky-OR convolution is associative but not
simply "divided out" the way log-linear products were in the simpler BP/SP.
"""
import numpy as np

VETO_PATTERNS = [(0, 0, 0), (1, 0, 0), (0, 1, 0), (0, 0, 1),
                 (1, 1, 0), (1, 0, 1), (0, 1, 1), (1, 1, 1)]
PATTERN_INDEX = {p: i for i, p in enumerate(VETO_PATTERNS)}


def _sticky_or_convolve(dist_a, dist_b):
    """dist_a, dist_b: (8,) distributions over veto-patterns. Returns their
    sticky-OR convolution (the pattern resulting from combining two
    independent sets of vetoes is the bitwise OR of the two patterns)."""
    out = np.zeros(8)
    for ia, pa in enumerate(VETO_PATTERNS):
        if dist_a[ia] == 0:
            continue
        for ib, pb in enumerate(VETO_PATTERNS):
            if dist_b[ib] == 0:
                continue
            combined = tuple(max(x, y) for x, y in zip(pa, pb))
            out[PATTERN_INDEX[combined]] += dist_a[ia] * dist_b[ib]
    return out


def _neighbor_state_to_pattern_dist(frozen_probs):
    """frozen_probs: (3,) [P(frozen to 1), P(frozen to 2), P(frozen to 3)].
    Returns (8,) distribution over veto-patterns contributed by ONE neighbor
    (either frozen to a single color -> vetoes exactly that color, or joker
    -> vetoes nothing)."""
    out = np.zeros(8)
    out[PATTERN_INDEX[(0, 0, 0)]] = max(0.0, 1 - frozen_probs.sum())  # joker
    out[PATTERN_INDEX[(1, 0, 0)]] = frozen_probs[0]
    out[PATTERN_INDEX[(0, 1, 0)]] = frozen_probs[1]
    out[PATTERN_INDEX[(0, 0, 1)]] = frozen_probs[2]
    return out


def _pattern_dist_to_frozen(pattern_dist):
    """Marginalize an 8-state veto-pattern distribution down to
    (P(frozen to 1), P(frozen to 2), P(frozen to 3)) for use as this vertex's
    OWN contribution to its other neighbors -- achievable set = complement of
    veto pattern; frozen-to-s means achievable set is exactly {s}, i.e. veto
    pattern vetoes exactly the OTHER two colors."""
    frozen = np.zeros(3)
    frozen[0] = pattern_dist[PATTERN_INDEX[(0, 1, 1)]]  # only color 1 achievable
    frozen[1] = pattern_dist[PATTERN_INDEX[(1, 0, 1)]]  # only color 2 achievable
    frozen[2] = pattern_dist[PATTERN_INDEX[(1, 1, 0)]]  # only color 3 achievable
    return frozen


def _prefix_suffix_leave_one_out(neighbor_pattern_dists):
    """neighbor_pattern_dists: list of (8,) arrays, one per neighbor. Returns
    a list of (8,) leave-one-out convolutions (excluding each index in turn),
    computed via prefix/suffix products in O(degree) convolutions total."""
    deg = len(neighbor_pattern_dists)
    identity = np.zeros(8)
    identity[0] = 1.0  # "no vetoes" pattern

    prefix = [identity] * (deg + 1)
    for i in range(deg):
        prefix[i + 1] = _sticky_or_convolve(prefix[i], neighbor_pattern_dists[i])

    suffix = [identity] * (deg + 1)
    for i in range(deg - 1, -1, -1):
        suffix[i] = _sticky_or_convolve(neighbor_pattern_dists[i], suffix[i + 1])

    return [_sticky_or_convolve(prefix[i], suffix[i + 1]) for i in range(deg)], prefix[deg]


def sp_full_reinforced_coloring(A, q=3, max_iter=300, damping=0.5, rho=0.1,
                                 seed=None, eps=1e-9):
    assert q == 3
    rng = np.random.default_rng(seed)
    n = A.shape[0]
    neighbors = [np.where(A[v] == 1)[0] for v in range(n)]

    # per-directed-edge message: frozen-prob (3,) for edge (i -> j)
    msg = {}
    for i in range(n):
        for j in neighbors[i]:
            m = rng.uniform(0.0, 0.1, size=3) + rng.normal(scale=0.01, size=3)
            msg[(i, j)] = np.clip(m, 1e-4, 0.9)

    belief_frozen = np.full((n, 3), 1.0 / 4)

    for t in range(1, max_iter + 1):
        new_msg = {}
        new_belief = np.zeros((n, 3))
        for i in range(n):
            nbrs = neighbors[i]
            deg = len(nbrs)
            if deg == 0:
                continue
            pattern_dists = [_neighbor_state_to_pattern_dist(msg[(k, i)]) for k in nbrs]
            leave_one_out, full_conv = _prefix_suffix_leave_one_out(pattern_dists)

            new_belief[i] = _pattern_dist_to_frozen(full_conv)  # unnormalized frozen mass (rest is joker/contradiction)

            for idx, j in enumerate(nbrs):
                cavity_frozen = _pattern_dist_to_frozen(leave_one_out[idx])
                # reinforcement: extra virtual neighbor, frozen-state prob
                # proportional to i's own current belief, growing with t
                reinforce_strength = min(1.0, rho * t)
                virtual = belief_frozen[i] * reinforce_strength
                virtual_pattern = _neighbor_state_to_pattern_dist(virtual)
                reinforced = _sticky_or_convolve(leave_one_out[idx], virtual_pattern)
                cavity_frozen_reinforced = _pattern_dist_to_frozen(reinforced)
                new_msg[(i, j)] = np.clip(cavity_frozen_reinforced, 1e-4, 0.9)

        diff = 0.0
        for key in new_msg:
            damped = damping * msg[key] + (1 - damping) * new_msg[key]
            diff = max(diff, np.abs(damped - msg[key]).max())
            msg[key] = damped
        belief_frozen = damping * belief_frozen + (1 - damping) * new_belief

    return belief_frozen, False
