"""
Survey propagation (zero-temperature / hard-constraint) for random-graph
q-coloring, following Mulet, Pagnani, Weigt & Zecchina, "Coloring random
graphs" (PRL 2002) -- the actual SP formalism, not plain BP.

The key difference from belief_propagation_coloring.py: BP's message is a
soft marginal ("probability i has color s"), which conflates "i is frozen to
s across the whole cluster" with "i merely leans toward s" -- this is exactly
why plain BP over/under-counts correlations once the solution space shatters
into clusters (the regime SP was invented for). SP's message instead tracks,
per directed edge, the probability that vertex i is FROZEN specifically to
color s in the cavity graph without j -- only a fully-frozen neighbor can
actually force a color away from you; a neighbor with >=2 remaining options
can always pick a different one, so it sends no real constraint at all.

For q=3, a vertex's cavity "achievable color set" (assuming per-color
independence of veto events, the standard mean-field cavity approximation)
has 8 possible outcomes: frozen to {1}, {2}, or {3}; free among {1,2},
{1,3}, or {2,3} (lumped into a single "joker" probability, since a
partially-free neighbor sends no warning either way); or the empty set
(local contradiction -- signals this cluster/branch has no solution).

Update per directed edge (i -> j), given per-color "available" probabilities
p(s) = prod over cavity neighbors k of (1 - pi_{k->i}(s)):
  P(frozen to s)     = p(s) * prod_{s' != s} (1 - p(s'))
  P(contradiction)   = prod_s (1 - p(s))
  pi_{i->j}(s) = P(frozen to s) / (1 - P(contradiction))   [renormalize away
                                                             the contradiction
                                                             mass]

Reinforcement (same role as in BP): treat the growing self-field as an extra
virtual neighbor contributing its own (1 - belief_frozen(s))^(rho*t) factor
into the "available" product, exactly like any other cavity neighbor would.
"""
import numpy as np

from belief_propagation_coloring import _build_directed_edges


def _frozen_and_contradiction(p):
    """p: (..., 3) per-color 'available' probabilities. Returns
    (frozen (..., 3), contradiction (...,))."""
    p1, p2, p3 = p[..., 0], p[..., 1], p[..., 2]
    frozen = np.stack([
        p1 * (1 - p2) * (1 - p3),
        (1 - p1) * p2 * (1 - p3),
        (1 - p1) * (1 - p2) * p3,
    ], axis=-1)
    contradiction = (1 - p1) * (1 - p2) * (1 - p3)
    return frozen, contradiction


def sp_reinforced_coloring(A, q=3, max_iter=1000, damping=0.5, rho=0.1,
                            tol=1e-9, seed=None, eps=1e-9):
    """Returns (belief_frozen (n,3), converged_bool). belief_frozen[i,s] is
    SP's estimate of the probability vertex i is frozen to color s (using
    the full, non-cavity neighbor set)."""
    assert q == 3, "this implementation is specialized to q=3 (8-outcome enumeration)"
    rng = np.random.default_rng(seed)
    n = A.shape[0]
    src, dst, reverse = _build_directed_edges(A)
    m2 = len(src)

    pi = rng.uniform(0.0, 0.15, size=(m2, q))  # start with mostly "joker", small random frozen mass
    pi += rng.normal(scale=0.01, size=(m2, q))
    pi = np.clip(pi, 1e-4, 0.95)

    prev_belief_frozen = np.full((n, q), 1.0 / (q + 1))

    for t in range(1, max_iter + 1):
        log_avail_term = np.log(np.clip(1.0 - pi, eps, None))  # (m2, q)

        S = np.zeros((n, q))
        np.add.at(S, dst, log_avail_term)
        p_avail_full = np.exp(S)  # (n, q), full neighbor set

        belief_frozen, belief_contradiction = _frozen_and_contradiction(p_avail_full)
        belief_frozen = belief_frozen / np.clip(1 - belief_contradiction[:, None], eps, None)

        p_avail_cavity = np.exp(S[src] - log_avail_term[reverse])  # (m2, q), leave-one-out

        # reinforcement: an extra virtual "available" factor with growing
        # weight rho*t, proportional to how strongly i already favors color s
        # -- the favored color's availability decays slowest as t grows, so
        # it eventually dominates (NOT 1-belief_frozen, which would decay the
        # favored color fastest and push the system away from its own lean).
        reinforce_factor = np.power(
            np.clip(belief_frozen[src], eps, None), rho * t
        )
        p_avail_reinforced = p_avail_cavity * reinforce_factor

        frozen_new, contradiction_new = _frozen_and_contradiction(p_avail_reinforced)
        pi_new = frozen_new / np.clip(1 - contradiction_new[:, None], eps, None)
        pi_new = np.clip(pi_new, 1e-4, 0.95)

        pi_damped = damping * pi + (1 - damping) * pi_new

        diff = np.abs(pi_damped - pi).max()
        pi = pi_damped
        prev_belief_frozen = belief_frozen

        if diff < tol:
            return prev_belief_frozen, True

    return prev_belief_frozen, False


def sp_best_of_restarts(A, q=3, n_restarts=5, **kwargs):
    best_belief = None
    best_score = -1.0
    for s in range(n_restarts):
        belief, converged = sp_reinforced_coloring(A, q=q, seed=s, **kwargs)
        score = belief.max(axis=1).mean()
        if score > best_score:
            best_score = score
            best_belief = belief
    return best_belief
