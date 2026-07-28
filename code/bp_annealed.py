"""
Idea #2 (2026-07-18): annealed BP for c=4.69. The existing BP is the
zero-temperature ("hard-constraint") limit: cavity term is log(1 - psi(s)),
i.e. a neighbor with color s completely forbids s. The finite-temperature
version replaces the hard 0 weight for "same color" with a soft Boltzmann
factor exp(-beta): term(s) = 1 - psi(s)*(1 - exp(-beta)). At beta->infinity
this recovers the current hard update; at beta=0 there's no constraint at
all (uniform mixing). Start at a small beta (easy to converge, escapes bad
basins) and ramp up to a large beta over iterations, instead of jumping
straight to the hard limit -- a standard simulated-annealing-style trick for
exactly the hardest, most marginal regime (the true phase transition).
Combined with the existing reinforcement mechanism (kept as-is).
"""
import numpy as np

from belief_propagation_coloring import _build_directed_edges, _logsumexp


def bp_annealed_reinforced_coloring(A, q=3, max_iter=3000, damping=0.5, rho=0.003,
                                     beta_start=0.5, beta_end=30.0, anneal_iters=1500,
                                     tol=1e-7, seed=None, eps=1e-12):
    rng = np.random.default_rng(seed)
    n = A.shape[0]
    src, dst, reverse = _build_directed_edges(A)
    m2 = len(src)

    psi = rng.uniform(0.9, 1.1, size=(m2, q)) * (1.0 / q)
    psi += rng.normal(scale=0.02, size=(m2, q))
    psi = np.clip(psi, 1e-6, None)
    psi /= psi.sum(axis=1, keepdims=True)

    prev_belief_log = np.full((n, q), -np.log(q))

    for t in range(1, max_iter + 1):
        beta = beta_start + (beta_end - beta_start) * min(1.0, t / anneal_iters)
        soft_factor = 1.0 - np.exp(-beta)  # -> 1 as beta grows (hard limit)
        soft_term = np.clip(1.0 - psi * soft_factor, eps, None)
        logterm = np.log(soft_term)  # (m2, q)

        S = np.zeros((n, q))
        np.add.at(S, dst, logterm)

        belief_log = S - _logsumexp(S, axis=1, keepdims=True)

        reinforce = rho * t * belief_log
        new_log_msg = S[src] - logterm[reverse] + reinforce[src]
        new_log_msg -= _logsumexp(new_log_msg, axis=1, keepdims=True)
        new_msg = np.exp(new_log_msg)

        psi_damped = damping * psi + (1 - damping) * new_msg
        psi_damped /= psi_damped.sum(axis=1, keepdims=True)

        diff = np.abs(psi_damped - psi).max()
        psi = psi_damped
        prev_belief_log = belief_log

        if diff < tol and t > anneal_iters:  # don't allow early exit mid-anneal
            return np.exp(prev_belief_log), True

    return np.exp(prev_belief_log), False


def bp_annealed_best_of_restarts(A, q=3, n_restarts=5, **kwargs):
    best_belief = None
    best_score = -1.0
    for s in range(n_restarts):
        belief, _ = bp_annealed_reinforced_coloring(A, q=q, seed=s, **kwargs)
        score = belief.max(axis=1).mean()
        if score > best_score:
            best_score = score
            best_belief = belief
    return best_belief
