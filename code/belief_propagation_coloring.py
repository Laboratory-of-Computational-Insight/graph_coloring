"""
Belief propagation for random-graph q-coloring, with the standard tricks from
the statistical-physics literature (Mulet-Pagnani-Weigt-Zecchina 2002;
Braunstein-Zecchina 2004 reinforcement trick; Zdeborova-Krzakala 2007 for
coloring specifically) used to make BP converge to a useful, decisive fixed
point in the clustered/hard phase instead of oscillating or sitting at the
symmetric uniform fixed point:

  - cavity (leave-one-out) message passing: psi_{i->j}(s) ~ prod_{k in d(i)\\j} (1 - psi_{k->i}(s))
  - reinforcement: a growing self-field toward each node's own running belief,
    which smoothly decimates the system to a single configuration without
    needing explicit discrete fix-and-resolve steps
  - damping for numerical stability
  - small random symmetry-breaking noise at init (uniform messages are an
    unstable fixed point of the symmetric equations -- this project has hit
    that exact 1/3-1/3-1/3 trap before in the NN-based approaches)
  - multiple independent random restarts, keep the most decisive run
"""
import numpy as np


def _build_directed_edges(A):
    n = A.shape[0]
    iu, ju = np.nonzero(np.triu(A, k=1))
    src = np.concatenate([iu, ju])
    dst = np.concatenate([ju, iu])
    m2 = len(src)
    half = m2 // 2
    reverse = np.concatenate([np.arange(half, m2), np.arange(0, half)])
    return src, dst, reverse


def bp_reinforced_coloring(A, q=3, max_iter=3000, damping=0.5, rho=0.003,
                            tol=1e-7, seed=None, eps=1e-12,
                            init_bias=None, bias_strength=0.3):
    """Returns (belief (n,q), converged_bool). belief[i] is BP's marginal
    color distribution for vertex i after reinforcement-driven decimation.

    init_bias (optional, shape (n,) int in [0,q)): a per-vertex classical
    guess (e.g. Bethe-Hessian spectral init) to seed psi toward instead of
    pure noise. psi[e] represents "belief that src[e] has color s", so the
    bias is added at psi[e][init_bias[src[e]]] for every directed edge e --
    biasing each vertex's OUTGOING messages toward its own classical guess.
    bias_strength controls how strong the nudge is relative to the ~1/q
    uniform baseline (0 = no bias, identical to the original unbiased init).
    """
    rng = np.random.default_rng(seed)
    n = A.shape[0]
    src, dst, reverse = _build_directed_edges(A)
    m2 = len(src)

    psi = rng.uniform(0.9, 1.1, size=(m2, q)) * (1.0 / q)
    psi += rng.normal(scale=0.02, size=(m2, q))
    if init_bias is not None:
        bias_onehot = np.zeros((n, q))
        bias_onehot[np.arange(n), init_bias] = 1.0
        psi += bias_strength * bias_onehot[src]
    psi = np.clip(psi, 1e-6, None)
    psi /= psi.sum(axis=1, keepdims=True)

    prev_belief_log = np.full((n, q), -np.log(q))

    for t in range(1, max_iter + 1):
        logterm = np.log(np.clip(1.0 - psi, eps, None))  # (m2, q)

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

        if diff < tol:
            return np.exp(prev_belief_log), True

    return np.exp(prev_belief_log), False


def _logsumexp(x, axis=None, keepdims=False):
    m = np.max(x, axis=axis, keepdims=True)
    out = m + np.log(np.sum(np.exp(x - m), axis=axis, keepdims=True))
    return out if keepdims else np.squeeze(out, axis=axis)


def bp_best_of_restarts(A, q=3, n_restarts=5, **kwargs):
    """Run several independent random-seeded BP-reinforcement chains, keep the
    most decisive one (highest mean max-belief -- i.e. the most confident /
    fully decimated run), a standard practical trick alongside damping and
    reinforcement itself."""
    best_belief = None
    best_score = -1.0
    for s in range(n_restarts):
        belief, converged = bp_reinforced_coloring(A, q=q, seed=s, **kwargs)
        score = belief.max(axis=1).mean()
        if score > best_score:
            best_score = score
            best_belief = belief
    return best_belief
