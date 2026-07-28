"""
Three ideas combining Alon-Kahale (AK) and belief propagation (BP), per user
request (2026-07-17):

1. AK with random restarts -- AK's phase 1 (spectral init) is fully
   deterministic (exact top-2 eigenvectors + exhaustive angle search), so
   "random init" has to come from perturbing the matrix slightly before
   eigendecomposition (small random noise -> different eigenvector alignment
   in near-degenerate directions), retried a few times.
2. AK's phase-3 "pruning" (support-threshold + exact backtracking on the
   small residual) applied to BP's raw output, instead of greedy_repair --
   reuses spectral_coloring.py's `_cleanup` on any coloring, not just AK's own.
3. BP as phase-1 init for AK -- skip AK's spectral init, feed BP's own
   argmax coloring into AK's phases 2 (propagate) + 3 (cleanup).
"""
import numpy as np
import torch

from spectral_coloring import _to_numpy_adj, _spectral_init, _propagate, _cleanup
from belief_propagation_coloring import bp_reinforced_coloring
from repair_utils import count_conflicts, greedy_repair

RHO_VALUES = [0.001, 0.003, 0.01, 0.03, 0.1]
DAMPING_VALUES = [0.3, 0.5, 0.7]
HYPER_COMBOS = [(r, d) for r in RHO_VALUES for d in DAMPING_VALUES]


def spectral_seeded_bp_ak(adj, bias_strength=0.3, max_component_size=18, cap_total=30, seed_base=0):
    """Full AK-style pipeline with BP swapped in for the middle phase:
    preprocessing = AK's own phase-1 spectral init (exact eigenvector guess,
    same as plain alon_kahale_3color); middle = BP (escalating plain ->
    reinforced -> hyperparameter-diverse) with its init psi biased toward
    that spectral guess (see belief_propagation_coloring.py's init_bias);
    postprocessing = AK's phase-3 cleanup (support-threshold + exact
    backtracking / DSATUR fallback) applied to BP's output, with
    greedy_repair as a final safety net if cleanup still leaves conflicts.
    Returns (colors, n_conflicts, success)."""
    A = _to_numpy_adj(adj)
    n = A.shape[0]

    spectral_guess = _spectral_init(A, n)

    per_rung = cap_total // 3
    plans = ([(0.0, 0.5, seed_base * 100000 + i) for i in range(per_rung)] +
             [(0.003, 0.5, seed_base * 100000 + per_rung + i) for i in range(per_rung)] +
             [(HYPER_COMBOS[i % len(HYPER_COMBOS)][0], HYPER_COMBOS[i % len(HYPER_COMBOS)][1],
               seed_base * 100000 + 2 * per_rung + i) for i in range(cap_total - 2 * per_rung)])

    best_colors, best_conf = None, np.inf
    for rho, damping, seed in plans:
        belief, _ = bp_reinforced_coloring(A, rho=rho, damping=damping, max_iter=3000, seed=seed,
                                            init_bias=spectral_guess, bias_strength=bias_strength)
        colors = belief.argmax(axis=1)
        ok, colors, method = _cleanup(A, colors, max_component_size=max_component_size)
        conf = count_conflicts(A, colors)
        if conf > 0:
            colors = greedy_repair(A, colors)
            conf = count_conflicts(A, colors)
        if conf < best_conf:
            best_conf, best_colors = conf, colors
        if conf == 0:
            return best_colors, 0, True
    return best_colors, best_conf, False


def ak_random_restart(adj, n_restarts=5, noise_scale=0.05, propagation_iterations=None,
                       max_component_size=18, seed=None):
    """Idea 1: perturb the adjacency matrix with small symmetric random noise
    before phase 1's eigendecomposition, retry n_restarts times, keep the best."""
    rng = np.random.default_rng(seed)
    A = _to_numpy_adj(adj)
    n = A.shape[0]
    if propagation_iterations is None:
        propagation_iterations = max(1, int(np.ceil(np.log(max(n, 2)))))

    best_colors, best_conf = None, np.inf
    for r in range(n_restarts):
        if r == 0:
            A_noisy = A
        else:
            noise = rng.normal(scale=noise_scale, size=A.shape)
            noise = (noise + noise.T) / 2
            A_noisy = A + noise * (A + A.T > 0)  # perturb existing edge weights only, keep sparsity pattern

        colors = _spectral_init(A_noisy, n)
        colors = _propagate(A, colors, propagation_iterations)  # propagate on the TRUE graph
        ok, colors, method = _cleanup(A, colors, max_component_size=max_component_size)
        conf = count_conflicts(A, colors)
        if conf < best_conf:
            best_conf = conf
            best_colors = colors
        if conf == 0:
            break
    return best_colors, best_conf


def bp_then_ak_cleanup(adj, use_ak_cleanup=True, bp_seed=0, rho=0.0, damping=0.5,
                        propagation_iterations=None, max_component_size=18):
    """Idea 2 + 3: BP produces an initial coloring (replacing AK's spectral
    phase 1 entirely, not biasing it); feed it into AK's phase 2 (propagate)
    + phase 3 (cleanup/peeling) exactly as AK would run them on its own
    spectral guess."""
    A = _to_numpy_adj(adj)
    n = A.shape[0]
    if propagation_iterations is None:
        propagation_iterations = max(1, int(np.ceil(np.log(max(n, 2)))))

    belief, _ = bp_reinforced_coloring(A, rho=rho, damping=damping, seed=bp_seed, max_iter=3000)
    colors = belief.argmax(axis=1)

    if use_ak_cleanup:
        colors = _propagate(A, colors, propagation_iterations)
        ok, colors, method = _cleanup(A, colors, max_component_size=max_component_size)
    return colors, count_conflicts(A, colors)


def bp_then_ak_cleanup_escalating(adj, cap_total=30, max_component_size=18):
    """bp_then_ak_cleanup with the same plain -> reinforced -> hyperdiverse
    restart ladder used elsewhere in this project, OR-combined on first
    zero-conflict result, plus greedy_repair as a final safety net."""
    per_rung = cap_total // 3
    plans = ([(0.0, 0.5, i) for i in range(per_rung)] +
             [(0.003, 0.5, per_rung + i) for i in range(per_rung)] +
             [(HYPER_COMBOS[i % len(HYPER_COMBOS)][0], HYPER_COMBOS[i % len(HYPER_COMBOS)][1],
               2 * per_rung + i) for i in range(cap_total - 2 * per_rung)])

    A = _to_numpy_adj(adj)
    best_colors, best_conf = None, np.inf
    for rho, damping, seed in plans:
        colors, conf = bp_then_ak_cleanup(adj, rho=rho, damping=damping, bp_seed=seed,
                                           max_component_size=max_component_size)
        if conf > 0:
            colors = greedy_repair(A, colors)
            conf = count_conflicts(A, colors)
        if conf < best_conf:
            best_conf, best_colors = conf, colors
        if conf == 0:
            return best_colors, 0, True
    return best_colors, best_conf, False


if __name__ == "__main__":
    import sys
    from belief_propagation_coloring import bp_best_of_restarts

    RESULTS_DIR = "G:/graph_col/graph_coloring/results"
    c_list = [int(x) for x in sys.argv[1:]] if len(sys.argv) > 1 else [5, 6]
    n_inst = 20

    for c in c_list:
        counts = {"bp_ak_cleanup": 0, "bp_greedy_repair": 0, "bp_both_or": 0}
        for i in range(n_inst):
            d = torch.load(f"{RESULTS_DIR}/shared_test_instances/c{c}_inst{i}.pt")
            adj = d["adj"]
            A = adj.numpy()
            n = A.shape[0]
            prop_iters = max(1, int(np.ceil(np.log(max(n, 2)))))

            belief = bp_best_of_restarts(A, n_restarts=3, max_iter=3000)
            colors_raw = belief.argmax(axis=1)
            conf_raw = count_conflicts(A, colors_raw)

            if conf_raw == 0:
                conf_ak = conf_gr = 0
            else:
                colors_ak = _propagate(A, colors_raw.copy(), prop_iters)
                ok, colors_ak, method = _cleanup(A, colors_ak, max_component_size=18)
                conf_ak = count_conflicts(A, colors_ak)

                colors_gr = greedy_repair(A, colors_raw)
                conf_gr = count_conflicts(A, colors_gr)

            counts["bp_ak_cleanup"] += (conf_ak == 0)
            counts["bp_greedy_repair"] += (conf_gr == 0)
            counts["bp_both_or"] += (conf_ak == 0 or conf_gr == 0)

            print(f"c={c} inst={i}: bp_raw_conf={conf_raw} +ak_cleanup_conf={conf_ak} +greedy_repair_conf={conf_gr}")

        print(f"\n=== c={c}: BP+AK_cleanup={counts['bp_ak_cleanup']}/{n_inst}  "
              f"BP+greedy_repair={counts['bp_greedy_repair']}/{n_inst}  "
              f"OR={counts['bp_both_or']}/{n_inst} ===\n")
