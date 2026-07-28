"""
Alon & Kahale, "A spectral technique for coloring random 3-colorable graphs"
(SIAM J. Comput. 26(6), 1997). Polynomial-time algorithm that properly
3-colors G_{3n,p,3} with high probability whenever p >= c/n for a sufficiently
large constant c -- this is the paper behind the planted-3-coloring excerpt
that motivated the classical baseline, and directly implements its actual
algorithm (classical_baseline.py's dsatur_3color was only ever a stand-in).

Three phases, following the paper's Section 1.2 (with the tuned variant from
Section 4 for phase 3):

  1. Spectral init: trim edges incident to vertices of degree > 5d (d = mean
     degree) to get G'; take the two eigenvectors of G' with the smallest
     (most negative) eigenvalues; find a linear combination t of them with
     median 0, normalized to ||t||_2 = sqrt(2 * n_per_class); threshold at
     +-1/2 into three initial classes. This is the same 2D eigenvector
     embedding the AAAI paper found GNN-GCP spontaneously learns (INSIGHTS.md
     "triangular geometry").

  2. Propagation: for ~log(n) iterations, recolor each vertex to whichever
     color is currently least popular among its neighbors -- a cheap
     relaxation-labeling pass over the *original* (untrimmed) graph.

  3. Cleanup: uncolor any vertex with fewer than i neighbors of some other
     color, for the smallest threshold i that leaves brute-forceable
     connected components (this "support < i" criterion is exactly the
     support metric from Alon & Kahale 1994 already documented in
     INSIGHTS.md). Brute-force each small uncolored component, respecting
     the fixed colors of its already-colored neighbors. If no threshold
     keeps every component small enough, fall back to DSATUR on the
     residual -- a practical robustness choice for finite n, not part of
     the original theorem (which only claims success almost surely as
     n -> infinity).
"""
import networkx as nx
import numpy as np
import torch


def _to_numpy_adj(adj):
    if isinstance(adj, torch.Tensor):
        return adj.detach().cpu().numpy().astype(np.float64)
    return np.asarray(adj, dtype=np.float64)


def _two_eigvecs_to_coloring(e1, e2, n, angle_steps=720):
    """
    Shared by both spectral variants: given the two "informative" eigenvectors,
    search linear combinations t = cos(theta)*e1 + sin(theta)*e2 for the angle
    whose median is closest to 0 (the paper requires an exact linear
    combination, not an affine shift, so we search the angle rather than
    subtracting the median), normalize to ||t||_2 = sqrt(2*n/3), threshold at
    +-1/2 into three initial classes.
    """
    thetas = np.linspace(0, np.pi, angle_steps, endpoint=False)
    best_theta, best_abs_median = 0.0, np.inf
    for theta in thetas:
        t = np.cos(theta) * e1 + np.sin(theta) * e2
        med = np.median(t)
        if abs(med) < best_abs_median:
            best_abs_median = abs(med)
            best_theta = theta

    t = np.cos(best_theta) * e1 + np.sin(best_theta) * e2
    t = t - np.median(t)  # tiny correction so thresholds at +-1/2 are well-centered
    norm = np.linalg.norm(t)
    target_norm = np.sqrt(2.0 * n / 3.0)
    if norm > 1e-12:
        t = t * (target_norm / norm)

    colors = np.full(n, 2, dtype=np.int64)  # default: third class (|t| <= 1/2)
    colors[t > 0.5] = 0
    colors[t < -0.5] = 1
    return colors


def _spectral_init(A, num_vertices, angle_steps=720):
    """Phase 1 (Alon-Kahale): returns an initial (possibly illegal) coloring."""
    n = num_vertices
    degree = A.sum(axis=1)
    d = max(degree.mean(), 1e-6)

    # Trim edges incident to high-degree vertices (robustness against outliers).
    high_deg = degree > 5 * d
    A_trim = A.copy()
    A_trim[high_deg, :] = 0
    A_trim[:, high_deg] = 0

    # Two smallest (most negative) eigenvalues/eigenvectors of the trimmed adjacency.
    eigvals, eigvecs = np.linalg.eigh(A_trim)  # ascending order
    e1, e2 = eigvecs[:, 0], eigvecs[:, 1]
    return _two_eigvecs_to_coloring(e1, e2, n, angle_steps=angle_steps)


def _bethe_hessian_init(A, num_vertices, angle_steps=720, r=None):
    """
    Phase 1 (non-backtracking-inspired): replaces the raw adjacency matrix
    with the Bethe Hessian H(r) = (r^2-1)*I - r*A + D (Saade, Krzakala &
    Zdeborova 2014), whose spectrum tracks the full non-backtracking matrix's
    but costs the same as a plain n x n eigendecomposition. The point: naive
    adjacency-spectral methods need a "sufficiently large" average degree
    because at low degree the two informative eigenvectors get swamped by
    noise from high-degree vertices and short cycles; the D term here
    down-weights exactly that noise directly, without the ad hoc 5d trimming
    _spectral_init needs. Recovers signal closer to the true (Kesten-Stigum)
    threshold than plain adjacency spectral clustering can.

    Sign note: Saade/Krzakala/Zdeborova's r=+sqrt(mean_degree) targets
    *assortative* communities (denser within than between). A proper coloring
    is the opposite structure -- disassortative, zero edges within a class --
    which is exactly why Alon-Kahale's phase 1 reads off the *smallest*
    (most negative) eigenvalues of the plain adjacency matrix rather than the
    largest. Empirically verified (agreement with the true planted partition,
    n=1000): r=+sqrt(d) gives ~chance-level accuracy (0.34-0.36) at every c
    tested; r=-sqrt(d) gives 0.46/0.64/0.80 at c=5/8/15, beating plain
    adjacency (0.41/0.54/0.77) at every one. Default r is therefore negative.

    The most negative eigenvalues of H(r) carry the community structure (H(r)
    is built so informative directions are negative, near-zero for noise) --
    take the two smallest, same downstream read-out as the adjacency variant.
    """
    n = num_vertices
    degree = A.sum(axis=1)
    if r is None:
        r = -np.sqrt(max(degree.mean(), 1e-6))

    H = (r * r - 1.0) * np.eye(n) - r * A + np.diag(degree)
    eigvals, eigvecs = np.linalg.eigh(H)  # ascending order
    e1, e2 = eigvecs[:, 0], eigvecs[:, 1]
    return _two_eigvecs_to_coloring(e1, e2, n, angle_steps=angle_steps)


def _non_backtracking_vertex_eigvecs(A, num_vertices, imag_tol=1e-6):
    """
    Shared by _non_backtracking_init and raw_spectral_coordinates: builds the
    true non-backtracking (NB) operator's 2n x 2n companion linearization and
    returns its two "informative" vertex-space eigenvectors as a real (n, 2)
    array, plus how many of the 2n eigenvalues came out numerically real
    (diagnostic, also useful for the NaN/complex-truncation spot-check).

    Theory (Krzakala, Moore, Mossel, Neeman, Sly, Zdeborova & Zhang 2013,
    "Spectral redemption in clustering sparse networks"): the (2m x 2m)
    directed-edge NB matrix B is expensive to build/eigendecompose directly,
    but its non-trivial spectrum equals that of the much smaller

        B' = [[0, D - I], [-I, A]]      (2n x 2n, D = diag degree, A = adjacency)

    B' is real but not symmetric (the -I block breaks symmetry), so we use
    np.linalg.eig (general eigendecomposition; eigenvalues/eigenvectors are
    complex in general) rather than eigh.

    Vertex-space eigenvector -- verified, NOT the naive "first n components":
    writing an eigenpair of B' as (lambda, z) with z = (x; y) split into its
    first-n and second-n blocks, the defining equations of B' give
    x = (D-I) y / lambda and x = (A - lambda I) y simultaneously, which forces

        lambda^2 * y - lambda * A * y + (D - I) * y = 0
        <=>  H(lambda) y = 0   where H(r) = (r^2-1)*I - r*A + D

    i.e. y (the SECOND block) is exactly a null vector of the Bethe Hessian
    _bethe_hessian_init already uses, at r = lambda -- the Ihara-Bass identity
    relating the NB operator's spectrum to the Bethe Hessian. x = (A - lambda
    I) y is a linear transform of y and empirically carries a noisier signal.
    Confirmed empirically (n=1000, accuracy vs. true planted partition, best
    color permutation): the second block (y) beats the first block (x) at
    every c in {5,...,10} (e.g. c=8: y=0.611 vs x=0.560; c=10: y=0.660 vs
    x=0.595) and tracks _bethe_hessian_init closely, as the identity above
    predicts. This module therefore reads off y, overriding the naive
    "first n components" reading of the companion form.

    Eigenvalue selection: for the same reason _spectral_init reads the
    smallest (most negative) adjacency eigenvalues and _bethe_hessian_init
    uses r<0 -- proper coloring is a disassortative structure -- we take the
    two most negative real eigenvalues of B'. "Real" is not automatic: most
    of B''s 2n eigenvalues are genuine complex-conjugate pairs (the "bulk"),
    and at low mean degree (c=5 in testing) a complex pair's real part can
    be more negative than the second-most-negative *real* eigenvalue, so
    sorting all 2n eigenvalues by raw real part and taking the top two can
    accidentally grab one eigenvalue out of a complex pair. Fix: filter to
    eigenvalues with |imag(lambda)| < imag_tol first (in every instance
    tested, n=1000, c=5..14, genuinely real eigenvalues came back with
    imaginary part exactly 0.0 or ~1e-12 float noise, while genuine complex
    pairs had |imag/real| ~0.1-1 -- imag_tol=1e-6 cleanly separates the two),
    then take the two most negative among that filtered set. If fewer than 2
    eigenvalues pass the filter (not observed at n=1000, c>=5, but a possible
    degenerate small-graph edge case), fall back to the two most negative by
    real part over the full unfiltered spectrum rather than raising.
    """
    n = num_vertices
    degree = A.sum(axis=1)
    D = np.diag(degree)
    I = np.eye(n)
    Bp = np.block([[np.zeros((n, n)), D - I], [-I, A]])

    eigvals, eigvecs = np.linalg.eig(Bp)
    real_mask = np.abs(eigvals.imag) < imag_tol
    n_real = int(real_mask.sum())
    candidates = np.where(real_mask)[0] if n_real >= 2 else np.arange(len(eigvals))
    candidates = candidates[np.argsort(eigvals[candidates].real)]  # ascending
    idx1, idx2 = candidates[0], candidates[1]

    # Second block (y) is the vertex-space eigenvector -- see docstring above.
    y1 = eigvecs[n:, idx1].real
    y2 = eigvecs[n:, idx2].real
    return y1, y2, n_real


def _non_backtracking_init(A, num_vertices, angle_steps=720):
    """
    Phase 1 (true non-backtracking): same read-out as _spectral_init /
    _bethe_hessian_init, but the two eigenvectors come from the exact
    non-backtracking operator's companion linearization rather than plain
    adjacency or the Bethe Hessian's scalar approximation to it. See
    _non_backtracking_vertex_eigvecs's docstring for the construction, the
    (verified, non-obvious) second-block vertex-eigenvector convention, and
    the eigenvalue-selection rule.

    Empirically (n=1000, accuracy vs. true planted partition, best color
    permutation, 8 instances/c, c=5..10): non_backtracking beats plain
    adjacency at every c (e.g. c=5: 0.405 vs 0.448; c=8: 0.587 vs 0.591;
    c=10: 0.634 vs 0.675) but tracks bethe_hessian closely without clearly
    beating it -- bethe_hessian is at or slightly above non_backtracking at
    every c tested (c=5: 0.476 vs 0.448; c=7: 0.578 vs 0.573, nearly tied;
    c=9: 0.652 vs 0.636; c=10: 0.689 vs 0.675). The exact non-backtracking
    operator does not show the hoped-for edge over the cheap Bethe-Hessian
    scalar approximation in this finite-n (n=1000) regime -- a legitimate,
    non-obvious negative result given the ~6.2s/call cost (vs. ~milliseconds
    for the n x n eigh-based variants), so bethe_hessian remains the better
    cost/accuracy choice for this project's hard window (c=5-10) unless a
    later, larger-n test finds the gap opening up.
    """
    n = num_vertices
    e1, e2, _ = _non_backtracking_vertex_eigvecs(A, n)
    return _two_eigvecs_to_coloring(e1, e2, n, angle_steps=angle_steps)


def raw_spectral_coordinates(adj, variant="adjacency", r=None):
    """
    Returns the raw (n, 2) continuous coordinates -- the two informative
    eigenvectors themselves, *before* the angle-search + threshold step that
    turns them into a discrete coloring. Exposed for use as an input feature
    elsewhere (e.g. spectral_nn.py's embedder init): a downstream trainable
    network can learn its own best combination of these two directions, so
    there's no need to run the angle search just to hand it a feature vector.
    """
    A = adj.detach().cpu().numpy() if isinstance(adj, torch.Tensor) else np.asarray(adj, dtype=np.float64)
    n = A.shape[0]
    degree = A.sum(axis=1)

    if variant == "non_backtracking":
        # Not an eigh-on-M path like the other variants -- see
        # _non_backtracking_vertex_eigvecs for the companion-matrix
        # construction and the (verified) second-block read-out convention.
        e1, e2, _ = _non_backtracking_vertex_eigvecs(A, n)
        return np.stack([e1, e2], axis=1)  # [n, 2]

    if variant == "bethe_hessian":
        rr = r if r is not None else -np.sqrt(max(degree.mean(), 1e-6))
        M = (rr * rr - 1.0) * np.eye(n) - rr * A + np.diag(degree)
    else:
        d = max(degree.mean(), 1e-6)
        high_deg = degree > 5 * d
        M = A.copy()
        M[high_deg, :] = 0
        M[:, high_deg] = 0

    eigvals, eigvecs = np.linalg.eigh(M)
    return np.stack([eigvecs[:, 0], eigvecs[:, 1]], axis=1)  # [n, 2]


def _propagate(A, colors, iterations):
    """Phase 2: iteratively recolor each vertex to its least-popular neighbor color."""
    n = A.shape[0]
    colors = colors.copy()
    for _ in range(iterations):
        new_colors = colors.copy()
        for v in range(n):
            neighbor_colors = colors[A[v] != 0]
            if neighbor_colors.size == 0:
                continue
            counts = [np.sum(neighbor_colors == c) for c in range(3)]
            new_colors[v] = int(np.argmin(counts))
        colors = new_colors
    return colors


def _support(A, colors, v):
    """min over the other two colors of v's neighbor count in that color."""
    neighbor_colors = colors[A[v] != 0]
    own = colors[v]
    others = [c for c in range(3) if c != own]
    return min(int(np.sum(neighbor_colors == c)) for c in others)


def _solve_component(component, A, colors, fixed_mask, node_budget=20_000):
    """
    Exact 3-coloring of one connected component of uncolored vertices,
    respecting the fixed colors of their already-colored neighbors. Returns
    True/False; mutates `colors` in place on success.

    NOT naive brute force: uses forward-checking + most-constrained-variable
    (MRV) ordering. Every assignment immediately prunes that color from
    same-component neighbors' domains, failing fast the instant any domain
    empties (rather than discovering the conflict only once that neighbor
    is reached); the next vertex branched on is always whichever currently
    has the fewest remaining options. This prunes the overwhelming majority
    of the search tree in practice while keeping the same exactness as
    plain backtracking.

    node_budget caps the search so a hard component (e.g. one straddling the
    phase transition) can't blow up combinatorially and stall the threshold
    search in _cleanup -- bail out (return False) and let _cleanup try the
    next threshold or fall back to DSATUR instead of hanging. Found via a
    real slowdown: c=5 at n=300 took 14+ minutes and 2% success because this
    function had no budget at all.
    """
    nodes = list(component)
    component_set = set(component)
    comp_neighbors = {
        v: [u for u in np.where(A[v] != 0)[0] if u in component_set]
        for v in nodes
    }

    domains = {}
    for v in nodes:
        neighbor_colors = {colors[u] for u in np.where(A[v] != 0)[0] if fixed_mask[u]}
        domains[v] = set(c for c in range(3) if c not in neighbor_colors)
        if not domains[v]:
            return False

    assignment = {}
    budget_counter = [0]

    def backtrack(cur_domains):
        budget_counter[0] += 1
        if budget_counter[0] > node_budget:
            return False
        if len(assignment) == len(nodes):
            return True

        v = min((u for u in nodes if u not in assignment), key=lambda u: len(cur_domains[u]))

        for c in cur_domains[v]:
            assignment[v] = c
            next_domains = dict(cur_domains)
            ok = True
            for u in comp_neighbors[v]:
                if u in assignment:
                    if assignment[u] == c:
                        ok = False
                        break
                    continue
                if c in next_domains[u]:
                    remaining = next_domains[u] - {c}
                    if not remaining:
                        ok = False
                        break
                    next_domains[u] = remaining
            if ok and backtrack(next_domains):
                return True
            del assignment[v]
        return False

    if not backtrack(domains):
        return False

    for v, c in assignment.items():
        colors[v] = c
    return True


def _cleanup(A, colors, max_component_size=18, max_threshold=None):
    """
    Phase 3: find the smallest support threshold that leaves every uncolored
    connected component within max_component_size, then brute-force each
    component. Returns (success, colors, method) where method is
    'spectral' if the threshold search succeeded, or 'spectral+dsatur_fallback'
    if no threshold worked and DSATUR was used on the residual.
    """
    n = A.shape[0]
    degree = A.sum(axis=1)
    d = max(degree.mean(), 1e-6)
    if max_threshold is None:
        max_threshold = int(d / 2) + 1

    for i in range(0, max_threshold + 1):
        fixed_mask = np.array([_support(A, colors, v) >= i for v in range(n)])
        working = colors.copy()

        uncolored = np.where(~fixed_mask)[0]
        if len(uncolored) == 0:
            same = working[:, None] == working[None, :]
            conflicts = int((A.astype(bool) & same).sum() // 2)
            if conflicts == 0:
                return True, working, "spectral"
            continue

        G_unc = nx.Graph()
        G_unc.add_nodes_from(uncolored.tolist())
        for u in uncolored:
            for v in np.where(A[u] != 0)[0]:
                if v in uncolored and v > u:
                    G_unc.add_edge(int(u), int(v))

        components = list(nx.connected_components(G_unc))
        if any(len(comp) > max_component_size for comp in components):
            continue

        ok = True
        for comp in components:
            if not _solve_component(comp, A, working, fixed_mask):
                ok = False
                break
        if not ok:
            continue

        same = working[:, None] == working[None, :]
        conflicts = int((A.astype(bool) & same).sum() // 2)
        if conflicts == 0:
            return True, working, "spectral"

    # Fallback: no threshold produced small-enough components. Use DSATUR on
    # the whole graph as a practical (not theorem-backed) completion.
    from classical_baseline import dsatur_3color
    ok, dsatur_colors = dsatur_3color(A, restarts=30)
    if ok:
        return True, dsatur_colors, "spectral+dsatur_fallback"
    return False, colors, "spectral+dsatur_fallback"


def spectral_init_coloring(adj, variant="adjacency"):
    """Phase 1 only -- exposed standalone for use as a GNN prior/diagnostic."""
    A = _to_numpy_adj(adj)
    n = A.shape[0]
    if variant == "bethe_hessian":
        return _bethe_hessian_init(A, n)
    if variant == "non_backtracking":
        return _non_backtracking_init(A, n)
    return _spectral_init(A, n)


def alon_kahale_3color(adj, propagation_iterations=None, max_component_size=18, variant="adjacency"):
    """
    Full 3-phase algorithm. Returns dict: success, colors, method, conflicts.

    variant="adjacency" is the original Alon-Kahale paper (phase 1 from the raw,
    degree-trimmed adjacency matrix). variant="bethe_hessian" swaps phase 1 for
    the non-backtracking-inspired Bethe Hessian init (_bethe_hessian_init) --
    same phases 2-3, aimed at recovering signal at lower average degree than
    the plain adjacency variant can. variant="non_backtracking" swaps phase 1
    for the exact non-backtracking companion-matrix init (_non_backtracking_init)
    instead of the Bethe Hessian's scalar approximation to it.
    """
    A = _to_numpy_adj(adj)
    n = A.shape[0]
    if propagation_iterations is None:
        propagation_iterations = max(1, int(np.ceil(np.log(max(n, 2)))))

    if variant == "bethe_hessian":
        colors = _bethe_hessian_init(A, n)
    elif variant == "non_backtracking":
        colors = _non_backtracking_init(A, n)
    else:
        colors = _spectral_init(A, n)
    colors = _propagate(A, colors, propagation_iterations)
    ok, colors, method = _cleanup(A, colors, max_component_size=max_component_size)

    same = colors[:, None] == colors[None, :]
    conflicts = int((A.astype(bool) & same).sum() // 2)
    method = method if variant == "adjacency" else method.replace("spectral", variant)
    return {"success": ok, "colors": colors, "method": method, "conflicts": conflicts}
