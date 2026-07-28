"""
"Spectral-NN" (user's proposal, 2026-07-14): a trained message-passing network
that produces a high-dimensional per-vertex embedding, such that a 2D PCA
projection of that embedding reveals the 3-coloring at *lower* average degree
c than plain adjacency-spectral (Alon-Kahale) or the hand-designed Bethe
Hessian variant can.

Unlike everything else trained tonight (QueryOptGNN_MP's unsupervised
conflict-minimization loss), this is trained with **direct supervision** on
the known planted partition -- we generate the data, so we know the true
classes. This sidesteps the "uniform fixed point" problem that motivated
noise/annealing/structured-prior ablations all night: a pairwise contrastive
loss (same-class pairs close, different-class pairs far) has no arbitrary
color-labeling to get stuck on.

At inference (no ground truth): project the embeddings onto their top-2
principal components (PCA -- eigenvectors of the d x d covariance matrix,
d = embedding dim, cheap regardless of n) to get two per-vertex coordinate
vectors, then feed those into spectral_coloring.py's *existing*,
already-verified phases 2 (propagation) and 3 (support-threshold cleanup +
brute force), exactly the way _spectral_init and _bethe_hessian_init do. Only
phase 1 (how the initial 2D picture is produced) is learned; everything
downstream is unchanged and already validated.
"""
import time

import numpy as np
import torch
import torch.nn as nn

from random_planted import create_planted_3col
from spectral_coloring import _two_eigvecs_to_coloring, _propagate, _cleanup, raw_spectral_coordinates


class SpectralEmbedder(nn.Module):
    """
    Simple message-passing embedder: per-vertex embeddings in R^embed_dim.

    Every mechanism found necessary tonight is an explicit constructor flag,
    not hardcoded -- so a saved checkpoint's exact config is always
    reconstructible, and no fix silently changes what an old checkpoint means.
    Defaults are the current best-known config; set flags to False to
    reproduce earlier (worse) variants exactly if ever needed for comparison.
    """

    def __init__(self, hidden_dim=32, embed_dim=8, rounds=8, reinject_degree_signal=True,
                 use_spectral_coords=False, spectral_coords_variant="adjacency"):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.rounds = rounds
        self.reinject_degree_signal = reinject_degree_signal
        # use_spectral_coords: user's suggestion (2026-07-14) -- init can come
        # from spectral (or other) signals too, not just degree. Feeds the raw
        # 2 eigenvectors of plain adjacency-spectral (or Bethe Hessian) in
        # alongside degree, since that classical signal already carries some
        # real information even in the hard region (that's the whole reason
        # alon_kahale_3color works at all) -- the network gets a head start
        # instead of rebuilding that signal from scratch via message passing.
        # Flagged, not hardcoded: default False preserves the degree-only
        # checkpoints already trained tonight exactly as they were.
        self.use_spectral_coords = use_spectral_coords
        self.spectral_coords_variant = spectral_coords_variant
        # Init from degree (normalized), not a constant -- an all-ones init gives
        # every vertex an identical starting point with no asymmetric signal to
        # break ties, which is exactly what caused a full representation collapse
        # (every vertex converging to the same embedding, verified empirically:
        # std ~1e-8 across the board) on the first training attempt. Degree is a
        # free, natural per-vertex asymmetry, same idea validated by A2 tonight.
        init_in_dim = 3 if use_spectral_coords else 1
        self.init = nn.Linear(init_in_dim, hidden_dim)
        self.msg_mlp = nn.Sequential(nn.Linear(hidden_dim, hidden_dim), nn.ReLU())
        self.gru = nn.GRUCell(hidden_dim, hidden_dim)
        self.readout = nn.Linear(hidden_dim, embed_dim)

    def forward(self, adj):
        """adj: [n,n] float tensor (0/1). Returns embeddings [n, embed_dim]."""
        n = adj.shape[0]
        device = adj.device
        degree = adj.sum(dim=1, keepdim=True).clamp(min=1.0)
        degree_norm = (degree - degree.mean()) / (degree.std() + 1e-6)

        if self.use_spectral_coords:
            with torch.no_grad():
                coords = raw_spectral_coordinates(adj, variant=self.spectral_coords_variant)
                coords_t = torch.as_tensor(coords, dtype=degree_norm.dtype, device=device)
                coords_norm = (coords_t - coords_t.mean(dim=0, keepdim=True)) / (coords_t.std(dim=0, keepdim=True) + 1e-6)
            init_feat = torch.cat([degree_norm, coords_norm], dim=1)
        else:
            init_feat = degree_norm

        degree_signal = self.init(init_feat)
        h = degree_signal
        for _ in range(self.rounds):
            msg = (adj @ self.msg_mlp(h)) / degree  # mean-aggregate neighbor messages
            h = self.gru(msg, h)
            if self.reinject_degree_signal:
                # Re-inject the degree signal every round, not just at init:
                # with many rounds (64+) the GRU's recurrent dynamics converge
                # to a single fixed point regardless of input and wash the
                # initial per-vertex asymmetry out entirely (verified: with
                # rounds=64 and only an init-time signal, all 3 class means
                # came out identical to float precision -- the exact A1
                # problem from QueryOptGNN_MP tonight, same fix: re-add the
                # anchor every round). Flagged (not hardcoded) so a checkpoint
                # trained with this off is never silently reinterpreted.
                h = h + degree_signal
        return self.readout(h)


def contrastive_loss(embeddings, assignment, margin=2.0, n_pairs=4000, rng=None,
                      variance_weight=1.0, variance_target=1.0):
    """
    Original loss (2026-07-14, first attempt): same-class pairs pulled
    together, different-class pairs pushed apart past a margin. Superseded by
    fisher_discriminant_loss as the default because it doesn't directly target
    the PCA-based readout (see that function's docstring) -- kept here,
    selectable via loss_fn="contrastive", rather than deleted, so it stays
    available for direct comparison instead of silently disappearing.
    """
    n = embeddings.shape[0]
    device = embeddings.device
    if rng is None:
        idx_i = torch.randint(0, n, (n_pairs,), device=device)
        idx_j = torch.randint(0, n, (n_pairs,), device=device)
    else:
        idx_i = torch.tensor(rng.integers(0, n, n_pairs), device=device)
        idx_j = torch.tensor(rng.integers(0, n, n_pairs), device=device)

    assignment = assignment.to(device)
    same = (assignment[idx_i] == assignment[idx_j]).float()
    dist2 = (embeddings[idx_i] - embeddings[idx_j]).pow(2).sum(-1)
    dist = dist2.clamp(min=1e-8).sqrt()

    same_loss = same * dist2
    diff_loss = (1 - same) * torch.clamp(margin - dist, min=0).pow(2)
    contrastive = (same_loss + diff_loss).mean()

    std = embeddings.std(dim=0)
    variance_penalty = torch.clamp(variance_target - std, min=0).mean()

    return contrastive + variance_weight * variance_penalty


def fisher_discriminant_loss(embeddings, assignment, k=3, variance_weight=1.0, variance_target=1.0):
    """
    Default loss since the fix (2026-07-14). Original pairwise contrastive_loss
    trained distances to respect class membership, but the actual readout
    (nn_spectral_init) takes the top-2 *principal components* of the whole
    embedding cloud -- a linear, global summary -- not pairwise distances. The
    two objectives are related but not the same: an embedding manifold can
    satisfy "same-class pairs close, different-class pairs far" while being
    curled up in a way PCA's linear projection doesn't cleanly separate.

    This loss targets the readout directly: minimize within-class scatter
    relative to between-class scatter (the Fisher/LDA criterion). Large
    between-class variance relative to within-class variance is exactly what
    makes the top-2 principal components of the *overall* covariance align
    with the class-separating directions -- so optimizing this ratio is
    optimizing for what nn_spectral_init actually uses, not a proxy for it.
    Also cheaper: 3 class means and sums, no pair sampling.
    """
    device = embeddings.device
    assignment = assignment.to(device)
    grand_mean = embeddings.mean(dim=0)

    within = embeddings.new_zeros(())
    between = embeddings.new_zeros(())
    for c in range(k):
        mask = assignment == c
        if mask.sum() == 0:
            continue
        emb_c = embeddings[mask]
        mean_c = emb_c.mean(dim=0)
        within = within + ((emb_c - mean_c) ** 2).sum()
        between = between + mask.sum() * ((mean_c - grand_mean) ** 2).sum()

    # within/(within+between) is bounded in [0,1) -- unlike within/between, it
    # can't blow up when between is near zero (verified this was happening:
    # raw within/between swung 237->73->353->932 loss over 100 steps).
    fisher = within / (within + between + 1e-6)

    std = embeddings.std(dim=0)
    variance_penalty = torch.clamp(variance_target - std, min=0).mean()

    return fisher + variance_weight * variance_penalty


def train_spectral_nn(
    n_train=300,
    c_range=(3, 20),
    hard_window=(4, 14),
    hard_window_prob=0.6,
    steps=2000,
    lr=1e-3,
    hidden_dim=32,
    embed_dim=8,
    rounds=64,
    reinject_degree_signal=True,
    use_spectral_coords=False,
    spectral_coords_variant="adjacency",
    loss_fn="fisher",
    device="cpu",
    log_every=100,
    seed=0,
):
    """
    Train from scratch, one planted graph per step (SGD). Biases c sampling
    toward the hard window (c=4-14, confirmed from real n=1000 data as the
    region where nothing else already works) since c=15+ is already solved by
    plain spectral and c<=3 by trivial peel/DSATUR -- no point spending
    capacity learning what's already solved.

    loss_fn: "fisher" (default, current best) or "contrastive" (the original
    pairwise loss, kept selectable -- not deleted -- for direct comparison).
    """
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)

    model = SpectralEmbedder(
        hidden_dim=hidden_dim, embed_dim=embed_dim, rounds=rounds,
        reinject_degree_signal=reinject_degree_signal,
        use_spectral_coords=use_spectral_coords,
        spectral_coords_variant=spectral_coords_variant,
    ).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    losses = []
    t0 = time.time()
    for step in range(1, steps + 1):
        if rng.random() < hard_window_prob:
            c = rng.uniform(*hard_window)
        else:
            c = rng.uniform(*c_range)

        assignment, adj = create_planted_3col(n_train, c)
        adj = adj.float().to(device)
        assignment = assignment.to(device)

        embeddings = model(adj)
        if loss_fn == "contrastive":
            loss = contrastive_loss(embeddings, assignment, rng=rng)
        else:
            loss = fisher_discriminant_loss(embeddings, assignment)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        losses.append(loss.item())

        if step % log_every == 0:
            recent = losses[-log_every:]
            print(f"step {step:5d}/{steps}  loss={sum(recent)/len(recent):.4f}  "
                  f"({time.time()-t0:.0f}s elapsed)")

    return model


def nn_spectral_init(adj, model, device="cpu"):
    """Phase 1 (learned): project embeddings onto their top-2 principal components."""
    A = adj.detach().cpu().numpy() if isinstance(adj, torch.Tensor) else np.asarray(adj)
    n = A.shape[0]
    with torch.no_grad():
        adj_t = torch.as_tensor(A, dtype=torch.float32, device=device)
        embeddings = model(adj_t).cpu().numpy()  # [n, embed_dim]

    mean = embeddings.mean(axis=0, keepdims=True)
    X = embeddings - mean
    cov = X.T @ X  # [d, d], cheap regardless of n
    eigvals, eigvecs = np.linalg.eigh(cov)  # ascending
    top2 = eigvecs[:, -2:]  # two largest-variance directions
    proj = X @ top2  # [n, 2]
    e1, e2 = proj[:, 0], proj[:, 1]
    return _two_eigvecs_to_coloring(e1, e2, n)


def spectral_nn_3color(adj, model, device="cpu", propagation_iterations=None, max_component_size=18):
    """Full pipeline: learned phase 1 + spectral_coloring.py's existing phases 2-3."""
    A = adj.detach().cpu().numpy() if isinstance(adj, torch.Tensor) else np.asarray(adj)
    n = A.shape[0]
    if propagation_iterations is None:
        propagation_iterations = max(1, int(np.ceil(np.log(max(n, 2)))))

    colors = nn_spectral_init(A, model, device=device)
    colors = _propagate(A, colors, propagation_iterations)
    ok, colors, method = _cleanup(A, colors, max_component_size=max_component_size)
    method = method.replace("spectral", "spectral_nn")

    same = colors[:, None] == colors[None, :]
    conflicts = int((A.astype(bool) & same).sum() // 2)
    return {"success": ok, "colors": colors, "method": method, "conflicts": conflicts}
