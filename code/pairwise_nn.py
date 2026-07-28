"""
"Pairwise same/different" reformulation (project idea, brainstormed 2026-07-14
night, built 2026-07-15): instead of predicting a per-vertex 3-way color
directly -- which has an arbitrary color-labeling symmetry (colors 0/1/2 are
interchangeable, a real source of training difficulty for spectral_nn.py's
embedder, worked around there via Fisher/contrastive losses that never name a
color) -- train a network to answer a binary question per PAIR of vertices:
"same planted class, or different?". This has no labeling symmetry at all:
"same" vs "different" doesn't require picking a canonical color numbering.

Architecture: a GRU-based message-passing embedder, structurally the same
recipe as spectral_nn.py's SpectralEmbedder (degree-seeded init, degree
re-injected every round to avoid the exact representation-collapse bug hit
twice there -- see that file's docstring/comments for the two failure modes:
constant-init collapse, and many-round GRU fixed-point collapse). This is
written as fully independent, new code (no import from spectral_nn.py) so
that file is untouched; only spectral_coloring.py's already-verified
propagation/cleanup phases and raw_spectral_coordinates helper are reused,
exactly the way spectral_nn.py itself reuses them.

Given the per-vertex embeddings, a pairwise "same-class" score is read out
cheaply via a learned negative-squared-distance head (default) or a small MLP
on the pair (both selectable, see `head=` flag) and trained with binary
cross-entropy against ground truth `assignment[u] == assignment[v]` from
create_planted_3col.

KNOWN DEGENERATE SOLUTION (must check for on every run, see check_collapse()):
in a balanced 3-partition, a random pair is "different-class" with probability
2/3 -- so a classifier that always predicts "different" gets ~67% raw accuracy
for free, without learning anything, and would look deceptively reasonable on
a bare accuracy number. Any reported accuracy MUST be compared against this
2/3 baseline, and predictions MUST be checked for actually varying across
different input pairs (not a constant output) before trusting a loss curve --
this is the pairwise analogue of the vertex-embedding collapse bug that hit
spectral_nn.py twice.

Inference-time decoding (no ground truth available): build the full n x n
same-class probability matrix P, center it into a signed similarity matrix
S = 2P - 1 (positive entries where the classifier thinks "same", negative
where "different"), and treat S as an assortative-community similarity
matrix -- take its two LARGEST-eigenvalue eigenvectors (opposite convention
from Alon-Kahale's plain adjacency phase 1, which takes the two SMALLEST
eigenvalues of a disassortative same-vs-different-only adjacency matrix; see
spectral_coloring.py's _bethe_hessian_init docstring for the same
assortative-vs-disassortative sign discussion). Feed those two eigenvectors
into spectral_coloring.py's existing, already-verified _two_eigvecs_to_coloring
/ _propagate / _cleanup pipeline -- only phase 1 is new, everything downstream
is the same proven machinery spectral_nn.py itself reuses.
"""
import time

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from random_planted import create_planted_3col
from spectral_coloring import _two_eigvecs_to_coloring, _propagate, _cleanup, raw_spectral_coordinates


def _raw_coords_k(adj, variant, k, r=None):
    """
    Thin duplicate of spectral_coloring.py's raw_spectral_coordinates,
    parameterized by k (number of eigenvectors) instead of hardcoded to 2 --
    kept as a local duplicate (not a modification of that shared file) per
    project convention (see multistage_nn.py's nn_spectral_init for the same
    pattern), and specifically to avoid touching spectral_coloring.py while
    other work is in flight there. Only used when num_spectral_eigvecs != 2;
    the k=2 default path still goes through the original imported function
    unchanged, so no existing checkpoint's behavior is affected.
    """
    A = adj.detach().cpu().numpy() if isinstance(adj, torch.Tensor) else np.asarray(adj, dtype=np.float64)
    n = A.shape[0]
    degree = A.sum(axis=1)

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
    return eigvecs[:, :k]


class PairwiseEmbedder(nn.Module):
    """
    Message-passing embedder producing per-vertex embeddings, plus a learned
    "same-class" pairwise head. Every mechanism is an explicit constructor
    flag (established convention, see spectral_nn.py's SpectralEmbedder) so a
    saved checkpoint's config is always reconstructible from the flags used
    to build it, never guessed from a filename alone.

    head="distance" (default): same-class score = sigmoid(bias - scale *
    ||e_u - e_v||^2), with scale, bias learned scalars. Cheap to vectorize
    into a full n x n matrix (a single torch.cdist call) -- important since
    inference at n=1000 needs the full pairwise matrix, not just sampled
    pairs.
    head="mlp": small MLP on [e_u, e_v, |e_u - e_v|]. More expressive, more
    expensive to materialize as a full n x n matrix (kept for comparison,
    not the default).

    use_dual_spectral_coords (2026-07-15, off by default): feed BOTH plain
    adjacency-spectral AND Bethe-Hessian raw coordinates at init (4 raw dims
    instead of 2) instead of forcing an either/or choice via
    spectral_coords_variant. These are two independently-derived classical
    estimates of the same underlying structure (Bethe-Hessian already beats
    plain adjacency at every c tested -- see spectral_coloring.py's
    _bethe_hessian_init docstring) -- concatenating both gives the network
    two hints to combine rather than only one.

    residual_spectral_coords (2026-07-15, off by default): concatenate the
    (normalized) raw spectral coordinates onto the FINAL embedding, right
    before the pairwise readout -- not just at init. Motivation: after
    `rounds` GRU applications, the initial classical signal has been
    transformed many times with no guarantee it survives; a skip connection
    at the output guarantees the classical signal is always literally present
    at decision time, so the network can never do worse than what plain
    spectral coordinates alone would give, and whatever it learned is pure
    upside on top -- not something that has to survive dozens of
    transformations intact.

    per_dim_scale (2026-07-15, off by default): FIX for a real negative result
    with residual_spectral_coords -- the plain distance head computes one
    GLOBAL Euclidean distance across all dims with a single scalar
    `exp(log_scale)`, so it cannot learn "trust the learned dims more than the
    noisy raw classical dims." Verified empirically (EXPERIMENTS.md set 14)
    that plain concatenation made results WORSE, not better, likely for
    exactly this reason. This flag makes log_scale a per-dimension vector
    (diagonal Mahalanobis distance) so the network can down-weight whichever
    dims turn out uninformative.

    use_common_neighbors (2026-07-15, off by default): adds a learned scalar
    weight on the pair's common-neighbor count (|N(u) intersect N(v)|,
    computed once per graph as `adj @ adj`) directly into the logit. Cheap
    (one matmul, same cost as one message-passing round), hand-engineered
    higher-order structure a GNN should in principle rediscover via message
    passing but might not reliably learn in the low-signal c=7-9 regime --
    same-class pairs may have systematically different shared-neighborhood
    statistics than different-class pairs in this planted model, worth
    testing directly rather than assuming message passing already covers it.
    `pair_logits`/`pairwise_prob_matrix` take an optional `cn` argument for
    this -- callers are responsible for computing and passing it (kept out of
    forward() so the embedder itself stays a pure per-vertex function).

    num_spectral_eigvecs (2026-07-15, default 2): use more than the top-2
    "informative" eigenvectors per classical variant (passed through to
    spectral_coloring.py's raw_spectral_coordinates(k=...)). Near the
    detection threshold the signal may not cleanly concentrate into exactly 2
    directions -- feeding more could retain weak-but-real signal otherwise
    discarded. Multiplies coord_dim accordingly (e.g. num_spectral_eigvecs=4
    with use_dual_spectral_coords=True -> 8 raw coordinate dims).
    """

    def __init__(self, hidden_dim=32, embed_dim=8, rounds=64, reinject_degree_signal=True,
                 use_spectral_coords=False, spectral_coords_variant="adjacency",
                 use_dual_spectral_coords=False, residual_spectral_coords=False,
                 per_dim_scale=False, use_common_neighbors=False, num_spectral_eigvecs=2,
                 head="distance"):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.embed_dim = embed_dim
        self.rounds = rounds
        self.reinject_degree_signal = reinject_degree_signal
        self.use_spectral_coords = use_spectral_coords
        self.spectral_coords_variant = spectral_coords_variant
        self.use_dual_spectral_coords = use_dual_spectral_coords
        self.residual_spectral_coords = residual_spectral_coords
        self.per_dim_scale = per_dim_scale
        self.use_common_neighbors = use_common_neighbors
        self.num_spectral_eigvecs = num_spectral_eigvecs
        self.head = head

        k = num_spectral_eigvecs
        self.coord_dim = (2 * k if use_dual_spectral_coords else (k if use_spectral_coords else 0))
        init_in_dim = 1 + self.coord_dim
        self.init = nn.Linear(init_in_dim, hidden_dim)
        self.msg_mlp = nn.Sequential(nn.Linear(hidden_dim, hidden_dim), nn.ReLU())
        self.gru = nn.GRUCell(hidden_dim, hidden_dim)
        self.readout = nn.Linear(hidden_dim, embed_dim)

        final_embed_dim = embed_dim + (self.coord_dim if residual_spectral_coords else 0)
        self.final_embed_dim = final_embed_dim

        if head == "distance":
            self.log_scale = nn.Parameter(torch.zeros(final_embed_dim) if per_dim_scale
                                           else torch.tensor(0.0))
            self.bias = nn.Parameter(torch.tensor(0.0))
            if use_common_neighbors:
                self.cn_weight = nn.Parameter(torch.tensor(0.0))
        elif head == "mlp":
            mlp_in_dim = final_embed_dim * 3 + (1 if use_common_neighbors else 0)
            self.head_mlp = nn.Sequential(
                nn.Linear(mlp_in_dim, hidden_dim), nn.ReLU(),
                nn.Linear(hidden_dim, 1),
            )
        else:
            raise ValueError(f"unknown head: {head!r}")

    def _compute_coords_norm(self, adj, degree_norm):
        """Shared by init and the residual connection -- computes the same
        normalized coordinate tensor once, either single-variant (2 dims) or
        dual-variant (4 dims: adjacency + Bethe-Hessian concatenated)."""
        device = adj.device
        k = self.num_spectral_eigvecs
        with torch.no_grad():
            if k == 2:
                get_coords = lambda variant: raw_spectral_coordinates(adj, variant=variant)
            else:
                get_coords = lambda variant: _raw_coords_k(adj, variant, k)
            if self.use_dual_spectral_coords:
                coords_a = get_coords("adjacency")
                coords_b = get_coords("bethe_hessian")
                coords = np.concatenate([coords_a, coords_b], axis=1)
            else:
                coords = get_coords(self.spectral_coords_variant)
            coords_t = torch.as_tensor(coords, dtype=degree_norm.dtype, device=device)
            coords_norm = (coords_t - coords_t.mean(dim=0, keepdim=True)) / (coords_t.std(dim=0, keepdim=True) + 1e-6)
        return coords_norm

    def forward(self, adj):
        """adj: [n,n] float tensor (0/1). Returns per-vertex embeddings
        [n, final_embed_dim] -- final_embed_dim == embed_dim unless
        residual_spectral_coords is set, in which case the raw classical
        coordinates are concatenated onto the output too."""
        n = adj.shape[0]
        device = adj.device
        degree = adj.sum(dim=1, keepdim=True).clamp(min=1.0)
        degree_norm = (degree - degree.mean()) / (degree.std() + 1e-6)

        coords_norm = None
        if self.use_spectral_coords or self.use_dual_spectral_coords:
            coords_norm = self._compute_coords_norm(adj, degree_norm)
            init_feat = torch.cat([degree_norm, coords_norm], dim=1)
        else:
            init_feat = degree_norm

        degree_signal = self.init(init_feat)
        h = degree_signal
        for _ in range(self.rounds):
            msg = (adj @ self.msg_mlp(h)) / degree
            h = self.gru(msg, h)
            if self.reinject_degree_signal:
                h = h + degree_signal
        out = self.readout(h)
        if self.residual_spectral_coords:
            if coords_norm is None:
                coords_norm = self._compute_coords_norm(adj, degree_norm)
            out = torch.cat([out, coords_norm], dim=1)
        return out

    def pair_logits(self, emb_u, emb_v, cn=None):
        """
        emb_u, emb_v: same-shaped tensors [..., embed_dim] (broadcastable),
        e.g. [n_pairs, embed_dim] for sampled-pair training. Returns raw
        (pre-sigmoid) logits for "same class". cn (optional): common-neighbor
        count per pair, same leading shape as emb_u/emb_v minus the feature
        dim -- only used if use_common_neighbors is set; caller computes it
        (see pairwise_bce_loss / nn_pairwise_init for how).
        """
        if self.head == "distance":
            diff2 = (emb_u - emb_v).pow(2)
            if self.per_dim_scale:
                dist2 = (diff2 * torch.exp(self.log_scale)).sum(-1)
            else:
                dist2 = diff2.sum(-1) * torch.exp(self.log_scale)
            logit = self.bias - dist2
        else:
            diff = emb_u - emb_v
            feat = torch.cat([emb_u, emb_v, diff.abs()], dim=-1)
            if self.use_common_neighbors:
                cn_feat = cn if cn is not None else torch.zeros_like(feat[..., :1].squeeze(-1))
                feat = torch.cat([feat, cn_feat.unsqueeze(-1)], dim=-1)
            logit = self.head_mlp(feat).squeeze(-1)

        if self.head == "distance" and self.use_common_neighbors and cn is not None:
            logit = logit + self.cn_weight * cn
        return logit

    def pairwise_prob_matrix(self, embeddings, cn=None):
        """
        Full n x n same-class probability matrix, vectorized -- used at
        inference (decoding needs the whole matrix, not sampled pairs).
        cn (optional): full n x n common-neighbor count matrix.
        """
        n = embeddings.shape[0]
        if self.head == "distance":
            if self.per_dim_scale:
                scaled = embeddings * torch.sqrt(torch.exp(self.log_scale))
                dist2 = torch.cdist(scaled, scaled, p=2).pow(2)
                logits = self.bias - dist2
            else:
                dist2 = torch.cdist(embeddings, embeddings, p=2).pow(2)
                logits = self.bias - torch.exp(self.log_scale) * dist2
            if self.use_common_neighbors and cn is not None:
                logits = logits + self.cn_weight * cn
        else:
            eu = embeddings.unsqueeze(1).expand(n, n, -1)
            ev = embeddings.unsqueeze(0).expand(n, n, -1)
            diff = eu - ev
            feat = torch.cat([eu, ev, diff.abs()], dim=-1)
            if self.use_common_neighbors:
                cn_feat = cn if cn is not None else torch.zeros(n, n, dtype=embeddings.dtype, device=embeddings.device)
                feat = torch.cat([feat, cn_feat.unsqueeze(-1)], dim=-1)
            logits = self.head_mlp(feat).squeeze(-1)
        return torch.sigmoid(logits)


def _fisher_term(embeddings, assignment, k=3):
    """within/(within+between) scatter ratio -- see spectral_nn.py's
    fisher_discriminant_loss for the full derivation. Used here only as an
    optional auxiliary term (see pairwise_bce_loss's fisher_weight flag)."""
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
    return within / (within + between + 1e-6)


def pairwise_bce_loss(model, embeddings, assignment, n_pairs=4000, rng=None,
                       variance_weight=1.0, variance_target=1.0, fisher_weight=0.0, adj=None,
                       entropy_weight=0.0):
    """
    Binary cross-entropy on sampled random pairs, label = 1 if same planted
    class else 0. Same sampling pattern as spectral_nn.py's contrastive_loss
    (random index pairs, optionally via a supplied numpy Generator for
    reproducibility) but a purpose-built binary-classification loss, not a
    copy of that function.

    Includes the same variance-floor regularizer used throughout tonight's
    work (penalize per-embedding-dimension std below target) as a defense
    against representation collapse -- necessary here too since this recurrent
    architecture has the identical many-round GRU fixed-point failure mode
    documented in spectral_nn.py.

    fisher_weight (new collapse finding, 2026-07-15): pure BCE on sampled
    pairs turned out to have ITS OWN, different collapse mode from the
    vertex-embedding one -- verified empirically (check_collapse()) that with
    fisher_weight=0 the model converges to predicting "different" for
    essentially every pair (pred_same_frac ~0.02-0.03 regardless of the
    actual same-class fraction ~0.33, and accuracy lands exactly on the 2/3
    "always different" baseline, with all 3 per-class embedding means
    numerically near-identical). Root cause: with imbalanced pairs (2/3
    "different" by construction) and a small-magnitude random-init distance
    head, the BCE gradient's easiest early win is just pushing the global
    bias down to predict "different" always, rather than learning any real
    class structure -- and once distances are uninformative, there's no
    gradient signal left to escape that fixed point. Adding this auxiliary
    Fisher/LDA scatter term (same one spectral_nn.py's fisher_discriminant_loss
    uses, imported as _fisher_term) directly rewards between-class separation
    regardless of the distance-head's current calibration, which reliably
    breaks the collapse (verified: fisher_weight=2.0 + use_spectral_coords=True
    gives accuracy meaningfully above baseline with clearly distinct per-class
    means -- see check_collapse() output in this file's dev log). Flagged,
    default 0.0, so it must be explicitly opted into.
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
    labels = (assignment[idx_i] == assignment[idx_j]).float()

    cn_pairs = None
    if model.use_common_neighbors:
        if adj is None:
            raise ValueError("model.use_common_neighbors=True requires adj to be passed to pairwise_bce_loss")
        cn_matrix = adj @ adj
        cn_norm = (cn_matrix - cn_matrix.mean()) / (cn_matrix.std() + 1e-6)
        cn_pairs = cn_norm[idx_i, idx_j]

    logits = model.pair_logits(embeddings[idx_i], embeddings[idx_j], cn=cn_pairs)
    bce = F.binary_cross_entropy_with_logits(logits, labels)

    std = embeddings.std(dim=0)
    variance_penalty = torch.clamp(variance_target - std, min=0).mean()

    loss = bce + variance_weight * variance_penalty
    if fisher_weight > 0:
        loss = loss + fisher_weight * _fisher_term(embeddings, assignment)
    if entropy_weight > 0:
        # Direct symmetry-breaking regularizer (2026-07-17, user's request): the
        # binary analog of the classic p=[1/3,1/3,1/3] uniform-fixed-point trap
        # is p=0.5 (maximally uncertain same/different prediction) -- unlike the
        # noise-injection hack used elsewhere in this project, this is a
        # deterministic penalty directly on prediction entropy, pushing logits
        # away from 0 (sigmoid(0)=0.5) regardless of which side they land on.
        probs = torch.sigmoid(logits).clamp(1e-6, 1 - 1e-6)
        entropy = -(probs * probs.log() + (1 - probs) * (1 - probs).log())
        loss = loss + entropy_weight * entropy.mean()
    return loss


def train_pairwise_nn(
    n_train=300,
    c_range=(1, 8),
    steps=1000,
    lr=1e-3,
    hidden_dim=32,
    embed_dim=8,
    rounds=64,
    reinject_degree_signal=True,
    use_spectral_coords=True,
    spectral_coords_variant="adjacency",
    use_dual_spectral_coords=False,
    residual_spectral_coords=False,
    per_dim_scale=False,
    use_common_neighbors=False,
    num_spectral_eigvecs=2,
    head="distance",
    n_pairs=4000,
    fisher_weight=2.0,
    entropy_weight=0.0,
    device="cpu",
    log_every=100,
    seed=0,
):
    """
    Train from scratch, one planted graph per step (SGD), c sampled uniformly
    within c_range every step -- mirrors train_spectral_nn's per-step sampling
    pattern in spectral_nn.py, but without that function's extra hard-window
    resampling bias: here c_range IS the band (small/medium/large), sampled
    uniformly throughout, per the task spec.

    Defaults for use_spectral_coords (True) and fisher_weight (2.0) differ
    from this module's low-level building blocks' own defaults (False / 0.0)
    -- those low-level defaults preserve "vanilla" behavior for anyone
    constructing a PairwiseEmbedder or calling pairwise_bce_loss directly;
    THESE defaults are the empirically-required fix for the pure-BCE collapse
    documented in pairwise_bce_loss's docstring, since plain BCE from a cold
    start never escapes the "always predict different" fixed point in
    practice. Every checkpoint trained by this function must still be saved
    under a filename that encodes these flags explicitly (see
    train_pairwise_bands.py) -- never assume the defaults from the filename.
    """
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)

    model = PairwiseEmbedder(
        hidden_dim=hidden_dim, embed_dim=embed_dim, rounds=rounds,
        reinject_degree_signal=reinject_degree_signal,
        use_spectral_coords=use_spectral_coords,
        spectral_coords_variant=spectral_coords_variant,
        use_dual_spectral_coords=use_dual_spectral_coords,
        residual_spectral_coords=residual_spectral_coords,
        per_dim_scale=per_dim_scale,
        use_common_neighbors=use_common_neighbors,
        num_spectral_eigvecs=num_spectral_eigvecs,
        head=head,
    ).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    losses = []
    t0 = time.time()
    for step in range(1, steps + 1):
        c = rng.uniform(*c_range)
        assignment, adj = create_planted_3col(n_train, c)
        adj = adj.float().to(device)
        assignment = assignment.to(device)

        embeddings = model(adj)
        loss = pairwise_bce_loss(model, embeddings, assignment, n_pairs=n_pairs, rng=rng,
                                  fisher_weight=fisher_weight, adj=adj, entropy_weight=entropy_weight)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        losses.append(loss.item())

        if step % log_every == 0:
            recent = losses[-log_every:]
            print(f"step {step:5d}/{steps}  loss={sum(recent)/len(recent):.4f}  "
                  f"({time.time()-t0:.0f}s elapsed)")

    return model


def _cn_matrix_norm(adj):
    """Normalized full n x n common-neighbor count matrix -- shared by
    check_collapse, nn_pairwise_init, and pairwise_bce_loss's caller sites
    that need the full matrix rather than sampled pairs."""
    cn = adj @ adj
    return (cn - cn.mean()) / (cn.std() + 1e-6)


@torch.no_grad()
def check_collapse(model, n=300, c_values=(6, 10, 14), n_pairs=4000, seed=123, device="cpu"):
    """
    Diagnostic, meant to be run and eyeballed before trusting any loss curve
    (established discipline tonight after two real collapse bugs in
    spectral_nn.py). Prints, per test graph:
      - per-class mean embedding vectors and per-dimension std (a collapsed
        representation has near-identical class means / near-zero std)
      - held-out-pair classification accuracy vs the 2/3 "always predict
        different" degenerate baseline
      - fraction of predictions that are "same" (a collapsed classifier
        predicts a near-constant fraction regardless of the actual pairs)
    """
    rng = np.random.default_rng(seed)
    for c in c_values:
        assignment, adj = create_planted_3col(n, c)
        adj = adj.float().to(device)
        embeddings = model(adj)

        print(f"\n-- collapse check: n={n}, c={c} --")
        for k in range(3):
            mask = assignment == k
            if mask.sum() == 0:
                continue
            emb_k = embeddings[mask]
            print(f"  class {k} (n={int(mask.sum())}): mean={emb_k.mean(dim=0).numpy().round(3)}  "
                  f"std={emb_k.std(dim=0).numpy().round(3)}")

        idx_i = torch.tensor(rng.integers(0, n, n_pairs))
        idx_j = torch.tensor(rng.integers(0, n, n_pairs))
        labels = (assignment[idx_i] == assignment[idx_j]).float()
        cn_pairs = None
        if model.use_common_neighbors:
            cn_pairs = _cn_matrix_norm(adj)[idx_i, idx_j]
        probs = torch.sigmoid(model.pair_logits(embeddings[idx_i], embeddings[idx_j], cn=cn_pairs))
        preds = (probs > 0.5).float()

        acc = (preds == labels).float().mean().item()
        baseline = 1.0 - labels.mean().item()  # "always predict different" accuracy
        frac_pred_same = preds.mean().item()
        frac_actual_same = labels.mean().item()
        print(f"  accuracy={acc:.3f}  'always-different' baseline={baseline:.3f}  "
              f"pred_same_frac={frac_pred_same:.3f} (actual same_frac={frac_actual_same:.3f})  "
              f"prob_std={probs.std().item():.4f}")


def nn_pairwise_init(adj, model, device="cpu"):
    """
    Phase 1 (learned, replaces Alon-Kahale's spectral init): build the full
    same-class probability matrix, center to a signed similarity matrix
    S = 2P - 1, and take its two LARGEST-eigenvalue eigenvectors (assortative
    convention -- see module docstring for why this is the opposite sign
    convention from plain adjacency-spectral phase 1) as the two "informative"
    directions fed to the existing _two_eigvecs_to_coloring angle-search.
    """
    A = adj.detach().cpu().numpy() if isinstance(adj, torch.Tensor) else np.asarray(adj)
    n = A.shape[0]
    with torch.no_grad():
        adj_t = torch.as_tensor(A, dtype=torch.float32, device=device)
        embeddings = model(adj_t)
        cn = _cn_matrix_norm(adj_t) if model.use_common_neighbors else None
        P = model.pairwise_prob_matrix(embeddings, cn=cn).cpu().numpy()

    np.fill_diagonal(P, 0.0)
    S = 2.0 * P - 1.0
    S = (S + S.T) / 2.0  # symmetrize away any float asymmetry
    eigvals, eigvecs = np.linalg.eigh(S)  # ascending
    e1, e2 = eigvecs[:, -1], eigvecs[:, -2]  # two largest eigenvalues
    return _two_eigvecs_to_coloring(e1, e2, n)


def pairwise_nn_3color(adj, model, device="cpu", propagation_iterations=None, max_component_size=18):
    """Full pipeline: learned phase 1 (pairwise same/different) + spectral_coloring.py's existing phases 2-3."""
    A = adj.detach().cpu().numpy() if isinstance(adj, torch.Tensor) else np.asarray(adj)
    n = A.shape[0]
    if propagation_iterations is None:
        propagation_iterations = max(1, int(np.ceil(np.log(max(n, 2)))))

    colors = nn_pairwise_init(A, model, device=device)
    colors = _propagate(A, colors, propagation_iterations)
    ok, colors, method = _cleanup(A, colors, max_component_size=max_component_size)
    method = method.replace("spectral", "pairwise_nn")

    same = colors[:, None] == colors[None, :]
    conflicts = int((A.astype(bool) & same).sum() // 2)
    return {"success": ok, "colors": colors, "method": method, "conflicts": conflicts}
