"""
"Multi-stage / nested-loop" embedder (2026-07-15), extending the single-cell
`SpectralEmbedder` idea from spectral_nn.py.

Motivating idea (from tonight's brainstorming, verbatim): "we might need not
just 1 GNN that passes many times... like we have multi attention, each
finding some attribute... loop many times (A), loop many times (B), loop many
times (C), and now a whole loop around it (so go back to A). With positional
encoding of the iteration of small and large loop." The hypothesis: going
from 8 to 64 rounds of the SAME recurrent cell already measurably helped
(confirmed tonight in spectral_nn.py) -- more reasoning *depth* helps. This
file tests whether more reasoning *shape* -- genuinely different learned
operations, composed and repeated -- helps further.

Architecture:
  - `num_stages` distinct GRU cells (stage A, B, C, ...), each with its own
    learned weights (NOT the same cell reused) and its own message MLP.
  - An outer loop of `outer_iters` iterations cycles through all stages in
    order (A -> B -> C -> A -> B -> C -> ...).
  - Each stage runs `inner_rounds` message-passing steps per visit.
  - Total effective rounds = num_stages * inner_rounds * outer_iters.
  - Sinusoidal positional encoding of (outer_iteration_index,
    inner_round_index) is projected (learned linear layer) and injected into
    every single message-passing step, not just once -- same reasoning as
    Transformer positional encoding, adapted to this recurrent setting.

This is a NEW architecture, not a flag on SpectralEmbedder, so it lives in
its own file per project convention -- nothing in spectral_nn.py is touched
or at risk here. Every mechanism is an explicit constructor flag (mirroring
SpectralEmbedder's discipline) so a saved checkpoint's config is always
exactly reconstructible from its filename + this file's defaults.

Known-risk, explicitly tested for (see check_collapse() at the bottom):
multiple sequential applications of ANY single recurrent cell tend to
converge to an input-independent fixed point (this hit spectral_nn.py twice
tonight: once from a constant init with no anti-collapse term, once again
from simply raising rounds 8->64 with only an init-time signal). With
num_stages x inner_rounds possibly compounding this risk across THREE
separate cells, the degree-based per-vertex asymmetric signal is re-injected
after every single stage-cell application (not just at the very start),
mirroring reinject_degree_signal in SpectralEmbedder -- but this must be
verified empirically per run, not assumed, which is exactly what
check_collapse() does.
"""
import time

import numpy as np
import torch
import torch.nn as nn

from random_planted import create_planted_3col
from spectral_coloring import _two_eigvecs_to_coloring, _propagate, _cleanup, raw_spectral_coordinates
from spectral_nn import fisher_discriminant_loss, contrastive_loss


def _sinusoidal_encoding(pos, dim, device, dtype):
    """Standard Transformer-style sinusoidal encoding of a single scalar position."""
    if dim <= 0:
        return torch.zeros(0, device=device, dtype=dtype)
    half = max(dim // 2, 1)
    rest = dim - half
    div_sin = torch.exp(torch.arange(half, device=device, dtype=dtype) * (-np.log(10000.0) / half))
    sin_part = torch.sin(pos * div_sin)
    if rest == 0:
        return sin_part
    div_cos = torch.exp(torch.arange(rest, device=device, dtype=dtype) * (-np.log(10000.0) / rest))
    cos_part = torch.cos(pos * div_cos)
    return torch.cat([sin_part, cos_part])


def _loop_positional_encoding(outer_idx, inner_idx, dim, device, dtype):
    """
    Concatenation of two independent sinusoidal encodings: one for the outer
    (large) loop index, one for the inner (small) loop index within the
    current stage -- half of `dim` each. This is the "positional encoding of
    the iteration of small and large loop" the brainstorming asked for.
    """
    half = dim // 2
    rest = dim - half
    outer_enc = _sinusoidal_encoding(float(outer_idx), half, device, dtype)
    inner_enc = _sinusoidal_encoding(float(inner_idx), rest, device, dtype)
    return torch.cat([outer_enc, inner_enc])


class MultiStageEmbedder(nn.Module):
    """
    Nested inner/outer loop message-passing embedder with `num_stages`
    genuinely distinct recurrent cells cycled through `outer_iters` times.

    All mechanisms are explicit constructor flags (never hardcoded), same
    discipline as SpectralEmbedder in spectral_nn.py:
      - num_stages / inner_rounds / outer_iters: the loop shape itself.
      - reinject_degree_signal: re-add the degree-based init signal after
        every stage-cell application (default True -- see module docstring;
        this is the fix for the fixed-point collapse bug hit twice tonight).
      - use_spectral_coords: same idea as SpectralEmbedder -- optionally feed
        raw plain-spectral eigenvector coordinates in at init, alongside
        degree, as a classical-signal head start.
      - pos_enc_dim: width of the sinusoidal (outer, inner) loop positional
        encoding injected into every single step (0 disables it entirely,
        for direct before/after comparison).
    """

    def __init__(self, hidden_dim=32, embed_dim=8, num_stages=3, inner_rounds=6, outer_iters=4,
                 reinject_degree_signal=True, use_spectral_coords=False, spectral_coords_variant="adjacency",
                 pos_enc_dim=16):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.num_stages = num_stages
        self.inner_rounds = inner_rounds
        self.outer_iters = outer_iters
        self.reinject_degree_signal = reinject_degree_signal
        self.use_spectral_coords = use_spectral_coords
        self.spectral_coords_variant = spectral_coords_variant
        self.pos_enc_dim = pos_enc_dim

        init_in_dim = 3 if use_spectral_coords else 1
        self.init = nn.Linear(init_in_dim, hidden_dim)

        # Genuinely separate learned weights per stage -- a ModuleList of
        # ModuleDicts, NOT one cell reused num_stages times. This is the
        # entire point of the architecture: different operations, not just
        # more applications of the same one.
        self.stage_msg_mlp = nn.ModuleList([
            nn.Sequential(nn.Linear(hidden_dim, hidden_dim), nn.ReLU())
            for _ in range(num_stages)
        ])
        self.stage_gru = nn.ModuleList([
            nn.GRUCell(hidden_dim, hidden_dim)
            for _ in range(num_stages)
        ])

        if pos_enc_dim > 0:
            self.pos_proj = nn.Linear(pos_enc_dim, hidden_dim)
        else:
            self.pos_proj = None

        self.readout = nn.Linear(hidden_dim, embed_dim)

    def forward(self, adj, return_trace=False):
        """
        adj: [n,n] float tensor (0/1). Returns embeddings [n, embed_dim].
        If return_trace=True, also returns a list of per-class-free hidden
        states h captured at each outer iteration boundary (used only by
        check_collapse() for empirical collapse diagnostics -- never used
        during training/inference).
        """
        n = adj.shape[0]
        device = adj.device
        dtype = adj.dtype
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

        trace = [] if return_trace else None

        for outer in range(self.outer_iters):
            for stage_idx in range(self.num_stages):
                msg_mlp = self.stage_msg_mlp[stage_idx]
                gru = self.stage_gru[stage_idx]
                for inner in range(self.inner_rounds):
                    msg = (adj @ msg_mlp(h)) / degree  # mean-aggregate neighbor messages
                    if self.pos_proj is not None:
                        pos = _loop_positional_encoding(outer, inner, self.pos_enc_dim, device, dtype)
                        msg = msg + self.pos_proj(pos).unsqueeze(0)  # broadcast [1,hidden] -> [n,hidden]
                    h = gru(msg, h)
                    if self.reinject_degree_signal:
                        # Re-inject after every single stage-cell step (not
                        # just at stage/outer boundaries) -- the exact fix
                        # verified necessary for the single-cell 64-round
                        # case in spectral_nn.py; with 3 stages compounding
                        # depth further, doing this less often is riskier,
                        # not safer.
                        h = h + degree_signal
            if return_trace:
                trace.append(h.detach().clone())

        out = self.readout(h)
        if return_trace:
            return out, trace
        return out


def train_multistage_nn(
    n_train=300,
    c_range=(3, 20),
    hard_window=(4, 14),
    hard_window_prob=0.6,
    steps=1000,
    lr=1e-3,
    hidden_dim=32,
    embed_dim=8,
    num_stages=3,
    inner_rounds=6,
    outer_iters=4,
    reinject_degree_signal=True,
    use_spectral_coords=False,
    spectral_coords_variant="adjacency",
    pos_enc_dim=16,
    loss_fn="fisher",
    device="cpu",
    log_every=100,
    seed=0,
):
    """
    Mirrors train_spectral_nn's loop structure exactly (one planted graph per
    SGD step, c sampled uniformly with a bias toward the hard window).
    loss_fn: "fisher" (default, targets the PCA readout directly -- see
    fisher_discriminant_loss's docstring in spectral_nn.py) or "contrastive".
    """
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)

    model = MultiStageEmbedder(
        hidden_dim=hidden_dim, embed_dim=embed_dim,
        num_stages=num_stages, inner_rounds=inner_rounds, outer_iters=outer_iters,
        reinject_degree_signal=reinject_degree_signal,
        use_spectral_coords=use_spectral_coords,
        spectral_coords_variant=spectral_coords_variant,
        pos_enc_dim=pos_enc_dim,
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
    """Phase 1 (learned): project embeddings onto their top-2 principal components.
    Identical pattern to spectral_nn.py's function of the same name -- kept as
    a thin duplicate (not an import) so this file has no risk of ever being
    broken by an edit made to spectral_nn.py's version, per the versioning
    discipline for this task."""
    A = adj.detach().cpu().numpy() if isinstance(adj, torch.Tensor) else np.asarray(adj)
    n = A.shape[0]
    with torch.no_grad():
        adj_t = torch.as_tensor(A, dtype=torch.float32, device=device)
        embeddings = model(adj_t).cpu().numpy()  # [n, embed_dim]

    mean = embeddings.mean(axis=0, keepdims=True)
    X = embeddings - mean
    cov = X.T @ X
    eigvals, eigvecs = np.linalg.eigh(cov)
    top2 = eigvecs[:, -2:]
    proj = X @ top2
    e1, e2 = proj[:, 0], proj[:, 1]
    return _two_eigvecs_to_coloring(e1, e2, n)


def multistage_nn_3color(adj, model, device="cpu", propagation_iterations=None, max_component_size=18):
    """Full pipeline: learned phase 1 (multi-stage embedder) + spectral_coloring.py's
    existing, already-validated phases 2-3 (_propagate, _cleanup)."""
    A = adj.detach().cpu().numpy() if isinstance(adj, torch.Tensor) else np.asarray(adj)
    n = A.shape[0]
    if propagation_iterations is None:
        propagation_iterations = max(1, int(np.ceil(np.log(max(n, 2)))))

    colors = nn_spectral_init(A, model, device=device)
    colors = _propagate(A, colors, propagation_iterations)
    ok, colors, method = _cleanup(A, colors, max_component_size=max_component_size)
    method = method.replace("spectral", "multistage_nn")

    same = colors[:, None] == colors[None, :]
    conflicts = int((A.astype(bool) & same).sum() // 2)
    return {"success": ok, "colors": colors, "method": method, "conflicts": conflicts}


def check_collapse(model, n=300, c=10, device="cpu", seed=1):
    """
    Empirical collapse diagnostic (mandatory before trusting any loss curve,
    per this task's instructions -- collapse is easy to miss just by
    watching the loss number since fisher/contrastive losses can look
    "fine" on a degenerate embedding for a while).

    Generates one planted graph, runs the model with return_trace=True,
    prints the per-class mean embedding vector (of the true planted
    partition) at each outer-loop boundary, plus the overall embedding std.
    If class means are numerically near-identical (or overall std ~1e-6 or
    below), the network has collapsed to an input-independent fixed point
    regardless of what the loss number says.
    """
    torch.manual_seed(seed)
    assignment, adj = create_planted_3col(n, c)
    adj = adj.float().to(device)
    with torch.no_grad():
        final_embed, trace = model(adj, return_trace=True)

    print(f"check_collapse: n={n} c={c}, num_stages={model.num_stages}, "
          f"inner_rounds={model.inner_rounds}, outer_iters={model.outer_iters}")
    for outer_idx, h in enumerate(trace):
        h = h.cpu().numpy()
        overall_std = h.std()
        class_means = [h[(assignment == k).cpu().numpy()].mean(axis=0) for k in range(3)]
        # pairwise distances between class means -- near-zero => collapsed
        d01 = np.linalg.norm(class_means[0] - class_means[1])
        d02 = np.linalg.norm(class_means[0] - class_means[2])
        d12 = np.linalg.norm(class_means[1] - class_means[2])
        print(f"  outer={outer_idx:2d}  overall_std={overall_std:.6f}  "
              f"class_mean_dists=({d01:.4f},{d02:.4f},{d12:.4f})")

    final = final_embed.detach().cpu().numpy()
    final_std = final.std()
    final_class_means = [final[(assignment == k).cpu().numpy()].mean(axis=0) for k in range(3)]
    fd01 = np.linalg.norm(final_class_means[0] - final_class_means[1])
    fd02 = np.linalg.norm(final_class_means[0] - final_class_means[2])
    fd12 = np.linalg.norm(final_class_means[1] - final_class_means[2])
    print(f"  FINAL (post-readout) overall_std={final_std:.6f}  "
          f"class_mean_dists=({fd01:.4f},{fd02:.4f},{fd12:.4f})")
    collapsed = final_std < 1e-4 or max(fd01, fd02, fd12) < 1e-4
    print(f"  => {'COLLAPSED' if collapsed else 'not collapsed (distinct class means found)'}")
    return not collapsed


if __name__ == "__main__":
    # Quick untrained-model sanity check: verify the architecture itself
    # doesn't collapse by construction (before spending any training time).
    torch.manual_seed(0)
    m = MultiStageEmbedder(hidden_dim=32, embed_dim=8, num_stages=3, inner_rounds=6, outer_iters=4)
    check_collapse(m, n=300, c=10)
