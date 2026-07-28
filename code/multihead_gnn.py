"""
"Multi-head GNN" (2026-07-15) -- genuinely PARALLEL heads, not the sequential
nested-loop composition in `multistage_nn.py` (which runs stage A, then B,
then C, then back to A -- a sequential pipeline). This is the literal
multi-head-attention structure applied to message passing: `num_heads`
independently-parameterized GNNs (own init projection, own message MLP, own
GRU cell -- mirroring attention's separate learned Q/K/V projection per head)
each run the SAME `rounds` of message passing INDEPENDENTLY AND IN PARALLEL
over the same graph, starting from the same raw signal. Their final per-vertex
hidden states are concatenated and passed through one output projection
(`out_proj`, playing the exact role of attention's `W_O`), exactly like
multi-head attention concatenates per-head outputs before the final linear
layer.

The hypothesis this tests, distinct from multistage_nn.py's: does giving
several independent "algorithms" a vote (concatenated, not sequentially
composed) on the same graph let different heads specialize on different
structure (e.g. one head good at sparse regions, another at locally dense
subgraphs), the way attention heads specialize on different relations? This
is a different bet than multistage's "more distinct sequential reasoning
steps" bet.

This is a NEW architecture, own file, per project convention -- nothing in
spectral_nn.py or multistage_nn.py is touched or at risk. Every mechanism is
an explicit constructor flag. Collapse is checked PER HEAD, not just on the
final concatenated output -- an aggregate check could hide 3 of 4 heads
collapsing while 1 healthy head keeps the concatenated stats looking fine,
which would be a materially different (and worse-understood) failure than
whole-network collapse.
"""
import time

import numpy as np
import torch
import torch.nn as nn

from random_planted import create_planted_3col
from spectral_coloring import _two_eigvecs_to_coloring, _propagate, _cleanup, raw_spectral_coordinates
from spectral_nn import fisher_discriminant_loss, contrastive_loss


class MultiHeadGNNEmbedder(nn.Module):
    """
    `num_heads` independent message-passing heads, each with its own
    (init projection, message MLP, GRU cell) -- never shared weights across
    heads, the entire point of the architecture. Each head runs `rounds`
    steps independently over the same adjacency, starting from the same raw
    per-vertex signal (degree, optionally + spectral coords). Final hidden
    states are concatenated (`[n, hidden_dim * num_heads]`) and projected
    through `out_proj` to `embed_dim` -- the multi-head-attention output
    projection, not a sequential composition.

    head_dim: hidden width per head (kept modest by default since num_heads
    heads run independently -- total compute is num_heads x rounds forward
    passes through similarly-sized cells, same order as a single wider head
    doing num_heads*rounds sequential steps, but structured as parallel votes
    instead of sequential depth).
    """

    def __init__(self, head_dim=16, embed_dim=8, num_heads=4, rounds=64,
                 reinject_degree_signal=True, use_spectral_coords=False,
                 spectral_coords_variant="adjacency"):
        super().__init__()
        self.head_dim = head_dim
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.rounds = rounds
        self.reinject_degree_signal = reinject_degree_signal
        self.use_spectral_coords = use_spectral_coords
        self.spectral_coords_variant = spectral_coords_variant

        init_in_dim = 3 if use_spectral_coords else 1
        # Separate init projection per head -- mirrors attention's separate
        # learned Q/K/V linear maps per head from the same input.
        self.head_init = nn.ModuleList([nn.Linear(init_in_dim, head_dim) for _ in range(num_heads)])
        self.head_msg_mlp = nn.ModuleList([
            nn.Sequential(nn.Linear(head_dim, head_dim), nn.ReLU()) for _ in range(num_heads)
        ])
        self.head_gru = nn.ModuleList([nn.GRUCell(head_dim, head_dim) for _ in range(num_heads)])

        # Output projection concatenated-heads -> embed_dim, i.e. attention's W_O.
        self.out_proj = nn.Linear(head_dim * num_heads, embed_dim)

    def forward(self, adj, return_per_head=False):
        """
        adj: [n,n] float tensor (0/1). Returns embeddings [n, embed_dim].
        If return_per_head=True, also returns the list of per-head final
        hidden states [n, head_dim] (used only by check_collapse(), never
        during training/inference) so each head's health can be inspected
        independently, not just the post-projection aggregate.
        """
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

        head_finals = []
        for head_idx in range(self.num_heads):
            degree_signal = self.head_init[head_idx](init_feat)
            h = degree_signal
            msg_mlp = self.head_msg_mlp[head_idx]
            gru = self.head_gru[head_idx]
            for _ in range(self.rounds):
                msg = (adj @ msg_mlp(h)) / degree
                h = gru(msg, h)
                if self.reinject_degree_signal:
                    h = h + degree_signal
            head_finals.append(h)

        concat = torch.cat(head_finals, dim=-1)
        out = self.out_proj(concat)
        if return_per_head:
            return out, head_finals
        return out


def train_multihead_gnn(
    n_train=300,
    c_range=(3, 20),
    hard_window=(4, 14),
    hard_window_prob=0.6,
    steps=1000,
    lr=1e-3,
    head_dim=16,
    embed_dim=8,
    num_heads=4,
    rounds=64,
    reinject_degree_signal=True,
    use_spectral_coords=False,
    spectral_coords_variant="adjacency",
    loss_fn="fisher",
    device="cpu",
    log_every=100,
    seed=0,
):
    """Mirrors train_spectral_nn's / train_multistage_nn's loop structure exactly."""
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)

    model = MultiHeadGNNEmbedder(
        head_dim=head_dim, embed_dim=embed_dim, num_heads=num_heads, rounds=rounds,
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
    """Phase 1 (learned): PCA top-2 readout, identical pattern to
    spectral_nn.py/multistage_nn.py's function of the same name -- kept as a
    thin duplicate (not an import) per this project's versioning discipline."""
    A = adj.detach().cpu().numpy() if isinstance(adj, torch.Tensor) else np.asarray(adj)
    n = A.shape[0]
    with torch.no_grad():
        adj_t = torch.as_tensor(A, dtype=torch.float32, device=device)
        embeddings = model(adj_t).cpu().numpy()

    mean = embeddings.mean(axis=0, keepdims=True)
    X = embeddings - mean
    cov = X.T @ X
    eigvals, eigvecs = np.linalg.eigh(cov)
    top2 = eigvecs[:, -2:]
    proj = X @ top2
    e1, e2 = proj[:, 0], proj[:, 1]
    return _two_eigvecs_to_coloring(e1, e2, n)


def multihead_gnn_3color(adj, model, device="cpu", propagation_iterations=None, max_component_size=18):
    """Full pipeline: learned phase 1 (multi-head embedder) + spectral_coloring.py's
    existing, already-validated phases 2-3 (_propagate, _cleanup)."""
    A = adj.detach().cpu().numpy() if isinstance(adj, torch.Tensor) else np.asarray(adj)
    n = A.shape[0]
    if propagation_iterations is None:
        propagation_iterations = max(1, int(np.ceil(np.log(max(n, 2)))))

    colors = nn_spectral_init(A, model, device=device)
    colors = _propagate(A, colors, propagation_iterations)
    ok, colors, method = _cleanup(A, colors, max_component_size=max_component_size)
    method = method.replace("spectral", "multihead_gnn")

    same = colors[:, None] == colors[None, :]
    conflicts = int((A.astype(bool) & same).sum() // 2)
    return {"success": ok, "colors": colors, "method": method, "conflicts": conflicts}


def check_collapse(model, n=300, c=10, device="cpu", seed=1):
    """
    Empirical collapse diagnostic, run PER HEAD (not just on the final
    concatenated+projected output) -- mandatory before trusting any loss
    curve. An aggregate-only check could hide 3 of 4 heads collapsing while
    one healthy head keeps the concatenated statistics looking non-degenerate,
    which would be a materially worse (and easy-to-miss) failure mode than
    whole-network collapse.
    """
    torch.manual_seed(seed)
    assignment, adj = create_planted_3col(n, c)
    adj = adj.float().to(device)
    with torch.no_grad():
        final_embed, head_finals = model(adj, return_per_head=True)

    print(f"check_collapse: n={n} c={c}, num_heads={model.num_heads}, rounds={model.rounds}")
    any_head_collapsed = False
    for head_idx, h in enumerate(head_finals):
        h = h.cpu().numpy()
        overall_std = h.std()
        class_means = [h[(assignment == k).cpu().numpy()].mean(axis=0) for k in range(3)]
        d01 = np.linalg.norm(class_means[0] - class_means[1])
        d02 = np.linalg.norm(class_means[0] - class_means[2])
        d12 = np.linalg.norm(class_means[1] - class_means[2])
        collapsed = overall_std < 1e-4 or max(d01, d02, d12) < 1e-4
        any_head_collapsed = any_head_collapsed or collapsed
        print(f"  head {head_idx}: std={overall_std:.6f}  class_mean_dists=({d01:.4f},{d02:.4f},{d12:.4f})"
              f"  {'COLLAPSED' if collapsed else 'ok'}")

    final = final_embed.detach().cpu().numpy()
    final_std = final.std()
    final_class_means = [final[(assignment == k).cpu().numpy()].mean(axis=0) for k in range(3)]
    fd01 = np.linalg.norm(final_class_means[0] - final_class_means[1])
    fd02 = np.linalg.norm(final_class_means[0] - final_class_means[2])
    fd12 = np.linalg.norm(final_class_means[1] - final_class_means[2])
    final_collapsed = final_std < 1e-4 or max(fd01, fd02, fd12) < 1e-4
    print(f"  FINAL (post-out_proj) std={final_std:.6f}  class_mean_dists=({fd01:.4f},{fd02:.4f},{fd12:.4f})"
          f"  {'COLLAPSED' if final_collapsed else 'ok'}")
    if any_head_collapsed and not final_collapsed:
        print("  *** WARNING: at least one individual head collapsed even though the "
              "final concatenated output looks fine -- inspect per-head lines above ***")
    return not (final_collapsed or any_head_collapsed)


if __name__ == "__main__":
    torch.manual_seed(0)
    m = MultiHeadGNNEmbedder(head_dim=16, embed_dim=8, num_heads=4, rounds=64)
    check_collapse(m, n=300, c=10)
