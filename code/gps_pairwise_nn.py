"""
GraphGPS (local message passing + global Transformer attention) backbone +
pairwise same/different head (2026-07-15) -- a 4th backbone option alongside
tonight's single-cell (pairwise_nn.py), multi-stage (multistage_pairwise_nn.py),
and multi-head (multihead_pairwise_nn.py) backbones, all sharing the same
pairwise-head framework.

Why pairwise same/different as the target here too (not a per-node distance
or 3-way color target): it's been the single best-performing target of the
whole project (sidesteps the color-labeling symmetry that hurts direct 3-way
prediction), and it fits GraphGPS naturally -- GPSConv already outputs
per-node embeddings [n, hidden], exactly the input shape pairwise_nn.py's
pair_logits/pairwise_prob_matrix expect.

Backbone: `num_layers` GPSConv layers (rugPyTorch Geometric's official
implementation -- local GINEConv message passing + global multi-head
attention per layer, combined, per Rampasek et al. "Recipe for a General,
Powerful, Scalable Graph Transformer"). Global attention is a genuinely
different reasoning mechanism from every custom GRU-based backbone used
elsewhere tonight -- every node can attend to every other node directly in
one layer, not just propagate through local message passing over many
rounds.

Our graphs are unweighted/simple (no edge features) -- GINEConv requires
edge_attr, so a single learned constant edge-type embedding is broadcast to
every edge (there being no real edge features to use).
"""
import time

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GPSConv, GINEConv

from random_planted import create_planted_3col
from spectral_coloring import _two_eigvecs_to_coloring, _propagate, _cleanup, raw_spectral_coordinates, _spectral_init


def _classical_coloring_onehot(adj):
    """
    Discrete classical coloring signal (2026-07-16, direct answer to "did you
    try feeding the classical algorithm's actual discrete output, not just
    raw eigenvector coordinates"): phase 1 (spectral init) + phase 2
    (propagation) ONLY -- deliberately skips phase 3 (cleanup/DSATUR
    fallback), which is expensive and, per diagnose_baseline_failure.py's
    finding, essentially always fails at c=7-9 and falls back to returning
    this exact phase-1+2 coloring unchanged anyway (just after wasting time
    on a doomed DSATUR search) -- so using phase 1+2 directly is both cheaper
    and no less informative in the regime this is meant for.

    One-hot encoded per vertex ([n,3], centered to [-1/3, 2/3] range) --
    a genuinely different kind of signal from raw_spectral_coordinates: a
    concrete discrete guess (even though noisy/likely wrong per-vertex) that
    the network can incorporate however it learns is useful, rather than raw
    continuous eigenvector coordinates it has to threshold itself.
    """
    A = adj.detach().cpu().numpy() if isinstance(adj, torch.Tensor) else np.asarray(adj, dtype=np.float64)
    n = A.shape[0]
    colors = _spectral_init(A, n)
    propagation_iterations = max(1, int(np.ceil(np.log(max(n, 2)))))
    colors = _propagate(A, colors, propagation_iterations)
    onehot = np.eye(3, dtype=np.float64)[colors]  # [n, 3], each row sums to 1
    return onehot - (1.0 / 3.0)  # center, same spirit as the other normalized features


def _raw_coords_k(adj, variant, k, r=None):
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


def _fisher_term(embeddings, assignment, k=3):
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


def _cn_matrix_norm(adj):
    cn = adj @ adj
    return (cn - cn.mean()) / (cn.std() + 1e-6)


def _adj_to_edge_index(adj):
    """Dense n x n 0/1 adjacency -> PyG edge_index [2, num_edges] (both
    directions, standard for undirected-graph message passing)."""
    return adj.nonzero(as_tuple=False).t().contiguous()


class GPSPairwiseEmbedder(nn.Module):
    def __init__(self, hidden_dim=32, embed_dim=8, num_layers=4, heads=4,
                 use_spectral_coords=False, spectral_coords_variant="adjacency", num_spectral_eigvecs=2,
                 per_dim_scale=False, use_common_neighbors=False, use_classical_coloring=False, head="distance"):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.embed_dim = embed_dim
        self.num_layers = num_layers
        self.heads = heads
        self.use_spectral_coords = use_spectral_coords
        self.spectral_coords_variant = spectral_coords_variant
        self.num_spectral_eigvecs = num_spectral_eigvecs
        self.per_dim_scale = per_dim_scale
        self.use_common_neighbors = use_common_neighbors
        self.use_classical_coloring = use_classical_coloring
        self.head = head

        k = num_spectral_eigvecs
        init_in_dim = (k if use_spectral_coords else 0) + (3 if use_classical_coloring else 0) + 1
        self.init = nn.Linear(init_in_dim, hidden_dim)

        # Constant learned edge-type embedding -- our graphs have no real
        # edge features, GINEConv still requires an edge_attr of width hidden_dim.
        self.edge_embed = nn.Parameter(torch.randn(hidden_dim) * 0.1)

        self.convs = nn.ModuleList()
        for _ in range(num_layers):
            local_gnn = GINEConv(nn.Sequential(
                nn.Linear(hidden_dim, hidden_dim), nn.ReLU(), nn.Linear(hidden_dim, hidden_dim),
            ))
            self.convs.append(GPSConv(channels=hidden_dim, conv=local_gnn, heads=heads, attn_type="multihead"))

        self.readout = nn.Linear(hidden_dim, embed_dim)

        if head == "distance":
            self.log_scale = nn.Parameter(torch.zeros(embed_dim) if per_dim_scale else torch.tensor(0.0))
            self.bias = nn.Parameter(torch.tensor(0.0))
            if use_common_neighbors:
                self.cn_weight = nn.Parameter(torch.tensor(0.0))
        elif head == "mlp":
            mlp_in_dim = embed_dim * 3 + (1 if use_common_neighbors else 0)
            self.head_mlp = nn.Sequential(nn.Linear(mlp_in_dim, hidden_dim), nn.ReLU(), nn.Linear(hidden_dim, 1))
        else:
            raise ValueError(f"unknown head: {head!r}")

    def forward(self, adj):
        n = adj.shape[0]
        device = adj.device
        degree = adj.sum(dim=1, keepdim=True).clamp(min=1.0)
        degree_norm = (degree - degree.mean()) / (degree.std() + 1e-6)

        feats = [degree_norm]
        if self.use_spectral_coords:
            with torch.no_grad():
                k = self.num_spectral_eigvecs
                coords = (raw_spectral_coordinates(adj, variant=self.spectral_coords_variant) if k == 2
                          else _raw_coords_k(adj, self.spectral_coords_variant, k))
                coords_t = torch.as_tensor(coords, dtype=degree_norm.dtype, device=device)
                coords_norm = (coords_t - coords_t.mean(dim=0, keepdim=True)) / (coords_t.std(dim=0, keepdim=True) + 1e-6)
            feats.append(coords_norm)
        if self.use_classical_coloring:
            with torch.no_grad():
                onehot = _classical_coloring_onehot(adj)
                onehot_t = torch.as_tensor(onehot, dtype=degree_norm.dtype, device=device)
            feats.append(onehot_t)
        init_feat = torch.cat(feats, dim=1) if len(feats) > 1 else feats[0]

        x = self.init(init_feat)
        edge_index = _adj_to_edge_index(adj)
        num_edges = edge_index.shape[1]
        edge_attr = self.edge_embed.unsqueeze(0).expand(num_edges, -1)
        batch = torch.zeros(n, dtype=torch.long, device=device)

        for conv in self.convs:
            x = conv(x, edge_index, batch=batch, edge_attr=edge_attr)

        return self.readout(x)

    def pair_logits(self, emb_u, emb_v, cn=None):
        if self.head == "distance":
            diff2 = (emb_u - emb_v).pow(2)
            dist2 = (diff2 * torch.exp(self.log_scale)).sum(-1) if self.per_dim_scale \
                else diff2.sum(-1) * torch.exp(self.log_scale)
            logit = self.bias - dist2
            if self.use_common_neighbors and cn is not None:
                logit = logit + self.cn_weight * cn
            return logit
        else:
            diff = emb_u - emb_v
            feat = torch.cat([emb_u, emb_v, diff.abs()], dim=-1)
            if self.use_common_neighbors:
                cn_feat = cn if cn is not None else torch.zeros_like(feat[..., :1].squeeze(-1))
                feat = torch.cat([feat, cn_feat.unsqueeze(-1)], dim=-1)
            return self.head_mlp(feat).squeeze(-1)

    def pairwise_prob_matrix(self, embeddings, cn=None):
        n = embeddings.shape[0]
        if self.head == "distance":
            if self.per_dim_scale:
                scaled = embeddings * torch.sqrt(torch.exp(self.log_scale))
                logits = self.bias - torch.cdist(scaled, scaled, p=2).pow(2)
            else:
                logits = self.bias - torch.exp(self.log_scale) * torch.cdist(embeddings, embeddings, p=2).pow(2)
            if self.use_common_neighbors and cn is not None:
                logits = logits + self.cn_weight * cn
        else:
            eu = embeddings.unsqueeze(1).expand(n, n, -1)
            ev = embeddings.unsqueeze(0).expand(n, n, -1)
            feat = torch.cat([eu, ev, (eu - ev).abs()], dim=-1)
            if self.use_common_neighbors:
                cn_feat = cn if cn is not None else torch.zeros(n, n, dtype=embeddings.dtype, device=embeddings.device)
                feat = torch.cat([feat, cn_feat.unsqueeze(-1)], dim=-1)
            logits = self.head_mlp(feat).squeeze(-1)
        return torch.sigmoid(logits)


def pairwise_bce_loss(model, embeddings, assignment, adj, n_pairs=4000, rng=None,
                       variance_weight=1.0, variance_target=1.0, fisher_weight=0.0, entropy_weight=0.0):
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
        cn_norm = _cn_matrix_norm(adj)
        cn_pairs = cn_norm[idx_i, idx_j]

    logits = model.pair_logits(embeddings[idx_i], embeddings[idx_j], cn=cn_pairs)
    bce = F.binary_cross_entropy_with_logits(logits, labels)

    std = embeddings.std(dim=0)
    variance_penalty = torch.clamp(variance_target - std, min=0).mean()

    loss = bce + variance_weight * variance_penalty
    if fisher_weight > 0:
        loss = loss + fisher_weight * _fisher_term(embeddings, assignment)
    if entropy_weight > 0:
        probs = torch.sigmoid(logits).clamp(1e-6, 1 - 1e-6)
        entropy = -(probs * probs.log() + (1 - probs) * (1 - probs).log())
        loss = loss + entropy_weight * entropy.mean()
    return loss


def train_gps_pairwise_nn(
    n_train=300, c_range=(1, 8), steps=800, lr=1e-3,
    hidden_dim=32, embed_dim=8, num_layers=4, heads=4,
    use_spectral_coords=True, spectral_coords_variant="adjacency", num_spectral_eigvecs=2,
    per_dim_scale=False, use_common_neighbors=False, use_classical_coloring=False, head="distance",
    n_pairs=4000, fisher_weight=2.0, entropy_weight=0.0, device="cpu", log_every=100, seed=0,
):
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)

    model = GPSPairwiseEmbedder(
        hidden_dim=hidden_dim, embed_dim=embed_dim, num_layers=num_layers, heads=heads,
        use_spectral_coords=use_spectral_coords, spectral_coords_variant=spectral_coords_variant,
        num_spectral_eigvecs=num_spectral_eigvecs, per_dim_scale=per_dim_scale,
        use_common_neighbors=use_common_neighbors, use_classical_coloring=use_classical_coloring, head=head,
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
        loss = pairwise_bce_loss(model, embeddings, assignment, adj, n_pairs=n_pairs, rng=rng,
                                  fisher_weight=fisher_weight, entropy_weight=entropy_weight)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        losses.append(loss.item())

        if step % log_every == 0:
            recent = losses[-log_every:]
            print(f"step {step:5d}/{steps}  loss={sum(recent)/len(recent):.4f}  ({time.time()-t0:.0f}s elapsed)")

    return model


@torch.no_grad()
def check_collapse(model, n=300, c_values=(6, 10, 14), n_pairs=4000, seed=123, device="cpu"):
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
        cn_pairs = _cn_matrix_norm(adj)[idx_i, idx_j] if model.use_common_neighbors else None
        probs = torch.sigmoid(model.pair_logits(embeddings[idx_i], embeddings[idx_j], cn=cn_pairs))
        preds = (probs > 0.5).float()

        acc = (preds == labels).float().mean().item()
        baseline = 1.0 - labels.mean().item()
        print(f"  accuracy={acc:.3f}  'always-different' baseline={baseline:.3f}  "
              f"pred_same_frac={preds.mean().item():.3f} (actual same_frac={labels.mean().item():.3f})  "
              f"prob_std={probs.std().item():.4f}")


def nn_pairwise_init(adj, model, device="cpu"):
    A = adj.detach().cpu().numpy() if isinstance(adj, torch.Tensor) else np.asarray(adj)
    n = A.shape[0]
    with torch.no_grad():
        adj_t = torch.as_tensor(A, dtype=torch.float32, device=device)
        embeddings = model(adj_t)
        cn = _cn_matrix_norm(adj_t) if model.use_common_neighbors else None
        P = model.pairwise_prob_matrix(embeddings, cn=cn).cpu().numpy()

    np.fill_diagonal(P, 0.0)
    S = 2.0 * P - 1.0
    S = (S + S.T) / 2.0
    eigvals, eigvecs = np.linalg.eigh(S)
    e1, e2 = eigvecs[:, -1], eigvecs[:, -2]
    return _two_eigvecs_to_coloring(e1, e2, n)


def gps_pairwise_nn_3color(adj, model, device="cpu", propagation_iterations=None, max_component_size=18):
    A = adj.detach().cpu().numpy() if isinstance(adj, torch.Tensor) else np.asarray(adj)
    n = A.shape[0]
    if propagation_iterations is None:
        propagation_iterations = max(1, int(np.ceil(np.log(max(n, 2)))))

    colors = nn_pairwise_init(A, model, device=device)
    colors = _propagate(A, colors, propagation_iterations)
    ok, colors, method = _cleanup(A, colors, max_component_size=max_component_size)
    method = method.replace("spectral", "gps_pairwise")

    same = colors[:, None] == colors[None, :]
    conflicts = int((A.astype(bool) & same).sum() // 2)
    return {"success": ok, "colors": colors, "method": method, "conflicts": conflicts}


if __name__ == "__main__":
    torch.manual_seed(0)
    m = GPSPairwiseEmbedder(hidden_dim=32, embed_dim=8, num_layers=4, heads=4)
    check_collapse(m, n=300, c_values=(10,), n_pairs=1000)
