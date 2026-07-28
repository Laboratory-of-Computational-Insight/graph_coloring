"""
Multi-head (parallel) backbone + pairwise same/different head (2026-07-15) --
the other missing combination (see multistage_pairwise_nn.py's docstring for
the full rationale: pairwise same/different was the project's best single
result, multi-head was only ever paired with the PCA-readout target).

Backbone: identical structure to multihead_gnn.py's MultiHeadGNNEmbedder
(num_heads independent GRU cells -- own init/msg-MLP/GRU weights each, never
shared -- run the same rounds count in parallel on the same graph, final
hidden states concatenated then projected, mirroring multi-head attention).

Head: identical structure to pairwise_nn.py's PairwiseEmbedder.

New file, not a modification of either existing file, per project convention.
"""
import time

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from random_planted import create_planted_3col
from spectral_coloring import _two_eigvecs_to_coloring, _propagate, _cleanup, raw_spectral_coordinates


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


class MultiHeadPairwiseEmbedder(nn.Module):
    def __init__(self, head_dim=16, embed_dim=8, num_heads=4, rounds=64,
                 reinject_degree_signal=True, use_spectral_coords=False, spectral_coords_variant="adjacency",
                 num_spectral_eigvecs=2, per_dim_scale=False, use_common_neighbors=False, head="distance"):
        super().__init__()
        self.head_dim = head_dim
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.rounds = rounds
        self.reinject_degree_signal = reinject_degree_signal
        self.use_spectral_coords = use_spectral_coords
        self.spectral_coords_variant = spectral_coords_variant
        self.num_spectral_eigvecs = num_spectral_eigvecs
        self.per_dim_scale = per_dim_scale
        self.use_common_neighbors = use_common_neighbors
        self.head = head

        k = num_spectral_eigvecs
        init_in_dim = (k if use_spectral_coords else 0) + 1
        self.head_init = nn.ModuleList([nn.Linear(init_in_dim, head_dim) for _ in range(num_heads)])
        self.head_msg_mlp = nn.ModuleList([
            nn.Sequential(nn.Linear(head_dim, head_dim), nn.ReLU()) for _ in range(num_heads)
        ])
        self.head_gru = nn.ModuleList([nn.GRUCell(head_dim, head_dim) for _ in range(num_heads)])
        self.out_proj = nn.Linear(head_dim * num_heads, embed_dim)

        if head == "distance":
            self.log_scale = nn.Parameter(torch.zeros(embed_dim) if per_dim_scale else torch.tensor(0.0))
            self.bias = nn.Parameter(torch.tensor(0.0))
            if use_common_neighbors:
                self.cn_weight = nn.Parameter(torch.tensor(0.0))
        elif head == "mlp":
            mlp_in_dim = embed_dim * 3 + (1 if use_common_neighbors else 0)
            self.head_mlp = nn.Sequential(nn.Linear(mlp_in_dim, head_dim), nn.ReLU(), nn.Linear(head_dim, 1))
        else:
            raise ValueError(f"unknown head: {head!r}")

    def forward(self, adj):
        n = adj.shape[0]
        device = adj.device
        degree = adj.sum(dim=1, keepdim=True).clamp(min=1.0)
        degree_norm = (degree - degree.mean()) / (degree.std() + 1e-6)

        if self.use_spectral_coords:
            with torch.no_grad():
                k = self.num_spectral_eigvecs
                coords = (raw_spectral_coordinates(adj, variant=self.spectral_coords_variant) if k == 2
                          else _raw_coords_k(adj, self.spectral_coords_variant, k))
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
        return self.out_proj(concat)

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


def train_multihead_pairwise_nn(
    n_train=300, c_range=(1, 8), steps=800, lr=1e-3,
    head_dim=16, embed_dim=8, num_heads=4, rounds=32,
    reinject_degree_signal=True, use_spectral_coords=True, spectral_coords_variant="adjacency",
    num_spectral_eigvecs=2, per_dim_scale=False, use_common_neighbors=False, head="distance",
    n_pairs=4000, fisher_weight=2.0, entropy_weight=0.0, device="cpu", log_every=100, seed=0,
):
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)

    model = MultiHeadPairwiseEmbedder(
        head_dim=head_dim, embed_dim=embed_dim, num_heads=num_heads, rounds=rounds,
        reinject_degree_signal=reinject_degree_signal, use_spectral_coords=use_spectral_coords,
        spectral_coords_variant=spectral_coords_variant, num_spectral_eigvecs=num_spectral_eigvecs,
        per_dim_scale=per_dim_scale, use_common_neighbors=use_common_neighbors, head=head,
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


def multihead_pairwise_nn_3color(adj, model, device="cpu", propagation_iterations=None, max_component_size=18):
    A = adj.detach().cpu().numpy() if isinstance(adj, torch.Tensor) else np.asarray(adj)
    n = A.shape[0]
    if propagation_iterations is None:
        propagation_iterations = max(1, int(np.ceil(np.log(max(n, 2)))))

    colors = nn_pairwise_init(A, model, device=device)
    colors = _propagate(A, colors, propagation_iterations)
    ok, colors, method = _cleanup(A, colors, max_component_size=max_component_size)
    method = method.replace("spectral", "multihead_pairwise")

    same = colors[:, None] == colors[None, :]
    conflicts = int((A.astype(bool) & same).sum() // 2)
    return {"success": ok, "colors": colors, "method": method, "conflicts": conflicts}


if __name__ == "__main__":
    torch.manual_seed(0)
    m = MultiHeadPairwiseEmbedder(head_dim=16, embed_dim=8, num_heads=4, rounds=32)
    check_collapse(m, n=300, c_values=(10,), n_pairs=1000)
