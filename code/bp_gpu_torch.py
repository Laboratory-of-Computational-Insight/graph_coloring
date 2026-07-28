"""
Torch/GPU-native variant of belief_propagation_coloring.py's bp_reinforced_
coloring, with one key design change beyond a straight port: it runs
`batch_size` INDEPENDENT random restarts simultaneously as one vectorized
tensor computation, instead of one restart per call. This is the actual
useful GPU workload here -- BP on a single n=1000 graph is too small to
saturate a GPU on its own, but this project routinely burns 100-1000
sequential restarts per hard instance (see bp_distance_escalation.py), and
those restarts are fully independent (different random seed each), which is
exactly what a GPU batch dimension is for.

Falls back to CPU if CUDA isn't available (still correct, just not fast --
useful for testing the math on a machine with no GPU before shipping to one
that has it).
"""
import numpy as np
import torch


def _build_directed_edges(A):
    """Same as belief_propagation_coloring.py's version -- kept independent
    here (no numpy/torch mixing needed at call sites) since this is the one
    piece of graph-structure setup that isn't itself a per-restart tensor op
    (same edge list is shared by every restart in the batch)."""
    n = A.shape[0]
    iu, ju = np.nonzero(np.triu(A, k=1))
    src = np.concatenate([iu, ju])
    dst = np.concatenate([ju, iu])
    m2 = len(src)
    half = m2 // 2
    reverse = np.concatenate([np.arange(half, m2), np.arange(0, half)])
    return src, dst, reverse


def bp_reinforced_coloring_batch_gpu(A, batch_size=100, q=3, max_iter=3000, damping=0.5,
                                      rho=0.003, tol=1e-7, seed=0, eps=1e-12,
                                      device=None, init_bias=None, bias_strength=0.3):
    """Runs `batch_size` independent BP+reinforcement chains in parallel.

    Returns (belief (batch_size, n, q) torch tensor, converged (batch_size,)
    bool torch tensor). All batch elements run the FULL max_iter unless
    every single one has converged (a per-element early stop would break
    the vectorization; the loop only exits early once the whole batch is
    done, same total cost as the numpy version's worst-of-batch case).

    init_bias (optional, shape (n,) int in [0,q)): same semantics as the
    numpy version -- a per-vertex classical guess to bias every batch
    element's init toward (the noise on top still differs per element).
    """
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    n = A.shape[0]
    A_np = A.detach().cpu().numpy() if isinstance(A, torch.Tensor) else np.asarray(A)
    src, dst, reverse = _build_directed_edges(A_np)
    m2 = len(src)
    src_t = torch.as_tensor(src, device=device, dtype=torch.long)
    dst_t = torch.as_tensor(dst, device=device, dtype=torch.long)
    reverse_t = torch.as_tensor(reverse, device=device, dtype=torch.long)

    gen = torch.Generator(device=device)
    gen.manual_seed(seed)

    psi = torch.rand((batch_size, m2, q), generator=gen, device=device) * 0.2 + 0.9
    psi = psi * (1.0 / q)
    psi = psi + torch.randn((batch_size, m2, q), generator=gen, device=device) * 0.02
    if init_bias is not None:
        bias_onehot = torch.zeros((n, q), device=device)
        bias_idx = torch.as_tensor(init_bias, device=device, dtype=torch.long)
        bias_onehot[torch.arange(n, device=device), bias_idx] = 1.0
        psi = psi + bias_strength * bias_onehot[src_t].unsqueeze(0)  # broadcast over batch
    psi = psi.clamp(min=1e-6)
    psi = psi / psi.sum(dim=-1, keepdim=True)

    prev_belief_log = torch.full((batch_size, n, q), -float(np.log(q)), device=device)
    converged = torch.zeros(batch_size, dtype=torch.bool, device=device)

    for t in range(1, max_iter + 1):
        logterm = torch.log(torch.clamp(1.0 - psi, min=eps))  # (batch, m2, q)

        S = torch.zeros((batch_size, n, q), device=device)
        S.index_add_(1, dst_t, logterm)

        belief_log = S - torch.logsumexp(S, dim=-1, keepdim=True)

        reinforce = rho * t * belief_log
        new_log_msg = S[:, src_t, :] - logterm[:, reverse_t, :] + reinforce[:, src_t, :]
        new_log_msg = new_log_msg - torch.logsumexp(new_log_msg, dim=-1, keepdim=True)
        new_msg = torch.exp(new_log_msg)

        psi_damped = damping * psi + (1 - damping) * new_msg
        psi_damped = psi_damped / psi_damped.sum(dim=-1, keepdim=True)

        diff = (psi_damped - psi).abs().amax(dim=(1, 2))  # (batch,) per-restart max diff
        psi = psi_damped
        prev_belief_log = belief_log

        newly_converged = diff < tol
        if newly_converged.all():
            converged[:] = True
            break

    belief = torch.exp(prev_belief_log)
    return belief, converged


if __name__ == "__main__":
    import sys
    import time

    sys.path.insert(0, ".")
    from random_planted import create_planted_3col
    from repair_utils import count_conflicts, greedy_repair

    torch.manual_seed(0)
    c = float(sys.argv[1]) if len(sys.argv) > 1 else 8
    n = int(sys.argv[2]) if len(sys.argv) > 2 else 1000
    batch = int(sys.argv[3]) if len(sys.argv) > 3 else 100

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"device={device}, c={c}, n={n}, batch={batch}")

    _, adj = create_planted_3col(n, c)
    A = adj.numpy()

    t0 = time.time()
    belief, converged = bp_reinforced_coloring_batch_gpu(A, batch_size=batch, max_iter=3000, device=device)
    dt = time.time() - t0

    colors_batch = belief.argmax(dim=-1).cpu().numpy()  # (batch, n)
    n_success = 0
    first_success_idx = None
    for b in range(batch):
        colors = colors_batch[b]
        conf = count_conflicts(A, colors)
        if conf > 0:
            colors = greedy_repair(A, colors)
            conf = count_conflicts(A, colors)
        if conf == 0:
            n_success += 1
            if first_success_idx is None:
                first_success_idx = b

    print(f"batch of {batch} restarts in {dt:.1f}s ({dt/batch:.3f}s/restart) -- "
          f"{n_success}/{batch} succeeded, first success at index {first_success_idx}")
