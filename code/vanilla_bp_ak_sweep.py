"""
STANDALONE 4th algorithm: vanilla (unreinforced, rho=0) BP + AK's FULL
pipeline (phase-2 propagation + phase-3 cleanup/peeling), + greedy_repair
safety net. This is the correct reading of "BP+AK": plain BP feeding AK's
post-processing, with NO reinforcement anywhere.

This file does not import, call, modify, or share output files with
bp_full_sweep_gpu.py. The existing three algorithms there (vanilla_bp,
bp_reinforce_cleanup, bp_ak_cleanup -- the last of which is REINFORCED
BP + AK, a genuinely different, already-reported result) are completely
untouched. This is an ADDITIONAL, independent measurement, not a
replacement or correction of anything already saved.

Restart budget: identical structure to bp_full_sweep_gpu.py's vanilla_bp
(up to n restarts, fresh seed each, max_iter=3n, chunked GPU batches so
easy instances exit early) -- but every candidate is checked via AK's
propagate+cleanup instead of repair alone.

Supports partial resume: if --out already has some instances saved for a
c-value (e.g. only 10 of a requested 100), running again with a larger
--n-inst continues from where it left off rather than restarting that
c-value's instances from scratch or skipping it entirely.
"""
import argparse
import base64
import json
import multiprocessing as mp
import os
import time

import numpy as np

from random_planted import create_planted_3col
from gc_utils import is_k_color
from repair_utils import count_conflicts, greedy_repair
from spectral_coloring import _propagate, _cleanup

CHUNK_SIZE = 100
N_CHECK_WORKERS = 16


def _build_directed_edges(A):
    n = A.shape[0]
    iu, ju = np.nonzero(np.triu(A, k=1))
    src = np.concatenate([iu, ju])
    dst = np.concatenate([ju, iu])
    m2 = len(src)
    half = m2 // 2
    reverse = np.concatenate([np.arange(half, m2), np.arange(0, half)])
    return src, dst, reverse


def _bp_batch_step_vanilla(A_np, src_t, dst_t, reverse_t, max_iter, seed, batch_size, q, device):
    """Plain (rho=0) BP -- no reinforcement term at all."""
    import torch
    n = A_np.shape[0]
    m2 = len(src_t)
    gen = torch.Generator(device=device)
    gen.manual_seed(seed)

    psi = torch.rand((batch_size, m2, q), generator=gen, device=device) * 0.2 + 0.9
    psi = psi * (1.0 / q)
    psi = psi + torch.randn((batch_size, m2, q), generator=gen, device=device) * 0.02
    psi = psi.clamp(min=1e-6)
    psi = psi / psi.sum(dim=-1, keepdim=True)

    eps = 1e-12
    prev_belief_log = torch.full((batch_size, n, q), -float(np.log(q)), device=device)

    for t in range(1, max_iter + 1):
        logterm = torch.log(torch.clamp(1.0 - psi, min=eps))
        S = torch.zeros((batch_size, n, q), device=device)
        S.index_add_(1, dst_t, logterm)
        belief_log = S - torch.logsumexp(S, dim=-1, keepdim=True)

        new_log_msg = S[:, src_t, :] - logterm[:, reverse_t, :]  # no reinforce term
        new_log_msg = new_log_msg - torch.logsumexp(new_log_msg, dim=-1, keepdim=True)
        new_msg = torch.exp(new_log_msg)

        psi_damped = 0.5 * psi + 0.5 * new_msg
        psi_damped = psi_damped / psi_damped.sum(dim=-1, keepdim=True)

        diff = (psi_damped - psi).abs().amax(dim=(1, 2))
        psi = psi_damped
        prev_belief_log = belief_log
        if (diff < 1e-7).all():
            break

    return torch.exp(prev_belief_log)


def _check_one(args):
    # NO _cleanup() call -- that function's residual-component step is exact
    # backtracking (brute-force search), explicitly removed from the
    # pipeline per instruction.
    A, colors, prop_iters = args
    colors = _propagate(A, colors, prop_iters)
    conf = count_conflicts(A, colors)
    if conf > 0:
        colors = greedy_repair(A, colors)
        conf = count_conflicts(A, colors)
    return conf == 0


def _check_batch(A, colors_batch, pool):
    batch_size = colors_batch.shape[0]
    n = A.shape[0]
    prop_iters = max(1, int(np.ceil(np.log(max(n, 2)))))
    args = [(A, colors_batch[b], prop_iters) for b in range(batch_size)]
    chunksize = max(1, batch_size // 8)
    results = pool.map(_check_one, args, chunksize=chunksize)
    success = np.asarray(results, dtype=bool)
    if success.any():
        first_idx = int(np.argmax(success))
        return success[:first_idx + 1], first_idx
    return success, None


def run_vanilla_bp_ak(A, n, device, pool, chunk_size=CHUNK_SIZE):
    import torch
    max_iter = 3 * n
    src, dst, reverse = _build_directed_edges(A)
    src_t = torch.as_tensor(src, device=device, dtype=torch.long)
    dst_t = torch.as_tensor(dst, device=device, dtype=torch.long)
    reverse_t = torch.as_tensor(reverse, device=device, dtype=torch.long)

    budget = n
    attempted = 0
    all_success = []
    while attempted < budget:
        bs = min(chunk_size, budget - attempted)
        belief = _bp_batch_step_vanilla(A, src_t, dst_t, reverse_t, max_iter, seed=attempted,
                                         batch_size=bs, q=3, device=device)
        colors_batch = belief.argmax(dim=-1).cpu().numpy()
        success, first_idx = _check_batch(A, colors_batch, pool)
        all_success.append(success)
        attempted += len(success)
        if first_idx is not None:
            break

    bitmask = np.concatenate(all_success)
    overall_success = bool(bitmask.any())
    restarts_used = int(np.argmax(bitmask)) + 1 if overall_success else attempted
    return overall_success, restarts_used, budget, bitmask


def pack_bitmask(bitmask):
    packed = np.packbits(bitmask)
    return {"shape": list(bitmask.shape), "data_b64": base64.b64encode(packed.tobytes()).decode("ascii")}


def build_c_values():
    coarse_low = [round(1.0 + 0.5 * i, 2) for i in range(7)]
    fine = [round(4.40 + 0.02 * i, 2) for i in range(16)]
    fine_full = sorted(set(fine + [4.69]))
    coarse_high = [round(5.0 + 0.5 * i, 2) for i in range(31)]
    return coarse_low + fine_full + coarse_high


def parse_args():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--n", type=int, required=True)
    p.add_argument("--c-values", type=float, nargs="+", default=None)
    p.add_argument("--n-inst", type=int, default=100)
    p.add_argument("--device", default=None)
    p.add_argument("--out", required=True)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--n-check-workers", type=int, default=N_CHECK_WORKERS)
    return p.parse_args()


def main():
    import torch
    args = parse_args()
    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    c_values = args.c_values if args.c_values is not None else build_c_values()
    torch.manual_seed(args.seed)

    if os.path.exists(args.out):
        with open(args.out) as f:
            results = json.load(f)
        print(f"resuming from existing {args.out}: c-values with data: {list(results.keys())}", flush=True)
    else:
        results = {}

    print(f"device={device}, n={args.n}, {len(c_values)} c-values, target {args.n_inst} instances each, "
          f"{args.n_check_workers} check-workers (persistent pool)", flush=True)

    with mp.Pool(args.n_check_workers) as pool:
      for c in c_values:
        c_key = str(c)
        existing = results.get(c_key, {})
        n_done = len(existing)

        if n_done >= args.n_inst:
            print(f"c={c}: already have {n_done}/{args.n_inst}, skipping", flush=True)
            continue

        # Replay RNG for instances [0, n_done) to keep instance i's graph
        # identical across resumes, then generate the NEW instances needed.
        torch.manual_seed(args.seed)
        for prior_c in c_values:
            if prior_c == c:
                break
            prior_n_done = len(results.get(str(prior_c), {}))
            for _ in range(max(prior_n_done, 0)):
                create_planted_3col(args.n, prior_c)
        for _ in range(n_done):
            create_planted_3col(args.n, c)

        t0 = time.time()
        n_success_new = 0
        for i in range(n_done, args.n_inst):
            _, adj = create_planted_3col(args.n, c)
            A = adj.numpy()

            ok, r, budget, bm = run_vanilla_bp_ak(A, args.n, device, pool)
            if ok:
                colors = None  # not retained; success/restart data is what's tracked
                n_success_new += ok

            existing[str(i)] = {"success": ok, "restarts": r, "budget": budget, "bitmask": pack_bitmask(bm)}
            print(f"  c={c} inst={i}: {'SUCCESS' if ok else 'FAIL'} (restarts={r})", flush=True)

        results[c_key] = existing
        dt = time.time() - t0
        total_ok = sum(v["success"] for v in existing.values())
        print(f"c={c}: {total_ok}/{len(existing)} ({dt:.0f}s for the {args.n_inst - n_done} new instances)", flush=True)

        with open(args.out, "w") as f:
            json.dump(results, f)

    print(f"\nsaved to {args.out}")


if __name__ == "__main__":
    mp.set_start_method("spawn", force=True)
    main()
