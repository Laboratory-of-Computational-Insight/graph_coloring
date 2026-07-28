"""
Full n=1000/2000, c=1-20 BP sweep, GPU-batched, three algorithms:

  vanilla_bp:          rho=0, up to n restarts (fresh seed each), max_iter=3n,
                        greedy_repair only (no AK cleanup).
  bp_reinforce_cleanup: up to n/5 OUTER restarts (fresh seed each); for each
                        outer seed, try all 5 established rho values
                        {0.001,0.003,0.01,0.03,0.1} (damping=0.5 fixed) --
                        n/5 x 5 = n total attempts, same total budget as
                        vanilla. AK's phase-3 cleanup (peeling/backtrack) as
                        postprocessing, greedy_repair as final safety net.
  bp_ak_cleanup:       same restart ladder as bp_reinforce_cleanup, but ALSO
                        runs AK's phase-2 propagation (least-popular-neighbor
                        recoloring) before phase-3 cleanup -- the full
                        AK pipeline with BP swapped in for phase 1, spectral
                        eigenvectors never used anywhere.

GPU batching is CHUNKED (chunk_size restarts/outer-restarts computed as one
vectorized torch call at a time), so easy instances that succeed in the
first chunk don't pay for the full n-sized budget -- same early-exit benefit
CPU multiprocessing.imap_unordered gave us, but with GPU-level throughput
per chunk.

Compact save format per (n, c, instance, algo): success bool, restarts_used,
restarts_budget, and a base64-packed bitmask of every attempt actually made
(chunks before the success are all-fail by construction; chunks after a
found success are simply not attempted) -- so "which worked and which
didn't" is fully reconstructable without storing full color arrays.
"""
import argparse
import base64
import json
import multiprocessing as mp
import time

import numpy as np
# torch is imported LAZILY inside each function that actually needs it, not
# at module level: this module is re-imported wholesale in every spawned
# check-worker (Windows multiprocessing has no fork/COW), and workers only
# ever call _check_one -- pure numpy, no GPU/torch use at all. A module-level
# `import torch` was making all 16 check-workers load CUDA runtime DLLs
# (cublas/cufft/curand/cusolver) on startup for no reason, which under
# concurrent load can exhaust the Windows paging file (WinError 1455).

from random_planted import create_planted_3col
from gc_utils import is_k_color
from repair_utils import count_conflicts, greedy_repair
from spectral_coloring import _propagate, _cleanup

RHO_VALUES = [0.001, 0.003, 0.01, 0.03, 0.1]
CHUNK_SIZE = 100
N_CHECK_WORKERS = 16


def _build_directed_edges(A):
    """Pure-numpy, no torch -- kept local (not imported from bp_gpu_torch.py)
    so check-workers never transitively pull in torch via this import."""
    n = A.shape[0]
    iu, ju = np.nonzero(np.triu(A, k=1))
    src = np.concatenate([iu, ju])
    dst = np.concatenate([ju, iu])
    m2 = len(src)
    half = m2 // 2
    reverse = np.concatenate([np.arange(half, m2), np.arange(0, half)])
    return src, dst, reverse


def _check_one(args):
    # NO _cleanup() call -- that function's residual-component step is exact
    # backtracking (brute-force search), explicitly removed from the
    # pipeline per instruction. use_cleanup is accepted for call-site
    # compatibility but no longer does anything.
    A, colors, use_propagate, use_cleanup, prop_iters, repair_passes = args
    if use_propagate:
        colors = _propagate(A, colors, prop_iters)
    conf = count_conflicts(A, colors)
    if conf > 0:
        colors = greedy_repair(A, colors, max_passes=repair_passes)
        conf = count_conflicts(A, colors)
    return conf == 0


def _bp_batch_step(A_np, src_t, dst_t, reverse_t, rho_t, damping, max_iter, seed, batch_size, q, device):
    """Core batched BP+reinforcement loop. rho_t: scalar or (batch_size,) tensor
    (per-batch-element rho, for the reinforced variant's rho sub-restarts)."""
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
    if not torch.is_tensor(rho_t):
        rho_t = torch.full((batch_size,), rho_t, device=device)
    rho_bc = rho_t.view(batch_size, 1, 1)

    prev_belief_log = torch.full((batch_size, n, q), -float(np.log(q)), device=device)

    for t in range(1, max_iter + 1):
        logterm = torch.log(torch.clamp(1.0 - psi, min=eps))
        S = torch.zeros((batch_size, n, q), device=device)
        S.index_add_(1, dst_t, logterm)
        belief_log = S - torch.logsumexp(S, dim=-1, keepdim=True)

        reinforce = rho_bc * t * belief_log
        new_log_msg = S[:, src_t, :] - logterm[:, reverse_t, :] + reinforce[:, src_t, :]
        new_log_msg = new_log_msg - torch.logsumexp(new_log_msg, dim=-1, keepdim=True)
        new_msg = torch.exp(new_log_msg)

        psi_damped = damping * psi + (1 - damping) * new_msg
        psi_damped = psi_damped / psi_damped.sum(dim=-1, keepdim=True)

        diff = (psi_damped - psi).abs().amax(dim=(1, 2))
        psi = psi_damped
        prev_belief_log = belief_log
        if (diff < 1e-7).all():
            break

    return torch.exp(prev_belief_log)  # (batch_size, n, q)


def _check_batch(A, colors_batch, use_propagate, use_cleanup, prop_iters, pool, repair_passes=50):
    """CPU-side postprocessing + conflict check for a batch of candidate
    colorings, parallelized across `pool`'s workers. `pool` is PERSISTENT
    across the whole run (created once in main(), not per instance --
    Windows has no fork/COW, so spawning a fresh 16-process pool per
    instance was the actual bottleneck, not the checking work itself).
    A is passed per-task, but since chunksize>1 batches several tasks into
    one pickled message and every tuple in a chunk references the SAME A
    object, pickle's memoization serializes it once per chunk, not once per
    candidate. Returns (success_bitmask (batch,), first_success_idx or None)
    -- earliest (lowest-index) success is reported, matching the old
    sequential version's "first success in seed order" semantics."""
    batch_size = colors_batch.shape[0]
    args = [(A, colors_batch[b], use_propagate, use_cleanup, prop_iters, repair_passes) for b in range(batch_size)]
    chunksize = max(1, batch_size // 8)
    results = pool.map(_check_one, args, chunksize=chunksize)
    success = np.asarray(results, dtype=bool)
    if success.any():
        first_idx = int(np.argmax(success))
        return success[:first_idx + 1], first_idx
    return success, None


def run_vanilla_bp(A, n, device, pool, chunk_size=CHUNK_SIZE, repair_passes=50):
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
        belief = _bp_batch_step(A, src_t, dst_t, reverse_t, 0.0, 0.5, max_iter, seed=attempted, batch_size=bs, q=3, device=device)
        colors_batch = belief.argmax(dim=-1).cpu().numpy()
        success, first_idx = _check_batch(A, colors_batch, use_propagate=False, use_cleanup=False, prop_iters=0, pool=pool, repair_passes=repair_passes)
        all_success.append(success)
        attempted += len(success)
        if first_idx is not None:
            break

    bitmask = np.concatenate(all_success)
    overall_success = bool(bitmask.any())
    restarts_used = int(np.argmax(bitmask)) + 1 if overall_success else attempted
    return overall_success, restarts_used, budget, bitmask


def _finalize_reinforced_result(all_success, n_rho, outer_budget):
    """Shared bitmask -> (success, restarts_used, budget, bitmask_2d, rho_at_success,
    info) reduction, used by both pipelines in run_reinforced_bp_both."""
    bitmask_flat = np.concatenate(all_success) if all_success else np.zeros(0, dtype=bool)
    pad = (-len(bitmask_flat)) % n_rho
    if pad:
        bitmask_flat = np.concatenate([bitmask_flat, np.zeros(pad, dtype=bool)])
    bitmask_2d = bitmask_flat.reshape(-1, n_rho)

    overall_success = bool(bitmask_flat.any())
    if overall_success:
        flat_idx = int(np.argmax(bitmask_flat))
        outer_restart_at_success = flat_idx // n_rho + 1
        rho_idx_at_success = flat_idx % n_rho
        rho_at_success = RHO_VALUES[rho_idx_at_success]
        total_subrestarts_used = flat_idx + 1
    else:
        outer_restart_at_success = None
        rho_idx_at_success = None
        rho_at_success = None
        total_subrestarts_used = len(bitmask_flat)

    info = {
        "outer_restarts_budget": outer_budget,
        "rho_values": RHO_VALUES,
        "outer_restarts_attempted": bitmask_2d.shape[0],
        "outer_restart_at_success": outer_restart_at_success,
        "rho_idx_at_success": rho_idx_at_success,
        "total_subrestarts_used": total_subrestarts_used,
    }
    return overall_success, total_subrestarts_used, outer_budget * n_rho, bitmask_2d, rho_at_success, info


def run_reinforced_bp_both(A, n, device, pool, chunk_outer=20, repair_passes=50):
    """Computes the SAME BP+reinforcement restart ladder ONCE per chunk, then
    checks it under BOTH postprocessing pipelines independently:
      "cleanup_only"       = bp_reinforce_cleanup (AK phase-3 only)
      "propagate_cleanup"  = bp_ak_cleanup (AK phase-2 propagate + phase-3)
    Previously these were two separate calls to what's now this function,
    each recomputing an IDENTICAL GPU BP trajectory (same seeds, same rho
    schedule) just to apply a different postprocessing step -- pure waste.
    Each pipeline tracks its own bitmask/success/restart-count independently
    (propagate CAN change which candidates pass, so they must not be
    assumed to tie), but the loop stops once BOTH have found a success (or
    the budget is exhausted), instead of running each to its own separate
    stopping point -- collapsing what were two sequential searches into one
    shared one.

    Returns (result_cleanup_only, result_propagate_cleanup), each a 6-tuple
    matching the old run_reinforced_bp's return signature.
    """
    import torch
    max_iter = 3 * n
    n_ = A.shape[0]
    prop_iters = max(1, int(np.ceil(np.log(max(n_, 2)))))
    src, dst, reverse = _build_directed_edges(A)
    src_t = torch.as_tensor(src, device=device, dtype=torch.long)
    dst_t = torch.as_tensor(dst, device=device, dtype=torch.long)
    reverse_t = torch.as_tensor(reverse, device=device, dtype=torch.long)

    n_rho = len(RHO_VALUES)
    outer_budget = max(1, n // 5)
    outer_attempted = 0

    all_success = {"cleanup_only": [], "propagate_cleanup": []}
    done = {"cleanup_only": False, "propagate_cleanup": False}

    while outer_attempted < outer_budget and not (done["cleanup_only"] and done["propagate_cleanup"]):
        n_outer = min(chunk_outer, outer_budget - outer_attempted)
        bs = n_outer * n_rho
        rho_t = torch.as_tensor(RHO_VALUES, device=device).repeat(n_outer)
        belief = _bp_batch_step(A, src_t, dst_t, reverse_t, rho_t, 0.5, max_iter,
                                 seed=outer_attempted, batch_size=bs, q=3, device=device)
        colors_batch = belief.argmax(dim=-1).cpu().numpy()

        for key, use_propagate in [("cleanup_only", False), ("propagate_cleanup", True)]:
            if done[key]:
                continue
            success, first_idx = _check_batch(A, colors_batch, use_propagate=use_propagate, use_cleanup=True,
                                               prop_iters=prop_iters, pool=pool, repair_passes=repair_passes)
            all_success[key].append(success)
            if first_idx is not None:
                done[key] = True

        outer_attempted += n_outer

    result_cleanup_only = _finalize_reinforced_result(all_success["cleanup_only"], n_rho, outer_budget)
    result_propagate_cleanup = _finalize_reinforced_result(all_success["propagate_cleanup"], n_rho, outer_budget)
    return result_cleanup_only, result_propagate_cleanup


def run_reinforced_bp_cleanup_only(A, n, device, pool, chunk_outer=20, repair_passes=50):
    """ADDITIVE, standalone: identical restart ladder and cleanup_only
    pipeline as run_reinforced_bp_both's first half, but never computes or
    checks the propagate_cleanup (AK) pipeline at all -- saves the real
    CPU-side propagate+cleanup cost when only bp_reinforce_cleanup is
    wanted (e.g. for a faster n=2000 pass while bp_ak_cleanup is on hold).
    Does not modify or call run_reinforced_bp_both; that function and its
    output are completely untouched."""
    import torch
    max_iter = 3 * n
    n_ = A.shape[0]
    src, dst, reverse = _build_directed_edges(A)
    src_t = torch.as_tensor(src, device=device, dtype=torch.long)
    dst_t = torch.as_tensor(dst, device=device, dtype=torch.long)
    reverse_t = torch.as_tensor(reverse, device=device, dtype=torch.long)

    n_rho = len(RHO_VALUES)
    outer_budget = max(1, n // 5)
    outer_attempted = 0
    all_success = []

    while outer_attempted < outer_budget:
        n_outer = min(chunk_outer, outer_budget - outer_attempted)
        bs = n_outer * n_rho
        rho_t = torch.as_tensor(RHO_VALUES, device=device).repeat(n_outer)
        belief = _bp_batch_step(A, src_t, dst_t, reverse_t, rho_t, 0.5, max_iter,
                                 seed=outer_attempted, batch_size=bs, q=3, device=device)
        colors_batch = belief.argmax(dim=-1).cpu().numpy()
        success, first_idx = _check_batch(A, colors_batch, use_propagate=False, use_cleanup=True,
                                           prop_iters=1, pool=pool, repair_passes=repair_passes)
        all_success.append(success)
        outer_attempted += n_outer
        if first_idx is not None:
            break

    return _finalize_reinforced_result(all_success, n_rho, outer_budget)


def run_reinforced_bp_ak_only(A, n, device, pool, chunk_outer=20, repair_passes=50):
    """ADDITIVE, standalone mirror of run_reinforced_bp_cleanup_only: same
    restart ladder, but only computes/checks the propagate pipeline
    (bp_ak_cleanup) -- never computes or checks the cleanup_only
    (bp_reinforce_cleanup) pipeline at all. Does not modify or call
    run_reinforced_bp_both or run_reinforced_bp_cleanup_only."""
    import torch
    max_iter = 3 * n
    n_ = A.shape[0]
    prop_iters = max(1, int(np.ceil(np.log(max(n_, 2)))))
    src, dst, reverse = _build_directed_edges(A)
    src_t = torch.as_tensor(src, device=device, dtype=torch.long)
    dst_t = torch.as_tensor(dst, device=device, dtype=torch.long)
    reverse_t = torch.as_tensor(reverse, device=device, dtype=torch.long)

    n_rho = len(RHO_VALUES)
    outer_budget = max(1, n // 5)
    outer_attempted = 0
    all_success = []

    while outer_attempted < outer_budget:
        n_outer = min(chunk_outer, outer_budget - outer_attempted)
        bs = n_outer * n_rho
        rho_t = torch.as_tensor(RHO_VALUES, device=device).repeat(n_outer)
        belief = _bp_batch_step(A, src_t, dst_t, reverse_t, rho_t, 0.5, max_iter,
                                 seed=outer_attempted, batch_size=bs, q=3, device=device)
        colors_batch = belief.argmax(dim=-1).cpu().numpy()
        success, first_idx = _check_batch(A, colors_batch, use_propagate=True, use_cleanup=True,
                                           prop_iters=prop_iters, pool=pool, repair_passes=repair_passes)
        all_success.append(success)
        outer_attempted += n_outer
        if first_idx is not None:
            break

    return _finalize_reinforced_result(all_success, n_rho, outer_budget)


def pack_bitmask(bitmask):
    """Packs a bool array (1D or 2D) into base64. Shape is recorded so a 2D
    (outer_restarts, n_rho) array can be reconstructed exactly on unpack."""
    packed = np.packbits(bitmask)
    return {"shape": list(bitmask.shape), "data_b64": base64.b64encode(packed.tobytes()).decode("ascii")}


def unpack_bitmask(packed_dict):
    raw = np.frombuffer(base64.b64decode(packed_dict["data_b64"]), dtype=np.uint8)
    n_bits = int(np.prod(packed_dict["shape"]))
    return np.unpackbits(raw)[:n_bits].reshape(packed_dict["shape"]).astype(bool)


def build_c_values():
    # TODO(paper): Dan wants the fine-grained (step 0.02) band to start at
    # c=4.0 instead of 4.40 -- i.e. extend `fine` below to
    # [round(4.00 + 0.02 * i, 2) for i in range(36)]. Not yet done; c=4.0
    # currently only has coarse-grid data (see tab:density-grid in the paper).
    coarse_low = [round(1.0 + 0.5 * i, 2) for i in range(7)]  # 1.0..4.0
    fine = [round(4.40 + 0.02 * i, 2) for i in range(16)]  # 4.40..4.70
    fine_full = sorted(set(fine + [4.69]))
    coarse_high = [round(5.0 + 0.5 * i, 2) for i in range(31)]  # 5.0..20.0
    return coarse_low + fine_full + coarse_high


def parse_args():
    import torch
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--n", type=int, required=True)
    p.add_argument("--c-values", type=float, nargs="+", default=None)
    p.add_argument("--n-inst", type=int, default=100)
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--out", required=True)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--n-check-workers", type=int, default=N_CHECK_WORKERS)
    p.add_argument("--skip-ak", action="store_true",
                    help="Skip computing/saving bp_ak_cleanup entirely (uses "
                         "run_reinforced_bp_cleanup_only instead of run_reinforced_bp_both, "
                         "saving the propagate+cleanup check's real CPU cost, not just the save step). "
                         "vanilla_bp and bp_reinforce_cleanup are unaffected either way.")
    p.add_argument("--vanilla-only", action="store_true",
                    help="Only compute/save vanilla_bp -- skips the reinforced ladder entirely "
                         "(not just its AK check). Much faster since it drops the n/5x5 restart "
                         "budget entirely, not just the cleanup step.")
    p.add_argument("--ak-only", action="store_true",
                    help="Only compute/save bp_ak_cleanup -- skips vanilla_bp and "
                         "bp_reinforce_cleanup entirely (not just their reporting).")
    p.add_argument("--repair-passes", type=int, default=50,
                    help="max_passes passed to greedy_repair (default 50, matching "
                         "repair_utils.greedy_repair's own default). Override to "
                         "measure the effect of capping the repair pass count, e.g. "
                         "--repair-passes 1 for a single sweep with no iteration to "
                         "convergence. Does not affect any other run's saved output "
                         "since it's a separate --out file.")
    return p.parse_args()


def main():
    import torch
    args = parse_args()
    c_values = args.c_values if args.c_values is not None else build_c_values()
    torch.manual_seed(args.seed)

    # Resume support: if --out already exists (e.g. restarting mid-sweep to
    # pick up a code change without losing already-completed c-values),
    # load it and skip whatever's already done. Skipped c-values still
    # consume the same amount of RNG state (re-drawing and discarding their
    # instances) so every REMAINING c-value gets the exact same graphs a
    # fresh, uninterrupted run would have produced.
    import os
    if os.path.exists(args.out):
        with open(args.out) as f:
            results = json.load(f)
        print(f"resuming from existing {args.out}: {len(results)} c-values already done", flush=True)
    else:
        results = {}

    print(f"device={args.device}, n={args.n}, {len(c_values)} c-values, {args.n_inst} instances each, "
          f"{args.n_check_workers} check-workers (persistent pool)", flush=True)

    # ONE pool for the entire run -- Windows has no fork/COW, so spawning a
    # fresh N-process pool per instance was the actual bottleneck (measured:
    # negligible speedup from parallelizing the checking alone). A is passed
    # per-task now (not via initializer), relying on pickle's memoization
    # within a chunk to avoid re-serializing it per candidate.
    with mp.Pool(args.n_check_workers) as pool:
      for c in c_values:
        c_key = str(c)
        existing = results.get(c_key, {})
        n_done = len(existing)

        if n_done >= args.n_inst:
            for i in range(args.n_inst):
                create_planted_3col(args.n, c)  # keep RNG stream in sync, discard
            print(f"c={c}: already have {n_done}/{args.n_inst}, skipping", flush=True)
            continue

        # Replay RNG for c-values already fully processed, then for THIS
        # c-value's already-done instances, so instance i's graph is
        # identical across resumes (same pattern as vanilla_bp_ak_sweep.py).
        torch.manual_seed(args.seed)
        for prior_c in c_values:
            if prior_c == c:
                break
            for _ in range(len(results.get(str(prior_c), {}))):
                create_planted_3col(args.n, prior_c)
        for _ in range(n_done):
            create_planted_3col(args.n, c)

        results[c_key] = existing
        t0 = time.time()
        if args.vanilla_only:
            counts = {"vanilla_bp": 0}
            restarts_sum = {"vanilla_bp": []}
        elif args.ak_only:
            counts = {"bp_ak_cleanup": 0}
            restarts_sum = {"bp_ak_cleanup": []}
        elif args.skip_ak:
            counts = {"vanilla_bp": 0, "bp_reinforce_cleanup": 0}
            restarts_sum = {"vanilla_bp": [], "bp_reinforce_cleanup": []}
        else:
            counts = {"vanilla_bp": 0, "bp_reinforce_cleanup": 0, "bp_ak_cleanup": 0}
            restarts_sum = {"vanilla_bp": [], "bp_reinforce_cleanup": [], "bp_ak_cleanup": []}

        for i in range(n_done, args.n_inst):
            _, adj = create_planted_3col(args.n, c)
            A = adj.numpy()

            if args.ak_only:
                ok3, r3, budget3, bm3, rho3, info3 = run_reinforced_bp_ak_only(A, args.n, args.device, pool=pool, repair_passes=args.repair_passes)
                counts["bp_ak_cleanup"] += ok3
                if ok3: restarts_sum["bp_ak_cleanup"].append(r3)
                results[c_key][str(i)] = {
                    "bp_ak_cleanup": {
                        "success": ok3, "budget": budget3, "bitmask": pack_bitmask(bm3), "rho_at_success": rho3,
                        "outer_restarts_budget": info3["outer_restarts_budget"],
                        "outer_restart_at_success": info3["outer_restart_at_success"],
                        "rho_idx_at_success": info3["rho_idx_at_success"],
                        "total_subrestarts_used": info3["total_subrestarts_used"],
                    }
                }
                continue

            ok1, r1, budget1, bm1 = run_vanilla_bp(A, args.n, args.device, pool, repair_passes=args.repair_passes)
            counts["vanilla_bp"] += ok1
            if ok1: restarts_sum["vanilla_bp"].append(r1)
            results[c_key][str(i)] = {
                "vanilla_bp": {"success": ok1, "restarts": r1, "budget": budget1, "bitmask": pack_bitmask(bm1)},
            }

            if args.vanilla_only:
                continue

            if args.skip_ak:
                ok2, r2, budget2, bm2, rho2, info2 = run_reinforced_bp_cleanup_only(A, args.n, args.device, pool=pool, repair_passes=args.repair_passes)
            else:
                (ok2, r2, budget2, bm2, rho2, info2), (ok3, r3, budget3, bm3, rho3, info3) = \
                    run_reinforced_bp_both(A, args.n, args.device, pool=pool, repair_passes=args.repair_passes)

            counts["bp_reinforce_cleanup"] += ok2
            if ok2: restarts_sum["bp_reinforce_cleanup"].append(r2)
            results[c_key][str(i)]["bp_reinforce_cleanup"] = {
                "success": ok2, "budget": budget2, "bitmask": pack_bitmask(bm2), "rho_at_success": rho2,
                "outer_restarts_budget": info2["outer_restarts_budget"],
                "outer_restart_at_success": info2["outer_restart_at_success"],
                "rho_idx_at_success": info2["rho_idx_at_success"],
                "total_subrestarts_used": info2["total_subrestarts_used"],
            }
            if not args.skip_ak:
                counts["bp_ak_cleanup"] += ok3
                if ok3: restarts_sum["bp_ak_cleanup"].append(r3)
                results[c_key][str(i)]["bp_ak_cleanup"] = {
                    "success": ok3, "budget": budget3, "bitmask": pack_bitmask(bm3), "rho_at_success": rho3,
                    "outer_restarts_budget": info3["outer_restarts_budget"],
                    "outer_restart_at_success": info3["outer_restart_at_success"],
                    "rho_idx_at_success": info3["rho_idx_at_success"],
                    "total_subrestarts_used": info3["total_subrestarts_used"],
                }

        dt = time.time() - t0
        summary = " ".join(f"{k}={counts[k]}/{args.n_inst}" for k in counts)
        print(f"c={c}: {summary}  ({dt:.0f}s)", flush=True)

        with open(args.out, "w") as f:
            json.dump(results, f)

    print(f"\nsaved to {args.out}")


if __name__ == "__main__":
    mp.set_start_method("spawn", force=True)
    main()
