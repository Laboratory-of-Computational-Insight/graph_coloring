"""
User's request (2026-07-18): for the 3 c=4.69 instances still unsolved by
everything tried so far (1, 10, 16), alternate between a plain-BP restart
attempt and a hyperparameter-diversified restart attempt, up to 500 plain-BP
attempts / 200 hyperparam-diverse attempts, interleaved 1:1 (bp, hyper, bp,
hyper, ...) until success or budget exhaustion.
"""
import json
import torch
import numpy as np

from belief_propagation_coloring import bp_reinforced_coloring
from repair_utils import count_conflicts, greedy_repair

RESULTS_DIR = "G:/graph_col/graph_coloring/results"

RHO_VALUES = [0.001, 0.003, 0.01, 0.03, 0.1]
DAMPING_VALUES = [0.3, 0.5, 0.7]
HYPERPARAM_COMBOS = [(r, d) for r in RHO_VALUES for d in DAMPING_VALUES]  # 15 combos

MAX_BP = 500
MAX_HYPER = 200


def try_plain_bp(A, seed):
    belief, _ = bp_reinforced_coloring(A, seed=seed, rho=0.003, damping=0.5)
    colors = belief.argmax(axis=1)
    conf = count_conflicts(A, colors)
    if conf > 0:
        colors = greedy_repair(A, colors)
        conf = count_conflicts(A, colors)
    return conf == 0


def try_hyperparam_bp(A, attempt_idx):
    rho, damping = HYPERPARAM_COMBOS[attempt_idx % len(HYPERPARAM_COMBOS)]
    seed = 1000 + attempt_idx  # distinct seed space from plain BP
    belief, _ = bp_reinforced_coloring(A, seed=seed, rho=rho, damping=damping)
    colors = belief.argmax(axis=1)
    conf = count_conflicts(A, colors)
    if conf > 0:
        colors = greedy_repair(A, colors)
        conf = count_conflicts(A, colors)
    return conf == 0


if __name__ == "__main__":
    targets = [1, 10, 16]
    results = {}

    for i in targets:
        d = torch.load(f"{RESULTS_DIR}/shared_test_instances/c4.69_inst{i}.pt")
        adj = d["adj"]
        A = adj.numpy()

        bp_count = 0
        hyper_count = 0
        success = False
        winner = None
        round_idx = 0

        while bp_count < MAX_BP or hyper_count < MAX_HYPER:
            round_idx += 1
            if bp_count < MAX_BP:
                if try_plain_bp(A, seed=bp_count):
                    success = True
                    winner = f"plain_bp(seed={bp_count})"
                    bp_count += 1
                    break
                bp_count += 1

            if hyper_count < MAX_HYPER:
                if try_hyperparam_bp(A, attempt_idx=hyper_count):
                    success = True
                    winner = f"hyperparam_bp(attempt={hyper_count})"
                    hyper_count += 1
                    break
                hyper_count += 1

            if bp_count % 20 == 0 or hyper_count % 20 == 0:
                print(f"  inst={i}: progress bp={bp_count}/{MAX_BP} hyper={hyper_count}/{MAX_HYPER}")

        results[str(i)] = {"success": success, "winner": winner,
                            "bp_attempts": bp_count, "hyper_attempts": hyper_count}
        print(f"inst={i}: {'SUCCESS via ' + winner if success else 'fail'} "
              f"(bp_attempts={bp_count}, hyper_attempts={hyper_count})")

    n_solved = sum(v["success"] for v in results.values())
    print(f"\n=== alternating BP/hyperparam on {targets}: {n_solved}/{len(targets)} newly solved ===")

    with open(f"{RESULTS_DIR}/phase_transition_bp_alternating.json", "w") as f:
        json.dump(results, f, indent=2)
