"""
Classical algorithms (Alon-Kahale spectral, DSATUR, degree-peeling) + repair,
evaluated on the shared c=4,5,6 test instances, for folding into the real
combined ensemble alongside the spectral-NN family, BP, and QueryOptGNN_MP.
No OR-Tools/CP-SAT here -- diagnostic-only per project rule, never counted.
"""
import sys
import json
import numpy as np
import torch

from spectral_coloring import alon_kahale_3color
from classical_baseline import dsatur_3color, peel_3color
from repair_utils import count_conflicts, greedy_repair
from gc_utils import is_k_color


def run_method(name, fn, adj, A):
    result = fn(adj)
    if isinstance(result, dict):
        colors = result["colors"]
    elif isinstance(result, tuple):
        _, colors = result  # (success_bool, colors)
    else:
        colors = result
    if colors is None:
        return False, None
    if torch.is_tensor(colors):
        colors = colors.numpy()
    colors = np.asarray(colors)
    conf = count_conflicts(A, colors)
    if conf > 0:
        colors = greedy_repair(A, colors)
        conf = count_conflicts(A, colors)
    ok, nconf, _ = is_k_color(adj.clone(), torch.tensor(colors))
    assert (nconf == 0) == (conf == 0)
    return conf == 0, colors


if __name__ == "__main__":
    RESULTS_DIR = "G:/graph_col/graph_coloring/results"
    c_list = [int(x) for x in sys.argv[1:]] if len(sys.argv) > 1 else [4, 5, 6]
    n_inst = 20

    methods = {
        "alon_kahale": lambda adj: alon_kahale_3color(adj),
        "dsatur": lambda adj: dsatur_3color(adj),
        "peel": lambda adj: peel_3color(adj),
    }

    all_results = {m: {} for m in methods}
    for c in c_list:
        counts = {m: 0 for m in methods}
        for i in range(n_inst):
            d = torch.load(f"{RESULTS_DIR}/shared_test_instances/c{c}_inst{i}.pt")
            adj = d["adj"]
            A = adj.numpy()
            for m, fn in methods.items():
                success, _ = run_method(m, fn, adj, A)
                all_results[m].setdefault(str(c), {})[str(i)] = bool(success)
                counts[m] += success
        print(f"c={c}: " + ", ".join(f"{m}={counts[m]}/{n_inst}" for m in methods))

    for m in methods:
        with open(f"{RESULTS_DIR}/shared_test_{m}_c4to6_repaired.json", "w") as f:
            json.dump(all_results[m], f, indent=2)
    print("done")
