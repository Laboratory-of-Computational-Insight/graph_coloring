"""
Test bp_then_ak_cleanup_escalating (ak_bp_hybrid.py): BP (plain ->
reinforced -> hyperdiverse escalation) REPLACES AK's spectral phase 1
entirely (not biasing it -- a genuinely different algorithm produces the
initial coloring), then AK's own phase 2 (propagate) + phase 3
(cleanup/peeling: raise a support threshold, uncolor everything below it,
brute-force or DSATUR-fallback on the shrinking residual core) run exactly
as they would on AK's own spectral guess. Requested first at c>=15, where
plain classical alon_kahale_3color is already known to be ~100% -- this run
validates the hybrid pipeline doesn't regress there before testing it
anywhere harder.
"""
import argparse
import time

import torch

from random_planted import create_planted_3col
from gc_utils import is_k_color
from ak_bp_hybrid import bp_then_ak_cleanup_escalating

RESULTS_DIR = "G:/graph_col/graph_coloring/results"


def parse_args():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--n", type=int, default=1000)
    p.add_argument("--c-values", type=float, nargs="+", default=[15, 16, 18, 20])
    p.add_argument("--n-inst", type=int, default=10)
    p.add_argument("--cap", type=int, default=30)
    return p.parse_args()


def main():
    args = parse_args()
    torch.manual_seed(0)
    print(f"{'c':>6}{'n_success':>11}{'time(s)':>10}")

    for c in args.c_values:
        c_disp = int(c) if c == int(c) else c
        n_success = 0
        t0 = time.time()
        for i in range(args.n_inst):
            planted, adj = create_planted_3col(args.n, c)
            colors, conf, success = bp_then_ak_cleanup_escalating(adj, cap_total=args.cap)
            if success:
                ok, nconf, _ = is_k_color(adj.clone(), colors.copy())
                assert ok, f"illegal at c={c} inst={i}, nconf={nconf}"
            n_success += success
            print(f"  c={c_disp} inst={i}: {'SUCCESS' if success else 'FAIL'} (conflicts={conf})", flush=True)

        dt = time.time() - t0
        print(f"{c_disp!s:>6}{f'{n_success}/{args.n_inst}':>11}{dt:>10.0f}", flush=True)

    print("\ndone")


if __name__ == "__main__":
    main()
