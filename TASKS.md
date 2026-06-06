# Tasks

## In Progress

_(none currently running)_

## Next Up

- `[ ]` **Sweep n=45 and n=100 as controls** — run `sweep_planted.py` with `N=45` and `N=100`
  over `c=1..10` to confirm within-distribution graphs yield non-zero success rates and to
  bracket where the model breaks down. Use `create_planted_3col`.

- `[ ]` **Multi-n success-rate plot** — combine results from n=45, 100, 500, 1000, 2000 into
  one figure (c on x-axis, success rate on y-axis, one line per n). Key figure for understanding
  GNN-GCP's generalization boundary.

- `[ ]` **Per-iteration vs. final-only comparison** — at n=45, compare "check k-means only at
  tmax=32" vs. "check at every iteration up to tmax=150". Quantify how much success rate is
  lost by checking only the final embedding (current sweep methodology).

- `[ ]` **Refactor `attribute_tests_v2.py` to accept in-memory graphs** — currently requires
  remote Wasabi S3 data. Decouple the data-loading from the analysis so it can run on
  `create_planted_3col`-generated graphs locally.

- `[ ]` **Run full XAI attribute analysis on local data** — once the above refactor is done,
  run `attributes_check()` on DS (success) and DF (failure) sets generated locally and
  reproduce the Spearman / confidence / support tables from the AAAI paper.

## Backlog

- `[ ]` Refactor `graph_coloring_attributes.py` — four support-variant functions defined, only
  one used in the paper. Remove or annotate the unused ones.
- `[ ]` `run_model.py:23` — remove debug `print(k, ...)` that fires on every eval run.
- `[ ]` Export `three_kol_files.py` to a plain JSON file; it is 26 KB of a Python list literal.
- `[ ]` Add minimal CLI flags to `sweep_planted.py` (--n, --c-max, --n-inst) so parameters
  don't require editing the source file.
- `[ ]` Upload sweep results to Wasabi S3 after completion (`utils/fs.py` already wired up).

## Done

- `[x]` **Sweep n=2000, c=1..30** — 500 inst/c, tmax=32, final-embedding k-means check.
  Result: 0/15,000 successes (complete OOD failure). `sweep_results_n2000.json` + `sweep_n2000.png`.
- `[x]` **Implement `create_planted_3col(n, c)`** — balanced partition into 3 classes of
  exactly n/3 vertices, cross-class edge probability p=3c/2n, expected degree=c.
  Lives at `random_planted.py:7`.
- `[x]` **Fix `gc_utils.py` top-level `import picos`** — moved to `try/except` so the module
  imports on machines without picos/cvxpy installed.
- `[x]` **Port GNN-GCP from TensorFlow to PyTorch** — `model.py`, `lstm.py`, `mlp.py`,
  weights loaded from `original.npz`.
- `[x]` **Paper published** — "From Black Box to Algorithmic Insight" (AAAI 2025,
  Shoham / Rika / Vilenchik).
