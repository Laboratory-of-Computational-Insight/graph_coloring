# Tasks

## In Progress

- `[ ]` **Ablation plan execution (2026-07-14 →)** — full plan tracked in memory
  (`gnn-coloring-ablation-plan`); working through it top-down, one change at a
  time, re-running the c-sweep after each. Step 0 is done — see `EXPERIMENTS.md`
  "Notes on Experiment set 2" for full results. Remaining:
  1. `[x]` **0a. Classical baseline** — `code/classical_baseline.py`. Result:
     ~100% success c=1..20 at n=45 and n=100 (dip to 78-99% at the phase
     transition c≈4.69). This is the ceiling everything below is measured against.
  2. `[x]` **0b. In-distribution control sweep** — n=45 and n=100, c=1..20, GCPNet.
     Confirms non-flat, non-zero success (unlike n=2000) — the earlier 0% is an
     OOD/size-generalization failure, not a fundamental one. Bonus finding: even
     in-distribution, GCPNet has a narrow density sweet-spot (peaks 31% at n=45,
     c=7; only 3.2% at n=100).
  3. `[x]` **Prerequisite: build training pipeline for `QueryOptGNN_MP`** — no
     `data/` dir existed. Built `code/exact_3col_solver.py` (exact backtracking
     3-coloring solver, correctness-tested) +
     `graph_color_improved/generate_adversarial_data.py` (solver-verified
     near-boundary adversarial pairs, NeuroSAT/Lemos-style, not naive random
     pairs) + `--train`/`--epochs` CLI flags on `hybrid4.py`. Generated 400
     train + 80 test pairs, trained 40 epochs from scratch: held-out
     `sat_acc` 0%→61.3%. Evaluated with the same `create_planted_3col` c-sweep
     as GCPNet: **85-99% success c=1-9 at n=45** — dramatically beats GCPNet's
     31% peak, even pre-ablation. `code/sweep_queryoptgnn.py` is the reusable
     eval script for all remaining ablations below.
  4. `[x]` **D1. Test-time round scaling** — evaluated the trained A2 checkpoint
     at rounds=64 (vs. trained rounds=32) on its two weakest regions, no
     retraining, no new data. Result: 8/12 c-values improved (avg +4.1pt),
     biggest gains exactly on the weakest points (c=5: 57.0%→69.5%; c=20:
     50.0%→66.0%). Confirms the NeuroSAT-style mechanism — the same trained
     weights generalize further with more test-time iterations, because
     nothing in the architecture is tied to an absolute round count. See
     `EXPERIMENTS.md` "Notes on Experiment set 9". Next: try rounds=128, and
     consider `sweep_queryoptgnn.py --rounds 64` as the new default eval
     setting for A2 going forward.
  5. `[ ]` **E1. Expose per-vertex conflict signal** — aggregate
     `QueryOptGNN_MP._compute_query_loss`'s `per_edge_conflict`
     (`graph_color_improved/hybrid4.py:349`) per vertex; log correlation with
     eventual miscoloring before wiring it into anything.
  6. `[x]` **A1. Re-inject color anchor every round** — `--reinject-color-embed`
     flag on `hybrid4.py`. Result: held-out sat_acc 61.3%→72.5%. On the c-sweep,
     a wash at c=3-9 (training density) but a consistent, real gain at c=12-19
     (e.g. c=13: 26.5%→43.0%) — helps OOD-density generalization specifically.
     See `EXPERIMENTS.md` "Notes on Experiment set 3".
  7. `[x]` **B1/B2. Temperature annealing** — `--anneal-temperature` flag
     (anneals `query_temp` 1.5→0.3 across rounds). Result: a real tradeoff, not
     a simple win/loss — worse than baseline/A1 at c=3-9 (training density),
     but the strongest OOD-density result so far at c=10-20 (e.g. c=13: 54.0%
     vs A1's 43.0% vs baseline's 26.5%). See `EXPERIMENTS.md` "Notes on
     Experiment set 4" — the held-out test-set metric alone was misleading
     here, the full c-sweep was needed to see the real picture.
  8. `[x]` **A2. Structured symmetry breaking** — `--structured-node-priors` flag
     (`degree_greedy_coloring` seeds the dominant-color prior instead of
     uniform-random). **Best ablation result so far**: matches/slightly beats
     baseline at c=3-9 (no B1-style tradeoff) AND gets B1-style OOD-density
     gains at c=10-20 (c=20: 50.0% vs baseline 11.5%, a 4.3x improvement).
     One real regression at c=1 (70.0% vs 98.5%) worth investigating. See
     `EXPERIMENTS.md` "Notes on Experiment set 5".
  9. `[ ]` **D2. Multi-cell / nested inner-outer loop** — only if D1 plateaus;
     split the shared `var_gru`/`edge_gru` into 2-3 alternating cells with
     `(outer_i, inner_j)` positional encoding.
  10. `[ ]` **C1/C2. Learned decimation/commit head** — per-vertex commit head
      that locks a color and shrinks neighbor domains (CSP arc-consistency
      analog of SAT decimation); differentiable soft-commit first, RL only if
      that's insufficient.
  11. `[ ]` **New: density-generalization axis** — the baseline `QueryOptGNN_MP`
      run also degrades away from its training density band (c∈[3,7]),
      mirroring the size-generalization pattern. Worth widening the training
      c-range and re-measuring before attributing gains/losses in A1/B1/A2 to
      the right cause.
  12. `[x]` **A1 + B1 combined — anti-synergistic, do not stack naively.**
      Held-out sat_acc=66.2% looked unremarkable, but the c-sweep shows the
      OOD-density tail (where both help individually) **collapses to 0%** at
      c=17-20, worse than baseline. The two mechanisms fight each other
      mechanistically. See `EXPERIMENTS.md` "Notes on Experiment set 6".
  13. `[x]` **A1 + A2 combined — a sidegrade, not a clean win.** Fixes A2's c=1
      regression (97.0% vs 70.0%) and extends the extreme tail (c=20: 73.5% vs
      A2's 50.0%), but gives up ground at c=9-17 vs A2 alone (avg 71.2% vs
      74.4%). No collapse anywhere (unlike A1+B1). This also explains A2's
      c=1 weakness: the anchor compensates for weak signal when most vertices
      have degree 0-1. See `EXPERIMENTS.md` "Notes on Experiment set 7".
  14. `[x]` **B1 + A2 combined** — well-behaved, no collapse (avg 71.0%,
      essentially tied with A1+A2's 71.2%); partially fixes A2's c=1 weakness
      (79.5% vs A2's 70.0%, short of A1+A2's 97.0%). Completes the pairwise
      combination matrix. See `EXPERIMENTS.md` "Notes on Experiment set 8".

  **Full pairwise ablation matrix complete (2026-07-14).** Every combination
  that includes A2 lands in a tight 71-74% average-success band (n=45,
  c=1-20), far above anything without it (43-60%) — A2 (structured,
  degree-informed symmetry breaking) is the dominant factor, and it barely
  matters what it's paired with. The one trap: A1+B1 *without* A2 is
  anti-synergistic and actively worse than baseline.

  ~~**Recommended default: A2 alone** (avg 74.4%)~~ — **superseded, see below.**

  **D1 pushed further (2026-07-14): rounds=300 + best-of-iterations tracking.**
  Added `track_best` to `QueryOptGNN_MP.forward()` (checks legality every
  round, keeps best/first-legal, stops early) and `--rounds 300
  --track-best` to `sweep_queryoptgnn.py`. On the A2 checkpoint: **avg
  74.4%→83.0%**, c=1 and c=2 now hit a perfect 100.0% (fixing A2's one real
  weakness without needing A1), c=20 goes 50.0%→78.5%. Zero retraining, zero
  new data — purely better use of inference-time compute. **New recommended
  default: A2 checkpoint, evaluated at rounds=300 with track_best=True.**
  Every future ablation (D2, C1/C2, E1) should be compared against *this*,
  not the rounds=32 numbers. See `EXPERIMENTS.md` "Notes on Experiment set 10".
  Worst point is now c=5 (68.0%, phase transition) — still short of the
  98%-for-all-c target but far closer than anything else tried.

  **⚠ Correction, n≥1000 required from now on (user instruction, 2026-07-14):**
  every number above was measured at n=45/100/300 and is flagged suspect —
  small n blurs the phase transition. Real n=1000 sweeps (all methods) show a
  much bigger picture: classical baseline has a genuine failure valley
  c=5-12 (0-52%, not a narrow dip); GCPNet is flat 0% everywhere; spectral
  algorithm hits 100% pure-spectral from c≥15 but 0% at c=5-9; and
  **`QueryOptGNN_MP` A2+rounds=300 (the "83.0% avg" config above) also
  collapses at n=1000** — 0% at c=4-5, and unlike the classical methods, it
  never recovers (stays at 0% from c=12 through c=20, no climb back to 100%).
  The exact hard/unsolved window at real scale is **c=4 through c=14**.

  **Spectral-NN (user's high-dim embedding idea, 2026-07-14)** —
  `code/spectral_nn.py`. Found and fixed 3 real bugs in sequence: (1) full
  representation collapse from a constant vertex init (fixed: degree-based
  init + variance regularizer); (2) the training loss didn't target the
  actual PCA-based readout (fixed: replaced pairwise contrastive loss with a
  Fisher-discriminant loss, aligned with what the readout actually uses; also
  fixed a division-by-near-zero instability this introduced); (3) raising
  rounds from 8 to 64 re-triggered collapse via GRU fixed-point convergence
  over many rounds (fixed: re-inject the degree signal every round — the same
  fix as A1). Trained 3 models specialized by c-band (small=1-8, medium=6-18,
  large=14-30). Result: c=5-8 (true hard core) still 0% for every model,
  learned or not; but **c=10-11 shows a real win** — small model 60% vs
  plain spectral's 27% at c=10 — at a small cost at c=14 (80-93% vs plain's
  100%). See `EXPERIMENTS.md` "Notes on Experiment set 13".

  Reference implementations cloned for comparison/ideas (not part of this repo's
  own code): `references/anycsp` (Tönshoff et al., "One Model, Any CSP"),
  `references/pi_gnn` (Schuetz et al., physics-inspired GNN, Amazon).

  **Spectral coordinates as NN input, multi-stage nested-loop, pairwise
  reformulation, and cheap ensembling — all cross-tested (2026-07-15).** Per
  the "try every insight as a flag, cross-test against all models" mandate:
  (1) `use_spectral_coords` flag on `SpectralEmbedder` — bounded gains,
  same story as before, c=5-7 still 0%. (2) `code/multistage_nn.py`
  (D2, 3 distinct GRU cells + sinusoidal loop positional encoding) — real
  but modest gains c=10-13 (e.g. c=12 small_multistage 9/10 vs plain 6/10).
  (3) `code/pairwise_nn.py` (same/different reformulation) — **best result
  of the project**, see TASKS entry above. (4) `--ensemble` OR-combination
  flag + `code/combined_ensemble_eval.py` (cross-architecture OR, no new
  training) — within spectral-NN family alone, ANY already reaches 10/10 at
  c=12-14. **Cross-family OR-ensemble (10 checkpoints, 20 inst/c, n=1000) is
  the headline result of the whole project: c=9 5%→70%, c=10 30%→95%, c=11
  50%→100%, c=12 60%→100% — closing the c=9-12 gap up to c=13-14's ~100%
  ceiling, with zero new training, directly delivering the "c>7 like c>14"
  mission.** c=8 barely moves (0%→10%). **c=5-7 remains an absolute floor for
  every one of the 10 checkpoints and their combination, 0/20 with no
  exception** (classical, spectral, GCPNet, QueryOptGNN_MP, every
  spectral-NN/pairwise/multistage variant, and their ensembles) — the
  strongest evidence yet that this sits at/below the true
  information-theoretic threshold, not a training gap. Full numbers in
  `EXPERIMENTS.md` "Notes on Experiment set 14".

## Next Up

- `[ ]` **Multi-n success-rate plot** — combine results from n=45, 100, 500, 1000, 2000 into
  one figure (c on x-axis, success rate on y-axis, one line per n). Key figure for understanding
  GNN-GCP's generalization boundary.

- `[x]` **Implemented Alon-Kahale spectral 3-coloring algorithm** (2026-07-14) —
  `code/spectral_coloring.py`, the actual paper's 3-phase method (spectral init via
  2 smallest eigenvectors + propagation + support-threshold cleanup), not the
  DSATUR stand-in used before. Result at n=300: **100% success and pure-spectral
  (no fallback) from c≥16**, confirming/beating "works for c>20" — but exposed a
  serious weak point at c=5 (2% success, 14+ min for 200 instances).
- `[ ]` **Fix `dsatur_3color`'s scaling at n≥300** — profiled the c=5 slowdown
  above: the real cost is 1.35M `numpy.sum()` calls in `dsatur_3color`'s
  per-vertex saturation scan (O(n) per step, compounded by restarts), not the
  spectral cleanup's component solver (a `node_budget` cap was added there but
  confirmed *not* the fix — it just cut off searches that would've succeeded,
  making results worse). Needs incremental saturation bookkeeping instead of
  re-summing per step, and/or fewer restarts in the fallback path specifically.
- `[x]` **"Same partition or not" pairwise reformulation** (2026-07-15) —
  `code/pairwise_nn.py`. Binary same/different-class classifier on vertex pairs,
  decoded via eigen-decomposition of the centered same-class probability matrix
  (assortative convention) into the existing propagation/cleanup pipeline. Fixed
  its own collapse mode (plain BCE → free "always different" solution) via
  spectral-coordinate init + an auxiliary Fisher/LDA term. **Result: best result
  of the whole project at c=9-11** — large_pairwise 6/10 and 9/10 at c=9/10 (n=1000)
  vs plain's 1/10 at both; medium_pairwise 10/10 at c=11 vs plain's 6/10. Still
  0/10 at c=5-7. See `EXPERIMENTS.md` "Notes on Experiment set 14".
- `[ ]` **Algorithm mixture/routing per c** — user idea (2026-07-14); flagged
  honestly that classical peel/DSATUR/spectral already gets ~100% across nearly
  all of c=1-30, so a mixture's main value is runtime/scale robustness or as a
  verification layer, not beating the GNN on raw success rate.

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
- `[ ]` Upload sweep results to Wasabi S3 after completion (`utils/fs.py` already wired up).

## Done

- `[x]` **Sweep n=2000, c=1..30** — 500 inst/c, tmax=32, final-embedding k-means check.
  Result: 0/15,000 successes (complete OOD failure). `sweep_results_n2000.json` + `sweep_n2000.png`.
- `[x]` **Implement `create_planted_3col(n, c)`** (2026-07-14) — was referenced everywhere
  (docs, `sweep_planted.py`) but not actually defined anywhere in the codebase; the 2026-06-06
  sweep above could not have run against the code as it stood. Implemented per the documented
  spec and validated against literature convention. Lives at `random_planted.py`.
- `[x]` **Fix `gc_utils.py` top-level `import picos`/`cvxpy`/`plotly`** (2026-07-14) — a
  previous "Done" entry here claimed this was fixed, but the imports were unguarded at the
  top of the file; now properly wrapped in `try/except`.
- `[x]` **Add CLI flags to `sweep_planted.py`** (2026-07-14) — `--n`, `--c-min`, `--c-max`,
  `--n-inst`, `--tmax`, `--results-file`, `--plot-file`, defaults preserve the original
  n=2000/c=1..30 behavior.
- `[x]` **Port GNN-GCP from TensorFlow to PyTorch** — `model.py`, `lstm.py`, `mlp.py`,
  weights loaded from `original.npz`.
- `[x]` **Paper published** — "From Black Box to Algorithmic Insight" (AAAI 2025,
  Shoham / Rika / Vilenchik).
- `[x]` **Ablation plan Step 0 (classical baseline + control sweeps + QueryOptGNN_MP
  training pipeline)** (2026-07-14) — see "In Progress" above and `EXPERIMENTS.md` for
  full detail and numbers.
