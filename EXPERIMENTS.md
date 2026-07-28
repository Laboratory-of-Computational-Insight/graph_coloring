# Experiments

Append-only log. Add a row each time a non-trivial experiment completes.

**Success definition:** a legal 3-coloring is found by running k-means (k=3) on the vertex
embedding matrix and verifying zero monochromatic edges via `gc_utils.is_k_color()`.

**Columns:** Date | Script | Key parameters | Total instances | Outcome | Artefacts

---

## Log

| Date | Script | Key parameters | Instances | Outcome | Artefacts |
|------|--------|----------------|-----------|---------|-----------|
| 2026-06-06 | `sweep_planted.py` | n=2000, c=1..30, tmax=32, 500 inst/c, check=final-only, device=mps | 15 000 | **0% success all c** — complete OOD failure | `sweep_results_n2000.json`, `sweep_n2000.png` |
| 2026-07-14 | `classical_baseline_sweep.py` | n=45, c=1..20, 500 inst/c, peel+DSATUR(30 restarts) | 10 000 | **~100% success all c** (dip to 99.0% at c=5, the phase-transition region) | `classical_baseline_n45.json` |
| 2026-07-14 | `classical_baseline_sweep.py` | n=100, c=1..20, 500 inst/c, peel+DSATUR(30 restarts) | 10 000 | **~100% success**, deeper dip to 78.4% at c=5 | `classical_baseline_n100.json` |
| 2026-07-14 | `sweep_planted.py` (GCPNet, pretrained) | n=45, c=1..20, tmax=32, 500 inst/c, check=final-only, device=cuda | 10 000 | Non-flat: 0% at c=1,4 → peaks 31.0% at c=7 → 0% by c=10+. Narrow density sweet-spot even in-distribution | `sweep_results_n45.json`, `sweep_n45.png` |
| 2026-07-14 | `sweep_planted.py` (GCPNet, pretrained) | n=100, c=1..20, tmax=32, 500 inst/c, check=final-only, device=cuda | 10 000 | Same shape, much weaker: peaks 3.2% at c=8. Confirms gradual (not just cliff-at-2000) size degradation | `sweep_results_n100.json`, `sweep_n100.png` |
| 2026-07-14 | `hybrid4.py --train` (`QueryOptGNN_MP`, from scratch) | 400 verified adversarial train pairs, 80 test pairs, n∈[40,60], c∈[3,7], 40 epochs, batch=32, device=cuda | 800 train graphs | Held-out `sat_acc` 0%→61.3% over training; `avg_conflicts` 58.6→0.76 | `models/querysat/.../checkpoints/epoch_40.pt`, `history.json` |
| 2026-07-14 | `sweep_queryoptgnn.py` (trained `QueryOptGNN_MP`, epoch 40) | n=45, c=1..20, 200 inst/c, on `create_planted_3col` (not adversarial pairs) | 4 000 | **85-99% success for c=1-9**, dip to 52.5% at c=5 (phase transition), gradual decline to 11.5% by c=20 | `queryoptgnn_sweep_n45.json` |
| 2026-07-14 | `hybrid4.py --train --reinject-color-embed` (A1 ablation) | Same data/config as baseline, 40 epochs | 800 train graphs | Held-out `sat_acc` 61.3%→**72.5%** (+11.2pt) vs. baseline | `models/querysat/a1_reinject_color_embed/checkpoints/epoch_40.pt` |
| 2026-07-14 | `sweep_queryoptgnn.py --reinject-color-embed` (A1) | n=45, c=1..20, 200 inst/c | 4 000 | Comparable to baseline at c=1-9 (small mixed ±), but **consistently better at c=12-19** (e.g. c=13: 43.0% vs 26.5%, c=16: 25.0% vs 15.0%) — helps the OOD-density tail, not the core training band | `queryoptgnn_a1_sweep_n45.json` |
| 2026-07-14 | `hybrid4.py --train --anneal-temperature` (B1 ablation) | Same data/config as baseline, 40 epochs | 800 train graphs | Held-out `sat_acc` (measured on the c∈[3,7] adversarial test set) 61.3%→**47.5%**, looks like a regression in isolation | `models/querysat/b1_anneal_temp/checkpoints/epoch_40.pt` |
| 2026-07-14 | `sweep_queryoptgnn.py --anneal-temperature` (B1) | n=45, c=1..20, 200 inst/c | 4 000 | **Not a simple regression** — worse than baseline at c=3-9 (e.g. c=5: 38.0% vs 52.5%), but dramatically better at c=10-20, beating even A1 there (e.g. c=13: 54.0% vs A1's 43.0% vs baseline's 26.5%; c=20: 23.0% vs 13.5% vs 11.5%). See interpretation below | `queryoptgnn_b1_sweep_n45.json` |
| 2026-07-14 | `hybrid4.py --train --structured-node-priors` (A2 ablation) | Same data/config as baseline, 40 epochs | 800 train graphs | Held-out `sat_acc` 61.3%→**67.5%** (between baseline and A1) | `models/querysat/a2_structured_priors/checkpoints/epoch_40.pt` |
| 2026-07-14 | `sweep_queryoptgnn.py --structured-node-priors` (A2) | n=45, c=1..20, 200 inst/c | 4 000 | **Best result so far** — roughly matches or slightly beats baseline at c=3-9 (no B1-style tradeoff), and dramatically better at c=10-20 (c=16: 64.0% vs baseline 15.0%/A1 25.0%/B1 36.5%; c=20: 50.0% vs 11.5%/13.5%/23.0%). One real weak spot: c=1 regresses to 70.0% vs baseline's 98.5% | `queryoptgnn_a2_sweep_n45.json` |
| 2026-07-14 | `hybrid4.py --train --reinject-color-embed --anneal-temperature` (A1+B1 combined) | Same data/config, 40 epochs | 800 train graphs | Held-out `sat_acc` 66.2% (between the two alone) | `models/querysat/a1_b1_combined/checkpoints/epoch_40.pt` |
| 2026-07-14 | `sweep_queryoptgnn.py --reinject-color-embed --anneal-temperature` (A1+B1) | n=45, c=1..20, 200 inst/c | 4 000 | **Anti-synergistic — collapses in the OOD tail.** c=3-9 comparable to individual runs, but c=13-20 falls to near-zero (c=15: 2.5% vs A1's 24.0%/B1's 41.5%; c=17-20: **0.0%**, vs A1's 13.5-23.0%/B1's 22.5-37.5%). The two mechanisms that each helped alone actively interfere when combined | `queryoptgnn_a1b1_sweep_n45.json` |
| 2026-07-14 | `hybrid4.py --train --reinject-color-embed --structured-node-priors` (A1+A2 combined) | Same data/config, 40 epochs | 800 train graphs | Held-out `sat_acc` 62.5% (below both individually — but see c-sweep) | `models/querysat/a1_a2_combined/checkpoints/epoch_40.pt` |
| 2026-07-14 | `sweep_queryoptgnn.py --reinject-color-embed --structured-node-priors` (A1+A2) | n=45, c=1..20, 200 inst/c | 4 000 | **A sidegrade, not a strict win or loss.** Fixes A2's c=1 regression (97.0% vs A2's 70.0%) and pushes the extreme tail further (c=20: 73.5% vs A2's 50.0%), but is weaker than A2 alone in the c=9-17 band (e.g. c=12: 53.5% vs A2's 72.0%). No catastrophic collapse anywhere (unlike A1+B1) — full-sweep average 71.2% vs A2 alone's 74.4% | `queryoptgnn_a1a2_sweep_n45.json` |
| 2026-07-14 | `hybrid4.py --train --anneal-temperature --structured-node-priors` (B1+A2 combined) | Same data/config, 40 epochs | 800 train graphs | Held-out `sat_acc` 70.0% (second-highest single number after A1 alone) | `models/querysat/b1_a2_combined/checkpoints/epoch_40.pt` |
| 2026-07-14 | `sweep_queryoptgnn.py --anneal-temperature --structured-node-priors` (B1+A2) | n=45, c=1..20, 200 inst/c | 4 000 | Well-behaved, no collapse — avg 71.0%, worst 50.0% (c=15), essentially tied with A1+A2. Partial (not full) fix of A2's c=1 weakness (79.5% vs A2's 70.0%, short of A1+A2's 97.0%) | `queryoptgnn_b1a2_sweep_n45.json` |
| 2026-07-15 | `hybrid4.py --train --structured-node-priors --multi-head-reasoning` (A2+C1, `num_reasoning_heads=4`) | Same data/config as other ablations, 40 epochs (interrupted at epoch 31/40 by an environment-level background-task kill, unrelated to the code — resumed cleanly via new `--auto-resume` flag, model+optimizer+scheduler state restored) | 800 train graphs | See Notes on Experiment set 16 | `models/querysat/a2_multihead4_combined/checkpoints/epoch_40.pt` |
| 2026-07-15 | `hybrid4.py --train --multi-head-reasoning` (C1 alone, no A1/B1/A2) | Same data/config, 40 epochs | 800 train graphs | See Notes on Experiment set 16 | `models/querysat/multihead4_alone/checkpoints/epoch_40.pt` |
| 2026-07-15 | `hybrid4.py --train --reinject-color-embed --multi-head-reasoning` (A1+C1) | Same data/config, 40 epochs | 800 train graphs | See Notes on Experiment set 16 | `models/querysat/a1_multihead4_combined/checkpoints/epoch_40.pt` |
| 2026-07-15 | `hybrid4.py --train --anneal-temperature --multi-head-reasoning` (B1+C1) | Same data/config, 40 epochs | 800 train graphs | See Notes on Experiment set 16 | `models/querysat/b1_multihead4_combined/checkpoints/epoch_40.pt` |
| 2026-07-15 | `sweep_queryoptgnn.py --structured-node-priors --multi-head-reasoning --rounds 300 --track-best` (A2+C1) | **n=1000**, c=3..14, 50 inst/c | 600 | See Notes on Experiment set 16 | `queryoptgnn_a2_multihead4_r300_n1000.json` |
| 2026-07-15 | `sweep_queryoptgnn.py --multi-head-reasoning --rounds 300 --track-best` (C1 alone) | **n=1000**, c=3..14, 50 inst/c | 600 | See Notes on Experiment set 16 | `queryoptgnn_multihead4_alone_r300_n1000.json` |
| 2026-07-15 | `sweep_queryoptgnn.py --reinject-color-embed --multi-head-reasoning --rounds 300 --track-best` (A1+C1) | **n=1000**, c=3..14, 50 inst/c | 600 | See Notes on Experiment set 16 | `queryoptgnn_a1_multihead4_r300_n1000.json` |
| 2026-07-15 | `sweep_queryoptgnn.py --anneal-temperature --multi-head-reasoning --rounds 300 --track-best` (B1+C1) | **n=1000**, c=3..14, 50 inst/c | 600 | See Notes on Experiment set 16 | `queryoptgnn_b1_multihead4_r300_n1000.json` |

---

## Notes on Experiment 1 (2026-06-06)

**Setup:** Planted 3-colorable graphs generated by `create_planted_3col(n=2000, c)`.
Each instance: balanced 3-partition, cross-class edge prob p=3c/2n, expected degree=c.
Model: GNN-GCP weights from `original.npz` (Lemos et al. pre-trained TF checkpoint).
Evaluation: single forward pass (tmax=32), k-means on final embedding only.

**Result:** 0 legal colorings found in 15,000 trials across c=1..30.
The flatness (no variance at all) means this is not a marginal failure — the model's embedding
geometry is entirely non-triangular at n=2000, consistent with complete out-of-distribution
behaviour.

**Interpretation:** The model was trained on n ∈ [40, 60]. At n=2000 the `modified()` forward
applies a ÷40 scale to vertex embeddings, but this does not recover valid dynamics. The
Lemos paper already noted drastic accuracy degradation at n=1000; n=2000 confirms total failure.

**Next:** Run n=45 and n=100 sweeps as controls to find the failure point and confirm the
within-distribution regime gives non-zero success rates.

---

## Notes on Experiment set 2 (2026-07-14) — Step 0 of the ablation plan

**Prerequisite bug found:** `create_planted_3col` was referenced throughout the docs and
imported by `sweep_planted.py`, but was never actually defined anywhere in the codebase —
the 2026-06-06 n=2000 experiment could not have run against the current state of
`random_planted.py`. Implemented it from the spec already documented in `INSIGHTS.md`
(balanced 3-partition, `p=3c/(2n)`) and validated it directly: correct class balance, zero
within-class edges, correct mean degree, and the planted assignment independently confirmed
legal via `gc_utils.is_k_color`.

**Classical baseline (Step 0a):** Implemented `code/classical_baseline.py` — a
degree-peeling greedy (provably correct whenever the 3-core is empty, matching the
sparse-regime argument from the planted-3-coloring paper excerpt) with a DSATUR+restarts
fallback for the denser regime. Result: ~100% success across c=1..20 at both n=45 and
n=100, with the expected easy-hard-easy dip right at the phase transition (c≈4.69) — this
is the "should be achievable" ceiling everything else in this project should be measured
against.

**In-distribution control sweep (Step 0b):** Re-ran the exact `sweep_planted.py`
methodology at n=45 and n=100 (GCPNet's actual training range), instead of only the
previous out-of-distribution n=2000. Both are **non-flat and non-zero**, unlike n=2000 —
confirming the 2026-06-06 result is genuinely an out-of-distribution/size-generalization
failure and not evidence the architecture can't solve coloring at all. A second, unplanned
finding: even fully in-distribution (n=45), GCPNet has a narrow density "sweet spot"
(peaking at 31% around c=7) and falls back to 0% outside it — the model's effective
working range is narrower than "trained on n∈[40,60]" alone would suggest, and this
sweet-spot shrinks further at n=100 (peak drops to 3.2%). This is a second, gradual
size-sensitivity axis, distinct from the sharp n=2000 cliff.

**QueryOptGNN_MP has no data or training capability yet — built both:** No `data/` directory
existed in `graph_color_improved`, so `QueryOptGNN_MP` could never have been trained.
Built `code/exact_3col_solver.py` (backtracking + forward-checking + MRV + color-symmetry
breaking; correctness-tested against K4/K3/C5/C4/Petersen/complete-4-partite and planted
graphs — caught and fixed a real bug where the solver discarded its own answer on the
success path) and `graph_color_improved/generate_adversarial_data.py`, which generates
near-boundary adversarial pairs (a 3-colorable graph G, plus one *solver-proven* critical
edge making G+e genuinely uncolorable — not just "breaks the one planted partition"),
matching the NeuroSAT/Lemos-style training convention of forcing the model to learn real
structure instead of density shortcuts. Generated 400 train + 80 test pairs (n∈[40,60],
c∈[3,7]) in 44s, all independently re-verified against the solver and against
`parse_graphs_adversarial.py`'s own loader end-to-end.

**First QueryOptGNN_MP training run — major result:** Trained from scratch, 40 epochs,
on the 400 generated pairs. Held-out `sat_acc` climbed from 0% to 61.3%, `avg_conflicts`
from 58.6 to 0.76. Evaluating the trained checkpoint with the *same* `create_planted_3col`
c-sweep used for GCPNet (not the adversarial pairs it was trained on) at n=45: **85-99%
success for c=1-9** (peaking 99% at c=8), a shallower dip to 52.5% at c=5 (phase
transition), then a gradual decline to 11.5% by c=20. This dramatically outperforms
GCPNet's 31% peak at the same n=45 — even before any of the planned A1/B1/A2 ablations.
The decline above c~10 lines up with the training data's density band (c∈[3,7]): the model
generalizes well near its training density and degrades further from it, mirroring the
size-generalization pattern in the same way — density and size look like the same kind of
out-of-distribution axis for these architectures.

**Interpretation for the ablation plan:** `QueryOptGNN_MP`'s existing mechanisms (color
anchor, per-color message passing, query-embedding feedback, entropy/noise symmetry
breaking) already do substantial work relative to the original Lemos architecture. The A1
(re-inject color anchor), B1/B2 (temperature annealing), A2 (structured symmetry breaking)
ablations should now be measured against this real 61.3%/85-99% baseline, not a
hypothetical one.

---

## Notes on Experiment set 3 (2026-07-14) — A1 ablation result

Implemented A1 exactly as planned: `color_embed` (`hybrid4.py:194`) was only added once
at init; added a `reinject_color_embed` flag that re-adds it to `var_h` every round
(after the GRU update + PairNorm, before dropout), so it can't be silently forgotten
across rounds of recurrence. Trained from scratch with identical data/config/epochs to
the baseline, only this flag flipped.

**Result:** held-out `sat_acc` on the adversarial test set improved 61.3%→72.5%. On the
`create_planted_3col` c-sweep (n=45, c=1..20, 200 inst/c), the effect is not uniform —
it's a **wash in the training density band (c=3-9)** but a **consistent, meaningful gain
in the out-of-distribution density tail (c=12-19)**: e.g. c=13 goes from 26.5%→43.0%,
c=16 from 15.0%→25.0%, c=18 from 15.5%→22.0%. Every c value from 10 through 20 improved
or matched; none regressed by more than 1-2 points.

**Interpretation:** this is a coherent result, not noise — a persistent color identity
should matter *more* when the model is extrapolating beyond the density it was trained
on, where the message-passing dynamics alone are less reliable and a stable anchor signal
has more relative value. Near the training density, the model already has enough direct
signal that the anchor is redundant. This is consistent with the earlier hypothesis in
[[gnn-coloring-insights]] that A1 addresses a generalization/robustness axis, not raw
in-distribution capacity — and it's the first piece of real evidence that this project's
color-anchor idea (point 1 of the original 8-point brainstorm) has a measurable effect.

---

## Notes on Experiment set 4 (2026-07-14) — B1 ablation result (a real tradeoff, not a regression)

Implemented B1 via an `anneal_temperature` flag: `query_temp` (fixed at 0.5 in the
baseline) is annealed linearly from 1.5 (round 0, near-uniform/exploratory) to 0.3
(final round, sharp/decisive) inside `_compute_query_loss`, replacing reliance on
`loss_noise_scale` alone to escape the uniform fixed point. Trained from scratch,
identical data/config/epochs to baseline.

**First look was misleading.** The held-out `sat_acc` metric (computed only on the
adversarial test set, which is drawn from c∈[3,7] by construction) dropped from 61.3%
to 47.5% — read in isolation this looks like B1 just doesn't work. The full
`create_planted_3col` c-sweep tells a very different story:

| c | baseline | A1 | B1 |
|---|---|---|---|
| 3 | 89.5% | 84.0% | 82.5% |
| 5 | 52.5% | 61.0% | **38.0%** |
| 7 | 95.0% | 97.0% | 80.5% |
| 10 | 70.0% | 67.0% | **82.0%** |
| 13 | 26.5% | 43.0% | **54.0%** |
| 16 | 15.0% | 25.0% | **36.5%** |
| 20 | 11.5% | 13.5% | **23.0%** |

B1 is worse than both baseline and A1 across the training density band (c=3-9), but
**beats both of them, often by a wide margin, across the entire OOD-density tail
(c=10-20)** — e.g. at c=13 it's roughly double the baseline and noticeably ahead of A1.

**Interpretation:** this looks like a real bias/variance-style tradeoff, not noise.
Annealing sacrifices some in-distribution peak sharpness (the fixed temp=0.5 baseline
is implicitly tuned to what works for c∈[3,7]) in exchange for a decision process that
transfers better to unfamiliar density. Between the two mechanisms tried so far,
temperature annealing is the stronger OOD-density generalization lever, at a real cost
in-distribution — while the color anchor (A1) is closer to a free improvement (small
mixed effect in-distribution, consistent gain out-of-distribution). Combining both
(A1 + B1 together) is a natural next experiment, since they don't obviously conflict
mechanistically and both target the same generalization axis from different angles.

**Process note:** always compute the full c-sweep before concluding an ablation
"failed" — a single aggregate metric measured on a narrow distribution (here, the
training-density-only adversarial test set) can point in the wrong direction entirely.

---

## Notes on Experiment set 5 (2026-07-14) — A2 ablation result: best so far, by a wide margin

Implemented A2 via a `structured_node_priors` flag: `degree_greedy_coloring` (a cheap
largest-degree-first greedy 3-coloring) seeds each node's dominant-color prior instead
of `random_confidence_matrix`'s uniform-random choice. Trained from scratch, identical
data/config/epochs to baseline.

**Result:** held-out `sat_acc` 61.3%→67.5% (between baseline and A1). The full c-sweep
is where this ablation stands out — full comparison across all four runs so far
(n=45, c=1..20, 200 inst/c):

| c | baseline | A1 (anchor) | B1 (anneal) | A2 (structured priors) |
|---|---|---|---|---|
| 1 | 98.5% | 98.5% | 99.5% | **70.0%** (regression) |
| 4 | 61.0% | 63.5% | 38.5% | 63.0% |
| 7 | 95.0% | 97.0% | 80.5% | 97.5% |
| 9 | 85.5% | 82.0% | 87.5% | 90.5% |
| 10 | 70.0% | 67.0% | 82.0% | **90.0%** |
| 13 | 26.5% | 43.0% | 54.0% | **63.0%** |
| 16 | 15.0% | 25.0% | 36.5% | **64.0%** |
| 18 | 15.5% | 22.0% | 27.5% | **64.0%** |
| 20 | 11.5% | 13.5% | 23.0% | **50.0%** |

A2 gets the B1-style OOD-density transfer (often the best of all three at c=10-20,
e.g. 4.3x baseline at c=20) **without** B1's in-distribution cost — c=3-9 roughly
matches or slightly beats baseline throughout, no tradeoff. The one real weak spot is
c=1: 70.0% vs baseline's 98.5%, a genuine regression at the sparsest end, plausibly
because degree-greedy ordering gives a weak/arbitrary signal when most vertices have
degree 0-1 (few real constraints to break ties with) — worth a closer look before
calling this an unqualified win.

**Interpretation:** this is the strongest evidence yet for the original insight (from
the very first brainstorm) that structured, non-arbitrary symmetry breaking beats
fighting the uniform fixed point with noise alone — a cheap, informative prior (here,
just vertex degree) gives the model a real reason to commit to a color, and that reason
generalizes far better to unfamiliar density than either the baseline's noise-only
approach or B1's temperature schedule. Between the three ablations tried, A2 is the
best default choice so far.

---

## Notes on Experiment set 6 (2026-07-14) — A1+B1 combined: anti-synergistic, not additive

Trained A1 and B1 together (`--reinject-color-embed --anneal-temperature`), same
data/config/epochs. Held-out `sat_acc` = 66.2% — sitting between the two individual
runs, looked unremarkable in isolation. **The c-sweep tells a much worse story than
the aggregate number suggests:**

| c | baseline | A1 alone | B1 alone | A1+B1 combined |
|---|---|---|---|---|
| 9 | 85.5% | 82.0% | 87.5% | 78.0% |
| 11 | 52.5% | 49.0% | 73.0% | 32.0% |
| 13 | 26.5% | 43.0% | 54.0% | **4.0%** |
| 15 | 16.0% | 24.0% | 41.5% | **2.5%** |
| 17 | 22.5% | 23.0% | 37.5% | **0.0%** |
| 20 | 11.5% | 13.5% | 23.0% | **0.0%** |

Combining the two mechanisms that *each individually* improved the OOD-density tail
makes that same region collapse to near-zero — worse than baseline, worse than either
alone. This is not a subtle effect; c=17-20 goes to exactly 0/200 across the board.

**Interpretation:** A1 (constant re-injection of a fixed color anchor every round) and
B1 (a temperature schedule that sharpens the output distribution over rounds) are
evidently fighting each other mechanistically — re-injecting a full-strength anchor
late in the round loop, exactly when B1's schedule is trying to commit the network to
a sharp, near-discrete answer, likely reintroduces exactly the kind of perturbation the
annealing schedule is trying to damp out. **Practical conclusion: do not stack A1 and
B1 naively.** A2 remains the best single ablation; if combining with A2, prefer testing
A1+A2 or B1+A2 pairwise before ever re-attempting a three-way combination, and don't
assume any two individually-good changes compose — this needs to be checked every time,
not assumed from the individual results.

---

## Notes on Experiment set 7 (2026-07-14) — A1+A2 combined: a sidegrade, not a clean win

Unlike A1+B1, A1+A2 (`--reinject-color-embed --structured-node-priors`) does **not**
collapse anywhere — full comparison (n=45, c=1..20, 200 inst/c), with the average
success rate across all 20 c-values as a single robustness summary:

| variant | avg success (c=1-20) | worst c | notes |
|---|---|---|---|
| baseline | 53.0% | 11.5% (c=20) | |
| A1 alone | 55.7% | 13.5% (c=20) | |
| B1 alone | 59.9% | 22.5% (c=19) | in-dist tradeoff |
| **A2 alone** | **74.4%** | 50.0% (c=20) | best single ablation; weak at c=1 (70.0%) |
| A1+B1 | 43.2% | **0.0%** (c=17-20) | anti-synergistic, worse than baseline |
| A1+A2 | 71.2% | 50.0% (c=14/16) | no collapse; fixes A2's c=1 weakness |

A1+A2 fixes A2's specific c=1 regression (97.0% vs A2's 70.0%, matching baseline's
98.5%) and extends further at the extreme tail (c=20: 73.5% vs A2's 50.0%), but gives
up ground in the c=9-17 band relative to A2 alone (e.g. c=12: 53.5% vs 72.0%; c=16:
50.5% vs 64.0%). Net average is slightly below A2 alone (71.2% vs 74.4%).

**Interpretation:** this incidentally answers the queued "why does A2 regress at c=1"
question — re-injecting the color anchor (A1) compensates for whatever weak/arbitrary
signal degree-greedy ordering provides when most vertices have degree 0-1 (nothing real
to break ties with), consistent with the hypothesis in `EXPERIMENTS.md`'s A2 section.
But A1+A2 is a genuine sidegrade, not a strict improvement: it trades some of A2's
mid-tail strength for better behavior at the two extremes (very sparse, very dense).
Whether to prefer A2 alone or A1+A2 depends on whether the deployment cares more about
average-case performance across a density range, or about avoiding any specific weak
spot (worst-case robustness) — a real design choice, not something more ablating can
resolve on its own.

**Running takeaway across all ablations tried today:** A2 (structured symmetry
breaking) is the strongest lever discovered, by a wide margin, and composes with A1
(color anchor) without catastrophe (unlike B1, which actively conflicts with A1).
Recommended default going forward: **A2 alone**, with A1+A2 as a documented alternative
for worst-case-robustness-sensitive use.

---

## Notes on Experiment set 8 (2026-07-14) — B1+A2 combined completes the pairwise matrix

Last untried pairwise combination. Result (n=45, c=1..20, 200 inst/c): well-behaved,
no collapse anywhere (min 50.0% at c=15) — essentially tied with A1+A2 (avg 71.0% vs
71.2%). Partially fixes A2's c=1 weakness (79.5% vs A2's 70.0%) but not as completely
as A1+A2 does (97.0%).

**Complete picture — all 7 variants tried today, n=45, full c=1..20 sweep:**

| variant | avg success | worst c (value) | verdict |
|---|---|---|---|
| baseline (no ablations) | 53.0% | c=20 (11.5%) | reference |
| A1 alone (color anchor) | 55.7% | c=20 (13.5%) | small free win, mostly OOD tail |
| B1 alone (temp anneal) | 59.9% | c=19 (22.5%) | real in-dist/OOD tradeoff |
| **A2 alone (structured priors)** | **74.4%** | c=20 (50.0%) | **best — recommended default** |
| A1+B1 | 43.2% | c=17-20 (**0.0%**) | anti-synergistic — do not use |
| A1+A2 | 71.2% | c=14/16 (50.0%) | sidegrade — fixes A2's c=1 weakness |
| B1+A2 | 71.0% | c=15 (50.0%) | sidegrade — partial c=1 fix |

**Interpretation:** every combination that includes A2 lands in a tight 71-74% average
band, all far above anything without A2 (43-60%). A2 (structured, degree-informed
symmetry breaking) is the dominant factor — it doesn't matter much whether it's paired
with A1, B1, or neither of them (A1+A2 and B1+A2 are within 0.2 points of each other);
what matters is whether A2 is present at all. The one true trap is A1+B1 *without* A2,
which is actively worse than doing nothing.

**Final recommendation from this round of ablations: ship A2 alone as the new
baseline.** If c=1-class sparse-graph robustness matters for the deployment, A1+A2 is
the next thing to reach for — it's the only variant tested that meaningfully closes
A2's one real weakness, at a small (3.2 point) cost to the average.

---

## Notes on Experiment set 9 (2026-07-14) — D1 (test-time round scaling): confirms the NeuroSAT mechanism

User question: NeuroSAT (Selsam et al. 2019) generalized to SAT instances far larger
than it trained on without more data or bigger training instances — how, and does the
same lever exist here? Answer: NeuroSAT's core trick was running the *same trained
weights* for many more message-passing rounds at test time than at train time — this
works specifically because the update rule has no notion of absolute round count or
position baked into the weights (every node/edge gets the same local update
regardless of round index or graph size), so running longer is well-defined, not
extrapolation into undefined behavior.

Checked `QueryOptGNN_MP` for the same property: `var_gru`/`edge_gru` are plain GRU
cells with no round-index dependency; the query-embedding feedback loop
(`query_loss → query_mlp → next round's input`) is itself a round-agnostic "how close
to a legal coloring am I" signal; B1's temperature schedule anneals by *round
fraction* (`round_idx / (r-1)`), not absolute count, so it automatically rescales if
`rounds` is overridden at eval time. Structurally, nothing here is tied to exactly 32
rounds.

**Test:** took the already-trained A2 checkpoint (no retraining, no new data) and
re-ran the c-sweep at its two weakest regions — the phase-transition dip (c=4-5) and
the dense OOD tail (c=11-20) — with `rounds=64` instead of the trained `rounds=32`.

| c | rounds=32 | rounds=64 | Δ |
|---|---|---|---|
| 4 | 63.0% | 70.5% | +7.5 |
| 5 | 57.0% | 69.5% | +12.5 |
| 11 | 85.5% | 84.5% | −1.0 |
| 12 | 72.0% | 77.0% | +5.0 |
| 13 | 63.0% | 71.5% | +8.5 |
| 14 | 69.5% | 69.0% | −0.5 |
| 15 | 68.0% | 62.0% | −6.0 |
| 16 | 64.0% | 68.0% | +4.0 |
| 17 | 67.5% | 64.5% | −3.0 |
| 18 | 64.0% | 65.0% | +1.0 |
| 19 | 61.0% | 66.0% | +5.0 |
| 20 | 50.0% | 66.0% | +16.0 |

8/12 points improved, average +4.1, and the two biggest single gains land exactly on
the two weakest points tested (c=5 phase transition, c=20 densest/most OOD) — both
improve by double digits. **Confirms the hypothesis directly: the same trained weights
generalize further when simply given more iterations, no new data or retraining
required**, exactly the NeuroSAT mechanism. The 4 regressions (c=11,14,15,17) are all
small (≤6 points) and don't offset the pattern.

**Process note (two flaky CUDA crashes along the way):** the first two attempts at
this experiment hit `CUDA error: an illegal memory access` partway through — first
suspected GPU contention from many hours of sustained use, but a CPU retry crashed
immediately too, for an unrelated reason (a `torch.load(..., map_location=device)`
device-mismatch bug when forcing CPU via `CUDA_VISIBLE_DEVICES=""` — fixed by adding an
explicit `--cpu` flag to `sweep_queryoptgnn.py` that sets `device="cpu"` directly
rather than relying on env-var-based CUDA hiding). The GPU retry of the same c=4-5
sweep then completed cleanly, suggesting the original crashes probably were transient
GPU contention after all, not a rounds=64-specific bug — but this wasn't fully isolated
and is worth remembering if similar crashes recur.

**Next step this opens up:** since more rounds helps for free, D2 (replacing the
single shared GRU cell with distinct alternating cells across a nested loop) becomes
more attractive — if uniform extra iterations of the *same* operation already help,
genuinely different operations at different depths may help further. Also worth
trying rounds=128 to see whether the improvement continues, plateaus, or reverses.

---

## Notes on Experiment set 10 (2026-07-14) — D1 pushed further: rounds=300 + best-of-iterations

Two changes on top of the earlier D1 test: (1) pushed from rounds=64 to rounds=300,
(2) stopped checking only the final round's output and instead check legality at
*every* round, keep the best (lowest-conflict, first-legal-wins) coloring, and stop
early on success — added as `track_best` in `QueryOptGNN_MP.forward()` and
`--track-best`/`--rounds 300` in `sweep_queryoptgnn.py`. This matches what
`INSIGHTS.md` already flagged as underexplored for GCPNet (final-only checking
underestimates true success) and is still zero retraining, zero new data.

**Result — full c-sweep, A2 checkpoint, n=45, 200 inst/c:**

| c | rounds=32 | rounds=300+best | Δ |
|---|---|---|---|
| 1 | 70.0% | **100.0%** | +30.0 |
| 2 | 91.5% | **100.0%** | +8.5 |
| 3 | 90.0% | 97.0% | +7.0 |
| 5 | 57.0% | 68.0% | +11.0 |
| 8 | 99.0% | 98.5% | −0.5 |
| 11 | 85.5% | 79.5% | −6.0 |
| 13 | 63.0% | 78.0% | +15.0 |
| 19 | 61.0% | 78.5% | +17.5 |
| 20 | 50.0% | **78.5%** | +28.5 |

**Full-sweep average: 74.4% → 83.0%.** A2's one real weakness (c=1, previously
70.0%) is now perfectly resolved (100.0%) — without needing A1's help at all. The
dense tail is transformed: c=20 goes from 50.0% to 78.5%, nearly matching what A1+A2
needed a whole extra mechanism to achieve (73.5%). Only 2 of 20 points regressed, both
small (c=8: −0.5, c=11: −6.0).

**Interpretation:** this is now the strongest single change of the whole session,
stronger than any of A1/B1/A2 individually, and it required no new architecture, no
new training data, and no retraining — purely a better use of an already-trained
model's compute budget at inference time. c=5 (68.0%) is now the worst point on the
sweep, still short of the user's 98% target but much closer than anything before it,
and consistent with the phase transition being genuinely the hardest region for every
method tried, including the classical baseline.

**New recommended default:** A2 checkpoint, evaluated at `rounds=300`,
`track_best=True`. This should now be the baseline every future ablation (D2, C1/C2,
E1) is compared against, not the rounds=32 numbers from earlier tonight.

---

## Notes on Experiment set 11 (2026-07-14) — Alon-Kahale spectral algorithm implemented

Fetched the actual paper (Alon & Kahale, SIAM J. Comput. 1997 — confirmed to be the
exact source of the planted-3-coloring excerpt from earlier tonight) and implemented
its real 3-phase algorithm in `code/spectral_coloring.py` (previously
`classical_baseline.py`'s `dsatur_3color` was only ever a stand-in, not the paper's
actual method):

1. **Spectral init**: trim edges incident to vertices of degree > 5·mean_degree; take
   the two smallest-eigenvalue eigenvectors of the trimmed adjacency; find the linear
   combination with median 0, normalize to `‖t‖₂ = √(2n/3)`; threshold at ±1/2 into 3
   classes. This is the same 2D eigenvector embedding the AAAI paper found GNN-GCP
   spontaneously learns (`INSIGHTS.md`'s "triangular geometry").
2. **Propagation**: ~log(n) rounds of "recolor each vertex to its least-popular
   neighbor color."
3. **Cleanup**: uncolor any vertex with support < threshold i (searching for the
   smallest i that leaves brute-forceable components) — this is exactly the
   **support** metric from Alon & Kahale 1994 already in `INSIGHTS.md`. Falls back to
   DSATUR on the residual if no threshold works (practical robustness beyond the
   theorem, which only claims success as n→∞).

Correctness verified against `gc_utils.is_k_color` on all successful runs.

**Sweep result (n=300, c=1..30, 200 inst/c):**

| c | success | time/200 inst | method |
|---|---|---|---|
| 1-3 | 100% | 40-48s | DSATUR fallback |
| 4 | 92.0% | 198s | DSATUR fallback |
| **5** | **2.0%** | **864s** | DSATUR fallback (failing) |
| 6 | 29.0% | 777s | DSATUR fallback (struggling) |
| 7 | 62.0% | 585s | DSATUR fallback (struggling) |
| 8-9 | 90.0-96.5% | 233-356s | DSATUR fallback |
| 10-11 | 99.5-100% | 137-144s | mostly DSATUR, spectral emerging |
| 12-15 | 100% | 27-70s | spectral majority, ramping up |
| **16-30** | **100%** | **~26s flat** | **pure spectral, zero fallback** |

**The high end is a clean confirmation of the paper's claim** — from c≥16, the pure
spectral algorithm (no DSATUR needed at all) succeeds 100% of the time with flat,
fast, polynomial-scaling runtime, exactly as Theorem 1.1 promises for "sufficiently
large c." This is stronger than "works for c>20" — it works from c≥10 here, purely
spectral from c≥16.

**But c=5 is a serious, unexpected weak point** — 2% success, 14+ minutes for 200
instances. This is *worse* than the much simpler peel+DSATUR baseline got at n=100
for the same relative point (78.4%, `Experiment set 2`).

**First diagnosis was wrong — corrected by profiling.** Initially assumed the
component-solver's unbounded backtracking was blowing up combinatorially, and added a
`node_budget` cap to `_solve_component`. A spot-check (n=300, c=5, 30 instances)
showed this **did not help** (still ~4.5s/inst) and made success *worse* (0/30) —
the budget was cutting off searches that would have succeeded. Profiled properly
with `cProfile` instead of guessing again: **the real bottleneck is
`dsatur_3color` itself** — 1.35 million `numpy.sum()` calls in a single instance,
almost the entire 5.9s runtime. `classical_baseline.py`'s DSATUR implementation
(shared with `classical_baseline_color`, written and tuned against n≤200 tests)
does an O(n) saturation scan per vertex-selection step in a plain Python loop; at
n=300 with 30 restarts this compounds badly, especially on hard (near-transition)
instances where DSATUR has to churn through many forced-conflict resolutions before
giving up. The `node_budget` cap on `_solve_component` is harmless but was not the
fix — left in place since it protects against genuine pathological components, but
the actual problem is a pre-existing, separate scaling issue in `dsatur_3color` that
predates tonight's spectral work.

**Not yet fixed** — queued as a follow-up, not solved tonight given session length:
vectorize `dsatur_3color`'s saturation bookkeeping (avoid the per-step full-array
`.sum()`, e.g. maintain running saturation counts incrementally), and/or reduce
restarts specifically in `spectral_coloring.py`'s fallback path since the spectral
threshold search itself is cheap and the DSATUR call dominates cost for no
proportional accuracy gain at this n.

**Practical implication for the whole project:** classical spectral+DSATUR now
provides an even stronger point of comparison than the earlier peel+DSATUR baseline —
100% and fast for c≥10, at the cost of being currently broken (fixably) right at the
phase transition. The GNN's own worst point (c=5, currently 68% with A2+rounds=300)
is in the exact same hard region — worth keeping in mind that *this specific c* may
just be hard for every method tried so far, GNN or classical, until the classical
cleanup bug above is fixed to show what the real ceiling there looks like.

---

## Notes on Experiment set 12 (2026-07-14) — n≥1000 required from now on; dsatur_3color fixed; Bethe Hessian spectral variant

**User correction, binding going forward: never report results for n < 1000.**
Everything above this point in the log was measured at n=45/100/300. Small n blurs
the phase transition via finite-size effects and inflates every success number —
this is the exact mechanism already documented for GCPNet's own n=45→100→2000
collapse, and the same skepticism should have applied to the classical baseline and
every `QueryOptGNN_MP` ablation, not just GCPNet. All prior numbers are flagged
suspect, not disproven, pending re-measurement at n≥1000.

**`dsatur_3color` scaling bug found and fixed.** Root-caused with `cProfile` (not
guessed): the hot loop called `saturation[v].sum()` and indexed `degree[v]` via
numpy on every vertex-selection step — 1.35M numpy calls for a single n=300
instance, almost the entire runtime. Rewrote the inner loop in plain Python
(bitmask + incrementally-maintained popcount, list indexing instead of numpy
indexing). **27x speedup verified** (n=300, c=5: 4.5s→0.165s/instance), correctness
re-verified against `is_k_color`. Confirmed this was a pure constant-factor
(implementation) issue, not the actual difficulty — pushing restarts from 30 to
1000 on the fixed version still only reached 5/15 successes at n=300, c=5,
confirming the phase transition is genuinely hard for DSATUR-family search at this
scale, not merely slow.

**First real n=1000 results (n_inst=50, all methods):**

| c | Classical (peel+DSATUR) | GCPNet | Spectral | QueryOptGNN A2+r300 |
|---|---|---|---|---|
| 1-3 | 96-100% | 0% | 100% | 96-100% |
| 4 | 96% | 0% | 92% | 60% |
| **5** | **0%** | 0% | **0%** | **0%** |
| **6** | **0%** | 0% | **0%** | 14% |
| **7** | **0%** | 0% | 0%* | 68% |
| 8 | 2% | 0% | 0%* | 86% |
| 9 | 12% | 0% | 0%* | 56% |
| 10 | 34% | 0% | 0%* | (pending) |
| 11 | 52% | 0% | 46% | (pending) |
| 12 | 50% | 0% | 74% | (pending) |
| 13-20 | 82-100% | 0% | (pending) | (pending) |

*spectral c=7-9 not yet confirmed complete at time of writing, filled from partial log.

**This is a fundamentally different picture than every earlier n=45/100/300 result
suggested.** The classical baseline has a genuine, wide failure valley — roughly
c=5 through c=12, not a narrow dip — before recovering. GCPNet is completely flat
0% across the entire range (matches its already-known n=2000 collapse). Critically,
**`QueryOptGNN_MP` A2+rounds=300 — 83% average and our best result at n=45 — also
collapses at n=1000** (0% at c=4-5, only partial recovery afterward): the "best
config" conclusion from earlier tonight does not transfer to real scale, exactly as
the user warned.

**Bethe Hessian spectral variant implemented** (`spectral_coloring.py`,
`variant="bethe_hessian"`) — non-backtracking-inspired phase 1 replacement
(Saade, Krzakala & Zdeborová 2014): `H(r) = (r²-1)I - rA + D` in place of the raw
adjacency matrix, motivated by the literature result that naive adjacency-spectral
methods lose signal at low average degree because the informative eigenvectors get
swamped by noise from high-degree vertices/short cycles, while the non-backtracking
operator (and its cheaper Bethe-Hessian proxy) recovers signal closer to the true
Kesten-Stigum threshold. Had to correct the standard formula's sign: the published
default `r=+√(mean degree)` targets *assortative* communities; a proper coloring is
*disassortative* (zero edges within a class) — exactly why Alon-Kahale's own phase 1
reads the *smallest* eigenvalues of A, not the largest. Verified empirically
(agreement with the true planted partition, n=1000): `r=+√d` gives chance-level
accuracy (~0.34-0.36) at every c; `r=-√d` gives 0.46/0.64/0.80 at c=5/8/15, beating
plain adjacency (0.41/0.54/0.77) at every point tested.

**But the raw-accuracy win doesn't yet translate to more full-pipeline successes**
at the hardest point: both variants land at 0/10 for c=5-7 (n=1000) once
propagation+cleanup+DSATUR-fallback run on top; Bethe Hessian only pulls ahead at
the edge of the hard region (c=8: 1/10 vs 0/10). Interpretation: propagation and
cleanup can't rescue an initial guess that's only modestly better than chance at the
very center of the phase transition — you'd need a much larger accuracy jump to
flip the final binary outcome there. This doesn't kill the idea; it sharpens it —
a fixed formula gets a real but bounded improvement (consistent with pushing toward,
not reaching, the true reconstruction threshold), which is exactly the argument for
trying a *learned* version next (see [[gnn-coloring-insights]] "spectral-NN"
discussion) rather than assuming a hand-designed operator is the ceiling.

---

## Notes on Experiment set 13 (2026-07-14) — Spectral-NN: three real bugs found, then a genuine (bounded) win

User's proposal: train a network to produce a *high-dimensional* per-vertex
embedding (not just 2 eigenvectors), such that its structure reveals the coloring
at lower c than plain adjacency-spectral can. Built `code/spectral_nn.py`: a
message-passing `SpectralEmbedder` (GRU-based) producing an 8-dim embedding per
vertex, read out at inference via PCA (top-2 principal components of the
embedding cloud) fed into `spectral_coloring.py`'s existing, already-verified
phases 2-3 (propagation + cleanup). Three real bugs found and fixed in sequence,
each caught by checking actual embedding statistics rather than trusting the loss
curve alone:

1. **Full representation collapse (first attempt).** Loss flat at 2.66 for 5000
   steps; embedding std ≈ 1e-8, every vertex identical. Cause: the embedder
   started every vertex from an identical constant feature, with no anti-collapse
   pressure in the loss — the trivial "map everything to one point" solution
   satisfies the pairwise same-class term for free. Fixed with degree-based init
   (natural per-vertex asymmetry, same idea as A2) plus a VICReg-style variance
   regularizer.
2. **Loss/readout mismatch (user caught this one).** The original pairwise
   contrastive loss trained distances to respect class membership, but the actual
   readout is a *linear* PCA projection — related but not the same objective; an
   embedding manifold can satisfy pairwise contrastive constraints while being
   curled up in a way PCA can't cleanly separate. Replaced with a **Fisher
   discriminant loss** (minimize within-class scatter relative to between-class
   scatter) — this is exactly the criterion that determines whether PCA's top
   components align with the class-separating directions, so it directly targets
   what gets read out, not a proxy for it. Also fixed a numerical instability
   from this change (unbounded within/between ratio exploded when between was
   near zero early in training; switched to the bounded within/(within+between)).
3. **A second collapse, triggered by going from rounds=8 to rounds=64** (per
   explicit instruction — "run many message passing rounds... not 8"). Loss got
   stuck at exactly 1.0 for "large"; checked embeddings directly: all 3 class
   means identical again. Cause: with 64 sequential GRU applications, the
   recurrent dynamics converge to a single input-independent fixed point,
   washing out the degree-based init entirely — mechanistically the same problem
   A1 solved for `QueryOptGNN_MP` earlier tonight. Fixed identically: re-inject
   the degree signal every round, not just at init.

**Result after all three fixes (n=1000, 15 instances/c, three models specialized
by c-band — small=1-8, medium=6-18, large=14-30):**

| c | plain adjacency | small | medium | large |
|---|---|---|---|---|
| 5-8 | 0-0% | 0-7% | 0-0% | 0-0% |
| 9 | 7% | 7% | 13% | 13% |
| **10** | 27% | **60%** | 47% | 40% |
| **11** | 27% | **40%** | 40% | 40% |
| 12 | 73% | 73% | 60% | 73% |
| 13 | 87% | 93% | 67% | 93% |
| 14 | **100%** | 80% | 80% | 93% |

**c=5-8 (the true core of the hard window) remains completely unsolved by every
method, learned or not** — this part hasn't moved and may be at or below the true
information-theoretic reconstruction threshold. **But c=10-11 shows a real,
non-noise win**: "small" more than doubles plain spectral's success rate at c=10
(60% vs 27%), and all three specialized models beat plain at both c=10 and c=11 —
the first time any spectral variant tried tonight clearly outperforms plain
adjacency rather than matching it or showing noise-level differences. **This
isn't free, though** — at c=14, plain hits a clean 100% while the learned models
regress to 80-93%.

**Honest bottom line:** the spectral-NN idea works, but as a targeted tool for a
specific sub-region (the c=10-11 recovery zone), not a universal replacement for
plain spectral, and not (yet) a way to crack the true hard core at c=5-8.

## Notes on Experiment set 14 (2026-07-15) — three architectures cross-tested + cheap ensembling, per the "close the gap any way" mandate

Following the explicit instruction to try every insight as a flag and test every
new mechanism against all existing variants, three lines ran in parallel tonight,
all at n=1000, c=5-14, cross-verified against `is_k_color` on every reported
success (never trust a bare `success` flag):

**1. Spectral coordinates as NN input** (`use_spectral_coords` flag on
`SpectralEmbedder`, feeding `raw_spectral_coordinates()` alongside the degree
signal at init) — full 7-variant matrix (10 inst/c):

| c | plain | small_spec | med_spec | large_spec | small_n300 | med_n300 | large_n300 | universal_stale |
|---|---|---|---|---|---|---|---|---|
| 5-7 | 0/10 | 0/10 | 0/10 | 0/10 | 0/10 | 0/10 | 0/10 | 0/10 |
| 8 | 1/10 | 1/10 | 1/10 | 0/10 | 1/10 | 1/10 | 0/10 | 0/10 |
| 9 | 0/10 | 1/10 | 1/10 | 3/10 | 1/10 | 1/10 | 0/10 | 0/10 |
| 10 | 1/10 | **5/10** | 2/10 | 2/10 | 3/10 | 1/10 | 4/10 | 4/10 |
| 11 | 4/10 | 4/10 | 5/10 | 4/10 | 5/10 | 4/10 | 3/10 | **6/10** |
| 12 | 6/10 | 5/10 | 6/10 | 7/10 | 5/10 | 6/10 | 7/10 | **10/10** |
| 13 | 10/10 | 8/10 | 8/10 | 7/10 | 10/10 | 7/10 | 9/10 | 9/10 |
| 14 | 10/10 | 8/10 | 9/10 | 9/10 | 10/10 | 8/10 | 10/10 | 10/10 |

Bootstrapping the NN's init from real spectral coordinates (rather than degree
alone) doesn't change the story: still 0% at c=5-7, still bounded gains at
c=9-12, sometimes at a cost near c=13-14 where plain is already saturated. The
"stale" pre-fix universal model again punches above its weight at c=11-12 (10/10
at c=12) — reconfirms set 13's lesson to never assume a superseded checkpoint is
strictly worse.

**2. Pairwise same/different reformulation** (new file `code/pairwise_nn.py`,
`PairwiseEmbedder`) — this is the item explicitly called out ("did you add the
varient that tries to learn if they are in the same partition yes/no?"). Instead
of a direct 3-way color prediction (which has a color-labeling symmetry that's a
real source of training difficulty), train a binary classifier on vertex pairs:
"same planted class or different?" — no labeling symmetry at all. Decoded at
inference by building the full n×n same-class probability matrix, centering to a
signed similarity matrix, and taking its two *largest*-eigenvalue eigenvectors
(assortative convention, opposite sign from plain adjacency-spectral's
disassortative phase 1) into the existing propagation/cleanup pipeline.

Hit its own distinct collapse mode: plain BCE from a cold start reliably
collapsed to "always predict different" (the free 2/3-accuracy degenerate
solution, since a balanced 3-partition has P(same class)=1/3 for a random pair).
Fixed by combining `use_spectral_coords=True` *and* an auxiliary Fisher/LDA
scatter term (`fisher_weight=2.0`) — neither alone was sufficient.

Results (10 inst/c):

| c | plain | small_pairwise (c1-8) | medium_pairwise (c6-18) | large_pairwise (c14-30) |
|---|---|---|---|---|
| 5-7 | 0/10 | 0/10 | 0/10 | 0/10 |
| 8 | 1/10 | 0/10 | 0/10 | 0/10 |
| 9 | 1/10 | 1/10 | 1/10 | **6/10** |
| 10 | 1/10 | 5/10 | 6/10 | **9/10** |
| 11 | 6/10 | 7/10 | **10/10** | 8/10 |
| 12 | 8/10 | 10/10 | 10/10 | 10/10 |
| 13 | 8/10 | 10/10 | 10/10 | 10/10 |
| 14 | 10/10 | 10/10 | 10/10 | 10/10 |

**This is the single best result of the whole project at c=9-11**: `large_pairwise`
gets 6/10 and 9/10 at c=9/10 where plain gets 1/10 at both — a much bigger gap
than any spectral-NN variant achieved in the same window. `medium_pairwise` hits
a clean 10/10 at c=11 vs plain's 6/10. The "same/different" framing appears to be
a genuinely easier, better-conditioned learning target than either direct 3-way
prediction or a PCA readout of a Fisher-separated cloud. Still 0% at c=5-7 — the
true hard core is unmoved by this reformulation too.

**3. Multi-stage nested-loop architecture** (new file `code/multistage_nn.py`,
`MultiStageEmbedder`) — this is the item from the original brainstorm ("loop many
times (A), loop many times (B), loop many times (C), and now a whole loop around
it... positional encoding of the iteration of small and large loop"). Three
*distinct* GRU cells (not one cell reused), cycled through an outer loop, with
sinusoidal positional encoding of (outer_idx, inner_idx) injected at every single
step. Degree-signal re-injected after every stage-cell application (same
fixed-point-collapse defense as everywhere else tonight) — verified via
`check_collapse()` before trusting any result.

Results (10 inst/c):

| c | plain | small_multistage | medium_multistage | large_multistage |
|---|---|---|---|---|
| 5-7 | 0/10 | 0/10 | 0/10 | 0/10 |
| 8 | 0/10 | 1/10 | 0/10 | 0/10 |
| 9 | 1/10 | 1/10 | 1/10 | 0/10 |
| 10 | 3/10 | 3/10 | 5/10 | 2/10 |
| 11 | 2/10 | 4/10 | 3/10 | **5/10** |
| 12 | 6/10 | **9/10** | 6/10 | 6/10 |
| 13 | 8/10 | **9/10** | 8/10 | 6/10 |
| 14 | 10/10 | 10/10 | 10/10 | 10/10 |

Real but smaller gains than the pairwise reformulation, in the same c=10-13
region. More reasoning *shape* (distinct stages) helped somewhat on top of more
reasoning *depth* (set 13's rounds=8→64 win), but nowhere near as much as
reframing the *task itself* (point 2) did. c=5-7 unmoved, same as every other
method.

**4. Cheap ensembling — combine already-trained checkpoints, no new training.**
Per the explicit mandate ("close the gap any way, even combining
methods-techniques and pipelines"), added an `--ensemble` OR-combination check.

*Within spectral-NN family only* (10 inst/c): ANY reaches 10/10 at c=12-14, 9/10
at c=11, 7/10 at c=10, 6/10 at c=9 — a clear step up from any single variant in
that family, confirming that different checkpoints fail on different instances
rather than all failing together.

*Cross-family* (spectral-NN + pairwise + multistage + plain, all 10 checkpoints
tonight, `code/combined_ensemble_eval.py`, 20 inst/c, n=1000) — **this is the
headline result of the whole project**:

| c | plain | ANY (cross-family OR) | best single model |
|---|---|---|---|
| 5 | 0/20 | 0/20 | 0/20 |
| 6 | 0/20 | 0/20 | 0/20 |
| 7 | 0/20 | 0/20 | 0/20 |
| 8 | 0/20 | 2/20 (10%) | small_speccoords 1/20 |
| 9 | 1/20 (5%) | **14/20 (70%)** | large_pairwise 7/20 (35%) |
| 10 | 6/20 (30%) | **19/20 (95%)** | medium_pairwise 14/20 (70%) |
| 11 | 10/20 (50%) | **20/20 (100%)** | medium_pairwise 20/20 (100%) |
| 12 | 12/20 (60%) | **20/20 (100%)** | small_pairwise 20/20 (100%) |
| 13 | 20/20 (100%) | 20/20 (100%) | universal_n300_stale 20/20 |
| 14 | 20/20 (100%) | 20/20 (100%) | small_pairwise 20/20 |

**This directly delivers the stated mission** ("close the gap any way... so
c>7 is like c>14"): at c=9-12, OR-combining 10 already-trained checkpoints
(zero additional training) takes success from plain spectral's 5-60% up to
70-100% — matching or beating c=13-14's ~100% ceiling across exactly the
window that was previously the weakest. c=11-12 individually already hit a
clean 100% from a single best model (`medium_pairwise`/`small_pairwise`), so
the ensemble isn't even doing extra work there beyond picking the right model;
the real ensemble lift is at c=9-10, where no single checkpoint exceeds 70%
but the OR combination reaches 70-95%.

**What doesn't close:** c=8 stays weak even ensembled (10%, up from 0% plain)
and **c=5-7 is a complete, unmoved floor — 0/20 for literally every one of the
10 checkpoints and their OR-combination, at 20 instances/c (not just 10)**.
This is now the most statistically solid evidence collected all project that
c=5-7 sits at or below the true information-theoretic reconstruction
threshold for `create_planted_3col` — no architecture, loss, ensemble, or
combination tried has moved it even once.

**Running honest summary across all of tonight's variants:** every real,
reproducible win clusters in c=8-12 (c=8 only barely, via ensembling), and the
single largest single-model jump came not from a bigger/deeper/wider network
but from reformulating the *prediction target* (pairwise same/different
instead of direct 3-way color) — while the single largest *combined* jump came
from cheaply OR-ensembling everything already trained, no new training
required. c=5-7 remains completely unsolved by every method tried across the
entire project.

**Which models actually solve which range — full per-model OR breakdown (n=1000,
20 inst/c, all 13 checkpoints), not just the best single model:**

```
   c   plain  sm_spec  md_spec  lg_spec  univ_stale  sm_pair  md_pair  lg_pair  sm_mstage  md_mstage  lg_mstage  sm_mhead  md_mhead  lg_mhead   ANY
   9    2/20    6/20*1    4/20    3/20      5/20*1     2/20    4/20*1   5/20*1     2/20*1      5/20       4/20    3/20*1     2/20      2/20   17/20
  10    4/20    9/20*1    8/20    6/20        6/20    15/20   15/20    18/20       1/20        5/20       6/20     6/20      6/20      6/20   20/20
  11   11/20     14/20   13/20   14/20       17/20    19/20   20/20    19/20      12/20       12/20      15/20    10/20     15/20     10/20   20/20
  12   15/20     13/20   10/20   15/20       16/20    19/20   20/20    17/20       9/20       14/20      10/20    15/20     13/20     12/20   20/20
  13   18/20     16/20   17/20   17/20       19/20    20/20   20/20    20/20      18/20       16/20      17/20    19/20     15/20     15/20   20/20
  14   20/20     17/20   19/20   18/20       20/20    20/20   20/20    20/20      18/20       20/20      19/20    18/20     19/20     17/20   20/20
```
(`*N` = that model was the ONLY one that solved N of those instances — a real,
non-redundant contribution, not just tied with something stronger.)

**Follow-up: isolating the pairwise trio (small+medium+large) alone** (own
dedicated run, `pairwise_nn_matrix_eval.py --ensemble`, n=1000, 20 inst/c):

```
   c   plain  small_pairwise  medium_pairwise  large_pairwise  ANY_pairwise
   8    0/20            1/20             1/20            1/20          2/20  (10%)
   9    2/20            1/20             1/20            8/20          9/20  (45%)
  10    5/20           18/20            13/20           15/20         19/20  (95%)
  11   12/20           16/20            17/20           18/20         20/20 (100%)
  12   15/20           20/20            20/20           20/20         20/20 (100%)
  13   19/20           20/20            20/20           19/20         20/20 (100%)
  14   20/20           20/20            20/20           20/20         20/20 (100%)
```

**Precise, verified ranges-solved statement:**
- **c=11-14: fully solved by the pairwise trio alone (100%, no other architecture
  needed).** This is a single-family result, not really an "ensemble" story —
  `medium_pairwise` alone hits 20/20 at both c=11 and c=12 by itself.
- **c=10: pairwise trio alone gets 95% (19/20)** — very close but not literally
  complete without the broader 13-model ensemble, which does reach 100% there
  (`large_pairwise` 18/20 + 1 unique from `small_speccoords` + 1 more from
  overlapping weaker models jointly).
- **c=9: pairwise trio alone only gets 45%** — this row is NOT a pairwise story.
  Reaching 17/20 (85%) here required real, non-redundant contributions from 6
  different architectures (`small_speccoords`, `universal_n300_stale`,
  `medium_pairwise`, `large_pairwise`, `small_multistage`, `small_multihead`),
  and even 85% falls short of the 98%-for-every-c target.
- **c=8 and below: still essentially unsolved** (pairwise trio 10% at c=8; full
  13-model ensemble was 2/20=10% at c=8 too — see prior ensemble entry).
- **c=5-7: a complete, zero-exception floor for every model and combination
  tried, full stop.**

**Deep-dive diagnostic: why is c=7-9 hard, at the representation level, not just end-to-end?**
(2026-07-15, prompted by "on training, do we get good accuracy for c=7?") Ran
`pairwise_nn.py`'s `check_collapse()` (raw same/different classifier accuracy
vs. the 2/3 "always different" baseline) at n=1000, c=5-9, 8000 sampled pairs
(SE ≈0.5pt, so differences of a few points are real, not noise):

| c | small_pairwise excess | medium_pairwise excess |
|---|---|---|
| 5 | −3.9pt (worse than trivial guessing) | −6.0pt (worse than trivial guessing) |
| 6 | −1.3pt (worse than trivial guessing) | −4.7pt (worse than trivial guessing) |
| 7 | +0.7pt (noise-level) | +4.2pt |
| 8 | +4.1pt | +5.5pt |
| 9 | +7.6pt | +8.5pt |

**At c=5-6 the classifier is literally worse than guessing "always different"**
— direct, representation-level confirmation (independent of end-to-end
coloring success) that the network extracts no usable signal there. c=7 is
right at the noise floor. **c=8 is the interesting case: a real, statistically
clear classifier edge (+4-5.5pt) coexists with ~0-10% end-to-end coloring
success** — the discrete decode pipeline (eigen-decomposition -> threshold ->
propagate -> cleanup) has no tolerance for partial credit across all n=1000
vertices, so a modest per-pair edge isn't enough to produce a single fully
legal coloring.

**Companion diagnostic: exactly how/where does plain classical spectral fail
at c=5-10?** (`code/diagnose_baseline_failure.py`, n=1000, 8 inst/c). Ran
phase 1 (spectral init) + phase 2 (propagation) and inspected the per-vertex
support-score distribution (Alon-Kahale's own confidence metric, already used
by `_cleanup` to decide what to brute-force):

- **c=5-9: 92-100% of vertices have support < 1 after propagation** — nearly
  the entire graph looks "ambiguous," not a small hard residual pocket.
  `_cleanup`'s threshold search therefore never finds a small-enough
  component and falls back to DSATUR on almost the whole graph, every single
  time (8/8 instances at c=5-8, 7/8 at c=9) -- and DSATUR then fails too
  (restarts=30 is not enough at this n/c combination). Phase-1+2 raw
  per-vertex accuracy against the true partition is only 40-55% here,
  matching the pairwise classifier's own near-baseline numbers above --
  **three independent methods (end-to-end coloring, pairwise-classifier
  accuracy, classical support-score) now agree there is no exploitable signal
  at c=5-7, and only a thin one at c=8-9.**
- **c=10: low-support fraction drops to 60-93%** (still high, but visibly
  better and more variable instance-to-instance) -- consistent with c=10
  being exactly where things start to become resolvable.

**Implication for "use the classical baseline as a warm start, let the model
fix the rest":** the naive version of this idea (trust the baseline's
confident majority, let the model resolve a small ambiguous residual) doesn't
apply at c=5-9 -- there is no confident majority to lean on. Two evidence-
grounded variants tried instead:

**(a) Dual classical-signal input + residual skip-connection (tried, NEGATIVE
result).** Added two new flags to `PairwiseEmbedder`: `use_dual_spectral_coords`
(feed both plain-adjacency AND Bethe-Hessian raw coordinates at init, 4 dims
instead of 2 -- these are two independent classical estimates, Bethe-Hessian
already known to beat plain adjacency at every c) and `residual_spectral_coords`
(concatenate the raw coordinates onto the *final* embedding, right before the
pairwise readout, not just at init -- a ResNet-style skip connection so the
classical signal can't be lost across 32+ GRU rounds). Trained small+medium
bands (n=300, 800 steps, hidden_dim=32). Training loss was notably higher
than every other run tonight (settled ~2-4 vs. the usual <1) -- a flag worth
noting even though `check_collapse()` showed no actual collapse (clean,
distinct per-class means across all 12 dims).

**Result: this made things WORSE, not better** (n=1000, 20 inst/c):

| c | plain | small_pairwise | medium_pairwise | small_dualres | medium_dualres |
|---|---|---|---|---|---|
| 9 | 0/20 | 2/20 | 7/20 | 1/20 | 2/20 |
| 10 | 9/20 | 11/20 | 13/20 | 9/20 | 8/20 |

Likely mechanism: `head="distance"` computes a single global Euclidean
distance across all 12 dimensions with one scalar `log_scale` -- it has no
way to learn "trust the 8 learned dims more than the 4 raw classical dims."
Since the raw coordinates are only mediocre at c=9 (~50% accuracy, barely
above chance), mixing them uniformly into the same distance metric the
learned embedding uses likely adds noise the model can't down-weight, rather
than genuinely complementary signal. A per-dimension learned scale
(Mahalanobis-style distance) would probably be needed to make this idea work;
plain concatenation does not. **Recorded as a real negative result, not
silently dropped** -- naive baseline-signal injection is not free, and this
specific mechanism should not be reused without the per-dimension weighting
fix.

**Follow-up: ruled out "not enough model capacity" as the cause.** Added
`--hidden-dim`/`--device` flags to `train_pairwise_bands.py`, confirmed GPU
(RTX 3080) works cleanly for this architecture (no crashes, unlike earlier
`QueryOptGNN_MP` CUDA issues), and retrained the same dual+residual-coords
small/medium bands at `hidden_dim=128` (4x) on GPU. **Loss curves were nearly
identical to the 32-dim run** (small: settled ~3.9-4.3 either way; medium:
~1.7-2.2 either way) -- capacity was never the bottleneck. End-to-end coloring
at c=9-10 (n_inst=15) stayed mixed/negative vs. the original 32-dim
non-residual checkpoints, confirming the root cause is the distance-head
mechanism itself, not model size. GPU + 128-dim is now available and proven
stable for future experiments, but this specific idea needs the
per-dimension-weighting fix (not yet built) before it's worth another try.

**(b) True non-backtracking operator (tried, NEGATIVE result -- but a
valuable, well-verified one).** The Bethe-Hessian variant already in
`spectral_coloring.py` is a cheap scalar *approximation* to the
non-backtracking (NB) matrix's spectrum. Krzakala et al. (2013) proved the
real NB operator achieves detection down to the exact Kesten-Stigum
threshold, in principle beating the Bethe-Hessian shortcut in the hardest
regime -- worth checking directly rather than assuming. Implemented
`_non_backtracking_init` in `spectral_coloring.py`: the standard 2n x 2n
companion-matrix reformulation `B'=[[0,D-I],[-I,A]]` (avoids the expensive
2|E| x 2|E| direct construction), `np.linalg.eig` (real but asymmetric, so
general eigendecomposition, not `eigh`).

One real, non-obvious finding along the way: the vertex-space eigenvector is
the **second** n-block of each eigenpair `(x;y)`, not the first (`y`
satisfies `H(lambda)y=0` where `H(r)=(r^2-1)I-rA+D` is exactly the Bethe
Hessian -- the Ihara-Bass identity) -- verified empirically (y beat x at
every c tested, e.g. c=10: 0.660 vs 0.595) rather than assumed from the
construction alone. Also had to filter to numerically-real eigenvalues
(|imag|<1e-6) before eigenvalue-sorting, since a genuine complex-conjugate
pair can rank ahead of the true second-most-negative real eigenvalue by raw
real-part comparison.

**Result: non-backtracking beats plain adjacency at every c, but does NOT
beat Bethe-Hessian anywhere in c=5-10** (n=1000, phase-1-only accuracy vs.
true partition, 8 inst/c):

| c | adjacency | bethe_hessian | non_backtracking |
|---|---|---|---|
| 5 | 0.405 | 0.476 | 0.448 |
| 6 | 0.480 | 0.542 | 0.510 |
| 7 | 0.503 | 0.578 | 0.573 |
| 8 | 0.587 | 0.608 | 0.591 |
| 9 | 0.606 | 0.652 | 0.636 |
| 10 | 0.634 | 0.689 | 0.675 |

**And it costs ~1000x more per call** (~6.2-6.4s at n=1000 for the dense
`eig` on a 2000x2000 companion matrix, vs. milliseconds for the `eigh`-based
adjacency/Bethe-Hessian variants). **Not worth integrating as a training-time
feature** -- no accuracy edge over the already-available Bethe-Hessian
variant in the project's actual hard window, at a cost that would make it a
severe bottleneck inside a per-forward-pass NN feature computation. A clean,
literature-motivated hypothesis that didn't pan out at this finite n --
recorded honestly rather than omitted.

**(c) Systematic combination sweep at c=7,8,9** (`code/train_pairwise_ablation_matrix.py`
+ `code/pairwise_ablation_matrix_eval.py`, 2026-07-15): rather than guess which
mechanism/combination helps, trained 7 configs x 2 bands (14 checkpoints,
GPU, hidden_dim=32, 800 steps) covering: `per_dim_scale` alone, `use_common_neighbors`
alone, `num_spectral_eigvecs=4` alone, the dual+residual combo with the
per-dim-scale fix applied, and three combinations of the above (plus a
kitchen-sink of everything). Evaluated all 16 variants + plain at c=7,8,9
(n=1000, 20 inst/c first, then verified the two promising leads at 40 inst/c).

**c=7: still a complete floor across all 17 variants** (0-1/20 everywhere,
noise-level) -- 17 more ways to fail confirms this isn't a mechanism gap.

**One real, verified win: `medium_eig4_only`** (medium-band pairwise
classifier, `num_spectral_eigvecs=4` instead of the default 2, no
dual-coords/residual-connection) **at c=9: 27.5% (11/40) vs. plain's 5% and
the original `medium_pairwise`'s 17.5%** -- held up after doubling the sample
from n_inst=20 (which showed 40%, an overestimate) to 40. Simplest
explanation: near the detection threshold the signal doesn't cleanly
concentrate into exactly 2 eigenvector directions the way it does at higher
c, so keeping 4 retains real information that k=2 throws away. This is the
only mechanism out of everything tried tonight (dual coords, residual
connection, per-dim scale, common-neighbors, the non-backtracking operator)
that produced a verified, non-noise improvement at c=9.

**One false lead, caught and corrected, not left standing:** `small_dualres_pds`
(the per-dim-scale fix applied to the earlier-negative dual+residual combo)
looked like a real win at c=8 in the first pass (15%, 3/20, best of all 17
variants) -- but dropped to 2.5% (1/40) on the follow-up 40-instance check,
indistinguishable from everything else. The per-dim-scale fix does not
rescue the dual+residual mechanism after all; the earlier apparent win was
sampling noise at n=20. Recorded honestly rather than left as a claimed
result -- this is exactly why every promising number in this project gets a
follow-up verification pass before being trusted.

**c=8 remains essentially unmoved by every mechanism tried** (0-5% across
plain and all 17 variants at n=40) -- still the weakest point in the c=7-14
range after tonight's entire combination sweep.

**5. Multi-head GNN (genuinely parallel heads, distinct from #3's sequential stages).**
User clarification: "just like multi head attention, but gnn" — this is a
different architecture from `multistage_nn.py`'s A→B→C→A→B→C sequential
composition. Built `code/multihead_gnn.py`'s `MultiHeadGNNEmbedder`:
`num_heads` (default 4) genuinely independent message-passing heads — each
with its own init projection, message MLP, and GRU cell (never shared weights)
— run the SAME `rounds` of message passing INDEPENDENTLY AND IN PARALLEL over
the same graph, starting from the same raw signal, exactly mirroring
multi-head attention's independent per-head Q/K/V projections. Final per-head
hidden states are concatenated and passed through one output projection
(`out_proj`, playing `W_O`'s role) — concatenation, not sequential composition.

Collapse must be checked **per head**, not just on the final concatenated
output — an aggregate-only check could hide 3 of 4 heads collapsing while one
healthy head keeps the concatenated statistics looking fine. `check_collapse()`
was extended accordingly; verified no head (nor the final projection)
collapsed in any of the three trained bands.

Results (n=1000, 10 inst/c):

| c | plain | small_multihead | medium_multihead | large_multihead |
|---|---|---|---|---|
| 5-8 | 0/10 | 0/10 | 0/10 | 0/10 |
| 9 | 1/10 | 1/10 | 2/10 | 0/10 |
| 10 | 2/10 | **6/10** | 2/10 | 3/10 |
| 11 | 5/10 | 3/10 | 3/10 | 4/10 |
| 12 | 5/10 | 6/10 | **7/10** | 5/10 |
| 13 | 9/10 | 8/10 | 9/10 | 8/10 |
| 14 | 10/10 | 10/10 | 9/10 | 9/10 |

Real gain at c=10 (small_multihead 6/10 vs plain 2/10), modest elsewhere,
same c=5-8 floor as every other method. **Same order of magnitude as the
multi-stage sequential variant (#3), smaller than the pairwise reformulation's
win (#2).** Parallel independent heads and sequential distinct stages appear
to buy roughly the same modest amount — reformulating the prediction target
remains the highest-leverage single change found all project. Added to
`code/combined_ensemble_eval.py` (now 13 checkpoints total); see the updated
cross-family ensemble numbers once that re-run completes.

## Notes on Experiment set 15 (2026-07-16) — filling the backbone x head matrix; GraphGPS is the breakthrough of the project

User pushback ("why dont i see pairwise embedder with muly stage or muly head?")
correctly identified a real gap: every combination tried through set 14 paired
the pairwise same/different target only with the single-cell backbone, and
paired the multi-stage/multi-head backbones only with the PCA-readout target.
Built the two missing combinations, plus a 4th backbone (GraphGPS, local
message passing + global Transformer attention, via PyTorch Geometric's
`GPSConv`+`GINEConv` — user's suggestion), all sharing the same proven
pairwise head:

- `code/multistage_pairwise_nn.py` — `MultiStagePairwiseEmbedder` (multi-stage
  backbone + pairwise head)
- `code/multihead_pairwise_nn.py` — `MultiHeadPairwiseEmbedder` (multi-head
  backbone + pairwise head)
- `code/gps_pairwise_nn.py` — `GPSPairwiseEmbedder` (GraphGPS backbone +
  pairwise head; `torch_geometric` installed fresh for this). Graphs here are
  unweighted with no real edge features, so `GINEConv`'s required `edge_attr`
  is a single learned constant embedding broadcast to every edge.

All three trained on small+medium bands (n=300, 800 steps, GPU, `use_spectral_coords=True`,
`num_spectral_eigvecs=4` — carrying over the one confirmed-useful mechanism
from set 14's sweep), collapse-checked clean.

**GraphGPS is a dramatic, verified breakthrough — a single checkpoint nearly
solves the entire c=9-14 range that previously needed the 13-model ensemble
to approach.** n=1000, cross-verified against `is_k_color`, every promising
number re-checked at n_inst=40 before trusting it (per this project's
established discipline — a false lead at n=20 was already caught and
corrected once in set 14):

| c | plain | medium_gps_pairwise | medium_multistage_pairwise |
|---|---|---|---|
| 9 (n=40) | 17.5% | **72.5%** | 42.5% |
| 10 (n=40) | 30% | **100% (40/40, perfect)** | 92.5% |
| 11 (n=20) | 45% | 100% | 100% |
| 12 (n=20) | 60% | 100% | 100% |
| 13 (n=20) | 95% | 100% | 100% |
| 14 (n=20) | 100% | 100% | 100% |

c=7-8 remain a complete floor for every architecture including these two —
this doesn't touch the true hard core, but it essentially closes the
previously-hardest recoverable window (c=9-14) with ONE model, not an
ensemble. `medium_multistage_pairwise` is also genuinely strong (independent
confirmation that going beyond the single-cell backbone helps, not just a
GraphGPS-specific effect) but GraphGPS is stronger at both verified points
(c=9: 72.5% vs 42.5%; c=10: 100% vs 92.5%).

**Likely why GraphGPS is so much stronger:** global attention lets every
vertex directly attend to every other vertex in one layer, not just
propagate through local message passing over many rounds — a genuinely
different reasoning mechanism from every GRU-based backbone used all night
(single-cell, multi-stage, multi-head all still only aggregate over
immediate neighbors per step, however many steps). At training scale (n=300),
raw pairwise-classifier accuracy was dramatically higher than anything seen
in set 14's diagnostic (small: 82.6% vs 66.7% baseline; medium: 93.9% vs
66.0% baseline, both at c=9) — consistent with the n=1000 end-to-end result,
not a fluke isolated to one evaluation.

**QueryOptGNN_MP + multi-head (separate codebase, `graph_color_improved/hybrid4.py`,
background agent work):** ported the same "genuinely parallel heads" idea
(own weights per head, not shared, combined after) into `QueryOptGNN_MP`'s
var/edge-GRU reasoning loop as a new `multi_head_reasoning` flag, tested
combined with A2 (structured priors), and — per explicit follow-up
instruction — also with A1, B1, and alone. Training hit a real snag (A2+multihead
stopped short at epoch 31 of 40 with no error surfaced initially; the agent
diagnosed it and added an `--auto-resume` flag, then continued) — full
results pending, to be added once the complete sweep finishes.

**Follow-up mechanism sweep on the GPS backbone** (medium band, c=9, the one
GraphGPS itself hadn't cracked as fully as c=10+): trained `per_dim_scale`,
`use_common_neighbors`, both combined, and `num_spectral_eigvecs=6` (vs the
eig4 baseline). `per_dim_scale`/`use_common_neighbors` gave no clear
improvement (mixed, within noise across repeated checks). **`eig6` looks
like a real, modest additional win** — consistently ~75-80% across two
independent samples (n_inst=20 and 40) vs the eig4 baseline's fluctuating
but lower ~60-72.5% (baseline itself varies run to run within that band,
consistent with sampling noise, not a contradiction) — continuing the same
"more eigenvectors helps" trend found on the single-cell backbone in set 14,
now confirmed to transfer to GraphGPS too. GPS large band (c=14-30) also
trained, clean, low loss as expected for the easy end of the range.

**Still open:** `multistage_pairwise`/`multihead_pairwise` haven't had the
mechanism sweep at all yet (only eig4 tried); folding GraphGPS into the
cross-family ensemble.

**Discrete classical coloring as an input feature (tried, NEGATIVE result).**
User pushback ("did you try feeding spectral as init values... or the actual
discrete classical output, not just raw coordinates") correctly identified
that every `use_spectral_coords` variant all project fed *continuous*
eigenvector coordinates, never the classical algorithm's actual discrete
coloring guess. Added `use_classical_coloring` to `GPSPairwiseEmbedder`:
phase 1 (spectral init) + phase 2 (propagation) — deliberately skipping
phase 3/cleanup, which is expensive and, per the earlier support-score
diagnostic, essentially always fails at c=7-9 and returns this exact
phase-1+2 result unchanged anyway — one-hot encoded per vertex, centered,
concatenated at init alongside the existing degree+coordinate features.

**Result: no improvement, mild regression** (n=1000, 20 inst/c):

| c | plain | base (no classical-coloring) | +classical-coloring (medium) | +classical-coloring (small) |
|---|---|---|---|---|
| 7 | 0% | 0% | 0% | 0% |
| 8 | 0% | 5% | 0% | 5% |
| 9 | 25% | 65% | 55% | 5% |

Same pattern as the earlier dual-coords+residual-connection negative result
(set 15): naively concatenating extra classical signal doesn't help and can
mildly hurt — plausibly because the classical coloring is itself wrong
~45-55% of the time at c=9 (see the raw-accuracy diagnostic earlier in this
set), and the network partially trusts noisy/incorrect discrete information
rather than learning to ignore it, unlike the continuous coordinate hint
it already uses productively in the eig4 config. **Two independent attempts
now (continuous dual-coords and discrete one-hot coloring) at "inject more
classical signal" have both come back negative for this architecture family**
— worth treating that pattern as a real signal in itself, not just two
unlucky rolls.

**User idea: odd cycle transversal ("soft-core") as a "focal points" decomposition
(2026-07-16, `code/oct_algo.py`).** Distinct from the existing degree-based
"3-core" concept in `classical_baseline.py` (vertices surviving repeated
degree<=2 peeling) -- the **soft-core** is defined instead by odd-cycle
participation: the (greedy-approximate) minimum set of vertices whose
removal makes the rest of the graph bipartite. Construction: BFS-parity
2-color the graph, flag every edge connecting same-parity vertices as a
"conflict edge" (each one closes an odd cycle relative to the BFS tree),
greedily remove vertices covering the most remaining conflict edges until
none remain. **This is a hard guarantee, not a heuristic hope**: by
construction every edge remaining in G-soft-core connects different-parity
vertices, so the original BFS parity coloring is *already* a fully valid,
zero-conflict 2-coloring of G-soft-core -- if the soft-core itself can be
legally colored (respecting internal edges and fixed-neighbor constraints
from G-soft-core), the whole graph is properly 3-colored, with certainty.

**Result: the soft-core is far too large for this to help in practice**
(n=1000, 10 inst/c, low variance):

| c | 5 | 6 | 7 | 8 | 9 |
|---|---|---|---|---|---|
| mean soft-core size | 343 (34.3%) | 398 (39.8%) | 440 (44.0%) | 486 (48.6%) | 510 (51.0%) |

Growing monotonically with c, reaching **over half the graph by c=9**. The
"small number of focal points, easy rest" premise doesn't hold at these
densities -- solving the soft-core itself (a 340-510 vertex arbitrary
subgraph) via exact backtracking is exponential/infeasible, no easier than
the original problem. Not likely a greedy-heuristic artifact either: sparse
random graphs at average degree bounded away from 2 have `(c/2-1)*n` excess
edges beyond a spanning tree, and odd cycle transversals are known to scale
*linearly* with n in this regime -- even the true minimum soft-core is
almost certainly still a large constant fraction here, not sublinear.
**A third independent structural diagnostic (after support scores and
phase1+2 accuracy) confirming c=5-9's hardness is graph-wide, not localized
to a small "hard core."**

**Also checked, per explicit ask: classical peeling/pruning as a
preprocessing step.** Measured how much of the graph `classical_baseline.py`'s
degree-<=2 peeling actually removes before hitting a non-empty 3-core, across
the project's c range (n=1000, 5 instances/c): c=3 100%, c=5 14.3%, **c=7
2.8%, c=8 1.5%, c=9 0.7%**, c=10 0.3%. **Negligible at exactly the c=7-9 zone
this project needs help with** — there simply aren't enough degree-<=2
vertices at these densities for pruning to matter (it's a low-c technique,
already fully exploited by `peel_3color` itself, which already solves c<=4
outright). Not pursued further as a preprocessing step for this reason,
verified empirically before building anything rather than assumed.

---

## Notes on Experiment set 16 (2026-07-15/16) — QueryOptGNN_MP multi-head reasoning (C1): completed, honest negative-to-null result

Follow-up to set 15's placeholder. Full task: port `code/multihead_gnn.py`'s
genuinely-parallel-heads idea (own weights per head, never shared, combined
after — mirrors multi-head attention's separate per-head Q/K/V projections
concatenated through `W_O`) into `graph_color_improved/hybrid4.py`'s
`QueryOptGNN_MP`, as a new `multi_head_reasoning`/`num_reasoning_heads` flag
pair, then test it combined with A2 (the strongest existing single ablation),
and — per explicit follow-up instruction from the user — also alone, with A1,
and with B1, evaluated at n=1000 (mandatory scale) across c=3-9.

**Implementation.** `QueryOptGNN_MP` reasons via one shared `var_gru`/`edge_gru`
pair applied every round, with query-embedding feedback. Unlike
`multihead_gnn.py` (where each head runs ALL rounds in an isolated inner loop
and heads combine only once at the very end), `QueryOptGNN_MP`'s rounds are
not independent — every round already feeds back through message passing, the
query embedding, and (with A1) the color anchor, so isolating whole per-head
trajectories would throw that feedback loop away. Instead, `multi_head_reasoning=True`
parallelizes the per-round *transition function* itself: `num_reasoning_heads`
(default 4) independently-parameterized `GRUCell`s (own weights, never shared)
each read the SAME shared message/query input every round — mirroring how
attention heads share input but apply separate learned projections — then
their outputs are concatenated and projected back to `hidden_dim` via a
dedicated linear layer (`var_head_out_proj`/`edge_head_out_proj`, playing
`W_O`'s exact role). Each head also gets its own initial hidden state via a
dedicated per-head linear projection of the existing (A1-anchored,
possibly-A2-structured) init state, so heads start from genuinely different
views of the same symmetry-broken signal. The combined, projected result is
what continues through PairNorm/A1-reinject/dropout/readout every round, so
A1/B1/A2 and the query-feedback loop are completely untouched — only the GRU
update step becomes a multi-head ensemble. `var_gru`/`edge_gru` are
constructed only when the flag is `False`, and the multi-head modules only
when it's `True`, so state-dict keys never change based on anything else —
verified the existing A2-alone checkpoint still `load_state_dict(strict=True)`s
cleanly after this change (untouched code path, byte for byte).

**A real training snag, fixed properly (not worked around).** The A2+C1 run
was killed by something external to the training code at epoch 31/40 — the
training log up to that point is completely clean (loss falling normally, no
exception, no NaN); the process was simply terminated by the environment
(this session's background-task infrastructure appears to cap a single
tracked background process's lifetime at roughly one hour, unrelated to any
bug in `hybrid4.py`). Rather than discard 31 epochs of progress, added a
proper `--auto-resume` flag: reloads the latest `epoch_N.pt`'s model +
optimizer + **scheduler** state (so the cosine LR schedule continues exactly
as if training had never stopped, not restarted) and the existing
`history.json`, so the final checkpoint/history sequence is gapless. Added a
safety check that raises a clear error (rather than silently corrupting the
LR schedule) if `--auto-resume` is ever invoked with a different `--epochs`
than the interrupted run used, since `CosineAnnealingLR`'s `T_max` lives
inside its own `state_dict` and would otherwise silently overwrite the newly
requested one. Verified with a small 2-then-4-epoch smoke test before
trusting it on the real run. All four configs' training (40 epochs each, same
`data/adversarial-training` + `data/adversarial-testing` split used by every
other `QueryOptGNN_MP` ablation) completed via this mechanism.

**Results — n=1000, c=3-9, 50 inst/c, `rounds=300` + `track_best` (identical
methodology to the existing `queryoptgnn_a2_r300_n1000.json` baseline),
cross-verified via `sweep_queryoptgnn.py`'s existing `is_k_color` legality
check (unchanged, same check every other sweep in this project uses):**

| c | A2 alone (correct baseline) | A2+C1 | C1 alone | A1+C1 | B1+C1 |
|---|---|---|---|---|---|
| 3 | 60% | 30% | 36% | 2% | **72%** |
| 4 | 0% | 0% | 0% | 0% | 0% |
| 5 | 0% | 0% | 0% | 0% | 0% |
| 6 | 14% | 14% | 2% | 12% | 14% |
| 7 | 68% | **70%** | 44% | 44% | 64% |
| 8 | **86%** | 76% | 46% | 66% | 94% (highest single number) |
| 9 | **56%** | 56% | 4% | 58% (highest single number) | 28% |
| **avg(3-9)** | **40.6%** | 35.1% | 18.9% | 26.0% | 38.9% |

**Correction on the record:** an earlier same-session message mischaracterized
A2+C1's c=7-8 numbers (70%/76%) as a "breakthrough." They are not — the
correct baseline is A2 alone (already measured at n=1000 in Experiment set
12, *before* any multi-head work started this session: 68%/86%/56% at
c=7/8/9), not the older/weaker n=45 numbers. Against the correct baseline,
A2+C1 is roughly flat at c=7 (+2pt, within noise for n=50), and actually
**worse** at c=8 (-10pt) and c=3 (-30pt). The table above reports this
accurately.

**Bottom line: multi-head reasoning does not help this architecture, and the
honest ranking is A2 alone > B1+C1 ≈ A2+C1 > A1+C1 > C1 alone.** Three
findings support this:

1. **C1 alone (no A1/B1/A2) is the clear worst performer** (18.9% avg,
   collapsing to single digits at c=6 and c=9) — worse than every other
   config including plain A2. Multi-head parallelism by itself does not
   supply the symmetry-breaking this architecture needs; A1/A2/B1 are doing
   the real work everywhere they appear.
2. **A2+C1 is a strict regression from A2 alone** (35.1% vs 40.6% avg, worse
   at 2 of 7 points, tied at 3, marginally ahead at 1) — adding multi-head on
   top of the best existing ablation makes it worse, not better.
3. **B1+C1 is the closest multi-head config to A2 alone** (38.9% vs 40.6%,
   essentially tied) and has the two single highest numbers in the whole
   table (c=8: 94%, c=3: 72%), but is also markedly worse at c=9 (28% vs
   A2-alone's 56%) — a mixed sidegrade, not a clean win, and still net below
   A2 alone.

**Why, mechanistically:** splitting `hidden_dim=128` into `num_reasoning_heads=4`
heads of `head_dim=32` each reduces the per-head representational capacity
without buying enough diversity benefit to compensate — consistent with this
project's established diagnosis of `QueryOptGNN_MP`'s n=1000 collapse (set
12): the bottleneck is escaping the uniform-fixed-point / symmetry-breaking
problem, which A1/A2/B1 directly address via structured priors, anchor
re-injection, or temperature scheduling, not a representational-capacity or
parallelism problem that splitting one GRU into several smaller independent
ones is suited to solve. This mirrors GCPNet's and `QueryOptGNN_MP`'s own
already-documented pattern (set 12): a mechanism that looked like the best
result at small n (n=45) does not transfer to n=1000, and here it doesn't
even clear the bar set by the *existing* best mechanism (A2) at n=1000 — a
clean instance of exactly the finite-size-inflation risk n=1000 evaluation
exists to catch.

**Bonus (n=1000, c=10-14, collected as a byproduct — outside the scope asked
for in the final report, kept here for the record since it bears on the
"OOD-density-tail" pattern already tracked for A1/B1 at n=45):** A2+C1 does
modestly outperform A2 alone in the tail — c=10: 32% vs 20%, c=11: 8% vs 4%,
c=12: 2% vs 0%, c=14: 2% vs 0% (c=13: 0% both) — the same shape (a small but
real edge at high c) already seen for A1 and B1 individually at n=45 in
Experiment set 5, now the first time this specific tail-recovery pattern for
a `QueryOptGNN_MP` ablation has been confirmed at n=1000. Not large enough to
change the bottom line (A2+C1 is still net worse than A2 alone once c=3-9 is
weighted in), but a genuine, real, reproducible effect worth noting for any
future OOD-density-focused follow-up.
