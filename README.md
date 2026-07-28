# From Black Box to Algorithmic Insight: Explainable AI in Graph Neural Networks for Graph Coloring

**Elad Shoham, Havana Rika, Dan Vilenchik** — AAAI 2025

This repository contains the code for the XAI analysis of **GNN-GCP** (Lemos et al. 2019), a
Graph Neural Network trained to decide 3-colorability. We show that the model spontaneously
learns algorithmic concepts from classical combinatorics — in particular the notion of *support*
(Alon & Kahale 1994) — and encodes them geometrically as a triangular embedding structure that
mirrors the SDP relaxation of k-coloring (Karger–Motwani–Sudan 1998).

---

## Repository Layout

```
code/               Python source files + pre-trained weights
  model.py          GCPNet — PyTorch port of Lemos et al. GNN-GCP
  lstm.py           LayerNormLSTM with 5 separate layer norms
  mlp.py            4-layer MLP for message functions and voting
  random_planted.py Graph generators: create_planted_3col(n, c)
  gc_utils.py       Coloring validation, k-means, support/confidence metrics
  sweep_planted.py  Density sweep: success rate vs. c for a fixed n
  attribute_tests_v2.py  Full XAI analysis pipeline (support, confidence, Spearman)
  results_analyse.py     Aggregates JSON result files from attribute analysis runs
  plot_for_paper.py      Renders paper figures from results/for_plot/*.json
  original.npz      Pre-trained TF weights (numpy arrays)
  utils/            S3 access and singleton helpers

results/            Output artefacts (gitignored except for_plot/)
  for_plot/         Plotly figure JSONs used in the AAAI paper
  sweep_results_n*.json  Sweep checkpoints
  sweep_n*.png      Success-rate plots

assets/             Reference PDFs (papers)
TASKS.md            Current + future work checklist
INSIGHTS.md         Accumulated research findings
EXPERIMENTS.md      Append-only experiment log
```

---

## Latest results (2026-07-17) — GNN-based 3-coloring at scale (n=1000)

Extension of the original AAAI work: pushing a GNN-based (and classical/hybrid) pipeline to
reliably 3-color **planted, n=1000** instances (`create_planted_3col(n, c)`) across the full
average-degree range c=3..14. Success = legal 3-coloring found (`is_k_color`, zero monochromatic
edges), verified independently on every reported success. **Never report results at n<1000** —
small n inflates success via finite-size effects (this is the exact reason we rebuilt every early
n=45/100/300 result at real scale). Full blow-by-blow experiment log: `EXPERIMENTS.md`.

### Status per c-value (n=1000)

| c | status | method |
|---|---|---|
| 3 | **100%** | classical `peel_3color` (degree-≤2 peeling) — guaranteed-correct, no learned model needed |
| 4 | **100%** | full ensemble (classical alon_kahale/dsatur + spectral-NN + BP, all + greedy repair) |
| 5 | **95%** | BP (belief propagation + reinforcement) with 50 independent restarts, each repaired and OR-combined (early-exit on first success) — up from an initial 10% that turned out to be a restart-methodology artifact, not a real ceiling. One instance remains stuck even at 50/50 restarts (same "structurally stuck" pattern seen at c=8) |
| 6 | **100%** | BP (belief propagation + reinforcement) alone gets 85% (17/20) standalone — the single strongest individual method found for this c — combined with `QueryOptGNN_MP`'s multi-restart spectral-prior variants closes the rest |
| 7 | **100%** | `QueryOptGNN_MP` + spectral-algorithm node priors, OR-combined across 10 independent random restarts |
| 8 | **95%** (not 100%) | same — one specific instance fails *every* method/restart/architecture tried, appears structurally stuck rather than seed-sensitive |
| 9 | **100%** | cross-architecture ensemble (spectral-NN/pairwise/GraphGPS family OR `QueryOptGNN_MP` OR spectral-prior variant); a **single** GPS model + entropy penalty alone already reaches 92.5% |
| 10–14 | **~92–100%** | single `medium_pairwise`/`medium_gps_pairwise`/`medium_multistage_pairwise` checkpoints (no ensembling needed) |

### What actually moved the needle (roughly in order of impact)

1. **Pairwise same/different reformulation** (`code/pairwise_nn.py`) — predict whether two vertices
   share a color instead of a direct 3-way color, sidestepping the color-labeling symmetry that
   traps direct 3-way prediction at a uniform p=[1/3,1/3,1/3] fixed point. Single biggest lever
   for c=9-14.
2. **GraphGPS backbone** (`code/gps_pairwise_nn.py`) — local message passing + global Transformer
   attention (PyTorch Geometric `GPSConv`+`GINEConv`), combined with the pairwise head. Global
   attention lets far-apart vertices interact in one layer instead of needing many message-passing
   rounds — the project's best single result at c=9 (72.5%, later 92.5% with the entropy penalty).
3. **Spectral-algorithm node priors for `QueryOptGNN_MP`** (`graph_color_improved/hybrid4.py`) —
   seed the model's per-vertex color prior from the real Alon-Kahale spectral algorithm's output
   instead of a cheap degree-greedy heuristic. Dramatic win at c=6/7/9 (14%→32%, 68%→86%, 56%→90%).
4. **Independent random restarts** (not just tracking the best state across one run's rounds) —
   OR-combining 10 independently-seeded runs of the same trained model closed nearly all of the
   remaining c=6-8 gap for free, no new training.
5. **Direct entropy penalty** on the prediction confidence — a deterministic symmetry-breaking
   regularizer (penalize uncertainty directly) rather than the training-time noise-injection hack
   used earlier in the project; gave a large, verified, cross-architecture win specifically at c=9.
6. **Cross-architecture ensembling** (OR-combine already-trained checkpoints, zero new training) —
   `QueryOptGNN_MP` and the spectral-NN/pairwise/GraphGPS family are strong in different, largely
   non-overlapping places; combining them closes gaps neither family closes alone.
7. **Belief propagation + reinforcement** (`code/belief_propagation_coloring.py`) — a genuinely
   different algorithm family (statistical-physics cavity method + the standard Braunstein-Zecchina
   reinforcement trick, not a GNN or classical peel/DSATUR/spectral method). By far the single
   strongest individual method at c=6 (85% standalone, beating the entire previous ensemble), and
   the only method in the whole project to get any signal at all at c=5 — 95% with 50
   independent restarts, each repaired and OR-combined (the naive "best-of-N-beliefs" approach
   only found 10%; true independent-restart OR-combination is what unlocked the rest).
8. **Greedy conflict-repair post-processing** (`code/repair_utils.py`) — applied after every
   model/algorithm's output, always. Cheap, strictly monotonic (never increases conflicts), and a
   real, verified win at c=4 specifically (cracked what looked like a universal floor).

**What didn't work**, recorded for completeness (see `EXPERIMENTS.md` for full detail): feeding the
discrete classical coloring (vs. raw spectral coordinates) as an extra input feature; naive
dual-spectral-coordinate + residual-connection concatenation; the true non-backtracking operator
(correct but ~1000x more expensive than Bethe-Hessian for no accuracy gain in this regime); odd
cycle transversal ("soft-core") decomposition approaches — the cycle-causing nodes are easy to
color in isolation but that coloring is ~uncorrelated with the true global solution, so it doesn't
help reconstruct the rest; classical peeling/clique-finding preprocessing beyond c≈3.

---

## Setup

```bash
pip install -r requirements.txt
```

Heavy optional dependencies (only needed for SDP comparison):
```bash
pip install picos cvxpy gurobipy
```

---

## Quick Start

Run the model on a randomly generated planted 3-colorable graph:

```python
import sys; sys.path.insert(0, "code")
import torch, numpy as np
from model import GCPNet
from random_planted import create_planted_3col
from gc_utils import is_k_color, sklearn_k_means
from sweep_planted import load_pretrained

gcp = GCPNet(64, tmax=32, device="cpu")
load_pretrained(gcp, "cpu")
gcp.eval()

assignment, adj = create_planted_3col(n=45, c=3.5)
M_vc = torch.ones(45, 3)
with torch.no_grad():
    _, _, V, _ = gcp.forward(adj.float(), M_vc, [45], cn=torch.tensor([[3]]))

colors, _ = sklearn_k_means(V.numpy(), k=3)
ok, n_conflicts, _ = is_k_color(adj, colors)
print("Legal coloring found:", ok)
```

Or run directly from the `code/` directory:

```bash
cd code
python sweep_planted.py          # n=2000, c=1..30, 500 instances/c → ../results/sweep_n2000.png
```

Edit `N`, `C_VALUES`, `N_INST` at the top of `sweep_planted.py` to change parameters.
Results are checkpointed to `../results/sweep_results_n{N}.json` after each c-value.

---

## XAI Attribute Analysis

`code/attribute_tests_v2.py` reproduces the support/confidence/Spearman metrics from the paper.
It currently expects data from the remote Wasabi S3 bucket (see `code/utils/secrets.py`); a
refactor to accept in-memory graphs is on the task list (see `TASKS.md`).

---

## Remote Data

Large graph datasets and model checkpoints are stored in a Wasabi S3 bucket
(`kcol`, `eu-central-1`). Credentials are in `code/utils/secrets.py`. Access via `code/utils/fs.py`.

---

## Citation

```bibtex
@inproceedings{shoham2025blackbox,
  title     = {From Black Box to Algorithmic Insight: Explainable AI in Graph Neural Networks for Graph Coloring},
  author    = {Shoham, Elad and Rika, Havana and Vilenchik, Dan},
  booktitle = {Proceedings of the AAAI Conference on Artificial Intelligence},
  year      = {2025}
}
```
