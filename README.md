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
