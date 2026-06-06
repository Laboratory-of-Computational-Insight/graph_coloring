# Insights

Accumulated findings from experiments and code analysis. Add entries as they emerge.

---

## Model Behavior

**Out-of-distribution failure at large n**
GNN-GCP was trained on n ∈ [40, 60]. The sweep over n=2000 (c=1..30, 500 inst/c) found
0/15,000 legal colorings — flat zero, no signal. The Lemos paper already warned that n=1000
causes drastic accuracy drop; n=2000 confirms complete generalization failure beyond the training
scale. The model has essentially memorized the size of its training graphs.

**Scale compensation in `model.modified()`**
The `modified()` forward path scales the initial vertex embedding by `1/((n-1)//50 + 1)`:
for n=2000 this is ÷40. This is a heuristic to keep activation magnitudes in range for larger
graphs, but it does not recover generalization — the LSTM dynamics are entirely out of their
trained regime.

**Final-embedding vs. per-iteration checking**
The sweep checks k-means only on the embedding at the last iteration (tmax=32). The paper's
methodology checks at every iteration up to 150 and stops at the first valid coloring. At n=45
this per-iteration approach finds valid colorings ~5% of the time; checking only the final
embedding likely underestimates this. At n=2000 both methods appear to give 0%, so the
distinction doesn't matter for the large-n regime, but it matters for within-distribution
evaluation.

**Two forward modes**
- `modified()` (default): single global color node, fixed initialization from TF checkpoint,
  vertex init scaled by 1/((n-1)//50+1).
- `converted()` (`c_rand=True`): k random per-color embeddings, closer to original Lemos paper.
  The XAI analysis was done with `modified()`.

---

## Key Algorithmic Concepts (from AAAI 2025 Paper)

**Support — s(v)**
For a vertex v with color assignment C(v) in a (not necessarily legal) 3-coloring:
`s(v) = min over the other two color classes of the number of v's neighbors in that class`.
Originally from Alon & Kahale (1994). High support means: flipping v's color would create
many new conflicts → v is likely correctly placed. The GNN rediscovers this without being
explicitly taught it.

**Confidence — conf(v)**
`conf(v) = (1/|N(v)|) Σ_{u∈N(v)} ||x_v − x_u||_2`
i.e., average Euclidean distance from v's embedding to its graph neighbors' embeddings.
Increases monotonically over message-passing iterations. Spearman(iteration, mean conf) ≈ 86%
for successful colorings, 76% for failures.

**Support–Confidence correlation**
Spearman(s(v), conf(v)) ≈ 84% (successes) and 93% (failures), averaged over instances and
iterations. The GNN uses support as a proxy for where to push vertices in embedding space:
high-support → high-confidence → pushed toward a triangle vertex.

**Triangular geometry (successful colorings only)**
The 2D PCA projection of vertex embeddings forms an equilateral triangle in successful
executions; each color class occupies one leg. This geometry is absent in failed colorings.
It mirrors the k-simplex solution of the max-k-cut SDP (Karger–Motwani–Sudan 1998), which is
the best-known polynomial-time algorithm for approximate k-coloring.

**Two-phase operation**
1. High-support vertices anchor the triangle vertices early.
2. Remaining low-support vertices fill in around them.
Both phases are driven by the model iteratively increasing each vertex's distance from its
graph neighbors.

**SDP connection**
GNN-GCP was trained only on binary colorability decisions (cross-entropy loss), yet its
embedding geometry resembles an SDP solver. Prior work (Yau et al. 2023) showed that GNNs can
emulate SDPs when *explicitly* designed for it; GNN-GCP does so spontaneously.

---

## Data / Distribution Notes

**Phase transition for random 3-coloring**
For G(n, c/n) random graphs, the 3-colorability phase transition is at c ≈ 4.69. Below this
the graph is almost surely 3-colorable; above it almost surely not.

**Planted graphs vs. random graphs**
`create_planted_3col(n, c)` constructs a graph that is *guaranteed* 3-colorable by construction
(balanced partition, cross-class edges only). At c < 4.69 a random G(n, c/n) is also almost
surely 3-colorable, so both distributions are valid below the threshold. Above the threshold,
planted graphs remain colorable while random ones do not.

**Edge probability in `create_planted_3col`**
p = 3c / (2n) ensures each vertex has expected degree c. Derivation: each vertex v in class Vi
has |Vj| + |Vk| ≈ 2n/3 potential cross-class neighbors, each connected with probability p, so
E[deg(v)] = (2n/3) · (3c/2n) = c.

---

## Codebase Quirks

- `model.modified()` uses one global color node (C_h sliced to [1,1,d]), not k per-color
  embeddings. Color broadcast: `(M_vc.shape[1]*C).repeat(n, 1)`. This is different from the
  Lemos paper formulation.
- TF weight keys need `.T` transposition when loading into PyTorch Linear layers (row-major vs.
  col-major kernel convention).
- `gc_utils.sklearn_k_means()` reorders cluster labels by PCA angle (clockwise from rightmost
  centroid) for consistent cluster numbering across instances.
- `for_plot/*.json` files are Plotly figure JSONs (output), not input graph data.
- `results_analyse.py` expects `results/results_data_n_{n}_{c_or_p}_{val}.json` naming —
  the output format produced by `attribute_tests_v2.py`.
