#!/usr/bin/env python3
"""
Sweep GNN-GCP on planted 3-colorable graphs.
For each c in 1..30, generates N_INST instances with n=2000,
runs the pre-trained model, checks success rate for finding a legal 3-coloring
via k-means on the final embedding, and plots success rate vs c.
"""
import json
import os
import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from tqdm import tqdm

from model import GCPNet
from random_planted import create_planted_3col
from gc_utils import is_k_color, sklearn_k_means

# ── Config ─────────────────────────────────────────────────────────────────
N        = 2000
K        = 3
N_INST   = 500
C_VALUES = list(range(1, 31))
TMAX     = 32
RESULTS_FILE = "../results/sweep_results_n2000.json"
PLOT_FILE    = "../results/sweep_n2000.png"

DEVICE = ("mps"  if torch.backends.mps.is_available()  else
          "cuda" if torch.cuda.is_available()           else "cpu")


# ── Weight loading ──────────────────────────────────────────────────────────
def load_pretrained(gcp: GCPNet, device: str) -> None:
    d = {k: v for k, v in np.load("./original.npz", allow_pickle=True).items()}

    gcp.v_normal.data = torch.tensor([
        -0.169884145, 0.0430190973, 0.091173254, -0.0339165181, 0.0643557236,
        0.145693019, 0.112589225, -0.0952830836, 0.0254595, -0.0693574101,
        -0.0650991499, 0.235404313, -0.31420821, -0.0290404037, -0.161913335,
        -0.09325625, 0.298154235, -0.169444725, -0.207124308, 0.0723744854,
        -0.0849481523, 0.0168008488, 0.00895659439, -0.0171319768, -0.127776787,
        -0.0971129909, -0.0536339432, 0.168108433, 0.177107826, 0.320735186,
        -0.0755678415, 0.139883056, -0.388966531, -0.0078522, -0.00130009966,
        0.143557593, 0.035293255, -0.12994355, 0.1157846, -0.121418417,
        -0.115577929, 0.0780592263, -0.194125444, 0.113405302, 0.244302094,
        -0.0874284953, -0.0544838, 0.0926826522, 0.0209452771, 0.0718942657,
        0.0228996184, 0.298201054, 0.0192331262, -0.0319460481, -0.17595163,
        -0.0833073, 0.0334902816, 0.14013885, -0.14659746, 0.181580797,
        -0.00996331591, -0.0195714869, 0.160506919, 0.0497409627,
    ]).to(device)

    for tf_pat, attr_pat in [
        ("graph-coloring/V_msg_C_MLP_layer_{i}/kernel:0", "mlpV.l{i}.weight"),
        ("graph-coloring/V_msg_C_MLP_layer_{i}/bias:0",   "mlpV.l{i}.bias"),
        ("graph-coloring/C_msg_V_MLP_layer_{i}/kernel:0", "mlpC.l{i}.weight"),
        ("graph-coloring/C_msg_V_MLP_layer_{i}/bias:0",   "mlpC.l{i}.bias"),
        ("V_vote_MLP_layer_{i}/kernel:0",                  "V_vote_mlp.l{i}.weight"),
        ("V_vote_MLP_layer_{i}/bias:0",                    "V_vote_mlp.l{i}.bias"),
    ]:
        for i in range(1, 5):
            key  = tf_pat.replace("{i}", str(i))
            path = attr_pat.replace("{i}", str(i))
            obj  = gcp
            for part in path.split("."): obj = getattr(obj, part)
            t = torch.Tensor(d[key]).to(device)
            obj.data = t.T if "kernel" in key else t

    def _load_lstm(lstm, prefix):
        lstm.fc.weight.data = torch.Tensor(d[f"{prefix}/kernel:0"]).to(device).T
        for gate, name in [
            ("ln_ih", "input"), ("ln_ho", "output"),
            ("ln_hf", "transform"), ("ln_hc", "forget"), ("ln_hcy", "state"),
        ]:
            getattr(lstm, gate).gamma.data = torch.Tensor(d[f"{prefix}/{name}/gamma:0"]).to(device)
            getattr(lstm, gate).beta.data  = torch.Tensor(d[f"{prefix}/{name}/beta:0"]).to(device)

    _load_lstm(gcp.LSTM_v, "graph-coloring/V_cell/layer_norm_basic_lstm_cell")
    _load_lstm(gcp.LSTM_c, "graph-coloring/C_cell/layer_norm_basic_lstm_cell")


# ── Per-instance evaluation ─────────────────────────────────────────────────
def try_color(gcp: GCPNet, adj: torch.Tensor, device: str) -> tuple[bool, float]:
    """Return (legal, mono_frac) where mono_frac = monochromatic edges / total edges."""
    n = adj.shape[0]
    M_vv = adj.to(device)
    M_vc = torch.ones(n, K, device=device)
    cn   = torch.tensor([[K]])

    with torch.no_grad():
        _, _, V, _ = gcp.forward(M_vv, M_vc, [n], cn=cn)

    emb    = V.detach().cpu().numpy()
    colors, _ = sklearn_k_means(emb, K)
    ok, _, _  = is_k_color(adj.cpu(), colors)

    colors_t  = torch.tensor(colors, dtype=torch.long)
    same      = (colors_t.unsqueeze(0) == colors_t.unsqueeze(1)).float()
    n_edges   = adj.sum().item() / 2
    mono_frac = (adj.cpu() * same).sum().item() / 2 / n_edges if n_edges > 0 else 0.0

    return ok, mono_frac


# ── Main sweep ──────────────────────────────────────────────────────────────
def run_sweep(gcp: GCPNet) -> dict:
    results = {}

    # resume from existing results (re-run any entry missing mono_rate_mean)
    if os.path.exists(RESULTS_FILE):
        with open(RESULTS_FILE) as f:
            results = json.load(f)
        done = sum(1 for v in results.values() if "mono_rate_mean" in v)
        print(f"Resuming — {done} c-values already done.")

    for c in C_VALUES:
        key = str(c)
        if key in results and "mono_rate_mean" in results[key]:
            continue

        successes  = 0
        mono_fracs = []
        for _ in tqdm(range(N_INST), desc=f"c={c:2d}", leave=False):
            _, adj = create_planted_3col(N, c)
            ok, mf = try_color(gcp, adj, DEVICE)
            if ok:
                successes += 1
            mono_fracs.append(mf)

        rate = successes / N_INST
        results[key] = {
            "successes":     successes,
            "n_inst":        N_INST,
            "rate":          rate,
            "mono_rate_mean": float(np.mean(mono_fracs)),
            "mono_rate_std":  float(np.std(mono_fracs)),
        }
        print(f"  c={c:2d}  success={rate:.3f}  mono={results[key]['mono_rate_mean']:.3f}"
              f"  ({successes}/{N_INST})")

        with open(RESULTS_FILE, "w") as f:
            json.dump(results, f, indent=2)

    return results


# ── Plotting ────────────────────────────────────────────────────────────────
def plot_results(results: dict) -> None:
    cs         = sorted(int(k) for k in results)
    rates      = [results[str(c)]["rate"]           for c in cs]
    mono_means = [results[str(c)]["mono_rate_mean"] for c in cs]
    mono_stds  = [results[str(c)]["mono_rate_std"]  for c in cs]

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(8, 8), sharex=True)

    # — top: success rate —
    ax1.plot(cs, rates, marker="o", linewidth=2, markersize=5, color="steelblue",
             label="Success rate")
    ax1.axvline(4.69, color="red", linestyle="--", linewidth=1.2, label="Phase transition c≈4.69")
    ax1.set_ylabel("Success rate (legal 3-coloring)", fontsize=12)
    ax1.set_ylim(-0.02, 1.02)
    ax1.legend(fontsize=10)
    ax1.grid(True, alpha=0.3)
    ax1.set_title(f"GNN-GCP on planted 3-colorable graphs  (n={N}, {N_INST} instances/c)",
                  fontsize=12)

    # — bottom: monochromatic edge fraction —
    mono_means_pct = [m * 100 for m in mono_means]
    mono_stds_pct  = [s * 100 for s in mono_stds]
    ax2.errorbar(cs, mono_means_pct, yerr=mono_stds_pct,
                 marker="o", linewidth=2, markersize=5, color="darkorange",
                 capsize=3, label="Avg mono-edge %")
    ax2.axhline(100 / 3, color="gray", linestyle="--", linewidth=1.5,
                label="Random baseline (33.3%)")
    ax2.axvline(4.69, color="red", linestyle="--", linewidth=1.2)
    ax2.set_xlabel("Average degree c", fontsize=12)
    ax2.set_ylabel("Monochromatic edges (%)", fontsize=12)
    ax2.legend(fontsize=10)
    ax2.grid(True, alpha=0.3)

    fig.tight_layout()
    fig.savefig(PLOT_FILE, dpi=150)
    print(f"Plot saved to {PLOT_FILE}")


# ── Entry point ─────────────────────────────────────────────────────────────
if __name__ == "__main__":
    print(f"Device: {DEVICE}  |  n={N}  |  {N_INST} instances per c  |  tmax={TMAX}")

    gcp = GCPNet(64, tmax=TMAX, device=DEVICE)
    gcp.to(DEVICE)
    gcp.eval()
    load_pretrained(gcp, DEVICE)
    print("Model loaded.\n")

    results = run_sweep(gcp)
    plot_results(results)
