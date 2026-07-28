#!/usr/bin/env python3
"""
Diagnostic (2026-07-15): is the pairwise same/different classifier's RAW
accuracy at c=5-9 actually informative (above the 2/3 "always different"
baseline), or is it also uninformative there -- matching the 0% end-to-end
coloring success? Separates "the network learned nothing" from "the network
learned something but the discrete decode pipeline throws it away."

Runs at n=1000 (matching real eval scale, not n=300 training scale) since
check_collapse() takes n as a parameter.
"""
import torch

from pairwise_nn import PairwiseEmbedder, check_collapse

RESULTS_DIR = "G:/graph_col/graph_coloring/results"

MANIFEST = {
    "small_pairwise":  ("pairwise_nn_small_n300_r32_bce_fisher2_speccoords.pt",
                     dict(hidden_dim=32, embed_dim=8, rounds=32, reinject_degree_signal=True,
                          use_spectral_coords=True, head="distance")),
    "medium_pairwise": ("pairwise_nn_medium_n300_r32_bce_fisher2_speccoords.pt",
                     dict(hidden_dim=32, embed_dim=8, rounds=32, reinject_degree_signal=True,
                          use_spectral_coords=True, head="distance")),
}

for name, (fname, kwargs) in MANIFEST.items():
    m = PairwiseEmbedder(**kwargs)
    m.load_state_dict(torch.load(f"{RESULTS_DIR}/{fname}"))
    m.eval()
    print(f"\n########## {name} ##########")
    check_collapse(m, n=1000, c_values=(5, 6, 7, 8, 9), n_pairs=8000, seed=42)
