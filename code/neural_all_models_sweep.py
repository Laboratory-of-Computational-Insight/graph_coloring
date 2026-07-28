"""
Per-model accuracy sweep across ALL neural architecture families we have,
one canonical (medium-band, best-config) checkpoint per family, at most 2
restarts each, with greedy_repair applied to every attempt (standing rule).

Families (7): spectral_nn, pairwise_nn, multistage_nn, multihead_gnn,
multistage_pairwise_nn, multihead_pairwise_nn, gps_pairwise_nn.

gps_pairwise_nn is NOT part of the production ensemble (locked decision from
earlier in the project -- never actually contributes to reported ensemble
numbers) but is included here since this is a standalone per-model
comparison, not an ensemble report.

Restart semantics: these models' `_3color` entry points have no seed
argument, but their `_cleanup` fallback path (spectral_coloring._cleanup ->
classical_baseline.dsatur_3color) uses an unseeded `np.random.default_rng()`,
so calling the same model twice on the same instance genuinely can differ.
"A restart" = calling the model again from scratch.
"""
import argparse
import json
import time

import torch

from random_planted import create_planted_3col
from gc_utils import is_k_color
from repair_utils import count_conflicts, greedy_repair

from spectral_nn import SpectralEmbedder, spectral_nn_3color
from pairwise_nn import PairwiseEmbedder, pairwise_nn_3color
from multistage_nn import MultiStageEmbedder, multistage_nn_3color
from multihead_gnn import MultiHeadGNNEmbedder, multihead_gnn_3color
from multistage_pairwise_nn import MultiStagePairwiseEmbedder, multistage_pairwise_nn_3color
from multihead_pairwise_nn import MultiHeadPairwiseEmbedder, multihead_pairwise_nn_3color
try:
    from gps_pairwise_nn import GPSPairwiseEmbedder, gps_pairwise_nn_3color
    _GPS_AVAILABLE = True
except ModuleNotFoundError:
    _GPS_AVAILABLE = False

RESULTS_DIR = "G:/graph_col/graph_coloring/results"
MAX_RESTARTS = 2

MANIFEST = {
    "spectral_nn": ("spectral_nn_medium_n300_r64_fisher_reinject_speccoords.pt",
        SpectralEmbedder, spectral_nn_3color,
        dict(hidden_dim=32, embed_dim=8, rounds=64, reinject_degree_signal=True, use_spectral_coords=True)),
    "pairwise_nn": ("pairwise_nn_medium_ablation_eig4_only.pt", PairwiseEmbedder, pairwise_nn_3color,
        dict(hidden_dim=32, embed_dim=8, rounds=32, reinject_degree_signal=True,
             use_spectral_coords=True, num_spectral_eigvecs=4, head="distance")),
    "multistage_nn": ("multistage_nn_medium_n300_s3_i6_o4_pos16_fisher_reinject.pt",
        MultiStageEmbedder, multistage_nn_3color,
        dict(hidden_dim=32, embed_dim=8, num_stages=3, inner_rounds=6, outer_iters=4,
             pos_enc_dim=16, reinject_degree_signal=True)),
    "multihead_gnn": ("multihead_gnn_medium_n300_h16_nh4_r64_fisher_reinject.pt",
        MultiHeadGNNEmbedder, multihead_gnn_3color,
        dict(head_dim=16, embed_dim=8, num_heads=4, rounds=64, reinject_degree_signal=True)),
    "multistage_pairwise_nn": ("multistage_pairwise_nn_medium_n300_eig4.pt",
        MultiStagePairwiseEmbedder, multistage_pairwise_nn_3color,
        dict(hidden_dim=32, embed_dim=8, num_stages=3, inner_rounds=6, outer_iters=4, pos_enc_dim=16,
             reinject_degree_signal=True, use_spectral_coords=True, num_spectral_eigvecs=4, head="distance")),
    "multihead_pairwise_nn": ("multihead_pairwise_nn_medium_n300_eig4.pt",
        MultiHeadPairwiseEmbedder, multihead_pairwise_nn_3color,
        dict(head_dim=16, embed_dim=8, num_heads=4, rounds=32, reinject_degree_signal=True,
             use_spectral_coords=True, num_spectral_eigvecs=4, head="distance")),
}
if _GPS_AVAILABLE:
    MANIFEST["gps_pairwise_nn"] = ("gps_pairwise_nn_medium_n300_eig4.pt", GPSPairwiseEmbedder, gps_pairwise_nn_3color,
        dict(hidden_dim=32, embed_dim=8, num_layers=4, heads=4, use_spectral_coords=True,
             num_spectral_eigvecs=4, head="distance"))


def load_all(names):
    models = {}
    for name in names:
        fname, cls, run_fn, kwargs = MANIFEST[name]
        m = cls(**kwargs)
        m.load_state_dict(torch.load(f"{RESULTS_DIR}/{fname}", map_location="cpu"))
        m.eval()
        models[name] = (m, run_fn)
    return models


def try_model(run_fn, m, adj, A):
    """One restart: run model, repair if needed, return (success, colors)."""
    r = run_fn(adj, m, device="cpu")
    colors = r["colors"]
    conf = count_conflicts(A, colors)
    if conf > 0:
        colors = greedy_repair(A, colors)
        conf = count_conflicts(A, colors)
    return conf == 0, colors


def parse_args():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--n", type=int, default=1000)
    p.add_argument("--c-values", type=float, nargs="+",
                    default=[5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 4.69])
    p.add_argument("--n-inst", type=int, default=20)
    p.add_argument("--models", nargs="+", default=list(MANIFEST.keys()))
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--out", default=f"{RESULTS_DIR}/neural_all_models_sweep.json")
    return p.parse_args()


def main():
    args = parse_args()
    torch.manual_seed(args.seed)
    models = load_all(args.models)
    print(f"loaded {len(models)} model families: {list(models.keys())}", flush=True)

    all_results = {}
    header = "c".rjust(6) + "".join(n.rjust(max(20, len(n) + 2)) for n in models) + "ANY".rjust(8)
    print(header, flush=True)

    t0 = time.time()
    for c in args.c_values:
        c_disp = int(c) if c == int(c) else c
        per_model_counts = {name: 0 for name in models}
        per_model_restarts = {name: [] for name in models}
        any_count = 0
        per_inst = {}

        for i in range(args.n_inst):
            _, adj = create_planted_3col(args.n, c)
            A = adj.numpy()
            solved_by = []
            inst_record = {}

            for name, (m, run_fn) in models.items():
                success = False
                used = 0
                for attempt in range(1, MAX_RESTARTS + 1):
                    ok, colors = try_model(run_fn, m, adj, A)
                    used = attempt
                    if ok:
                        success = True
                        break
                if success:
                    ok2, nconf, _ = is_k_color(adj.clone(), colors.copy())
                    assert ok2, f"{name} illegal at c={c}, nconf={nconf}"
                    per_model_counts[name] += 1
                    per_model_restarts[name].append(used)
                    solved_by.append(name)
                inst_record[name] = {"success": success, "restarts": used}

            if solved_by:
                any_count += 1
            per_inst[i] = inst_record

        row = f"{c_disp}".rjust(6)
        for name in models:
            row += f"{per_model_counts[name]}/{args.n_inst}".rjust(max(20, len(name) + 2))
        row += f"{any_count}/{args.n_inst}".rjust(8)
        print(row, flush=True)

        all_results[str(c_disp)] = {
            "per_instance": per_inst,
            "summary": {name: f"{per_model_counts[name]}/{args.n_inst}" for name in models},
            "any": f"{any_count}/{args.n_inst}",
        }

        with open(args.out, "w") as f:
            json.dump(all_results, f, indent=2)

    print(f"\n({time.time()-t0:.0f}s elapsed) -- saved to {args.out}", flush=True)


if __name__ == "__main__":
    main()
