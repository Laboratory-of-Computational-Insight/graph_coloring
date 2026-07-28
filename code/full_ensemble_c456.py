"""
TRUE combined ensemble accuracy for c=4,5,6, OR-combining every real
(non-CP-SAT) method/architecture tried on the shared 20-instance test set,
all with greedy_repair applied. Excludes GraphGPS (transformer) per explicit
user instruction (over-tuned for specific n, diagnostic-only going forward).
Excludes OR-Tools/CP-SAT entirely (diagnostic-only, never counted).

Sources combined:
  - spectral-NN family: pairwise_nn, multistage_pairwise_nn
  - BP (belief propagation + reinforcement)
  - classical: alon_kahale, dsatur (peel is 0% everywhere at c>=4, included for completeness)
  - QueryOptGNN_MP: A2, spectral_priors_alone, spectral_priors_bethe_hessian
  - QueryOptGNN_MP multi-restart (10 restarts each): spectral_priors_alone, bethe_hessian
"""
import json

RESULTS_DIR = "G:/graph_col/graph_coloring/results"


def load(name):
    with open(f"{RESULTS_DIR}/{name}") as f:
        return json.load(f)


spectral_nn = load("shared_test_c456_spectral_nn_repaired.json")
bp = load("shared_test_bp_repair_results.json")
alon_kahale = load("shared_test_alon_kahale_c4to6_repaired.json")
dsatur = load("shared_test_dsatur_c4to6_repaired.json")
peel = load("shared_test_peel_c4to6_repaired.json")
a2 = load("shared_test_a2_c4to6_repaired.json")
spectral_priors = load("shared_test_spectral_priors_c4to6_repaired.json")
bethe_hessian = load("shared_test_bethe_hessian_c4to6_repaired.json")
sp_multi = load("spectral_priors_multi_restart_repaired.json")
bh_multi = load("bethe_hessian_multi_restart_repaired.json")

print(f"{'c':>3} | {'pairwise':>8} {'multistg':>8} {'BP':>6} {'AK':>6} {'dsatur':>6} {'peel':>6} "
      f"{'A2':>6} {'sp_prio':>7} {'bethe':>6} {'sp_multi':>8} {'bh_multi':>8} | {'FULL OR':>7}")

for c in ["4", "5", "6"]:
    n_inst = 20
    counts = {k: 0 for k in ["pairwise", "multistage", "bp", "ak", "dsatur", "peel",
                              "a2", "sp", "bh", "sp_multi", "bh_multi"]}
    full_or = 0
    for i in range(n_inst):
        si = str(i)
        vals = {
            "pairwise": spectral_nn[c][si]["pairwise"],
            "multistage": spectral_nn[c][si]["multistage"],
            "bp": bp[c][si],
            "ak": alon_kahale[c][si],
            "dsatur": dsatur[c][si],
            "peel": peel[c][si],
            "a2": a2[c][si],
            "sp": spectral_priors[c][si],
            "bh": bethe_hessian[c][si],
            "sp_multi": any(sp_multi[c][si]),
            "bh_multi": any(bh_multi[c][si]),
        }
        for k, v in vals.items():
            key_map = {"pairwise": "pairwise", "multistage": "multistage", "bp": "bp",
                       "ak": "ak", "dsatur": "dsatur", "peel": "peel", "a2": "a2",
                       "sp": "sp", "bh": "bh", "sp_multi": "sp_multi", "bh_multi": "bh_multi"}
            counts[key_map[k]] += v
        full_or += any(vals.values())

    print(f"{c:>3} | {counts['pairwise']:>8} {counts['multistage']:>8} {counts['bp']:>6} "
          f"{counts['ak']:>6} {counts['dsatur']:>6} {counts['peel']:>6} {counts['a2']:>6} "
          f"{counts['sp']:>7} {counts['bh']:>6} {counts['sp_multi']:>8} {counts['bh_multi']:>8} "
          f"| {full_or:>4}/{n_inst} ({100.0*full_or/n_inst:.0f}%)")
