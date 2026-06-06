import numpy as np
import os

import sys

import json

# FOLDER = "./"
FOLDER = "results/"

#list all files with the prefix 'results_data' in the directory using os module
all_results_files = [f for f in os.listdir(FOLDER) if f.startswith('results_data')]

all_results_data = {}

if len(all_results_files)==0:
    print("no files found")
    sys.exit()

keys = []
with open(os.path.join(FOLDER, all_results_files[0]), 'r') as f:
    data = json.load(f)
    for k, v in data.items():
        if len(v) > 0: keys.append(k)
    print("'" + "','".join(keys) + "'")

# keys = ['n','p','c','max_degree','mean_degree','mean_degree_normalized','k','colorable','closesed_index','attmept_range','closest_ass_dist','regression_slope','node_distance_ordering','node_distance_ordering_p','min_disagree','spearman_support_mean_confidence','spearman_degree_mean_confidence','changed_ratio_mean','unchanged_ratio_mean','confidence_pair_contrudiction','spearman_avg_confidence_over_time','spearman_p_avg_confidence_over_time','signs_distance_neighbor_strange','signs_distance_stranger_color','triangle_regression','pc_corolation','degree_pc_corolation','triangle_metric','triangle_apex_regression','embeddings','centroid_from_triangle','centroid_diff_from_line']

for file in all_results_files:
    name = ".".join(file.split('.')[:-1])
    name = name.split('_')[3:6]
    n = int(name[0])
    c_or_p = name[1]
    cp = float(name[2])
    name = (n, c_or_p, cp)
    if name in all_results_data and len(all_results_data[name]["c"]) > 0: continue
    with open(os.path.join(FOLDER, file), 'r') as f:
        data = json.load(f)
        # keys = []
        # for k, v in data.items():
        #     if len(v) > 0: keys.append(k)
        # print("'" + "','".join(keys) + "'")
    indexes = []
    for i in range(len(data["n"])):
        if data["n"][i] == n:
            indexes.append(i)
    relavent_data = {
        key: [data[key][j] for j in indexes] for key in keys if key in data
    }
    all_results_data[name] = relavent_data

avg_spearman_slope = []
avg_spearman_slope_p = []

condifence_support = []

attmept_range = {}

above=[]
below = []

for (n, cp_type, cp), value in all_results_data.items():
    if cp_type == "c" and float(cp) >= 5.0:
        continue

    for i in range(len(value["c"])):
        att = value["attmept_range"][i]
        if att not in attmept_range:
            attmept_range[att] = 0
        attmept_range[att] += 1

        if value["min_disagree"][i] > max(0.0 * value["n"][i], 1): # if fail then skip -> show success
        # if value["min_disagree"][i] <= max(0.0 * value["n"][i], 1):  # if success then skip -> show fail
            continue

        # if value["c"][i] > 2.5: # show only sparse
        # if value["c"][i] < 2.5 or value["c"][i] > 5:  # show only dense
        if value[cp_type][i] < 5:  # show only externe
            continue

        avg_spearman_slope.append(value["node_distance_ordering"][i])
        avg_spearman_slope_p.append(value["node_distance_ordering_p"][i])

        condifence_support.append(value["spearman_support_mean_confidence"][i])

        above.append(value["above_below"][i][0])
        below.append(value["above_below"][i][1])



above = np.array([a['(70, 70)'] for a in above])
above = above[above>=0]
below = np.array([a['(20, 20)'] for a in below])
below = below[below>=0]
print("low", below.mean(), below.std(), len(below))
print("high", above.mean(), above.std(), len(above))


condifence_support = np.array(condifence_support)
print("confidence_support ", condifence_support[condifence_support>=0].mean(), condifence_support[condifence_support>=0].std(), len(condifence_support), len(condifence_support[condifence_support<0]))


avg_spearman_slope = np.array(avg_spearman_slope)
avg_spearman_slope = avg_spearman_slope[avg_spearman_slope!=-1]
print("spearman disatnce increasing", avg_spearman_slope[avg_spearman_slope>=0].mean(), avg_spearman_slope[avg_spearman_slope>=0].std(), len(avg_spearman_slope), len(avg_spearman_slope[avg_spearman_slope<0]))

avg_spearman_slope_p = np.array(avg_spearman_slope_p)
avg_spearman_slope_p = avg_spearman_slope_p[avg_spearman_slope_p!=-1]
print("spearman disatnce increasing p", avg_spearman_slope_p[avg_spearman_slope_p>=0].mean(), avg_spearman_slope_p[avg_spearman_slope_p>=0].std(), len(avg_spearman_slope_p))

