import io

import numpy as np
import argparse
import os

import torch
from scipy.stats import spearmanr as sci_spearmanr
from torch.utils.data import DataLoader

from GeneratedGraphDataSet import GeneratedGraphs
from GraphDataSet import GraphDataSet
from gc_utils import is_k_color, sklearn_k_means, find_least_common_neighbor_color, calc_dist_over_iteartions, confidance_pair
from model import GCPNet
import pandas as pd

import plotly.io as pio
import plotly.express as px

from random_planted import create_gnp, create_planted
from utils.fs import FS
from utils.secrets import data_path
import json

pio.renderers.default = 'browser'

DEVICE = 'cpu'  # 'cuda'  #

def load_from_tf(args, gcp):
    gcp.v_normal.data = torch.tensor(
        [-0.169884145, 0.0430190973, 0.091173254, -0.0339165181, 0.0643557236, 0.145693019, 0.112589225, -0.0952830836,
         0.0254595, -0.0693574101, -0.0650991499, 0.235404313, -0.31420821, -0.0290404037, -0.161913335, -0.09325625,
         0.298154235, -0.169444725, -0.207124308, 0.0723744854, -0.0849481523, 0.0168008488, 0.00895659439,
         -0.0171319768, -0.127776787, -0.0971129909, -0.0536339432, 0.168108433, 0.177107826, 0.320735186,
         -0.0755678415, 0.139883056, -0.388966531, -0.0078522, -0.00130009966, 0.143557593, 0.035293255, -0.12994355,
         0.1157846, -0.121418417, -0.115577929, 0.0780592263, -0.194125444, 0.113405302, 0.244302094, -0.0874284953,
         -0.0544838, 0.0926826522, 0.0209452771, 0.0718942657, 0.0228996184, 0.298201054, 0.0192331262, -0.0319460481,
         -0.17595163, -0.0833073, 0.0334902816, 0.14013885, -0.14659746, 0.181580797, -0.00996331591, -0.0195714869,
         0.160506919, 0.0497409627]).to(args.device)  # torch.Tensor(loaded_dict["V_init:0"]).to(args.device)
    gcp.c_rand.data = torch.tensor([[0.569332063, 0.19621861, 0.936044037, 0.0672274604, 0.989149, 0.916594744, 0.754,
                                     0.431524485, 0.445979536, 0.333774686, 0.732518792, 0.822434127, 0.711422324,
                                     0.753830671, 0.836414278, 0.209573701, 0.527794242, 0.3339068, 0.832167804,
                                     0.6979146, 0.807687044, 0.690893054, 0.00416331459, 0.971259296, 0.615243,
                                     0.69255811, 0.669207, 0.670641, 0.85558778, 0.00144830858, 0.76548326, 0.409540862,
                                     0.888088107, 0.717633903, 0.584715724, 0.263450205, 0.459266245, 0.986697912,
                                     0.698782682, 0.63641417, 0.400523841, 0.221628249, 0.405968219, 0.579900086,
                                     0.725307345, 0.455515683, 0.131517351, 0.763612092, 0.928811967, 0.349458158,
                                     0.832664609, 0.914531469, 0.495537758, 0.163773, 0.827578843, 0.815654,
                                     0.429762304, 0.835437894, 0.323074102, 0.756760597, 0.627905488, 0.249528378,
                                     0.8888852, 0.242653042]]).to(args.device)

    loaded_dict = {item: value for item, value in np.load('./original.npz', allow_pickle=True).items()}
    for k in loaded_dict:
        print(k, loaded_dict[k].shape)
    gcp.mlpV.l1.weight.data = torch.Tensor(loaded_dict["graph-coloring/V_msg_C_MLP_layer_1/kernel:0"]).to(args.device).T
    gcp.mlpV.l1.bias.data = torch.Tensor(loaded_dict["graph-coloring/V_msg_C_MLP_layer_1/bias:0"]).to(args.device)
    gcp.mlpV.l2.weight.data = torch.Tensor(loaded_dict["graph-coloring/V_msg_C_MLP_layer_2/kernel:0"]).to(args.device).T
    gcp.mlpV.l2.bias.data = torch.Tensor(loaded_dict["graph-coloring/V_msg_C_MLP_layer_2/bias:0"]).to(args.device)
    gcp.mlpV.l3.weight.data = torch.Tensor(loaded_dict["graph-coloring/V_msg_C_MLP_layer_3/kernel:0"]).to(args.device).T
    gcp.mlpV.l3.bias.data = torch.Tensor(loaded_dict["graph-coloring/V_msg_C_MLP_layer_3/bias:0"]).to(args.device)
    gcp.mlpV.l4.weight.data = torch.Tensor(loaded_dict["graph-coloring/V_msg_C_MLP_layer_4/kernel:0"]).to(args.device).T
    gcp.mlpV.l4.bias.data = torch.Tensor(loaded_dict["graph-coloring/V_msg_C_MLP_layer_4/bias:0"]).to(args.device)
    gcp.mlpC.l1.weight.data = torch.Tensor(loaded_dict["graph-coloring/C_msg_V_MLP_layer_1/kernel:0"]).to(args.device).T
    gcp.mlpC.l1.bias.data = torch.Tensor(loaded_dict["graph-coloring/C_msg_V_MLP_layer_1/bias:0"]).to(args.device)
    gcp.mlpC.l2.weight.data = torch.Tensor(loaded_dict["graph-coloring/C_msg_V_MLP_layer_2/kernel:0"]).to(args.device).T
    gcp.mlpC.l2.bias.data = torch.Tensor(loaded_dict["graph-coloring/C_msg_V_MLP_layer_2/bias:0"]).to(args.device)
    gcp.mlpC.l3.weight.data = torch.Tensor(loaded_dict["graph-coloring/C_msg_V_MLP_layer_3/kernel:0"]).to(args.device).T
    gcp.mlpC.l3.bias.data = torch.Tensor(loaded_dict["graph-coloring/C_msg_V_MLP_layer_3/bias:0"]).to(args.device)
    gcp.mlpC.l4.weight.data = torch.Tensor(loaded_dict["graph-coloring/C_msg_V_MLP_layer_4/kernel:0"]).to(args.device).T
    gcp.mlpC.l4.bias.data = torch.Tensor(loaded_dict["graph-coloring/C_msg_V_MLP_layer_4/bias:0"]).to(args.device)
    gcp.V_vote_mlp.l1.weight.data = torch.Tensor(loaded_dict["V_vote_MLP_layer_1/kernel:0"]).to(args.device).T
    gcp.V_vote_mlp.l1.bias.data = torch.Tensor(loaded_dict["V_vote_MLP_layer_1/bias:0"]).to(args.device)
    gcp.V_vote_mlp.l2.weight.data = torch.Tensor(loaded_dict["V_vote_MLP_layer_2/kernel:0"]).to(args.device).T
    gcp.V_vote_mlp.l2.bias.data = torch.Tensor(loaded_dict["V_vote_MLP_layer_2/bias:0"]).to(args.device)
    gcp.V_vote_mlp.l3.weight.data = torch.Tensor(loaded_dict["V_vote_MLP_layer_3/kernel:0"]).to(args.device).T
    gcp.V_vote_mlp.l3.bias.data = torch.Tensor(loaded_dict["V_vote_MLP_layer_3/bias:0"]).to(args.device)
    gcp.V_vote_mlp.l4.weight.data = torch.Tensor(loaded_dict["V_vote_MLP_layer_4/kernel:0"]).to(args.device).T
    gcp.V_vote_mlp.l4.bias.data = torch.Tensor(loaded_dict["V_vote_MLP_layer_4/bias:0"]).to(args.device)
    gcp.LSTM_v.fc.weight.data = torch.Tensor(
        loaded_dict["graph-coloring/V_cell/layer_norm_basic_lstm_cell/kernel:0"]).to(args.device).T
    gcp.LSTM_v.ln_ih.gamma.data = torch.Tensor(
        loaded_dict["graph-coloring/V_cell/layer_norm_basic_lstm_cell/input/gamma:0"]).to(args.device)
    gcp.LSTM_v.ln_ih.beta.data = torch.Tensor(
        loaded_dict["graph-coloring/V_cell/layer_norm_basic_lstm_cell/input/beta:0"]).to(args.device)
    gcp.LSTM_v.ln_ho.gamma.data = torch.Tensor(
        loaded_dict["graph-coloring/V_cell/layer_norm_basic_lstm_cell/output/gamma:0"]).to(args.device)
    gcp.LSTM_v.ln_ho.beta.data = torch.Tensor(
        loaded_dict["graph-coloring/V_cell/layer_norm_basic_lstm_cell/output/beta:0"]).to(args.device)
    gcp.LSTM_v.ln_hf.gamma.data = torch.Tensor(
        loaded_dict["graph-coloring/V_cell/layer_norm_basic_lstm_cell/transform/gamma:0"]).to(args.device)
    gcp.LSTM_v.ln_hf.beta.data = torch.Tensor(
        loaded_dict["graph-coloring/V_cell/layer_norm_basic_lstm_cell/transform/beta:0"]).to(args.device)
    gcp.LSTM_v.ln_hc.gamma.data = torch.Tensor(
        loaded_dict["graph-coloring/V_cell/layer_norm_basic_lstm_cell/forget/gamma:0"]).to(args.device)
    gcp.LSTM_v.ln_hc.beta.data = torch.Tensor(
        loaded_dict["graph-coloring/V_cell/layer_norm_basic_lstm_cell/forget/beta:0"]).to(args.device)
    gcp.LSTM_v.ln_hcy.gamma.data = torch.Tensor(
        loaded_dict["graph-coloring/V_cell/layer_norm_basic_lstm_cell/state/gamma:0"]).to(args.device)
    gcp.LSTM_v.ln_hcy.beta.data = torch.Tensor(
        loaded_dict["graph-coloring/V_cell/layer_norm_basic_lstm_cell/state/beta:0"]).to(args.device)
    gcp.LSTM_c.fc.weight.data = torch.Tensor(
        loaded_dict["graph-coloring/C_cell/layer_norm_basic_lstm_cell/kernel:0"]).to(args.device).T
    gcp.LSTM_c.ln_ih.gamma.data = torch.Tensor(
        loaded_dict["graph-coloring/C_cell/layer_norm_basic_lstm_cell/input/gamma:0"]).to(args.device)
    gcp.LSTM_c.ln_ih.beta.data = torch.Tensor(
        loaded_dict["graph-coloring/C_cell/layer_norm_basic_lstm_cell/input/beta:0"]).to(args.device)
    gcp.LSTM_c.ln_ho.gamma.data = torch.Tensor(
        loaded_dict["graph-coloring/C_cell/layer_norm_basic_lstm_cell/output/gamma:0"]).to(args.device)
    gcp.LSTM_c.ln_ho.beta.data = torch.Tensor(
        loaded_dict["graph-coloring/C_cell/layer_norm_basic_lstm_cell/output/beta:0"]).to(args.device)
    gcp.LSTM_c.ln_hf.gamma.data = torch.Tensor(
        loaded_dict["graph-coloring/C_cell/layer_norm_basic_lstm_cell/transform/gamma:0"]).to(args.device)
    gcp.LSTM_c.ln_hf.beta.data = torch.Tensor(
        loaded_dict["graph-coloring/C_cell/layer_norm_basic_lstm_cell/transform/beta:0"]).to(args.device)
    gcp.LSTM_c.ln_hc.gamma.data = torch.Tensor(
        loaded_dict["graph-coloring/C_cell/layer_norm_basic_lstm_cell/forget/gamma:0"]).to(args.device)
    gcp.LSTM_c.ln_hc.beta.data = torch.Tensor(
        loaded_dict["graph-coloring/C_cell/layer_norm_basic_lstm_cell/forget/beta:0"]).to(args.device)
    gcp.LSTM_c.ln_hcy.gamma.data = torch.Tensor(
        loaded_dict["graph-coloring/C_cell/layer_norm_basic_lstm_cell/state/gamma:0"]).to(args.device)
    gcp.LSTM_c.ln_hcy.beta.data = torch.Tensor(
        loaded_dict["graph-coloring/C_cell/layer_norm_basic_lstm_cell/state/beta:0"]).to(args.device)


def attributes_check(results_data, M_vv, M_vc, split,labels, cn, n, k, gcp, adj):

    iters_to_run = 150
    attempts = 2
    old = gcp.tmax
    gcp.tmax = iters_to_run
    pred, means, V_ret, C_ret = gcp.forward(M_vv, M_vc, split, cn=cn, attempts=attempts)
    gcp.tmax = old
    min_disagree = []
    assignments = []
    votes = []
    missmatches_indexes = []
    centroids = []
    closesdt_history = []
    local_assignments = []

    adj_cpu = adj.cpu()

    embeddings_list = [h.clone().detach().cpu().numpy() for h in gcp.histories]

    for i in range(len(embeddings_list)):
        assignment, centroid = sklearn_k_means(embeddings_list[i], k)
        is_k_col, disagree, missmatch_indexes = is_k_color(adj, assignment)

        centroids.append(centroid)
        min_disagree.append(disagree)
        assignments.append(assignment)
        local_assignments.append(assignment)
        missmatches_indexes.append(missmatch_indexes)
        votes.append(-1)

    old_closesed_index = np.argmin(min_disagree)

    min_disagree = []
    assignments = []
    votes = []
    missmatches_indexes = []
    centroid = centroids[old_closesed_index]
    closesdt_history = []
    new_centroids = []
    for i in range(len(embeddings_list)):
        assignment, new_centroid = sklearn_k_means(embeddings_list[i], k, centroids = centroid)
        is_k_col, disagree, missmatch_indexes = is_k_color(adj, assignment)
        # close_ass_iter = find_closest_kcoloring(assignment, adj, k)
        # miss_match_count = sum(x != y for x, y in zip(assignment, close_ass_iter))
        # closesdt_history.append(miss_match_count)
        new_centroids.append(new_centroid)
        min_disagree.append(disagree)
        assignments.append(assignment)
        missmatches_indexes.append(missmatch_indexes)
        votes.append(-1)
    new_closesed_index = np.argmin(min_disagree) # np.argmin(closesdt_history) # #np.argmin(min_disagree)
    attmept_range = new_closesed_index//iters_to_run
    closesed_index = new_closesed_index % iters_to_run
    gcp.histories = gcp.histories[attmept_range*iters_to_run: (attmept_range+1)*iters_to_run ]
    embeddings_list = embeddings_list[attmept_range * iters_to_run: (attmept_range + 1) * iters_to_run]
    min_disagree = min_disagree[attmept_range*iters_to_run: (attmept_range+1)*iters_to_run ]
    assignments = assignments[attmept_range*iters_to_run: (attmept_range+1)*iters_to_run ]
    local_assignments = local_assignments[attmept_range*iters_to_run: (attmept_range+1)*iters_to_run ]
    votes = votes[attmept_range*iters_to_run: (attmept_range+1)*iters_to_run ]
    gcp.clauses_hist = gcp.clauses_hist[attmept_range*iters_to_run: (attmept_range+1)*iters_to_run ]
    centroids = centroids[attmept_range*iters_to_run: (attmept_range+1)*iters_to_run ]
    closesdt_history = closesdt_history[attmept_range*iters_to_run: (attmept_range+1)*iters_to_run ]
    new_centroids = new_centroids[attmept_range*iters_to_run: (attmept_range+1)*iters_to_run ]

    results_data["attmept_range"].append(attmept_range.item())

    results_data["min_disagree"].append( min_disagree[closesed_index])

    distance_neighbors_strangers_color(adj, closesed_index, embeddings_list, results_data)

    distances_over_time = calc_dist_over_iteartions(adj_cpu, embeddings_list)

    conflict_sums = []
    least_common_colors = []
    fake_closesd_index = min(len(embeddings_list)-1, (closesed_index if min_disagree[closesed_index]==0 else (150 if n>250 else 50)))
    for i in range(fake_closesd_index + 1 ):
        emb = embeddings_list[i]
        least_common_color, cs, neighbor_dist = find_least_common_neighbor_color(adj, local_assignments[i], emb, k)
        conflict_sums.append(cs)
        least_common_colors.append(least_common_color)

    above, below = confidance_pair(adj, local_assignments, distances_over_time, fake_closesd_index, conflict_sums)
    results_data["above_below"].append([above, below])

    get_spearman_support_mean_confidence(distances_over_time, adj_cpu, fake_closesd_index, results_data, conflict_sums)
    return


def distance_neighbors_strangers_color(adj, closesed_index, embeddings_list, results_data):
    def calculate_average_distances_per_iteration(embeddings_list, adj):
        avg_distances_per_iteration_neighbors = []
        for embeddings in embeddings_list:
            avg_distances_neihobrs = []
            for i in range(len(embeddings)):
                neighbors = torch.where(adj[i] > 0)[0]
                neihbors_distances = [0]
                if len(neighbors) != 0:
                    neihbors_distances = [np.linalg.norm(embeddings[i] - embeddings[neighbor]) for neighbor in
                                          neighbors]
                avg_distances_neihobrs.append(np.mean(neihbors_distances))

            avg_distances_per_iteration_neighbors.append(np.mean(avg_distances_neihobrs))
        return avg_distances_per_iteration_neighbors

    avg_distances_per_iteration_neighbors = calculate_average_distances_per_iteration(embeddings_list[:closesed_index], adj)
    df = pd.DataFrame({
        'Average Distance': avg_distances_per_iteration_neighbors[:closesed_index],
    })
    average_distances = df['Average Distance']

    statistic = -1
    pval = -1
    if len(average_distances) > 0:
        stats = sci_spearmanr(list(range(len(average_distances))), average_distances)
        statistic = stats.statistic
        pval = stats.pvalue
    results_data["node_distance_ordering"].append(statistic)
    results_data["node_distance_ordering_p"].append(pval)


def get_spearman_support_mean_confidence(distances_over_time, adj, closesed_index, results_data, supports):
    rows = []
    columns = ["support", "confidence", "degree", "iteration"]
    degrees = adj.sum(0).numpy()
    for i in range(closesed_index + 1):
        support = supports[i][0]
        for j in range(len(support)):
            rows.append([support[j], distances_over_time[i][j], degrees[j], i])
    df = pd.DataFrame(rows, columns=columns)
    fig = px.scatter(df, x='support', y='confidence', animation_frame="iteration")
    # fig.show()
    iter_sup_mean_conf = df.groupby(["iteration", "support"])["confidence"].mean()
    spearman_corrs = []
    for i in range(len(df["iteration"].unique())):
        spearman_corrs.append(sci_spearmanr(iter_sup_mean_conf[i].index, iter_sup_mean_conf[i].values))
    results_data["spearman_support_mean_confidence"].append(np.mean([sc.statistic for sc in spearman_corrs]).item())


def run(args, gcp, critirion, results_data, dl):
    preds = []
    plot_loss = 0

    for j, b in enumerate(dl):
        if j > 49: break
        M_vv, labels, cn, split, M_vc = b
        labels = labels.squeeze()
        split = split.squeeze(0)
        split = [int(s) for s in split]
        M_vv = M_vv.squeeze()
        M_vc = M_vc.squeeze()

        M_vv = M_vv.to(device=args.device)
        M_vc = M_vc.to(device=args.device)

        pred, means, V_ret, C_ret = gcp.forward(M_vv, M_vc, split, cn=cn)
        l = critirion(means.to(DEVICE), torch.Tensor(labels).to(device=DEVICE))

        preds += pred.tolist() if len(pred.shape) != 0 else [pred.item()]
        plot_loss += l.detach()

        if len(labels.size()) > 0:
            k = cn[0, 0].item() + (1 - int(labels[0].item()))
        else:
            k = cn[0, 0].item() + (1 - int(labels.item()))
        k = k
        n = split[0]
        adj = M_vv[:n, :n]

        features = gcp.history[:n].clone().detach().cpu()
        degrees = adj.sum(axis=0).cpu()
        n = len(degrees)

        p = M_vv[:n, :n].sum() / (n * (n - 1))
        c = n * p
        results_data["n"] += [n]
        results_data["p"] += [p.item()]
        results_data["c"] += [c.item()]
        results_data["k"] += [k]

        print(n, int(degrees.max().item()), round(degrees.mean().item(), 2), round(degrees.mean().item()/(n-1), 3))

        attributes_check(results_data, M_vv, M_vc, split, labels, cn, n, k, gcp, adj)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--graph_dir', default="/home/elad/Documents/kcol/GCP_project/data_json/data")
    parser.add_argument('--embedding_size', type=int, default=64)
    parser.add_argument('--batch_size', type=int, default=1)
    parser.add_argument('--tmax', type=int, default=32)
    parser.add_argument('--epochs', type=int, default=5300)
    parser.add_argument('--device', type=str, default='cuda')
    parser.add_argument('--check_path', type=str, default=None)
    parser.add_argument('--test', type=bool, default=True)
    args = parser.parse_args()

    ds = GraphDataSet(
        args.graph_dir,
        batch_size=args.batch_size,
        filter=lambda f: f.split("/")[-1][3] == "3",
    )

    embedding_size = args.embedding_size
    gcp = GCPNet(embedding_size, tmax=args.tmax, device=args.device)
    gcp.to(args.device)
    gcp.eval()
    load_from_tf(args, gcp)
    critirion = torch.nn.BCEWithLogitsLoss(reduction="mean")

    results_data_original = {
        "n":[],
        "p":[],
        "c":[],
        "k": [],
        "attmept_range": [],
        "min_disagree": [],
        "node_distance_ordering": [],
        "node_distance_ordering_p": [],
        "spearman_support_mean_confidence": [],
        "above_below": [],

    }


    if args.check_path is not None:
        print("loading:", args.check_path)
        checkpoint = torch.load(args.check_path, map_location="cpu")
        gcp.load_all_attributes(checkpoint['model'])
    gcp.to(args.device)


    ns = [45, 100,500,1000, 2000]
    k = 3
    cs = [0.5, 1, 2, 3, 3.5, 4, 4.5,5.0, 5.5]
    ps = [0.3, 0.5, 0.8]

    ds.shuffle()
    dl_files = DataLoader(ds, batch_size=1, shuffle=False)
    datasets = []

    for n in ns:
        results_data = results_data_original.copy()
        for c in cs: # Random
            if os.path.exists(f"results/results_data_n_{n}_c_{c}.json"):
                continue
            print("starting nc", n,c)
            if FS().file_exists(os.path.join(data_path, "experiments", "random", str(n), str(c), "data.pt")):
                random = FS().get_data(os.path.join(data_path, "experiments", "random", str(n), str(c), "data.pt"))
                random = torch.load(io.BytesIO(random))["data"]
                dl = DataLoader(GeneratedGraphs(random), batch_size=1, shuffle=False)
                datasets.append(dl)
                res = run(args, gcp, critirion, results_data, dl)
                with open(f'results_data_n_{n}_c_{c}.json', 'w') as json_file:
                    json.dump(results_data, json_file, indent=4)
            else:
                random = []
                for _ in range(1_000):
                    p = c/n
                    ass, v_mat, bad, be = create_gnp(n, k=k, c=c)
                    random.append(v_mat)
                buffer = io.BytesIO()
                torch.save({"data": random}, buffer)
                buffer.seek(0)
                FS().upload_data(buffer, os.path.join(data_path, "experiments", "random",str(n), str(c), "data.pt"))
        results_data = results_data_original.copy()
        for p in ps: # planted
            if os.path.exists(f"results/results_data_n_{n}_p_{p}.json"):
                continue
            print("starting np", n, p)
            if FS().file_exists(os.path.join(data_path, "experiments", "planted",str(n), str(p), "data.pt")):
                # pass
                planted = FS().get_data(os.path.join(data_path, "experiments", "planted", str(n), str(p), "data.pt"))
                planted = torch.load(io.BytesIO(planted))["data"]
                dl = DataLoader(GeneratedGraphs(planted), batch_size=1, shuffle=False)
                datasets.append(dl)
                res = run(args, gcp, critirion, results_data, dl)
                with open(f'results_data_n_{n}_p_{p}.json', 'w') as json_file:
                    json.dump(results_data, json_file, indent=4)
            else:
                planted = []
                for _ in range(1_000):
                    c = n * p
                    ass, v_mat, bad, be = create_planted(n, k=k, p=p)
                    planted.append(v_mat)
                buffer = io.BytesIO()
                torch.save({"data": planted}, buffer)
                buffer.seek(0)
                FS().upload_data(buffer, os.path.join(data_path, "experiments", "planted",str(n), str(p), "data.pt"))

    datasets.append(dl_files)
    results_data = results_data_original.copy()
    for dl in datasets:
        run(args, gcp, critirion, results_data, dl)
    print("done")


if __name__ == '__main__':
    main()