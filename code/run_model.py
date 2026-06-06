import os

import numpy as np
import argparse

import torch
from torch.utils.data import DataLoader
from GraphDataSet import GraphDataSet
from model import GCPNet
from utils.fs import FS
from three_kol_files import col3_files


DEVICE =  'cuda'  # 'cuda'  #


def load_from_tf(gcp):
    gcp.v_normal.data = torch.tensor([-0.169884145,0.0430190973,0.091173254,-0.0339165181,0.0643557236,0.145693019,0.112589225,-0.0952830836,0.0254595,-0.0693574101,-0.0650991499,0.235404313,-0.31420821,-0.0290404037,-0.161913335,-0.09325625,0.298154235,-0.169444725,-0.207124308,0.0723744854,-0.0849481523,0.0168008488,0.00895659439,-0.0171319768,-0.127776787,-0.0971129909,-0.0536339432,0.168108433,0.177107826,0.320735186,-0.0755678415,0.139883056,-0.388966531,-0.0078522,-0.00130009966,0.143557593,0.035293255,-0.12994355,0.1157846,-0.121418417,-0.115577929,0.0780592263,-0.194125444,0.113405302,0.244302094,-0.0874284953,-0.0544838,0.0926826522,0.0209452771,0.0718942657,0.0228996184,0.298201054,0.0192331262,-0.0319460481,-0.17595163,-0.0833073,0.0334902816,0.14013885,-0.14659746,0.181580797,-0.00996331591,-0.0195714869,0.160506919,0.0497409627]).to(args.device) #torch.Tensor(loaded_dict["V_init:0"]).to(args.device)
    gcp.c_rand.data = torch.tensor([[0.569332063,0.19621861,0.936044037,0.0672274604,0.989149,0.916594744,0.754,0.431524485,0.445979536,0.333774686,0.732518792,0.822434127,0.711422324,0.753830671,0.836414278,0.209573701,0.527794242,0.3339068,0.832167804,0.6979146,0.807687044,0.690893054,0.00416331459,0.971259296,0.615243,0.69255811,0.669207,0.670641,0.85558778,0.00144830858,0.76548326,0.409540862,0.888088107,0.717633903,0.584715724,0.263450205,0.459266245,0.986697912,0.698782682,0.63641417,0.400523841,0.221628249,0.405968219,0.579900086,0.725307345,0.455515683,0.131517351,0.763612092,0.928811967,0.349458158,0.832664609,0.914531469,0.495537758,0.163773,0.827578843,0.815654,0.429762304,0.835437894,0.323074102,0.756760597,0.627905488,0.249528378,0.8888852,0.242653042]]).to(args.device)

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
    gcp.LSTM_v.fc.weight.data = torch.Tensor(loaded_dict["graph-coloring/V_cell/layer_norm_basic_lstm_cell/kernel:0"]).to(args.device).T
    gcp.LSTM_v.ln_ih.gamma.data = torch.Tensor(loaded_dict["graph-coloring/V_cell/layer_norm_basic_lstm_cell/input/gamma:0"]).to(args.device)
    gcp.LSTM_v.ln_ih.beta.data = torch.Tensor(loaded_dict["graph-coloring/V_cell/layer_norm_basic_lstm_cell/input/beta:0"]).to(args.device)
    gcp.LSTM_v.ln_ho.gamma.data = torch.Tensor(loaded_dict["graph-coloring/V_cell/layer_norm_basic_lstm_cell/output/gamma:0"]).to(args.device)
    gcp.LSTM_v.ln_ho.beta.data = torch.Tensor(loaded_dict["graph-coloring/V_cell/layer_norm_basic_lstm_cell/output/beta:0"]).to(args.device)
    gcp.LSTM_v.ln_hf.gamma.data = torch.Tensor(loaded_dict["graph-coloring/V_cell/layer_norm_basic_lstm_cell/transform/gamma:0"]).to(args.device)
    gcp.LSTM_v.ln_hf.beta.data = torch.Tensor(loaded_dict["graph-coloring/V_cell/layer_norm_basic_lstm_cell/transform/beta:0"]).to(args.device)
    gcp.LSTM_v.ln_hc.gamma.data = torch.Tensor(loaded_dict["graph-coloring/V_cell/layer_norm_basic_lstm_cell/forget/gamma:0"]).to(args.device)
    gcp.LSTM_v.ln_hc.beta.data = torch.Tensor(loaded_dict["graph-coloring/V_cell/layer_norm_basic_lstm_cell/forget/beta:0"]).to(args.device)
    gcp.LSTM_v.ln_hcy.gamma.data = torch.Tensor(loaded_dict["graph-coloring/V_cell/layer_norm_basic_lstm_cell/state/gamma:0"]).to(args.device)
    gcp.LSTM_v.ln_hcy.beta.data = torch.Tensor(loaded_dict["graph-coloring/V_cell/layer_norm_basic_lstm_cell/state/beta:0"]).to(args.device)
    gcp.LSTM_c.fc.weight.data = torch.Tensor(loaded_dict["graph-coloring/C_cell/layer_norm_basic_lstm_cell/kernel:0"]).to(args.device).T
    gcp.LSTM_c.ln_ih.gamma.data = torch.Tensor(loaded_dict["graph-coloring/C_cell/layer_norm_basic_lstm_cell/input/gamma:0"]).to(args.device)
    gcp.LSTM_c.ln_ih.beta.data = torch.Tensor(loaded_dict["graph-coloring/C_cell/layer_norm_basic_lstm_cell/input/beta:0"]).to(args.device)
    gcp.LSTM_c.ln_ho.gamma.data = torch.Tensor(loaded_dict["graph-coloring/C_cell/layer_norm_basic_lstm_cell/output/gamma:0"]).to(args.device)
    gcp.LSTM_c.ln_ho.beta.data = torch.Tensor(loaded_dict["graph-coloring/C_cell/layer_norm_basic_lstm_cell/output/beta:0"]).to(args.device)
    gcp.LSTM_c.ln_hf.gamma.data = torch.Tensor(loaded_dict["graph-coloring/C_cell/layer_norm_basic_lstm_cell/transform/gamma:0"]).to(args.device)
    gcp.LSTM_c.ln_hf.beta.data = torch.Tensor(loaded_dict["graph-coloring/C_cell/layer_norm_basic_lstm_cell/transform/beta:0"]).to(args.device)
    gcp.LSTM_c.ln_hc.gamma.data = torch.Tensor(loaded_dict["graph-coloring/C_cell/layer_norm_basic_lstm_cell/forget/gamma:0"]).to(args.device)
    gcp.LSTM_c.ln_hc.beta.data = torch.Tensor(loaded_dict["graph-coloring/C_cell/layer_norm_basic_lstm_cell/forget/beta:0"]).to(args.device)
    gcp.LSTM_c.ln_hcy.gamma.data = torch.Tensor(loaded_dict["graph-coloring/C_cell/layer_norm_basic_lstm_cell/state/gamma:0"]).to(args.device)
    gcp.LSTM_c.ln_hcy.beta.data = torch.Tensor(loaded_dict["graph-coloring/C_cell/layer_norm_basic_lstm_cell/state/beta:0"]).to(args.device)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--embedding_size', type=int, default=64)
    parser.add_argument('--batch_size', type=int, default=1)
    parser.add_argument('--tmax', type=int, default=32)
    parser.add_argument('--device', type=str, default='cuda')
    args = parser.parse_args()

    embedding_size = args.embedding_size
    gcp = GCPNet(embedding_size, tmax=args.tmax, device=args.device)
    gcp.to(args.device)

    folder = os.path.join("data/experiments", "accuracy")
    files = FS().list_items(folder)
    files = files[1:] if len(files) > 1 else files

    for file in files:
        FS().get_file(file)

    test_ds = GraphDataSet(
        os.path.join("tmp/data/experiments/accuracy"),
        filter=lambda f: sum([f.split("/")[-1] in c3f for c3f in col3_files]) > 0,
        batch_size=args.batch_size,
    )
    test_dl = DataLoader(test_ds, batch_size=1, shuffle=True)

    load_from_tf(gcp)

    opt = torch.optim.Adam(
        gcp.parameters(),
        lr=2e-5,
        weight_decay=1e-10
    )

    epoch_size = 100_000 / embedding_size #128

    gcp.to(args.device)

    prev_train_flag_test = gcp.training
    gcp.eval()
    loss_agg = []
    with torch.no_grad():
        test_acc = 0
        for j, b in enumerate(test_dl):
            # if j == epoch_size: break
            M_vv, labels, cn, split, M_vc = b
            labels = labels.squeeze()
            split = split.squeeze(0)
            split = [int(s) for s in split]
            M_vv = M_vv.squeeze()
            M_vc = M_vc.squeeze()
            v_size = M_vv.shape[0]
            M_vv = M_vv.to(device=args.device)
            M_vc = M_vc.to(device=args.device)
            pred, means, V_ret, C_ret = gcp.forward(M_vv, M_vc, split, cn=cn, c_rand=True)
            test_acc += ((pred.detach().cpu() > 0.5) == torch.Tensor(labels)).sum()/float(cn.shape[1])
        test_acc /= (j+1)
    gcp.train(prev_train_flag_test)

    print(test_acc)

