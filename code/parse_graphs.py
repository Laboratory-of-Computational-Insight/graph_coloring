import glob
import json
import math
import random
import torch
import numpy as np

# import networkx as nx
from networkx.algorithms import bipartite

from random_planted import create_planted, create_planted_with_one_side, create_planted_with_one_contradicting, \
    create_gnp, create_one_side, create_planted_with_one_side_single


def add_random_edge(v_mat, ass):
    n = v_mat.shape[0]
    n_range = torch.arange(n)
    while True:
        i = random.sample(range(n), 1)[0]
        indexes = n_range[(ass != ass[i]) & (v_mat[i] == 0)]
        j = torch.permute(indexes, dims=(0,))
        if len(j) == 0: continue
        j = j[0]
        print("adding edge", i, j)
        v_mat[i, j] = v_mat[j, i] = 1
        break


class ConvertToTensor(object):
    '''
    This class converts JSON graphs into torch tensors to deal with.
    '''
    def __init__(self, graph_dir_path, device='cuda', filter = lambda x: True):
        self._gp = glob.glob('{}/*.json'.format(graph_dir_path))
        self._gp = [g for g in self._gp if filter(g)]
        self.device = device

    BATCH_CACHE = {}
    @staticmethod
    def get_batch(jsons, device='cpu'):
        # load batch of graphs
        V_matricies = []
        C_matricies = []
        breaking_edges = []
        splits = []
        colors = []

        jsons2 = []
        for j in jsons:
            jsons2.append(j)
            jsons2.append(j)
        jsons = jsons2

        for j in jsons:
            if j not in ConvertToTensor.BATCH_CACHE:
                with open(j, 'r') as f:
                    l = json.load(f)
                n = l['v']
                v_mat = torch.Tensor(l['m']).reshape([n,n])
                col = l['c_number']
                be = l['change_coord']
                split = v_mat.shape[0]
                c_mat = torch.ones(n,col)
                k = col
                c = (v_mat.sum()/(n-1)).item()
                p = c/n

                ConvertToTensor.BATCH_CACHE[j] = (v_mat, c_mat, be, col, split)

            v_mat, c_mat, b_edges, c, s = ConvertToTensor.BATCH_CACHE[j]
            V_matricies.append(v_mat.clone())
            C_matricies.append(c_mat.clone())
            breaking_edges.append(b_edges)
            colors.append(c)
            splits.append(s)


        labels = torch.randint(0, 2, size=(len(V_matricies),)).float()

        for i in range(len(V_matricies)):
            if labels[i] == 0:
                v, u = breaking_edges[i]
                V_matricies[i][v][u] = V_matricies[i][u][v] = 1
        V_mat = torch.block_diag(*V_matricies)
        C_mat = torch.block_diag(*C_matricies)
        return V_mat.to(device), labels.to(device), torch.tensor(colors).to(device), torch.Tensor(splits).to(device), C_mat.to(device)

    def random_graph(self):
        return self.get_one(random.choice(self._gp))
    @staticmethod
    def get_one(g):
        # load graph
        jg = json.load(open(g))
        mat = torch.Tensor(jg['m']).reshape([jg['v'], jg['v']])
        mat_adv = mat.clone()
        cc = jg['change_coord']
        mat_adv[cc[0], cc[1]] = 1
        mat_adv[cc[1], cc[0]] = 1
        return mat, mat_adv, jg['c_number']
