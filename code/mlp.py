import torch
import torch.nn as nn


class MLP(nn.Module):
    def __init__(self, in_dim, hidden_dim, out_dim, with_act=True):
        super(MLP, self).__init__()
        self.in_dim = in_dim
        self.hidden_dim = hidden_dim
        self.out_dim = out_dim
        self.with_act = with_act

        self.l1 = nn.Linear(in_dim, hidden_dim)
        self.l2 = nn.Linear(hidden_dim, hidden_dim)
        self.l3 = nn.Linear(hidden_dim, hidden_dim)
        self.l4 = nn.Linear(hidden_dim, out_dim)

    def forward(self, x):
        original_x = x
        x = self.l1(x)
        if self.with_act: x = torch.relu(x)
        x = self.l2(x)
        if self.with_act: x = torch.relu(x)
        x = self.l3(x)
        if self.with_act: x = torch.relu(x)
        x = self.l4(x)
        return x

