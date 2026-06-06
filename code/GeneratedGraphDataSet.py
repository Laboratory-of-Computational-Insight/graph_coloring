import torch
from torch.utils.data import Dataset

class GeneratedGraphs(Dataset):
    def __init__(self, adj_matrices, k=3):
        self.adj_matrices = adj_matrices
        self.k = k

    def __len__(self):
        return len(self.adj_matrices)

    def __getitem__(self, idx):
        adj_matrix = self.adj_matrices[idx]
        return (
                    adj_matrix.to(torch.float32),
                    torch.tensor([1], dtype=torch.float64),
                    torch.tensor([self.k], dtype=torch.int64),
                    torch.tensor([adj_matrix.shape[0]], dtype=torch.float64),
                    torch.tensor([[1]*self.k]*adj_matrix.shape[0], dtype=torch.float64),
                )

