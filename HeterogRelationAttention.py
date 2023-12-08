import argparse
import os.path as osp
import numpy as np
import torch.nn as nn
import pickle
import torch
import scipy.sparse as sp
from torch_geometric.nn import AGNNConv
import torch.nn.functional as F

import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

class RelationAttention(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels):
        super().__init__()
        self.linear1 = torch.nn.Linear(in_channels, hidden_channels)
        self.prop1 = AGNNConv(requires_grad=False)
        self.prop2 = AGNNConv(requires_grad=True)
        self.linear2 = torch.nn.Linear(hidden_channels, hidden_channels)

        self.loss = nn.CrossEntropyLoss()

    def forward(self, x, adj):
        H = adj
        adj = adj.cpu().detach().numpy()
        adj = sp.coo_matrix(adj)
        values = adj.data  # weight
        indices = np.vstack((adj.row, adj.col))
        edge_index = torch.cuda.LongTensor(indices) # coo

        x = F.dropout(x, p=0.3, training=self.training)  # ACM：0.3  DBLP:0.2  IMDB 0.5

        x = F.relu(self.linear1(x))

        x = self.prop1(x, edge_index)
        x = self.prop2(x, edge_index)

        x = F.dropout(x, p=0.3, training=self.training)
        x = self.linear2(x)
        x = F.relu(x)
        return x
        #return torch.mm(H,x)

