from torch_geometric.nn import GATv2Conv
import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import math
import scipy.sparse as sp
from torch.nn import Parameter
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"


class NodeAttention(nn.Module):
    def __init__(self, input, output, heads=6):
        super(NodeAttention, self).__init__()
        self.conv1 = GATv2Conv(in_channels=input, out_channels=output, heads=heads, concat=False, negative_slope=0.2,
                               dropout=0.5)
        self.conv2 = GATv2Conv(in_channels=output, out_channels=output, heads=heads, concat=False, negative_slope=0.2,
                               dropout=0.5)
        # dropout----ACM：0.5  DBLP:0.2  IMDB 0.5
    def forward(self, x, adj):
        # Convert to sparse matrix
        adj = adj.cpu().detach().numpy()
        adj = sp.coo_matrix(adj)
        values = adj.data
        indices = np.vstack((adj.row, adj.col))  # coo形式
        edge_index = torch.cuda.LongTensor(indices)  # coo形式

        x = self.conv1(x, edge_index)
        x = F.relu(x)

        x = self.conv2(x, edge_index)
        x = F.relu(x)
        return x
