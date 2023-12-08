import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import math
from torch.nn import Parameter

from NodeAttention import NodeAttention

from HeterogRelationAttention import RelationAttention
import scipy.sparse as sp
from matplotlib import pyplot as plt
import pdb
import os

os.environ["CUDA_VISIBLE_DEVICES"] = "0"
device = f'cuda' if torch.cuda.is_available() else 'cpu'


class HGRAN(nn.Module):
    def __init__(self, num_edge, num_channels, w_in, w_out, num_class, norm, args=None):
        super(HGRAN, self).__init__()
        self.num_edge = num_edge  # 5
        self.num_channels = num_channels  # 1
        self.w_in = w_in
        self.w_out = w_out
        self.num_class = num_class
        self.norm = norm
        self.args = args
        self.HeterogeneousConv = HeterogeneousConv(in_channels=num_edge, out_channels=num_channels)
        self.HomogeneousConv = HomogeneousConv(in_channels=num_edge, out_channels=num_channels)
        self.HomogRelationAttention = NodeAttention(input=self.w_in, output=self.w_out)
        self.HeterogRelationAttention = RelationAttention(in_channels=self.w_in, hidden_channels=self.w_out)
        self.loss = nn.CrossEntropyLoss()
        self.register_buffer("epsilon", torch.FloatTensor([1e-12]))
        self.linear1 = nn.Linear(self.w_out * 2, self.w_out)
        self.linear2 = nn.Linear(self.w_out, self.num_class)

    def forward(self, A, X, target_x, target):

        A = A.unsqueeze(0).permute(0, 3, 1, 2)  # A.unsqueeze(0)=[1,N,N,edgeType]=>[1,edgeType,N,N];
        Ws = []
        Wh = []
        # Homogeneous relationship attention
        H_features = X.to(device)
        H = A
        H, W_h= self.HomogeneousConv(H)
        Wh.append(W_h)
        H = torch.squeeze(H).to(device)

        z_homog = self.HomogRelationAttention(H_features, H)  # NodeAttention


        # Heterogeneous Relationship Attention
        A, Wr = self.HeterogeneousConv(A)  # Heterogeneous RelationConv
        Ws.append(Wr)
        A = torch.squeeze(A).to(device)

        # RelationAttention
        z_heterog = self.HeterogRelationAttention(X, A)

        z = torch.cat((z_heterog, z_homog), dim=1)

        if self.norm == 'true':
            z = z / (torch.max(torch.norm(z, dim=1, keepdim=True), self.epsilon))   # Z--L2-norm
        else:
            z=z

        Z = self.linear1(z)
        Z = F.relu(Z)
        y = self.linear2(Z[target_x])
        loss = self.loss(y, target)
        return loss, y, Ws

class HomogeneousConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(HomogeneousConv, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.homogeconv=Conv2(in_channels, out_channels)

    def forward(self, H):
        H = self.homogeconv(H)
        W_h = [(F.softmax(self.homogeconv.weight, dim=1)).detach()]
        return H, W_h

class HeterogeneousConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(HeterogeneousConv, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.conv2 = Conv2(in_channels, out_channels)

    def forward(self, A):  # A:[1,edgeType,N,N]
        A = F.relu(self.conv2(A))  # ConvLayer=>[1, N, N]
        W2 = [(F.softmax(self.conv2.weight, dim=1)).detach()]
        return A, W2


# Conv卷积
class Conv2(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(Conv2, self).__init__()
        self.in_channels = in_channels  # >>>5
        self.out_channels = 1  # >>>1
        self.weight = nn.Parameter(torch.Tensor(out_channels, in_channels, 1, 1), requires_grad=True)
        #                                            (输出，输入，关系卷积)
        self.bias = None
        self.scale = nn.Parameter(torch.Tensor([0.1]), requires_grad=False)
        self.reset_parameters()

    # 更新参数
    def reset_parameters(self):
        nn.init.constant_(self.weight, 0.2)
        if self.bias is not None:
            fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.weight)
            bound = 1 / math.sqrt(fan_in)
            nn.init.uniform_(self.bias, -bound, bound)

    def forward(self, A):  # self.weight:带有channel的conv;
        '''
        0) 对weight(conv)进行softmax
        1) 对每个节点在每个edgeType上进行[1, 5, 1, 1]的卷积操作;
        '''
        A = torch.sum(A * F.softmax(self.weight, dim=1), dim=1)
        return A
