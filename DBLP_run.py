import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
from model import HRANN
import pdb
import pickle
import argparse
from utils import f1_score
from sklearn.metrics import f1_score as sk_f1_score

from sklearn.cluster import KMeans
from sklearn.metrics.cluster import normalized_mutual_info_score as nmi_score
from sklearn.metrics import adjusted_rand_score as ari_score

from matplotlib import pyplot as plt
import os
from sklearn.manifold import TSNE

os.environ["CUDA_VISIBLE_DEVICES"] = "0"
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 训练set
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--epoch', type=int, default=60,
                        help='Training Epochs')
    parser.add_argument('--node_dim', type=int, default=64,
                        help='Node dimension')
    parser.add_argument('--num_channels', type=int, default=1,
                        help='number of channels')
    parser.add_argument('--lr', type=float, default=0.02,
                        help='learning rate')
    parser.add_argument('--norm', type=str, default='true',
                        help='L2-normalization')
    parser.add_argument('--weight_decay', type=float, default=0.00001,
                        help='l2 reg')
    parser.add_argument('--adaptive_lr', type=str, default='True',
                        help='adaptive learning rate')

    args = parser.parse_args()
    args.dataset = 'DBLP'
    args.adaptive_lr = 'True'
    args.norm = 'true'

    epochs = args.epoch

    node_dim = args.node_dim
    num_channels = args.num_channels
    lr = args.lr
    norm = args.norm
    weight_decay = args.weight_decay
    adaptive_lr = args.adaptive_lr
    print(args)

    # dataset
    with open('data/' + args.dataset + '/node_features.pkl', 'rb') as f:
        node_features = pickle.load(f)
    with open('data/' + args.dataset + '/edges.pkl', 'rb') as f:
        edges = pickle.load(f)
    # label
    with open('data/' + args.dataset + '/labels.pkl', 'rb') as f:
        labels = pickle.load(f)  # train; val; test
    num_nodes = edges[0].shape[0]

    for i, edge in enumerate(edges):
        if i == 0:
            A = torch.from_numpy(edge.todense()).type(torch.cuda.FloatTensor).unsqueeze(-1)
        else:
            A = torch.cat([A, torch.from_numpy(edge.todense()).type(torch.cuda.FloatTensor).unsqueeze(-1)], dim=-1)
    A = torch.cat([A, torch.eye(num_nodes).type(torch.cuda.FloatTensor).unsqueeze(-1)], dim=-1)
    channels = len(edges)

    node_features = torch.from_numpy(node_features).type(torch.cuda.FloatTensor)
    train_node = torch.from_numpy(np.array(labels[0])[:, 0]).type(torch.cuda.LongTensor)  # labels
    train_target = torch.from_numpy(np.array(labels[0])[:, 1]).type(torch.cuda.LongTensor)  # val
    valid_node = torch.from_numpy(np.array(labels[1])[:, 0]).type(torch.cuda.LongTensor)
    valid_target = torch.from_numpy(np.array(labels[1])[:, 1]).type(torch.cuda.LongTensor)
    test_node = torch.from_numpy(np.array(labels[2])[:, 0]).type(torch.cuda.LongTensor)  # test
    test_target = torch.from_numpy(np.array(labels[2])[:, 1]).type(torch.cuda.LongTensor)

    print('node_features:', node_features.shape)
    # num_classes >>> 4
    num_classes = torch.max(train_target).item() + 1
    max_NMI, max_ARI = [], []
    best_NMI, best_ARI = [], []

    max_micro = []
    max_macro = []
    best_micro, best_macro = [], []

    for l in range(1):
        model = HRANN(num_edge=A.shape[- 1],
                    num_channels=num_channels,
                    w_in=node_features.shape[1],
                    w_out=node_dim,
                    num_class=num_classes,
                    norm=args.norm
                    )

        if adaptive_lr == 'False':
            optimizer = torch.optim.Adam(model.parameters(), lr=0.01, weight_decay=0.0)
        else:
            optimizer = torch.optim.Adam([{'params': model.linear1.parameters()},
                                          {'params': model.linear2.parameters()},
                                          {'params': model.HeterogeneousConv.parameters()},
                                          {'params': model.HomogRelationAttention.parameters(), "lr": 0.01, "weight_decay": 0.0},
                                          {'params': model.HeterogRelationAttention.parameters(), "lr": 0.01,"weight_decay": 0.0},
                                          ], lr=0.02, weight_decay=0.0)
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model.to(device)
        loss = nn.CrossEntropyLoss()
        # Train & Valid & Test
        best_val_loss = 10000
        best_test_loss = 10000
        best_train_loss = 10000
        max_val_loss = 10000
        max_test_loss = 10000
        best_train_f1, best_micro_train_f1 = 0, 0
        best_val_f1, best_micro_val_f1 = 0, 0
        best_test_f1, best_micro_test_f1 = 0, 0
        best_test_nmi, best_test_ari = 0, 0
        max_val_macro_f1, max_val_micro_f1 = 0, 0
        max_test_macro_f1, max_test_micro_f1 = 0, 0
        max_test_nmi, max_test_ari = 0, 0
        kmeans = KMeans(n_clusters=4, n_init=10)
        for i in range(epochs):
            for param_group in optimizer.param_groups:
                if param_group['lr'] > 0.0001:
                    param_group['lr'] = param_group['lr'] * 0.9
            print('Epoch:  ', i + 1)

            model.zero_grad()
            model.train()
            loss, y_train, Ws = model(A, node_features, train_node, train_target)
            y_pred = kmeans.fit_predict(y_train.detach().cpu().numpy())
            train_target1 = train_target.detach().cpu()
            train_nmi = nmi_score(train_target1, y_pred, average_method='arithmetic')
            train_ari = ari_score(train_target1, y_pred)

            train_f1 = torch.mean(
                f1_score(torch.argmax(y_train.detach(), dim=1), train_target, num_classes=num_classes)).cpu().numpy()
            sk_train_f1 = sk_f1_score(train_target.detach().cpu(), np.argmax(y_train.detach().cpu(), axis=1),
                                      average='micro')
            print('Train - Loss: {}, Macro_F1: {}, Micro_F1: {}'.format(loss.detach().cpu().numpy(), train_f1,
                                                                        sk_train_f1))
            print('Train:  nmi {:.4f}'.format(train_nmi), ', ari {:.4f}'.format(train_ari))

            loss.backward()
            optimizer.step()
            model.eval()
            # Valid
            with torch.no_grad():
                val_loss, y_valid, _ = model.forward(A, node_features, valid_node, valid_target)

                vallid_z = kmeans.fit_predict(y_valid.detach().cpu().numpy())
                valid_target1 = valid_target.detach().cpu()
                valid_nmi = nmi_score(valid_target1, vallid_z, average_method='arithmetic')
                valid_ari = ari_score(valid_target1, vallid_z)

                val_f1 = torch.mean(
                    f1_score(torch.argmax(y_valid, dim=1), valid_target, num_classes=num_classes)).cpu().numpy()
                sk_val_f1 = sk_f1_score(valid_target.detach().cpu(), np.argmax(y_valid.detach().cpu(), axis=1),
                                        average='micro')
                print('Valid - Loss: {}, Macro_F1: {}, Micro_F1: {}'.format(val_loss.detach().cpu().numpy(), val_f1,
                                                                            sk_val_f1))
                print('Valid:   nmi {:.4f}'.format(valid_nmi), ', ari {:.4f}'.format(valid_ari))

                # Test
                test_loss, y_test, W = model.forward(A, node_features, test_node, test_target)

                test_z = kmeans.fit_predict(y_test.detach().cpu().numpy())
                test_target1 = test_target.detach().cpu()

                test_nmi = nmi_score(test_target1, test_z, average_method='arithmetic')
                test_ari = ari_score(test_target1, test_z)

                test_f1 = torch.mean(
                    f1_score(torch.argmax(y_test, dim=1), test_target, num_classes=num_classes)).cpu().numpy()
                sk_test_f1 = sk_f1_score(test_target.detach().cpu(), np.argmax(y_test.detach().cpu(), axis=1),
                                         average='micro')

            if val_f1 > best_val_f1:
                best_val_loss = val_loss.detach().cpu().numpy()
                best_test_loss = test_loss.detach().cpu().numpy()
                best_train_loss = loss.detach().cpu().numpy()
                best_train_f1 = train_f1
                best_val_f1 = val_f1
                best_test_f1 = test_f1
                best_micro_train_f1 = sk_train_f1
                best_micro_val_f1 = sk_val_f1
                best_micro_test_f1 = sk_test_f1  # micro
                best_test_nmi = test_nmi
                best_test_ari = test_ari

            if val_f1 > max_val_macro_f1:
                max_val_loss = val_loss.detach().cpu().numpy()
                max_val_macro_f1 = val_f1
            if sk_val_f1 > max_val_micro_f1:
                max_val_loss = val_loss.detach().cpu().numpy()
                max_val_micro_f1 = sk_val_f1
            if test_f1 > max_test_macro_f1:
                max_test_loss = test_loss.detach().cpu().numpy()
                max_test_macro_f1 = test_f1
            if sk_test_f1 > max_test_micro_f1:
                max_test_loss = val_loss.detach().cpu().numpy()
                max_test_micro_f1 = sk_test_f1

            if test_nmi > max_test_nmi:
                max_test_loss = test_loss.detach().cpu().numpy()
                max_test_nmi = test_nmi
            if test_ari > max_test_ari:
                max_test_loss = val_loss.detach().cpu().numpy()
                max_test_ari = test_ari


        print('---------------Test Results--------------------')
        print('Valid - Loss: {:.4f}, maxMacro_F1: {:.4f}, maxMicro_F1: {:.4f}'.format(max_val_loss, max_val_macro_f1,
                                                                                      max_val_micro_f1))
        print('Test - Loss: {:.4f}, maxMacro_F1: {:.4f}, maxMicro_F1: {:.4f}'.format(max_test_loss, max_test_macro_f1,
                                                                                     max_test_micro_f1))

        print('Test - Loss: {:.4f}, max_nim: {:.4f}, max_ari: {:.4f}'.format(max_test_loss, max_test_nmi,
                                                                             max_test_ari))
        max_macro.append(max_test_macro_f1)
        max_micro.append(max_test_micro_f1)
        max_NMI.append(max_test_nmi)
        max_ARI.append(max_test_ari)

        best_macro.append(best_test_f1)
        best_micro.append(best_micro_test_f1)
        best_NMI.append(best_test_nmi)
        best_ARI.append(best_test_ari)
    print("Total run class results", max_macro, max_micro)
    print("Total run cluster results", max_NMI, max_ARI)
    print("Average results", sum(max_macro) / len(max_macro), sum(max_micro) / len(max_micro))
    print("Average results", sum(max_NMI) / len(max_NMI), sum(max_ARI) / len(max_ARI))

    f=open("datadblp.dat","w")
    print("Run Macro_F1 Micro_F1 NMI ARI",file=f)
    f=open("data.dat","a")
    f.close 

    f=open("datadblp.dat","a")
    for i in range(len(max_macro)):
        print(i," ",max_macro[i]," ",max_micro[i]," ",max_NMI[i]," ",max_ARI[i],file=f)
    f.close