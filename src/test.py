import torch
import torchvision
import torch.nn as nn
import torch.utils.data as data
import torch.nn.functional as F
from rdkit import Chem
from rdkit.Chem import AllChem
from rdkit.Chem import rdMolDescriptors
import numpy as np
import pandas as pd
import random
import math

class GraphConv(nn.Module):
    def __init__(self, input_size, output_size):
        super(GraphConv, self).__init__()
        self.input_size = input_size
        self.output_size = output_size
        self.weight = torch.nn.parameter.Parameter(torch.FloatTensor(input_size, output_size))
        self.bias = torch.nn.parameter.Parameter(torch.FloatTensor(output_size))

    def forward(self, X, A):
        in_wt = torch.matmul(X, self.weight)
        output = torch.bmm(A, in_wt) + self.bias
        return output


class GCN(nn.Module):
    def __init__(self, padding_size, input_size, hidden_size, output_size, class_num, dropout_rate):
        super(GCN, self).__init__()
        self.padding_size = padding_size
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.class_num = class_num
        self.dropout_rate = dropout_rate

        self.gcn1 = GraphConv(self.input_size, self.hidden_size)
        self.gcn2 = GraphConv(self.hidden_size, self.hidden_size * 2)
        self.gcn3 = GraphConv(self.hidden_size * 2, self.output_size)

        self.fc1 = nn.Linear(self.output_size * self.padding_size, self.padding_size)
        self.fc2 = nn.Linear(self.padding_size, self.class_num)
        self.sigmoid = nn.Sigmoid()

    def forward(self, X, A):
        X = F.relu(self.gcn1(X, A))
        X = F.relu(self.gcn2(X, A))
        X = F.relu(self.gcn3(X, A))
        X = F.dropout(X, self.dropout_rate, training=self.training)

        X = X.view(-1, self.output_size * self.padding_size)
        X = F.relu(self.fc1(X))
        X = self.fc2(X)
        X = self.sigmoid(X)
        return X

padding_size =132
unique = np.load('./unique.npy')
smiles_test = pd.read_csv('../test/names_smiles.txt')

adt_ft = 5
feature_size = len(unique) + adt_ft


def normalize(A):
    # A = A+I
    A = A + np.eye(A.shape[0])
    # 所有节点的度
    A_sum = A.sum(1)
    # D = D^-1/2
    D = np.diag(np.power(A_sum, -0.5))
    out = D.dot(A.dot(D))

    return out


def prework(data, length):
    _feature = np.zeros(shape=(length, padding_size, feature_size))
    _adj = np.zeros(shape=(length, padding_size, padding_size))

    cnt = 0
    for i in range(len(data)):
        feature = np.zeros(shape=(padding_size, feature_size))
        smile = data['SMILES'][i]
        mol = Chem.MolFromSmiles(smile)

        adj = Chem.GetAdjacencyMatrix(mol)
        adj = normalize(adj)
        adj_pad = np.pad(adj, ((0, padding_size - len(adj)), (0, padding_size - len(adj))), 'constant',
                         constant_values=0)

        cnt = 0
        for atom in mol.GetAtoms():
            onehots = np.zeros(shape=(len(unique)))
            idx = np.where(unique == atom.GetAtomicNum())
            onehots[idx[0]] += 1
            halogen = int(atom.GetAtomicNum() in [9, 17, 35, 53, 85])
            onehots = np.append(onehots, [atom.GetFormalCharge(), atom.GetTotalNumHs(), int(atom.GetIsAromatic()),
                                          int(atom.IsInRing()), halogen])
            feature[cnt] = onehots
            cnt += 1

        feature = np.array(feature)

        _feature[i] = feature
        _adj[i] = adj_pad
    return _feature, _adj

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# Assuming that we are on a CUDA machine, this should print a CUDA device:
print(device)


model_test = torch.load('./weights/model.pth')
model_test = model_test.to(device)
model_test.eval()

feature_test, adj_test = prework(smiles_test,len(smiles_test))

feature_test_ts = torch.tensor(torch.from_numpy(feature_test), dtype=torch.float32)
adj_test_ts = torch.tensor(torch.from_numpy(adj_test), dtype=torch.float32)

pred_test = model_test(feature_test_ts.to(device), adj_test_ts.to(device))
print(pred_test)
results = pred_test.cpu().detach().numpy()
print(results.shape)

import csv

f = open('output_518021910489.txt','w',encoding='utf-8',newline='')
csv_writer = csv.writer(f)

csv_writer.writerow(['Chemical','Label'])
for i in range(len(smiles_test)):
    csv_writer.writerow([smiles_test['Chemical'][i],results[i,0]])

f.close()