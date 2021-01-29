import torch
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

print(torch.__version__)

# load data
smiles_train = pd.read_csv('../train/names_smiles.txt')
smiles_val = pd.read_csv('../validation/names_smiles.txt')
smiles_test = pd.read_csv('../test/names_smiles.txt')
# get padding size
cnt = 0
padding_size = 0

for i in range(len(smiles_train)):
    smile = smiles_train['SMILES'][i]
    mol = Chem.MolFromSmiles(smile)
    for atom in mol.GetAtoms():
        cnt+=1
    if(cnt>=padding_size):
        padding_size = cnt
    cnt = 0
for i in range(len(smiles_val)):
    smile = smiles_val['SMILES'][i]
    mol = Chem.MolFromSmiles(smile)
    for atom in mol.GetAtoms():
        cnt+=1
    if(cnt>=padding_size):
        padding_size = cnt
    cnt = 0
for i in range(len(smiles_test)):
    smile = smiles_test['SMILES'][i]
    mol = Chem.MolFromSmiles(smile)
    for atom in mol.GetAtoms():
        cnt+=1
    if(cnt>=padding_size):
        padding_size = cnt
    cnt = 0
print(padding_size)


def get_unique(data):
    atom_unique = np.array([])
    for i in range(len(data)):
        smile = data['SMILES'][i]
        mol = Chem.MolFromSmiles(smile)

        for atom in mol.GetAtoms():
            atom_num = atom.GetAtomicNum()
            atom_unique = np.append(atom_unique, atom_num)
        atom_unique = np.unique(atom_unique)

    return atom_unique


# -----------------------------------------------
train_unique = get_unique(smiles_train)
val_unique = get_unique(smiles_val)
test_unique = get_unique(smiles_test)

unique = np.unique(np.concatenate([train_unique, val_unique, test_unique]))
# -----------------------------------------------

print(unique)
print(len(unique))

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


feature_train, adj_train = prework(smiles_train, len(smiles_train))
feature_val, adj_val = prework(smiles_val, len(smiles_val))

labels_train = pd.read_csv('../data/train/names_labels.txt')['Label']
labels_val = pd.read_csv('../data/validation/names_labels.txt')['Label']

labels_train = np.array(labels_train).reshape(-1,1)
labels_val = np.array(labels_val).reshape(-1,1)

nontox_idx = np.where((labels_train == 0).squeeze())
tox_idx = np.where((labels_train == 1).squeeze())

extend_scale = 1
disparity = nontox_idx[0].size - tox_idx[0].size
extend_length = int(extend_scale * disparity)

feature_extend = np.zeros(shape=(extend_length, padding_size, feature_size))
adj_extend = np.zeros(shape=(extend_length, padding_size, padding_size))
labels_extend = np.zeros(shape=(extend_length, 1))

for i in range(extend_length):
    rand_idx = np.random.choice(tox_idx[0])

    feature_extend[i] = feature_train[rand_idx]
    adj_extend[i] = adj_train[rand_idx]
    labels_extend[i] = labels_train[rand_idx]

print(feature_extend.shape, feature_train.shape)
print(adj_extend.shape, adj_train.shape)
print(labels_extend.shape, labels_train.shape)

feature_train = np.concatenate([feature_train, feature_extend])
adj_train = np.concatenate([adj_train, adj_extend])
labels_train = np.concatenate([labels_train, labels_extend])
print(feature_train.shape, adj_train.shape, labels_train.shape)

feature_train_ts = torch.tensor(torch.from_numpy(feature_train), dtype=torch.float32)
adj_train_ts = torch.tensor(torch.from_numpy(adj_train), dtype=torch.float32)
feature_val_ts = torch.tensor(torch.from_numpy(feature_val), dtype=torch.float32)
adj_val_ts = torch.tensor(torch.from_numpy(adj_val), dtype=torch.float32)

Y_train = torch.tensor(torch.from_numpy(labels_train), dtype=torch.float32)
Y_val = torch.tensor(torch.from_numpy(labels_val), dtype=torch.float32)

X_train = torch.cat([feature_train_ts,adj_train_ts],dim=2)
X_val = torch.cat([feature_val_ts,adj_val_ts],dim=2)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# Assuming that we are on a CUDA machine, this should print a CUDA device:
print(device)

def seed_torch(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True


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

#hyperparas
learning_rate = 1e-4
batch_size = 64
epoches = 25

model = GCN(padding_size, feature_size, hidden_size=128, output_size=64, class_num=1, dropout_rate=0.6)
print(model)
loss_func = torch.nn.BCELoss()
loss_func.to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

seed_torch(105)

for param in model.parameters():
    if param.dim() == 1:
        nn.init.constant_(param, 0)
    else:
        nn.init.xavier_uniform_(param)

model.to(device)


def GCN_train(model, dataset, epoch):
    model.train()
    loss_sum = 0

    for i, (x, y) in enumerate(dataset):
        X = x[:, :, :feature_size]
        A = x[:, :, feature_size:]
        X = X.to(device)
        A = A.to(device)

        pred = torch.squeeze(model(X, A))
        y = y.to(device)
        loss = loss_func(pred, y)
        loss_sum += loss
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    epoch_loss = loss_sum / len(dataset)
    #         train_loss_list.append(epoch_loss.item())
    print('epoch:', epoch, 'train loss:', epoch_loss)


#         return train_loss_list

def GCN_val(model, dataset, epoch):
    model.eval()
    loss = np.zeros(dataset.__len__())
    loss_sum = 0
    labels_auc = np.array([])
    pred_auc = np.array([])

    for i, (x, y) in enumerate(dataset):
        X = x[:, :, :feature_size]
        A = x[:, :, feature_size:]
        X = X.to(device)
        A = A.to(device)

        pred = torch.squeeze(model(X, A))
        y = y.to(device)
        loss = loss_func(pred, y)

        loss_sum += loss

        labels_auc = np.append(labels_auc, y.cpu().detach().numpy())
        pred_auc = np.append(pred_auc, pred.cpu().detach().numpy())

    epoch_loss = loss_sum / len(dataset)
    #         val_loss_list.append(epoch_loss.item())
    print('epoch:', epoch, 'valid loss:', epoch_loss)

    pos_cnt = labels_auc.sum()
    neg_cnt = len(pred_auc) - pos_cnt

    rank = np.argsort(pred_auc)
    ranklist = np.zeros(len(pred_auc))
    for i in range(len(pred_auc)):
        if labels_auc[rank[i]] == 1:
            ranklist[rank[i]] = i + 1
    auc = (ranklist.sum() - pos_cnt * (pos_cnt + 1) / 2) / (pos_cnt * neg_cnt)
    #         return auc,val_loss_list
    return auc


if __name__ == "__main__":
    AUC_best = 0

    dataset = data.TensorDataset(X_train, Y_train)
    dataset_val = data.TensorDataset(X_val, Y_val)
    data_iter = data.DataLoader(
        dataset=dataset,
        batch_size=batch_size,
        shuffle=True,
    )
    data_iter_val = data.DataLoader(
        dataset=dataset_val,
        batch_size=batch_size,
        shuffle=False,
    )

    for epoch in range(epoches):
        #         train_loss_list = GCN_train(model, data_iter,epoch)
        #         epoch_auc,val_loss_list = GCN_val(model, data_iter_val,epoch)
        GCN_train(model, data_iter, epoch)
        epoch_auc = GCN_val(model, data_iter_val, epoch)

        if epoch_auc > AUC_best:
            AUC_best = epoch_auc
            torch.save(model, 'model_best_auc.pth')
        print('validation AUC:', epoch_auc)
        #         if epoch_auc > 0.87:
        #             torch.save(model,str(epoch)+'model_ba4_e50_1215_5.pth')
        #         fig, ax = plt.subplots(2, 1)
        #         axis = ax.flatten()
        #         axis[1].set_xlabel("Epoch")
        #         axis[0].set_ylabel("Train Loss")
        #         axis[0].plot(train_loss_list)

        #         axis[1].set_xlabel("Epoch")
        #         axis[1].set_ylabel("Validation Loss")
        #         axis[1].plot(val_loss_list, color="red")
        #         plt.savefig('./train_val_loss.jpg')
        if (epoch + 1 in [16, 17, 18, 19, 20]):
            torch.save(model, str(epoch + 1) + '_model_epoch.pth')