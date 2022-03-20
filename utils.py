import numpy as np
import scipy.sparse as sp
import torch
import networkx as nx
from torch_geometric.utils.convert import to_networkx
from scipy.linalg import fractional_matrix_power, inv


# Normlization
def aug_normalized_adjacency(adj):
    adj = adj + sp.eye(adj.shape[0])
    adj = sp.coo_matrix(adj)
    row_sum = np.array(adj.sum(1))
    d_inv_sqrt = np.power(row_sum, -0.5).flatten()
    d_inv_sqrt[np.isinf(d_inv_sqrt)] = 0.
    d_mat_inv_sqrt = sp.diags(d_inv_sqrt)
    return d_mat_inv_sqrt.dot(adj).dot(d_mat_inv_sqrt).tocoo()


def fetch_normalization(type):
    switcher = {
        'AugNormAdj': aug_normalized_adjacency,  # A' = (D + I)^-1/2 * ( A + I ) * (D + I)^-1/2
    }
    func = switcher.get(type, lambda: "Invalid normalization technique.")
    return func


def row_normalize(mx):
    """Row-normalize sparse matrix"""
    rowsum = np.array(mx.sum(1))
    r_inv = np.power(rowsum, -1).flatten()
    r_inv[np.isinf(r_inv)] = 0.
    r_mat_inv = sp.diags(r_inv)
    mx = r_mat_inv.dot(mx)
    return mx


# Get Enhanced Adjacency Matrix
def get_Enhanced_Adjacency_Matrix(adj, r):
    adj = adj.to_dense()
    adj_label = adj
    if r == 1:
        adj_label = adj
    elif r == 2:
        adj_label = adj + adj @ adj
    elif r == 3:
        adj_label = adj + (adj @ adj) + (adj @ adj @ adj)
    elif r == 4:
        adj_label = adj + (adj @ adj) + (adj @ adj @ adj) + (adj @ adj @ adj @ adj)
    elif r == 5:
        adj_label = adj + (adj @ adj) + (adj @ adj @ adj) + (adj @ adj @ adj @ adj) + (adj @ adj @ adj @ adj @ adj)
    elif r == 6:
        adj_label = adj + (adj @ adj) + (adj @ adj @ adj) + (adj @ adj @ adj @ adj) + (adj @ adj @ adj @ adj @ adj) + (
                adj @ adj @ adj @ adj @ adj @ adj)
    elif r == 7:
        adj_label = adj + (adj @ adj) + (adj @ adj @ adj) + (adj @ adj @ adj @ adj) + (adj @ adj @ adj @ adj @ adj) + (
                adj @ adj @ adj @ adj @ adj @ adj) + (adj @ adj @ adj @ adj @ adj @ adj @ adj)
    return adj_label


def get_enhanced_Adjacency_Matrix(adj, r):
    adj = adj.to_dense()
    adj_label = adj
    if r == 1:
        adj_label = adj
    elif r == 2:
        adj_label = adj @ adj
    elif r == 3:
        adj_label = (adj @ adj @ adj)
    elif r == 4:
        adj_label = (adj @ adj @ adj @ adj)
    elif r == 5:
        adj_label = (adj @ adj @ adj @ adj @ adj)
    elif r == 6:
        adj_label = (adj @ adj @ adj @ adj @ adj @ adj)
    elif r == 7:
        adj_label = (adj @ adj @ adj @ adj @ adj @ adj @ adj)
    return adj_label

# Get Enhanced Adjacency Matrix
def get_coauthor_amazon_Enhanced_Adjacency_Matrix(adj, r):
    adj_label = adj
    if r == 1:
        adj_label = adj
    elif r == 2:
        adj_label = adj + adj @ adj
    elif r == 3:
        adj_label = adj + (adj @ adj) + (adj @ adj @ adj)
    elif r == 4:
        adj_label = adj + (adj @ adj) + (adj @ adj @ adj) + (adj @ adj @ adj @ adj)
    elif r == 5:
        adj_label = adj + (adj @ adj) + (adj @ adj @ adj) + (adj @ adj @ adj @ adj) + (adj @ adj @ adj @ adj @ adj)
    elif r == 6:
        adj_label = adj + (adj @ adj) + (adj @ adj @ adj) + (adj @ adj @ adj @ adj) + (adj @ adj @ adj @ adj @ adj) + (
                adj @ adj @ adj @ adj @ adj @ adj)
    elif r == 7:
        adj_label = adj + (adj @ adj) + (adj @ adj @ adj) + (adj @ adj @ adj @ adj) + (adj @ adj @ adj @ adj @ adj) + (
                adj @ adj @ adj @ adj @ adj @ adj) + (adj @ adj @ adj @ adj @ adj @ adj @ adj)
    return adj_label

# Get feature similarity Matrix
def get_feature_dis(x):
    """
    x :           batch_size x nhid
    x_dis(i,j):   item means the similarity between x(i) and x(j).
    """
    x_dis = x @ x.T
    mask = torch.eye(x_dis.shape[0]).cuda()
    x_sum = torch.sum(x ** 2, 1).reshape(-1, 1)
    x_sum = torch.sqrt(x_sum).reshape(-1, 1)
    x_sum = x_sum @ x_sum.T
    x_dis = x_dis * (x_sum ** (-1))
    x_dis = (1 - mask) * x_dis
    return x_dis


#
def parse_index_file(filename):
    """Parse index file."""
    index = []
    for line in open(filename):
        index.append(int(line.strip()))
    return index


def preprocess_citation(adj, features, normalization="FirstOrderGCN"):
    adj_normalizer = fetch_normalization(normalization)

    adj = adj_normalizer(adj)

    features = row_normalize(features)
    return adj, features


def sparse_mx_to_torch_sparse_tensor(sparse_mx):
    """Convert a scipy sparse matrix to a torch sparse tensor."""
    sparse_mx = sparse_mx.tocoo().astype(np.float32)
    indices = torch.from_numpy(
        np.vstack((sparse_mx.row, sparse_mx.col)).astype(np.int64))
    values = torch.from_numpy(sparse_mx.data)
    shape = torch.Size(sparse_mx.shape)
    return torch.sparse.FloatTensor(indices, values, shape)


def accuracy(output, labels):
    preds = output.max(1)[1].type_as(labels)
    correct = preds.eq(labels).double()
    correct = correct.sum()
    return correct / len(labels)


#
def get_batch(batch_size, adj_label, idx_train, features):
    """
    get a batch of feature & adjacency matrix
    """
    rand_indx = torch.tensor(np.random.choice(np.arange(adj_label.shape[0]), batch_size)).type(torch.long).cuda()
    rand_indx[0:len(idx_train)] = idx_train
    features_batch = features[rand_indx]
    adj_label_batch = adj_label[rand_indx, :][:, rand_indx]
    return features_batch, adj_label_batch


# Neighbor Contrast Loss
def Ncontrast(x_dis, adj_label, tau=1):
    """
    compute the Ncontrast loss
    """
    x_dis = torch.exp(tau * x_dis)
    x_dis_sum = torch.sum(x_dis, 1)
    x_dis_sum_pos = torch.sum(x_dis * adj_label, 1)
    loss = -torch.log(x_dis_sum_pos * (x_dis_sum ** (-1)) + 1e-8).mean()
    return loss


def get_A_hat(data):
    G = to_networkx(data)
    A = nx.convert_matrix.to_numpy_array(G)
    I = np.eye(A.shape[0])
    A_add_self = A + np.eye(A.shape[0])  # A^ = A + I_n
    dInfo = np.sum(A_add_self, 1)
    D = np.diag(dInfo)  # D^ = Sigma A^_ii
    dinv = fractional_matrix_power(D, -0.5)  # D^(-1/2)
    A_hat = np.matmul(np.matmul(dinv, A), dinv)  # A~ = D^(-1/2) x A^ x D^(-1/2)
    return A_hat

def index_to_mask(index, size):
    mask = torch.zeros(size, dtype=torch.bool, device=index.device)
    mask[index] = 1
    return mask

def random_coauthor_amazon_splits(data, num_classes, lcc_mask):
    # Set random coauthor/co-purchase splits:
    # * 20 * num_classes labels for training
    # * 30 * num_classes labels for validation
    # rest labels for testing

    indices = []
    if lcc_mask is not None:
        for i in range(num_classes):
            index = (data.y[lcc_mask] == i).nonzero().view(-1)
            index = index[torch.randperm(index.size(0))]
            indices.append(index)
    else:
        for i in range(num_classes):
            index = (data.y == i).nonzero().view(-1)
            index = index[torch.randperm(index.size(0))]
            indices.append(index)

    train_index = torch.cat([i[:20] for i in indices], dim=0)
    val_index = torch.cat([i[20:50] for i in indices], dim=0)

    rest_index = torch.cat([i[50:] for i in indices], dim=0)
    rest_index = rest_index[torch.randperm(rest_index.size(0))]

    data.train_mask = index_to_mask(train_index, size=data.num_nodes)
    data.val_mask = index_to_mask(val_index, size=data.num_nodes)
    data.test_mask = index_to_mask(rest_index, size=data.num_nodes)

    return data