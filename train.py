import torch
import torch.nn.functional as F
from torch.optim import Adam
from torch import tensor

from utils import get_batch, Ncontrast, accuracy
from tqdm import tqdm
from torch_geometric.utils.convert import to_networkx
import time
import networkx as nx

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def train_citation(alpha, beta, tau, batch_size, adj_label, idx_train, features, labels, model, optimizer):

    features_batch, adj_label_batch = get_batch(batch_size, adj_label, idx_train, features)
    model.train()
    optimizer.zero_grad()
    x_bar, x_dis, output = model(features_batch)
    loss_train_class = F.nll_loss(output[idx_train], labels[idx_train])
    loss_Ncontrast = Ncontrast(x_dis, adj_label_batch, tau=tau)
    loss_train = loss_train_class + alpha * loss_Ncontrast + beta * F.mse_loss(x_bar, features_batch)
    acc_train = accuracy(output[idx_train], labels[idx_train])
    loss_train.backward()
    optimizer.step()
    return


def evaluate_citation(model, features, labels, idx_train, idx_val, idx_test):
    model.eval()

    with torch.no_grad():
        logits = model(features)

    outs = {}
    train_loss = F.nll_loss(logits[idx_train], labels[idx_train]).item()
    train_pred = logits[idx_train].max(1)[1]
    train_acc = train_pred.eq(labels[idx_train]).sum().item() / len(idx_train)

    outs['train_loss'] = train_loss
    outs['train_acc'] = train_acc

    val_loss = F.nll_loss(logits[idx_val], labels[idx_val]).item()
    val_pred = logits[idx_val].max(1)[1]
    val_acc = val_pred.eq(labels[idx_val]).sum().item() / len(idx_val)

    outs['val_loss'] = val_loss
    outs['val_acc'] = val_acc

    test_loss = F.nll_loss(logits[idx_test], labels[idx_test]).item()
    test_pred = logits[idx_test].max(1)[1]
    test_acc = test_pred.eq(labels[idx_test]).sum().item() / len(idx_test)

    outs['test_loss'] = test_loss
    outs['test_acc'] = test_acc

    return outs


def train_coauthor_amazon(data, alpha, beta, tau, adj_label, model, optimizer):
    model.train()
    optimizer.zero_grad()
    x_bar, x_dis, output = model(data.x)
    loss_train_class = F.nll_loss(output[data.train_mask], data.y[data.train_mask])
    loss_Ncontrast = Ncontrast(x_dis, adj_label, tau)
    loss_train = loss_train_class + alpha * loss_Ncontrast + beta * F.mse_loss(x_bar, data.x)
    loss_train.backward()
    optimizer.step()
    return


def evaluate_coauthor_amazon(model, data):
    model.eval()

    with torch.no_grad():
        logits = model(data.x)

    outs = {}
    for key in ['train', 'val', 'test']:
        mask = data['{}_mask'.format(key)]
        loss = F.nll_loss(logits[mask], data.y[mask]).item()
        pred = logits[mask].max(1)[1]
        acc = pred.eq(data.y[mask]).sum().item() / mask.sum().item()

        outs['{}_loss'.format(key)] = loss
        outs['{}_acc'.format(key)] = acc

    return outs


def run_citation(model, runs, epochs, lr, weight_decay, early_stopping, alpha, beta, tau, batch_size, adj_label,
                 features, labels, idx_train, idx_val,
                 idx_test):
    val_losses, accs, durations = [], [], []


    print(
        f'raw_data_dim:{features.shape[1]}, classes:{torch.max(labels) + 1},',
        f'Train_data_num:{idx_train.shape[0]}, Val_data_num:{idx_val.shape[0]}, Test_data_num:{idx_test.shape[0]}')

    pbar = tqdm(range(runs), unit='run')

    for _ in pbar:

        model.to(device).reset_parameters()
        optimizer = Adam(model.parameters(), lr=lr, weight_decay=weight_decay)

        if torch.cuda.is_available():
            torch.cuda.synchronize()

        t_start = time.perf_counter()

        best_val_loss = float('inf')
        test_acc = 0
        val_loss_history = []

        for epoch in range(1, epochs + 1):
            out = train_citation(alpha, beta, tau, batch_size, adj_label, idx_train, features, labels, model, optimizer)
            eval_info = evaluate_citation(model, features, labels, idx_train, idx_val, idx_test)
            eval_info['epoch'] = epoch

            if eval_info['val_loss'] < best_val_loss:
                best_val_loss = eval_info['val_loss']
                test_acc = eval_info['test_acc']

            val_loss_history.append(eval_info['val_loss'])
            if early_stopping > 0 and epoch > epochs // 2:
                tmp = tensor(val_loss_history[-(early_stopping + 1):-1])
                if eval_info['val_loss'] > tmp.mean().item():
                    break

        if torch.cuda.is_available():
            torch.cuda.synchronize()

        t_end = time.perf_counter()

        val_losses.append(best_val_loss)
        accs.append(test_acc)
        durations.append(t_end - t_start)

    loss, acc, duration = tensor(val_losses), tensor(accs), tensor(durations)

    print('Val Loss: {:.4f}, Test Accuracy: {:.3f} ± {:.3f}, Duration: {:.3f}'.
          format(loss.mean().item(),
                 acc.mean().item(),
                 acc.std().item(),
                 duration.mean().item()))


def run_coauthor_amazon(dataset, model, runs, epochs, lr, weight_decay, early_stopping, alpha, beta, tau, adj_label,
                        permute_masks=None, lcc=False):
    val_losses, accs, durations = [], [], []

    lcc_mask = None
    if lcc:  # select largest connected component
        data_ori = dataset[0]
        data_nx = to_networkx(data_ori)
        data_nx = data_nx.to_undirected()
        print("Original #nodes:", data_nx.number_of_nodes())
        data_nx = data_nx.subgraph(max(nx.connected_components(data_nx), key=len))
        print("#Nodes after lcc:", data_nx.number_of_nodes())
        lcc_mask = list(data_nx.nodes)

    pbar = tqdm(range(runs), unit='run')
    data = dataset[0]

    for _ in pbar:

        if permute_masks is not None:
            data = permute_masks(data, dataset.num_classes, lcc_mask)
        data = data.to(device)
        if _ == 0:
            print(
                f'raw_data_dim:{data.x.shape[1]}, classes:{torch.max(data.y) + 1},',
                f'Train_data_num:{sum(data.train_mask)}, Val_data_num:{sum(data.val_mask)}, Test_data_num:{sum(data.test_mask)}')

        model.to(device).reset_parameters()
        optimizer = Adam(model.parameters(), lr=lr, weight_decay=weight_decay)

        if torch.cuda.is_available():
            torch.cuda.synchronize()

        t_start = time.perf_counter()

        best_val_loss = float('inf')
        test_acc = 0
        val_loss_history = []

        for epoch in range(1, epochs + 1):
            out = train_coauthor_amazon(data, alpha, beta, tau, adj_label, model, optimizer)
            eval_info = evaluate_coauthor_amazon(model, data)
            eval_info['epoch'] = epoch

            if eval_info['val_loss'] < best_val_loss:
                best_val_loss = eval_info['val_loss']
                test_acc = eval_info['test_acc']

            val_loss_history.append(eval_info['val_loss'])
            if early_stopping > 0 and epoch > epochs // 2:
                tmp = tensor(val_loss_history[-(early_stopping + 1):-1])
                if eval_info['val_loss'] > tmp.mean().item():
                    break

        if torch.cuda.is_available():
            torch.cuda.synchronize()

        t_end = time.perf_counter()

        val_losses.append(best_val_loss)
        accs.append(test_acc)
        durations.append(t_end - t_start)

    loss, acc, duration = tensor(val_losses), tensor(accs), tensor(durations)

    print('Val Loss: {:.4f}, Test Accuracy: {:.3f} ± {:.3f}, Duration: {:.3f}'.
          format(loss.mean().item(),
                 acc.mean().item(),
                 acc.std().item(),
                 duration.mean().item()))
