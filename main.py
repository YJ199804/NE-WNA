import argparse
from train import *
from model import NE_WNA
from utils import *
from datasets import *
import warnings

parser = argparse.ArgumentParser()
parser.add_argument('--dataset', type=str, required=True)
parser.add_argument('--random_splits', type=bool, default=False)
parser.add_argument('--runs', type=int, default=100)
parser.add_argument('--epochs', type=int, default=1000)
parser.add_argument('--batch_size', type=int, default=2000)
parser.add_argument('--lr', type=float, default=0.01)
parser.add_argument('--weight_decay', type=float, default=0.005)
parser.add_argument('--early_stopping', type=int, default=100)
parser.add_argument('--hidden', type=int, default=64)
parser.add_argument('--hidden_z', type=int, default=64)
parser.add_argument('--dropout', type=float, default=0.5)
parser.add_argument('--normalize_features', type=bool, default=False)
parser.add_argument('--order', type=int, default=4)
parser.add_argument('--alpha', type=float, default=1)
parser.add_argument('--beta', type=float, default=0.5)
parser.add_argument('--tau', type=float, default=0.5)

args = parser.parse_args()

warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=RuntimeWarning)

if args.dataset == "cora" or args.dataset == "citeseer" or args.dataset == "pubmed":
    adj, features, labels, idx_train, idx_val, idx_test = load_citation(args.dataset, True)
    print("Precompute the enhanced adjacency matrix...")
    adj = normalize_adj(adj + sp.eye(adj.shape[0]))
    sp_A_hat = sparse_mx_to_torch_sparse_tensor(adj).float().cuda()
    if args.dataset == "pubmed":
        enhanced_adj = get_A_hat_k_power(sp_A_hat, args.order)
        print("Precompute Finish!")
        model = NE_WNA(args.hidden, args.hidden_z, features.shape[1], labels.max().item() + 1, args.dropout)
        print(f'Dataset:{args.dataset}')
        run_citation(model, args.runs, args.epochs, args.lr, args.weight_decay, args.early_stopping, args.alpha,
                     args.beta,
                     args.tau, args.batch_size, enhanced_adj.to_dense(),
                     features, labels, idx_train, idx_val,
                     idx_test)
    else:
        enhanced_adj = get_ehanced_A_hat_k_power(sp_A_hat, args.order)
        print("Precompute Finish!")
        model = NE_WNA(args.hidden, args.hidden_z, features.shape[1], labels.max().item() + 1, args.dropout)
        print(f'Dataset:{args.dataset}')
        run_citation(model, args.runs, args.epochs, args.lr, args.weight_decay, args.early_stopping, args.alpha,
                     args.beta,
                     args.tau, args.batch_size, enhanced_adj.to_dense(),
                     features, labels, idx_train, idx_val,
                     idx_test)

elif args.dataset == "computers" or args.dataset == "photo":
    dataset = get_amazon_dataset(args.dataset, args.normalize_features)
    permute_masks = random_coauthor_amazon_splits
    print("Precompute the enhanced adjacency matrix...")
    A = get_adj_ori(dataset[0])
    A_hat = normalize_adj(A + sp.eye(A.shape[0]))
    sp_A_hat = sparse_mx_to_torch_sparse_tensor(A_hat).float()
    enhanced_adj = get_ehanced_A_hat_k_power(sp_A_hat, args.order)
    print("Precompute Finish!")
    print(f'Dataset:{args.dataset}')
    print("Dataset:", dataset[0])
    model = NE_WNA(args.hidden, args.hidden_z, dataset.num_features, dataset.num_classes, args.dropout)
    run_coauthor_amazon(dataset, model, args.runs, args.epochs, args.lr, args.weight_decay, args.early_stopping,
                        args.alpha, args.beta, args.tau, enhanced_adj.to_dense().to(device), permute_masks, lcc=True)

elif args.dataset == "cs":
    dataset = get_coauthor_dataset(args.dataset, args.normalize_features)
    permute_masks = random_coauthor_amazon_splits
    A = get_adj_ori(dataset[0])
    A_hat = normalize_adj(A + sp.eye(A.shape[0]))
    sp_A_hat = sparse_mx_to_torch_sparse_tensor(A_hat).float()
    enhanced_adj = get_ehanced_A_hat_k_power(sp_A_hat, args.order)
    print("Dataset:", dataset[0])
    model = NE_WNA(args.hidden, args.hidden_z, dataset.num_features, dataset.num_classes, args.dropout)
    run_coauthor_amazon(dataset, model, args.runs, args.epochs, args.lr, args.weight_decay, args.early_stopping,
                        args.alpha, args.beta, args.tau, enhanced_adj.to_dense().to(device), permute_masks, lcc=False)
