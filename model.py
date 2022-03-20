import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from tqdm import tqdm
from torch.nn import Dropout, Linear
from torch.optim import Adam
from utils import get_feature_dis


class AE(nn.Module):

    def __init__(self, n_hidden, n_input, n_z, dropout):
        super(AE, self).__init__()
        self.dropout = dropout
        self.enc_1 = Linear(n_input, n_hidden)
        self.z_layer = Linear(n_hidden, n_z)
        self.dec_1 = Linear(n_z, n_hidden)
        self.x_bar_layer = Linear(n_hidden, n_input)

        self._init_weights()

    def _init_weights(self):
        nn.init.xavier_uniform_(self.enc_1.weight)
        nn.init.xavier_uniform_(self.z_layer.weight)
        nn.init.xavier_uniform_(self.dec_1.weight)
        nn.init.xavier_uniform_(self.x_bar_layer.weight)
        nn.init.normal_(self.enc_1.bias, std=1e-6)
        nn.init.normal_(self.z_layer.bias, std=1e-6)
        nn.init.normal_(self.dec_1.bias, std=1e-6)
        nn.init.normal_(self.x_bar_layer.bias, std=1e-6)

    def reset_parameters(self):
        self.enc_1.reset_parameters()
        self.z_layer.reset_parameters()
        self.dec_1.reset_parameters()
        self.x_bar_layer.reset_parameters()


    def forward(self, x):
        enc_h1 = F.relu(self.enc_1(x))
        enc_h1 = F.dropout(enc_h1, p=self.dropout, training=self.training)

        z = self.z_layer(enc_h1)
        z_drop = F.dropout(z, p=self.dropout, training=self.training)

        dec_h1 = F.relu(self.dec_1(z_drop))
        dec_h1 = F.dropout(dec_h1, p=self.dropout, training=self.training)

        x_bar = self.x_bar_layer(dec_h1)

        return x_bar, z


class AE_ENC(nn.Module):
    def __init__(self, nhid, n_z, nfeat, nclass, dropout):
        super(AE_ENC, self).__init__()
        self.AE = AE(nhid, nfeat, n_z, dropout)
        self.classifier = Linear(n_z, nclass)

    def reset_parameters(self):
        self.AE.reset_parameters()
        self.classifier.reset_parameters()

    def forward(self, x):
        x_bar, x = self.AE(x)

        feature_cls = x
        Z = x
        x_dis = 0

        if self.training:
            x_dis = get_feature_dis(Z)

        class_feature = self.classifier(feature_cls)
        class_logits = F.log_softmax(class_feature, dim=1)

        if self.training:
            return x_bar, x_dis, class_logits
        else:
            return class_logits