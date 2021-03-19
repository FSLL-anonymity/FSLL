import torch.nn as nn
import torch.nn.functional as F
import torch
import numpy as np
import torchmeta
from torchmeta.modules import (MetaModule, MetaSequential, MetaConv2d,
                               MetaBatchNorm2d, MetaLinear)
from torchmeta.modules.utils import get_subdict
from collections import OrderedDict


class DNN3Word(nn.Module):

    def __init__(self, params):
        super(DNN3Word, self).__init__()
        self.W_m = nn.Sequential(
            nn.Linear(500, 1000),
            torch.nn.Dropout(0.1),
            nn.ReLU(),
            nn.Linear(1000, 1000),
            torch.nn.Dropout(0.1),
            nn.ReLU(),
            nn.Linear(1000, 1000),
            torch.nn.Dropout(0.1),
            nn.ReLU(),
            nn.Linear(1000, params.word2vecDIM), )

    def forward(self, x, word2vec):
        y_hat = self.W_m(x).mm(word2vec.t())
        return y_hat


class DNN2Word_Semantic(nn.Module):

    def __init__(self, params):
        super(DNN2Word_Semantic, self).__init__()
        self.W_m = nn.Sequential(
            nn.Linear(500, 1000),
            torch.nn.Dropout(0.1),
            nn.ReLU(),
            nn.Linear(1000, 1000),
            torch.nn.Dropout(0.1),
            nn.ReLU(),
            nn.Linear(1000, params.word2vecDIM)
        )
        self.W_q = nn.Linear(params.word2vecDIM, params.dw)
        self.W_K = nn.Linear(params.word2vecDIM, params.dw)
        self.Inference = nn.Linear(params.word2vecDIM, params.word2vecDIM)

    def forward(self, x, word2vec_base, word2vec_novel, params):
        z = self.W_m(x)
        p_b = z.mm(word2vec_base.t())
        p_I1 = p_b.mm(word2vec_base.t())
        p_I2 = self.Inference(p_I1)
        p_I = p_I2.mm(word2vec_novel.t())

        q = self.W_q(z)
        K = self.W_K(word2vec_base.t())
        t = np.sqrt(params.dw)
        alpha = (q.mm(K.t())/t).softmax(0)
        s = alpha.mm(word2vec_base.t()) + z
        p_A = s.mm(word2vec_novel.t())
        return p_I + p_A


class DNN2Word_fe(MetaModule):

    def __init__(self, params):
        super(DNN2Word_fe, self).__init__()
        self.W_m =MetaSequential(
            MetaLinear(500, 1000),
            torch.nn.Dropout(0.1),
            nn.ReLU(),
            MetaLinear(1000, 1000),
            torch.nn.Dropout(0.1),
            nn.ReLU(),
            MetaLinear(1000, params.word2vecDIM)
        )

    def forward(self, x, theta=None):
        y_hat = self.W_m(x, params=get_subdict(theta, 'W_m'))
        return y_hat


class Word_Semantic(torchmeta.modules.MetaModule):
    def __init__(self, params, word2vec_train, word2vec_val, word2vec_test):
        super(Word_Semantic, self).__init__()
        self.params = params
        self.word2vec_train = word2vec_train
        self.word2vec_val = word2vec_val
        self.word2vec_test = word2vec_test
        self.max_num_base_label = word2vec_train.shape[0]
        self.W_a = MetaSequential(
            MetaLinear(params.word2vecDIM, params.dw),
            torch.nn.Dropout(0.1),
            nn.ReLU(),
            MetaLinear(params.dw, params.word2vecDIM)
        )
        self.W_i = MetaLinear(self.max_num_base_label, params.word2vecDIM)
        self.W_z = MetaSequential(
            MetaLinear(params.word2vecDIM, params.dw),
            torch.nn.Dropout(0.1),
            nn.ReLU(),
            MetaLinear(params.dw, params.word2vecDIM)
        )
        # self.W_q = MetaLinear(params.word2vecDIM, params.dw)
        self.scale_z = nn.Parameter(torch.FloatTensor(1).fill_(1), requires_grad=True)
        self.scale_a = nn.Parameter(torch.FloatTensor(1).fill_(1), requires_grad=True)
        self.scale_i = nn.Parameter(torch.FloatTensor(1).fill_(1), requires_grad=True)

    def forward(self, phase, z, base_label_IDlist, novel_label_IDlist, theta=None):
        if phase == 'train':
            word2vec_base = self.word2vec_train[base_label_IDlist].to(self.params.device)
            word2vec_novel = self.word2vec_train[novel_label_IDlist].to(self.params.device)
        elif phase == 'val':
            word2vec_base = self.word2vec_train.to(self.params.device)
            word2vec_novel = self.word2vec_val[novel_label_IDlist].to(self.params.device)  # [batch_size x N x dim]
        elif phase == 'test':
            word2vec_base = self.word2vec_train.to(self.params.device)
            word2vec_novel = self.word2vec_test[novel_label_IDlist].to(self.params.device)  # [batch_size x N x dim]

        # q = self.W_q(z, params=get_subdict(theta, 'W_q'))
        # Attention Module
        alpha = (z.mm(word2vec_base.t())*4).sigmoid()
        attention = 1 / alpha.sum() * alpha.mm(word2vec_base)
        attention = F.normalize(attention, p=2, dim=attention.dim()-1, eps=1e-12)
        attention = self.W_a(attention, params=get_subdict(theta, 'W_a'))

        # Inference Module.
        p_i = z.mm(word2vec_base.t())
        p_i = F.normalize(p_i, p=2, dim=p_i.dim()-1, eps=1e-12)

        if theta is None:
            W_i_theta = OrderedDict(self.W_i.named_parameters())
        else:
            W_i_theta = get_subdict(theta, 'W_i')
        W_i_bias = W_i_theta.get('bias', None)
        W_i_weight = W_i_theta['weight'][:, base_label_IDlist]
        p_i = F.linear(p_i, W_i_weight, W_i_bias)

        # W_z.
        z = self.W_z(z, params=get_subdict(theta, 'W_z'))
        z = F.normalize(z, p=2, dim=z.dim()-1, eps=1e-12)
        z = z.mm(word2vec_novel.t())

        scale_z = self.scale_z
        scale_i = self.scale_i
        scale_a = self.scale_a

        p_i = p_i.mm(word2vec_novel.t())

        word2vec_novel = F.normalize(word2vec_novel, p=2, dim=word2vec_novel.dim() - 1, eps=1e-12)
        attention = attention.mm(word2vec_novel.t())
        output = z * scale_z + p_i * scale_i + scale_a * attention

        return output


class MetaMLPModel(MetaModule):

    def __init__(self, in_features, out_features, hidden_sizes):
        super(MetaMLPModel, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.hidden_sizes = hidden_sizes

        layer_sizes = [in_features] + hidden_sizes
        self.features = MetaSequential(OrderedDict([('layer{0}'.format(i + 1),
            MetaSequential(OrderedDict([
                ('linear', MetaLinear(hidden_size, layer_sizes[i + 1], bias=True)),
                ('relu', nn.ReLU())
            ]))) for (i, hidden_size) in enumerate(layer_sizes[:-1])]))
        self.classifier = MetaLinear(hidden_sizes[-1], out_features, bias=True)

    def forward(self, inputs, theta=None):
        features = self.features(inputs, params=get_subdict(theta, 'features'))
        logits = self.classifier(features, params=get_subdict(theta, 'classifier'))
        return logits


class Threshold(MetaModule):
    def __init__(self, params):
        super(Threshold, self).__init__()
        self.params = params
        self.threshold = MetaSequential(
            MetaLinear(params.word2vecDIM, 100),
            nn.ReLU(),
            MetaLinear(100, 30),
            nn.ReLU(),
            MetaLinear(30, params.N)
        )
        self.threshold_constant = nn.Parameter(torch.FloatTensor(1).fill_(-1), requires_grad=True)

    def forward(self, z, theta=None):
        z = F.normalize(z, p=2, dim=z.dim() - 1, eps=1e-12)
        threshold = self.threshold(z, params=get_subdict(theta, 'threshold')) + self.threshold_constant

        return threshold
