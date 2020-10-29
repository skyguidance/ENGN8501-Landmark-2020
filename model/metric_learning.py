from __future__ import print_function
from __future__ import division
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.autograd
from torch.nn import Parameter
import math
from scipy.special import binom


class AdMSoftmaxLoss(nn.Module):
    # Additive Margin Softmax Loss
    # Reference: The Additive Margin Softmax Loss was implemented from
    # https://github.com/Leethony/Additive-Margin-Softmax-Loss-Pytorch
    def __init__(self, in_features, out_features, s=30.0, m=0.4):
        '''
        AM Softmax Loss
        '''
        super(AdMSoftmaxLoss, self).__init__()
        self.s = s
        self.m = m
        self.in_features = in_features
        self.out_features = out_features
        self.fc = nn.Linear(in_features, out_features, bias=False)

    def forward(self, x, labels):
        '''
        input shape (N, in_features)
        '''
        assert len(x) == len(labels)
        assert torch.min(labels) >= 0
        assert torch.max(labels) < self.out_features

        if self.training == False:
            for W in self.fc.parameters():
                W = F.normalize(W, dim=1)

            x = F.normalize(x, dim=1)

            wf = self.fc(x)
            return wf

        for W in self.fc.parameters():
            W = F.normalize(W, dim=1)

        x = F.normalize(x, dim=1)

        wf = self.fc(x)
        numerator = self.s * (torch.diagonal(wf.transpose(0, 1)[labels]) - self.m)
        excl = torch.cat([torch.cat((wf[i, :y], wf[i, y + 1:])).cpu().unsqueeze(0) for i, y in enumerate(labels)],
                         dim=0)
        denominator = torch.exp(numerator.cpu()) + torch.sum(torch.exp(self.s * excl), dim=1)
        L = numerator.cpu() - torch.log(denominator)
        return wf, -torch.mean(L.cuda())


class LSoftmaxLinear(nn.Module):
    # Large-margin Softmax(L-Softmax)
    # Reference: The Large-margin Softmax(L-Softmax) was implemented from
    # https://github.com/amirhfarzaneh/lsoftmax-pytorch/blob/master/lsoftmax.py

    def __init__(self, input_features, output_features, margin):
        super().__init__()
        self.input_dim = input_features  # number of input feature i.e. output of the last fc layer
        self.output_dim = output_features  # number of output = class numbers
        self.margin = margin  # m
        self.beta = 100
        self.beta_min = 0
        self.scale = 0.99

        # self.device = device  # gpu or cpu

        # Initialize L-Softmax parameters
        self.weight = nn.Parameter(torch.FloatTensor(input_features, output_features))
        self.divisor = math.pi / self.margin  # pi/m
        self.C_m_2n = torch.Tensor(binom(margin, range(0, margin + 1, 2))).to(device='cuda')  # C_m{2n}
        self.cos_powers = torch.Tensor(range(self.margin, -1, -2)).to(device='cuda')  # m - 2n
        self.sin2_powers = torch.Tensor(range(len(self.cos_powers))).to(device='cuda')  # n
        self.signs = torch.ones(margin // 2 + 1).to(device='cuda')  # 1, -1, 1, -1, ...
        self.signs[1::2] = -1

    def calculate_cos_m_theta(self, cos_theta):
        sin2_theta = 1 - cos_theta ** 2
        cos_terms = cos_theta.cpu().unsqueeze(1) ** self.cos_powers.cpu().unsqueeze(0)  # cos^{m - 2n}
        sin2_terms = (sin2_theta.cpu().unsqueeze(1)  # sin2^{n}
                      ** self.sin2_powers.cpu().unsqueeze(0))

        cos_m_theta = (self.signs.cpu().unsqueeze(0) *  # -1^{n} * C_m{2n} * cos^{m - 2n} * sin2^{n}
                       self.C_m_2n.cpu().unsqueeze(0) *
                       cos_terms *
                       sin2_terms).sum(1)  # summation of all terms

        return cos_m_theta.cuda()

    def reset_parameters(self):
        nn.init.kaiming_normal_(self.weight.data.t())

    def find_k(self, cos):
        # to account for acos numerical errors
        eps = 1e-7
        cos = torch.clamp(cos, -1 + eps, 1 - eps)
        acos = cos.acos()
        k = (acos / self.divisor).floor().detach()
        return k

    def forward(self, input, target=None):
        if self.training == False:
            return input.mm(self.weight)
        x, w = input, self.weight
        beta = max(self.beta, self.beta_min)
        logit = x.mm(w)
        indexes = range(logit.size(0))
        logit_target = logit[indexes, target]

        # cos(theta) = w * x / ||w||*||x||
        w_target_norm = w[:, target].norm(p=2, dim=0)
        x_norm = x.norm(p=2, dim=1)
        cos_theta_target = logit_target / (w_target_norm * x_norm + 1e-10)

        # equation 7
        cos_m_theta_target = self.calculate_cos_m_theta(cos_theta_target)

        # find k in equation 6
        k = self.find_k(cos_theta_target)

        # f_y_i
        logit_target_updated = (w_target_norm *
                                x_norm *
                                (((-1) ** k * cos_m_theta_target) - 2 * k))
        logit_target_updated_beta = (logit_target_updated + beta * logit[indexes, target]) / (1 + beta)

        logit[indexes, target] = logit_target_updated_beta
        self.beta *= self.scale
        return logit

class ArcMarginProduct(nn.Module):
    # Arcface Loss function
    # Reference: Implement the arcface loss function from
    # https://github.com/ronghuaiyang/arcface-pytorch/blob/master/models/metrics.py
    r"""Implement of large margin arc distance: :
        Args:
            in_features: size of each input sample
            out_features: size of each output sample
            s: norm of input feature
            m: margin
            cos(theta + m)
        """

    def __init__(self, in_features, out_features, s=30.0, m=0.50, easy_margin=False, ls_eps=0.0):
        super(ArcMarginProduct, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.s = s
        self.m = m
        self.ls_eps = ls_eps  # label smoothing
        self.weight = Parameter(torch.FloatTensor(out_features, in_features))
        nn.init.xavier_uniform_(self.weight)

        self.easy_margin = easy_margin
        self.cos_m = math.cos(m)
        self.sin_m = math.sin(m)
        self.th = math.cos(math.pi - m)
        self.mm = math.sin(math.pi - m) * m

    def forward(self, input, label):
        if self.training == False:
            cosine = F.linear(F.normalize(input), F.normalize(self.weight))
            return cosine
        # --------------------------- cos(theta) & phi(theta) ---------------------------
        cosine = F.linear(F.normalize(input), F.normalize(self.weight))
        sine = torch.sqrt(1.0 - torch.pow(cosine, 2))
        phi = cosine * self.cos_m - sine * self.sin_m
        if self.easy_margin:
            phi = torch.where(cosine > 0, phi, cosine)
        else:
            phi = torch.where(cosine > self.th, phi, cosine - self.mm)
        # --------------------------- convert label to one-hot ---------------------------
        # one_hot = torch.zeros(cosine.size(), requires_grad=True, device='cuda')
        one_hot = torch.zeros(cosine.size(), device='cuda')
        one_hot.scatter_(1, label.view(-1, 1).long(), 1)
        if self.ls_eps > 0:
            one_hot = (1 - self.ls_eps) * one_hot + self.ls_eps / self.out_features
        # -------------torch.where(out_i = {x_i if condition_i else y_i) -------------
        output = (one_hot * phi) + ((1.0 - one_hot) * cosine)
        output *= self.s

        return output


class AddMarginProduct(nn.Module):
    # Cosface loss function
    # Reference: Implement cosface loss function from
    # https://github.com/lyakaap/Landmark2019-1st-and-3rd-Place-Solution/blob/master/src/modeling/metric_learning.py
    r"""Implement of large margin cosine distance: :
    Args:
        in_features: size of each input sample
        out_features: size of each output sample
        s: norm of input feature
        m: margin
        cos(theta) - m
    """

    def __init__(self, in_features, out_features, s=30.0, m=0.40):
        super(AddMarginProduct, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.s = s
        self.m = m
        self.weight = Parameter(torch.FloatTensor(out_features, in_features))
        nn.init.xavier_uniform_(self.weight)

    def forward(self, input, label):
        if self.training == False:
            return F.linear(F.normalize(input), F.normalize(self.weight))
        # --------------------------- cos(theta) & phi(theta) ---------------------------
        cosine = F.linear(F.normalize(input), F.normalize(self.weight))
        phi = cosine - self.m
        # --------------------------- convert label to one-hot ---------------------------
        one_hot = torch.zeros(cosine.size(), device='cuda')
        # one_hot = one_hot.cuda() if cosine.is_cuda else one_hot
        one_hot.scatter_(1, label.view(-1, 1).long(), 1)
        # -------------torch.where(out_i = {x_i if condition_i else y_i) -------------
        output = (one_hot * phi) + (
                (1.0 - one_hot) * cosine)  # you can use torch.where if your torch.__version__ is 0.4
        output *= self.s
        # print(output)

        return output

    def __repr__(self):
        return self.__class__.__name__ + '(' \
               + 'in_features=' + str(self.in_features) \
               + ', out_features=' + str(self.out_features) \
               + ', s=' + str(self.s) \
               + ', m=' + str(self.m) + ')'