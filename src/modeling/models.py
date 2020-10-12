import torch
import torch.nn as nn
import torch.nn.functional as F
import pretrainedmodels

from src import utils
from .metric_learning import *
import cirtorch

ROOT = '../'


class LandmarkNet(nn.Module):
    DIVIDABLE_BY = 32

    def __init__(self,
                 n_classes,
                 model_name='resnet50',
                 pooling='GeM',
                 args_pooling: dict = {},
                 use_fc=False,
                 fc_dim=512,
                 dropout=0.0,
                 loss_module = "",
                 s=30.0,
                 margin=0.50,
                 ls_eps=0.0,
                 theta_zero=0.785):
        """
        :param n_classes:
        :param model_name: name of model from pretrainedmodels
            e.g. resnet50, resnext101_32x4d, pnasnet5large
        :param pooling: One of ('SPoC', 'MAC', 'RMAC', 'GeM', 'Rpool', 'Flatten', 'CompactBilinearPooling')
        :param loss_module: One of ('arcface', 'cosface', 'softmax')
        """
        super(LandmarkNet, self).__init__()

        self.backbone = getattr(pretrainedmodels, model_name)(num_classes=1000)
        final_in_features = self.backbone.last_linear.in_features
        # HACK: work around for this issue https://github.com/Cadene/pretrained-models.pytorch/issues/120
        self.backbone = nn.Sequential(*list(self.backbone.children())[:-2])
        # TODO:CompactBilinearPooling
        self.pooling = getattr(cirtorch.pooling, pooling)(**args_pooling)

        self.use_fc = use_fc
        if use_fc:
            self.dropout = nn.Dropout(p=dropout)
            self.fc = nn.Linear(final_in_features, fc_dim)
            self.bn = nn.BatchNorm1d(fc_dim)
            self._init_params()
            final_in_features = fc_dim

        self.loss_module = loss_module
        print("Current Loss:{}".format(self.loss_module))
        if loss_module == 'arcface':
            self.final = ArcMarginProduct(final_in_features, n_classes,
                                          s=s, m=margin, easy_margin=False, ls_eps=ls_eps)
        elif loss_module == 'cosface':
            self.final = AddMarginProduct(final_in_features, n_classes, s=s, m=margin)
        elif loss_module == 'adacos':
            self.final = AdaCos(final_in_features, n_classes, m=margin, theta_zero=theta_zero)
        # New by Group
        elif loss_module == "AdditiveMarginSoftmaxLoss":
            self.final = AdMSoftmaxLoss(final_in_features, n_classes, s=s, m=margin)
        elif loss_module == "LSoftmax":
            self.final = LSoftmaxLinear(final_in_features, n_classes, margin=2)
        elif loss_module == "Softmax":
            self.final = nn.Linear(final_in_features, n_classes)
        else:
            raise NotImplementedError("Loss Not Implemented.")

    def _init_params(self):
        nn.init.xavier_normal_(self.fc.weight)
        nn.init.constant_(self.fc.bias, 0)
        nn.init.constant_(self.bn.weight, 1)
        nn.init.constant_(self.bn.bias, 0)

    def forward(self, x, label=None):
        feature = self.extract_feat(x)
        if self.loss_module in ['arcface', 'cosface', 'adacos', 'AdditiveMarginSoftmaxLoss', 'LSoftmax']:
            logits = self.final(feature, label)
        elif self.loss_module in ["Softmax"]:
            logits = self.final(feature)
        else:
            raise NotImplementedError("Loss Not Implemented.")
        return logits

    def extract_feat(self, x):
        batch_size = x.shape[0]
        x = self.backbone(x)
        x = self.pooling(x).view(batch_size, -1)

        if self.use_fc:
            x = self.dropout(x)
            x = self.fc(x)
            x = self.bn(x)

        return x
