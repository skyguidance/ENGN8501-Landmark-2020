import pretrainedmodels
import extra.cirtorch as cirtorch

from .metric_learning import *


class AttentionModel(nn.Module):

    def __init__(self, num_features_in, feature_size=256):
        super(AttentionModel, self).__init__()

        self.conv1 = nn.Conv2d(num_features_in, feature_size, kernel_size=3, padding=1)
        self.act1 = nn.ReLU()

        self.conv2 = nn.Conv2d(feature_size, feature_size, kernel_size=3, padding=1)
        self.act2 = nn.ReLU()

        self.conv3 = nn.Conv2d(feature_size, feature_size, kernel_size=3, padding=1)
        self.act3 = nn.ReLU()

        self.conv4 = nn.Conv2d(feature_size, feature_size, kernel_size=3, padding=1)
        self.act4 = nn.ReLU()

        self.conv5 = nn.Conv2d(feature_size, 1, kernel_size=3, padding=1)

        self.output_act = nn.Sigmoid()

    def forward(self, x):
        out = self.conv1(x)
        out = self.act1(out)

        out = self.conv2(out)
        out = self.act2(out)

        out = self.conv3(out)
        out = self.act3(out)

        out = self.conv4(out)
        out = self.act4(out)

        out = self.conv5(out)
        out_attention = self.output_act(out)

        return out_attention


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
                 loss_module="",
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

        self.layer1 = nn.Sequential(*list(self.backbone.children())[:5])
        self.layer2 = nn.Sequential(*list(self.backbone.children())[5:6])
        self.layer3 = nn.Sequential(*list(self.backbone.children())[6:7])
        self.layer4 = nn.Sequential(*list(self.backbone.children())[7:8])
        self.attention1 = AttentionModel(256)
        self.attention2 = AttentionModel(512)
        self.attention3 = AttentionModel(1024)

        self.use_fc = use_fc
        if use_fc:
            self.dropout = nn.Dropout(p=dropout)
            self.fc = nn.Linear(final_in_features, fc_dim)
            self.bn = nn.BatchNorm1d(fc_dim)
            self._init_params()
            final_in_features = fc_dim

        self.loss_module = loss_module
        if loss_module == 'arcface':
            # Arcface Loss function
            # Reference: Implement the arcface loss function from
            # https://github.com/ronghuaiyang/arcface-pytorch/blob/master/models/metrics.py
            self.final = ArcMarginProduct(final_in_features, n_classes,
                                          s=s, m=margin, easy_margin=False, ls_eps=ls_eps)
        elif loss_module == 'cosface':
            # Cosface loss function
            # Reference: Implement cosface loss function from
            # https://github.com/lyakaap/Landmark2019-1st-and-3rd-Place-Solution/blob/master/src/modeling/metric_learning.py
            self.final = AddMarginProduct(final_in_features, n_classes, s=s, m=margin)
        # New by Group
        elif loss_module == "AdditiveMarginSoftmaxLoss":
            # Additive Margin Softmax Loss
            # Reference: The Additive Margin Softmax Loss was implemented from
            # https://github.com/Leethony/Additive-Margin-Softmax-Loss-Pytorch
            self.final = AdMSoftmaxLoss(final_in_features, n_classes, s=s, m=margin)
        elif loss_module == "LSoftmax":
            # Large-margin Softmax(L-Softmax)
            # Reference: The Large-margin Softmax(L-Softmax) was implemented from
            # https://github.com/amirhfarzaneh/lsoftmax-pytorch/blob/master/lsoftmax.py
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

        x = self.layer1(x)
        att = self.attention1(x)
        x = x * torch.exp(att)
        x = self.layer2(x)
        att = self.attention2(x)
        x = x * torch.exp(att)
        x = self.layer3(x)
        att = self.attention3(x)
        x = x * torch.exp(att)
        x = self.layer4(x)

        x = self.pooling(x).view(batch_size, -1)

        if self.use_fc:
            x = self.dropout(x)
            x = self.fc(x)
            x = self.bn(x)

        return x

    # Use this extract_feat when training or inferencing without attention!!!
    # def extract_feat(self, x):
    #     batch_size = x.shape[0]
    #     x = self.backbone(x)
    #     x = self.pooling(x).view(batch_size, -1)
    #
    #     if self.use_fc:
    #         x = self.dropout(x)
    #         x = self.fc(x)
    #         x = self.bn(x)
    #
    #     return x