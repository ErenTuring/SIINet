# -*- coding: utf-8 -*-
import torch
from torch import nn
from torch.nn import functional as F
import time
from torchvision import models
# import math


class BasicModule(nn.Module):
    """
    封装了nn.Module,主要是提供了save和load两个方法
    """

    def __init__(self):
        super(BasicModule, self).__init__()
        self.model_name = str(type(self))  # 默认名字

    def load(self, path):
        """
        可加载指定路径的模型
        """
        # TODO: 把resnet.py里实现的残缺ckpt加载，融合进来
        self.load_state_dict(torch.load(path))

    def save(self, name=None):
        """
        保存模型，默认使用“模型名字+时间”作为文件名
        """
        if name is None:
            prefix = 'checkpoints/' + self.model_name + '_'
            name = time.strftime(prefix + '%m%d_%H:%M:%S.pth')
        torch.save(self.state_dict(), name)
        return name


class ResNet(BasicModule):
    ''' Use BasicModule package orginal pytorch resnet34 '''
    def __init__(self, pretrained=True, num_classes=1000, resnet_arch='resnet50'):
        super().__init__()
        if resnet_arch == 'resnet18':
            self.model = models.resnet18(pretrained, num_classes=num_classes)
        elif resnet_arch == 'resnet34':
            self.model = models.resnet34(pretrained, num_classes=num_classes)
        elif resnet_arch == 'resnet50':
            self.model = models.resnet50(pretrained, num_classes=num_classes)
        elif resnet_arch == 'resnet101':
            self.model = models.resnet101(pretrained, num_classes=num_classes)
        elif resnet_arch == 'resnet152':
            self.model = models.resnet152(pretrained, num_classes=num_classes)

    def forward(self, x):
        x = self.model(x)
        return x


def set_parameter_requires_grad(model, feature_extracting):
    ''' If we are feature extracting and only want to compute gradients
    for the newly initialized layer then we want all of the other parameters
    to not require gradients.  '''
    if feature_extracting:
        for param in model.parameters():
            param.requires_grad = False
