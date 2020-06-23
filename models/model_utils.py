'''
A collection of tools used in model building.

Version 1.0  2018-09-25 12:17:32

'''
import math
import torch.nn as nn
from torch.nn import functional as F


def resize(tensor, newsize):
    '''Resize tensor'''
    return F.interpolate(
        tensor, size=newsize, mode='bilinear', align_corners=True
    )


def custom_initialization(model, init_list=None):
    if init_list is None:
        for m in model.modules():
            if isinstance(m, nn.Conv2d):
                # 此处类似 resnet论文中的torch.nn.init.kaiming_normal(), 但又不一样
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
    else:
        for name, child_moudle in model.named_children():
            if name in init_list:
                for name, m in child_moudle.named_modules():
                    if isinstance(m, nn.Conv2d):
                        nn.init.kaiming_normal_(m.weight.data, nonlinearity='relu')
                    if isinstance(m, nn.BatchNorm2d):
                        m.weight.data.fill_(1)
                        m.weight.data.zero_()
