# -*- coding:utf-8 -*-
'''
Contract Nets with SIIS.
'''
import torch
from torch import nn
from models import resnet
from .BasicModule import BasicModule
from .model_utils import resize
from .SIIS_Kernel import SIIS


def conv3x3(in_planes, out_planes, stride=1):
    """ 3x3 convolution with padding """
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=1, bias=False)


class ResUnetBlock(nn.Module):
    ''' Residual block of ResUnet'''
    def __init__(self, inplanes, planes, stride=1, residual=True):
        super(ResUnetBlock, self).__init__()
        self.bn1 = nn.BatchNorm2d(inplanes)
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv2 = conv3x3(planes, planes)
        self.relu = nn.ReLU(inplace=True)
        # self.stride = stride
        if inplanes != planes or stride != 1:
            self.downsample = nn.Sequential(
                nn.Conv2d(inplanes, planes, 1, stride, bias=False),
                nn.BatchNorm2d(planes)
            )
        self.residual = residual

    def forward(self, x):

        out = self.bn1(x)
        out = self.relu(out)
        out = self.conv1(out)

        out = self.bn2(out)
        out = self.relu(out)
        out = self.conv2(out)

        if self.residual:
            residual = x
            if self.downsample is not None:
                residual = self.downsample(x)
            out += residual

        return out


class Resnet_seg(BasicModule):
    ''' Resnet for samtic segmentation '''
    def __init__(self, num_classes=2,
                 resnet_arch='resnet50', output_stride=8, layer_num=2):
        super(Resnet_seg, self).__init__()
        self.output_stride = output_stride
        self.layer_num = layer_num
        # 注意用的是50还是101
        if resnet_arch == 'resnet50':
            encoder = resnet.resnet50(True, output_stride=self.output_stride)
        elif resnet_arch == 'resnet101':
            encoder = resnet.resnet101(True, output_stride=self.output_stride)
        encoder = encoder._modules  # Covert class instance into orderdict

        # Encoder
        self.conv1 = nn.Sequential(encoder['conv1'], encoder['bn1'], encoder['relu'])
        self.pool1 = encoder['maxpool']  # s/4 - 64dim

        self.layers = nn.Sequential()
        for i in range(layer_num):
            self.layers.add_module('layer%d' % (i+1), encoder['layer%d' % (i+1)])
        layers_dim = [256, 512, 1024, 2048, 2048, 1024, 512]

        # Decoder
        self.decoder_conv1 = self._make_layer(layers_dim, 256, 3, padding=1)  # siis_size
        self.decoder_conv2 = self._make_layer(256, 256, 3, padding=1)  # s/4
        self.out_conv = nn.Conv2d(256, num_classes, 1, 1)  # s/1 - output

        self.model_name = resnet_arch + '_seg'

    def _make_layer(self, in_channel, out_channel, kernel_size, padding=0):
        return nn.Sequential(
            nn.Conv2d(in_channel, out_channel, kernel_size, 1, padding=padding),
            nn.BatchNorm2d(out_channel),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        size = x.shape[2:]  # need interpolate input size
        x = self.conv1(x)
        x = self.pool1(x)
        size_4 = x.shape[2:]

        x = self.layers(x)  # s/output_stride

        # Decoder
        x = self.decoder_conv1(x)  # s/output_stride
        x = resize(x, newsize=size_4)  # Upx? -> s/4
        x = self.decoder_conv2(x)  # s/4

        x = resize(x, newsize=size)  # Upx4 -> s/1
        x = self.out_conv(x)  # s/1

        return x


class ResUnet(BasicModule):
    def __init__(self, num_classes=2):
        super(ResUnet, self).__init__()
        self.model_name = 'res_unet'
        # Encoder
        self.layer1 = nn.Sequential(
            conv3x3(3, 64),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            conv3x3(64, 64)
        )  # s/1, dim=64, Addtion
        self.layer1_shortcut = nn.Conv2d(3, 64, 1, 1, bias=False)

        self.layer2 = ResUnetBlock(64, 128, 2)
        self.layer3 = ResUnetBlock(128, 256, 2)

        # Bridge
        self.layer4 = ResUnetBlock(256, 512, 2, False)

        # Deocder
        self.layer5 = ResUnetBlock(512+256, 256)
        self.layer6 = ResUnetBlock(256+128, 128)
        self.layer7 = ResUnetBlock(128+64, 64)

        self.out_conv = nn.Conv2d(64, num_classes, 1, 1)

        # Initalization
        # for m in self.modules():
        #     if isinstance(m, nn.Conv2d):
        #         # n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
        #         # m.weight.data.normal_(0, math.sqrt(2. / n))
        #         nn.init.kaiming_normal_(m.weight.data, nonlinearity='relu')
        #     elif isinstance(m, nn.BatchNorm2d):
        #         m.weight.data.fill_(1)
        #         m.bias.data.zero_()

    def forward(self, x):
        size = x.shape[2:]
        # Encoder
        residual = self.layer1_shortcut(x)
        x = self.layer1(x)
        layer1 = x + residual  # s/1, dim=64

        layer2 = self.layer2(layer1)  # s/2, dim=128
        size_2 = layer2.shape[2:]
        layer3 = self.layer3(layer2)  # s/4, dim=256
        size_4 = layer3.shape[2:]

        # Bridge
        layer4 = self.layer4(layer3)  # s/8, dim=512
        layer4 = resize(layer4, size_4)  # Upx2 -> s/4

        # Decoder
        layer4 = torch.cat([layer4, layer3], dim=1)  # s/4, dim=512+256
        layer5 = self.layer5(layer4)  # s/4, dim=256
        layer5 = resize(layer5, size_2)  # Upx2 -> s/2
        layer5 = torch.cat([layer5, layer2], dim=1)  # s/2, dim=256+128

        layer6 = self.layer6(layer5)  # s/2, dim=128
        layer6 = resize(layer6, size)  # Upx2 -> s/1
        layer6 = torch.cat([layer6, layer1], dim=1)  # s/1, dim=128+64

        layer7 = self.layer7(layer6)  # s/1, dim=64

        x = self.out_conv(layer7)
        return x


class ResUnet_SIIS(BasicModule):
    def __init__(self, num_classes=2,
                 siis_size=[32, 32], width=1, kw=9, dim=128, arch=1):
        super(ResUnet_SIIS, self).__init__()
        self.model_name = 'res_unet'
        # Encoder
        self.layer1 = nn.Sequential(
            conv3x3(3, 64),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            conv3x3(64, 64)
        )  # s/1, dim=64, Addtion
        self.layer1_shortcut = nn.Conv2d(3, 64, 1, 1, bias=False)

        self.layer2 = ResUnetBlock(64, 128, 2)
        self.layer3 = ResUnetBlock(128, 256, 2)

        # Bridge
        self.layer4 = ResUnetBlock(256, 512, 2, False)
        self.siis = SIIS(siis_size, width, kw, dim, arch)  # size=siis_size, dim=dim

        # Deocder
        self.layer5 = ResUnetBlock(512+256, 256)
        self.layer6 = ResUnetBlock(256+128, 128)
        self.layer7 = ResUnetBlock(128+64, 64)

        self.out_conv = nn.Conv2d(64, num_classes, 1, 1)

        # Initalization
        # for m in self.modules():
        #     if isinstance(m, nn.Conv2d):
        #         # n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
        #         # m.weight.data.normal_(0, math.sqrt(2. / n))
        #         nn.init.kaiming_normal_(m.weight.data, nonlinearity='relu')
        #     elif isinstance(m, nn.BatchNorm2d):
        #         m.weight.data.fill_(1)
        #         m.bias.data.zero_()

    def forward(self, x):
        size = x.shape[2:]
        # Encoder
        residual = self.layer1_shortcut(x)
        x = self.layer1(x)
        layer1 = x + residual  # s/1, dim=64

        layer2 = self.layer2(layer1)  # s/2, dim=128
        size_2 = layer2.shape[2:]
        layer3 = self.layer3(layer2)  # s/4, dim=256
        size_4 = layer3.shape[2:]

        # Bridge
        layer4 = self.layer4(layer3)  # s/8, dim=512
        layer4 = self.siis(layer4)  # SIIS
        layer4 = resize(layer4, size_4)  # Upx2 -> s/4

        # Decoder
        layer4 = torch.cat([layer4, layer3], dim=1)  # s/4, dim=512+256
        layer5 = self.layer5(layer4)  # s/4, dim=256
        layer5 = resize(layer5, size_2)  # Upx2 -> s/2
        layer5 = torch.cat([layer5, layer2], dim=1)  # s/2, dim=256+128

        layer6 = self.layer6(layer5)  # s/2, dim=128
        layer6 = resize(layer6, size)  # Upx2 -> s/1
        layer6 = torch.cat([layer6, layer1], dim=1)  # s/1, dim=128+64

        layer7 = self.layer7(layer6)  # s/1, dim=64

        x = self.out_conv(layer7)
        return x
