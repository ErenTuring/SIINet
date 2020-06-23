'''
main_model:
    Vgg：
    Resnet：
    DeepLab：

'''

import torch
from torch import nn
# from torch.nn import functional as F
# from torchvision.models import resnet
from models import resnet
from .model_utils import resize
from .DeeplabV3_plus import ASPP
from .BasicModule import BasicModule
from .SIIS_Kernel import SIIS


class Vgg_SIIS(BasicModule):
    """ Main module：Vgg_SIIS """
    def __init__(self, num_classes=2,
                 siis_size=[32, 32], width=1, kw=9, dim=128, arch=1):
        super(Vgg_SIIS, self).__init__()
        from torchvision.models import vgg

        self.siis_size = siis_size
        # Encoder
        features = vgg.vgg16_bn(pretrained=True).features
        # self.layer1 = features[0:7]  # s/1 - dim=64
        # self.pool_1 = features[6]  # s/2
        # self.layer2 = features[7:14]  # s/2 - dim=128
        # self.pool_2 = features[13]  # s/4
        # self.layer3 = features[14:24]  # s/4 - dim=256
        # self.pool_3 = features[23]  # s/8
        # self.layer4 = features[24:33]  # s/8 - dim=512
        # self.pool_4 = features[33]  # s/16

        self.layers = features[:33]  # s/8, dim=512

        # conv_s - Make sure SIIS input dim. in: layers(out), out: siis(in).
        self.conv_s = self._make_layer(512, dim, 1)
        # SIIS
        self.siis = SIIS(siis_size, width, kw, dim, arch)  # in: conv_s(out)

        # Decoder
        self.decoder_conv1 = self._make_layer(dim, dim, 3, padding=1)  # s/siis_size
        self.decoder_conv2 = self._make_layer(dim, dim, 3, padding=1)  # s/4
        self.out_conv = nn.Conv2d(dim, num_classes, 1, 1)  # s/1 - output

        self.model_name = 'vgg_' + self.siis.name

        # Initalization
        # init_list = ['conv_s', 'decoder_conv1', 'decoder_conv2', 'out_conv']
        # model_utils.custom_initialization(self, init_list)

    def _make_layer(self, in_channel, out_channel, kernel_size, padding=0):
        return nn.Sequential(
            nn.Conv2d(in_channel, out_channel, kernel_size, 1, padding=padding),
            nn.BatchNorm2d(out_channel),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        size = x.size()[2:]
        size_4 = [s//4 for s in size]
        x = self.layers(x)  # s/8

        x = self.conv_s(x)  # 512 -> dim
        # SIIS
        if list(x.shape[2:]) != self.siis_size:
            # s/output_stride != siis_size(default=[32, 32])
            x = resize(x, newsize=self.siis_size)
        x = self.siis(x)  # fix input size: siis_size

        # Decoder
        x = self.decoder_conv1(x)  # siis_size
        x = resize(x, newsize=size_4)  # Upx? -> s/4
        x = self.decoder_conv2(x)  # s/4
        x = resize(x, newsize=size)  # Upx4 -> s/1
        x = self.out_conv(x)

        return x


class Resnet_SIIS(BasicModule):
    """ Main module: Resnet_SIIS """
    def __init__(self, num_classes=2,
                 siis_size=[32, 32], width=1, kw=9, dim=128, arch=1,
                 resnet_arch='resnet50', output_stride=8, layer_num=2):
        super(Resnet_SIIS, self).__init__()
        self.siis_size = siis_size
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
        # layer_outSize = [s/4, s/output_stride, s/output_stride, ...]

        # conv_s - Make sure SIIS input dim. in: layers(out), out: siis(in).
        self.conv_s = self._make_layer(layers_dim[layer_num-1], dim, 1)
        self.siis = SIIS(siis_size, width, kw, dim, arch)  # in: conv_s(out)

        # Decoder
        self.decoder_conv1 = self._make_layer(dim, dim, 3, padding=1)  # siis_size
        self.decoder_conv2 = self._make_layer(dim, dim, 3, padding=1)  # s/4
        self.out_conv = nn.Conv2d(dim, num_classes, 1, 1)  # s/1 - output

        self.model_name = resnet_arch + self.siis.name

        # Initalization
        # init_list = ['conv_s', 'decoder_conv1', 'decoder_conv2', 'out_conv']
        # model_utils.custom_initialization(self, init_list)

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

        x = self.conv_s(x)  # s/output_stride, layers_dim -> dim
        # SIIS
        if list(x.shape[2:]) != self.siis_size:
            # s/output_stride != siis_size(default=[32, 32])
            x = resize(x, newsize=self.siis_size)
        x = self.siis(x)  # fix input size: siis_size

        # Decoder
        x = self.decoder_conv1(x)  # siis_size
        x = resize(x, newsize=size_4)  # Upx? -> s/4
        x = self.decoder_conv2(x)  # s/4

        x = resize(x, newsize=size)  # Upx4 -> s/1
        x = self.out_conv(x)  # s/1

        return x


class Deeplab_SIIS(BasicModule):
    """ Main Deeplab_SIIS model. """
    def __init__(self, num_classes=2,
                 siis_size=[32, 32], width=1, kw=9, dim=128, arch=1,
                 resnet_arch='resnet50', output_stride=8, layer_num=2):
        super(Deeplab_SIIS, self).__init__()
        self.siis_size = siis_size
        self.output_stride = output_stride
        self.layer_num = layer_num
        aspp_depth = 128

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
        # layer_outSize = [s/4, s/output_stride, s/output_stride, ...]

        self.conv2 = self._make_layer(64, 48, 1)  # in: pool1(out)
        # ASPP
        self.aspp = ASPP(
            in_channel=layers_dim[layer_num-1], depth=aspp_depth
        )  # ASPP: in: layers(out), fix_size=s/8

        # in: concat[conv2(out), Up(aspp(out))], out: siis(in)
        self.conv3 = self._make_layer(aspp_depth+48, dim, 1)  # s/4
        # SIIS
        self.siis = SIIS(siis_size, width, kw, dim, arch)  # size=siis_size, dim=dim

        # Decoder
        # in: siis(out),
        self.decoder_conv1 = self._make_layer(dim, dim, 3, padding=1)  # s/siis_size
        self.decoder_conv2 = self._make_layer(dim, dim, 3, padding=1)  # s/4

        self.out_conv = nn.Conv2d(dim, num_classes, 1, 1)  # s/1 - output

        self.model_name = 'deeplab2_'+self.siis.name

    def _make_layer(self, in_channel, out_channel, kernel_size, padding=0):
        return nn.Sequential(
            nn.Conv2d(in_channel, out_channel, kernel_size, 1, padding=padding),
            nn.BatchNorm2d(out_channel),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        size = x.shape[2:]  # need interpolate input size
        x = self.conv1(x)
        pool2 = self.pool1(x)
        size_4 = pool2.shape[2:]
        size_8 = [s//2 for s in size_4]

        x = self.layers(pool2)  # s/output_stride
        # x = self.layers[1:](x)  # s/output_stride

        # ASPP
        if list(x.shape[2:]) != size_8:
            x = resize(x, newsize=size_8)  # s/output_stride -> s/8
        decoder_features = self.aspp(x)  # fix input size s/8

        encoder_features = self.conv2(pool2)  # s/4
        encoder_features = resize(encoder_features, newsize=size_8)  # s/4 -> s/8

        x = torch.cat([encoder_features, decoder_features], dim=1)  # s/8

        x = self.conv3(x)  # s/8,  aspp_depth+48 -> dim
        # SIIS
        if list(x.shape[2:]) != self.siis_size:
            # s/8 != siis_size(default=[32, 32])
            x = resize(x, newsize=self.siis_size)
        x = self.siis(x)  # fix input size: siis_size

        # Decoder
        x = self.decoder_conv1(x)  # siis_size
        x = resize(x, newsize=size_4)  # Upx? -> s/4
        x = self.decoder_conv2(x)  # s/4

        x = resize(x, newsize=size)  # Upx4 -> s/1
        x = self.out_conv(x)  # s/1

        return x
