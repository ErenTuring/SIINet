'''
deeplab_v3+ : pytorch resnet 18/34 Basicblock
                      resnet 50/101/152 Bottleneck
'''
import torch
# import torchvision
from torch import nn
import torch.nn.functional as F
from .BasicModule import BasicModule
from models import resnet


def resize(tensor, newsize):
    return F.interpolate(
        tensor, size=newsize, mode='bilinear', align_corners=True)


class ASPP(nn.Module):
    '''
    ASPP consists of (a) one 1x1 convolution and three 3x3 convolutions
    with rates = (6, 12, 18) when output stride = 16 (all with 256 filters
    and batch normalization), and (b) the image-level features as described in the paper
    Careful!! Content the output 1x1 conv.
    '''
    def __init__(self, in_channel=512, depth=128):
        super().__init__()
        self.in_channel = in_channel
        self.depth = depth
        # Global average pooling
        self.global_avg_pooling = nn.AdaptiveAvgPool2d((1, 1))
        self.conv = self._make_layer(kernel_size=1)  # conv first, then upsample!

        self.atrous_block_1 = self._make_layer(kernel_size=1)
        self.atrous_block_2 = self._make_layer(3, 2, 2)
        self.atrous_block_6 = self._make_layer(3, 6, 6)
        self.atrous_block_12 = self._make_layer(3, 12, 12)

        self.conv_output = nn.Sequential(
            nn.Conv2d(depth*5, depth, kernel_size=1, stride=1),
            nn.BatchNorm2d(depth),
            nn.ReLU(inplace=True)
        )

    def _make_layer(self, kernel_size, padding=0, rate=1):
        ''' Let padding=dilation can make sure the input shape is same as output(ks=3) '''
        return nn.Sequential(
            nn.Conv2d(
                self.in_channel, self.depth, kernel_size, 1, padding=padding, dilation=rate),
            nn.BatchNorm2d(self.depth),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        size = x.shape[2:]

        image_feature = self.global_avg_pooling(x)
        image_feature = self.conv(image_feature)
        image_feature = resize(image_feature, newsize=size)
        block_1 = self.atrous_block_1(x)
        block_2 = self.atrous_block_2(x)
        block_6 = self.atrous_block_6(x)
        block_12 = self.atrous_block_12(x)

        concat = [image_feature, block_1, block_2, block_6, block_12]
        x = self.conv_output(torch.cat(concat, 1))
        return x


class ASPP_test(nn.Module):
    '''
    ASPP consists of (a) one 1x1 convolution and three 3x3 convolutions
    with rates = (6, 12, 18) when output stride = 16 (all with 256 filters
    and batch normalization), and (b) the image-level features as described in the paper
    Careful!! Content the output 1x1 conv.
    '''
    def __init__(self, in_channel=512, depth=128, rate=1):
        super().__init__()
        self.in_channel = in_channel
        self.depth = depth

        self.atrous_block = self._make_layer(3, rate, rate)

    def _make_layer(self, kernel_size, padding=0, rate=1):
        ''' Let padding=dilation can make sure the input shape is same as output(ks=3) '''
        return nn.Sequential(
            nn.Conv2d(
                self.in_channel, self.depth, kernel_size, 1, padding=padding, dilation=rate),
            nn.BatchNorm2d(self.depth),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        x = self.atrous_block(x)
        return x


class DeeplabV3_plus(BasicModule):
    ''' Main model: DeepLabV3+ '''
    def __init__(self, num_classes=2,
                 resnet_arch='resnet50', output_stride=8, layer_num=2,
                 aspp_rate=0):
        super().__init__()
        self.model_name = 'deeplabv3plus'
        self.layer_num = layer_num
        self.output_stride = output_stride
        aspp_depth = 256

        if resnet_arch == 'resnet50':
            encoder = resnet.resnet50(True, output_stride=self.output_stride)
        elif resnet_arch == 'resnet101':
            encoder = resnet.resnet101(True, output_stride=self.output_stride)
        encoder = encoder._modules  # Covert class instance into orderdict

        # decay=0.9997, epsilon=1e-5, scale=True
        self.conv1 = nn.Sequential(encoder['conv1'], encoder['bn1'], encoder['relu'])
        self.pool1 = encoder['maxpool']  # s/4 - 64dim

        self.layers = nn.Sequential()
        for i in range(layer_num):
            self.layers.add_module('layer%d' % (i+1), encoder['layer%d' % (i+1)])
        layers_dim = [256, 512, 1024, 2048, 2048, 1024, 512]
        # layer_outSize = [s/4, s/output_stride, s/output_stride, ...]
        self.conv2 = self._make_layer(64, 48, 1)  # in: pool1(out)

        rate_tabel = [1, 6, 12, 18, 24, 1, 3]
        if aspp_rate == 0:
            self.aspp = ASPP(
                in_channel=layers_dim[layer_num - 1],
                depth=aspp_depth)  # ASPP: in: layers(out), fix_size=s/8
        else:
            self.aspp = ASPP_test(layers_dim[layer_num - 1], aspp_depth,
                                  rate_tabel[aspp_rate])

        # Decoder
        self.decoder_conv1 = self._make_layer(
            aspp_depth + 48, aspp_depth, 3, padding=1
        )  # in: concat[conv2(out), Up(aspp(out))]
        self.decoder_conv2 = self._make_layer(
            aspp_depth, aspp_depth, 3, padding=1)  # s/4

        self.out_conv = nn.Conv2d(aspp_depth, num_classes, 1, 1)  # s/1 - output

        # Initalization
        # source code didn't specify the way of weight initalization of the decoder,
        # but slim default is zero initalization
        # init_list = ['conv2', 'aspp', 'decoder_conv1', 'decoder_conv2', 'out_conv']
        # for name, child_moudle in self.named_children():
        #     if name in init_list:
        #         for name, m in child_moudle.named_modules():
        #             if isinstance(m, nn.Conv2d):
        #                 nn.init.kaiming_normal_(m.weight.data, nonlinearity='relu')
        #             if isinstance(m, nn.BatchNorm2d):
        #                 m.weight.data.fill_(1)
        #                 m.weight.data.zero_()

    def _make_layer(self, in_channel, out_channel, kernel_size, padding=0):
        return nn.Sequential(
            nn.Conv2d(in_channel, out_channel, kernel_size, 1, padding=padding),
            nn.BatchNorm2d(out_channel),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        size = x.shape[2:]  # need upsample input size
        x = self.conv1(x)
        pool2 = self.pool1(x)  # s/4
        size_4 = pool2.shape[2:]
        size_8 = [s//2 for s in size_4]

        x = self.layers(pool2)  # s/output_stride

        # ASPP
        if list(x.shape[2:]) != size_8:
            x = resize(x, newsize=size_8)  # s/output_stride -> s/8
        x = self.aspp(x)  # fix input size s/8

        decoder_features = resize(x, newsize=pool2.shape[2:])  # s/4

        encoder_features = self.conv2(pool2)  # s/4
        x = torch.cat([encoder_features, decoder_features], dim=1)  # s/4
        x = self.decoder_conv1(x)
        x = self.decoder_conv2(x)
        x = resize(x, newsize=size)  # Upx4 -> s/1

        x = self.out_conv(x)  # s/1
        return x


def build_model(num_classes=5):
    model = DeeplabV3_plus(num_classes=num_classes)
    return model
