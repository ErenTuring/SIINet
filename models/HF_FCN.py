import torch
import torch.nn as nn
# import torch.utils.model_zoo as model_zoo
import torch.nn.functional as F
from torchvision.models import vgg
from .BasicModule import BasicModule


def resize(tensor, newsize):
    return F.interpolate(
        tensor, size=newsize, mode='bilinear', align_corners=True)


class HF_FCN_back_1(BasicModule):
    ''' HF_FCN(use vgg_16 with bn) '''
    def __init__(self, num_classes=1000):
        super(HF_FCN, self).__init__()
        self.model_name = 'hf_fcn'

        # Encoder
        features = vgg.vgg16_bn(pretrained=True).features

        self.conv1_1 = features[0:3]  # s/1 - dim=64
        self.conv1_2 = features[3:6]
        self.pool_1 = features[6]  # s/1 -> s/2

        self.conv2_1 = features[7:10]  # s/2 - dim=128
        self.conv2_2 = features[10:13]
        self.pool_2 = features[13]  # s/2 -> s/4

        self.conv3_1 = features[14:17]  # s/4 - dim=256
        self.conv3_2 = features[17:20]
        self.conv3_3 = features[20:23]
        self.pool_3 = features[23]  # s/4 -> s/8

        self.conv4_1 = features[24:27]  # s/8 - dim=512
        self.conv4_2 = features[27:30]
        self.conv4_3 = features[30:33]
        self.pool_4 = features[33]  # s/8 -> s/16

        self.conv5_1 = features[34:37]  # s/16 - dim=512
        self.conv5_2 = features[37:40]
        self.conv5_3 = features[40:43]

        # Decoder
        self.updsn1_1 = self._make_layer(64, 1, 1)
        self.updsn1_2 = self._make_layer(64, 1, 1)
        self.updsn2_1 = self._make_layer(128, 1, 1)
        self.updsn2_2 = self._make_layer(128, 1, 1)
        self.updsn3_1 = self._make_layer(256, 1, 1)
        self.updsn3_2 = self._make_layer(256, 1, 1)
        self.updsn3_3 = self._make_layer(256, 1, 1)
        self.updsn4_1 = self._make_layer(512, 1, 1)
        self.updsn4_2 = self._make_layer(512, 1, 1)
        self.updsn4_3 = self._make_layer(512, 1, 1)
        self.updsn5_1 = self._make_layer(512, 1, 1)
        self.updsn5_2 = self._make_layer(512, 1, 1)
        self.updsn5_3 = self._make_layer(512, 1, 1)

        self.out_conv = nn.Conv2d(13, num_classes, 1, 1)

        # Initalization
        # for name, m in self.modules():
        #     if name.find('updsn') != -1:
        #         if isinstance(m, nn.Conv2d):
        #             nn.init.kaiming_normal_(m.weight.data, nonlinearity='relu')
        #         if isinstance(m, nn.BatchNorm2d):
        #             m.weight.data.fill_(1)
        #             m.weight.data.zero_()

    def _make_layer(self, inchannel, outchannel, k_size, pad=0):
        return nn.Sequential(
            nn.Conv2d(inchannel, outchannel, k_size, padding=pad),
            nn.BatchNorm2d(outchannel),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        input_size = x.shape[2:]
        conv1_1 = self.conv1_1(x)
        conv1_2 = self.conv1_2(conv1_1)
        conv2_1 = self.conv2_1(conv1_2)
        conv2_2 = self.conv2_2(conv2_1)
        conv3_1 = self.conv3_1(conv2_2)
        conv3_2 = self.conv3_2(conv3_1)
        conv3_3 = self.conv3_3(conv3_2)
        conv4_1 = self.conv4_1(conv3_3)
        conv4_2 = self.conv4_2(conv4_1)
        conv4_3 = self.conv4_3(conv4_2)
        conv5_1 = self.conv5_1(conv4_3)
        conv5_2 = self.conv5_2(conv5_1)
        conv5_3 = self.conv5_3(conv5_2)

        updsn1_1 = self.updsn1_1(conv1_1)
        updsn1_2 = self.updsn1_2(conv1_2)

        updsn2_1 = self.updsn2_1(conv2_1)
        updsn2_1 = resize(updsn2_1, input_size)
        updsn2_2 = self.updsn2_2(conv2_2)
        updsn2_2 = resize(updsn2_2, input_size)

        updsn3_1 = self.updsn3_1(conv3_1)
        updsn3_1 = resize(updsn3_1, input_size)
        updsn3_2 = self.updsn3_2(conv3_2)
        updsn3_2 = resize(updsn3_2, input_size)
        updsn3_3 = self.updsn3_3(conv3_3)
        updsn3_3 = resize(updsn3_3, input_size)

        updsn4_1 = self.updsn4_1(conv4_1)
        updsn4_1 = resize(updsn4_1, input_size)
        updsn4_2 = self.updsn4_2(conv4_2)
        updsn4_2 = resize(updsn4_2, input_size)
        updsn4_3 = self.updsn4_3(conv3_3)
        updsn4_3 = resize(updsn4_3, input_size)

        updsn5_1 = self.updsn5_1(conv5_1)
        updsn5_1 = resize(updsn5_1, input_size)
        updsn5_2 = self.updsn5_2(conv5_2)
        updsn5_2 = resize(updsn5_2, input_size)
        updsn5_3 = self.updsn5_3(conv5_3)
        updsn5_3 = resize(updsn5_3, input_size)

        concat = [updsn1_1, updsn1_2, updsn2_1, updsn2_2,
                  updsn3_1, updsn3_2, updsn3_3, updsn4_1,
                  updsn4_2, updsn4_3, updsn5_1, updsn5_2,
                  updsn5_3]
        x = torch.cat(concat, 1)

        x = self.out_conv(x)
        return x


class HF_FCN(BasicModule):
    ''' HF_FCN(use vgg_16 with bn) '''
    def __init__(self, num_classes=1000):
        super(HF_FCN, self).__init__()
        self.model_name = 'hf_fcn'

        # Encoder
        features = vgg.vgg16_bn(pretrained=True).features

        # self.
        self.encoder = nn.Sequential()
        self.encoder.add_module('conv1_1', features[0:3])  # s/1 - dim=64
        self.encoder.add_module('conv1_2', features[3:6])
        self.encoder.add_module('pool_1', features[6])  # s/1 -> s/2

        self.encoder.add_module('conv2_1', features[7:10])  # s/2 - dim=128
        self.encoder.add_module('conv2_2', features[10:13])
        self.encoder.add_module('pool_2', features[13])  # s/2 -> s/4

        self.encoder.add_module('conv3_1', features[14:17])  # s/4 - dim=256
        self.encoder.add_module('conv3_2', features[17:20])
        self.encoder.add_module('conv3_3', features[20:23])
        self.encoder.add_module('pool_3', features[23])  # s/4 -> s/8

        self.encoder.add_module('conv4_1', features[24:27])  # s/8 - dim=512
        self.encoder.add_module('conv4_2', features[27:30])
        self.encoder.add_module('conv4_3', features[30:33])
        self.encoder.add_module('pool_4', features[33])  # s/8 -> s/16

        self.encoder.add_module('conv5_1', features[34:37])  # s/16 - dim=512
        self.encoder.add_module('conv5_2', features[37:40])
        self.encoder.add_module('conv5_3', features[40:43])

        # Decoder
        self.decoder = nn.Sequential()
        self.decoder.add_module('updsn1_1', self._make_layer(64, 1, 1))
        self.decoder.add_module('updsn1_2', self._make_layer(64, 1, 1))
        self.decoder.add_module('updsn2_1', self._make_layer(128, 1, 1))
        self.decoder.add_module('updsn2_2', self._make_layer(128, 1, 1))
        self.decoder.add_module('updsn3_1', self._make_layer(256, 1, 1))
        self.decoder.add_module('updsn3_2', self._make_layer(256, 1, 1))
        self.decoder.add_module('updsn3_3', self._make_layer(256, 1, 1))
        self.decoder.add_module('updsn4_1', self._make_layer(512, 1, 1))
        self.decoder.add_module('updsn4_2', self._make_layer(512, 1, 1))
        self.decoder.add_module('updsn4_3', self._make_layer(512, 1, 1))
        self.decoder.add_module('updsn5_1', self._make_layer(512, 1, 1))
        self.decoder.add_module('updsn5_2', self._make_layer(512, 1, 1))
        self.decoder.add_module('updsn5_3', self._make_layer(512, 1, 1))
        self.out_conv = nn.Conv2d(13, num_classes, 1, 1)

        # Initalization
        # for name, m in self.modules():
        #     if name.find('updsn') != -1:
        #         if isinstance(m, nn.Conv2d):
        #             nn.init.kaiming_normal_(m.weight.data, nonlinearity='relu')
        #         if isinstance(m, nn.BatchNorm2d):
        #             m.weight.data.fill_(1)
        #             m.weight.data.zero_()

    def _make_layer(self, inchannel, outchannel, k_size, pad=0):
        return nn.Sequential(
            nn.Conv2d(inchannel, outchannel, k_size, padding=pad),
            nn.BatchNorm2d(outchannel), nn.ReLU(inplace=True))

    def forward(self, x):
        input_size = x.shape[2:]
        conv1_1 = self.encoder.conv1_1(x)
        conv1_2 = self.encoder.conv1_2(conv1_1)
        x = self.encoder.pool_1(conv1_2)  # s/1 -> s/2
        conv2_1 = self.encoder.conv2_1(x)
        conv2_2 = self.encoder.conv2_2(conv2_1)
        x = self.encoder.pool_2(conv2_2)  # s/2 -> s/4
        conv3_1 = self.encoder.conv3_1(x)
        conv3_2 = self.encoder.conv3_2(conv3_1)
        conv3_3 = self.encoder.conv3_3(conv3_2)
        x = self.encoder.pool_3(conv3_3)  # s/4 -> s/8
        conv4_1 = self.encoder.conv4_1(x)
        conv4_2 = self.encoder.conv4_2(conv4_1)
        conv4_3 = self.encoder.conv4_3(conv4_2)
        x = self.encoder.pool_4(conv4_3)  # s/8 -> s/16
        conv5_1 = self.encoder.conv5_1(x)
        conv5_2 = self.encoder.conv5_2(conv5_1)
        conv5_3 = self.encoder.conv5_3(conv5_2)

        updsn1_1 = self.decoder.updsn1_1(conv1_1)
        updsn1_2 = self.decoder.updsn1_2(conv1_2)

        updsn2_1 = self.decoder.updsn2_1(conv2_1)
        updsn2_1 = resize(updsn2_1, input_size)
        updsn2_2 = self.decoder.updsn2_2(conv2_2)
        updsn2_2 = resize(updsn2_2, input_size)

        updsn3_1 = self.decoder.updsn3_1(conv3_1)
        updsn3_1 = resize(updsn3_1, input_size)
        updsn3_2 = self.decoder.updsn3_2(conv3_2)
        updsn3_2 = resize(updsn3_2, input_size)
        updsn3_3 = self.decoder.updsn3_3(conv3_3)
        updsn3_3 = resize(updsn3_3, input_size)

        updsn4_1 = self.decoder.updsn4_1(conv4_1)
        updsn4_1 = resize(updsn4_1, input_size)
        updsn4_2 = self.decoder.updsn4_2(conv4_2)
        updsn4_2 = resize(updsn4_2, input_size)
        updsn4_3 = self.decoder.updsn4_3(conv4_3)
        updsn4_3 = resize(updsn4_3, input_size)

        updsn5_1 = self.decoder.updsn5_1(conv5_1)
        updsn5_1 = resize(updsn5_1, input_size)
        updsn5_2 = self.decoder.updsn5_2(conv5_2)
        updsn5_2 = resize(updsn5_2, input_size)
        updsn5_3 = self.decoder.updsn5_3(conv5_3)
        updsn5_3 = resize(updsn5_3, input_size)

        concat = [updsn1_1, updsn1_2, updsn2_1, updsn2_2,
                  updsn3_1, updsn3_2, updsn3_3, updsn4_1,
                  updsn4_2, updsn4_3, updsn5_1, updsn5_2,
                  updsn5_3]
        x = torch.cat(concat, 1)

        x = self.out_conv(x)
        return x


def build_model(pretrained=False, num_classes=1000):
    '''Build model'''
    model = HF_FCN(num_classes)
    return model
