'''
SIIS
'''

import torch
from torch import nn
from torch.nn import functional as F
import math


class SegRNNCell(nn.Module):
    '''ConvRNN Cell for segmentation.
    Args:
        dim - The number of features in the input tensor.
        ks - The size of conv kernel.
    input: [N,dim,width,1,W] or [N,dim,width,H,1]
    '''
    def __init__(self, dim, ks, bias=True, nonlinearity='relu'):
        super(SegRNNCell, self).__init__()
        # self.input_size = input_size  # (N,C,width,W)
        self.dim = dim
        self.ks = ks
        self.bias = bias
        self.alpha = nn.Parameter(torch.ones((1, 1, 1, 1, 1), dtype=torch.float32))

        self.w_ih = self._make_layer(ks)
        self.w_hh = self._make_layer(ks)
        self.w_hz = self._make_layer((1, 1, 1))

    def _make_layer(self, k_size=(1, 1, 1), stride=1):
        ''' Use Conv3d filter as seg_rnn w. '''
        pad = ((k_size[0]-1)//2, (k_size[1]-1)//2, (k_size[2]-1)//2)  # same padding
        return nn.Sequential(
            nn.Conv3d(
                in_channels=self.dim, out_channels=self.dim,
                kernel_size=k_size, stride=stride, padding=pad, bias=False
            ),
            nn.BatchNorm3d(self.dim),
            nn.ReLU(inplace=True)
        )

    def forward(self, input):
        (x, hx) = input
        Wx = self.w_ih(x)  # w_ih * x + b_ih
        Wh = self.w_hh(hx)  # w_hh * h + b_hh
        # ht = self.w_ih(x) + torch.sigmoid(self.alpha) * hx
        ht = Wx + self.alpha * Wh
        hz = self.w_hz(ht)
        return hz


class BuildPass_4(nn.Module):
    '''
    Sub_modle of SIIS, complete the transfer of information on 4 paths.
    Note: BuildPass_4 is Conv3D
    Args:
        d - Direction(1:down-up; 2:right-left)
        width - Slice thickness along the H direction
        dim - Dim of feature map
        kw - Conv kener size
        Num - Actually the H/W of input feature map
    Notes：
        num - The number of slice(also the conv kernel) = Num//width
    '''
    def __init__(self, d, width, kw, dim, Num):
        super(BuildPass_4, self).__init__()
        self.d = d
        self.dim = dim
        self.kw = kw
        self.width = width
        self.num = Num // width
        assert (self.num * width) == Num

        # make two convs pass(Down-Up or Left-Right)
        self.pass_1 = nn.Sequential()  # Down / Left
        self.pass_2 = nn.Sequential()  # Up / Right
        # for i in range(self.num):
        if self.d == 1:
            self.pass_1.add_module(('SIIS_D'), SegRNNCell(dim, (width, 1, kw), False))
            self.pass_2.add_module(('SIIS_U'), SegRNNCell(dim, (width, 1, kw), False))
        else:
            self.pass_1.add_module(('SIIS_L'), SegRNNCell(dim, (width, kw, 1), False))
            self.pass_2.add_module(('SIIS_R'), SegRNNCell(dim, (width, kw, 1), False))

        # initialize
        std = math.sqrt(2 / (self.width * self.kw * self.dim * self.dim * 5))
        for m in self.modules():
            if isinstance(m, nn.Conv3d):
                m.weight.data.normal_(0, std)

    def forward(self, x):
        # [NCHW] -> n*[N,C,width,W] or n*[N,C,H,width]
        fm = torch.split(x, self.width, self.d+1)  # slice alonge H/W
        if len(fm) != self.num:
            raise ValueError('The number of feature map cuts is inconsistent with the number of filters')

        FM = []
        # 1. Down/Left first
        for i in range(self.num):
            # Feature: [N,C,width,W] -> [N,C,width,1,W]
            # Filter kernel: [width,kw] -> [width,1,kw] or [width,kw] -> [width,kw,1]
            FM.append(fm[i].unsqueeze(2).transpose(2, self.d+2))
        fm = []
        hx = FM[0].detach()
        for i in range(0, self.num):
            hx = self.pass_1((FM[i], hx))
            FM[i] = hx

        # 2. Up/Right second
        hx = FM[-1].detach()
        for i in range(self.num-2, -1, -1):
            hx = self.pass_2((FM[i], hx))
            FM[i] = hx

        FM = torch.split(torch.cat(FM, 2), 1, 2)  # n*[N,C,width,1,W]->[NCH1W]->H[NC11W]
        FM = [s.transpose(2, self.d+2).squeeze(2) for s in FM]  # 5D -> 4D
        x = torch.cat(FM, self.d+1)  # H*[NC1W] -> [NCHW]
        return x


class SIIS_Conv3dRNN(nn.Module):
    '''
    SIIS model No.4, Conv3d RNN version.
    Args:
        input_shape - [H, W]
    '''
    def __init__(self, input_shape, width=1, kw=9, dim=128):
        super(SIIS_Conv3dRNN, self).__init__()
        self.name = 'SIIS_Conv3dRNN'
        self.input_shape = input_shape
        self.new_size = None
        [H, W] = input_shape  # channel last!!
        while H % width != 0:
            H += 1
            self.new_size = [H, W]
        while W % width != 0:
            W += 1
            self.new_size = [H, W]

        self.DU = BuildPass_4(1, width, kw, dim, H)  # down-up pass
        self.LR = BuildPass_4(2, width, kw, dim, W)  # left-right pass

    def forward(self, x):
        if self.new_size is not None:  # size of input feature map need to be adjusted
            x = F.interpolate(x, size=self.new_size, mode='nearest')
            x = self.DU(x)
            x = self.LR(x)
            x = F.interpolate(x, size=self.input_shape, mode='nearest')
        else:
            x = self.DU(x)
            x = self.LR(x)
        return x


class SIIS_Conv1d(nn.Module):
    '''
    SIIS model No.7, conv1d.
    Args:
        input_shape - [H, W]
    '''
    def __init__(self, input_shape, width=1, kw=9, dim=128):
        super(SIIS_Conv1d, self).__init__()
        self.name = 'SIIS_Conv1d'
        self.input_shape = input_shape
        self.dim = dim
        self.kw = kw

        # self.DU = BuildPass_6(1, width, kw, dim, [H, W])  # down-up pass
        # self.LR = BuildPass_6(2, width, kw, dim, [H, W])  # left-right pass
        # down-up pass
        self.D = self._make_layer()
        self.U = self._make_layer()
        # left-right pass
        self.L = self._make_layer()
        self.R = self._make_layer()

    def _make_layer(self):
        return nn.Sequential(
            nn.Conv1d(self.dim, self.dim, self.kw, 1, padding=(self.kw-1)//2),
            nn.BatchNorm1d(self.dim),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        H, W = x.shape[:2]

        # 1. Down-up
        fms = torch.split(x, 1, dim=3)  # split along W
        FMs = []
        for fm in fms:
            # W*[N,C,H,1] -> W*[N,C,H]
            new_fm = self.D(fm.squeeze(3))
            new_fm = self.U(torch.flip(new_fm, [2]))
            FMs.append(torch.flip(new_fm, [2]))
        x = torch.stack(FMs, dim=3)

        # 2. Left-Right
        fms = torch.split(x, 1, dim=2)  # split along H
        FMs = []
        for fm in fms:
            # W*[N,C,1,W] -> H*[N,C,W]
            new_fm = self.L(fm.squeeze(2))
            new_fm = self.R(torch.flip(new_fm, [2]))
            FMs.append(torch.flip(new_fm, [2]))
        x = torch.stack(FMs, dim=2)

        return x


def SIIS(input_shape, width=1, kw=9, dim=128, arch=1):
    '''
    Args:
        input_shape - [H, W]
        width - Slice thickness along the H direction
        kw - Conv kener size
        dim - Dim of feature map
        arch - Select arch of SIIS
    '''
    if arch == 4:
        return SIIS_Conv3dRNN(input_shape, width, kw, dim)
    elif arch == 7:
        return SIIS_Conv1d(input_shape, width, kw, dim)
