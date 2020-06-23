# -*- coding:utf-8 -*-
'''
Some custom loss functions for PyTorch.

Version 1.0  2018-11-02 15:15:44
'''
import torch
import numpy as np
# import torch.nn as nn
# import torch.nn.functional as F
# import torch.autograd as autograd


def jaccard_loss(input, target):
    ''' Soft IoU loss.
    Args:
        input - net output tensor, one_hot [NCHW]
        target - gt label tensor, one_hot [NCHW]
    '''
    n, c, h, w = input.size()
    # nt, ht, wt = target.size()
    if input.size(0) != target.size(0):
        raise ValueError('Expected input batch_size ({}) to match target batch_size ({}).'
                         .format(input.size(0), target.size(0)))
    input = torch.sigmoid(input)
    # Expand target tensor dim(if target's Channel is 1)
    # target = torch.zeros(n, 2, h, w).scatter_(dim=1, index=target, value=1)
    intersection = input * target  # be careful about wether need to use abs
    union = input + target - intersection
    return (intersection / union).sum() / (n*h*w)


class dice_bce_loss(torch.nn.Module):
    def __init__(self, batch=True):
        super(dice_bce_loss, self).__init__()
        self.batch = batch
        self.bce_loss = torch.nn.BCELoss()

    def soft_dice_coeff(self, target, input):
        smooth = 0.0  # may change
        if self.batch:
            i = torch.sum(target)
            j = torch.sum(input)
            intersection = torch.sum(target * input)
        else:
            i = target.sum(1).sum(1).sum(1)
            j = input.sum(1).sum(1).sum(1)
            intersection = (target * input).sum(1).sum(1).sum(1)
        score = (2. * intersection + smooth) / (i + j + smooth)
        #score = (intersection + smooth) / (i + j - intersection + smooth)#iou
        return score.mean()

    def soft_dice_loss(self, target, input):
        loss = 1 - self.soft_dice_coeff(target, input)
        return loss

    def __call__(self, target, input):
        input = torch.sigmoid(input)
        a = self.bce_loss(input, target)
        b = self.soft_dice_loss(target, input)
        return a + b


def main():

    # 预测值f(x) 构造样本，神经网络输出层
    # input_tensor = torch.ones([3, 2, 5, 5], dtype=torch.float64)
    # tmp_mat = torch.ones([5, 5], dtype=torch.float64)
    # input_tensor[0, 0, :, :] = tmp_mat * 0.5
    # input_tensor[1, 1, :, :] = tmp_mat * 0.5
    # input_tensor[2, 1, :, :] = tmp_mat * 0.5
    # label = torch.argmax(input_tensor, 3)
    # print(label[0])
    # print(label[1])
    # print(label.size())
    # [0.8, 0.2] * [1, 0]: 0.8 / (0.8+0.2 + 1 - 0.8) = 0.8 / 1.2 = 2/3
    # [0.4, 0.6] * [1, 0]: 0.4 / (2 - 0.4) = 0.4 / 1.6 = 1/4
    # [0.0, 1.0] * [0, 1]: 0

    # 真值y
    # labels = torch.LongTensor([0, 1, 4, 7, 3, 2]).unsqueeze(1)
    # print(labels.size())
    # one_hot = torch.zeros(6, 8).scatter_(dim=1, index=labels, value=1)
    # print(one_hot)

    # target_tensor = torch.ones([3, 5, 5], dtype=torch.int64).unsqueeze(1)
    # target_tensor = torch.zeros(3, 2, 5, 5).scatter_(1, target_tensor, 1)
    # print(target_tensor.size())
    # J = input_tensor * target_tensor

    p = np.array([0.8, 0.2])
    t = np.array([1, 0])
    print()
    pass


if __name__ == '__main__':
    main()
