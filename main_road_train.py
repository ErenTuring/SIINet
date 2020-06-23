#!/usr/bin/env python3
# -*- coding: utf-8 -*-
'''
Main function for Segmentation
Build model and do training or predicting

'''
import os
import time

import numpy as np
import torch
from torch.utils.data import DataLoader
from torchnet import meter
from tqdm import tqdm

from utils.loss import jaccard_loss as mycriterion
from utils import dataset, tools  # visualize
# from models import DeeplabV3_plus as net
from models import build_model
from config.opt_mit import opt  # The Roadtracer dataset
# from config.opt_cvpr import opt  # The CVPR dataset
# from config.opt_rbdd import opt  # The Massachusetts road dataset
net_predict = tools.net_predict_enhance  # TTA


device, pin_memory = tools.setup_mode(opt.use_gpu)
print(opt.root)

# Basic initialization
# [main_net, siis, width, kw, dim, resnet_arch, layer_num, out_stride]
# NET_NUM = '2'+'4332'+'020'  # ResUnet + SIIS_Conv3dRNN
# NET_NUM = '3'+'4332'+'020'  # ResUnet + SIIS_Conv3dRNN
NET_NUM = '4'+'4330'+'020'  # Deeplabv3+ + SIIS_Conv3dRNN
# NET_NUM = '4'+'7330'+'020'  # Deeplabv3+ + SIIS_Conv1d
# NET_NUM = '6'+'0020'  # Deeplabv3
# NET_NUM = ['7', '8', '9'][1]  # ResUnet, HF_FCN, UNet
MAIN_MODE = ['train', 'val'][0]
C_EPOCH = 15
# class_names, label_values = get_label_info(opt.root+'/class_dict.txt')
class_names = ['BG', 'Road']
label_values = [(0, 0, 0), (255, 255, 255)]
CLASS_NUM = 2
# opt.dataset_dir += '/%s' % (['512', 'test', '512_theta0', '512_theta01'][3])
# opt.ckpt += ['/theta0', '', '/theta01'][2]

opt.num_workers = 0  # TODO: RNN cell not support multi workers
opt.start_epoch = 1
# opt.batch_size = 8
IF_VAL = True  # if do quick val while train mode
W_LABEL = True  # if save the color predict label while val mode

# Setup Model
model = build_model(CLASS_NUM, opt.siis_size, net_num=NET_NUM).to(device)
if NET_NUM[0] in ['4', '6']:
    NET_NUM = NET_NUM[:-1]  # ckpt name does not contain out_stride

# Load ckpt
if opt.start_epoch or MAIN_MODE != 'train':
    ckpt_epoch = C_EPOCH if MAIN_MODE != 'train' else opt.start_epoch
    print("Trying to load latest model")
    ckpt = opt.ckpt+'/%s_%d.pth' % (NET_NUM, ckpt_epoch)
    if os.path.isfile(ckpt):
        model_dict = torch.load(ckpt)
        model.load_state_dict(model_dict)  # ['model_state'])
        print("Loaded checkpoint %s(epoch %d)" % (NET_NUM, ckpt_epoch))
    elif MAIN_MODE != 'train':
        raise ValueError('Failed to load lastest model checkpoint.')
    else:
        print('Failed to load model checkpoint, re-initializtion.')


# Train
def train():
    # 1. Get data
    train_loader = DataLoader(
        dataset.Train_Dataset(opt.dataset_dir, 'train', opt),
        batch_size=opt.batch_size,
        shuffle=True,
        num_workers=opt.num_workers,
        drop_last=True,
        pin_memory=pin_memory)
    if IF_VAL:
        val_dataset = dataset.MyDataset_1(opt.dataset_dir, 'val', opt)
        val_dataset.crop_mode = 'random'  # quick val
        val_loader = DataLoader(val_dataset, 1, num_workers=0)

    # 2. Criterion
    if opt.loss == 'MSELoss':
        basic_criterion = torch.nn.MSELoss()
    elif opt.loss == 'BCEWithLogitsLoss':
        my_weight = torch.zeros(
            opt.batch_size, CLASS_NUM, opt.input_size[0], opt.input_size[1]
        )
        my_weight[:, 0, :, :] = opt.loss_weight[0]  # class 1(BG)
        my_weight[:, 1, :, :] = opt.loss_weight[1]  # class 2(Road)
        my_weight = my_weight.to(device)
        basic_criterion = torch.nn.BCEWithLogitsLoss(weight=my_weight)
    else:
        pass

    # 3.  Customization LR and Optimizer
    LR = opt.lr
    if NET_NUM[0] in ['4', '6']:
        # Resnet basic net
        encoder_params = list(map(id, model.conv1.parameters()))
        encoder_params += list(map(id, model.layers.parameters()))
        base_params = filter(
            lambda p: id(p) not in encoder_params, model.parameters()
        )
        optimizer = torch.optim.Adam(
            [
                {'params': base_params, 'lr': LR*10},
                {'params': model.conv1.parameters()},  # encoder部分学习率设定较低
                {'params': model.layers.parameters()}
            ],
            lr=LR, weight_decay=opt.weight_decay, betas=(0.9, 0.99)
        )
    elif NET_NUM[0] in ['8']:
        # Vgg basic net
        encoder_param_id = list(map(id, model.encoder.parameters()))
        base_params = filter(
            lambda p: id(p) not in encoder_param_id, model.parameters()
        )
        optimizer = torch.optim.Adam(
            [
                {'params': model.encoder.parameters()},  # encoder部分学习率设定较低
                {'params': base_params, 'lr': LR*10},
            ],
            lr=LR, weight_decay=opt.weight_decay, betas=(0.9, 0.99)
        )
    else:
        optimizer = torch.optim.Adam(
            model.parameters(), lr=LR, weight_decay=opt.weight_decay, betas=(0.9, 0.99)
        )
    # optimizer = torch.optim.SGD(
    #     model.parameters(), lr=LR, momentum=0.9, weight_decay=opt.weight_decay)

    # 4. Meters
    loss_meter = meter.AverageValueMeter()
    best_meter = 0  # [max(acc) or 1/min(loss)]

    # 5. Begain Train
    print('Begain Train! %s\n' % model.model_name,
          time.strftime('%Y-%m-%d %H:%M', time.localtime(time.time())))
    model.train()  # transform model mode
    tic = time.time()
    for epoch in range(opt.start_epoch, opt.max_epoch):
        loss_meter.reset()

        for i, data in enumerate(train_loader):
            input, label = data['image'].to(device), data['label'].unsqueeze(1)  # [NCHW], [N1HW]
            n, _, h, w = label.size()
            target = torch.zeros(n, 2, h, w).scatter_(dim=1, index=label, value=1).to(device)
            optimizer.zero_grad()
            # forward + backward + optimize
            output = model(input)  # no softmax layer; [NCHW]
            basic_loss = basic_criterion(output, target)  # .to(device)
            my_loss = mycriterion(output, target)  # .to(device)
            total_loss = opt.alpha*basic_loss + (1-opt.alpha)*(1-my_loss)
            total_loss.backward()
            optimizer.step()

            # meters update and visualize
            loss_meter.add(total_loss.item())

            if i % opt.print_freq == 0:
                tools.train_log(
                    "   Epoch:%3d iters:%6d Current_loss:%.3f Time:%.2f\r" % (
                        epoch+1, i, total_loss.item(), time.time()-tic))
                tic = time.time()  # update time

        # Validate and visualize
        log_str = ("\nepoch: [%d | %d], lr:%e, loss:%.4f" % (
            epoch+1, opt.max_epoch, LR, loss_meter.value()[0]))
        if IF_VAL:
            val_iou_class, val_iou_mean = quick_val(model, val_loader)
            log_str += (", val_IoU:%.5f" % val_iou_mean)
        print(log_str)

        # Save checkpoint
        if ((epoch+1) % opt.ckpt_freq) == 0 or epoch > (opt.max_epoch-5):
            model.save(opt.ckpt + '/%s_%d.pth' % (NET_NUM, epoch+1))
        if IF_VAL and best_meter < val_iou_mean:
            model.save(opt.ckpt+'/%s_best.pth' % NET_NUM)
            best_meter = val_iou_mean
        elif best_meter < (1 / loss_meter.value()[0]):
            model.save(opt.ckpt+'/%s_best.pth' % NET_NUM)
            best_meter = 1 / loss_meter.value()[0]

        LR = tools.adjust_lr(LR, opt.lr_decay, optimizer, epoch)
    print('\nFinish Train!')


def quick_val(model, dataloader):
    ''' IoU '''
    model.eval()
    with torch.no_grad():
        # iou = meter.AverageValueMeter()
        class_iou = np.zeros(CLASS_NUM)
        for ii, data in enumerate(dataloader):
            input, label = data['image'].to(device), data['label']
            label = np.squeeze(label.numpy())  # [NHW] -> [HW](n=1)
            output = model(input)[0]  # [NCHW] -> [CHW]
            predict = np.argmax(output.cpu().detach().numpy(), 0)  # [CHW] -> [HW]
            class_iou += tools.compute_class_iou(predict, label, CLASS_NUM)
    model.train()
    class_iou /= ii + 1
    return class_iou, np.mean(class_iou)  # mean_iou


def val():
    if W_LABEL:
        output_dir = opt.dataset_dir+'/%s_%d_pre' % (NET_NUM, C_EPOCH)
        if not os.path.exists(output_dir):
            os.mkdir(output_dir)

    # 1. Get data
    val_dataset = dataset.MyDataset_1(opt.dataset_dir, 'val', opt)
    val_loader = DataLoader(val_dataset, 1, num_workers=0)  # BS must be 1

    # 3. Begain val
    model.eval()
    class_iou = np.zeros(CLASS_NUM)
    for ii, data in tqdm(enumerate(val_loader)):
        image, label, name = data['image'], data['label'], data['name'][0]
        label = np.squeeze(label.numpy())  # [NHW] -> [HW](N=1)
        crop_info = np.array(data['crop_info']) if 'crop_info' in data.keys() else None
        # Do predict
        predict = net_predict(model, image, opt, crop_info).argmax(-1)
        class_iou += tools.compute_class_iou(predict, label, CLASS_NUM)
        # Visualize result
        if W_LABEL:
            tools.colour_code_label(predict, label_values, save_path=output_dir+'/'+name)

    class_iou /= ii + 1
    print('Class   \tIoU')
    for i in range(CLASS_NUM):
        print('class %s\t%.5f' % (class_names[i], class_iou[i]))
    print('Mean IoU\t%.5f' % (np.mean(class_iou)))


def main():
    if MAIN_MODE == 'train':
        train()
    elif MAIN_MODE == 'val':
        val()


if __name__ == '__main__':
    main()
