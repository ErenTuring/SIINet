#!/usr/bin/env python3
# -*- coding: utf-8 -*-
'''
Main function for Road Detection
Do val and predicting

'''
import os
import numpy as np
from torch.utils.data import DataLoader
from tqdm import tqdm

from utils import dataset, tools  # visualize, tools
from models import build_model
from config import get_opt
net_predict = tools.net_predict_enhance


DATASET = 'mit'
opt = get_opt(DATASET)
device, pin_memory = tools.setup_mode(opt.use_gpu)
print('root:', opt.root)

# 1. Basic initialization
# [main_net, siis, width, kw, dim, resnet_arch, layer_num, out_stride]
# NET_NUM = '4'+'7330'+'020'  # conv1d
# NET_NUM = '2'+'4332'+'020'  # Unet + SIIS
NET_NUM = '3'+'4332'+'020'  # ResUnet + SIIS
# NET_NUM = '4'+'4330'+'020'  # SII-Net
# NET_NUM = '6'+'0020'  # deeplab
# NET_NUM = ['7', '8', '9'][0]  # ResUnet, HF_FCN, UNet
E_MODE = ['test', 'val'][0]  # Evaluation mode: [test, val]
MAIN_MODE = 'e'  # Main mode: e-'val' or 'test'; x-self commands; c-collect label
C_EPOCH = 0  # current epoch

W_LABEL = 1  # if save color predict label
LOG = 0  # if log the evalution result


# 2. Get data
# opt.ckpt += ['/theta0', '', '/theta01'][2]  # 单纯的裁剪

if DATASET == 'mit':
    # opt.dataset_dir += '/test'
    opt.dataset_dir += '/test_all'

class_names = ['BG', 'Road']
label_values = [(0, 0, 0), (255, 255, 255)]
CLASS_NUM = 2
m_dataset = dataset.MyDataset_2(opt.dataset_dir, 'test', opt)
# dataset.crop_mode = 'random'  # quick val
dataloader = DataLoader(m_dataset, 1, num_workers=0)


def test_mit(model, out_dir):
    ''' Do test(predict) only for Roadtracer dataset'''
    import cv2
    T = 0.4  # Here we also adjusted the optimal threshold just as Roadtracer did.
    # Begain test
    for ii, data in tqdm(enumerate(dataloader)):
        image, name = data['image'], data['name'][0]
        # image = cv2.resize(image, )
        crop_info = np.array(data['crop_info']) if 'crop_info' in data.keys() else None
        predict = tools.net_predict_tta(model, image, opt, crop_info)[:, :, 1]
        predict[predict > T] = 1
        predict[predict <= T] = 0
        # Visualize and save result
        # name = rename_file(name, ifID=1, addstr='mask', extension='png')
        predict = cv2.resize(predict.astype(np.uint8), (8192, 8192), cv2.INTER_NEAREST)
        tools.colour_code_label(predict, label_values, save_path=out_dir+'/'+name)


def val(model, out_dir, result_file_name=None):
    ''' Do val and log the evaluation result. '''
    from utils import evaluation

    # runingscore = evaluation.RoadExtractionScore(CLASS_NUM)
    runingscore = evaluation.RelaxedRoadExtractionScore(3)

    if LOG:
        f = open(result_file_name, 'w')
        f.write('Name,' + runingscore.keys() + '\n')

    for ii, data in tqdm(enumerate(dataloader)):
        image, label, name = data['image'], data['label'], data['name'][0]
        label = np.squeeze(label.numpy())  # [NHW] -> [HW](N=1)
        crop_info = np.array(data['crop_info']) if 'crop_info' in data.keys() else None
        # Predict
        predict = net_predict(model, image, opt, crop_info).argmax(-1)
        if LOG:
            score = runingscore.update(label, predict)
            f.write(name+','+runingscore.print_score(score, 1))
        else:
            runingscore.add(label, predict)
        # Visualize result
        if W_LABEL:
            tools.colour_code_label(predict, label_values, save_path=out_dir+'/'+name)

    runingscore.print_score(runingscore.get_scores())  # print total scores


def get_net(num_classes=2, net_num=None):
    # 3. Setup Model
    model = build_model(num_classes, opt.siis_size, net_num=net_num).to(device)
    model.eval()  # predict mode

    if net_num[0] in ['4', '6']:
        net_num = net_num[:-1]  # ckpt name does not contain out_stride

    # 4. Load ckpt
    # if os.path.isfile(opt.ckpt + '/%s_%d.pth' % (net_num, C_EPOCH)):
    #     checkpoint = torch.load(opt.ckpt+'/%s_%d.pth' % (net_num, C_EPOCH))
    #     model.load_state_dict(checkpoint)  # ['model_state'])
    #     print("Loaded checkpoint %s(epoch %d)" % (net_num, C_EPOCH))
    if C_EPOCH:
        ckpt = opt.ckpt + '/%s_%d.pth' % (net_num, C_EPOCH)
    else:
        ckpt = opt.ckpt + '/%s_best.pth' % (net_num)
    tools.load_ckpt(model, ckpt, MAIN_MODE == 'train')

    return model


def main(net_num):
    # 3. Setup Model
    net = get_net(CLASS_NUM, net_num)

    # 5. Do some preparation
    output_dir = opt.dataset_dir+'/%s_%d_pre' % (net_num, C_EPOCH)

    if W_LABEL:  # if want to write result, make out dir
        if not os.path.exists(output_dir):
            os.mkdir(output_dir)

    if E_MODE == 'test':
        # test(net, output_dir)
        test_mit(net, output_dir)
    elif E_MODE == 'val':
        result_file_name = opt.dataset_dir+'/%s_%d.txt' % (net_num, C_EPOCH)
        val(net, output_dir, result_file_name)

    else:
        print('Invalid mode!')


if __name__ == '__main__':
    if MAIN_MODE == 'e':  # E_MODE = ['test', 'val']
        main(NET_NUM)
    elif MAIN_MODE == 'x':
        # **********************************************
        # Commands

        # Filter Data
        # filter_data_by_iou(opt.dataset_dir, result_file_name)
        main('2'+'4332'+'020')  # Unet + SIIS
        main('4'+'7330'+'020')  # SII-Net conv1d
        # main('3'+'4330'+'020')
        # main('4'+'4330'+'020')
        # main('6'+'0020')
        # main('6'+'1020')
        # main('6'+'2020')
        # main('6'+'3020')
        # main('6'+'4020')
        # main('6'+'5020')
        # main('7')  # ResUnet
        # main('8')  # HF_FCN
        # main('9')  # UNet
        # collect_label()

        # **********************************************
        pass
