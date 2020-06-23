#!/usr/bin/env python3
# -*- coding: utf-8 -*-
'''
Default Config about Massachusetts road dataset.
'''


class DefaultConfig(object):
    env = 'result'  # visdom 环境

    # Path and file
    # debug_file = '/tmp/debug'  # if os.path.exists(debug_file): enter ipdb
    # result_file = 'result.csv'
    root = '/home/tao/Data/RBDD'  # ISPRS,cvpr_road,Road_Car,Road,KITTIT
    dataset_dir = root
    ckpt = root + '/Model/cp'

    # Model related arguments
    continue_training = True
    start_epoch = 0  # use to continue from a checkpoint

    # Optimiztion related arguments
    use_gpu = True  # if use GPU
    batch_size = 16  # batch size
    max_epoch = 15  # 16-[256,256] dataset only need 8~9 epoch
    ckpt_freq = 3
    lr = 1e-5  # initial learning rate
    lr_decay = 0.98  # pre epoch
    weight_decay = 1e-5  # L2 loss
    loss = ['MSELoss', 'BCEWithLogitsLoss', ''][0]
    loss_weight = [0.2, 0.8]  # init loss weight: [bg, road]
    alpha = 0.1  # BCE_loss: IoU_loss = alpha: 1-alpha

    # Data related arguments
    num_class = 2
    num_workers = 2  # number of data loading workers
    siis_size = [32, 32]  # 32, 32
    input_size = [256, 256]  # final input size of network(random-crop use this)
    # zoom_size = [256, 256]   # input image(label) size
    crop_params = [250, 250, 125]  # [H, W, stride] only used by slide-crop
    crop_mode = 'slide'  # crop way of Val data, one of [random, slide]
    # random_flip = True  # if random filp images(labels) when training
    mean = [0.30663615, 0.34775069, 0.34375149]  # BGR, 此处的均值应该是0-1
    std = [0.5, 0.5, 0.5]  # [0.12283102, 0.1269429, 0.15580289]

    ont_hot = False  # Is the output of data_loader one_hot type

    # Misc arguments
    seed = 304
    print_freq = 20  # print info every N batch


opt = DefaultConfig()
