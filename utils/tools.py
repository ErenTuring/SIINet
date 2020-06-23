'''
Pytorch basic tools for main.
'''
import datetime
import os
import sys
import re
import cv2
import numpy as np
import torch
import torch.nn.functional as F


def TTA(net, image, mode='cls'):
    """
    Do test time augmentations on single image for classification or segmentation.
    Note: Only suport single image per time!
    Args:
        image: [N, C, H, W] tensor of image (have transformed)
        mode: 'cls' for classification, 'seg' for segmentation.

    """
    # predict a complete image
    aug_imgs = []
    for i in range(4):
        aug_imgs.append(torch.rot90(image.clone(), i, dims=(3, 2)))
    aug_imgs.append(torch.flip(image.clone(), [2]))  # filp H
    aug_imgs.append(torch.flip(image.clone(), [3]))  # filp W
    aug_imgs = torch.cat(aug_imgs, dim=0)
    outputs = net(aug_imgs)
    if mode == 'cls':
        # outputs: [NC]
        predict = outputs.mean(dim=0, keepdim=True)
    elif mode == 'seg':
        # outputs: [NCHW]
        predict = torch.flip(outputs[5, None].clone(), [3])
        predict += torch.flip(outputs[4, None].clone(), [2])
        for i in range(4):
            predict += torch.rot90(outputs[i, None].clone(), i, dims=(2, 3))

    return predict


def net_predict(net, image, opt, crop_info=None):
    '''Do predict use Net(only one image at a time).
    Args:
        net: The trained model
        image: Image to be predicted
        opt: An instance of config class(including size, crop_params, etc.)
        crop_info: [rows, cols]
            rows: Num of images in the row direction.
            cols: Num of images in the col direction.
    Return:
        predict: [HWC] array, where C is the num of classes
    '''
    if crop_info is None:
        # predict a complete image
        if opt.use_gpu:
            image = image.cuda()
        output = net(image)[0]  # [NCHW] -> [CHW]
        predict = np.transpose(output.cpu().detach().numpy(), (1, 2, 0))  # [CHW]->[HWC]
    else:
        # predict the list of croped images
        predict = []
        image.transpose_(0, 1)  # [NLCHW] -> [LNCHW]
        for input in image:
            if opt.use_gpu:
                input = input.cuda()  # [NCHW](N=1)
            output = net(input)[0]  # [NCHW] -> [CHW]
            output = np.transpose(output.cpu().detach().numpy(), (1, 2, 0))  # [CHW]->[HWC]
            if opt.input_size != opt.crop_params[:2]:
                output = cv2.resize(output, tuple(opt.crop_params[:2][::-1]), 1)
            predict.append(output)
        predict = vote_combine(predict, opt.crop_params, crop_info, 2)

    predict = F.softmax(F.normalize(torch.from_numpy(predict), dim=-1), dim=-1)
    return predict.numpy()  # [HWC] array


def net_predict_enhance(net, image, opt, crop_info=None):
    '''Do predict use Net with some trick(only one image at a time).
    Args:
        net: The trained model
        image: Image to be predicted
        opt: An instance of config class(including size, crop_params, etc.)
        crop_info: [rows, cols]
            rows: Num of images in the row direction.
            cols: Num of images in the col direction.
    Return:
        predict: [HWC] array, where C is the num of classes
    '''
    predict_list = []
    if crop_info is None:
        # predict a complete image
        for i in range(4):
            input = torch.from_numpy(np.rot90(image, i, axes=(3, 2)).copy())
            if opt.use_gpu:
                input = input.cuda()
            output = net(input)[0]  # [NCHW] -> [CHW]
            output = output.cpu().detach().numpy()  # Tensor -> array
            output = np.transpose(output, (1, 2, 0))  # [CHW]->[HWC]
            predict_list.append(np.rot90(output, i, axes=(0, 1)))  # counter-clockwise rotation

    else:
        # predict the list of croped images
        image.transpose_(0, 1)  # [NLCHW] -> [LNCHW]
        for i in range(4):
            predict = []
            for img in image:
                input = torch.from_numpy(np.rot90(img, i, axes=(3, 2)).copy())
                if opt.use_gpu:
                    input = input.cuda()  # [NCHW](N=1)
                output = net(input)[0]  # [NCHW] -> [CHW]
                output = output.cpu().detach().numpy()
                output = np.transpose(output, (1, 2, 0))  # [CHW]->[HWC]
                if opt.input_size != opt.crop_params[:2]:
                    output = cv2.resize(output, tuple(opt.crop_params[:2][::-1]), 1)
                predict.append(np.rot90(output, i, axes=(0, 1)))
            predict_list.append(vote_combine(predict, opt.crop_params, crop_info, 2))

    predict = predict_list[0]
    for i in range(1, 4):
        predict += predict_list[i]

    # 自适应归一化 TODO: 失败
    # m = np.max(predict, axis=-1, keepdims=True).repeat(2, -1)
    # n = np.min(predict, axis=-1, keepdims=True).repeat(2, -1)
    # predict = predict / (m - n)

    predict = F.softmax(F.normalize(torch.from_numpy(predict), dim=-1), dim=-1)
    return predict.numpy()  # [HWC] array


def net_predict_tta(net, image, opt, crop_info=None):
    '''Do predict use Net(only one image at a time).
    Args:
        net: The trained model
        image: Image to be predicted
        opt: An instance of config class(including size, crop_params, etc.)
        crop_info: [rows, cols]
            rows: Num of images in the row direction.
            cols: Num of images in the col direction.
    Return:
        predict: [HWC] array, where C is the num of classes
    '''
    if crop_info is None:
        # predict a complete image
        if opt.use_gpu:
            image = image.cuda()
        output = TTA(net, image, 'seg')[0]  # [NCHW] -> [CHW]
        predict = np.transpose(output.cpu().detach().numpy(), (1, 2, 0))  # [CHW]->[HWC]
    else:
        # predict the list of croped images
        predict = []
        image.transpose_(0, 1)  # [NLCHW] -> [LNCHW]
        for input in image:
            if opt.use_gpu:
                input = input.cuda()  # [NCHW](N=1)
            output = TTA(net, input, 'seg')[0]  # [NCHW] -> [CHW]
            output = np.transpose(output.cpu().detach().numpy(), (1, 2, 0))  # [CHW]->[HWC]
            if opt.input_size != opt.crop_params[:2]:
                output = cv2.resize(output, tuple(opt.crop_params[:2][::-1]), 1)
            predict.append(output)
        predict = vote_combine(predict, opt.crop_params, crop_info, 2)

    predict = F.softmax(F.normalize(torch.from_numpy(predict), dim=-1), dim=-1)
    return predict.numpy()  # [HWC] array


def vote_combine(label, crop_params, crop_info, mode, scale=False):
    '''
    Combine small scale predicted label into a big one,
    for 1.classification or 2.semantic segmantation.
    Args:
        label: One_hot label(may be tensor). 1.[NC]; 2.[NHWC]
        crop_params: [sub_h, sub_w, crop_stride]
        crop_info: [rows, cols]
            rows: Num of images in the row direction.
            cols: Num of images in the col direction.
        mode: 1-classification, 2-semantic segmantation.
        scale: If do adaptive normalize.
    Returns:
        out_label: [HWC] one hot array, uint8.
    '''
    rows, cols = crop_info
    h, w, stride = crop_params
    label = np.array(label)  # Tensor -> Array
    if mode == 1:
        # 1. For classification
        if len(label.shape) == 3:  # list转array后会多出一维
            label = np.squeeze(label)
    elif mode == 2:
        # 2. For semantic segmantation
        if len(label.shape) == 5:  # in case of label is [1NHWC]
            label = label[0]
    else:
        return ValueError('Incorrect mode!')
    out_h, out_w = (rows-1)*stride+h, (cols-1)*stride+w
    out_label = np.zeros([out_h, out_w, label.shape[-1]], dtype=label.dtype)  # [HWC]

    y = 0  # y=h=row
    for i in range(rows):
        x = 0  # x=w=col
        for j in range(cols):
            out_label[y:y+h, x:x+w] += label[i*cols+j]  # 此处融合用加法
            x += stride
        y += stride

    # 自适应归一化
    if scale:
        m = np.max(out_label, axis=-1, keepdims=True).repeat(2, -1)
        n = np.min(out_label, axis=-1, keepdims=True).repeat(2, -1)
        out_label = out_label / (m - n)
    return out_label  # np.uint8(out_label)


# **************************************************
# ***************** Training ***********************
# **************************************************
def adjust_lr(lr, factor, optimizer, vis=None):
    ''' Update learning rate per epoch.
    '''
    if factor is not None:
        new_lr = lr * factor
        for param_group in optimizer.param_groups:
            param_group['lr'] = new_lr

    if factor < 0.9:
        print('\nupdate learning rate: %f -> %f' % (lr, new_lr))
        if vis is not None:
            vis.log('\nupdate learning rate: %f -> %f' % (lr, new_lr))
    return new_lr


def load_ckpt(model, ckpt, is_training=False):
    ''' Load ckpt and setup some state. '''
    print("Trying to load model")
    if os.path.isfile(ckpt):
        model.load_state_dict(torch.load(ckpt))
        print('Load %s' % ckpt)
    else:
        if is_training:
            print('Checkpoint file is not exist, re-initializtion.')
        else:
            raise ValueError('Failed to load model checkpoint.')
    return model


def save_ckpt(state, is_best, ckpt='cp', filename='checkpoint.pth.tar'):
    ''' Save checkpoint.
    Args:
        ckpt - Dir of ckpt to save.
        filename - Only name of ckpt to save.
    Note: 这种模式下建议以直接模型名字作为filename，其中不带epoch等信息。
    '''
    import shutil
    filepath = ckpt+'/'+filename+'.pth.tar'
    torch.save(state, filepath)
    if is_best:
        shutil.copyfile(filepath, ckpt+'/'+filename+'_best.pth.tar')


def setup_mode(if_gpu, gpu_id='0', cudnn_benchmark=True):
    ''' Set up basic mode for training. '''
    if torch.cuda.is_available() and if_gpu:
        from torch.backends import cudnn
        # 让内置的cuDNN的auto-tuner自动寻找最适合当前配置的高效算法，以优化效率
        cudnn.benchmark = cudnn_benchmark

        os.environ["CUDA_VISIBLE_DEVICES"] = gpu_id  # 指定运行GPU
        device = torch.device("cuda")
        torch.cuda.current_device()
        pin_memory = True
        print('GPU Mode')
    else:
        device = torch.device("cpu")
        pin_memory = False
        print('CPU Mode')

    return device, pin_memory


def data_generator(root, split='train'):
    '''Simple data generator to return pairs of image and label and img_name.'''
    img_names = sorted(os.listdir(root+'/'+split))
    lbl_names = sorted(os.listdir(root+'/'+split+'_labels'))

    def data():
        for img_name, lbl_name in zip(img_names, lbl_names):
            image = cv2.imread(root+'/'+split+'/'+img_name, -1)
            label = cv2.imread(root+'/'+split+'_labels/'+lbl_name, -1)
            yield image, label, img_name

    return data(), len(img_names)


def rename_file(file_name, ifID=0, addstr=None, extension=None):
    '''Rename a file.
    Args:
        file_name: The name/path of file.
        ifID: 1 - only keep the number(ID) in old_name
            Carefully! if only keep ID, file_name can't be path.
        addstr: The addition str add between name and extension
        extension: Set the new extension(kind of image, such as: 'png').
    '''
    savename, extn = os.path.splitext(file_name)  # extn content '.'
    if ifID:
        # file_path = os.path.dirname(full_name)
        ID_nums = re.findall(r"\d+", savename)
        ID_str = str(ID_nums[0])
        for i in range(len(ID_nums)-1):
            ID_str += ('_'+(ID_nums[i+1]))
        savename = ID_str

    if addstr is not None:
        savename += '_' + addstr

    if extension is not None:
        extn = '.' + extension

    return savename + extn


def train_log(X, f=None):
    ''' Print with time. To console or a file(f) '''
    time_stamp = datetime.datetime.now().strftime("[%d %H:%M:%S]")
    if not f:
        sys.stdout.write(time_stamp + " " + X)
        sys.stdout.flush()
    else:
        f.write(time_stamp + " " + X)


def compute_class_iou(pred, gt, num_classes):
    '''
    Args:
        pred: Predict label [HW].
        gt: Ground truth label [HW].
    Return:
        （每一类的）intersection and union list.
    '''
    intersection = np.zeros(num_classes)
    union = np.zeros(num_classes)
    for i in range(num_classes):
        pred_i = pred == i
        label_i = gt == i
        intersection[i] = float(np.sum(np.logical_and(label_i, pred_i)))
        union[i] = float(np.sum(np.logical_or(label_i, pred_i)) + 1e-8)
    class_iou = intersection / union
    return class_iou


def colour_code_label(label, label_values, add_image=None, save_path=None):
    '''
    Given a [HW] array of class keys(or one hot[HWC]), colour code the label;
    also can weight the colour coded label and image, maybe save the final result.

    Args:
        label: single channel array where each value represents the class key.
        label_values
    Returns:
        Colour coded label or just save image return none.
    '''
    label, colour_codes = np.array(label), np.array(label_values)
    if len(label) == 3:
        label = np.argmax(label, axis=2)  # [HWC] -> [HW]
    color_label = colour_codes[label.astype(int)]
    color_label = color_label.astype(np.uint8)

    if add_image is not None:
        if add_image.shape != color_label.shape:
            cv2.resize(color_label, (add_image.shape[1], add_image.shape[0]),
                       interpolation=cv2.INTER_NEAREST)
        add_image = cv2.addWeighted(add_image, 0.7, color_label, 0.3, 0)
        if save_path is None:
            return color_label, add_image

    if save_path is not None:
        cv2.imwrite(save_path, color_label, [1, 100])
        if add_image is not None:
            cv2.imwrite(rename_file(save_path, addstr='mask'), add_image, [1, 100])
        return  # no need to return label if saved

    return color_label
