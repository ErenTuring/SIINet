'''
Comprehensive image operation
'''
import os

import cv2
import numpy as np
from skimage import morphology
from tqdm import tqdm

rate_threshold = 0.1


# **********************************************
# ********* Function Implementation ************
# **********************************************
def judge(label, num_classes=0):
    '''
    Judge if one class's protion is zero, don't support RGB label
    Args:
        label: 1. [HW]; 2. [HW1]; 3. one_hot-[HWC]; 4. rgb-[HWC]- not support rbg!!!
            Note: 1&2 class_num need to be specified!!!
        num_classes: default=0, only one_hot label don't have to specify.
    Returns:
        True or False
    '''
    sum_area = label.shape[0] * label.shape[1]  # h*w
    if len(label.shape) == 2 or label.shape[-1] == 1:  # 1,2
        for i in range(num_classes):
            indArr = [label == i]
            rate = np.sum(indArr) / sum_area
            if rate < rate_threshold:
                return False
    else:  # [HWC](one_hot) 暂无对rgb的支持
        num_classes = label.shape[-1]
        for i in range(num_classes):
            rate = np.sum(label[:, :, i]) / sum_area
            if rate < rate_threshold:
                return False

    return True


def crop_data_smartly(flaglabel, class_num=0, crop_params=[256, 256, 128]):
    '''
    Crop image and label and judge the class stitudation of per sub label
    Args:
        flaglabel: An [HW] or [HWC] array of label used to judge
        class_num: num of classes
    Return:
        box_list: A tuple of crop index: (y0, y1, x0, x1), img=img[y0:y1, x0:x1]
    '''
    h, w = flaglabel.shape[0], flaglabel.shape[1]
    crop_h, crop_w, stride = crop_params

    box_list = []  # (y0, y1, x0, x1)

    y = 0  # y-h
    for i in range((h-crop_h)//stride + 1):
        x = 0  # x-w
        for j in range((w-crop_w)//stride + 1):
            tp_label = flaglabel[y:y+crop_h, x:x+crop_w].copy()
            # 判断各类占比
            if judge(tp_label, class_num):
                # sub_label = label[y:y+crop_h, x:x+crop_w]
                box = (y, y+crop_h, x, x+crop_w)
                box_list.append(box)
            x += stride
        y += stride
    return box_list


def filelist(floder_dir, ifPath=False, extension=None):
    '''
    Get names(or whole path) of all files(with specify extension)
    in the floder_dir and return as a list.

    Args:
        floder_dir: The dir of the floder_dir.
        ifPath:
            True - Return whole path of files.
            False - Only return name of files.(Defualt)
        extension: Specify extension to only get that kind of file names.

    Returns:
        namelist: Name(or path) list of all files(with specify extension)
    '''
    namelist = sorted(os.listdir(floder_dir))

    if ifPath:
        for i in range(len(namelist)):
            namelist[i] = os.path.join(floder_dir, namelist[i])

    if extension is not None:
        n = len(namelist)-1  # orignal len of namelist
        for i in range(len(namelist)):
            if not namelist[n-i].endswith(extension):
                namelist.remove(namelist[n-i])  # discard the files with other extension

    return namelist


# **********************************************
# ************** Main function *****************
# **********************************************
def data_crop():
    '''
    Crop dataset(both images and labels) in some way.
    (当前是判断各类地物占比裁剪)
    '''
    # Basic setting
    data_dir = '/home/tao/Data/cvpr_road/1024/filtered_img'
    split = 'train'
    class_num = 2
    crop_params = [512, 512, 128]

    # Primary preparation work.
    global_num = 1  # 作为结果存储的编号
    image_names = filelist(data_dir+'/%s' % split, ifPath=True)
    label_names = filelist(data_dir+'/%s_labels' % split, ifPath=True)
    img_out_dir = data_dir+"/%s_out" % split
    label_out_dir = data_dir + "/%s_labels_out" % split
    if not os.path.exists(img_out_dir):
        os.mkdir(img_out_dir)
    if not os.path.exists(label_out_dir):
        os.mkdir(label_out_dir)

    # Judge crop and save the sub_images that meet the conditions
    for i in tqdm(range(len(image_names))):
        image = cv2.imread(image_names[i], -1)
        label = cv2.imread(label_names[i], -1)
        boxes = crop_data_smartly(label, class_num, crop_params)
        if boxes is None:
            continue  # 如果所有的子图都不满足条件，则跳过
        # Save them
        for (y0, y1, x0, x1) in boxes:
            img = image[y0:y1, x0:x1].copy()
            lbl = label[y0:y1, x0:x1].copy()

            # TODO: Only treat with RBDD dataset
            # equality = np.equal(img, [255, 255, 255])
            # eq_map = np.all(equality, axis=-1)  # 提取图像空白部分
            # ratio = np.sum(eq_map)/(img.shape[0]*img.shape[1])
            # if ratio > 0.04:
            #     continue  # 如果空白部分过大则去除此部分
            # elif ratio > 0.001:
            #     eq_map = morphology.remove_small_objects(eq_map, 20, connectivity=1)
            #     eq_map = morphology.dilation(eq_map, morphology.disk(3))
            #     lbl[eq_map] = 0  # 图像空白区域对应的标签设置为BG

            cv2.imwrite(img_out_dir+'/%.6d.jpg' % (global_num), img, [1, 100])
            cv2.imwrite(label_out_dir+'/%.6d.png' % (global_num), lbl)
            global_num += 1
    print("Finish, sum of %d pictures." % (global_num))


if __name__ == '__main__':
    # main()
    data_crop()

    pass
