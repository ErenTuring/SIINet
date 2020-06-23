# -*- coding:utf-8 -*-
'''
Do evaluation.
'''
import cv2
import os
import numpy as np

TRAIN_REGIONS = [
    'indianapolis', 'louisville', 'columbus', 'milwaukee', 'minneapolis',
    'seattle', 'portland', 'sf', 'san antonio', 'vegas', 'phoenix', 'dallas',
    'austin', 'san jose', 'houston', 'miami', 'tampa', 'orlando', 'atlanta',
    'st louis', 'nashville', 'dc', 'baltimore', 'philadelphia', 'london'
]
TEST_REGIONS = [
    'boston', 'new york', 'chicago', 'la', 'toronto', 'denver', 'kansas city',
    'san diego', 'pittsburgh', 'montreal', 'vancouver', 'tokyo', 'saltlakecity',
    'paris', 'amsterdam']


def crops():
    print('crops')
    import os
    import cv2
    from tqdm import tqdm
    from qjdltools import dlimage, fileop

    root = '/home/tao/Data/Seg/roadtracer_org/'
    split = ['train', 'val'][0]
    mode = 2
    in_dir = root + ['imagery', 'mask', 'mask_vis'][mode]
    out_dir = root + '%s%s' % (split, ['', '_labels', '_labels_vis'][mode])
    REGIONS = {
        'train': TRAIN_REGIONS,
        'val': ['chicago', 'toronto', 'saltlakecity', 'boston']
        # 'val': ['chicago', 'toronto', 'saltlakecity', 'boston']
    }

    if not os.path.exists(out_dir):
        os.mkdir(out_dir)

    # crop_params = [512, 512, 512] if split == 'val' else [512, 512, 256]
    crop_params = [2048, 2048, 2048]
    image_names = sorted(os.listdir(in_dir))
    for name in tqdm(image_names):
        if name.split('_')[0] not in REGIONS[split]:
            continue
        image = cv2.imread(in_dir+'/'+name, -1)
        # 附加操作
        if 'labels' in out_dir:
            inter = cv2.INTER_NEAREST
        else:
            inter = cv2.INTER_LINEAR
        image = cv2.resize(image, (2048, 2048), inter)
        imgs, _ = dlimage.slide_crop(image, crop_params, False)
        for i in range(len(imgs)):
            save_name = fileop.rename_file(name, addstr="_%d" % (i+1))
            cv2.imwrite(out_dir+'/'+save_name, imgs[i], [1, 100])


def code_lbls():
    import os
    import cv2
    from tqdm import tqdm

    root = '/home/tao/Data/Seg/roadtracer_org/'
    in_dir = root + 'mask_vis'  # 'val_labels_vis'
    out_dir = root + 'mask'  # 'val_labels'
    if not os.path.exists(out_dir):
        os.mkdir(out_dir)

    image_names = sorted(os.listdir(in_dir))
    for name in tqdm(image_names):
        image = cv2.imread(in_dir+'/'+name, -1)
        # image = image[:, :, 0]
        image[image > 0] = 1
        cv2.imwrite(out_dir+'/'+name, image, [1, 100])


def expand_lines(in_dir, out_dir, width=8):
    if not os.path.exists(out_dir):
        os.mkdir(out_dir)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (width, width))
    lbl_names = sorted(os.listdir(in_dir))
    for name in lbl_names:
        lbl = cv2.imread(in_dir + '/' + name, 0)
        lbl = cv2.dilate(lbl, kernel)
        # lbl = np.repeat(np.expand_dims(lbl, axis=2), 3, axis=2)
        cv2.imwrite(out_dir + "/" + name, lbl)


def certeral_crop(in_dir):
    img_names = sorted(os.listdir(in_dir))
    for img_name in img_names:
        img = cv2.imread(in_dir+'/'+img_name, -1)
        img = img[100:1400, 100:1400, :]
        cv2.imwrite(in_dir + "/" + img_name, img)


def vis_result(lbl_dir, out_dir, addstr=None):
    import os
    import cv2
    from tqdm import tqdm

    # traget_regions = ['boston', 'chicago', 'saltlakecity', 'toronto', 'denver', 'kansas']
    traget_regions = ['la']

    img_dir = 'D:/Data/mit_result/test_patch'

    if not os.path.exists(out_dir):
        os.mkdir(out_dir)
    img_names = sorted(os.listdir(img_dir))
    lbl_names = sorted(os.listdir(lbl_dir))

    # Execution operations
    for (img_name, lbl_name) in tqdm(zip(img_names, lbl_names)):
        if img_name.split('_')[0] not in traget_regions:
            continue
        image = cv2.imread(img_dir+'/'+img_name, -1)
        label = cv2.imread(lbl_dir+'/'+lbl_name, -1)
        assert image.shape[:2] == label.shape[:2]
        # Operatrions
        # BGR: (0, 255, 255)
        if len(label.shape) == 2:
            label = np.stack([label, label, label], axis=2)
        label[:, :, 0] = 0
        fusing_img = cv2.addWeighted(image, 0.7, label, 0.5, 0)

        if addstr is not None:
            savename, extn = os.path.splitext(lbl_name)
            lbl_name = savename + '_' + addstr + extn
        cv2.imwrite(out_dir+'/'+lbl_name, fusing_img)


def num_of_break(predict, target):
    """ Count the number of break in the RoadTracer dataset.
    The roadtracer dataset is too large to accurately count the number of road fractures,
    so we estimate the number of road fractures by automatic method.
    """
    # from skimage import morphology, measure

    pre, gt = predict.copy().astype(np.int16), target.copy().astype(np.int16)
    pre = cv2.dilate(pre, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (20, 20)))
    diff = gt - pre  # 缺口都为真（255）
    diff[diff < 0] = 0
    diff = diff.astype(np.uint8)

    _, bin_diff = cv2.threshold(diff, 127, 255, cv2.THRESH_BINARY)
    bin_diff = cv2.erode(bin_diff, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (16, 16)))
    bin_diff = cv2.dilate(bin_diff, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (30, 30)))

    cnts, _ = cv2.findContours(bin_diff, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    break_num = len(cnts)

    # diff = morphology.binary_erosion(diff, morphology.disk(7))
    # diff = morphology.binary_dilation(diff, morphology.disk(20))

    # ind_mat = measure.label(diff, connectivity=1)
    # break_num = np.max(ind_mat)

    # cv2.imwrite('debug.jpg', diff)

    return break_num


def val(lbl_dir, gt_dir):
    """ 此处读取的为0-255的label """
    from utils import evaluation

    runingscore = evaluation.RoadExtractionScore(2)
    # runingscore = evaluation.RelaxedRoadExtractionScore(3)
    # print("(relax)")

    lbl_names = sorted(os.listdir(lbl_dir))
    gt_names = sorted(os.listdir(gt_dir))
    # Execution operations
    nrb = 0
    for (lbl_name, gt_name) in zip(lbl_names, gt_names):
        predict = cv2.imread(lbl_dir+'/'+lbl_name, 0)
        gt = cv2.imread(gt_dir+'/'+gt_name, 0)
        # predict = predict[200:-200, 200:-200]
        # gt = gt[200:-200, 200:-200]
        # cv2.imshow('', cv2.resize(predict, (500, 500)))
        # cv2.waitKey(0)
        assert predict.shape[:2] == gt.shape[:2]
        nrb += num_of_break(predict, gt)

        predict[predict > 0] = 1
        gt[gt > 0] = 1
        runingscore.add(gt, predict)
    mean_nrb = nrb / len(lbl_names)
    print('Mean NRB: %.2f' % mean_nrb)
    runingscore.print_score(runingscore.get_scores())  # print total scores


def de_merge(lbl_dir, out_dir):
    import os
    import cv2
    from tqdm import tqdm

    if not os.path.exists(out_dir):
        os.mkdir(out_dir)

    lbl_names = sorted(os.listdir(lbl_dir))

    # Execution operations
    for lbl_name in tqdm(lbl_names):
        region = lbl_name.split('.')[0]
        label = cv2.imread(lbl_dir+'/'+lbl_name, -1)
        # Split
        # if region in 'boston':
        out_lbl = label[:4096, :4096]
        cv2.imwrite(out_dir+'/%s_0_0_osm.png' % region, out_lbl)
        out_lbl = label[4096:, :4096]
        cv2.imwrite(out_dir+'/%s_0_1_osm.png' % region, out_lbl)
        out_lbl = label[:4096, 4096:]
        cv2.imwrite(out_dir+'/%s_1_0_osm.png' % region, out_lbl)
        out_lbl = label[4096:, 4096:]
        cv2.imwrite(out_dir+'/%s_1_1_osm.png' % region, out_lbl)


def main():
    print('main')
    root = '/home/tao/Data/Seg/roadtracer_org/test_all'
    # in_dir = root + '/mask_org'
    # out_dir = root + '/mask_vis'
    # expand_lines(in_dir, out_dir, 11)

    # root = '/home/tao/Data/Seg/roadtracer_org'
    dataset = {
        1: 'gt', 2: 'rt', 3: 'sii_net_102',
        4: '60020_36', 5: '7', 6: '7s', 7: '8', 8: '9', 9: '9s',
        10: 'conv1d', 11: 'sii_net_101', 12: 'sii_net_102',
        13: '60020'
        # 10: 'conv1d', 11: 'SIINet_theta0', 12: 'SIINet_theta01',
    }[6]
    print('*'*50 + '\n' + dataset)
    # datasets = ['sii_net_102', '60020_36', '7', '8', '9']
    # datasets = ['sii_net_102', 'rt', 'gt']
    in_dir = root + '/out_masks/%s' % dataset
    out_dir = root + '/out_masks/%s_expand' % dataset
    expand_lines(in_dir, out_dir, 8)

    lbl_dir = root + '/out_masks/%s_expand' % dataset
    gt_dir = root + '/out_masks/gt_expand'
    val(lbl_dir, gt_dir)

    # certeral_crop('C:/Data/Syns_Data/Result/Roadtracer/test')

    # Visual
    # for ds in datasets:
    #     print('*'*50 + '\n' + ds)
    #     root = r'D:/Data/mit_result'

    #     # lbl_dir = root + '/%s_pre' % ds
    #     # out_dir = root + '/%s_patch2' % ds
    #     # de_merge(lbl_dir, out_dir)

    #     lbl_dir = root + '/%s_patch' % ds
    #     # lbl_dir = root + '/%s_patch2' % ds
    #     out_dir = root + '/all_vis'
    #     # out_dir = root + '/all_vis2'
    #     vis_result(lbl_dir, out_dir, ds)

    pass


if __name__ == "__main__":
    main()
    # crops()
    # code_lbls()
    # lbl_dir = "/home/tao/Data/Seg/roadtracer_org/test_all/out_masks/test_expand"
    # gt_dir = "/home/tao/Data/Seg/roadtracer_org/test_all/out_masks/gt_test_expand"
    # val(lbl_dir, gt_dir)

    # vis_result()
    pass
