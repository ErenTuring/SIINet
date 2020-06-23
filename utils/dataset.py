'''
Dataset
'''
import collections
import os
# import time
import cv2
import numpy as np
import random
from PIL import Image
import torch
from torch.utils import data
from torchvision import transforms as T


class SegData(data.Dataset):
    '''
    Sematic segmentation dataset loader for pytorch
    Args:
        root(str) - root of dataset_dir
        split(str) - one of 'train', 'val', 'test' to contorl the mode
        opt - all options
        file_list(str) - path of train(val/test) files name list files,
                         [image_name,label_name]
        transform - Custom transform object(only can change the pixel value)
    '''

    def __init__(self, root, split, opt, file_list=None, transform=None):
        self.root = root
        self.split = split
        self.transform = transform
        self.input_size = opt.input_size
        self.crop_params = opt.crop_params  # for val&test mode
        self.crop_mode = opt.crop_mode  # 'slide' or 'random'
        self.num_classes = 2

        # Get files name and collect them into dict
        self.files = collections.defaultdict(list)  # just record files name list
        if split == 'test':
            if file_list is not None:
                with open(file_list, 'r') as fobj:
                    image_names = [x.strip() for x in fobj]  # only iamge name
            else:
                image_names = sorted(os.listdir(root+'/'+split))  # get image names
        else:
            if file_list is not None:
                with open(file_list, 'r') as fobj:
                    name_list = [x.strip().split(',') for x in fobj]
                    image_names = [x[0] for x in name_list]
                    label_names = [x[1] for x in name_list]
            else:
                image_names = sorted(os.listdir(root+'/'+split))  # get image names
                label_names = sorted(os.listdir(root+'/'+split+'_labels'))  # label
            self.files[split + '_labels'] = label_names
        self.files[split] = image_names

        # Transform
        if transform is None:
            normalize = T.Normalize(mean=opt.mean, std=opt.std)
            transform = [T.ToTensor(), normalize]  # basic transform
            if split == 'train':
                transform.insert(0, T.ColorJitter(
                    brightness=0.5, contrast=0.3, saturation=0.2, hue=0.05
                ))  # change shoudn't be too big
            if opt.input_size != opt.crop_params[:2]:
                transform.insert(0, T.Resize(tuple(opt.input_size)))
            # Compose
            self.transform = T.Compose(transform)

    def __len__(self):
        return len(self.files[self.split])
        # return int(len(self.files[self.split])*0.01)

    def load_image(self, index):
        ''' Load image(and label) by index '''
        img_name = self.files[self.split][index]
        img_path = self.root + '/' + self.split + '/' + img_name
        img = cv2.imread(img_path, 1)  # BGR 与预训练模型(如果包含的话)统一
        if self.split != 'test':
            lbl_name = self.files[self.split + '_labels'][index]
            lbl_path = self.root + '/' + self.split + '_labels/' + lbl_name
            lbl = cv2.imread(lbl_path, -1)  # TIFF is not support
            if len(lbl.shape) == 3:  # BGR!!!
                if lbl.shape[-1] == 4:
                    lbl = lbl[:, :, :3]
            else:
                ValueError('Wrong label image.')
            return img, lbl, img_name

        return img, img_name

    def join_transform(self, image, label):
        ''' Simple data augment function for train model,
        to random crop,flip and roate image and label,
        image,label are PIL image.
        '''
        # Random Flip
        f = [1, 0, -1, 2, 2][np.random.randint(0, 5)]  # [1, 0, -1, 2, 2]
        if f != 2:
            image, label = cv2.flip(image, f), cv2.flip(label, f)

        # Random Roate (Only 0, 90, 180, 270)
        k = np.random.randint(0, 4)  # [0, 1, 2, 3]
        image = np.rot90(image, k, (1, 0))  # clockwise
        label = np.rot90(label, k, (1, 0))

        image, label = self.join_random_resize_crop(image, label)

        return image, label

    def join_random_crop(self, image, label):
        crop_h, crop_w = self.input_size[0], self.input_size[1]
        if image.shape[:2] != label.shape[:2]:
            raise Exception('Image and label must have the same shape')
        x = np.random.randint(0, image.shape[0] - crop_h)  # row
        y = np.random.randint(0, image.shape[1] - crop_w)  # column
        return image[x:x + crop_h, y:y + crop_w], label[
            x:x + crop_h, y:y + crop_w]

    def resized_crop(self, image, i, j, h, w, size, interpolation=cv2.INTER_LINEAR):
        '''Crop the given PIL Image and resize it to desired size.
        Args:
            i: Upper pixel coordinate.
            j: Left pixel coordinate.
            h: Height of the cropped image.
            w: Width of the cropped image.
            size: (Height, Width) must be tuple
        '''
        image = image[i:i+h, j:j+w]
        image = cv2.resize(image, size[::-1], interpolation)
        return image

    def join_random_resize_crop(self, image, label,
                                scale=(0.3, 1.0), ratio=(3./4., 4./3.)):
        '''Crop the given Image(np.ndarray) to random size and aspect ratio,
        and finally resized to given size.
        Args:
            size: expected output size of each edge(default: 0.6 to 1.4 of original size)
            scale: range of size of the origin size cropped
            ratio: range of aspect ratio of the origin aspect ratio cropped(default: 3/4 to 4/3)
            interpolation: Default=PIL.Image.BILINEAR
        '''
        size = tuple(self.input_size)  # tar_height, tar_width
        H, W = image.shape[:2]  # ori_height, ori_width
        area = H*W

        for attempt in range(10):
            target_area = np.random.uniform(*scale) * area
            aspect_ratio = np.random.uniform(*ratio)

            w = int(round(np.sqrt(target_area * aspect_ratio)))
            h = int(round(np.sqrt(target_area / aspect_ratio)))

            if random.random() < 0.5:
                w, h = h, w

            if w <= W and h <= H:
                i = random.randint(0, H - h)  # crop start point(row/y)
                j = random.randint(0, W - w)  # crop start point(col/x)
                image = self.resized_crop(image, i, j, h, w, size, cv2.INTER_LINEAR)
                label = self.resized_crop(label, i, j, h, w, size, cv2.INTER_NEAREST)
                return image, label
        # Fallback
        w = min(w, H)
        i, j = (H - w) // 2, (W - w) // 2
        image = self.resized_crop(image, i, j, h, w, size, cv2.INTER_LINEAR)
        label = self.resized_crop(label, i, j, h, w, size, cv2.INTER_NEAREST)
        return image, label

    def slide_crop(self, image):
        '''Slide crop image(np.ndarray) into small piece, transfrom them at the same time,
        finally return list of sub_image(Tensor) and crop information.
        Return:
            image_list
            size: [y, x]
                y: Num of images in the row direction.
                x: Num of images in the col direction.
        '''
        crop_h, crop_w, stride = self.crop_params
        h, w = image.shape[0], image.shape[1]
        if (h-crop_h) % stride or (w-crop_w) % stride:
            ValueError(
                'Provided crop-parameters [%d, %d, %d]' % (crop_h, crop_w, stride)
                + 'cannot completely slide-crop the target image')

        image_list = []
        y = 0  # y-height
        for i in range((h-crop_h)//stride + 1):
            x = 0  # x-width
            for j in range((w-crop_w)//stride + 1):
                tmp_img = image[y:y+crop_h, x:x+crop_w].copy()
                tmp_img = self.transform(Image.fromarray(tmp_img))  # Array->PIL->Tensor
                image_list.append(tmp_img)
                x += stride
            y += stride
        size = [i+1, j+1]
        image_list = torch.stack(image_list)  # Tensor
        return image_list, size

    def need_crop(self, input_shape):
        '''Judge if the input image needs to be cropped'''
        crop_H = self.input_size[0]
        crop_W = self.input_size[1]
        return input_shape[0] > crop_H and input_shape[1] > crop_W


class MyDataset_1(SegData):
    ''' Dataset. '''
    def __getitem__(self, index):
        data_dict = dict()
        if self.split == 'test':
            img, img_name = SegData.load_image(self, index)  # PIL
            if SegData.need_crop(self, img.shape):
                img, data_dict['crop_info'] = SegData.slide_crop(self, img)
            else:
                img = self.transform(Image.fromarray(img))  # 需确保是PIL才能兼容transform操作
        else:
            img, lbl, img_name = SegData.load_image(self, index)  # PIL

            if self.split == 'train':
                img, lbl = SegData.join_transform(self, img, lbl)
                img = self.transform(Image.fromarray(img))
                data_dict['label'] = torch.from_numpy(lbl).long()
            else:  # val mode
                if SegData.need_crop(self, img.shape):
                    if self.crop_mode == 'random':
                        img, lbl = self.join_random_crop(img, lbl)
                        img = self.transform(Image.fromarray(img))
                    elif self.crop_mode == 'slide':
                        img, data_dict['crop_info'] = SegData.slide_crop(self, img)
                else:
                    img = self.transform(Image.fromarray(img))
                data_dict['label'] = lbl  # a [HW] np.ndarray

        data_dict['image'], data_dict['name'] = img, img_name
        return data_dict


class MyDataset_2(SegData):
    ''' Dataset. '''
    def __getitem__(self, index):
        data_dict = dict()
        if self.split == 'test':
            img, img_name = SegData.load_image(self, index)  # PIL
            H, W = img.shape[:2]
            img = cv2.resize(img, (W//2, H//2))  # TODO: For roadtracer dataset to make sure performance
            if SegData.need_crop(self, img.shape):
                img, data_dict['crop_info'] = SegData.slide_crop(self, img)
            else:
                img = self.transform(Image.fromarray(img))  # 需确保是PIL才能兼容transform操作
        else:
            img, lbl, img_name = SegData.load_image(self, index)  # PIL

            if self.split == 'train':
                img, lbl = SegData.join_transform(self, img, lbl)
                img = self.transform(Image.fromarray(img))
                data_dict['label'] = torch.from_numpy(lbl).long()
            else:  # val mode
                if SegData.need_crop(self, img.shape):
                    if self.crop_mode == 'random':
                        img, lbl = self.join_random_crop(img, lbl)
                        img = self.transform(Image.fromarray(img))
                    elif self.crop_mode == 'slide':
                        img, data_dict['crop_info'] = SegData.slide_crop(self, img)
                else:
                    img = self.transform(Image.fromarray(img))
                data_dict['label'] = lbl  # a [HW] np.ndarray

        data_dict['image'], data_dict['name'] = img, img_name
        return data_dict


class Train_Dataset(SegData):
    def __getitem__(self, index):
        ''' simple datset for traning '''
        img, lbl, img_name = SegData.load_image(self, index)
        # lbl = one_hot_1(lbl, self.num_classes)
        # 功能：crop, zoom, color_label -> class_label
        img, lbl = SegData.join_transform(self, img, lbl)
        img = self.transform(Image.fromarray(img))
        lbl = torch.from_numpy(lbl).long()
        return {'image': img, 'label': lbl, 'name': img_name}
