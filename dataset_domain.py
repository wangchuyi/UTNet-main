import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torch.utils.data import Dataset, DataLoader
import pandas as pd
import SimpleITK as sitk
from skimage.measure import label, regionprops
import math
import pdb
import cv2
from tqdm import tqdm
from collections import defaultdict
from data_aug import makeAug

from utils.utils import *
from torch.utils import data


class CMRDataset(Dataset):
    def __init__(self, config, dataset_dir, mode='train', is_debug=False):
        # 基础参数
        self.config = config
        self.mode = mode  # 模式包含’train‘、’test‘、’debug‘
        self.dataset_dir = dataset_dir
        self.crop_size = config.crop_size


        # 配置数据增强
        if self.mode == 'train':
            self.transform = makeAug(config, mode)
        elif self.mode == 'test':
            self.transform = makeAug(config, mode)

        # 中间参数
        self.img_key_name_list = []
        self.img_key_to_path = defaultdict(str)  # str: key 值为 img_key_name
        self.label_key_to_path = defaultdict(str)  # str: key 值为 img_key_name
        self.has_label = {}  # BOOL: key 值为 img_key_name
        self.norm_thresh = {}  # float: key 值为 case_day

        # 根据模式选择数据路径
        if self.mode == 'train':
            self.pre_face = 'Training'
        elif self.mode == 'test':
            self.pre_face = 'Testing'
        if is_debug:
            self.pre_face = 'debug'

        path = os.path.join(self.dataset_dir, self.pre_face + '/')
        print('start loading data')

        # 根据文件目录读取训练图片
        case_list = os.listdir(path)  # ['case2', 'case6', ...]
        for case in tqdm(case_list):
            case_day_list = os.listdir(path + case)  # ['case2_day1', 'case2_day2', ...]
            for case_day in case_day_list:

                # 读取每天所有的 slice 计算归一化阈值
                imgs_per_day = []

                scans_path = os.path.join(path, case, case_day, 'scans')
                scans_list = os.listdir(scans_path)  # ['slice_0001_266_266_1.50_1.50.png', ...]
                for scan_name in scans_list:
                    _, idx, img_height, img_width, pixel_height, pixel_width = scan_name.split('_')
                    pixel_width = pixel_width[:-4]  # delete '.png'

                    # 生成诸如 'case2_day1_slice_0001' 的key name
                    img_key_name = case_day + '_' + scan_name[:10]
                    self.img_key_name_list.append(img_key_name)

                    # 保存所有 slice 和 label 的路径
                    img_path = os.path.join(scans_path, scan_name)
                    img = self.read_image(img_path)
                    self.img_key_to_path[img_key_name] = img_path

                    label_name = img_key_name + '.png'
                    label_path = os.path.join(self.dataset_dir, 'label', label_name)
                    if os.path.exists(label_path):
                        self.has_label[img_key_name] = True
                        self.label_key_to_path[img_key_name] = label_path
                    else:
                        self.has_label[img_key_name] = False

                    # 计算每天的多张slice归一化阈值
                    imgs_per_day.append(img)

                self.cal_norm_thresh(case_day, imgs_per_day)

        print('load done, length of dataset:', len(self.img_key_name_list))

    def __len__(self):
        return len(self.img_key_name_list)

    def __getitem__(self, idx):
        img_key_name = self.img_key_name_list[idx]
        img_key_part = img_key_name.split("_")  # 'case2', 'day1', 'slice', '0001'
        tensor_image, tensor_label,  tensor_lab_mid = self.get_img_label(img_key_part)
        return tensor_image, tensor_label, tensor_lab_mid, img_key_name

    def get_img_label(self, img_key_part):
        train_dimension = self.config.train_dimension
        USE_3C = self.config.USE_3C

        # 生成3d图像、标签列表
        slice_num = int(img_key_part[3])
        slice_range = [slice_num - 2, slice_num - 1, slice_num, slice_num + 1, slice_num + 2]
        slice_img = [self.load_slice(img_key_part.copy(), i, load_img=True) for i in slice_range]
        slice_label = [self.load_slice(img_key_part.copy(), i, load_img=False) for i in slice_range]

        # 处理空图像与空标签
        if slice_img[3] is None:
            slice_img[3] = slice_img[2]
        if slice_img[4] is None:
            slice_img[4] = slice_img[3]
        if slice_img[1] is None:
            slice_img[1] = slice_img[2]
        if slice_img[0] is None:
            slice_img[0] = slice_img[1]

        for idx, label in enumerate(slice_label):
            if label is None:
                slice_label[idx] = np.zeros((self.config.crop_size, self.config.crop_size, 3))

        # 是否使用三通道标签
        if not USE_3C:
            slice_label = [self.change_3c_to_1c(label) for label in slice_label]

        # 训练维度
        if train_dimension == '3d':
            pass
        elif train_dimension == '2.5d':
            slice_label = [slice_label[2]]
        elif train_dimension == '1d':
            slice_img = [slice_img[2]]
            slice_label = [slice_label[2]]
        else:
            raise NotImplementedError

        case_day = '_'.join(img_key_part[:2])
        img_key_name = "_".join(img_key_part)

        tensor_image, tensor_label, tensor_lab_mid = self.preprocess(slice_img, slice_label, case_day)

        return tensor_image, tensor_label, tensor_lab_mid

    def preprocess(self, slice_img, slice_label, case_day):
        img = np.stack(slice_img, axis=2)

        label_mid = 0
        if self.config.train_dimension == '3d':
            label_mid = slice_label[2]
            label = np.concatenate(slice_label, axis=2)
        else:
            label = slice_label[0]

        # 用每一组slice做归一化
        max_val = img.max()

        # 用每一天slice做归一化
        # max_val = self.norm_thresh[case_day]
        img = img.astype(np.float32)
        if max_val != 0:
            img /= max_val

        img = cv2.resize(img, (self.crop_size, self.crop_size), interpolation=cv2.INTER_AREA)
        label_mid = cv2.resize(label_mid, (self.crop_size, self.crop_size), interpolation=cv2.INTER_NEAREST).astype(np.float32)
        label = cv2.resize(label, (self.crop_size, self.crop_size), interpolation=cv2.INTER_NEAREST).astype(np.float32)

        tensor_img, tensor_lab, tensor_lab_mid = self.transform(img=img, mask=label, mask_mid=label_mid)

        if not self.config.USE_3C:
            tensor_lab = tensor_lab.unsqueeze(0).long()

        return tensor_img, tensor_lab, tensor_lab_mid

    def load_slice(self, img_key_part, slice_num, load_img = True):
        slice_str = str(slice_num).zfill(4)
        img_key_part[3] = slice_str
        img_path = self.img_key_to_path["_".join(img_key_part)]
        label_path = self.label_key_to_path["_".join(img_key_part)]

        if load_img:
            if os.path.exists(img_path):
                return cv2.imread(img_path, cv2.IMREAD_UNCHANGED)
            return None
        else:
            if os.path.exists(label_path):
                return cv2.imread(label_path, cv2.IMREAD_UNCHANGED)
            return None

    def read_image(self, path):
        '''Reads and converts the image.
        path: the full complete path to the .png file'''

        # Read image in a corresponding manner
        # convert int16 -> float32
        image = cv2.imread(path, cv2.IMREAD_UNCHANGED).astype('float32')
        # Scale to [0, 255]
        # image = cv2.normalize(image, None, alpha=0, beta=255,
        #                       norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_32F)
        # image = image.astype(np.uint8)

        return image

    def change_3c_to_1c(self, label):
        new_label = label[..., 0] * 1 + label[..., 1] * 2 + label[..., 2] * 3
        self.change_repeat_class(label, new_label)

        return new_label

    def change_repeat_class(self, label, newlabel):
        added_label = label[..., 0] + label[..., 1] + label[..., 2]
        try:
            x_list, y_list = np.where(added_label > 1)
        except:
            return np.zeros_like(label[..., 0])

        for x, y in zip(x_list, y_list):
            same_class_pixel_count = [0, 0, 0]
            for i in range(3):
                if label[x, y, i] != 0:
                    same_class_pixel_count[i] = label[x - 1:x + 2, y - 1:y + 2, i].flatten().tolist().count(1) - 1
            newlabel[x, y] = same_class_pixel_count.index(max(same_class_pixel_count)) + 1

        return newlabel

    def cal_norm_thresh(self, case_day, img):
        img = np.stack(img, axis=2)
        img = img.astype(np.float32)
        max_val = img.max()

        self.norm_thresh[case_day] = max_val


if __name__ == '__main__':
    config = get_config("./configs/config_fpn.py")

    testset_A = CMRDataset(config, config.data_path, mode='train')
    testLoader_A = data.DataLoader(testset_A, batch_size=1, shuffle=False, num_workers=0)

    for i, (img, label, img_names) in enumerate(testLoader_A, 0):
        print(img, label, img_names)