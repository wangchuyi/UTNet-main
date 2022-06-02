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
from data_aug import make_train_augmenter

Use3C = True

class CMRDataset(Dataset):
    def __init__(self, dataset_dir, mode='train', useUT = False, crop_size=320, is_debug = False):
        # 基础参数
        self.mode = mode  # 模式包含’train‘、’test‘、’debug‘
        self.dataset_dir = dataset_dir
        self.crop_size = crop_size
        self.useUT = False

        # ut 参数
        self.scale = 0.1
        self.rotate = 10

        # 配置数据增强
        self.transform = make_train_augmenter(image_size=crop_size)

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
        case_list = os.listdir(path)    # ['case2', 'case6', ...]
        for case in tqdm(case_list):
            case_day_list = os.listdir(path+case)    # ['case2_day1', 'case2_day2', ...]
            for case_day in case_day_list:

                # 读取每天所有的 slice 计算归一化阈值
                imgs_per_day = []

                scans_path = os.path.join(path, case, case_day, 'scans')
                scans_list = os.listdir(scans_path)    # ['slice_0001_266_266_1.50_1.50.png', ...]
                for scan_name in scans_list:
                    _, idx, img_height, img_width, pixel_height, pixel_width = scan_name.split('_')
                    pixel_width = pixel_width[:-4] # delete '.png'

                    # 生成诸如 'case2_day1_slice_0001' 的key name
                    img_key_name = case_day + '_' + scan_name[:10]
                    self.img_key_name_list.append(img_key_name)

                    # 读取每天所有的 slice 和 label
                    img_path = os.path.join(scans_path, scan_name)
                    img = self.read_image(img_path)

                    label_name = img_key_name + '.png'
                    label_path = os.path.join(self.dataset_dir, 'label', label_name)
                    if os.path.exists(label_path):
                        self.has_label[img_key_name] = True
                        self.label_key_to_path[img_key_name] = label_path
                    else:
                        self.has_label[img_key_name] = False

                    # 存储每张图片和label的地址
                    self.img_key_to_path[img_key_name] = img_path

                    imgs_per_day.append(img)

                self.cal_norm_thresh(case_day, imgs_per_day)

        print('load done, length of dataset:', len(self.img_key_name_list))

    def __len__(self):
        return len(self.img_key_name_list)

    def __getitem__(self, idx):
        img_key_name = self.img_key_name_list[idx]
        img_key_part = img_key_name.split("_")  # 'case2', 'day1', 'slice', '0001'
        tensor_image, tensor_label = self.generate_25d_img(img_key_part)
        return tensor_image, tensor_label, img_key_name

    def generate_25d_img(self,img_key_part, stride=1, collate_num=5):
        # 生成2.5d图像列表
        slice_num = int(img_key_part[3])
        slice_range = [slice_num-2,slice_num-1,slice_num,slice_num+1,slice_num+2]
        slice_img = [self.load_slice(img_key_part.copy(), i) for i in slice_range]

        if slice_img[3] is None:
            slice_img[3] = slice_img[2]
        if slice_img[4] is None:
            slice_img[4] = slice_img[3]
        if slice_img[1] is None:
            slice_img[1] = slice_img[2]
        if slice_img[0] is None:
            slice_img[0] = slice_img[1]
        tensor_image = 0
        tensor_label = 0

        case_day = '_'.join(img_key_part[:2])
        img_key_name = "_".join(img_key_part)

        if self.useUT:
            # 读取label
            label = self.read_label(img_key_name)

            # 预处理
            if Use3C:
                label = torch.from_numpy(label).permute(2,0,1).numpy()
            else:
                label = torch.from_numpy(label.astype('float32')).unsqueeze(0).numpy()

            img, lab = self.preprocess_ut(np.array(slice_img), label, case_day)

            # 数据增强
            tensor_image, tensor_label = self.aug_ut(img, lab)

        else:
            # 读取label
            label = self.read_label(img_key_name)

            # 预处理
            img = self.preprocess(slice_img, case_day)

            # 数据增强
            if self.mode == "train":
                result = self.transform(image=img, mask=label)
                img, label = result['image'], result['mask']
            else:
                img = torch.from_numpy(img).permute(2,0,1)
                label = torch.from_numpy(label).permute(2,0,1)

            tensor_image = img

            if Use3C:
                tensor_label = label.long()
            else:
                tensor_label = label.unsqueeze(0).long()

        return tensor_image, tensor_label

    def read_label(self, img_key_name):
        if self.useUT:
            if self.has_label[img_key_name]:
                label = self.read_image(self.label_key_to_path[img_key_name])
            else:
                w = self.read_image(self.img_key_to_path[img_key_name]).shape[0]
                label = np.zeros((w, w, 3))
        else:
            if self.has_label[img_key_name]:
                label = self.read_image(self.label_key_to_path[img_key_name])
                label = cv2.resize(label, (self.crop_size,self.crop_size), interpolation=cv2.INTER_NEAREST)
            else:
                label = np.zeros((self.crop_size,self.crop_size, 3))

        if Use3C:
            return label
        else:
            return self.change_3c_to_1c(label)

    def change_3c_to_1c(self, label):
        new_label = label[..., 0] * 1 + label[..., 1] * 2 + label[..., 2] * 3
        self.change_repeat_class(label, new_label)

        return new_label

    def change_repeat_class(self, label, newlabel):
        added_label = label[..., 0] + label[..., 1] + label[..., 2]
        x_list, y_list = np.where(added_label > 1)
        for x, y in zip(x_list, y_list):
            same_class_pixel_count = [0, 0, 0]
            for i in range(3):
                if label[x, y, i] != 0:
                    same_class_pixel_count[i] = label[x - 1:x + 2, y - 1:y + 2, i].flatten().tolist().count(1) - 1
            newlabel[x, y] = same_class_pixel_count.index(max(same_class_pixel_count)) + 1

        return newlabel


    def preprocess(self, slice_img, case_day):
        img = np.stack(slice_img, axis=2)
        max98 = self.norm_thresh[case_day]
        img = np.clip(img, 0, max98)
        img = img.astype(np.float32)
        img /= max98
        img = cv2.resize(img, (self.crop_size, self.crop_size), cv2.INTER_AREA)

        return img


    def aug_ut(self, img, lab):
        img = [x.unsqueeze(0).unsqueeze(0) for x in img]
        tensor_image = torch.cat(img, 1)
        tensor_label = lab.unsqueeze(0)

        if self.mode == 'train':
            # Gaussian Noise
            tensor_image += torch.randn(tensor_image.shape) * 0.02
            # Additive brightness
            rnd_bn = np.random.normal(0, 0.7)  # 0.03
            tensor_image += rnd_bn
            # gamma
            minm = tensor_image.min()
            rng = tensor_image.max() - minm
            gamma = np.random.uniform(0.5, 1.6)
            tensor_image = torch.pow((tensor_image - minm) / rng, gamma) * rng + minm

            tensor_image, tensor_label = self.random_zoom_rotate(tensor_image, tensor_label)
            tensor_image, tensor_label = self.randcrop(tensor_image, tensor_label)
        else:
            tensor_image, tensor_label = self.center_crop(tensor_image, tensor_label)

        return tensor_image, tensor_label

    def preprocess_ut(self, img, lab, case_day):
        max98 = self.norm_thresh[case_day]
        img = np.clip(img, 0, max98)

        z, y, x = img.shape
        if x < self.crop_size:
            diff = (self.crop_size + 10 - x) // 2
            img = np.pad(img, ((0, 0), (0, 0), (diff, diff)))
            lab = np.pad(lab, ((0, 0), (0, 0), (diff, diff)))
        if y < self.crop_size:
            diff = (self.crop_size + 10 - y) // 2
            img = np.pad(img, ((0, 0), (diff, diff), (0, 0)))
            lab = np.pad(lab, ((0, 0), (diff, diff), (0, 0)))

        img = img / max98
        img = img * 255

        tensor_img = torch.from_numpy(img).float()
        tensor_lab = torch.from_numpy(lab).long()

        return tensor_img, tensor_lab


    def load_slice(self,img_key_part, slice_num):
        slice_str = str(slice_num).zfill(4)
        img_key_part[3] = slice_str
        img_path = self.img_key_to_path["_".join(img_key_part)]

        # if self.useUT:
        if os.path.exists(img_path):
            return cv2.imread(img_path, cv2.IMREAD_UNCHANGED)
        return None

    def cal_norm_thresh(self, case_day, img):
        # img = np.stack(img, axis=2)
        # img = img.astype(np.float32)
        # max_val = img.max()

        # utnet 使用的方法
        max98 = np.percentile(img, 98)
        # img = np.clip(img, 0, max98)

        self.norm_thresh[case_day] = max98

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

    def randcrop(self, img, label):
        _, _, H, W = img.shape

        diff_H = H - self.crop_size
        diff_W = W - self.crop_size

        rand_x = np.random.randint(0, diff_H)
        rand_y = np.random.randint(0, diff_W)

        croped_img = img[0, :, rand_x:rand_x + self.crop_size, rand_y:rand_y + self.crop_size]
        croped_lab = label[0, :, rand_x:rand_x + self.crop_size, rand_y:rand_y + self.crop_size]

        return croped_img, croped_lab

    def center_crop(self, img, label):
        _, D, H, W = img.shape

        diff_H = H - self.crop_size
        diff_W = W - self.crop_size

        rand_x = diff_H // 2
        rand_y = diff_W // 2

        croped_img = img[0, :, rand_x:rand_x + self.crop_size, rand_y:rand_y + self.crop_size]
        croped_lab = label[0, :, rand_x:rand_x + self.crop_size, rand_y:rand_y + self.crop_size]

        return croped_img, croped_lab

    def center_crop_train(self, img, label):
        D, H, W = img.shape

        diff_H = H - self.crop_size
        diff_W = W - self.crop_size

        rand_x = diff_H // 2
        rand_y = diff_W // 2

        croped_img = img[:, :, rand_x:rand_x + self.crop_size, rand_y:rand_y + self.crop_size]
        croped_lab = label[:, :, rand_x:rand_x + self.crop_size, rand_y:rand_y + self.crop_size]

        return croped_img, croped_lab

    def random_zoom_rotate(self, img, label):
        scale_x = np.random.random() * 2 * self.scale + (1 - self.scale)
        scale_y = np.random.random() * 2 * self.scale + (1 - self.scale)

        theta_scale = torch.tensor([[scale_x, 0, 0],
                                    [0, scale_y, 0],
                                    [0, 0, 1]]).float()
        angle = (float(np.random.randint(-self.rotate, self.rotate)) / 180.) * math.pi

        theta_rotate = torch.tensor([[math.cos(angle), -math.sin(angle), 0],
                                     [math.sin(angle), math.cos(angle), 0],
                                     ]).float()

        theta_rotate = theta_rotate.unsqueeze(0)
        grid = F.affine_grid(theta_rotate, img.size())
        img = F.grid_sample(img, grid, mode='bilinear')
        label = F.grid_sample(label.float(), grid, mode='nearest').long()

        return img, label
