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
class CMRDataset(Dataset):
    def __init__(self, dataset_dir, mode='train', domain='A', crop_size=256, scale=0.1, rotate=10, debug=False):

        self.mode = mode
        self.dataset_dir = dataset_dir
        self.crop_size = crop_size
        self.scale = scale
        self.rotate = rotate
        self.img_name_list =[]

        if self.mode == 'train':
            pre_face = 'debug'
            if 'C' in domain or 'D' in domain:
                print('No domain C or D in Training set')
                raise StandardError

        elif self.mode == 'test':
            pre_face = 'debug'

        else:
            print('Wrong mode')
            raise StandardError
        if debug:
            # validation set is the smallest, need the shortest time for load data.
           pre_face = 'Testing'

        path = self.dataset_dir + pre_face + '/'
        print('start loading data')
        
        name_list = []

        img_list = []
        lab_list = []
        spacing_list = []

        case_list = os.listdir(path)
        for case in tqdm(case_list):
            day_list = os.listdir(path+case)
            for day in day_list:
                scans_path = os.path.join(path, case, day, 'scans')
                scans_list = os.listdir(scans_path)
                imgs_per_day = []
                labs_per_day = []
                for scan_name in scans_list:
                    _, idx, img_height, img_width, pixel_height, pixel_width = scan_name.split('_')
                    pixel_width = pixel_width[:-4] # delete '.png'

                    img = self.read_image(os.path.join(scans_path, scan_name))

                    label_name = case + '_' + day.split('_')[1] + '_' + 'slice' + '_' + idx + '.png'
                    if os.path.exists(os.path.join(self.dataset_dir, 'new_label', label_name)):
                        label = self.read_image(os.path.join(self.dataset_dir, 'new_label', label_name))
                    else:
                        continue
                        label = np.zeros_like(img)

                    imgs_per_day.append(img)
                    labs_per_day.append(label)
                    self.img_name_list.append(scan_name)

                img, lab = self.preprocess(np.array(imgs_per_day), np.array(labs_per_day))

                img_list.append(img)
                lab_list.append(lab)
                # spacing_list.append([])
        self.img_slice_list = []
        self.lab_slice_list = []
        # if self.mode == 'train':
        for i in range(len(img_list)):
            tmp_img = img_list[i]
            tmp_lab = lab_list[i]

            z, x, y = tmp_img.shape

            for j in range(z):
                self.img_slice_list.append(tmp_img[j])
                self.lab_slice_list.append(tmp_lab[j])

        print('load done, length of dataset:', len(self.img_slice_list))

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
    def __len__(self):
        return len(self.img_slice_list)

    def preprocess(self, itk_img, itk_lab):
        
        img = itk_img
        lab = itk_lab
        max98 = np.percentile(img, 98)
        img = np.clip(img, 0, max98)
            
        z, y, x = img.shape
        if x < self.crop_size:
            diff = (self.crop_size + 10 - x) // 2
            img = np.pad(img, ((0,0), (0,0), (diff, diff)))
            lab = np.pad(lab, ((0,0), (0,0), (diff,diff)))
        if y < self.crop_size:
            diff = (self.crop_size + 10 -y) // 2
            img = np.pad(img, ((0,0), (diff, diff), (0,0)))
            lab = np.pad(lab, ((0,0), (diff, diff), (0,0)))

        img = img / max98
        img = img *255
        # debug
        # n = img.shape[0]
        # for i in range(n):
        #     cv2.imwrite("/mnt/home/code/UTnet/UTNet-main/show_data/"+str(i)+".jpg",lab[i]*(255/3.0))
        tensor_img = torch.from_numpy(img).float()
        tensor_lab = torch.from_numpy(lab).long()

        return tensor_img, tensor_lab


    def __getitem__(self, idx):
        tensor_image = self.img_slice_list[idx]
        tensor_label = self.lab_slice_list[idx]
        img_name=self.img_name_list[idx]
        tensor_image = tensor_image.unsqueeze(0).unsqueeze(0)
        tensor_label = tensor_label.unsqueeze(0).unsqueeze(0)
        if self.mode == 'train':
            # Gaussian Noise
            tensor_image += torch.randn(tensor_image.shape) * 0.02
            # Additive brightness
            rnd_bn = np.random.normal(0, 0.7)#0.03
            tensor_image += rnd_bn
            # gamma
            minm = tensor_image.min()
            rng = tensor_image.max() - minm
            gamma = np.random.uniform(0.5, 1.6)
            tensor_image = torch.pow((tensor_image-minm)/rng, gamma)*rng + minm

            tensor_image, tensor_label = self.random_zoom_rotate(tensor_image, tensor_label)
            tensor_image, tensor_label = self.randcrop(tensor_image, tensor_label)
            #tensor_image, tensor_label = self.center_crop(tensor_image[0], tensor_label[0])
        else:
            tensor_image, tensor_label = self.center_crop(tensor_image, tensor_label)
        
        assert tensor_image.shape == tensor_label.shape
        
        if self.mode == 'train':
            return tensor_image, tensor_label,img_name
        else:
            return tensor_image, tensor_label,img_name

    def randcrop(self, img, label):
        _, _, H, W = img.shape
        
        diff_H = H - self.crop_size
        diff_W = W - self.crop_size
        
        rand_x = np.random.randint(0, diff_H)
        rand_y = np.random.randint(0, diff_W)
        
        croped_img = img[0, :, rand_x:rand_x+self.crop_size, rand_y:rand_y+self.crop_size]
        croped_lab = label[0, :, rand_x:rand_x+self.crop_size, rand_y:rand_y+self.crop_size]

        return croped_img, croped_lab


    def center_crop(self, img, label):
        _,D, H, W = img.shape
        
        diff_H = H - self.crop_size
        diff_W = W - self.crop_size
        
        rand_x = diff_H // 2
        rand_y = diff_W // 2

        croped_img = img[0,:, rand_x:rand_x+self.crop_size, rand_y:rand_y+self.crop_size]
        croped_lab = label[0,:, rand_x:rand_x+self.crop_size, rand_y:rand_y+self.crop_size]

        return croped_img, croped_lab

    def center_crop_train(self, img, label):
        D, H, W = img.shape
        
        diff_H = H - self.crop_size
        diff_W = W - self.crop_size
        
        rand_x = diff_H // 2
        rand_y = diff_W // 2

        croped_img = img[:,:, rand_x:rand_x+self.crop_size, rand_y:rand_y+self.crop_size]
        croped_lab = label[:,:, rand_x:rand_x+self.crop_size, rand_y:rand_y+self.crop_size]

        return croped_img, croped_lab

    def random_zoom_rotate(self, img, label):
        scale_x = np.random.random() * 2 * self.scale + (1 - self.scale)
        scale_y = np.random.random() * 2 * self.scale + (1 - self.scale)


        theta_scale = torch.tensor([[scale_x, 0, 0],
                                    [0, scale_y, 0],
                                    [0, 0, 1]]).float()
        angle = (float(np.random.randint(-self.rotate, self.rotate)) / 180.) * math.pi

        theta_rotate = torch.tensor( [  [math.cos(angle), -math.sin(angle), 0], 
                                        [math.sin(angle), math.cos(angle), 0], 
                                        ]).float()
        
    
        theta_rotate = theta_rotate.unsqueeze(0)
        grid = F.affine_grid(theta_rotate, img.size())
        img = F.grid_sample(img, grid, mode='bilinear')
        label = F.grid_sample(label.float(), grid, mode='nearest').long()
    
        return img, label


