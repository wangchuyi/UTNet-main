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
from data_aug import make_train_augmenter
UTBASE = False
class CMRDataset(Dataset):
    def __init__(self, dataset_dir, mode='train', domain='A', crop_size=320, scale=0.1, rotate=10, debug=False):
        self.mode = mode
        self.dataset_dir = dataset_dir
        self.crop_size = crop_size
        self.scale = scale
        self.rotate = rotate
        self.img_name_list =[]
        self.img_path_list =[]
        self.all_img_dic = {}
        self.has_label_img_name = []
        self.day_slice_num = {}
        self.pre_face = ""
        self.transform = make_train_augmenter(image_size=crop_size)

        if self.mode == 'train':
            self.pre_face = 'Training'
            if 'C' in domain or 'D' in domain:
                print('No domain C or D in Training set')
                raise StandardError

        elif self.mode == 'test':
            self.pre_face = 'Testing'

        else:
            print('Wrong mode')
            raise StandardError
        if debug:
           self.pre_face = 'debug'

        path = self.dataset_dir + self.pre_face + '/'
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
                has_label = []
                img_names_per_day =[]
                for scan_name in scans_list:
                    _, idx, img_height, img_width, pixel_height, pixel_width = scan_name.split('_')
                    pixel_width = pixel_width[:-4] # delete '.png'

                    img = self.read_image(os.path.join(scans_path, scan_name))
                    case_day_key = case + '_' + day.split('_')[1]
                    img_key_name = case_day_key + '_' +scan_name
                    self.img_name_list.append(img_key_name)
                    self.img_path_list.append(os.path.join(scans_path, scan_name))
                    if  case_day_key in self.day_slice_num:
                        self.day_slice_num[case_day_key]+=1
                    else:
                        self.day_slice_num[case_day_key]=1
                    img_names_per_day.append(img_key_name)
                    label_name = case_day_key + '_' + 'slice' + '_' + idx + '.png'
                    if os.path.exists(os.path.join(self.dataset_dir, 'new_label', label_name)):
                        label = self.read_image(os.path.join(self.dataset_dir, 'new_label', label_name))
                        self.has_label_img_name.append(img_key_name)
                    else:
                        label = np.zeros_like(img)
                    imgs_per_day.append(img)
                    labs_per_day.append(label)
                if UTBASE:
                    img, lab = self.preprocess(np.array(imgs_per_day), np.array(labs_per_day))
                    assert img.shape[0] == len (img_names_per_day)
                    for idx,name in enumerate(img_names_per_day):
                        self.all_img_dic[name] = (img[idx],lab[idx])

        print('load done, length of dataset:', len(self.has_label_img_name))

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
        return len(self.img_name_list)

    def preprocess(self, img, lab):
        
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

    def load_slice(self,img_key_part, slice_num,use_utnet_pre = UTBASE):
        slice_str = str(slice_num).zfill(4)
        img_key_part[3] = slice_str
        filename = self.img_name_to_path(img_key_part)
        if use_utnet_pre:
            img_key = "_".join(img_key_part)
            if  img_key in self.has_label_img_name:
                return self.all_img_dic[img_key][0].unsqueeze(0).unsqueeze(0)
            else :
                return None
        else:
            if os.path.exists(filename):
                return cv2.imread(filename, cv2.IMREAD_UNCHANGED)
            return None

    def img_name_to_path(self,img_key_part):
        base_path = os.path.join(self.dataset_dir,self.pre_face)
        case = img_key_part[0]
        day = img_key_part[1]
        dir_path = base_path+"/{0}/{1}_{2}/scans".format(case,case,day)
        slice_img_name = "_".join(img_key_part[2:])
        slice_img_path = os.path.join(dir_path,slice_img_name)
        return slice_img_path

    def img_path_to_label_path(self,img_key_part,path_base):
        case = img_key_part[0]
        day = img_key_part[1]
        label_path = os.path.join(path_base,"_".join(img_key_part[:4])+".png")
        return label_path

    def generate_25d_img(self,img_key_part,slice_num,stride = 1,collate_num = 5,use_utnet_pre = UTBASE):
        slice_range = [slice_num-2,slice_num-1,slice_num,slice_num+1,slice_num+2]
        slice_img =  [self.load_slice(img_key_part, i) for i in slice_range]
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
        if use_utnet_pre:
            tensor_image = torch.cat(slice_img,1)
            _,tensor_label = self.all_img_dic[img_key]
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
        else:
            img = np.stack(slice_img, axis=2)
            img = img.astype(np.float32)
            max_val = img.max()
            if max_val != 0:
                img /= max_val
            img = cv2.resize(img, (self.crop_size,self.crop_size),cv2.INTER_AREA)
            if self.mode == "train":
                msk_file = self.img_path_to_label_path(img_key_part,os.path.join(self.dataset_dir, 'new_label'))
                if os.path.exists(msk_file):
                    msk = cv2.imread(msk_file, cv2.IMREAD_UNCHANGED)
                    msk = cv2.resize(msk, (self.crop_size,self.crop_size),cv2.INTER_NEAREST)
                else:
                    msk = np.zeros((self.crop_size,self.crop_size))
                # data aug + to tensor
                result = self.transform(image=img, mask=msk)
                img, msk = result['image'], result['mask']
            else:
                msk_file = self.img_path_to_label_path(img_key_part,os.path.join(self.dataset_dir, 'new_label'))
                if os.path.exists(msk_file):
                    msk = cv2.imread(msk_file, cv2.IMREAD_UNCHANGED)
                    msk = cv2.resize(msk, (self.crop_size,self.crop_size),cv2.INTER_NEAREST)
                else:
                    msk = np.zeros((self.crop_size,self.crop_size))
                img = torch.from_numpy(img).permute(2,0,1)
                msk = torch.from_numpy(msk)
            tensor_image = img
            tensor_label = msk.unsqueeze(0).long()
        return tensor_image, tensor_label

    def __getitem__(self, idx):
        img_key=self.img_name_list[idx]
        img_key_part = img_key.split("_")
        slice_num = int(img_key_part[3])
        tensor_image, tensor_label = self.generate_25d_img(img_key_part,slice_num)
        return tensor_image, tensor_label,img_key

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


