import os
import cv2
import numpy as np
from tqdm import tqdm

def change_repeat_class(label, newlabel):
    added_label = label[..., 0] + label[..., 1] + label[..., 2]
    x_list, y_list = np.where(added_label > 1)
    for x, y in zip(x_list, y_list):
        same_class_pixel_count = [0, 0, 0]
        for i in range(3):
            if label[x, y, i] != 0:
                same_class_pixel_count[i] = label[x - 1:x + 2, y - 1:y + 2, i].flatten().tolist().count(1) - 1
        newlabel[x, y] = same_class_pixel_count.index(max(same_class_pixel_count)) + 1


if __name__ == '__main__':
    base ="/mnt/home/code/UTnet/dataset/label"
    dst_path = "/mnt/home/code/UTnet/dataset/new_label"
    all_label = os.listdir(base)

    for label_name in tqdm(all_label):
        label = cv2.imread(os.path.join(base,label_name))
        new_label = label[...,0]*1+label[...,1]*2+label[...,2]*3
        change_repeat_class(label, new_label)
        # print(np.max(label),np.max(new_label))
        assert np.max(new_label)<=3
        print(new_label.shape)
        cv2.imwrite(os.path.join(dst_path,label_name),new_label)
        # print(os.path.join(dst_path,label_name))