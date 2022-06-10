import albumentations as A
from albumentations.pytorch import ToTensorV2
import cv2

def makeAug(config, mode='train'):
    if config.model == "UTNet":
        raise NotImplementedError
    elif config.model == "FPN":
        return FPN_AUG(config, mode)
    else:
        raise NotImplementedError


class FPN_AUG():
    def __init__(self, config, mode='train'):
        self.mode = mode
        if mode == "train":
            self.transform = make_fpn_train_augmenter(config.crop_size)
        else:
            self.transform = make_fpn_test_augmenter(config.crop_size)

    def __call__(self, img, mask=0):
        result = self.transform(image=img, mask=mask)
        tensor_img, tensor_lab = result['image'], result['mask']

        return tensor_img, tensor_lab


def make_fpn_test_augmenter(image_size=256,crop_size=0.9):
    crop_size = round(image_size*crop_size)
    return  A.Compose([
        A.CenterCrop(height=crop_size, width=crop_size),
        ToTensorV2(transpose_mask=True)
    ])


def make_fpn_train_augmenter(image_size=256,crop_size=0.9,aug_prob=0.4,max_cutout=0,strong_aug=True):
    p = aug_prob
    crop_size = round(image_size*crop_size)
    if p <= 0:
        return A.Compose([
            A.RandomCrop(height=crop_size, width=crop_size, always_apply=True),
            ToTensorV2(transpose_mask=True)
        ])

    aug_list = []
    if max_cutout > 0:
        aug_list.extend([
            A.CoarseDropout(
                max_holes=max_cutout, min_holes=1,
                max_height=crop_size//10, max_width=crop_size//10,
                min_height=4, min_width=4, mask_fill_value=0, p=0.2*p),
        ])

    aug_list.extend([
        A.ShiftScaleRotate(
            shift_limit=0.0625, scale_limit=0.2, rotate_limit=25,
            interpolation=cv2.INTER_AREA, p=p),
        A.RandomCrop(height=crop_size, width=crop_size, always_apply=True),
        A.PadIfNeeded(min_height=image_size, 
                      min_width=image_size,border_mode=cv2.BORDER_CONSTANT,value = 0),
        A.HorizontalFlip(p=0.5*p),
        A.OneOf([
            A.MotionBlur(p=0.2*p),
            A.MedianBlur(blur_limit=3, p=0.1*p),
            A.Blur(blur_limit=3, p=0.1*p),
        ], p=0.2*p),
        A.Perspective(p=0.2*p),
    ])

    if strong_aug:
        aug_list.extend([
            A.GaussNoise(var_limit=0.001, p=0.2*p),
            A.OneOf([
                A.OpticalDistortion(p=0.3*p),
                A.GridDistortion(p=0.1*p),
                A.PiecewiseAffine(p=0.3*p),
            ], p=0.2*p),
            A.OneOf([
                A.Sharpen(p=0.2*p),
                A.Emboss(p=0.2*p),
                A.RandomBrightnessContrast(p=0.2*p),
            ], p=0.3*p),
        ])

    aug_list.extend([
        ToTensorV2(transpose_mask=True)
    ])

    return A.Compose(aug_list)
