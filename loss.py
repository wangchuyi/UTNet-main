import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import optim

import numpy as np
from model.utnet import UTNet, UTNet_Encoderonly

from dataset_domain import CMRDataset

from torch.utils import data
from losses import DiceLoss
from utils.utils import *
from utils import metrics
from optparse import OptionParser
import SimpleITK as sitk
import segmentation_models_pytorch as smp
from torch.utils.tensorboard import SummaryWriter
import time
import math
import os
import sys
import pdb
import warnings
import cv2


class Loss_func():
    def __init__(self, config):
        self.config = config

        self.ce_loss = nn.CrossEntropyLoss(weight=torch.tensor(config.weight).cuda())
        self.dice_loss = DiceLoss()

        self.soft_bce_loss = smp.losses.SoftBCEWithLogitsLoss()
        self.tvsky_loss = smp.losses.TverskyLoss(mode='multilabel', log_loss=False)

        self.loss_weight = config.loss_weight
        self.func_list = []

        if config.loss == "BCE_TV":
            self.func_list = [self.soft_bce_loss, self.tvsky_loss]
        if config.loss == "CE_DICE":
            self.func_list = [self.ce_loss, self.dice_loss]

    def caculate_loss(self, pred, label):
        loss = 0
        loss_list = []
        for w, func in zip(self.loss_weight,self.func_list):
            if func == self.ce_loss:
                part_loss = func(pred, label.squeeze(1))
            else:
                part_loss = func(pred, label)
            loss += w * part_loss
            loss_list.append(part_loss)
        return loss, [l.detach().item()for l in loss_list]

    def __call__(self, pred, label):
        loss = 0
        loss_list = []
        if  self.config.aux_loss and not (isinstance(pred, tuple) or isinstance(pred, list)):
                pred = pred[0]
                pritn("WARNING!!! can not do aux_loss!!!check net pred")
                assert False
        # 对unet在decode的不同阶段（resize到相同分辨率）图像做损失
        if self.config.aux_loss:
            for j in range(len(pred)):
                loss += self.config.aux_weight[j] * self.caculate_loss(pred[j], label)[0]
        else:
            loss, loss_list = self.caculate_loss(pred, label)
        return loss, loss_list


if __name__ == '__main__':
    config = get_config("/configs/config.py")
    l = Loss_func(config)
    pred = (torch.ones((2, 2)), torch.ones((2, 2)))
    label = torch.ones((2, 2))
    l(pred, label)
