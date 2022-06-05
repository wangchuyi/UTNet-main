import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import SimpleITK as sitk
import pdb 
import cv2
import importlib
import os
def log_evaluation_result(writer, dice_list, name, epoch):
    
    writer.add_scalar('Test_Dice/%s_AVG'%name, dice_list.mean(), epoch+1)
    for idx in range(3):
        writer.add_scalar('Test_Dice/%s_Dice%d'%(name, idx+1), dice_list[idx], epoch+1)
        print(dice_list[idx])
    # writer.add_scalar('Test_ASD/%s_AVG'%name, ASD_list.mean(), epoch+1)
    # for idx in range(3):
    #     writer.add_scalar('Test_ASD/%s_ASD%d'%(name, idx+1), ASD_list[idx], epoch+1)
    # writer.add_scalar('Test_HD/%s_AVG'%name, HD_list.mean(), epoch+1)
    # for idx in range(3):
    #     writer.add_scalar('Test_HD/%s_HD%d'%(name, idx+1), HD_list[idx], epoch+1)


def multistep_lr_scheduler_with_warmup(optimizer, init_lr, epoch, warmup_epoch, lr_decay_epoch, max_epoch, gamma=0.1):

    if epoch >= 0 and epoch <= warmup_epoch:
        lr = init_lr * 2.718 ** (10*(float(epoch) / float(warmup_epoch) - 1.))
        if epoch == warmup_epoch:
            lr = init_lr
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr

        return lr

    flag = False
    for i in range(len(lr_decay_epoch)):
        if epoch == lr_decay_epoch[i]:
            flag = True
            break

    if flag == True:
        lr = init_lr * gamma**(i+1)
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr

    else:
        return optimizer.param_groups[0]['lr']

    return lr

def exp_lr_scheduler_with_warmup(optimizer, init_lr, epoch, warmup_epoch, max_epoch):

    if epoch >= 0 and epoch <= warmup_epoch:
        lr = init_lr * 2.718 ** (10*(float(epoch) / float(warmup_epoch) - 1.))
        if epoch == warmup_epoch:
            lr = init_lr
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr

        return lr

    else:
        lr = init_lr * (1 - epoch / max_epoch)**0.9
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr
    return lr

class Exp_lr_scheduler_with_warmup():
    def __init__(self,optimizer,init_lr,warmup_epoch,max_epoch):
        self.optimizer = optimizer
        self.init_lr = init_lr
        self.warmup_epoch =warmup_epoch
        self.max_epoch =max_epoch
        self.epoch = 0
    
    def exp_lr_scheduler_with_warmup(self):
        if self.epoch >= 0 and self.epoch <= self.warmup_epoch:
            lr = self.init_lr * 2.718 ** (10*(float(self.epoch) / float(self.warmup_epoch) - 1.))
            if self.epoch == self.warmup_epoch:
                lr = self.init_lr
            for param_group in self.optimizer.param_groups:
                param_group['lr'] = lr
            return lr
        else:
            lr = self.init_lr * (1 -self.epoch / self.max_epoch)**0.9
            for param_group in self.optimizer.param_groups:
                param_group['lr'] = lr
            return lr

    def step(self):
        lr = self.exp_lr_scheduler_with_warmup()
        self.epoch += 1
        return lr

def cal_dice(pred, target, C): 
    with torch.no_grad():
        pred = F.softmax(pred, dim=1)
        _, pred = torch.max(pred, dim=1)
        pred = pred.view(-1, 1).cpu()
        target = target.view(-1, 1).cpu()

        N = pred.shape[0]
        target_mask = target.data.new(N, C).fill_(0)
        target_mask.scatter_(1, target, 1.) 

        pred_mask = pred.data.new(N, C).fill_(0)
        pred_mask.scatter_(1, pred, 1.) 

        intersection= pred_mask * target_mask
        summ = pred_mask + target_mask

        intersection = intersection.sum(0).type(torch.float32)
        summ = summ.sum(0).type(torch.float32)
        
        eps = torch.rand(C, dtype=torch.float32)
        eps = eps.fill_(1e-7)

        summ += eps
        intersection += eps/2
        dice = 2 * intersection / summ

        total_dice = 2*((target_mask[:,1:]*pred_mask[:,1:]).sum()+1e-7/2)/(target_mask[:,1:].sum() + pred_mask[:,1:].sum() + 1e-7)
        return total_dice,dice.numpy()
        
def cal_dice_3C(pred, target, C,thresh = 0.5):
    with torch.no_grad():
        sigmoid = nn.Sigmoid()
        pred = sigmoid(pred).round().to(torch.float32)
        pred = pred.cpu()
        target=target.cpu()

        pred = pred.reshape(pred.shape[0],pred.shape[1],-1)
        target =target.reshape(pred.shape[0],pred.shape[1],-1)
        d1,d2,d3= np.where(pred>thresh)
        pred[d1,d2,d3] = 1
        pred = pred.cpu()
        intersection= pred * target
        summ = pred + target

        intersection = intersection.sum(2).type(torch.float32)
        summ = summ.sum(2).type(torch.float32)
        
        eps = torch.rand(C, dtype=torch.float32)
        eps = eps.fill_(1e-7)

        summ += eps
        #(n,c,1)
        intersection += eps/2
        dice = 2 * intersection / summ
        dice = dice.mean(0)
        dice = dice.squeeze()
        total_dice = 2*((target*pred).sum()+1e-7/2)/(target.sum() + pred.sum() + 1e-7)

        return total_dice.item(),dice.numpy()

def cal_asd(itkPred, itkGT):
    
    reference_distance_map = sitk.Abs(sitk.SignedMaurerDistanceMap(itkGT, squaredDistance=False))
    reference_surface = sitk.LabelContour(itkGT)

    statistics_image_filter = sitk.StatisticsImageFilter()
    statistics_image_filter.Execute(reference_surface)
    num_reference_surface_pixels = int(statistics_image_filter.GetSum())

    segmented_distance_map = sitk.Abs(sitk.SignedMaurerDistanceMap(itkPred, squaredDistance=False))
    segmented_surface = sitk.LabelContour(itkPred)

    seg2ref_distance_map = reference_distance_map * sitk.Cast(segmented_surface, sitk.sitkFloat32)
    ref2seg_distance_map = segmented_distance_map * sitk.Cast(reference_surface, sitk.sitkFloat32)

    statistics_image_filter.Execute(segmented_surface)
    num_segmented_surface_pixels = int(statistics_image_filter.GetSum())

    seg2ref_distance_map_arr = sitk.GetArrayViewFromImage(seg2ref_distance_map)
    seg2ref_distances = list(seg2ref_distance_map_arr[seg2ref_distance_map_arr!=0])
    seg2ref_distances = seg2ref_distances + \
                        list(np.zeros(num_segmented_surface_pixels - len(seg2ref_distances)))
    ref2seg_distance_map_arr = sitk.GetArrayViewFromImage(ref2seg_distance_map)
    ref2seg_distances = list(ref2seg_distance_map_arr[ref2seg_distance_map_arr!=0])
    ref2seg_distances = ref2seg_distances + \
                        list(np.zeros(num_reference_surface_pixels - len(ref2seg_distances)))
    
    all_surface_distances = seg2ref_distances + ref2seg_distances

    ASD = np.mean(all_surface_distances)

    return ASD

def get_config(config_file):
    temp_config_name = os.path.basename(config_file)
    temp_module_name = os.path.splitext(temp_config_name)[0]
    config = importlib.import_module("configs.%s" % temp_module_name)
    cfg = config.config
    return cfg

if __name__ == '__main__':
    pred = torch.from_numpy(np.ones((12,3,128,128)))
    pred[:,0,:64,:64] = 0
    label = torch.from_numpy(np.ones((12,3,128,128)))
    cal_dice_3C(pred,label,3)