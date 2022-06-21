import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import optim

import numpy as np
from model.utnet import UTNet, UTNet_Encoderonly

from dataset_domain import CMRDataset

import ast
from torch.utils import data
from torch.utils.data import WeightedRandomSampler
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
from loss import Loss_func
warnings.filterwarnings("ignore", category=UserWarning)
config = None
import pandas as pd
from tqdm import tqdm

#画图函数，输入一个0-cls数值范围的二值图像,shape为(w,h)、返回一个bgr图像，shape为(w,h,3)
def decode_label(label):
    black_board = np.zeros((label.shape[-2],label.shape[-1],3))
    if (label.shape[0]==1):
        label = label[0]
        x,y = np.where(label==1)
        black_board[x,y,:] = config.colors[1]
        x,y = np.where(label==2)
        black_board[x,y,:] = config.colors[2]
        x,y = np.where(label==3)
        black_board[x,y,:] = config.colors[3]
    else :
        x,y = np.where(label[0]==1)
        black_board[x,y,:] = config.colors[1]
        x,y = np.where(label[1]==1)
        black_board[x,y,:] = config.colors[2]
        x,y = np.where(label[2]==1)
        black_board[x,y,:] = config.colors[3]
    return black_board

def c1_to_c3(class_channel_tensor,C=4):
    pred = class_channel_tensor.view(-1, 1)
    N = pred.shape[0]
    pred_mask = pred.data.new(N, C).fill_(0)
    pred_mask.scatter_(1, pred, 1.)
    pred_mask = pred_mask.reshape((class_channel_tensor.shape[-2],class_channel_tensor.shape[-1],C))
    return pred_mask.permute(2,0,1)

#画图函数，输入一个通道为cls的tensor,shape为(c,w,h)、返回一个bgr图像，shape为(w,h,3)
def decode_pred(result,thresh = 0.5):
    black_board = np.zeros((result.shape[-2],result.shape[-1],3))
    dic = {}
    if config.USE_3C:
        start = 0 
    else:
        start = 1
    for idx in range(start,result.shape[0]):
        x_s,y_s = np.where(result[idx,...]>thresh)
        for x,y in zip(x_s,y_s):
            if (x,y) not in dic:
                if config.USE_3C:
                    dic[(x,y)] = config.colors[idx+1]
                else:
                    dic[(x,y)] = config.colors[idx]
    for key in dic:
        x,y=key
        color = dic[key]
        black_board[x,y,:] = color
    return black_board
    
#解析网络输出，讲原图，pred和label画在一起。result,img,label输入均为tensor(cuda)
def decode_result(result,names,img,label,save_img_path,epoach=0,dice=None,loss=None,):
    if config.USE_3C:
        sigmoid = nn.Sigmoid()
        result = sigmoid(result).round().to(torch.float32) 
    else:
        if config.aux_loss:
            result = result[0]
        result = F.softmax(result, dim=1)
    result = result.cpu().detach().numpy()
    img = img.cpu().detach().numpy()
    label = label.cpu().detach().numpy()
    
    for batch_index in range(result.shape[0]):
        wrt_label = decode_label(label[batch_index,...])
        wrt_result = decode_pred(result[batch_index,...])
        if config.input_channel == 1:
            wrt_ori_img = img[batch_index,0,...]*255
        else:
            mid = (config.input_channel-1)/2
            wrt_ori_img = img[batch_index,int(mid),...]*255
        wrt_ori_img = np.expand_dims(wrt_ori_img,-1).repeat(3,axis=-1)

        mask_res = np.ones((wrt_result.shape[0],wrt_result.shape[1]))
        position_res = np.where((wrt_result[...,0]+wrt_result[...,1]+wrt_result[...,2])==0)
        mask_res[position_res]=0
        mask_res_rev = cv2.bitwise_xor(mask_res,np.ones_like(mask_res))
        mask_res  = np.expand_dims(mask_res,-1).repeat(3,axis=-1)
        mask_res_rev  = np.expand_dims(mask_res_rev,-1).repeat(3,axis=-1)

        mask_lab = np.ones((wrt_label.shape[0],wrt_label.shape[1]))
        position_lab = np.where((wrt_label[...,0]+wrt_label[...,1]+wrt_label[...,2])==0)
        mask_lab[position_lab]=0
        mask_lab_rev = cv2.bitwise_xor(mask_lab,np.ones_like(mask_lab))
        mask_lab = np.expand_dims(mask_lab,-1).repeat(3,axis=-1)
        mask_lab_rev = np.expand_dims(mask_lab_rev,-1).repeat(3,axis=-1)
        # cv2.imwrite(save_img_path+"{}".format("1.jpg"),mask_lab*255)
        # cv2.imwrite(save_img_path+"{}".format("2.jpg"),mask_lab_rev*255)

        combine1 = wrt_ori_img*mask_res_rev + wrt_ori_img*mask_res*0.8+wrt_result*0.2
        combine2 = wrt_ori_img*mask_lab_rev + wrt_ori_img*mask_lab*0.8+wrt_label*0.2
        if dice is not None:
            cv2.putText(wrt_ori_img, "dice:", (15,20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)
            for idx,d in enumerate(dice):
                cv2.putText(wrt_ori_img, str(np.round(d,3)), (60+50*idx,20), cv2.FONT_HERSHEY_SIMPLEX, 0.5,config.colors[idx+1], 1)
        if loss is not None:
            cv2.putText(wrt_ori_img, "loss:", (15,45), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)
            for idx,l in enumerate(loss):
                cv2.putText(wrt_ori_img, str(np.round(l,3)), (60+50*idx,45), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)
        final = np.concatenate([wrt_ori_img,combine1, combine2], axis=1)
        cv2.imwrite(save_img_path+"/{}.jpg".format(names[batch_index]),final)
    return

def create_sampler(weight_hard = 0.7, weight_medium = 0.2, weight_easy = 0.1):
    dice = 0

    weight_list = []

    trainset = CMRDataset(config,config.data_path, mode='train', is_debug = config.DEBUG)
    trainLoader = data.DataLoader(trainset, batch_size=1, shuffle=False, num_workers=0)

    df = pd.read_csv('./img_all_class.csv',index_col = 0)

    for i, (_, _, img_names) in tqdm(enumerate(trainLoader, 0)):
        dices = df.loc[str(img_names), 'dices']
        dice = min(ast.literal_eval(dices))

        if dice < 0.3:
            weight = weight_hard
        elif dice < 0.7:
            weight = weight_medium
        else:
            weight = weight_easy

        weight_list.append(weight)

    print(len(weight_list))

    sampler = WeightedRandomSampler(weight_list, num_samples=len(weight_list), replacement=False)

    return sampler

def train_net(net,optimizer,loss_func,exp_scheduler):
    if config.EVAL:
        print(eval(config, net,loss_func,show_log = False,write_result =  False))
        return
    data_path = config.data_path

    trainset = CMRDataset(config,data_path, mode='train', is_debug = config.DEBUG)
    trainLoader = data.DataLoader(trainset, batch_size=config.batch_size, shuffle=True, num_workers=16)

    testset_A = CMRDataset(config,data_path, mode='test', is_debug = config.DEBUG)
    testLoader_A = data.DataLoader(testset_A, batch_size=1, shuffle=False, num_workers=16)

    writer = SummaryWriter(os.path.join(config.log_path,config.unique_name))
    
    best_dice = 0
    for epoch in range(config.epochs):
        if epoch == 999:
            sampler = create_sampler(weight_hard=0.7, weight_medium=0.2, weight_easy=0.1)
            trainset = CMRDataset(config, data_path, mode='train', is_debug=config.DEBUG)
            trainLoader = data.DataLoader(trainset, batch_size=config.batch_size, sampler=sampler, num_workers=16)

        print('Starting epoch {}/{}'.format(epoch+1, config.epochs))
        epoch_loss = 0
        epoch_dice = 0
        for i, (img, label3d, label, img_names) in enumerate(trainLoader, 0):
            if config.DEBUG:
                label_channel= 3 # = 3
                mode = "3d" # = 3d
                for im,la,na in zip(img,label3d,img_names):
                    np_label_list = []
                    if mode == "2.5d":
                        if label_channel==1:
                            la = c1_to_c3(la)
                        np_label = la.numpy()
                        np_label_list.append(decode_pred(np_label))
                    elif mode == "3d":
                        for idx in range(img.shape[1]):
                            np_label_slice = la[idx*label_channel:(idx+1)*label_channel]
                            if label_channel==1:
                                np_label_slice = c1_to_c3(np_label_slice)
                            np_label = np_label_slice.numpy()
                            np_label_list.append(decode_pred(np_label))
                    for idx,splited_img in enumerate(im):
                        np_img = splited_img.numpy()
                        if mode == "2.5d":
                            wrt_label = np_label_list[0]
                        elif mode == "3d":
                            wrt_label = np_label_list[idx]
                        wrt_img = (np.expand_dims(np_img,-1).repeat(3,axis=-1)*255)*0.7+wrt_label*0.3
                        cv2.imwrite(os.path.join(data_path,"check_input","{0}_{1}.png".format(na,idx)),wrt_img)
            
            img = img.cuda()
            label = label.cuda()
            label3d = label3d.cuda()

            end = time.time()
            net.train()

            optimizer.zero_grad()
            
            result,result3d = net(img)


            #TODO:更改网络；3d loss计算
            loss,_ = loss_func(result,label)
            loss3d = 0
            
            loss3d += loss_func(result3d[:,:3,...],label3d[:,:3,...])[0]
            loss3d += loss_func(result3d[:,3:6,...],label3d[:,3:6,...])[0]
            loss3d += loss_func(result3d[:,6:9,...],label3d[:,6:9,...])[0]
            loss3d += loss_func(result3d[:,9:12,...],label3d[:,9:12,...])[0]
            loss3d += loss_func(result3d[:,12:15,...],label3d[:,12:15,...])[0]
            loss3d *=0.1
            
            loss += loss3d
            dice,dice_split = caculate_batch_dice(result,label)
            #debug
            #decode_result(result[0],img_names,epoch,img,label)
            loss.backward()
            optimizer.step()
            # print(loss.item())
            epoch_loss += loss.item()
            epoch_dice += dice
            batch_time = time.time() - end
            #print('batch loss: %.5f, batch_time:%.5f'%(loss.item(), batch_time))
        lr = exp_scheduler.step()
        if lr is not None:
            print("epoch_lr",lr)
        print('[epoch %d] epoch loss: %.5f'%(epoch+1, epoch_loss/(i+1)))
        print('[epoch %d] epoch dice: %.5f'%(epoch+1, epoch_dice/(i+1)))

        writer.add_scalar('Train/Loss', epoch_loss/(i+1), epoch+1)
        writer.add_scalar('Train/Dice', epoch_dice/(i+1), epoch+1)

        if epoch % 10 == 0 or epoch > config.epochs-10:
            print("save")
            torch.save(net.state_dict(), '%s/%s/CP%d.pth'%(config.cp_path, config.unique_name, epoch))
        
        if (epoch+1) >config.epochs-10 or (epoch+1) % 5 == 0:
            mean_dice,mean_loss=eval(config, net,loss_func,testLoader_A)
            writer.add_scalar('eval_dice', mean_dice, epoch+1)
            writer.add_scalar('eval_loss', mean_loss, epoch+1)
            if mean_dice >= best_dice:
                best_dice = mean_dice
                torch.save(net.state_dict(), '%s/%s/best.pth'%(config.cp_path, config.unique_name))
            print('save done',mean_dice,best_dice)

#计算一个batch的dice
def caculate_batch_dice(pred,label):
    if config.aux_loss:
        pred = pred[0]
    all_dice_split_list = np.zeros((config.num_class))
    all_dice = 0
    for batch_idx in range(label.shape[0]):
        temp = pred[batch_idx,...]
        if config.USE_3C:
            dice, dice_split_list= cal_dice_3C(pred[batch_idx,...].unsqueeze(0), label[batch_idx,...].unsqueeze(0), config.num_class)
        else:
            dice,dice_split_list = cal_dice(pred[batch_idx,...].unsqueeze(0), label[batch_idx,...].unsqueeze(0), config.num_class)
        all_dice+=dice
        all_dice_split_list+=dice_split_list
    return all_dice/label.shape[0],all_dice_split_list/label.shape[0]


#测试函数，输出可视化结果和指标,model可为字符串或模型
def eval(config,model,loss_func,dataloader=None,show_log=False,write_result = False):
    if (isinstance(model, str) ):
        net = UTNet(1, config.base_chan, config.num_class, reduce_size=config.reduce_size, block_list=config.block_list, num_blocks=config.num_blocks, num_heads=[4,4,4,4], projection='interp', attn_drop=0.1, proj_drop=0.1, rel_pos=True, aux_loss=config.aux_loss, maxpool=True)
        net.load_state_dict(torch.load(config.load))
        net.cuda()
    else:
        net = model
    net.eval()
    if  dataloader is not None:
        testLoader_A = dataloader
    else:
        testset_A = CMRDataset(config,config.data_path, mode='test', is_debug=config.DEBUG)
        testLoader_A = data.DataLoader(testset_A, batch_size=1, shuffle=False, num_workers=2)

    total_dice = 0
    total_num=0
    total_loss = 0
    with torch.no_grad():
        for i, (img, label3d, label, img_names) in enumerate(testLoader_A, 0):
            img = img.cuda()
            label = label.cuda()
            
            result,result3d = net(img)
            
            loss,loss_list = loss_func(result,label)
            
            dice,dice_split = caculate_batch_dice(result,label)

            if write_result:
                decode_result(result,img_names,img,label,config.save_img_path,loss = loss_list,dice=dice_split)
            if show_log:
                print("### losses ###:",loss_list)
                print("### dice ###:",dice,dice_split)
            total_num+=1
            total_dice+=dice
            total_loss+=loss.item()
    return total_dice/total_num,total_loss/total_num

if __name__ == '__main__':
    parser = OptionParser()
    parser.add_option('--config', dest='config', default="/mnt/home/code/UTnet/UTNet-main/configs/config_fpn.py")
    options, args = parser.parse_args()

    config = get_config(options.config)
    os.environ['CUDA_VISIBLE_DEVICES'] = config.gpu

    # if not os.path.isdir(os.path.join(config.cp_path, config.unique_name)):
    #     os.mkdir(os.path.join(config.cp_path, config.unique_name))
    # if not os.path.isdir(os.path.join(config.log_path, config.unique_name)):
    #     os.mkdir(os.path.join(config.log_path, config.unique_name))

    if config.model == 'UTNet':
        net = UTNet(config.input_channel, config.base_chan, config.num_class, reduce_size=config.reduce_size, block_list=config.block_list, num_blocks=config.num_blocks, num_heads=[4,4,4,4], projection='interp', attn_drop=0.1, proj_drop=0.1, rel_pos=True, aux_loss=config.aux_loss, maxpool=True)
    elif config.model == 'UTNet_encoder':
        # Apply transformer blocks only in the encoder
        net = UTNet_Encoderonly(1, config.base_chan, config.num_class, reduce_size=config.reduce_size, block_list=config.block_list, num_blocks=config.num_blocks, num_heads=[4,4,4,4], projection='interp', attn_drop=0.1, proj_drop=0.1, rel_pos=True, aux_loss=config.aux_loss, maxpool=True)
    elif config.model == 'FPN':
        arch = smp.FPN
        net = arch(
            encoder_name=config.backbone, encoder_weights=None, in_channels=config.input_channel,
            classes=config.num_class, activation=None)
    elif config.model == 'Unet':
        arch = smp.Unet
        model = arch(
            encoder_name=config.backbone, encoder_weights=None, in_channels=config.input_channel,
            classes=config.num_class, activation=None)
    else:
        raise NotImplementedError(config.model + " has not been implemented")

    if config.resume_model:
        net.load_state_dict(torch.load(config.resume_model))
        print('Model loaded from {}'.format(config.resume_model))
    
    param_num = sum(p.numel() for p in net.parameters() if p.requires_grad)
    
    print(net)
    print(param_num)
    
    net.cuda()

    optimizer = None
    if config.optim == 'sgd':
        optimizer = optim.SGD(net.parameters(), lr=config.lr, momentum=0.9, weight_decay=config.weight_decay)
    elif config.optim == 'adam':
        optimizer = torch.optim.AdamW(
                net.parameters(), lr=config.lr,
                weight_decay=config.weight_decay)
    else:
        print("check config.optim !!!!")
        assert False
    loss = Loss_func(config)
    
    exp_scheduler = None
    if config.scheduler == "warmup":
        exp_scheduler = Exp_lr_scheduler_with_warmup(optimizer, init_lr=config.lr, warmup_epoch=5, max_epoch=config.epochs)
    elif config.scheduler == "Exponential":
        exp_scheduler = torch.optim.lr_scheduler.ExponentialLR(
            optimizer, gamma=0.96)
    else:
        print("lr will never change!!!!")
        assert False

    train_net(net,optimizer,loss,exp_scheduler)

    print('done')

    sys.exit(0)
