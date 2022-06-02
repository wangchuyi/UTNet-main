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

from torch.utils.tensorboard import SummaryWriter
import time
import math
import os
import sys
import pdb
import warnings
import cv2
warnings.filterwarnings("ignore", category=UserWarning)

DEBUG = False
EVAL = False
colors = [(0,0,0),(0,0,255),(0,255,0),(255,0,0),(125,125,125)]
#画图函数，输入一个0-cls数值范围的二值图像,shape为(w,h)、返回一个bgr图像，shape为(w,h,3)
def decode_label(label):
    black_board = np.zeros((label.shape[-2],label.shape[-1],3))
    x,y = np.where(label==1)
    black_board[x,y,:] = colors[1]
    x,y = np.where(label==2)
    black_board[x,y,:] = colors[2]
    x,y = np.where(label==3)
    black_board[x,y,:] = colors[3]
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
    colors = [(0,0,0),(0,0,255),(0,255,0),(255,0,0),(125,125,125)]
    dic = {}
    for idx in range(1,result.shape[0]):
        x_s,y_s = np.where(result[idx,...]>thresh)
        for x,y in zip(x_s,y_s):
            if (x,y) not in dic:
                dic[(x,y)] = colors[idx]
        break
            # else:
            #     dic[(x,y)] = colors[-1]
            #     print("overlay!!!")
    for key in dic:
        x,y=key
        color = dic[key]
        black_board[x,y,:] = color
    return black_board
    
#解析网络输出，讲原图，pred和label画在一起。result,img,label输入均为tensor(cuda)
def decode_result(result,names,img,label,save_img_path,epoach=0,dice=None,loss=None):
    # if epoach==100:
    result = F.softmax(result, dim=1)
    result = result.cpu().detach().numpy()
    img = img.cpu().detach().numpy()
    label = label.cpu().detach().numpy()
    
    for batch_index in range(result.shape[0]):
        wrt_label = decode_label(label[batch_index,0,...])
        wrt_result = decode_pred(result[batch_index,...])

        wrt_ori_img = img[batch_index,3,...]
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
        combine2 = wrt_ori_img*mask_lab_rev +wrt_ori_img*mask_lab*0.8+wrt_label*0.2
        if dice is not None:
            cv2.putText(wrt_ori_img, "dice:", (15,20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)
            for idx,d in enumerate(dice[1:]):
                cv2.putText(wrt_ori_img, str(np.round(d,3)), (60+50*idx,20), cv2.FONT_HERSHEY_SIMPLEX, 0.5,colors[idx+1], 1)
        if loss is not None:
            cv2.putText(wrt_ori_img, "loss:", (15,45), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)
            for idx,l in enumerate(loss):
                cv2.putText(wrt_ori_img, str(np.round(l,3)), (60+50*idx,45), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)
        final = np.concatenate([wrt_ori_img,combine1, combine2], axis=1)
        cv2.imwrite(save_img_path+"{}".format(names[batch_index]),final)
    return 

def train_net(net, options):
    if EVAL:
        print(eval(options, net,show_log = True,write_result = False))
        return
    data_path = options.data_path
    
    trainset = CMRDataset(data_path, mode='train',useUT = False, crop_size=options.crop_size,is_debug = DEBUG)

    trainLoader = data.DataLoader(trainset, batch_size=options.batch_size, shuffle=True, num_workers=4)

    testset_A = CMRDataset(data_path, mode='test', useUT=False, crop_size=options.crop_size,is_debug = DEBUG)
    testLoader_A = data.DataLoader(testset_A, batch_size=32, shuffle=False, num_workers=2)

    writer = SummaryWriter(options.log_path + options.unique_name)

    optimizer = optim.SGD(net.parameters(), lr=options.lr, momentum=0.9, weight_decay=options.weight_decay)

    criterion = nn.CrossEntropyLoss(weight=torch.tensor(options.weight).cuda())
    criterion_dl = DiceLoss()
    # if DEBUG:
    #     dl,d=eval(options, net,testLoader_A)
    
    best_dice = 0
    for epoch in range(options.epochs):
        print('Starting epoch {}/{}'.format(epoch+1, options.epochs))
        epoch_loss = 0

        exp_scheduler = exp_lr_scheduler_with_warmup(optimizer, init_lr=options.lr, epoch=epoch, warmup_epoch=5, max_epoch=options.epochs)

        print('current lr:', exp_scheduler)

        for i, (img, label, img_names) in enumerate(trainLoader, 0):

            if DEBUG:
                label_channel= 1 # = 3
                mode = "2.5d" # = 3d
                for im,la,na in zip(img,label,img_names):
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

            end = time.time()
            net.train()

            optimizer.zero_grad()
            
            result = net(img)
            
            loss = 0

            pred = F.softmax(result[0], dim=1)
            _, label_pred = torch.max(pred, dim=1)
            label_pred = label_pred.view(-1, 1)
            label_true = label.view(-1, 1)
            dice, _, _ = cal_dice(label_pred, label_true, 4)


            if isinstance(result, tuple) or isinstance(result, list):
                for j in range(len(result)):
                    loss += options.aux_weight[j] * (criterion(result[j], label.squeeze(1)) + criterion_dl(result[j], label))
            else:
                loss = criterion(result, label.squeeze(1)) + criterion_dl(result, label)

            #debug
            #decode_result(result[0],img_names,epoch,img,label)
            loss.backward()
            optimizer.step()

            # print(loss.item())
            epoch_loss += loss.item()
            batch_time = time.time() - end
            print('batch loss: %.5f, batch_time:%.5f'%(loss.item(), batch_time))
            print('batch dice:',dice)
        print('[epoch %d] epoch loss: %.5f'%(epoch+1, epoch_loss/(i+1)))

        writer.add_scalar('Train/Loss', epoch_loss/(i+1), epoch+1)
        writer.add_scalar('LR', exp_scheduler, epoch+1)

        # if os.path.isdir('%s%s/'%(options.cp_path, options.unique_name)):
        #     pass
        # else:
        #     os.mkdir('%s%s/'%(options.cp_path, options.unique_name))

        if epoch % 20 == 0 or epoch > options.epochs-10:
            torch.save(net.state_dict(), '%s%s/CP%d.pth'%(options.cp_path, options.unique_name, epoch))
        
        if (epoch+1) >70 or (epoch+1) % 10 == 0:
            dl,d=eval(options, net,testLoader_A)
            writer.add_scalar('eval_dice', d, epoch+1)
            if d >= best_dice:
                best_dice = d
                torch.save(net.state_dict(), '%s%s/best.pth'%(options.cp_path, options.unique_name))

            print('save done')
            print('dice: %.5f/best dice: %.5f'%(d, best_dice))

#测试函数，输出可视化结果和指标,model可为字符串或模型
def eval(options,model,dataloader=None,show_log=False,write_result = False):
    if (isinstance(model, str) ):
        net = UTNet(1, options.base_chan, options.num_class, reduce_size=options.reduce_size, block_list=options.block_list, num_blocks=options.num_blocks, num_heads=[4,4,4,4], projection='interp', attn_drop=0.1, proj_drop=0.1, rel_pos=True, aux_loss=options.aux_loss, maxpool=True)
        # net.load_state_dict(torch.load(options.load))
        net.cuda()
    else:
        net = model
    net.eval()
    if  dataloader is not None:
        testLoader_A = dataloader
    else:
        testset_A = CMRDataset(options.data_path, mode='test', useUT=True, crop_size=options.crop_size)
        testLoader_A = data.DataLoader(testset_A, batch_size=1, shuffle=False, num_workers=2)

    criterion = nn.CrossEntropyLoss(weight=torch.tensor(options.weight).cuda())
    criterion_dl = DiceLoss()
    aux_weight = options.aux_weight
    dice_list = np.zeros(4)
    counter = [0, 0, 0, 0]
    with torch.no_grad():
        for i, (img, label, img_names) in enumerate(testLoader_A, 0):
            img = img.cuda()
            label = label.cuda()
            
            result = net(img)
            loss1= criterion(result[0], label.squeeze(1)).item()
            loss2= criterion_dl(result[0], label).item()
            loss = loss1+loss2
            loss_list = [loss1,loss2]
            
            pred = F.softmax(result[0], dim=1)
            _, label_pred = torch.max(pred, dim=1)
            label_pred = label_pred.view(-1, 1)
            label_true = label.view(-1, 1)
            dice, _, _ = cal_dice(label_pred, label_true, 4)
            dice_list += dice.cpu().numpy()
            for i, v in enumerate(dice.cpu().numpy()):
                if v != 0 or (i in label_true.cpu().numpy().reshape(1, -1).tolist()[0]):
                    counter[i] += 1
            if write_result:
                decode_result(result[0],img_names,img,label,options.save_img_path,loss=loss_list,dice=dice.cpu().numpy())
            if show_log:
                print("### losses ###:",[loss,loss1,loss2])
                print("### dice ###:",dice.cpu().numpy())
    dice_list = [(x / y) for x, y in zip(dice_list, counter)]
    return dice_list,np.mean(np.array(dice_list[1:]))

def cal_distance(label_pred, label_true, spacing):
    label_pred = label_pred.squeeze(1).cpu().numpy()
    label_true = label_true.squeeze(1).cpu().numpy()
    spacing = spacing.numpy()[0]

    ASD_list = np.zeros(3)
    HD_list = np.zeros(3)

    for i in range(3):
        tmp_surface = metrics.compute_surface_distances(label_true==(i+1), label_pred==(i+1), spacing)
        dis_gt_to_pred, dis_pred_to_gt = metrics.compute_average_surface_distance(tmp_surface)
        ASD_list[i] = (dis_gt_to_pred + dis_pred_to_gt) / 2

        HD = metrics.compute_robust_hausdorff(tmp_surface, 100)
        HD_list[i] = HD

    return ASD_list, HD_list


if __name__ == '__main__':
    parser = OptionParser()
    def get_comma_separated_int_args(option, opt, value, parser):
        value_list = value.split(',')
        value_list = [int(i) for i in value_list]
        setattr(parser.values, option.dest, value_list)

    parser.add_option('-e', '--epochs', dest='epochs', default=160, type='int', help='number of epochs')
    parser.add_option('-b', '--batch_size', dest='batch_size', default=12, type='int', help='batch size')
    parser.add_option('-l', '--learning-rate', dest='lr', default=0.05, type='float', help='learning rate')
    parser.add_option('-c', '--resume_model', type='str', dest='resume_model', default="", help='load pretrained model')
    #parser.add_option('-c', '--resume_model', type='str', dest='resume_model', default="/mnt/home/code/UTnet/UTNet-main/checkpoint/test0528_1/best.pth", help='load pretrained model')
    parser.add_option('-p', '--checkpoint-path', type='str', dest='cp_path', default='/mnt/home/code/UTnet/UTNet-main/checkpoint/', help='checkpoint path')
    parser.add_option('--data_path', type='str', dest='data_path', default='/research/cbim/vast/yg397/vision_transformer/dataset/resampled_dataset/', help='dataset path')

    parser.add_option('-o', '--log-path', type='str', dest='log_path', default='./log/', help='log path')
    parser.add_option('-s', '--save_img_path', type='str', dest='save_img_path', default='./show_data/', help='save path')
    parser.add_option('-m', type='str', dest='model', default='UTNet', help='use which model')
    parser.add_option('--num_class', type='int', dest='num_class', default=4, help='number of segmentation classes')
    parser.add_option('--base_chan', type='int', dest='base_chan', default=32, help='number of channels of first expansion in UNet')
    parser.add_option('-u', '--unique_name', type='str', dest='unique_name', default='test0527', help='unique experiment name')
    parser.add_option('--rlt', type='float', dest='rlt', default=1, help='relation between CE/FL and dice')
    parser.add_option('--weight', type='float', dest='weight',
                      default=[0.5,1,1,1] , help='weight each class in loss function')
    parser.add_option('--weight_decay', type='float', dest='weight_decay',
                      default=0.0001)
    parser.add_option('--scale', type='float', dest='scale', default=0.30)
    parser.add_option('--rotate', type='float', dest='rotate', default=180)
    parser.add_option('--crop_size', type='int', dest='crop_size', default=320)
    parser.add_option('--domain', type='str', dest='domain', default='A')
    parser.add_option('--aux_weight', type='float', dest='aux_weight', default=[1, 0.4, 0.2, 0.1])
    parser.add_option('--reduce_size', dest='reduce_size', default=10, type='int')
    parser.add_option('--block_list', dest='block_list', default='1234', type='str')
    parser.add_option('--num_blocks', dest='num_blocks', default=[1,1,1,1], type='string', action='callback', callback=get_comma_separated_int_args)
    parser.add_option('--aux_loss', dest='aux_loss', action='store_true', help='using aux loss for deep supervision')

    
    parser.add_option('--gpu', type='str', dest='gpu', default='0')
    options, args = parser.parse_args()

    os.environ['CUDA_VISIBLE_DEVICES'] = options.gpu


    if options.model == 'UTNet':
        net = UTNet(5, options.base_chan, options.num_class, reduce_size=options.reduce_size, block_list=options.block_list, num_blocks=options.num_blocks, num_heads=[4,4,4,4], projection='interp', attn_drop=0.1, proj_drop=0.1, rel_pos=True, aux_loss=options.aux_loss, maxpool=True)
    elif options.model == 'UTNet_encoder':
        # Apply transformer blocks only in the encoder
        net = UTNet_Encoderonly(1, options.base_chan, options.num_class, reduce_size=options.reduce_size, block_list=options.block_list, num_blocks=options.num_blocks, num_heads=[4,4,4,4], projection='interp', attn_drop=0.1, proj_drop=0.1, rel_pos=True, aux_loss=options.aux_loss, maxpool=True)
    elif options.model =='TransUNet':
        from model.transunet import VisionTransformer as ViT_seg
        from model.transunet import CONFIGS as CONFIGS_ViT_seg
        config_vit = CONFIGS_ViT_seg['R50-ViT-B_16']
        config_vit.n_classes = 4 
        config_vit.n_skip = 3 
        config_vit.patches.grid = (int(256/16), int(256/16))
        net = ViT_seg(config_vit, img_size=256, num_classes=4)
        #net.load_from(weights=np.load('./initmodel/R50+ViT-B_16.npz')) # uncomment this to use pretrain model download from TransUnet git repo

    elif options.model == 'ResNet_UTNet':
        from model.resnet_utnet import ResNet_UTNet
        net = ResNet_UTNet(1, options.num_class, reduce_size=options.reduce_size, block_list=options.block_list, num_blocks=options.num_blocks, num_heads=[4,4,4,4], projection='interp', attn_drop=0.1, proj_drop=0.1, rel_pos=True)
    
    elif options.model == 'SwinUNet':
        from model.swin_unet import SwinUnet, SwinUnet_config
        config = SwinUnet_config()
        net = SwinUnet(config, img_size=224, num_classes=options.num_class)
        net.load_from('./initmodel/swin_tiny_patch4_window7_224.pth')


    else:
        raise NotImplementedError(options.model + " has not been implemented")
    if options.resume_model:
        net.load_state_dict(torch.load(options.resume_model))
        print('Model loaded from {}'.format(options.resume_model))
    
    param_num = sum(p.numel() for p in net.parameters() if p.requires_grad)
    
    print(net)
    print(param_num)
    
    net.cuda()
    # print('Using model:', options.model)
    # dl,d = eval(options, net)
    # print(dl,d)


    train_net(net, options)

    print('done')

    sys.exit(0)
