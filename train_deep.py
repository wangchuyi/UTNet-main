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
            # else:
            #     dic[(x,y)] = colors[-1]
            #     print("overlay!!!")
    for key in dic:
        x,y=key
        color = dic[key]
        black_board[x,y,:] = color
    return black_board
    
#解析网络输出，讲原图，pred和label画在一起。result,img,label输入均为tensor(cuda)
def decode_result(result,names,epoach,img,label,dst_path = "./show_data/"):
    # if epoach==100:
    result = F.softmax(result, dim=1)
    result = result.cpu().detach().numpy()
    img = img.cpu().detach().numpy()
    label = label.cpu().detach().numpy()
    
    for batch_index in range(result.shape[0]):
        wrt_label = decode_label(label[batch_index,0,...])
        wrt_result = decode_pred(result[batch_index,...])
        wrt_ori_img = img[batch_index,0,...]
        wrt_ori_img = np.expand_dims(wrt_ori_img,-1).repeat(3,axis=-1)
        final = np.concatenate([wrt_ori_img,wrt_result, wrt_label], axis=1)
        cv2.imwrite(dst_path+"{}".format(names[batch_index]),final)

def train_net(net, options):
    data_path = options.data_path

    trainset = CMRDataset(data_path, mode='train', domain=options.domain, debug=DEBUG, scale=options.scale, rotate=options.rotate, crop_size=options.crop_size)
    trainLoader = data.DataLoader(trainset, batch_size=options.batch_size, shuffle=True, num_workers=16)

    testset_A = CMRDataset(data_path, mode='test', domain='A', debug=DEBUG, crop_size=options.crop_size)
    testLoader_A = data.DataLoader(testset_A, batch_size=1, shuffle=False, num_workers=2)

    writer = SummaryWriter(options.log_path + options.unique_name)

    optimizer = optim.SGD(net.parameters(), lr=options.lr, momentum=0.9, weight_decay=options.weight_decay)

    criterion = nn.CrossEntropyLoss(weight=torch.tensor(options.weight).cuda())
    criterion_dl = DiceLoss()


    best_dice = 0
    for epoch in range(options.epochs):
        print('Starting epoch {}/{}'.format(epoch+1, options.epochs))
        epoch_loss = 0

        exp_scheduler = exp_lr_scheduler_with_warmup(optimizer, init_lr=options.lr, epoch=epoch, warmup_epoch=5, max_epoch=options.epochs)

        print('current lr:', exp_scheduler)

        for i, (img, label, img_names) in enumerate(trainLoader, 0):

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

            decode_result(result[0],img_names,epoch,img,label)
            loss.backward()
            optimizer.step()

            print(loss.item())
            epoch_loss += loss.item()
            batch_time = time.time() - end
            print('batch loss: %.5f, batch_time:%.5f'%(loss.item(), batch_time))
            print('batch dice: %.5f, batch_time:%.5f'%(dice.item(), batch_time))
        print('[epoch %d] epoch loss: %.5f'%(epoch+1, epoch_loss/(i+1)))

        writer.add_scalar('Train/Loss', epoch_loss/(i+1), epoch+1)
        writer.add_scalar('LR', exp_scheduler, epoch+1)

        # if os.path.isdir('%s%s/'%(options.cp_path, options.unique_name)):
        #     pass
        # else:
        #     os.mkdir('%s%s/'%(options.cp_path, options.unique_name))

        if epoch % 20 == 0 or epoch > options.epochs-10:
            torch.save(net.state_dict(), '%s%s/CP%d.pth'%(options.cp_path, options.unique_name, epoch))
        
        # if (epoch+1) >0 or (epoch+1) % 10 == 0:
        #     dice_list_A, ASD_list_A, HD_list_A = validation(net, testLoader_A, options)
        #     log_evaluation_result(writer, dice_list_A, ASD_list_A, HD_list_A, 'A', epoch)
            
        #     dice_list_B, ASD_list_B, HD_list_B = validation(net, testLoader_B, options)
        #     log_evaluation_result(writer, dice_list_B, ASD_list_B, HD_list_B, 'B', epoch)

        #     dice_list_C, ASD_list_C, HD_list_C = validation(net, testLoader_C, options)
        #     log_evaluation_result(writer, dice_list_C, ASD_list_C, HD_list_C, 'C', epoch)

        #     dice_list_D, ASD_list_D, HD_list_D = validation(net, testLoader_D, options)
        #     log_evaluation_result(writer, dice_list_D, ASD_list_D, HD_list_D, 'D', epoch)


        #     AVG_dice_list = 20 * dice_list_A + 50 * dice_list_A + 50 * dice_list_A + 50 * dice_list_A
        #     AVG_dice_list /= 170

        #     AVG_ASD_list = 20 * ASD_list_A + 50 * ASD_list_A + 50 * ASD_list_A + 50 * ASD_list_A
        #     AVG_ASD_list /= 170

        #     AVG_HD_list = 20 * HD_list_A + 50 * HD_list_A + 50 * HD_list_A + 50 * HD_list_A
        #     AVG_HD_list /= 170

        #     log_evaluation_result(writer, AVG_dice_list, AVG_ASD_list, AVG_HD_list, 'mean', epoch)



        #     if dice_list_A.mean() >= best_dice:
        #         best_dice = dice_list_A.mean()
        #         torch.save(net.state_dict(), '%s%s/best.pth'%(options.cp_path, options.unique_name))

        #     print('save done')
        #     print('dice: %.5f/best dice: %.5f'%(dice_list_A.mean(), best_dice))

#测试函数，输出可视化结果和指标,model可为字符串或模型
def eval(options,model):
    if (isinstance(model, str) ):
        net = UTNet(1, options.base_chan, options.num_class, reduce_size=options.reduce_size, block_list=options.block_list, num_blocks=options.num_blocks, num_heads=[4,4,4,4], projection='interp', attn_drop=0.1, proj_drop=0.1, rel_pos=True, aux_loss=options.aux_loss, maxpool=True)
        # net.load_state_dict(torch.load(options.load))
        net.cuda()
    else:
        net = model
    net.eval() 
    testset_A = CMRDataset(options.data_path, mode='test', domain='A', debug=False, crop_size=256)
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
            
            decode_result(result[0],img_names,0,img,label)

            pred = F.softmax(result[0], dim=1)
            _, label_pred = torch.max(pred, dim=1)
            label_pred = label_pred.view(-1, 1)
            label_true = label.view(-1, 1)
            dice, _, _ = cal_dice(label_pred, label_true, 4)

            dice_list += dice.cpu().numpy()

            for i, v in enumerate(dice.cpu().numpy()):
                if v != 0 or (i in label_true.cpu().numpy().reshape(1, -1).tolist()[0]):
                    counter[i] += 1

            print("### losses ###:",[loss,loss1,loss2])
            print("### dice ###:",dice.cpu().numpy())

    return [(x / y) for x, y in zip(dice_list, counter)]

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

    parser.add_option('-e', '--epochs', dest='epochs', default=170, type='int', help='number of epochs')
    parser.add_option('-b', '--batch_size', dest='batch_size', default=32, type='int', help='batch size')
    parser.add_option('-l', '--learning-rate', dest='lr', default=0.05, type='float', help='learning rate')
    parser.add_option('-c', '--resume', type='str', dest='load', default=False, help='load pretrained model')
    parser.add_option('-p', '--checkpoint-path', type='str', dest='cp_path', default='./checkpoint/', help='checkpoint path')
    parser.add_option('--data_path', type='str', dest='data_path', default='/research/cbim/vast/yg397/vision_transformer/dataset/resampled_dataset/', help='dataset path')

    parser.add_option('-o', '--log-path', type='str', dest='log_path', default='./log/', help='log path')
    parser.add_option('-m', type='str', dest='model', default='UTNet', help='use which model')
    parser.add_option('--num_class', type='int', dest='num_class', default=4, help='number of segmentation classes')
    parser.add_option('--base_chan', type='int', dest='base_chan', default=32, help='number of channels of first expansion in UNet')
    parser.add_option('-u', '--unique_name', type='str', dest='unique_name', default='test', help='unique experiment name')
    parser.add_option('--rlt', type='float', dest='rlt', default=1, help='relation between CE/FL and dice')
    parser.add_option('--weight', type='float', dest='weight',
                      default=[0.5,1,1,1] , help='weight each class in loss function')
    parser.add_option('--weight_decay', type='float', dest='weight_decay',
                      default=0.0001)
    parser.add_option('--scale', type='float', dest='scale', default=0.30)
    parser.add_option('--rotate', type='float', dest='rotate', default=180)
    parser.add_option('--crop_size', type='int', dest='crop_size', default=256)
    parser.add_option('--domain', type='str', dest='domain', default='A')
    parser.add_option('--aux_weight', type='float', dest='aux_weight', default=[1, 0.4, 0.2, 0.1])
    parser.add_option('--reduce_size', dest='reduce_size', default=8, type='int')
    parser.add_option('--block_list', dest='block_list', default='1234', type='str')
    parser.add_option('--num_blocks', dest='num_blocks', default=[1,1,1,1], type='string', action='callback', callback=get_comma_separated_int_args)
    parser.add_option('--aux_loss', dest='aux_loss', action='store_true', help='using aux loss for deep supervision')

    
    parser.add_option('--gpu', type='str', dest='gpu', default='0')
    options, args = parser.parse_args()

    os.environ['CUDA_VISIBLE_DEVICES'] = options.gpu


    if options.model == 'UTNet':
        net = UTNet(1, options.base_chan, options.num_class, reduce_size=options.reduce_size, block_list=options.block_list, num_blocks=options.num_blocks, num_heads=[4,4,4,4], projection='interp', attn_drop=0.1, proj_drop=0.1, rel_pos=True, aux_loss=options.aux_loss, maxpool=True)
        # net.load_state_dict(torch.load("/mnt/home/code/UTnet/UTNet-main/checkpoint/test/CP169.pth"))
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
    if options.load:
        net.load_state_dict(torch.load(options.load))
        print('Model loaded from {}'.format(options.load))
    
    param_num = sum(p.numel() for p in net.parameters() if p.requires_grad)
    
    print(net)
    print(param_num)
    
    net.cuda()
    # print('Using model:', options.model)
    d = eval(options, net)
    print(d)


    train_net(net, options)

    print('done')

    sys.exit(0)
