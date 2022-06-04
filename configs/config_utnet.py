from easydict import EasyDict as edict
config = edict()

#mode
config.DEBUG = True
config.EVAL = False
config.USE_3C = False
config.colors = [(0,0,0),(0,0,255),(0,255,0),(255,0,0),(125,125,125)]

#base setting
config.data_path = "/mnt/home/code/UTnet/dataset/"
config.num_class =4

config.epochs = 160

config.log_path = './log'
config.save_img_path='./show_data'
config.cp_path='./checkpoint'
config.resume_model = ""
config.input_channel = 5

config.model = "UTNet"
config.optim = "sgd"
config.weight_decay = 0.0001
config.gpu = '0'
config.batch_size = 46

config.crop_size = 256
config.unique_name = "utnet_ori"
config.lr = 0.05
config.loss = "CE_DICE"
config.loss_weight =[1,1]

#only utnet
config.aux_loss = True
config.aux_weight = [1, 0.4, 0.2, 0.1]
config.num_blocks = [1,1,1,1]
config.reduce_size = 8
#weight each class in loss function
config.weight = [0.5,1,1,1]
config.base_chan = 32
config.block_list = '1234'
#only fpn
config.backbone = "efficientnet-b3"