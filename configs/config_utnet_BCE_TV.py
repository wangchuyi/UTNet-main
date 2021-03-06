from easydict import EasyDict as edict
config = edict()

#mode
config.DEBUG = False
config.EVAL = False
config.USE_3C = True
config.colors = [(0,0,0),(0,0,255),(0,255,0),(255,0,0),(125,125,125)]

#base setting
config.data_path = "/dataset/"
config.num_class = 3

config.epochs = 140
config.log_path = './log'
config.save_img_path='./show_data'
config.cp_path='./checkpoint'
config.resume_model = ""
config.input_channel = 5

config.model = "UTNet"
config.optim = "sgd"
config.weight_decay = 0.0001
config.gpu = '1'
config.batch_size = 46

config.crop_size = 256
config.unique_name = "utnet_bce_tv"
config.lr = 0.05
config.scheduler = "warmup"
config.loss = "BCE_TV"
config.loss_weight =[1,1]

#only utnet
config.aux_loss = False
config.aux_weight = [1, 0.4, 0.2, 0.1]
config.num_blocks = [1,1,1,1]
config.reduce_size = 8
#weight each class in loss function
config.weight = [0.5,1,1,1]
config.base_chan = 32
config.block_list = '1234'
#only fpn
config.backbone = "efficientnet-b3"