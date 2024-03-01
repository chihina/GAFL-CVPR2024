import sys
sys.path.append(".")
device_list = '2'
import os
os.environ['CUDA_VISIBLE_DEVICES'] = device_list
import wandb

from train_net_dynamic import *

dataset = 'volleyball'
cfg=Config(dataset)
cfg.inference_module_name = 'dual_ai_volleyball'
cfg.device_list = device_list
cfg.use_gpu = True
cfg.use_multi_gpu = True
cfg.training_stage = 2
cfg.train_backbone = True
cfg.test_before_train = False
cfg.test_interval_epoch = 1
cfg.image_size = 320, 640
# cfg.train_seqs = [1]
# cfg.test_seqs = [4]

# vgg16 setup
cfg.backbone = 'vgg16'
cfg.stage1_model_path = 'result/[GR ours_stage1]<2023-07-07_23-22-44>/stage1_epoch10_0.59%.pth'
# cfg.out_size = 22, 40
cfg.out_size = 10, 20
cfg.emb_features = 512

cfg.eval_only = False
cfg.batch_size = 8
cfg.test_batch_size = 1
cfg.num_frames = 10
cfg.load_backbone_stage2 = True
cfg.train_learning_rate = 1e-4
cfg.lr_plan = {11: 3e-5, 21: 1e-5}
# cfg.max_epoch = 60
# cfg.lr_plan = {11: 3e-5, 21: 1e-5}
# cfg.lr_plan = {11: 1e-5}
cfg.max_epoch = 30
cfg.actions_weights = [1., 1., 2., 3., 1., 2., 2., 0.2, 1.]

use_act_percent = 10
use_act_ratio = use_act_percent / 100
cfg.train_seqs = cfg.train_seqs[:int(len(cfg.train_seqs)*use_act_ratio)]

cfg.exp_note = f'GAR ours dual_ai_{use_act_percent}'
train_net(cfg)