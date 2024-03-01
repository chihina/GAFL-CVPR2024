import sys
sys.path.append(".")
device_list = '0'
import os
os.environ['CUDA_VISIBLE_DEVICES'] = device_list
import wandb

from train_net_dynamic import *

dataset = 'collective'
cfg=Config(dataset)
cfg.inference_module_name = 'dual_ai_collective'
cfg.device_list = device_list
cfg.use_gpu = True
cfg.use_multi_gpu = True
cfg.training_stage = 2
cfg.train_backbone = True
cfg.test_before_train = False
cfg.test_interval_epoch = 1
cfg.image_size = 240, 360
# cfg.train_seqs = [1]
# cfg.test_seqs = [5]

# vgg16 setup
cfg.backbone = 'inv3'
cfg.stage1_model_path = 'result/[CAD GR ours_stage1]<2023-07-09_21-47-14>/stage1_epoch17_0.92%.pth'
cfg.out_size=57,87

cfg.eval_only = False
cfg.num_boxes = 13
cfg.num_actions = 5
cfg.num_activities = 4
cfg.num_frames = 10
cfg.load_backbone_stage2 = True
cfg.batch_size = 8
cfg.test_batch_size = 8
cfg.train_learning_rate = 1e-4
cfg.lr_plan = {11: 3e-5, 21: 1e-5}
cfg.max_epoch = 50

cfg.exp_note = 'CAD GAR ours'
train_net(cfg)