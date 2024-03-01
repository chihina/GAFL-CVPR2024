import sys
sys.path.append(".")
device_list = '4'
import os
os.environ['CUDA_VISIBLE_DEVICES'] = device_list

from eval_net_gr import *

cfg=Config('volleyball')

cfg.device_list = device_list
cfg.use_gpu = True
cfg.use_multi_gpu = True
cfg.image_size = 320, 640
cfg.eval_only = True
cfg.eval_stage = 1
# cfg.train_seqs = [1]
# cfg.test_seqs = [4]

# vgg16 setup
cfg.backbone = 'vgg16'
cfg.out_size = 10, 20
cfg.emb_features = 512

# GR net setup
cfg.inference_module_name = 'group_relation_volleyball'
cfg.model_exp_name = '[GR ours_stage1]<2023-07-07_23-22-44>'
cfg.stage1model = f'result/{cfg.model_exp_name}/stage1_epoch10_0.59%.pth'

cfg.batch_size = 1
cfg.test_batch_size = 1
cfg.num_frames = 10


eval_net(cfg)
