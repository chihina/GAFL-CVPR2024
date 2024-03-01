import sys
sys.path.append(".")
device_list = '2'
import os
os.environ['CUDA_VISIBLE_DEVICES'] = device_list

from eval_net_gr import *

model_exp_name = '[CAD GR ours_stage1]<2023-07-09_21-47-14>'
cfg_pickle_path = os.path.join('result', model_exp_name, 'cfg.pickle')
with open(cfg_pickle_path, 'rb') as f:
    cfg = pickle.load(f)
cfg.model_exp_name = model_exp_name
cfg.inference_module_name = 'group_relation_collective'
cfg.stage1model = f'result/{cfg.model_exp_name}/stage1_epoch17_0.92%.pth'

if 'person_size' in dir(cfg):
    pass
else:
    cfg.person_size = (244,244)

cfg.device_list = device_list
cfg.use_gpu = True
cfg.use_multi_gpu = True
cfg.image_size = 320, 640
cfg.eval_only = True
cfg.eval_stage = 1
# cfg.train_seqs = [1]
# cfg.test_seqs = [4]

# vgg16 setup
cfg.backbone = 'inv3'

cfg.batch_size = 1
cfg.test_batch_size = 1
cfg.num_frames = 10


eval_net(cfg)
