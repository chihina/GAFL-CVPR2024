import argparse
parser = argparse.ArgumentParser()
parser.add_argument('gpu', help='gpu_number')
parser.add_argument('name', help='model_exp_name')
parser.add_argument('-eval_mask_num', help='eval_mask_num', default=0, type=int)
args = parser.parse_args()
device_list = str(args.gpu)
model_exp_name = str(args.name)
eval_mask_num = float(args.eval_mask_num)

import sys
sys.path.append(".")
import os
os.environ['CUDA_VISIBLE_DEVICES'] = device_list
import pickle

from eval_net_gr import *
from config_utils import update_config_all

if (model_exp_name == 'GAFL_PAC') or (model_exp_name == 'GAFL_PAF'):
    stage2model = f'result/{model_exp_name}/best_model.pth'
    cfg_pickle_path = os.path.join('result', model_exp_name, 'cfg.pickle')
    with open(cfg_pickle_path, 'rb') as f:
        cfg = pickle.load(f)
    cfg.model_exp_name = model_exp_name
    cfg.stage2model = stage2model
elif 'GA' in model_exp_name:
    stage2model = f'result/{model_exp_name}/best_model.pth'
    cfg=Config('volleyball')
    cfg.model_exp_name = model_exp_name
    cfg.stage2model = stage2model
    cfg.inference_module_name = 'group_activity_volleyball'
    cfg.emb_features = 512
    cfg.backbone = 'vgg16'
    cfg.out_size = (10, 20)
    cfg.actions_weights = [1., 1., 2., 3., 1., 2., 2., 0.2, 1.]
else:
    stage2model = f'result/{model_exp_name}/best_model.pth'
    cfg_pickle_path = os.path.join('result', model_exp_name, 'cfg.pickle')
    with open(cfg_pickle_path, 'rb') as f:
        cfg = pickle.load(f)
    cfg.model_exp_name = model_exp_name
    cfg.stage2model = stage2model

cfg = update_config_all(cfg)

cfg.device_list = device_list
cfg.use_gpu = True
cfg.use_multi_gpu = True
cfg.image_size = 320, 640
cfg.eval_only = True
cfg.eval_stage = 2
# cfg.train_seqs = [1]
# cfg.test_seqs = [4]
# cfg.old_act_rec = False
# cfg.old_act_rec = True
cfg.eval_mask_num = eval_mask_num

# vgg16 setup
# cfg.backbone = 'vgg16'
# cfg.out_size = 10, 20
# cfg.emb_features = 512

# GA net setup
# cfg.inference_module_name = 'group_activity_volleyball'
# cfg.model_exp_name = '[GA ours finetune_stage2]<2023-07-07_09-43-38>'
# cfg.stage2model = f'result/{cfg.model_exp_name}/stage2_epoch3_0.70%.pth'

# Prev net setup
# cfg.inference_module_name = 'group_relation_volleyball'

cfg.batch_size = 1
cfg.test_batch_size = 1
cfg.num_frames = 10

cfg.dataset_symbol = 'vol'
eval_net(cfg)
