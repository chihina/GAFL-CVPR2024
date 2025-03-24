import argparse
parser = argparse.ArgumentParser()
parser.add_argument('gpu', help='gpu_number')
parser.add_argument('name', help='model_exp_name')
args = parser.parse_args()
device_list = str(args.gpu)
model_exp_name = str(args.name)

import sys
sys.path.append(".")
import os
os.environ['CUDA_VISIBLE_DEVICES'] = device_list
import pickle

from eval_net_gr import *
from config_utils import update_config_all

# model_exp_name = '[CAD GR ours_stage2]'

stage2model = f'result/{model_exp_name}/best_model.pth'

cfg_pickle_path = os.path.join('result', model_exp_name, 'cfg.pickle')
with open(cfg_pickle_path, 'rb') as f:
    cfg = pickle.load(f)

cfg.model_exp_name = model_exp_name
cfg.stage2model = stage2model
# cfg=Config('volleyball')

cfg = update_config_all(cfg)

cfg.device_list = device_list
cfg.use_gpu = True
cfg.use_multi_gpu = True
cfg.eval_only = True
cfg.eval_stage = 2
# cfg.train_seqs = [1]
# cfg.test_seqs = [4]
# cfg.old_act_rec = False
# cfg.old_act_rec = True
cfg.eval_mask_num = 0

cfg.batch_size = 1
cfg.test_batch_size = 1
cfg.num_frames = 10

cfg.actions_weights = [1., 1., 1., 1., 1.]

cfg.dataset_symbol = 'cad'
eval_net(cfg)
