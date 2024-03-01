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

# model_exp_name = '[CAD GR ours_stage2]<2023-07-13_10-16-04>'
# model_exp_name = '[CAD GR ours recon_stage2]<2023-07-28_18-08-07>'

# model_exp_name = '[CAD GR ours recon_stage2]<2023-07-28_18-08-07>'
# model_exp_name = '[CAD GR ours layer 1 head 4_stage2]<2023-09-01_15-12-29>'
# model_exp_name = '[CAD GR ours layer 1 head 16_stage2]<2023-09-01_15-12-59>'
# model_exp_name = '[CAD GR ours recon rand mask 2_stage2]<2023-08-30_10-15-36>'
# model_exp_name = '[CAD GR ours recon rand mask 4_stage2]<2023-08-30_10-15-44>'
# model_exp_name = '[CAD GR ours recon rand mask 4 layer1 head 16_stage2]<2023-09-06_08-52-38>'
# model_exp_name = '[CAD GR ours recon rand mask 4 layer1 head 64_stage2]<2023-09-06_08-53-06>'

# model_exp_name = '[CAD GR prev ImageNet pretrain VGG_stage2]<2023-07-31_09-37-52>'
# model_exp_name = '[CAD GR prev ImageNet pretrain VGG crop_stage2]<2023-07-29_09-51-27>'
# model_exp_name = '[CAD GR prev ImageNet pretrain autoencoder_stage2]<2023-07-31_10-07-31>'
# model_exp_name = '[CAD GR prev ImageNet pretrain autoencoder crop_stage2]<2023-07-29_18-37-59>'
# model_exp_name = '[CAD GR prev ImageNet pretrain HRN_stage2]<2023-07-31_10-08-36>'
# model_exp_name = '[CAD GR prev ImageNet pretrain HRN crop_stage2]<2023-07-29_18-37-23>'

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

cfg.batch_size = 1
cfg.test_batch_size = 1
cfg.num_frames = 10

cfg.actions_weights = [1., 1., 1., 1., 1.]

cfg.dataset_symbol = 'cad'
eval_net(cfg)
