import sys
sys.path.append(".")
device_list = '3'
import os
os.environ['CUDA_VISIBLE_DEVICES'] = device_list
import wandb

from train_net_stage3_gr import *
from config_utils import update_config_all

# model_exp_name = '[GR ours rand mask 5_stage2]<2023-10-16_22-26-54>'
model_exp_name = '[GR ours recon feat random mask 6_stage2]<2023-10-18_09-34-15>'
# model_exp_name = '[GR ours rand mask 4 w temp cond_stage2]<2023-11-09_09-55-19>'
# model_exp_name = '[GR ours recon feat random mask 8 w temp cond_stage2]<2023-11-10_13-32-14>'

cfg_pickle_path = os.path.join('result', model_exp_name, 'cfg.pickle')
with open(cfg_pickle_path, 'rb') as f:
    cfg = pickle.load(f)
cfg.exp_name = None

# cfg=Config('volleyball')
cfg.device_list = device_list
cfg.use_gpu = True
cfg.use_multi_gpu = True
cfg.training_stage = 3
# cfg.train_backbone = False
cfg.train_backbone = True
cfg.test_before_train = False
# cfg.test_interval_epoch = 1
cfg.test_interval_epoch = 2
cfg.image_size = 320, 640
# cfg.train_seqs = [1]
# cfg.test_seqs = [4]
cfg.wandb_loss_list = ['activities_acc', 'activities_conf', 'activities_MPCA',
                        'actions_acc', 'actions_conf', 'actions_MPCA', 'loss']

# vgg16 setup
# cfg.backbone = 'vgg16'
# cfg.stage1_model_path = 'result/[GR ours_stage1]<2023-07-07_23-22-44>/stage1_epoch10_0.59%.pth'
# cfg.out_size = 22, 40
cfg.out_size = 10, 20
cfg.emb_features = 512

# additional setup for stage2
cfg.person_size = (244,244)
cfg = update_config_all(cfg)
cfg.stage2model = os.path.join('result', model_exp_name, 'best_model.pth')
# cfg.load_backbone_stage3 = False
cfg.load_backbone_stage3 = True

# additional setup for stage3
cfg.use_recon_loss = False
cfg.use_random_mask = False
cfg.use_act_loss = False
# cfg.use_act_loss = True
cfg.use_tmp_cond = False
# cfg.use_tmp_cond = True

cfg.eval_only = False
# cfg.batch_size = 2
cfg.batch_size = 8
cfg.test_batch_size = 1
cfg.num_frames = 10

cfg.max_epoch = 100
cfg.train_learning_rate = 1e-4
cfg.lr_plan = {11: 3e-5, 21: 1e-5}
# cfg.train_learning_rate = 1e-5
# cfg.lr_plan = {11: 3e-6, 21: 1e-6}
# cfg.train_learning_rate = 1e-6
# cfg.lr_plan = {11: 3e-7, 21: 1e-7}

cfg.actions_weights = [1., 1., 2., 3., 1., 2., 2., 0.2, 1.]

# cfg.exp_note = 'GR ours rand mask 5 w lr 1e4 ga layer_ori'
# cfg.exp_note = 'GR ours rand mask 5 w lr 1e4 ga layer_ori w act loss'
# cfg.exp_note = 'GR ours rand mask 5 w lr 1e4 ga layer_ori wo pretrain'
# cfg.exp_note = 'GR ours rand mask 5 w lr 1e4 ga layer_ori w pretrain'

# cfg.exp_note = 'GR ours rand mask 4 temp cond w lr 1e4 ga layer_ori'
# cfg.exp_note = 'GR ours recon feat random mask 8 temp cond w lr 1e4 ga layer_ori'
# cfg.exp_note = 'GR ours recon feat random mask 8 temp cond w lr 1e4 ga layer_ori wo train backbone'

# cfg.exp_note = 'GR ours recon feat random mask 6 w lr 1e4 ga layer_ori'
# cfg.exp_note = 'GR ours recon feat random mask 6 w lr 1e5 ga layer_ori'
# cfg.exp_note = 'GR ours recon feat random mask 6 w lr 1e6 ga layer_ori'
cfg.exp_note = 'GR ours recon feat random mask 6 w lr 1e4 wo schedule ga layer_ori'

print("===> Generate wandb system")
wandb.login()
wandb.init(project=f"DIN-Group-Activity-Recognition-Benchmark-Volley",
            name=f'{cfg.exp_note}_stage3', 
            config=cfg,
            settings=wandb.Settings(start_method='fork'),
            )

train_net(cfg)
