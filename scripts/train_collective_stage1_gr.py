import sys
sys.path.append(".")
device_list = '4'
import os
os.environ['CUDA_VISIBLE_DEVICES'] = device_list
import wandb

from train_net_stage1_gr import *

cfg=Config('collective')

cfg.device_list=device_list
cfg.training_stage=1
cfg.train_backbone=True
# cfg.train_seqs = [1]
# cfg.test_seqs = [4]
# cfg.wandb_loss_list = ['activities_acc', 'activities_conf', 'activities_MPCA',
                        # 'actions_acc', 'actions_conf', 'actions_MPCA', 'loss']
cfg.wandb_loss_list = ['activities_acc', 'actions_acc', 'loss']

cfg.image_size=480, 720
cfg.out_size=57,87
cfg.num_boxes=13
cfg.num_actions=6
cfg.num_activities=5
cfg.num_frames=10

cfg.eval_only = False
cfg.batch_size=16
cfg.test_batch_size=8 
cfg.train_learning_rate=1e-5
cfg.train_dropout_prob=0.5
cfg.weight_decay=1e-2
cfg.lr_plan={}
cfg.max_epoch=100

cfg.exp_note='CAD GR ours'
print("===> Generate wandb system")
wandb.login()
wandb.init(project=f"DIN-Group-Activity-Recognition-Benchmark-Collective",
            name=f'{cfg.exp_note}_stage1', 
            config=cfg,
            settings=wandb.Settings(start_method='fork'),
            )

train_net(cfg)
