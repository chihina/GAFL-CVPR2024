import sys
sys.path.append(".")
device_list = '4'
import os
os.environ['CUDA_VISIBLE_DEVICES'] = device_list
import wandb

from train_net_stage1_gr import *

cfg=Config('volleyball')

cfg.use_multi_gpu = False
cfg.device_list=device_list
cfg.training_stage=1
cfg.stage1_model_path=''
cfg.train_backbone=True
cfg.test_before_train = True
# cfg.train_seqs = [1]
# cfg.test_seqs = [4]
# cfg.wandb_loss_list = ['activities_acc', 'activities_conf', 'activities_MPCA',
                        # 'actions_acc', 'actions_conf', 'actions_MPCA', 'loss']
cfg.wandb_loss_list = ['activities_acc', 'actions_acc', 'loss']

# VGG16
cfg.backbone = 'vgg16'
cfg.image_size = 720, 1280
cfg.out_size = 22, 40
cfg.emb_features = 512

cfg.num_before = 5
cfg.num_after = 4

cfg.eval_only = False
cfg.batch_size=8
cfg.test_batch_size=1
cfg.num_frames=1
# cfg.train_learning_rate=1e-5
# cfg.lr_plan={}
# cfg.max_epoch=200
cfg.train_learning_rate=1e-4
cfg.lr_plan={30:5e-5, 60:2e-5, 90:1e-5}
cfg.max_epoch=120
cfg.set_bn_eval = False
cfg.actions_weights=[1., 1., 2., 3., 1., 2., 2., 0.2, 1.]

cfg.exp_note = 'GR ours'
print("===> Generate wandb system")
wandb.login()
wandb.init(project=f"DIN-Group-Activity-Recognition-Benchmark-Volley",
            name=f'{cfg.exp_note}_stage1', 
            config=cfg,
            settings=wandb.Settings(start_method='fork'),
            )

train_net(cfg)
