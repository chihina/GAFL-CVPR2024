import sys
sys.path.append(".")
device_list = '3'
import os
os.environ['CUDA_VISIBLE_DEVICES'] = device_list
import wandb

from train_net_stage3_gr import *
from config_utils import update_config_all

model_exp_name = '[CAD GA ours rand mask 0_stage2]<2023-10-23_13-28-02>'
cfg_pickle_path = os.path.join('result', model_exp_name, 'cfg.pickle')
with open(cfg_pickle_path, 'rb') as f:
    cfg = pickle.load(f)
cfg.exp_name = None

dataset = 'collective'
# cfg=Config(dataset)
cfg.device_list = device_list
cfg.use_gpu = True
cfg.use_multi_gpu = True
cfg.training_stage = 3
# cfg.train_backbone = False
cfg.train_backbone = True
cfg.test_before_train = False
cfg.test_interval_epoch = 1
# cfg.test_interval_epoch = 3
cfg.image_size = 240, 360
# cfg.train_seqs = [1]
# cfg.test_seqs = [5]
cfg.wandb_loss_list = ['activities_acc', 'activities_conf', 'activities_MPCA',
                        'actions_acc', 'actions_conf', 'actions_MPCA', 'loss']

# vgg16 setup
# cfg.image_size=480, 720
# cfg.backbone = 'inv3'
# cfg.stage1_model_path = 'result/[CAD GR ours_stage1]<2023-07-09_21-47-14>/stage1_epoch17_0.92%.pth'

# additional setup for stage2
cfg = update_config_all(cfg)
cfg.stage2model = os.path.join('result', model_exp_name, 'best_model.pth')

# additional setup for stage3
cfg.use_recon_loss = False
cfg.use_random_mask = False

cfg.eval_only = False
cfg.batch_size = 8
cfg.test_batch_size = 8
cfg.num_frames = 10
cfg.train_learning_rate = 5e-5
cfg.lr_plan = {}
# cfg.lr_plan = {11: 3e-5, 21: 1e-5}
cfg.max_epoch = 50

cfg.inference_module_name = 'group_relation_collective'
cfg.exp_note = 'CAD GR ours rand mask 0'

print("===> Generate wandb system")
wandb.login()
wandb.init(project=f"DIN-Group-Activity-Recognition-Benchmark-{dataset}",
            name=f'{cfg.exp_note}_stage3', 
            config=cfg,
            settings=wandb.Settings(start_method='fork'),
            )

train_net(cfg)
