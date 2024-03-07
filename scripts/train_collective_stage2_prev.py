import sys
sys.path.append(".")
device_list = '2'
import os
os.environ['CUDA_VISIBLE_DEVICES'] = device_list
import wandb

from train_net_stage2_gr import *

dataset = 'collective'
cfg=Config(dataset)
cfg.device_list = device_list
cfg.use_gpu = True
cfg.use_multi_gpu = True
cfg.training_stage = 2
cfg.train_backbone = False
# cfg.train_backbone = True
cfg.test_before_train = False
# cfg.test_interval_epoch = 1
cfg.image_size = 240, 360
cfg.test_interval_epoch = 3
# cfg.train_seqs = [1]
# cfg.test_seqs = [5]
cfg.wandb_loss_list = ['activities_acc', 'activities_conf', 'activities_MPCA',
                        'actions_acc', 'actions_conf', 'actions_MPCA', 'loss']

# vgg16 setup
# cfg.backbone = 'inv3'
# cfg.stage1_model_path = 'result/[CAD GR ours_stage1]<2023-07-09_21-47-14>/stage1_epoch17_0.92%.pth'
# cfg.emb_features = 512
# cfg.out_size=57,87
cfg.out_size = 10, 20
cfg.emb_features = 512
cfg.backbone = 'vgg19_flat'

cfg.eval_only = False
cfg.num_boxes = 13
cfg.num_actions = 5
cfg.num_activities = 4
cfg.num_frames = 10
cfg.load_backbone_stage2 = False
# cfg.load_backbone_stage2 = True
cfg.num_graph = 4
cfg.tau_sqrt=True
cfg.batch_size = 1
# cfg.batch_size = 2
# cfg.batch_size = 4
cfg.batch_size = 8
cfg.test_batch_size = 4
# cfg.test_batch_size = 8
cfg.test_interval_epoch = 1
cfg.train_learning_rate = 5e-5
cfg.train_dropout_prob = 0.5
cfg.weight_decay = 1e-4
cfg.lr_plan = {}
# cfg.max_epoch = 30
cfg.max_epoch = 100

# stage2 setup
cfg.use_random_mask = False
# cfg.use_random_mask = True
cfg.random_mask_type = 'random'
# cfg.random_mask_type = 2
# cfg.random_mask_type = 4
# cfg.random_mask_type = 6
# cfg.random_mask_type = 8
# cfg.random_mask_type = 10

# cfg.use_recon_loss = False
cfg.use_recon_loss = True

# cfg.use_loc_feat_prev = False
cfg.use_loc_feat_prev = True

# cfg.use_ind_feat_crop = 'roi_multi'
cfg.use_ind_feat_crop = 'crop_single'

# cfg.inference_module_name = 'group_relation_ident_collective'
# cfg.exp_note = 'CAD GR prev ImageNet pretrain VGG crop with loc'

cfg.inference_module_name = 'group_relation_ae_collective'
# cfg.exp_note = 'CAD GR prev ImageNet pretrain autoencoder crop with loc'

# cfg.inference_module_name = 'group_relation_hrn_collective'
# cfg.exp_note = 'CAD GR prev ImageNet pretrain HRN crop with loc'

print("===> Generate wandb system")
wandb.login()
wandb.init(project=f"DIN-Group-Activity-Recognition-Benchmark-{dataset}",
            name=f'{cfg.exp_note}_stage2', 
            config=cfg,
            settings=wandb.Settings(start_method='fork'),
            )

train_net(cfg)
