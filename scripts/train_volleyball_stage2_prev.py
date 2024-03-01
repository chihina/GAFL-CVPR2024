import sys
sys.path.append(".")
device_list = '1'
import os
os.environ['CUDA_VISIBLE_DEVICES'] = device_list
import wandb

from train_net_stage2_gr import *

cfg=Config('volleyball')
cfg.device_list = device_list
cfg.use_gpu = True
cfg.use_multi_gpu = True
cfg.training_stage = 2
cfg.train_backbone = False
# cfg.train_backbone = True
cfg.test_before_train = False
cfg.test_interval_epoch = 1
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
cfg.backbone = 'vgg19_flat'

cfg.eval_only = False
# cfg.batch_size = 1
# cfg.batch_size = 2
# cfg.batch_size = 4
cfg.batch_size = 8
# cfg.batch_size = 16
cfg.test_batch_size = 1
cfg.num_frames = 10
cfg.load_backbone_stage2 = False
# cfg.load_backbone_stage2 = True
# cfg.train_learning_rate = 1e-4
# cfg.lr_plan = {11: 3e-5, 21: 1e-5}
# cfg.max_epoch = 60
# cfg.max_epoch = 100
cfg.train_learning_rate = 1e-4
cfg.lr_plan = {100: 3e-5}
cfg.max_epoch = 100
# cfg.max_epoch = 1
# cfg.lr_plan = {11: 3e-5, 21: 1e-5}
# cfg.lr_plan = {11: 1e-5}
# cfg.max_epoch = 30
cfg.actions_weights = [1., 1., 2., 3., 1., 2., 2., 0.2, 1.]

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

# cfg.use_ind_feat_crop = 'roi_multi'
cfg.use_ind_feat_crop = 'crop_single'

# cfg.use_loc_feat_prev = False
cfg.use_loc_feat_prev = True

# cfg.inference_module_name = 'group_activity_volleyball'
# cfg.exp_note = 'GA ours finetune'

# cfg.inference_module_name = 'group_relation_ident_volleyball'
# cfg.exp_note = 'GR prev VOL finetune VGG'
# cfg.exp_note = 'GR prev ImageNet pretrain VGG'
# cfg.exp_note = 'GR prev ImageNet pretrain VGG crop'
# cfg.exp_note = 'GR prev ImageNet pretrain VGG crop with loc'

# cfg.inference_module_name = 'group_relation_ae_volleyball'
# cfg.exp_note = 'GR prev VOL finetune autoencoder'
# cfg.exp_note = 'GR prev ImageNet pretrain autoencoder'
# cfg.exp_note = 'GR prev ImageNet pretrain autoencoder crop'
# cfg.exp_note = 'GR prev ImageNet pretrain autoencoder crop with loc'

cfg.inference_module_name = 'group_relation_hrn_volleyball'
# cfg.exp_note = 'GR prev VOL finetune HRN'
# cfg.exp_note = 'GR prev ImageNet pretrain HRN'
# cfg.exp_note = 'GR prev ImageNet pretrain HRN crop'
# cfg.exp_note = 'GR prev ImageNet pretrain HRN crop sigmoid'
cfg.exp_note = 'GR prev ImageNet pretrain HRN crop sigmoid with loc'

print("===> Generate wandb system")
wandb.login()
wandb.init(project=f"DIN-Group-Activity-Recognition-Benchmark-Volley",
            name=f'{cfg.exp_note}_stage2', 
            config=cfg,
            settings=wandb.Settings(start_method='fork'),
            )

train_net(cfg)
