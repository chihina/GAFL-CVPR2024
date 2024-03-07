import sys
sys.path.append(".")
device_list = '6'
import os
os.environ['CUDA_VISIBLE_DEVICES'] = device_list
import wandb

from train_net_stage2_gr import *

mode = 'PAC'
# mode = 'PAF'

dataset = 'volleyball'
cfg=Config(dataset)

cfg.mode = mode
cfg.use_recon_loss = False
cfg.use_act_loss = False
cfg.use_pose_loss = False
cfg.use_recon_diff_loss = False
cfg.use_jae_loss = False

if mode == 'PAC':
    cfg.train_backbone = True
    cfg.load_backbone_stage2 = True
    cfg.use_act_loss = True
    cfg.max_epoch = 100
    cfg.batch_size = 8
elif mode == 'PAF':
    cfg.train_backbone = False
    cfg.load_backbone_stage2 = False
    cfg.use_recon_loss = True
    cfg.max_epoch = 100
    cfg.batch_size = 16

cfg.device_list = device_list
cfg.use_gpu = True
cfg.use_multi_gpu = True
cfg.training_stage = 2
cfg.test_before_train = False
cfg.test_interval_epoch = 3
cfg.image_size = 320, 640
cfg.wandb_loss_list = ['activities_acc', 'activities_conf', 'activities_MPCA',
                        'actions_acc', 'actions_conf', 'actions_MPCA', 
                        'loss', 
                        'loss_act', 'loss_recon',
                        'loss_pseudo_rec', 'loss_cluster_balance'
                        ]

# vgg16 setup
cfg.backbone = 'vgg16'
cfg.stage1_model_path = 'result/[GR ours_stage1]<2023-07-07_23-22-44>/stage1_epoch10_0.59%.pth'
cfg.out_size = 10, 20
cfg.emb_features = 512

cfg.eval_only = False
cfg.old_act_rec = False
cfg.test_batch_size = 1
cfg.num_frames = 10

cfg.train_learning_rate = 1e-4
cfg.lr_plan = {500: 3e-5}
cfg.actions_weights = [1., 1., 2., 3., 1., 2., 2., 0.2, 1.]

# stage2 setup
cfg.use_res_connect = False
# cfg.use_trans = False
cfg.use_trans = True
cfg.use_same_enc_dual_path = False
# cfg.use_same_enc_dual_path = True
cfg.trans_head_num = 1
cfg.trans_layer_num = 1

# cfg.num_before = 0
# cfg.num_after = 0
cfg.use_ind_feat_crop = 'roi_multi'
cfg.use_ind_feat = 'loc_and_app'
cfg.people_pool_type = 'max'
# cfg.use_pos_cond = False
cfg.use_pos_cond = True
cfg.use_tmp_cond = False
# cfg.use_tmp_cond = True
cfg.final_head_mid_num = 2
cfg.use_gen_iar = False
cfg.gen_iar_ratio = 0.0

# cfg.use_random_mask = False
cfg.use_random_mask = True
if cfg.use_random_mask:
    mk_num = 5
else:
    mk_num = 0

cfg.random_mask_type = f'random_to_{mk_num}'

# cfg.inference_module_name = 'group_activity_volleyball'
cfg.inference_module_name = 'group_relation_volleyball'

if mode == 'PAC':
    cfg.exp_note = f'GR ours PAC'
elif mode == 'PAF':
    cfg.exp_note = 'GR ours PAF'

print("===> Generate wandb system")
wandb.login()
wandb.init(project=f"DIN-Group-Activity-Recognition-Benchmark-Volley",
            name=f'{cfg.exp_note}_stage2', 
            config=cfg,
            settings=wandb.Settings(start_method='fork'),
            )

train_net(cfg)
