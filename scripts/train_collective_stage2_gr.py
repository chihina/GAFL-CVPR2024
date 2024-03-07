import sys
sys.path.append(".")
device_list = '3'
import os
os.environ['CUDA_VISIBLE_DEVICES'] = device_list
import wandb

from train_net_stage2_gr import *

# mode = 'PAC'
mode = 'PAF'

dataset = 'collective'
cfg=Config(dataset)

cfg.mode = mode
cfg.use_recon_loss = False
cfg.use_act_loss = False
cfg.use_pose_loss = False
cfg.use_recon_diff_loss = False

if mode == 'PAC':
    cfg.train_backbone = True
    cfg.load_backbone_stage2 = True
    cfg.use_act_loss = True
    cfg.max_epoch = 100
    cfg.batch_size = 8
else:
    cfg.train_backbone = False
    cfg.load_backbone_stage2 = False
    cfg.use_recon_loss = True
    cfg.max_epoch = 100
    # cfg.batch_size = 16
    cfg.batch_size = 4

cfg.device_list = device_list
cfg.use_gpu = True
cfg.use_multi_gpu = True
cfg.training_stage = 2
cfg.test_before_train = False
cfg.test_interval_epoch = 3
cfg.image_size = 240, 360
cfg.wandb_loss_list = ['activities_acc', 'activities_conf', 'activities_MPCA',
                        'actions_acc', 'actions_conf', 'actions_MPCA', 
                        'loss', 
                        # 'loss_act', 'loss_recon',
                        # 'loss_pseudo_rec', 'loss_cluster_balance'
                        ]

# vgg16 setup
cfg.backbone = 'inv3'
cfg.stage1_model_path = 'result/[CAD GR ours_stage1]<2023-07-09_21-47-14>/stage1_epoch17_0.92%.pth'
cfg.out_size=57,87

cfg.eval_only = False
cfg.old_act_rec = False
cfg.test_batch_size = 1
cfg.num_boxes = 13
cfg.num_actions = 5
cfg.num_activities = 4
cfg.num_frames = 10

cfg.train_learning_rate = 5e-5
cfg.train_dropout_prob = 0.5
cfg.weight_decay = 1e-4
cfg.lr_plan = {}
cfg.max_epoch = 100

# stage2 setup
cfg.use_res_connect = False
cfg.use_trans = True
cfg.trans_head_num = 1
cfg.trans_layer_num = 1

cfg.use_ind_feat_crop = 'roi_multi'
cfg.use_ind_feat = 'loc_and_app'
cfg.people_pool_type = 'max'
cfg.use_pos_cond = True
cfg.use_tmp_cond = False
# cfg.use_tmp_cond = True
cfg.final_head_mid_num = 2
cfg.use_gen_iar = False
cfg.gen_iar_ratio = 0.0

cfg.use_random_mask = False
# cfg.use_random_mask = True
mk_num = 12
cfg.random_mask_type = f'random_to_{mk_num}'

# cfg.inference_module_name = 'group_activity_collective'
cfg.inference_module_name = 'group_relation_collective'

if mode == 'PAC':
    cfg.exp_note = 'CAD GR ours DIN PAC'
elif mode == 'PAF':
    cfg.exp_note = 'CAD GR ours HIGCIN PAF'
else:
    raise NotImplementedError

print("===> Generate wandb system")
wandb.login()
wandb.init(project=f"DIN-Group-Activity-Recognition-Benchmark-{dataset}",
            name=f'{cfg.exp_note}_stage2', 
            config=cfg,
            settings=wandb.Settings(start_method='fork'),
            )

train_net(cfg)
