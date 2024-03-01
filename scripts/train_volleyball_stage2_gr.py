import sys
sys.path.append(".")
device_list = '6'
import os
os.environ['CUDA_VISIBLE_DEVICES'] = device_list
import wandb

from train_net_stage2_gr import *

# mode = 'PAC'
# mode = 'PAF'
# mode = 'PACF'
mode = 'PAC_DC'
# mode = 'PAC_REC'

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
elif mode == 'PAC_REC':
    cfg.train_backbone = True
    cfg.load_backbone_stage2 = True
    cfg.use_act_loss = True
    cfg.max_epoch = 100
    cfg.batch_size = 4
    # use_act_percent = 50
    # cfg.act_recognizer_path = 'result/[GAR ours dual_ai_50_stage2]<2024-01-26_22-38-41>/best_model.pth'
elif mode == 'PAF':
    cfg.train_backbone = False
    cfg.load_backbone_stage2 = False
    cfg.use_recon_loss = True
    cfg.max_epoch = 100
    cfg.batch_size = 16
elif mode == 'PAC_DC':
    cfg.train_backbone = False
    # cfg.train_backbone = True
    cfg.load_backbone_stage2 = False
    # cfg.load_backbone_stage2 = True
    # cfg.pseudo_classifier_mode = 'single_branch_transformer'
    # cfg.pseudo_classifier_mode = 'single_branch_transformer_wo_spatial'
    cfg.pseudo_classifier_mode = 'person_action_recognizor'
    # cfg.use_pseudo_act_gt = False
    cfg.use_pseudo_act_gt = True
    cfg.use_act_loss = False
    # cfg.use_act_loss = True
    cfg.max_epoch = 100
    # cfg.batch_size = 4
    cfg.batch_size = 8
    cfg.use_pseudo_act_loss_weight = False
    # cfg.use_pseudo_act_loss_weight = True
    cfg.use_cluster_balance_loss = False
    # cfg.use_cluster_balance_loss = True
else:
    cfg.train_backbone = False
    cfg.load_backbone_stage2 = True
    cfg.use_recon_loss = True
    cfg.use_act_loss = True
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
# cfg.train_learning_rate = 1e-5
# cfg.train_learning_rate = 1e-3
cfg.lr_plan = {500: 3e-5}
cfg.actions_weights = [1., 1., 2., 3., 1., 2., 2., 0.2, 1.]
# cfg.actions_weights = [1., 1., 1., 1., 1., 1., 1., 1., 1.]

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

# Dual AI setup
cfg.inference_module_name = 'group_relation_volleyball'

# HIGCIN Inference setup
# cfg.inference_module_name = 'group_relation_higcin_volleyball'
# cfg.crop_size = 7, 7

# Dynamic Inference setup
# cfg.inference_module_name = 'group_relation_din_volleyball'
# cfg.group = 1
# cfg.stride = 1
# cfg.ST_kernel_size = [(3, 3)] #[(3, 3),(3, 3),(3, 3),(3, 3)]
# cfg.dynamic_sampling = True
# cfg.sampling_ratio = [1]
# cfg.lite_dim = 128 # None # 128
# cfg.scale_factor = True
# cfg.beta_factor = False
# cfg.hierarchical_inference = False
# cfg.parallel_inference = False
# cfg.num_DIM = 1
# cfg.train_dropout_prob = 0.3

if mode == 'PAC':
    cfg.exp_note = f'GR dual_ai wo backbone pretrain wo imagenet pretrain'
elif mode == 'PAC_REC':
    cfg.exp_note = f'GR ours rand mask {mk_num} w REC act {use_act_percent}'
elif mode == 'PAF':
    cfg.exp_note = 'GR ours DIN PAF'
elif mode == 'PAC_DC':
    # cfg.exp_note = f'VOL_GR PAC_DC act loss {cfg.use_act_loss} pseudo act gt {cfg.use_pseudo_act_gt}'
    # cfg.exp_note = f'VOL_GR PAC_DC act loss {cfg.use_act_loss} pseudo act gt {cfg.use_pseudo_act_gt} 1e4 w backbone pretrain'
    # cfg.exp_note = f'VOL_GR PAC_DC act loss {cfg.use_act_loss} pseudo act gt {cfg.use_pseudo_act_gt} 1e5 w backbone pretrain'
    # cfg.exp_note = f'VOL_GR PAC_DC act loss {cfg.use_act_loss} pseudo act gt {cfg.use_pseudo_act_gt} 1e3 w backbone pretrain'
    # cfg.exp_note = f'VOL_GR PAC_DC act loss {cfg.use_act_loss} pseudo act gt {cfg.use_pseudo_act_gt} 1e4 w backbone pretrain'
    # cfg.exp_note = f'VOL_GR PAC_DC act loss {cfg.use_act_loss} pseudo act gt {cfg.use_pseudo_act_gt} w backbone pretrain'
    # cfg.exp_note = f'VOL_GR PAC_DC act loss {cfg.use_act_loss} pseudo act gt {cfg.use_pseudo_act_gt} {cfg.pseudo_classifier_mode} finetune {cfg.train_backbone}'
    cfg.exp_note = f'VOL_GR PAC_DC act loss {cfg.use_act_loss} pseudo act gt {cfg.use_pseudo_act_gt} wo weight {cfg.pseudo_classifier_mode} finetune {cfg.train_backbone}'
else:
    cfg.exp_note = f'GR ours rand mask {mk_num} w temp cond w paf loss'

print("===> Generate wandb system")
wandb.login()
wandb.init(project=f"DIN-Group-Activity-Recognition-Benchmark-Volley",
            name=f'{cfg.exp_note}_stage2', 
            config=cfg,
            settings=wandb.Settings(start_method='fork'),
            )

train_net(cfg)
