import sys
sys.path.append(".")
device_list = '6'
import os
os.environ['CUDA_VISIBLE_DEVICES'] = device_list
import wandb

from train_net_stage2_gr import *

cfg=Config('volleyball')
cfg.inference_module_name = 'higcin_volleyball'

cfg.device_list = device_list
cfg.use_gpu = True
cfg.use_multi_gpu = True
cfg.training_stage = 2
cfg.train_backbone = True
cfg.test_before_train = False
cfg.test_interval_epoch = 1
cfg.image_size = 320, 640
cfg.wandb_loss_list = ['activities_acc', 'activities_conf', 'activities_MPCA',
                        'actions_acc', 'actions_conf', 'actions_MPCA', 'loss']

# vgg16 setup
cfg.backbone = 'vgg16'
# cfg.stage1_model_path = 'result/basemodel_VD_vgg16.pth'
cfg.stage1_model_path = 'result/[GR ours_stage1]<2023-07-07_23-22-44>/stage1_epoch10_0.59%.pth'
# cfg.out_size = 22, 40
cfg.out_size = 10, 20
cfg.emb_features = 512

# res18 setup
# cfg.backbone = 'res18'
# cfg.stage1_model_path = 'result/basemodel_VD_res18.pth'
# cfg.out_size = 23, 40
# cfg.emb_features = 512

# HIGCIN
cfg.crop_size = 7, 7

# Dynamic Inference setup
cfg.group = 1
cfg.stride = 1
cfg.ST_kernel_size = (3, 3)
cfg.dynamic_sampling = True
cfg.sampling_ratio = [1]  # [1,2,4]
cfg.lite_dim = None # 128
cfg.scale_factor = True
cfg.beta_factor = False
cfg.hierarchical_inference = False
cfg.parallel_inference = False

cfg.eval_only = False
cfg.batch_size = 8
cfg.test_batch_size = 1
cfg.num_frames = 10
cfg.load_backbone_stage2 = True
# cfg.train_learning_rate = 1e-4
cfg.train_learning_rate = 1e-5
# cfg.lr_plan = {11: 3e-5, 21: 1e-5}
# cfg.max_epoch = 60
# cfg.lr_plan = {11: 3e-5, 21: 1e-5}
cfg.lr_plan = {11: 1e-5}
cfg.max_epoch = 30
# cfg.actions_weights = [[1., 1., 2., 3., 1., 2., 2., 0.2, 1.]]
cfg.actions_weights = [1., 1., 2., 3., 1., 2., 2., 0.2, 1.]

# cfg.exp_note = 'Dynamic Volleyball_stage2_res18_litedim128_reproduce_1'
cfg.exp_note='Higcin_volleyball'

print("===> Generate wandb system")
wandb.login()
wandb.init(project=f"DIN-Group-Activity-Recognition-Benchmark-Volley",
            name=f'{cfg.exp_note}_stage2', 
            config=cfg,
            settings=wandb.Settings(start_method='fork'),
            )

train_net(cfg)