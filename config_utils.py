import sys
import os
import pickle

def update_config_all(cfg):
    if 'use_recon_loss' in dir(cfg):
        pass
    else:
        cfg.use_recon_loss = False

    if 'use_ind_feat' in dir(cfg):
        pass
    else:
        cfg.use_ind_feat = 'loc_and_app'

    if 'use_ind_feat_crop' in dir(cfg):
        pass
    else:
        cfg.use_ind_feat_crop = 'roi_multi'

    if 'person_size' in dir(cfg):
        pass
    else:
        cfg.person_size = 224, 224

    if 'trans_head_num' in dir(cfg):
        pass
    else:
        cfg.trans_head_num = 1

    if 'trans_layer_num' in dir(cfg):
        pass
    else:
        cfg.trans_layer_num = 1

    if 'final_head_mid_num' in dir(cfg):
        pass
    else:
        cfg.final_head_mid_num = 2

    if 'use_res_connect' in dir(cfg):
        pass
    else:
        cfg.use_res_connect = False

    if 'people_pool_type' in dir(cfg):
        pass
    else:
        cfg.people_pool_type = 'max'

    if 'use_gen_iar' in dir(cfg):
        pass
    else:
        cfg.use_gen_iar = False

    if 'use_trans' in dir(cfg):
        pass
    else:
        cfg.use_trans = True

    if 'use_pos_cond' in dir(cfg):
        pass
    else:
        cfg.use_pos_cond = True

    if 'use_act_loss' in dir(cfg):
        pass
    else:
        if cfg.use_recon_loss:
            cfg.use_act_loss = False
        else:
            cfg.use_act_loss = True

    if 'use_tmp_cond' in dir(cfg):
        pass
    else:
        cfg.use_tmp_cond = False

    if 'use_loc_feat_prev' in dir(cfg):
        pass
    else:
        cfg.use_loc_feat_prev = False
    
    if 'use_pose_loss' in dir(cfg):
        pass
    else:
        cfg.use_pose_loss = False

    if 'use_recon_diff_loss' in dir(cfg):
        pass
    else:
        cfg.use_recon_diff_loss = False

    if 'use_same_enc_dual_path' in dir(cfg):
        pass
    else:
        cfg.use_same_enc_dual_path = False

    return cfg


def check_config(cfg):
    print(cfg)