'''
    Evaluate the trained model with various metrics and visualization.
'''

import torch
import torch.optim as optim

import time
import random
import os
os.environ['OMP_NUM_THREADS'] = '1'
os.environ['OPENBLAS_NUM_THREADS'] = '1'
import sys
import json
from tqdm import tqdm
from collections import OrderedDict
from sklearn.metrics import average_precision_score
from sklearn.metrics import label_ranking_average_precision_score
from sklearn.manifold import TSNE
from sklearn.metrics import confusion_matrix
from sklearn.neighbors import KNeighborsClassifier
from sklearn.cluster import KMeans
import warnings 
warnings.filterwarnings("ignore")
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from munkres import Munkres

from config import *
from volleyball import *
from collective import *
from dataset import *
from infer_model import *
from infer_model_prev import *
from infer_model_original import *
from base_model import *
from utils import *

def save_clustering_results(mode, cfg, info_test, kmeans_labels):
    test_gr = info_test['activities_in_all'].cpu().numpy().reshape(-1, 1)
    df_cluster_array = np.concatenate([kmeans_labels, test_gr], axis=1)
    df_cluster_results = pd.DataFrame(df_cluster_array, info_test['video_id_all'], ['cluster_id', 'GA'])
    save_excel_file_path = os.path.join(cfg.result_path, f'test_kmeans_{mode}.xlsx')
    df_cluster_results.to_excel(save_excel_file_path, sheet_name='all')

def clustering_net(cfg):
    """
    evaluating gcn net
    """
    os.environ['CUDA_VISIBLE_DEVICES']=cfg.device_list

    # Show config parameters
    cfg.init_config()
    show_config(cfg)
    
    # Reading dataset
    _, validation_set=return_dataset(cfg)
    
    params = {
        'batch_size': 1,
        'shuffle': False,
        'num_workers': 4, # 4,
    }
    params['batch_size']=cfg.test_batch_size

    # Set data position
    if cfg.use_gpu and torch.cuda.is_available():
        device = torch.device('cuda')
    else:
        device = torch.device('cpu')
    cfg.device = device
    
    # Build model and Load trained model
    basenet_list={'group_relation_volleyball':Basenet_volleyball,
                  'group_relation_collective':Basenet_collective,
                  }
    gcnnet_list={
                 'group_activity_volleyball':GroupActivity_volleyball,
                 'group_relation_volleyball':GroupRelation_volleyball,
                 'group_relation_ident_volleyball':GroupRelationIdentity_volleyball,
                 'group_relation_ae_volleyball':GroupRelationAutoEncoder_volleyball,
                 'group_relation_hrn_volleyball':GroupRelationHRN_volleyball,
                 'group_activity_collective':GroupActivity_volleyball,
                 'group_relation_collective':GroupRelation_volleyball,
                 'group_relation_ident_collective':GroupRelationIdentity_volleyball,
                 'group_relation_ae_collective':GroupRelationAutoEncoder_volleyball,
                 'group_relation_hrn_collective':GroupRelationHRN_volleyball,
                 'dynamic_volleyball':Dynamic_volleyball,
                 'dynamic_collective':Dynamic_volleyball,
                 'higcin_volleyball':HiGCIN_volleyball,
                 'higcin_collective':HiGCIN_volleyball,
                }

    if cfg.eval_stage == 1:
        model = basenet_list[cfg.inference_module_name](cfg)
        model.loadmodel(cfg.stage1model)
        print_log(cfg.log_path, f'Loading stage{cfg.eval_stage} model: ' + cfg.stage1model)
    elif cfg.eval_stage == 2:
        model = gcnnet_list[cfg.inference_module_name](cfg)

        if 'group_relation_ident' in cfg.inference_module_name:
            if cfg.load_backbone_stage2:
                model.loadmodel(cfg.stage1_model_path)
            else:
                pass
        elif 'dynamic_collective' in cfg.inference_module_name:
            state_dict = torch.load(cfg.stage2model)['state_dict']
            model.load_state_dict(state_dict)
            print_log(cfg.log_path, f'Loading stage{cfg.eval_stage} model: ' + cfg.stage2model)
        elif 'higcin_collective' in cfg.inference_module_name:
            state_dict = torch.load(cfg.stage2model)['state_dict']
            model.load_state_dict(state_dict)
            print_log(cfg.log_path, f'Loading stage{cfg.eval_stage} model: ' + cfg.stage2model)
        else:
            state_dict = torch.load(cfg.stage2model)['state_dict']
            new_state_dict=OrderedDict()
            for k, v in state_dict.items():
                name = k[7:] 
                new_state_dict[name] = v
            model.load_state_dict(new_state_dict)
            print_log(cfg.log_path, f'Loading stage{cfg.eval_stage} model: ' + cfg.stage2model)
    else:
        assert False, 'cfg.eval_stage should be 1 or 2'

    if cfg.use_multi_gpu:
        model=nn.DataParallel(model)
    model=model.to(device=device)

    test_list={'volleyball':test_volleyball, 'collective':test_collective}
    test=test_list[cfg.dataset_name]

    if cfg.use_debug:
        validation_set.frames = validation_set.frames[:cfg.debug_sample]
    cfg.test_data_size = len(validation_set)
    validation_loader=data.DataLoader(validation_set,**params)
    
    info_test=test(validation_loader, model, device, 0, cfg)
    clustering_xai(validation_loader, model, cfg, info_test)

def clustering_xai(data_loader, model, cfg, info_test):
    get_activities_list={'volleyball':volley_get_activities, 'collective':collective_get_activities}
    get_activities = get_activities_list[cfg.dataset_name]
    test_num = info_test['actions_in_all'].shape[0]
    num_people = info_test['actions_in_all'].shape[1]
    cfg.num_people = num_people

    print("===> Calculate distance between train and test features")
    feat_dic = {'people':'individual_features', 'scene':'group_features'}

    print("===> Evaluate clustering")
    mode = 'scene'
    group_feat_test = info_test[feat_dic[mode]]
    kmeans = KMeans(n_clusters=cfg.n_clusters, random_state=777)
    kmeans.fit(group_feat_test.cpu().numpy())
    cluster_labels = kmeans.labels_
    save_clustering_results(mode, cfg, info_test, cluster_labels.reshape(-1, 1))
    if cfg.use_debug:
        cluster_labels = info_test['activities_in_all'].cpu().numpy().astype(np.int32)
    print(collections.Counter(cluster_labels.reshape(-1)))

    print("===> Calculate cluster features")
    device = group_feat_test.device
    cluster_feat_center_tensor = torch.zeros((cfg.n_clusters, info_test[feat_dic[mode]].shape[1]), requires_grad=True).to(device=device)
    for cluster_idx in range(cfg.n_clusters):
        cluster_feat = info_test[feat_dic[mode]][cluster_labels==cluster_idx]
        if cfg.cluster_center_type == 'mean':
            cluster_feat_center = cluster_feat.mean(dim=0)
        elif cfg.cluster_center_type == 'median':
            cluster_feat_center = cluster_feat.median(dim=0).values
        else:
            assert False, 'cfg.cluster_center_type should be mean or median'
        cluster_feat_center_tensor[cluster_idx] = cluster_feat_center

    print("===> Estimate mask")
    model.module.set_explainable_type(cfg.mask_act_type)
    xai_mask_loss_all = torch.zeros((test_num, num_people)).to(device=device)
    xai_idx = 0
    for batch_data_test in tqdm(data_loader):
        for key in range(len(batch_data_test)):
            if torch.is_tensor(batch_data_test[key]):
                batch_data_test[key] = batch_data_test[key].to(device=device)

        input_data = {}
        input_data['images_in'] = batch_data_test[0]
        input_data['boxes_in'] = batch_data_test[1]
        input_data['images_person_in'] = batch_data_test[4]
        batch_size=batch_data_test[0].shape[0]
        num_frames=batch_data_test[0].shape[1]

        # obtain original features
        xai_feat_cluster_grp = cluster_feat_center_tensor[cluster_labels[xai_idx:xai_idx+batch_size]]

        if 'perturbation' in cfg.mask_act_type:
            with torch.no_grad():
                xai_mask_importance = torch.zeros(batch_size, num_people).to(device=device)
                for person_idx in range(num_people):
                    xai_person_mask = torch.ones(batch_size, num_people).to(device=device)
                    xai_person_mask[:, person_idx] = 0
                    input_data['xai_person_mask'] = xai_person_mask
                    ret = model(input_data)
                    xai_masked_feat_grp = ret['group_feat']
                    loss_feat = (xai_masked_feat_grp-xai_feat_cluster_grp).pow(2).mean(dim=1)
                    xai_mask_importance[:, person_idx] = loss_feat
        elif 'backprop' in cfg.mask_act_type:
            # freeze all parameters
            for param in model.parameters():
                # param.requires_grad = True
                param.requires_grad = False

            # # init and set xai person mask
            xai_mask_importance = torch.rand(batch_size, num_people, dtype=torch.float32, requires_grad=True, device=device)
            model.module.update_explainable_mask(xai_mask_importance)
            
            # set xai person mask for optimizer
            optimizer=optim.Adam([xai_mask_importance], lr=1e-2)

            # optimize xai person mask
            backprop_all = 200
            for backprop_iter in tqdm(range(backprop_all)):

                # obtain features and masks
                ret = model(input_data)
                xai_masked_feat_grp = ret['group_feat']
                xai_mask_importance_update = ret['xai_person_mask']

                # calculate loss functions
                loss_feat = F.mse_loss(xai_masked_feat_grp, xai_feat_cluster_grp)
                loss_mask = xai_mask_importance_update.mean() * 1e-1
                if 'energy' in cfg.mask_act_type:
                    loss = loss_feat + loss_mask
                else: 
                    loss = loss_feat

                # update masks
                optimizer.zero_grad()
                loss.backward(retain_graph=True)
                optimizer.step()
                model.module.update_explainable_mask(xai_mask_importance_update)

                if cfg.use_debug and backprop_iter % 40 == 0:
                    print(f'loss_feat: {loss_feat.item():.4f}, loss_mask: {loss_mask.item():.4f}')
                    print(xai_mask_importance_update[0])

            # save masks for all data
            xai_mask_importance = xai_mask_importance_update
        elif 'random' in cfg.mask_act_type:
            xai_mask_importance = torch.rand(batch_size, num_people, dtype=torch.float32, requires_grad=True, device=device)

        # save person mask for all data
        xai_mask_loss_all[xai_idx:xai_idx+batch_size] = xai_mask_importance

        # increament xai idx
        xai_idx += batch_size

        if cfg.use_debug:
            if (xai_idx > (cfg.debug_sample-1)):
                break

    print("===> Save optimized mask")
    test_gr_np = info_test['activities_in_all'].cpu().numpy().reshape(-1, 1)
    cluster_labels_np = cluster_labels.reshape(-1, 1)
    xai_mask_all_np = xai_mask_loss_all.detach().cpu().numpy()
    test_actions_np = info_test['actions_in_all'].cpu().numpy()
    save_mask_array = np.concatenate([test_gr_np, cluster_labels_np, test_actions_np, xai_mask_all_np], axis=1)
    save_mask_array_label = ['GA'] + ['CL'] + [f'P_act:{i}' for i in range(num_people)] + [f'P:{i}' for i in range(num_people)]
    # np.save(os.path.join(cfg.result_path, f'test_clusuer_{cfg.n_clusters}_{mode}_{cfg.cluster_center_type}_{cfg.mask_act_type}.npy'), xai_mask_all_np)

    df_mask = pd.DataFrame(save_mask_array, info_test['video_id_all'], save_mask_array_label)
    if cfg.use_debug:
        save_excel_file_path = os.path.join(cfg.result_path, f'test_cluster_{cfg.n_clusters}_{mode}_{cfg.cluster_center_type}_{cfg.mask_act_type}_debug.xlsx')
    else:
        save_excel_file_path = os.path.join(cfg.result_path, f'test_cluster_{cfg.n_clusters}_{mode}_{cfg.cluster_center_type}_{cfg.mask_act_type}.xlsx')
    df_mask.to_excel(save_excel_file_path, sheet_name='all')

def test_volleyball(data_loader, model, device, epoch, cfg):
    model.eval()
    train_with_action = False
    actions_meter=AverageMeter()
    activities_meter=AverageMeter()
    loss_meter=AverageMeter()
    activities_conf = ConfusionMeter(cfg.num_activities)
    actions_conf = ConfusionMeter(cfg.num_actions)
    epoch_timer=Timer()

    batch_idx = 0
    actions_in_all = torch.zeros(cfg.test_data_size, cfg.num_boxes).to(device=device)
    actions_labels_all = torch.zeros(cfg.test_data_size, cfg.num_boxes).to(device=device)
    activities_in_all = torch.zeros(cfg.test_data_size).to(device=device)
    locations_in_all = torch.zeros(cfg.test_data_size, cfg.num_boxes, 2).to(device=device)
    video_id_all = ['0' for i in range(cfg.test_data_size)]

    if cfg.eval_stage == 1:
        individual_features = torch.zeros(cfg.test_data_size, cfg.num_features_boxes*cfg.num_boxes).to(device=device)
        group_features = torch.zeros(cfg.test_data_size, cfg.num_features_boxes).to(device=device)
    elif cfg.eval_stage == 2:
        if 'group_relation_ident' in cfg.inference_module_name:
            if cfg.use_ind_feat_crop == 'roi_multi':
                ind_feat_dim = cfg.emb_features*cfg.crop_size[0]*cfg.crop_size[0]
            else:
                ind_feat_dim = 4096
            individual_features = torch.zeros(cfg.test_data_size, ind_feat_dim*cfg.num_boxes).to(device=device)
            group_features = torch.zeros(cfg.test_data_size, ind_feat_dim).to(device=device)
        elif 'group_relation_ae' in cfg.inference_module_name:
            if cfg.use_ind_feat_crop == 'roi_multi':
                ind_feat_dim = 1024
            else:
                ind_feat_dim = 128
            individual_features = torch.zeros(cfg.test_data_size, ind_feat_dim*cfg.num_boxes).to(device=device)
            group_features = torch.zeros(cfg.test_data_size, ind_feat_dim).to(device=device)
        elif 'group_relation_hrn' in cfg.inference_module_name:
            if cfg.use_ind_feat_crop == 'roi_multi':
                ind_feat_dim = 512
            else:
                ind_feat_dim = 128
            individual_features = torch.zeros(cfg.test_data_size, ind_feat_dim*cfg.num_boxes).to(device=device)
            group_features = torch.zeros(cfg.test_data_size, ind_feat_dim).to(device=device)
        elif 'dynamic' in cfg.inference_module_name:
            individual_features = torch.zeros(cfg.test_data_size, 128*cfg.num_boxes).to(device=device)
            group_features = torch.zeros(cfg.test_data_size, 128).to(device=device)
        elif 'higcin' in cfg.inference_module_name:
            individual_features = torch.zeros(cfg.test_data_size, 512*cfg.num_boxes).to(device=device)
            group_features = torch.zeros(cfg.test_data_size, 512).to(device=device)
        else:
            individual_features = torch.zeros(cfg.test_data_size, cfg.num_features_boxes*2*cfg.num_boxes).to(device=device)
            group_features = torch.zeros(cfg.test_data_size, cfg.num_features_boxes*2).to(device=device)

    with torch.no_grad():
        for batch_data_test in tqdm(data_loader):
            # prepare batch data
            for key in range(len(batch_data_test)):
                if torch.is_tensor(batch_data_test[key]):
                    batch_data_test[key] = batch_data_test[key].to(device=device)

            batch_size=batch_data_test[0].shape[0]
            num_frames=batch_data_test[0].shape[1]

            locations_in=batch_data_test[1].reshape((batch_size,num_frames,cfg.num_boxes, 4))
            actions_in=batch_data_test[2].reshape((batch_size,num_frames,cfg.num_boxes))
            activities_in=batch_data_test[3].reshape((batch_size,num_frames))
            
            # forward
            input_data = {}
            input_data['images_in'] = batch_data_test[0]
            input_data['boxes_in'] = batch_data_test[1]
            input_data['images_person_in'] = batch_data_test[4]
            ret= model(input_data)

            # Predict actions
            actions_in=actions_in[:,0,:].reshape((batch_size*cfg.num_boxes,))
            activities_in=activities_in[:,0].reshape((batch_size,))

            # Predict activities
            loss_list = []
            if 'activities' in list(ret.keys()):
                activities_scores = ret['activities']
                activities_loss = F.cross_entropy(activities_scores,activities_in)
                loss_list.append(activities_loss)
                activities_labels = torch.argmax(activities_scores,dim=1)

                activities_correct = torch.sum(torch.eq(activities_labels.int(),activities_in.int()).float())
                activities_accuracy = activities_correct.item() / activities_scores.shape[0]
                activities_meter.update(activities_accuracy, activities_scores.shape[0])
                activities_conf.add(activities_labels, activities_in)

            if 'actions' in list(ret.keys()):
                actions_scores = ret['actions']
                actions_labels=torch.argmax(actions_scores,dim=1)
                actions_correct = torch.sum(torch.eq(actions_labels.int(), actions_in.int()).float())
                actions_accuracy = actions_correct.item() / actions_scores.shape[0]
                actions_meter.update(actions_accuracy, actions_scores.shape[0])
                actions_conf.add(actions_labels, actions_in)
            
            # save features
            individual_features[batch_idx:batch_idx+batch_size] = ret['individual_feat']
            group_features[batch_idx:batch_idx+batch_size] = ret['group_feat']
            actions_in_all[batch_idx:batch_idx+batch_size] = actions_in.reshape((batch_size,-1))
            activities_in_all[batch_idx:batch_idx+batch_size] = activities_in
            locations_in_all_x = (locations_in[:, num_frames//2, :, 0]+locations_in[:, num_frames//2, :, 2])/2
            locations_in_all_y = (locations_in[:, num_frames//2, :, 1]+locations_in[:, num_frames//2, :, 3])/2
            locations_in_all[batch_idx:batch_idx+batch_size, :, 0] = locations_in_all_x
            locations_in_all[batch_idx:batch_idx+batch_size, :, 1] = locations_in_all_y
            video_id_all[batch_idx:batch_idx+batch_size] = batch_data_test[5]

            if cfg.use_recon_loss:
                pass
            elif cfg.inference_module_name == 'group_relation_ident_volleyball':
                pass
            else:
                actions_labels_all[batch_idx:batch_idx+batch_size] = actions_labels.reshape((batch_size,-1))

            batch_idx += batch_size
            if cfg.use_debug:
                if batch_idx > cfg.debug_sample:
                    break

    test_info={
        'time':epoch_timer.timeit(),
        'activities_acc':activities_meter.avg*100,
        'activities_conf': activities_conf.value(),
        'activities_MPCA': MPCA(activities_conf.value()),
        'actions_acc':actions_meter.avg*100,
        'actions_conf':actions_conf.value(),
        'actions_MPCA':MPCA(actions_conf.value()),
        'individual_features':individual_features,
        'group_features':group_features,
        'actions_in_all':actions_in_all,
        'actions_labels_all':actions_labels_all,
        'activities_in_all':activities_in_all,
        'locations_in_all':locations_in_all,
        'video_id_all':video_id_all,
    }
    
    return test_info

def test_collective(data_loader, model, device, epoch, cfg):
    model.eval()
    
    actions_meter=AverageMeter()
    activities_meter=AverageMeter()
    loss_meter=AverageMeter()
    
    epoch_timer=Timer()
    activities_conf = ConfusionMeter(cfg.num_activities)
    actions_conf = ConfusionMeter(cfg.num_actions)

    batch_idx = 0
    with torch.no_grad():
        actions_in_all = torch.zeros(cfg.test_data_size, cfg.num_boxes).to(device=device)
        actions_labels_all = torch.zeros(cfg.test_data_size, cfg.num_boxes).to(device=device)
        activities_in_all = torch.zeros(cfg.test_data_size).to(device=device)
        locations_in_all = torch.zeros(cfg.test_data_size, cfg.num_boxes, 2).to(device=device)
        video_id_all = ['0' for i in range(cfg.test_data_size)]

        if cfg.eval_stage == 1:
            individual_features = torch.zeros(cfg.test_data_size, cfg.num_features_boxes*cfg.num_boxes).to(device=device)
            group_features = torch.zeros(cfg.test_data_size, cfg.num_features_boxes).to(device=device)
        elif cfg.eval_stage == 2:
            if 'group_relation_ident' in cfg.inference_module_name:
                if cfg.use_ind_feat_crop == 'roi_multi':
                    ind_feat_dim = 1056*cfg.crop_size[0]*cfg.crop_size[0]
                else:
                    ind_feat_dim = 4096
                individual_features = torch.zeros(cfg.test_data_size, ind_feat_dim*cfg.num_boxes).to(device=device)
                group_features = torch.zeros(cfg.test_data_size, ind_feat_dim).to(device=device)
            elif 'group_relation_ae' in cfg.inference_module_name:
                if cfg.use_ind_feat_crop == 'roi_multi':
                    ind_feat_dim = 1024
                else:
                    ind_feat_dim = 128
                individual_features = torch.zeros(cfg.test_data_size, ind_feat_dim*cfg.num_boxes).to(device=device)
                group_features = torch.zeros(cfg.test_data_size, ind_feat_dim).to(device=device)
            elif 'group_relation_hrn' in cfg.inference_module_name:
                if cfg.use_ind_feat_crop == 'roi_multi':
                    ind_feat_dim = 512
                else:
                    ind_feat_dim = 128
                individual_features = torch.zeros(cfg.test_data_size, ind_feat_dim*cfg.num_boxes).to(device=device)
                group_features = torch.zeros(cfg.test_data_size, ind_feat_dim).to(device=device)
            elif 'dynamic' in cfg.inference_module_name:
                individual_features = torch.zeros(cfg.test_data_size, cfg.num_features_boxes*cfg.num_boxes).to(device=device)
                group_features = torch.zeros(cfg.test_data_size, cfg.num_features_boxes).to(device=device)
            elif 'higcin' in cfg.inference_module_name:
                individual_features = torch.zeros(cfg.test_data_size, 1056*cfg.num_boxes).to(device=device)
                group_features = torch.zeros(cfg.test_data_size, 1056).to(device=device)
            else:
                individual_features = torch.zeros(cfg.test_data_size, cfg.num_features_boxes*2*cfg.num_boxes).to(device=device)
                group_features = torch.zeros(cfg.test_data_size, cfg.num_features_boxes*2).to(device=device)

        for batch_data_test in tqdm(data_loader):
            # prepare batch data
            for key in range(len(batch_data_test)):
                if torch.is_tensor(batch_data_test[key]):
                    batch_data_test[key] = batch_data_test[key].to(device=device)

            batch_size=batch_data_test[0].shape[0]
            num_frames=batch_data_test[0].shape[1]
            
            locations_in=batch_data_test[1].reshape((batch_size,num_frames,cfg.num_boxes, 4))
            actions_in=batch_data_test[2].reshape((batch_size,num_frames,cfg.num_boxes))
            activities_in=batch_data_test[3].reshape((batch_size,num_frames))
            bboxes_num=batch_data_test[5].reshape(batch_size,num_frames)

            # forward
            if cfg.training_stage==1:
                # ret = model((batch_data[0], batch_data[1], batch_data[4], batch_data[5]))
                input_data = {}
                input_data['images_in'] = batch_data_test[0]
                input_data['boxes_in'] = batch_data_test[1]
                input_data['images_person_in'] = batch_data_test[4]
                input_data['bboxes_num'] = batch_data_test[5]
                ret= model(input_data)
            elif cfg.training_stage==2:
                # ret = model((batch_data[0], batch_data[1], batch_data[4]))
                input_data = {}
                input_data['images_in'] = batch_data_test[0]
                input_data['boxes_in'] = batch_data_test[1]
                input_data['images_person_in'] = batch_data_test[4]
                ret= model(input_data)

            actions_in_nopad=[]
            if cfg.training_stage==1:
                actions_in=actions_in.reshape((batch_size*num_frames,cfg.num_boxes,))
                bboxes_num=bboxes_num.reshape(batch_size*num_frames,)
                for bt in range(batch_size*num_frames):
                    N=bboxes_num[bt]
                    actions_in_nopad.append(actions_in[bt,:N])
            else:
                for b in range(batch_size):
                    N=bboxes_num[b][0]
                    actions_in_nopad.append(actions_in[b][0][:N])
            actions_in=torch.cat(actions_in_nopad,dim=0).reshape(-1,)  #ALL_N,
            
            if cfg.training_stage==1:
                activities_in=activities_in.reshape(-1,)
            else:
                activities_in=activities_in[:,0].reshape(batch_size,)

            if 'actions' in list(ret.keys()):
                actions_scores=ret['actions'].reshape(batch_size, cfg.num_boxes, -1)
                actions_scores_nopad=[]
                for b in range(batch_size):
                    N = bboxes_num[b][0]
                    actions_scores_nopad.append(actions_scores[b][:N])
                actions_scores=torch.cat(actions_scores_nopad,dim=0).reshape(-1, cfg.num_actions)
                actions_loss=F.cross_entropy(actions_scores,actions_in)
                actions_labels=torch.argmax(actions_scores,dim=1)  #ALL_N,
                actions_correct=torch.sum(torch.eq(actions_labels.int(),actions_in.int()).float())
                actions_accuracy = actions_correct.item() / actions_scores.shape[0]
                actions_meter.update(actions_accuracy, actions_scores.shape[0])

            # Total loss
            # total_loss=actions_loss
            # loss_meter.update(total_loss.item(), batch_size)

            # save features
            individual_features[batch_idx:batch_idx+batch_size] = ret['individual_feat']
            group_features[batch_idx:batch_idx+batch_size] = ret['group_feat']
            actions_in_all[batch_idx:batch_idx+batch_size] = batch_data_test[2].reshape((batch_size,num_frames,cfg.num_boxes))[:,0,:].reshape((batch_size*cfg.num_boxes,))
            activities_in_all[batch_idx:batch_idx+batch_size] = batch_data_test[3].reshape((batch_size,num_frames))[:,0].reshape((batch_size,))
            locations_in_all_x = (locations_in[:, num_frames//2, :, 0]+locations_in[:, num_frames//2, :, 2])/2
            locations_in_all_y = (locations_in[:, num_frames//2, :, 1]+locations_in[:, num_frames//2, :, 3])/2
            locations_in_all[batch_idx:batch_idx+batch_size, :, 0] = locations_in_all_x
            locations_in_all[batch_idx:batch_idx+batch_size, :, 1] = locations_in_all_y
            video_id_all[batch_idx] = batch_data_test[6]

            batch_idx += batch_size
            if cfg.use_debug:
                if batch_idx > cfg.debug_sample:
                    break

    test_info={
        'time':epoch_timer.timeit(),
        'activities_acc':activities_meter.avg*100,
        'activities_conf': activities_conf.value(),
        'activities_MPCA': MPCA(activities_conf.value()),
        'actions_acc':actions_meter.avg*100,
        'actions_conf':actions_conf.value(),
        'actions_MPCA':MPCA(actions_conf.value()),
        'individual_features':individual_features,
        'group_features':group_features,
        'actions_in_all':actions_in_all,
        'actions_labels_all':actions_labels_all,
        'activities_in_all':activities_in_all,
        'locations_in_all':locations_in_all,
        'video_id_all':video_id_all,
    }

    return test_info