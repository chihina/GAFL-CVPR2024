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

from config import *
from volleyball import *
from collective import *
from dataset import *
from infer_model import *
from infer_model_prev import *
from infer_model_original import *
from base_model import *
from utils import *

def set_bn_eval(m):
    classname = m.__class__.__name__
    if classname.find('BatchNorm') != -1:
        m.eval()
            
def adjust_lr(optimizer, new_lr):
    print('change learning rate:',new_lr)
    for param_group in optimizer.param_groups:
        param_group['lr'] = new_lr

def evaluate_iar_metrics(info_train, info_test, metrics_dic, cfg):
    iar_metrics_dic = {}
    get_actions_list={'volleyball':volley_get_actions, 'collective':collective_get_actions}
    get_actions = get_actions_list[cfg.dataset_name]

    print("=========> Calculate IAR accuracy")
    iar_acc_array = info_test['actions_in_all'] == info_test['actions_labels_all']
    iar_mca = torch.mean(iar_acc_array.float()).to('cpu').detach().item()
    iar_acc_mean_mpca_list = []
    for iar_idx, iar_name in enumerate(get_actions()):
        iar_acc_array_class = iar_acc_array[info_test['actions_in_all']==iar_idx]
        iar_acc_mean_class = torch.mean(iar_acc_array_class.float()).to('cpu').detach().item()
        metrics_dic[f'IAR accuracy {iar_name}'] = iar_acc_mean_class
        print(f'IAR accuracy {iar_name}', iar_acc_mean_class)
        iar_acc_mean_mpca_list.append(iar_acc_mean_class)
    iar_mpca = sum(iar_acc_mean_mpca_list)/len(iar_acc_mean_mpca_list)
    metrics_dic['IAR MCA'] = iar_mca
    metrics_dic['IAR MPCA'] = iar_mpca
    print('IAR MCA', iar_mca)
    print('IAR MPCA', iar_mpca)

    if cfg.dataset_name != 'volleyball':
        return

    if not cfg.use_debug:
        print("=========> Calculate confusion matrix")
        cm_iar = confusion_matrix(info_test['actions_in_all'].view(-1).cpu().numpy(), info_test['actions_labels_all'].view(-1).cpu().numpy(), normalize='true')
        cm_iar = pd.DataFrame(cm_iar, columns=get_actions(), index=get_actions())
        plt.figure(figsize=(10, 10))
        plt.tight_layout()
        s = sns.heatmap(cm_iar, annot=True, cmap='Blues', fmt=".2f", annot_kws={"size": 25}, vmin=0, vmax=1)
        s.set(xlabel='Pr', ylabel='GT')
        save_cm_path = os.path.join(cfg.result_path, 'confusion_matrix_iar.png')
        plt.savefig(save_cm_path)

def plot_action_iou_histgram(mode, cfg, action_iou_type, action_iou_array):
    print(f"===> Plot histgram {mode} {action_iou_type}")
    action_iou_array_np = action_iou_array.cpu().numpy()
    plt.figure()
    plt.hist(action_iou_array_np)
    save_action_iou_path = os.path.join(cfg.result_path, f'hist_{action_iou_type}_{mode}.png')
    plt.savefig(save_action_iou_path)
    save_action_iou_np_path = os.path.join(cfg.result_path, f'hist_{action_iou_type}_{mode}')
    np.save(save_action_iou_np_path, action_iou_array_np)

# def calc_action_iou_labels(actions_cnt_cat, action_iou_thresh, actions_weights):
#     action_cnt_cat_max, _ = actions_cnt_cat.max(dim=3)
#     action_cnt_cat_min, _ = actions_cnt_cat.min(dim=3)
#     action_cnt_cat_max_weighted = action_cnt_cat_max * actions_weights
#     action_cnt_cat_min_weighted = action_cnt_cat_min * actions_weights
#     action_iou_weighted = action_cnt_cat_min_weighted.sum(dim=2) / action_cnt_cat_max_weighted.sum(dim=2)
#     action_iou_weighted_labels = action_iou_weighted > action_iou_thresh

#     return action_iou_weighted, action_iou_weighted_labels

def calc_action_iou_labels(actions_cnt_cat, action_iou_thresh, mode='jaccard'):
    print(f'action iou {mode}')
    test_num, train_val_num, actions_num, _ = actions_cnt_cat.shape
    actions_cnt_cat_union, _ = actions_cnt_cat.max(dim=3)
    actions_cnt_cat_common, _ = actions_cnt_cat.min(dim=3)
    actions_cnt_cat_sum_train_val = actions_cnt_cat[:, :, :, 0].sum(dim=2)
    actions_cnt_cat_sum_test = actions_cnt_cat[:, :, :, 1].sum(dim=2)
    actions_cnt_cat_sum_common = torch.min(actions_cnt_cat_sum_train_val, actions_cnt_cat_sum_test)

    if mode == 'jaccard':
        action_iou = actions_cnt_cat_common.sum(dim=2) / actions_cnt_cat_union.sum(dim=2)
    elif mode == 'dice':
        action_iou = (2*actions_cnt_cat_common.sum(dim=2)) / (actions_cnt_cat_sum_train_val+actions_cnt_cat_sum_test)
    elif mode == 'simpson':
        action_iou = actions_cnt_cat_common.sum(dim=2) / actions_cnt_cat_sum_common
    elif mode == 'tfidf':
        actions_cnt_cat_tf_train_val = actions_cnt_cat[:, :, :, 0] / actions_cnt_cat_sum_train_val.view(test_num, train_val_num, 1)
        actions_cnt_cat_tf_train_val = actions_cnt_cat_tf_train_val.view(test_num, train_val_num, actions_num, 1)
        actions_cnt_cat_tf_test = actions_cnt_cat[:, :, :, 1] / actions_cnt_cat_sum_test.view(test_num, train_val_num, 1)
        actions_cnt_cat_tf_test = actions_cnt_cat_tf_test.view(test_num, train_val_num, actions_num, 1)
        actions_cnt_cat_tf = torch.cat([actions_cnt_cat_tf_train_val, actions_cnt_cat_tf_test], dim=-1)
        actions_cnt_train_val_freq = actions_cnt_cat[0, :, :, 0]
        actions_cnt_train_val_freq_flag = actions_cnt_train_val_freq > 0
        actions_cnt_train_val_freq_flag_sum = actions_cnt_train_val_freq_flag.sum(dim=0)
        actions_cnt_cat_idf =  torch.log(train_val_num/(actions_cnt_train_val_freq_flag_sum+1e-10))
        actions_cnt_cat_tfidf = actions_cnt_cat_idf.view(1, 1, actions_num, 1) * actions_cnt_cat_tf
        actions_cnt_cat_tfidf = actions_cnt_cat_tfidf / (torch.norm(actions_cnt_cat_tfidf, dim=2, keepdim=True))
        action_iou = (actions_cnt_cat_tfidf[:, :, :, 0] * actions_cnt_cat_tfidf[:, :, :, 1]).sum(dim=2)

    action_iou_labels = action_iou > action_iou_thresh

    return action_iou, action_iou_labels

def calc_hit_at_n(mode, cfg, action_iou_type, action_iou_labels_all, match_score, test_num, metrics_dic):
    print(f"======> Calculate hit@n ({action_iou_type}) ({mode})")
    match_score_argsort = torch.argsort(match_score, dim=1)
    hit_num = 5
    action_iou_flag = torch.zeros(test_num, hit_num, device=match_score.device)
    for hit_idx in range(hit_num):
        action_iou_labels_hit = action_iou_labels_all[torch.arange(test_num, device=match_score.device), match_score_argsort[:, hit_idx]]
        action_iou_flag[:, hit_idx] = action_iou_labels_hit
    for hit_idx in range(hit_num):
        action_iou_flag_hit = action_iou_flag[:, :(hit_idx+1)]
        action_iou_flag_hit_sum = action_iou_flag_hit.sum(dim=1) > 0
        action_iou_hit_score = torch.mean(action_iou_flag_hit_sum.float()).to('cpu').detach().item()
        metrics_dic[f'GAR accuracy hit@{hit_idx+1} ({action_iou_type}) ({mode})'] = action_iou_hit_score
        print(f'GAR accuracy hit@{hit_idx+1} ({action_iou_type}) ({mode})', action_iou_hit_score)

def calc_map(mode, cfg, action_iou_type, action_iou_labels_all, match_score, metrics_dic):
    print(f"======> Calculate mAP ({action_iou_type}) ({mode})")
    map_pred_conf = -1 * match_score.cpu()
    map_action_iou = average_precision_score(action_iou_labels_all.cpu().numpy(), map_pred_conf.numpy())
    metrics_dic[f'mAP ({action_iou_type}) ({mode})'] = map_action_iou
    print(f'mAP ({action_iou_type}) ({mode})', map_action_iou)
    map_action_iou_ranking = label_ranking_average_precision_score(action_iou_labels_all.cpu().numpy(), map_pred_conf.numpy())
    metrics_dic[f'mAP rank ({action_iou_type}) ({mode})'] = map_action_iou_ranking
    print(f'mAP rank ({action_iou_type}) ({mode})', map_action_iou_ranking)

def visualize_features(mode, cfg, info_test, test_tsne, use_legend, alpha = 0.7):
    get_activities_list={'volleyball':volley_get_activities, 'collective':collective_get_activities}
    get_activities = get_activities_list[cfg.dataset_name]
    plt.figure()
    for gar_idx, gar_name in enumerate(get_activities()):
        plot_feat = test_tsne[info_test['activities_in_all'].cpu().numpy()==gar_idx, :]
        plt.scatter(plot_feat[:, 0], plot_feat[:, 1], label=gar_name, alpha=alpha)
    
    if use_legend:
        plt.legend()
        plt.savefig(os.path.join(cfg.result_path, f'tsne_{mode}.png'))
    else:
        plt.savefig(os.path.join(cfg.result_path, f'tsne_{mode}_wo_legend.png'))

def visualize_features_wo_ga(mode, cfg, info_test, test_tsne, alpha = 0.7):
    get_activities_list={'volleyball':volley_get_activities, 'collective':collective_get_activities}
    get_activities = get_activities_list[cfg.dataset_name]
    plt.figure()
    plt.scatter(test_tsne[:, 0], test_tsne[:, 1], alpha=alpha, color='k')
    plt.savefig(os.path.join(cfg.result_path, f'tsne_{mode}_wo_ga.png'))

def save_visualized_features(mode, cfg, info_test, test_tsn):
    test_gr = info_test['activities_in_all'].cpu().numpy().reshape(-1, 1)
    test_vis_array = np.concatenate([test_tsn, test_gr], axis=1)
    df_vis_results = pd.DataFrame(test_vis_array, info_test['video_id_all'], ['dim1', 'dim2', 'GA'])
    save_excel_file_path = os.path.join(cfg.result_path, f'test_tsne_{mode}.xlsx')
    df_vis_results.to_excel(save_excel_file_path, sheet_name='all')

def save_clustering_results(mode, cfg, info_test, kmeans_labels):
    test_gr = info_test['activities_in_all'].cpu().numpy().reshape(-1, 1)
    df_cluster_array = np.concatenate([kmeans_labels, test_gr], axis=1)
    df_cluster_results = pd.DataFrame(df_cluster_array, info_test['video_id_all'], ['cluster_id', 'GA'])
    save_excel_file_path = os.path.join(cfg.result_path, f'test_kmeans_{mode}.xlsx')
    df_cluster_results.to_excel(save_excel_file_path, sheet_name='all')

def save_nn_video(mode, cfg, info_test, info_train, match_idx):
    video_id_all_train = info_train['video_id_all']
    video_id_all_test = info_test['video_id_all']
    activities_in_all_train = info_train['activities_in_all'].cpu().numpy()
    activities_in_all_test = info_test['activities_in_all'].cpu().numpy()

    video_id_all_train_nn = [video_id_all_train[nn_train_idx] for nn_train_idx in match_idx]
    activities_in_all_train_nn = [activities_in_all_train[nn_train_idx] for nn_train_idx in match_idx]

    # pack data
    nn_results_dic = {}
    nn_results_dic['nn_id'] = video_id_all_train_nn
    nn_results_dic['nn_ga'] = activities_in_all_train_nn
    nn_results_dic['gt_ga'] = activities_in_all_test
    nn_results_dic['fl_ga'] = activities_in_all_test == activities_in_all_train_nn
    
    # save data as a excel file
    nn_results_array = np.array(list(nn_results_dic.values())).transpose(1, 0)
    data_name_list = list(nn_results_dic.keys())
    df_nn_results = pd.DataFrame(nn_results_array, video_id_all_test, data_name_list)
    df_nn_results.to_excel(os.path.join(cfg.result_path, f'nn_video_{mode}.xlsx'), sheet_name='all')

def evaluate_gar_metrics(mode, info_train, info_test, metrics_dic, cfg):

    get_activities_list={'volleyball':volley_get_activities, 'collective':collective_get_activities}
    get_activities = get_activities_list[cfg.dataset_name]
    train_num = info_train['actions_in_all'].shape[0]
    test_num = info_test['actions_in_all'].shape[0]
    num_people = info_train['actions_in_all'].shape[1]

    print("===> Calculate distance between train and test features")
    feat_dic = {'people':'individual_features', 'scene':'group_features'}
    match_score = torch.cdist(info_test[feat_dic[mode]], info_train[feat_dic[mode]], p=2)
    match_idx = torch.argmin(match_score, dim=-1)

    print("===> Save a nearest neighbor video id")
    save_nn_video(mode, cfg, info_test, info_train, match_idx)

    print("===> Set a threshold of action iou")
    action_iou_thresh = 0.5

    print("===> Calculate match label of action iou")
    actions_cnt_train = torch.zeros(train_num, cfg.num_actions, device=match_score.device)
    actions_cnt_test = torch.zeros(test_num, cfg.num_actions, device=match_score.device)
    for action_idx in range(cfg.num_actions):
        actions_cnt_train[:, action_idx] = (info_train['actions_in_all'] == action_idx).sum(dim=1)
        actions_cnt_test[:, action_idx] = (info_test['actions_in_all'] == action_idx).sum(dim=1)
    actions_cnt_train_expand = actions_cnt_train.view(1, train_num, -1, 1).expand(test_num, train_num, cfg.num_actions, 1)
    actions_cnt_test_expand = actions_cnt_test.view(test_num, 1, -1, 1).expand(test_num, train_num, cfg.num_actions, 1)
    actions_cnt_cat = torch.cat([actions_cnt_train_expand, actions_cnt_test_expand], dim=3)
    action_iou_jac, action_iou_labels_jac = calc_action_iou_labels(actions_cnt_cat, action_iou_thresh, 'jaccard')
    action_iou_dice, action_iou_labels_dice = calc_action_iou_labels(actions_cnt_cat, action_iou_thresh, 'dice')
    action_iou_simp, action_iou_labels_simp = calc_action_iou_labels(actions_cnt_cat, action_iou_thresh, 'simpson')
    action_iou_tfidf, action_iou_labels_tfidf = calc_action_iou_labels(actions_cnt_cat, action_iou_thresh, 'tfidf')
    plot_action_iou_histgram(mode, cfg, 'action iou jaccard', action_iou_jac[torch.arange(test_num), match_idx])
    plot_action_iou_histgram(mode, cfg, 'action iou dice', action_iou_dice[torch.arange(test_num), match_idx])
    plot_action_iou_histgram(mode, cfg, 'action iou simpson', action_iou_simp[torch.arange(test_num), match_idx])
    plot_action_iou_histgram(mode, cfg, 'action iou tfidf', action_iou_tfidf[torch.arange(test_num), match_idx])

    print("===> Calculate hit@n")
    calc_hit_at_n(mode, cfg, 'action iou jaccard', action_iou_labels_jac, match_score, test_num, metrics_dic)
    calc_hit_at_n(mode, cfg, 'action iou dice', action_iou_labels_dice, match_score, test_num, metrics_dic)
    calc_hit_at_n(mode, cfg, 'action iou simpson', action_iou_labels_simp, match_score, test_num, metrics_dic)
    calc_hit_at_n(mode, cfg, 'action iou tfidf', action_iou_labels_tfidf, match_score, test_num, metrics_dic)

    print("===> Calculate mAP")
    calc_map(mode, cfg, 'action iou jaccard', action_iou_labels_jac, match_score, metrics_dic)
    calc_map(mode, cfg, 'action iou dice', action_iou_labels_dice, match_score, metrics_dic)
    calc_map(mode, cfg, 'action iou simpson', action_iou_labels_simp, match_score, metrics_dic)
    calc_map(mode, cfg, 'action iou tfidf', action_iou_labels_tfidf, match_score, metrics_dic)

    print(f"======> Calculate GAR accuracy (group activity) ({mode})")
    gar_acc_array = info_test['activities_in_all'] == info_train['activities_in_all'][match_idx]
    gar_acc_mean = torch.mean(gar_acc_array.float()).to('cpu').detach().item()
    metrics_dic[f'GAR accuracy (group activity) ({mode})'] = gar_acc_mean
    print(f'GAR accuracy (group activity) ({mode})', gar_acc_mean)
    for gar_idx, gar_name in enumerate(get_activities()):
        gar_acc_array_class = gar_acc_array[info_test['activities_in_all']==gar_idx]
        gar_acc_mean_class = torch.mean(gar_acc_array_class.float()).to('cpu').detach().item()
        metrics_dic[f'GAR accuracy {gar_name} (group activity) ({mode})'] = gar_acc_mean_class
        print(f'GAR accuracy {gar_name} (group activity) (scene)', gar_acc_mean_class)
    print(f"======> Calculate hit@n (group activity) ({mode})")
    match_score_argsort = torch.argsort(match_score, dim=1)
    for hit_idx in range(1, 5+1):
        gar_label_array_pred_hit = info_train['activities_in_all'][match_score_argsort[:, :hit_idx]]
        gar_label_array_test_hit = info_test['activities_in_all'].view(-1, 1)
        gar_label_array_test_hit = gar_label_array_test_hit.expand(info_test['activities_in_all'].shape[0], hit_idx)
        gar_acc_array_hit = gar_label_array_test_hit == gar_label_array_pred_hit
        gar_acc_array_hit_sum = torch.sum(gar_acc_array_hit, dim=-1)
        gar_acc_array_hit_sum_over_one = gar_acc_array_hit_sum >= 1
        gar_accuracy_hit = torch.mean(gar_acc_array_hit_sum_over_one.float()).to('cpu').detach().item()
        metrics_dic[f'GAR accuracy hit@{hit_idx} (group activity) ({mode})'] = gar_accuracy_hit
        print(f'GAR accuracy hit@{hit_idx} (group activity) ({mode})', gar_accuracy_hit)
    calc_map(mode, cfg, 'group activity', torch.cdist(info_test['activities_in_all'].view(-1, 1), info_train['activities_in_all'].view(-1, 1), p=2) == 0, match_score, metrics_dic)

    print(f"======> Calculate KNN (group activity) ({mode})")
    k_max = 5
    if cfg.use_debug:
        k_max = 2
    for k in range(1, k_max+1):
        knn = KNeighborsClassifier(n_neighbors=k)
        knn_feat_train, knn_feat_test = info_train[feat_dic[mode]].cpu().numpy(), info_test[feat_dic[mode]].cpu().numpy()
        gar_labels_train, gar_labels_test = info_train['activities_in_all'].cpu().numpy(),  info_test['activities_in_all'].cpu().numpy()
        knn.fit(knn_feat_train, gar_labels_train)
        gar_acc_knn = knn.score(knn_feat_test, gar_labels_test)
        metrics_dic[f'GAR accuracy {k}NN (group activity) ({mode})'] = gar_acc_knn
        print(f'GAR accuracy {k}NN (group activity) ({mode})', gar_acc_knn)

    print(f"======> Save confusion matrix ({mode})")
    annot_kws_size = 20
    if cfg.dataset_symbol == 'cad':
        annot_kws_size = 35

    if not cfg.use_debug:
        cm_gar_scene = confusion_matrix(info_test['activities_in_all'].cpu().numpy(), info_train['activities_in_all'][match_idx].cpu().numpy(), normalize='true')
        cm_gar_scene = pd.DataFrame(cm_gar_scene, columns=get_activities(), index=get_activities())
        plt.figure(figsize=(10, 10))
        plt.tight_layout()
        s = sns.heatmap(cm_gar_scene, annot=True, cmap='Blues', fmt=".2f", annot_kws={"size": annot_kws_size}, vmin=0, vmax=1)
        s.set(xlabel='Pr', ylabel='GT')
        save_cm_scene_path = os.path.join(cfg.result_path, f'confusion_matrix_gar_{mode}.png')
        plt.savefig(save_cm_scene_path)

    print("===> Evaluate clustering")
    print(f'======> K-means clsutering ({mode})')
    kmeans = KMeans(n_clusters=30, n_init="auto")
    kmeans.fit(info_test[feat_dic[mode]].cpu().numpy())
    save_clustering_results(mode, cfg, info_test, kmeans.labels_.reshape(-1, 1))

    print("===> Evaluate 2D visualization")
    print(f'======> TSNE analysis ({mode})')
    tsne = TSNE(n_components=2, random_state=41)
    test_tsne = tsne.fit_transform(info_test[feat_dic[mode]].cpu().numpy())
    visualize_features(mode, cfg, info_test, test_tsne, True)
    visualize_features(mode, cfg, info_test, test_tsne, False)
    visualize_features_wo_ga(mode, cfg, info_test, test_tsne)
    save_visualized_features(mode, cfg, info_test, test_tsne)

def evaluate_jae_metrics(info_test, metrics_dic, cfg):
    estimated_ja_all = info_test['estimated_ja_all']
    gt_ja_all = info_test['gt_ja_all']

    print("===> Calculate distance between estimated and groun-truth joint attention")
    ja_dist = F.l1_loss(estimated_ja_all, gt_ja_all)
    ja_dist_key = 'JA distance'
    metrics_dic[ja_dist_key] = ja_dist.item()
    print(f'{ja_dist_key}', ja_dist.item())

def eval_net(cfg):
    """
    evaluating gcn net
    """
    os.environ['CUDA_VISIBLE_DEVICES']=cfg.device_list

    cfg.use_debug = False
    # cfg.use_debug = True

    # Show config parameters
    cfg.init_config()
    # cfg.person_size = (368, 368)
    show_config(cfg)
    
    # Reading dataset
    training_set,validation_set=return_dataset(cfg)
    
    params = {
        'batch_size': 1,
        'shuffle': False,
        'num_workers': 4, # 4,
    }
    training_loader=data.DataLoader(training_set,**params)
    
    params['batch_size']=cfg.test_batch_size
    validation_loader=data.DataLoader(validation_set,**params)

    # Set data position
    if cfg.use_gpu and torch.cuda.is_available():
        device = torch.device('cuda')
    else:
        device = torch.device('cpu')
    
    # Build model and Load trained model
    basenet_list={'group_relation_volleyball':Basenet_volleyball,
                  'group_relation_collective':Basenet_collective,
                  }
    gcnnet_list={
                 'group_activity_volleyball':GroupActivity_volleyball,
                 'group_relation_volleyball':GroupRelation_volleyball,
                 'group_relation_higcin_volleyball':GroupRelation_HiGCIN_volleyball,
                 'group_relation_din_volleyball':GroupRelation_DIN_volleyball,
                 'group_relation_ident_volleyball':GroupRelationIdentity_volleyball,
                 'group_relation_ae_volleyball':GroupRelationAutoEncoder_volleyball,
                 'group_relation_hrn_volleyball':GroupRelationHRN_volleyball,
                 'group_activity_collective':GroupActivity_volleyball,
                 'group_relation_collective':GroupRelation_volleyball,
                 'group_relation_higcin_collective':GroupRelation_HiGCIN_volleyball,
                 'group_relation_din_collective':GroupRelation_DIN_volleyball,
                 'group_relation_ident_collective':GroupRelationIdentity_volleyball,
                 'group_relation_ae_collective':GroupRelationAutoEncoder_volleyball,
                 'group_relation_hrn_collective':GroupRelationHRN_volleyball,
                 'dynamic_volleyball':Dynamic_volleyball,
                 'dynamic_collective':Dynamic_volleyball,
                 'higcin_volleyball':HiGCIN_volleyball,
                 'higcin_collective':HiGCIN_volleyball,
                 'person_action_recognizor':PersonAction_volleyball,
                 'single_branch_transformer':PersonActionSigleBranch_volleyball,
                 'single_branch_transformer_wo_spatial':PersonActionSigleBranchTemporal_volleyball,
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
            new_state_dict=OrderedDict()
            for k, v in state_dict.items():
                name = k[7:] if k.startswith('module.') else k
                new_state_dict[name] = v
            model.load_state_dict(new_state_dict)
            print_log(cfg.log_path, f'Loading stage{cfg.eval_stage} model: ' + cfg.stage2model)
        else:
            state_dict = torch.load(cfg.stage2model)['state_dict']
            new_state_dict=OrderedDict()
            for k, v in state_dict.items():
                name = k[7:] if k.startswith('module.') else k
                new_state_dict[name] = v
            model.load_state_dict(new_state_dict)
            print_log(cfg.log_path, f'Loading stage{cfg.eval_stage} model: ' + cfg.stage2model)
    else:
        assert False, 'cfg.eval_stage should be 1 or 2'

    if cfg.use_multi_gpu:
        model=nn.DataParallel(model)
    model=model.to(device=device)

    # pack models
    models = {'model': model}
    test_list={'volleyball':test_volleyball, 'collective':test_collective}
    test=test_list[cfg.dataset_name]
    info_train=test(training_loader, models, device, 0, cfg)
    info_test=test(validation_loader, models, device, 0, cfg)

    print("======> Evaluation")
    cfg.eval_mask_num = int(cfg.eval_mask_num)
    eval_mask_use = cfg.eval_mask_num!=0

    print("======> Evaluate iar metrics")
    iar_metrics_dic = {}
    evaluate_iar_metrics(info_train, info_test, iar_metrics_dic, cfg)
    print("======> Save iar metrics")
    if eval_mask_use:
        save_iar_metrics_path = os.path.join(cfg.result_path, f'eval_iar_metrics_{cfg.eval_mask_num}.json')
    else:
        save_iar_metrics_path = os.path.join(cfg.result_path, f'eval_iar_metrics.json')
    with open(save_iar_metrics_path, 'w') as f:
        json.dump(iar_metrics_dic, f, indent=4)

    print("======> Evaluate gar metrics")
    gar_metrics_dic = {}
    evaluate_gar_metrics('people', info_train, info_test, gar_metrics_dic, cfg)
    evaluate_gar_metrics('scene', info_train, info_test, gar_metrics_dic, cfg)
    print("======> Save gar metrics")
    if eval_mask_use:
        save_gar_metrics_path = os.path.join(cfg.result_path, f'eval_gar_metrics_{cfg.eval_mask_num}.json')
    else:
        save_gar_metrics_path = os.path.join(cfg.result_path, f'eval_gar_metrics.json')
    with open(save_gar_metrics_path, 'w') as f:
        json.dump(gar_metrics_dic, f, indent=4)

    if cfg.use_jae_loss:
        print("======> Evaluate jae metrics")
        jae_metrics_dic = {}
        evaluate_jae_metrics(info_test, jae_metrics_dic, cfg)
        print("======> Save jae metrics")
        save_jae_metrics_path = os.path.join(cfg.result_path, f'eval_jae_metrics.json')
        with open(save_jae_metrics_path, 'w') as f:
            json.dump(jae_metrics_dic, f, indent=4)

def test_volleyball(data_loader, models, device, epoch, cfg):
    model = models['model']
    model.eval()

    train_with_action = False
    actions_meter=AverageMeter()
    activities_meter=AverageMeter()
    loss_meter=AverageMeter()
    loss_pseudo_rec_meter=AverageMeter()
    activities_conf = ConfusionMeter(cfg.num_activities)
    actions_conf = ConfusionMeter(cfg.num_actions)
    epoch_timer=Timer()

    batch_idx = 0
    actions_in_all = torch.zeros(len(data_loader), cfg.num_boxes).to(device=device)
    actions_labels_all = torch.zeros(len(data_loader), cfg.num_boxes).to(device=device)
    activities_in_all = torch.zeros(len(data_loader)).to(device=device)
    locations_in_all = torch.zeros(len(data_loader), cfg.num_boxes, 2).to(device=device)
    video_id_all = ['0' for i in range(len(data_loader))]
    estimated_ja_all = torch.zeros(len(data_loader), cfg.num_frames, 2).to(device=device)
    gt_ja_all = torch.zeros(len(data_loader), cfg.num_frames, 2).to(device=device)

    if cfg.eval_stage == 1:
        individual_features = torch.zeros(len(data_loader), cfg.num_features_boxes*cfg.num_boxes).to(device=device)
        group_features = torch.zeros(len(data_loader), cfg.num_features_boxes).to(device=device)
    elif cfg.eval_stage == 2:
        if 'group_relation_ident' in cfg.inference_module_name:
            if cfg.use_ind_feat_crop == 'roi_multi':
                ind_feat_dim = cfg.emb_features*cfg.crop_size[0]*cfg.crop_size[0]
            else:
                ind_feat_dim = 4096
            individual_features = torch.zeros(len(data_loader), ind_feat_dim*cfg.num_boxes).to(device=device)
            group_features = torch.zeros(len(data_loader), ind_feat_dim).to(device=device)
        elif 'group_relation_ae' in cfg.inference_module_name:
            if cfg.use_ind_feat_crop == 'roi_multi':
                ind_feat_dim = 1024
            else:
                ind_feat_dim = 128
            individual_features = torch.zeros(len(data_loader), ind_feat_dim*cfg.num_boxes).to(device=device)
            group_features = torch.zeros(len(data_loader), ind_feat_dim).to(device=device)
        elif 'group_relation_hrn' in cfg.inference_module_name:
            if cfg.use_ind_feat_crop == 'roi_multi':
                ind_feat_dim = 512
            else:
                ind_feat_dim = 128
            individual_features = torch.zeros(len(data_loader), ind_feat_dim*cfg.num_boxes).to(device=device)
            group_features = torch.zeros(len(data_loader), ind_feat_dim).to(device=device)
        elif 'dynamic' in cfg.inference_module_name or 'din' in cfg.inference_module_name:
            individual_features = torch.zeros(len(data_loader), 128*cfg.num_boxes).to(device=device)
            group_features = torch.zeros(len(data_loader), 128).to(device=device)
        elif 'higcin' in cfg.inference_module_name:
            individual_features = torch.zeros(len(data_loader), 512*cfg.num_boxes).to(device=device)
            group_features = torch.zeros(len(data_loader), 512).to(device=device)
        else:
            individual_features = torch.zeros(len(data_loader), cfg.num_features_boxes*2*cfg.num_boxes).to(device=device)
            group_features = torch.zeros(len(data_loader), cfg.num_features_boxes*2).to(device=device)

    with torch.no_grad():
        for batch_idx, batch_data_test in enumerate(tqdm(data_loader)):
            for key in range(len(batch_data_test)):
                if torch.is_tensor(batch_data_test[key]):
                    batch_data_test[key] = batch_data_test[key].to(device=device)

            batch_size=batch_data_test[0].shape[0]
            num_frames=batch_data_test[0].shape[1]

            locations_in=batch_data_test[1].reshape((batch_size,num_frames,cfg.num_boxes, 4))
            actions_in=batch_data_test[2].reshape((batch_size,num_frames,cfg.num_boxes))
            activities_in=batch_data_test[3].reshape((batch_size,num_frames))
            
            # define list for various losses
            loss_list = []

            # forward
            input_data = {}
            input_data['images_in'] = batch_data_test[0]
            input_data['boxes_in'] = batch_data_test[1]
            input_data['images_person_in'] = batch_data_test[4]
            ret = model(input_data)

            # Predict actions
            actions_in=actions_in[:,0,:].reshape((batch_size*cfg.num_boxes,))
            activities_in=activities_in[:,0].reshape((batch_size,))

            # Predict activities
            if 'estimated_ja' in list(ret.keys()):
                estimated_ja = ret['estimated_ja']
                gt_ja = batch_data_test[6]
                gt_ja_center = (gt_ja[:, :, 0:2] + gt_ja[:, :, 2:4]) / 2
                estimated_ja_all[batch_idx] = estimated_ja[0]
                gt_ja_all[batch_idx] = gt_ja_center[0]

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
            individual_features[batch_idx] = ret['individual_feat'][0]
            group_features[batch_idx] = ret['group_feat'][0]
            actions_in_all[batch_idx] = actions_in
            activities_in_all[batch_idx] = activities_in
            locations_in_all[batch_idx, :, 0] = (locations_in[0, num_frames//2, :, 0]+locations_in[0, num_frames//2, :, 2])/2
            locations_in_all[batch_idx, :, 1] = (locations_in[0, num_frames//2, :, 1]+locations_in[0, num_frames//2, :, 3])/2
            video_id_all[batch_idx] = batch_data_test[5][0]

            if cfg.use_recon_loss:
                pass
            elif cfg.use_recon_diff_loss:
                pass
            elif cfg.use_pose_loss:
                pass
            elif cfg.inference_module_name == 'group_relation_ident_volleyball':
                pass
            elif cfg.use_jae_loss:
                pass
            elif cfg.use_act_loss:
                actions_labels_all[batch_idx] = actions_labels

            batch_idx += 1
            if cfg.use_debug and batch_idx > 5:
                break
            # if batch_idx > 3:
                # break

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
        'estimated_ja_all': estimated_ja_all,
        'gt_ja_all': gt_ja_all,
    }

    return test_info

def test_collective(data_loader, models, device, epoch, cfg):
    model = models['model']
    model.eval()
    
    actions_meter=AverageMeter()
    activities_meter=AverageMeter()
    loss_meter=AverageMeter()
    
    epoch_timer=Timer()
    activities_conf = ConfusionMeter(cfg.num_activities)
    actions_conf = ConfusionMeter(cfg.num_actions)

    batch_idx = 0
    with torch.no_grad():
        actions_in_all = torch.zeros(len(data_loader), cfg.num_boxes).to(device=device)
        actions_labels_all = torch.zeros(len(data_loader), cfg.num_boxes).to(device=device)
        activities_in_all = torch.zeros(len(data_loader)).to(device=device)
        locations_in_all = torch.zeros(len(data_loader), cfg.num_boxes, 2).to(device=device)
        video_id_all = ['0' for i in range(len(data_loader))]

        if cfg.eval_stage == 1:
            individual_features = torch.zeros(len(data_loader), cfg.num_features_boxes*cfg.num_boxes).to(device=device)
            group_features = torch.zeros(len(data_loader), cfg.num_features_boxes).to(device=device)
        elif cfg.eval_stage == 2:
            if 'group_relation_ident' in cfg.inference_module_name:
                if cfg.use_ind_feat_crop == 'roi_multi':
                    ind_feat_dim = 1056*cfg.crop_size[0]*cfg.crop_size[0]
                else:
                    ind_feat_dim = 4096
                individual_features = torch.zeros(len(data_loader), ind_feat_dim*cfg.num_boxes).to(device=device)
                group_features = torch.zeros(len(data_loader), ind_feat_dim).to(device=device)
            elif 'group_relation_ae' in cfg.inference_module_name:
                if cfg.use_ind_feat_crop == 'roi_multi':
                    ind_feat_dim = 1024
                else:
                    ind_feat_dim = 128
                individual_features = torch.zeros(len(data_loader), ind_feat_dim*cfg.num_boxes).to(device=device)
                group_features = torch.zeros(len(data_loader), ind_feat_dim).to(device=device)
            elif 'group_relation_hrn' in cfg.inference_module_name:
                if cfg.use_ind_feat_crop == 'roi_multi':
                    ind_feat_dim = 512
                else:
                    ind_feat_dim = 128
                individual_features = torch.zeros(len(data_loader), ind_feat_dim*cfg.num_boxes).to(device=device)
                group_features = torch.zeros(len(data_loader), ind_feat_dim).to(device=device)
            elif 'dynamic' in cfg.inference_module_name:
                individual_features = torch.zeros(len(data_loader), cfg.num_features_boxes*cfg.num_boxes).to(device=device)
                group_features = torch.zeros(len(data_loader), cfg.num_features_boxes).to(device=device)
            elif 'din' in cfg.inference_module_name:
                individual_features = torch.zeros(len(data_loader), 128*cfg.num_boxes).to(device=device)
                group_features = torch.zeros(len(data_loader), 128).to(device=device)
            elif 'higcin' in cfg.inference_module_name:
                individual_features = torch.zeros(len(data_loader), 1056*cfg.num_boxes).to(device=device)
                group_features = torch.zeros(len(data_loader), 1056).to(device=device)
            else:
                individual_features = torch.zeros(len(data_loader), cfg.num_features_boxes*2*cfg.num_boxes).to(device=device)
                group_features = torch.zeros(len(data_loader), cfg.num_features_boxes*2).to(device=device)

        for batch_data_test in tqdm(data_loader):
            # prepare batch data
            # batch_data_test=[b.to(device=device) for b in batch_data_test]
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
            individual_features[batch_idx] = ret['individual_feat'][0]
            group_features[batch_idx] = ret['group_feat'][0]
            actions_in_all[batch_idx] = batch_data_test[2].reshape((batch_size,num_frames,cfg.num_boxes))[:,0,:].reshape((batch_size*cfg.num_boxes,))
            activities_in_all[batch_idx] = batch_data_test[3].reshape((batch_size,num_frames))[:,0].reshape((batch_size,))
            locations_in_all[batch_idx, :, 0] = (locations_in[0, num_frames//2, :, 0]+locations_in[0, num_frames//2, :, 2])/2
            locations_in_all[batch_idx, :, 1] = (locations_in[0, num_frames//2, :, 1]+locations_in[0, num_frames//2, :, 3])/2
            video_id_all[batch_idx] = batch_data_test[6][0]

            batch_idx += 1
            if cfg.use_debug and batch_idx > 5:
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