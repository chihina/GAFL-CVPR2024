import json
import os
import shutil
import sys
import random
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import cv2
import math
from tqdm import tqdm
from sklearn.metrics.cluster import normalized_mutual_info_score
from sklearn.metrics.cluster import adjusted_rand_score
sys.path.append(".")

def read_annotation(ann_path):
    with open(ann_path, 'r') as f:
        anns = f.readlines()
    ann_dic = {}
    for ann in anns:
        ann = ann.strip().split(' ')
        ann_dic[ann[0].split('.')[0]] = ann[1:]
    return ann_dic

def iou_np(a, b):
    a_area = (a[2] - a[0] + 1) \
             * (a[3] - a[1] + 1)
    b_area = (b[:,2] - b[:,0] + 1) \
             * (b[:,3] - b[:,1] + 1)
    
    abx_mn = np.maximum(a[0], b[:,0]) # xmin
    aby_mn = np.maximum(a[1], b[:,1]) # ymin
    abx_mx = np.minimum(a[2], b[:,2]) # xmax
    aby_mx = np.minimum(a[3], b[:,3]) # ymax
    w = np.maximum(0, abx_mx - abx_mn + 1)
    h = np.maximum(0, aby_mx - aby_mn + 1)
    intersect = w*h
    
    iou = intersect / (a_area + b_area - intersect)
    return iou

ACTIVITIES = ['r_set', 'r_spike', 'r-pass', 'r_winpoint',
            'l_set', 'l-spike', 'l-pass', 'l_winpoint']
ACTIVITIES_COLORS = ['b', 'g', 'r', 'magenta', 
            'cyan', 'lime', 'coral', 'purple']

ACTIONS = ['blocking', 'digging', 'falling', 'jumping',
           'moving', 'setting', 'spiking', 'standing',
           'waiting']

DISCRIMINATIVE_ACTIONS_DIC = {}
DISCRIMINATIVE_ACTIONS_DIC['set'] = ['setting']
DISCRIMINATIVE_ACTIONS_DIC['spike'] = ['spiking']
DISCRIMINATIVE_ACTIONS_DIC['pass'] = ['digging']
DISCRIMINATIVE_ACTIONS_DIC['winpoint'] = ['falling']

saved_result_dir = os.path.join('result')
saved_result_analysis_dir = os.path.join('result_analysis')

# define model name
# analyze_name = '[GR ours rand mask 5_stage2]<2023-10-16_22-26-54>'
analyze_name = '[GR ours_stage2]<2023-07-08_08-59-54>'
# analyze_name = '[GR ours recon feat random mask 0 w temp cond_stage2]<2023-11-10_13-26-06>'

cl_cent_type = 'mean'
# cl_cent_type = 'median'

cl_sum = 8
# cl_sum = 30
# cl_sum = 100
# cl_sum = 200
# cl_sum = 300

cl_activate_type = 'backprop_original'
# cl_activate_type = 'backprop_energy_original'
# cl_activate_type = 'perturbation_original'
# cl_activate_type = 'random'

# use_debug_model = False
use_debug_model = True

if use_debug_model:
    model_setting_name = f'test_cluster_{cl_sum}_scene_{cl_cent_type}_{cl_activate_type}_debug'
    save_dir = os.path.join(saved_result_analysis_dir, 'xai_la_on_volleyball', analyze_name, f'all_cluster_{cl_sum}_{cl_cent_type}_{cl_activate_type}_debug')
else:
    model_setting_name = f'test_cluster_{cl_sum}_scene_{cl_cent_type}_{cl_activate_type}'
    save_dir = os.path.join(saved_result_analysis_dir, 'xai_la_on_volleyball', analyze_name, f'all_cluster_{cl_sum}_{cl_cent_type}_{cl_activate_type}')

if os.path.exists(save_dir):
    shutil.rmtree(save_dir)
os.makedirs(save_dir)

# make a directory to save analized results
save_dir_ga = os.path.join(save_dir, 'ga')
if os.path.exists(save_dir_ga):
    shutil.rmtree(save_dir_ga)
os.makedirs(save_dir_ga)

# make a directory to save analized results
save_dir_cl = os.path.join(save_dir, 'cl')
if os.path.exists(save_dir_cl):
    shutil.rmtree(save_dir_cl)
os.makedirs(save_dir_cl)

read_excel_file_path = os.path.join(saved_result_dir, analyze_name, f'{model_setting_name}.xlsx')
df_original = pd.read_excel(read_excel_file_path, index_col=0)
df = df_original.sample(frac=1, random_state=777)
ga_class_list = df['GA'].values.tolist()
cl_class_list = df['CL'].values.tolist()

# calculate NMI to evaluate clustering performance
print(f'NMI: {normalized_mutual_info_score(ga_class_list, cl_class_list):.4f}')

# calculate ARI to evaluate clustering performance
print(f'ARI: {adjusted_rand_score(ga_class_list, cl_class_list):.4f}')

# visualize attention values for each person
att_val_dist_dic = {i:[] for i in ACTIVITIES}
att_idx_dist_dic = {i:[] for i in ACTIVITIES}
grp_idx_dist_dic = {i:[] for i in ACTIVITIES}

# read grouping information
print('Read grouping information')
grp_dic = {}
grp_info_dir = os.path.join('data_local', 'volleyball_tracking_annotation')
for vid_id in tqdm(os.listdir(grp_info_dir)):
    if 'txt' in vid_id:
        continue
    for seq_id in os.listdir(os.path.join(grp_info_dir, vid_id)):
        names_list = ['pid', 'x1', 'y1', 'x2', 'y2', 'frame', 'lost', 'grouping', 'generated', 'label', 'no']
        grp_txt_path = os.path.join(grp_info_dir, vid_id, seq_id, f'{seq_id}.txt')
        grp_dic[f'{vid_id}_{seq_id}'] = pd.read_csv(grp_txt_path, sep=' ', header=None, names=names_list)

# plot pie chart for all clusters
pie_width_length = 15
pie_height_length = math.ceil(cl_sum/pie_width_length)
fig = plt.figure(figsize=(pie_width_length*8, pie_height_length*8))
ax_list = [fig.add_subplot(pie_height_length, pie_width_length, i+1) for i in range(cl_sum)]
for cluster_idx in tqdm(range(cl_sum)):
    ax = ax_list[cluster_idx]
    cluster_ga_cnt = df[df['CL'] == cluster_idx]['GA'].value_counts()
    cluster_ga_cnt_name = [ACTIVITIES[idx] for idx in cluster_ga_cnt.index.tolist()]
    clsuter_ga_color = [ACTIVITIES_COLORS[idx] for idx in cluster_ga_cnt.index.tolist()]
    ax.pie(cluster_ga_cnt.values.tolist(), labels=cluster_ga_cnt_name, colors=clsuter_ga_color, textprops={'fontsize': 20})
    ax.set_title('cluster_'+str(cluster_idx)+f' ({cluster_ga_cnt.sum()})', fontsize=30)
fig.tight_layout()
plt.savefig(os.path.join(save_dir_cl, f'CL_all.png'))

# plot pie chart for each cluster
for cluster_idx in tqdm(range(cl_sum)):
    save_dir_cl_each = os.path.join(save_dir_cl, f'cluster_{cluster_idx}')
    if not os.path.exists(save_dir_cl_each):
        os.makedirs(save_dir_cl_each)

    plt.figure()
    cluster_ga_cnt = df[df['CL'] == cluster_idx]['GA'].value_counts()
    cluster_ga_cnt_name = [ACTIVITIES[idx] for idx in cluster_ga_cnt.index.tolist()]
    clsuter_ga_color = [ACTIVITIES_COLORS[idx] for idx in cluster_ga_cnt.index.tolist()]
    plt.pie(cluster_ga_cnt.values.tolist(), labels=cluster_ga_cnt_name, colors=clsuter_ga_color, textprops={'fontsize': 20})
    plt.title('cluster_'+str(cluster_idx)+f' ({cluster_ga_cnt.sum()})', fontsize=30)
    plt.savefig(os.path.join(save_dir_cl_each, f'CL_{cluster_idx}.png'))

    df_cluster = df[df['CL'] == cluster_idx]
    # for vis_att_idx in range(min((5, df_cluster.shape[0]))):
    for vis_att_idx in range(df_cluster.shape[0]):
        df_vis = df_cluster.iloc[vis_att_idx]
        vid_id, seq_id, img_id = df_vis.name.split('_')

        # read annotation
        ann_path = os.path.join('data', 'volleyball', 'videos', vid_id, f'annotations.txt')
        ann_dic = read_annotation(ann_path)
        ann_vis = ann_dic[seq_id]
        ga_class = ann_vis[0]
        person_info = ann_vis[1:]
        people_num = len(person_info)//5

        # read group information
        df_grp = grp_dic[f'{vid_id}_{seq_id}']
        df_grp = df_grp[df_grp['frame'] == int(seq_id)]
        df_grp_bbox = df_grp[['x1', 'y1', 'x2', 'y2']].values

        # get original index
        df_original_index = df_original.index.get_loc(df_vis.name)

        if 'set' in ga_class:
            discrimiate_action = DISCRIMINATIVE_ACTIONS_DIC['set']
        elif 'spike' in ga_class:
            discrimiate_action = DISCRIMINATIVE_ACTIONS_DIC['spike']
        elif 'pass' in ga_class:
            discrimiate_action = DISCRIMINATIVE_ACTIONS_DIC['pass']
        elif 'winpoint' in ga_class:
            discrimiate_action = DISCRIMINATIVE_ACTIONS_DIC['winpoint']

        att_val_array = np.array([df_vis[f'P:{i}'] for i in range(people_num)])
        att_val_array = np.sort(att_val_array)[::-1]
        att_val_vis_idx_array = np.zeros((people_num), dtype=np.int32)

        # read image
        if vis_att_idx < 5:
            img_path = os.path.join('data', 'volleyball', 'videos', vid_id, seq_id, f'{seq_id}.jpg')
            img = cv2.imread(img_path)
        
        for person_idx in range(people_num):
            # get annotation information
            x1, y1, width, height = map(int, person_info[person_idx*5:person_idx*5+4])
            x2, y2 = x1+width, y1+height
            ia_label = person_info[person_idx*5+4]
            att_val = df_vis[f'P:{person_idx}']
            att_val_idx = np.where(att_val_array == att_val)[0][0]

            # visualize attention value
            if vis_att_idx < 5:
                vis_color_inten = int((12-att_val_idx)/people_num*255)
                cm_vis = plt.get_cmap('jet')(vis_color_inten)
                cm_vis_scale = [int(cm_vis[2]*255), int(cm_vis[1]*255), int(cm_vis[0]*255), int(cm_vis[3]*255)]
                cv2.rectangle(img, (x1, y1), (x2, y2), cm_vis_scale, 2)
            
            # visualize bboxes
            if ia_label in discrimiate_action:
                att_val_dist_dic[ga_class].append(att_val)
                att_idx_dist_dic[ga_class].append(att_val_idx)
                # cv2.putText(img, f'{ia_label}', (x1, y1-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, cm_vis_scale, 2)

            # match person id with bouding box
            iou = iou_np(np.array([x1, y1, x2, y2]), df_grp_bbox)
            iou_max_idx = np.argmax(iou)
            df_grp_person = df_grp.iloc[iou_max_idx]
            if ia_label != df_grp_person['label']:
                iou[iou_max_idx] = 0
                iou_max_idx_update1 = np.argmax(iou)
                df_grp_person = df_grp.iloc[iou_max_idx_update1]
            if df_grp_person['grouping']:
                # grp_idx_dist_dic[ga_class].append(att_val_idx)
                grp_idx_dist_dic[ga_class].append(att_val_idx<df_grp['grouping'].sum())
                cv2.putText(img, f'{ia_label}', (x1, y1-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 2)

        # save images
        if vis_att_idx < 5:
            cv2.imwrite(os.path.join(save_dir_cl_each, f'{vid_id}_{seq_id}_{img_id}_{ga_class}.jpg'), img)

print('Average attention value for discriminative actions')
att_val_thr_max = 5
eval_activities = ['r_set', 'r_spike', 'l_set', 'l-spike']
disc_match_array = np.zeros((len(eval_activities)+1, att_val_thr_max))
for ga_idx, ga_class in enumerate(eval_activities):
    print(f'===GA_ID: {ga_class}===')
    for att_val_thr in range(1, att_val_thr_max+1):
        att_acc = np.mean(np.array(att_idx_dist_dic[ga_class]) < att_val_thr)
        print(f'Att. acc. (<{att_val_thr}): {att_acc:.2f}')
        disc_match_array[ga_idx, att_val_thr-1] = att_acc
disc_match_array[-1] = np.mean(disc_match_array[:-1], axis=0)

print('Average attention value for grouping actions')
grp_match_array = np.zeros((len(ACTIVITIES)+1, att_val_thr_max))
for ga_idx, ga_class in enumerate(ACTIVITIES):
    print(f'===GA_ID: {ga_class}===')
    for grp_idx_thr in range(1, att_val_thr_max+1):
        # grp_acc = np.mean(np.array(grp_idx_dist_dic[ga_class]) < grp_idx_thr)
        grp_acc = np.mean(np.array(grp_idx_dist_dic[ga_class]))
        print(f'Grp. acc. (<{grp_idx_thr}): {grp_acc:.2f}')
        grp_match_array[ga_idx, grp_idx_thr-1] = grp_acc
grp_match_array[-1] = np.mean(grp_match_array[:-1], axis=0)

print('Save evaluation metrics as a excel file')
save_metrics_array = np.concatenate((disc_match_array, grp_match_array), axis=0)
save_metrcis_index = eval_activities+['ALL']+ACTIVITIES+['ALL']
save_metrics_header = [f'<{i}' for i in range(1, att_val_thr_max+1)]
df_metrics = pd.DataFrame(save_metrics_array, save_metrcis_index, save_metrics_header)
df_metrics_disc = df_metrics.iloc[:len(eval_activities)+1]
df_metrics_grp = df_metrics.iloc[len(eval_activities)+1:]
save_excel_file_path = os.path.join(save_dir, f'{model_setting_name}.xlsx')
with pd.ExcelWriter(save_excel_file_path) as writer:
    df_metrics.to_excel(writer, sheet_name='all')
    df_metrics_disc.to_excel(writer, sheet_name='discriminative')
    df_metrics_grp.to_excel(writer, sheet_name='grouping')

# visualize group activity classes (all)
fig = plt.figure()
ax_list = [fig.add_subplot(2, 4, i+1) for i in range(len(ACTIVITIES))]
for ga_idx, ga_class in enumerate(ACTIVITIES):
    ax = ax_list[ga_idx]
    ax.hist(df[df['GA'] == ga_idx]['CL'], align='mid')
    ax.set_title(ga_class)
    ax.set_ylim(0, (df.shape[0]//len(ACTIVITIES)))
fig.tight_layout()
plt.savefig(os.path.join(save_dir_ga, f'GA_all.png'))

# visualize group activity classes (each)
for ga_idx, ga_class in enumerate(ACTIVITIES):
    plt.figure()
    plt.hist(df[df['GA'] == ga_idx]['CL'], align='mid')
    plt.ylim(0, (df.shape[0]//len(ACTIVITIES)))
    plt.title(ga_class)
    plt.savefig(os.path.join(save_dir_ga, f'GA_{ga_class}.png'))
    plt.close()
